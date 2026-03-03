"""
Token Forensics: Who Needs the Swordsman?

Deep analysis of per-token properties that predict MLP nonlinearity need.
For each layer, we collect per-position (activation, L_full, L_lin) triples
along with rich token-level features, then run multiple regression to identify
what actually predicts the need for nonlinear computation.

The "murder mystery": function/content word split has weak empirical support.
What really determines which tokens can go linear?

Hypotheses:
  H1 (Rarity):     log-frequency predicts early-layer nonlinearity need
  H2 (Entropy):    next-token entropy predicts late-layer nonlinearity need
  H3 (Relational): syntactic dependency type predicts mid-layer need
  H4 (Subword):    subword fragments vs whole words behave differently
  H5 (Position):   sentence position matters (early = more ambiguous)
  H6 (Surprise):   token-level surprise (self-info) predicts need

Output: per-layer feature importance profiles + rich qualitative token tables.

Usage:
  python token_forensics.py --model gpt2-medium --layers 1,6,12,18,23
  python token_forensics.py --model EleutherAI/pythia-410m --layers 0,2,6,12,18,23
  python token_forensics.py --model gpt2-medium --all-layers
  python token_forensics.py --model gpt2-medium --layers 1,12,23 --device mps
"""

import torch
import torch.nn as nn
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from collections import Counter
import json, time, argparse, os, sys, warnings

warnings.filterwarnings("ignore", category=FutureWarning)

# Reuse the ModelAdapter from scale_test_universal
sys.path.insert(0, os.path.dirname(__file__))
from scale_test_universal import ModelAdapter, compute_loss

loss_fn = nn.CrossEntropyLoss(reduction='none')


# ─── Token Feature Extraction ────────────────────────────────────────────────

class TokenFeatureExtractor:
    """Extract rich per-token features for forensic analysis."""

    # Closed-class / function word sets at different granularities
    DETERMINERS = {'the', 'a', 'an', 'this', 'that', 'these', 'those',
                   'my', 'your', 'his', 'her', 'its', 'our', 'their',
                   'some', 'any', 'no', 'every', 'each', 'all', 'both',
                   'few', 'many', 'much', 'more', 'most'}
    PREPOSITIONS = {'of', 'in', 'to', 'for', 'with', 'on', 'at', 'from',
                    'by', 'about', 'as', 'into', 'through', 'during',
                    'before', 'after', 'above', 'below', 'between',
                    'under', 'over', 'against', 'among', 'within'}
    PRONOUNS = {'i', 'me', 'my', 'mine', 'myself',
                'you', 'your', 'yours', 'yourself',
                'he', 'him', 'his', 'himself',
                'she', 'her', 'hers', 'herself',
                'it', 'its', 'itself',
                'we', 'us', 'our', 'ours', 'ourselves',
                'they', 'them', 'their', 'theirs', 'themselves',
                'who', 'whom', 'whose', 'which', 'what', 'that'}
    CONJUNCTIONS = {'and', 'but', 'or', 'nor', 'for', 'yet', 'so',
                    'because', 'although', 'while', 'if', 'when',
                    'since', 'unless', 'until', 'whether', 'though'}
    AUXILIARIES = {'is', 'am', 'are', 'was', 'were', 'be', 'been', 'being',
                   'have', 'has', 'had', 'having',
                   'do', 'does', 'did',
                   'will', 'would', 'shall', 'should',
                   'may', 'might', 'can', 'could', 'must'}
    NEGATION = {"n't", 'not', 'no', 'never', 'neither', 'nor', 'nothing',
                'nobody', 'nowhere', 'none'}
    DISCOURSE = {'however', 'therefore', 'moreover', 'furthermore',
                 'nevertheless', 'instead', 'meanwhile', 'otherwise',
                 'consequently', 'accordingly', 'subsequently',
                 'additionally', 'alternatively', 'similarly'}
    PUNCTUATION = {',', '.', '!', '?', ';', ':', '-', '--', '(', ')', '"',
                   "'", '/', '[', ']', '{', '}', '...'}

    ALL_FUNCTION = (DETERMINERS | PREPOSITIONS | PRONOUNS | CONJUNCTIONS |
                    AUXILIARIES | NEGATION | PUNCTUATION)

    def __init__(self, tokenizer, model, device='cpu'):
        self.tokenizer = tokenizer
        self.model = model
        self.device = device

        # Build frequency table from tokenizer vocab
        # (rough proxy — actual corpus frequency would be better)
        self.vocab_size = tokenizer.vocab_size

        # Precompute token properties
        self._precompute_token_properties()

    def _precompute_token_properties(self):
        """Precompute static per-token-id properties."""
        self.token_texts = {}
        self.token_is_subword = {}
        self.token_is_space_prefix = {}
        self.token_char_len = {}
        self.token_is_alpha = {}
        self.token_category = {}  # fine-grained category

        for token_id in range(self.vocab_size):
            text = self.tokenizer.decode([token_id])
            text_stripped = text.strip().lower()
            self.token_texts[token_id] = text

            # Subword detection: doesn't start with space and isn't punctuation
            is_subword = (not text.startswith(' ') and
                         not text.startswith('Ġ') and  # GPT-2 space marker
                         text_stripped not in self.PUNCTUATION and
                         len(text_stripped) > 0 and
                         text_stripped[0].isalpha())
            self.token_is_subword[token_id] = is_subword

            # Space prefix (whole word in GPT-2 tokenization)
            self.token_is_space_prefix[token_id] = (
                text.startswith(' ') or text.startswith('Ġ'))

            self.token_char_len[token_id] = len(text)
            self.token_is_alpha[token_id] = text_stripped.isalpha()

            # Fine-grained category
            if text_stripped in self.DETERMINERS:
                cat = 'determiner'
            elif text_stripped in self.PREPOSITIONS:
                cat = 'preposition'
            elif text_stripped in self.PRONOUNS:
                cat = 'pronoun'
            elif text_stripped in self.CONJUNCTIONS:
                cat = 'conjunction'
            elif text_stripped in self.AUXILIARIES:
                cat = 'auxiliary'
            elif text_stripped in self.NEGATION:
                cat = 'negation'
            elif text_stripped in self.DISCOURSE:
                cat = 'discourse_marker'
            elif text_stripped in self.PUNCTUATION:
                cat = 'punctuation'
            elif text.strip() == '':
                cat = 'whitespace'
            elif not text_stripped[0:1].isalpha():
                cat = 'symbol'
            elif is_subword:
                cat = 'subword'
            else:
                cat = 'content_word'
            self.token_category[token_id] = cat

    def extract_features(self, token_ids, positions_in_seq, logits_full,
                         logits_lin, token_freqs):
        """
        Extract per-position feature vectors.

        Args:
            token_ids: (N,) int array of token ids
            positions_in_seq: (N,) int array of position within sequence
            logits_full: (N, vocab) full MLP logits
            logits_lin: (N, vocab) linear MLP logits
            token_freqs: dict of token_id -> count in training corpus

        Returns:
            features: dict of feature_name -> (N,) float array
            feature_names: list of feature names
            metadata: dict of per-position metadata (token text, category, etc.)
        """
        N = len(token_ids)
        features = {}

        # ─── Feature 1: Log frequency ───
        total_count = sum(token_freqs.values())
        log_freq = np.array([
            np.log(token_freqs.get(int(t), 1) / total_count)
            for t in token_ids
        ])
        features['log_frequency'] = log_freq

        # ─── Feature 2: Token rank (by frequency) ───
        # Sort tokens by frequency, assign rank
        sorted_tokens = sorted(token_freqs.items(), key=lambda x: -x[1])
        rank_map = {tid: rank for rank, (tid, _) in enumerate(sorted_tokens)}
        log_rank = np.array([
            np.log(rank_map.get(int(t), len(rank_map)) + 1)
            for t in token_ids
        ])
        features['log_rank'] = log_rank

        # ─── Feature 3: Is subword ───
        is_subword = np.array([
            self.token_is_subword.get(int(t), 0) for t in token_ids
        ], dtype=float)
        features['is_subword'] = is_subword

        # ─── Feature 4: Character length ───
        char_len = np.array([
            self.token_char_len.get(int(t), 1) for t in token_ids
        ], dtype=float)
        features['char_length'] = char_len

        # ─── Feature 5: Is function word (binary) ───
        is_func = np.array([
            1.0 if self.token_texts.get(int(t), '').strip().lower()
                   in self.ALL_FUNCTION else 0.0
            for t in token_ids
        ])
        features['is_function_word'] = is_func

        # ─── Feature 6: Fine-grained category (one-hot) ───
        categories = [self.token_category.get(int(t), 'content_word')
                      for t in token_ids]
        unique_cats = sorted(set(categories))
        for cat in unique_cats:
            features[f'cat_{cat}'] = np.array(
                [1.0 if c == cat else 0.0 for c in categories])

        # ─── Feature 7: Position in sequence ───
        features['position_in_seq'] = positions_in_seq.astype(float)
        features['position_normalized'] = positions_in_seq.astype(float) / 512.0

        # ─── Feature 8: Self-information (surprise of this token) ───
        # How surprising is THIS token given the context?
        # Use full model's logits at position i-1 to get P(token_i)
        # (We approximate by using unigram surprise = -log(freq))
        features['unigram_surprise'] = -log_freq  # simple version

        # ─── Features 9-12: Entropy, KL, agreement (computed in chunks) ───
        if logits_full is not None:
            CHUNK = 500  # process in chunks to avoid memory explosion
            entropy_arr = np.zeros(N)
            entropy_lin_arr = np.zeros(N)
            kl_arr = np.zeros(N)
            top1_agree_arr = np.zeros(N)

            for i in range(0, N, CHUNK):
                j = min(i + CHUNK, N)
                lf = torch.tensor(logits_full[i:j], dtype=torch.float32)
                pf = torch.softmax(lf, dim=-1)
                log_pf = torch.log(pf + 1e-10)
                entropy_arr[i:j] = -(pf * log_pf).sum(dim=-1).numpy()

                if logits_lin is not None:
                    ll = torch.tensor(logits_lin[i:j], dtype=torch.float32)
                    pl = torch.softmax(ll, dim=-1)
                    log_pl = torch.log(pl + 1e-10)
                    entropy_lin_arr[i:j] = -(pl * log_pl).sum(dim=-1).numpy()
                    kl_arr[i:j] = (pf * (log_pf - log_pl)).sum(dim=-1).numpy()
                    top1_agree_arr[i:j] = (
                        lf.argmax(dim=-1) == ll.argmax(dim=-1)).float().numpy()

            features['next_token_entropy'] = entropy_arr
            if logits_lin is not None:
                features['entropy_delta'] = entropy_lin_arr - entropy_arr
                features['kl_full_to_linear'] = kl_arr
                features['top1_agree'] = top1_agree_arr
        else:
            features['next_token_entropy'] = np.zeros(N)

        # ─── Feature 13: Is alphabetic ───
        features['is_alpha'] = np.array([
            self.token_is_alpha.get(int(t), 0) for t in token_ids
        ], dtype=float)

        # ─── Feature 14: Starts a new word ───
        features['starts_word'] = np.array([
            self.token_is_space_prefix.get(int(t), 0) for t in token_ids
        ], dtype=float)

        # Build metadata
        metadata = {
            'token_texts': [self.token_texts.get(int(t), '?') for t in token_ids],
            'categories': categories,
            'token_ids': token_ids.tolist() if hasattr(token_ids, 'tolist') else list(token_ids),
        }

        # Filter to non-category features for the feature name list
        feature_names = [k for k in features.keys() if not k.startswith('cat_')]

        return features, feature_names, metadata


# ─── Data Collection ──────────────────────────────────────────────────────────

def collect_layer_data(adapter, tokenizer, layer_idx, n_tokens=25000,
                       device='cpu'):
    """
    Collect per-position (activation, L_full, L_lin, logits) data for one layer.

    Returns:
        token_ids:   (N,) token ids
        positions:   (N,) position in sequence
        L_full:      (N,) per-token loss with full MLP
        L_lin:       (N,) per-token loss with linear surrogate
        logits_full: (N, vocab) logits from full model
        logits_lin:  (N, vocab) logits from linear model
        activations: (N, d) MLP input activations
        token_freqs: dict of token_id -> count
    """
    model = adapter.model
    mlp = adapter.get_mlp(layer_idx)
    hidden_dim = adapter.hidden_dim

    # Load data
    dataset = load_dataset("wikitext", "wikitext-103-raw-v1", split="test")
    text = "\n".join([x for x in dataset["text"] if x.strip()])
    tokens = tokenizer.encode(text)

    # Count frequencies
    token_freqs = Counter(tokens[:100000])  # first 100k tokens

    # Fit linear approximation
    print(f"  Fitting linear approximation for layer {layer_idx}...", flush=True)
    fit_tokens = torch.tensor([tokens[:10000]], dtype=torch.long).to(device)

    acts_fit = []
    outs_fit = []
    original_forward = mlp.forward

    def capture_hook(x, *args, **kwargs):
        acts_fit.append(x.detach().cpu())
        out = original_forward(x, *args, **kwargs)
        outs_fit.append(out.detach().cpu())
        return out

    mlp.forward = capture_hook
    with torch.no_grad():
        model(fit_tokens[:, :512])
        for start in range(512, min(10000, len(tokens)), 512):
            end = min(start + 512, 10000)
            model(fit_tokens[:, start:end])
    mlp.forward = original_forward

    X_fit = torch.cat(acts_fit, dim=1).squeeze(0).numpy()
    Y_fit = torch.cat(outs_fit, dim=1).squeeze(0).numpy()

    # Ridge regression for linear surrogate
    X_mean = X_fit.mean(axis=0)
    Y_mean = Y_fit.mean(axis=0)
    Xc = X_fit - X_mean
    Yc = Y_fit - Y_mean

    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    lam = 0.01
    S_inv = S / (S**2 + lam)
    W_lin = (Vt.T * S_inv) @ U.T @ Yc
    b_lin = Y_mean - X_mean @ W_lin

    print(f"  Linear fit done. Collecting per-token data...", flush=True)

    # Now collect per-token data from separate split
    eval_start = 30000
    eval_tokens_raw = tokens[eval_start:eval_start + n_tokens + 512]

    all_token_ids = []
    all_positions = []
    all_L_full = []
    all_L_lin = []
    all_logits_full = []
    all_logits_lin = []
    all_acts = []

    for chunk_start in range(0, min(n_tokens, len(eval_tokens_raw) - 1), 512):
        chunk_end = min(chunk_start + 512, len(eval_tokens_raw))
        input_ids = torch.tensor(
            [eval_tokens_raw[chunk_start:chunk_end]], dtype=torch.long).to(device)

        if input_ids.shape[1] < 2:
            continue

        # Full model forward
        with torch.no_grad():
            out_full = model(input_ids)
            logits_full = out_full.logits[0, :-1].cpu()  # (seq-1, vocab)
            targets = input_ids[0, 1:].cpu()
            losses_full = nn.CrossEntropyLoss(reduction='none')(
                logits_full.float(), targets).numpy()

        # Linear surrogate forward
        captured_acts = []
        def capture_only(x, *args, **kwargs):
            captured_acts.append(x.detach().cpu())
            return original_forward(x, *args, **kwargs)

        mlp.forward = capture_only
        with torch.no_grad():
            model(input_ids)
        mlp.forward = original_forward

        acts = captured_acts[0].squeeze(0).numpy()  # (seq, d)

        # Compute linear MLP output
        lin_out = acts @ W_lin + b_lin  # (seq, d)

        # Replace MLP and get logits
        def linear_forward(x, *args, **kwargs):
            x_np = x.detach().cpu().numpy()
            out_np = x_np @ W_lin + b_lin
            return torch.tensor(out_np, dtype=x.dtype, device=x.device)

        mlp.forward = linear_forward
        with torch.no_grad():
            out_lin = model(input_ids)
            logits_lin = out_lin.logits[0, :-1].cpu()
            losses_lin = nn.CrossEntropyLoss(reduction='none')(
                logits_lin.float(), targets).numpy()
        mlp.forward = original_forward

        seq_len = logits_full.shape[0]
        positions = np.arange(chunk_start, chunk_start + seq_len)

        all_token_ids.append(targets.numpy())
        all_positions.append(positions)
        all_L_full.append(losses_full)
        all_L_lin.append(losses_lin)
        # Store logits as float16 to save memory
        all_logits_full.append(logits_full.half().numpy())
        all_logits_lin.append(logits_lin.half().numpy())
        all_acts.append(acts[:-1])  # align with targets

        collected = sum(len(x) for x in all_token_ids)
        if collected % 5000 < 512:
            print(f"    Collected {collected}/{n_tokens} tokens", flush=True)

        if collected >= n_tokens:
            break

    return {
        'token_ids': np.concatenate(all_token_ids),
        'positions': np.concatenate(all_positions),
        'L_full': np.concatenate(all_L_full),
        'L_lin': np.concatenate(all_L_lin),
        'logits_full': np.concatenate(all_logits_full, axis=0),
        'logits_lin': np.concatenate(all_logits_lin, axis=0),
        'activations': np.concatenate(all_acts, axis=0),
        'token_freqs': token_freqs,
    }


# ─── Analysis ─────────────────────────────────────────────────────────────────

def analyze_layer(data, extractor, layer_idx):
    """
    Full forensic analysis for one layer.

    Returns a rich result dict with:
      - Feature importances (multiple methods)
      - Per-category statistics
      - Qualitative token tables
      - Hypothesis test results
    """
    delta = data['L_lin'] - data['L_full']  # positive = MLP helps
    result = {'layer': layer_idx}

    # Extract features
    features, feature_names, metadata = extractor.extract_features(
        data['token_ids'],
        data['positions'],
        data['logits_full'],
        data['logits_lin'],
        data['token_freqs'],
    )

    # ─── 1. Multiple regression: what predicts delta? ───
    # Use core features (not one-hot categories)
    core_features = [
        'log_frequency', 'log_rank', 'is_subword', 'char_length',
        'is_function_word', 'position_normalized', 'unigram_surprise',
        'next_token_entropy', 'is_alpha', 'starts_word'
    ]
    available = [f for f in core_features if f in features]

    X = np.column_stack([features[f] for f in available])
    y = delta

    # Remove NaN/inf
    mask = np.isfinite(X).all(axis=1) & np.isfinite(y)
    X = X[mask]
    y = y[mask]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Ridge regression
    ridge = Ridge(alpha=1.0)
    ridge.fit(X_scaled, y)

    importances = dict(zip(available, ridge.coef_))
    r_squared = ridge.score(X_scaled, y)

    result['regression'] = {
        'feature_importances': {k: round(float(v), 6) for k, v in importances.items()},
        'r_squared': round(float(r_squared), 4),
        'n_samples': int(len(y)),
    }

    # Rank features by |coefficient|
    ranked = sorted(importances.items(), key=lambda x: abs(x[1]), reverse=True)
    result['regression']['ranked_features'] = [
        {'name': name, 'coef': round(float(coef), 6)} for name, coef in ranked
    ]

    # ─── 2. Per-category analysis ───
    categories = [metadata['categories'][i] for i in range(len(metadata['categories']))
                  if mask[i]] if len(mask) == len(metadata['categories']) else metadata['categories']
    cat_stats = {}
    for cat in sorted(set(categories)):
        cat_mask = np.array([c == cat for c in categories])
        if cat_mask.sum() < 5:
            continue
        cat_delta = y[cat_mask]
        cat_stats[cat] = {
            'count': int(cat_mask.sum()),
            'mean_delta': round(float(cat_delta.mean()), 6),
            'median_delta': round(float(np.median(cat_delta)), 6),
            'std_delta': round(float(cat_delta.std()), 6),
            'pct_mlp_helps': round(float((cat_delta > 0).mean() * 100), 1),
            'pct_linear_better': round(float((cat_delta < 0).mean() * 100), 1),
        }

    # Sort by mean_delta descending (most MLP-needy first)
    result['category_stats'] = dict(
        sorted(cat_stats.items(), key=lambda x: -x[1]['mean_delta']))

    # ─── 3. Top tokens that need MLP most / least ───
    # Aggregate by token_id: mean delta per unique token
    token_ids_clean = data['token_ids'][mask] if mask.sum() < len(data['token_ids']) else data['token_ids']
    token_deltas = {}
    token_counts = {}
    for tid, d in zip(token_ids_clean, y):
        tid = int(tid)
        if tid not in token_deltas:
            token_deltas[tid] = []
        token_deltas[tid].append(d)

    token_mean_delta = {}
    for tid, ds in token_deltas.items():
        if len(ds) >= 3:  # minimum 3 occurrences
            token_mean_delta[tid] = {
                'text': extractor.token_texts.get(tid, '?'),
                'mean_delta': round(float(np.mean(ds)), 6),
                'count': len(ds),
                'std': round(float(np.std(ds)), 6),
                'category': extractor.token_category.get(tid, '?'),
            }

    # Top 20 most MLP-needy tokens
    sorted_by_delta = sorted(token_mean_delta.items(),
                             key=lambda x: -x[1]['mean_delta'])
    result['top_need_mlp'] = [
        {'token_id': tid, **info} for tid, info in sorted_by_delta[:25]
    ]
    result['top_linear_ok'] = [
        {'token_id': tid, **info} for tid, info in sorted_by_delta[-25:][::-1]
    ]

    # ─── 4. Hypothesis tests ───
    hypotheses = {}

    # H1: Frequency predicts nonlinearity need
    freq_corr = np.corrcoef(features['log_frequency'][mask], y)[0, 1]
    hypotheses['H1_frequency'] = {
        'correlation': round(float(freq_corr), 4),
        'direction': 'rare tokens need MLP more' if freq_corr < 0
                     else 'frequent tokens need MLP more',
    }

    # H2: Next-token entropy
    if 'next_token_entropy' in features:
        ent_corr = np.corrcoef(features['next_token_entropy'][mask], y)[0, 1]
        hypotheses['H2_entropy'] = {
            'correlation': round(float(ent_corr), 4),
            'direction': 'high-entropy positions need MLP more' if ent_corr > 0
                         else 'low-entropy positions need MLP more',
        }

    # H3: Function word binary correlation
    fw_corr = np.corrcoef(features['is_function_word'][mask], y)[0, 1]
    hypotheses['H3_function_word'] = {
        'correlation': round(float(fw_corr), 4),
    }

    # H4: Subword
    sw_corr = np.corrcoef(features['is_subword'][mask], y)[0, 1]
    hypotheses['H4_subword'] = {
        'correlation': round(float(sw_corr), 4),
    }

    # H5: Position
    pos_corr = np.corrcoef(features['position_normalized'][mask], y)[0, 1]
    hypotheses['H5_position'] = {
        'correlation': round(float(pos_corr), 4),
    }

    # H6: Top-1 prediction agreement rate
    if 'top1_agree' in features:
        agree_corr = np.corrcoef(features['top1_agree'][mask], y)[0, 1]
        hypotheses['H6_prediction_agreement'] = {
            'correlation': round(float(agree_corr), 4),
            'agreement_rate': round(float(features['top1_agree'][mask].mean() * 100), 1),
        }

    # H7: KL divergence
    if 'kl_full_to_linear' in features:
        kl_vals = features['kl_full_to_linear'][mask]
        hypotheses['H7_kl_divergence'] = {
            'mean_kl': round(float(kl_vals.mean()), 4),
            'median_kl': round(float(np.median(kl_vals)), 4),
            'corr_with_delta': round(float(np.corrcoef(kl_vals, y)[0, 1]), 4),
        }

    result['hypotheses'] = hypotheses

    # ─── 5. Variance decomposition ───
    # How much variance in delta does each feature explain alone?
    var_explained = {}
    total_var = y.var()
    for fname in available:
        x_single = features[fname][mask].reshape(-1, 1)
        if np.isfinite(x_single).all():
            r = Ridge(alpha=1.0)
            r.fit(StandardScaler().fit_transform(x_single), y)
            var_explained[fname] = round(float(
                r.score(StandardScaler().fit_transform(x_single), y)), 4)

    result['univariate_r2'] = dict(
        sorted(var_explained.items(), key=lambda x: -x[1]))

    # ─── 6. Layer-specific narrative ───
    # What's the dominant story at this layer?
    top_feat = ranked[0][0] if ranked else 'unknown'
    top_cat = list(result['category_stats'].keys())[0] if result['category_stats'] else 'unknown'
    result['narrative'] = {
        'dominant_feature': top_feat,
        'most_mlp_needy_category': top_cat,
        'regression_r2': r_squared,
        'top3_need_mlp_tokens': [t['text'] for t in result['top_need_mlp'][:3]],
        'top3_linear_ok_tokens': [t['text'] for t in result['top_linear_ok'][:3]],
    }

    return result


# ─── Cross-layer summary ──────────────────────────────────────────────────────

def cross_layer_summary(all_results):
    """Generate cross-layer summary showing how feature importance shifts."""
    summary = {
        'feature_importance_by_layer': {},
        'dominant_feature_by_layer': {},
        'category_ranking_by_layer': {},
        'hypothesis_correlations_by_layer': {},
    }

    for res in all_results:
        layer = res['layer']
        # Feature importances
        summary['feature_importance_by_layer'][layer] = res['regression']['ranked_features']
        summary['dominant_feature_by_layer'][layer] = res['narrative']['dominant_feature']

        # Category ranking
        cats = list(res['category_stats'].keys())[:5]
        summary['category_ranking_by_layer'][layer] = cats

        # Hypothesis correlations
        h_corrs = {}
        for h_name, h_data in res['hypotheses'].items():
            if 'correlation' in h_data:
                h_corrs[h_name] = h_data['correlation']
        summary['hypothesis_correlations_by_layer'][layer] = h_corrs

    # Find if dominant feature shifts across layers
    layers = sorted(summary['dominant_feature_by_layer'].keys())
    if len(layers) >= 3:
        early = [summary['dominant_feature_by_layer'][l] for l in layers[:len(layers)//3]]
        mid = [summary['dominant_feature_by_layer'][l] for l in layers[len(layers)//3:2*len(layers)//3]]
        late = [summary['dominant_feature_by_layer'][l] for l in layers[2*len(layers)//3:]]
        summary['depth_pattern'] = {
            'early_dominant': Counter(early).most_common(1)[0][0] if early else '?',
            'mid_dominant': Counter(mid).most_common(1)[0][0] if mid else '?',
            'late_dominant': Counter(late).most_common(1)[0][0] if late else '?',
        }

    return summary


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Token Forensics: what predicts MLP nonlinearity need?")
    parser.add_argument("--model", required=True,
                        help="HuggingFace model name")
    parser.add_argument("--device", default="cpu",
                        help="cpu, cuda, or mps")
    parser.add_argument("--layers", default=None,
                        help="Comma-separated layer indices (e.g. 1,6,12,18,23)")
    parser.add_argument("--all-layers", action="store_true",
                        help="Run all layers")
    parser.add_argument("--n_tokens", type=int, default=15000,
                        help="Tokens per layer for analysis")
    parser.add_argument("--output", default=None,
                        help="Output JSON path (default: auto-named)")
    args = parser.parse_args()

    device = args.device
    print(f"Loading model: {args.model}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float32).to(device)
    model.eval()

    adapter = ModelAdapter(model, args.model)

    if args.all_layers:
        layer_indices = list(range(adapter.n_layers))
    elif args.layers:
        layer_indices = [int(x) for x in args.layers.split(',')]
    else:
        # Default: sample across depth
        n = adapter.n_layers
        layer_indices = sorted(set([0, n//6, n//3, n//2, 2*n//3, 5*n//6, n-1]))

    print(f"Analyzing layers: {layer_indices}", flush=True)
    extractor = TokenFeatureExtractor(tokenizer, model, device)

    all_results = []
    for layer_idx in layer_indices:
        print(f"\n{'='*60}", flush=True)
        print(f"Layer {layer_idx}", flush=True)
        print(f"{'='*60}", flush=True)

        t0 = time.time()
        data = collect_layer_data(adapter, tokenizer, layer_idx,
                                  n_tokens=args.n_tokens, device=device)
        t_collect = time.time() - t0
        print(f"  Data collection: {t_collect:.1f}s", flush=True)

        t0 = time.time()
        result = analyze_layer(data, extractor, layer_idx)
        t_analyze = time.time() - t0
        print(f"  Analysis: {t_analyze:.1f}s", flush=True)
        result['time_seconds'] = round(t_collect + t_analyze, 1)

        # Print summary
        print(f"\n  --- Layer {layer_idx} Summary ---", flush=True)
        print(f"  R² (all features): {result['regression']['r_squared']}", flush=True)
        print(f"  Top features:", flush=True)
        for feat in result['regression']['ranked_features'][:5]:
            r2_solo = result['univariate_r2'].get(feat['name'], '?')
            print(f"    {feat['name']:25s}  coef={feat['coef']:+.4f}  solo_R²={r2_solo}", flush=True)

        print(f"\n  Hypotheses:", flush=True)
        for h_name, h_data in result['hypotheses'].items():
            corr = h_data.get('correlation', '?')
            extra = h_data.get('direction', '')
            print(f"    {h_name:30s}  r={corr}  {extra}", flush=True)

        print(f"\n  Most MLP-needy categories:", flush=True)
        for cat, stats in list(result['category_stats'].items())[:5]:
            print(f"    {cat:20s}  mean_delta={stats['mean_delta']:+.4f}  "
                  f"n={stats['count']:5d}  {stats['pct_mlp_helps']:.0f}% MLP helps", flush=True)

        print(f"\n  Tokens needing MLP most:", flush=True)
        for t in result['top_need_mlp'][:8]:
            print(f"    {t['text']:15s}  delta={t['mean_delta']:+.4f}  "
                  f"n={t['count']:4d}  [{t['category']}]", flush=True)

        print(f"\n  Tokens fine with linear:", flush=True)
        for t in result['top_linear_ok'][:8]:
            print(f"    {t['text']:15s}  delta={t['mean_delta']:+.4f}  "
                  f"n={t['count']:4d}  [{t['category']}]", flush=True)

        all_results.append(result)

        # Free memory
        del data
        import gc
        gc.collect()

    # Cross-layer summary
    summary = cross_layer_summary(all_results)

    print(f"\n{'='*60}", flush=True)
    print(f"CROSS-LAYER SUMMARY", flush=True)
    print(f"{'='*60}", flush=True)
    if 'depth_pattern' in summary:
        dp = summary['depth_pattern']
        print(f"  Early layers dominated by: {dp['early_dominant']}", flush=True)
        print(f"  Mid layers dominated by:   {dp['mid_dominant']}", flush=True)
        print(f"  Late layers dominated by:  {dp['late_dominant']}", flush=True)

    print(f"\n  Hypothesis correlations across layers:", flush=True)
    for layer in sorted(summary['hypothesis_correlations_by_layer'].keys()):
        h_corrs = summary['hypothesis_correlations_by_layer'][layer]
        best_h = max(h_corrs.items(), key=lambda x: abs(x[1])) if h_corrs else ('?', 0)
        print(f"    L{layer:2d}: strongest = {best_h[0]} (r={best_h[1]:+.3f})", flush=True)

    # Save results
    output_path = args.output or os.path.join(
        os.path.dirname(__file__),
        f"forensics_{args.model.replace('/', '_')}.json")

    output = {
        'model': args.model,
        'n_tokens': args.n_tokens,
        'layers_analyzed': layer_indices,
        'per_layer': all_results,
        'cross_layer_summary': summary,
    }

    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to {output_path}", flush=True)


if __name__ == '__main__':
    main()
