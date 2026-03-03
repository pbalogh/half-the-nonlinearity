"""
Context vs Token Gating: Who Decides the Route?

Tests whether MLP nonlinearity need is determined by:
  (a) the token's identity (its embedding alone)
  (b) the context (everything attention/previous layers added)
  (c) both together (the full activation)

Method:
  At each layer, the MLP input activation is:
    x_i = e_i + c_i
  where:
    e_i = token embedding (+ positional embedding at L0)
    c_i = x_i - e_i = contextual contribution

  We train three gates:
    1. Full gate:    g(x_i)  — the standard gate (baseline)
    2. Token gate:   g(e_i)  — sees only token identity
    3. Context gate: g(c_i)  — sees only what context contributed

  If context gate ≈ full gate >> token gate:
    → routing is contextual ("read the room")
  If token gate ≈ full gate >> context gate:
    → routing is lexical ("check the guest list")
  If full gate >> both:
    → routing needs the interaction

Usage:
  python context_vs_token_gating.py --model gpt2-medium --layers 1,12,23
  python context_vs_token_gating.py --model gpt2-medium --all-layers
  python context_vs_token_gating.py --model EleutherAI/pythia-410m --layers 0,6,12,23
"""

import torch
import torch.nn as nn
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score
import json, time, argparse, os, sys, warnings

warnings.filterwarnings("ignore", category=FutureWarning)

sys.path.insert(0, os.path.dirname(__file__))
from scale_test_universal import ModelAdapter


# ─── Data Collection ──────────────────────────────────────────────────────────

def get_token_embeddings(adapter, token_ids, device='cpu'):
    """Get raw token embeddings (no positional) for a set of token ids."""
    embedding_layer = adapter.get_embedding()
    with torch.no_grad():
        ids = torch.tensor(token_ids, dtype=torch.long, device=device)
        embeds = embedding_layer(ids)
    return embeds.cpu().numpy()


def collect_decomposed_data(adapter, tokenizer, layer_idx, n_tokens=15000,
                            device='cpu'):
    """
    Collect per-position data decomposed into token embedding and context.

    Returns:
        x_full:     (N, d) full MLP input activations
        e_token:    (N, d) token embeddings (including positional if applicable)
        c_context:  (N, d) contextual contribution (x - e)
        delta:      (N,) loss difference (L_lin - L_full)
        token_ids:  (N,) token ids
        token_strs: list of token strings
    """
    model = adapter.model
    mlp = adapter.get_mlp(layer_idx)
    hidden_dim = adapter.hidden_dim

    # Load data
    dataset = load_dataset("wikitext", "wikitext-103-raw-v1", split="test")
    text = "\n".join([x for x in dataset["text"] if x.strip()])
    tokens = tokenizer.encode(text)

    # ─── Step 1: Fit linear approximation ───
    print(f"  Fitting linear approximation...", flush=True)
    fit_toks = torch.tensor([tokens[:10000]], dtype=torch.long).to(device)

    acts_fit, outs_fit = [], []
    original_forward = mlp.forward

    def capture_hook(x, *args, **kwargs):
        acts_fit.append(x.detach().cpu())
        out = original_forward(x, *args, **kwargs)
        outs_fit.append(out.detach().cpu())
        return out

    mlp.forward = capture_hook
    with torch.no_grad():
        for start in range(0, min(10000, len(tokens)), 512):
            end = min(start + 512, 10000)
            model(fit_toks[:, start:end])
    mlp.forward = original_forward

    X_fit = torch.cat(acts_fit, dim=1).squeeze(0).numpy()
    Y_fit = torch.cat(outs_fit, dim=1).squeeze(0).numpy()

    X_mean, Y_mean = X_fit.mean(0), Y_fit.mean(0)
    Xc, Yc = X_fit - X_mean, Y_fit - Y_mean
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    lam = 0.01
    S_inv = S / (S**2 + lam)
    W_lin = (Vt.T * S_inv) @ U.T @ Yc
    b_lin = Y_mean - X_mean @ W_lin

    # ─── Step 2: Get positional embeddings ───
    # For GPT-2: wte(token_id) + wpe(position)
    # For Pythia/NeoX: just wte(token_id) — rotary handles position differently
    print(f"  Extracting embeddings...", flush=True)

    # ─── Step 3: Collect per-token data ───
    print(f"  Collecting per-token decomposed data...", flush=True)
    eval_start = 30000
    eval_tokens = tokens[eval_start:eval_start + n_tokens + 512]

    all_x_full = []
    all_e_token = []
    all_delta = []
    all_token_ids = []

    for chunk_start in range(0, min(n_tokens, len(eval_tokens) - 1), 512):
        chunk_end = min(chunk_start + 512, len(eval_tokens))
        input_ids = torch.tensor(
            [eval_tokens[chunk_start:chunk_end]], dtype=torch.long).to(device)
        seq_len = input_ids.shape[1]
        if seq_len < 2:
            continue

        # Get token + positional embeddings
        with torch.no_grad():
            if adapter.family == 'gpt2':
                wte = model.transformer.wte(input_ids)
                positions = torch.arange(seq_len, device=device).unsqueeze(0)
                wpe = model.transformer.wpe(positions)
                embed = (wte + wpe).squeeze(0).cpu().numpy()  # (seq, d)
            elif adapter.family == 'gpt_neox':
                # Pythia: rotary embeddings, no additive positional
                embed_layer = model.gpt_neox.embed_in
                embed = embed_layer(input_ids).squeeze(0).cpu().numpy()
            elif adapter.family in ('llama', 'mistral'):
                embed = model.model.embed_tokens(input_ids).squeeze(0).cpu().numpy()
            else:
                embed = adapter.get_embedding()(input_ids).squeeze(0).cpu().numpy()

        # Get full MLP-input activations
        captured = []
        def capture_only(x, *args, **kwargs):
            captured.append(x.detach().cpu())
            return original_forward(x, *args, **kwargs)

        mlp.forward = capture_only
        with torch.no_grad():
            out_full = model(input_ids)
            logits_full = out_full.logits[0, :-1].cpu().float()
        mlp.forward = original_forward

        x_full = captured[0].squeeze(0).numpy()  # (seq, d)

        # Get per-token losses (full MLP)
        targets = input_ids[0, 1:].cpu()
        losses_full = nn.CrossEntropyLoss(reduction='none')(
            logits_full, targets).numpy()

        # Get per-token losses (linear MLP)
        def linear_fwd(x, *args, **kwargs):
            x_np = x.detach().cpu().numpy()
            out_np = x_np @ W_lin + b_lin
            return torch.tensor(out_np, dtype=x.dtype, device=x.device)

        mlp.forward = linear_fwd
        with torch.no_grad():
            out_lin = model(input_ids)
            logits_lin = out_lin.logits[0, :-1].cpu().float()
        mlp.forward = original_forward

        losses_lin = nn.CrossEntropyLoss(reduction='none')(
            logits_lin, targets).numpy()

        delta = losses_lin - losses_full  # positive = MLP helps

        # Align: we predict next token, so activations at position i
        # correspond to predicting token i+1. Use positions 0..seq-2.
        all_x_full.append(x_full[:-1])
        all_e_token.append(embed[:-1])
        all_delta.append(delta)
        all_token_ids.append(targets.numpy())

        collected = sum(len(x) for x in all_delta)
        if collected % 5000 < 512:
            print(f"    {collected}/{n_tokens} tokens", flush=True)
        if collected >= n_tokens:
            break

    x_full = np.concatenate(all_x_full)
    e_token = np.concatenate(all_e_token)
    c_context = x_full - e_token  # the key decomposition
    delta = np.concatenate(all_delta)
    token_ids = np.concatenate(all_token_ids)
    token_strs = [tokenizer.decode([t]) for t in token_ids]

    return x_full, e_token, c_context, delta, token_ids, token_strs


# ─── Gate Training & Evaluation ──────────────────────────────────────────────

def train_and_eval_gate(X_train, y_train, X_eval, y_eval, name="gate"):
    """
    Train a logistic regression gate and evaluate.

    Returns dict with accuracy, AUC, pct routed linear, and gate direction info.
    """
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_train)
    X_ev = scaler.transform(X_eval)

    clf = LogisticRegression(C=1.0, max_iter=1000, solver='lbfgs')
    clf.fit(X_tr, y_train)

    pred = clf.predict(X_ev)
    prob = clf.predict_proba(X_ev)[:, 1]

    acc = accuracy_score(y_eval, pred)
    try:
        auc = roc_auc_score(y_eval, prob)
    except ValueError:
        auc = 0.5

    pct_linear = (pred == 1).mean() * 100

    # Get gate direction in original space
    direction = clf.coef_[0] * scaler.scale_
    direction = direction / (np.linalg.norm(direction) + 1e-8)

    return {
        'name': name,
        'accuracy': round(float(acc), 4),
        'auc': round(float(auc), 4),
        'pct_linear': round(float(pct_linear), 1),
        'n_params': int(X_train.shape[1] + 1),
        'direction_norm': round(float(np.linalg.norm(clf.coef_[0])), 4),
    }


def analyze_layer(adapter, tokenizer, layer_idx, n_tokens=15000, device='cpu'):
    """Full context-vs-token analysis for one layer."""
    print(f"\n{'='*60}", flush=True)
    print(f"Layer {layer_idx}", flush=True)
    print(f"{'='*60}", flush=True)

    t0 = time.time()
    x_full, e_token, c_context, delta, token_ids, token_strs = \
        collect_decomposed_data(adapter, tokenizer, layer_idx, n_tokens, device)
    t_collect = time.time() - t0
    print(f"  Data collection: {t_collect:.1f}s", flush=True)

    N = len(delta)
    hidden_dim = x_full.shape[1]

    # Basic stats on decomposition
    e_norms = np.linalg.norm(e_token, axis=1)
    c_norms = np.linalg.norm(c_context, axis=1)
    x_norms = np.linalg.norm(x_full, axis=1)

    # How much of x is context vs token?
    context_fraction = c_norms / (x_norms + 1e-8)

    # Cosine between e and c
    cos_ec = np.sum(e_token * c_context, axis=1) / (e_norms * c_norms + 1e-8)

    decomp_stats = {
        'mean_token_norm': round(float(e_norms.mean()), 4),
        'mean_context_norm': round(float(c_norms.mean()), 4),
        'mean_full_norm': round(float(x_norms.mean()), 4),
        'context_fraction_mean': round(float(context_fraction.mean()), 4),
        'context_fraction_std': round(float(context_fraction.std()), 4),
        'cos_token_context_mean': round(float(cos_ec.mean()), 4),
        'cos_token_context_std': round(float(cos_ec.std()), 4),
    }
    print(f"  Decomposition: token_norm={decomp_stats['mean_token_norm']:.2f}, "
          f"context_norm={decomp_stats['mean_context_norm']:.2f}, "
          f"context_fraction={decomp_stats['context_fraction_mean']:.2f}", flush=True)

    # ─── Create binary labels ───
    # "Linear OK" = delta <= median (MLP doesn't help much)
    threshold = np.median(delta)
    labels = (delta <= threshold).astype(int)  # 1 = linear OK, 0 = need MLP

    # Also try a stricter threshold: bottom quartile vs rest
    q25 = np.percentile(delta, 25)
    labels_strict = (delta <= q25).astype(int)

    # ─── Train/eval split ───
    split = int(N * 0.7)
    idx = np.random.RandomState(42).permutation(N)
    train_idx, eval_idx = idx[:split], idx[split:]

    results = {'layer': layer_idx, 'n_tokens': N, 'decomposition': decomp_stats}

    # ─── Train gates on different inputs ───
    print(f"  Training gates...", flush=True)

    for label_name, y in [('median', labels), ('q25', labels_strict)]:
        y_train, y_eval = y[train_idx], y[eval_idx]

        gates = {}

        # 1. Full activation gate
        gates['full'] = train_and_eval_gate(
            x_full[train_idx], y_train,
            x_full[eval_idx], y_eval, "full")

        # 2. Token embedding gate
        gates['token_only'] = train_and_eval_gate(
            e_token[train_idx], y_train,
            e_token[eval_idx], y_eval, "token_only")

        # 3. Context-only gate
        gates['context_only'] = train_and_eval_gate(
            c_context[train_idx], y_train,
            c_context[eval_idx], y_eval, "context_only")

        # 4. Token type gate (mean embedding per token type — no context at all)
        # Average e_token across all instances of same token id
        token_means = {}
        for i in train_idx:
            tid = int(token_ids[i])
            if tid not in token_means:
                token_means[tid] = []
            token_means[tid].append(e_token[i])
        token_means = {tid: np.mean(vecs, axis=0) for tid, vecs in token_means.items()}
        # For eval, use the mean embedding (or raw if unseen)
        e_type_train = np.array([token_means.get(int(token_ids[i]),
                                 e_token[i]) for i in train_idx])
        e_type_eval = np.array([token_means.get(int(token_ids[i]),
                                e_token[i]) for i in eval_idx])
        gates['token_type'] = train_and_eval_gate(
            e_type_train, y_train,
            e_type_eval, y_eval, "token_type")

        # 5. Random baseline
        rng = np.random.RandomState(42)
        random_preds = rng.randint(0, 2, size=len(y_eval))
        random_acc = accuracy_score(y_eval, random_preds)
        gates['random'] = {
            'name': 'random',
            'accuracy': round(float(random_acc), 4),
            'auc': 0.5,
            'pct_linear': round(float(random_preds.mean() * 100), 1),
            'n_params': 0,
        }

        # 6. Context norm only (1D — just the magnitude of context contribution)
        c_norm_feat = c_norms.reshape(-1, 1)
        gates['context_norm_only'] = train_and_eval_gate(
            c_norm_feat[train_idx], y_train,
            c_norm_feat[eval_idx], y_eval, "context_norm_only")

        # 7. Delta between token embedding and context (element-wise interaction)
        interaction = e_token * c_context  # Hadamard product
        gates['interaction'] = train_and_eval_gate(
            interaction[train_idx], y_train,
            interaction[eval_idx], y_eval, "interaction")

        results[f'gates_{label_name}'] = gates

        # Print comparison
        print(f"\n  --- Gate comparison ({label_name} threshold) ---", flush=True)
        for gname in ['random', 'token_type', 'token_only', 'context_only',
                       'context_norm_only', 'full', 'interaction']:
            g = gates[gname]
            print(f"    {gname:20s}  acc={g['accuracy']:.3f}  "
                  f"auc={g['auc']:.3f}  params={g.get('n_params', 0)}", flush=True)

    # ─── Correlation analysis ───
    # How does delta correlate with norms and cosine?
    corr_results = {
        'delta_vs_token_norm': round(float(np.corrcoef(e_norms, delta)[0, 1]), 4),
        'delta_vs_context_norm': round(float(np.corrcoef(c_norms, delta)[0, 1]), 4),
        'delta_vs_full_norm': round(float(np.corrcoef(x_norms, delta)[0, 1]), 4),
        'delta_vs_context_fraction': round(float(np.corrcoef(context_fraction, delta)[0, 1]), 4),
        'delta_vs_cos_ec': round(float(np.corrcoef(cos_ec, delta)[0, 1]), 4),
    }
    results['correlations'] = corr_results

    print(f"\n  Correlations with delta:", flush=True)
    for k, v in corr_results.items():
        print(f"    {k:35s}  r={v:+.4f}", flush=True)

    # ─── Per-token consistency ───
    # For tokens that appear multiple times, how consistent is the routing?
    # If it's the token identity, same token should always route the same way.
    # If it's context, same token should vary.
    token_routing_var = {}
    for i in range(N):
        tid = int(token_ids[i])
        if tid not in token_routing_var:
            token_routing_var[tid] = []
        token_routing_var[tid].append(delta[i])

    # For tokens with 5+ occurrences, measure variance
    consistent_tokens = []
    variable_tokens = []
    for tid, deltas in token_routing_var.items():
        if len(deltas) >= 5:
            var = np.var(deltas)
            mean_d = np.mean(deltas)
            text = tokenizer.decode([tid])
            entry = {'token': text, 'count': len(deltas),
                     'mean_delta': round(float(mean_d), 4),
                     'var_delta': round(float(var), 4),
                     'cv': round(float(np.std(deltas) / (abs(mean_d) + 1e-6)), 2)}
            if var < np.percentile([np.var(d) for d in token_routing_var.values()
                                    if len(d) >= 5], 25):
                consistent_tokens.append(entry)
            else:
                variable_tokens.append(entry)

    # Most consistent (same routing regardless of context)
    consistent_tokens.sort(key=lambda x: x['var_delta'])
    # Most variable (routing depends heavily on context)
    variable_tokens.sort(key=lambda x: -x['var_delta'])

    results['routing_consistency'] = {
        'most_consistent': consistent_tokens[:15],
        'most_variable': variable_tokens[:15],
        'overall_mean_variance': round(float(np.mean([
            np.var(d) for d in token_routing_var.values() if len(d) >= 5
        ])), 6),
    }

    print(f"\n  Routing consistency:", flush=True)
    print(f"  Most CONSISTENT (same routing regardless of context):", flush=True)
    for t in consistent_tokens[:8]:
        print(f"    {t['token']:15s}  var={t['var_delta']:.4f}  "
              f"mean_delta={t['mean_delta']:+.4f}  n={t['count']}", flush=True)
    print(f"  Most VARIABLE (routing depends on context):", flush=True)
    for t in variable_tokens[:8]:
        print(f"    {t['token']:15s}  var={t['var_delta']:.4f}  "
              f"mean_delta={t['mean_delta']:+.4f}  n={t['count']}", flush=True)

    results['time_seconds'] = round(time.time() - t0, 1)
    return results


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Context vs Token: what decides MLP routing?")
    parser.add_argument("--model", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--layers", default=None)
    parser.add_argument("--all-layers", action="store_true")
    parser.add_argument("--n_tokens", type=int, default=15000)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    print(f"Loading model: {args.model}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float32).to(args.device)
    model.eval()

    adapter = ModelAdapter(model, args.model)

    if args.all_layers:
        layer_indices = list(range(adapter.n_layers))
    elif args.layers:
        layer_indices = [int(x) for x in args.layers.split(',')]
    else:
        n = adapter.n_layers
        layer_indices = sorted(set([0, n//4, n//2, 3*n//4, n-1]))

    all_results = []
    for layer_idx in layer_indices:
        result = analyze_layer(adapter, tokenizer, layer_idx,
                               args.n_tokens, args.device)
        all_results.append(result)

    # ─── Cross-layer summary ───
    print(f"\n{'='*60}", flush=True)
    print(f"CROSS-LAYER VERDICT", flush=True)
    print(f"{'='*60}", flush=True)
    print(f"\n  {'Layer':>5s}  {'Full':>7s}  {'Token':>7s}  {'Context':>7s}  "
          f"{'TokType':>7s}  {'Inter':>7s}  {'Random':>7s}  Winner", flush=True)
    print(f"  {'-'*5:>5s}  {'-'*7:>7s}  {'-'*7:>7s}  {'-'*7:>7s}  "
          f"{'-'*7:>7s}  {'-'*7:>7s}  {'-'*7:>7s}  ------", flush=True)

    for res in all_results:
        g = res['gates_median']
        full_auc = g['full']['auc']
        tok_auc = g['token_only']['auc']
        ctx_auc = g['context_only']['auc']
        typ_auc = g['token_type']['auc']
        int_auc = g['interaction']['auc']
        rnd_auc = g['random']['auc']

        # Determine winner
        scores = {'full': full_auc, 'token': tok_auc, 'context': ctx_auc,
                  'type': typ_auc, 'interaction': int_auc}
        winner = max(scores, key=scores.get)

        # Is context sufficient? (within 0.01 of full)
        ctx_sufficient = "✓" if abs(ctx_auc - full_auc) < 0.01 else ""
        tok_sufficient = "✓" if abs(tok_auc - full_auc) < 0.01 else ""

        print(f"  L{res['layer']:>3d}  {full_auc:.4f}  {tok_auc:.4f}  "
              f"{ctx_auc:.4f}  {typ_auc:.4f}  {int_auc:.4f}  "
              f"{rnd_auc:.4f}  {winner}"
              f"{'  (ctx≈full)' if ctx_sufficient else ''}"
              f"{'  (tok≈full)' if tok_sufficient else ''}",
              flush=True)

    # Save
    output_path = args.output or os.path.join(
        os.path.dirname(__file__),
        f"context_vs_token_{args.model.replace('/', '_')}.json")

    output = {
        'model': args.model,
        'per_layer': all_results,
    }
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nSaved to {output_path}", flush=True)


if __name__ == '__main__':
    main()
