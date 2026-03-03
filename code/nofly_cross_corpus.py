"""
NoFly Cross-Corpus Stability Test

Build NoFly lists from WikiText-103, test on completely different corpora.
If the same tokens show high deltas in different text, they're real.
If not, they're just "words about ironclads."

Uses large samples (100K+ tokens per corpus) for statistical power.

Usage:
  python3 nofly_cross_corpus.py --model gpt2-medium --layer 12
  python3 nofly_cross_corpus.py --model gpt2-medium --layers 1,12,23
"""

import torch
import torch.nn as nn
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from collections import defaultdict
import json, time, argparse, os, sys, warnings

warnings.filterwarnings("ignore", category=FutureWarning)

sys.path.insert(0, os.path.dirname(__file__))
from scale_test_universal import ModelAdapter


def fit_linear_mlp(adapter, tokens, layer_idx, n_fit=10000, device='cpu'):
    """Fit linear approximation to MLP at given layer."""
    model = adapter.model
    mlp = adapter.get_mlp(layer_idx)
    original_forward = mlp.forward

    fit_toks = torch.tensor([tokens[:n_fit]], dtype=torch.long).to(device)
    acts, outs = [], []

    def capture(x, *args, **kwargs):
        acts.append(x.detach().cpu())
        out = original_forward(x, *args, **kwargs)
        outs.append(out.detach().cpu())
        return out

    mlp.forward = capture
    with torch.no_grad():
        for start in range(0, n_fit, 512):
            end = min(start + 512, n_fit)
            model(fit_toks[:, start:end])
    mlp.forward = original_forward

    X = torch.cat(acts, dim=1).squeeze(0).numpy()
    Y = torch.cat(outs, dim=1).squeeze(0).numpy()
    Xm, Ym = X.mean(0), Y.mean(0)
    U, S, Vt = np.linalg.svd(X - Xm, full_matrices=False)
    S_inv = S / (S**2 + 0.01)
    W = (Vt.T * S_inv) @ U.T @ (Y - Ym)
    b = Ym - Xm @ W
    return W, b, original_forward


def collect_token_deltas(adapter, tokenizer, tokens, layer_idx, W_lin, b_lin,
                         original_forward, n_tokens=50000, start_offset=0,
                         device='cpu'):
    """Collect per-token-type deltas from a token sequence."""
    model = adapter.model
    mlp = adapter.get_mlp(layer_idx)

    token_deltas = defaultdict(list)
    eval_tokens = tokens[start_offset:start_offset + n_tokens + 512]

    collected = 0
    for chunk_start in range(0, min(n_tokens, len(eval_tokens) - 1), 512):
        chunk_end = min(chunk_start + 512, len(eval_tokens))
        input_ids = torch.tensor(
            [eval_tokens[chunk_start:chunk_end]], dtype=torch.long).to(device)
        if input_ids.shape[1] < 2:
            continue

        # Full MLP loss
        mlp.forward = original_forward
        with torch.no_grad():
            logits_full = model(input_ids).logits[0, :-1].cpu().float()
        targets = input_ids[0, 1:].cpu()
        losses_full = nn.CrossEntropyLoss(reduction='none')(
            logits_full, targets).numpy()

        # Linear MLP loss
        def linear_fwd(x, *args, **kwargs):
            x_np = x.detach().cpu().numpy()
            return torch.tensor(x_np @ W_lin + b_lin, dtype=x.dtype, device=x.device)

        mlp.forward = linear_fwd
        with torch.no_grad():
            logits_lin = model(input_ids).logits[0, :-1].cpu().float()
        mlp.forward = original_forward

        losses_lin = nn.CrossEntropyLoss(reduction='none')(
            logits_lin, targets).numpy()
        delta = losses_lin - losses_full

        for i, tid in enumerate(targets.numpy()):
            token_deltas[int(tid)].append(float(delta[i]))

        collected += len(targets)
        if collected % 10000 < 512:
            print(f"    {collected}/{n_tokens}", flush=True)
        if collected >= n_tokens:
            break

    return token_deltas


def load_corpus(name, tokenizer, max_tokens=200000):
    """Load and tokenize a corpus. Returns token list."""
    print(f"  Loading {name}...", flush=True)

    if name == 'wikitext':
        ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="test")
        text = "\n".join([x for x in ds["text"] if x.strip()])
    elif name == 'wikitext-train':
        # Different split = different articles
        ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="train")
        text = "\n".join([x for x in ds["text"][:5000] if x.strip()])
    elif name == 'bookcorpus':
        # Use a subset of books
        ds = load_dataset("bookcorpus", split="train", streaming=True)
        chunks = []
        total = 0
        for ex in ds:
            chunks.append(ex["text"])
            total += len(ex["text"])
            if total > max_tokens * 5:  # rough char-to-token ratio
                break
        text = "\n".join(chunks)
    elif name == 'openwebtext':
        ds = load_dataset("stas/openwebtext-10k", split="train")
        text = "\n".join([x for x in ds["text"] if x.strip()])
    elif name == 'c4':
        ds = load_dataset("allenai/c4", "en", split="validation", streaming=True)
        chunks = []
        total = 0
        for ex in ds:
            chunks.append(ex["text"])
            total += len(ex["text"])
            if total > max_tokens * 5:
                break
        text = "\n".join(chunks)
    elif name == 'lambada':
        ds = load_dataset("lambada", split="test")
        text = "\n".join([x for x in ds["text"] if x.strip()])
    else:
        raise ValueError(f"Unknown corpus: {name}")

    tokens = tokenizer.encode(text)
    print(f"  {name}: {len(tokens)} tokens", flush=True)
    return tokens[:max_tokens]


def identify_nofly(token_deltas, min_obs=10, min_mean_delta=0.05,
                   min_frac_positive=0.6):
    """Identify NoFly tokens from delta observations."""
    nofly = {}
    for tid, deltas in token_deltas.items():
        if len(deltas) < min_obs:
            continue
        arr = np.array(deltas)
        mean_d = arr.mean()
        frac_pos = (arr > 0).mean()
        if mean_d > min_mean_delta and frac_pos > min_frac_positive:
            nofly[tid] = {
                'mean': float(mean_d),
                'std': float(arr.std()),
                'n': len(deltas),
                'frac_pos': float(frac_pos),
            }
    return nofly


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='gpt2-medium')
    parser.add_argument('--layers', default='12')
    parser.add_argument('--n_tokens', type=int, default=50000,
                        help='Tokens per corpus for eval')
    parser.add_argument('--min_obs', type=int, default=10,
                        help='Min observations to classify a token')
    parser.add_argument('--output', default=None)
    args = parser.parse_args()

    layers = [int(x) for x in args.layers.split(',')]
    model_short = args.model.split('/')[-1]

    # Corpora to test
    # Build NoFly from wikitext-test, test on others
    train_corpus = 'wikitext'
    test_corpora = ['wikitext-train', 'openwebtext', 'lambada']

    print(f"Loading {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float32)
    model.eval()
    adapter = ModelAdapter(model, args.model)

    # Load all corpora upfront
    print("\n=== Loading corpora ===")
    corpus_tokens = {}
    corpus_tokens[train_corpus] = load_corpus(train_corpus, tokenizer)
    for name in test_corpora:
        try:
            corpus_tokens[name] = load_corpus(name, tokenizer)
        except Exception as e:
            print(f"  ⚠️  Skipping {name}: {e}")

    all_results = {}

    for layer_idx in layers:
        print(f"\n{'='*80}")
        print(f"Layer {layer_idx}")
        print(f"{'='*80}")

        # Fit linear MLP on wikitext
        print(f"\n  Fitting linear MLP on {train_corpus}...")
        W, b, orig_fwd = fit_linear_mlp(
            adapter, corpus_tokens[train_corpus], layer_idx)

        # Build NoFly list from training corpus
        print(f"\n  Building NoFly list from {train_corpus} "
              f"({args.n_tokens} tokens)...")
        train_deltas = collect_token_deltas(
            adapter, tokenizer, corpus_tokens[train_corpus], layer_idx,
            W, b, orig_fwd, n_tokens=args.n_tokens, start_offset=30000)

        nofly_train = identify_nofly(train_deltas, min_obs=args.min_obs)
        print(f"  NoFly tokens identified: {len(nofly_train)}")

        if not nofly_train:
            print("  No NoFly tokens found! Skipping layer.")
            continue

        # Show top NoFly tokens
        sorted_nf = sorted(nofly_train.items(),
                           key=lambda x: x[1]['mean'], reverse=True)
        print(f"\n  Top 20 NoFly tokens (from {train_corpus}):")
        print(f"  {'Token':>20} {'MeanΔ':>8} {'StdΔ':>8} {'N':>5} {'%Pos':>6}")
        for tid, info in sorted_nf[:20]:
            tok = tokenizer.decode([int(tid)]).strip() or repr(tokenizer.decode([int(tid)]))
            print(f"  {tok:>20} {info['mean']:>8.4f} {info['std']:>8.4f} "
                  f"{info['n']:>5} {info['frac_pos']:>6.1%}")

        # Test on each corpus
        layer_results = {
            'train_corpus': train_corpus,
            'n_nofly': len(nofly_train),
            'nofly_tokens': {
                int(tid): {
                    'text': tokenizer.decode([int(tid)]),
                    **info
                } for tid, info in sorted_nf
            },
            'cross_corpus': {}
        }

        for test_name in test_corpora:
            if test_name not in corpus_tokens:
                continue

            print(f"\n  Testing NoFly list on {test_name}...")

            # Refit linear MLP on THIS corpus (fair comparison)
            print(f"    Refitting linear MLP on {test_name}...")
            W_test, b_test, orig_fwd_test = fit_linear_mlp(
                adapter, corpus_tokens[test_name], layer_idx)

            test_deltas = collect_token_deltas(
                adapter, tokenizer, corpus_tokens[test_name], layer_idx,
                W_test, b_test, orig_fwd_test,
                n_tokens=args.n_tokens, start_offset=10000)

            # Check how NoFly tokens behave in test corpus
            n_found = 0
            n_still_nofly = 0
            n_flipped = 0
            n_insufficient = 0
            token_results = []

            for tid_str, train_info in nofly_train.items():
                tid = int(tid_str)
                tok = tokenizer.decode([tid]).strip() or repr(tokenizer.decode([tid]))

                if tid not in test_deltas or len(test_deltas[tid]) < 3:
                    n_insufficient += 1
                    continue

                n_found += 1
                test_arr = np.array(test_deltas[tid])
                test_mean = test_arr.mean()
                test_frac_pos = (test_arr > 0).mean()

                still_nofly = test_mean > 0.05 and test_frac_pos > 0.6
                if still_nofly:
                    n_still_nofly += 1
                elif test_mean < 0:
                    n_flipped += 1

                token_results.append({
                    'token': tok,
                    'token_id': tid,
                    'train_mean': train_info['mean'],
                    'train_n': train_info['n'],
                    'test_mean': float(test_mean),
                    'test_std': float(test_arr.std()),
                    'test_n': len(test_deltas[tid]),
                    'test_frac_pos': float(test_frac_pos),
                    'still_nofly': still_nofly,
                })

            # Correlation between train and test deltas
            if token_results:
                train_means = [r['train_mean'] for r in token_results]
                test_means = [r['test_mean'] for r in token_results]
                corr = np.corrcoef(train_means, test_means)[0, 1]
            else:
                corr = None

            print(f"\n  Results on {test_name}:")
            print(f"    NoFly tokens found in test corpus: {n_found}/{len(nofly_train)}")
            print(f"    Still NoFly: {n_still_nofly}/{n_found} "
                  f"({n_still_nofly/max(n_found,1):.1%})")
            print(f"    Flipped (negative delta): {n_flipped}/{n_found} "
                  f"({n_flipped/max(n_found,1):.1%})")
            print(f"    Insufficient obs: {n_insufficient}")
            if corr is not None:
                print(f"    Train↔Test delta correlation: r={corr:.3f}")

            # Show token-by-token comparison
            token_results.sort(key=lambda r: r['test_mean'], reverse=True)
            print(f"\n    {'Token':>20} {'TrainΔ':>8} {'TestΔ':>8} "
                  f"{'TrainN':>6} {'TestN':>6} {'Status':>10}")
            for r in token_results[:30]:
                status = "✓ NoFly" if r['still_nofly'] else \
                         "✗ FLIP" if r['test_mean'] < 0 else "~ weak"
                print(f"    {r['token']:>20} {r['train_mean']:>8.4f} "
                      f"{r['test_mean']:>8.4f} {r['train_n']:>6} "
                      f"{r['test_n']:>6} {status:>10}")

            layer_results['cross_corpus'][test_name] = {
                'n_found': n_found,
                'n_still_nofly': n_still_nofly,
                'n_flipped': n_flipped,
                'n_insufficient': n_insufficient,
                'retention_rate': n_still_nofly / max(n_found, 1),
                'correlation': float(corr) if corr is not None else None,
                'tokens': token_results,
            }

        all_results[f'L{layer_idx}'] = layer_results

    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    for layer_name, lr in all_results.items():
        print(f"\n{layer_name}: {lr['n_nofly']} NoFly tokens from {lr['train_corpus']}")
        for corpus, cr in lr.get('cross_corpus', {}).items():
            print(f"  → {corpus}: {cr['n_still_nofly']}/{cr['n_found']} retained "
                  f"({cr['retention_rate']:.1%}), "
                  f"r={cr['correlation']:.3f}" if cr['correlation'] else
                  f"  → {corpus}: {cr['n_still_nofly']}/{cr['n_found']} retained "
                  f"({cr['retention_rate']:.1%})")

    # Save
    out_path = args.output or f"nofly_cross_corpus_{model_short}.json"

    def clean(obj):
        if isinstance(obj, dict):
            return {k: clean(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [clean(v) for v in obj]
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating, np.float64)):
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        return obj

    with open(out_path, 'w') as f:
        json.dump(clean(all_results), f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == '__main__':
    main()
