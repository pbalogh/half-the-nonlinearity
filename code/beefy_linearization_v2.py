"""
Beefy Linearization v2: Pre-tokenized WikiText-103 train (103M tokens)

Same as v1 but uses pre-tokenized numpy array for training,
so we never see the same token twice in 2000 steps.
"""

import torch
import torch.nn as nn
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import json, argparse, os, sys

sys.path.insert(0, os.path.dirname(__file__))
from scale_test_universal import ModelAdapter
from variable_capacity_poc import LinearMLP, evaluate_ppl


def fit_linear_mlp_large(model, adapter, tokens, layer_idx, n_fit=50000):
    """Fit linear MLP on a larger token set."""
    mlp = adapter.get_mlp(layer_idx)
    original_forward = mlp.forward

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
            input_ids = torch.tensor([tokens[start:end]], dtype=torch.long)
            model(input_ids)
    mlp.forward = original_forward

    X = torch.cat(acts, dim=1).squeeze(0).numpy()
    Y = torch.cat(outs, dim=1).squeeze(0).numpy()

    print(f"    Fit data: {X.shape[0]} tokens, dim={X.shape[1]}", flush=True)

    Xm, Ym = X.mean(0), Y.mean(0)
    U, S, Vt = np.linalg.svd(X - Xm, full_matrices=False)
    S_inv = S / (S**2 + 0.01)
    W = (Vt.T * S_inv) @ U.T @ (Y - Ym)
    b = Ym - Xm @ W

    Y_pred = X @ W + b
    residual = np.mean((Y - Y_pred)**2)
    total = np.mean((Y - Ym)**2)
    r2 = 1 - residual / total
    print(f"    Linear fit R²: {r2:.6f}", flush=True)

    return W, b


def evaluate_ppl_corpus(model, tokens, start=0, n_eval=20000, batch_size=512):
    """Evaluate perplexity."""
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    eval_tokens = tokens[start:start + n_eval]

    with torch.no_grad():
        for chunk_start in range(0, len(eval_tokens) - 1, batch_size):
            chunk_end = min(chunk_start + batch_size, len(eval_tokens))
            input_ids = torch.tensor(
                [eval_tokens[chunk_start:chunk_end]], dtype=torch.long)
            if input_ids.shape[1] < 2:
                continue
            outputs = model(input_ids, labels=input_ids)
            n = input_ids.shape[1] - 1
            total_loss += outputs.loss.item() * n
            total_tokens += n

    return np.exp(total_loss / total_tokens)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='gpt2-medium')
    parser.add_argument('--linear_layers', default='10,11,12,13')
    parser.add_argument('--train_tokens', default='wikitext103_train_tokens.npy')
    parser.add_argument('--fit_tokens', type=int, default=50000)
    parser.add_argument('--steps', type=int, default=2000)
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--eval_every', type=int, default=200)
    parser.add_argument('--output', default=None)
    args = parser.parse_args()

    linear_layers = set(int(x) for x in args.linear_layers.split(','))

    print(f"Loading {args.model}...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float32)
    model.eval()
    adapter = ModelAdapter(model, args.model)

    # Load pre-tokenized training data
    print(f"\nLoading pre-tokenized training data: {args.train_tokens}", flush=True)
    train_tokens = np.load(args.train_tokens).tolist()
    print(f"  Train: {len(train_tokens):,} tokens", flush=True)

    # Load eval corpora
    print("Loading eval corpora...", flush=True)
    ds_test = load_dataset("wikitext", "wikitext-103-raw-v1", split="test")
    test_text = "\n".join([x for x in ds_test["text"] if x.strip()])
    test_tokens = tokenizer.encode(test_text)
    print(f"  WikiText test: {len(test_tokens):,} tokens", flush=True)

    ds_lambada = load_dataset("lambada", split="test")
    lambada_text = "\n".join([x for x in ds_lambada["text"] if x.strip()])
    lambada_tokens = tokenizer.encode(lambada_text)
    print(f"  LAMBADA: {len(lambada_tokens):,} tokens", flush=True)

    # Baselines
    print("\n=== Baselines ===", flush=True)
    base_wiki = evaluate_ppl_corpus(model, test_tokens, start=50000, n_eval=20000)
    base_lambada = evaluate_ppl_corpus(model, lambada_tokens, start=0, n_eval=20000)
    print(f"  WikiText test PPL: {base_wiki:.2f}", flush=True)
    print(f"  LAMBADA PPL:       {base_lambada:.2f}", flush=True)

    # Fit and install linear MLPs
    print(f"\n=== Linearizing layers {sorted(linear_layers)} ===", flush=True)
    print(f"  Using {args.fit_tokens} tokens for linear fit", flush=True)

    for layer_idx in sorted(linear_layers):
        print(f"\n  Layer {layer_idx}:", flush=True)
        W, b = fit_linear_mlp_large(model, adapter, test_tokens, layer_idx,
                                     n_fit=args.fit_tokens)
        linear_mlp = LinearMLP(W, b)
        # Install linear MLP (architecture-aware)
        if adapter.family == 'gpt2':
            model.transformer.h[layer_idx].mlp = linear_mlp
        elif adapter.family == 'gpt_neox':
            model.gpt_neox.layers[layer_idx].mlp = linear_mlp
        else:
            raise ValueError(f"Unsupported family: {adapter.family}")

    # Post-linearization
    print("\n=== Post-Linearization ===", flush=True)
    lin_wiki = evaluate_ppl_corpus(model, test_tokens, start=50000, n_eval=20000)
    lin_lambada = evaluate_ppl_corpus(model, lambada_tokens, start=0, n_eval=20000)
    print(f"  WikiText PPL: {lin_wiki:.2f} ({(lin_wiki-base_wiki)/base_wiki*100:+.1f}%)", flush=True)
    print(f"  LAMBADA PPL:  {lin_lambada:.2f} ({(lin_lambada-base_lambada)/base_lambada*100:+.1f}%)", flush=True)

    # Fine-tune with large corpus
    print(f"\n=== Fine-Tuning ({args.steps} steps on {len(train_tokens):,} token corpus) ===", flush=True)
    tokens_per_step = args.batch_size
    total_train_tokens = args.steps * tokens_per_step
    corpus_coverage = total_train_tokens / len(train_tokens) * 100
    print(f"  {total_train_tokens:,} token-steps = {corpus_coverage:.2f}% of corpus (no repeats)", flush=True)

    # Freeze linear layers, unfreeze everything else
    for param in model.parameters():
        param.requires_grad = False

    trainable = 0
    if adapter.family == 'gpt2':
        layers = list(model.transformer.h)
    elif adapter.family == 'gpt_neox':
        layers = list(model.gpt_neox.layers)
    else:
        raise ValueError(f"Unsupported family: {adapter.family}")
    for i in range(adapter.n_layers):
        layer = layers[i]
        # Attention
        attn = layer.attn if hasattr(layer, 'attn') else layer.attention
        for p in attn.parameters():
            p.requires_grad = True
            trainable += p.numel()
        # MLP (only if not linearized)
        if i not in linear_layers:
            mlp = adapter.get_mlp(i)
            for p in mlp.parameters():
                p.requires_grad = True
                trainable += p.numel()
        # Layer norms
        for name, p in layer.named_parameters():
            if 'norm' in name.lower() or 'ln' in name.lower():
                p.requires_grad = True
                trainable += p.numel()
    # LM head + final norm
    if hasattr(model, 'lm_head'):
        lm_head = model.lm_head
    elif hasattr(model, 'embed_out'):
        lm_head = model.embed_out
    else:
        lm_head = None
    if lm_head is not None:
        for p in lm_head.parameters():
            p.requires_grad = True
            trainable += p.numel()
    if adapter.family == 'gpt2':
        for p in model.transformer.ln_f.parameters():
            p.requires_grad = True
            trainable += p.numel()
    elif adapter.family == 'gpt_neox':
        for p in model.gpt_neox.final_layer_norm.parameters():
            p.requires_grad = True
            trainable += p.numel()

    total = sum(p.numel() for p in model.parameters())
    print(f"  Trainable: {trainable:,} / {total:,} ({trainable/total:.1%})", flush=True)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=0.01
    )

    from torch.optim.lr_scheduler import CosineAnnealingLR
    scheduler = CosineAnnealingLR(optimizer, T_max=args.steps, eta_min=args.lr/10)

    model.train()
    eval_trajectory = []
    train_len = len(train_tokens)

    # Use sequential sampling — walk through the corpus, never repeat
    train_pos = 0

    for step in range(args.steps):
        # Sequential walk through corpus (no repeats)
        if train_pos + args.batch_size + 1 >= train_len:
            train_pos = 0  # wrap only if we exhaust the corpus
        input_ids = torch.tensor(
            [train_tokens[train_pos:train_pos + args.batch_size]], dtype=torch.long)
        train_pos += args.batch_size

        outputs = model(input_ids, labels=input_ids)
        loss = outputs.loss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], 1.0)
        optimizer.step()
        scheduler.step()

        if (step + 1) % 50 == 0:
            lr_now = scheduler.get_last_lr()[0]
            print(f"  Step {step+1}/{args.steps}: loss={loss.item():.4f} lr={lr_now:.2e}",
                  flush=True)

        if (step + 1) % args.eval_every == 0:
            model.eval()
            wiki_ppl = evaluate_ppl_corpus(model, test_tokens, start=50000, n_eval=20000)
            lamb_ppl = evaluate_ppl_corpus(model, lambada_tokens, start=0, n_eval=20000)
            print(f"  → WikiText: {wiki_ppl:.2f} ({(wiki_ppl-base_wiki)/base_wiki*100:+.1f}%) "
                  f"| LAMBADA: {lamb_ppl:.2f} ({(lamb_ppl-base_lambada)/base_lambada*100:+.1f}%)",
                  flush=True)
            eval_trajectory.append({
                'step': step + 1,
                'wiki_ppl': float(wiki_ppl),
                'lambada_ppl': float(lamb_ppl),
                'wiki_delta': float((wiki_ppl-base_wiki)/base_wiki*100),
                'lambada_delta': float((lamb_ppl-base_lambada)/base_lambada*100),
                'train_pos': train_pos,
            })
            model.train()

    # Final eval
    print("\n=== Final Evaluation ===", flush=True)
    model.eval()
    final_wiki = evaluate_ppl_corpus(model, test_tokens, start=50000, n_eval=20000)
    final_lambada = evaluate_ppl_corpus(model, lambada_tokens, start=0, n_eval=20000)

    print(f"  {'':>15} {'Baseline':>10} {'Post-Lin':>10} {'Final':>10} {'Δ%':>8}", flush=True)
    print(f"  {'WikiText':>15} {base_wiki:>10.2f} {lin_wiki:>10.2f} {final_wiki:>10.2f} "
          f"{(final_wiki-base_wiki)/base_wiki*100:>+8.1f}%", flush=True)
    print(f"  {'LAMBADA':>15} {base_lambada:>10.2f} {lin_lambada:>10.2f} {final_lambada:>10.2f} "
          f"{(final_lambada-base_lambada)/base_lambada*100:>+8.1f}%", flush=True)

    if final_wiki < base_wiki:
        print(f"\n  ★ WikiText BEATS BASELINE! ({base_wiki:.2f} → {final_wiki:.2f})", flush=True)
    if final_lambada < base_lambada:
        print(f"  ★ LAMBADA BEATS BASELINE! ({base_lambada:.2f} → {final_lambada:.2f})", flush=True)

    # Save
    results = {
        'model': args.model,
        'linear_layers': sorted(linear_layers),
        'fit_tokens': args.fit_tokens,
        'train_corpus': args.train_tokens,
        'train_corpus_size': len(train_tokens),
        'steps': args.steps,
        'tokens_seen': train_pos,
        'corpus_coverage_pct': float(train_pos / len(train_tokens) * 100),
        'lr': args.lr,
        'batch_size': args.batch_size,
        'baseline': {'wiki': float(base_wiki), 'lambada': float(base_lambada)},
        'post_linearization': {'wiki': float(lin_wiki), 'lambada': float(lin_lambada)},
        'final': {'wiki': float(final_wiki), 'lambada': float(final_lambada)},
        'wiki_delta_pct': float((final_wiki-base_wiki)/base_wiki*100),
        'lambada_delta_pct': float((final_lambada-base_lambada)/base_lambada*100),
        'trajectory': eval_trajectory,
    }

    out_path = args.output or f"beefy_lin_v2_{args.model.split('/')[-1]}.json"
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_path}", flush=True)


if __name__ == '__main__':
    main()
