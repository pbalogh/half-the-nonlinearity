"""
Gated Linearization v3: Fixed sparsity + annealing.

Key fixes vs v1/v2:
  1. Sparsity penalizes ACTUAL gate activations per-token (not bias proxy)
  2. Annealed sparsity: starts at 0 for warmup, ramps to target over schedule
  3. Option to anneal gate temperature for sharper binary decisions
  4. Smaller default gate network (hidden=32) to reduce memory
  5. Gate values stored during forward pass for proper L1 penalty

The idea: let the model first learn good attention/LN compensation (warmup),
THEN progressively increase pressure to use the linear path.

Usage:
  python3 beefy_linearization_gated_v3.py --model gpt2-medium --linear_layers 10,11,12,13 \
    --train_tokens wikitext103_train_tokens.npy --steps 5000 \
    --sparsity 0.1 --warmup_frac 0.2 --batch_size 128
"""

import torch
import torch.nn as nn
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import json, argparse, os, sys

sys.path.insert(0, os.path.dirname(__file__))
from scale_test_universal import ModelAdapter


class GatedMLP(nn.Module):
    """
    Blends between a frozen linear approximation and the frozen original MLP.

    output = gate * original_mlp(x) + (1 - gate) * linear_mlp(x)

    gate = sigmoid(gate_net(x) / temperature)

    Key change from v1: stores gate values for external L1 penalty.
    """
    def __init__(self, original_mlp, linear_W, linear_b, hidden_dim, gate_hidden=32):
        super().__init__()
        self.original_mlp = original_mlp
        for p in self.original_mlp.parameters():
            p.requires_grad = False

        self.register_buffer('linear_W', torch.tensor(linear_W, dtype=torch.float32))
        self.register_buffer('linear_b', torch.tensor(linear_b, dtype=torch.float32))

        # Smaller gate network
        self.gate_net = nn.Sequential(
            nn.Linear(hidden_dim, gate_hidden),
            nn.ReLU(),
            nn.Linear(gate_hidden, 1),
        )
        # Initialize biased toward linear (gate ≈ 0)
        nn.init.zeros_(self.gate_net[2].weight)
        nn.init.constant_(self.gate_net[2].bias, -3.0)  # sigmoid(-3) ≈ 0.047

        # Temperature for annealing (lower = sharper decisions)
        self.temperature = 1.0

        # Store last gate values for sparsity penalty
        self.last_gate = None

    def forward(self, x, *args, **kwargs):
        logits = self.gate_net(x.detach())
        gate = torch.sigmoid(logits / self.temperature)

        # Store for sparsity loss (keep in graph for gradient flow)
        self.last_gate = gate

        # Linear path (cheap)
        linear_out = x @ self.linear_W + self.linear_b

        # Original MLP path (expensive, frozen)
        with torch.no_grad():
            mlp_out = self.original_mlp(x, *args, **kwargs)

        return gate * mlp_out + (1 - gate) * linear_out


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


def get_gate_stats(model, adapter, gated_layers, tokens, n_tokens=5000):
    """Collect gate activation statistics."""
    model.eval()
    stats = {}

    for layer_idx in sorted(gated_layers):
        if adapter.family == 'gpt2':
            gated_mlp = model.transformer.h[layer_idx].mlp
        elif adapter.family == 'gpt_neox':
            gated_mlp = model.gpt_neox.layers[layer_idx].mlp
        else:
            continue

        gate_vals = []

        def make_hook(gate_list):
            def hook(module, input, output):
                if module.last_gate is not None:
                    gate_list.append(module.last_gate.detach().squeeze(-1).cpu())
            return hook

        h = gated_mlp.register_forward_hook(make_hook(gate_vals))

        with torch.no_grad():
            for start in range(0, n_tokens, 512):
                end = min(start + 512, n_tokens)
                input_ids = torch.tensor([tokens[start:end]], dtype=torch.long)
                model(input_ids)

        h.remove()

        all_gates = torch.cat(gate_vals, dim=-1).numpy()
        stats[layer_idx] = {
            'mean': float(np.mean(all_gates)),
            'median': float(np.median(all_gates)),
            'std': float(np.std(all_gates)),
            'frac_below_0.1': float(np.mean(all_gates < 0.1)),
            'frac_below_0.5': float(np.mean(all_gates < 0.5)),
            'frac_above_0.9': float(np.mean(all_gates > 0.9)),
        }
        print(f"    Layer {layer_idx}: mean={stats[layer_idx]['mean']:.3f}, "
              f"<0.1: {stats[layer_idx]['frac_below_0.1']:.1%}, "
              f"<0.5: {stats[layer_idx]['frac_below_0.5']:.1%}, "
              f">0.9: {stats[layer_idx]['frac_above_0.9']:.1%}", flush=True)

    return stats


def collect_gate_sparsity_loss(model, adapter, gated_layers):
    """Collect ACTUAL gate activation values for L1 penalty."""
    total_gate = 0.0
    n_gates = 0
    for layer_idx in sorted(gated_layers):
        if adapter.family == 'gpt2':
            gated_mlp = model.transformer.h[layer_idx].mlp
        elif adapter.family == 'gpt_neox':
            gated_mlp = model.gpt_neox.layers[layer_idx].mlp
        else:
            continue

        if gated_mlp.last_gate is not None:
            total_gate = total_gate + gated_mlp.last_gate.mean()
            n_gates += 1

    if n_gates == 0:
        return torch.tensor(0.0)
    return total_gate / n_gates  # average gate activation across layers


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='gpt2-medium')
    parser.add_argument('--linear_layers', default='10,11,12,13')
    parser.add_argument('--train_tokens', default='wikitext103_train_tokens.npy')
    parser.add_argument('--fit_tokens', type=int, default=50000)
    parser.add_argument('--steps', type=int, default=5000)
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--gate_lr', type=float, default=1e-3)
    parser.add_argument('--sparsity', type=float, default=0.1,
                        help='Peak L1 penalty on actual gate activations')
    parser.add_argument('--warmup_frac', type=float, default=0.2,
                        help='Fraction of training for warmup (sparsity=0)')
    parser.add_argument('--anneal_frac', type=float, default=0.3,
                        help='Fraction of training to ramp sparsity from 0 to target')
    parser.add_argument('--temp_anneal', action='store_true',
                        help='Anneal gate temperature from 1.0 to 0.1 in last 20%')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--eval_every', type=int, default=200)
    parser.add_argument('--gate_hidden', type=int, default=32)
    parser.add_argument('--output', default=None)
    args = parser.parse_args()

    linear_layers = set(int(x) for x in args.linear_layers.split(','))

    print(f"=== Gated Linearization v3: Actual-activation sparsity + annealing ===", flush=True)
    print(f"  Sparsity: {args.sparsity} (peak), warmup: {args.warmup_frac:.0%}, "
          f"anneal: {args.anneal_frac:.0%}, temp_anneal: {args.temp_anneal}", flush=True)

    print(f"\nLoading {args.model}...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float32)
    model.eval()
    adapter = ModelAdapter(model, args.model)

    if hasattr(model.config, 'n_embd'):
        hidden_dim = model.config.n_embd
    elif hasattr(model.config, 'hidden_size'):
        hidden_dim = model.config.hidden_size
    else:
        raise ValueError("Cannot determine hidden dim")
    print(f"  Hidden dim: {hidden_dim}", flush=True)

    # Load data
    print(f"\nLoading pre-tokenized training data: {args.train_tokens}", flush=True)
    train_tokens = np.load(args.train_tokens).tolist()
    print(f"  Train: {len(train_tokens):,} tokens", flush=True)

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

    # Install gated MLPs
    print(f"\n=== Installing Gated MLPs for layers {sorted(linear_layers)} ===", flush=True)
    print(f"  Gate hidden dim: {args.gate_hidden}", flush=True)

    gated_layers = set()
    for layer_idx in sorted(linear_layers):
        print(f"\n  Layer {layer_idx}:", flush=True)

        if adapter.family == 'gpt2':
            original_mlp = model.transformer.h[layer_idx].mlp
        elif adapter.family == 'gpt_neox':
            original_mlp = model.gpt_neox.layers[layer_idx].mlp
        else:
            raise ValueError(f"Unsupported family: {adapter.family}")

        W, b = fit_linear_mlp_large(model, adapter, test_tokens, layer_idx,
                                     n_fit=args.fit_tokens)

        gated_mlp = GatedMLP(original_mlp, W, b, hidden_dim, gate_hidden=args.gate_hidden)

        if adapter.family == 'gpt2':
            model.transformer.h[layer_idx].mlp = gated_mlp
        elif adapter.family == 'gpt_neox':
            model.gpt_neox.layers[layer_idx].mlp = gated_mlp

        gated_layers.add(layer_idx)

    # Post-gating eval
    print("\n=== Post-Gating (gates ≈ 0, mostly linear) ===", flush=True)
    gated_wiki = evaluate_ppl_corpus(model, test_tokens, start=50000, n_eval=20000)
    gated_lambada = evaluate_ppl_corpus(model, lambada_tokens, start=0, n_eval=20000)
    print(f"  WikiText PPL: {gated_wiki:.2f} ({(gated_wiki-base_wiki)/base_wiki*100:+.1f}%)", flush=True)
    print(f"  LAMBADA PPL:  {gated_lambada:.2f} ({(gated_lambada-base_lambada)/base_lambada*100:+.1f}%)", flush=True)

    # Setup training
    print(f"\n=== Fine-Tuning ({args.steps} steps) ===", flush=True)

    for param in model.parameters():
        param.requires_grad = False

    gate_params = []
    main_params = []
    trainable = 0

    for layer_idx in sorted(gated_layers):
        if adapter.family == 'gpt2':
            gated_mlp = model.transformer.h[layer_idx].mlp
        elif adapter.family == 'gpt_neox':
            gated_mlp = model.gpt_neox.layers[layer_idx].mlp

        for p in gated_mlp.gate_net.parameters():
            p.requires_grad = True
            gate_params.append(p)
            trainable += p.numel()

    if adapter.family == 'gpt2':
        layers = list(model.transformer.h)
    elif adapter.family == 'gpt_neox':
        layers = list(model.gpt_neox.layers)

    for i in range(adapter.n_layers):
        layer = layers[i]
        attn = layer.attn if hasattr(layer, 'attn') else layer.attention
        for p in attn.parameters():
            p.requires_grad = True
            main_params.append(p)
            trainable += p.numel()

        if i not in gated_layers:
            mlp = adapter.get_mlp(i)
            for p in mlp.parameters():
                p.requires_grad = True
                main_params.append(p)
                trainable += p.numel()

        for name, p in layer.named_parameters():
            if ('norm' in name.lower() or 'ln' in name.lower()) and not p.requires_grad:
                p.requires_grad = True
                main_params.append(p)
                trainable += p.numel()

    if hasattr(model, 'lm_head'):
        for p in model.lm_head.parameters():
            p.requires_grad = True
            main_params.append(p)
            trainable += p.numel()
    if adapter.family == 'gpt2':
        for p in model.transformer.ln_f.parameters():
            p.requires_grad = True
            main_params.append(p)
            trainable += p.numel()
    elif adapter.family == 'gpt_neox':
        for p in model.gpt_neox.final_layer_norm.parameters():
            p.requires_grad = True
            main_params.append(p)
            trainable += p.numel()

    total = sum(p.numel() for p in model.parameters())
    gate_count = sum(p.numel() for p in gate_params)
    print(f"  Trainable: {trainable:,} / {total:,} ({trainable/total:.1%})", flush=True)
    print(f"  Gate params: {gate_count:,}", flush=True)

    optimizer = torch.optim.AdamW([
        {'params': main_params, 'lr': args.lr, 'weight_decay': 0.01},
        {'params': gate_params, 'lr': args.gate_lr, 'weight_decay': 0.0},
    ])

    from torch.optim.lr_scheduler import CosineAnnealingLR
    scheduler = CosineAnnealingLR(optimizer, T_max=args.steps, eta_min=args.lr/10)

    # Sparsity schedule
    warmup_steps = int(args.steps * args.warmup_frac)
    anneal_steps = int(args.steps * args.anneal_frac)
    anneal_end = warmup_steps + anneal_steps
    print(f"  Sparsity schedule: 0 for steps 0-{warmup_steps}, "
          f"ramp to {args.sparsity} by step {anneal_end}, "
          f"hold until {args.steps}", flush=True)

    if args.temp_anneal:
        temp_start_step = int(args.steps * 0.8)
        print(f"  Temperature anneal: 1.0 → 0.1 from step {temp_start_step} to {args.steps}", flush=True)

    model.train()
    eval_trajectory = []
    train_len = len(train_tokens)
    train_pos = 0

    for step in range(args.steps):
        if train_pos + args.batch_size + 1 >= train_len:
            train_pos = 0
        input_ids = torch.tensor(
            [train_tokens[train_pos:train_pos + args.batch_size]], dtype=torch.long)
        train_pos += args.batch_size

        # Compute current sparsity weight
        if step < warmup_steps:
            sparsity_w = 0.0
        elif step < anneal_end:
            sparsity_w = args.sparsity * (step - warmup_steps) / max(anneal_steps, 1)
        else:
            sparsity_w = args.sparsity

        # Temperature annealing
        if args.temp_anneal and step >= temp_start_step:
            progress = (step - temp_start_step) / (args.steps - temp_start_step)
            temp = 1.0 - 0.9 * progress  # 1.0 → 0.1
            for layer_idx in gated_layers:
                if adapter.family == 'gpt2':
                    model.transformer.h[layer_idx].mlp.temperature = temp
                elif adapter.family == 'gpt_neox':
                    model.gpt_neox.layers[layer_idx].mlp.temperature = temp

        # Forward pass
        outputs = model(input_ids, labels=input_ids)
        loss = outputs.loss

        # L1 penalty on ACTUAL gate activations
        if sparsity_w > 0:
            gate_loss = collect_gate_sparsity_loss(model, adapter, gated_layers)
            loss = loss + sparsity_w * gate_loss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], 1.0)
        optimizer.step()
        scheduler.step()

        if (step + 1) % 50 == 0:
            lr_now = scheduler.get_last_lr()[0]
            print(f"  Step {step+1}/{args.steps}: loss={outputs.loss.item():.4f} "
                  f"sparsity_w={sparsity_w:.4f} lr={lr_now:.2e}", flush=True)

        if (step + 1) % args.eval_every == 0:
            model.eval()
            wiki_ppl = evaluate_ppl_corpus(model, test_tokens, start=50000, n_eval=20000)
            lamb_ppl = evaluate_ppl_corpus(model, lambada_tokens, start=0, n_eval=20000)

            print(f"\n  → Step {step+1} WikiText: {wiki_ppl:.2f} ({(wiki_ppl-base_wiki)/base_wiki*100:+.1f}%) "
                  f"| LAMBADA: {lamb_ppl:.2f} ({(lamb_ppl-base_lambada)/base_lambada*100:+.1f}%)", flush=True)
            print(f"  Sparsity weight: {sparsity_w:.4f}", flush=True)
            print(f"  Gate statistics:", flush=True)
            gate_stats = get_gate_stats(model, adapter, gated_layers, test_tokens)

            eval_trajectory.append({
                'step': step + 1,
                'wiki_ppl': float(wiki_ppl),
                'lambada_ppl': float(lamb_ppl),
                'wiki_delta': float((wiki_ppl-base_wiki)/base_wiki*100),
                'lambada_delta': float((lamb_ppl-base_lambada)/base_lambada*100),
                'sparsity_weight': float(sparsity_w),
                'gate_stats': gate_stats,
            })
            model.train()

    # Final eval
    print("\n=== Final Evaluation ===", flush=True)
    model.eval()
    final_wiki = evaluate_ppl_corpus(model, test_tokens, start=50000, n_eval=20000)
    final_lambada = evaluate_ppl_corpus(model, lambada_tokens, start=0, n_eval=20000)

    print(f"\n  Gate statistics (final):", flush=True)
    final_gate_stats = get_gate_stats(model, adapter, gated_layers, test_tokens)

    print(f"\n  {'':>15} {'Baseline':>10} {'Post-Gate':>10} {'Final':>10} {'Δ%':>8}", flush=True)
    print(f"  {'WikiText':>15} {base_wiki:>10.2f} {gated_wiki:>10.2f} {final_wiki:>10.2f} "
          f"{(final_wiki-base_wiki)/base_wiki*100:>+8.1f}%", flush=True)
    print(f"  {'LAMBADA':>15} {base_lambada:>10.2f} {gated_lambada:>10.2f} {final_lambada:>10.2f} "
          f"{(final_lambada-base_lambada)/base_lambada*100:>+8.1f}%", flush=True)

    avg_gate = np.mean([final_gate_stats[l]['mean'] for l in sorted(gated_layers)])
    linear_frac = 1 - avg_gate
    print(f"\n  Effective linearization: {linear_frac:.1%} of gated-layer compute uses linear path", flush=True)
    print(f"  (avg gate = {avg_gate:.3f})", flush=True)

    if final_wiki < base_wiki:
        print(f"\n  ★ WikiText BEATS BASELINE! ({base_wiki:.2f} → {final_wiki:.2f})", flush=True)
    if final_lambada < base_lambada:
        print(f"  ★ LAMBADA BEATS BASELINE! ({base_lambada:.2f} → {final_lambada:.2f})", flush=True)

    # Save
    results = {
        'model': args.model,
        'method': 'gated_linearization_v3',
        'gated_layers': sorted(gated_layers),
        'gate_hidden': args.gate_hidden,
        'fit_tokens': args.fit_tokens,
        'train_corpus': args.train_tokens,
        'train_corpus_size': len(train_tokens),
        'steps': args.steps,
        'tokens_seen': train_pos,
        'lr': args.lr,
        'gate_lr': args.gate_lr,
        'sparsity': args.sparsity,
        'warmup_frac': args.warmup_frac,
        'anneal_frac': args.anneal_frac,
        'temp_anneal': args.temp_anneal,
        'batch_size': args.batch_size,
        'baseline': {'wiki': float(base_wiki), 'lambada': float(base_lambada)},
        'post_gating': {'wiki': float(gated_wiki), 'lambada': float(gated_lambada)},
        'final': {'wiki': float(final_wiki), 'lambada': float(final_lambada)},
        'wiki_delta_pct': float((final_wiki-base_wiki)/base_wiki*100),
        'lambada_delta_pct': float((final_lambada-base_lambada)/base_lambada*100),
        'final_gate_stats': {str(k): v for k, v in final_gate_stats.items()},
        'avg_gate': float(avg_gate),
        'effective_linear_frac': float(linear_frac),
        'trajectory': eval_trajectory,
    }

    out_path = args.output or f"beefy_lin_gated_v3_{args.model.split('/')[-1]}.json"
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_path}", flush=True)


if __name__ == '__main__':
    main()
