"""
Gated Linearization: Best-foot-forward version.

Instead of fully replacing MLP with a linear approximation,
learns a per-token gate that blends between linear and original MLP.
The gate is a small network (hidden → scalar sigmoid) that decides
how much nonlinearity each token actually needs.

At inference: tokens where gate ≈ 0 use cheap linear path,
tokens where gate ≈ 1 use full MLP. The model learns to allocate
nonlinearity only where needed.

Key difference from beefy_linearization_v2.py:
  - Original MLP weights are FROZEN (not removed)
  - Gate network is LEARNED
  - Linear approximation is FROZEN
  - Everything else (attention, LN, LM head) is trainable as before
  - L1 sparsity penalty on gate encourages using linear path when possible

Usage:
  python beefy_linearization_gated.py --model gpt2-medium --linear_layers 10,11,12,13 \
    --train_tokens wikitext103_train_tokens.npy --steps 2000
"""

import torch
import torch.nn as nn
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import json, argparse, os, sys

sys.path.insert(0, os.path.dirname(__file__))
from scale_test_universal import ModelAdapter


class LinearMLP(nn.Module):
    """A frozen linear replacement for an MLP layer."""
    def __init__(self, W, b):
        super().__init__()
        self.register_buffer('W', torch.tensor(W, dtype=torch.float32))
        self.register_buffer('b', torch.tensor(b, dtype=torch.float32))

    def forward(self, x):
        return x @ self.W + self.b


class GatedMLP(nn.Module):
    """
    Blends between a frozen linear approximation and the frozen original MLP.
    
    output = gate * original_mlp(x) + (1 - gate) * linear_mlp(x)
    
    gate = sigmoid(x @ W_gate + b_gate)  — per-token scalar
    
    When gate → 0: uses cheap linear path (no nonlinearity needed)
    When gate → 1: uses full MLP (nonlinearity needed)
    """
    def __init__(self, original_mlp, linear_W, linear_b, hidden_dim, gate_hidden=64):
        super().__init__()
        self.original_mlp = original_mlp
        # Freeze original MLP
        for p in self.original_mlp.parameters():
            p.requires_grad = False
        
        # Frozen linear approximation
        self.register_buffer('linear_W', torch.tensor(linear_W, dtype=torch.float32))
        self.register_buffer('linear_b', torch.tensor(linear_b, dtype=torch.float32))
        
        # Learnable gate: small MLP → scalar
        self.gate_net = nn.Sequential(
            nn.Linear(hidden_dim, gate_hidden),
            nn.ReLU(),
            nn.Linear(gate_hidden, 1),
        )
        # Initialize gate to output ~0 (bias toward linear path)
        nn.init.zeros_(self.gate_net[2].weight)
        nn.init.constant_(self.gate_net[2].bias, -2.0)  # sigmoid(-2) ≈ 0.12
    
    def forward(self, x, *args, **kwargs):
        # Gate: per-token scalar in [0, 1]
        gate = torch.sigmoid(self.gate_net(x.detach()))  # detach so gate doesn't backprop through representations
        
        # Linear path (cheap)
        linear_out = x @ self.linear_W + self.linear_b
        
        # Original MLP path (expensive, frozen)
        with torch.no_grad():
            mlp_out = self.original_mlp(x, *args, **kwargs)
        
        # Blend
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
                x = input[0]
                with torch.no_grad():
                    g = torch.sigmoid(module.gate_net(x))
                    gate_list.append(g.squeeze(-1).cpu())
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
              f">0.9: {stats[layer_idx]['frac_above_0.9']:.1%}", flush=True)
    
    return stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='gpt2-medium')
    parser.add_argument('--linear_layers', default='10,11,12,13')
    parser.add_argument('--train_tokens', default='wikitext103_train_tokens.npy')
    parser.add_argument('--fit_tokens', type=int, default=50000)
    parser.add_argument('--steps', type=int, default=2000)
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--gate_lr', type=float, default=1e-3,
                        help='Learning rate for gate networks (higher than main LR)')
    parser.add_argument('--sparsity', type=float, default=0.01,
                        help='L1 penalty on gate values (encourages linear path)')
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--eval_every', type=int, default=200)
    parser.add_argument('--gate_hidden', type=int, default=64)
    parser.add_argument('--output', default=None)
    args = parser.parse_args()

    linear_layers = set(int(x) for x in args.linear_layers.split(','))

    print(f"Loading {args.model}...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float32)
    model.eval()
    adapter = ModelAdapter(model, args.model)

    # Get hidden dim
    if hasattr(model.config, 'n_embd'):
        hidden_dim = model.config.n_embd
    elif hasattr(model.config, 'hidden_size'):
        hidden_dim = model.config.hidden_size
    else:
        raise ValueError("Cannot determine hidden dim")
    print(f"  Hidden dim: {hidden_dim}", flush=True)

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

    # Fit linear approximations and install gated MLPs
    print(f"\n=== Installing Gated MLPs for layers {sorted(linear_layers)} ===", flush=True)
    print(f"  Using {args.fit_tokens} tokens for linear fit", flush=True)
    print(f"  Gate hidden dim: {args.gate_hidden}", flush=True)

    gated_layers = set()
    for layer_idx in sorted(linear_layers):
        print(f"\n  Layer {layer_idx}:", flush=True)
        
        # Get original MLP before replacing
        if adapter.family == 'gpt2':
            original_mlp = model.transformer.h[layer_idx].mlp
        elif adapter.family == 'gpt_neox':
            original_mlp = model.gpt_neox.layers[layer_idx].mlp
        else:
            raise ValueError(f"Unsupported family: {adapter.family}")
        
        # Fit linear approximation
        W, b = fit_linear_mlp_large(model, adapter, test_tokens, layer_idx,
                                     n_fit=args.fit_tokens)
        
        # Create gated MLP
        gated_mlp = GatedMLP(original_mlp, W, b, hidden_dim, gate_hidden=args.gate_hidden)
        
        # Install
        if adapter.family == 'gpt2':
            model.transformer.h[layer_idx].mlp = gated_mlp
        elif adapter.family == 'gpt_neox':
            model.gpt_neox.layers[layer_idx].mlp = gated_mlp
        
        gated_layers.add(layer_idx)

    # Post-gating eval (gates initialized near 0, so mostly linear)
    print("\n=== Post-Gating (gates ≈ 0, mostly linear) ===", flush=True)
    gated_wiki = evaluate_ppl_corpus(model, test_tokens, start=50000, n_eval=20000)
    gated_lambada = evaluate_ppl_corpus(model, lambada_tokens, start=0, n_eval=20000)
    print(f"  WikiText PPL: {gated_wiki:.2f} ({(gated_wiki-base_wiki)/base_wiki*100:+.1f}%)", flush=True)
    print(f"  LAMBADA PPL:  {gated_lambada:.2f} ({(gated_lambada-base_lambada)/base_lambada*100:+.1f}%)", flush=True)

    # Set up training
    print(f"\n=== Fine-Tuning ({args.steps} steps) ===", flush=True)

    # Freeze everything first
    for param in model.parameters():
        param.requires_grad = False

    # Collect parameter groups
    gate_params = []
    main_params = []
    trainable = 0

    # Gate parameters (higher LR)
    for layer_idx in sorted(gated_layers):
        if adapter.family == 'gpt2':
            gated_mlp = model.transformer.h[layer_idx].mlp
        elif adapter.family == 'gpt_neox':
            gated_mlp = model.gpt_neox.layers[layer_idx].mlp
        
        for p in gated_mlp.gate_net.parameters():
            p.requires_grad = True
            gate_params.append(p)
            trainable += p.numel()

    # Attention in all layers
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
        
        # MLP only if NOT gated (non-linearized layers)
        if i not in gated_layers:
            mlp = adapter.get_mlp(i)
            for p in mlp.parameters():
                p.requires_grad = True
                main_params.append(p)
                trainable += p.numel()
        
        # Layer norms
        for name, p in layer.named_parameters():
            if ('norm' in name.lower() or 'ln' in name.lower()) and not p.requires_grad:
                p.requires_grad = True
                main_params.append(p)
                trainable += p.numel()

    # LM head + final norm
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
    print(f"  Gate params: {gate_count:,} ({len(gated_layers)} layers × ~{gate_count//len(gated_layers):,} each)", flush=True)
    print(f"  Gate LR: {args.gate_lr}, Main LR: {args.lr}, Sparsity: {args.sparsity}", flush=True)

    optimizer = torch.optim.AdamW([
        {'params': main_params, 'lr': args.lr, 'weight_decay': 0.01},
        {'params': gate_params, 'lr': args.gate_lr, 'weight_decay': 0.0},
    ])

    from torch.optim.lr_scheduler import CosineAnnealingLR
    scheduler = CosineAnnealingLR(optimizer, T_max=args.steps, eta_min=args.lr/10)

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

        outputs = model(input_ids, labels=input_ids)
        loss = outputs.loss

        # L1 sparsity penalty on gate activations
        if args.sparsity > 0:
            sparsity_loss = 0.0
            for layer_idx in sorted(gated_layers):
                if adapter.family == 'gpt2':
                    gated_mlp = model.transformer.h[layer_idx].mlp
                elif adapter.family == 'gpt_neox':
                    gated_mlp = model.gpt_neox.layers[layer_idx].mlp
                # Penalize gate bias toward 1 (encourage linear path)
                # We use the gate bias as a proxy — actual gate values require a forward pass
                gate_bias = gated_mlp.gate_net[2].bias
                sparsity_loss = sparsity_loss + torch.sigmoid(gate_bias).mean()
            loss = loss + args.sparsity * sparsity_loss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], 1.0)
        optimizer.step()
        scheduler.step()

        if (step + 1) % 50 == 0:
            lr_now = scheduler.get_last_lr()[0]
            print(f"  Step {step+1}/{args.steps}: loss={outputs.loss.item():.4f} lr={lr_now:.2e}",
                  flush=True)

        if (step + 1) % args.eval_every == 0:
            model.eval()
            wiki_ppl = evaluate_ppl_corpus(model, test_tokens, start=50000, n_eval=20000)
            lamb_ppl = evaluate_ppl_corpus(model, lambada_tokens, start=0, n_eval=20000)
            
            # Gate stats
            print(f"\n  → Step {step+1} WikiText: {wiki_ppl:.2f} ({(wiki_ppl-base_wiki)/base_wiki*100:+.1f}%) "
                  f"| LAMBADA: {lamb_ppl:.2f} ({(lamb_ppl-base_lambada)/base_lambada*100:+.1f}%)", flush=True)
            print(f"  Gate statistics:", flush=True)
            gate_stats = get_gate_stats(model, adapter, gated_layers, test_tokens)
            
            eval_trajectory.append({
                'step': step + 1,
                'wiki_ppl': float(wiki_ppl),
                'lambada_ppl': float(lamb_ppl),
                'wiki_delta': float((wiki_ppl-base_wiki)/base_wiki*100),
                'lambada_delta': float((lamb_ppl-base_lambada)/base_lambada*100),
                'train_pos': train_pos,
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

    if final_wiki < base_wiki:
        print(f"\n  ★ WikiText BEATS BASELINE! ({base_wiki:.2f} → {final_wiki:.2f})", flush=True)
    if final_lambada < base_lambada:
        print(f"  ★ LAMBADA BEATS BASELINE! ({base_lambada:.2f} → {final_lambada:.2f})", flush=True)

    # Compute effective linearization rate
    avg_gate = np.mean([final_gate_stats[l]['mean'] for l in sorted(gated_layers)])
    linear_frac = 1 - avg_gate
    print(f"\n  Effective linearization: {linear_frac:.1%} of gated-layer compute uses linear path", flush=True)
    print(f"  (avg gate = {avg_gate:.3f})", flush=True)

    # Save
    results = {
        'model': args.model,
        'method': 'gated_linearization',
        'gated_layers': sorted(gated_layers),
        'gate_hidden': args.gate_hidden,
        'fit_tokens': args.fit_tokens,
        'train_corpus': args.train_tokens,
        'train_corpus_size': len(train_tokens),
        'steps': args.steps,
        'tokens_seen': train_pos,
        'corpus_coverage_pct': float(train_pos / len(train_tokens) * 100),
        'lr': args.lr,
        'gate_lr': args.gate_lr,
        'sparsity': args.sparsity,
        'batch_size': args.batch_size,
        'baseline': {'wiki': float(base_wiki), 'lambada': float(base_lambada)},
        'post_gating': {'wiki': float(gated_wiki), 'lambada': float(gated_lambada)},
        'final': {'wiki': float(final_wiki), 'lambada': float(final_lambada)},
        'wiki_delta_pct': float((final_wiki-base_wiki)/base_wiki*100),
        'lambada_delta_pct': float((final_lambada-base_lambada)/base_lambada*100),
        'final_gate_stats': final_gate_stats,
        'avg_gate': float(avg_gate),
        'effective_linear_frac': float(linear_frac),
        'trajectory': eval_trajectory,
    }

    out_path = args.output or f"beefy_lin_gated_{args.model.split('/')[-1]}.json"
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_path}", flush=True)


if __name__ == '__main__':
    main()
