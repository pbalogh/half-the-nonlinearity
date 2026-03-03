"""
Phase 2: Gate-Only Training on a Pre-Linearized Model.

Two-phase approach:
  Phase 1 (already done): Full linearization + fine-tune → model learns to
           compensate for missing nonlinearity via attention/LN adjustments.
  Phase 2 (this script): Freeze everything. Install gates. Train ONLY the
           gate networks to learn which tokens benefit from nonlinearity.

This measures the marginal value of nonlinearity per-token, on a model
that has already adapted to its absence. Much cleaner signal than joint training.

Memory: Only ~262K trainable params (vs 320M in joint). Mac mini friendly.

Usage:
  # Option A: Run phase 1 inline first, then phase 2
  python3 phase2_gate_only.py --model gpt2-medium --linear_layers 10,11,12,13 \
    --train_tokens wikitext103_train_tokens.npy \
    --phase1_steps 2000 --phase2_steps 3000

  # Option B: Skip phase 1 (if you just want to test gate training on base model)
  python3 phase2_gate_only.py --model gpt2-medium --linear_layers 10,11,12,13 \
    --train_tokens wikitext103_train_tokens.npy \
    --phase1_steps 0 --phase2_steps 3000
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
    """Frozen linear replacement for an MLP layer."""
    def __init__(self, W, b):
        super().__init__()
        self.register_buffer('W', torch.tensor(W, dtype=torch.float32))
        self.register_buffer('b', torch.tensor(b, dtype=torch.float32))

    def forward(self, x, *args, **kwargs):
        return x @ self.W + self.b


class GatedMLP(nn.Module):
    """
    Phase 2 gated MLP. Blends frozen original MLP with frozen linear approx.
    
    ONLY the gate network has gradients.
    
    output = gate * original_mlp(x) + (1 - gate) * linear_mlp(x)
    """
    def __init__(self, original_mlp, linear_W, linear_b, hidden_dim, gate_hidden=32):
        super().__init__()
        # Both paths frozen
        self.original_mlp = original_mlp
        for p in self.original_mlp.parameters():
            p.requires_grad = False

        self.register_buffer('linear_W', torch.tensor(linear_W, dtype=torch.float32))
        self.register_buffer('linear_b', torch.tensor(linear_b, dtype=torch.float32))

        # Only trainable part: small gate network
        self.gate_net = nn.Sequential(
            nn.Linear(hidden_dim, gate_hidden),
            nn.ReLU(),
            nn.Linear(gate_hidden, 1),
        )
        # Initialize strongly toward linear (gate ≈ 0)
        nn.init.zeros_(self.gate_net[2].weight)
        nn.init.constant_(self.gate_net[2].bias, -3.0)  # sigmoid(-3) ≈ 0.047

        self.last_gate = None

    def forward(self, x, *args, **kwargs):
        gate = torch.sigmoid(self.gate_net(x))  # no detach — gate gets full gradient
        self.last_gate = gate

        linear_out = x @ self.linear_W + self.linear_b

        with torch.no_grad():
            mlp_out = self.original_mlp(x, *args, **kwargs)

        return gate * mlp_out + (1 - gate) * linear_out


def fit_linear_mlp(model, adapter, tokens, layer_idx, n_fit=50000):
    """Fit linear MLP approximation."""
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
    r2 = 1 - np.mean((Y - Y_pred)**2) / np.mean((Y - Ym)**2)
    print(f"    Linear fit R²: {r2:.6f}", flush=True)
    return W, b


def evaluate_ppl(model, tokens, start=0, n_eval=20000, batch_size=512):
    """Evaluate perplexity."""
    model.eval()
    total_loss, total_tokens = 0.0, 0
    eval_tokens = tokens[start:start + n_eval]

    with torch.no_grad():
        for cs in range(0, len(eval_tokens) - 1, batch_size):
            ce = min(cs + batch_size, len(eval_tokens))
            ids = torch.tensor([eval_tokens[cs:ce]], dtype=torch.long)
            if ids.shape[1] < 2:
                continue
            out = model(ids, labels=ids)
            n = ids.shape[1] - 1
            total_loss += out.loss.item() * n
            total_tokens += n

    return np.exp(total_loss / total_tokens)


def get_gate_stats(model, adapter, gated_layers, tokens, n_tokens=5000):
    """Collect gate statistics."""
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

        def make_hook(glist):
            def hook(mod, inp, out):
                if mod.last_gate is not None:
                    glist.append(mod.last_gate.detach().squeeze(-1).cpu())
            return hook

        h = gated_mlp.register_forward_hook(make_hook(gate_vals))
        with torch.no_grad():
            for start in range(0, n_tokens, 512):
                end = min(start + 512, n_tokens)
                ids = torch.tensor([tokens[start:end]], dtype=torch.long)
                model(ids)
        h.remove()

        g = torch.cat(gate_vals, dim=-1).numpy()
        stats[layer_idx] = {
            'mean': float(np.mean(g)),
            'median': float(np.median(g)),
            'std': float(np.std(g)),
            'frac_below_0.1': float(np.mean(g < 0.1)),
            'frac_below_0.5': float(np.mean(g < 0.5)),
            'frac_above_0.9': float(np.mean(g > 0.9)),
        }
        print(f"    Layer {layer_idx}: mean={stats[layer_idx]['mean']:.3f}, "
              f"<0.1: {stats[layer_idx]['frac_below_0.1']:.1%}, "
              f"<0.5: {stats[layer_idx]['frac_below_0.5']:.1%}, "
              f">0.9: {stats[layer_idx]['frac_above_0.9']:.1%}", flush=True)

    return stats


def collect_gate_loss(model, adapter, gated_layers):
    """L1 penalty on actual gate activations."""
    total, n = 0.0, 0
    for li in sorted(gated_layers):
        if adapter.family == 'gpt2':
            gm = model.transformer.h[li].mlp
        elif adapter.family == 'gpt_neox':
            gm = model.gpt_neox.layers[li].mlp
        else:
            continue
        if gm.last_gate is not None:
            total = total + gm.last_gate.mean()
            n += 1
    return total / n if n > 0 else torch.tensor(0.0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='gpt2-medium')
    parser.add_argument('--linear_layers', default='10,11,12,13')
    parser.add_argument('--train_tokens', default='wikitext103_train_tokens.npy')
    parser.add_argument('--fit_tokens', type=int, default=50000)
    # Phase 1
    parser.add_argument('--phase1_steps', type=int, default=2000,
                        help='Fine-tune steps with full linearization (0 to skip)')
    parser.add_argument('--phase1_lr', type=float, default=2e-5)
    # Phase 2
    parser.add_argument('--phase2_steps', type=int, default=3000,
                        help='Gate-only training steps')
    parser.add_argument('--phase2_lr', type=float, default=1e-3)
    parser.add_argument('--sparsity', type=float, default=0.1,
                        help='L1 penalty on gate activations')
    parser.add_argument('--warmup_frac', type=float, default=0.1,
                        help='Fraction of phase 2 with no sparsity')
    # Shared
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--eval_every', type=int, default=200)
    parser.add_argument('--gate_hidden', type=int, default=32)
    parser.add_argument('--output', default=None)
    args = parser.parse_args()

    linear_layers = set(int(x) for x in args.linear_layers.split(','))

    print(f"=== Two-Phase Gated Linearization ===", flush=True)
    print(f"  Phase 1: {args.phase1_steps} steps full linearization + fine-tune", flush=True)
    print(f"  Phase 2: {args.phase2_steps} steps gate-only training", flush=True)
    print(f"  Sparsity: {args.sparsity}, warmup: {args.warmup_frac:.0%}\n", flush=True)

    print(f"Loading {args.model}...", flush=True)
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

    # Load data
    print(f"Loading training data: {args.train_tokens}", flush=True)
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
    base_wiki = evaluate_ppl(model, test_tokens, start=50000)
    base_lambada = evaluate_ppl(model, lambada_tokens, start=0)
    print(f"  WikiText: {base_wiki:.2f}", flush=True)
    print(f"  LAMBADA:  {base_lambada:.2f}", flush=True)

    # Fit linear approximations
    print(f"\n=== Fitting linear MLPs for layers {sorted(linear_layers)} ===", flush=True)
    linear_fits = {}
    for li in sorted(linear_layers):
        print(f"  Layer {li}:", flush=True)
        linear_fits[li] = fit_linear_mlp(model, adapter, test_tokens, li, args.fit_tokens)

    # =========================================================
    # PHASE 1: Full linearization + fine-tune
    # =========================================================
    if args.phase1_steps > 0:
        print(f"\n{'='*60}", flush=True)
        print(f"  PHASE 1: Full linearization + fine-tune ({args.phase1_steps} steps)", flush=True)
        print(f"{'='*60}", flush=True)

        # Store original MLPs before replacing
        original_mlps = {}
        for li in sorted(linear_layers):
            if adapter.family == 'gpt2':
                original_mlps[li] = model.transformer.h[li].mlp
            elif adapter.family == 'gpt_neox':
                original_mlps[li] = model.gpt_neox.layers[li].mlp

        # Install linear MLPs
        for li in sorted(linear_layers):
            W, b = linear_fits[li]
            lmlp = LinearMLP(W, b)
            if adapter.family == 'gpt2':
                model.transformer.h[li].mlp = lmlp
            elif adapter.family == 'gpt_neox':
                model.gpt_neox.layers[li].mlp = lmlp

        post_lin_wiki = evaluate_ppl(model, test_tokens, start=50000)
        print(f"  Post-linearization WikiText: {post_lin_wiki:.2f} "
              f"({(post_lin_wiki-base_wiki)/base_wiki*100:+.1f}%)", flush=True)

        # Unfreeze everything except linear MLPs
        for p in model.parameters():
            p.requires_grad = False

        trainable = 0
        if adapter.family == 'gpt2':
            layers = list(model.transformer.h)
        elif adapter.family == 'gpt_neox':
            layers = list(model.gpt_neox.layers)

        for i in range(adapter.n_layers):
            layer = layers[i]
            attn = layer.attn if hasattr(layer, 'attn') else layer.attention
            for p in attn.parameters():
                p.requires_grad = True
                trainable += p.numel()
            if i not in linear_layers:
                for p in adapter.get_mlp(i).parameters():
                    p.requires_grad = True
                    trainable += p.numel()
            for name, p in layer.named_parameters():
                if ('norm' in name.lower() or 'ln' in name.lower()) and not p.requires_grad:
                    p.requires_grad = True
                    trainable += p.numel()

        if hasattr(model, 'lm_head'):
            for p in model.lm_head.parameters():
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

        print(f"  Phase 1 trainable: {trainable:,}", flush=True)

        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=args.phase1_lr, weight_decay=0.01)
        from torch.optim.lr_scheduler import CosineAnnealingLR
        scheduler = CosineAnnealingLR(optimizer, T_max=args.phase1_steps, eta_min=args.phase1_lr/10)

        model.train()
        train_pos = 0
        phase1_trajectory = []

        for step in range(args.phase1_steps):
            if train_pos + args.batch_size + 1 >= len(train_tokens):
                train_pos = 0
            ids = torch.tensor([train_tokens[train_pos:train_pos+args.batch_size]], dtype=torch.long)
            train_pos += args.batch_size

            out = model(ids, labels=ids)
            optimizer.zero_grad()
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.0)
            optimizer.step()
            scheduler.step()

            if (step+1) % 100 == 0:
                print(f"  P1 Step {step+1}/{args.phase1_steps}: loss={out.loss.item():.4f}", flush=True)

            if (step+1) % args.eval_every == 0:
                model.eval()
                w = evaluate_ppl(model, test_tokens, start=50000)
                l = evaluate_ppl(model, lambada_tokens, start=0)
                print(f"  → P1 Step {step+1} Wiki: {w:.2f} ({(w-base_wiki)/base_wiki*100:+.1f}%) "
                      f"| LAMBADA: {l:.2f} ({(l-base_lambada)/base_lambada*100:+.1f}%)", flush=True)
                phase1_trajectory.append({
                    'step': step+1, 'wiki_ppl': float(w), 'lambada_ppl': float(l),
                    'wiki_delta': float((w-base_wiki)/base_wiki*100),
                    'lambada_delta': float((l-base_lambada)/base_lambada*100),
                })
                model.train()

        # Phase 1 final eval
        model.eval()
        p1_wiki = evaluate_ppl(model, test_tokens, start=50000)
        p1_lambada = evaluate_ppl(model, lambada_tokens, start=0)
        print(f"\n  Phase 1 final: Wiki {p1_wiki:.2f} ({(p1_wiki-base_wiki)/base_wiki*100:+.1f}%) "
              f"| LAMBADA {p1_lambada:.2f} ({(p1_lambada-base_lambada)/base_lambada*100:+.1f}%)", flush=True)

        # Restore original MLPs (now we have a compensated model)
        for li in sorted(linear_layers):
            if adapter.family == 'gpt2':
                model.transformer.h[li].mlp = original_mlps[li]
            elif adapter.family == 'gpt_neox':
                model.gpt_neox.layers[li].mlp = original_mlps[li]

        # Clean up phase 1 optimizer to free memory
        del optimizer, scheduler
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        import gc; gc.collect()

    else:
        print("\n  Skipping Phase 1 (phase1_steps=0)", flush=True)
        p1_wiki = base_wiki
        p1_lambada = base_lambada
        phase1_trajectory = []
        original_mlps = {}
        for li in sorted(linear_layers):
            if adapter.family == 'gpt2':
                original_mlps[li] = model.transformer.h[li].mlp
            elif adapter.family == 'gpt_neox':
                original_mlps[li] = model.gpt_neox.layers[li].mlp

    # =========================================================
    # PHASE 2: Gate-only training
    # =========================================================
    print(f"\n{'='*60}", flush=True)
    print(f"  PHASE 2: Gate-only training ({args.phase2_steps} steps)", flush=True)
    print(f"{'='*60}", flush=True)

    # Freeze EVERYTHING
    for p in model.parameters():
        p.requires_grad = False

    # Install gated MLPs (original MLP + linear approx, both frozen; only gate trains)
    gated_layers = set()
    for li in sorted(linear_layers):
        if adapter.family == 'gpt2':
            orig_mlp = model.transformer.h[li].mlp
        elif adapter.family == 'gpt_neox':
            orig_mlp = model.gpt_neox.layers[li].mlp

        W, b = linear_fits[li]
        gmlp = GatedMLP(orig_mlp, W, b, hidden_dim, gate_hidden=args.gate_hidden)

        if adapter.family == 'gpt2':
            model.transformer.h[li].mlp = gmlp
        elif adapter.family == 'gpt_neox':
            model.gpt_neox.layers[li].mlp = gmlp

        gated_layers.add(li)

    # Only gate params are trainable
    gate_params = []
    for li in sorted(gated_layers):
        if adapter.family == 'gpt2':
            gm = model.transformer.h[li].mlp
        elif adapter.family == 'gpt_neox':
            gm = model.gpt_neox.layers[li].mlp
        for p in gm.gate_net.parameters():
            p.requires_grad = True
            gate_params.append(p)

    gate_count = sum(p.numel() for p in gate_params)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Trainable (gate only): {gate_count:,} / {total_params:,} ({gate_count/total_params:.4%})", flush=True)

    # Pre-gate eval (gates ≈ 0, so mostly linear — should match phase 1 linearized)
    model.eval()
    pre_gate_wiki = evaluate_ppl(model, test_tokens, start=50000)
    pre_gate_lambada = evaluate_ppl(model, lambada_tokens, start=0)
    print(f"  Pre-gate (mostly linear): Wiki {pre_gate_wiki:.2f} "
          f"({(pre_gate_wiki-base_wiki)/base_wiki*100:+.1f}%) "
          f"| LAMBADA {pre_gate_lambada:.2f} "
          f"({(pre_gate_lambada-base_lambada)/base_lambada*100:+.1f}%)", flush=True)

    print(f"\n  Initial gate stats:", flush=True)
    get_gate_stats(model, adapter, gated_layers, test_tokens)

    optimizer = torch.optim.Adam(gate_params, lr=args.phase2_lr)
    from torch.optim.lr_scheduler import CosineAnnealingLR
    scheduler = CosineAnnealingLR(optimizer, T_max=args.phase2_steps, eta_min=args.phase2_lr/10)

    warmup_steps = int(args.phase2_steps * args.warmup_frac)
    print(f"\n  Sparsity: {args.sparsity} (after {warmup_steps} warmup steps)", flush=True)

    model.train()
    train_pos = 0
    phase2_trajectory = []

    for step in range(args.phase2_steps):
        if train_pos + args.batch_size + 1 >= len(train_tokens):
            train_pos = 0
        ids = torch.tensor([train_tokens[train_pos:train_pos+args.batch_size]], dtype=torch.long)
        train_pos += args.batch_size

        out = model(ids, labels=ids)
        loss = out.loss

        # Sparsity after warmup
        sparsity_w = 0.0 if step < warmup_steps else args.sparsity
        if sparsity_w > 0:
            gate_loss = collect_gate_loss(model, adapter, gated_layers)
            loss = loss + sparsity_w * gate_loss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(gate_params, 1.0)
        optimizer.step()
        scheduler.step()

        if (step+1) % 50 == 0:
            print(f"  P2 Step {step+1}/{args.phase2_steps}: loss={out.loss.item():.4f} "
                  f"sparsity_w={sparsity_w:.3f}", flush=True)

        if (step+1) % args.eval_every == 0:
            model.eval()
            w = evaluate_ppl(model, test_tokens, start=50000)
            l = evaluate_ppl(model, lambada_tokens, start=0)
            print(f"\n  → P2 Step {step+1} Wiki: {w:.2f} ({(w-base_wiki)/base_wiki*100:+.1f}%) "
                  f"| LAMBADA: {l:.2f} ({(l-base_lambada)/base_lambada*100:+.1f}%)", flush=True)
            print(f"  Gate stats:", flush=True)
            gs = get_gate_stats(model, adapter, gated_layers, test_tokens)
            phase2_trajectory.append({
                'step': step+1, 'wiki_ppl': float(w), 'lambada_ppl': float(l),
                'wiki_delta': float((w-base_wiki)/base_wiki*100),
                'lambada_delta': float((l-base_lambada)/base_lambada*100),
                'sparsity_weight': float(sparsity_w),
                'gate_stats': gs,
            })
            model.train()

    # Final
    print(f"\n{'='*60}", flush=True)
    print(f"  FINAL EVALUATION", flush=True)
    print(f"{'='*60}", flush=True)
    model.eval()
    final_wiki = evaluate_ppl(model, test_tokens, start=50000)
    final_lambada = evaluate_ppl(model, lambada_tokens, start=0)

    print(f"\n  Final gate stats:", flush=True)
    final_gate_stats = get_gate_stats(model, adapter, gated_layers, test_tokens)

    avg_gate = np.mean([final_gate_stats[l]['mean'] for l in sorted(gated_layers)])
    linear_frac = 1 - avg_gate

    print(f"\n  {'':>20} {'Baseline':>10} {'Phase1':>10} {'Pre-Gate':>10} {'Final':>10} {'Δ%':>8}", flush=True)
    print(f"  {'WikiText':>20} {base_wiki:>10.2f} {p1_wiki:>10.2f} {pre_gate_wiki:>10.2f} "
          f"{final_wiki:>10.2f} {(final_wiki-base_wiki)/base_wiki*100:>+8.1f}%", flush=True)
    print(f"  {'LAMBADA':>20} {base_lambada:>10.2f} {p1_lambada:>10.2f} {pre_gate_lambada:>10.2f} "
          f"{final_lambada:>10.2f} {(final_lambada-base_lambada)/base_lambada*100:>+8.1f}%", flush=True)

    print(f"\n  Effective linearization: {linear_frac:.1%} (avg gate = {avg_gate:.3f})", flush=True)

    if final_wiki < p1_wiki:
        print(f"  ★ Gates IMPROVE over full linearization! ({p1_wiki:.2f} → {final_wiki:.2f})", flush=True)
    if final_wiki < base_wiki:
        print(f"  ★ BEATS original baseline! ({base_wiki:.2f} → {final_wiki:.2f})", flush=True)

    # Save
    results = {
        'model': args.model,
        'method': 'two_phase_gated',
        'gated_layers': sorted(gated_layers),
        'gate_hidden': args.gate_hidden,
        'phase1_steps': args.phase1_steps,
        'phase1_lr': args.phase1_lr,
        'phase2_steps': args.phase2_steps,
        'phase2_lr': args.phase2_lr,
        'sparsity': args.sparsity,
        'warmup_frac': args.warmup_frac,
        'batch_size': args.batch_size,
        'baseline': {'wiki': float(base_wiki), 'lambada': float(base_lambada)},
        'phase1_final': {'wiki': float(p1_wiki), 'lambada': float(p1_lambada)},
        'pre_gate': {'wiki': float(pre_gate_wiki), 'lambada': float(pre_gate_lambada)},
        'final': {'wiki': float(final_wiki), 'lambada': float(final_lambada)},
        'wiki_delta_pct': float((final_wiki-base_wiki)/base_wiki*100),
        'lambada_delta_pct': float((final_lambada-base_lambada)/base_lambada*100),
        'final_gate_stats': {str(k): v for k, v in final_gate_stats.items()},
        'avg_gate': float(avg_gate),
        'effective_linear_frac': float(linear_frac),
        'phase1_trajectory': phase1_trajectory,
        'phase2_trajectory': phase2_trajectory,
    }

    out_path = args.output or f"phase2_gated_{args.model.split('/')[-1]}.json"
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_path}", flush=True)


if __name__ == '__main__':
    main()
