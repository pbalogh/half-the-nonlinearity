"""
Compound gating: fit optimal gate size per layer, then run ALL gates
simultaneously to measure compound perplexity improvement.

Strategy:
1. For each layer 0-11, test gate sizes [0 (no gate), linear, b=1, b=3, b=6]
2. Pick best per layer
3. Run all 12 layers with their optimal gates simultaneously
4. Measure compound perplexity
"""

import torch
import torch.nn as nn
import numpy as np
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from datasets import load_dataset
import json, time

device = "cpu"


def fit_linear_approx(model, layer_idx, n_tokens=10000):
    ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="train[:500]")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    mlp = model.transformer.h[layer_idx].mlp
    inputs_list, outputs_list = [], []
    collected = 0
    for ex in ds:
        text = ex["text"]
        if not text.strip(): continue
        toks = tokenizer.encode(text, return_tensors="pt").to(device)
        if toks.shape[1] < 2: continue
        toks = toks[:, :512]
        with torch.no_grad():
            hidden = model.transformer.wte(toks) + model.transformer.wpe(
                torch.arange(toks.shape[1], device=device))
            for i in range(layer_idx):
                hidden = model.transformer.h[i](hidden)[0]
            ln_out = model.transformer.h[layer_idx].ln_2(hidden)
            mlp_out = mlp(ln_out)
            inputs_list.append(ln_out.squeeze(0).cpu())
            outputs_list.append(mlp_out.squeeze(0).cpu())
            collected += toks.shape[1]
        if collected >= n_tokens: break
    X = torch.cat(inputs_list, dim=0).numpy()[:n_tokens]
    Y = torch.cat(outputs_list, dim=0).numpy()[:n_tokens]
    lam = 1e-3
    W = np.linalg.solve(X.T @ X + lam * np.eye(768), X.T @ Y)
    b = Y.mean(0) - X.mean(0) @ W
    return torch.tensor(W, dtype=torch.float32), torch.tensor(b, dtype=torch.float32)


def collect_training_data(model, tokenizer, layer_idx, W_lin, b_lin, n_tokens=8000):
    ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="train[1000:1500]")
    mlp = model.transformer.h[layer_idx].mlp
    original_forward = mlp.forward
    loss_fn = nn.CrossEntropyLoss(reduction='none')
    all_acts, all_L_full, all_L_lin = [], [], []
    collected = 0
    for ex in ds:
        text = ex["text"]
        if not text.strip(): continue
        toks = tokenizer.encode(text, return_tensors="pt").to(device)
        if toks.shape[1] < 4: continue
        toks = toks[:, :256]
        with torch.no_grad():
            hidden = model.transformer.wte(toks) + model.transformer.wpe(
                torch.arange(toks.shape[1], device=device))
            for i in range(layer_idx):
                hidden = model.transformer.h[i](hidden)[0]
            ln_out = model.transformer.h[layer_idx].ln_2(hidden)
            out_full = model(toks)
            logits_full = out_full.logits[:, :-1, :]
            targets = toks[:, 1:]
            L_full = loss_fn(logits_full.reshape(-1, logits_full.size(-1)), targets.reshape(-1))
            mlp.forward = lambda x: x @ W_lin.to(device) + b_lin.to(device)
            out_lin = model(toks)
            logits_lin = out_lin.logits[:, :-1, :]
            L_lin = loss_fn(logits_lin.reshape(-1, logits_lin.size(-1)), targets.reshape(-1))
            mlp.forward = original_forward
            n = toks.shape[1] - 1
            all_acts.append(ln_out[0, :n].cpu())
            all_L_full.append(L_full.cpu())
            all_L_lin.append(L_lin.cpu())
            collected += n
        if collected >= n_tokens: break
    return (torch.cat(all_acts, 0)[:n_tokens],
            torch.cat(all_L_full, 0)[:n_tokens],
            torch.cat(all_L_lin, 0)[:n_tokens])


def make_gate(bottleneck, has_hidden=True):
    if has_hidden:
        gate = nn.Sequential(
            nn.Linear(768, bottleneck),
            nn.ReLU(),
            nn.Linear(bottleneck, 1),
            nn.Sigmoid()
        )
        n_params = 768 * bottleneck + bottleneck + bottleneck + 1
    else:
        gate = nn.Sequential(
            nn.Linear(768, 1),
            nn.Sigmoid()
        )
        n_params = 769
    return gate, n_params


def train_gate(acts, L_full, L_lin, bottleneck, has_hidden=True, epochs=500):
    deltas = L_lin - L_full
    gate, n_params = make_gate(bottleneck, has_hidden)
    optimizer = torch.optim.Adam(gate.parameters(), lr=1e-3)
    for ep in range(epochs):
        perm = torch.randperm(len(acts))[:512]
        g = gate(acts[perm]).squeeze()
        d = deltas[perm]
        loss = (g * d).mean() + 0.01 * g.mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    gate.eval()
    return gate, n_params


def eval_single_layer_gate(model, tokenizer, layer_idx, W_lin, b_lin, gate, n_tokens=12000):
    """Evaluate PPL with gate on a single layer."""
    ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="train[2000:2500]")
    mlp = model.transformer.h[layer_idx].mlp
    original_forward = mlp.forward
    total_loss = 0.0
    total_tokens = 0
    total_linear = 0
    total_full = 0

    def gated_forward(x):
        nonlocal total_linear, total_full
        with torch.no_grad():
            g = gate(x).squeeze(-1)
            use_linear = g > 0.5
        total_linear += use_linear.sum().item()
        total_full += (~use_linear).sum().item()
        linear_out = x @ W_lin.to(device) + b_lin.to(device)
        if use_linear.all():
            return linear_out
        elif not use_linear.any():
            return original_forward(x)
        else:
            full_out = original_forward(x)
            mask = use_linear.unsqueeze(-1).expand_as(x)
            return torch.where(mask, linear_out, full_out)

    mlp.forward = gated_forward
    try:
        for ex in ds:
            text = ex["text"]
            if not text.strip(): continue
            toks = tokenizer.encode(text, return_tensors="pt").to(device)
            if toks.shape[1] < 2: continue
            toks = toks[:, :512]
            with torch.no_grad():
                out = model(toks, labels=toks)
                total_loss += out.loss.item() * (toks.shape[1] - 1)
                total_tokens += toks.shape[1] - 1
            if total_tokens >= n_tokens: break
    finally:
        mlp.forward = original_forward

    ppl = np.exp(total_loss / total_tokens)
    pct_lin = total_linear / max(1, total_linear + total_full) * 100
    return ppl, pct_lin


def eval_compound(model, tokenizer, layer_gates, n_tokens=15000):
    """Evaluate PPL with gates on ALL layers simultaneously.
    layer_gates: dict of {layer_idx: (W_lin, b_lin, gate)} or None for no gate.
    """
    ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="train[2000:2500]")

    # Store originals and install gated forwards
    originals = {}
    layer_stats = {i: {"linear": 0, "full": 0} for i in layer_gates}

    for layer_idx, (W_lin, b_lin, gate) in layer_gates.items():
        mlp = model.transformer.h[layer_idx].mlp
        originals[layer_idx] = mlp.forward
        stats = layer_stats[layer_idx]

        # Need to capture variables properly in closure
        def make_gated(orig_fwd, w_lin, b_lin, g, st):
            def gated_forward(x):
                with torch.no_grad():
                    gv = g(x).squeeze(-1)
                    use_linear = gv > 0.5
                st["linear"] += use_linear.sum().item()
                st["full"] += (~use_linear).sum().item()
                linear_out = x @ w_lin.to(device) + b_lin.to(device)
                if use_linear.all():
                    return linear_out
                elif not use_linear.any():
                    return orig_fwd(x)
                else:
                    full_out = orig_fwd(x)
                    mask = use_linear.unsqueeze(-1).expand_as(x)
                    return torch.where(mask, linear_out, full_out)
            return gated_forward

        mlp.forward = make_gated(originals[layer_idx], W_lin, b_lin, gate, stats)

    total_loss = 0.0
    total_tokens = 0
    try:
        for ex in ds:
            text = ex["text"]
            if not text.strip(): continue
            toks = tokenizer.encode(text, return_tensors="pt").to(device)
            if toks.shape[1] < 2: continue
            toks = toks[:, :512]
            with torch.no_grad():
                out = model(toks, labels=toks)
                total_loss += out.loss.item() * (toks.shape[1] - 1)
                total_tokens += toks.shape[1] - 1
            if total_tokens >= n_tokens: break
    finally:
        for layer_idx in layer_gates:
            model.transformer.h[layer_idx].mlp.forward = originals[layer_idx]

    ppl = np.exp(total_loss / total_tokens)

    per_layer = {}
    for i, st in layer_stats.items():
        total = st["linear"] + st["full"]
        per_layer[i] = round(st["linear"] / max(1, total) * 100, 1)

    return ppl, per_layer


def main():
    print("=" * 80, flush=True)
    print("COMPOUND GATING: Optimal gate per layer, all running together", flush=True)
    print("=" * 80, flush=True)

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2").to(device).eval()

    # Baseline
    print("\nBaseline PPL...", flush=True)
    ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="train[2000:2500]")
    total_loss = 0.0
    total_tokens = 0
    for ex in ds:
        text = ex["text"]
        if not text.strip(): continue
        toks = tokenizer.encode(text, return_tensors="pt").to(device)
        if toks.shape[1] < 2: continue
        toks = toks[:, :512]
        with torch.no_grad():
            out = model(toks, labels=toks)
            total_loss += out.loss.item() * (toks.shape[1] - 1)
            total_tokens += toks.shape[1] - 1
        if total_tokens >= 15000: break
    baseline_ppl = np.exp(total_loss / total_tokens)
    print(f"  Baseline: {baseline_ppl:.2f}", flush=True)

    # Phase 1: Find optimal gate per layer
    gate_configs = [
        ("none", 0, False),
        ("linear", 0, False),
        ("b=1", 1, True),
        ("b=3", 3, True),
        ("b=6", 6, True),
    ]

    all_results = {"baseline_ppl": round(baseline_ppl, 2)}
    optimal_gates = {}  # layer_idx -> (name, gate, W_lin, b_lin, params, ppl, pct_lin)

    for layer_idx in range(12):
        print(f"\n{'='*60}", flush=True)
        print(f"LAYER {layer_idx}: Finding optimal gate", flush=True)
        print(f"{'='*60}", flush=True)

        t0 = time.time()
        W_lin, b_lin = fit_linear_approx(model, layer_idx)

        print("  Collecting training data...", flush=True)
        acts, L_full, L_lin = collect_training_data(model, tokenizer, layer_idx, W_lin, b_lin)

        best_name = "none"
        best_ppl = baseline_ppl
        best_gate = None
        best_params = 0
        best_pct = 0.0

        layer_results = []

        for name, b_size, has_hidden in gate_configs:
            if name == "none":
                layer_results.append({
                    "name": "none", "params": 0,
                    "ppl": round(baseline_ppl, 2), "delta_pct": 0.0, "pct_linear": 0.0
                })
                continue

            # Train gate (best of 2 trials)
            trial_best_ppl = float('inf')
            trial_best_gate = None
            trial_best_pct = 0
            trial_params = 0

            for trial in range(2):
                if name == "linear":
                    gate, n_params = train_gate(acts, L_full, L_lin, 0, has_hidden=False)
                else:
                    gate, n_params = train_gate(acts, L_full, L_lin, b_size, has_hidden=True)

                ppl, pct_lin = eval_single_layer_gate(
                    model, tokenizer, layer_idx, W_lin, b_lin, gate, n_tokens=10000)

                if ppl < trial_best_ppl:
                    trial_best_ppl = ppl
                    trial_best_gate = gate
                    trial_best_pct = pct_lin
                    trial_params = n_params

            delta = (trial_best_ppl - baseline_ppl) / baseline_ppl * 100
            layer_results.append({
                "name": name, "params": trial_params,
                "ppl": round(trial_best_ppl, 2), "delta_pct": round(delta, 2),
                "pct_linear": round(trial_best_pct, 1)
            })

            marker = " ⭐" if trial_best_ppl < best_ppl else ""
            print(f"  {name:>8} ({trial_params:>5} params): "
                  f"ppl={trial_best_ppl:.2f} ({delta:+.2f}%) "
                  f"{trial_best_pct:.1f}% linear{marker}", flush=True)

            if trial_best_ppl < best_ppl:
                best_name = name
                best_ppl = trial_best_ppl
                best_gate = trial_best_gate
                best_params = trial_params
                best_pct = trial_best_pct

        dt = time.time() - t0
        all_results[f"layer_{layer_idx}"] = {
            "results": layer_results,
            "best": best_name,
            "best_params": best_params,
            "best_ppl": round(best_ppl, 2),
            "best_pct_linear": round(best_pct, 1),
            "time": round(dt, 1)
        }

        if best_gate is not None:
            optimal_gates[layer_idx] = (best_name, best_gate, W_lin, b_lin,
                                        best_params, best_ppl, best_pct)
            print(f"  → Best: {best_name} (ppl={best_ppl:.2f}, {best_pct:.1f}% linear) [{dt:.0f}s]",
                  flush=True)
        else:
            print(f"  → Best: none (no gate improves this layer) [{dt:.0f}s]", flush=True)

    # Phase 2: Summary of per-layer optimal gates
    print("\n" + "=" * 80, flush=True)
    print("PER-LAYER OPTIMAL GATES", flush=True)
    print("=" * 80, flush=True)
    print(f"  {'Layer':>5} {'Gate':>8} {'Params':>8} {'PPL':>8} {'Δ%':>8} {'%Lin':>6}", flush=True)
    print(f"  {'-'*50}", flush=True)

    total_params = 0
    for layer_idx in range(12):
        r = all_results[f"layer_{layer_idx}"]
        delta = (r["best_ppl"] - baseline_ppl) / baseline_ppl * 100
        total_params += r["best_params"]
        marker = " ⭐" if r["best_ppl"] < baseline_ppl else ""
        print(f"  {layer_idx:>5} {r['best']:>8} {r['best_params']:>8} "
              f"{r['best_ppl']:>8.2f} {delta:>+8.2f} {r['best_pct_linear']:>5.1f}%{marker}", flush=True)

    print(f"\n  Total gate params: {total_params:,} ({total_params/124_000_000*100:.4f}% of GPT-2)", flush=True)

    # Phase 3: Compound evaluation
    print("\n" + "=" * 80, flush=True)
    print("COMPOUND EVALUATION: All optimal gates running simultaneously", flush=True)
    print("=" * 80, flush=True)

    # Build compound gate dict
    compound_gates = {}
    for layer_idx, (name, gate, W_lin, b_lin, params, ppl, pct) in optimal_gates.items():
        compound_gates[layer_idx] = (W_lin, b_lin, gate)

    print(f"  Gating {len(compound_gates)} layers: {sorted(compound_gates.keys())}", flush=True)

    compound_ppl, per_layer_pct = eval_compound(model, tokenizer, compound_gates)
    compound_delta = (compound_ppl - baseline_ppl) / baseline_ppl * 100

    print(f"\n  Compound PPL: {compound_ppl:.2f} ({compound_delta:+.2f}%)", flush=True)
    print(f"  Per-layer linear %:", flush=True)
    for i in sorted(per_layer_pct):
        print(f"    Layer {i}: {per_layer_pct[i]:.1f}% linear", flush=True)

    # Estimate total MLP compute saved
    total_mlp_positions = 12  # one per layer per position
    saved = sum(per_layer_pct[i] / 100 for i in per_layer_pct)
    avg_saved = saved / 12 * 100
    print(f"\n  Average MLP skip rate: {avg_saved:.1f}% across gated layers", flush=True)
    print(f"  Total MLP compute saved: {saved/12*100:.1f}% of all MLP operations", flush=True)

    all_results["compound"] = {
        "ppl": round(compound_ppl, 2),
        "delta_pct": round(compound_delta, 2),
        "per_layer_linear_pct": {str(k): v for k, v in per_layer_pct.items()},
        "total_gate_params": total_params,
        "layers_gated": len(compound_gates),
    }

    out_path = "/Users/peter/clawd/projects/sense-stack/code/compound_gate_results.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved to {out_path}", flush=True)


if __name__ == "__main__":
    main()
