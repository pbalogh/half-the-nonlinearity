"""
Compound gating at scale: fit gates on all layers, run simultaneously.

Usage:
  python mlp0_scale_compound.py --model gpt2-medium
  python mlp0_scale_compound.py --model gpt2-large --device cuda
  python mlp0_scale_compound.py --model gpt2-medium --device mps --gate-size 1
  
Options:
  --gate-size: Force a specific bottleneck for all layers (0=linear, 1, 3, 6)
               Default: auto-select best per layer from [linear, b=1, b=3]
  --skip-layers: Comma-separated layers to NOT gate (e.g., "0" to skip layer 0)
"""

import torch
import torch.nn as nn
import numpy as np
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from datasets import load_dataset
import json, time, argparse, os

loss_fn_global = nn.CrossEntropyLoss()

def compute_loss(model, toks):
    """Compute cross-entropy loss manually, using float64 for lm_head to avoid overflow."""
    with torch.no_grad():
        hidden = model.transformer(toks)[0]
        logits = torch.nn.functional.linear(
            hidden.double(), model.lm_head.weight.double(),
            model.lm_head.bias.double() if model.lm_head.bias is not None else None
        )
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = toks[:, 1:].contiguous()
        loss = loss_fn_global(shift_logits.view(-1, shift_logits.size(-1)),
                              shift_labels.view(-1))
    return loss.item(), toks.shape[1] - 1


def fit_linear_approx(model, layer_idx, hidden_dim, device, n_tokens=10000):
    ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="train[:500]")
    tokenizer = GPT2Tokenizer.from_pretrained(model.config._name_or_path)
    mlp = model.transformer.h[layer_idx].mlp
    inputs_list, outputs_list = [], []
    collected = 0
    for ex in ds:
        text = ex["text"]
        if not text.strip():
            continue
        toks = tokenizer.encode(text, return_tensors="pt").to(device)
        if toks.shape[1] < 2:
            continue
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
        if collected >= n_tokens:
            break
    X = torch.cat(inputs_list, dim=0).numpy()[:n_tokens].astype(np.float64)
    Y = torch.cat(outputs_list, dim=0).numpy()[:n_tokens].astype(np.float64)
    X_mean = X.mean(0)
    Y_mean = Y.mean(0)
    Xc = X - X_mean
    Yc = Y - Y_mean
    U, s, Vt = np.linalg.svd(Xc, full_matrices=False)
    lam = 0.01
    d = s / (s**2 + lam)
    W = (Vt.T * d) @ U.T @ Yc
    b = Y_mean - X_mean @ W
    if np.isnan(W).any() or np.isnan(b).any():
        W = np.eye(hidden_dim) * 0.01
        b = Y_mean
    return torch.tensor(W, dtype=torch.float32), torch.tensor(b, dtype=torch.float32)


def collect_training_data(model, tokenizer, layer_idx, W_lin, b_lin, device, n_tokens=8000):
    ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="train[1000:1500]")
    mlp = model.transformer.h[layer_idx].mlp
    original_forward = mlp.forward
    loss_fn = nn.CrossEntropyLoss(reduction='none')
    all_acts, all_L_full, all_L_lin = [], [], []
    collected = 0
    for ex in ds:
        text = ex["text"]
        if not text.strip():
            continue
        toks = tokenizer.encode(text, return_tensors="pt").to(device)
        if toks.shape[1] < 4:
            continue
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
        if collected >= n_tokens:
            break
    return (torch.cat(all_acts, 0)[:n_tokens],
            torch.cat(all_L_full, 0)[:n_tokens],
            torch.cat(all_L_lin, 0)[:n_tokens])


def train_gate(acts, L_full, L_lin, hidden_dim, bottleneck, has_hidden=True, epochs=500):
    deltas = L_lin - L_full
    if has_hidden and bottleneck > 0:
        gate = nn.Sequential(
            nn.Linear(hidden_dim, bottleneck),
            nn.ReLU(),
            nn.Linear(bottleneck, 1),
            nn.Sigmoid()
        )
        n_params = hidden_dim * bottleneck + bottleneck + bottleneck + 1
    else:
        gate = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        n_params = hidden_dim + 1
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


def eval_single_gate(model, tokenizer, layer_idx, W_lin, b_lin, gate, device, n_tokens=10000):
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
            g = gate(x.cpu()).squeeze(-1)
            use_linear = (g > 0.5).to(device)
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
            if not text.strip():
                continue
            toks = tokenizer.encode(text, return_tensors="pt").to(device)
            if toks.shape[1] < 2:
                continue
            toks = toks[:, :512]
            with torch.no_grad():
                loss, n_toks = compute_loss(model, toks)
                total_loss += loss * n_toks
                total_tokens += n_toks
            if total_tokens >= n_tokens:
                break
    finally:
        mlp.forward = original_forward

    ppl = np.exp(total_loss / total_tokens)
    pct_lin = total_linear / max(1, total_linear + total_full) * 100
    return ppl, pct_lin


def eval_compound(model, tokenizer, layer_gates, device, n_tokens=15000):
    ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="train[2000:2500]")
    originals = {}
    layer_stats = {i: {"linear": 0, "full": 0} for i in layer_gates}

    for layer_idx, (W_lin, b_lin, gate) in layer_gates.items():
        mlp = model.transformer.h[layer_idx].mlp
        originals[layer_idx] = mlp.forward
        stats = layer_stats[layer_idx]

        def make_gated(orig_fwd, w_lin, b_lin, g, st, dev):
            def gated_forward(x):
                with torch.no_grad():
                    gv = g(x.cpu()).squeeze(-1)
                    use_linear = (gv > 0.5).to(dev)
                st["linear"] += use_linear.sum().item()
                st["full"] += (~use_linear).sum().item()
                linear_out = x @ w_lin.to(dev) + b_lin.to(dev)
                if use_linear.all():
                    return linear_out
                elif not use_linear.any():
                    return orig_fwd(x)
                else:
                    full_out = orig_fwd(x)
                    mask = use_linear.unsqueeze(-1).expand_as(x)
                    return torch.where(mask, linear_out, full_out)
            return gated_forward

        mlp.forward = make_gated(originals[layer_idx], W_lin, b_lin, gate, stats, device)

    total_loss = 0.0
    total_tokens = 0
    try:
        for ex in ds:
            text = ex["text"]
            if not text.strip():
                continue
            toks = tokenizer.encode(text, return_tensors="pt").to(device)
            if toks.shape[1] < 2:
                continue
            toks = toks[:, :512]
            with torch.no_grad():
                loss, n_toks = compute_loss(model, toks)
                total_loss += loss * n_toks
                total_tokens += n_toks
            if total_tokens >= n_tokens:
                break
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gpt2-medium")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--gate-size", type=int, default=-1,
                        help="Force gate bottleneck (-1=auto, 0=linear, 1,3,6)")
    parser.add_argument("--skip-layers", default="", help="Comma-separated layers to skip")
    parser.add_argument("--epochs", type=int, default=500)
    args = parser.parse_args()

    model_name = args.model
    device = args.device
    skip = set(int(x) for x in args.skip_layers.split(",") if x)

    configs = {
        "gpt2":        {"n_layers": 12, "hidden": 768},
        "gpt2-medium": {"n_layers": 24, "hidden": 1024},
        "gpt2-large":  {"n_layers": 36, "hidden": 1280},
    }
    cfg = configs[model_name]
    hidden_dim = cfg["hidden"]
    n_layers = cfg["n_layers"]

    print("=" * 80, flush=True)
    print(f"COMPOUND GATING: {model_name} ({n_layers} layers, {hidden_dim} hidden)", flush=True)
    print(f"  Device: {device}, Gate size: {'auto' if args.gate_size < 0 else args.gate_size}", flush=True)
    print(f"  Skipping layers: {skip or 'none'}", flush=True)
    print("=" * 80, flush=True)

    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name).to(device).eval()

    # Baseline
    print("\nBaseline PPL...", flush=True)
    ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="train[2000:2500]")
    total_loss = 0.0
    total_tokens = 0
    for ex in ds:
        text = ex["text"]
        if not text.strip():
            continue
        toks = tokenizer.encode(text, return_tensors="pt").to(device)
        if toks.shape[1] < 2:
            continue
        toks = toks[:, :512]
        with torch.no_grad():
            loss, n_toks = compute_loss(model, toks)
            total_loss += loss * n_toks
            total_tokens += n_toks
        if total_tokens >= 15000:
            break
    baseline_ppl = np.exp(total_loss / total_tokens)
    print(f"  Baseline: {baseline_ppl:.2f}", flush=True)

    # Gate sizes to try per layer
    if args.gate_size >= 0:
        sizes_to_try = [(f"b={args.gate_size}" if args.gate_size > 0 else "linear",
                         args.gate_size, args.gate_size > 0)]
    else:
        sizes_to_try = [
            ("linear", 0, False),
            ("b=1", 1, True),
            ("b=3", 3, True),
        ]

    optimal_gates = {}  # layer -> (W_lin, b_lin, gate, name, params, ppl, pct)
    all_results = {"model": model_name, "baseline_ppl": round(baseline_ppl, 2), "layers": {}}
    total_gate_params = 0

    for layer_idx in range(n_layers):
        if layer_idx in skip:
            print(f"\n  Layer {layer_idx}: SKIPPED", flush=True)
            continue

        t0 = time.time()
        print(f"\n  Layer {layer_idx}: fitting...", end=" ", flush=True)
        W_lin, b_lin = fit_linear_approx(model, layer_idx, hidden_dim, device)

        print("training...", end=" ", flush=True)
        acts, L_full, L_lin = collect_training_data(
            model, tokenizer, layer_idx, W_lin, b_lin, device)

        best_name = "none"
        best_ppl = baseline_ppl
        best_gate = None
        best_params = 0
        best_pct = 0

        for name, b_size, has_hidden in sizes_to_try:
            gate, n_params = train_gate(acts, L_full, L_lin, hidden_dim,
                                       b_size, has_hidden, epochs=args.epochs)
            ppl, pct = eval_single_gate(model, tokenizer, layer_idx, W_lin, b_lin,
                                        gate, device, n_tokens=10000)
            if ppl < best_ppl:
                best_name = name
                best_ppl = ppl
                best_gate = gate
                best_params = n_params
                best_pct = pct

        dt = time.time() - t0
        if best_gate is not None:
            optimal_gates[layer_idx] = (W_lin, b_lin, best_gate, best_name,
                                        best_params, best_ppl, best_pct)
            total_gate_params += best_params
            delta = (best_ppl - baseline_ppl) / baseline_ppl * 100
            print(f"{best_name} ppl={best_ppl:.2f} ({delta:+.2f}%) "
                  f"{best_pct:.1f}% lin [{dt:.0f}s] ⭐", flush=True)
        else:
            print(f"no improvement [{dt:.0f}s]", flush=True)

        all_results["layers"][str(layer_idx)] = {
            "best": best_name, "params": best_params,
            "ppl": round(best_ppl, 2), "pct_linear": round(best_pct, 1),
            "time": round(dt, 1),
        }

    # Compound eval
    print(f"\n{'='*60}", flush=True)
    print(f"COMPOUND EVALUATION ({len(optimal_gates)} layers gated)", flush=True)
    print(f"Total gate params: {total_gate_params:,}", flush=True)
    print(f"{'='*60}", flush=True)

    compound_dict = {i: (W, b, g) for i, (W, b, g, *_) in optimal_gates.items()}
    compound_ppl, per_layer_pct = eval_compound(model, tokenizer, compound_dict, device)
    compound_delta = (compound_ppl - baseline_ppl) / baseline_ppl * 100

    print(f"\n  Compound PPL: {compound_ppl:.2f} ({compound_delta:+.2f}%)", flush=True)
    for i in sorted(per_layer_pct):
        print(f"    Layer {i:>2}: {per_layer_pct[i]:.1f}% linear", flush=True)

    total_skip = sum(per_layer_pct[i] for i in per_layer_pct) / n_layers
    print(f"\n  Avg MLP skip: {total_skip:.1f}%", flush=True)

    all_results["compound"] = {
        "ppl": round(compound_ppl, 2),
        "delta_pct": round(compound_delta, 2),
        "per_layer_linear_pct": {str(k): v for k, v in per_layer_pct.items()},
        "total_gate_params": total_gate_params,
    }

    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           f"scale_compound_{model_name.replace('-','_')}.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved to {out_path}", flush=True)


if __name__ == "__main__":
    main()
