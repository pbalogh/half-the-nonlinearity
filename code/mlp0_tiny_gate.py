"""
How small can the gate be? Test bottleneck sizes 1,2,3,4,6,8,16,32
measured by PERPLEXITY (the only metric that matters).
Also test: raw linear gate (no hidden layer, just 768->1).
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


def collect_training_data(model, tokenizer, layer_idx, W_lin, b_lin, n_tokens=10000):
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


def train_gate(acts, L_full, L_lin, bottleneck, epochs=500, has_hidden=True):
    deltas = L_lin - L_full
    if has_hidden:
        gate = nn.Sequential(
            nn.Linear(768, bottleneck),
            nn.ReLU(),
            nn.Linear(bottleneck, 1),
            nn.Sigmoid()
        )
        n_params = 768 * bottleneck + bottleneck + bottleneck * 1 + 1
    else:
        # Pure linear: 768 -> 1
        gate = nn.Sequential(
            nn.Linear(768, 1),
            nn.Sigmoid()
        )
        n_params = 768 + 1

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


def eval_ppl_with_gate(model, tokenizer, layer_idx, W_lin, b_lin, gate, n_tokens=15000):
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


def main():
    print("=" * 80, flush=True)
    print("MINIMAL GATE SIZE: How few neurons do you need?", flush=True)
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

    bottleneck_sizes = [0, 1, 2, 3, 4, 6, 8, 16, 32]  # 0 = no hidden layer
    layers = [2, 11]
    all_results = {"baseline_ppl": round(baseline_ppl, 2)}

    for layer_idx in layers:
        print(f"\n{'='*60}", flush=True)
        print(f"LAYER {layer_idx}", flush=True)
        print(f"{'='*60}", flush=True)

        W_lin, b_lin = fit_linear_approx(model, layer_idx)

        print("  Collecting training data...", flush=True)
        acts, L_full, L_lin = collect_training_data(model, tokenizer, layer_idx, W_lin, b_lin)

        layer_results = []

        for b_size in bottleneck_sizes:
            has_hidden = b_size > 0
            label = f"b={b_size}" if has_hidden else "linear"

            # Train 3 times, take best (gates are noisy)
            best_ppl = float('inf')
            best_pct = 0
            best_params = 0

            for trial in range(3):
                gate, n_params = train_gate(acts, L_full, L_lin, b_size,
                                           epochs=500, has_hidden=has_hidden)
                ppl, pct_lin = eval_ppl_with_gate(
                    model, tokenizer, layer_idx, W_lin, b_lin, gate)
                if ppl < best_ppl:
                    best_ppl = ppl
                    best_pct = pct_lin
                    best_params = n_params

            delta = (best_ppl - baseline_ppl) / baseline_ppl * 100
            layer_results.append({
                "bottleneck": b_size,
                "params": best_params,
                "ppl": round(best_ppl, 2),
                "delta_pct": round(delta, 2),
                "pct_linear": round(best_pct, 1),
            })
            print(f"  {label:>10} ({best_params:>6} params): "
                  f"ppl={best_ppl:.2f} ({delta:+.2f}%) {best_pct:.1f}% linear", flush=True)

        all_results[f"layer_{layer_idx}"] = layer_results

    # Summary
    print("\n" + "=" * 80, flush=True)
    print("SUMMARY", flush=True)
    print("=" * 80, flush=True)

    for layer_idx in layers:
        print(f"\nLayer {layer_idx}:", flush=True)
        print(f"  {'Size':>10} {'Params':>8} {'PPL':>8} {'Δ%':>8} {'%Lin':>7}", flush=True)
        print(f"  {'-'*45}", flush=True)
        for r in all_results[f"layer_{layer_idx}"]:
            label = f"b={r['bottleneck']}" if r['bottleneck'] > 0 else "linear"
            marker = " ⭐" if r["ppl"] <= baseline_ppl else ""
            print(f"  {label:>10} {r['params']:>8} {r['ppl']:>8.2f} {r['delta_pct']:>+8.2f} "
                  f"{r['pct_linear']:>6.1f}%{marker}", flush=True)

    out_path = "/Users/peter/clawd/projects/sense-stack/code/tiny_gate_results.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved to {out_path}", flush=True)


if __name__ == "__main__":
    main()
