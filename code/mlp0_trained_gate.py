"""
Trained routing gate: a tiny MLP that sees the actual hidden state
(post-attention, post-LayerNorm — the MLP's input) and decides whether
this activation needs the full nonlinear MLP or can use the linear shortcut.

This is the right abstraction: not "which token" but "which activation pattern."
Essentially a learned MoE gate where the two experts are {linear, full_MLP}.

Architecture:
  Gate: 768 -> 32 -> 1 (sigmoid)  [~25K params, trivial vs 4.7M MLP params]
  Training: minimize gate_prob * linear_loss + (1-gate_prob) * full_loss
            + sparsity_penalty * mean(gate_prob)
  
  The sparsity penalty encourages using linear path as much as possible.
"""

import torch
import torch.nn as nn
import numpy as np
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from datasets import load_dataset
import json, time, os

device = "cpu"


class ActivationGate(nn.Module):
    """Tiny gate: hidden_state -> probability of using linear path."""
    def __init__(self, hidden_dim=768, bottleneck=32):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim, bottleneck),
            nn.ReLU(),
            nn.Linear(bottleneck, 1),
        )
    
    def forward(self, x):
        """x: (batch, seq, hidden) -> (batch, seq, 1) probabilities"""
        return torch.sigmoid(self.gate(x))


def fit_linear_approx(model, layer_idx, n_tokens=10000):
    """Fit linear approximation for a layer's MLP."""
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
    r2 = 1 - ((Y - X @ W - b)**2).sum() / ((Y - Y.mean(0))**2).sum()
    print(f"    Linear R² = {r2:.4f}", flush=True)
    
    return torch.tensor(W, dtype=torch.float32), torch.tensor(b, dtype=torch.float32)


def collect_training_data(model, tokenizer, layer_idx, W_lin, b_lin, n_tokens=8000):
    """Collect (mlp_input, loss_with_linear, loss_with_full) triples.
    We need per-position losses under both routing decisions.
    """
    ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="train[1000:1500]")
    mlp = model.transformer.h[layer_idx].mlp
    original_forward = mlp.forward
    loss_fn = nn.CrossEntropyLoss(reduction='none')
    
    all_inputs = []  # MLP input activations
    all_loss_full = []
    all_loss_linear = []
    collected = 0
    
    for ex in ds:
        text = ex["text"]
        if not text.strip(): continue
        toks = tokenizer.encode(text, return_tensors="pt").to(device)
        if toks.shape[1] < 4: continue
        toks = toks[:, :256]
        
        # Capture MLP input
        mlp_inputs = []
        def hook_fn(module, input, output):
            mlp_inputs.append(input[0].detach())
        
        # We need to hook the layer norm before MLP
        handle = model.transformer.h[layer_idx].ln_2.register_forward_hook(
            lambda m, inp, out: mlp_inputs.append(out.detach()))
        
        # Full model loss
        with torch.no_grad():
            out_full = model(toks)
            logits_full = out_full.logits[:, :-1, :]
            targets = toks[:, 1:]
            loss_full = loss_fn(logits_full.reshape(-1, logits_full.size(-1)),
                              targets.reshape(-1))
        
        handle.remove()
        ln_input = mlp_inputs[-1]  # (1, seq, 768)
        mlp_inputs.clear()
        
        # Linear replacement loss
        def linear_fwd(x):
            return x @ W_lin.to(device) + b_lin.to(device)
        
        mlp.forward = linear_fwd
        with torch.no_grad():
            out_lin = model(toks)
            logits_lin = out_lin.logits[:, :-1, :]
            loss_lin = loss_fn(logits_lin.reshape(-1, logits_lin.size(-1)),
                             targets.reshape(-1))
        mlp.forward = original_forward
        
        # Store: each position's MLP input and both losses
        # Note: loss[i] corresponds to predicting token[i+1] given token[0:i+1]
        # The MLP input at position i contributes to loss[i]
        seq_len = toks.shape[1] - 1
        all_inputs.append(ln_input[0, :seq_len].cpu())
        all_loss_full.append(loss_full.cpu())
        all_loss_linear.append(loss_lin.cpu())
        collected += seq_len
        
        if collected >= n_tokens:
            break
    
    X = torch.cat(all_inputs, dim=0)[:n_tokens]
    L_full = torch.cat(all_loss_full, dim=0)[:n_tokens]
    L_lin = torch.cat(all_loss_linear, dim=0)[:n_tokens]
    
    return X, L_full, L_lin


def train_gate(X, L_full, L_lin, sparsity_weight=0.1, n_epochs=500, lr=1e-3):
    """Train the gate to minimize expected loss while maximizing linear usage.
    
    Loss = gate * L_linear + (1-gate) * L_full + sparsity * mean(gate)
    
    Where gate=1 means "use linear" and gate=0 means "use full MLP".
    The gate learns to route to linear when L_linear ≈ L_full,
    and to full MLP when L_linear >> L_full.
    """
    n = len(X)
    idx = torch.randperm(n)
    split = int(0.8 * n)
    train_idx, val_idx = idx[:split], idx[split:]
    
    gate = ActivationGate(X.shape[1], bottleneck=32)
    optimizer = torch.optim.Adam(gate.parameters(), lr=lr)
    
    best_val_loss = float('inf')
    best_state = None
    
    for epoch in range(n_epochs):
        gate.train()
        
        # Mini-batch
        perm = torch.randperm(len(train_idx))[:512]
        batch_idx = train_idx[perm]
        
        x_batch = X[batch_idx]
        lf_batch = L_full[batch_idx]
        ll_batch = L_lin[batch_idx]
        
        g = gate(x_batch.unsqueeze(0)).squeeze()  # (batch,)
        
        # Expected loss under routing
        expected_loss = g * ll_batch + (1 - g) * lf_batch
        
        # Sparsity: encourage using linear (g → 1)
        sparsity_loss = -sparsity_weight * g.mean()
        
        total_loss = expected_loss.mean() + sparsity_loss
        
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 50 == 0:
            gate.eval()
            with torch.no_grad():
                x_val = X[val_idx]
                lf_val = L_full[val_idx]
                ll_val = L_lin[val_idx]
                
                g_val = gate(x_val.unsqueeze(0)).squeeze()
                val_expected = (g_val * ll_val + (1 - g_val) * lf_val).mean()
                pct_linear = (g_val > 0.5).float().mean().item() * 100
                
                # Oracle: always pick the better option
                oracle_loss = torch.min(lf_val, ll_val).mean()
                # Always-linear loss
                always_lin = ll_val.mean()
                # Always-full loss
                always_full = lf_val.mean()
                
                if val_expected < best_val_loss:
                    best_val_loss = val_expected.item()
                    best_state = {k: v.clone() for k, v in gate.state_dict().items()}
                
                if (epoch + 1) % 100 == 0:
                    print(f"    ep{epoch+1}: gate_loss={val_expected:.4f} "
                          f"full={always_full:.4f} linear={always_lin:.4f} "
                          f"oracle={oracle_loss:.4f} %linear={pct_linear:.1f}%", flush=True)
    
    if best_state:
        gate.load_state_dict(best_state)
    
    # Final analysis
    gate.eval()
    with torch.no_grad():
        g_all = gate(X.unsqueeze(0)).squeeze()
        pct_linear = (g_all > 0.5).float().mean().item() * 100
        
        # What does the gate look at? Analyze weight structure
        W1 = gate.gate[0].weight.data  # (32, 768)
        # Which input dimensions matter most?
        dim_importance = W1.abs().sum(dim=0)  # sum across hidden units
        top_dims = torch.argsort(dim_importance, descending=True)[:20]
    
    return gate, pct_linear, top_dims


def eval_gated_ppl(model, tokenizer, layer_idx, W_lin, b_lin, gate, n_tokens=15000):
    """Evaluate perplexity using the trained gate for routing."""
    ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="train[2000:2500]")
    
    mlp = model.transformer.h[layer_idx].mlp
    original_forward = mlp.forward
    total_loss = 0.0
    total_tokens = 0
    total_linear = 0
    total_nonlinear = 0
    
    # We need to intercept after LN2 and route
    captured_ln_out = [None]
    handle = model.transformer.h[layer_idx].ln_2.register_forward_hook(
        lambda m, inp, out: captured_ln_out.__setitem__(0, out.detach()))
    
    def gated_forward(x):
        nonlocal total_linear, total_nonlinear
        
        with torch.no_grad():
            g = gate(x)  # (batch, seq, 1)
            use_linear = (g > 0.5).squeeze(-1)  # (batch, seq)
        
        linear_out = x @ W_lin.to(device) + b_lin.to(device)
        
        n_lin = use_linear.sum().item()
        n_nl = (~use_linear).sum().item()
        total_linear += n_lin
        total_nonlinear += n_nl
        
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
                outputs = model(toks, labels=toks)
                total_loss += outputs.loss.item() * (toks.shape[1] - 1)
                total_tokens += toks.shape[1] - 1
            
            if total_tokens >= n_tokens:
                break
    finally:
        mlp.forward = original_forward
        handle.remove()
    
    ppl = np.exp(total_loss / total_tokens)
    pct_linear = total_linear / max(1, total_linear + total_nonlinear) * 100
    return ppl, pct_linear


def analyze_gate_decisions(model, tokenizer, layer_idx, gate, W_lin, b_lin, n_tokens=5000):
    """Analyze what patterns the gate routes to linear vs full MLP."""
    ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="train[2500:3000]")
    mlp = model.transformer.h[layer_idx].mlp
    
    linear_tokens = []
    nonlinear_tokens = []
    
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
            
            g = gate(ln_out)  # (1, seq, 1)
            use_linear = (g > 0.5).squeeze()
        
        for pos in range(toks.shape[1]):
            tid = toks[0, pos].item()
            tok_str = tokenizer.decode([tid])
            if use_linear[pos]:
                linear_tokens.append((tid, tok_str, pos))
            else:
                nonlinear_tokens.append((tid, tok_str, pos))
        
        if len(linear_tokens) + len(nonlinear_tokens) >= n_tokens:
            break
    
    print(f"\n  Gate routing analysis:", flush=True)
    print(f"    Linear: {len(linear_tokens)} ({len(linear_tokens)/(len(linear_tokens)+len(nonlinear_tokens))*100:.1f}%)", flush=True)
    print(f"    Nonlinear: {len(nonlinear_tokens)} ({len(nonlinear_tokens)/(len(linear_tokens)+len(nonlinear_tokens))*100:.1f}%)", flush=True)
    
    # Token overlap analysis
    lin_tids = set(t[0] for t in linear_tokens)
    nl_tids = set(t[0] for t in nonlinear_tokens)
    both = lin_tids & nl_tids
    print(f"    Tokens appearing in BOTH routes: {len(both)} "
          f"({len(both)/max(1,len(lin_tids|nl_tids))*100:.1f}% of all types)", flush=True)
    
    # Position distribution
    lin_positions = [t[2] for t in linear_tokens]
    nl_positions = [t[2] for t in nonlinear_tokens]
    print(f"    Mean position: linear={np.mean(lin_positions):.1f} nonlinear={np.mean(nl_positions):.1f}", flush=True)
    
    # Sample decisions
    print(f"\n  Sample LINEAR tokens: {[t[1] for t in linear_tokens[:30]]}", flush=True)
    print(f"  Sample NONLINEAR tokens: {[t[1] for t in nonlinear_tokens[:30]]}", flush=True)
    
    # Context analysis: show same token routed differently
    if both:
        example_tid = list(both)[0]
        tok_str = tokenizer.decode([example_tid])
        lin_contexts = [(t[2],) for t in linear_tokens if t[0] == example_tid][:3]
        nl_contexts = [(t[2],) for t in nonlinear_tokens if t[0] == example_tid][:3]
        print(f"\n  Token '{tok_str}' (id={example_tid}) routed both ways:", flush=True)
        print(f"    Linear at positions: {[c[0] for c in lin_contexts]}", flush=True)
        print(f"    Nonlinear at positions: {[c[0] for c in nl_contexts]}", flush=True)
    
    return len(both), len(lin_tids | nl_tids)


def main():
    print("=" * 80, flush=True)
    print("TRAINED ACTIVATION GATE: Route by Hidden State", flush=True)
    print("=" * 80, flush=True)
    
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2").to(device).eval()
    
    # Get baselines
    print("\nBaselines...", flush=True)
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
    print(f"  Baseline PPL: {baseline_ppl:.2f}", flush=True)
    
    layers_to_test = [0, 1, 2, 5, 8, 11]
    all_results = {}
    
    for layer_idx in layers_to_test:
        print(f"\n{'='*70}", flush=True)
        print(f"LAYER {layer_idx}", flush=True)
        print(f"{'='*70}", flush=True)
        
        # 1. Fit linear approximation
        print("  Fitting linear approx...", flush=True)
        W_lin, b_lin = fit_linear_approx(model, layer_idx)
        
        # 2. Collect training data
        print("  Collecting training data...", flush=True)
        X, L_full, L_lin = collect_training_data(model, tokenizer, layer_idx, W_lin, b_lin)
        
        loss_delta = L_lin - L_full
        print(f"  Loss delta: mean={loss_delta.mean():.4f} std={loss_delta.std():.4f}", flush=True)
        print(f"  % where linear is better: {(loss_delta < 0).float().mean()*100:.1f}%", flush=True)
        
        # 3. Train gates with different sparsity levels
        layer_results = {"baseline_ppl": baseline_ppl}
        
        for sparsity in [0.0, 0.1, 0.5, 1.0, 2.0]:
            print(f"\n  --- Sparsity weight = {sparsity} ---", flush=True)
            gate, train_pct_linear, top_dims = train_gate(
                X, L_full, L_lin, sparsity_weight=sparsity)
            
            # 4. Evaluate perplexity
            ppl, eval_pct_linear = eval_gated_ppl(
                model, tokenizer, layer_idx, W_lin, b_lin, gate)
            delta = (ppl - baseline_ppl) / baseline_ppl * 100
            
            print(f"  Gate PPL: {ppl:.2f} ({delta:+.2f}%) | {eval_pct_linear:.1f}% linear", flush=True)
            print(f"  Top gate dimensions: {top_dims[:10].tolist()}", flush=True)
            
            layer_results[f"sp{sparsity}"] = {
                "ppl": round(ppl, 2),
                "delta_pct": round(delta, 2),
                "pct_linear": round(eval_pct_linear, 1),
                "top_dims": top_dims[:10].tolist(),
            }
            
            # 5. Analyze gate decisions for the best sparsity
            if sparsity == 0.5:
                n_both, n_total = analyze_gate_decisions(
                    model, tokenizer, layer_idx, gate, W_lin, b_lin)
                layer_results["context_dependent"] = {
                    "tokens_routed_both_ways": n_both,
                    "total_token_types": n_total,
                    "pct_context_dependent": round(n_both / max(1, n_total) * 100, 1)
                }
        
        # All-linear baseline for this layer
        print(f"\n  All-linear comparison:", flush=True)
        mlp = model.transformer.h[layer_idx].mlp
        orig_fwd = mlp.forward
        mlp.forward = lambda x: x @ W_lin.to(device) + b_lin.to(device)
        total_loss = 0.0
        total_tokens = 0
        ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="train[2000:2500]")
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
        mlp.forward = orig_fwd
        all_lin_ppl = np.exp(total_loss / total_tokens)
        print(f"  All-linear PPL: {all_lin_ppl:.2f} ({(all_lin_ppl-baseline_ppl)/baseline_ppl*100:+.2f}%)", flush=True)
        layer_results["all_linear_ppl"] = round(all_lin_ppl, 2)
        
        all_results[f"layer_{layer_idx}"] = layer_results
        
        out_path = "/Users/peter/clawd/projects/sense-stack/code/trained_gate_results.json"
        with open(out_path, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"  (saved)", flush=True)
    
    # Summary
    print("\n" + "=" * 80, flush=True)
    print("SUMMARY: Trained Gate Results", flush=True)
    print("=" * 80, flush=True)
    print(f"Baseline PPL: {baseline_ppl:.2f}", flush=True)
    print(f"\n{'Layer':>6} {'All-Lin':>10} {'Gate sp0':>10} {'Gate sp0.5':>12} "
          f"{'Gate sp1':>10} {'%Lin@0.5':>10} {'%CtxDep':>10}", flush=True)
    print("-" * 75, flush=True)
    for layer_idx in layers_to_test:
        key = f"layer_{layer_idx}"
        if key not in all_results: continue
        r = all_results[key]
        al = r.get("all_linear_ppl", "?")
        g0 = r.get("sp0.0", {}).get("ppl", "?")
        g05 = r.get("sp0.5", {}).get("ppl", "?")
        g1 = r.get("sp1.0", {}).get("ppl", "?")
        pl = r.get("sp0.5", {}).get("pct_linear", "?")
        cd = r.get("context_dependent", {}).get("pct_context_dependent", "?")
        print(f"{layer_idx:>6} {al:>10} {g0:>10} {g05:>12} {g1:>10} {pl:>9}% {cd:>9}%", flush=True)


if __name__ == "__main__":
    main()
