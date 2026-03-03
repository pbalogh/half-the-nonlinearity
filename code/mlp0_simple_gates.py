"""
Can simple geometric features match the learned 25K-param gate?
Test: kurtosis threshold, norm threshold, max_abs threshold,
PCA projection, and small combinations — measured by PERPLEXITY
(not classification accuracy, which can be misleading).

For each layer, compare:
1. Learned gate (768->32->1, 25K params)
2. Kurtosis threshold (1 param)
3. Norm threshold (1 param)
4. Max-abs threshold (1 param)
5. Top-1 PCA projection threshold (769 params — one direction + threshold)
6. Top-1 gate SVD direction threshold (769 params)
7. 3-feature logistic (kurtosis + norm + max_abs, 4 params)
8. Oracle (per-position, pick whichever is better)
"""

import torch
import torch.nn as nn
import numpy as np
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from datasets import load_dataset
from scipy.stats import kurtosis as scipy_kurtosis
import json, time, os

device = "cpu"


class ActivationGate(nn.Module):
    def __init__(self, hidden_dim=768, bottleneck=32):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim, bottleneck),
            nn.ReLU(),
            nn.Linear(bottleneck, 1),
        )
    def forward(self, x):
        return torch.sigmoid(self.gate(x))


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
    """Collect per-position losses under both conditions."""
    ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="train[1000:1500]")
    mlp = model.transformer.h[layer_idx].mlp
    original_forward = mlp.forward
    loss_fn = nn.CrossEntropyLoss(reduction='none')
    
    all_acts = []
    all_L_full = []
    all_L_lin = []
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
            
            # Full
            out_full = model(toks)
            logits_full = out_full.logits[:, :-1, :]
            targets = toks[:, 1:]
            L_full = loss_fn(logits_full.reshape(-1, logits_full.size(-1)),
                           targets.reshape(-1))
            
            # Linear
            mlp.forward = lambda x: x @ W_lin.to(device) + b_lin.to(device)
            out_lin = model(toks)
            logits_lin = out_lin.logits[:, :-1, :]
            L_lin = loss_fn(logits_lin.reshape(-1, logits_lin.size(-1)),
                          targets.reshape(-1))
            mlp.forward = original_forward
            
            seq_len = toks.shape[1] - 1
            all_acts.append(ln_out[0, :seq_len].cpu())
            all_L_full.append(L_full.cpu())
            all_L_lin.append(L_lin.cpu())
            collected += seq_len
        
        if collected >= n_tokens: break
    
    return (torch.cat(all_acts, dim=0)[:n_tokens],
            torch.cat(all_L_full, dim=0)[:n_tokens],
            torch.cat(all_L_lin, dim=0)[:n_tokens])


def eval_ppl_with_gate_fn(model, tokenizer, layer_idx, W_lin, b_lin, 
                           gate_fn, n_tokens=15000):
    """Evaluate PPL using a gate function: gate_fn(activation_tensor) -> bool mask (True=linear)."""
    ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="train[2000:2500]")
    mlp = model.transformer.h[layer_idx].mlp
    original_forward = mlp.forward
    total_loss = 0.0
    total_tokens = 0
    total_linear = 0
    total_nonlinear = 0
    
    def gated_forward(x):
        nonlocal total_linear, total_nonlinear
        batch, seq, hidden = x.shape
        
        with torch.no_grad():
            use_linear = gate_fn(x)  # (batch, seq) bool
        
        total_linear += use_linear.sum().item()
        total_nonlinear += (~use_linear).sum().item()
        
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
    pct_lin = total_linear / max(1, total_linear + total_nonlinear) * 100
    return ppl, pct_lin


def find_optimal_threshold(feature_vals, L_full, L_lin, n_candidates=50):
    """Find the threshold that minimizes expected loss under routing.
    feature > threshold -> use linear; else use full MLP.
    Also try feature < threshold -> linear (for inverted features).
    """
    deltas = (L_lin - L_full).numpy()
    vals = feature_vals if isinstance(feature_vals, np.ndarray) else feature_vals.numpy()
    
    percentiles = np.linspace(5, 95, n_candidates)
    candidates = np.percentile(vals, percentiles)
    
    best_loss = float('inf')
    best_threshold = None
    best_direction = None  # 'above' or 'below'
    
    for t in candidates:
        for direction in ['above', 'below']:
            if direction == 'above':
                use_linear = vals > t
            else:
                use_linear = vals < t
            
            # Expected loss: linear where routed linear, full elsewhere
            expected = np.where(use_linear, L_lin.numpy(), L_full.numpy())
            total_loss = expected.mean()
            
            if total_loss < best_loss:
                best_loss = total_loss
                best_threshold = t
                best_direction = direction
    
    # Also compute the pct that goes linear
    if best_direction == 'above':
        pct = (vals > best_threshold).mean() * 100
    else:
        pct = (vals < best_threshold).mean() * 100
    
    return best_threshold, best_direction, best_loss, pct


def main():
    print("=" * 80, flush=True)
    print("SIMPLE GEOMETRIC GATES vs LEARNED GATE", flush=True)
    print("Priority: PERPLEXITY ACCURACY", flush=True)
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
    
    layers_to_test = [2, 5, 8, 11]
    all_results = {}
    
    for layer_idx in layers_to_test:
        print(f"\n{'='*70}", flush=True)
        print(f"LAYER {layer_idx}", flush=True)
        print(f"{'='*70}", flush=True)
        
        W_lin, b_lin = fit_linear_approx(model, layer_idx)
        
        # Collect training data
        print("  Collecting training data...", flush=True)
        acts, L_full, L_lin_loss = collect_training_data(
            model, tokenizer, layer_idx, W_lin, b_lin)
        
        acts_np = acts.numpy()
        deltas = (L_lin_loss - L_full).numpy()
        
        # Compute features on training data
        print("  Computing geometric features...", flush=True)
        norms = np.linalg.norm(acts_np, axis=1)
        kurt = np.array([scipy_kurtosis(acts_np[i]) for i in range(len(acts_np))])
        max_abs = np.abs(acts_np).max(axis=1)
        
        # PCA direction
        from sklearn.decomposition import PCA
        pca = PCA(n_components=5)
        pca_projs = pca.fit_transform(acts_np)
        
        # Gate SVD direction
        # Train a quick gate to get its top direction
        gate = ActivationGate(768, 32)
        optimizer = torch.optim.Adam(gate.parameters(), lr=1e-3)
        for ep in range(300):
            perm = torch.randperm(len(acts))[:512]
            g = gate(acts[perm].unsqueeze(0)).squeeze()
            d = torch.tensor(deltas[perm.numpy()])
            loss = (g * d).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        gate.eval()
        W1 = gate.gate[0].weight.data.numpy()
        _, _, Vt = np.linalg.svd(W1, full_matrices=False)
        gate_dir = Vt[0]  # top direction
        gate_proj = acts_np @ gate_dir
        
        # Find optimal thresholds for each feature
        print("  Finding optimal thresholds...", flush=True)
        features = {
            "kurtosis": kurt,
            "norm": norms,
            "max_abs": max_abs,
            "PC0": pca_projs[:, 0],
            "PC1": pca_projs[:, 1],
            "PC2": pca_projs[:, 2],
            "gate_dir0": gate_proj,
        }
        
        thresholds = {}
        for name, vals in features.items():
            t, d, loss, pct = find_optimal_threshold(vals, L_full, L_lin_loss)
            thresholds[name] = (t, d, pct)
            print(f"    {name:>12}: threshold={t:.4f} dir={d} {pct:.1f}% linear", flush=True)
        
        # Also find optimal 3-feature logistic threshold
        from sklearn.linear_model import LogisticRegression
        # Label: 1 if linear is better
        labels = (deltas < 0).astype(int)
        feat_3 = np.column_stack([kurt, norms, max_abs])
        lr = LogisticRegression(max_iter=1000)
        lr.fit(feat_3, labels)
        lr_probs = lr.predict_proba(feat_3)[:, 1]
        t_lr, d_lr, _, pct_lr = find_optimal_threshold(lr_probs, L_full, L_lin_loss)
        thresholds["logistic_3feat"] = (t_lr, d_lr, pct_lr)
        print(f"    {'logistic_3f':>12}: threshold={t_lr:.4f} dir={d_lr} {pct_lr:.1f}% linear", flush=True)
        print(f"    Logistic weights: kurt={lr.coef_[0][0]:.4f} norm={lr.coef_[0][1]:.4f} "
              f"max_abs={lr.coef_[0][2]:.4f}", flush=True)
        
        # Now evaluate each gate by PERPLEXITY
        print(f"\n  Evaluating by perplexity...", flush=True)
        layer_results = {}
        
        # Store PCA and gate direction for eval
        pca_components = pca.components_  # (5, 768)
        
        # Define gate functions
        def make_threshold_gate(feature_name, threshold_val, direction, 
                                pca_comp=None, gate_direction=None, lr_model=None):
            def gate_fn(x):
                # x: (batch, seq, 768)
                batch, seq, hidden = x.shape
                x_flat = x.reshape(-1, hidden).cpu().numpy()
                
                if feature_name == "kurtosis":
                    vals = np.array([scipy_kurtosis(x_flat[i]) for i in range(len(x_flat))])
                elif feature_name == "norm":
                    vals = np.linalg.norm(x_flat, axis=1)
                elif feature_name == "max_abs":
                    np.abs(x_flat).max(axis=1)
                    vals = np.abs(x_flat).max(axis=1)
                elif feature_name.startswith("PC"):
                    idx = int(feature_name[2:])
                    vals = x_flat @ pca_comp[idx]
                elif feature_name == "gate_dir0":
                    vals = x_flat @ gate_direction
                elif feature_name == "logistic_3feat":
                    k = np.array([scipy_kurtosis(x_flat[i]) for i in range(len(x_flat))])
                    n = np.linalg.norm(x_flat, axis=1)
                    m = np.abs(x_flat).max(axis=1)
                    probs = lr_model.predict_proba(np.column_stack([k, n, m]))[:, 1]
                    vals = probs
                else:
                    vals = np.zeros(len(x_flat))
                
                if direction == 'above':
                    use_linear = vals > threshold_val
                else:
                    use_linear = vals < threshold_val
                
                return torch.tensor(use_linear, device=x.device).reshape(batch, seq)
            return gate_fn
        
        # All-linear baseline
        ppl_all_lin, _ = eval_ppl_with_gate_fn(
            model, tokenizer, layer_idx, W_lin, b_lin,
            lambda x: torch.ones(x.shape[0], x.shape[1], dtype=torch.bool, device=x.device))
        layer_results["all_linear"] = {"ppl": round(ppl_all_lin, 2), "pct_linear": 100.0}
        d_al = (ppl_all_lin - baseline_ppl) / baseline_ppl * 100
        print(f"  {'all_linear':>15}: ppl={ppl_all_lin:.2f} ({d_al:+.2f}%) 100.0% linear", flush=True)
        
        # Test each simple gate
        gates_to_test = [
            ("kurtosis", thresholds["kurtosis"], None, None, None),
            ("norm", thresholds["norm"], None, None, None),
            ("max_abs", thresholds["max_abs"], None, None, None),
            ("PC0", thresholds["PC0"], pca_components, None, None),
            ("PC1", thresholds["PC1"], pca_components, None, None),
            ("PC2", thresholds["PC2"], pca_components, None, None),
            ("gate_dir0", thresholds["gate_dir0"], None, gate_dir, None),
            ("logistic_3feat", thresholds["logistic_3feat"], None, None, lr),
        ]
        
        for name, (t, d, pct), pca_c, gate_d, lr_m in gates_to_test:
            t0 = time.time()
            gate_fn = make_threshold_gate(name, t, d, pca_c, gate_d, lr_m)
            ppl, actual_pct = eval_ppl_with_gate_fn(
                model, tokenizer, layer_idx, W_lin, b_lin, gate_fn)
            dt = time.time() - t0
            delta = (ppl - baseline_ppl) / baseline_ppl * 100
            layer_results[name] = {
                "ppl": round(ppl, 2), "delta_pct": round(delta, 2),
                "pct_linear": round(actual_pct, 1), "time": round(dt, 1)
            }
            print(f"  {name:>15}: ppl={ppl:.2f} ({delta:+.2f}%) {actual_pct:.1f}% linear [{dt:.1f}s]", flush=True)
        
        # Learned gate
        t0 = time.time()
        with torch.no_grad():
            gate_fn_learned = lambda x: (gate(x).squeeze(-1) > 0.5)
        ppl_learned, pct_learned = eval_ppl_with_gate_fn(
            model, tokenizer, layer_idx, W_lin, b_lin, gate_fn_learned)
        dt = time.time() - t0
        delta = (ppl_learned - baseline_ppl) / baseline_ppl * 100
        layer_results["learned_gate"] = {
            "ppl": round(ppl_learned, 2), "delta_pct": round(delta, 2),
            "pct_linear": round(pct_learned, 1), "time": round(dt, 1)
        }
        print(f"  {'learned_gate':>15}: ppl={ppl_learned:.2f} ({delta:+.2f}%) {pct_learned:.1f}% linear [{dt:.1f}s]", flush=True)
        
        # Baseline (all nonlinear)
        layer_results["baseline"] = {"ppl": round(baseline_ppl, 2), "pct_linear": 0.0}
        print(f"  {'baseline':>15}: ppl={baseline_ppl:.2f} (+0.00%) 0.0% linear", flush=True)
        
        all_results[f"layer_{layer_idx}"] = layer_results
        
        out_path = "/Users/peter/clawd/projects/sense-stack/code/simple_gates_results.json"
        with open(out_path, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"  (saved)", flush=True)
    
    # Summary
    print("\n" + "=" * 80, flush=True)
    print("SUMMARY: Simple Gates vs Learned Gate (by PPL)", flush=True)
    print("=" * 80, flush=True)
    
    for layer_idx in layers_to_test:
        key = f"layer_{layer_idx}"
        r = all_results[key]
        print(f"\nLayer {layer_idx}:", flush=True)
        
        # Sort by PPL (best first)
        entries = [(name, data) for name, data in r.items() if isinstance(data, dict) and "ppl" in data]
        entries.sort(key=lambda x: x[1]["ppl"])
        
        print(f"  {'Gate':>18} {'PPL':>8} {'Δ%':>8} {'%Lin':>7} {'Params':>8}", flush=True)
        print(f"  {'-'*55}", flush=True)
        
        param_counts = {
            "baseline": 0, "all_linear": 0,
            "kurtosis": 1, "norm": 1, "max_abs": 1,
            "PC0": 769, "PC1": 769, "PC2": 769,
            "gate_dir0": 769, "logistic_3feat": 4,
            "learned_gate": 25000,
        }
        
        for name, data in entries:
            params = param_counts.get(name, "?")
            delta = data.get("delta_pct", (data["ppl"] - baseline_ppl) / baseline_ppl * 100)
            marker = " ⭐" if name not in ["baseline", "all_linear"] and data["ppl"] <= baseline_ppl else ""
            print(f"  {name:>18} {data['ppl']:>8.2f} {delta:>+8.2f} {data['pct_linear']:>6.1f}% {params:>8}{marker}", flush=True)


if __name__ == "__main__":
    main()
