"""
Interpret the trained activation gate: what geometric property of the
hidden state separates "linear-ok" from "needs-MLP"?

Approaches:
1. Gate weight analysis — what directions in 768-d space does the gate attend to?
2. Activation projection — project linear vs nonlinear activations onto gate's learned directions
3. Geometric features — norm, variance, kurtosis, distance from cluster centers
4. Residual stream decomposition — how much comes from embedding vs attention?
5. Cosine similarity to sense prototypes — connection to polysemy work
"""

import torch
import torch.nn as nn
import numpy as np
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from datasets import load_dataset
from collections import defaultdict
import json, time, os
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

device = "cpu"


class ActivationGate(nn.Module):
    def __init__(self, hidden_dim=768, bottleneck=32):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim, bottleneck),
            nn.ReLU(),
            nn.Linear(bottleneck, bottleneck),
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


def collect_gate_data(model, tokenizer, layer_idx, W_lin, b_lin, n_tokens=10000):
    """Collect activations, gate inputs, and per-position loss deltas."""
    ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="train[1000:1800]")
    mlp = model.transformer.h[layer_idx].mlp
    original_forward = mlp.forward
    loss_fn = nn.CrossEntropyLoss(reduction='none')
    
    all_activations = []  # post-LN, pre-MLP activations
    all_loss_deltas = []  # L_linear - L_full (positive = MLP helps)
    all_token_ids = []
    all_positions = []
    all_pre_attn = []  # activation before attention (just embed+pos)
    all_attn_contrib = []  # attention's contribution
    collected = 0
    
    for ex in ds:
        text = ex["text"]
        if not text.strip(): continue
        toks = tokenizer.encode(text, return_tensors="pt").to(device)
        if toks.shape[1] < 4: continue
        toks = toks[:, :256]
        
        with torch.no_grad():
            # Track residual stream components
            embed = model.transformer.wte(toks) + model.transformer.wpe(
                torch.arange(toks.shape[1], device=device))
            
            # Run through layers before target, tracking what attention adds
            hidden = embed.clone()
            for i in range(layer_idx):
                hidden = model.transformer.h[i](hidden)[0]
            
            # Now at target layer: get pre-attention state
            pre_attn = hidden.clone()
            
            # Run attention
            attn_out = model.transformer.h[layer_idx].attn(
                model.transformer.h[layer_idx].ln_1(hidden))[0]
            post_attn = hidden + attn_out  # residual connection
            
            # LN2 -> MLP input
            ln_out = model.transformer.h[layer_idx].ln_2(post_attn)
            
            # Full model loss
            out_full = model(toks)
            logits_full = out_full.logits[:, :-1, :]
            targets = toks[:, 1:]
            loss_full = loss_fn(logits_full.reshape(-1, logits_full.size(-1)),
                              targets.reshape(-1))
            
            # Linear replacement loss
            mlp.forward = lambda x: x @ W_lin.to(device) + b_lin.to(device)
            out_lin = model(toks)
            logits_lin = out_lin.logits[:, :-1, :]
            loss_lin = loss_fn(logits_lin.reshape(-1, logits_lin.size(-1)),
                             targets.reshape(-1))
            mlp.forward = original_forward
            
            seq_len = toks.shape[1] - 1
            all_activations.append(ln_out[0, :seq_len].cpu())
            all_loss_deltas.append((loss_lin - loss_full).cpu())
            all_token_ids.append(toks[0, :seq_len].cpu())
            all_positions.append(torch.arange(seq_len))
            all_pre_attn.append(pre_attn[0, :seq_len].cpu())
            all_attn_contrib.append(attn_out[0, :seq_len].cpu())
            collected += seq_len
        
        if collected >= n_tokens:
            break
    
    acts = torch.cat(all_activations, dim=0)[:n_tokens]
    deltas = torch.cat(all_loss_deltas, dim=0)[:n_tokens]
    tids = torch.cat(all_token_ids, dim=0)[:n_tokens]
    positions = torch.cat(all_positions, dim=0)[:n_tokens]
    pre_attn = torch.cat(all_pre_attn, dim=0)[:n_tokens]
    attn_contrib = torch.cat(all_attn_contrib, dim=0)[:n_tokens]
    
    return acts, deltas, tids, positions, pre_attn, attn_contrib


def train_gate_for_analysis(acts, deltas, sparsity=0.0, n_epochs=500):
    """Train gate and return it for analysis."""
    L_full_proxy = torch.zeros_like(deltas)  # we only need deltas
    L_lin_proxy = deltas.clone()
    
    n = len(acts)
    idx = torch.randperm(n)
    split = int(0.8 * n)
    train_idx, val_idx = idx[:split], idx[split:]
    
    gate = ActivationGate(acts.shape[1], bottleneck=32)
    optimizer = torch.optim.Adam(gate.parameters(), lr=1e-3)
    
    for epoch in range(n_epochs):
        gate.train()
        perm = torch.randperm(len(train_idx))[:512]
        batch_idx = train_idx[perm]
        
        x = acts[batch_idx]
        d = deltas[batch_idx]
        
        g = gate(x.unsqueeze(0)).squeeze()  # prob of using linear
        # Loss: when g=1 (linear), pay d (the delta); when g=0 (full), pay 0
        expected_loss = g * d - sparsity * g.mean()
        
        total_loss = expected_loss.mean()
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
    
    gate.eval()
    return gate


def analyze_gate_geometry(gate, acts, deltas, tids, positions, pre_attn, attn_contrib, 
                          tokenizer, token_freqs, layer_idx):
    """Deep analysis of what the gate has learned."""
    results = {}
    
    with torch.no_grad():
        g = gate(acts.unsqueeze(0)).squeeze()
        use_linear = (g > 0.5).numpy()
        g_vals = g.numpy()
    
    acts_np = acts.numpy()
    deltas_np = deltas.numpy()
    linear_acts = acts_np[use_linear]
    nonlinear_acts = acts_np[~use_linear]
    
    n_lin = use_linear.sum()
    n_nl = (~use_linear).sum()
    print(f"\n  Routing split: {n_lin} linear ({n_lin/len(use_linear)*100:.1f}%), "
          f"{n_nl} nonlinear ({n_nl/len(use_linear)*100:.1f}%)", flush=True)
    
    # ===== 1. GATE WEIGHT ANALYSIS =====
    print(f"\n  === 1. GATE WEIGHT STRUCTURE ===", flush=True)
    
    W1 = gate.gate[0].weight.data.numpy()  # (32, 768)
    b1 = gate.gate[0].bias.data.numpy()
    
    # SVD of first layer weights
    U, S, Vt = np.linalg.svd(W1, full_matrices=False)
    print(f"  First layer weight singular values (top 10): {S[:10].round(3)}", flush=True)
    print(f"  Effective rank (90% variance): {(np.cumsum(S**2)/np.sum(S**2) < 0.9).sum() + 1}", flush=True)
    
    # Top principal directions the gate attends to
    top_directions = Vt[:5]  # top 5 directions in 768-d space
    
    # Project all activations onto these directions
    projs = acts_np @ top_directions.T  # (n, 5)
    print(f"\n  Projections onto gate's top 5 directions:", flush=True)
    print(f"  {'Dir':>4} {'Lin mean':>10} {'NL mean':>10} {'Diff':>10} {'t-stat':>10}", flush=True)
    for i in range(5):
        lin_mean = projs[use_linear, i].mean()
        nl_mean = projs[~use_linear, i].mean()
        # Welch's t-test
        lin_std = projs[use_linear, i].std()
        nl_std = projs[~use_linear, i].std()
        t_stat = (lin_mean - nl_mean) / np.sqrt(lin_std**2/n_lin + nl_std**2/n_nl)
        print(f"  {i:>4} {lin_mean:>10.4f} {nl_mean:>10.4f} {lin_mean-nl_mean:>10.4f} {t_stat:>10.2f}", flush=True)
    
    results["gate_effective_rank"] = int((np.cumsum(S**2)/np.sum(S**2) < 0.9).sum() + 1)
    
    # ===== 2. GEOMETRIC FEATURES =====
    print(f"\n  === 2. GEOMETRIC PROPERTIES ===", flush=True)
    
    metrics = {}
    
    # Activation norm
    norms = np.linalg.norm(acts_np, axis=1)
    metrics["norm"] = (norms[use_linear].mean(), norms[~use_linear].mean())
    
    # Per-dimension variance (how "spread" the activation is)
    dim_vars = acts_np ** 2  # variance proxy per dim
    act_entropy = -np.sum(dim_vars / dim_vars.sum(axis=1, keepdims=True) * 
                          np.log(dim_vars / dim_vars.sum(axis=1, keepdims=True) + 1e-10), axis=1)
    metrics["activation_entropy"] = (act_entropy[use_linear].mean(), act_entropy[~use_linear].mean())
    
    # Kurtosis (peakedness)
    from scipy.stats import kurtosis
    kurt = np.array([kurtosis(acts_np[i]) for i in range(len(acts_np))])
    metrics["kurtosis"] = (kurt[use_linear].mean(), kurt[~use_linear].mean())
    
    # Distance from mean activation
    mean_act = acts_np.mean(axis=0)
    dists = np.linalg.norm(acts_np - mean_act, axis=1)
    metrics["dist_from_mean"] = (dists[use_linear].mean(), dists[~use_linear].mean())
    
    # Max absolute activation (spikiness)
    max_abs = np.abs(acts_np).max(axis=1)
    metrics["max_abs_activation"] = (max_abs[use_linear].mean(), max_abs[~use_linear].mean())
    
    # Number of "active" dimensions (> 1 std above mean)
    threshold = acts_np.mean() + acts_np.std()
    n_active = (acts_np > threshold).sum(axis=1)
    metrics["n_active_dims"] = (n_active[use_linear].mean(), n_active[~use_linear].mean())
    
    print(f"  {'Metric':>25} {'Linear':>12} {'Nonlinear':>12} {'Ratio':>8}", flush=True)
    print(f"  {'-'*60}", flush=True)
    for name, (lin_val, nl_val) in metrics.items():
        ratio = lin_val / nl_val if nl_val != 0 else float('inf')
        print(f"  {name:>25} {lin_val:>12.4f} {nl_val:>12.4f} {ratio:>8.3f}", flush=True)
    
    results["geometric_metrics"] = {k: {"linear": round(v[0], 4), "nonlinear": round(v[1], 4)} 
                                     for k, v in metrics.items()}
    
    # ===== 3. ATTENTION CONTRIBUTION ANALYSIS =====
    print(f"\n  === 3. ATTENTION vs EMBEDDING CONTRIBUTION ===", flush=True)
    
    pre_attn_np = pre_attn.numpy()
    attn_np = attn_contrib.numpy()
    
    # How much of the activation comes from attention vs the residual?
    attn_norms = np.linalg.norm(attn_np, axis=1)
    pre_norms = np.linalg.norm(pre_attn_np, axis=1)
    attn_ratio = attn_norms / (attn_norms + pre_norms + 1e-10)
    
    print(f"  Attention contribution ratio: linear={attn_ratio[use_linear].mean():.4f} "
          f"nonlinear={attn_ratio[~use_linear].mean():.4f}", flush=True)
    
    # Cosine between activation and attention contribution
    cos_act_attn = np.array([
        np.dot(acts_np[i], attn_np[i]) / (np.linalg.norm(acts_np[i]) * np.linalg.norm(attn_np[i]) + 1e-10)
        for i in range(len(acts_np))])
    print(f"  Cos(activation, attn_contrib): linear={cos_act_attn[use_linear].mean():.4f} "
          f"nonlinear={cos_act_attn[~use_linear].mean():.4f}", flush=True)
    
    # Alignment with attention direction
    attn_dir = attn_np / (np.linalg.norm(attn_np, axis=1, keepdims=True) + 1e-10)
    proj_on_attn = np.sum(acts_np * attn_dir, axis=1)
    print(f"  Projection on attn direction: linear={proj_on_attn[use_linear].mean():.4f} "
          f"nonlinear={proj_on_attn[~use_linear].mean():.4f}", flush=True)
    
    results["attn_contrib_ratio"] = {
        "linear": round(attn_ratio[use_linear].mean(), 4),
        "nonlinear": round(attn_ratio[~use_linear].mean(), 4)
    }
    
    # ===== 4. PCA STRUCTURE =====
    print(f"\n  === 4. PCA STRUCTURE ===", flush=True)
    
    pca = PCA(n_components=20)
    pca_acts = pca.fit_transform(acts_np)
    
    print(f"  Variance explained (first 10 PCs): {pca.explained_variance_ratio_[:10].round(4)}", flush=True)
    
    # Which PCs separate linear from nonlinear?
    print(f"\n  PC separability (t-stat):", flush=True)
    for i in range(10):
        lin_mean = pca_acts[use_linear, i].mean()
        nl_mean = pca_acts[~use_linear, i].mean()
        lin_std = pca_acts[use_linear, i].std()
        nl_std = pca_acts[~use_linear, i].std()
        t = (lin_mean - nl_mean) / np.sqrt(lin_std**2/n_lin + nl_std**2/n_nl)
        if abs(t) > 2:
            print(f"    PC{i}: t={t:.2f} (lin={lin_mean:.4f}, nl={nl_mean:.4f}) ***", flush=True)
        else:
            print(f"    PC{i}: t={t:.2f}", flush=True)
    
    # ===== 5. CLUSTER ANALYSIS =====
    print(f"\n  === 5. ACTIVATION CLUSTERS ===", flush=True)
    
    kmeans = KMeans(n_clusters=8, n_init=5, random_state=42)
    clusters = kmeans.fit_predict(pca_acts[:, :10])
    
    print(f"  {'Cluster':>8} {'Size':>6} {'%Linear':>10} {'MeanDelta':>12} {'MeanNorm':>10}", flush=True)
    for c in range(8):
        mask = clusters == c
        if mask.sum() == 0: continue
        pct_lin = use_linear[mask].mean() * 100
        mean_delta = deltas_np[mask].mean()
        mean_norm = norms[mask].mean()
        print(f"  {c:>8} {mask.sum():>6} {pct_lin:>9.1f}% {mean_delta:>12.4f} {mean_norm:>10.2f}", flush=True)
    
    # ===== 6. CORRELATION WITH LOSS DELTA =====
    print(f"\n  === 6. WHAT PREDICTS LOSS DELTA? ===", flush=True)
    
    features = {
        "norm": norms,
        "kurtosis": kurt,
        "dist_from_mean": dists,
        "max_abs": max_abs,
        "n_active": n_active.astype(float),
        "attn_ratio": attn_ratio,
        "attn_norm": attn_norms,
        "position": positions.numpy().astype(float),
        "gate_score": g_vals,
    }
    
    # Add top PCs
    for i in range(5):
        features[f"PC{i}"] = pca_acts[:, i]
    
    # Add projections on gate directions
    for i in range(3):
        features[f"gate_dir_{i}"] = projs[:, i]
    
    print(f"  {'Feature':>20} {'Corr w/ delta':>15} {'Corr w/ gate':>15}", flush=True)
    print(f"  {'-'*55}", flush=True)
    sorted_feats = []
    for name, vals in features.items():
        r_delta = np.corrcoef(vals, deltas_np)[0, 1]
        r_gate = np.corrcoef(vals, g_vals)[0, 1]
        sorted_feats.append((name, r_delta, r_gate))
    
    sorted_feats.sort(key=lambda x: -abs(x[1]))
    for name, r_delta, r_gate in sorted_feats:
        marker = " ***" if abs(r_delta) > 0.1 else ""
        print(f"  {name:>20} {r_delta:>+15.4f} {r_gate:>+15.4f}{marker}", flush=True)
    
    results["correlations"] = {name: {"delta": round(rd, 4), "gate": round(rg, 4)} 
                                for name, rd, rg in sorted_feats}
    
    # ===== 7. INTERPRETABLE DIRECTION =====
    print(f"\n  === 7. THE GATE'S DECISION BOUNDARY ===", flush=True)
    
    # Fit a logistic regression on top geometric features to approximate gate
    from sklearn.linear_model import LogisticRegression
    
    feat_matrix = np.column_stack([
        norms, kurt, dists, max_abs, n_active.astype(float),
        attn_ratio, attn_norms, positions.numpy().astype(float),
        pca_acts[:, :5]
    ])
    feat_names = ["norm", "kurtosis", "dist_from_mean", "max_abs", "n_active",
                  "attn_ratio", "attn_norm", "position", "PC0", "PC1", "PC2", "PC3", "PC4"]
    
    lr = LogisticRegression(max_iter=1000, C=1.0)
    lr.fit(feat_matrix, use_linear.astype(int))
    lr_acc = lr.score(feat_matrix, use_linear.astype(int))
    
    print(f"  Logistic regression accuracy (approx gate): {lr_acc:.4f}", flush=True)
    print(f"\n  Feature weights (what predicts 'linear ok'):", flush=True)
    coefs = list(zip(feat_names, lr.coef_[0]))
    coefs.sort(key=lambda x: -abs(x[1]))
    for name, w in coefs:
        marker = " ***" if abs(w) > 0.1 else ""
        print(f"    {name:>20}: {w:>+8.4f}{marker}", flush=True)
    
    results["logistic_approx"] = {
        "accuracy": round(lr_acc, 4),
        "weights": {name: round(w, 4) for name, w in coefs}
    }
    
    return results


def load_token_frequencies(tokenizer):
    cache_path = "/Users/peter/clawd/projects/sense-stack/code/token_freqs.npy"
    if os.path.exists(cache_path):
        return np.load(cache_path)
    return np.zeros(50257)


def main():
    print("=" * 80, flush=True)
    print("GATE INTERPRETATION: What Geometry Separates Linear from Nonlinear?", flush=True)
    print("=" * 80, flush=True)
    
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2").to(device).eval()
    token_freqs = load_token_frequencies(tokenizer)
    
    layers_to_test = [2, 5, 8, 11]  # Skip 0 (untouchable) and 1 (minimal)
    all_results = {}
    
    for layer_idx in layers_to_test:
        print(f"\n{'='*70}", flush=True)
        print(f"LAYER {layer_idx}", flush=True)
        print(f"{'='*70}", flush=True)
        
        # 1. Fit linear approx
        W_lin, b_lin = fit_linear_approx(model, layer_idx)
        
        # 2. Collect data with decomposition
        print("  Collecting activation data with residual decomposition...", flush=True)
        acts, deltas, tids, positions, pre_attn, attn_contrib = \
            collect_gate_data(model, tokenizer, layer_idx, W_lin, b_lin)
        
        print(f"  Loss delta: mean={deltas.mean():.4f} std={deltas.std():.4f}", flush=True)
        print(f"  % where linear is better: {(deltas < 0).float().mean()*100:.1f}%", flush=True)
        
        # 3. Train fresh gate
        print("  Training gate...", flush=True)
        gate = train_gate_for_analysis(acts, deltas)
        
        # 4. Analyze
        results = analyze_gate_geometry(
            gate, acts, deltas, tids, positions, pre_attn, attn_contrib,
            tokenizer, token_freqs, layer_idx)
        
        all_results[f"layer_{layer_idx}"] = results
        
        # Save incrementally
        out_path = "/Users/peter/clawd/projects/sense-stack/code/gate_interpret_results.json"
        with open(out_path, "w") as f:
            json.dump(all_results, f, indent=2, default=str)
        print(f"\n  (saved)", flush=True)
    
    # Cross-layer comparison
    print("\n" + "=" * 80, flush=True)
    print("CROSS-LAYER COMPARISON", flush=True)
    print("=" * 80, flush=True)
    
    for layer_key in sorted(all_results.keys()):
        r = all_results[layer_key]
        print(f"\n{layer_key}:", flush=True)
        print(f"  Gate effective rank: {r.get('gate_effective_rank', '?')}", flush=True)
        print(f"  Logistic approx accuracy: {r.get('logistic_approx', {}).get('accuracy', '?')}", flush=True)
        
        # Top 3 logistic weights
        weights = r.get('logistic_approx', {}).get('weights', {})
        top3 = sorted(weights.items(), key=lambda x: -abs(x[1]))[:3]
        print(f"  Top features: {[(n, round(w, 3)) for n, w in top3]}", flush=True)
        
        # Attention contribution
        ac = r.get('attn_contrib_ratio', {})
        print(f"  Attn ratio: lin={ac.get('linear', '?')} nl={ac.get('nonlinear', '?')}", flush=True)


if __name__ == "__main__":
    main()
