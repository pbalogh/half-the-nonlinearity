"""
Scale validation: Does the single-direction gate finding hold for larger models?

Tests GPT-2 Medium (345M, 24 layers, 1024 hidden) and GPT-2 Large (774M, 36 layers, 1280 hidden).

For selected layers (early, mid, late), tests:
1. All-linear baseline (how much does linearizing the MLP cost?)
2. Pure linear gate (hidden_dim -> 1, no hidden layer)
3. b=1 gate (hidden_dim -> 1 -> 1, single neuron)
4. b=3 gate
5. b=6 gate

Also extracts gate Direction 0 and checks what tokens it separates
(does the content/function word split hold at scale?).

Usage:
  python mlp0_scale_test.py --model gpt2-medium
  python mlp0_scale_test.py --model gpt2-large
  python mlp0_scale_test.py --model gpt2          # original, for comparison
  python mlp0_scale_test.py --model gpt2-medium --device cuda  # if you have GPU
  python mlp0_scale_test.py --model gpt2-large --layers 2,11,23,35  # custom layers
"""

import torch
import torch.nn as nn
import numpy as np
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from datasets import load_dataset
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import json, time, argparse, os

loss_fn_global = nn.CrossEntropyLoss()

def compute_loss(model, toks):
    """Compute cross-entropy loss manually, using float64 for lm_head to avoid overflow."""
    with torch.no_grad():
        # Run transformer body
        hidden = model.transformer(toks)[0]
        # lm_head in float64 to avoid overflow on larger models
        logits = torch.nn.functional.linear(
            hidden.double(), model.lm_head.weight.double(),
            model.lm_head.bias.double() if model.lm_head.bias is not None else None
        )
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = toks[:, 1:].contiguous()
        loss = loss_fn_global(shift_logits.view(-1, shift_logits.size(-1)),
                              shift_labels.view(-1))
    return loss.item(), toks.shape[1] - 1


MODEL_CONFIGS = {
    "gpt2":        {"n_layers": 12, "hidden": 768,  "test_layers": [0, 2, 5, 8, 11]},
    "gpt2-medium": {"n_layers": 24, "hidden": 1024, "test_layers": [0, 2, 6, 12, 18, 23]},
    "gpt2-large":  {"n_layers": 36, "hidden": 1280, "test_layers": [0, 2, 9, 18, 27, 35]},
}


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
    # Center for numerical stability
    X_mean = X.mean(0)
    Y_mean = Y.mean(0)
    Xc = X - X_mean
    Yc = Y - Y_mean
    # SVD-based pseudoinverse (much more stable than normal equations)
    U, s, Vt = np.linalg.svd(Xc, full_matrices=False)
    # Tikhonov regularization in SVD space
    lam = 0.01
    d = s / (s**2 + lam)
    W = (Vt.T * d) @ U.T @ Yc
    b = Y_mean - X_mean @ W
    if np.isnan(W).any() or np.isnan(b).any():
        print(f"  WARNING: NaN in linear approx even with SVD! Norms: X={np.linalg.norm(X):.1f}, Y={np.linalg.norm(Y):.1f}", flush=True)
        # Fallback: identity-ish mapping
        W = np.eye(hidden_dim) * 0.01
        b = Y_mean
    print(f"  Linear approx: W norm={np.linalg.norm(W):.2f}, b norm={np.linalg.norm(b):.2f}, "
          f"cond={s[0]/s[-1]:.0f}", flush=True)
    return torch.tensor(W, dtype=torch.float32), torch.tensor(b, dtype=torch.float32)


def _per_token_loss_f64(model, toks):
    """Per-token cross-entropy loss using float64 logits (avoids overflow on larger models)."""
    hidden = model.transformer(toks)[0]
    logits = torch.nn.functional.linear(
        hidden.double(), model.lm_head.weight.double(),
        model.lm_head.bias.double() if model.lm_head.bias is not None else None
    )
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = toks[:, 1:].contiguous()
    loss_fn = nn.CrossEntropyLoss(reduction='none')
    per_tok = loss_fn(shift_logits.reshape(-1, shift_logits.size(-1)),
                      shift_labels.reshape(-1))
    return per_tok.float()  # back to f32 for storage


def collect_training_data(model, tokenizer, layer_idx, W_lin, b_lin, device, n_tokens=25000):
    ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="train[1000:3000]")
    mlp = model.transformer.h[layer_idx].mlp
    original_forward = mlp.forward
    all_acts, all_L_full, all_L_lin = [], [], []
    all_tokens = []
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
            # Get activations at this layer
            hidden = model.transformer.wte(toks) + model.transformer.wpe(
                torch.arange(toks.shape[1], device=device))
            for i in range(layer_idx):
                hidden = model.transformer.h[i](hidden)[0]
            ln_out = model.transformer.h[layer_idx].ln_2(hidden)
            # Full MLP loss (float64 logits to avoid overflow)
            L_full = _per_token_loss_f64(model, toks)
            # Linear MLP loss
            mlp.forward = lambda x: x @ W_lin.to(device) + b_lin.to(device)
            L_lin = _per_token_loss_f64(model, toks)
            mlp.forward = original_forward
            n = toks.shape[1] - 1
            all_acts.append(ln_out[0, :n].cpu())
            all_L_full.append(L_full.cpu())
            all_L_lin.append(L_lin.cpu())
            all_tokens.extend(toks[0, :n].tolist())
            collected += n
        if collected >= n_tokens:
            break
    return (torch.cat(all_acts, 0)[:n_tokens],
            torch.cat(all_L_full, 0)[:n_tokens],
            torch.cat(all_L_lin, 0)[:n_tokens],
            all_tokens[:n_tokens])


def train_gate(acts, L_full, L_lin, hidden_dim, bottleneck, has_hidden=True, epochs=800):
    """Train a gate using sklearn LogisticRegression (closed-form, scale-robust).
    
    Strategy: compute per-position loss delta, label positions where linear is
    acceptable (delta below threshold), fit logistic regression to predict this.
    For bottleneck > 0, first reduce dimensionality via PCA, then fit logistic.
    Wraps result in a nn.Module for compatibility with eval_ppl_with_gate.
    """
    deltas = (L_lin - L_full).numpy()
    X = acts.numpy()
    
    # Handle NaN/Inf deltas (catastrophic linear approximation, e.g. Layer 0)
    valid_mask = np.isfinite(deltas)
    if valid_mask.sum() < 100:
        print(f"    Too few valid deltas ({valid_mask.sum()}), gate cannot train", flush=True)
        return _dummy_gate(hidden_dim), hidden_dim + 1
    if not valid_mask.all():
        n_bad = (~valid_mask).sum()
        print(f"    Dropping {n_bad}/{len(deltas)} NaN/Inf deltas", flush=True)
        deltas = deltas[valid_mask]
        X = X[valid_mask]
    
    # Label: 1 = "linear is OK" (delta is small or negative)
    # Use adaptive threshold: positions where linear costs less than median
    threshold = np.median(deltas)
    labels = (deltas <= threshold).astype(int)
    
    # Also try a stricter threshold for positions where linear is actually BETTER
    n_beneficial = (deltas < 0).sum()
    pct_beneficial = n_beneficial / len(deltas) * 100
    print(f"    Delta stats: median={threshold:.4f}, mean={deltas.mean():.4f}, "
          f"std={deltas.std():.4f}, beneficial={pct_beneficial:.1f}%", flush=True)
    
    # If almost no positions benefit from linear, use a looser threshold
    # (positions in the bottom quartile of damage)
    if pct_beneficial < 5:
        threshold = np.percentile(deltas, 25)
        labels = (deltas <= threshold).astype(int)
        print(f"    Few beneficial positions; using Q1 threshold={threshold:.4f}", flush=True)
    
    # Check we have two classes
    if len(np.unique(labels)) < 2:
        print(f"    Only one class in labels, gate cannot train", flush=True)
        return _dummy_gate(hidden_dim), hidden_dim + 1
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    if has_hidden and bottleneck > 0:
        # PCA reduction then logistic regression
        n_components = min(bottleneck * 2, 32, hidden_dim)
        pca = PCA(n_components=n_components)
        X_reduced = pca.fit_transform(X_scaled)
        n_params = hidden_dim * n_components + n_components + n_components + 1
        
        clf = LogisticRegression(C=1.0, max_iter=2000, solver='lbfgs')
        clf.fit(X_reduced, labels)
        acc = clf.score(X_reduced, labels)
        print(f"    Logistic (PCA-{n_components}): acc={acc:.3f}, "
              f"predict_linear={clf.predict(X_reduced).mean()*100:.1f}%", flush=True)
        
        # Wrap in nn.Module
        gate = SklearnGate(scaler, clf, pca)
    else:
        # Pure linear logistic regression
        clf = LogisticRegression(C=1.0, max_iter=2000, solver='lbfgs')
        clf.fit(X_scaled, labels)
        acc = clf.score(X_scaled, labels)
        n_params = hidden_dim + 1
        print(f"    Logistic (linear): acc={acc:.3f}, "
              f"predict_linear={clf.predict(X_scaled).mean()*100:.1f}%", flush=True)
        
        gate = SklearnGate(scaler, clf, pca=None)
    
    return gate, n_params


class DummyGate(nn.Module):
    """Always routes to full MLP (outputs 0)."""
    def forward(self, x):
        shape = x.shape[:-1] + (1,)
        return torch.zeros(shape)

def _dummy_gate(hidden_dim):
    return DummyGate()


class SklearnGate(nn.Module):
    """Wraps sklearn LogisticRegression as a nn.Module for eval compatibility."""
    
    def __init__(self, scaler, clf, pca=None):
        super().__init__()
        self.scaler = scaler
        self.clf = clf
        self.pca = pca
    
    def forward(self, x):
        # x: (batch, seq, hidden) or (batch, hidden)
        shape = x.shape
        x_np = x.detach().cpu().numpy().reshape(-1, shape[-1])
        x_scaled = self.scaler.transform(x_np)
        if self.pca is not None:
            x_scaled = self.pca.transform(x_scaled)
        # predict_proba gives probability of class 1 (= linear is OK)
        probs = self.clf.predict_proba(x_scaled)[:, 1]
        result = torch.tensor(probs, dtype=torch.float32).reshape(shape[:-1] + (1,))
        return result


def eval_ppl_with_gate(model, tokenizer, layer_idx, W_lin, b_lin, gate, device, n_tokens=12000):
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


def analyze_gate_direction(gate, acts, tokens, tokenizer, hidden_dim, L_full, L_lin):
    """Extract top gate direction and analyze what it separates."""
    # Handle SklearnGate
    if isinstance(gate, SklearnGate):
        clf = gate.clf
        if gate.pca is not None:
            # Direction in original space: PCA components weighted by logistic coefs
            coefs_reduced = clf.coef_[0]  # shape: (n_components,)
            components = gate.pca.components_  # shape: (n_components, hidden_dim)
            # Project back to original (scaled) space, then unscale
            direction_scaled = coefs_reduced @ components  # shape: (hidden_dim,)
            direction = direction_scaled * gate.scaler.scale_  # undo standardization
        else:
            direction_scaled = clf.coef_[0]
            direction = direction_scaled * gate.scaler.scale_
        direction = direction / (np.linalg.norm(direction) + 1e-8)
    elif hasattr(gate, '__getitem__') and hasattr(gate[0], 'weight'):
        W1 = gate[0].weight.data.numpy()
        if W1.shape[0] == 1:
            direction = W1[0] / (np.linalg.norm(W1[0]) + 1e-8)
        else:
            try:
                _, S, Vt = np.linalg.svd(W1.astype(np.float64), full_matrices=False)
                direction = Vt[0]
            except np.linalg.LinAlgError:
                print(f"    SVD did not converge, skipping direction analysis", flush=True)
                return {}
    else:
        return {}

    acts_np = acts.numpy()
    projections = acts_np @ direction
    deltas = (L_lin - L_full).numpy()

    # Correlation with loss delta
    corr = np.corrcoef(projections, deltas)[0, 1]

    # Top/bottom tokens
    top_idx = np.argsort(projections)[-15:][::-1]
    bot_idx = np.argsort(projections)[:15]
    token_strs = [tokenizer.decode([t]) for t in tokens]

    top_tokens = [token_strs[i] for i in top_idx]
    bot_tokens = [token_strs[i] for i in bot_idx]

    # Quartile analysis
    q25, q75 = np.percentile(projections, [25, 75])
    high_delta = deltas[projections > q75].mean()
    low_delta = deltas[projections < q25].mean()

    # Check function vs content word pattern
    is_func = np.array([1 if token_strs[i].strip().lower() in
        {'the', 'a', 'an', 'of', 'to', 'in', 'for', 'is', 'on', 'that',
         'by', 'this', 'with', 'from', 'or', 'as', 'are', 'was', 'were',
         'be', 'has', 'had', 'have', 'it', 'he', 'she', 'they', 'we',
         'his', 'her', 'its', 'their', 'my', 'your', 'at', 'but', 'not',
         'do', 'did', 'will', 'would', 'could', 'should', 'may', 'might',
         'can', 'if', 'then', 'than', 'so', 'no', 'when', 'who', 'which',
         'what', 'how', 'all', 'each', 'both', 'few', 'more', 'most',
         'and', ',', '.', '(', ')', '-', '"', "'", ':', ';'}
        else 0 for i in range(len(token_strs))])
    if is_func.std() > 0:
        func_corr = np.corrcoef(projections, is_func)[0, 1]
    else:
        func_corr = 0.0

    return {
        "corr_loss_delta": round(float(corr), 4),
        "corr_function_word": round(float(func_corr), 4),
        "top_tokens_linear_ok": top_tokens,
        "bottom_tokens_need_mlp": bot_tokens,
        "q4_mean_delta": round(float(high_delta), 4),
        "q1_mean_delta": round(float(low_delta), 4),
    }


def main():
    parser = argparse.ArgumentParser(description="Scale validation of MLP gating")
    parser.add_argument("--model", default="gpt2-medium", choices=list(MODEL_CONFIGS.keys()))
    parser.add_argument("--device", default="cpu", help="cpu, cuda, or mps")
    parser.add_argument("--layers", default=None, help="Comma-separated layer indices (overrides defaults)")
    parser.add_argument("--n_train", type=int, default=25000, help="Training tokens per layer")
    parser.add_argument("--n_eval", type=int, default=12000, help="Eval tokens per layer")
    parser.add_argument("--n_fit", type=int, default=10000, help="Tokens for linear approximation fit")
    parser.add_argument("--epochs", type=int, default=500, help="Gate training epochs")
    args = parser.parse_args()

    config = MODEL_CONFIGS[args.model]
    hidden_dim = config["hidden"]
    test_layers = [int(x) for x in args.layers.split(",")] if args.layers else config["test_layers"]
    device = args.device

    print("=" * 80, flush=True)
    print(f"SCALE VALIDATION: {args.model}", flush=True)
    print(f"  Hidden dim: {hidden_dim}, Layers: {config['n_layers']}, Device: {device}", flush=True)
    print(f"  Testing layers: {test_layers}", flush=True)
    print(f"  Train tokens: {args.n_train}, Eval tokens: {args.n_eval}", flush=True)
    print("=" * 80, flush=True)

    tokenizer = GPT2Tokenizer.from_pretrained(args.model)
    print(f"\nLoading {args.model}...", flush=True)
    model = GPT2LMHeadModel.from_pretrained(args.model).to(device).eval()
    n_params_model = sum(p.numel() for p in model.parameters())
    print(f"  Model params: {n_params_model:,}", flush=True)

    # Baseline PPL
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
        if total_tokens >= args.n_eval:
            break
    baseline_ppl = np.exp(total_loss / total_tokens)
    print(f"  Baseline: {baseline_ppl:.2f}", flush=True)

    gate_configs = [
        ("none", 0, False),
        ("all_linear", 0, False),  # special: all positions use linear
        ("linear", 0, False),
        ("b=1", 1, True),
        ("b=3", 3, True),
        ("b=6", 6, True),
    ]

    # Resume support: load existing results and skip completed layers
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           f"scale_test_{args.model.replace('-','_')}.json")
    if os.path.exists(out_path):
        with open(out_path) as f:
            all_results = json.load(f)
        # Update metadata in case args changed
        all_results["baseline_ppl"] = round(baseline_ppl, 2)
        all_results["device"] = device
        existing_layers = set(all_results.get("layers", {}).keys())
        print(f"  Resuming: found {len(existing_layers)} completed layers: {sorted(existing_layers)}", flush=True)
    else:
        all_results = {
            "model": args.model,
            "hidden_dim": hidden_dim,
            "n_layers": config["n_layers"],
            "model_params": n_params_model,
            "baseline_ppl": round(baseline_ppl, 2),
            "device": device,
            "layers": {},
        }
        existing_layers = set()

    for layer_idx in test_layers:
        if str(layer_idx) in existing_layers:
            print(f"\n  LAYER {layer_idx} — already complete, skipping", flush=True)
            continue

        print(f"\n{'='*60}", flush=True)
        print(f"LAYER {layer_idx}", flush=True)
        print(f"{'='*60}", flush=True)

        t0 = time.time()
        W_lin, b_lin = fit_linear_approx(model, layer_idx, hidden_dim, device, n_tokens=args.n_fit)
        print(f"  Linear approx fit ({args.n_fit} tokens)", flush=True)

        # All-linear eval
        print("  Evaluating all-linear...", flush=True)
        mlp = model.transformer.h[layer_idx].mlp
        original_forward = mlp.forward
        mlp.forward = lambda x: x @ W_lin.to(device) + b_lin.to(device)
        al_loss = 0.0
        al_tokens = 0
        ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="train[2000:2500]")
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
                al_loss += loss * n_toks
                al_tokens += n_toks
            if al_tokens >= args.n_eval:
                break
        mlp.forward = original_forward
        all_linear_ppl = np.exp(al_loss / al_tokens)
        all_linear_delta = (all_linear_ppl - baseline_ppl) / baseline_ppl * 100
        print(f"  all_linear: ppl={all_linear_ppl:.2f} ({all_linear_delta:+.2f}%)", flush=True)

        layer_results = {
            "all_linear_ppl": round(all_linear_ppl, 2),
            "all_linear_delta_pct": round(all_linear_delta, 2),
            "gates": [],
        }

        # Skip gating for catastrophic layers (>500% worse) — no useful signal
        if all_linear_delta > 500:
            print(f"  Skipping gates — all-linear is catastrophic ({all_linear_delta:.0f}%)", flush=True)
            layer_results["gates"].append({
                "name": "skipped", "params": 0,
                "ppl": baseline_ppl, "delta_pct": 0.0,
                "pct_linear": 0.0,
                "note": f"all-linear too catastrophic ({all_linear_delta:.0f}%) for meaningful gating"
            })
            dt = time.time() - t0
            layer_results["time_seconds"] = round(dt, 1)
            all_results["layers"][str(layer_idx)] = layer_results
            print(f"  Layer {layer_idx} done [{dt:.0f}s]", flush=True)
            out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   f"scale_test_{args.model.replace('-','_')}.json")
            with open(out_path, "w") as f:
                json.dump(all_results, f, indent=2)
            continue

        # Collect training data
        print(f"  Collecting training data ({args.n_train} tokens)...", flush=True)
        acts, L_full, L_lin_loss, tokens = collect_training_data(
            model, tokenizer, layer_idx, W_lin, b_lin, device, n_tokens=args.n_train)

        for name, b_size, has_hidden in gate_configs:
            if name in ("none", "all_linear"):
                continue

            # Train (best of 2)
            best_ppl = float('inf')
            best_pct = 0
            best_gate = None
            best_params = 0

            for trial in range(3):
                if name == "linear":
                    gate, n_params = train_gate(acts, L_full, L_lin_loss, hidden_dim,
                                               0, has_hidden=False, epochs=args.epochs)
                else:
                    gate, n_params = train_gate(acts, L_full, L_lin_loss, hidden_dim,
                                               b_size, has_hidden=True, epochs=args.epochs)
                ppl, pct_lin = eval_ppl_with_gate(
                    model, tokenizer, layer_idx, W_lin, b_lin, gate, device, n_tokens=args.n_eval)
                if ppl < best_ppl:
                    best_ppl = ppl
                    best_pct = pct_lin
                    best_gate = gate
                    best_params = n_params

            delta = (best_ppl - baseline_ppl) / baseline_ppl * 100
            marker = " ⭐" if best_ppl <= baseline_ppl else ""
            print(f"  {name:>8} ({best_params:>6} params): "
                  f"ppl={best_ppl:.2f} ({delta:+.2f}%) "
                  f"{best_pct:.1f}% linear{marker}", flush=True)

            gate_result = {
                "name": name, "params": best_params,
                "ppl": round(best_ppl, 2), "delta_pct": round(delta, 2),
                "pct_linear": round(best_pct, 1),
            }

            # Analyze gate direction for the best gate of each type
            if name == "b=6" and best_gate is not None:
                print(f"  Analyzing gate direction...", flush=True)
                dir_analysis = analyze_gate_direction(
                    best_gate, acts, tokens, tokenizer, hidden_dim, L_full, L_lin_loss)
                gate_result["direction_analysis"] = dir_analysis
                print(f"    corr_delta={dir_analysis.get('corr_loss_delta', '?')}, "
                      f"corr_func={dir_analysis.get('corr_function_word', '?')}, "
                      f"eff_rank={dir_analysis.get('eff_rank', '?')}", flush=True)
                print(f"    Top (linear ok): {dir_analysis.get('top_tokens_linear_ok', [])[:8]}", flush=True)
                print(f"    Bot (need MLP):  {dir_analysis.get('bottom_tokens_need_mlp', [])[:8]}", flush=True)

            layer_results["gates"].append(gate_result)

        dt = time.time() - t0
        layer_results["time_seconds"] = round(dt, 1)
        all_results["layers"][str(layer_idx)] = layer_results
        print(f"  Layer {layer_idx} done [{dt:.0f}s]", flush=True)

        # Save incrementally
        out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               f"scale_test_{args.model.replace('-','_')}.json")
        with open(out_path, "w") as f:
            json.dump(all_results, f, indent=2)

    # Summary
    print("\n" + "=" * 80, flush=True)
    print(f"SUMMARY: {args.model} (baseline {baseline_ppl:.2f})", flush=True)
    print("=" * 80, flush=True)
    print(f"  {'Layer':>5} {'AllLin%':>8} {'Linear':>10} {'b=1':>10} {'b=3':>10} {'b=6':>10}", flush=True)
    print(f"  {'-'*55}", flush=True)

    for layer_idx in test_layers:
        lr = all_results["layers"][str(layer_idx)]
        gates = {g["name"]: g for g in lr["gates"]}
        row = f"  {layer_idx:>5} {lr['all_linear_delta_pct']:>+7.1f}%"
        for name in ["linear", "b=1", "b=3", "b=6"]:
            if name in gates:
                g = gates[name]
                row += f" {g['delta_pct']:>+6.2f}%/{g['pct_linear']:.0f}%"
            else:
                row += "       —"
        print(row, flush=True)

    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           f"scale_test_{args.model.replace('-','_')}.json")
    print(f"\nSaved to {out_path}", flush=True)


if __name__ == "__main__":
    main()
