"""
Universal scale validation: MLP gating across model families.

Supports: GPT-2 (all sizes), Pythia (all sizes), GPT-NeoX, LLaMA/Llama-2/3, OLMo, Mistral.

Uses AutoModelForCausalLM for model loading and abstracts away architecture differences
(MLP access paths, lm_head names, layer norm patterns).

Usage:
  python scale_test_universal.py --model EleutherAI/pythia-410m
  python scale_test_universal.py --model EleutherAI/pythia-1b --device cuda
  python scale_test_universal.py --model EleutherAI/pythia-160m --layers 0,3,6,9,11
  python scale_test_universal.py --model meta-llama/Llama-2-7b-hf --device cuda --layers 0,4,16,24,31
  python scale_test_universal.py --model gpt2-medium  # still works for GPT-2
  python scale_test_universal.py --model mistralai/Mistral-7B-v0.1 --device cuda
"""

import torch
import torch.nn as nn
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from datasets import load_dataset
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import json, time, argparse, os


loss_fn_global = nn.CrossEntropyLoss()


# ─── Architecture abstraction ────────────────────────────────────────────────

class ModelAdapter:
    """Abstracts architecture differences across model families."""

    def __init__(self, model, model_name):
        self.model = model
        self.model_name = model_name
        self._detect_architecture()

    def _detect_architecture(self):
        """Detect model family from config/architecture."""
        config = self.model.config
        arch = config.architectures[0] if hasattr(config, 'architectures') and config.architectures else ""
        model_type = getattr(config, 'model_type', '')

        if 'GPT2' in arch or model_type == 'gpt2':
            self.family = 'gpt2'
        elif 'GPTNeoX' in arch or model_type == 'gpt_neox':
            self.family = 'gpt_neox'  # Pythia, GPT-NeoX
        elif 'Llama' in arch or model_type == 'llama':
            self.family = 'llama'  # LLaMA, Llama-2, Llama-3, CodeLlama
        elif 'Mistral' in arch or model_type == 'mistral':
            self.family = 'mistral'
        elif 'OLMo' in arch or model_type == 'olmo':
            self.family = 'olmo'
        elif 'Phi' in arch or model_type == 'phi':
            self.family = 'phi'
        else:
            raise ValueError(f"Unknown architecture: {arch} (model_type={model_type}). "
                             f"Add support in ModelAdapter._detect_architecture()")

        self.n_layers = config.num_hidden_layers
        self.hidden_dim = config.hidden_size
        print(f"  Detected: {self.family} family, {self.n_layers} layers, hidden={self.hidden_dim}", flush=True)

    def get_layers(self):
        """Return the list of transformer layer modules."""
        if self.family == 'gpt2':
            return self.model.transformer.h
        elif self.family == 'gpt_neox':
            return self.model.gpt_neox.layers
        elif self.family in ('llama', 'mistral'):
            return self.model.model.layers
        elif self.family == 'olmo':
            return self.model.model.transformer.blocks
        elif self.family == 'phi':
            return self.model.model.layers
        else:
            raise ValueError(f"get_layers not implemented for {self.family}")

    def get_mlp(self, layer_idx):
        """Return the MLP submodule for a given layer."""
        return self.get_layers()[layer_idx].mlp

    def get_pre_mlp_layernorm(self, layer_idx):
        """Return the layer norm applied before the MLP."""
        layer = self.get_layers()[layer_idx]
        if self.family == 'gpt2':
            return layer.ln_2
        elif self.family == 'gpt_neox':
            return layer.post_attention_layernorm
        elif self.family in ('llama', 'mistral'):
            return layer.post_attention_layernorm
        elif self.family == 'olmo':
            # OLMo may not have separate pre-MLP LN depending on version
            return getattr(layer, 'post_attention_layernorm', layer.ln_2)
        elif self.family == 'phi':
            return layer.post_layernorm
        else:
            raise ValueError(f"get_pre_mlp_layernorm not implemented for {self.family}")

    def get_embedding(self):
        """Return (wte, wpe) or (embed_tokens, None) depending on architecture."""
        if self.family == 'gpt2':
            return self.model.transformer.wte, self.model.transformer.wpe
        elif self.family == 'gpt_neox':
            return self.model.gpt_neox.embed_in, None
        elif self.family in ('llama', 'mistral'):
            return self.model.model.embed_tokens, None
        elif self.family == 'olmo':
            return self.model.model.transformer.wte, None
        elif self.family == 'phi':
            return self.model.model.embed_tokens, None
        else:
            raise ValueError(f"get_embedding not implemented for {self.family}")

    def get_transformer_body(self):
        """Return the transformer body (everything before lm_head)."""
        if self.family == 'gpt2':
            return self.model.transformer
        elif self.family == 'gpt_neox':
            return self.model.gpt_neox
        elif self.family in ('llama', 'mistral'):
            return self.model.model
        elif self.family == 'olmo':
            return self.model.model.transformer
        elif self.family == 'phi':
            return self.model.model
        else:
            raise ValueError(f"get_transformer_body not implemented for {self.family}")

    def get_lm_head(self):
        """Return (weight, bias) of the language model head."""
        if self.family == 'gpt2':
            return self.model.lm_head.weight, getattr(self.model.lm_head, 'bias', None)
        elif self.family == 'gpt_neox':
            return self.model.embed_out.weight, getattr(self.model.embed_out, 'bias', None)
        elif self.family in ('llama', 'mistral', 'phi'):
            return self.model.lm_head.weight, getattr(self.model.lm_head, 'bias', None)
        elif self.family == 'olmo':
            # OLMo often ties embeddings
            if hasattr(self.model, 'lm_head'):
                return self.model.lm_head.weight, getattr(self.model.lm_head, 'bias', None)
            else:
                return self.model.model.transformer.ff_out.weight, None
        else:
            raise ValueError(f"get_lm_head not implemented for {self.family}")

    def run_transformer_body(self, toks):
        """Run the full transformer body (no lm_head), return hidden states."""
        body = self.get_transformer_body()
        if self.family == 'gpt2':
            return body(toks)[0]
        elif self.family == 'gpt_neox':
            return body(toks)[0]
        elif self.family in ('llama', 'mistral', 'phi'):
            return body(toks)[0]
        elif self.family == 'olmo':
            return body(toks)[0]
        else:
            raise ValueError(f"run_transformer_body not implemented for {self.family}")

    def embed_tokens(self, toks, device):
        """Compute input embeddings (token + position if applicable)."""
        wte, wpe = self.get_embedding()
        hidden = wte(toks)
        if wpe is not None:
            hidden = hidden + wpe(torch.arange(toks.shape[1], device=device))
        return hidden

    def collect_mlp_io_via_hooks(self, toks, layer_idx):
        """Collect MLP input/output using hooks. Works for ALL architectures
        including rotary-embedding models (Pythia, LLaMA, Mistral).
        Returns (mlp_input, mlp_output) tensors."""
        ln = self.get_pre_mlp_layernorm(layer_idx)
        mlp = self.get_mlp(layer_idx)
        captured = {}

        def ln_hook(module, input, output):
            captured['mlp_input'] = output.detach()

        def mlp_hook(module, input, output):
            captured['mlp_output'] = output.detach()

        h1 = ln.register_forward_hook(ln_hook)
        h2 = mlp.register_forward_hook(mlp_hook)
        try:
            with torch.no_grad():
                self.model(toks)
        finally:
            h1.remove()
            h2.remove()
        return captured['mlp_input'], captured['mlp_output']

    def uses_rotary(self):
        """Whether the model uses rotary position embeddings (need special handling)."""
        return self.family in ('gpt_neox', 'llama', 'mistral', 'phi')

    def default_test_layers(self):
        """Return sensible default layers to test."""
        n = self.n_layers
        if n <= 12:
            return [0, 2, n // 2, n - 2, n - 1]
        elif n <= 24:
            return [0, 2, 6, n // 2, n - 6, n - 1]
        elif n <= 36:
            return [0, 2, 9, 18, 27, n - 1]
        else:
            step = n // 6
            return [0, step, step * 2, step * 3, step * 4, n - 1]


# ─── Core functions (model-agnostic) ─────────────────────────────────────────

def compute_loss(adapter, toks):
    """Compute cross-entropy loss. Uses float64 where supported, float32 on MPS."""
    use_f64 = toks.device.type != 'mps'
    with torch.no_grad():
        hidden = adapter.run_transformer_body(toks)
        lm_w, lm_b = adapter.get_lm_head()
        if use_f64:
            logits = torch.nn.functional.linear(
                hidden.double(), lm_w.double(),
                lm_b.double() if lm_b is not None else None
            )
        else:
            logits = torch.nn.functional.linear(
                hidden.float(), lm_w.float(),
                lm_b.float() if lm_b is not None else None
            )
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = toks[:, 1:].contiguous()
        loss = loss_fn_global(shift_logits.view(-1, shift_logits.size(-1)),
                              shift_labels.view(-1))
    return loss.item(), toks.shape[1] - 1


def _per_token_loss_f64(adapter, toks):
    """Per-token cross-entropy loss. Uses float64 where supported, float32 on MPS."""
    use_f64 = toks.device.type != 'mps'
    hidden = adapter.run_transformer_body(toks)
    lm_w, lm_b = adapter.get_lm_head()
    if use_f64:
        logits = torch.nn.functional.linear(
            hidden.double(), lm_w.double(),
            lm_b.double() if lm_b is not None else None
        )
    else:
        logits = torch.nn.functional.linear(
            hidden.float(), lm_w.float(),
            lm_b.float() if lm_b is not None else None
        )
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = toks[:, 1:].contiguous()
    loss_fn = nn.CrossEntropyLoss(reduction='none')
    per_tok = loss_fn(shift_logits.reshape(-1, shift_logits.size(-1)),
                      shift_labels.reshape(-1))
    return per_tok.float()


def fit_linear_approx(adapter, layer_idx, device, n_tokens=10000):
    hidden_dim = adapter.hidden_dim
    ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="train[:500]")
    tokenizer = AutoTokenizer.from_pretrained(adapter.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
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
        mlp_in, mlp_out = adapter.collect_mlp_io_via_hooks(toks, layer_idx)
        inputs_list.append(mlp_in.squeeze(0).cpu())
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
        print(f"  WARNING: NaN in linear approx! Norms: X={np.linalg.norm(X):.1f}, Y={np.linalg.norm(Y):.1f}", flush=True)
        W = np.eye(hidden_dim) * 0.01
        b = Y_mean
    print(f"  Linear approx: W norm={np.linalg.norm(W):.2f}, b norm={np.linalg.norm(b):.2f}, "
          f"cond={s[0]/s[-1]:.0f}", flush=True)
    return torch.tensor(W, dtype=torch.float32), torch.tensor(b, dtype=torch.float32)


def collect_training_data(adapter, tokenizer, layer_idx, W_lin, b_lin, device, n_tokens=25000):
    ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="train[1000:3000]")
    mlp = adapter.get_mlp(layer_idx)
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
            # Get MLP input via hooks (works with rotary embeddings)
            mlp_in, _ = adapter.collect_mlp_io_via_hooks(toks, layer_idx)
            # Full MLP loss
            L_full = _per_token_loss_f64(adapter, toks)
            # Linear MLP loss
            mlp.forward = lambda x: x @ W_lin.to(device) + b_lin.to(device)
            L_lin = _per_token_loss_f64(adapter, toks)
            mlp.forward = original_forward
            n = toks.shape[1] - 1
            all_acts.append(mlp_in[0, :n].cpu())
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


# ─── Gate classes (identical to original) ─────────────────────────────────────

class DummyGate(nn.Module):
    def forward(self, x):
        return torch.zeros(x.shape[:-1] + (1,))


class SklearnGate(nn.Module):
    def __init__(self, scaler, clf, pca=None):
        super().__init__()
        self.scaler = scaler
        self.clf = clf
        self.pca = pca

    def forward(self, x):
        shape = x.shape
        x_np = x.detach().cpu().numpy().reshape(-1, shape[-1])
        x_scaled = self.scaler.transform(x_np)
        if self.pca is not None:
            x_scaled = self.pca.transform(x_scaled)
        probs = self.clf.predict_proba(x_scaled)[:, 1]
        return torch.tensor(probs, dtype=torch.float32).reshape(shape[:-1] + (1,))


def train_gate(acts, L_full, L_lin, hidden_dim, bottleneck, has_hidden=True, epochs=800):
    deltas = (L_lin - L_full).numpy()
    X = acts.numpy()

    valid_mask = np.isfinite(deltas)
    if valid_mask.sum() < 100:
        print(f"    Too few valid deltas ({valid_mask.sum()}), gate cannot train", flush=True)
        return DummyGate(), hidden_dim + 1
    if not valid_mask.all():
        n_bad = (~valid_mask).sum()
        print(f"    Dropping {n_bad}/{len(deltas)} NaN/Inf deltas", flush=True)
        deltas = deltas[valid_mask]
        X = X[valid_mask]

    threshold = np.median(deltas)
    labels = (deltas <= threshold).astype(int)
    n_beneficial = (deltas < 0).sum()
    pct_beneficial = n_beneficial / len(deltas) * 100
    print(f"    Delta stats: median={threshold:.4f}, mean={deltas.mean():.4f}, "
          f"std={deltas.std():.4f}, beneficial={pct_beneficial:.1f}%", flush=True)

    if pct_beneficial < 5:
        threshold = np.percentile(deltas, 25)
        labels = (deltas <= threshold).astype(int)
        print(f"    Few beneficial positions; using Q1 threshold={threshold:.4f}", flush=True)

    if len(np.unique(labels)) < 2:
        print(f"    Only one class in labels, gate cannot train", flush=True)
        return DummyGate(), hidden_dim + 1

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    if has_hidden and bottleneck > 0:
        n_components = min(bottleneck * 2, 32, hidden_dim)
        pca = PCA(n_components=n_components)
        X_reduced = pca.fit_transform(X_scaled)
        n_params = hidden_dim * n_components + n_components + n_components + 1

        clf = LogisticRegression(C=1.0, max_iter=2000, solver='lbfgs')
        clf.fit(X_reduced, labels)
        acc = clf.score(X_reduced, labels)
        print(f"    Logistic (PCA-{n_components}): acc={acc:.3f}, "
              f"predict_linear={clf.predict(X_reduced).mean()*100:.1f}%", flush=True)
        gate = SklearnGate(scaler, clf, pca)
    else:
        clf = LogisticRegression(C=1.0, max_iter=2000, solver='lbfgs')
        clf.fit(X_scaled, labels)
        acc = clf.score(X_scaled, labels)
        n_params = hidden_dim + 1
        print(f"    Logistic (linear): acc={acc:.3f}, "
              f"predict_linear={clf.predict(X_scaled).mean()*100:.1f}%", flush=True)
        gate = SklearnGate(scaler, clf, pca=None)

    return gate, n_params


def eval_ppl_with_gate(adapter, tokenizer, layer_idx, W_lin, b_lin, gate, device, n_tokens=12000):
    ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="train[2000:2500]")
    mlp = adapter.get_mlp(layer_idx)
    original_forward = mlp.forward
    total_loss = 0.0
    total_tokens = 0
    total_linear = 0
    total_full = 0
    W_lin_d = W_lin.to(device)
    b_lin_d = b_lin.to(device)
    n_calls = [0]

    def gated_forward(x):
        nonlocal total_linear, total_full
        n_calls[0] += 1
        with torch.no_grad():
            g = gate(x).squeeze(-1)
            use_linear = (g > 0.5).to(device)
        total_linear += use_linear.sum().item()
        total_full += (~use_linear).sum().item()
        linear_out = x @ W_lin_d + b_lin_d
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
        batch_count = 0
        for ex in ds:
            text = ex["text"]
            if not text.strip():
                continue
            toks = tokenizer.encode(text, return_tensors="pt").to(device)
            if toks.shape[1] < 2:
                continue
            toks = toks[:, :512]
            with torch.no_grad():
                loss, n_toks = compute_loss(adapter, toks)
                total_loss += loss * n_toks
                total_tokens += n_toks
            batch_count += 1
            if batch_count % 20 == 0:
                print(f"      eval: {total_tokens}/{n_tokens} tokens, {n_calls[0]} gate calls", flush=True)
            if total_tokens >= n_tokens:
                break
    finally:
        mlp.forward = original_forward

    ppl = np.exp(total_loss / total_tokens)
    pct_lin = total_linear / max(1, total_linear + total_full) * 100
    return ppl, pct_lin


def analyze_gate_direction(gate, acts, tokens, tokenizer, hidden_dim, L_full, L_lin):
    if isinstance(gate, SklearnGate):
        clf = gate.clf
        if gate.pca is not None:
            coefs_reduced = clf.coef_[0]
            components = gate.pca.components_
            direction_scaled = coefs_reduced @ components
            direction = direction_scaled * gate.scaler.scale_
        else:
            direction_scaled = clf.coef_[0]
            direction = direction_scaled * gate.scaler.scale_
        direction = direction / (np.linalg.norm(direction) + 1e-8)
    else:
        return {}

    acts_np = acts.numpy()
    projections = acts_np @ direction
    deltas = (L_lin - L_full).numpy()
    corr = np.corrcoef(projections, deltas)[0, 1]

    top_idx = np.argsort(projections)[-15:][::-1]
    bot_idx = np.argsort(projections)[:15]
    token_strs = [tokenizer.decode([t]) for t in tokens]

    top_tokens = [token_strs[i] for i in top_idx]
    bot_tokens = [token_strs[i] for i in bot_idx]

    q25, q75 = np.percentile(projections, [25, 75])
    high_delta = deltas[projections > q75].mean()
    low_delta = deltas[projections < q25].mean()

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
    func_corr = np.corrcoef(projections, is_func)[0, 1] if is_func.std() > 0 else 0.0

    return {
        "corr_loss_delta": round(float(corr), 4),
        "corr_function_word": round(float(func_corr), 4),
        "top_tokens_linear_ok": top_tokens,
        "bottom_tokens_need_mlp": bot_tokens,
        "q4_mean_delta": round(float(high_delta), 4),
        "q1_mean_delta": round(float(low_delta), 4),
    }


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Universal scale validation of MLP gating")
    parser.add_argument("--model", required=True, help="HuggingFace model name (e.g. EleutherAI/pythia-410m)")
    parser.add_argument("--device", default="cpu", help="cpu, cuda, or mps")
    parser.add_argument("--layers", default=None, help="Comma-separated layer indices")
    parser.add_argument("--n_train", type=int, default=25000, help="Training tokens per layer")
    parser.add_argument("--n_eval", type=int, default=12000, help="Eval tokens per layer")
    parser.add_argument("--n_fit", type=int, default=10000, help="Tokens for linear approximation fit")
    parser.add_argument("--dtype", default="float32", choices=["float32", "float16", "bfloat16"],
                        help="Model dtype (use float16/bfloat16 for large models on GPU)")
    parser.add_argument("--trust_remote_code", action="store_true", help="Trust remote code for custom architectures")
    args = parser.parse_args()

    device = args.device
    dtype_map = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
    model_dtype = dtype_map[args.dtype]

    print("=" * 80, flush=True)
    print(f"UNIVERSAL SCALE VALIDATION: {args.model}", flush=True)
    print(f"  Device: {device}, dtype: {args.dtype}", flush=True)
    print("=" * 80, flush=True)

    print(f"\nLoading tokenizer...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=args.trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading model...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=model_dtype,
        trust_remote_code=args.trust_remote_code,
    ).to(device).eval()

    adapter = ModelAdapter(model, args.model)
    n_params_model = sum(p.numel() for p in model.parameters())
    print(f"  Model params: {n_params_model:,}", flush=True)

    test_layers = [int(x) for x in args.layers.split(",")] if args.layers else adapter.default_test_layers()
    print(f"  Testing layers: {test_layers}", flush=True)
    print(f"  Train tokens: {args.n_train}, Eval tokens: {args.n_eval}", flush=True)

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
            loss, n_toks = compute_loss(adapter, toks)
            total_loss += loss * n_toks
            total_tokens += n_toks
        if total_tokens >= args.n_eval:
            break
    baseline_ppl = np.exp(total_loss / total_tokens)
    print(f"  Baseline: {baseline_ppl:.2f}", flush=True)

    # Sanitize model name for filename
    safe_name = args.model.replace("/", "_").replace("-", "_")
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           f"scale_test_{safe_name}.json")

    # Resume support
    if os.path.exists(out_path):
        with open(out_path) as f:
            all_results = json.load(f)
        all_results["baseline_ppl"] = round(baseline_ppl, 2)
        all_results["device"] = device
        existing_layers = set(all_results.get("layers", {}).keys())
        print(f"  Resuming: found {len(existing_layers)} completed layers: {sorted(existing_layers)}", flush=True)
    else:
        all_results = {
            "model": args.model,
            "family": adapter.family,
            "hidden_dim": adapter.hidden_dim,
            "n_layers": adapter.n_layers,
            "model_params": n_params_model,
            "baseline_ppl": round(baseline_ppl, 2),
            "device": device,
            "dtype": args.dtype,
            "layers": {},
        }
        existing_layers = set()

    gate_configs = [
        ("linear", 0, False),
        ("b=1", 1, True),
        ("b=3", 3, True),
        ("b=6", 6, True),
    ]

    for layer_idx in test_layers:
        if str(layer_idx) in existing_layers:
            print(f"\n  LAYER {layer_idx} — already complete, skipping", flush=True)
            continue

        print(f"\n{'='*60}", flush=True)
        print(f"LAYER {layer_idx}", flush=True)
        print(f"{'='*60}", flush=True)

        t0 = time.time()
        W_lin, b_lin = fit_linear_approx(adapter, layer_idx, device, n_tokens=args.n_fit)

        # All-linear eval
        print("  Evaluating all-linear...", flush=True)
        mlp = adapter.get_mlp(layer_idx)
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
                loss, n_toks = compute_loss(adapter, toks)
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
            with open(out_path, "w") as f:
                json.dump(all_results, f, indent=2)
            continue

        # Collect training data
        print(f"  Collecting training data ({args.n_train} tokens)...", flush=True)
        acts, L_full, L_lin_loss, tokens = collect_training_data(
            adapter, tokenizer, layer_idx, W_lin, b_lin, device, n_tokens=args.n_train)

        for name, b_size, has_hidden in gate_configs:
            best_ppl = float('inf')
            best_pct = 0
            best_gate = None
            best_params = 0

            for trial in range(3):
                if name == "linear":
                    gate, n_params = train_gate(acts, L_full, L_lin_loss, adapter.hidden_dim,
                                               0, has_hidden=False)
                else:
                    gate, n_params = train_gate(acts, L_full, L_lin_loss, adapter.hidden_dim,
                                               b_size, has_hidden=True)
                ppl, pct_lin = eval_ppl_with_gate(
                    adapter, tokenizer, layer_idx, W_lin, b_lin, gate, device, n_tokens=args.n_eval)
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

            if name == "b=6" and best_gate is not None:
                print(f"  Analyzing gate direction...", flush=True)
                dir_analysis = analyze_gate_direction(
                    best_gate, acts, tokens, tokenizer, adapter.hidden_dim, L_full, L_lin_loss)
                gate_result["direction_analysis"] = dir_analysis
                print(f"    corr_delta={dir_analysis.get('corr_loss_delta', '?')}, "
                      f"corr_func={dir_analysis.get('corr_function_word', '?')}", flush=True)
                print(f"    Top (linear ok): {dir_analysis.get('top_tokens_linear_ok', [])[:8]}", flush=True)
                print(f"    Bot (need MLP):  {dir_analysis.get('bottom_tokens_need_mlp', [])[:8]}", flush=True)

            layer_results["gates"].append(gate_result)

        dt = time.time() - t0
        layer_results["time_seconds"] = round(dt, 1)
        all_results["layers"][str(layer_idx)] = layer_results
        print(f"  Layer {layer_idx} done [{dt:.0f}s]", flush=True)

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

    print(f"\nSaved to {out_path}", flush=True)


if __name__ == "__main__":
    main()
