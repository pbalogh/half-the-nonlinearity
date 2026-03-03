"""
What do the 9 dimensions of the gate's decision boundary encode?

For each layer, train a gate (768->32->1), extract the 32->1 bottleneck,
SVD the 768->32 projection to get the 9 effective directions, then:

1. Project known linguistic probes onto these directions
   - Part of speech, position, frequency, capitalization, punctuation context
2. Find the top-activating tokens for each direction (what maximizes it?)
3. Correlate each direction with the linear→nonlinear loss delta
4. Check if directions correspond to known transformer circuits
   (induction heads, copy, etc.) by correlating with attention patterns
5. Visualize: for each direction, show the tokens that score highest/lowest
"""

import torch
import torch.nn as nn
import numpy as np
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from datasets import load_dataset
from scipy.stats import kurtosis as scipy_kurtosis
import json, time

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


def collect_data(model, tokenizer, layer_idx, W_lin, b_lin, n_tokens=10000):
    """Collect activations, tokens, positions, and per-position loss deltas."""
    ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="train[1000:1500]")
    mlp = model.transformer.h[layer_idx].mlp
    original_forward = mlp.forward
    loss_fn = nn.CrossEntropyLoss(reduction='none')

    all_acts, all_tokens, all_positions = [], [], []
    all_L_full, all_L_lin = [], []
    all_prev_tokens, all_next_tokens = [], []
    collected = 0

    for ex in ds:
        text = ex["text"]
        if not text.strip(): continue
        toks = tokenizer.encode(text, return_tensors="pt").to(device)
        if toks.shape[1] < 4: continue
        toks = toks[:, :256]
        seq_len = toks.shape[1]

        with torch.no_grad():
            hidden = model.transformer.wte(toks) + model.transformer.wpe(
                torch.arange(seq_len, device=device))
            for i in range(layer_idx):
                hidden = model.transformer.h[i](hidden)[0]
            ln_out = model.transformer.h[layer_idx].ln_2(hidden)

            out_full = model(toks)
            logits_full = out_full.logits[:, :-1, :]
            targets = toks[:, 1:]
            L_full = loss_fn(logits_full.reshape(-1, logits_full.size(-1)),
                           targets.reshape(-1))

            mlp.forward = lambda x: x @ W_lin.to(device) + b_lin.to(device)
            out_lin = model(toks)
            logits_lin = out_lin.logits[:, :-1, :]
            L_lin = loss_fn(logits_lin.reshape(-1, logits_lin.size(-1)),
                          targets.reshape(-1))
            mlp.forward = original_forward

            n = seq_len - 1
            all_acts.append(ln_out[0, :n].cpu())
            all_tokens.extend(toks[0, :n].tolist())
            all_positions.extend(list(range(n)))
            all_prev_tokens.extend([0] + toks[0, :n-1].tolist())
            all_next_tokens.extend(toks[0, 1:n+1].tolist())
            all_L_full.append(L_full.cpu())
            all_L_lin.append(L_lin.cpu())
            collected += n

        if collected >= n_tokens: break

    acts = torch.cat(all_acts, dim=0)[:n_tokens]
    L_full = torch.cat(all_L_full, dim=0)[:n_tokens]
    L_lin = torch.cat(all_L_lin, dim=0)[:n_tokens]
    tokens = all_tokens[:n_tokens]
    positions = all_positions[:n_tokens]
    prev_tokens = all_prev_tokens[:n_tokens]
    next_tokens = all_next_tokens[:n_tokens]

    return acts, L_full, L_lin, tokens, positions, prev_tokens, next_tokens


def train_gate(acts, L_full, L_lin, epochs=500):
    deltas = (L_lin - L_full)
    gate = ActivationGate(768, 32)
    optimizer = torch.optim.Adam(gate.parameters(), lr=1e-3)
    for ep in range(epochs):
        perm = torch.randperm(len(acts))[:512]
        g = gate(acts[perm].unsqueeze(0)).squeeze()
        d = deltas[perm]
        # Route: g=1 means use linear, g=0 means use full
        # Want g=1 where delta<0 (linear is better)
        # Loss = E[g * delta] + sparsity
        loss = (g * d).mean() + 0.01 * g.mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    gate.eval()
    return gate


def analyze_gate_dimensions(model, gate, acts, L_full, L_lin, tokens, positions,
                            prev_tokens, next_tokens, tokenizer, layer_idx):
    """Extract and interpret the gate's effective dimensions."""
    results = {}

    # Extract gate weights
    W1 = gate.gate[0].weight.data.numpy()  # (32, 768)
    b1 = gate.gate[0].bias.data.numpy()    # (32,)
    W2 = gate.gate[2].weight.data.numpy()  # (1, 32)
    b2 = gate.gate[2].bias.data.numpy()    # (1,)

    # SVD of W1 to get effective directions
    U, S, Vt = np.linalg.svd(W1, full_matrices=False)
    # Effective rank (90% energy)
    energy = np.cumsum(S**2) / np.sum(S**2)
    eff_rank = int(np.searchsorted(energy, 0.90)) + 1
    print(f"  Effective rank: {eff_rank} (singular values: {S[:12].round(2)})", flush=True)
    results["effective_rank"] = eff_rank
    results["singular_values"] = S[:16].tolist()

    # The effective directions in 768-d space
    directions = Vt[:eff_rank]  # (eff_rank, 768)

    # Project all activations onto these directions
    acts_np = acts.numpy()
    projections = acts_np @ directions.T  # (N, eff_rank)

    deltas = (L_lin - L_full).numpy()
    token_strs = [tokenizer.decode([t]) for t in tokens]

    # Load token frequencies
    freq_path = "/Users/peter/clawd/projects/sense-stack/code/token_freqs.npy"
    try:
        token_freqs = np.load(freq_path)
    except:
        token_freqs = np.ones(50257)

    results["directions"] = []

    for d_idx in range(min(eff_rank, 9)):
        proj = projections[:, d_idx]
        dir_info = {"dim": d_idx, "singular_value": float(S[d_idx])}

        # 1. Correlation with loss delta
        corr = np.corrcoef(proj, deltas)[0, 1]
        dir_info["corr_with_loss_delta"] = round(float(corr), 4)
        print(f"\n  --- Direction {d_idx} (σ={S[d_idx]:.2f}, corr_delta={corr:.3f}) ---", flush=True)

        # 2. Top/bottom tokens
        top_idx = np.argsort(proj)[-20:][::-1]
        bot_idx = np.argsort(proj)[:20]

        top_tokens = [(token_strs[i], round(float(proj[i]), 3),
                       round(float(deltas[i]), 3)) for i in top_idx]
        bot_tokens = [(token_strs[i], round(float(proj[i]), 3),
                       round(float(deltas[i]), 3)) for i in bot_idx]

        dir_info["top_tokens"] = [{"token": t, "proj": p, "delta": d} for t,p,d in top_tokens]
        dir_info["bottom_tokens"] = [{"token": t, "proj": p, "delta": d} for t,p,d in bot_tokens]

        print(f"    Top: {[t[0] for t in top_tokens[:10]]}", flush=True)
        print(f"    Bot: {[t[0] for t in bot_tokens[:10]]}", flush=True)

        # 3. Linguistic probes
        tok_arr = np.array(tokens)

        # Position correlation
        pos_corr = np.corrcoef(proj, positions)[0, 1]
        dir_info["corr_position"] = round(float(pos_corr), 4)

        # Frequency correlation
        freqs = np.array([token_freqs[t] for t in tokens])
        freq_corr = np.corrcoef(proj, np.log1p(freqs))[0, 1]
        dir_info["corr_log_frequency"] = round(float(freq_corr), 4)

        # Capitalization
        is_cap = np.array([1 if token_strs[i].strip() and token_strs[i].strip()[0].isupper() else 0
                          for i in range(len(token_strs))])
        if is_cap.std() > 0:
            cap_corr = np.corrcoef(proj, is_cap)[0, 1]
        else:
            cap_corr = 0
        dir_info["corr_capitalized"] = round(float(cap_corr), 4)

        # Punctuation (is the token punctuation?)
        is_punct = np.array([1 if all(not c.isalnum() for c in token_strs[i].strip()) and token_strs[i].strip()
                            else 0 for i in range(len(token_strs))])
        if is_punct.std() > 0:
            punct_corr = np.corrcoef(proj, is_punct)[0, 1]
        else:
            punct_corr = 0
        dir_info["corr_punctuation"] = round(float(punct_corr), 4)

        # Subword (starts with non-space = continuation token)
        is_subword = np.array([1 if not token_strs[i].startswith(" ") and token_strs[i].strip()
                              else 0 for i in range(len(token_strs))])
        if is_subword.std() > 0:
            sub_corr = np.corrcoef(proj, is_subword)[0, 1]
        else:
            sub_corr = 0
        dir_info["corr_subword"] = round(float(sub_corr), 4)

        # Token length
        tok_lens = np.array([len(token_strs[i].strip()) for i in range(len(token_strs))])
        len_corr = np.corrcoef(proj, tok_lens)[0, 1]
        dir_info["corr_token_length"] = round(float(len_corr), 4)

        # Norm of activation
        norms = np.linalg.norm(acts_np, axis=1)
        norm_corr = np.corrcoef(proj, norms)[0, 1]
        dir_info["corr_norm"] = round(float(norm_corr), 4)

        # Kurtosis of activation
        kurts = np.array([scipy_kurtosis(acts_np[i]) for i in range(len(acts_np))])
        kurt_corr = np.corrcoef(proj, kurts)[0, 1]
        dir_info["corr_kurtosis"] = round(float(kurt_corr), 4)

        print(f"    Correlations: pos={pos_corr:.3f} freq={freq_corr:.3f} "
              f"cap={cap_corr:.3f} punct={punct_corr:.3f} sub={sub_corr:.3f} "
              f"len={len_corr:.3f} norm={norm_corr:.3f} kurt={kurt_corr:.3f}", flush=True)

        # 4. What does this direction align with in the MLP weight matrices?
        mlp_W_fc = model.transformer.h[layer_idx].mlp.c_fc.weight.data.numpy()  # (768, 3072)
        mlp_W_proj = model.transformer.h[layer_idx].mlp.c_proj.weight.data.numpy()  # (3072, 768)

        direction = directions[d_idx]  # (768,)

        # Which MLP neurons does this direction activate most?
        neuron_alignment = direction @ mlp_W_fc  # (3072,)
        top_neurons = np.argsort(np.abs(neuron_alignment))[-10:][::-1]
        dir_info["top_mlp_neurons"] = top_neurons.tolist()
        dir_info["top_neuron_alignments"] = neuron_alignment[top_neurons].tolist()

        # 5. Alignment with embedding space
        wte = model.transformer.wte.weight.data.numpy()  # (50257, 768)
        embed_align = wte @ direction  # (50257,)
        top_embed_idx = np.argsort(embed_align)[-10:][::-1]
        bot_embed_idx = np.argsort(embed_align)[:10]
        dir_info["top_embedding_tokens"] = [tokenizer.decode([i]) for i in top_embed_idx]
        dir_info["bottom_embedding_tokens"] = [tokenizer.decode([i]) for i in bot_embed_idx]

        print(f"    Embedding top: {dir_info['top_embedding_tokens']}", flush=True)
        print(f"    Embedding bot: {dir_info['bottom_embedding_tokens']}", flush=True)

        # 6. Alignment with positional embeddings
        wpe = model.transformer.wpe.weight.data.numpy()  # (1024, 768)
        pos_align = wpe @ direction  # (1024,)
        # Is it monotonic? Check correlation with position index
        pos_idx = np.arange(1024)
        pos_mono_corr = np.corrcoef(pos_align, pos_idx)[0, 1]
        dir_info["positional_embedding_corr"] = round(float(pos_mono_corr), 4)
        # Variance explained
        pos_align_var = pos_align.var()
        dir_info["positional_embedding_variance"] = round(float(pos_align_var), 4)

        print(f"    Pos embed monotonicity: {pos_mono_corr:.3f}, var: {pos_align_var:.4f}", flush=True)

        # 7. Check if direction separates POS categories using simple heuristics
        # (We don't have POS tags, but we can check token category distributions)
        # Quartile analysis
        q25, q75 = np.percentile(proj, [25, 75])
        low_mask = proj < q25
        high_mask = proj > q75

        # Token properties in each quartile
        high_freq_mean = np.log1p(freqs[high_mask]).mean()
        low_freq_mean = np.log1p(freqs[low_mask]).mean()
        high_cap_frac = is_cap[high_mask].mean()
        low_cap_frac = is_cap[low_mask].mean()
        high_sub_frac = is_subword[high_mask].mean()
        low_sub_frac = is_subword[low_mask].mean()
        high_punct_frac = is_punct[high_mask].mean()
        low_punct_frac = is_punct[low_mask].mean()
        high_delta = deltas[high_mask].mean()
        low_delta = deltas[low_mask].mean()

        dir_info["quartile_analysis"] = {
            "high": {"freq": round(float(high_freq_mean), 3),
                    "cap": round(float(high_cap_frac), 3),
                    "subword": round(float(high_sub_frac), 3),
                    "punct": round(float(high_punct_frac), 3),
                    "loss_delta": round(float(high_delta), 4)},
            "low": {"freq": round(float(low_freq_mean), 3),
                   "cap": round(float(low_cap_frac), 3),
                   "subword": round(float(low_sub_frac), 3),
                   "punct": round(float(low_punct_frac), 3),
                   "loss_delta": round(float(low_delta), 4)},
        }

        print(f"    Q4 vs Q1: freq {high_freq_mean:.2f} vs {low_freq_mean:.2f}, "
              f"cap {high_cap_frac:.2f} vs {low_cap_frac:.2f}, "
              f"sub {high_sub_frac:.2f} vs {low_sub_frac:.2f}, "
              f"delta {high_delta:.3f} vs {low_delta:.3f}", flush=True)

        results["directions"].append(dir_info)

    # 8. Gate decision boundary geometry
    # Project gate decisions into the effective subspace
    with torch.no_grad():
        gate_scores = gate(acts.unsqueeze(0)).squeeze().numpy()

    linear_mask = gate_scores > 0.5
    nonlinear_mask = gate_scores <= 0.5

    # Centroid of linear vs nonlinear in the effective subspace
    if linear_mask.sum() > 0 and nonlinear_mask.sum() > 0:
        linear_centroid = projections[linear_mask].mean(axis=0)
        nonlinear_centroid = projections[nonlinear_mask].mean(axis=0)
        separation = linear_centroid - nonlinear_centroid
        separation_norm = np.linalg.norm(separation)

        results["boundary"] = {
            "linear_centroid": linear_centroid.tolist(),
            "nonlinear_centroid": nonlinear_centroid.tolist(),
            "separation_vector": separation.tolist(),
            "separation_norm": round(float(separation_norm), 4),
            "n_linear": int(linear_mask.sum()),
            "n_nonlinear": int(nonlinear_mask.sum()),
        }

        # Which dimensions contribute most to separation?
        dim_contributions = np.abs(separation) / (separation_norm + 1e-8)
        top_sep_dims = np.argsort(dim_contributions)[::-1]
        results["boundary"]["top_separating_dims"] = top_sep_dims[:5].tolist()
        results["boundary"]["dim_contributions"] = dim_contributions[top_sep_dims[:5]].tolist()

        print(f"\n  Gate boundary:", flush=True)
        print(f"    {linear_mask.sum()} linear, {nonlinear_mask.sum()} nonlinear", flush=True)
        print(f"    Separation norm: {separation_norm:.4f}", flush=True)
        print(f"    Top separating dims: {top_sep_dims[:5]} (contrib: {dim_contributions[top_sep_dims[:5]].round(3)})", flush=True)

    # 9. Cross-direction interactions: do pairs of directions matter?
    # Quick check: is the boundary linear in the projected space?
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score
    
    if linear_mask.sum() > 10:
        labels = linear_mask.astype(int)
        lr = LogisticRegression(max_iter=1000)
        lr.fit(projections[:, :eff_rank], labels)
        pred = lr.predict(projections[:, :eff_rank])
        acc = accuracy_score(labels, pred)
        results["boundary"]["logistic_accuracy_in_subspace"] = round(float(acc), 4)
        results["boundary"]["logistic_weights"] = lr.coef_[0].tolist()
        
        print(f"    Logistic accuracy in {eff_rank}-d subspace: {acc:.4f}", flush=True)
        print(f"    Logistic weights: {lr.coef_[0].round(3)}", flush=True)

    return results


def main():
    print("=" * 80, flush=True)
    print("GATE DIMENSION ANALYSIS: What do the 9 directions encode?", flush=True)
    print("=" * 80, flush=True)

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2").to(device).eval()

    all_results = {}

    for layer_idx in [2, 11]:  # Focus on the two most interesting layers
        print(f"\n{'='*70}", flush=True)
        print(f"LAYER {layer_idx}", flush=True)
        print(f"{'='*70}", flush=True)

        W_lin, b_lin = fit_linear_approx(model, layer_idx)

        print("  Collecting data...", flush=True)
        acts, L_full, L_lin, tokens, positions, prev_tokens, next_tokens = \
            collect_data(model, tokenizer, layer_idx, W_lin, b_lin)

        print(f"  Training gate ({len(acts)} samples)...", flush=True)
        gate = train_gate(acts, L_full, L_lin)

        print("  Analyzing gate dimensions...", flush=True)
        results = analyze_gate_dimensions(
            model, gate, acts, L_full, L_lin, tokens, positions,
            prev_tokens, next_tokens, tokenizer, layer_idx)

        all_results[f"layer_{layer_idx}"] = results

        out_path = "/Users/peter/clawd/projects/sense-stack/code/gate_dims_results.json"
        with open(out_path, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"  (saved)", flush=True)

    # Summary
    print("\n" + "=" * 80, flush=True)
    print("DIMENSION SUMMARY", flush=True)
    print("=" * 80, flush=True)

    for layer_idx in [2, 11]:
        key = f"layer_{layer_idx}"
        r = all_results[key]
        print(f"\nLayer {layer_idx} (effective rank {r['effective_rank']}):", flush=True)
        for d in r["directions"]:
            # Find the strongest correlation
            corrs = {
                "position": d["corr_position"],
                "frequency": d["corr_log_frequency"],
                "capitalized": d["corr_capitalized"],
                "punctuation": d["corr_punctuation"],
                "subword": d["corr_subword"],
                "token_length": d["corr_token_length"],
                "norm": d["corr_norm"],
                "kurtosis": d["corr_kurtosis"],
            }
            best = max(corrs.items(), key=lambda x: abs(x[1]))
            print(f"  Dir {d['dim']} (σ={d['singular_value']:.2f}): "
                  f"Δcorr={d['corr_with_loss_delta']:.3f}, "
                  f"best={best[0]}({best[1]:.3f}), "
                  f"embed_top={d['top_embedding_tokens'][:3]}", flush=True)


if __name__ == "__main__":
    main()
