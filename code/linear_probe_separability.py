"""
Linear Probe Experiment: Is the nonlinearity decision linearly separable?

For each layer with a trained gate, we:
1. Collect residual stream activations at the MLP input
2. Binarize gate decisions (>0.5 = needs nonlinearity, ≤0.5 = linear suffices)
3. Fit a logistic regression (linear probe) on activations → binary label
4. Compare to: (a) majority-class baseline, (b) MLP probe (1 hidden layer)
5. Report accuracy, F1, and AUROC

This directly tests the claim that "a single direction separates linear from nonlinear."

Usage:
  python3 linear_probe_separability.py --model gpt2-medium \
    --checkpoint_dir /path/to/phase2/checkpoints \
    --output results_linear_probe.json
"""

import torch
import torch.nn as nn
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
import json, argparse, os, sys

sys.path.insert(0, os.path.dirname(__file__))


def get_mlp_inputs_and_gates(model, tokenizer, gate_weights, layers, n_tokens=50000, seq_len=256):
    """
    Run model forward, hooking MLP inputs at specified layers.
    Also compute what the trained gate would output for each position.
    
    gate_weights: dict of layer_idx -> {'weight': tensor, 'bias': tensor}
    (for linear gates: gate = sigmoid(x @ weight + bias))
    
    Returns: dict of layer_idx -> {'activations': [n_tokens, hidden_dim], 'labels': [n_tokens]}
    """
    device = next(model.parameters()).device
    hidden_dim = model.config.n_embd
    
    # Prepare data
    dataset = load_dataset("wikitext", "wikitext-103-raw-v1", split="validation")
    text = "\n\n".join([t for t in dataset["text"] if len(t) > 100])
    tokens = tokenizer.encode(text)
    
    # Collect activations via hooks
    collected = {l: {'activations': [], 'gate_logits': []} for l in layers}
    hooks = []
    
    def make_hook(layer_idx):
        def hook_fn(module, input, output):
            x = input[0].detach()  # [batch, seq, hidden]
            collected[layer_idx]['activations'].append(x.cpu())
            # Compute gate logit
            w = gate_weights[layer_idx]['weight']  # [hidden_dim, 1] or [hidden_dim]
            b = gate_weights[layer_idx]['bias']     # [1] or scalar
            logit = (x.cpu().float() @ w.float()).squeeze(-1) + b.float()
            collected[layer_idx]['gate_logits'].append(logit)
        return hook_fn
    
    # Install hooks on MLP modules
    for layer_idx in layers:
        block = model.transformer.h[layer_idx]
        h = block.mlp.register_forward_hook(make_hook(layer_idx))
        hooks.append(h)
    
    # Forward pass
    total_collected = 0
    with torch.no_grad():
        for start in range(0, len(tokens) - seq_len, seq_len):
            if total_collected >= n_tokens:
                break
            input_ids = torch.tensor([tokens[start:start+seq_len]], device=device)
            model(input_ids)
            total_collected += seq_len
    
    # Remove hooks
    for h in hooks:
        h.remove()
    
    # Concatenate and binarize
    results = {}
    for l in layers:
        acts = torch.cat(collected[l]['activations'], dim=1).squeeze(0).numpy()  # [n, hidden]
        logits = torch.cat(collected[l]['gate_logits'], dim=1).squeeze(0).numpy()  # [n]
        # Gate output: sigmoid(logit). Our convention: high gate = use original MLP (needs nonlinearity)
        gate_vals = 1 / (1 + np.exp(-logits))
        labels = (gate_vals > 0.5).astype(int)
        
        # Subsample if too large
        if len(acts) > n_tokens:
            acts = acts[:n_tokens]
            labels = labels[:n_tokens]
            gate_vals = gate_vals[:n_tokens]
        
        results[l] = {
            'activations': acts,
            'labels': labels,
            'gate_vals': gate_vals,
        }
    
    return results


def extract_gate_weights_from_checkpoint(checkpoint_path, layers):
    """
    Extract trained linear gate weights from a phase2 checkpoint.
    Expects gate parameters named like 'gate_networks.{layer_idx}.linear.weight' etc.
    """
    state_dict = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    
    gate_weights = {}
    for l in layers:
        # Try different naming conventions
        w_key = None
        b_key = None
        for k in state_dict:
            if f'{l}' in k and 'weight' in k and 'gate' in k.lower():
                w_key = k
            if f'{l}' in k and 'bias' in k and 'gate' in k.lower():
                b_key = k
        
        if w_key and b_key:
            w = state_dict[w_key]
            b = state_dict[b_key]
            if w.dim() == 2:
                # [out, in] -> [in, out] for x @ w convention
                if w.shape[0] == 1:
                    w = w.T
            gate_weights[l] = {'weight': w, 'bias': b.squeeze()}
        else:
            print(f"  Warning: Could not find gate weights for layer {l}")
            print(f"  Available keys: {[k for k in state_dict if f'{l}' in k]}")
    
    return gate_weights


def extract_gate_weights_from_json(json_path):
    """
    If we saved gate parameters in the results JSON, extract them.
    Otherwise fall back to checkpoint.
    """
    with open(json_path) as f:
        data = json.load(f)
    
    # Check if gate weights are stored
    if 'gate_weights' in data:
        gate_weights = {}
        for l_str, gw in data['gate_weights'].items():
            l = int(l_str)
            gate_weights[l] = {
                'weight': torch.tensor(gw['weight']),
                'bias': torch.tensor(gw['bias']),
            }
        return gate_weights
    return None


def run_probes(activations, labels, gate_vals):
    """
    Fit linear and nonlinear probes. Return metrics.
    """
    X_train, X_test, y_train, y_test, g_train, g_test = train_test_split(
        activations, labels, gate_vals, test_size=0.2, random_state=42
    )
    
    n_pos = y_train.sum()
    n_neg = len(y_train) - n_pos
    majority_class = 1 if n_pos > n_neg else 0
    majority_acc = max(n_pos, n_neg) / len(y_train)
    
    # Linear probe (logistic regression)
    lr = LogisticRegression(max_iter=1000, C=1.0, solver='lbfgs', random_state=42)
    lr.fit(X_train, y_train)
    lr_pred = lr.predict(X_test)
    lr_prob = lr.predict_proba(X_test)[:, 1]
    
    lr_acc = accuracy_score(y_test, lr_pred)
    lr_f1 = f1_score(y_test, lr_pred)
    lr_auroc = roc_auc_score(y_test, lr_prob)
    
    # Nonlinear probe (MLP with 1 hidden layer)
    mlp = MLPClassifier(hidden_layer_sizes=(64,), max_iter=500, random_state=42)
    mlp.fit(X_train, y_train)
    mlp_pred = mlp.predict(X_test)
    mlp_prob = mlp.predict_proba(X_test)[:, 1]
    
    mlp_acc = accuracy_score(y_test, mlp_pred)
    mlp_f1 = f1_score(y_test, mlp_pred)
    mlp_auroc = roc_auc_score(y_test, mlp_prob)
    
    return {
        'n_samples': len(activations),
        'class_balance': float(labels.mean()),
        'majority_baseline_acc': float(majority_acc),
        'linear_probe': {
            'accuracy': float(lr_acc),
            'f1': float(lr_f1),
            'auroc': float(lr_auroc),
        },
        'mlp_probe': {
            'accuracy': float(mlp_acc),
            'f1': float(mlp_f1),
            'auroc': float(mlp_auroc),
        },
        'nonlinear_advantage': {
            'accuracy_delta': float(mlp_acc - lr_acc),
            'auroc_delta': float(mlp_auroc - lr_auroc),
        }
    }


def alternative_ground_truth(model, tokenizer, layers, n_tokens=50000, seq_len=256):
    """
    Alternative labeling: instead of gate decisions, use actual MLP delta magnitude.
    Median-split into high-delta (needs nonlinearity) vs low-delta (doesn't).
    This avoids circularity of probing a linear gate with a linear probe.
    """
    device = next(model.parameters()).device
    
    dataset = load_dataset("wikitext", "wikitext-103-raw-v1", split="validation")
    text = "\n\n".join([t for t in dataset["text"] if len(t) > 100])
    tokens = tokenizer.encode(text)
    
    collected = {l: {'activations': [], 'mlp_input': [], 'mlp_output': []} for l in layers}
    hooks = []
    
    def make_hook(layer_idx):
        def hook_fn(module, input, output):
            x = input[0].detach().cpu()
            y = output.detach().cpu()
            collected[layer_idx]['mlp_input'].append(x)
            collected[layer_idx]['mlp_output'].append(y)
        return hook_fn
    
    for layer_idx in layers:
        block = model.transformer.h[layer_idx]
        h = block.mlp.register_forward_hook(make_hook(layer_idx))
        hooks.append(h)
    
    total = 0
    with torch.no_grad():
        for start in range(0, len(tokens) - seq_len, seq_len):
            if total >= n_tokens:
                break
            input_ids = torch.tensor([tokens[start:start+seq_len]], device=device)
            model(input_ids)
            total += seq_len
    
    for h in hooks:
        h.remove()
    
    results = {}
    for l in layers:
        inputs = torch.cat(collected[l]['mlp_input'], dim=1).squeeze(0)
        outputs = torch.cat(collected[l]['mlp_output'], dim=1).squeeze(0)
        
        # Delta = nonlinear_output - linear_approx
        # But we don't have the linear approx here. Use ||MLP(x)|| as proxy for nonlinearity magnitude.
        # Actually: use ||MLP(x) - best_linear(x)|| but we'd need to fit that.
        # Simpler: just use ||MLP(x)|| magnitude — tokens where MLP does little = linearizable
        mlp_delta = outputs.numpy()
        delta_norms = np.linalg.norm(mlp_delta, axis=-1)
        
        # Median split
        median = np.median(delta_norms)
        labels = (delta_norms > median).astype(int)
        
        if len(inputs) > n_tokens:
            inputs = inputs[:n_tokens]
            labels = labels[:n_tokens]
            delta_norms = delta_norms[:n_tokens]
        
        results[l] = {
            'activations': inputs.numpy(),
            'labels': labels,
            'gate_vals': delta_norms / delta_norms.max(),  # normalized for AUROC
        }
    
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='gpt2-medium')
    parser.add_argument('--checkpoint', help='Path to phase2 checkpoint with gate weights')
    parser.add_argument('--results_json', help='Path to phase2 results JSON (may contain gate weights)')
    parser.add_argument('--layers', default='1,2,3,5,6,7,10,11,12,13,14,15,16,17,18,19,20,21,22,23',
                        help='Layers to probe')
    parser.add_argument('--n_tokens', type=int, default=50000)
    parser.add_argument('--output', default='results_linear_probe.json')
    parser.add_argument('--method', choices=['gate', 'delta', 'both'], default='both',
                        help='Ground truth: gate decisions, MLP delta magnitude, or both')
    args = parser.parse_args()
    
    layers = [int(l) for l in args.layers.split(',')]
    
    print(f"Loading {args.model}...")
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    model = AutoModelForCausalLM.from_pretrained(args.model).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    
    all_results = {'model': args.model, 'n_tokens': args.n_tokens, 'layers': {}}
    
    # Method 1: Probe using MLP delta magnitude (no gate needed — avoids circularity)
    if args.method in ('delta', 'both'):
        print("\n=== Method: MLP Delta Magnitude (median split) ===")
        delta_data = alternative_ground_truth(model, tokenizer, layers, args.n_tokens)
        
        for l in sorted(delta_data.keys()):
            d = delta_data[l]
            print(f"\n  Layer {l}: {d['labels'].sum()}/{len(d['labels'])} need nonlinearity "
                  f"(balance: {d['labels'].mean():.3f})")
            metrics = run_probes(d['activations'], d['labels'], d['gate_vals'])
            print(f"    Linear probe:  acc={metrics['linear_probe']['accuracy']:.4f}  "
                  f"AUROC={metrics['linear_probe']['auroc']:.4f}")
            print(f"    MLP probe:     acc={metrics['mlp_probe']['accuracy']:.4f}  "
                  f"AUROC={metrics['mlp_probe']['auroc']:.4f}")
            print(f"    Nonlinear advantage: "
                  f"Δacc={metrics['nonlinear_advantage']['accuracy_delta']:+.4f}  "
                  f"ΔAUROC={metrics['nonlinear_advantage']['auroc_delta']:+.4f}")
            
            if l not in all_results['layers']:
                all_results['layers'][l] = {}
            all_results['layers'][l]['delta_method'] = metrics
    
    # Method 2: Probe using trained gate decisions
    if args.method in ('gate', 'both') and (args.checkpoint or args.results_json):
        print("\n=== Method: Trained Gate Decisions ===")
        
        gate_weights = None
        if args.results_json:
            gate_weights = extract_gate_weights_from_json(args.results_json)
        if gate_weights is None and args.checkpoint:
            gate_weights = extract_gate_weights_from_checkpoint(args.checkpoint, layers)
        
        if gate_weights:
            gate_layers = [l for l in layers if l in gate_weights]
            gate_data = get_mlp_inputs_and_gates(model, tokenizer, gate_weights, gate_layers, args.n_tokens)
            
            for l in sorted(gate_data.keys()):
                d = gate_data[l]
                print(f"\n  Layer {l}: {d['labels'].sum()}/{len(d['labels'])} need nonlinearity "
                      f"(balance: {d['labels'].mean():.3f})")
                metrics = run_probes(d['activations'], d['labels'], d['gate_vals'])
                print(f"    Linear probe:  acc={metrics['linear_probe']['accuracy']:.4f}  "
                      f"AUROC={metrics['linear_probe']['auroc']:.4f}")
                print(f"    MLP probe:     acc={metrics['mlp_probe']['accuracy']:.4f}  "
                      f"AUROC={metrics['mlp_probe']['auroc']:.4f}")
                print(f"    Nonlinear advantage: "
                      f"Δacc={metrics['nonlinear_advantage']['accuracy_delta']:+.4f}  "
                      f"ΔAUROC={metrics['nonlinear_advantage']['auroc_delta']:+.4f}")
                
                if l not in all_results['layers']:
                    all_results['layers'][l] = {}
                all_results['layers'][l]['gate_method'] = metrics
        else:
            print("  No gate weights found — skipping gate method")
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    for l in sorted(all_results['layers'].keys(), key=int if isinstance(list(all_results['layers'].keys())[0], str) else lambda x: x):
        layer_data = all_results['layers'][l]
        for method_name, metrics in layer_data.items():
            lp = metrics['linear_probe']
            mp = metrics['mlp_probe']
            adv = metrics['nonlinear_advantage']
            print(f"  L{l:2d} ({method_name:12s}): "
                  f"linear AUROC={lp['auroc']:.3f}  "
                  f"MLP AUROC={mp['auroc']:.3f}  "
                  f"Δ={adv['auroc_delta']:+.3f}  "
                  f"(balance={metrics['class_balance']:.2f})")
    
    # Save
    # Convert int keys to strings for JSON
    json_results = {
        'model': all_results['model'],
        'n_tokens': all_results['n_tokens'],
        'layers': {str(k): v for k, v in all_results['layers'].items()}
    }
    output_path = os.path.join(os.path.dirname(__file__), '..', 'data', args.output)
    with open(output_path, 'w') as f:
        json.dump(json_results, f, indent=2)
    print(f"\nSaved to {output_path}")


if __name__ == '__main__':
    main()
