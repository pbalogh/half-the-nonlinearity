"""
Progressive Linearization: One Layer at a Time

Instead of linearizing 12 layers simultaneously (compound shock),
linearize one layer, fine-tune briefly, then linearize the next.
The model adapts incrementally.

Usage:
  python3 progressive_linearization.py --model gpt2-medium --linear_layers 4,5,6,7,8,9,10,11,12,13,14,15
  python3 progressive_linearization.py --model gpt2-medium --linear_layers 4,5,6,7,8,9,10,11,12,13,14,15 --steps_per_layer 100
"""

import torch
import torch.nn as nn
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import json, argparse, os, sys

sys.path.insert(0, os.path.dirname(__file__))
from scale_test_universal import ModelAdapter
from variable_capacity_poc import LinearMLP, fit_linear_mlp, evaluate_ppl


def progressive_fine_tune(model, adapter, tokens, trainable_param_fn,
                          n_steps=50, lr=5e-6, batch_size=128):
    """Quick fine-tune pass. Returns final training loss."""
    # Freeze everything
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze selected params
    trainable = 0
    trainable_param_fn(model, adapter)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    if trainable == 0:
        return 0.0

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr, weight_decay=0.01
    )

    train_tokens = tokens[10000:30000]
    model.train()
    losses = []

    for step in range(n_steps):
        start = np.random.randint(0, len(train_tokens) - batch_size - 1)
        input_ids = torch.tensor(
            [train_tokens[start:start + batch_size]], dtype=torch.long)

        outputs = model(input_ids, labels=input_ids)
        loss = outputs.loss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], 1.0)
        optimizer.step()
        losses.append(loss.item())

    model.eval()
    return np.mean(losses[-10:]) if losses else 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='gpt2-medium')
    parser.add_argument('--linear_layers', default='4,5,6,7,8,9,10,11,12,13,14,15')
    parser.add_argument('--steps_per_layer', type=int, default=50,
                        help='Fine-tuning steps after each linearization')
    parser.add_argument('--final_steps', type=int, default=200,
                        help='Final fine-tuning steps after all layers linearized')
    parser.add_argument('--lr', type=float, default=5e-6)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--output', default=None)
    args = parser.parse_args()

    linear_layers = [int(x) for x in args.linear_layers.split(',')]
    # Sort by cheapest to linearize (middle layers first)
    # Our data shows layers 8,6,2,3 are cheapest. Start from center outward.
    n_layers_total = 24  # GPT-2 Medium
    mid = n_layers_total // 2
    linear_layers.sort(key=lambda x: abs(x - mid))  # center-out order

    model_short = args.model.split('/')[-1]

    print(f"Loading {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float32)
    model.eval()
    adapter = ModelAdapter(model, args.model)

    print("Loading WikiText-103...")
    dataset = load_dataset("wikitext", "wikitext-103-raw-v1", split="test")
    text = "\n".join([x for x in dataset["text"] if x.strip()])
    tokens = tokenizer.encode(text)
    print(f"  {len(tokens)} tokens")

    # Baseline
    print("\n=== Baseline ===")
    baseline_ppl = evaluate_ppl(model, tokens)
    print(f"  PPL: {baseline_ppl:.2f}")

    # Progressive linearization
    print(f"\n=== Progressive Linearization ===")
    print(f"  Order: {linear_layers}")
    print(f"  Steps per layer: {args.steps_per_layer}")

    linearized_so_far = set()
    trajectory = [{'layer': 'baseline', 'ppl': float(baseline_ppl),
                    'n_linearized': 0}]

    for layer_idx in linear_layers:
        print(f"\n--- Linearizing layer {layer_idx} "
              f"({len(linearized_so_far)+1}/{len(linear_layers)}) ---")

        # Fit linear MLP
        print(f"  Fitting linear approximation...", flush=True)
        W, b = fit_linear_mlp(model, adapter, tokenizer, layer_idx, tokens)

        # Install it
        linear_mlp = LinearMLP(W, b)
        if adapter.family == 'gpt2':
            model.transformer.h[layer_idx].mlp = linear_mlp
        linearized_so_far.add(layer_idx)

        # Measure immediate damage
        model.eval()
        ppl_after = evaluate_ppl(model, tokens)
        delta = (ppl_after - baseline_ppl) / baseline_ppl * 100
        print(f"  PPL after linearizing: {ppl_after:.2f} ({delta:+.1f}%)")

        # Quick fine-tune
        if args.steps_per_layer > 0:
            print(f"  Fine-tuning {args.steps_per_layer} steps...", flush=True)

            def unfreeze_neighbors(model, adapter, li=layer_idx, ls=linearized_so_far):
                """Unfreeze attention everywhere + MLPs at non-linearized layers."""
                for i in range(adapter.n_layers):
                    # Attention always trainable
                    attn = model.transformer.h[i].attn
                    for p in attn.parameters():
                        p.requires_grad = True
                    # MLP only if not linearized
                    if i not in ls:
                        mlp = model.transformer.h[i].mlp
                        for p in mlp.parameters():
                            p.requires_grad = True
                # LM head
                for p in model.lm_head.parameters():
                    p.requires_grad = True

            avg_loss = progressive_fine_tune(
                model, adapter, tokens, unfreeze_neighbors,
                n_steps=args.steps_per_layer, lr=args.lr,
                batch_size=args.batch_size)

            model.eval()
            ppl_recovered = evaluate_ppl(model, tokens)
            delta_r = (ppl_recovered - baseline_ppl) / baseline_ppl * 100
            print(f"  PPL after fine-tuning: {ppl_recovered:.2f} ({delta_r:+.1f}%)")
            print(f"  Train loss: {avg_loss:.4f}")

            trajectory.append({
                'layer': int(layer_idx),
                'n_linearized': len(linearized_so_far),
                'ppl_before_ft': float(ppl_after),
                'ppl_after_ft': float(ppl_recovered),
                'delta_pct': float(delta_r),
                'train_loss': float(avg_loss),
            })
        else:
            trajectory.append({
                'layer': int(layer_idx),
                'n_linearized': len(linearized_so_far),
                'ppl': float(ppl_after),
                'delta_pct': float(delta),
            })

    # Final fine-tuning pass
    if args.final_steps > 0:
        print(f"\n=== Final Fine-Tuning ({args.final_steps} steps) ===")

        def unfreeze_all_nonlinear(model, adapter, ls=linearized_so_far):
            for i in range(adapter.n_layers):
                attn = model.transformer.h[i].attn
                for p in attn.parameters():
                    p.requires_grad = True
                if i not in ls:
                    mlp = model.transformer.h[i].mlp
                    for p in mlp.parameters():
                        p.requires_grad = True
            for p in model.lm_head.parameters():
                p.requires_grad = True

        avg_loss = progressive_fine_tune(
            model, adapter, tokens, unfreeze_all_nonlinear,
            n_steps=args.final_steps, lr=args.lr,
            batch_size=args.batch_size)

        model.eval()
        final_ppl = evaluate_ppl(model, tokens)
        final_delta = (final_ppl - baseline_ppl) / baseline_ppl * 100
        print(f"  Final PPL: {final_ppl:.2f} ({final_delta:+.1f}%)")

        trajectory.append({
            'layer': 'final_ft',
            'ppl': float(final_ppl),
            'delta_pct': float(final_delta),
            'train_loss': float(avg_loss),
        })
    else:
        final_ppl = trajectory[-1].get('ppl_after_ft', trajectory[-1].get('ppl', 0))

    # Summary
    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    print(f"  Model: {args.model}")
    print(f"  Layers linearized: {len(linearized_so_far)}/{adapter.n_layers} "
          f"({len(linearized_so_far)/adapter.n_layers:.0%})")
    print(f"  Baseline PPL:  {baseline_ppl:.2f}")
    print(f"  Final PPL:     {final_ppl:.2f} ({(final_ppl-baseline_ppl)/baseline_ppl*100:+.1f}%)")
    print(f"  Steps used:    {len(linear_layers) * args.steps_per_layer + args.final_steps}")

    if final_ppl < baseline_ppl:
        print(f"  ★ BEATS BASELINE!")
    elif (final_ppl - baseline_ppl) / baseline_ppl < 0.05:
        print(f"  ✓ Within 5% of baseline — VIABLE")
    elif (final_ppl - baseline_ppl) / baseline_ppl < 0.10:
        print(f"  ~ Within 10% of baseline — promising")
    else:
        print(f"  △ Still {(final_ppl-baseline_ppl)/baseline_ppl*100:.1f}% above baseline")

    # Trajectory table
    print(f"\n  {'Step':>20} {'#Lin':>5} {'PPL':>8} {'Δ%':>8}")
    for t in trajectory:
        layer = t['layer']
        n = t.get('n_linearized', '-')
        ppl = t.get('ppl_after_ft', t.get('ppl', 0))
        delta = t.get('delta_pct', (ppl-baseline_ppl)/baseline_ppl*100 if ppl else 0)
        print(f"  {str(layer):>20} {str(n):>5} {ppl:>8.2f} {delta:>+8.1f}%")

    # Save
    results = {
        'model': args.model,
        'linear_layers': sorted(linearized_so_far),
        'linearization_order': linear_layers,
        'baseline_ppl': float(baseline_ppl),
        'final_ppl': float(final_ppl),
        'final_delta_pct': float((final_ppl-baseline_ppl)/baseline_ppl*100),
        'steps_per_layer': args.steps_per_layer,
        'final_steps': args.final_steps,
        'total_steps': len(linear_layers) * args.steps_per_layer + args.final_steps,
        'trajectory': trajectory,
    }

    out_path = args.output or f"progressive_lin_{model_short}.json"
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == '__main__':
    main()
