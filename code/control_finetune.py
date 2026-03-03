"""
Control Experiment: Fine-tune WITHOUT linearization.

If vanilla fine-tuning on WikiText-103 train also drops WikiText test PPL
by 10-13%, then linearization adds nothing — we're just measuring the
fine-tuning effect. This is the critical control.

Usage:
  python3 control_finetune.py --model gpt2-medium --steps 2000
"""

import torch
import torch.nn as nn
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import json, argparse, os, sys

sys.path.insert(0, os.path.dirname(__file__))


def evaluate_ppl_corpus(model, tokens, start=0, n_eval=20000, batch_size=512):
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    eval_tokens = tokens[start:start + n_eval]

    with torch.no_grad():
        for chunk_start in range(0, len(eval_tokens) - 1, batch_size):
            chunk_end = min(chunk_start + batch_size, len(eval_tokens))
            input_ids = torch.tensor(
                [eval_tokens[chunk_start:chunk_end]], dtype=torch.long)
            if input_ids.shape[1] < 2:
                continue
            outputs = model(input_ids, labels=input_ids)
            n = input_ids.shape[1] - 1
            total_loss += outputs.loss.item() * n
            total_tokens += n

    return np.exp(total_loss / total_tokens)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='gpt2-medium')
    parser.add_argument('--train_tokens', default='wikitext103_train_tokens.npy')
    parser.add_argument('--steps', type=int, default=2000)
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--eval_every', type=int, default=200)
    parser.add_argument('--output', default=None)
    args = parser.parse_args()

    print(f"=== CONTROL EXPERIMENT: Fine-tune WITHOUT linearization ===", flush=True)
    print(f"Same training setup as linearization experiment, but NO layers linearized.", flush=True)
    print(f"If PPL drops similarly, linearization isn't the cause.\n", flush=True)

    print(f"Loading {args.model}...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float32)
    model.eval()

    # Load training data
    print(f"Loading pre-tokenized training data: {args.train_tokens}", flush=True)
    train_tokens = np.load(args.train_tokens).tolist()
    print(f"  Train: {len(train_tokens):,} tokens", flush=True)

    # Load eval corpora
    print("Loading eval corpora...", flush=True)
    ds_test = load_dataset("wikitext", "wikitext-103-raw-v1", split="test")
    test_text = "\n".join([x for x in ds_test["text"] if x.strip()])
    test_tokens = tokenizer.encode(test_text)
    print(f"  WikiText test: {len(test_tokens):,} tokens", flush=True)

    ds_lambada = load_dataset("lambada", split="test")
    lambada_text = "\n".join([x for x in ds_lambada["text"] if x.strip()])
    lambada_tokens = tokenizer.encode(lambada_text)
    print(f"  LAMBADA: {len(lambada_tokens):,} tokens", flush=True)

    # Baselines
    print("\n=== Baselines ===", flush=True)
    base_wiki = evaluate_ppl_corpus(model, test_tokens, start=50000, n_eval=20000)
    base_lambada = evaluate_ppl_corpus(model, lambada_tokens, start=0, n_eval=20000)
    print(f"  WikiText test PPL: {base_wiki:.2f}", flush=True)
    print(f"  LAMBADA PPL:       {base_lambada:.2f}", flush=True)

    # Fine-tune ALL parameters (no linearization)
    print(f"\n=== Fine-Tuning ALL parameters ({args.steps} steps) ===", flush=True)

    trainable = sum(p.numel() for p in model.parameters())
    print(f"  Trainable: {trainable:,} (100%)", flush=True)

    for p in model.parameters():
        p.requires_grad = True

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=0.01)

    from torch.optim.lr_scheduler import CosineAnnealingLR
    scheduler = CosineAnnealingLR(optimizer, T_max=args.steps, eta_min=args.lr/10)

    model.train()
    eval_trajectory = []
    train_len = len(train_tokens)
    train_pos = 0

    for step in range(args.steps):
        if train_pos + args.batch_size + 1 >= train_len:
            train_pos = 0
        input_ids = torch.tensor(
            [train_tokens[train_pos:train_pos + args.batch_size]], dtype=torch.long)
        train_pos += args.batch_size

        outputs = model(input_ids, labels=input_ids)
        loss = outputs.loss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        if (step + 1) % 50 == 0:
            lr_now = scheduler.get_last_lr()[0]
            print(f"  Step {step+1}/{args.steps}: loss={loss.item():.4f} lr={lr_now:.2e}",
                  flush=True)

        if (step + 1) % args.eval_every == 0:
            model.eval()
            wiki_ppl = evaluate_ppl_corpus(model, test_tokens, start=50000, n_eval=20000)
            lamb_ppl = evaluate_ppl_corpus(model, lambada_tokens, start=0, n_eval=20000)
            print(f"  → WikiText: {wiki_ppl:.2f} ({(wiki_ppl-base_wiki)/base_wiki*100:+.1f}%) "
                  f"| LAMBADA: {lamb_ppl:.2f} ({(lamb_ppl-base_lambada)/base_lambada*100:+.1f}%)",
                  flush=True)
            eval_trajectory.append({
                'step': step + 1,
                'wiki_ppl': float(wiki_ppl),
                'lambada_ppl': float(lamb_ppl),
                'wiki_delta': float((wiki_ppl-base_wiki)/base_wiki*100),
                'lambada_delta': float((lamb_ppl-base_lambada)/base_lambada*100),
            })
            model.train()

    # Final eval
    print("\n=== Final Evaluation ===", flush=True)
    model.eval()
    final_wiki = evaluate_ppl_corpus(model, test_tokens, start=50000, n_eval=20000)
    final_lambada = evaluate_ppl_corpus(model, lambada_tokens, start=0, n_eval=20000)

    print(f"  {'':>15} {'Baseline':>10} {'Final':>10} {'Δ%':>8}", flush=True)
    print(f"  {'WikiText':>15} {base_wiki:>10.2f} {final_wiki:>10.2f} "
          f"{(final_wiki-base_wiki)/base_wiki*100:>+8.1f}%", flush=True)
    print(f"  {'LAMBADA':>15} {base_lambada:>10.2f} {final_lambada:>10.2f} "
          f"{(final_lambada-base_lambada)/base_lambada*100:>+8.1f}%", flush=True)

    wiki_delta = (final_wiki-base_wiki)/base_wiki*100
    print(f"\n  === VERDICT ===", flush=True)
    print(f"  Control (no linearization): WikiText {wiki_delta:+.1f}%", flush=True)
    print(f"  Linearization (4 layers):   WikiText -10.2%", flush=True)
    if wiki_delta > -5:
        print(f"  ✓ Linearization effect is REAL — control doesn't explain the improvement", flush=True)
    else:
        print(f"  ⚠ Fine-tuning alone produces significant improvement — linearization effect may be confounded", flush=True)

    # Save
    results = {
        'experiment': 'control_finetune_no_linearization',
        'model': args.model,
        'linear_layers': [],
        'steps': args.steps,
        'lr': args.lr,
        'batch_size': args.batch_size,
        'train_corpus_size': len(train_tokens),
        'baseline': {'wiki': float(base_wiki), 'lambada': float(base_lambada)},
        'final': {'wiki': float(final_wiki), 'lambada': float(final_lambada)},
        'wiki_delta_pct': float(wiki_delta),
        'lambada_delta_pct': float((final_lambada-base_lambada)/base_lambada*100),
        'trajectory': eval_trajectory,
    }

    out_path = args.output or f"control_finetune_{args.model.split('/')[-1]}.json"
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_path}", flush=True)


if __name__ == '__main__':
    main()
