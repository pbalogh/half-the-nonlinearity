# Half the Nonlinearity Is Wasted

**Measuring and Reallocating the Transformer's MLP Budget**

Peter Balogh · [palexanderbalogh@gmail.com](mailto:palexanderbalogh@gmail.com)

---

## Abstract

We investigate when transformer MLP nonlinearity is actually necessary. Through systematic investigation across six models (162M–2.8B parameters), two architectures, three corpora, and 50,000+ tokens, we show that a substantial fraction of MLP computations can be replaced by a precomputed linear matrix at negligible cost. A gate with *d*+1 parameters—a single logistic classifier—decides when. But the gate's decision defies simple characterization: nonlinearity need cannot be predicted from token identity.

## Key Findings

| Finding | Number |
|---|---|
| GPT-2 Large layers that beat baseline with gating | **11 / 36** |
| PPL improvement via two-phase gated linearization | **17.3%** |
| Cross-corpus correlation of token routing lists | **r < 0.05** (dead end) |
| Max all-linear cost in GPT-2 Large | **3.7%** per layer |
| Min all-linear cost in Pythia-2.8B | **1.9%** (L3 beats baseline at −0.13%) |

- **Token-based routing is a dead end.** Per-token "No-Fly lists" show zero generalization across corpora. Over a quarter of flagged tokens flip behavior on new text.
- **Context dominates.** A context-only gate matches the full gate within 0.004 AUC at every layer.
- **Linearization as regularization.** At 4/23 GPT-2 Medium layers, the gated linear path *outperforms* the full MLP.
- **Architecture matters.** GPT-2 linearizes cheaply; Pythia shows higher costs, though Pythia-2.8B L3 narrowly beats baseline.

## Repository Structure

```
half-the-nonlinearity/
├── code/               # All experiment scripts
│   ├── scale_test_universal.py    # Main: sweep all layers across models
│   ├── beefy_linearization*.py    # Progressive linearization experiments
│   ├── context_vs_token_gating.py # Context vs token ablation
│   ├── nofly_cross_corpus.py      # Cross-corpus routing analysis
│   └── ...
├── data/               # Experiment results (33 JSON files)
├── paper/
│   ├── half-the-nonlinearity.tex  # Full paper
│   ├── figures/                   # All figures (PDF + PNG)
│   │   └── generate_figures_2.py  # Figure generation script
│   ├── gated-linearization-summary.md
│   └── SUBMISSION_STRATEGY.md
├── requirements.txt
├── LICENSE             # MIT (code), CC-BY-4.0 (paper)
└── README.md
```

## Reproducing Results

### Setup

```bash
pip install -r requirements.txt
```

### Run the main layer sweep

```bash
python code/scale_test_universal.py --model gpt2-medium --num-tokens 4096
```

This sweeps all layers, fits linear approximations, trains gates, and outputs JSON results to `data/`.

### Generate figures

```bash
cd paper/figures
python generate_figures_2.py
```

Reads from `../../data/` and produces the PDF/PNG figures used in the paper.

### Other experiments

- `code/beefy_linearization_gated_v3.py` — Two-phase gated progressive linearization
- `code/context_vs_token_gating.py` — Context vs. token decomposition
- `code/nofly_cross_corpus.py` — Cross-corpus routing list analysis
- `code/control_finetune.py` — Vanilla fine-tuning control

## Citation

```bibtex
@article{balogh2025half,
  title={Half the Nonlinearity Is Wasted: Measuring and Reallocating the Transformer's MLP Budget},
  author={Balogh, Peter},
  year={2025}
}
```

## License

- **Code:** MIT License
- **Paper and figures:** [CC-BY 4.0](https://creativecommons.org/licenses/by/4.0/)
