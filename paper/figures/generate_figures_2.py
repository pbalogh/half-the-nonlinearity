#!/usr/bin/env python3
"""Generate figures 2, 3, 4 for the linearization paper."""

import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    'font.size': 12,
    'font.family': 'serif',
    'axes.linewidth': 0.8,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'savefig.facecolor': 'white',
    'savefig.bbox': 'tight',
    'savefig.dpi': 300,
})

BLUE = '#0077BB'
TEAL = '#009988'
ORANGE = '#EE7733'
RED = '#CC3311'
OUTDIR = '/Users/peter/clawd/projects/linearization/docs/figures/'
DATADIR = '/Users/peter/clawd/projects/linearization/data/'

# ── Figure 2: Gate Architecture Comparison ──
with open(DATADIR + 'bootstrap_ci_gpt2_medium.json') as f:
    gpt2m = json.load(f)

fig, ax = plt.subplots(figsize=(10, 4.5))

layers_all = []
all_linear_vals = []
best_gate_vals = []
best_gate_beats = []  # layers where best_gate < 0

for i in range(24):
    layer = gpt2m['layers'][str(i)]
    al = layer['all_linear']
    bg = layer.get('best_gate', {})
    if bg.get('skipped'):
        continue  # skip layer 0
    layers_all.append(i)
    all_linear_vals.append(al['delta_pct'])
    best_gate_vals.append(bg['delta_pct'])
    if bg['delta_pct'] < 0:
        best_gate_beats.append(len(layers_all) - 1)

x = np.array(layers_all)
ax.plot(x, all_linear_vals, 'o-', color=ORANGE, linewidth=2, markersize=5, label='All-Linear', zorder=3)
ax.plot(x, best_gate_vals, 's-', color=TEAL, linewidth=2, markersize=5, label='Best-Gate', zorder=3)

# Highlight where best_gate beats baseline
if best_gate_beats:
    bx = [x[i] for i in best_gate_beats]
    by = [best_gate_vals[i] for i in best_gate_beats]
    ax.scatter(bx, by, s=120, facecolors='none', edgecolors=RED, linewidths=2, zorder=4, label='Beats baseline')

ax.axhline(0, color='#888888', linestyle='--', linewidth=1, alpha=0.7)
ax.set_xlabel('Layer', fontsize=13)
ax.set_ylabel('Perplexity Change (%)', fontsize=13)
ax.set_title('All-Linear vs. Best-Gate Perplexity Cost by Layer (GPT-2 Medium)', fontsize=14, fontweight='bold')
ax.legend(frameon=False, fontsize=11)
ax.set_xticks(x)
ax.grid(axis='y', alpha=0.3)

fig.savefig(OUTDIR + 'fig2_gate_comparison.png')
fig.savefig(OUTDIR + 'fig2_gate_comparison.pdf')
plt.close(fig)
print("✅ Figure 2 saved")

# ── Figure 3: Wanamaker Effect Scaling ──
models = [
    ('bootstrap_ci_EleutherAI_pythia_160m.json', 'Pythia-160M', 162e6, 'Pythia', 'bootstrap'),
    ('bootstrap_ci_gpt2_medium.json', 'GPT-2 Medium', 345e6, 'GPT-2', 'bootstrap'),
    ('bootstrap_ci_EleutherAI_pythia_410m.json', 'Pythia-410M', 405e6, 'Pythia', 'bootstrap'),
    ('bootstrap_ci_EleutherAI_pythia_1b.json', 'Pythia-1B', 1e9, 'Pythia', 'bootstrap'),
    ('scale_test_gpt2_large.json', 'GPT-2 Large', 774e6, 'GPT-2', 'scale_test'),
    ('scale_test_EleutherAI_pythia_2.8b.json', 'Pythia-2.8B', 2.8e9, 'Pythia', 'scale_test'),
]

fig, ax = plt.subplots(figsize=(7, 5))

for fname, label, size, family, fmt in models:
    with open(DATADIR + fname) as f:
        d = json.load(f)
    deltas = []
    for k, v in d['layers'].items():
        if k == '0':
            continue
        if fmt == 'bootstrap':
            al = v.get('all_linear', {})
            dp = al.get('delta_pct')
        else:  # scale_test format
            dp = v.get('all_linear_delta_pct')
        if dp is not None:
            deltas.append(dp)
    median_cost = np.median(deltas)
    color = BLUE if family == 'GPT-2' else ORANGE
    marker = 'D' if family == 'GPT-2' else 'o'
    ax.scatter(size, median_cost, s=120, c=color, marker=marker, zorder=4, edgecolors='white', linewidths=0.5)
    ax.annotate(label, (size, median_cost), textcoords='offset points',
                xytext=(8, 8), fontsize=10, color=color)

ax.set_xscale('log')
ax.set_xlabel('Model Parameters', fontsize=13)
ax.set_ylabel('Median Linearization Cost (% Δ PPL)', fontsize=13)
ax.set_title('The Wanamaker Effect:\nLinearization Cost Decreases with Scale', fontsize=14, fontweight='bold')
ax.grid(axis='both', alpha=0.3)

# Custom legend
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], marker='D', color='w', markerfacecolor=BLUE, markersize=10, label='GPT-2'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor=ORANGE, markersize=10, label='Pythia'),
]
ax.legend(handles=legend_elements, frameon=False, fontsize=11)

fig.savefig(OUTDIR + 'fig3_wanamaker_scaling.png')
fig.savefig(OUTDIR + 'fig3_wanamaker_scaling.pdf')
plt.close(fig)
print("✅ Figure 3 saved")

# ── Figure 4: Linguistic Processing Hierarchy ──
with open(DATADIR + 'forensics_gpt2-medium.json') as f:
    forensics = json.load(f)

# Collect data
layer_labels = ['Layer 1\n(Early)', 'Layer 12\n(Middle)', 'Layer 23\n(Late)']
layer_indices = [1, 12, 23]

# Get all categories, pick interesting ones (skip rare ones like negation, symbol)
keep_cats = ['content_word', 'subword', 'punctuation', 'preposition', 'determiner',
             'conjunction', 'pronoun', 'auxiliary']
cat_labels = {
    'content_word': 'Content Word',
    'subword': 'Subword',
    'punctuation': 'Punctuation',
    'preposition': 'Preposition',
    'determiner': 'Determiner',
    'conjunction': 'Conjunction',
    'pronoun': 'Pronoun',
    'auxiliary': 'Auxiliary',
}

# Build matrix: rows=categories, cols=layers
layer_data = {pl['layer']: pl['category_stats'] for pl in forensics['per_layer']}
matrix = np.zeros((len(keep_cats), len(layer_indices)))
for j, li in enumerate(layer_indices):
    stats = layer_data[li]
    for i, cat in enumerate(keep_cats):
        matrix[i, j] = stats[cat]['mean_delta']

fig, ax = plt.subplots(figsize=(7, 6))
im = ax.imshow(matrix, cmap='YlOrRd', aspect='auto', interpolation='nearest')

ax.set_xticks(range(len(layer_labels)))
ax.set_xticklabels(layer_labels, fontsize=12)
ax.set_yticks(range(len(keep_cats)))
ax.set_yticklabels([cat_labels[c] for c in keep_cats], fontsize=11)

# Annotate cells
for i in range(matrix.shape[0]):
    for j in range(matrix.shape[1]):
        val = matrix[i, j]
        color = 'white' if val > matrix.max() * 0.6 else 'black'
        ax.text(j, i, f'{val:.3f}', ha='center', va='center', fontsize=10, color=color)

cb = fig.colorbar(im, ax=ax, shrink=0.8, label='Mean Δ Loss (MLP contribution)')
ax.set_title('Nonlinearity Demand by Token Type and Depth', fontsize=14, fontweight='bold')

fig.savefig(OUTDIR + 'fig4_linguistic_hierarchy.png')
fig.savefig(OUTDIR + 'fig4_linguistic_hierarchy.pdf')
plt.close(fig)
print("✅ Figure 4 saved")
