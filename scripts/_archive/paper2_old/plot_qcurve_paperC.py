#!/usr/bin/env python
"""Paper C q-curve figure: NDCG@10 (full + tail) vs question budget q, from qcurve_paperC.csv.
Two panels (full, tail), one line per policy with +/-1std shading. Output PDF+PNG into the paper dir."""
import csv, os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

CSV = sys.argv[1] if len(sys.argv) > 1 else 'C:/dev/phd/casper/experiments/paper2/qcurve_paperC.csv'
OUT = sys.argv[2] if len(sys.argv) > 2 else 'C:/dev/phd/papers/paper3_casper/fig_qcurve_paperC'

# display order, labels, styles
SPEC = [
    ('actor_divw', 'Continuous actor + divisiveness (ours)', '#0b5394', '-',  'o', 2.4, 6),
    ('actor_cont', 'Continuous actor (continuity only)',      '#3d85c6', '--', 's', 1.8, 5),
    ('casper',     'CASPER-R (discrete SOTA)',                '#cc0000', '-.', '^', 1.6, 5),
    ('entropy',    'Entropy / divisiveness (discrete)',       '#e69138', ':',  'D', 1.4, 4),
    ('popular',    'Popularity (discrete)',                   '#999999', ':',  'v', 1.2, 4),
]

data = {}
with open(CSV) as fh:
    for row in csv.DictReader(fh):
        data.setdefault(row['policy'], {})[int(row['q'])] = (
            float(row['full']), float(row['full_std']), float(row['tail']), float(row['tail_std']))

fig, axes = plt.subplots(1, 2, figsize=(10, 4.0))
for ax, (idx, title, ylab) in zip(axes, [(0, 'Full NDCG@10', 'NDCG@10 (full)'), (2, 'Long-tail NDCG@10', 'NDCG@10 (tail)')]):
    for key, lab, col, ls, mk, lw, ms in SPEC:
        if key not in data:
            continue
        qs = sorted(data[key])
        mean = np.array([data[key][q][idx] for q in qs])
        std  = np.array([data[key][q][idx + 1] for q in qs])
        ax.plot(qs, mean, ls=ls, color=col, marker=mk, lw=lw, ms=ms, label=lab, zorder=3 if 'divw' in key else 2)
        ax.fill_between(qs, mean - std, mean + std, color=col, alpha=0.12, lw=0)
    ax.set_xlabel('questions asked $q$')
    ax.set_ylabel(ylab)
    ax.set_title(title)
    ax.set_xticks(qs)
    ax.grid(True, alpha=0.3)
axes[0].legend(loc='lower right', fontsize=8, framealpha=0.9)
fig.tight_layout()
for ext in ('pdf', 'png'):
    fig.savefig(f'{OUT}.{ext}', bbox_inches='tight', dpi=160)
    print('wrote', f'{OUT}.{ext}')
