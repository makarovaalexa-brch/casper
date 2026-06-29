#!/usr/bin/env python
"""Paper C q-curve figure, CANONICAL harness (continuous_policy2_st.py).
Discrete baselines (native answers) from evalcks_qcurve.csv; continuous actors from the
parsed qcanon/*.log (MODES=contactor, QPTS=0,2,4,8). 2-panel full+tail, seed-avg +/-1std.
Also prints a LaTeX-ready tab:qcurve block (tail, q=0,2,4,8)."""
import csv, os, re, glob
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE = 'C:/dev/phd/casper'
EVALCKS = f'{BASE}/data/movielens/.cache/evalcks_qcurve.csv'
QCANON  = f'{BASE}/experiments/paper2/qcanon'
OUT     = 'C:/dev/phd/papers/paper3_casper/fig_qcurve_paperC'
QS      = [0, 2, 4, 8]
SEEDS   = [1, 2, 3, 7, 11]

# ---- discrete baselines from EVALCKS csv: seed,name,q,FULL,TAIL,... ----
disc = {}  # name -> q -> list of (full,tail) per seed
with open(EVALCKS) as fh:
    for r in csv.reader(fh):
        if len(r) < 5: continue
        sd, nm, q = int(r[0]), r[1], int(r[2])
        if q not in QS: continue
        disc.setdefault(nm, {}).setdefault(q, []).append((float(r[3]), float(r[4])))

# ---- continuous actors parsed from qcanon logs: <tag>_seed<sd>.log ----
def parse_contactor(path):
    """returns (full_list, tail_list) aligned to QPTS order, from the two 'contactor: NDCG ...' lines."""
    blocks = []
    for line in open(path, encoding='utf-8', errors='ignore'):
        m = re.search(r'contactor:\s*NDCG\s+([0-9.\s]+?)\s*\|', line)
        if m: blocks.append([float(x) for x in m.group(1).split()])
    if len(blocks) < 2: return None
    return blocks[0], blocks[1]  # FULL block, TAIL block (printed in that order)

actors = {}  # tag -> q -> list of (full,tail) per seed
for path in glob.glob(f'{QCANON}/*_seed*.log'):
    base = os.path.basename(path)
    tag = base.split('_seed')[0]
    pr = parse_contactor(path)
    if pr is None: continue
    full, tail = pr
    for i, q in enumerate(QS):
        if i < len(full) and i < len(tail):
            actors.setdefault(tag, {}).setdefault(q, []).append((full[i], tail[i]))

# ---- assemble series: (key, label, color, ls, marker, lw, ms, source) ----
def series(store, name):
    qs = [q for q in QS if q in store.get(name, {})]
    fm = np.array([np.mean([v[0] for v in store[name][q]]) for q in qs])
    fs = np.array([np.std ([v[0] for v in store[name][q]]) for q in qs])
    tm = np.array([np.mean([v[1] for v in store[name][q]]) for q in qs])
    ts = np.array([np.std ([v[1] for v in store[name][q]]) for q in qs])
    return qs, fm, fs, tm, ts

SPEC = [
    ('d1',             actors, 'Continuous actor + divisiveness (ours)', '#0b5394', '-',  'o', 2.4, 6),
    ('cont',           actors, 'Continuous actor (continuity only)',     '#3d85c6', '--', 's', 1.8, 5),
    ('entdistill_ep4', disc,   'CASPER-R (discrete SOTA)',               '#cc0000', '-.', '^', 1.6, 5),
    ('entropy',        disc,   'Entropy / divisiveness (discrete)',      '#e69138', ':',  'D', 1.4, 4),
    ('conc_pop',       disc,   'Popularity (discrete)',                  '#999999', ':',  'v', 1.2, 4),
]

fig, axes = plt.subplots(1, 2, figsize=(10, 4.0))
for ax, idx, title, ylab in [(axes[0], 0, 'Full NDCG@10', 'NDCG@10 (full)'),
                             (axes[1], 1, 'Long-tail NDCG@10', 'NDCG@10 (tail)')]:
    for key, store, lab, col, ls, mk, lw, ms in SPEC:
        if key not in store: continue
        qs, fm, fs, tm, ts = series(store, key)
        m, s = (fm, fs) if idx == 0 else (tm, ts)
        ax.plot(qs, m, ls=ls, color=col, marker=mk, lw=lw, ms=ms, label=lab, zorder=3 if key == 'd1' else 2)
        ax.fill_between(qs, m - s, m + s, color=col, alpha=0.12, lw=0)
    ax.set_xlabel('questions asked $q$'); ax.set_ylabel(ylab); ax.set_title(title)
    ax.set_xticks(QS); ax.grid(True, alpha=0.3)
axes[0].legend(loc='lower right', fontsize=8, framealpha=0.9)
fig.suptitle('Cold-start NDCG@10 vs.\\ question budget (canonical ruler, seed-avg)', fontsize=11)
fig.tight_layout(rect=[0, 0, 1, 0.96])
for ext in ('pdf', 'png'):
    fig.savefig(f'{OUT}.{ext}', bbox_inches='tight', dpi=160); print('wrote', f'{OUT}.{ext}')

# ---- LaTeX tab:qcurve (tail) ----
print('\n% --- tab:qcurve TAIL (canonical, seed-avg) ---')
namemap = [('d1','Continuous actor + divisiveness (ours)'), ('cont','Continuous actor (continuity only)'),
           ('entdistill_ep4','CASPER-R'), ('entropy','Entropy'), ('conc_pop','Popularity')]
for key, lab in namemap:
    store = actors if key in ('d1','cont') else disc
    if key not in store: continue
    _, _, _, tm, _ = series(store, key)
    cells = ' & '.join(f'{v:.3f}' for v in tm)
    print(f'{lab} & {cells} \\\\')
