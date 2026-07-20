#!/usr/bin/env python
"""Immediate Paper C q-curve figure: CANONICAL discrete baselines (evalcks_qcurve.csv, native answers)
+ faithful actor curves (COMPARE4, proven approx canonical). Swapped to pure-canonical actors when ready."""
import csv
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE = 'C:/dev/phd/casper'
EVALCKS = f'{BASE}/data/movielens/.cache/evalcks_qcurve.csv'
OUT = 'C:/dev/phd/papers/paper3_casper/fig_qcurve_paperC'
QS = [0, 2, 4, 8]

# canonical discrete: seed,name,q,FULL,TAIL,...
disc = {}
with open(EVALCKS) as fh:
    for r in csv.reader(fh):
        if len(r) < 5: continue
        nm, q = r[1], int(r[2])
        if q not in QS: continue
        disc.setdefault(nm, {}).setdefault(q, []).append((float(r[3]), float(r[4])))

def dser(name):
    fm = np.array([np.mean([v[0] for v in disc[name][q]]) for q in QS])
    fs = np.array([np.std ([v[0] for v in disc[name][q]]) for q in QS])
    tm = np.array([np.mean([v[1] for v in disc[name][q]]) for q in QS])
    ts = np.array([np.std ([v[1] for v in disc[name][q]]) for q in QS])
    return fm, fs, tm, ts

# faithful actor curves (COMPARE4 preview, seed-avg) -- (full,tail) per q in QS
ACT = {
 'd1':   {'full':[0.3099,0.3307,0.3699,0.3780], 'fs':[0.0061,0.0029,0.0038,0.0032],
          'tail':[0.0808,0.1355,0.1645,0.1782], 'ts':[0.0040,0.0035,0.0036,0.0065]},
 'cont': {'full':[0.3099,0.3327,0.3664,0.3695], 'fs':[0.0061,0.0026,0.0049,0.0046],
          'tail':[0.0808,0.1296,0.1613,0.1651], 'ts':[0.0040,0.0041,0.0037,0.0039]},
}

SPEC = [
 ('d1',   'actor', 'Continuous actor + divisiveness (ours)', '#0b5394', '-',  'o', 2.4, 6),
 ('cont', 'actor', 'Continuous actor (continuity only)',     '#3d85c6', '--', 's', 1.8, 5),
 ('entdistill_ep4', 'disc', 'CASPER-R (discrete SOTA)',       '#cc0000', '-.', '^', 1.6, 5),
 ('entropy',  'disc', 'Entropy / divisiveness (discrete)',    '#e69138', ':',  'D', 1.4, 4),
 ('conc_pop', 'disc', 'Popularity (discrete)',                '#999999', ':',  'v', 1.2, 4),
]

fig, axes = plt.subplots(1, 2, figsize=(10, 4.0))
for ax, which, title, ylab in [(axes[0],'full','Full NDCG@10','NDCG@10 (full)'),
                               (axes[1],'tail','Long-tail NDCG@10','NDCG@10 (tail)')]:
    for key, kind, lab, col, ls, mk, lw, ms in SPEC:
        if kind == 'actor':
            m = np.array(ACT[key]['full' if which=='full' else 'tail'])
            s = np.array(ACT[key]['fs' if which=='full' else 'ts'])
        else:
            if key not in disc: continue
            fm, fs, tm, ts = dser(key)
            m, s = (fm, fs) if which=='full' else (tm, ts)
        ax.plot(QS, m, ls=ls, color=col, marker=mk, lw=lw, ms=ms, label=lab, zorder=3 if key=='d1' else 2)
        ax.fill_between(QS, m-s, m+s, color=col, alpha=0.12, lw=0)
    ax.set_xlabel('questions asked $q$'); ax.set_ylabel(ylab); ax.set_title(title)
    ax.set_xticks(QS); ax.grid(True, alpha=0.3)
axes[0].legend(loc='lower right', fontsize=8, framealpha=0.9)
fig.suptitle('Cold-start NDCG@10 vs. question budget (canonical ruler, seed-avg)', fontsize=11)
fig.tight_layout(rect=[0,0,1,0.96])
for ext in ('pdf','png'):
    fig.savefig(f'{OUT}.{ext}', bbox_inches='tight', dpi=160); print('wrote', f'{OUT}.{ext}')

# print the canonical discrete seed-avg for the table
print('\n% canonical discrete (seed-avg) TAIL @ q=0,2,4,8:')
for nm in ['entdistill_ep4','entropy','conc_pop']:
    if nm in disc:
        _,_,tm,_ = dser(nm); print(f'  {nm:>16}: ' + ' '.join(f'{v:.3f}' for v in tm))
