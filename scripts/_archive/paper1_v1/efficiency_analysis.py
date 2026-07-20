"""
Elicitation EFFICIENCY analysis (the right way to size elicitation value):
turns-to-match-random, per-budget AUAC, and speedup factors, from episode
logs. Re-run any time new results land.

Outputs:
  experiments/paper1/efficiency.json
  papers/paper1_casper/tables/efficiency_slate{1,2}.md
  papers/paper1_casper/figures/fig_efficiency_slate{1,2}.{pdf,png}

Run: poetry run python scripts/paper1/efficiency_analysis.py
"""

import sys
sys.path.insert(0, '.')
sys.path.insert(0, 'scripts/paper1')

import glob
import json
import os
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

EXP = Path('C:/dev/phd/casper/experiments/paper1')
TAB = Path('C:/dev/phd/papers/paper1_casper/tables')
FIG = Path('C:/dev/phd/papers/paper1_casper/figures')
for d in (TAB, FIG):
    d.mkdir(parents=True, exist_ok=True)
N = 15

SLATES = {
    'slate1': ('episodes_', 'Slate 1 (top-100 movies)'),
    'slate2': ('s2_episodes_', 'Slate 2 (mid-popularity + genome tags)'),
}
# display order / friendly names
NAME = {
    'random': 'Random', 'popularity': 'Popularity',
    'greedy_infogain': 'Greedy info-gain', 'scpr_entropy': 'SCPR-entropy',
    'thompson': 'Thompson', 'dqn': 'DQN', 'ppo': 'PPO', 'ppo_dual': 'PPO+ans',
    'botplay_rl': 'Bot-play', 'botplay_rl_v2': 'Bot-play v2',
    'botplay': 'Bot-play', 'botplay_dual': 'Bot-play+ans',
}
HILITE = ['random', 'popularity', 'scpr_entropy', 'ppo_dual', 'botplay_dual', 'ppo', 'botplay']


def curve(path):
    eps = [json.loads(l) for l in open(path) if l.strip()]
    C = np.full((len(eps), N + 1), np.nan)
    for i, e in enumerate(eps):
        a = e['accuracy'] + [e['accuracy'][-1]] * (N + 1 - len(e['accuracy']))
        C[i] = a[:N + 1]
    return C.mean(0), len(eps)


def analyse(prefix):
    out = {}
    for f in sorted(glob.glob(str(EXP / f'{prefix}*.jsonl'))):
        pol = os.path.basename(f)[len(prefix):-6]
        c, n = curve(f)
        out[pol] = {'curve': c, 'n': n}
    if 'random' not in out:
        return out, None
    prior = float(out['random']['curve'][0])
    rand_final = float(out['random']['curve'][-1])
    for pol, d in out.items():
        c = d['curve']
        hit = next((t for t in range(N + 1) if c[t] >= rand_final), None)
        d['final'] = float(c[-1])
        d['auac5'] = float(np.mean(c[:6]))
        d['auac10'] = float(np.mean(c[:11]))
        d['auac15'] = float(np.mean(c))
        d['turns_to_random'] = hit
        d['speedup'] = (N / hit) if hit and hit > 0 else (float('inf') if c[-1] >= rand_final else None)
        d['gain'] = float(c[-1] - prior)
        d['pct_random_gain'] = float((c[-1] - prior) / (rand_final - prior) * 100) if rand_final > prior else None
    return out, prior


def md_table(out, prior, title):
    rows = sorted([k for k in out if k != '_'], key=lambda k: -out[k]['auac15'])
    L = [f"### {title}  (prior={prior:.3f})", "",
         "| Policy | AUAC@5 | AUAC@15 | turns→random@15 | speedup | %info vs random |",
         "|---|---|---|---|---|---|"]
    for k in rows:
        d = out[k]
        tt = d['turns_to_random']
        tt_s = str(tt) if tt is not None else '—'
        sp = ('∞' if d['speedup'] == float('inf')
              else (f"{d['speedup']:.1f}×" if d['speedup'] else '—'))
        pg = f"{d['pct_random_gain']:.0f}%" if d['pct_random_gain'] is not None else '—'
        L.append(f"| {NAME.get(k,k)} | {d['auac5']:.4f} | {d['auac15']:.4f} | {tt_s} | {sp} | {pg} |")
    return "\n".join(L)


def figure(out, prior, title, path):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    rand_final = out['random']['curve'][-1] if 'random' in out else None
    for k in [p for p in HILITE if p in out]:
        ax.plot(range(N + 1), out[k]['curve'], lw=1.8, label=NAME.get(k, k))
    if rand_final is not None:
        ax.axhline(rand_final, ls=':', c='grey', lw=1)
        ax.text(0.2, rand_final + 0.001, "random @15", fontsize=7, c='grey')
    ax.set_xlabel('Conversational turn'); ax.set_ylabel('Mean accuracy')
    ax.set_title(title, fontsize=10); ax.legend(fontsize=8, ncol=2); ax.grid(alpha=0.3)
    for ext in ('pdf', 'png'):
        fig.savefig(f'{path}.{ext}', bbox_inches='tight', dpi=200)
    plt.close(fig)


def main():
    allout = {}
    for slate, (prefix, title) in SLATES.items():
        out, prior = analyse(prefix)
        if not out or prior is None:
            print(f"[{slate}] no data yet"); continue
        md = md_table(out, prior, title)
        (TAB / f'efficiency_{slate}.md').write_text(md, encoding='utf-8')
        figure(out, prior, title, FIG / f'fig_efficiency_{slate}')
        allout[slate] = {k: {m: v for m, v in d.items() if m != 'curve'}
                         for k, d in out.items()}
        print(md); print()
    (EXP / 'efficiency.json').write_text(json.dumps(allout, indent=2, default=str))
    print(f"Saved efficiency.json, tables/, figures/")


if __name__ == '__main__':
    main()
