"""
Efficiency / speedup analysis on the synthetic indicator world (T=5).

Synthetic runs stored only summaries, so we re-evaluate the non-learned
policies (instant, deterministic given the cached instrument) to capture
per-turn curves, then compute turns-to-match-random and speed-up — the
same framing as the real slates. Learned policies (PPO/REINFORCE/DQN) are
omitted here because their synthetic nets were not persisted; their
summary AUAC/final are already in synthetic_sanity.json.

Run: poetry run python scripts/paper1/synthetic_efficiency.py
"""

import sys
sys.path.insert(0, '.')
sys.path.insert(0, 'scripts/paper1')

import json
from pathlib import Path

import numpy as np

import synthetic_sanity as W
import synthetic_rl_improved as R
import synthetic_baselines as B

EXP = Path('C:/dev/phd/casper/experiments/paper1')
TAB = Path('C:/dev/phd/papers/paper1_casper/tables')
N = W.N_TURNS  # 5
SEED = 42


def curve(select_fn, instrument, profiles):
    C = np.full((len(profiles), N + 1), np.nan)
    for i, prof in enumerate(profiles):
        accs, _, _ = W.run_episode(select_fn, prof, instrument, N)
        C[i] = accs[:N + 1]
    return C.mean(0)


def main():
    instrument = R.load_instrument()
    eval_profiles = W.gen_users(W.N_EVAL_USERS, np.random.default_rng(SEED + 6))
    train_profiles = W.gen_users(2000, np.random.default_rng(SEED))
    vecs = np.stack(train_profiles)
    p_rated = (~np.isnan(vecs)).mean(0)

    policies = {
        'oracle_adaptive': W.oracle_adaptive_select(),
        'random': W.random_select(np.random.default_rng(SEED + 3)),
        'static_oracle': W.static_oracle_select(),
        'greedy_infogain': W.greedy_select(answerability=False),
        'greedy_answerability': W.greedy_select(answerability=True),
        'popularity': B.make_popularity(p_rated),
        'scpr_entropy': B.make_scpr(p_rated),
        'thompson': B.make_thompson(),
    }
    curves = {k: curve(fn, instrument, eval_profiles) for k, fn in policies.items()}
    prior = float(curves['random'][0])
    rfinal = float(curves['random'][-1])

    rows = []
    for k, c in curves.items():
        hit = next((t for t in range(N + 1) if c[t] >= rfinal), None)
        rows.append((k, float(np.mean(c)), float(c[-1]), hit,
                     (N / hit) if hit else None))
    rows.sort(key=lambda r: -r[1])

    L = [f"### Synthetic indicator world (T={N}, prior={prior:.3f}, "
         f"random@{N}={rfinal:.3f})", "",
         "| Policy | AUAC | final | turns→random | speed-up |",
         "|---|---|---|---|---|"]
    for k, au, fin, hit, sp in rows:
        ts = str(hit) if hit is not None else '—'
        sps = (f"{sp:.1f}x" if sp else '—')
        L.append(f"| {k} | {au:.3f} | {fin:.3f} | {ts} | {sps} |")
    md = "\n".join(L)
    (TAB / 'efficiency_synthetic.md').write_text(md, encoding='utf-8')

    # merge into efficiency.json
    effp = EXP / 'efficiency.json'
    allout = json.loads(effp.read_text()) if effp.exists() else {}
    allout['synthetic'] = {k: {'auac': float(np.mean(c)), 'final': float(c[-1]),
                               'curve': [round(float(x), 4) for x in c]}
                           for k, c in curves.items()}
    effp.write_text(json.dumps(allout, indent=2, default=str))
    print(md)


if __name__ == '__main__':
    main()
