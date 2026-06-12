"""
Replay episode logs through the instrument to backfill metrics without
re-running any policy (no API calls, no retraining):
  per-turn accuracy, NDCG@10, Hit@10; AUAC@{5,10,15}; answer rate
  (renamed from 'hit rate' to avoid collision with Hit@k).

Run from casper root: poetry run python scripts/paper1/recompute_metrics.py
Outputs: experiments/paper1/metrics_by_turn.json (+ markdown table
         papers/paper1_casper/tables/by_turn.md), updates
         benchmark_results.json rows in place.
"""

import sys
sys.path.insert(0, '.')
sys.path.insert(0, 'scripts/paper1')

import json
from pathlib import Path

import numpy as np

from test_instrument_lib import load_instrument_by_name
from testbed import build_profiles

EXP = Path('C:/dev/phd/casper/experiments/paper1')
TAB = Path('C:/dev/phd/papers/paper1_casper/tables')
TAB.mkdir(parents=True, exist_ok=True)
N_TURNS = 15


def ndcg_hit(preds, profile, n_movies, k=10):
    gt = profile[:n_movies]
    liked = np.where(gt == 1.0)[0]
    if len(liked) == 0:
        return np.nan, np.nan
    order = np.argsort(-preds[:n_movies])[:k]
    rel = np.isin(order, liked).astype(float)
    dcg = float((rel / np.log2(np.arange(2, k + 2))).sum())
    ideal = min(len(liked), k)
    idcg = float((1 / np.log2(np.arange(2, ideal + 2))).sum())
    return dcg / idcg, float(rel.any())


def main():
    instrument, ckpt = load_instrument_by_name('instrument_v5_set')
    items = ckpt['items']
    n_movies = instrument.n_movies

    uids = set()
    files = sorted(EXP.glob('episodes_*.jsonl'))
    episodes = {}
    for f in files:
        pname = f.stem.replace('episodes_', '')
        eps = [json.loads(l) for l in f.read_text().splitlines() if l.strip()]
        episodes[pname] = eps
        uids.update(e['uid'] for e in eps)
    profiles = build_profiles(items, user_ids=np.array(sorted(uids)),
                              attr_min_support=3, taste_margin=None)
    print(f"{len(files)} policies, {len(profiles)} profiles")

    results_path = EXP / 'benchmark_results.json'
    results = json.loads(results_path.read_text())
    by_turn = {}

    for pname, eps in episodes.items():
        acc = np.full((len(eps), N_TURNS + 1), np.nan)
        ndcg = np.full((len(eps), N_TURNS + 1), np.nan)
        hit = np.full((len(eps), N_TURNS + 1), np.nan)
        ans = []
        for i, e in enumerate(eps):
            prof = profiles.get(e['uid'])
            if prof is None:
                continue
            revealed = []
            # turn 0 prior
            preds = instrument.predict(revealed)
            n, h = ndcg_hit(preds, prof, n_movies)
            acc[i, 0] = e['accuracy'][0]
            ndcg[i, 0], hit[i, 0] = n, h
            for t, (q, a) in enumerate(zip(e['questions'], e['answers'])):
                if a == 'liked':
                    revealed.append((q, 1.0))
                elif a == 'disliked':
                    revealed.append((q, 0.0))
                preds = instrument.predict(revealed)
                n, h = ndcg_hit(preds, prof, n_movies)
                tt = t + 1
                acc[i, tt] = (e['accuracy'][tt]
                              if tt < len(e['accuracy']) else e['accuracy'][-1])
                ndcg[i, tt], hit[i, tt] = n, h
            # pad curves for episodes shorter than N_TURNS
            last = len(e['questions'])
            for tt in range(last + 1, N_TURNS + 1):
                acc[i, tt] = acc[i, last]
                ndcg[i, tt] = ndcg[i, last]
                hit[i, tt] = hit[i, last]
            if e['answers']:
                ans.append(np.mean([a != 'unknown' for a in e['answers']]))

        m_acc = np.nanmean(acc, axis=0)
        m_ndcg = np.nanmean(ndcg, axis=0)
        m_hit = np.nanmean(hit, axis=0)
        by_turn[pname] = {
            'accuracy': [round(float(x), 4) for x in m_acc],
            'ndcg10': [round(float(x), 4) for x in m_ndcg],
            'hit10': [round(float(x), 4) for x in m_hit],
        }
        upd = {
            'answer_rate': float(np.mean(ans)) if ans else np.nan,
            'auac@5': float(np.nanmean(acc[:, :6])),
            'auac@10': float(np.nanmean(acc[:, :11])),
            'auac@15': float(np.nanmean(acc)),
            'ndcg10_final': float(m_ndcg[-1]),
            'ndcg10_turn0': float(m_ndcg[0]),
            'hit10_final': float(m_hit[-1]),
        }
        if pname in results:
            results[pname].update(upd)
        else:
            results[pname] = upd
        print(f"{pname:<28} auac@5={upd['auac@5']:.4f} @10={upd['auac@10']:.4f} "
              f"@15={upd['auac@15']:.4f} ndcg10={upd['ndcg10_final']:.4f} "
              f"ans={upd['answer_rate']:.0%}")

    results_path.write_text(json.dumps(results, indent=2))
    (EXP / 'metrics_by_turn.json').write_text(json.dumps(by_turn, indent=2))

    # markdown by-turn accuracy table (turns 0,1,2,3,5,10,15)
    cols = [0, 1, 2, 3, 5, 10, 15]
    lines = ['| policy | ' + ' | '.join(f't{c}' for c in cols) + ' |',
             '|' + '---|' * (len(cols) + 1)]
    order = sorted(by_turn, key=lambda p: -np.mean(by_turn[p]['accuracy']))
    for p in order:
        a = by_turn[p]['accuracy']
        lines.append(f"| {p} | " + ' | '.join(f"{a[c]:.3f}" for c in cols) + ' |')
    md = '\n'.join(lines)
    (TAB / 'by_turn.md').write_text(md)
    print('\nBY-TURN ACCURACY (also saved to tables/by_turn.md):\n')
    print(md)


if __name__ == '__main__':
    main()
