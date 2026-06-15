"""
APPROPRIATE baselines for item cold-start elicitation, full-catalog, on the
hard popularity-matched LOO metric. Replaces the CRS-dialogue heuristics
(greedy_infogain/scpr) with the cold-start active-learning literature:

- random (full catalog)
- popularity (most-rated first)            [answerable, non-discriminative]
- HELF = harmonic(logPopularity, Entropy)  [Rashid et al. IUI 2002: answerable x discriminative]
- answer_uncertainty = P(rated) x p(1-p)   [answerability-weighted uncertainty, dual head]
- thompson (PEBOL-style Bayesian)

Usage: INST_NAME=instrument_ml1m_rank DATASET_NPZ=.../ml1m_profiles.npz \
       poetry run python scripts/paper1/bench_appropriate.py
"""
import os, sys, time
sys.path.insert(0, '.'); sys.path.insert(0, 'scripts/paper1')
import numpy as np
from test_instrument_lib import load_instrument_by_name
from testbed import SimulatedUser
import policies as P

NAME = os.environ.get('DATASET_NAME', 'ml1m')
INST = os.environ.get('INST_NAME', f'instrument_{NAME}_rank')
NPZ = os.environ.get('DATASET_NPZ', 'C:/dev/phd/casper/data/movielens/ml1m_profiles.npz')
T = 15; N_NEG = 50; N_USERS = 300; SEED = 42


def hit10(scores_cand):
    return 1.0 if 1 + int((scores_cand[1:] >= scores_cand[0]).sum()) <= 10 else 0.0


def main():
    w, _ = load_instrument_by_name(INST)
    d = np.load(NPZ, allow_pickle=True)
    train, test = d['train'], d['test']; nt = int(d['n_targets']); ni = train.shape[1]
    items = [tuple(x) for x in d['items'].tolist()]
    p_rated = (~np.isnan(train)).mean(0)                       # popularity / answerability
    # entropy per item: split among raters
    cnt = (~np.isnan(train)).sum(0)
    fl = np.array([np.nan if cnt[e] == 0 else np.nanmean(train[:, e] == 1) for e in range(ni)])
    fc = np.clip(fl, 1e-6, 1 - 1e-6)
    H = -(fc * np.log2(fc) + (1 - fc) * np.log2(1 - fc))       # rating entropy
    LF = np.log1p(cnt) / np.log1p(cnt.max())
    helf = 2 * LF * H / (LF + H + 1e-9)                        # harmonic mean
    MOVIES_ONLY = os.environ.get('MOVIES_ONLY', '1') == '1'
    keep = (lambda arr: [e for e in arr if e < nt]) if MOVIES_ONLY else (lambda arr: arr)
    helf_order = keep(list(np.argsort(-np.nan_to_num(helf))))
    pop_order = keep(list(np.argsort(-p_rated)))
    cand_items = list(range(nt)) if MOVIES_ONLY else list(range(ni))

    # pop-matched negatives
    pop = p_rated[:nt]; dec = np.zeros(nt, int); order = np.argsort(pop)
    for q in range(10):
        dec[order[q * nt // 10:(q + 1) * nt // 10]] = q
    by = {q: set(np.where(dec == q)[0].tolist()) for q in range(10)}
    rng = np.random.default_rng(SEED); cases = []
    for ui, prof in enumerate(test):
        liked = np.where(prof[:nt] == 1)[0]
        rated = set(np.where(~np.isnan(prof[:nt]))[0].tolist())
        if len(liked) < 3:
            continue
        tgt = int(liked[rng.integers(len(liked))])
        negpool = [i for i in by[dec[tgt]] if i not in rated]
        if len(negpool) < N_NEG:
            continue
        cases.append((ui, prof, tgt, np.array([tgt] + list(rng.choice(negpool, N_NEG, replace=False)))))
        if len(cases) >= N_USERS:
            break

    def run(selector_factory, name):
        t0 = time.time(); curves = []
        for (ui, prof, tgt, cand) in cases:
            sel = selector_factory(); user = SimulatedUser(ui, prof)
            revealed = []; asked = {tgt}; row = [hit10(np.asarray(w.predict([]))[cand])]
            for _ in range(T):
                q = sel(asked, revealed)
                if q is None or q in asked:
                    row.append(row[-1]); continue
                a = user.answer(q); asked.add(q)
                if a == 'liked': revealed.append((q, 1.0))
                elif a == 'disliked': revealed.append((q, 0.0))
                row.append(hit10(np.asarray(w.predict(revealed))[cand]))
            curves.append(row)
        cur = np.array(curves).mean(0)
        print(f"  {name:<20} turn0={cur[0]:.3f} t5={cur[5]:.3f} t15={cur[-1]:.3f} "
              f"AUC={cur.mean():.4f}  ({time.time()-t0:.0f}s)", flush=True)
        return [float(x) for x in cur]

    # selector factories (each returns a select(asked, revealed)->item over FULL catalog)
    def rand_fac():
        r = np.random.default_rng(123)
        return lambda asked, rev: int(r.choice([i for i in cand_items if i not in asked]))
    def order_fac(order):
        return lambda asked, rev: next((int(e) for e in order if e not in asked), None)
    def au_fac():
        def sel(asked, rev):
            p = np.asarray(w.predict_full(rev)); r = np.asarray(w.predict_rated(rev))
            score = r * p * (1 - p)
            if MOVIES_ONLY:
                score[nt:] = -2
            for e in asked:
                score[e] = -1
            return int(np.argmax(score))
        return sel
    def thom_fac():
        pol = P.ThompsonPolicy(items); pol.reset(rng=np.random.default_rng(7))
        if MOVIES_ONLY:
            pol.cand_pool = cand_items
        return lambda asked, rev: pol.select(asked=asked, history=None, instrument=w, revealed=rev)

    print(f"=== APPROPRIATE baselines, full-catalog hard-LOO ({INST}, {len(cases)} users) ===")
    res = {}
    res['random'] = run(rand_fac, 'random')
    res['popularity'] = run(lambda: order_fac(pop_order), 'popularity')
    res['HELF'] = run(lambda: order_fac(helf_order), 'HELF(pop x entropy)')
    res['answer_uncertainty'] = run(au_fac, 'answer_uncertainty')
    res['thompson'] = run(thom_fac, 'thompson')
    import json
    open(f'C:/dev/phd/casper/experiments/paper1/appropriate_{NAME}.json', 'w').write(json.dumps(res, indent=2))
    print("references: no-question prior 0.440 | 20 true reveals (ceiling) 0.647 | pop-rank floor 0.385")


if __name__ == '__main__':
    main()
