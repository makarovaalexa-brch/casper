"""
Core experiment: evaluate ELICITATION POLICIES under the honest hard-LOO
protocol. For each test user we hold out one liked target and 50 popularity-
matched negatives; the policy asks T questions (never the target); after each
turn we rank target-vs-negatives with the instrument -> Hit@10 curve per turn.
A good policy lifts the target into the top-10 in FEWER turns.

Baselines: random, popularity, greedy_infogain, scpr_entropy,
greedy_answerability, thompson. (Learned method benchmarked separately once
trained on this instrument.)

Usage: INST_NAME=instrument_ml1m_rank DATASET_NPZ=.../ml1m_profiles.npz \
       poetry run python scripts/paper1/bench_policies_loo.py
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
T = 15; N_NEG = 50; N_USERS = 300; TOPK_MOVIES = 300; SEED = 42


def main():
    w, _ = load_instrument_by_name(INST)
    d = np.load(NPZ, allow_pickle=True)
    train, test = d['train'], d['test']; nt = int(d['n_targets']); ni = train.shape[1]
    items = [tuple(x) for x in d['items'].tolist()]
    p_rated = (~np.isnan(train)).mean(0)
    # popularity deciles for matched negatives
    pop = p_rated[:nt]; dec = np.zeros(nt, int); order = np.argsort(pop)
    for q in range(10):
        dec[order[q * nt // 10:(q + 1) * nt // 10]] = q
    by = {q: set(np.where(dec == q)[0].tolist()) for q in range(10)}
    # candidate ASK pool (speed): attributes + top popular movies
    attrs = list(range(nt, ni))
    toppop = [int(m) for m in np.argsort(-pop)[:TOPK_MOVIES]]
    ask_pool = sorted(set(attrs) | set(toppop))

    rng = np.random.default_rng(SEED)
    # precompute per-user target + negatives + ask pool (shared across policies)
    cases = []
    for ui, prof in enumerate(test[:N_USERS * 2]):
        liked = np.where(prof[:nt] == 1)[0]
        rated = set(np.where(~np.isnan(prof[:nt]))[0].tolist())
        if len(liked) < 3:
            continue
        tgt = int(liked[rng.integers(len(liked))])
        negpool = [i for i in by[dec[tgt]] if i not in rated]
        if len(negpool) < N_NEG:
            continue
        negs = list(rng.choice(negpool, N_NEG, replace=False))
        cases.append((ui, prof, tgt, np.array([tgt] + negs)))
        if len(cases) >= N_USERS:
            break

    pols = {'random': lambda: P.RandomPolicy(items),
            'popularity': lambda: P.PopularityPolicy(items, p_rated),
            'greedy_infogain': lambda: P.GreedyInfoGainPolicy(items, p_rated),
            'scpr_entropy': lambda: P.SCPREntropyPolicy(items, p_rated),
            'greedy_answerability': lambda: P.GreedyAnswerabilityPolicy(items),
            'thompson': lambda: P.ThompsonPolicy(items)}

    print(f"=== policy hard-LOO Hit@10 vs turns ({INST}, {len(cases)} users, {N_NEG} neg) ===")
    results = {}
    for name, fac in pols.items():
        t0 = time.time(); curves = []
        for (ui, prof, tgt, cand) in cases:
            pol = fac(); pol.cand_pool = [c for c in ask_pool if c != tgt]
            pol.reset(rng=np.random.default_rng(SEED * 7 + ui))
            user = SimulatedUser(ui, prof)
            revealed = []; asked = {tgt}; qs = []; ans = []
            row = []
            s = np.asarray(w.predict(revealed))[cand]
            row.append(1.0 if 1 + int((s[1:] >= s[0]).sum()) <= 10 else 0.0)
            for _ in range(T):
                q = pol.select(asked=asked, history=list(zip(qs, ans)),
                               instrument=w, revealed=list(revealed))
                if q is None or q in asked:
                    row.append(row[-1]); continue
                a = user.answer(q); asked.add(q); qs.append(q); ans.append(a)
                if a == 'liked':
                    revealed.append((q, 1.0))
                elif a == 'disliked':
                    revealed.append((q, 0.0))
                s = np.asarray(w.predict(revealed))[cand]
                row.append(1.0 if 1 + int((s[1:] >= s[0]).sum()) <= 10 else 0.0)
            curves.append(row)
        cur = np.array(curves).mean(0)
        results[name] = cur
        print(f"  {name:<22} turn0={cur[0]:.3f} turn5={cur[5]:.3f} turn15={cur[-1]:.3f} "
              f"AUC={cur.mean():.4f}  ({time.time()-t0:.0f}s)", flush=True)
    import json
    out = {k: [float(x) for x in v] for k, v in results.items()}
    open(f'C:/dev/phd/casper/experiments/paper1/policy_loo_{NAME}.json', 'w').write(json.dumps(out, indent=2))
    print("saved policy_loo json")


if __name__ == '__main__':
    main()
