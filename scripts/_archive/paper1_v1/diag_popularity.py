"""
Diagnostic for the popularity puzzle: per-turn answer-rate (how many of the
asked questions the user could actually answer) and Hit@10, for popularity vs
random, on the hard pop-matched LOO metric. Full catalog.
"""
import os, sys
sys.path.insert(0, '.'); sys.path.insert(0, 'scripts/paper1')
import numpy as np
from test_instrument_lib import load_instrument_by_name

INST = os.environ.get('INST_NAME', 'instrument_ml1m_rank')
NPZ = os.environ.get('DATASET_NPZ', 'C:/dev/phd/casper/data/movielens/ml1m_profiles.npz')
T = 15; N_NEG = 50; N_USERS = 300; SEED = 42


def main():
    w, _ = load_instrument_by_name(INST)
    d = np.load(NPZ, allow_pickle=True)
    train, test = d['train'], d['test']; nt = int(d['n_targets']); ni = train.shape[1]
    pop = (~np.isnan(train[:, :nt])).mean(0)
    pop_order = list(np.argsort(-pop))           # movies+? -> use full p_rated
    p_all = (~np.isnan(train)).mean(0)
    pop_order = list(np.argsort(-p_all))
    dec = np.zeros(nt, int); order = np.argsort(pop)
    for q in range(10):
        dec[order[q * nt // 10:(q + 1) * nt // 10]] = q
    by = {q: set(np.where(dec == q)[0].tolist()) for q in range(10)}
    rng = np.random.default_rng(SEED); cases = []
    for prof in test:
        liked = np.where(prof[:nt] == 1)[0]; rated = set(np.where(~np.isnan(prof[:nt]))[0].tolist())
        if len(liked) < 3:
            continue
        tgt = int(liked[rng.integers(len(liked))])
        negpool = [i for i in by[dec[tgt]] if i not in rated]
        if len(negpool) < N_NEG:
            continue
        cases.append((prof, tgt, np.array([tgt] + list(rng.choice(negpool, N_NEG, replace=False)))))
        if len(cases) >= N_USERS:
            break

    def hit(rev, cand):
        s = np.asarray(w.predict(rev))[cand]
        return 1.0 if 1 + int((s[1:] >= s[0]).sum()) <= 10 else 0.0

    def run(kind):
        ans = np.zeros(T + 1); hits = np.zeros(T + 1); n = len(cases)
        for (prof, tgt, cand) in cases:
            rev = []; asked = {tgt}; na = 0
            hits[0] += hit([], cand)
            r = np.random.default_rng(7)
            for t in range(1, T + 1):
                if kind == 'popularity':
                    q = next((e for e in pop_order if e not in asked), None)
                else:
                    q = int(r.choice([e for e in range(ni) if e not in asked]))
                asked.add(q); v = prof[q]
                if not np.isnan(v):
                    na += 1; rev.append((int(q), float(v)))
                ans[t] += na; hits[t] += hit(rev, cand)
        return ans / n, hits / n

    pa, ph = run('popularity'); ra, rh = run('random')
    print(f"=== popularity vs random: per-turn answered & Hit@10 ({INST}) ===")
    print(f"{'turn':>4} | {'pop#ans':>8} {'popHit@10':>10} | {'rnd#ans':>8} {'rndHit@10':>10}")
    for t in range(T + 1):
        print(f"{t:>4} | {pa[t]:>8.2f} {ph[t]:>10.3f} | {ra[t]:>8.2f} {rh[t]:>10.3f}")


if __name__ == '__main__':
    main()
