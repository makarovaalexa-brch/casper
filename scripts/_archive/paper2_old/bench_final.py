"""
Fair head-to-head: method (argmax, movies-only) vs popularity / HELF / thompson
/ random, ALL on the IDENTICAL hard-LOO cases (same seed -> same targets,
negatives, prior). Movie-scoped. This removes subset variance.
"""
import os, sys, time
sys.path.insert(0, '.'); sys.path.insert(0, 'scripts/paper1'); sys.path.insert(0, 'scripts/paper2')
import numpy as np, torch
from test_instrument_lib import load_instrument_by_name
import policies as P
from equivariant_actor import EquivariantActor

INST = 'instrument_ml1m_rank'; NPZ = 'C:/dev/phd/casper/data/movielens/ml1m_profiles.npz'
CK = 'C:/dev/phd/casper/experiments/paper2/method_ml1m.pt'
T = 15; N_NEG = 50; N_TEST = 200; SEED = 42


def main():
    w, _ = load_instrument_by_name(INST)
    d = np.load(NPZ, allow_pickle=True); train, test = d['train'], d['test']
    nt = int(d['n_targets']); ni = train.shape[1]; items = [tuple(x) for x in d['items'].tolist()]
    p_rated = (~np.isnan(train)).mean(0); pop = p_rated[:nt]
    cnt = (~np.isnan(train[:, :nt])).sum(0)
    fl = np.array([np.nan if cnt[e] == 0 else np.nanmean(train[:, e] == 1) for e in range(nt)])
    fc = np.clip(fl, 1e-6, 1 - 1e-6); H = -(fc*np.log2(fc) + (1-fc)*np.log2(1-fc))
    LF = np.log1p(cnt)/np.log1p(cnt.max()); helf = 2*LF*H/(LF+H+1e-9)
    helf_order = [int(e) for e in np.argsort(-np.nan_to_num(helf))]
    pop_order = [int(e) for e in np.argsort(-pop)]
    dec = np.zeros(nt, int); o = np.argsort(pop)
    for q in range(10): dec[o[q*nt//10:(q+1)*nt//10]] = q
    by = {q: set(np.where(dec == q)[0].tolist()) for q in range(10)}
    rng = np.random.default_rng(SEED); cases = []
    for prof in test:
        liked = np.where(prof[:nt] == 1)[0]; rated = set(np.where(~np.isnan(prof[:nt]))[0].tolist())
        if len(liked) < 3: continue
        tgt = int(liked[rng.integers(len(liked))]); negp = [i for i in by[dec[tgt]] if i not in rated]
        if len(negp) < N_NEG: continue
        cases.append((prof, tgt, np.array([tgt] + list(rng.choice(negp, N_NEG, replace=False)))))
        if len(cases) >= N_TEST: break

    actor = EquivariantActor(ni, 'dual'); actor.load_state_dict(torch.load(CK)['actor_state_dict']); actor.eval()

    def state_from(rev):
        bel = np.asarray(w.predict_full(rev)); rat = np.asarray(w.predict_rated(rev))
        s = np.zeros((ni, 3), np.float32); s[:, 2] = 1
        for (e, p) in rev: s[e, 2] = 0; s[e, 1 if p >= 0.5 else 0] = 1
        return np.concatenate([s.reshape(-1), bel.astype(np.float32), rat.astype(np.float32)])

    def hit(rev, cand):
        sc = np.asarray(w.predict(rev))[cand]; return 1.0 if 1+int((sc[1:] >= sc[0]).sum()) <= 10 else 0.0

    def run(name, sel_factory):
        t0 = time.time(); curves = []
        for (prof, tgt, cand) in cases:
            sel = sel_factory(); rev = []; asked = {tgt}; row = [hit([], cand)]
            for _ in range(T):
                q = sel(asked, rev)
                if q is None or q in asked: row.append(row[-1]); continue
                asked.add(q); v = prof[q]
                if not np.isnan(v): rev.append((int(q), float(v)))
                row.append(hit(rev, cand))
            curves.append(row)
        c = np.array(curves).mean(0)
        print(f"  {name:<16} turn0={c[0]:.3f} t5={c[5]:.3f} t15={c[-1]:.3f} AUC={c.mean():.4f} ({time.time()-t0:.0f}s)", flush=True)

    def method_sel():
        def s(asked, rev):
            with torch.no_grad():
                lg = actor(torch.from_numpy(state_from(rev)).unsqueeze(0))[0].numpy()
            lg[list(asked)] = -1e9; lg[nt:] = -1e9
            return int(np.argmax(lg))
        return s
    def order_sel(order):
        return lambda asked, rev: next((e for e in order if e not in asked), None)
    def rand_sel():
        r = np.random.default_rng(1)
        return lambda asked, rev: int(r.choice([i for i in range(nt) if i not in asked]))
    def thom_sel():
        pol = P.ThompsonPolicy(items); pol.reset(rng=np.random.default_rng(7)); pol.cand_pool = list(range(nt))
        return lambda asked, rev: pol.select(asked=asked, history=None, instrument=w, revealed=rev)

    print(f"=== FAIR head-to-head, identical {len(cases)} cases (movie-scoped) ===")
    run('METHOD(argmax)', method_sel)
    run('HELF', lambda: order_sel(helf_order))
    run('popularity', lambda: order_sel(pop_order))
    run('thompson', thom_sel)
    run('random', rand_sel)
    print("  ceiling (20 true reveals) ~0.647")


if __name__ == '__main__':
    main()
