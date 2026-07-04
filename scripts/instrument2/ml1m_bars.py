"""
ml1m_bars.py -- honest bars on the canonical ML-1M arena (same protocol as ml1m_recvae):
  (1) V1 FIDELITY CHECK: load the V1 enc_concept recommender + Ql_concept and reproduce
      the V1 reference numbers (MOSTPOP q0 ~0.310, full-profile ~0.407 full / ~0.216 tail).
      This validates the arena is byte-identical to the V1 papers before any comparison.
  (2) EASE (Steck 2019): closed-form item-item, ni=3706 dense (trivially feasible).
      lambda sweep on val (te[:300]); test at val-selected lambda.
  (3) iALS (implicit ALS, Hu et al. 2008): cheap latent-factor bar.
All eval on te[300:] test cohort, held-out disjoint targets, NDCG@10 full+tail.
Usage: python scripts/instrument2/ml1m_bars.py --mode all
"""
import os, sys, json, time, argparse
import numpy as np
import torch, torch.nn as nn
from scipy import sparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ml1m_arena as A

BASE = A.BASE


class Enc(nn.Module):
    def __init__(s, D=64):
        super().__init__()
        s.inp = nn.Sequential(nn.Linear(D + 1, 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU())
        s.att = nn.Linear(128, 1); s.val = nn.Linear(128, D)

    def forward(s, t, m):
        h = s.inp(t); a = s.att(h).squeeze(-1).masked_fill(m == 0, -1e9)
        al = torch.softmax(a, 1); return (al.unsqueeze(-1) * s.val(h)).sum(1)


def load_v1(ar):
    D = 64
    Q = np.load(f'{BASE}/.cache/Q_svd.npy'); bi = np.load(f'{BASE}/.cache/bi_svd.npy')
    Ql = np.load(f'{BASE}/.cache/Ql_concept.npy')
    enc = Enc(D); enc.load_state_dict(torch.load(f'{BASE}/.cache/enc_concept.pt')); enc.eval()
    for p in enc.parameters():
        p.requires_grad_(False)
    # mu over trU ratings (matches continuous_policy2_st)
    sm = c = 0.0
    for x in ar['trU']:
        for j, r in ar['rat_by_u'][x]:
            sm += r; c += 1
    mu = sm / c
    resid = {x: {j: (r - mu - bi[j]) for j, r in ar['rat_by_u'][x]} for x in (ar['trU'] + ar['te'])}

    def enc_u(toks):
        if not toks:
            return np.zeros(D)
        arr = np.zeros((1, len(toks), D + 1), np.float32); m = np.ones((1, len(toks)), np.float32)
        for q, (f, v) in enumerate(toks):
            arr[0, q, :D] = f; arr[0, q, D] = v
        with torch.no_grad():
            return enc(torch.tensor(arr), torch.tensor(m)).numpy()[0]
    return Q, Ql, resid, enc_u


def v1_fidelity(ar):
    Q, Ql, resid, enc_u = load_v1(ar)
    popb = ar['popb'].astype(np.float64)
    users = ar['test_users']
    # MOSTPOP q0
    mp = A_eval(ar, users, lambda x, profset: popb.copy())
    # full-profile fold through V1 encoder
    def v1score(x, profset):
        u = enc_u([(Q[j], resid[x][j]) for j in profset])
        return (popb + Ql @ u).astype(np.float64)
    fp = A_eval(ar, users, v1score)
    print('=== V1 FIDELITY (te[300:]) ===', flush=True)
    print(f'  MOSTPOP q0     full {mp[0]:.4f} tail {mp[1]:.4f}   (V1 ref ~0.310)', flush=True)
    print(f'  V1 full-profile full {fp[0]:.4f} tail {fp[1]:.4f}   (V1 ref ~0.407 / ~0.216)', flush=True)
    return mp, fp


def A_eval(ar, users, score_fn):
    accf = acct = 0.0; mf = mt = 0
    rat = {x: dict(v) for x, v in ar['rat_by_u'].items()}
    for x in users:
        profset, test = ar['SPL'][x]; rd = rat[x]
        tlike = set(j for j in test if rd[j] >= 4)
        if not tlike:
            continue
        s = score_fn(x, profset)
        nf = A.ndcg_at10(s, tlike, profset, ar['headmask'], False)
        if nf is not None:
            accf += nf; mf += 1
        nt = A.ndcg_at10(s, tlike, profset, ar['headmask'], True)
        if nt is not None:
            acct += nt; mt += 1
    return (accf / max(mf, 1), acct / max(mt, 1), mf, mt)


def build_like_matrix(ar, users):
    rows, cols = [], []
    for r, x in enumerate(users):
        for j in ar['likes_by_u'].get(x, []):
            rows.append(r); cols.append(j)
    return sparse.csr_matrix((np.ones(len(rows), np.float32), (rows, cols)),
                             dtype=np.float32, shape=(len(users), ar['ni']))


def ease_B(X, lam):
    G = np.asarray((X.T @ X).todense(), dtype=np.float64)
    G[np.diag_indices_from(G)] += lam
    P = np.linalg.inv(G); d = np.diag(P).copy()
    B = -P / d[None, :]; B[np.diag_indices_from(B)] = 0.0
    return B


def _ease_scorer(B, ar, rat):
    def f(x, profset):
        likes = [j for j in profset if rat[x][j] >= 4]
        return B[likes].sum(0) if likes else np.full(ar['ni'], -1e30)
    return f


def ease_bar(ar, eval_seeds):
    rat = {x: dict(v) for x, v in ar['rat_by_u'].items()}
    X = build_like_matrix(ar, ar['trU'])  # seed-independent (trU=rng(0))
    print(f'=== EASE (ni={ar["ni"]}, train X nnz={X.nnz}) ===', flush=True)
    best = None
    for lam in [1.0, 10.0, 100.0, 500.0, 1000.0]:
        B = ease_B(X, lam)
        vf, vt, _, _ = A_eval(ar, ar['val_users'], _ease_scorer(B, ar, rat))
        print(f'  [val lam={lam:>6.0f}] full {vf:.4f} tail {vt:.4f}', flush=True)
        if best is None or vf > best[1]:
            best = (lam, vf, B)
    lam, _, B = best
    tfs, tts = [], []
    for sd in eval_seeds:
        ars = A.load_arena(seed=sd)
        tf, tt, _, _ = A_eval(ars, ars['test_users'], _ease_scorer(B, ars, rat))
        tfs.append(tf); tts.append(tt)
    tf, tt = float(np.mean(tfs)), float(np.mean(tts))
    print(f'  >>> val-selected lam={lam}; TEST seed-avg{eval_seeds} full {tf:.4f} tail {tt:.4f}', flush=True)
    return {'lam': lam, 'test_full': tf, 'test_tail': tt}


def ials_bar(ar, eval_seeds, d=64, reg=0.1, alpha=40.0, iters=15):
    """Hu et al. 2008 implicit ALS on trU like-matrix. Cheap; ni=3706, users~4827."""
    rat = {x: dict(v) for x, v in ar['rat_by_u'].items()}
    X = build_like_matrix(ar, ar['trU']).tocsr()
    nU, nI = X.shape
    rng = np.random.default_rng(0)
    Uf = rng.standard_normal((nU, d)).astype(np.float64) * 0.01
    Vf = rng.standard_normal((nI, d)).astype(np.float64) * 0.01
    Xt = X.T.tocsr()

    def als_step(P_csr, W, Wt_other):  # solve W given fixed other factors
        Y = Wt_other; YtY = Y.T @ Y; reg_eye = reg * np.eye(d)
        out = np.zeros((P_csr.shape[0], d))
        for i in range(P_csr.shape[0]):
            st, en = P_csr.indptr[i], P_csr.indptr[i + 1]
            idx = P_csr.indices[st:en]
            if len(idx) == 0:
                continue
            cu = alpha  # confidence for positives = 1 + alpha*1
            Yi = Y[idx]
            A_ = YtY + (Yi.T * cu) @ Yi + reg_eye
            b_ = (Yi * (1 + cu)).sum(0)
            out[i] = np.linalg.solve(A_, b_)
        return out
    for it in range(iters):
        Uf = als_step(X, Uf, Vf)
        Vf = als_step(Xt, Vf, Uf)
    print(f'=== iALS (d={d}, reg={reg}, alpha={alpha}, {iters} iters) ===', flush=True)
    YtY_all = Vf.T @ Vf

    def make_scorer(ratx):
        def scorer(x, profset):
            likes = [j for j in profset if ratx[x][j] >= 4]
            if not likes:
                return np.full(ar['ni'], -1e30)
            Yi = Vf[likes]
            A_ = YtY_all + (Yi.T * alpha) @ Yi + reg * np.eye(Vf.shape[1])
            u = np.linalg.solve(A_, (Yi * (1 + alpha)).sum(0))
            return Vf @ u
        return scorer
    tfs, tts = [], []
    for sd in eval_seeds:
        ars = A.load_arena(seed=sd)
        ratx = {x: dict(v) for x, v in ars['rat_by_u'].items()}
        tf, tt, _, _ = A_eval(ars, ars['test_users'], make_scorer(ratx))
        tfs.append(tf); tts.append(tt)
    tf, tt = float(np.mean(tfs)), float(np.mean(tts))
    print(f'  >>> TEST seed-avg{eval_seeds} full {tf:.4f} tail {tt:.4f}', flush=True)
    return {'d': d, 'test_full': tf, 'test_tail': tt}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', default='all', help='v1|ease|ials|all')
    ap.add_argument('--seed', type=int, default=123)
    args = ap.parse_args()
    ar = A.load_arena(seed=args.seed)
    eval_seeds = [int(s) for s in os.environ.get('SEEDS', '1,2,3,7,11').split(',')]
    out = {'eval_seeds': eval_seeds}
    if args.mode in ('v1', 'all'):
        # seed-averaged fidelity
        mpf = mpt = fpf = fpt = 0.0
        Q, Ql, resid, enc_u = load_v1(ar)
        for sd in eval_seeds:
            ars = A.load_arena(seed=sd); popb = ars['popb'].astype(np.float64)
            mp = A_eval(ars, ars['test_users'], lambda x, ps: popb.copy())
            fp = A_eval(ars, ars['test_users'],
                        lambda x, ps: (popb + Ql @ enc_u([(Q[j], resid[x][j]) for j in ps])).astype(np.float64))
            mpf += mp[0]; mpt += mp[1]; fpf += fp[0]; fpt += fp[1]
        n = len(eval_seeds)
        print(f'=== V1 FIDELITY seed-avg{eval_seeds} (te[300:]) ===', flush=True)
        print(f'  MOSTPOP q0     full {mpf/n:.4f} tail {mpt/n:.4f}   (V1 ref ~0.310)', flush=True)
        print(f'  V1 full-profile full {fpf/n:.4f} tail {fpt/n:.4f}   (V1 ref ~0.407 / ~0.216)', flush=True)
        out['mostpop'] = {'full': mpf / n, 'tail': mpt / n}
        out['v1_fullprofile'] = {'full': fpf / n, 'tail': fpt / n}
    if args.mode in ('ease', 'all'):
        out['ease'] = ease_bar(ar, eval_seeds)
    if args.mode in ('ials', 'all'):
        out['ials'] = ials_bar(ar, eval_seeds)
    json.dump(out, open(os.path.join('.cache', 'instrument2', 'ml1m_bars.json'), 'w'), indent=2)
    print('[saved] ml1m_bars.json', flush=True)


if __name__ == '__main__':
    main()
