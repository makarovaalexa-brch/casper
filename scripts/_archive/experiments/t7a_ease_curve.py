"""
t7a_ease_curve.py -- REVIEW RESPONSE T7a: EASE vs I2 (RecVAE-d512) COLD-START curves.

Reviewer charge: "the EASE == I2 tie is warm-start (full-profile) only." This script tests
whether the tie also holds under COLD-START elicitation with few revealed items (k=1..8).

For each user we reveal k liked items and score with the SAME k-item history two ways:
  I2   : fold the k liked items through the frozen RecVAE-d512 encoder -> decode -> NDCG@10.
         (identical mechanism to P4a's `item8_fold` arm; item-fold, NOT the additive actor.)
  EASE : cold-start EASE score = B[revealed_k].sum(0), B = trU-trained item-item, val-selected
         lambda (identical closed form to ml1m_bars.ease_*; same B, restricted history).

Item-selection convention: matches P4a's k=8 item-fold arm, which uses
`rng = np.random.default_rng(sd)` random selection of profile likes. For a NESTED k=1..8 curve
we draw ONE random permutation of each user's profile likes (seeded per eval seed) and take the
first k. When a user has < k profile likes the fold saturates on all available likes (exactly as
P4a's item8_fold only trims when len>8).

Protocol (frozen invariants): instrument .cache/instrument2/ml1m_recvae_d512_best.pt; arena
ml1m_arena byte-identical splits; eval seeds {1,2,3,7,11}; te[300:] TEST (304 users);
NDCG@10 full + Cremonesi tail. Paired I2-EASE bootstrap reuses p4a_bootstrap.boot.

Usage:  python scripts/instrument2/t7a_ease_curve.py
"""
import os, sys, json, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ml1m_arena as A
import ml1m_bars as M
import p4a_battery as P
import p4a_bootstrap as B0

CK = '.cache/instrument2'
SEEDS = P.SEEDS               # {1,2,3,7,11}
KMAX = 8
OUTJSON = f'{CK}/t7a_ease_coldstart.json'


def select_ease_B(ar):
    """Train EASE on trU (seed-independent), select lambda on VAL by full NDCG (mirrors
    ml1m_bars.ease_bar exactly). Returns (lam, B)."""
    rat = {x: dict(v) for x, v in ar['rat_by_u'].items()}
    X = M.build_like_matrix(ar, ar['trU'])
    print(f'[ease] train X nnz={X.nnz}', flush=True)
    best = None
    for lam in [1.0, 10.0, 100.0, 500.0, 1000.0]:
        B = M.ease_B(X, lam)
        vf, vt, _, _ = M.A_eval(ar, ar['val_users'], M._ease_scorer(B, ar, rat))
        print(f'  [val lam={lam:>6.0f}] full {vf:.4f} tail {vt:.4f}', flush=True)
        if best is None or vf > best[1]:
            best = (lam, vf, B)
    print(f'[ease] val-selected lam={best[0]}', flush=True)
    return best[0], best[2]


def main():
    t0 = time.time()
    model, d = P.load_model()
    # EASE B: trained on trU (seed-independent), lambda selected on seed-123 arena val
    # (byte-identical to ml1m_bars.ease_bar which uses the default seed-123 arena).
    lam, B_ease = select_ease_B(A.load_arena(seed=123))

    # per-user seed-averaged NDCG for I2 and EASE at each k (for paired bootstrap)
    # key: (user,) -> {k: {'i2_full':[..seeds], 'i2_tail':[], 'ease_full':[], 'ease_tail':[]}}
    per_user = {}
    # per-seed cohort means at each k
    seed_means = {k: {'i2_full': [], 'i2_tail': [], 'ease_full': [], 'ease_tail': []}
                  for k in range(1, KMAX + 1)}

    for sd in SEEDS:
        ar = A.load_arena(seed=sd)
        rat = {x: dict(v) for x, v in ar['rat_by_u'].items()}
        test = ar['test_users']
        rng = np.random.default_rng(sd)       # same generator convention as P4a item8_fold

        acc = {k: {'i2_full': 0.0, 'i2_tail': 0.0, 'ease_full': 0.0, 'ease_tail': 0.0,
                   'nf': 0, 'nt': 0} for k in range(1, KMAX + 1)}
        for x in test:
            profset, tst = ar['SPL'][x]
            rd = rat[x]
            tlike = set(j for j in tst if rd[j] >= 4)
            if not tlike:
                continue
            likes = [j for j in profset if rd[j] >= 4]
            if not likes:
                continue
            perm = list(rng.permutation(len(likes)))
            lk = [likes[i] for i in perm]      # random order of this user's profile likes

            pu = per_user.setdefault((x,), {k: {'i2_full': [], 'i2_tail': [],
                                                'ease_full': [], 'ease_tail': []}
                                            for k in range(1, KMAX + 1)})
            for k in range(1, KMAX + 1):
                rev = lk[:k]                    # first k revealed (saturates when <k available)
                # ---- I2 item-fold ----
                z = P.enc_mu(model, P.bag_from_likes(rev)[None, :])[0]
                s_i2 = P.decode_np(model, z[None, :])[0]
                # ---- EASE cold-start (restricted history) ----
                s_ea = B_ease[rev].sum(0)
                for tail, itag, etag, cnt in [(False, 'i2_full', 'ease_full', 'nf'),
                                              (True, 'i2_tail', 'ease_tail', 'nt')]:
                    ni2 = A.ndcg_at10(s_i2, tlike, profset, ar['headmask'], tail)
                    nea = A.ndcg_at10(s_ea, tlike, profset, ar['headmask'], tail)
                    if ni2 is None or nea is None:
                        continue
                    acc[k][itag] += ni2; acc[k][etag] += nea; acc[k][cnt] += 1
                    pu[k][itag].append(ni2); pu[k][etag].append(nea)
        for k in range(1, KMAX + 1):
            nf = max(acc[k]['nf'], 1); nt = max(acc[k]['nt'], 1)
            seed_means[k]['i2_full'].append(acc[k]['i2_full'] / nf)
            seed_means[k]['ease_full'].append(acc[k]['ease_full'] / nf)
            seed_means[k]['i2_tail'].append(acc[k]['i2_tail'] / nt)
            seed_means[k]['ease_tail'].append(acc[k]['ease_tail'] / nt)
        print(f'[eval] seed{sd} done  k8 I2 full {seed_means[8]["i2_full"][-1]:.4f} '
              f'EASE full {seed_means[8]["ease_full"][-1]:.4f}  [{time.time()-t0:.0f}s]', flush=True)

    # aggregate curves (5-seed mean +/- sd)
    curves = {}
    for k in range(1, KMAX + 1):
        c = {}
        for tag in ('i2_full', 'i2_tail', 'ease_full', 'ease_tail'):
            v = np.array(seed_means[k][tag])
            c[tag] = float(v.mean()); c[tag + '_sd'] = float(v.std())
        curves[str(k)] = c

    # paired bootstrap of I2 - EASE at each k (per-user seed-averaged), reuse p4a_bootstrap.boot
    boot = {}
    for k in range(1, KMAX + 1):
        df, dt = [], []
        for key, kk in per_user.items():
            fi = kk[k]['i2_full']; fe = kk[k]['ease_full']
            ti = kk[k]['i2_tail']; te = kk[k]['ease_tail']
            if len(fi) and len(fe) and not (np.isnan(fi).any() or np.isnan(fe).any()):
                df.append(np.mean(fi) - np.mean(fe))
            if len(ti) and len(te) and not (np.isnan(ti).any() or np.isnan(te).any()):
                dt.append(np.mean(ti) - np.mean(te))
        boot[str(k)] = {'full': B0.boot(df), 'tail': B0.boot(dt)}
        print(f'[boot] k={k} full dMean {boot[str(k)]["full"]["mean_diff"]:+.4f} '
              f'CI {boot[str(k)]["full"]["ci95"]}  tail dMean '
              f'{boot[str(k)]["tail"]["mean_diff"]:+.4f}', flush=True)

    out = {'meta': {'seeds': SEEDS, 'kmax': KMAX, 'ease_lambda': lam,
                    'n_test_users': len(per_user),
                    'item_selection': 'random perm of profile likes, seeded per eval seed, first-k (nested); matches P4a item8_fold rng convention',
                    'i2_mechanism': 'item-fold through RecVAE-d512 encoder (P4a item8_fold arm)',
                    'ease_mechanism': 'cold-start EASE B[revealed_k].sum(0), trU-trained val-selected lambda'},
           'curves': curves, 'bootstrap': boot}
    json.dump(out, open(OUTJSON, 'w'), indent=2)
    print(f'[saved] {OUTJSON}', flush=True)

    # concise console verdict
    print('\n=== T7a COLD-START CURVE (5-seed TEST mean) ===', flush=True)
    print(' k |  I2 full  EASE full   d(I2-EASE)  |  I2 tail  EASE tail', flush=True)
    for k in range(1, KMAX + 1):
        c = curves[str(k)]; bk = boot[str(k)]
        print(f' {k} |  {c["i2_full"]:.4f}   {c["ease_full"]:.4f}    {bk["full"]["mean_diff"]:+.4f}'
              f'   |  {c["i2_tail"]:.4f}   {c["ease_tail"]:.4f}', flush=True)


if __name__ == '__main__':
    main()
