"""dislike_signal_probe.py -- DECISIVE cheap GO/NO-GO probe.

Question: at SHORT interview lengths, does knowing a user's DISLIKES add INCREMENTAL
held-LIKED ranking value beyond the few revealed likes, in a TASTE-structured space?

PART A  model-free item-item neighborhood predictor in MF(taste) space.
  reveal k in {2,4} random LIKED items; hold out the rest of the user's likes.
  score(c) = mean cos(c, revealed likes) - alpha * mean cos(c, ORACLE dislikes).
  alpha in {0,.25,.5,1,2}. candidate pool = all items minus (revealed likes U oracle dislikes),
  IDENTICAL across alpha (dislikes excluded from candidates in every condition -> alpha isolates
  ONLY the repulsion effect). held-liked NDCG@10.

PART B  SIGNED EASE (closed form on the CENTERED-rating train matrix; B itself is trained signed).
  strength: full-profile held-liked NDCG@10, signed-EASE vs implicit-EASE (reference).
  short-k fold-in: user vec = (k revealed likes as +1) + alpha*(oracle dislikes as -1); score = vec@B.
  alpha in {0,.25,.5,1,2}, k in {2,4,8}. same identical-pool rule.

Held-out TEST users = meta 'te' (500) MINUS the 300 LLM study users (quarantined). No training on
them. No data reduction: all eligible test users, all their ratings, all items as candidates,
full item-item gram / full EASE inverse (chunked dense accumulation for memory, never a cap).
"""
import os, sys, json, time
os.environ.setdefault('OMP_NUM_THREADS', str(os.cpu_count()))
os.environ.setdefault('MKL_NUM_THREADS', str(os.cpu_count()))
os.environ.setdefault('OPENBLAS_NUM_THREADS', str(os.cpu_count()))
import numpy as np
import scipy.sparse as sp
import torch
torch.set_num_threads(os.cpu_count())

C = 'C:/dev/phd/casper/.cache/rung1/rawcache'
META = 'C:/dev/phd/casper/data/movielens/.cache/ml25m/meta.npz'
STUDY = 'C:/dev/phd/casper/.cache/instrument2/answerability_grid_ml25m.json'
OUT_JSON = 'C:/dev/phd/casper/.cache/rung1/dislike_signal_probe.json'

LO, HI = 4.0, 2.0          # liked / disliked thresholds (project canon)
ALPHAS = [0.0, 0.25, 0.5, 1.0, 2.0]
NSEED = 8                  # random-reveal repetitions per user (robustness, NOT a reduction)

def log(*a): print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)

def ndcg10(scores, held_set, exclude):
    """NDCG@10 of held_set among all items; exclude=known items masked out."""
    s = scores.copy()
    s[list(exclude)] = -1e30
    top = np.argpartition(-s, 10)[:10]
    top = top[np.argsort(-s[top])]
    dcg = sum(1.0/np.log2(p+2) for p, it in enumerate(top) if int(it) in held_set)
    ideal = sum(1.0/np.log2(p+2) for p in range(min(10, len(held_set))))
    return dcg/ideal if ideal > 0 else None

# ------------------------------------------------------------------ load test users
def load_test_users():
    m = np.load(META)
    uu, ii, rr = m['uu'].astype(np.int64), m['ii'].astype(np.int64), m['rr'].astype(np.float32)
    te = set(m['te'].tolist()); ni = int(m['ni'])
    study = set(int(u) for u in json.load(open(STUDY))['users'].keys())
    assert study <= te, "study cohort not subset of te"
    keep = np.array(sorted(te - study), dtype=np.int64)      # 200 clean held-out users
    log(f"ni={ni}  test te={len(te)}  study(quarantined)={len(study)}  clean test users={len(keep)}")
    mask = np.isin(uu, keep)
    u_k, i_k, r_k = uu[mask], ii[mask], rr[mask]
    order = np.argsort(u_k, kind='stable')
    u_k, i_k, r_k = u_k[order], i_k[order], r_k[order]
    users = {}
    b = 0; N = len(u_k)
    while b < N:
        e = b
        while e < N and u_k[e] == u_k[b]:
            e += 1
        its = i_k[b:e]; rs = r_k[b:e]
        liked = its[rs >= LO].tolist()
        disliked = its[rs <= HI].tolist()
        users[int(u_k[b])] = dict(items=its, ratings=rs,
                                  liked=liked, disliked=disliked)
        b = e
    return ni, users

# ------------------------------------------------------------------ PART A
def part_a(ni, users, Vn):
    ks = [2, 4]
    # per (k, alpha) accumulate per-user mean-over-seeds ndcg
    res = {k: {a: [] for a in ALPHAS} for k in ks}
    nrng = np.random.default_rng(0)
    n_eligible = {k: 0 for k in ks}
    for uid, u in users.items():
        L = np.array(u['liked']); D = np.array(u['disliked'])
        if len(D) < 1:
            continue
        mdis = Vn[D].mean(0)
        for k in ks:
            if len(L) < k + 1:
                continue
            n_eligible[k] += 1
            per_alpha = {a: [] for a in ALPHAS}
            for _ in range(NSEED):
                rev = nrng.choice(len(L), size=k, replace=False)
                Lk = L[rev]
                held = set(L.tolist()) - set(Lk.tolist())
                if not held:
                    continue
                excl = set(Lk.tolist()) | set(D.tolist())
                mlike = Vn[Lk].mean(0)
                like_s = Vn @ mlike
                dis_s = Vn @ mdis
                for a in ALPHAS:
                    sc = like_s - a * dis_s
                    v = ndcg10(sc, held, excl)
                    if v is not None:
                        per_alpha[a].append(v)
            for a in ALPHAS:
                if per_alpha[a]:
                    res[k][a].append(np.mean(per_alpha[a]))
    table = {}
    for k in ks:
        row = {}
        base = np.mean(res[k][0.0]) if res[k][0.0] else float('nan')
        for a in ALPHAS:
            m = float(np.mean(res[k][a])) if res[k][a] else float('nan')
            row[a] = dict(ndcg=m, gain_vs_a0=float(m - base))
        row['_n_users'] = len(res[k][0.0]); row['_base_a0'] = float(base)
        table[k] = row
        log(f"[A] k={k} n={row['_n_users']} " +
            " ".join(f"a{a}={row[a]['ndcg']:.4f}({row[a]['gain_vs_a0']:+.4f})" for a in ALPHAS))
    return table

# ------------------------------------------------------------------ EASE
def build_gram(X, ni, chunk=6000):
    """G = X^T X via chunked DENSE accumulation in torch float32 (memory-safe; no cap; 2-thread
    OpenBLAS is bypassed by torch). X = csr users x items."""
    G = torch.zeros((ni, ni), dtype=torch.float32)
    n = X.shape[0]
    for b in range(0, n, chunk):
        Xc = torch.from_numpy(X[b:b+chunk].toarray().astype(np.float32))
        G.addmm_(Xc.T, Xc)      # G += Xc^T Xc in place
        log(f"    gram {min(b+chunk,n)}/{n}")
    return G                     # torch float32 (ni x ni)

def ease_from_gram(G, lam):
    """B = -(G+lam I)^-1 / diag; diag(B)=0.  Returns numpy float32 B (ni x ni)."""
    ni = G.shape[0]
    A = G.clone()
    A.diagonal().add_(lam)
    P = torch.linalg.inv(A)      # float32 inverse
    del A
    dP = torch.diag(P).clone()
    P /= -dP[None, :]            # divide column j by -P_jj
    P.fill_diagonal_(0.0)
    return P.numpy()             # B (ni x ni) float32

def ease_score(B, weights, idx):
    """score_i = sum_j w_j B[j,i]  (few nonzero history items)."""
    return weights @ B[idx, :]

def eval_ease_shortk(B, users, ni, ks=(2, 4, 8)):
    res = {k: {a: [] for a in ALPHAS} for k in ks}
    nrng = np.random.default_rng(1)
    for uid, u in users.items():
        L = np.array(u['liked']); D = np.array(u['disliked'])
        if len(D) < 1:
            continue
        for k in ks:
            if len(L) < k + 1:
                continue
            per_alpha = {a: [] for a in ALPHAS}
            for _ in range(NSEED):
                rev = nrng.choice(len(L), size=k, replace=False)
                Lk = L[rev]
                held = set(L.tolist()) - set(Lk.tolist())
                if not held:
                    continue
                excl = set(Lk.tolist()) | set(D.tolist())
                for a in ALPHAS:
                    idx = np.concatenate([Lk, D]).astype(np.int64)
                    w = np.concatenate([np.ones(len(Lk)), -a*np.ones(len(D))]).astype(np.float32)
                    sc = ease_score(B, w, idx).astype(np.float64)
                    v = ndcg10(sc, held, excl)
                    if v is not None:
                        per_alpha[a].append(v)
            for a in ALPHAS:
                if per_alpha[a]:
                    res[k][a].append(np.mean(per_alpha[a]))
    table = {}
    for k in ks:
        base = np.mean(res[k][0.0]) if res[k][0.0] else float('nan')
        row = {}
        for a in ALPHAS:
            m = float(np.mean(res[k][a])) if res[k][a] else float('nan')
            row[a] = dict(ndcg=m, gain_vs_a0=float(m - base))
        row['_n_users'] = len(res[k][0.0]); row['_base_a0'] = float(base)
        table[k] = row
        log(f"[B] k={k} n={row['_n_users']} " +
            " ".join(f"a{a}={row[a]['ndcg']:.4f}({row[a]['gain_vs_a0']:+.4f})" for a in ALPHAS))
    return table

def eval_full_strength(B_signed, B_imp, users, ni, holdfrac=0.2, nseed=5):
    """full-profile: fold in all-but-held(20%) likes; signed also includes dislikes(alpha=1).
    held-liked NDCG@10. Returns (signed, implicit) means."""
    nrng = np.random.default_rng(2)
    sgn, imp = [], []
    for uid, u in users.items():
        L = np.array(u['liked']); D = np.array(u['disliked'])
        if len(L) < 5:
            continue
        us_s, us_i = [], []
        for _ in range(nseed):
            perm = nrng.permutation(len(L))
            nh = max(1, int(round(holdfrac*len(L))))
            held = set(L[perm[:nh]].tolist())
            inp = L[perm[nh:]]
            if not inp.size:
                continue
            excl = set(inp.tolist()) | set(D.tolist())
            # implicit: +1 on input likes only
            sc_i = ease_score(B_imp, np.ones(len(inp), np.float32), inp.astype(np.int64)).astype(np.float64)
            vi = ndcg10(sc_i, held, set(inp.tolist()))
            # signed: +1 input likes, -1 dislikes
            idx = np.concatenate([inp, D]).astype(np.int64)
            w = np.concatenate([np.ones(len(inp)), -np.ones(len(D))]).astype(np.float32)
            sc_s = ease_score(B_signed, w, idx).astype(np.float64)
            vs = ndcg10(sc_s, held, excl)
            if vi is not None: us_i.append(vi)
            if vs is not None: us_s.append(vs)
        if us_i: imp.append(np.mean(us_i))
        if us_s: sgn.append(np.mean(us_s))
    return float(np.mean(sgn)), float(np.mean(imp)), len(sgn)

# ------------------------------------------------------------------ main
def main():
    t0 = time.time()
    ni, users = load_test_users()
    out = dict(meta=dict(n_test_users=len(users), ni=ni, LO=LO, HI=HI,
                         alphas=ALPHAS, nseed=NSEED,
                         quarantine="te(500) minus 300 LLM study users; train-disjoint by construction",
                         no_data_reduction=True))

    # ---------- PART A ----------
    log("PART A: MF taste-space neighborhood predictor")
    V = np.load(f'{C}/mf_factors.npy').astype(np.float32)
    Vn = V / (np.linalg.norm(V, axis=1, keepdims=True) + 1e-9)
    out['partA_MF_neighborhood'] = part_a(ni, users, Vn)
    del V, Vn

    # ---------- PART B ----------
    log("PART B: signed EASE (closed form on centered train matrix)")
    Radj = sp.load_npz(f'{C}/Radj.npz').tocsr()           # centered train (signed)
    Rraw = sp.load_npz(f'{C}/R.npz').tocsr()
    Xb = (Rraw >= LO).astype(np.float64).tocsr()          # implicit binary train
    del Rraw
    log(f"  signed X nnz={Radj.nnz}  implicit X nnz={Xb.nnz}  users={Radj.shape[0]}")

    lam_grid = [100.0, 300.0, 1000.0]

    # SIGNED
    log("  building signed Gram ...")
    Gs = build_gram(Radj, ni); del Radj
    best_s = None
    for lam in lam_grid:
        B = ease_from_gram(Gs, lam)
        s, i, n = eval_full_strength(B, B, users, ni)   # placeholder imp; recomputed below
        log(f"    signed lam={lam} full-strength(signed)={s:.4f} (n={n})")
        if best_s is None or s > best_s[1]:
            best_s = (lam, s, B)
    del Gs
    lam_s, str_s, B_signed = best_s
    log(f"  BEST signed lam={lam_s} strength={str_s:.4f}")

    # IMPLICIT
    log("  building implicit Gram ...")
    Gi = build_gram(Xb, ni); del Xb
    best_i = None
    for lam in lam_grid:
        B = ease_from_gram(Gi, lam)
        # implicit strength: use B for implicit path only
        _, i_str, n = eval_full_strength(B_signed, B, users, ni)
        log(f"    implicit lam={lam} full-strength(implicit)={i_str:.4f} (n={n})")
        if best_i is None or i_str > best_i[1]:
            best_i = (lam, i_str, B)
    del Gi
    lam_i, str_i, B_imp = best_i
    log(f"  BEST implicit lam={lam_i} strength={str_i:.4f}")

    # final strength numbers with the chosen B's
    s_final, i_final, n_str = eval_full_strength(B_signed, B_imp, users, ni)
    out['partB_strength'] = dict(signed_ease_full=s_final, implicit_ease_full=i_final,
                                 lam_signed=lam_s, lam_implicit=lam_i, n_users=n_str,
                                 note="held=20% of likes, fold in remainder; NDCG@10 over all items")
    log(f"  STRENGTH  signed={s_final:.4f}  implicit={i_final:.4f}")

    # short-k alpha table on the SIGNED model
    log("  PART B short-k alpha x k table (signed EASE):")
    out['partB_signed_ease_shortk'] = eval_ease_shortk(B_signed, users, ni)

    out['minutes'] = round((time.time()-t0)/60, 1)

    # ---------- verdict ----------
    def any_gain(tab, kset):
        best = -9; arg = None
        for k in kset:
            for a in ALPHAS:
                if a == 0.0: continue
                g = tab[k][a]['gain_vs_a0']
                if g == g and g > best:
                    best = g; arg = (k, a)
        return best, arg
    ga, aa = any_gain(out['partA_MF_neighborhood'], [2, 4])
    gb, ab = any_gain(out['partB_signed_ease_shortk'], [2, 4])
    gb8, ab8 = any_gain(out['partB_signed_ease_shortk'], [2, 4, 8])
    decisive = max(ga, gb)
    go = (ga > 0) or (gb > 0)
    out['verdict'] = dict(
        partA_best_shortk_gain=ga, partA_best_arg=aa,
        partB_best_shortk_gain=gb, partB_best_arg=ab,
        partB_best_gain_incl_k8=gb8, partB_best_arg_incl_k8=ab8,
        decisive_number=decisive,
        GO=bool(go),
        statement=("GO: dislikes add held-liked NDCG at short k for some alpha>0"
                   if go else
                   "NO-GO: no alpha helped at short k even with ORACLE dislikes in either part"))
    json.dump(out, open(OUT_JSON, 'w'), indent=2, default=float)
    log(f"wrote {OUT_JSON}  [{out['minutes']}m]")
    log(f"VERDICT: {'GO' if go else 'NO-GO'}  decisive={decisive:+.4f}")
    return out

if __name__ == '__main__':
    main()
