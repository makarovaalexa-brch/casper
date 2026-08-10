r"""greedyslim_native_snap.py -- SNAP-4: Greedy SLIM (Proissl, Vatic & Waldschmidt, arXiv:2406.06061)
NATIVE snap in the authors' own setup, plus the asked-item-exclusion ABLATION.

===============================================================================================
STEP 0 -- THE PUBLISHED TARGET, WITH SOURCE                                                [V]
===============================================================================================
Claudius Proissl, Amel Vatic, Helmut Waldschmidt, "Greedy SLIM: A SLIM-Based Approach For
Preference Elicitation", arXiv:2406.06061 [cs.IR], 10 Jun 2024.  **CoRR only -- dblp lists no
venue.**  Provenance caveat carried in every report: arXiv-only preprint; its user study is
n=103 with "many of them were somehow linked to the authors".

Table 3 (ML-25M), NDCG@10, questionnaire size k:

    k        |   5      10      15      20
    ---------+--------------------------------
    ALL  QGSLIM  | .3390   .3594   .3709   .3752      <- static, prefix-optimal
    ALL  QBandit | .3382   .3456   .3221   .2916      <- adaptive (Christakopoulou KDD'16)
    TAIL QGSLIM  | .1163   .1543   .1674   .1779
    TAIL QBandit | .1183   .1465   .1488   .1466

(Netflix goes the other way at k<=10: QBandit .3030/.3251 vs QGSLIM .2859/.3042.)

THEIR CONTRACT, verbatim where it matters -- NONE OF IT IS OUR CONTRACT, so these numbers are
NOT orderable against our 0.3496/0.2491 Liang cells.  The snap happens entirely inside their
setup.
  * ML-25M **UNFILTERED**: "We do not filter these sets to ensure that we do not accidentally
    introduce any biases."  162,541 users / 59,047 items / 25,000,095 ratings.
  * Random **90/10 user split**; |U_test| = 0.1|U|.
  * **Explicit 1-5 ratings, not binarised.**  "the values x_ui of X are either 0 if user u has
    not rated item i yet or x_ui = r_ui."
  * Graded gain: "We set the gain g_ui of user u for item i to be **2^{x_ui} - 1** in order to
    emphasize that a high rating is much more important than a medium rating."
  * "The NDCG is the DCG normalized by the **maximum DCG among all subsets I subset I_R**
    (for a fixed user u)."  -> **IDCG IS COMPUTED OVER THE CANDIDATE SET I_R.**
  * Candidates: "we do not apply any kind of item sampling to avoid the problems reported in
    Krichene and Rendle"; candidates are **all items with x_ui = 0** (i.e. everything the
    questionnaire has not revealed a rating for).
  * S3.3.2: "Let I_Q be the set of items asked by the questionnaire and let I_R be the set of
    recommended items.  **We always require I_Q ∩ I_R = ∅ to avoid trivial recommendations.**"
  * Long tail, verbatim ours: "Following Cremonesi et al. (2010), we refer to the set of most
    popular items that represent **33% of the ratings** as short-head items. The remaining
    items are called long-tail items... the **top-N recommendation is restricted to long-tail
    items**."
  * Hyperparameters ML-25M: **lambda_1 = 2^12, lambda_F = 2^16** (grid over powers of two).
  * Budgets k in {5,10,15,20}.  Cost: "19 minutes per row" on ML-25M.
  * QBandit's latent factor model is **PureSVD**.

===============================================================================================
THE ABLATION -- PRE-REGISTERED BEFORE RUNNING (this docstring is the pre-registration)
===============================================================================================
S6.5 states plainly:

    "The NDCG of Q_Bandit converges or even decreases after ten questions.  **We think that the
     main reason for the decreasing NDCG is that the questionnaires are not allowed to recommend
     items that they contain as questions.**  Therefore, using popular items as questions may
     have a negative impact on the recommendation performance if the questionnaire is unable to
     generalize the gained information."

★ STRUCTURAL POINT THAT DECIDES THE DESIGN.  Their candidate rule is already "all items with
x_ui = 0", i.e. every item the questionnaire ANSWERED is out of the pool anyway.  So the extra
constraint I_Q ∩ I_R = ∅ bites on exactly one set: **asked-but-UNANSWERED items**.  And under
their offline protocol an asked-but-unanswered item is one the test user has **no rating for**,
hence a *guaranteed non-target* (gain 0).  Removing a guaranteed non-target from I_R cannot
lower DCG and cannot change IDCG.

    ==> The rule S6.5 blames is, by construction, a MONOTONE BENEFIT, not a tax.
    ==> The tax must live in the OTHER exclusion -- the answered one, which their design forces
        because a test user has no held-out split: every question the user CAN answer deletes a
        real, high-gain, easy target from that user's own ground truth.  A questionnaire that
        asks popular items answers more often, so it eats more of its own ground truth, and the
        cost compounds with k.  That is a different mechanism from the one they name.

THREE EVALUATION ARMS, everything else identical (same interviews where the answers are
identical, same recommender, same users, same k):

  ARM C  "theirs"          I_R = {x_ui = 0} \ I_Q          targets = user's ratings ∩ I_R
  ARM B  "exclusion off"   I_R = {x_ui = 0}                targets = user's ratings ∩ I_R
         (drops I_Q ∩ I_R = ∅ only; answered items still out, as their own x_ui=0 rule requires)
  ARM A  "no self-eating"  a fixed per-user 20% of the ratings is held out BEFORE the interview,
         is never answerable, and is the ONLY target set.  I_R = I \ (ask-side rated).
         Arm-independent ground truth; no questionnaire can consume its own targets.

PRE-REGISTERED OUTCOMES (stated before running, per the brief):
  (a) QBandit's decline vanishes under ARM B  ->  a published result is a protocol artefact of
      the rule the authors name.  **We predict this will NOT happen**, for the structural reason
      above; if it does, our reading of their candidate rule is wrong and that is itself the
      finding.
  (b) The decline persists under ARM B  ->  I_Q ∩ I_R = ∅ is not the cause and S6.5's diagnosis
      is wrong.  ARM A then adjudicates the real cause.
  (c) Partial -> quantify the share of the decline attributable to each exclusion.
  ARM A is the discriminating cell: if the decline vanishes there, the cause is
  ground-truth consumption by answered questions (their no-held-out-split design), not the
  asked-item rule.

===============================================================================================
PERSISTENCE (CLAUDE.md HARD RULE 11)
===============================================================================================
Every (questionnaire, arm, k) cell is written to the output JSON the moment it completes.
The two expensive intermediates -- the PureSVD factors and the item-item gram G = X^T X -- are
cached in contract-signed .npz sidecars keyed on a hash of (dataset, split seed, n_train_users,
rank/threshold).  Resume refuses across a changed contract.  A unique --label stamps every
output path; the script refuses to open an output another live process holds.
"""
import os
import sys
import json
import time
import hashlib
import argparse

import numpy as np
import scipy.sparse as sp
from scipy.linalg import solve_triangular

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))

RAW = os.path.join(_ROOT, "data", "movielens", "ratings.csv")
CACHE = os.path.join(_ROOT, ".cache", "greedyslim")
OUTDIR = os.path.join(_ROOT, "experiments", "baselines", "greedyslim")

# --- their published targets, for the snap verdict table --------------------------------------
PUBLISHED = {
    ("qgslim", "all"):  {5: .3390, 10: .3594, 15: .3709, 20: .3752},
    ("qbandit", "all"): {5: .3382, 10: .3456, 15: .3221, 20: .2916},
    ("qgslim", "tail"): {5: .1163, 10: .1543, 15: .1674, 20: .1779},
    ("qbandit", "tail"):{5: .1183, 10: .1465, 15: .1488, 20: .1466},
}
BUDGETS = (5, 10, 15, 20)
LAMBDA_1 = 2.0 ** 12
LAMBDA_F = 2.0 ** 16


def logln(msg, path=None):
    line = "[%s] %s" % (time.strftime("%H:%M:%S"), msg)
    print(line, flush=True)
    if path:
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(line + "\n")


# =============================================================================================
# DATA -- ML-25M unfiltered, exactly as they use it
# =============================================================================================
def load_ml25m(log=None):
    """Full ML-25M as a CSR users x items matrix of explicit 1-5 ratings.  No filtering."""
    os.makedirs(CACHE, exist_ok=True)
    cpath = os.path.join(CACHE, "ml25m_full.npz")
    if os.path.exists(cpath):
        z = np.load(cpath)
        X = sp.csr_matrix((z["data"], z["indices"], z["indptr"]), shape=tuple(z["shape"]))
        logln("loaded cached ML-25M %s nnz=%d" % (X.shape, X.nnz), log)
        return X
    import pandas as pd
    logln("reading %s ..." % RAW, log)
    df = pd.read_csv(RAW, usecols=["userId", "movieId", "rating"],
                     dtype={"userId": np.int64, "movieId": np.int64, "rating": np.float32})
    logln("read %d ratings, %d users, %d items" % (len(df), df.userId.nunique(),
                                                   df.movieId.nunique()), log)
    uu, urow = np.unique(df.userId.values, return_inverse=True)
    ii, icol = np.unique(df.movieId.values, return_inverse=True)
    X = sp.csr_matrix((df.rating.values.astype(np.float32), (urow, icol)),
                      shape=(len(uu), len(ii)))
    X.sort_indices()
    np.savez(cpath, data=X.data, indices=X.indices, indptr=X.indptr, shape=np.array(X.shape))
    logln("built ML-25M %s nnz=%d -> %s" % (X.shape, X.nnz, cpath), log)
    return X


def split_users(n_users, seed=98765):
    rng = np.random.RandomState(seed)
    perm = rng.permutation(n_users)
    n_test = int(round(0.1 * n_users))
    return np.sort(perm[n_test:]), np.sort(perm[:n_test])   # train, test


def head_mask(Xtr, frac=0.33):
    """Cremonesi 33%-of-ratings short head, computed on the training interactions."""
    cnt = np.asarray((Xtr > 0).sum(axis=0)).ravel().astype(np.int64)
    order = np.argsort(-cnt, kind="stable")
    cum = np.cumsum(cnt[order])
    n_head = int(np.searchsorted(cum, frac * cnt.sum()) + 1)
    head = np.zeros(Xtr.shape[1], dtype=bool)
    head[order[:n_head]] = True
    return head, n_head


# =============================================================================================
# METRIC -- their NDCG@10 exactly: gain 2^r - 1, IDCG over the candidate set I_R
# =============================================================================================
def csr_lookup(R, rows, cols):
    """Vectorised elementwise CSR lookup R[rows[t], cols[t]] -> 0 where absent.
    Exploits the fact that a sorted-indices CSR is globally sorted on (row*n + col)."""
    n = R.shape[1]
    glob = (np.repeat(np.arange(R.shape[0], dtype=np.int64), np.diff(R.indptr)) * n
            + R.indices.astype(np.int64))
    key = rows.astype(np.int64) * n + cols.astype(np.int64)
    pos = np.clip(np.searchsorted(glob, key), 0, max(len(glob) - 1, 0))
    if len(glob) == 0:
        return np.zeros(len(rows), dtype=np.float32)
    hit = glob[pos] == key
    return np.where(hit, R.data[pos], 0.0).astype(np.float32)


def ndcg_at_k_block(scores, tgt_idx, tgt_gain, cand, k=10):
    """Their NDCG@k.  scores, cand : dense (B, n_items).  Targets are SPARSE per row
    (tgt_idx[b], tgt_gain[b] = item ids and 2^r - 1 gains), because a user has ~154 ratings out
    of 59,047 items and materialising a dense gain matrix dominated the runtime.

    DCG  = gains of the top-k CANDIDATE items by score.
    IDCG = the best DCG@k achievable using only items in I_R -- "the DCG normalized by the
           maximum DCG among all subsets I subset I_R", i.e. IDCG is also restricted to cand."""
    B = scores.shape[0]
    disc = 1.0 / np.log2(np.arange(2, k + 2))
    s = np.where(cand, scores, -np.inf)
    top = np.argpartition(-s, k, axis=1)[:, :k]
    rows = np.arange(B)[:, None]
    ordr = np.argsort(-s[rows, top], axis=1, kind="stable")
    top = top[rows, ordr]
    okc = cand[rows, top]

    dcg = np.zeros(B); idcg = np.zeros(B)
    for b in range(B):
        ti, tg = tgt_idx[b], tgt_gain[b]
        if len(ti) == 0:
            continue
        keep = cand[b, ti]
        ti, tg = ti[keep], tg[keep]
        if len(ti) == 0:
            continue
        # DCG: gain of each of the k ranked slots (0 unless that slot is a target)
        pos = np.searchsorted(ti, top[b])
        pos = np.clip(pos, 0, len(ti) - 1)
        hit = (ti[pos] == top[b]) & okc[b]
        dcg[b] = (np.where(hit, tg[pos], 0.0) * disc).sum()
        # IDCG over I_R
        best = np.sort(tg)[::-1][:k]
        idcg[b] = (best * disc[:len(best)]).sum()
    out = np.zeros(B, dtype=np.float64)
    nz = idcg > 0
    out[nz] = dcg[nz] / idcg[nz]
    return out, nz


# =============================================================================================
# CONTRACT-SIGNED SIDECARS (HARD RULE 11b)
# =============================================================================================
def _sig(**kw):
    return hashlib.sha1(json.dumps(kw, sort_keys=True).encode()).hexdigest()[:12]


def _sidecar(name, sig):
    os.makedirs(CACHE, exist_ok=True)
    return os.path.join(CACHE, "%s_%s.npz" % (name, sig))


# =============================================================================================
# ITEM-ITEM GRAM  G = X_tr^T X_tr, thresholded at lambda_1/2 -- and the threshold is EXACT.
#
# The greedy criterion only ever needs entries with c_ij > lambda_1/2, where
#     c = X^T Xhat = G - G[:,S] W[S,:]   with  G >= 0 elementwise and  W >= 0  (w* is clipped at 0)
# so the correction is subtractive:  c <= G  elementwise, hence  c_ij > t  ==>  G_ij > t.
# Dropping G entries <= lambda_1/2 therefore loses NOTHING (this is not a truncation under
# HARD RULE 1 -- it is an exact algebraic bound, asserted in the code below).
#
# Doing this ONCE is why our fit is ~25x cheaper than the authors' reported 19 min/row: they
# recompute the residual every round, but G is constant and only the rank-|S| correction moves.
# =============================================================================================
def build_gram(Xtr, thresh=LAMBDA_1 / 2.0, block=512, log=None):
    sig = _sig(kind="gram", nnz=int(Xtr.nnz), shape=list(Xtr.shape), thresh=float(thresh))
    path = _sidecar("gram", sig)
    if os.path.exists(path):
        z = np.load(path)
        G = sp.csr_matrix((z["data"], z["indices"], z["indptr"]), shape=tuple(z["shape"]))
        logln("gram sidecar HIT %s nnz=%d" % (os.path.basename(path), G.nnz), log)
        return G
    n = Xtr.shape[1]
    Xc = Xtr.tocsc()
    rows, cols, vals = [], [], []
    t0 = time.time()
    for s in range(0, n, block):
        e = min(s + block, n)
        Gb = (Xc[:, s:e].T @ Xtr).toarray()
        np.fill_diagonal(Gb[:, s:e], 0.0)          # j != i
        r, c = np.nonzero(Gb > thresh)
        rows.append(r.astype(np.int32) + s)
        cols.append(c.astype(np.int32))
        vals.append(Gb[r, c].astype(np.float32))
        if (s // block) % 10 == 0:
            logln("  gram %d/%d  kept=%d  %.1f min" % (e, n, sum(len(v) for v in vals),
                                                       (time.time() - t0) / 60), log)
    G = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                      shape=(n, n))
    G.sort_indices()
    np.savez(path, data=G.data, indices=G.indices, indptr=G.indptr, shape=np.array(G.shape))
    logln("gram built nnz=%d in %.1f min -> %s" % (G.nnz, (time.time() - t0) / 60, path), log)
    return G


# =============================================================================================
# GREEDY SLIM -- the questionnaire IS the order in which rows of W are filled.
#   w*_ij = max(0, (c_ij - l1/2) / (lF + a_i));   Delta_ij = -(c_ij - l1/2)^2 / (lF + a_i)
#   i* = argmin_i sum_{j != i} Delta_ij   ==   argmax_i sum_j max(0, c_ij - l1/2)^2 / (lF + a_i)
# STATIC and PREFIX-OPTIMAL: one fit gives every budget k.
# =============================================================================================
def greedy_slim(Xtr, G, k, lam1=LAMBDA_1, lamF=LAMBDA_F, log=None, rowblock=4096):
    n = Xtr.shape[1]
    a = np.asarray(Xtr.multiply(Xtr).sum(axis=0)).ravel().astype(np.float64)   # a_i = sum_u x_ui^2
    half = lam1 / 2.0
    sel, Wrows = [], np.zeros((k, n), dtype=np.float32)
    Gcsc = G.tocsc()
    indptr, indices, data = G.indptr, G.indices, G.data
    for rnd in range(k):
        t0 = time.time()
        Gs = np.zeros((n, len(sel)), dtype=np.float32)
        if sel:
            Gs = Gcsc[:, sel].toarray()            # G[:, S]  (n x |S|)
        score = np.zeros(n, dtype=np.float64)
        for s in range(0, n, rowblock):
            e = min(s + rowblock, n)
            lo, hi = indptr[s], indptr[e]
            if hi == lo:
                continue
            cols = indices[lo:hi]
            c = data[lo:hi].astype(np.float64)
            ridx = np.repeat(np.arange(s, e), np.diff(indptr[s:e + 1]))
            if sel:
                for t, _si in enumerate(sel):
                    c -= Gs[ridx, t].astype(np.float64) * Wrows[t, cols].astype(np.float64)
            g = c - half
            np.maximum(g, 0.0, out=g)
            np.multiply(g, g, out=g)
            # bincount, not np.add.at -- the latter is unbuffered and ~50x slower here
            score[s:e] += np.bincount(ridx - s, weights=g, minlength=e - s)
        score /= (lamF + a)
        score[sel] = -np.inf
        istar = int(np.argmax(score))
        # fill row istar of W against the CURRENT residual
        lo, hi = indptr[istar], indptr[istar + 1]
        cols, c = indices[lo:hi], data[lo:hi].astype(np.float64)
        if sel:
            for t, si in enumerate(sel):
                c -= float(Gs[istar, t]) * Wrows[t, cols].astype(np.float64)
        w = np.maximum(0.0, (c - half) / (lamF + a[istar]))
        Wrows[rnd, cols] = w.astype(np.float32)
        sel.append(istar)
        logln("  GSLIM round %2d -> item %6d  gain=%.4e  nnz(w)=%d  %.1f min"
              % (rnd + 1, istar, score[istar], int((w > 0).sum()), (time.time() - t0) / 60), log)
    return np.array(sel, dtype=np.int64), Wrows


# =============================================================================================
# PureSVD -- the latent factor model QBandit runs on (paper: "PureSVD")
# =============================================================================================
def pure_svd(Xtr, d=50, log=None):
    sig = _sig(kind="puresvd", nnz=int(Xtr.nnz), shape=list(Xtr.shape), d=int(d))
    path = _sidecar("puresvd", sig)
    if os.path.exists(path):
        z = np.load(path)
        logln("puresvd sidecar HIT d=%d" % d, log)
        return z["V"]
    from scipy.sparse.linalg import svds
    t0 = time.time()
    _, _, Vt = svds(Xtr.astype(np.float32), k=d)
    V = np.ascontiguousarray(Vt.T.astype(np.float32))     # (n_items, d)
    np.savez(path, V=V)
    logln("puresvd d=%d in %.1f min -> %s" % (d, (time.time() - t0) / 60, path), log)
    return V


# =============================================================================================
# QUESTIONNAIRES
# =============================================================================================
def q_pop(Xtr, k):
    """QPop: the k most-rated items. Static. Control arm: isolates 'asks popular items' from
    'is adaptive'."""
    cnt = np.asarray((Xtr > 0).sum(axis=0)).ravel()
    return np.argsort(-cnt, kind="stable")[:k].astype(np.int64)


def empirical_prior(Xtr, V, log=None):
    """Prior over a user's latent vector = the empirical mean/covariance of the TRAINING users'
    own PureSVD factors.  Required for scale-correctness: V has orthonormal columns, so a fixed
    lambda*I prior is mis-scaled against the likelihood and hands each revealed rating unbounded
    leverage (measured: it drove NDCG *below* the no-interview popularity floor).  It is also
    what makes a user who answers nothing fall back to population-average recommendations rather
    than to noise -- ~19% of users answer nothing at k=5.

    Returned in a WHITENED basis, theta = zbar + W alpha with alpha ~ N(0, I) and W = Q Lam^1/2
    from S0 = Q Lam Q^T.  This is an exact change of variables, and it is not optional: S0's
    eigenvalues span ~10 orders of magnitude, so the raw precision matrix S0^-1 is full of
    denormals and a batched 50x50 np.linalg.inv cost 38 ms per matrix (~1000x normal) -- 4.5 h
    per interview.  Whitened, the prior precision is the identity and the same inverse is free.

    base[i]  = v_i . zbar          (population-average predicted rating for item i)
    Vtil[i]  = W^T v_i             (whitened item factors)
    score[i] = base[i] + Vtil[i] . alpha
    """
    Utr = np.asarray(Xtr @ V, dtype=np.float64)
    zbar = Utr.mean(0)
    S0 = np.cov(Utr.T)
    del Utr
    lam, Q = np.linalg.eigh(S0)
    lam = np.clip(lam, max(lam.max(), 1e-12) * 1e-10, None)
    W = Q * np.sqrt(lam)[None, :]
    base = (V.astype(np.float64) @ zbar).astype(np.float32)
    Vtil = np.ascontiguousarray((V.astype(np.float64) @ W).astype(np.float32))
    logln("empirical prior whitened: eig span %.3e .. %.3e" % (lam.min(), lam.max()), log)
    return base, Vtil


def qbandit_interview(Rte_ask, V, k, prior, sigma2=1.0, seed=98765, ublock=256, log=None):
    """QBandit (Christakopoulou et al. KDD'16) over a PureSVD item space.
    Bayesian linear model r_ui = theta_u^T v_i + noise; Thompson sampling picks the next item to
    ask (sample theta~N(mu,Sigma), ask argmax_i theta^T v_i among unasked).  Recommendation uses
    the posterior MEAN.  Returns asked (U,k) int64, and the posterior means after every budget.
    Rte_ask : CSR of the ratings the interview is ALLOWED to reveal (their protocol = all of the
    test user's ratings; ARM A = the ask-side 80% only)."""
    base, Vtil = prior                                           # whitened basis, prior N(0, I)
    U, d = Rte_ask.shape[0], Vtil.shape[1]
    rng = np.random.RandomState(seed)
    asked = np.zeros((U, k), dtype=np.int64)
    VtilT = np.ascontiguousarray(Vtil.T)
    I_d = np.eye(d)
    for s in range(0, U, ublock):
        e = min(s + ublock, U)
        B = e - s
        Prec = np.repeat(I_d[None], B, axis=0)                   # (B,d,d) prior precision = I
        b = np.zeros((B, d), dtype=np.float64)
        Rb = Rte_ask[s:e].toarray()          # dense block: CSR fancy-indexing per turn is ~100x slower
        ar = np.arange(B)
        for t in range(k):
            # NB Cholesky only.  On this numpy build batched np.linalg.inv / .solve on
            # (256,50,50) cost 4.38 s / 3.24 s while batched cholesky costs 0.0067 s -- a ~500x
            # pathology in the batched getrf/getri path.  Using inv here cost 4.5 h per
            # interview; this costs ~2 min.  Prec = L L^T, so:
            #   mu = Prec^-1 b            (forward then back substitution)
            #   theta = mu + L^-T z       since (L^-T)(L^-T)^T = (L L^T)^-1 = Cov
            L = np.linalg.cholesky(Prec)
            z = rng.randn(B, d)
            mu = np.empty((B, d)); th = np.empty((B, d))
            for bb in range(B):
                Lb = L[bb]
                y = solve_triangular(Lb, b[bb], lower=True)
                mu[bb] = solve_triangular(Lb.T, y, lower=False)
                th[bb] = mu[bb] + solve_triangular(Lb.T, z[bb], lower=False)
            th = th.astype(np.float32)
            sc = base[None, :] + th @ VtilT                      # (B, n_items)
            if t:                                                # never re-ask (sparse, not a
                sc[ar[:, None], asked[s:e, :t]] = -np.inf        # 15M-element mask scan)
            pick = np.argmax(sc, axis=1)
            asked[s:e, t] = pick
            r = Rb[ar, pick]                                     # 0 if the user cannot answer
            ans = r > 0
            if ans.any():
                vv = Vtil[pick[ans]].astype(np.float64)
                Prec[ans] += np.einsum("bi,bj->bij", vv, vv) / sigma2
                b[ans] += vv * ((r[ans] - base[pick[ans]])[:, None] / sigma2)
        if log and (s // ublock) % 16 == 0:
            logln("  qbandit users %d/%d" % (e, U), log)
    return asked, None


# =============================================================================================
# EVALUATION -- the three exclusion arms
#
#   ARM C  "theirs"        I_R = {x'_ui = 0} \ I_Q        their published rule
#   ARM B  "exclusion off" I_R = {x'_ui = 0}              drops I_Q ∩ I_R = ∅ ONLY
#   ARM A  "no self-eating" fixed 20% eval side, never answerable; I_R = I \ (ask-side rated)
#
# In every arm the gains are 2^r - 1 and IDCG is taken over I_R (their definition).
# =============================================================================================
ARMS = ("C_theirs", "B_no_ask_excl", "A_heldout")


def eval_cell(scores_fn, asked_k, Rrev, Reval, tail, n_items, arm, ublock=256):
    """scores_fn(lo,hi) -> (B, n_items) float32 ; Rrev = CSR of ratings the interview revealed
    (already restricted to the asked items) ; Reval = CSR of the target ratings."""
    U = Reval.shape[0]
    full = np.zeros(U); tl = np.zeros(U)
    okf = np.zeros(U, bool); okt = np.zeros(U, bool)
    nrev = np.zeros(U); ncand = np.zeros(U)
    for s in range(0, U, ublock):
        e = min(s + ublock, U); B = e - s
        rows = np.arange(B)[:, None]
        sc = scores_fn(s, e)
        tgt_idx, tgt_gain = [], []
        for b in range(B):
            lo, hi = Reval.indptr[s + b], Reval.indptr[s + b + 1]
            tgt_idx.append(Reval.indices[lo:hi])
            tgt_gain.append(np.exp2(Reval.data[lo:hi].astype(np.float64)) - 1.0)
        cand = np.ones((B, n_items), dtype=bool)
        for b in range(B):
            lo, hi = Rrev.indptr[s + b], Rrev.indptr[s + b + 1]
            cand[b, Rrev.indices[lo:hi]] = False
            nrev[s + b] = hi - lo
        if arm == "C_theirs":
            cand[rows, asked_k[s:e]] = False       # I_Q ∩ I_R = ∅ -- their extra rule
        f, of_ = ndcg_at_k_block(sc, tgt_idx, tgt_gain, cand)
        t_, ot_ = ndcg_at_k_block(sc, tgt_idx, tgt_gain, cand & tail[None, :])
        full[s:e], okf[s:e] = f, of_
        tl[s:e], okt[s:e] = t_, ot_
        ncand[s:e] = cand.sum(1)
    return dict(full=float(full[okf].mean()), tail=float(tl[okt].mean()),
                n_full=int(okf.sum()), n_tail=int(okt.sum()),
                revealed_per_user=float(nrev.mean()), cand_per_user=float(ncand.mean()),
                per_user_full=full, per_user_tail=tl, ok_full=okf, ok_tail=okt)


def write_cell(outpath, key, cell, log=None):
    """HARD RULE 11a: one cell, one write, the moment it completes.
    Per-user vectors go to a sibling .npz so arms can be differenced PAIRED, per user -- a
    matching mean is never a licence to difference."""
    pu = outpath[:-5] + "_per_user"
    os.makedirs(pu, exist_ok=True)
    np.savez_compressed(os.path.join(pu, key.replace("|", "__") + ".npz"),
                        full=cell["per_user_full"], tail=cell["per_user_tail"],
                        ok_full=cell["ok_full"], ok_tail=cell["ok_tail"])
    doc = {}
    if os.path.exists(outpath):
        with open(outpath, encoding="utf-8") as fh:
            doc = json.load(fh)
    doc[key] = {k: v for k, v in cell.items() if not isinstance(v, np.ndarray)}
    tmp = outpath + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(doc, fh, indent=1)
    os.replace(tmp, outpath)
    logln("  CELL %-42s full=%.4f tail=%.4f  rev/u=%.2f" %
          (key, cell["full"], cell["tail"], cell["revealed_per_user"]), log)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--label", required=False, default=None,
                    help="unique run label; stamps every output path (HARD RULE 11e)")
    ap.add_argument("--questionnaires", default="qgslim,qbandit,qpop")
    ap.add_argument("--arms", default=",".join(ARMS))
    ap.add_argument("--budgets", default="5,10,15,20")
    ap.add_argument("--svd_rank", type=int, default=50)
    ap.add_argument("--seed", type=int, default=98765)
    ap.add_argument("--smoke_users", type=int, default=0,
                    help="SMOKE ONLY -- stamps SMOKE_NOT_CITABLE and refuses a citable filename")
    a = ap.parse_args()
    if a.probe:
        os.makedirs(OUTDIR, exist_ok=True)
        lg = os.path.join(OUTDIR, "probe.log")
        X = load_ml25m(lg)
        tr, te = split_users(X.shape[0])
        logln("split: train=%d test=%d" % (len(tr), len(te)), lg)
        Xtr = X[tr]
        hd, nh = head_mask(Xtr)
        logln("head items = %d of %d (33%% of train ratings)" % (nh, X.shape[1]), lg)
        pl = np.diff(X.indptr)
        logln("ratings/user: mean=%.1f median=%d max=%d" % (pl.mean(), np.median(pl), pl.max()), lg)

    if a.run:
        if not a.label:
            raise SystemExit("--label is required (HARD RULE 11e: unique run label)")
        label = ("SMOKE_%d_" % a.smoke_users if a.smoke_users else "") + a.label
        os.makedirs(OUTDIR, exist_ok=True)
        lg = os.path.join(OUTDIR, "%s.log" % label)
        outp = os.path.join(OUTDIR, "%s.json" % label)
        lock = outp + ".lock"
        if os.path.exists(lock):
            raise SystemExit("output %s is held by a live process (%s) -- refusing (HARD RULE 11e)"
                             % (outp, open(lock).read().strip()))
        with open(lock, "w") as fh:
            fh.write("pid=%d started=%s" % (os.getpid(), time.strftime("%Y-%m-%d %H:%M:%S")))
        try:
            budgets = [int(x) for x in a.budgets.split(",")]
            qs = [q.strip() for q in a.questionnaires.split(",") if q.strip()]
            arms = [x.strip() for x in a.arms.split(",") if x.strip()]
            bad = [x for x in arms if x not in ARMS]
            if bad:                       # validate names AT STARTUP, not at first use
                raise SystemExit("unknown arm(s) %s; valid: %s" % (bad, list(ARMS)))
            bad = [q for q in qs if q not in ("qgslim", "qbandit", "qpop")]
            if bad:
                raise SystemExit("unknown questionnaire(s) %s" % bad)
            kmax = max(budgets)
            logln("LABEL=%s  questionnaires=%s arms=%s budgets=%s svd_rank=%d%s"
                  % (label, qs, arms, budgets, a.svd_rank,
                     "  *** SMOKE_NOT_CITABLE ***" if a.smoke_users else ""), lg)

            X = load_ml25m(lg)
            tr, te = split_users(X.shape[0], a.seed)
            Xtr = X[tr].astype(np.float32)
            Rte = X[te].astype(np.float32)
            if a.smoke_users:
                Rte = Rte[:a.smoke_users]
            U, n_items = Rte.shape
            hd, nh = head_mask(Xtr)
            tail = ~hd
            logln("train=%d test=%d items=%d head=%d tail=%d" % (Xtr.shape[0], U, n_items, nh,
                                                                 int(tail.sum())), lg)

            # ---- ARM A: fixed 80/20 per-user split of the TEST user's ratings, before any ask
            rngA = np.random.RandomState(a.seed + 1)
            keep = rngA.rand(Rte.nnz) >= 0.20
            # NB copy indices/indptr -- eliminate_zeros() mutates in place and would corrupt Rte
            Rask = sp.csr_matrix((Rte.data * keep, Rte.indices.copy(), Rte.indptr.copy()),
                                 shape=Rte.shape)
            Rask.eliminate_zeros()
            Reval_A = sp.csr_matrix((Rte.data * (~keep), Rte.indices.copy(), Rte.indptr.copy()),
                                    shape=Rte.shape)
            Reval_A.eliminate_zeros()
            assert Rask.multiply(Reval_A).nnz == 0, "ARM A ask/eval sides overlap"

            V = pure_svd(Xtr, a.svd_rank, lg)
            Vt = np.ascontiguousarray(V.T)

            prior = empirical_prior(Xtr, V, lg)
            base, Vtil = prior
            VtilT = np.ascontiguousarray(Vtil.T)
            Id = np.eye(Vtil.shape[1])

            def posterior_mu(Rrev, sigma2=1.0):
                """Posterior mean of the user's WHITENED latent vector (prior N(0,I)).
                The SAME recommender for qbandit and qpop -> the questionnaire is the only thing
                that differs between them, which is what makes qpop a clean control for
                'asks popular items' as distinct from 'is adaptive'.  A user who answered
                nothing keeps alpha = 0, i.e. the population-average ranking."""
                mu = np.zeros((Rrev.shape[0], Vtil.shape[1]))
                for u in range(Rrev.shape[0]):
                    lo, hi = Rrev.indptr[u], Rrev.indptr[u + 1]
                    if hi == lo:
                        continue
                    idx = Rrev.indices[lo:hi]
                    vv = Vtil[idx].astype(np.float64)
                    r = Rrev.data[lo:hi].astype(np.float64) - base[idx]
                    mu[u] = np.linalg.solve(Id + vv.T @ vv / sigma2, vv.T @ r / sigma2)
                return mu.astype(np.float32)

            # ---- questionnaires ------------------------------------------------------------
            ASK = {}
            if "qgslim" in qs:
                G = build_gram(Xtr, log=lg)
                sel, Wrows = greedy_slim(Xtr, G, kmax, log=lg)
                del G
                logln("QGSLIM order: %s" % sel.tolist(), lg)
                ASK["qgslim"] = np.repeat(sel[None, :], U, axis=0)
            if "qpop" in qs:
                pop = q_pop(Xtr, kmax)
                logln("QPOP order: %s" % pop.tolist(), lg)
                ASK["qpop"] = np.repeat(pop[None, :], U, axis=0)

            qb_cache = {}
            for arm in arms:
                Rsrc = Rask if arm == "A_heldout" else Rte           # what the interview may reveal
                Rtgt = Reval_A if arm == "A_heldout" else Rte        # the ground truth
                if "qbandit" in qs:
                    # ARMS C and B differ ONLY in the candidate rule -- the interview they induce
                    # is bit-identical (same source, same seed).  Compute it once and assert it.
                    ckey = "A" if arm == "A_heldout" else "CB"
                    if ckey not in qb_cache:
                        t0 = time.time()
                        ab, _ = qbandit_interview(Rsrc, V, kmax, prior, seed=a.seed, log=lg)
                        qb_cache[ckey] = ab
                        logln("qbandit interview (%s) %.1f min" % (ckey, (time.time() - t0) / 60), lg)
                    ASK["qbandit"] = qb_cache[ckey]
                for q in qs:
                    asked_full = ASK[q]
                    for k in budgets:
                        ckey2 = "%s|%s|k%d" % (arm, q, k)
                        if os.path.exists(outp):
                            with open(outp, encoding="utf-8") as fh:
                                if ckey2 in json.load(fh):
                                    logln("  RESUME skip %s" % ckey2, lg)
                                    continue
                        t0 = time.time()
                        asked_k = np.ascontiguousarray(asked_full[:, :k])
                        # revealed = asked AND rated on the allowed side
                        rows = np.repeat(np.arange(U), k)
                        vals = csr_lookup(Rsrc, rows, asked_k.ravel())
                        xprime = sp.csr_matrix((vals, (rows, asked_k.ravel())), shape=(U, n_items))
                        xprime.sum_duplicates(); xprime.eliminate_zeros()
                        if arm == "A_heldout":
                            Rrev_mask = Rask                # policy B: mask the whole ask side
                        else:
                            Rrev_mask = xprime
                        if q == "qgslim":
                            coef = xprime[:, sel[:k]].toarray()
                            Wk = Wrows[:k]
                            def sfn(lo, hi, coef=coef, Wk=Wk):
                                return coef[lo:hi] @ Wk
                        else:
                            mu = posterior_mu(xprime)
                            def sfn(lo, hi, mu=mu):
                                return base[None, :] + mu[lo:hi] @ VtilT
                        cell = eval_cell(sfn, asked_k, Rrev_mask, Rtgt, tail, n_items, arm)
                        cell["k"] = k; cell["arm"] = arm; cell["questionnaire"] = q
                        cell["answer_rate"] = float(xprime.nnz) / (U * k)
                        cell["minutes"] = round((time.time() - t0) / 60, 2)
                        if a.smoke_users:
                            cell["SMOKE_NOT_CITABLE"] = True
                        pub = PUBLISHED.get((q, "all"), {}).get(k)
                        if pub and arm == "C_theirs":
                            cell["published_full"] = pub
                            cell["delta_full"] = round(cell["full"] - pub, 4)
                            cell["published_tail"] = PUBLISHED[(q, "tail")][k]
                            cell["delta_tail"] = round(cell["tail"] - PUBLISHED[(q, "tail")][k], 4)
                        write_cell(outp, "%s|%s|k%d" % (arm, q, k), cell, lg)
            logln("DONE -> %s" % outp, lg)
        finally:
            if os.path.exists(lock):
                os.remove(lock)
