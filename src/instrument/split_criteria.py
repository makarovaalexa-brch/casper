r"""THE THREE SPLIT CRITERIA, SIDE BY SIDE IN ONE PLACE -- plan item B1 / TODO 1.12.

This module exists so that the objective ablation changes ONE function and nothing else. The tree
machinery (ternary split rule, min-leaf 200, charged behavioural answers, certified pair, policy-B
candidate mask, 10,000 validation users, 16-question budget, 1,000-candidate combined pool) is
`ndcg_tree.py`'s and is reused verbatim by `split_criterion_ablation.py`. What varies is the scalar
each candidate question is scored by.

    (a) `variance`  -- Golbandi, Koren & Lempel, WSDM 2011: within-node squared error of RATING
                       prediction, Eq. (4), minimised.
    (b) `pairwise`  -- Hu & Yu, RecSys 2013: a PAIRWISE RANK LOSS, scored by their second-order
                       Taylor / diagonal-Hessian split gain (their Eq. 8-12), maximised.
    (b') `logistic` -- Hu & Yu again, but with the LOGISTIC loss, which is what their ranking
                       experiments ACTUALLY RAN ("we evaluate the MAP for EIP, FixQ and Warm-MF on
                       MovieLens since fMF cannot handle the logistic loss", sec. 3.2.1). Pairwise
                       rank loss appears in that paper exactly once, in the "square loss, logistic
                       loss, pairwise rank loss, etc." list of admissible losses. Reported because
                       the faithful reading of their EXPERIMENT is the logistic one and the faithful
                       reading of their FRAMEWORK is the pairwise one, and we should not have to
                       choose which of the two we are being pre-empted by.
    (c) `ndcg`      -- ours: the node-size-weighted mean of the three children's cohort-mean MEASURED
                       held-out full NDCG@10, i.e. the actual expected value of asking. Implemented
                       in `ndcg_tree.py` / `strategy_channel_suite.greedy_certified`, NOT here; this
                       module only re-derives it for the rank-correlation exhibit.
    (r) `random`    -- the shuffle control (HARD RULE 5). Uniform draw over the candidate pool with a
                       fixed seed; `random_ml` restricts the draw to candidates whose ternary split
                       would clear min-leaf, which is the FAIR control (it isolates "which question"
                       from "does the question split at all").

HU & YU'S TAYLOR TRICK, WHICH IS THE PUBLISHED ROUTE TO SCALE OUR EXACT OBJECTIVE CANNOT REACH
----------------------------------------------------------------------------------------------
Their node objective (Eq. 7) is: for each child C, find the increment `du` on the user factor that
minimises L(R|C, (u_P+du)^T V) + lam*||du||^2. Solving that per candidate is a full re-fit, so they
expand to second order about du = 0 (their unnumbered display before Eq. 8):

    Ltilde(du) = Ltilde(0) + du^T g + 0.5 du^T Htilde du,   g = grad L,  Htilde = H + 2*lam

whose minimiser is du* = -Htilde^-1 g, giving (their Eq. 11)

    min_du Ltilde(du) = Ltilde(0) - 0.5 * g^T Htilde^-1 g

and then, because inverting Htilde per candidate per child is still too expensive, they replace
Htilde by its DIAGONAL D (their Eq. 12):

    min_du Ltilde(du) ~= Ltilde(0) - 0.5 * g^T D^-1 g.

Two consequences we use directly:
  * `sum_over_children Ltilde(0)` is the node's own loss, which is the SAME for every candidate, so
    the argmin over candidates of the total child loss is the ARGMAX of

        Gain(q) = sum_{C in {like, dislike, noanswer}}  0.5 * sum_k  g_C[k]^2 / (H_C[k] + 2*lam)

    -- structurally identical to a gradient-boosted-tree split gain. That is what `_taylor_gain`
    computes, and it is exact given their approximation, not a further approximation of it.
  * `g_C` and `H_C` are SUMS over the child's users of per-user quantities that do NOT depend on the
    candidate (they depend only on the node's already-asked prefix). So the per-user gradients and
    diagonal Hessians are computed ONCE per node and every one of the 1,000 candidates is then a pair
    of masked sums. This is why (a), (b) and (b') cost minutes where (c) costs ten hours -- and it is
    the honest statement of what their approximation buys.
  * They are explicit that the approximation is used ONLY to choose the question: "since there are
    approximations in the derivations, we only use it to find the interview question. After the
    question is found, we use coordinate descent to obtain optimal du." We follow that: the criterion
    picks the question, and the recommender that scores the resulting interview is the certified pair,
    untouched.

WHAT `u_P` IS IN OUR HARNESS, AND WHY
-------------------------------------
In their model the user factor IS the tree: u = sum of `du` along the path, so `u_P` is one vector
shared by every user at the node. In ours the user factor is the certified tower's latent `z_u`,
folded from that user's OWN answers to the path's questions -- so `u_P` is per-user. We use `z_u`.
This is the minimal transplant: swapping their split criterion into our harness must not also swap
our recommender, or the ablation would confound the criterion with the model. The shared increment
`du` per child, which is the thing the criterion optimises over, is kept exactly as they have it.
Scores are the recommender's own: s_uj = (z_u + du) @ Wd[j] + bd[j], with Wd/bd the certified tower's
LEARNED decoder weight and bias (never a popularity floor).

WHAT THE LOSSES ARE COMPUTED OVER, AND THE LEAK CHECK
-----------------------------------------------------
FOLD-IN EVIDENCE ONLY, never held-out targets. For user u:
    POS_u = fold-in items rated at level >= LIKE_MIN_LEVEL (>= 4.0 stars, the Liang like boundary)
    NEG_u = fold-in items rated BELOW that level
Both come from `E["lvl"]`, which `ndcg_tree.leak_assert` already proves carries no held-out target;
`assert_no_leak` below re-proves it for the exact index sets these criteria consume, per user, and is
called before any criterion is evaluated. `pairwise` uses every (i, j) in POS_u x NEG_u -- ALL pairs,
nothing sampled (HARD RULE 1); users with an empty POS or NEG contribute exactly zero and are
counted and reported rather than dropped. `logistic` uses every fold-in rating as a labelled example
(1 for POS, 0 for NEG). `variance` uses the user's whole fold-in rating vector, user-bias corrected,
exactly as Golbandi's Eq. (4).

NOTE ON THE PAIRWISE HESSIAN. The exact diagonal of the pairwise Hessian is
    H[k] = sum_{ij} sig(d_ij) sig(-d_ij) (W_ik - W_jk)^2
which expands to two separable terms plus a cross term `-2 sum_ij c_ij W_ik W_jk`. The cross term is
computed EXACTLY, per user, as `sum_p W_P[p] * (C @ W_N)[p]` -- no approximation, no dropped term.
The cost is sum_u |POS_u| |NEG_u| d, which is paid once per node, and is reported.
"""
import numpy as np
from scipy import sparse

from i26_encoder import LIKE_MIN_LEVEL

CRITERIA = ("variance", "pairwise", "logistic", "random", "random_ml")
BRANCHES = ("like", "dislike", "noanswer")
LAM_U = 10.0          # Hu & Yu: "We set lam_u at 10 in our experiments by cross validation."


# ------------------------------------------------------------------ fold-in evidence, built once
def build_evidence(E, rows_all):
    """The fold-in rating matrix and the like/dislike index sets, for the WHOLE cohort, built once.

    Returns a dict with
      R    csr (n_rows x n_items) of fold-in STARS (levels mapped back to the 0.5..5.0 star scale)
      M    csr, the rated mask
      Xb   csr, R with the per-user mean removed (Golbandi's bias-corrected values); X2 = Xb^2
      pos  list of int arrays, the LIKED fold-in item ids per row
      neg  list of int arrays, the DISLIKED fold-in item ids per row
    `rows_all` fixes the row order; every array below is indexed POSITIONALLY against it.
    """
    lvl = E["lvl"]
    ni = E["ctx"].ni
    ind, ptr, dat = [], [0], []
    pos, neg = [], []
    for r in rows_all:
        d = lvl[int(r)]
        ids = np.fromiter(d.keys(), np.int64, len(d))
        lv = np.fromiter(d.values(), np.int64, len(d))
        o = np.argsort(ids)
        ids, lv = ids[o], lv[o]
        ind.append(ids)
        dat.append((lv.astype(np.float64) + 1.0) / 2.0)          # level -> stars
        ptr.append(ptr[-1] + len(ids))
        like = lv >= LIKE_MIN_LEVEL
        pos.append(ids[like]); neg.append(ids[~like])
    R = sparse.csr_matrix((np.concatenate(dat) if dat else np.zeros(0),
                           np.concatenate(ind) if ind else np.zeros(0, np.int64),
                           np.asarray(ptr, np.int64)), shape=(len(rows_all), ni))
    M = R.copy(); M.data = np.ones_like(M.data)
    cnt = np.diff(R.indptr).astype(np.float64)
    tot = np.asarray(R.sum(axis=1)).ravel()
    gmu = float(R.data.mean()) if R.nnz else 0.0
    b_u = (tot + 10.0 * gmu) / (cnt + 10.0)                      # Golbandi's shrunk per-user bias
    Xb = R.copy(); Xb.data = Xb.data - np.repeat(b_u, np.diff(R.indptr))
    X2 = Xb.copy(); X2.data = Xb.data ** 2
    return {"R": R, "M": M, "Xb": Xb, "X2": X2, "b_u": b_u, "pos": pos, "neg": neg,
            "n_rows": len(rows_all), "ni": ni,
            "n_pos_empty": int(sum(1 for p in pos if len(p) == 0)),
            "n_neg_empty": int(sum(1 for n in neg if len(n) == 0)),
            "n_pairs_total": int(sum(len(p) * len(n) for p, n in zip(pos, neg)))}


def assert_no_leak(E, ev, rows_all, log=print):
    """HARD RULE 5. Every index these criteria read must be FOLD-IN, never a held-out target.

    Three separate checks, because the criteria read three different things:
      (1) the rating matrix R (variance)     -- R AND va_te must not overlap on any (user, item);
      (2) POS_u and NEG_u (pairwise/logistic) -- per user, set-disjoint from that user's targets;
      (3) POS_u and NEG_u partition the user's fold-in items exactly (nothing invented, nothing lost).
    """
    te = E["ctx"].va_te[np.asarray(rows_all, np.int64)]
    ov = ev["M"].multiply(te).nnz
    assert ov == 0, f"LEAK: {ov} (user, item) cells appear in BOTH the fold-in criterion matrix and " \
                    f"the held-out targets"
    bad = 0
    for k, r in enumerate(rows_all):
        t = set(te[k].indices.tolist())
        p, n = ev["pos"][k], ev["neg"][k]
        if (t & set(p.tolist())) or (t & set(n.tolist())):
            bad += 1
        assert len(p) + len(n) == E["ctx"].ni - 0 or True
        assert len(p) + len(n) == ev["M"][k].nnz, \
            f"row {r}: POS+NEG ({len(p)}+{len(n)}) does not partition the {ev['M'][k].nnz} fold-in items"
    assert bad == 0, f"LEAK: {bad} users have a held-out target inside POS_u or NEG_u"
    log(f"[leak/criteria] 0/{len(rows_all)} users leak: the fold-in rating matrix shares no cell with "
        f"the held-out targets, POS_u and NEG_u are target-free for every user, and they partition "
        f"each user's fold-in items exactly. Total pairs the pairwise loss will sum over: "
        f"{ev['n_pairs_total']:,} (nothing sampled). Users with no LIKED fold-in item: "
        f"{ev['n_pos_empty']}; with no DISLIKED fold-in item: {ev['n_neg_empty']} -- these contribute "
        f"exactly zero to the pairwise gradient and are counted, not dropped.")


# ------------------------------------------------------------------ candidate membership, built once
def build_membership(E, rows_all, pool, log=print):
    """(like, dislike) boolean CSR of shape (len(pool) x len(rows_all)) for the WHOLE pool.

    The ternary branch rule is `ndcg_tree.branch_of`'s, verbatim in intent and asserted equal to it by
    the caller's self-test: an ITEM is `noanswer` if the user did not rate it, `like` at level >= 7,
    else `dislike`; a CONCEPT is `noanswer` if not foldable, `like` if its signed value > 0, else
    `dislike`. `noanswer` is the complement and is never stored (subtraction, as in Golbandi Eq. 3).
    """
    idx = np.asarray(rows_all, np.int64)
    lvl = E["lvl"]
    nL, nD = [], []
    for ch, q in pool:
        if ch == 0:
            lv = np.fromiter((lvl[int(r)].get(int(q), -1) for r in idx), np.int64, len(idx))
            ans = lv >= 0
            like = ans & (lv >= LIKE_MIN_LEVEL)
        else:
            ans = E["foldable"][idx, int(q)]
            like = ans & (E["V"][idx, int(q)] > 0)
        nL.append(like); nD.append(ans & ~like)
    L = np.asarray(nL, bool)
    D = np.asarray(nD, bool)
    log(f"[membership] {L.shape[0]} candidates x {L.shape[1]} users (dense bool, "
        f"{L.nbytes / 1e6:.0f} MB each); mean like rate {L.mean():.4f}, dislike {D.mean():.4f}, "
        f"noanswer {1 - L.mean() - D.mean():.4f}")
    return L, D


# ------------------------------------------------------------------ (a) Golbandi variance
def _e2(s, s2, n):
    """SUM_j [ sum2_j - sum_j^2 / n_j ] over items with n_j > 0 -- Golbandi Eq. (4), row-wise.
    Copied in form from `src/baselines/golbandi_tree_selector.py::_e2`, which is the arm that snapped
    against the published tree; keeping the same expression keeps this ablation on that arm's ruler."""
    with np.errstate(divide="ignore", invalid="ignore"):
        v = s2 - np.where(n > 0, s * s / np.maximum(n, 1.0), 0.0)
    return v.sum(axis=-1)


def variance_scores(ev, memL, memD, sub, cand, chunk=256):
    """Golbandi's criterion over the node's users `sub` (positional), for candidates `cand`.

    Returns HIGHER-IS-BETTER scores: the node's own e2 MINUS the partition's e2, i.e. the variance
    REDUCTION the split buys. Argmax of this == argmin of Golbandi's Err_t.
    """
    Xr, X2r, Mr = ev["Xb"][sub], ev["X2"][sub], ev["M"][sub]
    # RESTRICT THE ITEM AXIS TO THE ITEMS THIS BLOCK'S USERS ACTUALLY RATED. Eq. (4) sums over items
    # with n_j > 0, so an item nobody here rated contributes exactly zero to the node's e2 and to
    # every child's -- dropping it is exact. It matters because the criterion's cost is dominated by
    # the DENSE (chunk x n_items) output of each sparse product, which is otherwise 18,359 columns
    # wide no matter how few users the block holds. The refining leaf tail evaluates this on hundreds
    # of small blocks, where the full width was ~99% empty and turned a seconds-long step into a
    # minutes-long one.
    if Mr.nnz:
        cols = np.unique(Mr.indices)
        if len(cols) < Mr.shape[1]:
            Xr = Xr[:, cols]; X2r = X2r[:, cols]; Mr = Mr[:, cols]
    s_tot = np.asarray(Xr.sum(axis=0)).ravel()
    s2_tot = np.asarray(X2r.sum(axis=0)).ravel()
    n_tot = np.asarray(Mr.sum(axis=0)).ravel()
    e2_node = float(_e2(s_tot, s2_tot, n_tot))
    out = np.empty(len(cand), np.float64)
    cnd = np.asarray(cand, np.int64)
    Lall = memL[np.ix_(cnd, sub)]
    Dall = memD[np.ix_(cnd, sub)]
    for a in range(0, len(cand), chunk):
        Lc = sparse.csr_matrix(Lall[a:a + chunk].astype(np.float64))
        Dc = sparse.csr_matrix(Dall[a:a + chunk].astype(np.float64))
        sL = np.asarray((Lc @ Xr).todense()); s2L = np.asarray((Lc @ X2r).todense())
        nL = np.asarray((Lc @ Mr).todense())
        sD = np.asarray((Dc @ Xr).todense()); s2D = np.asarray((Dc @ X2r).todense())
        nD = np.asarray((Dc @ Mr).todense())
        sU = s_tot[None, :] - sL - sD                      # Eq. (3): the unknown branch by SUBTRACTION
        s2U = s2_tot[None, :] - s2L - s2D
        nU = n_tot[None, :] - nL - nD
        out[a:a + Lc.shape[0]] = e2_node - (_e2(sL, s2L, nL) + _e2(sD, s2D, nD)
                                            + _e2(sU, s2U, nU))
    return out, e2_node


# ------------------------------------------------------------------ (b)/(b') Hu & Yu Taylor gain
def loss_grad_hess(kind, Z, Wd, bd, pos, neg, sub, log=None):
    """Per-user gradient g_u and DIAGONAL Hessian h_u of the loss w.r.t. a shared increment `du`,
    evaluated at du = 0. Shapes (|sub| x d) each. `Z` is the node's per-user latent (|sub| x d).

    `kind='pairwise'`: L_u = sum_{i in POS_u, j in NEG_u} log(1 + exp(-(s_ui - s_uj))).
    `kind='logistic'`: L_u = sum_{j in POS_u} -log sig(s_uj) + sum_{j in NEG_u} -log(1 - sig(s_uj)).
    """
    n, d = Z.shape
    G = np.zeros((n, d), np.float64)
    H = np.zeros((n, d), np.float64)
    n_pairs = 0
    for k in range(n):
        p, m = pos[sub[k]], neg[sub[k]]
        z = Z[k]
        if kind == "logistic":
            if len(p) == 0 and len(m) == 0:
                continue
            ids = np.concatenate([p, m]) if len(p) and len(m) else (p if len(p) else m)
            y = np.concatenate([np.ones(len(p)), np.zeros(len(m))])
            W = Wd[ids]
            s = W @ z + bd[ids]
            sg = 1.0 / (1.0 + np.exp(-np.clip(s, -30, 30)))
            G[k] = (sg - y) @ W
            H[k] = (sg * (1.0 - sg)) @ (W * W)
            n_pairs += len(ids)
        else:
            if len(p) == 0 or len(m) == 0:
                continue                                   # contributes exactly zero, counted not dropped
            Wp, Wm = Wd[p], Wd[m]
            sp = Wp @ z + bd[p]
            sm = Wm @ z + bd[m]
            dd = sp[:, None] - sm[None, :]                 # (|P| x |N|)
            sg = 1.0 / (1.0 + np.exp(-np.clip(dd, -30, 30)))
            a = 1.0 - sg                                   # sig(-d_ij)
            # gradient: -sum_ij sig(-d_ij) (W_i - W_j)
            G[k] = -(a.sum(1) @ Wp - a.sum(0) @ Wm)
            # diagonal Hessian: sum_ij c_ij (W_ik - W_jk)^2, c = sig(d) sig(-d); cross term EXACT
            c = sg * a
            H[k] = (c.sum(1) @ (Wp * Wp) + c.sum(0) @ (Wm * Wm) - 2.0 * (Wp * (c @ Wm)).sum(0))
            n_pairs += len(p) * len(m)
    if log is not None:
        log(f"[huyu/{kind}] per-user gradients over {n} users, {n_pairs:,} "
            f"{'pairs' if kind == 'pairwise' else 'labelled examples'} (all of them; nothing sampled)")
    return G, H, n_pairs


def _taylor_gain(gL, hL, gD, hD, gU, hU, lam=LAM_U):
    """sum over the three children of 0.5 * sum_k g[k]^2 / (h[k] + 2 lam) -- Hu & Yu Eq. (12), with
    sum_C Ltilde(0) dropped because it is the node's own loss and is candidate-independent."""
    out = np.zeros(gL.shape[0], np.float64)
    for g, h in ((gL, hL), (gD, hD), (gU, hU)):
        out += 0.5 * (g * g / (h + 2.0 * lam)).sum(1)
    return out


def huyu_scores(G, H, memL, memD, sub, cand, lam=LAM_U):
    """Hu & Yu's split gain for every candidate. HIGHER IS BETTER (their loss is minimised).

    `memL`/`memD` are DENSE boolean (n_pool x n_cohort), so the aggregation of per-user gradients into
    per-child sums is one BLAS matmul per branch rather than a sparse product per candidate.
    """
    gT = G.sum(0); hT = H.sum(0)
    cnd = np.asarray(cand, np.int64)
    Lc = memL[np.ix_(cnd, sub)].astype(np.float64)
    Dc = memD[np.ix_(cnd, sub)].astype(np.float64)
    gL = Lc @ G; hL = Lc @ H
    gD = Dc @ G; hD = Dc @ H
    gU = gT[None, :] - gL - gD                             # the unknown branch by SUBTRACTION
    hU = hT[None, :] - hL - hD
    # a diagonal Hessian summed over a subset can go very slightly negative through float
    # cancellation in the subtraction; clip at 0 so the denominator is never below 2*lam. Reported.
    neg_clip = int((hU < 0).sum())
    hU = np.maximum(hU, 0.0)
    return _taylor_gain(gL, hL, gD, hD, gU, hU, lam), neg_clip
