r"""Replication of Hu & Yu, "Interview Process Learning for Top-N Recommendation"
(RecSys 2013, short paper) on MovieLens-1M, under THEIR protocol.

Their method is EIP: layer-wise functional matrix factorisation.  The user latent
factor is the output of a ternary interview decision tree over the user's answers
(rating "higher than threshold" / "lower than threshold" / "Unknown"); the tree is
grown by an XGBoost-style gain derived from a second-order Taylor expansion with a
DIAGONAL Hessian (their Eq. 12); the chosen node's Delta-u is then re-fitted by
coordinate descent; item factors V are fitted by regularised coordinate descent,
alternating with the tree fit.  lambda_u = 10 (stated in the paper).

Loss: the paper says L "can be square loss, logistic loss, pairwise rank loss, etc.";
the ranking experiments they ACTUALLY ran used LOGISTIC loss ("we evaluate the MAP for
EIP, FixQ and Warm-MF on MovieLens since fMF cannot handle the logistic loss").  We run
both logistic and pairwise (BPR) rank loss.

Protocol (theirs, verbatim where stated):
  * users split into disjoint TRAIN / TEST sets;
  * each TEST user's ratings split in two: first half generates the interview
    responses, second half evaluates;
  * ranking: negatives sampled for training and test data separately; for test,
    "randomly sample negative items 10 times as positive items for each user";
  * MAP = (1/|U|) sum_i (1/|O_i|) sum_{j in O_i} P@(rank_i(j))   (their Eq. 13,
    UNTRUNCATED average precision, no positional discount).

Arms: EIP (logistic / pairwise / square), FixQ (top-k popular static interview,
answers as features in a feature-based MF), Warm-MF (train + answer half, free u),
fMF (Zhou et al. 2011: square loss, EXACT closed-form split criterion instead of the
diagonal-Taylor approximation), plus controls: most-popular (canonical snap) and
RandQ (shuffle control: random question + threshold at each node).

Usage
-----
  python scripts/baselines/huyu_eip_ml1m.py --stage sweep     # d / lambda_v selection
  python scripts/baselines/huyu_eip_ml1m.py --stage main      # the snap
  python scripts/baselines/huyu_eip_ml1m.py --stage controls

Every (arm, depth) cell is written to experiments/baselines/huyu2013/cells/ the
moment it completes.
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

ROOT = r"C:\dev\phd\casper"
RATINGS = os.path.join(ROOT, "data", "ml-1m", "ratings.dat")
OUT = os.path.join(ROOT, "experiments", "baselines", "huyu2013")
CELLS = os.path.join(OUT, "cells")

# ----------------------------------------------------------------------------- ruler
# Stated ONCE and reused by every arm (HARD RULE 8).
SEED_USER_SPLIT = 12345      # train / test user partition
SEED_HALF = 24680            # answer / evaluation half of each test user
SEED_TRAIN_NEG = 13579       # negatives for the training data
SEED_TEST_NEG = 20130101     # negatives for the test data  (paired across arms)
SEED_TIEBREAK = 4242         # deterministic candidate tie-break (paired across arms)
TEST_FRAC = 0.25
VAL_FRAC = 0.20              # carved out of TRAIN users for the d / lambda sweep only
NEG_PER_POS_TRAIN = 10
NEG_PER_POS_TEST = 10        # THEIR number
LAMBDA_U = 10.0              # theirs
MIN_LEAF = 30                # not specified by the paper
OUTER_ITERS = 4              # tree <-> V alternations; not specified
V_SWEEPS = 3                 # coordinate-descent sweeps over V per outer iteration
U_SWEEPS = 3                 # coordinate-descent sweeps for Delta-u per tree level
FIXQ_LIKE_THRESHOLD = 3      # FixQ's fixed "higher than" threshold (EIP searches it)

log_fh = None


def log(*a):
    msg = " ".join(str(x) for x in a)
    print(msg, flush=True)
    if log_fh is not None:
        log_fh.write(msg + "\n")
        log_fh.flush()


# ------------------------------------------------------------------------------ data
def load_ratings():
    cache = os.path.join(ROOT, "data", "ml-1m", "ratings_cache.npz")
    if os.path.exists(cache):
        z = np.load(cache)
        return z["u"], z["i"], z["r"]
    txt = open(RATINGS, "r", encoding="latin-1").read().replace("::", " ")
    arr = np.fromstring(txt, dtype=np.int64, sep=" ").reshape(-1, 4)
    uraw, iraw, r = arr[:, 0], arr[:, 1], arr[:, 2]
    uu, u = np.unique(uraw, return_inverse=True)
    ii, i = np.unique(iraw, return_inverse=True)
    u, i, r = u.astype(np.int32), i.astype(np.int32), r.astype(np.int8)
    np.savez(cache, u=u, i=i, r=r, users=uu, items=ii)
    return u, i, r


class Split:
    """Disjoint train/test users; test users' ratings halved into answer / eval."""

    def __init__(self, u, i, r, holdout_users=None, drop_users=None):
        self.n_users = int(u.max()) + 1
        self.n_items = int(i.max()) + 1
        rng = np.random.default_rng(SEED_USER_SPLIT)
        perm = rng.permutation(self.n_users)
        n_test = int(round(TEST_FRAC * self.n_users))
        test_users = np.sort(perm[:n_test])
        train_users = np.sort(perm[n_test:])
        if holdout_users is not None:      # sweep stage: pseudo-test cohort
            test_users = np.sort(holdout_users)
        if drop_users is not None:
            keep = np.ones(self.n_users, bool)
            keep[drop_users] = False
            train_users = train_users[keep[train_users]]
        self.train_users, self.test_users = train_users, test_users
        is_test = np.zeros(self.n_users, bool)
        is_test[test_users] = True
        is_train = np.zeros(self.n_users, bool)
        is_train[train_users] = True

        # answer / eval half for every user (deterministic, independent of which
        # cohort the user lands in, so the halves are identical across stages)
        rng2 = np.random.default_rng(SEED_HALF)
        half = np.zeros(len(u), np.int8)
        order = np.argsort(u, kind="stable")
        starts = np.searchsorted(u[order], np.arange(self.n_users))
        ends = np.searchsorted(u[order], np.arange(self.n_users), side="right")
        for uu in range(self.n_users):
            idx = order[starts[uu]:ends[uu]]
            n = len(idx)
            pick = rng2.permutation(n)
            half[idx[pick[: n // 2]]] = 0     # first set  -> interview responses
            half[idx[pick[n // 2:]]] = 1      # second set -> evaluation
        self.half = half

        m_train = is_train[u]
        m_ans = is_test[u] & (half == 0)
        m_eval = is_test[u] & (half == 1)
        self.tr = (u[m_train], i[m_train], r[m_train])
        self.ans = (u[m_ans], i[m_ans], r[m_ans])
        self.ev = (u[m_eval], i[m_eval], r[m_eval])
        # every (user,item) the user has touched -> excluded from negative sampling
        self.all_u, self.all_i = u, i

    def leak_check(self):
        """(i) halves disjoint, (ii) no eval rating in any training/interview set."""
        out = {}
        au, ai, _ = self.ans
        eu, ei, _ = self.ev
        tu, ti, _ = self.tr
        key = lambda a, b: a.astype(np.int64) * 100000 + b
        ka, ke, kt = key(au, ai), key(eu, ei), key(tu, ti)
        out["ans_eval_pair_overlap"] = int(np.intersect1d(ka, ke).size)
        out["train_eval_pair_overlap"] = int(np.intersect1d(kt, ke).size)
        out["train_test_user_overlap"] = int(
            np.intersect1d(self.train_users, self.test_users).size)
        out["eval_users_in_train_ratings"] = int(
            np.intersect1d(np.unique(tu), self.test_users).size)
        out["n_train_users"] = int(len(self.train_users))
        out["n_test_users"] = int(len(self.test_users))
        out["n_train_ratings"] = int(len(tu))
        out["n_answer_ratings"] = int(len(au))
        out["n_eval_ratings"] = int(len(eu))
        return out


# ------------------------------------------------------------------- example builders
def sample_negatives(u_arr, i_arr, n_users, n_items, rated_sets, ratio, seed,
                     users_subset=None):
    """`ratio` negatives per positive, drawn from items the user never rated."""
    rng = np.random.default_rng(seed)
    ru, ri = [], []
    order = np.argsort(u_arr, kind="stable")
    us = u_arr[order]
    uniq, starts = np.unique(us, return_index=True)
    ends = np.append(starts[1:], len(us))
    for k, uu in enumerate(uniq):
        npos = ends[k] - starts[k]
        need = npos * ratio
        seen = rated_sets[uu]
        cand = rng.integers(0, n_items, size=int(need * 1.4) + 16)
        cand = cand[~np.isin(cand, seen)]
        while len(cand) < need:
            more = rng.integers(0, n_items, size=int(need))
            more = more[~np.isin(more, seen)]
            cand = np.concatenate([cand, more])
        ru.append(np.full(need, uu, np.int32))
        ri.append(cand[:need].astype(np.int32))
    return np.concatenate(ru), np.concatenate(ri)


def rated_sets_of(u, i, n_users):
    order = np.argsort(u, kind="stable")
    us, isort = u[order], i[order]
    starts = np.searchsorted(us, np.arange(n_users))
    ends = np.searchsorted(us, np.arange(n_users), side="right")
    return [isort[starts[k]:ends[k]] for k in range(n_users)]


# ------------------------------------------------------------------------------ model
class Loss:
    """z = raw score; per-example user-side grad = c1 * w, diag hess = c2 * w^2."""

    def __init__(self, kind):
        self.kind = kind

    def coefs(self, z, y=None):
        if self.kind == "logistic":          # z already carries the label sign
            sg = 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))
            return -(1.0 - sg), sg * (1.0 - sg)
        if self.kind == "pairwise":
            sg = 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))
            return -(1.0 - sg), sg * (1.0 - sg)
        if self.kind == "square":
            return 2.0 * (z - y), np.full_like(z, 2.0)
        raise ValueError(self.kind)


class Data:
    """Example container.  Holds the E-length index arrays and the current scores."""

    def __init__(self, kind, n_users, n_items, d, **kw):
        self.kind, self.n_users, self.n_items, self.d = kind, n_users, n_items, d
        self.loss = Loss(kind)
        if kind in ("logistic", "square"):
            self.eu = kw["eu"].astype(np.int32)
            self.ei = kw["ei"].astype(np.int32)
            self.ey = kw["ey"].astype(np.float32)      # +-1 sign, or the rating
        else:
            self.eu = kw["eu"].astype(np.int32)
            self.ep = kw["ep"].astype(np.int32)
            self.en = kw["en"].astype(np.int32)
        self.E = len(self.eu)

    def w(self, V, k):
        if self.kind == "pairwise":
            return V[self.ep, k] - V[self.en, k]
        if self.kind == "logistic":
            return V[self.ei, k] * self.ey
        return V[self.ei, k]

    def z_of(self, U, V):
        z = np.zeros(self.E, np.float32)
        for k in range(self.d):
            z += U[self.eu, k] * self.w(V, k)
        return z

    def coefs(self, z):
        y = self.ey if self.kind == "square" else None
        return self.loss.coefs(z, y)


def user_grad_hess(data, U, V, z):
    """Per-user gradient and diagonal Hessian of the loss w.r.t. that user's u."""
    c1, c2 = data.coefs(z)
    G = np.zeros((data.n_users, data.d), np.float64)
    H = np.zeros((data.n_users, data.d), np.float64)
    for k in range(data.d):
        wk = data.w(V, k)
        G[:, k] = np.bincount(data.eu, weights=c1 * wk, minlength=data.n_users)
        H[:, k] = np.bincount(data.eu, weights=c2 * wk * wk, minlength=data.n_users)
    return G, H


def fit_delta_groups(data, V, z, group_of_user, n_groups, lam_u, sweeps,
                     delta=None, frozen=None):
    """Coordinate descent for one Delta-u per group (a tree LEVEL at a time)."""
    if delta is None:
        delta = np.zeros((n_groups, data.d), np.float64)
    ge = group_of_user[data.eu]
    ok = ge >= 0
    for _ in range(sweeps):
        for k in range(data.d):
            wk = data.w(V, k)
            c1, c2 = data.coefs(z)
            g = np.bincount(ge[ok], weights=(c1 * wk)[ok], minlength=n_groups)
            h = np.bincount(ge[ok], weights=(c2 * wk * wk)[ok], minlength=n_groups)
            g += 2.0 * lam_u * delta[:, k]
            h += 2.0 * lam_u
            step = -g / np.maximum(h, 1e-9)
            if frozen is not None:
                step[frozen] = 0.0
            delta[:, k] += step
            z[ok] += (step[ge[ok]] * wk[ok]).astype(np.float32)
    return delta, z


def update_V(data, U, V, z, lam_v, sweeps):
    """Regularised coordinate descent on V (their Eq. 5), one dimension at a time."""
    for _ in range(sweeps):
        for k in range(data.d):
            c1, c2 = data.coefs(z)
            uk = U[data.eu, k]
            if data.kind == "pairwise":
                gp = np.bincount(data.ep, weights=c1 * uk, minlength=data.n_items)
                gn = np.bincount(data.en, weights=-c1 * uk, minlength=data.n_items)
                hp = np.bincount(data.ep, weights=c2 * uk * uk, minlength=data.n_items)
                hn = np.bincount(data.en, weights=c2 * uk * uk, minlength=data.n_items)
                g, h = gp + gn, hp + hn
            else:
                sg = data.ey if data.kind == "logistic" else 1.0
                g = np.bincount(data.ei, weights=c1 * uk * sg, minlength=data.n_items)
                h = np.bincount(data.ei, weights=c2 * uk * uk, minlength=data.n_items)
            g = g + 2.0 * lam_v * V[:, k]
            h = h + 2.0 * lam_v
            step = -g / np.maximum(h, 1e-9)
            V[:, k] += step
            if data.kind == "pairwise":
                z += (uk * (step[data.ep] - step[data.en])).astype(np.float32)
            else:
                sg = data.ey if data.kind == "logistic" else 1.0
                z += (uk * step[data.ei] * sg).astype(np.float32)
    return V, z


# ------------------------------------------------------------------------------- tree
class Interview:
    """The fitted interview: nodes with (item, threshold) and a cumulative u."""

    def __init__(self, d):
        self.q = []          # item asked at this node (-1 = terminal)
        self.t = []          # rating threshold ("higher than t")
        self.ch = []         # (left, right, mid) child ids
        self.u = []          # cumulative user factor at this node
        self.d = d

    def add(self, u):
        self.q.append(-1); self.t.append(0); self.ch.append((-1, -1, -1))
        self.u.append(np.asarray(u, np.float64).copy())
        return len(self.q) - 1

    def traverse(self, ans_by_user, users):
        """ans_by_user[user] = dict item -> rating (the ANSWER half only)."""
        U = np.zeros((len(users), self.d))
        depth_reached = np.zeros(len(users), int)
        for row, uu in enumerate(users):
            node, dep = 0, 0
            a = ans_by_user[uu]
            while self.q[node] >= 0:
                r = a.get(self.q[node], None)
                l, rr, m = self.ch[node]
                node = m if r is None else (rr if r > self.t[node] else l)
                dep += 1
            U[row] = self.u[node]
            depth_reached[row] = dep
        return U, depth_reached


def _split_scores(Gnode, Hnode, cnt, Gtot, Htot, ntot, lam_u, min_leaf, banned):
    """Diagonal-Taylor gain (their Eq. 12) for every (item, threshold) candidate."""
    M, _, d = Gnode.shape
    Grated, Hrated = Gnode.sum(1), Hnode.sum(1)
    crated = cnt.sum(1)
    Gm, Hm = Gtot - Grated, Htot - Hrated
    cm = ntot - crated
    best = (-np.inf, -1, -1)
    two = 2.0 * lam_u
    for t in range(1, 5):
        Gl, Hl = Gnode[:, :t].sum(1), Hnode[:, :t].sum(1)
        cl = cnt[:, :t].sum(1)
        Gr, Hr = Grated - Gl, Hrated - Hl
        cr = crated - cl
        sc = 0.5 * ((Gl ** 2 / (Hl + two)).sum(1)
                    + (Gr ** 2 / (Hr + two)).sum(1)
                    + (Gm ** 2 / (Hm + two)).sum(1))
        bad = (cl < min_leaf) | (cr < min_leaf) | ((cm > 0) & (cm < min_leaf))
        sc[bad] = -np.inf
        sc[banned] = -np.inf
        j = int(np.argmax(sc))
        if sc[j] > best[0]:
            best = (float(sc[j]), j, t)
    return best


def _split_scores_exact(Anode, bnode, cnt, Atot, btot, ntot, lam_u, min_leaf, banned):
    """fMF (Zhou et al. 2011): EXACT closed-form square-loss reduction per split."""
    M, _, d, _ = Anode.shape
    Ar, br = Anode.sum(1), bnode.sum(1)
    crated = cnt.sum(1)
    Am, bm = Atot - Ar, btot - br
    cm = ntot - crated
    eye = np.eye(d) * lam_u
    best = (-np.inf, -1, -1)

    def quad(A, b):
        sol = np.linalg.solve(A + eye, b[..., None])[..., 0]
        return (b * sol).sum(-1)

    for t in range(1, 5):
        Al, bl = Anode[:, :t].sum(1), bnode[:, :t].sum(1)
        cl = cnt[:, :t].sum(1)
        Arr, brr = Ar - Al, br - bl
        cr = crated - cl
        sc = quad(Al, bl) + quad(Arr, brr) + quad(Am, bm)
        bad = (cl < min_leaf) | (cr < min_leaf) | ((cm > 0) & (cm < min_leaf))
        sc[bad] = -np.inf
        sc[banned] = -np.inf
        j = int(np.argmax(sc))
        if sc[j] > best[0]:
            best = (float(sc[j]), j, t)
    return best


def fit_tree(data, V, depth, rec, n_users, lam_u, exact=False, random_q=False,
             rng=None):
    """Grow the ternary interview tree.  `rec` = (user, item, rating) train records."""
    d = data.d
    rec_u, rec_i, rec_r = rec
    M = data.n_items
    tree = Interview(d)
    node_of_user = np.full(n_users, -1, np.int64)
    node_of_user[np.unique(data.eu)] = 0
    U = np.zeros((n_users, d))
    z = data.z_of(U, V)
    delta, z = fit_delta_groups(data, V, z, node_of_user, 1, lam_u, U_SWEEPS)
    U[node_of_user == 0] = delta[0]
    root = tree.add(delta[0])
    live = {0: root}                       # group id at this level -> tree node id
    banned_of = {0: set()}
    terminal = set()

    for level in range(depth):
        z = data.z_of(U, V)
        G, H = user_grad_hess(data, U, V, z)
        Auser = user_gram(data, V) if exact else None
        m = node_of_user >= 0
        gid = node_of_user
        rec_g = gid[rec_u]
        order = np.argsort(rec_g, kind="stable")
        rg = rec_g[order]
        bounds = {}
        uniq, st = np.unique(rg[rg >= 0], return_index=True)
        st = st + int((rg < 0).sum())
        en = np.append(st[1:], len(rg))
        for k, g in enumerate(uniq):
            bounds[int(g)] = (st[k], en[k])

        new_group = np.full(n_users, -1, np.int64)
        n_new, new_nodes, new_banned = 0, {}, {}
        splits = []
        for g, node in sorted(live.items()):
            users_g = np.where(gid == g)[0]
            ntot = len(users_g)
            Gtot, Htot = G[users_g].sum(0), H[users_g].sum(0)
            if ntot < 2 * MIN_LEAF or g not in bounds:
                terminal.add(node)
                new_group[users_g] = -1
                continue
            s, e = bounds[g]
            sl = order[s:e]
            it, rv, uu = rec_i[sl], rec_r[sl], rec_u[sl]
            idx = it.astype(np.int64) * 5 + (rv - 1)
            cnt = np.bincount(idx, minlength=M * 5).reshape(M, 5)
            banned = np.zeros(M, bool)
            banned[list(banned_of[g])] = True
            if exact:
                # fMF: EXACT closed-form square-loss reduction needs, per user,
                # A_i = sum_j v_j v_j^T and b_i = sum_j (r_ij - u_P.v_j) v_j.
                # Both are additive over users, so group them by (item, rating).
                A = np.zeros((M * 5, d, d))
                b = np.zeros((M * 5, d))
                for k1 in range(d):
                    b[:, k1] = np.bincount(idx, weights=-0.5 * G[uu, k1],
                                           minlength=M * 5)
                    for k2 in range(k1, d):
                        val = np.bincount(idx, weights=Auser[uu, k1, k2],
                                          minlength=M * 5)
                        A[:, k1, k2] = val
                        A[:, k2, k1] = val
                Atot = Auser[users_g].sum(0)
                btot = -0.5 * G[users_g].sum(0)
                sc, q, t = _split_scores_exact(A.reshape(M, 5, d, d),
                                               b.reshape(M, 5, d), cnt, Atot, btot,
                                               ntot, lam_u, MIN_LEAF, banned)
            else:
                Gn = np.zeros((M * 5, d))
                Hn = np.zeros((M * 5, d))
                for k in range(d):
                    Gn[:, k] = np.bincount(idx, weights=G[uu, k], minlength=M * 5)
                    Hn[:, k] = np.bincount(idx, weights=H[uu, k], minlength=M * 5)
                sc, q, t = _split_scores(Gn.reshape(M, 5, d), Hn.reshape(M, 5, d), cnt,
                                         Gtot, Htot, ntot, lam_u, MIN_LEAF, banned)
            if q < 0:
                terminal.add(node)
                new_group[users_g] = -1
                continue
            if random_q:                     # SHUFFLE CONTROL
                cand = []
                for tt in range(1, 5):
                    cl = cnt[:, :tt].sum(1)
                    cr = cnt.sum(1) - cl
                    cmid = ntot - cnt.sum(1)
                    ok = np.where((cl >= MIN_LEAF) & (cr >= MIN_LEAF)
                                  & ((cmid == 0) | (cmid >= MIN_LEAF)) & ~banned)[0]
                    cand += [(int(x), tt) for x in ok]
                if not cand:
                    terminal.add(node)
                    new_group[users_g] = -1
                    continue
                q, t = cand[rng.integers(len(cand))]
            # partition
            rq = np.full(n_users, 0, np.int8)          # 0 unknown
            sel = it == q
            rq[uu[sel]] = rv[sel]
            isr = rq[users_g]
            gl, gr, gm = n_new, n_new + 1, n_new + 2
            n_new += 3
            new_group[users_g[(isr > 0) & (isr <= t)]] = gl
            new_group[users_g[isr > t]] = gr
            new_group[users_g[isr == 0]] = gm
            splits.append((node, q, t, gl, gr, gm))
            for gg in (gl, gr, gm):
                new_banned[gg] = banned_of[g] | {q}

        if n_new == 0:
            break
        z = data.z_of(U, V)
        delta, z = fit_delta_groups(data, V, z, new_group, n_new, lam_u, U_SWEEPS)
        for node, q, t, gl, gr, gm in splits:
            ids = []
            for gg in (gl, gr, gm):
                nid = tree.add(tree.u[node] + delta[gg])
                new_nodes[gg] = nid
                ids.append(nid)
            tree.q[node], tree.t[node] = q, t
            tree.ch[node] = tuple(ids)
        upd = new_group >= 0
        U[upd] += delta[new_group[upd]]
        node_of_user = new_group
        gid = node_of_user
        live = {g: new_nodes[g] for g in new_nodes}
        banned_of = new_banned
    return tree, U


# ------------------------------------------------------------------------ evaluation
class Evaluator:
    """Their Eq. 13 MAP on 10 sampled negatives per positive, paired across arms."""

    def __init__(self, split, n_items):
        eu, ei, er = split.ev
        self.users = np.unique(eu)
        rated = rated_sets_of(split.all_u, split.all_i, split.n_users)
        rng = np.random.default_rng(SEED_TEST_NEG)
        rng_tb = np.random.default_rng(SEED_TIEBREAK)
        order = np.argsort(eu, kind="stable")
        eu_s, ei_s, er_s = eu[order], ei[order], er[order]
        st = np.searchsorted(eu_s, self.users)
        en = np.searchsorted(eu_s, self.users, side="right")
        self.cand, self.npos, self.pos_items, self.pos_r = [], [], [], []
        for k, uu in enumerate(self.users):
            pos = ei_s[st[k]:en[k]]
            need = len(pos) * NEG_PER_POS_TEST
            seen = rated[uu]
            neg = np.empty(0, np.int64)
            while len(neg) < need:
                c = rng.integers(0, n_items, size=int(need * 1.5) + 16)
                c = c[~np.isin(c, seen)]
                neg = np.concatenate([neg, c])
            neg = neg[:need]
            self.cand.append(np.concatenate([pos, neg]).astype(np.int32))
            self.npos.append(len(pos))
            self.pos_items.append(pos.astype(np.int32))
            self.pos_r.append(er_s[st[k]:en[k]].astype(np.float64))
        self.tb = [rng_tb.random(len(c)) for c in self.cand]

    def map_score(self, U):
        aps = np.empty(len(self.users))
        for k in range(len(self.users)):
            c = self.cand[k]
            s = U[k] @ np.ascontiguousarray(self.V[c].T)
            npos = self.npos[k]
            o = np.lexsort((self.tb[k], -s))
            is_pos = np.zeros(len(c), bool)
            is_pos[:npos] = True
            hit = is_pos[o]
            ranks = np.arange(1, len(c) + 1)
            prec = np.cumsum(hit) / ranks
            aps[k] = prec[hit].mean()
        return float(aps.mean()), aps

    def rmse(self, U):
        se, n = 0.0, 0
        for k in range(len(self.users)):
            p = U[k] @ np.ascontiguousarray(self.V[self.pos_items[k]].T)
            se += float(((np.clip(p, 1, 5) - self.pos_r[k]) ** 2).sum())
            n += len(p)
        return float(np.sqrt(se / n))

    def map_score_static(self, scores_by_item):
        """For the most-popular canonical snap: one global item score vector."""
        aps = np.empty(len(self.users))
        for k in range(len(self.users)):
            c = self.cand[k]
            s = scores_by_item[c]
            npos = self.npos[k]
            o = np.lexsort((self.tb[k], -s))
            is_pos = np.zeros(len(c), bool)
            is_pos[:npos] = True
            hit = is_pos[o]
            prec = np.cumsum(hit) / np.arange(1, len(c) + 1)
            aps[k] = prec[hit].mean()
        return float(aps.mean()), aps


def user_gram(data, V):
    """A_i = sum over user i's examples of v_j v_j^T  (square loss only)."""
    d, N = data.d, data.n_users
    A = np.zeros((N, d, d))
    for k1 in range(d):
        for k2 in range(k1, d):
            w = V[data.ei, k1] * V[data.ei, k2]
            val = np.bincount(data.eu, weights=w, minlength=N)
            A[:, k1, k2] = val
            A[:, k2, k1] = val
    return A


# ------------------------------------------------------------------------------- arms
def build_data(kind, pos_u, pos_i, pos_r, rated, n_users, n_items, d, seed):
    """Positives + sampled negatives -> the example container for a loss."""
    if kind == "square":
        return Data("square", n_users, n_items, d,
                    eu=pos_u, ei=pos_i, ey=pos_r.astype(np.float32))
    nu, ni = sample_negatives(pos_u, pos_i, n_users, n_items, rated,
                              NEG_PER_POS_TRAIN, seed)
    if kind == "logistic":
        eu = np.concatenate([pos_u, nu])
        ei = np.concatenate([pos_i, ni])
        ey = np.concatenate([np.ones(len(pos_u), np.float32),
                             -np.ones(len(nu), np.float32)])
        return Data("logistic", n_users, n_items, d, eu=eu, ei=ei, ey=ey)
    if kind == "pairwise":
        # pair positive j of user u with its own block of NEG_PER_POS negatives
        order = np.argsort(pos_u, kind="stable")
        pu, pi = pos_u[order], pos_i[order]
        ep = np.repeat(pi, NEG_PER_POS_TRAIN)
        eu = np.repeat(pu, NEG_PER_POS_TRAIN)
        onN = np.argsort(nu, kind="stable")
        en = ni[onN]
        assert (nu[onN] == eu).all()
        return Data("pairwise", n_users, n_items, d, eu=eu, ep=ep, en=en)
    raise ValueError(kind)


def init_V(n_items, d, seed=7):
    return np.random.default_rng(seed).normal(0, 0.05, size=(n_items, d))


def answers_by_user(au, ai, ar):
    out = {}
    for k in range(len(au)):
        out.setdefault(int(au[k]), {})[int(ai[k])] = int(ar[k])
    return out


def run_eip(kind, depth, split, d, lam_v, exact=False, random_q=False, seed=0,
            outer=None):
    outer = OUTER_ITERS if outer is None else outer
    tu, ti, tr = split.tr
    N, M = split.n_users, split.n_items
    rated = rated_sets_of(split.all_u, split.all_i, N)
    data = build_data(kind, tu, ti, tr, rated, N, M, d, SEED_TRAIN_NEG)
    V = init_V(M, d)
    rng = np.random.default_rng(1000 + seed)
    tree = None
    for it in range(outer + 1):
        t0 = time.time()
        tree, U = fit_tree(data, V, depth, (tu, ti, tr), N, LAMBDA_U,
                           exact=exact, random_q=random_q, rng=rng)
        t1 = time.time()
        if it < outer:
            z = data.z_of(U, V)
            V, z = update_V(data, U, V, z, lam_v, V_SWEEPS)
        log("    [%s%s%s depth=%d] iter %d: tree %.1fs V %.1fs nodes=%d"
            % (kind, "/exact" if exact else "", "/rand" if random_q else "",
               depth, it, t1 - t0, time.time() - t1, len(tree.q)))
    return tree, V


def eip_predict(tree, split, ev_users):
    abu = answers_by_user(*split.ans)
    for uu in ev_users:
        abu.setdefault(int(uu), {})
    U, dep = tree.traverse(abu, ev_users)
    return U, dep


def _feature_matrix(items, ans, users, thr=FIXQ_LIKE_THRESHOLD):
    """FixQ features: intercept + (higher-than-thr, lower-or-equal) per question."""
    F = 1 + 2 * len(items)
    rows = []
    for uu in users:
        a = ans.get(int(uu), {})
        f = [0]
        for qi, q in enumerate(items):
            r = a.get(int(q))
            if r is None:
                continue
            f.append(1 + 2 * qi + (0 if r > thr else 1))
        rows.append(np.array(f, np.int64))
    return rows, F


def fit_feature_mf(data, V, rows_by_user, F, users, lam_u, lam_v, outer, d,
                   n_users):
    """Feature-based MF: u_i = sum_f x_if a_f, fitted by coordinate descent on a
    per-dimension quadratic model of the loss (exact CD for square loss)."""
    A = np.zeros((F, d))
    U = np.zeros((n_users, d))
    ptr = np.zeros(len(users) + 1, np.int64)
    for k, rr in enumerate(rows_by_user):
        ptr[k + 1] = ptr[k] + len(rr)
    cols = (np.concatenate(rows_by_user) if len(rows_by_user)
            else np.zeros(0, np.int64))
    urow = np.repeat(np.arange(len(users)), np.diff(ptr))
    uid = np.asarray(users)
    feat_rows = [urow[cols == f] for f in range(F)]
    for it in range(outer + 1):
        for _ in range(U_SWEEPS):
            z = data.z_of(U, V)
            for k in range(d):
                c1, c2 = data.coefs(z)
                wk = data.w(V, k)
                g_i = np.bincount(data.eu, weights=c1 * wk, minlength=n_users)
                h_i = np.bincount(data.eu, weights=c2 * wk * wk, minlength=n_users)
                gi, hi = g_i[uid], h_i[uid]
                du = np.zeros(len(users))
                for _gs in range(8):                 # Gauss-Seidel over features
                    for f in range(F):
                        rr = feat_rows[f]
                        if len(rr) == 0:
                            continue
                        num = (gi[rr] + hi[rr] * du[rr]).sum() + 2 * lam_u * A[f, k]
                        den = hi[rr].sum() + 2 * lam_u
                        step = -num / max(den, 1e-9)
                        A[f, k] += step
                        du[rr] += step
                U[uid, k] += du
                z = data.z_of(U, V)
        if it < outer:
            V, _ = update_V(data, U, V, data.z_of(U, V), lam_v, V_SWEEPS)
    return A, V


def run_fixq(kind, k, split, d, lam_v, outer=None):
    outer = OUTER_ITERS if outer is None else outer
    tu, ti, tr = split.tr
    N, M = split.n_users, split.n_items
    rated = rated_sets_of(split.all_u, split.all_i, N)
    pop = np.bincount(ti, minlength=M)
    items = np.argsort(-pop)[:k]
    data = build_data(kind, tu, ti, tr, rated, N, M, d, SEED_TRAIN_NEG)
    V = init_V(M, d)
    iset = set(int(x) for x in items)
    train_ans = {}
    for a, b, c in zip(tu, ti, tr):
        if int(b) in iset:
            train_ans.setdefault(int(a), {})[int(b)] = int(c)
    users = np.sort(split.train_users)
    rows, F = _feature_matrix(items, train_ans, users)
    A, V = fit_feature_mf(data, V, rows, F, users, LAMBDA_U, lam_v, outer, d, N)
    return items, A, V, F


def fixq_predict(items, A, F, split, ev_users, d):
    abu = answers_by_user(*split.ans)
    rows, _ = _feature_matrix(items, abu, ev_users)
    U = np.zeros((len(ev_users), d))
    for k, rr in enumerate(rows):
        U[k] = A[rr].sum(0)
    return U


def run_warm(kind, split, d, lam_v, outer=None):
    """Train ratings + the test users' ANSWER half, one combined training set."""
    outer = OUTER_ITERS if outer is None else outer
    tu, ti, tr = split.tr
    au, ai, ar = split.ans
    pu = np.concatenate([tu, au])
    pi = np.concatenate([ti, ai])
    pr = np.concatenate([tr, ar])
    N, M = split.n_users, split.n_items
    rated = rated_sets_of(split.all_u, split.all_i, N)
    data = build_data(kind, pu, pi, pr, rated, N, M, d, SEED_TRAIN_NEG)
    V = init_V(M, d)
    U = np.zeros((N, d))
    gid = np.arange(N)
    for it in range(outer + 1):
        z = data.z_of(U, V)
        U, z = fit_delta_groups(data, V, z, gid, N, LAMBDA_U, U_SWEEPS,
                                delta=U.copy())
        if it < outer:
            z = data.z_of(U, V)
            V, z = update_V(data, U, V, z, lam_v, V_SWEEPS)
    return U, V


# ------------------------------------------------------------------------------ main
def cell_path(tag):
    os.makedirs(CELLS, exist_ok=True)
    return os.path.join(CELLS, tag + ".json")


def write_cell(tag, rec):
    with open(cell_path(tag), "w") as fh:
        json.dump(rec, fh, indent=2)
    log("  wrote", cell_path(tag))


def main():
    global log_fh
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="main",
                    choices=["sweep", "main", "controls", "rmse", "smoke"])
    ap.add_argument("--d", type=int, default=10)
    ap.add_argument("--lam-v", type=float, default=10.0)
    ap.add_argument("--depths", default="3,5,8,10")
    ap.add_argument("--outer", type=int, default=OUTER_ITERS)
    ap.add_argument("--arms", default="")
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    log_fh = open(os.path.join(OUT, "run_%s.log" % args.stage), "a")
    depths = [int(x) for x in args.depths.split(",")]
    arms = set(a for a in args.arms.split(",") if a)

    u, i, r = load_ratings()
    log("ML-1M: %d ratings, %d users, %d rated items (3952 movie ids in the file)"
        % (len(u), u.max() + 1, i.max() + 1))

    if args.stage == "sweep":
        base = Split(u, i, r)
        rng = np.random.default_rng(777)
        val = np.sort(rng.permutation(base.train_users)[
            :int(round(VAL_FRAC * len(base.train_users)))])
        sp = Split(u, i, r, holdout_users=val, drop_users=val)
        log("SWEEP cohort: " + json.dumps(sp.leak_check()))
        ev = Evaluator(sp, sp.n_items)
        res = {}
        for dd in (5, 10, 20):
            for lam_v in (1.0, 10.0, 100.0):
                t0 = time.time()
                tree, V = run_eip("logistic", 5, sp, dd, lam_v, outer=args.outer)
                U, _ = eip_predict(tree, sp, ev.users)
                ev.V = V
                m, _ = ev.map_score(U)
                res["d%d_lv%g" % (dd, lam_v)] = m
                log("SWEEP d=%d lam_v=%g: val MAP=%.4f (%.0fs)"
                    % (dd, lam_v, m, time.time() - t0))
                write_cell("sweep", res)
        return

    sp = Split(u, i, r)
    lk = sp.leak_check()
    log("LEAK CHECK: " + json.dumps(lk))
    ev = Evaluator(sp, sp.n_items)
    d, lam_v = args.d, args.lam_v

    if args.stage in ("main", "smoke", "controls"):
        pop = np.bincount(sp.tr[1], minlength=sp.n_items).astype(float)
        mp, _ = ev.map_score_static(pop)
        rnd, _ = ev.map_score_static(np.random.default_rng(0).random(sp.n_items))
        log("CANONICAL SNAP  most-popular MAP=%.4f   random MAP=%.4f" % (mp, rnd))
        write_cell("canonical", {"most_popular_MAP": mp, "random_MAP": rnd,
                                 "leak_check": lk})

    if args.stage == "smoke":
        return

    if args.stage == "main":
        for kind, tag in (("logistic", "eip_logistic"), ("pairwise", "eip_pairwise")):
            if arms and tag not in arms and ("fixq_" + kind) not in arms \
                    and ("warm_" + kind) not in arms:
                continue
            if not arms or tag in arms:
                for k in depths:
                    t0 = time.time()
                    tree, V = run_eip(kind, k, sp, d, lam_v, outer=args.outer)
                    U, dep = eip_predict(tree, sp, ev.users)
                    ev.V = V
                    m, aps = ev.map_score(U)
                    log("%s depth=%d: MAP=%.4f (%.0fs, mean questions used=%.2f)"
                        % (tag, k, m, time.time() - t0, dep.mean()))
                    write_cell("%s_d%d" % (tag, k),
                               dict(arm=tag, depth=k, MAP=m, d=d, lam_v=lam_v,
                                    seconds=time.time() - t0,
                                    mean_depth_reached=float(dep.mean()),
                                    n_test_users=len(ev.users), n_nodes=len(tree.q)))
            if not arms or ("fixq_" + kind) in arms:
                for k in depths:
                    t0 = time.time()
                    items, A, V, F = run_fixq(kind, k, sp, d, lam_v, outer=args.outer)
                    U = fixq_predict(items, A, F, sp, ev.users, d)
                    ev.V = V
                    m, _ = ev.map_score(U)
                    log("fixq_%s k=%d: MAP=%.4f (%.0fs)"
                        % (kind, k, m, time.time() - t0))
                    write_cell("fixq_%s_d%d" % (kind, k),
                               dict(arm="fixq_" + kind, depth=k, MAP=m, d=d,
                                    lam_v=lam_v))
            if not arms or ("warm_" + kind) in arms:
                U, V = run_warm(kind, sp, d, lam_v, outer=args.outer)
                ev.V = V
                m, _ = ev.map_score(U[ev.users])
                log("warm_%s: MAP=%.4f" % (kind, m))
                write_cell("warm_%s" % kind,
                           dict(arm="warm_" + kind, MAP=m, d=d, lam_v=lam_v))

    if args.stage == "controls":
        for k in depths:
            t0 = time.time()
            tree, V = run_eip("logistic", k, sp, d, lam_v, random_q=True,
                              outer=args.outer)
            U, _ = eip_predict(tree, sp, ev.users)
            ev.V = V
            m, _ = ev.map_score(U)
            log("randq depth=%d: MAP=%.4f (%.0fs)" % (k, m, time.time() - t0))
            write_cell("randq_d%d" % k,
                       dict(arm="randq", depth=k, MAP=m, d=d, lam_v=lam_v))

    if args.stage == "rmse":
        for k in depths:
            for exact, tag in ((False, "eip_square"), (True, "fmf")):
                t0 = time.time()
                tree, V = run_eip("square", k, sp, d, lam_v, exact=exact,
                                  outer=args.outer)
                U, _ = eip_predict(tree, sp, ev.users)
                ev.V = V
                rm = ev.rmse(U)
                log("%s depth=%d: RMSE=%.4f (%.0fs)"
                    % (tag, k, rm, time.time() - t0))
                write_cell("%s_d%d" % (tag, k),
                           dict(arm=tag, depth=k, RMSE=rm, d=d, lam_v=lam_v))
            items, A, V, F = run_fixq("square", k, sp, d, lam_v, outer=args.outer)
            U = fixq_predict(items, A, F, sp, ev.users, d)
            ev.V = V
            rm = ev.rmse(U)
            log("fixq_square k=%d: RMSE=%.4f" % (k, rm))
            write_cell("fixq_square_d%d" % k,
                       dict(arm="fixq_square", depth=k, RMSE=rm, d=d, lam_v=lam_v))
        U, V = run_warm("square", sp, d, lam_v, outer=args.outer)
        ev.V = V
        rm = ev.rmse(U[ev.users])
        log("warm_square: RMSE=%.4f" % rm)
        write_cell("warm_square", dict(arm="warm_square", RMSE=rm, d=d, lam_v=lam_v))


if __name__ == "__main__":
    main()
