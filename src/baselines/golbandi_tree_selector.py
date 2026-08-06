r"""golbandi_tree_selector.py -- THE SPLIT SEARCH. Golbandi, Koren & Lempel, WSDM 2011,
"Adaptive Bootstrapping of Recommender Systems Using Decision Trees", DOI 10.1145/1935826.1935910.

WHAT WAS MISSING AND WHY THIS FILE EXISTS. The WSDM'11 method has two halves. The NODE MODEL (each
node holds the shrunk mean profile of the training users routed to it) has been in the harness for a
while as `golbandi_leaf.py` -- it CONSUMES answers. The other half, the half that makes the method
ADAPTIVE, is the tree GROWTH: at every node, search the candidate items for the one whose ternary
lovers/haters/unknowns partition minimises the within-group squared error, and make that item the
question asked at that node. Nothing in the repo chose the next question; every arm in the interview
table was a static global order. That is the hole this file fills.

THE CRITERION, VERBATIM FROM THE PAPER (Eq. 1). The authors state explicitly that they "deviate from
common error metrics typical to classification tree construction (like entropy or Gini impurity), but
rather rely on the squared loss metric". With S_t the users at node t, mu(t)_i the node's mean for
item i, and

    e2(t)_i = SUM_{u in S_t ^ R(i)} ( r_ui - mu(t)_i )^2 ,        e2(t) = SUM_i e2(t)_i

the splitter is

    Err_t(i) = e2(t_L) + e2(t_H) + e2(t_U) ,      splitter(t) = argmin_i Err_t(i)          (Eq. 1)

t_L = users who LOVE i (r >= like_min; the paper's Netflix cut is 4-5 love / 1-3 hate against a 3.6
dataset mean), t_H = users who HATE i, t_U = users who never rated i. Note what the criterion is NOT:
it is not the error on the split item, it is the total within-group error over the WHOLE catalogue.
That is why one question can pay off -- it partitions users, not items.

SUFFICIENT STATISTICS AND THE SUBTRACTION TRICK (Eq. 3/4). e2 of a group is computed from three
per-item accumulators, never from a pass over residuals:

    e2(group) = SUM_j [ sum2(group)_j - sum(group)_j^2 / n(group)_j ]        (Eq. 4, n_j > 0)

and the UNKNOWN branch is never gathered directly -- it is the node total minus lovers minus haters
(Eq. 3). So one pass over the lovers and haters of each candidate gives all three children. Here that
is expressed as six sparse products per candidate CHUNK (indicator^T @ {ratings, ratings^2, mask}),
which is the vectorised form of the paper's O(SUM_u |R(u)|^2) per-level bound. The chunking is a
MEMORY device only -- every candidate item is still scored at every node, exhaustively, as the paper
prescribes. Nothing is sampled, capped or truncated (HARD RULE 1).

BIAS. The paper removes only the USER bias when growing (item bias washes out, because the squared
error always compares values on the same item), with

    b_u = ( SUM_{i in R(u)} r_ui + beta * mu ) / ( |R(u)| + beta ) ,   beta = 7

and observes that "while biases are important while constructing the tree in the training phase, they
become immaterial during the online phase".

STOPPING. Three rules: maximum depth; no strict error reduction (Err_t(best) >= e2(t)); and node
rating mass below xi (the paper's Netflix value is xi = 200,000 out of >100M ratings, i.e. ~0.2% of
the mass -- on any smaller corpus it must be rescaled or it stops the tree at the root, and the
rescaling is reported, not hidden).

LEAF PROFILE (optional here; `golbandi_leaf.py` is the harness's consumer). Per-item HIERARCHICAL
shrinkage toward the PARENT node, not a raw mean and not pruning -- the paper rejects pruning
explicitly because "pruning a node as a whole would be unappealing in our sparse data scenario":

    rhat_t(i) = ( sum(t)_i + lam1 * rhat_parent(i) ) / ( n(t)_i + lam1 ) ,   lam1 = 200

Profiles are OPT-IN (`store_profiles`) because a dense (n_nodes x n_items) table is 1.4 GB at
ML-25M scale and the SELECTOR does not need it: the selector only picks questions, and
`golbandi_leaf` consumes the answers.

DEPTH IS OUR QUESTION BUDGET. A depth-d tree asks exactly d questions, so `MAX_DEPTH` in
`golbandi_leaf` (8, because 3^k answer patterns fragment the training cohort) is the same constraint
seen from the consumer side. When a user reaches a leaf before the budget is spent -- which happens
constantly at small node counts -- the selector must still return a question. It falls back to the
next unasked item of a fixed global order and COUNTS how often that happened; that count is reported
in `stats()`, because a "Golbandi" arm that is 60% popularity fallback is not Golbandi.

INTERFACE (the `Selector` contract from `adaptive_harness.py`):

    sel = GolbandiTreeSelector(g_train, n_items, depth=k, log=logln)
    sel.reset(ctx)                                  # ctx: n_users, n_items, q_max, D, glob, bank, log
    sel.select(turn, users, asked, answered)        # -> np.int64[len(users)], never a repeat
    sel.stats()                                     # tree shape + fallback + occupancy diagnostics

The tree is grown ONCE, on the TRAIN users only, at the first `reset` (or eagerly via `fit()`), and
cached across budgets: growth never sees a test user, and re-growing per budget would be both slower
and a place for two budgets to disagree about the same root question.
"""
import json
import os
import time
import numpy as np
from scipy import sparse

# --------------------------------------------------------------------------- published constants
BETA_USER_BIAS = 7.0        # paper: b_u shrinkage
LAMBDA1_LEAF = 200.0        # paper: leaf shrinkage toward parent (Netflix scale)
XI_NETFLIX = 200_000.0      # paper: min node rating mass, out of >100M ratings
XI_FRACTION = XI_NETFLIX / 100_000_000.0    # = 0.002 of the corpus; the scale-free form of the rule
LIKE_MIN_LEVEL = 7          # harness half-star level >= 7  <=>  r >= 4.0  <=>  the paper's 4-5 cut


# ============================================================================ tree growth
class GolbandiTree:
    """The WSDM'11 ternary tree, grown on a graded (ratings) CSR of TRAINING users.

    Parameters
    ----------
    R            : csr_matrix (n_train x n_items) of RATINGS (any scale; explicit zeros are "unrated")
    like_min     : float. r >= like_min is a LOVER, 0 < r < like_min is a HATER, absent is UNKNOWN.
    depth        : int, maximum tree depth == the maximum number of questions the tree can ask.
    xi           : float or None. Minimum node rating mass to consider splitting. None -> the paper's
                   scale-free 0.2% of the corpus's total rating count (see XI_FRACTION).
    min_child_users : int. 0 = paper-faithful (the paper relies on shrinkage, not on a size floor).
                   >0 rejects splits producing a tiny child; used only as a labelled ablation.
    bias_correct : bool. Remove the paper's user bias before computing the split criterion.
    candidates   : None (exhaustive over all items, the paper's own rule) or an explicit id array.
    forced_order : None, or an item-id sequence. If given, the node at depth d does NOT search -- it
                   asks forced_order[d]. This is the SYMMETRIC STATIC COMPARATOR: identical tree
                   machinery, identical leaf model, identical question count, only the SELECTION
                   differs. It is the only way to attribute a difference to adaptivity rather than to
                   the node model.
    store_profiles : bool. Compute per-node shrunk profiles (needs a dense n_nodes x n_items table).
    profile_kind : "mean" (the paper's rating mean -> RMSE) or "like_rate" (the fraction of the
                   node's users who liked each item -> the ranking adaptation documented in
                   `golbandi_leaf.py`; a mean rating is a known disaster as a top-N score).
    """

    def __init__(self, R, like_min=4.0, depth=6, xi=None, min_child_users=0, bias_correct=True,
                 candidates=None, forced_order=None, store_profiles=False, profile_kind="mean",
                 lam1=LAMBDA1_LEAF, beta=BETA_USER_BIAS, chunk=256, store_rows=False, log=print):
        self.R = R.tocsr().astype(np.float64)
        self.R.eliminate_zeros()
        self.n_users, self.n_items = self.R.shape
        self.like_min = float(like_min)
        self.depth = int(depth)
        self.min_child_users = int(min_child_users)
        self.bias_correct = bool(bias_correct)
        self.forced_order = None if forced_order is None else list(forced_order)
        self.store_profiles = bool(store_profiles)
        self.profile_kind = profile_kind
        self.lam1 = float(lam1)
        self.beta = float(beta)
        self.chunk = int(chunk)
        self.log = log
        n_ratings = float(self.R.nnz)
        self.xi = float(xi) if xi is not None else XI_FRACTION * n_ratings
        self.candidates = (np.arange(self.n_items, dtype=np.int64) if candidates is None
                           else np.asarray(candidates, dtype=np.int64))

        # ---- matrices the criterion needs. M = rated mask, X = (bias-corrected) values, X2 = X^2.
        self.M = self.R.copy(); self.M.data = np.ones_like(self.M.data)
        if self.bias_correct:
            cnt = np.diff(self.R.indptr).astype(np.float64)
            tot = np.asarray(self.R.sum(axis=1)).ravel()
            gmu = n_ratings and float(self.R.data.sum() / n_ratings) or 0.0
            b_u = (tot + self.beta * gmu) / (cnt + self.beta)
            X = self.R.copy()
            X.data = X.data - np.repeat(b_u, np.diff(self.R.indptr))
            self.b_u = b_u
        else:
            X = self.R.copy()
            self.b_u = np.zeros(self.n_users)
        self.X = X
        self.X2 = X.copy(); self.X2.data = X.data ** 2
        # LOVER / HATER indicators, on the RAW ratings (the branch rule is on r, not on the residual)
        Lo = self.R.copy(); Lo.data = (Lo.data >= self.like_min).astype(np.float64); Lo.eliminate_zeros()
        Ha = self.R.copy(); Ha.data = (Ha.data < self.like_min).astype(np.float64); Ha.eliminate_zeros()
        self.Lo, self.Ha = Lo.tocsr(), Ha.tocsr()
        # like-rate global prior (for profile_kind="like_rate")
        self.global_like = np.asarray(self.Lo.sum(axis=0)).ravel() / max(self.n_users, 1)
        s = np.asarray(self.R.sum(axis=0)).ravel()
        c = np.asarray(self.M.sum(axis=0)).ravel()
        gmu_all = float(self.R.data.mean()) if n_ratings else 0.0
        self.global_mean = np.where(c > 0, s / np.maximum(c, 1), gmu_all)

        self.nodes = []              # each: dict(item, children{L,H,U}, n_users, n_ratings, depth)
        self.profiles = None
        self._rows_of = [] if store_rows else None
        self._grown = False

    # ------------------------------------------------------------------ criterion
    @staticmethod
    def _e2(s, s2, n):
        """SUM_j [ sum2_j - sum_j^2 / n_j ] over items with n_j > 0. Eq. (4). Works row-wise."""
        with np.errstate(divide="ignore", invalid="ignore"):
            v = s2 - np.where(n > 0, s * s / np.maximum(n, 1.0), 0.0)
        return v.sum(axis=-1)

    def _best_split(self, rows, banned):
        """Exhaustive argmin_i Err_t(i) over `self.candidates` minus `banned` (path ancestors).

        Returns (item, Err, e2_node) or (None, inf, e2_node)."""
        Xr, X2r, Mr = self.X[rows], self.X2[rows], self.M[rows]
        s_tot = np.asarray(Xr.sum(axis=0)).ravel()
        s2_tot = np.asarray(X2r.sum(axis=0)).ravel()
        n_tot = np.asarray(Mr.sum(axis=0)).ravel()
        e2_node = float(self._e2(s_tot, s2_tot, n_tot))

        cand = self.candidates[~np.isin(self.candidates, list(banned))] if banned else self.candidates
        if len(cand) == 0:
            return None, np.inf, e2_node
        # Transposed ONCE per node: (n_items x n_S) so a candidate chunk is a cheap ROW slice.
        LrT, HrT = self.Lo[rows].T.tocsr(), self.Ha[rows].T.tocsr()
        best_i, best_e = None, np.inf
        best_nL = best_nH = 0
        for a in range(0, len(cand), self.chunk):
            cc = cand[a:a + self.chunk]
            Lc = LrT[cc]                                   # (|cc| x n_S)
            Hc = HrT[cc]
            sL = np.asarray((Lc @ Xr).todense()); s2L = np.asarray((Lc @ X2r).todense())
            nL = np.asarray((Lc @ Mr).todense())
            sH = np.asarray((Hc @ Xr).todense()); s2H = np.asarray((Hc @ X2r).todense())
            nH = np.asarray((Hc @ Mr).todense())
            sU = s_tot[None, :] - sL - sH                  # Eq. (3): the unknown branch by SUBTRACTION
            s2U = s2_tot[None, :] - s2L - s2H
            nU = n_tot[None, :] - nL - nH
            err = self._e2(sL, s2L, nL) + self._e2(sH, s2H, nH) + self._e2(sU, s2U, nU)
            if self.min_child_users > 0:
                cL = np.asarray(Lc.sum(axis=1)).ravel(); cH = np.asarray(Hc.sum(axis=1)).ravel()
                err = np.where((cL >= self.min_child_users) & (cH >= self.min_child_users),
                               err, np.inf)
            k = int(np.argmin(err))
            if err[k] < best_e:
                best_e, best_i = float(err[k]), int(cc[k])
                best_nL = int(np.asarray(Lc.sum(axis=1)).ravel()[k])
                best_nH = int(np.asarray(Hc.sum(axis=1)).ravel()[k])
        self._last_child_sizes = (best_nL, best_nH, len(rows) - best_nL - best_nH)
        return best_i, best_e, e2_node

    # ------------------------------------------------------------------ growth
    def fit(self):
        if self._grown:
            return self
        assert not getattr(self, "_released", False), (
            "this tree was loaded from cache / released; it has no training matrices and cannot grow")
        t0 = time.time()
        rows_root = np.arange(self.n_users, dtype=np.int64)
        prof_root = (self.global_like.copy() if self.profile_kind == "like_rate"
                     else self.global_mean.copy())
        self._profiles = [] if self.store_profiles else None
        self._build(rows_root, 0, banned=set(), parent_profile=prof_root)
        self.profiles = (np.asarray(self._profiles, dtype=np.float32)
                         if self.store_profiles else None)
        self._grown = True
        self.log(f"[golbandi_tree] grown: depth<={self.depth} nodes={len(self.nodes)} "
                 f"internal={sum(1 for nd in self.nodes if nd['item'] is not None)} "
                 f"xi={self.xi:.0f} bias={self.bias_correct} forced={self.forced_order is not None} "
                 f"({time.time() - t0:.1f}s)")
        return self

    def recompute_profiles(self, lam1=None, profile_kind=None):
        """Re-derive every node's profile on the ALREADY-GROWN structure. Growth does not depend on
        lam1 or on the profile kind (the criterion is Eq. 1 on the ratings), so a shrinkage sweep or
        a mean-vs-like-rate comparison must reuse ONE tree -- re-growing would risk two arms that
        differ in their questions being reported as differing only in their leaf model."""
        assert self._grown and self._rows_of is not None, "grow with store_rows=True first"
        if lam1 is not None:
            self.lam1 = float(lam1)
        if profile_kind is not None:
            self.profile_kind = profile_kind
        root = (self.global_like.copy() if self.profile_kind == "like_rate"
                else self.global_mean.copy())
        P = np.empty((len(self.nodes), self.n_items), dtype=np.float32)

        def rec(nid, parent_profile):
            P[nid] = self._profile(self._rows_of[nid], parent_profile)
            nd = self.nodes[nid]
            if nd["item"] is not None:
                for key in ("L", "H", "U"):
                    rec(nd[key], P[nid])
        rec(0, root)
        self.profiles = P
        self.store_profiles = True
        return self

    def _profile(self, rows, parent_profile):
        if self.profile_kind == "like_rate":
            s = np.asarray(self.Lo[rows].sum(axis=0)).ravel()
            return ((s + self.lam1 * parent_profile) / (len(rows) + self.lam1)).astype(np.float32)
        s = np.asarray(self.R[rows].sum(axis=0)).ravel()
        n = np.asarray(self.M[rows].sum(axis=0)).ravel()
        return ((s + self.lam1 * parent_profile) / (n + self.lam1)).astype(np.float32)

    def _build(self, rows, depth, banned, parent_profile):
        prof = self._profile(rows, parent_profile) if (self.store_profiles or depth == 0) else None
        nid = len(self.nodes)
        node = {"item": None, "L": None, "H": None, "U": None, "depth": depth,
                "n_users": int(len(rows)), "n_ratings": int(self.M[rows].nnz)}
        self.nodes.append(node)
        if self._rows_of is not None:
            self._rows_of.append(rows)
        if self.store_profiles:
            self._profiles.append(prof if prof is not None else parent_profile)

        if depth >= self.depth or node["n_ratings"] < self.xi:
            return nid
        if self.forced_order is not None:
            if depth >= len(self.forced_order):
                return nid
            item = int(self.forced_order[depth])
            if item in banned:
                return nid
            col = self.R[rows][:, item].toarray().ravel()
            mL = col >= self.like_min
            mH = (col > 0) & ~mL
        else:
            item, err, e2_node = self._best_split(rows, banned)
            if item is None or not (err < e2_node):      # paper: stop on NO STRICT error reduction
                return nid
            col = self.R[rows][:, item].toarray().ravel()
            mL = col >= self.like_min
            mH = (col > 0) & ~mL
        mU = ~(mL | mH)
        node["item"] = int(item)
        node["child_sizes"] = [int(mL.sum()), int(mH.sum()), int(mU.sum())]
        nb = banned | {int(item)}
        pp = prof if prof is not None else parent_profile
        node["L"] = self._build(rows[mL], depth + 1, nb, pp)
        node["H"] = self._build(rows[mH], depth + 1, nb, pp)
        node["U"] = self._build(rows[mU], depth + 1, nb, pp)
        return nid

    # ------------------------------------------------------------------ traversal / diagnostics
    def route(self, answers, max_steps=None):
        """answers: dict item -> rating (absent == unknown). Returns the list of node ids visited,
        root first. len(path)-1 == the number of questions actually asked by the tree."""
        max_steps = self.depth if max_steps is None else int(max_steps)
        nid, path = 0, [0]
        while len(path) - 1 < max_steps:
            nd = self.nodes[nid]
            if nd["item"] is None:
                break
            r = answers.get(nd["item"])
            key = "U" if r is None else ("L" if r >= self.like_min else "H")
            nid = nd[key]
            path.append(nid)
        return path

    def occupancy(self):
        """Per-depth leaf/node occupancy -- the mechanism behind any depth-vs-budget degradation."""
        out = {}
        for d in range(self.depth + 1):
            ns = [nd["n_users"] for nd in self.nodes if nd["depth"] == d]
            rs = [nd["n_ratings"] for nd in self.nodes if nd["depth"] == d]
            if not ns:
                continue
            ns = np.asarray(ns)
            out[d] = {"n_nodes": int(len(ns)), "users_mean": float(ns.mean()),
                      "users_median": float(np.median(ns)), "users_min": int(ns.min()),
                      "users_max": int(ns.max()),
                      "n_lt_10": int((ns < 10).sum()), "n_lt_25": int((ns < 25).sum()),
                      "ratings_median": float(np.median(rs)),
                      "internal": int(sum(1 for nd in self.nodes
                                          if nd["depth"] == d and nd["item"] is not None))}
        return out

    def to_dict(self):
        return {"depth": self.depth, "xi": self.xi, "like_min": self.like_min,
                "bias_correct": self.bias_correct, "lam1": self.lam1, "beta": self.beta,
                "min_child_users": self.min_child_users,
                "forced": self.forced_order is not None, "n_items": self.n_items,
                "n_train": self.n_users, "nodes": self.nodes}

    def save(self, path):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        json.dump(self.to_dict(), open(path, "w"))
        return path

    # ---------------------------------------------------------------- persistence / memory (RULE 11)
    def release(self):
        """Drop every training matrix. Growth is over; ROUTING needs only `nodes`, `depth` and
        `like_min`, so at ML-25M scale this returns ~1.5 GB (six float64 CSRs over 21.7M nnz) that the
        selector would otherwise hold for the whole scoring run. Calling anything that re-reads the
        data (fit, recompute_profiles, _profile) after this is an error, and asserts as one."""
        for attr in ("R", "M", "X", "X2", "Lo", "Ha"):
            setattr(self, attr, None)
        self._rows_of = None
        self._profiles = None
        self._released = True
        return self


def load_tree(path, log=print):
    """Rebuild a ROUTING-ONLY tree from `GolbandiTree.save()` output.

    A grown tree at ML-25M scale costs hours; the harness must be able to crash, be relaunched and
    reuse it (HARD RULE 11b). What comes back has `nodes`, `depth`, `like_min` and `occupancy()` --
    everything `GolbandiTreeSelector` touches at select time -- and no matrices, so it cannot silently
    be re-fitted against a different contract: `fit()` on a loaded tree raises."""
    blob = json.load(open(path))
    t = GolbandiTree.__new__(GolbandiTree)
    t.nodes = blob["nodes"]
    t.depth = int(blob["depth"])
    t.like_min = float(blob["like_min"])
    t.xi = float(blob["xi"])
    t.lam1 = float(blob.get("lam1", LAMBDA1_LEAF))
    t.beta = float(blob.get("beta", BETA_USER_BIAS))
    t.bias_correct = bool(blob.get("bias_correct", True))
    t.min_child_users = int(blob.get("min_child_users", 0))
    t.n_items = int(blob["n_items"])
    t.n_users = int(blob["n_train"])
    t.forced_order = None
    t.profiles = None
    t._rows_of = None
    t._grown = True
    t._released = True
    for attr in ("R", "M", "X", "X2", "Lo", "Ha"):
        setattr(t, attr, None)
    t.log = log
    log(f"[golbandi_tree] LOADED cached tree {os.path.basename(path)}: depth<={t.depth} "
        f"nodes={len(t.nodes)} internal={sum(1 for nd in t.nodes if nd['item'] is not None)} "
        f"xi={t.xi:.0f} n_train={t.n_users} n_items={t.n_items}")
    return t


def tree_contract(g_train, n_items, depth, like_min, xi, min_child_users, bias_correct):
    """The signature a cached tree must match before it may be reused. A cache that was grown under a
    different contract is not the same tree, and silently reusing it is exactly the failure HARD RULE
    11(b) names."""
    return {"n_train": int(g_train.shape[0]), "n_items": int(n_items), "nnz": int(g_train.nnz),
            "depth": int(depth), "like_min": float(like_min),
            "xi": None if xi is None else float(xi),
            "min_child_users": int(min_child_users), "bias_correct": bool(bias_correct)}


# ============================================================================ the Selector
try:
    from adaptive_harness import Selector as _Selector
except Exception:                                        # importable standalone (e.g. the snap script)
    class _Selector:                                     # pragma: no cover
        name = "selector"

        def reset(self, ctx):
            self.ctx = ctx

        def select(self, turn, users, asked, answered):
            raise NotImplementedError

        def stats(self):
            return {}


class GolbandiTreeSelector(_Selector):
    """WSDM'11 tree growth as a question SELECTOR for `adaptive_harness.run_interview`.

    Grown once on the TRAIN users; never sees a test user. At turn t, user u is routed by the answers
    they have actually given so far and is asked that node's splitter. A user who reaches a leaf
    before the budget runs out falls back to the next unasked item of `fallback_order` (popularity by
    default, taken from `ctx["glob"]` when the harness supplies it), and the fallback rate is
    reported by `stats()` -- an arm that is mostly fallback is not this method.
    """
    name = "golbandi_tree"

    def __init__(self, g_train=None, n_items=None, depth=None, like_level=LIKE_MIN_LEVEL,
                 like_min=None, xi=None, min_child_users=0, bias_correct=True, chunk=256,
                 candidates=None, fallback_order=None, tree=None, cache_path=None, log=print):
        self.cache_path = cache_path
        self.g_train, self.n_items = g_train, n_items
        self.depth = depth
        self.like_level = int(like_level)     # harness half-star level cut (>=7 == r>=4.0)
        self.like_min = like_min              # rating-scale cut used to GROW (defaults to like_level)
        self.xi, self.min_child_users = xi, min_child_users
        self.bias_correct, self.chunk, self.candidates = bias_correct, chunk, candidates
        self.fallback_order = fallback_order
        self.tree = tree
        self.log = log
        self._fallbacks = 0
        self._asks = 0
        self._leaf_depths = []

    def fit(self, g_train=None, n_items=None, depth=None, cache_path=None):
        """Grow the tree, or REUSE a cached one whose contract matches exactly.

        `cache_path` is the HARD RULE 11(b) resumable sidecar: growth at ML-25M scale is the one
        genuinely expensive step here, and it is deterministic, so it is written once and reloaded.
        The contract (`tree_contract`) is stored alongside the nodes and a mismatch REFUSES to load
        rather than quietly reusing a tree grown on other data."""
        g_train = self.g_train if g_train is None else g_train
        n_items = self.n_items if n_items is None else n_items
        depth = self.depth if depth is None else depth
        lm = self.like_min if self.like_min is not None else float(self.like_level)
        want = tree_contract(g_train, n_items, depth, lm, self.xi, self.min_child_users,
                             self.bias_correct)
        if cache_path and os.path.exists(cache_path):
            got = json.load(open(cache_path)).get("_contract")
            if got == want:
                self.tree = load_tree(cache_path, log=self.log)
                return self
            self.log(f"[golbandi_tree] cache {cache_path} REFUSED: contract mismatch\n"
                     f"    cached={got}\n    wanted={want}")
        self.tree = GolbandiTree(g_train, like_min=lm, depth=int(depth), xi=self.xi,
                                 min_child_users=self.min_child_users,
                                 bias_correct=self.bias_correct, candidates=self.candidates,
                                 store_profiles=False, chunk=self.chunk, log=self.log).fit()
        if cache_path:
            blob = self.tree.to_dict()
            blob["_contract"] = want
            blob["occupancy"] = self.tree.occupancy()
            os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
            json.dump(blob, open(cache_path, "w"))
            self.log(f"[golbandi_tree] tree CHECKPOINTED -> {cache_path}")
        self.tree.release()          # routing needs only `nodes`; free the six training CSRs
        return self

    def reset(self, ctx):
        self.ctx = ctx
        if self.g_train is None:
            self.g_train = ctx["D"]["g_train"]
        if self.n_items is None:
            self.n_items = ctx["n_items"]
        q_max = int(ctx["q_max"])
        if self.fallback_order is None:
            glob = ctx.get("glob") or {}
            self.fallback_order = np.asarray(glob.get("popularity", np.arange(self.n_items)))
        if self.tree is None or self.tree.depth < q_max:
            # ONE tree, grown to the LARGEST budget seen; a depth-d prefix of it is exactly the
            # depth-d tree, because growth is top-down and depth-independent.
            self.fit(depth=max(q_max, int(self.depth or 0)), cache_path=self.cache_path)
        self._fallbacks = 0
        self._asks = 0
        self._leaf_depths = []

    def _node_of(self, asked_u, answered_u):
        """Route by the answers actually given. An asked-but-unrated item is the UNKNOWN branch,
        which is exactly the paper's third branch -- the non-answer is evidence."""
        lv = {int(i): int(l) for (i, l) in answered_u}
        nid = 0
        for _ in range(self.tree.depth + 1):
            nd = self.tree.nodes[nid]
            it = nd["item"]
            if it is None or it not in set(asked_u):
                return nid, nd
            l = lv.get(it)
            key = "U" if l is None else ("L" if l >= self.like_level else "H")
            nid = nd[key]
        return nid, self.tree.nodes[nid]

    def select(self, turn, users, asked, answered):
        out = np.empty(len(users), dtype=np.int64)
        for k, u in enumerate(users.tolist() if hasattr(users, "tolist") else users):
            a = asked[u]
            nid, nd = self._node_of(a, answered[u])
            it = nd["item"]
            self._asks += 1
            if it is None or it in a:
                self._fallbacks += 1
                self._leaf_depths.append(len(a))
                seen = set(a)
                it = next((int(j) for j in self.fallback_order if int(j) not in seen), None)
                if it is None:
                    it = next(j for j in range(self.n_items) if j not in seen)
            out[k] = int(it)
        return out

    def stats(self):
        s = {"tree_nodes": len(self.tree.nodes) if self.tree else 0,
             "tree_depth": self.tree.depth if self.tree else 0,
             "fallback_asks": int(self._fallbacks), "total_asks": int(self._asks),
             "fallback_rate": round(self._fallbacks / max(self._asks, 1), 4)}
        if self.tree:
            s["occupancy"] = self.tree.occupancy()
        if self._leaf_depths:
            s["leaf_reached_depth_mean"] = round(float(np.mean(self._leaf_depths)), 3)
        return s


class StaticOrderTreeSelector(_Selector):
    """THE SYMMETRIC COMPARATOR. Same ternary routing, same node model downstream, same number of
    questions -- but the question at depth d is fixed in advance by a global order, so nothing is
    conditioned on the answers. Any difference against `GolbandiTreeSelector` is attributable to the
    SPLIT SEARCH and to nothing else. (It is also what `interview_strategies.ask()` already does; this
    class exists so the snap script can run both through one code path.)"""
    name = "static_order_tree"

    def __init__(self, order, log=print):
        self.order = np.asarray(order, dtype=np.int64)
        self.log = log

    def reset(self, ctx):
        self.ctx = ctx

    def select(self, turn, users, asked, answered):
        return np.full(len(users), int(self.order[turn]), dtype=np.int64)
