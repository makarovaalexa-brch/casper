r"""igcn_selector.py -- IGCN (Information Gain through Clustered Neighbours).
Rashid, Karypis & Riedl, "Learning Preferences of New Users in Recommender Systems: An Information
Theoretic Approach", SIGKDD Explorations 10(2) 2008, pp. 90-100, DOI 10.1145/1540276.1540302.

WHY THIS FILE EXISTS. `interview_strategies.CITES` already credits this paper for the `entropy0` and
`helf` arms, and that credit is CORRECT: the paper's own words are "All of the heuristics except the
Igcn can be computed prior to the simulation", so a static precomputed order IS the faithful form of
entropy0/HELF. But the paper's own headline method, IGCN, is the one member of the family that is
ADAPTIVE, and it had never been implemented here -- a repo-wide grep found prose references only. The
2025 attribute-tree literature still benchmarks against it, so its absence is visible to a reviewer.

============================ THE METHOD, FROM THE PAPER [V] ============================
Source of record: the repo's own full-text pass, `external_literature/findings/
rating_elicitation_family_deep.md` section 2.3. The publisher PDF is CLOSED everywhere reachable
(Semantic Scholar `openAccessPdf: {"url": "", "status": "CLOSED"}` on DOI 10.1145/1540276.1540302;
kdd.org mirrors 404, ACM 403, karypis.github.io 404) -- the same situation SNAP-5 recorded for
Golbandi, and it is why the extraction, not a re-fetch, is cited.

1. INFORMATION GAIN OVER USER-CLUSTER LABELS, not over ratings:

       IG(a) = H(C) - SUM_r ( |C_a^r| / |C| ) H(C_a^r)

   C = the cluster label of each reference user; C_a^r = the reference users whose rating of item a
   falls in category r. Asking about `a` is worth exactly as much as it tells you WHICH KIND OF USER
   you are talking to. The missing-value-weighted variant the paper also gives is

       IG(a;W) = H(C) - SUM_r [ w_r |C_a^r| H(C_a^r) ] / E(C;W)

   and it is the one implemented here, because our protocol has a genuine "did not rate" outcome. The
   weights are the paper's OWN Guiasu weights from its Entropy0 definition in the same section --
   w = 1/2 on the not-rated category, 1 on every real rating category -- so nothing is invented: the
   number that decides how much an unanswerable question is punished comes from the paper.

2. CLUSTERS: bisecting k-means over the training users, c = 300.

3. ADAPTIVITY -- THE THING THAT MAKES IT IGCN AND NOT A SEVENTH STATIC ARM. Verbatim: "One issue with
   information theoretic measures such as entropy and its variants discussed so far is that they are
   not adaptive to a user's rating history." Algorithm 3.1 is two-phase:
     - NON-PERSONALIZED step: order by IG computed over ALL training users, until the user has rated
       at least `i` items;
     - PERSONALIZED step: find the user's best `l` neighbours and RE-COMPUTE IG using only those
       `l` users' ratings, repeating as the neighbour set moves.
   Published parameters (c, n, i, l) = (300 clusters, 15 items/screen, switch after 5 ratings,
   150 neighbours).

4. CONTRACT: MovieLens ~11,000 users / 9,000 items / ~3M ratings, budgets 15/30/45/60/75 (pages of
   15), metrics #seen / MAE / Expected Utility. Plus a 20-day online study, 381 valid subjects.

============================ THE TWO PLACES WE DEVIATE, BOTH LABELLED ============================
(a) RE-COMPUTATION PERIOD. The paper re-ranks per PAGE of n=15 items. Our budgets are k <= 16, i.e.
    at most ONE page, so the paper-literal period would fire at most once and the arm would be a
    static order at every budget we run -- which is precisely the failure mode this implementation
    exists to avoid. `recompute_every` therefore defaults to 1 (re-rank after every turn), the
    strictly more adaptive reading of the same algorithm, and `--igcn_page 15` recovers the literal
    setting for anyone who wants it. The default is reported in `stats()`, never assumed.
(b) SWITCH THRESHOLD. `switch_i` defaults to the paper's 5. At k <= 4 no user can have 5 answers, so
    the arm is EXACTLY the non-personalized global-IG order there, by construction and not by
    accident -- read a k<=4 IGCN cell as "global IG", and check `personalised_rate` in the cell. The
    `igcn_i1` spec (switch_i=1) is the labelled deviation that exercises the adaptive machinery at
    every budget; it is a DIFFERENT ROW, never a repair of the first.

Neighbour similarity is cosine over the graded rating vectors. The paper says "best l neighbours" of
its user-kNN recommender without pinning the similarity; cosine is stated here rather than implied.

============================ COST, AND WHY IT IS SHAPED THIS WAY ============================
The personalized step is a 150-NN search over 140,768 training users plus an information-gain sweep
per user per turn. Two things make it affordable without truncating anything (HARD RULE 1):

  * ONLY TOUCHED ITEMS CAN HAVE NONZERO GAIN. If none of a user's l neighbours rated item a, then
    C_a^{not-rated} is the whole neighbour set and IG(a) = H(C) - H(C) = 0 identically. So the sweep
    is over the (at most l * mean-profile-length) items the neighbours actually rated, and that is an
    EXACT statement about the criterion, not a candidate-pool reduction: every other item is scored,
    it just provably scores zero.
  * THE UNRATED BRANCH IS DERIVED BY SUBTRACTION, never gathered. Its cluster histogram is the
    neighbour histogram minus the item's rater histogram, and since entropy is
    H = log N - (1/N) SUM_c n_c log n_c, only the clusters an item's raters touch change -- the same
    accounting trick Golbandi's Eq. 3 uses for its own unknown branch.

Everything is vectorised over a CHUNK of users at a time; the chunk is a memory device only (HARD
RULE 11c), every user of the cohort is processed every turn.

INTERFACE (the `Selector` contract from `adaptive_harness.py`):

    sel = IGCNSelector(g_train, n_items, n_clusters=300, n_nbr=150, switch_i=5)
    sel.reset(ctx); sel.select(turn, users, asked, answered); sel.stats()
"""
import json
import os
import time
import numpy as np
from scipy import sparse

# --------------------------------------------------------------------------- published constants
N_CLUSTERS = 300            # paper: c
N_NEIGHBOURS = 150          # paper: l
SWITCH_AFTER = 5            # paper: i  (ratings before the personalized step takes over)
PAGE_SIZE = 15              # paper: n  (items per screen; the literal re-rank period)
NLEV = 10                   # our half-star bands -- the "r" categories of a real rating
W_UNRATED = 0.5             # paper's Guiasu weight w_0 for the not-rated category
W_RATED = 1.0               # paper's w_1..w_5


def _xlogx(x):
    """x*log(x) with 0*log 0 = 0, elementwise, no warnings."""
    x = np.asarray(x, np.float64)
    return np.where(x > 0.0, x * np.log(np.where(x > 0.0, x, 1.0)), 0.0)


def _ranges(starts, counts):
    """Concatenated aranges: [starts[j], starts[j]+counts[j]) for every j, in one array.

    Rows with count 0 are dropped by the caller; leaving them in would make the cumulative-sum trick
    write two values to the same position and silently corrupt the expansion."""
    starts = np.asarray(starts, np.int64)
    counts = np.asarray(counts, np.int64)
    total = int(counts.sum())
    if total == 0:
        return np.empty(0, np.int64)
    idx = np.ones(total, np.int64)
    idx[0] = starts[0]
    if len(starts) > 1:
        pos = np.cumsum(counts)[:-1]
        idx[pos] = starts[1:] - (starts[:-1] + counts[:-1]) + 1
    return np.cumsum(idx)


def _group_starts(sorted_keys):
    """Start offsets of each run of equal values in an already-sorted key array."""
    if len(sorted_keys) == 0:
        return np.empty(0, np.int64)
    new = np.empty(len(sorted_keys), bool)
    new[0] = True
    np.not_equal(sorted_keys[1:], sorted_keys[:-1], out=new[1:])
    return np.flatnonzero(new)


# ============================================================================ clustering
def bisecting_kmeans(X, n_clusters=N_CLUSTERS, seed=98765, n_iter=10, log=print):
    """Spherical bisecting k-means over the rows of X -> (n_rows,) int32 cluster labels.

    The paper says "bisecting k-means" and nothing more; this is the standard CLUTO-style form its
    own authors' toolkit implements -- L2-normalise the rows, repeatedly take the LARGEST cluster and
    split it in two by cosine 2-means, until c clusters exist. Largest-first is stated here because
    the alternative (worst-cohesion-first) gives a different tree and the choice is not in the paper.

    Deterministic given `seed`. Rows with no ratings at all form their own degenerate group and are
    never chosen as a seed (a zero row is equidistant from everything and would collapse a split)."""
    t0 = time.time()
    X = sparse.csr_matrix(X, dtype=np.float32)
    n = X.shape[0]
    norms = np.sqrt(np.asarray(X.multiply(X).sum(1)).ravel())
    Xn = sparse.diags(1.0 / np.maximum(norms, 1e-12)).dot(X).tocsr()
    labels = np.zeros(n, np.int32)
    sizes = {0: np.arange(n, dtype=np.int64)}
    frozen = set()                      # clusters that cannot be split (all-empty rows, or a
    rng = np.random.default_rng(seed)   # degenerate 2-means); never reconsidered
    nxt = 1
    while nxt < n_clusters:
        # largest cluster that can actually be split
        cand = [c for c, m in sizes.items() if len(m) >= 2 and c not in frozen]
        if not cand:
            log(f"[igcn] bisecting k-means stopped early at {nxt} clusters: nothing left to split")
            break
        c = max(cand, key=lambda c: len(sizes[c]))
        mem = sizes[c]
        S = Xn[mem]
        pool = np.flatnonzero(np.asarray(S.getnnz(axis=1) > 0).ravel())
        if len(pool) < 2:
            frozen.add(c)               # all-empty (or all-but-one-empty): not splittable
            continue
        a, b = rng.choice(pool, size=2, replace=False)
        Cn = sparse.vstack([S[a], S[b]]).tocsr()
        side = None
        for _ in range(n_iter):
            sim = np.asarray(S.dot(Cn.T).todense())            # (m x 2) cosine, rows unit norm
            new_side = (sim[:, 1] > sim[:, 0]).astype(np.int8)
            if new_side.all() or not new_side.any():           # degenerate: re-seed once, then stop
                new_side = np.zeros(len(mem), np.int8)
                new_side[rng.choice(len(mem), size=max(1, len(mem) // 2), replace=False)] = 1
            if side is not None and np.array_equal(side, new_side):
                break
            side = new_side
            cen = []
            for s in (0, 1):
                m = S[side == s]
                v = np.asarray(m.sum(0)).ravel()
                cen.append(v / max(np.linalg.norm(v), 1e-12))
            Cn = sparse.csr_matrix(np.vstack(cen).astype(np.float32))
        left, right = mem[side == 0], mem[side == 1]
        if len(left) == 0 or len(right) == 0:                  # never emit an empty cluster
            frozen.add(c)
            continue
        sizes[c] = left
        sizes[nxt] = right
        labels[right] = nxt
        nxt += 1
    # compact the (possibly frozen-keyed) label space to 0..K-1
    uniq, labels = np.unique(labels, return_inverse=True)
    labels = labels.astype(np.int32)
    k = len(uniq)
    cnt = np.bincount(labels, minlength=k)
    log(f"[igcn] bisecting k-means: {k} clusters over {n} users in {time.time() - t0:.1f}s "
        f"(sizes min={cnt.min()} median={int(np.median(cnt))} max={cnt.max()})")
    return labels, k


def cluster_contract(n_users, n_items, nnz, n_clusters, seed, n_iter):
    return {"n_users": int(n_users), "n_items": int(n_items), "nnz": int(nnz),
            "n_clusters": int(n_clusters), "seed": int(seed), "n_iter": int(n_iter),
            "algo": "spherical bisecting 2-means, largest-first"}


def load_or_fit_clusters(g_train, n_clusters=N_CLUSTERS, seed=98765, n_iter=10, cache_path=None,
                         log=print):
    """Contract-signed cluster sidecar (HARD RULE 11b). Clustering 140,768 users is the one expensive
    setup step and it is deterministic, so it is written once and reloaded; a cache whose contract
    does not match is REFUSED and recomputed, never silently reused."""
    G = g_train.tocsr()
    want = cluster_contract(G.shape[0], G.shape[1], G.nnz, n_clusters, seed, n_iter)
    if cache_path and os.path.exists(cache_path):
        try:
            blob = np.load(cache_path, allow_pickle=False)
            got = json.loads(str(blob["contract"]))
            if got == want:
                lab = blob["labels"].astype(np.int32)
                log(f"[igcn] clusters loaded from {os.path.basename(cache_path)} "
                    f"({int(blob['k'])} clusters)")
                return lab, int(blob["k"])
            log(f"[igcn] cluster cache REFUSED (contract mismatch)\n    cached={got}\n    wanted={want}")
        except Exception as e:
            log(f"[igcn] cluster cache unreadable ({e}) -- recomputing")
    lab, k = bisecting_kmeans(G, n_clusters=n_clusters, seed=seed, n_iter=n_iter, log=log)
    if cache_path:
        os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
        tmp = cache_path + ".tmp.npz"
        np.savez_compressed(tmp, labels=lab, k=k, contract=json.dumps(want))
        os.replace(tmp, cache_path)
        log(f"[igcn] clusters CHECKPOINTED -> {cache_path}")
    return lab, k


# ============================================================================ information gain
def _bands_of(G):
    """Half-star band 0..NLEV-1 for every stored rating of a graded CSR (same map as
    interview_strategies._star_to_level, so the IG categories and the entropy0 arm agree)."""
    return np.clip(np.rint(np.asarray(G.data, np.float64) * 2).astype(np.int64) - 1, 0, NLEV - 1)


def global_ig(g_train, labels, k_clusters, n_items):
    """IG(a) over ALL training users -- the paper's NON-PERSONALIZED step. Returns (n_items,) float64.

    Same weighted criterion as the personalized step, computed once. An item nobody rated scores
    exactly 0: its not-rated category is the whole cohort, so H(C_a^unrated) = H(C)."""
    G = g_train.tocsr()
    n_u = G.shape[0]
    band = _bands_of(G)
    item = G.indices.astype(np.int64)
    clus = np.repeat(labels.astype(np.int64), np.diff(G.indptr))

    hist = np.bincount(labels.astype(np.int64), minlength=k_clusters).astype(np.float64)
    S_all = _xlogx(hist).sum()
    H_C = np.log(n_u) - S_all / n_u

    # ---- (item, band, cluster) counts
    key = (item * NLEV + band) * k_clusters + clus
    k3, n3 = np.unique(key, return_counts=True)
    n3 = n3.astype(np.float64)
    c3 = k3 % k_clusters
    ib = k3 // k_clusters                                   # = item*NLEV + band

    st = _group_starts(ib)
    N_ib = np.add.reduceat(n3, st)
    S_ib = np.add.reduceat(_xlogx(n3), st)
    H_ib = np.log(N_ib) - S_ib / N_ib
    ib_u = ib[st]

    # ---- rated mass and weighted rated entropy, per item
    it_of_ib = ib_u // NLEV
    st2 = _group_starts(it_of_ib)
    mass_r = np.add.reduceat(W_RATED * N_ib, st2)
    num_r = np.add.reduceat(W_RATED * N_ib * H_ib, st2)
    items = it_of_ib[st2]

    # ---- the unrated branch by subtraction: histogram = global - this item's raters
    key2 = (k3 // (NLEV * k_clusters)) * k_clusters + c3     # = item*k + cluster
    k2, inv = np.unique(key2, return_inverse=True)
    m2 = np.bincount(inv, weights=n3)
    c2 = k2 % k_clusters
    i2 = k2 // k_clusters
    delta = _xlogx(hist[c2]) - _xlogx(hist[c2] - m2)
    st3 = _group_starts(i2)
    dsum = np.add.reduceat(delta, st3)
    assert np.array_equal(i2[st3], items), "global_ig: item grouping disagreed between the two passes"

    N_unr = n_u - np.add.reduceat(m2, st3)
    S_unr = S_all - dsum
    H_unr = np.where(N_unr > 0, np.log(np.maximum(N_unr, 1.0)) - S_unr / np.maximum(N_unr, 1.0), 0.0)

    num = num_r + W_UNRATED * N_unr * H_unr
    mass = mass_r + W_UNRATED * N_unr
    ig = np.zeros(n_items, np.float64)
    ig[items] = H_C - num / np.maximum(mass, 1e-12)
    return ig


def personalised_ig(nbr, g_train, band_all, labels, k_clusters, n_items):
    """IG per (user, item) restricted to each user's own `l` neighbours -- the PERSONALIZED step.

    nbr : (U, l) int array of TRAIN user ids, one row per cold user in this chunk.
    Returns (u_of, item_of, ig) flat arrays covering exactly the (user, item) pairs with a neighbour
    who rated the item -- every other item has IG identically 0 (see the module header)."""
    G = g_train
    U, l = nbr.shape
    flat = nbr.ravel().astype(np.int64)
    urow = np.repeat(np.arange(U, dtype=np.int64), l)
    lab_flat = labels[flat].astype(np.int64)

    # neighbour-set cluster histogram, per user
    hist = np.zeros((U, k_clusters), np.float64)
    np.add.at(hist, (urow, lab_flat), 1.0)
    S_G = _xlogx(hist).sum(1)
    N_u = float(l)
    H_C = np.log(N_u) - S_G / N_u

    cnts = (G.indptr[flat + 1] - G.indptr[flat]).astype(np.int64)
    keep = cnts > 0
    if not keep.any():
        return (np.empty(0, np.int64), np.empty(0, np.int64), np.empty(0, np.float64))
    pos = _ranges(G.indptr[flat[keep]], cnts[keep])
    E_u = np.repeat(urow[keep], cnts[keep])
    E_c = np.repeat(lab_flat[keep], cnts[keep])
    E_i = G.indices[pos].astype(np.int64)
    E_b = band_all[pos]

    key = ((E_u * n_items + E_i) * NLEV + E_b) * k_clusters + E_c
    k4, n4 = np.unique(key, return_counts=True)
    n4 = n4.astype(np.float64)
    c4 = k4 % k_clusters
    ib = k4 // k_clusters                                    # = (u*n_items + i)*NLEV + b

    st = _group_starts(ib)
    N_ib = np.add.reduceat(n4, st)
    S_ib = np.add.reduceat(_xlogx(n4), st)
    H_ib = np.log(N_ib) - S_ib / N_ib
    ui_of_ib = ib[st] // NLEV                                # = u*n_items + i

    st2 = _group_starts(ui_of_ib)
    mass_r = np.add.reduceat(W_RATED * N_ib, st2)
    num_r = np.add.reduceat(W_RATED * N_ib * H_ib, st2)
    ui = ui_of_ib[st2]

    key2 = (k4 // (NLEV * k_clusters)) * k_clusters + c4      # = (u*n_items+i)*k + cluster
    k2, inv = np.unique(key2, return_inverse=True)
    m2 = np.bincount(inv, weights=n4)
    c2 = k2 % k_clusters
    ui2 = k2 // k_clusters
    u2 = ui2 // n_items
    g2 = hist[u2, c2]
    delta = _xlogx(g2) - _xlogx(g2 - m2)
    st3 = _group_starts(ui2)
    assert np.array_equal(ui2[st3], ui), "personalised_ig: (user,item) grouping disagreed"

    N_unr = N_u - np.add.reduceat(m2, st3)
    S_unr = S_G[ui // n_items] - np.add.reduceat(delta, st3)
    H_unr = np.where(N_unr > 0, np.log(np.maximum(N_unr, 1.0)) - S_unr / np.maximum(N_unr, 1.0), 0.0)

    num = num_r + W_UNRATED * N_unr * H_unr
    mass = mass_r + W_UNRATED * N_unr
    ig = H_C[ui // n_items] - num / np.maximum(mass, 1e-12)
    return ui // n_items, ui % n_items, ig


# ============================================================================ the Selector
try:
    from adaptive_harness import Selector as _Selector
except Exception:                                            # importable standalone (snap script)
    class _Selector:                                         # pragma: no cover
        name = "selector"

        def reset(self, ctx):
            self.ctx = ctx

        def select(self, turn, users, asked, answered):
            raise NotImplementedError

        def stats(self):
            return {}


class IGCNSelector(_Selector):
    """Rashid'08 Algorithm 3.1 as a question SELECTOR for `adaptive_harness.run_interview`.

    Clusters and the global IG order are computed ONCE on the TRAIN users; no test user is ever seen
    by either. Per turn, a user with fewer than `switch_i` answers walks the global order (the paper's
    non-personalized step); a user at or past it gets their `n_nbr` nearest training neighbours and an
    IG recomputed from those neighbours alone.

    `stats()` reports `personalised_rate`, the share of asks that came from the personalized step. A
    row with personalised_rate 0 is the global-IG static order and must be read as such."""
    name = "igcn"

    def __init__(self, g_train=None, n_items=None, n_clusters=N_CLUSTERS, n_nbr=N_NEIGHBOURS,
                 switch_i=SWITCH_AFTER, recompute_every=1, seed=98765, kmeans_iter=10,
                 nn_chunk=128, ig_chunk=256, cache_path=None, fallback_order=None, log=print):
        self.g_train, self.n_items = g_train, n_items
        self.n_clusters, self.n_nbr = int(n_clusters), int(n_nbr)
        self.switch_i, self.recompute_every = int(switch_i), max(1, int(recompute_every))
        self.seed, self.kmeans_iter = int(seed), int(kmeans_iter)
        self.nn_chunk, self.ig_chunk = int(nn_chunk), int(ig_chunk)
        self.cache_path, self.fallback_order, self.log = cache_path, fallback_order, log
        self.labels = self.k_clusters = self.order = None
        self._nbr_cache = {}                                 # answered-state -> neighbour ids
        self._pick_cache = {}                                # (state, asked) -> chosen item
        self._n_pers = self._n_glob = self._n_degen = 0
        self._pers_per_turn = []

    # ---------------------------------------------------------------- setup
    def fit(self):
        G = sparse.csr_matrix(self.g_train)
        self.labels, self.k_clusters = load_or_fit_clusters(
            G, n_clusters=self.n_clusters, seed=self.seed, n_iter=self.kmeans_iter,
            cache_path=self.cache_path, log=self.log)
        t0 = time.time()
        ig = global_ig(G, self.labels, self.k_clusters, self.n_items)
        cnt = np.asarray((G > 0).sum(0)).ravel().astype(np.float64)
        # tie-break by train popularity so the head of the order is well defined (many items score a
        # hard 0 -- nobody rated them -- and an arbitrary argsort there is not reproducible)
        self.order = np.lexsort((-cnt, -ig)).astype(np.int64)
        self.ig_global = ig
        self.log(f"[igcn] global IG over {G.shape[0]} train users, {self.k_clusters} clusters, "
                 f"{time.time() - t0:.1f}s; top-5 items {self.order[:5].tolist()} "
                 f"IG {np.round(ig[self.order[:5]], 4).tolist()}; "
                 f"{int((ig > 0).sum())}/{self.n_items} items have nonzero gain")
        self._G = G
        self._band = _bands_of(G)
        gn = np.sqrt(np.asarray(G.multiply(G).sum(1)).ravel())
        self._Gn = sparse.diags(1.0 / np.maximum(gn, 1e-12)).dot(G).tocsr().astype(np.float32)
        self._GnT = self._Gn.T.tocsr()
        return self

    def reset(self, ctx):
        self.ctx = ctx
        if self.g_train is None:
            self.g_train = ctx["D"]["g_train"]
        if self.n_items is None:
            self.n_items = int(ctx["n_items"])
        if self.order is None:
            self.fit()
        if self.fallback_order is None:
            glob = ctx.get("glob") or {}
            self.fallback_order = np.asarray(glob.get("popularity", self.order), np.int64)
        self._pers_per_turn = []

    # ---------------------------------------------------------------- neighbours
    def _neighbours(self, states):
        """states: list of answered-tuples (each non-empty). -> (len(states), l) train-user ids.

        Cosine over the graded vectors, computed in chunks of `nn_chunk` states. Memoised on the
        answered state itself, so a BURNED turn (an asked-but-unrated item adds no answer) costs
        nothing -- the neighbour set is a pure function of the answers, not of the asks."""
        todo = [s for s in states if s not in self._nbr_cache]
        for s0 in range(0, len(todo), self.nn_chunk):
            blk = todo[s0:s0 + self.nn_chunk]
            rows, cols, vals = [], [], []
            for j, st in enumerate(blk):
                for (i, lv) in st:
                    rows.append(j)
                    cols.append(int(i))
                    vals.append((int(lv) + 1) / 2.0)         # half-star level -> star rating
            A = sparse.csr_matrix((np.asarray(vals, np.float32), (rows, cols)),
                                  shape=(len(blk), self.n_items))
            an = np.sqrt(np.asarray(A.multiply(A).sum(1)).ravel())
            A = sparse.diags(1.0 / np.maximum(an, 1e-12)).dot(A).tocsr().astype(np.float32)
            S = np.asarray((A @ self._GnT).todense())        # (b x n_train) cosine
            l = min(self.n_nbr, S.shape[1])
            top = np.argpartition(-S, l - 1, axis=1)[:, :l]
            for j, st in enumerate(blk):
                self._nbr_cache[st] = top[j].astype(np.int64)
        return np.vstack([self._nbr_cache[s] for s in states])

    # ---------------------------------------------------------------- score vector (gate G-F)
    def score_vector(self, turn, users, asked, answered):
        """See `Selector.score_vector`. EXACT for the GLOBAL phase; the personalised phase is
        reported as UNSCOREABLE rather than approximated, and the split is counted.

        Two facts about this arm force that shape, and both are already in the chapter:

        (1) THE GLOBAL PHASE DOES NOT ARGMAX `self.ig_global`. It walks the precomputed lexsorted
            `self.order` (`_next_global`), whose key is `(-information gain, -rating count)` -- and
            the gain has a large hard-zero tail where an argsort of `ig` alone is not even
            reproducible. So the honest score vector for this phase is the RANK IN THAT ORDER, i.e.
            `-position`, with the user's asked items unscoreable. That reproduces `_next_global`
            item for item, which `ig_global` plus an asked-mask would not.

        (2) THE PERSONALISED PHASE HAS NO DENSE VECTOR. `personalised_ig` returns sparse triples
            over (user, item) pairs with neighbour support only, then applies a strict positivity
            gate and, if it fails, DEGENERATES BACK to the global order. Emitting a dense
            approximation of that would be inventing a ranking the method never forms.

        The cost of (2) is bounded and is reported, not waved away: the phase fires only once a user
        has given `switch_i` ANSWERS, so at the Act-2 reference states it is inactive for the great
        majority of users at every budget this study scores. `score_vector_coverage()` returns the
        exact per-call split so the results table can print it beside the row."""
        rank = np.empty(self.n_items, dtype=np.float64)
        rank[self.order] = -np.arange(len(self.order), dtype=np.float64)
        out = np.full((len(users), self.n_items), -np.inf, dtype=np.float64)
        n_glob = n_pers = 0
        for j, u in enumerate(users):
            u = int(u)
            personalised = (len(answered[u]) >= self.switch_i
                            and (turn % self.recompute_every) == 0)
            if personalised:
                out[j, :] = np.nan          # UNSCOREABLE, and counted -- never approximated
                n_pers += 1
                continue
            v = rank.copy()
            a = np.asarray(asked[u], dtype=np.int64)
            if len(a):
                v[a[a < self.n_items]] = -np.inf
            out[j] = v
            n_glob += 1
        self._sv_glob = getattr(self, "_sv_glob", 0) + n_glob
        self._sv_pers = getattr(self, "_sv_pers", 0) + n_pers
        return out

    def score_vector_coverage(self):
        """(global, personalised) user-turns emitted by `score_vector` so far. The personalised
        count is the user mass this row's rank correlation is NOT measured on, and it is printed
        beside the row rather than left implicit."""
        g, p = getattr(self, "_sv_glob", 0), getattr(self, "_sv_pers", 0)
        return {"global_user_turns": int(g), "personalised_user_turns": int(p),
                "scoreable_fraction": (float(g) / (g + p)) if (g + p) else None}

    # ---------------------------------------------------------------- select
    def select(self, turn, users, asked, answered):
        out = np.empty(len(users), np.int64)
        pers_idx, pers_state = [], []
        n_pers_turn = 0
        # phase split: paper's rule, on the number of ANSWERS the user has actually given
        for j, u in enumerate(users):
            u = int(u)
            n_ans = len(answered[u])
            if n_ans < self.switch_i:
                out[j] = self._next_global(asked[u])
                self._n_glob += 1
                continue
            # the paper re-ranks per page of n items; with recompute_every > 1 a user between
            # re-ranks keeps walking the order fixed at the last re-rank -- here that is expressed by
            # simply not re-ranking, i.e. falling back to the global order for that turn.
            if (turn % self.recompute_every) != 0:
                out[j] = self._next_global(asked[u])
                self._n_glob += 1
                continue
            pers_idx.append(j)
            pers_state.append((u, tuple(answered[u]), tuple(sorted(int(x) for x in asked[u]))))
        if pers_idx:
            # group by (answered state, asked set): the pick is a pure function of the pair
            need = [(st, ak) for (_u, st, ak) in pers_state if (st, ak) not in self._pick_cache]
            uniq = list(dict.fromkeys(need))
            for s0 in range(0, len(uniq), self.ig_chunk):
                blk = uniq[s0:s0 + self.ig_chunk]
                nbr = self._neighbours([st for st, _ak in blk])
                uu, ii, ig = personalised_ig(nbr, self._G, self._band, self.labels,
                                             self.k_clusters, self.n_items)
                st_grp = _group_starts(uu) if len(uu) else np.empty(0, np.int64)
                bounds = np.append(st_grp, len(uu))
                seen = {int(uu[s]): (s, e) for s, e in zip(bounds[:-1], bounds[1:])}
                for j, (st, ak) in enumerate(blk):
                    rng = seen.get(j)
                    pick = -1
                    if rng is not None:
                        s, e = rng
                        it, g = ii[s:e], ig[s:e]
                        ok = ~np.isin(it, np.fromiter(ak, np.int64, len(ak)))
                        if ok.any():
                            g2 = np.where(ok, g, -np.inf)
                            b = int(np.argmax(g2))
                            if np.isfinite(g2[b]) and g2[b] > 0.0:
                                pick = int(it[b])
                    self._pick_cache[(st, ak)] = pick
            for j, (u, st, ak) in zip(pers_idx, pers_state):
                pick = self._pick_cache[(st, ak)]
                if pick < 0 or pick in set(asked[u]):
                    # every neighbour-touched candidate is already asked, or none has positive gain:
                    # the personalized step has nothing to say, so walk the global order and COUNT it
                    out[j] = self._next_global(asked[u])
                    self._n_degen += 1
                    self._n_glob += 1
                else:
                    out[j] = pick
                    self._n_pers += 1
                    n_pers_turn += 1
        self._pers_per_turn.append(n_pers_turn)
        # bound the memoisation tables (HARD RULE 11c); eviction costs recomputation, never a
        # different answer, because every cached value is a pure function of its key
        for tbl, cap in ((self._nbr_cache, 200_000), (self._pick_cache, 400_000)):
            while len(tbl) > cap:
                tbl.pop(next(iter(tbl)))
        return out

    def _next_global(self, asked_u):
        s = set(int(x) for x in asked_u)
        for i in self.order:
            if int(i) not in s:
                return int(i)
        for i in self.fallback_order:                        # unreachable at k <= 16
            if int(i) not in s:
                return int(i)
        raise RuntimeError("igcn: catalogue exhausted")

    def stats(self):
        tot = self._n_pers + self._n_glob
        return {"igcn_personalised_asks": int(self._n_pers),
                "igcn_global_asks": int(self._n_glob),
                "igcn_total_asks": int(tot),
                "igcn_personalised_rate": round(self._n_pers / max(tot, 1), 4),
                "igcn_degenerate_falls": int(self._n_degen),
                "igcn_personalised_per_turn": list(self._pers_per_turn),
                "igcn_n_clusters": int(self.k_clusters or 0),
                "igcn_n_neighbours": int(self.n_nbr),
                "igcn_switch_i": int(self.switch_i),
                "igcn_recompute_every": int(self.recompute_every),
                "igcn_nbr_states_cached": int(len(self._nbr_cache))}
