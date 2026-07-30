r"""golbandi_leaf.py -- Golbandi's NODE MODEL as an interview recommender, under an external strategy.

WHAT THIS IS, AND WHY IT NEEDS NO TREE. Golbandi, Koren & Lempel (WSDM 2011) do two things: their tree
CHOOSES which item to ask next, and each node CONSUMES the answers by holding the shrunk mean profile of
the training users routed to it. The strategy-portability experiment supplies the questions externally,
so only the second job is needed. Given k asked items and the user's like / dislike / unknown answers,
the "leaf" is simply the set of training users with the SAME answer pattern over those k items, and the
prediction is that group's shrunk mean profile. No tree growth, no candidate-pool choice, no sampling.

    branch(v, i) = unknown   if v never rated i
                   like      if r_vi > 3.5          <- on half-star data this is exactly r >= 4.0,
                   dislike   otherwise                 which is exactly Golbandi's own like/dislike cut
                                                       AND exactly Liang's keep/discard boundary.
    profile(S)_i = ( #{v in S : v LIKED i}  +  lam * global_like_rate_i ) / ( |S| + lam )   [lam=8]

RANKING ADAPTATION, stated plainly. Golbandi's node profile is the group's MEAN RATING, because their
metric is RMSE. Ranking by mean rating is a known disaster for top-N (an obscure film with one
enthusiastic rater outranks a film forty people liked) and, measured directly, it scores 0.0165 -- an
order of magnitude BELOW the no-information popularity prior of 0.1626. Reporting that would be
strawmanning the method, not reproducing it. So the node profile becomes the group's LIKE RATE: the
fraction of the leaf's users who liked each item, shrunk toward the global like rate. This is the same
implicit-feedback adaptation every other ratings-native baseline in the bank receives, it is exactly
what our binary user-kNN node reduction already computes, and it preserves the mechanism that matters
-- the profile is conditioned on the answer pattern. Note the denominator is the GROUP SIZE, not the
per-item rater count: that is what makes it a frequency rather than an average.

The shrinkage toward the global mean is Golbandi's (our archived ML-100k replication uses LAM=8) and it
is load-bearing: answer patterns fragment fast, so most leaves are small and an unshrunk group mean
would be noise.

DEPTH LIMIT. There are up to 3^k patterns. At k=8 that is 6,561 groups over 140,768 train users, ~21
users each -- thin but workable with shrinkage. Beyond k~8 the groups collapse and the shrunk mean just
returns the global mean, i.e. Most-Popular in disguise. Report this row only to k<=8 and say so rather
than printing a number that is not the method.

EFFICIENCY. Global strategies (popularity, entropy, entropy0, helf) ask EVERY user the same items, so
the partition is computed once and cached: an indicator matrix M (n_codes x n_train) times the graded
train matrix gives all group sums in one sparse product, with nnz bounded by nnz(g_train) because the
groups partition the users. Per-user strategies (the two random arms) get the complement trick instead:
only the raters of the k asked items have a nonzero code, and for random items that set is small.
"""
import numpy as np
from scipy import sparse

LIKE_MIN = 3.5          # r > 3.5, i.e. r >= 4.0 on half-star data
MAX_DEPTH = 8


class GolbandiLeaf:
    def __init__(self, g_train, n_items, lam=8.0, log=print):
        self.G = g_train.tocsr().astype(np.float32)
        self.Gc = self.G.tocsc()
        self.n_train, self.n_items = self.G.shape[0], n_items
        self.lam = float(lam)
        # LIKE indicator (r > 3.5), not the rating -- see the ranking-adaptation note in the header.
        L = self.G.copy()
        L.data = (L.data > LIKE_MIN).astype(np.float32)
        L.eliminate_zeros()
        self.L = L.tocsr()
        self.lsum = np.asarray(self.L.sum(axis=0)).ravel().astype(np.float64)
        # Global (root) profile = catalogue like RATE: where an all-unknown pattern, or an
        # over-fragmented leaf, correctly falls back to.
        self.global_mean = self.lsum / float(self.n_train)
        self._cache = {}
        log(f"[golbandi_leaf] node model over {self.n_train} train users, lambda={self.lam}")

    def _codes(self, asked):
        """(n_train,) ternary answer code of every TRAIN user over the asked items. 0 == all-unknown."""
        codes = np.zeros(self.n_train, dtype=np.int64)
        for j, i in enumerate(asked):
            s, e = self.Gc.indptr[i], self.Gc.indptr[i + 1]
            rows = self.Gc.indices[s:e]
            vals = self.Gc.data[s:e]
            branch = np.where(vals > LIKE_MIN, 2, 1).astype(np.int64)   # 1=dislike, 2=like, 0=unknown
            codes[rows] += branch * (3 ** j)
        return codes

    def _group_profiles(self, asked):
        """Shrunk profile for every answer code present, as (code -> row index) plus a sparse sum
        matrix. One sparse product; groups partition the users so nnz stays bounded by nnz(g_train)."""
        key = tuple(asked)
        if key in self._cache:
            return self._cache[key]
        codes = self._codes(asked)
        uniq, inv = np.unique(codes, return_inverse=True)
        M = sparse.csr_matrix((np.ones(self.n_train, np.float32),
                               (inv, np.arange(self.n_train))), shape=(len(uniq), self.n_train))
        S = np.asarray((M @ self.L).todense(), dtype=np.float64)        # (n_codes x n_items) LIKE counts
        gsz = np.asarray(M.sum(axis=1)).ravel()[:, None]                # leaf SIZE (not rater count)
        P = (S + self.lam * self.global_mean[None, :]) / (gsz + self.lam)
        out = ({int(c): r for r, c in enumerate(uniq)}, P.astype(np.float32))
        if len(uniq) <= 20000:                                          # cache only if it is worth it
            self._cache[key] = out
        return out

    def _profile_one(self, asked, code):
        """Complement-trick single-group profile, for per-user asked sets where building the whole
        partition would be wasteful. Only raters of the asked items have a nonzero code."""
        codes = self._codes(asked)
        idx = np.flatnonzero(codes == code)
        if len(idx) == 0:
            return self.global_mean.astype(np.float32)
        if len(idx) > self.n_train // 2:                                # big group: subtract the rest
            other = np.flatnonzero(codes != code)
            s = self.lsum - np.asarray(self.L[other].sum(axis=0)).ravel()
        else:
            s = np.asarray(self.L[idx].sum(axis=0)).ravel()
        return ((s + self.lam * self.global_mean) / (len(idx) + self.lam)).astype(np.float32)

    def scores(self, asked_per_user, answered_per_user, global_asked=None):
        """(n_users x n_items) dense scores. `global_asked` is the shared ask-list when the strategy is
        global (all four scored arms), which lets the whole partition be computed once."""
        n = len(asked_per_user)
        out = np.zeros((n, self.n_items), dtype=np.float32)
        if global_asked is not None:
            code_of, P = self._group_profiles(list(global_asked))
            for u in range(n):
                d = dict(answered_per_user[u])
                c = 0
                for j, i in enumerate(global_asked):
                    if i in d:
                        c += (2 if d[i] >= 7 else 1) * (3 ** j)         # level>=7 == r>=4.0 == like
                out[u] = P[code_of[c]] if c in code_of else self.global_mean
        else:
            for u in range(n):
                asked = asked_per_user[u]
                d = dict(answered_per_user[u])
                c = 0
                for j, i in enumerate(asked):
                    if i in d:
                        c += (2 if d[i] >= 7 else 1) * (3 ** j)
                out[u] = self._profile_one(asked, c)
        return out
