r"""igcn_selftest.py -- prove the vectorised information gain equals a naive textbook implementation.

The fast path in `igcn_selector` does three things that could silently be wrong and that no downstream
number would reveal: it derives the NOT-RATED branch by subtracting the item's rater histogram from
the neighbour histogram (never gathering it), it computes group entropies through a sorted-key
reduceat instead of a loop, and it scores only the items some neighbour rated (asserting the rest are
identically zero). Each of those is checked here against a slow reference that does the obvious thing.

    python scripts/baselines/igcn_selftest.py
"""
import os
import sys
import numpy as np
from scipy import sparse

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..",
                                "src", "baselines"))
import igcn_selector as IG


def naive_ig(R_levels, labels, k_clusters, n_items, ref_rows):
    """Textbook IG(a;W) over the reference users `ref_rows`, one item at a time, no tricks."""
    lab = labels[ref_rows]
    N = len(ref_rows)

    def H(counts):
        counts = np.asarray(counts, np.float64)
        n = counts.sum()
        if n <= 0:
            return 0.0
        p = counts[counts > 0] / n
        return float(-(p * np.log(p)).sum())

    H_C = H(np.bincount(lab, minlength=k_clusters))
    out = np.zeros(n_items, np.float64)
    sub = R_levels[ref_rows].tocsc()
    for a in range(n_items):
        s, e = sub.indptr[a], sub.indptr[a + 1]
        rows, bands = sub.indices[s:e], sub.data[s:e].astype(int)
        num = den = 0.0
        rated = np.zeros(len(ref_rows), bool)
        for b in range(IG.NLEV):
            sel = rows[bands == b]
            if len(sel) == 0:
                continue
            rated[sel] = True
            cc = np.bincount(lab[sel], minlength=k_clusters)
            num += IG.W_RATED * len(sel) * H(cc)
            den += IG.W_RATED * len(sel)
        unr = np.flatnonzero(~rated)
        if len(unr):
            num += IG.W_UNRATED * len(unr) * H(np.bincount(lab[unr], minlength=k_clusters))
            den += IG.W_UNRATED * len(unr)
        out[a] = H_C - num / max(den, 1e-12)
    return out


def main():
    rng = np.random.default_rng(7)
    n_u, n_i, k = 240, 60, 6
    dens = 0.10
    M = (rng.random((n_u, n_i)) < dens)
    lev = rng.integers(0, IG.NLEV, size=(n_u, n_i))
    stars = (lev + 1) / 2.0
    G = sparse.csr_matrix(np.where(M, stars, 0.0))
    Glev = sparse.csr_matrix(np.where(M, lev + 1, 0))          # +1 so band 0 is not an implicit zero
    Glev.data = Glev.data - 1
    labels = rng.integers(0, k, size=n_u).astype(np.int32)

    # ---- 1. GLOBAL IG over every training user
    fast = IG.global_ig(G, labels, k, n_i)
    slow = naive_ig(Glev, labels, k, n_i, np.arange(n_u))
    d1 = np.abs(fast - slow).max()
    print(f"[selftest] global IG      max |fast - naive| = {d1:.3e}   over {n_i} items")
    assert d1 < 1e-10, "global_ig disagrees with the naive reference"

    # ---- 2. PERSONALISED IG for several users with different neighbour sets
    l = 25
    nbr = np.vstack([rng.choice(n_u, size=l, replace=False) for _ in range(5)])
    band = IG._bands_of(G)
    uu, ii, ig = IG.personalised_ig(nbr, G, band, labels, k, n_i)
    worst = 0.0
    for j in range(nbr.shape[0]):
        ref = naive_ig(Glev, labels, k, n_i, nbr[j])
        got = np.zeros(n_i)
        m = uu == j
        got[ii[m]] = ig[m]
        worst = max(worst, float(np.abs(got - ref).max()))
        # the sparsity claim: an item no neighbour rated must be EXACTLY zero in the reference too
        touched = np.zeros(n_i, bool)
        touched[ii[m]] = True
        assert np.abs(ref[~touched]).max() < 1e-12, (
            "an item no neighbour rated has nonzero naive IG -- the sparsity shortcut is invalid")
    print(f"[selftest] personalised IG max |fast - naive| = {worst:.3e}   over 5 neighbour sets")
    assert worst < 1e-10, "personalised_ig disagrees with the naive reference"

    # ---- 3. the _ranges expansion helper
    starts = np.array([3, 10, 0, 7], np.int64)
    counts = np.array([2, 3, 1, 4], np.int64)
    want = np.concatenate([np.arange(s, s + c) for s, c in zip(starts, counts)])
    assert np.array_equal(IG._ranges(starts, counts), want), "_ranges is wrong"
    print("[selftest] _ranges expansion OK")

    # ---- 4. clustering is deterministic and partitions every user
    lab1, k1 = IG.bisecting_kmeans(G, n_clusters=8, seed=1, log=lambda *_: None)
    lab2, k2 = IG.bisecting_kmeans(G, n_clusters=8, seed=1, log=lambda *_: None)
    assert np.array_equal(lab1, lab2) and k1 == k2, "bisecting k-means is not deterministic"
    assert set(np.unique(lab1)) == set(range(k1)) and len(lab1) == n_u, "labels are not a partition"
    print(f"[selftest] bisecting k-means deterministic, {k1} non-empty clusters, all users assigned")
    print("[selftest] PASS")


if __name__ == "__main__":
    main()
