r"""CLUSTER-LINKED OPENERS -- one seeded generator, shared by the fitter and the evaluator.

Author's spec, 2026-08-19: *"we need some opener, and a mix of both. and probably some variation of
how diagnostic, but should be linkable to that cluster."*

So, per user in a cluster:
  * ALWAYS an opener -- nobody in a cluster arrives silent;
  * a MIX OF BOTH CHANNELS -- films and attributes, in the proportions the M1b census measured on
    real requests (films only / signature only / both, renormalised because everyone speaks here);
  * VARIATION IN DIAGNOSTICITY -- the opener is drawn from a high / mid / low band of lift within the
    cluster's qualifying pool, so the fitted tree meets weak and strong statements of the same taste;
  * NEVER ARBITRARY -- every candidate clears a lift FLOOR against the global rate, so it is
    attributable to the cluster. This is the difference from M6a, whose uniform draw admitted
    anything the user happened to like and was correctly rejected as noise.

WHY A LIFT FLOOR AND NOT A TOP-N. M6c ranked by lift and took the top 60. That selected the sparse
tail -- `arthouse` -> *Sans Soleil*, `marvel` -> *Dracula Untold* -- and covered only 41.6% of the
`arthouse` cohort. A floor plus a minimum in-cluster support admits far more material while keeping
every item attributable, and lets diagnosticity vary WITHIN the qualifying pool rather than pinning
it at the extreme.

Lift is computed on TRAIN users only, so neither the cohort nor the opener is defined on the split a
measurement runs on. The draw is a pure function of (cluster, user, seed): the same user wears the
same opener at fit time and at scoring time, which is required or the tree would be fitted on one set
of states and scored on another.
"""
from __future__ import annotations

import numpy as np

# census shares of real requests that name a film / an attribute / both (M1B_OPENER_CENSUS.md),
# renormalised over the three speaking cells because every user in a cluster gets an opener here.
CENSUS = {"film_only": 0.271, "conc_only": 0.199, "both": 0.359}
BANDS = ("high", "mid", "low")


def build_lift_tables(E, cohort_rows, like_min_level, min_support=15, lift_floor=1.5):
    """Per-item and per-concept lift for one cluster, over the env's own cohort.

    lift(x) = P(likes/endorses x | in cluster) / P(likes/endorses x).  Items below `min_support`
    in-cluster likes are dropped: a ratio built on a handful of users is noise, and it is exactly
    what put *Sans Soleil* at the top of the arthouse list.
    """
    lvl = E["lvl"]
    V, F = np.asarray(E["V"]), np.asarray(E["foldable"])
    endorse = (F > 0) & (V > 0)
    n_all, n_items = endorse.shape[0], E["ctx"].ni

    all_cnt = np.zeros(n_items, np.float64)
    likes = []
    for r in range(n_all):
        ls = np.fromiter((int(i) for i, v in lvl[r].items() if v >= like_min_level), np.int64)
        likes.append(ls)
        all_cnt[ls] += 1
    in_cnt = np.zeros(n_items, np.float64)
    for r in cohort_rows:
        in_cnt[likes[int(r)]] += 1

    p_all = all_cnt / max(n_all, 1)
    p_in = in_cnt / max(len(cohort_rows), 1)
    lift_i = np.where(in_cnt >= min_support, p_in / np.maximum(p_all, 1e-9), 0.0)

    pc_all = endorse.mean(0)
    pc_in = endorse[cohort_rows].mean(0)
    sup_c = endorse[cohort_rows].sum(0)
    lift_c = np.where(sup_c >= min_support, pc_in / np.maximum(pc_all, 1e-9), 0.0)

    return {"lift_item": lift_i, "lift_conc": lift_c, "floor": float(lift_floor),
            "qual_items": np.nonzero(lift_i >= lift_floor)[0],
            "qual_concs": np.nonzero(lift_c >= lift_floor)[0]}


def _band_edges(vals):
    """Tertile cuts of lift within the qualifying pool -- the 'how diagnostic' axis."""
    if len(vals) < 3:
        return (0.0, 0.0)
    return (float(np.percentile(vals, 33.3)), float(np.percentile(vals, 66.7)))


def make_openers(E, cohort_rows, tables, like_min_level, seed, exclude_concepts=()):
    """Per-user opener as a list of (channel, id). Returns (openers, meta).

    Deterministic in `seed`: user u always receives the same opener for a given cluster and seed.
    A user with no qualifying material in the drawn channel falls back to the other channel; one with
    none at all is reported in `meta['unserved']` and receives an empty opener rather than a
    fabricated one.
    """
    rng = np.random.default_rng(seed)
    lvl = E["lvl"]
    V, F = np.asarray(E["V"]), np.asarray(E["foldable"])
    endorse = (F > 0) & (V > 0)
    li, lc = tables["lift_item"], tables["lift_conc"]
    banned_c = set(int(c) for c in exclude_concepts)

    qi, qc = set(int(x) for x in tables["qual_items"]), set(
        int(x) for x in tables["qual_concs"] if int(x) not in banned_c)
    ei = _band_edges(li[list(qi)]) if qi else (0.0, 0.0)
    ec = _band_edges(lc[list(qc)]) if qc else (0.0, 0.0)

    def band_of(v, e):
        return "low" if v < e[0] else ("mid" if v < e[1] else "high")

    p = np.array([CENSUS["film_only"], CENSUS["conc_only"], CENSUS["both"]], np.float64)
    p /= p.sum()
    shapes = rng.choice(["film_only", "conc_only", "both"], size=len(cohort_rows), p=p)
    want_band = rng.choice(BANDS, size=len(cohort_rows))

    openers, meta = [], {"shape": [], "band": [], "unserved": 0, "fallback": 0,
                         "band_edges_item": ei, "band_edges_conc": ec,
                         "n_qual_items": len(qi), "n_qual_concs": len(qc)}
    for k, r in enumerate(cohort_rows):
        r = int(r)
        mine_f = [i for i, v in lvl[r].items() if v >= like_min_level and int(i) in qi]
        mine_c = [int(c) for c in np.nonzero(endorse[r])[0] if int(c) in qc]
        b = want_band[k]
        fb = [i for i in mine_f if band_of(li[i], ei) == b] or mine_f
        cb = [c for c in mine_c if band_of(lc[c], ec) == b] or mine_c
        shape = shapes[k]
        if shape == "film_only" and not fb:
            shape, meta["fallback"] = "conc_only", meta["fallback"] + 1
        elif shape == "conc_only" and not cb:
            shape, meta["fallback"] = "film_only", meta["fallback"] + 1
        if shape == "both" and not (fb and cb):
            shape = "film_only" if fb else "conc_only"
            meta["fallback"] += 1

        op = []
        if shape in ("film_only", "both") and fb:
            k_f = 2 if (len(fb) >= 2 and rng.random() < 0.5) else 1
            op += [(0, int(x)) for x in rng.choice(fb, k_f, replace=False)]
        if shape in ("conc_only", "both") and cb:
            op += [(1, int(rng.choice(cb)))]
        if not op:
            meta["unserved"] += 1
        openers.append(op)
        meta["shape"].append(shape)
        meta["band"].append(b)
    return openers, meta
