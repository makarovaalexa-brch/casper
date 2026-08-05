r"""golbandi_ml100k_snap.py -- SNAP-5 step S: the WSDM'11 ternary tree, natively.

WHAT IS AND IS NOT BEING SNAPPED. Golbandi, Koren & Lempel (WSDM 2011) report on **Netflix**
(>100M ratings, 480k users, 17,770 items), **RMSE only**, test = the Qualifying set. We do not have
Netflix and it cannot be obtained, so a NUMERIC snap to their table is impossible and is not
attempted. What IS snapped is the QUALITATIVE CLAIM plus an archived internal precedent:

    the adaptive squared-loss tree beats a STATIC question order of the same length,
    on RMSE, at every depth >= 1,

on ML-100k, where `scripts/_archive/paper1_v1/golbandi_tree.py` previously reported adaptive 0.9804
vs static 0.9885 at depth 6. Say this out loud in any write-up: **shape snap on a different dataset,
not a numeric snap to Netflix.**

WHAT THIS SCRIPT ADDS OVER THE ARCHIVED ONE (all four are things the archived script got wrong or
could not do, and all four move the number):

 1. NO ANSWER LEAK. The archived script let the interview be answered from the cold user's WHOLE
    profile, including the very ratings it then scored RMSE on. Here each cold user's ratings are
    split into a fold-in side (which alone may answer questions) and a disjoint eval side.
 2. NO CANDIDATE-POOL TRUNCATION. The archived script searched only the top-80 items by train count
    (`POOL=80`); the paper is exhaustive over all n items at every node. Here the search is over all
    1,682 items (HARD RULE 1: no silent reduction).
 3. THE PAPER'S ACTUAL CONSTANTS. User-bias shrinkage beta=7 during growth; per-item hierarchical
    leaf shrinkage toward the PARENT (lam1), not a flat shrink toward the global mean; the node
    rating-mass stopping rule xi, rescaled from the paper's Netflix 200,000/100M = 0.2% of corpus
    mass and reported as such.
 4. THREE STATIC COMPARATORS, NOT ONE. Popularity (the archived comparator), Var = sqrt(|R(i)|)*std(i)
    and HELF -- i.e. the paper's own Fig. 3 static alternatives in spirit. A shape snap against one
    weak static is not evidence.

The static arms run through the SAME tree machinery with the question at depth d fixed in advance
(`forced_order`), so the node model, the ternary routing, the shrinkage and the question count are
literally identical and the only difference is the SELECTION.

Secondary, and reported separately with a warning: NDCG@10 from a like-rate leaf. Golbandi's own leaf
is a mean RATING, which is a known disaster as a top-N score (their own words: "a decision-tree
recommender delivers a quite high RMSE ... this makes decision-trees undesirable as a traditional
recommendation system"). The like-rate variant is the ranking adaptation documented in
`golbandi_leaf.py`. RMSE is the snap; NDCG is context for the inversion question.

PERSISTENCE (HARD RULE 11): every (arm, depth) cell is appended to the output JSON the moment it is
computed, and every grown tree is checkpointed to .cache/ before evaluation starts.

  python scripts/baselines/golbandi_ml100k_snap.py --depth 6
"""
import argparse
import json
import os
import sys
import time
import numpy as np
from scipy import sparse

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
sys.path.insert(0, os.path.join(_ROOT, "src", "baselines"))

from golbandi_tree_selector import GolbandiTree, XI_FRACTION      # noqa: E402

ML100K = os.path.join(_ROOT, "data", "movielens", "ml-100k", "u.data")
OUT = os.path.join(_ROOT, "experiments", "results", "snap5")
CACHE = os.path.join(_ROOT, ".cache", "snap5")


def log(*a):
    print(*a, flush=True)


# --------------------------------------------------------------------------- data
def load_ml100k(path=ML100K):
    u, i, r = [], [], []
    with open(path) as f:
        for line in f:
            a = line.split("\t")
            u.append(int(a[0])); i.append(int(a[1])); r.append(float(a[2]))
    u = np.asarray(u); i = np.asarray(i); r = np.asarray(r, dtype=np.float64)
    uu = {x: k for k, x in enumerate(np.unique(u))}
    ii = {x: k for k, x in enumerate(np.unique(i))}
    ur = np.asarray([uu[x] for x in u]); ir = np.asarray([ii[x] for x in i])
    R = sparse.csr_matrix((r, (ur, ir)), shape=(len(uu), len(ii)))
    return R


def split_cold(R, n_cold=200, fold_frac=0.7, seed=0):
    """n_cold users are held out entirely (never seen by the tree). Each cold user's ratings are cut
    into a fold-in side (may answer questions) and a disjoint eval side (RMSE / NDCG target)."""
    rng = np.random.default_rng(seed)
    n = R.shape[0]
    cold = np.sort(rng.choice(n, size=n_cold, replace=False))
    warm = np.setdiff1d(np.arange(n), cold)
    fold, evl = {}, {}
    for cu in cold:
        s, e = R.indptr[cu], R.indptr[cu + 1]
        items, vals = R.indices[s:e], R.data[s:e]
        if len(items) < 10:
            continue
        p = rng.permutation(len(items))
        cut = max(1, int(round(fold_frac * len(items))))
        fi, ei = p[:cut], p[cut:]
        if len(ei) == 0:
            continue
        fold[int(cu)] = dict(zip(items[fi].tolist(), vals[fi].tolist()))
        evl[int(cu)] = (items[ei], vals[ei])
    return R[warm].tocsr(), warm, fold, evl


# --------------------------------------------------------------------------- static orders
def static_orders(Rw, like_min):
    """The three static comparators, all computed on the TRAIN users only."""
    M = Rw.copy(); M.data = np.ones_like(M.data)
    cnt = np.asarray(M.sum(axis=0)).ravel()
    s = np.asarray(Rw.sum(axis=0)).ravel()
    s2 = Rw.copy(); s2.data = s2.data ** 2
    s2 = np.asarray(s2.sum(axis=0)).ravel()
    with np.errstate(invalid="ignore", divide="ignore"):
        mu = np.where(cnt > 0, s / np.maximum(cnt, 1), 0.0)
        var = np.where(cnt > 1, s2 / np.maximum(cnt, 1) - mu ** 2, 0.0)
    orders = {"popularity": np.argsort(-cnt)}
    orders["var"] = np.argsort(-(np.sqrt(cnt) * np.sqrt(np.maximum(var, 0))))
    # HELF: harmonic mean of normalised log-frequency and the rating-value entropy (Rashid 2008)
    lf = np.log(np.maximum(cnt, 1)) / max(np.log(max(cnt.max(), 2)), 1e-9)
    ent = np.zeros_like(lf)
    Rc = Rw.tocsc()
    for j in range(Rw.shape[1]):
        v = Rc.data[Rc.indptr[j]:Rc.indptr[j + 1]]
        if len(v) == 0:
            continue
        _, c = np.unique(np.round(v).astype(int), return_counts=True)
        p = c / c.sum()
        ent[j] = float(-(p * np.log(p)).sum()) / np.log(5.0)
    helf = np.where((lf > 0) & (ent > 0), 2 * lf * ent / np.maximum(lf + ent, 1e-9), 0.0)
    orders["helf"] = np.argsort(-helf)
    return orders


# --------------------------------------------------------------------------- evaluation
def ndcg_at_k(scores, eval_items, eval_vals, fold_items, like_min, k=10):
    """NDCG@10 over the cold user's EVAL-side liked items, with the fold-in side masked out."""
    rel = eval_items[eval_vals >= like_min]
    if len(rel) == 0:
        return None
    sc = scores.copy()
    sc[np.asarray(fold_items, dtype=np.int64)] = -np.inf
    top = np.argpartition(-sc, k)[:k]
    top = top[np.argsort(-sc[top])]
    relset = set(rel.tolist())
    dcg = sum(1.0 / np.log2(2 + r) for r, it in enumerate(top.tolist()) if it in relset)
    idcg = sum(1.0 / np.log2(2 + r) for r in range(min(k, len(rel))))
    return dcg / idcg if idcg > 0 else None


def eval_tree(tree, fold, evl, depth, like_min, clip=(1.0, 5.0), ranking=False):
    """Returns (per_depth_summary, per_user).

    `per_user` carries the PAIRED quantities -- each cold user's summed squared error and eval count
    at every depth, or their NDCG -- in a FIXED user order, so the adaptive-minus-static contrast can
    be bootstrapped per user rather than compared as two unpaired means. The chapter's own scar is
    exactly this (`paired-test-kills-the-tie-claim`): an unpaired CI made a real separation look like
    a tie."""
    users = sorted(fold.keys())
    per = {d: {"occ": [], "leafed": 0} for d in range(depth + 1)}
    se = np.zeros((len(users), depth + 1)); cn = np.zeros((len(users), depth + 1))
    nd = np.full((len(users), depth + 1), np.nan)
    for ui, cu in enumerate(users):
        ans = fold[cu]
        path = tree.route(ans, max_steps=depth)
        ei, ev = evl[cu]
        for d in range(depth + 1):
            nid = path[min(d, len(path) - 1)]
            per[d]["occ"].append(tree.nodes[nid]["n_users"])
            if len(path) - 1 < d:
                per[d]["leafed"] += 1
            prof = tree.profiles[nid]
            if not ranking:
                pred = np.clip(prof[ei], clip[0], clip[1])
                se[ui, d] = float(((ev - pred) ** 2).sum()); cn[ui, d] = len(ev)
            else:
                g = ndcg_at_k(prof.astype(np.float64), ei, ev, list(ans.keys()), like_min)
                if g is not None:
                    nd[ui, d] = g
    out = {}
    for d in range(depth + 1):
        p = per[d]
        tot_n = float(cn[:, d].sum())
        out[d] = {"rmse": (float(np.sqrt(se[:, d].sum() / tot_n)) if tot_n else None),
                  "ndcg10": (float(np.nanmean(nd[:, d])) if not np.all(np.isnan(nd[:, d]))
                             else None),
                  "n_eval_ratings": int(tot_n), "n_users": len(users),
                  "node_users_mean": float(np.mean(p["occ"])) if p["occ"] else None,
                  "node_users_median": float(np.median(p["occ"])) if p["occ"] else None,
                  "leafed_early_frac": round(p["leafed"] / max(len(users), 1), 4)}
    pu = {"users": users}
    if ranking:
        pu["ndcg"] = np.round(nd, 6).tolist()
    else:
        pu["se"] = np.round(se, 6).tolist(); pu["n"] = cn.astype(int).tolist()
    return out, pu


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--depth", type=int, default=6)
    ap.add_argument("--n_cold", type=int, default=200)
    ap.add_argument("--fold_frac", type=float, default=0.7)
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--like_min", type=float, default=4.0)
    ap.add_argument("--lams", default="8,16,32,64,128,200,400",
                    help="leaf shrinkage sweep. The paper's Netflix value is lam1=200; the archived "
                         "ML-100k script used 8. Both are run, plus the range between, because the "
                         "shrinkage turns out to decide whether depth helps or hurts -- reporting "
                         "one value would have made the snap verdict a function of an unstated "
                         "constant. Every value reuses the SAME grown tree.")
    ap.add_argument("--xi", type=float, default=None,
                    help="node rating-mass stop. Default = the paper's 0.2%% of corpus mass.")
    ap.add_argument("--min_child_users", type=int, default=0)
    ap.add_argument("--no_bias", action="store_true")
    ap.add_argument("--label", default="ml100k")
    a = ap.parse_args()

    os.makedirs(OUT, exist_ok=True); os.makedirs(CACHE, exist_ok=True)
    outp = os.path.join(OUT, f"golbandi_snap_{a.label}_d{a.depth}.json")
    R = load_ml100k()
    log(f"[snap5] ML-100k: {R.shape[0]} users x {R.shape[1]} items, {R.nnz} ratings")

    blob = {"meta": {"target": "Golbandi/Koren/Lempel WSDM 2011 (Netflix, RMSE); THIS IS A SHAPE "
                               "SNAP ON ML-100k, NOT a numeric snap to the published table",
                     "depth": a.depth, "n_cold": a.n_cold, "fold_frac": a.fold_frac,
                     "like_min": a.like_min, "lams": a.lams,
                     "xi": a.xi, "xi_rule": f"{XI_FRACTION:.4f} of corpus rating mass (paper: "
                                            f"200,000/100M on Netflix)",
                     "min_child_users": a.min_child_users, "bias_correct": not a.no_bias,
                     "candidate_pool": "EXHAUSTIVE over all items (no truncation)",
                     "seeds": a.seeds},
            "results": {}}

    def flush():
        json.dump(blob, open(outp, "w"), indent=2)

    for seed in [int(s) for s in a.seeds.split(",")]:
        Rw, warm, fold, evl = split_cold(R, n_cold=a.n_cold, fold_frac=a.fold_frac, seed=seed)
        log(f"[snap5] seed {seed}: train {Rw.shape[0]} users / {Rw.nnz} ratings; "
            f"cold cases {len(fold)}")
        orders = static_orders(Rw, a.like_min)
        arms = {"adaptive": None}
        for nm, o in orders.items():
            arms[f"static_{nm}"] = o
        blob["results"].setdefault(str(seed), {})
        for arm, order in arms.items():
            # ONE growth per (seed, arm). The criterion (Eq. 1) does not depend on lam1 or on the
            # profile kind, so every shrinkage / leaf-model variant below is the SAME TREE with its
            # profiles recomputed -- otherwise two arms could differ in their QUESTIONS while being
            # reported as differing only in their leaf.
            t0 = time.time()
            tree = GolbandiTree(Rw, like_min=a.like_min, depth=a.depth, xi=a.xi,
                                min_child_users=a.min_child_users, bias_correct=not a.no_bias,
                                forced_order=(None if order is None else order[:a.depth]),
                                store_profiles=False, store_rows=True, log=log).fit()
            ck = os.path.join(CACHE, f"tree_{a.label}_s{seed}_{arm}_d{a.depth}.json")
            tree.save(ck)                                        # HARD RULE 11: checkpoint the tree
            grow_s = round(time.time() - t0, 1)
            for lam1 in [float(x) for x in a.lams.split(",")]:
                for kind in ("mean", "like_rate"):
                    key = f"{arm}|lam1={lam1:g}|{kind}"
                    tree.recompute_profiles(lam1=lam1, profile_kind=kind)
                    res, pu = eval_tree(tree, fold, evl, a.depth, a.like_min,
                                        ranking=(kind == "like_rate"))
                    blob["results"][str(seed)][key] = {
                        "per_depth": {str(d): v for d, v in res.items()},
                        "occupancy": tree.occupancy(),
                        "root_question": tree.nodes[0]["item"],
                        "n_nodes": len(tree.nodes),
                        "n_internal": int(sum(1 for nd in tree.nodes if nd["item"] is not None)),
                        "grow_seconds": grow_s,
                        "checkpoint": os.path.relpath(ck, _ROOT),
                        "per_user": pu}
                    flush()                                      # HARD RULE 11: one cell, one write
                    m = res[a.depth]
                    log(f"[snap5] seed{seed} {key:38s} d{a.depth}: "
                        f"rmse={m['rmse'] if m['rmse'] is None else round(m['rmse'], 4)} "
                        f"ndcg={m['ndcg10'] if m['ndcg10'] is None else round(m['ndcg10'], 4)} "
                        f"leafed={m['leafed_early_frac']:.2f}")
    flush()
    log(f"[snap5] done -> {outp}")


if __name__ == "__main__":
    main()
