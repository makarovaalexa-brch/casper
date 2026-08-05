r"""golbandi_snap_answerability.py -- WHY the split search can lose: what fraction of the questions
it asks can a cold user actually answer?

The WSDM'11 criterion (Eq. 1) rewards an item for PARTITIONING the training cohort. It contains no
term for the probability that the next user has seen the item. On Netflix (>100M ratings, 480k users)
that omission is nearly free. On a sparse corpus it is not: the exhaustive argmin can settle on an
item that splits the training users beautifully and that four cold users in a hundred have seen, in
which case every one of the other ninety-six takes the UNKNOWN branch and the question bought
nothing. That is a concrete, measurable mechanism for an inversion, and it is exactly the mechanism
our own concept-vs-item bank result describes from the other side (a divisive ITEM is one few people
have seen).

This script reads the trees checkpointed by `golbandi_ml100k_snap.py` and reports, per depth and per
arm: the answer rate of the question asked, the popularity rank of the question, and the share of
cold users routed down the UNKNOWN branch.
"""
import argparse
import json
import os
import sys
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_ROOT, "src", "baselines"))

from golbandi_ml100k_snap import load_ml100k, split_cold      # noqa: E402

CACHE = os.path.join(_ROOT, ".cache", "snap5")


def route(nodes, answers, like_min, depth):
    nid, path, qs = 0, [0], []
    while len(path) - 1 < depth:
        nd = nodes[nid]
        if nd["item"] is None:
            break
        qs.append(nd["item"])
        r = answers.get(nd["item"])
        nid = nd["U" if r is None else ("L" if r >= like_min else "H")]
        path.append(nid)
    return path, qs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--depth", type=int, default=8)
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--n_cold", type=int, default=200)
    ap.add_argument("--fold_frac", type=float, default=0.7)
    ap.add_argument("--like_min", type=float, default=4.0)
    ap.add_argument("--arms", default="adaptive,static_popularity,static_var,static_helf")
    a = ap.parse_args()

    R = load_ml100k()
    arms = a.arms.split(",")
    agg = {arm: {d: {"ans": [], "unk": [], "rank": []} for d in range(a.depth)} for arm in arms}
    for seed in [int(s) for s in a.seeds.split(",")]:
        Rw, warm, fold, evl = split_cold(R, n_cold=a.n_cold, fold_frac=a.fold_frac, seed=seed)
        M = Rw.copy(); M.data = np.ones_like(M.data)
        cnt = np.asarray(M.sum(axis=0)).ravel()
        poprank = np.empty(len(cnt), dtype=np.int64)
        poprank[np.argsort(-cnt)] = np.arange(len(cnt))
        for arm in arms:
            p = os.path.join(CACHE, f"tree_s{seed}_{arm}_d{a.depth}.json")
            if not os.path.exists(p):
                print(f"  (missing checkpoint {p})")
                continue
            nodes = json.load(open(p))["nodes"]
            for cu, ans in fold.items():
                _, qs = route(nodes, ans, a.like_min, a.depth)
                for d, q in enumerate(qs):
                    agg[arm][d]["ans"].append(1.0 if q in ans else 0.0)
                    agg[arm][d]["unk"].append(0.0 if q in ans else 1.0)
                    agg[arm][d]["rank"].append(int(poprank[q]))

    print(f"\n=== QUESTION ANSWERABILITY by depth (cold users, fold-in side only), "
          f"{len(a.seeds.split(','))} seeds ===")
    print(f"{'depth':>5} | " + " | ".join(f"{arm.replace('static_',''):>26}" for arm in arms))
    print(f"{'':>5} | " + " | ".join(f"{'ansrate  medPopRank    n':>26}" for _ in arms))
    for d in range(a.depth):
        row = f"{d + 1:>5} | "
        cells = []
        for arm in arms:
            v = agg[arm][d]
            if not v["ans"]:
                cells.append(f"{'-':>26}")
                continue
            cells.append(f"{np.mean(v['ans']):>8.3f} {np.median(v['rank']):>11.0f} "
                         f"{len(v['ans']):>5d}")
        print(row + " | ".join(cells))
    print("\nansrate = fraction of cold users who had rated the question they were asked at that "
          "depth (the rest take the UNKNOWN branch and the question buys almost nothing).")
    print("medPopRank = median popularity rank of the asked item (0 = most-rated item in train).")


if __name__ == "__main__":
    main()
