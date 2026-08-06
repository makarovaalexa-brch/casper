r"""golbandi_grow_ml25m.py -- grow the WSDM'11 ternary tree ONCE on the canonical ML-25M Liang train
cohort and checkpoint it, so the interview harness never has to pay for it twice.

WHY IT IS A SEPARATE SCRIPT. Growth is the only expensive step in the golbandi_tree arm: an exhaustive
argmin over all 18,359 candidates at every node, over 140,768 train users and 21.7M ratings. It is also
deterministic and independent of the budget, the consumer and the scoring contract -- a depth-d prefix
of the depth-8 tree IS the depth-d tree. So it is grown here, written to a contract-signed sidecar, and
loaded by `adaptive_harness --golbandi_tree_cache`. That is HARD RULE 11(b): the expensive intermediate
is checkpointed and resumed, and a run that dies in scoring does not repay the growth.

It also prints the OCCUPANCY table, which is the mechanism behind the harness's fallback rate: a cold
user whose node is a leaf before the budget is spent gets the next unasked popularity item instead, and
an arm that is mostly fallback is not this method.

    python scripts/baselines/golbandi_grow_ml25m.py --depth 8
"""
import os
import sys
import json
import time
import argparse

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
sys.path.insert(0, os.path.join(_ROOT, "src", "baselines"))
sys.path.insert(0, os.path.join(_ROOT, "src", "instrument"))

from arm_n import load_arm_n                                   # noqa: E402
from run_interview_table import logln                          # noqa: E402
import golbandi_tree_selector as GT                            # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--depth", type=int, default=8,
                    help="8 = golbandi_leaf.MAX_DEPTH, the consumer-side cap (3^k leaf collapse)")
    ap.add_argument("--xi", type=float, default=None,
                    help="min node rating mass; default = the paper's scale-free 0.2%% of corpus mass")
    ap.add_argument("--min_child", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(_ROOT, ".cache", "snap5",
                                                  "tree_ml25m_liang_d8.json"))
    a = ap.parse_args()

    D = load_arm_n(log=logln)
    g, ni = D["g_train"], D["n_items"]
    logln(f"[grow] train {g.shape[0]} users x {ni} items, nnz={g.nnz}; "
          f"like cut = star >= 4.0 (== half-star level >= {GT.LIKE_MIN_LEVEL})")
    sel = GT.GolbandiTreeSelector(g, ni, depth=a.depth, like_level=GT.LIKE_MIN_LEVEL, like_min=4.0,
                                  xi=a.xi, min_child_users=a.min_child, log=logln)
    t0 = time.time()
    sel.fit(cache_path=a.out)
    logln(f"[grow] total {(time.time() - t0) / 60:.1f} min -> {a.out}")

    occ = sel.tree.occupancy()
    logln("[grow] occupancy (depth: nodes / internal / users mean / median / #nodes<10u / ratings med)")
    for d, v in sorted(occ.items()):
        logln(f"[grow]   d={d}: nodes={v['n_nodes']:5d} internal={v['internal']:5d} "
              f"users_mean={v['users_mean']:9.1f} median={v['users_median']:8.1f} "
              f"n<10u={v['n_lt_10']:5d} ratings_median={v['ratings_median']:9.0f}")
    # Depth actually REACHED by a routed user is what decides the fallback rate; report the fraction of
    # train users sitting under an internal node at each depth (i.e. still askable).
    tot = float(sel.tree.nodes[0]["n_users"])
    for d in sorted(occ):
        askable = sum(nd["n_users"] for nd in sel.tree.nodes
                      if nd["depth"] == d and nd["item"] is not None)
        logln(f"[grow]   depth {d}: {askable}/{int(tot)} train users ({askable / tot:.1%}) sit at a "
              f"node that still has a question")
    json.dump(occ, open(a.out.replace(".json", "_occupancy.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
