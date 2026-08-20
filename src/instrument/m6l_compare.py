r"""M6l -- the cluster teacher against the incumbents, at the states it was fitted on.

Author, 2026-08-19: *"compare its performance with old tree teacher and cold tree and static."*

FOUR ARMS, every one at the SAME per-user cluster-linked opener states, the opener folded free and
masked from being asked in all of them:

  cluster   the tree fitted ON these varied opener states (`m6k_<cluster>.json`)
  anchor    that group's BANKED teacher, fitted at one state -- its anchor concept folded
  cold      the incumbent cold tree, depth 11
  static    the policy-B combined greedy order, context rather than headline

The openers are READ FROM THE FIT ARTIFACT, not regenerated, so the states scored are bit-identical
to the states fitted -- regenerating them would be a second draw and a different experiment.

WHAT A RESULT MEANS HERE
  cluster > cold      a teacher fitted on realistic openers earns its keep where the banked one did
                      not (M6a/M6c: the anchor teacher goes NEGATIVE at these states).
  cluster ~ cold      conditioning on cluster membership buys nothing once the opener is realistic,
                      and the forest's advantage is specific to a stated attribute.
  cluster < cold      fitting on heterogeneous states costs more than the conditioning returns.
All three are reportable; the second and third are not retried.

⚠ IN-SAMPLE, as M3 and M4a were: this tree is fitted and scored on the same validation users, so the
level is optimistic and only the CONTRAST across arms is informative. The anchor and cold arms carry
the same in-sample advantage, and the static arm carries none, which is why it is reported.

VALIDATION ONLY. The 10,000 canonical TEST users are never built, loaded or named.
"""
from __future__ import annotations

import os
import sys

_H = os.path.dirname(os.path.abspath(__file__))
_R = os.path.abspath(os.path.join(_H, "..", ".."))
sys.path.insert(0, _H)

import argparse
import json
import time

import numpy as np
import torch

B = os.path.join(_R, "experiments", "battery")
OUT = os.path.join(B, "m6")
SEED = 98765
TOTAL_L = 15
VAL_M0_FULL = 0.15686538118354684
ANCHORS = {"arthouse": 68, "dialogue": 275, "marvel": 581,
           "gore": 417, "teenagers": 922, "inspiring": 503}
BUDGETS = [1, 2, 4, 8, 15]


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def paired_boot(d, n=10000, seed=SEED):
    d = np.asarray(d, np.float64)
    d = d[np.isfinite(d)]
    if len(d) == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    bs = d[rng.integers(0, len(d), size=(n, len(d)))].mean(1)
    return float(d.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def main():
    ap = argparse.ArgumentParser(description="M6l -- cluster teacher vs anchor teacher vs cold vs static")
    ap.add_argument("--cluster", default="teenagers")
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--i_have_author_goahead", action="store_true")
    a = ap.parse_args()
    assert a.i_have_author_goahead, "REFUSING TO RUN without --i_have_author_goahead"
    torch.set_num_threads(a.threads)
    np.random.seed(SEED)

    from strategy_channel_suite import (build_greedy_env, genv_mean, genv_eval_multi,
                                        CERT_TOWER, CERT_CONC)
    from tree_route import route
    from m4b_eval import load_tree

    log("#" * 100)
    log(f"# M6l -- '{a.cluster}' cluster teacher vs anchor teacher vs cold tree vs static")
    log("#" * 100)

    fit = json.load(open(os.path.join(OUT, f"m6k_{a.cluster}.json"), encoding="utf-8"))
    nodes_cl = fit["nodes"]
    openers = {int(k): [(int(c), int(i)) for c, i in v] for k, v in fit["openers"].items()}
    cohort = np.asarray(sorted(openers), np.int64)
    log(f"[fit] {a.cluster}: {len(nodes_cl)} nodes, {len(cohort)} users, "
        f"fitted in {fit.get('minutes', float('nan')):.1f} min")

    E = build_greedy_env(CERT_TOWER, "validation", CERT_CONC, f"m6l_{a.cluster}")
    f0, _ = genv_mean(E, [], np.asarray(E["rows"], np.int64))
    assert round(f0, 8) == round(VAL_M0_FULL, 8), f"CANONICAL SNAP FAILED: {f0!r}"
    log(f"[gate] canonical m=0 snap full={f0!r} -- exact")

    static = [(int(x), int(y)) for x, y in
              json.load(open(os.path.join(B, "greedy_static_policyB_combined.json")))["sequence"]]
    _cd, cold_nodes = load_tree(os.path.join(B, "coldtree_v2.json"))
    _ad, anch_nodes = load_tree(os.path.join(B, f"m3v2_{a.cluster}.json"))
    cid = ANCHORS[a.cluster]

    umask = {int(r): list(openers[int(r)]) for r in cohort}
    seqs = {}
    for name, nodes in (("cluster", nodes_cl), ("anchor", anch_nodes), ("cold", cold_nodes)):
        s, dg = route(E, cohort, nodes, TOTAL_L, static=static, mask=umask, on_conflict="burn")
        seqs[name] = s
        if dg["n_fallback"]:
            log(f"[{name}] {dg['n_fallback']} users hit the continuation rule: {dg['sites']}")
    # the static arm carries the same per-user mask, so no arm may re-ask volunteered evidence
    seqs["static"] = [[q for q in static if q not in set(openers[int(r)])][:TOTAL_L] for r in cohort]

    # ⚠ the ANCHOR teacher's path begins with its free seed (1, cid); routing masks it out of the
    # asked list, and the anchor is NOT folded here -- these users volunteered an opener instead.
    res = {"cluster": a.cluster, "n": int(len(cohort)), "budgets": {}}
    for K in BUDGETS:
        def ev(name):
            iseq, cseq = [], []
            for r, s in zip(cohort, seqs[name]):
                op = openers[int(r)]
                ask = s[:K]
                iseq.append(list(dict.fromkeys([i for c, i in op if c == 0] +
                                               [q[1] for q in ask if q[0] == 0])))
                cseq.append(list(dict.fromkeys([i for c, i in op if c == 1] +
                                               [q[1] for q in ask if q[0] == 1])))
            full, tail = genv_eval_multi(E, cohort, iseq, cseq)
            return np.asarray(full, np.float64), np.asarray(tail, np.float64)

        f = {n: ev(n) for n in ("cluster", "anchor", "cold", "static")}
        row = {n: float(np.nanmean(f[n][0])) for n in f}
        row.update({n + "_tail": float(np.nanmean(f[n][1])) for n in f})
        for opp in ("cold", "anchor", "static"):
            m, lo, hi = paired_boot(f["cluster"][0] - f[opp][0])
            row[f"cluster_minus_{opp}"] = m
            row[f"ci_{opp}"] = [lo, hi]
            row[f"cluster_minus_{opp}_tail"] = paired_boot(f["cluster"][1] - f[opp][1])[0]
        res["budgets"][str(K)] = row
        st = lambda k: ("*" if row[f"ci_{k}"][0] > 0 else ("-" if row[f"ci_{k}"][1] < 0 else " "))
        log(f"  K={K:<3} cluster {row['cluster']:.5f} anchor {row['anchor']:.5f} "
            f"cold {row['cold']:.5f} static {row['static']:.5f} | "
            f"vs cold {row['cluster_minus_cold']:+.5f}{st('cold')} "
            f"vs anchor {row['cluster_minus_anchor']:+.5f}{st('anchor')} "
            f"vs static {row['cluster_minus_static']:+.5f}{st('static')}")
        json.dump({"analysis": "m6l_cluster_vs_incumbents", "split": "validation",
                   "test_touched": False, "results": res},
                  open(os.path.join(OUT, f"m6l_{a.cluster}.json"), "w"), indent=1)
    log("[done]")


if __name__ == "__main__":
    main()
