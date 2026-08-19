r"""M6a GATE -- does a cluster's teacher survive a REDDIT-SHAPED opener that is not its anchor?

Design sheet: `docs/design/DS_M6_REDDIT_SHAPED_OPENERS.md` (binding). Committed before the run.

Every teacher was fitted at ONE state: its anchor concept folded at the root. Its optimality has only
ever been measured there. The author's M6b proposal would label many different in-cluster opener
states -- a film, two films, a concept, two films and a concept -- with that same teacher. This gate
asks whether that labelling is legitimate:

    at Reddit-shaped NON-ANCHOR opener states inside cluster A, does tree-A still beat the cold tree?

PASS -> build the M6b training set.  FAIL -> the teacher is anchored to its own opener, the training
set would be mislabelled, and the design needs per-shape teachers. Report and stop.

The opener is folded into BOTH arms, so it is never the teacher's advantage. Openers are instantiated
from each user's OWN FOLD-IN half -- never from held-out targets, which would be clairvoyant.

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
SHAPES = ["anchor", "film1", "film2", "conc1", "film2conc1"]


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
    ap = argparse.ArgumentParser(description="M6a -- do teachers survive Reddit-shaped openers?")
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--i_have_author_goahead", action="store_true")
    a = ap.parse_args()
    assert a.i_have_author_goahead, "REFUSING TO RUN without --i_have_author_goahead"
    torch.set_num_threads(a.threads)
    np.random.seed(SEED)
    rng = np.random.default_rng(SEED)
    os.makedirs(OUT, exist_ok=True)

    from strategy_channel_suite import (build_greedy_env, genv_mean, genv_eval_multi,
                                        CERT_TOWER, CERT_CONC)
    from ndcg_tree import leak_assert, LIKE_MIN_LEVEL
    from tree_route import route
    from m4b_eval import load_tree

    log("#" * 100)
    log("# M6a GATE -- does a cluster's teacher survive a Reddit-shaped, non-anchor opener?")
    log("#" * 100)

    E = build_greedy_env(CERT_TOWER, "validation", CERT_CONC, "m6a")
    leak_assert(E, E["rows"], "m6a/val")
    rows_all = np.asarray(E["rows"], np.int64)
    f0, _ = genv_mean(E, [], rows_all)
    assert round(f0, 8) == round(VAL_M0_FULL, 8), f"CANONICAL SNAP FAILED: {f0!r}"
    log(f"[gate] canonical m=0 snap full={f0!r} -- exact")

    static = [(int(x), int(y)) for x, y in
              json.load(open(os.path.join(B, "greedy_static_policyB_combined.json")))["sequence"]]
    _cd, cold_nodes = load_tree(os.path.join(B, "coldtree_v2.json"))
    V, F = np.asarray(E["V"]), np.asarray(E["foldable"])
    endorse = (F > 0) & (V > 0)
    anchor_ids = set(ANCHORS.values())
    lvl = E["lvl"]
    results = []

    for name, cid in ANCHORS.items():
        _td, t_nodes = load_tree(os.path.join(B, f"m3v2_{name}.json"))
        cohort = rows_all[endorse[rows_all, cid]]
        log(f"[{name}] cohort {len(cohort)}")

        # ---- per-user opener material, from the user's OWN FOLD-IN half only -------------------
        liked, concs = {}, {}
        for r in cohort:
            r = int(r)
            ls = [int(i) for i, v in lvl[r].items() if v >= LIKE_MIN_LEVEL]
            liked[r] = sorted(ls)
            cs = [int(c) for c in np.nonzero(endorse[r])[0]
                  if int(c) not in anchor_ids]
            concs[r] = cs

        for shape in SHAPES:
            keep, op_items, op_concs = [], [], []
            for r in cohort:
                r = int(r)
                it, cc = [], []
                if shape == "anchor":
                    cc = [cid]
                elif shape == "film1":
                    if len(liked[r]) < 1:
                        continue
                    it = list(rng.choice(liked[r], 1, replace=False))
                elif shape == "film2":
                    if len(liked[r]) < 2:
                        continue
                    it = list(rng.choice(liked[r], 2, replace=False))
                elif shape == "conc1":
                    if len(concs[r]) < 1:
                        continue
                    cc = [int(rng.choice(concs[r]))]
                elif shape == "film2conc1":
                    if len(liked[r]) < 2 or len(concs[r]) < 1:
                        continue
                    it = list(rng.choice(liked[r], 2, replace=False))
                    cc = [int(rng.choice(concs[r]))]
                keep.append(r)
                op_items.append([int(x) for x in it])
                op_concs.append([int(x) for x in cc])
            if len(keep) < 150:
                log(f"   [{name}/{shape}] only {len(keep)} users can wear this opener -- skipping")
                continue
            keep = np.asarray(keep, np.int64)
            # The opener's own questions must never be ASKED as well as folded -- PER USER.
            # Flattening these into one set was a real bug: it masked every user's evidence out of
            # every other user's interview and truncated them all (caught by the router's assert).
            umask = {int(r): [(0, i) for i in oi] + [(1, c) for c in oc]
                     for r, oi, oc in zip(keep, op_items, op_concs)}
            seq_cold, _dc = route(E, keep, cold_nodes, TOTAL_L, static=static, mask=umask)
            seq_teach, _dt = route(E, keep, t_nodes, TOTAL_L, static=static, mask=umask)

            row = {"anchor": name, "shape": shape, "n": int(len(keep)), "budgets": {}}
            for K in BUDGETS:
                def ev(seqs):
                    iseq, cseq = [], []
                    for s, oi, oc in zip(seqs, op_items, op_concs):
                        ask = s[:K]
                        iseq.append(list(dict.fromkeys(list(oi) +
                                                       [q[1] for q in ask if q[0] == 0])))
                        cseq.append(list(dict.fromkeys(list(oc) +
                                                       [q[1] for q in ask if q[0] == 1])))
                    full, tail = genv_eval_multi(E, keep, iseq, cseq)
                    return np.asarray(full, np.float64), np.asarray(tail, np.float64)
                cf, ct = ev(seq_cold)
                tf, tt = ev(seq_teach)
                m, lo, hi = paired_boot(tf - cf)
                row["budgets"][str(K)] = {
                    "cold_full": float(np.nanmean(cf)), "teacher_full": float(np.nanmean(tf)),
                    "teacher_minus_cold_full": m, "ci_full": [lo, hi],
                    "teacher_minus_cold_tail": paired_boot(tt - ct)[0]}
            c = row["budgets"]["15"]
            star = "*" if c["ci_full"][0] > 0 else ("-" if c["ci_full"][1] < 0 else " ")
            log(f"   [{name}/{shape:<11}] n={len(keep):<5} K=15 cold {c['cold_full']:.5f} "
                f"teacher {c['teacher_full']:.5f} | {c['teacher_minus_cold_full']:+.5f} "
                f"[{c['ci_full'][0]:+.5f},{c['ci_full'][1]:+.5f}] {star}")
            results.append(row)
            json.dump({"analysis": "m6a_opener_shapes", "split": "validation",
                       "test_touched": False, "results": results},
                      open(os.path.join(OUT, "m6a_opener_shapes.json"), "w"), indent=1)

    log("=" * 100)
    log("[summary] teacher minus cold at K=15, by opener shape (mean over clusters)")
    for shape in SHAPES:
        sub = [r for r in results if r["shape"] == shape]
        if not sub:
            continue
        g = [r["budgets"]["15"]["teacher_minus_cold_full"] for r in sub]
        sep = sum(1 for r in sub if r["budgets"]["15"]["ci_full"][0] > 0)
        neg = sum(1 for r in sub if r["budgets"]["15"]["ci_full"][1] < 0)
        log(f"  {shape:<12} clusters={len(sub)}  mean {np.mean(g):+.5f}  "
            f"positive {sum(1 for x in g if x > 0)}/{len(g)}  separating +{sep} -{neg}")
    log("[done]")


if __name__ == "__main__":
    main()
