r"""M6r -- THE FAIR COMPARISON: every tree fitted on TRAIN, every number reported on VALIDATION.

Author's design, 2026-08-21: *"i think fitting all of the trees on train and applying to val is a
fair comparison?"* -- yes, and this measures it.

THE PROBLEM IT FIXES
--------------------
The incumbent cold tree was fitted on VALIDATION; the opener-fitted group trees on TRAIN. So every
comparison so far has been played on one arm's home ground, and the home advantage is not small --
the same arthouse tree scored +0.01762 against cold on train users and -0.00573 on validation, and
the two arms' own population effects (tree +0.01645, cold +0.00690) account for the entire gap.
See memory `train-user-contrast-does-not-cancel-inflation`.

With `coldtree_v2_train10000` both arms are fitted on TRAIN and both are applied to VALIDATION,
where neither has ever been fitted. Nobody is at home.

TWO CELLS
---------
**A. HOME ADVANTAGE, MEASURED DIRECTLY.** All 10,000 validation users at COLD states (nothing
volunteered): the validation-fitted incumbent versus the train-fitted rebuild. Same recipe, same N,
same pool, same min-leaf -- the ONLY difference is which population it was fitted on. `cold_val
minus cold_train` IS the home advantage, at n=10,000 rather than the +0.00782 I previously inferred
indirectly from `cold - static` across two populations.

**B. THE FAIR COMPARISON.** The arthouse cohort at opener states, sweeping the opener half:
the opener-fitted group tree against BOTH cold trees. `group - cold_train` is the honest number.
`group - cold_val` is what we have been quoting. The difference between them should be cell A.

⚠ The group tree carries an N handicap (2,000 fitting users vs cold's 10,000) that runs AGAINST it,
so a win here is conservative and a loss does not yet separate "no group signal" from "not enough
fitting data". Stated up front so it is not discovered afterwards.

**THE TEST SET IS NEVER BUILT, LOADED OR NAMED IN THIS FILE.**
"""
from __future__ import annotations

import os
import sys

_H = os.path.dirname(os.path.abspath(__file__))
_R = os.path.abspath(os.path.join(_H, "..", ".."))
sys.path.insert(0, _H)

import argparse
import itertools
import json
import time

import numpy as np
import torch

B = os.path.join(_R, "experiments", "battery")
M6 = os.path.join(B, "m6")
SEED = 98765
TOTAL_L = 15
ANCHORS = {"arthouse": 68, "dialogue": 275, "marvel": 581,
           "gore": 417, "teenagers": 922, "inspiring": 503}
BUDGETS = [1, 4, 15]


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


def score(E, rows, seqs, ops, budgets=BUDGETS):
    from strategy_channel_suite import genv_eval_multi
    out = {}
    for K in budgets:
        iseq, cseq = [], []
        for s, op in zip(seqs, ops):
            ask = s[:K]
            iseq.append(list(dict.fromkeys([i for c, i in op if c == 0] +
                                           [q[1] for q in ask if q[0] == 0])))
            cseq.append(list(dict.fromkeys([i for c, i in op if c == 1] +
                                           [q[1] for q in ask if q[0] == 1])))
        full, _t = genv_eval_multi(E, rows, iseq, cseq)
        out[K] = np.asarray(full, np.float64)
    return out


def table(sc, arms, budgets=BUDGETS):
    rec = {}
    for K in budgets:
        row = {a: float(np.nanmean(sc[a][K])) for a in arms}
        for a, b in itertools.combinations(arms, 2):
            m, lo, hi = paired_boot(sc[a][K] - sc[b][K])
            row[f"{a}_minus_{b}"] = m
            row[f"ci_{a}_minus_{b}"] = [lo, hi]
        rec[str(K)] = row
    return rec


def sig(rec, a, b, K="15"):
    lo, hi = rec[K][f"ci_{a}_minus_{b}"]
    return "*" if lo > 0 else ("-" if hi < 0 else " ")


def main():
    ap = argparse.ArgumentParser(description="M6r -- all trees train-fitted, reported on validation")
    ap.add_argument("--cold_train", default=os.path.join(B, "coldtree_v2_train10000.json"))
    ap.add_argument("--group_tree", default=os.path.join(M6, "m6k_arthouse_train2000.json"))
    ap.add_argument("--cluster", default="arthouse")
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--i_have_author_goahead", action="store_true")
    a = ap.parse_args()
    assert a.i_have_author_goahead, "REFUSING TO RUN without --i_have_author_goahead"
    torch.set_num_threads(a.threads)
    np.random.seed(SEED)

    from strategy_channel_suite import (build_greedy_env, CERT_TOWER, CERT_CONC)
    from ndcg_tree import build_pool, LIKE_MIN_LEVEL
    from tree_route import route
    from m4b_eval import load_tree
    from m3_show_paths import build_names
    from opener_gen import centroid_lift, make_openers
    cname, _iname = build_names()

    for p in (a.cold_train, a.group_tree):
        assert os.path.exists(p), f"missing artifact: {p}"
    ct = json.load(open(a.cold_train, encoding="utf-8"))
    assert ct.get("split") == "train", f"cold_train was fitted on {ct.get('split')!r}, not train"
    cold_train = ct["nodes"]
    if isinstance(cold_train, list):
        cold_train = {r["nid"]: r for r in cold_train}
    gt = json.load(open(a.group_tree, encoding="utf-8"))
    group = gt["nodes"]
    if isinstance(group, list):
        group = {r["nid"]: r for r in group}
    _cd, cold_val = load_tree(os.path.join(B, "coldtree_v2.json"))
    _ad, anchor = load_tree(os.path.join(B, f"m3v2_{a.cluster}.json"))
    static = [(int(x), int(y)) for x, y in
              json.load(open(os.path.join(B, "greedy_static_policyB_combined.json")))["sequence"]]

    log("#" * 100)
    log("# M6r -- ALL TREES TRAIN-FITTED, ALL NUMBERS ON VALIDATION. Nobody is at home.")
    log(f"# cold_train {len(cold_train)} nodes (n_cohort={ct.get('n_cohort')}) | "
        f"cold_val {len(cold_val)} nodes | group {len(group)} nodes "
        f"(n_fit={len(gt.get('fit_rows', []))})")
    log("#" * 100)

    E = build_greedy_env(CERT_TOWER, "validation", CERT_CONC, "m6r")
    pool, _i, _c = build_pool(E, 500, 500)
    vrows = np.asarray(E["rows"], np.int64)
    out = {"analysis": "m6r_fair_compare", "ruler": "validation_clean", "test_touched": False,
           "cold_train_artifact": os.path.basename(a.cold_train),
           "group_artifact": os.path.basename(a.group_tree), "cells": {}}

    # ---- CELL A: home advantage, all 10,000 validation users at cold states -------------------
    log("-" * 100)
    log(f"[A] HOME ADVANTAGE: {len(vrows)} validation users, COLD states (nothing volunteered).")
    log("[A] cold_val was fitted HERE; cold_train was not. Same recipe, same N, same pool.")
    ops0 = [[] for _ in vrows]
    scA = {}
    for arm, nd in (("cold_val", cold_val), ("cold_train", cold_train)):
        seqs, dg = route(E, vrows, nd, TOTAL_L, static=static)
        if dg["n_fallback"]:
            log(f"[A] {arm}: {dg['n_fallback']}/{len(vrows)} continuation-rule fires {dg['sites']}")
        scA[arm] = score(E, vrows, seqs, ops0)
    scA["static"] = score(E, vrows, [static[:TOTAL_L]] * len(vrows), ops0)
    armsA = ["cold_val", "cold_train", "static"]
    recA = table(scA, armsA)
    out["cells"]["A_home_advantage"] = {"n": int(len(vrows)), "budgets": recA}
    for K in ("1", "4", "15"):
        r = recA[K]
        lo, hi = r["ci_cold_val_minus_cold_train"]
        log(f"[A] K={K:>2} cold_val {r['cold_val']:.5f} cold_train {r['cold_train']:.5f} "
            f"static {r['static']:.5f} | HOME ADVANTAGE {r['cold_val_minus_cold_train']:+.5f} "
            f"[{lo:+.5f},{hi:+.5f}]{sig(recA, 'cold_val', 'cold_train', K)}")
    json.dump(out, open(os.path.join(M6, "m6r_fair_compare.json"), "w"), indent=1)

    # ---- CELL B: the fair comparison, arthouse cohort at opener states -------------------------
    cid = ANCHORS[a.cluster]
    V, F = np.asarray(E["V"]), np.asarray(E["foldable"])
    rows = vrows[((F > 0) & (V > 0))[vrows, cid]]
    zf = np.load(os.path.join(B, "m1", "z_full.npz"))
    tab = centroid_lift(E, rows, zf["z"].astype(np.float64), zf["rows"])
    log("-" * 100)
    log(f"[B] FAIR COMPARISON: {len(rows)} '{a.cluster}' validation users at opener states.")
    armsB = ["group", "cold_train", "cold_val", "anchor", "static"]
    for half, label in ((None, "full_pool"), ("A", "half_A"), ("B", "half_B")):
        ops, meta = make_openers(E, rows, tab, cname, LIKE_MIN_LEVEL, seed=SEED,
                                 half=half, group_key=a.cluster)
        umask = {int(r): list(o) for r, o in zip(rows, ops)}
        sc = {}
        for arm, nd in (("group", group), ("cold_train", cold_train),
                        ("cold_val", cold_val), ("anchor", anchor)):
            seqs, dg = route(E, rows, nd, TOTAL_L, static=static, mask=umask, on_conflict="burn")
            if dg["n_fallback"]:
                log(f"[B/{label}] {arm}: {dg['n_fallback']}/{len(rows)} continuation-rule fires")
            sc[arm] = score(E, rows, seqs, ops)
        sc["static"] = score(E, rows, [static[:TOTAL_L]] * len(rows), ops)
        rec = table(sc, armsB)
        out["cells"][f"B_{label}"] = {"n": int(len(rows)), "pool_size": int(meta["pool_size"]),
                                      "budgets": rec}
        r = rec["15"]
        log(f"[B/{label}] K=15 group {r['group']:.5f} cold_train {r['cold_train']:.5f} "
            f"cold_val {r['cold_val']:.5f} static {r['static']:.5f}")
        for opp in ("cold_train", "cold_val", "anchor", "static"):
            lo, hi = r[f"ci_group_minus_{opp}"]
            log(f"[B/{label}]    group - {opp:<11} {r[f'group_minus_{opp}']:+.5f} "
                f"[{lo:+.5f},{hi:+.5f}]{sig(rec, 'group', opp)}")
        json.dump(out, open(os.path.join(M6, "m6r_fair_compare.json"), "w"), indent=1)

    log("=" * 100)
    ha = recA["15"]["cold_val_minus_cold_train"]
    fp = out["cells"]["B_full_pool"]["budgets"]["15"]
    log(f"[VERDICT] measured home advantage (cell A, n={len(vrows)}): {ha:+.5f} "
        f"(previously INFERRED indirectly as +0.00782)")
    log(f"[VERDICT] group vs the FAIR comparator   : {fp['group_minus_cold_train']:+.5f} "
        f"[{fp['ci_group_minus_cold_train'][0]:+.5f},{fp['ci_group_minus_cold_train'][1]:+.5f}]")
    log(f"[VERDICT] group vs the at-home comparator: {fp['group_minus_cold_val']:+.5f} "
        f"(what we have been quoting)")
    log("[done]")


if __name__ == "__main__":
    main()
