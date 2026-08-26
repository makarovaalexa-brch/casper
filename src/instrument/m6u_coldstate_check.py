r"""M6u -- COLD STATES: does the train-fitted tree beat the train-fitted static order?

The missing cell. Everything in `m6t` compares against `static_train`, but the cold-state comparison
on record (`m6r` cell A) used the VALIDATION-fitted static order, which is at home there. This closes
that so the chapter can quote ONE static comparator throughout instead of explaining two.

  FAIR (both fitted on TRAIN, neither has seen validation)
    static_train   greedy_static_policyB_combined_train10000
    cold_train     coldtree_v2_train10000

  REFERENCE ONLY (validation-fitted -- AT HOME here, not a fair contrast)
    static_val, cold_val

All 10,000 validation users, COLD states: nothing volunteered, no opener, no mask.

Already established elsewhere, for context rather than repetition:
  * ON TEST, K=16: NDCG tree 0.26936 vs rebuilt policy-B static 0.25900
    = +0.01035 [+0.00749,+0.01328] full / +0.00705 tail (`CONDITIONED_ARMS_ON_TEST.md`).
  * m6r cell A, K=15: cold_train 0.26156 vs static_VAL 0.25415 = +0.00741, i.e. already positive
    against an at-home comparator.

K=1 is identically zero by construction: a tree's root splitter is the global order's first
question, so one question cannot separate them. That is a harness check, not a result.

**THE TEST SET IS NEVER BUILT, LOADED OR NAMED IN THIS FILE.**
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
M6 = os.path.join(B, "m6")
SEED = 98765
TOTAL_L = 15
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
    ap = argparse.ArgumentParser(description="M6u -- cold-state check, train-fitted arms")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--i_have_author_goahead", action="store_true")
    a = ap.parse_args()
    assert a.i_have_author_goahead, "REFUSING TO RUN without --i_have_author_goahead"
    torch.set_num_threads(a.threads)
    np.random.seed(SEED)

    from strategy_channel_suite import (build_greedy_env, genv_eval_multi, CERT_TOWER, CERT_CONC)
    from ndcg_tree import build_pool
    from tree_route import route
    from m4b_eval import load_tree
    from run_guard import require_mains
    require_mains("m6u cold-state check", log=log)

    _ct, cold_train = load_tree(os.path.join(B, "coldtree_v2_train10000.json"))
    _cv, cold_val = load_tree(os.path.join(B, "coldtree_v2.json"))
    st_tr = [(int(x), int(y)) for x, y in json.load(
        open(os.path.join(B, "greedy_static_policyB_combined_train10000.json")))["sequence"]]
    st_va = [(int(x), int(y)) for x, y in json.load(
        open(os.path.join(B, "greedy_static_policyB_combined.json")))["sequence"]][:TOTAL_L]
    log("#" * 100)
    log(f"# M6u -- COLD STATES. cold_train {len(cold_train)} nodes vs static_train {len(st_tr)} picks")
    log("# reference arms are VALIDATION-fitted and AT HOME -- not a fair contrast")
    log("#" * 100)

    E = build_greedy_env(CERT_TOWER, "validation", CERT_CONC, "m6u")
    pool, _i, _c = build_pool(E, 500, 500)
    rows = np.asarray(E["rows"], np.int64)
    log(f"[cohort] {len(rows):,} validation users, COLD states (nothing volunteered)")
    ops = [[] for _ in rows]

    seqs = {}
    for arm, nd in (("cold_train", cold_train), ("cold_val", cold_val)):
        seqs[arm], dg = route(E, rows, nd, TOTAL_L, static=st_tr)
        if dg["n_fallback"]:
            log(f"[{arm}] {dg['n_fallback']}/{len(rows)} continuation-rule fires {dg['sites']}")
    seqs["static_train"] = [st_tr[:TOTAL_L]] * len(rows)
    seqs["static_val"] = [st_va[:TOTAL_L]] * len(rows)

    sc = {}
    for arm in seqs:
        sc[arm] = {}
        for K in BUDGETS:
            iseq, cseq = [], []
            for s, op in zip(seqs[arm], ops):
                ask = s[:K]
                iseq.append([q[1] for q in ask if q[0] == 0])
                cseq.append([q[1] for q in ask if q[0] == 1])
            full, _t = genv_eval_multi(E, rows, iseq, cseq)
            sc[arm][K] = np.asarray(full, np.float64)
        log(f"[scored] {arm}")

    out = {"analysis": "m6u_coldstate_check", "ruler": "validation_clean", "test_touched": False,
           "n": int(len(rows)), "fair_arms": ["static_train", "cold_train"],
           "reference_arms": ["static_val", "cold_val"], "budgets": {}}
    log("=" * 100)
    log(f"{'K':>3} {'static_tr':>10} {'cold_tr':>9} | {'cold-static (FAIR)':>19} {'95% CI':>22} {'':>3}"
        f" | {'static_val':>10} {'cold_val':>9}")
    for K in BUDGETS:
        row = {arm: float(np.nanmean(sc[arm][K])) for arm in sc}
        m, lo, hi = paired_boot(sc["cold_train"][K] - sc["static_train"][K])
        row["cold_train_minus_static_train"] = m
        row["ci"] = [lo, hi]
        mv, lv, hv = paired_boot(sc["cold_val"][K] - sc["static_val"][K])
        row["cold_val_minus_static_val_REFERENCE"] = mv
        row["ci_reference"] = [lv, hv]
        out["budgets"][str(K)] = row
        s = "*" if lo > 0 else ("-" if hi < 0 else " ")
        log(f"{K:>3} {row['static_train']:10.5f} {row['cold_train']:9.5f} | {m:>+19.5f} "
            f"[{lo:+.5f},{hi:+.5f}] {s:>3} | {row['static_val']:10.5f} {row['cold_val']:9.5f}")
    json.dump(out, open(os.path.join(M6, "m6u_coldstate_check.json"), "w"), indent=1)
    b = out["budgets"]["15"]
    log("=" * 100)
    log(f"[VERDICT] K=15 cold_train - static_train = {b['cold_train_minus_static_train']:+.5f} "
        f"[{b['ci'][0]:+.5f},{b['ci'][1]:+.5f}]")
    log(f"[context] on TEST, both validation-fitted: +0.01035 [+0.00749,+0.01328] at K=16")
    log(f"[context] m6r cell A, cold_train vs static_VAL (at home): +0.00741")
    log("[done]")


if __name__ == "__main__":
    main()
