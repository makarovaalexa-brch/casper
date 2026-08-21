r"""M6q -- BENCH the per-evaluation cost at the cohort sizes we are considering.

Author's question, 2026-08-21: *"but it will run for eternity?"*

Project rule: **cost estimates on this box are not properties of a method** -- past projections were
off by 4x high and 2.6x low. So this measures the ASSEMBLED path (real env, real pool, real
`mean_score` over per-user opener prefixes) instead of extrapolating from the 2,000-user run.

Calibration point already measured: `arthouse` at 2,000 train users ran the root seed pass at
**1.88 s/eval**; the cold tree recorded `baseline_sec_per_eval_10k = 13.543` at 10,000 VALIDATION
users and finished 164 nodes in 6.9 h.

Reports, for each cohort size, the measured s/eval and the implied cost of ONE 1,000-candidate seed
pass -- which after the heap-reuse fix is the only full sweep a node pays.

VALIDATION + TRAIN only. **THE TEST SET IS NEVER BUILT, LOADED OR NAMED IN THIS FILE.**
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
ANCHORS = {"arthouse": 68, "dialogue": 275, "marvel": 581,
           "gore": 417, "teenagers": 922, "inspiring": 503}


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def main():
    ap = argparse.ArgumentParser(description="M6q -- measured per-eval cost by cohort size")
    ap.add_argument("--n_evals", type=int, default=20)
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--i_have_author_goahead", action="store_true")
    a = ap.parse_args()
    assert a.i_have_author_goahead, "REFUSING TO RUN without --i_have_author_goahead"
    torch.set_num_threads(a.threads)
    np.random.seed(SEED)

    from strategy_channel_suite import (build_greedy_env, CERT_TOWER, CERT_CONC)
    from ndcg_tree import build_pool, LIKE_MIN_LEVEL
    from group_greedy_bands import mean_score
    from m3_show_paths import build_names
    from opener_gen import centroid_lift, make_openers
    cname, _iname = build_names()

    zf = np.load(os.path.join(B, "m1", "z_full.npz"))
    ZF, ZR = zf["z"].astype(np.float64), zf["rows"]

    log("#" * 100)
    log("# M6q -- MEASURED per-eval cost (assembled path). Calibration: arthouse@2000 = 1.88 s/eval")
    log("#" * 100)

    # the arthouse validation cohort, for the group centroid (a property of the GROUP)
    Ev = build_greedy_env(CERT_TOWER, "validation", CERT_CONC, "m6q_cent")
    vr = np.asarray(Ev["rows"], np.int64)
    Vv, Fv = np.asarray(Ev["V"]), np.asarray(Ev["foldable"])
    valcoh = vr[((Fv > 0) & (Vv > 0))[vr, ANCHORS["arthouse"]]]
    del Ev
    log(f"[centroid] arthouse validation cohort {len(valcoh)} users")

    E = build_greedy_env(CERT_TOWER, "train", CERT_CONC, "m6q", cohort_slice=(0, 34000))
    pool, _i, _c = build_pool(E, 500, 500)
    tab = centroid_lift(E, valcoh, ZF, ZR)
    V, F = np.asarray(E["V"]), np.asarray(E["foldable"])
    endorse = (F > 0) & (V > 0)
    rows_all = np.arange(E["ctx"].n, dtype=np.int64)
    art = rows_all[endorse[rows_all, ANCHORS["arthouse"]]]
    log(f"[cohort] train slice 0..34000: {len(rows_all)} users, {len(art)} arthouse endorsers")

    out = {"analysis": "m6q_cost_bench", "test_touched": False, "n_evals": a.n_evals, "cells": {}}
    rng = np.random.default_rng(SEED)

    cells = [("cold_on_train_10k", rows_all[:10000], False),
             ("arthouse_all_train", art, True),
             ("arthouse_2k_calibration", art[:2000], True)]
    for name, rows, with_openers in cells:
        if len(rows) == 0:
            continue
        if with_openers:
            ops, _m = make_openers(E, rows, tab, cname, LIKE_MIN_LEVEL, seed=SEED,
                                   group_key="arthouse")
            seeds = [[tuple(q) for q in o] for o in ops]
        else:
            seeds = [[] for _ in rows]      # the cold tree's root state: nothing volunteered
        qs = [tuple(pool[i]) for i in rng.integers(0, len(pool), a.n_evals)]
        t0 = time.time()
        for q in qs:
            mean_score(E, rows, seeds, [q], "full")
        sec = (time.time() - t0) / a.n_evals
        out["cells"][name] = {"n_users": int(len(rows)), "sec_per_eval": round(sec, 3),
                              "seed_pass_hours": round(sec * 1000 / 3600, 2),
                              "openers": with_openers}
        log(f"[{name:<24}] n={len(rows):>6}  {sec:6.2f} s/eval  "
            f"-> one 1,000-candidate seed pass = {sec * 1000 / 3600:5.2f} h")

    json.dump(out, open(os.path.join(M6, "m6q_cost_bench.json"), "w"), indent=1)
    log("=" * 100)
    log("[reference] cold tree: 164 nodes / 6.9 h wall at 10,000 VALIDATION users")
    log("[note] only the ROOT pays the full-cohort cost; nodes shrink fast as the tree partitions")
    log("[done]")


if __name__ == "__main__":
    main()
