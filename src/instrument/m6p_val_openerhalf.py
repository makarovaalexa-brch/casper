r"""M6p -- IS IT THE POPULATION OR THE OPENER HALF?

Two numbers disagree about the same train-fitted `arthouse` tree:

  m6n  TRAIN held-out users, FULL-pool openers   tree-cold  +0.01762  [+0.01090,+0.02436]
  m6d  VALIDATION users,     half-B openers      tree-cold  -0.00319  (CI includes zero)

They differ in TWO ways at once -- the population (train, whose levels the tower memorised, versus
clean validation) and the opener draw (the full film pool versus a disjoint 50% half). A single
number cannot be attributed while both move together.

This run holds the POPULATION fixed at validation and sweeps ONLY the opener half:

  full pool  (`half=None`)  -- exactly what the tree was FITTED on
  half A                    -- the half the distillation trajectories were generated from
  half B                    -- the generalisation half m6d used

Same users, same tree, same cold/anchor/static comparators, same scoring path. If `full` recovers
something near +0.017 then the opener half is the lever and m6d's null is an off-manifold cost. If
`full` also sits near zero then the train-held-out win was inflated by the memorised ruler and the
opener half is irrelevant.

⚠ The comparators (cold tree, anchor teacher, static order) are all VALIDATION-fitted, so on these
users they are IN sample while the group tree is OUT of sample. That handicap runs AGAINST the
tree, so a positive result here is conservative; a null is not automatically damning. Stated so the
asymmetry is visible rather than assumed away.

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


def main():
    ap = argparse.ArgumentParser(description="M6p -- opener half at fixed population")
    ap.add_argument("--artifact", default=os.path.join(M6, "m6k_arthouse_train2000.json"))
    ap.add_argument("--cluster", default="arthouse")
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--i_have_author_goahead", action="store_true")
    a = ap.parse_args()
    assert a.i_have_author_goahead, "REFUSING TO RUN without --i_have_author_goahead"
    torch.set_num_threads(a.threads)
    np.random.seed(SEED)

    from strategy_channel_suite import (build_greedy_env, genv_eval_multi, CERT_TOWER, CERT_CONC)
    from ndcg_tree import (build_pool, LIKE_MIN_LEVEL, branch_of_per_user, branch_of_selftest)
    from tree_route import route
    from m4b_eval import load_tree
    from m3_show_paths import build_names
    from opener_gen import centroid_lift, make_openers
    cname, _iname = build_names()

    fit = json.load(open(a.artifact, encoding="utf-8"))
    nodes = fit["nodes"]
    if isinstance(nodes, list):
        nodes = {r["nid"]: r for r in nodes}
    cid = ANCHORS[a.cluster]
    log("#" * 100)
    log(f"# M6p -- {a.cluster}: population FIXED at validation, opener half swept")
    log(f"# tree fitted on {len(fit['fit_rows'])} TRAIN users (split={fit.get('split')}), "
        f"{len(nodes)} nodes")
    log("#" * 100)

    _cd, cold_nodes = load_tree(os.path.join(B, "coldtree_v2.json"))
    _ad, anch_nodes = load_tree(os.path.join(B, f"m3v2_{a.cluster}.json"))
    static = [(int(x), int(y)) for x, y in
              json.load(open(os.path.join(B, "greedy_static_policyB_combined.json")))["sequence"]]
    zf = np.load(os.path.join(B, "m1", "z_full.npz"))
    ZF, ZR = zf["z"].astype(np.float64), zf["rows"]

    E = build_greedy_env(CERT_TOWER, "validation", CERT_CONC, "m6p")
    pool, _i, _c = build_pool(E, 500, 500)
    vrows = np.asarray(E["rows"], np.int64)
    V, F = np.asarray(E["V"]), np.asarray(E["foldable"])
    rows = vrows[((F > 0) & (V > 0))[vrows, cid]]
    log(f"[cohort] {len(rows)} validation users endorse '{a.cluster}'")
    tab = centroid_lift(E, rows, ZF, ZR)          # same population -- no cross-split index use

    out = {"analysis": "m6p_val_openerhalf", "ruler": "validation_clean", "test_touched": False,
           "cluster": a.cluster, "artifact": os.path.basename(a.artifact),
           "n": int(len(rows)), "comparators_in_sample": True, "conditions": {}}

    for half, label in ((None, "full_pool"), ("A", "half_A"), ("B", "half_B")):
        ops, meta = make_openers(E, rows, tab, cname, LIKE_MIN_LEVEL, seed=SEED,
                                 half=half, group_key=a.cluster)
        umask = {int(r): list(o) for r, o in zip(rows, ops)}
        seqs = {}
        for arm, nd in (("tree", nodes), ("anchor", anch_nodes), ("cold", cold_nodes)):
            seqs[arm], dg = route(E, rows, nd, TOTAL_L, static=static, mask=umask,
                                  on_conflict="burn")
            if dg["n_fallback"]:
                log(f"[{label}] {arm}: {dg['n_fallback']}/{len(rows)} continuation-rule fires")
        seqs["static"] = [static[:TOTAL_L]] * len(rows)
        sc = {}
        for arm in seqs:
            sc[arm] = {}
            for K in BUDGETS:
                iseq, cseq = [], []
                for s, op in zip(seqs[arm], ops):
                    ask = s[:K]
                    iseq.append(list(dict.fromkeys([i for c, i in op if c == 0] +
                                                   [q[1] for q in ask if q[0] == 0])))
                    cseq.append(list(dict.fromkeys([i for c, i in op if c == 1] +
                                                   [q[1] for q in ask if q[0] == 1])))
                full, _t = genv_eval_multi(E, rows, iseq, cseq)
                sc[arm][K] = np.asarray(full, np.float64)

        branch_of_selftest(E, rows, [tuple(seqs["tree"][0][0])])
        codes = np.stack([branch_of_per_user(E, rows, [tuple(x[t]) for x in seqs["tree"]])
                          for t in range(TOTAL_L)], 1)
        n_ans = (codes != 2).sum(1).astype(np.int16)
        np.savez_compressed(os.path.join(M6, f"m6p_{a.cluster}_{label}_peruser.npz"),
                            rows=rows, n_answered_tree=n_ans,
                            n_opener=np.asarray([len(o) for o in ops], np.int16),
                            **{f"{arm}_K{K}": sc[arm][K] for arm in sc for K in BUDGETS})

        rec = {"n": int(len(rows)), "pool_size": int(meta["pool_size"]),
               "film_rate": meta["n_film"] / max(meta["n"], 1),
               "unserved": int(meta["unserved"]),
               "answered_mean": float(n_ans.mean()), "budgets": {}}
        for K in BUDGETS:
            row = {arm: float(np.nanmean(sc[arm][K])) for arm in sc}
            for opp in ("cold", "anchor", "static"):
                m, l, h = paired_boot(sc["tree"][K] - sc[opp][K])
                row[f"tree_minus_{opp}"] = m
                row[f"ci_{opp}"] = [l, h]
            rec["budgets"][str(K)] = row
        out["conditions"][label] = rec
        c = rec["budgets"]["15"]
        st = lambda k: ("*" if c[f"ci_{k}"][0] > 0 else ("-" if c[f"ci_{k}"][1] < 0 else " "))
        log(f"[{label}] pool {meta['pool_size']} film {rec['film_rate']:.3f} "
            f"answered {rec['answered_mean']:.2f}/15 | K=15 tree {c['tree']:.5f} "
            f"cold {c['cold']:.5f} | vs cold {c['tree_minus_cold']:+.5f}{st('cold')} "
            f"[{c['ci_cold'][0]:+.5f},{c['ci_cold'][1]:+.5f}] "
            f"vs anchor {c['tree_minus_anchor']:+.5f}{st('anchor')} "
            f"vs static {c['tree_minus_static']:+.5f}{st('static')}")
        json.dump(out, open(os.path.join(M6, f"m6p_{a.cluster}_openerhalf.json"), "w"), indent=1)

    log("=" * 100)
    log("[reference] m6n TRAIN held-out, FULL pool: tree-cold +0.01762 [+0.01090,+0.02436]")
    fp = out["conditions"]["full_pool"]["budgets"]["15"]["tree_minus_cold"]
    hb = out["conditions"]["half_B"]["budgets"]["15"]["tree_minus_cold"]
    log(f"[verdict] validation full-pool {fp:+.5f} vs validation half-B {hb:+.5f} "
        f"-> opener-half effect {fp - hb:+.5f}")
    log(f"[verdict] validation full-pool {fp:+.5f} vs TRAIN full-pool +0.01762 "
        f"-> population effect {fp - 0.01762:+.5f}")
    log("[done]")


if __name__ == "__main__":
    main()
