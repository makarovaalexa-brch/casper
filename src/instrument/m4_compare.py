r"""M4a -- COLD TREE vs DEDICATED TEACHER at opener states. VALIDATION ONLY.

The question the forest rests on: at the states where a user has volunteered an anchor, does that
anchor's dedicated teacher beat the incumbent cold tree -- with the anchor's own advantage
NEUTRALISED by folding it into BOTH arms?

This is M5's per-anchor cell run on VALIDATION, so the single TEST look stays unspent.

ARMS (per anchor cohort = validation users who ENDORSE the anchor under the env's SEL rule):
  cold      anchor folded, then the COLD TREE's interview, routed by the user's own answers.
  teacher   anchor folded, then that anchor's TEACHER, routed by the user's own answers.
  static    anchor folded, then the policy-B combined static order (context, not the headline).
The anchor is free in every arm and masked from nobody -- it is folded evidence, not a question.
Budgets K in {1,2,4,8,15} count ASKED questions.

Paired percentile bootstrap over per-user differences, 10,000 resamples, seed 98765.

IN-SAMPLE, AND NOT SYMMETRICALLY SO (author, 2026-08-19): each teacher was fitted on precisely the
cohort it is scored on, while the cold tree was fitted on all 10,000 and merely covers it. The
specialised arm has an advantage the incumbent does not, so a positive teacher-minus-cold is an UPPER
BOUND on what conditioning is worth, not an estimate of it. Held-out confirmation is the M5 test look.

RE-RUN 2026-08-19 after a ROUTING DEFECT: the previous run reconstructed each user's interview by
accumulating splitters during the descent and then appending the leaf's `steps`. For a below-floor
child that re-supplies the parent's splitter, so 18.5-45.3% of users were asked one question twice
and received 14 distinct questions on a budget of 15. Routing now uses the tree's own
`path + steps` (`tree_route`), verified to give 15.00 distinct questions for every tree.
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
OUT = os.path.join(B, "m4")
SEED = 98765
VAL_M0_FULL = 0.15686538118354684
ANCHORS = {"arthouse": 68, "dialogue": 275, "marvel": 581,
           "gore": 417, "teenagers": 922, "inspiring": 503}
BUDGETS = [1, 2, 4, 8, 15]


def log(m: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def paired_boot(d, n=10000, seed=SEED):
    # tail NDCG is NaN for users with no tail target; drop those PAIRS (never impute)
    d = np.asarray(d, np.float64)
    d = d[np.isfinite(d)]
    if len(d) == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(n, len(d)))
    bs = d[idx].mean(1)
    return float(d.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def load_tree(path):
    d = json.load(open(path, encoding="utf-8"))
    nodes = d.get("nodes")
    if isinstance(nodes, list):
        nodes = {r["nid"]: r for r in nodes}
    if not nodes:
        part = path.replace(".json", "_partial.json")
        nodes = json.load(open(part, encoding="utf-8"))["nodes"]
    return d, nodes


def main() -> None:
    ap = argparse.ArgumentParser(description="M4a -- cold tree vs dedicated teacher at opener states")
    ap.add_argument("--total_L", type=int, default=15)
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--i_have_author_goahead", action="store_true")
    a = ap.parse_args()
    assert a.i_have_author_goahead, "REFUSING TO RUN without --i_have_author_goahead"
    torch.set_num_threads(a.threads); np.random.seed(SEED)
    os.makedirs(OUT, exist_ok=True)

    from strategy_channel_suite import (build_greedy_env, contract_selftest, genv_mean,
                                        genv_eval_multi, CERT_TOWER, CERT_CONC)
    from ndcg_tree import leak_assert
    from tree_route import route

    log("#" * 100)
    log("# M4a -- COLD TREE vs DEDICATED TEACHER at opener states. VALIDATION ONLY.")
    log("#" * 100)

    E = build_greedy_env(CERT_TOWER, "validation", CERT_CONC, "m4a")
    contract_selftest(E, E["rows"]); leak_assert(E, E["rows"], "m4a/val")
    rows_all = np.asarray(E["rows"], np.int64)
    f0, _ = genv_mean(E, [], rows_all)
    assert round(f0, 8) == round(VAL_M0_FULL, 8), f"CANONICAL SNAP FAILED: {f0!r}"
    log(f"[gate] canonical m=0 snap full={f0!r} -- exact")

    static = [(int(x), int(y)) for x, y in
              json.load(open(os.path.join(B, "greedy_static_policyB_combined.json")))["sequence"]]
    cold_d, cold_nodes = load_tree(os.path.join(B, "coldtree_v2.json"))
    log(f"[trees] cold tree depth {cold_d.get('depth_reached')}, {cold_d.get('n_nodes')} nodes")

    V, F = np.asarray(E["V"]), np.asarray(E["foldable"])
    endorse = (F > 0) & (V > 0)
    results = []

    for name, cid in ANCHORS.items():
        tpath = os.path.join(B, f"m3v2_{name}.json")
        if not os.path.exists(tpath):
            log(f"[skip] {name}: no teacher artifact")
            continue
        t_d, t_nodes = load_tree(tpath)
        cohort = rows_all[endorse[rows_all, cid]]
        log(f"[{name}] cohort {len(cohort)} · teacher depth {t_d.get('depth_reached')}, "
            f"{t_d.get('n_nodes')} nodes")

        seq_cold, dg_c = route(E, cohort, cold_nodes, a.total_L, static=static, mask=[(1, cid)])
        seq_teach, dg_t = route(E, cohort, t_nodes, a.total_L, static=static, mask=[(1, cid)])

        row = {"anchor": name, "concept": cid, "cohort": int(len(cohort)),
               "teacher_depth": t_d.get("depth_reached"),
               "fallback_cold": dg_c["n_fallback"], "fallback_teacher": dg_t["n_fallback"],
               "budgets": {}}
        for K in BUDGETS:
            def ev(seqs):
                iseq, cseq = [], []
                for s in seqs:
                    ask = s[:K]
                    it = [q[1] for q in ask if q[0] == 0]
                    cc = [cid] + [q[1] for q in ask if q[0] == 1]   # anchor folded in EVERY arm
                    iseq.append(list(dict.fromkeys(it)))     # a question can recur across a path
                    cseq.append(list(dict.fromkeys(cc)))
                full, tail = genv_eval_multi(E, cohort, iseq, cseq)
                return np.asarray(full, np.float64), np.asarray(tail, np.float64)

            cf, ct = ev(seq_cold)
            tf, tt = ev(seq_teach)
            sf, st = ev([static] * len(cohort))
            mf, lof, hif = paired_boot(tf - cf)
            mt, lot, hit = paired_boot(tt - ct)
            row["budgets"][str(K)] = {
                "cold_full": float(np.nanmean(cf)), "teacher_full": float(np.nanmean(tf)),
                "static_full": float(np.nanmean(sf)),
                "cold_tail": float(np.nanmean(ct)), "teacher_tail": float(np.nanmean(tt)),
                "teacher_minus_cold_full": mf, "ci_full": [lof, hif],
                "teacher_minus_cold_tail": mt, "ci_tail": [lot, hit]}
            star = "*" if lof > 0 else (" " if hif > 0 else "-")
            log(f"   [{name}] K={K:<3} cold {np.nanmean(cf):.5f} teacher {np.nanmean(tf):.5f} "
                f"static {np.nanmean(sf):.5f} | teacher-cold {mf:+.5f} [{lof:+.5f},{hif:+.5f}] {star}"
                f" | tail {mt:+.5f}")
        results.append(row)
        json.dump({"analysis": "m4a_cold_vs_teacher", "split": "validation",
                   "test_touched": False, "results": results},
                  open(os.path.join(OUT, "m4a_compare.json"), "w"), indent=1)

    log("=" * 100)
    log("[summary] teacher minus cold tree, anchor folded in BOTH arms (validation, in-sample)")
    for r in results:
        c = r["budgets"]["15"]
        log(f"  {r['anchor']:<11} n={r['cohort']:<5} d={r['teacher_depth']} "
            f"K=15 {c['teacher_minus_cold_full']:+.5f} "
            f"[{c['ci_full'][0]:+.5f},{c['ci_full'][1]:+.5f}] full · "
            f"{c['teacher_minus_cold_tail']:+.5f} tail")
    log("[done]")


if __name__ == "__main__":
    main()
