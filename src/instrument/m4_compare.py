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

⚠ IN-SAMPLE FOR BOTH TREES: the cold tree and every teacher were grown on these same validation
users. This measures whether the teacher's extra conditioning helps AT ALL before spending the test
look; it does not establish generalisation. Both arms are equally in-sample, so the COMPARISON is
fair even though the levels are optimistic.
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


def route(E, rows, nodes, seed_prefix, total_L):
    """Per-user question sequence produced by descending `nodes` from the root, branching on each
    user's OWN answer. Returns a list of per-user ASKED question lists (the free seed excluded)."""
    from ndcg_tree import branch_of
    seq = {int(r): [] for r in rows}
    def rec(nid, rs):
        if not rs:
            return
        rec_ = nodes.get(nid)
        assert rec_ is not None, f"node {nid} missing"
        steps = [(int(a), int(b)) for a, b in rec_["steps"]]
        if not rec_.get("split"):
            for r in rs:
                seq[int(r)].extend(steps)
            return
        q = (int(rec_["splitter"][0]), int(rec_["splitter"][1]))
        for r in rs:
            seq[int(r)].append(q)
        lab, _ar, _nz = branch_of(E, rs, q)
        kids = {"like": [], "dislike": [], "noanswer": []}
        for r, l in zip(rs, lab):
            kids[str(l)].append(int(r))
        for br in ("like", "dislike", "noanswer"):
            if kids[br]:
                child = f"{nid}.{br}"
                if child in nodes:
                    rec(child, kids[br])
                else:                       # below-floor child: inherits this node's tail
                    tail = [(int(a), int(b)) for a, b in rec_.get("tail", rec_["steps"])]
                    for r in kids[br]:
                        seq[int(r)].extend(tail[1:])   # the splitter is already appended
    rec("root", [int(r) for r in rows])
    out = []
    for r in rows:
        s = seq[int(r)]
        assert len(s) >= total_L, f"user {r} got {len(s)} questions, need {total_L}"
        out.append(s[:total_L])
    return out


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

        seq_cold = route(E, cohort, cold_nodes, [(1, cid)], a.total_L)
        seq_teach = route(E, cohort, t_nodes, [(1, cid)], a.total_L)

        row = {"anchor": name, "concept": cid, "cohort": int(len(cohort)),
               "teacher_depth": t_d.get("depth_reached"), "budgets": {}}
        for K in BUDGETS:
            def ev(seqs):
                iseq, cseq = [], []
                for s in seqs:
                    ask = s[:K]
                    it = [q[1] for q in ask if q[0] == 0]
                    cc = [cid] + [q[1] for q in ask if q[0] == 1]   # anchor folded in EVERY arm
                    iseq.append(it); cseq.append(list(dict.fromkeys(cc)))
                full, tail = genv_eval_multi(E, cohort, iseq, cseq)
                return np.asarray(full, np.float64), np.asarray(tail, np.float64)

            cf, ct = ev(seq_cold)
            tf, tt = ev(seq_teach)
            sf, st = ev([static] * len(cohort))
            mf, lof, hif = paired_boot(tf - cf)
            mt, lot, hit = paired_boot(tt - ct)
            row["budgets"][str(K)] = {
                "cold_full": float(cf.mean()), "teacher_full": float(tf.mean()),
                "static_full": float(sf.mean()),
                "cold_tail": float(ct.mean()), "teacher_tail": float(tt.mean()),
                "teacher_minus_cold_full": mf, "ci_full": [lof, hif],
                "teacher_minus_cold_tail": mt, "ci_tail": [lot, hit]}
            star = "*" if lof > 0 else (" " if hif > 0 else "-")
            log(f"   [{name}] K={K:<3} cold {cf.mean():.5f} teacher {tf.mean():.5f} "
                f"static {sf.mean():.5f} | teacher-cold {mf:+.5f} [{lof:+.5f},{hif:+.5f}] {star}"
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
