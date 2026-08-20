r"""M6n -- DOES MORE FITTING DATA CLOSE THE OVERFITTING GAP?

The question this run exists for. `arthouse` fitted on 587 validation users scored **+0.01926**
against the cold tree on those same users and **+0.00733** on held-out train users -- a gap of
**0.0119**, which is what "selection overfitting" looks like when 15 picks are chosen from 1,000
candidates on a small cohort.

This evaluates the tree refitted on **2,000 train users** (2.5x the data, and deep enough to reach
depth 2 -- the first of these trees that could) in BOTH conditions:

  IN-SAMPLE   the 2,000 users it was fitted on (`fit_rows`, banked in the artifact)
  HELD-OUT    arthouse users from a DISJOINT train slice, never seen at fitting

If the in-sample/held-out gap shrinks materially below 0.0119, selection overfitting is confirmed as
the mechanism and the remaining five groups are worth the ~60 h. If the gap holds, more data is not
the lever and the pre-declared stop rule stands.

FAIRNESS. The anchor teacher, the cold tree and the static order are all VALIDATION-fitted, so they
are out of sample on every train user in both conditions. Only the group tree changes condition,
which is exactly the contrast being measured.

⚠ ruler = train_memorised: the recommender was trained on these users, so LEVELS are inflated and
must not be quoted beside validation or test numbers. The CONTRAST is the result.

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
    ap = argparse.ArgumentParser(description="M6n -- in-sample vs held-out for a train-fitted tree")
    ap.add_argument("--artifact", default=os.path.join(M6, "m6k_arthouse_train2k.json"))
    ap.add_argument("--cluster", default="arthouse")
    ap.add_argument("--held_lo", type=int, default=40000)
    ap.add_argument("--held_hi", type=int, default=70000)
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--i_have_author_goahead", action="store_true")
    a = ap.parse_args()
    assert a.i_have_author_goahead, "REFUSING TO RUN without --i_have_author_goahead"
    torch.set_num_threads(a.threads)
    np.random.seed(SEED)

    from strategy_channel_suite import (build_greedy_env, genv_eval_multi, CERT_TOWER, CERT_CONC)
    from ndcg_tree import build_pool, LIKE_MIN_LEVEL
    from tree_route import route
    from m4b_eval import load_tree
    from m3_show_paths import build_names
    from opener_gen import centroid_lift, make_openers
    cname, _iname = build_names()

    fit = json.load(open(a.artifact, encoding="utf-8"))
    nodes = fit["nodes"]
    if isinstance(nodes, list):
        nodes = {r["nid"]: r for r in nodes}
    fit_rows = set(int(r) for r in fit["fit_rows"])
    slice_hi = int(fit.get("slice_hi", 34000))
    cid = ANCHORS[a.cluster]
    log("#" * 100)
    log(f"# M6n -- {a.cluster}: fitted on {len(fit_rows)} TRAIN users, depth-2, {len(nodes)} nodes")
    log(f"# ruler = train_memorised. LEVELS INFLATED; the in-sample/held-out CONTRAST is the result.")
    log("#" * 100)

    _cd, cold_nodes = load_tree(os.path.join(B, "coldtree_v2.json"))
    _ad, anch_nodes = load_tree(os.path.join(B, f"m3v2_{a.cluster}.json"))
    static = [(int(x), int(y)) for x, y in
              json.load(open(os.path.join(B, "greedy_static_policyB_combined.json")))["sequence"]]
    zf = np.load(os.path.join(B, "m1", "z_full.npz"))
    ZF, ZR = zf["z"].astype(np.float64), zf["rows"]
    # the group centroid comes from VALIDATION, as it did at fitting -- a group-level object
    Ev = build_greedy_env(CERT_TOWER, "validation", CERT_CONC, "m6n_cent")
    vrows = np.asarray(Ev["rows"], np.int64)
    Vv, Fv = np.asarray(Ev["V"]), np.asarray(Ev["foldable"])
    valcoh = vrows[((Fv > 0) & (Vv > 0))[vrows, cid]]
    del Ev

    out = {"analysis": "m6n_train2k", "ruler": "train_memorised", "test_touched": False,
           "cluster": a.cluster, "n_fit": len(fit_rows), "conditions": {}}

    for label, lo, hi in (("in_sample", 0, slice_hi), ("held_out", a.held_lo, a.held_hi)):
        E = build_greedy_env(CERT_TOWER, "train", CERT_CONC, f"m6n_{label}", cohort_slice=(lo, hi))
        pool, _i, _c = build_pool(E, 500, 500)
        V, F = np.asarray(E["V"]), np.asarray(E["foldable"])
        endorse = (F > 0) & (V > 0)
        rows_loc = np.arange(E["ctx"].n, dtype=np.int64)
        rows = rows_loc[endorse[rows_loc, cid]]
        if label == "in_sample":
            rows = np.asarray([r for r in rows if int(r) in fit_rows], np.int64)
        else:
            assert lo >= slice_hi, "held-out slice must not overlap the fitting pool"
        log(f"[{label}] {len(rows)} users (slice {lo}:{hi})")
        tab = centroid_lift(E, valcoh, ZF, ZR)
        ops, _m = make_openers(E, rows, tab, cname, LIKE_MIN_LEVEL, seed=SEED,
                               group_key=a.cluster)
        umask = {int(r): list(o) for r, o in zip(rows, ops)}
        seqs = {}
        for arm, nd in (("tree", nodes), ("anchor", anch_nodes), ("cold", cold_nodes)):
            seqs[arm], _dg = route(E, rows, nd, TOTAL_L, static=static, mask=umask,
                                   on_conflict="burn")
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
        rec = {"n": int(len(rows)), "budgets": {}}
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
        log(f"[{label}] K=15 tree {c['tree']:.5f} anchor {c['anchor']:.5f} cold {c['cold']:.5f} "
            f"static {c['static']:.5f} | vs cold {c['tree_minus_cold']:+.5f}{st('cold')} "
            f"vs anchor {c['tree_minus_anchor']:+.5f}{st('anchor')} "
            f"vs static {c['tree_minus_static']:+.5f}{st('static')}")
        del E

    ins = out["conditions"]["in_sample"]["budgets"]["15"]["tree_minus_cold"]
    hel = out["conditions"]["held_out"]["budgets"]["15"]["tree_minus_cold"]
    out["overfit_gap"] = {"n_fit": len(fit_rows), "in_sample": ins, "held_out": hel,
                          "gap": ins - hel, "reference_gap_at_587_users": 0.01193}
    log("=" * 100)
    log(f"[GAP] fitted on {len(fit_rows)} users: in-sample {ins:+.5f} · held-out {hel:+.5f} "
        f"· gap {ins - hel:+.5f}")
    log(f"[GAP] reference, fitted on 587 validation users: gap +0.01193 "
        f"(+0.01926 in-sample -> +0.00733 held-out)")
    json.dump(out, open(os.path.join(M6, f"m6n_{a.cluster}_train2k.json"), "w"), indent=1)
    log("[done]")


if __name__ == "__main__":
    main()
