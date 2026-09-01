r"""M6v -- WHERE DO THE ROUTING MISTAKES GO?

`m6d` records only agreement with the CORRECT tree, so a cohort routing at 0.404 tells us it is
wrong 60% of the time and nothing about what it is wrong *toward*. That distinction decides the fix:

  * mistakes concentrated on ONE cohort  -> class imbalance / a default attractor. `inspiring` holds
    24,372 of 75,270 grouped training users (32%), so it is the obvious suspect. Fix = rebalance the
    distillation loss or subsample the bank. Cheap.
  * mistakes spread across cohorts       -> the folded opener genuinely does not separate them.
    Fix = a better state, not a better loss. Expensive.
  * mistakes on COLD                     -> the policy is not recognising a warm state at all.

Every tree has a DISTINCT root splitter (verified: no two of the seven share one), so the policy's
first question identifies which tree it is imitating, with no ambiguity to resolve.

This measures ONLY the first question. Routing is not value -- a policy can route perfectly and earn
nothing, which by this project's rules is not evidence. This is a DIAGNOSTIC for choosing the next
experiment, never a result.

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
CONCEPTS = {"arthouse": 68, "dialogue": 275, "marvel": 581,
            "gore": 417, "teenagers": 922, "inspiring": 503}


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def main():
    ap = argparse.ArgumentParser(description="M6v -- routing confusion matrix")
    ap.add_argument("--ckpt", default=os.path.join(B, "m4", "m6forest_policy.pt"))
    ap.add_argument("--cold_tree", default="coldtree_v2_train10000.json")
    ap.add_argument("--tree_suffix", default="_train2000")
    ap.add_argument("--disjoint", default="",
                    help="m7b_disjoint.npz: banked disjoint groups + exclusive opener "
                         "material. MUST be passed when scoring a policy distilled from "
                         "the m6k_dj* teachers.")
    ap.add_argument("--half", default="B", choices=["A", "B"])
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--i_have_author_goahead", action="store_true")
    a = ap.parse_args()
    assert a.i_have_author_goahead, "REFUSING TO RUN without --i_have_author_goahead"
    torch.set_num_threads(a.threads)
    np.random.seed(SEED)
    _DJN = ["arthouse", "dialogue", "marvel", "gore", "teenagers", "inspiring"]
    if a.disjoint:
        _d = np.load(a.disjoint)
        _DJI, _DJC = _d["own_item"], _d["own_concept"]
        _DJR, _DJG, _DJH = _d["val_rows"], _d["val_group"], _d["val_has"]

    from strategy_channel_suite import build_greedy_env, CERT_TOWER, CERT_CONC
    from ndcg_tree import build_pool, LIKE_MIN_LEVEL
    from m4b_eval import load_tree
    from m3_show_paths import build_names
    from opener_gen import centroid_lift, make_openers
    from m6d_eval import rollout_with_openers
    from policy_net import Policy, build_phi
    from run_guard import require_mains
    cname, _iname = build_names()
    require_mains("m6v route confusion", log=log)

    # ---- every tree's ROOT question: the label the policy's first pick reveals -------------------
    roots = {}
    _cd, cold_nodes = load_tree(os.path.join(B, a.cold_tree))
    roots["cold"] = tuple(cold_nodes["root"]["splitter"])
    trees = {"cold": cold_nodes}
    for g in CONCEPTS:
        p = os.path.join(M6, (f"m6k_dj{g}{a.tree_suffix}.json" if a.disjoint
                              else f"m6k_{g}{a.tree_suffix}.json"))
        if not os.path.exists(p):
            continue
        d = json.load(open(p, encoding="utf-8"))
        nd = d["nodes"]
        nd = nd if isinstance(nd, dict) else {r["nid"]: r for r in nd}
        trees[g] = nd
        roots[g] = tuple(nd["root"]["splitter"])
    assert len(set(roots.values())) == len(roots), \
        f"two trees share a root question -- the first pick would not identify the tree: {roots}"
    inv = {q: g for g, q in roots.items()}
    log(f"[roots] {len(roots)} distinct root questions: " +
        ", ".join(f"{g}={q}" for g, q in roots.items()))

    E = build_greedy_env(CERT_TOWER, "validation", CERT_CONC, "m6v")
    pool, _i, _c = build_pool(E, 500, 500)
    vrows = np.asarray(E["rows"], np.int64)
    V, F = np.asarray(E["V"]), np.asarray(E["foldable"])
    zf = np.load(os.path.join(B, "m1", "z_full.npz"))
    ZF, ZR = zf["z"].astype(np.float64), zf["rows"]
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    log(f"[ckpt] epoch {ck['epoch']} arch={ck['arch']} encoding={ck.get('anchor_encoding')}")
    assert ck.get("anchor_encoding") == "none", "state-only policies only"
    assert [list(q) for q in pool] == ck["pool"], "not the pool the policy was trained on"
    norm = (np.asarray(ck["norm"][0], np.float32), np.asarray(ck["norm"][1], np.float32))
    phi_ext, _ = build_phi(E, [tuple(q) for q in ck["ext"]], norm=norm)
    pol = Policy(ck["d_z"], ck["d_phi"], len(pool), arch=ck["arch"], state=ck["state"],
                 use_pos=True)
    pol.load_state_dict(ck["sd"])
    pol.eval()

    names = ["cold"] + sorted(CONCEPTS)
    out = {"analysis": "m6v_route_confusion", "test_touched": False, "half": a.half,
           "ckpt": os.path.basename(a.ckpt), "note": "DIAGNOSTIC ONLY -- routing is not value",
           "roots": {g: list(q) for g, q in roots.items()}, "rows": {}}
    log("=" * 100)
    hdr = "".join(f"{n[:8]:>9}" for n in names)
    log(f"{'TRUE\\PRED':<12}{hdr}{'other':>9}{'correct':>9}")
    for g in sorted(CONCEPTS):
        if g not in trees:
            continue
        cid = CONCEPTS[g]
        coh = vrows[((F > 0) & (V > 0))[vrows, cid]]   # centroid: overlapping, as m7b/m6d
        rows = (_DJR[(_DJG == _DJN.index(g)) & _DJH] if a.disjoint else coh)
        tab = centroid_lift(E, coh, ZF, ZR)
        _dj = None if not a.disjoint else (_DJI, _DJC, _DJN.index(g))
        ops_l, _m = make_openers(E, rows, tab, cname, LIKE_MIN_LEVEL, seed=SEED,
                                 half=a.half, group_key=g, disjoint=_dj)
        ops = {int(r): list(o) for r, o in zip(rows, ops_l)}
        seqs = rollout_with_openers(E, pol, phi_ext, pool, rows, ops, TOTAL_L)
        first = [tuple(s[0]) for s in seqs]
        cnt = {n: 0 for n in names}
        other = 0
        for q in first:
            if q in inv:
                cnt[inv[q]] += 1
            else:
                other += 1
        n = len(first)
        line = "".join(f"{cnt[x] / n:>9.3f}" for x in names)
        log(f"{g:<12}{line}{other / n:>9.3f}{cnt[g] / n:>9.3f}")
        out["rows"][g] = {"n": int(n), "pred": {x: cnt[x] / n for x in names},
                          "other": other / n, "correct": cnt[g] / n}
        json.dump(out, open(os.path.join(M6, "m6v_route_confusion.json"), "w"), indent=1)

    log("=" * 100)
    tot = {x: 0.0 for x in names}
    N = 0
    for g, r in out["rows"].items():
        for x in names:
            tot[x] += r["pred"][x] * r["n"]
        N += r["n"]
    log("[where the mistakes go] share of ALL warm users routed to each tree:")
    for x in names:
        log(f"    {x:<11} {tot[x] / max(N,1):.3f}")
    _oth = sum(r["other"] * r["n"] for r in out["rows"].values()) / max(N, 1)
    assert _oth < 0.5, (
        f"ROUTING GUARD FAILED: {_oth:.3f} of warm users opened with a question that is NO "
        f"tree's root. A policy distilled from these teachers cannot do that, so the trees "
        f"loaded here are NOT the trees it was distilled from. Check --disjoint/--tree_suffix. "
        f"Seen 2026-08-29: m6v loaded the overlapping m6k_<g> trees against a policy distilled "
        f"from m6k_dj<g> and reported routing 0.000 for five of six groups.")
    log("[reading] one dominant wrong column => class imbalance / default attractor (cheap fix);")
    log("          spread => the folded opener does not separate the cohorts (expensive fix).")
    log("[done]")


if __name__ == "__main__":
    main()
