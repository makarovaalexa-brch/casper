r"""M6w -- WHY DOES `marvel` DRAG THE POLICY DOWN?

`marvel` is 734 of 5,406 pooled rows and carries a -0.015..-0.020 deficit against its own teacher --
40% of the entire shortfall from 15% of the users. It is the only cohort where the distilled policy
scores BELOW the static order (0.39377 vs 0.39483), and its cold baseline is far the highest of the
six (0.40569 vs 0.25-0.34), so there is least room and most to lose.

Three candidate explanations, and this separates them:

  A. THE WRONG TREE IS GENUINELY HARMFUL. Route a marvel user to `gore` and they lose real value.
     -> the value matrix below has a bad `marvel x gore` cell. Fix = routing.
  B. THE WRONG TREE IS FINE, ROUTING IS MISLABELLED. marvel users are multi-membership, so `gore`
     is a legitimate tree for them and the deficit lies elsewhere.
     -> value matrix rows are flat; overlap is high. Fix = the assignment rule, not the policy.
  C. THE POLICY IS WORSE THAN ANY TREE. It is not imitating a wrong teacher, it is improvising.
     -> the policy scores below EVERY tree's row. Fix = fidelity, not routing.

Measures, on the marvel validation cohort at their own openers:
  1. the full value matrix: every tree scored on marvel users
  2. cohort overlap: how many marvel users also endorse each other concept
  3. the policy's own score, decomposed by WHICH tree it routed each user to

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
    ap = argparse.ArgumentParser(description="M6w -- why marvel drags the policy down")
    ap.add_argument("--cohort", default="marvel")
    ap.add_argument("--ckpt", default=os.path.join(B, "m4", "m6forest_policy.pt"))
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
    from m6d_eval import rollout_with_openers
    from policy_net import Policy, build_phi
    from run_guard import require_mains
    cname, _iname = build_names()
    require_mains(f"m6w {a.cohort} diagnosis", log=log)

    st = [(int(x), int(y)) for x, y in json.load(
        open(os.path.join(B, "greedy_static_policyB_combined_train10000.json")))["sequence"]]
    trees, roots = {}, {}
    _cd, trees["cold"] = load_tree(os.path.join(B, "coldtree_v2_train10000.json"))
    for g in CONCEPTS:
        p = os.path.join(M6, f"m6k_{g}_train2000.json")
        if os.path.exists(p):
            d = json.load(open(p, encoding="utf-8"))
            nd = d["nodes"]
            trees[g] = nd if isinstance(nd, dict) else {r["nid"]: r for r in nd}
    for g, nd in trees.items():
        roots[tuple(nd["root"]["splitter"])] = g

    E = build_greedy_env(CERT_TOWER, "validation", CERT_CONC, "m6w")
    pool, _i, _c = build_pool(E, 500, 500)
    vrows = np.asarray(E["rows"], np.int64)
    V, F = np.asarray(E["V"]), np.asarray(E["foldable"])
    end = (F > 0) & (V > 0)
    zf = np.load(os.path.join(B, "m1", "z_full.npz"))
    ZF, ZR = zf["z"].astype(np.float64), zf["rows"]

    cid = CONCEPTS[a.cohort]
    rows = vrows[end[vrows, cid]]
    tab = centroid_lift(E, rows, ZF, ZR)
    ops_l, _m = make_openers(E, rows, tab, cname, LIKE_MIN_LEVEL, seed=SEED, group_key=a.cohort)
    ops = {int(r): list(o) for r, o in zip(rows, ops_l)}
    log(f"[cohort] {a.cohort}: {len(rows)} validation users at their own openers")

    # ---- 2. overlap: how multi-membership is this cohort? -----------------------------------------
    log("-" * 92)
    log(f"[overlap] share of {a.cohort} users who ALSO endorse each other concept:")
    for g2, c2 in sorted(CONCEPTS.items()):
        if g2 == a.cohort:
            continue
        log(f"    {g2:<11} {float(end[rows, c2].mean()):.3f}")
    n_extra = sum(end[rows, c2] for g2, c2 in CONCEPTS.items() if g2 != a.cohort)
    log(f"[overlap] mean OTHER cohorts endorsed per {a.cohort} user: {float(np.mean(n_extra)):.2f}")

    def score(seqs):
        iseq, cseq = [], []
        for r, s in zip(rows, seqs):
            op = ops[int(r)]
            iseq.append(list(dict.fromkeys([i for c, i in op if c == 0] +
                                           [q[1] for q in s[:TOTAL_L] if q[0] == 0])))
            cseq.append(list(dict.fromkeys([i for c, i in op if c == 1] +
                                           [q[1] for q in s[:TOTAL_L] if q[0] == 1])))
        return np.asarray(genv_eval_multi(E, rows, iseq, cseq)[0], np.float64)

    # ---- 1. the value matrix: every tree on these users ------------------------------------------
    log("-" * 92)
    log(f"[value] every arm scored on the SAME {len(rows)} {a.cohort} users, K=15:")
    per = {}
    umask = {int(r): list(o) for r, o in zip(rows, ops_l)}
    for g, nd in trees.items():
        sq, _dg = route(E, rows, nd, TOTAL_L, static=st, mask=umask, on_conflict="burn")
        per[g] = score(sq)
    per["static"] = score([st[:TOTAL_L]] * len(rows))
    base = float(np.nanmean(per[a.cohort]))
    for g in sorted(per, key=lambda k: -float(np.nanmean(per[k]))):
        d = float(np.nanmean(per[g])) - base
        log(f"    {g:<11} {float(np.nanmean(per[g])):.5f}   vs its own tree {d:+.5f}")

    # ---- 3. the policy, decomposed by where it routed ---------------------------------------------
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    assert ck.get("anchor_encoding") == "none"
    assert [list(q) for q in pool] == ck["pool"]
    nm = (np.asarray(ck["norm"][0], np.float32), np.asarray(ck["norm"][1], np.float32))
    phi_ext, _ = build_phi(E, [tuple(q) for q in ck["ext"]], norm=nm)
    pol = Policy(ck["d_z"], ck["d_phi"], len(pool), arch=ck["arch"], state=ck["state"], use_pos=True)
    pol.load_state_dict(ck["sd"]); pol.eval()
    pseq = rollout_with_openers(E, pol, phi_ext, pool, rows, umask, TOTAL_L)
    pv = score(pseq)
    dest = [roots.get(tuple(s[0]), "OTHER") for s in pseq]
    log("-" * 92)
    log(f"[policy] overall {float(np.nanmean(pv)):.5f} "
        f"(own tree {base:.5f}, cold {float(np.nanmean(per['cold'])):.5f})")
    log(f"[policy] decomposed by the tree it routed each user to:")
    log(f"    {'routed to':<11} {'n':>5} {'share':>7} {'policy':>9} {'that tree':>10} {'policy-tree':>12}")
    out = {"analysis": "m6w_diag", "cohort": a.cohort, "n": int(len(rows)),
           "test_touched": False, "by_destination": {}}
    for g in sorted(set(dest), key=lambda x: -dest.count(x)):
        m = np.asarray([d == g for d in dest])
        tv = float(np.nanmean(per[g][m])) if g in per else float("nan")
        log(f"    {g:<11} {int(m.sum()):>5} {m.mean():>7.3f} {float(np.nanmean(pv[m])):>9.5f} "
            f"{tv:>10.5f} {float(np.nanmean(pv[m])) - tv:>+12.5f}")
        out["by_destination"][g] = {"n": int(m.sum()), "share": float(m.mean()),
                                    "policy": float(np.nanmean(pv[m])), "that_tree": tv}
    out["arms"] = {g: float(np.nanmean(v)) for g, v in per.items()}
    out["policy_overall"] = float(np.nanmean(pv))
    json.dump(out, open(os.path.join(M6, f"m6w_{a.cohort}_diag.json"), "w"), indent=1)
    log("-" * 92)
    log("[reading] bad cells in the value matrix => routing is the problem (A);")
    log("          flat rows + high overlap  => the assignment rule is (B);")
    log("          policy below EVERY tree   => fidelity, it is improvising (C).")
    log("[done]")


if __name__ == "__main__":
    main()
