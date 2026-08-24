r"""M6d / M6e -- does the opener-fitted forest policy GENERALISE, and does it survive a group mix?

Pre-registered in `docs/design/DS_M6_REDDIT_SHAPED_OPENERS.md` before any of this data existed.
Author, 2026-08-20: *"continue with the tree teachers, distillation, generalisation. do not touch
test dataset yet, still experimenting on val"*. **VALIDATION ONLY. There is no code path in this file
that can build a test environment.**

TWO MODES, one machinery.

--mode generalise  (M6d)
    Validation users in each group wear an opener built from **film-pool half B** -- films the policy
    has never seen in that group's openers, since training drew only from half A. The split is a
    seeded hash of the item id, fixed before any fit.
    Arms: the distilled POLICY, that group's opener-fitted TREE, and the COLD tree.
    ⚠ THE CRITERION IS VALUE, NOT ROUTING (pre-registered, and it is the M4b failure mode): a policy
    that routes to the right group but does not beat the cold tree has changed its behaviour without
    earning anything, which by this project's own rules is not evidence. Routing agreement is
    reported as diagnosis, never as the result.

--mode mix  (M6e)
    All 10,000 validation users at once. Group members wear a half-B opener; everyone else is cold and
    volunteers nothing. ONE policy serves both.
    Arms: the POLICY, the COLD tree, the STATIC order.
    ★ THE DO-NO-HARM CELL IS LOAD-BEARING: the ~60% who volunteer nothing are reported SEPARATELY.
    A mix win bought by degrading the silent majority is arithmetic, not a result.

MULTI-GROUP USERS. 27.8% of group members endorse two or more anchors, so "which group" is undefined
for them. Rule, declared here rather than chosen later: assign each user to the group whose **centroid
their full-profile latent is nearest** -- the taste geometry decides, not the anchor ordering.
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
VAL_M0_FULL = 0.15686538118354684
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
    ap = argparse.ArgumentParser(description="M6d/M6e -- generalisation and the group mix")
    ap.add_argument("--mode", default="generalise", choices=["generalise", "mix"])
    ap.add_argument("--ckpt", default=os.path.join(M6, "m6_policy.pt"))
    ap.add_argument("--half", default="B", choices=["A", "B"])
    ap.add_argument("--tree_suffix", default="")
    ap.add_argument("--groups", default="", help="restrict to these groups")
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--cold_tree", default="coldtree_v2.json",
                    help="the COLD teacher. Pass coldtree_v2_train10000.json to keep BOTH teachers train-fitted; mixing a validation-fitted cold arm with train-fitted group arms puts one teacher at home and the other away, which is the confound m6r exists to remove (home advantage measured at +0.02881).")
    ap.add_argument("--i_have_author_goahead", action="store_true")
    a = ap.parse_args()
    assert a.i_have_author_goahead, "REFUSING TO RUN without --i_have_author_goahead"
    torch.set_num_threads(a.threads)
    np.random.seed(SEED)

    from strategy_channel_suite import (build_greedy_env, genv_mean, genv_eval_multi,
                                        CERT_TOWER, CERT_CONC)
    from ndcg_tree import build_pool, leak_assert, LIKE_MIN_LEVEL
    from policy_net import Policy, build_phi
    from tree_route import route
    from m4b_eval import load_tree
    from m3_show_paths import build_names
    from opener_gen import centroid_lift, make_openers
    cname, _iname = build_names()

    log("#" * 100)
    log(f"# M6 {a.mode.upper()} -- VALIDATION ONLY, half {a.half!r} openers")
    log("#" * 100)
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    log(f"[ckpt] epoch {ck['epoch']} arch={ck['arch']} encoding={ck.get('anchor_encoding')}")
    assert ck.get("anchor_encoding") == "none", \
        "M6 policies are STATE-ONLY by necessity -- there is no token to announce"

    E = build_greedy_env(CERT_TOWER, "validation", CERT_CONC, f"m6{a.mode}")
    leak_assert(E, E["rows"], f"m6{a.mode}")
    rows_all = np.asarray(E["rows"], np.int64)
    f0, _ = genv_mean(E, [], rows_all)
    assert round(f0, 8) == round(VAL_M0_FULL, 8), f"CANONICAL SNAP FAILED: {f0!r}"
    log(f"[gate] canonical m=0 snap full={f0!r} -- exact")

    pool, _i, _c = build_pool(E, 500, 500)
    assert [list(q) for q in pool] == ck["pool"], "the pool is not the pool the policy was trained on"
    n_pool = len(pool)
    ext = [tuple(q) for q in ck["ext"]]
    norm = (np.asarray(ck["norm"][0], np.float32), np.asarray(ck["norm"][1], np.float32))
    phi_ext, _ = build_phi(E, ext, norm=norm)
    pol = Policy(ck["d_z"], ck["d_phi"], n_pool, arch=ck["arch"], state=ck["state"], use_pos=True)
    pol.load_state_dict(ck["sd"])
    pol.eval()

    static = [(int(x), int(y)) for x, y in
              json.load(open(os.path.join(B, "greedy_static_policyB_combined.json")))["sequence"]]
    _cd, cold_nodes = load_tree(os.path.join(B, a.cold_tree))
    log(f"[cold] teacher = {a.cold_tree} ({len(cold_nodes)} nodes)")
    V, F = np.asarray(E["V"]), np.asarray(E["foldable"])
    endorse = (F > 0) & (V > 0)
    zf = np.load(os.path.join(B, "m1", "z_full.npz"))
    ZF, ZR = zf["z"].astype(np.float64), zf["rows"]
    zpos = {int(r): k for k, r in enumerate(ZR)}

    groups, tabs, trees = [], {}, {}
    want = {x.strip() for x in a.groups.split(",") if x.strip()} or set(ANCHORS)
    for g in ANCHORS:
        if g not in want:
            continue
        p = os.path.join(M6, f"m6k_{g}{a.tree_suffix}.json")
        if not os.path.exists(p):
            log(f"[skip] {g}: no opener-fitted tree")
            continue
        d = json.load(open(p, encoding="utf-8"))
        nd = d["nodes"]
        trees[g] = {r["nid"]: r for r in nd} if isinstance(nd, list) else nd
        coh = rows_all[endorse[rows_all, ANCHORS[g]]]
        tabs[g] = centroid_lift(E, coh, ZF, ZR)
        groups.append(g)
    assert groups, "no opener-fitted trees found"
    log(f"[groups] {groups}")

    # ---- assign every user to at most ONE group, by nearest centroid (declared rule) ----------
    cent = {g: ZF[[zpos[int(r)] for r in rows_all[endorse[rows_all, ANCHORS[g]]]
                   if int(r) in zpos]].mean(0) for g in groups}
    assign = {}
    for r in rows_all:
        mine = [g for g in groups if endorse[int(r), ANCHORS[g]]]
        if not mine:
            continue
        if len(mine) == 1:
            assign[int(r)] = mine[0]
        else:
            z = ZF[zpos[int(r)]]
            assign[int(r)] = min(mine, key=lambda g: float(np.linalg.norm(z - cent[g])))
    warm = np.asarray(sorted(assign), np.int64)
    cold = np.asarray([int(r) for r in rows_all if int(r) not in assign], np.int64)
    log(f"[mix] {len(warm)} in a group ({len(warm) / len(rows_all):.1%}), {len(cold)} cold")

    # ---- per-group half-B openers -------------------------------------------------------------
    openers = {}
    for g in groups:
        rows_g = np.asarray([r for r in warm if assign[int(r)] == g], np.int64)
        if len(rows_g) == 0:
            continue
        ops, meta = make_openers(E, rows_g, tabs[g], cname, LIKE_MIN_LEVEL,
                                 seed=SEED, half=a.half, group_key=g)
        for r, op in zip(rows_g, ops):
            openers[int(r)] = op
        log(f"  [{g}] {len(rows_g)} users · pool(half {a.half}) {meta['pool_size']} · "
            f"film {meta['n_film'] / max(meta['n'], 1):.3f} unserved {meta['unserved']}")

    results = {"mode": a.mode, "half": a.half, "split": "validation", "test_touched": False,
               "n_warm": int(len(warm)), "n_cold": int(len(cold)), "cells": {}}

    def score(rows, ops, arms, tag):
        """arms: dict name -> per-user asked sequence. Returns per-arm (full, tail) arrays."""
        out = {}
        for name, seqs in arms.items():
            per_k = {}
            for K in BUDGETS:
                iseq, cseq = [], []
                for r, s in zip(rows, seqs):
                    op = ops[int(r)]
                    ask = s[:K]
                    iseq.append(list(dict.fromkeys([i for c, i in op if c == 0] +
                                                   [q[1] for q in ask if q[0] == 0])))
                    cseq.append(list(dict.fromkeys([i for c, i in op if c == 1] +
                                                   [q[1] for q in ask if q[0] == 1])))
                full, tail = genv_eval_multi(E, rows, iseq, cseq)
                per_k[K] = (np.asarray(full, np.float64), np.asarray(tail, np.float64))
            out[name] = per_k
        return out

    if a.mode == "generalise":
        for g in groups:
            rows_g = np.asarray([r for r in warm if assign[int(r)] == g], np.int64)
            if len(rows_g) < 150:
                log(f"[skip] {g}: only {len(rows_g)} users after assignment")
                continue
            ops = {int(r): openers[int(r)] for r in rows_g}
            umask = {int(r): list(ops[int(r)]) for r in rows_g}
            # the opener is PER USER, so `m4b_eval.policy_rollout` (one shared seed)
            # does not apply -- the rollout below folds each user's own opener.
            s_pol = rollout_with_openers(E, pol, phi_ext, pool, rows_g, ops, TOTAL_L)
            s_tree, _d1 = route(E, rows_g, trees[g], TOTAL_L, static=static, mask=umask,
                                on_conflict="burn")
            s_cold, _d2 = route(E, rows_g, cold_nodes, TOTAL_L, static=static, mask=umask,
                                on_conflict="burn")
            agree_root = float(np.mean([tuple(p[0]) == tuple(t[0])
                                        for p, t in zip(s_pol, s_tree)]))
            agree_all = float(np.mean([[tuple(q) for q in p] == [tuple(q) for q in t]
                                       for p, t in zip(s_pol, s_tree)]))
            sc = score(rows_g, ops, {"policy": s_pol, "tree": s_tree, "cold": s_cold}, g)
            cell = {"n": int(len(rows_g)), "route_root": agree_root, "route_whole": agree_all,
                    "budgets": {}}
            for K in BUDGETS:
                pf = sc["policy"][K][0]; cf = sc["cold"][K][0]; tf = sc["tree"][K][0]
                m1, l1, h1 = paired_boot(pf - cf)
                m2, l2, h2 = paired_boot(pf - tf)
                cell["budgets"][str(K)] = {
                    "policy": float(np.nanmean(pf)), "tree": float(np.nanmean(tf)),
                    "cold": float(np.nanmean(cf)),
                    "policy_minus_cold": m1, "ci_cold": [l1, h1],
                    "policy_minus_tree": m2, "ci_tree": [l2, h2]}
            results["cells"][g] = cell
            c = cell["budgets"]["15"]
            st = "*" if c["ci_cold"][0] > 0 else ("-" if c["ci_cold"][1] < 0 else " ")
            log(f"[{g}] n={len(rows_g):<5} route root {agree_root:.3f} whole {agree_all:.3f} | "
                f"K=15 policy {c['policy']:.5f} tree {c['tree']:.5f} cold {c['cold']:.5f} | "
                f"VALUE vs cold {c['policy_minus_cold']:+.5f}{st} vs tree "
                f"{c['policy_minus_tree']:+.5f}")
    else:
        allrows = np.concatenate([warm, cold])
        ops = {int(r): openers.get(int(r), []) for r in allrows}
        umask = {int(r): list(ops[int(r)]) for r in allrows}
        s_pol = rollout_with_openers(E, pol, phi_ext, pool, allrows, ops, TOTAL_L)
        s_cold, _d = route(E, allrows, cold_nodes, TOTAL_L, static=static, mask=umask,
                           on_conflict="burn")
        s_stat = [[q for q in static if q not in set(ops[int(r)])][:TOTAL_L] for r in allrows]
        sc = score(allrows, ops, {"policy": s_pol, "cold": s_cold, "static": s_stat}, "mix")
        iswarm = np.isin(allrows, warm)
        for label, sel in (("ALL", np.ones(len(allrows), bool)),
                           ("warm", iswarm), ("COLD (do-no-harm)", ~iswarm)):
            cell = {"n": int(sel.sum()), "budgets": {}}
            for K in BUDGETS:
                pf = sc["policy"][K][0][sel]; cf = sc["cold"][K][0][sel]
                sf = sc["static"][K][0][sel]
                m1, l1, h1 = paired_boot(pf - cf)
                m2, l2, h2 = paired_boot(pf - sf)
                cell["budgets"][str(K)] = {
                    "policy": float(np.nanmean(pf)), "cold": float(np.nanmean(cf)),
                    "static": float(np.nanmean(sf)),
                    "policy_minus_cold": m1, "ci_cold": [l1, h1],
                    "policy_minus_static": m2, "ci_static": [l2, h2]}
            results["cells"][label] = cell
            c = cell["budgets"]["15"]
            st = "*" if c["ci_cold"][0] > 0 else ("-" if c["ci_cold"][1] < 0 else " ")
            log(f"[{label:<18}] n={cell['n']:<6} K=15 policy {c['policy']:.5f} cold {c['cold']:.5f} "
                f"static {c['static']:.5f} | vs cold {c['policy_minus_cold']:+.5f}{st} "
                f"vs static {c['policy_minus_static']:+.5f}")
    json.dump(results, open(os.path.join(M6, f"m6{a.mode}_{a.half}.json"), "w"), indent=1)
    log("[done]")


def rollout_with_openers(E, pol, phi_ext, pool, rows, ops, L):
    """Policy rollout where every user carries their OWN folded opener and nothing is announced."""
    from strategy_channel_suite import genv_eval_multi
    from ndcg_tree import branch_of_per_user
    n, P = len(rows), len(pool)
    asked = np.full((n, L), -1, np.int64)
    outs = np.full((n, L), -1, np.int8)
    amask = np.zeros((n, P), bool)
    pidx = {tuple(q): i for i, q in enumerate(pool)}
    base_i = [[i for c, i in ops[int(r)] if c == 0] for r in rows]
    base_c = [[i for c, i in ops[int(r)] if c == 1] for r in rows]
    for u, r in enumerate(rows):                       # a volunteered question is never re-asked
        for q in ops[int(r)]:
            if tuple(q) in pidx:
                amask[u, pidx[tuple(q)]] = True
    z = np.asarray(genv_eval_multi(E, rows, [list(x) for x in base_i], [list(x) for x in base_c],
                                   score=False, return_z=True)[-1], np.float32)
    phi_cand = torch.from_numpy(phi_ext[:P])
    pol.eval()
    for t in range(L):
        hq = np.zeros((n, L), np.int64); ho = np.zeros((n, L), np.int64)
        hm = np.zeros((n, L), np.float32)
        for j in range(t):
            hq[:, j] = asked[:, j]; ho[:, j] = outs[:, j]; hm[:, j] = 1.0
        pos = np.tile(np.asarray([[t, L - t]], np.float32), (n, 1))
        with torch.no_grad():
            for st in range(0, n, 1000):
                sl = slice(st, min(st + 1000, n))
                s = pol(torch.from_numpy(z[sl]), torch.from_numpy(phi_ext[hq[sl]]),
                        torch.from_numpy(ho[sl]), torch.from_numpy(hm[sl]),
                        torch.from_numpy(pos[sl]), phi_cand, torch.from_numpy(amask[sl]))
                asked[sl, t] = s.argmax(1).numpy()
        outs[:, t] = branch_of_per_user(E, rows, [pool[int(j)] for j in asked[:, t]])
        amask[np.arange(n), asked[:, t]] = True
        if t + 1 < L:
            iseq = [list(dict.fromkeys(base_i[u] + [pool[int(j)][1] for j in asked[u, :t + 1]
                                                    if pool[int(j)][0] == 0])) for u in range(n)]
            cseq = [list(dict.fromkeys(base_c[u] + [pool[int(j)][1] for j in asked[u, :t + 1]
                                                    if pool[int(j)][0] == 1])) for u in range(n)]
            z = np.asarray(genv_eval_multi(E, rows, iseq, cseq, score=False, return_z=True)[-1],
                           np.float32)
    assert (asked >= 0).all()
    return [[tuple(pool[int(j)]) for j in asked[u]] for u in range(n)]


if __name__ == "__main__":
    main()
