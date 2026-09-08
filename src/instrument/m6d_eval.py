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
    ap.add_argument("--disjoint", default="",
                    help="m7b_disjoint.npz: disjoint validation groups and exclusive opener "
                         "material; teachers read from m6k_dj<group>.")
    ap.add_argument("--scramble_openers", action="store_true",
                    help="CONTROL: hand each user ANOTHER user's real opener. Breaks the "
                         "correspondence, preserves the distribution. If routing survives "
                         "this, the opener is not what the router reads.")
    ap.add_argument("--split", default="validation", choices=["validation", "test"],
                    help="THE CONFIRMATORY SPLIT. With --split test you MUST also pass --load_tabs; "
                         "recomputing the opener material on test fits the opener generator on the "
                         "confirmatory set and silently invalidates the run.")
    ap.add_argument("--bank_tabs", default="",
                    help="write the per-group opener material (lift_item, lift_conc, pool) so a "
                         "later run can REUSE it instead of refitting.")
    ap.add_argument("--load_tabs", default="",
                    help="read banked opener material instead of computing it. REQUIRED for "
                         "--split test.")
    ap.add_argument("--i_have_author_goahead", action="store_true")
    ap.add_argument("--static_train", default="",
                    help="DS-A9 (2026-09-08): path (relative to experiments/battery) of the TRAIN-fitted "
                         "static order, e.g. greedy_static_policyB_combined_train10000.json. In "
                         "generalise mode adds the arm `static_train` per group plus the contrasts "
                         "tree-cold, cold-static_train, tree-static_train, policy-static_train and a "
                         "pooled block, and banks every per-user score vector to a .npz. The existing "
                         "policy/tree/cold numbers are unchanged (the gate).")
    a = ap.parse_args()
    assert a.i_have_author_goahead, "REFUSING TO RUN without --i_have_author_goahead"
    assert not (a.split == "test" and not a.load_tabs), (
        "--split test REQUIRES --load_tabs. The opener material (each group's centroid lift and "
        "diagnostic film pool) is a FITTED quantity; recomputing it on test users would fit the "
        "opener generator on the confirmatory split and produce plausible, meaningless numbers.")
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
    log(f"# M6 {a.mode.upper()} -- SPLIT={a.split.upper()}, half {a.half!r} openers")
    if a.split == "test":
        log("# " + "!" * 96)
        log("# CONFIRMATORY TEST LOOK. Material is REUSED from banked validation artefacts, never")
        log("# refitted. Only the users and their own-data group assignment come from test.")
        log("# " + "!" * 96)
    log("#" * 100)
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    log(f"[ckpt] epoch {ck['epoch']} arch={ck['arch']} encoding={ck.get('anchor_encoding')}")
    assert ck.get("anchor_encoding") == "none", \
        "M6 policies are STATE-ONLY by necessity -- there is no token to announce"

    E = build_greedy_env(CERT_TOWER, a.split, CERT_CONC, f"m6{a.mode}")
    leak_assert(E, E["rows"], f"m6{a.mode}")
    rows_all = np.asarray(E["rows"], np.int64)
    f0, _ = genv_mean(E, [], rows_all)
    if a.split == "validation":
        assert round(f0, 8) == round(VAL_M0_FULL, 8), f"CANONICAL SNAP FAILED: {f0!r}"
        log(f"[gate] canonical m=0 snap full={f0!r} -- exact")
    else:
        # No banked m=0 exists for test, so this cannot be an equality gate. It is still recorded:
        # a value far from the validation snap means a broken environment, not a real effect.
        log(f"[gate] test m=0 snap full={f0!r} (validation reference {VAL_M0_FULL!r})")
        assert 0.5 * VAL_M0_FULL < f0 < 2.0 * VAL_M0_FULL, (
            f"test m=0 {f0!r} is implausible against the validation reference "
            f"{VAL_M0_FULL!r} -- the environment is broken, do not read these numbers")

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
    _DJN = ["arthouse", "dialogue", "marvel", "gore", "teenagers", "inspiring"]
    _DJI = _DJC = None
    if a.disjoint:
        _Dd = np.load(a.disjoint); _DJI, _DJC = _Dd["own_item"], _Dd["own_concept"]
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
        p = os.path.join(M6, (f"m6k_dj{g}{a.tree_suffix}.json" if a.disjoint
                              else f"m6k_{g}{a.tree_suffix}.json"))
        if not os.path.exists(p):
            log(f"[skip] {g}: no opener-fitted tree")
            continue
        d = json.load(open(p, encoding="utf-8"))
        nd = d["nodes"]
        trees[g] = {r["nid"]: r for r in nd} if isinstance(nd, list) else nd
        if a.load_tabs:
            _bk = np.load(a.load_tabs, allow_pickle=True)
            tabs[g] = {"lift_item": _bk[f"{g}__lift_item"], "lift_conc": _bk[f"{g}__lift_conc"],
                       "pool": set(int(x) for x in _bk[f"{g}__pool"])}
        else:
            coh = rows_all[endorse[rows_all, ANCHORS[g]]]
            tabs[g] = centroid_lift(E, coh, ZF, ZR)
        groups.append(g)
    assert groups, "no opener-fitted trees found"
    log(f"[groups] {groups}")
    if a.load_tabs:
        log(f"[material] REUSED from {os.path.basename(a.load_tabs)} -- not refitted on {a.split}. "
            + " ".join(f"{g}:pool={len(tabs[g]['pool'])}" for g in groups))
    if a.bank_tabs:
        _out = {}
        for g in groups:
            _out[f"{g}__lift_item"] = np.asarray(tabs[g]["lift_item"])
            _out[f"{g}__lift_conc"] = np.asarray(tabs[g]["lift_conc"])
            _out[f"{g}__pool"] = np.asarray(sorted(int(x) for x in tabs[g]["pool"]), np.int64)
        np.savez_compressed(a.bank_tabs, **_out)
        log(f"[material] banked {len(groups)} groups -> {a.bank_tabs}")

    # ---- assign every user to at most ONE group, by nearest centroid (declared rule) ----------
    cent = {g: ZF[[zpos[int(r)] for r in rows_all[endorse[rows_all, ANCHORS[g]]]
                   if int(r) in zpos]].mean(0) for g in groups}
    assign = {}
    if a.disjoint:
        # Use the BANKED assignment, not nearest-centroid. The teachers were fitted on exactly
        # these groups; re-deriving membership by a different rule here would score each tree on a
        # population it was not fitted for, which is the confound this whole redesign removed.
        # The RULE (m7b_disjoint.assign_users): each user -> the group they endorse most strongly.
        # It reads only the user's OWN signed values, so it fits nothing and is applied identically
        # on any split. On validation we additionally assert it reproduces the banked assignment
        # exactly, which is what licenses using it on test.
        _cids = np.array([ANCHORS[g] for g in _DJN], np.int64)
        _Vs = V[:, _cids].astype(np.float64).copy()
        _Es = endorse[:, _cids]
        _Vs[~_Es] = -np.inf
        _best, _has = _Vs.argmax(1).astype(np.int32), _Es.any(1)
        for r in rows_all:
            if _has[int(r)] and _DJN[int(_best[int(r)])] in groups:
                assign[int(r)] = _DJN[int(_best[int(r)])]
        if a.split == "validation":
            _bank = {int(r): _DJN[int(gi)] for r, gi, ok in
                     zip(_Dd["val_rows"], _Dd["val_group"], _Dd["val_has"])
                     if ok and _DJN[int(gi)] in groups}
            assert _bank == assign, (
                f"the assignment RULE does not reproduce the banked validation assignment "
                f"({len(_bank)} banked vs {len(assign)} derived) -- it may not be used on test")
            log(f"[assign] rule reproduces the banked validation assignment EXACTLY "
                f"({len(assign)} warm users)")
        else:
            log(f"[assign] rule applied to {a.split}: {len(assign)} warm users, one group each "
                f"(own-data argmax; nothing fitted)")
    else:
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
        _dj = None if not a.disjoint else (_DJI, _DJC, _DJN.index(g))
        ops, meta = make_openers(E, rows_g, tabs[g], cname, LIKE_MIN_LEVEL, disjoint=_dj,
                                 seed=SEED, half=a.half, group_key=g)
        for r, op in zip(rows_g, ops):
            openers[int(r)] = op
        log(f"  [{g}] {len(rows_g)} users · pool(half {a.half}) {meta['pool_size']} · "
            f"film {meta['n_film'] / max(meta['n'], 1):.3f} unserved {meta['unserved']}")

    if a.scramble_openers:
        # ⚠ Scramble the CORRESPONDENCE, never the distribution. Each user is handed ANOTHER user's
        # REAL opener, drawn from the whole warm set -- never within-group, which would leave the
        # group signal intact and prove nothing. Film popularity, length, coherence and the
        # group-marginal mix survive exactly; only the link between this user and this opener dies.
        # If routing survives this, the router is not reading the opener and the claim that the
        # volunteered request carries the routing signal is DEAD.
        ks = sorted(openers)
        rng = np.random.default_rng(SEED)
        src = np.asarray(ks, np.int64)[rng.permutation(len(ks))]
        fixed = sum(1 for u, v in zip(ks, src) if int(u) == int(v))
        for _ in range(20):
            if fixed == 0:
                break
            src = np.asarray(ks, np.int64)[rng.permutation(len(ks))]
            fixed = sum(1 for u, v in zip(ks, src) if int(u) == int(v))
        assert fixed <= 1, f"{fixed} users kept their own opener -- that is not a scramble"
        openers = {int(u): openers[int(v)] for u, v in zip(ks, src)}
        log(f"[SCRAMBLE] openers permuted ACROSS {len(ks)} warm users, {fixed} fixed points -- "
            f"correspondence broken, distribution identical")

    results = {"mode": a.mode, "half": a.half, "split": a.split,
               "test_touched": (a.split == "test"),
               "material_reused_from": os.path.basename(a.load_tabs) if a.load_tabs else None,
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
        st_tr = None
        if a.static_train:
            _sp = os.path.join(B, a.static_train)
            st_tr = [(int(x), int(y)) for x, y in json.load(open(_sp))["sequence"]][:TOTAL_L]
            log(f"[static_train] {a.static_train}: {len(st_tr)} questions (DS-A9 extra arm)")
            results["static_train"] = a.static_train
        peruser = {}                      # DS-A9: banked per-user score vectors, one look
        pooled = {}                       # arm -> K -> list of per-user arrays, concatenated below
        pooled_rows = []
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
            arms = {"policy": s_pol, "tree": s_tree, "cold": s_cold}
            if st_tr is not None:
                # the static order asked after the user's own opener, opener questions skipped,
                # exactly as m6t_order_check scores `static_train` on validation
                arms["static_train"] = [[q for q in st_tr if q not in set(ops[int(r)])][:TOTAL_L]
                                        for r in rows_g]
            sc = score(rows_g, ops, arms, g)
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
                if st_tr is not None:
                    sf = sc["static_train"][K][0]
                    row = cell["budgets"][str(K)]
                    row["static_train"] = float(np.nanmean(sf))
                    for name, diff in (("tree_minus_cold", tf - cf),
                                       ("cold_minus_static_train", cf - sf),
                                       ("tree_minus_static_train", tf - sf),
                                       ("policy_minus_static_train", pf - sf)):
                        m, l, h = paired_boot(diff)
                        row[name] = m; row[f"ci_{name}"] = [l, h]
                for arm in sc:
                    peruser[f"{g}__{arm}__K{K}__full"] = sc[arm][K][0]
                    peruser[f"{g}__{arm}__K{K}__tail"] = sc[arm][K][1]
                    pooled.setdefault(arm, {}).setdefault(K, []).append(sc[arm][K][0])
            peruser[f"{g}__rows"] = rows_g
            pooled_rows.append(rows_g)
            results["cells"][g] = cell
            c = cell["budgets"]["15"]
            st = "*" if c["ci_cold"][0] > 0 else ("-" if c["ci_cold"][1] < 0 else " ")
            log(f"[{g}] n={len(rows_g):<5} route root {agree_root:.3f} whole {agree_all:.3f} | "
                f"K=15 policy {c['policy']:.5f} tree {c['tree']:.5f} cold {c['cold']:.5f} | "
                f"VALUE vs cold {c['policy_minus_cold']:+.5f}{st} vs tree "
                f"{c['policy_minus_tree']:+.5f}")
            if st_tr is not None:
                for name in ("tree_minus_cold", "cold_minus_static_train", "tree_minus_static_train"):
                    l, h = c[f"ci_{name}"]
                    s_ = "*" if l > 0 else ("-" if h < 0 else " ")
                    log(f"        {name:<26} {c[name]:+.5f} [{l:+.5f},{h:+.5f}]{s_}  "
                        f"(static_train {c['static_train']:.5f})")
        # ---- DS-A9: pooled over groups (each user in at most one group under --disjoint) ----
        if pooled:
            uid = np.concatenate(pooled_rows)
            assert len(uid) == len(np.unique(uid)) or not a.disjoint, "a user appears in two groups"
            prec = {"n_rows": int(len(uid)), "n_users": int(len(np.unique(uid))), "budgets": {}}
            for K in BUDGETS:
                P = {arm: np.concatenate(pooled[arm][K]) for arm in pooled}
                row = {arm: float(np.nanmean(P[arm])) for arm in P}
                pairs = [("policy", "cold"), ("policy", "tree"), ("tree", "cold")]
                if st_tr is not None:
                    pairs += [("cold", "static_train"), ("tree", "static_train"),
                              ("policy", "static_train")]
                for hi, lo in pairs:
                    m, l, h = paired_boot(P[hi] - P[lo])
                    row[f"{hi}_minus_{lo}"] = m; row[f"ci_{hi}_minus_{lo}"] = [l, h]
                prec["budgets"][str(K)] = row
            results["pooled"] = prec
            c = prec["budgets"]["15"]
            log(f"[POOLED] {prec['n_rows']} rows, {prec['n_users']} users | K=15 " +
                " ".join(f"{arm} {c[arm]:.5f}" for arm in sorted(pooled)))
            for hi, lo in pairs:
                l, h = c[f"ci_{hi}_minus_{lo}"]
                s_ = "*" if l > 0 else ("-" if h < 0 else " ")
                log(f"        {hi} - {lo:<13} {c[f'{hi}_minus_{lo}']:+.5f} [{l:+.5f},{h:+.5f}]{s_}")
            _sfx = "" if a.split == "validation" else f"_{a.split}"
            _pp = os.path.join(M6, f"m6generalise_{a.half}{_sfx}_peruser.npz")
            np.savez_compressed(_pp, **{k: np.asarray(v) for k, v in peruser.items()})
            log(f"[peruser] banked {len(peruser)} arrays -> {_pp}")
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
    # ⚠ The split is IN THE FILENAME. Before 2026-08-29 both splits wrote to the same path with a
    # hardcoded "split": "validation", so a test run silently overwrote the validation results and
    # labelled itself test_touched=False. A filename is not provenance, but a shared filename is a
    # provenance HAZARD: two populations must never contend for one path.
    _sfx = "" if a.split == "validation" else f"_{a.split}"
    json.dump(results, open(os.path.join(M6, f"m6{a.mode}_{a.half}{_sfx}.json"), "w"), indent=1)
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
