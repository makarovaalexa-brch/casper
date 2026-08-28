r"""M6t -- DOES THE ORDER HOLD?  static < cold < opener, on each cohort's own opener states.

Every arm is fitted on TRAIN and evaluated on VALIDATION, so no arm is on its own fitting data and
the comparison is symmetric. Reference arms fitted on validation are scored alongside, clearly
labelled, because they are the incumbents the chapter has been quoting -- they are AT HOME here and
their numbers are not comparable to the train-fitted arms.

  ARMS ON EQUAL FOOTING (all fitted on TRAIN, none has seen validation)
    static_train   the 15-question global greedy order, 10,000 train users
    cold_train     coldtree_v2_train10000, 144 nodes, 10,000 train users
    opener         this cohort's own opener-fitted tree, 2,000 train users

  REFERENCE ONLY (fitted on VALIDATION -- at home, do not read as a fair contrast)
    static_val, cold_val

Each cohort is evaluated at ITS OWN opener states: per-user Reddit-shaped openers, deterministic in
the user id, drawn from the full film pool.

POOLING. A user may endorse several concepts (1.37 on average), so pooled rows are (user, cohort)
pairs and a user can appear more than once. The pooled bootstrap therefore RESAMPLES USERS, not
rows, carrying all of a user's rows together -- otherwise the repeated users would be treated as
independent and the interval would be too narrow.

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
BUDGETS = [1, 4, 15]
FAIR = ["static_train", "cold_train", "opener", "policy"]
REF = ["static_val", "cold_val"]


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def boot_users(diff, uids, n=10000, seed=SEED):
    """Cluster bootstrap over USERS: a user's rows move together."""
    diff = np.asarray(diff, np.float64)
    uids = np.asarray(uids, np.int64)
    ok = np.isfinite(diff)
    diff, uids = diff[ok], uids[ok]
    if len(diff) == 0:
        return float("nan"), float("nan"), float("nan")
    uniq, inv = np.unique(uids, return_inverse=True)
    by = [np.flatnonzero(inv == k) for k in range(len(uniq))]
    rng = np.random.default_rng(seed)
    out = np.empty(n)
    for b in range(n):
        pick = rng.integers(0, len(uniq), len(uniq))
        idx = np.concatenate([by[j] for j in pick])
        out[b] = diff[idx].mean()
    return float(diff.mean()), float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def main():
    ap = argparse.ArgumentParser(description="M6t -- does static < cold < opener hold on validation?")
    ap.add_argument("--cohorts", default="", help="default: every cohort with a train2000 tree")
    ap.add_argument("--ckpt", default="",
                    help="if given, add the DISTILLED POLICY as a fourth fair arm, rolled out on "
                         "the SAME openers as every other arm so all four are on one ruler")
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
    from run_guard import require_mains
    cname, _iname = build_names()
    require_mains("m6t order check", log=log)
    pol = phi_ext = None
    if a.ckpt:
        from m6d_eval import rollout_with_openers
        from policy_net import Policy, build_phi

    want = [g.strip() for g in a.cohorts.split(",") if g.strip()] or list(CONCEPTS)
    trees = {}
    for g in want:
        p = os.path.join(M6, f"m6k_{g}_train2000.json")
        if not os.path.exists(p):
            log(f"[skip] {g}: no train2000 tree yet")
            continue
        d = json.load(open(p, encoding="utf-8"))
        assert d.get("split") == "train", f"{g} tree is {d.get('split')}-fitted, not train"
        nd = d["nodes"]
        trees[g] = nd if isinstance(nd, dict) else {r["nid"]: r for r in nd}
    assert trees, "no train-fitted cohort trees found"

    _cd, cold_train = load_tree(os.path.join(B, "coldtree_v2_train10000.json"))
    _cv, cold_val = load_tree(os.path.join(B, "coldtree_v2.json"))
    st_tr = [(int(x), int(y)) for x, y in json.load(
        open(os.path.join(B, "greedy_static_policyB_combined_train10000.json")))["sequence"]]
    st_va = [(int(x), int(y)) for x, y in json.load(
        open(os.path.join(B, "greedy_static_policyB_combined.json")))["sequence"]][:TOTAL_L]

    log("#" * 104)
    log(f"# M6t -- ORDER CHECK on VALIDATION. cohorts: {sorted(trees)}")
    log(f"# fair arms (all TRAIN-fitted): {FAIR}   |   reference (validation-fitted, AT HOME): {REF}")
    log("#" * 104)

    E = build_greedy_env(CERT_TOWER, "validation", CERT_CONC, "m6t")
    pool, _i, _c = build_pool(E, 500, 500)
    vrows = np.asarray(E["rows"], np.int64)
    V, F = np.asarray(E["V"]), np.asarray(E["foldable"])
    zf = np.load(os.path.join(B, "m1", "z_full.npz"))
    ZF, ZR = zf["z"].astype(np.float64), zf["rows"]

    if a.ckpt:
        ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
        log(f"[policy] {os.path.basename(a.ckpt)} epoch {ck['epoch']} arch={ck['arch']} "
            f"encoding={ck.get('anchor_encoding')}")
        assert ck.get("anchor_encoding") == "none", "state-only policies only"
        assert [list(q) for q in pool] == ck["pool"], "not the pool the policy was trained on"
        _nm = (np.asarray(ck["norm"][0], np.float32), np.asarray(ck["norm"][1], np.float32))
        phi_ext, _ = build_phi(E, [tuple(q) for q in ck["ext"]], norm=_nm)
        pol = Policy(ck["d_z"], ck["d_phi"], len(pool), arch=ck["arch"], state=ck["state"],
                     use_pos=True)
        pol.load_state_dict(ck["sd"]); pol.eval()

    out = {"analysis": "m6t_order_check", "ruler": "validation_clean", "test_touched": False,
           "fair_arms": FAIR, "reference_arms": REF, "cohorts": {}}
    pooled = {arm: {K: [] for K in BUDGETS} for arm in FAIR + REF}
    pooled_uid = []

    for g in sorted(trees):
        cid = CONCEPTS[g]
        rows = vrows[((F > 0) & (V > 0))[vrows, cid]]
        tab = centroid_lift(E, rows, ZF, ZR)
        ops, meta = make_openers(E, rows, tab, cname, LIKE_MIN_LEVEL, seed=SEED, group_key=g)
        umask = {int(r): list(o) for r, o in zip(rows, ops)}
        seqs = {}
        for arm, nd in (("opener", trees[g]), ("cold_train", cold_train), ("cold_val", cold_val)):
            seqs[arm], dg = route(E, rows, nd, TOTAL_L, static=st_tr, mask=umask, on_conflict="burn")
            if dg["n_fallback"]:
                log(f"[{g}] {arm}: {dg['n_fallback']}/{len(rows)} continuation-rule fires")
        seqs["static_train"] = [st_tr[:TOTAL_L]] * len(rows)
        seqs["static_val"] = [st_va[:TOTAL_L]] * len(rows)
        if pol is not None:
            seqs["policy"] = rollout_with_openers(E, pol, phi_ext, pool, rows, umask, TOTAL_L)

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
                pooled[arm][K].append(sc[arm][K])
        pooled_uid.append(np.asarray(rows, np.int64))

        rec = {"n": int(len(rows)), "film_rate": meta["n_film"] / max(meta["n"], 1), "budgets": {}}
        for K in BUDGETS:
            row = {arm: float(np.nanmean(sc[arm][K])) for arm in sc}
            pairs = [("opener", "cold_train"), ("cold_train", "static_train"),
                     ("opener", "static_train")]
            if pol is not None:
                pairs += [("policy", "opener"), ("policy", "cold_train"),
                          ("policy", "static_train")]
            for hi, lo in pairs:
                m, l, h = boot_users(sc[hi][K] - sc[lo][K], rows)
                row[f"{hi}_minus_{lo}"] = m
                row[f"ci_{hi}_minus_{lo}"] = [l, h]
            rec["budgets"][str(K)] = row
        out["cohorts"][g] = rec
        c = rec["budgets"]["15"]
        ordered = c["static_train"] < c["cold_train"] < c["opener"]
        log(f"[{g:<10}] n={rec['n']:<5} K=15 | static {c['static_train']:.5f} < cold "
            f"{c['cold_train']:.5f} < opener {c['opener']:.5f} ? {'YES' if ordered else 'NO'}")
        for hi, lo in pairs:
            l, h = c[f"ci_{hi}_minus_{lo}"]
            s = "*" if l > 0 else ("-" if h < 0 else " ")
            log(f"             {hi} - {lo:<13} {c[f'{hi}_minus_{lo}']:+.5f} [{l:+.5f},{h:+.5f}]{s}")
        log(f"             (reference, AT HOME: cold_val {c['cold_val']:.5f}  "
            f"static_val {c['static_val']:.5f})")
        json.dump(out, open(os.path.join(M6, "m6t_order_check.json"), "w"), indent=1)

    # ---- pooled over (user, cohort) rows, bootstrapped over USERS -------------------------------
    uid = np.concatenate(pooled_uid)
    # only arms that were actually scored -- `policy` is absent when --ckpt is not given
    P = {arm: {K: np.concatenate(pooled[arm][K]) for K in BUDGETS}
         for arm in pooled if pooled[arm][BUDGETS[0]]}
    prec = {"n_rows": int(len(uid)), "n_users": int(len(np.unique(uid))), "budgets": {}}
    for K in BUDGETS:
        row = {arm: float(np.nanmean(P[arm][K])) for arm in P}
        ppairs = [("opener", "cold_train"), ("cold_train", "static_train"),
                  ("opener", "static_train")]
        if pol is not None:
            ppairs += [("policy", "opener"), ("policy", "cold_train"), ("policy", "static_train")]
        for hi, lo in ppairs:
            m, l, h = boot_users(P[hi][K] - P[lo][K], uid)
            row[f"{hi}_minus_{lo}"] = m
            row[f"ci_{hi}_minus_{lo}"] = [l, h]
        prec["budgets"][str(K)] = row
    out["pooled"] = prec
    log("=" * 104)
    c = prec["budgets"]["15"]
    log(f"[POOLED] {prec['n_rows']} rows over {prec['n_users']} distinct users "
        f"({prec['n_rows']/max(prec['n_users'],1):.2f} cohorts/user)")
    log(f"[POOLED] K=15 | static {c['static_train']:.5f} < cold {c['cold_train']:.5f} < opener "
        f"{c['opener']:.5f} ? "
        f"{'YES' if c['static_train'] < c['cold_train'] < c['opener'] else 'NO'}")
    for hi, lo in ppairs:
        l, h = c[f"ci_{hi}_minus_{lo}"]
        s = "*" if l > 0 else ("-" if h < 0 else " ")
        log(f"         {hi} - {lo:<13} {c[f'{hi}_minus_{lo}']:+.5f} [{l:+.5f},{h:+.5f}]{s}")
    json.dump(out, open(os.path.join(M6, "m6t_order_check.json"), "w"), indent=1)
    log("[done]")


if __name__ == "__main__":
    main()
