r"""M6c -- the M6a gate, RE-RUN with CLUSTER-DIAGNOSTIC openers.

⚠ WHY THIS SUPERSEDES M6a (author, 2026-08-19): *"we want openers that correspond to a cluster
taste-wise, this is what reddit openers were. what you've done could be anything. introducing noise
for no reason ... we need to prioritise taste that actually puts user in that cluster."*

M6a drew the opener UNIFORMLY from whatever the user happened to like or endorse. That is arbitrary
evidence, not a taste statement: a random film from someone's history says nothing about which
cluster they are in. A real request names something DIAGNOSTIC -- "something like Holy Motors" places
the requester; "I liked Toy Story" does not.

So M6a tested whether a teacher survives an ARBITRARY opener. It did NOT test whether a teacher
survives A DIFFERENT WAY OF STATING THE SAME TASTE, which is the question that matters and the one
the author asked. Its conclusion ("the teachers are anchor teachers") is narrowed accordingly and
must not be quoted beyond arbitrary openers until this run reports.

DIAGNOSTIC OPENERS, by the project's own established measure
------------------------------------------------------------
LIFT over the global rate -- the same statistic M1 used for its "distinctive films" column, adopted
there precisely because raw top-N collapsed to popularity (Shawshank in 4 of 6 clusters):

    lift(x | cluster) = P(likes/endorses x | in cluster) / P(likes/endorses x)

Computed on TRAIN users, so cohort and opener definition never touch the split the measurement runs
on. A user's opener is drawn from the intersection of the cluster's top-lift material and what THEY
actually like or endorse, so it stays their own honest evidence -- diagnostic AND true of them.

SHAPES (the anchor row reproduces M4a and is the reference):
    anchor        the anchor concept alone -- the state the teacher was fitted at
    dfilm1        one diagnostic film
    dfilm2        two diagnostic films
    dconc1        one diagnostic concept, NOT the anchor
    dfilm2conc1   two diagnostic films + one diagnostic non-anchor concept
Each is contrasted with M6a's arbitrary counterpart, already banked.

Coverage is REPORTED, not assumed: a user who likes none of the cluster's distinctive films cannot
wear a diagnostic opener, and how many such users there are is itself a finding.

VALIDATION ONLY for measurement. The 10,000 canonical TEST users are never built, loaded or named.
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
OUT = os.path.join(B, "m6")
SEED = 98765
TOTAL_L = 15
VAL_M0_FULL = 0.15686538118354684
ANCHORS = {"arthouse": 68, "dialogue": 275, "marvel": 581,
           "gore": 417, "teenagers": 922, "inspiring": 503}
BUDGETS = [1, 2, 4, 8, 15]
SHAPES = ["anchor", "dfilm1", "dfilm2", "dconc1", "dfilm2conc1"]


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
    ap = argparse.ArgumentParser(description="M6c -- gate with cluster-diagnostic openers")
    ap.add_argument("--top_films", type=int, default=60, help="size of each cluster's diagnostic film set")
    ap.add_argument("--top_concs", type=int, default=25)
    ap.add_argument("--min_likes", type=int, default=20, help="min in-cluster likes for an item to qualify")
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--i_have_author_goahead", action="store_true")
    a = ap.parse_args()
    assert a.i_have_author_goahead, "REFUSING TO RUN without --i_have_author_goahead"
    torch.set_num_threads(a.threads)
    np.random.seed(SEED)
    rng = np.random.default_rng(SEED)
    os.makedirs(OUT, exist_ok=True)

    from strategy_channel_suite import (build_greedy_env, genv_mean, genv_eval_multi,
                                        CERT_TOWER, CERT_CONC)
    from ndcg_tree import leak_assert, LIKE_MIN_LEVEL
    from tree_route import route
    from m4b_eval import load_tree
    from m3_show_paths import build_names

    log("#" * 100)
    log("# M6c -- does a cluster teacher survive a CLUSTER-DIAGNOSTIC non-anchor opener?")
    log("#" * 100)
    cname, iname = build_names()

    # ---------------------------------------------- diagnostic material, from TRAIN users only
    Et = build_greedy_env(CERT_TOWER, "train", CERT_CONC, "m6c_lift", cohort_slice=(0, 20000))
    Vt, Ft = np.asarray(Et["V"]), np.asarray(Et["foldable"])
    end_t = (Ft > 0) & (Vt > 0)
    lvl_t = Et["lvl"]
    n_tr = end_t.shape[0]
    n_items = Et["ctx"].ni
    like_cnt = np.zeros(n_items, np.float64)
    per_user_likes = []
    for r in range(n_tr):
        ls = np.fromiter((int(i) for i, v in lvl_t[r].items() if v >= LIKE_MIN_LEVEL),
                         np.int64)
        per_user_likes.append(ls)
        like_cnt[ls] += 1
    base_item = like_cnt / n_tr
    base_conc = end_t.mean(0)

    diag = {}
    for an, cid in ANCHORS.items():
        m = np.nonzero(end_t[:, cid])[0]
        cnt = np.zeros(n_items, np.float64)
        for r in m:
            cnt[per_user_likes[r]] += 1
        p_in = cnt / max(len(m), 1)
        lift_i = p_in / np.maximum(base_item, 1e-9)
        ok = cnt >= a.min_likes                      # never rank a lift built on a handful of users
        lift_i = np.where(ok, lift_i, 0.0)
        films = [int(i) for i in np.argsort(-lift_i)[:a.top_films]]
        p_c = end_t[m].mean(0)
        lift_c = p_c / np.maximum(base_conc, 1e-9)
        lift_c[list(ANCHORS.values())] = 0.0
        concs = [int(c) for c in np.argsort(-lift_c)[:a.top_concs]]
        diag[an] = {"films": set(films), "concs": set(concs)}
        log(f"[diag] {an:<11} top films: " +
            ", ".join(iname.get(i, str(i))[:28] for i in films[:5]))
        log(f"{'':<19} top concepts: " + ", ".join(cname.get(c, str(c)) for c in concs[:6]))
    del Et

    # ------------------------------------------------------------------------- the measurement
    E = build_greedy_env(CERT_TOWER, "validation", CERT_CONC, "m6c")
    leak_assert(E, E["rows"], "m6c/val")
    rows_all = np.asarray(E["rows"], np.int64)
    f0, _ = genv_mean(E, [], rows_all)
    assert round(f0, 8) == round(VAL_M0_FULL, 8), f"CANONICAL SNAP FAILED: {f0!r}"
    log(f"[gate] canonical m=0 snap full={f0!r} -- exact")
    static = [(int(x), int(y)) for x, y in
              json.load(open(os.path.join(B, "greedy_static_policyB_combined.json")))["sequence"]]
    _cd, cold_nodes = load_tree(os.path.join(B, "coldtree_v2.json"))
    V, F = np.asarray(E["V"]), np.asarray(E["foldable"])
    endorse = (F > 0) & (V > 0)
    lvl = E["lvl"]
    results = []

    for name, cid in ANCHORS.items():
        _td, t_nodes = load_tree(os.path.join(B, f"m3v2_{name}.json"))
        cohort = rows_all[endorse[rows_all, cid]]
        dfilms, dconcs = diag[name]["films"], diag[name]["concs"]
        have_f, have_c = {}, {}
        for r in cohort:
            r = int(r)
            have_f[r] = sorted(int(i) for i, v in lvl[r].items()
                               if v >= LIKE_MIN_LEVEL and int(i) in dfilms)
            have_c[r] = sorted(int(c) for c in np.nonzero(endorse[r])[0] if int(c) in dconcs)
        cov1 = np.mean([len(have_f[int(r)]) >= 1 for r in cohort])
        cov2 = np.mean([len(have_f[int(r)]) >= 2 for r in cohort])
        covc = np.mean([len(have_c[int(r)]) >= 1 for r in cohort])
        log(f"[{name}] cohort {len(cohort)} | can wear a diagnostic opener: "
            f">=1 film {cov1:.3f}, >=2 films {cov2:.3f}, >=1 concept {covc:.3f}")

        for shape in SHAPES:
            keep, op_i, op_c = [], [], []
            for r in cohort:
                r = int(r)
                it, cc = [], []
                if shape == "anchor":
                    cc = [cid]
                elif shape == "dfilm1":
                    if not have_f[r]:
                        continue
                    it = [int(rng.choice(have_f[r]))]
                elif shape == "dfilm2":
                    if len(have_f[r]) < 2:
                        continue
                    it = [int(x) for x in rng.choice(have_f[r], 2, replace=False)]
                elif shape == "dconc1":
                    if not have_c[r]:
                        continue
                    cc = [int(rng.choice(have_c[r]))]
                elif shape == "dfilm2conc1":
                    if len(have_f[r]) < 2 or not have_c[r]:
                        continue
                    it = [int(x) for x in rng.choice(have_f[r], 2, replace=False)]
                    cc = [int(rng.choice(have_c[r]))]
                keep.append(r); op_i.append(it); op_c.append(cc)
            if len(keep) < 150:
                log(f"   [{name}/{shape}] only {len(keep)} users can wear it -- skipping")
                continue
            keep = np.asarray(keep, np.int64)
            umask = {int(r): [(0, i) for i in oi] + [(1, c) for c in oc]
                     for r, oi, oc in zip(keep, op_i, op_c)}
            seq_cold, _dc = route(E, keep, cold_nodes, TOTAL_L, static=static, mask=umask)
            seq_teach, _dt = route(E, keep, t_nodes, TOTAL_L, static=static, mask=umask)

            row = {"anchor": name, "shape": shape, "n": int(len(keep)),
                   "coverage_1film": float(cov1), "coverage_2film": float(cov2),
                   "coverage_conc": float(covc), "budgets": {}}
            for K in BUDGETS:
                def ev(seqs):
                    iseq, cseq = [], []
                    for s, oi, oc in zip(seqs, op_i, op_c):
                        ask = s[:K]
                        iseq.append(list(dict.fromkeys(list(oi) +
                                                       [q[1] for q in ask if q[0] == 0])))
                        cseq.append(list(dict.fromkeys(list(oc) +
                                                       [q[1] for q in ask if q[0] == 1])))
                    full, tail = genv_eval_multi(E, keep, iseq, cseq)
                    return np.asarray(full, np.float64), np.asarray(tail, np.float64)
                cf, ct = ev(seq_cold)
                tf, tt = ev(seq_teach)
                m, lo, hi = paired_boot(tf - cf)
                row["budgets"][str(K)] = {
                    "cold_full": float(np.nanmean(cf)), "teacher_full": float(np.nanmean(tf)),
                    "teacher_minus_cold_full": m, "ci_full": [lo, hi],
                    "teacher_minus_cold_tail": paired_boot(tt - ct)[0]}
            c = row["budgets"]["15"]
            star = "*" if c["ci_full"][0] > 0 else ("-" if c["ci_full"][1] < 0 else " ")
            log(f"   [{name}/{shape:<12}] n={len(keep):<5} K=15 cold {c['cold_full']:.5f} "
                f"teacher {c['teacher_full']:.5f} | {c['teacher_minus_cold_full']:+.5f} "
                f"[{c['ci_full'][0]:+.5f},{c['ci_full'][1]:+.5f}] {star}")
            results.append(row)
            json.dump({"analysis": "m6c_diagnostic_openers", "split": "validation",
                       "test_touched": False, "results": results},
                      open(os.path.join(OUT, "m6c_diagnostic_openers.json"), "w"), indent=1)

    log("=" * 100)
    log("[summary] teacher minus cold at K=15, by DIAGNOSTIC opener shape (mean over clusters)")
    for shape in SHAPES:
        sub = [r for r in results if r["shape"] == shape]
        if not sub:
            continue
        g = [r["budgets"]["15"]["teacher_minus_cold_full"] for r in sub]
        sep = sum(1 for r in sub if r["budgets"]["15"]["ci_full"][0] > 0)
        neg = sum(1 for r in sub if r["budgets"]["15"]["ci_full"][1] < 0)
        log(f"  {shape:<13} clusters={len(sub)}  mean {np.mean(g):+.5f}  "
            f"positive {sum(1 for x in g if x > 0)}/{len(g)}  separating +{sep} -{neg}")
    log("[done]")


if __name__ == "__main__":
    main()
