r"""M4b STAGE 3 -- ARM C: the distilled policy against the cold tree and the dedicated teachers.
VALIDATION ONLY. The 10,000 canonical TEST users are never built, loaded or named here.

This completes the comparison the author asked for: at the opener states, three arms side by side.

  A  cold      the incumbent COLD TREE (depth 11), routed by the user's own answers
  B  teacher   that anchor's DEDICATED teacher, routed the same way
  C  policy    the SINGLE distilled policy, picking its OWN questions from the full pool

The anchor is folded into ALL THREE arms and is free everywhere, so nothing here is the anchor's own
contribution. Budgets K in {1,2,4,8,15} count ASKED questions. Paired percentile bootstrap over
per-user differences, 10,000 resamples, seed 98765. NaN tail pairs are dropped, never imputed.

TWO THINGS THIS RUN REPORTS THAT M4A COULD NOT
-----------------------------------------------
1. HELD-OUT FIDELITY. The policy was distilled on TRAIN users; validation users are held out from it
   entirely. Top-1 agreement with each teacher at that teacher's own validation states therefore
   measures imitation that GENERALISES, not replay. E1's precedent: 1.0000 at all 16 positions, but
   that was one tree and the states were the tree's own.
2. THE ARM-C DIRECTION IS PREDICTED, NOT DISCOVERED. E1 established that a distilled policy
   reproduces its teacher at best and never exceeds it, so arm C is expected AT OR BELOW arm B. The
   claim being tested is not "C beats B" -- it is that ONE parametric object holds all seven teachers
   at once, which is what makes the forest deployable. C materially below B is the falsifier.

WHY THE COMPARISON STILL FAVOURS THE TEACHERS (author, 2026-08-19)
------------------------------------------------------------------
Each teacher was fitted on precisely the cohort it is scored on; the cold tree was fitted on all
10,000 and merely covers it. Arms A and B are both in-sample here and B is the more specialised, so
B's margin over A is an UPPER BOUND. Arm C is the only arm whose parameters never saw these users.
That asymmetry runs AGAINST arm C and must be stated wherever this table is quoted.
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
TOTAL_L = 15
VAL_M0_FULL = 0.15686538118354684
ANCHORS = {"arthouse": 68, "dialogue": 275, "marvel": 581,
           "gore": 417, "teenagers": 922, "inspiring": 503}
BUDGETS = [1, 2, 4, 8, 15]


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


def load_tree(path):
    d = json.load(open(path, encoding="utf-8"))
    nodes = d.get("nodes")
    if isinstance(nodes, list):
        nodes = {r["nid"]: r for r in nodes}
    return d, nodes


def policy_rollout(E, pol, phi_ext, pool, rows, cid, a_phi_idx, a_pool_idx, L=TOTAL_L,
                   base_extra=None):
    """The policy's OWN interview, with the anchor folded and carried as an answered question.

    Mirrors `policy_net.rollout` (same scorer, same answer rule, same scoring funnel) with three
    differences the seeded setting requires: the initial latent already has the anchor folded, the
    anchor occupies history slot 0 with a LIKE outcome, and the anchor is masked out of the candidate
    table for the whole interview.
    """
    from strategy_channel_suite import genv_eval_multi
    from ndcg_tree import branch_of_per_user
    n, P = len(rows), len(pool)
    seeded = cid is not None
    asked = np.full((n, L), -1, np.int64)
    outs = np.full((n, L), -1, np.int8)
    amask = np.zeros((n, P), bool)
    if seeded and a_pool_idx >= 0:
        amask[:, a_pool_idx] = True
    # `base_extra` folds evidence WITHOUT announcing it in the anchor slot (M4c arm 2)
    base_c = ([cid] if seeded else []) + list(base_extra or [])
    z = np.asarray(genv_eval_multi(E, rows, [[] for _ in range(n)], [list(base_c) for _ in range(n)],
                                   score=False, return_z=True)[-1], np.float32)
    phi_cand = torch.from_numpy(phi_ext[:P])
    pol.eval()
    for t in range(L):
        hq = np.zeros((n, TOTAL_L), np.int64)
        ho = np.zeros((n, TOTAL_L), np.int64)
        hm = np.zeros((n, TOTAL_L), np.float32)
        s0 = 0
        if seeded:
            hq[:, 0] = a_phi_idx
            ho[:, 0] = 0
            hm[:, 0] = 1.0
            s0 = 1
        for j in range(t):
            hq[:, s0 + j] = asked[:, j]
            ho[:, s0 + j] = outs[:, j]
            hm[:, s0 + j] = 1.0
        pos = np.tile(np.asarray([[t, TOTAL_L - t]], np.float32), (n, 1))
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
            iseq = [[pool[int(j)][1] for j in asked[u, :t + 1] if pool[int(j)][0] == 0]
                    for u in range(n)]
            cseq = [list(base_c) + [pool[int(j)][1] for j in asked[u, :t + 1] if pool[int(j)][0] == 1]
                    for u in range(n)]
            z = np.asarray(genv_eval_multi(E, rows, iseq, cseq, score=False, return_z=True)[-1],
                           np.float32)
    assert (asked >= 0).all() and (outs >= 0).all()
    assert all(len(set(asked[u].tolist())) == L for u in range(min(n, 2000))), \
        "a question was asked twice inside one interview"
    return [[tuple(pool[int(j)]) for j in asked[u]] for u in range(n)]


def main():
    ap = argparse.ArgumentParser(description="M4b stage 3 -- arm C against the trees, on validation")
    ap.add_argument("--ckpt", default=os.path.join(OUT, "m4b_policy.pt"))
    ap.add_argument("--total_L", type=int, default=TOTAL_L)
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--i_have_author_goahead", action="store_true")
    a = ap.parse_args()
    assert a.i_have_author_goahead, "REFUSING TO RUN without --i_have_author_goahead"
    torch.set_num_threads(a.threads)
    np.random.seed(SEED)

    from strategy_channel_suite import (build_greedy_env, contract_selftest, genv_mean,
                                        genv_eval_multi, CERT_TOWER, CERT_CONC)
    from ndcg_tree import leak_assert, build_pool
    from policy_net import Policy, build_phi
    from tree_route import route

    log("#" * 100)
    log("# M4b STAGE 3 -- ARM C: the distilled policy vs the cold tree and the teachers. VALIDATION.")
    log("#" * 100)

    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    log(f"[ckpt] epoch {ck['epoch']} arch={ck['arch']} state={ck['state']} "
        f"dev top-1 " + " ".join(f"{k}={v:.4f}" for k, v in ck["dev_top1"].items()))

    E = build_greedy_env(CERT_TOWER, "validation", CERT_CONC, "m4b_eval")
    contract_selftest(E, E["rows"])
    leak_assert(E, E["rows"], "m4b_eval/val")
    rows_all = np.asarray(E["rows"], np.int64)
    f0, _ = genv_mean(E, [], rows_all)
    assert round(f0, 8) == round(VAL_M0_FULL, 8), f"CANONICAL SNAP FAILED: {f0!r}"
    log(f"[gate] canonical m=0 snap full={f0!r} -- exact")

    pool, _ip, _cp = build_pool(E, 500, 500)
    assert [list(q) for q in pool] == ck["pool"], \
        "the evaluation pool is NOT the pool the policy was trained on"
    ext = [tuple(q) for q in ck["ext"]]
    norm = (np.asarray(ck["norm"][0], np.float32), np.asarray(ck["norm"][1], np.float32))
    phi_ext, _ = build_phi(E, ext, norm=norm)          # the TRAINING pool's standardisation
    pol = Policy(ck["d_z"], ck["d_phi"], ck["n_pool"], arch=ck["arch"], state=ck["state"],
                 use_pos=True)
    pol.load_state_dict(ck["sd"])
    pol.eval()

    static = [(int(x), int(y)) for x, y in
              json.load(open(os.path.join(B, "greedy_static_policyB_combined.json")))["sequence"]]
    _cd, cold_nodes = load_tree(os.path.join(B, "coldtree_v2.json"))
    V, F = np.asarray(E["V"]), np.asarray(E["foldable"])
    endorse = (F > 0) & (V > 0)
    results = []

    for name, cid in ANCHORS.items():
        tpath = os.path.join(B, f"m3v2_{name}.json")
        if not os.path.exists(tpath):
            log(f"[skip] {name}: no teacher artifact")
            continue
        _td, t_nodes = load_tree(tpath)
        cohort = rows_all[endorse[rows_all, cid]]
        log(f"[{name}] cohort {len(cohort)}")

        s_cold, d_cold = route(E, cohort, cold_nodes, TOTAL_L, static=static, mask=[(1, cid)])
        s_teach, d_teach = route(E, cohort, t_nodes, TOTAL_L, static=static, mask=[(1, cid)])
        s_pol = policy_rollout(E, pol, phi_ext, pool, cohort, cid,
                               ck["a_phi"][name], ck["a_pool"][name])

        # Held-out fidelity: does the policy pick the teacher's question at the teacher's own
        # validation states? These users were never trained on. Reported PER POSITION, because root
        # agreement of 1.0000 with a K=15 shortfall can only mean the trajectories separate later,
        # and where they separate is the diagnosis. `never_deviates` is E1's statistic: the fraction
        # of users whose whole interview is the teacher's, which is what bounds the achievable gap.
        by_pos = [float(np.mean([1.0 if tuple(p[k]) == tuple(t[k]) else 0.0
                                 for p, t in zip(s_pol, s_teach)])) for k in range(a.total_L)]
        never = float(np.mean([1.0 if [tuple(x) for x in p] == [tuple(x) for x in t] else 0.0
                               for p, t in zip(s_pol, s_teach)]))
        agree = by_pos[0]
        log(f"   [{name}] agreement by position " + " ".join(f"{v:.3f}" for v in by_pos))
        log(f"   [{name}] whole interview identical to the teacher for {never:.4f} of users")

        row = {"anchor": name, "concept": cid, "cohort": int(len(cohort)),
               "fallback_cold": d_cold["n_fallback"], "fallback_teacher": d_teach["n_fallback"],
               "policy_matches_teacher_at_root": agree, "agreement_by_position": by_pos,
               "frac_never_deviates": never, "budgets": {}}
        for K in BUDGETS:
            def ev(seqs):
                iseq, cseq = [], []
                for s in seqs:
                    ask = s[:K]
                    iseq.append(list(dict.fromkeys([q[1] for q in ask if q[0] == 0])))
                    cseq.append(list(dict.fromkeys([cid] + [q[1] for q in ask if q[0] == 1])))
                full, tail = genv_eval_multi(E, cohort, iseq, cseq)
                return np.asarray(full, np.float64), np.asarray(tail, np.float64)

            cf, ct = ev(s_cold)
            tf, tt = ev(s_teach)
            pf, pt = ev(s_pol)
            mpc, lpc, hpc = paired_boot(pf - cf)
            mpt, lpt, hpt = paired_boot(pf - tf)
            row["budgets"][str(K)] = {
                "cold_full": float(np.nanmean(cf)), "teacher_full": float(np.nanmean(tf)),
                "policy_full": float(np.nanmean(pf)),
                "cold_tail": float(np.nanmean(ct)), "teacher_tail": float(np.nanmean(tt)),
                "policy_tail": float(np.nanmean(pt)),
                "policy_minus_cold_full": mpc, "ci_policy_minus_cold": [lpc, hpc],
                "policy_minus_teacher_full": mpt, "ci_policy_minus_teacher": [lpt, hpt],
                "policy_minus_cold_tail": paired_boot(pt - ct)[0],
                "policy_minus_teacher_tail": paired_boot(pt - tt)[0]}
            log(f"   [{name}] K={K:<3} cold {np.nanmean(cf):.5f} teacher {np.nanmean(tf):.5f} "
                f"policy {np.nanmean(pf):.5f} | P-C {mpc:+.5f} [{lpc:+.5f},{hpc:+.5f}] | "
                f"P-T {mpt:+.5f} [{lpt:+.5f},{hpt:+.5f}]")
        results.append(row)
        json.dump({"analysis": "m4b_arm_c", "split": "validation", "test_touched": False,
                   "ckpt_epoch": ck["epoch"], "results": results},
                  open(os.path.join(OUT, "m4b_armc.json"), "w"), indent=1)

    log("=" * 100)
    log("[summary] K=15, anchor folded in ALL THREE arms (validation)")
    for r in results:
        c = r["budgets"]["15"]
        log(f"  {r['anchor']:<11} n={r['cohort']:<5} cold {c['cold_full']:.5f} "
            f"teacher {c['teacher_full']:.5f} policy {c['policy_full']:.5f} | "
            f"P-C {c['policy_minus_cold_full']:+.5f} P-T {c['policy_minus_teacher_full']:+.5f} | "
            f"root agreement with teacher {r['policy_matches_teacher_at_root']:.4f}")
    log("[done]")


if __name__ == "__main__":
    main()
