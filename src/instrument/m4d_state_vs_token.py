r"""M4d -- DOES THE POLICY ROUTE ON THE STATE, OR ONLY ON THE ANNOUNCED TOKEN?

Author's question, 2026-08-19: *"what happens if a user comes with a state that resembles one of the
clusters but slightly different? it will not figure out a question that is relevant for that
cluster?"*

M4c answered half of it: a user endorsing a cluster-ADJACENT concept, folded but not announced, gets
the identical cold interview. But that left the premise unverified -- maybe the adjacent fold never
moved the state near the cluster in the first place, in which case nothing was actually tested.

THE DECISIVE CONTROL, and it needs no new cohorts. Take a TAUGHT anchor's OWN cohort -- the exact
users its teacher was fitted on, the state as close to the cluster as a state can possibly be -- and
fold the anchor WITHOUT announcing it in the history slot.

  if the policy still produces that teacher's interview  -> it routes on the STATE, and M4c's null is
                                                            about adjacency, not about routing
  if it falls back to the cold interview                 -> it routes ENTIRELY on the announced token
                                                            and the latent carries no routing signal
                                                            at all. Then the answer to the author is
                                                            an unambiguous NO, and it is a property of
                                                            the encoding, not of the clusters.

ARMS, per anchor cohort (the anchor is folded and free in every one):
  teacher      that anchor's dedicated tree, routed by the user's own answers
  cold         the incumbent cold tree
  announced    the policy WITH the anchor in the history slot   (M4b: reproduces the teacher)
  fold-only    the policy WITHOUT it -- the same evidence, silent

GEOMETRY, reported alongside so the premise is checked rather than assumed: the distance between the
anchor-folded state centroid and the cold centroid, against the spread of the cohort itself. If
folding an anchor barely moves the latent, "the state resembles the cluster" was never true and the
routing question is moot -- that too is an answer, and a different one.

VALIDATION ONLY. The 10,000 canonical TEST users are never built, loaded or named.
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
    ap = argparse.ArgumentParser(description="M4d -- state routing vs token routing")
    ap.add_argument("--ckpt", default=os.path.join(OUT, "m4b_policy.pt"))
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--i_have_author_goahead", action="store_true")
    a = ap.parse_args()
    assert a.i_have_author_goahead, "REFUSING TO RUN without --i_have_author_goahead"
    torch.set_num_threads(a.threads)
    np.random.seed(SEED)

    from strategy_channel_suite import (build_greedy_env, genv_mean, genv_eval_multi,
                                        CERT_TOWER, CERT_CONC)
    from ndcg_tree import build_pool
    from policy_net import Policy, build_phi
    from tree_route import route
    from m4b_eval import load_tree, policy_rollout

    log("#" * 100)
    log("# M4d -- DOES THE POLICY ROUTE ON THE STATE OR ONLY ON THE ANNOUNCED TOKEN? VALIDATION.")
    log("#" * 100)
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)

    E = build_greedy_env(CERT_TOWER, "validation", CERT_CONC, "m4d")
    rows_all = np.asarray(E["rows"], np.int64)
    f0, _ = genv_mean(E, [], rows_all)
    assert round(f0, 8) == round(VAL_M0_FULL, 8), f"CANONICAL SNAP FAILED: {f0!r}"
    log(f"[gate] canonical m=0 snap full={f0!r} -- exact")
    pool, _i, _c = build_pool(E, 500, 500)
    assert [list(q) for q in pool] == ck["pool"], "pool moved since training"
    pidx = {tuple(q): i for i, q in enumerate(pool)}
    n_pool = len(pool)
    ext = [tuple(q) for q in ck["ext"]]
    norm = (np.asarray(ck["norm"][0], np.float32), np.asarray(ck["norm"][1], np.float32))
    phi_ext, _ = build_phi(E, ext, norm=norm)
    pol = Policy(ck["d_z"], ck["d_phi"], n_pool, arch=ck["arch"], state=ck["state"], use_pos=True)
    pol.load_state_dict(ck["sd"])
    pol.eval()

    static = [(int(x), int(y)) for x, y in
              json.load(open(os.path.join(B, "greedy_static_policyB_combined.json")))["sequence"]]
    _cd, cold_nodes = load_tree(os.path.join(B, "coldtree_v2.json"))
    V, F = np.asarray(E["V"]), np.asarray(E["foldable"])
    endorse = (F > 0) & (V > 0)
    results = []

    for name, cid in ANCHORS.items():
        _td, t_nodes = load_tree(os.path.join(B, f"m3v2_{name}.json"))
        cohort = rows_all[endorse[rows_all, cid]]

        # --- geometry: does folding the anchor actually move the state? -----------------------
        z0 = np.asarray(genv_eval_multi(E, cohort, [[] for _ in cohort], [[] for _ in cohort],
                                        score=False, return_z=True)[-1], np.float64)
        za = np.asarray(genv_eval_multi(E, cohort, [[] for _ in cohort], [[cid] for _ in cohort],
                                        score=False, return_z=True)[-1], np.float64)
        shift = float(np.linalg.norm(za.mean(0) - z0.mean(0)))
        spread = float(np.mean(np.linalg.norm(z0 - z0.mean(0), axis=1)))
        cos = float(np.dot(za.mean(0), z0.mean(0)) /
                    max(np.linalg.norm(za.mean(0)) * np.linalg.norm(z0.mean(0)), 1e-9))

        seq_teach, _ = route(E, cohort, t_nodes, TOTAL_L, static=static, mask=[(1, cid)])
        seq_cold, _ = route(E, cohort, cold_nodes, TOTAL_L, static=static, mask=[(1, cid)])
        seq_ann = policy_rollout(E, pol, phi_ext, pool, cohort, cid,
                                 ck["a_phi"][name], ck["a_pool"][name])
        seq_fold = policy_rollout(E, pol, phi_ext, pool, cohort, None, 0, -1,
                                  L=TOTAL_L, base_extra=[cid])

        def agree(x, y, whole=False):
            if whole:
                return float(np.mean([[tuple(q) for q in p] == [tuple(q) for q in t]
                                      for p, t in zip(x, y)]))
            return float(np.mean([tuple(p[0]) == tuple(t[0]) for p, t in zip(x, y)]))

        row = {"anchor": name, "concept": cid, "cohort": int(len(cohort)),
               "geometry": {"anchor_fold_shift": shift, "cohort_spread": spread,
                            "shift_over_spread": shift / max(spread, 1e-9), "cosine": cos},
               "announced_vs_teacher_root": agree(seq_ann, seq_teach),
               "announced_vs_teacher_whole": agree(seq_ann, seq_teach, True),
               "foldonly_vs_teacher_root": agree(seq_fold, seq_teach),
               "foldonly_vs_teacher_whole": agree(seq_fold, seq_teach, True),
               "foldonly_vs_cold_root": agree(seq_fold, seq_cold),
               "foldonly_vs_cold_whole": agree(seq_fold, seq_cold, True),
               "budgets": {}}

        def ev(seqs, K):
            iseq, cseq = [], []
            for s in seqs:
                ask = s[:K]
                iseq.append(list(dict.fromkeys([q[1] for q in ask if q[0] == 0])))
                cseq.append(list(dict.fromkeys([cid] + [q[1] for q in ask if q[0] == 1])))
            full, tail = genv_eval_multi(E, cohort, iseq, cseq)
            return np.asarray(full, np.float64), np.asarray(tail, np.float64)

        for K in BUDGETS:
            tf, _tt = ev(seq_teach, K)
            cf, _ct = ev(seq_cold, K)
            af, _af2 = ev(seq_ann, K)
            ff, _ff2 = ev(seq_fold, K)
            m1, l1, h1 = paired_boot(ff - cf)
            m2, l2, h2 = paired_boot(ff - tf)
            row["budgets"][str(K)] = {
                "teacher": float(np.nanmean(tf)), "cold": float(np.nanmean(cf)),
                "announced": float(np.nanmean(af)), "foldonly": float(np.nanmean(ff)),
                "foldonly_minus_cold": m1, "ci_fc": [l1, h1],
                "foldonly_minus_teacher": m2, "ci_ft": [l2, h2]}
        log(f"[{name}] n={len(cohort):<5} | anchor fold moves the state {shift:.4f} "
            f"({shift / max(spread, 1e-9):.2f}x the cohort spread, cos {cos:.4f})")
        log(f"    announced -> teacher root {row['announced_vs_teacher_root']:.3f} "
            f"whole {row['announced_vs_teacher_whole']:.3f}   |   "
            f"fold-only -> teacher root {row['foldonly_vs_teacher_root']:.3f} "
            f"whole {row['foldonly_vs_teacher_whole']:.3f}, -> cold root "
            f"{row['foldonly_vs_cold_root']:.3f} whole {row['foldonly_vs_cold_whole']:.3f}")
        c = row["budgets"]["15"]
        log(f"    K=15 teacher {c['teacher']:.5f} cold {c['cold']:.5f} announced "
            f"{c['announced']:.5f} fold-only {c['foldonly']:.5f} | "
            f"fold-only-cold {c['foldonly_minus_cold']:+.5f} "
            f"fold-only-teacher {c['foldonly_minus_teacher']:+.5f}")
        results.append(row)
        json.dump({"analysis": "m4d_state_vs_token", "split": "validation", "test_touched": False,
                   "results": results}, open(os.path.join(OUT, "m4d_state_vs_token.json"), "w"),
                  indent=1)

    log("=" * 100)
    ar = np.mean([r["foldonly_vs_teacher_root"] for r in results])
    ac = np.mean([r["foldonly_vs_cold_root"] for r in results])
    log(f"[summary] with the anchor FOLDED BUT NOT ANNOUNCED, on the teacher's OWN cohort: "
        f"root agreement with the teacher {ar:.3f}, with the cold tree {ac:.3f}")
    log("[done]")


if __name__ == "__main__":
    main()
