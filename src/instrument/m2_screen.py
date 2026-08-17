r"""M2 -- THE SCREEN. Does an opener change which question should be asked next?

Design sheet `PAPER_B_POLICY_DESIGN.md` Part M (M2), `docs/design/DS_M1_STATE_MATCHING.md`.
Committed before running (HARD RULE 10). VALIDATION ONLY -- the 10,000 canonical TEST users are
never built, never loaded, never named.

THE QUESTION
------------
A user states an opener (one of the six anchors mined in M1). Their evidence is folded, so they now
sit at a state the cold tree never reaches. At THAT state, is there a question better than the one
the cold tree would ask -- its root -- by enough to be worth a teacher?

This is the front-loaded bet of the whole programme. If no opener changes the best first question,
there is nothing for a forest to teach and the beat-the-tree line ends here.

COHORT = WHO STATES THE ANCHOR, NOT WHO IS SECRETLY IN THE CLUSTER
-----------------------------------------------------------------
Routing at deployment is by what a user SAYS. So a teacher's cohort is every validation user who
behaviourally endorses that anchor under the env's own SEL rule -- the same rule M1 used on train.
Cluster membership was the device for FINDING the anchor; it is not the routing signature.

POWER, DECLARED BEFORE THE RUN
------------------------------
A cohort is ~500 users. The tree's paired interval is +/-0.003 at n=10,000, so at n=500 it is
roughly +/-0.013: a +0.004 effect is NOT detectable per teacher. Therefore:
  * HEADLINE  = the POOLED paired test over all six teachers (n ~ 3,000), best-vs-root, with the
    per-user difference vector concatenated across teachers.
  * PER-TEACHER numbers are DESCRIPTIVE and are reported with their (wide) intervals.
  * Selection is on a FIT half, evaluation on the HELD-OUT half, so the reported difference is not
    the maximum of a search on its own data.

ARMS, at matched budget (the seed is free in both arms; exactly ONE question is asked)
  root        the cold tree's root question, asked on top of the seed. The incumbent.
  best        the candidate with the highest measured gain on the FIT half, asked on the HOLDOUT.
  shuffle     control: a candidate drawn at random per teacher, same budget. Must not beat root.
"""
from __future__ import annotations

import os
import sys

_H = os.path.dirname(os.path.abspath(__file__))
_R = os.path.abspath(os.path.join(_H, "..", ".."))
sys.path.insert(0, _H)

import argparse
import csv
import json
import time

import numpy as np
import torch

M1 = os.path.join(_R, "experiments", "battery", "m1")
OUT = os.path.join(_R, "experiments", "battery", "m2")
TAGS = os.path.join(_R, "data", "movielens", "genome-tags.csv")
SEED = 98765
VAL_M0_FULL = 0.15686538118354684
ROOT_Q = (1, 114)                    # the cold tree's root splitter (read from ndcgtree16.json below)


def log(m: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def paired_boot(d, n=10000, seed=SEED):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(n, len(d)))
    bs = d[idx].mean(1)
    return float(d.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def main() -> None:
    ap = argparse.ArgumentParser(description="M2 screen -- does the opener change the best question?")
    ap.add_argument("--anchors", default="gore,art house,dialogue driven,inspiring,teenagers,marvel")
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--i_have_author_goahead", action="store_true")
    a = ap.parse_args()
    assert a.i_have_author_goahead, "REFUSING TO RUN without --i_have_author_goahead"
    torch.set_num_threads(a.threads); np.random.seed(SEED)
    os.makedirs(OUT, exist_ok=True)

    from strategy_channel_suite import (build_greedy_env, contract_selftest, genv_mean,
                                        genv_eval_multi, CERT_TOWER, CERT_CONC)
    from ndcg_tree import build_pool, leak_assert

    log("#" * 100)
    log("# M2 SCREEN -- does a stated opener change the best next question? VALIDATION ONLY.")
    log("#" * 100)

    E = build_greedy_env(CERT_TOWER, "validation", CERT_CONC, "m2screen")
    contract_selftest(E, E["rows"])
    leak_assert(E, E["rows"], "m2/val")
    rows = np.asarray(E["rows"], np.int64)
    assert len(rows) == 10000
    f0, _t = genv_mean(E, [], rows)
    assert round(f0, 8) == round(VAL_M0_FULL, 8), f"CANONICAL SNAP FAILED: {f0!r}"
    log(f"[gate] canonical m=0 snap full={f0!r} -- exact")

    # The cold tree's root splitter. `ndcgtree16.json` stores leaf tables rather than the node
    # structure, so this is the constant `policy_asif.py` uses -- and L1 verified it behaviourally:
    # seeding it made the policy reproduce the tree's post-root arm for 100.00% of users.
    from policy_asif import ROOT_Q as ASIF_ROOT
    root = tuple(ASIF_ROOT)
    assert root == ROOT_Q, f"root constant drifted: {root} vs {ROOT_Q}"
    log(f"[tree] cold-tree root question = {root} (concept {root[1]}, verified by L1)")

    pool, _ip, _cp = build_pool(E, 500, 500)
    log(f"[pool] {len(pool)} candidates")

    tagname = {}
    with open(TAGS, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            tagname[int(r["tagId"])] = r["tag"].strip()
    tags = list(E["sh"]["tags"])
    byname = {}
    for cid, t in enumerate(tags):
        s = str(tagname.get(int(t), t)).lower()
        byname.setdefault(s, cid); byname.setdefault(s.replace("-", " "), cid)

    V, F = np.asarray(E["V"]), np.asarray(E["foldable"])
    endorse = (F > 0) & (V > 0)

    rng = np.random.default_rng(SEED)
    results, pooled_best, pooled_shuf = [], [], []

    for name in [x.strip() for x in a.anchors.split(",") if x.strip()]:
        cid = byname.get(name.lower())
        if cid is None:
            log(f"[skip] anchor {name!r} is not a concept in the bank")
            continue
        cohort = np.flatnonzero(endorse[rows, cid])          # who STATES it -- the routing signature
        if len(cohort) < 100:
            log(f"[skip] anchor {name!r}: only {len(cohort)} validation users state it")
            continue
        r_idx = rows[cohort]
        fit = np.zeros(len(cohort), bool)
        fit[rng.permutation(len(cohort))[: len(cohort) // 2]] = True
        log(f"[{name}] cohort {len(cohort)} val users state it · fit {int(fit.sum())} / "
            f"holdout {int((~fit).sum())}")

        seed_c = [[int(cid)] for _ in range(len(cohort))]
        seed_i = [[] for _ in range(len(cohort))]

        def arm(q):
            ch, it = int(q[0]), int(q[1])
            iseq = [list(s) + ([it] if ch == 0 else []) for s in seed_i]
            cseq = [list(s) + ([it] if ch == 1 else []) for s in seed_c]
            full, _tail = genv_eval_multi(E, r_idx, iseq, cseq)
            return np.asarray(full, np.float64)

        t0 = time.time()
        base = arm(root)
        cands = [q for q in pool if not (int(q[0]) == 1 and int(q[1]) == cid) and tuple(q) != tuple(root)]
        gains, best_q, best_g = {}, None, -9e9
        for n_done, q in enumerate(cands, 1):
            v = arm(q)
            g = float((v - base)[fit].mean())               # SELECT on the fit half only
            gains[tuple(q)] = g
            if g > best_g:
                best_g, best_q = g, tuple(q)
            if n_done % 200 == 0:
                log(f"   [{name}] {n_done}/{len(cands)} candidates · best so far {best_q} "
                    f"{best_g:+.5f} · {(time.time()-t0)/60:.1f} min")

        vb = arm(best_q)
        d_best = (vb - base)[~fit]                          # EVALUATE on the holdout half
        shuf_q = tuple(cands[int(rng.integers(len(cands)))])
        d_shuf = (arm(shuf_q) - base)[~fit]
        m, lo, hi = paired_boot(d_best)
        ms, los, his = paired_boot(d_shuf)
        log(f"[{name}] BEST {best_q} (fit gain {best_g:+.5f}) -> HOLDOUT {m:+.5f} [{lo:+.5f},{hi:+.5f}]")
        log(f"[{name}] shuffle {shuf_q} -> HOLDOUT {ms:+.5f} [{los:+.5f},{his:+.5f}]")
        results.append({"anchor": name, "concept": int(cid), "cohort": int(len(cohort)),
                        "root": list(root), "best": list(best_q), "fit_gain": best_g,
                        "holdout_mean": m, "holdout_ci": [lo, hi],
                        "shuffle": list(shuf_q), "shuffle_mean": ms, "shuffle_ci": [los, his],
                        "minutes": round((time.time() - t0) / 60, 1)})
        pooled_best.append(d_best); pooled_shuf.append(d_shuf)
        json.dump({"analysis": "m2_screen", "split": "validation", "test_touched": False,
                   "root": list(root), "results": results},
                  open(os.path.join(OUT, "m2_screen_partial.json"), "w"), indent=1)

    if pooled_best:
        db = np.concatenate(pooled_best); ds = np.concatenate(pooled_shuf)
        m, lo, hi = paired_boot(db); ms, los, his = paired_boot(ds)
        log("=" * 90)
        log(f"[POOLED n={len(db)}] best-minus-root {m:+.5f} [{lo:+.5f}, {hi:+.5f}]")
        log(f"[POOLED n={len(ds)}] shuffle-minus-root {ms:+.5f} [{los:+.5f}, {his:+.5f}]")
        log(f"[VERDICT] {'SEPARATES from zero' if lo > 0 else 'does NOT separate from zero'}")
        json.dump({"analysis": "m2_screen", "split": "validation", "test_touched": False,
                   "root": list(root), "results": results,
                   "pooled": {"n": int(len(db)), "best_mean": m, "best_ci": [lo, hi],
                              "shuffle_mean": ms, "shuffle_ci": [los, his]}},
                  open(os.path.join(OUT, "m2_screen.json"), "w"), indent=1)
    log("[done]")


if __name__ == "__main__":
    main()
