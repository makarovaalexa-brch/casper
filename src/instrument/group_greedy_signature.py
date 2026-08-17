r"""T1' -- THE STATED-GENRE TEACHER, REBUILT OVER THE TREE'S OWN ACTION SPACE.

Design sheet: `docs/design/PAPER_B_POLICY_DESIGN.md` Part L.2, and the measured reason this exists:
`docs/results/L2_FOREST.md`. VALIDATION ONLY; the parser has no `--split`; `test_touched=false`.

WHY
---
L2a measured that one policy can hold the cold tree and the stated-genre orders at the same time,
exactly, with no interference -- and that following the stated-genre teacher COSTS 0.026-0.033 full
NDCG against simply running the tree at the same states. The obvious suspect is the action space, not
the conditioning: the banked genre orders (`groupgreedy_genre15.json`) were searched over **500 items
only**, because in that experiment the genre answer was free and the comparison was item-for-item. The
tree spends most of its 16 turns on CONCEPTS.

So the question this file answers, and it is the one that decides whether the forest is worth
expanding: **given the tree's own 1,000-candidate pool and the tree's own budget, does conditioning on
a stated genre beat the tree inside that genre's cohort?**

  * if YES -- T1 failed for its action space, the conditioning is real, and T2/T3 (evidence-count and
    taste-popularity bands) are worth their search time.
  * if NO -- conditioning on a volunteered category genuinely does not pay against a cold interview
    that is already allowed to ask about concepts, and the forest stops at one teacher. Reported as
    the finding either way.

THE SEARCH
----------
For each teacher genre G: prefix = the G concept, already asked (it is what the user stated), then a
LAZY-GREEDY (CELF) 15-question order over the COMBINED pool -- the same 500 popular items + 500
answerable concepts the tree and the static comparator use, minus the G concept itself -- maximising
the cohort-mean held-out FULL NDCG@10. Identical objective, identical ruler, identical budget as the
tree; the ONLY difference is that the cohort is one stated-genre group and the interview opens on a
fact the user volunteered.

FIREWALL: the search runs on that group's **FIT users only** (the 8,000/2,000 by-user split, seed
98765, from `pu_labels.fit_holdout_split`). The banked genre orders of 2026-08-11 were searched on all
10,000 validation users; these are not, so a HOLDOUT comparison against them is clean on this side.

PERSISTENCE (HARD RULE 11): every pick is written to `<out>_partial.json` the moment it lands, and the
CELF heap is persisted under a contract signature so an interrupted search resumes for the price of
the picks it has not yet made rather than the 1,000-candidate seed pass.
"""
import os
import sys
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
sys.path.insert(0, _HERE)

import json
import time
import argparse
import numpy as np

from policy_net import OUTDIR, SEED, TOTAL_L, STATIC_JSON, STATIC_VAL_FULL, VAL_M0_FULL, log

OUT_DIR = os.path.join(OUTDIR, "forest")
FOREST_STATES = os.path.join(OUT_DIR, "forest_states.json")


def main():
    ap = argparse.ArgumentParser(description="T1': stated-genre orders over the COMBINED pool")
    ap.add_argument("--genres", default="", help="default: every teacher in forest_states.json, "
                                                 "largest cohort first")
    ap.add_argument("--L", type=int, default=TOTAL_L - 1, help="new questions after the stated genre")
    ap.add_argument("--objective", default="full", choices=("full", "tail"))
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--full_threads", action="store_true")
    ap.add_argument("--i_have_author_goahead", action="store_true")
    args = ap.parse_args()
    assert args.i_have_author_goahead, "REFUSING TO RUN without --i_have_author_goahead"
    assert "--split" not in sys.argv, "there is no --split here: this file is validation-only"
    t00 = time.time()

    from strategy_channel_suite import (build_greedy_env, contract_selftest, genv_eval, genv_mean,
                                        greedy_certified, CERT_TOWER, CERT_CONC)
    from ndcg_tree import build_pool, leak_assert
    import pu_labels as PL
    from group_greedy_genre import load_item_genres, partition_by_genre

    log("#" * 110)
    log("# T1' -- STATED-GENRE ORDERS OVER THE COMBINED POOL. VALIDATION ONLY, FIT USERS ONLY.")
    log("# The 10,000 canonical TEST users are NOT built, NOT loaded, NOT touched.")
    log("#" * 110)

    E = build_greedy_env(CERT_TOWER, "validation", CERT_CONC, "t1prime")
    contract_selftest(E, E["rows"])
    leak_assert(E, E["rows"], "t1prime/val")
    rows = E["rows"]
    assert len(rows) == 10000
    f0, t0_ = genv_mean(E, [], rows)
    assert round(f0, 8) == VAL_M0_FULL, f"CANONICAL SNAP FAILED: {f0!r}"
    log(f"[gate] canonical m=0 snap full={f0!r} tail={t0_!r} -- exact")
    pool, _ip, _cp = build_pool(E, 500, 500)
    static_seq = [(int(a), int(b)) for a, b in json.load(open(STATIC_JSON))["sequence"]]
    sf, _ = genv_eval(E, static_seq, rows)
    assert abs(float(np.nanmean(sf)) - STATIC_VAL_FULL) < 1e-12, "BANKED-HEADLINE ASSERT FAILED"
    log(f"[gate] greedy_static_policyB_combined reproduces EXACTLY: {float(np.nanmean(sf))!r}")

    fit, hold, fw = PL.fit_holdout_split(np.asarray(rows, np.int64))
    fs = set(int(x) for x in fit.tolist())
    log(f"[firewall] {len(fit):,} FIT / {len(hold):,} HOLDOUT, seed {fw['seed']}; the search sees "
        f"FIT only, and HOLDOUT is where the order is judged")

    sb = json.load(open(FOREST_STATES))
    teachers = [g for g in sb["teachers"]]
    gmap = {g: int(sb["per_teacher"][g]["concept_id"]) for g in teachers}
    gen, _v = load_item_genres(E["ctx"].ni)
    groups, _a, _b = partition_by_genre(E, rows, gen)
    want = [g.strip() for g in args.genres.split(",") if g.strip()] or teachers
    for g in want:
        assert g in gmap, f"{g!r} is not a teacher in {os.path.basename(FOREST_STATES)}"

    os.makedirs(OUT_DIR, exist_ok=True)
    summary = {}
    for g in want:
        grows = [int(r) for r in sorted(groups[g]) if int(r) in fs]
        cq = (1, int(gmap[g]))
        cand = [q for q in pool if tuple(q) != cq]
        stem = os.path.join(OUT_DIR, f"t1prime_{g.replace('-', '')}")
        log("=" * 100)
        log(f"=== {g}: {len(grows):,} FIT users, prefix {cq} (the stated genre, already asked), "
            f"{len(cand)} candidates, L={args.L}, objective {args.objective}")
        base_f, base_t = genv_mean(E, [cq], grows)
        log(f"[{g}] the stated genre alone: full {base_f:.5f} tail {base_t:.5f}")

        def on_pick(seq, gains, nev, _g=g, _stem=stem, _rows=grows, _cq=cq):
            with open(_stem + "_partial.json", "w") as fh:
                json.dump({"analysis": "t1prime_partial", "genre": _g, "test_touched": False,
                           "split": "validation", "cohort": "FIT", "n_users": len(_rows),
                           "prefix": [list(_cq)], "order": [[int(a), int(b)] for a, b in seq],
                           "gains": [float(x) for x in gains], "n_evals": int(nev)},
                          fh, indent=1)

        seq, gains, nev = greedy_certified(E, grows, cand, args.L, args.objective, f"t1prime/{g}",
                                           prefix=[cq], on_pick=on_pick,
                                           heap_path=stem + "_heap.json")
        full, tail = genv_mean(E, [cq] + list(seq), grows)
        blob = {"analysis": "t1prime_stated_genre_combined_pool", "genre": g, "citable": True,
                "design_sheet": "docs/design/PAPER_B_POLICY_DESIGN.md L.2", "test_touched": False,
                "split": "validation", "cohort": "FIT users of this genre group",
                "n_users": len(grows), "concept_id": int(gmap[g]),
                "prefix": [list(cq)], "order": [[int(a), int(b)] for a, b in seq],
                "gains": [float(x) for x in gains], "n_evals": int(nev),
                "pool": "combined 500 popular items + 500 answerable concepts, minus the genre "
                        "concept itself", "L": args.L, "objective": args.objective,
                "n_concepts_in_order": int(sum(1 for a, _b in seq if int(a) == 1)),
                "in_sample_full": float(full), "in_sample_tail": float(tail),
                "in_sample_warning": "these are SELECTION-cohort numbers; the citable comparison is "
                                     "the HOLDOUT one in the forest evaluation",
                "banked_item_only_order": json.load(open(
                    os.path.join(OUTDIR, "groupgreedy_genre15.json")))["groups"][g]["order"],
                "tower": E["tower"], "conc_ckpt": E["conc_ckpt"],
                "pool_contract": E["pool_contract"], "firewall": fw,
                "minutes": round((time.time() - t00) / 60, 2)}
        with open(stem + ".json", "w") as fh:
            json.dump(blob, fh, indent=2, default=float)
        summary[g] = {"full": float(full), "tail": float(tail), "n_evals": int(nev),
                      "n_concepts_in_order": blob["n_concepts_in_order"]}
        log(f"[{g}] DONE: in-sample full {full:.5f} tail {tail:.5f}; "
            f"{blob['n_concepts_in_order']}/{args.L} picks are concepts; {nev} evals; "
            f"{blob['minutes']:.1f}m elapsed")
    log(f"[done] {json.dumps(summary, default=float)} ({round((time.time() - t00) / 60, 2)}m)")


if __name__ == "__main__":
    main()
