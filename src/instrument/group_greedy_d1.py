"""Depth-1 group-conditioned greedy — the ONE-BRANCH-POINT control for Act 1.

WHY. Our greedy static order is the only arm in the battery that maximises the metric we report, and
it is also the only static arm. So every adaptive-vs-static comparison in the battery confounds two
things: "does conditioning on the answer help" and "is this arm even optimising the reported
quantity". This run isolates conditioning. Same objective (cohort-mean held-out full NDCG@10), same
optimiser (`greedy_certified`, lazy CELF), same pool (top-500 items by popularity, the identical
`_ranked_pool` call the global build makes), same cohort — ONE branch point, on the answer to the
question the global greedy itself chose first.

CONTRACT, stated so it cannot drift:
  * Cohort = the canonical 10,000 VALIDATION users (`split="validation"`), restricted as everywhere
    else to rows with a non-empty target. FORCED, not a choice: `data/ml-25m/proc/` has no
    train_tr/train_te, so train users have no fold-in/target split, and the tower was TRAINED on them
    — scoring held-out items for train users measures memorisation. Selection has always run on
    validation; this run keeps that.
  * q1 is READ from `experiments/battery/greedy_static_i26c.json`, arm "items-only", step 0. It is
    never re-derived here.
  * The partition is the harness's OWN answer rule, behavioural, off the graded fold-in lookup
    `sh["lvl_lookup"]` (targets already excluded): not present -> DID-NOT-ANSWER; level >=
    LIKE_MIN_LEVEL (7, i.e. >= 4.0 stars) -> LIKE; below -> DISLIKE. No user is dropped, capped or
    sampled (HARD RULE 1): the three groups partition the cohort exactly, and this is asserted.
  * Each group then runs `greedy_certified` for 15 further steps with prefix=[q1] and q1 removed from
    the pool. Its sequence is [q1] + its own 15.
  * VALIDATION ONLY. The TEST cohort is not built, not loaded and not touched. That is a separate,
    author-authorised step.

SANITY CHECK BUILT IN. On the selection cohort a group's own order can never be worse than the global
order on that group's users — the global order is inside the search space the group's greedy explored
(modulo CELF's own guarantee, which is exact for the argmax at each step, not for the whole
sequence). Both are measured per group and a violation is logged as a screaming FAIL, not silently.

PERSISTENCE (HARD RULE 11a). Everything is written to `*_partial.json` after EVERY pick, not at the
end. A kill costs one pick.
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

from strategy_channel_suite import (log, build_greedy_env, set_active_rows, greedy_certified,
                                    genv_mean, genv_answered, contract_selftest, _ranked_pool,
                                    CERT_TOWER, CERT_CONC, BUDGETS, pool_contract, resume_ok)
from i26_encoder import LIKE_MIN_LEVEL

LABEL = "groupgreedy_d1"
# Per-dataset results dir, single source of truth (run_battery_phaseA keys it off
# CASPER_DATASET). Was hardcoded here, so a Goodreads run would have written on top of the
# certified ML-25M tree/greedy artifacts -- HARD RULE 11e.
from run_battery_phaseA import OUTDIR                  # noqa: E402
GLOBAL_JSON = os.path.join(OUTDIR, "greedy_static_i26c.json")
GROUPS = ("like", "dislike", "noanswer")


def _read_q1_and_global(path):
    """q1 and the full 16-step global items-only order, READ from the frozen global build."""
    blob = json.load(open(path))
    arm = blob["arms"]["items-only"]
    seq = [(int(a), int(b)) for a, b in arm["sequence"]]
    assert len(seq) == 16, f"global items-only order is {len(seq)} steps, expected 16"
    assert all(ch == 0 for ch, _ in seq), "the items-only arm contains a non-item question"
    assert blob["objective"] == "full", f"global build objective is {blob['objective']}, expected full"
    assert blob["item_bank"] == "pop" and blob["n_items_pool"] == 500, \
        f"global build pool is {blob['item_bank']}/{blob['n_items_pool']}, expected pop/500"
    return seq[0], seq, blob


def _partition(E, rows, q1):
    """LIKE / DISLIKE / DID-NOT-ANSWER on q1, by the harness's own behavioural rule."""
    ch, iid = q1
    assert ch == 0, "depth-1 branching is defined here on an ITEM question"
    lvl = E["lvl"]
    g = {k: [] for k in GROUPS}
    for r in rows:
        d = lvl[r]
        if iid not in d:
            g["noanswer"].append(int(r))
        elif int(d[iid]) >= LIKE_MIN_LEVEL:
            g["like"].append(int(r))
        else:
            g["dislike"].append(int(r))
    tot = sum(len(v) for v in g.values())
    assert tot == len(rows), f"partition lost users: {tot} != {len(rows)}"
    assert len(set(g["like"]) | set(g["dislike"]) | set(g["noanswer"])) == len(rows), \
        "partition is not disjoint"
    return g


def _curve(E, seq, rows):
    out = {}
    for q in BUDGETS:
        f, t = genv_mean(E, seq[:q], rows)
        ia, _ca = genv_answered(E, seq[:q], rows)
        out[str(q)] = {"full@10": f, "tail@10": t, "item_answered": round(ia, 3)}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot", default=CERT_TOWER)
    ap.add_argument("--concept_ckpt", default=CERT_CONC)
    ap.add_argument("--n_items", type=int, default=500)
    ap.add_argument("--objective", default="full", choices=["full", "tail"])
    ap.add_argument("--L", type=int, default=15, help="steps AFTER the fixed q1 (16 total)")
    ap.add_argument("--label", default=LABEL)
    ap.add_argument("--sanity", action="store_true",
                    help="end-to-end pipeline check: L=2 on the SMALLEST group from a 12-item pool. "
                         "Writes *_SANITY.json and is NEVER citable.")
    args = ap.parse_args()

    t00 = time.time()
    sfx = "_SANITY" if args.sanity else ""
    OUT = os.path.join(OUTDIR, f"{args.label}{sfx}.json")
    PARTIAL = os.path.join(OUTDIR, f"{args.label}{sfx}_partial.json")
    os.makedirs(OUTDIR, exist_ok=True)

    # ---- 2-phase startup validation: every FILE path exists before anything expensive runs -------
    for nm, p in (("snapshot", args.snapshot), ("concept_ckpt", args.concept_ckpt),
                  ("global greedy json", GLOBAL_JSON)):
        assert os.path.isabs(p) and os.path.exists(p), f"{nm} missing or not absolute: {p}"
    if args.sanity:
        log("#" * 100)
        log("# SANITY MODE: L=2 on the smallest group, 12-item pool. Pipeline check ONLY, NOT citable.")
        log("#" * 100)

    q1, global_seq, gblob = _read_q1_and_global(GLOBAL_JSON)
    log(f"[q1] read from {os.path.basename(GLOBAL_JSON)} (tower={gblob['tower']}, "
        f"conc={gblob['concept_ckpt']}): q1 = channel {q1[0]} item {q1[1]}; "
        f"global 16-step order = {[i for _, i in global_seq]}")
    assert os.path.basename(args.snapshot) == gblob["tower"], \
        f"tower mismatch: {os.path.basename(args.snapshot)} vs global build's {gblob['tower']}"
    assert os.path.basename(args.concept_ckpt) == gblob["concept_ckpt"], \
        f"concept ckpt mismatch vs the global build ({gblob['concept_ckpt']})"

    # ---- the ONE environment: validation. The test cohort is never built. ----------------------
    E = build_greedy_env(args.snapshot, "validation", args.concept_ckpt, f"{args.label}(val)")
    rows = E["rows"]
    contract_selftest(E, rows)
    log(f"[cohort] {len(rows)} canonical VALIDATION users (rows with a non-empty target). "
        f"TEST IS NOT BUILT AND NOT TOUCHED.")

    groups = _partition(E, rows, q1)
    sizes = {k: len(groups[k]) for k in GROUPS}
    log(f"[partition] on q1 = item {q1[1]} (rule: absent -> noanswer; level >= {LIKE_MIN_LEVEL} "
        f"(>= 4.0 stars) -> like; else dislike)")
    for k in GROUPS:
        log(f"[partition]   {k:9s} = {sizes[k]:6d} users ({100.0 * sizes[k] / len(rows):5.2f}%)")
    for k in GROUPS:
        if sizes[k] == 0:
            log(f"[partition] !!! GROUP '{k}' IS EMPTY -- reported, NOT merged and NOT dropped.")
        elif sizes[k] < 200:
            log(f"[partition] !!! GROUP '{k}' IS TINY ({sizes[k]}) -- reported, NOT merged, NOT dropped.")

    # ---- the pool: the SAME _ranked_pool call the global build makes, minus q1 ------------------
    item_tables = {"pop": np.asarray(E["ctx"].cnt, np.float64)}
    pool_all = [(0, int(i)) for i in _ranked_pool("pop", item_tables, args.n_items, "item")]
    assert q1 in pool_all, f"q1 {q1} is not in the top-{args.n_items} popularity pool"
    pool = [q for q in pool_all if q != q1]
    log(f"[pool] items-only, bank='pop', top-{args.n_items} -> {len(pool)} candidates after removing "
        f"q1 (top-5 {[i for _, i in pool_all[:5]]})")
    L = args.L
    if args.sanity:
        L = 2
        # The global head must stay IN the truncated pool, or the built-in >= check is not a check:
        # it would fail because the reference order was outside the search space, not because
        # anything is wrong.
        pool = list(dict.fromkeys([q for q in global_seq[1:1 + L]] + pool[:12]))

    sig = {"tower": os.path.basename(args.snapshot), "conc": os.path.basename(args.concept_ckpt),
           "objective": args.objective, "item_bank": "pop", "n_items": len(pool_all),
           "q1": [int(q1[0]), int(q1[1])], "L_after_q1": L, "split": "validation",
           "n_cohort": len(rows), "sizes": sizes, "sanity": bool(args.sanity),
           # 2026-08-12 audit: derived from the mask; a missing field refuses rather than passes.
           "pool_contract": pool_contract(E["pool"], rows, policy=E["pool_policy"])}
    part = json.load(open(PARTIAL)) if os.path.exists(PARTIAL) else {}
    if not resume_ok(part.get("sig"), sig, log, os.path.basename(PARTIAL)):
        part = {"sig": sig, "groups": {}}

    def flush():
        json.dump(part, open(PARTIAL, "w"), indent=2, default=float)

    order_groups = sorted(GROUPS, key=lambda k: sizes[k]) if args.sanity else list(GROUPS)
    if args.sanity:
        order_groups = order_groups[:1]

    for gname in order_groups:
        grows = groups[gname]
        if not grows:
            log(f"[{gname}] EMPTY GROUP -- no greedy to run. Recorded as empty.")
            part["groups"][gname] = {"n_users": 0, "empty": True}
            flush()
            continue
        if part["groups"].get(gname, {}).get("done"):
            log(f"[{gname}] already complete in the sidecar -- skipping")
            continue
        set_active_rows(E, grows)
        E["memo"] = {}                       # memo is keyed on the SEQUENCE only: it MUST NOT cross
        E["n_evals"] = 0; E["t_evals"] = 0.0  # cohorts. Reset with the cohort, every time.
        t0 = time.time()
        rec = {"n_users": len(grows), "q1": [int(q1[0]), int(q1[1])], "done": False}
        part["groups"][gname] = rec
        flush()

        def on_pick(seq, gains, nev, _g=gname, _r=rec, _t=t0):
            _r["steps_after_q1"] = [[int(a), int(b)] for a, b in seq]
            _r["sequence"] = [[int(q1[0]), int(q1[1])]] + _r["steps_after_q1"]
            _r["gains"] = [float(x) for x in gains]
            _r["n_evals"] = int(nev)
            _r["sec_per_eval"] = round(E["t_evals"] / max(E["n_evals"], 1), 3)
            _r["minutes"] = round((time.time() - _t) / 60, 2)
            flush()                          # HARD RULE 11(a): on disk after EVERY pick

        seq, gains, nev = greedy_certified(E, grows, pool, L, args.objective, gname,
                                           prefix=[q1], on_pick=on_pick)
        own = [q1] + seq
        spe = E["t_evals"] / max(E["n_evals"], 1)
        rec.update({"steps_after_q1": [[int(a), int(b)] for a, b in seq],
                    "sequence": [[int(a), int(b)] for a, b in own],
                    "gains": [float(g) for g in gains], "n_evals": int(nev),
                    "sec_per_eval": round(spe, 3),
                    "minutes": round((time.time() - t0) / 60, 2)})
        rec["curve_own"] = _curve(E, own, grows)
        rec["curve_global"] = _curve(E, global_seq, grows)
        # LENGTH-MATCHED, always. The check is "same budget, one conditioned on q1's answer and one
        # not"; comparing a 3-step sanity order against 16 global questions would fail by construction.
        jj = 0 if args.objective == "full" else 1
        K = len(own)
        o16 = genv_mean(E, own, grows)[jj]
        g16 = genv_mean(E, global_seq[:K], grows)[jj]
        rec["cmp_budget"] = K
        rec[f"own_minus_global@{K}"] = o16 - g16
        rec["sanity_pass"] = bool(o16 >= g16 - 1e-9)
        rec["done"] = True
        flush()
        log(f"[{gname}] DONE {len(grows)} users, {nev} evals, {spe:.3f}s/eval "
            f"({(time.time() - t0) / 60:.1f}m) | order {[i for _, i in own]}")
        if rec["sanity_pass"]:
            log(f"[{gname}] SANITY OK: own {o16:.5f} >= global {g16:.5f} at budget {K} on its own users "
                f"({o16 - g16:+.5f})")
        else:
            log("!" * 100)
            log(f"[{gname}] SANITY *** FAIL *** : the group's OWN greedy order scores BELOW the "
                f"GLOBAL order on the group's OWN selection users at budget {K}: own {o16:.5f} < global "
                f"{g16:.5f} ({o16 - g16:+.5f}). The global order is inside the search space, so "
                f"this cannot happen if the search, the cohort masking and the memo are correct. "
                f"DO NOT INTERPRET THIS GROUP.")
            log("!" * 100)

    out = {"analysis": "group_conditioned_greedy_depth1",
           "label": args.label, "SANITY": bool(args.sanity), "citable": not args.sanity,
           "split": "validation", "test_touched": False,
           "tower": os.path.basename(args.snapshot), "concept_ckpt": os.path.basename(args.concept_ckpt),
           "objective": args.objective, "item_bank": "pop", "n_items_pool": len(pool_all),
           "q1": [int(q1[0]), int(q1[1])], "q1_source": os.path.basename(GLOBAL_JSON),
           "global_sequence": [[int(a), int(b)] for a, b in global_seq],
           "answer_rule": f"lvl_lookup absent -> noanswer; level >= {LIKE_MIN_LEVEL} -> like; else dislike",
           "n_cohort": len(rows), "group_sizes": sizes, "budgets": list(BUDGETS),
           "groups": part["groups"], "sig": sig,
           # this run's ruler, derived (2026-08-12 audit). q1 above comes from GLOBAL_JSON and carries
           # ITS policy, not this one -- read `q1_source`'s own pool_contract before pairing them.
           "pool_policy": E["pool_policy"], "pool_contract": sig["pool_contract"],
           "seconds": round(time.time() - t00, 1),
           "baseline_sec_per_eval_10k": gblob.get("sec_per_eval_build")}
    json.dump(out, open(OUT, "w"), indent=2, default=float)
    log(f"[done] wrote {os.path.basename(OUT)} in {(time.time() - t00) / 60:.1f}m")
    log("[cost] s/eval by group vs the 13.543 s/eval global build at 10,000 users: " + " | ".join(
        f"{g}: n={part['groups'][g].get('n_users', 0)} "
        f"{part['groups'][g].get('sec_per_eval', float('nan'))}s"
        for g in part["groups"]))
    return out


if __name__ == "__main__":
    main()
