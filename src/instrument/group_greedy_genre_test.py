"""Genre-split group-conditioned greedy, STAGE 2 (evaluate) -- label `groupgreedy_genre16_TEST`.

ONE-SHOT EVALUATION ON THE 10,000 CANONICAL TEST USERS. The analysis plan below was PRE-SPECIFIED
and COMMITTED before any test number existed (`docs/design/DS_GROUP_GREEDY_GENRE.md` §Stage 2). If a
result looks wrong it is reported as it stands, with a note on what would have been done
differently. Nothing in this file may be tuned after a test number is seen: not the threshold, not
the grouping rule, not the pool, not the budget, not the number of resamples.

WHAT IS CARRIED OVER, AND WHAT IS RECOMPUTED
  * CARRIED: the per-group 16-item orders built on VALIDATION by `group_greedy_genre.py`, read from
    `groupgreedy_genre16.json`. They are never rebuilt here -- that is the whole point.
  * RECOMPUTED on the test side: each test user's genre group, by the SAME rule, from their OWN
    fold-in ratings (`partition_by_genre`, imported, not re-implemented). The leak assertion is
    re-run: 0/10,000 test users may have a held-out target inside the grouping lookup.

THE TWO PRE-SPECIFIED ARMS (both reported, always)
  * ALL-GROUPS  -- every test user is asked their own group's validation order, including groups
                   whose order was fitted on a handful of validation users.
  * FALLBACK-200 -- test users whose genre group had FEWER THAN 200 VALIDATION users are asked the
                   GLOBAL order instead. This is the deployable policy: you do not ship an order
                   fitted on 3 people. The threshold is 200, fixed in advance, and it is the same
                   threshold stage 1 already uses to flag small groups. It is NOT a tuned parameter
                   and must never become one.

PRIMARY ENDPOINT (pre-specified): cohort-weighted group-order minus global-order, FULL NDCG@10 at
K=16, over all 10,000 test users, with a PAIRED percentile bootstrap CI.

INFERENCE: paired percentile bootstrap over the PER-USER DIFFERENCE vector, 10,000 resamples, seed
98765. Never the unpaired `1.96*sqrt(2)*SE` form -- that signature is what made the earlier "tie"
claim wrong (memory `paired-test-kills-the-tie-claim`). Reported at K in {1,2,4,8,16}, full and tail.

ALSO REPORTED (all decided in advance): the test-side genre distribution beside the validation one;
per-group TEST gain against VALIDATION group size; order overlap with the global order; and the
VALIDATION-to-TEST SHRINKAGE -- each group's validation gain beside its test gain. That last one is
the overfitting measurement and is the most likely thing to carry the finding.

CONTRACT: full AND tail NDCG@10, learned decoder bias, NO floor, policy-B candidate pool, canonical
head mask -- i.e. `genv_eval`, the same function stage 1 used, unchanged.
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

from strategy_channel_suite import (log, build_greedy_env, set_active_rows, genv_eval,
                                    genv_answered, contract_selftest, CERT_TOWER, CERT_CONC)
from group_greedy_genre import (load_item_genres, partition_by_genre, leak_assert, order_overlap,
                                UNIQUE_SID, SMALL_GROUP, NONE_GROUP, ITEM_BUDGETS, GENEROUS)

LABEL = "groupgreedy_genre16_TEST"
OUTDIR = os.path.join(_ROOT, "experiments", "battery")
STAGE1_JSON = os.path.join(OUTDIR, "groupgreedy_genre16.json")
N_BOOT = 10000
BOOT_SEED = 98765
ARMS = ("all_groups", "fallback_200")


def paired_bootstrap(d, n_boot=N_BOOT, seed=BOOT_SEED):
    """Percentile CI over the PER-USER paired difference vector. Users, not cells, are resampled."""
    d = np.asarray(d, np.float64)
    d = d[np.isfinite(d)]
    if len(d) == 0:
        return {"mean": float("nan"), "lo": float("nan"), "hi": float("nan"), "n": 0,
                "excludes_zero": False}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(n_boot, len(d)))
    bs = d[idx].mean(axis=1)
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return {"mean": float(d.mean()), "lo": float(lo), "hi": float(hi), "n": int(len(d)),
            "n_boot": n_boot, "seed": seed, "excludes_zero": bool(lo > 0 or hi < 0)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot", default=CERT_TOWER)
    ap.add_argument("--concept_ckpt", default=CERT_CONC)
    ap.add_argument("--label", default=LABEL)
    args = ap.parse_args()

    t00 = time.time()
    OUT = os.path.join(OUTDIR, f"{args.label}.json")
    PARTIAL = os.path.join(OUTDIR, f"{args.label}_partial.json")
    NPZ = os.path.join(OUTDIR, f"{args.label}_per_user.npz")

    for nm, p in (("snapshot", args.snapshot), ("concept_ckpt", args.concept_ckpt),
                  ("stage-1 build", STAGE1_JSON), ("unique_sid.txt", UNIQUE_SID)):
        assert os.path.isabs(p) and os.path.exists(p), f"{nm} missing or not absolute: {p}"

    s1 = json.load(open(STAGE1_JSON))
    assert s1.get("citable") and not s1.get("SANITY"), "stage 1 build is a SANITY artifact"
    assert s1["split"] == "validation", "stage 1 was not built on validation"
    assert os.path.basename(args.snapshot) == s1["tower"], "tower mismatch vs the stage-1 build"
    assert os.path.basename(args.concept_ckpt) == s1["concept_ckpt"], "concept ckpt mismatch"
    global_seq = [(int(a), int(b)) for a, b in s1["global_sequence"]]
    val_sizes = dict(s1["group_sizes"])
    orders = {}
    val_gain = {}
    for g, r in s1["groups"].items():
        assert r.get("done"), f"stage-1 group '{g}' did not finish -- refusing to evaluate on test"
        orders[g] = [(int(a), int(b)) for a, b in r["items"]]
        assert len(orders[g]) == 16, f"group '{g}' order is {len(orders[g])} items, expected 16"
        val_gain[g] = {K: r["curve_item_matched"][str(K)]["diff_full@10"]
                       for K in ITEM_BUDGETS if str(K) in r["curve_item_matched"]}
    log("#" * 108)
    log("# STAGE 2 -- ONE-SHOT TEST EVALUATION. The plan was pre-specified and committed.")
    log("# GENEROUS ASSUMPTION: " + GENEROUS)
    log(f"# Carrying {len(orders)} validation-built 16-item orders onto the TEST cohort.")
    log("#" * 108)

    E = build_greedy_env(args.snapshot, "test", args.concept_ckpt, f"{args.label}(test)")
    rows = E["rows"]
    contract_selftest(E, rows)
    log(f"[cohort] {len(rows)} canonical TEST users (rows with a non-empty target)")
    leak_assert(E, rows, "test")

    gen, _vocab = load_item_genres(E["ctx"].ni)
    tgroups, n_no_liked, n_ungenred = partition_by_genre(E, rows, gen)
    tsizes = {k: len(v) for k, v in tgroups.items()}
    log("[partition] TEST-side genre distribution beside VALIDATION:")
    log(f"[partition]   {'group':<14}{'test n':>9}{'val n':>9}{'test %':>9}{'val %':>9}   note")
    allg = sorted(set(tsizes) | set(val_sizes), key=lambda k: -(val_sizes.get(k, 0)))
    for k in allg:
        tn, vn = tsizes.get(k, 0), val_sizes.get(k, 0)
        note = []
        if vn == 0:
            note.append("NO VALIDATION ORDER -> global order used, arm-independent")
        if tn == 0:
            note.append("EMPTY ON TEST (reported, not merged)")
        if 0 < vn < SMALL_GROUP:
            note.append(f"val n < {SMALL_GROUP}: global order under FALLBACK-200")
        log(f"[partition]   {k:<14}{tn:>9}{vn:>9}{100.0 * tn / len(rows):>8.2f}%"
            f"{100.0 * vn / max(s1['n_cohort'], 1):>8.2f}%   {'; '.join(note)}")

    # ---- assemble the two arms: which order does each test user get? ---------------------------
    # Pre-specified, and written down before any test number is computed.
    assign = {a: {} for a in ARMS}            # arm -> group -> rows
    n_fallback = 0
    for g, grows in tgroups.items():
        has_order = g in orders
        assign["all_groups"].setdefault(g if has_order else "__global__", []).extend(grows)
        small = (not has_order) or val_sizes.get(g, 0) < SMALL_GROUP
        if small:
            n_fallback += len(grows)
            assign["fallback_200"].setdefault("__global__", []).extend(grows)
        else:
            assign["fallback_200"].setdefault(g, []).extend(grows)
    log(f"[arms] ALL-GROUPS: every test user gets their own group's validation order "
        f"({sum(1 for g in tgroups if g in orders)} groups with an order).")
    log(f"[arms] FALLBACK-200: {n_fallback}/{len(rows)} test users ({100.0 * n_fallback / len(rows):.2f}%) "
        f"fall in a group with fewer than {SMALL_GROUP} VALIDATION users and are asked the GLOBAL "
        f"order instead. Threshold fixed in advance; never tuned.")

    # ---- score. Per-user vectors, so every CI is paired. ----------------------------------------
    part = json.load(open(PARTIAL)) if os.path.exists(PARTIAL) else {"groups": {}}
    banked = {}
    per_group = {}
    # The global order's per-user scores, once, over the WHOLE test cohort -- the shared comparator.
    set_active_rows(E, rows)
    E["memo"] = {}; E["n_evals"] = 0; E["t_evals"] = 0.0
    gl = {}
    for K in ITEM_BUDGETS:
        f, t = genv_eval(E, global_seq[:K], rows)
        gl[K] = (f, t)
        banked[f"__global__|{K}|full"] = f.astype(np.float32)
        banked[f"__global__|{K}|tail"] = t.astype(np.float32)
        log(f"[global] K={K:<3} full {np.nanmean(f):.5f}  tail {np.nanmean(t):.5f} "
            f"({E['t_evals'] / max(E['n_evals'], 1):.2f}s/eval)")
    banked["__rows__"] = np.asarray(rows, np.int64)
    pos = {int(r): i for i, r in enumerate(rows)}
    json.dump({"stage": 2, "global_done": True}, open(PARTIAL, "w"))

    # Each group's own order, scored on that group's TEST users only.
    for g in sorted(tgroups, key=lambda k: -tsizes[k]):
        grows = tgroups[g]
        if not grows or g not in orders:
            per_group[g] = {"n_test": len(grows), "has_order": g in orders,
                            "note": "empty on test" if not grows else "no validation order"}
            continue
        set_active_rows(E, grows)
        E["memo"] = {}; E["n_evals"] = 0; E["t_evals"] = 0.0
        t0 = time.time()
        rec = {"n_test": len(grows), "n_val": val_sizes.get(g, 0), "has_order": True,
               "small_on_val": bool(val_sizes.get(g, 0) < SMALL_GROUP),
               "order": [int(i) for _, i in orders[g]], "curve": {}}
        jj = np.asarray([pos[int(r)] for r in grows], np.int64)
        for K in ITEM_BUDGETS:
            of, ot = genv_eval(E, orders[g][:K], grows)
            lf, lt = gl[K][0][jj], gl[K][1][jj]
            banked[f"{g}|{K}|own_full"] = of.astype(np.float32)
            banked[f"{g}|{K}|own_tail"] = ot.astype(np.float32)
            ia, _ = genv_answered(E, orders[g][:K], grows)
            rec["curve"][str(K)] = {
                "group_full@10": float(np.nanmean(of)), "group_tail@10": float(np.nanmean(ot)),
                "global_full@10": float(np.nanmean(lf)), "global_tail@10": float(np.nanmean(lt)),
                "diff_full@10": float(np.nanmean(of - lf)), "diff_tail@10": float(np.nanmean(ot - lt)),
                "paired_ci_full": paired_bootstrap(of - lf),
                "val_diff_full@10": val_gain.get(g, {}).get(K),
                "shrinkage_full@10": (None if val_gain.get(g, {}).get(K) is None
                                      else float(np.nanmean(of - lf)) - val_gain[g][K]),
                "item_answered": round(ia, 3),
                "overlap": order_overlap(orders[g], global_seq, K)}
        banked[f"{g}|rows"] = np.asarray(grows, np.int64)
        rec["seconds"] = round(time.time() - t0, 1)
        per_group[g] = rec
        part["groups"][g] = rec
        json.dump(part, open(PARTIAL, "w"), indent=2, default=float)
        np.savez_compressed(NPZ, **banked)
        c = rec["curve"]["16"]
        log(f"[{g}] test n={len(grows)} (val n={rec['n_val']}{' SMALL' if rec['small_on_val'] else ''}) "
            f"@K=16 own {c['group_full@10']:.5f} vs global {c['global_full@10']:.5f} = "
            f"{c['diff_full@10']:+.5f} [{c['paired_ci_full']['lo']:+.5f}, {c['paired_ci_full']['hi']:+.5f}] "
            f"| val was {c['val_diff_full@10']} | shrinkage {c['shrinkage_full@10']} "
            f"| overlap {c['overlap']['set_overlap']}/16")

    # ---- the two arms, cohort-wide, paired ------------------------------------------------------
    arms_out = {}
    for arm in ARMS:
        arms_out[arm] = {}
        for K in ITEM_BUDGETS:
            df = np.full(len(rows), np.nan); dt = np.full(len(rows), np.nan)
            for g, grows in assign[arm].items():
                if not grows:
                    continue
                jj = np.asarray([pos[int(r)] for r in grows], np.int64)
                if g == "__global__":
                    df[jj] = 0.0; dt[jj] = 0.0            # asked the global order: difference is 0
                    dt[jj] = np.where(np.isfinite(gl[K][1][jj]), 0.0, np.nan)
                    df[jj] = np.where(np.isfinite(gl[K][0][jj]), 0.0, np.nan)
                else:
                    df[jj] = banked[f"{g}|{K}|own_full"] - gl[K][0][jj]
                    dt[jj] = banked[f"{g}|{K}|own_tail"] - gl[K][1][jj]
            arms_out[arm][str(K)] = {
                "diff_full@10": float(np.nanmean(df)), "diff_tail@10": float(np.nanmean(dt)),
                "paired_ci_full": paired_bootstrap(df), "paired_ci_tail": paired_bootstrap(dt),
                "global_full@10": float(np.nanmean(gl[K][0])),
                "global_tail@10": float(np.nanmean(gl[K][1])),
                "n_users_on_own_order": int(sum(len(v) for g2, v in assign[arm].items()
                                                if g2 != "__global__"))}
            banked[f"ARM_{arm}|{K}|diff_full"] = df.astype(np.float32)
            banked[f"ARM_{arm}|{K}|diff_tail"] = dt.astype(np.float32)

    prim = arms_out["all_groups"]["16"]["paired_ci_full"]
    prim2 = arms_out["fallback_200"]["16"]["paired_ci_full"]
    log("=" * 108)
    log("PRIMARY ENDPOINT (pre-specified): cohort-weighted group minus global, FULL NDCG@10, K=16, "
        f"all {len(rows)} TEST users, paired percentile bootstrap {N_BOOT} resamples seed {BOOT_SEED}")
    log(f"  ALL-GROUPS   {prim['mean']:+.5f}  95% CI [{prim['lo']:+.5f}, {prim['hi']:+.5f}]  "
        f"excludes zero: {prim['excludes_zero']}")
    log(f"  FALLBACK-200 {prim2['mean']:+.5f}  95% CI [{prim2['lo']:+.5f}, {prim2['hi']:+.5f}]  "
        f"excludes zero: {prim2['excludes_zero']}")
    log("-" * 108)
    log(f"{'group':<14}{'val n':>7}{'test n':>8}{'val gain':>11}{'test gain':>11}"
        f"{'shrinkage':>11}{'overlap16':>11}")
    for g in sorted(per_group, key=lambda k: -(val_sizes.get(k, 0))):
        r = per_group[g]
        if not r.get("has_order") or "curve" not in r:
            log(f"{g:<14}{val_sizes.get(g, 0):>7}{r.get('n_test', 0):>8}   {r.get('note', '')}")
            continue
        c = r["curve"]["16"]
        log(f"{g:<14}{r['n_val']:>7}{r['n_test']:>8}{(c['val_diff_full@10'] or 0):>11.5f}"
            f"{c['diff_full@10']:>11.5f}{(c['shrinkage_full@10'] or 0):>11.5f}"
            f"{c['overlap']['set_overlap']:>11}")
    log("=" * 108)

    out = {"analysis": "group_conditioned_greedy_genre_split_16v16", "stage": "2_evaluate_on_test",
           "label": args.label, "citable": True, "split": "test",
           "prespecified": True,
           "prespecification": (
               "Plan fixed and committed before any test number existed: primary endpoint = "
               "cohort-weighted group minus global, FULL NDCG@10, K=16, all test users, paired "
               "percentile bootstrap (10,000 resamples, seed 98765). Two arms, ALL-GROUPS and "
               "FALLBACK-200 (threshold 200 VALIDATION users, the same one stage 1 flags with). "
               "One look; no variant re-runs; no post-hoc tuning of threshold, rule, pool or budget."),
           "generous_assumption": GENEROUS,
           "stage1_source": os.path.basename(STAGE1_JSON),
           "tower": os.path.basename(args.snapshot),
           "concept_ckpt": os.path.basename(args.concept_ckpt),
           "n_test_users": len(rows), "n_val_users": s1["n_cohort"],
           "test_group_sizes": tsizes, "val_group_sizes": val_sizes,
           "n_users_no_liked_folding_test": n_no_liked,
           "n_users_liked_but_ungenred_test": n_ungenred,
           "n_test_users_on_global_under_fallback": n_fallback,
           "small_group_threshold": SMALL_GROUP,
           "global_sequence": [[int(a), int(b)] for a, b in global_seq],
           "budgets_items": list(ITEM_BUDGETS),
           "arms": arms_out, "per_group": per_group,
           "primary_endpoint": {"definition": "arms.all_groups.16.paired_ci_full",
                                "all_groups": prim, "fallback_200": prim2},
           "per_user_npz": os.path.basename(NPZ),
           "seconds": round(time.time() - t00, 1)}
    json.dump(out, open(OUT, "w"), indent=2, default=float)
    np.savez_compressed(NPZ, **banked)
    log(f"[done] wrote {os.path.basename(OUT)} + {os.path.basename(NPZ)} "
        f"in {(time.time() - t00) / 60:.1f}m")
    return out


if __name__ == "__main__":
    main()
