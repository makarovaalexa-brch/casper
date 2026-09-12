r"""A3/A4 downstream -- the three POLICY-B static channel orders on the 10,000 canonical TEST users.

Two numbers the chapter needs and could not have cleanly until the single-channel orders were rebuilt:

  * THE BANK-VS-CONDITIONING DECOMPOSITION. "Static items-only -> static combined, with no
    conditioning at all" was +0.01544 on test, which EXCEEDED the tree's entire conditioning gain
    (+0.01035) -- but the items side of it was a policy-A-SELECTED order, so the pool effect was an
    upper bound exactly the way the tree's +0.01371 was before A1. This script recomputes it against
    the rebuilt items order, AT EQUAL BUDGET (16 vs 16, not 15 vs 16), with a paired percentile CI.
  * ACT 4's CHANNEL COMPARISON at 16 questions -- items vs concepts vs combined -- on rebuilt orders.

CONTRACT: `genv_eval` unchanged, policy-B pool, learned decoder bias, no floor, canonical head mask,
certified pair. Paired percentile bootstrap over per-user differences, 10,000 resamples, seed 98765.

THE BANKED ASSERT (memory `banked-global-greedy-is-a-policy-A-artifact`): the combined order is
re-scored here from scratch, and its per-user test vectors must come back BIT-IDENTICAL to the ones
`ndcgtree16test_policyB` banked, at every budget, on both axes -- reassembled from that run's 15 leaf
slices and compared in the banked payload's own dtype. A new arm nested in a banked one at some
budget is a free contract check on the whole chain, and this one covers the comparator every tree
number in the chapter is measured against.
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
from group_greedy_genre import leak_assert

# Per-dataset results dir, single source of truth (run_battery_phaseA keys it off
# CASPER_DATASET). Was hardcoded here, so a Goodreads run would have written on top of the
# certified ML-25M tree/greedy artifacts -- HARD RULE 11e.
from run_battery_phaseA import OUTDIR                  # noqa: E402
LABEL = "channel_orders_policyB_test"
BUDGETS = (1, 2, 4, 8, 16)
N_BOOT = 10000
BOOT_SEED = 98765
ARMS = ("items", "concepts", "combined")


def paired_bootstrap(d, n_boot=N_BOOT, seed=BOOT_SEED):
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


def load_order(path, channel, n_pool, snapshot, conc_ckpt):
    b = json.load(open(path))
    assert b.get("citable") and not b.get("SANITY"), f"{path} is not a citable artifact"
    assert b["split"] == "validation" and not b["test_touched"], f"{path} was not built on validation"
    assert b["mask_policy"] == "B", f"{path} is not a policy-B build"
    assert b["objective"] == "full", f"{path} objective is {b['objective']}"
    assert b["n_pool"] == n_pool, f"{path} pool is {b['n_pool']}, expected {n_pool}"
    assert b["tower"] == os.path.basename(snapshot) and b["concept_ckpt"] == os.path.basename(conc_ckpt)
    seq = [(int(a), int(c)) for a, c in b["sequence"]]
    assert len(seq) >= max(BUDGETS), f"{path} has {len(seq)} picks"
    if channel in ("items", "concepts"):
        want = 0 if channel == "items" else 1
        assert all(q[0] == want for q in seq), f"{path} is not a {channel}-only order"
    return seq[:max(BUDGETS)], b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot", default=CERT_TOWER)
    ap.add_argument("--concept_ckpt", default=CERT_CONC)
    ap.add_argument("--items_json", default=os.path.join(OUTDIR, "greedy_static_policyB_items.json"))
    ap.add_argument("--concepts_json",
                    default=os.path.join(OUTDIR, "greedy_static_policyB_concepts.json"))
    ap.add_argument("--combined_json",
                    default=os.path.join(OUTDIR, "greedy_static_policyB_combined.json"))
    ap.add_argument("--tree_npz",
                    default=os.path.join(OUTDIR, "ndcgtree16test_policyB_per_user.npz"))
    ap.add_argument("--label", default=LABEL)
    ap.add_argument("--full_threads", action="store_true")
    args = ap.parse_args()

    t00 = time.time()
    OUT = os.path.join(OUTDIR, f"{args.label}.json")
    NPZ = os.path.join(OUTDIR, f"{args.label}_per_user.npz")

    orders = {}
    srcs = {}
    for arm, path, npool in (("items", args.items_json, 500),
                             ("concepts", args.concepts_json, 500),
                             ("combined", args.combined_json, 1000)):
        assert os.path.exists(path), f"{arm} order missing: {path}"
        orders[arm], srcs[arm] = load_order(path, arm, npool, args.snapshot, args.concept_ckpt)
        log(f"[order] {arm:<9} {os.path.basename(path)}: {orders[arm]}")

    E = build_greedy_env(args.snapshot, "test", args.concept_ckpt, f"{args.label}(test)")
    rows = E["rows"]
    contract_selftest(E, rows)
    leak_assert(E, rows, "test")
    assert len(rows) == 10000, f"test cohort is {len(rows)} users"
    set_active_rows(E, rows)
    E["memo"] = {}; E["n_evals"] = 0; E["t_evals"] = 0.0

    banked = {"__rows__": np.asarray(rows, np.int64)}
    per_user = {}
    table = {}
    f0, t0v = genv_eval(E, [], rows)
    banked["m0|full"] = f0.astype(np.float32); banked["m0|tail"] = t0v.astype(np.float32)
    log(f"[m0] ask nothing: full {np.nanmean(f0):.5f} tail {np.nanmean(t0v):.5f}")
    for arm in ARMS:
        table[arm] = {}
        for K in BUDGETS:
            f, t = genv_eval(E, orders[arm][:K], rows)
            per_user[(arm, K)] = (f, t)
            banked[f"{arm}|{K}|full"] = f.astype(np.float32)
            banked[f"{arm}|{K}|tail"] = t.astype(np.float32)
            ia, ca = genv_answered(E, orders[arm][:K], rows)
            table[arm][str(K)] = {"full@10": float(np.nanmean(f)), "tail@10": float(np.nanmean(t)),
                                  "items_answered": round(ia, 3), "concepts_answered": round(ca, 3)}
        np.savez_compressed(NPZ, **banked)
        c = table[arm][str(max(BUDGETS))]
        log(f"[test] {arm:<9} @K={max(BUDGETS)} full {c['full@10']:.5f} tail {c['tail@10']:.5f}")
    # the legacy 15-question items number, for continuity with the published 15-vs-16 row
    f15, t15 = genv_eval(E, orders["items"][:15], rows)
    per_user[("items", 15)] = (f15, t15)
    banked["items|15|full"] = f15.astype(np.float32); banked["items|15|tail"] = t15.astype(np.float32)
    table["items"]["15"] = {"full@10": float(np.nanmean(f15)), "tail@10": float(np.nanmean(t15))}

    # ---------------------------------------------------------------- THE BANKED ASSERT
    gate = {"npz": os.path.basename(args.tree_npz), "checked": [], "all_bit_identical": None}
    z = np.load(args.tree_npz)
    leaves = sorted({k.split("|")[1] for k in z.keys() if k.count("|") >= 2})
    pos = {int(r): i for i, r in enumerate(rows)}
    ok_all = True
    for K in BUDGETS:
        for axis in ("full", "tail"):
            ref = np.full(len(rows), np.nan, np.float32)
            seen = 0
            for nid in leaves:
                lr = z[f"test|{nid}|rows"]
                jj = np.asarray([pos[int(r)] for r in lr], np.int64)
                ref[jj] = z[f"test|{nid}|{K}|policyB_{axis}"]
                seen += len(lr)
            assert seen == len(rows), f"banked leaf slices cover {seen} of {len(rows)} test users"
            mine = banked[f"combined|{K}|{axis}"]
            # `equal_nan=True` is load-bearing, not cosmetic: 719 of the 10,000 test users have no
            # tail target and carry NaN, and plain `array_equal` calls NaN != NaN, so a bit-identical
            # pair fails the gate. Checked against the genre re-run's K=1 vectors, where the full
            # axis (no NaN) compared equal and the tail axis did not, at max abs diff 0.0.
            same = bool(np.array_equal(mine, ref, equal_nan=True))
            ok_all &= same
            gate["checked"].append({"K": K, "axis": axis, "bit_identical": same,
                                    "nan_masks_agree": bool(np.array_equal(np.isnan(mine),
                                                                           np.isnan(ref))),
                                    "max_abs_diff": float(np.nanmax(np.abs(
                                        mine.astype(np.float64) - ref.astype(np.float64))))})
    gate["all_bit_identical"] = bool(ok_all)
    assert ok_all, ("BANKED ASSERT FAILED: the rebuilt combined comparator does not reproduce the "
                    "per-user test vectors banked by ndcgtree16test_policyB. " + json.dumps(gate))
    log(f"[gate] BANKED ASSERT PASS: the combined comparator's per-user TEST vectors are "
        f"BIT-IDENTICAL to ndcgtree16test_policyB's, at all {len(BUDGETS)} budgets on both axes")

    # ---------------------------------------------------------------- the decomposition + channels
    contrasts = {}

    def add(name, a, b, note):
        """a - b, paired, per user."""
        (fa, ta), (fb, tb) = a, b
        contrasts[name] = {"note": note,
                           "diff_full@10": float(np.nanmean(fa - fb)),
                           "diff_tail@10": float(np.nanmean(ta - tb)),
                           "paired_ci_full": paired_bootstrap(fa - fb),
                           "paired_ci_tail": paired_bootstrap(ta - tb)}
        c = contrasts[name]
        log(f"[contrast] {name:<34} full {c['diff_full@10']:+.5f} "
            f"[{c['paired_ci_full']['lo']:+.5f},{c['paired_ci_full']['hi']:+.5f}]  tail "
            f"{c['diff_tail@10']:+.5f} [{c['paired_ci_tail']['lo']:+.5f},"
            f"{c['paired_ci_tail']['hi']:+.5f}]")

    for K in BUDGETS:
        add(f"combined_minus_items@K{K}", per_user[("combined", K)], per_user[("items", K)],
            "THE BANK EFFECT at equal budget: change the question bank, condition on nothing.")
        add(f"combined_minus_concepts@K{K}", per_user[("combined", K)], per_user[("concepts", K)],
            "combined pool vs the concepts-only pool, no conditioning either side")
        add(f"concepts_minus_items@K{K}", per_user[("concepts", K)], per_user[("items", K)],
            "which single channel is the better bank, at equal budget")
    add("combined16_minus_items15", per_user[("combined", 16)], per_user[("items", 15)],
        "the published unequal-budget form (items 15 vs combined 16), for continuity only")

    out = {"analysis": "policyB_static_channel_orders_on_test",
           "label": args.label, "citable": True, "split": "test",
           "purpose": ("the bank-vs-conditioning decomposition and Act 4's channel comparison, both "
                       "on orders SELECTED under policy B (A3/A4) rather than under the retired "
                       "policy-A mask"),
           "tower": os.path.basename(args.snapshot),
           "concept_ckpt": os.path.basename(args.concept_ckpt),
           "mask_policy": "B", "pool_policy": "B", "pool_contract": E["pool_contract"],
           "n_test_users": len(rows), "budgets": list(BUDGETS),
           "orders": {a: [[int(x), int(y)] for x, y in orders[a]] for a in ARMS},
           "order_sources": {a: {"file": os.path.basename(
               {"items": args.items_json, "concepts": args.concepts_json,
                "combined": args.combined_json}[a]),
               "label": srcs[a]["label"],
               "positions_identical_to_policyA_order":
                   srcs[a].get("positions_identical_to_policyA_order")} for a in ARMS},
           "m0_full@10": float(np.nanmean(f0)), "m0_tail@10": float(np.nanmean(t0v)),
           "test_curves": table,
           "validation_curves": {a: srcs[a].get("curve_policyB") for a in ARMS},
           "contrasts": contrasts,
           "banked_assert": gate,
           "inference": (f"paired percentile bootstrap over per-user differences, {N_BOOT} "
                         f"resamples, seed {BOOT_SEED}"),
           "per_user_npz": os.path.basename(NPZ),
           "seconds": round(time.time() - t00, 1)}
    with open(OUT + ".tmp", "w") as fh:
        json.dump(out, fh, indent=2, default=float)
    os.replace(OUT + ".tmp", OUT)
    np.savez_compressed(NPZ, **banked)
    log(f"[done] wrote {os.path.basename(OUT)} in {(time.time() - t00) / 60:.1f}m")
    return out


if __name__ == "__main__":
    main()
