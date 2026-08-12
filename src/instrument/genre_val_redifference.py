r"""A3 follow-on -- re-difference the genre arm's VALIDATION gains against the rebuilt comparator.

The stage-2 re-run (`groupgreedy_genre15_TEST_policyB`) measures each group's test gain against the
REBUILT policy-B items order. Its `shrinkage` column, however, subtracts stage 1's validation gain,
which was measured against the POLICY-A order -- two rulers in one subtraction. The val->test
shrinkage fact ("all 17 groups shrank") is load-bearing for the chapter, so it gets its own ruler.

Nothing is re-searched. Stage 1 banked every group's OWN per-user validation vectors
(`groupgreedy_genre15_per_user.npz`, `<group>|<K>|own_full`), so all this job computes is the
REBUILT global order's per-user validation vectors, and re-differences.

FREE CONTRACT CHECK (memory `banked-global-greedy-is-a-policy-A-artifact`): the POLICY-A global
order is also re-scored here and must reproduce stage 1's banked `<group>|<K>|glob_full` vectors
bit-identically (NaN-aware -- users with no tail target carry NaN, and NaN != NaN would fail a naive
comparison). If that fails, nothing else in this file is trustworthy and it stops.
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
                                    contract_selftest, CERT_TOWER, CERT_CONC)
from group_greedy_genre import leak_assert

OUTDIR = os.path.join(_ROOT, "experiments", "battery")
STAGE1_JSON = os.path.join(OUTDIR, "groupgreedy_genre15.json")
STAGE1_NPZ = os.path.join(OUTDIR, "groupgreedy_genre15_per_user.npz")
TEST_JSON = os.path.join(OUTDIR, "groupgreedy_genre15_TEST_policyB.json")
LABEL = "groupgreedy_genre15_VAL_policyB"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot", default=CERT_TOWER)
    ap.add_argument("--concept_ckpt", default=CERT_CONC)
    ap.add_argument("--global_from",
                    default=os.path.join(OUTDIR, "greedy_static_policyB_items.json"))
    ap.add_argument("--label", default=LABEL)
    ap.add_argument("--full_threads", action="store_true")
    args = ap.parse_args()

    t00 = time.time()
    OUT = os.path.join(OUTDIR, f"{args.label}.json")
    NPZ = os.path.join(OUTDIR, f"{args.label}_per_user.npz")

    s1 = json.load(open(STAGE1_JSON))
    assert s1["split"] == "validation" and s1.get("citable")
    K_LIST = tuple(int(k) for k in s1["budgets_items"])
    L = int(s1["L_items"])
    old_global = [(int(a), int(b)) for a, b in s1["global_sequence"]]
    gb = json.load(open(args.global_from))
    assert gb["channel"] == "items" and gb["mask_policy"] == "B" and gb.get("citable")
    new_global = [(int(a), int(b)) for a, b in gb["sequence"]]
    t2 = json.load(open(TEST_JSON))
    assert t2["global_comparator"]["label"] == gb["label"], "test re-run used a different comparator"

    E = build_greedy_env(args.snapshot, "validation", args.concept_ckpt, f"{args.label}(val)")
    rows = E["rows"]
    contract_selftest(E, rows)
    leak_assert(E, rows, "validation")
    assert len(rows) == s1["n_cohort"], f"cohort drift {len(rows)} vs {s1['n_cohort']}"
    set_active_rows(E, rows)
    pos = {int(r): i for i, r in enumerate(rows)}

    z1 = np.load(STAGE1_NPZ)
    banked = {"__rows__": np.asarray(rows, np.int64)}
    gl = {}
    for tagname, seq in (("policyB", new_global), ("policyA", old_global)):
        for K in K_LIST:
            f, t = genv_eval(E, seq[:K], rows)
            gl[(tagname, K)] = (f, t)
            banked[f"__global_{tagname}__|{K}|full"] = f.astype(np.float32)
            banked[f"__global_{tagname}__|{K}|tail"] = t.astype(np.float32)
            log(f"[global {tagname}] K={K:<3} full {np.nanmean(f):.5f} tail {np.nanmean(t):.5f}")

    # ---- the free contract check: policy-A global must reproduce stage 1's banked vectors --------
    gate = []
    for g, rec in s1["groups"].items():
        grows = [int(r) for r in rec["rows"]]
        jj = np.asarray([pos[r] for r in grows], np.int64)
        for K in K_LIST:
            key = f"{g}|{K}|glob_full"
            if key not in z1:
                continue
            mine = gl[("policyA", K)][0][jj].astype(np.float32)
            same = bool(np.array_equal(mine, z1[key], equal_nan=True))
            gate.append({"group": g, "K": K, "bit_identical": same,
                         "max_abs_diff": float(np.nanmax(np.abs(
                             mine.astype(np.float64) - z1[key].astype(np.float64))))})
    bad = [c for c in gate if not c["bit_identical"]]
    assert not bad, f"BANKED ASSERT FAILED on {len(bad)} cells: {bad[:3]}"
    log(f"[gate] BANKED ASSERT PASS: the policy-A global order reproduces stage 1's banked "
        f"per-user validation vectors bit-identically in all {len(gate)} group x budget cells")

    out_groups = {}
    for g, rec in s1["groups"].items():
        grows = [int(r) for r in rec["rows"]]
        jj = np.asarray([pos[r] for r in grows], np.int64)
        row = {"n_val": len(grows), "curve": {}}
        for K in K_LIST:
            of = z1[f"{g}|{K}|own_full"].astype(np.float64)
            ot = z1[f"{g}|{K}|own_tail"].astype(np.float64)
            dB_f = float(np.nanmean(of - gl[("policyB", K)][0][jj]))
            dB_t = float(np.nanmean(ot - gl[("policyB", K)][1][jj]))
            dA_f = float(np.nanmean(of - gl[("policyA", K)][0][jj]))
            row["curve"][str(K)] = {"val_diff_full_vs_policyB": dB_f,
                                    "val_diff_tail_vs_policyB": dB_t,
                                    "val_diff_full_vs_policyA": dA_f,
                                    "stage1_val_diff_full": rec["curve_item_matched"][str(K)]["diff_full@10"]}
        out_groups[g] = row

    # cohort-weighted validation gain against each comparator, and the honest shrinkage
    weighted = {}
    for K in K_LIST:
        dfB = np.full(len(rows), np.nan); dfA = np.full(len(rows), np.nan)
        dtB = np.full(len(rows), np.nan)
        for g, rec in s1["groups"].items():
            jj = np.asarray([pos[int(r)] for r in rec["rows"]], np.int64)
            of = z1[f"{g}|{K}|own_full"].astype(np.float64)
            ot = z1[f"{g}|{K}|own_tail"].astype(np.float64)
            dfB[jj] = of - gl[("policyB", K)][0][jj]
            dtB[jj] = ot - gl[("policyB", K)][1][jj]
            dfA[jj] = of - gl[("policyA", K)][0][jj]
        weighted[str(K)] = {"all_groups_full_vs_policyB": float(np.nanmean(dfB)),
                            "all_groups_tail_vs_policyB": float(np.nanmean(dtB)),
                            "all_groups_full_vs_policyA": float(np.nanmean(dfA)),
                            "global_policyB_full": float(np.nanmean(gl[("policyB", K)][0])),
                            "global_policyA_full": float(np.nanmean(gl[("policyA", K)][0]))}

    log("-" * 104)
    log(f"{'group':<14}{'n_val':>7}{'val vs A':>11}{'val vs B':>11}{'test vs B':>11}{'shrinkage':>11}")
    shr = {}
    n_shrank = 0
    for g in sorted(out_groups, key=lambda k: -out_groups[k]["n_val"]):
        c = out_groups[g]["curve"][str(L)]
        tg = t2["per_group"].get(g, {}).get("curve", {}).get(str(L))
        if tg is None:
            continue
        s = tg["diff_full@10"] - c["val_diff_full_vs_policyB"]
        shr[g] = s
        n_shrank += int(s < 0)
        log(f"{g:<14}{out_groups[g]['n_val']:>7}{c['val_diff_full_vs_policyA']:>11.5f}"
            f"{c['val_diff_full_vs_policyB']:>11.5f}{tg['diff_full@10']:>11.5f}{s:>11.5f}")
    log(f"[shrinkage] {n_shrank} of {len(shr)} groups shrank val->test on ONE ruler (the rebuilt "
        f"policy-B comparator on both sides)")

    out = {"analysis": "genre_group_orders_validation_redifferenced_vs_policyB_comparator",
           "label": args.label, "citable": True, "split": "validation", "test_touched": False,
           "why": ("the stage-2 re-run's shrinkage column mixed rulers: test gains against the "
                   "rebuilt policy-B items order, validation gains against the policy-A one. This "
                   "puts both sides on the rebuilt comparator. Nothing is re-searched -- the group "
                   "orders' own per-user validation vectors are read from stage 1's npz."),
           "tower": os.path.basename(args.snapshot),
           "concept_ckpt": os.path.basename(args.concept_ckpt),
           "mask_policy": "B", "pool_policy": "B", "pool_contract": E["pool_contract"],
           "n_cohort": len(rows), "budgets_items": list(K_LIST), "L_items": L,
           "comparator_policyB": {"file": os.path.basename(args.global_from), "label": gb["label"],
                                  "sequence": [[int(a), int(b)] for a, b in new_global]},
           "comparator_policyA": [[int(a), int(b)] for a, b in old_global],
           "banked_assert_cells": len(gate), "banked_assert_all_bit_identical": True,
           "groups": out_groups, "cohort_weighted": weighted,
           "shrinkage_on_one_ruler@L": shr,
           "n_groups_shrank": n_shrank, "n_groups": len(shr),
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
