"""A12b - price the BURN, so the A12 answer-matched reading is not confounded by it.

A12 found that at a matched number of answers the item channel buys far more than the concept
channel, which contradicts the breadth hypothesis. One confound stands in the way of that reading:
the item arm reaches its 3.108 answers by asking SIXTEEN questions, burning ~12.9 of them, whereas
the concept arm reaches a comparable count in TWO or FOUR questions. A burned film question is not
free information: it folds the i26 exposure token, and "this user has not seen a top-500 popular
film" is a real signal. So part of the item arm's lift may be bought by burns rather than answers,
which would inflate its lift-per-answer.

This prices the burn directly, from artifacts already on disk, with no new scoring:

  * Users who answered NOTHING in an arm at budget K have received burns only. Their entire lift
    over the answer-free reference IS the burn value. That is an exact measurement, not an estimate.
  * Subtracting the burn-only lift from each arm gives an answer-attributable lift, and the
    answer-matched contrast can be recomputed on it.

If the item channel still wins per answer after the burn is removed, the A12 reading stands. If the
whole gap is burn value, the honest finding changes: it is not that a film answer is worth more, but
that asking a film question is informative even when it fails.

Inputs: experiments/battery/channel_orders_policyB_test_per_user.npz (per-user scores, banked)
        experiments/battery/a12_answerability_vs_breadth_per_user.npz (per-user answer counts, A12)
Run:    python src/instrument/a12b_burn_value.py
"""
import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
OUTDIR = os.path.join(_ROOT, "experiments", "battery")
SRC = "channel_orders_policyB_test"
A12 = "a12_answerability_vs_breadth"
LABEL = "a12b_burn_value"
BUDGETS = (1, 2, 4, 8, 16)
ARMS = ("items", "concepts", "combined")
N_BOOT = 10000
BOOT_SEED = 98765


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
            "excludes_zero": bool(lo > 0 or hi < 0)}


def main():
    Z = np.load(os.path.join(OUTDIR, f"{SRC}_per_user.npz"))
    A = np.load(os.path.join(OUTDIR, f"{A12}_per_user.npz"))
    assert np.array_equal(Z["__rows__"], A["__rows__"]), "row order differs; pairing would be wrong"
    m0f, m0t = Z["m0|full"].astype(np.float64), Z["m0|tail"].astype(np.float64)
    out = {"label": LABEL, "citable": True, "split": "test", "n_users": int(len(m0f)),
           "design_sheet": "docs/design/DS_A12_ANSWERABILITY_VS_BREADTH.md",
           "purpose": "price the burn so A12's answer-matched reading is not confounded by it",
           "inference": f"paired percentile bootstrap, {N_BOOT} resamples clustered over users, "
                        f"seed {BOOT_SEED}",
           "burn_only": {}, "answer_attributable": {}, "answer_matched_net": {}}

    print("BURN-ONLY LIFT: users who answered NOTHING, so their whole lift is the burn")
    print(f"{'arm':10s} {'K':>3s} {'n_zero':>7s} {'share':>7s} {'burn full':>10s} "
          f"{'burn tail':>10s} {'arm full':>9s} {'burn share':>11s}")
    burn = {}
    for arm in ARMS:
        for K in BUDGETS:
            a = A[f"ans|{arm}|{K}"].astype(np.float64)
            zero = a <= 0
            f = Z[f"{arm}|{K}|full"].astype(np.float64)
            t = Z[f"{arm}|{K}|tail"].astype(np.float64)
            lf_all = float((f - m0f).mean())
            if zero.sum() < 30:
                burn[(arm, K)] = None
                print(f"{arm:10s} {K:>3d} {int(zero.sum()):>7d}    n/a (too few zero-answer users)")
                continue
            bf = float((f - m0f)[zero].mean())
            bt = float((t - m0t)[zero].mean())
            burn[(arm, K)] = (bf, bt)
            out["burn_only"][f"{arm}|K{K}"] = {
                "n_zero_answer_users": int(zero.sum()),
                "share_of_cohort": float(zero.mean()),
                "burn_lift_full": bf, "burn_lift_tail": bt,
                "arm_lift_full": lf_all,
                "burn_share_of_arm_lift": (bf / lf_all if lf_all != 0 else float("nan")),
                "ci_full": paired_bootstrap((f - m0f)[zero])}
            print(f"{arm:10s} {K:>3d} {int(zero.sum()):>7d} {zero.mean():7.3f} {bf:10.5f} "
                  f"{bt:10.5f} {lf_all:9.5f} {bf / lf_all if lf_all else float('nan'):11.3f}")

    print("\nANSWER-ATTRIBUTABLE LIFT: arm lift minus the burn-only lift, per answer")
    print(f"{'arm':10s} {'K':>3s} {'lift':>9s} {'burn':>9s} {'net':>9s} {'answers':>8s} {'net/ans':>9s}")
    for arm in ARMS:
        for K in BUDGETS:
            b = burn.get((arm, K))
            if b is None:
                continue
            f = Z[f"{arm}|{K}|full"].astype(np.float64)
            lf = float((f - m0f).mean())
            na = float(A[f"ans|{arm}|{K}"].mean())
            net = lf - b[0]
            out["answer_attributable"][f"{arm}|K{K}"] = {
                "arm_lift_full": lf, "burn_lift_full": b[0], "net_full": net,
                "answers": na, "net_per_answer_full": (net / na if na > 0 else float("nan"))}
            print(f"{arm:10s} {K:>3d} {lf:9.5f} {b[0]:9.5f} {net:9.5f} {na:8.3f} "
                  f"{net / na if na > 0 else float('nan'):9.5f}")

    # The decisive contrast, recomputed per user with each arm's own burn baseline removed.
    print("\nANSWER-MATCHED, NET OF BURN (per-user: arm score minus that arm's burn-only baseline)")
    for a1, k1, a2, k2 in (("items", 16, "concepts", 4), ("items", 8, "concepts", 2),
                           ("items", 4, "concepts", 1)):
        b1, b2 = burn.get((a1, k1)), burn.get((a2, k2))
        if b1 is None or b2 is None:
            continue
        n1 = float(A[f"ans|{a1}|{k1}"].mean()); n2 = float(A[f"ans|{a2}|{k2}"].mean())
        rec = {"arm_a": f"{a1}@K{k1}", "answers_a": n1, "arm_b": f"{a2}@K{k2}", "answers_b": n2,
               "burn_a_full": b1[0], "burn_b_full": b2[0]}
        for axis, bi in (("full", 0), ("tail", 1)):
            d = ((Z[f"{a1}|{k1}|{axis}"].astype(np.float64) - burn[(a1, k1)][bi])
                 - (Z[f"{a2}|{k2}|{axis}"].astype(np.float64) - burn[(a2, k2)][bi]))
            rec[axis] = paired_bootstrap(d)
        out["answer_matched_net"][f"{a1}@K{k1}_minus_{a2}@K{k2}"] = rec
        s = "*" if rec["full"]["excludes_zero"] else " "
        print(f"  {a1}@K{k1} ({n1:.3f} ans, burn {b1[0]:+.5f}) - {a2}@K{k2} "
              f"({n2:.3f} ans, burn {b2[0]:+.5f}): full {rec['full']['mean']:+.5f}{s} "
              f"[{rec['full']['lo']:+.5f},{rec['full']['hi']:+.5f}]")

    op = os.path.join(OUTDIR, f"{LABEL}.json")
    json.dump(out, open(op, "w"), indent=2)
    print(f"\n[done] {op}")


if __name__ == "__main__":
    main()
