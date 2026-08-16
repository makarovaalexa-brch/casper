r"""EXPERIMENT V -- the side-by-side. Reads banked per-user vectors only; searches nothing.

Puts four numbers on ONE ruler (10,000 validation users, policy-B mask, certified pair, paired
percentile bootstrap 10,000 resamples seed 98765, over PER-USER differences):

  genre arm    cohort = the answer to one ASKED genre question, scored from the EMPTY state against
               the rebuilt policy-B items comparator (`groupgreedy_genre15_VAL_policyB`).
  V            cohort = the genre signature of the k=4 VOLUNTEERED items, scored from the SEEDED
               state against the same comparator seeded the same way.
  V placebo    random cohorts with V's exact size profile -- the null level for an in-sample
               cohort-conditioned greedy.

Each arm is differenced against ITS OWN no-conditioning baseline (the global order at the same
state), so the three gains are commensurable even though the states differ. The paired
difference-of-differences (V - genre, V - placebo) is computed per user over the same 10,000 rows.
"""
import os
import sys
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
sys.path.insert(0, _HERE)

import json
import argparse
import numpy as np

B = os.path.join(_ROOT, "experiments", "battery")
VDIR = os.path.join(B, "volcohorts")
N_BOOT, BOOT_SEED = 10000, 98765
SMALL = 200


def pb(d, n_boot=N_BOOT, seed=BOOT_SEED):
    d = np.asarray(d, np.float64)
    d = d[np.isfinite(d)]
    if len(d) == 0:
        return {"mean": float("nan"), "lo": float("nan"), "hi": float("nan"), "n": 0,
                "excludes_zero": False}
    rng = np.random.default_rng(seed)
    bs = d[rng.integers(0, len(d), size=(n_boot, len(d)))].mean(1)
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return {"mean": float(d.mean()), "lo": float(lo), "hi": float(hi), "n": int(len(d)),
            "n_boot": n_boot, "seed": seed, "excludes_zero": bool(lo > 0 or hi < 0)}


def fmt(c):
    return f"{c['mean']:+.5f} [{c['lo']:+.5f}, {c['hi']:+.5f}]"


def genre_vectors(K):
    """Per-user (all_groups, fallback_200) difference vectors for the genre arm, in canonical order."""
    s1 = json.load(open(os.path.join(B, "groupgreedy_genre15.json")))
    z1 = np.load(os.path.join(B, "groupgreedy_genre15_per_user.npz"))
    zb = np.load(os.path.join(B, "groupgreedy_genre15_VAL_policyB_per_user.npz"))
    rows = [int(r) for r in zb["__rows__"]]
    pos = {r: i for i, r in enumerate(rows)}
    gf = zb[f"__global_policyB__|{K}|full"].astype(np.float64)
    gt = zb[f"__global_policyB__|{K}|tail"].astype(np.float64)
    out = {}
    for nm in ("all_groups", "fallback_200"):
        out[nm] = (np.full(len(rows), np.nan), np.full(len(rows), np.nan))
    per = {}
    for g, rec in s1["groups"].items():
        jj = np.asarray([pos[int(r)] for r in rec["rows"]], np.int64)
        of = z1[f"{g}|{K}|own_full"].astype(np.float64)
        ot = z1[f"{g}|{K}|own_tail"].astype(np.float64)
        df, dt = of - gf[jj], ot - gt[jj]
        out["all_groups"][0][jj] = df; out["all_groups"][1][jj] = dt
        small = rec["n_users"] < SMALL
        out["fallback_200"][0][jj] = 0.0 if small else df
        out["fallback_200"][1][jj] = 0.0 if small else dt
        per[g] = {"n": rec["n_users"], "small": bool(small), "full": pb(df), "tail": pb(dt)}
    return rows, out, per


def v_vectors(label, K):
    blob = json.load(open(os.path.join(VDIR, f"{label}.json")))
    z = np.load(os.path.join(VDIR, f"{label}_per_user.npz"))
    rows = [int(r) for r in z["__rows__"]]
    pos = {r: i for i, r in enumerate(rows)}
    out = {nm: (np.full(len(rows), np.nan), np.full(len(rows), np.nan))
           for nm in ("all_groups", "fallback_200")}
    per = {}
    for g, rec in blob["groups"].items():
        jj = np.asarray([pos[int(r)] for r in rec["rows"]], np.int64)
        df = z[f"{g}|{K}|own_full"].astype(np.float64) - z[f"{g}|{K}|glob_full"].astype(np.float64)
        dt = z[f"{g}|{K}|own_tail"].astype(np.float64) - z[f"{g}|{K}|glob_tail"].astype(np.float64)
        out["all_groups"][0][jj] = df; out["all_groups"][1][jj] = dt
        small = rec["small_group"]
        out["fallback_200"][0][jj] = 0.0 if small else df
        out["fallback_200"][1][jj] = 0.0 if small else dt
        c = rec["curve_item_matched"][str(K)]
        per[g] = {"n": rec["n_users"], "small": bool(small), "full": pb(df), "tail": pb(dt),
                  "set_overlap": c["overlap"]["set_overlap"],
                  "reask_own_mean": c["reask_own_mean"],
                  "reask_global_mean": c["reask_global_mean"],
                  "order": rec["order"]}
    return rows, out, per, blob


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--K", type=int, default=15)
    ap.add_argument("--arms", default="volcohorts_k4_volunteered,volcohorts_k4_placebo")
    args = ap.parse_args()
    K = args.K
    res = {"K": K, "n_boot": N_BOOT, "seed": BOOT_SEED, "split": "validation",
           "test_touched": False}

    grows, gvec, gper = genre_vectors(K)
    res["genre"] = {a: {"full": pb(gvec[a][0]), "tail": pb(gvec[a][1])} for a in gvec}
    res["genre_per_group"] = gper

    arms = {}
    for lb in [x for x in args.arms.split(",") if x.strip()]:
        p = os.path.join(VDIR, f"{lb}.json")
        if not os.path.exists(p):
            print(f"[skip] {lb}: not on disk yet")
            continue
        r, v, per, blob = v_vectors(lb, K)
        assert r == grows, f"{lb} row order differs from the genre arm's"
        arms[lb] = (v, per, blob)
        res[lb] = {a: {"full": pb(v[a][0]), "tail": pb(v[a][1])} for a in v}
        res[lb + "_per_group"] = per
        res[lb + "_sizes"] = blob["group_sizes"]

    print("=" * 104)
    print(f"COHORT-CONDITIONING GAIN AT K={K}, 10,000 VALIDATION USERS, POLICY-B RULER")
    print("=" * 104)
    print(f"{'arm':<34}{'all_groups full':>28}{'all_groups tail':>28}")
    print(f"{'genre (ASKED question)':<34}{fmt(res['genre']['all_groups']['full']):>28}"
          f"{fmt(res['genre']['all_groups']['tail']):>28}")
    for lb in arms:
        nm = "V VOLUNTEERED (k=4)" if "volunteered" in lb else "V PLACEBO (random cohorts)"
        print(f"{nm:<34}{fmt(res[lb]['all_groups']['full']):>28}"
              f"{fmt(res[lb]['all_groups']['tail']):>28}")
    print("-" * 104)
    print(f"{'arm (fallback_200)':<34}{'full':>28}{'tail':>28}")
    print(f"{'genre':<34}{fmt(res['genre']['fallback_200']['full']):>28}"
          f"{fmt(res['genre']['fallback_200']['tail']):>28}")
    for lb in arms:
        nm = "V VOLUNTEERED" if "volunteered" in lb else "V PLACEBO"
        print(f"{nm:<34}{fmt(res[lb]['fallback_200']['full']):>28}"
              f"{fmt(res[lb]['fallback_200']['tail']):>28}")
    print("=" * 104)

    # ---- paired contrasts, per user, same 10,000 rows -------------------------------------------
    con = {}
    vl = "volcohorts_k4_volunteered"
    pl = "volcohorts_k4_placebo"
    if vl in arms:
        for a in ("all_groups", "fallback_200"):
            con[f"V_minus_genre|{a}"] = {"full": pb(arms[vl][0][a][0] - gvec[a][0]),
                                         "tail": pb(arms[vl][0][a][1] - gvec[a][1])}
        if pl in arms:
            for a in ("all_groups", "fallback_200"):
                con[f"V_minus_placebo|{a}"] = {"full": pb(arms[vl][0][a][0] - arms[pl][0][a][0]),
                                               "tail": pb(arms[vl][0][a][1] - arms[pl][0][a][1])}
    if pl in arms:
        for a in ("all_groups", "fallback_200"):
            con[f"placebo_minus_genre|{a}"] = {"full": pb(arms[pl][0][a][0] - gvec[a][0]),
                                              "tail": pb(arms[pl][0][a][1] - gvec[a][1])}
    res["paired_contrasts"] = con
    print("PAIRED DIFFERENCE OF DIFFERENCES (same user, each arm against its own global baseline)")
    for k, v in con.items():
        print(f"  {k:<34} full {fmt(v['full'])}   tail {fmt(v['tail'])}")
    print("=" * 104)

    if vl in arms:
        per = arms[vl][1]
        print(f"{'group':<14}{'n':>7}{'full':>11}{'lo':>11}{'hi':>11}{'ex0':>6}{'tail':>11}"
              f"{'ovlap':>7}{'reask':>8}{'reaskG':>8}")
        for g, v in sorted(per.items(), key=lambda x: -x[1]["n"]):
            print(f"{g:<14}{v['n']:>7}{v['full']['mean']:>11.5f}{v['full']['lo']:>11.5f}"
                  f"{v['full']['hi']:>11.5f}{str(v['full']['excludes_zero']):>6}"
                  f"{v['tail']['mean']:>11.5f}{v['set_overlap']:>7}{v['reask_own_mean']:>8.3f}"
                  f"{v['reask_global_mean']:>8.3f}")
        print("=" * 104)

    p = os.path.join(VDIR, f"volcohorts_report_K{K}.json")
    json.dump(res, open(p, "w"), indent=2, default=float)
    print(f"[done] wrote {os.path.relpath(p, _ROOT)}")


if __name__ == "__main__":
    main()
