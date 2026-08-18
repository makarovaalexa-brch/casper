r"""Build the two EXTRA teacher cohorts: the RESIDUAL group and taste cluster c19.

  residual  validation users who state NONE of the six anchors. Routable by elimination, so a
            teacher for them is DEPLOYABLE and directly comparable to the cold tree, which was
            fitted on everyone including the users routed away from it.

  c19       the taste cluster with no nameable attribute -- best attributes `goofy` 0.038 /
            `silly` 0.031 against a 0.061 purity requirement, margins -0.077 to -0.115, and films
            that are unrelated obscurities. Reddit arrival cluster r7 lands 27% here.
            ** ORACLE-ROUTED: membership comes from the user's FULL PROFILE, which is not observable
            at ask time. Its teacher is a CEILING -- it answers "does this region have different
            optimal questions?", not "can we route to it?". If it beats the cold tree, hunting for a
            routable signature is worthwhile; if not, no signature would be worth finding. **

Writes experiments/battery/m3_cohort_{residual,c19}.json (lists of validation row ids).
Read-only w.r.t. the recommender: no NDCG, no targets, no TEST.
"""
import csv
import json
import os
import re
import sys

_H = os.path.dirname(os.path.abspath(__file__))
_R = os.path.abspath(os.path.join(_H, "..", ".."))
sys.path.insert(0, _H)

import numpy as np

B = os.path.join(_R, "experiments", "battery")
M1 = os.path.join(B, "m1")
ANCHORS = {"arthouse": 68, "dialogue": 275, "marvel": 581,
           "gore": 417, "teenagers": 922, "inspiring": 503}
TASTE_K, C19 = 20, 19


def main():
    from strategy_channel_suite import build_greedy_env, CERT_TOWER, CERT_CONC
    from m1_ideal import kmeans

    E = build_greedy_env(CERT_TOWER, "validation", CERT_CONC, "m3cohorts")
    rows = np.asarray(E["rows"], np.int64)
    V, F = np.asarray(E["V"]), np.asarray(E["foldable"])
    endorse = (F > 0) & (V > 0)

    # ---- residual: states none of the six anchors ---------------------------------------------
    states_any = np.zeros(len(rows), bool)
    for nm, cid in ANCHORS.items():
        m = endorse[rows, cid]
        states_any |= m
        print(f"  [{nm:<10}] {int(m.sum()):>5} validation users state it")
    residual = rows[~states_any]
    print(f"[residual] {len(residual)} of {len(rows)} state none of the six "
          f"({len(residual)/len(rows):.1%})")
    json.dump([int(r) for r in residual],
              open(os.path.join(B, "m3_cohort_residual.json"), "w"))

    # ---- c19: nearest train taste centroid, de-era'd space -------------------------------------
    files = sorted((f for f in os.listdir(os.path.join(M1, "train")) if f.startswith("deera_")),
                   key=lambda x: int(x.split("_")[1]))
    Ztr = np.concatenate([np.load(os.path.join(M1, "train", f))["z"] for f in files], 0)
    lab, _ = kmeans(Ztr, TASTE_K)                      # deterministic: same seed as the sweep
    cent = np.stack([Ztr[lab == i].mean(0) for i in range(TASTE_K)])
    cent /= np.maximum(np.linalg.norm(cent, axis=1, keepdims=True), 1e-9)
    print(f"[taste] recomputed k={TASTE_K} on de-era'd train states; sizes "
          f"{np.bincount(lab, minlength=TASTE_K).tolist()}")

    zf = np.load(os.path.join(M1, "z_full.npz"))
    assert list(zf["rows"]) == list(rows), "z_full rows are not the env's rows, in order"
    Zv = zf["z"]

    # the SAME era axis, rebuilt from item release years (m1_deera.py's construction)
    Wd = np.asarray(E["ctx"].Wd)
    with open(os.path.join(_R, "data", "ml-25m", "proc", "unique_sid.txt")) as fh:
        sids = [int(x) for x in fh.read().split()]
    year = {}
    with open(os.path.join(_R, "data", "movielens", "movies.csv"), newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            m = re.search(r"\((\d{4})\)\s*$", r["title"].strip())
            if m:
                year[int(r["movieId"])] = int(m.group(1))
    yr = np.array([year.get(s, np.nan) for s in sids], float)
    ok = ~np.isnan(yr)
    y = np.zeros(len(sids))
    y[ok] = (yr[ok] - yr[ok].mean()) / yr[ok].std()
    d_era = Wd.T @ y
    d_era /= np.linalg.norm(d_era)
    Zv = Zv - np.outer(Zv @ d_era, d_era)

    Zvn = Zv / np.maximum(np.linalg.norm(Zv, axis=1, keepdims=True), 1e-9)
    assign = np.argmax(Zvn @ cent.T, axis=1)
    c19_rows = rows[assign == C19]
    print(f"[c19] {len(c19_rows)} validation users assign to taste cluster {C19} "
          f"({len(c19_rows)/len(rows):.1%})")
    print(f"[c19] full validation assignment histogram: "
          f"{np.bincount(assign, minlength=TASTE_K).tolist()}")
    json.dump([int(r) for r in c19_rows], open(os.path.join(B, "m3_cohort_c19.json"), "w"))
    print("[done] wrote m3_cohort_residual.json and m3_cohort_c19.json")


if __name__ == "__main__":
    main()
