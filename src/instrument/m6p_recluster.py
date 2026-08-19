r"""M6p -- recreate the CLEAN taste clusters, and BANK THE LABELS this time.

Author, 2026-08-19: *"where did the clusters go then? can you recreate clean ones?"*

WHERE THEY WENT. M1 clustered validation users by taste with the release-era axis projected out and
reported six coherent groups. The pipeline then MERGED them -- `m1_train_clusters` ran start_k=40 to
final_k=4 under a `min_size >= 1200` gate -- and only the merged labels were written to disk. The six
groups survive as a table in `M1_TASTE_CLUSTERS.md`; their labels were never banked. Inspecting what
IS on disk shows the cost: the k=4 partition is era structure again -- c0 is a 1995-96 cohort
(*Up Close and Personal*, *Sudden Death*, *Broken Arrow*), c3 a 1997-99 one -- because merging up to
clear 1,200 re-admitted exactly the axis the projection had removed.

WHAT THIS DOES. The same construction, without the merge, with the labels written out:

  1. validation states `z_full` (banked, all 10,000 users, their own fold-in half folded);
  2. project out `d_era = Wd^T y` (y = standardised release year) -- `m1_deera`'s axis, rebuilt here
     rather than assumed, and its rank correlation with year re-reported as the check;
  3. cosine k-means at `--k` (the memory-safe form: rows L2-normalised, distance via 2-2cos, never
     the pairwise-difference tensor that materialised 1.6 GB at n=140,768);
  4. characterise every cluster by CENTROID LIFT over films and attributes -- the measure validated
     against Reddit naming in M6m (real named films land at percentile 76-91 of their own cluster's
     ranking against ~50 for random);
  5. report an ERA CHECK per cluster -- the mean release year of its distinctive films, and how far
     that sits from the global mean. A cluster whose identity is a decade is the failure mode this
     file exists to avoid, and it is measured rather than eyeballed.

NOTHING IS MERGED. Clusters below `--min_size` are reported and flagged, not folded into a neighbour:
merging is what destroyed the previous attempt, and a small coherent cluster is a more useful object
than a large incoherent one.

VALIDATION ONLY. The 10,000 canonical TEST users are never built, loaded or named.
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
import re
import time

import numpy as np
import torch

B = os.path.join(_R, "experiments", "battery")
M1 = os.path.join(B, "m1")
OUT = os.path.join(B, "m6")
SEED = 98765
ANCHORS = {"arthouse": 68, "dialogue": 275, "marvel": 581,
           "gore": 417, "teenagers": 922, "inspiring": 503}


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def main():
    ap = argparse.ArgumentParser(description="M6p -- recreate and bank clean taste clusters")
    ap.add_argument("--k", type=int, default=20)
    ap.add_argument("--min_size", type=int, default=400,
                    help="a cluster below this is FLAGGED, never merged")
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--i_have_author_goahead", action="store_true")
    a = ap.parse_args()
    assert a.i_have_author_goahead, "REFUSING TO RUN without --i_have_author_goahead"
    torch.set_num_threads(a.threads)
    os.makedirs(OUT, exist_ok=True)

    from strategy_channel_suite import build_greedy_env, CERT_TOWER, CERT_CONC
    from m1_ideal import kmeans
    from m3_show_paths import build_names
    from scipy import stats
    cname, iname = build_names()

    E = build_greedy_env(CERT_TOWER, "validation", CERT_CONC, "m6p")
    Wd = np.asarray(E["ctx"].Wd.detach().numpy(), np.float64)
    emb = np.asarray(E["net"].emb.weight.detach().numpy(), np.float64)

    # ---- the era axis, rebuilt (not assumed) ------------------------------------------------
    with open(os.path.join(_R, "data", "ml-25m", "proc", "unique_sid.txt")) as fh:
        sids = [int(x) for x in fh.read().split()]
    year = {}
    with open(os.path.join(_R, "data", "movielens", "movies.csv"), newline="",
              encoding="utf-8") as fh:
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
    rho = stats.spearmanr((d_era @ Wd.T)[ok], yr[ok]).statistic
    log(f"[era-axis] rho(decoded score, release year) = {rho:+.3f} -- the direction removed")

    zf = np.load(os.path.join(M1, "z_full.npz"))
    Z = zf["z"].astype(np.float64)
    rows = zf["rows"]
    Zd = Z - np.outer(Z @ d_era, d_era)
    log(f"[remove] state norm {np.linalg.norm(Z, axis=1).mean():.3f} -> "
        f"{np.linalg.norm(Zd, axis=1).mean():.3f} over {len(Zd)} validation users")

    lab, _C = kmeans(Zd, a.k, seed=SEED)
    sizes = {int(c): int((lab == c).sum()) for c in sorted(set(lab.tolist()))}
    log(f"[kmeans] k={a.k}: sizes {sizes}")

    gm = Zd.mean(0)
    gsi, gsc = Wd @ gm, emb @ gm
    ymean = float(np.nanmean(yr[ok]))
    V, F = np.asarray(E["V"]), np.asarray(E["foldable"])
    endorse = (F > 0) & (V > 0)
    out = []
    for c in sorted(sizes):
        sel = np.asarray([int(r) for r, l in zip(rows, lab) if l == c], np.int64)
        cm = Zd[lab == c].mean(0)
        li, lc = Wd @ cm - gsi, emb @ cm - gsc
        top_i = [int(i) for i in np.argsort(-li)[:12]]
        top_c = [int(i) for i in np.argsort(-lc)[:10]]
        yrs = [yr[i] for i in top_i if ok[i]]
        era_gap = float(np.mean(yrs) - ymean) if yrs else float("nan")
        era_sd = float(np.std(yrs)) if yrs else float("nan")
        anc = {an: float(np.mean(endorse[sel, cid])) for an, cid in ANCHORS.items()}
        best = max(anc, key=anc.get)
        rec = {"cluster": int(c), "n": int(sizes[c]),
               "films": [iname.get(i, str(i)) for i in top_i[:6]],
               "attrs": [cname.get(i, str(i)) for i in top_c],
               "mean_year_of_top_films": float(np.mean(yrs)) if yrs else None,
               "era_gap_years": era_gap, "era_sd_years": era_sd,
               "anchor_rates": anc, "best_anchor": best, "best_anchor_rate": anc[best],
               "below_min_size": bool(sizes[c] < a.min_size)}
        out.append(rec)
        flag = "  <-- SMALL" if rec["below_min_size"] else ""
        era = "  <-- ERA?" if abs(era_gap) > 8 and era_sd < 9 else ""
        log(f"  c{c:<3} n={sizes[c]:<5} yr {rec['mean_year_of_top_films']:.0f} "
            f"({era_gap:+.1f} vs global, sd {era_sd:.1f}){era}{flag}")
        log(f"       films: {', '.join(rec['films'])}")
        log(f"       attrs: {', '.join(rec['attrs'][:8])}")
        log(f"       best anchor: {best} {anc[best]:.2f}")

    np.savez_compressed(os.path.join(OUT, f"m6p_clusters_k{a.k}.npz"),
                        labels=lab.astype(np.int32), rows=rows.astype(np.int64),
                        d_era=d_era.astype(np.float32))
    json.dump({"analysis": "m6p_recluster", "split": "validation", "test_touched": False,
               "k": a.k, "min_size": a.min_size, "era_axis_rho": float(rho),
               "merged": False, "sizes": sizes, "clusters": out},
              open(os.path.join(OUT, f"m6p_clusters_k{a.k}.json"), "w"), indent=1)
    log(f"[done] labels BANKED to m6p_clusters_k{a.k}.npz -- this is what was missing before")


if __name__ == "__main__":
    main()
