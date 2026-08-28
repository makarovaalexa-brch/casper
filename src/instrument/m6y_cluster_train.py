r"""M6y -- TASTE CLUSTERS FITTED ON TRAIN, APPLIED TO VALIDATION.

The partition the router will read. `m6p_recluster` clusters VALIDATION users, but every teacher in
this project is fitted on TRAIN and reported on VALIDATION, so the partition must be defined the same
way or the cluster teachers would be fitted on one population's geometry and scored on another's.

WHAT THIS DOES
  1. cold-state latents for the TRAIN slice (each user's own fold-in half folded, no opener)
  2. the release-era axis `d_era = Wd^T y`, REBUILT from the decoder and release years rather than
     assumed, its rank correlation with year reported as the check that it is the era direction,
     and then PROJECTED OUT. This is the step the author asked for: era variation removed before
     clustering, not after.
  3. cosine k-means on the era-removed train states -> the partition, with CENTROIDS banked
  4. VALIDATION users assigned to those same centroids, after removing the same axis. The partition
     is therefore FITTED ON TRAIN and merely APPLIED to validation, like every other arm.
  5. per-cluster era diagnostics, so residual era structure is disclosed rather than discovered

⚠ Projecting the era axis out does NOT guarantee era-free clusters -- the axis is one direction and
release year can survive in others. `m6p_recluster` on validation left 6 of 20 clusters era-like
(|mean-year gap| > 8 yrs and sd < 9 yrs) after the same projection. That residue is reported here
per cluster and must be disclosed in the paper, not buried.

**THE TEST SET IS NEVER BUILT, LOADED OR NAMED IN THIS FILE.**
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
from scipy import stats

B = os.path.join(_R, "experiments", "battery")
M1 = os.path.join(B, "m1")
M6 = os.path.join(B, "m6")
SEED = 98765


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def kmeans(X, k, seed=SEED, iters=60):
    """Cosine k-means: rows L2-normalised, distance via 2-2cos. Never materialises a pairwise tensor."""
    rng = np.random.default_rng(seed)
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
    C = Xn[rng.permutation(len(Xn))[:k]].copy()
    lab = np.zeros(len(Xn), np.int32)
    for it in range(iters):
        sim = Xn @ C.T
        new = sim.argmax(1).astype(np.int32)
        if it and (new == lab).all():
            break
        lab = new
        for c in range(k):
            m = lab == c
            if m.sum():
                v = Xn[m].mean(0)
                C[c] = v / (np.linalg.norm(v) + 1e-12)
            else:                                   # re-seed an empty cluster on the worst-fit point
                C[c] = Xn[np.argmin(sim.max(1))]
    return lab, C


def main():
    ap = argparse.ArgumentParser(description="M6y -- train-fitted taste clusters")
    ap.add_argument("--k", type=int, default=20)
    ap.add_argument("--slice_lo", type=int, default=0)
    ap.add_argument("--slice_hi", type=int, default=34000)
    ap.add_argument("--chunk", type=int, default=17000)
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--i_have_author_goahead", action="store_true")
    a = ap.parse_args()
    assert a.i_have_author_goahead, "REFUSING TO RUN without --i_have_author_goahead"
    torch.set_num_threads(a.threads)

    from strategy_channel_suite import (build_greedy_env, genv_eval_multi, CERT_TOWER, CERT_CONC)
    from run_guard import require_mains
    require_mains(f"m6y clustering k={a.k}", log=log)

    # ---- the era axis, rebuilt from the decoder and release years ---------------------------------
    Ev = build_greedy_env(CERT_TOWER, "validation", CERT_CONC, "m6y_val")
    Wd = np.asarray(Ev["ctx"].Wd.detach().numpy(), np.float64)   # same access as m6p_recluster
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
    rho = float(stats.spearmanr((d_era @ Wd.T)[ok], yr[ok]).statistic)
    log(f"[era-axis] rho(decoded score, release year) = {rho:+.3f} -- this is the direction REMOVED")

    # ---- validation states (banked) ---------------------------------------------------------------
    zf = np.load(os.path.join(M1, "z_full.npz"))
    Zv, vrows = zf["z"].astype(np.float64), zf["rows"]
    del Ev

    # ---- TRAIN cold-state latents -----------------------------------------------------------------
    Zt, trows = [], []
    for lo in range(a.slice_lo, a.slice_hi, a.chunk):
        hi = min(lo + a.chunk, a.slice_hi)
        E = build_greedy_env(CERT_TOWER, "train", CERT_CONC, f"m6y_{lo}", cohort_slice=(lo, hi))
        rows = np.asarray(E["rows"], np.int64)
        z = np.asarray(genv_eval_multi(E, rows, [[] for _ in rows], [[] for _ in rows],
                                       score=False, return_z=True)[-1], np.float64)
        Zt.append(z); trows.append(rows + lo)          # GLOBAL train ids, not slice-local
        log(f"[train] slice {lo}:{hi} -> {len(rows)} cold-state latents")
        del E
    Zt = np.concatenate(Zt); trows = np.concatenate(trows)

    # ---- remove the era axis from BOTH, then cluster TRAIN ----------------------------------------
    Ztd = Zt - np.outer(Zt @ d_era, d_era)
    Zvd = Zv - np.outer(Zv @ d_era, d_era)
    log(f"[remove] train state norm {np.linalg.norm(Zt, axis=1).mean():.3f} -> "
        f"{np.linalg.norm(Ztd, axis=1).mean():.3f} over {len(Ztd)} users")
    lab_t, C = kmeans(Ztd, a.k, seed=SEED)
    Cn = C / (np.linalg.norm(C, axis=1, keepdims=True) + 1e-12)
    Zvn = Zvd / (np.linalg.norm(Zvd, axis=1, keepdims=True) + 1e-12)
    lab_v = (Zvn @ Cn.T).argmax(1).astype(np.int32)    # validation ASSIGNED, never re-fitted

    # ---- describe each cluster, and disclose the era residue --------------------------------------
    gm = Ztd.mean(0)
    gsi = Wd @ gm
    ymean = float(np.nanmean(yr[ok]))
    out, era_like = [], 0
    log("=" * 100)
    log(f"{'cl':>3} {'n_train':>8} {'n_val':>6} {'era_gap':>8} {'era_sd':>7}  era-like?")
    for c in range(a.k):
        m = lab_t == c
        cm = Ztd[m].mean(0)
        li = Wd @ cm - gsi
        top_i = [int(i) for i in np.argsort(-li)[:12]]
        yrs = [yr[i] for i in top_i if ok[i]]
        gap = float(np.mean(yrs) - ymean) if yrs else float("nan")
        sd = float(np.std(yrs)) if yrs else float("nan")
        flag = abs(gap) > 8 and sd < 9
        era_like += bool(flag)
        nv = int((lab_v == c).sum())
        out.append({"cluster": int(c), "n_train": int(m.sum()), "n_val": nv,
                    "era_gap_years": gap, "era_sd_years": sd, "era_like": bool(flag),
                    "top_items": top_i})
        log(f"{c:>3} {int(m.sum()):>8} {nv:>6} {gap:>+8.1f} {sd:>7.1f}  {'YES' if flag else 'no':>9}")
    log("=" * 100)
    log(f"[era] {era_like} of {a.k} clusters remain era-like after the projection "
        f"(validation-side reference: 6 of 20). DISCLOSE THIS, do not bury it.")

    np.savez_compressed(os.path.join(M6, f"m6y_clusters_train_k{a.k}.npz"),
                        train_rows=trows.astype(np.int64), train_labels=lab_t,
                        val_rows=vrows.astype(np.int64), val_labels=lab_v,
                        centroids=Cn.astype(np.float32), d_era=d_era.astype(np.float32))
    json.dump({"analysis": "m6y_cluster_train", "fitted_on": "train", "applied_to": "validation",
               "test_touched": False, "k": a.k, "slice": [a.slice_lo, a.slice_hi],
               "era_axis_rho": rho, "n_era_like": era_like, "clusters": out},
              open(os.path.join(M6, f"m6y_clusters_train_k{a.k}.json"), "w"), indent=1)
    big = sorted([o for o in out if not o["era_like"]], key=lambda o: -o["n_train"])
    log(f"[gate] largest NON-era-like cluster: c{big[0]['cluster']} "
        f"({big[0]['n_train']} train / {big[0]['n_val']} val users) -- the gate candidate")
    log("[done]")


if __name__ == "__main__":
    main()
