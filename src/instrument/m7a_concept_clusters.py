r"""M7a -- TASTE CLUSTERS IN CONCEPT-DECODED LATENT SPACE.

WHY NOT RAW LATENTS. `m6y` clustered the latents directly and got release-year cohorts: 14 of 20
clusters had their distinctive films spanning under 2.5 years (c12 all 1994, c7 all 1995-96, c14 all
1993-95). In MovieLens a user's film set is dominated by WHEN THEY USED THE SERVICE -- people rate
what was available while they were active -- so the dominant variance in `z` is joining cohort, not
taste. Projecting out one linear era direction cannot fix that; era lives in many directions.

WHY NOT RAW CONCEPT VALUES EITHER. The obvious alternative, clustering the behavioural concept matrix
`V`, would give coherent vibe groups but they would NOT be a function of `z` -- and the router only
ever sees the folded opener state. That is exactly how endorsement cohorts died: a linear probe
recovered the cohort from the opener state at only 0.371 on the worst group, and the policy matched
it at 0.364, i.e. it was at the information ceiling, not the optimisation ceiling.

WHAT THIS DOES. Clusters the CONCEPT-DECODED latent, `emb @ z`, which satisfies both constraints:

  * it is a DETERMINISTIC FUNCTION OF z, so the policy can compute it from the folded opener and
    separability is near-guaranteed by construction rather than hoped for;
  * it is expressed in vibe coordinates -- `dark humor`, `feel-good`, `gritty` -- where "1994" is not
    a coordinate, so era variance is compressed relative to film-identity variance.

It stays nameable: a cluster is described by the top concepts of its centroid, which is what makes
"chick flicks" or "family films" expressible at all.

★ THE ERA TEST IS `sd` ALONE. `m6y` flagged a cluster era-like only if |mean-year gap| > 8 AND
sd < 9, which misses a cluster whose films are all from one year sitting AT the global mean -- c12
(everything 1994) passed as clean. That under-reported the problem by more than 2x. A cluster whose
distinctive films span under ~3 years is an era bucket wherever its mean sits.

VERDICT ONLY. This file clusters and diagnoses. It fits no teacher and reports no NDCG.

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

B = os.path.join(_R, "experiments", "battery")
M1 = os.path.join(B, "m1")
M6 = os.path.join(B, "m6")
SEED = 98765
ERA_SD_MAX = 3.0            # distinctive films spanning less than this = an era bucket


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def kmeans(X, k, seed=SEED, iters=80):
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
            else:
                C[c] = Xn[np.argmin(sim.max(1))]
    return lab, C


def main():
    ap = argparse.ArgumentParser(description="M7a -- concept-space taste clusters")
    ap.add_argument("--k", type=int, default=12)
    ap.add_argument("--slice_hi", type=int, default=34000)
    ap.add_argument("--chunk", type=int, default=17000)
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--i_have_author_goahead", action="store_true")
    a = ap.parse_args()
    assert a.i_have_author_goahead, "REFUSING TO RUN without --i_have_author_goahead"
    torch.set_num_threads(a.threads)

    from strategy_channel_suite import build_greedy_env, CERT_TOWER, CERT_CONC
    from concept_fold import fold_items_frozen
    from i26_encoder import UNSEEN_LEVEL
    from m3_show_paths import build_names
    from run_guard import require_mains
    cname, iname = build_names()
    require_mains(f"m7a concept clustering k={a.k}", log=log)
    LIKED_MIN = 7

    Ev = build_greedy_env(CERT_TOWER, "validation", CERT_CONC, "m7a_val")
    Wd = np.asarray(Ev["ctx"].Wd.detach().numpy(), np.float64)          # item decoder
    emb = np.asarray(Ev["net"].emb.weight.detach().numpy(), np.float64)  # CONCEPT decoder
    log(f"[decoder] items {Wd.shape}  concepts {emb.shape}")

    # release years, for the era diagnostic only -- never used in the clustering
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
    ymean = float(np.nanmean(yr[ok]))

    zf = np.load(os.path.join(M1, "z_full.npz"))
    Zv, vrows = zf["z"].astype(np.float64), zf["rows"]
    del Ev

    # ---- TRAIN latents, folded from each user's own liked history ---------------------------------
    Zt, trows = [], []
    for lo in range(0, a.slice_hi, a.chunk):
        hi = min(lo + a.chunk, a.slice_hi)
        E = build_greedy_env(CERT_TOWER, "train", CERT_CONC, f"m7a_{lo}", cohort_slice=(lo, hi))
        rows = np.asarray(E["rows"], np.int64)
        lvl = E["lvl"]
        seqs = []
        for r in rows:
            d = lvl[int(r)]
            lk = [(int(i), int(L)) for i, L in d.items() if LIKED_MIN <= int(L) < UNSEEN_LEVEL]
            ids = np.asarray([i for i, _ in lk], np.int64) if lk else np.zeros(0, np.int64)
            lv = np.asarray([L for _, L in lk], np.int64) if lk else np.zeros(0, np.int64)
            seqs.append((ids, lv))
        Zt.append(np.asarray(fold_items_frozen(E["ctx"].enc, seqs, batch=500).numpy(), np.float64))
        trows.append(rows + lo)
        log(f"[train] slice {lo}:{hi} -> {len(rows)} states")
        del E
    Zt = np.concatenate(Zt); trows = np.concatenate(trows)

    # ---- DECODE TO CONCEPT SPACE, then cluster ----------------------------------------------------
    Ct = Zt @ emb.T                     # (n_users, n_concepts) -- a function of z, in vibe coords
    Cv = Zv @ emb.T
    log(f"[decode] concept profiles: train {Ct.shape}, validation {Cv.shape}")
    lab_t, K = kmeans(Ct, a.k, seed=SEED)
    sz = np.bincount(lab_t, minlength=a.k)
    assert sz.max() < 0.5 * len(lab_t), f"k-means COLLAPSED: sizes {sz.tolist()}"
    assert (sz > 0).sum() >= a.k - 1, f"empty clusters: {sz.tolist()}"
    Kn = K / (np.linalg.norm(K, axis=1, keepdims=True) + 1e-12)
    Cvn = Cv / (np.linalg.norm(Cv, axis=1, keepdims=True) + 1e-12)
    lab_v = (Cvn @ Kn.T).argmax(1).astype(np.int32)

    # ---- describe: what vibe is this, and is it an era bucket in disguise? -------------------------
    gm_i = Wd @ Zt.mean(0)
    gm_c = emb @ Zt.mean(0)
    out, n_era = [], 0
    log("=" * 104)
    for c in range(a.k):
        m = lab_t == c
        cm = Zt[m].mean(0)
        li, lc = Wd @ cm - gm_i, emb @ cm - gm_c
        top_i = [int(i) for i in np.argsort(-li)[:12]]
        top_c = [int(i) for i in np.argsort(-lc)[:8]]
        yrs = [yr[i] for i in top_i if ok[i]]
        gap = float(np.mean(yrs) - ymean) if yrs else float("nan")
        sd = float(np.std(yrs)) if yrs else float("nan")
        era = sd < ERA_SD_MAX
        n_era += bool(era)
        out.append({"cluster": int(c), "n_train": int(m.sum()),
                    "n_val": int((lab_v == c).sum()), "era_gap_years": gap, "era_sd_years": sd,
                    "era_bucket": bool(era),
                    "concepts": [cname.get(i, str(i)) for i in top_c],
                    "films": [iname.get(i, str(i)) for i in top_i[:5]]})
        log(f"c{c:<3} n={int(m.sum()):<5} val={int((lab_v == c).sum()):<4} "
            f"sd={sd:>5.1f}{'  ERA-BUCKET' if era else '            '}")
        log(f"      vibes: {', '.join(cname.get(i, str(i)) for i in top_c)}")
        log(f"      films: {'; '.join(iname.get(i, str(i))[:34] for i in top_i[:4])}")
    log("=" * 104)
    log(f"[era] {n_era} of {a.k} clusters have distinctive films spanning < {ERA_SD_MAX} years "
        f"(m6y's RAW-LATENT clustering: 14 of 20). Lower is better.")
    np.savez_compressed(os.path.join(M6, f"m7a_concept_clusters_k{a.k}.npz"),
                        train_rows=trows.astype(np.int64), train_labels=lab_t,
                        val_rows=vrows.astype(np.int64), val_labels=lab_v,
                        centroids=Kn.astype(np.float32))
    json.dump({"analysis": "m7a_concept_clusters", "space": "concept-decoded latent (emb @ z)",
               "fitted_on": "train", "applied_to": "validation", "test_touched": False,
               "k": a.k, "era_sd_max": ERA_SD_MAX, "n_era_bucket": n_era, "clusters": out},
              open(os.path.join(M6, f"m7a_concept_clusters_k{a.k}.json"), "w"), indent=1)
    log("[done]")


if __name__ == "__main__":
    main()
