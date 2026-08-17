r"""M1 step 4 -- cluster the REDDIT ARRIVAL states and compare them with the taste partition.

Author's four-step design, step 4: *"cluster based on reddit openers and compare drift"*.

Steps 1-3 (`m1_sweep.py`, `M1_TASTE_CLUSTERS.md`) partitioned users by TASTE, from their full liked
history, and named each group by an attribute. That is where people BELONG. This file asks where
people ARRIVE: fold each real request's own volunteered bundle and partition THAT.

THE COMPARISON IS THE POINT
  - Reddit arrival states are folded from ~2.9 named films plus any stated genre/tone; a train taste
    state is folded from ~62 liked films. Different evidence quantities, so the comparison is made
    on DIRECTION (states are L2-normalised), and norms are reported separately.
  - The era axis is projected out of the request states with the SAME `d_era` used on the train
    states (`m1_deera.json`), or the two populations would sit in different geometries. Real requests
    name recent films disproportionately, so era is expected to matter MORE here, not less.

OUTPUTS
  1. how the 3,078 arrival states distribute over the TASTE clusters (nearest centroid);
  2. whether arrival states have their own cluster structure, and how it aligns (contingency +
     adjusted Rand index against the nearest-taste-cluster assignment);
  3. per taste cluster, the share of real requests that land there -- i.e. which teachers real
     traffic would actually reach.

Read-only w.r.t. the recommender: no NDCG, no targets, no TEST.
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
import time

import numpy as np
import torch

M1 = os.path.join(_R, "experiments", "battery", "m1")
TRAIN_DIR = os.path.join(M1, "train")
TAGS = os.path.join(_R, "data", "movielens", "genome-tags.csv")
MOVIES = os.path.join(_R, "data", "movielens", "movies.csv")
SEED = 98765


def log(m: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def adjusted_rand(a, b):
    """ARI without sklearn."""
    a = np.asarray(a); b = np.asarray(b)
    ka, kb = a.max() + 1, b.max() + 1
    cont = np.zeros((ka, kb), np.int64)
    for x, y in zip(a, b):
        cont[x, y] += 1
    n = len(a)
    def c2(x):
        return x * (x - 1) / 2.0
    sij = c2(cont).sum()
    si = c2(cont.sum(1)).sum()
    sj = c2(cont.sum(0)).sum()
    exp = si * sj / c2(n)
    mx = (si + sj) / 2.0
    return float((sij - exp) / (mx - exp)) if mx != exp else 0.0, cont


def main() -> None:
    ap = argparse.ArgumentParser(description="M1 step 4 -- Reddit arrival-state clustering")
    ap.add_argument("--taste_k", type=int, default=20)
    ap.add_argument("--req_k", type=int, default=8)
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--i_have_author_goahead", action="store_true")
    a = ap.parse_args()
    assert a.i_have_author_goahead, "REFUSING TO RUN without --i_have_author_goahead"
    torch.set_num_threads(a.threads); np.random.seed(SEED)

    from strategy_channel_suite import build_greedy_env, CERT_TOWER, CERT_CONC
    from m1_ideal import kmeans

    # ---- the two populations, in one geometry ---------------------------------------------------
    files = sorted((f for f in os.listdir(TRAIN_DIR) if f.startswith("deera_")),
                   key=lambda x: int(x.split("_")[1]))
    Ztr = np.concatenate([np.load(os.path.join(TRAIN_DIR, f))["z"] for f in files], 0)
    req = np.load(os.path.join(M1, "z_requests.npz"))
    Zrq_raw = req["z"]
    tids = req["thread_ids"]
    n_items = req["n_items"]
    log(f"[in] train de-era'd {Ztr.shape} · requests {Zrq_raw.shape}")

    E = build_greedy_env(CERT_TOWER, "validation", CERT_CONC, "m1r4")
    Wd = np.asarray(E["ctx"].Wd); bd = np.asarray(E["ctx"].bd)
    with open(os.path.join(_R, "data", "ml-25m", "proc", "unique_sid.txt")) as fh:
        sids = [int(x) for x in fh.read().split()]
    import re
    year = {}
    with open(MOVIES, newline="", encoding="utf-8") as fh:
        rows_m = list(csv.DictReader(fh))
    mt = {int(r["movieId"]): r["title"] for r in rows_m}
    for r in rows_m:
        m = re.search(r"\((\d{4})\)\s*$", r["title"].strip())
        if m:
            year[int(r["movieId"])] = int(m.group(1))
    iname = {i: mt.get(m, str(m)) for i, m in enumerate(sids)}
    yr = np.array([year.get(s, np.nan) for s in sids], float)
    ok = ~np.isnan(yr)
    y = np.zeros(len(sids)); y[ok] = (yr[ok] - yr[ok].mean()) / yr[ok].std()
    d_era = Wd.T @ y; d_era /= np.linalg.norm(d_era)

    Zrq = Zrq_raw - np.outer(Zrq_raw @ d_era, d_era)
    log(f"[era] request states projected onto the SAME era axis; "
        f"mean |proj| train-scale check: requests {np.abs(Zrq_raw @ d_era).mean():.3f}")
    log(f"[norm] state norms: train {np.linalg.norm(Ztr,axis=1).mean():.2f} · "
        f"requests {np.linalg.norm(Zrq,axis=1).mean():.2f} "
        f"(requests carry ~2.9 films, train ~62 -- compare DIRECTION, not magnitude)")

    def unit(X):
        return X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-9)

    # ---- 1. where do arrivals land in the TASTE partition? --------------------------------------
    lab_tr, _ = kmeans(Ztr, a.taste_k)
    cent = np.stack([Ztr[lab_tr == i].mean(0) for i in range(a.taste_k)])
    cent_u = unit(cent)
    assign = np.argmax(unit(Zrq) @ cent_u.T, axis=1)
    share_req = np.bincount(assign, minlength=a.taste_k) / len(assign)
    share_tr = np.bincount(lab_tr, minlength=a.taste_k) / len(lab_tr)
    log("[land] taste cluster : train share -> request share (ratio)")
    order = np.argsort(-share_req)
    for i in order:
        log(f"    c{i:<2} n_train={int((lab_tr==i).sum()):>6}  train {share_tr[i]:6.2%} -> "
            f"requests {share_req[i]:6.2%}  ({share_req[i]/max(share_tr[i],1e-9):5.2f}x)")

    # ---- 2. do arrivals have their OWN structure, and does it align? -----------------------------
    lab_rq, _ = kmeans(Zrq, a.req_k)
    ari, cont = adjusted_rand(lab_rq, assign)
    log(f"[align] adjusted Rand index between the arrival partition (k={a.req_k}) and the "
        f"nearest-taste-cluster assignment: {ari:+.4f}")
    log("[align] each arrival cluster's dominant taste cluster:")
    for j in range(a.req_k):
        row = cont[j]
        tot = row.sum()
        if tot == 0:
            continue
        top = int(np.argmax(row))
        cen = Zrq[lab_rq == j].mean(0)
        sc = cen @ Wd.T + bd
        gl = Zrq.mean(0) @ Wd.T + bd
        films = [iname.get(int(t), "?") for t in np.argsort(-(sc - gl))[:4]]
        log(f"    r{j} n={int(tot):>5} -> taste c{top} ({row[top]/tot:.0%} of it) · "
            f"films: {', '.join(films)}")

    res = {"analysis": "m1_step4_reddit_arrival_clusters", "taste_k": a.taste_k,
           "req_k": a.req_k, "n_requests": int(len(Zrq)), "test_touched": False,
           "adjusted_rand": ari,
           "train_share": share_tr.tolist(), "request_share": share_req.tolist(),
           "contingency": cont.tolist(),
           "state_norm_train": float(np.linalg.norm(Ztr, axis=1).mean()),
           "state_norm_requests": float(np.linalg.norm(Zrq, axis=1).mean())}
    json.dump(res, open(os.path.join(M1, "m1_reddit_clusters.json"), "w"), indent=1)
    np.savez_compressed(os.path.join(M1, "m1_reddit_assign.npz"),
                        thread_ids=tids, taste_cluster=assign, arrival_cluster=lab_rq,
                        n_items=n_items)
    log("[done] wrote m1_reddit_clusters.json")


if __name__ == "__main__":
    main()
