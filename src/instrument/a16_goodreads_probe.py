"""A16 probe -- can the FULL UCSD Goodreads book graph carry Paper B's experiment?

Design sheet: docs/design/DS_A15_GOODBOOKS_PROBE.md (same pre-flight, new candidate)

WHY THIS CANDIDATE, after two rejected ones:
  * goodbooks-10k passed on the numbers but is a GitHub/Kaggle release with no paper to cite.
    Author's ruling: the dataset must be one people actively use, or reviewers read it as reaching.
  * The UCSD Goodreads GENRE subsets are citable but far too shallow -- the poetry subset has a
    MEDIAN OF 2 ratings per user, because a genre slice cuts each user's history down to one corner
    of it. Measured, not assumed: 2,734,350 interactions, 1,229,059 rated, 267,821 users, median 2.
  * The FULL book graph keeps the whole history: ~104.5M ratings over ~876k users. This probe
    measures whether that is really so, and what catalogue survives our own item floor.

THE TWO QUESTIONS, both of which decided against the previous candidates:
  1. DEPTH. ML-25M gives 141 fold-in items per user. Anything near single digits cannot support an
     interview, because no arm collects an answer and there is nothing to measure.
  2. CATALOGUE. We keep films with >= 20 ratings on ML-25M, which yields 18,430 items. The same
     floor on 2.36M books yields something much larger, and cost scales as (eval users x catalogue),
     so the size decides whether a greedy search is affordable at all.

NO TRUNCATION IS APPLIED HERE. This counts; it does not filter anything away for convenience. The
item floor is reported at several values so the choice is made on evidence and stated, not tuned.

Run:  python src/instrument/a16_goodreads_probe.py
"""
import json
import os
import time

import numpy as np
import pandas as pd

_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
DATA = os.path.join(_ROOT, "data", "goodreads_ucsd")
OUT = os.path.join(_ROOT, "experiments", "goodreads")
CHUNK = 20_000_000
ML25M = {"foldin_items_per_user": 141.2, "catalogue": 18359, "item_floor": 20,
         "item_answers_at_16": 3.11, "concept_answers_at_16": 12.36}


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def main():
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()
    n_books = sum(1 for _ in open(os.path.join(DATA, "book_id_map.csv"), encoding="utf-8")) - 1
    n_users = sum(1 for _ in open(os.path.join(DATA, "user_id_map.csv"), encoding="utf-8")) - 1
    log(f"[maps] {n_users} users, {n_books} books in the id maps")

    ucnt = np.zeros(n_users + 1, np.int32)      # ratings per user
    bcnt = np.zeros(n_books + 1, np.int32)      # ratings per book
    total = rated = 0
    rdr = pd.read_csv(os.path.join(DATA, "goodreads_interactions.csv"),
                      usecols=["user_id", "book_id", "rating"],
                      dtype={"user_id": np.int32, "book_id": np.int32, "rating": np.int8},
                      chunksize=CHUNK)
    for i, ch in enumerate(rdr):
        total += len(ch)
        m = ch.rating.values > 0                # rating 0 = shelved but not rated
        rated += int(m.sum())
        np.add.at(ucnt, ch.user_id.values[m], 1)
        np.add.at(bcnt, ch.book_id.values[m], 1)
        log(f"  chunk {i+1}: {total:,} rows read, {rated:,} rated ({(time.time()-t0)/60:.1f}m)")

    u = ucnt[ucnt > 0]
    log("")
    log(f"[totals] {total:,} interactions, {rated:,} RATED, "
        f"{len(u):,} users with >=1 rating, {(bcnt>0).sum():,} books with >=1 rating")
    log(f"[depth] ratings per user: mean {u.mean():.1f} median {np.median(u):.0f} "
        f"p10 {np.percentile(u,10):.0f} p25 {np.percentile(u,25):.0f} "
        f"p75 {np.percentile(u,75):.0f} p90 {np.percentile(u,90):.0f}   "
        f"(ML-25M fold-in: {ML25M['foldin_items_per_user']})")

    rep = {"dataset": "goodreads-ucsd-full", "ml25m_reference": ML25M,
           "interactions": int(total), "rated": int(rated),
           "users_with_rating": int(len(u)), "books_with_rating": int((bcnt > 0).sum()),
           "depth": {"mean": float(u.mean()), "median": float(np.median(u)),
                     "p10": float(np.percentile(u, 10)), "p25": float(np.percentile(u, 25)),
                     "p75": float(np.percentile(u, 75)), "p90": float(np.percentile(u, 90))},
           "users_at_floor": {}, "catalogue_at_floor": {}}

    log("")
    log("[users] how many users survive a minimum-ratings floor")
    for k in (5, 10, 20, 50, 100, 200):
        n = int((u >= k).sum())
        rep["users_at_floor"][str(k)] = n
        log(f"   >= {k:3d} ratings: {n:8,d} users")

    log("")
    log("[catalogue] how many books survive an item floor, and the mass they keep")
    for k in (5, 10, 20, 50, 100, 200, 500):
        keep = bcnt >= k
        n = int(keep.sum())
        mass = float(bcnt[keep].sum() / max(rated, 1))
        rep["catalogue_at_floor"][str(k)] = {"books": n, "share_of_ratings": mass}
        rel = n / ML25M["catalogue"]
        log(f"   >= {k:3d} ratings: {n:9,d} books  keeps {mass:6.1%} of ratings  "
            f"({rel:5.1f}x ML-25M's catalogue)")

    # cost implication: the dominant stages scale as (eval users x catalogue)
    log("")
    log("[cost] eval-bound stages scale as (eval users x catalogue); ML-25M = 10,000 x 18,359")
    base = 10000 * ML25M["catalogue"]
    for k in (20, 50, 100, 200):
        nb = rep["catalogue_at_floor"][str(k)]["books"]
        for nu in (5000, 10000):
            log(f"   floor {k:3d}, {nu:,} eval users: {nu*nb/base:5.2f}x ML-25M per evaluation")

    rep["seconds"] = time.time() - t0
    json.dump(rep, open(os.path.join(OUT, "a16_goodreads_probe.json"), "w"), indent=2)
    log(f"\n[done] {os.path.join(OUT, 'a16_goodreads_probe.json')} ({rep['seconds']/60:.1f}m)")


if __name__ == "__main__":
    main()
