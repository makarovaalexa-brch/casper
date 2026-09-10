"""A21 -- emit Goodreads in the RAW formats the ML-25M pipeline already expects.

Design sheet: docs/design/DS_A17_GOODREADS_ARENA.md

WHY THIS EXISTS, and why it replaces a18_goodreads_split.py.
`train_tower_t2.py` does not read the split alone: it RE-DERIVES the user partition from the raw
ratings file (`reproduce_partition`, and the catalogue assertion at :423-427) so the graded half-star
profiles line up with the binarised split. A hand-rolled split therefore drifts from what the
pipeline expects, however faithful its recipe looks. The faithful move is the other way round: emit
Goodreads in the RAW shapes, then run the project's OWN scripts unmodified.

So this writes two files that are drop-in replacements for the MovieLens ones:

  ratings.csv        userId, movieId, rating       <- the shape liang_split.py and train_tower_t2 read
  genome-scores.csv  movieId, tagId, relevance     <- the shape the concept channel reads

MAPPING DECISIONS, all stated:
  * movieId := the Goodreads csv_book_id of an arena book. Arena = the 18,430 most-rated books,
    matched to ML-25M's arena SIZE (a17_goodreads_arena.py).
  * rating  := the Goodreads 1-5 integer, unchanged. Liang's `rating > 3.5` predicate means `>= 4`
    on this scale, which is exactly what it means on MovieLens half-stars. No rescaling.
    ⚠ The tower's level table is 10-wide for half-stars (`LIKE_MIN_LEVEL = 7` <=> star >= 4.0).
    Goodreads integers occupy levels 1,3,5,7,9 of that same table, so `>= 7` still means `>= 4`
    stars and the like boundary is preserved. Levels 0,2,4,6,8 are simply never observed.
  * userId  := a sampled subset of eligible users, see below.
  * relevance := 1.0 for every (book, shelf) membership pair. The concept channel thresholds at
    >= 0.5 and then takes >= 30 members, so a binary membership is exactly what it consumes
    (`belief_layer.py:632`, `strategy_channel_suite.py:1422` builds with np.ones).

THE USER SAMPLE, which is the one reduction and is stated.
liang_split.py holds out 10,000 validation + 10,000 test and puts EVERY remaining user in train.
Goodreads has 683,636 eligible users, which would give a train cohort 4.7x ML-25M's and an
instrument that is better-resourced rather than comparable. So the raw file is emitted for
160,768 users sampled at seed 98765, and liang_split then yields 140,768 / 10,000 / 10,000 by its
own recipe -- the same cohort shape as ML-25M.

Run:  python src/instrument/a21_goodreads_rawform.py
      python scripts/baselines/liang_split.py --data goodreads
"""
import json
import os
import time

import numpy as np
import pandas as pd

_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
DATA = os.path.join(_ROOT, "data", "goodreads_ucsd")
ARENA = os.path.join(DATA, "arena")
SEED = 98765
LIKE_GT = 3.5
MIN_UC = 5
N_USERS = 160768          # 140,768 train + 10,000 val + 10,000 test, matching ML-25M
CATALOG_MIN_RATINGS = 20  # liang_split's own constant, applied to the emitted file


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def main():
    t0 = time.time()
    z = np.load(os.path.join(ARENA, "graded.npz"))
    u, i, r = z["user"], z["item"], z["rating"]
    log(f"[graded] {len(u):,} rated events on {len(np.unique(i)):,} arena books, "
        f"{len(np.unique(u)):,} users")

    like = r > LIKE_GT
    uu, cnt = np.unique(u[like], return_counts=True)
    elig = uu[cnt >= MIN_UC]
    log(f"[eligible] {len(elig):,} users with >= {MIN_UC} likes on arena books")
    if len(elig) < N_USERS:
        raise SystemExit(f"only {len(elig)} eligible, need {N_USERS}")

    rng = np.random.default_rng(SEED)
    pick = np.sort(rng.choice(elig, size=N_USERS, replace=False))
    log(f"[sample] {len(pick):,} users sampled at seed {SEED} "
        f"(liang_split will hold out 20,000 of them, leaving 140,768 train)")

    keep = np.isin(u, pick)
    U, I, R = u[keep], i[keep], r[keep]
    log(f"[emit] {len(U):,} rated events for the sampled users")

    # the catalogue assertion liang_split will apply to THIS file
    icnt = pd.Series(I).value_counts()
    n_cat = int((icnt >= CATALOG_MIN_RATINGS).sum())
    log(f"[catalogue] {n_cat:,} books have >= {CATALOG_MIN_RATINGS} ratings in the emitted file "
        f"(arena is {len(np.unique(i)):,}); min per-book count = {int(icnt.min())}")
    if n_cat != len(np.unique(i)):
        log(f"!! {len(np.unique(i)) - n_cat} arena books fall below {CATALOG_MIN_RATINGS} ratings "
            f"once restricted to the sampled users -- liang_split's CATALOG_N_EXPECTED for goodreads "
            f"must be set to {n_cat}, not {len(np.unique(i))}")

    out = os.path.join(DATA, "ratings.csv")
    pd.DataFrame({"userId": U, "movieId": I, "rating": R.astype(np.float32)}).to_csv(
        out, index=False)
    log(f"[wrote] {out} ({os.path.getsize(out)/1e6:.0f} MB)")

    # ---- genome-scores.csv shaped concept file ----
    cz = np.load(os.path.join(ARENA, "concepts.npz"), allow_pickle=True)
    M, tags, vocab = cz["M"], list(cz["tags"]), cz["vocab"]
    rows_i, rows_t = np.nonzero(M)
    gpath = os.path.join(DATA, "genome-scores.csv")
    pd.DataFrame({"movieId": vocab[rows_i],
                  "tagId": rows_t.astype(np.int32),
                  "relevance": np.ones(len(rows_i), np.float32)}).to_csv(gpath, index=False)
    log(f"[wrote] {gpath} ({len(rows_i):,} membership pairs, {len(tags):,} concepts)")
    pd.DataFrame({"tagId": np.arange(len(tags)), "tag": tags}).to_csv(
        os.path.join(DATA, "genome-tags.csv"), index=False)

    json.dump({"n_users_emitted": int(N_USERS), "n_events": int(len(U)),
               "n_arena_books": int(len(np.unique(i))),
               "catalog_n_at_min20": n_cat, "seed": SEED,
               "eligible_users": int(len(elig)),
               "n_concepts": len(tags), "n_membership_pairs": int(len(rows_i)),
               "rating_scale": "goodreads integers 1-5; Liang's rating>3.5 means >=4, as on "
                               "MovieLens half-stars; the tower's 10-wide level table is occupied "
                               "at levels 1,3,5,7,9 and LIKE_MIN_LEVEL=7 still means star>=4",
               "seconds": time.time() - t0},
              open(os.path.join(DATA, "rawform_meta.json"), "w"), indent=2)
    log(f"[done] {(time.time()-t0)/60:.1f}m")


if __name__ == "__main__":
    main()
