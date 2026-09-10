"""A17 -- build the Goodreads arena: catalogue, shelf vocabulary, and the graded rating store.

Design sheet: docs/design/DS_A17_GOODREADS_ARENA.md

MIRRORS the ML-25M construction, with ONE calibrated deviation that the author signed off:

  ML-25M  raw 59,047 films -> keep films with >= 20 ratings -> arena 18,430 -> Liang user-side
          verbatim -> train vocab 18,359. That >= 20 rule keeps 99.2% of all rating events.
  GOODREADS  raw 2,360,650 books. The same >= 20 rule leaves 434,531 books, which cannot be
          trained on this box. Matching the rule is infeasible and matching the coverage (99.2%)
          needs >1.1M books, so we match the ARENA SIZE: the 18,430 most-rated books. This makes
          the ranking problem the same size in both datasets, which is what makes the replication
          comparable. It keeps 56.7% of rating events -- stated, not hidden, because Goodreads is a
          raw crawl with a far longer tail than a curated MovieLens catalogue.

Nothing else is reduced. All 816,371 rated users are candidates for the cohorts; the rating scale is
kept at full resolution; no sampling anywhere.

Outputs, under data/goodreads_ucsd/arena/:
  catalogue.csv     csv_book_id, goodreads_book_id, n_ratings, rank      (the 18,430-book arena)
  shelves.csv       csv_book_id, shelf, count                            (content shelves only)
  graded.npz        user, item, rating  -- ALL bands, arena books only   (the answer oracle)

Run:  python src/instrument/a17_goodreads_arena.py
"""
import gzip
import json
import os
import re
import time

import numpy as np
import pandas as pd

_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
DATA = os.path.join(_ROOT, "data", "goodreads_ucsd")
ARENA = os.path.join(DATA, "arena")
CHUNK = 20_000_000
ARENA_N = 18430          # ML-25M's arena size, matched exactly
TOP_SHELVES_PER_BOOK = 200   # generous; the concept bank applies its own top-K and floor later

# Process / non-content shelves. Extends scripts/_archive/paper2_old/gr_books.py, written for this
# same job on the same corpus.
JUNK_EXACT = set("""
to-read currently-reading read books owned own i-own my-books favorites favourites
default all-time-favorites my-library library wish-list wishlist to-buy bought
kindle ebook ebooks audiobook audiobooks audio paperback hardcover book-club
series finished abandoned dnf did-not-finish unfinished re-read reread
english books-i-own my-book shelfari-favorites giveaway giveaways netgalley arc arcs
general adult fiction non-fiction nonfiction novels novel literature books-read
""".split())
JUNK_SUBSTR = ("to-read", "own", "wish", "buy", "shelf", "kindle", "audio", "read-in-",
               "reading-", "-read", "read-", "favorite", "favourite", "library", "tbr",
               "20", "19", "star", "rating", "review", "club", "challenge", "borrowed",
               "default", "list", "have", "want", "need", "maybe", "someday", "next")


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def is_junk(name):
    n = str(name).strip().lower()
    if not n or n in JUNK_EXACT or len(n) <= 2:
        return True
    if not re.search(r"[a-z]", n):
        return True
    return any(sub in n for sub in JUNK_SUBSTR)


def main():
    os.makedirs(ARENA, exist_ok=True)
    t0 = time.time()
    n_books = sum(1 for _ in open(os.path.join(DATA, "book_id_map.csv"), encoding="utf-8")) - 1

    # ---------- pass 1: rating counts per book, and bank the graded events for the arena ----------
    bcnt = np.zeros(n_books + 1, np.int64)
    parts = []
    rdr = pd.read_csv(os.path.join(DATA, "goodreads_interactions.csv"),
                      usecols=["user_id", "book_id", "rating"],
                      dtype={"user_id": np.int32, "book_id": np.int32, "rating": np.int8},
                      chunksize=CHUNK)
    for i, ch in enumerate(rdr):
        ch = ch[ch.rating.values > 0]                 # rating 0 = shelved but never rated
        np.add.at(bcnt, ch.book_id.values, 1)
        parts.append(ch.to_numpy(np.int32))
        log(f"  pass1 chunk {i+1}: {sum(len(p) for p in parts):,} rated rows "
            f"({(time.time()-t0)/60:.1f}m)")
    A = np.vstack(parts)
    del parts
    log(f"[pass1] {len(A):,} rated events over {(bcnt>0).sum():,} books")

    # ---------- the arena: the ARENA_N most-rated books ----------
    order = np.argsort(-bcnt, kind="stable")[:ARENA_N]
    keep = np.zeros(n_books + 1, bool)
    keep[order] = True
    share = bcnt[order].sum() / bcnt.sum()
    log(f"[arena] {ARENA_N} most-rated books keep {share:.1%} of rating events; "
        f"min ratings in arena = {bcnt[order].min()}, max = {bcnt[order].max()}")

    bmap = pd.read_csv(os.path.join(DATA, "book_id_map.csv"))
    csv2gr = dict(zip(bmap.book_id_csv.values, bmap.book_id.values))
    cat = pd.DataFrame({"csv_book_id": order,
                        "goodreads_book_id": [csv2gr.get(int(x), -1) for x in order],
                        "n_ratings": bcnt[order]})
    cat["rank"] = np.arange(len(cat))
    cat.to_csv(os.path.join(ARENA, "catalogue.csv"), index=False)
    log(f"[arena] wrote catalogue.csv ({len(cat)} books, "
        f"{(cat.goodreads_book_id<0).sum()} unmapped)")

    # ---------- the graded store, arena books only ----------
    sel = keep[A[:, 1]]
    G = A[sel]
    np.savez_compressed(os.path.join(ARENA, "graded.npz"),
                        user=G[:, 0], item=G[:, 1], rating=G[:, 2].astype(np.int8))
    nu = len(np.unique(G[:, 0]))
    log(f"[graded] {len(G):,} rated events on arena books, {nu:,} users "
        f"({len(G)/max(nu,1):.1f} per user)")
    del A, G

    # ---------- shelves for the arena books ----------
    want = {int(csv2gr.get(int(x), -1)) for x in order}
    want.discard(-1)
    rows = []
    n_seen = 0
    with gzip.open(os.path.join(DATA, "goodreads_books.json.gz"), "rt", encoding="utf-8") as f:
        for line in f:
            n_seen += 1
            if n_seen % 400000 == 0:
                log(f"  shelves: {n_seen:,} books scanned, {len(rows):,} rows "
                    f"({(time.time()-t0)/60:.1f}m)")
            j = line.find('"book_id"')
            if j < 0:
                continue
            d = json.loads(line)
            try:
                bid = int(d["book_id"])
            except (KeyError, ValueError, TypeError):
                continue
            if bid not in want:
                continue
            sh = d.get("popular_shelves") or []
            got = 0
            for s in sh:
                nm = s.get("name")
                if is_junk(nm):
                    continue
                try:
                    c = int(s.get("count", 0))
                except (TypeError, ValueError):
                    continue
                rows.append((bid, nm, c))
                got += 1
                if got >= TOP_SHELVES_PER_BOOK:
                    break
    sdf = pd.DataFrame(rows, columns=["goodreads_book_id", "shelf", "count"])
    sdf = sdf.merge(cat[["csv_book_id", "goodreads_book_id"]], on="goodreads_book_id", how="inner")
    sdf[["csv_book_id", "shelf", "count"]].to_csv(os.path.join(ARENA, "shelves.csv"), index=False)
    log(f"[shelves] {len(sdf):,} (book, shelf) rows over {sdf.csv_book_id.nunique():,} arena books; "
        f"{sdf.shelf.nunique():,} distinct content shelves")

    meta = {"dataset": "goodreads-ucsd-full", "arena_n": ARENA_N,
            "arena_share_of_ratings": float(share),
            "arena_min_ratings": int(bcnt[order].min()),
            "graded_events": int(len(sdf)) and int(nu) and int(sel.sum()),
            "graded_users": int(nu),
            "distinct_shelves": int(sdf.shelf.nunique()),
            "books_with_shelves": int(sdf.csv_book_id.nunique()),
            "calibration": ("arena size matched to ML-25M's 18,430-film arena so the ranking "
                            "problem is the same size in both datasets; matching ML-25M's >=20 "
                            "rating RULE would leave 434,531 books and matching its 99.2% coverage "
                            "would need >1.1M, neither trainable here"),
            "seconds": time.time() - t0}
    json.dump(meta, open(os.path.join(ARENA, "arena_meta.json"), "w"), indent=2)
    log(f"\n[done] {ARENA} ({meta['seconds']/60:.1f}m)")


if __name__ == "__main__":
    main()
