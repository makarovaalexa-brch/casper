"""A19 -- the Goodreads concept bank, calibrated to ML-25M's per-item concept density.

Design sheet: docs/design/DS_A17_GOODREADS_ARENA.md

WHAT THE CONCEPT CHANNEL ACTUALLY NEEDS is a BINARY item-by-concept membership matrix plus a
>= 30-member floor. It never sees graded relevance: `belief_layer.py:632` thresholds the Tag Genome
at relevance >= 0.5 and throws the score away, and `strategy_channel_suite.py:1422` builds the matrix
with `np.ones(...)`. So a mined folksonomy is a drop-in, which is why Goodreads shelves can stand in
for the Tag Genome at all.

CALIBRATION, not tuning. The membership rule is "a book's top-K shelves by application count", and K
is chosen so that the CONCEPTS-PER-BOOK density matches ML-25M's, measured directly from
genome-scores.csv at relevance >= 0.5:

    ML-25M: 1,127 tags, 13,816 movies covered, median 37 / mean 44.5 concepts per movie,
            1,031 concepts surviving the >= 30-member floor.

Calibrating the density rather than the bank size is what makes a concept question mean a comparable
thing in the two datasets. The bank size is then whatever falls out, and is reported, not chosen.

A first attempt on goodbooks-10k used a fractional rule (a shelf counts if it is >= 2% of the book's
tag mass) and produced only 103 concepts: popular books carry enormous shelf mass, so a fractional
rule is a popularity filter in disguise. Recorded here so the same mistake is not repeated.

Outputs under data/goodreads_ucsd/arena/:
  concepts.json   {tags: [...], members: {tag: [sid, ...]}, K, density, floor}
  concepts.npz    membership matrix (n_items x n_concepts, uint8) + tag names

Run:  python src/instrument/a19_goodreads_concepts.py
"""
import json
import os
import time

import numpy as np
import pandas as pd

_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
DATA = os.path.join(_ROOT, "data", "goodreads_ucsd")
ARENA = os.path.join(DATA, "arena")
PROC = os.path.join(DATA, "proc")
CONC_FLOOR = 30
ML25M_DENSITY = {"median": 37, "mean": 44.5, "concepts": 1031, "tags": 1127}
K_GRID = (10, 15, 20, 25, 30, 37, 45, 50, 60, 80, 120)

# A SECOND junk pass, added after inspecting the first bank's highest-membership concepts.
# A17's filter caught reading-process shelves but let three classes through, all of which are
# highly answerable and carry NO taste signal -- exactly the way to inflate the concept channel's
# answer rate with vacuous questions:
#   FORMAT   "e-book", "audible", "nook", "calibre"  -- the reader's device, not the book
#   STRUCTURAL "part-of-a-series", "first-in-series", "stand-alone"
#   PROCESS  "didn-t-finish", "recommended"
#   LIST     "1001-books", "100-essential-novels", "33-keepers" -- list membership, not an attribute
# Dropping these is conservative for our own claim: it removes easy-to-answer questions from the
# bank, so the concept channel's measured answer rate can only fall.
JUNK2_EXACT = set("""
e-book e-books ebook ebooks nook calibre kobo epub pdf mobi hardback mass-market
part-of-a-series first-in-series stand-alone standalone stand-alones sequel sequels
didn-t-finish didnt-finish gave-up not-finished couldn-t-finish
recommended recommendations rec general-fiction adult-fiction fiction-general
signed borrowed loaned unread skimmed started
""".split())
JUNK2_SUBSTR = ("e-book", "ebook", "audible", "calibre", "kobo", "epub", "-pdf", "hardback",
                "mass-market", "in-series", "part-of", "stand-alone", "standalone",
                "didn-t", "didnt", "did-not", "gave-up", "give-up", "not-finish",
                "recommend", "1001", "100-", "-100", "top-", "best-of", "must-read",
                "bucket", "keepers", "essential", "boxed", "omnibus", "duplicate")


def is_junk2(name):
    """True for shelves that are format, structural, process or list membership rather than taste."""
    n = str(name).strip().lower()
    if n in JUNK2_EXACT:
        return True
    return any(sub in n for sub in JUNK2_SUBSTR)



def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def main():
    t0 = time.time()
    meta = json.load(open(os.path.join(PROC, "meta.json")))
    vocab = np.loadtxt(os.path.join(PROC, "unique_sid.txt"), dtype=np.int64)
    sid = {int(b): k for k, b in enumerate(vocab)}
    ni = len(vocab)
    log(f"[vocab] {ni} books in the train vocabulary")

    sh = pd.read_csv(os.path.join(ARENA, "shelves.csv"))
    sh = sh[sh.csv_book_id.isin(sid.keys())]
    sh["sid"] = sh.csv_book_id.map(sid)
    n0, t0_ = len(sh), sh.shelf.nunique()
    sh = sh[~sh.shelf.map(is_junk2)]
    log(f"[shelves] {len(sh):,} (book, shelf) rows, {sh.shelf.nunique():,} distinct shelves, "
        f"{sh.sid.nunique():,} books covered")
    log(f"[junk2] dropped {n0-len(sh):,} rows and {t0_-sh.shelf.nunique():,} shelves that are "
        f"format, structural, process or list membership rather than taste")

    sh = sh.sort_values("count", ascending=False)
    log("")
    log(f"[calibrate] choosing K so concepts-per-book matches ML-25M "
        f"(median {ML25M_DENSITY['median']}, mean {ML25M_DENSITY['mean']})")
    best, best_err = None, None
    for K in K_GRID:
        top = sh.groupby("sid").head(K)
        mem = top.groupby("shelf")["sid"].nunique()
        keep = set(mem[mem >= CONC_FLOOR].index)
        d = top[top.shelf.isin(keep)].groupby("sid").size()
        med, mean = float(d.median()), float(d.mean())
        err = abs(med - ML25M_DENSITY["median"]) / ML25M_DENSITY["median"]
        log(f"   K={K:3d}: {len(keep):5,d} concepts >= {CONC_FLOOR} members | "
            f"per book median {med:5.1f} mean {mean:5.1f} | density error {err:5.1%}")
        if best_err is None or err < best_err:
            best, best_err = K, err
    K = best
    log(f"[calibrate] chosen K = {K} (closest per-book median to ML-25M's "
        f"{ML25M_DENSITY['median']})")

    top = sh.groupby("sid").head(K)
    mem = top.groupby("shelf")["sid"].nunique()
    tags = sorted(mem[mem >= CONC_FLOOR].index)
    t2j = {t: j for j, t in enumerate(tags)}
    top = top[top.shelf.isin(t2j)]
    M = np.zeros((ni, len(tags)), np.uint8)
    M[top.sid.values, top.shelf.map(t2j).values] = 1
    dens = M.sum(1)
    members = M.sum(0)
    log("")
    log(f"[bank] {len(tags):,} concepts (ML-25M: {ML25M_DENSITY['concepts']})")
    log(f"[bank] concepts per book: median {np.median(dens):.0f} mean {dens.mean():.1f} "
        f"(ML-25M: median {ML25M_DENSITY['median']} mean {ML25M_DENSITY['mean']})")
    log(f"[bank] members per concept: median {np.median(members):.0f} mean {members.mean():.0f}")
    log(f"[bank] books with no concept: {(dens == 0).sum()}")
    log(f"[bank] examples: {tags[:20]}")

    np.savez_compressed(os.path.join(ARENA, "concepts.npz"),
                        M=M, tags=np.array(tags, dtype=object), vocab=vocab)
    json.dump({"K": int(K), "floor": CONC_FLOOR, "n_concepts": len(tags), "n_items": ni,
               "concepts_per_book_median": float(np.median(dens)),
               "concepts_per_book_mean": float(dens.mean()),
               "members_per_concept_median": float(np.median(members)),
               "books_with_no_concept": int((dens == 0).sum()),
               "ml25m_reference": ML25M_DENSITY,
               "calibration": "K chosen to match ML-25M's per-item concept density, not its bank "
                              "size; the bank size is whatever falls out",
               "seconds": time.time() - t0},
              open(os.path.join(ARENA, "concepts_meta.json"), "w"), indent=2)
    log(f"\n[done] {ARENA}/concepts.npz ({(time.time()-t0)/60:.1f}m)")


if __name__ == "__main__":
    main()
