"""A15 probe -- can goodbooks-10k carry Paper B's experiment at all?

Design sheet: docs/design/DS_A15_GOODBOOKS_PROBE.md

This is the PRE-FLIGHT probe from DS_A14: it needs no tower, no teacher and no gates, only the
split, the tag mining and some counting. It exists to kill a bad candidate in hours rather than
discover it is bad after weeks of training.

THE GO/NO-GO is step 4. On ML-25M, sixteen questions collect **3.11 item answers and 12.36 concept
answers**, and the paper's entire finding is the ratio between those. A dataset that does not land in
a comparable regime is not a cheaper instance of this experiment, it is a different one.

Mirrors the ML-25M definitions exactly where they exist:
  * item answerable  <=> the user rated it (any band)          [interview_strategies.py:115]
  * concept answerable <=> n_c >= 2 AND e_c >= 1.5             [signed_answers.py:130,138]
        n_c = user's rated members of c ; e_c = |profile| * pexp_c ; pexp from TRAIN counts
  * concept bank     <=> >= 30 member items                    [conversational_bandit_selectors.py:415-419]
  * item pool        <=> top-500 by train popularity           [ndcg_tree.build_pool:307]
  * concept pool     <=> top-500 by answerability
  * split            <=> Liang strong generalisation, seed 98765, per-user 80/20 fold-in/target

NO TRUNCATION. The full 10,000-book catalogue is used; goodbooks is small enough that the item
universe never needs restricting, which is the defect that dogged the old Goodreads work.

Run:  python src/instrument/a15_goodbooks_probe.py
"""
import json
import os
import re
import sys
import time

import numpy as np
import pandas as pd

_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
DATA = os.path.join(_ROOT, "data", "goodbooks-10k")
OUT = os.path.join(_ROOT, "experiments", "goodbooks")
SEED = 98765
N_VAL = 5000
N_TEST = 5000
TEST_PROP = 0.2
MIN_UC = 5
LIKE_GT = 3.5          # Liang's rating > 3.5 for the binary training matrix
CONC_FLOOR = 30        # >= 30 member items, mirroring ML-25M
TAU_REF = 1.5          # e_c refusal threshold, signed_answers.py:29
NC_MIN = 2             # n_c >= 2, signed_answers.py:138
POOL_N = 500
BUDGET = 16
SHELF_MINCOUNT_FRAC = 0.02   # a tag counts as membership if it is >=2% of that book's tag mass

# Process / non-content shelves. Extends gr_books.py's list, which was written for the same job.
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
    if not n or n in JUNK_EXACT:
        return True
    if not re.search(r"[a-z]", n):
        return True
    if len(n) <= 2:
        return True
    return any(sub in n for sub in JUNK_SUBSTR)


def main():
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()
    rep = {"label": "a15_goodbooks_probe", "dataset": "goodbooks-10k", "seed": SEED,
           "design_sheet": "docs/design/DS_A15_GOODBOOKS_PROBE.md",
           "ml25m_reference": {"item_answers_at_16": 3.11, "concept_answers_at_16": 12.36,
                               "concepts": 1031, "foldin_items_per_user": 141.2}}

    # ---------- 1. ratings, split ----------
    r = pd.read_csv(os.path.join(DATA, "ratings.csv"))
    r = r.drop_duplicates(subset=["user_id", "book_id"])
    cnt = r.groupby("user_id").size()
    keep = cnt[cnt >= MIN_UC].index
    r = r[r.user_id.isin(keep)]
    users = np.array(sorted(r.user_id.unique()))
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(len(users))
    te_u = set(users[perm[:N_TEST]]); va_u = set(users[perm[N_TEST:N_TEST + N_VAL]])
    tr_u = set(users[perm[N_TEST + N_VAL:]])
    log(f"[split] {len(tr_u)} train / {len(va_u)} val / {len(te_u)} test users; "
        f"{r.book_id.nunique()} books; {len(r)} ratings")
    rep["split"] = {"train_users": len(tr_u), "val_users": len(va_u), "test_users": len(te_u),
                    "items": int(r.book_id.nunique()), "ratings": int(len(r))}

    books = np.array(sorted(r.book_id.unique()))
    b2i = {b: i for i, b in enumerate(books)}
    ni = len(books)

    # per-user 80/20 fold-in / target, on the HELD-OUT cohorts only
    def fold(df):
        out = {}
        for u, g in df.groupby("user_id"):
            it = g.book_id.values.copy()
            st = g.rating.values.copy()
            p = rng.permutation(len(it))
            it, st = it[p], st[p]
            ntg = max(1, int(round(TEST_PROP * len(it))))
            out[u] = (it[ntg:], st[ntg:], it[:ntg])      # fold-in items, fold-in stars, targets
        return out

    r_te = r[r.user_id.isin(te_u)]
    te = fold(r_te)
    fold_sizes = np.array([len(v[0]) for v in te.values()])
    log(f"[depth] fold-in items per test user: mean {fold_sizes.mean():.1f} "
        f"median {np.median(fold_sizes):.0f} p10 {np.percentile(fold_sizes,10):.0f} "
        f"(ML-25M: 141.2)")
    rep["foldin_items_per_user"] = {"mean": float(fold_sizes.mean()),
                                    "median": float(np.median(fold_sizes)),
                                    "p10": float(np.percentile(fold_sizes, 10))}

    # train popularity from the TRAIN cohort's likes only (mirrors ctx.cnt)
    r_tr = r[r.user_id.isin(tr_u)]
    likes_tr = r_tr[r_tr.rating > LIKE_GT]
    pop = np.zeros(ni, np.float64)
    vc = likes_tr.book_id.value_counts()
    for b, c in vc.items():
        pop[b2i[b]] = c
    log(f"[pop] train likes {len(likes_tr)}; top book has {pop.max():.0f} likes")

    # ---------- 2. concept bank from the tag folksonomy ----------
    bk = pd.read_csv(os.path.join(DATA, "books.csv"), usecols=["book_id", "goodreads_book_id"])
    g2b = dict(zip(bk.goodreads_book_id.values, bk.book_id.values))
    bt = pd.read_csv(os.path.join(DATA, "book_tags.csv"))
    tg = pd.read_csv(os.path.join(DATA, "tags.csv"))
    id2name = dict(zip(tg.tag_id.values, tg.tag_name.values))

    bt["book"] = bt.goodreads_book_id.map(g2b)
    bt = bt.dropna(subset=["book"])
    bt = bt[bt.book.isin(b2i.keys())]
    tot = bt.groupby("book")["count"].transform("sum")
    bt = bt[bt["count"] >= SHELF_MINCOUNT_FRAC * tot]          # membership rule
    bt["name"] = bt.tag_id.map(id2name)
    bt = bt[~bt.name.map(is_junk)]
    log(f"[tags] {bt.tag_id.nunique()} distinct content tags after junk filter and the "
        f"{SHELF_MINCOUNT_FRAC:.0%} membership rule")

    memb = bt.groupby("tag_id")["book"].apply(lambda s: np.array([b2i[x] for x in s.unique()]))
    memb = {int(t): v for t, v in memb.items() if len(v) >= CONC_FLOOR}
    tags = sorted(memb)
    nc = len(tags)
    log(f"[bank] {nc} concepts with >= {CONC_FLOOR} members (ML-25M: 1031)")
    rep["concept_bank"] = {"n_concepts": nc, "floor": CONC_FLOOR,
                           "median_members": float(np.median([len(memb[t]) for t in tags])) if nc else 0,
                           "example_names": [str(id2name[t]) for t in tags[:25]]}
    if nc < 100:
        log("!! concept bank is too small to stand in for ML-25M's 1,031 -- this is a NO-GO signal")

    # membership matrix and the exposure prior, mirroring strategy_channel_suite.py:454-455
    M = np.zeros((ni, nc), np.float32)
    for j, t in enumerate(tags):
        M[memb[t], j] = 1.0
    pexp = (pop @ M) / max(pop.sum(), 1e-9)

    # ---------- 3. pools ----------
    item_pool = np.argsort(-pop)[:POOL_N]

    # concept answerability over the TRAIN cohort, to rank the concept pool without touching test
    tr_users = np.array(sorted(tr_u))[:5000]
    sub = r_tr[r_tr.user_id.isin(set(tr_users.tolist()))]
    cans = np.zeros(nc, np.float64)
    n_probe = 0
    for u, g in sub.groupby("user_id"):
        idx = np.array([b2i[b] for b in g.book_id.values])
        n_c = M[idx].sum(0)
        e_c = len(idx) * pexp
        cans += ((n_c >= NC_MIN) & (e_c >= TAU_REF)).astype(np.float64)
        n_probe += 1
    cans /= max(n_probe, 1)
    conc_pool = np.argsort(-cans)[:POOL_N]
    log(f"[pool] item pool {len(item_pool)}; concept pool {len(conc_pool)}; "
        f"best concept answerable by {cans.max():.1%} of train users")
    rep["concept_answerability_train"] = {"max": float(cans.max()), "mean": float(cans.mean()),
                                          "pool_mean": float(cans[conc_pool].mean())}

    # ---------- 4. THE GO/NO-GO: answers collected in 16 questions, test cohort ----------
    ipool_set = item_pool[:BUDGET]                 # a static popularity order asks the top-16
    ia, ca = [], []
    for u, (fi, fs, tg_) in te.items():
        idx = np.array([b2i[b] for b in fi])
        rated = np.zeros(ni, bool); rated[idx] = True
        ia.append(int(rated[ipool_set].sum()))
        n_c = M[idx].sum(0)
        e_c = len(idx) * pexp
        ok = (n_c >= NC_MIN) & (e_c >= TAU_REF)
        ca.append(int(ok[conc_pool[:BUDGET]].sum()))
    ia = np.array(ia, np.float64); ca = np.array(ca, np.float64)
    log("")
    log("=== GO / NO-GO ===")
    log(f"item answers per 16 questions   : {ia.mean():.2f}   (ML-25M 3.11)")
    log(f"concept answers per 16 questions: {ca.mean():.2f}   (ML-25M 12.36)")
    log(f"ratio concept/item              : {ca.mean()/max(ia.mean(),1e-9):.2f}   (ML-25M 3.98)")
    log(f"users answering NOTHING in 2 item questions   : {(ia[:]>=0).sum() and int((np.array([1 for _ in ia])*0).sum())}")
    rep["go_no_go"] = {"item_answers_at_16": float(ia.mean()),
                       "concept_answers_at_16": float(ca.mean()),
                       "ratio": float(ca.mean() / max(ia.mean(), 1e-9)),
                       "item_zero_share": float((ia == 0).mean()),
                       "concept_zero_share": float((ca == 0).mean())}
    log(f"share of users with ZERO item answers   : {(ia==0).mean():.1%}")
    log(f"share of users with ZERO concept answers: {(ca==0).mean():.1%}")

    rep["seconds"] = time.time() - t0
    json.dump(rep, open(os.path.join(OUT, "a15_goodbooks_probe.json"), "w"), indent=2)
    log(f"\n[done] {os.path.join(OUT, 'a15_goodbooks_probe.json')} ({rep['seconds']:.1f}s)")


if __name__ == "__main__":
    main()
