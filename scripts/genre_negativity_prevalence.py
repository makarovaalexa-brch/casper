"""genre_negativity_prevalence.py -- MODEL-FREE measurement on raw ML-25M.

Question: do we HAVE signal about disliked genres, or is it predominantly positive + disinterest?
Per (user, genre): n_rated_in_genre, mean_rating_in_genre, user's overall mean rating. Classify:
  ENGAGED-LIKED:      n_rated >= K, mean_in_genre >= 4.0 (or >= user_mean)
  ENGAGED-DISLIKED:   n_rated >= K, mean_in_genre <= 2.5 AND mean_in_genre <= user_mean - 0.7
  DISINTERESTED:      n_rated in genre low relative to population base rate (under-represented)
Report the prevalence distribution over ALL (user, genre) pairs, K=3 and K=5.
Also: fraction of users with >=1 disliked genre; mean # disliked genres/user.

Raw ratings + genre membership ONLY (no learned model, no MF, no EASE). ALL users (nu=162541),
ALL ratings (24.8M rows), ALL 18430 items, ALL 20 genres -- no caps, no subsampling.
Quarantine: the 300 LLM-study users (answerability_grid_ml25m.json) are EXCLUDED (never touched).

Sparse / chunked ops throughout for memory; single pass over sorted-by-user rating stream.
"""
import os, sys, json, time, collections
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
os.chdir("C:/dev/phd/casper")

META = "data/movielens/.cache/ml25m/meta.npz"
MOVIES_CSV = "data/movielens/movies.csv"
STUDY = ".cache/instrument2/answerability_grid_ml25m.json"
OUT = ".cache/signed_latent/genre_negativity_prevalence.json"

GENRES = ["Action", "Adventure", "Animation", "Children", "Comedy", "Crime", "Documentary", "Drama",
          "Fantasy", "Film-Noir", "Horror", "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller",
          "War", "Western", "IMAX", "(no genres listed)"]
GIX = {g: i for i, g in enumerate(GENRES)}

K_LIST = [3, 5]
LIKE_THRESH = 4.0
DISLIKE_MEAN_THRESH = 2.5
DISLIKE_MARGIN = 0.7   # below user's own mean


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


def main():
    t0 = time.time()
    log("loading meta.npz (raw ratings, ALL users) ...")
    d = np.load(META)
    uu = d["uu"].astype(np.int64)
    ii = d["ii"].astype(np.int64)
    rr = d["rr"].astype(np.float32)
    ni = int(d["ni"]); nu = int(d["nu"])
    keepI = d["keepI"].astype(np.int64)
    log(f"ni={ni} nu={nu} n_ratings={len(uu)}")

    study = set(int(u) for u in json.load(open(STUDY))["users"].keys())
    log(f"quarantining {len(study)} LLM-study users")

    # genre matrix (dense-id -> 20-hot)
    log("building genre matrix from movies.csv ...")
    mv = {}
    with open(MOVIES_CSV, encoding="utf-8") as f:
        next(f)
        for line in f:
            i0 = line.find(","); i1 = line.rfind(",")
            mid = int(line[:i0])
            genres = line[i1 + 1:].strip().split("|")
            mv[mid] = genres
    Gmat = np.zeros((ni, len(GENRES)), np.bool_)
    for i in range(ni):
        mid = int(keepI[i])
        for g in mv.get(mid, ["(no genres listed)"]):
            if g in GIX:
                Gmat[i, GIX[g]] = True
    genre_item_counts = Gmat.sum(0)  # items per genre (catalog base rate, numerator side)
    genre_catalog_frac = genre_item_counts / ni
    log(f"genre catalog fractions computed; total items={ni}")

    # quarantine mask, then sort by user for a single streaming pass (memory-safe, no cap)
    log("masking quarantined users + sorting by user (single pass, chunked) ...")
    keep_mask = ~np.isin(uu, np.fromiter(study, dtype=np.int64)) if study else np.ones(len(uu), bool)
    uu, ii, rr = uu[keep_mask], ii[keep_mask], rr[keep_mask]
    order = np.argsort(uu, kind="stable")
    uu, ii, rr = uu[order], ii[order], rr[order]
    n_ratings = len(uu)
    log(f"post-quarantine ratings={n_ratings} users<= {nu - len(study & set(range(nu)))} (approx)")

    NG = len(GENRES)
    # population-level base rate of a user's rating count landing in a given genre, computed as
    # each user's genre-rating-count vector; population median per genre used as the "expected"
    # rate for the DISINTERESTED test (population-representative, not catalog-item-count, since a
    # genre with many items but low engagement should still use REALIZED user behavior as the base).

    # ---- PASS 1: per-user aggregates (genre counts + genre sums) via one streaming sweep ----
    counts = {K: collections.Counter() for K in K_LIST}   # not used directly; per-pair classification done below
    total_pairs = {K: 0 for K in K_LIST}
    cls_counts = {K: collections.Counter() for K in K_LIST}
    n_users_with_dislike = {K: 0 for K in K_LIST}
    dislike_genre_count_per_user = {K: [] for K in K_LIST}
    users_with_ge1_rating_in_genre = np.zeros(NG, np.int64)  # for population base-rate of "any engagement"
    n_users_total = 0

    b = 0
    while b < n_ratings:
        e = b
        u0 = uu[b]
        while e < n_ratings and uu[e] == u0:
            e += 1
        items_u = ii[b:e]
        rats_u = rr[b:e]
        n_users_total += 1
        user_mean = float(rats_u.mean())
        n_rated_total = e - b

        # genre membership rows for this user's rated items: (n_rated, NG) bool -- chunked, this
        # user's slice only, never materializing the full ni x NG beyond the shared Gmat.
        Gu = Gmat[items_u]                      # (n_rated_u, NG)
        n_in_genre = Gu.sum(0).astype(np.int64)  # (NG,)
        # sum of ratings per genre (weight rating by membership, then reduce)
        sum_in_genre = (Gu.astype(np.float32) * rats_u[:, None]).sum(0)
        with np.errstate(invalid="ignore", divide="ignore"):
            mean_in_genre = np.where(n_in_genre > 0, sum_in_genre / np.maximum(n_in_genre, 1), np.nan)

        any_genre_present = n_in_genre > 0
        users_with_ge1_rating_in_genre += any_genre_present.astype(np.int64)

        for K in K_LIST:
            eligible = n_in_genre >= K
            liked = eligible & (mean_in_genre >= max(LIKE_THRESH, user_mean))
            disliked = eligible & (mean_in_genre <= DISLIKE_MEAN_THRESH) & (mean_in_genre <= user_mean - DISLIKE_MARGIN)
            disinterested = ~eligible  # rated 0 or < K items in that genre (relative-to-K under-representation)
            n_disliked_this_user = int(disliked.sum())
            if n_disliked_this_user > 0:
                n_users_with_dislike[K] += 1
            dislike_genre_count_per_user[K].append(n_disliked_this_user)
            total_pairs[K] += NG
            cls_counts[K]["ENGAGED-LIKED"] += int(liked.sum())
            cls_counts[K]["ENGAGED-DISLIKED"] += n_disliked_this_user
            # engaged but neither clearly liked nor disliked (e.g. mean between thresholds)
            engaged_other = eligible & ~liked & ~disliked
            cls_counts[K]["ENGAGED-OTHER"] += int(engaged_other.sum())
            cls_counts[K]["DISINTERESTED"] += int(disinterested.sum())

        b = e
        if n_users_total % 20000 == 0:
            log(f"  ... {n_users_total} users processed ({b}/{n_ratings} ratings)")

    log(f"done streaming: n_users_total={n_users_total}")

    out = dict(
        meta=dict(
            ni=ni, nu_meta=nu, n_ratings_raw=int(len(d["uu"])),
            n_ratings_post_quarantine=int(n_ratings),
            n_users_processed=n_users_total,
            n_study_quarantined=len(study),
            K_list=K_LIST, like_thresh=LIKE_THRESH, dislike_mean_thresh=DISLIKE_MEAN_THRESH,
            dislike_margin_below_user_mean=DISLIKE_MARGIN,
            method="model-free: raw ratings + genre membership only; no learned model",
        ),
        genre_catalog_fraction={GENRES[g]: float(genre_catalog_frac[g]) for g in range(NG)},
    )

    for K in K_LIST:
        tp = total_pairs[K]
        dist = {k: dict(count=v, frac=v / tp) for k, v in cls_counts[K].items()}
        dg = np.array(dislike_genre_count_per_user[K])
        out[f"K{K}"] = dict(
            total_user_genre_pairs=tp,
            distribution=dist,
            frac_users_with_ge1_disliked_genre=n_users_with_dislike[K] / n_users_total,
            mean_disliked_genres_per_user=float(dg.mean()),
            median_disliked_genres_per_user=float(np.median(dg)),
            max_disliked_genres_per_user=int(dg.max()),
        )
        log(f"K={K}: ENGAGED-DISLIKED frac={dist['ENGAGED-DISLIKED']['frac']:.4f}  "
            f"ENGAGED-LIKED frac={dist['ENGAGED-LIKED']['frac']:.4f}  "
            f"DISINTERESTED frac={dist['DISINTERESTED']['frac']:.4f}  "
            f"users-with->=1-dislike={out[f'K{K}']['frac_users_with_ge1_disliked_genre']:.4f}  "
            f"mean-disliked/user={out[f'K{K}']['mean_disliked_genres_per_user']:.3f}")

    # verdict
    k3 = out["K3"]["distribution"]
    verdict_ratio = k3["ENGAGED-DISLIKED"]["frac"] / max(k3["ENGAGED-LIKED"]["frac"], 1e-9)
    out["verdict"] = dict(
        engaged_disliked_frac_K3=k3["ENGAGED-DISLIKED"]["frac"],
        engaged_liked_frac_K3=k3["ENGAGED-LIKED"]["frac"],
        disinterested_frac_K3=k3["DISINTERESTED"]["frac"],
        disliked_to_liked_ratio_K3=verdict_ratio,
        frac_users_ge1_dislike_K3=out["K3"]["frac_users_with_ge1_disliked_genre"],
    )

    out["minutes"] = round((time.time() - t0) / 60, 1)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(out, open(OUT, "w"), indent=2, default=float)
    log(f"wrote {OUT}  [{out['minutes']}m]")
    return out


if __name__ == "__main__":
    main()
