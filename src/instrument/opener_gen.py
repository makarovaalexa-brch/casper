r"""CLUSTER-LINKED OPENERS -- one seeded generator, shared by the fitter and the evaluator.

Author's spec, 2026-08-19: every user in a cluster speaks; films and coarse attributes mixed; the
statement varies in how diagnostic it is; and nothing is admitted that is not linkable to the cluster.

WHAT IS MATCHED, AND AGAINST WHAT
----------------------------------
The targets are MEASURED on the 3,078 foldable Reddit requests and the 4,431-thread slot census
(`m6e_reddit_targets.json`, `M1B_OPENER_CENSUS.md`), never guessed:

  film present            0.630   (census `anchors_pos` marginal)
  genre present           0.398   (census `genres_pos` marginal)
  tone present            0.214   (census `tone` marginal)
  films per request       measured distribution, 1/2/3+
  coherence               mean pairwise cosine of co-named films = +0.2243, against a null of
                          +0.1718 for two independent likes of one ML user
  popularity              named films sit at median percentile 93.3

★ DIAGNOSTIC MATERIAL IS SELECTED BY CENTROID LIFT, not by frequency lift. Both were built and
compared: they agree in rank (Spearman +0.56..+0.93) but overlap only 24-65% in the top 100, which is
exactly where openers are drawn from -- and the centroid head is far better. Frequency lift gave
`marvel` -> *Justice League, Dracula Untold*; centroid lift gives *Iron Man 3, Captain America: The
Winter Soldier, Avengers: Age of Ultron, Ant-Man*, independently reproducing M1's validated
distinctive-films column. Centroid lift is also the measure M1 adopted after raw top-N collapsed to
popularity.

  item lift(i | cluster)    = W_d[i] . mean(z_full | cluster)  -  W_d[i] . mean(z_full)
  concept lift(c | cluster) = emb[c] . mean(z_full | cluster)  -  emb[c] . mean(z_full)

Everything drawn is the USER'S OWN evidence -- a film they liked, an attribute they endorse -- so the
opener is both diagnostic of the cluster and true of the person. The draw is a pure function of
(cluster, user, seed), so a user wears the identical opener at fit time and at scoring time; anything
else would fit on one set of states and score on another.

DECLARED NON-MATCHES (measured, not hidden)
  * popularity bottoms out near the 96th percentile against a target of 93.3: ML users' liked sets
    are more blockbuster-heavy than what Reddit posters name, and closing it would mean abandoning
    "the opener is something this user actually liked".
  * era openers are largely unrepresentable -- 1980s/1990s/2000s have no genome tag at the >=30
    member threshold, and they are three of the census's top four era values.
  * `wholesome`, `uplifting`, `cozy` have no genome tag; the census flagged them as the affect words
    our vocabulary cannot express.
  * negative constraints ("no horror") and situational requests ("something for a first date") are
    outside the fold entirely.
"""
from __future__ import annotations

import numpy as np

TARGETS = {"film": 0.6303, "genre": 0.3979, "tone": 0.2135,
           "coherence": 0.2243, "pop_median_pct": 93.3}

GENRE_WORDS = ["horror", "thriller", "comedy", "sci-fi", "action", "romance", "drama",
               "documentary", "fantasy", "adventure", "crime", "war", "mystery", "western",
               "animation", "musical"]
TONE_WORDS = ["feel-good", "fun", "sad", "scary", "emotional", "dark", "disturbing", "melancholic",
              "quirky", "gritty", "heartwarming", "funny", "tense", "bleak"]


def coarse_ids(cname):
    """Genome ids for the coarse vocabulary real requests actually use."""
    nm = {v.lower(): k for k, v in cname.items()}
    return ([nm[w] for w in GENRE_WORDS if w in nm], [nm[w] for w in TONE_WORDS if w in nm])


def centroid_lift(E, cohort_rows, z_full, z_rows, top_frac=0.05):
    """Per-item and per-concept centroid lift for one cluster, plus the qualifying film pool."""
    Wd = E["ctx"].Wd.detach().numpy().astype(np.float64)
    emb = np.asarray(E["net"].emb.weight.detach().numpy(), np.float64)
    pos = {int(r): k for k, r in enumerate(z_rows)}
    idx = [pos[int(r)] for r in cohort_rows if int(r) in pos]
    assert idx, "no cluster user has a banked full-profile latent"
    cmean, gmean = z_full[idx].mean(0), z_full.mean(0)
    li = Wd @ cmean - Wd @ gmean
    lc = emb @ cmean - emb @ gmean
    pool = np.argsort(-li)[:max(1, int(top_frac * len(li)))]
    return {"lift_item": li, "lift_conc": lc, "pool": set(int(i) for i in pool)}


def make_openers(E, cohort_rows, tab, cname, like_min_level, seed, beta=1.5,
                 films_per=(0.53, 0.31, 0.16)):
    """Per-user opener as a list of (channel, id), matched to the measured Reddit targets."""
    rng = np.random.default_rng(seed)
    genres, tones = coarse_ids(cname)
    lc, li = tab["lift_conc"], tab["lift_item"]
    gq = sorted([c for c in genres if lc[c] > 0], key=lambda c: -lc[c])[:6]
    tq = sorted([c for c in tones if lc[c] > 0], key=lambda c: -lc[c])[:6]
    lvl = E["lvl"]
    V, F = np.asarray(E["V"]), np.asarray(E["foldable"])
    endorse = (F > 0) & (V > 0)
    cnt = np.asarray(E["ctx"].cnt, np.float64)
    o = np.argsort(-cnt)
    pop = np.empty(len(cnt), np.int64)
    pop[o] = np.arange(len(cnt))
    pop = pop / len(cnt)
    Wd = E["ctx"].Wd.detach().numpy().astype(np.float32)
    Wn = Wd / np.maximum(np.linalg.norm(Wd, axis=1, keepdims=True), 1e-9)
    # the real popularity profile of named films, as a sampling distribution
    tp = np.concatenate([rng.uniform(0, .10, 590), rng.uniform(.10, .25, 235),
                         rng.uniform(.25, .60, 140), rng.uniform(.60, 1., 35)])
    fk = np.asarray(films_per, np.float64)
    fk /= fk.sum()

    openers, meta = [], {"n_film": 0, "n_genre": 0, "n_tone": 0, "n": 0, "unserved": 0,
                         "genres": [cname.get(c, str(c)) for c in gq],
                         "tones": [cname.get(c, str(c)) for c in tq],
                         "pool_size": len(tab["pool"])}
    for r in cohort_rows:
        r = int(r)
        meta["n"] += 1
        op, got_g = [], False
        myg = [c for c in gq if endorse[r, c]]
        myt = [c for c in tq if endorse[r, c]]
        if rng.random() < TARGETS["genre"] and myg:
            op.append((1, int(rng.choice(myg)))); meta["n_genre"] += 1; got_g = True
        if rng.random() < TARGETS["tone"] and myt:
            op.append((1, int(rng.choice(myt)))); meta["n_tone"] += 1
        mine = [i for i, v in lvl[r].items() if v >= like_min_level and int(i) in tab["pool"]]
        want = rng.random() < TARGETS["film"]
        if not op and not want:                 # nobody arrives silent; prefer the coarse channel
            if myt:
                op.append((1, int(rng.choice(myt)))); meta["n_tone"] += 1
            elif myg and not got_g:
                op.append((1, int(rng.choice(myg)))); meta["n_genre"] += 1
            else:
                want = True
        if want and mine:
            pv = np.asarray([pop[i] for i in mine])
            s = int(mine[int(np.argmin(np.abs(pv - rng.choice(tp))))])
            op.append((0, s)); meta["n_film"] += 1
            k = int(rng.choice([1, 2, 3], p=fk))
            if k > 1 and len(mine) > 1:
                rest = [i for i in mine if i != s]
                w = np.exp(beta * (Wn[rest] @ Wn[s]))
                w /= w.sum()
                for x in rng.choice(rest, min(k - 1, len(rest)), replace=False, p=w):
                    op.append((0, int(x)))
        if not op:
            if mine:
                op = [(0, int(rng.choice(mine)))]; meta["n_film"] += 1
            else:
                meta["unserved"] += 1
        openers.append(op)
    return openers, meta
