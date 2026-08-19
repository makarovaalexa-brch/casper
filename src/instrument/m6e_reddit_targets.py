r"""M6e -- MEASURE the Reddit opener structure, so a generator can be matched to it.

Author, 2026-08-19: *"please think about best ways to approximate reddit data without directly
mapping reddit to ml, it just needs to be similar in spirit on all dimensions that matter."*

So: do not transplant, and do not hand-tune. NAME the dimensions, MEASURE each on the real corpus,
and later report a synthetic-vs-real table on every one of them. This file does the measuring.

DIMENSIONS
  A cardinality        films per request, attributes per request, and their JOINT distribution
  B channel mix        films only / attributes only / both / neither
  C vocabulary level   which attributes people actually use, and how coarse they are
  D internal coherence do the named films point at ONE region? -- the thing the first generator got
                       wrong (it drew films independently, producing "Chinatown + La La Land")
  E popularity profile how well known are named films, against the catalogue
  F diagnosticity      how strongly an opener localises a taste cluster
  G cross-channel      does the attribute repeat what the films already say, or add to it

D and E need the named MOVIE IDS, which `z_requests.npz` does not carry -- but they are a column in
the ruler CSV (`op_named_movieIds`, already linked and in-catalogue), so this is a measurement of the
Reddit side, never a mapping of Reddit onto ML.

Coherence (D) is measured against the honest null: the SAME number of films drawn independently from
one ML user's liked set -- i.e. exactly what the rejected generator did. If real pairs are no more
similar than that null, coherence is not a dimension that matters and the fix is unnecessary.

Nothing is fitted here and nothing is claimed. VALIDATION-side ML data only for the null; the 10,000
canonical TEST users are never built, loaded or named.
"""
from __future__ import annotations

import os
import sys

_H = os.path.dirname(os.path.abspath(__file__))
_R = os.path.abspath(os.path.join(_H, "..", ".."))
sys.path.insert(0, _H)

import ast
import collections
import csv
import gzip
import json
import time

import numpy as np
import torch

RULER = os.path.join(_R, "data", "reddit_he", "ruler", "ruler_v1.csv.gz")
M1 = os.path.join(_R, "experiments", "battery", "m1")
OUT = os.path.join(_R, "experiments", "battery", "m6")
SEED = 98765


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def main():
    import argparse
    ap = argparse.ArgumentParser(description="M6e -- measure Reddit opener structure")
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--i_have_author_goahead", action="store_true")
    a = ap.parse_args()
    assert a.i_have_author_goahead, "REFUSING TO RUN without --i_have_author_goahead"
    torch.set_num_threads(a.threads)
    rng = np.random.default_rng(SEED)
    os.makedirs(OUT, exist_ok=True)

    from strategy_channel_suite import build_greedy_env, CERT_TOWER, CERT_CONC
    from ndcg_tree import LIKE_MIN_LEVEL
    from m3_show_paths import build_names
    cname, iname = build_names()

    # ---------------------------------------------------------- the named films, from the ruler
    named = {}
    with gzip.open(RULER, "rt", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            s = row["op_named_movieIds"].strip()
            named[row["thread_id"]] = [int(x) for x in s.split()] if s else []
    log(f"[ruler] {len(named):,} threads; {sum(1 for v in named.values() if v):,} name >=1 film")

    z = np.load(os.path.join(M1, "z_requests.npz"), allow_pickle=True)
    tid = [str(t) for t in z["thread_ids"]]
    conc = [ast.literal_eval(str(c)) for c in z["concept_ids"]]

    E = build_greedy_env(CERT_TOWER, "validation", CERT_CONC, "m6e")
    ctx = E["ctx"]
    # movieId -> item row, the canonical Liang show2id (same construction as m1_states)
    sid2row = {}
    with open(os.path.join(_R, "data", "ml-25m", "proc", "unique_sid.txt")) as fh:
        for i, line in enumerate(fh):
            sid2row[int(line.strip())] = i
    cnt = np.asarray(ctx.cnt, np.float64)
    order = np.argsort(-cnt)
    pop_rank = np.empty(len(cnt), np.int64)
    pop_rank[order] = np.arange(len(cnt))
    Wd = ctx.Wd.detach().numpy().astype(np.float64)
    Wn = Wd / np.maximum(np.linalg.norm(Wd, axis=1, keepdims=True), 1e-9)

    rows_named = []
    for t, c in zip(tid, conc):
        ids = [sid2row[m] for m in named.get(t, []) if m in sid2row]
        rows_named.append((ids, c))

    nf = np.array([len(i) for i, _ in rows_named])
    nc = np.array([len(c) for _, c in rows_named])
    log("=" * 96)
    log("[A cardinality] films per request: " +
        " ".join(f"{k}:{int((nf == k).sum())}" for k in range(6)) +
        f" 6+:{int((nf >= 6).sum())}")
    log("[A cardinality] attributes per request: " +
        " ".join(f"{k}:{int((nc == k).sum())}" for k in range(5)))
    joint = collections.Counter((min(int(f), 3), min(int(c), 2)) for f, c in zip(nf, nc))
    log("[A joint] (films capped 3, attrs capped 2) -> " +
        ", ".join(f"{k}:{v}" for k, v in sorted(joint.items())))

    both = int(((nf > 0) & (nc > 0)).sum()); fo = int(((nf > 0) & (nc == 0)).sum())
    co = int(((nf == 0) & (nc > 0)).sum()); nn = int(((nf == 0) & (nc == 0)).sum())
    tot = len(nf)
    log(f"[B channel mix] films only {fo / tot:.3f} · attrs only {co / tot:.3f} · "
        f"both {both / tot:.3f} · neither {nn / tot:.3f}   (n={tot})")

    cc = collections.Counter()
    for _, c in rows_named:
        for x in c:
            cc[cname.get(int(x), str(x))] += 1
    top = cc.most_common(20)
    head = sum(n for _, n in top)
    log(f"[C vocabulary] {len(cc)} distinct attributes used; top 20 carry {head / max(sum(cc.values()), 1):.1%}")
    log("[C vocabulary] " + ", ".join(f"{w}({n})" for w, n in top))

    # ---- D coherence: real multi-film requests vs an ML user's independent draw ---------------
    def mean_pair_cos(ids):
        if len(ids) < 2:
            return None
        v = Wn[ids]
        g = v @ v.T
        iu = np.triu_indices(len(ids), 1)
        return float(g[iu].mean())

    real = [mean_pair_cos(i) for i, _ in rows_named if len(i) >= 2]
    real = [x for x in real if x is not None]
    lvl = E["lvl"]
    rows_all = np.asarray(E["rows"], np.int64)
    null = []
    for r in rng.choice(rows_all, 1500, replace=False):
        liked = [int(i) for i, v in lvl[int(r)].items() if v >= LIKE_MIN_LEVEL]
        if len(liked) < 2:
            continue
        pick = rng.choice(liked, 2, replace=False)
        null.append(mean_pair_cos([int(pick[0]), int(pick[1])]))
    log(f"[D coherence] REAL co-named film pairs: mean cos {np.mean(real):+.4f} "
        f"(median {np.median(real):+.4f}, n={len(real)})")
    log(f"[D coherence] NULL two independent likes of one ML user: mean cos {np.mean(null):+.4f} "
        f"(median {np.median(null):+.4f}, n={len(null)})")
    log(f"[D coherence] ==> real openers are {np.mean(real) - np.mean(null):+.4f} more coherent "
        f"than the rejected generator's draw")

    # ---- E popularity of named films ----------------------------------------------------------
    flat = [i for ids, _ in rows_named for i in ids]
    pr = pop_rank[flat] / len(cnt)
    log(f"[E popularity] named films: median popularity percentile "
        f"{100 * (1 - np.median(pr)):.1f} (100 = most popular); "
        f"share in the top decile {np.mean(pr < 0.10):.3f}, top quartile {np.mean(pr < 0.25):.3f}")
    liked_pr = []
    for r in rng.choice(rows_all, 1500, replace=False):
        liked = [int(i) for i, v in lvl[int(r)].items() if v >= LIKE_MIN_LEVEL]
        if liked:
            liked_pr.append(pop_rank[int(rng.choice(liked))] / len(cnt))
    log(f"[E popularity] NULL a uniform draw from an ML user's likes: median percentile "
        f"{100 * (1 - np.median(liked_pr)):.1f}; top decile {np.mean(np.asarray(liked_pr) < 0.10):.3f}")

    json.dump({"analysis": "m6e_reddit_targets", "test_touched": False,
               "n_requests": tot,
               "films_per_request": {str(k): int((nf == k).sum()) for k in range(8)},
               "attrs_per_request": {str(k): int((nc == k).sum()) for k in range(6)},
               "channel_mix": {"film_only": fo / tot, "conc_only": co / tot,
                               "both": both / tot, "neither": nn / tot},
               "top_attributes": top,
               "coherence": {"real_mean_cos": float(np.mean(real)),
                             "null_mean_cos": float(np.mean(null))},
               "popularity": {"named_median_pct": float(100 * (1 - np.median(pr))),
                              "named_top_decile": float(np.mean(pr < 0.10)),
                              "null_median_pct": float(100 * (1 - np.median(liked_pr)))}},
              open(os.path.join(OUT, "m6e_reddit_targets.json"), "w"), indent=1)
    log("[done] targets written")


if __name__ == "__main__":
    main()
