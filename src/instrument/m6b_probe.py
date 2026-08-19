r"""M6b TIMING PROBE -- what does a CLUSTER teacher cost, when every user wears a different opener?

Author's proposal, 2026-08-19: *"can we build teachers on clusters rather than fixed openers? so
fold-in different things for those users, group them by cluster and grow a tree?"*

That is the correct remedy for M6a, which found the existing teachers are ANCHOR teachers -- their
advantage over the cold tree exists only at the one state they were fitted at and vanishes at
realistic film/bundle openers. Fitting the tree ON those varied states makes it a cluster teacher by
construction instead of by hope.

The machinery already exists and needs no new search:
  `group_greedy_bands.mean_score`          cohort-mean NDCG where each user carries their own prefix
  `group_greedy_bands.celf_per_user_prefix` lazy greedy over that, structurally identical to
                                            `greedy_certified` and gated against it by `--mode selftest`

WHAT IS UNKNOWN IS THE COST, and it must be BENCHED, not projected (project rule: cost estimates on
this box are not properties of a method; over-projected 4x and under-projected 2.6x before). The
shared-prefix search is fast partly because every concept candidate reuses one item-state fold for
the whole block. Per-user prefixes destroy that reuse: each user has a different item list. This
probe measures ONE root search on ONE cohort and reports seconds/eval, so the six-tree job can be
projected from a measured number.

Nothing is fitted, kept or claimed here. VALIDATION ONLY.
"""
from __future__ import annotations

import os
import sys

_H = os.path.dirname(os.path.abspath(__file__))
_R = os.path.abspath(os.path.join(_H, "..", ".."))
sys.path.insert(0, _H)

import argparse
import json
import time

import numpy as np
import torch

B = os.path.join(_R, "experiments", "battery")
SEED = 98765
ANCHORS = {"arthouse": 68, "dialogue": 275, "marvel": 581,
           "gore": 417, "teenagers": 922, "inspiring": 503}


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def main():
    ap = argparse.ArgumentParser(description="M6b timing probe")
    ap.add_argument("--anchor", default="arthouse")
    ap.add_argument("--n_cand", type=int, default=60, help="candidates to time (a sample of the pool)")
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--i_have_author_goahead", action="store_true")
    a = ap.parse_args()
    assert a.i_have_author_goahead, "REFUSING TO RUN without --i_have_author_goahead"
    torch.set_num_threads(a.threads)
    rng = np.random.default_rng(SEED)

    from strategy_channel_suite import build_greedy_env, genv_mean, CERT_TOWER, CERT_CONC
    from ndcg_tree import build_pool, LIKE_MIN_LEVEL
    from group_greedy_bands import mean_score

    E = build_greedy_env(CERT_TOWER, "validation", CERT_CONC, "m6bprobe")
    rows_all = np.asarray(E["rows"], np.int64)
    pool, _i, _c = build_pool(E, 500, 500)
    V, F = np.asarray(E["V"]), np.asarray(E["foldable"])
    endorse = (F > 0) & (V > 0)
    cid = ANCHORS[a.anchor]
    cohort = rows_all[endorse[rows_all, cid]]
    lvl = E["lvl"]
    anchor_ids = set(ANCHORS.values())

    # a Reddit-shaped opener per user, from their OWN fold-in half
    seeds, keep = [], []
    for r in cohort:
        r = int(r)
        liked = [int(i) for i, v in lvl[r].items() if v >= LIKE_MIN_LEVEL]
        cs = [int(c) for c in np.nonzero(endorse[r])[0] if int(c) not in anchor_ids]
        if len(liked) < 2 or not cs:
            continue
        it = rng.choice(liked, 2, replace=False)
        seeds.append([(0, int(it[0])), (0, int(it[1])), (1, int(rng.choice(cs)))])
        keep.append(r)
    keep = np.asarray(keep, np.int64)
    log(f"[probe] {a.anchor}: {len(keep)} users, each wearing a DIFFERENT 2-film + 1-concept opener")

    # baseline: the cohort mean with only the openers folded
    t0 = time.time()
    base = mean_score(E, keep, seeds, [])
    t_base = time.time() - t0
    log(f"[probe] base score with openers only = {base:.6f}  ({t_base:.2f}s for one pass)")

    cand = [tuple(pool[i]) for i in rng.choice(len(pool), a.n_cand, replace=False)]
    t0 = time.time()
    for q in cand:
        mean_score(E, keep, seeds, [q])
    dt = time.time() - t0
    per = dt / len(cand)
    log(f"[probe] {len(cand)} candidate evals in {dt:.1f}s -> {per:.3f} s/eval "
        f"on {len(keep)} users")

    # projection, stated as a measured extrapolation and nothing more
    n_pool = len(pool)
    seed_pass = per * n_pool
    log("=" * 90)
    log(f"[project] one full seed pass over {n_pool} candidates: {seed_pass / 60:.1f} min")
    log(f"[project] a 15-pick greedy tail with CELF laziness (~2-4x a seed pass, measured elsewhere): "
        f"{seed_pass * 3 / 60:.0f} min per node-order")
    log(f"[project] a depth-2 tree is ~4 such orders -> ~{seed_pass * 3 * 4 / 60 / 60:.1f} h per "
        f"cluster; six clusters -> ~{seed_pass * 3 * 4 * 6 / 60 / 60:.1f} h")
    log("[project] ^ EXTRAPOLATION from one measured cohort. Cohort size varies 587-1822, so the "
        "true figure scales roughly linearly with n.")
    log("[done] nothing fitted, nothing kept")


if __name__ == "__main__":
    main()
