r"""M6k -- fit a CLUSTER teacher: one tree grown on varied, cluster-linked opener states.

Author, 2026-08-19: *"try building the tree teacher on just one cluster for now, and then compare its
performance with old tree teacher and cold tree and static."*

WHY THIS EXISTS. The banked teachers were each fitted at ONE state -- their anchor concept folded --
and M6a/M6c showed their advantage over the cold tree lives only there: at a realistic opener it
vanishes, and at a strongly diagnostic one it goes negative. That is a property of WHERE they were
fitted, not a verdict on cluster teachers. This fits the tree ON the varied states instead, so it is
a cluster teacher by construction rather than by hope.

Each user wears their own cluster-linked opener (`opener_gen`, matched to measured Reddit targets),
folded free and masked from being asked. The split search therefore runs over PER-USER prefixes,
which `group_greedy_bands.celf_per_user_prefix` already supports -- it is structurally identical to
`greedy_certified` and gated against it by that module's selftest.

OUTPUT SCHEMA is the tree schema `tree_route` already consumes: every leaf satisfies
`len(path) + len(steps) == total_L`, with `path` holding ancestor splitters only (the opener is
per-user and enters as folded evidence at routing time, never as a node field).

DEPTH 1 BY DEFAULT: root splitter plus a fitted order per branch. That is the structure `marvel` has,
and it is enough to answer the question this run exists for. Deeper is a flag, not a default, because
the cost is measured (0.722 s/eval on 587 users) and unmeasured depth is how a one-hour job becomes
an overnight one.

VALIDATION ONLY. The 10,000 canonical TEST users are never built, loaded or named.
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
OUT = os.path.join(B, "m6")
SEED = 98765
TOTAL_L = 15
VAL_M0_FULL = 0.15686538118354684
ANCHORS = {"arthouse": 68, "dialogue": 275, "marvel": 581,
           "gore": 417, "teenagers": 922, "inspiring": 503}
BRANCHES = ("like", "dislike", "noanswer")


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def main():
    ap = argparse.ArgumentParser(description="M6k -- fit one cluster teacher on varied openers")
    ap.add_argument("--cluster", default="teenagers")
    ap.add_argument("--depth", type=int, default=1)
    ap.add_argument("--min_leaf", type=int, default=200)
    ap.add_argument("--top_frac", type=float, default=0.05)
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--i_have_author_goahead", action="store_true")
    a = ap.parse_args()
    assert a.i_have_author_goahead, "REFUSING TO RUN without --i_have_author_goahead"
    torch.set_num_threads(a.threads)
    np.random.seed(SEED)
    os.makedirs(OUT, exist_ok=True)

    from strategy_channel_suite import (build_greedy_env, genv_mean, CERT_TOWER, CERT_CONC)
    from ndcg_tree import build_pool, branch_of, leak_assert, LIKE_MIN_LEVEL
    from group_greedy_bands import celf_per_user_prefix, mean_score
    from m3_show_paths import build_names
    from opener_gen import centroid_lift, make_openers

    log("#" * 100)
    log(f"# M6k -- CLUSTER TEACHER for '{a.cluster}', fitted on varied cluster-linked openers")
    log("#" * 100)
    cname, iname = build_names()
    E = build_greedy_env(CERT_TOWER, "validation", CERT_CONC, f"m6k_{a.cluster}")
    leak_assert(E, E["rows"], f"m6k/{a.cluster}")
    rows_all = np.asarray(E["rows"], np.int64)
    f0, _ = genv_mean(E, [], rows_all)
    assert round(f0, 8) == round(VAL_M0_FULL, 8), f"CANONICAL SNAP FAILED: {f0!r}"
    log(f"[gate] canonical m=0 snap full={f0!r} -- exact")

    cid = ANCHORS[a.cluster]
    V, F = np.asarray(E["V"]), np.asarray(E["foldable"])
    endorse = (F > 0) & (V > 0)
    cohort = rows_all[endorse[rows_all, cid]]
    zf = np.load(os.path.join(B, "m1", "z_full.npz"))
    tab = centroid_lift(E, cohort, zf["z"].astype(np.float64), zf["rows"], top_frac=a.top_frac)
    openers, meta = make_openers(E, cohort, tab, cname, LIKE_MIN_LEVEL, seed=SEED)
    n = len(cohort)
    log(f"[cohort] {n} users · film {meta['n_film'] / n:.3f} genre {meta['n_genre'] / n:.3f} "
        f"tone {meta['n_tone'] / n:.3f} · unserved {meta['unserved']} · pool {meta['pool_size']}")
    log(f"[vocab] genres {meta['genres']} · tones {meta['tones']}")
    for j in range(4):
        log("   e.g. " + " + ".join((iname.get(i, str(i)) if c == 0 else cname.get(i, str(i)))
                                    for c, i in openers[j]))

    pool, _i, _c = build_pool(E, 500, 500)
    seeds = {int(r): [tuple(q) for q in op] for r, op in zip(cohort, openers)}
    sig = {"cluster": a.cluster, "seed": SEED, "top_frac": a.top_frac, "n": int(n),
           "pool": len(pool), "total_L": TOTAL_L}
    nodes = {}

    def fit(nid, rows, path, budget):
        """Fit one node: a splitter if it may split, then the fitted order for each branch."""
        sd = [seeds[int(r)] + [tuple(q) for q in path] for r in rows]
        hp = os.path.join(OUT, f"heap_{a.cluster}_{nid}.json")
        depth = len(path)
        may_split = depth < a.depth
        if may_split:
            t0 = time.time()
            seq1, g1, nev = celf_per_user_prefix(E, rows, sd, pool, 1, "full",
                                                 f"{nid}/split", heap_path=hp, sig=sig)
            q = (int(seq1[0][0]), int(seq1[0][1]))
            lab, ar, _z = branch_of(E, rows, q)
            kids = {k: [int(r) for r, l in zip(rows, lab) if str(l) == k] for k in BRANCHES}
            sizes = {k: len(v) for k, v in kids.items()}
            big = [k for k in BRANCHES if sizes[k] >= a.min_leaf]
            nonempty = [k for k in BRANCHES if sizes[k] > 0]
            will = len(nonempty) >= 2 and len(big) >= 1
            log(f"[{nid}] splitter {q} = "
                f"{iname.get(q[1], q[1]) if q[0] == 0 else cname.get(q[1], q[1])} "
                f"gain {g1[0]:+.5f} answer-rate {ar:.3f} sizes {sizes} "
                f"({(time.time() - t0) / 60:.1f}m, {nev} evals) -> split={will}")
        else:
            will = False
        if not will:
            sdl = [seeds[int(r)] + [tuple(x) for x in path] for r in rows]
            t0 = time.time()
            seqL, gL, nev = celf_per_user_prefix(E, rows, sdl, pool, budget, "full",
                                                 f"{nid}/tail", heap_path=hp + ".tail", sig=sig)
            nodes[nid] = {"nid": nid, "split": False, "n_users": len(rows),
                          "path": [[int(x), int(y)] for x, y in path],
                          "steps": [[int(x), int(y)] for x, y in seqL],
                          "gains": [float(g) for g in gL]}
            log(f"[{nid}] LEAF: {len(seqL)} picks for {len(rows)} users "
                f"({(time.time() - t0) / 60:.1f}m, {nev} evals)")
            json.dump({"analysis": "m6k_cluster_tree", "split": "validation", "test_touched": False,
                       "cluster": a.cluster, "sig": sig, "nodes": nodes,
                       "openers": {str(int(r)): [[int(x), int(y)] for x, y in op]
                                   for r, op in zip(cohort, openers)}},
                      open(os.path.join(OUT, f"m6k_{a.cluster}.json"), "w"))
            return
        # a splitting node: record the splitter, then a fitted TAIL for inheritance, then recurse
        t0 = time.time()
        seqT, gT, _n = celf_per_user_prefix(E, rows, sd, pool, budget, "full",
                                            f"{nid}/tailfull", heap_path=hp + ".tail", sig=sig)
        nodes[nid] = {"nid": nid, "split": True, "n_users": len(rows),
                      "splitter": [q[0], q[1]], "child_sizes": sizes, "answer_rate": float(ar),
                      "path": [[int(x), int(y)] for x, y in path],
                      "steps": [[q[0], q[1]]],
                      "tail": [[int(x), int(y)] for x, y in seqT],
                      "children_inheriting": [k for k in BRANCHES
                                              if 0 < sizes[k] < a.min_leaf]}
        log(f"[{nid}] tail fitted ({(time.time() - t0) / 60:.1f}m)")
        for k in BRANCHES:
            if not kids[k]:
                continue
            cn = f"{nid}.{k}"
            if sizes[k] >= a.min_leaf:
                fit(cn, np.asarray(kids[k], np.int64), path + [q], budget - 1)
            else:
                nodes[cn] = {"nid": cn, "split": False, "n_users": sizes[k],
                             "inherited_from": nid,
                             "path": [[int(x), int(y)] for x, y in path],
                             "steps": [[int(x), int(y)] for x, y in seqT]}
                log(f"[{cn}] below floor ({sizes[k]}) -- inherits the parent tail")

    t0 = time.time()
    fit("root", cohort, [], TOTAL_L)
    for nid, nd in nodes.items():
        if not nd.get("split"):
            assert len(nd["path"]) + len(nd["steps"]) == TOTAL_L, \
                f"{nid}: path+steps = {len(nd['path']) + len(nd['steps'])}, expected {TOTAL_L}"
    json.dump({"analysis": "m6k_cluster_tree", "split": "validation", "test_touched": False,
               "cluster": a.cluster, "sig": sig, "n_nodes": len(nodes),
               "minutes": (time.time() - t0) / 60, "opener_meta":
                   {k: v for k, v in meta.items() if k != "shape"}, "nodes": nodes,
               "openers": {str(int(r)): [[int(x), int(y)] for x, y in op]
                           for r, op in zip(cohort, openers)}},
              open(os.path.join(OUT, f"m6k_{a.cluster}.json"), "w"), indent=1)
    log(f"[done] {len(nodes)} nodes in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
