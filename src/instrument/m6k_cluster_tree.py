r"""M6k -- one tree grown on varied, group-linked opener states.

⚠ NAMING, corrected 2026-08-19 after the author caught the conflation. This file fits on an
ENDORSEMENT COHORT -- `rows[endorse[rows, cid]]`, the users who behaviourally endorse one concept --
NOT on a k-means taste cluster. Those are different objects: a cluster is a partition of taste space,
an endorsement cohort is an overlapping semantic set (the six overlap by 27.8%). Every tree in this
project so far -- the six M3 teachers, the cold tree, and this one -- uses endorsement cohorts;
`run_m3_teachers.sh` passes `--cohort_concept`, and the artifacts record `n_cohort` = 786 / 587,
exactly the endorsement counts. No tree has ever been built on a cluster label.

The k-means labels were used ONCE, in M1, to choose WHICH six concepts to use as anchor words. They
were never banked, which is why they could not be re-checked; `m6p_recluster.py` re-derives and banks
them. Fitting on a true cluster is a one-line cohort change from here and has not yet been done.

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
import hashlib
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
    ap.add_argument("--split", default="validation", choices=["validation", "train"],
                    help="fit on TRAIN to get a larger cohort; the held-out evaluation then uses "
                         "OTHER train users, and cold/static stay validation-fitted so every arm is "
                         "out of sample on the evaluation half")
    ap.add_argument("--slice_hi", type=int, default=34000,
                    help="train users 0..slice_hi are the fitting pool (--split train only)")
    ap.add_argument("--n_users", type=int, default=0,
                    help="cap the fitting cohort at this many users (0 = all in the pool)")
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--disjoint", default="",
                    help="m7b_disjoint.npz. Fit on the DISJOINT group: users assigned to their "
                         "single strongest endorsement, openers drawn only from material this group "
                         "owns outright. Probe separability rises 0.371 -> 0.959 on marvel.")
    ap.add_argument("--global_openers", action="store_true",
                    help="M7 k=1: fit ONE tree over ALL users at GROUP-FREE opener states. No "
                         "cohort, no cluster, no group pool. This is the control that separates "
                         "'fitting at opener states' from 'fitting per group' -- the two effects "
                         "the +0.00760 cohort result currently conflates.")
    ap.add_argument("--cluster_labels", default="",
                    help="npz from m6y_cluster_train: fit on a TASTE CLUSTER (a latent-space "
                         "partition) instead of an endorsement cohort. With this, --cluster is the "
                         "integer cluster id.")
    ap.add_argument("--i_have_author_goahead", action="store_true")
    a = ap.parse_args()
    assert a.i_have_author_goahead, "REFUSING TO RUN without --i_have_author_goahead"
    from run_guard import require_mains
    require_mains(f"{a.cluster} group-tree fit", log=log)
    torch.set_num_threads(a.threads)
    np.random.seed(SEED)
    os.makedirs(OUT, exist_ok=True)

    from strategy_channel_suite import (build_greedy_env, genv_mean, CERT_TOWER, CERT_CONC)
    from ndcg_tree import build_pool, branch_of, leak_assert, LIKE_MIN_LEVEL
    from group_greedy_bands import celf_per_user_prefix, mean_score
    from m3_show_paths import build_names
    from opener_gen import centroid_lift, make_openers, make_openers_free

    log("#" * 100)
    log(f"# M6k -- CLUSTER TEACHER for '{a.cluster}', fitted on varied cluster-linked openers")
    log("#" * 100)
    cname, iname = build_names()
    # ⚠ 2026-08-20: a --split train run overwrote the validation-fitted arthouse tree AND its heap
    # sidecars, because both used the bare cluster name. Every artifact is now tagged by the fitting
    # condition, so the two can never collide.
    # A cluster tree and an endorsement-cohort tree are DIFFERENT OBJECTS and must never share a
    # path -- `--cluster 16` means concept-cohort "16" in one mode and taste-cluster 16 in the other.
    _pre = ("GLOBALopen" if a.global_openers else
            f"dj{a.cluster}" if a.disjoint else
            f"clu{a.cluster}" if a.cluster_labels else a.cluster)
    tag = _pre if a.split == "validation" else f"{_pre}_{a.split}{a.n_users or a.slice_hi}"
    if a.split == "train":
        E = build_greedy_env(CERT_TOWER, "train", CERT_CONC, f"m6k_{a.cluster}_tr",
                             cohort_slice=(0, a.slice_hi))
        rows_all = np.arange(E["ctx"].n, dtype=np.int64)
        log(f"[split] fitting on TRAIN users 0..{a.slice_hi} -- the criterion is measured on users "
            f"the tower memorised, so LEVELS here are inflated; the fitted STRUCTURE is what is "
            f"carried, and it is scored on held-out users elsewhere")
    else:
        E = build_greedy_env(CERT_TOWER, "validation", CERT_CONC, f"m6k_{a.cluster}")
        leak_assert(E, E["rows"], f"m6k/{a.cluster}")
        rows_all = np.asarray(E["rows"], np.int64)
    f0, _ = genv_mean(E, [], rows_all)
    if a.split == "validation":
        assert round(f0, 8) == round(VAL_M0_FULL, 8), f"CANONICAL SNAP FAILED: {f0!r}"
        log(f"[gate] canonical m=0 snap full={f0!r} -- exact")
    else:
        log(f"[gate] train-pool m=0 snap full={f0!r} (no canonical value exists for a train slice)")

    if a.disjoint:
        # ---- M7b: disjoint group, disjoint opener material ---------------------------------------
        _D = np.load(a.disjoint)
        NAMES = ["arthouse", "dialogue", "marvel", "gore", "teenagers", "inspiring"]
        gi = NAMES.index(a.cluster)
        cid = ANCHORS[a.cluster]
        Vm, Fm = np.asarray(E["V"]), np.asarray(E["foldable"])
        endorse = (Fm > 0) & (Vm > 0)
        _cids = np.array([ANCHORS[g] for g in NAMES])
        _Vs = Vm[:, _cids].astype(np.float64).copy()
        _Es = endorse[:, _cids]
        _Vs[~_Es] = -np.inf
        _best = _Vs.argmax(1)
        cohort = rows_all[(_best[rows_all] == gi) & _Es[rows_all].any(1)]
        cent_rows_cluster = None
        DJ = (_D["own_item"], _D["own_concept"], gi)
        log(f"[disjoint] {a.cluster}: {len(cohort)} train users assigned here EXCLUSIVELY "
            f"(of {int(endorse[rows_all, cid].sum())} who endorse it at all)")
    elif a.global_openers:
        # ---- M7 k=1: everyone, group-free -------------------------------------------------------
        cohort = rows_all
        cent_rows_cluster = None; DJ = None
        log(f"[global] ONE tree over all {len(cohort)} users at group-free opener states")
    elif a.cluster_labels:
        # ---- TASTE-CLUSTER MODE ------------------------------------------------------------------
        # A cluster is a latent-space partition (era axis projected out, k-means on train), not an
        # overlapping semantic set. Membership therefore comes from banked labels, not from an
        # endorsement test, and every user belongs to exactly one.
        cl = np.load(a.cluster_labels)
        cnum = int(a.cluster)
        tr_rows, tr_lab = cl["train_rows"], cl["train_labels"]
        va_rows, va_lab = cl["val_rows"], cl["val_labels"]
        assert a.split == "train", "cluster teachers are fitted on train"
        want = set(tr_rows[tr_lab == cnum].tolist())
        # m6k builds its train env as cohort_slice=(0, slice_hi), so a row's LOCAL index is already
        # its global train id -- the same convention m6y banked its labels under. Asserted, not
        # assumed, because an index is only meaningful with its population.
        cohort = np.asarray([r for r in rows_all if int(r) in want], np.int64); DJ = None
        in_slice = int(((tr_rows >= 0) & (tr_rows < a.slice_hi) & (tr_lab == cnum)).sum())
        matched = len(cohort) / max(in_slice, 1)
        assert matched > 0.8, (
            f"cluster {cnum}: matched only {len(cohort)} of {in_slice} banked members in slice "
            f"0..{a.slice_hi} ({matched:.1%}). A PARTIAL index match is indistinguishable from a "
            f"small cohort -- check the label file's population against this env's.")
        log(f"[cluster] index match {len(cohort)}/{in_slice} = {matched:.1%}")
        cent_rows_cluster = va_rows[va_lab == cnum]
        log(f"[cluster] c{cnum}: {len(cohort)} train users in this slice, "
            f"{len(cent_rows_cluster)} validation users")
        assert len(cohort) >= 200, f"cluster {cnum} has only {len(cohort)} users in this slice"
    else:
        cid = ANCHORS[a.cluster]
        V, F = np.asarray(E["V"]), np.asarray(E["foldable"])
        endorse = (F > 0) & (V > 0)
        cohort = rows_all[endorse[rows_all, cid]]
        cent_rows_cluster = None; DJ = None
    if a.n_users and len(cohort) > a.n_users:
        cohort = cohort[:a.n_users]
        log(f"[cohort] capped at {a.n_users} users")
    zf = np.load(os.path.join(B, "m1", "z_full.npz"))
    # ⚠ THE GROUP CENTROID IS ALWAYS COMPUTED ON VALIDATION ROWS -- it is a property of the group,
    # not of the fitting split, and `z_full` holds validation users. Passing the train cohort here
    # silently averaged the ~600 train indices that happened to collide with a validation id.
    if a.global_openers:
        # No group => no group centroid. make_openers_free needs no diagnostic pool, which is the
        # whole point: the opener is the user's own history, so nothing has to be computed from a
        # cohort first. (`cid` does not exist on this path.)
        tab = None
        log("[centroid] SKIPPED -- group-free openers need no diagnostic pool")
    elif cent_rows_cluster is not None:
        cent_rows = np.asarray(cent_rows_cluster, np.int64)
        log(f"[centroid] computed on {len(cent_rows)} VALIDATION users of cluster {a.cluster} "
            f"(assigned to the train-fitted centroid, never re-clustered on validation)")
    elif a.split == "train":
        Ev = build_greedy_env(CERT_TOWER, "validation", CERT_CONC, "m6k_cent")
        vr = np.asarray(Ev["rows"], np.int64)
        Vv, Fv = np.asarray(Ev["V"]), np.asarray(Ev["foldable"])
        cent_rows = vr[((Fv > 0) & (Vv > 0))[vr, cid]]
        del Ev
        log(f"[centroid] computed on {len(cent_rows)} VALIDATION {a.cluster} users "
            f"(the group centroid is split-independent)")
    else:
        cent_rows = cohort
    if not a.global_openers:
        tab = centroid_lift(E, cent_rows, zf["z"].astype(np.float64), zf["rows"],
                            top_frac=a.top_frac)
    if a.global_openers:
        openers, meta = make_openers_free(E, cohort, cname, LIKE_MIN_LEVEL, seed=SEED)
    else:
        gkey = f"cluster{a.cluster}" if a.cluster_labels else a.cluster
        openers, meta = make_openers(E, cohort, tab, cname, LIKE_MIN_LEVEL, seed=SEED,
                                     group_key=gkey, disjoint=DJ)
    n = len(cohort)
    log(f"[cohort] {n} users · film {meta['n_film'] / n:.3f} genre {meta['n_genre'] / n:.3f} "
        f"tone {meta['n_tone'] / n:.3f} · unserved {meta['unserved']}"
        + (" · GROUP-FREE (own history, no pool)" if a.global_openers
           else f" · pool {meta['pool_size']}"))
    if not a.global_openers:
        log(f"[vocab] genres {meta['genres']} · tones {meta['tones']}")
    else:
        log("[vocab] group-free: each user offers their OWN endorsed genres and tones, "
            "not a group's top-6")
    for j in range(4):
        log("   e.g. " + " + ".join((iname.get(i, str(i)) if c == 0 else cname.get(i, str(i)))
                                    for c, i in openers[j]))

    pool, _i, _c = build_pool(E, 500, 500)
    seeds = {int(r): [tuple(q) for q in op] for r, op in zip(cohort, openers)}
    # ⚠ THE SIGNATURE MUST IDENTIFY THE CANDIDATE SET, NOT ITS SIZE (2026-08-26). It used to carry
    # `"pool": len(pool)` and was computed ONCE for the whole tree, so shrinking the pool per node
    # left the signature unchanged: banked heaps validated, resumed, and faithfully reproduced the
    # very tree the pool fix was meant to correct -- 22 nodes "refitted" in 0.0 minutes. A length is
    # a PROXY for a candidate set. It is now per-node and hashes the actual candidates and path.
    sig_base = {"cluster": a.cluster, "seed": SEED, "top_frac": a.top_frac, "n": int(n),
                "total_L": TOTAL_L, "v": 2}

    def _sha(o):
        return hashlib.sha1(json.dumps(o, sort_keys=True).encode()).hexdigest()[:16]

    def node_sig(path, pool_n):
        return dict(sig_base,
                    path=[[int(x), int(y)] for x, y in path],
                    n_pool=len(pool_n),
                    pool_sha1=_sha([[int(q[0]), int(q[1])] for q in pool_n]))
    nodes = {}

    def fit(nid, rows, path, budget):
        """Fit one node: a splitter if it may split, then the fitted order for each branch."""
        sd = [seeds[int(r)] + [tuple(q) for q in path] for r in rows]
        # EXCLUDE THIS NODE'S OWN PATH FROM THE POOL (2026-08-26). Without this a search can
        # re-pick a question already folded into the state: it scores ~0 gain, which WINS at a deep
        # node where nothing else is positive, and the user is then asked the same thing twice.
        # `ndcg_tree` has always done this (`pool_n`, line 674) and `greedy_certified` asserts it;
        # `celf_per_user_prefix` had neither, so m6k was the one fitter that could produce it.
        # Cost: `inspiring` had 3 such nodes over 307 users.
        _banned = {(int(a), int(b)) for a, b in path}
        pool_n = [q for q in pool if (int(q[0]), int(q[1])) not in _banned]
        assert len(pool_n) == len(pool) - len(_banned & {(int(q[0]), int(q[1])) for q in pool}),             f"{nid}: pool exclusion dropped the wrong number of candidates"
        sig = node_sig(path, pool_n)
        hp = os.path.join(OUT, f"heap_{tag}_{nid}.json")
        sw = os.path.join(OUT, "sweeps", f"sweep_{tag}_{nid}.npz")   # C1 gain table, see ndcg_tree
        depth = len(path)
        may_split = depth < a.depth
        if may_split:
            t0 = time.time()
            seq1, g1, nev = celf_per_user_prefix(E, rows, sd, pool_n, 1, "full",
                                                 f"{nid}/split", heap_path=hp, sig=sig, sweep_path=sw)
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
            # The split search WAS the CELF seed pass over this exact (rows, sd, pool) state, so
            # resume its heap rather than re-seeding: the first pick a fresh greedy would make IS
            # the splitter, by construction. Same reuse the cold-tree fitter documents at
            # `ndcg_tree.py:642`. Halves the cost of every node. If `may_split` was False the file
            # does not exist and this seeds cold, as before.
            seqL, gL, nev = celf_per_user_prefix(E, rows, sdl, pool_n, budget, "full",
                                                 f"{nid}/tail", heap_path=hp, sig=sig, sweep_path=sw)
            nodes[nid] = {"nid": nid, "split": False, "n_users": len(rows),
                          "path": [[int(x), int(y)] for x, y in path],
                          "steps": [[int(x), int(y)] for x, y in seqL],
                          "gains": [float(g) for g in gL]}
            log(f"[{nid}] LEAF: {len(seqL)} picks for {len(rows)} users "
                f"({(time.time() - t0) / 60:.1f}m, {nev} evals)")
            json.dump({"analysis": "m6k_cluster_tree", "split": a.split, "test_touched": False,
               "fit_rows": [int(r) for r in cohort], "slice_hi": a.slice_hi,
                       "cluster": a.cluster, "sig": sig, "nodes": nodes,
                       "openers": {str(int(r)): [[int(x), int(y)] for x, y in op]
                                   for r, op in zip(cohort, openers)}},
                      open(os.path.join(OUT, f"m6k_{tag}.json"), "w"))
            return
        # a splitting node: record the splitter, then a fitted TAIL for inheritance, then recurse
        t0 = time.time()
        seqT, gT, _n = celf_per_user_prefix(E, rows, sd, pool_n, budget, "full",
                                            f"{nid}/tailfull", heap_path=hp, sig=sig, sweep_path=sw)
        # the equivalence the reuse rests on, asserted rather than assumed
        assert tuple(seqT[0]) == q, (
            f"{nid}: resumed tail opened with {tuple(seqT[0])}, splitter was {q} -- heap reuse is "
            f"NOT equivalent here; revert to a separate .tail heap before trusting this tree")
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
    json.dump({"analysis": "m6k_cluster_tree", "split": a.split, "test_touched": False,
               "fit_rows": [int(r) for r in cohort], "slice_hi": a.slice_hi,
               "cluster": a.cluster, "sig": sig_base, "n_nodes": len(nodes),
               "minutes": (time.time() - t0) / 60, "opener_meta":
                   {k: v for k, v in meta.items() if k != "shape"}, "nodes": nodes,
               "openers": {str(int(r)): [[int(x), int(y)] for x, y in op]
                           for r, op in zip(cohort, openers)}},
              open(os.path.join(OUT, f"m6k_{tag}.json"), "w"), indent=1)
    log(f"[done] {len(nodes)} nodes in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
