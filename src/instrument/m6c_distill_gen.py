r"""M6c STAGE 1 -- trajectories for the OPENER-FITTED forest, on TRAIN users.

Pre-registered in `docs/design/DS_M6_REDDIT_SHAPED_OPENERS.md` (M6c/M6d/M6e) before any of this data
existed. Author's goal, verbatim: *"continue with all"* -- fit every group, distil state-only, check
on a group mix, and check whether it generalises.

WHAT IS DIFFERENT FROM M4b. There, every user of a group sat in ONE state (its anchor concept folded)
and the policy learned a six-valued token lookup that generalised to nothing (M4c/M4d/M4e). Here each
user wears their OWN group-linked opener -- a film, a genre, a tone, or a mix -- so a group is a
CLOUD of states rather than a point, and fitting the training data at all requires learning
"these many different states get this group's questions". That is the mapping that can extend to an
opener never seen.

STATE-ONLY BY NECESSITY. There is no token to announce: the opener differs per user, so nothing
plays the role the anchor token played. Routing must come from the folded latent. `m4b_policy.py
--anchor_encoding none` is the trainer, and M4e already showed that encoding reproduces teachers at
identical fidelity.

THE HOLD-OUT. Train openers draw films only from **half A** of each group's diagnostic pool
(`opener_gen.pool_half`, a seeded hash of the item id, declared before any fit). Half B is reserved
for the generalisation test. The attribute channel is shared between halves -- the coarse vocabulary
is ~12 words per group and splitting it would empty a channel, so the generalisation claim is about
UNSEEN FILMS. Stated here because it must travel with the result.

NO NDCG IS EVER MEASURED ON A TRAIN USER (D.6b). The trees are frozen and only APPLIED; `z` comes
from `genv_eval_multi(..., score=False, return_z=True)`, the same object a policy rollout consumes.

PERSISTENCE (HARD RULE 11): one npz per (group, slice), written as it completes, skipped on resume.
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
M6 = os.path.join(B, "m6")
OUT = os.path.join(M6, "traj")
SEED = 98765
N_TRAIN = 140768
TOTAL_L = 15
ANCHORS = {"arthouse": 68, "dialogue": 275, "marvel": 581,
           "gore": 417, "teenagers": 922, "inspiring": 503}


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def main():
    ap = argparse.ArgumentParser(description="M6c stage 1 -- opener-fitted forest trajectories")
    ap.add_argument("--chunk", type=int, default=20000)
    ap.add_argument("--half", default="A", choices=["A", "B"])
    ap.add_argument("--groups", default="")
    ap.add_argument("--tree_suffix", default="",
                    help="use m6k_<group><suffix>.json instead of m6k_<group>.json")
    ap.add_argument("--out_dir", default="",
                    help="bank directory; defaults to m6/traj")
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--cold_tree", default="coldtree_v2.json",
                    help="the COLD teacher. Pass coldtree_v2_train10000.json to keep BOTH teachers train-fitted; mixing a validation-fitted cold arm with train-fitted group arms puts one teacher at home and the other away, which is the confound m6r exists to remove (home advantage measured at +0.02881).")
    ap.add_argument("--i_have_author_goahead", action="store_true")
    a = ap.parse_args()
    assert a.i_have_author_goahead, "REFUSING TO RUN without --i_have_author_goahead"
    torch.set_num_threads(a.threads)
    np.random.seed(SEED)
    global OUT
    if a.out_dir:
        OUT = a.out_dir if os.path.isabs(a.out_dir) else os.path.join(M6, a.out_dir)
    os.makedirs(OUT, exist_ok=True)

    from strategy_channel_suite import (build_greedy_env, genv_eval_multi, CERT_TOWER, CERT_CONC)
    from ndcg_tree import build_pool, branch_of_per_user, branch_of_selftest, LIKE_MIN_LEVEL
    from tree_route import route
    from m3_show_paths import build_names
    from opener_gen import centroid_lift, make_openers
    from m4b_eval import load_tree
    cname, _iname = build_names()

    # ---- which groups have an opener-fitted tree? --------------------------------------------
    want = [g.strip() for g in a.groups.split(",") if g.strip()] or list(ANCHORS)
    trees, valcoh, need_val = {}, {}, set()
    for g in want:
        p = os.path.join(M6, f"m6k_{g}{a.tree_suffix}.json")
        if not os.path.exists(p):
            log(f"[skip] {g}: no opener-fitted tree yet")
            continue
        d = json.load(open(p, encoding="utf-8"))
        nodes = d["nodes"]
        if isinstance(nodes, list):
            nodes = {r["nid"]: r for r in nodes}
        trees[g] = nodes
        # The group centroid is a property of the GROUP and is always computed on VALIDATION rows.
        # For a validation-fitted tree the artifact's opener keys ARE that cohort. For a TRAIN-fitted
        # tree they are train indices, which must never be used here -- that mismatch made the
        # centroid the mean of ~600 arbitrary users. Recover the cohort from validation instead.
        if d.get("split", "validation") == "validation":
            valcoh[g] = np.asarray(sorted(int(k) for k in d["openers"]), np.int64)
        else:
            need_val.add(g)
        log(f"[tree] {g}: {len(nodes)} nodes, split={d.get('split', 'validation')}")
    assert trees, "no opener-fitted trees found -- nothing to generate"
    if need_val:
        Ev = build_greedy_env(CERT_TOWER, "validation", CERT_CONC, "m6c_cent")
        vr = np.asarray(Ev["rows"], np.int64)
        Vv, Fv = np.asarray(Ev["V"]), np.asarray(Ev["foldable"])
        end_v = (Fv > 0) & (Vv > 0)
        for g in need_val:
            valcoh[g] = vr[end_v[vr, ANCHORS[g]]]
            log(f"[centroid] {g}: {len(valcoh[g])} VALIDATION users (tree was train-fitted)")
        del Ev
    _cd, cold_nodes = load_tree(os.path.join(B, a.cold_tree))
    log(f"[cold] teacher = {a.cold_tree} ({len(cold_nodes)} nodes)")

    zf = np.load(os.path.join(B, "m1", "z_full.npz"))
    ZF, ZR = zf["z"].astype(np.float64), zf["rows"]
    static = [(int(x), int(y)) for x, y in
              json.load(open(os.path.join(B, "greedy_static_policyB_combined.json")))["sequence"]]

    log("# M6c STAGE 1 -- OPENER-FITTED FOREST ON TRAIN USERS. NO NDCG IS MEASURED ON TRAIN.")
    log(f"# openers drawn from film-pool half {a.half!r}; half B is reserved for M6d")
    for lo in range(0, N_TRAIN, a.chunk):
        hi = min(lo + a.chunk, N_TRAIN)
        todo = [g for g in list(trees) + ["cold"]
                if not os.path.exists(os.path.join(OUT, f"{g}_{lo}_{hi}.npz"))]
        if not todo:
            log(f"[slice {lo}:{hi}] all banked -- skipping")
            continue
        t0 = time.time()
        E = build_greedy_env(CERT_TOWER, "train", CERT_CONC, f"m6cgen_{lo}", cohort_slice=(lo, hi))
        pool, _ip, _cp = build_pool(E, 500, 500)
        pidx = {tuple(q): i for i, q in enumerate(pool)}
        V, F = np.asarray(E["V"]), np.asarray(E["foldable"])
        endorse = (F > 0) & (V > 0)
        rows_loc = np.arange(E["ctx"].n, dtype=np.int64)
        in_any = np.zeros(len(rows_loc), bool)
        for cid in ANCHORS.values():
            in_any |= endorse[rows_loc, cid]
        gated = False

        for g in todo:
            if g == "cold":
                rows = rows_loc[~in_any]              # volunteers nothing -> the incumbent tree
                nodes, openers = cold_nodes, [[] for _ in rows]
            else:
                cid = ANCHORS[g]
                rows = rows_loc[endorse[rows_loc, cid]]
                if len(rows) < 50:
                    continue
                tab = centroid_lift(E, valcoh[g], ZF, ZR)
                # seed does NOT depend on the slice: a train user must wear the same opener
                # whichever batch they are generated in.
                openers, meta = make_openers(E, rows, tab, cname, LIKE_MIN_LEVEL,
                                             seed=SEED, half=a.half, group_key=g)
                nodes = trees[g]
                log(f"  [{g}] slice {lo}:{hi}: {len(rows)} users · film "
                    f"{meta['n_film'] / max(meta['n'], 1):.3f} genre "
                    f"{meta['n_genre'] / max(meta['n'], 1):.3f} tone "
                    f"{meta['n_tone'] / max(meta['n'], 1):.3f} · unserved {meta['unserved']} "
                    f"· pool {meta['pool_size']}")
            if len(rows) < 50:
                continue
            umask = {int(r): list(op) for r, op in zip(rows, openers)}
            seqs, diag = route(E, rows, nodes, TOTAL_L, static=static, mask=umask, on_conflict="burn")
            if diag["n_fallback"]:
                log(f"  [{g}] WARN {diag['n_fallback']}/{len(rows)} outside the fitted support "
                    f"-- continuation rule fired: {diag['sites']}")
            valid = np.ones((len(rows), TOTAL_L), bool)
            for u, r in enumerate(rows):
                fb = diag["fb_from"].get(int(r))
                if fb is not None:
                    valid[u, fb:] = False
            lab = np.full((len(rows), TOTAL_L), -1, np.int16)
            for u, s in enumerate(seqs):
                for p, q in enumerate(s):
                    assert tuple(q) in pidx, f"{g}: question {q} outside the pool"
                    lab[u, p] = pidx[tuple(q)]
            if not gated:
                branch_of_selftest(E, rows[:2000],
                                   sorted({tuple(x) for s in seqs for x in s})[:25])
                gated = True
            outc = np.zeros((len(rows), TOTAL_L), np.int8)
            for p in range(TOTAL_L):
                outc[:, p] = branch_of_per_user(E, rows, [s[p] for s in seqs])
            z = np.zeros((len(rows), TOTAL_L, E["ctx"].d), np.float32)
            for p in range(TOTAL_L):
                iseq, cseq = [], []
                for s, op in zip(seqs, openers):
                    pre = s[:p]
                    iseq.append(list(dict.fromkeys([i for c, i in op if c == 0] +
                                                   [q[1] for q in pre if q[0] == 0])))
                    cseq.append(list(dict.fromkeys([i for c, i in op if c == 1] +
                                                   [q[1] for q in pre if q[0] == 1])))
                _f, _t, zz = genv_eval_multi(E, rows, iseq, cseq, score=False, return_z=True)
                z[:, p, :] = np.asarray(zz, np.float32)
            np.savez_compressed(
                os.path.join(OUT, f"{g}_{lo}_{hi}.npz"),
                rows=rows.astype(np.int32), label_idx=lab, outcome=outc, z=z, valid=valid,
                anchor=-1 if g == "cold" else ANCHORS[g], half=a.half,
                opener=np.asarray([json.dumps([[int(c), int(i)] for c, i in op])
                                   for op in openers]),
                n_fallback=diag["n_fallback"])
            log(f"  [{g}] banked {len(rows)} x {TOTAL_L} "
                f"({int((~valid).sum())} positions masked as fallback)")
        log(f"[slice {lo}:{hi}] done in {(time.time() - t0) / 60:.1f}m")
        del E
    log("[done] all slices banked under " + OUT)


if __name__ == "__main__":
    main()
