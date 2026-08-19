r"""M4b stage 1 -- trajectory generation for the FOREST, on TRAIN users.

Emits, for every tree in the forest, the (state, label) examples a policy is distilled from:

    label_idx (n, L) int16   pool index of the question asked at position t == the LABEL at state t
    outcome   (n, L) int8    like / dislike / did-not-answer of that question, per user
    z         (n, L, d) f32  the latent AFTER folding the anchor (if any) + positions 0..t-1

Trees: the cold tree (all train users, no anchor) and the six teachers (train users who ENDORSE the
anchor, with the anchor folded at every position and masked from the label vocabulary).

NO NDCG IS EVER MEASURED ON A TRAIN USER (D.6b). The trees are frozen and only APPLIED; z comes from
`genv_eval_multi(..., score=False, return_z=True)`, the same object a policy rollout consumes, so
the policy's state and the scored interview are one implementation.

PERSISTENCE (HARD RULE 11): one npz per (tree, slice), written as it completes and skipped on resume.
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
OUT = os.path.join(B, "m4", "traj")
SEED = 98765
N_TRAIN = 140768
TOTAL_L = 15
ANCHORS = {"arthouse": 68, "dialogue": 275, "marvel": 581,
           "gore": 417, "teenagers": 922, "inspiring": 503}


def log(m: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def load_tree(path):
    d = json.load(open(path, encoding="utf-8"))
    nodes = d.get("nodes")
    if isinstance(nodes, list):
        nodes = {r["nid"]: r for r in nodes}
    if not nodes:
        nodes = json.load(open(path.replace(".json", "_partial.json"), encoding="utf-8"))["nodes"]
    return d, nodes


def route(E, rows, nodes, total_L):
    """Per-user ASKED question sequence, branching on each user's own answer."""
    from ndcg_tree import branch_of
    seq = {int(r): [] for r in rows}

    def rec(nid, rs):
        if not rs:
            return
        nd = nodes.get(nid)
        assert nd is not None, f"node {nid} missing"
        if not nd.get("split"):
            steps = [(int(a), int(b)) for a, b in nd["steps"]]
            for r in rs:
                seq[int(r)].extend(steps)
            return
        q = (int(nd["splitter"][0]), int(nd["splitter"][1]))
        for r in rs:
            seq[int(r)].append(q)
        lab, _ar, _nz = branch_of(E, rs, q)
        kids = {"like": [], "dislike": [], "noanswer": []}
        for r, l in zip(rs, lab):
            kids[str(l)].append(int(r))
        for br in ("like", "dislike", "noanswer"):
            if not kids[br]:
                continue
            child = f"{nid}.{br}"
            if child in nodes:
                rec(child, kids[br])
            else:
                tail = [(int(a), int(b)) for a, b in nd.get("tail", nd["steps"])]
                for r in kids[br]:
                    seq[int(r)].extend(tail[1:])
    rec("root", [int(r) for r in rows])
    out = []
    for r in rows:
        s = seq[int(r)]
        assert len(s) >= total_L, f"user {r}: {len(s)} questions, need {total_L}"
        out.append(s[:total_L])
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="M4b stage 1 -- forest trajectories on train users")
    ap.add_argument("--chunk", type=int, default=20000)
    ap.add_argument("--trees", default="cold,arthouse,dialogue,marvel,gore,teenagers,inspiring")
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--i_have_author_goahead", action="store_true")
    a = ap.parse_args()
    assert a.i_have_author_goahead, "REFUSING TO RUN without --i_have_author_goahead"
    torch.set_num_threads(a.threads); np.random.seed(SEED)
    os.makedirs(OUT, exist_ok=True)

    from strategy_channel_suite import (build_greedy_env, genv_eval_multi,
                                        CERT_TOWER, CERT_CONC)
    from ndcg_tree import build_pool

    want = [t.strip() for t in a.trees.split(",") if t.strip()]
    trees = {}
    for t in want:
        path = os.path.join(B, "coldtree_v2.json" if t == "cold" else f"m3v2_{t}.json")
        d, nodes = load_tree(path)
        trees[t] = (nodes, None if t == "cold" else ANCHORS[t])
        log(f"[tree] {t}: depth {d.get('depth_reached')}, {d.get('n_nodes')} nodes, "
            f"anchor {trees[t][1]}")

    log("# M4b STAGE 1 -- FOREST TRAJECTORIES ON TRAIN USERS. NO NDCG IS MEASURED ON TRAIN.")
    for lo in range(0, N_TRAIN, a.chunk):
        hi = min(lo + a.chunk, N_TRAIN)
        todo = [t for t in want if not os.path.exists(os.path.join(OUT, f"{t}_{lo}_{hi}.npz"))]
        if not todo:
            log(f"[slice {lo}:{hi}] all trees banked -- skipping")
            continue
        t0 = time.time()
        E = build_greedy_env(CERT_TOWER, "train", CERT_CONC, f"m4bgen_{lo}", cohort_slice=(lo, hi))
        pool, _ip, _cp = build_pool(E, 500, 500)
        pidx = {tuple(q): i for i, q in enumerate(pool)}
        V, F = np.asarray(E["V"]), np.asarray(E["foldable"])
        endorse = (F > 0) & (V > 0)
        rows_loc = np.arange(E["ctx"].n, dtype=np.int64)

        for t in todo:
            nodes, cid = trees[t]
            rows = rows_loc if cid is None else rows_loc[endorse[rows_loc, cid]]
            if len(rows) < 50:
                log(f"  [{t}] slice has only {len(rows)} endorsers -- writing empty")
                np.savez_compressed(os.path.join(OUT, f"{t}_{lo}_{hi}.npz"),
                                    rows=rows.astype(np.int32), label_idx=np.zeros((0, TOTAL_L), np.int16),
                                    z=np.zeros((0, TOTAL_L, E["ctx"].d), np.float32), anchor=-1 if cid is None else cid)
                continue
            seqs = route(E, rows, nodes, TOTAL_L)
            lab = np.full((len(rows), TOTAL_L), -1, np.int16)
            for u, s in enumerate(seqs):
                for p, q in enumerate(s):
                    assert tuple(q) in pidx, f"{t}: question {q} outside the pool"
                    lab[u, p] = pidx[tuple(q)]
            z = np.zeros((len(rows), TOTAL_L, E["ctx"].d), np.float32)
            for p in range(TOTAL_L):
                iseq, cseq = [], []
                for s in seqs:
                    pre = s[:p]
                    it = [q[1] for q in pre if q[0] == 0]
                    cc = ([cid] if cid is not None else []) + [q[1] for q in pre if q[0] == 1]
                    iseq.append(list(dict.fromkeys(it)))
                    cseq.append(list(dict.fromkeys(cc)))
                _f, _t2, zz = genv_eval_multi(E, rows, iseq, cseq, score=False, return_z=True)
                z[:, p, :] = np.asarray(zz, np.float32)
            np.savez_compressed(os.path.join(OUT, f"{t}_{lo}_{hi}.npz"),
                                rows=rows.astype(np.int32), label_idx=lab, z=z,
                                anchor=-1 if cid is None else cid)
            log(f"  [{t}] slice {lo}:{hi}: {len(rows)} users x {TOTAL_L} positions banked")
        log(f"[slice {lo}:{hi}] done in {(time.time() - t0) / 60:.1f}m")
        del E
    log("[done] all slices banked under " + OUT)


if __name__ == "__main__":
    main()
