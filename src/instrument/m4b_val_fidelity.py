r"""M4b DIAGNOSTIC 2 -- teacher-forced agreement per position on VALIDATION states, all seven trees.

THE CONTROL THIS SUPPLIES. Diagnostic 1 showed the policy reproduces every teacher at >= 0.992 at
every position on held-out TRAIN states, while stage 3's rollout on VALIDATION drops to 0.63-0.90 at
the first branching decision. The pooled headline was not hiding the drop; train and validation
states are simply not exchangeable.

Two explanations remain, and the COLD TREE separates them:

  (i) TEACHER-SPECIFIC. Each teacher is fitted on 8-24k train users, against 140,768 for the cold
      tree, and its splits are cohort-specific. Then the cold tree should still transfer at ~1.0 --
      which is exactly what E1 measured for a policy distilled on the cold tree alone (top-1 1.0000
      at all sixteen positions on validation states).
  (ii) A GENERAL STATE-DISTRIBUTION GAP. The certified tower was TRAINED on the 140,768 train users,
      so their latents are in-sample for the encoder and validation users' are not. The policy
      consumes that latent, so it would inherit the tower's own generalisation gap -- and then the
      cold tree must degrade too.

This measures TEACHER-FORCED agreement (states supplied by the tree, exactly as in diagnostic 1), so
nothing here can be rollout compounding. VALIDATION ONLY; no NDCG is computed; the TEST cohort is
never built, loaded or named.
"""
from __future__ import annotations

import os
import sys

_H = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _H)

import argparse
import json

import numpy as np
import torch

from m4b_policy import OUT, TOTAL_L, TREES, log

B = os.path.dirname(OUT)
ANCHORS = {"arthouse": 68, "dialogue": 275, "marvel": 581,
           "gore": 417, "teenagers": 922, "inspiring": 503}


def main():
    ap = argparse.ArgumentParser(description="per-position teacher-forced agreement on VALIDATION")
    ap.add_argument("--ckpt", default=os.path.join(OUT, "m4b_policy.pt"))
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--i_have_author_goahead", action="store_true")
    a = ap.parse_args()
    assert a.i_have_author_goahead, "REFUSING TO RUN without --i_have_author_goahead"
    torch.set_num_threads(a.threads)

    from strategy_channel_suite import (build_greedy_env, genv_eval_multi, CERT_TOWER, CERT_CONC)
    from ndcg_tree import build_pool, branch_of_per_user
    from policy_net import Policy, build_phi
    from tree_route import route
    from m4b_eval import load_tree

    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    E = build_greedy_env(CERT_TOWER, "validation", CERT_CONC, "m4bvalfid")
    rows_all = np.asarray(E["rows"], np.int64)
    pool, _i, _c = build_pool(E, 500, 500)
    assert [list(q) for q in pool] == ck["pool"], "pool moved since training"
    pidx = {tuple(q): i for i, q in enumerate(pool)}
    ext = [tuple(q) for q in ck["ext"]]
    norm = (np.asarray(ck["norm"][0], np.float32), np.asarray(ck["norm"][1], np.float32))
    phi_ext, _ = build_phi(E, ext, norm=norm)
    n_pool, d_z = ck["n_pool"], ck["d_z"]
    pol = Policy(d_z, ck["d_phi"], n_pool, arch=ck["arch"], state=ck["state"], use_pos=True)
    pol.load_state_dict(ck["sd"])
    pol.eval()
    phi_cand = torch.from_numpy(phi_ext[:n_pool])

    static = [(int(x), int(y)) for x, y in
              json.load(open(os.path.join(B, "greedy_static_policyB_combined.json")))["sequence"]]
    V, F = np.asarray(E["V"]), np.asarray(E["foldable"])
    endorse = (F > 0) & (V > 0)
    out = {}

    for t in TREES:
        cid = ANCHORS.get(t)
        path = os.path.join(B, "coldtree_v2.json" if t == "cold" else f"m3v2_{t}.json")
        _d, nodes = load_tree(path)
        rows = rows_all if cid is None else rows_all[endorse[rows_all, cid]]
        seqs, _diag = route(E, rows, nodes, TOTAL_L, static=static,
                            mask=[] if cid is None else [(1, cid)])
        n = len(rows)
        a_phi = ck["a_phi"][t]
        a_pool = ck["a_pool"][t]
        by = []
        for p in range(TOTAL_L):
            # the state the TREE puts the user in at position p (teacher-forced, no compounding)
            iseq, cseq = [], []
            for s in seqs:
                pre = s[:p]
                iseq.append(list(dict.fromkeys([q[1] for q in pre if q[0] == 0])))
                cseq.append(list(dict.fromkeys(([cid] if cid is not None else []) +
                                               [q[1] for q in pre if q[0] == 1])))
            z = np.asarray(genv_eval_multi(E, rows, iseq, cseq, score=False, return_z=True)[-1],
                           np.float32)
            hq = np.zeros((n, TOTAL_L), np.int64)
            ho = np.zeros((n, TOTAL_L), np.int64)
            hm = np.zeros((n, TOTAL_L), np.float32)
            s0 = 0
            if cid is not None:
                hq[:, 0] = a_phi
                hm[:, 0] = 1.0
                s0 = 1
            asked_m = np.zeros((n, n_pool), bool)
            if a_pool >= 0:
                asked_m[:, a_pool] = True
            for j in range(p):
                qs = [s[j] for s in seqs]
                hq[:, s0 + j] = [pidx[tuple(q)] for q in qs]
                ho[:, s0 + j] = branch_of_per_user(E, rows, qs)
                hm[:, s0 + j] = 1.0
                asked_m[np.arange(n), hq[:, s0 + j]] = True
            y = np.asarray([pidx[tuple(s[p])] for s in seqs], np.int64)
            asked_m[np.arange(n), y] = False
            pos = np.tile(np.asarray([[p, TOTAL_L - p]], np.float32), (n, 1))
            hit = 0
            with torch.no_grad():
                for st in range(0, n, 2048):
                    sl = slice(st, min(st + 2048, n))
                    s = pol(torch.from_numpy(z[sl]), torch.from_numpy(phi_ext[hq[sl]]),
                            torch.from_numpy(ho[sl]), torch.from_numpy(hm[sl]),
                            torch.from_numpy(pos[sl]), phi_cand, torch.from_numpy(asked_m[sl]))
                    hit += int((s.argmax(1).numpy() == y[sl]).sum())
            by.append(hit / n)
        out[t] = by
        log(f"  [{t}] n={n:<6} " + " ".join(f"{v:.3f}" for v in by) +
            f"   pooled {np.mean(by):.4f}")
    json.dump({"analysis": "m4b_fidelity_by_position", "split": "validation",
               "test_touched": False, "ckpt_epoch": ck["epoch"], "by_position": out},
              open(os.path.join(OUT, "m4b_val_fidelity.json"), "w"), indent=1)
    log("[done]")


if __name__ == "__main__":
    main()
