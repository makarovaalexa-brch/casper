r"""M6m -- the opener-fitted trees on HELD-OUT USERS: the 140,768 TRAIN users.

Author's ruling, 2026-08-20: *"train is ok, final numbers will be on test anyway"* — amending D.6b.

WHY TRAIN USERS ARE THE RIGHT HELD-OUT COHORT HERE
----------------------------------------------------
The trees were fitted on VALIDATION users, so every train user is genuinely out of sample. Using
them costs no refitting and gives ~11,000 users per group instead of ~400-1,800 — roughly a 5x cut
in standard error, which is the direct answer to "the fresh-draw spread is about one noise-width".

⚠ D.6b FORBADE NDCG ON TRAIN USERS. Its two stated reasons were "no fold-in/target split" and "the
tower memorised them". The FIRST NO LONGER HOLDS -- `train_tr_te` provides a cached 80/20 split,
2,114,192 target-half rows over all 140,768 users. The SECOND STILL HOLDS: the recommender was
trained on these users, so **LEVELS ARE INFLATED AND MUST NOT BE QUOTED BESIDE VALIDATION OR TEST
NUMBERS**. What is reported here is a CONTRAST between arms on the same users, where the
memorisation advantage is common to every arm and cancels -- the same argument the chapter already
makes for the shared encoder. Every number this file writes is stamped `ruler: train_memorised`.

FOUR ARMS, identical opener per user in all of them, folded free, and a question the user already
volunteered is folded once with THE TURN BURNED (`on_conflict="burn"`, matching how the trees were
fitted):
  opener-fitted tree · the banked anchor teacher · the cold tree · the static order

Openers are DETERMINISTIC IN THE USER (`opener_gen`, per-user seeding), so there is no draw to
overfit to and no seed to re-roll.

**THE TEST SET IS NEVER BUILT, LOADED OR NAMED IN THIS FILE.**
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
SEED = 98765
TOTAL_L = 15
N_TRAIN = 140768
ANCHORS = {"arthouse": 68, "dialogue": 275, "marvel": 581,
           "gore": 417, "teenagers": 922, "inspiring": 503}
BUDGETS = [1, 4, 15]


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def paired_boot(d, n=10000, seed=SEED):
    d = np.asarray(d, np.float64)
    d = d[np.isfinite(d)]
    if len(d) == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    bs = d[rng.integers(0, len(d), size=(n, len(d)))].mean(1)
    return float(d.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def main():
    ap = argparse.ArgumentParser(description="M6m -- opener-fitted trees on held-out TRAIN users")
    ap.add_argument("--chunk", type=int, default=20000)
    ap.add_argument("--max_users", type=int, default=N_TRAIN)
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--i_have_author_goahead", action="store_true")
    a = ap.parse_args()
    assert a.i_have_author_goahead, "REFUSING TO RUN without --i_have_author_goahead"
    torch.set_num_threads(a.threads)
    np.random.seed(SEED)

    from strategy_channel_suite import (build_greedy_env, genv_eval_multi, CERT_TOWER, CERT_CONC)
    from ndcg_tree import build_pool, LIKE_MIN_LEVEL
    from tree_route import route
    from m4b_eval import load_tree
    from m3_show_paths import build_names
    from opener_gen import centroid_lift, make_openers
    cname, _iname = build_names()

    log("#" * 100)
    log("# M6m -- OPENER-FITTED TREES ON HELD-OUT TRAIN USERS (ruler: train_memorised)")
    log("# levels are inflated and must not be quoted beside val/test; CONTRASTS are the result")
    log("#" * 100)

    trees, valcoh, anchors = {}, {}, {}
    for g in ANCHORS:
        p = os.path.join(M6, f"m6k_{g}.json")
        if not os.path.exists(p):
            continue
        d = json.load(open(p, encoding="utf-8"))
        nd = d["nodes"]
        trees[g] = {r["nid"]: r for r in nd} if isinstance(nd, list) else nd
        valcoh[g] = np.asarray(sorted(int(k) for k in d["openers"]), np.int64)
        _ad, anchors[g] = load_tree(os.path.join(B, f"m3v2_{g}.json"))
    assert trees, "no opener-fitted trees"
    _cd, cold_nodes = load_tree(os.path.join(B, "coldtree_v2.json"))
    static = [(int(x), int(y)) for x, y in
              json.load(open(os.path.join(B, "greedy_static_policyB_combined.json")))["sequence"]]
    zf = np.load(os.path.join(B, "m1", "z_full.npz"))
    ZF, ZR = zf["z"].astype(np.float64), zf["rows"]
    log(f"[trees] {sorted(trees)}")

    # per-user score vectors, accumulated across slices
    acc = {g: {arm: {K: [] for K in BUDGETS} for arm in ("tree", "anchor", "cold", "static")}
           for g in trees}
    nseen = {g: 0 for g in trees}

    for lo in range(0, min(a.max_users, N_TRAIN), a.chunk):
        hi = min(lo + a.chunk, min(a.max_users, N_TRAIN))
        t0 = time.time()
        E = build_greedy_env(CERT_TOWER, "train", CERT_CONC, f"m6m_{lo}", cohort_slice=(lo, hi))
        pool, _i, _c = build_pool(E, 500, 500)
        V, F = np.asarray(E["V"]), np.asarray(E["foldable"])
        endorse = (F > 0) & (V > 0)
        rows_loc = np.arange(E["ctx"].n, dtype=np.int64)
        for g in trees:
            cid = ANCHORS[g]
            rows = rows_loc[endorse[rows_loc, cid]]
            if len(rows) < 50:
                continue
            tab = centroid_lift(E, valcoh[g], ZF, ZR)
            ops, _meta = make_openers(E, rows, tab, cname, LIKE_MIN_LEVEL, seed=SEED, group_key=g)
            umask = {int(r): list(o) for r, o in zip(rows, ops)}
            seqs = {}
            for arm, nodes in (("tree", trees[g]), ("anchor", anchors[g]), ("cold", cold_nodes)):
                seqs[arm], _dg = route(E, rows, nodes, TOTAL_L, static=static, mask=umask,
                                       on_conflict="burn")
            seqs["static"] = [static[:TOTAL_L]] * len(rows)
            for arm in seqs:
                for K in BUDGETS:
                    iseq, cseq = [], []
                    for s, op in zip(seqs[arm], ops):
                        ask = s[:K]
                        iseq.append(list(dict.fromkeys([i for c, i in op if c == 0] +
                                                       [q[1] for q in ask if q[0] == 0])))
                        cseq.append(list(dict.fromkeys([i for c, i in op if c == 1] +
                                                       [q[1] for q in ask if q[0] == 1])))
                    full, _tail = genv_eval_multi(E, rows, iseq, cseq)
                    acc[g][arm][K].append(np.asarray(full, np.float64))
            nseen[g] += len(rows)
        log(f"[slice {lo}:{hi}] done in {(time.time() - t0) / 60:.1f}m · "
            + " ".join(f"{g}={nseen[g]}" for g in sorted(nseen)))
        del E

    log("=" * 100)
    out = {"analysis": "m6m_heldout_train", "ruler": "train_memorised", "test_touched": False,
           "note": "levels inflated (tower trained on these users); CONTRASTS are the result",
           "groups": {}}
    for g in sorted(trees):
        if not acc[g]["tree"][15]:
            continue
        v = {arm: {K: np.concatenate(acc[g][arm][K]) for K in BUDGETS} for arm in acc[g]}
        rec = {"n": int(nseen[g]), "budgets": {}}
        for K in BUDGETS:
            row = {arm: float(np.nanmean(v[arm][K])) for arm in v}
            for opp in ("cold", "anchor", "static"):
                m, l, h = paired_boot(v["tree"][K] - v[opp][K])
                row[f"tree_minus_{opp}"] = m
                row[f"ci_{opp}"] = [l, h]
            rec["budgets"][str(K)] = row
        out["groups"][g] = rec
        c = rec["budgets"]["15"]
        st = lambda k: ("*" if c[f"ci_{k}"][0] > 0 else ("-" if c[f"ci_{k}"][1] < 0 else " "))
        log(f"  {g:<11} n={rec['n']:<6} K=15 tree {c['tree']:.5f} anchor {c['anchor']:.5f} "
            f"cold {c['cold']:.5f} static {c['static']:.5f} | "
            f"vs cold {c['tree_minus_cold']:+.5f}{st('cold')} "
            f"vs anchor {c['tree_minus_anchor']:+.5f}{st('anchor')} "
            f"vs static {c['tree_minus_static']:+.5f}{st('static')}")
        json.dump(out, open(os.path.join(M6, "m6m_heldout_train.json"), "w"), indent=1)
    log("[done]")


if __name__ == "__main__":
    main()
