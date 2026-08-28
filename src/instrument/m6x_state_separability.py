r"""M6x -- IS THE COHORT RECOVERABLE FROM THE OPENER STATE AT ALL?

The policy routes from one thing: the latent after the user's volunteered opener is folded in. If
that latent does not separate the cohorts, no amount of policy tuning will fix routing, and the
`marvel` deficit is a property of the DATA, not of the network.

Two measurements decide it.

1. A LINEAR PROBE, cohort from opener-state. This is the ceiling any router could reach with a
   linear read-out of `z`. If the probe confuses `marvel` with `teenagers` the same way the policy
   does, the policy is already near the ceiling and the state is the binding constraint. If the
   probe is far more accurate than the policy, routing is a training problem after all.
   ⚠ The probe is fitted and scored on DISJOINT halves of the validation cohort -- an in-sample
   probe would report the ceiling as higher than it is.

2. THE SAME USER UNDER TWO COHORTS. A user endorsing both `marvel` and `teenagers` receives a
   DIFFERENT opener under each (openers are seeded by (seed, cohort, user)), so they appear twice in
   training with different labels. That is only a contradiction if the two states are close. This
   measures how far apart they actually are, against the typical distance between two random users
   as the scale.

Neither number is a result. Both exist to choose the next experiment.

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
CONCEPTS = {"arthouse": 68, "dialogue": 275, "marvel": 581,
            "gore": 417, "teenagers": 922, "inspiring": 503}


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def main():
    ap = argparse.ArgumentParser(description="M6x -- can the cohort be read off the opener state?")
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--i_have_author_goahead", action="store_true")
    a = ap.parse_args()
    assert a.i_have_author_goahead, "REFUSING TO RUN without --i_have_author_goahead"
    torch.set_num_threads(a.threads)
    rng = np.random.default_rng(SEED)

    from strategy_channel_suite import (build_greedy_env, genv_eval_multi, CERT_TOWER, CERT_CONC)
    from ndcg_tree import build_pool, LIKE_MIN_LEVEL
    from m3_show_paths import build_names
    from opener_gen import centroid_lift, make_openers
    from run_guard import require_mains
    cname, _iname = build_names()
    require_mains("m6x separability", log=log)

    E = build_greedy_env(CERT_TOWER, "validation", CERT_CONC, "m6x")
    _pool, _i, _c = build_pool(E, 500, 500)
    vrows = np.asarray(E["rows"], np.int64)
    V, F = np.asarray(E["V"]), np.asarray(E["foldable"])
    end = (F > 0) & (V > 0)
    zf = np.load(os.path.join(B, "m1", "z_full.npz"))
    ZF, ZR = zf["z"].astype(np.float64), zf["rows"]

    # ---- the opener state for every (user, cohort) pair ------------------------------------------
    Z, lab, uid = [], [], []
    per_user = {}
    for g in sorted(CONCEPTS):
        rows = vrows[end[vrows, CONCEPTS[g]]]
        tab = centroid_lift(E, rows, ZF, ZR)
        ops, _m = make_openers(E, rows, tab, cname, LIKE_MIN_LEVEL, seed=SEED, group_key=g)
        iseq = [[i for c, i in o if c == 0] for o in ops]
        cseq = [[i for c, i in o if c == 1] for o in ops]
        z = np.asarray(genv_eval_multi(E, rows, iseq, cseq, score=False, return_z=True)[-1],
                       np.float64)
        Z.append(z); lab += [g] * len(rows); uid.append(np.asarray(rows, np.int64))
        for r, zz in zip(rows, z):
            per_user.setdefault(int(r), {})[g] = zz
        log(f"[state] {g:<11} {len(rows):>5} users")
    Z = np.concatenate(Z); uid = np.concatenate(uid)
    names = sorted(CONCEPTS)
    y = np.asarray([names.index(x) for x in lab])
    log(f"[state] {len(Z)} (user, cohort) states, dim {Z.shape[1]}")

    # ---- 1. linear probe, fitted and scored on DISJOINT users -------------------------------------
    users = np.unique(uid)
    perm = rng.permutation(len(users))
    tr_u = set(users[perm[: len(users) // 2]].tolist())
    tr = np.asarray([int(u) in tr_u for u in uid])
    mu, sd = Z[tr].mean(0), Z[tr].std(0) + 1e-8
    Zs = (Z - mu) / sd
    Xtr = torch.tensor(Zs[tr], dtype=torch.float32)
    ytr = torch.tensor(y[tr])
    W = torch.nn.Linear(Z.shape[1], len(names))
    opt = torch.optim.Adam(W.parameters(), lr=0.01, weight_decay=1e-4)
    for _ in range(400):
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(W(Xtr), ytr)
        loss.backward(); opt.step()
    with torch.no_grad():
        pred = W(torch.tensor(Zs[~tr], dtype=torch.float32)).argmax(1).numpy()
    yte = y[~tr]
    acc = float((pred == yte).mean())
    log("=" * 96)
    log(f"[probe] held-out accuracy {acc:.3f}   (chance {1/len(names):.3f})")
    log(f"[probe] confusion, TRUE row / PRED column:")
    log(f"{'':<11}" + "".join(f"{n[:9]:>10}" for n in names) + f"{'correct':>10}")
    conf = {}
    for i, g in enumerate(names):
        m = yte == i
        row = [float((pred[m] == j).mean()) for j in range(len(names))]
        conf[g] = dict(zip(names, row))
        log(f"{g:<11}" + "".join(f"{v:>10.3f}" for v in row) + f"{row[i]:>10.3f}")

    # ---- 2. the same user under two cohorts -------------------------------------------------------
    log("=" * 96)
    dual = {(g1, g2): [] for i, g1 in enumerate(names) for g2 in names[i + 1:]}
    for u, d in per_user.items():
        for i, g1 in enumerate(names):
            for g2 in names[i + 1:]:
                if g1 in d and g2 in d:
                    dual[(g1, g2)].append(np.linalg.norm(d[g1] - d[g2]))
    idx = rng.integers(0, len(Z), (4000, 2))
    ref = float(np.median(np.linalg.norm(Z[idx[:, 0]] - Z[idx[:, 1]], axis=1)))
    log(f"[dual] median distance between two RANDOM opener states = {ref:.4f}  (the scale)")
    log(f"[dual] the SAME user under two cohorts -- median distance, and as a fraction of that:")
    out_dual = {}
    for (g1, g2), v in sorted(dual.items(), key=lambda kv: -len(kv[1]))[:8]:
        if len(v) < 20:
            continue
        med = float(np.median(v))
        log(f"    {g1:<11}/{g2:<11} n={len(v):<5} {med:.4f}   {med/ref:.2f}x random")
        out_dual[f"{g1}|{g2}"] = {"n": len(v), "median": med, "frac_of_random": med / ref}
    json.dump({"analysis": "m6x_separability", "test_touched": False, "probe_acc": acc,
               "probe_confusion": conf, "random_state_distance": ref, "dual": out_dual},
              open(os.path.join(M6, "m6x_separability.json"), "w"), indent=1)
    log("=" * 96)
    log("[reading] probe ~= policy accuracy => the STATE is the constraint, tuning will not help;")
    log("          probe >> policy         => routing is trainable and the policy is underfit;")
    log("          dual distance << random => a user's two opener states really are near-identical.")
    log("[done]")


if __name__ == "__main__":
    main()
