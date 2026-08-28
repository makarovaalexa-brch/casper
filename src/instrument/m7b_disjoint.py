r"""M7b -- MAKE THE SIX GROUPS A PARTITION, IN BOTH PLACES THEY OVERLAP.

THE PROBLEM, MEASURED. The router must pick which group's interview to run from what the user
volunteered. It cannot: a linear probe recovers the group from the opener state at 0.371 on the worst
group and the trained policy achieves 0.364 -- the policy is at the INFORMATION ceiling, not the
optimisation ceiling, which is why rebalancing and re-encoding both failed.

THE CAUSE, MEASURED. The groups advertise the same material, so what a user volunteers does not
identify them:

    marvel / inspiring share 10 of their 12 characteristic genres and tones
    31% of the films used in openers appear in more than one group's pool

A user who volunteers `sad`, `emotional` or `adventure` has named a word belonging to four groups at
once. Pool overlap predicts routing confusion group by group: marvel/inspiring overlap most and
marvel leaks to inspiring at 0.202; arthouse/marvel overlap 0.000 and never confuse.

HOW IT HAPPENED. The six groups began as disjoint taste clusters in M1. They were converted into six
concept WORDS, and cohorts were then defined as "everyone who endorses that word" -- which overlaps.
The partition was discarded at that step and the labels were never banked, so it could not be
recovered. This restores it without returning to k-means.

WHAT THIS DOES -- two argmaxes over quantities we already compute:

  USERS     each user -> the single group they endorse most strongly (highest signed value V).
            No user is lost; a user in two groups is placed in one.
  MATERIAL  each film and each concept -> the group whose centroid lift for it is highest.
            A group's opener pool becomes what it wins outright.

⚠ ASYMMETRY, EXPECTED AND REPORTED. When two groups shared a film only one keeps it, so pools become
uneven. `inspiring` and `dialogue` lose most of their films to `arthouse` and `marvel` and will lean
on their vocabulary instead. `dialogue`'s exclusive words are `gritty, drama` -- generic -- so if
anything stays unroutable it should be `dialogue`. That is a prediction, recorded before the fit.

Lift is computed on VALIDATION users, as everywhere else in this project: the group centroid is a
property of the GROUP, not of the fitting split.

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

B = os.path.join(_R, "experiments", "battery")
M1 = os.path.join(B, "m1")
M6 = os.path.join(B, "m6")
SEED = 98765
CONCEPTS = {"arthouse": 68, "dialogue": 275, "marvel": 581,
            "gore": 417, "teenagers": 922, "inspiring": 503}
NAMES = list(CONCEPTS)


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def assign_users(V, endorse):
    """Each user -> the group they endorse most strongly. Returns (best, has_any)."""
    cids = np.array([CONCEPTS[g] for g in NAMES])
    Vs = V[:, cids].astype(np.float64).copy()
    Es = endorse[:, cids]
    Vs[~Es] = -np.inf
    return Vs.argmax(1).astype(np.int32), Es.any(1)


def main():
    ap = argparse.ArgumentParser(description="M7b -- disjoint groups and disjoint opener material")
    ap.add_argument("--top_frac", type=float, default=0.05)
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--i_have_author_goahead", action="store_true")
    a = ap.parse_args()
    assert a.i_have_author_goahead, "REFUSING TO RUN without --i_have_author_goahead"
    import torch
    torch.set_num_threads(a.threads)

    from strategy_channel_suite import build_greedy_env, CERT_TOWER, CERT_CONC
    from m3_show_paths import build_names
    from opener_gen import centroid_lift, coarse_ids
    from run_guard import require_mains
    cname, iname = build_names()
    require_mains("m7b disjoint assignment", log=log)

    E = build_greedy_env(CERT_TOWER, "validation", CERT_CONC, "m7b")
    vr = np.asarray(E["rows"], np.int64)
    V, F = np.asarray(E["V"]), np.asarray(E["foldable"])
    endorse = (F > 0) & (V > 0)
    zf = np.load(os.path.join(M1, "z_full.npz"))
    ZF, ZR = zf["z"].astype(np.float64), zf["rows"]

    # ---- material: who wins each film and each concept -------------------------------------------
    lift_i, lift_c, pools = [], [], {}
    for g in NAMES:
        tab = centroid_lift(E, vr[endorse[vr, CONCEPTS[g]]], ZF, ZR, top_frac=a.top_frac)
        lift_i.append(tab["lift_item"]); lift_c.append(tab["lift_conc"])
        pools[g] = set(int(x) for x in tab["pool"])
    LI, LC = np.stack(lift_i), np.stack(lift_c)
    own_i = LI.argmax(0).astype(np.int32)
    own_c = LC.argmax(0).astype(np.int32)

    # ---- users --------------------------------------------------------------------------------
    best, has = assign_users(V, endorse)
    genres, tones = coarse_ids(cname)
    out = {"analysis": "m7b_disjoint", "test_touched": False, "names": NAMES,
           "top_frac": a.top_frac, "groups": {}}
    log("=" * 96)
    log(f"{'group':<11} {'users(ov)':>10} {'users(ex)':>10} {'pool(ov)':>9} {'pool(ex)':>9}  exclusive vocabulary")
    for i, g in enumerate(NAMES):
        ov_u = int(endorse[vr, CONCEPTS[g]].sum())
        ex_u = int(((best[vr] == i) & has[vr]).sum())
        ex_p = sorted(f for f in pools[g] if own_i[f] == i)
        lc = LC[i]
        top = sorted([c for c in list(genres) + list(tones) if lc[c] > 0], key=lambda c: -lc[c])[:12]
        excl = [c for c in top if own_c[c] == i]
        out["groups"][g] = {"users_overlapping": ov_u, "users_exclusive": ex_u,
                            "pool_overlapping": len(pools[g]), "pool_exclusive": len(ex_p),
                            "vocab_exclusive": [cname.get(c, str(c)) for c in excl],
                            "pool": ex_p}
        log(f"{g:<11} {ov_u:>10} {ex_u:>10} {len(pools[g]):>9} {len(ex_p):>9}  "
            f"{', '.join(cname.get(c, str(c)) for c in excl[:6])}")
    log("=" * 96)
    n_u = int(has[vr].sum())
    log(f"[users] {int(endorse[vr][:, [CONCEPTS[g] for g in NAMES]].sum())} memberships -> "
        f"{n_u} users, each in exactly ONE group. No user is dropped.")
    thin = [g for g in NAMES if out["groups"][g]["pool_exclusive"] < 200]
    log(f"[warn] groups leaning on vocabulary rather than films (pool < 200): {thin or 'none'}")

    np.savez_compressed(os.path.join(M6, "m7b_disjoint.npz"),
                        own_item=own_i, own_concept=own_c,
                        val_rows=vr.astype(np.int64), val_group=best[vr].astype(np.int32),
                        val_has=has[vr])
    json.dump(out, open(os.path.join(M6, "m7b_disjoint.json"), "w"), indent=1)
    log("[done] banked m7b_disjoint.npz (own_item, own_concept, per-user group)")


if __name__ == "__main__":
    main()
