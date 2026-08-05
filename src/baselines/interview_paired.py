r"""interview_paired.py -- paired per-user bootstrap CIs on the k=8 interview table.

WHY THIS EXISTS. `tab:instrdep` prints bare point estimates and makes two claims on them:

  (1) THE INVERSION -- the ranking of question-selection strategies depends on which recommender
      consumes the answers. Our instrument orders E0 > P > H; the Golbandi node model reproduces
      Rashid's published H > P > E0. That is a WITHIN-model claim about strategy gaps.
  (2) NO DEFICIT AT THE SHORT END -- a recommender that is strong at full profile is not thereby weak
      on a short interview. That is a CROSS-model claim at each model's own best strategy.

Neither had an interval. The chapter has already been bitten once by an unpaired CI on a
model-vs-model difference (`ARM_N_PHASE1.md` section 4b), so both are measured PAIRED here.

PAIRING IS VALID, and not merely convenient: every arm scores the SAME 10,000 held-out users against
the SAME targets under the SAME fixed arm-N candidate pool; only the asked questions (within-model) or
the consuming recommender (cross-model) changes. `metrics.evaluate` drops users with no held-out target
(and, for tail, users whose held-out set is all head), but those subsets are properties of the DATA,
so a given key stays row-aligned across arms -- asserted below rather than assumed.

Runs at k=8 only and NO FLOOR, which is the contract `tab:instrdep` states.

  python src/baselines/interview_paired.py --boot 10000
"""
import os
import sys
import json
import time
import argparse

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import interview_strategies as ST
from arm_n import load_arm_n, _ROOT
from run_interview_table import (build_recommenders, mostpop_row, score_arm, GLOBAL_ARMS, MAX_DEPTH,
                                 SNAP_DEFAULT)

OUT = os.path.join(_ROOT, "experiments", "baselines", "interview")
ARMS = ("helf", "entropy0", "popularity")
RECS = ("ours", "golbandi_leaf", "rbmf", "belief_mf", "tanp")
K = 8


def logln(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "interview_paired.log"), "a") as f:
        f.write(line + "\n")


def boot_ci(d, n_boot, seed=98765):
    """Percentile bootstrap over the per-user DIFFERENCE vector."""
    rng = np.random.RandomState(seed)
    idx = rng.randint(0, len(d), size=(n_boot, len(d)))
    means = d[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def paired(a, b, n_boot, label, out):
    """a - b, paired. Records the delta, its CI, and whether the interval excludes zero."""
    if len(a) != len(b):
        raise ValueError(f"{label}: vectors misaligned ({len(a)} vs {len(b)}) -- pairing invalid")
    d = np.asarray(a, float) - np.asarray(b, float)
    lo, hi = boot_ci(d, n_boot)
    sep = bool(lo > 0 or hi < 0)
    out[label] = {"n": int(len(d)), "delta": float(d.mean()), "ci": [lo, hi], "separated": sep}
    logln(f"    {label:52s} {d.mean():+.4f} CI [{lo:+.4f},{hi:+.4f}] {'SEP' if sep else 'ns '}")
    return out[label]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot", default=os.path.join(_ROOT, "models", "i26c_ep31",
                                                       "t2i26c_ep31.pt"))
    ap.add_argument("--arch", default="i26", choices=["i25", "i26"])
    ap.add_argument("--boot", type=int, default=10000)
    a = ap.parse_args()

    D = load_arm_n(log=logln)
    ni, n = D["n_items"], D["te_tr"].shape[0]
    glob, bank = ST.build_orders(D["g_train"], D["train"], ni, log=logln)
    lvl = ST.level_lookup(D["g_te_tr"])
    built = build_recommenders(list(RECS), D, arch=a.arch, snapshot=a.snapshot, log=logln)
    pop_row, prior = mostpop_row(D, log=logln)

    # ---------------------------------------------------------------- collect per-user vectors
    cells = {}          # cells[model][arm] = {"full": vec, "tail": vec, "mean_full": float, ...}
    for name in RECS:
        mode, obj = built[name]
        cells[name] = {}
        if name == "golbandi_leaf" and K > MAX_DEPTH:
            continue
        for arm in ARMS:
            asked, answered = ST.ask(arm, glob, bank, n, ni, lvl, K)
            ga = asked[0] if arm in GLOBAL_ARMS else None
            c = score_arm(name, mode, obj, asked, answered, D, pop_row, no_floor=True,
                          global_asked=ga, per_user=True)
            cells[name][arm] = c
            logln(f"[cell] {name:14s} {arm:11s} k={K} full={c['full']:.4f} tail={c['tail']:.4f} "
                  f"(n_full={len(c['full_per_user'])}, n_tail={len(c['tail_per_user'])})")

    # ---------------------------------------------------------------- alignment canary
    # Every arm must expose the same user subsets, or none of the pairing below is meaningful.
    nf = {len(c["full_per_user"]) for m in cells for c in cells[m].values()}
    nt = {len(c["tail_per_user"]) for m in cells for c in cells[m].values()}
    assert len(nf) == 1 and len(nt) == 1, f"user subsets differ across arms: full={nf} tail={nt}"
    logln(f"[canary] all arms row-aligned: n_full={nf.pop()}, n_tail={nt.pop()}")

    res = {"k": K, "no_floor": True, "snapshot": os.path.basename(a.snapshot), "boot": a.boot,
           "mostpop_prior": prior, "points": {}, "within_model_strategy": {}, "cross_model": {}}
    for m in cells:
        res["points"][m] = {arm: {"full": cells[m][arm]["full"], "tail": cells[m][arm]["tail"]}
                            for arm in cells[m]}

    # ---------------------------------------------------------------- (1) THE INVERSION
    logln("=== within-model strategy gaps (the inversion claim) ===")
    for m in cells:
        logln(f"  -- {m}")
        for ax in ("full", "tail"):
            for x, y in (("entropy0", "helf"), ("entropy0", "popularity"), ("popularity", "helf")):
                if x not in cells[m] or y not in cells[m]:
                    continue
                paired(cells[m][x][f"{ax}_per_user"], cells[m][y][f"{ax}_per_user"],
                       a.boot, f"{m}/{ax}/{x}-minus-{y}", res["within_model_strategy"])

    # ---------------------------------------------------------------- (2) NO DEFICIT AT k=8
    # Each model at ITS OWN best strategy -- the comparison a reader actually makes.
    logln("=== cross-model at each model's best strategy (the no-deficit claim) ===")
    best = {m: max(cells[m], key=lambda arm: cells[m][arm]["full"]) for m in cells}
    logln(f"  best strategy per model: {best}")
    res["best_strategy"] = best
    for m in cells:
        if m == "ours":
            continue
        for ax in ("full", "tail"):
            paired(cells["ours"][best["ours"]][f"{ax}_per_user"],
                   cells[m][best[m]][f"{ax}_per_user"],
                   a.boot, f"ours({best['ours']})-minus-{m}({best[m]})/{ax}", res["cross_model"])

    outp = os.path.join(OUT, "interview_paired.json")
    json.dump(res, open(outp, "w"), indent=2)
    logln(f"done -> {outp}")


if __name__ == "__main__":
    main()
