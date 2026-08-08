r"""ials_ladder_paired.py -- paired per-user CIs on the iALS graded ladder.

WHY. run_ials_conf_decompose.py separated iALS-confidence's +0.0451 into reg (+0.0257), support (-0.0088)
and grades (+0.0282) using MEAN differences only. Mean differences are not evidence of separation, and
this chapter has already been burned once by an UNPAIRED interval (the retracted "parity with RecVAE"
claim, where the 1.96*sqrt(2)*SE signature made a real gap look like a tie). The grades delta is about to
overturn a stated claim in the chapter, so it must carry a paired interval before it is written down.

FOUR RUNGS, one variable per step, alphas as selected on validation by the decomposition run:
  r0 binary, likes-only, reg 0.1,  alpha 50    the published binary iALS row
  r1 binary, likes-only, reg 0.01, alpha 10    isolates the regularisation
  r2 binary, ALL-rated,  reg 0.01, alpha 5     isolates support size
  r3 graded, ALL-rated,  reg 0.01, alpha 2.5   iALS-confidence, c = 1 + alpha*r

CONTRASTS TESTED (all paired on the same 10,000 test users, full and tail):
  r1-r0  regularisation
  r2-r1  support: what the dislike ITEMS are worth when their values are erased
  r3-r2  GRADES at fixed support -- the quantity the chapter's claim turns on
  r3-r1  grades+support against the best honest binary cell, which is the comparison a
         reviewer will actually make, since r1 beats r0 outright
  r3-r0  the headline, for completeness

Alphas are NOT re-tuned here; they are the validation-selected values from the decomposition run, so this
script measures the same four models it names. G1 (MostPop through this eval path) still runs.

  python src/baselines/ials_ladder_paired.py
"""
import os
import sys
import json
import time
import argparse

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
sys.path.insert(0, _HERE)

import metrics as M
import pop
import ials
from arm_n import load_arm_n

OUTDIR = os.path.join(_ROOT, "experiments", "baselines", "arm_n")

# (label, graded?, all_rated?, reg, alpha) -- alphas val-selected by run_ials_conf_decompose.py
RUNGS = [
    ("r0_binary_likes_reg01",     False, False, 0.1,  50.0),
    ("r1_binary_likes_reg001",    False, False, 0.01, 10.0),
    ("r2_binary_allrated_reg001", False, True,  0.01,  5.0),
    ("r3_graded_allrated_reg001", True,  True,  0.01,  2.5),
]
CONTRASTS = [
    ("regularisation            r1-r0", "r1_binary_likes_reg001", "r0_binary_likes_reg01"),
    ("support (values erased)   r2-r1", "r2_binary_allrated_reg001", "r1_binary_likes_reg001"),
    ("GRADES at fixed support   r3-r2", "r3_graded_allrated_reg001", "r2_binary_allrated_reg001"),
    ("grades vs best binary     r3-r1", "r3_graded_allrated_reg001", "r1_binary_likes_reg001"),
    ("headline                  r3-r0", "r3_graded_allrated_reg001", "r0_binary_likes_reg01"),
]


def logln(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    os.makedirs(OUTDIR, exist_ok=True)
    with open(os.path.join(OUTDIR, "ials_ladder_paired.log"), "a") as f:
        f.write(line + "\n")


def boot_ci(d, n_boot, seed=98765):
    rng = np.random.RandomState(seed)
    idx = rng.randint(0, len(d), size=(n_boot, len(d)))
    means = d[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def binarise(X):
    B = X.tocsr().copy().astype(np.float32)
    B.data[:] = 1.0
    return B


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--boot", type=int, default=10000)
    ap.add_argument("--factors", type=int, default=200)
    ap.add_argument("--iterations", type=int, default=15)
    a0 = ap.parse_args()

    D = load_arm_n(log=logln)
    ni, hm, pool, te_te = D["n_items"], D["head_mask"], D["pool"], D["te_te"]

    pr_pop = pop.fit(D["train"], ni, log=logln)
    rp = M.evaluate(pr_pop, D["te_tr"], te_te, batch_size=500, head_mask=hm, mask_X=pool)
    canon = json.load(open(os.path.join(OUTDIR, "pop.json")))["N"]
    ok = abs(rp["ndcg@10"] - canon["ndcg@10"]) <= 1e-4
    logln(f"[G1] MostPop this path {rp['ndcg@10']:.4f} vs arm-N {canon['ndcg@10']:.4f} -> "
          f"{'PASS' if ok else '*** FAIL ***'}")
    if not ok:
        raise SystemExit("[G1] eval path does not reproduce MostPop; fix wiring first")

    inputs = {
        (False, False): (D["train"], D["te_tr"]),
        (False, True): (binarise(D["g_train"]), binarise(D["g_te_tr"])),
        (True, True): (D["g_train"], D["g_te_tr"]),
    }

    vecs, means = {}, {}
    for label, graded, allrated, reg, alpha in RUNGS:
        tr, te_in = inputs[(graded, allrated)]
        t0 = time.time()
        model = ials._fit_model(tr, a0.factors, reg, alpha, a0.iterations, log=lambda *_: None)
        prd = ials._make_predict(model, reg, alpha)
        r = M.evaluate(prd, te_in, te_te, batch_size=500, head_mask=hm, mask_X=pool, per_user=True)
        vecs[label] = {"full": np.asarray(r["ndcg@10_per_user"], float),
                       "tail": np.asarray(r["tail_ndcg@10_per_user"], float)}
        means[label] = {"full": r["ndcg@10"], "tail": r["tail_ndcg@10"]}
        logln(f"[rung] {label:28s} reg={reg:<5} alpha={alpha:<5} "
              f"full={r['ndcg@10']:.4f} tail={r['tail_ndcg@10']:.4f} ({(time.time()-t0)/60:.1f}m)")

    out = {"rungs": {k: {**means[k], "reg": r[3], "alpha": r[4]}
                     for k, r in zip([x[0] for x in RUNGS], RUNGS)}, "contrasts": {}}
    logln("---- paired, per-user, %d bootstrap resamples ----" % a0.boot)
    for name, A, B in CONTRASTS:
        cell = {}
        for axis in ("full", "tail"):
            va, vb = vecs[A][axis], vecs[B][axis]
            # A tail vector can contain NaN for users with no tail target; pair on the users where BOTH
            # are defined, and record how many that is -- an unstated n is how an unpaired CI hides.
            keep = np.isfinite(va) & np.isfinite(vb)
            d = va[keep] - vb[keep]
            lo, hi = boot_ci(d, a0.boot)
            sep = bool(lo > 0 or hi < 0)
            cell[axis] = {"n": int(keep.sum()), "delta": float(d.mean()), "ci": [lo, hi],
                          "separated": sep}
            logln(f"  {name:34s} {axis:4s} {d.mean():+.4f} CI [{lo:+.4f},{hi:+.4f}] "
                  f"{'SEP' if sep else 'ns '} (n={int(keep.sum())})")
        out["contrasts"][name] = cell

    g = out["contrasts"]["GRADES at fixed support   r3-r2"]
    logln("=" * 78)
    logln("[verdict] grades at fixed support: full %+.4f %s | tail %+.4f %s"
          % (g["full"]["delta"], "SEPARATED" if g["full"]["separated"] else "not separated",
             g["tail"]["delta"], "SEPARATED" if g["tail"]["separated"] else "not separated"))
    logln("=" * 78)

    p = os.path.join(OUTDIR, "ials_ladder_paired.json")
    json.dump(out, open(p, "w"), indent=2)
    logln(f"done -> {p}")


if __name__ == "__main__":
    main()
