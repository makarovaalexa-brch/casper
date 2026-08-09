r"""rbmf_graded_paired.py -- paired per-user CI on RBMF's graded-vs-binary contrast.

WHY THIS IS NOT OPTIONAL. run_rbmf_graded_reselected.py returned graded minus binary = -0.0045 full and
+0.0046 tail, and its coded verdict printed "still loses" purely because the full delta is negative. A
0.0045 gap with a positive tail delta is not a loss in any useful sense -- but it is not a tie either
until an interval says so, and this project has a specific scar there: the retracted "parity with RecVAE"
claim came from an UNPAIRED interval whose 1.96*sqrt(2)*SE signature made a real gap look like a wash.
Every claim of the form "these two are the same" needs a PAIRED per-user bootstrap on this ruler.

Re-creates the two selected arms exactly as that run selected them -- one frozen basis at the re-selected
setting, ls_reg as chosen on validation for EACH arm separately -- collects per-user NDCG, and reports the
paired difference.

  B_likes  constant target, likes-only support,  ls_reg 100.0   (the binary build)
  G_raw    raw ratings,     ALL-RATED support,   ls_reg   1.0   (RBMF's seed rule, native input)

Also reports the fixed-support contrast G_raw - B_allrated, the +0.0454 that carries the mechanism.

  python src/baselines/rbmf_graded_paired.py
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
import ials
from arm_n import load_arm_n
from run_rbmf_graded_reselected import make_seed, FACTORS, ITERS, BATCH, BASIS_REG, BASIS_ALPHA

OUTDIR = os.path.join(_ROOT, "experiments", "baselines", "arm_n")
OUTP = os.path.join(OUTDIR, "rbmf_graded_paired.json")


def logln(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(os.path.join(OUTDIR, "rbmf_graded_paired.log"), "a") as f:
        f.write(line + "\n")


def boot_ci(d, n_boot=10000, seed=98765):
    rng = np.random.RandomState(seed)
    idx = rng.randint(0, len(d), size=(n_boot, len(d)))
    m = d[idx].mean(axis=1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def main():
    argparse.ArgumentParser().parse_args()
    D = load_arm_n(log=logln)
    ni, hm, pool, te_te = D["n_items"], D["head_mask"], D["pool"], D["te_te"]

    base = ials._fit_model(D["train"], FACTORS, BASIS_REG, BASIS_ALPHA, ITERS, log=logln)
    Y = np.asarray(base.item_factors, dtype=np.float32)

    arms = {}
    for tag, mode, ls, te_in in (("B_likes", "const", 100.0, D["te_tr"]),
                                 ("B_allrated", "const", 100.0, D["g_te_tr"]),
                                 ("G_raw", "raw", 1.0, D["g_te_tr"])):
        r = M.evaluate(make_seed(Y, ls, mode), te_in, te_te, batch_size=BATCH, head_mask=hm,
                       mask_X=pool, per_user=True)
        arms[tag] = {"full": np.asarray(r["ndcg@10_per_user"], float),
                     "tail": np.asarray(r["tail_ndcg@10_per_user"], float),
                     "mean_full": r["ndcg@10"], "mean_tail": r["tail_ndcg@10"], "ls_reg": ls}
        logln(f"[{tag}] ls_reg={ls} full={r['ndcg@10']:.4f} tail={r['tail_ndcg@10']:.4f}")

    out = {"arms": {k: {kk: v[kk] for kk in ("mean_full", "mean_tail", "ls_reg")}
                    for k, v in arms.items()}, "contrasts": {}}
    for label, A, B in (("graded vs binary (contract)", "G_raw", "B_likes"),
                        ("values at fixed support", "G_raw", "B_allrated")):
        cell = {}
        for axis in ("full", "tail"):
            a, b = arms[A][axis], arms[B][axis]
            keep = np.isfinite(a) & np.isfinite(b)
            d = a[keep] - b[keep]
            lo, hi = boot_ci(d)
            sep = bool(lo > 0 or hi < 0)
            cell[axis] = {"n": int(keep.sum()), "delta": float(d.mean()), "ci": [lo, hi],
                          "separated": sep}
            logln(f"  {label:30s} {axis:4s} {d.mean():+.4f} CI [{lo:+.4f},{hi:+.4f}] "
                  f"{'SEPARATED' if sep else 'not separated'} (n={int(keep.sum())})")
        out["contrasts"][label] = cell

    c = out["contrasts"]["graded vs binary (contract)"]
    logln("=" * 78)
    if not c["full"]["separated"] and not c["tail"]["separated"]:
        logln("[verdict] neither axis separates: RBMF's graded and binary builds are INDISTINGUISHABLE "
              "on this ruler once configured. The loss on record was configuration and centring.")
    else:
        logln("[verdict] at least one axis separates -- state the direction per axis, never as a tie.")
    logln("=" * 78)
    out["indistinguishable"] = bool(not c["full"]["separated"] and not c["tail"]["separated"])
    json.dump(out, open(OUTP, "w"), indent=2)
    logln(f"done -> {OUTP}")


if __name__ == "__main__":
    main()
