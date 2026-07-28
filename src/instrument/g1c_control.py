r"""g1c_control.py -- THE MISSING CONTROL for G1c.

G1c asks: does the belief's stated uncertainty predict how wrong the model actually is?
Measured as Spearman(sqrt(w' Sigma w), held-item NLL) = 0.2393 on the fitted alphas.

The control nobody ran: **how well does a TRIVIAL scalar do on the same target?** If simply counting
a user's answers (or reading off how popular their held-out items are) predicts NLL just as well, then
the covariance is adding nothing over bookkeeping and G1c is not worth improving. If the trivial
proxies score far lower, then 0.2393 is a real (if modest) result.

Mirrors the G1c computation in run_battery_phaseB.py verbatim (same rows, same k=8 item prefix, same
seed, same held-out NLL) and swaps only the predictor. No truncation: all eligible rows, as G1 used.

  python src/instrument/g1c_control.py [--snapshot PATH] [--belief_ckpt PATH]
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "6")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "6")
os.environ.setdefault("MKL_NUM_THREADS", "6")
import sys
import json
import time
import argparse
import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
sys.path.insert(0, os.path.join(_ROOT, "src", "baselines"))
sys.path.insert(0, _HERE)

from train_tower_t2 import log
import run_battery_phaseA as PA
from run_battery_phaseA import build_real_ctx, SEED, spearman
from run_battery_phaseB import attach_belief, sigma_from_items
from belief_layer import fold_items

OUT = os.path.join(_ROOT, "experiments", "battery", "g1c_control.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot", default=PA.SNAP_DEFAULT)
    ap.add_argument("--belief_ckpt", default=os.path.join(_ROOT, ".cache", "instrument", "belief_i25.pt"))
    args = ap.parse_args()

    t0 = time.time()
    ctx = build_real_ctx(args.snapshot)
    attach_belief(ctx, args.belief_ckpt)
    log(f"[g1c-control] belief_fitted={getattr(ctx, 'belief_fitted', None)}")
    rows = [r for r in range(ctx.n) if ctx.va_te[r].nnz > 0]
    log(f"[g1c-control] {len(rows)} rows (all eligible; no subsample)")

    Wd, bd = ctx.Wd, ctx.bd
    cnt = np.asarray(ctx.cnt, np.float64)                      # train popularity per item
    bdn = bd.numpy().astype(np.float64)                        # learned decoder bias per item

    belief_sigma, err = [], []
    p_nfold, p_profile, p_nheld, p_pop, p_bias = [], [], [], [], []

    for r in rows:
        s = np.asarray(ctx.allb[r][0], np.int64)
        l = np.asarray(ctx.allb[r][1], np.int64)
        n = min(8, len(s))
        sel = np.random.default_rng(SEED + r).choice(len(s), n, replace=False)
        mu = fold_items(ctx.enc, [(s[sel], l[sel])])[0]
        ops = sigma_from_items(ctx, s[sel])
        hl = ctx.va_te[r].indices
        w = Wd[torch.from_numpy(hl)].mean(0)
        w = (w / w.norm().clamp_min(1e-8)).unsqueeze(0)

        belief_sigma.append(float(torch.sqrt(ops.quad_user(0, w)[0].clamp_min(1e-12))))
        logsm = torch.log_softmax(mu @ Wd.T + bd, -1)
        err.append(float(-logsm[torch.from_numpy(hl)].mean()))

        # --- trivial proxies: bookkeeping a system could read off without any covariance ---
        p_nfold.append(float(n))                               # answers actually folded
        p_profile.append(float(len(s)))                        # how much we know about the user
        p_nheld.append(float(len(hl)))                         # how many targets we must hit
        p_pop.append(float(np.log1p(cnt[hl]).mean()))          # how popular the targets are
        p_bias.append(float(bdn[hl].mean()))                   # decoder's own prior on the targets

    res = {
        "gate": "G1c-control", "n_users": len(rows),
        "belief_fitted": bool(getattr(ctx, "belief_fitted", False)),
        "rho": {
            "belief_sigma": round(spearman(belief_sigma, err), 4),
            "n_answers_folded": round(spearman(p_nfold, err), 4),
            "profile_size": round(spearman(p_profile, err), 4),
            "n_held_targets": round(spearman(p_nheld, err), 4),
            "mean_log_pop_of_targets": round(spearman(p_pop, err), 4),
            "mean_decoder_bias_of_targets": round(spearman(p_bias, err), 4),
        },
        "seconds": round(time.time() - t0, 1),
    }
    # n_answers_folded is constant by construction (k=8 prefix for every user) -> rho is undefined;
    # exclude NaNs rather than let max() propagate one.
    best_triv = max((abs(v), k) for k, v in res["rho"].items()
                    if k != "belief_sigma" and v == v)
    res["best_trivial_proxy"] = {"name": best_triv[1], "abs_rho": round(best_triv[0], 4)}
    res["belief_beats_best_trivial"] = bool(abs(res["rho"]["belief_sigma"]) > best_triv[0])
    res["verdict"] = ("belief covariance carries information beyond bookkeeping"
                      if res["belief_beats_best_trivial"] else
                      "a trivial scalar predicts held-out error at least as well -- G1c is not "
                      "measuring the covariance's contribution and should be dropped, not improved")

    json.dump(res, open(OUT, "w"), indent=1)
    log(f"[g1c-control] {json.dumps(res['rho'])}")
    log(f"[g1c-control] best trivial = {res['best_trivial_proxy']} | "
        f"belief_beats_trivial={res['belief_beats_best_trivial']}")
    log(f"[g1c-control] VERDICT: {res['verdict']}")
    log(f"[g1c-control] -> {OUT}")


if __name__ == "__main__":
    main()
