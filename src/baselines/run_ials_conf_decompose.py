r"""run_ials_conf_decompose.py -- what actually produced iALS-confidence's +0.045? Three variables, one at a time.

THE PROBLEM WITH THE HEADLINE NUMBER. run_native_ials_conf.py returned .3495/.2566 against binary iALS's
.3044/.2293 and printed "grades help". That contrast is not admissible as it stands, because THREE things
differ between the two cells, not one:

  (a) VALUES    binary indicator          vs  c = 1 + alpha*r
  (b) SUPPORT   likes-only fold-in        vs  every rating   (625,946 vs 1,411,761 nnz -- 2.26x)
  (c) REG       reg=0.1 (binary's tuned)  vs  reg=0.01 (the confidence script's default)

(b) is the dangerous one. arm_n.py's own header names it: the graded fold-in is 2.26x richer, and that
richness is what produced the void "Golbandi 0.3894" when it leaked into the mask. Here it does not touch
the mask -- the pool is protocol-level and identical -- but it still means the graded arm reads more than
twice as much evidence per user. More evidence beats less evidence in any collaborative model, so +0.045
is fully explainable without grades carrying any information at all.

THE LADDER. Each rung changes exactly one thing from the rung above:

  rung 0  binary, likes-only,  reg 0.1    = .3044/.2293   (on record, published binary iALS)
  rung 1  binary, likes-only,  reg 0.01   -> isolates (c), the regularisation I changed by accident
  rung 2  binary, ALL-rated,   reg 0.01   -> isolates (b), support size, the big one
  rung 3  graded, ALL-rated,   reg 0.01   = .3495/.2566   (on record, iALS-confidence)

The quantity the chapter needs is rung3 - rung2: the value of the GRADES, holding support and reg fixed.
rung2 - rung1 is the value of hearing about dislikes at all, which is a different and also interesting
claim, but it is NOT a claim about graded input and must not be reported as one.

alpha is tuned on validation NDCG@10 separately inside every rung, because the confidence scale means
different things at different supports and inheriting it across a contract change is the exact failure
that has already bitten this experiment three times (unseeded ALS basis; OPENBLAS after numpy; beta
across a 680x loss rescale).

  python src/baselines/run_ials_conf_decompose.py
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
from graded_data import load_graded_val

OUTDIR = os.path.join(_ROOT, "experiments", "baselines", "arm_n")
ALPHA_GRID = [2.5, 5.0, 10.0, 25.0, 50.0, 100.0]


def logln(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    os.makedirs(OUTDIR, exist_ok=True)
    with open(os.path.join(OUTDIR, "ials_conf_decompose.log"), "a") as f:
        f.write(line + "\n")


def binarise(X):
    """Indicator on exactly the same support -- values dropped, who-rated-what preserved."""
    B = X.tocsr().copy().astype(np.float32)
    B.data[:] = 1.0
    return B


def rung(name, tr, va_in, te_in, reg, factors, iters, D, va_te, alphas=ALPHA_GRID):
    ni, hm, pool, te_te = D["n_items"], D["head_mask"], D["pool"], D["te_te"]
    best = (None, -1.0, None)
    for alpha in alphas:
        model = ials._fit_model(tr, factors, reg, alpha, iters, log=lambda *_: None)
        prd = ials._make_predict(model, reg, alpha)
        m = M.evaluate(prd, va_in, va_te, batch_size=500, head_mask=hm)
        logln(f"  [{name}] alpha={alpha:6.1f} val full={m['ndcg@10']:.4f} tail={m['tail_ndcg@10']:.4f}")
        if m["ndcg@10"] > best[1]:
            best = (alpha, m["ndcg@10"], prd)
    alpha, val, prd = best
    edge = alpha in (alphas[0], alphas[-1])
    r = M.evaluate(prd, te_in, te_te, batch_size=500, head_mask=hm, mask_X=pool)
    logln(f"[RUNG {name}] alpha*={alpha} (val {val:.4f}){'  *** ALPHA AT GRID EDGE ***' if edge else ''} "
          f"-> TEST full={r['ndcg@10']:.4f} tail={r['tail_ndcg@10']:.4f}")
    return {"alpha": alpha, "val_ndcg@10": val, "alpha_at_edge": bool(edge),
            "ndcg@10": r["ndcg@10"], "tail_ndcg@10": r["tail_ndcg@10"], "ndcg@100": r["ndcg@100"],
            "reg": reg, "train_nnz": int(tr.nnz), "foldin_nnz": int(te_in.nnz)}


def main():
    ap = argparse.ArgumentParser()
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

    va_tr_bin, va_te = M.load_val(ni, os.path.join(_ROOT, "data", "ml-25m", "proc"))
    g_va = load_graded_val(va_te, log=logln)

    b_train, b_te_tr = D["train"], D["te_tr"]                       # binary, likes-only
    ga_train, ga_te_tr = binarise(D["g_train"]), binarise(D["g_te_tr"])   # binary, ALL-rated
    logln(f"[support] likes-only train nnz={b_train.nnz} foldin={b_te_tr.nnz} | "
          f"all-rated train nnz={ga_train.nnz} foldin={ga_te_tr.nnz} "
          f"({ga_te_tr.nnz / max(b_te_tr.nnz, 1):.2f}x)")
    gva_bin = binarise(g_va)

    R = {}
    logln("--- rung 1: binary, likes-only, reg 0.01  (isolates the regularisation I changed) ---")
    R["r1_binary_likes_reg001"] = rung("r1", b_train, va_tr_bin, b_te_tr, 0.01, a0.factors, a0.iterations,
                                       D, va_te)
    logln("--- rung 2: binary, ALL-rated, reg 0.01   (isolates SUPPORT SIZE -- the decisive one) ---")
    R["r2_binary_allrated_reg001"] = rung("r2", ga_train, gva_bin, ga_te_tr, 0.01, a0.factors,
                                          a0.iterations, D, va_te)

    r0 = json.load(open(os.path.join(OUTDIR, "ials.json")))["N"]
    r3 = json.load(open(os.path.join(OUTDIR, "ials_confidence_armn.json")))
    R["r0_binary_likes_reg01"] = {"ndcg@10": r0["ndcg@10"], "tail_ndcg@10": r0["tail_ndcg@10"],
                                  "reg": 0.1, "alpha": 50.0, "source": "ials.json (on record)"}
    R["r3_graded_allrated_reg001"] = {"ndcg@10": r3["ndcg@10"], "tail_ndcg@10": r3["tail_ndcg@10"],
                                      "reg": 0.01, "alpha": r3["alpha"],
                                      "source": "ials_confidence_armn.json (on record)"}

    f = lambda k: R[k]["ndcg@10"]
    t = lambda k: R[k]["tail_ndcg@10"]
    d_reg = f("r1_binary_likes_reg001") - f("r0_binary_likes_reg01")
    d_sup = f("r2_binary_allrated_reg001") - f("r1_binary_likes_reg001")
    d_val = f("r3_graded_allrated_reg001") - f("r2_binary_allrated_reg001")
    logln("=" * 78)
    logln(f"[decompose] total .3044 -> .3495 = {f('r3_graded_allrated_reg001') - f('r0_binary_likes_reg01'):+.4f} full")
    logln(f"[decompose]   regularisation  {d_reg:+.4f} full  {t('r1_binary_likes_reg001') - t('r0_binary_likes_reg01'):+.4f} tail")
    logln(f"[decompose]   SUPPORT (likes -> all rated, still binary)  {d_sup:+.4f} full  "
          f"{t('r2_binary_allrated_reg001') - t('r1_binary_likes_reg001'):+.4f} tail")
    logln(f"[decompose]   GRADES  (values on identical support)       {d_val:+.4f} full  "
          f"{t('r3_graded_allrated_reg001') - t('r2_binary_allrated_reg001'):+.4f} tail")
    logln("=" * 78)
    share = abs(d_val) / max(abs(d_val) + abs(d_sup) + abs(d_reg), 1e-9)
    logln(f"[verdict] grades account for {share:.0%} of the movement. " + (
        "The grades carry real signal on fixed support -- the chapter's claim that no ratings-native "
        "baseline benefits from its grades is WRONG and must be rewritten." if d_val > 0.005 else
        "The apparent graded win is SUPPORT SIZE, not gradedness -- hearing about dislikes at all is what "
        "pays, and the values on top of that support buy ~nothing. Report it as a support effect."))

    out = {"question": "does iALS-confidence's +0.045 come from the grades, the support, or the reg?",
           "rungs": R, "deltas": {"reg": d_reg, "support": d_sup, "grades": d_val},
           "gates": {"mostpop": bool(ok)}}
    p = os.path.join(OUTDIR, "ials_conf_decompose.json")
    json.dump(out, open(p, "w"), indent=2)
    logln(f"done -> {p}")


if __name__ == "__main__":
    main()
