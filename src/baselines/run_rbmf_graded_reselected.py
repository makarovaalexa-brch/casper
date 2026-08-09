r"""run_rbmf_graded_reselected.py -- does RBMF's graded arm still lose under the re-selected configuration?

WHY THIS IS OPEN. RBMF's graded arms on record (.2314 centred, .2950 uncentred) were fitted under the
superseded hyperparameters. Re-selecting on the reported metric moved RBMF's BINARY build from .3015 to
.3616, so the old pair is not a comparison any more and the chapter currently quotes no graded RBMF
number. Two facts make the question live rather than academic:

  * the mechanism test's E arm -- ridge least squares regressing the user's actual ratings on a frozen
    basis -- IS RBMF's published seed rule, and there the VALUES were worth +0.0454 over the same
    regression against a constant target;
  * most of RBMF's +0.0601 came from its OWN knob (ls_reg 1.0 -> 10.0), not the shared basis, and the
    graded arm never had that knob re-selected either.

So "RBMF is worse on its own ratings" may well be an artefact of configuration. This settles it.

FOUR ARMS on ONE frozen basis (the re-selected iALS: f=200, reg=0.1, alpha=10), ls_reg re-selected on
validation NDCG@10 SEPARATELY for each -- the whole point is that a knob tuned for one contract is not
valid for another:

  B_likes    constant target, likes-only support     the binary build; must reproduce ~.3616
  B_allrated constant target, ALL-RATED support      isolates the support change alone
  G_raw      raw ratings,     ALL-RATED support      RBMF's seed rule on its native input
  G_centred  ratings - 3.5,   ALL-RATED support      the published centring, 41.4% of targets negative

That gives both the contract-level comparison the NATIVE column asks for (G vs B_likes, where support and
values move together, as the two published contracts genuinely differ) and the fixed-support decomposition
(G vs B_allrated) that says which of the two did the work.

  python src/baselines/run_rbmf_graded_reselected.py
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
OUTP = os.path.join(OUTDIR, "rbmf_graded_reselected.json")
FACTORS, ITERS, BATCH = 200, 15, 250
BASIS_REG, BASIS_ALPHA = 0.1, 10.0
LSREG_GRID = [0.1, 1.0, 10.0, 100.0]
THRESH = 3.5
STATE = {}


def logln(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    os.makedirs(OUTDIR, exist_ok=True)
    with open(os.path.join(OUTDIR, "rbmf_graded_reselected.log"), "a") as f:
        f.write(line + "\n")


def persist():
    json.dump(STATE, open(OUTP, "w"), indent=2)


def make_seed(Y, ls_reg, mode):
    """RBMF's seed: ridge LS of a target on the frozen basis, over the SUPPORT items only.

    mode 'const'   -> target 1        (the binary build)
         'raw'     -> target r        (native ratings)
         'centred' -> target r - 3.5  (the published centring; negative for sub-threshold answers)
    """
    from scipy.linalg import cho_factor, cho_solve
    f = Y.shape[1]
    I = (ls_reg * np.eye(f)).astype(np.float32)
    YT = np.ascontiguousarray(Y.T)

    def predict(X_csr):
        X = X_csr.tocsr().astype(np.float32)
        out = np.zeros((X.shape[0], Y.shape[0]), dtype=np.float32)
        for u in range(X.shape[0]):
            lo, hi = X.indptr[u], X.indptr[u + 1]
            if lo == hi:
                continue
            idx = X.indices[lo:hi]
            v = X.data[lo:hi]
            t = (np.ones(hi - lo, dtype=np.float32) if mode == "const"
                 else (v if mode == "raw" else v - THRESH))
            Ys = Y[idx]
            G = Ys.T @ Ys
            G += I
            x = cho_solve(cho_factor(G, lower=True, check_finite=False), Ys.T @ t,
                          check_finite=False)
            out[u] = x @ YT
        return out
    return predict


def main():
    argparse.ArgumentParser().parse_args()
    D = load_arm_n(log=logln)
    ni, hm, pool, te_te = D["n_items"], D["head_mask"], D["pool"], D["te_te"]

    pr_pop = pop.fit(D["train"], ni, log=logln)
    rp = M.evaluate(pr_pop, D["te_tr"], te_te, batch_size=BATCH, head_mask=hm, mask_X=pool)
    canon = json.load(open(os.path.join(OUTDIR, "pop.json")))["N"]
    ok = abs(rp["ndcg@10"] - canon["ndcg@10"]) <= 1e-4
    logln(f"[G1] MostPop {rp['ndcg@10']:.4f} vs {canon['ndcg@10']:.4f} -> {'PASS' if ok else 'FAIL'}")
    if not ok:
        raise SystemExit("[G1] eval path does not reproduce MostPop")
    STATE["gates"] = {"mostpop": bool(ok)}
    persist()

    va_tr, va_te = M.load_val(ni, os.path.join(_ROOT, "data", "ml-25m", "proc"))
    g_va = load_graded_val(va_te, log=logln)

    logln(f"[basis] one frozen iALS basis at the RE-SELECTED setting (f={FACTORS}, reg={BASIS_REG}, "
          f"alpha={BASIS_ALPHA}); every arm below folds onto it")
    base = ials._fit_model(D["train"], FACTORS, BASIS_REG, BASIS_ALPHA, ITERS, log=logln)
    Y = np.asarray(base.item_factors, dtype=np.float32)
    STATE["basis"] = {"factors": FACTORS, "reg": BASIS_REG, "alpha": BASIS_ALPHA}
    persist()

    ARMS = (("B_likes",    "const",   va_tr, D["te_tr"],     "constant target, likes-only support"),
            ("B_allrated", "const",   g_va,  D["g_te_tr"],   "constant target, ALL-RATED support"),
            ("G_raw",      "raw",     g_va,  D["g_te_tr"],   "raw ratings, ALL-RATED support"),
            ("G_centred",  "centred", g_va,  D["g_te_tr"],   "ratings - 3.5, ALL-RATED support"))

    for tag, mode, va_in, te_in, note in ARMS:
        logln(f"--- {tag}: {note} ---")
        best = (None, -1.0, None)
        for ls in LSREG_GRID:
            prd = make_seed(Y, ls, mode)
            v = M.evaluate(prd, va_in, va_te, batch_size=BATCH, head_mask=hm)
            logln(f"  [{tag}] ls_reg={ls:6.1f} val full={v['ndcg@10']:.4f} tail={v['tail_ndcg@10']:.4f}")
            if v["ndcg@10"] > best[1]:
                best = (ls, v["ndcg@10"], prd)
        ls, val, prd = best
        r = M.evaluate(prd, te_in, te_te, batch_size=BATCH, head_mask=hm, mask_X=pool)
        logln(f"[{tag}] ls_reg={ls} -> TEST full={r['ndcg@10']:.4f} tail={r['tail_ndcg@10']:.4f}")
        STATE[tag] = {"ls_reg": ls, "val": val, "ndcg@10": r["ndcg@10"],
                      "tail_ndcg@10": r["tail_ndcg@10"], "note": note}
        persist()

    f = lambda k: STATE[k]["ndcg@10"]
    t = lambda k: STATE[k]["tail_ndcg@10"]
    best_g = "G_raw" if f("G_raw") >= f("G_centred") else "G_centred"
    logln("=" * 78)
    logln(f"[contract] best graded ({best_g}) minus binary likes-only: "
          f"{f(best_g) - f('B_likes'):+.4f} full, {t(best_g) - t('B_likes'):+.4f} tail")
    logln(f"[decompose] support alone (B_allrated - B_likes): {f('B_allrated') - f('B_likes'):+.4f} full")
    logln(f"[decompose] values at fixed support ({best_g} - B_allrated): "
          f"{f(best_g) - f('B_allrated'):+.4f} full")
    logln(f"[centring] raw minus centred: {f('G_raw') - f('G_centred'):+.4f} full")
    logln("[verdict] " + ("RBMF's graded arm WINS under the re-selected configuration -- the loss on "
                          "record was configuration, not the rating contract. The chapter must say so."
                          if f(best_g) > f("B_likes") else
                          "RBMF's graded arm still loses under the re-selected configuration -- the "
                          "finding survives re-tuning and can be stated as a property of the method."))
    logln("=" * 78)
    STATE["verdict"] = {"best_graded_arm": best_g,
                        "graded_minus_binary_full": f(best_g) - f("B_likes"),
                        "graded_minus_binary_tail": t(best_g) - t("B_likes"),
                        "support_alone": f("B_allrated") - f("B_likes"),
                        "values_at_fixed_support": f(best_g) - f("B_allrated"),
                        "graded_wins": bool(f(best_g) > f("B_likes"))}
    persist()
    logln(f"done -> {OUTP}")


if __name__ == "__main__":
    main()
