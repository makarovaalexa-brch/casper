r"""run_refit_shared_mf.py -- refit RBMF and belief-MF under the same selection rule as iALS.

WHY. iALS, RBMF and belief-MF all carried {factors 200, reg 0.1, alpha 50}, inherited from a real 9-point
validation sweep run on ML-20M in July that selected on NDCG@100. Re-running that grid on THIS ruler and
selecting on the PRIMARY metric, NDCG@10, picks alpha 10 instead of alpha 50 and moves iALS by +0.0275
(arm A) and +0.0281 (arm N). RBMF and belief-MF sit on the same setting, so leaving them would put one
re-selected row beside two that are not -- which is worse than leaving all three alone, because the
inconsistency is invisible in the table.

NOTE ON WHAT WAS ACTUALLY MIS-SET, since I got this wrong once already: `reg` is unchanged at 0.1. The
mis-set knob is ALPHA, the confidence scale, 50 -> 10. My earlier ladder called that rung
"regularisation" because it moved reg and alpha together and I attributed the gain to reg. It is alpha.

RULE, stated once and applied to all three: sweep on this ruler's validation users, select on validation
NDCG@10, report both arms. alpha is NOT inherited from iALS's answer -- each model sweeps its own, because
"they all share one setting" was the problem, not the solution.

Scope of the grid here, stated rather than silently narrowed: reg is held at 0.1 because iALS's full
9-point sweep on this ruler preferred it under BOTH criteria and 0.001/0.01 lost at every alpha; alpha is
swept over its two contenders {10, 50}. Crucially, each model's SECOND knob is swept too -- RBMF's ls_reg
and belief-MF's obs_noise are read by fit() separately from the basis hyperparameters, and a first draft
of this script swept only the basis and would have left those two at their defaults, reproducing the very
mistake it exists to correct.

GATE: MostPop must reproduce its arm-N row through this eval path before anything is fitted.

  python src/baselines/run_refit_shared_mf.py
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
import rbmf_seed
import belief_mf
from arm_n import load_arm_n

OUTDIR = os.path.join(_ROOT, "experiments", "baselines", "arm_n")
OUTP = os.path.join(OUTDIR, "refit_shared_mf.json")
# reg is fixed at 0.1: iALS's full 9-point sweep on this ruler preferred it under BOTH criteria, and
# 0.001/0.01 were worse at every alpha. alpha is the knob that was actually mis-set (50 -> 10), so the
# two contenders are swept here against each model's OWN second knob -- which fit() reads separately and
# which my first draft of this script left at its default while sweeping parameters it does pass through.
REG_GRID = [0.1]
ALPHA_GRID = [10.0, 50.0]
# each model's own knob: RBMF's ridge on the per-user LS seed, belief-MF's observation noise
KNOB = {"rbmf": ("ls_reg", [0.1, 1.0, 10.0]), "belief_mf": ("obs_noise", [0.1, 1.0, 10.0])}
FACTORS, ITERS, BATCH = 200, 15, 250
STATE = {"selection_rule": "validation NDCG@10 on this ruler; per-model sweep; alpha NOT inherited",
         "old_shared_setting": {"factors": 200, "reg": 0.1, "alpha": 50.0}}


def logln(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    os.makedirs(OUTDIR, exist_ok=True)
    with open(os.path.join(OUTDIR, "refit_shared_mf.log"), "a") as f:
        f.write(line + "\n")


def persist():
    json.dump(STATE, open(OUTP, "w"), indent=2)


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

    for name, mod, oldfile in (("rbmf", rbmf_seed, "rbmf_seed.json"),
                               ("belief_mf", belief_mf, "belief_mf.json")):
        logln(f"--- {name}: sweeping basis alpha and the model's own knob, selecting on val NDCG@10 ---")
        STATE.setdefault("sweeps", {})[name] = []
        knob_name, knob_grid = KNOB[name]
        best = (None, None, None, -1.0)
        for reg in REG_GRID:
            for alpha in ALPHA_GRID:
                for kv in knob_grid:
                    t0 = time.time()
                    a = argparse.Namespace(factors=FACTORS, reg=reg, alpha=alpha, iterations=ITERS,
                                           **{knob_name: kv})
                    prd = mod.fit(D["train"], ni, args=a, log=lambda *_: None)
                    v = M.evaluate(prd, va_tr, va_te, batch_size=BATCH, head_mask=hm)
                    logln(f"  [{name}] alpha={alpha:5.1f} {knob_name}={kv:<6} "
                          f"val @10={v['ndcg@10']:.4f} @100={v['ndcg@100']:.4f} "
                          f"({(time.time()-t0)/60:.1f}m)")
                    STATE["sweeps"][name].append({"reg": reg, "alpha": alpha, knob_name: kv,
                                                  "val_ndcg@10": v["ndcg@10"],
                                                  "val_ndcg@100": v["ndcg@100"]})
                    persist()
                    if v["ndcg@10"] > best[3]:
                        best = (reg, alpha, kv, v["ndcg@10"])
        reg, alpha, kv, val = best
        logln(f"[{name}] val-selected alpha={alpha} {knob_name}={kv} (val @10 {val:.4f})")

        a = argparse.Namespace(factors=FACTORS, reg=reg, alpha=alpha, iterations=ITERS,
                               **{knob_name: kv})
        prd = mod.fit(D["train"], ni, args=a, log=logln)
        rows = {}
        for arm in ("A", "N"):
            kw = dict(batch_size=BATCH, head_mask=hm)
            if arm == "N":
                kw["mask_X"] = pool
            r = M.evaluate(prd, D["te_tr"], te_te, **kw)
            rows[arm] = {"ndcg@10": r["ndcg@10"], "tail_ndcg@10": r["tail_ndcg@10"],
                         "ndcg@100": r["ndcg@100"]}
        old = json.load(open(os.path.join(OUTDIR, oldfile)))
        for arm in ("A", "N"):
            o = old.get(arm) or {}
            if o:
                logln(f"[{name} arm {arm}] {rows[arm]['ndcg@10']:.4f}/{rows[arm]['tail_ndcg@10']:.4f} "
                      f"(was {o['ndcg@10']:.4f}/{o['tail_ndcg@10']:.4f}, "
                      f"{rows[arm]['ndcg@10'] - o['ndcg@10']:+.4f} full)")
        STATE[name] = {"reg": reg, "alpha": alpha, knob_name: kv, "val_ndcg@10": val, "rows": rows,
                       "on_record": {a_: {k: (old.get(a_) or {}).get(k)
                                          for k in ("ndcg@10", "tail_ndcg@10")} for a_ in ("A", "N")}}
        persist()

    logln("done -> " + OUTP)


if __name__ == "__main__":
    main()
