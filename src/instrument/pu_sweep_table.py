r"""Tabulate the Part K temperature sweep. Reads only; computes nothing new.

One row per tau: the TARGET's own shape (entropy / effective support), then what the trained arm
achieved against the two things that make a number readable -- the uniform-prediction CE for that
same target (the free baseline) and the shuffled-target control at the SAME tau (the signal measure).

The verdict rule, stated before the table is read: the design is alive at a tau only if the main arm
beats BOTH its shuffled control AND the flat prediction, with a non-trivial top-1/top-5.
"""
import os
import sys
import glob
import json

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
PU_DIR = os.path.join(_ROOT, "experiments", "battery", "pu_labels")


def load(arm, tag):
    p = os.path.join(PU_DIR, f"pu_{arm}_{tag}_history.json")
    return json.load(open(p)) if os.path.exists(p) else None


# Effective support below this many candidates means the target has collapsed to a hard argmax on a
# single noisy per-user measurement -- the label form K.1 forbids by name, because argmax-of-noise is
# how the value head died. A number obtained there is not a result, it is the forbidden regime.
EFF_SUPPORT_FLOOR = 5.0


def main():
    taus = sys.argv[1:] or ["0.0001", "0.0003", "0.001", "0.003", "0.01", "0.03", "0.1", "0.3", "1.0"]
    rows = []
    for pre, lbl in (("t", ""), ("cold", " COLD/no-replay")):
        for t in taus:
            m, s = load("main", f"{pre}{t}"), load("shuffle", f"{pre}{t}")
            if not m:
                continue
            rows.append(_row(m, s, t, lbl))
    return _emit(rows)


def _row(m, s, t, lbl):
    if True:
        td = m["target_diagnostic"]
        bm = min(r["dev_soft_ce"] for r in m["history"])
        be = min(m["history"], key=lambda r: r["dev_soft_ce"])
        bs = min(r["dev_soft_ce"] for r in s["history"]) if s else float("nan")
        return {
            "lbl": lbl, "tau": float(t), "H": td["target_entropy_nats_mean"],
            "eff": td["target_eff_support_median"],
            "onehot": td["frac_effectively_one_hot_eff_lt_2"] * 100,
            "unif": td["frac_effectively_uniform_eff_gt_half_pool"] * 100,
            "uce": td["uniform_prediction_ce"], "main": bm, "shuf": bs,
            "gap": bs - bm, "flat_margin": td["uniform_prediction_ce"] - bm,
            "top1": be["dev_top1_vs_measured_argmax"] * 100,
            "top5": be["dev_top5_vs_measured_argmax"] * 100,
            "zero_top1": be["dev_zeroed_state_top1"] * 100,
            "ep": be["epoch"], "n_ep": len(m["history"])}


def _emit(rows):
    if not rows:
        raise SystemExit("no sweep artifacts found")
    print()
    print("PART K -- TEMPERATURE SWEEP on the banked 1,000-user labels (FIT-dev = 100 users)")
    print("grid chosen from the MEASURED per-user gain spread: median per-user std 0.0344,")
    print("median (max - median) 0.1710. tau spans ~0.1x to ~30x that std; 0.01 is the known-bad anchor.")
    print()
    hdr = (f"{'tau':>7} | {'H':>6} {'eff':>7} | {'flatCE':>7} {'MAIN':>7} {'SHUF':>7} | "
           f"{'main<flat':>9} {'shuf-main':>9} | {'top1%':>6} {'top5%':>6} {'zero%':>6} | "
           f"{'bestEp':>7} | regime")
    print(hdr); print("-" * len(hdr))
    last = None
    for r in rows:
        if r["lbl"] != last:
            print(f"--- {'WARM START + REPLAY (the pilot configuration)' if not r['lbl'] else 'COLD START, REPLAY-FREE (diagnostic: nothing forces peakedness)'} ---")
            last = r["lbl"]
        reg = "FORBIDDEN (argmax-of-noise)" if r["eff"] < EFF_SUPPORT_FLOOR else "ok"
        print(f"{r['tau']:>7g} | {r['H']:>6.3f} {r['eff']:>7.1f} | "
              f"{r['uce']:>7.4f} {r['main']:>7.4f} {r['shuf']:>7.4f} | {r['flat_margin']:>+9.4f} "
              f"{r['gap']:>+9.4f} | {r['top1']:>6.1f} {r['top5']:>6.1f} {r['zero_top1']:>6.1f} | "
              f"{r['ep']:>3}/{r['n_ep']:<3} | {reg}")
    print()
    print("  H     target entropy in nats (max = ln(n available) ~ 6.908; 0 = one-hot)")
    print("  eff   median effective support = exp(H), in candidates, of 1,000")
    print("  flatCE  CE of a CONSTANT prediction against that same target -- the free baseline")
    print("  main<flat  positive = the trained arm beats a flat prediction")
    print("  shuf-main  positive = the trained arm beats its shuffled-target control (the signal)")
    print("  zero%   top-1 with the state ZEROED -- if it equals top1%, the state is not being read")
    print()
    print(f"  regime  effective support < {EFF_SUPPORT_FLOOR:g} candidates = the target has collapsed to a hard")
    print("          argmax on ONE noisy per-user measurement, which K.1 forbids by name. Numbers")
    print("          obtained there are not results.")
    print()
    alive = [r for r in rows if r["flat_margin"] > 0 and r["gap"] > 0 and r["top5"] > 1.0
             and r["eff"] >= EFF_SUPPORT_FLOOR]
    if alive:
        b = max(alive, key=lambda r: r["gap"])
        print(f"  VERDICT: ALIVE at tau={b['tau']:g}{b['lbl']} -- beats flat by {b['flat_margin']:+.4f} "
              f"and its control by {b['gap']:+.4f}, top-5 {b['top5']:.1f}%")
    else:
        fm = [r for r in rows if r["flat_margin"] > 0]
        print("  VERDICT: NO tau clears both bars. " +
              (f"{len(fm)} tau beat the flat prediction but not the control (or top-5 is trivial)."
               if fm else "NO tau beats even a flat prediction."))
    return rows


if __name__ == "__main__":
    main()
