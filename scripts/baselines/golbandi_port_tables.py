r"""golbandi_port_tables.py -- tables + PAIRED bootstrap CIs for TASK #13 (Golbandi tree, P and X).

Reads `adaptive_table_<label>.json` and its `_per_user.npz` sidecar and prints, per consumer:
  full/tail NDCG@10 at every budget, answered counts, and the SELECTOR FALLBACK RATE (a golbandi_tree
  row with a high fallback rate is part popularity and must be read that way), plus the
  adaptive-minus-static delta against each matched static comparator with a paired percentile
  bootstrap over users.

The bootstrap is PAIRED, resampling USERS and recomputing the per-user difference. The unpaired
`1.96*sqrt(2)*SE` form is banned: it overstated the true interval ~4.6x and produced a "tie" claim that
had to be retracted (ARM_N_PHASE1 sec 4b).

  python scripts/baselines/golbandi_port_tables.py --label gtport13 --boot 10000
"""
import os
import sys
import json
import argparse
import numpy as np

_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
OUT = os.path.join(_ROOT, "experiments", "baselines", "interview")


def boot_ci(d, n_boot, seed):
    rng = np.random.RandomState(seed)
    idx = rng.randint(0, len(d), size=(n_boot, len(d)))
    m = d[idx].mean(axis=1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", default="gtport13")
    ap.add_argument("--boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=98765)
    ap.add_argument("--adaptive", default="golbandi_tree")
    a = ap.parse_args()

    jp = os.path.join(OUT, f"adaptive_table_{a.label}.json")
    blob = json.load(open(jp))
    meta, res = blob["meta"], blob["results"]
    vp = jp.replace(".json", "_per_user.npz")
    V = np.load(vp) if os.path.exists(vp) else None

    print(f"# Golbandi tree on our ruler -- {a.label}")
    print(f"snapshot={meta.get('snapshot')} arch={meta.get('arch')} no_floor={meta.get('no_floor')} "
          f"n_test_users={meta.get('n_test_users')} golbandi={meta.get('golbandi')}")
    print(f"MostPop prior: full={res['mostpop']['_prior']['full']:.4f} "
          f"tail={res['mostpop']['_prior']['tail']:.4f}\n")

    budgets = meta["budgets"]
    for cons in [c for c in meta["consumers"] if c in res]:
        print(f"\n## consumer = {cons}")
        hdr = "| selector | " + " | ".join(f"k={k}" for k in budgets) + " | answered@max | fallback@max |"
        print(hdr)
        print("|" + "---|" * (len(budgets) + 3))
        for spec in meta["selectors"]:
            cells = res[cons].get(spec, {})
            row = []
            for k in budgets:
                c = cells.get(f"k{k}")
                row.append(f"{c['full']:.4f}/{c['tail']:.4f}" if c else "—")
            last = [cells[f"k{k}"] for k in budgets if f"k{k}" in cells]
            ans = f"{last[-1]['answered']:.2f}" if last else "—"
            fb = (f"{last[-1]['fallback_rate']:.1%}" if last and "fallback_rate" in last[-1]
                  else "n/a (static)")
            print(f"| {spec} | " + " | ".join(row) + f" | {ans} | {fb} |")

        if V is None:
            print("\n(no per-user sidecar: paired CIs OWED)")
            continue
        print(f"\n### {a.adaptive} minus each static comparator, paired bootstrap "
              f"({a.boot} resamples, seed {a.seed}); * = CI excludes zero")
        print("| comparator | axis | " + " | ".join(f"k={k}" for k in budgets) + " |")
        print("|" + "---|" * (len(budgets) + 2))
        for spec in meta["selectors"]:
            if spec == a.adaptive:
                continue
            for axis in ("full", "tail"):
                cells = []
                for k in budgets:
                    ka = f"{cons}|{a.adaptive}|k{k}|{axis}"
                    kb = f"{cons}|{spec}|k{k}|{axis}"
                    if ka not in V or kb not in V:
                        cells.append("—")
                        continue
                    x, y = np.asarray(V[ka], float), np.asarray(V[kb], float)
                    assert len(x) == len(y), f"pairing broken {ka} vs {kb}: {len(x)} vs {len(y)}"
                    d = x - y
                    lo, hi = boot_ci(d, a.boot, a.seed)
                    star = "*" if (lo > 0) == (hi > 0) else ""
                    cells.append(f"{d.mean():+.4f}{star} [{lo:+.4f},{hi:+.4f}]")
                print(f"| {spec} | {axis} | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()
