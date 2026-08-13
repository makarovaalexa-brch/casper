r"""a9_render.py -- render the A9 deliverable tables from `a9_battery_policyB.json`.

Tabulation only: every number printed here comes from the analysis blob, none is recomputed.
"""
import os
import sys
import json

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
J = os.path.join(_ROOT, "experiments", "baselines", "interview", "a9_battery_policyB.json")
if len(sys.argv) > 1:                 # a labelled re-run (e.g. the TODO 1.3(b) EDDI rows)
    J = sys.argv[1] if os.path.isabs(sys.argv[1]) else os.path.join(
        _ROOT, "experiments", "baselines", "interview", sys.argv[1])
BUDGETS = (1, 2, 4, 8, 16)


def f4(x):
    """the chapter's own format: leading zero dropped, four decimals"""
    s = f"{x:.4f}"
    return s[1:] if s.startswith("0.") else s


def main():
    d = json.load(open(J))
    S = d["static_rows"]

    print("### tab:battery / tab:greedy -- the three rebuilt rows, test cohort\n")
    print("| arm | " + " | ".join(f"q={k}" for k in BUDGETS) + " | answered@16 |")
    print("|---|" + "---|" * (len(BUDGETS) + 1))
    for arm in ("items", "concepts", "combined"):
        cells = [f"{f4(S[arm][str(k)]['full@10'])}/{f4(S[arm][str(k)]['tail@10'])}" for k in BUDGETS]
        a = S[arm]["ans@16"]
        ans = (f"{a['items']:.2f} i" if a["concepts"] == 0 else
               (f"{a['concepts']:.2f} c" if a["items"] == 0 else
                f"{a['items']:.2f} i / {a['concepts']:.2f} c"))
        print(f"| {arm} | " + " | ".join(cells) + f" | {ans} |")

    print("\n### every selector minus each rebuilt static order\n")
    for comparator in ("combined", "items", "concepts"):
        print(f"\n#### vs greedy {comparator} (policy-B)\n")
        print("| selector | type | axis | " + " | ".join(f"k={k}" for k in BUDGETS) + " |")
        print("|---|---|---|" + "---|" * len(BUDGETS))
        for name, meta in d["selector_points"].items():
            for ax in ("full", "tail"):
                cells = []
                for k in BUDGETS:
                    lab = f"{name}|K{k}|{ax}|minus|{comparator}"
                    if lab not in d["vs_static"]:
                        cells.append("---")
                        continue
                    r = d["vs_static"][lab]
                    s = f"{r['delta']:+.4f} [{r['lo']:+.4f},{r['hi']:+.4f}]"
                    cells.append(f"**{s}**" if r["separates"] else s)
                print(f"| {name} | {meta['type']} | {ax} | " + " | ".join(cells) + " |")

    print("\n### narrative verification\n")
    for key, v in d["narrative"].items():
        print(f"* `{key}`: {v['static_leads_and_separates']}/{v['cells']} cells where the static "
              f"order leads AND the interval excludes zero "
              f"({v['static_leads']}/{v['cells']} lead on the point estimate) -- "
              f"**{'HOLDS' if v['claim_holds'] else 'BROKEN'}**")
        for e in v["exceptions"]:
            print(f"    * exception `{e['cell']}`: {e['delta']:+.5f} "
                  f"[{e['lo']:+.5f},{e['hi']:+.5f}] "
                  f"({'separates the other way' if e['separates'] else 'does not separate'})")


if __name__ == "__main__":
    main()
