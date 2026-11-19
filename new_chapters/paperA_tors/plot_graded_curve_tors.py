"""plot_graded_curve_tors.py -- the graded-vs-likes-only figure for the TORS paper, redrawn from the
banked result experiments/battery/premium_vs_k_t2i26c_ep31.json (the certified tower's curve; the same
file the chapter's figure was drawn from). No number is recomputed: panel (a) plots the two arms' full
NDCG@10, panel (b) the banked paired premium (graded - likes-only) with its 95% bootstrap interval.
Output: fig_graded_curve_tors.pdf in this directory.
"""
import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
INP = os.path.join(ROOT, "experiments", "battery", "premium_vs_k_t2i26c_ep31.json")
d = json.load(open(INP))
ks = ["2", "4", "8", "16", "full"]
x = np.arange(len(ks))
true = np.array([d["per_k"][k]["arms"]["true"]["full@10"] for k in ks])
likes = np.array([d["per_k"][k]["arms"]["likes"]["full@10"] for k in ks])
prem = np.array([d["per_k"][k]["deltas"]["graded_over_binary(true-likes)"]["mean"] for k in ks])
lo = np.array([d["per_k"][k]["deltas"]["graded_over_binary(true-likes)"]["ci95"][0] for k in ks])
hi = np.array([d["per_k"][k]["deltas"]["graded_over_binary(true-likes)"]["ci95"][1] for k in ks])

plt.rcParams.update({"font.size": 8, "axes.labelsize": 8, "legend.fontsize": 7,
                     "xtick.labelsize": 7, "ytick.labelsize": 7, "font.family": "serif"})
fig, (a, b) = plt.subplots(1, 2, figsize=(6.2, 2.35))
a.plot(x, true, marker="o", ms=3.5, lw=1.2, color="#1f4e79", label="graded (signed star levels)")
a.plot(x, likes, marker="s", ms=3.5, lw=1.2, color="#c55a11", ls="--", label="likes-only (sub-threshold dropped)")
a.set_xticks(x); a.set_xticklabels(ks); a.set_xlabel("interview length $k$ (answered items)")
a.set_ylabel("full NDCG@10"); a.grid(alpha=0.25); a.legend(loc="upper left", frameon=False)
a.set_title("(a) the two input encodings", fontsize=8)
b.axhline(0, color="gray", lw=0.8)
b.errorbar(x, prem, yerr=[prem - lo, hi - prem], fmt="o-", ms=3.5, lw=1.2, capsize=2.5, color="#1f4e79")
b.set_xticks(x); b.set_xticklabels(ks); b.set_xlabel("interview length $k$ (answered items)")
b.set_ylabel("graded $-$ likes-only, full NDCG@10"); b.grid(alpha=0.25)
b.set_title("(b) paired premium, 95% bootstrap interval", fontsize=8)
for i, v in enumerate(prem):
    b.annotate(f"{v:+.4f}", (x[i], hi[i]), textcoords="offset points", xytext=(0, 3), ha="center", fontsize=6)
b.set_ylim(top=hi.max() + 0.0022)
fig.tight_layout()
fig.savefig(os.path.join(HERE, "fig_graded_curve_tors.pdf"))
fig.savefig(os.path.join(HERE, "fig_graded_curve_tors.png"), dpi=200)
print("premiums:", dict(zip(ks, prem.round(4))))
