"""plot_graded_curve.py -- PROVISIONAL graded-vs-binarized interview-length chart for Chapter A.

Reads experiments/battery/premium_vs_k.json (from premium_vs_k.py) and plots full@10 NDCG vs
interview length k for three input encodings of the SAME revealed set:
  - graded (true): real star levels (signed, hated..loved)
  - likes-only:    sub-threshold answers dropped (what a positive-only tower, e.g. RecVAE, does)
  - all-as-like:   every answered item recorded as a 4-star like (naive implicit binarization)
The honest read: graded ~ likes-only (fine levels add little beyond the like/dislike sign, and only
on the short interview), and BOTH crush all-as-like (mislabeling dislikes as likes craters). The
graded channel's value is representing dislikes at all -- impossible in a positive-only basis.
Output: new_chapters/chapterA_v2/fig_graded_curve.png (+ .pdf).
"""
import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
INP = os.path.join(_ROOT, "experiments", "battery", "premium_vs_k.json")
OUT = os.path.join(_ROOT, "new_chapters", "chapterA_v2", "fig_graded_curve")

d = json.load(open(INP))
ks = ["2", "4", "8", "16", "full"]
x = np.arange(len(ks))
true = [d["per_k"][k]["arms"]["true"]["full@10"] for k in ks]
likes = [d["per_k"][k]["arms"]["likes"]["full@10"] for k in ks]
memb = [d["per_k"][k]["arms"]["memb4"]["full@10"] for k in ks]

fig, ax = plt.subplots(figsize=(5.2, 3.4))
ax.plot(x, true, "-o", color="#1b6ca8", lw=2, label="graded (signed star levels)")
ax.plot(x, likes, "-s", color="#0a9396", lw=1.8, label="likes-only (positive-only, drops dislikes)")
ax.plot(x, memb, "-^", color="#bb3e03", lw=1.8, label="all-as-like (naive binarization)")
ax.set_xticks(x); ax.set_xticklabels(ks)
ax.set_xlabel("interview length $k$ (revealed items)")
ax.set_ylabel("NDCG@10 (full)")
ax.set_title("Graded vs. binarized input, frozen tower", fontsize=10)
ax.grid(True, alpha=0.3)
ax.legend(fontsize=7.5, loc="upper left", framealpha=0.9)
# annotate the short-interview graded-over-likes premium
ax.annotate(f"+{true[0]-likes[0]:.3f} @k2", (x[0], true[0]), textcoords="offset points",
            xytext=(6, 6), fontsize=7, color="#1b6ca8")
fig.tight_layout()
fig.savefig(OUT + ".png", dpi=160)
fig.savefig(OUT + ".pdf")
print("wrote", OUT + ".png / .pdf")
print(f"graded-over-likes premium: " + " ".join(f"k{k}={t-l:+.4f}" for k, t, l in zip(ks, true, likes)))
