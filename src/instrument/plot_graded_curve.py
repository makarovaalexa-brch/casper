"""plot_graded_curve.py -- PROVISIONAL graded-vs-binarized interview-length chart for Chapter A.

Reads experiments/battery/premium_vs_k.json (from premium_vs_k.py). Two panels, full NDCG@10 vs
interview length k:
  (a) three input encodings of the SAME revealed set: graded (real signed star levels), likes-only
      (sub-threshold answers dropped -- what a positive-only tower e.g. RecVAE does at r>3.5), and
      all-as-like (every answered item recorded as a 4-star like -- naive implicit binarization).
  (b) the graded-over-binary premium (graded - likes-only) with bootstrap CI band and a zero line:
      stars help over the fair positive-only baseline MOST on the short interview, fading as
      collaborative signal fills in. (The larger graded - all-as-like premium is shown faint for
      context; it grows with k only because that control mislabels dislikes as likes.)
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
ks = [k for k in ["2", "4", "8", "16", "full"] if k in d["per_k"]]
x = np.arange(len(ks))
true = np.array([d["per_k"][k]["arms"]["true"]["full@10"] for k in ks])
likes = np.array([d["per_k"][k]["arms"]["likes"]["full@10"] for k in ks])
memb = np.array([d["per_k"][k]["arms"]["memb4"]["full@10"] for k in ks])

# graded-over-binary premium (true - likes) with CI, if present; else derive the mean
gob = d["per_k"]["2"]["deltas"].get("graded_over_binary(true-likes)")
prem = np.array([d["per_k"][k]["deltas"]["graded_over_binary(true-likes)"]["mean"]
                 if "graded_over_binary(true-likes)" in d["per_k"][k]["deltas"]
                 else true[i] - likes[i] for i, k in enumerate(ks)])
lo = np.array([d["per_k"][k]["deltas"].get("graded_over_binary(true-likes)", {}).get("ci95", [np.nan, np.nan])[0]
               for k in ks])
hi = np.array([d["per_k"][k]["deltas"].get("graded_over_binary(true-likes)", {}).get("ci95", [np.nan, np.nan])[1]
               for k in ks])
prem_naive = true - memb  # graded over naive all-as-like binarization

fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(8.6, 3.4))

# (a) NDCG curves
ax0.plot(x, true, "-o", color="#1b6ca8", lw=2, label="graded (signed star levels)")
ax0.plot(x, likes, "-s", color="#0a9396", lw=1.8, label="likes-only (positive-only)")
ax0.plot(x, memb, "-^", color="#bb3e03", lw=1.8, label="all-as-like (naive binarization)")
ax0.set_xticks(x); ax0.set_xticklabels(ks)
ax0.set_xlabel("interview length $k$"); ax0.set_ylabel("NDCG@10 (full)")
ax0.set_title("(a) input encodings", fontsize=9)
ax0.grid(True, alpha=0.3); ax0.legend(fontsize=7, loc="upper left", framealpha=0.9)

# (b) graded-over-binary premium with CI
ax1.axhline(0, color="#888", lw=0.8, ls="--")
if not np.isnan(lo).all():
    ax1.fill_between(x, lo, hi, color="#1b6ca8", alpha=0.18, label="95% CI")
ax1.plot(x, prem, "-o", color="#1b6ca8", lw=2, label="graded $-$ likes-only (fair binary)")
ax1.plot(x, prem_naive, ":", color="#bb3e03", lw=1.4, alpha=0.8,
         label="graded $-$ all-as-like (mislabels dislikes)")
ax1.set_xticks(x); ax1.set_xticklabels(ks)
ax1.set_xlabel("interview length $k$"); ax1.set_ylabel("$\\Delta$ NDCG@10 (full)")
ax1.set_title("(b) stars over binary: help most when $k$ small", fontsize=9)
ax1.grid(True, alpha=0.3); ax1.legend(fontsize=7, loc="upper left", framealpha=0.9)
ax1.annotate(f"+{prem[0]:.3f}", (x[0], prem[0]), textcoords="offset points", xytext=(4, 5),
             fontsize=7.5, color="#1b6ca8")

fig.tight_layout()
fig.savefig(OUT + ".png", dpi=160)
fig.savefig(OUT + ".pdf")
print("wrote", OUT + ".png / .pdf")
print("graded-over-binary premium (true-likes):",
      " ".join(f"k{k}={p:+.4f}" for k, p in zip(ks, prem)))
