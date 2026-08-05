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
import sys
# default to the CERTIFIED tower's curve; pass a path to plot the pre-certification one instead
INP = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
    _ROOT, "experiments", "battery", "premium_vs_k_t2final.json")
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

fig, ax0 = plt.subplots(1, 1, figsize=(5.2, 3.4))

# Single panel: the three encodings as curves. The former panel (b) -- the graded-minus-
# likes-only premium -- was DROPPED 2026-08-04 (author): it is positive and CI-clean only
# at k=2 and k=4 and sits at ~0 for k>=8, so three of five plotted points were at or below
# zero and the chart read against the claim it illustrated. Two numbers belong in prose,
# not a panel. It also invited confusion with the separate full-profile A/B (+0.0039 on
# identical support), which is a DIFFERENT protocol from this harness's k=full point.
# (a) NDCG curves
ax0.plot(x, true, "-o", color="#1b6ca8", lw=2, label="graded (signed star levels)")
ax0.plot(x, likes, "-s", color="#0a9396", lw=1.8, label="likes-only (positive-only)")
ax0.set_xticks(x); ax0.set_xticklabels(ks)
ax0.set_xlabel("interview length $k$"); ax0.set_ylabel("NDCG@10 (full)")
ax0.set_title("input encodings", fontsize=9)   # no "(a)": panel (b) was dropped, this is single-panel
ax0.grid(True, alpha=0.3); ax0.legend(fontsize=7, loc="upper left", framealpha=0.9)

fig.tight_layout()
fig.savefig(OUT + ".png", dpi=160)
fig.savefig(OUT + ".pdf")
print("wrote", OUT + ".png / .pdf")
print("graded-over-binary premium (true-likes):",
      " ".join(f"k{k}={p:+.4f}" for k, p in zip(ks, prem)))
