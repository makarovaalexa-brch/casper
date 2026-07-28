"""plot_g0_landscape.py -- measured full-profile NDCG@10 on our ML-25M Liang ruler (G0 landscape).

Reads the committed on-ruler JSONs in experiments/baselines/ml25m_liang/ and draws a grouped
horizontal bar chart of full-profile NDCG@10 (test, 10k users), tail@10 shown as a lighter
inset bar. Groups: accuracy-corner published baselines (snapped), belief-corner best-effort
reconstructions (not snapped), our frozen tower, and floors. Cold-intercept + RecVAE-bar
reference lines. Output: new_chapters/chapterA_v2/fig_g0_landscape.png (+ .pdf).
"""
import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
D = os.path.join(_ROOT, "experiments", "baselines", "ml25m_liang")
OUT = os.path.join(_ROOT, "new_chapters", "chapterA_v2", "fig_g0_landscape")

def load(name):
    j = json.load(open(os.path.join(D, name + ".json")))
    return j.get("ndcg@10") or j.get("G0_strength", {}).get("tower_full@10"), j.get("tail_ndcg@10")

# (label, file, group)  groups: acc=accuracy corner, bel=belief corner, our=ours, flo=floor
SPEC = [
    ("RecVAE (bar)",         "recvae",        "acc"),
    ("EASE",                 "ease",          "acc"),
    ("EDLAE",                "edlae",         "acc"),
    ("iALS",                 "ials",          "acc"),
    ("Item-kNN",             "itemknn",       "acc"),
    ("Golbandi node-rec",    "golbandi_node", "bel"),
    ("belief-MF (Bıyık/ConTS core)", "belief_mf", "bel"),
    ("Most-Popular",         "pop",           "flo"),
    ("Tower T2′ (ours)","tower_t2",      "our"),
]
COLD = 0.1279  # interview t=0 intercept
COLOR = {"acc": "#0a9396", "bel": "#bb3e03", "our": "#1b6ca8", "flo": "#adb5bd"}
LABELG = {"acc": "accuracy corner (published, snapped)", "bel": "belief corner (best-effort recon.)",
          "our": "our frozen tower", "flo": "floor"}

data = []
for label, f, grp in SPEC:
    full, tail = load(f)
    data.append((label, full, tail, grp))
data.sort(key=lambda r: r[1])  # ascending full@10

labels = [d[0] for d in data]
full = np.array([d[1] for d in data])
tail = np.array([d[2] for d in data])
cols = [COLOR[d[3]] for d in data]
y = np.arange(len(data))

fig, ax = plt.subplots(figsize=(7.2, 4.2))
ax.barh(y, full, color=cols, height=0.62, zorder=3, label=None)
ax.barh(y, tail, color="white", height=0.24, alpha=0.55, zorder=4)  # tail inset (lighter)
for i, (fv, tv) in enumerate(zip(full, tail)):
    ax.text(fv + 0.004, i, f"{fv:.3f}", va="center", fontsize=7.5, color="#333")
ax.axvline(COLD, color="#666", ls=":", lw=1.2, zorder=2)
ax.text(COLD, len(data) - 0.4, " cold intercept 0.128", fontsize=7, color="#666", ha="left")
recvae = dict(data and [(d[0], d[1]) for d in data]).get("RecVAE (bar)")
if recvae:
    ax.axvline(recvae, color="#0a9396", ls="--", lw=1.0, alpha=0.6, zorder=2)
ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=8)
ax.set_xlabel("full-profile NDCG@10 (ML-25M Liang ruler, test)")
ax.set_title("Measured full-profile accuracy on our ruler (white inset = tail@10)", fontsize=9.5)
ax.set_xlim(0, max(full) * 1.12)
ax.grid(True, axis="x", alpha=0.3, zorder=0)
handles = [plt.Rectangle((0, 0), 1, 1, color=COLOR[g]) for g in ["acc", "bel", "our", "flo"]]
ax.legend(handles, [LABELG[g] for g in ["acc", "bel", "our", "flo"]], fontsize=7, loc="lower right")
fig.tight_layout()
fig.savefig(OUT + ".png", dpi=160)
fig.savefig(OUT + ".pdf")
print("wrote", OUT + ".png / .pdf")
for label, fv, tv, grp in sorted(data, key=lambda r: -r[1]):
    print(f"  {label:<30} full={fv:.4f} tail={tv:.4f}")
