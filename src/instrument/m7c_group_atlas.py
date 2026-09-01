r"""M7c -- THE GROUP ATLAS: what the six shipped groups ARE, shown rather than asserted.

The paper has to answer "why these six groups?". `m7b_disjoint` reports their SIZES; this reports
their CONTENT and their GEOMETRY, which is what a reader needs to believe they are taste groups and
not an artefact of the construction.

WHAT IT PRODUCES
  1. a t-SNE of validation users in the recommender's own latent space, coloured by shipped group,
     with ungrouped users as grey background -- the visual claim that the partition tracks real
     structure rather than cutting a single blob into six
  2. per group: the highest-lift films drawn from that group's EXCLUSIVE pool, plus its exclusive
     vocabulary -- the exemplars that let a reader judge coherence for themselves
  3. a machine-readable json so the chapter table is generated, never retyped

WHY LATENT SPACE AND NOT RATINGS. The grouping only earns its place if it is visible to the
recommender, so the atlas is drawn in the space the policy actually reads. t-SNE is used for DISPLAY
ONLY -- no claim rests on it, and the separability numbers come from the linear probe (`m6x`), not
from how tidy this picture looks.

**THE TEST SET IS NEVER BUILT, LOADED OR NAMED IN THIS FILE.**
"""
from __future__ import annotations

import os
import sys

_H = os.path.dirname(os.path.abspath(__file__))
_R = os.path.abspath(os.path.join(_H, "..", ".."))
sys.path.insert(0, _H)

import argparse
import json
import time

import numpy as np

B = os.path.join(_R, "experiments", "battery")
M1 = os.path.join(B, "m1")
M6 = os.path.join(B, "m6")
FIG = os.path.join(_R, "docs", "figures")
SEED = 98765
NAMES = ["arthouse", "dialogue", "marvel", "gore", "teenagers", "inspiring"]
COLOURS = {"arthouse": "#4C72B0", "dialogue": "#937860", "marvel": "#C44E52",
           "gore": "#55A868", "teenagers": "#CCB974", "inspiring": "#8172B3"}


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def main():
    ap = argparse.ArgumentParser(description="M7c -- group atlas: figure + exemplars")
    ap.add_argument("--top_films", type=int, default=12)
    ap.add_argument("--perplexity", type=float, default=40.0)
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--i_have_author_goahead", action="store_true")
    a = ap.parse_args()
    assert a.i_have_author_goahead, "REFUSING TO RUN without --i_have_author_goahead"
    import torch
    torch.set_num_threads(a.threads)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE

    from strategy_channel_suite import build_greedy_env, CERT_TOWER, CERT_CONC
    from m3_show_paths import build_names
    from run_guard import require_mains
    cname, iname = build_names()
    require_mains("m7c group atlas", log=log)
    os.makedirs(FIG, exist_ok=True)

    D = np.load(os.path.join(M6, "m7b_disjoint.npz"))
    vr, vg, vh = D["val_rows"], D["val_group"], D["val_has"]
    meta = json.load(open(os.path.join(M6, "m7b_disjoint.json"), encoding="utf-8"))["groups"]

    zf = np.load(os.path.join(M1, "z_full.npz"))
    Z, ZR = zf["z"].astype(np.float64), zf["rows"]
    zpos = {int(r): i for i, r in enumerate(ZR)}
    keep = np.array([int(r) in zpos for r in vr])
    idx = np.array([zpos[int(r)] for r in vr[keep]])
    Zg, grp, has = Z[idx], vg[keep], vh[keep]
    log(f"[users] {len(Zg)} validation users with a banked latent; {int(has.sum())} in a group")

    E = build_greedy_env(CERT_TOWER, "validation", CERT_CONC, "m7c")
    Wd = np.asarray(E["ctx"].Wd.detach().numpy(), np.float64)
    gmean = Wd @ Zg.mean(0)

    # ---- exemplars: highest-lift films from each group's OWN exclusive pool --------------------
    out = {"analysis": "m7c_group_atlas", "test_touched": False, "groups": {}}
    log("=" * 104)
    for gi, g in enumerate(NAMES):
        m = has & (grp == gi)
        lift = Wd @ Zg[m].mean(0) - gmean
        pool = [int(f) for f in meta[g]["pool"]]
        top = sorted(pool, key=lambda f: -lift[f])[:a.top_films]
        out["groups"][g] = {"n_users": int(m.sum()), "pool_exclusive": len(pool),
                            "vocab_exclusive": meta[g]["vocab_exclusive"],
                            "top_films": [iname.get(f, str(f)) for f in top]}
        log(f"{g:<11} n={int(m.sum()):<5} pool={len(pool):<5} "
            f"vocab: {', '.join(meta[g]['vocab_exclusive'][:5])}")
        for t in out["groups"][g]["top_films"][:5]:
            log(f"            {t[:70]}")
    log("=" * 104)

    # ---- the picture ---------------------------------------------------------------------------
    log(f"[t-SNE] {len(Zg)} points, perplexity {a.perplexity} -- DISPLAY ONLY, no claim rests on it")
    xy = TSNE(n_components=2, perplexity=a.perplexity, init="pca", random_state=SEED,
              max_iter=1000).fit_transform(Zg)
    fig, ax = plt.subplots(figsize=(9, 8), dpi=170)
    ax.scatter(xy[~has, 0], xy[~has, 1], s=3, c="#DDDDDD", linewidths=0,
               label=f"no group ({int((~has).sum()):,})", rasterized=True)
    for gi, g in enumerate(NAMES):
        m = has & (grp == gi)
        ax.scatter(xy[m, 0], xy[m, 1], s=7, c=COLOURS[g], linewidths=0, alpha=0.75,
                   label=f"{g} ({int(m.sum()):,})", rasterized=True)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.legend(loc="upper right", frameon=False, markerscale=2.4, fontsize=9)
    ax.set_title("Validation users in the recommender's latent space,\n"
                 "coloured by shipped group (t-SNE, display only)", fontsize=11)
    fig.tight_layout()
    p_png = os.path.join(FIG, "m7c_group_atlas.png")
    fig.savefig(p_png); fig.savefig(p_png.replace(".png", ".pdf"))
    log(f"[fig] {p_png} (+ .pdf for the chapter)")

    json.dump(out, open(os.path.join(M6, "m7c_group_atlas.json"), "w"), indent=1)
    log("[done]")


if __name__ == "__main__":
    main()
