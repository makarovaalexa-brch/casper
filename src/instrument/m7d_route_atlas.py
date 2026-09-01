r"""M7d -- THE TWO SPACES: why the groups look like a continuum and route like a partition.

`m7c` plotted validation users in the FULL-PROFILE latent and the groups overlapped heavily. That
picture is true and it is also the wrong space: the router never sees a user's full profile. It sees
the OPENER STATE -- the fold of what the user volunteered -- and a linear probe recovers the group
from that at 0.956.

So this draws both, side by side:

  LEFT   full-profile latent, t-SNE. Taste is a CONTINUUM; the six groups are a partition drawn
         across it, not six islands waiting to be found. This is the honest picture of the data.
  RIGHT  opener state, projected on the two leading discriminant directions. This is the space the
         policy reads, and in it the groups separate.

★ THE RIGHT PANEL IS NOT ALLOWED TO CHEAT. A discriminant fitted and plotted on the same users
would separate anything, so the projection is fitted on a DISJOINT HALF of the users and every point
drawn is from the held-out half. The separation shown is therefore earned on users the projection
never saw, and the caption carries the held-out accuracy to match.

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
PRETTY = {"arthouse": "art house", "dialogue": "dialogue driven", "marvel": "marvel",
          "gore": "gore", "teenagers": "teenagers", "inspiring": "inspiring"}
COLOURS = {"arthouse": "#4C72B0", "dialogue": "#937860", "marvel": "#C44E52",
           "gore": "#55A868", "teenagers": "#CCB974", "inspiring": "#8172B3"}


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def main():
    ap = argparse.ArgumentParser(description="M7d -- full-profile continuum vs router-space partition")
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
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

    from strategy_channel_suite import (build_greedy_env, genv_eval_multi, CERT_TOWER, CERT_CONC)
    from ndcg_tree import LIKE_MIN_LEVEL
    from m3_show_paths import build_names
    from opener_gen import centroid_lift, make_openers
    from run_guard import require_mains
    cname, _iname = build_names()
    require_mains("m7d route atlas", log=log)
    os.makedirs(FIG, exist_ok=True)

    D = np.load(os.path.join(M6, "m7b_disjoint.npz"))
    vr, vg, vh = D["val_rows"], D["val_group"], D["val_has"]
    own_i, own_c = D["own_item"], D["own_concept"]

    E = build_greedy_env(CERT_TOWER, "validation", CERT_CONC, "m7d")
    zf = np.load(os.path.join(M1, "z_full.npz"))
    ZF, ZR = zf["z"].astype(np.float64), zf["rows"]
    zpos = {int(r): i for i, r in enumerate(ZR)}

    # ---- opener states, per group, exactly as the probe and the policy see them -----------------
    Zo, lab, uid = [], [], []
    for gi, g in enumerate(NAMES):
        rows = vr[(vg == gi) & vh].astype(np.int64)
        # centroid from the group's own users, matching how `m6x` builds the states this figure
        # is about -- the probe's space, not m6d's evaluation path.
        tab = centroid_lift(E, rows, ZF, ZR)
        ops, _m = make_openers(E, rows, tab, cname, LIKE_MIN_LEVEL, seed=SEED, group_key=g,
                               disjoint=(own_i, own_c, gi))
        iseq = [[i for c, i in o if c == 0] for o in ops]
        cseq = [[i for c, i in o if c == 1] for o in ops]
        z = np.asarray(genv_eval_multi(E, rows, iseq, cseq, score=False, return_z=True)[-1],
                       np.float64)
        Zo.append(z); lab.append(np.full(len(rows), gi, np.int32)); uid.append(rows)
        log(f"[opener] {g:<11} {len(rows)} states")
    Zo = np.concatenate(Zo); lab = np.concatenate(lab); uid = np.concatenate(uid)

    # ---- full-profile latents for the SAME users ------------------------------------------------
    keep = np.array([int(r) in zpos for r in uid])
    Zfull = ZF[[zpos[int(r)] for r in uid[keep]]]
    lab_full = lab[keep]
    log(f"[users] {len(Zo)} opener states, {len(Zfull)} with a banked full-profile latent")

    # ---- fit the discriminant on ONE HALF, plot the OTHER ---------------------------------------
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(len(Zo))
    fit_i, hold_i = perm[: len(perm) // 2], perm[len(perm) // 2:]
    lda = LDA(n_components=2).fit(Zo[fit_i], lab[fit_i])
    acc = float((lda.predict(Zo[hold_i]) == lab[hold_i]).mean())
    P = lda.transform(Zo[hold_i])
    ph = lab[hold_i]
    log(f"[lda] fitted on {len(fit_i)} users, held-out accuracy on {len(hold_i)} = {acc:.3f} "
        f"(chance {1/len(NAMES):.3f}) -- the picture on the right is the HELD-OUT half only")

    log(f"[t-SNE] {len(Zfull)} full-profile points -- display only")
    xy = TSNE(n_components=2, perplexity=a.perplexity, init="pca", random_state=SEED,
              max_iter=1000).fit_transform(Zfull)

    fig, axes = plt.subplots(1, 2, figsize=(13.6, 6.4), dpi=170)
    for gi, g in enumerate(NAMES):
        m = lab_full == gi
        axes[0].scatter(xy[m, 0], xy[m, 1], s=6, c=COLOURS[g], linewidths=0, alpha=0.7,
                        rasterized=True, label=PRETTY[g])
    axes[0].set_title("full profile: taste is a continuum\n(t-SNE, all assigned users)", fontsize=11)
    for gi, g in enumerate(NAMES):
        m = ph == gi
        axes[1].scatter(P[m, 0], P[m, 1], s=7, c=COLOURS[g], linewidths=0, alpha=0.75,
                        rasterized=True, label=PRETTY[g])
    axes[1].set_title(f"opener state: the space the router reads\n"
                      f"(discriminant fitted on a disjoint half; held-out accuracy {acc:.3f})",
                      fontsize=11)
    for ax in axes:
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
    axes[1].legend(loc="best", frameon=False, markerscale=2.2, fontsize=9)
    fig.tight_layout()
    p_png = os.path.join(FIG, "m7d_route_atlas.png")
    fig.savefig(p_png); fig.savefig(p_png.replace(".png", ".pdf"))
    log(f"[fig] {p_png} (+ .pdf)")
    json.dump({"analysis": "m7d_route_atlas", "test_touched": False,
               "lda_heldout_accuracy": acc, "n_fit": int(len(fit_i)), "n_held": int(len(hold_i)),
               "chance": 1.0 / len(NAMES)},
              open(os.path.join(M6, "m7d_route_atlas.json"), "w"), indent=1)
    log("[done]")


if __name__ == "__main__":
    main()
