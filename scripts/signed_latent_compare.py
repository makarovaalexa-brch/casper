"""signed_latent_compare.py -- per-alpha selection table.

PRIMARY metric = held-liked NDCG@10 (full-catalog, arena, seed-avg {1,2,3,7,11}), with bootstrap CIs.
For EACH trained alpha checkpoint (best.pt), report:
  (A) full-profile strength NDCG@10 (full + tail),
  (B) Gate-3 dislike INPUT fold-in lift at k in {2,4,8}: held-liked NDCG with profile-half oracle
      dislikes fed as SIGNED-NEGATIVE input (add_dislikes=True) vs likes-only.
Selection is by strength + fold-in lift (NOT suppression/IG2). IG2 is secondary (battery).

Run AFTER training all alphas. Reuses signed_latent.py eval verbatim.
"""
import os, sys, json, time
import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import signed_latent as S
from signed_latent import (load_arena_base, build_splits, cohort, ndcg10, SignedAE, eval_model,
                           scale_rating, LO, HI, SEEDS, OUTDIR)


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


def boot_ci(vals, nb=2000, seed=0):
    v = np.asarray(vals, np.float64)
    if len(v) == 0:
        return (float("nan"), float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    means = v[rng.integers(0, len(v), size=(nb, len(v)))].mean(1)
    return float(v.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def collect(model, base, mode, k=None, add_dislikes=False):
    """Pool per-user held-liked NDCG@10 (full + tail) across seeds."""
    ni = base["ni"]; headmask = base["headmask"]
    allf, allt = [], []
    for si, seed in enumerate(SEEDS):
        SPL = build_splits(base, seed); users = cohort(base, SPL, "test")
        rng = np.random.default_rng(4242 + si)
        xs = []; metas = []
        for u in users:
            profset, held, prof_r, held_r = SPL[u]
            tlike = [j for j in held if held_r[j] >= LO]
            if not tlike:
                continue
            liked_prof = [j for j in prof_r if prof_r[j] >= LO]
            dis_prof = [j for j in prof_r if prof_r[j] <= HI]
            if mode == "full":
                idx = list(liked_prof)
            else:
                if len(liked_prof) < k:
                    continue
                sel = rng.choice(len(liked_prof), size=k, replace=False)
                idx = [liked_prof[i] for i in sel]
            xv = np.zeros(ni, np.float32)
            for j in idx:
                xv[j] = scale_rating(prof_r[j])
            if add_dislikes:
                for j in dis_prof:
                    xv[j] = scale_rating(prof_r[j])
            xs.append(xv); metas.append((profset, tlike))
        if not xs:
            continue
        with torch.no_grad():
            scores = model(torch.from_numpy(np.stack(xs)), 0.0)[0].numpy().astype(np.float64)
        for r, (profset, tlike) in enumerate(metas):
            nf = ndcg10(scores[r], tlike, profset, headmask, False)
            nt = ndcg10(scores[r], tlike, profset, headmask, True)
            if nf is not None: allf.append(nf)
            if nt is not None: allt.append(nt)
    return allf, allt


def main():
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument("--tags", required=True,
                    help="comma list, e.g. a0c,a03b,a01b,a10b")
    args = ap.parse_args()
    base = load_arena_base()
    S._POP = base["cnt"].astype(np.float64)
    ni = base["ni"]
    rows = {}
    for tag in args.tags.split(","):
        tag = tag.strip()
        best = os.path.join(OUTDIR, f"{tag}_best.pt")
        if not os.path.exists(best):
            log(f"SKIP {tag}: no {best}")
            continue
        blob = torch.load(best, map_location="cpu")
        model = SignedAE(ni, use_mask=True); model.load_state_dict(blob["model"]); model.eval()
        alpha = blob.get("alpha"); val_full = blob.get("val_full")
        # full-profile strength
        ff, tt = collect(model, base, "full")
        sf = boot_ci(ff); st = boot_ci(tt)
        row = {"alpha": alpha, "best_val_full": val_full,
               "full_profile_full": sf[0], "full_profile_full_ci": [sf[1], sf[2]],
               "full_profile_tail": st[0], "full_profile_tail_ci": [st[1], st[2]],
               "gate3_foldin": {}}
        log(f"[{tag}] alpha={alpha} full-profile full={sf[0]:.4f} [{sf[1]:.4f},{sf[2]:.4f}] tail={st[0]:.4f}")
        # gate3 fold-in lift
        for k in (2, 4, 8):
            l0, _ = collect(model, base, "k", k=k, add_dislikes=False)
            l1, _ = collect(model, base, "k", k=k, add_dislikes=True)
            m0 = boot_ci(l0); m1 = boot_ci(l1)
            lift = m1[0] - m0[0]
            row["gate3_foldin"][k] = {"likes_only": m0[0], "likes_only_ci": [m0[1], m0[2]],
                                      "plus_dislikes": m1[0], "plus_dislikes_ci": [m1[1], m1[2]],
                                      "lift": lift}
            log(f"    k={k} likes={m0[0]:.4f} +dis(input)={m1[0]:.4f} lift={lift:+.4f}")
        rows[tag] = row
    json.dump(rows, open(os.path.join(OUTDIR, "compare_alphas.json"), "w"), indent=2)
    log("wrote compare_alphas.json")


if __name__ == "__main__":
    main()
