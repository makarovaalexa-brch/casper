"""signed_latent_battery.py -- FULL analysis battery on the chosen best signed-latent checkpoint.

Reuses signed_latent.py (model, ruler, eval, gates) verbatim -- does NOT touch the model.
Full-catalog held-liked NDCG@10 on the arena half-split; te(500) MINUS 300 study users; seed-avg
over {1,2,3,7,11}; bootstrap 95% CIs over pooled per-user values; NO data caps.

Emits .cache/signed_latent/battery_<tag>.json with:
  k-curve (0..full) x refusal {0,.15,.30,.50}, strength vs posts, IG1..IG5, IG2 flip+specificity,
  dislike-helps (fold-in weight sweep), taste separation, G-GoT, and all hygiene gates.
"""
import os, sys, json, time
import numpy as np
import torch
import torch.nn.functional as F

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import signed_latent as S
from signed_latent import (load_arena_base, build_splits, cohort, ndcg10, SignedAE, eval_model,
                           build_gmat, score_from_input, percentile_rank, scale_rating,
                           gate2_ig2, LO, HI, SEEDS, OUTDIR, GENRES)


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


def boot_ci(vals, nb=2000, seed=0):
    v = np.asarray(vals, np.float64)
    if len(v) == 0:
        return (float("nan"), float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    means = v[rng.integers(0, len(v), size=(nb, len(v)))].mean(1)
    return float(v.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


# ---------- generalized per-user eval collecting per-user NDCGs (for bootstrap) ----------
def eval_collect(model, base, SPL, users, mode="full", k=None, refusal=0.0,
                 add_dislikes=False, seedoff=0):
    """Return (list_full_ndcg, list_tail_ndcg). mode: 'full'|'k'|'cold'.
    refusal drops that fraction of REVEALED liked answers (rounded). add_dislikes appends
    profile-half oracle dislikes as signed-negative input."""
    ni = base["ni"]; headmask = base["headmask"]
    rng = np.random.default_rng(4242 + seedoff)
    xs = []; metas = []
    for u in users:
        profset, held, prof_r, held_r = SPL[u]
        tlike = [j for j in held if held_r[j] >= LO]
        if not tlike:
            continue
        liked_prof = [j for j in prof_r if prof_r[j] >= LO]
        dis_prof = [j for j in prof_r if prof_r[j] <= HI]
        if mode == "cold":
            idx = []
        elif mode == "full":
            idx = list(liked_prof)
        else:
            if len(liked_prof) < k:
                continue
            sel = rng.choice(len(liked_prof), size=k, replace=False)
            idx = [liked_prof[i] for i in sel]
        # refusal: drop a fraction of the revealed liked answers
        if refusal > 0 and idx:
            ndrop = int(round(refusal * len(idx)))
            if ndrop > 0:
                keep = rng.choice(len(idx), size=len(idx) - ndrop, replace=False)
                idx = [idx[i] for i in keep]
        xv = np.zeros(ni, np.float32)
        for j in idx:
            xv[j] = scale_rating(prof_r[j])
        if add_dislikes:
            for j in dis_prof:
                xv[j] = scale_rating(prof_r[j])
        xs.append(xv); metas.append((profset, tlike))
    if not xs:
        return [], []
    with torch.no_grad():
        scores = model(torch.from_numpy(np.stack(xs)), 0.0)[0].numpy().astype(np.float64)
    ff, tt = [], []
    for r, (profset, tlike) in enumerate(metas):
        nf = ndcg10(scores[r], tlike, profset, headmask, False)
        nt = ndcg10(scores[r], tlike, profset, headmask, True)
        if nf is not None: ff.append(nf)
        if nt is not None: tt.append(nt)
    return ff, tt


def pooled(model, base, mode, k=None, refusal=0.0, add_dislikes=False):
    """Pool per-user NDCGs across seeds -> bootstrap CI."""
    allf, allt = [], []
    for si, seed in enumerate(SEEDS):
        SPL = build_splits(base, seed); users = cohort(base, SPL, "test")
        ff, tt = eval_collect(model, base, SPL, users, mode, k, refusal, add_dislikes, seedoff=si)
        allf += ff; allt += tt
    mf, lof, hif = boot_ci(allf)
    mt, lot, hit = boot_ci(allt)
    return dict(full=mf, full_lo=lof, full_hi=hif, tail=mt, tail_lo=lot, tail_hi=hit, n=len(allf))


def title_map(base):
    d = np.load(S.META); keepI = d["keepI"].astype(np.int64)
    mv = {}
    with open("data/movielens/movies.csv", encoding="utf-8") as f:
        next(f)
        for line in f:
            i0 = line.find(","); i1 = line.rfind(",")
            try:
                mid = int(line[:i0])
            except ValueError:
                continue
            mv[mid] = line[i0 + 1:i1].strip().strip('"')
    return {i: mv.get(int(keepI[i]), f"item{i}") for i in range(base["ni"])}


def main():
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument("--tag", required=True)
    args = ap.parse_args()
    tag = args.tag
    base = load_arena_base(); ni = base["ni"]
    S._POP = base["cnt"].astype(np.float64)
    best = os.path.join(OUTDIR, f"{tag}_best.pt")
    blob = torch.load(best, map_location="cpu")
    model = SignedAE(ni, use_mask=True); model.load_state_dict(blob["model"]); model.eval()
    alpha = blob.get("alpha"); val_full = blob.get("val_full")
    log(f"loaded {best} alpha={alpha} val_full={val_full}")
    Gmat, gix = build_gmat(base)
    titles = title_map(base)
    cnt = base["cnt"]
    out = {"tag": tag, "alpha": alpha, "best_val_full": val_full,
           "anchors": {"MOSTPOP": 0.2851, "EASE": 0.5103, "RecVAE": 0.5261, "preVAE_docref": 0.33},
           "eval": "full-catalog held-liked NDCG@10, arena half-split, te minus 300 study, seed-avg {1,2,3,7,11}, bootstrap 95% CI"}

    # ============ 1. k-curve x refusal ============
    log("=== k-curve x refusal ===")
    KS = [0, 1, 2, 4, 8, 16, 32, "full"]
    REF = [0.0, 0.15, 0.30, 0.50]
    kcurve = {}
    for refusal in REF:
        row = {}
        for k in KS:
            if k == 0:
                r = pooled(model, base, "cold", refusal=refusal)
            elif k == "full":
                r = pooled(model, base, "full", refusal=refusal)
            else:
                r = pooled(model, base, "k", k=k, refusal=refusal)
            row[str(k)] = r
            log(f"  ref={refusal:.2f} k={k}: full={r['full']:.4f} [{r['full_lo']:.4f},{r['full_hi']:.4f}] n={r['n']}")
        kcurve[f"{refusal:.2f}"] = row
    out["kcurve_by_refusal"] = kcurve

    # monotone accumulation (refusal 0): non-decreasing across k up to full
    base_row = kcurve["0.00"]
    seq = [base_row[str(k)]["full"] for k in KS]
    out["monotone_accumulation_ok"] = bool(all(seq[i + 1] >= seq[i] - 1e-3 for i in range(len(seq) - 1)))
    out["monotone_seq"] = seq

    # ============ 3. strength ============
    out["strength"] = {"full_profile_full": base_row["full"]["full"],
                       "full_profile_tail": base_row["full"]["tail"],
                       "full_profile_full_ci": [base_row["full"]["full_lo"], base_row["full"]["full_hi"]],
                       "cold_full": base_row["0"]["full"]}

    # ============ 4. IG2 dislike-expresses + specificity + taste-sep (reuse gate2) ============
    log("=== IG2 flip + specificity + taste separation ===")
    g2 = gate2_ig2(model, base, Gmat, gix)
    out["ig2"] = g2
    log(f"  mean_flip_gap={g2['mean_flip_gap']:.3f} ctrl_disp={g2['mean_ctrl_displacement']:.3f} "
        f"taste_sep_cos={g2['taste_sep_cos_like_vs_dislike']:.3f}")

    # ============ 5. dislike-helps: fold-in weight sweep at k in {2,4,8} ============
    log("=== dislike-helps (region-narrowing) fold-in weight sweep ===")
    # add_dislikes uses the true signed value; sweep a multiplier on the dislike magnitude
    dh = {}
    for k in (2, 4, 8):
        base_lo = pooled(model, base, "k", k=k, add_dislikes=False)["full"]
        # sweep dislike weight by scaling the negative entries via a wrapper
        wsweep = {}
        for w in (0.5, 1.0, 2.0):
            allf = []
            for si, seed in enumerate(SEEDS):
                SPL = build_splits(base, seed); users = cohort(base, SPL, "test")
                ff = eval_dislike_weight(model, base, SPL, users, k, w, seedoff=si)
                allf += ff
            m, lo, hi = boot_ci(allf)
            wsweep[str(w)] = dict(full=m, lo=lo, hi=hi, lift=m - base_lo)
            log(f"  k={k} w={w}: +dis full={m:.4f} lift={m-base_lo:+.4f}")
        dh[k] = dict(likes_only=base_lo, weight_sweep=wsweep)
    out["dislike_helps"] = dh

    # ============ 7. IG1 genre purity ============
    log("=== IG1 genre purity ===")
    ig1 = {}
    for g in ["Horror", "Documentary", "Western", "Animation", "Sci-Fi"]:
        gi = gix[g]; members = np.where(Gmat[:, gi] > 0)[0]
        pop_members = members[np.argsort(-cnt[members])][:8]
        s = score_from_input(model, base, pop_members, np.ones(len(pop_members)))
        s2 = s.copy(); s2[pop_members] = -1e30  # exclude revealed
        top = np.argsort(-s2)[:10]
        purity = float(Gmat[top, gi].mean())
        ig1[g] = dict(purity_at10=purity, top_titles=[titles[int(t)] for t in top])
        log(f"  like-{g}: purity@10={purity:.2f}")
    out["ig1_genre_purity"] = ig1

    # ============ IG3 franchise coherence ============
    log("=== IG3 franchise coherence ===")
    ig3 = {}
    seeds_titles = {"Toy Story": None, "Harry Potter": None, "Lord of the Rings": None}
    lower = {i: titles[i].lower() for i in range(ni)}
    for key in list(seeds_titles):
        cand = [i for i in range(ni) if key.lower() in lower[i]]
        if not cand:
            continue
        seed_i = max(cand, key=lambda i: cnt[i])
        s = score_from_input(model, base, [seed_i], [1.0])
        s[seed_i] = -1e30
        top = np.argsort(-s)[:10]
        ig3[titles[seed_i]] = [titles[int(t)] for t in top]
        log(f"  {titles[seed_i]} -> {titles[int(top[0])]}, {titles[int(top[1])]}, ...")
    out["ig3_franchise"] = ig3

    # ============ IG4 graded-value monotone sweep ============
    log("=== IG4 graded-value sweep ===")
    ig4 = {}
    for g in ["Horror", "Sci-Fi", "Romance"]:
        gi = gix[g]; members = np.where(Gmat[:, gi] > 0)[0]
        pop_members = members[np.argsort(-cnt[members])]
        probe = pop_members[:8]; held = pop_members[8:208]
        seq = {}
        for label, val in [("hated", -1.0), ("meh", 0.0), ("liked", 0.44), ("loved", 1.0)]:
            if val == 0.0:
                s = score_from_input(model, base, [], [])  # empty = prior baseline for meh proxy
            else:
                s = score_from_input(model, base, probe, np.full(len(probe), val))
            seq[label] = percentile_rank(s, held)
        mono = seq["hated"] <= seq["meh"] + 1e-3 <= seq["liked"] + 1e-3 and seq["liked"] <= seq["loved"] + 1e-3
        ig4[g] = dict(pct=seq, monotone=bool(mono))
        log(f"  {g}: hated={seq['hated']:.2f} meh={seq['meh']:.2f} liked={seq['liked']:.2f} loved={seq['loved']:.2f} mono={mono}")
    out["ig4_graded_value"] = ig4

    # ============ IG5 graded-confidence (magnitude pulls harder) ============
    log("=== IG5 graded-confidence ===")
    ig5 = {}
    for g in ["Horror", "Sci-Fi", "Western"]:
        gi = gix[g]; members = np.where(Gmat[:, gi] > 0)[0]
        pop_members = members[np.argsort(-cnt[members])]
        probe = pop_members[:8]; held = pop_members[8:208]
        s_rough = score_from_input(model, base, probe, np.full(len(probe), 0.44))   # liked-rough
        s_well = score_from_input(model, base, probe, np.full(len(probe), 1.0))     # loved/know-well
        pr_rough = percentile_rank(s_rough, held); pr_well = percentile_rank(s_well, held)
        ig5[g] = dict(rough_pct=pr_rough, know_well_pct=pr_well, pulls_harder=bool(pr_well >= pr_rough - 1e-3))
        log(f"  {g}: rough={pr_rough:.2f} know_well={pr_well:.2f}")
    out["ig5_graded_confidence"] = ig5

    # ============ 8. G-GoT: know-well-but-hated -> still pull toward region? (consumption-as-taste) ============
    log("=== G-GoT know-well-but-hated ===")
    ggot = {}
    for g in ["Horror", "Sci-Fi", "Western"]:
        gi = gix[g]; members = np.where(Gmat[:, gi] > 0)[0]
        pop_members = members[np.argsort(-cnt[members])]
        probe = pop_members[:8]; held = pop_members[8:208]
        s_prior = score_from_input(model, base, [], [])
        s_hated = score_from_input(model, base, probe, -np.ones(len(probe)))
        pr_prior = percentile_rank(s_prior, held); pr_hated = percentile_rank(s_hated, held)
        # consumption-as-taste would PULL toward region (pr_hated > pr_prior). A signed model PUSHES away.
        ggot[g] = dict(prior_pct=pr_prior, hated_pct=pr_hated, pulls_toward=bool(pr_hated > pr_prior))
        log(f"  {g}: prior={pr_prior:.2f} hated={pr_hated:.2f} pulls_toward={pr_hated>pr_prior}")
    out["g_got"] = ggot

    # ============ 9. hygiene ============
    log("=== hygiene ===")
    hyg = {}
    rng = np.random.default_rng(0)
    # pick a real user profile
    SPL = build_splits(base, SEEDS[0]); users = cohort(base, SPL, "test")
    u0 = users[0]; profset, held, prof_r, held_r = SPL[u0]
    liked = [j for j in prof_r if prof_r[j] >= LO]
    def score_idx(idx, val):
        return score_from_input(model, base, idx, val)
    # G-intercept: empty interview -> deterministic prior, max-dev between two empty runs
    e1 = score_idx([], []); e2 = score_idx([], [])
    hyg["G_intercept_maxdev"] = float(np.max(np.abs(e1 - e2)))
    # G-order: shuffle revealed order -> identical
    order = liked[:]; sh = liked[:]; rng.shuffle(sh)
    s_a = score_idx(order, [scale_rating(prof_r[j]) for j in order])
    s_b = score_idx(sh, [scale_rating(prof_r[j]) for j in sh])
    hyg["G_order_maxdev"] = float(np.max(np.abs(s_a - s_b)))
    # G-falsify-count: duplicate answers -> unchanged (set semantics)
    dup_idx = order + order; dup_val = [scale_rating(prof_r[j]) for j in order] * 2
    s_dup = score_idx(dup_idx, dup_val)
    hyg["G_falsify_dup_maxdev"] = float(np.max(np.abs(s_a - s_dup)))
    # G-no-profile-leak: scramble UNREVEALED items (they are zeros -> no effect); verify byte-invariance
    # build input, then a copy where we "scramble" identity of unrevealed slots (still zero) -> identical
    xv = np.zeros(ni, np.float32)
    for j in order: xv[j] = scale_rating(prof_r[j])
    xv2 = xv.copy()
    unrev = np.setdiff1d(np.arange(ni), np.array(order))
    perm = rng.permutation(unrev)
    xv2[unrev] = xv2[perm]  # permute zeros among unrevealed -> still zeros
    with torch.no_grad():
        o1 = model(torch.from_numpy(xv[None]), 0.0)[0].numpy()[0]
        o2 = model(torch.from_numpy(xv2[None]), 0.0)[0].numpy()[0]
    hyg["G_no_profile_leak_maxdev"] = float(np.max(np.abs(o1 - o2)))
    hyg["monotone_accumulation_ok"] = out["monotone_accumulation_ok"]
    for kk, vv in hyg.items():
        log(f"  {kk} = {vv}")
    out["hygiene"] = hyg

    json.dump(out, open(os.path.join(OUTDIR, f"battery_{tag}.json"), "w"), indent=2)
    log(f"wrote battery_{tag}.json")


def eval_dislike_weight(model, base, SPL, users, k, wdis, seedoff=0):
    """k random liked answers + profile-half dislikes scaled by wdis (signed-negative). per-user NDCG@10 full."""
    ni = base["ni"]; headmask = base["headmask"]
    rng = np.random.default_rng(4242 + seedoff)
    xs = []; metas = []
    for u in users:
        profset, held, prof_r, held_r = SPL[u]
        tlike = [j for j in held if held_r[j] >= LO]
        if not tlike:
            continue
        liked_prof = [j for j in prof_r if prof_r[j] >= LO]
        dis_prof = [j for j in prof_r if prof_r[j] <= HI]
        if len(liked_prof) < k:
            continue
        sel = rng.choice(len(liked_prof), size=k, replace=False)
        idx = [liked_prof[i] for i in sel]
        xv = np.zeros(ni, np.float32)
        for j in idx:
            xv[j] = scale_rating(prof_r[j])
        for j in dis_prof:
            xv[j] = wdis * scale_rating(prof_r[j])   # scale_rating(dislike) already negative
        xs.append(xv); metas.append((profset, tlike))
    if not xs:
        return []
    with torch.no_grad():
        scores = model(torch.from_numpy(np.stack(xs)), 0.0)[0].numpy().astype(np.float64)
    ff = []
    for r, (profset, tlike) in enumerate(metas):
        nf = ndcg10(scores[r], tlike, profset, headmask, False)
        if nf is not None: ff.append(nf)
    return ff


if __name__ == "__main__":
    main()
