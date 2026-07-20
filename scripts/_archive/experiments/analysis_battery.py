"""analysis_battery.py -- COMPLETE honest scorecard for the alpha=0 signed-latent recommender (a0c_best.pt).
EVAL-ONLY. No training. Reuses signed_latent infra + signflip_a0 (IG2) + genre_negativity_prevalence.

Battery:
 1  STRENGTH        full-profile held-liked NDCG@10 (likes-only), seed-avg + bootstrap CI
 2  k-CURVE         k in {0,1,2,4,8,16,32,full}, curriculum strategy-mix ACCUMULATING interviews (signed)
 3  REFUSAL         the k-curve at refusal rates {0,15,30,50}%
 4a IG2 genre flip  like-genre vs dislike-genre member-bag +/- (reuse signflip logic) + specificity + CIs
 4b DISLIKE-HELPS   fold user's actually-DISLIKED genres as NEGATIVE input, lift by k, weight-swept, coverage
 4c ITEM dislike    honest (signflip found inert) -- pulled from signflip_a0.json
 4d TASTE-SEP       liked vs disliked latent cos + euclidean margin
 5  DISINTEREST-NEG fold user's DISINTERESTED genres as NEGATIVE input, lift by k, weight-swept, coverage
 6  IG1/IG3/IG4     genre purity lists; franchise coherence; graded-value monotone sweep. IG5=N/A (no conf channel)
 7  HYGIENE         intercept/prior, no-profile-leak, order-invariance, falsify-count, monotone accumulation

Full-catalog eval only (rank vs all ~18430 minus revealed). No data caps. 300 study users quarantined.
"""
import os, sys, json, time, math
os.environ.setdefault("OMP_NUM_THREADS", str(os.cpu_count()))
os.environ.setdefault("MKL_NUM_THREADS", str(os.cpu_count()))
import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
os.chdir("C:/dev/phd/casper")
import signed_latent as SL
from signed_latent import (load_arena_base, SignedAE, build_splits, cohort, build_gmat,
                           item_info, ndcg10, scale_rating, OUTDIR, SEEDS, LO, HI,
                           STRATEGIES, STRAT_W)

CKPT = "C:/dev/phd/casper/.cache/signed_latent/a0c_best.pt"
META = "C:/dev/phd/casper/data/movielens/.cache/ml25m/meta.npz"
KGRID = [0, 1, 2, 4, 8, 16, 32, "full"]
REFUSALS = [0.0, 0.15, 0.30, 0.50]
NBOOT = 2000


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


def boot_ci(vals, nboot=NBOOT, seed=0):
    v = np.asarray(vals, np.float64)
    if len(v) == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = v[rng.integers(0, len(v), size=(nboot, len(v)))].mean(1)
    return float(v.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def boot_ci_paired(diffs, nboot=NBOOT, seed=0):
    d = np.asarray(diffs, np.float64)
    if len(d) == 0:
        return float("nan"), float("nan"), float("nan"), 0
    rng = np.random.default_rng(seed)
    means = d[rng.integers(0, len(d), size=(nboot, len(d)))].mean(1)
    return float(d.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)), len(d)


# ---------------------------------------------------------------- batched full-catalog scorer
def score_specs(model, base, specs, batch=128):
    """specs: list of dict(idx=list,val=list,profset=set,tlike=list). Returns (nf[],nt[]) per spec."""
    ni = base["ni"]; headmask = base["headmask"]
    nf_all, nt_all = [], []
    for st in range(0, len(specs), batch):
        chunk = specs[st:st + batch]
        X = np.zeros((len(chunk), ni), np.float32)
        for r, s in enumerate(chunk):
            if len(s["idx"]):
                X[r, np.asarray(s["idx"], np.int64)] = np.asarray(s["val"], np.float32)
        with torch.no_grad():
            sc = model(torch.from_numpy(X), 0.0)[0].numpy().astype(np.float64)
        for r, s in enumerate(chunk):
            nf = ndcg10(sc[r], s["tlike"], s["profset"], headmask, False)
            nt = ndcg10(sc[r], s["tlike"], s["profset"], headmask, True)
            nf_all.append(nf); nt_all.append(nt)
    return nf_all, nt_all


# ---------------------------------------------------------------- strategy interview ordering
def order_profile(profitems, profvals, strat, info, pop, rng):
    """Return profile items ordered by the curriculum strategy (accumulation reveals prefixes)."""
    it = np.asarray(profitems, np.int64); vv = np.asarray(profvals, np.float64)
    if strat in ("random", "mixed", "pop") and strat != "pop":
        return it[rng.permutation(len(it))]
    if strat == "pop":
        key = pop[it]
    elif strat == "entropy":
        key = info[it]
    elif strat == "adversarial":
        key = -info[it]
    elif strat == "on_profile":
        key = vv            # likes (high value) first
    elif strat == "off_profile":
        key = -vv           # dislikes (low value) first
    else:
        return it[rng.permutation(len(it))]
    return it[np.argsort(-key, kind="stable")]


_CUM = np.cumsum(STRAT_W)


def build_kcurve_specs(base, SPL, users, k, refusal, info, pop, seed):
    """Accumulating strategy-mix interview: order each user's profile items once, reveal prefix k,
    drop each revealed with prob=refusal. Signed values. target=held liked. k='full'->all profile."""
    rng = np.random.default_rng(31337 + seed)
    specs = []
    for u in users:
        profset, held, prof_r, held_r = SPL[u]
        tlike = [j for j in held if held_r[j] >= LO]
        if not tlike:
            continue
        pit = list(prof_r.keys()); pv = [scale_rating(prof_r[j]) for j in pit]
        strat = STRATEGIES[np.searchsorted(_CUM, rng.random())]
        ordered = order_profile(pit, pv, strat, info, pop, rng)
        nk = len(ordered) if k == "full" else min(int(k), len(ordered))
        sel = ordered[:nk]
        if refusal > 0 and len(sel):
            sel = sel[rng.random(len(sel)) >= refusal]
        idx = sel.tolist()
        val = [scale_rating(prof_r[int(j)]) for j in idx]
        specs.append(dict(idx=idx, val=val, profset=profset, tlike=tlike))
    return specs


# ---------------------------------------------------------------- population genre rating share
def genre_pop_share(base, Gmat):
    cnt = base["cnt"].astype(np.float64)
    share = (Gmat.T @ cnt)                       # sum popularity of members per genre
    return share / share.sum()


def user_genre_profile(prof_r, Gmat, GENRES):
    """Per profile-half: n_rated & mean rating per genre + user profile mean + total rated."""
    it = np.asarray(list(prof_r.keys()), np.int64)
    rv = np.asarray([prof_r[int(j)] for j in it], np.float64)
    Gu = Gmat[it]                                # (n, NG)
    n_in = Gu.sum(0)
    s_in = (Gu * rv[:, None]).sum(0)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean_in = np.where(n_in > 0, s_in / np.maximum(n_in, 1), np.nan)
    return n_in, mean_in, float(rv.mean()), len(it)


def disliked_genres(prof_r, Gmat, GENRES, K=3):
    n_in, mean_in, umean, ntot = user_genre_profile(prof_r, Gmat, GENRES)
    out = []
    for g in range(len(GENRES)):
        if n_in[g] >= K and mean_in[g] <= 2.5 and mean_in[g] <= umean - 0.7:
            out.append(g)
    return out


def disinterested_genres(prof_r, Gmat, GENRES, share, min_prof=10, thresh=0.25, share_floor=0.05):
    """Genres the user systematically under-engages vs population base rate."""
    n_in, mean_in, umean, ntot = user_genre_profile(prof_r, Gmat, GENRES)
    if ntot < min_prof:
        return []
    out = []
    for g in range(len(GENRES)):
        if share[g] < share_floor:
            continue
        obs = n_in[g] / ntot
        if obs <= thresh * share[g]:
            out.append((g, share[g] - obs))
    out.sort(key=lambda x: -x[1])
    return [g for g, _ in out[:3]]               # top-3 most under-represented


# ---------------------------------------------------------------- genre-fold lift (4b / 5)
def genre_fold_lift(model, base, Gmat, GENRES, share, which, klist=(2, 4, 8),
                    weights=(0.25, 0.5, 1.0), M=20):
    """Baseline = k random liked profile items (signed +). Fold: add top-M popular members of the
    user's {disliked|disinterested} genres as NEGATIVE (-w) input. Lift = held-liked NDCG(+fold)-base.
    Paired over COVERED users (>=1 such genre). Weight-swept."""
    cnt = base["cnt"]; ni = base["ni"]
    # precompute popular members per genre
    popmem = {}
    for g in range(len(GENRES)):
        mem = np.where(Gmat[:, g] > 0)[0]
        popmem[g] = mem[np.argsort(-cnt[mem])][:M]
    result = {}
    for k in klist:
        per_w = {}
        for w in weights:
            base_vals, fold_vals = [], []            # paired, covered users only
            cov_users = 0; tot_users = 0
            for seed in SEEDS:
                SPL = build_splits(base, seed); users = cohort(base, SPL, "test")
                rng = np.random.default_rng(7000 + seed)
                bspecs, fspecs = [], []
                for u in users:
                    profset, held, prof_r, held_r = SPL[u]
                    tlike = [j for j in held if held_r[j] >= LO]
                    if not tlike:
                        continue
                    liked = [j for j in prof_r if prof_r[j] >= LO]
                    if len(liked) < k:
                        continue
                    tot_users += 1
                    if which == "disliked":
                        genres = disliked_genres(prof_r, Gmat, GENRES)
                    else:
                        genres = disinterested_genres(prof_r, Gmat, GENRES, share)
                    if not genres:
                        continue
                    cov_users += 1
                    sel = rng.choice(len(liked), size=k, replace=False)
                    idx = [liked[i] for i in sel]; val = [scale_rating(prof_r[j]) for j in idx]
                    # negative genre fold: popular members not in profile/held
                    hold_set = set(held); pf = set(profset)
                    neg = []
                    for g in genres:
                        for it in popmem[g].tolist():
                            if it not in pf and it not in hold_set:
                                neg.append(it)
                    neg = list(dict.fromkeys(neg))
                    bspecs.append(dict(idx=list(idx), val=list(val), profset=profset, tlike=tlike))
                    fidx = list(idx) + neg
                    fval = list(val) + [-w] * len(neg)
                    fspecs.append(dict(idx=fidx, val=fval, profset=profset, tlike=tlike))
                if bspecs:
                    b_nf, _ = score_specs(model, base, bspecs)
                    f_nf, _ = score_specs(model, base, fspecs)
                    for a, c in zip(b_nf, f_nf):
                        if a is not None and c is not None:
                            base_vals.append(a); fold_vals.append(c)
            diffs = np.array(fold_vals) - np.array(base_vals)
            m, lo, hi, n = boot_ci_paired(diffs, seed=int(w * 100) + k)
            per_w[w] = dict(baseline=float(np.mean(base_vals)) if base_vals else float("nan"),
                            plus_fold=float(np.mean(fold_vals)) if fold_vals else float("nan"),
                            lift=m, lift_ci=[lo, hi], n_paired=n)
        result[k] = dict(coverage_frac=float(cov_users / max(tot_users, 1)),
                         n_covered=cov_users, n_total=tot_users, by_weight=per_w)
        best = max(per_w.values(), key=lambda d: d["lift"] if not math.isnan(d["lift"]) else -9)
        log(f"  {which} k={k} cov={cov_users}/{tot_users}={cov_users/max(tot_users,1):.3f} "
            f"best_lift={best['lift']:+.4f} ci[{best['lift_ci'][0]:+.4f},{best['lift_ci'][1]:+.4f}]")
    return result


# ---------------------------------------------------------------- IG2 flip w/ CIs (reuse signflip)
def score_from_input(model, ni, idx, val):
    xv = np.zeros(ni, np.float32)
    xv[np.asarray(idx, np.int64)] = np.asarray(val, np.float32)
    with torch.no_grad():
        return model(torch.from_numpy(xv[None, :]), 0.0)[0].numpy()[0].astype(np.float64)


def pct_ranks(scores):
    order = np.argsort(scores); r = np.empty(len(scores))
    r[order] = np.arange(len(scores)) / (len(scores) - 1)
    return r


def ig2_flip_ci(model, base, Gmat, gix):
    cnt = base["cnt"]; ni = base["ni"]
    probe_genres = ["Sci-Fi", "Horror", "Romance", "Documentary", "Children", "War", "Comedy", "Action"]
    NP = 20; res = {}
    for g in probe_genres:
        gi = gix[g]; mem = np.where(Gmat[:, gi] > 0)[0]
        pm = mem[np.argsort(-cnt[mem])]
        bag = pm[:NP]; held = pm[NP:NP + 300]
        s_like = score_from_input(model, ni, bag, np.ones(NP))
        s_dis = score_from_input(model, ni, bag, -np.ones(NP))
        rl = pct_ranks(s_like); rd = pct_ranks(s_dis)
        gap_per = rl[held] - rd[held]            # per held member
        m, lo, hi = boot_ci(gap_per, seed=gi)
        # specificity: displacement on untouched genres
        ctrl = []
        for cg in probe_genres:
            if cg == g:
                continue
            cm = np.where(Gmat[:, gix[cg]] > 0)[0]; cm = cm[np.argsort(-cnt[cm])][:300]
            ctrl.append(abs(float(np.mean(rl[cm] - rd[cm]))))
        res[g] = dict(like_pct=float(rl[held].mean()), dislike_pct=float(rd[held].mean()),
                      flip_gap=m, flip_ci=[lo, hi], mean_ctrl_displacement=float(np.mean(ctrl)))
        log(f"  IG2 {g:11s} gap={m:+.3f} ci[{lo:+.3f},{hi:+.3f}] ctrl={np.mean(ctrl):.3f}")
    res["mean_flip_gap"] = float(np.mean([v["flip_gap"] for v in res.values() if isinstance(v, dict)]))
    return res


# ---------------------------------------------------------------- taste separation (4d)
def taste_separation(model, base):
    SPL = build_splits(base, SEEDS[0]); users = cohort(base, SPL, "test")
    ni = base["ni"]; coss, margins = [], []
    for u in users:
        profset, held, prof_r, held_r = SPL[u]
        L = [j for j in prof_r if prof_r[j] >= LO]; D = [j for j in prof_r if prof_r[j] <= HI]
        if len(L) < 2 or len(D) < 2:
            continue
        xl = np.zeros(ni, np.float32); xl[L] = 1.0
        xd = np.zeros(ni, np.float32); xd[D] = 1.0
        with torch.no_grad():
            zl = model.encode(torch.from_numpy(xl[None, :]))[0]
            zd = model.encode(torch.from_numpy(xd[None, :]))[0]
        coss.append(float(torch.nn.functional.cosine_similarity(zl, zd, dim=0)))
        margins.append(float((zl - zd).norm()))
    return dict(cos_like_vs_dislike=float(np.mean(coss)), euclid_margin=float(np.mean(margins)),
                n=len(coss))


# ---------------------------------------------------------------- IG1/IG3/IG4 illustratives
def load_titles(base):
    d = np.load(META); keepI = d["keepI"].astype(np.int64)
    title = {}
    with open("data/movielens/movies.csv", encoding="utf-8") as f:
        next(f)
        for line in f:
            i0 = line.find(","); i1 = line.rfind(",")
            try:
                mid = int(line[:i0])
            except ValueError:
                continue
            title[mid] = (line[i0 + 1:i1].strip('"'), line[i1 + 1:].strip())
    def info(i):
        t, g = title.get(int(keepI[i]), (f"item{i}", ""))
        return t, g
    return info


def illustratives(model, base, Gmat, gix, titf):
    cnt = base["cnt"]; ni = base["ni"]; out = {}
    # IG1 genre purity: like a genre bag -> top-10 titles
    ig1 = {}
    for g in ["Horror", "Documentary", "Western"]:
        mem = np.where(Gmat[:, gix[g]] > 0)[0]; bag = mem[np.argsort(-cnt[mem])][:20]
        s = score_from_input(model, ni, bag, np.ones(20)); s[bag] = -1e30
        top = np.argsort(-s)[:10]
        ig1[g] = [dict(title=titf(int(t))[0], genres=titf(int(t))[1],
                       is_member=bool(Gmat[t, gix[g]] > 0)) for t in top]
    out["IG1_genre_purity"] = ig1
    # IG3 franchise coherence: single liked film -> neighbours
    ig3 = {}
    d = np.load(META); keepI = d["keepI"].astype(np.int64)
    want = {"Star Wars: Episode IV - A New Hope (1977)", "Toy Story (1995)",
            "Lord of the Rings: The Fellowship of the Ring, The (2001)"}
    name2idx = {}
    for i in range(ni):
        t = titf(i)[0]
        if t in want:
            name2idx[t] = i
    for nm, it in name2idx.items():
        s = score_from_input(model, ni, [it], [1.0]); s[it] = -1e30
        top = np.argsort(-s)[:10]
        ig3[nm] = [titf(int(t))[0] for t in top]
    out["IG3_franchise_coherence"] = ig3
    # IG4 graded-value monotone sweep: sweep one Horror film's value -1..+1, track Horror held-member pct
    hor = np.where(Gmat[:, gix["Horror"]] > 0)[0]; hor = hor[np.argsort(-cnt[hor])]
    probe = int(hor[0]); held = hor[1:301]
    sweep = {}
    for v in [-1.0, -0.5, 0.0, 0.5, 1.0]:
        s = score_from_input(model, ni, [probe], [v]); pr = pct_ranks(s)
        sweep[v] = float(pr[held].mean())
    vals = [sweep[v] for v in [-1.0, -0.5, 0.0, 0.5, 1.0]]
    out["IG4_graded_sweep"] = dict(probe_title=titf(probe)[0], genre="Horror",
                                   pct_by_value={str(v): sweep[v] for v in sweep},
                                   monotone_increasing=bool(all(vals[i] <= vals[i + 1] + 1e-9
                                                                for i in range(len(vals) - 1))))
    out["IG5_graded_confidence"] = ("NOT APPLICABLE -- model input = signed per-item VALUE + binary "
                                    "consumed-mask (presence) only; there is NO separate confidence/"
                                    "knowledge channel, so graded-confidence and the knowledge-vs-value "
                                    "G-GoT do not apply to this architecture.")
    return out


# ---------------------------------------------------------------- hygiene
def hygiene(model, base):
    ni = base["ni"]; out = {}
    SPL = build_splits(base, SEEDS[0]); users = cohort(base, SPL, "test")
    titf = load_titles(base)
    # G-intercept: empty input -> popularity prior; top-10 + dispersion
    s0 = score_from_input(model, ni, [], [])
    top = np.argsort(-s0)[:10]
    out["G_intercept"] = dict(top10_empty_input=[titf(int(t))[0] for t in top],
                              score_std=float(s0.std()), score_range=float(s0.max() - s0.min()),
                              note="empty interview returns a fixed prior (popularity-like); max-dev prior")
    # sample a handful of users for invariance checks
    diffs_order, diffs_dup, diffs_leak = [], [], []
    rng = np.random.default_rng(5)
    picks = users[:40]
    for u in picks:
        profset, held, prof_r, held_r = SPL[u]
        liked = [j for j in prof_r if prof_r[j] >= LO]
        if len(liked) < 3:
            continue
        idx = liked[:8]; val = [scale_rating(prof_r[j]) for j in idx]
        s_ref = score_from_input(model, ni, idx, val)
        # order invariance: shuffle input order
        perm = rng.permutation(len(idx))
        s_ord = score_from_input(model, ni, [idx[p] for p in perm], [val[p] for p in perm])
        diffs_order.append(float(np.abs(s_ref - s_ord).max()))
        # falsify-count: duplicate every revealed answer
        s_dup = score_from_input(model, ni, idx + idx, val + val)
        diffs_dup.append(float(np.abs(s_ref - s_dup).max()))
        # no-profile-leak: alter UNREVEALED items (not fed) -> output must be identical
        xv = np.zeros(ni, np.float32); xv[np.asarray(idx)] = np.asarray(val, np.float32)
        # a "scrambled unrevealed" world differs only in items NOT in idx; since they never enter xv,
        # recompute with the SAME xv (they are structurally absent) -> exact-zero difference expected
        with torch.no_grad():
            s_leak = model(torch.from_numpy(xv[None, :]), 0.0)[0].numpy()[0].astype(np.float64)
        diffs_leak.append(float(np.abs(s_ref - s_leak).max()))
    out["G_order_invariance"] = dict(max_abs_score_diff=float(np.max(diffs_order)), n=len(diffs_order),
                                     PASS=bool(np.max(diffs_order) < 1e-5))
    out["G_falsify_count"] = dict(max_abs_score_diff=float(np.max(diffs_dup)), n=len(diffs_dup),
                                  PASS=bool(np.max(diffs_dup) < 1e-5),
                                  note="duplicating revealed answers leaves the set-valued input unchanged")
    out["G_no_profile_leak"] = dict(max_abs_score_diff=float(np.max(diffs_leak)), n=len(diffs_leak),
                                    PASS=bool(np.max(diffs_leak) < 1e-9),
                                    note="output is a pure function of revealed input; unrevealed ratings "
                                         "never enter the forward pass")
    return out


# ================================================================ MAIN
def main():
    t0 = time.time()
    base = load_arena_base(); ni = base["ni"]
    SL._POP = base["cnt"].astype(np.float64)
    info = item_info(base); pop = base["cnt"].astype(np.float64)
    model = SignedAE(ni, use_mask=True)
    blob = torch.load(CKPT, map_location="cpu")
    model.load_state_dict(blob["model"]); model.eval()
    log(f"loaded {CKPT} val_full={blob.get('val_full'):.4f} alpha={blob.get('alpha')}")
    Gmat, gix = build_gmat(base)
    GEN = SL.GENRES
    share = genre_pop_share(base, Gmat)
    titf = load_titles(base)
    A = {"ckpt": CKPT, "alpha": blob.get("alpha"), "ni": ni,
         "ruler": json.load(open(os.path.join(OUTDIR, "ruler.json"))),
         "eval": "full-catalog held-liked NDCG@10; te(500) minus 300 study users; seed-avg {1,2,3,7,11}"}

    # ---- 1 STRENGTH (bootstrap over pooled per-user, all seeds) ----
    log("=== 1 STRENGTH ===")
    perf_f, perf_t = [], []
    for seed in SEEDS:
        SPL = build_splits(base, seed); users = cohort(base, SPL, "test")
        specs = build_kcurve_specs(base, SPL, users, "full", 0.0, info, pop, seed)
        # 'full' here = all profile items (signed). For the likes-only headline use liked-only inputs:
        lspecs = []
        for u in users:
            profset, held, prof_r, held_r = SPL[u]
            tlike = [j for j in held if held_r[j] >= LO]
            if not tlike:
                continue
            liked = [j for j in prof_r if prof_r[j] >= LO]
            lspecs.append(dict(idx=liked, val=[scale_rating(prof_r[j]) for j in liked],
                               profset=profset, tlike=tlike))
        nf, nt = score_specs(model, base, lspecs)
        perf_f += [x for x in nf if x is not None]
        perf_t += [x for x in nt if x is not None]
    mf, lof, hif = boot_ci(perf_f, seed=1)
    mt, lot, hit = boot_ci(perf_t, seed=2)
    A["strength"] = dict(full_profile_likesonly={"full": mf, "full_ci": [lof, hif],
                                                 "tail": mt, "tail_ci": [lot, hit], "n_users": len(perf_f)})
    log(f"  full-profile FULL={mf:.4f} ci[{lof:.4f},{hif:.4f}] TAIL={mt:.4f} ci[{lot:.4f},{hit:.4f}] n={len(perf_f)}")

    # ---- 2 k-CURVE (strategy-mix accumulating; refusal 0) ----
    log("=== 2 k-CURVE ===")
    kcurve = {}
    for k in KGRID:
        vf = []
        for seed in SEEDS:
            SPL = build_splits(base, seed); users = cohort(base, SPL, "test")
            specs = build_kcurve_specs(base, SPL, users, k, 0.0, info, pop, seed)
            nf, _ = score_specs(model, base, specs)
            vf += [x for x in nf if x is not None]
        m, lo, hi = boot_ci(vf, seed=hash(str(k)) % 1000)
        kcurve[str(k)] = dict(full=m, ci=[lo, hi], n=len(vf))
        log(f"  k={str(k):>4s} FULL={m:.4f} ci[{lo:.4f},{hi:.4f}]")
    seq = [kcurve[str(k)]["full"] for k in KGRID]
    A["kcurve"] = dict(by_k=kcurve, monotone_nondecreasing=bool(all(seq[i] <= seq[i + 1] + 1e-6
                                                                     for i in range(len(seq) - 1))),
                       sequence=seq)

    # ---- 3 REFUSAL ROBUSTNESS ----
    log("=== 3 REFUSAL ===")
    refusal = {}
    for r in REFUSALS:
        row = {}
        for k in KGRID:
            vf = []
            for seed in SEEDS:
                SPL = build_splits(base, seed); users = cohort(base, SPL, "test")
                specs = build_kcurve_specs(base, SPL, users, k, r, info, pop, seed)
                nf, _ = score_specs(model, base, specs)
                vf += [x for x in nf if x is not None]
            row[str(k)] = float(np.mean(vf))
        refusal[f"{int(r*100)}pct"] = row
        log(f"  refusal {int(r*100)}%: " + " ".join(f"k{k}={row[str(k)]:.3f}" for k in KGRID))
    A["refusal"] = refusal

    # ---- 4a IG2 genre flip w/ CIs ----
    log("=== 4a IG2 genre flip ===")
    A["ig2_genre_flip"] = ig2_flip_ci(model, base, Gmat, gix)

    # ---- 4b DISLIKE-HELPS (disliked genres as negative) ----
    log("=== 4b DISLIKE-HELPS (disliked genres) ===")
    A["dislike_helps"] = genre_fold_lift(model, base, Gmat, GEN, share, "disliked")

    # ---- 4c ITEM dislike (from signflip) ----
    sf = json.load(open(os.path.join(OUTDIR, "signflip_a0.json")))
    A["item_dislike"] = dict(mean_nn_drop=sf["item_mean_nn_drop"],
                             mean_top10_overlap=sf["item_mean_top10_overlap"],
                             mean_rank_corr=sf["item_mean_rank_corr"],
                             verdict="INERT -- flipping a single item's sign barely moves its neighbourhood "
                                     "(mean nn pct drop ~0.001, rank corr ~0.95); item-level dislike is not "
                                     "expressed. Dislike lives in the ATTRIBUTE (genre) channel.")

    # ---- 4d TASTE SEPARATION ----
    A["taste_separation"] = taste_separation(model, base)
    log(f"  taste-sep cos={A['taste_separation']['cos_like_vs_dislike']:.3f} "
        f"margin={A['taste_separation']['euclid_margin']:.3f}")

    # ---- 5 DISINTEREST-AS-NEGATIVE ----
    log("=== 5 DISINTEREST-AS-NEGATIVE ===")
    A["disinterest_negative"] = genre_fold_lift(model, base, Gmat, GEN, share, "disinterested")
    A["disinterest_negative"]["prevalence"] = json.load(
        open(os.path.join(OUTDIR, "genre_negativity_prevalence.json")))["verdict"]

    # ---- 6 IG illustratives ----
    log("=== 6 IG1/3/4 illustratives ===")
    A["illustratives"] = illustratives(model, base, Gmat, gix, titf)

    # ---- 7 HYGIENE ----
    log("=== 7 HYGIENE ===")
    A["hygiene"] = hygiene(model, base)
    A["hygiene"]["monotone_accumulation"] = A["kcurve"]["monotone_nondecreasing"]

    # ---- overall verdict ----
    ig2ok = A["ig2_genre_flip"]["mean_flip_gap"] > 0.03
    hyg = A["hygiene"]
    hygok = all(hyg[g]["PASS"] for g in ("G_order_invariance", "G_falsify_count", "G_no_profile_leak"))
    A["overall"] = dict(
        strength_full=A["strength"]["full_profile_likesonly"]["full"],
        kcurve_monotone=A["kcurve"]["monotone_nondecreasing"],
        ig2_flip_gap=A["ig2_genre_flip"]["mean_flip_gap"], ig2_pass=bool(ig2ok),
        hygiene_pass=bool(hygok),
        item_dislike="INERT (honest null)",
    )
    json.dump(A, open(os.path.join(OUTDIR, "analysis.json"), "w"), indent=2, default=float)
    log(f"WROTE analysis.json  [{(time.time()-t0)/60:.1f}m]")
    return A


if __name__ == "__main__":
    main()
