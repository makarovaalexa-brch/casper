"""dans_iter2.py -- ITERATION-2 of the Distilled Answerer (author-directed amendment, signed 2026-07-09).

Contract: DESIGN_SHEET_DISTILLED_ANSWERER.md AMENDMENTS block. NO LLM calls. Deterministic seeds (123).
Scope (exactly this, then STOP for review regardless of outcome):
  1. BUFFNESS user-level features (dans_build.user_features / Universe): mean+median popularity
     percentile of rated items; SHARE + log-COUNT outside top-1000 and top-5000; niche-tag engagement
     (share+logcount of relevance-weighted contact with low-fame genome tags); foreign markers
     (genome) + era marker (share pre-1980); log rating-count and its interactions with obscurity.
  2. REFIT all knowledge channels with the enriched features (LOUO over the 173); report the NEW
     explained-vs-residual decomposition per channel (R^2 old iter1-feats vs new buff-feats) and the
     random-intercept sigma_u refit to the NEW residual (vs iter1 0.305/0.567/1.048).
  3. Two G2 calibration fixes: (a) attribute (entity) per-cut-margin user random effects (the
     ICC(k>=1) miss); (b) concept taste-gradient coefficient calibrated so the synthetic concept
     near-far(k1) gap matches the measured [0.095,0.115] (iter1 overshot at 0.125).
  4. FULL G2 re-run (13 decisive stats, CIs); G1 side-by-side with iter1; G3 error profile.
Writes to experiments/DANS_BUILD.md (clearly-marked ITERATION-2 section) + .cache/dans/*_iter2.*.
STOP after G2. No generation, no lazy answerer, no policies.

Run: python scripts/dans_iter2.py --stage fit2|g1_2|g2_2|g3_2|all
"""
import os, sys, json, time, argparse
import numpy as np
from sklearn.metrics import cohen_kappa_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "instrument2"))
import warnings
warnings.filterwarnings("ignore")

import dans_build as B
from dans_build import (Universe, load_173, ord_fit, ord_prob, fit_scale, zscale, md, sha,
                        eb_percut_sigma, DANS, WORKING, KLAB, VLAB, VSTAR, SEED)
import dans_stages as S
from dans_stages import (collect, fit_know_channel, fit_value_channel, KNOW_COLS, VAL_COLS, UFEATS,
                         UFEATS_ITER1, UFEATS_BUFF, UVEC_SLICE, N_ITEM_KNOW, know_probs, value_probs,
                         _sample_cat, _stagea, _cells_from_real, _cells_from_synth, _band_by)
from adaptivity_battery_v1 import icc_oneway, paired_boot

MODELS2 = f"{DANS}/models_iter2.json"
SIGMA_U_ITER1 = {"concept": 0.305, "entity": 0.567, "item": 1.048}   # from fit_iter1.log
TASTE_TARGET = 0.105            # center of measured concept near-far(k1) [0.095,0.115]
TASTE_BETA_IDX = 5             # theta index of concept taste_align: 2 cuts + [sat,logmemb,logpw,TASTE]


# ---------------------------------------------------------------- decomposition: user-variance explained
def _decomp(acc):
    """Per channel: regress per-user answer-rate mean(k>=1) on the user-feature block; report R^2 for the
    ITER1 (4) feature set vs the full (iter1+buffness) set = 'explained % old vs new'. Also raw between-
    user variance of the answer rate."""
    out = {}
    for ch in ("concept", "entity", "item"):
        lin = acc[ch]["lin"]; y = (acc[ch]["ky"] >= 1).astype(float); grp = acc[ch]["grp"]
        lo, hi = UVEC_SLICE[ch]
        U = lin[:, lo:hi]                                  # user block (constant within user)
        uq = np.unique(grp)
        Xu, yu = [], []
        for u in uq:
            m = grp == u
            Xu.append(U[m][0]); yu.append(float(y[m].mean()))
        Xu = np.asarray(Xu); yu = np.asarray(yu)
        mu = Xu.mean(0); sd = np.where(Xu.std(0) > 1e-9, Xu.std(0), 1.0)
        Xz = (Xu - mu) / sd
        ss = float(np.sum((yu - yu.mean()) ** 2))
        def r2_in(cols):
            A = np.column_stack([np.ones(len(Xz)), Xz[:, cols]])
            beta, *_ = np.linalg.lstsq(A, yu, rcond=None)
            return float(1.0 - np.sum((yu - A @ beta) ** 2) / max(ss, 1e-12))
        def r2_cv(cols, alpha=1.0):
            """5-fold CV ridge R2 -- the honest number with ~34 features on 173 users."""
            rng = np.random.default_rng(0); perm = rng.permutation(len(yu))
            folds = np.array_split(perm, 5)
            resid2 = 0.0
            for f in folds:
                tr = np.setdiff1d(perm, f)
                A = np.column_stack([np.ones(len(tr)), Xz[tr][:, cols]])
                At = np.column_stack([np.ones(len(f)), Xz[f][:, cols]])
                reg = alpha * np.eye(A.shape[1]); reg[0, 0] = 0.0
                beta = np.linalg.solve(A.T @ A + reg, A.T @ yu[tr])
                resid2 += float(np.sum((yu[f] - At @ beta) ** 2))
            return float(1.0 - resid2 / max(ss, 1e-12))
        n1 = len(UFEATS_ITER1)
        out[ch] = dict(n_users=int(len(uq)), var_rate=float(np.var(yu, ddof=1)),
                       r2_old=r2_in(list(range(n1))), r2_new=r2_in(list(range(len(UFEATS)))),
                       r2_old_cv=r2_cv(list(range(n1))), r2_new_cv=r2_cv(list(range(len(UFEATS)))))
    return out


# ---------------------------------------------------------------- concept taste-gradient calibration
def _precompute_concept(uni, users, cmodel):
    """Per user: zscaled concept design matrix Xs + concept-cell (rid, align) list for near-far(k1)."""
    g = cmodel["gamma"]; mu = np.asarray(cmodel["mu"]); sd = np.asarray(cmodel["sd"])
    pc = []
    for rec in users:
        f = uni.user_features(rec["known"])
        X = np.column_stack([np.power(np.maximum(f["concept_raw"], 0.0), g), f["concept_lin"]])
        Xs = zscale(X, mu, sd)
        align = f["concept_lin"][:, 2]
        rids = [rid for ch, rid, k, v, st in rec["cells"] if ch == "concept"]
        pc.append((rec["u"], Xs, np.asarray(rids, int), align))
    return pc


def _concept_nearfar_k1(pc, theta, sigma_u, seeds=(1, 2, 3, 7, 11, 13, 17, 23)):
    """Mean over users of within-user (near-far) gap in P(k>=1), by SAMPLING (matches G2 A2), averaged
    over several seeds to denoise the calibration objective."""
    gaps = []
    for seed in seeds:
        per_user = []
        for u, Xs, rids, align in pc:
            if len(rids) < 20:
                continue
            rng = np.random.default_rng(seed * 1000 + u)
            P = ord_prob(theta, Xs, 3, shift=float(rng.normal(0.0, sigma_u)))
            k = _sample_cat(P, rng)
            kc = k[rids]; al = align[rids]
            y1 = (kc >= 1).astype(float); med = np.median(al)
            nm = al >= med; fm = al < med
            if nm.any() and fm.any():
                per_user.append(float(y1[nm].mean() - y1[fm].mean()))
        gaps.append(np.mean(per_user))
    return float(np.mean(gaps))


def _calibrate_taste(pc, cmodel):
    """Search a multiplier on the concept taste_align coefficient so the synthetic near-far(k1) gap
    matches TASTE_TARGET (0.105). Returns (multiplier, gap_before, gap_after, calibrated_theta)."""
    theta0 = np.asarray(cmodel["theta"], float); sigma_u = cmodel["sigma_u"]
    gap0 = _concept_nearfar_k1(pc, theta0, sigma_u)
    best = None
    for m in [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.85, 0.75, 0.65, 0.55]:
        th = theta0.copy(); th[TASTE_BETA_IDX] = theta0[TASTE_BETA_IDX] * m
        gap = _concept_nearfar_k1(pc, th, sigma_u)
        err = abs(gap - TASTE_TARGET)
        if best is None or err < best[0]:
            best = (err, m, gap, th)
    _, m, gap1, th = best
    return m, gap0, gap1, th


# ---------------------------------------------------------------- STAGE fit2
def stage_fit2():
    t0 = time.time()
    uni = Universe()
    users = load_173(uni)
    acc = collect(uni, users)
    print(f"\n==== ITERATION-2 FIT: enriched (buffness) features, LOUO over {len(users)} ====", flush=True)
    print(f"[fit2] {len(UFEATS)} user-level features ({len(UFEATS_ITER1)} iter1 + {len(UFEATS_BUFF)} buff): "
          f"{UFEATS}", flush=True)
    # value models unchanged by buffness (value features don't use the user block) -> reuse iter1 fit.
    iter1 = json.load(open(f"{DANS}/models.json"))
    models = dict(knowledge={}, value=iter1["value"],
                  meta=dict(n_users=len(users), seed=SEED, grid_sha=sha(WORKING), iteration=2,
                            ufeats=UFEATS))
    kfit = {}
    for ch in ("concept", "entity", "item"):
        mk = fit_know_channel(ch, acc[ch], has_gamma=(ch in ("concept", "entity")))
        kfit[ch] = mk
    # ---- calibration fix (a): entity per-cut-margin random effects ----
    e = kfit["entity"]
    sig_cuts = eb_percut_sigma(np.asarray(e["theta"]), e["_Xs"], e["_y"].astype(int), e["_grp"], 3)
    print(f"    [entity per-cut sigma] cut1(k>=1)={sig_cuts[0]:.3f} cut2(k>=2)={sig_cuts[1]:.3f} "
          f"(single-shift was {e['sigma_u']:.3f})", flush=True)
    # decomposition (before dropping _Xs)
    decomp = _decomp(acc)
    # stash LOUO predictions for G1 (avoid a second LOUO pass), matching iter-1's fit-LOUO convention
    louo = {}
    for ch in ("concept", "entity", "item"):
        louo[f"{ch}__y"] = kfit[ch]["_y"]; louo[f"{ch}__pred"] = kfit[ch]["_pred"]
        louo[f"{ch}__strat"] = acc[ch]["lin"][:, 1] if ch != "item" else acc[ch]["lin"][:, 4]
    np.savez(f"{DANS}/louo_iter2.npz", **louo)
    # assemble knowledge models (drop private arrays)
    for ch in ("concept", "entity", "item"):
        mk = kfit[ch]
        for pk in ("_Xs", "_y", "_grp", "_pred", "_prob"):
            mk.pop(pk, None)
        models["knowledge"][ch] = mk
    models["knowledge"]["entity"]["sigma_cuts"] = [float(sig_cuts[0]), float(sig_cuts[1])]
    # ---- calibration fix (b): concept taste-gradient multiplier ----
    cmodel = dict(models["knowledge"]["concept"])
    cmodel["mu"] = np.asarray(cmodel["mu"]); cmodel["sd"] = np.asarray(cmodel["sd"])
    cmodel["theta"] = np.asarray(cmodel["theta"])
    pc = _precompute_concept(uni, users, cmodel)
    mult, gap0, gap1, th_cal = _calibrate_taste(pc, cmodel)
    print(f"    [concept taste calib] multiplier={mult:.2f}  near-far(k1) {gap0:.3f} -> {gap1:.3f} "
          f"(target {TASTE_TARGET})", flush=True)
    beta_before = float(models["knowledge"]["concept"]["theta"][TASTE_BETA_IDX])
    models["knowledge"]["concept"]["theta"] = [float(x) for x in th_cal]
    models["knowledge"]["concept"]["taste_calib_mult"] = float(mult)
    models["knowledge"]["concept"]["taste_beta_before"] = beta_before
    # serialize
    json.dump(models, open(MODELS2, "w"), indent=1, default=float)
    assert os.path.exists(MODELS2)
    # ---- report ----
    print("\n---- DECOMPOSITION: user-level answer-rate variance explained (R^2) ----", flush=True)
    for ch in ("concept", "entity", "item"):
        d = decomp[ch]
        print(f"    {ch:8s} R2_old={d['r2_old']:.3f} R2_new={d['r2_new']:.3f} "
              f"R2_old_cv={d['r2_old_cv']:.3f} R2_new_cv={d['r2_new_cv']:.3f} "
              f"(var_rate={d['var_rate']:.4f}, n={d['n_users']})", flush=True)
    print("\n---- sigma_u (random intercept refit to NEW residual) ----", flush=True)
    for ch in ("concept", "entity", "item"):
        su = models["knowledge"][ch]["sigma_u"]
        extra = f" | per-cut [{sig_cuts[0]:.3f},{sig_cuts[1]:.3f}]" if ch == "entity" else ""
        print(f"    {ch:8s} sigma_u iter1={SIGMA_U_ITER1[ch]:.3f} -> iter2={su:.3f}{extra}", flush=True)
    # ---- coefficient table (all census signs are a deliverable) ----
    rows = []
    for ch in ("concept", "entity", "item"):
        m = models["knowledge"][ch]; beta = np.asarray(m["theta"])[2:]
        for cn, bv in zip(m["cols"], beta):
            rows.append((f"know:{ch}", cn, float(bv)))
        if m.get("gamma") is not None:
            rows.append((f"know:{ch}", "saturation_gamma", m["gamma"]))
    # ---- FEATURE IMPORTANCE STUDY (what does an LLM read in a rating profile?) ----
    imp = {}
    for ch in ("concept", "entity", "item"):
        m = models["knowledge"][ch]
        bm = np.asarray(m["fold_beta_mean"]); bs = np.asarray(m["fold_beta_sd"])
        sc = np.asarray(m["fold_sign_cons"])
        imp[ch] = dict(zip(UFEATS, zip(bm.tolist(), bs.tolist(), sc.tolist())))
    rank_score = {fn: sum(abs(imp[ch][fn][0]) for ch in imp) for fn in UFEATS}
    ranked = sorted(UFEATS, key=lambda f: -rank_score[f])
    print("\n---- FEATURE IMPORTANCE (user block; ranked by sum |beta| across channels) ----", flush=True)
    for fn in ranked[:15]:
        print(f"    {fn:22s} " + "  ".join(f"{ch[:4]}={imp[ch][fn][0]:+.3f}(sd{imp[ch][fn][1]:.3f})"
              for ch in ("concept", "entity", "item")), flush=True)
    # write md
    md("\n\n---\n\n# ITERATION-2 (author-directed amendment; signed 2026-07-09)\n\n"
       "Contract: DESIGN_SHEET AMENDMENTS block + coordinator addenda (full feature census, "
       "distinctiveness, pockets, regularized importance study). Enriched user-level features + refit + "
       "two G2 calibration fixes; full G2 re-run; G1/G3 side-by-side. NO LLM calls; seeds 123. Author "
       "rationale (quoted): *\"the LLM saw ONLY the profile, so its per-user behaviour is "
       "profile-predictable in principle; the old 90% item residual measured feature poverty, not "
       "randomness.\"* Census constraint: every feature derives from WHAT THE LLM SAW -- verified: the "
       "gate's profile_text showed the FULL known half (no 200-cap exists in the gate code; max shown "
       "702 titles), so known-half features == shown-profile features exactly.\n\n"
       "## ITER-2 feature census (user-level, known-half only; identical transform real & synthetic)\n\n"
       f"{len(UFEATS)} user features = {len(UFEATS_ITER1)} iter-1 consumption + {len(UFEATS_BUFF)} census: "
       "volume, obscurity (mean/median pop percentile, out-top-1000/5000 share+logcount), era (pre-1980 "
       "and pre-1970 share+count, era spread, per-question era density+distance), breadth (genre entropy, "
       "taste-cloud dispersion, top-pocket concentration, effective pocket count), distinctiveness (taste "
       "typicality vs population centroid, neighborhood density vs 4000 non-study reference users), "
       "territory (bank coverage + per-question co-knowledge/genre/era), markers (foreign genome tags "
       f"({uni.n_foreign_tags}), foreign-title, franchise, documentary, animation, niche-tag engagement), "
       "rating style (mean, sd, share-max, share-extreme), interactions (volume x out-top-5000, volume x "
       "niche, volume x dispersion). Fitted with a RIDGE penalty on the user block, lambda by 5-fold "
       "user-grouped CV (" +
       ", ".join(f"{ch} lambda={models['knowledge'][ch]['lambda_u']}" for ch in
                 ("concept", "entity", "item")) + ").\n\n"
       "### Fitted knowledge coefficients (standardized; sign = effect on higher knowledge)\n\n"
       "| model | feature | coefficient |\n|---|---|--:|\n" +
       "".join(f"| {r[0]} | {r[1]} | {r[2]:+.3f} |\n" for r in rows) + "\n"
       "(concept taste_align shown is POST-calibration; pre-calibration beta = "
       f"{beta_before:+.3f}, multiplier {mult:.2f}.)\n\n"
       "## STANDALONE STUDY -- what does an LLM read in a rating profile? (anatomy of judged "
       "answerability)\n\n"
       "Ranked user-feature importance: LOUO fold-mean beta (173 refits), fold sd, and sign-consistency "
       "per knowledge channel; ranked by sum |beta| across channels. (Flagged by the author as a "
       "potentially publishable standalone table.)\n\n"
       "| rank | feature | concept beta (sd, sign%) | entity beta (sd, sign%) | item beta (sd, sign%) |\n"
       "|--:|---|---|---|---|\n" +
       "".join(f"| {i+1} | {fn} | " + " | ".join(
               f"{imp[ch][fn][0]:+.3f} ({imp[ch][fn][1]:.3f}, {imp[ch][fn][2]*100:.0f}%)"
               for ch in ("concept", "entity", "item")) + " |\n"
               for i, fn in enumerate(ranked)) + "\n"
       "## DECOMPOSITION -- user answer-rate variance explained by observable features (R^2)\n\n"
       "Per channel: OLS of per-user mean(knowledge>=1) on the user-feature block; R^2 = fraction of "
       "between-user variance explained (residual = what the random intercept carries). 'Old' = the 4 "
       "iter-1 consumption features; 'New' = the full census. CV = 5-fold ridge cross-validated R^2 "
       "(the honest number with ~34 features on 173 users).\n\n"
       "| channel | explained OLD | explained NEW | OLD (CV) | NEW (CV) | between-user var(rate) |\n"
       "|---|--:|--:|--:|--:|--:|\n" +
       "".join(f"| {ch} | {decomp[ch]['r2_old']*100:.0f}% | {decomp[ch]['r2_new']*100:.0f}% | "
               f"{decomp[ch]['r2_old_cv']*100:.0f}% | {decomp[ch]['r2_new_cv']*100:.0f}% | "
               f"{decomp[ch]['var_rate']:.4f} |\n" for ch in ("concept", "entity", "item")) + "\n"
       "## sigma_u -- random intercept shrinkage (refit to the NEW residual)\n\n"
       "| channel | sigma_u iter-1 | sigma_u iter-2 | note |\n|---|--:|--:|---|\n" +
       f"| concept | {SIGMA_U_ITER1['concept']:.3f} | {models['knowledge']['concept']['sigma_u']:.3f} | single shift |\n"
       f"| entity/attribute | {SIGMA_U_ITER1['entity']:.3f} | {models['knowledge']['entity']['sigma_u']:.3f} | "
       f"replaced by PER-CUT [k>=1: {sig_cuts[0]:.3f}, k>=2: {sig_cuts[1]:.3f}] (fix a) |\n"
       f"| item | {SIGMA_U_ITER1['item']:.3f} | {models['knowledge']['item']['sigma_u']:.3f} | single shift |\n\n"
       f"Calibration fix (b): concept taste_align coefficient scaled x{mult:.2f} "
       f"(near-far(k1) {gap0:.3f} -> {gap1:.3f}, target {TASTE_TARGET}).\n\n")
    json.dump(dict(decomp=decomp, sigma_u={ch: models["knowledge"][ch]["sigma_u"] for ch in
                   ("concept", "entity", "item")}, sigma_cuts_entity=[float(sig_cuts[0]), float(sig_cuts[1])],
                   sigma_u_iter1=SIGMA_U_ITER1, taste_mult=float(mult), taste_gap_before=gap0,
                   taste_gap_after=gap1, taste_target=TASTE_TARGET,
                   importance={ch: imp[ch] for ch in imp}, importance_rank=ranked),
              open(f"{DANS}/decomp_iter2.json", "w"), indent=1, default=float)
    print(f"\n[fit2] wrote {MODELS2}, {DANS}/decomp_iter2.json  [{time.time()-t0:.0f}s]", flush=True)


# ---------------------------------------------------------------- load iter2 models for gen/gates
def _load_models2():
    m = json.load(open(MODELS2))
    for grp in ("knowledge", "value"):
        for ch in m[grp]:
            for k in ("mu", "sd", "theta"):
                m[grp][ch][k] = np.asarray(m[grp][ch][k], float)
    return m


# ---------------------------------------------------------------- STAGE g1_2 (agreement, LOUO, side-by-side)
def stage_g1_2():
    """Re-run G1 LOUO agreement with enriched features; side-by-side with iter1 g1.json.
    Consumes the LOUO predictions stashed by fit2 (same fit-LOUO convention as iter-1)."""
    iter1 = json.load(open(f"{DANS}/g1.json"))
    dat = np.load(f"{DANS}/louo_iter2.npz")
    print("\n==== ITER-2 G1 AGREEMENT (LOUO) -- enriched features; side-by-side with iter1 ====", flush=True)
    rep = dict(knowledge={}, value={})
    for ch in ("concept", "entity", "item"):
        y = dat[f"{ch}__y"].astype(int); pred = dat[f"{ch}__pred"].astype(int)
        band = _band_by(dat[f"{ch}__strat"])
        rep["knowledge"][ch] = {}
        for sname in ("all", "low", "mid", "high"):
            m = np.ones(len(y), bool) if sname == "all" else (band == sname)
            if m.sum() == 0:
                continue
            acc_m = float((pred[m] == y[m]).mean())
            maj = np.bincount(y[m], minlength=3).argmax(); accb = float((y[m] == maj).mean())
            kap = float(cohen_kappa_score(y[m], pred[m])) if len(np.unique(y[m])) > 1 else float("nan")
            rep["knowledge"][ch][sname] = dict(n=int(m.sum()), acc=acc_m, acc_base=accb, kappa=kap)
    # value channels unchanged (reuse iter1 value MAE)
    rep["value"] = iter1["value"]
    # print + md side-by-side
    md("## ITER-2 G1 AGREEMENT (LOUO) -- side-by-side with iter-1\n\n"
       "Knowledge accuracy / kappa vs LLM (baseline = per-stratum majority). Value MAE unchanged "
       "(value channels do not use the buffness user block).\n\n"
       "| channel | stratum | n | acc iter1 | acc iter2 | kappa iter1 | kappa iter2 | base | verdict |\n"
       "|---|---|--:|--:|--:|--:|--:|--:|:--:|\n")
    print("  channel  stratum      acc1   acc2   kap1   kap2  verdict", flush=True)
    all_ok = True
    for ch in ("concept", "entity", "item"):
        for sname in ("all", "low", "mid", "high"):
            n = rep["knowledge"][ch].get(sname)
            if n is None:
                continue
            o = iter1["knowledge"][ch].get(sname, {})
            a1 = o.get("acc", float("nan")); a2 = n["acc"]; k1 = o.get("kappa", float("nan")); k2 = n["kappa"]
            base = n["acc_base"]
            vd = "beat" if a2 > base + 1e-9 else "tie/lose"
            if sname in ("all", "low") and a2 < a1 - 0.005:
                all_ok = False       # enriched features must not degrade accuracy at all/niche
            niche = " (NICHE)" if sname == "low" else ""
            print(f"  {ch:8s} {sname:5s}     {a1:.3f}  {a2:.3f}  {k1:.3f}  {k2:.3f}  {vd}", flush=True)
            md(f"| {ch} | {sname}{niche} | {n['n']} | {a1:.3f} | {a2:.3f} | {k1:.3f} | {k2:.3f} | "
               f"{base:.3f} | {vd} |\n")
    passed = all(rep["knowledge"][ch]["all"]["acc"] > rep["knowledge"][ch]["all"]["acc_base"]
                 for ch in rep["knowledge"]) and \
             all(rep["knowledge"][ch]["low"]["acc"] >= rep["knowledge"][ch]["low"]["acc_base"] - 1e-9
                 for ch in rep["knowledge"])
    print(f"\n[g1_2] beats-baseline(all+niche)={passed}  no-degradation-vs-iter1={all_ok}", flush=True)
    md(f"\n**ITER-2 G1 verdict**: beats population-rate baseline (all channels + niche) = {passed}; "
       f"enriched features do NOT degrade accuracy vs iter-1 at all/niche strata = {all_ok}.\n\n")
    json.dump(dict(rep=rep, passed=bool(passed), no_degradation=bool(all_ok)),
              open(f"{DANS}/g1_iter2.json", "w"), indent=1, default=float)
    print(f"[g1_2] wrote {DANS}/g1_iter2.json", flush=True)


# ---------------------------------------------------------------- STAGE g2_2 (fuel reproduction, 13 decisive)
def stage_g2_2():
    uni = Universe()
    users = load_173(uni)
    models = _load_models2()
    print("\n==== ITER-2 G2 FUEL REPRODUCTION (DECISIVE) -- pre-registered, 13 decisive ====", flush=True)
    print("Threshold (printed before results): each Stage-A statistic's SYNTHETIC value must fall within "
          "the REAL 95% CI (or CIs overlap). Sampling only. Decisive (13): A1 ICC(k>=1),ICC(k>=2) x "
          "{concept,attribute,item}; A2 concept near-far(k1), item-mid near-far(k2); A3 concept "
          "niche/broad k>=1, item niche/broad k>=2, overall rough share.", flush=True)
    real = _stagea(_cells_from_real(uni, users), uni, seed=0)
    synth = _stagea(_cells_from_synth(uni, users, models), uni, seed=0)
    rows = []
    TOL_POINT = 0.03
    def add(name, rv, rci, sv, sci):
        if rci[0] == rci[1]:
            ok = bool(abs(sv - rv) <= TOL_POINT)
        else:
            ok = bool((rci[0] <= sv <= rci[1]) or not (sci[1] < rci[0] or rci[1] < sci[0]))
        rows.append((name, rv, rci, sv, sci, ok))
    for ch in ("concept", "attribute", "item"):
        a = real["A1"][ch]; b = synth["A1"][ch]
        add(f"A1 {ch} ICC(k>=1)", a["icc_k1"], a["ci_k1"], b["icc_k1"], b["ci_k1"])
        add(f"A1 {ch} ICC(k>=2)", a["icc_k2"], a["ci_k2"], b["icc_k2"], b["ci_k2"])
        add(f"A1 {ch} base(k>=1)", a["base_k1"], [a["base_k1"]] * 2, b["base_k1"], [b["base_k1"]] * 2)
    for band in ("low", "mid", "high"):
        a = real["A2"]["item_bands"][band]; b = synth["A2"]["item_bands"][band]
        add(f"A2 item-{band} near-far(k2)", a["delta"], a["ci"], b["delta"], b["ci"])
    a = real["A2"]["concept"]; b = synth["A2"]["concept"]
    add("A2 concept near-far(k1)", a["delta"], a["ci"], b["delta"], b["ci"])
    for nm in ("niche", "broad"):
        if nm in real["A3"]["concept_k1"] and nm in synth["A3"]["concept_k1"]:
            rv = real["A3"]["concept_k1"][nm]; sv = synth["A3"]["concept_k1"][nm]
            add(f"A3 concept-{nm} k>=1", rv, [rv, rv], sv, [sv, sv])
        if nm in real["A3"]["item_k2"] and nm in synth["A3"]["item_k2"]:
            rv = real["A3"]["item_k2"][nm]; sv = synth["A3"]["item_k2"][nm]
            add(f"A3 item-{nm} k>=2", rv, [rv, rv], sv, [sv, sv])
    add("A3 overall rough share", real["A3"]["rough_share"], [real["A3"]["rough_share"]] * 2,
        synth["A3"]["rough_share"], [synth["A3"]["rough_share"]] * 2)
    print("\n  stat                              real [95% CI]            synth [95% CI]        match", flush=True)
    npass = 0
    for nm, rv, rci, sv, sci, ok in rows:
        npass += ok
        print(f"    {nm:33s} {rv:+.3f} [{rci[0]:+.3f},{rci[1]:+.3f}]  {sv:+.3f} [{sci[0]:+.3f},{sci[1]:+.3f}]  "
              f"{'OK' if ok else 'MISS'}", flush=True)
    decisive = [r for r in rows if "ICC" in r[0] or r[0] in
                ("A2 concept near-far(k1)", "A2 item-mid near-far(k2)") or r[0].startswith("A3")]
    dpass = sum(r[5] for r in decisive)
    verdict = dpass == len(decisive)
    misses = [r[0] for r in decisive if not r[5]]
    print(f"\n  MATCH: {npass}/{len(rows)} all rows; DECISIVE {dpass}/{len(decisive)}", flush=True)
    print(f"  ITER-2 G2 VERDICT: {'PASS' if verdict else 'FAIL'}"
          + ("" if verdict else f"  (misses: {misses})"), flush=True)
    md("## ITER-2 G2 -- FUEL REPRODUCTION (decisive, 13 stats)\n\n**Pre-registered**: each statistic's "
       "synthetic value within the real 95% CI (or CIs overlap). Sampling only. Fixes applied: entity "
       "per-cut random effects (a); concept taste_align calibrated (b).\n\n"
       "| statistic | real [95% CI] | synth [95% CI] | match |\n|---|---|---|:--:|\n" +
       "".join(f"| {r[0]} | {r[1]:+.3f} [{r[2][0]:+.3f},{r[2][1]:+.3f}] | {r[3]:+.3f} "
               f"[{r[4][0]:+.3f},{r[4][1]:+.3f}] | {'OK' if r[5] else 'MISS'} |\n" for r in rows) +
       f"\n**ITER-2 G2 verdict**: decisive {dpass}/{len(decisive)} within CI -> "
       f"{'PASS' if verdict else 'FAIL'}." + ("" if verdict else f" Misses: {', '.join(misses)}.") + "\n\n")
    json.dump(dict(rows=[(r[0], r[1], r[2], r[3], r[4], r[5]) for r in rows], decisive_pass=int(dpass),
                   decisive_n=len(decisive), verdict=bool(verdict), misses=misses),
              open(f"{DANS}/g2_iter2.json", "w"), indent=1, default=float)
    print(f"[g2_2] wrote {DANS}/g2_iter2.json", flush=True)


# ---------------------------------------------------------------- STAGE g3_2 (error profile)
def stage_g3_2():
    uni = Universe()
    users = load_173(uni)
    models = _load_models2()
    print("\n==== ITER-2 G3 ERROR PROFILE (rated cells, value passthrough disabled) ====", flush=True)
    errs = []; tp = []
    rng = np.random.default_rng(SEED)
    for rec in users:
        f = uni.user_features(rec["known"])
        Pv = value_probs(uni, f, models)["item"]
        for rid, star in rec["data"]:
            if rid is None:
                continue
            draw = _sample_cat(Pv[rid][None, :], rng)[0]
            errs.append(abs(VSTAR[draw] - star)); tp.append((star, VSTAR[draw]))
    errs = np.array(errs); tp = np.array(tp)
    mae = float(errs.mean()); disp = float(errs.std())
    corr = float(np.corrcoef(tp[:, 0], tp[:, 1])[0, 1]) if tp[:, 1].std() > 0 else float("nan")
    too_clean = mae < 0.50 or (not np.isnan(corr) and corr > 0.70)
    print(f"  n={len(errs)} MAE={mae:.3f} dispersion={disp:.3f} corr={corr:.3f} "
          f"(LLM ref MAE~0.70 corr~0.5) too_clean={too_clean}", flush=True)
    md("## ITER-2 G3 -- ERROR PROFILE (rated cells, passthrough disabled)\n\n"
       f"Sampled value vs real rating on {len(errs)} rated cells: MAE={mae:.3f}, |err| dispersion={disp:.3f}, "
       f"pred-true corr={corr:.3f}. LLM reference MAE~0.70, corr~0.5. Too-clean flag={too_clean}. "
       f"**ITER-2 G3 verdict**: {'PASS' if not too_clean else 'FLAG'}. (value channels unchanged from "
       "iter-1; rated cells pass through the true rating in real generation.)\n\n")
    json.dump(dict(n=len(errs), mae=mae, dispersion=disp, corr=corr, too_clean=bool(too_clean)),
              open(f"{DANS}/g3_iter2.json", "w"), indent=1)
    print(f"[g3_2] wrote {DANS}/g3_iter2.json", flush=True)


# ---------------------------------------------------------------- ADDENDUM: era-pocket user check
def stage_erapocket():
    """Direct test that per-question territory matching fixes what a global dial cannot: LOUO item
    knowledge accuracy for era-pocket users (rated-era mass >= 1.5 decades from the bank's era mass --
    the user-62120 type) WITH vs WITHOUT the era features (item decade_align density + era_dist distance)."""
    uni = Universe()
    users = load_173(uni)
    acc = collect(uni, users)
    print(f"\n==== ADDENDUM: era-pocket check (bank era mass = {uni.bank_era_mean:.1f}) ====", flush=True)
    era_mass = {}
    for rec in users:
        decs = [uni.decade[j] for j in rec["known"] if uni.decade[j] > 0]
        era_mass[rec["u"]] = float(np.mean(decs)) if decs else uni.bank_era_mean
    pocket = {u for u, m in era_mass.items() if abs(m - uni.bank_era_mean) >= 15.0}
    print(f"[erapocket] {len(pocket)}/{len(users)} users are era-pocket (|rated-era - bank-era| >= 15y)",
          flush=True)
    a = acc["item"]; grp = a["grp"]; y = a["ky"].astype(int)
    mk_full = fit_know_channel("item", a, has_gamma=False)                    # WITH era features
    a_red = dict(a); a_red["lin"] = np.delete(a["lin"], [5, N_ITEM_KNOW - 1], axis=1)  # drop dec_align+era_dist
    lo, hi = UVEC_SLICE["item"]
    mk_red = fit_know_channel("item", a_red, has_gamma=False,
                              ublock=list(range(lo - 1, hi - 1)))             # user block shifts left by 1
    pf = mk_full["_pred"]; pr = mk_red["_pred"]
    pm = np.array([g in pocket for g in grp]); npm = ~pm
    def a_on(mask, p): return float((p[mask] == y[mask]).mean()) if mask.sum() else float("nan")
    rows = [("era-pocket users", int(pm.sum()), a_on(pm, pr), a_on(pm, pf)),
            ("non-pocket users", int(npm.sum()), a_on(npm, pr), a_on(npm, pf)),
            ("all", len(y), a_on(np.ones(len(y), bool), pr), a_on(np.ones(len(y), bool), pf))]
    print("  subset             n_cells   acc(no-era)  acc(+era)   delta", flush=True)
    for nm, n, r0, r1 in rows:
        print(f"  {nm:18s} {n:8d}   {r0:.3f}       {r1:.3f}     {r1-r0:+.3f}", flush=True)
    md("## ADDENDUM -- era-pocket user check (territory matching vs a global dial)\n\n"
       f"Bank era mass = {uni.bank_era_mean:.1f}. Era-pocket users = rated-era mass >= 1.5 decades away "
       f"({len(pocket)}/{len(users)} users; the user-62120 type: literate but off the bank's modern era). "
       "LOUO item-knowledge accuracy WITHOUT era features (decade_align density + era_dist distance dropped) "
       "vs WITH:\n\n"
       "| subset | n cells | acc (no era) | acc (+era) | delta |\n|---|--:|--:|--:|--:|\n" +
       "".join(f"| {nm} | {n} | {r0:.3f} | {r1:.3f} | {r1-r0:+.3f} |\n" for nm, n, r0, r1 in rows) +
       ("\nThe era features lift era-pocket users more than the population -- territory matching, not a "
        "global knowledgeability dial.\n\n" if rows[0][3] - rows[0][2] > max(0.005, rows[2][3] - rows[2][2])
        else "\n**Verdict: NULL** -- no era-feature accuracy lift for era-pocket users; hypothesis not "
        "supported on this bank (reported without softening).\n\n"))
    json.dump(dict(bank_era_mean=uni.bank_era_mean, n_pocket=len(pocket),
                   rows=[(nm, n, r0, r1) for nm, n, r0, r1 in rows]),
              open(f"{DANS}/erapocket_iter2.json", "w"), indent=1, default=float)
    print(f"[erapocket] wrote {DANS}/erapocket_iter2.json", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True,
                    choices=["fit2", "g1_2", "g2_2", "g3_2", "erapocket", "all"])
    a = ap.parse_args()
    stages = ["fit2", "g1_2", "g2_2", "g3_2", "erapocket"] if a.stage == "all" else [a.stage]
    for st in stages:
        globals()[f"stage_{st}"]()
