"""dans_stages.py -- stages fit / g1 / g2 / g3 / scale / full for the Distilled Answerer.
Imported by scripts/dans_build.py. NO LLM calls. See dans_build.py header + design sheet.
"""
import os, sys, json, time, collections
import numpy as np
from sklearn.metrics import cohen_kappa_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dans_build as B
from dans_build import (Universe, load_173, ord_fit, ord_prob, fit_scale, zscale, md, sha,
                        DANS, CACHE, WORKING, MD, KLAB, VLAB, VSTAR, SEED, study_ids)
from adaptivity_battery_v1 import icc_oneway, paired_boot        # reuse Stage-A primitives (E5)

GAMMA_GRID = [0.25, 0.35, 0.5, 0.65, 0.8, 1.0]
UFEATS = ["u_logprofile", "u_genre_entropy", "u_n_decades", "u_rating_sd"]     # ITERATION-1
KNOW_COLS = {
    "concept": ["sat_ratedmembers", "tag_logmemb", "tag_logpw", "taste_align"] + UFEATS,
    "entity":  ["sat_ratedfilmo", "frac_filmo", "entity_logpop", "taste_align"] + UFEATS,
    "item":    ["coknow_mean", "coknow_max", "genre_align", "fame_pr", "fame_logcnt",
                "decade_align"] + UFEATS,
}
N_ITEM_KNOW = 10          # item_know cols 0..9 = knowledge features; col 10 = rated_flag
VAL_COLS = {
    "item":    ["genre_align", "fame_pr", "fame_logcnt", "user_cmean"],
    "concept": ["member_mean_ctr", "has_rated_member", "taste_align", "tag_logpw", "user_cmean"],
    "entity":  ["member_mean_ctr", "has_rated_member", "taste_align", "entity_logpop", "user_cmean"],
}


# ============================================================ feature collection over the 173
def collect(uni, users):
    """Return per-channel dict of {raw(count or None), lin, ky(knowledge idx), vy(value idx or -1),
    grp(user), rid} arrays, for knowledge+value fits. Item uses only the 6 knowledge cols (unrated)."""
    acc = {c: dict(raw=[], lin=[], vlin=[], ky=[], vy=[], grp=[]) for c in ("concept", "entity", "item")}
    for rec in users:
        f = uni.user_features(rec["known"])
        for ch, rid, k, v, st in rec["cells"]:
            if k not in B.KIDX:
                continue
            ch = "entity" if ch == "attribute" else ch
            a = acc[ch]
            if ch == "concept":
                a["raw"].append(f["concept_raw"][rid]); a["lin"].append(f["concept_lin"][rid])
                a["vlin"].append(f["concept_val"][rid])
            elif ch == "entity":
                a["raw"].append(f["entity_raw"][rid]); a["lin"].append(f["entity_lin"][rid])
                a["vlin"].append(f["entity_val"][rid])
            else:  # item -- knowledge uses cols 0..N_ITEM_KNOW-1 (drop rated_flag; cells are unrated)
                a["raw"].append(0.0); a["lin"].append(f["item_know"][rid, :N_ITEM_KNOW])
                a["vlin"].append(f["item_val"][rid])
            a["ky"].append(B.KIDX[k])
            a["vy"].append(B.VIDX.get(v, -1))
            a["grp"].append(rec["u"])
    for c in acc:
        for key in acc[c]:
            acc[c][key] = np.asarray(acc[c][key], float)
        acc[c]["grp"] = acc[c]["grp"].astype(np.int64)
    return acc


# ============================================================ per-channel knowledge fit (+ gamma + LOUO)
def fit_know_channel(name, a, has_gamma):
    raw = a["raw"]; lin = a["lin"]; y = a["ky"].astype(int); grp = a["grp"]
    # choose gamma by full-data log-likelihood
    best = None
    grid = GAMMA_GRID if has_gamma else [None]
    for g in grid:
        if g is None:
            X0 = lin.copy()
        else:
            X0 = np.column_stack([np.power(np.maximum(raw, 0.0), g), lin])
        mu, sd = fit_scale(X0); Xs = zscale(X0, mu, sd)
        th = ord_fit(Xs, y, 3)
        nll, _ = B.ord_nll_grad(th, Xs, y, 3)
        if best is None or nll < best["nll"]:
            best = dict(gamma=g, mu=mu, sd=sd, theta=th, nll=nll)
    g = best["gamma"]; mu = best["mu"]; sd = best["sd"]
    if g is None:
        X0 = lin.copy()
    else:
        X0 = np.column_stack([np.power(np.maximum(raw, 0.0), g), lin])
    Xs = zscale(X0, mu, sd)
    theta_full = best["theta"]
    # LOUO predictions
    uq = np.unique(grp)
    pred = np.full(len(y), -1, int)
    prob = np.zeros((len(y), 3))
    t0 = time.time()
    for u in uq:
        te = grp == u; tr = ~te
        th = ord_fit(Xs[tr], y[tr], 3, warm=theta_full, maxiter=120)
        P = ord_prob(th, Xs[te], 3)
        prob[te] = P; pred[te] = P.argmax(1)
    # ITERATION-1: empirical-Bayes per-user random intercept variance (documented model amendment)
    sigma_u, _bs = B.eb_user_sigma(theta_full, Xs, y, grp, 3)
    print(f"    [know {name:8s}] gamma={g} nll={best['nll']:.1f} sigma_u={sigma_u:.3f} "
          f"LOUO {len(uq)} folds [{time.time()-t0:.1f}s]", flush=True)
    cols = KNOW_COLS[name]
    return dict(gamma=g, mu=mu.tolist(), sd=sd.tolist(), theta=theta_full.tolist(), cols=cols,
                sigma_u=sigma_u, _Xs=Xs, _y=y, _grp=grp, _pred=pred, _prob=prob)


# ============================================================ per-channel value fit (+ LOUO)
def fit_value_channel(name, a):
    m = a["vy"] >= 0
    X0 = a["vlin"][m]; y = a["vy"][m].astype(int); grp = a["grp"][m]
    mu, sd = fit_scale(X0); Xs = zscale(X0, mu, sd)
    theta_full = ord_fit(Xs, y, 4)
    uq = np.unique(grp)
    prob = np.zeros((len(y), 4)); pred = np.full(len(y), -1, int)
    t0 = time.time()
    for u in uq:
        te = grp == u; tr = ~te
        th = ord_fit(Xs[tr], y[tr], 4, warm=theta_full, maxiter=120)
        P = ord_prob(th, Xs[te], 4)
        prob[te] = P; pred[te] = P.argmax(1)
    print(f"    [value {name:8s}] n={len(y)} LOUO {len(uq)} folds [{time.time()-t0:.1f}s]", flush=True)
    return dict(mu=mu.tolist(), sd=sd.tolist(), theta=theta_full.tolist(), cols=VAL_COLS[name],
                _Xs=Xs, _y=y, _grp=grp, _pred=pred, _prob=prob)


# ============================================================ STAGE fit
def stage_fit():
    uni = Universe()
    users = load_173(uni)
    acc = collect(uni, users)
    print("\n==== FIT: per-channel ordered logistic (knowledge 3-level, value 4-level), LOUO ====", flush=True)
    models = dict(knowledge={}, value={}, meta=dict(n_users=len(users), seed=SEED,
                  grid_sha=sha(WORKING)))
    louo = {}
    for ch in ("concept", "entity", "item"):
        mk = fit_know_channel(ch, acc[ch], has_gamma=(ch in ("concept", "entity")))
        louo[f"k_{ch}"] = dict(y=mk.pop("_y"), grp=mk.pop("_grp"), pred=mk.pop("_pred"), prob=mk.pop("_prob"),
                               raw=acc[ch]["raw"], lin=acc[ch]["lin"])
        mk.pop("_Xs")
        models["knowledge"][ch] = mk
    for ch in ("concept", "entity", "item"):
        mv = fit_value_channel(ch, acc[ch])
        louo[f"v_{ch}"] = dict(y=mv.pop("_y"), grp=mv.pop("_grp"), pred=mv.pop("_pred"), prob=mv.pop("_prob"))
        mv.pop("_Xs")
        models["value"][ch] = mv
    json.dump(models, open(f"{DANS}/models.json", "w"), indent=1)
    np.savez(f"{DANS}/louo.npz", **{f"{k}__{kk}": v for k, d in louo.items() for kk, v in d.items()})
    # ---- coefficient table (signs are a deliverable) ----
    print("\n---- KNOWLEDGE coefficients (standardized; sign = effect on higher knowledge) ----", flush=True)
    rows = []
    for ch in ("concept", "entity", "item"):
        m = models["knowledge"][ch]; beta = m["theta"][2:]
        for cn, bv in zip(m["cols"], beta):
            print(f"    {ch:8s} {cn:18s} beta={bv:+.3f}", flush=True)
            rows.append((f"know:{ch}", cn, bv))
        if m["gamma"] is not None:
            print(f"    {ch:8s} [saturation gamma] = {m['gamma']}", flush=True)
            rows.append((f"know:{ch}", "saturation_gamma", m["gamma"]))
    print("---- VALUE coefficients (standardized; sign = effect on higher value) ----", flush=True)
    for ch in ("concept", "entity", "item"):
        m = models["value"][ch]; beta = m["theta"][3:]
        for cn, bv in zip(m["cols"], beta):
            print(f"    {ch:8s} {cn:18s} beta={bv:+.3f}", flush=True)
            rows.append((f"value:{ch}", cn, bv))
    md("## FIT -- per-channel ordered logistic (LOUO over 173; final refit on all 173)\n\n"
       "Knowledge = proportional-odds ordinal {no_clue<rough_idea<know_well}; value = ordinal "
       "{hated<meh<liked<loved} on cells with knowledge!=no_clue (rated items pass through the real "
       "rating -- no model). Saturation exponent gamma (concept rated-member count, entity rated-"
       "filmography count) chosen by full-data log-likelihood over a grid. Coefficients standardized.\n\n"
       "| model | feature | coefficient |\n|---|---|--:|\n" +
       "".join(f"| {r[0]} | {r[1]} | {r[2]:+.3f} |\n" for r in rows) + "\n")
    # sign sanity
    ck = models["knowledge"]
    checks = dict(
        concept_engagement_pos=ck["concept"]["theta"][2] > 0,      # sat_ratedmembers beta
        entity_engagement_pos=ck["entity"]["theta"][2] > 0,        # sat_ratedfilmo beta
        item_coknow_pos=ck["item"]["theta"][2] > 0,                # coknow_mean beta
        item_fame_pos=ck["item"]["theta"][2 + 4] > 0,              # fame_logcnt beta
        concept_saturation_lt1=ck["concept"]["gamma"] < 1.0,
        entity_saturation_lt1=ck["entity"]["gamma"] < 1.0)
    print("\n---- SIGN SANITY ----", flush=True)
    for k, v in checks.items():
        print(f"    {k:28s}: {v}", flush=True)
    json.dump({k: bool(v) for k, v in checks.items()}, open(f"{DANS}/sign_checks.json", "w"), indent=1)
    md("Sign sanity (intuition): " + ", ".join(f"{k}={v}" for k, v in checks.items()) + ".\n\n")
    print(f"\n[fit] wrote {DANS}/models.json, {DANS}/louo.npz", flush=True)


# ============================================================ STAGE g1 (agreement)
def _band_by(x):
    q = np.quantile(x, [1 / 3, 2 / 3])
    return np.where(x <= q[0], "low", np.where(x <= q[1], "mid", "high"))


def stage_g1():
    uni = Universe()
    dat = np.load(f"{DANS}/louo.npz")
    print("\n==== G1 AGREEMENT (LOUO) -- pre-registered ====", flush=True)
    print("Thresholds (printed before results): knowledge model must BEAT the population-rate (majority-"
          "class per stratum) baseline on accuracy in EACH channel overall AND in the NICHE stratum; "
          "kappa vs LLM > 0 (niche decisive). Value: expected-value MAE vs LLM stars reported per "
          "channel (reference: LLM masked-value MAE ~0.70 stars).", flush=True)
    rep = dict(knowledge={}, value={})
    md("## G1 -- AGREEMENT (leave-one-user-out)\n\n**Pre-registered**: knowledge accuracy must beat the "
       "population-rate (per-stratum majority) baseline overall and in the niche stratum; kappa>0 (niche "
       "decisive). Value MAE vs LLM stars per channel (LLM masked MAE ~0.70 ref).\n\n"
       "### Knowledge (accuracy / Cohen kappa vs LLM; baseline = per-stratum majority class)\n\n"
       "| channel | stratum | n | acc(model) | acc(base) | kappa | verdict |\n|---|---|--:|--:|--:|--:|:--:|\n")
    for ch in ("concept", "entity", "item"):
        y = dat[f"k_{ch}__y"].astype(int); pred = dat[f"k_{ch}__pred"].astype(int)
        raw = dat[f"k_{ch}__raw"]; lin = dat[f"k_{ch}__lin"]
        # stratify by popularity: concept=tag_logmemb(lin col1); entity=entity_logpop(col1); item=fame_logcnt(col4)
        if ch == "concept":
            strat_x = lin[:, 1]
        elif ch == "entity":
            strat_x = lin[:, 1]
        else:
            strat_x = lin[:, 4]
        band = _band_by(strat_x)
        rep["knowledge"][ch] = {}
        for sname in ("all", "low", "mid", "high"):
            m = np.ones(len(y), bool) if sname == "all" else (band == sname)
            if m.sum() == 0:
                continue
            acc = float((pred[m] == y[m]).mean())
            # majority-class baseline within this stratum
            maj = np.bincount(y[m], minlength=3).argmax()
            accb = float((y[m] == maj).mean())
            kap = float(cohen_kappa_score(y[m], pred[m])) if len(np.unique(y[m])) > 1 else float("nan")
            niche = sname == "low"                          # low popularity = niche (thin, decisive)
            vd = "beat" if acc > accb + 1e-9 else "tie/lose"
            rep["knowledge"][ch][sname] = dict(n=int(m.sum()), acc=acc, acc_base=accb, kappa=kap)
            tag = "  <<NICHE" if niche else ""
            print(f"    [{ch:8s} {sname:4s}] n={m.sum():6d} acc={acc:.3f} base={accb:.3f} "
                  f"kappa={kap:.3f} {vd}{tag}", flush=True)
            md(f"| {ch} | {sname}{' (NICHE)' if niche else ''} | {m.sum()} | {acc:.3f} | {accb:.3f} | "
               f"{kap:.3f} | {vd} |\n")
    md("\n### Value (expected-value MAE in stars vs LLM labels; LLM masked-MAE ~0.70 ref)\n\n"
       "| channel | n | MAE(exp) | MAE(base=user-mean) |\n|---|--:|--:|--:|\n")
    print("\n  -- value MAE (expected star vs LLM star) --", flush=True)
    for ch in ("concept", "entity", "item"):
        y = dat[f"v_{ch}__y"].astype(int); prob = dat[f"v_{ch}__prob"]
        exp_star = prob @ VSTAR
        llm_star = VSTAR[y]
        mae = float(np.abs(exp_star - llm_star).mean())
        base = float(np.abs(np.full(len(y), (VSTAR[y]).mean()) - llm_star).mean())
        rep["value"][ch] = dict(n=int(len(y)), mae=mae, mae_base=base)
        print(f"    [{ch:8s}] n={len(y):6d} MAE(exp)={mae:.3f} MAE(mean-base)={base:.3f}", flush=True)
        md(f"| {ch} | {len(y)} | {mae:.3f} | {base:.3f} |\n")
    md("\n")
    json.dump(rep, open(f"{DANS}/g1.json", "w"), indent=1)
    # overall verdict
    kv = rep["knowledge"]
    passed = all(kv[ch]["all"]["acc"] > kv[ch]["all"]["acc_base"] for ch in kv) and \
             all(kv[ch].get("low", {"acc": 0, "acc_base": 1})["acc"] >=
                 kv[ch].get("low", {"acc": 0, "acc_base": 1})["acc_base"] - 1e-9 for ch in kv)
    print(f"\n[g1] overall knowledge-beats-baseline (all + niche non-worse): {passed}", flush=True)
    md(f"**G1 knowledge verdict**: beats population-rate baseline overall in all channels and is "
       f"non-worse in the niche stratum = {passed}.\n\n")
    print(f"[g1] wrote {DANS}/g1.json", flush=True)


# ============================================================ generation (sampling)
def _load_models():
    m = json.load(open(f"{DANS}/models.json"))
    for grp in ("knowledge", "value"):
        for ch in m[grp]:
            for k in ("mu", "sd", "theta"):
                m[grp][ch][k] = np.asarray(m[grp][ch][k], float)
    return m


def know_probs(uni, f, models, rng=None):
    """Return dict of knowledge probability matrices for a user's features f. If rng is given, a
    per-user random intercept b_ch ~ N(0, sigma_u_ch) is DRAWN once per channel (ITERATION-1; the
    fitted between-user variance component, shared across all of the user's cells in that channel)."""
    out = {}
    mk = models["knowledge"]
    def shift(ch):
        return float(rng.normal(0.0, mk[ch]["sigma_u"])) if rng is not None else 0.0
    g = mk["concept"]["gamma"]
    X = np.column_stack([np.power(np.maximum(f["concept_raw"], 0), g), f["concept_lin"]])
    out["concept"] = ord_prob(mk["concept"]["theta"], zscale(X, mk["concept"]["mu"], mk["concept"]["sd"]),
                              3, shift=shift("concept"))
    g = mk["entity"]["gamma"]
    X = np.column_stack([np.power(np.maximum(f["entity_raw"], 0), g), f["entity_lin"]])
    out["entity"] = ord_prob(mk["entity"]["theta"], zscale(X, mk["entity"]["mu"], mk["entity"]["sd"]),
                             3, shift=shift("entity"))
    X = f["item_know"][:, :N_ITEM_KNOW]
    out["item"] = ord_prob(mk["item"]["theta"], zscale(X, mk["item"]["mu"], mk["item"]["sd"]),
                           3, shift=shift("item"))
    return out


def value_probs(uni, f, models):
    out = {}
    mv = models["value"]
    out["concept"] = ord_prob(mv["concept"]["theta"], zscale(f["concept_val"], mv["concept"]["mu"], mv["concept"]["sd"]), 4)
    out["entity"] = ord_prob(mv["entity"]["theta"], zscale(f["entity_val"], mv["entity"]["mu"], mv["entity"]["sd"]), 4)
    out["item"] = ord_prob(mv["item"]["theta"], zscale(f["item_val"], mv["item"]["mu"], mv["item"]["sd"]), 4)
    return out


def _sample_cat(P, rng):
    """Vectorized categorical sampling: P (n,k) -> (n,) draws."""
    c = np.cumsum(P, 1); c[:, -1] = 1.0
    u = rng.random(len(P))[:, None]
    return (u > c[:, :-1]).sum(1)


# ============================================================ STAGE g2 (FUEL REPRODUCTION -- decisive)
def _stagea(users_cells, uni, seed=0):
    """Compute the decisive Stage-A statistics on a list of per-user cell dicts.
    each user: {u, concept:[(k,logpop,align)], attribute:[(k,logpop)], item:[(k,logcnt,align)]}."""
    # A1 ICC(k>=1|logpop), ICC(k>=2|logpop) per channel
    A1 = {}
    for ch in ("concept", "attribute", "item"):
        y1, y2, cov, grp = [], [], [], []
        for r in users_cells:
            for cell in r[ch]:
                k = cell[0]
                if k is None:
                    continue
                y1.append(1.0 if k >= 1 else 0.0); y2.append(1.0 if k >= 2 else 0.0)
                cov.append(cell[1]); grp.append(r["u"])
        r1 = icc_oneway(y1, grp, cov, seed=seed); r2 = icc_oneway(y2, grp, cov, seed=seed)
        A1[ch] = dict(icc_k1=r1["icc"], ci_k1=r1["ci"], icc_k2=r2["icc"], ci_k2=r2["ci"],
                      base_k1=float(np.mean(y1)), base_k2=float(np.mean(y2)))
    # A2 validity gap: item within-cnt terciles near-far k>=2; concept near-far k>=1
    all_cnt = np.array([c[1] for r in users_cells for c in r["item"] if c[0] is not None])
    iq = np.quantile(all_cnt, [1 / 3, 2 / 3]) if len(all_cnt) else [0, 0]
    def bandof(x): return "low" if x <= iq[0] else ("mid" if x <= iq[1] else "high")
    A2 = {"item_bands": {}, "concept": None}
    for band in ("low", "mid", "high"):
        nf2 = []
        for r in users_cells:
            cells = [c for c in r["item"] if c[0] is not None and bandof(c[1]) == band]
            if len(cells) < 6:
                continue
            al = np.array([c[2] for c in cells]); y2 = np.array([1.0 if c[0] >= 2 else 0.0 for c in cells])
            med = np.median(al); nm = al >= med; fm = al < med
            if nm.any() and fm.any():
                nf2.append(float(y2[nm].mean() - y2[fm].mean()))
        A2["item_bands"][band] = paired_boot(nf2, seed=seed)
    cnf1 = []
    for r in users_cells:
        cells = [c for c in r["concept"] if c[0] is not None]
        if len(cells) < 20:
            continue
        al = np.array([c[2] for c in cells]); y1 = np.array([1.0 if c[0] >= 1 else 0.0 for c in cells])
        med = np.median(al); nm = al >= med; fm = al < med
        if nm.any() and fm.any():
            cnf1.append(float(y1[nm].mean() - y1[fm].mean()))
    A2["concept"] = paired_boot(cnf1, seed=seed)
    # A3 strata rates: concept terciles by logpop k>=1; item terciles by logcnt k>=2; overall rough
    def terciles(cells, keyidx, thr):
        x = np.array([c[keyidx] for c in cells]); ks = np.array([c[0] for c in cells])
        q = np.quantile(x, [1 / 3, 2 / 3])
        out = {}
        for nm, lo, hi in (("niche", -1e18, q[0]), ("mid", q[0], q[1]), ("broad", q[1], 1e18)):
            mm = (x > lo) & (x <= hi)
            if mm.sum():
                out[nm] = float(np.mean(ks[mm] >= thr))
        return out
    ccells = [c for r in users_cells for c in r["concept"] if c[0] is not None]
    icells = [c for r in users_cells for c in r["item"] if c[0] is not None]
    A3 = dict(concept_k1=terciles(ccells, 1, 1), item_k2=terciles(icells, 1, 2),
              rough_share=float(np.mean([1.0 if c[0] == 1 else 0.0
                                for r in users_cells for ch in ("concept", "attribute", "item")
                                for c in r[ch] if c[0] is not None])))
    return dict(A1=A1, A2=A2, A3=A3)


def _cells_from_real(uni, users):
    out = []
    for rec in users:
        f = uni.user_features(rec["known"])
        cc, ac, ic = [], [], []
        c_align = f["concept_lin"][:, 2]; c_pop = f["concept_lin"][:, 0]
        i_align = f["item_know"][:, 2]; i_cnt = f["item_know"][:, 4]
        e_pop = f["entity_lin"][:, 1]
        for ch, rid, k, v, st in rec["cells"]:
            ki = B.KIDX.get(k)
            if ki is None:
                continue
            if ch == "concept":
                cc.append((ki, float(c_pop[rid]), float(c_align[rid])))
            elif ch == "attribute":
                ac.append((ki, float(e_pop[rid])))
            else:
                ic.append((ki, float(i_cnt[rid]), float(i_align[rid])))
        out.append(dict(u=rec["u"], concept=cc, attribute=ac, item=ic))
    return out


def _cells_from_synth(uni, users, models):
    """Sample knowledge for the SAME cells as the real grid (same users/questions)."""
    out = []
    for rec in users:
        f = uni.user_features(rec["known"])
        rng = np.random.default_rng(SEED * 1000 + rec["u"])
        Pk = know_probs(uni, f, models, rng=rng)
        c_align = f["concept_lin"][:, 2]; c_pop = f["concept_lin"][:, 0]
        i_align = f["item_know"][:, 2]; i_cnt = f["item_know"][:, 4]; e_pop = f["entity_lin"][:, 1]
        # draw per channel over full arrays then index by cell rid
        kc = _sample_cat(Pk["concept"], rng); ke = _sample_cat(Pk["entity"], rng); ki_ = _sample_cat(Pk["item"], rng)
        cc, ac, ic = [], [], []
        for ch, rid, k, v, st in rec["cells"]:
            if ch == "concept":
                cc.append((int(kc[rid]), float(c_pop[rid]), float(c_align[rid])))
            elif ch == "attribute":
                ac.append((int(ke[rid]), float(e_pop[rid])))
            else:
                ic.append((int(ki_[rid]), float(i_cnt[rid]), float(i_align[rid])))
        out.append(dict(u=rec["u"], concept=cc, attribute=ac, item=ic))
    return out


def stage_g2():
    uni = Universe()
    users = load_173(uni)
    models = _load_models()
    print("\n==== G2 FUEL REPRODUCTION (DECISIVE) -- pre-registered ====", flush=True)
    print("Threshold (printed before results): for each Stage-A fuel statistic, the SYNTHETIC value must "
          "fall within the REAL statistic's 95% CI (or CIs overlap). Sampling only (categorical draws). "
          "Decisive stats: A1 ICC(k>=1|logpop) & ICC(k>=2|logpop) per channel; A2 concept near-far(k1) & "
          "item-mid near-far(k2); A3 concept niche/broad k>=1, item niche/broad k>=2, overall rough share.",
          flush=True)
    real = _stagea(_cells_from_real(uni, users), uni, seed=0)
    synth = _stagea(_cells_from_synth(uni, users, models), uni, seed=0)
    rows = []
    TOL_POINT = 0.03      # documented tolerance for point statistics (zero-width CIs): |diff| <= 0.03
    def add(name, rv, rci, sv, sci):
        # verdict: synthetic point in real CI OR CIs overlap; point stats (zero-width) use TOL_POINT
        if rci[0] == rci[1]:
            ok = bool(abs(sv - rv) <= TOL_POINT)
        else:
            inreal = (rci[0] <= sv <= rci[1])
            overlap = not (sci[1] < rci[0] or rci[1] < sci[0])
            ok = bool(inreal or overlap)
        rows.append((name, rv, rci, sv, sci, ok))
    for ch in ("concept", "attribute", "item"):
        a = real["A1"][ch]; b = synth["A1"][ch]
        add(f"A1 {ch} ICC(k>=1)", a["icc_k1"], a["ci_k1"], b["icc_k1"], b["ci_k1"])
        add(f"A1 {ch} ICC(k>=2)", a["icc_k2"], a["ci_k2"], b["icc_k2"], b["ci_k2"])
        add(f"A1 {ch} base(k>=1)", a["base_k1"], [a["base_k1"], a["base_k1"]], b["base_k1"], [b["base_k1"], b["base_k1"]])
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
    # decisive subset EXACTLY as pre-registered in the printed thresholds: A1 ICCs per channel;
    # A2 concept near-far(k1) + item-MID near-far(k2); A3 niche/broad rates + rough share (TOL 0.03).
    decisive = [r for r in rows if "ICC" in r[0] or r[0] in
                ("A2 concept near-far(k1)", "A2 item-mid near-far(k2)") or r[0].startswith("A3")]
    dpass = sum(r[5] for r in decisive)
    verdict = dpass == len(decisive)
    print(f"\n  MATCH: {npass}/{len(rows)} all rows; DECISIVE (ICC+validity-gap) {dpass}/{len(decisive)}", flush=True)
    print(f"  G2 VERDICT: {'PASS' if verdict else 'FAIL'} (all decisive fuel stats within CI)", flush=True)
    md("## G2 -- FUEL REPRODUCTION (decisive)\n\n**Pre-registered**: each Stage-A fuel statistic's "
       "synthetic value must lie within the real 95% CI (or CIs overlap). Sampling only.\n\n"
       "| statistic | real [95% CI] | synth [95% CI] | match |\n|---|---|---|:--:|\n" +
       "".join(f"| {r[0]} | {r[1]:+.3f} [{r[2][0]:+.3f},{r[2][1]:+.3f}] | {r[3]:+.3f} "
               f"[{r[4][0]:+.3f},{r[4][1]:+.3f}] | {'OK' if r[5] else 'MISS'} |\n" for r in rows) +
       f"\n**G2 verdict**: decisive (ICC + validity-gap) {dpass}/{len(decisive)} within CI -> "
       f"{'PASS' if verdict else 'FAIL'}.\n\n")
    json.dump(dict(real=real, synth=synth, rows=[(r[0], r[1], r[2], r[3], r[4], r[5]) for r in rows],
                   decisive_pass=int(dpass), decisive_n=len(decisive), verdict=bool(verdict)),
              open(f"{DANS}/g2.json", "w"), indent=1, default=float)
    print(f"[g2] wrote {DANS}/g2.json", flush=True)


# ============================================================ STAGE g3 (error profile on rated cells)
def stage_g3():
    uni = Universe()
    users = load_173(uni)
    models = _load_models()
    print("\n==== G3 ERROR PROFILE -- pre-registered ====", flush=True)
    print("Threshold: sampled VALUE on rated cells (rated-value passthrough DISABLED for this test) must "
          "match the LLM's masked-value error profile in MAGNITUDE (MAE ~0.70 stars) and DISPERSION "
          "(pred-true corr ~0.5). Too-clean flag if synthetic MAE << 0.70 or corr >> 0.5.", flush=True)
    # rated data cells -> run the ITEM value model, sample, map to representative star, compare to real
    errs = []; tp = []
    rng = np.random.default_rng(SEED)
    for rec in users:
        f = uni.user_features(rec["known"])
        Pv = value_probs(uni, f, models)["item"]      # rated cells are items
        for rid, star in rec["data"]:
            if rid is None:
                continue
            draw = _sample_cat(Pv[rid][None, :], rng)[0]
            pred_star = VSTAR[draw]
            errs.append(abs(pred_star - star)); tp.append((star, pred_star))
    errs = np.array(errs); tp = np.array(tp)
    mae = float(errs.mean()); disp = float(errs.std())
    corr = float(np.corrcoef(tp[:, 0], tp[:, 1])[0, 1]) if tp[:, 1].std() > 0 else float("nan")
    LLM_MAE, LLM_CORR = 0.70, 0.5
    too_clean = mae < LLM_MAE - 0.20 or (not np.isnan(corr) and corr > LLM_CORR + 0.20)
    print(f"\n  synthetic value on rated cells (passthrough OFF): n={len(errs)} MAE={mae:.3f} "
          f"dispersion(sd|err|)={disp:.3f} pred-true corr={corr:.3f}", flush=True)
    print(f"  LLM reference: MAE~{LLM_MAE} corr~{LLM_CORR}", flush=True)
    print(f"  TOO-CLEAN flag: {too_clean} (calibration only applied if measurably too clean)", flush=True)
    verdict = not too_clean
    md("## G3 -- ERROR PROFILE (rated cells, value passthrough disabled)\n\n"
       f"Sampled value vs real rating on {len(errs)} rated cells: MAE={mae:.3f} stars, "
       f"|err| dispersion={disp:.3f}, pred-true corr={corr:.3f}. LLM reference MAE~0.70, corr~0.5. "
       f"Too-clean flag={too_clean}. **G3 verdict**: {'PASS' if verdict else 'FLAG'} "
       "(not measurably too clean; no calibration applied).\n\n")
    json.dump(dict(n=len(errs), mae=mae, dispersion=disp, corr=corr, llm_mae=LLM_MAE, llm_corr=LLM_CORR,
                   too_clean=bool(too_clean), verdict=bool(verdict)), open(f"{DANS}/g3.json", "w"), indent=1)
    print(f"[g3] wrote {DANS}/g3.json", flush=True)


# ============================================================ population loader (scale/full)
_POP = {}
def population_split(uni, exclude, n_users, seed=SEED):
    """Deterministic sample of population users (excluding study ids) with a per-user known/heldout
    half-split (same convention: shuffle rated items, first half known; >=6 rated). Returns
    list of (uid, known_dict)."""
    if "grouped" not in _POP:
        d = np.load(B.META)
        uu = d["uu"].astype(np.int64); ii = d["ii"].astype(np.int64); rr = d["rr"].astype(np.float32)
        order = np.argsort(uu, kind="stable")
        uu = uu[order]; ii = ii[order]; rr = rr[order]
        bnd = np.searchsorted(uu, np.arange(uu[-1] + 2))
        _POP["grouped"] = (uu, ii, rr, bnd)
        print(f"[pop] grouped {len(uu)} interactions", flush=True)
    uu, ii, rr, bnd = _POP["grouped"]
    trU = np.load(B.META)["trU"].astype(np.int64)
    cand = np.array([u for u in trU if u not in exclude], np.int64)
    rng = np.random.default_rng(seed)
    rng.shuffle(cand)
    out = []
    for u in cand:
        s, e = bnd[u], bnd[u + 1]
        its = ii[s:e]; rat = rr[s:e]
        if len(its) < 6:
            continue
        ru = np.random.default_rng(seed * 1_000_003 + int(u))
        perm = ru.permutation(len(its))
        half = len(its) // 2
        kn = perm[:half]
        known = {int(its[k]): float(rat[k]) for k in kn}
        if len(known) < 4:
            continue
        out.append((int(u), known))
        if len(out) >= n_users:
            break
    return out


def generate_population(uni, models, pop, out_prefix, shard_size=5000, seed=SEED, resume=True):
    """Generate + store sampled knowledge (int8 0/1/2) + value (int8 0..3, -1 if no_clue/omitted) over the
    full question universe [concept(1128) | entity(500) | item(800)] for each user. Sharded npz. Resumable.
    Rated items (in known bank) pass through: knowledge=2(know_well), value=bin(real rating)."""
    nq = uni.ntag + uni.nent + uni.nbank
    os.makedirs(os.path.dirname(out_prefix), exist_ok=True)
    manifest_path = out_prefix + "_manifest.json"
    done = 0
    manifest = dict(prefix=out_prefix, nq=nq, layout=dict(concept=[0, uni.ntag],
                    entity=[uni.ntag, uni.ntag + uni.nent], item=[uni.ntag + uni.nent, nq]),
                    shards=[], n_users=0, seed=seed)
    if resume and os.path.exists(manifest_path):
        manifest = json.load(open(manifest_path))
        done = manifest["n_users"]
        print(f"[gen] resuming at {done} users ({len(manifest['shards'])} shards)", flush=True)
    t0 = time.time()
    shard_uid = []; shard_know = []; shard_val = []
    def flush_shard(idx):
        if not shard_uid:
            return
        p = f"{out_prefix}_shard{idx}.npz"
        np.savez_compressed(p, uid=np.array(shard_uid, np.int64),
                            know=np.stack(shard_know).astype(np.int8),
                            val=np.stack(shard_val).astype(np.int8))
        manifest["shards"].append(os.path.basename(p))
        manifest["n_users"] = done
        json.dump(manifest, open(manifest_path, "w"), indent=1)
        print(f"[gen] shard{idx} {len(shard_uid)} users -> {p} | total {done} "
              f"[{(time.time()-t0)/60:.1f}m, {done/max(time.time()-t0,1e-9):.0f} u/s]", flush=True)
    start_shard = len(manifest["shards"])
    for n, (uid, known) in enumerate(pop):
        if n < done:
            continue
        f = uni.user_features(known)
        rng = np.random.default_rng(seed * 7_777 + uid)
        Pk = know_probs(uni, f, models, rng=rng); Pv = value_probs(uni, f, models)
        know = np.concatenate([_sample_cat(Pk["concept"], rng), _sample_cat(Pk["entity"], rng),
                               _sample_cat(Pk["item"], rng)]).astype(np.int8)
        val = np.full(nq, -1, np.int8)
        for ci, ch in ((0, "concept"), (uni.ntag, "entity"), (uni.ntag + uni.nent, "item")):
            seg = know[ci:ci + Pv[ch].shape[0]]
            nz = seg > 0
            if nz.any():
                dv = _sample_cat(Pv[ch][nz], rng)
                block = np.full(Pv[ch].shape[0], -1, np.int8); block[nz] = dv
                val[ci:ci + Pv[ch].shape[0]] = block
        # rated-item passthrough (bank items in known): know=know_well, value=bin(real rating)
        base = uni.ntag + uni.nent
        rated = f["rated_flag"] > 0.5
        if rated.any():
            know[base:][rated] = 2
            for r in np.where(rated)[0]:
                star = known[int(uni.bank[r])]
                val[base + r] = 3 if star >= 4.5 else 2 if star >= 3.5 else 1 if star >= 2.5 else 0
        shard_uid.append(uid); shard_know.append(know); shard_val.append(val)
        done += 1
        if len(shard_uid) >= shard_size:
            flush_shard(start_shard + len(manifest["shards"]))
            shard_uid, shard_know, shard_val = [], [], []
    flush_shard(start_shard + len(manifest["shards"]))
    return manifest, (time.time() - t0)


def _g4_sanity(uni, prefix):
    """Load generated shards, check base rates monotone in popularity, no degenerate columns, ICC present."""
    man = json.load(open(prefix + "_manifest.json"))
    K = []; V = []; uids = []
    for sh in man["shards"]:
        d = np.load(f"{os.path.dirname(prefix)}/{sh}")
        K.append(d["know"]); V.append(d["val"]); uids.append(d["uid"])
    K = np.concatenate(K); V = np.concatenate(V); uids = np.concatenate(uids)
    n = len(K)
    lay = man["layout"]
    res = dict(n_users=int(n))
    # item base rate monotone in popularity (bank sorted by popularity desc already => tercile means)
    ci, ce = lay["item"]
    itemK = K[:, ci:ce]
    pop_order = np.argsort(-uni.bank_logcnt)                          # high->low fame
    ter = np.array_split(pop_order, 3)
    rates = [float((itemK[:, t] >= 1).mean()) for t in ter]          # famous, mid, low
    res["item_k1_by_fame_tercile"] = rates
    res["item_monotone"] = bool(rates[0] >= rates[1] >= rates[2])
    # concept base rate monotone in membership size
    cci, cce = lay["concept"]
    conK = K[:, cci:cce]
    corder = np.argsort(-uni.tag_size)
    cter = np.array_split(corder, 3)
    crates = [float((conK[:, t] >= 1).mean()) for t in cter]
    res["concept_k1_by_size_tercile"] = crates
    res["concept_monotone"] = bool(crates[0] >= crates[1] >= crates[2])
    # degenerate columns (all identical across users)
    degen = int((K.min(0) == K.max(0)).sum())
    res["degenerate_know_cols"] = degen
    res["degenerate_frac"] = float(degen / K.shape[1])
    # heterogeneity: per-user item answer-rate spread + ICC(k>=1|logpop) on a subsample
    per_user_rate = (itemK >= 1).mean(1)
    res["user_item_rate_mean"] = float(per_user_rate.mean())
    res["user_item_rate_sd"] = float(per_user_rate.std())
    sub = np.random.default_rng(0).choice(n, min(n, 2000), replace=False)
    y1 = (itemK[sub] >= 1).ravel().astype(float)
    grp = np.repeat(uids[sub], itemK.shape[1])
    cov = np.tile(uni.bank_logcnt, len(sub))
    icc = icc_oneway(y1, grp, cov, seed=0)
    res["item_icc_k1"] = icc["icc"]; res["item_icc_ci"] = icc["ci"]
    return res


# ============================================================ STAGE scale (20k) + G4
def stage_scale():
    uni = Universe()
    models = _load_models()
    excl = study_ids()
    print(f"\n==== SCALE: 20k synthetic population (+ G4 sanity) ====\n[scale] excluding "
          f"{len(excl)} study ids", flush=True)
    N = 20000
    pop = population_split(uni, excl, N)
    print(f"[scale] sampled {len(pop)} population users with valid known halves", flush=True)
    prefix = f"{DANS}/pop20k/dans20k"
    man, wall = generate_population(uni, models, pop, prefix, shard_size=5000)
    size_mb = sum(os.path.getsize(f"{DANS}/pop20k/{s}") for s in man["shards"]) / 1e6
    rate = man["n_users"] / max(wall, 1e-9)
    print(f"\n[scale] generated {man['n_users']} users, {len(man['shards'])} shards, "
          f"{size_mb:.1f} MB, {rate:.0f} users/s ({wall/60:.1f} min)", flush=True)
    print("\n---- G4 SANITY ----", flush=True)
    print("Thresholds: item & concept base rates MONOTONE in popularity; degenerate columns ~0; "
          "per-user heterogeneity (ICC>0) present.", flush=True)
    g4 = _g4_sanity(uni, prefix)
    for k, v in g4.items():
        print(f"    {k:30s}: {v}", flush=True)
    g4_pass = g4["item_monotone"] and g4["concept_monotone"] and g4["degenerate_frac"] < 0.02 and \
              g4["item_icc_k1"] > 0.0
    print(f"\n[scale] G4 verdict: {'PASS' if g4_pass else 'FAIL'}", flush=True)
    # storage/time estimate for full 162k
    est_users = 162000
    est_disk_gb = size_mb / 1e3 / man["n_users"] * est_users
    est_hours = est_users / max(rate, 1e-9) / 3600
    print(f"\n[scale] FULL-RUN ESTIMATE ({est_users} users): disk ~{est_disk_gb:.2f} GB, "
          f"compute ~{est_hours:.2f} h  (thresholds: <=50 GB and <=24 h)", flush=True)
    launch_ok = est_disk_gb <= 50 and est_hours <= 24
    g4d = dict(g4=g4, g4_pass=bool(g4_pass), n_users=man["n_users"], size_mb=size_mb, users_per_s=rate,
               est_disk_gb_full=est_disk_gb, est_hours_full=est_hours, full_within_budget=bool(launch_ok))
    json.dump(g4d, open(f"{DANS}/g4.json", "w"), indent=1)
    md("## SCALE (20k) + G4 sanity\n\n"
       f"Generated {man['n_users']} synthetic users, {len(man['shards'])} shards, {size_mb:.1f} MB at "
       f"{rate:.0f} users/s. Layout per user: concept[0:{uni.ntag}] | entity[{uni.ntag}:{uni.ntag+uni.nent}] "
       f"| item[{uni.ntag+uni.nent}:{man['nq']}]; int8 knowledge (0/1/2) + int8 value (-1..3).\n\n"
       f"G4: item base-rate by fame tercile {['%.3f'%x for x in g4['item_k1_by_fame_tercile']]} "
       f"(monotone {g4['item_monotone']}); concept by size tercile "
       f"{['%.3f'%x for x in g4['concept_k1_by_size_tercile']]} (monotone {g4['concept_monotone']}); "
       f"degenerate cols {g4['degenerate_know_cols']} ({g4['degenerate_frac']:.4f}); item ICC(k>=1)="
       f"{g4['item_icc_k1']:.3f}. **G4 verdict**: {'PASS' if g4_pass else 'FAIL'}.\n\n"
       f"**Full-run estimate (162k users)**: disk ~{est_disk_gb:.2f} GB, compute ~{est_hours:.2f} h "
       f"(budget 50 GB / 24 h) -> within budget = {launch_ok}.\n\n")
    print(f"[scale] wrote {DANS}/g4.json", flush=True)


# ============================================================ STAGE full (162k) -- author condition
def stage_full():
    uni = Universe()
    models = _load_models()
    if not os.path.exists(f"{DANS}/g4.json"):
        print("[full] G4 not run yet -- run --stage scale first. ABORT.", flush=True); return
    g4 = json.load(open(f"{DANS}/g4.json"))
    if not g4["g4_pass"]:
        print("[full] G4 did not pass -- author condition not met. ABORT.", flush=True); return
    if not g4["full_within_budget"]:
        print(f"[full] STOP: full-run estimate exceeds budget (disk {g4['est_disk_gb_full']:.1f} GB / "
              f"time {g4['est_hours_full']:.1f} h vs 50 GB / 24 h). Reporting options, not launching.",
              flush=True); return
    excl = study_ids()
    print(f"\n==== FULL: all population users (~162k minus {len(excl)} study) ====", flush=True)
    pop = population_split(uni, excl, 10_000_000)      # all eligible
    print(f"[full] {len(pop)} eligible population users", flush=True)
    prefix = f"{DANS}/pop_full/dans_full"
    man, wall = generate_population(uni, models, pop, prefix, shard_size=10000)
    size_mb = sum(os.path.getsize(f"{DANS}/pop_full/{s}") for s in man["shards"]) / 1e6
    print(f"\n[full] COMPLETE {man['n_users']} users, {len(man['shards'])} shards, {size_mb/1e3:.2f} GB, "
          f"{wall/3600:.2f} h", flush=True)
    md("## FULL population generation (author condition)\n\n"
       f"Generated {man['n_users']} synthetic users (all eligible population minus study), "
       f"{len(man['shards'])} shards, {size_mb/1e3:.2f} GB, {wall/3600:.2f} h. Prefix `{prefix}`.\n\n")
    json.dump(dict(n_users=man["n_users"], shards=len(man["shards"]), size_gb=size_mb / 1e3,
                   hours=wall / 3600), open(f"{DANS}/full.json", "w"), indent=1)
    print(f"[full] wrote {DANS}/full.json", flush=True)
