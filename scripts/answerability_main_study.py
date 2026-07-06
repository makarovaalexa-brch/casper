"""ANSWERABILITY MAIN STUDY (post-gate; both cross-checks PASSED).

Per ANSWERABILITY_RESULTS_LOG "NEXT" + PLAN Q-B/Q-D. The gate judged a ~91-question BANK to test
FUEL; the agent needs answerability for ANY item/concept. So this pass:

 (a) Judges the FULL breadth-tiered concept set (all 200 genome tags) x the study users.
 (b) Judges a LARGE stratified ITEM sample: a global pool (popularity strata x genre coverage,
     ~1800 distinct items) ROTATED across users + per-user taste-adjacent items => thousands of
     distinct items in the union, cached + resumable (gate cache pattern; never re-calls cached).
 (c) FITS P(answerable | features) — features: pop_pct, log_ratings_count, release_decade,
     genre_match_to_profile, franchise/sequel_flag, is_concept; logistic; VALIDATED on HELD-OUT
     USERS (fit on train users only) => AUC + calibration curve. This fitted model = the SCALE
     SURROGATE (LLM-derived plumbing per Q-D, NEVER the independent structural witness).
 (d) Masked-item validation across the larger sample -> primary value predictor (LLM vs EASE MAE)
     + counterfactual-channel fidelity sigma.

Model + split PINNED (gpt-5.4-mini, answerer split seed 123, same as the gate). Reuses the gate's
already-paid interview judgments as additional fitting data (same judge/model/split). Batches +
caches everything; logs response.usage + running $.

Run collection (chunked, resumable):  python scripts/answerability_main_study.py --collect --limit 50
Run fit + value (all cached):         python scripts/answerability_main_study.py --analyze
"""
import os, sys, re, json, time, argparse, collections
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import warnings
warnings.filterwarnings("ignore")
import llm_answerability_gate as G   # reuse load_data, prompts, _call_json, feature helpers

# ---- pins (inherit the gate's) ----
ANSWERER_SPLIT_SEED = G.ANSWERER_SPLIT_SEED     # 123
USER_SAMPLE_SEED = G.USER_SAMPLE_SEED           # 0
N_CONCEPTS = 200                # ALL genome tags (breadth-tiered)
POOL_TARGET = 1800             # global stratified item pool size (distinct)
POOL_PER_USER = 50             # rotating stratified slice per user
TASTE_ADJ_PER_USER = 20        # per-user taste-adjacent items
MASK_PER_USER = 12             # masked known-half items for value validation (gate used 8)

GRID_CACHE = ".cache/instrument2/answerability_mainstudy_grid.json"
USAGE_SIDE = ".cache/instrument2/answerability_mainstudy_usage.json"
PMODEL_JSON = ".cache/instrument2/answerability_pmodel.json"
PMODEL_NPZ = ".cache/instrument2/answerability_pmodel.npz"
OUT_JSON = "experiments/answerability_mainstudy_results.json"

FRANCHISE_RE = re.compile(r"(\b(II|III|IV|VI|VII|VIII|IX|XI|XII)\b|:|\bPart\b|\bChapter\b|"
                          r"\bEpisode\b|\bVol\b|\b[2-9]\b)", re.I)
YEAR_RE = re.compile(r"\((\d{4})\)")


# ----------------------------------------------------------------------------- banks
def item_year(title):
    m = YEAR_RE.search(title or "")
    return int(m.group(1)) if m else None


def is_franchise(title):
    base = YEAR_RE.sub("", title or "")
    return 1 if FRANCHISE_RE.search(base) else 0


def full_concept_bank(D):
    """ALL genome tags, breadth-tiered by coverage (broad/mid/niche)."""
    cov = D["concepts"]["coverage"]; names = D["concepts"]["names"]
    order = np.argsort(-cov)
    thirds = np.array_split(order, 3)
    label = {}
    for tag, arr in zip(["broad", "mid", "niche"], thirds):
        for ci in arr:
            label[int(ci)] = tag
    return [(int(ci), str(names[ci]), label[int(ci)]) for ci in order]


def global_item_pool(D, target=POOL_TARGET):
    """Distinct items stratified by popularity tier x primary genre (rotated across users)."""
    rng = np.random.default_rng(20260706)
    per_cell = max(1, target // (3 * len(G.GENRES)))
    pool = []
    for tier in ["famous", "moderate", "obscure"]:
        idx = np.where(D["tier"] == tier)[0]
        rng.shuffle(idx)
        by_g = collections.defaultdict(list)
        for j in idx:
            by_g[D["dgen"][j][0]].append(int(j))
        for g, lst in by_g.items():
            pool.extend((j, tier) for j in lst[:per_cell])
    # dedup, stable
    seen = set(); out = []
    for j, t in pool:
        if j not in seen:
            seen.add(j); out.append((j, t))
    rng.shuffle(out)
    return out


def user_item_slice(pool, uindex, k=POOL_PER_USER):
    n = len(pool)
    base = (uindex * k) % n
    return [pool[(base + i) % n] for i in range(k)]


# ----------------------------------------------------------------------------- one user LLM pass
def run_answer(D, split, u, concepts, items):
    kn, ho = split[u]; rat = dict(D["rat_by_u"][u])
    known_ids = sorted(kn)
    prof_lines = G.profile_text(D, known_ids, rat)
    n_shown, N_total = len(prof_lines), len(dict(D["rat_by_u"][u]))
    Q = []
    for ci, name, breadth in concepts:
        Q.append(("concept", dict(name=name, breadth=breadth, ctag=ci)))
    for j, tier, sub in items:
        Q.append(("item", dict(j=int(j), tier=tier, sub=sub)))
    lines = []
    for i, (kind, meta) in enumerate(Q):
        if kind == "concept":
            lines.append(f"{i}: [concept] Could this viewer state a preference about "
                         f"\"{meta['name']}\" films?")
        else:
            lines.append(f"{i}: [item] Has this viewer seen / could they give a real opinion on "
                         f"'{D['title'][meta['j']]}'?")
    prof = "; ".join(prof_lines) if prof_lines else "(no ratings shown)"
    user_msg = (f"Viewer's rating sample (size {n_shown} of {N_total} total; they have seen many more "
                f"films than listed):\n{prof}\n\nQuestions:\n" + "\n".join(lines))
    arr = G._call_json(G.SYS_ANSWER, user_msg)
    ans = {}
    for o in arr:
        if isinstance(o, dict) and "q" in o:
            try:
                ans[int(o["q"])] = o
            except (ValueError, TypeError):
                pass
    return Q, ans


def run_mask(D, split, u, kmax=MASK_PER_USER):
    kn, ho = split[u]; rat = dict(D["rat_by_u"][u])
    known_ids = sorted(kn)
    rng = np.random.default_rng(3000 + u)
    cand = known_ids[:]; rng.shuffle(cand); masked = cand[:kmax]
    prof_lines = G.profile_text(D, known_ids, rat, exclude=masked)
    n_shown, N_total = len(prof_lines), len(dict(D["rat_by_u"][u]))
    lines = [f"{i}: '{D['title'][j]}'" for i, j in enumerate(masked)]
    prof = "; ".join(prof_lines) if prof_lines else "(no ratings shown)"
    user_msg = (f"Viewer's rating sample (size {n_shown} of {N_total} total; they have seen many more "
                f"films than listed):\n{prof}\n\nPredict this viewer's rating for each target film:\n"
                + "\n".join(lines))
    arr = G._call_json(G.SYS_MASK, user_msg)
    preds = {}
    for o in arr:
        if isinstance(o, dict) and "q" in o and "pred_value" in o:
            try:
                preds[int(o["q"])] = float(o["pred_value"])
            except (ValueError, TypeError):
                pass
    rows = []
    for i, j in enumerate(masked):
        if i in preds:
            rows.append(dict(j=int(j), true=float(rat[j]), pred=float(preds[i])))
    return rows


# ----------------------------------------------------------------------------- collection driver
def collect(limit=0, chunk=10):
    D = G.load_data()
    split = G.build_split(D)
    grid_gate = json.load(open(G.GRID_CACHE))["users"]
    study_users = sorted(int(u) for u in grid_gate)   # SAME 300 study users as the gate
    concepts = full_concept_bank(D)
    pool = global_item_pool(D)
    print(f"[main] {len(study_users)} study users | {len(concepts)} concepts | pool {len(pool)} items "
          f"| slice {POOL_PER_USER}+taste {TASTE_ADJ_PER_USER} | mask {MASK_PER_USER}", flush=True)

    grid = {}
    if os.path.exists(GRID_CACHE):
        b = json.load(open(GRID_CACHE))
        if b.get("split_seed") == ANSWERER_SPLIT_SEED and b.get("model") == G.MODEL:
            grid = b.get("users", {})
            print(f"[cache] resuming {len(grid)} cached users", flush=True)

    new = 0
    for uindex, u in enumerate(study_users):
        if str(u) in grid:
            continue
        if limit and new >= limit:
            break
        kn, ho = split[u]; rat = dict(D["rat_by_u"][u])
        dg_name, dg_vec = G.dominant_genre(D, sorted(kn), rat)
        slice_items = [(j, t, "pool") for j, t in user_item_slice(pool, uindex)]
        taste = [(j, D["tier"][j], "taste_adj") for j, _ in
                 G.taste_adjacent_items(D, sorted(kn), rat, dg_vec, k=TASTE_ADJ_PER_USER)]
        items = slice_items + taste
        Q, ans = run_answer(D, split, u, concepts, items)
        mask = run_mask(D, split, u)
        grid[str(u)] = dict(
            Q=[[k, {kk: (int(vv) if isinstance(vv, np.integer) else vv) for kk, vv in m.items()}]
               for k, m in Q],
            ans={str(i): o for i, o in ans.items()}, mask=mask,
            dom=dg_name, dg_vec=[float(x) for x in dg_vec])
        new += 1
        iv = [i for i, (k, m) in enumerate(Q)]
        yes = sum(1 for i in iv if G.is_yes(ans.get(i, {})))
        print(f"  [{len(grid)} cached | +{new}] user {u}: {len(Q)} Q, yes {yes}/{len(iv)} parsed "
              f"{len(ans)}, mask {len(mask)} | running ${G._USAGE['usd']:.3f}", flush=True)
        if new % chunk == 0:
            _flush(grid)
    _flush(grid)
    # accumulate usage sidecar
    cum = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "usd": 0.0}
    if os.path.exists(USAGE_SIDE):
        cum = json.load(open(USAGE_SIDE))
    if new > 0:
        cum["calls"] += G._USAGE["calls"]; cum["prompt_tokens"] += G._USAGE["prompt_tokens"]
        cum["completion_tokens"] += G._USAGE["completion_tokens"]; cum["usd"] += G._USAGE["usd"]
        cum["snapshot"] = G._USAGE["snapshot"]
        json.dump(cum, open(USAGE_SIDE, "w"), indent=1)
    n_cached = sum(1 for u in study_users if str(u) in grid)
    print(f"[collect] {n_cached}/{len(study_users)} cached (+{new} new); running ${G._USAGE['usd']:.3f}",
          flush=True)
    print("ALL CACHED — run --analyze" if n_cached == len(study_users)
          else "RERUN --collect to continue", flush=True)


def _flush(grid):
    json.dump(dict(split_seed=ANSWERER_SPLIT_SEED, model=G.MODEL, snapshot=G._USAGE["snapshot"],
                   users=grid), open(GRID_CACHE, "w"))


# ----------------------------------------------------------------------------- feature builder
def concept_pop_pct(D):
    mass = D["concepts"]["mass"]
    return mass.argsort().argsort() / (len(mass) - 1)


def build_rows(D, grid_users, concept_pct):
    """Flatten (user, question) judgments -> feature rows. Works for main-study + gate grids."""
    rows = []
    for u_s, rec in grid_users.items():
        u = int(u_s)
        ans = {int(k): v for k, v in rec["ans"].items()}
        dgv = np.asarray(rec.get("dg_vec")) if rec.get("dg_vec") is not None else None
        if dgv is None:
            # gate grid stores no dg_vec; recompute from split known-half
            kn, _ = SPLIT[u]; rat = dict(D["rat_by_u"][u])
            _, dgv = G.dominant_genre(D, sorted(kn), rat)
        nv = np.linalg.norm(dgv)
        for i, (k, m) in enumerate(rec["Q"]):
            if i not in ans:
                continue
            y = 1 if G.is_yes(ans[i]) else 0
            if k == "concept":
                cv = G.concept_genre_vec(D, m["ctag"])
                gm = float(dgv @ cv / nv) if nv > 0 else 0.0
                pop = float(concept_pct[m["ctag"]])
                lrc = float(np.log(D["concepts"]["coverage"][m["ctag"]] + 1.0))
                dec = 0.0; fr = 0; isc = 1
            elif k in ("item", "valid_rated", "valid_never"):
                j = m["j"]
                gv = D["Gmat"][j].astype(np.float64); n2 = np.linalg.norm(gv)
                gm = float(dgv @ gv / (nv * n2)) if (nv > 0 and n2 > 0) else 0.0
                pop = float(D["pr"][j])
                lrc = float(np.log(D["cnt"][j] + 1.0))
                yr = item_year(D["title"][j]); dec = ((yr - 1900) / 100.0) if yr else 0.5
                fr = is_franchise(D["title"][j]); isc = 0
            else:
                continue
            rows.append(dict(user=u, kind=k, y=y, pop_pct=pop, log_rcount=lrc,
                             decade=dec, genre_match=gm, franchise=float(fr), is_concept=float(isc)))
    return rows


# ----------------------------------------------------------------------------- fit + validate
FEATS = ["pop_pct", "log_rcount", "decade", "genre_match", "franchise", "is_concept"]


def fit_pmodel(rows):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    X = np.array([[r[f] for f in FEATS] for r in rows], float)
    y = np.array([r["y"] for r in rows], float)
    users = np.array([r["user"] for r in rows])
    mu = X.mean(0); sd = X.std(0) + 1e-9
    Xz = (X - mu) / sd

    # held-out USER split (fit on train users only)
    uniq = np.array(sorted(set(users.tolist())))
    rng = np.random.default_rng(0); rng.shuffle(uniq)
    n_eval = max(1, int(round(0.30 * len(uniq))))
    eval_u = set(uniq[:n_eval].tolist()); train_u = set(uniq[n_eval:].tolist())
    tr = np.array([u in train_u for u in users]); ev = ~tr
    lr = LogisticRegression(penalty="l2", C=1.0, max_iter=3000).fit(Xz[tr], y[tr])
    p_ev = lr.predict_proba(Xz[ev])[:, 1]
    auc_heldout = float(roc_auc_score(y[ev], p_ev))

    # grouped 5-fold CV by user (robustness)
    folds = np.array_split(uniq, 5); umap = {u: k for k, f in enumerate(folds) for u in f}
    fo = np.array([umap[u] for u in users])
    yt, pt = [], []
    for k in range(5):
        trk = fo != k; evk = fo == k
        if y[trk].min() == y[trk].max() or evk.sum() == 0:
            continue
        m = LogisticRegression(penalty="l2", C=1.0, max_iter=3000).fit(Xz[trk], y[trk])
        pt.append(m.predict_proba(Xz[evk])[:, 1]); yt.append(y[evk])
    yt = np.concatenate(yt); pt = np.concatenate(pt)
    auc_cv = float(roc_auc_score(yt, pt))

    # calibration curve on held-out-user eval (10 equal-width prob bins)
    bins = np.linspace(0, 1, 11)
    bi = np.clip(np.digitize(p_ev, bins) - 1, 0, 9)
    calib = []
    for b in range(10):
        msk = bi == b
        if msk.sum() >= 5:
            calib.append(dict(bin=b, p_mean=float(p_ev[msk].mean()),
                              emp=float(y[ev][msk].mean()), n=int(msk.sum())))
    # ECE
    ece = float(sum(c["n"] * abs(c["p_mean"] - c["emp"]) for c in calib) / max(sum(c["n"] for c in calib), 1))

    # final model on ALL data (the deployable surrogate)
    lr_full = LogisticRegression(penalty="l2", C=1.0, max_iter=3000).fit(Xz, y)
    coef = {f: float(c) for f, c in zip(FEATS, lr_full.coef_[0])}
    art = dict(features=FEATS, mean=mu.tolist(), std=sd.tolist(),
               coef=lr_full.coef_[0].tolist(), intercept=float(lr_full.intercept_[0]),
               note="P(answerable)=sigmoid(intercept + coef . (x-mean)/std). LLM-derived scale "
                    "surrogate (Q-D plumbing), NOT the independent structural witness.")
    json.dump(art, open(PMODEL_JSON, "w"), indent=1)
    np.savez(PMODEL_NPZ, mean=mu, std=sd, coef=lr_full.coef_[0],
             intercept=lr_full.intercept_[0], features=np.array(FEATS))
    return dict(n_obs=len(rows), n_users=int(len(uniq)), n_eval_users=n_eval,
                auc_heldout_users=auc_heldout, auc_grouped5cv=auc_cv, ece_heldout=ece,
                calibration=calib, coef_standardized=coef,
                intercept=float(lr_full.intercept_[0]), base_rate=float(y.mean()))


# ----------------------------------------------------------------------------- EASE on the main mask set
def ease_on_masked(D, masked_rows, topk=4000, lam=500.0):
    import scipy.sparse as sp
    d = np.load(G.META)
    uu = d["uu"].astype(np.int64); ii = d["ii"].astype(np.int64); rr = d["rr"].astype(np.float64)
    cnt = d["cnt"].astype(np.float64); ni = int(d["ni"])
    masked_items = set(j for _, j, _ in masked_rows)
    universe = sorted(set(int(x) for x in np.argsort(-cnt)[:topk]) | masked_items)
    uix = {j: k for k, j in enumerate(universe)}; nU = len(universe)
    # drop gate held-out
    heldout = set()
    for u_s in SPLITKEYS:
        for j in SPLIT[int(u_s)][1]:
            heldout.add(int(u_s) * ni + int(j))
    drop = np.isin(uu * ni + ii, np.fromiter(heldout, np.int64, len(heldout)))
    in_uni = np.zeros(ni, bool); in_uni[universe] = True
    keep = in_uni[ii] & (~drop)
    uu_k = uu[keep]; rr_k = rr[keep]
    ii_k = np.array([uix[int(j)] for j in ii[keep]], np.int64)
    M = sp.csr_matrix((np.ones(len(uu_k)), (uu_k, ii_k)), shape=(int(d["nu"]), nU)); M.data[:] = 1.0
    Rs = sp.csr_matrix((rr_k, (uu_k, ii_k)), shape=(int(d["nu"]), nU))
    icnt = np.asarray(M.sum(0)).ravel(); irs = np.asarray(Rs.sum(0)).ravel()
    mu = np.divide(irs, np.maximum(icnt, 1.0)); gmu = float(rr_k.mean()); mu[icnt == 0] = gmu
    Gr = (M.T @ M).toarray().astype(np.float64); Gr[np.diag_indices_from(Gr)] += lam
    P = np.linalg.inv(Gr); B = -P / np.diag(P)[None, :]; np.fill_diagonal(B, 0.0)
    # per-user context = known-half minus that user's masked set
    from collections import defaultdict
    masked_by_u = defaultdict(set)
    for u, j, _ in masked_rows:
        masked_by_u[u].add(j)
    want = set(u for u, _, _ in masked_rows)
    ratmap = defaultdict(dict)
    selu = np.isin(uu, np.fromiter(want, np.int64, len(want)))
    for u, j, r in zip(uu[selu].tolist(), ii[selu].tolist(), rr[selu].tolist()):
        ratmap[u][j] = float(r)
    errs = []; tp = []
    for u, jt, tr in masked_rows:
        if jt not in uix:
            continue
        rc = np.zeros(nU)
        for jj in (set(int(x) for x in SPLIT[u][0]) - masked_by_u[u]):
            if jj in uix and jj in ratmap[u]:
                rc[uix[jj]] = ratmap[u][jj] - mu[uix[jj]]
        pred = float(np.clip(mu[uix[jt]] + rc @ B[:, uix[jt]], 0.5, 5.0))
        errs.append(abs(pred - tr)); tp.append((tr, pred))
    tp = np.array(tp)
    return dict(ease_mae=float(np.mean(errs)),
                ease_corr=float(np.corrcoef(tp[:, 0], tp[:, 1])[0, 1]),
                n=len(errs), universe=nU)


# ----------------------------------------------------------------------------- analyze
SPLIT = None; SPLITKEYS = None


def analyze():
    global SPLIT, SPLITKEYS
    D = G.load_data()
    SPLIT = G.build_split(D)
    if not os.path.exists(GRID_CACHE):
        print("no main-study grid yet; run --collect first"); return
    grid = json.load(open(GRID_CACHE))["users"]
    gate = json.load(open(G.GRID_CACHE))["users"]
    SPLITKEYS = [int(u) for u in gate]     # gate = study users; their held-out excluded from EASE
    concept_pct = concept_pop_pct(D)
    print(f"[analyze] main-study users {len(grid)} | gate users {len(gate)}", flush=True)

    # distinct-item / concept coverage
    main_items = set(); main_concepts = set()
    for rec in grid.values():
        for k, m in rec["Q"]:
            if k == "concept":
                main_concepts.add(m["ctag"])
            else:
                main_items.add(m["j"])
    # fit rows = main study + gate interview judgments (same judge/model/split)
    rows = build_rows(D, grid, concept_pct) + build_rows(D, gate, concept_pct)
    fit = fit_pmodel(rows)
    print(f"  P(answerable) AUC held-out-users={fit['auc_heldout_users']:.3f} "
          f"grouped5CV={fit['auc_grouped5cv']:.3f} ECE={fit['ece_heldout']:.3f} "
          f"(n={fit['n_obs']} over {fit['n_users']} users)", flush=True)

    # masked-value validation (main study mask): LLM MAE vs EASE MAE
    masked_rows = []
    llm_err = []; llm_tp = []
    for u_s, rec in grid.items():
        for m in rec["mask"]:
            masked_rows.append((int(u_s), int(m["j"]), float(m["true"])))
            llm_err.append(abs(m["pred"] - m["true"])); llm_tp.append((m["true"], m["pred"]))
    llm_tp = np.array(llm_tp)
    llm = dict(llm_mae=float(np.mean(llm_err)),
               llm_corr=float(np.corrcoef(llm_tp[:, 0], llm_tp[:, 1])[0, 1]), n=len(llm_err))
    print(f"  masked-value LLM MAE={llm['llm_mae']:.3f} corr={llm['llm_corr']:.3f} (n={llm['n']})",
          flush=True)
    ease = ease_on_masked(D, masked_rows)
    print(f"  masked-value EASE MAE={ease['ease_mae']:.3f} corr={ease['ease_corr']:.3f} "
          f"(n={ease['n']}, universe={ease['universe']})", flush=True)

    usage = json.load(open(USAGE_SIDE)) if os.path.exists(USAGE_SIDE) else {}
    out = dict(
        config=dict(model=G.MODEL, snapshot=usage.get("snapshot", G._USAGE["snapshot"]),
                    answerer_split_seed=ANSWERER_SPLIT_SEED, n_study_users=len(grid),
                    n_concepts_bank=N_CONCEPTS, pool_target=POOL_TARGET,
                    pool_per_user=POOL_PER_USER, taste_adj_per_user=TASTE_ADJ_PER_USER,
                    mask_per_user=MASK_PER_USER),
        coverage=dict(distinct_items_judged=len(main_items), distinct_concepts_judged=len(main_concepts),
                      total_fit_observations=fit["n_obs"], total_fit_users=fit["n_users"]),
        pmodel=fit,
        value=dict(llm=llm, ease=ease,
                   fidelity_sigma_stars=llm["llm_mae"],
                   note="Primary value predictor = LLM masked-rating; EASE = independent-CF "
                        "sensitivity control (Hole-1). Value is sensitivity-only per prereg."),
        cost=dict(calls=usage.get("calls"), prompt_tokens=usage.get("prompt_tokens"),
                  completion_tokens=usage.get("completion_tokens"),
                  usd_estimate=round(usage.get("usd", 0.0), 4),
                  price_in_per_m=G.PRICE_IN_PER_M, price_out_per_m=G.PRICE_OUT_PER_M),
        artifact=dict(pmodel_json=PMODEL_JSON, pmodel_npz=PMODEL_NPZ))
    os.makedirs("experiments", exist_ok=True)
    json.dump(out, open(OUT_JSON, "w"), indent=1, default=str)
    print(f"\n[saved] {OUT_JSON}\n[saved] {PMODEL_JSON}", flush=True)
    print(json.dumps(dict(coverage=out["coverage"],
                          auc_heldout=fit["auc_heldout_users"], auc_cv=fit["auc_grouped5cv"],
                          ece=fit["ece_heldout"], base_rate=fit["base_rate"],
                          llm_mae=llm["llm_mae"], ease_mae=ease["ease_mae"],
                          cost_usd=out["cost"]["usd_estimate"]), indent=1), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--collect", action="store_true")
    ap.add_argument("--analyze", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    if a.collect:
        collect(limit=a.limit)
    elif a.analyze:
        analyze()
    else:
        print("pass --collect [--limit N] or --analyze")
