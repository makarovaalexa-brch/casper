"""i25_phase4_a4.py -- A4 "pmodel-blind item prober": a deployable adaptive arm on the FAIR Phase-4
harness that ranks item probes by the VALIDATED answerability model P(answerable|features), with the
one user-specific feature (genre_match) imputed ONLINE from observed answers/refusals.

Imports and REUSES scripts/i25_phase4_fair.py (FA) and scripts/i25_phase4_a3.py (A3) UNMODIFIED
(build_universe, eval_fair, anytime/endpoint, tok_of, cold_ndcg, plan_sched, build_greedy, P4.boot;
A3.build_cooc, A3.make_a3_plan, A3.contrast, A3.first_sep). NO LLM calls; local compute; deterministic.

WHY A4 (vs A3): A3's knowledge signal was raw co-rating cosine (de-popularized -> it wandered into
low-coverage refusal territory; +1.0 hit rate over s3 but LOST -0.037 any@24, extra answers landed on
low-value neighbors). A4 instead uses the fitted answerability surrogate
   .cache/instrument2/answerability_pmodel.{json,npz}  (AUC .921 held-out users)
   features = [pop_pct, log_rcount, decade, genre_match, franchise, is_concept]
   log_rcount coef +1.55 (DOMINANT popularity term), genre_match coef +0.63 (user-specific tilt).
All features except genre_match are PUBLIC/STATIC per item. genre_match is imputed ONLINE from a
running per-user genre estimate g-hat -> popularity stays dominant, discovered taste only TILTS the
ranking. This is exactly the signal A3 lacked (A3 threw popularity away via the cosine normalisation).

ARM (BLIND, deployable):
  Pool = the same 160-item top-coverage ladder bank s3/a3 draw from (item probes only).
  Online state g-hat (per-genre evidence vector):
    - init = population genre prior (coverage-weighted catalog genre mass) with LOW evidence weight w0.
    - ANSWERED item (any polarity -- a dislike still proves knowledge): g-hat += Gmat[j] (raw multi-hot).
    - REFUSED item: g-hat -= REFUSAL_SCALE * p_hat_chosen * Gmat[j], then clip >=0 (surprising refusals,
      i.e. high predicted p_hat, subtract MORE).
  Per-turn selection: for each UNASKED bank item j,
    p_hat(j) = pmodel(pop_pct_j, log_rcount_j, decade_j, genre_match(j|g-hat), franchise_j, is_concept=0),
    genre_match(j|g-hat) = cos(g-hat_unit, Gmat[j]);
    pick argmax  p_hat(j) * V(j),  V(j) = study-cohort coverage prior (pop_rate) -- the SAME coverage
    prior s3's pool ranking and a3's coverage_prior used. Tie-break: higher V, then stable cid.

  Variants:
    a4-blind    : as above (deployable).
    a4-explore  : same + small multiplicative bonus (1 + EXPLORE_W*anneal_t*novelty(j)) for items whose
                  genre region carries LOW g-hat evidence mass; anneal_t = max(0, 1 - t/EXPLORE_TURNS)
                  (directed exploration, annealed off after ~6 turns). Deployable.

PRE-REGISTERED READ (printed): a4 WINS iff it beats s3 with CI excl 0 at T=24 anytime. If a4 lifts the
hit rate toward a3-table's 12.9/24 but still loses on NDCG, report where the value leaks. If hit rate
does NOT rise above a3-blind's 4.4, the online g-hat is not learning -> report g-hat convergence
(cosine of g-hat to the user's true known-half genre distribution -- PRIVILEGED-INFO DIAGNOSTIC ONLY).

Run:  python scripts/i25_phase4_a4.py
"""
import os, sys, json, time, re
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import i25_phase4_fair as FA     # UNMODIFIED reuse
import i25_phase4_a3 as A3       # UNMODIFIED reuse (build_cooc, make_a3_plan, contrast, first_sep)
P4 = FA.P4
L = FA.L

TMAX = FA.TMAX                   # 24
BUDGETS = FA.BUDGETS             # (8,16,24)
NG = FA.NG                       # 20 genres

# ---- a4 hyper-parameters (deterministic; documented in ASSUMPTIONS) ----
PRIOR_W = 1.0                    # g-hat prior evidence weight (~one pseudo-item of population genre mass)
REFUSAL_SCALE = 0.5             # refusal subtracts REFUSAL_SCALE * p_hat_chosen * Gmat[j] from g-hat
EXPLORE_W = 0.30               # a4-explore multiplicative novelty bonus weight
EXPLORE_TURNS = 6              # anneal explore bonus off linearly by this turn

PMODEL_JSON = ".cache/instrument2/answerability_pmodel.json"
OUT_JSON = "experiments/I25_phase4_a4.json"
OUT_MD = "experiments/I25_PHASE4_FAIR.md"

# franchise / year regexes (replicated from answerability_main_study.py -- pure functions, no LLM dep)
FRANCHISE_RE = re.compile(r"(\b(II|III|IV|VI|VII|VIII|IX|XI|XII)\b|:|\bPart\b|\bChapter\b|"
                          r"\bEpisode\b|\bVol\b|\b[2-9]\b)", re.I)
YEAR_RE = re.compile(r"\((\d{4})\)")


def item_year(title):
    m = YEAR_RE.search(title or "")
    return int(m.group(1)) if m else None


def is_franchise(title):
    base = YEAR_RE.sub("", title or "")
    return 1 if FRANCHISE_RE.search(base) else 0


def load_pmodel():
    art = json.load(open(PMODEL_JSON))
    feats = list(art["features"])
    assert feats == ["pop_pct", "log_rcount", "decade", "genre_match", "franchise", "is_concept"], feats
    return (np.asarray(art["mean"], float), np.asarray(art["std"], float),
            np.asarray(art["coef"], float), float(art["intercept"]), feats)


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


# ================================================================= a4 per-user trajectory
def make_a4_runner(D, CANDS, item_cids, pmodel, prior_unit, mode):
    """Returns run(rec) -> (plan, diag). mode in {'blind','explore'}.
    plan = list of (token,native|None) over TMAX (fed to FA.eval_fair, same shape as a3/statics).
    diag = per-turn dict arrays: chosen_phat, chosen_ans(bool), gcos(privileged), chosen_j."""
    mu, sd, coef, intc, _ = pmodel
    explore = (mode == "explore")

    nb = len(item_cids)
    col = {c: k for k, c in enumerate(item_cids)}
    jof = np.array([int(CANDS[c]["key"].split(":")[1]) for c in item_cids], dtype=np.int64)
    # ---- static per-item features (public) ----
    pop_pct = D["pr"][jof].astype(np.float64)
    lrc = np.log(D["cnt"][jof].astype(np.float64) + 1.0)
    Gm = D["Gmat"][jof].astype(np.float64)                      # (nb, NG) raw multi-hot
    gn = np.linalg.norm(Gm, axis=1, keepdims=True)
    G_unit = Gm / (gn + 1e-9)                                   # (nb, NG) unit genre vec per item
    dec = np.array([((item_year(D["title"][int(j)]) - 1900) / 100.0)
                    if item_year(D["title"][int(j)]) else 0.5 for j in jof], float)
    fr = np.array([is_franchise(D["title"][int(j)]) for j in jof], float)
    isc = np.zeros(nb)                                          # items: is_concept = 0
    V = np.array([CANDS[c]["pop_rate"] for c in item_cids], float)   # study-cohort coverage prior
    cids_arr = np.array(item_cids, dtype=np.int64)                   # stable tie-break key

    # static z-scored logit contribution (all but genre_match, col 3)
    static_logit = intc
    for k, x in ((0, pop_pct), (1, lrc), (2, dec), (4, fr), (5, isc)):
        static_logit = static_logit + coef[k] * ((x - mu[k]) / sd[k])

    def p_hat(gm_col):
        z3 = (gm_col - mu[3]) / sd[3]
        return _sigmoid(static_logit + coef[3] * z3)

    def run(rec):
        ans = rec["ans_arr"]                                    # global cid -> bool answerable (rated)
        # true known-half genre distribution (PRIVILEGED diagnostic only)
        tg = np.zeros(NG)
        for j, r in rec["known"].items():
            if 0 <= int(j) < D["Gmat"].shape[0]:
                tg = tg + D["Gmat"][int(j)].astype(np.float64)
        tg_unit = tg / (np.linalg.norm(tg) + 1e-9)

        g_hat = PRIOR_W * prior_unit.copy()                     # start at population prior, low weight
        used = np.zeros(nb, bool)
        plan = []
        d_phat, d_ans, d_gcos, d_j = [], [], [], []
        for t in range(TMAX):
            gh_norm = np.linalg.norm(g_hat)
            gh_unit = g_hat / (gh_norm + 1e-9)
            gcos = float(gh_unit @ tg_unit)
            gm_col = G_unit @ gh_unit                           # genre_match(j | g-hat) per item
            ph = p_hat(gm_col)
            score = ph * V
            if explore:
                anneal = max(0.0, 1.0 - t / float(EXPLORE_TURNS))
                if anneal > 0 and gh_norm > 0:
                    g_frac = g_hat / (g_hat.sum() + 1e-9)       # evidence fraction per genre
                    region_mass = (G_unit / (G_unit.sum(1, keepdims=True) + 1e-9)) @ g_frac
                    novelty = np.clip(1.0 - region_mass, 0.0, 1.0)
                    score = score * (1.0 + EXPLORE_W * anneal * novelty)
            score = np.where(used, -1e18, score)
            # deterministic argmax: max score, tie-break higher V then lowest cid
            j_sel = int(np.lexsort((cids_arr, -V, -score))[0])
            used[j_sel] = True
            cid = item_cids[j_sel]
            d_phat.append(float(ph[j_sel])); d_j.append(int(jof[j_sel])); d_gcos.append(gcos)
            if ans[cid]:
                plan.append((FA.tok_of(CANDS, rec, cid), rec["nat_arr"][cid]))
                g_hat = g_hat + Gm[j_sel]                       # answered (any polarity) -> add genre mass
                d_ans.append(True)
            else:
                plan.append((None, None))
                g_hat = np.clip(g_hat - REFUSAL_SCALE * ph[j_sel] * Gm[j_sel], 0.0, None)
                d_ans.append(False)
        diag = dict(phat=np.array(d_phat), ans=np.array(d_ans, bool),
                    gcos=np.array(d_gcos), j=np.array(d_j))
        return plan, diag
    return run


def main():
    t0 = time.time()
    print("[a4] loading data + frozen RecVAE + trained fold ...", flush=True)
    D = L.G.load_data(); FR = L.Frozen(D)
    model = L.Fold(); import torch
    blob = torch.load(P4.CKPT_BEST, map_location="cpu")
    model.load_state_dict(blob["model"]); model.eval()
    P4.T = TMAX
    users, _ = P4.assemble(D, FR)
    n = len(users)
    print(f"[a4] {n} users; building ladder universe (reuse FA.build_universe) ...", flush=True)
    CANDS, meta = FA.build_universe(D, FR, users)
    cold = FA.cold_ndcg(FR, users, 10)

    item_cids = [m["cid"] for m in CANDS if m["kind"] == "item"]
    concept_cids = [m["cid"] for m in CANDS if m["kind"] == "concept"]
    pop_rate = {m["cid"]: m["pop_rate"] for m in CANDS if m["kind"] == "item"}
    item_by_cov = sorted(item_cids, key=lambda c: -CANDS[c]["pop_rate"])
    s3_pool = item_by_cov[:FA.ITEM_POOL_S3]

    # ---- reproduce s1/s3 (existing harness) BEFORE adding anything ----
    print("[a4] single-question values (all candidates) ...", flush=True)
    v1 = FA.single_q_values(FR, model, users, cold, CANDS, [m["cid"] for m in CANDS])
    print("[a4] greedy s1 (concepts) ...", flush=True)
    s1 = FA.build_greedy(FR, model, users, cold, CANDS, concept_cids, TMAX, v1=v1, tag="s1")
    print("[a4] greedy s3 (popular items) ...", flush=True)
    s3 = FA.build_greedy(FR, model, users, cold, CANDS, s3_pool, TMAX, v1=v1, tag="s3")
    pt_s1, ans_s1 = FA.eval_fair(FR, model, users, lambda rec: FA.plan_sched(CANDS, rec, s1), cold)
    pt_s3, ans_s3 = FA.eval_fair(FR, model, users, lambda rec: FA.plan_sched(CANDS, rec, s3), cold)
    s1_any24 = float(FA.anytime(pt_s1, 24).mean())
    s3_any24 = float(FA.anytime(pt_s3, 24).mean())

    # ---- reproduce a3-blind (reuse A3 UNMODIFIED) ----
    print("[a4] building 160x160 population co-known cosine (reuse A3.build_cooc) ...", flush=True)
    bank_j = [int(CANDS[c]["key"].split(":")[1]) for c in item_cids]
    study_uids = [rec["u"] for rec in users]
    C, self_cnt, nU_pop = A3.build_cooc(bank_j, study_uids)
    colof = {c: k for k, c in enumerate(item_cids)}
    global_order = sorted(item_cids, key=lambda c: (-CANDS[c]["pop_rate"], c))
    a3_pf = A3.make_a3_plan(CANDS, item_cids, s3[0], global_order, colof, C, pop_rate, "blind")
    pt_a3, ans_a3 = FA.eval_fair(FR, model, users, a3_pf, cold)
    a3_any24 = float(FA.anytime(pt_a3, 24).mean())

    print(f"\n[a4] REPRODUCTION: s1 any@24={s1_any24:.4f} (t 0.2103) | s3 any@24={s3_any24:.4f} "
          f"(t 0.2786) | a3-blind any@24={a3_any24:.4f} (t 0.2412)", flush=True)
    ok = (abs(s1_any24 - 0.2103) < 0.0015 and abs(s3_any24 - 0.2786) < 0.0015
          and abs(a3_any24 - 0.2412) < 0.0020)
    if not ok:
        print("[a4] STOP: s1/s3/a3-blind did NOT reproduce within tolerance -- NOT adding a4. Diagnose first.",
              flush=True)
        return
    print("[a4] reproduction OK -- proceeding to build a4.\n", flush=True)

    # ---- pre-registered read print ----
    s3_hit = float(ans_s3.mean()); a3_hit = float(ans_a3.mean())
    print("=" * 78, flush=True)
    print("PRE-REGISTERED READ (a4 pmodel-blind prober):", flush=True)
    print("  a4 WINS iff it beats s3 with CI excl 0 at T=24 anytime.", flush=True)
    print(f"  Hit-rate anchors: s3 {s3_hit:.1f}/24 | a3-blind {a3_hit:.1f}/24 | a3-table 12.9/24 (ceiling).",
          flush=True)
    print("  If hit rises toward 12.9 but NDCG loses -> report where value leaks.", flush=True)
    print("  If hit does NOT rise above a3-blind 4.4 -> g-hat not learning; report g-hat convergence.",
          flush=True)
    print("=" * 78 + "\n", flush=True)

    # ---- population genre prior (coverage-weighted catalog genre mass over the bank) ----
    jof = np.array(bank_j, dtype=np.int64)
    Gm_bank = D["Gmat"][jof].astype(np.float64)
    Vw = np.array([CANDS[c]["pop_rate"] for c in item_cids], float)
    prior_raw = (Vw[:, None] * Gm_bank).sum(0)
    prior_unit = prior_raw / (np.linalg.norm(prior_raw) + 1e-9)
    pmodel = load_pmodel()
    print(f"[a4] pmodel loaded (intercept {pmodel[3]:+.3f}, coef {np.round(pmodel[2],3).tolist()}); "
          f"prior top-genre idx {int(prior_unit.argmax())} mass {prior_unit.max():.3f}\n", flush=True)

    # ---- run a4 arms (precompute plans+diags once per user) ----
    pt, ansn, diags = {}, {}, {}
    for mode in ("blind", "explore"):
        print(f"[a4] eval a4-{mode} ...", flush=True)
        runner = make_a4_runner(D, CANDS, item_cids, pmodel, prior_unit, mode)
        plans = {}; dg = {}
        for rec in users:
            pl, d = runner(rec)
            plans[rec["u"]] = pl; dg[rec["u"]] = d
        pt[mode], ansn[mode] = FA.eval_fair(FR, model, users, lambda rec: plans[rec["u"]], cold)
        diags[mode] = dg

    # ---- contrasts (reuse A3 helpers) ----
    vs_s3 = {m: A3.contrast(pt[m], pt_s3) for m in pt}
    vs_a3 = {m: A3.contrast(pt[m], pt_a3) for m in pt}
    sep_s3 = {m: A3.first_sep(pt[m], pt_s3) for m in pt}

    def row(name, ptm, ansm):
        r = dict(name=name, mean_ans_turns=float(ansm.mean()), hit_rate=float(ansm.mean() / TMAX))
        for T in BUDGETS:
            r[f"any@{T}"] = float(FA.anytime(ptm, T).mean())
            r[f"end@{T}"] = float(FA.endpoint(ptm, T).mean())
        return r
    rows = {m: row(f"a4-{m}", pt[m], ansn[m]) for m in pt}
    rows["s3 popular-item"] = row("s3 popular-item", pt_s3, ans_s3)
    rows["a3-blind"] = row("a3-blind", pt_a3, ans_a3)
    rows["s1 concepts"] = row("s1 concepts", pt_s1, ans_s1)

    blind_hit = rows["blind"]["mean_ans_turns"]
    hit_rose_vs_s3 = blind_hit > s3_hit + 1e-6
    hit_rose_vs_a3 = blind_hit > a3_hit + 1e-6
    blind_wins = (vs_s3["blind"][24]["ci"][0] > 0)          # pre-registered: beats s3 CI excl 0 at T=24

    # ---- calibration diagnostic (per turn: mean chosen p_hat vs realized answer rate) ----
    def calib(dg):
        ph = np.array([dg[u]["phat"] for u in [r["u"] for r in users]])   # (n, TMAX)
        an = np.array([dg[u]["ans"] for u in [r["u"] for r in users]]).astype(float)
        return ph.mean(0), an.mean(0)
    calib_phat, calib_ans = calib(diags["blind"])
    calib_gap = float(np.mean(calib_phat - calib_ans))
    calib_corr = float(np.corrcoef(calib_phat, calib_ans)[0, 1])

    # ---- g-hat convergence diagnostic (PRIVILEGED-INFO ONLY) ----
    gcos_blind = np.array([diags["blind"][u]["gcos"] for u in [r["u"] for r in users]]).mean(0)

    # ---- where does value leak? answered-but-low-value: mean V of answered probes per turn ----
    #     (a4-blind answered probes' coverage vs s3 answered probes' coverage)
    def answered_cov(dg):
        cov_by_t = np.full(TMAX, np.nan)
        for t in range(TMAX):
            vals = []
            for u in [r["u"] for r in users]:
                if dg[u]["ans"][t]:
                    jj = int(dg[u]["j"][t]); vals.append(float(D["pr"][jj]))
            cov_by_t[t] = np.mean(vals) if vals else np.nan
        return cov_by_t
    a4_ans_cov = answered_cov(diags["blind"])

    # ---- console summary ----
    print("\n==== A4 pmodel-blind prober (any/end NDCG@10; hit = mean answered turns / 24) ====", flush=True)
    for m in ("s3 popular-item", "a3-blind", "s1 concepts"):
        r = rows[m]
        print(f"  {m:16s} any24 {r['any@24']:.4f} end24 {r['end@24']:.4f} | hit {r['mean_ans_turns']:.1f}/24",
              flush=True)
    for m in ("blind", "explore"):
        r = rows[m]; c3 = vs_s3[m]; ca = vs_a3[m]
        print(f"  a4-{m:12s} any24 {r['any@24']:.4f} end24 {r['end@24']:.4f} | hit {r['mean_ans_turns']:.1f}/24 "
              f"| vs s3@24 {c3[24]['delta']:+.4f}[{c3[24]['ci'][0]:+.4f},{c3[24]['ci'][1]:+.4f}] "
              f"| vs a3@24 {ca[24]['delta']:+.4f}[{ca[24]['ci'][0]:+.4f},{ca[24]['ci'][1]:+.4f}]", flush=True)
    print(f"\n  HIT ROSE vs s3 ({blind_hit:.1f}>{s3_hit:.1f})? {hit_rose_vs_s3} | "
          f"vs a3-blind ({blind_hit:.1f}>{a3_hit:.1f})? {hit_rose_vs_a3}", flush=True)
    print(f"  a4-blind WINS (beats s3 CI excl 0 @T24)? {blind_wins}", flush=True)
    for T in BUDGETS:
        c = vs_s3["blind"][T]
        print(f"  a4-blind vs s3 @T{T}: {c['delta']:+.4f}[{c['ci'][0]:+.4f},{c['ci'][1]:+.4f}]", flush=True)
    print(f"  CALIBRATION: mean chosen p_hat vs realized answer-rate gap {calib_gap:+.3f} "
          f"(corr {calib_corr:+.2f}); p_hat@t1 {calib_phat[0]:.3f} vs ans@t1 {calib_ans[0]:.3f}", flush=True)
    print(f"  g-hat convergence (PRIV): cos@t1 {gcos_blind[0]:.3f} -> cos@t24 {gcos_blind[-1]:.3f} "
          f"(delta {gcos_blind[-1]-gcos_blind[0]:+.3f})", flush=True)

    # ---- verdict text ----
    if blind_wins:
        verdict = ("A4 pmodel-blind BEATS the best static s3 (CI excl 0 at T=24 anytime) -- the validated "
                   "answerability surrogate with online genre_match captures part of the Branch-B prize.")
    elif hit_rose_vs_a3:
        verdict = (f"A4-blind RAISES the hit rate ({blind_hit:.1f} vs a3-blind {a3_hit:.1f}, s3 {s3_hit:.1f}/24) "
                   "but does NOT beat s3 on NDCG at T=24 -- extra answers still land on lower-value items; hit "
                   "does not convert. Branch B stands.")
    else:
        verdict = (f"A4-blind does NOT raise the hit rate above a3-blind ({blind_hit:.1f} vs {a3_hit:.1f}/24) -- "
                   "the online g-hat is not learning enough per-user answerability tilt to help. Branch B stands.")
    print(f"\n  VERDICT: {verdict}", flush=True)

    # ---- persist json ----
    def cdump(cd):
        return {str(T): cd[T] for T in BUDGETS}
    out = dict(
        config=dict(dataset="ML-25M", instrument="RecVAE-d512 + I2.5 learned fold", TMAX=TMAX,
                    budgets=list(BUDGETS), n_users=n, bank_items=len(item_cids),
                    pmodel=PMODEL_JSON, pmodel_coef=pmodel[2].tolist(), pmodel_intercept=pmodel[3],
                    prior_w=PRIOR_W, refusal_scale=REFUSAL_SCALE, explore_w=EXPLORE_W,
                    explore_turns=EXPLORE_TURNS, V_prior="study-cohort coverage (pop_rate)",
                    fold_ckpt=P4.CKPT_BEST, best_val=blob["state"]["best_val"]),
        reproduction=dict(s1_any24=s1_any24, s3_any24=s3_any24, a3_blind_any24=a3_any24,
                          s1_target=0.2103, s3_target=0.2786, a3_target=0.2412, ok=bool(ok)),
        hit_anchors=dict(s3=s3_hit, a3_blind=a3_hit, a3_table=12.9),
        arms={rows[m]["name"]: rows[m] for m in rows},
        vs_s3={f"a4-{m}": cdump(vs_s3[m]) for m in pt},
        vs_a3_blind={f"a4-{m}": cdump(vs_a3[m]) for m in pt},
        first_separation_vs_s3={f"a4-{m}": dict(turn=sep_s3[m][0], sign=sep_s3[m][1]) for m in pt},
        ndcg_curves={f"a4-{m}": [float(x) for x in pt[m].mean(axis=0)] for m in pt},
        calibration=dict(phat_by_turn=[float(x) for x in calib_phat],
                         ans_rate_by_turn=[float(x) for x in calib_ans],
                         mean_gap=calib_gap, corr=calib_corr),
        ghat_convergence_privileged=[float(x) for x in gcos_blind],
        answered_coverage_by_turn=[None if np.isnan(x) else float(x) for x in a4_ans_cov],
        hypothesis=dict(blind_wins=bool(blind_wins), hit_rose_vs_s3=bool(hit_rose_vs_s3),
                        hit_rose_vs_a3=bool(hit_rose_vs_a3), blind_hit=blind_hit,
                        s3_hit=s3_hit, a3_hit=a3_hit),
        verdict=verdict, wall_min=round((time.time() - t0) / 60, 2))
    os.makedirs("experiments", exist_ok=True)
    json.dump(out, open(OUT_JSON, "w"), indent=1, default=str)

    # ---- append MD section ----
    def frow(name, r, cd=None):
        base = (f"| {name} | {r['any@8']:.4f}/{r['end@8']:.4f} | {r['any@16']:.4f}/{r['end@16']:.4f} | "
                f"{r['any@24']:.4f}/{r['end@24']:.4f} | {r['mean_ans_turns']:.1f} ({100*r['hit_rate']:.0f}%) |")
        if cd is not None:
            c = cd[24]; base += f" {c['delta']:+.4f}[{c['ci'][0]:+.4f},{c['ci'][1]:+.4f}] |"
        else:
            base += " -- |"
        return base

    md = []
    md.append("\n## A4 pmodel-blind prober\n\n")
    md.append("Date 2026-07-08. Script `scripts/i25_phase4_a4.py` (imports/reuses `i25_phase4_fair.py` "
              "and `i25_phase4_a3.py` UNMODIFIED). NO LLM calls; deterministic; local compute.\n\n")
    md.append("**Reproduction gate (existing harness, before adding anything):** "
              f"s1 any@24={s1_any24:.4f} (t 0.2103), s3 any@24={s3_any24:.4f} (t 0.2786), "
              f"a3-blind any@24={a3_any24:.4f} (t 0.2412) -> reproduced={ok}.\n\n")
    md.append("**Idea.** A3's co-rating cosine de-popularised the signal and wandered into low-coverage "
              "refusal territory (hit +1.0 but any@24 -0.037 vs s3). A4 instead ranks item probes by the "
              "VALIDATED answerability surrogate `.cache/instrument2/answerability_pmodel` "
              "(AUC .921 held-out users; features [pop_pct, log_rcount, decade, genre_match, franchise, "
              "is_concept]; log_rcount coef +1.55 DOMINANT, genre_match +0.63 the user tilt). All features "
              "except genre_match are public/static; genre_match is imputed ONLINE from a running per-user "
              "genre estimate g-hat. Popularity stays dominant; discovered taste only TILTS the ranking.\n\n")
    md.append("**Arm.** Pool = the same 160-item top-coverage ladder bank s3/a3 draw from (item probes only). "
              "Online g-hat: init = coverage-weighted population genre prior (low weight w0=%.1f); ANSWERED "
              "item (any polarity) += Gmat[j]; REFUSED item -= %.1f*p_hat_chosen*Gmat[j] then clip>=0 "
              "(surprising refusals subtract more). Per turn pick argmax p_hat(j)*V(j), "
              "p_hat=pmodel(., genre_match=cos(g-hat,Gmat[j])), V=study-cohort coverage prior (pop_rate). "
              "a4-explore adds a small annealed novelty bonus (1+%.2f*anneal_t*novelty) for low-g-hat-mass "
              "genre regions, off by turn %d.\n\n" % (PRIOR_W, REFUSAL_SCALE, EXPLORE_W, EXPLORE_TURNS))
    md.append("| arm | any/end @8 | any/end @16 | any/end @24 | hit (ansT) | delta-any@24 vs s3 [CI] |\n"
              "|---|---|---|---|---|---|\n")
    md.append(frow("s3 popular-item (opponent)", rows["s3 popular-item"]) + "\n")
    md.append(frow("a3-blind (prior arm)", rows["a3-blind"]) + "\n")
    md.append(frow("s1 concepts", rows["s1 concepts"]) + "\n")
    md.append(frow("a4-blind", rows["blind"], vs_s3["blind"]) + "\n")
    md.append(frow("a4-explore", rows["explore"], vs_s3["explore"]) + "\n")

    md.append("\n**Contrast a4-blind vs s3 (THE contrast) and vs a3-blind, per budget:**\n\n"
              "| budget T | a4-blind vs s3 [CI] | a4-blind vs a3-blind [CI] |\n|---|---|---|\n")
    for T in BUDGETS:
        c3 = vs_s3["blind"][T]; ca = vs_a3["blind"][T]
        md.append(f"| {T} | {c3['delta']:+.4f}[{c3['ci'][0]:+.4f},{c3['ci'][1]:+.4f}] | "
                  f"{ca['delta']:+.4f}[{ca['ci'][0]:+.4f},{ca['ci'][1]:+.4f}] |\n")

    md.append("\n**Mechanism metric -- hit rate (mean answered turns / 24):** "
              f"s3 = {s3_hit:.1f} (~{100*s3_hit/24:.0f}%); a3-blind = {a3_hit:.1f}; "
              f"a4-blind = {blind_hit:.1f} (~{100*rows['blind']['hit_rate']:.0f}%); "
              f"a4-explore = {rows['explore']['mean_ans_turns']:.1f}; a3-table ceiling = 12.9. "
              f"Hit rose vs s3? **{hit_rose_vs_s3}**. Hit rose vs a3-blind? **{hit_rose_vs_a3}**. "
              f"a4-blind BEATS s3 at T=24 (CI excl 0)? **{blind_wins}**.\n\n")
    fs = sep_s3["blind"]
    md.append(f"**First separation (a4-blind belief(t) vs s3 belief(t), CI excl 0):** "
              f"{('turn ' + str(fs[0]) + ' (sign ' + fs[1] + ')') if fs[0] else 'NEVER within T=24'}.\n\n")

    md.append("**Calibration diagnostic (is the online p_hat honest?): per turn, mean chosen-probe p_hat "
              "vs realized answer rate.**\n\n| t | " + " | ".join(str(t + 1) for t in range(TMAX)) + " |\n")
    md.append("|" + "---|" * (TMAX + 1) + "\n")
    md.append("| mean chosen p_hat | " + " | ".join(f"{x:.3f}" for x in calib_phat) + " |\n")
    md.append("| realized answer rate | " + " | ".join(f"{x:.3f}" for x in calib_ans) + " |\n")
    md.append(f"\nMean p_hat - answer-rate gap = {calib_gap:+.3f} (corr {calib_corr:+.2f}). "
              f"{'p_hat is OPTIMISTIC (over-predicts answerability)' if calib_gap > 0.02 else ('p_hat is PESSIMISTIC' if calib_gap < -0.02 else 'p_hat is well-calibrated in level')}"
              " -- the surrogate ranks, it is not a per-user probability oracle.\n\n")

    md.append("**g-hat convergence (PRIVILEGED-INFO DIAGNOSTIC ONLY -- cos of online g-hat to the user's "
              "true known-half genre distribution; NOT used by the policy):**\n\n| t | "
              + " | ".join(str(t + 1) for t in range(TMAX)) + " |\n")
    md.append("|" + "---|" * (TMAX + 1) + "\n")
    md.append("| cos(g-hat, true genre) | " + " | ".join(f"{x:.3f}" for x in gcos_blind) + " |\n")
    md.append(f"\ncos@t1 {gcos_blind[0]:.3f} -> cos@t24 {gcos_blind[-1]:.3f} "
              f"(delta {gcos_blind[-1]-gcos_blind[0]:+.3f}) -- "
              f"{'g-hat DOES converge toward the true genre profile' if gcos_blind[-1]-gcos_blind[0] > 0.02 else 'g-hat barely moves (few answers to learn from)'}.\n\n")

    md.append("**NDCG@10(t) curves (t=1..24):**\n\n| arm | " + " | ".join(f"t{t+1}" for t in range(TMAX)) + " |\n")
    md.append("|" + "---|" * (TMAX + 1) + "\n")
    for lab, m in (("s3 popular-item", None), ("a3-blind", "A3"), ("a4-blind", "blind"), ("a4-explore", "explore")):
        cur = pt_s3.mean(axis=0) if m is None else (pt_a3.mean(axis=0) if m == "A3" else pt[m].mean(axis=0))
        md.append(f"| {lab} | " + " | ".join(f"{c:.3f}" for c in cur) + " |\n")

    md.append(f"\n**VERDICT:** {verdict}\n\n")
    md.append("**ASSUMPTIONS / judgment calls (a4):**\n"
              "1. Probe bank = the 160 top-coverage ladder items (coverage>=3) s3/a3 draw from; item probes only.\n"
              "2. p_hat = the fitted logistic answerability surrogate (answerability_pmodel; sigmoid(intercept "
              "+ coef.(x-mean)/std)). Features pop_pct=D.pr[j], log_rcount=log(D.cnt[j]+1), decade=(year-1900)/100 "
              "(0.5 if no year), genre_match=cos(g-hat, Gmat[j]), franchise=title-regex, is_concept=0. All but "
              "genre_match are public/static per item.\n"
              "3. g-hat init = coverage-weighted population genre prior (sum_j pop_rate[j]*Gmat[j], unit-normed) "
              f"with evidence weight w0={PRIOR_W} (~one pseudo-item). ANSWERED (any polarity, incl dislikes -- a "
              "dislike still proves knowledge) += raw Gmat[j]. REFUSED -= "
              f"{REFUSAL_SCALE}*p_hat_chosen*Gmat[j], clipped >=0 (down-weight proportional to the p_hat we "
              "predicted -> surprising refusals count more).\n"
              f"4. V(j) = study-cohort coverage prior (pop_rate) -- the SAME coverage prior that defines s3's "
              "pool and a3's coverage_prior. Selection = argmax p_hat(j)*V(j); popularity stays dominant "
              "(V and the pmodel's log_rcount term), genre_match only tilts.\n"
              f"5. a4-explore bonus = (1 + {EXPLORE_W}*anneal_t*novelty(j)), anneal_t=max(0,1-t/{EXPLORE_TURNS}), "
              "novelty(j)=1 - (item genre distribution . g-hat evidence fraction) -- cheap directed exploration, "
              "annealed off after ~6 turns.\n"
              "6. No turn-1 seeding: turn 1 uses g-hat=prior only, so the first pick emerges from the model "
              "(deployable, fully model-driven; refusals do NOT switch to a hard fallback order -- every turn "
              "is argmax p_hat*V).\n"
              "7. Refusal = no-op turn (belief unchanged), user retained; all 298 users in every mean (fair, "
              "inherited from the harness). Bootstrap paired per-user BOOT=%d seed=%d; selectors deterministic "
              "(V/cid tie-breaks).\n"
              "8. g-hat-to-true-genre cosine and per-turn realized answer rate are DIAGNOSTICS; the true "
              "known-half genre distribution is PRIVILEGED and never enters the policy.\n\n"
              % (P4.BOOT, P4.SEED))

    with open(OUT_MD, "a", encoding="utf-8") as f:
        f.write("".join(md))
    print(f"\n[a4] wall {out['wall_min']}m -> {OUT_JSON}, appended section to {OUT_MD}", flush=True)


if __name__ == "__main__":
    main()
