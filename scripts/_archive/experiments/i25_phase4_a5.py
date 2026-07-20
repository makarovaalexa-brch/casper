"""i25_phase4_a5.py -- A5 "K-map item prober": the deployable blind item-prober scaffold (same as
a3/a4) with the BELIEF replaced by the validated LEARNED K-map (kmap_build.py + kmap_validate.py).

Imports and REUSES scripts/i25_phase4_fair.py (FA), scripts/i25_phase4_a3.py (A3),
scripts/i25_phase4_a4.py (A4) UNMODIFIED, and the ONLINE-INFERENCE code from scripts/kmap_validate.py
(the `Kmap` class: MAP-Newton for (alpha_u, k_u) on fixed item embeddings). NO LLM calls; deterministic;
local compute only.

WHY A5 (vs a3/a4): a3 used raw co-rating cosine (de-popularised -> wandered into refusal territory);
a4 used the feature pmodel with an online genre-match g-hat (popularity dominant, weak per-user tilt,
hit stuck at 4.4/24). A5 uses the K-map -- a population-scale LEARNED knowledge geometry
P(u knows j)=sigmoid(alpha_u + b_j + k_u . e_j). Offline (STRUCTURAL arena) the K-map's per-user term
k_u.e_j is the ONLY thing that lifts held-out AUC over popularity (t=8 full 0.669 vs pop 0.623,
+0.046) -- so A5 is the test of whether that validated offline superiority converts to a blind IN-LOOP
adaptivity WIN over the fixed popular-item static s3.

BELIEF (the K-map, online): start each user at the prior (k_u=0, alpha_u=0 -> P=sigmoid(b_j+beta0),
the popularity-only predictor). After every turn, re-infer (alpha_u, k_u) by the MAP-Newton online
inference in kmap_validate.Kmap.infer from ALL observed events so far (answered=1 / refused=0 for the
probed dense item id). The probe-answer outcome IS the structural label (user rated it in the known
half), exactly the arena the K-map passed its four offline gates in.

SELECTION each turn (BLIND, deployable): argmax over UNASKED bank items of P_kmap(u knows j) x V(j),
  V(j) = study-cohort coverage prior (pop_rate) -- IDENTICAL to a3/a4's V for comparability.
  P_kmap = sigmoid(alpha_u + b_j + k_u.e_j + beta0), beta0 = the level-only structural-arena
  calibration (see below). Tie-break: higher V, then lowest global cid (same rule as a4).

CALIBRATION (level only, as in the offline validation): the K-map's b_j is the population logit of
  P(rated-ever); the STRUCTURAL arena label is rated-in-the-KNOWN-HALF (a lower base rate). We fit a
  single global logit shift beta0 (1-parameter Platt, slope fixed to 1) on a deterministic TRAIN HALF
  of the 298 users (seed 0) so that the t=0 population prediction mean_j sigmoid(b_j+beta0) equals the
  empirical structural base rate over (train users x 160-item bank). Applied to ALL users at ALL turns.
  AUC is invariant to this monotone shift (map-health uncalibrated); it only sets the reported P level.

VARIANTS:
  a5-blind : as above (deployable).
  a5-ucb   : same, but one-standard-error OPTIMISM on the knowledge LOGIT from the Laplace posterior of
             (alpha_u,k_u). The posterior covariance is H^{-1} where H is the Newton Hessian at the MAP
             (cheaply available -- reconstructed at the returned MAP point, NOT a duplicated inference).
             logit_ucb(j) = (alpha+b_j+k.e_j) + sqrt(a_j^T H^{-1} a_j), a_j=[1, e_j]. Principled
             exploration, no hand rules.

PRE-REGISTERED (printed BEFORE results): a5 targets = hit rate above a3/a4's 4.4/24 moving toward
  a3-table's 12.9/24, and NDCG vs s3. a5 WINS iff it beats s3 anytime with CI excl 0 at T=24. Also
  report a5 vs a3-blind and a5 vs a4-blind (is the learned map better than both heuristics IN THE LOOP,
  matching its offline superiority?).

DIAGNOSTICS: hit-rate per arm; per-turn NDCG curve; mean P_kmap(chosen probe) vs realized answer rate
  per turn (calibration in the loop); map-health = per-user held-out structural AUC at t=8 IN THE LOOP
  (a5-blind's own probe order) vs the offline t=8 number 0.669 (does the policy starve the map?).

Run:  python scripts/i25_phase4_a5.py
"""
import os, sys, json, time
import numpy as np
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import i25_phase4_fair as FA     # UNMODIFIED reuse
import i25_phase4_a3 as A3       # UNMODIFIED reuse (build_cooc, make_a3_plan, contrast, first_sep)
import i25_phase4_a4 as A4       # UNMODIFIED reuse (make_a4_runner, load_pmodel)
import kmap_validate as KV       # ONLINE-INFERENCE code (Kmap class); imported, NOT duplicated
P4 = FA.P4
L = FA.L

TMAX = FA.TMAX                   # 24
BUDGETS = FA.BUDGETS             # (8,16,24)
NG = FA.NG

CALIB_SEED = 0                   # deterministic train/eval split for the beta0 level calibration
OFFLINE_T8_AUC = 0.669           # the offline structural full-AUC at t=8 (KMAP_OFFLINE.md) for map-health
OUT_JSON = "experiments/I25_phase4_a5.json"
OUT_MD = "experiments/I25_PHASE4_FAIR.md"


# ================================================================= level-only calibration (beta0)
def calibrate_beta0(b_bank, ans_bank_by_user, users, seed=CALIB_SEED):
    """Fit a single logit shift beta0 so the t=0 population prediction matches the empirical STRUCTURAL
    base rate on a deterministic TRAIN HALF. b_bank = K-map intercepts for the bank (nb,);
    ans_bank_by_user[u] = bool array over the bank (rated-in-known-half). Returns (beta0, base_rate)."""
    n = len(users)
    rng = np.random.default_rng(seed)
    idx = np.arange(n); rng.shuffle(idx)
    tr = idx[:n // 2]
    base = float(np.mean([ans_bank_by_user[users[i]["u"]].mean() for i in tr]))
    # solve mean_j sigmoid(b_bank[j] + beta0) = base (monotone increasing in beta0) by bisection
    lo, hi = -30.0, 30.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if float(KV.sigmoid(b_bank + mid).mean()) < base:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi), base


# ================================================================= Laplace posterior curvature (a5-ucb)
def laplace_cov(KM, J_ev, y_ev, alpha, k):
    """Posterior covariance of x=[alpha,k] at the MAP -- H^{-1} with H the Newton Hessian evaluated at
    the returned MAP point (reconstructed from KM.emb_of/b_of; the iterative inference is NOT duplicated).
    Empty history -> prior covariance."""
    d = KM.d
    pv = np.concatenate([[KV.TAU_A ** 2], np.full(d, KV.TAU ** 2)])
    if len(J_ev) == 0:
        return np.diag(pv)
    A = np.column_stack([np.ones(len(J_ev)), KM.emb_of(J_ev)])
    off = KM.b_of(J_ev)
    x = np.concatenate([[alpha], k])
    p = KV.sigmoid(A @ x + off)
    w = p * (1.0 - p)
    H = (A * w[:, None]).T @ A + np.diag(1.0 / pv)
    try:
        return np.linalg.inv(H)
    except np.linalg.LinAlgError:
        return np.diag(pv)


# ================================================================= a5 per-user runner
def make_a5_runner(KM, CANDS, item_cids, jof, V, beta0, mode):
    """Returns run(rec) -> (plan, diag). mode in {'blind','ucb'}.
    plan = list of (token,native|None) over TMAX (fed to FA.eval_fair, same shape as a3/a4/statics)."""
    ucb = (mode == "ucb")
    nb = len(item_cids)
    b_bank = KM.b_of(jof).astype(np.float64)                    # (nb,) population knowledge intercept
    E_bank = KM.emb_of(jof).astype(np.float64)                  # (nb, d) item knowledge embeddings
    A_bank = np.column_stack([np.ones(nb), E_bank])             # (nb, 1+d) feature rows [1, e_j]
    cids_arr = np.array(item_cids, dtype=np.int64)
    Varr = np.asarray(V, np.float64)

    def run(rec):
        ans = rec["ans_arr"]                                    # global cid -> bool answerable (rated)
        used = np.zeros(nb, bool)
        J_ev, y_ev = [], []
        plan = []
        d_p, d_ans, d_j = [], [], []
        for t in range(TMAX):
            alpha, k = KM.infer(np.array(J_ev, np.int64), np.array(y_ev, np.float64))
            logit = b_bank + alpha + E_bank @ k                 # per-bank knowledge logit
            if ucb:
                Hinv = laplace_cov(KM, J_ev, y_ev, alpha, k)
                var = np.einsum("ij,jk,ik->i", A_bank, Hinv, A_bank)   # a_j^T H^-1 a_j
                logit = logit + np.sqrt(np.clip(var, 0.0, None))       # +1 sigma optimism
            P = KV.sigmoid(logit + beta0)                       # calibrated P_kmap(knows j)
            score = np.where(used, -1e18, P * Varr)
            j_sel = int(np.lexsort((cids_arr, -Varr, -score))[0])
            used[j_sel] = True
            cid = item_cids[j_sel]
            d_p.append(float(P[j_sel])); d_j.append(int(jof[j_sel]))
            if ans[cid]:
                plan.append((FA.tok_of(CANDS, rec, cid), rec["nat_arr"][cid]))
                J_ev.append(int(jof[j_sel])); y_ev.append(1.0); d_ans.append(True)
            else:
                plan.append((None, None))
                J_ev.append(int(jof[j_sel])); y_ev.append(0.0); d_ans.append(False)
        diag = dict(P=np.array(d_p), ans=np.array(d_ans, bool), j=np.array(d_j, np.int64))
        return plan, diag
    return run


# ================================================================= map-health (in-loop held-out AUC @ t)
def map_health_auc(KM, users, diags, jof, t_eval=8):
    """Per-user held-out STRUCTURAL AUC at t=t_eval using the arm's OWN probe order (diags[u]['j','ans']).
    Held-out = bank items not among the first t_eval probes; label = rated-in-known-half. Mean over users
    with both classes present. Compare to the offline (s3-order) number."""
    jof = np.asarray(jof, np.int64)
    aucs = []
    for rec in users:
        d = diags[rec["u"]]
        J8 = d["j"][:t_eval]; y8 = d["ans"][:t_eval].astype(np.float64)
        probed = set(int(x) for x in J8)
        held = np.array([int(j) for j in jof if int(j) not in probed], dtype=np.int64)
        labels = np.array([1 if int(j) in rec["known"] else 0 for j in held])
        if labels.min() == labels.max():
            continue
        alpha, k = KM.infer(J8, y8)
        scores = KM.score(held, alpha, k, True, True)          # logit; AUC calibration-invariant
        aucs.append(float(roc_auc_score(labels, scores)))
    return (float(np.mean(aucs)), len(aucs)) if aucs else (float("nan"), 0)


def main():
    t0 = time.time()
    print("[a5] loading data + frozen RecVAE + trained fold ...", flush=True)
    D = L.G.load_data(); FR = L.Frozen(D)
    model = L.Fold(); import torch
    blob = torch.load(P4.CKPT_BEST, map_location="cpu")
    model.load_state_dict(blob["model"]); model.eval()
    P4.T = TMAX
    users, _ = P4.assemble(D, FR)
    n = len(users)
    print(f"[a5] {n} users; building ladder universe (reuse FA.build_universe) ...", flush=True)
    CANDS, meta = FA.build_universe(D, FR, users)
    cold = FA.cold_ndcg(FR, users, 10)

    item_cids = [m["cid"] for m in CANDS if m["kind"] == "item"]
    concept_cids = [m["cid"] for m in CANDS if m["kind"] == "concept"]
    pop_rate = {m["cid"]: m["pop_rate"] for m in CANDS if m["kind"] == "item"}
    item_by_cov = sorted(item_cids, key=lambda c: -CANDS[c]["pop_rate"])
    s3_pool = item_by_cov[:FA.ITEM_POOL_S3]
    jof = np.array([int(CANDS[c]["key"].split(":")[1]) for c in item_cids], dtype=np.int64)   # dense bank ids
    V = np.array([CANDS[c]["pop_rate"] for c in item_cids], float)                             # coverage prior

    # ---- reproduce s1 / s3 (existing harness) BEFORE adding anything ----
    print("[a5] single-question values (all candidates) ...", flush=True)
    v1 = FA.single_q_values(FR, model, users, cold, CANDS, [m["cid"] for m in CANDS])
    print("[a5] greedy s1 (concepts) ...", flush=True)
    s1 = FA.build_greedy(FR, model, users, cold, CANDS, concept_cids, TMAX, v1=v1, tag="s1")
    print("[a5] greedy s3 (popular items) ...", flush=True)
    s3 = FA.build_greedy(FR, model, users, cold, CANDS, s3_pool, TMAX, v1=v1, tag="s3")
    pt_s1, ans_s1 = FA.eval_fair(FR, model, users, lambda rec: FA.plan_sched(CANDS, rec, s1), cold)
    pt_s3, ans_s3 = FA.eval_fair(FR, model, users, lambda rec: FA.plan_sched(CANDS, rec, s3), cold)
    s1_any24 = float(FA.anytime(pt_s1, 24).mean())
    s3_any24 = float(FA.anytime(pt_s3, 24).mean())

    # ---- reproduce a3-blind (reuse A3 UNMODIFIED) ----
    print("[a5] building 160x160 population co-known cosine (reuse A3.build_cooc) ...", flush=True)
    bank_j = [int(j) for j in jof]
    study_uids = [rec["u"] for rec in users]
    C, self_cnt, nU_pop = A3.build_cooc(bank_j, study_uids)
    colof = {c: k for k, c in enumerate(item_cids)}
    global_order = sorted(item_cids, key=lambda c: (-CANDS[c]["pop_rate"], c))
    a3_pf = A3.make_a3_plan(CANDS, item_cids, s3[0], global_order, colof, C, pop_rate, "blind")
    pt_a3, ans_a3 = FA.eval_fair(FR, model, users, a3_pf, cold)
    a3_any24 = float(FA.anytime(pt_a3, 24).mean())

    print(f"\n[a5] REPRODUCTION: s1 any@24={s1_any24:.4f} (t 0.2103) | s3 any@24={s3_any24:.4f} "
          f"(t 0.2786) | a3-blind any@24={a3_any24:.4f} (t 0.2412)", flush=True)
    ok = (abs(s3_any24 - 0.2786) < 0.0015 and abs(a3_any24 - 0.2412) < 0.0020
          and abs(s1_any24 - 0.2103) < 0.0015)
    if not ok:
        print("[a5] STOP: s3 / a3-blind did NOT reproduce within tolerance -- NOT adding a5. Diagnose first.",
              flush=True)
        return
    print("[a5] reproduction OK -- proceeding to build a5.\n", flush=True)

    # ---- build a4-blind (reuse A4 UNMODIFIED) for the in-loop comparison ----
    print("[a5] building a4-blind (reuse A4.make_a4_runner) ...", flush=True)
    Gm_bank = D["Gmat"][jof].astype(np.float64)
    prior_raw = (V[:, None] * Gm_bank).sum(0)
    prior_unit = prior_raw / (np.linalg.norm(prior_raw) + 1e-9)
    pmodel = A4.load_pmodel()
    a4_runner = A4.make_a4_runner(D, CANDS, item_cids, pmodel, prior_unit, "blind")
    a4_plans = {rec["u"]: a4_runner(rec)[0] for rec in users}
    pt_a4, ans_a4 = FA.eval_fair(FR, model, users, lambda rec: a4_plans[rec["u"]], cold)
    a4_any24 = float(FA.anytime(pt_a4, 24).mean())

    # ---- K-map + level calibration ----
    print("[a5] loading K-map (kmap_validate.Kmap) + fitting level calibration beta0 ...", flush=True)
    KM = KV.Kmap(D)
    ans_bank_by_user = {rec["u"]: rec["ans_arr"][np.array(item_cids)] for rec in users}
    beta0, base_rate = calibrate_beta0(KM.b_of(jof).astype(np.float64), ans_bank_by_user, users)
    p0_mean = float(KV.sigmoid(KM.b_of(jof).astype(np.float64) + beta0).mean())
    print(f"[a5] structural base rate (train half) = {base_rate:.4f}; beta0 = {beta0:+.4f}; "
          f"calibrated mean P@t0 = {p0_mean:.4f}\n", flush=True)

    # ---- pre-registered read print ----
    s3_hit = float(ans_s3.mean()); a3_hit = float(ans_a3.mean()); a4_hit = float(ans_a4.mean())
    print("=" * 78, flush=True)
    print("PRE-REGISTERED READ (a5 K-map prober):", flush=True)
    print("  a5 WINS iff it beats s3 with CI excl 0 at T=24 anytime.", flush=True)
    print(f"  Hit-rate ladder: s3 {s3_hit:.1f} | a3-blind {a3_hit:.1f} | a4-blind {a4_hit:.1f} | "
          "a3-table 12.9/24 (class ceiling).", flush=True)
    print("  a5 target: hit above 4.4 moving toward 12.9; also report a5 vs a3-blind and a5 vs a4-blind.", flush=True)
    print("  Map-health: in-loop held-out structural AUC @t8 vs the offline 0.669 (starvation check).", flush=True)
    print("=" * 78 + "\n", flush=True)

    # ---- run a5 arms ----
    pt, ansn, diags = {}, {}, {}
    for mode in ("blind", "ucb"):
        print(f"[a5] eval a5-{mode} ...", flush=True)
        runner = make_a5_runner(KM, CANDS, item_cids, jof, V, beta0, mode)
        plans = {}; dg = {}
        for rec in users:
            pl, d = runner(rec)
            plans[rec["u"]] = pl; dg[rec["u"]] = d
        pt[mode], ansn[mode] = FA.eval_fair(FR, model, users, lambda rec: plans[rec["u"]], cold)
        diags[mode] = dg

    # ---- contrasts (reuse A3 helpers) ----
    vs_s3 = {m: A3.contrast(pt[m], pt_s3) for m in pt}
    vs_a3 = {m: A3.contrast(pt[m], pt_a3) for m in pt}
    vs_a4 = {m: A3.contrast(pt[m], pt_a4) for m in pt}
    sep_s3 = {m: A3.first_sep(pt[m], pt_s3) for m in pt}

    def row(name, ptm, ansm):
        r = dict(name=name, mean_ans_turns=float(ansm.mean()), hit_rate=float(ansm.mean() / TMAX))
        for T in BUDGETS:
            r[f"any@{T}"] = float(FA.anytime(ptm, T).mean())
            r[f"end@{T}"] = float(FA.endpoint(ptm, T).mean())
        return r
    rows = {m: row(f"a5-{m}", pt[m], ansn[m]) for m in pt}
    rows["s3 popular-item"] = row("s3 popular-item", pt_s3, ans_s3)
    rows["a3-blind"] = row("a3-blind", pt_a3, ans_a3)
    rows["a4-blind"] = row("a4-blind", pt_a4, ans_a4)
    rows["s1 concepts"] = row("s1 concepts", pt_s1, ans_s1)

    blind_hit = rows["blind"]["mean_ans_turns"]
    hit_rose_vs_s3 = blind_hit > s3_hit + 1e-6
    hit_rose_vs_a4 = blind_hit > a4_hit + 1e-6
    blind_wins = (vs_s3["blind"][24]["ci"][0] > 0)

    # ---- calibration in the loop (a5-blind): mean chosen P vs realized answer rate per turn ----
    uids = [r["u"] for r in users]
    calib_P = np.array([diags["blind"][u]["P"] for u in uids]).mean(0)
    calib_ans = np.array([diags["blind"][u]["ans"] for u in uids]).astype(float).mean(0)
    calib_gap = float(np.mean(calib_P - calib_ans))
    calib_corr = float(np.corrcoef(calib_P, calib_ans)[0, 1])

    # ---- map-health: in-loop held-out structural AUC @ t=8 (a5-blind own order) vs offline 0.669 ----
    mh_auc, mh_n = map_health_auc(KM, users, diags["blind"], jof, t_eval=8)
    mh_auc_ucb, mh_n_ucb = map_health_auc(KM, users, diags["ucb"], jof, t_eval=8)

    # ---- console summary ----
    print("\n==== A5 K-map prober (any/end NDCG@10; hit = mean answered turns / 24) ====", flush=True)
    for m in ("s3 popular-item", "a3-blind", "a4-blind", "s1 concepts"):
        r = rows[m]
        print(f"  {m:16s} any24 {r['any@24']:.4f} end24 {r['end@24']:.4f} | hit {r['mean_ans_turns']:.1f}/24",
              flush=True)
    for m in ("blind", "ucb"):
        r = rows[m]; c3 = vs_s3[m]; ca3 = vs_a3[m]; ca4 = vs_a4[m]
        print(f"  a5-{m:11s} any24 {r['any@24']:.4f} end24 {r['end@24']:.4f} | hit {r['mean_ans_turns']:.1f}/24 "
              f"| vs s3@24 {c3[24]['delta']:+.4f}[{c3[24]['ci'][0]:+.4f},{c3[24]['ci'][1]:+.4f}]", flush=True)
        print(f"       {'':11s} vs a3@24 {ca3[24]['delta']:+.4f}[{ca3[24]['ci'][0]:+.4f},{ca3[24]['ci'][1]:+.4f}] "
              f"| vs a4@24 {ca4[24]['delta']:+.4f}[{ca4[24]['ci'][0]:+.4f},{ca4[24]['ci'][1]:+.4f}]", flush=True)
    print(f"\n  HIT ROSE vs s3 ({blind_hit:.1f}>{s3_hit:.1f})? {hit_rose_vs_s3} | "
          f"vs a4-blind ({blind_hit:.1f}>{a4_hit:.1f})? {hit_rose_vs_a4}", flush=True)
    print(f"  a5-blind WINS (beats s3 CI excl 0 @T24)? {blind_wins}", flush=True)
    for T in BUDGETS:
        c = vs_s3["blind"][T]
        print(f"  a5-blind vs s3 @T{T}: {c['delta']:+.4f}[{c['ci'][0]:+.4f},{c['ci'][1]:+.4f}]", flush=True)
    print(f"  CALIBRATION (in loop): mean chosen P vs realized answer-rate gap {calib_gap:+.3f} "
          f"(corr {calib_corr:+.2f}); P@t1 {calib_P[0]:.3f} vs ans@t1 {calib_ans[0]:.3f}", flush=True)
    print(f"  MAP-HEALTH: in-loop held-out structural AUC @t8 (a5-blind order) = {mh_auc:.3f} "
          f"(n={mh_n}) vs offline {OFFLINE_T8_AUC:.3f}", flush=True)

    # ---- verdict text ----
    if blind_wins:
        verdict = ("A5 K-map prober BEATS the best static s3 (CI excl 0 at T=24 anytime) -- the validated "
                   "learned knowledge map makes BLIND adaptivity win where the a3/a4 heuristics could not.")
    elif hit_rose_vs_s3:
        verdict = (f"A5-blind RAISES the hit rate ({blind_hit:.1f} vs a4-blind {a4_hit:.1f}, s3 {s3_hit:.1f}/24) "
                   "but does NOT beat s3 on NDCG at T=24 -- the K-map's per-user knowledge tilt finds more "
                   "answerable items than the fixed list, yet the extra answers do not add enough ranking "
                   "value to overturn the popular-item static. Branch B stands.")
    else:
        verdict = (f"A5-blind does NOT raise the hit rate above s3 ({blind_hit:.1f} vs {s3_hit:.1f}/24) -- the "
                   "learned map's in-loop probe choices do not surface more answerable items than the fixed "
                   "popular list. Branch B stands.")
    print(f"\n  VERDICT: {verdict}", flush=True)

    # ---- persist json ----
    def cdump(cd):
        return {str(T): cd[T] for T in BUDGETS}
    out = dict(
        config=dict(dataset="ML-25M", instrument="RecVAE-d512 + I2.5 learned fold", TMAX=TMAX,
                    budgets=list(BUDGETS), n_users=n, bank_items=len(item_cids),
                    kmap_emb=KV.EMB, kmap_int=KV.INT, kmap_d=KM.d, tau=KV.TAU, tau_a=KV.TAU_A,
                    newton_it=KV.NEWTON_IT, beta0=beta0, structural_base_rate=base_rate,
                    calibrated_p_t0=p0_mean, calib_seed=CALIB_SEED,
                    V_prior="study-cohort coverage (pop_rate)", fold_ckpt=P4.CKPT_BEST,
                    best_val=blob["state"]["best_val"]),
        reproduction=dict(s1_any24=s1_any24, s3_any24=s3_any24, a3_blind_any24=a3_any24, a4_blind_any24=a4_any24,
                          s1_target=0.2103, s3_target=0.2786, a3_target=0.2412, ok=bool(ok)),
        hit_anchors=dict(s3=s3_hit, a3_blind=a3_hit, a4_blind=a4_hit, a3_table=12.9),
        arms={rows[m]["name"]: rows[m] for m in rows},
        vs_s3={f"a5-{m}": cdump(vs_s3[m]) for m in pt},
        vs_a3_blind={f"a5-{m}": cdump(vs_a3[m]) for m in pt},
        vs_a4_blind={f"a5-{m}": cdump(vs_a4[m]) for m in pt},
        first_separation_vs_s3={f"a5-{m}": dict(turn=sep_s3[m][0], sign=sep_s3[m][1]) for m in pt},
        ndcg_curves={f"a5-{m}": [float(x) for x in pt[m].mean(axis=0)] for m in pt},
        calibration=dict(P_by_turn=[float(x) for x in calib_P],
                         ans_rate_by_turn=[float(x) for x in calib_ans],
                         mean_gap=calib_gap, corr=calib_corr),
        map_health=dict(inloop_auc_t8_blind=mh_auc, inloop_n_blind=mh_n,
                        inloop_auc_t8_ucb=mh_auc_ucb, inloop_n_ucb=mh_n_ucb,
                        offline_auc_t8=OFFLINE_T8_AUC, delta_vs_offline=mh_auc - OFFLINE_T8_AUC),
        hypothesis=dict(blind_wins=bool(blind_wins), hit_rose_vs_s3=bool(hit_rose_vs_s3),
                        hit_rose_vs_a4=bool(hit_rose_vs_a4), blind_hit=blind_hit,
                        s3_hit=s3_hit, a3_hit=a3_hit, a4_hit=a4_hit),
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
    md.append("\n## A5 K-map prober\n\n")
    md.append("Date 2026-07-08. Script `scripts/i25_phase4_a5.py` (imports/reuses `i25_phase4_fair.py`, "
              "`i25_phase4_a3.py`, `i25_phase4_a4.py` UNMODIFIED, and the ONLINE-INFERENCE `Kmap` class "
              "from `scripts/kmap_validate.py`). NO LLM calls; deterministic; local compute.\n\n")
    md.append("**Reproduction gate (existing harness, before adding anything):** "
              f"s1 any@24={s1_any24:.4f} (t 0.2103), s3 any@24={s3_any24:.4f} (t 0.2786), "
              f"a3-blind any@24={a3_any24:.4f} (t 0.2412) -> reproduced={ok}. "
              f"(a4-blind any@24={a4_any24:.4f} for the in-loop comparison.)\n\n")
    md.append("**Idea.** The scaffold is a3/a4's blind item prober; the BELIEF is the validated LEARNED "
              "K-map `P(u knows j)=sigmoid(alpha_u + b_j + k_u.e_j)` (kmap_build/kmap_validate). Offline "
              "in this STRUCTURAL arena the per-user term k_u.e_j is the ONLY thing that lifts held-out AUC "
              f"over popularity (t8 full 0.669 vs pop 0.623). A5 asks whether that offline superiority "
              "converts to a blind IN-LOOP win. Each user starts at the prior (k=0, alpha=0 -> "
              "P=sigmoid(b_j+beta0)); after every turn (alpha_u,k_u) are re-inferred by the MAP-Newton "
              "online inference from ALL observed events (answered=1/refused=0 for the probed dense id -- "
              "the probe outcome IS the structural label). Selection = argmax_{unasked} P_kmap(j)*V(j), "
              "V=study-cohort coverage prior (IDENTICAL to a3/a4).\n\n")
    md.append(f"**Calibration (level only, as offline).** b_j is the population rated-ever logit; the "
              f"STRUCTURAL label is rated-in-the-known-half (lower base rate). We fit ONE global logit shift "
              f"beta0 (1-param Platt, slope=1) on a deterministic TRAIN HALF (seed {CALIB_SEED}) so that the "
              f"t=0 population prediction mean_j sigmoid(b_j+beta0) equals the empirical structural base rate "
              f"over (train users x 160 bank items). Fitted: base rate = {base_rate:.4f} -> beta0 = "
              f"{beta0:+.4f} (calibrated mean P@t0 = {p0_mean:.4f}). AUC is invariant to this monotone shift "
              f"(map-health uses the uncalibrated logit); in the deep low-probability regime P*V argmax is "
              f"near beta0-invariant, so beta0 sets the reported P LEVEL far more than the picks.\n\n")
    md.append("**Variants.** a5-blind (above, deployable). a5-ucb: one-standard-error optimism on the "
              "knowledge logit from the Laplace posterior of (alpha_u,k_u) -- H^{-1} with H the Newton "
              "Hessian reconstructed AT the returned MAP (the posterior variance IS cheaply available; the "
              "iterative inference is not duplicated); logit_ucb(j)=(alpha+b_j+k.e_j)+sqrt(a_j^T H^{-1} a_j), "
              "a_j=[1,e_j]. Principled exploration, no hand rules.\n\n")
    md.append("| arm | any/end @8 | any/end @16 | any/end @24 | hit (ansT) | delta-any@24 vs s3 [CI] |\n"
              "|---|---|---|---|---|---|\n")
    md.append(frow("s3 popular-item (opponent)", rows["s3 popular-item"]) + "\n")
    md.append(frow("a3-blind (co-known)", rows["a3-blind"]) + "\n")
    md.append(frow("a4-blind (pmodel)", rows["a4-blind"]) + "\n")
    md.append(frow("s1 concepts", rows["s1 concepts"]) + "\n")
    md.append(frow("a5-blind (K-map)", rows["blind"], vs_s3["blind"]) + "\n")
    md.append(frow("a5-ucb (K-map + Laplace 1se)", rows["ucb"], vs_s3["ucb"]) + "\n")

    md.append("\n**Contrast a5-blind vs s3 (THE contrast), vs a3-blind, vs a4-blind, per budget:**\n\n"
              "| budget T | a5-blind vs s3 [CI] | a5-blind vs a3-blind [CI] | a5-blind vs a4-blind [CI] |\n"
              "|---|---|---|---|\n")
    for T in BUDGETS:
        c3 = vs_s3["blind"][T]; ca3 = vs_a3["blind"][T]; ca4 = vs_a4["blind"][T]
        md.append(f"| {T} | {c3['delta']:+.4f}[{c3['ci'][0]:+.4f},{c3['ci'][1]:+.4f}] | "
                  f"{ca3['delta']:+.4f}[{ca3['ci'][0]:+.4f},{ca3['ci'][1]:+.4f}] | "
                  f"{ca4['delta']:+.4f}[{ca4['ci'][0]:+.4f},{ca4['ci'][1]:+.4f}] |\n")

    md.append("\n**Mechanism metric -- hit rate (mean answered turns / 24):** "
              f"s3 = {s3_hit:.1f} (~{100*s3_hit/24:.0f}%); a3-blind = {a3_hit:.1f}; a4-blind = {a4_hit:.1f}; "
              f"a5-blind = {blind_hit:.1f} (~{100*rows['blind']['hit_rate']:.0f}%); "
              f"a5-ucb = {rows['ucb']['mean_ans_turns']:.1f}; a3-table ceiling = 12.9. "
              f"Hit rose vs s3? **{hit_rose_vs_s3}**. Hit rose vs a4-blind? **{hit_rose_vs_a4}**. "
              f"a5-blind BEATS s3 at T=24 (CI excl 0)? **{blind_wins}**.\n\n")
    fs = sep_s3["blind"]
    md.append(f"**First separation (a5-blind belief(t) vs s3 belief(t), CI excl 0):** "
              f"{('turn ' + str(fs[0]) + ' (sign ' + fs[1] + ')') if fs[0] else 'NEVER within T=24'}.\n\n")

    md.append("**Calibration in the loop (mean chosen-probe P_kmap vs realized answer rate per turn):**\n\n"
              "| t | " + " | ".join(str(t + 1) for t in range(TMAX)) + " |\n")
    md.append("|" + "---|" * (TMAX + 1) + "\n")
    md.append("| mean chosen P_kmap | " + " | ".join(f"{x:.3f}" for x in calib_P) + " |\n")
    md.append("| realized answer rate | " + " | ".join(f"{x:.3f}" for x in calib_ans) + " |\n")
    lev = ("OPTIMISTIC (over-predicts)" if calib_gap > 0.02 else
           ("PESSIMISTIC (under-predicts)" if calib_gap < -0.02 else "well-calibrated in level"))
    md.append(f"\nMean P_kmap - answer-rate gap = {calib_gap:+.3f} (corr {calib_corr:+.2f}) -- the calibrated "
              f"K-map probability is {lev} against the in-loop realized answer rate.\n\n")

    md.append("**Map-health (does the policy starve the map?): per-user held-out STRUCTURAL AUC @t=8 using "
              "a5-blind's OWN probe order, vs the offline (s3-order) t8 number.**\n\n"
              f"- In-loop a5-blind AUC@t8 = **{mh_auc:.3f}** (n={mh_n}) vs offline **{OFFLINE_T8_AUC:.3f}** "
              f"-> delta {mh_auc - OFFLINE_T8_AUC:+.3f}. a5-ucb AUC@t8 = {mh_auc_ucb:.3f} (n={mh_n_ucb}).\n"
              f"- Reading: {'in-loop map is HEALTHY (>= offline within noise) -- the policy is not starving the map' if mh_auc >= OFFLINE_T8_AUC - 0.02 else 'in-loop map is DEGRADED vs offline -- the policy concentrates probes on high-P items and starves the K-map of the diverse evidence its offline s3-order run had'}.\n\n")

    md.append("**NDCG@10(t) curves (t=1..24):**\n\n| arm | " + " | ".join(f"t{t+1}" for t in range(TMAX)) + " |\n")
    md.append("|" + "---|" * (TMAX + 1) + "\n")
    for lab, cur in (("s3 popular-item", pt_s3.mean(axis=0)), ("a3-blind", pt_a3.mean(axis=0)),
                     ("a4-blind", pt_a4.mean(axis=0)), ("a5-blind", pt["blind"].mean(axis=0)),
                     ("a5-ucb", pt["ucb"].mean(axis=0))):
        md.append(f"| {lab} | " + " | ".join(f"{c:.3f}" for c in cur) + " |\n")

    md.append(f"\n**VERDICT:** {verdict}\n\n")
    md.append("**ASSUMPTIONS / judgment calls (a5):**\n"
              "1. Probe bank = the 160 top-coverage ladder items (coverage>=3) s3/a3/a4 draw from; item "
              "probes only; V(j)=study-cohort coverage prior (pop_rate) -- identical to a3/a4.\n"
              "2. Belief = the LEARNED K-map (kmap_emb.npz/kmap_intercepts.npz), online inference = "
              f"kmap_validate.Kmap.infer (MAP-Newton, {KV.NEWTON_IT} it, priors tau(k)={KV.TAU}, "
              f"tau_a(alpha)={KV.TAU_A}); imported UNMODIFIED, not duplicated. t=0 => k=alpha=0 => "
              "P=sigmoid(b_j+beta0) (popularity-only), so turn 1 is fully model-driven and deployable.\n"
              "3. Event label = the probe outcome (answered=1 iff the user rated the dense item id in the "
              "known half, i.e. the STRUCTURAL arena the K-map's four offline gates were passed in; "
              "refused=0). No privileged features enter the policy.\n"
              f"4. Level calibration beta0: 1-parameter logit shift (Platt slope=1) fit on a deterministic "
              f"TRAIN HALF (seed {CALIB_SEED}) so mean_j sigmoid(b_j+beta0)=structural base rate "
              f"{base_rate:.4f} over (train users x bank); applied to all users/turns. AUC-invariant "
              "(map-health uncalibrated). In the low-P regime P*V argmax is near beta0-invariant -> "
              "calibration sets the reported P LEVEL, barely the picks.\n"
              "5. Selection = argmax_{unasked} P_kmap(j)*V(j); tie-break higher V then lowest global cid "
              "(same deterministic rule as a4).\n"
              "6. a5-ucb: Laplace posterior cov = H^{-1}, H = Newton Hessian reconstructed at the returned "
              "MAP from KM.emb_of/b_of (the posterior variance is a by-product of the same infer; the "
              "iterative fitting is not re-run). logit_ucb=(alpha+b_j+k.e_j)+sqrt(a_j^T H^{-1} a_j), 1-sigma "
              "optimism; empty history -> prior covariance diag(tau_a^2, tau^2 I).\n"
              "7. Map-health = held-out structural AUC at t=8 over bank items NOT among the arm's first 8 "
              "probes (a5-blind's OWN order), label=rated-in-known-half, per-user AUC needs both classes; "
              "compared to the offline s3-order 0.669 to detect probe-choice starvation.\n"
              "8. Refusal = no-op turn (belief unchanged for the fold; the K-map still records the refusal "
              "event), user retained; all 298 users in every mean (fair, inherited). Bootstrap paired "
              f"per-user BOOT={P4.BOOT} seed={P4.SEED}; all selectors deterministic.\n\n")

    with open(OUT_MD, "a", encoding="utf-8") as f:
        f.write("".join(md))
    print(f"\n[a5] wall {out['wall_min']}m -> {OUT_JSON}, appended section to {OUT_MD}", flush=True)


if __name__ == "__main__":
    main()
