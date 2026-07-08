"""i25_phase4_arena2.py -- ARENA-V2 (LLM-MEASURED answerability) rerun of the key Phase-4 arms.

The FAIR harness (i25_phase4_fair.py) and its A3/A4 probers use ARENA-V1: an ITEM probe is answerable
ONLY if the user rated it in the KNOWN half (structural lower bound, ~14-18% hit rate). A4's diagnosis
showed this HARSH rule is what makes blind adaptive probing lose: the project's VALIDATED answerability
measurement (the cached LLM judged grid + the fitted pmodel, base answer-rate ~0.73) says real users can
answer FAR more item probes (seen-but-not-rated famous films). ARENA-V2 implements that MEASURED model.

The two answerability models are the study's TWO PRE-REGISTERED environments:
  arena-v1 = structural lower-bound answerability (rated-in-known-half only);
  arena-v2 = LLM-measured answerability (this script).
Results are reported under BOTH; the adaptivity verdict may legitimately differ (that contrast is itself
a headline finding).

ARENA-V2 ITEM ANSWER MODEL (environment-side only; AGENT arms stay exactly as blind/privileged as before):
  ANSWERABLE(user,item) iff:
    * the user rated it in the known half (as v1), OR
    * the cached LLM judged grid (gate + main-study grids) says YES for (user,item), OR
    * (item not in the grid for that user) the fitted answerability pmodel, evaluated with the user's
      TRUE known-half genre_match (env-side => true-profile features LEGAL here), has p_hat >= the
      calibrated grid base answer-rate BASE_RATE=0.732.
  ANSWER VALUE:
    * rated in known half  -> real centered rating (unchanged from v1);
    * answerable-but-unrated -> independent-CF prediction (bank-restricted EASE trained on KNOWN-portion
      ratings, all 300 study-user held-out rows dropped) + Gaussian noise sigma=0.70 stars, seeded per
      (user,item), clipped [0.5,5], then centered. LABEL: LLM/CF-predicted (sensitivity-only convention).
      Native recall channel: nat=item iff the (noised) predicted star >= 4 -- the SAME channel v1 uses for
      a rated-liked item.
  Concepts / attributes: UNCHANGED from the fair harness.

Everything is applied by OVERRIDING the item entries of each user's ans_arr/val_arr/nat_arr AFTER
FA.build_universe; all downstream machinery (greedy s3 rebuild, a3, a4, u1) then reads the v2 answer model
automatically. NO LLM calls. Deterministic. Run: python scripts/i25_phase4_arena2.py
"""
import os, sys, json, time, collections
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import i25_phase4_fair as FA        # UNMODIFIED reuse
import i25_phase4_a3 as A3          # UNMODIFIED reuse (build_cooc, make_a3_plan, contrast, first_sep)
import i25_phase4_a4 as A4          # UNMODIFIED reuse (make_a4_runner, load_pmodel, item_year, is_franchise)
P4 = FA.P4
L = FA.L

TMAX = FA.TMAX                      # 24
BUDGETS = FA.BUDGETS                # (8,16,24)
NG = FA.NG

BASE_RATE = 0.732                   # calibrated grid base answer-rate (MAIN STUDY: base_rate 0.732); pmodel threshold
SIGMA_STAR = 0.70                   # value-channel noise on CF-predicted unrated ratings (stars)
EASE_TOPK_POP = 4000                # EASE universe = top-pop UNION bank (matches cross-check A -> validated MAE 0.74)
EASE_LAMBDA = 500.0
META = "data/movielens/.cache/ml25m/meta.npz"
GATE_GRID = ".cache/instrument2/answerability_grid_ml25m.json"
MAIN_GRID = ".cache/instrument2/answerability_mainstudy_grid.json"
EASE_CACHE = ".cache/instrument2/arena2_ease_bankpred.npz"
PMODEL_JSON = A4.PMODEL_JSON
OUT_JSON = "experiments/I25_phase4_arena2.json"
OUT_MD = "experiments/I25_PHASE4_FAIR.md"


# ==================================================================== grid item answerability (measured)
def build_grid_item_ans(users):
    """Per study user, {dense_item_j: answerable_bool} from the UNION of the gate + main-study LLM grids
    (answerable if ANY grid judged it YES). Returns (per_user, coverage_stats)."""
    ga = json.load(open(GATE_GRID))["users"]
    gm = json.load(open(MAIN_GRID))["users"]
    per_user = {}
    judged_yes = judged_tot = 0
    for rec in users:
        u = rec["u"]; d = {}
        for grid in (ga, gm):
            g = grid.get(str(u))
            if not g:
                continue
            for i, q in enumerate(g["Q"]):
                if q[0] != "item":
                    continue
                j = int(q[1]["j"]); a = g["ans"].get(str(i))
                if a is None:
                    continue
                yes = (a.get("can_answer") == "yes")
                d[j] = d.get(j, False) or yes
        per_user[u] = d
    return per_user


# ==================================================================== bank-restricted EASE (independent CF)
def _pmodel_phat_item(pm, pop_pct, log_rcount, decade, genre_match, franchise):
    mu, sd, coef, intc, _ = pm
    x = np.array([pop_pct, log_rcount, decade, genre_match, franchise, 0.0], float)  # is_concept=0
    z = (x - mu) / sd
    return float(1.0 / (1.0 + np.exp(-(z @ coef + intc))))


def build_or_load_ease_bankpred(D, users, bank_j):
    """Bank-restricted EASE (cross-check A recipe): universe = top-EASE_TOPK_POP popular UNION bank.
    Train item-item weights on KNOWN-portion ratings of the full population with ALL study users'
    held-out interactions dropped; predict every bank item for every study user from their known-half
    context. Cached to EASE_CACHE (rebuilt if bank/uids mismatch). Returns pred[u] = {j: star}."""
    uids = np.array([rec["u"] for rec in users], np.int64)
    bank_arr = np.array(sorted(int(j) for j in bank_j), np.int64)
    if os.path.exists(EASE_CACHE):
        z = np.load(EASE_CACHE)
        if (len(z["bank"]) == len(bank_arr) and np.array_equal(z["bank"], bank_arr)
                and np.array_equal(z["uids"], uids)):
            print(f"[ease] loaded cached bank predictions {z['P'].shape} <- {EASE_CACHE}", flush=True)
            P = z["P"]
            return {int(u): {int(j): float(P[r, k]) for k, j in enumerate(bank_arr)}
                    for r, u in enumerate(uids)}, EASE_LAMBDA, int(z["nUni"])
    import scipy.sparse as sp
    d = np.load(META)
    uu = d["uu"].astype(np.int64); ii = d["ii"].astype(np.int64); rr = d["rr"].astype(np.float64)
    cnt = d["cnt"].astype(np.float64); ni = int(d["ni"]); nu = int(d["nu"])
    split = L.G.build_split(D)                                     # seed-123 answerer split (same as arena)
    study_uids = set(int(rec["u"]) for rec in users)

    top_pop = set(int(j) for j in np.argsort(-cnt)[:EASE_TOPK_POP])
    universe = sorted(top_pop | set(int(j) for j in bank_arr))
    uni_index = {j: k for k, j in enumerate(universe)}
    nUni = len(universe)
    print(f"[ease] universe={nUni} (top{EASE_TOPK_POP} pop + bank; bank already in pop: "
          f"{len(set(int(j) for j in bank_arr) & top_pop)}/{len(bank_arr)})", flush=True)

    # drop ALL study users' held-out interactions (never train on recommendation targets)
    heldout_codes = set()
    for u in study_uids:
        if u in split:
            for j in split[u][1]:
                heldout_codes.add(u * ni + int(j))
    codes = uu * ni + ii
    drop = np.isin(codes, np.fromiter(heldout_codes, np.int64, len(heldout_codes)))
    in_uni = np.zeros(ni, bool); in_uni[universe] = True
    keep = in_uni[ii] & (~drop)
    uu_k = uu[keep]; rr_k = rr[keep]
    ii_k = np.array([uni_index[int(j)] for j in ii[keep]], np.int64)
    print(f"[ease] dropped {int(drop.sum())} study held-out rows; train interactions {len(uu_k)}", flush=True)

    M = sp.csr_matrix((np.ones(len(uu_k)), (uu_k, ii_k)), shape=(nu, nUni)); M.data[:] = 1.0
    Rsum = sp.csr_matrix((rr_k, (uu_k, ii_k)), shape=(nu, nUni))
    item_cnt = np.asarray(M.sum(0)).ravel(); item_rsum = np.asarray(Rsum.sum(0)).ravel()
    mu = np.divide(item_rsum, np.maximum(item_cnt, 1.0)); global_mu = float(rr_k.mean())
    mu[item_cnt == 0] = global_mu
    print("[ease] Gram + inverse ...", flush=True)
    G = (M.T @ M).toarray().astype(np.float64); G[np.diag_indices_from(G)] += EASE_LAMBDA
    Pm = np.linalg.inv(G); diagP = np.diag(Pm).copy()
    B = -Pm / diagP[None, :]; np.fill_diagonal(B, 0.0); del G, Pm

    bank_cols = np.array([uni_index[int(j)] for j in bank_arr], np.int64)
    Bbank = B[:, bank_cols]                                        # (nUni, nbank)
    mu_bank = mu[bank_cols]
    P = np.zeros((len(uids), len(bank_arr)), np.float32)
    for r, rec in enumerate(users):
        rc = np.zeros(nUni)
        for j, rate in rec["known"].items():                      # known-half context (mirror v1 profile)
            jj = int(j)
            if jj in uni_index:
                rc[uni_index[jj]] = rate - mu[uni_index[jj]]
        pred = mu_bank + rc @ Bbank
        P[r] = np.clip(pred, 0.5, 5.0)
    os.makedirs(os.path.dirname(EASE_CACHE), exist_ok=True)
    np.savez(EASE_CACHE, bank=bank_arr, uids=uids, P=P, nUni=nUni)
    print(f"[ease] built + cached bank predictions {P.shape} -> {EASE_CACHE}", flush=True)
    return {int(u): {int(j): float(P[r, k]) for k, j in enumerate(bank_arr)}
            for r, u in enumerate(uids)}, EASE_LAMBDA, nUni


def noise_uj(u, j):
    """Deterministic per-(user,item) Gaussian(0,1) draw (hash-free, reproducible)."""
    s = ((int(u) & 0xffffffff) * 2654435761 + (int(j) & 0xffffffff)) & 0xffffffff
    return float(np.random.default_rng(s).standard_normal())


# ==================================================================== apply arena-v2 override
def apply_arena2(D, CANDS, users, grid_ans, ease_pred, pm):
    """Override the ITEM entries of ans_arr/val_arr/nat_arr per user to the measured (arena-v2) model.
    Concepts/attrs untouched. Returns per-user + aggregate provenance stats."""
    item_cids = [m["cid"] for m in CANDS if m["kind"] == "item"]
    jof = {cid: int(CANDS[cid]["key"].split(":")[1]) for cid in item_cids}
    stat = collections.Counter()
    bank_grid_yes = bank_grid_tot = 0
    cover_counts = []
    for rec in users:
        u = rec["u"]; d = grid_ans.get(u, {}); ep = ease_pred.get(u, {})
        cm = rec["cmean"]
        # TRUE known-half genre distribution (env-side legal for the answering rule)
        tg = np.zeros(NG)
        for j in rec["known"]:
            if 0 <= int(j) < D["Gmat"].shape[0]:
                tg = tg + D["Gmat"][int(j)].astype(np.float64)
        tg_unit = tg / (np.linalg.norm(tg) + 1e-9)
        cov = 0
        for cid in item_cids:
            j = jof[cid]
            if j in d:
                cov += 1; bank_grid_tot += 1; bank_grid_yes += int(d[j])
            if rec["ans_arr"][cid]:
                stat["rated"] += 1                                # keep real centered rating (v1)
                continue
            # unrated in the known half -> consult the measured model
            if j in d:
                answerable = bool(d[j]); src = "grid"
            else:
                gv = D["Gmat"][j].astype(np.float64); gn = np.linalg.norm(gv)
                gm = float(tg_unit @ (gv / (gn + 1e-9))) if gn > 0 else 0.0
                yr = A4.item_year(D["title"][j]); dec = ((yr - 1900) / 100.0) if yr else 0.5
                ph = _pmodel_phat_item(pm, float(D["pr"][j]), float(np.log(D["cnt"][j] + 1.0)),
                                       dec, gm, float(A4.is_franchise(D["title"][j])))
                answerable = (ph >= BASE_RATE); src = "pmodel"
            if answerable:
                pred = float(np.clip(ep.get(j, cm) + SIGMA_STAR * noise_uj(u, j), 0.5, 5.0))
                rec["ans_arr"][cid] = True
                rec["val_arr"][cid] = pred - cm
                rec["nat_arr"][cid] = j if pred >= 4.0 else None
                stat[f"answerable_{src}"] += 1
            else:
                stat[f"refuse_{src}"] += 1
        cover_counts.append(cov)
    prov = dict(stat=dict(stat), bank_grid_yes=bank_grid_yes, bank_grid_tot=bank_grid_tot,
                grid_bank_yes_rate=(bank_grid_yes / bank_grid_tot if bank_grid_tot else 0.0),
                mean_bank_grid_coverage=float(np.mean(cover_counts)),
                mean_bank_grid_coverage_frac=float(np.mean(cover_counts) / len(item_cids)),
                n_bank=len(item_cids))
    return prov


# ==================================================================== helpers
def item_hit_rate(users, plan_fn):
    ans = np.array([sum(1 for p in plan_fn(rec)[:TMAX] if p[0] is not None) for rec in users], float)
    return ans


def rowdict(name, ptm, ansm):
    r = dict(name=name, mean_ans_turns=float(ansm.mean()), hit_rate=float(ansm.mean() / TMAX))
    for T in BUDGETS:
        r[f"any@{T}"] = float(FA.anytime(ptm, T).mean())
        r[f"end@{T}"] = float(FA.endpoint(ptm, T).mean())
    return r


def main():
    t0 = time.time()
    print("[arena2] loading data + frozen RecVAE + trained fold ...", flush=True)
    D = L.G.load_data(); FR = L.Frozen(D)
    model = L.Fold(); blob = torch.load(P4.CKPT_BEST, map_location="cpu")
    model.load_state_dict(blob["model"]); model.eval()
    P4.T = TMAX
    users, _ = P4.assemble(D, FR)
    n = len(users)
    print(f"[arena2] {n} users; building ladder universe (reuse FA.build_universe) ...", flush=True)
    CANDS, meta = FA.build_universe(D, FR, users)
    cold = FA.cold_ndcg(FR, users, 10)

    concept_cids = [m["cid"] for m in CANDS if m["kind"] == "concept"]
    item_cids = [m["cid"] for m in CANDS if m["kind"] == "item"]
    pop_rate = {m["cid"]: m["pop_rate"] for m in CANDS if m["kind"] == "item"}
    item_by_cov = sorted(item_cids, key=lambda c: -CANDS[c]["pop_rate"])
    s3_pool = item_by_cov[:FA.ITEM_POOL_S3]
    bank_j = [int(CANDS[c]["key"].split(":")[1]) for c in item_cids]

    # =============================================================== ARENA-V1 reproduction (before switch)
    print("[arena2] ARENA-V1 reproduction: greedy s1 + s3 under the ORIGINAL (rated-only) rule ...", flush=True)
    s1 = FA.build_greedy(FR, model, users, cold, CANDS, concept_cids, TMAX, tag="s1")
    s3_v1 = FA.build_greedy(FR, model, users, cold, CANDS, s3_pool, TMAX, tag="s3v1")
    pt_s1, ans_s1 = FA.eval_fair(FR, model, users, lambda rec: FA.plan_sched(CANDS, rec, s1), cold)
    pt_s3v1, ans_s3v1 = FA.eval_fair(FR, model, users, lambda rec: FA.plan_sched(CANDS, rec, s3_v1), cold)
    s1_any24_v1 = float(FA.anytime(pt_s1, 24).mean())
    s3_any24_v1 = float(FA.anytime(pt_s3v1, 24).mean())
    v1_ok = abs(s3_any24_v1 - 0.2786) < 0.0015 and abs(s1_any24_v1 - 0.2103) < 0.0015
    print(f"[arena2] V1 s1 any@24={s1_any24_v1:.4f} (t 0.2103) | s3 any@24={s3_any24_v1:.4f} (t 0.2786) "
          f"| hit s3_v1={ans_s3v1.mean():.1f}/24 -> reproduced={v1_ok}", flush=True)
    if not v1_ok:
        print("[arena2] STOP: arena-v1 s1/s3 did NOT reproduce within tolerance. Diagnose before switching.",
              flush=True)
        return
    v1_hit_s3 = float(ans_s3v1.mean())

    # =============================================================== switch to ARENA-V2 measured answerability
    print("[arena2] building measured-answerability model (grids + EASE + pmodel fallback) ...", flush=True)
    grid_ans = build_grid_item_ans(users)
    ease_pred, ease_lambda, ease_nuni = build_or_load_ease_bankpred(D, users, bank_j)
    pm = A4.load_pmodel()
    prov = apply_arena2(D, CANDS, users, grid_ans, ease_pred, pm)
    print(f"[arena2] provenance: {prov['stat']}", flush=True)
    print(f"[arena2] bank grid coverage/user mean={prov['mean_bank_grid_coverage']:.1f}/{prov['n_bank']} "
          f"({100*prov['mean_bank_grid_coverage_frac']:.1f}%); grid bank yes-rate={prov['grid_bank_yes_rate']:.3f}",
          flush=True)

    # rebuild s3 greedy UNDER v2 (its optimal list may change); s1 concepts unchanged by item override
    print("[arena2] ARENA-V2: rebuild greedy s3 under the measured model ...", flush=True)
    s3 = FA.build_greedy(FR, model, users, cold, CANDS, s3_pool, TMAX, tag="s3v2")
    pt_s3, ans_s3 = FA.eval_fair(FR, model, users, lambda rec: FA.plan_sched(CANDS, rec, s3), cold)
    s3_any24 = float(FA.anytime(pt_s3, 24).mean()); v2_hit_s3 = float(ans_s3.mean())
    print(f"[arena2] V2 s3 any@24={s3_any24:.4f} | hit s3_v2={v2_hit_s3:.1f}/24 "
          f"(v1 was {s3_any24_v1:.4f} / {v1_hit_s3:.1f})", flush=True)

    # =============================================================== a3-blind + a3-table (reuse A3 UNMODIFIED)
    print("[arena2] building 160x160 population co-known cosine (reuse A3.build_cooc) ...", flush=True)
    study_uids = [rec["u"] for rec in users]
    C, self_cnt, nU_pop = A3.build_cooc(bank_j, study_uids)
    colof = {c: k for k, c in enumerate(item_cids)}
    global_order = sorted(item_cids, key=lambda c: (-CANDS[c]["pop_rate"], c))
    print("[arena2] eval a3-blind + a3-table under v2 ...", flush=True)
    a3b_pf = A3.make_a3_plan(CANDS, item_cids, s3[0], global_order, colof, C, pop_rate, "blind")
    a3t_pf = A3.make_a3_plan(CANDS, item_cids, s3[0], global_order, colof, C, pop_rate, "table")
    pt_a3b, ans_a3b = FA.eval_fair(FR, model, users, a3b_pf, cold)
    pt_a3t, ans_a3t = FA.eval_fair(FR, model, users, a3t_pf, cold)

    # =============================================================== a4-blind (reuse A4 runner UNMODIFIED)
    print("[arena2] eval a4-blind under v2 (online g-hat pmodel prober) ...", flush=True)
    jarr = np.array(bank_j, np.int64)
    Gm_bank = D["Gmat"][jarr].astype(np.float64)
    Vw = np.array([CANDS[c]["pop_rate"] for c in item_cids], float)
    prior_unit = (Vw[:, None] * Gm_bank).sum(0); prior_unit = prior_unit / (np.linalg.norm(prior_unit) + 1e-9)
    a4_runner = A4.make_a4_runner(D, CANDS, item_cids, pm, prior_unit, "blind")
    a4_plans, a4_diags = {}, {}
    for rec in users:
        pl, dg = a4_runner(rec); a4_plans[rec["u"]] = pl; a4_diags[rec["u"]] = dg
    pt_a4, ans_a4 = FA.eval_fair(FR, model, users, lambda rec: a4_plans[rec["u"]], cold)

    # =============================================================== u1 clairvoyant ceiling (PRIV, reuse FA)
    print("[arena2] eval u1 clairvoyant ceiling under v2 (reuse FA.eval_u1) ...", flush=True)
    pt_u1, ans_u1 = FA.eval_u1(FR, model, users, CANDS, cold, 10)

    # =============================================================== contrasts + verdicts
    def contrast(a, b):
        return {T: P4.boot(FA.anytime(a, T), FA.anytime(b, T)) for T in BUDGETS}
    vs_s3 = dict(a3_blind=contrast(pt_a3b, pt_s3), a4_blind=contrast(pt_a4, pt_s3),
                 a3_table=contrast(pt_a3t, pt_s3), u1=contrast(pt_u1, pt_s3), s1=contrast(pt_s1, pt_s3))
    a3b_beats = any(vs_s3["a3_blind"][T]["ci"][0] > 0 for T in BUDGETS)
    a4b_beats = any(vs_s3["a4_blind"][T]["ci"][0] > 0 for T in BUDGETS)
    u1_beats = any(vs_s3["u1"][T]["ci"][0] > 0 and vs_s3["u1"][T]["delta"] >= 0.010 for T in BUDGETS)
    sep_a3 = A3.first_sep(pt_a3b, pt_s3); sep_a4 = A3.first_sep(pt_a4, pt_s3)

    # a4 calibration (mean chosen p_hat vs realized answer rate; should be ~right by construction now)
    uid_order = [r["u"] for r in users]
    ph = np.array([a4_diags[u]["phat"] for u in uid_order])
    an = np.array([a4_diags[u]["ans"] for u in uid_order]).astype(float)
    calib_phat = ph.mean(0); calib_ans = an.mean(0)
    calib_gap = float(np.mean(calib_phat - calib_ans)); calib_corr = float(np.corrcoef(calib_phat, calib_ans)[0, 1])
    gcos = np.array([a4_diags[u]["gcos"] for u in uid_order]).mean(0)

    # rows
    rows = {
        "s1 concepts": rowdict("s1 concepts", pt_s1, ans_s1),
        "s3 popular-item (v2)": rowdict("s3 popular-item (v2)", pt_s3, ans_s3),
        "a3-blind": rowdict("a3-blind", pt_a3b, ans_a3b),
        "a4-blind": rowdict("a4-blind", pt_a4, ans_a4),
        "a3-table (PRIV)": rowdict("a3-table (PRIV)", pt_a3t, ans_a3t),
        "u1 clairvoyant (PRIV)": rowdict("u1 clairvoyant (PRIV)", pt_u1, ans_u1),
    }

    if a3b_beats or a4b_beats:
        branch = "C"; vtxt = ("BRANCH C (arena-v2) -- blind adaptive item-probing LIVES under measured "
                              "answerability: a blind prober beats the rebuilt s3 static (CI excl 0) at >=1 budget.")
    elif u1_beats:
        branch = "B"; vtxt = ("BRANCH B (arena-v2) -- prize EXISTS, DISCOVERY still the bottleneck: the "
                              "clairvoyant ceiling beats the rebuilt s3 static, but blind probing cannot capture it.")
    else:
        branch = "A"; vtxt = ("BRANCH A (arena-v2) -- no adaptivity prize over the rebuilt s3 static even with "
                              "measured answerability + a privileged ceiling.")

    # =============================================================== console
    print("\n==== ARENA-V2 (LLM-measured answerability) -- key arms (any/end NDCG@10; hit=ans turns/24) ====",
          flush=True)
    for name in ("s1 concepts", "s3 popular-item (v2)", "a3-blind", "a4-blind", "a3-table (PRIV)",
                 "u1 clairvoyant (PRIV)"):
        r = rows[name]
        print(f"  {name:24s} any24 {r['any@24']:.4f} end24 {r['end@24']:.4f} | hit {r['mean_ans_turns']:.1f}/24",
              flush=True)
    print(f"\n  HIT RATE s3: v1 {v1_hit_s3:.1f}/24 ({100*v1_hit_s3/24:.0f}%) -> v2 {v2_hit_s3:.1f}/24 "
          f"({100*v2_hit_s3/24:.0f}%); grid bank yes-rate {prov['grid_bank_yes_rate']:.3f}", flush=True)
    for lab, key in (("a3-blind vs s3", "a3_blind"), ("a4-blind vs s3", "a4_blind"),
                     ("a3-table(PRIV) vs s3", "a3_table"), ("u1(PRIV) vs s3", "u1")):
        for T in BUDGETS:
            c = vs_s3[key][T]
            print(f"  {lab} @T{T}: {c['delta']:+.4f}[{c['ci'][0]:+.4f},{c['ci'][1]:+.4f}]", flush=True)
    print(f"  CALIB a4: mean p_hat-answer gap {calib_gap:+.3f} (corr {calib_corr:+.2f}); "
          f"p_hat@t1 {calib_phat[0]:.3f} vs ans@t1 {calib_ans[0]:.3f}", flush=True)
    print(f"  a3-blind beats s3? {a3b_beats} | a4-blind beats s3? {a4b_beats} | u1(PRIV) beats s3? {u1_beats}",
          flush=True)
    print(f"  VERDICT: {vtxt}", flush=True)

    # =============================================================== persist JSON
    out = dict(
        config=dict(dataset="ML-25M", instrument="RecVAE-d512 + I2.5 learned fold", TMAX=TMAX,
                    budgets=list(BUDGETS), n_users=n, base_rate_threshold=BASE_RATE, sigma_star=SIGMA_STAR,
                    ease_topk_pop=EASE_TOPK_POP, ease_lambda=ease_lambda, ease_universe=ease_nuni,
                    fold_ckpt=P4.CKPT_BEST, best_val=blob["state"]["best_val"], bank_items=len(item_cids)),
        arena_v1_reproduction=dict(s1_any24=s1_any24_v1, s1_target=0.2103, s3_any24=s3_any24_v1,
                                   s3_target=0.2786, s3_hit=v1_hit_s3, ok=bool(v1_ok)),
        provenance=prov,
        hit_rates=dict(s3_v1=v1_hit_s3, s3_v2=v2_hit_s3, a3_blind=float(ans_a3b.mean()),
                       a4_blind=float(ans_a4.mean()), a3_table=float(ans_a3t.mean()),
                       u1=float(ans_u1.mean()), s1=float(ans_s1.mean())),
        schedules=dict(s1=[CANDS[c]["key"] for c in s1], s3_v1=[CANDS[c]["key"] for c in s3_v1],
                       s3_v2=[CANDS[c]["key"] for c in s3]),
        arms=rows,
        vs_s3={k: {str(T): vs_s3[k][T] for T in BUDGETS} for k in vs_s3},
        first_separation_vs_s3=dict(a3_blind=dict(turn=sep_a3[0], sign=sep_a3[1]),
                                    a4_blind=dict(turn=sep_a4[0], sign=sep_a4[1])),
        a4_calibration=dict(phat_by_turn=[float(x) for x in calib_phat],
                            ans_rate_by_turn=[float(x) for x in calib_ans],
                            mean_gap=calib_gap, corr=calib_corr,
                            ghat_true_genre_cos=[float(x) for x in gcos]),
        ndcg_curves={"s1 concepts": [float(x) for x in pt_s1.mean(0)],
                     "s3 popular-item (v2)": [float(x) for x in pt_s3.mean(0)],
                     "a3-blind": [float(x) for x in pt_a3b.mean(0)],
                     "a4-blind": [float(x) for x in pt_a4.mean(0)],
                     "a3-table (PRIV)": [float(x) for x in pt_a3t.mean(0)],
                     "u1 clairvoyant (PRIV)": [float(x) for x in pt_u1.mean(0)]},
        verdict=dict(branch=branch, text=vtxt, a3_blind_beats_s3=bool(a3b_beats),
                     a4_blind_beats_s3=bool(a4b_beats), u1_beats_s3=bool(u1_beats)),
        wall_min=round((time.time() - t0) / 60, 2))
    os.makedirs("experiments", exist_ok=True)
    json.dump(out, open(OUT_JSON, "w"), indent=1, default=str)
    print(f"[arena2] wrote {OUT_JSON}", flush=True)

    # =============================================================== append MD section
    def frow(name, r, cd=None):
        base = (f"| {name} | {r['any@8']:.4f}/{r['end@8']:.4f} | {r['any@16']:.4f}/{r['end@16']:.4f} | "
                f"{r['any@24']:.4f}/{r['end@24']:.4f} | {r['mean_ans_turns']:.1f} ({100*r['hit_rate']:.0f}%) |")
        if cd is not None:
            c = cd[24]; base += f" {c['delta']:+.4f}[{c['ci'][0]:+.4f},{c['ci'][1]:+.4f}] |"
        else:
            base += " -- |"
        return base

    md = []
    md.append("\n## ARENA-V2 (LLM-measured answerability)\n\n")
    md.append("Date 2026-07-08. Script `scripts/i25_phase4_arena2.py` (imports/reuses `i25_phase4_fair.py`, "
              "`i25_phase4_a3.py`, `i25_phase4_a4.py` UNMODIFIED). NO LLM calls (grids cached); deterministic; "
              "local compute.\n\n")
    md.append("**Two pre-registered answerability models.** The study fixes TWO environments and reports the "
              "adaptivity verdict under BOTH: **arena-v1** = STRUCTURAL lower-bound answerability (an item probe "
              "is answerable only if the user rated it in the known half; the sections above), and **arena-v2** "
              "(this section) = the project's VALIDATED **LLM-MEASURED** answerability (cached LLM judged grid + "
              "fitted pmodel, base answer-rate 0.732). The A4 diagnosis showed the v1 harsh rule is what starves "
              "blind adaptivity of hits; arena-v2 tests whether blind adaptive probing separates from the static "
              "once answerability is the measured, human-like model. The verdict may legitimately differ between "
              "the two -- that contrast is itself a headline finding.\n\n")
    md.append(f"**Arena-v1 reproduction (old rule, before switching):** s1 any@24 = {s1_any24_v1:.4f} "
              f"(target 0.2103), **s3 any@24 = {s3_any24_v1:.4f} (target 0.2786)**, s3 hit {v1_hit_s3:.1f}/24 "
              f"-> reproduced = {v1_ok}.\n\n")
    md.append("**Arena-v2 item answer model (environment-side only; agent arms stay exactly as blind/privileged "
              "as before).** An item probe is ANSWERABLE iff (a) the user rated it in the known half (as v1), OR "
              "(b) the cached LLM judged grid (gate + main-study, union; answerable if any grid judged YES) says "
              f"yes, OR (c) for a bank item NOT in that user's grid, the fitted pmodel with the user's TRUE "
              f"known-half genre_match has p_hat >= BASE_RATE={BASE_RATE} (env-side => true-profile features are "
              "LEGAL in the answering rule). ANSWER VALUE: rated -> real centered rating; answerable-but-unrated "
              f"-> **LLM/CF-predicted** = bank-restricted EASE (universe = top-{EASE_TOPK_POP} popular UNION the "
              f"{prov['n_bank']}-item bank, lambda={ease_lambda:.0f}, {ease_nuni} items; trained on KNOWN-portion "
              "ratings with all study-user held-out rows dropped -- cross-check A recipe) + Gaussian noise "
              f"sigma={SIGMA_STAR} stars seeded per (user,item), clipped [0.5,5], centered; native recall channel "
              "nat=item iff the noised predicted star >= 4 (the same channel v1 uses for a rated-liked item). "
              "Concepts/attributes UNCHANGED.\n\n")
    md.append(f"**Grid coverage of the {prov['n_bank']}-item probe bank:** mean "
              f"{prov['mean_bank_grid_coverage']:.1f}/{prov['n_bank']} items judged per user "
              f"({100*prov['mean_bank_grid_coverage_frac']:.1f}%); on those judged bank items the LLM YES-rate is "
              f"**{prov['grid_bank_yes_rate']:.3f}** (popular bank items are far more answerable than the 0.269 "
              "all-strata item base). Bank items outside a user's grid fall back to the pmodel threshold. "
              f"Provenance of item-probe outcomes across all (user,bank-item) cells: {prov['stat']}.\n\n")
    md.append("| arm | any/end @8 | any/end @16 | any/end @24 | hit (ansT) | delta-any@24 vs s3(v2) [CI] |\n"
              "|---|---|---|---|---|---|\n")
    md.append(frow("s3 popular-item (v2, opponent)", rows["s3 popular-item (v2)"]) + "\n")
    md.append(frow("s1 concepts", rows["s1 concepts"]) + "\n")
    md.append(frow("a3-blind", rows["a3-blind"], vs_s3["a3_blind"]) + "\n")
    md.append(frow("a4-blind", rows["a4-blind"], vs_s3["a4_blind"]) + "\n")
    md.append(frow("a3-table (PRIV, class ceiling)", rows["a3-table (PRIV)"], vs_s3["a3_table"]) + "\n")
    md.append(frow("u1 clairvoyant (PRIV, arena ceiling)", rows["u1 clairvoyant (PRIV)"], vs_s3["u1"]) + "\n")

    md.append("\n**Hit rates v1 vs v2 (mean answered turns / 24):** s3 v1 = "
              f"{v1_hit_s3:.1f} (~{100*v1_hit_s3/24:.0f}%) -> **s3 v2 = {v2_hit_s3:.1f} "
              f"(~{100*v2_hit_s3/24:.0f}%)**; a3-blind = {rows['a3-blind']['mean_ans_turns']:.1f}; "
              f"a4-blind = {rows['a4-blind']['mean_ans_turns']:.1f}; a3-table (ceiling) = "
              f"{rows['a3-table (PRIV)']['mean_ans_turns']:.1f}; u1 = {rows['u1 clairvoyant (PRIV)']['mean_ans_turns']:.1f}.\n\n")

    md.append("**THE verdict contrasts (per budget):**\n\n"
              "| budget T | a3-blind vs s3 [CI] | a4-blind vs s3 [CI] |\n|---|---|---|\n")
    for T in BUDGETS:
        c3 = vs_s3["a3_blind"][T]; c4 = vs_s3["a4_blind"][T]
        md.append(f"| {T} | {c3['delta']:+.4f}[{c3['ci'][0]:+.4f},{c3['ci'][1]:+.4f}] | "
                  f"{c4['delta']:+.4f}[{c4['ci'][0]:+.4f},{c4['ci'][1]:+.4f}] |\n")

    md.append(f"\n**First separation vs s3 (belief(t) paired CI excl 0):** a3-blind "
              f"{('turn '+str(sep_a3[0])+' (sign '+sep_a3[1]+')') if sep_a3[0] else 'NEVER within T=24'}; "
              f"a4-blind {('turn '+str(sep_a4[0])+' (sign '+sep_a4[1]+')') if sep_a4[0] else 'NEVER within T=24'}.\n\n")

    md.append("**a4 calibration under arena-v2 (per turn, mean chosen-probe p_hat vs realized answer rate -- "
              "roughly right by construction now, since the answer model IS the pmodel/grid):**\n\n| t | "
              + " | ".join(str(t + 1) for t in range(TMAX)) + " |\n")
    md.append("|" + "---|" * (TMAX + 1) + "\n")
    md.append("| mean chosen p_hat | " + " | ".join(f"{x:.3f}" for x in calib_phat) + " |\n")
    md.append("| realized answer rate | " + " | ".join(f"{x:.3f}" for x in calib_ans) + " |\n")
    md.append(f"\nMean p_hat - answer-rate gap = {calib_gap:+.3f} (corr {calib_corr:+.2f}) -- vs arena-v1's "
              "+0.720 gap: the surrogate is now "
              f"{'well-aligned in level' if abs(calib_gap) < 0.15 else ('still optimistic' if calib_gap > 0 else 'pessimistic')} "
              "because the environment answers by the SAME measured model the agent ranks with.\n\n")

    md.append("**NDCG@10(t) curves (t=1..24):**\n\n| arm | " + " | ".join(f"t{t+1}" for t in range(TMAX)) + " |\n")
    md.append("|" + "---|" * (TMAX + 1) + "\n")
    for lab, ptm in (("s3 popular-item (v2)", pt_s3), ("a3-blind", pt_a3b), ("a4-blind", pt_a4),
                     ("a3-table (PRIV)", pt_a3t), ("u1 clairvoyant (PRIV)", pt_u1)):
        md.append(f"| {lab} | " + " | ".join(f"{c:.3f}" for c in ptm.mean(0)) + " |\n")

    md.append(f"\n**VERDICT:** {vtxt} (a3-blind beats s3={a3b_beats}, a4-blind beats s3={a4b_beats}, "
              f"u1 PRIV beats s3={u1_beats}.) Read against the arena-v1 verdict (Branch B, blind loses): the "
              "adaptivity verdict is reported under BOTH pre-registered answerability models.\n\n")

    md.append("**ASSUMPTIONS / judgment calls (arena-v2):**\n"
              f"1. Grid coverage/fallback: an item probe is answerable if the user rated it in the known half, "
              "OR the gate+main-study LLM grids (union; YES if ANY grid judged YES) say yes; for bank items NOT "
              f"in that user's grid (mean grid coverage {prov['mean_bank_grid_coverage']:.1f}/{prov['n_bank']} "
              f"items/user), fall back to the fitted pmodel with the user's TRUE known-half genre_match, "
              f"thresholded at p_hat >= BASE_RATE={BASE_RATE} (the calibrated MAIN-STUDY grid base answer-rate). "
              "The answering rule is environment-side, so true-profile features are legal here; the AGENT arms "
              "(a3/a4 blind) never see it.\n"
              f"2. EASE rebuild: bank-restricted EASE (cross-check A recipe), universe = top-{EASE_TOPK_POP} "
              f"popular UNION the {prov['n_bank']} bank items = {ease_nuni} items, L2 lambda={ease_lambda:.0f}, "
              "binary item-item weights, item-mean-centered prediction. Trained on the KNOWN-portion ratings of "
              "the full ML-25M population with ALL study users' seed-123 held-out interactions dropped (never "
              "trains on recommendation targets). Predicts each bank item for each study user from their "
              f"known-half context. Cached to `{EASE_CACHE}` (rebuilt on bank/uid mismatch).\n"
              f"3. Value-channel noise: answerable-but-unrated value = EASE predicted star + N(0,{SIGMA_STAR}^2) "
              "stars, seeded DETERMINISTICALLY per (user,item) via rng((u*2654435761+j) mod 2^32), clipped "
              "[0.5,5], then centered by the user's known-half mean. Labelled LLM/CF-predicted (sensitivity-only "
              "convention). Rated items keep their REAL centered rating with no noise.\n"
              "4. Native recall channel: an answerable item contributes a native (full-factor) recall token iff "
              "its star (real, or noised-predicted) >= 4 -- identical to the v1 rule for rated-liked items; the "
              "probe is system-selected (not open recall).\n"
              "5. s3 greedy REBUILT under arena-v2 over the same top-60-coverage candidate pool; s1 concepts and "
              "the coverage prior / granularity g are unchanged (structural coverage) for comparability with v1.\n"
              "6. All harness conventions inherited UNMODIFIED: refusal = no-op turn (belief unchanged), user "
              f"retained, all {n} users in every mean; paired per-user bootstrap BOOT={P4.BOOT} seed={P4.SEED}; "
              "u1/a3-table use the (now v2) TRUE answerability table (privileged, labelled); a3/a4 blind arms "
              "condition only on answers/refusals; co-known cosine from trU minus study users (leak=0 asserted).\n\n")

    with open(OUT_MD, "a", encoding="utf-8") as f:
        f.write("".join(md))
    print(f"[arena2] appended section to {OUT_MD}; wall {out['wall_min']}m", flush=True)


if __name__ == "__main__":
    main()
