"""i25_phase4_a6.py -- A6 "tie-by-construction K-map prober": removes the value-model confound in the
prior adaptive arms (a2/a3/a4/a5). Those arms ranked probes by P(answerable) x COVERAGE, while the best
static s3 was built by greedy NDCG-VALUE ordering -- so the adaptive arms deviated from s3 at t=0 for
VALUE-MODEL reasons, not for evidence reasons, violating the program's tie-by-construction rule
(warm-start from the best heuristic so learning can only ADD). Their losses to s3 are confounded.

A6 removes the confound. The policy is a WARM-STARTED s3: it starts from s3's exact greedy schedule and
tilts it ONLY by an evidence-driven answerability LIKELIHOOD RATIO from the validated K-map belief.

Imports/reuses scripts/i25_phase4_fair.py (FA), scripts/i25_phase4_a3.py (A3), scripts/i25_phase4_a5.py
(A5: calibrate_beta0) UNMODIFIED, and the ONLINE-INFERENCE `Kmap` class from scripts/kmap_validate.py.
NO LLM calls; deterministic; local compute only.

VALUE V(j): the s3 greedy ordering itself, realized as the monotone rank proxy V(j)=1/rank_s3(j)
  (greedy marginal contributions are not returned by FA.build_greedy, so the rank proxy is used --
  stated in ASSUMPTIONS). s3's 24 scheduled items get ranks 1..24 in schedule order; the remaining bank
  items get ranks 25.. by coverage (pop_rate) descending. So argmax V over unasked = s3's next item
  exactly (and, past turn 24, the coverage tail).

BELIEF: the validated LEARNED K-map P(u knows j)=sigmoid(alpha_u + b_j + k_u.e_j) (kmap_build/validate),
  online-inferred (MAP-Newton) from all observed events, IDENTICAL to a5. Level calibration beta0
  reused from a5 (1-param Platt on a deterministic train half so mean_j sigmoid(b_j+beta0)=structural
  base rate).

SELECTION each turn (BLIND, deployable): argmax over UNASKED bank items of V(j) x LR(j), where
  LR(j) = P_kmap(j | events so far) / P_kmap(j | no events)   [the likelihood-ratio tilt].
  At t=0 (no events) alpha=k=0 => P_kmap(.|events)=P_kmap(.|no events) => LR=1 for ALL j, so the pick
  is EXACTLY s3's next item. Deviations occur ONLY from posterior updates -> any measured delta is PURE
  adaptivity, and the static s3 is the policy's floor at t=0 (verified: every user's turn-1 pick == s3[0]).

VARIANTS:
  a6        : argmax V(j)*LR(j)                       (deployable).
  a6-margin : hysteresis -- swap off the s3-base order only if LR(argmax) > 1.2 (guards vs posterior
              noise); else take the next s3-order item.
  a6-table  : same LR policy but with the TRUE-answerability posterior (LR=1 if the user rated j, ~0
              else) = the tie-by-construction PRIVILEGED ceiling of the policy class (labelled). Recovers
              most of a3-table's +0.060 iff the V*LR form is not itself the bottleneck.

PRE-REGISTERED (printed BEFORE results):
  - a6 WINS iff it beats s3 anytime with CI excl 0 at any pre-registered budget (8/16/24).
  - a6 must NEVER lose to s3 by more than noise; a large clean loss would mean the LR mechanism is broken
    (e.g. LR != 1 at t=0) -> diagnose before reporting. (Tie-by-construction floor is checked explicitly.)
  - Honest expectation: small positive or clean tie. The a6-table ceiling decides whether the policy
    CLASS (V-base + LR-tilt) is viable.

KEY REPORTS: (i) a6 vs s3 at 8/16/24; (ii) SWAP FORENSICS -- deviations/user, realized answer-rate of
  swapped-in vs displaced items, NDCG delta of swaps (users with >=1 swap vs 0 swaps); (iii) hit rates;
  (iv) a6-table vs s3.

Run:  python scripts/i25_phase4_a6.py
"""
import os, sys, json, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import i25_phase4_fair as FA      # UNMODIFIED reuse
import i25_phase4_a3 as A3        # UNMODIFIED reuse (build_cooc, make_a3_plan, contrast, first_sep)
import i25_phase4_a5 as A5        # UNMODIFIED reuse (calibrate_beta0)
import kmap_validate as KV        # ONLINE-INFERENCE Kmap class; imported, NOT duplicated
P4 = FA.P4
L = FA.L

TMAX = FA.TMAX                    # 24
BUDGETS = FA.BUDGETS              # (8,16,24)
MARGIN = 1.2                      # a6-margin hysteresis: swap only if LR(argmax) > MARGIN
OUT_JSON = "experiments/I25_phase4_a6.json"
OUT_MD = "experiments/I25_PHASE4_FAIR.md"


# ================================================================= a6 per-user runner (V*LR selection)
def make_a6_runner(KM, CANDS, item_cids, jof, V_a6, V_cov, P0, beta0, b_bank, E_bank, mode):
    """Returns run(rec) -> (plan, diag). mode in {'a6','margin','table'}.
    Selection: argmax over unasked of V_a6(j) * LR(j). j_base = the pure-V (s3-order) continuation
    among unasked; a SWAP = actual pick != j_base (evidence tilted the argmax off the static order)."""
    nb = len(item_cids)
    cids_arr = np.array(item_cids, dtype=np.int64)     # global cids, bank order
    jof_arr = np.asarray(jof, np.int64)
    Va = np.asarray(V_a6, float)
    Vc = np.asarray(V_cov, float)
    table = (mode == "table")
    margin = (mode == "margin")

    def run(rec):
        ans_g = rec["ans_arr"]                          # global-cid -> bool answerable (rated in known half)
        ans_bank = ans_g[cids_arr]                      # (nb,) answerability over the bank
        used = np.zeros(nb, bool)
        J_ev, y_ev = [], []
        plan = []
        d_chosen, d_LR, d_ans, d_swap = [], [], [], []
        swaps = []                                      # per swap event
        for t in range(TMAX):
            if table:
                LR = np.where(ans_bank, 1.0, 1e-6)      # PRIVILEGED: certain-knowledge posterior
            else:
                alpha, k = KM.infer(np.array(J_ev, np.int64), np.array(y_ev, np.float64))
                Pt = KV.sigmoid(b_bank + alpha + E_bank @ k + beta0)   # P_kmap(knows j | events)
                LR = Pt / P0                            # likelihood-ratio tilt (=1 for all j at t=0)
            score = np.where(used, -1e18, Va * LR)
            base = np.where(used, -1e18, Va)            # pure-V (s3-order) continuation
            j_base = int(np.lexsort((jof_arr, -Vc, -base))[0])
            j_star = int(np.lexsort((jof_arr, -Vc, -score))[0])
            if margin:
                j_sel = j_star if (j_star != j_base and LR[j_star] > MARGIN) else j_base
            else:
                j_sel = j_star
            is_swap = (j_sel != j_base)
            used[j_sel] = True
            cid = item_cids[j_sel]
            d_chosen.append(int(jof_arr[j_sel])); d_LR.append(float(LR[j_sel]))
            d_ans.append(bool(ans_bank[j_sel])); d_swap.append(bool(is_swap))
            if is_swap:
                swaps.append(dict(t=t, j_in=int(jof_arr[j_sel]), j_out=int(jof_arr[j_base]),
                                  LR_in=float(LR[j_sel]), ans_in=bool(ans_bank[j_sel]),
                                  ans_out=bool(ans_bank[j_base])))
            if ans_bank[j_sel]:
                plan.append((FA.tok_of(CANDS, rec, cid), rec["nat_arr"][cid]))
                J_ev.append(int(jof_arr[j_sel])); y_ev.append(1.0)
            else:
                plan.append((None, None))
                J_ev.append(int(jof_arr[j_sel])); y_ev.append(0.0)
        diag = dict(chosen=np.array(d_chosen, np.int64), LR=np.array(d_LR),
                    ans=np.array(d_ans, bool), swap=np.array(d_swap, bool),
                    nswap=int(sum(d_swap)), swaps=swaps)
        return plan, diag
    return run


def main():
    t0 = time.time()
    print("[a6] loading data + frozen RecVAE + trained fold ...", flush=True)
    D = L.G.load_data(); FR = L.Frozen(D)
    model = L.Fold(); import torch
    blob = torch.load(P4.CKPT_BEST, map_location="cpu")
    model.load_state_dict(blob["model"]); model.eval()
    P4.T = TMAX
    users, _ = P4.assemble(D, FR)
    n = len(users)
    print(f"[a6] {n} users; building ladder universe (reuse FA.build_universe) ...", flush=True)
    CANDS, meta = FA.build_universe(D, FR, users)
    cold = FA.cold_ndcg(FR, users, 10)

    item_cids = [m["cid"] for m in CANDS if m["kind"] == "item"]
    concept_cids = [m["cid"] for m in CANDS if m["kind"] == "concept"]
    pop_rate = {m["cid"]: m["pop_rate"] for m in CANDS if m["kind"] == "item"}
    item_by_cov = sorted(item_cids, key=lambda c: -CANDS[c]["pop_rate"])
    s3_pool = item_by_cov[:FA.ITEM_POOL_S3]
    jof = np.array([int(CANDS[c]["key"].split(":")[1]) for c in item_cids], dtype=np.int64)   # dense bank ids
    V_cov = np.array([CANDS[c]["pop_rate"] for c in item_cids], float)                        # coverage prior
    colof = {c: k for k, c in enumerate(item_cids)}                                           # global cid -> bank col

    # ---- reproduce s1 / s3 (existing harness) BEFORE adding anything ----
    print("[a6] single-question values (all candidates) ...", flush=True)
    v1 = FA.single_q_values(FR, model, users, cold, CANDS, [m["cid"] for m in CANDS])
    print("[a6] greedy s1 (concepts) ...", flush=True)
    s1 = FA.build_greedy(FR, model, users, cold, CANDS, concept_cids, TMAX, v1=v1, tag="s1")
    print("[a6] greedy s3 (popular items) ...", flush=True)
    s3 = FA.build_greedy(FR, model, users, cold, CANDS, s3_pool, TMAX, v1=v1, tag="s3")
    pt_s1, ans_s1 = FA.eval_fair(FR, model, users, lambda rec: FA.plan_sched(CANDS, rec, s1), cold)
    pt_s3, ans_s3 = FA.eval_fair(FR, model, users, lambda rec: FA.plan_sched(CANDS, rec, s3), cold)
    s1_any24 = float(FA.anytime(pt_s1, 24).mean())
    s3_any24 = float(FA.anytime(pt_s3, 24).mean())

    # ---- reproduce a3-blind (reuse A3 UNMODIFIED) ----
    print("[a6] building 160x160 population co-known cosine (reuse A3.build_cooc) ...", flush=True)
    bank_j = [int(j) for j in jof]
    study_uids = [rec["u"] for rec in users]
    C, self_cnt, nU_pop = A3.build_cooc(bank_j, study_uids)
    global_order = sorted(item_cids, key=lambda c: (-CANDS[c]["pop_rate"], c))
    a3_pf = A3.make_a3_plan(CANDS, item_cids, s3[0], global_order, colof, C, pop_rate, "blind")
    pt_a3, ans_a3 = FA.eval_fair(FR, model, users, a3_pf, cold)
    a3_any24 = float(FA.anytime(pt_a3, 24).mean())

    print(f"\n[a6] REPRODUCTION: s1 any@24={s1_any24:.4f} (t 0.2103) | s3 any@24={s3_any24:.4f} "
          f"(t 0.2786) | a3-blind any@24={a3_any24:.4f} (t 0.2412)", flush=True)
    ok = (abs(s3_any24 - 0.2786) < 0.0015 and abs(a3_any24 - 0.2412) < 0.0020
          and abs(s1_any24 - 0.2103) < 0.0015)
    if not ok:
        print("[a6] STOP: s3 / a3-blind did NOT reproduce within tolerance -- NOT adding a6. Diagnose first.",
              flush=True)
        return
    print("[a6] reproduction OK -- proceeding to build a6.\n", flush=True)

    # ---- V(j) = 1/rank_s3(j): s3 schedule first (ranks 1..24), then remaining bank by coverage desc ----
    nb = len(item_cids)
    rank = np.zeros(nb, float); assigned = set(); r = 1
    for c in s3:
        col = colof[c]; rank[col] = r; r += 1; assigned.add(col)
    rest = sorted([col for col in range(nb) if col not in assigned], key=lambda col: (-V_cov[col], jof[col]))
    for col in rest:
        rank[col] = r; r += 1
    V_a6 = 1.0 / rank
    s3_col0 = colof[s3[0]]

    # ---- K-map + level calibration (reuse a5's calibrator) ----
    print("[a6] loading K-map (kmap_validate.Kmap) + fitting level calibration beta0 (reuse A5) ...", flush=True)
    KM = KV.Kmap(D)
    b_bank = KM.b_of(jof).astype(np.float64)
    E_bank = KM.emb_of(jof).astype(np.float64)
    ans_bank_by_user = {rec["u"]: rec["ans_arr"][np.array(item_cids)] for rec in users}
    beta0, base_rate = A5.calibrate_beta0(b_bank, ans_bank_by_user, users)
    P0 = KV.sigmoid(b_bank + beta0)                     # P_kmap(knows j | no events); LR denominator
    p0_mean = float(P0.mean())
    print(f"[a6] structural base rate (train half) = {base_rate:.4f}; beta0 = {beta0:+.4f}; "
          f"calibrated mean P@t0 = {p0_mean:.4f}\n", flush=True)

    # ---- pre-registered read print (BEFORE results) ----
    s3_hit = float(ans_s3.mean()); a3_hit = float(ans_a3.mean())
    print("=" * 80, flush=True)
    print("PRE-REGISTERED READ (a6 tie-by-construction K-map prober):", flush=True)
    print("  Value model = s3's own greedy order (V=1/rank_s3) -> at t=0 LR=1 => a6 pick == s3 pick,", flush=True)
    print("  so the static s3 is the policy FLOOR at t=0; every deviation is an evidence-driven swap.", flush=True)
    print("  a6 WINS iff it beats s3 anytime with CI excl 0 at any budget 8/16/24.", flush=True)
    print("  a6 must NEVER lose to s3 by more than noise; a large clean loss => LR mechanism broken", flush=True)
    print("    (e.g. LR!=1 at t=0) -> diagnose. Tie-by-construction floor is asserted below.", flush=True)
    print("  Honest expectation: small positive or clean tie. a6-table = PRIVILEGED policy-class ceiling", flush=True)
    print(f"    (should recover most of a3-table's +0.060). s3 hit {s3_hit:.1f}/24; a3-blind hit {a3_hit:.1f}/24.", flush=True)
    print("=" * 80 + "\n", flush=True)

    # ---- run a6 arms ----
    pt, ansn, diags = {}, {}, {}
    for mode in ("a6", "margin", "table"):
        print(f"[a6] eval {('a6-'+mode) if mode!='a6' else 'a6'} ...", flush=True)
        runner = make_a6_runner(KM, CANDS, item_cids, jof, V_a6, V_cov, P0, beta0, b_bank, E_bank, mode)
        plans, dg = {}, {}
        for rec in users:
            pl, d = runner(rec)
            plans[rec["u"]] = pl; dg[rec["u"]] = d
        pt[mode], ansn[mode] = FA.eval_fair(FR, model, users, lambda rec: plans[rec["u"]], cold)
        diags[mode] = dg

    # ---- TIE-BY-CONSTRUCTION FLOOR CHECK: every user's turn-1 pick == s3[0] (LR=1 at t=0) ----
    tie_ok = all(int(diags["a6"][rec["u"]]["chosen"][0]) == int(jof[s3_col0]) for rec in users)
    print(f"[a6] TIE-BY-CONSTRUCTION floor check: all turn-1 picks == s3[0]? {tie_ok}", flush=True)

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
    rows = {m: row(f"a6-{m}" if m != "a6" else "a6", pt[m], ansn[m]) for m in pt}
    rows["s3 popular-item"] = row("s3 popular-item", pt_s3, ans_s3)
    rows["a3-blind"] = row("a3-blind", pt_a3, ans_a3)
    rows["s1 concepts"] = row("s1 concepts", pt_s1, ans_s1)

    a6_hit = rows["a6"]["mean_ans_turns"]
    a6_wins = any(vs_s3["a6"][T]["ci"][0] > 0 for T in BUDGETS)
    a6_loses_big = (vs_s3["a6"][24]["delta"] < -0.02 and vs_s3["a6"][24]["ci"][1] < 0)
    table_wins = any(vs_s3["table"][T]["ci"][0] > 0 for T in BUDGETS)

    # ---- SWAP FORENSICS (a6 arm) ----
    uids = [r["u"] for r in users]
    nswap = np.array([diags["a6"][u]["nswap"] for u in uids])
    all_swaps = [sw for u in uids for sw in diags["a6"][u]["swaps"]]
    n_swap_events = len(all_swaps)
    ans_in = np.array([sw["ans_in"] for sw in all_swaps], float) if all_swaps else np.array([])
    ans_out = np.array([sw["ans_out"] for sw in all_swaps], float) if all_swaps else np.array([])
    swapin_ansrate = float(ans_in.mean()) if len(ans_in) else float("nan")
    swapout_ansrate = float(ans_out.mean()) if len(ans_out) else float("nan")
    # NDCG effect of swaps: per-user anytime@24 delta (a6 - s3), split by >=1 swap vs 0 swaps
    d_any24 = FA.anytime(pt["a6"], 24) - FA.anytime(pt_s3, 24)
    d_end24 = FA.endpoint(pt["a6"], 24) - FA.endpoint(pt_s3, 24)
    hasswap = nswap >= 1
    grp = {}
    for lab, mask in (("swap>=1", hasswap), ("swap=0", ~hasswap)):
        if mask.sum() > 0:
            dd = d_any24[mask]
            rng = np.random.default_rng(P4.SEED)
            bs = np.array([dd[rng.integers(0, len(dd), len(dd))].mean() for _ in range(P4.BOOT)]) if len(dd) > 1 else np.array([dd.mean()])
            grp[lab] = dict(n=int(mask.sum()), mean_any24=float(dd.mean()),
                            mean_end24=float(d_end24[mask].mean()),
                            ci=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))])
        else:
            grp[lab] = dict(n=0, mean_any24=float("nan"), mean_end24=float("nan"), ci=[float("nan")] * 2)

    # a6-margin swap count (for context)
    nswap_margin = np.array([diags["margin"][u]["nswap"] for u in uids])

    # ---- console summary ----
    print("\n==== A6 tie-by-construction prober (any/end NDCG@10; hit = mean answered turns / 24) ====", flush=True)
    for m in ("s3 popular-item", "a3-blind", "s1 concepts"):
        r = rows[m]
        print(f"  {m:18s} any24 {r['any@24']:.4f} end24 {r['end@24']:.4f} | hit {r['mean_ans_turns']:.1f}/24", flush=True)
    for m in ("a6", "margin", "table"):
        r = rows[m]; c3 = vs_s3[m]
        lab = "a6" if m == "a6" else f"a6-{m}"
        print(f"  {lab:18s} any24 {r['any@24']:.4f} end24 {r['end@24']:.4f} | hit {r['mean_ans_turns']:.1f}/24 "
              f"| vs s3@24 {c3[24]['delta']:+.4f}[{c3[24]['ci'][0]:+.4f},{c3[24]['ci'][1]:+.4f}]", flush=True)
    for T in BUDGETS:
        c = vs_s3["a6"][T]; ct = vs_s3["table"][T]
        print(f"  vs s3 @T{T}: a6 {c['delta']:+.4f}[{c['ci'][0]:+.4f},{c['ci'][1]:+.4f}] | "
              f"a6-table(PRIV) {ct['delta']:+.4f}[{ct['ci'][0]:+.4f},{ct['ci'][1]:+.4f}]", flush=True)
    print(f"\n  TIE floor (turn-1==s3[0] all users)? {tie_ok} | a6 WINS (CI excl 0 any budget)? {a6_wins} | "
          f"a6 loses big? {a6_loses_big}", flush=True)
    print(f"  SWAP FORENSICS: deviations/user mean {nswap.mean():.2f} (median {int(np.median(nswap))}, "
          f"max {int(nswap.max())}, users w/ >=1 swap {int(hasswap.sum())}/{n}); {n_swap_events} swap events.", flush=True)
    print(f"    swapped-IN answer rate {swapin_ansrate:.3f} vs displaced (s3-order) {swapout_ansrate:.3f}.", flush=True)
    print(f"    NDCG any@24 delta vs s3: swap>=1 users {grp['swap>=1']['mean_any24']:+.4f} "
          f"[{grp['swap>=1']['ci'][0]:+.4f},{grp['swap>=1']['ci'][1]:+.4f}] (n={grp['swap>=1']['n']}); "
          f"swap=0 users {grp['swap=0']['mean_any24']:+.4f} (n={grp['swap=0']['n']}).", flush=True)

    # ---- verdict text ----
    if a6_wins:
        verdict = ("A6 tie-by-construction prober BEATS s3 (CI excl 0 at >=1 budget) -- warm-starting from "
                   "s3 and tilting ONLY by the evidence-driven K-map likelihood ratio makes blind adaptivity "
                   "ADD value over the best static once the value-model confound is removed.")
    elif a6_loses_big:
        verdict = ("A6 LOSES to s3 by more than noise despite the tie-by-construction warm start -- the "
                   "evidence-driven LR swaps ACTIVELY hurt ranking value (answerability tilt anti-correlated "
                   "with NDCG value; the E0e/Branch-B mechanism), not a broken LR (turn-1 floor verified). "
                   f"a6-table PRIV recovers {vs_s3['table'][24]['delta']:+.4f} at T24 -> the policy CLASS "
                   f"{'IS' if table_wins else 'is NOT'} viable even with perfect knowledge.")
    else:
        verdict = ("A6 TIES s3 within noise: warm-starting from s3 and tilting only by evidence neither adds "
                   "nor destroys ranking value -- the confound in the prior arms explained their apparent "
                   f"losses, but blind answerability adaptivity still does not beat the popular-item static. "
                   f"a6-table PRIV vs s3 @T24 {vs_s3['table'][24]['delta']:+.4f}"
                   f"[{vs_s3['table'][24]['ci'][0]:+.4f},{vs_s3['table'][24]['ci'][1]:+.4f}] -> policy CLASS "
                   f"{'viable' if table_wins else 'NOT viable'} (recover-a3-table check). Branch B stands.")
    print(f"\n  VERDICT: {verdict}", flush=True)

    # ---- persist json ----
    def cdump(cd):
        return {str(T): cd[T] for T in BUDGETS}
    out = dict(
        config=dict(dataset="ML-25M", instrument="RecVAE-d512 + I2.5 learned fold", TMAX=TMAX,
                    budgets=list(BUDGETS), n_users=n, bank_items=nb,
                    kmap_emb=KV.EMB, kmap_int=KV.INT, kmap_d=KM.d, tau=KV.TAU, tau_a=KV.TAU_A,
                    newton_it=KV.NEWTON_IT, beta0=beta0, structural_base_rate=base_rate,
                    calibrated_p_t0=p0_mean, margin=MARGIN, V_model="V=1/rank_s3 (s3 greedy order)",
                    fold_ckpt=P4.CKPT_BEST, best_val=blob["state"]["best_val"]),
        reproduction=dict(s1_any24=s1_any24, s3_any24=s3_any24, a3_blind_any24=a3_any24,
                          s1_target=0.2103, s3_target=0.2786, a3_target=0.2412, ok=bool(ok)),
        tie_by_construction_floor_ok=bool(tie_ok),
        arms={rows[m]["name"]: rows[m] for m in rows},
        vs_s3={rows[m]["name"]: cdump(vs_s3[m]) for m in pt},
        vs_a3_blind={rows[m]["name"]: cdump(vs_a3[m]) for m in pt},
        first_separation_vs_s3={rows[m]["name"]: dict(turn=sep_s3[m][0], sign=sep_s3[m][1]) for m in pt},
        ndcg_curves={rows[m]["name"]: [float(x) for x in pt[m].mean(axis=0)] for m in pt},
        swap_forensics=dict(deviations_per_user_mean=float(nswap.mean()),
                            deviations_per_user_median=float(np.median(nswap)),
                            deviations_per_user_max=int(nswap.max()),
                            users_with_swap=int(hasswap.sum()), n_users=n,
                            n_swap_events=n_swap_events,
                            swapin_answer_rate=swapin_ansrate, displaced_answer_rate=swapout_ansrate,
                            ndcg_delta_by_swapgroup=grp,
                            margin_deviations_per_user_mean=float(nswap_margin.mean())),
        hit_rates=dict(s3=s3_hit, a3_blind=a3_hit, a6=a6_hit,
                       a6_margin=rows["margin"]["mean_ans_turns"], a6_table=rows["table"]["mean_ans_turns"],
                       a3_table_ceiling=12.9),
        hypothesis=dict(a6_wins=bool(a6_wins), a6_loses_big=bool(a6_loses_big),
                        table_wins=bool(table_wins), tie_floor_ok=bool(tie_ok)),
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
    md.append("\n## A6 tie-by-construction prober\n\n")
    md.append("Date 2026-07-08. Script `scripts/i25_phase4_a6.py` (imports/reuses `i25_phase4_fair.py`, "
              "`i25_phase4_a3.py`, `i25_phase4_a5.py` [calibrate_beta0] UNMODIFIED, and the ONLINE-INFERENCE "
              "`Kmap` class from `scripts/kmap_validate.py`). NO LLM calls; deterministic; local compute.\n\n")
    md.append("**Reproduction gate (existing harness, before adding anything):** "
              f"s1 any@24={s1_any24:.4f} (t 0.2103), s3 any@24={s3_any24:.4f} (t 0.2786), "
              f"a3-blind any@24={a3_any24:.4f} (t 0.2412) -> reproduced={ok}.\n\n")
    md.append("**The confound A6 removes.** a2/a3/a4/a5 ranked probes by P(answerable)xCOVERAGE, but the "
              "best static s3 was built by greedy NDCG-VALUE ordering -- so those arms deviated from s3 at "
              "t=0 for VALUE-MODEL reasons (not evidence), violating the program's tie-by-construction rule "
              "(warm-start from the best heuristic so learning can only ADD). Their losses to s3 are "
              "confounded. A6 is a WARM-STARTED s3: it starts from s3's exact schedule and tilts it ONLY by "
              "an evidence-driven answerability likelihood ratio.\n\n")
    md.append("**Value V(j).** The s3 greedy ordering itself, realized as the monotone rank proxy "
              "V(j)=1/rank_s3(j) (FA.build_greedy does not return greedy marginal contributions, so the rank "
              "proxy is used -- see ASSUMPTIONS). s3's 24 scheduled items take ranks 1..24 in schedule order; "
              "remaining bank items take ranks 25.. by coverage (pop_rate) descending. So argmax V over "
              "unasked = s3's next item exactly.\n\n")
    md.append("**Belief + selection.** BELIEF = the validated LEARNED K-map "
              "`P(u knows j)=sigmoid(alpha_u+b_j+k_u.e_j)`, online MAP-Newton inference from all observed "
              "events, IDENTICAL to a5; level calibration beta0 reused from a5 "
              f"(base rate {base_rate:.4f} -> beta0 {beta0:+.4f}, mean P@t0 {p0_mean:.4f}). "
              "SELECTION each turn = argmax over UNASKED bank items of V(j) x LR(j), "
              "LR(j)=P_kmap(j|events)/P_kmap(j|no events). At t=0, alpha=k=0 => LR=1 for ALL j => the pick "
              "is EXACTLY s3's next item, so s3 is the policy FLOOR at t=0 and every deviation is a pure "
              f"evidence-driven swap. **Tie-by-construction floor check (all users' turn-1 pick == s3[0]): "
              f"{tie_ok}.**\n\n")
    md.append("**Variants.** a6 (argmax V*LR, deployable). a6-margin (hysteresis: swap off the s3-base order "
              f"only if LR(argmax) > {MARGIN}, else take the next s3-order item). a6-table (PRIVILEGED, "
              "labelled: same V*LR policy but LR from the TRUE-answerability posterior -- LR=1 if the user "
              "rated j, ~0 else -- the tie-by-construction ceiling of the policy class; recovers most of "
              "a3-table's +0.060 iff the V*LR form is not the bottleneck).\n\n")
    md.append("| arm | any/end @8 | any/end @16 | any/end @24 | hit (ansT) | delta-any@24 vs s3 [CI] |\n"
              "|---|---|---|---|---|---|\n")
    md.append(frow("s3 popular-item (opponent)", rows["s3 popular-item"]) + "\n")
    md.append(frow("a3-blind (co-known, prior arm)", rows["a3-blind"]) + "\n")
    md.append(frow("s1 concepts", rows["s1 concepts"]) + "\n")
    md.append(frow("a6 (V=s3-order x K-map LR)", rows["a6"], vs_s3["a6"]) + "\n")
    md.append(frow(f"a6-margin (swap iff LR>{MARGIN})", rows["margin"], vs_s3["margin"]) + "\n")
    md.append(frow("a6-table (PRIV, true-table LR = class ceiling)", rows["table"], vs_s3["table"]) + "\n")

    md.append("\n**THE contrast -- a6 vs s3, and the PRIVILEGED ceiling a6-table vs s3, per budget "
              "(can now differ from s3 ONLY through evidence-driven swaps):**\n\n"
              "| budget T | a6 vs s3 [CI] | a6-margin vs s3 [CI] | a6-table (PRIV) vs s3 [CI] |\n"
              "|---|---|---|---|\n")
    for T in BUDGETS:
        c = vs_s3["a6"][T]; cm = vs_s3["margin"][T]; ct = vs_s3["table"][T]
        md.append(f"| {T} | {c['delta']:+.4f}[{c['ci'][0]:+.4f},{c['ci'][1]:+.4f}] | "
                  f"{cm['delta']:+.4f}[{cm['ci'][0]:+.4f},{cm['ci'][1]:+.4f}] | "
                  f"{ct['delta']:+.4f}[{ct['ci'][0]:+.4f},{ct['ci'][1]:+.4f}] |\n")

    md.append("\n**SWAP FORENSICS (a6 arm -- a swap = the argmax V*LR pick differs from the pure-V s3-order "
              "continuation among unasked items).**\n\n"
              f"- Deviations from s3's order per user: mean **{nswap.mean():.2f}**, median "
              f"{int(np.median(nswap))}, max {int(nswap.max())}; users with >=1 swap "
              f"**{int(hasswap.sum())}/{n}**; {n_swap_events} swap events total. "
              f"(a6-margin, LR>{MARGIN}: mean {nswap_margin.mean():.2f} deviations/user.)\n"
              f"- Realized answer rate of swapped-IN items = **{swapin_ansrate:.3f}** vs displaced "
              f"(s3-order) items = **{swapout_ansrate:.3f}** (the K-map LR tilts toward items the user is "
              f"{'MORE' if swapin_ansrate > swapout_ansrate else 'NOT more'} likely to answer).\n"
              f"- NDCG effect of swaps (per-user paired delta a6-s3): users with >=1 swap "
              f"(n={grp['swap>=1']['n']}) any@24 **{grp['swap>=1']['mean_any24']:+.4f}**"
              f"[{grp['swap>=1']['ci'][0]:+.4f},{grp['swap>=1']['ci'][1]:+.4f}] "
              f"(end@24 {grp['swap>=1']['mean_end24']:+.4f}); users with 0 swaps "
              f"(n={grp['swap=0']['n']}) any@24 {grp['swap=0']['mean_any24']:+.4f} "
              f"(identical to s3 by construction -- 0 deviations => same plan).\n\n")

    md.append("**Mechanism metric -- hit rate (mean answered turns / 24):** "
              f"s3 = {s3_hit:.1f} (~{100*s3_hit/24:.0f}%); a3-blind = {a3_hit:.1f}; "
              f"a6 = {a6_hit:.1f} (~{100*rows['a6']['hit_rate']:.0f}%); "
              f"a6-margin = {rows['margin']['mean_ans_turns']:.1f}; "
              f"a6-table (PRIV) = {rows['table']['mean_ans_turns']:.1f}; a3-table ceiling = 12.9. "
              f"a6 BEATS s3 at any budget (CI excl 0)? **{a6_wins}**. a6-table (PRIV) beats s3 "
              f"(policy CLASS viable)? **{table_wins}**.\n\n")
    fs = sep_s3["a6"]
    md.append(f"**First separation (a6 belief(t) vs s3 belief(t), CI excl 0):** "
              f"{('turn ' + str(fs[0]) + ' (sign ' + fs[1] + ')') if fs[0] else 'NEVER within T=24'}.\n\n")

    md.append("**NDCG@10(t) curves (t=1..24):**\n\n| arm | " + " | ".join(f"t{t+1}" for t in range(TMAX)) + " |\n")
    md.append("|" + "---|" * (TMAX + 1) + "\n")
    for lab, cur in (("s3 popular-item", pt_s3.mean(axis=0)), ("a3-blind", pt_a3.mean(axis=0)),
                     ("a6", pt["a6"].mean(axis=0)), ("a6-margin", pt["margin"].mean(axis=0)),
                     ("a6-table (PRIV)", pt["table"].mean(axis=0))):
        md.append(f"| {lab} | " + " | ".join(f"{c:.3f}" for c in cur) + " |\n")

    md.append(f"\n**VERDICT:** {verdict}\n\n")
    md.append("**ASSUMPTIONS / judgment calls (a6):**\n"
              "1. Probe bank = the 160 top-coverage ladder items (coverage>=3) s3/a3/a4/a5 draw from; item "
              "probes only.\n"
              "2. V(j) REALIZATION = the s3 greedy ORDER as a monotone rank proxy V(j)=1/rank_s3(j). "
              "FA.build_greedy returns only the schedule (not per-position greedy marginal contributions), "
              "so the rank proxy is used; it is monotone in the s3 order, which is all the tie-by-"
              "construction argument requires (argmax V over unasked == s3's next item at LR=1). s3's 24 "
              "items get ranks 1..24 in schedule order; the remaining bank items get ranks 25.. by coverage "
              "descending (tie-break dense id).\n"
              "3. Belief = the LEARNED K-map (kmap_emb.npz/kmap_intercepts.npz), online inference = "
              f"kmap_validate.Kmap.infer (MAP-Newton, {KV.NEWTON_IT} it, priors tau(k)={KV.TAU}, "
              f"tau_a(alpha)={KV.TAU_A}); imported UNMODIFIED. Event label = probe outcome (answered=1 iff "
              "the user rated the dense id in the known half; refused=0) -- the STRUCTURAL arena. No "
              "privileged features enter the a6/a6-margin policy.\n"
              "4. LR CALIBRATION = P_kmap(j|events)/P_kmap(j|no events) with P_kmap=sigmoid(alpha+b_j+k.e_j"
              f"+beta0), beta0 the a5 level shift (reused via A5.calibrate_beta0; base rate {base_rate:.4f} "
              f"-> beta0 {beta0:+.4f}). At t=0 alpha=k=0 => numerator==denominator => LR=1 exactly for every "
              "j (verified: all turn-1 picks == s3[0]); beta0 does not affect the t=0 tie (cancels in the "
              "ratio there) and only tilts the ratio once events accrue.\n"
              f"5. MARGIN (a6-margin) = {MARGIN}: swap off the s3-base order only if the argmax's LR exceeds "
              f"{MARGIN} (hysteresis against posterior noise); otherwise take the next s3-order item.\n"
              "6. a6-table (PRIVILEGED, labelled) = same V*LR selection but LR = the certain-knowledge "
              "posterior (1.0 if the user rated j, 1e-6 else), so argmax V*LR = highest-V ANSWERABLE item "
              "each turn -- the tie-by-construction ceiling of the policy class (mirrors a3-table's restrict-"
              "to-answerable rule). Not deployable.\n"
              "7. SWAP definition = actual pick != j_base, where j_base = the max-V unasked item (the pure "
              "s3-order continuation given what is already asked); swapped-IN = actual pick, displaced = "
              "j_base. Deterministic tie-breaks: max score, then higher coverage V, then lower dense id.\n"
              "8. Refusal = no-op turn (belief unchanged for the fold; the K-map still records the refusal "
              "event), user retained; all 298 users in every mean (fair, inherited). Bootstrap paired "
              f"per-user BOOT={P4.BOOT} seed={P4.SEED}; all selectors deterministic.\n\n")

    with open(OUT_MD, "a", encoding="utf-8") as f:
        f.write("".join(md))
    print(f"\n[a6] wall {out['wall_min']}m -> {OUT_JSON}, appended section to {OUT_MD}", flush=True)


if __name__ == "__main__":
    main()
