"""fair_reruns.py -- THE FAIR RE-RUNS: router-vs-static contrasts against SPLIT-CONSTRUCTED statics.

The contamination run (experiments/STATIC_CONTAMINATION.md) CONFIRMED the greedy statics are test-fit
contaminated (+0.0283 pooled at T=12, s-item). Every prior router deficit was measured against that
INFLATED baseline. This script re-prices the three key routers against the FAIR (split-constructed)
static -- the routers AND the static are both CONSTRUCTED on the construction half only, evaluated on
the eval half only. The routers' value model V(q|z) is fitted on POPULATION trU sims (firewall-clean,
reused as-is from repair_probes.fit_value_model). NO LLM API calls; all local compute.

DESIGN (mirrors the contamination run): seeds {0,1} x eval-half {A,B} = 4 split estimates.
  construction half = the OTHER half; eval half = evaluated-on half.
  s-fair          = split-built greedy static on the construction half (s-item family = strongest fair
                    static per the contamination run; s-mixed-fair also reported).
  r-value-blind   = P3 V(q|z) tilt on s-fair's schedule (warm-start rebuilt on the CONSTRUCTION half).
  r-value+k       = + LOUO/kmap answerability (knowledge) tilt (P3, use_k=True).
  r-blind-a6      = a6-style pure LR-tilt on s-fair's backbone (base_v * kmap LR; no value model).
  All routers tie-by-construction at t=0 == s-fair[0] (E2 floor). Evaluated on eval half only, T=8/12/24.

PRE-REGISTERED READS (printed BEFORE results): for each router, pooled (router - s-fair) with paired
per-user bootstrap over the 4 estimates' concatenated per-user deltas:
  (i)  CI excl 0 POSITIVE  = THE FIRST FAIR ADAPTIVITY WIN OF THE PROGRAM.
  (ii) CI incl 0           = honest tie on a fair baseline (still an upgrade from 'loses').
  (iii)CI excl 0 NEGATIVE  = routers genuinely lose even fairly (reported without softening).
Sanity link: s-fair vs the OLD cohort-fit static (greedy built on the eval-half itself) on the same
eval users -- should reproduce ~-0.028 (= -contamination), linking the two experiments.

E-rules E1 (same-user paired bootstrap) / E5 (population-firewalled beliefs) / E6 (pre-registered
thresholds printed before results) honored. DIRECTIONAL 173/300.

Run:  python scripts/fair_reruns.py --do all
"""
import os, sys, json, time, argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import warnings
warnings.filterwarnings("ignore")
import repair_probes as RP
import static_contamination as SC
import battery_stage_b as B

OUT_MD = "experiments/FAIR_RERUNS.md"
OUT_JSON = "experiments/fair_reruns.json"
BUDGETS = [8, 12, 24]
TMAX = max(BUDGETS)
BOOT = 5000
PRIMARY_T = 12

RESULTS = {"banner": "DIRECTIONAL 173/300, grid unfrozen (v2 fold answerer-v1 working grid)"}


def md(txt, mode="a"):
    os.makedirs("experiments", exist_ok=True)
    open(OUT_MD, mode, encoding="utf-8").write(txt)


# ------------------------------------------------------------ router (warm-started on s-fair)
def make_router(env, gbm, KM, s_fair_sched, mode):
    """mode in {blind, k, a6}. Warm-start ranking = s_fair_sched (built on the CONSTRUCTION half).
    blind = V(q|z)/V(q|cold) tilt only; k = + kmap item-LR; a6 = pure kmap LR tilt on base_v (no
    value model). All tie-by-construction at t=0 (LR=1, vratio=1 -> argmax base_v == s_fair[0])."""
    FR, model, CANDS = env["FR"], env["model"], env["CANDS"]
    allc = [m["cid"] for m in CANDS]
    rank = {c: r + 1 for r, c in enumerate(s_fair_sched)}
    rest = sorted([m["cid"] for m in CANDS if m["cid"] not in rank], key=lambda c: -CANDS[c]["cov"])
    for r, c in enumerate(rest):
        rank[c] = len(s_fair_sched) + 1 + r
    base_v = {c: 1.0 / rank[c] for c in rank}
    z_cold = np.zeros(FR.W.shape[1])

    def feats(rec, cd, z0, t):
        m = CANDS[cd]; emb = m["emb"]; v = rec["val_arr"][cd]
        taste = float(z0 @ emb / (np.linalg.norm(z0) * np.linalg.norm(emb) + 1e-9)) if np.linalg.norm(z0) > 0 else 0.0
        ch = [0, 0, 0]; ch[m["typ"]] = 1
        fo = [0, 0, 0]; fo[int(rec["fid_arr"][cd])] = 1
        return ch + fo + [abs(v), taste, t]

    use_value = mode in ("blind", "k")
    use_k = mode in ("k", "a6")

    def plan(rec):
        used = set(); pl = []; ev = []; j_ev, y_ev = [], []
        if use_value:
            Vp0 = gbm.predict(np.array([feats(rec, c, z_cold, 0) for c in allc]))
            Vprior = {c: Vp0[i] for i, c in enumerate(allc)}
        for t in range(TMAX):
            z = RP.V2.fold_np_v2(FR, model, [RP.tok_of(CANDS, rec, c) for c in ev],
                                 [rec["nat_arr"][c] for c in ev if rec["nat_arr"][c] is not None]) if ev else z_cold
            if use_k and j_ev:
                al, k = B.infer_user(KM.Efull[j_ev], KM.bfull[j_ev], np.array(y_ev, float), KM.d)
            else:
                al, k = 0.0, np.zeros(KM.d)
            cand = [c for c in allc if c not in used]
            if use_value:
                Vp = gbm.predict(np.array([feats(rec, c, z, len(ev)) for c in cand]))
            best, bc = -1e18, None
            for i, c in enumerate(cand):
                if use_value:
                    vratio = Vp[i] / Vprior[c] if abs(Vprior[c]) > 1e-6 else 1.0
                    vratio = max(vratio, 0.0)
                else:
                    vratio = 1.0
                lr = KM.lr(CANDS[c]["key"], al, k) if (use_k and CANDS[c]["kind"] == "item") else 1.0
                score = base_v[c] * vratio * lr
                if score > best or (score == best and (bc is None or rank[c] < rank[bc])):
                    best, bc = score, c
            if bc is None:
                break
            used.add(bc)
            pl.append(bc if rec["ans_arr"][bc] else None)
            if rec["ans_arr"][bc]:
                ev.append(bc)
            if CANDS[bc]["kind"] == "item":
                j_ev.append(CANDS[bc]["key"]); y_ev.append(1 if rec["ans_arr"][bc] else 0)
        return pl
    return plan


def eval_plans_peruser(env, plans, sub_users, sub_cold, T):
    """Per-user anytime NDCG@10 over the first T turns of per-user router plans (entries = cid-if-
    answered-or-None). Identical accumulation to SC.eval_static_peruser but with per-user plans."""
    FR, model, CANDS = env["FR"], env["model"], env["CANDS"]
    n = len(sub_users); per_turn = np.empty((n, T))
    for t in range(T):
        col = sub_cold.copy(); tl, nl, idx = [], [], []
        for i, rec in enumerate(sub_users):
            cids = [c for c in plans[i][:t + 1] if c is not None]
            if not cids:
                continue
            tl.append([RP.tok_of(CANDS, rec, c) for c in cids])
            nl.append([rec["nat_arr"][c] for c in cids if rec["nat_arr"][c] is not None])
            idx.append(i)
        if tl:
            col[idx] = RP.batch_ndcg(FR, model, sub_users, tl, nl, idx)
        per_turn[:, t] = col
    return per_turn


PREREG = (
    "## Pre-registered reads (printed BEFORE results)\n\n"
    "For each router, pooled (router - s-fair) with paired per-user bootstrap over the 4 split\n"
    "estimates' concatenated per-user deltas (seeds {0,1} x eval-half {A,B}). s-fair = split-built\n"
    "greedy s-item on the CONSTRUCTION half, evaluated on the EVAL half. Routers warm-start on\n"
    "s-fair's construction-half schedule; value model V(q|z) fitted on population trU (firewall).\n\n"
    "- **(i) CI excl 0 POSITIVE** = THE FIRST FAIR ADAPTIVITY WIN OF THE PROGRAM.\n"
    "- **(ii) CI incl 0** = honest tie on a fair baseline (still a massive upgrade from 'loses').\n"
    "- **(iii) CI excl 0 NEGATIVE** = routers genuinely lose even fairly (reported without softening).\n\n"
    "Verdict uses the POOLED estimate at T=12. Sanity link: s-fair vs the OLD cohort-fit static\n"
    "(greedy built on the eval-half itself) on the same eval users -> should reproduce ~-0.028\n"
    "(= -contamination), linking to STATIC_CONTAMINATION.md. NO LLM calls; DIRECTIONAL 173/300.\n\n")


def run():
    t0 = time.time()
    env = RP.setup()
    users = env["users"]; n = len(users); CANDS = env["CANDS"]
    item_pool = [m["cid"] for m in CANDS if m["kind"] == "item"]
    mixed_pool = [m["cid"] for m in CANDS]

    md("# THE FAIR RE-RUNS -- routers vs SPLIT-CONSTRUCTED statics (v2 fold; DIRECTIONAL 173/300)\n\n", mode="w")
    md("> **DIRECTIONAL ONLY** -- 173/300 users, answerer-v1 working grid NOT frozen; v2 fold\n"
       "> `.cache/i25_fold_v2_best.pt`. Re-run on the frozen 300-user grid before any citation.\n\n"
       f"Date 2026-07-09. Script `scripts/fair_reruns.py`. NO LLM API calls; local compute.\n"
       f"{n} users; {len(CANDS)} candidates; cold NDCG@10 {env['cold'].mean():.4f}; fold val "
       f"{env['foldval']:.4f}. Paired per-user bootstrap BOOT={BOOT} seed=0. Greedy prefix-consistent,\n"
       f"one T={TMAX} build serves budgets T={BUDGETS}.\n\n")
    md(PREREG)
    print("\n" + PREREG, flush=True)

    # ---- population-firewalled beliefs (fit ONCE, reused across all 4 estimates) ----
    print("==== fitting V(q|z) on population trU simulations (firewall) ... ====", flush=True)
    gbm, n_fit = RP.fit_value_model(env)
    print(f"  value model fitted on {n_fit} population (candidate,evidence) samples.", flush=True)
    KM = RP._kmap(env["D"])
    print("  kmap (population item answerability posterior) loaded.", flush=True)

    ROUTERS = ["r-value-blind", "r-value+k", "r-blind-a6"]
    MODE = {"r-value-blind": "blind", "r-value+k": "k", "r-blind-a6": "a6"}

    # pooled per-user delta accumulators (concatenated across the 4 estimates), per T
    pooled = {rt: {T: {"r": [], "s": []} for T in BUDGETS} for rt in ROUTERS}
    pooled_mixed = {T: {"r": [], "s": []} for T in BUDGETS}   # s-mixed-fair vs s-item-fair reference
    pooled_sanity = {T: {"fair": [], "cohort": []} for T in BUDGETS}
    est = {}   # per-estimate summaries

    for seed in (0, 1):
        rng = np.random.default_rng(seed)
        perm = rng.permutation(n); half = n // 2
        A_idx = sorted(perm[:half].tolist()); B_idx = sorted(perm[half:].tolist())
        halves = {"A": A_idx, "B": B_idx}
        for evalhalf in ("B", "A"):
            constr = "A" if evalhalf == "B" else "B"
            ev_users = [users[i] for i in halves[evalhalf]]
            cn_users = [users[i] for i in halves[constr]]
            ev_cold = SC.cold_of(env, ev_users)
            cn_cold = SC.cold_of(env, cn_users)
            tag = f"seed{seed}_eval{evalhalf}"
            print(f"\n---- estimate {tag}: construct on {constr} (n={len(cn_users)}), "
                  f"eval on {evalhalf} (n={len(ev_users)}) ----", flush=True)

            # s-fair (s-item) + s-mixed-fair: built on construction half
            s_item_fair = SC.build_greedy_sub(env, item_pool, cn_users, cn_cold, TMAX)
            s_mixed_fair = SC.build_greedy_sub(env, mixed_pool, cn_users, cn_cold, TMAX)
            # OLD cohort-fit static (in-sample): built on the eval half itself (sanity link)
            s_item_cohort = SC.build_greedy_sub(env, item_pool, ev_users, ev_cold, TMAX)

            pt_sfair = SC.eval_static_peruser(env, s_item_fair, ev_users, ev_cold, TMAX)
            pt_smix = SC.eval_static_peruser(env, s_mixed_fair, ev_users, ev_cold, TMAX)
            pt_cohort = SC.eval_static_peruser(env, s_item_cohort, ev_users, ev_cold, TMAX)
            print(f"  s-fair(item) T12={pt_sfair[:, :12].mean():.4f}  s-mixed-fair T12={pt_smix[:, :12].mean():.4f}"
                  f"  old-cohort-fit T12={pt_cohort[:, :12].mean():.4f}", flush=True)

            # routers, warm-started on s-fair's construction-half schedule
            router_pt = {}
            for rt in ROUTERS:
                plan_fn = make_router(env, gbm, KM, s_item_fair, MODE[rt])
                plans = [plan_fn(rec) for rec in ev_users]
                # tie-by-construction check on a sample
                tie_ok = all((plans[i][0] == s_item_fair[0]) for i in range(0, len(ev_users), 15))
                pt = eval_plans_peruser(env, plans, ev_users, ev_cold, TMAX)
                router_pt[rt] = pt
                d12 = RP.paired(list(pt[:, :12].mean(axis=1)), list(pt_sfair[:, :12].mean(axis=1)))
                print(f"  {rt:14s} T12={pt[:, :12].mean():.4f}  (vs s-fair {d12['delta']:+.4f}"
                      f"[{d12['ci'][0]:+.4f},{d12['ci'][1]:+.4f}])  tie@t0={tie_ok}", flush=True)

            # accumulate pooled deltas
            e = {"n_eval": len(ev_users)}
            for T in BUDGETS:
                sfa = list(pt_sfair[:, :T].mean(axis=1))
                for rt in ROUTERS:
                    rvec = list(router_pt[rt][:, :T].mean(axis=1))
                    pooled[rt][T]["r"].extend(rvec); pooled[rt][T]["s"].extend(sfa)
                    e.setdefault(rt, {})[str(T)] = RP.paired(rvec, sfa)
                pooled_mixed[T]["r"].extend(list(pt_smix[:, :T].mean(axis=1))); pooled_mixed[T]["s"].extend(sfa)
                pooled_sanity[T]["fair"].extend(sfa)
                pooled_sanity[T]["cohort"].extend(list(pt_cohort[:, :T].mean(axis=1)))
                e.setdefault("s-mixed-fair", {})[str(T)] = RP.paired(
                    list(pt_smix[:, :T].mean(axis=1)), sfa)
                e.setdefault("sanity_fair_minus_cohort", {})[str(T)] = RP.paired(
                    sfa, list(pt_cohort[:, :T].mean(axis=1)))
            est[tag] = e

    # ---- pooled CIs ----
    pooled_ci = {rt: {T: RP.paired(pooled[rt][T]["r"], pooled[rt][T]["s"]) for T in BUDGETS} for rt in ROUTERS}
    mixed_ci = {T: RP.paired(pooled_mixed[T]["r"], pooled_mixed[T]["s"]) for T in BUDGETS}
    sanity_ci = {T: RP.paired(pooled_sanity[T]["fair"], pooled_sanity[T]["cohort"]) for T in BUDGETS}

    # ---- verdicts (primary T=12) ----
    def verdict(cb):
        lo, hi = cb["ci"]
        if lo > 0:
            return "FAIR-WIN (CI excl 0, positive) -- THE FIRST FAIR ADAPTIVITY WIN"
        if hi < 0:
            return "FAIR-LOSS (CI excl 0, negative) -- routers lose even fairly"
        return "TIE (CI incl 0) -- honest tie on a fair baseline"

    RESULTS.update(dict(
        n_value_fit=n_fit,
        pooled={rt: {str(T): pooled_ci[rt][T] for T in BUDGETS} for rt in ROUTERS},
        s_mixed_fair_vs_s_item_fair={str(T): mixed_ci[T] for T in BUDGETS},
        sanity_fair_minus_cohort={str(T): sanity_ci[T] for T in BUDGETS},
        verdicts={rt: verdict(pooled_ci[rt][PRIMARY_T]) for rt in ROUTERS},
        estimates=est,
        wall_min=round((time.time() - t0) / 60, 2)))

    _write_md(pooled_ci, mixed_ci, sanity_ci, est, ROUTERS)
    json.dump(RESULTS, open(OUT_JSON, "w"), indent=1, default=str)

    print("\n==== POOLED VERDICTS (T=12) ====", flush=True)
    for rt in ROUTERS:
        cb = pooled_ci[rt][PRIMARY_T]
        print(f"  {rt:14s} {cb['delta']:+.4f}[{cb['ci'][0]:+.4f},{cb['ci'][1]:+.4f}] "
              f"(n={cb['n']}) -> {RESULTS['verdicts'][rt]}", flush=True)
    sc = sanity_ci[PRIMARY_T]
    print(f"  sanity: s-fair - old-cohort-fit = {sc['delta']:+.4f}[{sc['ci'][0]:+.4f},{sc['ci'][1]:+.4f}] "
          f"(expect ~-0.028 = -contamination)", flush=True)
    print(f"\n[done] wrote {OUT_MD} + {OUT_JSON} (wall {RESULTS['wall_min']}m)", flush=True)
    return RESULTS


def _write_md(pooled_ci, mixed_ci, sanity_ci, est, ROUTERS):
    md("## Pooled router-vs-s-fair contrasts (4 estimates concatenated)\n\n"
       "| router | budget | pooled (router - s-fair) [95% CI] | n | verdict |\n|---|---|---|--:|---|\n")
    for rt in ROUTERS:
        for T in BUDGETS:
            cb = pooled_ci[rt][T]
            lo, hi = cb["ci"]
            v = "WIN" if lo > 0 else ("LOSS" if hi < 0 else "tie")
            md(f"| {rt} | T={T} | {cb['delta']:+.4f}[{lo:+.4f},{hi:+.4f}] | {cb['n']} | {v} |\n")
    md("\n**Primary verdicts (T=12, pooled):**\n\n")
    for rt in ROUTERS:
        cb = pooled_ci[rt][PRIMARY_T]
        md(f"- **{rt}:** {cb['delta']:+.4f}[{cb['ci'][0]:+.4f},{cb['ci'][1]:+.4f}] -> "
           f"{RESULTS['verdicts'][rt]}\n")

    md("\n## Reference: s-mixed-fair vs s-item-fair (which fair static is stronger)\n\n"
       "| budget | s-mixed-fair - s-item-fair [95% CI] | n |\n|---|---|--:|\n")
    for T in BUDGETS:
        cb = mixed_ci[T]
        md(f"| T={T} | {cb['delta']:+.4f}[{cb['ci'][0]:+.4f},{cb['ci'][1]:+.4f}] | {cb['n']} |\n")

    md("\n## Sanity link to STATIC_CONTAMINATION.md\n\n"
       "s-fair (built on construction half) minus OLD cohort-fit static (built on the eval half itself),\n"
       "both evaluated on the SAME eval users. This equals -(contamination); expect ~-0.028 at T=12.\n\n"
       "| budget | s-fair - old-cohort-fit [95% CI] | n |\n|---|---|--:|\n")
    for T in BUDGETS:
        cb = sanity_ci[T]
        md(f"| T={T} | {cb['delta']:+.4f}[{cb['ci'][0]:+.4f},{cb['ci'][1]:+.4f}] | {cb['n']} |\n")

    md("\n## Per-estimate detail (T=12)\n\n"
       "| estimate | n_eval | r-value-blind | r-value+k | r-blind-a6 | s-mixed-fair | sanity |\n"
       "|---|--:|---|---|---|---|---|\n")
    for tag, e in est.items():
        def fc(key):
            cb = e[key]["12"]
            return f"{cb['delta']:+.4f}[{cb['ci'][0]:+.4f},{cb['ci'][1]:+.4f}]"
        md(f"| {tag} | {e['n_eval']} | {fc('r-value-blind')} | {fc('r-value+k')} | {fc('r-blind-a6')} | "
           f"{fc('s-mixed-fair')} | {fc('sanity_fair_minus_cohort')} |\n")
    md("\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--do", choices=["all"], default="all")
    ap.parse_args()
    run()
