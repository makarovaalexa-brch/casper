"""arena_eval.py -- THE POLICY ARENA driver v2 (gated world; per DESIGN_SHEET_POLICY_ARENA.md,
rebuilt per ARENA_CODE_AUDIT.md + BLIND_VALIDATION.md F14, 2026-07-10).

Stages:
  build : cohorts + gated answer-table generation + WORLD FUEL CHECK (gate stamp) + b1/b2/b3-tail/
          Golbandi/scorer-A construction on TRAIN.
  eval  : all arms on DEV-TEST; DEV-VAL selects b2-anchored TAU; curves T=1..24, budgets 8/16/24,
          endpoint@50 primary + endpoint@10 + anytime@10; paired bootstrap + MDE; mechanism
          readouts; transcripts; pre-registered verdicts printed BEFORE results.

DEV synthetic users ONLY; the 173/300 are NEVER touched. NO LLM calls. ASCII. Deterministic.

Run: python scripts/arena_eval.py build
     python scripts/arena_eval.py eval
"""
import os, sys, json, time, argparse
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE); sys.path.insert(0, os.path.join(_HERE, "instrument2"))
import warnings
warnings.filterwarnings("ignore")

import arena_core as AC
import arena_policies as AP
from arena_core import paired_ci, world_fuel_check
from arena_policies import (StaticSeq, B0Cold, B4Myopic, TrueTableRouter, Clairvoyant, AskGradient,
                            CATRouter, GolbandiTree, ScorerA, B2Anchored, run_policy,
                            TYPE_ITEM, TYPE_CONCEPT, TYPE_ENTITY)

MD = "experiments/ARENA_BUILD.md"
RES = f"{AC.CACHE_DIR}/results.json"
Tmax = 24
KS = (50, 10)
BUDGETS = (8, 16, 24)
TAU_GRID = [0, 8, 24]


def md(txt, mode="a"):
    open(MD, mode, encoding="utf-8").write(txt)


def get_world(args):
    ar = AC.Arena()
    coh = AC.make_cohorts(ar, n_train=args.n_train, n_devval=args.n_devval,
                          n_devtest=args.n_devtest)
    cfg = coh["cfg"]
    ar.prefill_answers(coh["train"], "train", cfg)
    ar.prefill_answers(coh["devval"], "devval", cfg)
    ar.prefill_answers(coh["devtest"], "devtest", cfg)
    # shared blind population prior (TRAIN answer rates; E7-symmetric across all blind arms)
    ar.set_pop_prior(AP.train_pop_prior(ar, coh["train"]))
    return ar, coh


# ================================================================= BUILD stage
def stage_build(args):
    t0 = time.time()
    ar, coh = get_world(args)
    print(f"[build] cohorts train={len(coh['train'])} devval={len(coh['devval'])} "
          f"devtest={len(coh['devtest'])}", flush=True)
    # WORLD FUEL CHECK (gate stamp; fix #7) -- on TRAIN synthetic users, >=400
    world_fuel_check(ar, coh["train"][:max(400, min(600, len(coh["train"])))])
    b2_users = coh["train"][:args.n_b2]
    print(f"[build] b2 greedy static on {len(b2_users)} TRAIN users (fix #3; prescreen top "
          f"{args.b2_prescreen}) ...", flush=True)
    b2_seq, _ = AP.build_b2(ar, b2_users, Tmax=Tmax, K=AP.PRIMARY_K,
                            prescreen_top=args.b2_prescreen)
    AP.b3_tail(ar, b2_users, b2_seq)                       # materializes the prescreen cache
    print("[build] b1 concept-entropy static ...", flush=True)
    AP.build_b1(ar, coh["train"])
    print("[build] Golbandi tree ...", flush=True)
    AP.build_golbandi(ar, coh["train"][:args.n_gol], max_depth=5, min_users=25)
    print("[build] scorer-A GBM ...", flush=True)
    AP.build_scorerA(ar, coh["train"][:args.n_scorer], n_samples=args.n_labels)
    print(f"[build] DONE [{(time.time()-t0)/60:.1f} min]", flush=True)


# ================================================================= report header (E5/E7 before results)
def write_header(ar, coh, args, fuel):
    md("# THE POLICY ARENA -- BUILD + DEV RESULTS (v2, GATED WORLD)\n\n", mode="w")
    md("Per DESIGN_SHEET_POLICY_ARENA.md (signed 2026-07-09), rebuilt after the pre-verdict audits "
       "(experiments/ARENA_CODE_AUDIT.md; experiments/BLIND_VALIDATION.md F14). DEV evaluation on "
       "SYNTHETIC users ONLY; the 173/300 real-judged users are NEVER touched. NO LLM calls. "
       f"Deterministic (seed {AC.SEED}).\n\n")
    md("## FOLD PICK (STEP 0)\n\n")
    md("**Deep-Sets fold-v3** (`.cache/i25_fold_v3_best.pt`, clean_frac=0.30, val NDCG@10 "
       f"{ar.fold_state.get('best_val'):.4f}) is the arena belief. The Set-Transformer contingency "
       "FINISHED: best val 0.4123 (ep6), decisively below Deep-Sets' 0.4311, and re-gated WORSE -- "
       "FAIL on G2/G3/G4/G5/G6 (only G1/G2b/G7 pass), including the decisive G2 GoT gate that DS "
       "PASSES, and even G6 (its own remedy target) worse than DS (-0.0119 vs -0.0035). The contract "
       "required strict dominance (all gates + val + G4) to displace; it fails on every count. "
       "DS fold-v3 posture: decisive two-channel gates (G2 GoT +0.184, G2b prolific +0.316, G7 "
       "implicit-ablation +0.020) PASS; G5 (-0.028 vs v2 on v2's item-heavy regime) and G6 (-0.0035 "
       "one-step, marginal) are diagnosed failures, not the two-channel purpose. `ALL GATES PASS: "
       "False` is the honest record; we do NOT fall back to fold-v2.\n\n")
    md("## THE WORLD (fix #7 -- certification chain repaired)\n\n")
    md("The arena answerer is THE FITTED, GATED v2.1 MODEL, loaded and run (NOT the hand-"
       "parameterized sampler the blind validation flagged as credential transfer):\n\n")
    md(f"- knowledge: per-channel ordinal logistics from `.cache/dans/models_v21.json` (sha "
       f"{ar.models_sha}) -- full feature set (co-knowledge, fame, census/buffness, era), EQUATED "
       "flutter-free trait sigma (concept 0.199 / entity 0.301 / item 0.621), entity per-cut random "
       "effects; generation recipe = dans_stages.know_probs verbatim (per-user seed 123*7777+uid).\n")
    md(f"- values: v2.1 EASE-backbone value models (t(u,i) = real rating if rated else EASE "
       f"prediction over the {len(ar.ease_uni)}-item universe); rated bank items pass through "
       "(know_well + real centered rating, fid=data).\n")
    md(f"- universe (fix #5): ALL judged questions = {ar.ntag} concepts + {ar.nent} attributes "
       f"(IMDb entities) + {ar.nbank} bank items = {ar.nQ} -- the signed sheet's 2,428. No pools, "
       "no coverage sampling, no alphabetical ties.\n")
    md("- belief: the picked Deep-Sets fold-v3, FIXED. Documented deviation: the fold was trained "
       "on the earlier sampler world and is deployed unchanged on the gated world (same token "
       "vocabulary; mild distribution shift; retraining the fold on the gated world is future "
       "work).\n")
    md("- LENIENT regime: know>=1 answered; no_clue = refusal = consumed turn, belief unchanged "
       "(refusal tokens NEVER fold -- the fold-v3 no-clue-not-negative caveat), user kept (E1).\n\n")
    md("### WORLD FUEL CHECK (the world's own gate stamp, not inherited credentials)\n\n")
    md(f"Trait ICC per channel on the ARENA WORLD AS BUILT ({fuel['n_users']} synthetic users, "
       "question-residualized one-way user share; flutter=0 by construction) vs the CORRECTED "
       "targets (equate_v21.json icc_trait_corrected), tolerance 0.02 (the v2.1 G2 rule):\n\n")
    md("| channel | margin | arena trait ICC | corrected target | verdict | base rate |\n"
       "|---|---|--:|--:|:--:|--:|\n")
    for ch in ("concept", "entity", "item"):
        r = fuel[ch]
        md(f"| {ch} | k>={r['margin']} | {r['arena_trait_icc']:.4f} | {r['corrected_target']:.4f} | "
           f"{'MATCH' if r['match'] else 'MISS'} | {r['base_rate']:.3f} |\n")
    md(f"\n**ALL MATCH: {fuel['all_match']}**\n\n")
    md("## AUDIT FIXES APPLIED (methods; credit: the blind pre-verdict code audit + program "
       "validation)\n\n")
    md("1. b4 target-peek FIXED: objective is now the blind smooth ranking utility J(z)=tau*"
       "logsumexp(decode(z)/tau); expected gain = p_ans_blind * (J(z')-J(z)); never touches held-out "
       "targets.\n"
       "2. b2 partial-cache FIXED: resume-from-partial + assert len(seq)==Tmax (the 10/24 truncated "
       "cache was discarded with the old world's caches).\n"
       "3. b2 construction cohort raised to "
       f"{args.n_b2} TRAIN users (was 50); candidate PRE-SCREEN = top-{args.b2_prescreen} by "
       "1-question cohort gain over ALL 2,428 (documented compute deviation; seed-repeat robustness "
       "in the addenda).\n"
       "4. answer caches keyed by cohort-config sha + per-user known-set hashes verified on load; "
       "ALL old-world caches invalidated (archived to .cache/arena_old_prefix4/).\n"
       "5. universe rebuilt to the signed sheet (2,428 judged questions; was 1,530 sampler vocab).\n"
       "6. b3 gets the pre-registered value-ranked tail (prescreen-gain order) when its list "
       "exhausts under refunds.\n"
       "7. THE WORLD runs the fitted gated v2.1 models (see above) + carries its own fuel stamp.\n"
       "8. one shared cand_M="
       f"{args.cand_M} for b4/A/B/C; all blind arms share ONE population prior (per-question TRAIN "
       "answer rate); policies receive only the observed dialogue, never the answerer's internals.\n\n")
    md("## DEPLOYABILITY (blind agents)\n\n")
    md("b4/A/B/C choose questions from the current belief z + the shared TRAIN population prior "
       "only; value forecasts decode z over region members; the true answer is observed only AFTER "
       "asking. D branches on OBSERVED answer polarity of already-asked questions. Only the "
       "labelled context arms (true-table, clairvoyant) peek at realized answers -- never cited.\n\n")
    md("## E7 SYMMETRY TABLE\n\n")
    md("All arms share: gated v2.1 world, fold-v3 belief, full 2,428 universe, lenient regime, SAME "
       "DEV-TEST users (E1), SAME shared population prior + cand_M for blind arms. Differs ONLY by "
       "selection policy + construction cohort:\n\n")
    md("| arm | family | learns from | construction cohort | reachable set/turn | privileged? |\n")
    md("|---|---|---|---|---|---|\n")
    rows = [
        ("b0 cold", "orientation", "nothing", "-", "-", "no"),
        ("b1 concept-entropy", "static", "TRAIN answer-rates", f"{args.n_train} TRAIN", "1128 concepts", "no"),
        ("b2 learned static", "static", "TRAIN NDCG greedy", f"{args.n_b2} TRAIN", f"prescreen top-{args.b2_prescreen}", "no"),
        ("b3 = b2+skip+tail", "static+skip", "= b2", "= b2", "= b2 + gain-ranked tail", "no"),
        ("b4 myopic (J)", "model-based blind", "NONE", "-", f"cand_M={args.cand_M}", "no"),
        ("A scorer-v2", "adaptive learned", "TRAIN 1+2-step gains", f"{args.n_scorer} TRAIN", f"cand_M={args.cand_M}", "no"),
        ("B ask-gradient", "adaptive", "calibration only", "-", f"cand_M={args.cand_M}", "no"),
        ("C CAT-router", "adaptive", "NONE", "-", f"cand_M={args.cand_M}", "no"),
        ("D Golbandi tree", "adaptive learned", "TRAIN split NDCG", f"{args.n_gol} TRAIN", "tree + b2 + tail", "no"),
        ("ttab router", "CONTEXT labelled", "peeks realized answers", "-", f"cand_M={args.cand_M}", "YES -- never cited"),
        ("clairvoyant", "CONTEXT labelled approx", "peeks realized answers", "-", "M=200 answered", "YES -- never cited"),
    ]
    for r in rows:
        md("| " + " | ".join(r) + " |\n")
    md("\n## PRE-REGISTERED DEV VERDICTS (printed BEFORE results; E5)\n\n")
    md("Primary: endpoint NDCG@50 at T=24; also endpoint@10, anytime@10, budgets 8/16/24; paired "
       "bootstrap CI + MDE on every delta (E4).\n\n")
    md("- **b4 vs b2** (computation-suffices): b4 WIN (CI excl 0) = adaptivity pays via computation; "
       "learned policies must then beat b4. TIE = computation alone does not pay here.\n")
    md("- **each class (A,B,C,D) vs b2 AND vs b4**: WIN = CI excl 0 above b2 AND >= b4. If only b4 "
       "wins: 'adaptivity pays via computation; learning adds nothing yet' (stated exactly).\n")
    md("- TAU on DEV-VAL (TAU=24 == b2 exactly); pure TAU=0 reported alongside. No post-hoc metric "
       "selection. DEV ONLY -- no headline claims; the 173/300 are out of bounds.\n\n")


def _print_e7(ar):
    print("\n" + "=" * 72, flush=True)
    print("E7 SYMMETRY: all arms share the GATED v2.1 world (fitted models, sha "
          f"{ar.models_sha}), fold-v3 belief, the full {ar.nQ}-question universe, lenient regime,",
          flush=True)
    print("SAME DEV-TEST users (E1), one shared population prior + cand_M for all blind arms.",
          flush=True)
    print("Differs ONLY by selection policy + its construction cohort (see ARENA_BUILD.md table).",
          flush=True)
    print("=" * 72, flush=True)


# ================================================================= helpers
def endpoint(curves, K, t):
    return float(np.nanmean(curves[K][:, t]))


def anytime(curves, K):
    return float(np.nanmean(np.nanmean(curves[K][:, 1:], axis=1)))


def run_arm(ar, recs, policy, skip=False, tag=""):
    t0 = time.time()
    out = run_policy(ar, recs, policy, Tmax, Ks=KS, skip=skip, verbose=False, tag=tag)
    print(f"    [{tag}] done [{time.time()-t0:.0f}s] endpoint@50={endpoint(out['curves'],50,Tmax):.4f} "
          f"@10={endpoint(out['curves'],10,Tmax):.4f}", flush=True)
    return out


def per_user_endpoint(out, K, t=Tmax):
    return out["curves"][K][:, t]


def _fmt_ci(c):
    return f"{c['mean']:+.4f} [{c['lo']:+.4f},{c['hi']:+.4f}] MDE {c['mde']:.4f} (n={c['n']})"


# ================================================================= EVAL stage
def stage_eval(args):
    ar, coh = get_world(args)
    dv, dt = coh["devval"], coh["devtest"]
    fuel = world_fuel_check(ar, coh["train"][:max(400, min(600, len(coh["train"])))])

    b2_users = coh["train"][:args.n_b2]
    b2_seq, b2_gain = AP.build_b2(ar, b2_users, Tmax=Tmax, K=AP.PRIMARY_K,
                                  prescreen_top=args.b2_prescreen, verbose=False)
    assert len(b2_seq) >= Tmax, "b2 must be complete before eval (fix #2)"
    tail = AP.b3_tail(ar, b2_users, b2_seq)
    b1_seq = AP.build_b1(ar, coh["train"], verbose=False)
    gol_tree = AP.build_golbandi(ar, coh["train"][:args.n_gol], max_depth=5, min_users=25,
                                 verbose=False)
    gbm = AP.build_scorerA(ar, coh["train"][:args.n_scorer], n_samples=args.n_labels, verbose=False)

    write_header(ar, coh, args, fuel)
    _print_e7(ar)

    R = {}
    print("\n=== DEV-TEST baselines ===", flush=True)
    R["b0"] = run_arm(ar, dt, B0Cold(), tag="b0")
    R["b1"] = run_arm(ar, dt, StaticSeq("b1", b1_seq), tag="b1")
    R["b2"] = run_arm(ar, dt, StaticSeq("b2", b2_seq), tag="b2")
    R["b3"] = run_arm(ar, dt, StaticSeq("b3", b2_seq, tail=tail), skip=True, tag="b3")
    R["b4"] = run_arm(ar, dt, B4Myopic(M=args.cand_M), tag="b4")

    classes = {
        "A": lambda: ScorerA(gbm, M=args.cand_M, Tmax=Tmax),
        "B": lambda: AskGradient(M=args.cand_M),
        "C": lambda: CATRouter(M=args.cand_M),
        "D": lambda: GolbandiTree(gol_tree, b2_seq, tail),
    }
    sel = {}
    print("\n=== DEV-VAL model selection (b2-anchored TAU) ===", flush=True)
    b2_val = endpoint(run_policy(ar, dv, StaticSeq("b2", b2_seq), Tmax, Ks=KS)["curves"], 50, Tmax)
    print(f"    [b2] DEV-VAL endpoint@50={b2_val:.4f} (== TAU={Tmax} for every class)", flush=True)
    for cls, mk in classes.items():
        best_tau, best_v = Tmax, b2_val
        for tau in [t for t in TAU_GRID if t < Tmax]:
            o = run_policy(ar, dv, B2Anchored(f"{cls}_tau{tau}", b2_seq, mk(), tau), Tmax, Ks=KS)
            v = endpoint(o["curves"], 50, Tmax)
            print(f"    [{cls}] DEV-VAL TAU={tau:2d}  endpoint@50={v:.4f}", flush=True)
            if v > best_v:
                best_v = v; best_tau = tau
        sel[cls] = best_tau
        print(f"  [{cls}] selected TAU={best_tau} (DEV-VAL endpoint@50={best_v:.4f})", flush=True)

    print("\n=== DEV-TEST adaptive arms (selected TAU + pure TAU=0) ===", flush=True)
    for cls, mk in classes.items():
        R[cls] = run_arm(ar, dt, B2Anchored(f"{cls}_sel", b2_seq, mk(), sel[cls]),
                         tag=f"{cls}_sel(tau{sel[cls]})")
        R[cls + "_pure"] = run_arm(ar, dt, B2Anchored(f"{cls}_pure", b2_seq, mk(), 0),
                                   tag=f"{cls}_pure")

    print("\n=== CONTEXT arms (labelled, never cited) ===", flush=True)
    ctx_n = min(args.ctx_n, len(dt))
    R["ttab"] = run_arm(ar, dt[:ctx_n], TrueTableRouter(M=args.cand_M), tag=f"ttab(n={ctx_n})")
    R["clair"] = run_arm(ar, dt[:ctx_n], Clairvoyant(), tag=f"clair(n={ctx_n})")

    _write_results(ar, dt, R, sel, b2_gain, ctx_n)
    _mechanism(ar, dt, R)
    _transcripts(ar, dt, R)
    _verdicts(R)
    dump = {a: {f"ndcg{K}": [float(np.nanmean(R[a]["curves"][K][:, t])) for t in range(Tmax + 1)]
                for K in KS} for a in R}
    json.dump({"selected_tau": sel, "curves_mean": dump, "fuel_check": fuel,
               "ctx_n": ctx_n}, open(RES, "w"), indent=1, default=float)
    print(f"\n[eval] wrote {MD} + {RES}", flush=True)


def _write_results(ar, dt, R, sel, b2_gain, ctx_n):
    md("## DEV-TEST RESULTS (endpoint = T=24)\n\n")
    md("| arm | NDCG@50 T24 | NDCG@10 T24 | anytime@10 | @50 T8 | @50 T16 |\n|---|--:|--:|--:|--:|--:|\n")
    order = ["b0", "b1", "b2", "b3", "b4", "A", "A_pure", "B", "B_pure", "C", "C_pure",
             "D", "D_pure", "ttab", "clair"]
    label = {"b0": "b0 cold", "b1": "b1 concept-entropy", "b2": "b2 learned static",
             "b3": "b3 b2+skip+tail", "b4": "b4 myopic (blind J)",
             "A": f"A scorer (tau{sel['A']})", "A_pure": "A scorer pure",
             "B": f"B gradient (tau{sel['B']})", "B_pure": "B gradient pure",
             "C": f"C CAT (tau{sel['C']})", "C_pure": "C CAT pure",
             "D": f"D Golbandi (tau{sel['D']})", "D_pure": "D Golbandi pure",
             "ttab": f"[ctx] true-table (n={ctx_n})", "clair": f"[ctx] clairvoyant approx (n={ctx_n})"}
    for a in order:
        if a not in R:
            continue
        c = R[a]["curves"]
        md(f"| {label[a]} | {endpoint(c,50,24):.4f} | {endpoint(c,10,24):.4f} | {anytime(c,10):.4f} | "
           f"{endpoint(c,50,8):.4f} | {endpoint(c,50,16):.4f} |\n")
    md("\n(context rows are computed on the first "
       f"{ctx_n} DEV-TEST users -- do NOT difference them against full-cohort rows.)\n")
    md("\n### Paired deltas vs b2 and vs b4 (DEV-TEST endpoint@50 T=24; E4)\n\n")
    md("| arm | vs b2 | vs b4 |\n|---|---|---|\n")
    b2e = per_user_endpoint(R["b2"], 50); b4e = per_user_endpoint(R["b4"], 50)
    for a in ["b1", "b3", "b4", "A", "A_pure", "B", "B_pure", "C", "C_pure", "D", "D_pure"]:
        if a not in R:
            continue
        ae = per_user_endpoint(R[a], 50)
        d2 = paired_ci([x - y for x, y in zip(ae, b2e)])
        d4 = paired_ci([x - y for x, y in zip(ae, b4e)]) if a != "b4" else None
        md(f"| {label[a]} | {_fmt_ci(d2)} | {_fmt_ci(d4) if d4 else '-'} |\n")
    d = paired_ci([x - y for x, y in zip(b4e, b2e)])
    md(f"\n**b4 vs b2 (computation-suffices check): {_fmt_ci(d)}**\n\n")
    md(f"b2 per-step greedy gains (first 8): {[round(g, 4) for g in b2_gain[:8]]}\n\n")
    print("\n=== DEV-TEST endpoint@50 T24 ===", flush=True)
    for a in order:
        if a in R:
            print(f"  {label[a]:34s} {endpoint(R[a]['curves'],50,24):.4f}  "
                  f"@10 {endpoint(R[a]['curves'],10,24):.4f}", flush=True)
    print(f"\n  b4 vs b2 @50: {_fmt_ci(d)}", flush=True)


def _mechanism(ar, dt, R):
    md("## MECHANISM READOUTS (does coarse-to-fine EMERGE?)\n\n")
    md("| arm | refusal% (T1-4/T21-24) | channel mix T1-4 -> T21-24 (conc/ent/item %) | "
       "mean prior-p_ans asked (early/late) | divergence-from-b2 turn |\n|---|---|---|---|---|\n")
    b2_asked = R["b2"]["asked"]
    p0 = 1.0 / (1.0 + np.exp(-ar.p0_logit))
    lines = []
    for a in ["b1", "b2", "b3", "b4", "A", "B", "C", "D"]:
        if a not in R:
            continue
        asked = R[a]["asked"]; ansf = R[a]["ansf"]
        early_ref = _rate(ansf, 0, 4); late_ref = _rate(ansf, 20, 24)
        cm_e = _chanmix(ar, asked, 0, 4); cm_l = _chanmix(ar, asked, 20, 24)
        gr_e = _granularity(p0, asked, 0, 4); gr_l = _granularity(p0, asked, 20, 24)
        div = _divergence(asked, b2_asked)
        md(f"| {a} | {early_ref:.0%}/{late_ref:.0%} | {cm_e} -> {cm_l} | {gr_e:.2f}/{gr_l:.2f} | "
           f"{div:.1f} |\n")
        lines.append((a, cm_e, cm_l))
    md("\n**Channel-mix one-liners:**\n")
    for a, e, l in lines:
        md(f"- {a}: " + (f"STATIC mix ({e})" if e == l else f"mix moved {e} -> {l}") + "\n")
    md("\n")


def _rate(ansf, t0, t1):
    vals = []
    for f in ansf:
        seg = f[t0:t1]
        if seg:
            vals.append(np.mean([not x for x in seg]))
    return float(np.mean(vals)) if vals else 0.0


def _chanmix(ar, asked, t0, t1):
    grp = {"c": 0, "e": 0, "i": 0}; tot = 0
    for qs in asked:
        for qi in qs[t0:t1]:
            ch = int(ar.q_channel[qi])
            g = "c" if ch == TYPE_CONCEPT else ("e" if ch == TYPE_ENTITY else "i")
            grp[g] += 1; tot += 1
    if tot == 0:
        return "-"
    return "/".join(f"{100*grp[k]//tot}" for k in ("c", "e", "i"))


def _granularity(p0, asked, t0, t1):
    vals = [p0[qi] for qs in asked for qi in qs[t0:t1]]
    return float(np.mean(vals)) if vals else 0.0


def _divergence(asked, b2_asked):
    divs = []
    for i, qs in enumerate(asked):
        b = b2_asked[i] if i < len(b2_asked) else []
        d = Tmax
        for t in range(min(len(qs), len(b))):
            if qs[t] != b[t]:
                d = t + 1; break
        divs.append(d)
    return float(np.mean(divs)) if divs else float(Tmax)


def _transcripts(ar, dt, R):
    md("## TRANSCRIPTS (3 DEV-TEST users, b4 arm; question names + answers)\n\n")
    arm = "b4" if "b4" in R else "A"
    asked = R[arm]["asked"]
    lvl_name = {0: "no_clue(REFUSE)", 1: "rough", 2: "know_well"}
    for i in range(min(3, len(dt))):
        ctx = ar.user_ctx(dt[i])
        md(f"**user {dt[i]['u']}** ({arm}):\n\n")
        for t, qi in enumerate(asked[i][:12]):
            k = ar.answer_level(dt[i]["u"], qi, ctx)
            v = int(ctx["table"]["val"][qi])
            vv = f" val={['hated','meh','liked','loved'][v]}" if k >= 1 and v >= 0 else ""
            md(f"- T{t+1}: {ar.q_names[qi]} -> {lvl_name[k]}{vv}\n")
        md("\n")


def _verdicts(R):
    md("## VERDICTS (DEV; pre-registered contrasts)\n\n")
    b2e = per_user_endpoint(R["b2"], 50); b4e = per_user_endpoint(R["b4"], 50)
    d = paired_ci([x - y for x, y in zip(b4e, b2e)])
    if d["lo"] > 0:
        md(f"- **b4 vs b2: WIN** ({_fmt_ci(d)}) -> adaptivity pays via COMPUTATION; learned "
           "policies must now beat b4.\n")
    elif d["hi"] < 0:
        md(f"- **b4 vs b2: LOSS** ({_fmt_ci(d)}) -> blind model-based myopia hurts here.\n")
    else:
        md(f"- **b4 vs b2: TIE** ({_fmt_ci(d)}) -> computation alone does not pay in this arena.\n")
    for cls in ["A", "B", "C", "D"]:
        if cls not in R:
            continue
        ae = per_user_endpoint(R[cls], 50)
        d2 = paired_ci([x - y for x, y in zip(ae, b2e)])
        d4 = paired_ci([x - y for x, y in zip(ae, b4e)])
        v2 = "WIN" if d2["lo"] > 0 else ("LOSS" if d2["hi"] < 0 else "TIE")
        v4 = "WIN" if d4["lo"] > 0 else ("LOSS" if d4["hi"] < 0 else "TIE")
        md(f"- **{cls} vs b2: {v2}** ({_fmt_ci(d2)}); **vs b4: {v4}** ({_fmt_ci(d4)})\n")
    md("\nNO headline claim -- DEV only; the 173/300 real users are out of bounds (execution scope "
       "guard).\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["build", "eval"])
    ap.add_argument("--n_train", type=int, default=1000)
    ap.add_argument("--n_devval", type=int, default=80)
    ap.add_argument("--n_devtest", type=int, default=160)
    ap.add_argument("--n_b2", type=int, default=300)         # fix #3
    ap.add_argument("--b2_prescreen", type=int, default=300)
    ap.add_argument("--n_gol", type=int, default=300)
    ap.add_argument("--n_scorer", type=int, default=1000)
    ap.add_argument("--n_labels", type=int, default=3500)
    ap.add_argument("--cand_M", type=int, default=100)       # fix: ONE shared cap for b4/A/B/C
    ap.add_argument("--ctx_n", type=int, default=50)
    a = ap.parse_args()
    if a.stage == "build":
        stage_build(a)
    else:
        stage_eval(a)
