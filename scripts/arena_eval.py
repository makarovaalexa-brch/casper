"""arena_eval.py -- THE POLICY ARENA driver (per DESIGN_SHEET_POLICY_ARENA.md).

Stages (checkpointed; re-runs resume from cache):
  build : cohorts + dense answer memoization + b1/b2/Golbandi/scorer-A construction on TRAIN.
  eval  : run every arm on DEV-TEST; DEV-VAL selects each adaptive class's b2-anchored TAU; curves,
          budgets, paired bootstrap CIs + MDE, mechanism readouts, transcripts, pre-registered verdicts.
          Writes experiments/ARENA_BUILD.md incrementally + .cache/arena/*.json.

DEV evaluation on SYNTHETIC users ONLY. The 173 real users are NEVER touched (execution scope guard).
NO LLM calls. ASCII. Deterministic. Print E7 symmetry table + pre-registered verdicts BEFORE results.

Run: python scripts/arena_eval.py build   [--n_train 1200 --n_devval 250 --n_devtest 400]
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
from arena_core import paired_ci, CH_NAME
from arena_policies import (StaticSeq, B0Cold, B4Myopic, TrueTableRouter, Clairvoyant, AskGradient,
                            CATRouter, GolbandiTree, ScorerA, B2Anchored, TYPE_ITEM, TYPE_CONCEPT,
                            TYPE_ATTR, TYPE_ENTITY, p_ans_pop, run_policy)

MD = "experiments/ARENA_BUILD.md"
RES = f"{AC.CACHE_DIR}/results.json"
Tmax = 24
KS = (50, 10)
BUDGETS = (8, 16, 24)
TAU_GRID = [0, 8, 24]           # b2-anchored handover turn; 24 == exactly b2 (tie by construction)


def md(txt, mode="a"):
    open(MD, mode, encoding="utf-8").write(txt)


# ================================================================= COHORTS + config
def get_cohorts(args):
    ar = AC.Arena(n_item_universe=args.n_item)
    coh = AC.make_cohorts(ar, n_train=args.n_train, n_devval=args.n_devval, n_devtest=args.n_devtest)
    return ar, coh


# ================================================================= BUILD stage
def stage_build(args):
    t0 = time.time()
    ar, coh = get_cohorts(args)
    print(f"[build] cohorts: train={len(coh['train'])} devval={len(coh['devval'])} "
          f"devtest={len(coh['devtest'])}", flush=True)
    # dense answer memoization (firewall: population users only)
    ar.prefill_answers(coh["train"], f"train{len(coh['train'])}")
    ar.prefill_answers(coh["devval"], f"devval{len(coh['devval'])}")
    ar.prefill_answers(coh["devtest"], f"devtest{len(coh['devtest'])}")
    # b2 construction on a subset for tractability (documented)
    b2_users = coh["train"][:args.n_b2]
    ar.prefill_answers(b2_users, f"train{len(coh['train'])}")  # already cached
    print("[build] constructing b2 greedy static (this is the long pole) ...", flush=True)
    b2_seq, _ = AP.build_b2(ar, b2_users, Tmax=Tmax, K=AP.PRIMARY_K)
    print("[build] b1 concept-entropy static ...", flush=True)
    AP.build_b1(ar, coh["train"][:400])
    print("[build] Golbandi tree ...", flush=True)
    AP.build_golbandi(ar, coh["train"][:args.n_gol], b2_seq, max_depth=5, min_users=25)
    print("[build] scorer-A GBM ...", flush=True)
    AP.build_scorerA(ar, coh["train"][:args.n_scorer], n_samples=args.n_labels)
    print(f"[build] DONE [{(time.time()-t0)/60:.1f} min]. All artifacts cached in {AC.CACHE_DIR}/",
          flush=True)


# ================================================================= E7 + pre-registered verdicts
def write_header(ar, coh):
    md("# THE POLICY ARENA -- BUILD + DEV RESULTS\n\n", mode="w")
    md("Per DESIGN_SHEET_POLICY_ARENA.md (signed 2026-07-09). DEV evaluation on SYNTHETIC users ONLY; "
       "the 173 real users are NEVER touched (execution scope guard). NO LLM calls. Deterministic "
       f"(seed {AC.SEED}).\n\n")
    md("## FOLD PICK (STEP 0)\n\n")
    md("**Deep-Sets fold-v3** (`.cache/i25_fold_v3_best.pt`, clean_frac=0.30, val NDCG@10 "
       f"{ar.fold_state.get('best_val'):.4f}) is the arena belief. The Set-Transformer contingency "
       "(cst30, fired by DS's marginal G6) FINISHED: best val 0.4123 (ep6), decisively below Deep-Sets' "
       "0.4311, and it re-gated WORSE -- FAIL on G2/G3/G4/G5/G6 (only G1/G2b/G7 pass), including the "
       "decisive G2 GoT gate that DS PASSES, and even G6 (the anti-saturation gate the ST was meant to "
       "fix) worse than DS (-0.0119 vs -0.0035). The contract required the ST to STRICTLY DOMINATE -- "
       "pass all gates AND match/beat DS on val AND on G4 -- to displace; it fails on every count. "
       "Deep-Sets wins outright. DS fold-v3 gate posture: the DECISIVE two-channel gates (G2 GoT, G2b "
       "prolific, G7 implicit-ablation) all PASS; G5 (no-harm vs v2, -0.028) and G6 (anti-sat, -0.0035, "
       "marginal one-step) are diagnosed failures (item-explicit specialization tradeoff), not the two-"
       "channel purpose -- so we do NOT fall back to fold-v2.\n\n")
    md("## WORLD CONFIG CAVEAT (fold-v3 no-clue semantics)\n\n")
    md("The trained Deep-Sets fold-v3 did NOT learn no-clue implicit tokens as NEGATIVE evidence: a "
       "no-clue token pulls TOWARD the asked region's area nearly as strongly as know_well (+0.244 vs "
       "+0.202 region-score pull -- the embedding direction dominates, the level flag modulates weakly). "
       "Arena consequences, enforced in code: (1) at INTERVIEW time refusal/no-clue tokens are NEVER "
       "folded into the taste belief (arena_core.tokens_for returns [] on no_clue) -- refusal = consumed "
       "turn, belief unchanged (E1); this forbids any variant that folds refusals. (2) No policy or "
       "mechanism readout leans on refusals-as-negative-taste; refusals inform only a policy's "
       "answerability/knowledge estimate (e.g. b4/A candidate answerability), never the taste vector. "
       "(3) Documented here so downstream readers see it.\n\n")
    md("## DEPLOYABILITY (blind agents; STATE 3.1/3.3, E3)\n\n")
    md("The adaptive arms (b4, A, B, C) are BLIND and deployable: they start cold and may NOT peek at "
       "the user's hidden profile. Each CHOOSES its next question using ONLY the current belief z and "
       "population channel priors -- answerability is estimated blind (channel base rate {item .95 / "
       "concept .74 / attr .82 / entity .55} bumped by belief-region alignment; at cold start = the "
       "channel prior) and the answer's value is FORECAST from the belief (decode z over the region's "
       "members), never from the true ratings. The agent OBSERVES the true answer only AFTER asking "
       "(the realized token then updates the belief). b2/b3/b1 are fixed sequences built offline on "
       "TRAIN users (no deploy-time peek); D branches only on the user's REALIZED answers (observed, "
       "not peeked). ONLY the labelled context arms (true-table, clairvoyant) peek at realized answers "
       "-- never cited as results. This forbids the a3/a4 privileged-answerability confound.\n\n")
    md("## E7 SYMMETRY TABLE (every arm's data / cohort / beliefs / fold, side by side)\n\n")
    md("All arms share: v2.1 answerer world (population trU users), the SAME picked Deep-Sets fold-v3 "
       "belief, the SAME full question universe, the LENIENT regime (refusal=consumed turn, belief "
       "unchanged, user kept), the SAME DEV-TEST users (E1). What differs is ONLY the question-"
       "selection policy and its construction data:\n\n")
    md("| arm | family | learns from | construction cohort | belief/fold | privileged? |\n")
    md("|---|---|---|---|---|---|\n")
    rows = [
        ("b0 cold", "orientation", "nothing", "-", "fold-v3", "no"),
        ("b1 concept-entropy", "static", "TRAIN answer-rates", f"{min(400,len(coh['train']))} TRAIN", "fold-v3", "no"),
        ("b2 learned static", "static", "TRAIN NDCG (greedy)", f"{args_n_b2} TRAIN", "fold-v3", "no"),
        ("b3 = b2+skip", "static+skip", "= b2", "= b2", "fold-v3", "no"),
        ("b4 myopic greedy", "model-based", "NONE (population answerer model)", "-", "fold-v3", "no"),
        ("A scorer-v2", "adaptive/learned", "TRAIN 1+2-step gain labels", f"{args_n_scorer} TRAIN", "fold-v3", "no"),
        ("B ask-gradient", "adaptive", "calibration only", "-", "fold-v3", "no"),
        ("C CAT-router", "adaptive", "NONE (Fisher proxy)", "-", "fold-v3", "no"),
        ("D Golbandi tree", "adaptive/learned", "TRAIN split+leaf NDCG", f"{args_n_gol} TRAIN", "fold-v3", "no"),
        ("ttab router", "CONTEXT (labelled)", "peeks realized answers", "-", "fold-v3", "YES-never cited"),
        ("clairvoyant", "CONTEXT (labelled)", "peeks realized answers", "-", "fold-v3", "YES-never cited"),
    ]
    for r in rows:
        md("| " + " | ".join(r) + " |\n")
    md("\nStatics train on the SAME synthetic world as policies (E2: no test-fit leak -- b2 is built on "
       "TRAIN users, evaluated on disjoint DEV-TEST).\n\n")
    md("## PRE-REGISTERED DEV VERDICTS (printed BEFORE results; E5)\n\n")
    md("Primary metric: endpoint NDCG@50 at T=24. Also reported: endpoint@10, anytime@10, budgets "
       "8/16/24. Paired per-user bootstrap CI + MDE on every delta (E4).\n\n")
    md("- **b4 vs b2** (the computation-suffices check): if b4 beats b2 (CI excl 0), that is the "
       "program's first honest adaptive win by COMPUTATION -- and every learned policy must then beat "
       "b4, not just b2. If b4 ties b2, computation alone does not pay here.\n")
    md("- **each adaptive class (A,B,C,D) vs b2 AND vs b4**: WIN = CI excl 0 above b2 AND >= b4. If "
       "only b4 wins: 'adaptivity pays via computation; learning adds nothing yet' (stated exactly).\n")
    md("- b2-anchored TAU is selected on DEV-VAL (TAU=24 reproduces b2 exactly = tie by construction); "
       "the pure policy (TAU=0) is reported alongside. No post-hoc metric selection.\n")
    md("- DEV ONLY. No headline claim. The 173 are out of bounds.\n\n")


# ================================================================= run + summarize an arm
def endpoint(curves, K, t):
    return np.nanmean(curves[K][:, t])


def anytime(curves, K):
    return np.nanmean(np.nanmean(curves[K][:, 1:], axis=1))


def run_arm(ar, recs, policy, skip=False, tag=""):
    t0 = time.time()
    out = run_policy(ar, recs, policy, Tmax, Ks=KS, skip=skip, verbose=False, tag=tag)
    print(f"    [{tag}] done [{time.time()-t0:.0f}s] endpoint@50={endpoint(out['curves'],50,Tmax):.4f} "
          f"@10={endpoint(out['curves'],10,Tmax):.4f}", flush=True)
    return out


def per_user_endpoint(out, K, t=Tmax):
    return out["curves"][K][:, t]


# ================================================================= EVAL stage
args_n_b2 = args_n_scorer = args_n_gol = None   # filled in eval for the header


def stage_eval(args):
    global args_n_b2, args_n_scorer, args_n_gol
    args_n_b2, args_n_scorer, args_n_gol = args.n_b2, args.n_scorer, args.n_gol
    ar, coh = get_cohorts(args)
    ar.prefill_answers(coh["train"], f"train{len(coh['train'])}")
    ar.prefill_answers(coh["devval"], f"devval{len(coh['devval'])}")
    ar.prefill_answers(coh["devtest"], f"devtest{len(coh['devtest'])}")
    dv, dt = coh["devval"], coh["devtest"]

    # ---- load constructed arms ----
    b2_seq, b2_gain = AP.build_b2(ar, coh["train"][:args.n_b2], Tmax=Tmax, K=AP.PRIMARY_K)
    b1_seq = AP.build_b1(ar, coh["train"][:400], verbose=False)
    gol_tree, gol_tail = AP.build_golbandi(ar, coh["train"][:args.n_gol], b2_seq, max_depth=5,
                                           min_users=25, verbose=False)
    gbm = AP.build_scorerA(ar, coh["train"][:args.n_scorer], n_samples=args.n_labels, verbose=False)

    write_header(ar, coh)
    _print_e7(ar)

    R = {}   # arm -> DEV-TEST out (curves etc.)
    print("\n=== DEV-TEST baselines ===", flush=True)
    R["b0"] = run_arm(ar, dt, B0Cold(), tag="b0")
    R["b1"] = run_arm(ar, dt, StaticSeq("b1", b1_seq), tag="b1")
    R["b2"] = run_arm(ar, dt, StaticSeq("b2", b2_seq), tag="b2")
    R["b3"] = run_arm(ar, dt, StaticSeq("b3", b2_seq), skip=True, tag="b3")
    R["b4"] = run_arm(ar, dt, B4Myopic(M=args.b4_M), tag="b4")

    # ---- adaptive classes: DEV-VAL selects b2-anchored TAU; report pure + selected on DEV-TEST ----
    classes = {
        "A": lambda: ScorerA(gbm, M=args.cand_M, Tmax=Tmax),
        "B": lambda: AskGradient(M=args.cand_M),
        "C": lambda: CATRouter(M=args.cand_M),
        "D": lambda: GolbandiTree(gol_tree, gol_tail),
    }
    sel = {}
    print("\n=== DEV-VAL model selection (b2-anchored TAU) ===", flush=True)
    # TAU=Tmax reproduces b2 exactly (deterministic) -> reuse ONE b2-on-DEV-VAL run for that option.
    b2_val = endpoint(run_policy(ar, dv, StaticSeq("b2", b2_seq), Tmax, Ks=KS, tag="b2@val")["curves"],
                      50, Tmax)
    print(f"    [b2] DEV-VAL endpoint@50={b2_val:.4f} (== TAU={Tmax} for every class)", flush=True)
    for cls, mk in classes.items():
        best_tau, best_v = Tmax, b2_val         # TAU=Tmax (=b2) is the tie-by-construction default
        for tau in [t for t in TAU_GRID if t < Tmax]:
            pol = B2Anchored(f"{cls}_tau{tau}", b2_seq, mk(), tau)
            o = run_policy(ar, dv, pol, Tmax, Ks=KS, tag=f"{cls}@val tau{tau}")
            v = endpoint(o["curves"], 50, Tmax)
            print(f"    [{cls}] DEV-VAL TAU={tau:2d}  endpoint@50={v:.4f}", flush=True)
            if v > best_v:
                best_v = v; best_tau = tau
        sel[cls] = best_tau
        print(f"  [{cls}] selected TAU={best_tau} (DEV-VAL endpoint@50={best_v:.4f})", flush=True)

    print("\n=== DEV-TEST adaptive arms (selected TAU + pure TAU=0) ===", flush=True)
    for cls, mk in classes.items():
        tau = sel[cls]
        R[cls] = run_arm(ar, dt, B2Anchored(f"{cls}_sel", b2_seq, mk(), tau), tag=f"{cls}_sel(tau{tau})")
        R[cls + "_pure"] = run_arm(ar, dt, B2Anchored(f"{cls}_pure", b2_seq, mk(), 0),
                                   tag=f"{cls}_pure")

    # ---- context arms (LABELLED, never cited) on a subset ----
    print("\n=== CONTEXT arms (labelled, never cited) ===", flush=True)
    ctx_n = min(args.ctx_n, len(dt))
    R["ttab"] = run_arm(ar, dt[:ctx_n], TrueTableRouter(M=args.b4_M), tag="ttab")
    R["clair"] = run_arm(ar, dt[:ctx_n], Clairvoyant(), tag="clair")

    _write_results(ar, dt, R, sel, b2_seq, b2_gain)
    _mechanism(ar, dt, R, b2_seq)
    _transcripts(ar, dt, R)
    _verdicts(R)
    # persist raw curves (means only, compact)
    dump = {a: {f"ndcg{K}": [float(np.nanmean(R[a]["curves"][K][:, t])) for t in range(Tmax + 1)]
                for K in KS} for a in R}
    json.dump({"selected_tau": sel, "curves_mean": dump}, open(RES, "w"), indent=1)
    print(f"\n[eval] wrote {MD} + {RES}", flush=True)


def _print_e7(ar):
    print("\n" + "=" * 70, flush=True)
    print("E7 SYMMETRY: all arms share world=v2.1 answerer(population trU), belief=Deep-Sets fold-v3,",
          flush=True)
    print("full universe, LENIENT regime (refusal=no-op turn, user kept), SAME DEV-TEST users (E1).",
          flush=True)
    print("Differs ONLY by selection policy + its construction cohort. Statics built on TRAIN (E2).",
          flush=True)
    print("=" * 70, flush=True)


def _fmt_ci(c):
    return f"{c['mean']:+.4f} [{c['lo']:+.4f},{c['hi']:+.4f}] MDE {c['mde']:.4f} (n={c['n']})"


def _write_results(ar, dt, R, sel, b2_seq, b2_gain):
    md("## DEV-TEST RESULTS (endpoint = T=24)\n\n")
    md("| arm | NDCG@50 T24 | NDCG@10 T24 | anytime@10 | @50 T8 | @50 T16 |\n|---|--:|--:|--:|--:|--:|\n")
    order = ["b0", "b1", "b2", "b3", "b4", "A", "A_pure", "B", "B_pure", "C", "C_pure",
             "D", "D_pure", "ttab", "clair"]
    label = {"b0": "b0 cold", "b1": "b1 concept-entropy", "b2": "b2 learned static",
             "b3": "b3 b2+skip", "b4": "b4 myopic greedy",
             "A": f"A scorer (tau{sel['A']})", "A_pure": "A scorer pure",
             "B": f"B gradient (tau{sel['B']})", "B_pure": "B gradient pure",
             "C": f"C CAT (tau{sel['C']})", "C_pure": "C CAT pure",
             "D": f"D Golbandi (tau{sel['D']})", "D_pure": "D Golbandi pure",
             "ttab": "[ctx] true-table", "clair": "[ctx] clairvoyant"}
    for a in order:
        if a not in R:
            continue
        c = R[a]["curves"]
        md(f"| {label[a]} | {endpoint(c,50,24):.4f} | {endpoint(c,10,24):.4f} | {anytime(c,10):.4f} | "
           f"{endpoint(c,50,8):.4f} | {endpoint(c,50,16):.4f} |\n")
    md("\n### Paired deltas vs b2 and vs b4 (DEV-TEST, endpoint@50 T=24; E4 bootstrap + MDE)\n\n")
    md("| arm | vs b2 | vs b4 |\n|---|---|---|\n")
    b2e = per_user_endpoint(R["b2"], 50); b4e = per_user_endpoint(R["b4"], 50)
    for a in ["b1", "b3", "b4", "A", "A_pure", "B", "B_pure", "C", "C_pure", "D", "D_pure"]:
        if a not in R:
            continue
        ae = per_user_endpoint(R[a], 50)
        d2 = paired_ci([x - y for x, y in zip(ae, b2e)])
        d4 = paired_ci([x - y for x, y in zip(ae, b4e)]) if a != "b4" else None
        md(f"| {label[a]} | {_fmt_ci(d2)} | {_fmt_ci(d4) if d4 else '-'} |\n")
    # b4 vs b2 headline row
    d = paired_ci([x - y for x, y in zip(b4e, b2e)])
    md(f"\n**b4 vs b2 (computation-suffices check): {_fmt_ci(d)}**\n\n")
    md(f"b2 greedy per-step gains (first 8): {[round(g,4) for g in b2_gain[:8]]}\n\n")
    # console
    print("\n=== DEV-TEST endpoint@50 T24 ===", flush=True)
    for a in order:
        if a in R:
            print(f"  {label[a]:26s} {endpoint(R[a]['curves'],50,24):.4f}  "
                  f"@10 {endpoint(R[a]['curves'],10,24):.4f}", flush=True)
    print(f"\n  b4 vs b2 @50: {_fmt_ci(d)}", flush=True)


def _mechanism(ar, dt, R, b2_seq):
    md("## MECHANISM READOUTS (does coarse-to-fine EMERGE?)\n\n")
    md("| arm | refusal% (early/late) | channel mix T1-4 -> T21-24 | mean p_ans asked (early/late) "
       "| divergence-from-b2 turn |\n|---|---|---|---|---|\n")
    b2_asked = R["b2"]["asked"]
    lines = []
    for a in ["b1", "b2", "b3", "b4", "A", "B", "C", "D"]:
        if a not in R:
            continue
        asked = R[a]["asked"]; ansf = R[a]["ansf"]
        n = len(asked)
        # refusal early/late
        early_ref = _rate(ansf, 0, 4, refuse=True); late_ref = _rate(ansf, 20, 24, refuse=True)
        # channel mix early/late (grouped concept/item/attribute[attr+entity])
        cm_e = _chanmix(ar, asked, 0, 4); cm_l = _chanmix(ar, asked, 20, 24)
        # granularity: mean population answerability of asked q
        gr_e = _granularity(ar, dt, asked, 0, 4); gr_l = _granularity(ar, dt, asked, 20, 24)
        # divergence from b2
        div = _divergence(asked, b2_asked)
        md(f"| {a} | {early_ref:.0%}/{late_ref:.0%} | {cm_e} -> {cm_l} | {gr_e:.2f}/{gr_l:.2f} | "
           f"{div:.1f} |\n")
        # one-liner: did channel mix change?
        shift = _mix_shift(cm_e, cm_l)
        lines.append((a, shift, cm_e, cm_l))
    md("\n**Channel-mix shift one-liners:**\n")
    for a, shift, e, l in lines:
        md(f"- {a}: {shift}\n")
    md("\n")


def _rate(ansf, t0, t1, refuse=False):
    vals = []
    for f in ansf:
        seg = f[t0:t1]
        if seg:
            r = np.mean([(not x) if refuse else x for x in seg])
            vals.append(r)
    return float(np.mean(vals)) if vals else 0.0


def _chanmix(ar, asked, t0, t1):
    grp = {"concept": 0, "item": 0, "attribute": 0}
    tot = 0
    for qs in asked:
        for qi in qs[t0:t1]:
            ch = ar.Q[qi][0]
            g = "concept" if ch == TYPE_CONCEPT else ("item" if ch == TYPE_ITEM else "attribute")
            grp[g] += 1; tot += 1
    if tot == 0:
        return "-"
    return "/".join(f"{k[:4]}{100*grp[k]//tot}" for k in ("concept", "item", "attribute"))


def _granularity(ar, dt, asked, t0, t1):
    vals = []
    for i, qs in enumerate(asked):
        ctx = ar.user_ctx(dt[i])
        for qi in qs[t0:t1]:
            _, _, _, surp = ar._answer_raw(dt[i]["u"], qi, ctx)
            vals.append(p_ans_pop(ar, ar.Q[qi][0], surp))
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


def _mix_shift(cm_e, cm_l):
    if cm_e == "-" or cm_l == "-":
        return "no data"
    if cm_e == cm_l:
        return f"STATIC channel mix ({cm_e})"
    return f"mix moved {cm_e} -> {cm_l}"


def _transcripts(ar, dt, R):
    md("## TRANSCRIPTS (3 DEV-TEST users, b4 arm; question names + answers)\n\n")
    arm = "b4" if "b4" in R else "A"
    asked = R[arm]["asked"]
    for i in range(min(3, len(dt))):
        ctx = ar.user_ctx(dt[i])
        md(f"**user {dt[i]['u']}** ({arm}):\n\n")
        for t, qi in enumerate(asked[i][:12]):
            lvl, val, fid, surp = ar._answer_raw(dt[i]["u"], qi, ctx)
            an = ("REFUSE" if lvl == 0 and False else
                  {0: "rough", 1: "know_well", 2: "no_clue"}.get(int(lvl), "?"))
            an = "no_clue(REFUSE)" if int(lvl) == 2 else an
            vv = f" val={val:+.2f}" if np.isfinite(val) else ""
            md(f"- T{t+1}: {ar.q_names[qi]} -> {an}{vv}\n")
        md("\n")


def _verdicts(R):
    md("## VERDICTS (DEV; pre-registered contrasts)\n\n")
    b2e = per_user_endpoint(R["b2"], 50); b4e = per_user_endpoint(R["b4"], 50)
    d = paired_ci([x - y for x, y in zip(b4e, b2e)])
    if d["lo"] > 0:
        md(f"- **b4 vs b2: WIN** ({_fmt_ci(d)}) -> adaptivity pays via COMPUTATION; learned policies "
           "must now beat b4.\n")
    elif d["hi"] < 0:
        md(f"- **b4 vs b2: LOSS** ({_fmt_ci(d)}) -> model-based myopia hurts here.\n")
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
    md("\nNO headline claim -- DEV only; the 173 real users are out of bounds (execution scope guard).\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["build", "eval"])
    ap.add_argument("--n_item", type=int, default=800)
    ap.add_argument("--n_train", type=int, default=1000)
    ap.add_argument("--n_devval", type=int, default=100)
    ap.add_argument("--n_devtest", type=int, default=200)
    ap.add_argument("--n_b2", type=int, default=100)
    ap.add_argument("--n_gol", type=int, default=200)
    ap.add_argument("--n_scorer", type=int, default=1000)
    ap.add_argument("--n_labels", type=int, default=3500)
    ap.add_argument("--cand_M", type=int, default=110)
    ap.add_argument("--b4_M", type=int, default=90)
    ap.add_argument("--ctx_n", type=int, default=60)
    a = ap.parse_args()
    args_n_b2, args_n_scorer, args_n_gol = a.n_b2, a.n_scorer, a.n_gol
    if a.stage == "build":
        stage_build(a)
    else:
        stage_eval(a)
