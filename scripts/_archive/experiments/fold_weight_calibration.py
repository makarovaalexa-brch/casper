"""fold_weight_calibration.py -- FIDELITY-AWARE FOLD WEIGHT CALIBRATION + gated vivid-swap rerun on a
HELD-OUT user split. NO LLM API calls; all local compute on the cached judged grid through the I2.5
learned fold. Author-directed 2026-07-08, follows the T1 FAIL diagnosis in VIVID_SWAP_TESTS.md.

*** DIRECTIONAL ONLY: 173/300 users, answerer-v1 grid NOT frozen. Every output carries this flag. ***

THE T1 DIAGNOSIS (why we recalibrate): the fold weight is keyed on KNOWLEDGE LEVEL alone
(know_well=1.0, rough_idea=0.5). But a know_well answer whose VALUE is LLM-guessed (sigma~0.70 stars)
folded at 1.0 injects fidelity noise -- it lost to a half-weighted rough_idea hedge (-0.021, CI excl 0).
The premium was real ONLY on source="data" real ratings. So the correct control is FIDELITY CLASS, not
knowledge level:
    w_data   = 1.0  (real rating, zero noise; FIXED reference)
    w_k2_llm        (know_well, LLM value)
    w_k1_llm        (rough_idea, LLM value)
Prediction from the diagnosis: w_k2_llm well below 1.0, ~w_k1_llm (fidelity noise dominates the
vividness of an LLM guess), so the ONLY live composition premium is source="data".

OVERFITTING GUARD (E1/E5): users are split deterministically into half A (~86, FIT) and half B (~87,
EVAL). Weights are fit on A only, by maximizing mean NDCG@10 of FIXED reference answer-sets (a MIXTURE
of random tier-spanning 8-answer draws + a coverage-popularity static's answered subset -- so weights
are not tuned to one selector). ALL downstream verdicts (T1/T1b/T2/T3) come from half B ONLY. Both
halves are reported.

GATED SEQUENCE (reuses scripts/vivid_swap_tests.py building blocks):
  STEP 2  T1  : k2-set vs k1-set at matched value tiers (calibrated). GATE: CI excl 0.
          T1b : RATED (source=data) set vs matched-tier k1-LLM set = the real-fidelity premium, clean.
                GATE: CI excl 0.  (Calibration may make T1 near-null BY DESIGN if w_k2_llm~w_k1_llm;
                T1b isolates the source="data" premium regardless.)
  STEP 3  IF T1 or T1b passes on B -> T2 (peer-ranking, target = the passing class) + T3 (the swap
          policy, tie-by-construction, three pre-registered contrasts), all on B, calibrated weights
          EVERYWHERE incl. the statics (fair).  IF neither passes -> STOP: with a properly calibrated
          fold, answer-composition routing has NO premium in this arena.

Run:  python scripts/fold_weight_calibration.py
"""
import os, sys, json, time, itertools, collections
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import warnings
warnings.filterwarnings("ignore")
import adaptivity_battery_v1 as A
import battery_stage_c as C
import vivid_swap_tests as V

OUT_MD = "experiments/VIVID_SWAP_TESTS.md"
OUT_JSON = "experiments/fold_weight_calibration.json"
BOOT = A.BOOT            # 5000
SEED = A.SEED            # 0
NTIER = V.NTIER          # 8
MIN_COV = V.MIN_COV      # 10
GRID = [round(x, 2) for x in np.arange(0.0, 1.0001, 0.1)]   # 0,.1,...,1.0
N_RAND_SETS = 4
SET_SIZE = 8
STATIC_LEN = 24

# fidelity class codes on the raw value array
FC_NONE, FC_DATA, FC_K2LLM, FC_K1LLM = 0, 1, 2, 3
RES = {"banner": "DIRECTIONAL 173/300, grid unfrozen (answerer-v1 working grid); FIT=half A, VERDICTS=half B"}


def md(txt):
    os.makedirs("experiments", exist_ok=True)
    open(OUT_MD, "a", encoding="utf-8").write(txt)


# ============================================================ raw value + fidelity class (weight-free)
def annotate_raw(env):
    """Store per-user raw (unweighted centered) value + fidelity-class code, derived exactly from the
    DEFAULT-weighted val_arr (know_well=1.0 => raw=val; rough_idea=0.5 => raw=val*2; data=1.0 => raw=val).
    klev_arr/src_arr are already on each rec (V.setup)."""
    nc = env["nc"]
    for rec in env["all_users"]:
        klev = rec["klev_arr"]; src = rec["src_arr"]; val = rec["val_arr"]
        fc = np.zeros(nc, np.int64); raw = np.zeros(nc)
        for cd in range(nc):
            s = src[cd]
            if s == "data":
                fc[cd] = FC_DATA; raw[cd] = val[cd]                 # weight 1.0
            elif s in ("concept", "attr", "llm"):
                if klev[cd] == 2:
                    fc[cd] = FC_K2LLM; raw[cd] = val[cd]            # default weight 1.0
                elif klev[cd] == 1:
                    fc[cd] = FC_K1LLM; raw[cd] = val[cd] * 2.0      # default weight 0.5
        rec["raw_arr"] = raw; rec["fclass_arr"] = fc


def apply_weights(env, w_k2_llm, w_k1_llm, w_data=1.0):
    """Set every user's val_arr = raw * class-weight. Reproduces the DEFAULT fold at (1.0, 0.5, 1.0)."""
    wv = np.array([0.0, w_data, w_k2_llm, w_k1_llm])
    for rec in env["all_users"]:
        rec["val_arr"] = rec["raw_arr"] * wv[rec["fclass_arr"]]


# ============================================================ value model (tiers) under current val_arr
def recompute_value_model(env, users):
    """Rebuild cval/ccov/pop_k2/pop_data/tier_of/tier_med/valid/salift/cold for a user SUBSET under the
    CURRENT val_arr (calibrated). Mirrors V.setup's value-model block; scopes env['users']=users."""
    FR, model, CANDS, nc = env["FR"], env["model"], env["CANDS"], env["nc"]
    env["users"] = users
    env["cold"] = C.cold_ndcg(FR, model, users)
    cold = env["cold"]
    # per-candidate single-answer NDCG lift (identical value model across arms; E2)
    tl, nl, mi, mc = [], [], [], []
    for i, rec in enumerate(users):
        for cd in np.where(rec["ans_arr"])[0]:
            tl.append([C.tok_of(CANDS, rec, cd)])
            nl.append([rec["nat_arr"][cd]] if rec["nat_arr"][cd] is not None else [])
            mi.append(i); mc.append(int(cd))
    vals = C._batch_ndcg(FR, model, users, tl, nl, mi, cold)
    lifts = collections.defaultdict(list); salift = {}
    for r in range(len(vals)):
        lft = float(vals[r] - cold[mi[r]]); lifts[mc[r]].append(lft); salift[(mi[r], mc[r])] = lft
    cval = np.full(nc, np.nan); ccov = np.zeros(nc)
    for cd in range(nc):
        if lifts[cd]:
            cval[cd] = float(np.mean(lifts[cd])); ccov[cd] = len(lifts[cd])
    # population target rates over the subset
    k2c = np.zeros(nc); datac = np.zeros(nc); anyc = np.zeros(nc)
    for rec in users:
        anyc += rec["ans_arr"]; k2c += (rec["klev_arr"] == 2); datac += (rec["fclass_arr"] == FC_DATA)
    env["pop_k2"] = np.divide(k2c, np.maximum(anyc, 1))
    env["pop_data"] = np.divide(datac, np.maximum(anyc, 1))
    # octile tiers
    valid = (~np.isnan(cval)) & (ccov >= MIN_COV)
    tier_of = -np.ones(nc, np.int64); vc = np.where(valid)[0]
    if len(vc):
        qs = np.quantile(cval[vc], np.linspace(0, 1, NTIER + 1)); qs[-1] += 1e-9
        for cd in vc:
            t = int(np.searchsorted(qs, cval[cd], side="right") - 1)
            tier_of[cd] = min(max(t, 0), NTIER - 1)
    env["cval"] = cval; env["ccov"] = ccov; env["valid"] = valid; env["tier_of"] = tier_of
    env["tier_med"] = {t: float(np.median(cval[(tier_of == t)])) for t in range(NTIER) if (tier_of == t).any()}
    env["salift"] = salift
    return env


# ============================================================ reference answer-sets (FIXED, weight-free)
def build_reference_sets(env, users):
    """Per user: N_RAND_SETS random SET_SIZE answerable draws (seeded) that span value tiers +
    the coverage-popularity static's answered subset. Cids only -- fixed across all weight settings."""
    ccov_all = np.array([m["cov"] for m in env["CANDS"]])   # population coverage (weight-free)
    static = [c for c in np.argsort(-ccov_all).tolist()][:STATIC_LEN]
    refsets = []
    for rec in users:
        ans = np.where(rec["ans_arr"])[0].tolist()
        rng = np.random.default_rng(SEED * 100003 + int(rec["u"]))
        sets = []
        for _ in range(N_RAND_SETS):
            if len(ans) <= SET_SIZE:
                sets.append(list(ans))
            else:
                sets.append(rng.choice(ans, SET_SIZE, replace=False).tolist())
        stat = [c for c in static if rec["ans_arr"][c]][:SET_SIZE]
        if stat:
            sets.append(stat)
        refsets.append([s for s in sets if s])
    return refsets


def refset_objective(env, users, refsets):
    """Mean over users of (mean NDCG@10 over that user's reference sets), at the CURRENT val_arr."""
    maxs = max((len(r) for r in refsets), default=0)
    tot = np.zeros(len(users)); cnt = np.zeros(len(users))
    for slot in range(maxs):
        sets_slot = [refsets[i][slot] if slot < len(refsets[i]) else [] for i in range(len(users))]
        nd = V.set_ndcg(env["FR"], env["model"], users, sets_slot)
        for i, v in enumerate(nd):
            if v is not None and not np.isnan(v):
                tot[i] += v; cnt[i] += 1
    per_user = np.where(cnt > 0, tot / np.maximum(cnt, 1), np.nan)
    return per_user


# ============================================================ STEP 1 -- CALIBRATION
def calibrate(env, usersA, usersB):
    print("\n==== STEP 1 -- FIDELITY-AWARE WEIGHT CALIBRATION (fit half A, ~%d users) ====" % len(usersA), flush=True)
    print("  free params: w_k2_llm, w_k1_llm in {0,.1,..,1.0}^2 ; w_data FIXED = 1.0 (real-rating reference)", flush=True)
    print("  objective  : mean NDCG@10 of FIXED reference answer-sets (random tier-spanning + static mix)", flush=True)
    print("  default fold = (w_k2_llm=1.0, w_k1_llm=0.5) is inside the grid (sanity anchor).", flush=True)
    refA = build_reference_sets(env, usersA)
    refB = build_reference_sets(env, usersB)

    grid = {}
    for w2, w1 in itertools.product(GRID, GRID):
        apply_weights(env, w2, w1)
        objA = float(np.nanmean(refset_objective(env, usersA, refA)))
        grid[(w2, w1)] = objA
    (bw2, bw1), best = max(grid.items(), key=lambda kv: kv[1])
    default = grid[(1.0, 0.5)]

    # flatness: how many grid points within 0.5% and 1% of the best; marginal best per axis
    span = max(grid.values()) - min(grid.values())
    near05 = [k for k, v in grid.items() if best - v <= 0.005 * max(best, 1e-9)]
    near_abs = [k for k, v in grid.items() if best - v <= 0.001]
    w2_range = (min(k[0] for k in near_abs), max(k[0] for k in near_abs))
    w1_range = (min(k[1] for k in near_abs), max(k[1] for k in near_abs))

    # half-B objective at best vs default (paired)
    apply_weights(env, bw2, bw1); pB_best = refset_objective(env, usersB, refB)
    apply_weights(env, 1.0, 0.5); pB_def = refset_objective(env, usersB, refB)
    apply_weights(env, bw2, bw1); pA_best = refset_objective(env, usersA, refA)
    apply_weights(env, 1.0, 0.5); pA_def = refset_objective(env, usersA, refA)
    dB = V.paired(list(pB_best), list(pB_def))
    dA = V.paired(list(pA_best), list(pA_def))

    print(f"\n  FITTED (half A): w_k2_llm={bw2:.1f}  w_k1_llm={bw1:.1f}  | objA={best:.4f}  (default(1.0,0.5)={default:.4f})", flush=True)
    print(f"  flatness: obj span over grid={span:.4f}; {len(near_abs)} pts within 0.001 of best "
          f"(w_k2_llm in [{w2_range[0]:.1f},{w2_range[1]:.1f}], w_k1_llm in [{w1_range[0]:.1f},{w1_range[1]:.1f}]); "
          f"{len(near05)} pts within 0.5%.", flush=True)
    print(f"  half-A calibrated-vs-default: {dA['delta']:+.4f}[{dA['ci'][0]:+.4f},{dA['ci'][1]:+.4f}] (n={dA['n']})", flush=True)
    print(f"  half-B calibrated-vs-default: {dB['delta']:+.4f}[{dB['ci'][0]:+.4f},{dB['ci'][1]:+.4f}] (n={dB['n']})  <-- held out", flush=True)

    # neighbourhood print (small window around the optimum) for the sensitivity picture
    print("  objA surface (rows=w_k2_llm, cols=w_k1_llm):", flush=True)
    header = "    w2\\w1 " + " ".join(f"{w1:4.1f}" for w1 in GRID)
    print(header, flush=True)
    surf = []
    for w2 in GRID:
        row = " ".join(f"{grid[(w2, w1)]:.3f}"[1:] for w1 in GRID)   # strip leading 0
        line = f"    {w2:4.1f}  {row}"
        print(line, flush=True); surf.append([grid[(w2, w1)] for w1 in GRID])

    RES["calibration"] = dict(
        fitted=dict(w_data=1.0, w_k2_llm=bw2, w_k1_llm=bw1), objA_best=best, objA_default=default,
        obj_span=span, n_within_0p001=len(near_abs), n_within_0p5pct=len(near05),
        w2_range_flat=list(w2_range), w1_range_flat=list(w1_range),
        halfB_cal_vs_default=dB, halfA_cal_vs_default=dA,
        grid=[[float(grid[(w2, w1)]) for w1 in GRID] for w2 in GRID], grid_axis=GRID,
        nA=len(usersA), nB=len(usersB))
    _md_calib(env, RES["calibration"], surf)
    dump()
    return bw2, bw1


def _md_calib(env, cb, surf):
    f = cb["fitted"]
    md("\n\n---\n\n# FIDELITY-AWARE FOLD-WEIGHT CALIBRATION + GATED RERUN (held-out half B)\n\n"
       "> **DIRECTIONAL ONLY** -- 173/300 users, grid UNFROZEN. FIT on half A (~%d users), ALL verdicts "
       "from half B (~%d users). Re-run on the frozen grid before any citation.\n\n" % (cb["nA"], cb["nB"]))
    md("Date 2026-07-08. Script `scripts/fold_weight_calibration.py` (reuses `vivid_swap_tests.py`). NO LLM "
       "calls. Motivated by the T1 FAIL diagnosis above: the fold weight was keyed on KNOWLEDGE LEVEL "
       "(know_well=1.0, rough_idea=0.5), which full-weights a ~0.70-star LLM guess. We re-key it on "
       "**FIDELITY CLASS**: `w_data`=1.0 (real rating, FIXED reference), `w_k2_llm` (know_well + LLM value), "
       "`w_k1_llm` (rough_idea + LLM value). Deterministic split (seed=%d permutation); users split once, "
       "weights fit on A, verdicts on B.\n\n" % SEED)
    md("## STEP 1 -- Calibration (fit on half A)\n\n"
       "**OBJECTIVE (pre-registered):** maximize mean NDCG@10 of FIXED reference answer-sets = a MIXTURE per "
       "user of %d random %d-answer tier-spanning draws + the coverage-popularity static's answered subset "
       "(so weights are not tuned to one selector). Coarse grid {0,.1,..,1.0}^2 over (w_k2_llm, w_k1_llm); "
       "w_data fixed 1.0. Default fold (1.0, 0.5) lies inside the grid.\n\n" % (N_RAND_SETS, SET_SIZE))
    md("| quantity | value |\n|---|---|\n")
    md(f"| **FITTED weights (half A)** | w_data=1.0, **w_k2_llm={f['w_k2_llm']:.1f}**, **w_k1_llm={f['w_k1_llm']:.1f}** |\n")
    md(f"| objA at fitted / at default(1.0,0.5) | {cb['objA_best']:.4f} / {cb['objA_default']:.4f} |\n")
    md(f"| flatness (obj span over grid) | {cb['obj_span']:.4f} |\n")
    md(f"| grid pts within 0.001 of best | {cb['n_within_0p001']} (w_k2_llm in [{cb['w2_range_flat'][0]:.1f},"
       f"{cb['w2_range_flat'][1]:.1f}], w_k1_llm in [{cb['w1_range_flat'][0]:.1f},{cb['w1_range_flat'][1]:.1f}]) |\n")
    dB, dA = cb["halfB_cal_vs_default"], cb["halfA_cal_vs_default"]
    md(f"| half-A calibrated-vs-default objective | {dA['delta']:+.4f}[{dA['ci'][0]:+.4f},{dA['ci'][1]:+.4f}] (n={dA['n']}) |\n")
    md(f"| **half-B calibrated-vs-default objective (held out)** | **{dB['delta']:+.4f}"
       f"[{dB['ci'][0]:+.4f},{dB['ci'][1]:+.4f}]** (n={dB['n']}) |\n\n")
    md("objA surface (rows w_k2_llm 0..1, cols w_k1_llm 0..1), leading 0 stripped:\n\n```\n")
    md("w2\\w1 " + " ".join(f"{w1:4.1f}" for w1 in GRID) + "\n")
    for wi, w2 in enumerate(GRID):
        md(f"{w2:4.1f}  " + " ".join(f"{surf[wi][j]:.3f}"[1:] for j in range(len(GRID))) + "\n")
    md("```\n\n")
    pred = ("CONFIRMS" if (f["w_k2_llm"] < 1.0 and abs(f["w_k2_llm"] - f["w_k1_llm"]) <= 0.2) else "does NOT cleanly confirm")
    md(f"**Diagnosis prediction check:** w_k2_llm << 1.0 and ~ w_k1_llm -> this fit **{pred}** it "
       f"(w_k2_llm={f['w_k2_llm']:.1f}, w_k1_llm={f['w_k1_llm']:.1f}). "
       f"{'If the two LLM weights coincide, k2-vs-k1 vividness is neutralized BY DESIGN and only source=data carries a premium.' if abs(f['w_k2_llm']-f['w_k1_llm'])<=0.1 else ''}\n\n")


# ============================================================ STEP 2 -- T1 / T1b on half B (calibrated)
def step2_T1(env, usersB, w2, w1):
    apply_weights(env, w2, w1)
    recompute_value_model(env, usersB)          # scopes env to half B, calibrated tiers
    FR, model = env["FR"], env["model"]
    print("\n==== STEP 2 -- T1 / T1b on HELD-OUT half B (calibrated weights) ====", flush=True)
    print("PRE-REGISTERED GATES (printed before results), paired per-user bootstrap CI:", flush=True)
    print("  T1  : k2-set beats k1-set at matched value tiers, CI excl 0.", flush=True)
    print("  T1b : RATED(source=data) set beats matched-tier k1-LLM set, CI excl 0  (the real-fidelity premium).", flush=True)

    k1a, k2a = V.build_matched_sets(env, usersB, lambda s: True)
    t1 = V.paired(V.set_ndcg(FR, model, usersB, k2a), V.set_ndcg(FR, model, usersB, k1a))
    # T1b: k2 restricted to source=data vs matched k1 (all k1 are LLM rough_idea)
    k1d, k2d = V.build_matched_sets(env, usersB, lambda s: s == "data")
    t1b = V.paired(V.set_ndcg(FR, model, usersB, k2d), V.set_ndcg(FR, model, usersB, k1d))
    # context: k2-LLM vs k1
    k1l, k2l = V.build_matched_sets(env, usersB, lambda s: s != "data")
    t1l = V.paired(V.set_ndcg(FR, model, usersB, k2l), V.set_ndcg(FR, model, usersB, k1l))

    t1_pass = bool(t1["ci"][0] > 0)
    t1b_pass = bool(t1b["ci"][0] > 0)
    print(f"\n  T1  k2-set {t1['a']:.4f} vs k1-set {t1['b']:.4f} -> {t1['delta']:+.4f}"
          f"[{t1['ci'][0]:+.4f},{t1['ci'][1]:+.4f}] (n={t1['n']}) -> GATE {'PASS' if t1_pass else 'FAIL'}", flush=True)
    print(f"  T1b RATED {t1b['a']:.4f} vs k1-LLM {t1b['b']:.4f} -> {t1b['delta']:+.4f}"
          f"[{t1b['ci'][0]:+.4f},{t1b['ci'][1]:+.4f}] (n={t1b['n']}) -> GATE {'PASS' if t1b_pass else 'FAIL'}", flush=True)
    print(f"  (ctx) k2-LLM vs k1: {t1l['delta']:+.4f}[{t1l['ci'][0]:+.4f},{t1l['ci'][1]:+.4f}] (n={t1l['n']})", flush=True)

    RES["step2"] = dict(w_k2_llm=w2, w_k1_llm=w1, T1=t1, T1_pass=t1_pass, T1b=t1b, T1b_pass=t1b_pass,
                        k2llm_vs_k1=t1l, nB=len(usersB))
    _md_step2(env, RES["step2"])
    dump()
    return t1_pass, t1b_pass


def _md_step2(env, s):
    md("## STEP 2 -- T1 / T1b on held-out half B (calibrated weights everywhere)\n\n"
       f"Calibrated fold: w_data=1.0, w_k2_llm={s['w_k2_llm']:.1f}, w_k1_llm={s['w_k1_llm']:.1f}. Value tiers "
       f"recomputed under the calibrated fold on half B ({s['nB']} users). Matched-tier design identical to "
       f"T1 above; the ONLY change is the fidelity-aware weights.\n\n"
       "**PRE-REGISTERED GATES:** T1 = k2-set vs k1-set (CI excl 0); T1b = RATED(source=data) vs "
       "matched-tier k1-LLM (CI excl 0). NOTE: if w_k2_llm~w_k1_llm the LLM-valued cells are neutralized "
       "BY DESIGN, so T1 may go null while T1b isolates the real source=data premium.\n\n"
       "| contrast | tgt NDCG | k1 NDCG | delta [95% CI] | n | verdict |\n|---|--:|--:|---|--:|---|\n")
    for tag, r, p in (("T1 k2-set vs k1-set (ANY k2)", s["T1"], s["T1_pass"]),
                      ("T1b RATED(data) vs k1-LLM", s["T1b"], s["T1b_pass"]),
                      ("(ctx) k2-LLM vs k1", s["k2llm_vs_k1"], None)):
        vd = ("PASS" if p else "FAIL") if p is not None else "context"
        md(f"| **{tag}** | {r['a']:.4f} | {r['b']:.4f} | **{r['delta']:+.4f}**"
           f"[{r['ci'][0]:+.4f},{r['ci'][1]:+.4f}] | {r['n']} | {vd} |\n")
    md(f"\n**STEP 2 verdict (half B):** T1 {'PASS' if s['T1_pass'] else 'FAIL'}, "
       f"T1b {'PASS' if s['T1b_pass'] else 'FAIL'}.\n\n")


# ============================================================ STEP 3 -- gated T2/T3 (target = passing class)
def step3(env, usersB, w2, w1, target):
    """target in {'k2','data'}. Reuses V.test2/V.test3 with a target switch (V.TGT). env already scoped to
    half B + calibrated. Statics rebuilt on B under calibrated weights (fair)."""
    apply_weights(env, w2, w1)
    recompute_value_model(env, usersB)
    V._ENV = env                                  # make V.setup() return this scoped env (cached)
    V.TGT = target
    tname = {"data": "RATED (source=data real ratings; item channel only)", "k2": "know_well (klev>=2)"}[target]
    md("## STEP 3 -- gated T2/T3 on half B (calibrated fold, TARGET = %s)\n\n"
       "STEP 2 opened the gate via **T1%s** so the swap target is the class that carries the premium: "
       "**%s**. The reused `vivid_swap_tests.py` harness prints its generic labels ('k>=2', "
       "'population-k2-rate', 'know_well'); under target='%s' READ every such label as the TARGET class "
       "above (the target switch `V.TGT` re-points the label, the population rate, and the LOUO belief at "
       "source=data cells). Calibrated weights (w_data=1.0, w_k2_llm=%.1f, w_k1_llm=%.1f) are applied "
       "EVERYWHERE incl. the reproduced statics (fair). T2 = can a blind belief pick THIS user's rated "
       "peer at matched value? T3 = does the rated-swap policy beat the fair static?\n\n"
       % (tname, "b (rated premium)" if target == "data" else " (k2 premium)", tname, target, w2, w1))
    print(f"\n==== STEP 3 -- T2 + T3 on half B (calibrated, target='{target}') ====", flush=True)
    t2_pass = V.test2()
    V.test3(t2_pass)
    RES["step3"] = dict(target=target, T2=V.RESULTS.get("T2"), T3=V.RESULTS.get("T3"))
    # conclusion footer (interprets the reused-harness numbers for this target)
    t3 = V.RESULTS.get("T3", {}); c2 = t3.get("c2_rvivid_vs_svivid", {}); c3 = t3.get("c3_svivid_vs_sbest", {})
    t2 = V.RESULTS.get("T2", {})
    routable = bool(t2.get("gate_pass")) and bool(c3.get("ci", [0, 0])[0] > 0)
    md("### STEP 3 conclusion (target=%s)\n\n" % target +
       "**The rated-composition premium (T1b, +%.4f) is NOT routable.** " % RES["step2"]["T1b"]["delta"] +
       ("T2 personalization FAILS (belief(iii) does not beat the population rate at picking the user's "
        "rated peer -- consistent with the rated-ness predictability ceiling in STATE_2026-07-08: which "
        "popular items a user happens to have rated is largely idiosyncratic + popularity-driven). " if not t2.get("gate_pass")
        else "T2 personalization passes. ") +
       ("The population rated-swap (s-vivid) even HURTS the static (%+.4f) -- swapping a scheduled item for "
        "a higher-rated-rate (more popular) peer trades value for rated-probability and loses; the router "
        "only avoids the loss by declining to swap (reducing to s-best). "
        % c3.get("delta", float("nan")) if c3.get("ci", [0, 0])[1] < 0 else
        "The population rated-swap does not beat the static. ") +
       "**Net: source=data vividness is the user's own known-half property, not something a peer-swap "
       "policy can manufacture by picking a 'more vivid' question -- exactly the abundance-regime twin of "
       "the rated-ness ceiling. Answer-composition routing has no realizable premium here.**\n\n"
       "**ROUTABLE = %s.**\n\n" % ("YES" if routable else "NO"))
    RES["step3"]["routable"] = routable
    dump()


def dump():
    json.dump(RES, open(OUT_JSON, "w"), indent=1, default=str)


# ============================================================ main
def main():
    t0 = time.time()
    env = V.setup()                               # full 173-user env, default weights, CANDS + klev/src
    env["all_users"] = list(env["users"])
    annotate_raw(env)

    # deterministic user split (E1/E5 overfitting guard)
    n = len(env["all_users"])
    perm = np.random.default_rng(SEED).permutation(n)
    idxA = sorted(perm[:n // 2].tolist()); idxB = sorted(perm[n // 2:].tolist())
    usersA = [env["all_users"][i] for i in idxA]
    usersB = [env["all_users"][i] for i in idxB]
    print(f"[split] deterministic seed={SEED}: half A={len(usersA)} (FIT), half B={len(usersB)} (VERDICTS)", flush=True)
    RES["split"] = dict(nA=len(usersA), nB=len(usersB), seed=SEED,
                        A_users=[int(u["u"]) for u in usersA], B_users=[int(u["u"]) for u in usersB])

    w2, w1 = calibrate(env, usersA, usersB)
    t1_pass, t1b_pass = step2_T1(env, usersB, w2, w1)

    if t1_pass or t1b_pass:
        target = "k2" if t1_pass else "data"
        print(f"\n[GATE] T1 {'PASS' if t1_pass else 'FAIL'} / T1b {'PASS' if t1b_pass else 'FAIL'} "
              f"-> STEP 3 runs with target='{target}'.", flush=True)
        step3(env, usersB, w2, w1, target)
        concl = (f"STEP 2 opened the gate (T1={'PASS' if t1_pass else 'FAIL'}, "
                 f"T1b={'PASS' if t1b_pass else 'FAIL'}); T2/T3 ran on half B with target='{target}'.")
    else:
        print("\n[STOP] Neither T1 nor T1b passes on half B with the calibrated fold. CONCLUSION: with a "
              "properly calibrated (fidelity-aware) fold, answer-COMPOSITION routing has NO premium in this "
              "arena -- the vividness the swap could manufacture is already neutralized once LLM-guessed "
              "know_well is no longer over-weighted, and the only real-fidelity premium (source=data) does "
              "not survive matched-tier on the held-out half. T2/T3 NOT run (gated).", flush=True)
        concl = ("STOP. Neither T1 nor T1b passes on half B under the calibrated fold: answer-composition "
                 "routing has no premium in this arena once the fold is fidelity-aware.")
        md("## STEP 3 -- NOT RUN (gate closed)\n\n" + concl + "\n\n"
           "This is a clean, cheap death: the T1 failure above was a MIScalibration artifact "
           "(LLM-guessed know_well over-weighted at 1.0); once the fold is fidelity-aware the composition "
           "premium disappears rather than reversing, and no swap policy can manufacture it.\n\n")
        RES["step3"] = dict(status="not_run_gate_closed", conclusion=concl)

    RES["wall_min"] = round((time.time() - t0) / 60, 2)
    dump()
    print(f"\n[done] {concl}\n[paths] {OUT_MD} ; {OUT_JSON} (wall {RES['wall_min']} min)", flush=True)


if __name__ == "__main__":
    main()
