"""static_contamination.py -- measure the STATIC-BASELINE TEST-FIT CONTAMINATION.

HYPOTHESIS: every greedy static baseline in the program was CONSTRUCTED by greedy forward selection
maximizing cohort-mean NDCG@10 on the SAME users it is evaluated on (their held-out targets included).
Routers' online beliefs were firewalled to population data (E5); the statics never were. If the statics
carry a test-fit bonus, every static-vs-router contrast forfeits it -- biasing all adaptivity verdicts
AGAINST the router.

MEASUREMENT (all on the v2 fold .cache/i25_fold_v2_best.pt, 173-user answerer-v1 working grid):
  0. REPRODUCE: greedy static built on ALL 173, eval on ALL 173 -> must match anchor 0.2251 +/- 0.002.
  For each user split (seed 0, seed 1) into halves A/B, and each eval-half (B then A = swapped roles):
    IN-SAMPLE  arm: greedy built ON the eval-half,  eval ON the eval-half.
    OUT-SAMPLE arm: greedy built on the OTHER half, eval ON the eval-half (identical code, only the
                    construction cohort differs).
    CONTAMINATION = IN - OUT, paired per-user bootstrap over the eval-half users.
  => 4 estimates (seed0/B, seed0/A, seed1/B, seed1/A) + pooled.
  Context columns: T=8 / T=12 / T=24 (greedy is prefix-consistent, one T=24 build serves all); families
  s-item and s-mixed.
  Divergence: how many of the 12 picks differ between the A-built and B-built schedule + channel/tier
  composition of each.

NO LLM calls; all local. E-rules E1 (same-user paired bootstrap) / E5 (population-firewalled beliefs
n/a here) / E6 (pre-registered thresholds printed before results) honored. DIRECTIONAL 173/300.

Run:  python scripts/static_contamination.py --do all
"""
import os, sys, json, time, argparse, collections
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import warnings
warnings.filterwarnings("ignore")
import repair_probes as RP

OUT_MD = "experiments/STATIC_CONTAMINATION.md"
OUT_JSON = "experiments/static_contamination.json"
ANCHOR = 0.2251
ANCHOR_TOL = 0.002
BUDGETS = [8, 12, 24]
TMAX = max(BUDGETS)
BOOT = 5000

RESULTS = {"banner": "DIRECTIONAL 173/300, grid unfrozen (v2 fold answerer-v1 working grid)"}


def md(txt, mode="a"):
    os.makedirs("experiments", exist_ok=True)
    open(OUT_MD, mode, encoding="utf-8").write(txt)


# ------------------------------------------------------------ parametrized greedy build (users/cold/T)
def build_greedy_sub(env, pool_cids, sub_users, sub_cold, T):
    """Greedy forward selection maximizing cohort-mean NDCG@10 over sub_users (identical logic to
    RP.build_greedy, but on an explicit user subset + cold + budget T). Prefix-consistent."""
    FR, model, CANDS = env["FR"], env["model"], env["CANDS"]
    pool = sorted(pool_cids, key=lambda c: -CANDS[c]["cov"])[:RP.POOL]
    n = len(sub_users); sched = []
    for pos in range(T):
        cands = [c for c in pool if c not in sched]
        if not cands:
            break
        tl, nl, idx, owner = [], [], [], []
        base_ans = [[c for c in sched if rec["ans_arr"][c]] for rec in sub_users]
        for ci, c in enumerate(cands):
            for i, rec in enumerate(sub_users):
                cids = base_ans[i] + ([c] if rec["ans_arr"][c] else [])
                if not cids:
                    continue
                tl.append([RP.tok_of(CANDS, rec, cc) for cc in cids])
                nl.append([rec["nat_arr"][cc] for cc in cids if rec["nat_arr"][cc] is not None])
                idx.append(i); owner.append(ci)
        ndcg = RP.batch_ndcg(FR, model, sub_users, tl, nl, idx)
        sums = np.zeros(len(cands)); got = collections.defaultdict(set)
        for r, ci in enumerate(owner):
            sums[ci] += ndcg[r]; got[ci].add(idx[r])
        best, bc = -1.0, None
        for ci, c in enumerate(cands):
            mean = (sums[ci] + sum(sub_cold[i] for i in range(n) if i not in got[ci])) / n
            if mean > best:
                best, bc = mean, c
        sched.append(bc)
    return sched


def eval_static_peruser(env, sched, sub_users, sub_cold, T):
    """Per-user anytime NDCG@10 over the first T turns of a static schedule (refusal = no-op turn, cold
    fallback), on sub_users. Returns per-user anytime array (mean over turns 0..T-1)."""
    FR, model, CANDS = env["FR"], env["model"], env["CANDS"]
    n = len(sub_users)
    per_turn = np.empty((n, T))
    for t in range(T):
        col = sub_cold.copy(); tl, nl, idx = [], [], []
        for i, rec in enumerate(sub_users):
            cids = [c for c in sched[:t + 1] if (c is not None and rec["ans_arr"][c])]
            if not cids:
                continue
            tl.append([RP.tok_of(CANDS, rec, c) for c in cids])
            nl.append([rec["nat_arr"][c] for c in cids if rec["nat_arr"][c] is not None])
            idx.append(i)
        if tl:
            col[idx] = RP.batch_ndcg(FR, model, sub_users, tl, nl, idx)
        per_turn[:, t] = col
    return per_turn  # (n, T); caller slices/means


def cold_of(env, sub_users):
    return RP.cold_ndcg(env["FR"], sub_users)


# ------------------------------------------------------------ composition helpers
def sched_composition(env, sched, T=12):
    CANDS = env["CANDS"]; tier_of = env.get("tier_of")
    s = sched[:T]
    ch = collections.Counter(CANDS[c]["kind"] for c in s)
    tiers = None
    if tier_of is not None:
        tiers = [int(tier_of[c]) if tier_of[c] >= 0 else -1 for c in s]
    keys = [f"{CANDS[c]['kind']}:{CANDS[c]['key']}" for c in s]
    return dict(channels=dict(ch), tiers=tiers, keys=keys)


def divergence(env, sched_a, sched_b, T=12):
    """Set-based (how many of the T picks are not shared) + positionwise differences."""
    a, b = sched_a[:T], sched_b[:T]
    sa, sb = set(a), set(b)
    set_diff = len(sa ^ sb) // 1  # symmetric-diff count (elements not in both)
    n_a_only = len(sa - sb)
    pos_diff = sum(1 for i in range(min(len(a), len(b))) if a[i] != b[i])
    return dict(a_only=int(n_a_only), b_only=int(len(sb - sa)), shared=int(len(sa & sb)),
                symmetric_diff=int(len(sa ^ sb)), pos_diff=int(pos_diff), T=T)


# ------------------------------------------------------------ pre-registered thresholds
PREREG = (
    "## Pre-registered reads (printed BEFORE results)\n\n"
    "CONTAMINATION = (in-sample greedy static, built AND evaluated on the eval-half) minus\n"
    "(out-of-sample greedy static, built on the OTHER half, evaluated on the eval-half). Positive =\n"
    "the static gains from being fit to its own evaluation cohort. Paired per-user bootstrap over the\n"
    "eval-half users (E1).\n\n"
    "- **CONTAMINATED**: pooled contamination CI EXCLUDES 0 AND point estimate >= 0.005 ->\n"
    "  every prior static-vs-router contrast needs split-constructed re-runs; the true (fair) static\n"
    "  scores ~X lower, so each router's reported deficit shrinks by ~X.\n"
    "- **HYPOTHESIS DEAD**: pooled contamination CI INCLUDES 0 OR point estimate < 0.005 ->\n"
    "  the greedy statics do NOT meaningfully overfit their cohort; statics-optimal is strengthened\n"
    "  (this outcome is equally valuable -- it removes a confound from the closed adaptivity verdict).\n\n"
    "Verdict uses the POOLED estimate (all 4 per-user delta vectors concatenated). Per-estimate CIs\n"
    "reported alongside. Reproduction gate: all-173 greedy s-item anytime@T=12 == 0.2251 +/- 0.002.\n\n"
)


# ------------------------------------------------------------ main
def run():
    t0 = time.time()
    env = RP.setup()
    env = RP.value_tiers(env)  # adds tier_of for composition reporting
    users = env["users"]; n = len(users)
    CANDS = env["CANDS"]
    item_pool = [m["cid"] for m in CANDS if m["kind"] == "item"]
    mixed_pool = [m["cid"] for m in CANDS]
    families = {"s-item": item_pool, "s-mixed": mixed_pool}

    md("# STATIC-BASELINE TEST-FIT CONTAMINATION (v2 fold; DIRECTIONAL 173/300)\n\n", mode="w")
    md("> **DIRECTIONAL ONLY** -- 173/300 users, answerer-v1 working grid NOT frozen; v2 fold\n"
       "> `.cache/i25_fold_v2_best.pt`. Re-run on the frozen 300-user grid before any citation.\n\n"
       f"Date 2026-07-08. Script `scripts/static_contamination.py`. NO LLM API calls; local compute.\n"
       f"{n} users; {len(CANDS)} candidates; cold NDCG@10 {env['cold'].mean():.4f}; fold val "
       f"{env['foldval']:.4f}. Paired per-user bootstrap BOOT={BOOT} seed=0. Greedy forward selection is\n"
       f"prefix-consistent, so ONE T={TMAX} build serves budgets T={BUDGETS}.\n\n")
    md(PREREG)
    print("\n" + PREREG, flush=True)

    # ---- 0. REPRODUCE (anchor gate) ----
    print("==== 0. REPRODUCE (all-173 greedy s-item, eval all-173) ====", flush=True)
    cold_all = env["cold"]
    repro_sched = build_greedy_sub(env, item_pool, users, cold_all, TMAX)
    repro_pt = eval_static_peruser(env, repro_sched, users, cold_all, TMAX)
    repro_any12 = float(repro_pt[:, :12].mean())
    ok = abs(repro_any12 - ANCHOR) <= ANCHOR_TOL
    print(f"  all-173 s-item anytime@T12 = {repro_any12:.4f}  (anchor {ANCHOR} +/- {ANCHOR_TOL}) -> "
          f"{'MATCH' if ok else 'MISMATCH'}", flush=True)
    md("## 0. Reproduction (anchor gate)\n\n"
       f"All-173 greedy s-item, anytime NDCG@10 @T=12 = **{repro_any12:.4f}** vs anchor {ANCHOR} "
       f"+/- {ANCHOR_TOL} -> **{'MATCH' if ok else 'MISMATCH'}**.\n\n")
    RESULTS["reproduce"] = dict(anytime_T12=repro_any12, anchor=ANCHOR, match=bool(ok),
                                sched=[f"{CANDS[c]['kind']}:{CANDS[c]['key']}" for c in repro_sched[:12]])
    if not ok:
        md("**STOP: reproduction failed; contamination measurement not run.**\n\n")
        json.dump(RESULTS, open(OUT_JSON, "w"), indent=1, default=str)
        print("[STOP] reproduction mismatch.", flush=True)
        return

    # ---- build the split schedules ----
    # For each seed, split into halves A/B; build s-item and s-mixed greedy on A and on B (T=TMAX).
    splits = {}
    for seed in (0, 1):
        rng = np.random.default_rng(seed)
        perm = rng.permutation(n)
        half = n // 2  # 86
        A_idx = sorted(perm[:half].tolist())
        B_idx = sorted(perm[half:].tolist())
        A_users = [users[i] for i in A_idx]
        B_users = [users[i] for i in B_idx]
        coldA = cold_of(env, A_users); coldB = cold_of(env, B_users)
        scheds = {}
        for fam, pool in families.items():
            print(f"  [seed {seed}] greedy {fam} on A (n={len(A_users)}) ...", flush=True)
            sA = build_greedy_sub(env, pool, A_users, coldA, TMAX)
            print(f"  [seed {seed}] greedy {fam} on B (n={len(B_users)}) ...", flush=True)
            sB = build_greedy_sub(env, pool, B_users, coldB, TMAX)
            scheds[fam] = dict(A=sA, B=sB)
        splits[seed] = dict(A_idx=A_idx, B_idx=B_idx, A_users=A_users, B_users=B_users,
                            coldA=coldA, coldB=coldB, scheds=scheds)

    # ---- contamination estimates ----
    # 4 estimates: (seed0, eval B), (seed0, eval A), (seed1, eval B), (seed1, eval A)
    # In-sample = built on eval-half; Out-sample = built on the OTHER half.
    est = {}
    pooled = {fam: {T: dict(inv=[], outv=[]) for T in BUDGETS} for fam in families}
    for seed in (0, 1):
        S = splits[seed]
        for evalhalf in ("B", "A"):
            otherhalf = "A" if evalhalf == "B" else "B"
            ev_users = S[f"{evalhalf}_users"]; ev_cold = S[f"cold{evalhalf}"]
            for fam in families:
                sched_in = S["scheds"][fam][evalhalf]    # built on eval-half (IN-SAMPLE)
                sched_out = S["scheds"][fam][otherhalf]   # built on other half (OUT-OF-SAMPLE)
                pt_in = eval_static_peruser(env, sched_in, ev_users, ev_cold, TMAX)
                pt_out = eval_static_peruser(env, sched_out, ev_users, ev_cold, TMAX)
                for T in BUDGETS:
                    inv = list(pt_in[:, :T].mean(axis=1))
                    outv = list(pt_out[:, :T].mean(axis=1))
                    cb = RP.paired(inv, outv)
                    est.setdefault((seed, evalhalf, fam), {})[T] = dict(
                        in_mean=cb["b"] if False else float(np.mean(inv)),
                        out_mean=float(np.mean(outv)), contamination=cb)
                    pooled[fam][T]["inv"].extend(inv); pooled[fam][T]["outv"].extend(outv)
                key = f"seed{seed}_eval{evalhalf}_{fam}"
                cb12 = est[(seed, evalhalf, fam)][12]["contamination"]
                print(f"  [{key}] T12 in={np.mean(pt_in[:,:12].mean(axis=1)):.4f} "
                      f"out={np.mean(pt_out[:,:12].mean(axis=1)):.4f} "
                      f"contam={cb12['delta']:+.4f}[{cb12['ci'][0]:+.4f},{cb12['ci'][1]:+.4f}] "
                      f"(n={cb12['n']})", flush=True)

    # pooled CIs
    pooled_ci = {fam: {} for fam in families}
    for fam in families:
        for T in BUDGETS:
            cb = RP.paired(pooled[fam][T]["inv"], pooled[fam][T]["outv"])
            pooled_ci[fam][T] = cb

    # ---- divergence + composition (per seed, per family, at T=12) ----
    diverg = {}
    comps = {}
    for seed in (0, 1):
        S = splits[seed]
        for fam in families:
            sA = S["scheds"][fam]["A"]; sB = S["scheds"][fam]["B"]
            diverg[(seed, fam)] = divergence(env, sA, sB, T=12)
            comps[(seed, fam, "A")] = sched_composition(env, sA, T=12)
            comps[(seed, fam, "B")] = sched_composition(env, sB, T=12)
            dv = diverg[(seed, fam)]
            print(f"  [divergence seed{seed} {fam}] {12 - dv['shared']}/12 of A's picks not in B "
                  f"(shared {dv['shared']}/12; pos-diff {dv['pos_diff']}/12; sym-diff {dv['symmetric_diff']}/24)",
                  flush=True)

    # ---- assemble + verdict ----
    RESULTS["estimates"] = {
        f"seed{seed}_eval{eh}_{fam}": {str(T): {
            "in_mean": est[(seed, eh, fam)][T]["in_mean"],
            "out_mean": est[(seed, eh, fam)][T]["out_mean"],
            "contamination": est[(seed, eh, fam)][T]["contamination"]} for T in BUDGETS}
        for seed in (0, 1) for eh in ("B", "A") for fam in families}
    RESULTS["pooled"] = {fam: {str(T): pooled_ci[fam][T] for T in BUDGETS} for fam in families}
    RESULTS["divergence"] = {f"seed{seed}_{fam}": diverg[(seed, fam)]
                             for seed in (0, 1) for fam in families}
    RESULTS["composition"] = {f"seed{seed}_{fam}_{h}": comps[(seed, fam, h)]
                              for seed in (0, 1) for fam in families for h in ("A", "B")}

    # verdict on the PRIMARY family (s-item = the anchor's family), pooled, T=12
    prim = pooled_ci["s-item"][12]
    contam = prim["delta"]; lo, hi = prim["ci"]
    ci_excl_0 = bool(lo > 0 or hi < 0)
    verdict_contaminated = bool(ci_excl_0 and contam >= 0.005)
    RESULTS["verdict"] = dict(primary_family="s-item", budget=12, pooled_contamination=contam,
                              ci=[lo, hi], ci_excludes_0=ci_excl_0,
                              CONTAMINATED=verdict_contaminated)

    write_md(env, est, pooled_ci, diverg, comps, families)
    _interp_table(pooled_ci)
    RESULTS["wall_min"] = round((time.time() - t0) / 60, 2)
    json.dump(RESULTS, open(OUT_JSON, "w"), indent=1, default=str)

    print("\n==== VERDICT ====", flush=True)
    print(f"  pooled s-item T=12 contamination = {contam:+.4f}[{lo:+.4f},{hi:+.4f}]", flush=True)
    vl = ("CONTAMINATED -> all prior static-vs-router contrasts need split-constructed re-runs"
          if verdict_contaminated else
          "HYPOTHESIS DEAD -> statics do NOT meaningfully overfit their cohort; statics-optimal strengthened")
    print(f"  VERDICT: {vl}", flush=True)
    print(f"\n[done] wrote {OUT_MD} + {OUT_JSON} (wall {RESULTS['wall_min']}m)", flush=True)
    return RESULTS


def write_md(env, est, pooled_ci, diverg, comps, families):
    md("## Contamination estimates (per split x eval-half x family)\n\n")
    for fam in families:
        md(f"### Family: {fam}\n\n"
           "| estimate | budget | in-sample | out-of-sample | contamination [95% CI] | n |\n"
           "|---|---|--:|--:|---|--:|\n")
        for seed in (0, 1):
            for eh in ("B", "A"):
                for T in BUDGETS:
                    e = est[(seed, eh, fam)][T]; cb = e["contamination"]
                    md(f"| seed{seed} eval-{eh} | T={T} | {e['in_mean']:.4f} | {e['out_mean']:.4f} | "
                       f"{cb['delta']:+.4f}[{cb['ci'][0]:+.4f},{cb['ci'][1]:+.4f}] | {cb['n']} |\n")
        md("\n**Pooled (all 4 estimates' per-user deltas concatenated):**\n\n"
           "| budget | pooled contamination [95% CI] | n | CI excl 0? | >=0.005? |\n|---|---|--:|---|---|\n")
        for T in BUDGETS:
            cb = pooled_ci[fam][T]
            excl = (cb["ci"][0] > 0 or cb["ci"][1] < 0)
            md(f"| T={T} | {cb['delta']:+.4f}[{cb['ci'][0]:+.4f},{cb['ci'][1]:+.4f}] | {cb['n']} | "
               f"{'yes' if excl else 'no'} | {'yes' if abs(cb['delta'])>=0.005 else 'no'} |\n")
        md("\n")

    md("## Schedule divergence (A-built vs B-built, first 12 picks)\n\n"
       "`differ/12` = how many of A's 12 selected questions are NOT in B's schedule (= 12 - shared); "
       "`pos-diff/12` = positions holding a different question; `sym-diff/24` = total unshared across "
       "the union of both size-12 sets.\n\n"
       "| seed | family | differ/12 | pos-diff/12 | shared/12 | sym-diff/24 |\n|---|---|--:|--:|--:|--:|\n")
    for seed in (0, 1):
        for fam in families:
            dv = diverg[(seed, fam)]
            md(f"| {seed} | {fam} | {12 - dv['shared']}/12 | {dv['pos_diff']}/12 | {dv['shared']}/12 | "
               f"{dv['symmetric_diff']}/24 |\n")
    md("\n### Channel / tier composition of each schedule (first 12)\n\n"
       "| seed | family | half | channels | value-tiers (0=lowest..7=highest; -1=untiered) |\n"
       "|---|---|---|---|---|\n")
    for seed in (0, 1):
        for fam in families:
            for h in ("A", "B"):
                cp = comps[(seed, fam, h)]
                chs = ", ".join(f"{k}:{v}" for k, v in sorted(cp["channels"].items()))
                md(f"| {seed} | {fam} | {h} | {chs} | {cp['tiers']} |\n")
    md("\n")


def _interp_table(pooled_ci):
    """Annotate every prior contrast that used a cohort-fit static baseline with 'biased against router
    by ~X' using the measured pooled contamination X. NO re-runs; annotation math only."""
    # X per budget from the primary (s-item) family; use the closest-budget X for each contrast.
    def X(T):
        cb = pooled_ci["s-item"][T]
        return cb["delta"], cb["ci"]
    X8 = X(8); X12 = X(12); X24 = X(24)
    md("## Interpretation table -- prior static-vs-router contrasts (annotation math only, NO re-runs)\n\n"
       "Each contrast reported (router - cohort-fit static). If the static carries a test-fit bonus X,\n"
       "the FAIR (split-constructed) static would score ~X lower, so the router's DE-BIASED delta is\n"
       "~ (reported delta + X). Positive de-biased delta => the router would win / tie once the baseline\n"
       "is de-contaminated. X taken from the pooled s-item contamination at the nearest budget:\n\n"
       f"- X@T8  = {X8[0]:+.4f}[{X8[1][0]:+.4f},{X8[1][1]:+.4f}]\n"
       f"- X@T12 = {X12[0]:+.4f}[{X12[1][0]:+.4f},{X12[1][1]:+.4f}]\n"
       f"- X@T24 = {X24[0]:+.4f}[{X24[1][0]:+.4f},{X24[1][1]:+.4f}]\n\n"
       "| prior contrast | fold/arena | reported (router - static) | budget | de-biased ~= reported + X |\n"
       "|---|---|---|---|---|\n")
    rows = [
        ("E0 schedule B (adaptive vs greedy static B)", "additive-operator fold (VOID)", "tie (~0)", 12, X12[0]),
        ("i25_phase4_fair blind vs s3/s4", "I2.5 fold i25_fold_best (v1)", "-0.037..~tie (s3 best)", 24, X24[0]),
        ("i25_phase4 a6 vs s3", "I2.5 fold (v1)", "-0.0021[-0.0056,+0.0015]", 24, X24[0]),
        ("battery Stage-C r-blind vs s-best", "I2.5 fold i25_fold_best (v1)", "-0.0060[-0.0155,+0.0020]", 12, X12[0]),
        ("vivid-swap T3 r-vivid vs s-best", "I2.5 fold i25_fold_best (v1)", "+0.0000[0,0] (declined)", 12, X12[0]),
        ("repair P3 r-value-blind vs s-best", "v2 fold i25_fold_v2 (THIS)", "-0.0068[-0.0116,-0.0023]", 12, X12[0]),
        ("repair P3 r-value+k vs s-best", "v2 fold i25_fold_v2 (THIS)", "-0.0126[-0.0225,-0.0049]", 12, X12[0]),
    ]
    for name, fold, rep, T, x in rows:
        md(f"| {name} | {fold} | {rep} | T={T} | reported {rep.split('[')[0].strip()} + {x:+.4f} "
           f"= **{_debias(rep, x)}** |\n")
    md("\nNOTE: the annotation adds X to the reported deficit. Contrasts on the v1 (i25_fold_best) or\n"
       "the VOID additive-operator fold are annotated with the v2-measured X as an ORDER-OF-MAGNITUDE\n"
       "guide only -- the exact bonus is fold-specific and would need a per-fold split-construction to\n"
       "pin down. The two repair P3 rows are on the SAME fold as this measurement, so their de-biased\n"
       "values are directly valid (still DIRECTIONAL 173/300).\n\n")


def _debias(rep, x):
    try:
        base = float(rep.split("[")[0].strip().split("..")[0])
    except Exception:
        return "n/a (qualitative)"
    return f"{base + x:+.4f}"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--do", choices=["all"], default="all")
    ap.parse_args()
    run()
