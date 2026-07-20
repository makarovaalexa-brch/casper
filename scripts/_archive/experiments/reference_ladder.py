"""reference_ladder.py -- THE CLEAN REFERENCE LADDER.

ONE table, everything on the SAME instrument (fold-v2 .cache/i25_fold_v2_best.pt) and the SAME 173
users (answerer-v1 WORKING grid), so mixed-version numbers stop circulating. NO LLM API calls.

Rungs (NDCG@10 AND @50, all 173 users; paired per-user bootstrap CIs where deltas are shown):
  1. COLD                      : z=0, no answers.
  2. learned ITEM static       : greedy s-item (built on all 173), full per-turn endpoint curve t=1..24
                                 + anytime@{8,12,24}; also the FAIR cross-fit (out-of-fold, seed-0 split)
                                 anytime numbers + the measured contamination gap. Concept static curve
                                 for contrast. Monotonicity of the endpoint curve verified explicitly.
  3. FULL-PROFILE via fold-v2  : all known-half rated items as data-value tokens through the v2 fold.
  4. FULL-PROFILE via native   : liked known-half items through the RecVAE native encoder (its own
                                 interface) = the instrument ceiling.
  5. orientation               : static t=24 endpoint as % of (3) and of (4).
Plus: BOTH learned static schedules decoded to human-readable names -- items via D['title'], concepts
via the arena's OWN concept-key map (tagId -> 'tag' field on the grid Q cells; the exact path the
harness used), NOT the tag_questions.json order.

Reuses scripts/repair_probes.py (setup/tok/paired), static_contamination.build_greedy_sub, i25_fold_v2,
i25_phase4_fair.ndcg_batch (kk-parametrized), i25_lib.

Run:  python scripts/reference_ladder.py
"""
import os, sys, json, time, collections
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import warnings
warnings.filterwarnings("ignore")
import repair_probes as RP
import static_contamination as SC
import i25_fold_v2 as V2
import i25_phase4_fair as FA
import i25_lib as L

OUT_MD = "experiments/REFERENCE_LADDER.md"
OUT_JSON = "experiments/reference_ladder.json"
TMAX = 24
BUDGETS = [8, 12, 24]
BOOT = 5000
KKS = [10, 50]


def fold_chunked(env, tl, nl, chunk=3000):
    FR, model = env["FR"], env["model"]
    Z = np.empty((len(tl), FR.W.shape[1]))
    for s in range(0, len(tl), chunk):
        e = min(s + chunk, len(tl))
        Z[s:e] = V2.fold_batch_v2(FR, model, tl[s:e], nl[s:e])
    return Z


def sched_curve(env, sched, users=None, cold=None):
    """Per-turn ENDPOINT NDCG@10 and @50 over users, t=1..TMAX, for a fixed static schedule.
    Refusal (unanswerable cell for a user) = no-op turn (cold fallback). Returns (pt10, pt50) each
    (n, TMAX). Identical accumulation to SC.eval_static_peruser but dual-metric."""
    FR, CANDS = env["FR"], env["CANDS"]
    users = users if users is not None else env["users"]
    n = len(users)
    c10 = FA.cold_ndcg(FR, users, 10) if cold is None else cold[10]
    c50 = FA.cold_ndcg(FR, users, 50) if cold is None else cold[50]
    pt10 = np.empty((n, TMAX)); pt50 = np.empty((n, TMAX))
    for t in range(TMAX):
        col10 = c10.copy(); col50 = c50.copy()
        tl, nl, idx = [], [], []
        for i, rec in enumerate(users):
            cids = [c for c in sched[:t + 1] if (c is not None and rec["ans_arr"][c])]
            if not cids:
                continue
            tl.append([RP.tok_of(CANDS, rec, c) for c in cids])
            nl.append([rec["nat_arr"][c] for c in cids if rec["nat_arr"][c] is not None])
            idx.append(i)
        if tl:
            Z = fold_chunked(env, tl, nl)
            recs = [users[i] for i in idx]
            col10[idx] = FA.ndcg_batch(FR, Z, recs, 10)
            col50[idx] = FA.ndcg_batch(FR, Z, recs, 50)
        pt10[:, t] = col10; pt50[:, t] = col50
    return pt10, pt50


def monotone_report(curve_mean):
    """max per-step decline of a cohort-mean endpoint curve (t=1..TMAX). Returns (min_step, at_t)."""
    steps = np.diff(curve_mean)
    j = int(np.argmin(steps))
    return float(steps[j]), j + 2  # step j is turn (j+1)->(j+2)


def full_profile(env):
    """Per-user full-profile NDCG@10/@50 through (a) v2 fold (all rated known items as data tokens)
    and (b) native RecVAE encoder (liked known items only, its own interface)."""
    FR, model, users = env["FR"], env["model"], env["users"]
    FID = V2.FID
    v2_tl, v2_nl, nat_liked = [], [], []
    for rec in users:
        known, cm = rec["known"], rec["cmean"]
        toks = [(0, FID["data"], FR.Wn[j].numpy().astype(np.float32), float(known[j] - cm)) for j in known]
        liked = [j for j in known if known[j] >= 4]
        v2_tl.append(toks); v2_nl.append(liked); nat_liked.append(liked)
    Z2 = fold_chunked(env, v2_tl, v2_nl)
    Zn = np.array([FR.native_fold_np(lk) for lk in nat_liked])
    out = {}
    for kk in KKS:
        out[kk] = dict(v2=FA.ndcg_batch(FR, Z2, users, kk), native=FA.ndcg_batch(FR, Zn, users, kk))
    return out


def concept_map(env):
    """tagId -> human name via the arena's OWN mapping: the 'tag' field on the grid concept Q cells
    (the exact path the harness used to build the concept pool). NOT tag_questions.json order."""
    m = {}
    for us, rec in env["grid"].items():
        for _, c in rec.get("Q", {}).items():
            if c.get("channel") == "concept" and "tagId" in c and "tag" in c:
                m[int(c["tagId"])] = c["tag"]
    return m


def decode_sched(env, sched, cmap):
    D, CANDS = env["D"], env["CANDS"]
    out = []
    for c in sched:
        m = CANDS[c]
        if m["kind"] == "item":
            j = int(m["key"])
            try:
                nm = str(D["title"][j])
            except Exception:
                nm = f"item#{j}"
            out.append(dict(kind="item", key=j, name=nm, cov=round(m["cov"], 3)))
        elif m["kind"] == "concept":
            tid = int(m["key"])
            out.append(dict(kind="concept", key=tid, name=cmap.get(tid, f"<tagId {tid} UNMAPPED>"),
                            cov=round(m["cov"], 3)))
        else:
            out.append(dict(kind="attr", key=str(m["key"]), name=str(m["key"]), cov=round(m["cov"], 3)))
    return out


def main():
    t0 = time.time()
    env = RP.setup()
    FR, users, CANDS = env["FR"], env["users"], env["CANDS"]
    n = len(users)
    item_pool = [m["cid"] for m in CANDS if m["kind"] == "item"]
    conc_pool = [m["cid"] for m in CANDS if m["kind"] == "concept"]
    print(f"[ladder] {n} users; {len(CANDS)} cands ({len(item_pool)}i/{len(conc_pool)}c); "
          f"fold_val {env['foldval']:.4f}", flush=True)

    R = dict(banner="CLEAN REFERENCE LADDER -- fold-v2 (.cache/i25_fold_v2_best.pt), 173 users, "
                    "answerer-v1 WORKING grid, NO LLM calls. DIRECTIONAL 173/300 (grid unfrozen).",
             n_users=n, n_cands=len(CANDS), fold_val=float(env["foldval"]))

    # ---- 1. COLD ----
    cold = {kk: FA.cold_ndcg(FR, users, kk) for kk in KKS}
    R["cold"] = {kk: float(cold[kk].mean()) for kk in KKS}
    print(f"[1] COLD  @10 {R['cold'][10]:.4f}  @50 {R['cold'][50]:.4f}", flush=True)

    # ---- 2. learned ITEM static (all-173 in-sample = the decoded canonical schedule) ----
    print("[2] building greedy s-item / s-concept on all 173 (T=24) ...", flush=True)
    s_item = SC.build_greedy_sub(env, item_pool, users, cold[10], TMAX)
    s_conc = SC.build_greedy_sub(env, conc_pool, users, cold[10], TMAX)
    it10, it50 = sched_curve(env, s_item, users, cold)
    cc10, cc50 = sched_curve(env, s_conc, users, cold)

    def summ(pt10, pt50):
        d = {}
        for kk, pt in ((10, pt10), (50, pt50)):
            cm = pt.mean(axis=0)
            step, at_t = monotone_report(cm)
            d[kk] = dict(
                curve=[round(float(x), 4) for x in cm],
                endpoint={str(T): float(pt[:, T - 1].mean()) for T in BUDGETS},
                anytime={str(T): float(pt[:, :T].mean()) for T in BUDGETS},
                max_decline=step, max_decline_at_turn=at_t)
        return d

    R["item_static_insample"] = summ(it10, it50)
    R["concept_static_insample"] = summ(cc10, cc50)
    print(f"    s-item   @10 endpoint t24 {R['item_static_insample'][10]['endpoint']['24']:.4f} | "
          f"anytime@12 {R['item_static_insample'][10]['anytime']['12']:.4f} | "
          f"max-decline {R['item_static_insample'][10]['max_decline']:+.4f}", flush=True)
    print(f"    s-concept@10 endpoint t24 {R['concept_static_insample'][10]['endpoint']['24']:.4f} | "
          f"anytime@12 {R['concept_static_insample'][10]['anytime']['12']:.4f} | "
          f"max-decline {R['concept_static_insample'][10]['max_decline']:+.4f}", flush=True)

    # anchor cross-check (all-173 s-item anytime@10 T12 should be ~0.2251)
    R["anchor_check"] = dict(anytime10_T12=R["item_static_insample"][10]["anytime"]["12"],
                             anchor=0.2251, tol=0.002,
                             match=abs(R["item_static_insample"][10]["anytime"]["12"] - 0.2251) <= 0.002)
    print(f"    anchor: s-item anytime@10 T12 {R['anchor_check']['anytime10_T12']:.4f} vs 0.2251 -> "
          f"{'MATCH' if R['anchor_check']['match'] else 'MISMATCH'}", flush=True)

    # ---- 2b. FAIR cross-fit item static (out-of-fold, seed-0 split) ----
    print("[2b] fair cross-fit s-item (out-of-fold, seed 0) ...", flush=True)
    rng = np.random.default_rng(0)
    perm = rng.permutation(n); half = n // 2
    A_idx = sorted(perm[:half].tolist()); B_idx = sorted(perm[half:].tolist())
    A_users = [users[i] for i in A_idx]; B_users = [users[i] for i in B_idx]
    coldA = {kk: FA.cold_ndcg(FR, A_users, kk) for kk in KKS}
    coldB = {kk: FA.cold_ndcg(FR, B_users, kk) for kk in KKS}
    sA = SC.build_greedy_sub(env, item_pool, A_users, coldA[10], TMAX)
    sB = SC.build_greedy_sub(env, item_pool, B_users, coldB[10], TMAX)
    # eval each half with the OTHER half's schedule (out-of-fold)
    ptB10, ptB50 = sched_curve(env, sA, B_users, coldB)   # B eval'd by A-built
    ptA10, ptA50 = sched_curve(env, sB, A_users, coldA)   # A eval'd by B-built
    # concatenate per-user out-of-fold anytime vectors
    fair = {}
    for kk, ptA, ptB in ((10, ptA10, ptB10), (50, ptA50, ptB50)):
        fair[kk] = {}
        for T in BUDGETS:
            vec = list(ptA[:, :T].mean(axis=1)) + list(ptB[:, :T].mean(axis=1))
            fair[kk][str(T)] = float(np.mean(vec))
    R["item_static_faircrossfit"] = {str(kk): fair[kk] for kk in KKS}
    # contamination = in-sample - fair (anytime@10) per budget (pooled per-user paired where alignable)
    # in-sample per-user anytime on the SAME users, aligned to A_users+B_users order:
    insA10 = it10[A_idx, :]; insB10 = it10[B_idx, :]
    contam = {}
    for T in BUDGETS:
        ins_vec = list(insA10[:, :T].mean(axis=1)) + list(insB10[:, :T].mean(axis=1))
        fair_vec = list(ptA10[:, :T].mean(axis=1)) + list(ptB10[:, :T].mean(axis=1))
        contam[str(T)] = RP.paired(ins_vec, fair_vec)
    R["contamination_insample_minus_fair@10"] = contam
    print(f"     fair cross-fit @10 anytime@12 {fair[10]['12']:.4f} | contamination(in-fair)@12 "
          f"{contam['12']['delta']:+.4f}[{contam['12']['ci'][0]:+.4f},{contam['12']['ci'][1]:+.4f}]",
          flush=True)

    # ---- 3 + 4. full-profile fold-v2 vs native ----
    print("[3/4] full-profile fold-v2 vs native RecVAE ...", flush=True)
    fp = full_profile(env)
    R["full_profile"] = {}
    for kk in KKS:
        v2 = fp[kk]["v2"]; nat = fp[kk]["native"]
        cb = RP.paired(list(v2), list(nat))
        R["full_profile"][str(kk)] = dict(v2_fold=float(np.mean(v2)), native=float(np.mean(nat)),
                                           v2_minus_native=cb)
        print(f"    @{kk}: v2-fold {np.mean(v2):.4f}  native {np.mean(nat):.4f}  "
              f"delta {cb['delta']:+.4f}[{cb['ci'][0]:+.4f},{cb['ci'][1]:+.4f}]", flush=True)

    # ---- 5. orientation: static t=24 endpoint as % of full-profile ceilings ----
    R["orientation"] = {}
    for kk in KKS:
        st24 = R["item_static_insample"][kk]["endpoint"]["24"]
        v2fp = R["full_profile"][str(kk)]["v2_fold"]
        natfp = R["full_profile"][str(kk)]["native"]
        R["orientation"][str(kk)] = dict(
            static_t24=st24, pct_of_v2fold=round(100 * st24 / v2fp, 1) if v2fp else None,
            pct_of_native=round(100 * st24 / natfp, 1) if natfp else None)
        print(f"    @{kk}: static t24 {st24:.4f} = {R['orientation'][str(kk)]['pct_of_v2fold']}% of v2-fp, "
              f"{R['orientation'][str(kk)]['pct_of_native']}% of native", flush=True)

    # ---- decode both schedules ----
    cmap = concept_map(env)
    R["decoded_item_schedule"] = decode_sched(env, s_item, cmap)
    R["decoded_concept_schedule"] = decode_sched(env, s_conc, cmap)
    n_unmapped = sum(1 for e in R["decoded_concept_schedule"] if "UNMAPPED" in e["name"])
    R["concept_map_size"] = len(cmap); R["concept_unmapped_in_sched"] = n_unmapped
    print(f"[decode] concept map size {len(cmap)}; unmapped in schedule {n_unmapped}", flush=True)

    R["wall_min"] = round((time.time() - t0) / 60, 2)
    write_md(R, s_item, s_conc)
    json.dump(R, open(OUT_JSON, "w"), indent=1, default=str)
    print(f"\n[done] wrote {OUT_MD} + {OUT_JSON} (wall {R['wall_min']}m)", flush=True)
    return R


def _fmt_cb(cb):
    return f"{cb['delta']:+.4f}[{cb['ci'][0]:+.4f},{cb['ci'][1]:+.4f}]"


def write_md(R, s_item, s_conc):
    L_ = []
    w = L_.append
    w("# THE CLEAN REFERENCE LADDER (fold-v2; 173 users; NO LLM)\n\n")
    w("> **DIRECTIONAL 173/300** -- answerer-v1 WORKING grid (`.cache/instrument2/"
      "answerer_v1_grid173_WORKING.json`), grid NOT frozen. Re-run on the frozen 300-user grid before "
      "citation. Instrument = fold-v2 `.cache/i25_fold_v2_best.pt` (val {:.4f}). Script "
      "`scripts/reference_ladder.py`; NO LLM API calls.\n\n".format(R["fold_val"]))
    w(f"{R['n_users']} users; {R['n_cands']} candidates. Greedy statics built to maximize cohort-mean "
      f"NDCG@10 (canonical objective), reported at BOTH @10 and @50. Paired per-user bootstrap "
      f"BOOT={BOOT}, seed 0. Everything below is on the SAME instrument and the SAME 173 users.\n\n")

    # ---- the one table ----
    w("## The ladder (NDCG@10 / NDCG@50)\n\n")
    w("| # | rung | @10 | @50 | notes |\n|---|---|--:|--:|---|\n")
    w(f"| 1 | COLD (z=0, no answers) | {R['cold'][10]:.4f} | {R['cold'][50]:.4f} | floor |\n")
    isi = R["item_static_insample"]; csi = R["concept_static_insample"]
    w(f"| 2 | learned ITEM static -- anytime@12 (in-sample all-173) | {isi[10]['anytime']['12']:.4f} | "
      f"{isi[50]['anytime']['12']:.4f} | anchor@10 {R['anchor_check']['anytime10_T12']:.4f} vs 0.2251 "
      f"({'MATCH' if R['anchor_check']['match'] else 'MISMATCH'}); carries test-fit bonus (see 2b) |\n")
    w(f"| 2 | learned ITEM static -- endpoint t=24 (in-sample) | {isi[10]['endpoint']['24']:.4f} | "
      f"{isi[50]['endpoint']['24']:.4f} | endpoint after 24 turns |\n")
    fc = R["item_static_faircrossfit"]
    w(f"| 2b | learned ITEM static -- anytime@12 (FAIR cross-fit, out-of-fold) | {fc['10']['12']:.4f} | "
      f"{fc['50']['12']:.4f} | de-contaminated; the honest static number |\n")
    w(f"| 2c | concept static -- anytime@12 (in-sample, contrast) | {csi[10]['anytime']['12']:.4f} | "
      f"{csi[50]['anytime']['12']:.4f} | for contrast only |\n")
    fp = R["full_profile"]
    w(f"| 3 | FULL-PROFILE via fold-v2 (all rated known items) | {fp['10']['v2_fold']:.4f} | "
      f"{fp['50']['v2_fold']:.4f} | noise-robust fold, full clean profile (out-of-regime) |\n")
    w(f"| 4 | FULL-PROFILE via NATIVE RecVAE (liked items) | {fp['10']['native']:.4f} | "
      f"{fp['50']['native']:.4f} | **instrument ceiling** (its own interface) |\n")
    o = R["orientation"]
    w(f"| 5 | static t=24 as % of (3) / (4) | {o['10']['pct_of_v2fold']}% / {o['10']['pct_of_native']}% | "
      f"{o['50']['pct_of_v2fold']}% / {o['50']['pct_of_native']}% | orientation |\n\n")

    # ---- full per-turn curves ----
    w("## Learned ITEM static -- full per-turn ENDPOINT curve (in-sample all-173)\n\n")
    w("| metric | " + " | ".join(f"t{t}" for t in range(1, TMAX + 1)) + " |\n")
    w("|---|" + "|".join("--:" for _ in range(TMAX)) + "|\n")
    for kk in KKS:
        w(f"| @{kk} | " + " | ".join(f"{x:.4f}" for x in isi[kk]["curve"]) + " |\n")
    w("\n**Endpoint** @10: " + ", ".join(f"t{T}={isi[10]['endpoint'][str(T)]:.4f}" for T in BUDGETS) +
      "  |  @50: " + ", ".join(f"t{T}={isi[50]['endpoint'][str(T)]:.4f}" for T in BUDGETS) + "\n\n")
    w("**Anytime** @10: " + ", ".join(f"T{T}={isi[10]['anytime'][str(T)]:.4f}" for T in BUDGETS) +
      "  |  @50: " + ", ".join(f"T{T}={isi[50]['anytime'][str(T)]:.4f}" for T in BUDGETS) + "\n\n")

    w("## Concept static -- full per-turn ENDPOINT curve (in-sample, contrast)\n\n")
    w("| metric | " + " | ".join(f"t{t}" for t in range(1, TMAX + 1)) + " |\n")
    w("|---|" + "|".join("--:" for _ in range(TMAX)) + "|\n")
    for kk in KKS:
        w(f"| @{kk} | " + " | ".join(f"{x:.4f}" for x in csi[kk]["curve"]) + " |\n")
    w("\n")

    # ---- monotonicity ----
    w("## Monotonicity verdict (cohort-mean endpoint curve, max per-step decline)\n\n")
    w("| schedule | metric | max per-step decline | at turn | verdict |\n|---|---|--:|--:|---|\n")
    for nm, s in (("item", isi), ("concept", csi)):
        for kk in KKS:
            md_ = s[kk]["max_decline"]; att = s[kk]["max_decline_at_turn"]
            verd = "MONOTONE (no decline)" if md_ >= -1e-9 else (
                "near-monotone (decline < 0.002)" if md_ >= -0.002 else "NON-monotone (decline >= 0.002)")
            w(f"| {nm} | @{kk} | {md_:+.4f} | t{att - 1}->t{att} | {verd} |\n")
    w("\n")

    # ---- contamination ----
    w("## Static test-fit contamination (in-sample minus fair cross-fit, anytime NDCG@10)\n\n")
    w("| budget | in-sample - fair [95% CI] | n |\n|---|---|--:|\n")
    for T in BUDGETS:
        cb = R["contamination_insample_minus_fair@10"][str(T)]
        w(f"| T={T} | {_fmt_cb(cb)} | {cb['n']} |\n")
    w("\n> Positive = the all-173 greedy static gains from being fit to its own evaluation cohort. "
      "The FAIR (out-of-fold) numbers in rung 2b are the de-contaminated reference; the in-sample "
      "rung-2 curve is kept because it is the single canonical decodable schedule (and reproduces the "
      "0.2251 anchor). Consistent with experiments/STATIC_CONTAMINATION.md.\n\n")

    # ---- full-profile detail ----
    w("## Full-profile: fold-v2 vs native RecVAE\n\n")
    w("| metric | v2 fold | native RecVAE | v2 - native [95% CI] |\n|---|--:|--:|---|\n")
    for kk in KKS:
        d = R["full_profile"][str(kk)]
        w(f"| @{kk} | {d['v2_fold']:.4f} | {d['native']:.4f} | {_fmt_cb(d['v2_minus_native'])} |\n")
    w("\n> The v2 fold is trained for PARTIAL, NOISY interviews (k<=16, mixed fidelity, sigma=0.70); a "
      "full CLEAN profile of all rated items is out-of-regime, so the noise-robust fold is deliberately "
      "conservative and gives up NDCG to the native encoder on clean full profiles. Anyone needing a "
      "full-profile score uses the native RecVAE encoder (rung 4 = the instrument ceiling).\n\n")

    # ---- decoded schedules ----
    w("## Decoded schedule -- learned ITEM static (first 24, greedy order)\n\n")
    w("Item names via `D['title'][key]`.\n\n")
    w("| turn | tagId/itemId | name | pop-cov |\n|---|--:|---|--:|\n")
    for t, e in enumerate(R["decoded_item_schedule"], 1):
        w(f"| {t} | {e['key']} | {e['name']} | {e['cov']} |\n")
    w("\n## Decoded schedule -- learned CONCEPT static (first 24, greedy order)\n\n")
    w("Concept names via the arena's OWN map (tagId -> `tag` field on the grid Q cells). "
      f"Map size {R['concept_map_size']}; unmapped in schedule {R['concept_unmapped_in_sched']}.\n\n")
    w("| turn | tagId | tag name | pop-cov |\n|---|--:|---|--:|\n")
    for t, e in enumerate(R["decoded_concept_schedule"], 1):
        w(f"| {t} | {e['key']} | {e['name']} | {e['cov']} |\n")
    w("\n")

    os.makedirs("experiments", exist_ok=True)
    open(OUT_MD, "w", encoding="utf-8").write("".join(L_))


if __name__ == "__main__":
    main()
