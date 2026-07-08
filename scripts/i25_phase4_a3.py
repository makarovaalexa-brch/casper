"""i25_phase4_a3.py -- A3 "co-known item prober": ONE new adaptive arm on the FAIR Phase-4 harness.

Imports and REUSES scripts/i25_phase4_fair.py machinery UNMODIFIED (build_universe, single_q_values,
build_greedy, eval_fair, plan_sched, tok_of, cold_ndcg, anytime/endpoint, P4.boot). NO LLM calls,
local compute only, deterministic.

WHAT A3 IS (BLIND, deployable):
  Question pool = the 160-item top-coverage ladder bank (coverage>=3) that s3 draws from -- item probes
  only. Policy: turn 1 = the globally best s3 item (s3[0]). After each turn:
    - if the last probe was ANSWERED  -> next probe = highest-value UNASKED bank item in the ANSWERED
      set's co-known neighborhood: score(j) = coverage_prior(j) x mean_affinity(j | answered set),
      affinity = population item-item co-rating cosine. Tie-break by global coverage.
    - if REFUSED -> fall back to the next unasked item in the global coverage order.
  Co-known statistics: item-item co-rating cosine built from ML-25M TRAINING users (trU) with ALL 298
  study users' rows dropped (zero leak; study users are eval-split, disjoint from trU -- asserted).
  Restricted to the 160-item bank (160x160).

  Variants:
    a3-blind  : as above (deployable).
    a3-decay  : same, but a sponsoring anchor's neighborhood weight decays 1 -> 0.5 -> 0 as its
                neighbors are refused (2 refusals in X's region stop probing X's region).
    a3-table  : same policy but the candidate set is restricted to TRUE-answerable items (user rated it)
                -> 100% hit rate = the hit-rate CEILING for this policy class. PRIVILEGED (labelled).

PRE-REGISTERED HYPOTHESIS (printed at runtime): a3-blind raises the mean-answered-turns (hit rate)
above s3's ~14% (3.4/24) by exploiting answered-item neighborhoods. If the hit rate does NOT rise,
adaptive conditioning found no purchase and Branch B stands unqualified.

Run:  python scripts/i25_phase4_a3.py
"""
import os, sys, json, time, collections
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import i25_phase4_fair as FA     # UNMODIFIED reuse
P4 = FA.P4
L = FA.L

TMAX = FA.TMAX                   # 24
BUDGETS = FA.BUDGETS             # (8,16,24)
DECAY_STEP = 0.5                 # anchor weight 1 -> 0.5 -> 0.0 at 2 refusals in its region
OUT_JSON = "experiments/I25_phase4_a3.json"
OUT_MD = "experiments/I25_PHASE4_FAIR.md"     # append a section
META = L.G.META


# ================================================================= population co-known matrix
def build_cooc(bank_j, study_uids):
    """160x160 item-item co-rating COSINE from ML-25M training users (trU), with ALL study users'
    rows removed (zero leak). bank_j = list of dense item ids (the probe bank). Returns (C, self_cnt).
    C[a,b] = cooc(a,b)/sqrt(cooc(a,a)*cooc(b,b)); population = users who rated BOTH in their FULL profile."""
    d = np.load(META)
    uu, ii = d["uu"], d["ii"]
    trU = set(d["trU"].astype(np.int64).tolist())
    allowed = np.array(sorted(trU - set(int(u) for u in study_uids)), dtype=np.int64)
    leak = len(set(int(u) for u in study_uids) & trU)
    assert leak == 0, f"LEAK: {leak} study users are in trU (must be 0)"

    col = {int(j): k for k, j in enumerate(bank_j)}
    bank_arr = np.array(sorted(bank_j), dtype=np.int64)
    m_item = np.isin(ii, bank_arr)                      # rows touching the bank
    m_user = np.isin(uu, allowed)
    sel = m_item & m_user
    us = uu[sel].astype(np.int64); it = ii[sel].astype(np.int64)
    cols = np.array([col[int(j)] for j in it], dtype=np.int64)
    # compress user ids -> contiguous rows
    uniq, rows = np.unique(us, return_inverse=True)
    nU = len(uniq); nB = len(bank_j)
    M = np.zeros((nU, nB), dtype=np.float32)
    M[rows, cols] = 1.0                                 # binary rated-or-not (co-rating count)
    cooc = (M.T @ M)                                    # (nB,nB) integer co-rating counts (float)
    self_cnt = np.diag(cooc).copy()
    denom = np.sqrt(np.outer(self_cnt, self_cnt)) + 1e-9
    C = cooc / denom
    np.fill_diagonal(C, 0.0)                            # never score an item against itself
    return C.astype(np.float64), self_cnt, nU


# ================================================================= a3 policy plan builder
def make_a3_plan(CANDS, item_cids, turn1_cid, global_order, colof, C, pop_rate, mode):
    """mode in {'blind','decay','table'}. Returns plan_fn(rec) -> list of (token,native|None) over TMAX."""
    table = (mode == "table"); decay = (mode == "decay")
    item_set = set(item_cids)

    def first_in_order(used, avail_set):
        for c in global_order:
            if c not in used and c in avail_set:
                return c
        return None

    def pick_neighbor(avail, answered, refus, used):
        """avail = list of candidate cids; answered = list of answered cids (anchors)."""
        best_key, bc, bsp = None, None, None
        acols = [colof[a] for a in answered]
        for c in avail:
            cc = colof[c]
            num = den = 0.0
            for a, ac in zip(answered, acols):
                w = max(0.0, 1.0 - DECAY_STEP * refus[a]) if decay else 1.0
                num += w * C[ac, cc]; den += w
            aff = (num / den) if den > 0 else 0.0
            score = pop_rate[c] * aff
            key = (score, pop_rate[c], -c)              # tie-break: coverage, then stable cid
            if best_key is None or key > best_key:
                best_key, bc = key, c
        if bc is not None:                              # sponsoring anchor = raw-cos argmax
            bcc = colof[bc]
            bsp = max(answered, key=lambda a: (C[colof[a], bcc], -a))
        return bc, bsp

    def plan_fn(rec):
        ans = rec["ans_arr"]
        used = set(); plan = []; answered = []; sponsor = {}
        refus = collections.defaultdict(int); last_ans = False
        for t in range(TMAX):
            if table:
                avail = [c for c in item_cids if c not in used and ans[c]]
                avail_set = set(avail)
            else:
                avail = [c for c in item_cids if c not in used]
                avail_set = item_set - used
            if not avail:
                break
            if t == 0:
                cid = turn1_cid if (not table and turn1_cid not in used) else None
                if cid is None or cid not in avail_set:
                    cid = first_in_order(used, avail_set)   # table (or safety): best answerable in order
            elif last_ans and answered:
                cid, sp = pick_neighbor(avail, answered, refus, used)
                if cid is not None:
                    sponsor[cid] = sp
            else:
                cid = first_in_order(used, avail_set)       # refusal -> global coverage order
            if cid is None:
                break
            used.add(cid)
            if ans[cid]:
                plan.append((FA.tok_of(CANDS, rec, cid), rec["nat_arr"][cid]))
                answered.append(cid); last_ans = True
            else:
                plan.append((None, None)); last_ans = False
                if decay and cid in sponsor and sponsor[cid] is not None:
                    refus[sponsor[cid]] += 1
        return plan
    return plan_fn


# ================================================================= reporting helpers
def contrast(pt_a, pt_b):
    """paired bootstrap of anytime(a)-anytime(b) at each budget (a beats b iff ci[0]>0)."""
    return {T: P4.boot(FA.anytime(pt_a, T), FA.anytime(pt_b, T)) for T in BUDGETS}

def first_sep(pt_a, pt_b):
    """first turn t where the per-turn (belief(t)) paired CI excludes 0; returns (turn, sign)."""
    for t in range(TMAX):
        b = P4.boot(pt_a[:, t], pt_b[:, t])
        if b["ci"][0] > 0:
            return t + 1, "+"
        if b["ci"][1] < 0:
            return t + 1, "-"
    return None, "0"


def main():
    t0 = time.time()
    print("[a3] loading data + frozen RecVAE + trained fold ...", flush=True)
    D = L.G.load_data(); FR = L.Frozen(D)
    model = L.Fold(); import torch
    blob = torch.load(P4.CKPT_BEST, map_location="cpu")
    model.load_state_dict(blob["model"]); model.eval()
    P4.T = TMAX
    users, _ = P4.assemble(D, FR)
    n = len(users)
    print(f"[a3] {n} users; building ladder universe (reuse FA.build_universe) ...", flush=True)
    CANDS, meta = FA.build_universe(D, FR, users)
    cold = FA.cold_ndcg(FR, users, 10)

    # ---- reproduce s1 / s3 with the EXISTING harness before adding anything ----
    concept_cids = [m["cid"] for m in CANDS if m["kind"] == "concept"]
    item_cids = [m["cid"] for m in CANDS if m["kind"] == "item"]
    pop_rate = {m["cid"]: m["pop_rate"] for m in CANDS if m["kind"] == "item"}
    item_by_cov = sorted(item_cids, key=lambda c: -CANDS[c]["pop_rate"])
    s3_pool = item_by_cov[:FA.ITEM_POOL_S3]
    print("[a3] single-question values (all candidates) ...", flush=True)
    v1 = FA.single_q_values(FR, model, users, cold, CANDS, [m["cid"] for m in CANDS])
    print("[a3] greedy s1 (concepts) ...", flush=True)
    s1 = FA.build_greedy(FR, model, users, cold, CANDS, concept_cids, TMAX, v1=v1, tag="s1")
    print("[a3] greedy s3 (popular items) ...", flush=True)
    s3 = FA.build_greedy(FR, model, users, cold, CANDS, s3_pool, TMAX, v1=v1, tag="s3")

    pt_s1, ans_s1 = FA.eval_fair(FR, model, users, lambda rec: FA.plan_sched(CANDS, rec, s1), cold)
    pt_s3, ans_s3 = FA.eval_fair(FR, model, users, lambda rec: FA.plan_sched(CANDS, rec, s3), cold)
    s1_any24 = float(FA.anytime(pt_s1, 24).mean())
    s3_any24 = float(FA.anytime(pt_s3, 24).mean())
    print(f"\n[a3] REPRODUCTION: s1 any@24 = {s1_any24:.4f} (target 0.2103) | "
          f"s3 any@24 = {s3_any24:.4f} (target 0.2786)", flush=True)
    ok = (abs(s1_any24 - 0.2103) < 0.0015) and (abs(s3_any24 - 0.2786) < 0.0015)
    if not ok:
        print("[a3] STOP: s1/s3 did NOT reproduce within tolerance -- NOT adding a3. Diagnose first.", flush=True)
        return
    print("[a3] reproduction OK -- proceeding to build a3.\n", flush=True)

    # ---- pre-registered hypothesis print ----
    s3_hit = float(ans_s3.mean())
    print("=" * 78, flush=True)
    print("PRE-REGISTERED HYPOTHESIS: a3-blind raises mean-answered-turns (hit rate) above s3's", flush=True)
    print(f"  {s3_hit:.1f}/24 (~{100*s3_hit/24:.0f}%) by exploiting answered-item co-known neighborhoods.", flush=True)
    print("  If hit rate does NOT rise, adaptive conditioning found no purchase -> Branch B unqualified.", flush=True)
    print("=" * 78 + "\n", flush=True)

    # ---- co-known matrix (population; zero leak) ----
    bank_j = [int(CANDS[c]["key"].split(":")[1]) for c in item_cids]
    study_uids = [rec["u"] for rec in users]
    print("[a3] building 160x160 population co-known cosine (trU minus study users) ...", flush=True)
    C, self_cnt, nU_pop = build_cooc(bank_j, study_uids)
    colof = {c: k for k, c in enumerate(item_cids)}
    turn1_cid = s3[0]
    global_order = sorted(item_cids, key=lambda c: (-CANDS[c]["pop_rate"], c))
    print(f"[a3] cooc from {nU_pop} population users; bank={len(item_cids)} items; "
          f"turn1 = {CANDS[turn1_cid]['key']} (globally best s3 item)\n", flush=True)

    # ---- evaluate a3 arms ----
    pt, ansn = {}, {}
    for mode in ("blind", "decay", "table"):
        print(f"[a3] eval a3-{mode} ...", flush=True)
        pf = make_a3_plan(CANDS, item_cids, turn1_cid, global_order, colof, C, pop_rate, mode)
        pt[mode], ansn[mode] = FA.eval_fair(FR, model, users, pf, cold)

    # ---- contrasts ----
    vs_s3 = {m: contrast(pt[m], pt_s3) for m in pt}
    vs_s1 = {m: contrast(pt[m], pt_s1) for m in pt}
    sep = {m: first_sep(pt[m], pt_s3) for m in pt}

    def row(name, ptm, ansm):
        r = dict(name=name, mean_ans_turns=float(ansm.mean()),
                 hit_rate=float(ansm.mean() / TMAX))
        for T in BUDGETS:
            r[f"any@{T}"] = float(FA.anytime(ptm, T).mean())
            r[f"end@{T}"] = float(FA.endpoint(ptm, T).mean())
        return r
    rows = {m: row(f"a3-{m}", pt[m], ansn[m]) for m in pt}
    rows["s3 popular-item"] = row("s3 popular-item", pt_s3, ans_s3)
    rows["s1 concepts"] = row("s1 concepts", pt_s1, ans_s1)

    # blind hit-rate verdict
    blind_hit = rows["blind"]["mean_ans_turns"]
    hit_rose = blind_hit > s3_hit + 1e-6
    blind_beats_s3 = any(vs_s3["blind"][T]["ci"][0] > 0 for T in BUDGETS)

    # ---- console summary ----
    print("\n==== A3 co-known prober (any/end NDCG@10; hit = mean answered turns / 24) ====", flush=True)
    for m in ("s3 popular-item", "s1 concepts"):
        r = rows[m]
        print(f"  {m:16s} any24 {r['any@24']:.4f} end24 {r['end@24']:.4f} | hit {r['mean_ans_turns']:.1f}/24", flush=True)
    for m in ("blind", "decay", "table"):
        r = rows[m]; lab = "a3-" + m + (" [PRIV]" if m == "table" else "")
        c = vs_s3[m]
        print(f"  {lab:16s} any24 {r['any@24']:.4f} end24 {r['end@24']:.4f} | hit {r['mean_ans_turns']:.1f}/24 "
              f"| vs s3 @24 {c[24]['delta']:+.4f}[{c[24]['ci'][0]:+.4f},{c[24]['ci'][1]:+.4f}]", flush=True)
    print(f"\n  HIT ROSE (a3-blind {blind_hit:.1f} > s3 {s3_hit:.1f})? {hit_rose}", flush=True)
    print(f"  a3-blind BEATS s3 (CI excl 0) at any budget? {blind_beats_s3}", flush=True)
    for T in BUDGETS:
        c = vs_s3["blind"][T]
        print(f"  a3-blind vs s3 @T{T}: {c['delta']:+.4f}[{c['ci'][0]:+.4f},{c['ci'][1]:+.4f}]", flush=True)

    # ---- verdict text ----
    if blind_beats_s3:
        verdict = ("A3 co-known conditioning BEATS the best static s3 (CI excl 0) at >=1 budget -- blind "
                   "item-neighborhood probing captures part of the Branch-B prize.")
    elif hit_rose:
        verdict = (f"A3-blind RAISES the hit rate ({blind_hit:.1f} vs s3 {s3_hit:.1f}/24) but does NOT beat s3 on "
                   "NDCG -- extra answers land on lower-value neighbors; hit rate does not convert. Branch B stands.")
    else:
        verdict = (f"A3-blind does NOT raise the hit rate ({blind_hit:.1f} vs s3 {s3_hit:.1f}/24) -- adaptive "
                   "co-known conditioning finds no purchase over the fixed popular list. Branch B stands unqualified.")
    print(f"\n  VERDICT: {verdict}", flush=True)

    # ---- persist json ----
    out = dict(
        config=dict(dataset="ML-25M", instrument="RecVAE-d512 + I2.5 learned fold", TMAX=TMAX,
                    budgets=list(BUDGETS), n_users=n, bank_items=len(item_cids),
                    cooc_population_users=int(nU_pop), cooc_norm="cosine of co-rating counts",
                    coverage_prior="study-cohort coverage (pop_rate)", decay_step=DECAY_STEP,
                    turn1=CANDS[turn1_cid]["key"], fold_ckpt=P4.CKPT_BEST, best_val=blob["state"]["best_val"]),
        reproduction=dict(s1_any24=s1_any24, s1_target=0.2103, s3_any24=s3_any24, s3_target=0.2786, ok=bool(ok)),
        s3_hit_turns=s3_hit,
        arms={rows[m]["name"]: rows[m] for m in rows},
        vs_s3={f"a3-{m}": {str(T): vs_s3[m][T] for T in BUDGETS} for m in pt},
        vs_s1={f"a3-{m}": {str(T): vs_s1[m][T] for T in BUDGETS} for m in pt},
        first_separation_vs_s3={f"a3-{m}": dict(turn=sep[m][0], sign=sep[m][1]) for m in pt},
        ndcg_curves={f"a3-{m}": [float(x) for x in pt[m].mean(axis=0)] for m in pt},
        hypothesis=dict(hit_rose=bool(hit_rose), blind_hit=blind_hit, s3_hit=s3_hit,
                        blind_beats_s3=bool(blind_beats_s3)),
        verdict=verdict, wall_min=round((time.time() - t0) / 60, 2))
    os.makedirs("experiments", exist_ok=True)
    json.dump(out, open(OUT_JSON, "w"), indent=1, default=str)

    # ---- append MD section ----
    def frow(name, r, cdict=None):
        base = (f"| {name} | {r['any@8']:.4f}/{r['end@8']:.4f} | {r['any@16']:.4f}/{r['end@16']:.4f} | "
                f"{r['any@24']:.4f}/{r['end@24']:.4f} | {r['mean_ans_turns']:.1f} ({100*r['hit_rate']:.0f}%) |")
        if cdict is not None:
            c = cdict[24]
            base += f" {c['delta']:+.4f}[{c['ci'][0]:+.4f},{c['ci'][1]:+.4f}] |"
        else:
            base += " -- |"
        return base

    md = []
    md.append("\n## A3 co-known prober\n\n")
    md.append(f"Date 2026-07-08. Script `scripts/i25_phase4_a3.py` (imports/reuses `i25_phase4_fair.py` "
              "UNMODIFIED). NO LLM calls; deterministic; local compute.\n\n")
    md.append("**Reproduction gate (existing harness, before adding anything):** "
              f"s1 any@24 = {s1_any24:.4f} (target 0.2103), s3 any@24 = {s3_any24:.4f} (target 0.2786) -> "
              f"reproduced = {ok}.\n\n")
    md.append("**Pre-registered hypothesis:** a3-blind raises the mean-answered-turns (hit rate) above s3's "
              f"{s3_hit:.1f}/24 (~{100*s3_hit/24:.0f}%) by exploiting answered-item co-known neighborhoods. "
              "If the hit rate does not rise, adaptive conditioning found no purchase and Branch B stands "
              "unqualified.\n\n")
    md.append("**Arm.** Pool = the 160-item top-coverage ladder bank s3 draws from (coverage>=3), item probes "
              "only. Turn 1 = the globally best s3 item (%s). After each turn: if ANSWERED -> next = highest "
              "coverage_prior(j) x mean co-known cosine(j | answered set), unasked, tie-break global coverage; "
              "if REFUSED -> next unasked item in global coverage order. Co-known cosine built from ML-25M "
              "training users (trU, n=%d) with all 298 study users' rows dropped (zero leak, asserted). "
              "a3-decay: a sponsoring anchor's weight decays 1->0.5->0 as its neighbors are refused (2 "
              "refusals in X's region stop probing X). a3-table (PRIVILEGED): candidates restricted to "
              "TRUE-answerable items -> hit-rate ceiling for this policy class.\n\n"
              % (CANDS[turn1_cid]["key"], nU_pop))
    md.append("| arm | any/end @8 | any/end @16 | any/end @24 | hit (ansT) | delta-any@24 vs s3 [CI] |\n"
              "|---|---|---|---|---|---|\n")
    md.append(frow("s3 popular-item (opponent)", rows["s3 popular-item"]) + "\n")
    md.append(frow("s1 concepts", rows["s1 concepts"]) + "\n")
    md.append(frow("a3-blind", rows["blind"], vs_s3["blind"]) + "\n")
    md.append(frow("a3-decay", rows["decay"], vs_s3["decay"]) + "\n")
    md.append(frow("a3-table (PRIV)", rows["table"], vs_s3["table"]) + "\n")

    md.append("\n**Contrast a3-blind vs s3 (THE contrast) and vs s1, per budget:**\n\n"
              "| budget T | a3-blind vs s3 [CI] | a3-blind vs s1 [CI] |\n|---|---|---|\n")
    for T in BUDGETS:
        c3 = vs_s3["blind"][T]; c1 = vs_s1["blind"][T]
        md.append(f"| {T} | {c3['delta']:+.4f}[{c3['ci'][0]:+.4f},{c3['ci'][1]:+.4f}] | "
                  f"{c1['delta']:+.4f}[{c1['ci'][0]:+.4f},{c1['ci'][1]:+.4f}] |\n")

    md.append("\n**Mechanism metric -- hit rate (mean answered turns / 24):** "
              f"s3 = {s3_hit:.1f} (~{100*s3_hit/24:.0f}%); a3-blind = {rows['blind']['mean_ans_turns']:.1f} "
              f"(~{100*rows['blind']['hit_rate']:.0f}%); a3-decay = {rows['decay']['mean_ans_turns']:.1f}; "
              f"a3-table (ceiling) = {rows['table']['mean_ans_turns']:.1f}. "
              f"Hit rose vs s3? **{hit_rose}**. Did it convert to NDCG (beat s3, CI excl 0)? **{blind_beats_s3}**.\n\n")
    fs = sep["blind"]
    md.append(f"**First separation (a3-blind belief(t) vs s3 belief(t), CI excl 0):** "
              f"{('turn ' + str(fs[0]) + ' (sign ' + fs[1] + ')') if fs[0] else 'NEVER within T=24'}.\n\n")

    md.append("**NDCG@10(t) curves (t=1..24):**\n\n| arm | " + " | ".join(f"t{t+1}" for t in range(TMAX)) + " |\n")
    md.append("|" + "---|" * (TMAX + 1) + "\n")
    for lab, m in (("s3 popular-item", None), ("a3-blind", "blind"), ("a3-decay", "decay"), ("a3-table (PRIV)", "table")):
        cur = pt_s3.mean(axis=0) if m is None else pt[m].mean(axis=0)
        md.append(f"| {lab} | " + " | ".join(f"{c:.3f}" for c in cur) + " |\n")

    md.append(f"\n**VERDICT:** {verdict}\n\n")
    md.append("**ASSUMPTIONS / judgment calls (a3):**\n"
              "1. Probe bank = the 160 top-coverage ladder items (coverage>=3) that s3 draws from; item probes only.\n"
              "2. Co-known statistic = COSINE of binary co-rating counts, cooc(a,b)/sqrt(cooc(a,a)cooc(b,b)); "
              "diagonal zeroed (an item never scores against itself). Cosine removes each item's marginal "
              "popularity so the separate coverage prior is not double-counted.\n"
              "3. Neighbor value = coverage_prior(j) x mean cosine affinity of j to the answered set. "
              "coverage_prior = study-cohort coverage (pop_rate). Affinity = MEAN over answered anchors "
              "(not sum -> not confounded with #answers).\n"
              "4. Population for cooc = ML-25M training users (trU, n=%d) with all 298 study users' rows dropped; "
              "study users are eval-split and disjoint from trU (asserted leak=0). Co-rating uses each "
              "population user's FULL rated profile intersected with the bank.\n"
              "5. Turn 1 = s3[0] (the globally best s3 item). Fallback (refusal) order = global coverage "
              "descending, tie-break cid. Neighborhood tie-break = coverage.\n"
              "6. a3-decay: each neighborhood-selected probe records a SPONSORING anchor = raw-cosine argmax "
              "over the answered set; a refusal increments that sponsor's counter; anchor weight = "
              "max(0, 1 - 0.5*refusals) -> 1, 0.5, 0 (hard stop at 2 refusals in that region).\n"
              "7. a3-table (PRIVILEGED, labelled): candidate set restricted to true-answerable items each turn "
              "-> every probe answered -> hit-rate ceiling min(24, #answerable bank items); turn 1 = first "
              "answerable item in global order.\n"
              "8. No neighborhood-size cap (full 160-item bank scored each turn); refusal = no-op turn "
              "(belief unchanged), user retained; all 298 users in every mean (fair, inherited from the harness).\n"
              "9. Deterministic: bootstrap paired per-user BOOT=%d seed=%d; all selectors deterministic "
              "(coverage/cid tie-breaks).\n\n"
              % (nU_pop, P4.BOOT, P4.SEED))

    with open(OUT_MD, "a", encoding="utf-8") as f:
        f.write("".join(md))
    print(f"\n[a3] wall {out['wall_min']}m -> {OUT_JSON}, appended section to {OUT_MD}", flush=True)


if __name__ == "__main__":
    main()
