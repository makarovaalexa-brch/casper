"""E0d — HARD-CODED DESCENT PROBE (coarse->fine channel switch).

Owner's challenge: E0's routers never actually did coarse-to-fine. This probe hand-builds a
two-phase schedule and asks: does SWITCHING CHANNELS (concepts -> items) after concept saturation
beat staying on the concept-heavy static?

  Phase 1 (turns 1..k): the static schedule's top-k CONCEPT questions, SAME for everyone.
  Phase 2 (turns k+1..8): (8-k) ITEM questions chosen PER USER.

Arms:
  a) k=4, items chosen by the TRUE answerability table, ranked by the population value the statics
     use (value-when-answered single-question NDCG = popval). Upper bound.
  a-div) same but items ranked by divisiveness div(q)=q^T Cov(W) q (belief-independent).
  b) k=4, items chosen by the p_hat SURROGATE (rank all items by p_hat*popval; outcome gated by the
     true table -> a wrong pick that is truly unanswerable is a refusal, turn wasted). Realizable.
  c) switch-point sweep: k=3 and k=5 for arm (a).

Baselines (recomputed on the SAME user set): static B (E.sel_static, BLIND_SCHEDULE) and static+skip
(B's diversified head ++ popval fills, per-user refusal skip).

REUSES e0_gonogo machinery EXACTLY (fold-in, belief z'=z+eta*a*q with a=cos(z*,q), NDCG, bootstrap).
NO LLM calls. The answer VALUE is geometric (a=cos(z*,q)) for EVERY question in this harness --
real ratings never enter the answer; the LLM answerability table only gates which questions are
askable (see ASSUMPTIONS). So there is no rated/unrated answer-value distinction to make here;
no SENSITIVITY-ONLY arm is needed.

Run:  python scripts/e0d_descent.py
"""
import os, sys, json, collections
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "instrument2"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import warnings; warnings.filterwarnings("ignore")
import llm_answerability_gate as G
import e0_gonogo as E

OUT = "experiments/E0d_descent.json"
SEED = 0


def boot(diff, nb=5000):
    rng = np.random.default_rng(SEED)
    b = np.array([diff[rng.integers(0, len(diff), len(diff))].mean() for _ in range(nb)])
    return float(diff.mean()), [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))], float((b > 0).mean())


def main():
    D = G.load_data()
    FI = E.build_foldin(D)
    pm = E.PModel(E.PMODEL_JSON)
    cpct = E.concept_pct_vec(D)
    per_user, split = E.per_user_from_gate(D)
    U = E.assemble_users(D, per_user, split, FI, pm, cpct)
    T, ETA = E.T, E.ETA
    B = E.BLIND_SCHEDULE
    print(f"[E0d] {len(U)} eligible users; schedule={B}", flush=True)

    for rec in U:
        rec["kmap"] = {k: c for c, k in enumerate(rec["key"])}
        rec["ansset"] = {k: bool(rec["ans"][c]) for c, k in enumerate(rec["key"])}

    # ---- population value-when-answered single-question NDCG (popval), over ALL keys ----
    # popval[k] = mean over users who CAN answer k of the single-question NDCG (the population value
    # the statics greedy-forward on). Computed per candidate DIRECTION (concepts shared; item dirs
    # are per-item Wn rows so identical across users).
    keydir = {}
    for rec in U:
        for c, k in enumerate(rec["key"]):
            if k not in keydir:
                keydir[k] = rec["dir"][c]
    keycount = collections.Counter(k for rec in U for k in rec["key"])
    popval = {}; popn = {}
    for k, q in keydir.items():
        vals = []
        for rec in U:
            c = rec["kmap"].get(k)
            if c is None or not rec["ans"][c]:
                continue
            a = float((q @ rec["zstar"]) / rec["nz"])
            n = E.ndcg(FI, ETA * a * q, rec)
            if n is not None:
                vals.append(n)
        popval[k] = float(np.mean(vals)) if vals else -1.0
        popn[k] = len(vals)

    # divisiveness per key (belief-independent, = q^T Cov(W) q) — already in rec["div"]
    div_of = {}
    for rec in U:
        for c, k in enumerate(rec["key"]):
            if k not in div_of:
                div_of[k] = float(rec["div"][c])

    concept_sched = [k for k in B if k.startswith("C")]
    print(f"[E0d] phase-1 concept order (from static schedule): {concept_sched}", flush=True)

    # -------- descent runner --------
    def phase1_concepts(rec, z, k, curve):
        """Ask the first k schedule concepts (fixed for everyone); update if answerable; turn consumed
        either way (hard-coded schedule)."""
        for t in range(k):
            key = concept_sched[t] if t < len(concept_sched) else None
            if key is not None:
                c = rec["kmap"].get(key)
                if c is not None and rec["ans"][c]:
                    q = rec["dir"][c]; a = float((q @ rec["zstar"]) / rec["nz"])
                    z = z + ETA * a * q
            curve.append(E.ndcg(FI, z, rec))
        return z

    def descent_true(rec, k, rank="pop"):
        """Phase1: k schedule concepts. Phase2: (8-k) answerable ITEMS ranked by popval|div (true table).
        Returns curve + turn-index (0-based) of the first item question actually asked."""
        z = np.zeros(FI["d"]); curve = []
        z = phase1_concepts(rec, z, k, curve)
        # answerable items for this user, ranked
        items = [key for key, is_a in rec["ansset"].items() if key.startswith("I:") and is_a]
        val = popval if rank == "pop" else div_of
        items.sort(key=lambda kk: -val.get(kk, -1.0))
        first_item_turn = None
        need = T - k
        for key in items[:need]:
            c = rec["kmap"][key]; q = rec["dir"][c]
            a = float((q @ rec["zstar"]) / rec["nz"]); z = z + ETA * a * q
            if first_item_turn is None:
                first_item_turn = len(curve)  # 0-based turn index
            curve.append(E.ndcg(FI, z, rec))
        while len(curve) < T:                 # ran out of answerable items -> hold
            curve.append(curve[-1] if curve else E.ndcg(FI, np.zeros(FI["d"]), rec))
        return curve, first_item_turn

    def descent_phat(rec, k):
        """Phase2 items chosen by p_hat*popval over ALL items (not just answerable); outcome gated by
        the true table (unanswerable pick = refusal, turn consumed, no update)."""
        z = np.zeros(FI["d"]); curve = []
        z = phase1_concepts(rec, z, k, curve)
        all_items = [(key, rec["kmap"][key]) for key in rec["kmap"] if key.startswith("I:")]
        all_items.sort(key=lambda kc: -(rec["phat"][kc[1]] * max(popval.get(kc[0], 0.0), 0.0)))
        need = T - k; picked = 0
        for key, c in all_items:
            if picked >= need:
                break
            q = rec["dir"][c]
            if rec["ans"][c]:                              # truly answerable -> answer + update
                a = float((q @ rec["zstar"]) / rec["nz"]); z = z + ETA * a * q
            # else refusal -> turn consumed, no update
            picked += 1
            curve.append(E.ndcg(FI, z, rec))
        while len(curve) < T:
            curve.append(curve[-1] if curve else E.ndcg(FI, np.zeros(FI["d"]), rec))
        return curve

    # -------- baselines: static B and static+skip (same user set) --------
    pop_order = sorted([k for k in keydir if popn[k] > 0], key=lambda k: -popval[k])
    priority = list(B) + [k for k in pop_order if k not in B]

    def run_static(rec):
        return E.sel_static(FI, rec, B)

    def run_skip(rec):
        z = np.zeros(FI["d"]); curve = []; asked = 0
        for key in priority:
            if asked >= T:
                break
            c = rec["kmap"].get(key)
            if c is not None and rec["ans"][c]:
                q = rec["dir"][c]; a = float((q @ rec["zstar"]) / rec["nz"]); z = z + ETA * a * q
                asked += 1; curve.append(E.ndcg(FI, z, rec))
        while len(curve) < T:
            curve.append(curve[-1] if curve else E.ndcg(FI, np.zeros(FI["d"]), rec))
        return curve

    # -------- run all arms --------
    arms = {}
    firstturn = {}
    arms["static"] = [run_static(r) for r in U]
    arms["static_skip"] = [run_skip(r) for r in U]
    for name, k, rank in [("descent_k4_true_pop", 4, "pop"), ("descent_k4_true_div", 4, "div"),
                          ("descent_k3_true_pop", 3, "pop"), ("descent_k5_true_pop", 5, "pop")]:
        curves = []; fts = []
        for r in U:
            cur, ft = descent_true(r, k, rank)
            curves.append(cur); fts.append(ft)
        arms[name] = curves; firstturn[name] = fts
    arms["descent_k4_phat"] = [descent_phat(r, 4) for r in U]

    # keep users with all curves fully defined
    idx = [i for i in range(len(U))
           if all(arms[n][i][t] is not None for n in arms for t in range(T))]
    n = len(idx)
    print(f"[E0d] {n} users with all arms defined", flush=True)

    def anyt(name): return np.array([np.mean(arms[name][i]) for i in idx])
    def endp(name): return np.array([arms[name][i][-1] for i in idx])
    def pturn(name): return [float(np.mean([arms[name][i][t] for i in idx])) for t in range(T)]

    res = {"n_users": n, "arms": {}}
    static_any = anyt("static"); static_end = endp("static")
    skip_any = anyt("static_skip"); skip_end = endp("static_skip")
    for name in arms:
        a_any = anyt(name); a_end = endp(name)
        dvsB_any, ci_vB_any, p_vB_any = boot(a_any - static_any)
        dvsB_end, ci_vB_end, _ = boot(a_end - static_end)
        dvsS_any, ci_vS_any, p_vS_any = boot(a_any - skip_any)
        dvsS_end, ci_vS_end, _ = boot(a_end - skip_end)
        res["arms"][name] = dict(
            anytime=float(a_any.mean()), endpoint=float(a_end.mean()),
            per_turn=pturn(name),
            vs_static=dict(anytime_delta=dvsB_any, anytime_ci=ci_vB_any, anytime_p_gt0=p_vB_any,
                           endpoint_delta=dvsB_end, endpoint_ci=ci_vB_end),
            vs_static_skip=dict(anytime_delta=dvsS_any, anytime_ci=ci_vS_any, anytime_p_gt0=p_vS_any,
                                endpoint_delta=dvsS_end, endpoint_ci=ci_vS_end))

    # emergence: for descent, phase-2 first item turn is deterministic (turn k), but record how many
    # users actually reached an item (had >=1 answerable item) for each k
    emg = {}
    for name in firstturn:
        fts = [firstturn[name][i] for i in idx]
        reached = [f for f in fts if f is not None]
        emg[name] = dict(n_reached_item=len(reached), n_total=len(fts),
                         first_item_turn_1based=(int(np.median(reached)) + 1) if reached else None,
                         frac_reached=float(len(reached) / max(len(fts), 1)))
    res["descent_reach"] = emg
    res["popval_top_items"] = [(k, round(popval[k], 4), popn[k])
                               for k in sorted([kk for kk in keydir if kk.startswith("I:")],
                                               key=lambda kk: -popval[kk])[:12]]
    res["config"] = dict(schedule=B, concept_phase_order=concept_sched,
                         answer_value="geometric a=cos(z*,q) for ALL questions; ratings never enter answer",
                         static_anytime=float(static_any.mean()), static_endpoint=float(static_end.mean()),
                         static_skip_anytime=float(skip_any.mean()), static_skip_endpoint=float(skip_end.mean()))
    json.dump(res, open(OUT, "w"), indent=1, default=str)

    print(f"\n[E0d] static B          anytime={static_any.mean():.4f} end={static_end.mean():.4f}")
    print(f"[E0d] static+skip       anytime={skip_any.mean():.4f} end={skip_end.mean():.4f}")
    for name in ["descent_k4_true_pop", "descent_k4_true_div", "descent_k3_true_pop",
                 "descent_k5_true_pop", "descent_k4_phat"]:
        r = res["arms"][name]
        print(f"[E0d] {name:22s} anytime={r['anytime']:.4f} end={r['endpoint']:.4f} | "
              f"vsB any d={r['vs_static']['anytime_delta']:+.4f} CI{[round(x,4) for x in r['vs_static']['anytime_ci']]} | "
              f"vsSkip any d={r['vs_static_skip']['anytime_delta']:+.4f}")
    print(f"[saved] {OUT}", flush=True)


if __name__ == "__main__":
    main()
