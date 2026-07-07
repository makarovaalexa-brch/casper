"""E0f FOLD-MISCALIBRATION DIAGNOSIS (triggered because the mandatory control FAILED).

The cold single-item control HURTS for the real-rating answer (Δ=-0.053, CI excludes 0) and does NOT
help for the geometric answer either (Δ=-0.011, CI spans 0). Since it is NOT specific to real ratings,
this rules the real-rating rescale OUT as the cause and points at the ITEM-TOKEN FOLD itself. This
script quantifies WHY, isolating: (A) the z=0 baseline; (B) concept vs item single-token folds from
cold; (C) an eta sweep for the item token (is eta=16 an overshoot for a single narrow direction?);
(D) matched-step-norm concept vs item (breadth vs magnitude); (E) the descent-context item fold at t=5.

NO LLM calls.  Run:  python scripts/e0f_fold_diag.py
"""
import os, sys, json
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "instrument2"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import warnings; warnings.filterwarnings("ignore")
import llm_answerability_gate as G
import e0_gonogo as E
import e0f_lib as L

OUT = "experiments/E0f_fold_diag.json"


def boot(diff, nb=5000):
    rng = np.random.default_rng(0)
    b = np.array([diff[rng.integers(0, len(diff), len(diff))].mean() for _ in range(nb)])
    return float(diff.mean()), [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))], float((b > 0).mean())


def main():
    D = G.load_data()
    FI = E.build_foldin(D)
    pm = E.PModel(E.PMODEL_JSON)
    cpct = E.concept_pct_vec(D)
    per_user, split = E.per_user_from_gate(D)
    U = E.assemble_users(D, per_user, split, FI, pm, cpct)
    L.augment_users(D, split, U)
    beta, bstats = L.compute_beta(FI, U)
    for rec in U:
        rec["kmap"] = {k: c for c, k in enumerate(rec["key"])}
    Wn = FI["Wn"]; B = E.BLIND_SCHEDULE
    concept_sched = [k for k in B if k.startswith("C")]
    res = {"beta": beta, "beta_stats": bstats}

    # ---- A: base NDCG at z=0 ----
    base = np.array([E.ndcg(FI, np.zeros(FI["d"]), rec) for rec in U], float)
    res["A_base_ndcg_z0"] = dict(mean=float(np.nanmean(base)))

    # magnitude of answers: mean |a| for concepts vs items (geometric)
    conc_a, item_a = [], []
    for rec in U:
        for c, k in enumerate(rec["key"]):
            a = abs(float((rec["dir"][c] @ rec["zstar"]) / rec["nz"]))
            (conc_a if k.startswith("C") else item_a).append(a)
    res["answer_magnitude"] = dict(mean_abs_a_concept=float(np.mean(conc_a)),
                                   mean_abs_a_item_geo=float(np.mean(item_a)),
                                   note="fold step norm = eta*|a|*1 (all dirs unit-norm)")

    # deterministic per-user random known-rated item pick (same as control)
    pick = {}
    for rec in U:
        if rec["known_rated"]:
            rng = np.random.default_rng(1234 + rec["u"])
            pick[rec["u"]] = rec["known_rated"][int(rng.integers(len(rec["known_rated"])))]

    # ---- B: single CONCEPT fold from cold (first answerable schedule concept) ----
    dC = []
    for rec in U:
        b0 = E.ndcg(FI, np.zeros(FI["d"]), rec)
        if b0 is None:
            continue
        folded = None
        for key in concept_sched:
            c = rec["kmap"].get(key)
            if c is not None and rec["ans"][c]:
                q = rec["dir"][c]; a = float((q @ rec["zstar"]) / rec["nz"])
                folded = E.ndcg(FI, E.ETA * a * q, rec); break
        if folded is not None:
            dC.append(folded - b0)
    mC, ciC, pC = boot(np.array(dC))
    res["B_single_concept_from_cold"] = dict(mean_delta=mC, ci95=ciC, p_gt0=pC, n=len(dC))

    # ---- C: eta sweep for the single item token (geometric + real) ----
    etas = [1, 2, 4, 8, 16, 24, 32]
    sweep = {"geo": {}, "real": {}}
    for kind in ("geo", "real"):
        for eta in etas:
            dd = []
            for rec in U:
                if rec["u"] not in pick:
                    continue
                b0 = E.ndcg(FI, np.zeros(FI["d"]), rec)
                if b0 is None:
                    continue
                j, r = pick[rec["u"]]; q = Wn[j]
                a = L.geo_answer(rec, q) if kind == "geo" else L.real_answer(rec, j, r, beta)
                n = E.ndcg(FI, eta * a * q, rec)
                if n is not None:
                    dd.append(n - b0)
            m, ci, p = boot(np.array(dd))
            sweep[kind][str(eta)] = dict(mean_delta=m, ci95=ci, p_gt0=p, n=len(dd))
    res["C_item_eta_sweep"] = sweep

    # ---- D: matched-step-norm concept vs item (fix step norm s = eta*|a|) ----
    # For each user use the SAME first-answerable concept and the SAME picked item, set fold step to a
    # fixed norm s along each unit direction with the answer's SIGN; compare NDCG delta. Isolates
    # direction BREADTH from answer magnitude.
    matched = {}
    for s in [1.0, 2.0, 4.0, 8.0]:
        dCs, dIs = [], []
        for rec in U:
            b0 = E.ndcg(FI, np.zeros(FI["d"]), rec)
            if b0 is None:
                continue
            # concept
            for key in concept_sched:
                c = rec["kmap"].get(key)
                if c is not None and rec["ans"][c]:
                    q = rec["dir"][c]; sgn = np.sign(float((q @ rec["zstar"]) / rec["nz"])) or 1.0
                    nC = E.ndcg(FI, s * sgn * q, rec)
                    if nC is not None:
                        dCs.append(nC - b0)
                    break
            # item
            if rec["u"] in pick:
                j, r = pick[rec["u"]]; q = Wn[j]
                sgn = np.sign(L.geo_answer(rec, q)) or 1.0
                nI = E.ndcg(FI, s * sgn * q, rec)
                if nI is not None:
                    dIs.append(nI - b0)
        matched[str(s)] = dict(concept_delta=float(np.mean(dCs)), item_delta=float(np.mean(dIs)),
                               n_concept=len(dCs), n_item=len(dIs))
    res["D_matched_step_norm"] = matched

    # ---- E: descent-context item fold (after 4 concepts) ----
    def fold4(rec):
        z = np.zeros(FI["d"])
        for key in concept_sched[:4]:
            c = rec["kmap"].get(key)
            if c is not None and rec["ans"][c]:
                q = rec["dir"][c]; a = float((q @ rec["zstar"]) / rec["nz"]); z = z + E.ETA * a * q
        return z
    dE_geo, dE_real = [], []
    for rec in U:
        if rec["u"] not in pick:
            continue
        z4 = fold4(rec); n4 = E.ndcg(FI, z4, rec)
        if n4 is None:
            continue
        j, r = pick[rec["u"]]; q = Wn[j]
        ng = E.ndcg(FI, z4 + E.ETA * L.geo_answer(rec, q) * q, rec)
        nr = E.ndcg(FI, z4 + E.ETA * L.real_answer(rec, j, r, beta) * q, rec)
        if ng is not None:
            dE_geo.append(ng - n4)
        if nr is not None:
            dE_real.append(nr - n4)
    res["E_item_after_4_concepts"] = dict(
        geo=dict(mean_delta=float(np.mean(dE_geo)), n=len(dE_geo)),
        real=dict(mean_delta=float(np.mean(dE_real)), n=len(dE_real)))

    os.makedirs("experiments", exist_ok=True)
    json.dump(res, open(OUT, "w"), indent=1, default=str)

    # print
    print(f"\n[A] base NDCG@10 (z=0) = {res['A_base_ndcg_z0']['mean']:.4f}")
    print(f"    mean|a| concept={res['answer_magnitude']['mean_abs_a_concept']:.4f} "
          f"item(geo)={res['answer_magnitude']['mean_abs_a_item_geo']:.4f}")
    print(f"[B] single CONCEPT from cold: d={mC:+.4f} CI[{ciC[0]:+.4f},{ciC[1]:+.4f}] n={len(dC)}")
    print("[C] item eta sweep (delta vs z=0):")
    for kind in ("geo", "real"):
        row = " ".join(f"eta{e}={sweep[kind][str(e)]['mean_delta']:+.4f}" for e in etas)
        print(f"    {kind:4s}: {row}")
    print("[D] matched-step-norm (concept vs item delta):")
    for s, v in matched.items():
        print(f"    s={s}: concept={v['concept_delta']:+.4f} item={v['item_delta']:+.4f}")
    print(f"[E] item after 4 concepts: geo={res['E_item_after_4_concepts']['geo']['mean_delta']:+.4f} "
          f"real={res['E_item_after_4_concepts']['real']['mean_delta']:+.4f}")
    print(f"[saved] {OUT}", flush=True)


if __name__ == "__main__":
    main()
