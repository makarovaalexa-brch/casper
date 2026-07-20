"""E0e — EMERGENT SWITCH (no hard-coded phases) + emergence diagnostic.

Greedy selector over the WHOLE bank (concepts + items). Each turn scores every remaining candidate q
by:
    score(q) = marginal_info(q | answered directions so far) * answerability(q) [ * divisiveness(q) ]
where marginal_info(q) = || q - proj_{span(answered dirs)} q ||  (belief-only; NEVER touches z*).
answerability = the TRUE table (arm 'true': restrict to answerable) OR p_hat (arm 'phat': all
candidates; a truly-unanswerable pick = refusal, turn consumed, no update, and NOT added to the
answered span). Two value variants: with and without the divisiveness multiplier.

Belief updates realizably: z'=z+eta*a*q, a=cos(z*,q). Only ANSWERED directions enter the
answered-span basis (Gram-Schmidt orthonormal).

EMERGENCE DIAGNOSTIC: per user, the turn at which the selector first picks an ITEM question.
Distribution across users. If it never switches, report the score gap at the argmax.

REUSES e0_gonogo machinery EXACTLY. NO LLM calls.

Run:  python scripts/e0e_emergent.py
"""
import os, sys, json
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "instrument2"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import warnings; warnings.filterwarnings("ignore")
import llm_answerability_gate as G
import e0_gonogo as E

OUT = "experiments/E0e_emergent.json"
SEED = 0


def boot(diff, nb=5000):
    rng = np.random.default_rng(SEED)
    b = np.array([diff[rng.integers(0, len(diff), len(diff))].mean() for _ in range(nb)])
    return float(diff.mean()), [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))], float((b > 0).mean())


def marginal_norms(dirs, basis):
    """||q - proj_basis q|| for every row q of dirs. basis = orthonormal rows (k x d) or None."""
    if basis is None or len(basis) == 0:
        return np.linalg.norm(dirs, axis=1)
    Bm = np.array(basis)                       # k x d
    proj = dirs @ Bm.T                         # n x k
    resid = dirs - proj @ Bm
    return np.linalg.norm(resid, axis=1)


def gram_add(basis, q):
    """Add q to an orthonormal basis (list of rows) via Gram-Schmidt; ignore if ~dependent."""
    v = q.astype(np.float64).copy()
    for b in basis:
        v = v - (v @ b) * b
    nv = np.linalg.norm(v)
    if nv > 1e-8:
        basis.append(v / nv)
    return basis


def run_selector(FI, rec, use_true, use_div, ETA, T):
    """Greedy emergent selector. Returns (curve, first_item_turn_1based, never_switch_gap)."""
    d = FI["d"]; dirs = rec["dir"]; n = len(dirs)
    kinds = [1 if k.startswith("I:") else 0 for k in rec["key"]]   # 1=item
    z = np.zeros(d); basis = []; used = set(); curve = []
    first_item = None; never_gap = None
    for turn in range(T):
        cand = [c for c in range(n) if c not in used]
        if use_true:
            cand = [c for c in cand if rec["ans"][c]]              # true table: only answerable
        if not cand:
            curve.append(E.ndcg(FI, z, rec)); continue
        mi = marginal_norms(dirs[cand], basis)
        if use_true:
            w = np.ones(len(cand))
        else:
            w = np.array([rec["phat"][c] for c in cand])
        score = mi * w
        if use_div:
            score = score * np.array([max(float(rec["div"][c]), 0.0) for c in cand])
        bi = int(np.argmax(score)); c = cand[bi]; used.add(c)
        # emergence: is the argmax an item?
        if kinds[c] == 1 and first_item is None:
            first_item = turn                                      # 0-based
        answered = bool(rec["ans"][c])                            # true outcome
        if answered:
            q = dirs[c]; a = float((q @ rec["zstar"]) / rec["nz"])
            z = z + ETA * a * q; basis = gram_add(basis, q)
        # else refusal (only possible in phat arm): turn consumed, no update, not added to span
        curve.append(E.ndcg(FI, z, rec))
    # if the selector never picked an item, record the best item score vs best concept score at t=1
    if first_item is None:
        mi0 = marginal_norms(dirs, basis if basis else [])
        if use_true:
            mask = rec["ans"].astype(bool)
        else:
            mask = np.ones(n, bool)
        item_mask = np.array([k == 1 for k in kinds]) & mask
        conc_mask = np.array([k == 0 for k in kinds]) & mask
        sc = mi0 * (np.ones(n) if use_true else rec["phat"])
        if use_div:
            sc = sc * np.maximum(rec["div"], 0.0)
        best_item = float(sc[item_mask].max()) if item_mask.any() else None
        best_conc = float(sc[conc_mask].max()) if conc_mask.any() else None
        if best_item is not None and best_conc is not None:
            never_gap = best_conc - best_item
    return curve, (first_item + 1 if first_item is not None else None), never_gap


def main():
    D = G.load_data()
    FI = E.build_foldin(D)
    pm = E.PModel(E.PMODEL_JSON)
    cpct = E.concept_pct_vec(D)
    per_user, split = E.per_user_from_gate(D)
    U = E.assemble_users(D, per_user, split, FI, pm, cpct)
    T, ETA = E.T, E.ETA
    B = E.BLIND_SCHEDULE
    print(f"[E0e] {len(U)} eligible users", flush=True)

    for rec in U:
        rec["kmap"] = {k: c for c, k in enumerate(rec["key"])}

    # baselines
    def run_skip(rec):
        # static+skip priority (recompute here for a matched user set)
        return None
    static = [E.sel_static(FI, rec, B) for rec in U]

    variants = {
        "emergent_true_noDiv": dict(use_true=True, use_div=False),
        "emergent_true_div":   dict(use_true=True, use_div=True),
        "emergent_phat_noDiv": dict(use_true=False, use_div=False),
        "emergent_phat_div":   dict(use_true=False, use_div=True),
    }
    arms = {"static": static}
    emergence = {}
    for name, cfg in variants.items():
        curves = []; fts = []; gaps = []
        for rec in U:
            cur, ft, gap = run_selector(FI, rec, cfg["use_true"], cfg["use_div"], ETA, T)
            curves.append(cur); fts.append(ft); gaps.append(gap)
        arms[name] = curves; emergence[name] = dict(first_item_turn=fts, never_gap=gaps)

    idx = [i for i in range(len(U)) if all(arms[nm][i][t] is not None for nm in arms for t in range(T))]
    n = len(idx)
    print(f"[E0e] {n} users with all arms defined", flush=True)

    def anyt(name): return np.array([np.mean(arms[name][i]) for i in idx])
    def endp(name): return np.array([arms[name][i][-1] for i in idx])
    def pturn(name): return [float(np.mean([arms[name][i][t] for i in idx])) for t in range(T)]

    static_any = anyt("static"); static_end = endp("static")
    res = {"n_users": n, "arms": {}, "emergence": {}}
    res["arms"]["static"] = dict(anytime=float(static_any.mean()), endpoint=float(static_end.mean()),
                                 per_turn=pturn("static"))
    for name in variants:
        a_any = anyt(name); a_end = endp(name)
        dB_any, ciB_any, pB_any = boot(a_any - static_any)
        dB_end, ciB_end, _ = boot(a_end - static_end)
        res["arms"][name] = dict(
            anytime=float(a_any.mean()), endpoint=float(a_end.mean()), per_turn=pturn(name),
            vs_static=dict(anytime_delta=dB_any, anytime_ci=ciB_any, anytime_p_gt0=pB_any,
                           endpoint_delta=dB_end, endpoint_ci=ciB_end))
        # emergence diagnostic on the kept users
        fts = [emergence[name]["first_item_turn"][i] for i in idx]
        gaps = [emergence[name]["never_gap"][i] for i in idx if emergence[name]["never_gap"][i] is not None]
        switched = [f for f in fts if f is not None]
        hist = {t: int(sum(1 for f in switched if f == t)) for t in range(1, T + 1)}
        res["emergence"][name] = dict(
            n_users=n, n_switched=len(switched), frac_switched=float(len(switched) / max(n, 1)),
            first_item_turn_hist=hist,
            first_item_turn_quantiles=(dict(q25=float(np.percentile(switched, 25)),
                                            median=float(np.median(switched)),
                                            q75=float(np.percentile(switched, 75))) if switched else None),
            never_switch_score_gap_mean=(float(np.mean(gaps)) if gaps else None),
            never_switch_score_gap_median=(float(np.median(gaps)) if gaps else None))

    json.dump(res, open(OUT, "w"), indent=1, default=str)

    print(f"\n[E0e] static anytime={static_any.mean():.4f} end={static_end.mean():.4f}")
    for name in variants:
        r = res["arms"][name]; e = res["emergence"][name]
        print(f"[E0e] {name:22s} any={r['anytime']:.4f} end={r['endpoint']:.4f} | "
              f"vsB any d={r['vs_static']['anytime_delta']:+.4f} CI{[round(x,4) for x in r['vs_static']['anytime_ci']]} | "
              f"switched {e['n_switched']}/{n} ({100*e['frac_switched']:.0f}%) "
              f"gap={e['never_switch_score_gap_mean']}")
    print(f"[saved] {OUT}", flush=True)


if __name__ == "__main__":
    main()
