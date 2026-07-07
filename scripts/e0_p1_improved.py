"""E0 P1 — faithful O-ans with the POPULATION-VALUE machinery selector B actually uses.

The first pass (e0_gonogo.py) scored O-ans's questions with a belief-only divisiveness*novelty
info value. That value is z*-free (correct) but taste-blind: with eta=16 it injects large steps
along divisive off-taste directions and DEPRESSES NDCG (O-ans < static) — a value-model artifact,
not the absence of an answerability prize (the design doc explicitly warns of exactly this).

This script implements the honest O-ans: the value model is the SAME cohort/population NDCG value
selector B is built from (population knowledge, z*-free — B is allowed it too), and O-ans differs
from B ONLY by (i) being able to route to the questions THIS user can answer (the answerability
table) and (ii) a realizable belief path (z from answers received). That isolates the
answerability-discovery prize cleanly.

  popval[k] = mean over users who CAN answer k of the single-question NDCG (value-when-answered).
  O-ans per user = ask, in descending popval order, the SHARED-pool questions this user can answer
                   (T of them), updating belief z'=z+eta*a*q realizably (a=cos(z*,q)). No refusals
                   (true table known). Same shared candidate pool B's schedule was built from.

Reuses e0_gonogo assembly + fold-in + NDCG exactly. NO LLM calls.
"""
import os, sys, json, collections
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "instrument2"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import warnings; warnings.filterwarnings("ignore")
import llm_answerability_gate as G
import e0_gonogo as E

OUT = "experiments/E0_p1_improved.json"
SEED = 0


def main():
    D = G.load_data()
    FI = E.build_foldin(D)
    pm = E.PModel(E.PMODEL_JSON)
    cpct = E.concept_pct_vec(D)
    per_user, split = E.per_user_from_gate(D)
    U = E.assemble_users(D, per_user, split, FI, pm, cpct)
    T, ETA = E.T, E.ETA
    print(f"[P1*] {len(U)} users", flush=True)

    # shared candidate pool (== B's pool): keys present for >=50% of users
    keycount = collections.Counter(k for rec in U for k in rec["key"])
    shared = [k for k, c in keycount.items() if c >= max(2, int(0.5 * len(U)))]
    sset = set(shared)
    key_dir = {}
    for rec in U:
        for c, k in enumerate(rec["key"]):
            if k in sset and k not in key_dir:
                key_dir[k] = rec["dir"][c]
    print(f"[P1*] shared pool = {len(shared)} keys", flush=True)

    # per-user answerable lookup over shared keys
    for rec in U:
        rec["kmap"] = {k: c for c, k in enumerate(rec["key"])}

    # popval[k] = mean over answerable users of single-question NDCG (value-when-answered)
    popval = {}
    for k in shared:
        q = key_dir[k]; vals = []
        for rec in U:
            c = rec["kmap"].get(k)
            if c is None or not rec["ans"][c]:
                continue
            a = float((q @ rec["zstar"]) / rec["nz"])
            z = ETA * a * q
            n = E.ndcg(FI, z, rec)
            if n is not None:
                vals.append(n)
        popval[k] = float(np.mean(vals)) if vals else -1.0
    order = sorted(shared, key=lambda k: -popval[k])
    print(f"[P1*] top popval keys: {[(k, round(popval[k],3)) for k in order[:8]]}", flush=True)

    # ---- O-ans (population-value routing, realizable belief) ----
    def oans_pv(rec):
        z = np.zeros(FI["d"]); curve = []; asked = 0
        for k in order:
            if asked >= T:
                break
            c = rec["kmap"].get(k)
            if c is None or not rec["ans"][c]:
                continue
            a = float((rec["dir"][c] @ rec["zstar"]) / rec["nz"])
            z = z + ETA * a * rec["dir"][c]; asked += 1
            curve.append(E.ndcg(FI, z, rec))
        while len(curve) < T:                      # ran out of answerable shared questions
            curve.append(E.ndcg(FI, z, rec) if len(curve) == 0 else curve[-1])
        return curve

    # static B from e0_gonogo (identical machinery); O-full endpoint = 0.5681 from the G2 repro (context)
    rows = {"oans_pv": [], "static": []}
    for rec in U:
        rows["oans_pv"].append(oans_pv(rec))
        rows["static"].append(E.sel_static(FI, rec, E.BLIND_SCHEDULE))

    def clean(name):
        keep = [i for i in range(len(U))
                if all(rows[n][i][t] is not None for n in rows for t in range(T))]
        return keep
    idx = [i for i in range(len(U))
           if all(rows[n][i][t] is not None for n in rows for t in range(T))]
    n = len(idx)

    def anytime_arr(name):
        return np.array([np.mean(rows[name][i]) for i in idx])
    def end_arr(name):
        return np.array([rows[name][i][-1] for i in idx])
    def per_turn(name):
        return [float(np.mean([rows[name][i][t] for i in idx])) for t in range(T)]

    def boot(diff, nb=5000):
        rng = np.random.default_rng(SEED)
        b = np.array([diff[rng.integers(0, len(diff), len(diff))].mean() for _ in range(nb)])
        return float(diff.mean()), [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))], float((b > 0).mean())

    at = {k: anytime_arr(k) for k in rows}; en = {k: end_arr(k) for k in rows}
    d_any, ci_any, p_any = boot(at["oans_pv"] - at["static"])
    d_end, ci_end, p_end = boot(en["oans_pv"] - en["static"])

    out = dict(
        n_users=n,
        value_model="population single-question cohort-NDCG (value-when-answered); O-ans routes to this "
                    "user's answerable questions in descending population value; realizable belief; "
                    "same shared pool B was built from; z*-free selection.",
        anytime=dict(oans_pv=float(at["oans_pv"].mean()), static=float(at["static"].mean())),
        endpoint=dict(oans_pv=float(en["oans_pv"].mean()), static=float(en["static"].mean()),
                      ofull_ref=0.5681),
        per_turn_ndcg=dict(oans_pv=per_turn("oans_pv"), static=per_turn("static")),
        prize_anytime=dict(delta=d_any, ci95=ci_any, p_gt0=p_any, threshold=0.015,
                           GO=bool(ci_any[0] > 0 and d_any >= 0.015)),
        prize_endpoint=dict(delta=d_end, ci95=ci_end, p_gt0=p_end),
        popval_order_top=[(k, round(popval[k], 4)) for k in order[:12]])
    json.dump(out, open(OUT, "w"), indent=1, default=str)
    print(f"[P1*] O-ans_pv anytime={out['anytime']['oans_pv']:.4f} static={out['anytime']['static']:.4f}",
          flush=True)
    print(f"[P1*] prize anytime Δ={d_any:.4f} CI={ci_any} GO={out['prize_anytime']['GO']}", flush=True)
    print(f"[P1*] prize endpoint Δ={d_end:.4f} CI={ci_end}", flush=True)
    print(f"[P1*] endpoint O-ans_pv={out['endpoint']['oans_pv']:.4f} static={out['endpoint']['static']:.4f} "
          f"(O-full ref 0.5681)", flush=True)
    print(f"[saved] {OUT}", flush=True)


if __name__ == "__main__":
    main()
