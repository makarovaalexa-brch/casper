"""E0 P1 static+skip (fast, faithful) — B's diversified schedule as priority head + popval-ordered fills.

The design's "conditional static": take B's frozen greedy-forward sequence and refund a refused
question's turn to the next item on the (extended) priority list. Priority list =
  B's 8-schedule (its diversification, unchanged)  ++  population-value order for the remaining keys.
Per user, walk the priority list, ask the answerable ones until 8 are asked, updating belief
realizably (z'=z+eta*a*q, a=cos(z*,q)). z*-free selection. This is the cheapest realizable
answerability router (design rung 7). static+skip − static = the cheapest realizable
answerability-routing prize.

Reuses e0_gonogo. NO LLM calls.
"""
import os, sys, json, collections
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "instrument2"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import warnings; warnings.filterwarnings("ignore")
import llm_answerability_gate as G
import e0_gonogo as E

OUT = "experiments/E0_p1_skip.json"
SEED = 0


def main():
    D = G.load_data()
    FI = E.build_foldin(D)
    pm = E.PModel(E.PMODEL_JSON)
    cpct = E.concept_pct_vec(D)
    per_user, split = E.per_user_from_gate(D)
    U = E.assemble_users(D, per_user, split, FI, pm, cpct)
    T, ETA = E.T, E.ETA
    B = E.BLIND_SCHEDULE
    print(f"[skip2] {len(U)} users", flush=True)

    keycount = collections.Counter(k for rec in U for k in rec["key"])
    shared = [k for k, c in keycount.items() if c >= max(2, int(0.5 * len(U)))]
    sset = set(shared)
    key_dir = {}
    for rec in U:
        for c, k in enumerate(rec["key"]):
            if k in sset and k not in key_dir:
                key_dir[k] = rec["dir"][c]
    for rec in U:
        rec["kset"] = {k: bool(rec["ans"][c]) for c, k in enumerate(rec["key"])}

    # popval[k] = mean over answerable users of single-question NDCG (value-when-answered)
    popval = {}
    for k in shared:
        q = key_dir[k]; vals = []
        for rec in U:
            if not rec["kset"].get(k, False):
                continue
            a = float((q @ rec["zstar"]) / rec["nz"]); n = E.ndcg(FI, ETA * a * q, rec)
            if n is not None:
                vals.append(n)
        popval[k] = float(np.mean(vals)) if vals else -1.0
    pop_order = sorted(shared, key=lambda k: -popval[k])
    priority = list(B) + [k for k in pop_order if k not in B]
    ans8 = float(np.mean([sum(1 for k in B if rec["kset"].get(k, False)) for rec in U]))
    print(f"[skip2] priority head={priority[:12]} | mean answerable of B8 = {ans8:.2f}/8", flush=True)

    def run_static(rec):
        z = np.zeros(FI["d"]); curve = []
        for t in range(T):
            k = B[t] if t < len(B) else None
            if k is not None and rec["kset"].get(k, False):
                q = key_dir[k]; a = float((q @ rec["zstar"]) / rec["nz"]); z = z + ETA * a * q
            curve.append(E.ndcg(FI, z, rec))
        return curve

    def run_skip(rec):
        z = np.zeros(FI["d"]); curve = []; asked = 0
        for k in priority:
            if asked >= T:
                break
            if rec["kset"].get(k, False):
                q = key_dir[k]; a = float((q @ rec["zstar"]) / rec["nz"]); z = z + ETA * a * q
                asked += 1; curve.append(E.ndcg(FI, z, rec))
        while len(curve) < T:
            curve.append(curve[-1] if curve else E.ndcg(FI, np.zeros(FI["d"]), rec))
        return curve

    rows = {"static": [run_static(r) for r in U], "skip": [run_skip(r) for r in U]}
    idx = [i for i in range(len(U)) if all(rows[n][i][t] is not None for n in rows for t in range(T))]

    def anyt(name): return np.array([np.mean(rows[name][i]) for i in idx])
    def endp(name): return np.array([rows[name][i][-1] for i in idx])
    def pturn(name): return [float(np.mean([rows[name][i][t] for i in idx])) for t in range(T)]

    def boot(diff, nb=5000):
        rng = np.random.default_rng(SEED)
        b = np.array([diff[rng.integers(0, len(diff), len(diff))].mean() for _ in range(nb)])
        return float(diff.mean()), [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))], float((b > 0).mean())

    a_s, a_k, e_s, e_k = anyt("static"), anyt("skip"), endp("static"), endp("skip")
    d_any, ci_any, p_any = boot(a_k - a_s)
    d_end, ci_end, p_end = boot(e_k - e_s)
    out = dict(
        n_users=len(idx), method="static+skip: B8 diversified head ++ popval-ordered fills; per-user "
        "skip unanswerable; realizable belief; z*-free.", priority_head=priority[:14],
        mean_answerable_of_static8=ans8,
        anytime=dict(static=float(a_s.mean()), static_skip=float(a_k.mean()), ofull_ref=0.5256),
        endpoint=dict(static=float(e_s.mean()), static_skip=float(e_k.mean()), ofull_ref=0.5681),
        per_turn_ndcg=dict(static=pturn("static"), static_skip=pturn("skip")),
        prize_anytime=dict(delta=d_any, ci95=ci_any, p_gt0=p_any, threshold=0.015,
                           GO=bool(ci_any[0] > 0 and d_any >= 0.015)),
        prize_endpoint=dict(delta=d_end, ci95=ci_end, p_gt0=p_end))
    json.dump(out, open(OUT, "w"), indent=1, default=str)
    print(f"[skip2] static anytime={a_s.mean():.4f} end={e_s.mean():.4f} | "
          f"static+skip anytime={a_k.mean():.4f} end={e_k.mean():.4f}", flush=True)
    print(f"[skip2] PRIZE anytime={d_any:.4f} CI={ci_any} GO={out['prize_anytime']['GO']} | "
          f"endpoint={d_end:.4f} CI={ci_end}", flush=True)
    print(f"[saved] {OUT}", flush=True)


if __name__ == "__main__":
    main()
