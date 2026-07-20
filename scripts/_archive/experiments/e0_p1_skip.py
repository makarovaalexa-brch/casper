"""E0 P1 (faithful, diversified) — static+skip: the cheapest realizable answerability router.

Both value-ranked O-ans variants (divisiveness; population value-when-answered) lose to the STRONG
static B because B is a DIVERSIFIED greedy-forward schedule and single-question value ranking is
redundant. The design's pre-registered diversified realizable router is static+skip ("conditional
static"): take B's own greedy-forward PRIORITY LIST and, per user, skip questions this user cannot
answer, refunding the turn to the next answerable item on the list. It uses B's exact diversified
value machinery + the answerability table + a realizable belief path (z from answers). z*-free.

static+skip − static = the cheapest realizable answerability-routing prize (design rung 7 vs 8; the
baseline "most likely to embarrass the agent"). If even this is ~0, the T=8 answerability prize is
captured by nothing beyond the population schedule.

We extend B's greedy-forward order to length 16 (same cohort-mean-NDCG objective, per-user answer
gating, exactly as g2_exploitability builds B), then per user ask the first 8 answerable keys.
Reuses e0_gonogo assembly + NDCG. NO LLM calls.
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
LGREEDY = 16


def main():
    D = G.load_data()
    FI = E.build_foldin(D)
    pm = E.PModel(E.PMODEL_JSON)
    cpct = E.concept_pct_vec(D)
    per_user, split = E.per_user_from_gate(D)
    U = E.assemble_users(D, per_user, split, FI, pm, cpct)
    T, ETA = E.T, E.ETA
    print(f"[skip] {len(U)} users", flush=True)

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
        rec["kmap"] = {k: c for c, k in enumerate(rec["key"])}
    print(f"[skip] shared pool = {len(shared)}; building greedy-forward order to {LGREEDY} ...", flush=True)

    def cohort_mean_ndcg(schedule):
        tot, m = 0.0, 0
        for rec in U:
            z = np.zeros(FI["d"])
            for k in schedule:
                if rec["kset"].get(k, False):
                    q = key_dir[k]; av = float((q @ rec["zstar"]) / rec["nz"])
                    z = z + ETA * av * q
            n = E.ndcg(FI, z, rec)
            if n is not None:
                tot += n; m += 1
        return tot / max(m, 1)

    schedule = []
    for pos in range(LGREEDY):
        best, bk = -1.0, None
        for k in shared:
            if k in schedule:
                continue
            v = cohort_mean_ndcg(schedule + [k])
            if v > best:
                best, bk = v, k
        if bk is None:
            break
        schedule.append(bk)
        print(f"  pos{pos+1}: +{bk} cohortNDCG={best:.4f}", flush=True)
    print(f"[skip] greedy16 = {schedule}", flush=True)

    # plain static (first 8) sanity + static+skip
    def run_static(rec, sched8):
        z = np.zeros(FI["d"]); curve = []
        for t in range(T):
            k = sched8[t] if t < len(sched8) else None
            if k is not None and rec["kset"].get(k, False):
                q = key_dir[k]; a = float((q @ rec["zstar"]) / rec["nz"]); z = z + ETA * a * q
            curve.append(E.ndcg(FI, z, rec))
        return curve

    def run_skip(rec, order):
        z = np.zeros(FI["d"]); curve = []; asked = 0
        for k in order:
            if asked >= T:
                break
            if rec["kset"].get(k, False):
                q = key_dir[k]; a = float((q @ rec["zstar"]) / rec["nz"]); z = z + ETA * a * q
                asked += 1; curve.append(E.ndcg(FI, z, rec))
        while len(curve) < T:
            curve.append(curve[-1] if curve else E.ndcg(FI, np.zeros(FI["d"]), rec))
        return curve

    sched8 = schedule[:8]
    rows = {"static": [], "skip": []}
    for rec in U:
        rows["static"].append(run_static(rec, sched8))
        rows["skip"].append(run_skip(rec, schedule))
    idx = [i for i in range(len(U)) if all(rows[n][i][t] is not None for n in rows for t in range(T))]
    n = len(idx)

    def anyt(name): return np.array([np.mean(rows[name][i]) for i in idx])
    def endp(name): return np.array([rows[name][i][-1] for i in idx])
    def pturn(name): return [float(np.mean([rows[name][i][t] for i in idx])) for t in range(T)]

    def boot(diff, nb=5000):
        rng = np.random.default_rng(SEED)
        b = np.array([diff[rng.integers(0, len(diff), len(diff))].mean() for _ in range(nb)])
        return float(diff.mean()), [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))], float((b > 0).mean())

    a_s, a_k = anyt("static"), anyt("skip"); e_s, e_k = endp("static"), endp("skip")
    d_any, ci_any, p_any = boot(a_k - a_s)
    d_end, ci_end, p_end = boot(e_k - e_s)

    # refusal accounting: mean # of B's first-8 that each user can answer
    ans8 = np.mean([sum(1 for k in sched8 if rec["kset"].get(k, False)) for rec in U])

    out = dict(
        n_users=n, greedy_order=schedule, static_schedule8=sched8,
        mean_answerable_of_static8=float(ans8),
        anytime=dict(static=float(a_s.mean()), static_skip=float(a_k.mean()), ofull_ref=0.5256),
        endpoint=dict(static=float(e_s.mean()), static_skip=float(e_k.mean()), ofull_ref=0.5681),
        per_turn_ndcg=dict(static=pturn("static"), static_skip=pturn("skip")),
        prize_anytime=dict(delta=d_any, ci95=ci_any, p_gt0=p_any, threshold=0.015,
                           GO=bool(ci_any[0] > 0 and d_any >= 0.015)),
        prize_endpoint=dict(delta=d_end, ci95=ci_end, p_gt0=p_end))
    json.dump(out, open(OUT, "w"), indent=1, default=str)
    print(f"[skip] static anytime={a_s.mean():.4f} end={e_s.mean():.4f} | "
          f"static+skip anytime={a_k.mean():.4f} end={e_k.mean():.4f}", flush=True)
    print(f"[skip] PRIZE anytime d={d_any:.4f} CI={ci_any} GO={out['prize_anytime']['GO']} | "
          f"endpoint d={d_end:.4f} CI={ci_end}", flush=True)
    print(f"[skip] mean answerable of static8 = {ans8:.2f}/8", flush=True)
    print(f"[saved] {OUT}", flush=True)


if __name__ == "__main__":
    main()
