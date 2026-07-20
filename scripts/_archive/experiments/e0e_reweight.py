"""E0e P3 — FOLD RE-WEIGHTING quick probe (the owner's "re-weigh them" ask, fold side).

Down-weight redundant answer tokens when building the belief: fold the t-th answered direction q_t
with weight w_t = marginal_info(q_t | span of PREVIOUSLY answered dirs) = || q_t - proj ||, so a near-
duplicate (low-marginal) concept token contributes less to z. Targets the non-monotone-fold dilution
directly. Applied to BOTH static B AND the descent_k4_true_pop arm (fair comparison).

  reweighted fold:  z <- z + eta * a * w_t * q_t ,  w_t = ||q_t - proj_{answered span} q_t||
  (baseline fold:   z <- z + eta * a * q_t)

REUSES e0_gonogo + e0d machinery. NO LLM calls.  Run: python scripts/e0e_reweight.py
"""
import os, sys, json, collections
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "instrument2"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import warnings; warnings.filterwarnings("ignore")
import llm_answerability_gate as G
import e0_gonogo as E

OUT = "experiments/E0e_reweight.json"
SEED = 0


def boot(diff, nb=5000):
    rng = np.random.default_rng(SEED)
    b = np.array([diff[rng.integers(0, len(diff), len(diff))].mean() for _ in range(nb)])
    return float(diff.mean()), [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))], float((b > 0).mean())


def gram_add(basis, q):
    v = q.astype(np.float64).copy()
    for b in basis:
        v = v - (v @ b) * b
    nv = np.linalg.norm(v)
    if nv > 1e-8:
        basis.append(v / nv)
    return basis


def marg(q, basis):
    v = q.astype(np.float64).copy()
    for b in basis:
        v = v - (v @ b) * b
    return float(np.linalg.norm(v))


def main():
    D = G.load_data()
    FI = E.build_foldin(D)
    pm = E.PModel(E.PMODEL_JSON)
    cpct = E.concept_pct_vec(D)
    per_user, split = E.per_user_from_gate(D)
    U = E.assemble_users(D, per_user, split, FI, pm, cpct)
    T, ETA = E.T, E.ETA
    B = E.BLIND_SCHEDULE
    for rec in U:
        rec["kmap"] = {k: c for c, k in enumerate(rec["key"])}
        rec["ansset"] = {k: bool(rec["ans"][c]) for c, k in enumerate(rec["key"])}
    print(f"[reweight] {len(U)} users", flush=True)

    # popval for descent item ranking (same as e0d)
    keydir = {}
    for rec in U:
        for c, k in enumerate(rec["key"]):
            keydir.setdefault(k, rec["dir"][c])
    popval = {}
    for k, q in keydir.items():
        vals = []
        for rec in U:
            c = rec["kmap"].get(k)
            if c is None or not rec["ans"][c]:
                continue
            a = float((q @ rec["zstar"]) / rec["nz"]); n = E.ndcg(FI, ETA * a * q, rec)
            if n is not None:
                vals.append(n)
        popval[k] = float(np.mean(vals)) if vals else -1.0
    concept_sched = [k for k in B if k.startswith("C")]

    def fold(rec, keyseq, reweight):
        """Ask keys in keyseq (skip unanswerable, turn NOT consumed for descent items list already
        filtered; for the fixed schedule a refused turn IS consumed). Here keyseq entries are (key, fixed)
        where fixed=True means it's a hard schedule slot (turn consumed on refusal)."""
        z = np.zeros(FI["d"]); basis = []; curve = []
        for key, fixed in keyseq:
            if len(curve) >= T:
                break
            c = rec["kmap"].get(key)
            answerable = c is not None and rec["ans"][c]
            if answerable:
                q = rec["dir"][c]; a = float((q @ rec["zstar"]) / rec["nz"])
                w = marg(q, basis) if reweight else 1.0
                z = z + ETA * a * w * q
                basis = gram_add(basis, q)
                curve.append(E.ndcg(FI, z, rec))
            elif fixed:
                curve.append(E.ndcg(FI, z, rec))          # refused fixed slot: turn consumed
        while len(curve) < T:
            curve.append(curve[-1] if curve else E.ndcg(FI, z, rec))
        return curve

    def static_seq(rec):
        return [(k, True) for k in B]

    def descent_seq(rec, k=4):
        seq = [(concept_sched[t], True) for t in range(k)]
        items = [key for key, a in rec["ansset"].items() if key.startswith("I:") and a]
        items.sort(key=lambda kk: -popval.get(kk, -1.0))
        seq += [(key, False) for key in items[:T - k]]
        return seq

    arms = {}
    arms["static_base"]   = [fold(r, static_seq(r), False) for r in U]
    arms["static_rw"]     = [fold(r, static_seq(r), True) for r in U]
    arms["descent_base"]  = [fold(r, descent_seq(r), False) for r in U]
    arms["descent_rw"]    = [fold(r, descent_seq(r), True) for r in U]

    idx = [i for i in range(len(U)) if all(arms[n][i][t] is not None for n in arms for t in range(T))]
    n = len(idx)

    def anyt(name): return np.array([np.mean(arms[name][i]) for i in idx])
    def endp(name): return np.array([arms[name][i][-1] for i in idx])
    def pturn(name): return [float(np.mean([arms[name][i][t] for i in idx])) for t in range(T)]

    res = {"n_users": n, "arms": {k: dict(anytime=float(anyt(k).mean()), endpoint=float(endp(k).mean()),
                                          per_turn=pturn(k)) for k in arms}}
    # reweight effect on each schedule
    for base, rw, tag in [("static_base", "static_rw", "static"),
                          ("descent_base", "descent_rw", "descent")]:
        d_any, ci_any, p_any = boot(anyt(rw) - anyt(base))
        d_end, ci_end, _ = boot(endp(rw) - endp(base))
        res[f"reweight_effect_{tag}"] = dict(anytime_delta=d_any, anytime_ci=ci_any, anytime_p_gt0=p_any,
                                             endpoint_delta=d_end, endpoint_ci=ci_end)
    json.dump(res, open(OUT, "w"), indent=1, default=str)
    print(f"[reweight] n={n}")
    for k in arms:
        print(f"  {k:16s} any={res['arms'][k]['anytime']:.4f} end={res['arms'][k]['endpoint']:.4f}")
    for tag in ("static", "descent"):
        e = res[f"reweight_effect_{tag}"]
        print(f"  reweight effect {tag:8s} any d={e['anytime_delta']:+.4f} CI{[round(x,4) for x in e['anytime_ci']]} "
              f"end d={e['endpoint_delta']:+.4f}")
    print(f"[saved] {OUT}", flush=True)


if __name__ == "__main__":
    main()
