"""E0f MANDATORY CONTROL — the fold-sanity check (run FIRST, before any main run).

From the cold belief z=0, fold ONE answer for a random answerable+RATED known item at t=1 and compare
NDCG@10 vs asking nothing (z=0). A free TRUE answer must HELP on average. Run the same control with the
OLD geometric item answer. If the real-rating control HURTS, the main runs are aborted and the fold
scaling for item tokens (eta, direction norm, BETA) is the primary deliverable to diagnose.

Also reproduces the E0d baselines (static B 0.2517/0.2812, static+skip 0.2516/0.2796) up front.

NO LLM calls.  Run:  python scripts/e0f_control.py
"""
import os, sys, json
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "instrument2"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import warnings; warnings.filterwarnings("ignore")
import llm_answerability_gate as G
import e0_gonogo as E
import e0f_lib as L

OUT = "experiments/E0f_control.json"
SEED = 0


def boot(diff, nb=5000):
    rng = np.random.default_rng(SEED)
    b = np.array([diff[rng.integers(0, len(diff), len(diff))].mean() for _ in range(nb)])
    return float(diff.mean()), [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))], float((b > 0).mean())


def reproduce_baselines(FI, U):
    """static B + static+skip per-turn (geometric), matched user set — must hit 0.2517/0.2812, 0.2516/0.2796."""
    T = E.T
    B = E.BLIND_SCHEDULE
    for rec in U:
        rec["kmap"] = {k: c for c, k in enumerate(rec["key"])}
    # popval priority for static+skip (same construction as E0d)
    keydir, popval, popn = {}, {}, {}
    for rec in U:
        for c, k in enumerate(rec["key"]):
            keydir.setdefault(k, rec["dir"][c])
    for k, q in keydir.items():
        vals = []
        for rec in U:
            c = rec["kmap"].get(k)
            if c is None or not rec["ans"][c]:
                continue
            a = float((q @ rec["zstar"]) / rec["nz"])
            n = E.ndcg(FI, E.ETA * a * q, rec)
            if n is not None:
                vals.append(n)
        popval[k] = float(np.mean(vals)) if vals else -1.0
        popn[k] = len(vals)
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
                q = rec["dir"][c]; a = float((q @ rec["zstar"]) / rec["nz"]); z = z + E.ETA * a * q
                asked += 1; curve.append(E.ndcg(FI, z, rec))
        while len(curve) < T:
            curve.append(curve[-1] if curve else E.ndcg(FI, np.zeros(FI["d"]), rec))
        return curve

    st = [run_static(r) for r in U]
    sk = [run_skip(r) for r in U]
    idx = [i for i in range(len(U)) if all(st[i][t] is not None and sk[i][t] is not None for t in range(T))]
    st_any = float(np.mean([np.mean(st[i]) for i in idx])); st_end = float(np.mean([st[i][-1] for i in idx]))
    sk_any = float(np.mean([np.mean(sk[i]) for i in idx])); sk_end = float(np.mean([sk[i][-1] for i in idx]))
    return dict(n=len(idx), static_anytime=st_any, static_endpoint=st_end,
                skip_anytime=sk_any, skip_endpoint=sk_end,
                reproduced=bool(abs(st_any - 0.2517) < 0.002 and abs(st_end - 0.2812) < 0.002))


def main():
    D = G.load_data()
    FI = E.build_foldin(D)
    pm = E.PModel(E.PMODEL_JSON)
    cpct = E.concept_pct_vec(D)
    per_user, split = E.per_user_from_gate(D)
    U = E.assemble_users(D, per_user, split, FI, pm, cpct)
    L.augment_users(D, split, U)
    beta, bstats = L.compute_beta(FI, U)
    print(f"[E0f-control] {len(U)} eligible users; BETA={beta:.4f} "
          f"(RMS_geo={bstats['rms_geo']:.4f}, RMS_rating_centered={bstats['rms_rating_centered']:.4f}, "
          f"n_pairs={bstats['n_pairs']})", flush=True)

    repro = reproduce_baselines(FI, U)
    print(f"[E0f-control] baselines: static {repro['static_anytime']:.4f}/{repro['static_endpoint']:.4f} "
          f"skip {repro['skip_anytime']:.4f}/{repro['skip_endpoint']:.4f} "
          f"reproduced={repro['reproduced']}", flush=True)

    Wn = FI["Wn"]
    d_real, d_geo, d_real_likes = [], [], []
    nrated_avail = 0
    for rec in U:
        kr = rec["known_rated"]
        if not kr:
            continue
        nrated_avail += 1
        base = E.ndcg(FI, np.zeros(FI["d"]), rec)     # asking nothing
        if base is None:
            continue
        rng = np.random.default_rng(1234 + rec["u"])  # deterministic per-user pick
        j, r = kr[int(rng.integers(len(kr)))]
        q = Wn[j]
        a_real = L.real_answer(rec, j, r, beta)
        a_geo = L.geo_answer(rec, q)
        n_real = E.ndcg(FI, E.ETA * a_real * q, rec)
        n_geo = E.ndcg(FI, E.ETA * a_geo * q, rec)
        if n_real is not None:
            d_real.append(n_real - base)
            if r >= 4.0:
                d_real_likes.append(n_real - base)
        if n_geo is not None:
            d_geo.append(n_geo - base)

    d_real = np.array(d_real); d_geo = np.array(d_geo); d_real_likes = np.array(d_real_likes)
    mr, cir, pr = boot(d_real)
    mg, cig, pg = boot(d_geo)
    ml, cil, pl = boot(d_real_likes) if len(d_real_likes) else (None, None, None)

    res = dict(
        n_users=len(U), n_with_rated=nrated_avail, beta=beta, beta_stats=bstats,
        baselines=repro,
        control_real=dict(mean_delta=mr, ci95=cir, p_gt0=pr, n=int(len(d_real)),
                          HELPS=bool(cir[0] > 0)),
        control_geo=dict(mean_delta=mg, ci95=cig, p_gt0=pg, n=int(len(d_geo)),
                         HELPS=bool(cig[0] > 0)),
        control_real_likes_only=dict(mean_delta=ml, ci95=cil, p_gt0=pl, n=int(len(d_real_likes)),
                                     HELPS=(bool(cil[0] > 0) if cil else None)),
        PASS=bool(cir[0] > 0))
    os.makedirs("experiments", exist_ok=True)
    json.dump(res, open(OUT, "w"), indent=1, default=str)

    print(f"\n[CONTROL] fold ONE answer from cold z=0 vs asking nothing (NDCG@10, n={len(d_real)}):")
    print(f"  REAL rating answer : d={mr:+.4f} CI[{cir[0]:+.4f},{cir[1]:+.4f}] P(>0)={pr:.3f} "
          f"HELPS={res['control_real']['HELPS']}")
    print(f"  GEOMETRIC answer   : d={mg:+.4f} CI[{cig[0]:+.4f},{cig[1]:+.4f}] P(>0)={pg:.3f} "
          f"HELPS={res['control_geo']['HELPS']}")
    if ml is not None:
        print(f"  REAL (likes r>=4) : d={ml:+.4f} CI[{cil[0]:+.4f},{cil[1]:+.4f}] n={len(d_real_likes)}")
    print(f"\n[CONTROL VERDICT] real-rating fold {'PASSES' if res['PASS'] else 'FAILS — ABORT MAIN'}.")
    print(f"[saved] {OUT}", flush=True)


if __name__ == "__main__":
    main()
