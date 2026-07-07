"""E0f MAIN RUNS — descent under the NON-CIRCULAR real-rating item answer model.

Run AFTER e0f_control.py. NOTE: the mandatory fold-sanity control FAILED (a single real-rating item
from cold HURTS, and even the OLD geometric single item does not help at the calibrated eta=16 — a
fold-miscalibration diagnosed in e0f_fold_diag.py / _diag2.py). These main runs are therefore reported
under that caveat: they quantify the downstream descent verdict and test whether a CORRECTED per-item
eta rescues it.

Item channel = the user's KNOWN-portion RATED items (trivially answerable per study-design §3), answered
with the real rating (centered on the user's known-profile mean, rescaled by the global BETA that RMS-
matches the geometric item answers, + sigma=0.70-star fidelity noise). Concepts unchanged (geometric),
so the comparison isolates the ITEM VALUE channel. Static B / static+skip stay fully geometric = refs.

NO LLM calls.  Run: python scripts/e0f_main.py
"""
import os, sys, json
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "instrument2"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import warnings; warnings.filterwarnings("ignore")
import llm_answerability_gate as G
import e0_gonogo as E
import e0f_lib as L

OUT = "experiments/E0f_main.json"
ETA = E.ETA          # 16.0 (canonical fold step)
T = E.T              # 8
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
    L.augment_users(D, split, U)
    beta, bstats = L.compute_beta(FI, U)
    Wn = FI["Wn"]; C = FI["C"]; B = E.BLIND_SCHEDULE
    concept_sched = [k for k in B if k.startswith("C")]
    dgv_by_u = {int(us): np.asarray(v["dg_vec"]) for us, v in per_user.items()}
    for rec in U:
        rec["kmap"] = {k: c for c, k in enumerate(rec["key"])}
    print(f"[E0f-main] {len(U)} users; beta={beta:.4f}; concept phase order={concept_sched}", flush=True)

    # ---- precompute per-user rated-item value tables (one-shot NDCG improvement) ----
    # For each user, for each known-rated item: one-shot NDCG from cold under real (eta16, eta1) and geo.
    # This is a PER-USER ONE-SHOT ORACLE value (peeks at held-out targets) = upper-bound ranking.
    for rec in U:
        rows = []
        for j, r in rec["known_rated"]:
            q = Wn[j]
            a_real = L.real_answer(rec, j, r, beta)
            a_geo = L.geo_answer(rec, q)
            n_r16 = E.ndcg(FI, ETA * a_real * q, rec)
            n_r1 = E.ndcg(FI, 1.0 * a_real * q, rec)
            n_g16 = E.ndcg(FI, ETA * a_geo * q, rec)
            feats = E.cand_features(D, "item", {"j": j}, dgv_by_u.get(rec["u"], np.zeros(len(G.GENRES))), cpct)
            phat = float(pm.p(feats)[0])
            rows.append(dict(j=j, r=r, a_real=a_real, a_geo=a_geo, div=float(q @ C @ q), phat=phat,
                             one_real16=(n_r16 or 0.0), one_real1=(n_r1 or 0.0), one_geo16=(n_g16 or 0.0),
                             abscen=abs(r - rec["known_mean"])))
        rec["ritems"] = rows

    def phase1(rec, k):
        """k schedule concepts (geometric); turn consumed even if refused; returns z, curve list."""
        z = np.zeros(FI["d"]); curve = []
        for t in range(k):
            key = concept_sched[t] if t < len(concept_sched) else None
            if key is not None:
                c = rec["kmap"].get(key)
                if c is not None and rec["ans"][c]:
                    q = rec["dir"][c]; a = float((q @ rec["zstar"]) / rec["nz"]); z = z + ETA * a * q
            curve.append(E.ndcg(FI, z, rec))
        return z, curve

    def descent(rec, k, rank, answer, eta_item=ETA):
        """Phase2: (T-k) rated items ranked by `rank`, answered by `answer` (real|geo), folded at eta_item."""
        z, curve = phase1(rec, k)
        rows = rec["ritems"]
        if rank == "oracle":
            key = ("one_real16" if answer == "real" else "one_geo16")
            order = sorted(rows, key=lambda x: -x[key])
        elif rank == "abscen":
            order = sorted(rows, key=lambda x: -x["abscen"])
        elif rank == "phat":
            order = sorted(rows, key=lambda x: -(x["phat"] * max(x["abscen"], 0.0)))
        else:
            order = rows
        need = T - k
        for row in order[:need]:
            q = Wn[row["j"]]; a = row["a_real"] if answer == "real" else row["a_geo"]
            z = z + eta_item * a * q
            curve.append(E.ndcg(FI, z, rec))
        while len(curve) < T:
            curve.append(curve[-1] if curve else E.ndcg(FI, np.zeros(FI["d"]), rec))
        return curve

    def emergent(rec, use_div=True, answer="real"):
        """Marginal-info x true-table x div over {bank concepts} U {known-rated items}; real item answers."""
        # candidate list: (dir, answer_value, is_item, div)
        cand = []
        for c, k in enumerate(rec["key"]):
            if k.startswith("C") and rec["ans"][c]:
                q = rec["dir"][c]; cand.append((q, float((q @ rec["zstar"]) / rec["nz"]), False, float(rec["div"][c])))
        for row in rec["ritems"]:
            q = Wn[row["j"]]; a = row["a_real"] if answer == "real" else row["a_geo"]
            cand.append((q, a, True, row["div"]))
        if not cand:
            return [E.ndcg(FI, np.zeros(FI["d"]), rec)] * T
        dirs = np.array([c[0] for c in cand]); avals = np.array([c[1] for c in cand])
        divs = np.array([max(c[3], 0.0) for c in cand])
        z = np.zeros(FI["d"]); basis = []; used = set(); curve = []
        for _ in range(T):
            avail = [i for i in range(len(cand)) if i not in used]
            if not avail:
                curve.append(E.ndcg(FI, z, rec)); continue
            Dv = dirs[avail]
            if basis:
                Bm = np.array(basis); resid = Dv - (Dv @ Bm.T) @ Bm
            else:
                resid = Dv
            mi = np.linalg.norm(resid, axis=1)
            score = mi * (divs[avail] if use_div else 1.0)
            bi = avail[int(np.argmax(score))]; used.add(bi)
            q = dirs[bi]; z = z + ETA * avals[bi] * q
            v = q.astype(np.float64).copy()
            for b in basis:
                v = v - (v @ b) * b
            nv = np.linalg.norm(v)
            if nv > 1e-8:
                basis.append(v / nv)
            curve.append(E.ndcg(FI, z, rec))
        return curve

    # baselines (geometric)
    def run_skip(rec):
        keydir, popval, popn = {}, {}, {}
        return None
    # popval priority for skip (shared bank keys) — same as E0d
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
            a = float((q @ rec["zstar"]) / rec["nz"]); n = E.ndcg(FI, ETA * a * q, rec)
            if n is not None:
                vals.append(n)
        popval[k] = float(np.mean(vals)) if vals else -1.0; popn[k] = len(vals)
    priority = list(B) + [k for k in sorted([kk for kk in keydir if popn[kk] > 0], key=lambda kk: -popval[kk]) if k not in B]

    def skip(rec):
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

    arms = {}
    arms["static"] = [E.sel_static(FI, rec, B) for rec in U]
    arms["static_skip"] = [skip(rec) for rec in U]
    # --- REALIZABLE descent arms (no target peek): data-side |centered rating| or p-hat ranking ---
    arms["descent_k4_real_datarank"] = [descent(r, 4, "abscen", "real") for r in U]
    arms["descent_k4_real_datarank_eta1"] = [descent(r, 4, "abscen", "real", eta_item=1.0) for r in U]
    arms["descent_k4_real_phat"] = [descent(r, 4, "phat", "real") for r in U]
    arms["descent_k3_real_datarank"] = [descent(r, 3, "abscen", "real") for r in U]
    arms["descent_k5_real_datarank"] = [descent(r, 5, "abscen", "real") for r in U]
    arms["emergent_true_div_real"] = [emergent(r, True, "real") for r in U]
    # --- PER-USER TARGET-PEEK UPPER BOUNDS (NOT realizable: rank rated items by own held-out NDCG) ---
    arms["ub_descent_k4_real_peekrank"] = [descent(r, 4, "oracle", "real") for r in U]
    arms["ub_descent_k4_real_peekrank_eta1"] = [descent(r, 4, "oracle", "real", eta_item=1.0) for r in U]
    arms["ub_descent_k4_geo_peekrank_samePool"] = [descent(r, 4, "oracle", "geo") for r in U]
    arms["ub_all8_real_peekrank"] = [descent(r, 0, "oracle", "real") for r in U]

    idx = [i for i in range(len(U)) if all(arms[n][i][t] is not None for n in arms for t in range(T))]
    n = len(idx)
    print(f"[E0f-main] {n} users with all arms defined", flush=True)

    def anyt(name): return np.array([np.mean(arms[name][i]) for i in idx])
    def endp(name): return np.array([arms[name][i][-1] for i in idx])
    def pturn(name): return [float(np.mean([arms[name][i][t] for i in idx])) for t in range(T)]

    st_any, st_end = anyt("static"), endp("static")
    sk_any, sk_end = anyt("static_skip"), endp("static_skip")
    res = {"n_users": n, "beta": beta, "beta_stats": bstats,
           "reproduced": bool(abs(st_any.mean() - 0.2517) < 0.002 and abs(st_end.mean() - 0.2812) < 0.002),
           "arms": {}}
    for name in arms:
        a_any, a_end = anyt(name), endp(name)
        dB_any, ciB_any, pB_any = boot(a_any - st_any)
        dB_end, ciB_end, _ = boot(a_end - st_end)
        dS_any, ciS_any, _ = boot(a_any - sk_any)
        res["arms"][name] = dict(
            anytime=float(a_any.mean()), endpoint=float(a_end.mean()), per_turn=pturn(name),
            vs_static=dict(anytime_delta=dB_any, anytime_ci=ciB_any, anytime_p_gt0=pB_any,
                           endpoint_delta=dB_end, endpoint_ci=ciB_end),
            vs_skip=dict(anytime_delta=dS_any, anytime_ci=ciS_any))
    os.makedirs("experiments", exist_ok=True)
    json.dump(res, open(OUT, "w"), indent=1, default=str)

    print(f"\n[E0f] static B    anytime={st_any.mean():.4f} end={st_end.mean():.4f} reproduced={res['reproduced']}")
    print(f"[E0f] static+skip anytime={sk_any.mean():.4f} end={sk_end.mean():.4f}")
    for name in arms:
        if name in ("static", "static_skip"):
            continue
        r = res["arms"][name]
        print(f"[E0f] {name:32s} any={r['anytime']:.4f} end={r['endpoint']:.4f} | "
              f"vsB any d={r['vs_static']['anytime_delta']:+.4f} "
              f"CI[{r['vs_static']['anytime_ci'][0]:+.4f},{r['vs_static']['anytime_ci'][1]:+.4f}]")
    print(f"[saved] {OUT}", flush=True)


if __name__ == "__main__":
    main()
