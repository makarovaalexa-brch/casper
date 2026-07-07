"""E0f fold diagnosis v2 — WHY does the real rating hurt even at eta=1 and noiseless?

The fold operator z' = z + eta*a*q treats the answer `a` as the coordinate of TRUE taste z* along the
question direction q (that is exactly what a=cos(z*,q) is). For a real rating this holds only if the
centered rating is POSITIVELY aligned with cos(z*, Wn[j]). We measure that correlation directly, and
run the real answer NOISELESS to separate the sigma=0.70 fidelity noise from the operator mismatch.

NO LLM calls.  Run: python scripts/e0f_fold_diag2.py
"""
import os, sys, json
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "instrument2"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import warnings; warnings.filterwarnings("ignore")
import llm_answerability_gate as G
import e0_gonogo as E
import e0f_lib as L


def boot(diff, nb=4000):
    rng = np.random.default_rng(0)
    b = np.array([diff[rng.integers(0, len(diff), len(diff))].mean() for _ in range(nb)])
    return float(diff.mean()), [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]


def main():
    D = G.load_data()
    FI = E.build_foldin(D)
    pm = E.PModel(E.PMODEL_JSON)
    cpct = E.concept_pct_vec(D)
    per_user, split = E.per_user_from_gate(D)
    U = E.assemble_users(D, per_user, split, FI, pm, cpct)
    L.augment_users(D, split, U)
    beta, bstats = L.compute_beta(FI, U)
    Wn = FI["Wn"]
    res = {"beta": beta}

    # correlation between centered real rating and cos(z*, Wn[j]) over ALL known-rated pairs
    rc, ag = [], []
    for rec in U:
        for j, r in rec["known_rated"]:
            rc.append(r - rec["known_mean"])
            ag.append(float((Wn[j] @ rec["zstar"]) / rec["nz"]))
    rc = np.array(rc); ag = np.array(ag)
    res["corr_centered_rating_vs_cosZstar"] = float(np.corrcoef(rc, ag)[0, 1])
    res["frac_signs_agree"] = float(np.mean(np.sign(rc) == np.sign(ag)))
    res["mean_cos_when_like_r>=4"] = float(np.mean([ag[i] for i in range(len(ag)) if rc[i] > 0]))
    res["mean_cos_when_dislike"] = float(np.mean([ag[i] for i in range(len(ag)) if rc[i] < 0]))

    # deterministic per-user pick (same seed as control)
    pick = {}
    for rec in U:
        if rec["known_rated"]:
            rng = np.random.default_rng(1234 + rec["u"])
            pick[rec["u"]] = rec["known_rated"][int(rng.integers(len(rec["known_rated"])))]

    etas = [0.5, 1, 2, 4, 8, 16]
    variants = {}
    for name, noisy in (("real_noisy", True), ("real_noiseless", False)):
        variants[name] = {}
        for eta in etas:
            dd = []
            for rec in U:
                if rec["u"] not in pick:
                    continue
                b0 = E.ndcg(FI, np.zeros(FI["d"]), rec)
                if b0 is None:
                    continue
                j, r = pick[rec["u"]]; q = Wn[j]
                a = L.real_answer(rec, j, r, beta, add_noise=noisy)
                n = E.ndcg(FI, eta * a * q, rec)
                if n is not None:
                    dd.append(n - b0)
            m, ci = boot(np.array(dd))
            variants[name][str(eta)] = dict(mean_delta=m, ci95=ci, n=len(dd))
    res["real_eta_sweep_noise_vs_noiseless"] = variants

    os.makedirs("experiments", exist_ok=True)
    json.dump(res, open("experiments/E0f_fold_diag2.json", "w"), indent=1, default=str)
    print(f"corr(centered rating, cos(z*,Wn[j])) = {res['corr_centered_rating_vs_cosZstar']:+.4f}")
    print(f"sign-agreement = {res['frac_signs_agree']:.3f}; "
          f"mean cos when like={res['mean_cos_when_like_r>=4']:+.4f} dislike={res['mean_cos_when_dislike']:+.4f}")
    for name in ("real_noisy", "real_noiseless"):
        row = " ".join(f"eta{e}={variants[name][str(e)]['mean_delta']:+.4f}" for e in etas)
        print(f"{name:15s}: {row}")


if __name__ == "__main__":
    main()
