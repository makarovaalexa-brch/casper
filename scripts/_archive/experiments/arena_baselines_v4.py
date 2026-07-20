"""arena_baselines_v4.py -- rerun the standard cold-start baselines on the DE-OOD fold-v4, side by side
with fold-v3.1, on the SAME seed-123 devtest subsample (paired, fair). DEV/synthetic ONLY; the 173/300
LLM-judged users are NEVER touched. NO LLM calls. $0. Deterministic (seed 123).

Reuses arena_core (world), arena_policies (run_policy), and arena_baselines_v2 (RandomPolicy,
full_profile_ndcg, curve_means). For each fold we run: COLD, RANDOM (5 seeds), POPULARITY, ENTROPY,
INFO-GAIN (blind-EIG), to the headline endpoint turn 8. pop/entropy orders are fold-independent;
info-gain order is fold-derived (recomputed per fold -- honest deployment analog). random+entropy were
SEEN by fold-v4; popularity+info-gain were HELD OUT (the generalization headline).

THE QUESTION: does de-OOD-ing raise the honest elicitation lift and/or stop info-gain going negative?
"""
import os, sys, json, time, argparse
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE); sys.path.insert(0, os.path.join(_HERE, "instrument2"))
import warnings
warnings.filterwarnings("ignore")
import torch

import arena_core as AC
import arena_policies as AP
from arena_policies import StaticSeq, B0Cold, run_policy, train_pop_prior, prescreen_gains
from arena_baselines_v2 import RandomPolicy, full_profile_ndcg
from i25_fold_v4 import _sanitize_qemb
import i25_fold_v31 as FV31

RESULTS = ".cache/arena/fold_v4_results.json"
V4_CKPT = ".cache/i25_fold_v4_best.pt"
V31_CKPT = AC.FOLD_CKPT
MD = "experiments/ARENA_BUILD.md"
K = 10


def endpoint_curve(out, Tmax):
    c = out["curves"][K]
    return [float(np.nanmean(c[:, t])) for t in range(Tmax + 1)]


def load_fold(path):
    blob = torch.load(path, map_location="cpu")
    m = FV31.FoldV31(); m.load_state_dict(blob["model"]); m.eval()
    return m, blob.get("state", {})


def run_all(ar, dt, pop_order, ent_order, ig_order, seeds, Tmax):
    res = {}
    res["cold"] = endpoint_curve(run_policy(ar, dt, B0Cold(), Tmax, Ks=(K,)), Tmax)
    res["popularity"] = endpoint_curve(run_policy(ar, dt, StaticSeq("popularity", pop_order),
                                                   Tmax, Ks=(K,)), Tmax)
    res["entropy"] = endpoint_curve(run_policy(ar, dt, StaticSeq("entropy", ent_order),
                                               Tmax, Ks=(K,)), Tmax)
    res["infogain"] = endpoint_curve(run_policy(ar, dt, StaticSeq("infogain", ig_order),
                                                Tmax, Ks=(K,)), Tmax)
    rc = []
    for sd in seeds:
        rc.append(endpoint_curve(run_policy(ar, dt, RandomPolicy(sd), Tmax, Ks=(K,)), Tmax))
    rc = np.array(rc)
    res["random_mean"] = rc.mean(0).tolist(); res["random_std"] = rc.std(0).tolist()
    fp, _ = full_profile_ndcg(ar, dt)
    res["full_profile"] = fp
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_devtest", type=int, default=1200, help="devtest subsample (memory-light)")
    ap.add_argument("--n_train_sub", type=int, default=800)
    ap.add_argument("--random_seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--tmax", type=int, default=8)
    a = ap.parse_args()
    t0 = time.time()

    ar = AC.Arena()
    _sanitize_qemb(ar)
    # SAME seed-123 split as baselines_v2 (14000/3000/3000); take a devtest SUBSAMPLE for compute.
    coh = AC.make_cohorts(ar, n_train=14000, n_devval=3000, n_devtest=3000)
    train, dt_full = coh["train"], coh["devtest"]
    dt = dt_full[:a.n_devtest]
    print(f"[v4-base] split seed {AC.SEED}: train {len(train)} / devtest {len(dt_full)} "
          f"(using {len(dt)} devtest subsample); Tmax={a.tmax}", flush=True)

    train_sub = train[:a.n_train_sub]
    ar.prefill_answers(train_sub, "trainsub", dict(n_train=14000, n_devval=3000, n_devtest=3000,
                                                   sub=a.n_train_sub))
    ar.prefill_answers(dt, "v4devtest", dict(role="v4devtest", n=a.n_devtest))
    ar.set_pop_prior(train_pop_prior(ar, train_sub))

    # fold-independent orders
    pop_order = [int(q) for q in np.argsort(-ar.q_popmass)]
    Vtab = np.stack([ar.user_table(r["u"], r["known"])["val"] for r in train_sub])
    ent = np.zeros(ar.nQ)
    for q in range(ar.nQ):
        vq = Vtab[:, q]; vq = vq[vq >= 0].astype(int)
        if vq.size >= 2:
            c = np.bincount(vq, minlength=4).astype(float); pr = c[c > 0] / c.sum()
            ent[q] = float(-(pr * np.log2(pr)).sum())
    ent_order = [int(q) for q in np.argsort(-ent)]

    out = {}
    for name, ckpt in [("v31", V31_CKPT), ("v4", V4_CKPT)]:
        m, st = load_fold(ckpt)
        ar.model = m; ar.fold_state = st
        ar._emb_cache = ar._emb_cache  # unchanged
        # info-gain order is FOLD-DERIVED -> recompute per fold (honest deployment analog)
        ig = prescreen_gains(ar, train_sub, K=K, tag=f"v4base_{name}", verbose=False)
        ig_order = [int(q) for q in np.argsort(-ig)]
        print(f"\n=== FOLD {name} (val {st.get('best_val')}) : running baselines to turn {a.tmax} ===",
              flush=True)
        res = run_all(ar, dt, pop_order, ent_order, ig_order, a.random_seeds, a.tmax)
        cold0 = res["cold"][0]
        T = a.tmax
        lift = {k: res[k if k != "random" else "random_mean"][T] - cold0
                for k in ("random", "popularity", "infogain", "entropy")}
        out[name] = dict(cold0=cold0, full_profile=res["full_profile"], curves=res,
                         endpoint=T, lift=lift,
                         endpoint_ndcg={k: (res[k] if k != "random" else res["random_mean"])[T]
                                        for k in ("random", "popularity", "infogain", "entropy")})
        print(f"  [{name}] cold {cold0:.4f}  full-profile {res['full_profile']:.4f}", flush=True)
        for k in ("random", "popularity", "infogain", "entropy"):
            print(f"    {k:11s} endpoint@{T} {out[name]['endpoint_ndcg'][k]:.4f}  lift {lift[k]:+.4f}",
                  flush=True)
        print(f"  [{name}] [{(time.time()-t0)/60:.1f}m]", flush=True)

    # ---- merge into results json ----
    R = {}
    if os.path.exists(RESULTS):
        try:
            R = json.load(open(RESULTS))
        except Exception:
            R = {}
    R["baseline_rerun"] = dict(n_devtest=len(dt), tmax=a.tmax, random_seeds=a.random_seeds,
                               v31=out["v31"], v4=out["v4"],
                               note="v31 = the reference fold (arena default); v4 = de-OOD. random+"
                                    "entropy SEEN by v4; popularity+infogain HELD OUT.")
    json.dump(R, open(RESULTS, "w"), indent=1, default=float)

    # ---- append to ARENA_BUILD.md ----
    T = a.tmax
    L = []
    L.append("\n\n## FOLD V4 (de-OOD) + BASELINE RERUN\n\n")
    L.append(f"De-OOD retrain of the belief fold (`{V4_CKPT}`), same FoldV31 architecture, trained on "
             f"an ENSEMBLE of REAL selection strategies run through the real gated v2.1 answerer "
             f"(answers never faked; SELECTION randomized, biased to the hard tails). SEEN strategies: "
             f"random, popularity, entropy, on-profile, off-niche, adversarial(famous-refusal), mixed. "
             f"HELD OUT (never trained): blind-EIG(info-gain). 30% clean full profiles retained. Fold trained on "
             f"seed-123 users [0:5000] (arena TRAIN range), DISJOINT from the devtest split. NO LLM; "
             f"$0; 173/300 untouched.\n\n")
    R4 = R.get("fold_v4", {}); GT = R.get("gates", {})
    if R4:
        L.append(f"**Fold-v4 val NDCG@10 = {R4.get('best_val'):.4f}** @ep{R4.get('best_epoch')} "
                 f"(n_train {R4.get('n_train')}, n_val {R4.get('n_val')}).\n\n")
    if GT:
        gc = GT.get("G_clean", {}); gg = GT.get("G_generalize", {})
        L.append(f"- **G-clean**: fold {gc.get('fold_ndcg'):.4f} vs native {gc.get('native_ndcg'):.4f} "
                 f"(gap {gc.get('gap'):+.4f}) -> {'PASS' if gc.get('pass') else 'FAIL'}.\n")
        L.append(f"- **G-generalize**: held-out popularity NDCG@8 {gg.get('heldout_popularity_ndcg8'):.4f} "
                 f"vs mean(seen) {gg.get('mean_seen_ndcg8'):.4f} (tol 0.02) -> "
                 f"{'PASS' if gg.get('pass') else 'FAIL'}.\n\n")
    L.append(f"### Baseline endpoint@{T} on the SAME seed-123 devtest subsample (n={len(dt)}), "
             f"v3.1-fold vs v4-fold\n\n")
    L.append(f"| baseline | v3.1 endpoint@{T} | v3.1 lift | v4 endpoint@{T} | v4 lift | seen by v4? |\n")
    L.append("|---|--:|--:|--:|--:|:--:|\n")
    seenmap = {"random": "SEEN", "entropy": "SEEN", "popularity": "SEEN", "infogain": "HELD OUT"}
    for k in ("random", "popularity", "entropy", "infogain"):
        v31 = out["v31"]; v4 = out["v4"]
        L.append(f"| {k} | {v31['endpoint_ndcg'][k]:.4f} | {v31['lift'][k]:+.4f} | "
                 f"{v4['endpoint_ndcg'][k]:.4f} | {v4['lift'][k]:+.4f} | {seenmap[k]} |\n")
    L.append(f"\nCOLD v3.1 {out['v31']['cold0']:.4f} / v4 {out['v4']['cold0']:.4f}; "
             f"full-profile ceiling v3.1 {out['v31']['full_profile']:.4f} / v4 "
             f"{out['v4']['full_profile']:.4f}.\n")
    ig31 = out["v31"]["lift"]["infogain"]; ig4 = out["v4"]["lift"]["infogain"]
    best31 = max(out["v31"]["lift"], key=out["v31"]["lift"].get)
    best4 = max(out["v4"]["lift"], key=out["v4"]["lift"].get)
    L.append(f"\n**Verdict:** info-gain (blind-EIG) endpoint lift moved {ig31:+.4f} (v3.1) -> "
             f"{ig4:+.4f} (v4); expected-gain-goes-negative {'FIXED' if ig4 >= 0 else 'PERSISTS'}. "
             f"Strongest baseline v3.1={best31} ({out['v31']['lift'][best31]:+.4f}), "
             f"v4={best4} ({out['v4']['lift'][best4]:+.4f}).\n")
    open(MD, "a", encoding="utf-8").write("".join(L))

    print("\n" + "=" * 66, flush=True)
    print(f"BASELINE RERUN endpoint@{T} (v3.1 -> v4), devtest n={len(dt)}", flush=True)
    for k in ("random", "popularity", "entropy", "infogain"):
        print(f"  {k:11s} v3.1 lift {out['v31']['lift'][k]:+.4f} -> v4 lift {out['v4']['lift'][k]:+.4f}"
              f"  [{seenmap[k]}]", flush=True)
    print(f"\n[done] wrote {RESULTS} + appended {MD} [{(time.time()-t0)/60:.1f}m]", flush=True)


if __name__ == "__main__":
    main()
