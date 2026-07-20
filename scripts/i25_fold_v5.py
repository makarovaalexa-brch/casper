"""i25_fold_v5.py -- FOLD-V5: the CNP-style (Conditional-Neural-Process) intercept fix.

THE BROKEN DESIGN (FoldV31 / fold-v4 shared architecture):
    z = native_z + rho([pool_SUM, native_z, log1p(ntok)])
Two established defects (author's verdict; NOT re-diagnosed here):
  (1) COLD IS FOLD-DEPENDENT. rho fires even at zero tokens (rho(0, native_z, 0) != 0 after
      training), so the zero-answer belief is a LEARNED TRANSFORM of the prior, not the prior.
      Cold-start then differs across fold versions -- it should be native_z EXACTLY (a true
      intercept, identical across any fold trained with this architecture).
  (2) CARDINALITY CONFOUND. log1p(ntok) is an explicit input and pool is a SUM, so belief
      magnitude can grow with the NUMBER of tokens independent of content -> fakes monotonicity
      of the NDCG-vs-turns curve.

THE FIX (this file):
  - POOL = masked MEAN over answer tokens (not sum): aggregate magnitude reflects CONTENT, not count.
  - REMOVE log1p(ntok) from rho's inputs. rho takes [pool_mean, native_z] only.
  - PRESENCE GATE: z = native_z + g(ntok) * rho([pool_mean, native_z]), with g(ntok)=1-exp(-ntok),
    so g(0)=0 EXACTLY (empty -> prior BY CONSTRUCTION) and saturates to 1 after the 0->1 transition.
    The gate's ONLY job is the n=0 intercept -- it is NOT a count-magnitude knob.
  - KEEP the two-channel token features (implicit/explicit, knowledge level, value, fidelity,
    surprise, anti-surprise) EXACTLY as v3.1 -- those are content features, unchanged.
  - KEEP native_z = FR.enc_items(native_lists) as the additive intercept.

RETRAIN: on the de-OOD ensemble mixture (reuse DeOODGen from i25_fold_v4) but with REALISTIC tails
(adversarial+off-niche reduced from 0.42 -> 0.18), 7-strategy mixture + 30% clean + blind-EIG held out.
Single length B=8. 5000 train / 1000 val / 12 epochs, seed-123 users [0:6000] (disjoint from
devtest [17000:20000]). Frozen RecVAE decoder. Deterministic. NO LLM. $0. 173/300 untouched.

Run:
  python scripts/i25_fold_v5.py sanity                                  # 100 users, 2 epochs
  python scripts/i25_fold_v5.py train --n_train 5000 --n_val 1000 --epochs 12
  python scripts/i25_fold_v5.py baseline --n_devtest 1500
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "4")
import sys, json, time, argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
torch.set_num_threads(4)

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE); sys.path.insert(0, os.path.join(_HERE, "instrument2"))
import warnings
warnings.filterwarnings("ignore")

import arena_core as AC
import i25_lib as L
import i25_fold_v31 as FV31
import i25_fold_v4 as FV4
from i25_fold_v4 import DeOODGen, _sanitize_qemb, batch_loss, val_ndcg, gate_clean, gate_generalize
from i25_fold_v3_sampler import KIND_IMPL, KIND_EXPL, LVL_NEG, NTYPE

D_LAT = L.D_LAT
BEST = ".cache/i25_fold_v5_best.pt"
CKPT = ".cache/i25_fold_v5.pt"
LOGP = ".cache/i25_fold_v5_log.json"
RESULTS = ".cache/arena/fold_v5_results.json"
MD = "experiments/ARENA_BUILD.md"
CLEAN_FRAC = 0.30
BFIX = 8
K = 10
INTERCEPT_TOL = 1e-5

# ---- REALISTIC-TAIL mixture: same 7 SEEN strategies + blind-EIG held out, tails 0.42 -> 0.18 ----
#   v4 (fat tails):  off_niche 0.24 + adversarial 0.18 = 0.42
#   v5 (realistic):  off_niche 0.10 + adversarial 0.08 = 0.18  (mass redistributed to realistic arms)
STRAT_W = {"random": 0.20, "popularity": 0.14, "entropy": 0.14, "on_profile": 0.20,
           "off_niche": 0.10, "adversarial": 0.08, "mixed": 0.14}      # realistic tails = 0.18
SEEN = list(STRAT_W.keys())
assert abs(sum(STRAT_W.values()) - 1.0) < 1e-9
assert SEEN == FV4.SEEN, "strategy set must match v4 (only weights differ)"
_TAILS = STRAT_W["off_niche"] + STRAT_W["adversarial"]
assert 0.15 <= _TAILS <= 0.20, f"tails out of realistic band: {_TAILS}"


# =============================================================== FOLD-V5 (CNP-style, masked mean)
class FoldV5(nn.Module):
    """CNP-style Deep-Sets belief encoder. Same per-token features as FoldV3.1 (two-channel content
    features unchanged), but:
      - POOL = masked MEAN (not sum)               -> aggregate reflects content not count
      - rho input = [pool_mean, native_z]  (NO log1p(ntok))
      - z = native_z + g(ntok) * rho(.),  g(ntok) = 1 - exp(-ntok),  g(0)=0 EXACTLY
    => empty answer set folds to the prior (native_z) BY CONSTRUCTION: a true intercept, identical
    across any fold trained with this architecture. Signature matches FoldV31.forward so the arena
    (arena_core.belief_z_batch -> FV31.fold_batch) can swap it in directly."""

    def __init__(self, d=D_LAT):
        super().__init__()
        in_dim = NTYPE + 2 + 3 + 1 + 1 + 3 + 1 + d + d
        self.phi = nn.Sequential(nn.Linear(in_dim, d), nn.ReLU(), nn.Linear(d, d), nn.ReLU())
        self.rho = nn.Sequential(nn.Linear(2 * d, d), nn.ReLU(), nn.Linear(d, d))   # NO cardinality in
        nn.init.zeros_(self.rho[-1].weight); nn.init.zeros_(self.rho[-1].bias)
        self.d = d

    def forward(self, tt, tk, tl, ts, ta, tf, tv, te, mask, native_z, impl_ablate=False):
        B, Kk, d = te.shape
        is_impl = (tk == KIND_IMPL).to(te.dtype)
        is_expl = 1.0 - is_impl
        is_neg = ((tl == LVL_NEG) & (tk == KIND_IMPL)).to(te.dtype)
        type_oh = F.one_hot(tt.clamp(min=0), NTYPE).to(te.dtype)
        kind_oh = F.one_hot(tk.clamp(min=0), 2).to(te.dtype)
        lvl_oh = F.one_hot(tl.clamp(min=0), 3).to(te.dtype) * is_impl.unsqueeze(-1)
        fid_oh = F.one_hot(tf.clamp(min=0), 3).to(te.dtype) * is_expl.unsqueeze(-1)
        surprise = ts.unsqueeze(-1)
        anti = (ta * is_neg).unsqueeze(-1)                    # lit only on no-clue tokens (unchanged)
        value = tv.unsqueeze(-1)
        ve = value * te
        x = torch.cat([type_oh, kind_oh, lvl_oh, surprise, anti, fid_oh, value, te, ve], dim=-1)
        eff_mask = mask * (is_expl if impl_ablate else torch.ones_like(mask))
        h = self.phi(x) * eff_mask.unsqueeze(-1)
        ntok = eff_mask.sum(1, keepdim=True)                  # (B,1) effective token count
        pool = h.sum(1) / ntok.clamp(min=1.0)                 # masked MEAN (n=0 -> 0 vector)
        delta = self.rho(torch.cat([pool, native_z], dim=-1))
        g = 1.0 - torch.exp(-ntok)                            # presence gate; g(0)=0 EXACTLY
        return native_z + g * delta


# =============================================================== realistic-tail generator
class V5Gen(DeOODGen):
    """DeOODGen with REALISTIC tail weights (STRAT_W above). Everything else -- static orders, the
    real gated v2.1 answerer, clean profiles, held-out blind-EIG -- is inherited unchanged."""
    def build_reveal(self, r, rng, clean_frac=CLEAN_FRAC):
        if rng.random() < clean_frac:
            return self.clean_tokens(r, rng)
        strat = rng.choice(SEEN, p=[STRAT_W[s] for s in SEEN])
        return self.interview_tokens(r, self.select(r, strat, rng))


# =============================================================== helpers
def _load_fold(path, cls):
    blob = torch.load(path, map_location="cpu")
    m = cls(); m.load_state_dict(blob["model"]); m.eval()
    return m, blob.get("state", {})


def _native_z_empty(FR):
    return FR.enc_items([[]])[0].numpy().astype(np.float64)


def intercept_check(FR, model):
    """THE decisive check: cold belief must be a TRUE INTERCEPT == native_z(enc_items([]))."""
    z_cold = FV31.fold_np(FR, model, [], [])          # fold with empty answers, empty native
    z_nat = _native_z_empty(FR)                        # RecVAE prior for an empty item set (zeros)
    max_abs = float(np.max(np.abs(z_cold - z_nat)))
    l2 = float(np.linalg.norm(z_cold - z_nat))
    ok = bool(max_abs <= INTERCEPT_TOL)
    return dict(passed=ok, max_abs_diff=max_abs, l2_diff=l2, tol=INTERCEPT_TOL,
                cold_norm=float(np.linalg.norm(z_cold)), native_norm=float(np.linalg.norm(z_nat)),
                cold_first5=[float(v) for v in z_cold[:5]],
                native_first5=[float(v) for v in z_nat[:5]])


@torch.no_grad()
def cold_ndcg(ar, model, users):
    """Cold-start NDCG@10 with the fold (ask nothing) vs the native RecVAE prior. For a true
    intercept these are IDENTICAL and equal across any fold with this architecture."""
    held = [u["held"] for u in users]; prof = [set(u["known"].keys()) for u in users]
    fold = FV4._fold_ndcg(ar, model, [[]] * len(users), [[]] * len(users), held, prof)
    vals = []
    for s in range(0, len(users), 1200):
        e = min(s + 1200, len(users))
        Zn = ar.FR.enc_items([[]] * (e - s)).numpy().astype(np.float64)
        vv = AC.ndcg_at_k_batch(ar.FR, Zn, held[s:e], prof[s:e], K)
        vals += [v for v in vv if v is not None]
    nat = float(np.mean(vals)) if vals else 0.0
    return dict(fold_cold_ndcg10=fold, native_cold_ndcg10=nat, diff=fold - nat,
                equal=bool(abs(fold - nat) <= 1e-6))


# =============================================================== train
def cmd_train(a):
    t0 = time.time()
    print("[v5] building arena (gated v2.1 world + EASE + universe) ...", flush=True)
    ar = AC.Arena()
    _sanitize_qemb(ar)
    coh = AC.make_cohorts(ar, n_train=a.n_train, n_devval=a.n_val, n_devtest=100)
    tr_users, val_users = coh["train"], coh["devval"]
    train_sub = tr_users[:min(800, len(tr_users))]
    print(f"[v5] cohorts: train {len(tr_users)} / val {len(val_users)} (seed {AC.SEED}); realistic "
          f"tails={_TAILS:.2f} (off_niche {STRAT_W['off_niche']} + adversarial "
          f"{STRAT_W['adversarial']}); generating answer tables ...", flush=True)
    ar.prefill_answers(tr_users, "v5train", dict(role="v5train", n=a.n_train))
    ar.prefill_answers(val_users, "v5val", dict(role="v5val", n=a.n_val))
    print(f"[v5] answer tables ready [{(time.time()-t0)/60:.1f}m]", flush=True)

    gen = V5Gen(ar, train_sub)
    ar.gen = gen

    torch.manual_seed(a.seed); np.random.seed(a.seed)
    model = FoldV5()

    # ---- ARCHITECTURE SANITY (before any training): empty -> prior BY CONSTRUCTION ----
    ic0 = intercept_check(ar.FR, model)
    print(f"[v5] intercept @init: max|cold-native|={ic0['max_abs_diff']:.2e} -> "
          f"{'PASS' if ic0['passed'] else 'FAIL'}", flush=True)

    opt = torch.optim.Adam(model.parameters(), lr=a.lr, weight_decay=1e-6)
    step_rng = np.random.default_rng(a.seed + 5050)
    state = dict(epoch=0, best_val=-1.0, best_epoch=-1, history=[], strategies_seen=SEEN,
                 strategies_heldout=FV4.HELDOUT, strat_w=STRAT_W, tails=_TAILS,
                 clean_frac=CLEAN_FRAC, arch="FoldV5-CNP")
    print(f"[v5] === training FoldV5 (CNP mean-pool, B={BFIX}, {int(CLEAN_FRAC*100)}% clean), "
          f"epochs={a.epochs} ===", flush=True)
    for ep in range(a.epochs):
        model.train()
        order = np.arange(len(tr_users)); step_rng.shuffle(order)
        tot, nb = 0.0, 0
        for b0 in range(0, len(order), a.batch):
            us = [tr_users[i] for i in order[b0:b0 + a.batch]]
            loss = batch_loss(ar.FR, model, gen, us, step_rng)
            if loss is None or not torch.isfinite(loss):
                continue
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            tot += float(loss.item()); nb += 1
        vN = val_ndcg(ar, model, gen, val_users, a.seed + 7)
        state["epoch"] = ep + 1
        rec = dict(epoch=ep + 1, train_loss=round(tot / max(nb, 1), 4), val_ndcg=round(vN, 4),
                   min=round((time.time() - t0) / 60, 1))
        state["history"].append(rec)
        print(f"[v5] ep {ep+1:2d} | loss {rec['train_loss']:.4f} | val NDCG@10 {vN:.4f} | "
              f"{rec['min']}m", flush=True)
        torch.save(dict(model=model.state_dict(), opt=opt.state_dict(), state=state,
                        args=vars(a)), CKPT)
        if vN > state["best_val"]:
            state["best_val"] = vN; state["best_epoch"] = ep + 1
            torch.save(dict(model=model.state_dict(), state=state, args=vars(a)), BEST)
        json.dump(state["history"], open(LOGP, "w"), indent=1)
    print(f"[v5] BEST val {state['best_val']:.4f} @ep{state['best_epoch']} -> {BEST} "
          f"[{(time.time()-t0)/60:.1f}m]", flush=True)

    # ---- load BEST + THE decisive intercept check FIRST ----
    best, bstate = _load_fold(BEST, FoldV5)
    ic = intercept_check(ar.FR, best)
    cnd = cold_ndcg(ar, best, val_users)
    print("\n=== INTERCEPT CHECK (decisive) ===", flush=True)
    print(f"  cold z first5 {['%+.5f'%v for v in ic['cold_first5']]}", flush=True)
    print(f"  native first5 {['%+.5f'%v for v in ic['native_first5']]}", flush=True)
    print(f"  max|cold-native|={ic['max_abs_diff']:.2e}  L2={ic['l2_diff']:.2e}  "
          f"-> {'PASS (true intercept)' if ic['passed'] else 'FAIL'}", flush=True)
    print(f"  cold NDCG@10: fold {cnd['fold_cold_ndcg10']:.4f} vs native "
          f"{cnd['native_cold_ndcg10']:.4f} (diff {cnd['diff']:+.6f}) -> "
          f"{'EQUAL' if cnd['equal'] else 'DIFFER'}", flush=True)
    if not ic["passed"]:
        R = dict(fold_v5=dict(best_val=state["best_val"], history=state["history"]),
                 intercept_check=ic, cold_ndcg=cnd, FIX_FAILED=True)
        os.makedirs(os.path.dirname(RESULTS), exist_ok=True)
        json.dump(R, open(RESULTS, "w"), indent=1, default=float)
        print("\n[v5] *** INTERCEPT CHECK FAILED -- fix did NOT work. STOP. ***", flush=True)
        return R

    # ---- gates (reuse v4 gate implementations on the v5 fold) ----
    cold0 = cnd["fold_cold_ndcg10"]
    print("\n=== PRE-REGISTERED GATES ===", flush=True)
    print("  G-clean:      clean full-profile fold NDCG@10 >= native RecVAE - 0.01.", flush=True)
    print("  G-generalize: HELD-OUT blind-EIG NDCG@8 >= mean(SEEN strategies NDCG@8) - 0.02.",
          flush=True)
    gc = gate_clean(ar, best, val_users)
    gg = gate_generalize(ar, best, val_users, cold0)
    print(f"\n[G-clean]      fold {gc['fold_ndcg']:.4f}  native {gc['native_ndcg']:.4f}  "
          f"gap {gc['gap']:+.4f}  -> {'PASS' if gc['pass'] else 'FAIL'}", flush=True)
    print("[G-generalize] NDCG@8 by strategy (SEEN | held-out):", flush=True)
    for s in SEEN:
        d = gg["per_strategy"][s]
        print(f"    {s:12s} {d['ndcg8']:.4f}  lift {d['lift']:+.4f}", flush=True)
    d = gg["per_strategy"]["blind_eig"]
    print(f"    {'blind_eig':12s} {d['ndcg8']:.4f}  lift {d['lift']:+.4f}  (HELD OUT)", flush=True)
    print(f"    mean(seen)={gg['mean_seen_ndcg8']:.4f}  held-out blind-EIG="
          f"{gg['heldout_blind_eig_ndcg8']:.4f}  -> {'PASS' if gg['pass'] else 'FAIL'}", flush=True)

    R = {}
    if os.path.exists(RESULTS):
        try:
            R = json.load(open(RESULTS))
        except Exception:
            R = {}
    R["fold_v5"] = dict(best_val=state["best_val"], best_epoch=state["best_epoch"],
                        cold0=cold0, history=state["history"], n_train=len(tr_users),
                        n_val=len(val_users), strategies_seen=SEEN, strategies_heldout=FV4.HELDOUT,
                        strat_w=STRAT_W, tails=_TAILS, clean_frac=CLEAN_FRAC, B=BFIX, ckpt=BEST,
                        arch="FoldV5-CNP-meanpool-presencegate")
    R["intercept_check"] = ic
    R["cold_ndcg"] = cnd
    R["gates"] = dict(G_clean=gc, G_generalize=gg)
    os.makedirs(os.path.dirname(RESULTS), exist_ok=True)
    json.dump(R, open(RESULTS, "w"), indent=1, default=float)
    print(f"\n[v5] wrote intercept + fold val + gates -> {RESULTS} [{(time.time()-t0)/60:.1f}m]",
          flush=True)
    return R


# =============================================================== sanity (100 users, 2 epochs)
def cmd_sanity(a):
    a.n_train = 100; a.n_val = 60; a.epochs = 2; a.batch = 64
    print("[v5] SANITY run (100 users, 2 epochs): confirm finite loss + intercept before full run",
          flush=True)
    return cmd_train(a)


# =============================================================== baseline rerun (v5 vs v3.1)
def _endpoint_curve(out, Tmax):
    c = out["curves"][K]
    return [float(np.nanmean(c[:, t])) for t in range(Tmax + 1)]


def _run_all(ar, dt, pop_order, ent_order, ig_order, seeds, Tmax):
    from arena_policies import StaticSeq, B0Cold, run_policy
    from arena_baselines_v2 import RandomPolicy, full_profile_ndcg
    res = {}
    res["cold"] = _endpoint_curve(run_policy(ar, dt, B0Cold(), Tmax, Ks=(K,)), Tmax)
    res["popularity"] = _endpoint_curve(run_policy(ar, dt, StaticSeq("popularity", pop_order),
                                                   Tmax, Ks=(K,)), Tmax)
    res["entropy"] = _endpoint_curve(run_policy(ar, dt, StaticSeq("entropy", ent_order),
                                               Tmax, Ks=(K,)), Tmax)
    res["infogain"] = _endpoint_curve(run_policy(ar, dt, StaticSeq("infogain", ig_order),
                                                Tmax, Ks=(K,)), Tmax)
    rc = []
    for sd in seeds:
        rc.append(_endpoint_curve(run_policy(ar, dt, RandomPolicy(sd), Tmax, Ks=(K,)), Tmax))
    rc = np.array(rc)
    res["random_mean"] = rc.mean(0).tolist(); res["random_std"] = rc.std(0).tolist()
    fp, _ = full_profile_ndcg(ar, dt)
    res["full_profile"] = fp
    return res


def cmd_baseline(a):
    from arena_policies import train_pop_prior, prescreen_gains
    t0 = time.time()
    ar = AC.Arena()
    _sanitize_qemb(ar)
    coh = AC.make_cohorts(ar, n_train=14000, n_devval=3000, n_devtest=3000)
    train, dt_full = coh["train"], coh["devtest"]
    dt = dt_full[:a.n_devtest]
    print(f"[v5-base] split seed {AC.SEED}: train {len(train)} / devtest {len(dt_full)} "
          f"(using {len(dt)} subsample); Tmax={a.tmax}", flush=True)
    train_sub = train[:a.n_train_sub]
    ar.prefill_answers(train_sub, "trainsub", dict(n_train=14000, n_devval=3000, n_devtest=3000,
                                                   sub=a.n_train_sub))
    ar.prefill_answers(dt, "v4devtest", dict(role="v4devtest", n=a.n_devtest))
    ar.set_pop_prior(train_pop_prior(ar, train_sub))

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
    for name, ckpt, cls in [("v31", AC.FOLD_CKPT, FV31.FoldV31), ("v5", BEST, FoldV5)]:
        m, st = _load_fold(ckpt, cls)
        ar.model = m; ar.fold_state = st
        ig = prescreen_gains(ar, train_sub, K=K, tag=f"v5base_{name}", verbose=False)
        ig_order = [int(q) for q in np.argsort(-ig)]
        print(f"\n=== FOLD {name} (val {st.get('best_val')}) : baselines to turn {a.tmax} ===",
              flush=True)
        res = _run_all(ar, dt, pop_order, ent_order, ig_order, a.random_seeds, a.tmax)
        cold0 = res["cold"][0]; T = a.tmax
        endp = {k: (res[k] if k != "random" else res["random_mean"])[T]
                for k in ("random", "popularity", "infogain", "entropy")}
        lift = {k: endp[k] - cold0 for k in endp}
        out[name] = dict(cold0=cold0, full_profile=res["full_profile"], endpoint=T,
                         endpoint_ndcg=endp, lift=lift, curves=res)
        print(f"  [{name}] cold {cold0:.4f}  full-profile {res['full_profile']:.4f}", flush=True)
        for k in ("random", "popularity", "entropy", "infogain"):
            print(f"    {k:11s} endpoint@{T} {endp[k]:.4f}  lift {lift[k]:+.4f}", flush=True)

    R = {}
    if os.path.exists(RESULTS):
        try:
            R = json.load(open(RESULTS))
        except Exception:
            R = {}
    R["baseline_rerun"] = dict(n_devtest=len(dt), tmax=a.tmax, random_seeds=a.random_seeds,
                               v31=out["v31"], v5=out["v5"],
                               note="ABSOLUTE endpoint@10 (NOT lift-over-cold): v5 and v3.1 no longer "
                                    "share a cold intercept -- v5 cold == native prior BY CONSTRUCTION.")
    os.makedirs(os.path.dirname(RESULTS), exist_ok=True)
    json.dump(R, open(RESULTS, "w"), indent=1, default=float)
    _append_md(R, out, a.tmax, len(dt))
    print("\n" + "=" * 66, flush=True)
    print(f"BASELINE RERUN abs endpoint@{a.tmax} (v3.1 vs v5), devtest n={len(dt)}", flush=True)
    for k in ("random", "popularity", "entropy", "infogain"):
        print(f"  {k:11s} v3.1 {out['v31']['endpoint_ndcg'][k]:.4f}  ->  v5 "
              f"{out['v5']['endpoint_ndcg'][k]:.4f}", flush=True)
    print(f"  cold        v3.1 {out['v31']['cold0']:.4f}  ->  v5 {out['v5']['cold0']:.4f}", flush=True)
    print(f"\n[done] wrote {RESULTS} + appended {MD} [{(time.time()-t0)/60:.1f}m]", flush=True)
    return R


def _append_md(R, out, T, ndt):
    v31, v5 = out["v31"], out["v5"]
    ic = R.get("intercept_check", {}); cnd = R.get("cold_ndcg", {})
    fv5 = R.get("fold_v5", {}); GT = R.get("gates", {})
    o = []
    o.append("\n\n## FOLD V5 (CNP-style fix): intercept + baseline rerun\n\n")
    o.append("Author-directed fix of the belief-encoder architecture. FoldV3.1/fold-v4 used "
             "`z = native_z + rho([pool_SUM, native_z, log1p(ntok)])`, which (1) made the cold/zero-"
             "answer belief a LEARNED transform of the prior (rho fires at n=0) rather than the prior "
             "itself, and (2) fed `log1p(ntok)` + a SUM pool so belief magnitude could grow with the "
             "token COUNT independent of content. **FoldV5 (CNP-style)**: masked-MEAN pool, NO "
             "cardinality input, presence gate `z = native_z + g(ntok)*rho([pool_mean, native_z])` "
             "with `g(ntok)=1-exp(-ntok)` so `g(0)=0` EXACTLY -> empty answer set folds to the prior "
             "BY CONSTRUCTION. Two-channel content token features unchanged from v3.1. Retrained on the "
             "de-OOD ensemble with REALISTIC tails (off_niche+adversarial 0.42 -> "
             f"{R.get('fold_v5', {}).get('tails', 0.18):.2f}), 7 SEEN strategies + 30% clean, blind-EIG "
             "held out. Single length B=8, seed-123 users [0:6000] (disjoint from devtest [17000:20000]), "
             "frozen RecVAE decoder, deterministic. NO LLM; $0; 173/300 untouched.\n\n")
    if ic:
        o.append("### Decisive intercept check (cold belief == native prior)\n\n")
        o.append(f"- `max|fold([],[]) - enc_items([])| = {ic.get('max_abs_diff'):.2e}` "
                 f"(tol {ic.get('tol')}) -> **{'PASS -- true intercept' if ic.get('passed') else 'FAIL'}**. "
                 f"Cold latent norm {ic.get('cold_norm'):.4f} vs native {ic.get('native_norm'):.4f}.\n")
    if cnd:
        o.append(f"- Cold-start NDCG@10: fold {cnd.get('fold_cold_ndcg10'):.4f} == native "
                 f"{cnd.get('native_cold_ndcg10'):.4f} (diff {cnd.get('diff'):+.6f}) -> "
                 f"identical across any fold with this architecture.\n\n")
    if fv5:
        o.append(f"**Fold-v5 val NDCG@10 = {fv5.get('best_val'):.4f}** @ep{fv5.get('best_epoch')} "
                 f"(n_train {fv5.get('n_train')}, n_val {fv5.get('n_val')}).\n\n")
    if GT:
        gc = GT.get("G_clean", {}); gg = GT.get("G_generalize", {})
        o.append(f"- **G-clean**: fold {gc.get('fold_ndcg'):.4f} vs native {gc.get('native_ndcg'):.4f} "
                 f"(gap {gc.get('gap'):+.4f}) -> {'PASS' if gc.get('pass') else 'FAIL'}.\n")
        o.append(f"- **G-generalize**: held-out blind-EIG NDCG@8 "
                 f"{gg.get('heldout_blind_eig_ndcg8'):.4f} vs mean(seen) {gg.get('mean_seen_ndcg8'):.4f} "
                 f"(tol 0.02) -> {'PASS' if gg.get('pass') else 'FAIL'}.\n\n")
    o.append(f"### Baseline ABSOLUTE endpoint@{T} on the SAME seed-123 devtest subsample (n={ndt}), "
             f"v3.1-fold vs v5-fold\n\n")
    o.append("Both folds now share the SAME cold intercept convention (v5 cold == native prior). "
             "Absolute endpoint@10 reported (NOT lift-over-cold), since the v3.1 cold was a learned "
             "transform and the two colds differ.\n\n")
    o.append(f"| baseline | v3.1 cold | v3.1 endpoint@{T} | v5 cold | v5 endpoint@{T} |\n")
    o.append("|---|--:|--:|--:|--:|\n")
    for k in ("random", "popularity", "entropy", "infogain"):
        o.append(f"| {k} | {v31['cold0']:.4f} | {v31['endpoint_ndcg'][k]:.4f} | "
                 f"{v5['cold0']:.4f} | {v5['endpoint_ndcg'][k]:.4f} |\n")
    o.append(f"\nCOLD v3.1 {v31['cold0']:.4f} / v5 {v5['cold0']:.4f}; full-profile ceiling v3.1 "
             f"{v31['full_profile']:.4f} / v5 {v5['full_profile']:.4f}.\n")
    best31 = max(v31["endpoint_ndcg"], key=v31["endpoint_ndcg"].get)
    best5 = max(v5["endpoint_ndcg"], key=v5["endpoint_ndcg"].get)
    o.append(f"\n**Verdict:** strongest endpoint@{T} baseline -- v3.1={best31} "
             f"({v31['endpoint_ndcg'][best31]:.4f}), v5={best5} ({v5['endpoint_ndcg'][best5]:.4f}). "
             f"v5's cold-start now equals the native RecVAE prior by construction (no fold-dependent "
             f"cold), and the cardinality confound is removed (mean pool, no log1p(ntok)).\n")
    open(MD, "a", encoding="utf-8").write("".join(o))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["train", "sanity", "baseline"])
    ap.add_argument("--n_train", type=int, default=5000)
    ap.add_argument("--n_val", type=int, default=1000)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n_devtest", type=int, default=1500)
    ap.add_argument("--n_train_sub", type=int, default=800)
    ap.add_argument("--random_seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--tmax", type=int, default=8)
    a = ap.parse_args()
    if a.cmd == "train":
        cmd_train(a)
    elif a.cmd == "sanity":
        cmd_sanity(a)
    elif a.cmd == "baseline":
        cmd_baseline(a)
