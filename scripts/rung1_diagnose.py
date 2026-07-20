"""rung1_diagnose.py -- READ-ONLY diagnosis of the Rung-I value-INERTNESS failure.

Does NOT modify rung1_encoder.py. Imports it and runs cheap controlled micro-experiments (2k train /
1k val, 3 epochs each, SAME split seed as the pilot) to isolate WHY the graded value channel is inert
(value-zeroing dNDCG 0.0002 << 0.01 MDE; gamma spread 0.004; gamma(hated) never crosses 0).

HARD RULE #1 restated: NO data reduction beyond the author-sanctioned 2k pilot cohort. Same load_split
seed, same prep_users, all users kept, no caps/subsamples/top-N. Micro-exps change exactly ONE knob each.

Each config reports: value_delta (present & absent regime), gamma-spread (loved-hated), gamma(hated)
value + sign, IG2 flip percentiles, and per-loss-term GRADIENT NORM reaching the gamma value-slope
(the decisive instrumentation for suspect #1/#5).
"""
import os, sys, json, time, math, collections
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "instrument2"))
import warnings; warnings.filterwarnings("ignore")
import i25_lib as L
import rung1_encoder as R1

OUT = ".cache/rung1"
os.makedirs(OUT, exist_ok=True)


# ----------------------------------------------------------- AUX: in-interview region-ordinal loss
def aux_region_ordinal(z, args):
    """Suspect #5 test: give the value field a DIRECT gradient. For every EXPLICIT (valued) token, the
    user latent z should align with that token's own member-bag region emb in proportion to the revealed
    value: loved (v=+1) -> z.e > 0, hated (v=-1) -> z.e < 0. softplus(-v * z.e_hat), bounded gradient.
    This is a proxy for 'rank the region's OWN members by value' (z.e_hat ~ region mean score direction)."""
    tt, tk, tl, tf, tv, te, mask, nz = args
    is_expl = (tk == R1.KIND_EXPL).float() * mask                  # (B,K)
    denom = is_expl.sum().clamp(min=1.0)
    e = F.normalize(te, dim=-1)                                    # unit region dirs
    score = (z.unsqueeze(1) * e).sum(-1)                           # (B,K) z . e_hat
    l = F.softplus(-(tv * score))                                  # push v*score > 0
    return (l * is_expl).sum() / denom


# ----------------------------------------------------------- one training batch with knobs
def batch_terms(FR, model, S, recs, cfg, rng, gen):
    tok_lists, nat_lists, prof, likes, held_ids, held_stars = R1.build_batch(S, recs, cfg["clean_frac"], rng)
    if not tok_lists:
        return None
    args, _ = R1.pack(FR, tok_lists, nat_lists)
    B = len(tok_lists)
    kA, kB, kC = R1._channel_keeps(B, cfg["p_drop"], rng)
    if cfg["force_drop_native"]:
        kA = torch.zeros(B, dtype=torch.float32)
    out = model(*args, keepA=kA, keepB=kB, keepC=kC, sample=True, gen=gen)
    z = out["z"]
    Smat = z @ FR.W.T + FR.bdec
    Lc = R1.consume_loss(Smat, prof, likes)
    Lo = R1.ordinal_loss(Smat, held_ids, held_stars, cfg["margin"])
    Lkl = R1.kl_loss(out["mu"], out["logsigma"], out["native"])
    Laux = aux_region_ordinal(z, args) if cfg["lam_aux"] > 0 else torch.zeros((), requires_grad=True)
    return dict(Lo=Lo, Lc=Lc, Lkl=Lkl, Laux=Laux)


def total_loss(terms, cfg, beta):
    return (cfg["lam_ord"] * terms["Lo"] + cfg["lam_c"] * terms["Lc"]
            + beta * terms["Lkl"] + cfg["lam_aux"] * terms["Laux"])


# ----------------------------------------------------------- gradient probe (the decisive instrument)
def grad_probe(FR, model, S, recs, cfg, rng, gen, beta):
    """On ONE batch, backprop each loss TERM separately and record the grad-norm reaching the gamma
    VALUE-SLOPE (gamma_head.weight[:,0]) and tau head. Answers: is the ordinal gradient to gamma
    negligible vs consume/KL? does the aux term deliver a large gamma gradient?"""
    terms = batch_terms(FR, model, S, recs, cfg, rng, gen)
    if terms is None:
        return {}
    vs = model.gamma_head.weight                 # (d, VAL_IN); col 0 = value feature
    th = model.tau_head.weight
    rho = model.rho_mu[-1].weight
    probe = {}
    for name in ["Lo", "Lc", "Lkl", "Laux"]:
        model.zero_grad(set_to_none=True)
        t = terms[name]
        if not t.requires_grad or float(t) == 0.0 and t.grad_fn is None:
            probe[name] = dict(gamma_valslope=0.0, gamma_full=0.0, tau=0.0, rho_last=0.0, val=float(t))
            continue
        t.backward(retain_graph=True)
        gv = 0.0 if vs.grad is None else float(vs.grad[:, 0].norm())
        gf = 0.0 if vs.grad is None else float(vs.grad.norm())
        gt = 0.0 if th.grad is None else float(th.grad.norm())
        gr = 0.0 if rho.grad is None else float(rho.grad.norm())
        probe[name] = dict(gamma_valslope=gv, gamma_full=gf, tau=gt, rho_last=gr, val=float(t))
    model.zero_grad(set_to_none=True)
    return probe


# ----------------------------------------------------------- value-zeroing eval at a chosen keepA
def eval_vz(FR, model, S, recs, seed, keepA):
    model.eval()
    rng = np.random.default_rng(seed)
    tls, nls, idx = [], [], []
    for i, rec in enumerate(recs):
        toks, native = R1.sample_reveal(S, rec, 0.3, rng)
        if toks:
            tls.append(toks); nls.append(native); idx.append(i)
    meh = R1._value_to_meh(tls)
    dl = []
    for s in range(0, len(tls), 600):
        e = min(s + 600, len(tls))
        Zf = R1.fold_mu_batch(FR, model, tls[s:e], nls[s:e], keepA=keepA)
        Zm = R1.fold_mu_batch(FR, model, meh[s:e], nls[s:e], keepA=keepA)
        for r in range(s, e):
            rec = recs[idx[r]]
            a = L.ndcg10(FR, Zf[r - s], rec["held_likes"], set(rec["known"].keys()))
            b = L.ndcg10(FR, Zm[r - s], rec["held_likes"], set(rec["known"].keys()))
            if None not in (a, b):
                dl.append(a - b)
    dl = np.array(dl, float)
    if len(dl) < 2:
        return float("nan"), [float("nan"), float("nan")]
    ci = R1._boot_ci(dl, seed=1)
    return float(dl.mean()), list(ci)


def ig2_quick(FR, model, S, D, nmax=12):
    genres = [("gen", g) for g in range(D["Gmat"].shape[1])]
    love, hate = [], []
    for gk in genres:
        if len(S.attr_set[gk]) < 20:
            continue
        emb = S.region_emb(R1.TYPE_ATTR, gk); members = list(S.attr_set[gk])
        zl = R1.fold_mu(FR, model, [R1._mktok(R1.TYPE_ATTR, R1.KIND_EXPL, R1.LVL_ROUGH, R1.FID_EASE,
                                              R1.CENTERED_FOLD["loved"], emb)], [])
        zh = R1.fold_mu(FR, model, [R1._mktok(R1.TYPE_ATTR, R1.KIND_EXPL, R1.LVL_ROUGH, R1.FID_EASE,
                                              R1.CENTERED_FOLD["hated"], emb)], [])
        love.append(R1._region_percentile(R1._scores(FR, zl), members))
        hate.append(R1._region_percentile(R1._scores(FR, zh), members))
        if len(love) >= nmax:
            break
    return dict(pctile_love=float(np.mean(love)), pctile_hate=float(np.mean(hate)),
                drop=float(np.mean(love) - np.mean(hate)))


# ----------------------------------------------------------- train one config (few epochs)
def run_config(FR, S, D, tr_recs, val_recs, name, cfg, epochs, seed=0):
    t0 = time.time()
    torch.manual_seed(seed); np.random.seed(seed)
    model = R1.Rung1Encoder()
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=1e-6)
    gen = torch.Generator().manual_seed(seed + 5)
    step_rng = np.random.default_rng(seed + 100)
    nsteps_total = epochs * math.ceil(len(tr_recs) / cfg["batch"])
    warm = max(1, int(0.6 * nsteps_total)); step = 0
    hist = []
    for ep in range(epochs):
        model.train()
        order = np.arange(len(tr_recs)); step_rng.shuffle(order)
        tot = collections.defaultdict(float); nb = 0
        for b0 in range(0, len(order), cfg["batch"]):
            recs = [tr_recs[i] for i in order[b0:b0 + cfg["batch"]]]
            beta = cfg["beta_max"] * min(1.0, step / warm)
            terms = batch_terms(FR, model, S, recs, cfg, step_rng, gen)
            if terms is None:
                continue
            loss = total_loss(terms, cfg, beta)
            if not torch.isfinite(loss):
                continue
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            for k in ["Lo", "Lc", "Lkl", "Laux"]:
                tot[k] += float(terms[k])
            nb += 1; step += 1
        tel = R1.film_telemetry(model)
        hist.append(dict(epoch=ep + 1, gamma_hated=round(tel["gamma_hated"], 4),
                         gamma_loved=round(tel["gamma_loved"], 4),
                         spread=round(tel["gamma_loved"] - tel["gamma_hated"], 4),
                         **{k: round(v / max(nb, 1), 3) for k, v in tot.items()}))
    # ---- diagnostics ----
    tel = R1.film_telemetry(model)
    d_pres, ci_pres = eval_vz(FR, model, S, val_recs, seed + 21, keepA=1.0)
    d_abs, ci_abs = eval_vz(FR, model, S, val_recs, seed + 22, keepA=0.0)
    ig2 = ig2_quick(FR, model, S, D)
    probe = grad_probe(FR, model, S, tr_recs[:256], cfg, np.random.default_rng(seed + 99),
                       torch.Generator().manual_seed(seed + 7), cfg["beta_max"])
    # intercept-after-training check
    z_empty = R1.fold_mu(FR, model, [], [])
    intercept = float(np.abs(z_empty).max())
    res = dict(
        name=name, minutes=round((time.time() - t0) / 60, 2), epochs=epochs, cfg=cfg,
        gamma_hated=tel["gamma_hated"], gamma_meh=tel["gamma_meh"],
        gamma_liked=tel["gamma_liked"], gamma_loved=tel["gamma_loved"],
        gamma_spread=tel["gamma_loved"] - tel["gamma_hated"],
        gamma_hated_negative=bool(tel["gamma_hated"] < 0.0),
        tau_rough=tel["tau_rough"], tau_kw=tel["tau_know_well"], tau_no_clue=tel["tau_no_clue"],
        value_delta_present=d_pres, ci_present=ci_pres,
        value_delta_absent=d_abs, ci_absent=ci_abs,
        ig2=ig2, grad_probe=probe, intercept_after_train=intercept, history=hist)
    print(f"\n=== {name} [{res['minutes']}m] ===", flush=True)
    print(f"  gamma h/m/l/L: {tel['gamma_hated']:+.4f}/{tel['gamma_meh']:+.4f}/"
          f"{tel['gamma_liked']:+.4f}/{tel['gamma_loved']:+.4f}  spread {res['gamma_spread']:+.4f}  "
          f"hated<0: {res['gamma_hated_negative']}", flush=True)
    print(f"  value_delta present {d_pres:+.4f} CI{[round(x,4) for x in ci_pres]}  "
          f"absent {d_abs:+.4f} CI{[round(x,4) for x in ci_abs]}", flush=True)
    print(f"  IG2 love {ig2['pctile_love']:.3f} hate {ig2['pctile_hate']:.3f} drop {ig2['drop']:+.4f}", flush=True)
    print(f"  intercept-after-train {intercept:.3e}", flush=True)
    for k in ["Lo", "Lc", "Lkl", "Laux"]:
        p = probe.get(k, {})
        print(f"  grad[{k:4s}]->gamma_valslope {p.get('gamma_valslope',0):.3e}  "
              f"gamma_full {p.get('gamma_full',0):.3e}  tau {p.get('tau',0):.3e}  "
              f"rho_last {p.get('rho_last',0):.3e}  (val {p.get('val',0):.3f})", flush=True)
    return res


def main():
    epochs = int(os.environ.get("DIAG_EPOCHS", "3"))
    t0 = time.time()
    print("[diagnose] loading data (same split seed as pilot) ...", flush=True)
    D = L.G.load_data(); FR = L.Frozen(D); S = R1.V3Sampler(D, FR)
    seed = 0
    tr, va, _te = R1.load_split(D, n_train=2000, n_val=1000, n_test=0, seed=seed + 1234)
    tr_recs = R1.prep_users(tr, np.random.default_rng(seed), S)
    val_recs = R1.prep_users(va, np.random.default_rng(seed + 1), S)
    print(f"[diagnose] train {len(tr_recs)} / val {len(val_recs)} users  [{time.time()-t0:.0f}s]", flush=True)
    print(f"[diagnose] CENTERED_FOLD={R1.CENTERED_FOLD}  epochs/config={epochs}", flush=True)

    base = dict(lr=1e-3, batch=128, p_drop=0.5, lam_c=0.3, margin=0.5, beta_max=0.05,
                clean_frac=0.3, lam_ord=1.0, lam_aux=0.0, aux_margin=0.0, force_drop_native=False)

    def mk(**kw):
        c = dict(base); c.update(kw); return c

    configs = [
        ("C0_baseline", mk()),
        ("C1_beta0", mk(beta_max=0.0)),                        # suspect #2 KL
        ("C2_ord10", mk(lam_ord=10.0)),                        # suspect #1 drowned ordinal
        ("C3_ord100", mk(lam_ord=100.0)),                      # suspect #1 extreme
        ("C4_dropnative", mk(force_drop_native=True)),         # suspect #4 native free-ride
        ("C5_aux", mk(lam_aux=1.0)),                           # suspect #5 no direct path  <-- decisive
        ("C6_aux_beta0", mk(lam_aux=1.0, beta_max=0.0)),       # compounding #5 + #2
    ]
    results = []
    for name, cfg in configs:
        try:
            results.append(run_config(FR, S, D, tr_recs, val_recs, name, cfg, epochs, seed=seed))
        except Exception as ex:
            import traceback; traceback.print_exc()
            results.append(dict(name=name, error=str(ex), cfg=cfg))
        json.dump(dict(results=results, epochs=epochs,
                       centered_fold=R1.CENTERED_FOLD,
                       note="2k train/1k val, same split seed as pilot; NO data reduction"),
                  open(f"{OUT}/diagnosis.json", "w"), indent=2, default=float)
    print(f"\n[diagnose] DONE [{round((time.time()-t0)/60,1)}m] -> {OUT}/diagnosis.json", flush=True)


if __name__ == "__main__":
    main()
