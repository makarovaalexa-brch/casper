r"""concepts_only_curve.py -- CONCEPTS-AS-A-STANDALONE-CHANNEL scaling (author follow-on, 2026-07-24).

On the frozen ep4 i25 snapshot, VAL cohort: fold m in {1,2,4,8} CONCEPT answers ONLY (no items),
concepts = the user's top-m by SEL watch-lift on the fold-in (held te targets excluded by construction --
the C2 leak clause), values = SEL-derived graded, mean update = SEQUENTIAL Arm-A latent shifts
(the shifts are linear, so sequential application == their sum; asserted). Two cold operators reported
side by side (the two G5-fix candidate configs, author-specified):
    A "pure whitened":  beta_cold = 1.0, rho = 0    (whitened member direction only)
    B "floor blend":    beta_cold = 2.0, rho = 0.75 (75% raw popularity-bearing centroid at k=0;
                        beta 2.0 = the g5fix harness's floor-blend operating point, author gave rho only)
Per m: FULL + TAIL NDCG@10, monotonicity verdict, and the ITEMS-ONLY comparison at k=m from the
existing cold-val protocol (truncate_graded fixed-seed subsets of the canonical proc fold-in; k=2/k=8
use the training-time seeds COLD_SEED/COLD_SEED+1 so they snap to the recorded coldk2/coldk8 numbers).
Plus the MIXED row: m=2 concepts + k=2 items (concept-on-context operator: beta_ctx=5 pure whitened --
the certified +0.0084 config; z_shift on the k2 tower fold).

BELIEF ARTIFACTS NOTE (honest): Sigma/the fitted alphas play NO role here -- design (ii) is decoupled
and this analysis is mean-path only, so it runs without the belief fit (not yet launched).

SEL-GRADED VALUE CONVENTION (pre-C3, documented): within a user's selected top-m concepts, the valence
magnitude is the log-lift of concept j normalized by the user's top concept's log-lift, floored at 0.25:
    v_j = clip( log(lift_j) / log(lift_1), 0.25, 1.0 )     (lift_1 = the user's strongest lift > 1)
so the strongest behavioral concept folds at full strength and weaker ones fold graded-down. Users whose
top lift <= 1 (no positive behavioral signal) are EXCLUDED from the cohort (reported).

RESOURCE CAP (a training run owns the CPU): OMP=4 threads + LOW (idle) process priority.

Output: experiments/battery/concepts_only_curve.json + a printed table.
Usage:
  python src/instrument/concepts_only_curve.py [--snapshot PATH] [--smoke]
"""
import os
import sys
# --full_threads (author 2026-07-24: retrain stopped, battery owns the CPU): no OMP cap, normal priority
_FULL = "--full_threads" in sys.argv
_NT = str(os.cpu_count()) if _FULL else "4"
os.environ["OMP_NUM_THREADS"] = _NT
os.environ.setdefault("OPENBLAS_NUM_THREADS", _NT)
os.environ.setdefault("MKL_NUM_THREADS", _NT)
import json
import time
import argparse
import numpy as np
import torch
from scipy import sparse

torch.set_num_threads(int(_NT))

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
sys.path.insert(0, os.path.join(_ROOT, "src", "baselines"))
sys.path.insert(0, _HERE)

from train_tower_t2 import truncate_graded, COLD_SEED, log
import run_battery_phaseA as PA
from run_battery_phaseA import (build_real_ctx, build_smoke_ctx, eval_tokens, ndcg10_from_scores,
                                bootstrap_ci, load_genome, spearman, OUTDIR, SEED)
from belief_layer import build_concept_dirs, ConceptMean

assert not hasattr(sys.modules[__name__], "load_answerer")           # retired-answerer ban

M_LIST = (1, 2, 4, 8)
# cold-k truncation seeds: k2/k8 = the training-time seeds (snap to recorded coldk2/coldk8);
# k1/k4 = new fixed seeds (stable across reruns)
K_SEED = {1: COLD_SEED + 2, 2: COLD_SEED, 4: COLD_SEED + 3, 8: COLD_SEED + 1}
CONFIGS = {"pure_whitened_b1": {"beta_cold": 1.0, "rho": 0.0},
           "floor_blend_r075": {"beta_cold": 2.0, "rho": 0.75}}


def set_low_priority():
    """LOW (idle) process priority -- a training run owns the CPU."""
    try:
        import psutil
        psutil.Process().nice(psutil.IDLE_PRIORITY_CLASS)
        log("[prio] process priority -> IDLE (psutil)")
    except Exception:
        try:
            import ctypes
            ctypes.windll.kernel32.SetPriorityClass(
                ctypes.windll.kernel32.GetCurrentProcess(), 0x00000040)   # IDLE_PRIORITY_CLASS
            log("[prio] process priority -> IDLE (win32)")
        except Exception as e:                                            # non-fatal
            log(f"[prio] could not lower priority ({e}); OMP=4 cap still active")


def sel_top_concepts(ctx, members, tags, m_max):
    """Per val user: the top-m_max concepts by SEL watch-lift on the FOLD-IN (va_tr; te excluded by
    construction), with the SEL-graded values. Returns {row: (concept_idx array, v array)} for users
    with >=1 positive-lift concept (>=2 supporting fold-in items)."""
    Mm = sparse.csr_matrix((np.ones(sum(len(members[t]) for t in tags), np.float32),
                            (np.concatenate([members[t] for t in tags]),
                             np.concatenate([np.full(len(members[t]), i)
                                             for i, t in enumerate(tags)]))),
                           shape=(ctx.ni, len(tags)))
    counts = np.asarray((ctx.va_tr @ Mm).todense())                       # (n, C)
    nu = np.asarray(ctx.va_tr.sum(axis=1)).ravel().clip(min=1)
    gmass = ctx.cnt @ np.asarray(Mm.todense())
    grate = gmass / max(ctx.cnt.sum(), 1e-9)
    lift = (counts / nu[:, None]) / np.maximum(grate[None, :], 1e-12)
    lift[counts < 2] = -np.inf                                            # need >=2 supporting items
    out = {}
    for r in range(ctx.n):
        lr = lift[r]
        pos = np.flatnonzero(np.isfinite(lr) & (lr > 1.0))                # positive behavioral signal
        if len(pos) == 0:
            continue
        order = pos[np.argsort(-lr[pos])][:m_max]
        l1 = np.log(lr[order[0]])
        v = np.clip(np.log(lr[order]) / max(l1, 1e-9), 0.25, 1.0)
        out[r] = (order.astype(np.int64), v.astype(np.float32))
    return out


def concepts_only_scores(ctx, cmean, sel, rows, m):
    """Cold (k=0) concepts-only mean = SUM of the m sequential Arm-A shifts (linear -> sum; the i25
    empty fold is exactly 0). Returns per-user (full, tail) NDCG@10 under cold-parity masking (va_tr)."""
    full = np.full(ctx.n, np.nan); tail = np.full(ctx.n, np.nan)
    for st in range(0, len(rows), 500):
        chunk = rows[st:st + 500]
        Z = np.zeros((len(chunk), ctx.d), np.float32)
        for j, r in enumerate(chunk):
            cs, vs = sel[r]
            for c, v in zip(cs[:m], vs[:m]):                              # sequential == sum (linear)
                Z[j] += cmean.shift(int(c), float(v), cold=True).numpy()
        S = (torch.from_numpy(Z) @ ctx.Wd.T + ctx.bd).numpy().astype(np.float32)
        f, t = ndcg10_from_scores(S, ctx.va_tr[chunk], ctx.va_te[chunk], ctx.head_mask)
        full[np.asarray(chunk)] = f; tail[np.asarray(chunk)] = t
    return full, tail


# ============================================================================= FIX A/B diagnostics
# (coordinator 2026-07-24: two CHEAP eval-only diagnostics BEFORE Arm C-lite trains -- the m=8 crater
#  may be a correlation artifact (Fix A) or an accumulation artifact (Fix B), not a channel limit.)
COS_MAX = 0.5
Y_SCALES = (0.5, 1.0, 2.0, 5.0, 10.0)


def diversify_sel(sel_ranked, d_c, m_max=8, cos_max=COS_MAX):
    """FIX A greedy diversified selection: highest-SEL first, then each next-highest whose whitened
    direction has |cos| < cos_max to ALL already-picked. sel_ranked: {r: (cids, vals)} with a DEEP
    ranked pool (m_max=40). Returns {r: (cids, vals)} (up to m_max picked)."""
    out = {}
    D = d_c.numpy()
    for r, (cs, vs) in sel_ranked.items():
        picked = []; pv = []
        for c, v in zip(cs, vs):
            d = D[int(c)]
            if all(abs(float(d @ D[int(p)])) < cos_max for p in picked):
                picked.append(int(c)); pv.append(float(v))
            if len(picked) >= m_max:
                break
        if picked:
            out[r] = (np.asarray(picked, np.int64), np.asarray(pv, np.float32))
    return out


def mean_pairwise_cos(sel, d_c, rows, m):
    """Diagnostic: mean pairwise |cos| of the picked whitened dirs at budget m (users with >=2)."""
    D = d_c.numpy(); vals = []
    for r in rows:
        cs = sel[r][0][:m]
        if len(cs) < 2:
            continue
        G = D[cs] @ D[cs].T
        iu = np.triu_indices(len(cs), 1)
        vals.append(float(np.abs(G[iu]).mean()))
    return float(np.mean(vals)) if vals else float("nan")


def gs_orth_shifts(cs, vs, d_c, w_c, beta):
    """Gram-Schmidt variant (stricter Fix A): fold only the component of each new direction orthogonal
    to the span of the previously picked set (residual magnitude kept, NOT renormalized)."""
    basis = []
    Z = np.zeros(d_c.shape[1], np.float32)
    for c, v in zip(cs, vs):
        d = d_c[int(c)].numpy().astype(np.float64).copy()
        for b in basis:
            d -= (d @ b) * b
        Z += (beta * float(w_c[int(c)]) * float(v) * d).astype(np.float32)
        n = np.linalg.norm(d)
        if n > 1e-8:
            basis.append(d / n)
    return Z


def bayes_posterior_mean(Z0, conc_lists, d_c, v0, alpha_conc, y_scale):
    """FIX B: conjugate posterior mean. Prior N(z0, diag(v0)); each concept answer = observation along
    unit whitened d_c with target y = y_scale*v and precision alpha_conc (shakedown-fitted).
      mu = Sigma_post (z0/v0 + sum alpha y_j d_j),  Sigma_post = (diag(1/v0) + alpha D D^T)^{-1}
    Woodbury per user (m <= 8): Sigma x = v0*x - (v0*U) M^{-1} U^T (v0*x), M = I + U^T diag(v0) U,
    U = sqrt(alpha) D^T. Shrinkage is automatic: over-accumulation impossible by construction.
    NOTE: no IDF w_c here -- the per-observation precision replaces hand weighting (documented)."""
    B, d = Z0.shape
    out = Z0.clone()
    v0n = v0.numpy().astype(np.float64)
    Dn = d_c.numpy().astype(np.float64)
    for r in range(B):
        cl = conc_lists[r]
        if not cl:
            continue
        Dm = Dn[[int(c) for c, _ in cl]]                                  # (m,d)
        y = np.array([y_scale * float(v) for _, v in cl])                 # (m,)
        U = (np.sqrt(alpha_conc) * Dm).T                                  # (d,m)
        x = Z0[r].numpy().astype(np.float64) / v0n + (alpha_conc * y) @ Dm    # Lambda0 z0 + b
        Uv = U * v0n[:, None]
        M = np.eye(U.shape[1]) + U.T @ Uv
        sol = np.linalg.solve(M, U.T @ (v0n * x))
        mu = v0n * x - Uv @ sol
        out[r] = torch.from_numpy(mu.astype(np.float32))
    return out


def run_fixab(ctx, args):
    """Four-arm diagnostic (topSEL-additive / div-additive / topSEL-Bayes / div-Bayes) + GS variant +
    |cos| diagnostic + controls -> experiments/battery/concepts_only_fixAB.json"""
    from belief_layer import BeliefLayer, CH_CONC, VAGUE
    t00 = time.time()
    members = load_genome(ctx)
    tags = sorted(members.keys())
    d_c, d_raw, w_c = build_concept_dirs(ctx.Wd, members, tags)
    # fitted belief layer (SHAKEDOWN-calibrated -- labeled, not certified)
    bel = BeliefLayer(np.ones(ctx.d, np.float32))
    alpha_src = "INIT (smoke/dim-mismatch fallback)"
    if os.path.exists(args.belief_ckpt):
        try:
            blob = torch.load(args.belief_ckpt, map_location="cpu")
            bel.load_state_dict(blob["belief"])
            alpha_src = os.path.basename(args.belief_ckpt) + " (SHAKEDOWN fit)"
        except Exception as e:
            assert getattr(args, "smoke", False), f"belief ckpt load failed on a REAL run: {e}"
            bel = BeliefLayer(np.ones(ctx.d, np.float32))     # load_state_dict is NOT atomic: rebuild
    else:
        assert getattr(args, "smoke", False), f"belief ckpt missing on a REAL run: {args.belief_ckpt}"
    alpha_conc = float(bel.alphas()[CH_CONC, VAGUE])
    v0 = bel.v0().detach()
    log(f"[fixab] alphas from {alpha_src}: alpha_conc(vague)={alpha_conc:.3f} "
        f"v0 mean={float(v0.mean()):.3f} (SHAKEDOWN-calibrated, not certified)")
    # deep ranked pool (m_max=40) -> top-SEL prefix == the original run's selection (same ordering)
    sel40 = sel_top_concepts(ctx, members, tags, 40)
    sel_top = {r: (cs[:max(M_LIST)], vs[:max(M_LIST)]) for r, (cs, vs) in sel40.items()}
    sel_div = diversify_sel(sel40, d_c, m_max=max(M_LIST))
    rows = [r for r in range(ctx.n) if ctx.va_te[r].nnz > 0 and r in sel_top and r in sel_div]
    log(f"[fixab] cohort {len(rows)} (top-SEL AND div-pickable)")
    # ---- controls (HARD RULE 5) ----
    empty = [(np.empty(0, np.int64), np.empty(0, np.int64))] * ctx.n
    f_int, t_int = eval_tokens(ctx, empty, ctx.va_tr, rows=rows)
    out = {"analysis": "concepts_only_fixAB", "n_users": len(rows), "cos_max": COS_MAX,
           "belief_ckpt": os.path.basename(args.belief_ckpt),
           "alpha_conc_vague_SHAKEDOWN": alpha_conc,
           "note": "alpha from the SHAKEDOWN fit (belief_i25.pt) -- calibrated, NOT certified; "
                   "Bayes arm uses no IDF w_c (precision replaces hand weighting)",
           "intercept": {"full": float(np.nanmean(f_int[rows])), "tail": float(np.nanmean(t_int[rows]))},
           "arms": {}, "diag_mean_pairwise_abs_cos": {}}
    cmA = ConceptMean(d_c, d_raw, w_c, beta_ctx=5.0, beta_cold=1.0, floor_rho=0.0)
    # comparator bit-exact check vs the recorded curve (same operator, same users where cohorts match)
    prev_path = os.path.join(OUTDIR, "concepts_only_curve.json")
    prev = json.load(open(prev_path)) if os.path.exists(prev_path) else None
    # ---- y_scale pick: cold topSEL-Bayes m=8, argmax full ----
    ys_curve = {}
    for ys in Y_SCALES:
        Z = bayes_posterior_mean(torch.zeros(len(rows), ctx.d),
                                 [list(zip(sel_top[r][0][:8], sel_top[r][1][:8])) for r in rows],
                                 d_c, v0, alpha_conc, ys)
        S = (Z @ ctx.Wd.T + ctx.bd).numpy().astype(np.float32)
        f = np.full(len(rows), np.nan)
        for st in range(0, len(rows), 500):
            ch = rows[st:st + 500]
            ff, _ = ndcg10_from_scores(S[st:st + len(ch)], ctx.va_tr[ch], ctx.va_te[ch], None)
            f[st:st + len(ch)] = ff
        ys_curve[ys] = float(np.nanmean(f))
    y_scale = max(ys_curve, key=lambda y: ys_curve[y])
    out["y_scale_sweep_m8_full"] = {str(k): v for k, v in ys_curve.items()}
    out["y_scale"] = y_scale
    log(f"[fixab] y_scale sweep {ys_curve} -> {y_scale}")

    def score_batchZ(Zfull):
        f = np.full(ctx.n, np.nan); t = np.full(ctx.n, np.nan)
        for st in range(0, len(rows), 500):
            ch = rows[st:st + 500]
            S = (Zfull[st:st + len(ch)] @ ctx.Wd.T + ctx.bd).numpy().astype(np.float32)
            ff, tt = ndcg10_from_scores(S, ctx.va_tr[ch], ctx.va_te[ch], ctx.head_mask)
            f[np.asarray(ch)] = ff; t[np.asarray(ch)] = tt
        return f, t

    arms = {"topSEL_additive": ("add", sel_top), "div_additive": ("add", sel_div),
            "topSEL_bayes": ("bayes", sel_top), "div_bayes": ("bayes", sel_div),
            "div_gramschmidt_additive": ("gs", sel_div)}
    for name, (kind, sel) in arms.items():
        cur = {"full@10": {}, "tail@10": {}}
        for m in M_LIST:
            if kind == "add":
                f, t = concepts_only_scores(ctx, cmA, sel, rows, m)
            elif kind == "gs":
                Z = torch.zeros(len(rows), ctx.d)
                for j, r in enumerate(rows):
                    cs, vs = sel[r]
                    Z[j] = torch.from_numpy(gs_orth_shifts(cs[:m], vs[:m], d_c, w_c, beta=1.0))
                f, t = score_batchZ(Z)
            else:
                Z = bayes_posterior_mean(torch.zeros(len(rows), ctx.d),
                                         [list(zip(sel[r][0][:m], sel[r][1][:m])) for r in rows],
                                         d_c, v0, alpha_conc, y_scale)
                f, t = score_batchZ(Z)
            cur["full@10"][str(m)] = float(np.nanmean(f[rows]))
            cur["tail@10"][str(m)] = float(np.nanmean(t[rows]))
        seq = [cur["full@10"][str(m)] for m in M_LIST]
        cur["monotone_full"] = bool(all(seq[i + 1] >= seq[i] - 1e-9 for i in range(len(seq) - 1)))
        seqt = [cur["tail@10"][str(m)] for m in M_LIST]
        cur["monotone_tail"] = bool(all(seqt[i + 1] >= seqt[i] - 1e-9 for i in range(len(seqt) - 1)))
        out["arms"][name] = cur
        log(f"[fixab {name}] " + " ".join(f"m{m}={cur['full@10'][str(m)]:.4f}/"
                                          f"{cur['tail@10'][str(m)]:.4f}" for m in M_LIST))
    # comparator reproduction check (same cohort caveat: fixab cohort = topSEL AND div users)
    if prev is not None:
        out["comparator_note"] = ("topSEL_additive here is the same operator/users as "
                                  "pure_whitened_b1; exact equality only if the div-pickable filter "
                                  "removed no users -- deltas reported")
        out["comparator_delta_vs_recorded"] = {
            str(m): out["arms"]["topSEL_additive"]["full@10"][str(m)]
            - prev["configs"]["pure_whitened_b1"]["full@10"][str(m)] for m in M_LIST}
    # |cos| diagnostic
    for m in (2, 4, 8):
        out["diag_mean_pairwise_abs_cos"][str(m)] = {
            "topSEL": mean_pairwise_cos(sel_top, d_c, rows, m),
            "div": mean_pairwise_cos(sel_div, d_c, rows, m)}
    # items k2/k8 snap control (real ctx only; smoke ctx has no canonical L_val)
    if hasattr(ctx, "L_val"):
        snap = {}
        for m in (2, 8):
            Lm = truncate_graded(ctx.L_val, m, K_SEED[m])
            toks = [(Lm[i].indices.astype(np.int64), (Lm[i].data - 1).astype(np.int64))
                    for i in range(ctx.n)]
            f, _ = eval_tokens(ctx, toks, ctx.va_tr, rows=None)
            allrows = [r for r in range(ctx.n) if ctx.va_te[r].nnz > 0]
            snap[str(m)] = float(np.nanmean(f[allrows]))
        out["items_snap_control"] = {"k2": snap["2"], "k8": snap["8"],
                                     "expected": {"k2": 0.1965, "k8": 0.2584},
                                     "PASS": bool(abs(snap["2"] - 0.1965) < 5e-4
                                                  and abs(snap["8"] - 0.2584) < 5e-4)}
        log(f"[fixab] items snap control k2={snap['2']:.4f} k8={snap['8']:.4f} "
            f"PASS={out['items_snap_control']['PASS']}")
    else:
        out["items_snap_control"] = "SKIPPED (smoke ctx)"
    # mixed m2k2 through Bayes (prior mean = k2 tower fold), topSEL and div
    from run_battery_phaseA import fold_z
    Zk2 = torch.from_numpy(fold_z(ctx.enc, ctx.k2_tokens, rows))
    f_k2, _ = eval_tokens(ctx, ctx.k2_tokens, ctx.va_tr, rows=rows)
    for name, sel in (("topSEL", sel_top), ("div", sel_div)):
        Zm = bayes_posterior_mean(Zk2, [list(zip(sel[r][0][:2], sel[r][1][:2])) for r in rows],
                                  d_c, v0, alpha_conc, y_scale)
        f, t = score_batchZ(Zm)
        dmix = bootstrap_ci(f[rows] - f_k2[rows])
        out[f"mixed_m2_k2_bayes_{name}"] = {"full@10": float(np.nanmean(f[rows])),
                                            "tail@10": float(np.nanmean(t[rows])),
                                            "vs_items_k2": {"mean": dmix[0], "ci95": dmix[1]},
                                            "armA_additive_bar": 0.0125}
        log(f"[fixab mixed bayes {name}] {float(np.nanmean(f[rows])):.4f} vs k2 {dmix[0]:+.4f}")
    out["seconds"] = round(time.time() - t00, 1)
    outdir = ctx.outdir if getattr(args, "smoke", False) else OUTDIR
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, "concepts_only_fixAB.json")
    json.dump(out, open(path, "w"), indent=2)
    log(f"[out] -> {path}")
    # printed four-arm table
    print("\n=== FIX A/B FOUR-ARM TABLE (concepts-only, full/tail @ m) ===")
    print(f"intercept {out['intercept']['full']:.4f}/{out['intercept']['tail']:.4f}; "
          f"y_scale={y_scale}; alpha_conc={alpha_conc:.2f} (shakedown)")
    for name in ("topSEL_additive", "div_additive", "topSEL_bayes", "div_bayes",
                 "div_gramschmidt_additive"):
        c = out["arms"][name]
        rowtxt = " ".join(f"m{m}={c['full@10'][str(m)]:.4f}/{c['tail@10'][str(m)]:.4f}"
                          for m in M_LIST)
        print(f"{name:>26}: {rowtxt}  mono_full={c['monotone_full']} mono_tail={c['monotone_tail']}")
    print(f"|cos| diag (top vs div): " + " ".join(
        f"m{m}: {out['diag_mean_pairwise_abs_cos'][str(m)]['topSEL']:.3f}/"
        f"{out['diag_mean_pairwise_abs_cos'][str(m)]['div']:.3f}" for m in (2, 4, 8)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--snapshot", default=PA.SNAP_DEFAULT)
    ap.add_argument("--fixab", action="store_true",
                    help="run the FIX A/B diagnostics instead of the base curve")
    ap.add_argument("--belief_ckpt", default=os.path.join(_ROOT, ".cache", "instrument",
                                                          "belief_i25.pt"))
    ap.add_argument("--full_threads", action="store_true",
                    help="no OMP cap + normal priority (the CPU is ours)")
    args = ap.parse_args()
    if not args.full_threads:
        set_low_priority()
    else:
        log(f"[prio] FULL THREADS ({_NT}) at normal priority (author: battery owns the CPU)")
    t00 = time.time()
    ctx = build_smoke_ctx() if args.smoke else build_real_ctx(args.snapshot)
    ctx.Wd = ctx.decoder.weight.detach().float(); ctx.bd = ctx.decoder.bias.detach().float()
    ctx.d = ctx.enc.d_out
    if args.fixab:
        run_fixab(ctx, args)
        return
    members = load_genome(ctx)
    tags = sorted(members.keys())
    d_c, d_raw, w_c = build_concept_dirs(ctx.Wd, members, tags)

    # linearity assert: sequential shift application == summed shift (documents "sequential" honestly)
    cm_chk = ConceptMean(d_c, d_raw, w_c, beta_ctx=5.0, beta_cold=1.0, floor_rho=0.0)
    seq = cm_chk.shift(0, 1.0, cold=True) + cm_chk.shift(min(1, len(tags) - 1), 0.5, cold=True)
    summed = sum([cm_chk.shift(0, 1.0, cold=True), cm_chk.shift(min(1, len(tags) - 1), 0.5, cold=True)])
    assert torch.allclose(seq, summed), "Arm-A shifts must compose additively"

    sel = sel_top_concepts(ctx, members, tags, max(M_LIST))
    rows = [r for r in range(ctx.n) if ctx.va_te[r].nnz > 0 and r in sel]
    avail = np.array([len(sel[r][0]) for r in rows])
    log(f"[cohort] {len(rows)} val users with >=1 positive-lift SEL concept "
        f"(mean avail {avail.mean():.2f}; frac with >=8: {(avail >= 8).mean():.2f})")

    # ---- intercept (k=0, empty fold) ----
    empty = [(np.empty(0, np.int64), np.empty(0, np.int64))] * ctx.n
    f_int, t_int = eval_tokens(ctx, empty, ctx.va_tr, rows=rows)
    intercept = {"full": float(np.nanmean(f_int[rows])), "tail": float(np.nanmean(t_int[rows]))}

    out = {"analysis": "concepts_only_curve", "snapshot": os.path.basename(args.snapshot),
           "n_users": len(rows), "m_list": list(M_LIST),
           "avail_concepts": {"mean": float(avail.mean()),
                              "frac_ge_m": {str(m): float((avail >= m).mean()) for m in M_LIST}},
           "value_convention": "SEL-graded: v_j = clip(log(lift_j)/log(lift_1), 0.25, 1.0); "
                               "users with top lift <= 1 excluded",
           "belief_artifacts_note": "mean-path only (design (ii) decoupled) -- runs without the "
                                    "belief fit; Sigma plays no role in this curve",
           "intercept_k0": intercept, "configs": {}}

    # ---- concepts-only curves, both cold configs ----
    curves_full = {}
    for cname, cfg in CONFIGS.items():
        cm = ConceptMean(d_c, d_raw, w_c, beta_ctx=5.0,
                         beta_cold=cfg["beta_cold"], floor_rho=cfg["rho"])
        cur = {"beta_cold": cfg["beta_cold"], "rho": cfg["rho"], "full@10": {}, "tail@10": {},
               "vs_intercept": {}}
        fs = {}
        for m in M_LIST:
            f, t = concepts_only_scores(ctx, cm, sel, rows, m)
            fs[m] = f
            cur["full@10"][str(m)] = float(np.nanmean(f[rows]))
            cur["tail@10"][str(m)] = float(np.nanmean(t[rows]))
            d = bootstrap_ci(f[rows] - f_int[rows])
            cur["vs_intercept"][str(m)] = {"mean": d[0], "ci95": d[1]}
            log(f"[conc {cname}] m={m}: full {cur['full@10'][str(m)]:.4f} "
                f"tail {cur['tail@10'][str(m)]:.4f} vs intercept {d[0]:+.4f} {d[1]}")
        seq_f = [cur["full@10"][str(m)] for m in M_LIST]
        cur["monotone_full"] = bool(all(seq_f[i + 1] >= seq_f[i] - 1e-9 for i in range(len(seq_f) - 1)))
        cur["spearman_full_vs_m"] = spearman(list(M_LIST), seq_f)
        out["configs"][cname] = cur
        curves_full[cname] = fs

    # ---- items-only comparison at k=m (existing cold-val protocol; fixed-seed subsets) ----
    items = {"full@10": {}, "tail@10": {}}
    items_f = {}
    for m in M_LIST:
        Lk = ctx.k2_tokens if (m == 2 and hasattr(ctx, "k2_tokens") and not hasattr(ctx, "L_val")) \
            else None
        if hasattr(ctx, "L_val"):
            Lm = truncate_graded(ctx.L_val, m, K_SEED[m])
            toks = [(Lm[i].indices.astype(np.int64), (Lm[i].data - 1).astype(np.int64))
                    for i in range(ctx.n)]
        else:                                                             # smoke ctx: reuse allb prefixes
            toks = [(np.asarray(s[:m], np.int64), np.asarray(l[:m], np.int64)) for s, l in ctx.allb]
        f, t = eval_tokens(ctx, toks, ctx.va_tr, rows=rows)
        items_f[m] = f
        items["full@10"][str(m)] = float(np.nanmean(f[rows]))
        items["tail@10"][str(m)] = float(np.nanmean(t[rows]))
        log(f"[items] k={m}: full {items['full@10'][str(m)]:.4f} tail {items['tail@10'][str(m)]:.4f}")
    out["items_only_at_k"] = items
    out["items_seed_note"] = "k2/k8 = training cold-val seeds (snap to recorded coldk2/coldk8); k1/k4 new fixed seeds"

    # ---- mixed row: m=2 concepts + k=2 items (context operator, beta_ctx=5 pure whitened) ----
    if hasattr(ctx, "k2_tokens"):
        cm_ctx = ConceptMean(d_c, d_raw, w_c, beta_ctx=5.0)
        Zs = np.zeros((ctx.n, ctx.d), np.float32)
        for r in rows:
            cs, vs = sel[r]
            for c, v in zip(cs[:2], vs[:2]):
                Zs[r] += cm_ctx.shift(int(c), float(v), cold=False).numpy()
        f_mix, t_mix = eval_tokens(ctx, ctx.k2_tokens, ctx.va_tr, rows=rows, z_shift=Zs)
        d_mix = bootstrap_ci(f_mix[rows] - items_f[2][rows]) if 2 in items_f else (float("nan"), (0, 0), 0)
        out["mixed_m2_k2"] = {"full@10": float(np.nanmean(f_mix[rows])),
                              "tail@10": float(np.nanmean(t_mix[rows])),
                              "vs_items_k2": {"mean": d_mix[0], "ci95": d_mix[1]},
                              "operator": "concept-on-context beta_ctx=5 pure whitened (certified config)"}
        log(f"[mixed] m2+k2: full {out['mixed_m2_k2']['full@10']:.4f} "
            f"tail {out['mixed_m2_k2']['tail@10']:.4f} vs items-k2 {d_mix[0]:+.4f}")

    out["seconds"] = round(time.time() - t00, 1)
    outdir = ctx.outdir if args.smoke else OUTDIR
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, "concepts_only_curve.json")
    json.dump(out, open(path, "w"), indent=2)
    log(f"[out] -> {path}")

    # ---- printed table ----
    print("\n=== CONCEPTS-ONLY CURVE (val, cold k=0 operators) ===")
    print(f"cohort n={len(rows)}; intercept full {intercept['full']:.4f} / tail {intercept['tail']:.4f}")
    hdr = f"{'m':>3} | {'pureW b1 full/tail':>20} | {'floor r.75 full/tail':>21} | {'items k=m full/tail':>20}"
    print(hdr); print("-" * len(hdr))
    for m in M_LIST:
        a = out["configs"]["pure_whitened_b1"]; b = out["configs"]["floor_blend_r075"]
        print(f"{m:>3} | {a['full@10'][str(m)]:.4f} / {a['tail@10'][str(m)]:.4f}      | "
              f"{b['full@10'][str(m)]:.4f} / {b['tail@10'][str(m)]:.4f}       | "
              f"{items['full@10'][str(m)]:.4f} / {items['tail@10'][str(m)]:.4f}")
    for cname in CONFIGS:
        c = out["configs"][cname]
        print(f"monotone[{cname}] = {c['monotone_full']} (spearman {c['spearman_full_vs_m']:.2f})")
    if "mixed_m2_k2" in out:
        mx = out["mixed_m2_k2"]
        print(f"mixed m2+k2: {mx['full@10']:.4f} / {mx['tail@10']:.4f} "
              f"(vs items-k2 {mx['vs_items_k2']['mean']:+.4f})")


if __name__ == "__main__":
    main()
