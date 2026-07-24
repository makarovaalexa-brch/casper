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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--snapshot", default=PA.SNAP_DEFAULT)
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
