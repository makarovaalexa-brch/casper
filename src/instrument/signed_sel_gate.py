r"""signed_sel_gate.py -- EVAL-ONLY TRIPLE GATE on the signed-SEL redesign (adversarial-review verdict
2026-07-25: NO RETRAIN until this passes). One batch, the tradeoff_ledger deployment harness, the
cfold_best.pt operator, the same COLD_SEED 10k val users, the realizable fixed-bank concepts-only
protocol that produced the clip-up decline 0.1304 -> 0.1160 (below the 0.1279 intercept by q8).

THE SUSPECT: the clip-up value convention v = clip(log-lift ratio, +0.25, 1) folds EVERY answerable
concept as a (weak) LIKE -- down the fixed bank most answers are lift<=1, so accumulated weak-positive
poison explains the decline. Three arms test whether SIGNED SEL fixes it and whether the signal is
taste or a popularity/volume artifact:

  ARM 1 SIGNED-EVAL   sign-aware binning, NO fitting: positive lift -> the EXISTING graded like value
                      (unchanged); significantly negative lift (r <= r_neg, solid support) -> negative
                      value graded by |log-lift| (the tower flip gate already proved -v is repulsive:
                      -0.2012 CI-clean); lift~1 with solid support -> honest neutral v=0; thin support
                      -> REFUSE (no fold, budget burned).
  ARM 2 POP-COUNTERFEIT  the SAME binning rule computed from counterfeit counts with the taste residual
                      DESTROYED: each user's fold-in watch set is re-sampled uniformly WITHIN
                      popularity-decile strata (volume and popularity profile preserved exactly, item
                      identity randomized). Ask/answer structure kept REAL (identical interviews);
                      only the folded values/bins are counterfeit. If this reproduces Arm 1's gain,
                      the signed signal is a volume x popularity artifact.
  ARM 3 MEH-ONLY      every non-like folded as honest neutral v=0 (no dislikes, no clip-up); likes
                      unchanged. Isolates whether the old decline is CLIP-specific.
  COMPARATOR          the existing clip-up convention, CANONICAL-SNAPPED against the committed ledger
                      row (0.1304/0.1291/0.1242/0.1160).

PRE-REGISTERED (computed from the TRAIN-user lift distribution ONLY, written to the JSON BEFORE any
eval scoring): sig-negative cutoff r_neg (train 25th pct of log-lift among answerable pairs), dislike
saturation scale s_neg (train 10th pct magnitude), solid-support min expected member count, neutral
band (r_neg, 0]. DECISION RULES (pre-registered):
  SUPPORTED iff (i) Arm1 full@q >= its q2 level - eps for all q AND q16 > q2;
            AND (ii) Arm2 gain over clip-up at q16 < 70% of Arm1's gain;
            AND (iii) Arm3 q16 >= Arm3 q2 - eps while clip-up declines (decline is clip-specific).
  Any other pattern = KILL or RECLASSIFY; reported honestly. eps = 0.002.
Also: per-arm answer-type counts (like/meh/dislike/refuse) at each q; KT-A3 leak probe (ridge readout
R^2 from the q8 fold latent to log user volume, signed vs clip-up -- an R^2 jump = volume leak).

HARD RULES: no truncation (train thresholds over ALL 140,768 train users; eval over the full 10k
cohort); commit before run; intercept + canonical-snap controls. Eval-only; OMP=4 + idle priority by
default (the DAE/MultVAE snap owns the CPU).

Usage:
  python src/instrument/signed_sel_gate.py --smoke
  python src/instrument/signed_sel_gate.py [--full_threads]
"""
import os
import sys
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

import metrics as M
from train_tower_t2 import log
import run_battery_phaseA as PA
from run_battery_phaseA import (build_real_ctx, build_smoke_ctx, ndcg10_from_scores, bootstrap_ci,
                                load_genome, OUTDIR, SEED)
from belief_layer import build_concept_dirs
from concept_fold import ConceptFoldNet, build_member_matrix

assert not hasattr(sys.modules[__name__], "load_answerer")

BUDGETS = (2, 4, 8, 16)
EPS = 0.002                      # pre-registered flat/trend tolerance
MIN_EXPECT = 3.0                 # v1 solid-support bar (kept for the arm3 row's provenance)
LEDGER_SNAP = {2: 0.1304, 4: 0.1291, 8: 0.1242, 16: 0.1160}   # committed clite concepts-only row
N_POP_STRATA = 10
# ---- AMENDMENTS (author rulings 2026-07-25) ----
# (1) OOD ASYMMETRY: C-lite trained on v in [0.25,1] ONLY -> dislike/neutral are OOD for the module.
#     Arm1 improving = STRONG SUPPORT; Arm1 flat/worse = INCONCLUSIVE (not a kill). The ONLY hard KILL
#     is Arm2 reproducing >= 70% of any Arm1 gain. A signed RETRAIN is planned regardless (fair
#     in-envelope test) -- NOT launched from here.
# (2) SIGNED VALUE = the R2-validated bpool_r2 construction (Jul 18: SEL 0.376 + VAL 0.042 of affinity
#     variance), ported EXACTLY:
#         SEL = log2((n_c + 0.5) / (e_c + 0.5))          [watch-lift, add-0.5 smoothing]
#         VAL = (n_c / (n_c + LAM)) * mean(rating - item_mean),  LAM = 3.0
#         REFUSE iff EXPO = e_c < TAU,  TAU = 1.5        [bpool_r2's own refusal rule]
#     combined v_raw = SEL + VAL (unweighted sum of the two validated components -- the only
#     non-fitted combination; SEL-dominated per the R2 ordering), fold value
#     v = clip(v_raw / S, -1, 1), S = train p90 of |v_raw| (pre-registered scale).
#     CLIP PROVENANCE (for the record): bpool_r2 never clipped; the [0.25,1] clip entered DOWNSTREAM in
#     sel_top_concepts() (src/instrument/concepts_only_curve.py, commit 7f45fd2) and propagated into
#     concept_fold.make_example/quick_val and train_tower_t2.sel_value_to_level.
# (3) Circularity downgraded: SEL from fold-in-only w/ targets excluded = generalization; the
#     VALUE-PERMUTATION control kept as a cheap sanity row.
# (4) KT-A4 (humans/transfer) OUT OF SCOPE; the KT-A3 volume-leak probe stays.
LAM_VAL = 3.0
TAU_REF = 1.5


def set_low_priority():
    try:
        import ctypes
        ctypes.windll.kernel32.SetPriorityClass(
            ctypes.windll.kernel32.GetCurrentProcess(), 0x00000040)
        log("[prio] IDLE priority (snap owns the CPU); --full_threads to lift")
    except Exception:
        pass


# ============================================================================= pre-registration v2
def prereg_selval(ctx, Mm, pexp, smoke=False, chunk=20000):
    """AMENDED pre-registration: the bpool_r2 SEL+VAL construction evaluated over ALL train users
    (graded, all bands, train-only item means) -> the fold scale S = p90 |SEL+VAL| on answerable
    (n>=2, EXPO>=TAU) cells. No eval data touched."""
    if smoke or not hasattr(ctx, "raw"):
        return {"source": "SMOKE synthetic", "S_scale_p90_abs": 1.5, "TAU_refuse": TAU_REF,
                "LAM_val": LAM_VAL, "item_mean_source": "n/a"}
    import pandas as pd
    df = ctx.raw[ctx.raw["userId"].isin(ctx.tr_set)]
    sid = df["movieId"].map(ctx.show2id)
    ok = sid.notna()
    sid = sid[ok].astype(np.int64).values
    stars = df.loc[ok, "rating"].values.astype(np.float32)
    uid, _ = pd.factorize(df.loc[ok, "userId"].values)
    n_u = int(uid.max()) + 1
    imm = np.zeros(ctx.ni); ctt = np.zeros(ctx.ni)
    np.add.at(imm, sid, stars); np.add.at(ctt, sid, 1.0)
    item_mean = np.where(ctt > 0, imm / np.maximum(ctt, 1), stars.mean())
    resid = (stars - item_mean[sid]).astype(np.float32)
    Xbin = sparse.csr_matrix((np.ones(len(sid), np.float32), (uid, sid)), shape=(n_u, ctx.ni))
    Xres = sparse.csr_matrix((resid, (uid, sid)), shape=(n_u, ctx.ni))
    nu = np.asarray(Xbin.sum(axis=1)).ravel()
    vals = []
    npairs = 0
    for st in range(0, n_u, chunk):
        nb = np.asarray((Xbin[st:st + chunk] @ Mm).todense(), np.float32)
        rb = np.asarray((Xres[st:st + chunk] @ Mm).todense(), np.float32)
        e = nu[st:st + chunk, None] * pexp[None, :]
        selb = np.log2((nb + 0.5) / (e + 0.5))
        mrb = np.divide(rb, nb, out=np.zeros_like(rb), where=nb > 0)
        valb = (nb / (nb + LAM_VAL)) * mrb
        m = (nb >= 2) & (e >= TAU_REF)
        vals.append((selb + valb)[m].astype(np.float32))
        npairs += int(m.sum())
    vraw = np.concatenate(vals)
    prereg_selval.item_mean = item_mean                      # stashed for the eval pass (train-only)
    return {"source": f"ALL {n_u} train users, {npairs:,} answerable cells (n>=2, EXPO>=TAU)",
            "formula": "v_raw = SEL + VAL; SEL=log2((n+.5)/(e+.5)); "
                       "VAL=(n/(n+3))*mean(star - train item mean); REFUSE iff e < 1.5",
            "S_scale_p90_abs": float(np.percentile(np.abs(vraw), 90)),
            "vraw_quantiles": {q: float(np.percentile(vraw, q)) for q in (5, 25, 50, 75, 95)},
            "TAU_refuse": TAU_REF, "LAM_val": LAM_VAL,
            "item_mean_source": "TRAIN users only (leak-free)",
            "clip_provenance": "bpool_r2 never clipped; [0.25,1] entered downstream in "
                               "sel_top_concepts (concepts_only_curve.py, 7f45fd2) -> "
                               "concept_fold.make_example/quick_val + train_tower_t2.sel_value_to_level"}


def eval_selval(items_list, stars_list, item_mean, Mm, pexp, ni):
    """bpool_r2 components for the EVAL cohort from the all-bands fold-in (te excluded upstream).
    Returns (SEL, VAL, EXPO, n_c) dense (n_users, C)."""
    n = len(items_list); C = Mm.shape[1]
    rows = []; cols = []; bdat = []; rdat = []
    for r, (its, st) in enumerate(zip(items_list, stars_list)):
        rows.extend([r] * len(its)); cols.extend(its.tolist())
        bdat.extend([1.0] * len(its)); rdat.extend((st - item_mean[its]).tolist())
    Xb = sparse.csr_matrix((np.asarray(bdat, np.float32), (rows, cols)), shape=(n, ni))
    Xr = sparse.csr_matrix((np.asarray(rdat, np.float32), (rows, cols)), shape=(n, ni))
    nc = np.asarray((Xb @ Mm).todense(), np.float32)
    rs = np.asarray((Xr @ Mm).todense(), np.float32)
    nu = np.asarray(Xb.sum(axis=1)).ravel()
    E = nu[:, None] * pexp[None, :]
    SEL = np.log2((nc + 0.5) / (E + 0.5))
    mr = np.divide(rs, nc, out=np.zeros_like(rs), where=nc > 0)
    VAL = (nc / (nc + LAM_VAL)) * mr
    return SEL, VAL, E.astype(np.float32), nc


def train_thresholds(ctx, Mm, grate, smoke=False):
    """r = log(count/E) distribution over ALL train users x answerable concepts (count>=2).
    Returns the pre-registered thresholds dict. NO eval data touched."""
    if smoke or not hasattr(ctx, "raw"):
        rs = np.random.RandomState(1).normal(-0.1, 0.6, 20000)          # synthetic stand-in (labeled)
        src = "SMOKE synthetic"
        nu_med = 20.0
    else:
        train = M.load_train(ctx.ni, os.path.join(_ROOT, "data", "ml-25m", "proc"))
        counts = np.asarray((train @ Mm).todense(), np.float32)          # (140768, C)
        nu = np.asarray(train.sum(axis=1)).ravel().clip(min=1)
        E = nu[:, None] * grate[None, :]
        ans = counts >= 2
        with np.errstate(divide="ignore"):
            r = np.log(np.maximum(counts, 1e-9) / np.maximum(E, 1e-12))
        rs = r[ans]
        src = f"ALL {train.shape[0]} train users, {int(ans.sum()):,} answerable pairs"
        nu_med = float(np.median(nu))
        # the train quantile MIN_EXPECT corresponds to (context for the pre-registered constant)
        e_ans = E[ans]
        min_expect_q = float((e_ans < MIN_EXPECT).mean())
    r_neg = float(np.percentile(rs, 25))
    if r_neg >= -0.05:
        r_neg = -0.35                                                    # pre-registered fallback
    s_neg = float(abs(np.percentile(rs, 10)))
    s_neg = max(s_neg, 0.5)
    th = {"source": src, "r_neg_cutoff_p25": r_neg, "s_neg_scale_p10_abs": s_neg,
          "min_expected_members": MIN_EXPECT,
          "neutral_band": f"({r_neg:.4f}, 0]", "eps": EPS,
          "train_median_volume": nu_med}
    if not smoke and hasattr(ctx, "raw"):
        th["min_expect_train_quantile"] = min_expect_q
    return th


# ============================================================================= per-arm valuation
def bin_answers(r, E, th):
    """(r, E) -> (bin, v). bins: like / meh / dislike / refuse. The SIGNED rule (Arm 1)."""
    if r > 0:
        return "like", None                                              # v filled from clip-up grade
    if E < th["min_expected_members"]:
        return "refuse", 0.0
    if r <= th["r_neg_cutoff_p25"]:
        return "dislike", -float(np.clip(abs(r) / th["s_neg_scale_p10_abs"], 0.25, 1.0))
    return "meh", 0.0


def build_arm_values(ctx, sh_counts, sh_E, clip_v, th, arm):
    """Per (user, concept): (fold?, v, bin) under the arm's rule. clip_v = existing graded value."""
    with np.errstate(divide="ignore"):
        r = np.log(np.maximum(sh_counts, 1e-9) / np.maximum(sh_E, 1e-12))
    n, C = sh_counts.shape
    V = np.zeros((n, C), np.float32); F = np.zeros((n, C), bool)
    B = np.zeros((n, C), np.int8)                                        # 0 like 1 meh 2 dislike 3 refuse
    like = r > 0
    if arm == "clipup":
        F[:] = True; V[:] = clip_v; B[:] = 0
        return F, V, B
    if arm == "meh_only":
        F[:] = True
        V[like] = clip_v[like]; B[like] = 0
        V[~like] = 0.0; B[~like] = 1
        return F, V, B
    # signed (arm1 real counts / arm2 counterfeit counts)
    solid = sh_E >= th["min_expected_members"]
    neg = (r <= th["r_neg_cutoff_p25"]) & solid & ~like
    meh = (~like) & (~neg) & solid
    refuse = (~like) & (~solid)
    F[like] = True; V[like] = clip_v[like]; B[like] = 0
    F[meh] = True; V[meh] = 0.0; B[meh] = 1
    F[neg] = True
    V[neg] = -np.clip(np.abs(r[neg]) / th["s_neg_scale_p10_abs"], 0.25, 1.0)
    B[neg] = 2
    F[refuse] = False; B[refuse] = 3
    return F, V, B


# ============================================================================= the deployment walk
def walk_and_score(ctx, net, bank, answerable, F, V, B, rows, d):
    """Fixed-bank concepts-only interview (identical ask order for ALL arms): ask bank[j] in global
    order; every ask burns budget; ANSWERED iff structurally answerable (real count>=2); folded iff
    the arm folds it (F). Returns per-q (full, tail, type-counts)."""
    out = {}
    ev = {q: [] for q in BUDGETS}
    tc = {q: {"like": 0, "meh": 0, "dislike": 0, "refuse": 0, "unanswerable": 0} for q in BUDGETS}
    qmax = max(BUDGETS)
    binname = {0: "like", 1: "meh", 2: "dislike", 3: "refuse"}
    for r in rows:
        folded = []
        counts_local = {"like": 0, "meh": 0, "dislike": 0, "refuse": 0, "unanswerable": 0}
        for q in range(1, qmax + 1):
            if q - 1 >= len(bank):                                       # bank exhausted (smoke only)
                counts_local["unanswerable"] += 1
                if q in BUDGETS:
                    ev[q].append(list(folded))
                    for k in tc[q]:
                        tc[q][k] += counts_local[k]
                continue
            c = int(bank[q - 1])                                         # global order, ask burns
            if not answerable[r, c]:
                counts_local["unanswerable"] += 1
            else:
                b = binname[int(B[r, c])]
                counts_local[b] += 1
                if F[r, c]:
                    folded.append((c, float(V[r, c])))
            if q in BUDGETS:
                ev[q].append(list(folded))
                for k in tc[q]:
                    tc[q][k] += counts_local[k]
    for q in BUDGETS:
        full = np.full(len(rows), np.nan); tail = np.full(len(rows), np.nan)
        Zq = torch.zeros(len(rows), d)
        with torch.no_grad():
            for st in range(0, len(rows), 500):
                chunk = list(range(st, min(st + 500, len(rows))))
                cids = []; cvals = []
                for j in chunk:
                    ci, vv = ConceptFoldNet.canonical_order([c for c, _ in ev[q][j]],
                                                            [v for _, v in ev[q][j]])
                    cids.append(ci); cvals.append(vv)
                Z = net(torch.zeros(len(chunk), d), cids, cvals)
                Zq[chunk[0]:chunk[-1] + 1] = Z
                S = (Z @ ctx.Wd.T + ctx.bd).numpy().astype(np.float32)
                rr = [rows[j] for j in chunk]
                f, t = ndcg10_from_scores(S, ctx.va_tr[rr], ctx.va_te[rr], ctx.head_mask)
                full[chunk] = f; tail[chunk] = t
        nq = len(rows)
        out[q] = {"full@10": float(np.nanmean(full)), "tail@10": float(np.nanmean(tail)),
                  "types_per_user": {k: round(v / nq, 3) for k, v in tc[q].items()},
                  "_full_vec": full, "_Z": Zq}
    return out


def ridge_r2(Z, y, lam=1.0, seed=SEED):
    """Ridge readout R^2 (fit 80%, score 20%) from fold latent to y (the KT-A3 leak probe)."""
    rng = np.random.default_rng(seed)
    n = Z.shape[0]; idx = rng.permutation(n); ntr = int(0.8 * n)
    tr, te = idx[:ntr], idx[ntr:]
    X = Z.numpy().astype(np.float64); yv = np.asarray(y, np.float64)
    Xm = X[tr].mean(0); ym = yv[tr].mean()
    Xc = X[tr] - Xm; yc = yv[tr] - ym
    w = np.linalg.solve(Xc.T @ Xc + lam * np.eye(X.shape[1]), Xc.T @ yc)
    pred = (X[te] - Xm) @ w + ym
    ss = float(((yv[te] - pred) ** 2).sum()); st = float(((yv[te] - yv[te].mean()) ** 2).sum())
    return 1.0 - ss / max(st, 1e-12)


# ============================================================================= main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--ckpt", default=os.path.join(_ROOT, ".cache", "instrument", "cfold_best.pt"))
    ap.add_argument("--snapshot", default=PA.SNAP_DEFAULT)
    ap.add_argument("--out_tag", default="", help="suffix for the output JSON (retrain acceptance)")
    ap.add_argument("--full_threads", action="store_true")
    args = ap.parse_args()
    if not args.full_threads:
        set_low_priority()
    t00 = time.time()
    ctx = build_smoke_ctx() if args.smoke else build_real_ctx(args.snapshot)
    ctx.Wd = ctx.decoder.weight.detach().float(); ctx.bd = ctx.decoder.bias.detach().float()
    ctx.d = ctx.enc.d_out
    members = load_genome(ctx)
    tags = sorted(members.keys())
    d_c, d_raw, w_c = build_concept_dirs(ctx.Wd, members, tags)
    if args.smoke:
        net = ConceptFoldNet(len(tags), d=ctx.d, h=32, conc_init=d_c.numpy())
        with torch.no_grad():
            for p in net.mlp[-1].parameters():
                p.add_(torch.randn_like(p) * 0.05)
    else:
        blob = torch.load(args.ckpt, map_location="cpu")
        assert blob["tags"] == tags, "concept vocabulary drift"
        net = ConceptFoldNet(len(tags), d=ctx.d, h=blob["hidden"])
        net.load_state_dict(blob["net"])
    net.eval()
    Mm = build_member_matrix(members, tags, ctx.ni)
    gmass = ctx.cnt @ np.asarray(Mm.todense())
    grate = gmass / max(ctx.cnt.sum(), 1e-9)
    bank = np.argsort(-gmass)                                            # the ledger's global order
    # ---- eval-user REAL counts / E / clip-up values (fold-in only; te excluded by construction) ----
    counts = np.asarray((ctx.va_tr @ Mm).todense(), np.float32)
    nu = np.asarray(ctx.va_tr.sum(axis=1)).ravel().clip(min=1)
    E = (nu[:, None] * grate[None, :]).astype(np.float32)
    answerable = counts >= 2
    lift = (counts / nu[:, None]) / np.maximum(grate[None, :], 1e-12)
    top = np.where(lift > 1.0, lift, 1.0).max(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        cv = np.log(np.maximum(lift, 1.0)) / np.maximum(np.log(np.maximum(top, 1.0 + 1e-9)),
                                                        1e-9)[:, None]
    clip_v = np.clip(np.nan_to_num(cv, nan=0.25), 0.25, 1.0).astype(np.float32)
    rows = [r for r in range(ctx.n) if ctx.va_te[r].nnz > 0]
    # ---- AMENDED PRE-REGISTRATION (train-only; written to disk BEFORE scoring) ----
    th = train_thresholds(ctx, Mm, grate, smoke=args.smoke)          # v1 rules (arm3 provenance only)
    pexp = grate                                                     # = Mbin @ p_item (same quantity)
    sv = prereg_selval(ctx, Mm, pexp, smoke=args.smoke)
    item_mean = getattr(prereg_selval, "item_mean",
                        np.full(ctx.ni, 3.5, np.float32))            # smoke fallback
    outdir = ctx.outdir if args.smoke else OUTDIR
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, f"signed_sel_gate{(chr(95)+args.out_tag) if args.out_tag else ''}.json")
    prereg = {"gate": "signed_sel_triple_gate_AMENDED",
              "amendments": "2026-07-25 author rulings: OOD asymmetry (arm1 worse = INCONCLUSIVE, "
                            "only arm2>=70% = HARD KILL); value = bpool_r2 SEL+VAL port; "
                            "value-permutation sanity row kept; KT-A4 dropped",
              "pre_registered_selval": sv, "v1_thresholds_for_arm3_row": th,
              "decision_rules": {
                  "HARD_KILL_iff": "arm2 gain over clipup at q16 >= 0.70 * arm1 gain (given arm1 gain>0)",
                  "STRONG_SUPPORT_iff": "arm1 full@q >= arm1@q2 - eps for all q AND q16 > q2",
                  "otherwise": "INCONCLUSIVE (dislike/neutral OOD for the [0.25,1]-trained module); "
                               "signed retrain proceeds regardless, gated by counterfeit-eval on the "
                               "retrained model -- NOT launched from here",
                  "eps": EPS, "arm2_fraction": 0.70},
              "status": "PRE-REGISTERED (AMENDED), scoring not yet run"}
    json.dump(prereg, open(path, "w"), indent=2)
    log(f"[prereg AMENDED] written BEFORE scoring -> {path}")
    log(f"[prereg] S={sv.get('S_scale_p90_abs'):.4f} TAU={TAU_REF} LAM={LAM_VAL}")
    # ---- eval SEL+VAL components from the ALL-BANDS fold-in (te excluded upstream; bpool_r2's
    #      'watched' = rated at any band -- richer than the likes-only va_tr the clip-up used) ----
    items_list = [np.asarray(s, np.int64) for s, l in ctx.allb]
    stars_list = [((np.asarray(l, np.float64) + 1.0) / 2.0).astype(np.float32) for s, l in ctx.allb]
    SELe, VALe, Ee, nce = eval_selval(items_list, stars_list, item_mean, Mm, pexp, ctx.ni)
    S_scale = float(sv["S_scale_p90_abs"])
    # ---- ARM 2 counterfeit (stratified within-decile permutation of the ALL-BANDS watch set;
    #      counterfeit items carry item-mean ratings -> VAL residual = 0 by construction) ----
    strata = np.digitize(ctx.cnt, np.quantile(ctx.cnt, np.linspace(0, 1, N_POP_STRATA + 1)[1:-1]))
    stratum_items = [np.flatnonzero(strata == s) for s in range(N_POP_STRATA)]
    rng = np.random.default_rng(SEED)
    items_cf = []; stars_cf = []
    for r in range(ctx.n):
        its = items_list[r]
        picked = []
        for s in range(N_POP_STRATA):
            k = int((strata[its] == s).sum())
            if k:
                pool = stratum_items[s]
                picked.append(rng.choice(pool, size=min(k, len(pool)), replace=False))
        cf = np.concatenate(picked) if picked else np.empty(0, np.int64)
        items_cf.append(cf.astype(np.int64)); stars_cf.append(item_mean[cf].astype(np.float32))
    SELc, VALc, Ec, ncc = eval_selval(items_cf, stars_cf, item_mean, Mm, pexp, ctx.ni)

    def signed_arm(SELx, VALx, Ex):
        """AMENDED signed valuation: v = clip((SEL+VAL)/S, -1, 1); REFUSE iff EXPO < TAU.
        Reporting bins: like v>0.1 / meh |v|<=0.1 / dislike v<-0.1."""
        vraw = SELx + VALx
        V = np.clip(vraw / S_scale, -1.0, 1.0).astype(np.float32)
        F = Ex >= TAU_REF
        B = np.where(V > 0.1, 0, np.where(V < -0.1, 2, 1)).astype(np.int8)
        B[~F] = 3
        return F, V, B

    F1, V1, B1 = signed_arm(SELe, VALe, Ee)
    F2, V2, B2 = signed_arm(SELc, VALc, Ec)
    # value-permutation sanity row: same fold set as arm1, values permuted within user across the
    # concepts actually folded in the q16 window of the bank
    bank16 = set(int(c) for c in bank[:max(BUDGETS)])
    Vp = V1.copy()
    rngp = np.random.default_rng(SEED + 7)
    for r in rows:
        cs = [c for c in bank16 if answerable[r, c] and F1[r, c]]
        if len(cs) > 1:
            vv = Vp[r, cs]
            Vp[r, cs] = vv[rngp.permutation(len(cs))]
    # ---- the arms (identical interviews -- answered set fixed at the structural rule; only the
    #      valuation differs) ----
    arms_def = {
        "clipup": build_arm_values(ctx, counts, E, clip_v, th, "clipup"),
        "arm1_signed": (F1, V1, B1),
        "arm1p_value_permuted": (F1, Vp, B1),
        "arm2_pop_counterfeit": (F2, V2, B2),
        "arm3_meh_only": build_arm_values(ctx, counts, E, clip_v, th, "meh_only"),
    }
    # intercept control
    empty = [(np.empty(0, np.int64), np.empty(0, np.int64))] * ctx.n
    from run_battery_phaseA import eval_tokens
    f_int, t_int = eval_tokens(ctx, empty, ctx.va_tr, rows=rows)
    res = dict(prereg)
    res["status"] = "SCORED"
    res["n_users"] = len(rows)
    res["intercept"] = {"full": float(np.nanmean(f_int[rows])), "tail": float(np.nanmean(t_int[rows]))}
    arm_out = {}
    Zq8 = {}
    for name, (F, V, B) in arms_def.items():
        t0 = time.time()
        cur = walk_and_score(ctx, net, bank, answerable, F, V, B, rows, ctx.d)
        Zq8[name] = cur[8]["_Z"]
        arm_out[name] = {str(q): {k: v for k, v in cur[q].items() if not k.startswith("_")}
                         for q in BUDGETS}
        log(f"[{name}] " + " ".join(f"q{q}={cur[q]['full@10']:.4f}" for q in BUDGETS)
            + f" ({(time.time()-t0)/60:.1f}m) types@8={cur[8]['types_per_user']}")
        arm_out[name]["_fullvecs"] = {q: cur[q]["_full_vec"] for q in BUDGETS}
    # ---- canonical snap: clip-up vs the committed ledger row ----
    if not args.smoke:
        snap = {str(q): arm_out["clipup"][str(q)]["full@10"] - LEDGER_SNAP[q] for q in BUDGETS}
        res["snap_clipup_vs_ledger"] = {"deltas": snap,
                                        "PASS": bool(all(abs(v) < 5e-4 for v in snap.values()))}
        log(f"[snap] clip-up vs ledger row: {snap}")
    # ---- AMENDED decision rules (OOD asymmetry; only arm2 can hard-kill) ----
    a1 = {q: arm_out["arm1_signed"][str(q)]["full@10"] for q in BUDGETS}
    a2 = {q: arm_out["arm2_pop_counterfeit"][str(q)]["full@10"] for q in BUDGETS}
    a3 = {q: arm_out["arm3_meh_only"][str(q)]["full@10"] for q in BUDGETS}
    cu = {q: arm_out["clipup"][str(q)]["full@10"] for q in BUDGETS}
    strong = bool(all(a1[q] >= a1[2] - EPS for q in BUDGETS) and a1[16] > a1[2])
    g1 = a1[16] - cu[16]; g2 = a2[16] - cu[16]
    hard_kill = bool(g1 > 0 and g2 >= 0.70 * g1)
    clip_specific = bool(a3[16] >= a3[2] - EPS and cu[16] < cu[2] - EPS)   # informational
    if hard_kill:
        verdict = "HARD KILL (popularity counterfeit reproduces >=70% of the signed gain)"
    elif strong:
        verdict = "STRONG SUPPORT (arm1 holds its q2 level through q16 and trends up)"
    else:
        verdict = ("INCONCLUSIVE (OOD asymmetry: dislike/neutral are out-of-envelope for the "
                   "[0.25,1]-trained module) -- signed retrain proceeds as planned, gated by "
                   "counterfeit-eval on the retrained model")
    d16 = bootstrap_ci(arm_out["arm1_signed"]["_fullvecs"][16] - arm_out["clipup"]["_fullvecs"][16])
    dperm = bootstrap_ci(arm_out["arm1_signed"]["_fullvecs"][16]
                         - arm_out["arm1p_value_permuted"]["_fullvecs"][16])
    for name in arm_out:
        del arm_out[name]["_fullvecs"]
    res["arms"] = arm_out
    res["decision"] = {"strong_support": strong, "hard_kill_arm2": hard_kill,
                       "arm1_gain_q16": g1, "arm2_gain_q16": g2,
                       "clip_specific_informational": clip_specific,
                       "arm1_minus_clipup_q16": {"mean": d16[0], "ci95": d16[1]},
                       "arm1_minus_value_permuted_q16": {"mean": dperm[0], "ci95": dperm[1]},
                       "VERDICT": verdict}
    # ---- KT-A3 leak probe ----
    logvol = np.log1p(nu[np.asarray(rows)])
    r2 = {name: ridge_r2(Zq8[name], logvol) for name in ("clipup", "arm1_signed")}
    res["kt_a3_leak_probe"] = {"ridge_R2_logvol_from_q8_latent": r2,
                               "leak_tell": bool(r2["arm1_signed"] > r2["clipup"] + 0.1)}
    res["seconds"] = round(time.time() - t00, 1)
    json.dump(res, open(path, "w"), indent=2, default=float)
    log(f"[out] -> {path}")
    print(f"\n=== SIGNED-SEL TRIPLE GATE, AMENDED (intercept {res['intercept']['full']:.4f}) ===")
    for name in ("clipup", "arm1_signed", "arm1p_value_permuted", "arm2_pop_counterfeit",
                 "arm3_meh_only"):
        c = arm_out[name]
        print(f"{name:>22}: " + " ".join(f"q{q}={c[str(q)]['full@10']:.4f}" for q in BUDGETS))
    print(f"strong={strong} | hard_kill={hard_kill} (g1={g1:+.4f} g2={g2:+.4f}) | "
          f"clip-specific(info)={clip_specific}")
    print(f"arm1-clipup@q16 {d16[0]:+.4f} CI {d16[1]} | arm1-perm@q16 {dperm[0]:+.4f} CI {dperm[1]} "
          f"| leak R2 {r2}")
    print(f"VERDICT: {verdict}")


if __name__ == "__main__":
    main()
