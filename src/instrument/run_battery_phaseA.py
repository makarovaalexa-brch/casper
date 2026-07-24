"""run_battery_phaseA.py -- Instrument gate battery, PHASE A (no covariance layer needed), on a frozen
i25 tower snapshot. Spec: docs/design/GATE_BATTERY_INSTRUMENT.md + DESIGN_SHEET_STEP2_BELIEF.md.

REDO-OR-PROCEED verdict run (author priority pivot 2026-07-24): gates BEFORE more training.
Target snapshot: .cache/instrument/t2i25_EP4_SNAP.pt (i25 arm A; val full 0.3435 / coldk2 0.1965 /
coldk8 0.2584). Scoring only -- NO training. Thread-cap OMP=6.

GATES (most decisive first; one JSON per gate -> experiments/battery/):
  G3a FLIP + INTENSITY STAIRCASE  (sign fidelity; the sign-blind-fold killer)
  G3b VALUE-NONINERT              (neutralize levels to 3 stars; the value-inertness wall)
  G6  EXISTENTIAL NULLS           (wrong-user / shuffled-levels / placebo-constant / duplicate assert)
  G9  INGESTION-ISOLATION         (fixed user-independent bank; content-driven rise; vs random bank)
  G5  CONCEPT ARM-A OPERATOR      (whitened-centroid latent shift vs member-bag comparator)

ANSWER ENVIRONMENT (post-Jul-22 rules): values = REAL ratings only (raw stars re-derived to half-star
levels); concepts = genome-tag membership + SEL behavioral watch-lift on the FOLD-IN (held targets
EXCLUDED -- C2 leak clause); NO LLM artifacts anywhere; `load_answerer` is retired and asserted absent.

FOLD ENVIRONMENT for G3/G6 (documented): the ALL-BANDS graded reconstruction -- every raw catalog rating
of a val user EXCLUDING their held te-half items (the canonical proc fold-in is likes-only; the flip /
neutralize / shuffle tests are about DISLIKE and intensity handling, which need the full band range).
G9 folds true graded answers to a fixed bank; masking = full canonical tr fold-in UNION revealed tokens
(tower cold parity + credit-neutral masking of asked items).

STATS: paired per-user deltas with a 10k-resample bootstrap 95% CI (the standing protocol rule).
Usage:
  python src/instrument/run_battery_phaseA.py --smoke              # synthetic code-path check per gate
  python src/instrument/run_battery_phaseA.py [--only g3a,g6] [--snapshot PATH] [--force]
"""
import os
os.environ["OMP_NUM_THREADS"] = "6"
os.environ.setdefault("OPENBLAS_NUM_THREADS", "6")
os.environ.setdefault("MKL_NUM_THREADS", "6")
import sys
import json
import time
import argparse
import numpy as np
import torch
from scipy import sparse

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
sys.path.insert(0, os.path.join(_ROOT, "src", "baselines"))
sys.path.insert(0, _HERE)
import metrics as M
import recvae as R
from train_tower_t2 import (reproduce_partition, build_graded_eval_matrix, truncate_graded,
                            pack_tokens, build_model, COLD_SEED, LIKE_MIN_LEVEL, NLEV,
                            level_to_sv, log)
import train_tower_t2 as T2

# Retired-answerer ban (Jul-22 audit): the retired loader must not exist anywhere in the import chain.
assert not hasattr(T2, "load_answerer"), "retired answerer loader must not exist in the tower module"
assert not hasattr(sys.modules[__name__], "load_answerer")

PROC = os.path.join(_ROOT, "data", "ml-25m", "proc")
GENOME = os.path.join(_ROOT, "data", "movielens", "genome-scores.csv")
SNAP_DEFAULT = os.path.join(_ROOT, ".cache", "instrument", "t2i25_EP4_SNAP.pt")
OUTDIR = os.path.join(_ROOT, "experiments", "battery")
BOOT = 10_000
SEED = 4242


# ============================================================================= shared machinery
def bootstrap_ci(delta, n_boot=BOOT, seed=SEED):
    """Paired per-user bootstrap 95% CI of the mean of `delta` (NaNs dropped)."""
    d = np.asarray(delta, np.float64); d = d[~np.isnan(d)]
    if len(d) == 0:
        return float("nan"), (float("nan"), float("nan")), 0
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(n_boot, len(d)))
    means = d[idx].mean(axis=1)
    return float(d.mean()), (float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))), len(d)


def spearman(a, b):
    from scipy.stats import spearmanr
    return float(spearmanr(a, b).statistic)


def fold_z(enc, tokens, rows, batch=256):
    """z vectors for the given rows' token lists. tokens: list of (sids, lvls). Returns (len(rows), d)."""
    Z = np.zeros((len(rows), enc.d_out), np.float32)
    enc.eval()
    with torch.no_grad():
        for st in range(0, len(rows), batch):
            chunk = rows[st:st + batch]
            packrows = []
            for r in chunk:
                s, lv = tokens[r]
                packrows.append((np.asarray(s, np.int64), np.asarray(lv, np.int64),
                                 level_to_sv(np.asarray(lv))))
            ids, vals, pad, lvs = pack_tokens(packrows)
            Z[st:st + len(chunk)] = enc(ids, vals, pad, lvs).numpy()
    return Z


def ndcg10_from_scores(S, mask_rows, te_rows, head_mask=None):
    """Per-user full (and tail) NDCG@10 for a score batch. mask_rows/te_rows: csr slices aligned to S.
    Returns (full, tail) with NaN where the user has no (tail) target."""
    S = S.copy()
    S[mask_rows.nonzero()] = -np.inf
    n = S.shape[0]
    full = np.full(n, np.nan); tail = np.full(n, np.nan)
    keep = np.asarray(te_rows.getnnz(axis=1)).ravel() > 0
    if keep.any():
        full[keep] = M.NDCG_binary_at_k_batch(S[keep], te_rows[keep], k=10)
    if head_mask is not None:
        tail_row = (~head_mask).astype("float32")[None, :]
        te_t = te_rows.multiply(tail_row).tocsr(); te_t.eliminate_zeros()
        tkeep = np.asarray(te_t.getnnz(axis=1)).ravel() > 0
        if tkeep.any():
            St = S[tkeep].copy(); St[:, head_mask] = -np.inf
            tail[tkeep] = M.NDCG_binary_at_k_batch(St, te_t[tkeep], k=10)
    return full, tail


def eval_tokens(ctx, tokens, mask_csr, rows=None, batch=256, z_shift=None):
    """Fold tokens -> scores -> per-user full/tail NDCG@10. z_shift: optional (n,d) added to z."""
    rows = rows if rows is not None else list(range(ctx.n))
    Wd, bd = ctx.decoder.weight.detach(), ctx.decoder.bias.detach()
    full = np.full(ctx.n, np.nan); tail = np.full(ctx.n, np.nan)
    ctx.enc.eval()
    with torch.no_grad():
        for st in range(0, len(rows), batch):
            chunk = rows[st:st + batch]
            packrows = []
            for r in chunk:
                s, lv = tokens[r]
                packrows.append((np.asarray(s, np.int64), np.asarray(lv, np.int64),
                                 level_to_sv(np.asarray(lv))))
            ids, vals, pad, lvs = pack_tokens(packrows)
            z = ctx.enc(ids, vals, pad, lvs)
            if z_shift is not None:
                z = z + torch.from_numpy(z_shift[chunk])
            S = (z @ Wd.T + bd).numpy().astype(np.float32)
            f, t = ndcg10_from_scores(S, mask_csr[chunk], ctx.va_te[chunk], ctx.head_mask)
            full[chunk] = f; tail[chunk] = t
    return full, tail


def rows_to_csr(tokens, n, ni):
    r, c = [], []
    for i in range(n):
        s = tokens[i][0]
        r.extend([i] * len(s)); c.extend(list(np.asarray(s)))
    return sparse.csr_matrix((np.ones(len(r), np.float32), (r, c)), shape=(n, ni))


# ============================================================================= context builders
class Ctx:
    pass


def build_real_ctx(snapshot):
    ctx = Ctx()
    meta = M.load_meta(PROC); ctx.ni = meta["n_items"]
    train = M.load_train(ctx.ni, PROC)
    cnt = np.asarray(train.sum(axis=0)).ravel().astype(np.float64)
    order = np.argsort(-cnt); cum = np.cumsum(cnt[order]) / cnt.sum()
    hm = np.zeros(ctx.ni, bool); hm[order[:np.searchsorted(cum, 0.33) + 1]] = True
    ctx.head_mask, ctx.cnt = hm, cnt
    ctx.va_tr, ctx.va_te = M.load_val(ctx.ni, PROC)
    ctx.n = ctx.va_tr.shape[0]
    unique_uid, tr_set, vd_set, te_set, ntr, raw, show2id, usid = reproduce_partition()
    ctx.raw, ctx.tr_set, ctx.show2id = raw, tr_set, show2id      # stashed for strategy_ladder entropies
    # model from snapshot (i25 arm A)
    a = argparse.Namespace(arch="i25", teacher="warm_init", t_hidden=600, t_latent=200, token="film",
                           train_decoder=False, sign_prior=True, unfreeze_emb=False, lr=3e-4,
                           warm_lr_scale=0.1, full_kd=False, full_kd_w=0.3)
    enc, decoder, teacher, params, groups = build_model(a, ctx.ni, cnt)
    blob = torch.load(snapshot, map_location="cpu")
    enc.load_state_dict(blob["enc"]); decoder.load_state_dict(blob["decoder"])
    enc.eval(); ctx.enc, ctx.decoder = enc, decoder
    log(f"[ctx] snapshot loaded: {os.path.basename(snapshot)} ep={blob.get('epoch', '?')} "
        f"val_full={blob.get('val_full', float('nan')):.4f}")
    # ALL-BANDS graded reconstruction for val users (raw catalog ratings EXCLUDING held te items)
    import pandas as pd
    n_users_all = len(unique_uid)
    start_vd = n_users_all - 2 * 10000
    val_userIds = unique_uid[start_vd:start_vd + 10000]
    uid2row = {int(u): i for i, u in enumerate(val_userIds)}
    vdf = raw[raw["userId"].isin(set(val_userIds.tolist()))].copy()
    vdf["sid"] = vdf["movieId"].map(show2id); vdf = vdf[vdf["sid"].notna()]
    vdf["sid"] = vdf["sid"].astype(np.int64)
    vdf["lvl"] = np.clip(np.rint(vdf["rating"].values * 2).astype(np.int64) - 1, 0, NLEV - 1)
    te_sets = [set(ctx.va_te[i].indices.tolist()) for i in range(ctx.n)]
    allb = [(np.empty(0, np.int64), np.empty(0, np.int64))] * ctx.n
    for uidv, g in vdf.groupby("userId"):
        r = uid2row[int(uidv)]
        keepm = ~g["sid"].isin(te_sets[r]).values
        allb[r] = (g["sid"].values[keepm].astype(np.int64), g["lvl"].values[keepm].astype(np.int64))
    ctx.allb = allb
    ctx.allb_mask = rows_to_csr(allb, ctx.n, ctx.ni)
    log(f"[ctx] all-bands reconstruction: {int(ctx.allb_mask.nnz)} tokens; "
        f"{sum(1 for s, l in allb if (l <= 4).sum() >= 3)} users with >=3 dislikes")
    # canonical k2 graded subset (tower cold parity) for G5 context
    L_val, _ = build_graded_eval_matrix(raw, unique_uid, show2id, usid, "validation")
    ctx.L_val = L_val                        # stashed for downstream cold-k protocols (concepts_only_curve)
    Lk2 = truncate_graded(L_val, 2, COLD_SEED)
    ctx.k2_tokens = [(Lk2[i].indices.astype(np.int64), (Lk2[i].data - 1).astype(np.int64))
                     for i in range(ctx.n)]
    # G9 banks: top-200-popularity + random-200; per-user TRUE graded answers (te excluded)
    bank_pop = order[:200].astype(np.int64)
    bank_rand = np.random.default_rng(SEED).choice(ctx.ni, size=200, replace=False).astype(np.int64)
    lvl_lookup = [dict(zip(s.tolist(), l.tolist())) for s, l in allb]   # te already excluded
    def bank_answers(bank):
        out = []
        for r in range(ctx.n):
            d = lvl_lookup[r]
            sids = [int(b) for b in bank if int(b) in d]
            out.append((np.array(sids, np.int64), np.array([d[s] for s in sids], np.int64)))
        return out
    ctx.banks = {"pop": (bank_pop, bank_answers(bank_pop)),
                 "rand": (bank_rand, bank_answers(bank_rand))}
    ctx.raw_for_g5 = True
    ctx.outdir = OUTDIR
    return ctx


def build_smoke_ctx():
    """Tiny synthetic context (code-path only, labeled). Fake RecVAE-backed i25 tower."""
    log("[SMOKE] synthetic battery context (NOT the canonical split)")
    ctx = Ctx(); rng = np.random.RandomState(0)
    ctx.ni, ctx.n = 130, 60
    ctx.cnt = rng.rand(ctx.ni) * 100
    hm = np.zeros(ctx.ni, bool); hm[np.argsort(-ctx.cnt)[:20]] = True
    ctx.head_mask = hm
    fake = R.RecVAE(24, 16, ctx.ni)
    a = argparse.Namespace(arch="i25", teacher="warm_init", t_hidden=24, t_latent=16, token="film",
                           train_decoder=False, sign_prior=True, unfreeze_emb=False, lr=3e-4,
                           warm_lr_scale=0.1, full_kd=False, full_kd_w=0.3)
    enc, decoder, teacher, params, groups = build_model(a, ctx.ni, ctx.cnt, teacher_override=fake)
    with torch.no_grad():                                    # non-zero rho so folds actually move z
        for p in enc.rho[-1].parameters():
            p.add_(torch.randn_like(p) * 0.05)
    ctx.enc, ctx.decoder = enc.eval(), decoder
    allb, te_r, te_c = [], [], []
    for i in range(ctx.n):
        k = rng.randint(12, 30)
        items = rng.choice(ctx.ni, size=k, replace=False)
        lv = rng.randint(0, NLEV, size=k)
        lv[:4] = [0, 1, 8, 9]                                # guarantee dislikes + likes
        allb.append((items[:k - 3].astype(np.int64), lv[:k - 3].astype(np.int64)))
        te_r.extend([i] * 3); te_c.extend(items[k - 3:].tolist())
    ctx.allb = allb
    ctx.allb_mask = rows_to_csr(allb, ctx.n, ctx.ni)
    ctx.va_te = sparse.csr_matrix((np.ones(len(te_r), np.float32), (te_r, te_c)),
                                  shape=(ctx.n, ctx.ni))
    ctx.va_tr = ctx.allb_mask.copy()
    ctx.k2_tokens = [(s[:2], l[:2]) for s, l in allb]
    bank_pop = np.argsort(-ctx.cnt)[:30].astype(np.int64)
    bank_rand = rng.choice(ctx.ni, 30, replace=False).astype(np.int64)
    lookup = [dict(zip(s.tolist(), l.tolist())) for s, l in allb]
    def bank_answers(bank):
        return [(np.array([b for b in bank if b in lookup[r]], np.int64),
                 np.array([lookup[r][b] for b in bank if b in lookup[r]], np.int64))
                for r in range(ctx.n)]
    ctx.banks = {"pop": (bank_pop, bank_answers(bank_pop)),
                 "rand": (bank_rand, bank_answers(bank_rand))}
    ctx.raw_for_g5 = False
    ctx.outdir = os.path.join(OUTDIR, "_smoke")
    return ctx


def save_gate(ctx, name, payload):
    os.makedirs(ctx.outdir, exist_ok=True)
    p = os.path.join(ctx.outdir, f"{name}.json")
    json.dump(payload, open(p, "w"), indent=2)
    log(f"[{name}] -> {p}")
    return p


# ============================================================================= G3a FLIP + staircase
def gate_g3a(ctx):
    t0 = time.time()
    flip = [(s, np.clip(10 - l, 0, NLEV - 1)) for s, l in ctx.allb]      # mirror around 3 stars
    elig = [r for r in range(ctx.n)
            if (ctx.allb[r][1] <= 4).sum() >= 3 and ctx.va_te[r].nnz > 0]
    f_true, t_true = eval_tokens(ctx, ctx.allb, ctx.allb_mask, rows=elig)
    f_flip, t_flip = eval_tokens(ctx, flip, ctx.allb_mask, rows=elig)
    d = f_flip[elig] - f_true[elig]
    mean, ci, nn_ = bootstrap_ci(d)
    verdict = bool(mean < 0 and ci[1] < 0)
    # ---- intensity staircase (FIG-1): one held liked item folded at every level on a k=4 context ----
    rng = np.random.default_rng(SEED)
    st_rows, ctx4, tgt = [], {}, {}
    for r in range(ctx.n):
        s, l = ctx.allb[r]
        te = ctx.va_te[r].indices
        if len(s) >= 4 and len(te) >= 2:
            pick = rng.choice(len(s), size=4, replace=False)
            ctx4[r] = (s[pick], l[pick])
            tgt[r] = int(rng.choice(te))
            st_rows.append(r)
    def stair_tokens(level):
        toks = []
        for r in range(ctx.n):
            if r in ctx4:
                s4, l4 = ctx4[r]
                if level is None:
                    toks.append((s4, l4))
                else:
                    toks.append((np.append(s4, tgt[r]), np.append(l4, level)))
            else:
                toks.append((np.empty(0, np.int64), np.empty(0, np.int64)))
        return toks
    def stair_eval(level):
        toks = stair_tokens(level)
        mask = rows_to_csr([(np.append(ctx4[r][0], tgt[r]) if (level is not None and r in ctx4)
                             else (ctx4[r][0] if r in ctx4 else np.empty(0, np.int64)),
                             None) for r in range(ctx.n)], ctx.n, ctx.ni)
        # NDCG on remaining targets, target item masked when folded (credit-neutral)
        te_rest = ctx.va_te.copy().tolil()
        for r in st_rows:
            te_rest[r, tgt[r]] = 0
        te_rest = te_rest.tocsr(); te_rest.eliminate_zeros()
        # rank of target: context-only mask (target NEVER masked for the rank readout)
        Wd, bd = ctx.decoder.weight.detach(), ctx.decoder.bias.detach()
        fulls = np.full(ctx.n, np.nan); ranks = np.full(ctx.n, np.nan)
        with torch.no_grad():
            for st in range(0, len(st_rows), 256):
                chunk = st_rows[st:st + 256]
                packrows = [(np.asarray(toks[r][0], np.int64), np.asarray(toks[r][1], np.int64),
                             level_to_sv(np.asarray(toks[r][1]))) for r in chunk]
                ids, vals, pad, lvs = pack_tokens(packrows)
                S = (ctx.enc(ids, vals, pad, lvs) @ Wd.T + bd).numpy().astype(np.float32)
                Sr = S.copy()
                for j, r in enumerate(chunk):                # rank readout: mask context only
                    Sr[j, ctx4[r][0]] = -np.inf
                    ranks[r] = float((Sr[j] > Sr[j, tgt[r]]).sum() + 1)
                f, _ = ndcg10_from_scores(S, mask[chunk], te_rest[chunk], None)
                fulls[chunk] = f
        return fulls, ranks
    base_f, base_rank = stair_eval(None)
    stair = {}; shifts = {}                                  # per-level per-user rank shifts (for CI)
    for lvl in range(NLEV):
        f_l, r_l = stair_eval(lvl)
        shifts[lvl] = (base_rank - r_l)[st_rows]
        stair[lvl] = {"mean_dNDCG": float(np.nanmean(f_l[st_rows] - base_f[st_rows])),
                      "mean_rank_shift": float(np.nanmean(shifts[lvl]))}
    lvls = list(range(NLEV))
    rho_rank = spearman(lvls, [stair[l]["mean_rank_shift"] for l in lvls])
    rho_ndcg = spearman(lvls, [stair[l]["mean_dNDCG"] for l in lvls])
    # bootstrap CI on the rank-shift Spearman: resample USERS, recompute the mean curve, spearman
    rngb = np.random.default_rng(SEED + 1)
    Smat = np.stack([shifts[l] for l in lvls])               # (10, n_st)
    rhos = []
    for _ in range(1000):
        pick = rngb.integers(0, Smat.shape[1], size=Smat.shape[1])
        rhos.append(spearman(lvls, np.nanmean(Smat[:, pick], axis=1)))
    rho_ci = (float(np.quantile(rhos, 0.025)), float(np.quantile(rhos, 0.975)))
    payload = {"gate": "G3a", "flip": {"n_eligible": nn_, "mean_delta_full@10": mean,
               "ci95": ci, "true_full@10": float(np.nanmean(f_true[elig])),
               "flipped_full@10": float(np.nanmean(f_flip[elig])), "PASS": verdict},
               "staircase": {"n_users": len(st_rows), "per_level": stair,
                             "spearman_rank_shift": rho_rank, "spearman_rank_shift_ci95": rho_ci,
                             "spearman_dNDCG": rho_ndcg,
                             "note": "FIG-1 data; rank readout unmasked target, NDCG credit-neutral"},
               "seconds": round(time.time() - t0, 1)}
    payload["PASS"] = bool(verdict and rho_rank > 0 and rho_ci[0] > 0)
    save_gate(ctx, "g3a", payload)
    return payload


# ============================================================================= G3b VALUE-NONINERT
def gate_g3b(ctx, mde=0.005):
    t0 = time.time()
    neutral = [(s, np.full(len(l), 5, np.int64)) for s, l in ctx.allb]     # 3 stars everywhere
    rows = [r for r in range(ctx.n) if ctx.va_te[r].nnz > 0]
    f_true, t_true = eval_tokens(ctx, ctx.allb, ctx.allb_mask, rows=rows)
    f_neu, t_neu = eval_tokens(ctx, neutral, ctx.allb_mask, rows=rows)
    d = f_true[rows] - f_neu[rows]
    mean, ci, nn_ = bootstrap_ci(d)
    payload = {"gate": "G3b", "n": nn_, "true_full@10": float(np.nanmean(f_true[rows])),
               "neutral3star_full@10": float(np.nanmean(f_neu[rows])),
               "drop_mean": mean, "ci95": ci, "mde": mde,
               "true_tail@10": float(np.nanmean(t_true[rows])),
               "neutral_tail@10": float(np.nanmean(t_neu[rows])),
               "PASS": bool(mean >= mde and ci[0] > 0),
               "seconds": round(time.time() - t0, 1)}
    save_gate(ctx, "g3b", payload)
    return payload


# ============================================================================= G6 EXISTENTIAL NULLS
def gate_g6(ctx):
    t0 = time.time()
    rows = [r for r in range(ctx.n) if ctx.va_te[r].nnz > 0]
    f_true, _ = eval_tokens(ctx, ctx.allb, ctx.allb_mask, rows=rows)
    # (i) wrong-user derangement: fold user (r-1)'s tokens for row r; own mask/targets
    wrong = [ctx.allb[(r - 1) % ctx.n] for r in range(ctx.n)]
    f_wrong, _ = eval_tokens(ctx, wrong, ctx.allb_mask, rows=rows)
    # intercept: empty fold
    empty = [(np.empty(0, np.int64), np.empty(0, np.int64))] * ctx.n
    f_int, _ = eval_tokens(ctx, empty, ctx.allb_mask, rows=rows)
    # (ii) shuffled levels within user
    rng = np.random.default_rng(SEED)
    shuf = [(s, l[rng.permutation(len(l))] if len(l) else l) for s, l in ctx.allb]
    f_shuf, _ = eval_tokens(ctx, shuf, ctx.allb_mask, rows=rows)
    # (iii) placebo constant 4 stars (level 7)
    plac = [(s, np.full(len(l), 7, np.int64)) for s, l in ctx.allb]
    f_plac, _ = eval_tokens(ctx, plac, ctx.allb_mask, rows=rows)
    m_true = float(np.nanmean(f_true[rows])); m_wrong = float(np.nanmean(f_wrong[rows]))
    m_int = float(np.nanmean(f_int[rows]))
    d_shuf = bootstrap_ci(f_true[rows] - f_shuf[rows])
    d_plac = bootstrap_ci(f_true[rows] - f_plac[rows])
    d_wrong_int = bootstrap_ci(f_wrong[rows] - f_int[rows])
    # (iv) duplicate: assert-only (fold-boundary dedup)
    try:
        pack_tokens([(np.array([1, 1]), np.array([8, 8]), level_to_sv(np.array([8, 8])))])
        dup_fired = False
    except AssertionError:
        dup_fired = True
    payload = {"gate": "G6",
               "true_full@10": m_true, "intercept_full@10": m_int,
               "wrong_user": {"full@10": m_wrong, "delta_vs_intercept": d_wrong_int[0],
                              "ci95": d_wrong_int[1],
                              "PASS_near_intercept": bool(abs(m_wrong - m_int)
                                                          < 0.25 * max(m_true - m_int, 1e-9))},
               "shuffled_levels": {"full@10": float(np.nanmean(f_shuf[rows])),
                                   "true_minus_shuffled": d_shuf[0], "ci95": d_shuf[1],
                                   "PASS": bool(d_shuf[0] > 0 and d_shuf[1][0] > 0)},
               "placebo_4star": {"full@10": float(np.nanmean(f_plac[rows])),
                                 "true_minus_placebo": d_plac[0], "ci95": d_plac[1],
                                 "PASS": bool(d_plac[0] > 0 and d_plac[1][0] > 0)},
               "duplicate_assert_fired": dup_fired,
               "seconds": round(time.time() - t0, 1)}
    payload["PASS"] = bool(payload["wrong_user"]["PASS_near_intercept"]
                           and payload["shuffled_levels"]["PASS"]
                           and payload["placebo_4star"]["PASS"] and dup_fired)
    save_gate(ctx, "g6", payload)
    return payload


# ============================================================================= G9 INGESTION-ISOLATION
def gate_g9(ctx, budgets=(2, 4, 8, 16)):
    t0 = time.time()
    rows = [r for r in range(ctx.n) if ctx.va_te[r].nnz > 0]
    out = {"gate": "G9", "budgets": list(budgets), "banks": {}}
    per_bank_full = {}
    for bname, (bank, answers) in ctx.banks.items():
        cur = {"answer_rate": {}, "full@10": {}, "tail@10": {}}
        fq = {}
        for q in budgets:
            toks = []
            for r in range(ctx.n):
                s, l = answers[r]
                toks.append((s[:q], l[:q]))
            mask = ctx.va_tr + rows_to_csr(toks, ctx.n, ctx.ni)      # tower parity + credit-neutral
            mask.data[:] = 1.0
            f, tl = eval_tokens(ctx, toks, mask.tocsr(), rows=rows)
            fq[q] = f
            cur["full@10"][q] = float(np.nanmean(f[rows]))
            cur["tail@10"][q] = float(np.nanmean(tl[rows]))
            avail = np.array([min(q, len(answers[r][0])) for r in rows], float)
            cur["answer_rate"][q] = {"mean_answered": float(avail.mean()),
                                     "frac_full_budget": float((avail >= q).mean())}
        rise = bootstrap_ci(fq[budgets[-1]][rows] - fq[budgets[0]][rows])
        cur["rise_last_minus_first"] = {"mean": rise[0], "ci95": rise[1]}
        cur["monotone_spearman"] = spearman(list(budgets),
                                            [cur["full@10"][q] for q in budgets])
        out["banks"][bname] = cur
        per_bank_full[bname] = fq
    # pop bank must beat random bank at each q (paired)
    vs = {}
    for q in budgets:
        d = bootstrap_ci(per_bank_full["pop"][q][rows] - per_bank_full["rand"][q][rows])
        vs[q] = {"pop_minus_rand": d[0], "ci95": d[1], "PASS": bool(d[0] > 0 and d[1][0] > 0)}
    out["pop_vs_rand"] = vs
    pop = out["banks"]["pop"]
    out["PASS"] = bool(pop["rise_last_minus_first"]["mean"] > 0
                       and pop["rise_last_minus_first"]["ci95"][0] > 0
                       and pop["monotone_spearman"] > 0
                       and all(v["PASS"] for v in vs.values()))
    out["seconds"] = round(time.time() - t0, 1)
    save_gate(ctx, "g9", out)
    return out


# ============================================================================= G5 CONCEPT ARM-A
def load_genome(ctx, min_members=30):
    """tag -> member sids (relevance >= 0.5, vocab-restricted)."""
    import pandas as pd
    if not ctx.raw_for_g5:                                   # smoke: random membership
        rng = np.random.RandomState(3)
        return {c: np.sort(rng.choice(ctx.ni, size=rng.randint(min_members, 60), replace=False))
                for c in range(8)}
    usid = [int(x) for x in open(os.path.join(PROC, "unique_sid.txt")).read().split()]
    m2s = {m: i for i, m in enumerate(usid)}
    log("[g5] loading genome-scores.csv (relevance >= 0.5, vocab-restricted)")
    g = pd.read_csv(GENOME)
    g = g[g["relevance"] >= 0.5]
    g = g[g["movieId"].isin(m2s)]
    g["sid"] = g["movieId"].map(m2s).astype(np.int64)
    members = {int(t): np.sort(sub["sid"].values) for t, sub in g.groupby("tagId")}
    members = {t: m for t, m in members.items() if len(m) >= min_members}
    log(f"[g5] {len(members)} concepts with >= {min_members} members")
    return members


def gate_g5(ctx, betas=(2, 5, 10, 20, 50)):
    t0 = time.time()
    members = load_genome(ctx)
    tags = sorted(members.keys())
    Wd = ctx.decoder.weight.detach().numpy().astype(np.float64)          # (ni, d)
    # July whitening recipe: center + strip top-1 PC of the item embeddings (documented)
    Ec = Wd - Wd.mean(0, keepdims=True)
    cov = Ec.T @ Ec
    w, V = np.linalg.eigh(cov); u1 = V[:, -1]
    Ew = Ec - np.outer(Ec @ u1, u1)
    d_c = {}; w_c = {}
    for t in tags:
        v = Ew[members[t]].mean(0)
        d_c[t] = (v / max(np.linalg.norm(v), 1e-12)).astype(np.float32)
        w_c[t] = 1.0 / np.log1p(len(members[t]))
    # SEL top behavioral concept per user (fold-in likes; te-half EXCLUDED by construction -- C2 clause)
    Mm = sparse.csr_matrix((np.ones(sum(len(members[t]) for t in tags), np.float32),
                            (np.concatenate([members[t] for t in tags]),
                             np.concatenate([np.full(len(members[t]), i) for i, t in enumerate(tags)]))),
                           shape=(ctx.ni, len(tags)))
    counts = np.asarray((ctx.va_tr @ Mm).todense())                       # (n, C)
    nu = np.asarray(ctx.va_tr.sum(axis=1)).ravel().clip(min=1)
    gmass = ctx.cnt @ np.asarray(Mm.todense())                            # member train mass per concept
    grate = gmass / max(ctx.cnt.sum(), 1e-9)
    lift = (counts / nu[:, None]) / np.maximum(grate[None, :], 1e-12)
    lift[counts < 2] = -np.inf                                            # need >=2 supporting items
    top_idx = lift.argmax(1)
    has_c = np.isfinite(lift.max(1))
    user_tag = {r: tags[int(top_idx[r])] for r in range(ctx.n) if has_c[r]}
    rows = [r for r in range(ctx.n) if ctx.va_te[r].nnz > 0 and r in user_tag]
    log(f"[g5] {len(rows)} users with a SEL top concept (>=2 supporting fold-in items)")
    # popularity-matched non-members per concept (nearest train count, unique, once)
    used_tags = sorted(set(user_tag[r] for r in rows))
    match = {}
    order_cnt = np.argsort(ctx.cnt)
    for t in used_tags:
        mem = members[t]; non = np.setdiff1d(np.arange(ctx.ni), mem, assume_unique=False)
        non_sorted = non[np.argsort(ctx.cnt[non])]
        pos = np.searchsorted(ctx.cnt[non_sorted], ctx.cnt[mem])
        match[t] = non_sorted[np.clip(pos, 0, len(non_sorted) - 1)]
    bd = ctx.decoder.bias.detach().numpy()
    WdT = ctx.decoder.weight.detach().numpy().T.astype(np.float32)        # (d, ni)
    empty = [(np.empty(0, np.int64), np.empty(0, np.int64))] * ctx.n
    mask_parity = ctx.va_tr
    f_int, t_int = eval_tokens(ctx, empty, mask_parity, rows=rows)
    Zk2 = fold_z(ctx.enc, ctx.k2_tokens, list(range(ctx.n)))
    f_k2, t_k2 = eval_tokens(ctx, ctx.k2_tokens, mask_parity, rows=rows)
    def shift_mat(beta):
        Zs = np.zeros((ctx.n, ctx.decoder.weight.shape[1]), np.float32)
        for r in rows:
            t = user_tag[r]
            Zs[r] = beta * w_c[t] * 1.0 * d_c[t]                          # v=+1 (loved)
        return Zs
    def auc_members(Smat_rows, rows_sel):
        aucs = []
        for j, r in enumerate(rows_sel):
            t = user_tag[r]; sm = Smat_rows[j][members[t]]; sn = Smat_rows[j][match[t]]
            k = len(sm)
            ranks = np.argsort(np.argsort(np.concatenate([sm, sn])))[:k].sum()
            aucs.append((ranks - k * (k - 1) / 2) / (k * k))
        return float(np.mean(aucs))
    armA = {}
    for beta in betas:
        Zs = shift_mat(beta)
        # k0: z = shift only -> scores directly (no encoder pass)
        S0 = Zs @ WdT + bd[None, :]
        f0 = np.full(ctx.n, np.nan)
        for st in range(0, len(rows), 500):
            chunk = rows[st:st + 500]
            f, _ = ndcg10_from_scores(S0[chunk], mask_parity[chunk], ctx.va_te[chunk], None)
            f0[chunk] = f
        auc0 = auc_members(S0[rows], rows)
        f2, t2 = eval_tokens(ctx, ctx.k2_tokens, mask_parity, rows=rows, z_shift=Zs)
        d0 = bootstrap_ci(f0[rows] - f_int[rows])
        d2 = bootstrap_ci(f2[rows] - f_k2[rows])
        armA[beta] = {"k0_full@10": float(np.nanmean(f0[rows])), "k0_vs_intercept": d0[0],
                      "k0_ci95": d0[1], "k0_member_AUC_popmatched": auc0,
                      "k2_full@10": float(np.nanmean(f2[rows])), "k2_vs_k2only": d2[0],
                      "k2_ci95": d2[1], "k2_tail@10": float(np.nanmean(t2[rows]))}
    best_beta = max(armA, key=lambda b: armA[b]["k0_full@10"])
    # Arm B comparator: member-bag token fold (level 8 tokens), k0 and k2
    bag0, bag2 = [], []
    for r in range(ctx.n):
        if r in user_tag:
            mem = members[user_tag[r]]
            s2, l2 = ctx.k2_tokens[r]
            mem0 = np.setdiff1d(mem, s2)
            bag0.append((mem, np.full(len(mem), 8, np.int64)))
            bag2.append((np.concatenate([s2, mem0]),
                         np.concatenate([l2, np.full(len(mem0), 8, np.int64)])))
        else:
            bag0.append((np.empty(0, np.int64), np.empty(0, np.int64)))
            bag2.append(ctx.k2_tokens[r])
    fb0, _ = eval_tokens(ctx, bag0, mask_parity, rows=rows, batch=32)
    fb2, _ = eval_tokens(ctx, bag2, mask_parity, rows=rows, batch=32)
    db0 = bootstrap_ci(fb0[rows] - f_int[rows])
    payload = {"gate": "G5-armA-firstpass", "n_users": len(rows), "n_concepts": len(used_tags),
               "whitening": "center + strip top-1 PC of decoder rows (July recipe); IDF=1/log1p(|m|)",
               "armA_by_beta": armA, "best_beta": best_beta,
               "G_collinearity_PASS": bool(armA[best_beta]["k0_full@10"]
                                           > float(np.nanmean(f_int[rows]))),
               "intercept_full@10": float(np.nanmean(f_int[rows])),
               "k2only_full@10": float(np.nanmean(f_k2[rows])),
               "armB_memberbag": {"k0_full@10": float(np.nanmean(fb0[rows])),
                                  "k0_vs_intercept": db0[0], "ci95": db0[1],
                                  "k2_full@10": float(np.nanmean(fb2[rows]))},
               "armA_beats_armB_k0": bool(armA[best_beta]["k0_full@10"]
                                          > float(np.nanmean(fb0[rows]))),
               "seconds": round(time.time() - t0, 1)}
    save_gate(ctx, "g5", payload)
    return payload


# ============================================================================= main
GATES = {"g3a": gate_g3a, "g3b": gate_g3b, "g6": gate_g6, "g9": gate_g9, "g5": gate_g5}
ORDER = ["g3a", "g3b", "g6", "g9", "g5"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--only", default=None)
    ap.add_argument("--snapshot", default=SNAP_DEFAULT)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    ctx = build_smoke_ctx() if args.smoke else build_real_ctx(args.snapshot)
    todo = ORDER if not args.only else [g for g in ORDER if g in set(args.only.split(","))]
    results = {}
    for g in todo:
        outp = os.path.join(ctx.outdir, f"{g}.json")
        if os.path.exists(outp) and not args.force and not args.smoke:
            log(f"[skip] {g}: exists"); results[g] = json.load(open(outp)); continue
        log(f"=== {g.upper()} ===")
        results[g] = GATES[g](ctx)
    print("\n=== PHASE A VERDICTS ===")
    for g in todo:
        p = results[g].get("PASS", results[g].get("G_collinearity_PASS", "n/a"))
        print(f"  {g}: PASS={p}")


if __name__ == "__main__":
    main()
