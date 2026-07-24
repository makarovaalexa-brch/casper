r"""strategy_ladder.py -- LIT-ANCHORED STRATEGY LADDER (author requirement 2026-07-24): the G2 harness
extended to the classic new-user item-elicitation selectors, each annotated with its literature-expected
rank and auto-flagged on any CI-clean inversion of the published ordering.

ARMS (static ask-orders; an asked-but-unrated item BURNS the question -- the lit protocol, and the whole
point of random-catalog's low answer rate):
  random_catalog  uniform over the FULL 18,359 vocab, per-user permutation  [lit rank 6 -- worst: tiny
                  answer rate, near-zero info per budget]
  random_bank     the existing G9 random-200 bank, per-user order            [lit rank 5 -- random within
                  a fixed shortlist; still popularity-agnostic]
  pure_entropy    rating entropy over the 10 half-star bands from the GRADED train reconstruction
                  (raters only)                                              [lit rank 4 -- Rashid 2002:
                  "Pure Entropy was the worst... shockingly poor" among informed strategies: high-entropy
                  items are obscure, users can't rate them]
  popularity      descending train count                                     [lit rank 3 -- strong simple
                  baseline; answerable but redundant/low-discrimination]
  entropy0        entropy with 'not rated' as an EXTRA (11th) category       [lit rank 2 -- Rashid 2008's
                  fix: folds answerability into the entropy itself]
  helf            harmonic mean of normalized rating-entropy (graded, raters-only) and normalized
                  log-frequency                                              [lit rank 1 -- Rashid 2002/
                  2008's winner: informative AND answerable]
CITATIONS (keys verified in external_literature/INDEX.md): rashid2002getting, rashid2008learning,
golbandi2011adaptive (adaptive upper reference, not an arm here), elahi2016survey (survey anchor).
golbandi2010 (RecSys'10 static bootstrapping) is referenced in the annotations -- FLAG FOR BIB ADD if
absent from the bib.

PROTOCOL: VAL cohort, budgets q in {0,1,2,4,8,16} (headline q8), answers = REAL graded ratings from the
all-bands reconstruction (held te excluded), CREDIT-NEUTRAL masking of every ASKED item (answered or not:
it was shown) -- masked from candidates AND dropped from the IDCG denominator; candidate mask also unions
the canonical va_tr fold-in (tower cold parity, the G9 convention). Paired per-user bootstrap CIs.

AUTO REPLICATION FLAGS: for every arm pair (i,j) with lit_rank_i < lit_rank_j (i expected better), the
measured q8 full@10 paired delta d = arm_i - arm_j:
  d > 0, CI excl. 0   -> lit order REPLICATED
  CI spans 0          -> 'indistinguishable, no flag'
  d < 0, CI excl. 0   -> INVERSION FLAG (measured CI-clean in the wrong direction)

KNOWN PROTOCOL CAVEATS (documented in the JSON, per the author):
  * our answers fold GRADED values into a set-encoder posterior scored by NDCG@10; the lit mostly
    measured rating-prediction MAE (Rashid) or RMSE (Golbandi) -- expected ranks transfer as priors,
    not as certified numbers;
  * our random_bank (fixed random-200 shortlist) has no exact lit counterpart -- lit's 'random' is our
    random_catalog;
  * entropy from a catalog-restricted (>=20 ratings) matrix compresses the entropy range vs the lit's
    unrestricted catalogs.

Usage:
  python src/instrument/strategy_ladder.py --smoke
  python src/instrument/strategy_ladder.py [--full_threads] [--snapshot PATH]
Do NOT run against a live training run without OMP=4/Idle (default); --full_threads when the CPU is ours.
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

from train_tower_t2 import level_to_sv, pack_tokens, NLEV, log
import run_battery_phaseA as PA
from run_battery_phaseA import (build_real_ctx, build_smoke_ctx, ndcg10_from_scores, bootstrap_ci,
                                spearman, OUTDIR, SEED)

assert not hasattr(sys.modules[__name__], "load_answerer")           # retired-answerer ban

BUDGETS = (0, 1, 2, 4, 8, 16)
HEADLINE_Q = 8

ARMS = {  # name -> (lit_expected_rank at ~q8 [1=best], citation, annotation)
    "helf": (1, "rashid2002getting; rashid2008learning",
             "HELF = harmonic(entropy, log-freq): informative AND answerable -- Rashid's winner"),
    "entropy0": (2, "rashid2008learning",
                 "entropy incl. 'not rated' as a category -- folds answerability into the score"),
    "popularity": (3, "rashid2002getting; golbandi2010 [FLAG FOR BIB ADD if absent]",
                   "strong simple baseline: answerable but redundant"),
    "pure_entropy": (4, "rashid2002getting",
                     "'Pure Entropy was the worst... shockingly poor' -- obscure high-entropy items"),
    "random_bank": (5, "no exact lit counterpart (protocol caveat)",
                    "random within a fixed 200-item shortlist"),
    "random_catalog": (6, "rashid2002getting; elahi2016survey",
                       "uniform over the full vocab: tiny answer rate, near-zero info"),
}


# ============================================================================= entropies (graded train)
def train_entropies(ctx):
    """Per-item level-band counts over TRAIN-partition users from the raw graded ratings (all bands,
    catalog-restricted). Returns (H raters-only, H0 with 'not rated' as an 11th category, n_raters).
    Smoke ctx (no raw): synthetic counts."""
    if hasattr(ctx, "raw"):
        df = ctx.raw[ctx.raw["userId"].isin(ctx.tr_set)]
        sid = df["movieId"].map(ctx.show2id)
        ok = sid.notna()
        sid = sid[ok].astype(np.int64).values
        lvl = np.clip(np.rint(df.loc[ok, "rating"].values * 2).astype(np.int64) - 1, 0, NLEV - 1)
        C = np.zeros((ctx.ni, NLEV), np.float64)
        np.add.at(C, (sid, lvl), 1.0)
        n_train_users = float(len(ctx.tr_set))
    else:                                                             # smoke: synthetic
        rng = np.random.RandomState(5)
        C = rng.poisson(lam=rng.rand(ctx.ni, 1) * 20, size=(ctx.ni, NLEV)).astype(np.float64)
        n_train_users = float(C.sum(1).max() * 2 + 10)
    n_raters = C.sum(1)
    with np.errstate(divide="ignore", invalid="ignore"):
        p = C / np.maximum(n_raters[:, None], 1e-12)
        H = -np.nansum(np.where(p > 0, p * np.log2(p), 0.0), axis=1)      # raters-only entropy
        C0 = np.concatenate([C, np.maximum(n_train_users - n_raters, 0.0)[:, None]], axis=1)
        p0 = C0 / max(n_train_users, 1e-12)
        H0 = -np.nansum(np.where(p0 > 0, p0 * np.log2(p0), 0.0), axis=1)  # entropy0 (Rashid 2008)
    H[n_raters == 0] = 0.0
    return H, H0, n_raters


def arm_orders(ctx, H, H0):
    """Static ask-order per arm. random arms are PER-USER (seeded); score arms are global orders."""
    Hn = H / max(H.max(), 1e-9)
    Fn = np.log1p(ctx.cnt) / max(np.log1p(ctx.cnt).max(), 1e-9)
    helf = 2.0 * Hn * Fn / np.clip(Hn + Fn, 1e-9, None)
    glob = {"popularity": np.argsort(-ctx.cnt),
            "pure_entropy": np.argsort(-H),
            "entropy0": np.argsort(-H0),
            "helf": np.argsort(-helf)}
    bank_rand = ctx.banks["rand"][0]
    return glob, bank_rand, helf


# ============================================================================= simulation + eval
def simulate_arm(ctx, name, glob, bank_rand, rows, lvl_lookup, q_max):
    """Walk the arm's ask-order; each asked item burns a question (answered only if rated). Returns
    per-user (asked_list, answered_tokens list [(sid, lvl)])."""
    asked = {}; toks = {}
    for r in rows:
        rng = np.random.default_rng(SEED * 1000 + r)
        if name == "random_catalog":
            # non-lossy: sample WITHOUT materialising an 18k permutation -- draw q_max distinct sids
            order = rng.choice(ctx.ni, size=min(q_max * 4, ctx.ni), replace=False)
        elif name == "random_bank":
            order = rng.permutation(bank_rand)
        else:
            order = glob[name]
        d = lvl_lookup[r]
        a = []; t = []
        for i in order:
            if len(a) >= q_max:
                break
            i = int(i)
            a.append(i)
            if i in d:
                t.append((i, d[i]))
        asked[r] = a; toks[r] = t
    return asked, toks


def eval_budget(ctx, asked, toks, rows, q, batch=256):
    """Fold each user's answered tokens within the first q ASKED; credit-neutral mask over asked[:q]
    (mask from candidates AND drop from IDCG), unioned with the canonical va_tr fold-in (G9 parity)."""
    Wd = ctx.decoder.weight.detach(); bd = ctx.decoder.bias.detach()
    full = np.full(ctx.n, np.nan); tail = np.full(ctx.n, np.nan)
    n_ans = np.zeros(ctx.n)
    ctx.enc.eval()
    with torch.no_grad():
        for st in range(0, len(rows), batch):
            chunk = rows[st:st + batch]
            packrows = []; mrows = []; mcols = []; te_drop = []
            for j, r in enumerate(chunk):
                aq = set(asked[r][:q])
                tq = [(s, l) for s, l in toks[r] if s in aq]
                n_ans[r] = len(tq)
                s = np.array([x[0] for x in tq], np.int64); l = np.array([x[1] for x in tq], np.int64)
                packrows.append((s, l, level_to_sv(l)))
                for i in aq:
                    mrows.append(j); mcols.append(i)
                te_drop.append(aq)
            ids, vals, pad, lvs = pack_tokens(packrows)
            z = ctx.enc(ids, vals, pad, lvs)
            S = (z @ Wd.T + bd).numpy().astype(np.float32)
            extra = sparse.csr_matrix((np.ones(len(mrows), np.float32), (mrows, mcols)),
                                      shape=(len(chunk), ctx.ni))
            mask = ctx.va_tr[chunk] + extra; mask.data[:] = 1.0
            te = ctx.va_te[chunk].tolil()
            for j, aq in enumerate(te_drop):
                for i in aq:
                    te[j, i] = 0
            te = te.tocsr(); te.eliminate_zeros()
            f, t = ndcg10_from_scores(S, mask.tocsr(), te, ctx.head_mask)
            full[np.asarray(chunk)] = f; tail[np.asarray(chunk)] = t
    return full, tail, n_ans


# ============================================================================= main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--snapshot", default=PA.SNAP_DEFAULT)
    ap.add_argument("--full_threads", action="store_true")
    args = ap.parse_args()
    if not args.full_threads:
        try:
            import ctypes
            ctypes.windll.kernel32.SetPriorityClass(
                ctypes.windll.kernel32.GetCurrentProcess(), 0x00000040)
            log("[prio] IDLE priority (default; --full_threads to lift)")
        except Exception:
            pass
    else:
        log(f"[prio] FULL THREADS ({_NT}) normal priority")
    t00 = time.time()
    ctx = build_smoke_ctx() if args.smoke else build_real_ctx(args.snapshot)
    H, H0, n_raters = train_entropies(ctx)
    glob, bank_rand, helf = arm_orders(ctx, H, H0)
    log(f"[entropy] H raters-only mean {H.mean():.3f}; H0 mean {H0.mean():.3f}; "
        f"corr(H, log cnt) = {np.corrcoef(H, np.log1p(ctx.cnt))[0,1]:.3f}")
    rows = [r for r in range(ctx.n) if ctx.va_te[r].nnz > 0]
    lvl_lookup = [dict(zip(s.tolist(), l.tolist())) for s, l in ctx.allb]     # te excluded upstream
    q_max = max(BUDGETS)

    out = {"analysis": "strategy_ladder", "snapshot": os.path.basename(args.snapshot),
           "n_users": len(rows), "budgets": list(BUDGETS), "headline_q": HEADLINE_Q,
           "caveats": [
               "answers fold GRADED values into a set-encoder posterior scored NDCG@10; the lit mostly "
               "measured rating-prediction MAE/RMSE -- expected ranks are priors, not certified numbers",
               "random_bank (fixed random-200 shortlist) has no exact lit counterpart; lit's 'random' "
               "is our random_catalog",
               "entropy from a catalog-restricted (>=20 ratings) matrix compresses the entropy range",
               "asked-but-unrated burns the question in ALL arms (the lit protocol)"],
           "arms": {}}

    q8_full = {}
    # intercept (q=0) shared
    for name in ARMS:
        t0 = time.time()
        asked, toks = simulate_arm(ctx, name, glob, bank_rand, rows, lvl_lookup, q_max)
        cur = {"lit_expected_rank": ARMS[name][0], "citation": ARMS[name][1],
               "annotation": ARMS[name][2], "full@10": {}, "tail@10": {}, "answer_rate": {}}
        for q in BUDGETS:
            f, t, n_ans = eval_budget(ctx, asked, toks, rows, q)
            cur["full@10"][str(q)] = float(np.nanmean(f[rows]))
            cur["tail@10"][str(q)] = float(np.nanmean(t[rows]))
            cur["answer_rate"][str(q)] = {"mean_answered": float(n_ans[rows].mean()),
                                          "frac_any": float((n_ans[rows] > 0).mean())}
            if q == HEADLINE_Q:
                q8_full[name] = f
        seq = [cur["full@10"][str(q)] for q in BUDGETS]
        cur["monotone_full"] = bool(all(seq[i + 1] >= seq[i] - 1e-9 for i in range(len(seq) - 1)))
        out["arms"][name] = cur
        log(f"[{name}] q8 full {cur['full@10'][str(HEADLINE_Q)]:.4f} tail "
            f"{cur['tail@10'][str(HEADLINE_Q)]:.4f} answered@8 "
            f"{cur['answer_rate'][str(HEADLINE_Q)]['mean_answered']:.2f} "
            f"({(time.time() - t0)/60:.1f}m)")

    # ---- auto replication flags: every pair vs the lit order, paired CI at q8 ----
    names = sorted(ARMS.keys(), key=lambda n: ARMS[n][0])                 # lit order best->worst
    pairs = {}
    n_flags = 0
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]                                     # a expected BETTER than b
            d = bootstrap_ci(q8_full[a][rows] - q8_full[b][rows])
            if d[1][0] > 0:
                verdict = "replicated"
            elif d[1][1] < 0:
                verdict = "INVERSION_FLAG"; n_flags += 1
            else:
                verdict = "indistinguishable, no flag"
            pairs[f"{a} > {b}"] = {"measured_delta_q8_full": d[0], "ci95": d[1], "verdict": verdict}
    out["pairwise_vs_lit_order"] = pairs
    out["n_inversion_flags"] = n_flags
    meas_rank = {n: r + 1 for r, n in enumerate(
        sorted(names, key=lambda n: -out["arms"][n]["full@10"][str(HEADLINE_Q)]))}
    out["measured_rank_q8"] = meas_rank
    out["rank_spearman_lit_vs_measured"] = spearman([ARMS[n][0] for n in names],
                                                    [meas_rank[n] for n in names])
    out["seconds"] = round(time.time() - t00, 1)
    outdir = ctx.outdir if args.smoke else OUTDIR
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, "strategy_ladder.json")
    json.dump(out, open(path, "w"), indent=2)
    log(f"[out] -> {path}")

    # ---- printed table ----
    print("\n=== LIT-ANCHORED STRATEGY LADDER (val, q8 headline; full curve in JSON) ===")
    hdr = (f"{'arm':>15} | {'lit':>3} | {'meas':>4} | {'q8 full':>8} | {'q8 tail':>8} | "
           f"{'ans@8':>6} | citation")
    print(hdr); print("-" * (len(hdr) + 20))
    for n in names:
        c = out["arms"][n]
        print(f"{n:>15} | {ARMS[n][0]:>3} | {meas_rank[n]:>4} | "
              f"{c['full@10'][str(HEADLINE_Q)]:>8.4f} | {c['tail@10'][str(HEADLINE_Q)]:>8.4f} | "
              f"{c['answer_rate'][str(HEADLINE_Q)]['mean_answered']:>6.2f} | {ARMS[n][1]}")
    print(f"\nrank Spearman(lit, measured) = {out['rank_spearman_lit_vs_measured']:.2f}; "
          f"inversion flags = {n_flags}")
    for k, v in pairs.items():
        if v["verdict"] == "INVERSION_FLAG":
            print(f"  FLAG: {k} inverted ({v['measured_delta_q8_full']:+.4f} CI {v['ci95']})")


if __name__ == "__main__":
    main()
