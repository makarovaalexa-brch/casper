r"""strategy_channel_suite.py -- THE COMPLETE strategy-by-channel curve suite (author redirect
2026-07-25; replaces U3b, subsumes its gap-closure numbers; U3c parked pending author review).

ONE harness (the ledger deployment protocol), per-question currency (refuse/unanswerable burns),
q grid {1,2,4,8,16} (q1 explicitly requested), full AND tail NDCG@10, same COLD_SEED 10k users,
SIGNED modules: sclite primary (all arms) + scfull subset row (member-mass / greedy / oracle /
items-pop) because its tower folds are the expensive ones.

ITEMS bank strategies (recomputed on THIS harness -- kills the ladder/ledger protocol seam):
  items-pop, items-entropy (raters-only rating entropy, all bands), items-HELF (harmonic of
  normalized entropy x log-freq), items-random-bank (the G9 random-200 bank, per-user order).
CONCEPT bank strategies:
  conc-member-mass (the 'pop' analog), conc-polarization (train signed-value dispersion std|v| --
  the ENTROPY analog), conc-HELF-analog (harmonic of normalized polarization x answer-rate;
  answer-rate = fraction of train users with support >= tau), conc-random, ADAPTIVE-greedy (the
  realizable U3b selector), ADAPTIVE-oracle (PRIVILEGED ceiling, labeled).
MIXED:
  mixed-c4-then-items (4 greedy-concept questions then popularity items),
  mixed-interleave (alternate concept-greedy / item-pop).
  (adaptive-mixed channel-chooser SKIPPED per the delivery clause -- needs a calibrated cross-
   channel expected-response scale; noted in the JSON.)

Deliverables: experiments/battery/strategy_channel_suite.json + ONE multi-panel PNG
(strategy_channel_curves_2026-07-25.png): {full, tail} x {items, concepts, mixed}, intercept line
on every panel, fixed CVD-safe colors (Okabe-Ito), PRIVILEGED tags. Gap-closure (oracle vs
member-mass static floor vs greedy) folded into the JSON.

Usage:
  python src/instrument/strategy_channel_suite.py --smoke
  python src/instrument/strategy_channel_suite.py [--full_threads] [--scfull]
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

from train_tower_t2 import log, I25Encoder
import run_battery_phaseA as PA
from run_battery_phaseA import (build_real_ctx, build_smoke_ctx, ndcg10_from_scores, bootstrap_ci,
                                OUTDIR, SEED)
from tradeoff_ledger import build_shared, Rung
from strategy_ladder import train_entropies
from adaptive_concept_arms import train_concept_stats, score_users

assert not hasattr(sys.modules[__name__], "load_answerer")

BUDGETS = (1, 2, 4, 8, 16)
PNG_NAME = "strategy_channel_curves_2026-07-25.png"
QBANK_JSON = os.path.join(_ROOT, "experiments", "battery", "qbank_expansion.json")
QBANK_PNG = os.path.join(_ROOT, "experiments", "battery", "qbank_expansion.png")
# Okabe-Ito CVD-safe, FIXED assignment (never cycled; shared arms share colors across panels)
COLORS = {"items-pop": "#0072B2", "items-entropy": "#D55E00", "items-HELF": "#009E73",
          "items-random-bank": "#999999",
          "conc-member-mass": "#0072B2", "conc-polarization": "#D55E00",
          "conc-HELF-analog": "#009E73", "conc-random": "#999999",
          "ADAPTIVE-greedy": "#CC79A7", "ADAPTIVE-oracle (PRIVILEGED)": "#E69F00",
          "mixed-c4-then-items": "#56B4E9", "mixed-interleave": "#F0E442"}


# ============================================================================= walk engines
def snapshot_score(rung, rows, items_ev, conc_ev, budgets_hit, store, arm, q):
    full, tail, _ = score_users(rung, items_ev, conc_ev, rows)
    store[arm][q] = {"full@10": float(np.nanmean(full)), "tail@10": float(np.nanmean(tail)),
                     "mean_answered": float(np.mean([len(s[0]) for s in items_ev])
                                            + np.mean([len(c) for c in conc_ev])),
                     "_fv": full, "_tv": tail}


def items_walk(rung, rows, order_fn, budgets=BUDGETS):
    """Static item strategy: order_fn(r) -> iterable of sids. Burn per ask; answered iff rated."""
    sh = rung.sh
    out = {}
    ev_at = {q: [] for q in budgets}
    qmax = max(budgets)
    for r in rows:
        d = sh["lvl_lookup"][r]
        items = []
        nq = 0
        for i in order_fn(r):
            i = int(i); nq += 1
            if i in d:
                items.append((i, d[i]))
            if nq in budgets:
                ev_at[nq].append((np.asarray([s for s, _ in items], np.int64),
                                  np.asarray([l for _, l in items], np.int64)))
            if nq >= qmax:
                break
        while nq < qmax:                                     # bank exhausted: burn to the end
            nq += 1
            if nq in budgets:
                ev_at[nq].append((np.asarray([s for s, _ in items], np.int64),
                                  np.asarray([l for _, l in items], np.int64)))
    res = {}
    for q in budgets:
        full, tail, _ = score_users(rung, ev_at[q], [[] for _ in rows], rows)
        res[q] = {"full@10": float(np.nanmean(full)), "tail@10": float(np.nanmean(tail)),
                  "mean_answered": float(np.mean([len(s[0]) for s in ev_at[q]])),
                  "_fv": full, "_tv": tail}
    return res


def concept_walk(rung, rows, mode, mean_abs, std_v, static_order=None, budgets=BUDGETS):
    """Concept strategies. mode: 'static' (static_order = global array or 'random'), 'greedy',
    'oracle'. Signed fold rule: answered iff structural; refuse band burns without folding."""
    ctx, sh = rung.ctx, rung.sh
    C = sh["d_c"].shape[0]
    qmax = max(budgets)
    empty_i = (np.empty(0, np.int64), np.empty(0, np.int64))
    D = sh["d_c"].numpy().astype(np.float32)
    Vs = sh["conc_value_signed"]; Fs = sh["conc_fold_signed"]; answ = sh["conc_answerable"]
    asked = np.zeros((len(rows), C), bool)
    folded = [[] for _ in rows]
    Z = torch.zeros(len(rows), ctx.d)
    per_user_rand = None
    if mode == "static" and isinstance(static_order, str) and static_order == "random":
        rng = np.random.default_rng(SEED)
        per_user_rand = [rng.permutation(C) for _ in rows]
    if mode == "oracle":
        prio = np.where(answ & Fs, np.abs(Vs), -1.0)
    out = {}
    for step in range(1, qmax + 1):
        if mode == "static":
            picks = []
            for j, r in enumerate(rows):
                order = per_user_rand[j] if per_user_rand is not None else static_order
                nxt = next((int(c) for c in order if not asked[j, int(c)]), -1)
                picks.append(nxt)
        elif mode == "oracle":
            pr = prio[np.asarray(rows)].copy(); pr[asked] = -2.0
            picks = pr.argmax(1).tolist()
        else:
            if step == 1:
                base = np.tile(mean_abs, (len(rows), 1))
            else:
                base = np.abs(Z.numpy() @ D.T) * std_v[None, :]
            base[asked] = -np.inf
            picks = base.argmax(1).tolist()
        changed = False
        for j, r in enumerate(rows):
            c = int(picks[j])
            if c < 0:
                continue
            asked[j, c] = True
            if answ[r, c] and Fs[r, c]:
                folded[j].append((c, float(Vs[r, c]))); changed = True
        if mode == "greedy" and changed and step < qmax:
            Z = rung.z_batch([empty_i] * len(rows), [list(f) for f in folded])
        if step in budgets:
            full, tail, _ = score_users(rung, [empty_i] * len(rows),
                                        [list(f) for f in folded], rows)
            out[step] = {"full@10": float(np.nanmean(full)), "tail@10": float(np.nanmean(tail)),
                         "mean_answered": float(np.mean([len(f) for f in folded])),
                         "_fv": full, "_tv": tail}
    return out


def mixed_walk(rung, rows, recipe, mean_abs, std_v, budgets=BUDGETS):
    """Mixed strategies. recipe 'c4' = 4 greedy concepts then pop items; 'alt' = alternate
    concept-greedy (odd q) / item-pop (even q)."""
    ctx, sh = rung.ctx, rung.sh
    C = sh["d_c"].shape[0]
    qmax = max(budgets)
    empty_i = (np.empty(0, np.int64), np.empty(0, np.int64))
    D = sh["d_c"].numpy().astype(np.float32)
    Vs = sh["conc_value_signed"]; Fs = sh["conc_fold_signed"]; answ = sh["conc_answerable"]
    asked_c = np.zeros((len(rows), C), bool)
    folded = [[] for _ in rows]
    items = [[] for _ in rows]
    item_ptr = [0] * len(rows)
    Z = torch.zeros(len(rows), ctx.d)
    pop = sh["order_pop"]
    out = {}
    for step in range(1, qmax + 1):
        concept_turn = (step <= 4) if recipe == "c4" else (step % 2 == 1)
        if concept_turn:
            if step == 1:
                base = np.tile(mean_abs, (len(rows), 1))
            else:
                base = np.abs(Z.numpy() @ D.T) * std_v[None, :]
            base[asked_c] = -np.inf
            picks = base.argmax(1).tolist()
            changed = False
            for j, r in enumerate(rows):
                c = int(picks[j]); asked_c[j, c] = True
                if answ[r, c] and Fs[r, c]:
                    folded[j].append((c, float(Vs[r, c]))); changed = True
        else:
            for j, r in enumerate(rows):
                d = sh["lvl_lookup"][r]
                while item_ptr[j] < len(pop):
                    i = int(pop[item_ptr[j]]); item_ptr[j] += 1
                    if i in d:
                        items[j].append((i, d[i]))
                    break                                    # one ASK per turn (answered or not)
            changed = True
        if changed and step < qmax:
            Z = rung.z_batch([(np.asarray([s for s, _ in it], np.int64),
                               np.asarray([l for _, l in it], np.int64)) for it in items],
                             [list(f) for f in folded])
        if step in budgets:
            iev = [(np.asarray([s for s, _ in it], np.int64),
                    np.asarray([l for _, l in it], np.int64)) for it in items]
            full, tail, _ = score_users(rung, iev, [list(f) for f in folded], rows)
            out[step] = {"full@10": float(np.nanmean(full)), "tail@10": float(np.nanmean(tail)),
                         "mean_answered": float(np.mean([len(f) for f in folded])
                                                + np.mean([len(it) for it in items])),
                         "_fv": full, "_tv": tail}
    return out


def unified_walk(rung, rows, item_score, conc_score, budgets=BUDGETS):
    """Unified question bank: merge ALL items + ALL concepts into ONE global static ranking by a
    single cross-channel score (item_score, conc_score already on a comparable [0,1] scale), ask the
    top-q, fold what each user can answer (item iff rated; concept iff structurally answerable). The
    order is the SAME for every user (static bank) -- the answer never changes which question is
    asked, so any concept-first-then-items shape here EMERGES from the score, it is not scheduled."""
    sh = rung.sh
    C = sh["d_c"].shape[0]
    Vs = sh["conc_value_signed"]; Fs = sh["conc_fold_signed"]; answ = sh["conc_answerable"]
    ni = len(item_score)
    key = np.concatenate([np.asarray(item_score, np.float64), np.asarray(conc_score, np.float64)])
    chan = np.concatenate([np.zeros(ni, np.int8), np.ones(C, np.int8)])      # 0 item, 1 concept
    ids = np.concatenate([np.arange(ni), np.arange(C)]).astype(np.int64)
    order = np.argsort(-key, kind="stable")
    qmax = max(budgets)
    top = order[:qmax]
    iev_at = {q: [] for q in budgets}; cev_at = {q: [] for q in budgets}
    for j, r in enumerate(rows):
        d = sh["lvl_lookup"][r]
        items = []; folded = []
        for nq, o in enumerate(top, start=1):
            if chan[o] == 0:
                i = int(ids[o])
                if i in d: items.append((i, d[i]))
            else:
                c = int(ids[o])
                if answ[r, c] and Fs[r, c]: folded.append((c, float(Vs[r, c])))
            if nq in budgets:
                iev_at[nq].append((np.asarray([s for s, _ in items], np.int64),
                                   np.asarray([l for _, l in items], np.int64)))
                cev_at[nq].append(list(folded))
    res = {}
    for q in budgets:
        nc = int((chan[top[:q]] == 1).sum())
        full, tail, _ = score_users(rung, iev_at[q], cev_at[q], rows)
        res[q] = {"full@10": float(np.nanmean(full)), "tail@10": float(np.nanmean(tail)),
                  "mean_answered": float(np.mean([len(s[0]) for s in iev_at[q]])
                                         + np.mean([len(f) for f in cev_at[q]])),
                  "bank_conc": nc, "bank_item": q - nc, "_fv": full, "_tv": tail}
    return res


def make_qbank_png(results, intercept, path, balanced=False):
    """Question-bank-expansion figure: full & tail with EQUAL y-axes; per strategy (prevalence,
    entropy, HELF), items-only bank (dashed) vs unified items+concepts bank (solid)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    STRAT = [("prevalence", "items-pop", "unified-prevalence", "#0072B2"),
             ("entropy", "items-entropy", "unified-entropy", "#D55E00"),
             ("HELF", "items-HELF", "unified-HELF", "#009E73")]
    allv = []
    for _, io, un, _ in STRAT:
        for arm in (io, un):
            for q in BUDGETS:
                allv += [results[arm][q]["full@10"], results[arm][q]["tail@10"]]
    ymin = min(min(allv), intercept["full"], intercept["tail"]); ymax = max(allv)
    pad = (ymax - ymin) * 0.06
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True)
    for ax, metric, ikey in ((axes[0], "full@10", "full"), (axes[1], "tail@10", "tail")):
        ax.axhline(intercept[ikey], color="#888888", ls=":", lw=1.1, zorder=1,
                   label="no-answer intercept")
        for sname, io, un, col in STRAT:
            ax.plot(BUDGETS, [results[io][q][metric] for q in BUDGETS], "--o", color=col,
                    lw=1.4, ms=4, alpha=0.65, zorder=2, label=f"{sname} · items-only")
            ax.plot(BUDGETS, [results[un][q][metric] for q in BUDGETS], "-s", color=col,
                    lw=2.3, ms=5, zorder=3, label=f"{sname} · unified bank")
        ax.set_xscale("log", base=2); ax.set_xticks(BUDGETS); ax.set_xticklabels(BUDGETS)
        ax.set_xlabel("questions asked $q$"); ax.set_title(metric, fontsize=10)
        ax.set_ylim(ymin - pad, ymax + pad); ax.grid(alpha=0.3, zorder=0)
    axes[0].set_ylabel("NDCG@10")
    axes[0].legend(fontsize=6.8, loc="upper left", ncol=1, framealpha=0.9)
    fig.suptitle(f"Question-bank expansion: unified items+concepts "
                 f"({'balanced z-score merge' if balanced else 'raw-score merge'}) vs items-only "
                 f"(static bank; answer does not steer selection)", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=160); fig.savefig(path.replace(".png", ".pdf"))
    plt.close(fig)


# ============================================================================= figure
def make_png(results, intercept, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    panels = {"items": [k for k in results if k.startswith("items-")],
              "concepts": [k for k in results if k.startswith("conc-") or "ADAPTIVE" in k],
              "mixed": [k for k in results if k.startswith("mixed-")]}
    fig, axes = plt.subplots(2, 3, figsize=(16, 9), sharex=True)
    fig.suptitle("Strategy-by-channel deployment curves — signed C-lite, per-question currency, "
                 "10k COLD_SEED val users (2026-07-25)", fontsize=13)
    for col, (pname, arms) in enumerate(panels.items()):
        for row, metric in enumerate(("full@10", "tail@10")):
            ax = axes[row][col]
            iv = intercept["full"] if metric == "full@10" else intercept["tail"]
            ax.axhline(iv, color="#888888", lw=1.2, ls="--", zorder=1)
            ax.annotate(f"intercept {iv:.3f}", xy=(1, iv), fontsize=7, color="#666666",
                        xytext=(2, 3), textcoords="offset points")
            for arm in arms:
                cur = results[arm]
                xs = list(BUDGETS)
                ys = [cur[q][metric] for q in BUDGETS]
                ax.plot(xs, ys, marker="o", ms=4, lw=2, label=arm, color=COLORS.get(arm, "#333"))
            ax.set_xscale("log", base=2)
            ax.set_xticks(list(BUDGETS)); ax.set_xticklabels([str(q) for q in BUDGETS])
            ax.grid(alpha=0.25, lw=0.5)
            if row == 0:
                ax.set_title(pname, fontsize=11)
            if row == 1:
                ax.set_xlabel("questions asked (budget q)")
            if col == 0:
                ax.set_ylabel(("FULL" if metric == "full@10" else "TAIL") + " NDCG@10")
            ax.legend(fontsize=7, loc="best", framealpha=0.9)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ============================================================================= main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--snapshot", default=PA.SNAP_DEFAULT)
    ap.add_argument("--sclite_ckpt", default=os.path.join(_ROOT, ".cache", "instrument",
                                                          "cfold_signed_best.pt"))
    ap.add_argument("--scfull_ckpt", default=os.path.join(_ROOT, ".cache", "instrument",
                                                          "cfull_signed_best.pt"))
    ap.add_argument("--scfull", action="store_true", default=True)
    ap.add_argument("--no_scfull", dest="scfull", action="store_false")
    ap.add_argument("--full_threads", action="store_true")
    ap.add_argument("--qbank", action="store_true",
                    help="question-bank-expansion mode: items-only vs unified items+concepts bank "
                         "under prevalence/entropy/HELF; equal-y figure; no oracle/mixed")
    ap.add_argument("--balanced", action="store_true",
                    help="qbank: z-score each channel's score before merging (balanced interleave) "
                         "instead of raw [0,1] scores (which let concepts crowd out items)")
    args = ap.parse_args()
    t00 = time.time()
    ctx = build_smoke_ctx() if args.smoke else build_real_ctx(args.snapshot)
    sh = build_shared(ctx)
    if args.smoke and not sh.get("signed_available"):
        rng = np.random.RandomState(4)
        C = sh["d_c"].shape[0]
        sh["conc_value_signed"] = (rng.rand(ctx.n, C).astype(np.float32) * 2 - 1)
        sh["conc_fold_signed"] = rng.rand(ctx.n, C) > 0.15
        sh["signed_available"] = True
    assert sh.get("signed_available"), "signed prereg cache required"
    Mm = sparse.csr_matrix((np.ones(sum(len(sh["members"][t]) for t in sh["tags"]), np.float32),
                            (np.concatenate([sh["members"][t] for t in sh["tags"]]),
                             np.concatenate([np.full(len(sh["members"][t]), i)
                                             for i, t in enumerate(sh["tags"])]))),
                           shape=(ctx.ni, len(sh["tags"])))
    pexp = (ctx.cnt @ np.asarray(Mm.todense())) / max(ctx.cnt.sum(), 1e-9)
    mean_abs, std_v = train_concept_stats(ctx, Mm, pexp, smoke=args.smoke)
    # concept answer-rate (HELF-analog): fraction of train users with support >= tau
    CONCSTATS = os.path.join(_ROOT, ".cache", "instrument", "signed_train_concstats.npz")
    if args.smoke or not os.path.exists(CONCSTATS):
        ans_rate = np.random.RandomState(6).rand(len(std_v)).astype(np.float32)
    else:
        z = np.load(CONCSTATS)
        ans_rate = (z["n"] / 140768.0).astype(np.float32)
    pol_n = std_v / max(std_v.max(), 1e-9)
    ar_n = ans_rate / max(ans_rate.max(), 1e-9)
    helf_c = 2 * pol_n * ar_n / np.clip(pol_n + ar_n, 1e-9, None)
    # item orders (recomputed on THIS harness)
    H, H0, n_raters = train_entropies(ctx)
    Hn = H / max(H.max(), 1e-9)
    Fn = np.log1p(ctx.cnt) / max(np.log1p(ctx.cnt).max(), 1e-9)
    helf_i = 2 * Hn * Fn / np.clip(Hn + Fn, 1e-9, None)
    ord_pop = sh["order_pop"]; ord_ent = np.argsort(-H); ord_helf = np.argsort(-helf_i)
    bank_rand = ctx.banks["rand"][0]
    rng_rb = np.random.default_rng(SEED)
    rand_orders = {}
    rows = [r for r in range(ctx.n) if ctx.va_te[r].nnz > 0]
    for r in rows:
        rand_orders[r] = rng_rb.permutation(bank_rand)
    # rungs
    def load_rung(name):
        if args.smoke:
            from concept_fold import ConceptFoldNet
            net = ConceptFoldNet(len(sh["tags"]), d=ctx.d, h=32, conc_init=sh["d_c"].numpy())
            with torch.no_grad():
                for p in net.mlp[-1].parameters():
                    p.add_(torch.randn_like(p) * 0.05)
            net.eval()
            return Rung(name, ctx, sh, clite_net=net, val_source="signed")
        if name == "sclite":
            from concept_fold import ConceptFoldNet
            blob = torch.load(args.sclite_ckpt, map_location="cpu")
            net = ConceptFoldNet(len(blob["tags"]), d=ctx.d, h=blob["hidden"])
            net.load_state_dict(blob["net"]); net.eval()
            return Rung(name, ctx, sh, clite_net=net, val_source="signed")
        blob = torch.load(args.scfull_ckpt, map_location="cpu")
        native = ctx.enc._native[0]
        enc_sf = I25Encoder(ctx.ni, native, d_lat=ctx.d, token_mode="film",
                            n_concepts=len(sh["tags"]))
        enc_sf.load_state_dict(blob["enc"]); enc_sf.eval()
        return Rung(name, ctx, sh, cfull_enc=enc_sf, val_source="signed")
    rung = load_rung("sclite")
    # intercept
    empty_i = (np.empty(0, np.int64), np.empty(0, np.int64))
    f0, t0v, _ = score_users(rung, [empty_i] * len(rows), [[] for _ in rows], rows)
    intercept = {"full": float(np.nanmean(f0)), "tail": float(np.nanmean(t0v))}
    results = {}
    def runlog(name, fn, *a, **k):
        t0 = time.time()
        results[name] = fn(*a, **k)
        cur = results[name]
        log(f"[{name}] " + " ".join(f"q{q}={cur[q]['full@10']:.4f}/{cur[q]['tail@10']:.4f}"
                                    for q in BUDGETS) + f" ({(time.time()-t0)/60:.1f}m)")
    if args.qbank:
        # question-bank expansion: items-only vs unified items+concepts, 3 cross-channel scores.
        runlog("items-pop", items_walk, rung, rows, lambda r: ord_pop)
        runlog("items-entropy", items_walk, rung, rows, lambda r: ord_ent)
        runlog("items-HELF", items_walk, rung, rows, lambda r: ord_helf)
        item_prev = n_raters.astype(np.float64) / 140768.0          # fraction of train users who rated

        def _zsc(x):
            x = np.asarray(x, np.float64); return (x - x.mean()) / (x.std() + 1e-9)
        tf = _zsc if args.balanced else (lambda x: np.asarray(x, np.float64))
        runlog("unified-prevalence", unified_walk, rung, rows, tf(item_prev), tf(ans_rate))
        runlog("unified-entropy", unified_walk, rung, rows, tf(Hn), tf(pol_n))
        runlog("unified-HELF", unified_walk, rung, rows, tf(helf_i), tf(helf_c))
        jpath = QBANK_JSON.replace(".json", "_balanced.json") if args.balanced else QBANK_JSON
        ppath = QBANK_PNG.replace(".png", "_balanced.png") if args.balanced else QBANK_PNG
        out = {"analysis": "qbank_expansion", "merge": "balanced-zscore" if args.balanced else "raw",
               "n_users": len(rows), "budgets": BUDGETS, "rung": "sclite", "intercept": intercept,
               "arms": {k: {str(q): {kk: vv for kk, vv in results[k][q].items()
                                     if not kk.startswith("_")} for q in BUDGETS} for k in results}}
        os.makedirs(os.path.dirname(jpath), exist_ok=True)
        json.dump(out, open(jpath, "w"), indent=2)
        make_qbank_png(results, intercept, ppath, balanced=args.balanced)
        log(f"[qbank] wrote {os.path.basename(jpath)} + {os.path.basename(ppath)} "
            f"({(time.time()-t00)/60:.1f}m)")
        return
    # ITEMS
    runlog("items-pop", items_walk, rung, rows, lambda r: ord_pop)
    runlog("items-entropy", items_walk, rung, rows, lambda r: ord_ent)
    runlog("items-HELF", items_walk, rung, rows, lambda r: ord_helf)
    runlog("items-random-bank", items_walk, rung, rows, lambda r: rand_orders[r])
    # CONCEPTS
    runlog("conc-member-mass", concept_walk, rung, rows, "static", mean_abs, std_v,
           static_order=sh["bank_conc"])
    runlog("conc-polarization", concept_walk, rung, rows, "static", mean_abs, std_v,
           static_order=np.argsort(-std_v))
    runlog("conc-HELF-analog", concept_walk, rung, rows, "static", mean_abs, std_v,
           static_order=np.argsort(-helf_c))
    runlog("conc-random", concept_walk, rung, rows, "static", mean_abs, std_v,
           static_order="random")
    runlog("ADAPTIVE-greedy", concept_walk, rung, rows, "greedy", mean_abs, std_v)
    runlog("ADAPTIVE-oracle (PRIVILEGED)", concept_walk, rung, rows, "oracle", mean_abs, std_v)
    # MIXED
    runlog("mixed-c4-then-items", mixed_walk, rung, rows, "c4", mean_abs, std_v)
    runlog("mixed-interleave", mixed_walk, rung, rows, "alt", mean_abs, std_v)
    # gap closure (subsumes U3b; static floor = member-mass)
    gap = {}
    for q in BUDGETS:
        st = results["conc-member-mass"][q]; gr = results["ADAPTIVE-greedy"][q]
        orc = results["ADAPTIVE-oracle (PRIVILEGED)"][q]
        go_f = orc["full@10"] - st["full@10"]; gg_f = gr["full@10"] - st["full@10"]
        go_t = orc["tail@10"] - st["tail@10"]; gg_t = gr["tail@10"] - st["tail@10"]
        gap[str(q)] = {"oracle_minus_static_full": go_f,
                       "greedy_closes_full": gg_f / go_f if abs(go_f) > 1e-9 else float("nan"),
                       "oracle_minus_static_tail": go_t,
                       "greedy_closes_tail": gg_t / go_t if abs(go_t) > 1e-9 else float("nan")}
    d8 = bootstrap_ci(results["ADAPTIVE-greedy"][8]["_fv"] - results["conc-member-mass"][8]["_fv"])
    # scfull subset row
    scfull_res = None
    if args.scfull and (args.smoke or os.path.exists(args.scfull_ckpt)):
        if not args.smoke:
            rung2 = load_rung("scfull")
            scfull_res = {}
            for nm, fn, a in (("items-pop", items_walk, (rung2, rows, lambda r: ord_pop)),
                              ("conc-member-mass", concept_walk,
                               (rung2, rows, "static", mean_abs, std_v)),
                              ("ADAPTIVE-greedy", concept_walk,
                               (rung2, rows, "greedy", mean_abs, std_v)),
                              ("ADAPTIVE-oracle (PRIVILEGED)", concept_walk,
                               (rung2, rows, "oracle", mean_abs, std_v))):
                t0 = time.time()
                kw = {"static_order": sh["bank_conc"]} if nm == "conc-member-mass" else {}
                scfull_res[nm] = {q: {k: v for k, v in cur.items() if not k.startswith("_")}
                                  for q, cur in fn(*a, **kw).items()}
                log(f"[scfull {nm}] q8={scfull_res[nm][8]['full@10']:.4f} "
                    f"({(time.time()-t0)/60:.1f}m)")
    out = {"analysis": "strategy_channel_suite", "n_users": len(rows), "budgets": list(BUDGETS),
           "rung_primary": "sclite (signed C-lite)", "intercept": intercept,
           "adaptive_mixed_skipped": "needs a calibrated cross-channel expected-response scale; "
                                     "skipped per the delivery clause",
           "arms": {k: {str(q): {kk: vv for kk, vv in results[k][q].items()
                                 if not kk.startswith("_")} for q in BUDGETS} for k in results},
           "gap_closure_u3b_subsumed": gap,
           "greedy_minus_membermass_q8_full": {"mean": d8[0], "ci95": d8[1]},
           "scfull_subset": scfull_res}
    out["seconds"] = round(time.time() - t00, 1)
    outdir = ctx.outdir if args.smoke else OUTDIR
    os.makedirs(outdir, exist_ok=True)
    jpath = os.path.join(outdir, "strategy_channel_suite.json")
    json.dump(out, open(jpath, "w"), indent=2, default=float)
    ppath = os.path.join(outdir, PNG_NAME)
    make_png(results, intercept, ppath)
    log(f"[out] -> {jpath} + {ppath}")
    print("\n=== STRATEGY-BY-CHANNEL SUITE (sclite; intercept "
          f"{intercept['full']:.4f}/{intercept['tail']:.4f}) ===")
    for k in results:
        cur = results[k]
        print(f"{k:>32}: " + " ".join(f"q{q}={cur[q]['full@10']:.4f}" for q in BUDGETS))
    print("gap closed by greedy (full): " + " ".join(
        f"q{q}={gap[str(q)]['greedy_closes_full']:.0%}" for q in BUDGETS))


if __name__ == "__main__":
    main()
