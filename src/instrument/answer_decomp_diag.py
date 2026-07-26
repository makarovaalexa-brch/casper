r"""answer_decomp_diag.py -- THREE focused diagnostics on the strong stack (frozen i25 tower
t2i25_EP4_SNAP.pt + signed C-lite cfold_signed_best.pt), 10k COLD_SEED users, ML-25M Liang ruler,
credit-neutral masking, leak-safe. Diagnostic-for-understanding (NOT paper). Reuses the
answer_contrast.py / tradeoff_ledger.py machinery.

D1 -- PER-SINGLE-QUESTION ITEM-ASK BREAKDOWN. Item-asking under HONEST behavioral (B) answers at
   EVERY integer budget q=0..8, three question orders (popularity / random / entropy). Per q:
   full@10, full@100, cumulative items answered, marginal delta full@10, and for the q-th question:
   pop rank, answered fraction, mean rating if answered. Plus a MASK-ONLY control (score the
   intercept belief but credit-neutrally mask the answered items) to test the masking-artifact
   hypothesis. GOAL: locate + explain the non-smooth NDCG jump.

D2 -- WHY FINE CONCEPTS HURT. From a per-(user,concept) marginal single-fold lift matrix L:
   (a) DIRECTION QUALITY: per-concept member-lift AUC of the cold unit-positive fold delta, binned
       fine/medium/broad (are sparse-member centroids just noisy directions?).
   (b) THRESHOLD SWEEP: mean per-answer lift restricting to concepts with >=100 / >=300 members
       (does raising min-members remove the harm?).
   (c) POPULARITY-FLOOR DISPLACEMENT: per-concept Spearman(cold fold-delta, log-pop) + mean delta on
       head vs tail items, binned (does the harmful fine fold push mass toward popular items?).
   Verdict: sparse-centroid direction-quality artifact vs real operator problem. Also confirms the
   per-granularity G2 canary violation.

D3 -- UTILITY-BASED CONCEPT SELECTION (the productive lever). Rank each user's answerable concepts by
   their held-out marginal NDCG@10 utility (L), ask top-utility concepts (honest B answers), q0..8.
   vs K (|SEL|-selection), the polarization bank (B), and reference U (utility-ANSWER oracle). Reports
   the granularity distribution of picked concepts. Candidate TEACHER for a realizable selector.

CONTROLS (HARD RULE 5): q0 == cold intercept (canonical-snap); leak check (fold-in INTERSECT held-out
empty). Utility selection + L are PRIVILEGED (peek held-out) -- labeled, diagnostic only.

Usage:
  python src/instrument/answer_decomp_diag.py --smoke [--diag all]
  python src/instrument/answer_decomp_diag.py [--diag 1|23|all] [--full_threads]
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

from train_tower_t2 import log
import run_battery_phaseA as PA
from run_battery_phaseA import build_real_ctx, build_smoke_ctx, OUTDIR, spearman
from tradeoff_ledger import build_shared, Rung, fold_items_enc
from adaptive_concept_arms import train_concept_stats, score_users
from answer_contrast import (Behavioral, walk_static_item, walk_static_concept,
                             walk_oracle_select_concept, score_snapshots, score_users_multi,
                             INTERCEPT_REF)

assert not hasattr(sys.modules[__name__], "load_answerer")

INT_BUDGETS = tuple(range(0, 9))                                  # every integer q 0..8


# ============================================================================= rung loader
def load_sclite_rung(ctx, sh, args):
    from concept_fold import ConceptFoldNet
    if args.smoke:
        net = ConceptFoldNet(len(sh["tags"]), d=ctx.d, h=32, conc_init=sh["d_c"].numpy())
        with torch.no_grad():
            for p in net.mlp[-1].parameters():
                p.add_(torch.randn_like(p) * 0.05)
    else:
        blob = torch.load(args.sclite_ckpt, map_location="cpu")
        net = ConceptFoldNet(len(blob["tags"]), d=ctx.d, h=blob["hidden"])
        net.load_state_dict(blob["net"])
    net.eval()
    return Rung("sclite", ctx, sh, clite_net=net, val_source="signed")


def q_metadata(order, lvl_lookup, rows, ctx, qmax):
    """For each asked position 1..qmax under `order`: the item id, its popularity rank (0=most pop),
    fraction of users who answered (rated in fold-in), and mean rating level if answered."""
    poprank = np.empty(ctx.ni, np.int64)
    poprank[ctx.order_pop] = np.arange(ctx.ni)                    # rank of every item in pop order
    meta = []
    for step in range(1, qmax + 1):
        i = int(order[step - 1])
        rated = [lvl_lookup[r][i] for r in rows if i in lvl_lookup[r]]
        meta.append({"q": step, "item": i, "pop_rank": int(poprank[i]),
                     "answered_frac": len(rated) / len(rows),
                     "mean_rating_lvl_if_answered": float(np.mean(rated)) if rated else None})
    return meta


def score_maskonly(rung, rows, ev, budgets):
    """MASK-ONLY control: score the COLD-INTERCEPT belief (no fold) but apply the SAME credit-neutral
    masking of the answered items as the folded arm. If NDCG rises purely from masking the asked
    popular items, the jump is a masking artifact (mechanism c)."""
    ctx = rung.ctx
    empty_i = [(np.empty(0, np.int64), np.empty(0, np.int64)) for _ in rows]
    z0 = rung.z_batch(empty_i, [[] for _ in rows])               # (n, d) intercept, identical per row
    out = {}
    for q in budgets:
        item_seqs = ev[q]
        full = np.full(len(rows), np.nan)
        for st in range(0, len(rows), 500):
            ch = rows[st:st + 500]
            S = (z0[st:st + len(ch)] @ ctx.Wd.T + ctx.bd).numpy().astype(np.float32)
            mask = ctx.va_tr[ch].copy(); te = ctx.va_te[ch].tolil()
            for j in range(len(ch)):
                s = item_seqs[st + j][0]
                if len(s):
                    extra = sparse.csr_matrix((np.ones(len(s), np.float32), (np.zeros(len(s)), s)),
                                              shape=(1, ctx.ni))
                    mask[j] = mask[j] + extra
                    for i in np.asarray(s).tolist():
                        te[j, i] = 0
            mask.data[:] = 1.0; te = te.tocsr(); te.eliminate_zeros()
            from run_battery_phaseA import ndcg10_from_scores
            f, _ = ndcg10_from_scores(S, mask.tocsr(), te, ctx.head_mask)
            full[st:st + len(ch)] = f
        out[q] = float(np.nanmean(full))
    return out


# ============================================================================= D1
def diagnostic1(rung, rows, sh, ctx, args):
    log("=== DIAGNOSTIC 1: per-single-question item-ask breakdown ===")
    ctx.order_pop = sh["order_pop"]
    bmodel = Behavioral(sh, sh["lvl_lookup"])
    rng = np.random.default_rng(4242)
    rand_order = rng.permutation(ctx.ni).astype(np.int64)
    orders = {"popularity": sh["order_pop"], "random": rand_order,
              "entropy": sh["order_entropy"]}
    qmax = max(INT_BUDGETS)
    d1 = {"budgets": list(INT_BUDGETS), "orders": {}}
    for name, order in orders.items():
        ev = walk_static_item(rung, rows, bmodel, order, budgets=INT_BUDGETS)
        sc = score_snapshots(rung, rows, ev, "item", budgets=INT_BUDGETS)
        meta = q_metadata(order, sh["lvl_lookup"], rows, ctx, qmax)
        maskonly = score_maskonly(rung, rows, ev, INT_BUDGETS)
        curve = {}
        prev = None
        for q in INT_BUDGETS:
            f10 = sc[q]["full@10"]
            curve[q] = {"full@10": f10, "full@100": sc[q]["full@100"],
                        "tail@10": sc[q]["tail@10"], "cum_items_answered": sc[q]["mean_answered"],
                        "marginal_full@10": (None if prev is None else f10 - prev),
                        "maskonly_full@10": maskonly[q],
                        "question": (None if q == 0 else meta[q - 1])}
            prev = f10
        d1["orders"][name] = curve
        log(f"[D1 {name:>10}] " + " ".join(f"q{q}={curve[q]['full@10']:.4f}"
                                           f"(d{'' if curve[q]['marginal_full@10'] is None else '%+.4f'%curve[q]['marginal_full@10']})"
                                           for q in INT_BUDGETS))
        log(f"[D1 {name:>10}] maskonly " + " ".join(f"q{q}={maskonly[q]:.4f}" for q in INT_BUDGETS))
    return d1


# ============================================================================= L matrix (D2+D3 core)
def compute_marginal_L(rung, rows, sh, ctx):
    """Per-(user,concept) marginal single-fold lift matrix over answerable-and-fold cells (honest B
    value). L_full/L_tail: (n_rows, C) float32, NaN where not answerable. PRIVILEGED (held-out NDCG).
    Same total fold cost as the granularity A2 pass."""
    Vb = sh["conc_value_signed"]; AF = sh["conc_answerable"] & sh["conc_fold_signed"]
    C = len(sh["tags"])
    rows_arr = np.asarray(rows)
    row_pos = {r: j for j, r in enumerate(rows)}
    empty = (np.empty(0, np.int64), np.empty(0, np.int64))
    f0, t0, _ = score_users(rung, [empty] * len(rows), [[] for _ in rows], rows)
    L_full = np.full((len(rows), C), np.nan, np.float32)
    L_tail = np.full((len(rows), C), np.nan, np.float32)
    nans = np.zeros(C, np.int64)
    t0a = time.time()
    for c in range(C):
        sel = AF[rows_arr, c]
        rc = rows_arr[sel]
        nans[c] = len(rc)
        if len(rc) < 1:
            continue
        conc = [[(c, float(Vb[r, c]))] for r in rc]
        fc, tc, _ = score_users(rung, [empty] * len(rc), conc, list(rc))
        idx = np.array([row_pos[r] for r in rc.tolist()])
        L_full[idx, c] = fc - f0[idx]
        L_tail[idx, c] = tc - t0[idx]
        if c % 100 == 0:
            log(f"[L] concept {c}/{C} ({(time.time()-t0a)/60:.1f}m)")
    return L_full, L_tail, f0, t0, nans


# ============================================================================= D2
def diagnostic2(rung, rows, sh, ctx, L_full, L_tail, nans):
    log("=== DIAGNOSTIC 2: why fine concepts hurt ===")
    C = len(sh["tags"])
    member_count = np.array([len(sh["members"][t]) for t in sh["tags"]], np.int64)
    log_cnt = np.log1p(ctx.cnt)
    head = ctx.head_mask

    # ---- (a) DIRECTION QUALITY: per-concept cold unit-positive fold delta; member-AUC + pop-corr ----
    empty1 = [(np.empty(0, np.int64), np.empty(0, np.int64))]
    z0 = rung.z_batch(empty1, [[]])[0].numpy()
    member_auc = np.full(C, np.nan)
    delta_pop_spear = np.full(C, np.nan)
    head_minus_tail_delta = np.full(C, np.nan)
    t0a = time.time()
    for c in range(C):
        mem = sh["members"][sh["tags"][c]]
        non = sh["get_match"](c)                                  # pop-matched non-members
        zc = rung.z_batch(empty1, [[(c, 1.0)]])[0].numpy()
        delta = (zc - z0) @ ctx.Wd.T.numpy()                     # per-item fold contribution
        sm = delta[mem]; sn = delta[non]; k = len(sm)
        ranks = np.argsort(np.argsort(np.concatenate([sm, sn])))[:k].sum()
        member_auc[c] = (ranks - k * (k - 1) / 2) / (k * k)
        delta_pop_spear[c] = spearman(delta, log_cnt)
        head_minus_tail_delta[c] = float(delta[head].mean() - delta[~head].mean())
        if c % 200 == 0:
            log(f"[D2a] concept {c}/{C} ({(time.time()-t0a)/60:.1f}m)")

    # per-concept mean lift from L (answerable cells)
    with np.errstate(invalid="ignore"):
        lift_full = np.nanmean(L_full, axis=0)                    # (C,)
        lift_tail = np.nanmean(L_tail, axis=0)
    valid = ~np.isnan(lift_full)
    b1, b2 = np.percentile(member_count[valid], [33.333, 66.667])
    bins = {"fine": member_count <= b1, "medium": (member_count > b1) & (member_count <= b2),
            "broad": member_count > b2}

    def binstat(m):
        mm = m & valid
        return {"n_concepts": int(mm.sum()),
                "median_member_count": float(np.median(member_count[mm])),
                "mean_lift_full@10": float(np.nanmean(lift_full[mm])),
                "mean_lift_tail@10": float(np.nanmean(lift_tail[mm])),
                "mean_member_AUC": float(np.nanmean(member_auc[mm])),
                "mean_delta_pop_spearman": float(np.nanmean(delta_pop_spear[mm])),
                "mean_head_minus_tail_delta": float(np.nanmean(head_minus_tail_delta[mm]))}

    d2 = {"tercile_bounds_membercount": [float(b1), float(b2)],
          "n_concepts_scored": int(valid.sum()),
          "direction_quality_note": "member_AUC = does the cold unit-positive fold delta rank concept "
                                    "members above pop-matched non-members (0.5=noise, 1=perfect direction)",
          "bins": {k: binstat(m) for k, m in bins.items()},
          "spearman_memberAUC_vs_membercount": spearman(member_count[valid], member_auc[valid]),
          "spearman_lift_full_vs_memberAUC": spearman(member_auc[valid], lift_full[valid]),
          "spearman_lift_full_vs_membercount": spearman(member_count[valid], lift_full[valid])}

    # ---- (b) THRESHOLD SWEEP: mean per-answer lift over answerable cells, concepts >= threshold ----
    sweep = {}
    for thr in (0, 30, 100, 300, 1000):
        m = valid & (member_count >= thr)
        # cell-weighted (per-answer) mean lift over all answerable cells of qualifying concepts
        cell_full = np.nanmean(L_full[:, m]) if m.any() else float("nan")
        cell_tail = np.nanmean(L_tail[:, m]) if m.any() else float("nan")
        sweep[f">={thr}"] = {"n_concepts": int(m.sum()),
                             "cellmean_lift_full@10": float(cell_full),
                             "cellmean_lift_tail@10": float(cell_tail),
                             "conceptmean_lift_full@10": float(np.nanmean(lift_full[m])) if m.any() else None}
    d2["min_member_threshold_sweep"] = sweep

    # ---- G2 canary confirmation ----
    fine_neg = d2["bins"]["fine"]["mean_lift_full@10"] < 0
    d2["g2_canary_violation_per_granularity"] = {
        "fine_mean_lift_full_negative": bool(fine_neg),
        "statement": "G2 out-of-envelope canary ('one true answer helps every channel') is VIOLATED at "
                     "the per-granularity level: fine concepts give NEGATIVE mean single-answer lift, "
                     "even though the aggregate concept channel passed."}

    fb = d2["bins"]["fine"]; bb = d2["bins"]["broad"]
    log(f"[D2] fine lift {fb['mean_lift_full@10']:+.4f} AUC {fb['mean_member_AUC']:.3f} "
        f"popcorr {fb['mean_delta_pop_spearman']:+.3f} | broad lift {bb['mean_lift_full@10']:+.4f} "
        f"AUC {bb['mean_member_AUC']:.3f} popcorr {bb['mean_delta_pop_spearman']:+.3f}")
    log(f"[D2] spearman(memberAUC,memcount)={d2['spearman_memberAUC_vs_membercount']:+.3f} "
        f"spearman(lift,memberAUC)={d2['spearman_lift_full_vs_memberAUC']:+.3f}")
    log(f"[D2] threshold sweep cellmean full: " + " ".join(
        f"{k}={v['cellmean_lift_full@10']:+.4f}(n{v['n_concepts']})" for k, v in sweep.items()))
    return d2, member_count


# ============================================================================= D3
def diagnostic3(rung, rows, sh, ctx, L_full, member_count, args):
    log("=== DIAGNOSTIC 3: utility-based concept selection ===")
    Vb = sh["conc_value_signed"]; AF = sh["conc_answerable"] & sh["conc_fold_signed"]
    rows_arr = np.asarray(rows)
    # ---- utility-selection per-user orders: answerable concepts ranked by marginal held-out lift ----
    U_orders = []
    picked_counts = []                                            # member-count of top-8 picked
    for j, r in enumerate(rows):
        elig = np.flatnonzero(AF[r])
        if len(elig) == 0:
            U_orders.append(np.empty(0, np.int64)); continue
        util = L_full[j, elig]
        # NaN-safe: unanswerable cells are already excluded; rank by descending marginal utility
        order = elig[np.argsort(-np.nan_to_num(util, nan=-1e9))]
        U_orders.append(order)
        picked_counts.extend(member_count[order[:8]].tolist())
    ev_u = walk_oracle_select_concept(rung, rows, U_orders, Vb, budgets=INT_BUDGETS)
    sc_u = score_snapshots(rung, rows, ev_u, "concept", budgets=INT_BUDGETS)

    # ---- K arm: |SEL|-selection (reuse machinery) ----
    K_orders = []
    for r in rows:
        elig = np.flatnonzero(AF[r])
        K_orders.append(elig[np.argsort(-np.abs(Vb[r, elig]))])
    ev_k = walk_oracle_select_concept(rung, rows, K_orders, Vb, budgets=INT_BUDGETS)
    sc_k = score_snapshots(rung, rows, ev_k, "concept", budgets=INT_BUDGETS)

    # ---- polarization bank (B static) comparator ----
    _, std_v = train_concept_stats(ctx, sh["_Mm_cache"], sh["_pexp_cache"], smoke=args.smoke)
    conc_order = np.argsort(-std_v)
    bmodel = Behavioral(sh, sh["lvl_lookup"])
    ev_p = walk_static_concept(rung, rows, bmodel, conc_order, budgets=INT_BUDGETS)
    sc_p = score_snapshots(rung, rows, ev_p, "concept", budgets=INT_BUDGETS)

    def strip(sc):
        return {q: {"full@10": sc[q]["full@10"], "tail@10": sc[q]["tail@10"],
                    "full@100": sc[q]["full@100"], "mean_answered": sc[q]["mean_answered"]}
                for q in INT_BUDGETS}

    def dist(vals):
        v = np.asarray(vals, np.float64)
        return {"n": int(len(v)), "median": float(np.median(v)), "mean": float(v.mean()),
                "p33": float(np.percentile(v, 33)), "p66": float(np.percentile(v, 66))}

    d3 = {"budgets": list(INT_BUDGETS),
          "utility_selection": strip(sc_u), "sel_selection_K": strip(sc_k),
          "polarization_bank_B": strip(sc_p),
          "picked_granularity_top8": dist(picked_counts),
          "note": "utility_selection ranks each user's answerable concepts by their held-out marginal "
                  "NDCG@10 utility (PRIVILEGED, myopic-from-intercept), folds honest B answers. "
                  "Compare to reference U (utility-ANSWER oracle) from answer_contrast (~0.282 @q8)."}
    log(f"[D3 utility-sel] " + " ".join(f"q{q}={sc_u[q]['full@10']:.4f}/{sc_u[q]['tail@10']:.4f}"
                                        for q in INT_BUDGETS))
    log(f"[D3 K |SEL|-sel] " + " ".join(f"q{q}={sc_k[q]['full@10']:.4f}" for q in INT_BUDGETS))
    log(f"[D3 polar bank ] " + " ".join(f"q{q}={sc_p[q]['full@10']:.4f}" for q in INT_BUDGETS))
    log(f"[D3] picked-concept member-count median {d3['picked_granularity_top8']['median']:.0f}")
    return d3


# ============================================================================= T4
def compute_K_curve(rung, rows, sh, budgets):
    """Oracle CONCEPT selection K: each user asked their top-|behavioral v| answerable concepts."""
    Vb = sh["conc_value_signed"]; AF = sh["conc_answerable"] & sh["conc_fold_signed"]
    K_orders = []
    for r in rows:
        elig = np.flatnonzero(AF[r])
        K_orders.append(elig[np.argsort(-np.abs(Vb[r, elig]))])
    ev = walk_oracle_select_concept(rung, rows, K_orders, Vb, budgets=budgets)
    return score_snapshots(rung, rows, ev, "concept", budgets=budgets)


def diagnostic4(rung, rows, sh, ctx, args):
    """T4 RULER SENSITIVITY CEILING: fold N of each user's OWN best (highest-rated) fold-in items --
    the strongest honest short item signal ('your N favorite films'). vs intercept, vs the full-profile
    fold ceiling (fold ALL fold-in items), side by side with oracle CONCEPT selection K."""
    log("=== TEST 4: ruler sensitivity ceiling (oracle item selection) ===")
    budgets = (0, 1, 2, 4, 8, 16)
    lvl = sh["lvl_lookup"]
    poprank = np.empty(ctx.ni, np.int64); poprank[sh["order_pop"]] = np.arange(ctx.ni)
    # per-user favorite order: rating level DESC, tie-break popularity DESC (recognizable favorites)
    fav_order = []
    for r in rows:
        d = lvl[r]
        its = np.array(list(d.keys()), np.int64)
        if len(its) == 0:
            fav_order.append(np.empty(0, np.int64)); continue
        lv = np.array([d[int(i)] for i in its], np.int64)
        fav_order.append(its[np.lexsort((poprank[its], -lv))])       # primary -lv, tie poprank asc(=pop desc)
    qmax = max(budgets)
    ev = {q: None for q in budgets}
    if 0 in budgets:
        ev[0] = [(np.empty(0, np.int64), np.empty(0, np.int64)) for _ in rows]
    seqs = [[] for _ in rows]
    for step in range(1, qmax + 1):
        for j in range(len(rows)):
            o = fav_order[j]
            if step - 1 < len(o):
                i = int(o[step - 1]); seqs[j].append((i, lvl[rows[j]][i]))
        if step in budgets:
            ev[step] = [(np.asarray([s for s, _ in it], np.int64),
                         np.asarray([l for _, l in it], np.int64)) for it in seqs]
    sc = score_snapshots(rung, rows, ev, "item", budgets=budgets)
    # full-profile fold ceiling: fold ALL fold-in items
    allseq = [(np.asarray(list(lvl[r].keys()), np.int64),
               np.asarray(list(lvl[r].values()), np.int64)) for r in rows]
    acc_all, _ = score_users_multi(rung, allseq, [[] for _ in rows], rows)
    ceiling = {"full@10": float(np.nanmean(acc_all["full@10"])),
               "tail@10": float(np.nanmean(acc_all["tail@10"])),
               "full@100": float(np.nanmean(acc_all["full@100"])),
               "mean_items": float(np.mean([len(s[0]) for s in allseq]))}
    scK = compute_K_curve(rung, rows, sh, budgets)
    strip = lambda s: {q: {"full@10": s[q]["full@10"], "tail@10": s[q]["tail@10"],
                           "full@100": s[q]["full@100"], "mean_answered": s[q]["mean_answered"]}
                       for q in budgets}
    t4 = {"budgets": list(budgets),
          "oracle_item_selection": strip(sc),
          "oracle_concept_selection_K": strip(scK),
          "full_profile_fold_ceiling": ceiling,
          "note": "oracle item selection = top-N highest-rated fold-in items ('your N favorite films'), "
                  "honest (fold-in only, te-excluded). Fair oracle-vs-oracle vs K (top-|SEL| concepts)."}
    log(f"[T4 oracle-item ] " + " ".join(f"N{q}={sc[q]['full@10']:.4f}/{sc[q]['tail@10']:.4f}"
                                         for q in budgets))
    log(f"[T4 oracle-K    ] " + " ".join(f"N{q}={scK[q]['full@10']:.4f}" for q in budgets))
    log(f"[T4] full-profile ceiling {ceiling['full@10']:.4f}/{ceiling['tail@10']:.4f} "
        f"(mean {ceiling['mean_items']:.0f} items)")
    return t4


# ============================================================================= T5
def diagnostic5(rung, rows, sh, ctx, args, n_per_tercile=40, max_users=2500):
    """T5 CONCEPT-FOLD vs MEMBER-ITEM-FOLD: for sampled concepts across granularity terciles, compare
    per answerable user (a) folding the CONCEPT via signed C-lite vs (b) folding the concept's rated
    MEMBER ITEMS directly. If (b) >> (a), the concept fold is LOSSY (operator/training gap)."""
    log("=== TEST 5: concept-fold vs member-item-fold ===")
    Vb = sh["conc_value_signed"]; AF = sh["conc_answerable"] & sh["conc_fold_signed"]
    lvl = sh["lvl_lookup"]
    C = len(sh["tags"])
    member_count = np.array([len(sh["members"][t]) for t in sh["tags"]], np.int64)
    rows_arr = np.asarray(rows); row_pos = {r: j for j, r in enumerate(rows)}
    empty = (np.empty(0, np.int64), np.empty(0, np.int64))
    f0, t0, _ = score_users(rung, [empty] * len(rows), [[] for _ in rows], rows)
    b1, b2 = np.percentile(member_count, [33.333, 66.667])
    terciles = {"fine": np.flatnonzero(member_count <= b1),
                "medium": np.flatnonzero((member_count > b1) & (member_count <= b2)),
                "broad": np.flatnonzero(member_count > b2)}
    rng = np.random.default_rng(4242)
    t5 = {"tercile_bounds_membercount": [float(b1), float(b2)],
          "n_concepts_per_tercile": n_per_tercile, "max_users_per_concept": max_users, "bins": {}}
    memberset = {c: set(sh["members"][sh["tags"][c]].tolist()) for c in range(C)}
    for name, pool in terciles.items():
        csamp = rng.choice(pool, size=min(n_per_tercile, len(pool)), replace=False)
        la, lb, ratios, memfold_n = [], [], [], []
        for c in csamp:
            c = int(c)
            rc = rows_arr[AF[rows_arr, c]]
            if len(rc) > max_users:
                rc = rng.choice(rc, size=max_users, replace=False)
            if len(rc) < 10:
                continue
            idx = np.array([row_pos[int(r)] for r in rc])
            # (a) concept fold via C-lite
            conc = [[(c, float(Vb[int(r), c]))] for r in rc]
            fa, _, _ = score_users(rung, [empty] * len(rc), conc, list(rc))
            # (b) member-item fold: the user's RATED member items (fold-in), real ratings
            mseq = []
            for r in rc:
                d = lvl[int(r)]
                mi = [i for i in d if i in memberset[c]]
                mseq.append((np.asarray(mi, np.int64), np.asarray([d[i] for i in mi], np.int64)))
            fb, _, _ = score_users(rung, mseq, [[] for _ in rc], list(rc))
            lift_a = float(np.nanmean(fa - f0[idx])); lift_b = float(np.nanmean(fb - f0[idx]))
            la.append(lift_a); lb.append(lift_b)
            ratios.append(lift_b / lift_a if abs(lift_a) > 1e-6 else float("nan"))
            memfold_n.append(float(np.mean([len(s[0]) for s in mseq])))
        t5["bins"][name] = {"n_concepts": len(la),
                            "median_member_count": float(np.median(member_count[csamp])),
                            "mean_concept_fold_lift@10": float(np.mean(la)),
                            "mean_member_item_fold_lift@10": float(np.mean(lb)),
                            "mean_lift_ratio_member_over_concept": float(np.nanmean(ratios)),
                            "mean_member_items_folded": float(np.mean(memfold_n))}
        b = t5["bins"][name]
        log(f"[T5 {name:>6}] concept-fold {b['mean_concept_fold_lift@10']:+.4f} vs member-item-fold "
            f"{b['mean_member_item_fold_lift@10']:+.4f} (ratio {b['mean_lift_ratio_member_over_concept']:.2f}, "
            f"{b['mean_member_items_folded']:.1f} items)")
    return t5


# ============================================================================= T7
def _score_restricted(rung, rows, item_seqs, conc_lists, topN_mask, ctx):
    """Score a belief against a top-N popularity-restricted candidate set. Held-out targets outside
    top-N are dropped (rank-within-shortlist). Returns (full@10, tail@10, coverage) mean scalars."""
    import metrics as M
    Z = rung.z_batch(item_seqs, conc_lists)
    n = len(rows)
    full = np.full(n, np.nan); tail = np.full(n, np.nan); cov = np.full(n, np.nan)
    head = ctx.head_mask
    for st in range(0, n, 500):
        ch = rows[st:st + 500]
        S = (Z[st:st + len(ch)] @ ctx.Wd.T + ctx.bd).numpy().astype(np.float32)
        mask = ctx.va_tr[ch].copy(); te = ctx.va_te[ch].tolil()
        for j in range(len(ch)):
            s = item_seqs[st + j][0]
            if len(s):
                extra = sparse.csr_matrix((np.ones(len(s), np.float32), (np.zeros(len(s)), s)),
                                          shape=(1, ctx.ni))
                mask[j] = mask[j] + extra
                for i in np.asarray(s).tolist():
                    te[j, i] = 0
        mask.data[:] = 1.0; te = te.tocsr(); te.eliminate_zeros()
        te_orig_nnz = np.asarray(te.getnnz(axis=1)).ravel()
        # restrict candidates + targets to top-N popular
        S = S.copy(); S[:, ~topN_mask] = -np.inf
        te_r = te.multiply(topN_mask[None, :]).tocsr(); te_r.eliminate_zeros()
        S[mask.tocsr().nonzero()] = -np.inf
        keep = np.asarray(te_r.getnnz(axis=1)).ravel() > 0
        if keep.any():
            full[st:st + len(ch)][keep] = M.NDCG_binary_at_k_batch(S[keep], te_r[keep], k=10)
        cov[st:st + len(ch)] = np.where(te_orig_nnz > 0,
                                        np.asarray(te_r.getnnz(axis=1)).ravel() / np.maximum(te_orig_nnz, 1),
                                        np.nan)
        tail_row = (~head).astype("float32")[None, :]
        te_t = te_r.multiply(tail_row).tocsr(); te_t.eliminate_zeros()
        tkeep = np.asarray(te_t.getnnz(axis=1)).ravel() > 0
        if tkeep.any():
            St = S.copy(); St[:, head] = -np.inf
            tail[st:st + len(ch)][tkeep] = M.NDCG_binary_at_k_batch(St[tkeep], te_t[tkeep], k=10)
    return float(np.nanmean(full)), float(np.nanmean(tail)), float(np.nanmean(cov))


def diagnostic7(rung, rows, sh, ctx, args):
    """T7 CANDIDATE-CATALOG RESTRICTION SWEEP (Krichene-Rendle fixed popularity-top-N; rank-within-
    shortlist). Re-score concept-ask B / item-ask B / oracle-concept K interview beliefs (q0 vs q8)
    against top-N most-rated candidate sets. Does the elicitation delta grow (ruler was the problem)
    or shrink (popb prior strengthens) as the catalog shrinks toward ML-1M scale?"""
    log("=== TEST 7: candidate-catalog restriction sweep ===")
    Ns = (ctx.ni, 8000, 5000, 3706, 2000)
    bmodel = Behavioral(sh, sh["lvl_lookup"])
    _, std_v = train_concept_stats(ctx, sh["_Mm_cache"], sh["_pexp_cache"], smoke=args.smoke)
    conc_order = np.argsort(-std_v)
    Vb = sh["conc_value_signed"]; AF = sh["conc_answerable"] & sh["conc_fold_signed"]
    K_orders = [np.flatnonzero(AF[r])[np.argsort(-np.abs(Vb[r, np.flatnonzero(AF[r])]))] for r in rows]
    empty_i = [(np.empty(0, np.int64), np.empty(0, np.int64)) for _ in rows]
    budg = (0, 8)
    # per-arm beliefs at q0 and q8
    ev_item = walk_static_item(rung, rows, bmodel, sh["order_pop"], budgets=budg)
    ev_conc = walk_static_concept(rung, rows, bmodel, conc_order, budgets=budg)
    ev_K = walk_oracle_select_concept(rung, rows, K_orders, Vb, budgets=budg)
    arms = {"item_ask_B": ("item", ev_item), "concept_ask_B": ("concept", ev_conc),
            "oracle_concept_K": ("concept", ev_K)}
    t7 = {"Ns": list(Ns), "restriction": "fixed popularity top-N (Krichene-Rendle 2020); "
          "rank-within-shortlist; held-out targets outside top-N dropped", "arms": {}}
    for aname, (chan, ev) in arms.items():
        t7["arms"][aname] = {}
        for N in Ns:
            topN = np.zeros(ctx.ni, bool); topN[sh["order_pop"][:N]] = True
            res = {}
            for q in budg:
                snap = ev[q]
                if chan == "item":
                    iseq = snap; clist = [[] for _ in rows]
                else:
                    iseq = empty_i; clist = snap
                f, t, cov = _score_restricted(rung, rows, iseq, clist, topN, ctx)
                res[f"q{q}"] = {"full@10": f, "tail@10": t}
                res.setdefault("coverage", cov)
            res["delta_full@10"] = res["q8"]["full@10"] - res["q0"]["full@10"]
            res["delta_tail@10"] = res["q8"]["tail@10"] - res["q0"]["tail@10"]
            t7["arms"][aname][f"N{N}"] = res
        log(f"[T7 {aname:>16}] " + " ".join(
            f"N{N}:d_full{t7['arms'][aname][f'N{N}']['delta_full@10']:+.4f}"
            f"/d_tail{t7['arms'][aname][f'N{N}']['delta_tail@10']:+.4f}"
            f"(cov{t7['arms'][aname][f'N{N}']['coverage']:.2f})" for N in Ns))
    return t7


# ============================================================================= T6
def diagnostic6(rung, rows, sh, ctx, L_full, d3, args):
    """T6 THE MEH HYPOTHESIS. (a) band distribution of the generic polarization bank vs oracle K;
    (b) per-band per-answer NDCG delta (is meh ~ unanswered?); (c) polarized-only concept-ask curve
    (skip meh) vs generic vs K; (d) coverage: per-user fraction meh vs polarized."""
    log("=== TEST 6: the MEH hypothesis ===")
    from signed_answers import BAND_LIKE, BAND_MEH, BAND_DISLIKE, BAND_REFUSE
    Vb = sh["conc_value_signed"]; B = sh["conc_band_signed"]
    ansB = sh["conc_answerable"]; foldB = sh["conc_fold_signed"]
    C = len(sh["tags"]); rows_arr = np.asarray(rows)
    _, std_v = train_concept_stats(ctx, sh["_Mm_cache"], sh["_pexp_cache"], smoke=args.smoke)
    conc_order = np.argsort(-std_v)
    N = 8

    def band_frac(asked_cells):
        """asked_cells: list of (r, c). Categorize into folded like/meh/dislike + burned."""
        cnt = {"like": 0, "meh": 0, "dislike": 0, "burned_refuse_or_unanswerable": 0}
        for r, c in asked_cells:
            if not (ansB[r, c] and foldB[r, c]):
                cnt["burned_refuse_or_unanswerable"] += 1
            elif B[r, c] == BAND_LIKE:
                cnt["like"] += 1
            elif B[r, c] == BAND_MEH:
                cnt["meh"] += 1
            elif B[r, c] == BAND_DISLIKE:
                cnt["dislike"] += 1
        tot = max(sum(cnt.values()), 1)
        return {k: v / tot for k, v in cnt.items()}, cnt

    # (a) band distribution -- generic polarization bank (first-8 concepts) and oracle K (top-8 |v|)
    polar_cells = [(r, int(conc_order[k])) for r in rows for k in range(N)]
    K_cells = []
    for r in rows:
        elig = np.flatnonzero(ansB[r] & foldB[r])
        top = elig[np.argsort(-np.abs(Vb[r, elig]))][:N]
        K_cells.extend((r, int(c)) for c in top)
    pf, pc = band_frac(polar_cells); kf, kc = band_frac(K_cells)

    # (b) per-band per-answer NDCG delta from L (answerable&fold cells)
    B_rows = B[rows_arr]
    perband = {}
    for name, bid in (("like", BAND_LIKE), ("meh", BAND_MEH), ("dislike", BAND_DISLIKE)):
        m = (B_rows == bid) & ~np.isnan(L_full)
        perband[name] = {"n_cells": int(m.sum()),
                         "mean_delta_full@10": float(np.nanmean(L_full[m])) if m.any() else None}

    # (c) polarized-only concept-ask: skip meh, ask down the bank to the next polarized concept
    polar_orders = []
    order_set = conc_order.tolist()
    for r in rows:
        pol = (ansB[r] & foldB[r] & ((B[r] == BAND_LIKE) | (B[r] == BAND_DISLIKE)))
        polar_orders.append(np.array([c for c in order_set if pol[c]], np.int64))
    ev_pol = walk_oracle_select_concept(rung, rows, polar_orders, Vb, budgets=INT_BUDGETS)
    sc_pol = score_snapshots(rung, rows, ev_pol, "concept", budgets=INT_BUDGETS)
    strip = lambda s: {q: {"full@10": s[q]["full@10"], "tail@10": s[q]["tail@10"],
                           "full@100": s[q]["full@100"], "mean_answered": s[q]["mean_answered"]}
                       for q in INT_BUDGETS}

    # (d) coverage: per-user band fractions over all C concepts + among answerable
    frac_meh_all = []; frac_pol_all = []; frac_meh_ans = []; frac_pol_ans = []
    for r in rows:
        br = B[r]; an = ansB[r] & foldB[r]
        pol = an & ((br == BAND_LIKE) | (br == BAND_DISLIKE))
        meh = an & (br == BAND_MEH)
        frac_meh_all.append(meh.sum() / C); frac_pol_all.append(pol.sum() / C)
        na = max(an.sum(), 1)
        frac_meh_ans.append(meh.sum() / na); frac_pol_ans.append(pol.sum() / na)

    def med_mean(v):
        v = np.asarray(v); return {"median": float(np.median(v)), "mean": float(v.mean())}

    t6 = {"asked_first_k": N,
          "a_band_distribution": {"generic_polarization_bank": {"fractions": pf, "counts": pc},
                                  "oracle_selection_K": {"fractions": kf, "counts": kc}},
          "b_per_band_per_answer_delta": perband,
          "c_polarized_only_curve": {"polarized_only": strip(sc_pol),
                                     "generic_polarization_bank": d3["polarization_bank_B"],
                                     "oracle_selection_K": d3["sel_selection_K"]},
          "d_coverage": {"note": "per-user fractions; polarized = like|dislike, over answerable&fold",
                         "frac_meh_over_all_1031": med_mean(frac_meh_all),
                         "frac_polarized_over_all_1031": med_mean(frac_pol_all),
                         "frac_meh_among_answerable": med_mean(frac_meh_ans),
                         "frac_polarized_among_answerable": med_mean(frac_pol_ans)}}
    log(f"[T6a] generic bank bands like/meh/dislike/burned = "
        f"{pf['like']:.2f}/{pf['meh']:.2f}/{pf['dislike']:.2f}/{pf['burned_refuse_or_unanswerable']:.2f}"
        f" | K = {kf['like']:.2f}/{kf['meh']:.2f}/{kf['dislike']:.2f}/{kf['burned_refuse_or_unanswerable']:.2f}")
    log(f"[T6b] per-answer delta like={perband['like']['mean_delta_full@10']:+.4f} "
        f"meh={perband['meh']['mean_delta_full@10']:+.4f} "
        f"dislike={perband['dislike']['mean_delta_full@10']:+.4f}")
    log(f"[T6c] polarized-only " + " ".join(f"q{q}={sc_pol[q]['full@10']:.4f}" for q in INT_BUDGETS))
    log(f"[T6d] median-user frac meh(answerable)={t6['d_coverage']['frac_meh_among_answerable']['median']:.2f} "
        f"polarized(answerable)={t6['d_coverage']['frac_polarized_among_answerable']['median']:.2f}")
    return t6


# ============================================================================= main
def _flush(out, ctx, args):
    outdir = ctx.outdir if args.smoke else OUTDIR
    os.makedirs(outdir, exist_ok=True)
    json.dump(out, open(os.path.join(outdir, args.out), "w"), indent=2, default=float)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--diag", default="all")                     # 1 | 23 | 45 | all
    ap.add_argument("--snapshot", default=PA.SNAP_DEFAULT)
    ap.add_argument("--sclite_ckpt", default=os.path.join(_ROOT, ".cache", "instrument",
                                                          "cfold_signed_best.pt"))
    ap.add_argument("--out", default="answer_decomp_diag.json")
    ap.add_argument("--full_threads", action="store_true")
    args = ap.parse_args()
    t00 = time.time()
    ctx = build_smoke_ctx() if args.smoke else build_real_ctx(args.snapshot)
    sh = build_shared(ctx)
    if args.smoke and not sh.get("signed_available"):
        rng = np.random.RandomState(4); C = sh["d_c"].shape[0]
        sh["conc_value_signed"] = (rng.rand(ctx.n, C).astype(np.float32) * 2 - 1)
        sh["conc_fold_signed"] = rng.rand(ctx.n, C) > 0.15
        sh["signed_available"] = True
    assert sh.get("signed_available"), "signed prereg cache required"
    rows = [r for r in range(ctx.n) if ctx.va_te[r].nnz > 0]

    # ---- leak check (HARD RULE 5) ----
    leaks = sum(1 for r in rows
                if len(np.intersect1d(ctx.allb[r][0], ctx.va_te[r].indices)) > 0)
    assert leaks == 0, f"LEAK: {leaks} users have fold-in/held-out overlap"
    log(f"[leak] fold-in INTERSECT held-out empty for all {len(rows)} users: PASS")

    rung = load_sclite_rung(ctx, sh, args)
    # cache Mm/pexp for D3 polarization bank
    Mm = sparse.csr_matrix((np.ones(sum(len(sh["members"][t]) for t in sh["tags"]), np.float32),
                            (np.concatenate([sh["members"][t] for t in sh["tags"]]),
                             np.concatenate([np.full(len(sh["members"][t]), i)
                                             for i, t in enumerate(sh["tags"])]))),
                           shape=(ctx.ni, len(sh["tags"])))
    sh["_Mm_cache"] = Mm
    sh["_pexp_cache"] = (ctx.cnt @ np.asarray(Mm.todense())) / max(ctx.cnt.sum(), 1e-9)

    # intercept control
    empty = (np.empty(0, np.int64), np.empty(0, np.int64))
    f0, t0, _ = score_users(rung, [empty] * len(rows), [[] for _ in rows], rows)
    intercept = {"full@10": float(np.nanmean(f0)), "tail@10": float(np.nanmean(t0))}
    snap_ok = (abs(intercept["full@10"] - INTERCEPT_REF[0]) < 5e-3
               and abs(intercept["tail@10"] - INTERCEPT_REF[1]) < 5e-3)
    log(f"[intercept] {intercept['full@10']:.4f}/{intercept['tail@10']:.4f} snap={snap_ok}")

    out = {"analysis": "answer_decomp_diag", "n_users": len(rows),
           "stack": "frozen t2i25_EP4 tower + signed C-lite (cfold_signed_best.pt)",
           "intercept": intercept, "intercept_ref": list(INTERCEPT_REF),
           "controls": {"q0_canonical_snap_PASS": bool(snap_ok),
                        "leak_foldin_heldout_overlap_users": int(leaks)},
           "diagnostic": "UNDERSTANDING run (not for paper)"}

    do1 = args.diag in ("1", "all")
    do45 = args.diag in ("45", "all")
    do23 = args.diag in ("23", "all")
    # cheap diagnostics first (D1, T4, T5); the ~80-min L pass (D2/D3) runs last
    if do1:
        out["D1_item_breakdown"] = diagnostic1(rung, rows, sh, ctx, args)
    if do45:
        out["T4_ruler_ceiling"] = diagnostic4(rung, rows, sh, ctx, args)
        out["T5_concept_vs_memberitem_fold"] = diagnostic5(rung, rows, sh, ctx, args)
        # flush partial results so T4/T5 are readable before the long L pass finishes
        _flush(out, ctx, args)
    if args.diag in ("7", "all"):
        out["T7_catalog_restriction"] = diagnostic7(rung, rows, sh, ctx, args)
        _flush(out, ctx, args)
    if do23:
        L_full, L_tail, _, _, nans = compute_marginal_L(rung, rows, sh, ctx)
        d2, member_count = diagnostic2(rung, rows, sh, ctx, L_full, L_tail, nans)
        out["D2_fine_concept_harm"] = d2
        d3 = diagnostic3(rung, rows, sh, ctx, L_full, member_count, args)
        out["D3_utility_selection"] = d3
        out["T6_meh_hypothesis"] = diagnostic6(rung, rows, sh, ctx, L_full, d3, args)

    out["seconds"] = round(time.time() - t00, 1)
    outdir = ctx.outdir if args.smoke else OUTDIR
    os.makedirs(outdir, exist_ok=True)
    jpath = os.path.join(outdir, args.out)
    json.dump(out, open(jpath, "w"), indent=2, default=float)
    log(f"[out] -> {jpath}  ({out['seconds']}s)")


if __name__ == "__main__":
    main()
