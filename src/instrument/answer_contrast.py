r"""answer_contrast.py -- ANSWER-MODEL CONTRAST on the strong stack (frozen i25 tower + signed C-lite).

QUESTION: is "concepts beat items" driven by the ANSWER SIMULATOR rather than the concepts? Hold the
recommender (sclite = ConceptFoldNet on the frozen t2i25_EP4 tower) and the interview QUESTION ORDER
FIXED, and vary ONLY the answer model. If the concept win survives the honest answer model and does not
collapse to a self-preferencing geometric oracle, the concepts carry it; if it only shows up under the
recommender-geometry answer, the answer simulator was load-bearing.

FIXED STATIC ORDERS (identical across every answer model -- the answer model is the ONLY variable):
  item-ask    = popularity order (sh['order_pop'])          -- the best long-interview item arm
  concept-ask = polarization order (argsort(-train std|v|)) -- the best realizable concept opener
Per-question deployment currency: every asked question burns a budget slot; unanswered burns w/o folding.

ANSWER MODELS (pluggable; a 4th middle-ground SEL+ arm 'S' drops in via ANSWER_MODELS):
  (G) GEOMETRIC   PRIVILEGED / recommender-geometry / UNCITABLE. u* = tower fold of the user's KNOWN-HALF
      profile. For each queried entity the simulated like/dislike = the sign that, when folded, moves the
      running belief CLOSER (cosine) to u*. Concept magnitude = matched |v_B| (isolates DIRECTION, the
      self-preference), same answerable set as B; item candidate levels = {like 9, dislike 0}.
  (B) BEHAVIORAL  the honest/headline answer: signed four-band SEL (NPMI watch-lift + shrunk residual
      rating) over the FOLD-IN history only (sh['conc_value_signed']); items = the real fold-in rating.
  (O) ORACLE-B    PRIVILEGED ceiling (uses held-out): the SAME signed-SEL formula computed over the user's
      FULL history (fold-in + held-out); items = the real rating over full history (answerable on held-out
      items too, credit-neutral masked).
  (S) SEL+        assembled non-circular candidate (added in --arms G,B,O,S; see build_selplus).

CONTROLS (HARD RULE 5): q0 == cold intercept (canonical-snap 0.12794/0.01923); a wrong-user shuffle on the
concept arm (must collapse toward intercept); leak check (fold-in item set INTERSECT held-out == empty).
GUARDS (coordinator): linear CKA of each model's induced answer-geometry (concept belief Z@q8) vs u*
(circularity meter -- G should be HIGH); Spearman of each imputer's per-(user,concept) value vs oracle-B.

Usage:
  python src/instrument/answer_contrast.py --smoke
  python src/instrument/answer_contrast.py [--arms G,B,O[,S]] [--full_threads]
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

from train_tower_t2 import log, reproduce_partition, NLEV
import run_battery_phaseA as PA
from run_battery_phaseA import (build_real_ctx, build_smoke_ctx, bootstrap_ci, spearman, OUTDIR, SEED)
from tradeoff_ledger import build_shared, Rung, fold_items_enc
from adaptive_concept_arms import train_concept_stats, score_users
import metrics as M

assert not hasattr(sys.modules[__name__], "load_answerer")

BUDGETS = (0, 1, 2, 4, 8)
KS = (10, 100)                                                # NDCG cutoffs reported for every arm
INTERCEPT_REF = (0.12793773315625726, 0.01923195232020529)   # canonical-snap (strategy_channel_suite)


# ============================================================================= oracle-B (full history)
def build_full_history(ctx, smoke=False):
    """Reconstruct the per-row FULL graded history (fold-in + held-out targets, real half-star levels)
    from the raw catalog -- the ORACLE-B (privileged) answer basis. Returns list[(sids, lvls)] indexed
    like ctx.allb. In smoke: fold-in UNION te (te at level 8) since raw is unavailable."""
    if smoke:
        full = []
        for r in range(ctx.n):
            s, l = ctx.allb[r]
            te = ctx.va_te[r].indices.astype(np.int64)
            full.append((np.concatenate([s, te]),
                         np.concatenate([l, np.full(len(te), 8, np.int64)])))
        return full
    import pandas as pd
    unique_uid, tr_set, vd_set, te_set, ntr, raw, show2id, usid = reproduce_partition()
    n_users_all = len(unique_uid)
    start_vd = n_users_all - 2 * 10000
    val_userIds = unique_uid[start_vd:start_vd + 10000]
    uid2row = {int(u): i for i, u in enumerate(val_userIds)}
    vdf = raw[raw["userId"].isin(set(val_userIds.tolist()))].copy()
    vdf["sid"] = vdf["movieId"].map(show2id); vdf = vdf[vdf["sid"].notna()]
    vdf["sid"] = vdf["sid"].astype(np.int64)
    vdf["lvl"] = np.clip(np.rint(vdf["rating"].values * 2).astype(np.int64) - 1, 0, NLEV - 1)
    full = [(np.empty(0, np.int64), np.empty(0, np.int64))] * ctx.n
    for uidv, g in vdf.groupby("userId"):
        r = uid2row[int(uidv)]
        full[r] = (g["sid"].values.astype(np.int64), g["lvl"].values.astype(np.int64))
    return full


def build_oracle_arrays(ctx, sh, full_hist):
    """signed_values over the FULL history -> (V, F, answerable) at (n, C). The privileged ceiling."""
    from signed_answers import load_prereg, signed_values
    prereg, item_mean = load_prereg()
    tags = sh["tags"]; members = sh["members"]
    Mm = sparse.csr_matrix((np.ones(sum(len(members[t]) for t in tags), np.float32),
                            (np.concatenate([members[t] for t in tags]),
                             np.concatenate([np.full(len(members[t]), i) for i, t in enumerate(tags)]))),
                           shape=(ctx.ni, len(tags)))
    pexp = (ctx.cnt @ np.asarray(Mm.todense())) / max(ctx.cnt.sum(), 1e-9)
    items_l = [np.asarray(s, np.int64) for s, l in full_hist]
    stars_l = [((np.asarray(l, np.float64) + 1) / 2).astype(np.float32) for s, l in full_hist]
    V, F, B, ans = signed_values(items_l, stars_l, item_mean, Mm, pexp, ctx.ni, prereg)
    return V.astype(np.float32), F, ans, [dict(zip(np.asarray(s).tolist(), np.asarray(l).tolist()))
                                          for s, l in full_hist]


# ============================================================================= linear CKA (Kornblith'19)
def linear_cka(X, Y):
    """Linear CKA between two n x d representations (columns centered). Returns scalar in [0, 1]."""
    X = np.asarray(X, np.float64); Y = np.asarray(Y, np.float64)
    X = X - X.mean(0, keepdims=True); Y = Y - Y.mean(0, keepdims=True)
    hsic_xy = np.linalg.norm(Y.T @ X, "fro") ** 2
    hsic_xx = np.linalg.norm(X.T @ X, "fro") ** 2
    hsic_yy = np.linalg.norm(Y.T @ Y, "fro") ** 2
    return float(hsic_xy / max(np.sqrt(hsic_xx * hsic_yy), 1e-12))


# ============================================================================= answer models (pluggable)
class AnswerModel:
    """Clean interface: per user-row r and queried entity, return the folded answer or None (burns).
    concept_value(r, c) -> (value, ok) : ok False = unanswerable/refuse (burns budget, no fold).
    item_value(r, i)    -> (level, ok) . Geometric models override the *walk* (need running belief)."""
    geometric = False

    def concept_value(self, r, c):
        raise NotImplementedError

    def item_value(self, r, i):
        raise NotImplementedError


class Behavioral(AnswerModel):
    """(B) signed four-band SEL over fold-in; items = real fold-in rating."""
    def __init__(self, sh, lvl_lookup):
        self.V = sh["conc_value_signed"]; self.F = sh["conc_fold_signed"]
        self.ans = sh["conc_answerable"]; self.lk = lvl_lookup

    def concept_value(self, r, c):
        if self.ans[r, c] and self.F[r, c]:
            return float(self.V[r, c]), True
        return 0.0, False

    def item_value(self, r, i):
        d = self.lk[r]
        return (float(d[i]), True) if i in d else (0.0, False)


class OracleB(AnswerModel):
    """(O) signed four-band SEL over FULL history; items = real rating over full history (privileged)."""
    def __init__(self, Vf, Ff, ansf, lvl_lookup_full):
        self.V = Vf; self.F = Ff; self.ans = ansf; self.lk = lvl_lookup_full

    def concept_value(self, r, c):
        if self.ans[r, c] and self.F[r, c]:
            return float(self.V[r, c]), True
        return 0.0, False

    def item_value(self, r, i):
        d = self.lk[r]
        return (float(d[i]), True) if i in d else (0.0, False)


class Geometric(AnswerModel):
    """(G) privileged recommender-geometry. Concepts: matched |v_B| magnitude, geometric SIGN toward u*,
    B's answerable set. Items: candidate levels {9, 0}, B's answerable set. The walk is sequential."""
    geometric = True

    def __init__(self, sh, lvl_lookup):
        self.V = sh["conc_value_signed"]; self.F = sh["conc_fold_signed"]
        self.ans = sh["conc_answerable"]; self.lk = lvl_lookup

    def concept_gate(self, r, c):
        return bool(self.ans[r, c] and self.F[r, c])

    def concept_mag(self, r, c):
        return abs(float(self.V[r, c]))

    def item_gate(self, r, i):
        return i in self.lk[r]


class UtilityOracle(AnswerModel):
    """(U) PRIVILEGED true answer-channel CEILING. For each asked (answerable) question, greedily pick the
    answer -- concept sign in {+1, -1, skip}, item level in {9, 0, skip} -- that MAXIMIZES the user's
    held-out NDCG@10 after folding. This is the ceiling of the ANSWER, distinct from oracle-B (which only
    ceilings the SEL FORMULA). Peeks at held-out targets -> UNCITABLE; the walk is sequential."""
    utility = True

    def __init__(self, sh, lvl_lookup):
        self.ans = sh["conc_answerable"]; self.F = sh["conc_fold_signed"]; self.lk = lvl_lookup

    def concept_gate(self, r, c):
        return bool(self.ans[r, c] and self.F[r, c])

    def item_gate(self, r, i):
        return i in self.lk[r]


# ============================================================================= interview walks
def walk_static_concept(rung, rows, model, order, budgets=BUDGETS):
    """Fixed public concept order; model.concept_value gives the fold. Snapshots evidence at each q."""
    folded = [[] for _ in rows]
    qmax = max(budgets)
    ev = {q: None for q in budgets}
    if 0 in budgets:
        ev[0] = [list(f) for f in folded]
    for step in range(1, qmax + 1):
        c = int(order[step - 1])
        for j, r in enumerate(rows):
            v, ok = model.concept_value(r, c)
            if ok:
                folded[j].append((c, v))
        if step in budgets:
            ev[step] = [list(f) for f in folded]
    return ev


def walk_oracle_select_concept(rung, rows, per_user_orders, Vb, budgets=BUDGETS):
    """(K) ORACLE-CONCEPT-SELECTION: each user asked their OWN top-|value| concepts (privileged question
    SELECTION), folded with the honest behavioral B value. Isolates FOLD health from the SELECTION
    problem -- the upper bound on WHICH concepts to ask, honest answers. per_user_orders aligned to rows."""
    folded = [[] for _ in rows]
    qmax = max(budgets)
    ev = {q: None for q in budgets}
    if 0 in budgets:
        ev[0] = [list(f) for f in folded]
    for step in range(1, qmax + 1):
        for j, r in enumerate(rows):
            order = per_user_orders[j]
            if step - 1 < len(order):
                c = int(order[step - 1])
                folded[j].append((c, float(Vb[r, c])))
        if step in budgets:
            ev[step] = [list(f) for f in folded]
    return ev


def walk_static_item(rung, rows, model, order, budgets=BUDGETS):
    """Fixed public item order; model.item_value gives the fold. Snapshots evidence at each q."""
    items = [[] for _ in rows]
    qmax = max(budgets)
    ev = {q: None for q in budgets}
    if 0 in budgets:
        ev[0] = [(np.empty(0, np.int64), np.empty(0, np.int64)) for _ in rows]
    for step in range(1, qmax + 1):
        i = int(order[step - 1])
        for j, r in enumerate(rows):
            lv, ok = model.item_value(r, i)
            if ok:
                items[j].append((i, int(lv)))
        if step in budgets:
            ev[step] = [(np.asarray([s for s, _ in it], np.int64),
                         np.asarray([l for _, l in it], np.int64)) for it in items]
    return ev


def walk_geo_concept(rung, rows, model, order, u_star, budgets=BUDGETS):
    """Geometric concept walk: at each fixed-order concept, pick the SIGN whose folded belief is closer
    (cosine) to u*. Magnitude matched to |v_B|. Two C-lite folds per step (plus/minus)."""
    empty_i = [(np.empty(0, np.int64), np.empty(0, np.int64)) for _ in rows]
    us = u_star / np.maximum(np.linalg.norm(u_star, axis=1, keepdims=True), 1e-12)
    folded = [[] for _ in rows]
    qmax = max(budgets)
    ev = {q: None for q in budgets}
    if 0 in budgets:
        ev[0] = [list(f) for f in folded]
    for step in range(1, qmax + 1):
        c = int(order[step - 1])
        gate = np.array([model.concept_gate(r, c) for r in rows])
        mag = np.array([model.concept_mag(r, c) for r in rows], np.float32)
        plus = [f + ([(c, float(mag[j]))] if gate[j] else []) for j, f in enumerate(folded)]
        minus = [f + ([(c, float(-mag[j]))] if gate[j] else []) for j, f in enumerate(folded)]
        zp = rung.z_batch(empty_i, plus).numpy()
        zm = rung.z_batch(empty_i, minus).numpy()
        cp = (zp / np.maximum(np.linalg.norm(zp, axis=1, keepdims=True), 1e-12) * us).sum(1)
        cm = (zm / np.maximum(np.linalg.norm(zm, axis=1, keepdims=True), 1e-12) * us).sum(1)
        for j in range(len(rows)):
            if gate[j]:
                s = float(mag[j]) if cp[j] >= cm[j] else float(-mag[j])
                folded[j].append((c, s))
        if step in budgets:
            ev[step] = [list(f) for f in folded]
    return ev


def walk_geo_item(rung, rows, model, order, u_star, budgets=BUDGETS):
    """Geometric item walk: candidate levels {9, 0}, pick the one whose folded belief is closer to u*."""
    us = u_star / np.maximum(np.linalg.norm(u_star, axis=1, keepdims=True), 1e-12)
    items = [[] for _ in rows]
    qmax = max(budgets)
    ev = {q: None for q in budgets}
    if 0 in budgets:
        ev[0] = [(np.empty(0, np.int64), np.empty(0, np.int64)) for _ in rows]
    for step in range(1, qmax + 1):
        i = int(order[step - 1])
        gate = np.array([model.item_gate(r, i) for r in rows])
        def seqs(level):
            return [(np.asarray([s for s, _ in it] + ([i] if gate[j] else []), np.int64),
                     np.asarray([l for _, l in it] + ([level] if gate[j] else []), np.int64))
                    for j, it in enumerate(items)]
        zp = rung.z_batch(seqs(9), [[] for _ in rows]).numpy()
        zm = rung.z_batch(seqs(0), [[] for _ in rows]).numpy()
        cp = (zp / np.maximum(np.linalg.norm(zp, axis=1, keepdims=True), 1e-12) * us).sum(1)
        cm = (zm / np.maximum(np.linalg.norm(zm, axis=1, keepdims=True), 1e-12) * us).sum(1)
        for j in range(len(rows)):
            if gate[j]:
                items[j].append((i, 9 if cp[j] >= cm[j] else 0))
        if step in budgets:
            ev[step] = [(np.asarray([s for s, _ in it], np.int64),
                         np.asarray([l for _, l in it], np.int64)) for it in items]
    return ev


def ndcg_multi(S, mask_rows, te_rows, head_mask, ks=KS):
    """Per-user full/tail NDCG at several cutoffs for one score batch (credit-neutral masking applied
    upstream). Returns {full@k, tail@k -> (n,) arrays with NaN where no (tail) target}."""
    S = S.copy()
    S[mask_rows.nonzero()] = -np.inf
    n = S.shape[0]
    res = {}
    keep = np.asarray(te_rows.getnnz(axis=1)).ravel() > 0
    for k in ks:
        full = np.full(n, np.nan)
        if keep.any():
            full[keep] = M.NDCG_binary_at_k_batch(S[keep], te_rows[keep], k=k)
        res[f"full@{k}"] = full
    if head_mask is not None:
        tail_row = (~head_mask).astype("float32")[None, :]
        te_t = te_rows.multiply(tail_row).tocsr(); te_t.eliminate_zeros()
        tkeep = np.asarray(te_t.getnnz(axis=1)).ravel() > 0
        St = S.copy(); St[:, head_mask] = -np.inf
        for k in ks:
            tail = np.full(n, np.nan)
            if tkeep.any():
                tail[tkeep] = M.NDCG_binary_at_k_batch(St[tkeep], te_t[tkeep], k=k)
            res[f"tail@{k}"] = tail
    return res


def score_users_multi(rung, item_seqs, conc_lists, rows, ks=KS):
    """Rung fold -> per-user NDCG at cutoffs `ks`, credit-neutral masking of revealed ITEMS (parity with
    adaptive_concept_arms.score_users). Returns (acc dict of (n,) arrays, Z)."""
    ctx = rung.ctx
    Z = rung.z_batch(item_seqs, conc_lists)
    acc = {f"{m}@{k}": np.full(len(rows), np.nan) for k in ks for m in ("full", "tail")}
    for st in range(0, len(rows), 500):
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
        r = ndcg_multi(S, mask.tocsr(), te, ctx.head_mask, ks)
        for kk in acc:
            acc[kk][st:st + len(ch)] = r[kk]
    return acc, Z


def score_full10(rung, item_seqs, conc_lists, rows):
    """Fast selection metric for the utility oracle: per-user full NDCG@10 only."""
    f, _, _ = score_users(rung, item_seqs, conc_lists, rows)
    return f


def walk_utility_concept(rung, rows, model, order, budgets=BUDGETS):
    """Greedy true-utility concept walk: at each fixed-order answerable concept, pick sign in
    {+1, -1, skip} maximizing held-out NDCG@10 after folding. Skip = decline (no fold). PRIVILEGED."""
    empty_i = [(np.empty(0, np.int64), np.empty(0, np.int64)) for _ in rows]
    folded = [[] for _ in rows]
    qmax = max(budgets)
    ev = {q: None for q in budgets}
    if 0 in budgets:
        ev[0] = [list(f) for f in folded]
    cur = np.nan_to_num(score_full10(rung, empty_i, folded, rows), nan=-1e9)
    for step in range(1, qmax + 1):
        c = int(order[step - 1])
        gate = np.array([model.concept_gate(r, c) for r in rows])
        plus = [f + ([(c, 1.0)] if gate[j] else []) for j, f in enumerate(folded)]
        minus = [f + ([(c, -1.0)] if gate[j] else []) for j, f in enumerate(folded)]
        fp = np.nan_to_num(score_full10(rung, empty_i, plus, rows), nan=-1e9)
        fm = np.nan_to_num(score_full10(rung, empty_i, minus, rows), nan=-1e9)
        for j in range(len(rows)):
            if gate[j]:
                if fp[j] >= cur[j] and fp[j] >= fm[j]:
                    folded[j].append((c, 1.0))
                elif fm[j] >= cur[j] and fm[j] > fp[j]:
                    folded[j].append((c, -1.0))
                # else skip (folding either sign would not beat the current belief)
        cur = np.nan_to_num(score_full10(rung, empty_i, folded, rows), nan=-1e9)
        if step in budgets:
            ev[step] = [list(f) for f in folded]
    return ev


def walk_utility_item(rung, rows, model, order, budgets=BUDGETS):
    """Greedy true-utility item walk: at each fixed-order rated item, pick level in {9, 0, skip}
    maximizing held-out NDCG@10 after folding. PRIVILEGED."""
    empty_c = [[] for _ in rows]
    items = [[] for _ in rows]
    qmax = max(budgets)
    ev = {q: None for q in budgets}
    if 0 in budgets:
        ev[0] = [(np.empty(0, np.int64), np.empty(0, np.int64)) for _ in rows]

    def seqs(i, gate, level):
        return [(np.asarray([s for s, _ in it] + ([i] if (gate[j] and level is not None) else []),
                            np.int64),
                 np.asarray([l for _, l in it] + ([level] if (gate[j] and level is not None) else []),
                            np.int64)) for j, it in enumerate(items)]

    for step in range(1, qmax + 1):
        i = int(order[step - 1])
        gate = np.array([model.item_gate(r, i) for r in rows])
        cur = np.nan_to_num(score_full10(rung, seqs(i, gate, None), empty_c, rows), nan=-1e9)
        f9 = np.nan_to_num(score_full10(rung, seqs(i, gate, 9), empty_c, rows), nan=-1e9)
        f0 = np.nan_to_num(score_full10(rung, seqs(i, gate, 0), empty_c, rows), nan=-1e9)
        for j in range(len(rows)):
            if gate[j]:
                if f9[j] >= cur[j] and f9[j] >= f0[j]:
                    items[j].append((i, 9))
                elif f0[j] >= cur[j] and f0[j] > f9[j]:
                    items[j].append((i, 0))
                # else skip
        if step in budgets:
            ev[step] = [(np.asarray([s for s, _ in it], np.int64),
                         np.asarray([l for _, l in it], np.int64)) for it in items]
    return ev


def score_snapshots(rung, rows, ev, channel, budgets=BUDGETS):
    """ev[q] -> per-arm metrics (full/tail @10 and @100) + mean_answered, credit-neutral item masking."""
    empty_i = [(np.empty(0, np.int64), np.empty(0, np.int64)) for _ in rows]
    out = {}
    qmax = max(budgets)
    for q in budgets:
        snap = ev[q]
        if channel == "concept":
            item_seqs = empty_i; conc_lists = snap
            n_ans = float(np.mean([len(c) for c in conc_lists]))
        else:
            item_seqs = snap; conc_lists = [[] for _ in rows]
            n_ans = float(np.mean([len(s[0]) for s in item_seqs]))
        acc, Z = score_users_multi(rung, item_seqs, conc_lists, rows)
        out[q] = {"full@10": float(np.nanmean(acc["full@10"])), "tail@10": float(np.nanmean(acc["tail@10"])),
                  "full@100": float(np.nanmean(acc["full@100"])),
                  "tail@100": float(np.nanmean(acc["tail@100"])),
                  "mean_answered": n_ans,
                  "_fv": acc["full@10"], "_tv": acc["tail@10"], "_fv100": acc["full@100"],
                  "_Z": Z.numpy() if q == qmax else None}
    return out


# ============================================================================= main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--snapshot", default=PA.SNAP_DEFAULT)
    ap.add_argument("--arms", default="G,B,O,K,U,S,C")
    ap.add_argument("--sclite_ckpt", default=os.path.join(_ROOT, ".cache", "instrument",
                                                          "cfold_signed_best.pt"))
    ap.add_argument("--full_threads", action="store_true")
    args = ap.parse_args()
    t00 = time.time()
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    ctx = build_smoke_ctx() if args.smoke else build_real_ctx(args.snapshot)
    sh = build_shared(ctx)
    if args.smoke and not sh.get("signed_available"):
        rng = np.random.RandomState(4); C = sh["d_c"].shape[0]
        sh["conc_value_signed"] = (rng.rand(ctx.n, C).astype(np.float32) * 2 - 1)
        sh["conc_fold_signed"] = rng.rand(ctx.n, C) > 0.15
        sh["signed_available"] = True
    assert sh.get("signed_available"), "signed prereg cache required"
    rows = [r for r in range(ctx.n) if ctx.va_te[r].nnz > 0]

    # ---- leak check (HARD RULE 5): fold-in items INTERSECT held-out == empty, every user ----
    leaks = 0
    for r in rows:
        if len(np.intersect1d(ctx.allb[r][0], ctx.va_te[r].indices)) > 0:
            leaks += 1
    assert leaks == 0, f"LEAK: {leaks} users have fold-in/held-out overlap"
    log(f"[leak] fold-in INTERSECT held-out empty for all {len(rows)} users: PASS")

    # ---- rung (sclite: C-lite on the frozen tower) ----
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
    rung = Rung("sclite", ctx, sh, clite_net=net, val_source="signed")

    lvl_lookup = sh["lvl_lookup"]                                    # fold-in graded lookup (te excluded)
    # ---- fixed static orders (identical across answer models) ----
    Mm = sparse.csr_matrix((np.ones(sum(len(sh["members"][t]) for t in sh["tags"]), np.float32),
                            (np.concatenate([sh["members"][t] for t in sh["tags"]]),
                             np.concatenate([np.full(len(sh["members"][t]), i)
                                             for i, t in enumerate(sh["tags"])]))),
                           shape=(ctx.ni, len(sh["tags"])))
    pexp = (ctx.cnt @ np.asarray(Mm.todense())) / max(ctx.cnt.sum(), 1e-9)
    mean_abs, std_v = train_concept_stats(ctx, Mm, pexp, smoke=args.smoke)
    conc_order = np.argsort(-std_v)                                 # polarization (best concept opener)
    item_order = sh["order_pop"]                                    # popularity (best item arm)

    # ---- u* = tower fold of the KNOWN-HALF profile (aligned to rows) ----
    u_star = fold_items_enc(ctx.enc, [ctx.allb[r] for r in rows]).numpy()

    # ---- oracle-B full-history arrays ----
    full_hist = build_full_history(ctx, smoke=args.smoke)
    Vf, Ff, ansf, lk_full = build_oracle_arrays(ctx, sh, full_hist)
    # oracle leak-safety note: held-out items ARE answerable under O; score_users masks any answered item
    # credit-neutrally (removed from candidates AND IDCG), so no target leaks into the ranking metric.

    models = {}                                                     # built in the requested arm order
    for tag in arms:
        if tag == "B": models["B"] = Behavioral(sh, lvl_lookup)
        elif tag == "O": models["O"] = OracleB(Vf, Ff, ansf, lk_full)
        elif tag == "G": models["G"] = Geometric(sh, lvl_lookup)
        elif tag == "U": models["U"] = UtilityOracle(sh, lvl_lookup)
        elif tag == "S":
            from selplus import build_selplus_arrays, SELPlus
            Vs, Fs, anss = build_selplus_arrays(ctx, sh, Mm, pexp, smoke=args.smoke)
            models["S"] = SELPlus(Vs, Fs, anss, lvl_lookup); sh["conc_value_selplus"] = Vs
        elif tag == "C":
            from selplus import build_content_arrays, Content
            Vc, Fc, ansc = build_content_arrays(ctx, sh, Mm, pexp, smoke=args.smoke)
            models["C"] = Content(Vc, Fc, ansc, lvl_lookup); sh["conc_value_content"] = Vc
        elif tag == "K":
            models["K"] = "oracle_select"                           # sentinel; concept-only arm

    # per-user oracle concept SELECTION order (top-|behavioral v| among answerable+fold), for arm K
    Vb = sh["conc_value_signed"]; ansB = sh["conc_answerable"]; foldB = sh["conc_fold_signed"]
    K_orders = None
    if "K" in arms:
        K_orders = []
        for r in rows:
            elig = np.flatnonzero(ansB[r] & foldB[r])
            K_orders.append(elig[np.argsort(-np.abs(Vb[r, elig]))])

    results = {"item": {}, "concept": {}}
    zq8 = {}                                                        # concept belief Z@q8 per model (CKA)
    for tag, model in models.items():
        channels = ("concept",) if tag == "K" else ("concept", "item")
        for channel in channels:
            t0 = time.time()
            if tag == "K":
                ev = walk_oracle_select_concept(rung, rows, K_orders, Vb)
            elif getattr(model, "utility", False):
                ev = (walk_utility_concept(rung, rows, model, conc_order) if channel == "concept"
                      else walk_utility_item(rung, rows, model, item_order))
            elif model.geometric:
                ev = (walk_geo_concept(rung, rows, model, conc_order, u_star) if channel == "concept"
                      else walk_geo_item(rung, rows, model, item_order, u_star))
            else:
                ev = (walk_static_concept(rung, rows, model, conc_order) if channel == "concept"
                      else walk_static_item(rung, rows, model, item_order))
            sc = score_snapshots(rung, rows, ev, channel)
            if channel == "concept":
                zq8[tag] = sc[max(BUDGETS)]["_Z"]
            results[channel][tag] = sc
            log(f"[{tag} {channel:>7}] " + " ".join(
                f"q{q}={sc[q]['full@10']:.4f}/{sc[q]['tail@10']:.4f}" for q in BUDGETS)
                + f" ({(time.time()-t0)/60:.1f}m)")

    # ---- controls ----
    intercept = {"full": results["concept"][arms[0]][0]["full@10"],   # any arm's q0 == cold intercept
                 "tail": results["concept"][arms[0]][0]["tail@10"]}
    snap_ok = (abs(intercept["full"] - INTERCEPT_REF[0]) < (5e-3 if not args.smoke else 1e9)
               and abs(intercept["tail"] - INTERCEPT_REF[1]) < (5e-3 if not args.smoke else 1e9))
    # wrong-user shuffle on the concept arm (B): fold user (r-1)'s answers -> must collapse to ~intercept
    shuf_model = Behavioral(sh, lvl_lookup)
    perm = [rows[(k - 1) % len(rows)] for k in range(len(rows))]
    ev_sh = {q: None for q in BUDGETS}
    folded = [[] for _ in rows]
    if 0 in BUDGETS: ev_sh[0] = [[] for _ in rows]
    for step in range(1, max(BUDGETS) + 1):
        c = int(conc_order[step - 1])
        for j in range(len(rows)):
            v, ok = shuf_model.concept_value(perm[j], c)             # WRONG user's answer
            if ok:
                folded[j].append((c, v))
        if step in BUDGETS:
            ev_sh[step] = [list(f) for f in folded]
    sc_sh = score_snapshots(rung, rows, ev_sh, "concept")
    shuffle_q8 = {"full@10": sc_sh[8]["full@10"], "tail@10": sc_sh[8]["tail@10"]}
    shuffle_collapses = bool(sc_sh[8]["full@10"] < intercept["full"] + 0.25 *
                             max(results["concept"].get("B", results["concept"][arms[0]])[8]["full@10"]
                                 - intercept["full"], 1e-9))

    # ---- guards: CKA(model concept Z@q8, u*) ----
    cka = {tag: linear_cka(zq8[tag], u_star) for tag in zq8}
    cka["u*_self"] = 1.0
    # Spearman of each imputer's per-(user,concept) value vs oracle-B, over cells answerable under both
    spear = {}
    Vb = sh["conc_value_signed"]
    for tag in arms:
        if tag == "O":
            spear[tag] = 1.0; continue
        Vt = {"B": Vb, "K": Vb, "S": sh.get("conc_value_selplus"),
              "C": sh.get("conc_value_content")}.get(tag)          # K folds honest B values (selection arm)
        if Vt is None:                                              # G / U answers have no fixed value
            spear[tag] = None; continue                            # table -> N/A

        both = ansf & Ff & sh["conc_answerable"] & sh["conc_fold_signed"]
        a = Vt[both]; b = Vf[both]
        spear[tag] = spearman(a, b)

    # ---- key quantities (paired bootstrap on concept-asking deltas @ q8; full@10, tail@10, full@100) ----
    MK = {"full": "_fv", "tail": "_tv", "full100": "_fv100"}
    def pair(tag_a, tag_b, metric):
        va = results["concept"][tag_a][8][MK[metric]]
        vb = results["concept"][tag_b][8][MK[metric]]
        m, ci, _ = bootstrap_ci(va - vb)
        return {"mean": m, "ci95": ci}
    def trip(a, b, meaning=None):
        d = {"full": pair(a, b, "full"), "tail": pair(a, b, "tail"), "full100": pair(a, b, "full100")}
        if meaning: d["meaning"] = meaning
        return d
    A = set(arms); key = {}
    if {"G", "B"} <= A:
        key["G_minus_B_concept_q8"] = trip("G", "B", "self-preference inflation of the geometric answer")
    if {"O", "B"} <= A:
        key["O_minus_B_concept_q8"] = trip("O", "B", "headroom an honest better SEL imputer could capture")
    if {"G", "O"} <= A:
        key["G_minus_O_concept_q8"] = trip("G", "O", "does geometric converge to the SEL-oracle")
    if {"K", "B"} <= A:
        key["K_minus_B_concept_q8"] = trip("K", "B", "SELECTION gap: oracle per-user concept selection "
                                                     "over the generic polarization bank (honest B answer)")
    if {"U", "K"} <= A:
        key["U_minus_K_concept_q8"] = trip("U", "K", "answer-value gain beyond best selection")
    if {"U", "B"} <= A:
        key["U_minus_B_concept_q8"] = trip("U", "B", "TRUE answer-channel ceiling over behavioral SEL")
    if {"U", "O"} <= A:
        key["U_minus_O_concept_q8"] = trip("U", "O", "how far the SEL-formula oracle sits below true NDCG "
                                                     "ceiling (SEL is NOT the ceiling if >0)")
    for imp in ("S", "C"):
        if {imp, "B"} <= A:
            key[f"{imp}_minus_B_concept_q8"] = trip(imp, "B", f"{imp} imputer lift over behavioral SEL")
        if {imp, "O"} <= A:
            key[f"{imp}_minus_O_concept_q8"] = trip(imp, "O", f"does {imp} BEAT the SEL-oracle "
                                                              "(>0 => SEL formula is not the ceiling)")
        if {"U", imp} <= A:
            key[f"U_minus_{imp}_concept_q8"] = trip("U", imp, f"gap from {imp} to the true utility ceiling")
    # item-ask vs concept-ask WITHIN each answer model (does the answer model flip the winner?)
    flip = {}
    for tag in arms:
        if tag not in results["item"]:                              # concept-only arms (K) have no flip
            continue
        flip[tag] = {}
        for q in BUDGETS:
            cf = results["concept"][tag][q]["full@10"]; itf = results["item"][tag][q]["full@10"]
            flip[tag][str(q)] = {"concept_full": cf, "item_full": itf, "concept_wins": bool(cf > itf)}

    strip = lambda sc: {str(q): {k: v for k, v in sc[q].items() if not k.startswith("_")}
                        for q in BUDGETS}
    out = {"analysis": "answer_contrast_newrec", "n_users": len(rows), "budgets": list(BUDGETS),
           "arms": arms, "stack": "frozen t2i25_EP4 tower + signed C-lite (cfold_signed_best.pt)",
           "fixed_orders": {"item": "popularity", "concept": "polarization (argsort -std|v|)"},
           "diagnostic": "UNDERSTANDING run (not for paper): NDCG@100 + mean_answered + utility oracle "
                         "+ imputer arms S/C added to the G/B/O anchors",
           "geometric_def": "matched |v_B| magnitude, geometric sign toward u*=tower-fold(known-half); "
                            "items candidate levels {9,0}; B's answerable set (isolates answer DIRECTION)",
           "utility_oracle_def": "PRIVILEGED true answer-channel ceiling: greedy per question, pick "
                                 "concept sign {+1,-1,skip} / item level {9,0,skip} maximizing held-out "
                                 "NDCG@10 after folding (ceiling of the ANSWER, not of the SEL formula)",
           "imputer_def": "S=SEL+ (BM25 lift + EB-shrunk genome content prior); C=content-projection "
                          "(genome affinity only) -- both model-free/non-circular",
           "ks": list(KS),
           "intercept": intercept, "intercept_ref": INTERCEPT_REF,
           "controls": {"q0_canonical_snap_PASS": bool(snap_ok),
                        "wrong_user_shuffle_concept_q8": shuffle_q8,
                        "shuffle_collapses_toward_intercept": shuffle_collapses,
                        "leak_foldin_heldout_overlap_users": leaks},
           "guards": {"linear_CKA_answergeom_vs_ustar": cka,
                      "spearman_value_vs_oracleB": spear,
                      "cka_note": "G should be HIGH (visible circularity); an SEL+ arm must not exceed "
                                  "SEL(B)'s CKA while agreeing MORE with oracle-B (Spearman)."},
           "item_ask": {tag: strip(results["item"][tag]) for tag in arms if tag in results["item"]},
           "concept_ask": {tag: strip(results["concept"][tag]) for tag in arms},
           "channel_flip_within_model": flip,
           "key_quantities": key}
    out["seconds"] = round(time.time() - t00, 1)
    outdir = ctx.outdir if args.smoke else OUTDIR
    os.makedirs(outdir, exist_ok=True)
    jpath = os.path.join(outdir, "answer_contrast_newrec.json")
    json.dump(out, open(jpath, "w"), indent=2, default=float)
    log(f"[out] -> {jpath}")

    # ---- printed table (full@10 / tail@10 / full@100 per cell) ----
    print(f"\n=== ANSWER-MODEL CONTRAST (sclite; intercept {intercept['full']:.4f}/{intercept['tail']:.4f}"
          f"/{results['concept'][arms[0]][0]['full@100']:.4f}; snap={snap_ok}) ===")
    print(f"{'row':>16} | " + " ".join(f"q{q:<18}" for q in BUDGETS) + "  (f@10/t@10/f@100)")
    for channel, label in (("item", "item-ask"), ("concept", "concept-ask")):
        for tag in arms:
            if tag not in results[channel]:
                continue
            sc = results[channel][tag]
            print(f"{label+'x'+tag:>16} | " + " ".join(
                f"{sc[q]['full@10']:.4f}/{sc[q]['tail@10']:.4f}/{sc[q]['full@100']:.4f}"
                for q in BUDGETS))
    print("\nMEAN_ANSWERED (questions actually folded per budget):")
    for channel, label in (("item", "item-ask"), ("concept", "concept-ask")):
        for tag in arms:
            if tag not in results[channel]:
                continue
            sc = results[channel][tag]
            print(f"{label+'x'+tag:>16} | " + " ".join(
                f"q{q}={sc[q]['mean_answered']:.2f}" for q in BUDGETS))
    print("\nGUARDS  CKA(answer-geom vs u*): " + " ".join(f"{k}={v:.3f}" for k, v in cka.items()))
    print("        Spearman(value vs oracle-B): " + " ".join(
        f"{k}={('%.3f' % v) if isinstance(v, float) else v}" for k, v in spear.items()))
    print("\nKEY QUANTITIES (concept-asking, q8, paired bootstrap 95% CI):")
    for k, v in key.items():
        print(f"  {k}: f@10 {v['full']['mean']:+.4f} CI{v['full']['ci95']} | "
              f"t@10 {v['tail']['mean']:+.4f} | f@100 {v['full100']['mean']:+.4f} "
              f"CI{v['full100']['ci95']}" + (f"  <- {v['meaning']}" if "meaning" in v else ""))
    print(f"\nCONTROLS: wrong-user shuffle q8 full={shuffle_q8['full@10']:.4f} "
          f"(collapses={shuffle_collapses}); leak_users={leaks}")


if __name__ == "__main__":
    main()
