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

assert not hasattr(sys.modules[__name__], "load_answerer")

BUDGETS = (0, 1, 2, 4, 8)
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


def score_snapshots(rung, rows, ev, channel, budgets=BUDGETS):
    """ev[q] -> per-user full/tail arrays + means, with credit-neutral item masking. channel selects
    whether the snapshot is item evidence or concept evidence."""
    empty_i = [(np.empty(0, np.int64), np.empty(0, np.int64)) for _ in rows]
    out = {}
    for q in budgets:
        snap = ev[q]
        if channel == "concept":
            item_seqs = empty_i; conc_lists = snap
        else:
            item_seqs = snap; conc_lists = [[] for _ in rows]
        full, tail, Z = score_users(rung, item_seqs, conc_lists, rows)
        out[q] = {"full@10": float(np.nanmean(full)), "tail@10": float(np.nanmean(tail)),
                  "mean_answered": float(np.mean([len(c) for c in conc_lists]) if channel == "concept"
                                         else np.mean([len(s[0]) for s in item_seqs])),
                  "_fv": full, "_tv": tail, "_Z": Z.numpy() if q == max(budgets) else None}
    return out


# ============================================================================= main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--snapshot", default=PA.SNAP_DEFAULT)
    ap.add_argument("--arms", default="G,B,O")
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

    models = {}
    if "B" in arms: models["B"] = Behavioral(sh, lvl_lookup)
    if "O" in arms: models["O"] = OracleB(Vf, Ff, ansf, lk_full)
    if "G" in arms: models["G"] = Geometric(sh, lvl_lookup)
    if "S" in arms:
        from selplus import build_selplus_arrays, SELPlus
        Vs, Fs, anss = build_selplus_arrays(ctx, sh, Mm, pexp, smoke=args.smoke)
        models["S"] = SELPlus(Vs, Fs, anss, lvl_lookup)
        sh["conc_value_selplus"] = Vs

    results = {"item": {}, "concept": {}}
    zq8 = {}                                                        # concept belief Z@q8 per model (CKA)
    for tag, model in models.items():
        for channel in ("concept", "item"):
            t0 = time.time()
            if model.geometric:
                if channel == "concept":
                    ev = walk_geo_concept(rung, rows, model, conc_order, u_star)
                else:
                    ev = walk_geo_item(rung, rows, model, item_order, u_star)
            else:
                if channel == "concept":
                    ev = walk_static_concept(rung, rows, model, conc_order)
                else:
                    ev = walk_static_item(rung, rows, model, item_order)
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
        Vt = {"B": Vb, "S": sh.get("conc_value_selplus")}.get(tag)
        if Vt is None:                                              # G answer value is geometric-signed;
            spear[tag] = None; continue                            # not a fixed table -> skip (N/A)
        both = ansf & Ff & sh["conc_answerable"] & sh["conc_fold_signed"]
        a = Vt[both]; b = Vf[both]
        spear[tag] = spearman(a, b)

    # ---- key quantities (paired bootstrap on concept-asking full deltas @ q8) ----
    def pair(tag_a, tag_b, metric):
        va = results["concept"][tag_a][8]["_fv" if metric == "full" else "_tv"]
        vb = results["concept"][tag_b][8]["_fv" if metric == "full" else "_tv"]
        m, ci, _ = bootstrap_ci(va - vb)
        return {"mean": m, "ci95": ci}
    key = {}
    if "G" in arms and "B" in arms:
        key["G_minus_B_concept_q8"] = {"full": pair("G", "B", "full"), "tail": pair("G", "B", "tail"),
                                       "meaning": "self-preference inflation of the geometric answer"}
    if "O" in arms and "B" in arms:
        key["O_minus_B_concept_q8"] = {"full": pair("O", "B", "full"), "tail": pair("O", "B", "tail"),
                                       "meaning": "headroom an honest better imputer could capture"}
    if "G" in arms and "O" in arms:
        key["G_minus_O_concept_q8"] = {"full": pair("G", "O", "full"), "tail": pair("G", "O", "tail"),
                                       "meaning": "does geometric converge to the honest ceiling"}
    if "S" in arms and "B" in arms:
        key["S_minus_B_concept_q8"] = {"full": pair("S", "B", "full"), "tail": pair("S", "B", "tail")}
    if "S" in arms and "O" in arms:
        key["O_minus_S_concept_q8"] = {"full": pair("O", "S", "full"), "tail": pair("O", "S", "tail"),
                                       "meaning": "residual gap SEL+ leaves to the oracle"}
    # item-ask vs concept-ask WITHIN each answer model (does the answer model flip the winner?)
    flip = {}
    for tag in arms:
        flip[tag] = {}
        for q in BUDGETS:
            cf = results["concept"][tag][q]["full@10"]; itf = results["item"][tag][q]["full@10"]
            flip[tag][str(q)] = {"concept_full": cf, "item_full": itf, "concept_wins": bool(cf > itf)}

    strip = lambda sc: {str(q): {k: v for k, v in sc[q].items() if not k.startswith("_")}
                        for q in BUDGETS}
    out = {"analysis": "answer_contrast_newrec", "n_users": len(rows), "budgets": list(BUDGETS),
           "arms": arms, "stack": "frozen t2i25_EP4 tower + signed C-lite (cfold_signed_best.pt)",
           "fixed_orders": {"item": "popularity", "concept": "polarization (argsort -std|v|)"},
           "geometric_def": "matched |v_B| magnitude, geometric sign toward u*=tower-fold(known-half); "
                            "items candidate levels {9,0}; B's answerable set (isolates answer DIRECTION)",
           "intercept": intercept, "intercept_ref": INTERCEPT_REF,
           "controls": {"q0_canonical_snap_PASS": bool(snap_ok),
                        "wrong_user_shuffle_concept_q8": shuffle_q8,
                        "shuffle_collapses_toward_intercept": shuffle_collapses,
                        "leak_foldin_heldout_overlap_users": leaks},
           "guards": {"linear_CKA_answergeom_vs_ustar": cka,
                      "spearman_value_vs_oracleB": spear,
                      "cka_note": "G should be HIGH (visible circularity); an SEL+ arm must not exceed "
                                  "SEL(B)'s CKA while agreeing MORE with oracle-B (Spearman)."},
           "item_ask": {tag: strip(results["item"][tag]) for tag in arms},
           "concept_ask": {tag: strip(results["concept"][tag]) for tag in arms},
           "channel_flip_within_model": flip,
           "key_quantities": key}
    out["seconds"] = round(time.time() - t00, 1)
    outdir = ctx.outdir if args.smoke else OUTDIR
    os.makedirs(outdir, exist_ok=True)
    jpath = os.path.join(outdir, "answer_contrast_newrec.json")
    json.dump(out, open(jpath, "w"), indent=2, default=float)
    log(f"[out] -> {jpath}")

    # ---- printed table ----
    print(f"\n=== ANSWER-MODEL CONTRAST (sclite; intercept {intercept['full']:.4f}/{intercept['tail']:.4f}"
          f"; snap={snap_ok}) ===")
    print(f"{'row':>18} | " + " ".join(f"q{q:<12}" for q in BUDGETS) + "  (full/tail)")
    for channel, label in (("item", "item-ask"), ("concept", "concept-ask")):
        for tag in arms:
            sc = results[channel][tag]
            print(f"{label+'x'+tag:>18} | " + " ".join(
                f"{sc[q]['full@10']:.4f}/{sc[q]['tail@10']:.4f}" for q in BUDGETS))
    print("\nGUARDS  CKA(answer-geom vs u*): " + " ".join(f"{k}={v:.3f}" for k, v in cka.items()))
    print("        Spearman(value vs oracle-B): " + " ".join(
        f"{k}={('%.3f' % v) if isinstance(v, float) else v}" for k, v in spear.items()))
    print("\nKEY QUANTITIES (concept-asking, q8, paired bootstrap 95% CI):")
    for k, v in key.items():
        print(f"  {k}: full {v['full']['mean']:+.4f} CI{v['full']['ci95']} | "
              f"tail {v['tail']['mean']:+.4f} CI{v['tail']['ci95']}"
              + (f"  <- {v['meaning']}" if "meaning" in v else ""))
    print(f"\nCONTROLS: wrong-user shuffle q8 full={shuffle_q8['full@10']:.4f} "
          f"(collapses={shuffle_collapses}); leak_users={leaks}")


if __name__ == "__main__":
    main()
