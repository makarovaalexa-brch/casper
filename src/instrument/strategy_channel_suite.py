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
from concept_fold import fold_items_frozen
from i26_encoder import UNSEEN_LEVEL

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


def _greedy_build_ev(rung, rows, seq, fold_refusals=False, refuse_level=4):
    """LEGACY (i25/sclite contract, 2026-07-27). Superseded for the certified stack by
    `genv_eval` below; kept because greedy_polish.py imports this and `_greedy_avg`, and because the
    recorded Jul-27 numbers were produced by it. DO NOT use it with an i26 tower: it has no exposure
    (UNSEEN) token and no refusal token, so it measures a different information contract.

    Per-user (item_ev, conc_ev) for a fixed sequence of (chan, id): item iff rated, concept iff
    structurally answerable. Static -> same seq for every user.
    If fold_refusals: an asked-but-unrated item that is NOT the user's held-out target is folded as a
    mild-negative 'refusal' at level=refuse_level (heuristic; the model was not trained for refusals).
    Targets are masked from the refusal set (at deployment the user would answer them)."""
    sh = rung.sh; ctx = rung.ctx
    Vs = sh["conc_value_signed"]; Fs = sh["conc_fold_signed"]; answ = sh["conc_answerable"]
    iev, cev = [], []
    for r in rows:
        d = sh["lvl_lookup"][r]; items = []; fold = []
        te = set(ctx.va_te[r].indices.tolist()) if fold_refusals else None
        for ch, idx in seq:
            if ch == 0:
                if idx in d:
                    items.append((idx, d[idx]))
                elif fold_refusals and idx not in te:            # item refusal (not seen, not a target)
                    items.append((idx, refuse_level))
            else:
                if answ[r, idx] and Fs[r, idx]: fold.append((idx, float(Vs[r, idx])))
        iev.append((np.asarray([s for s, _ in items], np.int64),
                    np.asarray([l for _, l in items], np.int64)))
        cev.append(fold)
    return iev, cev


def _greedy_avg(rung, rows, seq, fold_refusals=False, refuse_level=4):
    """Mean full/tail NDCG@10 over rows of folding a fixed sequence."""
    iev, cev = _greedy_build_ev(rung, rows, seq, fold_refusals, refuse_level)
    full, tail, _ = score_users(rung, iev, cev, rows)
    return float(np.nanmean(full)), float(np.nanmean(tail))


def _seq_answered(sh, rows, seq):
    """Mean #items answered and #concepts answered per user for the fixed sequence (answerability):
    how many of the ASKED questions the average user could actually answer."""
    Fs = sh["conc_fold_signed"]; answ = sh["conc_answerable"]
    ii = cc = 0
    for r in rows:
        d = sh["lvl_lookup"][r]
        for ch, idx in seq:
            if ch == 0:
                if idx in d: ii += 1
            elif answ[r, idx] and Fs[r, idx]:
                cc += 1
    n = max(len(rows), 1)
    return ii / n, cc / n


def greedy_static_seq(rung, rows, pool, L, fold_refusals=False, refuse_level=4, objective="full"):
    """LEGACY (i25/sclite contract). The certified-stack driver is `greedy_certified` below.

    Lazy-greedy (CELF) best static sequence of length L from pool=[(chan,id),...], maximizing
    mean NDCG@10 over rows. Exploits submodularity: recompute a candidate's marginal gain only
    when it reaches the top with a stale bound. Returns (seq, gains, n_evals).

    objective: "full" (default; the 2026-07-27 run used this) or "tail". The original run maximised
    FULL only and discarded tail, so every tail number it produced was incidental -- which is why the
    tail objective is worth running separately rather than re-reading the full-optimised sequence.
    """
    import heapq
    assert objective in ("full", "tail"), objective
    pick = (lambda ft: ft[0]) if objective == "full" else (lambda ft: ft[1])
    base = pick(_greedy_avg(rung, rows, [], fold_refusals, refuse_level))
    heap = []; nev = 0
    for q in pool:
        v = pick(_greedy_avg(rung, rows, [q], fold_refusals, refuse_level)); nev += 1
        heapq.heappush(heap, (-(v - base), q, 0))
    seq, gains, cur = [], [], base
    while len(seq) < L and heap:
        neg, q, upd = heapq.heappop(heap)
        if upd == len(seq):
            seq.append(q); gains.append(-neg); cur += -neg
        else:
            v = pick(_greedy_avg(rung, rows, seq + [q], fold_refusals, refuse_level)); nev += 1
            heapq.heappush(heap, (-(v - cur), q, len(seq)))
    return seq, gains, nev


# ==========================================================================================
# CERTIFIED-STACK GREEDY  (tower i26c ep31 + concept channel cv3 ep08)          2026-08-04
# ==========================================================================================
# WHY A SECOND IMPLEMENTATION AND NOT AN EDIT. The Jul-27 greedy ran the i25 contract: an asked item
# the user had not rated contributed NOTHING (or, under --fold_refusals, a hand-picked mild-negative
# level the tower was never trained on), and an unanswerable concept was silently dropped. The
# certified stack has native tokens for both cases and was TRAINED on them:
#   * item asked, not rated -> exposure token at UNSEEN_LEVEL (i26's exposure branch)
#   * concept asked, not answerable -> refusal token, value 0.0 with the refusal flag set
#     (cv3 was built with use_refusal=True; 41.6% of its training concept tokens were refusals)
# Folding those away instead of folding them in would hand both channels a contract the models were
# not trained for, and would silently make an unanswered question free. The conventions below are
# copied from the certified-era callers (concept_cold_check.py / bank_sweep.py / concept_train_v2.py)
# rather than re-derived:
#   - ask order, NOT canonical desc-value order: the trainer feeds cids in ask order.
#   - values from signed_for_pool's construction with keep_meh=False, cap_negatives applied over the
#     REVEALED list only (the population-wide cap in build_shared is a different, harsher object).
#   - only ANSWERED items are masked out of the candidate list; an asked-but-unseen item is not
#     revealed to the user, so it neither burns a target nor leaves the ranking.
# LEGACY code above is untouched so greedy_polish.py and the recorded Jul-27 numbers stay readable.
CERT_TOWER = os.path.join(_ROOT, "models", "i26c_ep31", "t2i26c_ep31.pt")
CERT_CONC = os.path.join(_ROOT, "models", "i26c_ep31", "concept_cv3_ep08.pt")
CONCSTATS = os.path.join(_ROOT, ".cache", "instrument", "signed_train_concstats.npz")
N_TRAIN_USERS = 140768.0                       # canonical Liang split train cohort (answer-rate denom)


def load_certified_concept_net(path, n_concepts, d, conc_init, tower_basename=None):
    """Load the concept channel EXACTLY as bank_sweep.py does: architecture and answer contract read
    from the checkpoint, never from CLI defaults. Refuses rather than partially loads."""
    from concept_fold import ConceptFoldNet
    blob = torch.load(path, map_location="cpu")
    sd = blob["net"]
    h = int(sd["mlp.0.weight"].shape[0])
    use_refusal = bool(blob.get("use_refusal", False))
    keep_meh = bool(blob.get("keep_meh", False))
    # keep_meh governs the ANSWER VALUES, so it must match how this harness builds them. We build with
    # keep_meh=False (meh -> 0.0), which is what cv3 trained on; a keep_meh=True checkpoint (cv5) would
    # be scored on values it never saw.
    assert not keep_meh, (f"{os.path.basename(path)} was trained with keep_meh=True; this harness "
                          f"builds keep_meh=False values (meh zeroed). Scoring it here would measure "
                          f"a different system.")
    exp_in = 2 * d + (2 if use_refusal else 1)
    assert int(sd["mlp.0.weight"].shape[1]) == exp_in, \
        f"fold input width {int(sd['mlp.0.weight'].shape[1])} != expected {exp_in} (d={d}, " \
        f"use_refusal={use_refusal}) -- wrong tower latent or wrong answer contract"
    assert tuple(sd["emb.weight"].shape) == (n_concepts, d), \
        f"concept vocabulary drift: ckpt {tuple(sd['emb.weight'].shape)} vs harness ({n_concepts}, {d})"
    if "tags" in blob:
        assert len(blob["tags"]) == n_concepts, "concept vocabulary drift (tags length)"
    if tower_basename is not None and blob.get("snapshot"):
        assert os.path.basename(blob["snapshot"]) == tower_basename, \
            f"PAIR MISMATCH: concept ckpt was trained on {blob['snapshot']}, tower is {tower_basename}"
    net = ConceptFoldNet(n_concepts, d=d, h=h, conc_init=conc_init, use_refusal=use_refusal)
    net.load_state_dict(sd)                                  # strict: must fully populate
    net.eval()
    # G0 item-tie: with no concept answers the fold is bit-identical to the tower.
    zr = torch.randn(4, d)
    with torch.no_grad():
        z_out = net(zr, [[]] * 4, [[]] * 4, [[]] * 4 if use_refusal else None)
    assert torch.equal(z_out, zr), "G0 broken: empty-concept fold is not bit-identical to the tower"
    log(f"[cert] concept channel {os.path.basename(path)} ep{blob.get('epoch')} hidden={h} "
        f"use_refusal={use_refusal} keep_meh={keep_meh} params={net.n_params():,} "
        f"(trained on {blob.get('snapshot')}) | G0 item-tie PASS")
    return net, blob


def build_greedy_env(snapshot, split, conc_ckpt, label, need_entropy=False):
    """One scoring environment (tower + concept channel + answers) on ONE canonical cohort.

    split='validation' -> the 10k VAL users (the SELECTION cohort)
    split='test'       -> the disjoint 10k TEST users (the REPORTING cohort)
    """
    from concept_fold import build_member_matrix
    from signed_answers import load_prereg, signed_values
    log(f"=== building {label} env (split={split}) ===")
    ctx = build_real_ctx(snapshot, split=split)
    sh = build_shared(ctx)                                    # sets ctx.Wd / ctx.bd / ctx.d
    assert sh.get("signed_available"), "signed prereg cache required"
    tags = sh["tags"]; C = len(tags)
    # ANSWER VALUES, certified convention. build_shared's own conc_value_signed applies the C_NEG cap
    # across ALL 1031 concepts at once; the trainer and every certified-era caller apply it over the
    # REVEALED list only (cap_negatives), which is a much weaker shrink. Recompute uncapped here and
    # cap per revealed set at fold time.
    Mm = build_member_matrix(sh["members"], tags, ctx.ni)
    pexp = (ctx.cnt @ np.asarray(Mm.todense())) / max(ctx.cnt.sum(), 1e-9)
    prereg, item_mean = load_prereg()
    items_l = [np.asarray(s, np.int64) for s, l in ctx.allb]
    stars_l = [((np.asarray(l, np.float64) + 1) / 2).astype(np.float32) for s, l in ctx.allb]
    V, Fb, Bb, ansm = signed_values(items_l, stars_l, item_mean, Mm, pexp, ctx.ni, prereg,
                                    apply_neg_cap=False)
    foldable = (Fb & ansm)                                   # ans[c] and B[c] != BAND_REFUSE
    net, cblob = load_certified_concept_net(conc_ckpt, C, ctx.d, sh["d_c"].numpy(),
                                            tower_basename=os.path.basename(snapshot))
    rows = [r for r in range(ctx.n) if ctx.va_te[r].nnz > 0]
    item_H = train_entropies(ctx)[0] if need_entropy else None
    # ctx.raw is a full copy of the 25M-row ratings frame, stashed only for train_entropies (already
    # consumed inside build_shared). Two cohorts alive at once would double it for nothing. Deleted
    # rather than set to None because train_entropies branches on hasattr(ctx, "raw").
    del ctx.raw
    E = {"name": label, "split": split, "ctx": ctx, "sh": sh, "net": net, "cblob": cblob,
         "V": V, "foldable": foldable, "rows": rows, "lvl": sh["lvl_lookup"], "C": C, "item_H": item_H,
         "cneg": float(prereg.get("C_NEG", 2.0)), "cache": {}, "n_evals": 0, "t_evals": 0.0}
    set_active_rows(E, rows)
    ans_rate_c = float(foldable[np.asarray(rows)].mean())
    log(f"[{label}] {len(rows)} cohort users; {C} concepts; concept answer rate over the whole bank "
        f"= {ans_rate_c:.3f}; tower latent d={ctx.d}")
    return E


def set_active_rows(E, rows):
    """Fix the cohort this env scores. Everything below is indexed POSITIONALLY against it."""
    a = np.asarray(rows, np.int64)
    pos = np.full(E["ctx"].n, -1, np.int64); pos[a] = np.arange(len(a))
    E["active"] = a; E["posmap"] = pos; E["cache"] = {}
    return a


def _env_item_state(E, item_ids):
    """(Zi, answered-item CSR) over the env's ACTIVE rows for a fixed asked-ITEM list.

    Cached on the item part alone: in a CELF sweep every CONCEPT candidate shares the prefix's item
    set, so half the candidate evaluations skip the tower fold entirely. Exact, not an approximation.
    """
    key = tuple(item_ids)
    c = E["cache"]
    if c.get("key") == key:
        return c["Zi"], c["ans"]
    ctx = E["ctx"]; lvl = E["lvl"]; act = E["active"]
    ids = np.asarray(item_ids, np.int64)
    seqs = []; ar = []; ac = []
    for j, r in enumerate(act):
        d = lvl[r]
        lv = (np.fromiter((d.get(int(i), UNSEEN_LEVEL) for i in ids), np.int64, len(ids))
              if len(ids) else np.empty(0, np.int64))
        seqs.append((ids, lv))
        if len(ids):
            m = lv < UNSEEN_LEVEL                            # ANSWERED (the user rated it)
            k = int(m.sum())
            if k:
                ar.extend([j] * k); ac.extend(ids[m].tolist())
    # ALWAYS fold through the tower, even with zero items: i26's no-evidence latent is the trainable
    # z0, not the zero vector. Handing the concept net zeros is the documented G5 base-mismatch bug.
    Zi = fold_items_frozen(ctx.enc, seqs, batch=500)
    ans = sparse.csr_matrix((np.ones(len(ar), np.float32), (ar, ac)), shape=(len(act), ctx.ni))
    # An asked-but-unrated item is never revealed to the user, so it must not leave the ranking; an
    # ANSWERED item can never be a held-out target (lvl_lookup is built with te excluded). Assert it.
    assert ans.multiply(ctx.va_te[act]).nnz == 0, "an answered item is also a held-out target"
    E["cache"] = {"key": key, "Zi": Zi, "ans": ans}
    return Zi, ans


def genv_eval(E, seq, rows, batch=500):
    """Per-user (full, tail) NDCG@10 for ONE static ask sequence under the certified contract.

    seq = [(chan, id), ...] in ASK ORDER; chan 0 = item, 1 = concept. Every asked question burns the
    budget for every user. NO FLOOR, learned decoder bias, canonical head mask.
    """
    t0 = time.time()
    item_ids = [int(i) for ch, i in seq if ch == 0]
    conc_ids = [int(c) for ch, c in seq if ch == 1]
    Zi, ans = _env_item_state(E, item_ids)
    ctx = E["ctx"]; net = E["net"]
    ridx = np.asarray(rows, np.int64)
    jj = E["posmap"][ridx]                                   # positions inside the ACTIVE cohort
    assert (jj >= 0).all(), "genv_eval called on rows outside the env's active cohort"
    full = np.full(len(rows), np.nan); tail = np.full(len(rows), np.nan)
    cvals_all = crefs_all = None
    if conc_ids:
        cc = np.asarray(conc_ids, np.int64)
        ok = E["foldable"][np.ix_(ridx, cc)]                 # answerable AND not the refuse band
        vv = np.where(ok, E["V"][np.ix_(ridx, cc)], 0.0).astype(np.float64)
        # cap_negatives over the REVEALED set, vectorised (identical formula, asserted in the smoke)
        s = -np.minimum(vv, 0.0).sum(1)
        sc = np.minimum(1.0, E["cneg"] / np.maximum(s, 1e-9))
        cvals_all = np.where(vv < 0, vv * sc[:, None], vv)
        crefs_all = (~ok).astype(np.float64)                 # refusal token: "asked, cannot answer"
    with torch.no_grad():
        for st in range(0, len(rows), batch):
            ch = ridx[st:st + batch]; cj = jj[st:st + batch]
            Zb = Zi[torch.from_numpy(cj)]
            if conc_ids:
                B = len(ch)
                z = net(Zb, [conc_ids] * B, cvals_all[st:st + B].tolist(),
                        crefs_all[st:st + B].tolist() if net.use_refusal else None)
            else:
                z = Zb
            S = (z @ ctx.Wd.T + ctx.bd).numpy().astype(np.float32)
            mask = ctx.va_tr[ch] + ans[cj]                   # tower cold parity + answered items
            mask.data[:] = 1.0
            f, t = ndcg10_from_scores(S, mask.tocsr(), ctx.va_te[ch], ctx.head_mask)
            full[st:st + len(ch)] = f; tail[st:st + len(ch)] = t
    E["n_evals"] += 1; E["t_evals"] += time.time() - t0
    return full, tail


def contract_selftest(E, rows, n=8):
    """The vectorised revealed-set negative cap must equal signed_answers.cap_negatives exactly.
    Cheap, and it is the one place where this harness re-implements a certified-era primitive."""
    from signed_answers import cap_negatives
    rng = np.random.default_rng(SEED)
    cc = rng.choice(E["C"], size=min(6, E["C"]), replace=False)
    for r in rng.choice(np.asarray(rows), size=min(n, len(rows)), replace=False):
        r = int(r)
        ref = cap_negatives([float(E["V"][r, c]) if E["foldable"][r, c] else 0.0 for c in cc],
                            c_neg=E["cneg"])
        ok = E["foldable"][r, cc]
        vv = np.where(ok, E["V"][r, cc], 0.0).astype(np.float64)
        s = -np.minimum(vv, 0.0).sum()
        sc = min(1.0, E["cneg"] / max(s, 1e-9))
        assert np.allclose(ref, np.where(vv < 0, vv * sc, vv), atol=1e-6), "cap_negatives mismatch"
    log("[selftest] vectorised cap == signed_answers.cap_negatives over the REVEALED set")


def genv_mean(E, seq, rows):
    """Cohort-mean (full, tail) for a sequence, MEMOISED on the exact sequence.

    Exact, not an approximation: a static sequence over a fixed cohort has one value. It matters
    because the three arms share a seeding pass -- every length-1 item sequence the items-only arm
    scores is scored again by the combined arm, and likewise for concepts. That is ~1000 of the ~3100
    evaluations in a 500+500 run, i.e. a third of the wall clock, recomputed for nothing.
    """
    memo = E.setdefault("memo", {})
    key = tuple(seq) if len(rows) == len(E["active"]) else None
    if key is not None and key in memo:
        E["n_memo"] = E.get("n_memo", 0) + 1
        return memo[key]
    f, t = genv_eval(E, seq, rows)
    val = (float(np.nanmean(f)), float(np.nanmean(t)))
    if key is not None:
        memo[key] = val
    return val


def genv_answered(E, seq, rows):
    """Mean ANSWERED items and ANSWERED concepts per user for the sequence (the answerability row)."""
    item_ids = [int(i) for ch, i in seq if ch == 0]
    conc_ids = [int(c) for ch, c in seq if ch == 1]
    ridx = np.asarray(rows, np.int64)
    ia = 0.0
    if item_ids:
        lvl = E["lvl"]
        ia = float(np.mean([sum(1 for i in item_ids if i in lvl[r]) for r in rows]))
    ca = (float(E["foldable"][np.ix_(ridx, np.asarray(conc_ids, np.int64))].sum(1).mean())
          if conc_ids else 0.0)
    return ia, ca


def greedy_certified(E, rows, pool, L, objective, tag, log_every=25):
    """Lazy-greedy (CELF) best STATIC sequence of length L from pool=[(chan,id),...], maximising the
    cohort-mean NDCG@10 of `objective` ('full' or 'tail'). Returns (seq, gains, n_evals)."""
    import heapq
    assert objective in ("full", "tail"), objective
    j = 0 if objective == "full" else 1
    t0 = time.time()
    base = genv_mean(E, [], rows)[j]
    heap = []; nev = 0
    for q in pool:
        v = genv_mean(E, [q], rows)[j]; nev += 1
        heapq.heappush(heap, (-(v - base), q, 0))
        if nev % log_every == 0:
            el = time.time() - t0
            log(f"[greedy {tag}] seed {nev}/{len(pool)} ({el / 60:.1f}m, {el / nev:.2f}s/eval)")
    seq = []; gains = []; cur = base
    while len(seq) < L and heap:
        neg, q, upd = heapq.heappop(heap)
        if upd == len(seq):
            seq.append(q); gains.append(-neg); cur += -neg
            log(f"[greedy {tag}] pick {len(seq)}/{L} = {'concept' if q[0] else 'item'} {q[1]} "
                f"gain {-neg:+.5f} -> {cur:.5f} (nev={nev}, {(time.time() - t0) / 60:.1f}m)")
        else:
            v = genv_mean(E, seq + [q], rows)[j]; nev += 1
            heapq.heappush(heap, (-(v - cur), q, len(seq)))
    return seq, gains, nev


def _ranked_pool(name, tables, n_take, kind):
    """Resolve a NAMED bank to a real ranking, or die.

    This is the assert from concept_cold_check.py (fix 26fe795) in its general form. There, `--bank
    expinfo` was missing from the lookup, fell through to `crank=None` -- i.e. RANDOM draws -- and
    still logged "bank=expinfo", so a run labelled with a ranked bank silently measured a different
    contract for eleven hours. A KeyError here (not a .get default) makes that unreachable, and the
    monotonicity assert makes it impossible for an unranked order to masquerade as a named one.
    """
    key = np.asarray(tables[name], np.float64)               # KeyError = unknown bank, never a default
    rank = np.argsort(-key, kind="stable")
    assert rank.ndim == 1 and len(np.unique(rank)) == len(rank), f"{kind} bank '{name}' is not a permutation"
    kk = key[rank]
    assert np.all(np.diff(kk) <= 1e-9), \
        (f"{kind} bank '{name}' is NOT sorted by its own criterion -- an unranked/RANDOM order must "
         f"never masquerade as a named bank (the --bank expinfo fallthrough, fixed 26fe795)")
    take = rank[:n_take]
    assert np.isfinite(key[take]).all(), f"{kind} bank '{name}' has non-finite scores in the top-{n_take}"
    return take


def make_greedy_png(results, intercept, path):
    """Best-static curves: items-only / concepts-only / combined; full & tail, EQUAL y-axes."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    CURVES = [("items-only", "#0072B2", "--o"), ("concepts-only", "#009E73", "--^"),
              ("combined", "#D55E00", "-s")]
    allv = [results[n]["curve"][str(q)][m] for n, _, _ in CURVES for q in BUDGETS
            for m in ("full@10", "tail@10")]
    ymin = min(min(allv), intercept["full"], intercept["tail"]); ymax = max(allv)
    pad = (ymax - ymin) * 0.06
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True)
    for ax, metric, ikey in ((axes[0], "full@10", "full"), (axes[1], "tail@10", "tail")):
        ax.axhline(intercept[ikey], color="#888888", ls=":", lw=1.1, label="no-answer intercept")
        for name, col, mk in CURVES:
            c = results[name]["curve"]
            ax.plot(BUDGETS, [c[str(q)][metric] for q in BUDGETS], mk, color=col, lw=2, ms=5,
                    label=name)
        ax.set_xscale("log", base=2); ax.set_xticks(BUDGETS); ax.set_xticklabels(BUDGETS)
        ax.set_xlabel("questions asked $q$"); ax.set_title(metric, fontsize=10)
        ax.set_ylim(ymin - pad, ymax + pad); ax.grid(alpha=0.3)
    axes[0].set_ylabel("NDCG@10"); axes[0].legend(fontsize=7.5, loc="upper left")
    fig.suptitle("Best static question sequence (greedy NDCG): does adding concepts help?",
                 fontsize=10)
    fig.tight_layout(); fig.savefig(path, dpi=160); fig.savefig(path.replace(".png", ".pdf"))
    plt.close(fig)


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


# ============================================================================= greedy driver
def greedy_main(args):
    """Best-static greedy question sequence on the CERTIFIED stack.

    USER SPLIT -- stated plainly, because getting this wrong is a recorded fatal review point:
      SELECTION (the greedy search, i.e. which questions get chosen) runs on the canonical 10k
      VALIDATION users and NOTHING else. The TEST cohort is built only after the sequences are frozen
      and is touched exactly once per (arm, budget) to report. No checkpoint, epoch, bank, pool or
      question is ever chosen using a test number. --eval_test is therefore the paper-grade mode;
      without it the run reports on a disjoint slice of VAL and is labelled a PEEK.
    """
    t00 = time.time()
    SMOKE = args.smoke_users > 0
    assert not args.fold_refusals, (
        "--fold_refusals is the LEGACY i25 heuristic (an invented mild-negative level the tower was "
        "never trained on). The certified i26 tower folds an asked-but-unrated item as a native "
        "UNSEEN exposure token, which this path always does.")
    sfx = ("" if args.objective == "full" else f"_{args.objective}") + ("_SMOKE" if SMOKE else "")
    GRJSON = os.path.join(_ROOT, "experiments", "battery", f"greedy_static_i26c{sfx}.json")
    GRPNG = os.path.join(_ROOT, "experiments", "battery", f"greedy_static_i26c{sfx}.png")
    if SMOKE:
        log("#" * 100)
        log(f"# SMOKE MODE: {args.smoke_users} users per cohort. Pipeline check ONLY -- these numbers "
            f"are NOT citable and land in *_SMOKE.json.")
        log("#" * 100)
    tblob = torch.load(args.snapshot, map_location="cpu")
    log(f"[cert] tower {os.path.basename(args.snapshot)} arch={tblob.get('arch')} "
        f"ep{tblob.get('epoch')} val_full={tblob.get('val_full'):.4f} "
        f"val_tail={tblob.get('val_tail'):.4f} k0={tblob.get('k0'):.4f} sel={tblob.get('sel'):.4f}")
    assert tblob.get("arch") == "i26", \
        f"certified greedy expects the i26 tower; got arch={tblob.get('arch')}"

    # ---- SELECTION cohort (validation) --------------------------------------------------------
    Eb = build_greedy_env(args.snapshot, "validation", args.concept_ckpt, "BUILD(val)",
                          need_entropy=(args.item_bank == "entropy"))
    build_rows = Eb["rows"]
    if SMOKE:
        build_rows = build_rows[:args.smoke_users]
        set_active_rows(Eb, build_rows)          # SMOKE ONLY: labelled user truncation, never citable

    contract_selftest(Eb, build_rows)

    # ---- GATE (a): the empty interview must reproduce the tower's own certified cold number -----
    i_full, i_tail = genv_mean(Eb, [], build_rows)
    ref = float(tblob["k0"])
    ok = abs(i_full - ref) <= args.intercept_tol
    log(f"[gate a] m=0 empty interview on {len(build_rows)} val users: full={i_full:.5f} "
        f"tail={i_tail:.5f} | tower's recorded cold k0={ref:.5f} | delta={i_full - ref:+.5f} "
        f"tol={args.intercept_tol} -> {'PASS' if ok else 'FAIL'}")
    if not ok:
        assert SMOKE, ("GATE (a) FAILED: the empty-interview row does not reproduce the tower's "
                       "certified cold number. Something in the fold/mask/metric path differs from "
                       "the contract that certified the tower -- do not run 10 hours on top of it.")
        log("[gate a] SMOKE: not enforced (a 50-user subset is not the 10k cohort the k0 reference "
            "was measured on). It IS enforced on the real run.")
    intercept = {"full": i_full, "tail": i_tail}

    # ---- banks + pools ------------------------------------------------------------------------
    # Every bank is a RANKED, user-independent order; the pool is its top-N. No random draws anywhere.
    assert os.path.exists(CONCSTATS), f"train concept stats missing ({CONCSTATS})"
    z = np.load(CONCSTATS)
    ans_rate = (z["n"] / N_TRAIN_USERS).astype(np.float64)   # fraction of TRAIN users who can answer
    std_v = np.asarray(z["std"], np.float64)                 # train spread of the signed value
    pol_n = std_v / max(std_v.max(), 1e-9)
    ar_n = ans_rate / max(ans_rate.max(), 1e-9)
    conc_tables = {"answerable": ans_rate, "polar": std_v,
                   "helf": 2 * pol_n * ar_n / np.clip(pol_n + ar_n, 1e-9, None)}
    assert len(ans_rate) == Eb["C"], f"concept stats width {len(ans_rate)} != {Eb['C']} concepts"
    item_tables = {"pop": np.asarray(Eb["ctx"].cnt, np.float64)}
    if Eb["item_H"] is not None:
        item_tables["entropy"] = np.asarray(Eb["item_H"], np.float64)
    n_it, n_cc = args.n_items, args.n_concepts
    item_pool = [(0, int(i)) for i in _ranked_pool(args.item_bank, item_tables, n_it, "item")]
    conc_pool = [(1, int(c)) for c in _ranked_pool(args.conc_bank, conc_tables, n_cc, "concept")]
    log(f"[banks] item bank='{args.item_bank}' -> pool {len(item_pool)} items | "
        f"concept bank='{args.conc_bank}' -> pool {len(conc_pool)} concepts | "
        f"combined {len(item_pool) + len(conc_pool)} | L={max(BUDGETS)} | objective={args.objective}")
    log(f"[banks] top-5 items {[i for _, i in item_pool[:5]]} | "
        f"top-5 concepts {[c for _, c in conc_pool[:5]]} "
        f"(answer rate {[round(float(ans_rate[c]), 3) for _, c in conc_pool[:5]]})")

    # ---- the greedy search: VALIDATION USERS ONLY ---------------------------------------------
    # Each arm's frozen sequence is written to a sidecar the moment it finishes, and a relaunch with
    # the SAME contract resumes from it. This is a multi-hour search with no natural checkpoint; the
    # sidecar is signed with the contract so a changed tower/ckpt/bank/pool/cohort cannot be resumed
    # into silently.
    PARTIAL = GRJSON.replace(".json", ".partial.json")
    sig = {"tower": os.path.basename(args.snapshot), "conc": os.path.basename(args.concept_ckpt),
           "objective": args.objective, "item_bank": args.item_bank, "conc_bank": args.conc_bank,
           "n_items": len(item_pool), "n_concepts": len(conc_pool), "L": max(BUDGETS),
           "n_build_users": len(build_rows), "build_head": int(build_rows[0]),
           "build_tail": int(build_rows[-1])}
    part = json.load(open(PARTIAL)) if os.path.exists(PARTIAL) else {}
    if part.get("sig") != sig:
        if part:
            log(f"[greedy] sidecar {os.path.basename(PARTIAL)} has a DIFFERENT contract -- ignoring it")
        part = {"sig": sig, "arms": {}}
    log(f"[greedy] SELECTION on {len(build_rows)} VALIDATION users; the test cohort is not built yet.")
    seqs = {}
    for name, pool in [("items-only", item_pool), ("concepts-only", conc_pool),
                       ("combined", item_pool + conc_pool)]:
        if name in part["arms"]:
            p = part["arms"][name]
            seqs[name] = ([(int(a), int(b)) for a, b in p["sequence"]], p["gains"], p["n_evals"])
            log(f"[greedy {name}] RESUMED from sidecar (order={p['order']}, nev={p['n_evals']})")
            continue
        t0 = time.time()
        seq, gains, nev = greedy_certified(Eb, build_rows, pool, max(BUDGETS), args.objective, name)
        seqs[name] = (seq, gains, nev)
        order = "".join("c" if ch else "i" for ch, _ in seq)
        part["arms"][name] = {"sequence": [[int(a), int(b)] for a, b in seq],
                              "gains": [float(g) for g in gains], "n_evals": nev, "order": order,
                              "minutes": round((time.time() - t0) / 60, 1)}
        json.dump(part, open(PARTIAL, "w"), indent=2)
        log(f"[greedy {name}] DONE order={order} nev={nev} ({(time.time() - t0) / 60:.1f}m) "
            f"-> sidecar {os.path.basename(PARTIAL)}")

    # ---- REPORTING cohort: built only now, read only here -------------------------------------
    if args.eval_test:
        Ee = build_greedy_env(args.snapshot, "test", args.concept_ckpt, "EVAL(test)")
        eval_rows = Ee["rows"]
        if SMOKE:
            eval_rows = eval_rows[:args.smoke_users]
            set_active_rows(Ee, eval_rows)
        eval_split = "test"
    else:
        Ee = Eb
        nfit = args.fit_users if args.fit_users > 0 else len(build_rows) // 2
        eval_rows = [r for r in Eb["rows"] if r not in set(build_rows[:nfit])][:nfit]
        set_active_rows(Eb, sorted(set(build_rows) | set(eval_rows)))
        eval_split = "val-disjoint(PEEK)"
        log("[greedy] PEEK MODE: no test cohort. Selection and evaluation share a cohort's users' "
            "answer table; report nothing from this run.")
    ie_full, ie_tail = genv_mean(Ee, [], eval_rows)
    log(f"[gate a/eval] m=0 on {len(eval_rows)} {eval_split} users: {ie_full:.5f}/{ie_tail:.5f}")

    greedy_res = {}
    for name, (seq, gains, nev) in seqs.items():
        curve = {}
        for q in BUDGETS:
            f, t = genv_mean(Ee, seq[:q], eval_rows)          # held-out (the reported number)
            bf, bt = genv_mean(Eb, seq[:q], build_rows)       # build set (the overfit reference)
            ia, ca = genv_answered(Ee, seq[:q], eval_rows)
            curve[str(q)] = {"full@10": f, "tail@10": t, "build_full@10": bf, "build_tail@10": bt,
                             "item_answered": round(ia, 3), "conc_answered": round(ca, 3),
                             "item_asked": sum(1 for ch, _ in seq[:q] if ch == 0),
                             "conc_asked": sum(1 for ch, _ in seq[:q] if ch == 1)}
        comp = "".join("c" if ch == 1 else "i" for ch, _ in seq)
        greedy_res[name] = {"curve": curve, "order": comp,
                            "sequence": [[int(ch), int(idx)] for ch, idx in seq],
                            "gains": [float(g) for g in gains], "n_evals": nev}
        log(f"[greedy {name}] order={comp} " + " ".join(
            f"q{q}:{curve[str(q)]['full@10']:.4f}/{curve[str(q)]['tail@10']:.4f}" for q in BUDGETS)
            + f" | answered@16={curve['16']['item_answered']}i/{curve['16']['conc_answered']}c")

    out = {"analysis": "greedy_static_seq_certified",
           "SMOKE": bool(SMOKE),
           "citable": not SMOKE,
           "tower": os.path.basename(args.snapshot), "tower_epoch": tblob.get("epoch"),
           "concept_ckpt": os.path.basename(args.concept_ckpt),
           "concept_epoch": Eb["cblob"].get("epoch"),
           "use_refusal": bool(Eb["net"].use_refusal),
           "contract": ("NO-FLOOR, learned decoder bias, full+tail NDCG@10, canonical Liang ML-25M "
                        "split. Every asked question burns the budget: an unrated item is folded as "
                        "an i26 UNSEEN exposure token, an unanswerable concept as a cv3 refusal "
                        "token. Only ANSWERED items are masked out of the candidate list."),
           "selection_users": f"{len(build_rows)} canonical VALIDATION users",
           "reporting_users": f"{len(eval_rows)} {eval_split} users",
           "n_build_users": len(build_rows), "n_eval_users": len(eval_rows),
           "eval_split": eval_split, "objective": args.objective,
           "item_bank": args.item_bank, "conc_bank": args.conc_bank,
           "n_items_pool": len(item_pool), "n_concepts_pool": len(conc_pool),
           "gate_m0_full": i_full, "gate_m0_tail": i_tail, "gate_m0_ref_tower_k0": ref,
           "gate_m0_pass": bool(ok),
           "budgets": list(BUDGETS), "intercept": intercept,
           "intercept_eval": {"full": ie_full, "tail": ie_tail}, "arms": greedy_res,
           "seconds": round(time.time() - t00, 1),
           "sec_per_eval_build": round(Eb["t_evals"] / max(Eb["n_evals"], 1), 3),
           "n_evals_total": Eb["n_evals"] + Ee["n_evals"],
           "n_memo_hits_build": Eb.get("n_memo", 0)}
    os.makedirs(os.path.dirname(GRJSON), exist_ok=True)
    json.dump(out, open(GRJSON, "w"), indent=2, default=float)
    try:
        make_greedy_png(greedy_res, {"full": ie_full, "tail": ie_tail}, GRPNG)
    except Exception as e:                                    # a figure must never lose the numbers
        log(f"[greedy] PNG failed ({type(e).__name__}: {e}); JSON is written")
    spe = Eb["t_evals"] / max(Eb["n_evals"], 1)
    log(f"[greedy] wrote {os.path.basename(GRJSON)} ({(time.time() - t00) / 60:.1f}m total; "
        f"{Eb['n_evals']} build evals at {spe:.2f}s each, {Eb.get('n_memo', 0)} memo hits)")
    if SMOKE:
        n_ev_ref = 754 + 878 + 1423 + 30                     # the Jul-27 CELF eval counts + curves
        scale = 10000.0 / max(len(build_rows), 1)
        log("#" * 100)
        log(f"# SMOKE PROJECTION: {spe:.2f}s/eval at {len(build_rows)} users -> "
            f"~{spe * scale:.1f}s/eval at 10,000 users; "
            f"~{n_ev_ref} evals (Jul-27 CELF counts) => ~{spe * scale * n_ev_ref / 3600:.1f} h. "
            f"Pools here were {len(item_pool)}i/{len(conc_pool)}c vs 500/500 in the real run -- "
            f"scale the CELF seeding pass accordingly.")
        log("# SMOKE numbers are NOT citable.")
        log("#" * 100)
    return out


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
    ap.add_argument("--greedy", action="store_true",
                    help="best-static greedy sequence (lazy CELF) for NDCG: items-only / concepts-only "
                         "/ combined; shows the emergent optimal concept+item order. Runs on the "
                         "CERTIFIED stack (i26c ep31 + concept cv3 ep08) unless --snapshot is given.")
    ap.add_argument("--objective", default="full", choices=["full", "tail"],
                    help="metric the greedy maximises. The 2026-07-27 run used 'full' and discarded "
                         "tail, so its tail numbers were incidental; 'tail' gives the concept channel "
                         "its first tail-optimised sequence. Output filenames are suffixed by objective.")
    ap.add_argument("--concept_ckpt", default=CERT_CONC,
                    help="greedy: concept-channel checkpoint (certified = concept_cv3_ep08.pt). Its "
                         "architecture and answer contract are read from the file, never from flags.")
    ap.add_argument("--n_items", type=int, default=500, help="greedy: top-N items in the pool")
    ap.add_argument("--n_concepts", type=int, default=500, help="greedy: top-M concepts in the pool")
    ap.add_argument("--item_bank", default="pop", choices=["pop", "entropy"],
                    help="greedy: which RANKED item bank the pool is the top of (Jul-27 used pop)")
    ap.add_argument("--conc_bank", default="answerable", choices=["answerable", "polar", "helf"],
                    help="greedy: which RANKED concept bank the pool is the top of. 'answerable' = "
                         "train answer rate (what Jul-27 used); 'polar' = train std of the signed "
                         "value; 'helf' = their harmonic mean. Every name must resolve to a real "
                         "ranking -- see _ranked_pool.")
    ap.add_argument("--fit_users", type=int, default=0,
                    help="greedy: #val users to fit+eval on (0=all val). PEEK: small value.")
    ap.add_argument("--fold_refusals", action="store_true",
                    help="LEGACY i25 heuristic; refused on the certified stack (i26 has a native "
                         "UNSEEN exposure token for an asked-but-unrated item)")
    ap.add_argument("--refuse_level", type=int, default=4, help="LEGACY (see --fold_refusals)")
    ap.add_argument("--eval_test", action="store_true",
                    help="greedy: BUILD on all val, EVAL on the disjoint TEST cohort (paper-grade "
                         "split-construct). Ignores --fit_users.")
    ap.add_argument("--smoke_users", type=int, default=0,
                    help="greedy SMOKE: run the WHOLE pipeline end-to-end on this many users per "
                         "cohort. Output is written to *_SMOKE.json and is NEVER citable.")
    ap.add_argument("--intercept_tol", type=float, default=0.002,
                    help="greedy gate (a): allowed |m=0 full@10 - the tower's own certified cold k0|")
    args = ap.parse_args()
    if args.greedy and args.snapshot == PA.SNAP_DEFAULT:
        args.snapshot = CERT_TOWER            # the certified pair is the default for the greedy run
    if args.greedy:
        return greedy_main(args)
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
    # fold provenance: the concept arm is only as good as the fold it rides. cfold_signed_best.pt is the
    # SUPERSEDED Jul-25 fold (pre-recipe-fix); the deployment fold of record is cd_s1_l10_best.pt (lam=1.0).
    log(f"[rung] sclite concept fold = {os.path.basename(args.sclite_ckpt)}")
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
