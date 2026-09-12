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
# `--threads N` overrides both. This box is a 4-PHYSICAL-core i7-7820HQ reporting 8 logical, so
# `--full_threads` (= os.cpu_count() = 8) oversubscribes it whenever anything else is alive; the
# measured remedy on 2026-08-13 was 6. The number is a property of THIS box on THIS night, not of
# any method.
_NT = str(os.cpu_count()) if _FULL else "4"
if "--threads" in sys.argv:
    _i = sys.argv.index("--threads")
    if _i + 1 < len(sys.argv) and sys.argv[_i + 1].isdigit():
        _NT = sys.argv[_i + 1]
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
from mask_contract import pool_contract, resume_ok      # re-exported for the group/tree drivers
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
# NB: _DS is defined further down (with the certified-pair table), so this computes its own suffix.
QBANK_JSON = os.path.join(
    _ROOT, "experiments",
    "battery" + ("" if os.environ.get("CASPER_DATASET", "ml-25m") == "ml-25m"
                 else "_" + os.environ["CASPER_DATASET"]),
    "qbank_expansion.json")
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
#   - CANDIDATE POOL: policy B since 2026-08-05 (was: only ANSWERED items masked, on top of the
#     liked fold-in). Every rated fold-in-side item is masked for every arm, ask-independent; an
#     asked item is never removed from the ranking because it was asked. See genv_eval.
# LEGACY code above is untouched so greedy_polish.py and the recorded Jul-27 numbers stay readable.
# Dataset selection (2026-09-12): follows CASPER_DATASET, default ml-25m so the certified ML-25M
# path is unchanged. The concept STATS are per-dataset quantities -- sharing one filename across
# datasets is the same bug class that produced four silent portability faults in the tower port,
# each of which would have been invisible on a dataset with a matching item count.
_DS = os.environ.get("CASPER_DATASET", "ml-25m")
_SFX = "" if _DS == "ml-25m" else "_" + _DS
# CERTIFIED PAIR per dataset. ml-25m: the Aug-3 certified pair, untouched (author ruling
# 2026-09-12: Paper B's artifact set does not move). goodreads: tower selected by the
# pre-registered rule (t2i26g ep53), concept net cvg_ep10 -- the epoch that same rule picks,
# which A17 missed by taking the run's LAST epoch (ep12) instead.
_CERT = {
    "ml-25m": (os.path.join(_ROOT, "models", "i26c_ep31", "t2i26c_ep31.pt"),
               os.path.join(_ROOT, "models", "i26c_ep31", "concept_cv3_ep08.pt")),
    "goodreads": (os.path.join(_ROOT, ".cache", "instrument", "t2i26g_ep53.pt"),
                  os.path.join(_ROOT, ".cache", "instrument", "cvg_ep10.pt")),
}[_DS]
CERT_TOWER, CERT_CONC = _CERT
CONCSTATS = os.path.join(_ROOT, ".cache", "instrument", f"signed_train_concstats{_SFX}.npz")
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


def build_greedy_env(snapshot, split, conc_ckpt, label, need_entropy=False, cohort_slice=None):
    """One scoring environment (tower + concept channel + answers) on ONE canonical cohort.

    split='validation' -> the 10k VAL users (the SELECTION cohort)
    split='test'       -> the disjoint 10k TEST users (the REPORTING cohort)
    split='train'      -> a contiguous slice of the 140,768 TRAIN users, `cohort_slice=(lo, hi)`.
                          WORKSTREAM E1 ONLY, and NDCG IS NEVER REPORTED ON IT: the tower was trained
                          on these users, so any NDCG measured here is memorisation. What the slice
                          supplies is interview STATES and the frozen tree's LABELS -- both functions
                          of the user's own answers only. See PAPER_B_POLICY_DESIGN.md D.6b.
    """
    from concept_fold import build_member_matrix
    from signed_answers import load_prereg, signed_values
    log(f"=== building {label} env (split={split}"
        f"{'' if cohort_slice is None else f', users {cohort_slice[0]}..{cohort_slice[1]}'}) ===")
    ctx = build_real_ctx(snapshot, split=split, cohort_slice=cohort_slice)
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
    # POLICY-B CANDIDATE POOL, built ONCE per environment, before any question exists. It is a
    # property of the PROTOCOL, not of the arm: every sequence this env ever scores gets this exact
    # mask (2026-08-05 ruling, external_literature/findings/candidate_masking_protocol.md).
    pool_b = PA.ctx_pool_b(ctx)
    item_H = train_entropies(ctx)[0] if need_entropy else None
    # ctx.raw is a full copy of the 25M-row ratings frame, stashed only for train_entropies (already
    # consumed inside build_shared). Two cohorts alive at once would double it for nothing. Deleted
    # rather than set to None because train_entropies branches on hasattr(ctx, "raw").
    del ctx.raw
    E = {"name": label, "split": split, "cohort_slice": cohort_slice,
         "ctx": ctx, "sh": sh, "net": net, "cblob": cblob,
         "V": V, "foldable": foldable, "rows": rows, "lvl": sh["lvl_lookup"], "C": C, "item_H": item_H,
         "cneg": float(prereg.get("C_NEG", 2.0)), "cache": {}, "n_evals": 0, "t_evals": 0.0,
         "pool": pool_b, "pool_policy": "B", "contract_tag": f"genv:{label}:{split}",
         # the STACK, so a contract signature can key on it instead of on `label`, which is a human's
         # logging string (PAPER_B_TODO 1.19).
         "tower": os.path.basename(snapshot), "conc_ckpt": os.path.basename(conc_ckpt),
         # DERIVED identity of the mask, not a label (2026-08-12 audit). Every resume signature and
         # every output blob downstream signs THIS, so an artifact can always state its own ruler and
         # a sidecar written under a different pool can never be resumed into.
         "pool_contract": pool_contract(pool_b, rows, policy="B")}
    log(f"[{label}] pool contract (DERIVED): policy=B nnz={E['pool_contract']['nnz']} "
        f"per_user={E['pool_contract']['per_user']} fp={E['pool_contract']['fp']}")
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

    CANDIDATE POOL = POLICY B, 2026-08-05 ruling. Was `ctx.va_tr[ch] + ans[cj]` -- policy A (liked
    fold-in only) plus the answered asked items. Two things were wrong with it. The `ans` term made
    the pool a property of the ARM: an arm that asks answerable items deletes guaranteed non-targets
    that the just-informed model would have ranked high, which FREES top-10 slots -- a lift
    proportional to answer rate, landing hardest on the item channel (a concept question masks
    nothing at all, so the rule was channel-dependent in a comparison about channels). And the policy
    A base left the user's unasked dislikes in the pool as permanent unmaskable non-targets, putting
    this harness on a different ruler from the arm-N battery (measured: +0.027 at the cold state).

    Under policy B `ans` is redundant -- an answered item is a rated fold-in-side item and is already
    out of the pool for EVERY arm, whether or not anybody asked about it -- which is the point. The
    assert in `_env_item_state` (no answered item is a held-out target) is kept: it is a leak check
    on the answer simulator, independent of the pool.
    """
    t0 = time.time()
    ctx = E["ctx"]
    ridx = np.asarray(rows, np.int64)
    z_all = genv_latent(E, seq, rows, batch=batch, _caller="genv_eval")
    full = np.full(len(rows), np.nan); tail = np.full(len(rows), np.nan)
    with torch.no_grad():
        for st in range(0, len(rows), batch):
            ch = ridx[st:st + batch]
            S = (z_all[st:st + len(ch)] @ ctx.Wd.T + ctx.bd).numpy().astype(np.float32)
            f, t = ndcg10_from_scores(S, E["pool"][ch], ctx.va_te[ch], ctx.head_mask,
                                      contract=(E["contract_tag"], ch))
            full[st:st + len(ch)] = f; tail[st:st + len(ch)] = t
    E["n_evals"] += 1; E["t_evals"] += time.time() - t0
    return full, tail


def genv_latent(E, seq, rows, batch=500, _caller="genv_latent"):
    """The per-user TOWER+CONCEPT LATENT after folding `seq` -- i.e. exactly the `z` that `genv_eval`
    hands to the decoder, and nothing else.

    This is the state the parametric policy reads (workstream E1, design sheet A.8 part 1). It was
    extracted OUT of `genv_eval` rather than reimplemented beside it, and `genv_eval` now calls it, so
    the policy's state and the scored interview are the same object by construction and cannot drift.
    The extraction is regression-gated on the canonical m=0 snap (0.15686538 on validation) and on the
    tree's banked node bases, which are exact-equality checks.

    Returns a (len(rows), d) float32 tensor, rows in the given order.
    """
    item_ids = [int(i) for ch, i in seq if ch == 0]
    conc_ids = [int(c) for ch, c in seq if ch == 1]
    # `_ans` is no longer a mask (policy B already excludes every rated item); the call is kept
    # because _env_item_state carries the answered-item-is-never-a-target leak assert.
    Zi, _ans = _env_item_state(E, item_ids)
    ctx = E["ctx"]; net = E["net"]
    ridx = np.asarray(rows, np.int64)
    jj = E["posmap"][ridx]                                   # positions inside the ACTIVE cohort
    assert (jj >= 0).all(), f"{_caller} called on rows outside the env's active cohort"
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
    out = torch.empty(len(rows), ctx.d, dtype=torch.float32)
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
            out[st:st + len(ch)] = z
    return out


def genv_eval_multi(E, rows, item_seqs, conc_seqs, batch=500, Zi_items=None, return_Zi=False,
                    return_z=False, score=True):
    """Per-user (full, tail) NDCG@10 where EVERY USER CARRIES ITS OWN ask sequence.

    `genv_eval` scores ONE sequence for a whole cohort. TODO 1.11 needs the opposite shape: 10,000
    users sitting in up to 5,767 DIFFERENT interview states, each asked the SAME candidate question,
    scored in one pass. Calling `genv_eval` once per state would pay ~5.8M small-batch calls for what
    is one 10,000-user pass, so this function exists -- but it is the SAME contract, not a second
    ruler: same policy-B pool, same learned decoder bias, same `ndcg10_from_scores` funnel with the
    mask contract enabled, same certified concept fold with `cap_negatives` over the REVEALED set.
    `genv_eval_multi == genv_eval` bit-for-bit whenever every user carries the same sequence, and
    that is asserted by the caller's G-C gate rather than left as a comment.

      rows        cohort rows to score, inside E's ACTIVE set. THE ORDER IS SIGNIFICANT and must be
                  held fixed across calls: `ndcg10_from_scores`'s mask contract fingerprints each
                  batch's csr slice, so re-ordering users between candidates would trip it. The
                  caller passes state-major order once and reuses it.
      item_seqs   list (len == len(rows)) of ITEM id lists, in ask order, for that user.
      conc_seqs   list (len == len(rows)) of CONCEPT id lists for that user.
      Zi_items    optional precomputed item-fold latents for exactly these rows/sequences. Every
                  CONCEPT candidate leaves the item prefix untouched, so the whole 500-concept block
                  of a budget shares one fold; `return_Zi=True` on the baseline pass hands it over.

    THREE EXACT OPTIMISATIONS, NO APPROXIMATION ANYWHERE:
      (1) state-major user order, so a batch is state-homogeneous;
      (2) inside a batch, users whose recommender input is BIT-IDENTICAL (same item ids, same exact
          levels, same concept ids/values/refusals) are deduplicated before the tower fold, the
          concept fold and the decoder matmul, and the scores are scattered back. Exact because the
          state key carries the exact half-star levels, not the like/dislike bucket. At k=1 a
          500-user batch holds at most 11*11 distinct inputs;
      (3) NDCG stays per-user -- it must, every user has their own mask and their own targets.
    Memory is bounded by `batch`: at most `batch` x 18,359 float32 scores are alive (37 MB at 500).
    """
    t0 = time.time()
    ctx = E["ctx"]; net = E["net"]; lvl = E["lvl"]
    ridx = np.asarray(rows, np.int64)
    n = len(ridx)
    assert len(item_seqs) == n and len(conc_seqs) == n, "per-user sequence lists must match `rows`"
    jj = E["posmap"][ridx]
    assert (jj >= 0).all(), "genv_eval_multi called on rows outside the env's active cohort"

    # ---- per-user recommender input, built once ------------------------------------------------
    ikeys = []                                            # (item ids, exact levels) per user
    for u in range(n):
        ids = tuple(int(i) for i in item_seqs[u])
        d = lvl[int(ridx[u])]
        ikeys.append((ids, tuple(int(d.get(i, UNSEEN_LEVEL)) for i in ids)))
    # An ANSWERED asked item can never be a held-out target (lvl_lookup is built with te excluded).
    # `_env_item_state` asserts this for the single-sequence path; the heterogeneous path must not
    # lose the leak check just because the sequences differ per user.
    ar, ac = [], []
    for u in range(n):
        ids, lvs = ikeys[u]
        for i, L in zip(ids, lvs):
            if L < UNSEEN_LEVEL:
                ar.append(u); ac.append(i)
    ansm = sparse.csr_matrix((np.ones(len(ar), np.float32), (ar, ac)), shape=(n, ctx.ni))
    assert ansm.multiply(ctx.va_te[ridx]).nnz == 0, "an answered item is also a held-out target"

    ckeys = []                                            # (ids, capped values, refusal flags)
    any_conc = False
    for u in range(n):
        cids = [int(c) for c in conc_seqs[u]]
        if not cids:
            ckeys.append(((), (), ()))
            continue
        any_conc = True
        r = int(ridx[u])
        ok = np.asarray([bool(E["foldable"][r, c]) for c in cids])
        vv = np.where(ok, np.asarray([float(E["V"][r, c]) for c in cids], np.float64), 0.0)
        s = -np.minimum(vv, 0.0).sum()                    # cap_negatives over the REVEALED set
        sc = min(1.0, E["cneg"] / max(s, 1e-9))
        vv = np.where(vv < 0, vv * sc, vv)
        ckeys.append((tuple(cids), tuple(float(x) for x in vv),
                      tuple(0.0 if o else 1.0 for o in ok)))

    # ---- item fold (deduplicated), reusable across a whole concept block ------------------------
    if Zi_items is None:
        uniq, order = {}, []
        inv = np.empty(n, np.int64)
        for u, k in enumerate(ikeys):
            j = uniq.get(k)
            if j is None:
                j = len(order); uniq[k] = j; order.append(k)
            inv[u] = j
        seqs = [(np.asarray(k[0], np.int64), np.asarray(k[1], np.int64)) for k in order]
        Zu = fold_items_frozen(ctx.enc, seqs, batch=500)
        Zi_items = Zu[torch.from_numpy(inv)]
    assert Zi_items.shape[0] == n, "Zi_items does not match `rows`"

    # ---- score ---------------------------------------------------------------------------------
    full = np.full(n, np.nan); tail = np.full(n, np.nan)
    # `return_z` hands back the per-user latent the decoder was given -- the parametric policy's state
    # input (design sheet A.8) for a cohort where every user carries a DIFFERENT interview, which is
    # what a policy rollout is. It is read off the same tensor that produced the scores; there is no
    # second fold path.
    # `score=False` skips ONLY the decoder matmul and the NDCG reduction, for the 15 intermediate turns
    # of a rollout where the policy needs its state and nobody needs a number. It cannot change any
    # reported value: the returned full/tail stay NaN and the caller must ask for the score at the turn
    # it reports. The fold path above is untouched.
    z_out = torch.empty(n, ctx.d, dtype=torch.float32) if return_z else None
    with torch.no_grad():
        for st in range(0, n, batch):
            en = min(st + batch, n)
            ch = ridx[st:en]
            # dedup INSIDE the batch on the full recommender input
            bmap, binv = {}, np.empty(en - st, np.int64)
            for u in range(st, en):
                k = (ikeys[u], ckeys[u])
                j = bmap.get(k)
                if j is None:
                    j = len(bmap); bmap[k] = j
                binv[u - st] = j
            pick = np.empty(len(bmap), np.int64)
            for u in range(st, en):
                pick[binv[u - st]] = u
            Zb = Zi_items[torch.from_numpy(pick)]
            if any_conc and any(len(ckeys[int(p)][0]) for p in pick):
                cid = [list(ckeys[int(p)][0]) for p in pick]
                cvl = [list(ckeys[int(p)][1]) for p in pick]
                crf = [list(ckeys[int(p)][2]) for p in pick]
                z = net(Zb, cid, cvl, crf if net.use_refusal else None)
            else:
                z = Zb
            if return_z:
                z_out[st:en] = z[torch.from_numpy(binv)]
            if score:
                Su = (z @ ctx.Wd.T + ctx.bd).numpy().astype(np.float32)
                S = Su[binv]
                f, t = ndcg10_from_scores(S, E["pool"][ch], ctx.va_te[ch], ctx.head_mask,
                                          contract=(E["contract_tag"], ch))
                full[st:en] = f; tail[st:en] = t
    E["n_evals"] += 1; E["t_evals"] += time.time() - t0
    out = (full, tail)
    if return_Zi:
        out = out + (Zi_items,)
    if return_z:
        out = out + (z_out,)
    return out


def genv_intercept_policyA(E, rows, batch=500):
    """The m=0 intercept under POLICY A (liked fold-in only) -- the CERTIFICATION BRIDGE only.

    Gate (a) checks that the empty interview reproduces the tower's own recorded cold k0, and that
    k0 was measured under policy A. Moving the harness to policy B lifts the intercept by ~+0.027,
    which would fail a 0.002-tolerance gate for a reason that has nothing to do with the fold path.
    So the gate keeps its own ruler.

    It takes NO sequence and returns ONE number by construction: no arm can route a curve through
    it, which is what keeps policy A out of the reported battery.
    """
    ctx = E["ctx"]
    ridx = np.asarray(rows, np.int64)
    jj = E["posmap"][ridx]
    assert (jj >= 0).all(), "genv_intercept_policyA called on rows outside the env's active cohort"
    Zi, _ans = _env_item_state(E, [])
    full = np.full(len(rows), np.nan); tail = np.full(len(rows), np.nan)
    with torch.no_grad():
        for st in range(0, len(rows), batch):
            ch = ridx[st:st + batch]; cj = jj[st:st + batch]
            S = (Zi[torch.from_numpy(cj)] @ ctx.Wd.T + ctx.bd).numpy().astype(np.float32)
            f, t = ndcg10_from_scores(S, ctx.va_tr[ch], ctx.va_te[ch], ctx.head_mask)
            full[st:st + len(ch)] = f; tail[st:st + len(ch)] = t
    return float(np.nanmean(full)), float(np.nanmean(tail))


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


def celf_heap_sig(E, rows, pool, objective, prefix):
    """The contract a persisted CELF heap must match before it may be resumed (HARD RULE 11b).

    Everything the heap's contents depend on: the scoring environment, the objective, the fixed
    prefix, the exact candidate pool IN ORDER (the seed order decides tie-breaks) and the exact user
    set. `L` is deliberately NOT in it -- extending a banked order by one more pick is the whole point.

    v2 (2026-08-12, PAPER_B_TODO 1.19). `contract_tag` is a DISPLAY LABEL, `genv:{label}:{split}`,
    chosen by a human for logging -- so it is invariant under a change of candidate-mask policy and was
    never a guard, while simultaneously forcing any job that wants to resume a banked heap to rebuild
    its environment under a literal log string. It is replaced by the things it was standing in for and
    never actually carried: `tower`, `conc_ckpt`, and `mask_pool` -- the derived fingerprint of the mask
    the heap's evaluations actually went through. Bumping `v` makes every v1 heap refuse.

    Applied 2026-08-12 only after the in-flight policy-B rebuild finished, because it invalidates every
    heap already on disk; the items-only and concepts-only rebuilds are cold searches over different
    pools, so they could not have resumed those heaps anyway and lose nothing.
    """
    import hashlib

    def _sha(o):
        return hashlib.sha1(json.dumps(o, separators=(",", ":")).encode()).hexdigest()
    return {"v": 2, "split": E["split"], "objective": objective,
            "tower": E["tower"], "conc_ckpt": E["conc_ckpt"],
            "mask_pool": pool_contract(E["pool"], rows, policy=E["pool_policy"]),
            "prefix": [[int(a), int(b)] for a, b in prefix],
            "n_pool": len(pool), "pool_sha1": _sha([[int(a), int(b)] for a, b in pool]),
            "n_rows": len(rows), "rows_sha1": _sha(sorted(int(r) for r in rows))}


def greedy_certified(E, rows, pool, L, objective, tag, log_every=25, prefix=(), on_pick=None,
                     heap_path=None, sweep_path=None):
    """Lazy-greedy (CELF) best STATIC sequence of length L from pool=[(chan,id),...], maximising the
    cohort-mean NDCG@10 of `objective` ('full' or 'tail'). Returns (seq, gains, n_evals).

    `prefix` = a FIXED, already-asked head that every candidate sequence carries and that the search
    may not change. With prefix=() (the default, and what greedy_main uses) this is bit-identical to
    the original: every genv_mean key below reduces to the old one. The returned `seq` is the L NEW
    steps only -- the caller owns the concatenation, so a partial written to disk is unambiguous.
    `on_pick(seq, gains, nev)` fires after every pick, for HARD RULE 11(a) incremental persistence.

    `heap_path` (2026-08-11) PERSISTS THE LAZY HEAP, which pays the standing debt that a banked order
    could only ever be EXTENDED by paying the whole cold seed pass again (~485 evals ~= 1.8 h at 10k
    users) for what costs ~55 evals in process. The heap plus (base, cur, seq, gains, nev) is the
    complete state of the search, so it is written next to the picks after the seed pass and after
    EVERY pick, under a contract signature (`celf_heap_sig`). A resume with a DIFFERENT contract is
    ignored loudly, never merged. Two invariants make the resumed search bit-identical to a cold one:
    every heap entry carries a distinct question, so `heappop` order is fully determined by the tuple
    ordering and not by the internal array layout; and the seed order (which sets those tuples) is
    part of the signature. With `heap_path=None` NOTHING below changes -- not one extra evaluation,
    not one different branch.
    """
    import heapq
    assert objective in ("full", "tail"), objective
    j = 0 if objective == "full" else 1
    pre = [tuple(q) for q in prefix]
    assert not (set(pre) & set(tuple(q) for q in pool)), \
        "a prefix question is still in the candidate pool -- it would be asked twice"
    t0 = time.time()
    sig = celf_heap_sig(E, rows, pool, objective, pre) if heap_path else None

    def _save(heap, seq, gains, base, cur, nev):
        if not heap_path:
            return
        blob = {"sig": sig, "tag": tag, "base": float(base), "cur": float(cur), "nev": int(nev),
                "seq": [[int(a), int(b)] for a, b in seq], "gains": [float(g) for g in gains],
                "heap": [[float(a), [int(q[0]), int(q[1])], int(u)] for a, q, u in heap]}
        os.makedirs(os.path.dirname(heap_path) or ".", exist_ok=True)
        with open(heap_path + ".tmp", "w") as fh:
            json.dump(blob, fh)
        os.replace(heap_path + ".tmp", heap_path)

    def _save_sweep(heap, base, nev):
        """PERSIST THE FULL PER-CANDIDATE SWEEP BEFORE THE FIRST POP (2026-08-21).

        The heap sidecar is NOT a gain table. `C1_GAIN_TABLE_AUDIT.md` proved why: when a candidate
        is first popped its key is still its node-state gain, and `heappop` returns the maximum, so
        re-evaluation overwrites *exactly the top of the ranking*. At a node whose tail re-evaluated
        D distinct candidates, rank 1 survives (it is the recorded splitter gain), **ranks 2..D+1 are
        irrecoverable** (median D = 192), and only ranks D+2.. remain exact. That left a top-m blend
        with m >= 2 buildable at just 42 of `coldtree_v2`'s 164 nodes.

        Right here, immediately after the seed pass and before any pop, EVERY entry is still at
        level 0 -- an exact gain at this node's own state for every candidate in the pool. Writing it
        now costs one ~16 KB array per node and no evaluations at all; recovering it later costs a
        repair campaign. Same lesson as `seven-states-count-not-composition`: persist the sweep at
        step 0, before the first pop.

        The file is written ONCE and never rewritten, so unlike the heap sidecar it cannot decay.
        """
        if not sweep_path:
            return
        cand = np.asarray([[int(q[0]), int(q[1])] for _a, q, _u in heap], np.int32)
        gain = np.asarray([-float(a) for a, _q, _u in heap], np.float64)
        assert len(cand) == len(pool), (
            f"{tag}: sweep holds {len(cand)} of {len(pool)} candidates -- the seed pass did not "
            f"evaluate the whole pool, so this is not a complete gain table")
        assert all(int(u) == 0 for _a, _q, u in heap), (
            f"{tag}: a heap entry is not at level 0 -- _save_sweep ran AFTER a pop and the top of "
            f"the ranking is already destroyed")
        os.makedirs(os.path.dirname(sweep_path) or ".", exist_ok=True)
        np.savez_compressed(sweep_path + ".tmp.npz", cand=cand, gain=gain,
                            base=np.float64(base), n_evals=np.int64(nev),
                            prefix=np.asarray([[int(a), int(b)] for a, b in pre], np.int32),
                            objective=np.str_(objective), tag=np.str_(tag),
                            sig=np.str_(json.dumps(sig, sort_keys=True, default=str)))
        os.replace(sweep_path + ".tmp.npz", sweep_path)
        log(f"[greedy {tag}] SWEEP banked: {len(cand)} exact node-state gains "
            f"(best {gain.max():+.5f}) -> {os.path.basename(sweep_path)}")

    banked = None
    if heap_path and os.path.exists(heap_path):
        try:
            blob = json.load(open(heap_path))
        except Exception as e:                                   # a truncated sidecar is not a resume
            blob = None
            log(f"[greedy {tag}] heap sidecar unreadable ({e}) -- ignored, seeding cold")
        if blob is not None and blob.get("sig") == sig:
            banked = blob
        elif blob is not None:
            log(f"[greedy {tag}] heap sidecar {os.path.basename(heap_path)} carries a DIFFERENT "
                f"contract -- IGNORED (nothing merged), seeding cold")

    if banked is not None:
        base = float(banked["base"]); cur = float(banked["cur"]); nev = int(banked["nev"])
        seq = [(int(a), int(b)) for a, b in banked["seq"]]
        gains = [float(g) for g in banked["gains"]]
        heap = [(float(a), (int(q[0]), int(q[1])), int(u)) for a, q, u in banked["heap"]]
        heapq.heapify(heap)
        log(f"[greedy {tag}] RESUMED from {os.path.basename(heap_path)}: {len(seq)} pick(s) banked, "
            f"{len(heap)} heap entries, {nev} evals already paid -> extending to L={L} "
            f"(a cold restart would re-pay the {len(pool)}-candidate seed pass)")
        if sweep_path and not os.path.exists(sweep_path):
            log(f"[greedy {tag}] WARNING: resumed from a heap with {len(seq)} pick(s) and no banked "
                f"sweep -- the top of this node's ranking is already destroyed and CANNOT be "
                f"recovered here. This node will have no gain table.")
        if len(seq) >= L:
            log(f"[greedy {tag}] banked order already has {len(seq)} >= L={L} picks -- returning its "
                f"first {L} (greedy prefixes are exact: pick k does not depend on L)")
            return seq[:L], gains[:L], nev
    else:
        base = genv_mean(E, pre, rows)[j]
        heap = []; nev = 0
        for q in pool:
            v = genv_mean(E, pre + [q], rows)[j]; nev += 1
            heapq.heappush(heap, (-(v - base), q, 0))
            if nev % log_every == 0:
                el = time.time() - t0
                log(f"[greedy {tag}] seed {nev}/{len(pool)} ({el / 60:.1f}m, {el / nev:.2f}s/eval)")
        seq = []; gains = []; cur = base
        _save_sweep(heap, base, nev)          # BEFORE the first pop -- every entry still level 0
        _save(heap, seq, gains, base, cur, nev)
    while len(seq) < L and heap:
        neg, q, upd = heapq.heappop(heap)
        if upd == len(seq):
            seq.append(q); gains.append(-neg); cur += -neg
            log(f"[greedy {tag}] pick {len(seq)}/{L} = {'concept' if q[0] else 'item'} {q[1]} "
                f"gain {-neg:+.5f} -> {cur:.5f} (nev={nev}, {(time.time() - t0) / 60:.1f}m)")
            _save(heap, seq, gains, base, cur, nev)
            if on_pick is not None:
                on_pick(list(seq), list(gains), nev)
        else:
            v = genv_mean(E, pre + seq + [q], rows)[j]; nev += 1
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
    # The tower's k0 was certified under POLICY A, so the bridge check is measured under policy A.
    # The REPORTED intercept below it is the policy-B one, which is what every arm is scored on.
    a_full, a_tail = genv_intercept_policyA(Eb, build_rows)
    i_full, i_tail = genv_mean(Eb, [], build_rows)
    ref = float(tblob["k0"])
    ok = abs(a_full - ref) <= args.intercept_tol
    log(f"[gate a] m=0 empty interview on {len(build_rows)} val users, POLICY A (bridge): "
        f"full={a_full:.5f} tail={a_tail:.5f} | tower's recorded cold k0={ref:.5f} | "
        f"delta={a_full - ref:+.5f} tol={args.intercept_tol} -> {'PASS' if ok else 'FAIL'}")
    log(f"[gate a] m=0 REPORTED intercept, POLICY B (the battery pool): {i_full:.5f}/{i_tail:.5f} "
        f"(A->B pool shift {i_full - a_full:+.5f} full / {i_tail - a_tail:+.5f} tail)")
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
           "build_tail": int(build_rows[-1]),
           # THE FIELD THE 2026-08-12 AUDIT FOUND MISSING. Without it this signature was invariant
           # under a change of candidate-mask POLICY, so a policy-B re-run writing to an existing
           # policy-A path (`greedy_static_i26c.partial.json`) would have matched it and resumed the
           # policy-A picks in silence. Derived from the mask, never typed.
           "pool_contract": pool_contract(Eb["pool"], build_rows, policy=Eb["pool_policy"])}
    part = json.load(open(PARTIAL)) if os.path.exists(PARTIAL) else {}
    if not resume_ok(part.get("sig"), sig, log, os.path.basename(PARTIAL)):
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
                        "token. CANDIDATE POOL = POLICY B (2026-08-05 ruling): every rated "
                        "fold-in-side item, likes AND dislikes, fixed per user before the interview "
                        "starts, identical for every arm/channel/budget and independent of what was "
                        "asked; the target matrix is never modified. An asked item is NEVER removed "
                        "from the candidate list because it was asked."),
           "pool_policy": "B (arm_n.build_pool equivalent)",
           # the DERIVED stamp: nnz/per-user/fingerprint of the mask actually used, on the build and
           # the reporting cohorts. An artifact that cannot state its own ruler is one migration away
           # from needing another provenance audit (2026-08-12).
           "pool_contract_build": pool_contract(Eb["pool"], build_rows, policy=Eb["pool_policy"]),
           "pool_contract_eval": pool_contract(Ee["pool"], eval_rows, policy=Ee["pool_policy"]),
           "gate_m0_policyA_bridge": {"full": a_full, "tail": a_tail},
           "selection_users": f"{len(build_rows)} canonical VALIDATION users",
           "reporting_users": f"{len(eval_rows)} {eval_split} users",
           "n_build_users": len(build_rows), "n_eval_users": len(eval_rows),
           "eval_split": eval_split, "objective": args.objective,
           "item_bank": args.item_bank, "conc_bank": args.conc_bank,
           "n_items_pool": len(item_pool), "n_concepts_pool": len(conc_pool),
           # the BRIDGE gate is on policy A (the pool the tower's k0 was certified under); the
           # reported m=0 row is the policy-B intercept in "intercept" below.
           "gate_m0_full_policyA": a_full, "gate_m0_tail_policyA": a_tail,
           "gate_m0_ref_tower_k0": ref, "gate_m0_pass": bool(ok),
           "m0_policyB_full": i_full, "m0_policyB_tail": i_tail,
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
    # The sclite rung's concept fold. ml-25m keeps cfold_signed_best.pt exactly. goodreads gets
    # its own certified net -- the ml-25m file carries a 1,031-concept embedding and Goodreads has
    # 2,408, so it was being loaded with the CHECKPOINT's bank rather than the DATASET's, which
    # silently reinterprets every concept id.
    ap.add_argument("--sclite_ckpt", default=os.path.join(
        _ROOT, ".cache", "instrument",
        {"ml-25m": "cfold_signed_best.pt", "goodreads": "cvg_ep10.pt"}[_DS]))
    # The scfull rung's full signed encoder. cfull_signed_best.pt is pure ml-25m (18,359 items,
    # 1,031 concepts) and would crash a Goodreads run mid-flight on an item_emb size mismatch.
    # There is no Goodreads counterpart -- training one is a separate job -- so point Goodreads at
    # a name that does not exist and let the os.path.exists guard below skip the rung cleanly.
    ap.add_argument("--scfull_ckpt", default=os.path.join(
        _ROOT, ".cache", "instrument",
        {"ml-25m": "cfull_signed_best.pt",
         "goodreads": "cfull_signed_best_goodreads.pt"}[_DS]))
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
    # was a local rebind to the hardcoded ml-25m filename, shadowing the module constant
    CONCSTATS = os.path.join(_ROOT, ".cache", "instrument", f"signed_train_concstats{_SFX}.npz")
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
            # BANK-SIZE GUARD. This used to size the net from len(blob["tags"]) -- the
            # CHECKPOINT's bank -- so an ml-25m fold (1,031 concepts) loaded happily inside a
            # Goodreads run (2,408) and every concept id meant something else. Size from the
            # DATASET and refuse anything that disagrees. Checkpoints written by
            # concept_train_v2 carry no "tags" list, so fall back to the embedding shape.
            _nc_ck = (len(blob["tags"]) if "tags" in blob
                      else int(blob["net"]["emb.weight"].shape[0]))
            _nc_ds = len(sh["tags"])
            assert _nc_ck == _nc_ds, (
                f"concept-fold bank mismatch: {os.path.basename(args.sclite_ckpt)} has "
                f"{_nc_ck} concepts, the {_DS} bank has {_nc_ds}")
            net = ConceptFoldNet(_nc_ds, d=ctx.d, h=blob["hidden"],
                                 use_refusal=bool(blob.get("use_refusal", False)))
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
