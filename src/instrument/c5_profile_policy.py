r"""PAPER C / STAGE C5 -- the PROFILE-PARAMETERISED emitter: predict the gain PROFILE over candidates,
then CONSTRUCT the emission as a convex combination of real bank entries.

Design sheet: `docs/design/DS_C5_PROFILE_POLICY.md`.  Predecessor + every carried caveat:
`docs/design/DS_C1_FINAL.md` + `docs/results/C1_FINAL.md`.  Scoping (artifact locations):
`docs/design/C1_FINAL_SCOPING.md`.  This module IMPORTS `c1_final` -- the bank loader, the descent,
the label contract, the sweep gate, the batcher, the two blocking gates and reading 1 are that
file's, unchanged and unforked.

WHY THIS RE-PARAMETERISATION EXISTS
-----------------------------------
C1-final regressed the policy onto a 128-d blend DIRECTION: one supervised vector per state, 72
states.  It fits (cos 0.970 at trained states) and does not extrapolate (-0.119 at held-out states),
and the one statistic that looked like extrapolation was matched by the UNTRAINED warm start.  That
is action-shaped supervision, and Paper B already measured which shapes generalise on this stack:

  * the per-question REACTION head generalised to unseen questions (`HELDOUT_QUESTIONS.md`, 0.98167);
  * the ACTION policy did not (`P1_HELDOUT_ACTIONS.md`, `P1B_NEVER_TAUGHT.md`).

C5 makes the emission reaction-shaped.  The supervised object is the per-candidate gain PROFILE
`s(q | state)` -- which the incumbent head ALREADY computes, one scalar per candidate -- and the
emission is then *derived*, not regressed:

    p_hat = softmax(s / tau_m)  over the state's own (unasked) table
    z_hat = sum_{j <= m} p_j e_{q_j}     over the predicted top-m, p renormalised on that top-m

FOUR PROPERTIES, EACH ASSERTED IN CODE RATHER THAN CLAIMED IN PROSE
------------------------------------------------------------------
(a) SUPERVISION WIDTH.  ~1,000 numbers per state (the state's whole exact table) instead of 128.
    Asserted: `Pstar` is dense over the unasked pool and its row support is `n_pool - depth`.
(b) NO OFF-MANIFOLD DRIFT, BY CONSTRUCTION.  `z_hat` is a CONVEX COMBINATION of real bank entries:
    weights are a softmax, hence non-negative and summing to one, so `||z_hat|| <= max_j ||e_{q_j}||`
    by the triangle inequality.  The 38x scale blow-up the pilot suffered, and the `lambda`-weighted
    scale term C1-final bolted on to contain it, are BOTH removed by construction -- C5 carries no
    scale term at all.  `assert_convex()` checks this on every emission it computes.
    ! HONEST CAVEAT, and it is a place the brief did not survive contact with the code: convexity
      bounds the norm ABOVE only.  A convex combination of five vectors can be shorter than every
      one of them (they can cancel).  So the registered assert is the upper bound; where `||z_hat||`
      sits inside `[min_j ||e_j||, max_j ||e_j||]` is REPORTED, never asserted.
(c) EXACT NESTING OF THE DISCRETE POLICY.  As `tau -> 0` the softmax over the top-m becomes one-hot
    on the argmax, so `z_hat -> e_{argmax_q s(q)}` -- the emission degenerates to the policy's own
    picked question.  `selftest_nesting()` checks the limit numerically; the run additionally
    reports how often that argmax IS the tree's splitter at trained states.
(d) DIFFERENTIABILITY.  topk + softmax + a weighted sum of constant bank rows: the whole emission is
    differentiable in `head.q` / `enc` (the topk INDICES are piecewise constant, as in any
    straight-through top-k; the WEIGHTS carry the gradient).

AND ONE MORE, WHICH IS THE REASON THE OBJECTIVE IS WORTH REGISTERING: with `tau_t -> 0` the target
distribution `p*` collapses onto the argmax-gain candidate = the tree's splitter, and
`KL(p* || p_hat)` becomes EXACTLY the one-hot cross-entropy the incumbent distillation uses.  C5's
objective therefore NESTS C1-final's replay objective as a limit; `selftest_tau_t_zero_is_ce()`
asserts it to 1e-5.

WHAT IS INHERITED VERBATIM FROM C1-FINAL, AND WHY
------------------------------------------------
  * splits -- primary `root.like.*` (15 held out / 57 trained), secondary depth >= 5 (30 / 42).
    Random state splits stay FORBIDDEN: node cohorts nest and a random split leaks.
  * warm start `m7dj_policy.pt`; TRAIN canonical snap 0.15831 (NOT the validation 0.15687); the
    1,004-entry phi bank asserted; contract-signed output; atomic writes; per-epoch flush;
    HARD RULE 10 dirty-src guard.
  * `head.k` / `head.b` FROZEN.  Same ruling as C1-final and for the same reason: the bank
    `e_q = [k(phi_q), b(phi_q)]` is the space the registered cosines live in, and the warm-start null
    is measured against it.  If the loss could move it, reading 1 would compare numbers from two
    different banks.  (In C5 there is a second reason: the emission is built OUT of that bank, so a
    moving bank would make "convex combination of real bank entries" a moving claim.)
  * a HELD-OUT STATE RECEIVES NO LOSS AT ALL -- not the profile KL, not the incumbent CE.
  * readings 1 (generalisation, WITH the warm-start null that turned C1-final's headline), 2 (fit),
    3 (NO-COLLAPSE, blocking), 4 (RENDERABILITY, blocking), 5 (off-grid geometry).

WHAT IS NEW
-----------
  * READING 6, PROFILE-LEVEL: Spearman and NDCG@5 of the predicted profile against the measured
    gains at HELD-OUT states, against a popularity-prior baseline, a global-mean-profile baseline
    and the untrained warm start.  This is the reading the re-parameterisation exists to move: it is
    scored on the object C5 supervises, at states it never saw.
  * `--space {kphi,tower}`.  The PROFILE is space-free -- it is a distribution over candidates, and
    the head's scores do not change -- so the flag moves only the READ-OUT: which embedding the
    convex combination is taken in, and therefore which space reading 1/4/5 are measured in.
    (a) `kphi`  : `e_q = [k(phi_q), b(phi_q)]`, C1-final's space, the DEFAULT so the comparison with
        C1-final is like-for-like.
    (b) `tower` : `e_q` = the centred, popularity-stripped decoder row `Ew[i]` for an item and
        `build_concept_dirs`' direction `d_c[c]` for a concept -- the space `TowerBridge` and the
        renderer's target actually live in.  First-class: because the profile is space-free, BOTH
        spaces are computed in every run from ONE cache of top-m logits, and `--space` only names
        which one is the headline.

FIREWALL.  Train-side states only, on the 85,289-user cold bank.  Nothing here can build a test
environment; the 10,000 canonical TEST users are not built, not loaded, not touched.
"""
from __future__ import annotations

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
sys.path.insert(0, _HERE)
if "--threads" not in sys.argv:                        # 4-physical-core box, shared with other work
    sys.argv += ["--threads", "4"]
_NT = sys.argv[sys.argv.index("--threads") + 1]
os.environ["OMP_NUM_THREADS"] = _NT
os.environ.setdefault("MKL_NUM_THREADS", _NT)
os.environ.setdefault("OPENBLAS_NUM_THREADS", _NT)

import argparse
import copy
import json
import time

import numpy as np
import torch

from policy_net import Policy, build_phi, log, paired_bootstrap
from c0_geometry import (PAPERC, N_BANK, sha256, git_block, savez_atomic, index_map_assert,
                         bank_vectors, dist, wilson)
from c1_pilot import build_targets, state_geometry, geom_summary
import c1_final as C1
from c1_final import (TREE_JSON, WARM, TOTAL_L, SEED, N_EXT, TRAIN_M0_FULL,
                      load_cold_bank, descend, verify_labels, load_sweeps, make_batch,
                      snap_argmax, top1, reading1)

# ------------------------------------------------------------------ DECLARED CONSTANTS (not tuned)
TAU_M = 1.0          # the head's own logit scale. Fixing it at 1.0 is what makes the blend KL and
                     # the incumbent replay CE live on ONE temperature, and is what makes the
                     # tau_t -> 0 nesting an identity rather than an approximation.
M_BLEND = 5          # blend width, C1-final's registered m
TAU_FLOOR = 1e-12    # numerical floor on any temperature
W1_STAR_SOURCE = ("C1_FINAL.md: median top-1 weight of the gain-PROPORTIONAL top-5 blend over the "
                  "72 searched states = 0.2254. Recomputed here from the sweeps, never hard-coded.")


# ============================================================================ profile targets
def gain_rows(states, n_pool):
    """Per state: the exact measured gain over its own table, dense over the pool, plus the mask.

    `mask[i, q]` is True where q is ASKED in state i's prefix -- exactly the set the policy's own
    argmax excludes and exactly the set the sweep does not measure. Everything else is measured:
    `load_sweeps` has already asserted the table is the whole available pool, `1000 - len(prefix)`.
    """
    S = len(states)
    G = np.full((S, n_pool), np.nan, np.float64)
    asked = np.zeros((S, n_pool), bool)
    for i, s in enumerate(states):
        for q in s["pool_prefix_idx"]:
            asked[i, int(q)] = True
        for q, g in s["gains"].items():
            G[i, int(q)] = g
    got = np.isfinite(G)
    assert (got == ~asked).all(), "the measured table is not exactly the unasked pool"
    G[~got] = 0.0
    return G, asked


def _softmax_rows(x, mask, tau):
    """softmax(x / tau) over the unmasked entries of each row; 0 on masked entries."""
    z = np.where(mask, -np.inf, x / max(tau, TAU_FLOOR))
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    e[mask] = 0.0
    return e / e.sum(1, keepdims=True)


def calibrate_tau_t(states, m, lo=1e-8, hi=1e3, iters=200):
    r"""THE DECLARED RULE for `tau_t`, applied ONCE to the targets alone -- no model, no held-out
    feedback, nothing swept.

    "Choose `tau_t` so the target's entropy matches the measured profile's own spread."  Operationally,
    and this is the whole rule: `p*` RESTRICTED to the top-m and renormalised must reproduce, in the
    MEDIAN over states, the top-1 weight of the gain-PROPORTIONAL blend C1-final used as its target.
    Restriction-and-renormalisation of a softmax IS the softmax on the subset, so the criterion is a
    scalar equation in `tau` on the top-m gains only, and it is monotone (tau -> 0 gives w1 -> 1,
    tau -> inf gives w1 -> 1/m), so a bisection is exact.

    Two things this buys:
      * `p*`'s peakedness is fixed by the MEASUREMENT, not by a knob;
      * at `tau_t` the top-m of `p*` is (in the median) C1-final's own blend, so the two runs'
        targets are the same object read at two widths -- the comparison is like-for-like.
    """
    tops = []
    for s in states:
        asked = set(s["pool_prefix_idx"])
        g = np.asarray([v for q, v in s["gains"].items() if q not in asked], np.float64)
        g = np.sort(g)[::-1][:m]
        tops.append(g)
    Gm = np.stack(tops)                                    # (S, m), descending
    Wp = np.maximum(Gm, 0.0)
    assert (Wp.sum(1) > 0).all()
    Wp = Wp / Wp.sum(1, keepdims=True)
    w1_star = float(np.median(Wp[:, 0]))                   # the measured spread, in one number

    def med_w1(tau):
        z = (Gm - Gm[:, :1]) / max(tau, TAU_FLOOR)
        e = np.exp(z)
        return float(np.median(e[:, 0] / e.sum(1)))

    assert 1.0 / m < w1_star < 1.0, f"median proportional w1 {w1_star} outside (1/m, 1)"
    a, b = lo, hi
    for _ in range(iters):                                 # med_w1 is DECREASING in tau
        mid = float(np.sqrt(a * b))
        if med_w1(mid) > w1_star:
            a = mid
        else:
            b = mid
    tau = float(np.sqrt(a * b))
    return tau, w1_star, med_w1(tau), Wp


def build_profile_targets(states, n_pool, tau_t):
    """`p* = softmax(max(gain, 0) / tau_t)` over the state's own exact table."""
    G, asked = gain_rows(states, n_pool)
    Pstar = _softmax_rows(np.maximum(G, 0.0), asked, tau_t)
    assert (Pstar >= 0).all()
    assert np.abs(Pstar.sum(1) - 1.0).max() < 1e-9
    assert (Pstar[asked] == 0).all()
    supp = (~asked).sum(1)
    ent = -(np.where(Pstar > 0, Pstar * np.log(np.maximum(Pstar, 1e-300)), 0.0)).sum(1)
    return (Pstar.astype(np.float32), asked, ent,
            {"n_states": len(states),
             "supervision_width_per_state": dist(supp),
             "target_entropy_nats": dist(ent),
             "target_effective_support_expH": dist(np.exp(ent)),
             "target_top1_weight_full_table": dist(Pstar.max(1)),
             "n_positive_gain_candidates": dist((G > 0).sum(1))})


# ============================================================================ the emission
def topm_weights(v, tau):
    """Softmax weights over the top-m logits `v` (B, m), computed shift-first so no temperature can
    overflow. `tau -> 0` gives exactly one-hot on column 0 when v is sorted descending."""
    z = (v - v[..., :1]) / max(float(tau), TAU_FLOOR)
    if isinstance(v, torch.Tensor):
        return torch.softmax(z, dim=-1)
    z = z - z.max(-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(-1, keepdims=True)


def emit_convex(v, idx, Eemb, tau):
    """z_hat = sum_j p_j e_{q_j} over the top-m. `v` (B,m) logits descending, `idx` (B,m) pool ids."""
    w = topm_weights(v, tau)
    if isinstance(v, torch.Tensor):
        return (w.unsqueeze(-1) * Eemb[idx]).sum(1), w
    return np.einsum("bm,bmd->bd", w, Eemb[idx]), w


def assert_convex(w, idx, Eemb, zh, tol=1e-4):
    """PROPERTY (b), checked rather than claimed. Returns the norm-range report."""
    w = np.asarray(w, np.float64); zh = np.asarray(zh, np.float64)
    assert (w >= -tol).all(), f"a blend weight is negative (min {w.min():g}) -- not a convex combination"
    assert np.abs(w.sum(1) - 1.0).max() < tol, "the blend weights do not sum to 1"
    n = np.linalg.norm(zh, axis=1)
    ing = np.linalg.norm(Eemb[idx], axis=2)                       # (B, m)
    nmax, nmin = ing.max(1), ing.min(1)
    assert (n <= nmax * (1 + 1e-3) + 1e-6).all(), (
        "||z_hat|| exceeds the largest ingredient norm -- the triangle inequality is violated, so "
        "this is NOT a convex combination of bank entries")
    return {"norm_zhat": dist(n), "ingredient_norm_max": dist(nmax), "ingredient_norm_min": dist(nmin),
            "frac_inside_ingredient_norm_range": float(np.mean((n >= nmin) & (n <= nmax))),
            "frac_at_or_below_max": float(np.mean(n <= nmax * (1 + 1e-6))),
            "note": "convexity bounds the norm ABOVE only; a convex combination can be shorter than "
                    "every ingredient, so the lower end of the range is REPORTED, not asserted"}


# ============================================================================ the top-m cache
def collect_topm(pol, bank, phi_ext, srows, spos, n_pool, kc, bc, m, tau_m, batch=2048,
                 want_profile=True):
    """ONE forward pass per state, caching the top-m (logit, pool index) pairs and the mean predicted
    profile.  Every emission at every temperature, in every space, is then a pure array op on this
    cache -- which is why `--space` can report BOTH spaces and the emission temperature can be
    calibrated after the fact without a second pass over 510k states.
    """
    S = len(srows)
    cache = []
    prof = np.zeros((S, n_pool), np.float64) if want_profile else None
    kcT = kc.T.contiguous()
    with torch.no_grad():
        for i in range(S):
            r, p = srows[i], spos[i]
            vs, ix = [], []
            for s in range(0, len(r), batch):
                sl = slice(s, min(s + batch, len(r)))
                z, hp, ho, hm, pf, am, _y = make_batch(bank, phi_ext, r[sl], p[sl], n_pool)
                u = pol.head.q(pol.enc(z, hp, ho, hm, pf))
                lg = (u @ kcT + bc[None, :]).masked_fill(am, -1e20)
                v, j = torch.topk(lg, m, dim=1)
                vs.append(v.numpy().astype(np.float32)); ix.append(j.numpy().astype(np.int32))
                if want_profile:
                    prof[i] += torch.softmax(lg / tau_m, 1).double().sum(0).numpy()
            cache.append((np.concatenate(vs), np.concatenate(ix)))
            if want_profile:
                prof[i] /= max(len(r), 1)
    return cache, prof


def emission_from_cache(cache, Eemb, Tt, mu, tau, chunk=4096):
    """The per-state emission (cohort mean of the per-user convex combination) and the per-user
    cosines to the state's own target. Same dict shape `c1_final.reading1` consumes."""
    S = len(cache)
    d = Eemb.shape[1]
    zbar = np.zeros((S, d), np.float32)
    cos_own = np.zeros(S); disp = np.zeros(S); ratio_med = np.zeros(S)
    w1 = np.zeros(S); per_user = {}
    conv = []
    Tn = Tt / (np.linalg.norm(Tt, axis=1, keepdims=True) + 1e-30)
    Tmr = np.linalg.norm(Tt - mu[None, :], axis=1)
    E64 = Eemb.astype(np.float64)
    for i in range(S):
        v, idx = cache[i]
        n = len(v)
        Z = np.empty((n, d), np.float32)
        W = np.empty((n, v.shape[1]), np.float32)
        for s in range(0, n, chunk):
            sl = slice(s, min(s + chunk, n))
            zh, w = emit_convex(v[sl].astype(np.float64), idx[sl].astype(np.int64), E64, tau)
            Z[sl] = zh.astype(np.float32); W[sl] = w.astype(np.float32)
        if i < 4 or (i % 8) == 0:                       # spot-checked across the state list
            conv.append(assert_convex(W[:chunk], idx[:chunk].astype(np.int64), Eemb, Z[:chunk]))
        Zn = Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-30)
        cu = Zn @ Tn[i]
        per_user[i] = cu.astype(np.float32)
        cos_own[i] = float(cu.mean())
        zbar[i] = Z.mean(0)
        disp[i] = float((Zn @ (zbar[i] / (np.linalg.norm(zbar[i]) + 1e-30))).mean())
        ratio_med[i] = float(np.median(np.linalg.norm(Z - mu[None, :], axis=1) / max(Tmr[i], 1e-30)))
        w1[i] = float(W[:, 0].mean())
    return {"zbar": zbar, "cos_own": cos_own, "within_state": disp,
            "norm_ratio_median": ratio_med, "per_user": per_user, "mean_top1_weight": w1,
            "convexity": conv[0] if conv else None,
            "convexity_n_states_checked": len(conv)}


def calibrate_tau_emit(cache, trained, w1_star, lo=1e-6, hi=1e6, iters=200):
    """The ONE scalar C5 fits at read-out time, and it is fitted on TRAINED states only.

    The head's logit SCALE is arbitrary -- nothing in the objective pins it -- so `softmax(s/1.0)`
    over the top-5 can be arbitrarily peaked or arbitrarily flat, and with it the emission slides
    between "exactly the argmax bank entry" and "the uniform centroid of five".  `tau_emit` is
    therefore declared by the same rule as `tau_t`: match the median top-1 blend weight to the
    MEASURED `w1_star`.  It is a single global scalar, shared by every state including the held-out
    ones, so it cannot carry per-state information and cannot manufacture generalisation; it is
    disclosed as the only fitted read-out constant in the run.
    """
    V = np.stack([np.asarray(cache[int(i)][0], np.float64).mean(0) for i in trained])   # (T, m)

    def med_w1(tau):
        return float(np.median(topm_weights(V, tau)[:, 0]))

    if med_w1(hi) > w1_star:                       # logits already flatter than the target: no fix
        return float(hi), med_w1(hi), False
    if med_w1(lo) < w1_star:
        return float(lo), med_w1(lo), False
    a, b = lo, hi
    for _ in range(iters):
        mid = float(np.sqrt(a * b))
        if med_w1(mid) > w1_star:
            a = mid
        else:
            b = mid
    tau = float(np.sqrt(a * b))
    return tau, med_w1(tau), True


# ============================================================================ profile-level readings
def _rank(x):
    """Descending ranks with TIES AVERAGED. The tie handling is load-bearing: `max(gain, 0)` puts
    roughly half the pool at exactly 0, and a naive argsort would invent an ordering inside that
    block and hand every baseline a spurious correlation."""
    x = np.asarray(x, np.float64)
    order = np.argsort(-x, kind="stable")
    r = np.empty(len(x), np.float64)
    s = x[order]
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        r[order[i:j + 1]] = 0.5 * (i + j)
        i = j + 1
    return r


def spearman(a, b):
    ra, rb = _rank(a), _rank(b)
    ra = ra - ra.mean(); rb = rb - rb.mean()
    den = float(np.linalg.norm(ra) * np.linalg.norm(rb))
    return float(ra @ rb / den) if den > 0 else 0.0


def ndcg_at_k(score, rel, k=5):
    """Relevance = max(gain, 0); the ideal list is the true top-k of the same table."""
    rel = np.maximum(np.asarray(rel, np.float64), 0.0)
    disc = 1.0 / np.log2(np.arange(2, k + 2))
    top = np.argsort(-np.asarray(score, np.float64), kind="stable")[:k]
    dcg = float(rel[top] @ disc)
    ideal = np.sort(rel)[::-1][:k]
    idcg = float(ideal @ disc[:len(ideal)])
    return dcg / idcg if idcg > 0 else 0.0


def profile_readings(prof, G, asked, held, trained, popv, label, k=5):
    """READING 6. The predicted profile against the MEASURED gains, at the given states, with two
    declared baselines and the same two rank statistics on each.

    Baselines
      * POPULARITY PRIOR -- `policy_p1.py`'s declared dumb comparator, one scale across the two
        channels: the number of train users who can answer the question.
      * GLOBAL MEAN PROFILE -- the mean of `p*` over the TRAINED states, masked to the scored state's
        own unasked set. This is the "no state information at all" predictor and is the baseline the
        reaction-shaped claim has to beat: if a single fixed profile does as well, the head is not
        conditioning on the state.
    """
    gm = np.zeros(G.shape[1], np.float64)
    for i in trained:
        gm += np.where(asked[i], 0.0, np.maximum(G[i], 0.0))
    gm /= max(len(trained), 1)
    out = {}
    for name, sel in (("held_out", np.asarray(held, np.int64)),
                      ("trained", np.asarray(trained, np.int64))):
        rows = {"model": [], "popularity": [], "global_mean_profile": []}
        nd = {"model": [], "popularity": [], "global_mean_profile": []}
        for i in sel:
            u = ~asked[i]
            g = G[i][u]
            for kk, sc in (("model", prof[i][u]), ("popularity", popv[u]),
                           ("global_mean_profile", gm[u])):
                rows[kk].append(spearman(sc, g))
                nd[kk].append(ndcg_at_k(sc, g, k))
        out[name] = {
            "n_states": int(len(sel)),
            "spearman": {kk: dist(v) for kk, v in rows.items()},
            f"ndcg@{k}": {kk: dist(v) for kk, v in nd.items()},
            "model_minus_popularity_spearman": paired_bootstrap(
                np.asarray(rows["model"]) - np.asarray(rows["popularity"])),
            "model_minus_global_mean_spearman": paired_bootstrap(
                np.asarray(rows["model"]) - np.asarray(rows["global_mean_profile"])),
            f"model_minus_global_mean_ndcg@{k}": paired_bootstrap(
                np.asarray(nd["model"]) - np.asarray(nd["global_mean_profile"]))}
    out["label"] = label
    out["baselines"] = {
        "popularity": "train users who can answer: ctx.cnt[i] for an item, answer_rate[c]*140768 "
                      "for a concept (policy_p1.py's declared dumb comparator)",
        "global_mean_profile": "mean of max(gain,0) over the TRAINED states, remasked per state"}
    return out


# ============================================================================ the losses
def kl_blend_loss(logits, Pstar_rows, ent_rows, tau_m, topk=0):
    """KL(p* || softmax(s / tau_m)). Returns the SUM over the batch.

    `logits` already carries -1e20 at asked candidates (finite, so 0 * log p is 0 and never NaN).
    `topk = 0` is the registered form: the WHOLE measured table, no truncation.
    """
    lp = torch.log_softmax(logits / tau_m, dim=1)
    if topk:                                     # UNRUN option; a truncation, needs sign-off
        v, j = torch.topk(Pstar_rows, topk, dim=1)
        v = v / v.sum(1, keepdim=True)
        lp = torch.gather(lp, 1, j)
        return -(v * lp).sum()
    return -(Pstar_rows * lp).sum() - ent_rows.sum()


def approx_ndcg_loss(logits, rel_rows, temp=1.0, k=50):
    """UNRUN ALTERNATIVE (`--loss approxndcg`). ApproxNDCG (Qin et al. 2010): the rank of item i is
    approximated by `1 + sum_{j != i} sigmoid((s_j - s_i) / temp)`, which is differentiable, and the
    NDCG built on it is maximised directly.

    ! It is restricted to the `k` highest-relevance candidates because the pairwise term is O(P^2)
      per example and P = 1000. That restriction is a TRUNCATION of the supervision (HARD RULE 1) and
      would need explicit sign-off before this arm is ever run. It is implemented, tested for shape
      and gradient only, and NOT registered.
    """
    r, j = torch.topk(rel_rows, k, dim=1)
    s = torch.gather(logits, 1, j)
    d = (s.unsqueeze(2) - s.unsqueeze(1)) / temp
    pi = 1.0 + (torch.sigmoid(-d).sum(2) - 0.5)
    g = torch.pow(2.0, r) - 1.0
    dcg = (g / torch.log2(1.0 + pi)).sum(1)
    ideal, _ = torch.sort(r, dim=1, descending=True)
    idisc = 1.0 / torch.log2(torch.arange(2, k + 2, dtype=torch.float32))[None, :]
    idcg = ((torch.pow(2.0, ideal) - 1.0) * idisc).sum(1).clamp_min(1e-12)
    return -(dcg / idcg).sum()


def finetune(pol, bank, phi_ext, tr_rows, tr_pos, tr_state, Pstar, ent, n_pool, args, hook):
    """The profile KL at blend states + the incumbent distillation CE everywhere else.

    The replay CE is C1-final's, unchanged and for the same reason: it is the single difference
    between C1-final's policy and the pilot's collapse, and the no-collapse gate is blocking here too.
    There is NO scale term: property (b) removes the failure mode it existed to contain.
    """
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    for p in pol.parameters():
        p.requires_grad_(False)
    train = list(pol.enc.parameters()) + list(pol.head.q.parameters())
    for p in train:                    # head.k / head.b stay FROZEN -- see the module docstring
        p.requires_grad_(True)
    opt = torch.optim.AdamW(train, lr=args.lr, weight_decay=1e-4)
    phi_c = torch.from_numpy(phi_ext[:n_pool])
    Pt = torch.from_numpy(Pstar)
    Et = torch.from_numpy(ent.astype(np.float32))
    Rt = torch.from_numpy(np.ascontiguousarray(Pstar))          # relevance for the unrun rank arm
    rng = np.random.default_rng(args.seed)
    N = len(tr_rows)
    hist = []
    with torch.no_grad():                  # head.k / head.b are frozen: their bank is a CONSTANT
        kc = pol.head.k(phi_c).clone()
        bc = pol.head.b(phi_c).squeeze(-1).clone()
    kcT = kc.T.contiguous()
    for ep in range(1, args.epochs + 1):
        perm = rng.permutation(N)
        t0 = time.time()
        tot = kl_s = ce_s = 0.0
        n_seen = n_bl = n_ce = 0
        for s in range(0, N, args.batch):
            e = perm[s:s + args.batch]
            r, p, st = tr_rows[e], tr_pos[e], tr_state[e]
            z, hp, ho, hm, pf, am, y = make_batch(bank, phi_ext, r, p, n_pool)
            u = pol.head.q(pol.enc(z, hp, ho, hm, pf))
            bl = torch.from_numpy(np.flatnonzero(st >= 0))
            ce = torch.from_numpy(np.flatnonzero(st < 0))
            terms = []
            if len(bl):
                si = torch.from_numpy(st[st >= 0].astype(np.int64))
                lg = (u[bl] @ kcT + bc[None, :]).masked_fill(am[bl], -1e20)
                if args.loss == "kl":
                    lk = kl_blend_loss(lg, Pt[si], Et[si], args.tau_m, topk=args.kl_topk)
                else:
                    lk = approx_ndcg_loss(lg, Rt[si], temp=args.rank_temp, k=args.rank_topk)
                terms.append(args.alpha * lk)
                kl_s += float(lk); n_bl += len(bl)
            if len(ce):
                lg = (u[ce] @ kcT + bc[None, :]).masked_fill(am[ce], float("-inf"))
                lc = torch.nn.functional.cross_entropy(lg, y[ce], reduction="sum")
                terms.append(lc)
                ce_s += float(lc); n_ce += len(ce)
            loss = sum(terms) / len(e)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss) * len(e); n_seen += len(e)
            if (s // args.batch) % 400 == 399:
                log(f"    ep{ep} {n_seen:,}/{N:,} loss {tot / n_seen:.5f} "
                    f"{args.loss} {kl_s / max(n_bl, 1):.5f} ce {ce_s / max(n_ce, 1):.4f} "
                    f"({(time.time() - t0) / 60:.1f}m)")
        rec = {"epoch": ep, "loss": tot / n_seen,
               f"mean_{args.loss}_train_blend": kl_s / max(n_bl, 1),
               "mean_ce_replay": ce_s / max(n_ce, 1),
               "n_blend_examples": int(n_bl), "n_replay_examples": int(n_ce),
               "minutes": round((time.time() - t0) / 60, 2)}
        hist.append(rec)
        log(f"  [ep{ep}] loss {rec['loss']:.5f} | train-blend {args.loss} "
            f"{rec[f'mean_{args.loss}_train_blend']:.5f} | replay CE {rec['mean_ce_replay']:.4f} "
            f"| {rec['minutes']:.1f}m")
        hook(ep, hist)
    for p in pol.parameters():
        p.requires_grad_(False)
    return hist


# ============================================================================ spaces
def tower_bank(E, ni, C):
    r"""The TOWER read-out space, built exactly as `c3b_adapter.TowerBridge` builds its question side:
    an ITEM is its decoder row centred and popularity-stripped (`Ew[i]`), a CONCEPT is
    `build_concept_dirs`' direction `d_c[c]` from the env.  The reproduction control is the bridge's
    own: averaging `Ew` over each concept's members must return `d_c` to float tolerance.
    """
    ctx, sh = E["ctx"], E["sh"]
    Wd = np.asarray(ctx.Wd.detach().numpy(), np.float64)
    u1 = np.asarray(sh["u1"].detach().numpy(), np.float64)
    u1 = u1 / np.linalg.norm(u1)
    Ec = Wd - Wd.mean(0, keepdims=True)
    Ew = Ec - np.outer(Ec @ u1, u1)
    d_c = np.asarray(sh["d_c"].detach().numpy(), np.float64)
    tags, members = list(sh["tags"]), sh["members"]
    worst = 0.0
    for i, t in enumerate(tags):
        v = Ew[np.asarray(members[t], np.int64)].mean(0)
        v = v / max(np.linalg.norm(v), 1e-12)
        worst = max(worst, float(np.abs(np.abs(v @ d_c[i]) - 1.0)))
    assert worst < 1e-3, (f"the tower whitening does not reproduce the env's concept directions "
                          f"(worst 1-|cos| {worst:g})")
    Bt = np.concatenate([Ew, d_c], 0).astype(np.float32)
    assert Bt.shape[0] == ni + C == N_BANK
    return Bt, {"dc_reproduction_worst_1_minus_abs_cos": worst,
                "item_norm": dist(np.linalg.norm(Ew, axis=1)),
                "concept_norm": dist(np.linalg.norm(d_c, axis=1)),
                "caveat": "concept rows are UNIT-normalised by build_concept_dirs while item rows "
                          "are not, so a cross-channel convex combination in this space is "
                          "scale-inhomogeneous. Reported, not corrected: correcting it would be a "
                          "new space, not the renderer's."}


# ============================================================================ the run
def run(args):
    from strategy_channel_suite import build_greedy_env, contract_selftest, genv_mean, CONCSTATS
    from strategy_channel_suite import N_TRAIN_USERS
    from ndcg_tree import build_pool, branch_of_selftest
    t00 = time.time()
    os.makedirs(PAPERC, exist_ok=True)
    torch.manual_seed(SEED); np.random.seed(SEED)
    GB = git_block(allow_dirty=args.allow_dirty, threads=args.threads)     # HARD RULE 10, first
    log(f"[hardrule10] src/instrument clean={GB['src_instrument_clean']} rev={GB['rev'][:8]} "
        f"threads={GB['threads']}")
    torch.set_num_threads(args.threads)

    # ---- the warm start ---------------------------------------------------------------------
    blob = torch.load(WARM, map_location="cpu")
    assert blob["arch"] == "equivariant" and blob["state"] == "both"
    assert len(blob["ext"]) == N_EXT, (
        f"m7dj's phi bank is {len(blob['ext'])} entries, expected {N_EXT} -- assert, do not assume")
    assert int(blob["n_pool"]) == 1000 and int(blob["d_phi"]) == 205 and int(blob["d_z"]) == 200
    warm_sha = sha256(WARM)
    log(f"[warm] m7dj_policy.pt epoch={blob['epoch']} ext={len(blob['ext'])} sha256 {warm_sha[:12]}...")

    # ---- the environment, the pool, the canonical TRAIN snap --------------------------------
    E = build_greedy_env(args.snapshot, "train", args.concept_ckpt, "paperC/c5profile",
                         cohort_slice=(0, args.snap_users))
    contract_selftest(E, E["rows"])
    rows_env = E["rows"]
    assert len(rows_env) == args.snap_users
    f0, t0_ = genv_mean(E, [], rows_env)
    assert f0 == TRAIN_M0_FULL, (
        f"CANONICAL SNAP FAILED: train m=0 full {f0!r}, banked {TRAIN_M0_FULL!r}.")
    log(f"[gate] canonical TRAIN m=0 snap full={f0!r} tail={t0_!r} -- full asserted exact")

    pool, _ip, _cp = build_pool(E, 500, 500)
    pool = [(int(a), int(b)) for a, b in pool]
    assert pool == [(int(a), int(b)) for a, b in blob["pool"]], "the env's pool is not m7dj's pool"
    n_pool = len(pool)
    pool_index = {q: i for i, q in enumerate(pool)}
    branch_of_selftest(E, rows_env[:2000], [pool[0], pool[500], (1, 114)])

    ext = [(int(a), int(b)) for a, b in blob["ext"]]
    norm = (np.asarray(blob["norm"][0], np.float32), np.asarray(blob["norm"][1], np.float32))
    phi_ext, _ = build_phi(E, ext, norm=norm)
    assert phi_ext.shape == (N_EXT, 205)
    ni, C = int(E["ctx"].ni), int(E["C"])
    full_pool = [(0, i) for i in range(ni)] + [(1, c) for c in range(C)]
    assert len(full_pool) == N_BANK
    phi_full, _ = build_phi(E, full_pool, norm=norm)
    pool2full = np.asarray([(i if ch == 0 else ni + i) for ch, i in pool], np.int64)

    pol0 = Policy(200, 205, n_pool, arch="equivariant", state="both", use_pos=True)
    pol0.load_state_dict(blob["sd"]); pol0.eval()
    assert sum(p.numel() for p in pol0.parameters()) == 526542

    # ---- THE TWO READ-OUT SPACES -------------------------------------------------------------
    SP = {}
    Ebank_k, Kbank = bank_vectors(pol0, phi_full)
    Epool_k, Kpool = bank_vectors(pol0, phi_ext[:n_pool])
    map_err = index_map_assert(Ebank_k, pool2full, Epool_k)
    SP["kphi"] = {"bank": Ebank_k, "pool": Epool_k, "mu": Ebank_k.astype(np.float64).mean(0),
                  "meta": {"e_q": "[k(phi_q), b(phi_q)] -- C1-final's space",
                           "pool2full_index_map_max_abs_err": map_err}}
    Bt, tmeta = tower_bank(E, ni, C)
    SP["tower"] = {"bank": Bt, "pool": Bt[pool2full], "mu": Bt.astype(np.float64).mean(0),
                   "meta": dict(tmeta, e_q="item = centred+popularity-stripped decoder row Ew[i]; "
                                           "concept = build_concept_dirs direction d_c[c]")}
    log(f"[space] kphi bank {Ebank_k.shape} | tower bank {Bt.shape}; primary = {args.space}")

    # ---- the tree, its sweeps, and the bank --------------------------------------------------
    tree = json.load(open(TREE_JSON))
    assert tree["split"] == "train" and int(tree["total_L"]) == TOTAL_L
    assert tree["test_touched"] is False
    nodes = tree["nodes"]
    states, gain_chk = load_sweeps(nodes, pool_index)
    bank = load_cold_bank()
    node_at, term, unfitted, node_ids, nidx = descend(nodes, bank)
    contract = verify_labels(nodes, node_ids, node_at, term, unfitted, bank, pool_index)
    log(f"[contract] the re-derived descent reproduces ALL {contract['n_cells_checked']:,} banked "
        f"labels; {contract['n_users_unfitted_branch']} users in an unfitted branch")

    sidx = {s["sid"]: i for i, s in enumerate(states)}
    node_to_state = np.full(len(node_ids), -1, np.int64)
    for nid, i in sidx.items():
        node_to_state[nidx[nid]] = i
    depth_of_node = np.asarray([int(nodes[k]["depth"]) for k in node_ids], np.int64)
    blend = np.full((bank["n"], TOTAL_L), -1, np.int64)
    posgrid = np.arange(TOTAL_L)[None, :]
    on_chain = node_at >= 0
    own_depth = np.zeros_like(node_at, np.int64)
    own_depth[on_chain] = depth_of_node[node_at[on_chain]]
    is_own = on_chain & (own_depth == posgrid)
    st_of = np.full_like(blend, -1)
    st_of[on_chain] = node_to_state[node_at[on_chain]]
    blend[is_own & (st_of >= 0)] = st_of[is_own & (st_of >= 0)]
    n_blend_cells = int((blend >= 0).sum())
    occ = np.bincount(blend[blend >= 0], minlength=len(states))
    assert occ.min() > 0, "a searched node state has no bank users"
    log(f"[states] {n_blend_cells:,} of {bank['n'] * TOTAL_L:,} bank cells are one of the 72 "
        f"searched node states; occupancy min {occ.min():,} median {int(np.median(occ)):,}")

    srows, spos = [], []
    for i in range(len(states)):
        r, p = np.nonzero(blend == i)
        srows.append(r.astype(np.int64)); spos.append(p.astype(np.int64))

    # ---- the PROFILE targets ------------------------------------------------------------------
    tau_t, w1_star, w1_got, Wprop = calibrate_tau_t(states, args.m)
    if args.tau_t is not None:
        tau_t = float(args.tau_t)
    Pstar, asked_m, ent, tmeta_p = build_profile_targets(states, n_pool, tau_t)
    G, _ = gain_rows(states, n_pool)
    log(f"[tau_t] declared rule solved: tau_t = {tau_t:.6g} reproduces median top-{args.m} w1 "
        f"{w1_got:.4f} against the measured proportional w1* {w1_star:.4f}")
    log(f"[target] supervision width median {tmeta_p['supervision_width_per_state']['median']:.0f} "
        f"numbers/state | effective support exp(H) median "
        f"{tmeta_p['target_effective_support_expH']['median']:.1f} | full-table top-1 weight median "
        f"{tmeta_p['target_top1_weight_full_table']['median']:.5f}")

    # C1-final's VECTOR target, rebuilt unchanged: reading 1/2/4/5 are scored against it so the two
    # runs answer the same question about the same object.
    Tt_k, _TK, meta = build_targets(states, Epool_k, Kpool, pool_index, args.m)
    Wm = np.asarray([m["weights"] for m in meta], np.float64)
    Im = np.asarray([m["top_m_pool_idx"] for m in meta], np.int64)
    Tt = {"kphi": Tt_k,
          "tower": np.einsum("sm,smd->sd", Wm, SP["tower"]["pool"][Im]).astype(np.float32)}
    G0 = {sp: {k: np.asarray(v) for k, v in state_geometry(
        Tt[sp], states, SP[sp]["bank"], pool2full, chunk=args.chunk, label=f"target/{sp}").items()}
        for sp in SP}

    # the popularity prior, on ONE scale across channels (policy_p1.py's declared comparator)
    zc = np.load(CONCSTATS)
    c_ar = (zc["n"] / N_TRAIN_USERS).astype(np.float64)
    cnt = np.asarray(E["ctx"].cnt, np.float64)
    popv = np.asarray([cnt[int(i)] if int(ch) == 0 else c_ar[int(i)] * N_TRAIN_USERS
                       for ch, i in pool], np.float64)

    out = {"analysis": "paperC_C5_profile_policy",
           "design_sheet": "docs/design/DS_C5_PROFILE_POLICY.md",
           "citable": True, "test_touched": False, "split": "train_states",
           "teacher": os.path.basename(TREE_JSON),
           "predecessor": "docs/results/C1_FINAL.md (the generalisation failure this re-parameterises)",
           "warm_start": {"path": os.path.relpath(WARM, _ROOT), "sha256": warm_sha,
                          "epoch": int(blob["epoch"]), "n_ext": len(ext)},
           "git": GB,
           "declared_constants": {
               "tau_m": args.tau_m, "m": args.m, "tau_t": tau_t,
               "tau_t_rule": ("solved ONCE so the median top-m top-1 weight of p* equals the "
                              "MEASURED gain-proportional w1; " + W1_STAR_SOURCE),
               "w1_star_measured": w1_star, "w1_at_tau_t": w1_got,
               "tau_t_overridden_on_cli": args.tau_t is not None,
               "loss": args.loss, "kl_topk": args.kl_topk,
               "no_scale_term": "REMOVED BY CONSTRUCTION: the emission is a convex combination, so "
                                "its norm cannot blow up. C1-final's lambda term has no analogue.",
               "frozen": ["head.k", "head.b"]},
           "hyperparameters_not_tuned": {"epochs": args.epochs, "batch": args.batch, "lr": args.lr,
                                         "alpha": args.alpha, "optimiser": "AdamW wd=1e-4",
                                         "seed": args.seed,
                                         "note": "chosen once, before the run, never swept"},
           "spaces": {sp: SP[sp]["meta"] for sp in SP},
           "primary_space": args.space,
           "gates": {"canonical_snap_train_m0_full": {"got": f0, "expected": TRAIN_M0_FULL,
                                                      "pass": True},
                     "canonical_snap_train_m0_tail": t0_,
                     "phi_bank_is_1004": True,
                     "pool2full_index_map_max_abs_err": map_err,
                     "sweep_argmax_equals_splitter_max_abs_gain_err": gain_chk,
                     "descent_reproduces_banked_labels": contract},
           "population": {
               "cohort": "the 85,289-user COLD arm of m6/traj_dj -- train users endorsing NO anchor "
                         "concept, a selected 60.6% of the 140,768 train users, by design of Paper "
                         "B's forest split. Every number below is conditioned on that population.",
               "n_users": bank["n"], "n_cells": int(bank["n"] * TOTAL_L),
               "n_blend_cells": n_blend_cells,
               "bank_occupancy_per_searched_state": dist(occ)},
           "profile_target": tmeta_p,
           "ingredient_stability": {
               "carried_prior_split_half_meanremoved_cos": C1.PILOT_SPLIT_HALF_MEANREMOVED,
               "status": "UNMET, exactly as in C1-final: node cohorts min 200 / median 356, 43 of 72 "
                         "below 500, and no per-user gain sidecar exists for this tree. C5 widens "
                         "the SUPERVISION from 5 numbers to ~1,000 but it does not re-measure the "
                         "gains, so it cannot and does not discharge this.",
               "node_cohorts": dist([s["n_fit_users"] for s in states]),
               "n_nodes_below_500_fit_users": int(sum(s["n_fit_users"] < 500 for s in states)),
               "binding_wording": "we train toward a MEASURED profile; we do NOT claim the profile's "
                                  "top five are the right five"},
           "cells": {}}
    p_json = os.path.join(PAPERC, f"c5_profile{args.out_tag}.json")
    p_npz = os.path.join(PAPERC, f"c5_profile{args.out_tag}_per_state.npz")
    npz = {"Pstar": Pstar, "gains": G.astype(np.float32), "asked": asked_m,
           "target_kphi": Tt["kphi"], "target_tower": Tt["tower"],
           "bank_occupancy": occ,
           "state_depth": np.asarray([s["depth"] for s in states]),
           "state_n_fit_users": np.asarray([s["n_fit_users"] for s in states]),
           "state_sid": np.asarray([s["sid"] for s in states])}

    def flush():
        out["minutes"] = round((time.time() - t00) / 60, 2)
        with open(p_json + ".tmp", "w") as fh:
            json.dump(out, fh, indent=1, default=float)
        os.replace(p_json + ".tmp", p_json)
        savez_atomic(p_npz, **npz)

    flush()

    # ---- the warm start's own behaviour -------------------------------------------------------
    with torch.no_grad():
        phi_c = torch.from_numpy(phi_ext[:n_pool])
        kc0 = pol0.head.k(phi_c).clone(); bc0 = pol0.head.b(phi_c).squeeze(-1).clone()
    rows_all, pos_all = np.nonzero(bank["valid"])
    rows_all = rows_all.astype(np.int64); pos_all = pos_all.astype(np.int64)
    st_all = blend[rows_all, pos_all]
    dev = bank["uid"][rows_all] >= args.dev_lo
    _r = np.random.default_rng(SEED + 1)
    dsel = np.flatnonzero(dev)
    if len(dsel) > 40000:
        dsel = dsel[_r.choice(len(dsel), 40000, replace=False)]
    warm_dev = top1(pol0, bank, phi_ext, rows_all[dsel], pos_all[dsel], n_pool)
    assert warm_dev > 0.99, (
        f"the rebuilt pipeline scores the warm start at {warm_dev:.5f} on its own cold dev states "
        f"-- the phi/pool/batch reconstruction is wrong, STOP")
    out["gates"]["warm_start_cold_dev_top1_rebuilt"] = {"got": warm_dev, "n_states": int(len(dsel))}
    log(f"[contract] warm start teacher-forced top-1 on {len(dsel):,} cold dev states: {warm_dev:.5f}")

    grng = np.random.default_rng(SEED)
    ord_pool = np.flatnonzero(st_all < 0)
    gsel = ord_pool[grng.choice(len(ord_pool), args.n_gate, replace=False)]
    g_rows, g_pos = rows_all[gsel], pos_all[gsel]
    a0 = snap_argmax(pol0, bank, phi_ext, g_rows, g_pos, n_pool)
    n_distinct_warm = int(len(np.unique(a0)))
    out["gate3_population"] = {"n_states": int(args.n_gate),
                               "n_distinct_questions_warm_start": n_distinct_warm,
                               "bar_distinct": 0.75 * n_distinct_warm, "bar_agreement": 0.90}
    log(f"[gate3] warm start emits {n_distinct_warm} distinct questions over {args.n_gate:,} "
        f"ordinary non-blend states")
    npz["gate3_rows"] = g_rows; npz["gate3_pos"] = g_pos; npz["gate3_warm_argmax"] = a0

    cache0, prof0 = collect_topm(pol0, bank, phi_ext, srows, spos, n_pool, kc0, bc0, args.m,
                                 args.tau_m)
    flush()

    # ---- the two registered cells -------------------------------------------------------------
    depth = np.asarray([s["depth"] for s in states])
    sids = [s["sid"] for s in states]
    SPLITS = {
        "primary_root_like_subtree": np.asarray(
            [i for i, s in enumerate(sids) if s == "root.like" or s.startswith("root.like.")]),
        "secondary_depth_ge_5": np.flatnonzero(depth >= 5)}
    assert len(SPLITS["primary_root_like_subtree"]) == 15
    assert len(SPLITS["secondary_depth_ge_5"]) == 30
    splitter_pool_idx = np.asarray([pool_index[(int(nodes[s["nid"]]["splitter"][0]),
                                                int(nodes[s["nid"]]["splitter"][1]))]
                                    for s in states], np.int64)

    for cell, held in SPLITS.items():
        log("#" * 100)
        log(f"# CELL {cell}: {len(held)} states held out, {len(states) - len(held)} trained")
        heldset = set(int(x) for x in held)
        trained = np.asarray([i for i in range(len(states)) if i not in heldset])
        keep = ~np.isin(st_all, held)          # held-out blend states receive NO loss at all
        tr_rows, tr_pos, tr_state = rows_all[keep], pos_all[keep], st_all[keep]
        pol = copy.deepcopy(pol0)
        cellrec = {"n_held_out_states": int(len(held)), "n_trained_states": int(len(trained)),
                   "held_out_sids": [sids[int(i)] for i in held],
                   "n_train_examples": int(len(tr_rows)),
                   "n_blend_examples": int((tr_state >= 0).sum()),
                   "n_replay_examples": int((tr_state < 0).sum()),
                   "n_examples_excluded_heldout": int((~keep).sum()),
                   "heldout_states_receive": "no loss (neither the profile KL nor the incumbent CE)"}
        out["cells"][cell] = cellrec
        log(f"  [data] {len(tr_rows):,} training examples "
            f"({cellrec['n_blend_examples']:,} blend / {cellrec['n_replay_examples']:,} replay)")
        flush()

        def hook(ep, hist, _c=cellrec):
            _c["epochs"] = hist
            flush()

        cellrec["epochs"] = finetune(pol, bank, phi_ext, tr_rows, tr_pos, tr_state, Pstar, ent,
                                     n_pool, args, hook)

        with torch.no_grad():
            kc = pol.head.k(phi_c).clone(); bc = pol.head.b(phi_c).squeeze(-1).clone()
        assert float((kc - kc0).abs().max()) == 0.0 and float((bc - bc0).abs().max()) == 0.0, \
            "head.k / head.b moved during the fine-tune -- the bank is not a constant, STOP"
        cache, prof = collect_topm(pol, bank, phi_ext, srows, spos, n_pool, kc, bc, args.m,
                                   args.tau_m)

        # ---- the ONE fitted read-out constant, on TRAINED states only ----------------------
        tau_e, w1_e, solved = calibrate_tau_emit(cache, trained, w1_star)
        tau_w, w1_w, solved_w = calibrate_tau_emit(cache0, trained, w1_star)
        cellrec["tau_emit"] = {
            "value": tau_e, "median_top1_weight_at_tau_emit": w1_e, "w1_star": w1_star,
            "solved": solved, "warm_start_tau_emit": tau_w, "warm_start_solved": solved_w,
            "rule": "one GLOBAL scalar, solved on TRAINED states only, matching the median top-1 "
                    "emission weight to the measured w1*. Shared by every state including the "
                    "held-out ones, so it cannot carry per-state information.",
            "note": "the warm start gets its OWN tau_emit under the same rule, so reading 1b "
                    "compares two policies each read out at its own entropy-matched temperature "
                    "-- otherwise the comparison would be confounded by logit scale alone."}
        log(f"  [tau_emit] {tau_e:.6g} (median top-1 weight {w1_e:.4f} vs w1* {w1_star:.4f}, "
            f"solved={solved}) | warm start {tau_w:.6g}")

        # ---- emissions, in BOTH spaces, at the registered tau and at tau_m ------------------
        cellrec["by_space"] = {}
        for sp in ("kphi", "tower"):
            Eemb, mu = SP[sp]["pool"], SP[sp]["mu"]
            ev = emission_from_cache(cache, Eemb, Tt[sp], mu, tau_e)
            ev0 = emission_from_cache(cache0, Eemb, Tt[sp], mu, tau_w)
            evM = emission_from_cache(cache, Eemb, Tt[sp], mu, args.tau_m)
            R = {}
            R["convexity"] = ev["convexity"]
            R["convexity_n_states_checked"] = ev["convexity_n_states_checked"]
            R["mean_top1_blend_weight"] = dist(ev["mean_top1_weight"])

            # READING 1 -- generalisation (the headline), C1-final's function verbatim
            R["reading1_generalisation"] = reading1(ev, ev0, Tt[sp], held,
                                                    np.random.default_rng(SEED))
            # the WARM-START NULL that turned C1-final's headline: the same statistic computed with
            # the warm start in BOTH slots. Registered, never optional.
            R["reading1_warm_start_reference"] = reading1(ev0, ev0, Tt[sp], held,
                                                          np.random.default_rng(SEED))
            r1, r1w = R["reading1_generalisation"], R["reading1_warm_start_reference"]
            log(f"  [{sp}][reading1] held-out cos own {r1['cos_own_target']['median']:+.5f} | "
                f"a own-other {r1['a_own_minus_other']['mean']:+.5f} "
                f"[{r1['a_own_minus_other']['lo']:+.5f}, {r1['a_own_minus_other']['hi']:+.5f}] "
                f"| NULL (warm start, same pairing) {r1w['a_own_minus_other']['mean']:+.5f} "
                f"[{r1w['a_own_minus_other']['lo']:+.5f}, {r1w['a_own_minus_other']['hi']:+.5f}] "
                f"| b own-warm {r1['b_own_minus_warm_start']['mean']:+.5f} "
                f"[{r1['b_own_minus_warm_start']['lo']:+.5f}, "
                f"{r1['b_own_minus_warm_start']['hi']:+.5f}]")

            # READING 2 -- fit at TRAINED blend states
            ct, c0t = ev["cos_own"][trained], ev0["cos_own"][trained]
            R["reading2_fit_trained_states"] = {
                "unit": "state", "n_states": int(len(trained)),
                "cos_own_target": dist(ct), "warm_start_cos": dist(c0t),
                "minus_warm_start": paired_bootstrap(ct - c0t),
                "frac_states_cos_ge_0.90": float((ct >= 0.90).mean()),
                "within_state_cos_to_mean": dist(ev["within_state"][trained])}
            log(f"  [{sp}][reading2] trained-state fit median cos "
                f"{R['reading2_fit_trained_states']['cos_own_target']['median']:.5f}")

            # READING 4 -- RENDERABILITY GATE
            med_all = float(np.median(ev["norm_ratio_median"]))
            g4 = {"median_norm_ratio_all_72": med_all,
                  "median_norm_ratio_trained": float(np.median(ev["norm_ratio_median"][trained])),
                  "median_norm_ratio_held_out": float(np.median(ev["norm_ratio_median"][held])),
                  "warm_start_median": float(np.median(ev0["norm_ratio_median"])),
                  "band": [0.5, 2.0], "distribution": dist(ev["norm_ratio_median"])}
            g4["PASS"] = bool(0.5 <= med_all <= 2.0)
            R["reading4_renderability_GATE"] = g4
            log(f"  [{sp}][GATE 4] median mean-removed norm ratio {med_all:.4f} "
                f"{'PASS' if g4['PASS'] else '*** FAIL ***'} (trained "
                f"{g4['median_norm_ratio_trained']:.4f} held-out "
                f"{g4['median_norm_ratio_held_out']:.4f})")

            # READING 5 -- off-grid geometry
            Gg = {k: np.asarray(v) for k, v in state_geometry(
                ev["zbar"], states, SP[sp]["bank"], pool2full, chunk=args.chunk,
                label=f"{cell}/{sp}").items()}
            R["reading5_offgrid_geometry"] = {
                "emitted": geom_summary(Gg), "target": geom_summary(G0[sp]),
                "same_nearest_bank_question_as_target": float((Gg["snap"] == G0[sp]["snap"]).mean()),
                "held_out_only": {"emitted_cos1": dist(Gg["cos1"][held]),
                                  "same_nearest_as_target": float(
                                      (Gg["snap"][held] == G0[sp]["snap"][held]).mean())}}
            # the tau_m read-out, reported alongside so the entropy-matching is auditable
            R["at_tau_m_secondary"] = {
                "tau": args.tau_m,
                "mean_top1_blend_weight": dist(evM["mean_top1_weight"]),
                "cos_own_trained": dist(evM["cos_own"][trained]),
                "cos_own_held_out": dist(evM["cos_own"][held]),
                "median_norm_ratio": float(np.median(evM["norm_ratio_median"]))}
            npz[f"zbar_{cell}_{sp}"] = ev["zbar"]
            npz[f"cos_own_{cell}_{sp}"] = ev["cos_own"]
            npz[f"emit_cos1_{cell}_{sp}"] = Gg["cos1"]
            npz[f"emit_snap_{cell}_{sp}"] = Gg["snap"]
            cellrec["by_space"][sp] = R
            flush()

        # ---- READING 3 -- NO-COLLAPSE GATE (hard, blocking), space-free --------------------
        a1 = snap_argmax(pol, bank, phi_ext, g_rows, g_pos, n_pool)
        ndq = int(len(np.unique(a1)))
        agree = float((a0 == a1).mean())
        g3 = {"n_states": int(args.n_gate),
              "n_distinct_questions_warm_start": n_distinct_warm,
              "n_distinct_questions_finetuned": ndq,
              "distinct_ratio": ndq / max(n_distinct_warm, 1),
              "distinct_bar": 0.75, "distinct_pass": bool(ndq >= 0.75 * n_distinct_warm),
              "snap_agreement_vs_warm_start": agree,
              "snap_agreement_ci": wilson(int((a0 == a1).sum()), int(len(a0))),
              "agreement_bar": 0.90, "agreement_pass": bool(agree >= 0.90)}
        g3["PASS"] = bool(g3["distinct_pass"] and g3["agreement_pass"])
        cellrec["reading3_no_collapse_GATE"] = g3
        npz[f"gate3_argmax_{cell}"] = a1
        log(f"  [GATE 3 no-collapse] distinct {ndq}/{n_distinct_warm} "
            f"(ratio {g3['distinct_ratio']:.3f}) | agreement {agree:.4f} => "
            f"{'PASS' if g3['PASS'] else '*** FAIL ***'}")

        # ---- NESTING: the discrete policy the profile implies -------------------------------
        arg_state = np.asarray([int(np.bincount(cache[i][1][:, 0]).argmax())
                                for i in range(len(states))], np.int64)
        peruser_hit = np.asarray([float((cache[i][1][:, 0] == splitter_pool_idx[i]).mean())
                                  for i in range(len(states))])
        cellrec["nesting_tau_to_zero"] = {
            "claim": "as tau -> 0 the emission becomes e_{argmax_q s(q)}; the discrete policy is "
                     "therefore nested EXACTLY, and this reports whether that argmax is the tree's "
                     "own splitter",
            "modal_argmax_equals_splitter_trained": float(
                (arg_state[trained] == splitter_pool_idx[trained]).mean()),
            "modal_argmax_equals_splitter_held_out": float(
                (arg_state[held] == splitter_pool_idx[held]).mean()),
            "per_user_argmax_equals_splitter": {"trained": dist(peruser_hit[trained]),
                                                "held_out": dist(peruser_hit[held])},
            "warm_start_modal_equals_splitter": float(np.mean(
                [int(np.bincount(cache0[i][1][:, 0]).argmax()) == splitter_pool_idx[i]
                 for i in range(len(states))]))}
        log(f"  [nesting] modal argmax == the tree's splitter at "
            f"{cellrec['nesting_tau_to_zero']['modal_argmax_equals_splitter_trained']:.3f} of "
            f"trained / {cellrec['nesting_tau_to_zero']['modal_argmax_equals_splitter_held_out']:.3f}"
            f" of held-out states (warm start "
            f"{cellrec['nesting_tau_to_zero']['warm_start_modal_equals_splitter']:.3f})")

        # ---- READING 6 -- the PROFILE-level readings (the new ones) --------------------------
        cellrec["reading6_profile"] = profile_readings(prof, G, asked_m, held, trained, popv,
                                                       "trained policy")
        cellrec["reading6_profile_warm_start"] = profile_readings(prof0, G, asked_m, held, trained,
                                                                  popv, "untrained warm start")
        r6 = cellrec["reading6_profile"]["held_out"]
        r6w = cellrec["reading6_profile_warm_start"]["held_out"]
        log(f"  [reading6 held-out] Spearman model {r6['spearman']['model']['median']:+.4f} "
            f"(warm start {r6w['spearman']['model']['median']:+.4f}) | popularity "
            f"{r6['spearman']['popularity']['median']:+.4f} | global-mean profile "
            f"{r6['spearman']['global_mean_profile']['median']:+.4f} | NDCG@5 model "
            f"{r6['ndcg@5']['model']['median']:.4f} vs global-mean "
            f"{r6['ndcg@5']['global_mean_profile']['median']:.4f}")
        npz[f"profile_{cell}"] = prof.astype(np.float32)

        cellrec["VERDICT"] = ("BLOCKING GATE FAILED -- the run fails regardless of readings 1/2/6"
                              if not (g3["PASS"] and
                                      cellrec["by_space"][args.space]["reading4_renderability_GATE"]
                                      ["PASS"]) else "both blocking gates pass")
        flush()

    npz["profile_warm_start"] = prof0.astype(np.float32)
    out["headline"] = {
        "blocking_gates": {c: {
            "gate3_no_collapse": out["cells"][c]["reading3_no_collapse_GATE"]["PASS"],
            "gate4_renderability_primary_space":
                out["cells"][c]["by_space"][args.space]["reading4_renderability_GATE"]["PASS"]}
            for c in out["cells"]},
        "all_blocking_gates_pass": bool(all(
            out["cells"][c]["reading3_no_collapse_GATE"]["PASS"] and
            out["cells"][c]["by_space"][args.space]["reading4_renderability_GATE"]["PASS"]
            for c in out["cells"]))}
    flush()
    log(f"[done] wrote {p_json} in {out['minutes']:.1f} minutes")
    return out


# ============================================================================ self-tests
def _tiny_fixture(seed=0, B=64, P=40, m=5):
    g = np.random.default_rng(seed)
    Eemb = g.standard_normal((P, 7)).astype(np.float32)
    logits = torch.from_numpy(g.standard_normal((B, P)).astype(np.float32))
    return Eemb, logits


def selftest_nesting():
    """(c) tau -> 0: the emission becomes the argmax candidate's embedding, exactly."""
    Eemb, logits = _tiny_fixture(1)
    v, idx = torch.topk(logits, 5, dim=1)
    zh, w = emit_convex(v.numpy().astype(np.float64), idx.numpy().astype(np.int64),
                        Eemb.astype(np.float64), 1e-9)
    ref = Eemb[idx[:, 0].numpy()]
    d = float(np.abs(zh - ref).max())
    assert d < 1e-9, f"tau->0 emission differs from e_argmax by {d:g}"
    assert float(np.abs(w[:, 0] - 1.0).max()) < 1e-9
    # and the argmax of the top-m IS the argmax of the whole row
    assert (idx[:, 0].numpy() == logits.argmax(1).numpy()).all()
    return {"max_abs_diff_to_e_argmax": d, "max_abs_w1_minus_1": float(np.abs(w[:, 0] - 1).max())}


def selftest_convexity():
    """(b) convexity + the norm bound, at several temperatures."""
    Eemb, logits = _tiny_fixture(2)
    v, idx = torch.topk(logits, 5, dim=1)
    rep = {}
    for tau in (1e-6, 0.01, 1.0, 100.0):
        zh, w = emit_convex(v.numpy().astype(np.float64), idx.numpy().astype(np.int64),
                            Eemb.astype(np.float64), tau)
        rep[tau] = assert_convex(w, idx.numpy().astype(np.int64), Eemb, zh)
    # the flat limit is the plain centroid of the five ingredients
    zh, _ = emit_convex(v.numpy().astype(np.float64), idx.numpy().astype(np.int64),
                        Eemb.astype(np.float64), 1e9)
    d = float(np.abs(zh - Eemb[idx.numpy()].mean(1)).max())
    assert d < 1e-6, f"tau->inf emission is not the uniform centroid ({d:g})"
    return {"tau_inf_max_abs_diff_to_centroid": d,
            "frac_inside_ingredient_norm_range_at_tau_1": rep[1.0]["frac_inside_ingredient_norm_range"],
            "frac_at_or_below_max": {str(t): rep[t]["frac_at_or_below_max"] for t in rep}}


def selftest_tau_t_zero_is_ce():
    """C5's objective NESTS C1-final's: at tau_t -> 0 the KL target is the teacher's one-hot and
    KL(p*||p_hat) is the incumbent cross-entropy."""
    g = np.random.default_rng(3)
    B, P = 32, 60
    gain = g.standard_normal((B, P)) * 0.01
    mask = np.zeros((B, P), bool)
    mask[:, :3] = True                                   # a prefix of asked candidates
    gain[mask] = np.nan
    Pstar = _softmax_rows(np.nan_to_num(np.maximum(gain, 0.0)), mask, 1e-9)
    y = np.asarray([int(np.nanargmax(np.where(mask[i], -np.inf, gain[i]))) for i in range(B)],
                   np.int64)
    assert (Pstar.argmax(1) == y).all(), "the tau_t->0 target is not the argmax-gain candidate"
    assert float(Pstar.max(1).min()) > 1 - 1e-9
    logits = torch.from_numpy(g.standard_normal((B, P)).astype(np.float32))
    logits = logits.masked_fill(torch.from_numpy(mask), -1e20)
    ent = np.zeros(B, np.float32)
    kl = float(kl_blend_loss(logits, torch.from_numpy(Pstar), torch.from_numpy(ent), 1.0))
    ce = float(torch.nn.functional.cross_entropy(
        logits.masked_fill(torch.from_numpy(mask), float("-inf")),
        torch.from_numpy(y), reduction="sum"))
    d = abs(kl - ce) / max(abs(ce), 1e-9)
    assert d < 1e-5, f"KL at tau_t->0 is not the one-hot CE (rel diff {d:g})"
    return {"kl": kl, "ce": ce, "rel_diff": d}


def selftest_kl_decreases(steps=20):
    """20 optimiser steps on a tiny synthetic fixture: the KL must fall and the argmax must move to
    the target's argmax."""
    torch.manual_seed(0)
    g = np.random.default_rng(4)
    B, P, D = 128, 50, 12
    mask = np.zeros((B, P), bool); mask[:, :2] = True
    gain = g.standard_normal((B, P)) * 0.01
    Pstar = _softmax_rows(np.maximum(gain, 0.0), mask, 0.004)
    ent = -(np.where(Pstar > 0, Pstar * np.log(np.maximum(Pstar, 1e-300)), 0)).sum(1)
    K = torch.from_numpy(g.standard_normal((P, D)).astype(np.float32))      # a FROZEN bank
    b = torch.from_numpy(g.standard_normal(P).astype(np.float32))
    x = torch.from_numpy(g.standard_normal((B, 16)).astype(np.float32))
    q = torch.nn.Sequential(torch.nn.Linear(16, 32), torch.nn.ReLU(), torch.nn.Linear(32, D))
    opt = torch.optim.AdamW(q.parameters(), lr=3e-2)
    mt = torch.from_numpy(mask)
    Pt = torch.from_numpy(Pstar); Et = torch.from_numpy(ent.astype(np.float32))
    curve = []
    for _ in range(steps):
        lg = (q(x) @ K.T + b[None, :]).masked_fill(mt, -1e20)
        loss = kl_blend_loss(lg, Pt, Et, TAU_M) / B
        curve.append(float(loss))
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        lg = (q(x) @ K.T + b[None, :]).masked_fill(mt, -1e20)
        hit = float((lg.argmax(1).numpy() == Pstar.argmax(1)).mean())
        curve.append(float(kl_blend_loss(lg, Pt, Et, TAU_M) / B))
    assert curve[-1] < curve[0], f"KL did not decrease ({curve[0]:.5f} -> {curve[-1]:.5f})"
    assert min(curve) >= -1e-6, f"KL went negative ({min(curve):g}) -- the entropy term is wrong"
    return {"kl_start": curve[0], "kl_end": curve[-1], "curve": [round(c, 5) for c in curve],
            "argmax_agreement_with_target_after_20_steps": hit}


def selftest_tau_calibration():
    """The declared tau_t rule is a monotone bisection and it hits its criterion."""
    g = np.random.default_rng(5)
    states = []
    for i in range(30):
        # the sweeps bank a gain for EVERY unasked candidate and none for the prefix
        gains = {int(q): float(x) for q, x in zip(range(2, 40), g.random(38) * 0.02)}
        states.append({"sid": f"s{i}", "gains": gains, "pool_prefix_idx": [0, 1]})
    tau, w1s, w1g, _ = calibrate_tau_t(states, 5)
    assert abs(w1g - w1s) < 1e-3, f"the bisection missed its criterion ({w1g} vs {w1s})"
    Pstar, mask, ent, meta = build_profile_targets(states, 40, tau)
    assert np.abs(Pstar.sum(1) - 1).max() < 1e-6
    top5 = np.sort(Pstar, 1)[:, ::-1][:, :5]
    top5 = top5 / top5.sum(1, keepdims=True)
    assert abs(float(np.median(top5[:, 0])) - w1s) < 1e-3, \
        "restricting p* to the top-5 does not reproduce the criterion"
    return {"tau_t": tau, "w1_star": w1s, "w1_at_tau_t": w1g,
            "effective_support_median": meta["target_effective_support_expH"]["median"]}


def selftest_rank_loss():
    """The UNRUN ApproxNDCG arm: shape, finiteness and a non-zero gradient. Not a correctness proof."""
    g = np.random.default_rng(6)
    B, P = 8, 60
    rel = torch.from_numpy(np.maximum(g.standard_normal((B, P)) * 0.01, 0).astype(np.float32))
    lg = torch.from_numpy(g.standard_normal((B, P)).astype(np.float32)).requires_grad_(True)
    v = approx_ndcg_loss(lg, rel, k=20)
    v.backward()
    assert torch.isfinite(v) and float(lg.grad.abs().sum()) > 0
    return {"value": float(v), "grad_l1": float(lg.grad.abs().sum())}


def selftest_readout_contract():
    """The read-out plumbing: `emission_from_cache` must produce exactly the dict
    `c1_final.reading1` consumes (it is imported, not reimplemented, so a shape drift here would
    only surface 40 minutes into a run), and `profile_readings` must run end to end."""
    g = np.random.default_rng(7)
    S, P, D, m = 12, 40, 6, 5
    Eemb = g.standard_normal((P, D)).astype(np.float32)
    cache, cache0 = [], []
    for i in range(S):
        n = int(g.integers(20, 60))
        v = -np.sort(-g.standard_normal((n, m)).astype(np.float32), axis=1)
        idx = np.stack([g.permutation(P)[:m] for _ in range(n)]).astype(np.int32)
        cache.append((v, idx))
        cache0.append((v + g.standard_normal((n, m)).astype(np.float32), idx))
    Tt = g.standard_normal((S, D)).astype(np.float32)
    mu = Eemb.mean(0).astype(np.float64)
    ev = emission_from_cache(cache, Eemb, Tt, mu, 1.0)
    ev0 = emission_from_cache(cache0, Eemb, Tt, mu, 1.0)
    held = np.arange(4)
    r1 = reading1(ev, ev0, Tt, held, np.random.default_rng(SEED))
    assert set(("cos_own_target", "a_own_minus_other", "b_own_minus_warm_start")) <= set(r1)
    assert r1["n_held_out_states"] == 4
    # the profile readings, on a synthetic table with a prefix mask
    mask = np.zeros((S, P), bool); mask[:, :2] = True
    G = g.standard_normal((S, P)) * 0.01; G[mask] = 0.0
    prof = _softmax_rows(np.maximum(G, 0) + g.standard_normal((S, P)) * 1e-3, mask, 0.01)
    popv = g.random(P)
    r6 = profile_readings(prof, G, mask, held, np.arange(4, S), popv, "selftest")
    assert set(r6["held_out"]["spearman"]) == {"model", "popularity", "global_mean_profile"}
    # a perfect predictor must score Spearman 1 and NDCG@5 1
    perfect = np.where(mask, -np.inf, G)
    assert abs(spearman(perfect[0][~mask[0]], G[0][~mask[0]]) - 1.0) < 1e-9
    assert abs(ndcg_at_k(perfect[0][~mask[0]], G[0][~mask[0]], 5) - 1.0) < 1e-9
    return {"reading1_keys_ok": True,
            "held_out_spearman_model_median": r6["held_out"]["spearman"]["model"]["median"],
            "convexity_states_checked": ev["convexity_n_states_checked"]}


def selftest():
    out = {}
    for name, fn in (("nesting_tau_to_zero", selftest_nesting),
                     ("convexity_and_norm_range", selftest_convexity),
                     ("tau_t_zero_reduces_to_one_hot_CE", selftest_tau_t_zero_is_ce),
                     ("kl_decreases_20_steps", selftest_kl_decreases),
                     ("tau_calibration", selftest_tau_calibration),
                     ("approxndcg_unrun_arm", selftest_rank_loss),
                     ("readout_contract_with_c1_final", selftest_readout_contract)):
        t = time.time()
        out[name] = fn()
        out[name]["seconds"] = round(time.time() - t, 3)
        log(f"[selftest] {name}: PASS ({out[name]['seconds']:.2f}s)")
    print(json.dumps(out, indent=1, default=float))
    return out


def main():
    ap = argparse.ArgumentParser(description="Paper C stage C5 -- see docs/design/DS_C5_PROFILE_POLICY.md")
    ap.add_argument("--selftest", action="store_true", help="run the unit tests and exit")
    ap.add_argument("--space", choices=("kphi", "tower"), default="kphi",
                    help="which read-out space is the HEADLINE; both are always computed")
    ap.add_argument("--m", type=int, default=M_BLEND, help="blend width (registered m=5)")
    ap.add_argument("--tau_m", type=float, default=TAU_M)
    ap.add_argument("--tau_t", type=float, default=None,
                    help="override the DECLARED tau_t (recorded in the JSON if used)")
    ap.add_argument("--loss", choices=("kl", "approxndcg"), default="kl")
    ap.add_argument("--kl_topk", type=int, default=0,
                    help="0 = the whole measured table (registered). Non-zero TRUNCATES the "
                         "supervision and needs sign-off (HARD RULE 1)")
    ap.add_argument("--rank_topk", type=int, default=50)
    ap.add_argument("--rank_temp", type=float, default=1.0)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--n_gate", type=int, default=1000)
    ap.add_argument("--dev_lo", type=int, default=120000)
    ap.add_argument("--snap_users", type=int, default=10000)
    ap.add_argument("--chunk", type=int, default=200)
    ap.add_argument("--out_tag", default="")
    ap.add_argument("--snapshot", default=None)
    ap.add_argument("--concept_ckpt", default=None)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--allow_dirty", action="store_true",
                    help="EMERGENCY ONLY: run with an uncommitted src/instrument (HARD RULE 10)")
    ap.add_argument("--i_have_author_goahead", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        return selftest()
    assert args.i_have_author_goahead, "REFUSING TO RUN without --i_have_author_goahead"
    if args.loss != "kl":
        assert args.i_have_author_goahead, "the ranking arm is UNRUN and unregistered"
        log("*** WARNING: --loss approxndcg is the UNREGISTERED, UNRUN alternative ***")
    from strategy_channel_suite import CERT_TOWER, CERT_CONC
    args.snapshot = args.snapshot or CERT_TOWER
    args.concept_ckpt = args.concept_ckpt or CERT_CONC
    log("#" * 100)
    log("# PAPER C / C5 -- the PROFILE-parameterised emitter on coldtree_v2_train10000: predict the "
        "measured gain profile, construct the emission as a convex combination of real bank "
        "entries. TRAIN-SIDE ONLY; the 10,000 canonical TEST users are not built, not loaded, not "
        "touched.")
    log("#" * 100)
    return run(args)


if __name__ == "__main__":
    main()
