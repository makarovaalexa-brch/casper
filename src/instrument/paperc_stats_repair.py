r"""PAPER C -- THE REPAIR STATISTICS (hostile-review pass, 2026-08-24).

Every number here is one the C0/C1 write-ups needed and did not have. Nothing is re-measured and
nothing is re-trained: this file reads the banked per-state arrays, the banked E1 per-user rollout,
and the per-user gain sidecars that the C1 targets were built from, and computes the intervals,
decompositions and stability checks the original runs reported as bare point estimates.

WHAT IT ADDS, AND WHY EACH ONE

C0 (from `c0_geometry_per_state.npz` + `policy_main_equivariant_both_gates_per_user.npz`)
  1. THE NUMBER OF DISTINCT ASKED-HISTORY STATES. C0's headline distributions are over "160,000
     states", but a state is (asked prefix, answer prefix) and 10,000 users share very few of them.
     Every median in C0_GEOMETRY.md is therefore a median over user-rows, not over states, and the
     effective n is not 160,000. This counts them exactly, from the banked `asked`/`outs`.
  2. THE VARIANCE DECOMPOSITION, per position: how much of the spread in cos1 is BETWEEN states
     rather than within one. If it is nearly all between-state, the per-row distribution is a
     handful of atoms and its IQR is a statement about cohort sizes.
  3. MAX cos1 -- the single most favourable state, reported so the "0 states above 0.98" line has
     its distance to the bar attached.
  4. STATE-WEIGHTED medians beside the user-row-weighted ones: the same statistics with each
     distinct state counting once.
  5. A CLUSTER BOOTSTRAP over USERS (10,000 resamples, seed 98765) for the headline medians. The 16
     rows of one user are not independent, so a naive interval would be wrong by the cluster factor.
  6. THE MEAN-REMOVED cos1 for the emissions -- the diagnostic C0 never ran. C0 recorded that the
     bank has an enormous shared component (raw mean pairwise cosine 0.96 against 0.0018 centred)
     and that E1's cos1 is -0.182, but never asked what the cosine is once that shared component is
     removed from BOTH sides. The emissions are regenerated through `policy_net.rollout` with
     C0's own hook, and the regenerated raw cos1 is asserted equal to the banked one before the
     new number is read -- so this is C0's emission, not a lookalike.
  7. THE REGISTERED READING RULE APPLIED TO THE RANDOM FLOOR. C0 registered a rule, applied it to
     the emission, and never applied it to its own control. If the floor lands in the same cell,
     the cell is not a property of the trained policy.

C1 (from the `policy_labels_deep_partial` per-user gain sidecars)
  8. PER-STATE BOOTSTRAP (200 resamples, seed 98765) of the TOP-5 SELECTION. The m=5 blend target is
     built from "the top-5 measured candidates", and how many of those five survive resampling the
     cohort is the sampling stability of the target's INGREDIENTS -- never measured.
  9. THE SPLIT-HALF COSINE of the m=5 target direction: two disjoint halves of the same cohort,
     each building its own target from its own gains. This bounds how much of the target direction
     is cohort noise, and it is the number that says whether "the network fitted its target" is
     fitting a direction or fitting an estimate.

The 7 single-pick heap states carry no per-user sidecar (their gains come from banked heaps), so
8 and 9 cover the 35 deep-sweep states of the registered 42, and say so.

VALIDATION ONLY. Nothing is trained. The 10,000 canonical TEST users are not built, not loaded,
not touched.
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
sys.path.insert(0, _HERE)
if "--threads" not in sys.argv:
    sys.argv += ["--threads", "4"]
_NT = sys.argv[sys.argv.index("--threads") + 1]
os.environ["OMP_NUM_THREADS"] = _NT
os.environ.setdefault("MKL_NUM_THREADS", _NT)
os.environ.setdefault("OPENBLAS_NUM_THREADS", _NT)

import json
import time
import argparse
import numpy as np
import torch

from policy_net import Policy, build_phi, load_traj, rollout, log, TOTAL_L, OUTDIR
from c0_geometry import (PAPERC, E1_CKPT, E1_PER_USER, E1_PER_USER_SHA, N_BANK, VAL_M0_FULL,
                         VAL_M0_TAIL, sha256, git_block, index_map_assert, bank_vectors, augment,
                         Emissions, dist)
import c1_pilot as C1

OUT = os.path.join(PAPERC, "stats_repair.json")
C0_NPZ = os.path.join(PAPERC, "c0_geometry_per_state.npz")
C0_JSON = os.path.join(PAPERC, "c0_geometry.json")
DEEP_DIR = os.path.join(OUTDIR, "policy_labels_deep_partial")
BOOT_C0 = 10000          # cluster-bootstrap resamples over users
BOOT_C1 = 200            # per-state bootstrap resamples over a cohort
SEED = 98765


# ============================================================================ helpers
def weighted_median(sorted_vals, w_sorted):
    """Median of `sorted_vals` under row weights `w_sorted`, both already in ascending value order."""
    c = np.cumsum(w_sorted)
    return float(sorted_vals[int(np.searchsorted(c, c[-1] * 0.5))])


def reading_rule(median_cos1, median_m, frac_ge3_band_abs):
    """DS_C0_GEOMETRY.md's registered cell assignment, verbatim from `c0_geometry.run`."""
    on = bool(median_cos1 >= 0.98 and median_m >= 0.90)
    between = bool(median_m <= 0.60 or frac_ge3_band_abs >= 0.25)
    return {"median_cos1": float(median_cos1), "median_m": float(median_m),
            "frac_states_ge3_in_5pct_band_abs": float(frac_ge3_band_abs),
            "on_the_grid": on, "between_candidates": between,
            "cell": "on_the_grid" if on else ("between_candidates" if between else "intermediate")}


# ============================================================================ C0
def c0_stats(args, env=None):
    Z = np.load(C0_NPZ)
    sha = sha256(E1_PER_USER)
    assert sha == E1_PER_USER_SHA, f"the E1 per-user artifact is not the pinned one ({sha})"
    B = np.load(E1_PER_USER)
    asked = np.asarray(B["asked"], np.int64)          # (10000, 16) pool indices, in ask order
    outs = np.asarray(B["outs"], np.int64)            # (10000, 16) ternary answers
    rows_e1 = np.asarray(B["rows"], np.int64)
    nU, L = asked.shape
    assert L == TOTAL_L
    cos1 = np.asarray(Z["pool_cos1"], np.float64)
    m = np.asarray(Z["pool_m"], np.float64)
    band = np.asarray(Z["pool_n_band_abs"], np.int64)
    pos = np.asarray(Z["pool_position"], np.int64)
    urow = np.asarray(Z["pool_user_row"], np.int64)
    assert len(cos1) == nU * L
    # the npz's row layout is user*L + position; confirm it against the banked user rows
    assert np.array_equal(urow, np.repeat(rows_e1, L)), \
        "the per-state npz's user_row column is not the banked rollout's user order"
    uidx = np.repeat(np.arange(nU), L)

    out = {"cohort": {"n_users": int(nU), "n_positions": int(L), "n_user_rows": int(nU * L)},
           "e1_per_user_npz_sha256": sha}

    # ---- 1. how many DISTINCT asked-history states are there really? -----------------------------
    # A state is (what was asked so far, how the user answered so far). Position 0 is one state for
    # everybody; the count can only grow with depth, and it is bounded by the tree's own branching.
    per_pos_states, state_id = [], np.zeros(nU * L, np.int64)
    off = 0
    for t in range(L):
        if t == 0:
            inv = np.zeros(nU, np.int64); nuq = 1
        else:
            key = np.concatenate([asked[:, :t], outs[:, :t]], 1)
            _u, inv = np.unique(key, axis=0, return_inverse=True)
            inv = np.asarray(inv, np.int64).ravel()      # numpy 2 returns a column for axis=0
            nuq = len(_u)
        per_pos_states.append(int(nuq))
        state_id[np.arange(nU) * L + t] = off + inv      # globally unique (position, history) id
        off += nuq
    n_states_total = int(off)
    out["distinct_asked_history_states"] = {
        "total_over_all_positions": n_states_total,
        "by_position": per_pos_states,
        "n_user_rows": int(nU * L),
        "users_per_state_mean": float(nU * L / n_states_total),
        "n_distinct_full_interviews": int(len(np.unique(
            np.concatenate([asked, outs], 1), axis=0))),
        "note": ("a state is (asked prefix, answer prefix), read from the banked rollout. C0's "
                 "distributions are over the 160,000 USER-ROWS; this is how many distinct states "
                 "those rows cover.")}
    log(f"[c0] {n_states_total} distinct asked-history states over {nU * L:,} user-rows "
        f"(by position: {per_pos_states})")

    # ---- 2. variance decomposition, per position -------------------------------------------------
    def between_share(x, lab):
        gm = np.bincount(lab, weights=x) / np.maximum(np.bincount(lab), 1)
        n = np.bincount(lab)
        keep = n > 0
        mu = x.mean()
        bet = float((n[keep] * (gm[keep] - mu) ** 2).sum() / len(x))
        tot = float(x.var())
        return {"between": bet, "total": tot, "between_share": (bet / tot) if tot > 0 else None,
                "n_groups": int(keep.sum())}

    vd = {"cos1": {}, "m": {}}
    for name, x in (("cos1", cos1), ("m", m)):
        vd[name]["overall"] = between_share(x, state_id)
        vd[name]["by_position"] = []
        for t in range(L):
            sel = pos == t
            lab = state_id[sel]
            _u, l2 = np.unique(lab, return_inverse=True)
            vd[name]["by_position"].append(between_share(x[sel], l2))
    out["variance_decomposition"] = vd
    log("[c0] between-state share of variance -- cos1 overall "
        f"{vd['cos1']['overall']['between_share']:.4f}, m overall "
        f"{vd['m']['overall']['between_share']:.4f}; cos1 by position "
        + " ".join(f"{d['between_share']:.2f}" for d in vd["cos1"]["by_position"]))

    # ---- 3. the maximum ---------------------------------------------------------------------------
    i = int(np.argmax(cos1))
    out["max_cos1"] = {"value": float(cos1[i]), "position": int(pos[i]),
                       "user_row": int(urow[i]), "state_id": int(state_id[i]),
                       "n_states_above_0.98": int((cos1 >= 0.98).sum()),
                       "n_states_above_0": int((cos1 > 0).sum()),
                       "frac_above_0": float((cos1 > 0).mean())}
    log(f"[c0] max cos1 {cos1[i]:.5f} (position {pos[i]}); above 0 at {(cos1 > 0).mean():.5f} of rows")

    # ---- 4. state-weighted medians ----------------------------------------------------------------
    # each DISTINCT state contributes once: its own median (its own fraction, for the band statistic)
    order = np.argsort(state_id, kind="stable")
    sid_s = state_id[order]
    bounds = np.flatnonzero(np.diff(sid_s)) + 1
    groups = np.split(order, bounds)
    sw = {}
    for name, x in (("cos1", cos1), ("m", m)):
        per = np.asarray([np.median(x[g]) for g in groups])
        sw[name] = {"state_weighted_median": float(np.median(per)),
                    "state_weighted_mean_of_state_medians": float(per.mean()),
                    "user_row_weighted_median": float(np.median(x)),
                    "n_states": int(len(groups))}
    perb = np.asarray([float((band[g] >= 3).mean()) for g in groups])
    sw["frac_ge3_in_5pct_band_abs"] = {
        "state_weighted_mean_of_state_fractions": float(perb.mean()),
        "state_weighted_median_of_state_fractions": float(np.median(perb)),
        "user_row_weighted": float((band >= 3).mean()), "n_states": int(len(groups))}
    out["state_weighted_vs_user_row_weighted"] = sw
    log(f"[c0] cos1 median: user-row {sw['cos1']['user_row_weighted_median']:.5f} vs state-weighted "
        f"{sw['cos1']['state_weighted_median']:.5f}; m: "
        f"{sw['m']['user_row_weighted_median']:.5f} vs {sw['m']['state_weighted_median']:.5f}")

    # ---- 5. cluster bootstrap over USERS -----------------------------------------------------------
    # The 16 rows of one user share that user's latent, so they are one cluster. Resample USERS.
    rng = np.random.default_rng(SEED)
    boot = {}
    for name, x in (("cos1", cos1), ("m", m)):
        o = np.argsort(x, kind="stable")
        xs = x[o]
        us = uidx[o]
        meds = np.empty(args.boot_c0, np.float64)
        t0 = time.time()
        for b in range(args.boot_c0):
            mult = rng.multinomial(nU, np.full(nU, 1.0 / nU)).astype(np.float64)
            meds[b] = weighted_median(xs, mult[us])
            if b == 0 or (b + 1) % 2500 == 0:
                log(f"  [bootstrap/{name}] {b + 1}/{args.boot_c0} "
                    f"({(time.time() - t0) / 60:.1f}m)")
        lo, hi = np.percentile(meds, [2.5, 97.5])
        boot[name] = {"point_median": float(np.median(x)), "n_resamples": int(args.boot_c0),
                      "seed": SEED, "cluster": "user (16 rows each)",
                      "ci95_lo": float(lo), "ci95_hi": float(hi),
                      "bootstrap_median_of_medians": float(np.median(meds)),
                      "bootstrap_sd": float(meds.std(ddof=1))}
        log(f"[c0] cluster bootstrap median {name}: {np.median(x):.5f} "
            f"95% CI [{lo:.5f}, {hi:.5f}]")
    out["cluster_bootstrap"] = boot

    # ---- 7. the registered reading rule, applied to the RANDOM FLOOR -------------------------------
    rc = np.asarray(Z["pool_rand_cos1"], np.float64)
    rm = np.asarray(Z["pool_rand_m"], np.float64)
    rb = np.asarray(Z["pool_rand_n_band_abs"], np.int64)
    out["registered_reading_rule_on_the_random_floor"] = dict(
        reading_rule(np.median(rc), np.median(rm), float((rb >= 3).mean())),
        _comment=("DS_C0_GEOMETRY.md registered a cell assignment, applied it to the emission and "
                  "never applied it to its own scale control. This is that rule, unchanged, run on "
                  "the norm-matched random direction C0 already banked."),
        emission_cell=reading_rule(np.median(cos1), np.median(m),
                                   float((band >= 3).mean()))["cell"])
    rr = out["registered_reading_rule_on_the_random_floor"]
    log(f"[c0] ★ the RANDOM FLOOR lands in cell '{rr['cell']}' "
        f"(median cos1 {rr['median_cos1']:.5f}, median m {rr['median_m']:.5f}, band "
        f"{rr['frac_states_ge3_in_5pct_band_abs']:.4f}); the emission's cell is "
        f"'{rr['emission_cell']}'")

    # ---- 6. mean-removed cos1 for the EMISSIONS ----------------------------------------------------
    if env is not None:
        out["mean_removed_emission_cos1"] = c0_mean_removed(env, Z, cos1)
    else:
        out["mean_removed_emission_cos1"] = {"skipped": True}
    return out


def c0_mean_removed(env, Z, cos1_banked):
    """★ The diagnostic C0 never ran: cos1 with the bank's shared component removed from BOTH sides.

    The emissions are not in the banked npz, so they are regenerated through `policy_net.rollout`
    with C0's own forward hook. The regenerated RAW cos1 is asserted equal to the banked column
    first -- if that holds, these are C0's emissions and not a lookalike, and the mean-removed
    number is about the same object every other C0 statistic describes."""
    E, pol, phi_small, pool, Ebank, Kbank, rows, pool2full, banked_asked = env
    log("[c0/mean-removed] regenerating the pool-arm emissions through rollout (C0's own hook)")
    cap = Emissions(pol)
    try:
        ro = rollout(E, pol, phi_small, pool, rows, batch=1000)
    finally:
        cap.close()
    U = cap.stack(len(rows))
    cap.buf = []
    assert np.array_equal(ro["asked"], banked_asked), (
        "GATE FAILED: the regenerated rollout is not the banked E1 rollout -- refusing to read a "
        "geometry number off it")
    Zhat = augment(U)
    snap = np.asarray(Z["pool_snap"], np.int64)
    A = Ebank.astype(np.float64)
    mu = A.mean(0)
    zh = Zhat.astype(np.float64)
    e = A[snap]
    raw = ((zh * e).sum(1) / (np.linalg.norm(zh, axis=1) * np.linalg.norm(e, axis=1) + 1e-30))
    d = float(np.abs(raw - np.asarray(cos1_banked, np.float64)).max())
    assert d < 1e-5, (f"the regenerated emissions do not reproduce the banked cos1 (max|diff| {d:g})")
    log(f"[c0/mean-removed] regenerated raw cos1 reproduces the banked column (max|diff| {d:g})")
    zc = zh - mu
    ec = e - mu
    mr = ((zc * ec).sum(1) / (np.linalg.norm(zc, axis=1) * np.linalg.norm(ec, axis=1) + 1e-30))
    # the same statistic for the random floor, so the new number has its own control
    rng = np.random.default_rng(SEED)
    R = rng.standard_normal(U.shape).astype(np.float32)
    R *= (np.linalg.norm(U, axis=1) / np.linalg.norm(R, axis=1))[:, None]
    Rz = augment(R).astype(np.float64) - mu
    mrr = ((Rz * ec).sum(1) / (np.linalg.norm(Rz, axis=1) * np.linalg.norm(ec, axis=1) + 1e-30))
    log(f"[c0/mean-removed] ★ median mean-removed cos1 {np.median(mr):.5f} "
        f"(raw {np.median(raw):.5f}); norm-matched random direction against the SAME snapped "
        f"entry: {np.median(mrr):.5f}")
    return {"raw_cos1": dist(raw), "mean_removed_cos1": dist(mr),
            "bank_mean_norm": float(np.linalg.norm(mu)),
            "bank_mean_norm_over_mean_entry_norm":
                float(np.linalg.norm(mu) / np.linalg.norm(A, axis=1).mean()),
            "regenerated_raw_cos1_max_abs_diff_vs_banked": d,
            "random_direction_mean_removed_cos1_to_the_same_snapped_entry": dist(mrr),
            "frac_mean_removed_above_0": float((mr > 0).mean()),
            "_comment": ("cos(z_hat - mu, e_snap - mu), mu = the bank mean. C0 reported the RAW "
                         "cosine (-0.182) and, separately, that the bank is dominated by one "
                         "shared direction; it never reported the cosine with that direction "
                         "removed from both sides. The random row uses a norm-matched Gaussian "
                         "emission scored against the SAME snapped bank entry, so it is a floor "
                         "for this statistic and not a different comparison.")}


# ============================================================================ C1
def load_peruser_gains(state, pool_index):
    """(n_cand, n_users) per-user FULL-NDCG gains for a deep sweep, plus the pool indices.

    The sidecar is append-only float32: record i is `peruser_order[i]`, laid out as n_users FULL
    values then n_users TAIL values; record 0 is the state's own per-user vector before any
    candidate, so the per-user GAIN is record i minus record 0. The cohort mean of the rebuilt gain
    is asserted against the banked `vals[q][0]` -- the number `c1_pilot.build_targets` uses -- so a
    layout misreading cannot pass silently.
    """
    fp = os.path.join(DEEP_DIR, state["source"])
    d = json.load(open(fp))
    n = int(d["n_users"])
    order = d["peruser_order"]
    raw = np.fromfile(os.path.join(DEEP_DIR, d["peruser_file"]), dtype=np.float32)
    assert raw.size == len(order) * 2 * n, (
        f"{state['sid']}: sidecar has {raw.size} floats, expected {len(order) * 2 * n}")
    M = raw.reshape(len(order), 2, n)[:, 0, :].astype(np.float64)      # FULL half only
    assert order[0] == "__base__"
    G = M[1:] - M[0][None, :]
    cand, keep = [], []
    for i, q in enumerate(order[1:]):
        a, b = q.split("_")
        pi = pool_index.get((int(a), int(b)))
        if pi is not None:
            cand.append(pi); keep.append(i)
    G = G[np.asarray(keep, np.int64)]
    cand = np.asarray(cand, np.int64)
    # contract: the cohort mean of the rebuilt per-user gains IS the banked scalar gain
    banked = np.asarray([float(d["vals"][order[1 + i]][0]) for i in keep], np.float64)
    err = float(np.abs(G.mean(1) - banked).max())
    assert err < 1e-6, (f"{state['sid']}: rebuilt per-user gains do not average to the banked "
                        f"gains (max|diff| {err:g})")
    return G, cand, np.asarray(d["rows"], np.int64), err


def c1_stats(args, env, m=5):
    E, pol, phi_small, pool, Ebank, Kbank, rows, pool2full, _ba = env
    pool_index = {q: i for i, q in enumerate(pool)}
    Epool, Kpool = bank_vectors(pol, phi_small)
    Ep = Epool.astype(np.float64)
    mu_bank = Ebank.astype(np.float64).mean(0)
    Dva = load_traj(train=False)
    reg, _extra = C1.load_the_42(pool_index, Dva)
    deep = [s for s in reg if s["group"] == "deep_sweep"]
    log(f"[c1] {len(deep)} of the registered {len(reg)} states carry a per-user gain sidecar "
        f"(the {len(reg) - len(deep)} single-pick heap states do not)")

    def target_from(gain_by_cand, cand, asked, mm):
        sel = np.asarray([j for j, c in enumerate(cand) if c not in asked], np.int64)
        g = gain_by_cand[sel]
        o = np.argsort(-g)[:mm]
        idx = cand[sel][o]
        w = np.maximum(g[o], 0.0)
        if w.sum() <= 0:
            return None, None, None
        w = w / w.sum()
        return (w[:, None] * Ep[idx]).sum(0), idx, w

    rows_out = []
    rng = np.random.default_rng(SEED)
    for s in deep:
        G, cand, urows, contract_err = load_peruser_gains(s, pool_index)
        n = G.shape[1]
        asked = {pool_index[qq] for qq in s["prefix"]}
        gmean = G.mean(1)
        ref_t, ref_idx, ref_w = target_from(gmean, cand, asked, m)
        # the reference target must be the pilot's target, built the pilot's way
        assert ref_idx is not None
        # ---- 8. bootstrap the cohort, watch the top-5 ------------------------------------------
        ov = np.empty(args.boot_c1, np.int64)
        cosb = np.empty(args.boot_c1, np.float64)
        refset = set(int(x) for x in ref_idx)
        for b in range(args.boot_c1):
            w = rng.multinomial(n, np.full(n, 1.0 / n)).astype(np.float64)
            gb = (G @ w) / n
            tb, ib, _wb = target_from(gb, cand, asked, m)
            ov[b] = len(refset & set(int(x) for x in ib))
            cosb[b] = float(tb @ ref_t / (np.linalg.norm(tb) * np.linalg.norm(ref_t) + 1e-30))
        # ---- 9. split-half --------------------------------------------------------------------
        perm = np.random.default_rng(SEED).permutation(n)
        h1, h2 = perm[:n // 2], perm[n // 2:]
        t1, i1, _ = target_from(G[:, h1].mean(1), cand, asked, m)
        t2, i2, _ = target_from(G[:, h2].mean(1), cand, asked, m)
        cos_sh = float(t1 @ t2 / (np.linalg.norm(t1) * np.linalg.norm(t2) + 1e-30))
        c1c, c2c = t1 - mu_bank, t2 - mu_bank
        cos_sh_mr = float(c1c @ c2c / (np.linalg.norm(c1c) * np.linalg.norm(c2c) + 1e-30))
        rows_out.append({
            "sid": s["sid"], "n_users": int(n), "n_candidates": int(len(cand)),
            "prefix_len": len(s["prefix"]),
            "peruser_gain_contract_max_abs_err": contract_err,
            "top5_pool_idx": [int(x) for x in ref_idx],
            "top5_weights": [float(x) for x in ref_w],
            "bootstrap": {"n_resamples": int(args.boot_c1), "seed": SEED,
                          "overlap_of_5_mean": float(ov.mean()),
                          "overlap_of_5_median": float(np.median(ov)),
                          "overlap_of_5_min": int(ov.min()),
                          "frac_resamples_all_5_survive": float((ov == m).mean()),
                          "frac_resamples_top5_overlap_ge_3": float((ov >= 3).mean()),
                          "target_cos_to_reference": {
                              "mean": float(cosb.mean()), "median": float(np.median(cosb)),
                              "p5": float(np.percentile(cosb, 5)),
                              "min": float(cosb.min())}},
            "split_half": {"n_half1": int(len(h1)), "n_half2": int(len(h2)),
                           "target_cos": cos_sh, "target_cos_mean_removed": cos_sh_mr,
                           "top5_overlap_between_halves": int(
                               len(set(int(x) for x in i1) & set(int(x) for x in i2)))}})
        r = rows_out[-1]
        log(f"[c1] {s['sid'][:46]:<46} n={n:>5} | boot top-5 overlap "
            f"{r['bootstrap']['overlap_of_5_mean']:.2f}/5 (all 5 survive "
            f"{r['bootstrap']['frac_resamples_all_5_survive']:.3f}) | boot target cos "
            f"{r['bootstrap']['target_cos_to_reference']['median']:.4f} | split-half cos "
            f"{cos_sh:.4f} (mean-removed {cos_sh_mr:.4f}, overlap "
            f"{r['split_half']['top5_overlap_between_halves']}/5)")

    def agg(f):
        return dist([f(r) for r in rows_out])

    summary = {
        "n_states": len(rows_out), "m": m,
        "_comment": ("The 35 deep-sweep states of the registered 42. The 7 single-pick heap states "
                     "have no per-user sidecar and are excluded, not imputed."),
        "bootstrap_overlap_of_5_mean": agg(lambda r: r["bootstrap"]["overlap_of_5_mean"]),
        "bootstrap_frac_all_5_survive": agg(
            lambda r: r["bootstrap"]["frac_resamples_all_5_survive"]),
        "bootstrap_target_cos_median": agg(
            lambda r: r["bootstrap"]["target_cos_to_reference"]["median"]),
        "split_half_target_cos": agg(lambda r: r["split_half"]["target_cos"]),
        "split_half_target_cos_mean_removed": agg(
            lambda r: r["split_half"]["target_cos_mean_removed"]),
        "split_half_top5_overlap": agg(
            lambda r: r["split_half"]["top5_overlap_between_halves"]),
        "n_states_with_all_5_surviving_in_over_half_the_resamples": int(sum(
            r["bootstrap"]["frac_resamples_all_5_survive"] > 0.5 for r in rows_out))}
    log(f"[c1] SUMMARY over {len(rows_out)} states: split-half target cos median "
        f"{summary['split_half_target_cos']['median']:.4f} "
        f"[{summary['split_half_target_cos']['min']:.4f}, "
        f"{summary['split_half_target_cos']['max']:.4f}]; mean-removed median "
        f"{summary['split_half_target_cos_mean_removed']['median']:.4f}; bootstrap top-5 overlap "
        f"mean {summary['bootstrap_overlap_of_5_mean']['median']:.3f}/5")
    return {"summary": summary, "per_state": rows_out}


# ============================================================================ the run
def build_env(args):
    from strategy_channel_suite import build_greedy_env, contract_selftest, genv_mean
    from ndcg_tree import build_pool, leak_assert
    blob = torch.load(args.ckpt, map_location="cpu")
    sha = sha256(args.ckpt)
    assert sha == C1.C0_CKPT_SHA, f"this is not the C0/C1-measured checkpoint ({sha})"
    E = build_greedy_env(args.snapshot, "validation", args.concept_ckpt, "paperC/repair")
    contract_selftest(E, E["rows"]); leak_assert(E, E["rows"], "paperC/repair")
    rows = E["rows"]
    assert len(rows) == 10000
    f0, t0_ = genv_mean(E, [], rows)
    assert f0 == VAL_M0_FULL and t0_ == VAL_M0_TAIL, f"CANONICAL SNAP FAILED: {f0!r}/{t0_!r}"
    log(f"[gate] canonical m=0 snap full={f0!r} tail={t0_!r} -- both exact")
    pool = [(int(a), int(b)) for a, b in blob["pool"]]
    pool_e, _, _ = build_pool(E, 500, 500)
    assert pool_e == pool
    full_pool = [(0, i) for i in range(E["ctx"].ni)] + [(1, c) for c in range(E["C"])]
    assert len(full_pool) == N_BANK
    pool2full = np.asarray([(i if ch == 0 else E["ctx"].ni + i) for ch, i in pool], np.int64)
    norm = (blob["norm_mu"], blob["norm_sd"])
    phi_small, _ = build_phi(E, pool, norm=norm)
    phi_full, _ = build_phi(E, full_pool, norm=norm)
    pol = Policy(E["ctx"].d, phi_full.shape[1], len(full_pool), arch=blob["arch"],
                 state=blob["state"], use_pos=blob["use_pos"])
    pol.load_state_dict(blob["model"]); pol.eval()
    Ebank, Kbank = bank_vectors(pol, phi_full)
    Epool, _ = bank_vectors(pol, phi_small)
    err = index_map_assert(Ebank, pool2full, Epool)
    log(f"[gate] pool2full maps the pool onto the bank (max|diff| {err:g})")
    banked_asked = np.asarray(np.load(E1_PER_USER)["asked"], np.int64)
    return (E, pol, phi_small, pool, Ebank, Kbank, rows, pool2full, banked_asked), sha


def main():
    ap = argparse.ArgumentParser(description="Paper C -- repair statistics (C0 + C1)")
    ap.add_argument("--ckpt", default=E1_CKPT)
    ap.add_argument("--boot_c0", type=int, default=BOOT_C0)
    ap.add_argument("--boot_c1", type=int, default=BOOT_C1)
    ap.add_argument("--blocks", default="c0,c1")
    ap.add_argument("--no_emissions", action="store_true",
                    help="skip the mean-removed emission diagnostic (it needs a rollout)")
    ap.add_argument("--snapshot", default=None)
    ap.add_argument("--concept_ckpt", default=None)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--allow_dirty", action="store_true",
                    help="EMERGENCY ONLY: run with an uncommitted src/instrument (HARD RULE 10)")
    args = ap.parse_args()
    args.blocks = [b for b in args.blocks.split(",") if b]
    from strategy_channel_suite import CERT_TOWER, CERT_CONC
    args.snapshot = args.snapshot or CERT_TOWER
    args.concept_ckpt = args.concept_ckpt or CERT_CONC
    log("#" * 100)
    log("# PAPER C -- REPAIR STATISTICS. Read-only over banked artifacts (plus one regenerated "
        "rollout, gated against the banked one). VALIDATION ONLY; nothing trained; the 10,000 TEST "
        "users are not built, not loaded, not touched.")
    log("#" * 100)
    t00 = time.time()
    GB = git_block(allow_dirty=args.allow_dirty, threads=args.threads)
    log(f"[hardrule10] src/instrument clean={GB['src_instrument_clean']} rev={GB['rev'][:8]}")

    out = {"analysis": "paperC_stats_repair",
           "produced_by": "src/instrument/paperc_stats_repair.py",
           "_comment": ("Intervals, decompositions and stability checks the C0 and C1 runs "
                        "reported as bare point estimates. Nothing here re-measures or re-trains."),
           "citable": True, "test_touched": False, "split": "validation",
           "git": GB, "seed": SEED,
           "checkpoint": {"path": os.path.relpath(args.ckpt, _ROOT)}}

    def flush():
        out["minutes"] = round((time.time() - t00) / 60, 2)
        with open(OUT + ".tmp", "w") as fh:
            json.dump(out, fh, indent=1, default=float)
        os.replace(OUT + ".tmp", OUT)

    flush()
    env = None
    if not args.no_emissions or "c1" in args.blocks:
        env, sha = build_env(args)
        out["checkpoint"]["sha256"] = sha
    if "c0" in args.blocks:
        out["C0"] = c0_stats(args, None if args.no_emissions else env)
        flush()
    if "c1" in args.blocks:
        out["C1"] = c1_stats(args, env)
        flush()
    flush()
    log(f"[done] wrote {OUT} in {out['minutes']:.1f} min")
    return out


if __name__ == "__main__":
    main()
