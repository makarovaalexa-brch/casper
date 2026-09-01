r"""PAPER C / C7 -- LOCAL (NON-PARAMETRIC) PROFILE TRANSFER at held-out interview states.

Read-only over BANKED artifacts. Nothing is trained (a distance-weighted average of banked profiles
is not a fit), no LM is called, no test user is built, loaded or touched. `test_touched` is False.

WHY THIS RUNS
-------------
`experiments/paperC/c5_diagnostics.json` says three things that point at exactly one untested idea:

  D2  the NEAREST measured state by cohort-mean latent predicts a state's gain profile at Spearman
      +0.617 against +0.150 for a mean/random other state -- margin +0.4666 [+0.4237, +0.5069],
      better at 71/72 states.
  D4/D5  every GLOBAL LINEAR probe (ridge over the latent, the warm emission, structural features)
      FAILS to beat a state-blind baseline on the primary split.
  D3  a state-blind CONSTANT beats the trained 526k-parameter policy at held-out states by ~+0.89
      raw cosine.

So the signal is LOCAL, not globally linear, and no local predictor has been tested. This is that
test: predict a held-out state's gain profile as a distance-weighted average of its k nearest
TRAINING states' profiles, rebuild the emission from the prediction with `c5_diagnostics`' own rule,
and score both against the SAME splits C1-final registered.

THE FLOOR RULE (docs/HARSH_REVIEW_POINTS.md, 2026-09-01)
--------------------------------------------------------
EVERY number here is printed beside its floor, and the headline verdict is a PAIRED contrast against
the state-blind floor, not an absolute. The floors carried throughout:

  profile level   global-mean TRAINING profile | popularity prior | a randomly permuted profile
  emission level  the same three, rebuilt into blends, PLUS the two state-blind CONSTANT targets
                  (D3's leave-one-out mean of the other 71 targets, and the split-clean mean of the
                  TRAINING targets), PLUS random directions with matched norm, PLUS the trained
                  policy's own banked held-out cosines.

A kNN result that does not beat the state-blind floor is a NULL and is reported as one.

MEAN-REMOVAL CONVENTION, STATED ONCE AND USED EVERYWHERE
--------------------------------------------------------
Identical to C5/C1/C0. Anything in the 129-d homogeneous emission space -- targets, candidate
embeddings, rebuilt blends -- is mean-removed by `mu`, the mean of the FULL 19,390-entry bank. RAW
cosines are reported ALONGSIDE wherever C1-final's own registered number is raw (its `cos_own` is a
raw cosine). The 200-d latent has no bank; it is mean-removed by the mean over the TRAINING states of
the split being evaluated (NOT over all 72 -- centring on the held-out states would let their inputs
into the predictor, and the point of this run is to be clean about exactly that).

WHAT IS REUSED AND WHAT IS REBUILT
----------------------------------
REUSED from `experiments/paperC/c5_diagnostics_pairs.npz` (no recomputation):
  gain_profiles_shared (72 x 957)   the measured gain profiles on the shared candidate set
  shared_candidates (957,)          the pool indices of that set
  state_mean_latent (72 x 200)      the cohort-mean latent per state
  pair_tree_path_distance (72x72)   the tree-path metric
  state_sid / state_depth           the split definitions
  d3_loo_* / d5_*                   the banked floors these numbers are asserted against
REUSED from `experiments/paperC/c1_final_per_state.npz`:
  target_m5 (72 x 129)              the true blend targets
  cos_own_<split> (72,)             the trained policy's own RAW held-out cosines
REBUILT (the only reason an environment is built at all): `Epool` and `mu`, the 129-d candidate
embeddings and the bank mean, which the npz does not carry and without which no emission can be
constructed. The rebuild is VERIFIED, not trusted: gates G1-G4 below reproduce four banked C5
quantities to <1e-9 from the rebuilt objects. The expensive parts of C5 (the 85,289-user bank
descent, the cohort-latent accumulation, the 72 gain-table sweeps) are SKIPPED because their outputs
are already banked.

FIREWALL. Train-side states only, on the 85,289-user cold bank. The 10,000 canonical TEST users are
not touched.
"""
from __future__ import annotations

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
sys.path.insert(0, _HERE)
if "--threads" not in sys.argv:
    sys.argv += ["--threads", "2"]
_NT = sys.argv[sys.argv.index("--threads") + 1]
os.environ["OMP_NUM_THREADS"] = _NT
os.environ.setdefault("MKL_NUM_THREADS", _NT)
os.environ.setdefault("OPENBLAS_NUM_THREADS", _NT)

import argparse
import json
import time

import numpy as np
import torch

from policy_net import Policy, build_phi, log, paired_bootstrap
from c0_geometry import (PAPERC, N_BANK, sha256, git_block, savez_atomic, index_map_assert,
                         bank_vectors, dist)
import c1_final as CF
# IMPORTED, never reimplemented: the helpers C5's own numbers were produced with.
from c5_diagnostics import cosrows, ranks, boot_ci, SEED, N_BOOT

KS = (1, 3, 5, 10)
# bandwidth grid, as multipliers of the median TRAIN-TRAIN distance in whichever metric is in play
H_MULT = np.logspace(-2.0, 1.0, 25)


# ============================================================================ scoring primitives
def ndcg_at(pred, true, k=5):
    """C5's `ndcg_at`, character for character -- the same object must be scored the same way."""
    rel = np.maximum(true, 0.0)
    o = np.argsort(-pred)[:k]
    dcg = float((rel[o] / np.log2(np.arange(2, k + 2))).sum())
    io = np.argsort(-rel)[:k]
    idcg = float((rel[io] / np.log2(np.arange(2, k + 2))).sum())
    return dcg / idcg if idcg > 0 else np.nan


def sp_score(a, b):
    """C5's `sp_score`: Spearman as the Pearson correlation of average ranks."""
    ra, rb = ranks(a), ranks(b)
    ra -= ra.mean(); rb -= rb.mean()
    return float(ra @ rb / (np.linalg.norm(ra) * np.linalg.norm(rb) + 1e-300))


def corr_boot(x, y, kind="pearson", n_boot=N_BOOT, seed=SEED):
    """Correlation over the held-out states with a percentile bootstrap over those same states."""
    x = np.asarray(x, np.float64); y = np.asarray(y, np.float64)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    n = len(x)

    def c(a, b):
        if kind == "spearman":
            a, b = ranks(a), ranks(b)
        a = a - a.mean(); b = b - b.mean()
        d = np.linalg.norm(a) * np.linalg.norm(b)
        return float(a @ b / d) if d > 0 else np.nan
    v = c(x, y)
    rng = np.random.default_rng(seed)
    bs = np.empty(n_boot)
    for t in range(n_boot):
        s = rng.integers(0, n, n)
        bs[t] = c(x[s], y[s])
    bs = bs[np.isfinite(bs)]
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return {"r": v, "lo": float(lo), "hi": float(hi), "kind": kind, "n_states": int(n),
            "ci": "percentile bootstrap over the held-out states, 10k, seed 98765",
            "excludes_zero": bool(lo > 0 or hi < 0)}


# ============================================================================ the local predictor
def knn_predict(D, G, query, train, k, h):
    """Distance-weighted average of the k nearest TRAINING profiles.

    `D` is a full 72x72 distance matrix; only the (query, train) block is ever read, so a held-out
    state can never be its own neighbour and two held-out states can never see each other.
    Weights are `softmax(-d/h)`; the max is subtracted first for numerical safety, which leaves the
    softmax unchanged. `h=None` means k=1 (the weight is 1 regardless of any bandwidth).
    """
    train = np.asarray(train, np.int64)
    P = np.empty((len(query), G.shape[1]), np.float64)
    W = np.zeros((len(query), k), np.float64)
    NB = np.zeros((len(query), k), np.int64)
    DD = np.zeros((len(query), k), np.float64)
    for r, q in enumerate(query):
        d = D[q, train]
        o = np.argsort(d, kind="stable")[:k]
        nb, dd = train[o], d[o]
        if k == 1 or h is None:
            w = np.ones(len(nb), np.float64)
        else:
            e = -(dd - dd.min()) / max(float(h), 1e-12)
            w = np.exp(e - e.max())
        w = w / w.sum()
        P[r] = w @ G[nb]
        W[r, :len(w)] = w; NB[r, :len(nb)] = nb; DD[r, :len(dd)] = dd
    return P, NB, DD, W


def choose_h_loo(D, G, train, k):
    """Bandwidth by leave-one-out WITHIN THE TRAINING STATES ONLY. No held-out state is read.

    Returns (best_h, curve, best_loo_spearman). For k=1 the bandwidth is inert and reported as None.
    """
    train = np.asarray(train, np.int64)
    med = float(np.median(D[np.ix_(train, train)][~np.eye(len(train), dtype=bool)]))
    med = max(med, 1e-9)
    if k == 1:
        sc = []
        for s in train:
            others = train[train != s]
            p, _, _, _ = knn_predict(D, G, [s], others, 1, None)
            sc.append(sp_score(p[0], G[s]))
        return None, [], float(np.mean(sc))
    curve = []
    for m in H_MULT:
        h = m * med
        sc = []
        for s in train:
            others = train[train != s]
            p, _, _, _ = knn_predict(D, G, [s], others, k, h)
            sc.append(sp_score(p[0], G[s]))
        curve.append((float(h), float(np.mean(sc))))
    best = max(curve, key=lambda t: t[1])
    return best[0], curve, best[1]


# ============================================================================ the run
def run(args):
    from strategy_channel_suite import build_greedy_env, contract_selftest, genv_mean
    from ndcg_tree import build_pool, branch_of_selftest, CONCSTATS, N_TRAIN_USERS
    t00 = time.time()
    os.makedirs(PAPERC, exist_ok=True)
    np.random.seed(SEED); torch.manual_seed(SEED)
    GB = git_block(allow_dirty=args.allow_dirty, threads=args.threads)
    log(f"[hardrule10] src/instrument clean={GB['src_instrument_clean']} rev={GB['rev'][:8]}")
    torch.set_num_threads(args.threads)

    # ---- banked inputs (the whole point: C5 already measured these) ----------------------------
    p_pairs = os.path.join(PAPERC, "c5_diagnostics_pairs.npz")
    p_c1 = os.path.join(PAPERC, "c1_final_per_state.npz")
    Z5 = np.load(p_pairs, allow_pickle=True)
    Z1 = np.load(p_c1, allow_pickle=True)
    S = np.asarray(Z5["shared_candidates"], np.int64)             # pool indices of the shared set
    Xz = np.asarray(Z5["state_mean_latent"], np.float64)          # 72 x 200 cohort-mean latents
    Dtree = np.asarray(Z5["pair_tree_path_distance"], np.float64)  # 72 x 72 tree-path distance
    sids = [str(s) for s in Z5["state_sid"]]
    depth = np.asarray(Z5["state_depth"], np.int64)
    ns = len(sids)
    assert ns == 72 and Xz.shape[0] == 72
    assert list(Z1["state_sid"]) == sids, "the two banked npz files are not the same 72 states"
    pol_cos = {"primary_root_like_subtree": np.asarray(Z1["cos_own_primary_root_like_subtree"],
                                                       np.float64),
               "secondary_depth_ge_5": np.asarray(Z1["cos_own_secondary_depth_ge_5"], np.float64)}
    log(f"[banked] {ns} states | {len(S)} shared candidates "
        f"| reusing C5's cohort latents and tree metric verbatim")

    # ---- the ONE thing the npz does not carry: Epool and mu ------------------------------------
    blob = torch.load(CF.WARM, map_location="cpu")
    assert blob["arch"] == "equivariant" and blob["state"] == "both"
    assert len(blob["ext"]) == CF.N_EXT
    warm_sha = sha256(CF.WARM)
    E = build_greedy_env(args.snapshot, "train", args.concept_ckpt, "paperC/c7",
                         cohort_slice=(0, args.snap_users))
    contract_selftest(E, E["rows"])
    f0, t0_ = genv_mean(E, [], E["rows"])
    assert f0 == CF.TRAIN_M0_FULL, f"CANONICAL SNAP FAILED: {f0!r} vs {CF.TRAIN_M0_FULL!r}"
    log(f"[gate G5] canonical TRAIN m=0 snap full={f0!r} tail={t0_!r}")

    pool, _ip, _cp = build_pool(E, 500, 500)
    pool = [(int(a), int(b)) for a, b in pool]
    assert pool == [(int(a), int(b)) for a, b in blob["pool"]]
    n_pool = len(pool)
    branch_of_selftest(E, E["rows"][:2000], [pool[0], pool[500], (1, 114)])
    ext = [(int(a), int(b)) for a, b in blob["ext"]]
    norm = (np.asarray(blob["norm"][0], np.float32), np.asarray(blob["norm"][1], np.float32))
    phi_ext, _ = build_phi(E, ext, norm=norm)
    full_pool = [(0, i) for i in range(E["ctx"].ni)] + [(1, c) for c in range(E["C"])]
    assert len(full_pool) == N_BANK
    phi_full, _ = build_phi(E, full_pool, norm=norm)
    pool2full = np.asarray([(i if ch == 0 else E["ctx"].ni + i) for ch, i in pool], np.int64)
    pol0 = Policy(200, 205, n_pool, arch="equivariant", state="both", use_pos=True)
    pol0.load_state_dict(blob["sd"]); pol0.eval()
    assert sum(p.numel() for p in pol0.parameters()) == 526542
    Ebank, _ = bank_vectors(pol0, phi_full)
    Epool, _Kpool = bank_vectors(pol0, phi_ext[:n_pool])
    map_err = index_map_assert(Ebank, pool2full, Epool)
    mu = Ebank.astype(np.float64).mean(0)
    log(f"[gate G6] {Ebank.shape} descriptors; ||mu|| {np.linalg.norm(mu):.6f}; map err {map_err:g}")
    del Ebank, phi_full

    cz = np.load(CONCSTATS)
    ans_rate = (cz["n"] / N_TRAIN_USERS).astype(np.float64)
    icnt = np.asarray(E["ctx"].cnt, np.float64)
    pop_prior = np.asarray([(icnt[i] / max(icnt.max(), 1e-9)) if ch == 0 else
                            (ans_rate[i] / max(ans_rate.max(), 1e-9)) for ch, i in pool], np.float64)

    # ---- the profiles and targets: reconstructed by C5's OWN code, then ASSERTED against the -----
    # banked arrays. Reconstructing rather than reading the float32 npz is what lets the gates below
    # run at 1e-9 instead of at the float32 rounding of the banked copies.
    from c1_pilot import build_targets
    pool_index = {q: i for i, q in enumerate(pool)}
    tree = json.load(open(CF.TREE_JSON))
    assert tree["split"] == "train" and int(tree["total_L"]) == CF.TOTAL_L
    assert tree["test_touched"] is False
    states, gain_chk = CF.load_sweeps(tree["nodes"], pool_index)
    assert [s["sid"] for s in states] == sids, "the sweeps are not C5's 72 states, in C5's order"
    Tt, _TKt, _meta = build_targets(states, Epool, _Kpool, pool_index, args.m)
    dT = float(np.abs(Tt - np.asarray(Z1["target_m5"], np.float64)).max())
    assert dT < 1e-4, (f"the reconstructed m={args.m} targets differ from C1-final's banked ones by "
                       f"{dT:g} -- STOP, this run would be describing a different object")
    G = np.asarray([[s["gains"][int(q)] for q in S] for s in states], np.float64)
    dG = float(np.abs(G - np.asarray(Z5["gain_profiles_shared"], np.float64)).max())
    assert dG < 1e-5, f"the reconstructed gain profiles differ from C5's banked ones by {dG:g}"
    out_gates_profiles = {"sweep_argmax_equals_splitter_max_abs_gain_err": gain_chk,
                          "targets_match_c1_final_banked_max_abs_diff": dT,
                          "profiles_match_c5_banked_max_abs_diff": dG}
    log(f"[gate G0] targets rebuilt from the SAME code; max|diff| vs C1-final banked {dT:g}; "
        f"gain profiles vs C5 banked {dG:g}")

    Tt = np.asarray(Tt, np.float64)
    Tmr = Tt - mu[None, :]
    Epool_S = Epool[S].astype(np.float64)

    def blend_from_profile(pred, m=5):
        """C5's `blend_from_profile`, unchanged: top-`m` of the predicted profile, weights the
        normalised POSITIVE part, over the shared-set candidate embeddings."""
        o = np.argsort(-pred)[:m]
        w = np.maximum(pred[o], 0.0)
        if w.sum() <= 0:
            return None
        w = w / w.sum()
        return (w[:, None] * Epool_S[o]).sum(0)

    # ---- splits, defined exactly as C1-final / C5 define them ----------------------------------
    held_pri = np.asarray([i for i, s in enumerate(sids)
                           if s == "root.like" or s.startswith("root.like.")], np.int64)
    held_sec = np.flatnonzero(depth >= 5).astype(np.int64)
    assert len(held_pri) == 15 and len(held_sec) == 30
    SPLITS = {"primary_root_like_subtree": held_pri, "secondary_depth_ge_5": held_sec}

    out = {"analysis": "paperC_C7_knn_profile_transfer", "citable": True, "test_touched": False,
           "split": "train_states",
           "purpose": "test LOCAL (non-parametric) profile transfer at held-out interview states, "
                      "the one lead C5's diagnostics point at and no run has followed",
           "reads": ["experiments/paperC/c5_diagnostics.json",
                     "experiments/paperC/c5_diagnostics_pairs.npz",
                     "experiments/paperC/c1_final_per_state.npz",
                     "docs/results/C1_FINAL.md (CORRECTION 2026-09-01)",
                     "docs/HARSH_REVIEW_POINTS.md (the floor rule)"],
           "reuse": {"from_c5_diagnostics_pairs_npz_USED_AS_INPUT":
                         ["shared_candidates", "state_mean_latent", "pair_tree_path_distance",
                          "state_sid", "state_depth"],
                     "from_the_banked_npz_USED_AS_A_GATE":
                         ["gain_profiles_shared", "d3_loo_MEAN_REMOVED_bank_mu",
                          "d5_<cell>_baseline__global_mean_train_profile_*",
                          "d5_<cell>_baseline__popularity_prior_*", "c1_final target_m5"],
                     "from_c1_final_per_state_npz": ["cos_own_<split>"],
                     "rebuilt_because_not_banked": ["Epool (1000 x 129 candidate embeddings)",
                                                    "mu (the 19,390-entry bank mean)",
                                                    "pop_prior (the pool's popularity ranking)",
                                                    "the 72 gain tables + targets, rebuilt by C5's "
                                                    "own load_sweeps/build_targets so the gates can "
                                                    "run at 1e-9 rather than at the float32 "
                                                    "rounding of the banked copies"],
                     "skipped_because_banked": ["the 85,289-user bank descent", "CF.descend",
                                                "CF.verify_labels", "state_mean_z (cohort latents)"],
                     "imported_not_reimplemented": ["c5_diagnostics.cosrows/ranks/boot_ci",
                                                    "policy_net.paired_bootstrap",
                                                    "c0_geometry.dist/savez_atomic/git_block",
                                                    "c5_diagnostics' ndcg_at/sp_score/"
                                                    "blend_from_profile rules, copied verbatim"]},
           "git": GB,
           "mean_removal_convention": {
               "emission_space_129d": "mean-removed by mu = the mean of the FULL 19,390-entry bank "
                                      "(C0/C1/C5 convention). RAW cosines reported alongside "
                                      "wherever C1-final's own registered number is raw.",
               "latent_space_200d": "mean-removed by the mean over the TRAINING states of the split "
                                    "being evaluated -- NOT over all 72, so no held-out state's "
                                    "input enters the predictor's own centring.",
               "mu_norm": float(np.linalg.norm(mu))},
           "warm_start": {"path": os.path.relpath(CF.WARM, _ROOT), "sha256": warm_sha},
           "n_states": ns, "n_shared_candidates": int(len(S)),
           "splits": {k: {"n_held_out": int(len(v)), "held_out_sids": [sids[int(i)] for i in v]}
                      for k, v in SPLITS.items()},
           "gates": dict({"canonical_snap_train_m0_full": f0, "canonical_snap_train_m0_tail": t0_,
                          "pool2full_index_map_max_abs_err": map_err}, **out_gates_profiles),
           "knn_design": {
               "predictor": "profile_hat(s) = sum_j softmax(-d(s,j)/h) * profile(j) over the k "
                            "nearest TRAINING states j; then C5's blend rule (top-5, weights the "
                            "normalised positive part) turns the predicted profile into an emission",
               "k_grid": list(KS),
               "bandwidth": "h = m * median(train-train distance), m over 25 log-spaced multipliers "
                            "in [1e-2, 1e1]; chosen by LEAVE-ONE-OUT SPEARMAN WITHIN THE TRAINING "
                            "STATES ONLY. k=1 makes h inert and reports h=None.",
               "distances": {
                   "latent": "1 - cos in the 200-d cohort-mean latent, centred on the TRAIN mean",
                   "tree": "the tree-path distance d(i)+d(j)-2*depth(LCA), banked by C5",
                   "both": "the sum of the two, each z-scored on the TRAIN x TRAIN off-diagonal"},
               "selection_rule": "the headline arm is the (distance, k) whose LOO-within-train "
                                 "Spearman is highest -- chosen before any held-out state is scored"},
           "cells": {}}
    p_json = os.path.join(PAPERC, f"c7_knn{args.out_tag}.json")
    p_npz = os.path.join(PAPERC, f"c7_knn{args.out_tag}_per_state.npz")
    npz = {"state_sid": np.asarray(sids), "state_depth": depth,
           "shared_candidates": S}

    def flush():
        out["minutes"] = round((time.time() - t00) / 60, 2)
        with open(p_json + ".tmp", "w") as fh:
            json.dump(out, fh, indent=1, default=float)
        os.replace(p_json + ".tmp", p_json)
        if npz:
            savez_atomic(p_npz, **npz)

    flush()

    # ======================================================================== the by-product the
    # paper wants regardless: how much of a target IS the common component?
    d3_banked = np.asarray(Z5["d3_loo_MEAN_REMOVED_bank_mu"], np.float64)
    fr_norm = np.empty(ns); fr_en = np.empty(ns); loo_cos = np.empty(ns)
    for i in range(ns):
        t = Tmr[i]
        o = np.setdiff1d(np.arange(ns), [i])
        c = Tmr[o].mean(0)
        ch = c / (np.linalg.norm(c) + 1e-300)
        a = float(t @ ch)
        nt = float(np.linalg.norm(t))
        loo_cos[i] = a / (nt + 1e-300)
        fr_norm[i] = abs(a) / (nt + 1e-300)
        fr_en[i] = (a * a) / (nt * nt + 1e-300)
    g3 = float(np.abs(loo_cos - d3_banked).max())
    assert g3 < 1e-9, f"GATE G3 FAILED: LOO cosine differs from C5's banked D3 by {g3:g}"
    out["gates"]["G3_reproduces_c5_D3_loo_cos_max_abs_err"] = g3
    resid_en = 1.0 - fr_en
    out["common_vs_residual"] = {
        "definition": "for each state, decompose its MEAN-REMOVED target t into the component along "
                      "the LOO mean of the OTHER 71 mean-removed targets (the COMMON direction) and "
                      "the orthogonal STATE-SPECIFIC residual. The norm fraction is |cos| (C5's D3 "
                      "number); the ENERGY fraction is cos^2 and is the one to quote.",
        "common_ENERGY_fraction_of_mean_removed_target": dist(fr_en),
        "common_ENERGY_fraction_boot": boot_ci(fr_en, N_BOOT, SEED),
        "state_specific_residual_ENERGY_fraction": dist(resid_en),
        "state_specific_residual_ENERGY_fraction_boot": boot_ci(resid_en, N_BOOT, SEED),
        "common_NORM_fraction_equals_abs_cos": dist(fr_norm),
        "headline": f"the common component carries {float(fr_en.mean()):.4f} of a target's "
                    f"mean-removed ENERGY; the state-specific residual carries "
                    f"{float(resid_en.mean()):.4f}"}
    npz["common_energy_fraction"] = fr_en
    npz["residual_energy_fraction"] = resid_en
    npz["d3_loo_cos_mr"] = loo_cos
    log(f"[common/residual] common energy {fr_en.mean():.4f} "
        f"[{boot_ci(fr_en, 2000, SEED)['lo']:.4f}, {boot_ci(fr_en, 2000, SEED)['hi']:.4f}] "
        f"| residual energy {resid_en.mean():.4f} | norm fraction (=|cos|) {fr_norm.mean():.4f}")
    flush()

    # ======================================================================== per split
    rng0 = np.random.default_rng(SEED)
    for cell, held in SPLITS.items():
        tr = np.setdiff1d(np.arange(ns), held)
        log(f"=== {cell}: {len(tr)} training states, {len(held)} held out ===")

        # ---- the three distances, built with TRAIN-ONLY centring ------------------------------
        Xc = Xz - Xz[tr].mean(0, keepdims=True)
        Xn = Xc / (np.linalg.norm(Xc, axis=1, keepdims=True) + 1e-300)
        Dlat = 1.0 - Xn @ Xn.T
        off = ~np.eye(len(tr), dtype=bool)
        m1 = float(Dlat[np.ix_(tr, tr)][off].mean()); s1 = float(Dlat[np.ix_(tr, tr)][off].std())
        m2 = float(Dtree[np.ix_(tr, tr)][off].mean()); s2 = float(Dtree[np.ix_(tr, tr)][off].std())
        Dboth = (Dlat - m1) / max(s1, 1e-12) + (Dtree - m2) / max(s2, 1e-12)
        DISTS = {"latent": Dlat, "tree": Dtree, "both": Dboth}

        # ---- the floors -----------------------------------------------------------------------
        gm = G[tr].mean(0)
        perm = rng0.permutation(len(S))
        prof_floors = {
            "floor__global_mean_train_profile": np.tile(gm, (len(held), 1)),
            "floor__popularity_prior": np.tile(pop_prior[S], (len(held), 1)),
            "floor__chance_random_permuted_profile": np.tile(gm[perm], (len(held), 1))}
        # state-blind CONSTANT targets (emission level only -- they are targets, not profiles)
        const_train = Tmr[tr].mean(0)
        const_d3 = np.stack([Tmr[np.setdiff1d(np.arange(ns), [i])].mean(0) for i in held])
        RD = rng0.standard_normal((len(held), Tt.shape[1]))
        RD *= (np.linalg.norm(Tmr[held], axis=1) / np.linalg.norm(RD, axis=1))[:, None]
        emis_only = {
            "floor__state_blind_constant_target_TRAIN_MEAN":
                np.tile(const_train, (len(held), 1)),
            "floor__state_blind_constant_target_D3_LOO_mean_of_other_71": const_d3,
            "floor__chance_random_direction": RD}

        # ---- the kNN arms ---------------------------------------------------------------------
        arms_prof = {}
        arm_meta = {}
        for dn, D in DISTS.items():
            for k in KS:
                h, curve, loo = choose_h_loo(D, G, tr, k)
                P, NB, DD, W = knn_predict(D, G, held, tr, k, h)
                an = f"knn__{dn}__k{k}"
                arms_prof[an] = P
                arm_meta[an] = {"distance": dn, "k": int(k), "h_loo": h,
                                "loo_spearman_within_train": float(loo),
                                "h_curve": curve[:] if curve else None}
                npz[f"c7_{cell}_{an}_neighbour_idx"] = NB
                npz[f"c7_{cell}_{an}_neighbour_dist"] = DD
                npz[f"c7_{cell}_{an}_neighbour_weight"] = W
                log(f"  [{cell}/{an}] LOO-within-train spearman {loo:+.4f} | h={h}")

        # ---- score everything, profile level and emission level -------------------------------
        def score_profiles(P):
            sp = np.array([sp_score(P[r], G[i]) for r, i in enumerate(held)])
            nd = np.array([ndcg_at(P[r], G[i], 5) for r, i in enumerate(held)])
            bl = [blend_from_profile(P[r]) for r in range(len(held))]
            ok = np.array([b is not None for b in bl])
            assert ok.all(), "a held-out state had no constructible blend"
            BL = np.stack(bl)
            cr = cosrows(BL, Tt[held])
            cm = cosrows(BL - mu[None, :], Tmr[held])
            return {"sp": sp, "nd": nd, "cr": cr, "cm": cm}

        res = {}
        for an, P in list(arms_prof.items()) + list(prof_floors.items()):
            res[an] = score_profiles(P)
        for an, V in emis_only.items():
            # a CONSTANT TARGET has no profile; only the emission-level rows are defined. V lives in
            # MEAN-REMOVED space, so the RAW row puts mu back before scoring against the raw target.
            Vr = V + mu[None, :]
            res[an] = {"sp": np.full(len(held), np.nan), "nd": np.full(len(held), np.nan),
                       "cr": cosrows(Vr, Tt[held]), "cm": cosrows(V, Tmr[held])}
        # the ceiling and the incumbent
        ORC = np.stack([blend_from_profile(G[i]) for i in held])
        res["ceiling__oracle_TRUE_profile_on_shared_set"] = {
            "sp": np.ones(len(held)), "nd": np.ones(len(held)),
            "cr": cosrows(ORC, Tt[held]), "cm": cosrows(ORC - mu[None, :], Tmr[held])}
        res["incumbent__trained_policy_C1_final"] = {
            "sp": np.full(len(held), np.nan), "nd": np.full(len(held), np.nan),
            "cr": pol_cos[cell][held], "cm": np.full(len(held), np.nan)}

        # ---- GATES G1/G2/G4: the rebuilt Epool/mu must reproduce C5's banked baselines ---------
        gates = {}
        for an, key in (("floor__global_mean_train_profile", "baseline__global_mean_train_profile"),
                        ("floor__popularity_prior", "baseline__popularity_prior")):
            b_cm = np.asarray(Z5[f"d5_{cell}_{key}_blend_cos_mr"], np.float64)
            b_sp = np.asarray(Z5[f"d5_{cell}_{key}_spearman"], np.float64)
            e_cm = float(np.abs(res[an]["cm"] - b_cm).max())
            e_sp = float(np.abs(res[an]["sp"] - b_sp).max())
            gates[f"{key}_blend_cos_mr_max_abs_err_vs_C5"] = e_cm
            gates[f"{key}_spearman_max_abs_err_vs_C5"] = e_sp
            assert e_cm < 1e-9 and e_sp < 1e-9, (
                f"GATE FAILED for {cell}/{key}: the rebuilt Epool/mu do not reproduce C5's banked "
                f"baseline (blend cos err {e_cm:g}, spearman err {e_sp:g}). STOP -- this run would "
                f"be describing a different object.")
        log(f"  [gates G1/G2/G4] rebuilt Epool/mu reproduce C5's banked baselines to "
            f"{max(gates.values()):g}")

        # ---- paired contrasts against EVERY floor ---------------------------------------------
        FLOORS = (["floor__global_mean_train_profile", "floor__popularity_prior",
                   "floor__chance_random_permuted_profile"] + list(emis_only))
        rec = {"n_train": int(len(tr)), "n_held_out": int(len(held)),
               "gates_vs_c5_banked": gates, "arms": {}}
        for an, v in res.items():
            row = {"held_out_profile_spearman": dist(v["sp"]),
                   "held_out_profile_spearman_boot": boot_ci(v["sp"], N_BOOT, SEED)
                   if np.isfinite(v["sp"]).any() else None,
                   "held_out_profile_ndcg5": dist(v["nd"]),
                   "held_out_profile_ndcg5_boot": boot_ci(v["nd"], N_BOOT, SEED)
                   if np.isfinite(v["nd"]).any() else None,
                   "emission_cos_to_true_target_RAW": dist(v["cr"]),
                   "emission_cos_to_true_target_RAW_boot": boot_ci(v["cr"], N_BOOT, SEED),
                   "emission_cos_to_true_target_MEAN_REMOVED": dist(v["cm"])
                   if np.isfinite(v["cm"]).any() else None,
                   "emission_cos_MEAN_REMOVED_boot": boot_ci(v["cm"], N_BOOT, SEED)
                   if np.isfinite(v["cm"]).any() else None,
                   "vs_floors_PAIRED": {}}
            row.update(arm_meta.get(an, {}))
            for fn in FLOORS:
                if fn == an:
                    continue
                f = res[fn]
                e = {}
                if np.isfinite(v["sp"]).any() and np.isfinite(f["sp"]).any():
                    e["profile_spearman"] = paired_bootstrap(v["sp"] - f["sp"], N_BOOT, SEED)
                    e["profile_ndcg5"] = paired_bootstrap(v["nd"] - f["nd"], N_BOOT, SEED)
                e["emission_cos_RAW"] = paired_bootstrap(v["cr"] - f["cr"], N_BOOT, SEED)
                if np.isfinite(v["cm"]).any() and np.isfinite(f["cm"]).any():
                    e["emission_cos_MEAN_REMOVED"] = paired_bootstrap(v["cm"] - f["cm"],
                                                                      N_BOOT, SEED)
                row["vs_floors_PAIRED"][fn] = e
            row["vs_trained_policy_PAIRED_emission_cos_RAW"] = paired_bootstrap(
                v["cr"] - pol_cos[cell][held], N_BOOT, SEED)
            rec["arms"][an] = row
            npz[f"c7_{cell}_{an}_spearman"] = v["sp"]
            npz[f"c7_{cell}_{an}_ndcg5"] = v["nd"]
            npz[f"c7_{cell}_{an}_emission_cos_raw"] = v["cr"]
            npz[f"c7_{cell}_{an}_emission_cos_mr"] = v["cm"]
            if an.startswith(("knn__", "floor__", "ceiling__")):
                log(f"  [{cell}/{an}] spearman {np.nanmean(v['sp']):+.4f} "
                    f"| ndcg@5 {np.nanmean(v['nd']):.4f} | emission cos raw "
                    f"{np.nanmean(v['cr']):+.4f} mr {np.nanmean(v['cm']):+.4f}")

        # ---- the pre-registered selection: best LOO-within-train, then read it out -------------
        best_an = max(arm_meta, key=lambda a: arm_meta[a]["loo_spearman_within_train"])
        gmn = res["floor__global_mean_train_profile"]
        rec["selected_arm"] = {
            "arm": best_an, "selected_by": "highest LOO Spearman WITHIN the training states",
            "loo_spearman_within_train": arm_meta[best_an]["loo_spearman_within_train"],
            "held_out_profile_spearman": float(res[best_an]["sp"].mean()),
            "state_blind_floor_profile_spearman": float(gmn["sp"].mean()),
            "PAIRED_profile_spearman_vs_state_blind_floor":
                paired_bootstrap(res[best_an]["sp"] - gmn["sp"], N_BOOT, SEED),
            "PAIRED_profile_ndcg5_vs_state_blind_floor":
                paired_bootstrap(res[best_an]["nd"] - gmn["nd"], N_BOOT, SEED),
            "held_out_emission_cos_MEAN_REMOVED": float(res[best_an]["cm"].mean()),
            "state_blind_floor_emission_cos_MEAN_REMOVED": float(gmn["cm"].mean()),
            "PAIRED_emission_cos_MEAN_REMOVED_vs_state_blind_floor":
                paired_bootstrap(res[best_an]["cm"] - gmn["cm"], N_BOOT, SEED),
            "held_out_emission_cos_RAW": float(res[best_an]["cr"].mean()),
            "state_blind_constant_target_D3_cos_RAW":
                float(res["floor__state_blind_constant_target_D3_LOO_mean_of_other_71"]["cr"].mean()),
            "PAIRED_emission_cos_RAW_vs_state_blind_constant_target_D3":
                paired_bootstrap(
                    res[best_an]["cr"]
                    - res["floor__state_blind_constant_target_D3_LOO_mean_of_other_71"]["cr"],
                    N_BOOT, SEED),
            "trained_policy_emission_cos_RAW": float(pol_cos[cell][held].mean()),
            "PAIRED_emission_cos_RAW_vs_trained_policy":
                paired_bootstrap(res[best_an]["cr"] - pol_cos[cell][held], N_BOOT, SEED)}
        v = rec["selected_arm"]["PAIRED_profile_spearman_vs_state_blind_floor"]
        rec["selected_arm"]["VERDICT_profile_beats_state_blind_floor"] = bool(v["lo"] > 0)
        v2 = rec["selected_arm"]["PAIRED_emission_cos_MEAN_REMOVED_vs_state_blind_floor"]
        rec["selected_arm"]["VERDICT_emission_beats_state_blind_floor"] = bool(v2["lo"] > 0)

        # ---- the honest scaling question: does accuracy need a near neighbour? -----------------
        deg = {}
        for dn, D in DISTS.items():
            dnn = np.array([D[i, tr].min() for i in held])
            npz[f"c7_{cell}_{dn}_nearest_train_distance"] = dnn
            per_arm = {}
            for an in [a for a in arm_meta if arm_meta[a]["distance"] == dn]:
                per_arm[an] = {
                    "nearest_distance__vs__profile_spearman_pearson":
                        corr_boot(dnn, res[an]["sp"], "pearson"),
                    "nearest_distance__vs__profile_spearman_spearman":
                        corr_boot(dnn, res[an]["sp"], "spearman"),
                    "nearest_distance__vs__emission_cos_mr_pearson":
                        corr_boot(dnn, res[an]["cm"], "pearson"),
                    "near_half_vs_far_half_profile_spearman": {
                        "near": float(res[an]["sp"][dnn <= np.median(dnn)].mean()),
                        "far": float(res[an]["sp"][dnn > np.median(dnn)].mean()),
                        "floor_near": float(gmn["sp"][dnn <= np.median(dnn)].mean()),
                        "floor_far": float(gmn["sp"][dnn > np.median(dnn)].mean())},
                    "near_half_vs_far_half_emission_cos_mr": {
                        "near": float(res[an]["cm"][dnn <= np.median(dnn)].mean()),
                        "far": float(res[an]["cm"][dnn > np.median(dnn)].mean()),
                        "floor_near": float(gmn["cm"][dnn <= np.median(dnn)].mean()),
                        "floor_far": float(gmn["cm"][dnn > np.median(dnn)].mean())}}
            deg[dn] = {"nearest_train_distance_at_held_out_states": dist(dnn),
                       "nearest_train_distance_within_train_states":
                           dist(np.array([D[i, tr[tr != i]].min() for i in tr])),
                       "per_arm": per_arm}
        rec["distance_degradation"] = deg
        rec["per_state_table"] = [
            {"sid": sids[int(i)], "depth": int(depth[int(i)]),
             "nearest_train_dist_latent": float(Dlat[int(i), tr].min()),
             "nearest_train_dist_tree": float(Dtree[int(i), tr].min()),
             "nearest_train_sid_latent": sids[int(tr[int(np.argmin(Dlat[int(i), tr]))])],
             "selected_arm_profile_spearman": float(res[best_an]["sp"][r]),
             "selected_arm_emission_cos_mr": float(res[best_an]["cm"][r]),
             "floor_profile_spearman": float(gmn["sp"][r]),
             "floor_emission_cos_mr": float(gmn["cm"][r]),
             "trained_policy_cos_raw": float(pol_cos[cell][int(i)])}
            for r, i in enumerate(held)]
        out["cells"][cell] = rec
        flush()

    # ======================================================================== the headline verdict
    ver = {}
    for cell in SPLITS:
        sa = out["cells"][cell]["selected_arm"]
        ver[cell] = {
            "selected_arm": sa["arm"],
            "profile_spearman": sa["held_out_profile_spearman"],
            "state_blind_floor": sa["state_blind_floor_profile_spearman"],
            "PAIRED": sa["PAIRED_profile_spearman_vs_state_blind_floor"],
            "kNN_BEATS_THE_STATE_BLIND_FLOOR_profile": sa[
                "VERDICT_profile_beats_state_blind_floor"],
            "emission_cos_mr": sa["held_out_emission_cos_MEAN_REMOVED"],
            "emission_state_blind_floor": sa["state_blind_floor_emission_cos_MEAN_REMOVED"],
            "PAIRED_emission": sa["PAIRED_emission_cos_MEAN_REMOVED_vs_state_blind_floor"],
            "kNN_BEATS_THE_STATE_BLIND_FLOOR_emission": sa[
                "VERDICT_emission_beats_state_blind_floor"]}
    out["verdict"] = ver
    flush()
    log(f"[done] wrote {p_json} in {out['minutes']:.1f} min")
    return out


def main():
    ap = argparse.ArgumentParser(description="Paper C C7 kNN profile transfer (banked artifacts)")
    ap.add_argument("--m", type=int, default=5)
    ap.add_argument("--snap_users", type=int, default=10000)
    ap.add_argument("--out_tag", default="")
    ap.add_argument("--snapshot", default=None)
    ap.add_argument("--concept_ckpt", default=None)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--allow_dirty", action="store_true")
    args = ap.parse_args()
    from strategy_channel_suite import CERT_TOWER, CERT_CONC
    args.snapshot = args.snapshot or CERT_TOWER
    args.concept_ckpt = args.concept_ckpt or CERT_CONC
    log("#" * 100)
    log("# PAPER C / C7 -- LOCAL kNN profile transfer at held-out states. Nothing is trained; "
        "TRAIN-side banked artifacts only; the 10,000 canonical TEST users are not touched.")
    log("#" * 100)
    return run(args)


if __name__ == "__main__":
    main()
