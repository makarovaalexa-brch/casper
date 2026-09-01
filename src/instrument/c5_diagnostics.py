r"""PAPER C / C5 -- DIAGNOSTICS on the C1-final generalisation failure: is it the MODEL or the INPUTS?

Read-only over BANKED artifacts. Nothing is trained (a ridge regression on 72 rows is a closed-form
solve, not a fit of the policy), no LM is called, no test user is built, loaded or touched.

WHAT THIS ANSWERS
-----------------
`docs/results/C1_FINAL.md` records a policy that fits the 57 trained node states at cos 0.970 and
lands at cos -0.119 on the 15 held-out `root.like.*` states. Six cheap diagnostics decide whether
that is a MODEL failure (the policy/optimisation could not learn a mapping that exists) or an INPUT
failure (the state does not determine the target, so no architecture can extrapolate):

  D1  how different are the 72 blend targets from each other?
  D2  is the measured gain PROFILE predictable from the state?  (the load-bearing test)
  D3  the state-blind leave-one-out baseline C1-final should have had
  D4  ridge probe, policy input -> the 129-d blend target
  D5  ridge probe, policy input -> the measured gain profile, then a blend built from the prediction
  D6  input sufficiency: are the 72 states even distinguishable, and do size/depth explain the profile?

THE MEAN-REMOVAL CONVENTION, STATED ONCE AND USED EVERYWHERE
------------------------------------------------------------
`docs/results/C0_GEOMETRY.md` measured a bank-wide mean pairwise cosine of 0.96 RAW against 0.0018
after removing the bank mean: raw cosines in this space are dominated by a common-mode offset and are
close to meaningless as similarity. So:

  * anything living in the 129-d homogeneous emission space `[k(phi), b(phi)]` -- targets, candidate
    embeddings, bank entries, emitted vectors, ridge predictions -- is mean-removed by `mu`, the mean
    of the FULL 19,390-entry bank, exactly as C1/C4 do. Called MEAN-REMOVED (bank mu) below.
  * the 200-d latent state vectors have no bank; their mean-removal is the mean over the 72 states.
    Called MEAN-REMOVED (across states).
  * RAW cosines are reported ALONGSIDE wherever C1-final's own registered numbers are raw (its
    `cos_own` is a raw cosine), because the comparison would otherwise be between two conventions.

WHAT "THE POLICY INPUT STATE VECTOR" IS HERE
--------------------------------------------
The policy sees, at a (user, position) cell: the folded latent `z` (200-d), the answered history and
the position. Within one node state every user shares the history and the position, so the only thing
that varies is `z`, and the emitted per-state vector C1-final scores is itself a cohort mean. The
state-level input summaries used here are therefore:

  X_z     the cohort-mean latent `z` at the state (200-d), from the banked trajectory bank.
  X_w     the WARM START's own emitted vector at the state (129-d, banked `warm_zbar`) -- the
          policy's own representation of the state, before any C1 fine-tuning. Not a raw input, and
          labelled as such; it is the representation the fine-tune actually starts from.
  X_s     structural features: depth, last-branch one-hot, log fit-cohort size, log bank occupancy.
  X_all   the concatenation, standardised.

If NONE of these predicts the target, the inputs are insufficient and no architecture change fixes it.

FIREWALL. Train-side states only, on the 85,289-user cold bank. `test_touched` is False everywhere.
"""
from __future__ import annotations

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

import argparse
import glob
import json
import time

import numpy as np
import torch

from policy_net import Policy, build_phi, log, paired_bootstrap
from c0_geometry import (PAPERC, N_BANK, sha256, git_block, savez_atomic, index_map_assert,
                         bank_vectors, dist)
from c1_pilot import build_targets
import c1_final as CF

SEED = 98765
N_BOOT = 10000
ALPHAS = np.logspace(-2, 7, 28)


# ============================================================================ small helpers
def cosrows(A, B):
    A = np.asarray(A, np.float64); B = np.asarray(B, np.float64)
    return ((A * B).sum(1) / (np.linalg.norm(A, axis=1) * np.linalg.norm(B, axis=1) + 1e-300))


def cosmat(A):
    """Pairwise cosine matrix of the rows of A (A already mean-removed by the caller)."""
    A = np.asarray(A, np.float64)
    An = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-300)
    return An @ An.T


def offdiag(M):
    n = M.shape[0]
    iu = np.triu_indices(n, 1)
    return np.asarray(M)[iu]


try:
    from scipy.stats import rankdata as _rankdata

    def ranks(x):
        """Average ranks, the Spearman convention (scipy's C implementation)."""
        return np.asarray(_rankdata(np.asarray(x, np.float64)), np.float64)
except Exception:                                                          # pragma: no cover
    def ranks(x):
        x = np.asarray(x, np.float64)
        o = np.argsort(x, kind="mergesort")
        r = np.empty(len(x), np.float64)
        r[o] = np.arange(len(x), dtype=np.float64)
        sx = x[o]
        i = 0
        while i < len(sx):
            j = i
            while j + 1 < len(sx) and sx[j + 1] == sx[i]:
                j += 1
            if j > i:
                r[o[i:j + 1]] = (i + j) / 2.0
            i = j + 1
        return r


def pearson_mat(A):
    """Pairwise Pearson correlation of the rows of A."""
    A = np.asarray(A, np.float64)
    Ac = A - A.mean(1, keepdims=True)
    Ac /= (np.linalg.norm(Ac, axis=1, keepdims=True) + 1e-300)
    return Ac @ Ac.T


def boot_ci(x, n_boot=N_BOOT, seed=SEED):
    x = np.asarray(x, np.float64)
    x = x[np.isfinite(x)]
    rng = np.random.default_rng(seed)
    bs = x[rng.integers(0, len(x), size=(n_boot, len(x)))].mean(1)
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return {"mean": float(x.mean()), "median": float(np.median(x)), "lo": float(lo),
            "hi": float(hi), "n": int(len(x)), "excludes_zero": bool(lo > 0 or hi < 0),
            "ci": "percentile bootstrap 10k, seed 98765"}


def cluster_boot_corr(X, Y, n_boot=N_BOOT, seed=SEED, kind="pearson"):
    """Correlation between two PAIR-level quantities, with a CLUSTER bootstrap over STATES.

    The 2,556 pairs are not independent (each state appears in 71 of them), so a bootstrap over pairs
    would understate the interval by a lot. Resampling the 72 STATES with replacement and recomputing
    the correlation over the induced off-diagonal pairs is the honest version.
    """
    X = np.asarray(X, np.float64); Y = np.asarray(Y, np.float64)
    n = X.shape[0]
    iu = np.triu_indices(n, 1)

    def c(a, b):
        a = a - a.mean(); b = b - b.mean()
        d = np.linalg.norm(a) * np.linalg.norm(b)
        return float(a @ b / d) if d > 0 else np.nan

    if kind == "spearman":
        XX, YY = X.copy(), Y.copy()
        v = c(ranks(X[iu]), ranks(Y[iu]))
    else:
        XX, YY = X, Y
        v = c(X[iu], Y[iu])
    rng = np.random.default_rng(seed)
    bs = np.empty(n_boot)
    for t in range(n_boot):
        s = rng.integers(0, n, n)
        a = XX[np.ix_(s, s)]; b = YY[np.ix_(s, s)]
        keep = s[:, None] != s[None, :]
        m = np.triu(np.ones((n, n), bool), 1) & keep
        aa, bb = a[m], b[m]
        if len(aa) < 3:
            bs[t] = np.nan
            continue
        bs[t] = c(ranks(aa), ranks(bb)) if kind == "spearman" else c(aa, bb)
    bs = bs[np.isfinite(bs)]
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return {"r": v, "lo": float(lo), "hi": float(hi), "kind": kind, "n_pairs": int(len(iu[0])),
            "ci": "CLUSTER bootstrap over the 72 states, 10k, seed 98765",
            "excludes_zero": bool(lo > 0 or hi < 0)}


# ============================================================================ ridge
def ridge_fit(X, Y, alpha):
    xm = X.mean(0); ym = Y.mean(0)
    Xc = X - xm; Yc = Y - ym
    d = X.shape[1]
    A = Xc.T @ Xc + alpha * np.eye(d)
    W = np.linalg.solve(A, Xc.T @ Yc)
    return xm, ym, W


def ridge_pred(M, X):
    xm, ym, W = M
    return (X - xm) @ W + ym


def standardise(Xtr, Xall):
    m = Xtr.mean(0); s = Xtr.std(0)
    s = np.where(s < 1e-9, 1.0, s)
    return (Xall - m) / s


def sweep_alpha_loo(X, Y, score):
    """Leave-one-out over the TRAIN states ONLY; returns (best_alpha, curve).

    One thin SVD per fold serves every alpha (the ridge solution is
    `V diag(s/(s^2+a)) U^T Yc`), so the sweep costs 57 small SVDs rather than 28*57 solves. No
    held-out state is touched: the caller passes the training rows only.
    """
    n = len(X)
    sc = np.zeros((len(ALPHAS), n))
    for i in range(n):
        k = np.ones(n, bool); k[i] = False
        Xk, Yk = X[k], Y[k]
        xm, ym = Xk.mean(0), Yk.mean(0)
        Xc, Yc = Xk - xm, Yk - ym
        U, s, Vt = np.linalg.svd(Xc, full_matrices=False)
        UtY = U.T @ Yc                                    # r x p
        v0 = (X[i] - xm) @ Vt.T                           # r
        for ai, a in enumerate(ALPHAS):
            pred = ((v0 * s / (s * s + a)) @ UtY) + ym
            sc[ai, i] = score(pred, Y[i])
    m = np.nanmean(sc, 1)
    curve = [(float(a), float(v)) for a, v in zip(ALPHAS, m)]
    return float(ALPHAS[int(np.nanargmax(m))]), curve


# ============================================================================ the bank, streamed
def bank_light():
    """Only the SMALL fields of the 85,289-user cold bank, in c1_final's own slice order."""
    parts = sorted(glob.glob(os.path.join(CF.BANK_DIR, "cold_*.npz")),
                   key=lambda p: int(os.path.basename(p)[5:].split("_")[0]))
    assert parts, f"no cold slices under {CF.BANK_DIR}"
    lab, out, val, uid, n = [], [], [], [], 0
    for p in parts:
        z = np.load(p)
        if not len(z["rows"]):
            continue
        lo = int(os.path.basename(p)[5:].split("_")[0])
        assert int(z["anchor"]) == -1 and int(z["n_fallback"]) == 0
        lab.append(z["label_idx"]); out.append(z["outcome"]); val.append(z["valid"])
        uid.append(lo + z["rows"].astype(np.int64))
        n += len(z["rows"])
    lab = np.concatenate(lab); out = np.concatenate(out)
    val = np.concatenate(val); uid = np.concatenate(uid)
    assert len(np.unique(uid)) == len(uid)
    assert lab.shape[1] == CF.TOTAL_L
    log(f"[bank] {n:,} cold users x {CF.TOTAL_L} positions (small fields only; z is STREAMED)")
    return {"lab": lab, "out": out, "valid": val, "uid": uid, "n": n,
            "parts": parts, "n_slices": len(parts)}


def state_mean_z(bank, blend, n_states):
    """Cohort-mean latent `z` per blend state, accumulated one slice at a time (never all resident)."""
    d = None
    S = None
    cnt = np.zeros(n_states, np.int64)
    at = 0
    for p in bank["parts"]:
        z = np.load(p)
        k = len(z["rows"])
        if not k:
            continue
        Z = z["z"]
        if S is None:
            d = int(Z.shape[2]); S = np.zeros((n_states, d), np.float64)
        b = blend[at:at + k]
        r, c = np.nonzero(b >= 0)
        if len(r):
            idx = b[r, c]
            np.add.at(S, idx, Z[r, c].astype(np.float64))
            cnt += np.bincount(idx, minlength=n_states)
        at += k
        del Z, z
    assert at == bank["n"]
    assert cnt.min() > 0
    return (S / cnt[:, None]).astype(np.float64), cnt


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

    # ---- warm start + bank space (the ONE environment build) ---------------------------------
    blob = torch.load(CF.WARM, map_location="cpu")
    assert blob["arch"] == "equivariant" and blob["state"] == "both"
    assert len(blob["ext"]) == CF.N_EXT
    warm_sha = sha256(CF.WARM)
    E = build_greedy_env(args.snapshot, "train", args.concept_ckpt, "paperC/c5",
                         cohort_slice=(0, args.snap_users))
    contract_selftest(E, E["rows"])
    f0, t0_ = genv_mean(E, [], E["rows"])
    assert f0 == CF.TRAIN_M0_FULL, f"CANONICAL SNAP FAILED: {f0!r} vs {CF.TRAIN_M0_FULL!r}"
    log(f"[gate] canonical TRAIN m=0 snap full={f0!r} tail={t0_!r}")

    pool, _ip, _cp = build_pool(E, 500, 500)
    pool = [(int(a), int(b)) for a, b in pool]
    assert pool == [(int(a), int(b)) for a, b in blob["pool"]]
    n_pool = len(pool)
    pool_index = {q: i for i, q in enumerate(pool)}
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
    Ebank, Kbank = bank_vectors(pol0, phi_full)
    Epool, Kpool = bank_vectors(pol0, phi_ext[:n_pool])
    map_err = index_map_assert(Ebank, pool2full, Epool)
    mu = Ebank.astype(np.float64).mean(0)
    log(f"[bank] {Ebank.shape} descriptors; ||mu|| {np.linalg.norm(mu):.4f}; map err {map_err:g}")

    # the popularity prior the pool itself was ranked by (items: train count; concepts: answer rate)
    cz = np.load(CONCSTATS)
    ans_rate = (cz["n"] / N_TRAIN_USERS).astype(np.float64)
    icnt = np.asarray(E["ctx"].cnt, np.float64)
    pop_prior = np.asarray([(icnt[i] / max(icnt.max(), 1e-9)) if ch == 0 else
                            (ans_rate[i] / max(ans_rate.max(), 1e-9)) for ch, i in pool], np.float64)

    # ---- the tree, the 72 exact gain tables, the targets --------------------------------------
    tree = json.load(open(CF.TREE_JSON))
    assert tree["split"] == "train" and int(tree["total_L"]) == CF.TOTAL_L
    assert tree["test_touched"] is False
    nodes = tree["nodes"]
    states, gain_chk = CF.load_sweeps(nodes, pool_index)
    Tt, TKt, meta = build_targets(states, Epool, Kpool, pool_index, args.m)
    banked = np.load(os.path.join(PAPERC, "c1_final_per_state.npz"), allow_pickle=True)
    dT = float(np.abs(Tt - banked["target_m5"]).max())
    assert dT < 1e-4, (f"the reconstructed m=5 targets differ from C1-final's banked ones by {dT:g} "
                       f"-- STOP, this diagnostic would be describing a different object")
    assert list(banked["state_sid"]) == [s["sid"] for s in states]
    log(f"[contract] targets reconstructed from the SAME code; max|diff| vs banked {dT:g}")

    warm_zbar = np.asarray(banked["warm_zbar"], np.float64)
    zbar_pri = np.asarray(banked["zbar_primary_root_like_subtree"], np.float64)
    zbar_sec = np.asarray(banked["zbar_secondary_depth_ge_5"], np.float64)
    cos_pri = np.asarray(banked["cos_own_primary_root_like_subtree"], np.float64)
    cos_sec = np.asarray(banked["cos_own_secondary_depth_ge_5"], np.float64)

    ns = len(states)
    sids = [s["sid"] for s in states]
    depth = np.asarray([s["depth"] for s in states], np.int64)
    nfit = np.asarray([s["n_fit_users"] for s in states], np.float64)

    # ---- the bank, the descent, the cohort-mean latents ----------------------------------------
    bank = bank_light()
    node_at, term, unfitted, node_ids, nidx = CF.descend(nodes, bank)
    contract = CF.verify_labels(nodes, node_ids, node_at, term, unfitted, bank, pool_index)
    sidx = {s["sid"]: i for i, s in enumerate(states)}
    node_to_state = np.full(len(node_ids), -1, np.int64)
    for nid, i in sidx.items():
        node_to_state[nidx[nid]] = i
    depth_of_node = np.asarray([int(nodes[k]["depth"]) for k in node_ids], np.int64)
    blend = np.full((bank["n"], CF.TOTAL_L), -1, np.int64)
    posgrid = np.arange(CF.TOTAL_L)[None, :]
    on_chain = node_at >= 0
    own_depth = np.zeros_like(node_at, np.int64)
    own_depth[on_chain] = depth_of_node[node_at[on_chain]]
    st_of = np.full_like(blend, -1)
    st_of[on_chain] = node_to_state[node_at[on_chain]]
    is_own = on_chain & (own_depth == posgrid)
    blend[is_own & (st_of >= 0)] = st_of[is_own & (st_of >= 0)]
    Xz, occ = state_mean_z(bank, blend, ns)
    log(f"[states] cohort-mean latent built at all {ns} states; bank occupancy "
        f"min {occ.min():,} median {int(np.median(occ)):,} max {occ.max():,}")

    # ---- splits --------------------------------------------------------------------------------
    held_pri = np.asarray([i for i, s in enumerate(sids)
                           if s == "root.like" or s.startswith("root.like.")], np.int64)
    held_sec = np.flatnonzero(depth >= 5).astype(np.int64)
    assert len(held_pri) == 15 and len(held_sec) == 30
    SPLITS = {"primary_root_like_subtree": held_pri, "secondary_depth_ge_5": held_sec}

    out = {"analysis": "paperC_C5_diagnostics", "citable": True, "test_touched": False,
           "split": "train_states",
           "purpose": "decide whether C1-final's generalisation failure is the MODEL or the INPUTS",
           "reads": ["docs/results/C1_FINAL.md", "docs/design/DS_C1_FINAL.md",
                     "docs/design/C1_FINAL_SCOPING.md", "docs/results/C0_GEOMETRY.md"],
           "git": GB,
           "mean_removal_convention": {
               "emission_space_129d": "mean-removed by mu = the mean of the FULL 19,390-entry bank "
                                      "(C0/C1 convention). RAW cosines reported alongside where "
                                      "C1-final's own registered number is raw.",
               "latent_space_200d": "mean-removed by the mean over the 72 states (no bank exists).",
               "mu_norm": float(np.linalg.norm(mu))},
           "warm_start": {"path": os.path.relpath(CF.WARM, _ROOT), "sha256": warm_sha},
           "gates": {"canonical_snap_train_m0_full": f0, "canonical_snap_train_m0_tail": t0_,
                     "pool2full_index_map_max_abs_err": map_err,
                     "sweep_argmax_equals_splitter_max_abs_gain_err": gain_chk,
                     "targets_match_c1_final_banked_max_abs_diff": dT,
                     "descent_reproduces_banked_labels": contract},
           "n_states": ns,
           "splits": {k: {"n_held_out": int(len(v)), "held_out_sids": [sids[int(i)] for i in v]}
                      for k, v in SPLITS.items()}}
    p_json = os.path.join(PAPERC, f"c5_diagnostics{args.out_tag}.json")
    p_npz = os.path.join(PAPERC, f"c5_diagnostics{args.out_tag}_pairs.npz")
    npz = {}

    def flush():
        out["minutes"] = round((time.time() - t00) / 60, 2)
        with open(p_json + ".tmp", "w") as fh:
            json.dump(out, fh, indent=1, default=float)
        os.replace(p_json + ".tmp", p_json)
        if npz:
            savez_atomic(p_npz, **npz)

    flush()
    rng = np.random.default_rng(SEED)

    # ======================================================================== D1 target diversity
    Tmr = Tt.astype(np.float64) - mu[None, :]
    top1_idx = np.asarray([m["top_m_pool_idx"][0] for m in meta], np.int64)
    C1_ = Epool[top1_idx].astype(np.float64) - mu[None, :]
    rb = rng.choice(N_BANK, ns, replace=False)
    B_ = Ebank[rb].astype(np.float64) - mu[None, :]
    RD = rng.standard_normal((ns, Tt.shape[1]))
    RD *= (np.linalg.norm(Tmr, axis=1) / np.linalg.norm(RD, axis=1))[:, None]

    M_t_mr = cosmat(Tmr)
    M_t_raw = cosmat(Tt.astype(np.float64))
    d1 = {"unit": "the 2,556 unordered pairs of the 72 blend targets",
          "targets_MEAN_REMOVED_bank_mu": dist(offdiag(M_t_mr)),
          "targets_RAW": dist(offdiag(M_t_raw)),
          "top1_candidate_embeddings_MEAN_REMOVED": dist(offdiag(cosmat(C1_))),
          "random_bank_entries_MEAN_REMOVED": dist(offdiag(cosmat(B_))),
          "random_directions": dist(offdiag(cosmat(RD))),
          "warm_start_emissions_MEAN_REMOVED": dist(offdiag(cosmat(warm_zbar - mu[None, :]))),
          "note": "if the targets are far more mutually similar than the candidate embeddings they "
                  "are built from, then FITTING them is easy and IDENTIFYING them is hard"}
    out["D1_target_diversity"] = d1
    npz["pair_target_cos_mr"] = M_t_mr.astype(np.float32)
    npz["pair_target_cos_raw"] = M_t_raw.astype(np.float32)
    log(f"[D1] target pairwise cos: mean-removed median {d1['targets_MEAN_REMOVED_bank_mu']['median']:.4f} "
        f"(IQR {d1['targets_MEAN_REMOVED_bank_mu']['q1']:.4f}..{d1['targets_MEAN_REMOVED_bank_mu']['q3']:.4f}) "
        f"| RAW median {d1['targets_RAW']['median']:.4f} | top-1 cand "
        f"{d1['top1_candidate_embeddings_MEAN_REMOVED']['median']:.4f} | random bank "
        f"{d1['random_bank_entries_MEAN_REMOVED']['median']:.4f}")
    flush()

    # ======================================================================== the gain profiles
    # the SHARED candidate set: pool entries measured at EVERY state (i.e. in no state's own prefix)
    allc = [set(int(q) for q in s["gains"]) for s in states]
    shared = sorted(set.intersection(*allc))
    S = np.asarray(shared, np.int64)
    G = np.asarray([[s["gains"][int(q)] for q in S] for s in states], np.float64)
    in_shared = np.zeros(n_pool, bool); in_shared[S] = True
    top5_cov = float(np.mean([in_shared[np.asarray(m["top_m_pool_idx"])].mean() for m in meta]))
    log(f"[profiles] shared candidate set: {len(S)} of {n_pool} pool entries; "
        f"{top5_cov:.4f} of the states' true top-5 members lie inside it")

    RG = np.stack([ranks(G[i]) for i in range(ns)])
    Y_sp = pearson_mat(RG)                                   # Spearman over the shared set
    P = np.maximum(G, 0.0)
    Y_pos_raw = cosmat(P)
    Y_pos_mr = cosmat(P - P.mean(0, keepdims=True))          # candidate-mean removed
    try:
        from scipy.stats import kendalltau
        Y_kt = np.eye(ns)
        for i in range(ns):
            for j in range(i + 1, ns):
                Y_kt[i, j] = Y_kt[j, i] = float(kendalltau(G[i], G[j])[0])
        have_kt = True
    except Exception as e:                                   # pragma: no cover
        log(f"[D2] kendall tau unavailable ({e}); reported as null")
        Y_kt = np.full((ns, ns), np.nan); have_kt = False

    # ======================================================================== D2 predictability
    Xz_mr = Xz - Xz.mean(0, keepdims=True)
    X_zcos = cosmat(Xz_mr)
    X_wcos = cosmat(warm_zbar - mu[None, :])
    X_zraw = cosmat(Xz)

    def lca_depth(a, b):
        pa, pb = a.split("."), b.split(".")
        k = 0
        while k < min(len(pa), len(pb)) and pa[k] == pb[k]:
            k += 1
        return k - 1                                          # depth of the LCA node
    Dpath = np.zeros((ns, ns))
    Sib = np.zeros((ns, ns))
    for i in range(ns):
        for j in range(ns):
            l = lca_depth(sids[i], sids[j])
            Dpath[i, j] = depth[i] + depth[j] - 2 * l
            Sib[i, j] = float(sids[i].rsplit(".", 1)[0] == sids[j].rsplit(".", 1)[0]
                              and sids[i] != sids[j])

    d2 = {"unit": "the 2,556 unordered state pairs; CIs are a CLUSTER bootstrap over the 72 states",
          "n_shared_candidates": int(len(S)),
          "frac_true_top5_inside_shared_set": top5_cov,
          "profile_similarity_distributions": {
              "spearman": dist(offdiag(Y_sp)),
              "kendall_tau": dist(offdiag(Y_kt)) if have_kt else None,
              "positive_part_cosine_RAW": dist(offdiag(Y_pos_raw)),
              "positive_part_cosine_CANDIDATE_MEAN_REMOVED": dist(offdiag(Y_pos_mr))},
          "input_similarity_distributions": {
              "cohort_mean_latent_MEAN_REMOVED_across_states": dist(offdiag(X_zcos)),
              "cohort_mean_latent_RAW": dist(offdiag(X_zraw)),
              "warm_start_emission_MEAN_REMOVED_bank_mu": dist(offdiag(X_wcos)),
              "tree_path_distance": dist(offdiag(Dpath)),
              "frac_pairs_sharing_a_parent": float(offdiag(Sib).mean())},
          "correlation_input_similarity_vs_profile_similarity": {}}
    for xn, X in (("cohort_mean_latent_mr", X_zcos), ("warm_start_emission_mr", X_wcos),
                  ("neg_tree_path_distance", -Dpath), ("shares_a_parent", Sib)):
        for yn, Y in (("spearman", Y_sp), ("positive_part_cos_mr", Y_pos_mr),
                      ("kendall_tau", Y_kt if have_kt else None)):
            if Y is None:
                continue
            d2["correlation_input_similarity_vs_profile_similarity"][f"{xn}__vs__{yn}"] = {
                "pearson": cluster_boot_corr(X, Y, kind="pearson"),
                "spearman": cluster_boot_corr(X, Y, kind="spearman")}
    # k-NN version
    knn = {}
    for xn, X in (("cohort_mean_latent_mr", X_zcos), ("warm_start_emission_mr", X_wcos)):
        Xo = X.copy(); np.fill_diagonal(Xo, -np.inf)
        nn = np.argmax(Xo, 1)
        for yn, Y in (("spearman", Y_sp), ("positive_part_cos_mr", Y_pos_mr)):
            Yo = Y.copy(); np.fill_diagonal(Yo, np.nan)
            y_nn = Y[np.arange(ns), nn]
            y_mean_other = np.nanmean(Yo, 1)
            r2 = np.random.default_rng(SEED + 7)
            rand_other = np.array([r2.choice([k for k in range(ns) if k != i]) for i in range(ns)])
            y_rand = Y[np.arange(ns), rand_other]
            knn[f"{xn}__{yn}"] = {
                "profile_sim_to_nearest_state": dist(y_nn),
                "profile_sim_to_mean_other_state": dist(y_mean_other),
                "profile_sim_to_one_seeded_random_other": dist(y_rand),
                "nearest_minus_mean_other": paired_bootstrap(y_nn - y_mean_other, N_BOOT, SEED),
                "nearest_minus_one_random_other": paired_bootstrap(y_nn - y_rand, N_BOOT, SEED),
                "n_states_nearest_beats_mean_other": int((y_nn > y_mean_other).sum())}
    d2["knn_nearest_state_test"] = knn
    out["D2_profile_predictable_from_state"] = d2
    npz.update({"pair_profile_spearman": Y_sp.astype(np.float32),
                "pair_profile_poscos_mr": Y_pos_mr.astype(np.float32),
                "pair_profile_kendall": Y_kt.astype(np.float32),
                "pair_input_latent_cos_mr": X_zcos.astype(np.float32),
                "pair_input_warm_cos_mr": X_wcos.astype(np.float32),
                "pair_tree_path_distance": Dpath.astype(np.float32),
                "pair_shares_parent": Sib.astype(np.float32),
                "shared_candidates": S, "gain_profiles_shared": G.astype(np.float32),
                "state_mean_latent": Xz.astype(np.float32),
                "state_depth": depth, "state_n_fit_users": nfit, "state_bank_occ": occ,
                "state_sid": np.asarray(sids)})
    for k, v in knn.items():
        log(f"[D2 kNN] {k}: nearest {v['profile_sim_to_nearest_state']['mean']:+.4f} vs mean-other "
            f"{v['profile_sim_to_mean_other_state']['mean']:+.4f} => "
            f"{v['nearest_minus_mean_other']['mean']:+.4f} "
            f"[{v['nearest_minus_mean_other']['lo']:+.4f}, {v['nearest_minus_mean_other']['hi']:+.4f}]")
    flush()

    # ======================================================================== D3 state-blind LOO
    d3 = {"definition": "for each state, cos(mean of the OTHER 71 targets, this state's own target)"}
    for tag, Tm in (("MEAN_REMOVED_bank_mu", Tmr), ("RAW", Tt.astype(np.float64))):
        loo = np.empty(ns)
        for i in range(ns):
            k = np.ones(ns, bool); k[i] = False
            m_ = Tm[k].mean(0)
            loo[i] = float(m_ @ Tm[i] / (np.linalg.norm(m_) * np.linalg.norm(Tm[i]) + 1e-300))
        d3[f"loo_cos_{tag}"] = dist(loo)
        d3[f"loo_cos_{tag}_boot"] = boot_ci(loo)
        npz[f"d3_loo_{tag}"] = loo
        for cell, held in SPLITS.items():
            pol_cos = cos_pri if cell.startswith("primary") else cos_sec
            d3[f"{cell}__{tag}"] = {
                "loo_at_held_out_states": dist(loo[held]),
                "policy_cos_at_held_out_states_RAW": dist(pol_cos[held]),
                "loo_minus_policy": paired_bootstrap(loo[held] - pol_cos[held], N_BOOT, SEED),
                "n_held_out_states_loo_beats_policy": int((loo[held] > pol_cos[held]).sum()),
                "policy_cos_at_TRAINED_states_RAW": dist(pol_cos[np.setdiff1d(np.arange(ns), held)]),
                "warning": "the policy's cos_own is a RAW cosine (C1-final's registered convention); "
                           "the MEAN_REMOVED row of this comparison is therefore NOT like-for-like "
                           "and is given only to show what the same baseline is worth once the "
                           "common mode is taken out"}
    out["D3_state_blind_LOO_baseline"] = d3
    log(f"[D3] LOO cos: raw median {d3['loo_cos_RAW']['median']:.4f}, mean-removed median "
        f"{d3['loo_cos_MEAN_REMOVED_bank_mu']['median']:.4f}")
    flush()

    # ======================================================================== input matrices
    last = np.asarray([{"like": 0, "dislike": 1, "noanswer": 2}.get(s.rsplit(".", 1)[-1], 3)
                       for s in sids], np.int64)
    Xs = np.concatenate([depth[:, None].astype(np.float64),
                         np.eye(4)[last],
                         np.log(nfit)[:, None], np.log(occ.astype(np.float64))[:, None]], 1)
    XIN = {"latent_zbar_200d": Xz,
           "warm_start_emission_129d": warm_zbar,
           "structural_only": Xs,
           "latent_plus_warm_plus_structural": np.concatenate([Xz, warm_zbar, Xs], 1),
           "size_and_depth_only": np.concatenate(
               [depth[:, None].astype(np.float64), np.log(nfit)[:, None]], 1)}

    # ======================================================================== D4 ridge -> target
    def cos_score(a, b):
        return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-300))

    d4 = {"design": "ridge from the state's input vector to the 129-d blend target; alpha chosen by "
                    "LEAVE-ONE-OUT INSIDE THE TRAINING STATES ONLY; evaluated at the held-out states",
          "cells": {}}
    for cell, held in SPLITS.items():
        tr = np.setdiff1d(np.arange(ns), held)
        pol_cos = cos_pri if cell.startswith("primary") else cos_sec
        rec = {"n_train": int(len(tr)), "n_held_out": int(len(held)),
               "trained_policy_cos_held_out_RAW": dist(pol_cos[held]),
               "trained_policy_cos_trained_RAW": dist(pol_cos[tr]),
               "arms": {}}
        for space, Tm in (("MEAN_REMOVED_bank_mu", Tmr), ("RAW", Tt.astype(np.float64))):
            for xn, X0 in XIN.items():
                Xstd = standardise(X0[tr], X0)
                a, curve = sweep_alpha_loo(Xstd[tr], Tm[tr], cos_score)
                M = ridge_fit(Xstd[tr], Tm[tr], a)
                ph = ridge_pred(M, Xstd[held]); pt = ridge_pred(M, Xstd[tr])
                ch = cosrows(ph, Tm[held]); ct = cosrows(pt, Tm[tr])
                # chance: a random direction, scored the same way
                rr = np.random.default_rng(SEED + 11).standard_normal(ph.shape)
                rec["arms"][f"{space}__{xn}"] = {
                    "alpha_loo": a,
                    "held_out_cos": dist(ch), "held_out_cos_boot": boot_ci(ch, N_BOOT, SEED),
                    "train_fit_cos": dist(ct),
                    "loo_cos_within_train": max(v for _, v in curve),
                    "chance_random_direction_cos": dist(cosrows(rr, Tm[held]))}
                log(f"  [D4 {cell}/{space}/{xn}] alpha {a:.3g} | held-out cos "
                    f"{np.mean(ch):+.4f} [{boot_ci(ch, 2000, SEED)['lo']:+.4f}, "
                    f"{boot_ci(ch, 2000, SEED)['hi']:+.4f}] | train {np.mean(ct):+.4f}")
        d4["cells"][cell] = rec
        out["D4_ridge_state_to_target"] = d4
        flush()

    # ======================================================================== D5 ridge -> profile
    def ndcg_at(pred, true, k=5):
        rel = np.maximum(true, 0.0)
        o = np.argsort(-pred)[:k]
        dcg = float((rel[o] / np.log2(np.arange(2, k + 2))).sum())
        io = np.argsort(-rel)[:k]
        idcg = float((rel[io] / np.log2(np.arange(2, k + 2))).sum())
        return dcg / idcg if idcg > 0 else np.nan

    def sp_score(a, b):
        ra, rb = ranks(a), ranks(b)
        ra -= ra.mean(); rb -= rb.mean()
        return float(ra @ rb / (np.linalg.norm(ra) * np.linalg.norm(rb) + 1e-300))

    def blend_from_profile(pred, m=5):
        o = np.argsort(-pred)[:m]
        w = np.maximum(pred[o], 0.0)
        if w.sum() <= 0:
            return None
        w = w / w.sum()
        return (w[:, None] * Epool[S[o]].astype(np.float64)).sum(0)

    d5 = {"design": "ridge from the state's input vector to the measured gain profile over the "
                    f"{len(S)} shared candidates; then the blend is REBUILT from the predicted "
                    "profile by build_targets' own rule (top-5, weights proportional to the positive "
                    "part) and scored against the true target",
          "n_shared_candidates": int(len(S)), "cells": {}}
    for cell, held in SPLITS.items():
        tr = np.setdiff1d(np.arange(ns), held)
        pol_cos = cos_pri if cell.startswith("primary") else cos_sec
        rec = {"n_train": int(len(tr)), "n_held_out": int(len(held)),
               "trained_policy_cos_held_out_RAW": dist(pol_cos[held]), "arms": {}}
        # baselines
        gm = G[tr].mean(0)
        base = {"global_mean_train_profile": np.tile(gm, (len(held), 1)),
                "popularity_prior": np.tile(pop_prior[S], (len(held), 1))}
        arms = {}
        for xn, X0 in XIN.items():
            Xstd = standardise(X0[tr], X0)
            a, curve = sweep_alpha_loo(Xstd[tr], G[tr], sp_score)
            M = ridge_fit(Xstd[tr], G[tr], a)
            arms[f"ridge__{xn}"] = (ridge_pred(M, Xstd[held]), a,
                                    max(v for _, v in curve))
        for bn, Pm in base.items():
            arms[f"baseline__{bn}"] = (Pm, None, None)
        # the ceiling for the reconstruction: the TRUE top-5 restricted to the shared set
        oracle = np.array([blend_from_profile(G[i]) for i in held])
        rec["oracle_blend_from_TRUE_profile_on_shared_set"] = {
            "cos_to_true_target_RAW": dist(cosrows(oracle, Tt[held].astype(np.float64))),
            "cos_to_true_target_MEAN_REMOVED": dist(cosrows(oracle - mu[None, :], Tmr[held])),
            "note": "the ceiling any D5 reconstruction can reach, given the shared-candidate "
                    "restriction; a gap to 1.0 is the restriction's cost, not a model failure"}
        for an, (Pm, a, loo) in arms.items():
            sp = np.array([sp_score(Pm[k], G[i]) for k, i in enumerate(held)])
            nd = np.array([ndcg_at(Pm[k], G[i], 5) for k, i in enumerate(held)])
            bl = [blend_from_profile(Pm[k]) for k in range(len(held))]
            ok = np.array([b is not None for b in bl])
            BL = np.stack([b for b in bl if b is not None]) if ok.any() else np.zeros((0, Tt.shape[1]))
            cr = cosrows(BL, Tt[held][ok].astype(np.float64)) if ok.any() else np.array([np.nan])
            cm = cosrows(BL - mu[None, :], Tmr[held][ok]) if ok.any() else np.array([np.nan])
            rec["arms"][an] = {
                "alpha_loo": a, "loo_spearman_within_train": loo,
                "held_out_spearman": dist(sp), "held_out_spearman_boot": boot_ci(sp, N_BOOT, SEED),
                "held_out_ndcg5": dist(nd), "held_out_ndcg5_boot": boot_ci(nd, N_BOOT, SEED),
                "reconstructed_blend_cos_to_true_target_RAW": dist(cr),
                "reconstructed_blend_cos_to_true_target_MEAN_REMOVED": dist(cm),
                "reconstructed_blend_cos_MEAN_REMOVED_boot": boot_ci(cm, N_BOOT, SEED),
                "n_states_with_a_constructible_blend": int(ok.sum())}
            log(f"  [D5 {cell}/{an}] spearman {np.mean(sp):+.4f} | ndcg@5 {np.nanmean(nd):.4f} | "
                f"blend cos raw {np.nanmean(cr):+.4f} mr {np.nanmean(cm):+.4f}")
        d5["cells"][cell] = rec
        out["D5_profile_parameterised_alternative"] = d5
        flush()

    # ======================================================================== D6 input sufficiency
    d6 = {"pairwise_state_input_similarity": {
              "cohort_mean_latent_MEAN_REMOVED": dist(offdiag(X_zcos)),
              "cohort_mean_latent_RAW": dist(offdiag(X_zraw)),
              "warm_start_emission_MEAN_REMOVED": dist(offdiag(X_wcos)),
              "n_distinct_states": ns,
              "min_pairwise_cos_mean_removed": float(offdiag(X_zcos).min()),
              "max_pairwise_cos_mean_removed": float(offdiag(X_zcos).max())},
          "trivial_confounds": {}}
    for cell, held in SPLITS.items():
        tr = np.setdiff1d(np.arange(ns), held)
        X0 = XIN["size_and_depth_only"]
        Xstd = standardise(X0[tr], X0)
        a, curve = sweep_alpha_loo(Xstd[tr], G[tr], sp_score)
        M = ridge_fit(Xstd[tr], G[tr], a)
        Pm = ridge_pred(M, Xstd[held])
        sp = np.array([sp_score(Pm[k], G[i]) for k, i in enumerate(held)])
        # variance of the profile explained by size+depth, per candidate, on held-out states
        gm = G[tr].mean(0)
        ss_tot = float(((G[held] - gm[None, :]) ** 2).sum())
        ss_res = float(((G[held] - Pm) ** 2).sum())
        d6["trivial_confounds"][cell] = {
            "held_out_spearman_size_and_depth_only": dist(sp),
            "held_out_spearman_boot": boot_ci(sp, N_BOOT, SEED),
            "R2_vs_global_mean_profile": 1.0 - ss_res / max(ss_tot, 1e-300),
            "alpha_loo": a}
    # how much of the pairwise profile similarity is explained by depth/size similarity alone
    dd = -np.abs(depth[:, None] - depth[None, :]).astype(np.float64)
    ls = np.log(nfit)
    ds = -np.abs(ls[:, None] - ls[None, :])
    d6["pair_level_confounds"] = {
        "neg_abs_depth_diff__vs__profile_spearman": cluster_boot_corr(dd, Y_sp, kind="pearson"),
        "neg_abs_log_size_diff__vs__profile_spearman": cluster_boot_corr(ds, Y_sp, kind="pearson"),
        "neg_abs_depth_diff__vs__latent_input_cos": cluster_boot_corr(dd, X_zcos, kind="pearson")}
    out["D6_input_sufficiency"] = d6
    flush()

    # ======================================================================== verdicts
    def _v(cell):
        r4 = out["D4_ridge_state_to_target"]["cells"][cell]
        best4 = max((v["held_out_cos"]["mean"], k) for k, v in r4["arms"].items()
                    if k.startswith("MEAN_REMOVED"))
        r5 = out["D5_profile_parameterised_alternative"]["cells"][cell]
        best5 = max((v["held_out_spearman"]["mean"], k) for k, v in r5["arms"].items()
                    if k.startswith("ridge__"))
        gmean = r5["arms"]["baseline__global_mean_train_profile"]["held_out_spearman"]["mean"]
        return {"D4_best_meanremoved_heldout_cos": best4[0], "D4_best_arm": best4[1],
                "D4_trained_policy_heldout_cos_RAW": r4["trained_policy_cos_held_out_RAW"]["mean"],
                "D5_best_ridge_heldout_spearman": best5[0], "D5_best_arm": best5[1],
                "D5_global_mean_baseline_spearman": gmean,
                "D5_ridge_beats_global_mean": bool(best5[0] > gmean)}
    out["verdict_inputs"] = {c: _v(c) for c in SPLITS}
    flush()
    log(f"[done] wrote {p_json} in {out['minutes']:.1f} min")
    return out


def main():
    ap = argparse.ArgumentParser(description="Paper C C5 diagnostics (read-only, banked artifacts)")
    ap.add_argument("--m", type=int, default=5)
    ap.add_argument("--snap_users", type=int, default=10000)
    ap.add_argument("--out_tag", default="")
    ap.add_argument("--snapshot", default=None)
    ap.add_argument("--concept_ckpt", default=None)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--allow_dirty", action="store_true")
    args = ap.parse_args()
    from strategy_channel_suite import CERT_TOWER, CERT_CONC
    args.snapshot = args.snapshot or CERT_TOWER
    args.concept_ckpt = args.concept_ckpt or CERT_CONC
    log("#" * 100)
    log("# PAPER C / C5 -- diagnostics on C1-final's generalisation failure. Nothing is trained; "
        "TRAIN-side banked artifacts only; the 10,000 canonical TEST users are not touched.")
    log("#" * 100)
    return run(args)


if __name__ == "__main__":
    main()
