r"""PAPER C / STAGE C1 -- THE PILOT: can a gain-weighted BLEND target pull the emitted vector off the
question grid?

Design sheet: `docs/design/DS_C1_PILOT.md` (PRE-REGISTERED 2026-08-21, BINDING; author approved a
pilot/PoC the same date). Baseline it is read against: `docs/results/C0_GEOMETRY.md` +
`src/instrument/c0_geometry.py`, whose `geometry()` / `bilinear_cos()` / `dist()` / `bank_vectors()`
are IMPORTED here, never reimplemented. Data provenance: `docs/design/C1_GAIN_TABLE_AUDIT.md`.

WHAT IS MEASURED
----------------
E1's head scores `s(q|state) = <u(h), k(phi_q)> + b(phi_q)`, i.e. in C0's homogeneous coordinates
`z_hat = [u(h), 1]` against `e_q = [k(phi_q), b(phi_q)]`. C0 measured that a cross-entropy-trained
E1 emits a vector that is nowhere near its own snapped question (median cos1 -0.182). C1 replaces the
one-hot target with `target = sum_j w_j e_{q_j}`, `w_j ∝ max(gain_j, 0)` over the top-m MEASURED
candidates at the state, and asks whether the SAME architecture can be trained to land on it.

STAGE 0 IS A GATE, AND IT IS FREE. If the measured gain profiles are so peaked that the blend
collapses onto the top-1 embedding (median target-cos1 > 0.99), the premise is at fault and the pilot
STOPS before training.

★ THE ANALYTIC CEILING (added at implementation time, reported in Stage 0, NOT a deviation).
The emitted vector's last coordinate is pinned to 1 by construction, so for a target
`t = [t_u, t_b]` the best cosine ANY setting of `u(h)` can achieve is

    ceiling = 1                       if t_b >= 0     (attained at u = t_u / t_b)
    ceiling = |t_u| / |t|             if t_b <  0     (a supremum as |u| -> infinity, not attained)

This is arithmetic about the parameterisation, not an experimental result, and it is computed and
printed BEFORE any optimiser runs, because a registered criterion of `cos >= 0.90` is only meaningful
if 0.90 is reachable at all.

THE 42 STATES (design sheet "Data"; `C1_GAIN_TABLE_AUDIT.md` section 2.1)
------------------------------------------------------------------------
  35   `policy_labels_deep_partial` sweeps whose `nid` AND cohort size match a `coldtree_v2` node
       (full ~1,000-candidate exact tables; the audit's contract check is re-run here as an assert)
   7   single-pick (`len(seq) == 1`) `ndcgtree_heaps` nodes -- untouched seed heaps, whole-pool exact
  ---
  42   the ENTIRE exact-coverage population. Not a sample of it (HARD RULE 1). No partial heap, no
       stale bound, no tail state, no new measurement.

  +7   the deep sweeps that do NOT map to a `coldtree_v2` node (they are children of nodes the tree
       left as leaves). Paid-for and exact, but outside the registered 42, so they are carried as a
       DESCRIPTIVE secondary set: measured in Stage 0, never trained on, never in a registered cell.

VALIDATION ONLY. The 10,000 canonical TEST users are not built, not loaded, not touched.
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
sys.path.insert(0, _HERE)
if "--threads" not in sys.argv:                        # 4-physical-core box, shared with other work
    sys.argv += ["--threads", "3"]
_NT = sys.argv[sys.argv.index("--threads") + 1]
os.environ["OMP_NUM_THREADS"] = _NT
os.environ.setdefault("MKL_NUM_THREADS", _NT)
os.environ.setdefault("OPENBLAS_NUM_THREADS", _NT)

import json
import time
import copy
import argparse
import numpy as np
import torch

from policy_net import Policy, build_phi, load_traj, make_batch, log, TOTAL_L, OUTDIR
from c0_geometry import (PAPERC, E1_CKPT, VAL_M0_FULL, N_BANK, sha256, git, bank_vectors, augment,
                         geometry, bilinear_cos, dist)

C0_JSON = os.path.join(PAPERC, "c0_geometry.json")
C0_CKPT_SHA = "9c4c61e7387d29c61f4cee62600627eff6178d2dbe3fb4ea6b5a19f7e6003058"
COLDTREE = os.path.join(OUTDIR, "coldtree_v2.json")


# ============================================================================ the 42 states
def load_the_42(pool_index, Dva):
    """The registered population, assembled from the two banked sources and contract-checked."""
    import policy_p1b as P1B
    A, _B = P1B.load_states(pool_index)                 # 42 deep sweeps (group A), full tables
    C = P1B.load_onmanifold(pool_index, Dva)            # single-pick heaps, whole-pool exact
    ct = json.load(open(COLDTREE))["nodes"]
    mapped, unmapped = [], []
    for s in A:
        nid = s["sid"].split("@")[0]
        n = ct.get(nid)
        (mapped if (n is not None and int(n["n_users"]) == len(s["rows"])) else unmapped).append(s)
        s["nid"] = nid
    for s in mapped:
        s["group"] = "deep_sweep"; s["registered"] = True
    for s in unmapped:
        s["group"] = "deep_sweep_unmapped"; s["registered"] = False
    for s in C:
        s["group"] = "single_pick_heap"; s["registered"] = True
        s["nid"] = s["sid"]
    assert len(mapped) == 35, f"expected 35 coldtree_v2-mapped deep sweeps, got {len(mapped)}"
    assert len(C) == 7, f"expected 7 single-pick heap states, got {len(C)}"
    assert len(unmapped) == 7
    reg = mapped + C
    assert len({s["sid"] for s in reg}) == 42
    return reg, unmapped


def build_targets(states, Epool, Kpool, pool_index, m):
    """`target = sum_j w_j e_{q_j}`, `w_j ∝ max(gain_j, 0)` over the top-m measured candidates.

    Candidates already asked in the state's own prefix are excluded, exactly as the policy's argmax
    excludes them. Both C0 spaces are built: homogeneous (primary, `e_q = [k, b]`) and bias-free
    (secondary, `k` only) -- the split that shows a high cosine bought with the common-mode component.
    """
    T, TK, meta = [], [], []
    for s in states:
        asked = {pool_index[q] for q in s["prefix"]}
        cand = np.asarray([q for q in sorted(s["gains"]) if q not in asked], np.int64)
        g = np.asarray([s["gains"][int(q)] for q in cand], np.float64)
        o = np.argsort(-g)[:m]
        q, w = cand[o], np.maximum(g[o], 0.0)
        assert w.sum() > 0, f"{s['sid']}: no positive gain in the top-{m}"
        w = w / w.sum()
        T.append((w[:, None] * Epool[q]).sum(0).astype(np.float32))
        TK.append((w[:, None] * Kpool[q]).sum(0).astype(np.float32))
        meta.append({"sid": s["sid"], "top_m_pool_idx": [int(x) for x in q],
                     "top_m_gain": [float(x) for x in g[o]], "weights": [float(x) for x in w],
                     "w1": float(w[0]), "n_positive_gain_candidates": int((g > 0).sum()),
                     "n_candidates_measured": int(len(cand))})
    return np.stack(T), np.stack(TK), meta


def ceiling(T):
    """The best cos(z_hat, target) the parameterisation admits (see the module docstring)."""
    tb = T[:, -1].astype(np.float64)
    tu = np.linalg.norm(T[:, :-1].astype(np.float64), axis=1)
    tn = np.linalg.norm(T.astype(np.float64), axis=1)
    return np.where(tb >= 0, 1.0, tu / np.maximum(tn, 1e-30))


# ============================================================================ geometry on 42 rows
def state_geometry(V, states, Ebank, pool2full, chunk=1000, label=""):
    """C0's `geometry()` verbatim, on one vector per state.

    `geometry()` indexes states as `row = user*TOTAL_L + position` and excludes the questions asked
    before that position. State i is therefore placed at row `i*TOTAL_L + len(prefix_i)` in a padded
    array, its prefix written into `asked_full[i]`, and the padding rows are dropped afterwards. This
    is bookkeeping so the SAME function runs; no geometry is re-derived.
    """
    n = len(states)
    Z = np.zeros((n * TOTAL_L, V.shape[1]), np.float32)
    af = np.zeros((n, TOTAL_L - 1), np.int64)
    idx = np.zeros(n, np.int64)
    for i, s in enumerate(states):
        k = len(s["prefix"])
        assert k < TOTAL_L
        idx[i] = i * TOTAL_L + k
        Z[i * TOTAL_L: (i + 1) * TOTAL_L] = V[i]        # padding rows carry the same vector
        for j, q in enumerate(s["prefix"]):
            af[i, j] = pool2full[s["pool_prefix_idx"][j]]
    G = geometry(Z, Ebank, af, chunk=chunk, label=label)
    return {k: np.asarray(v)[idx] for k, v in G.items()}


def geom_summary(G, extra=None):
    o = {k: dist(G[k]) for k in ("cos1", "m", "ent50", "cos1_bilinear") if k in G}
    for k in ("n_band_abs", "n_band_range"):
        if k in G:
            o[k + "_frac_ge3"] = float(np.mean(np.asarray(G[k]) >= 3))
    if extra:
        o.update(extra)
    return o


# ============================================================================ state features
class Feats:
    """Everything the state encoder consumes, precomputed ONCE per state cohort.

    `pooled`, `cnt`, `n_asked`, `pos` and `z` are pure functions of the data -- no policy parameter
    enters them -- so precomputing them is an identity, and `selftest_forward` asserts that against
    `pol.enc` itself rather than trusting this docstring. The trainable path is then
    `mlp(cat(z, relu(hist(pooled)), cnt, n_asked, pos)) -> head.q`, which is what makes twelve
    fine-tunes cost minutes instead of hours.
    """

    def __init__(self, z, pooled, cnt, nasked, pos, sid, hq, ho, hm):
        self.z, self.pooled, self.cnt, self.nasked, self.pos = z, pooled, cnt, nasked, pos
        self.sid, self.hq, self.ho, self.hm = sid, hq, ho, hm


def precompute(E, states, pool_index, phi_t, log_every=5, ubatch=1000):
    from strategy_channel_suite import genv_latent
    from ndcg_tree import branch_of_per_user
    Z, PO, CN, NA, PS, SI, HQ, HO, HM = [], [], [], [], [], [], [], [], []
    t0 = time.time()
    for i, s in enumerate(states):
        rws = s["rows"]
        n = len(rws)
        prefix = s["prefix"]
        k = len(prefix)
        s["pool_prefix_idx"] = [pool_index[q] for q in prefix]
        z = genv_latent(E, prefix, rws).numpy().astype(np.float32)
        outs = (np.stack([np.asarray(branch_of_per_user(E, rws, [q] * n), np.int64)
                          for q in prefix], 1) if k else np.zeros((n, 0), np.int64))
        hq = np.zeros((n, TOTAL_L - 1), np.int64)
        ho = np.zeros((n, TOTAL_L - 1), np.int64)
        hm = np.zeros((n, TOTAL_L - 1), np.float32)
        for j in range(k):
            hq[:, j] = s["pool_prefix_idx"][j]; ho[:, j] = outs[:, j]; hm[:, j] = 1.0
        po, cn = [], []
        for st in range(0, n, ubatch):                   # the (n, 15, d_phi*3) token tensor is the
            sl = slice(st, min(st + ubatch, n))          # only large intermediate -- batch it
            hp = torch.from_numpy(phi_t[hq[sl]])
            oh = torch.nn.functional.one_hot(torch.from_numpy(ho[sl]), 3).to(hp.dtype)
            tok = (hp.unsqueeze(-1) * oh.unsqueeze(-2)).reshape(sl.stop - sl.start, TOTAL_L - 1, -1)
            mm = torch.from_numpy(hm[sl]).unsqueeze(-1)
            po.append(((tok * mm).sum(1) / mm.sum(1).clamp(min=1.0)).numpy().astype(np.float32))
            cn.append((oh * mm).sum(1).numpy().astype(np.float32))
            del tok
        Z.append(z)
        PO.append(np.concatenate(po))
        CN.append(np.concatenate(cn))
        NA.append(hm.sum(1, keepdims=True).astype(np.float32))
        PS.append(np.tile(np.asarray([[k, TOTAL_L - k]], np.float32), (n, 1)))
        SI.append(np.full(n, i, np.int64))
        HQ.append(hq); HO.append(ho); HM.append(hm)
        if (i + 1) % log_every == 0 or i == len(states) - 1:
            log(f"  [precompute] {i + 1}/{len(states)} states, "
                f"{sum(len(x) for x in Z):,} user-rows ({(time.time() - t0) / 60:.1f}m)")
    return Feats(np.concatenate(Z), np.concatenate(PO), np.concatenate(CN), np.concatenate(NA),
                 np.concatenate(PS), np.concatenate(SI), np.concatenate(HQ), np.concatenate(HO),
                 np.concatenate(HM))


def enc_h(pol, F, sl):
    """h = the state encoding, through the policy's OWN modules."""
    parts = [torch.from_numpy(F.z[sl]),
             torch.relu(pol.enc.hist(torch.from_numpy(F.pooled[sl]))),
             torch.from_numpy(F.cnt[sl]), torch.from_numpy(F.nasked[sl])]
    if pol.enc.use_pos:
        parts.append(torch.from_numpy(F.pos[sl]))
    return pol.enc.mlp(torch.cat(parts, -1))


def emit(pol, F, sl):
    """u(h) for a slice of user-rows."""
    return pol.head.q(enc_h(pol, F, sl))


def selftest_forward(pol, F, phi_t, n=256):
    """The precomputed path must equal `pol.enc` + `pol.head.q` EXACTLY (up to float32 rounding)."""
    sl = slice(0, min(n, F.z.shape[0]))
    with torch.no_grad():
        got = emit(pol, F, sl).numpy()
        ref = pol.head.q(pol.enc(torch.from_numpy(F.z[sl]), torch.from_numpy(phi_t[F.hq[sl]]),
                                 torch.from_numpy(F.ho[sl]), torch.from_numpy(F.hm[sl]),
                                 torch.from_numpy(F.pos[sl]))).numpy()
    d = float(np.abs(got - ref).max())
    assert d < 1e-4, f"the precomputed encoder path is not the policy's own forward (max |diff| {d:g})"
    return d


def emit_all(pol, F, batch=4096):
    with torch.no_grad():
        return np.concatenate([emit(pol, F, slice(st, min(st + batch, F.z.shape[0]))).numpy()
                               for st in range(0, F.z.shape[0], batch)])


# ============================================================================ the fine-tune
def finetune(pol, F, Tt, w_row, arm, seed, epochs, batch, lr):
    """Loss `1 - cos(z_hat, target)` at the states, each state weighted equally regardless of cohort
    size. `head.k` / `head.b` receive no gradient from this loss by construction (the target is a
    constant built from them), so the bank the geometry is measured against cannot move -- in EITHER
    arm. That is a property of the loss, not a freezing decision."""
    torch.manual_seed(seed); np.random.seed(seed)
    for p in pol.parameters():
        p.requires_grad_(False)
    train = list(pol.head.q.parameters())
    if arm == "full":
        train += list(pol.enc.parameters())
    for p in train:
        p.requires_grad_(True)
    opt = torch.optim.Adam(train, lr=lr)
    N = F.z.shape[0]
    Hc = None
    if arm == "head_only":            # the encoder is frozen, so h is a CONSTANT: cache it once.
        with torch.no_grad():         # An identity, not an approximation -- and the head-only arm
            Hc = np.concatenate([enc_h(pol, F, slice(st, min(st + 4096, N))).numpy()
                                 for st in range(0, N, 4096)])   # then costs one small MLP a step.
    Tn = torch.nn.functional.normalize(torch.from_numpy(Tt), dim=1)
    wr = torch.from_numpy(w_row)
    si = torch.from_numpy(F.sid)
    rng = np.random.default_rng(seed)
    hist = []
    t0 = time.time()
    for ep in range(epochs):
        perm = rng.permutation(N)
        tot = wt = 0.0
        for st in range(0, N, batch):
            idx = np.sort(perm[st:min(st + batch, N)])
            u = (pol.head.q(torch.from_numpy(Hc[idx])) if Hc is not None else emit(pol, F, idx))
            zh = torch.cat([u, torch.ones(len(idx), 1)], 1)
            c = (torch.nn.functional.normalize(zh, dim=1) * Tn[si[idx]]).sum(1)
            w = wr[idx]
            loss = ((1.0 - c) * w).sum() / w.sum()
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss) * float(w.sum()); wt += float(w.sum())
        hist.append(tot / wt)
        if ep == 0 or (ep + 1) % 10 == 0 or ep == epochs - 1:
            log(f"  [{arm}/seed{seed}] epoch {ep + 1}/{epochs} weighted loss {hist[-1]:.6f} "
                f"({(time.time() - t0) / 60:.1f}m)")
    for p in pol.parameters():
        p.requires_grad_(False)
    return hist


def per_state_eval(pol, F, states, Tt, TKt):
    """Per-user cosine to the target, and the per-state emitted vector (the cohort mean of z_hat)."""
    U = emit_all(pol, F)
    Z = np.concatenate([U, np.ones((U.shape[0], 1), np.float32)], 1)
    Tn = Tt / (np.linalg.norm(Tt, axis=1, keepdims=True) + 1e-30)
    cu = (Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-30) * Tn[F.sid]).sum(1)
    n = len(states)
    zbar = np.zeros((n, Z.shape[1]), np.float32)
    ubar = np.zeros((n, U.shape[1]), np.float32)
    cos_mean_user = np.zeros(n); cos_bar = np.zeros(n); disp = np.zeros(n); cos_bf = np.zeros(n)
    TKn = TKt / (np.linalg.norm(TKt, axis=1, keepdims=True) + 1e-30)
    for i in range(n):
        m = F.sid == i
        zbar[i] = Z[m].mean(0); ubar[i] = U[m].mean(0)
        cos_mean_user[i] = float(cu[m].mean())
        cos_bar[i] = float(zbar[i] @ Tn[i] / (np.linalg.norm(zbar[i]) + 1e-30))
        zn = Z[m] / (np.linalg.norm(Z[m], axis=1, keepdims=True) + 1e-30)
        disp[i] = float((zn @ (zbar[i] / (np.linalg.norm(zbar[i]) + 1e-30))).mean())
        cos_bf[i] = float(ubar[i] @ TKn[i] / (np.linalg.norm(ubar[i]) + 1e-30))
    return {"z_hat_bar": zbar, "u_bar": ubar, "cos_target_user_mean": cos_mean_user,
            "cos_target_of_state_mean": cos_bar, "within_state_cos_to_mean": disp,
            "cos_target_bias_free": cos_bf, "cos_target_per_user": cu.astype(np.float32)}


# ============================================================================ collateral
def collateral(pol, pol0, Dva, phi_t, P, n_states, seed=98765, batch=1000):
    """Snap agreement with the banked E1 at random ORDINARY states (the validation trajectory bank's
    own (user, position) states). REPORTED, not gated -- the pilot claims capability, not a
    deployable policy (design sheet, Stage 1)."""
    rng = np.random.default_rng(seed)
    n = len(Dva["uid"])
    u = rng.integers(0, n, n_states)
    t = rng.integers(0, TOTAL_L, n_states)
    a0, a1 = [], []
    phi_g = torch.from_numpy(phi_t)
    with torch.no_grad():
        for st in range(0, n_states, batch):
            sl = slice(st, min(st + batch, n_states))
            z, hp, ho, hm, pos, am, _y = make_batch(Dva, phi_t, u[sl], t[sl], P)
            a0.append(pol0(z, hp, ho, hm, pos, phi_g, am).argmax(1).numpy())
            a1.append(pol(z, hp, ho, hm, pos, phi_g, am).argmax(1).numpy())
    a0 = np.concatenate(a0); a1 = np.concatenate(a1)
    return {"n_states": int(n_states), "snap_agreement_vs_banked_E1": float((a0 == a1).mean()),
            "n_distinct_questions_banked_E1": int(len(np.unique(a0))),
            "n_distinct_questions_finetuned": int(len(np.unique(a1)))}


# ============================================================================ the run
def run(args):
    from strategy_channel_suite import build_greedy_env, contract_selftest, genv_mean
    from ndcg_tree import build_pool, leak_assert, branch_of_selftest
    t00 = time.time()
    os.makedirs(PAPERC, exist_ok=True)
    torch.manual_seed(98765); np.random.seed(98765)

    blob = torch.load(args.ckpt, map_location="cpu")
    assert blob["arch"] == "equivariant" and blob["state"] == "both"
    sha = sha256(args.ckpt)
    assert sha == C0_CKPT_SHA, f"this is not the C0-measured checkpoint (sha {sha})"
    log(f"[ckpt] {os.path.basename(args.ckpt)} epoch={blob['epoch']} val_full={blob['val_full']!r} "
        f"sha256 matches c0_geometry.json")

    E = build_greedy_env(args.snapshot, "validation", args.concept_ckpt, "paperC/c1")
    contract_selftest(E, E["rows"]); leak_assert(E, E["rows"], "paperC/c1")
    rows = E["rows"]
    assert len(rows) == 10000
    f0, t0_ = genv_mean(E, [], rows)
    assert f0 == VAL_M0_FULL, f"CANONICAL SNAP FAILED: {f0!r}"
    log(f"[gate] canonical m=0 snap full={f0!r} -- exact")

    pool = [(int(a), int(b)) for a, b in blob["pool"]]
    pool_e, _, _ = build_pool(E, 500, 500)
    assert pool_e == pool, "the env's pool is not the checkpoint's pool"
    branch_of_selftest(E, rows[:2000], [pool[0], pool[500], (1, 114)])
    pool_index = {q: i for i, q in enumerate(pool)}
    P = len(pool)
    full_pool = [(0, i) for i in range(E["ctx"].ni)] + [(1, c) for c in range(E["C"])]
    assert len(full_pool) == N_BANK
    pool2full = np.asarray([(i if ch == 0 else E["ctx"].ni + i) for ch, i in pool], np.int64)

    norm = (blob["norm_mu"], blob["norm_sd"])
    phi_t, _ = build_phi(E, pool, norm=norm)
    phi_full, _ = build_phi(E, full_pool, norm=norm)
    pol0 = Policy(E["ctx"].d, phi_full.shape[1], len(full_pool), arch=blob["arch"],
                  state=blob["state"], use_pos=blob["use_pos"])
    pol0.load_state_dict(blob["model"]); pol0.eval()
    Ebank, Kbank = bank_vectors(pol0, phi_full)
    Epool, Kpool = bank_vectors(pol0, phi_t)
    log(f"[bank] {Ebank.shape[0]:,} x {Ebank.shape[1]} (d_k={Kbank.shape[1]} + bias coordinate); "
        f"b(phi) over the bank: median {np.median(Ebank[:, -1]):.3f} "
        f"[{Ebank[:, -1].min():.3f}, {Ebank[:, -1].max():.3f}]")

    Dva = load_traj(train=False)
    assert Dva["pool"] == pool
    reg, extra = load_the_42(pool_index, Dva)
    states = reg + extra
    log(f"[states] registered 42 = {sum(s['group'] == 'deep_sweep' for s in reg)} coldtree_v2-mapped "
        f"deep sweeps + {sum(s['group'] == 'single_pick_heap' for s in reg)} single-pick heaps; "
        f"descriptive secondary {len(extra)}")

    out = {"analysis": "paperC_C1_pilot", "design_sheet": "docs/design/DS_C1_PILOT.md",
           "citable": True, "test_touched": False, "split": "validation",
           "checkpoint": {"path": os.path.relpath(args.ckpt, _ROOT), "sha256": sha,
                          "epoch": int(blob["epoch"]), "val_full": float(blob["val_full"])},
           "git": {"rev": git("rev-parse", "HEAD"),
                   "branch": git("rev-parse", "--abbrev-ref", "HEAD")},
           "c0_random_floor": json.load(open(C0_JSON))["arms"]["pool"]["random_floor"],
           "states": {"n_registered": len(reg), "n_secondary_unmapped": len(extra),
                      "cohort_users_registered": int(sum(len(s["rows"]) for s in reg)),
                      "composition": [{"sid": s["sid"], "group": s["group"],
                                       "n_users": len(s["rows"]), "prefix_len": len(s["prefix"]),
                                       "n_candidates": s["n_measured_in_pool"]} for s in states]},
           "stage0": {}, "stage1": {}}
    p_json = os.path.join(PAPERC, "c1_pilot.json")
    npz = {}

    def flush():
        out["minutes"] = round((time.time() - t00) / 60, 2)
        with open(p_json + ".tmp", "w") as fh:
            json.dump(out, fh, indent=1, default=float)
        os.replace(p_json + ".tmp", p_json)
        np.savez_compressed(os.path.join(PAPERC, "c1_pilot_per_state.npz"), **npz)

    # ---- precompute the states' features (needed by Stage 0 only for the prefix indices) ---------
    log("[precompute] building per-user state features for all states")
    F = precompute(E, states, pool_index, phi_t)
    _mreg = F.sid < len(reg)
    F_reg = Feats(*[getattr(F, k)[_mreg] for k in
                    ("z", "pooled", "cnt", "nasked", "pos", "sid", "hq", "ho", "hm")])
    d_id = selftest_forward(pol0, F, phi_t)
    log(f"[selftest] precomputed encoder path == pol.enc + head.q, max |diff| {d_id:.3g}")
    out["precompute"] = {"n_user_rows": int(F.z.shape[0]),
                         "n_user_rows_registered": int(F_reg.z.shape[0]),
                         "encoder_path_identity_max_abs_err": d_id}
    flush()

    # ================================================================= STAGE 0 -- the free gate
    log("#" * 100)
    log("# STAGE 0 -- are the blend targets themselves off the grid?")
    S0 = {}
    targets = {}
    for m in (5, 3, 10):
        Tt, TKt, meta = build_targets(states, Epool, Kpool, pool_index, m)
        targets[m] = (Tt, TKt, meta)
        cl = ceiling(Tt)
        G = state_geometry(Tt, states, Ebank, pool2full, chunk=args.chunk, label=f"target_m{m}")
        G["cos1_bilinear"] = bilinear_cos(TKt, Kbank, G["snap"])
        rng = np.random.default_rng(98765)
        R = rng.standard_normal(TKt.shape).astype(np.float32)
        R *= (np.linalg.norm(TKt, axis=1) / np.linalg.norm(R, axis=1))[:, None]
        GR = state_geometry(augment(R), states, Ebank, pool2full, chunk=args.chunk,
                            label=f"rand_m{m}")
        GR["cos1_bilinear"] = bilinear_cos(R, Kbank, GR["snap"])
        cell = {}
        for name, sel in (("registered42", np.arange(len(reg))),
                          ("secondary_unmapped7", np.arange(len(reg), len(states)))):
            sub = {k: np.asarray(v)[sel] for k, v in G.items()}
            cell[name] = geom_summary(sub, {
                "n_states": int(len(sel)),
                "analytic_ceiling_on_cos_to_target": dist(cl[sel]),
                "frac_states_ceiling_ge_0.90": float((cl[sel] >= 0.90).mean()),
                "target_bias_coordinate": dist(Tt[sel, -1]),
                "top1_weight_w1": dist([meta[i]["w1"] for i in sel]),
                "n_positive_gain_candidates": dist(
                    [meta[i]["n_positive_gain_candidates"] for i in sel]),
                "target_snap_equals_top_gain_candidate": float(np.mean(
                    [G["snap"][i] == pool2full[meta[i]["top_m_pool_idx"][0]] for i in sel])),
                "random_floor": geom_summary({k: np.asarray(v)[sel] for k, v in GR.items()})})
        S0[f"m{m}"] = cell
        npz[f"target_m{m}"] = Tt
        npz[f"target_m{m}_cos1"] = np.asarray(G["cos1"])
        npz[f"target_m{m}_m"] = np.asarray(G["m"])
        npz[f"target_m{m}_snap"] = np.asarray(G["snap"])
        npz[f"target_m{m}_ceiling"] = cl
        r = cell["registered42"]
        log(f"[stage0 m={m}] registered 42: median target-cos1 {r['cos1']['median']:.5f} "
            f"| median margin {r['m']['median']:.5f} | median w1 {r['top1_weight_w1']['median']:.4f} "
            f"| median analytic ceiling {r['analytic_ceiling_on_cos_to_target']['median']:.5f} "
            f"| random floor cos1 {r['random_floor']['cos1']['median']:.5f}")
    prim = S0["m5"]["registered42"]
    gate = bool(prim["cos1"]["median"] > 0.99)
    S0["gate"] = {"criterion": "STOP if median target-cos1 (m=5, registered 42) > 0.99",
                  "median_target_cos1_m5": prim["cos1"]["median"],
                  "targets_collapse_onto_top1": gate,
                  "verdict": "STOP -- the blends collapse onto e_(1)" if gate else
                             "PROCEED -- the targets are genuinely off the grid"}
    out["stage0"] = S0
    out["stage0_meta_m5"] = targets[5][2]
    flush()
    log(f"[stage0 GATE] {S0['gate']['verdict']}")
    if gate or args.stage0_only:
        out["stopped_after_stage0"] = True
        flush()
        return out

    # ================================================================= STAGE 1 -- fine-tune
    log("#" * 100)
    log("# STAGE 1 -- fine-tune from the banked E1 checkpoint")
    Tt, TKt, _meta = targets[args.m]
    Tt_reg, TKt_reg = Tt[:len(reg)], TKt[:len(reg)]
    G0 = {k: np.asarray(v) for k, v in state_geometry(
        Tt, states, Ebank, pool2full, chunk=args.chunk, label="tgt").items()}
    nst = np.asarray([len(s["rows"]) for s in states], np.float64)
    w_row = (1.0 / nst)[F.sid].astype(np.float32)
    w_row_reg = (1.0 / nst[:len(reg)])[F_reg.sid].astype(np.float32)

    base = per_state_eval(pol0, F, states, Tt, TKt)
    out["stage1"]["banked_E1_before_finetune"] = {
        "cos_target_user_mean": dist(base["cos_target_user_mean"]),
        "cos_target_of_state_mean": dist(base["cos_target_of_state_mean"]),
        "cos_target_bias_free": dist(base["cos_target_bias_free"])}
    log(f"[stage1] banked E1 BEFORE fine-tuning: median cos(z_hat, target) "
        f"{np.median(base['cos_target_user_mean']):.5f}")
    npz["before_zbar"] = base["z_hat_bar"]
    flush()

    _rng = np.random.default_rng(98765)
    perm = _rng.permutation(len(reg))
    while (perm == np.arange(len(reg))).any():           # a derangement: no state keeps its own
        perm = _rng.permutation(len(reg))                # target under the shuffle control
    runs = [("head_only", s, False) for s in args.seeds] + \
           [("full", s, False) for s in args.seeds] + \
           [("head_only", args.seeds[0], True), ("full", args.seeds[0], True)]
    res = {}
    for arm, seed, shuf in runs:
        key = f"{arm}/seed{seed}" + ("/SHUFFLED" if shuf else "")
        log("-" * 90)
        log(f"[run] {key}")
        Ta = Tt_reg[perm] if shuf else Tt_reg
        TKa = TKt_reg[perm] if shuf else TKt_reg
        pol = copy.deepcopy(pol0)
        hist = finetune(pol, F_reg, Ta, w_row_reg, arm, seed, args.epochs, args.batch, args.lr)
        ev = per_state_eval(pol, F_reg, reg, Ta, TKa)
        Ggot = {k: np.asarray(v) for k, v in state_geometry(
            ev["z_hat_bar"], reg, Ebank, pool2full, chunk=args.chunk, label=key).items()}
        Ggot["cos1_bilinear"] = bilinear_cos(ev["u_bar"], Kbank, Ggot["snap"])
        tgt_cos1 = G0["cos1"][:len(reg)][perm] if shuf else G0["cos1"][:len(reg)]
        tgt_snap = G0["snap"][:len(reg)][perm] if shuf else G0["snap"][:len(reg)]
        r = {"arm": arm, "seed": seed, "shuffled_targets": shuf,
             "loss_history": [float(x) for x in hist], "final_weighted_loss": float(hist[-1]),
             "criterion1_cos_target": dist(ev["cos_target_user_mean"]),
             "cos_target_of_state_mean": dist(ev["cos_target_of_state_mean"]),
             "cos_target_bias_free_secondary": dist(ev["cos_target_bias_free"]),
             "within_state_cos_to_mean": dist(ev["within_state_cos_to_mean"]),
             "frac_states_cos_ge_0.90": float((ev["cos_target_user_mean"] >= 0.90).mean()),
             "criterion2_abs_cos1_diff_vs_target": dist(np.abs(Ggot["cos1"] - tgt_cos1)),
             "criterion2_same_nearest_bank_question": float((Ggot["snap"] == tgt_snap).mean()),
             "emitted_geometry": geom_summary(Ggot)}
        if not shuf:
            r["collateral"] = collateral(pol, pol0, Dva, phi_t, P, args.n_collateral)
        res[key] = r
        npz[f"zbar_{key.replace('/', '_')}"] = ev["z_hat_bar"]
        npz[f"cos_{key.replace('/', '_')}"] = ev["cos_target_user_mean"]
        out["stage1"]["runs"] = res
        flush()
        log(f"[{key}] median cos(z_hat,target) {r['criterion1_cos_target']['median']:.5f} "
            f"| |dcos1| {r['criterion2_abs_cos1_diff_vs_target']['median']:.5f} "
            f"| same nearest {r['criterion2_same_nearest_bank_question']:.4f} "
            f"| emitted median cos1 {r['emitted_geometry']['cos1']['median']:.5f}"
            + ("" if shuf else f" | collateral snap agreement "
                               f"{r['collateral']['snap_agreement_vs_banked_E1']:.4f}"))

    # ---- the registered reading -------------------------------------------------------------------
    reading = {}
    for arm in ("head_only", "full"):
        ks = [f"{arm}/seed{s}" for s in args.seeds]
        Zs = [npz[f"zbar_{k.replace('/', '_')}"] for k in ks]
        pw = []
        for a in range(len(Zs)):
            for b in range(a + 1, len(Zs)):
                na = Zs[a] / (np.linalg.norm(Zs[a], axis=1, keepdims=True) + 1e-30)
                nb = Zs[b] / (np.linalg.norm(Zs[b], axis=1, keepdims=True) + 1e-30)
                pw.append((na * nb).sum(1))
        pw = np.concatenate(pw)
        c1 = float(np.mean([res[k]["criterion1_cos_target"]["mean"] for k in ks]))
        c1med = float(np.mean([res[k]["criterion1_cos_target"]["median"] for k in ks]))
        c2a = float(np.mean([res[k]["criterion2_abs_cos1_diff_vs_target"]["median"] for k in ks]))
        c2b = float(np.mean([res[k]["criterion2_same_nearest_bank_question"] for k in ks]))
        sk = f"{arm}/seed{args.seeds[0]}/SHUFFLED"
        reading[arm] = {
            "criterion1_seed_avg_mean_cos": c1, "criterion1_seed_avg_median_cos": c1med,
            "criterion1_pass": bool(c1 >= 0.90),
            "criterion2_median_abs_cos1_diff": c2a, "criterion2_same_nearest_rate": c2b,
            "criterion2_pass": bool(c2a <= 0.05 and c2b >= 0.80),
            "criterion3_pairwise_seed_cos": dist(pw), "criterion3_pass": bool(np.median(pw) >= 0.95),
            "shuffle_control": {
                "true_assignment_median_cos": c1med,
                "shuffled_assignment_median_cos": res[sk]["criterion1_cos_target"]["median"],
                "true_fits_better": bool(
                    c1med > res[sk]["criterion1_cos_target"]["median"])},
            "collateral_snap_agreement_seed_avg": float(np.mean(
                [res[k]["collateral"]["snap_agreement_vs_banked_E1"] for k in ks]))}
        reading[arm]["SUCCESS_all_three"] = bool(reading[arm]["criterion1_pass"] and
                                                 reading[arm]["criterion2_pass"] and
                                                 reading[arm]["criterion3_pass"])
        log("=" * 90)
        log(f"[REGISTERED READING / {arm}] c1 cos {c1:.5f} {'PASS' if reading[arm]['criterion1_pass'] else 'FAIL'}"
            f" | c2 |dcos1| {c2a:.5f} same-nearest {c2b:.4f} "
            f"{'PASS' if reading[arm]['criterion2_pass'] else 'FAIL'}"
            f" | c3 seed cos {np.median(pw):.5f} "
            f"{'PASS' if reading[arm]['criterion3_pass'] else 'FAIL'}"
            f" | shuffle true>shuffled {reading[arm]['shuffle_control']['true_fits_better']}")
    out["stage1"]["registered_reading"] = reading
    flush()
    log(f"[done] wrote {p_json} in {out['minutes']:.1f} minutes")
    return out


def main():
    ap = argparse.ArgumentParser(description="Paper C stage C1 -- see docs/design/DS_C1_PILOT.md")
    ap.add_argument("--ckpt", default=E1_CKPT)
    ap.add_argument("--m", type=int, default=5, help="blend width for Stage 1 (primary m=5)")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seeds", default="1,2,3")
    ap.add_argument("--n_collateral", type=int, default=1000)
    ap.add_argument("--chunk", type=int, default=200)
    ap.add_argument("--stage0_only", action="store_true")
    ap.add_argument("--snapshot", default=None)
    ap.add_argument("--concept_ckpt", default=None)
    ap.add_argument("--threads", type=int, default=3)
    args = ap.parse_args()
    args.seeds = [int(s) for s in args.seeds.split(",") if s]
    from strategy_channel_suite import CERT_TOWER, CERT_CONC
    args.snapshot = args.snapshot or CERT_TOWER
    args.concept_ckpt = args.concept_ckpt or CERT_CONC
    log("#" * 100)
    log("# PAPER C / C1 PILOT -- blend targets at the 42 exact-gain states. VALIDATION ONLY; the "
        "10,000 TEST users are not built, not loaded, not touched.")
    log("#" * 100)
    return run(args)


if __name__ == "__main__":
    main()
