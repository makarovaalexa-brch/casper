r"""PAPER C / STAGE C0 -- THE GEOMETRY CHECK on E1's emitted vector.

Design sheet: `docs/design/DS_C0_GEOMETRY.md` (PRE-REGISTERED 2026-08-21, BINDING). Read it first.
Nothing here is trained. No test user is built, loaded or touched: the only environment this file can
construct is `split="validation"`.

THE QUESTION
------------
E1's head scores a candidate as

    s(q | state) = <u(h_state), k(phi_q)> + b(phi_q)

so the selection is an argmax of an inner product between ONE emitted vector and a per-candidate
descriptor. Paper C wants to name a question the bank does not contain, which requires the emitted
vector to carry off-catalogue content. E1 was trained by cross-entropy toward the tree's one-hot pick,
so nothing in its loss rewards landing BETWEEN two questions (blocker 1). C0 measures whether it does
anyway.

★ THE BIAS TERM, AND WHY THE EMITTED VECTOR IS 129-DIMENSIONAL
---------------------------------------------------------------
The design sheet writes the snap as `argmax_q <z_hat, e_q>`. The head's score carries an extra
per-candidate bias `b(phi_q)`, so a bare `u(h)` / `k(phi)` inner product is NOT the object the policy
actually maximises. Dropping it would make every geometry number describe a scoring rule the policy
does not use, and gate 1 (the snap must reproduce E1's banked pick) could not pass exactly.

The lossless fix is homogeneous coordinates -- no approximation, an identity:

    z_hat = [ u(h) , 1 ]   in R^(d_k+1)          e_q = [ k(phi_q) , b(phi_q) ]   in R^(d_k+1)
    <z_hat, e_q> = <u(h), k(phi_q)> + b(phi_q) = s(q | state)   EXACTLY

That is the primary representation, and `--selftest` asserts the identity against `pol.head(...)`
itself rather than trusting this docstring. The bilinear-only variant (`u(h)` vs `k(phi)`, bias
discarded) is reported alongside as a SECONDARY number, because it is the quantity a reader who
ignores the bias would compute, and the two are not the same.

WHY A FORWARD HOOK AND NOT A REIMPLEMENTED ROLLOUT
--------------------------------------------------
The E1 lesson is that only the same code path proves anything. So the states come from
`policy_net.rollout` UNCHANGED -- the policy picks its own questions, the answers come from
`branch_of_per_user`, the folding comes from `genv_eval_multi` -- and `u(h)` is captured by a forward
hook on `pol.head.q`. Nothing about the rollout is re-derived here; if the hook's ordering were wrong,
gate 1 would fail, because the reconstructed argmax would not be the rollout's own pick.

THE TWO ARMS
------------
  pool        (PRIMARY, gate-carrying) the canonical E1 validation rollout over the 1,000-candidate
              pool -- the rollout whose realised interviews are banked in
              `policy_main_equivariant_both_gates_per_user.npz`. The emitted vector at each of its
              160,000 states is then scored against the FULL 19,390-question bank, so "in-pool vs
              out-of-pool snap" is a property of where the open bank would send the vector.
  catalogue   (SECONDARY) the scalability rollout: the same policy choosing from all 19,390
              candidates, where 29.99% of the picks are out of pool by E1's measurement. Its states
              are NOT the states G-E1b scored, so it carries no gate; it exists so the in/out-of-pool
              split can also be read on E1's own realised open-vocabulary interviews.

HARD RULE 1: no sampling anywhere. All 10,000 validation users, all 16 positions, all 19,390 bank
entries. Memory is handled by chunking over states (lossless), never by subsetting the bank.
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
sys.path.insert(0, _HERE)
if "--threads" not in sys.argv:                        # 4-physical-core box; the box may be shared
    sys.argv += ["--threads", "4"]
_NT = sys.argv[sys.argv.index("--threads") + 1]
os.environ["OMP_NUM_THREADS"] = _NT
os.environ.setdefault("MKL_NUM_THREADS", _NT)
os.environ.setdefault("OPENBLAS_NUM_THREADS", _NT)

import json
import time
import hashlib
import argparse
import subprocess
import numpy as np
import torch

from policy_net import Policy, build_phi, rollout, log, TOTAL_L, OUTDIR

PAPERC = os.path.join(_ROOT, "experiments", "paperC")
E1_CKPT = os.path.join(OUTDIR, "policy_main_equivariant_both.pt")
E1_GATES = os.path.join(OUTDIR, "policy_main_equivariant_both_gates.json")
E1_PER_USER = os.path.join(OUTDIR, "policy_main_equivariant_both_gates_per_user.npz")
# Banked E1 numbers, ASSERTED rather than trusted (design sheet gate 2).
VAL_M0_FULL = 0.15686538118354684
E1_VAL_FULL = 0.28354971825495506
E1_VAL_TAIL = 0.13421311934575303
N_BANK = 19390


def sha256(path, buf=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            b = fh.read(buf)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def git(*args):
    try:
        return subprocess.run(["git"] + list(args), cwd=_ROOT, capture_output=True,
                              text=True, timeout=60).stdout.strip()
    except Exception as e:                                             # pragma: no cover
        return f"<git failed: {e}>"


# ============================================================================ the emitted vector
class Emissions:
    """Captures `u(h)` for every state the rollout visits, in visit order, via a forward hook on
    `pol.head.q`. The rollout itself is untouched."""

    def __init__(self, pol):
        self.buf = []
        self.h = pol.head.q.register_forward_hook(lambda m, i, o: self.buf.append(o.detach().clone()))

    def close(self):
        self.h.remove()

    def stack(self, n_users, L=TOTAL_L):
        """(n_users*L, d_k) float32, row index = u*L + t. `rollout` loops t on the OUTSIDE and the
        user batches on the INSIDE, so the captures concatenate to L blocks of n_users, in order."""
        assert self.buf, "no emission was captured -- the hook never fired"
        z = torch.cat(self.buf, 0).numpy().astype(np.float32)
        assert z.shape[0] == n_users * L, f"captured {z.shape[0]} emissions, expected {n_users * L}"
        return z.reshape(L, n_users, z.shape[1]).transpose(1, 0, 2).reshape(n_users * L, z.shape[1])


def bank_vectors(pol, phi):
    """e_q = [k(phi_q), b(phi_q)] for every bank entry, float32."""
    with torch.no_grad():
        k = pol.head.k(torch.from_numpy(phi)).numpy().astype(np.float32)
        b = pol.head.b(torch.from_numpy(phi)).numpy().astype(np.float32).reshape(-1, 1)
    return np.concatenate([k, b], 1), k


def augment(u):
    return np.concatenate([u, np.ones((u.shape[0], 1), np.float32)], 1)


def identity_selftest(pol, phi, u):
    """The homogeneous-coordinate identity, asserted against the head itself on real emissions."""
    E, _ = bank_vectors(pol, phi)
    got = augment(u[:64]) @ E.T
    # the head's own scoring expression, on the same emissions
    with torch.no_grad():
        ref = (torch.from_numpy(u[:64]) @ pol.head.k(torch.from_numpy(phi)).T
               + pol.head.b(torch.from_numpy(phi)).squeeze(-1)[None, :]).numpy()
    d = float(np.abs(got - ref).max())
    assert d < 1e-3, f"the augmented inner product is not the head's score (max |diff| {d:g})"
    return d


# ============================================================================ geometry
def geometry(Zhat, Ebank, asked_full, chunk=1000, top=50, label=""):
    """Per-state geometry against the FULL bank, chunked over states (lossless).

    `asked_full[u, t]` = the bank index the rollout asked at position t. At state (u, t) the
    candidates asked at positions < t are excluded, exactly as the policy's own argmax excludes them.
    """
    N, P = Zhat.shape[0], Ebank.shape[0]
    En = np.linalg.norm(Ebank, axis=1).astype(np.float32)
    Et = torch.from_numpy(np.ascontiguousarray(Ebank.T))
    out = {k: np.zeros(N, np.float32) for k in
           ("cos1", "m", "s1", "s2", "s3", "s10", "ent50", "cos1_bilinear")}
    for k in ("n_band_abs", "n_band_range", "snap"):
        out[k] = np.zeros(N, np.int64)
    zn = np.linalg.norm(Zhat, axis=1).astype(np.float32)
    t0 = time.time()
    for st in range(0, N, chunk):
        sl = slice(st, min(st + chunk, N))
        C = sl.stop - sl.start
        S = (torch.from_numpy(np.ascontiguousarray(Zhat[sl])) @ Et)
        # exclude the already-asked, exactly as the rollout's own argmax does
        rows = np.arange(sl.start, sl.stop)
        u, t = rows // TOTAL_L, rows % TOTAL_L
        mr, mc = [], []
        for j in range(TOTAL_L - 1):
            m = t > j
            if not m.any():
                break
            mr.append(np.flatnonzero(m)); mc.append(asked_full[u[m], j])
        if mr:
            S[np.concatenate(mr), np.concatenate(mc)] = float("-inf")
        v, ix = torch.topk(S, top, dim=1)
        v64 = v.double()
        s1 = v64[:, 0]
        out["s1"][sl] = s1.numpy(); out["s2"][sl] = v64[:, 1].numpy()
        out["s3"][sl] = v64[:, 2].numpy(); out["s10"][sl] = v64[:, 9].numpy()
        out["m"][sl] = ((v64[:, 0] - v64[:, 1]) / (v64[:, 0] - v64[:, 9])).numpy()
        p = torch.softmax(v64 - s1[:, None], 1)
        out["ent50"][sl] = (-(p * torch.log(p.clamp_min(1e-300))).sum(1)).numpy()
        i1 = ix[:, 0].numpy()
        out["snap"][sl] = i1
        num = (torch.from_numpy(np.ascontiguousarray(Zhat[sl]))
               * torch.from_numpy(Ebank[i1])).sum(1).numpy()
        out["cos1"][sl] = num / (zn[sl] * En[i1] + 1e-30)
        # two readings of "within 5% of s1", both reported: absolute (5% of |s1|) and
        # scale-free (5% of the s1..s10 range, the same range the margin ratio normalises by)
        thr_a = (s1 - 0.05 * s1.abs()).float()[:, None]
        thr_r = (s1 - 0.05 * (s1 - v64[:, 9])).float()[:, None]
        out["n_band_abs"][sl] = (S >= thr_a).sum(1).numpy()
        out["n_band_range"][sl] = (S >= thr_r).sum(1).numpy()
        del S
        if (st // chunk) % 40 == 0:
            log(f"  [{label}] {sl.stop:,}/{N:,} states ({(time.time() - t0) / 60:.1f}m)")
    return out


def bilinear_cos(Zhat_k, Kbank, snap):
    """SECONDARY: cosine in the bias-free bilinear space, at the SAME snapped question."""
    num = (Zhat_k * Kbank[snap]).sum(1)
    return (num / (np.linalg.norm(Zhat_k, axis=1) * np.linalg.norm(Kbank[snap], axis=1) + 1e-30)
            ).astype(np.float32)


def dist(x):
    x = np.asarray(x, np.float64)
    x = x[np.isfinite(x)]
    if not len(x):
        return None
    q = np.percentile(x, [0, 1, 5, 10, 25, 50, 75, 90, 95, 99, 100])
    return {"n": int(len(x)), "mean": float(x.mean()), "std": float(x.std()),
            "min": float(q[0]), "p1": float(q[1]), "p5": float(q[2]), "p10": float(q[3]),
            "q1": float(q[4]), "median": float(q[5]), "q3": float(q[6]), "p90": float(q[7]),
            "p95": float(q[8]), "p99": float(q[9]), "max": float(q[10]),
            "iqr": float(q[6] - q[4])}


def summarise(G, mask=None):
    sel = slice(None) if mask is None else mask
    o = {k: dist(G[k][sel]) for k in ("cos1", "m", "ent50", "cos1_bilinear")}
    for k in ("n_band_abs", "n_band_range"):
        o[k] = dist(G[k][sel])
        o[k + "_frac_ge3"] = float(np.mean(np.asarray(G[k][sel]) >= 3))
        o[k + "_frac_ge2"] = float(np.mean(np.asarray(G[k][sel]) >= 2))
    o["n_states"] = int(len(np.asarray(G["cos1"])[sel]))
    return o


# ============================================================================ the run
def run(args):
    from strategy_channel_suite import build_greedy_env, contract_selftest, genv_mean
    from ndcg_tree import build_pool, leak_assert, branch_of_selftest
    t00 = time.time()
    os.makedirs(PAPERC, exist_ok=True)
    torch.manual_seed(98765); np.random.seed(98765)

    blob = torch.load(args.ckpt, map_location="cpu")
    assert blob["arch"] == "equivariant", "C0 reads the equivariant head's emitted vector"
    log(f"[ckpt] {os.path.basename(args.ckpt)} epoch={blob['epoch']} "
        f"val_full={blob['val_full']!r} arm={blob.get('arm')}")

    E = build_greedy_env(args.snapshot, "validation", args.concept_ckpt, "paperC/c0")
    contract_selftest(E, E["rows"]); leak_assert(E, E["rows"], "paperC/c0")
    rows = E["rows"]
    assert len(rows) == 10000, f"the validation cohort is {len(rows)} users, expected 10,000"

    # ---- GATE 2a: the canonical snap, before any geometry number is read ------------------------
    f0, t0_ = genv_mean(E, [], rows)
    assert f0 == VAL_M0_FULL, f"CANONICAL SNAP FAILED: m=0 full {f0!r}, banked {VAL_M0_FULL!r}"
    log(f"[gate2a] m=0 intercept full={f0!r} tail={t0_!r} -- exact")

    pool = [(int(a), int(b)) for a, b in blob["pool"]]
    pool_e, _, _ = build_pool(E, 500, 500)
    assert pool_e == pool, "the env's pool is not the checkpoint's pool"
    branch_of_selftest(E, rows[:2000], [pool[0], pool[500], (1, 114)])
    full_pool = [(0, i) for i in range(E["ctx"].ni)] + [(1, c) for c in range(E["C"])]
    assert len(full_pool) == N_BANK, f"the bank is {len(full_pool)}, expected {N_BANK}"
    pool2full = np.asarray([(i if ch == 0 else E["ctx"].ni + i) for ch, i in pool], np.int64)
    in_pool_full = np.zeros(N_BANK, bool); in_pool_full[pool2full] = True
    chan_full = np.asarray([ch for ch, _ in full_pool], np.int8)

    norm = (blob["norm_mu"], blob["norm_sd"])
    phi_small, _ = build_phi(E, pool, norm=norm)
    phi_full, _ = build_phi(E, full_pool, norm=norm)
    pol = Policy(E["ctx"].d, phi_full.shape[1], len(full_pool), arch=blob["arch"],
                 state=blob["state"], use_pos=blob["use_pos"])
    pol.load_state_dict(blob["model"]); pol.eval()
    Ebank, Kbank = bank_vectors(pol, phi_full)
    Epool, Kpool = bank_vectors(pol, phi_small)
    log(f"[bank] {Ebank.shape[0]:,} x {Ebank.shape[1]} emitted-space descriptors "
        f"(d_k={Kbank.shape[1]} + 1 bias coordinate)")

    banked = np.load(E1_PER_USER)
    banked_asked = np.asarray(banked["asked"], np.int64)
    assert np.array_equal(np.asarray(banked["rows"], np.int64), np.asarray(rows, np.int64))

    out = {"analysis": "paperC_C0_geometry",
           "design_sheet": "docs/design/DS_C0_GEOMETRY.md", "citable": True,
           "test_touched": False,
           "cohort": "10,000 VALIDATION users x 16 positions = 160,000 states",
           "checkpoint": {"path": os.path.relpath(args.ckpt, _ROOT), "sha256": sha256(args.ckpt),
                          "epoch": int(blob["epoch"]), "val_full": float(blob["val_full"]),
                          "arm": blob.get("arm")},
           "git": {"rev": git("rev-parse", "HEAD"), "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
                   "status_short": git("status", "--short")},
           "emitted_vector": {
               "definition": "z_hat = [u(h_state), 1]; e_q = [k(phi_q), b(phi_q)]; "
                             "<z_hat, e_q> = the head's score EXACTLY (homogeneous coordinates, "
                             "not an approximation -- the head carries a per-candidate bias and a "
                             "bias-free inner product is not the rule the policy maximises)",
               "d_k": int(Kbank.shape[1]), "d_emitted": int(Ebank.shape[1])},
           "gates": {"canonical_snap_m0_full": {"got": f0, "expected": VAL_M0_FULL, "pass": True},
                     "canonical_snap_m0_tail": t0_},
           "arms": {}}
    p_json = os.path.join(PAPERC, "c0_geometry.json")

    def flush():
        out["minutes"] = round((time.time() - t00) / 60, 2)
        with open(p_json + ".tmp", "w") as fh:
            json.dump(out, fh, indent=1, default=float)
        os.replace(p_json + ".tmp", p_json)

    flush()
    per_state = {}

    for arm in args.arms:
        log("#" * 100)
        log(f"# ARM {arm}")
        phi_a, pool_a = ((phi_small, pool) if arm == "pool" else (phi_full, full_pool))
        cap = Emissions(pol)
        try:
            ro = rollout(E, pol, phi_a, pool_a, rows, batch=args.batch)
        finally:
            cap.close()
        U = cap.stack(len(rows))
        cap.buf = []
        d_id = identity_selftest(pol, phi_a, U)
        Zhat = augment(U)
        asked = ro["asked"]
        vf, vt = ro["curves"][TOTAL_L]
        A = {"n_states": int(U.shape[0]),
             "augmented_identity_max_abs_err_vs_head": d_id,
             "val_full": float(np.nanmean(vf)), "val_tail": float(np.nanmean(vt)),
             "answer_rate": float((ro["outs"] != 2).mean()),
             "n_distinct_interviews": len({tuple(r) for r in asked})}

        # ---- GATES ------------------------------------------------------------------------------
        if arm == "pool":
            A["gate_realised_interviews_match_banked"] = bool(np.array_equal(asked, banked_asked))
            A["gate_val_full_matches_banked"] = {"got": A["val_full"], "expected": E1_VAL_FULL,
                                                 "abs_diff": abs(A["val_full"] - E1_VAL_FULL)}
            A["gate_val_tail_matches_banked"] = {"got": A["val_tail"], "expected": E1_VAL_TAIL,
                                                 "abs_diff": abs(A["val_tail"] - E1_VAL_TAIL)}
            assert A["gate_realised_interviews_match_banked"], (
                "GATE 2 FAILED: the regenerated rollout's realised interviews are NOT the banked E1 "
                "interviews. Stop -- this is a harness bug, not a geometry result.")
            assert A["gate_val_full_matches_banked"]["abs_diff"] < 1e-9
            log("[gate2b] realised interviews byte-identical to the banked E1 rollout; "
                f"val full {A['val_full']!r} == banked")
            asked_full = pool2full[asked]
            snap_ref = asked_full
            Eref, aref = Epool, asked
        else:
            asked_full = asked
            snap_ref = asked
            Eref, aref = Ebank, asked

        # GATE 1: the snap reproduces the rollout's own pick at EVERY state, from the reconstructed
        # emitted vector and the reconstructed bank -- over the arm's OWN candidate set.
        Zh_ref = Zhat
        n_bad = 0
        Ert = torch.from_numpy(np.ascontiguousarray(Eref.T))
        for st in range(0, Zh_ref.shape[0], args.chunk):
            sl = slice(st, min(st + args.chunk, Zh_ref.shape[0]))
            S = torch.from_numpy(np.ascontiguousarray(Zh_ref[sl])) @ Ert
            r = np.arange(sl.start, sl.stop); u, t = r // TOTAL_L, r % TOTAL_L
            mr, mc = [], []
            for j in range(TOTAL_L - 1):
                m = t > j
                if not m.any():
                    break
                mr.append(np.flatnonzero(m)); mc.append(aref[u[m], j])
            if mr:
                S[np.concatenate(mr), np.concatenate(mc)] = float("-inf")
            n_bad += int((S.argmax(1).numpy() != aref[u, t]).sum())
            del S
        A["gate_snap_reproduces_pick"] = {"n_states": int(Zh_ref.shape[0]), "n_mismatch": n_bad,
                                          "top1_agreement": 1.0 - n_bad / Zh_ref.shape[0]}
        assert n_bad == 0, (
            f"GATE 1 FAILED: the reconstructed snap disagrees with the rollout's own pick at "
            f"{n_bad} of {Zh_ref.shape[0]} states. Stop -- harness bug.")
        log(f"[gate1] snap reproduces the pick at all {Zh_ref.shape[0]:,} states (agreement 1.0000)")

        # ---- the geometry, against the FULL 19,390 bank ------------------------------------------
        G = geometry(Zhat, Ebank, asked_full, chunk=args.chunk, label=arm)
        G["cos1_bilinear"] = bilinear_cos(U, Kbank, G["snap"])
        r = np.arange(Zhat.shape[0])
        G["position"] = (r % TOTAL_L).astype(np.int16)
        G["user_row"] = np.asarray(rows, np.int64)[r // TOTAL_L]
        G["snap_in_pool"] = in_pool_full[G["snap"]]
        G["snap_channel"] = chan_full[G["snap"]]
        G["pick_bank_idx"] = snap_ref.reshape(-1)
        G["snap_is_the_pick"] = (G["snap"] == G["pick_bank_idx"])
        A["full_bank_snap_equals_the_arm_pick"] = float(G["snap_is_the_pick"].mean())

        A["overall"] = summarise(G)
        A["by_position"] = [summarise(G, G["position"] == t) for t in range(TOTAL_L)]
        A["by_snap_in_pool"] = {
            "in_pool": summarise(G, G["snap_in_pool"]),
            "out_of_pool": summarise(G, ~G["snap_in_pool"]),
            "frac_states_whose_full_bank_snap_is_out_of_pool": float((~G["snap_in_pool"]).mean())}
        A["by_snap_channel"] = {
            "item": summarise(G, G["snap_channel"] == 0),
            "concept": summarise(G, G["snap_channel"] == 1),
            "concept_share_of_full_bank_snaps": float((G["snap_channel"] == 1).mean())}

        # ---- GATE 3: the scale control, scored by the SAME code ----------------------------------
        rng = np.random.default_rng(98765)
        R = rng.standard_normal(U.shape).astype(np.float32)
        R *= (np.linalg.norm(U, axis=1) / np.linalg.norm(R, axis=1))[:, None]   # norm-matched
        GR = geometry(augment(R), Ebank, asked_full, chunk=args.chunk,
                      label=arm + "/rand")
        GR["cos1_bilinear"] = bilinear_cos(R, Kbank, GR["snap"])
        A["random_floor"] = summarise(GR)
        A["random_floor"]["note"] = ("a random Gaussian direction in the emitted space, rescaled to "
                                     "the SAME per-state norm as u(h) and augmented with the same "
                                     "bias coordinate, scored by the same geometry() call")
        A["random_floor"]["n_distinct_snapped_questions"] = int(len(np.unique(GR["snap"])))

        # registered reading rule (DS_C0_GEOMETRY.md)
        mc1, mm = A["overall"]["cos1"]["median"], A["overall"]["m"]["median"]
        f3 = A["overall"]["n_band_abs_frac_ge3"]
        A["registered_reading_rule"] = {
            "median_cos1": mc1, "median_m": mm, "frac_states_ge3_in_5pct_band_abs": f3,
            "frac_states_ge3_in_5pct_band_range": A["overall"]["n_band_range_frac_ge3"],
            "on_the_grid": bool(mc1 >= 0.98 and mm >= 0.90),
            "between_candidates": bool(mm <= 0.60 or f3 >= 0.25),
            "cell": ("on_the_grid" if (mc1 >= 0.98 and mm >= 0.90) else
                     ("between_candidates" if (mm <= 0.60 or f3 >= 0.25) else "intermediate"))}
        out["arms"][arm] = A
        flush()
        for k in ("cos1", "m", "ent50", "cos1_bilinear", "n_band_abs", "n_band_range", "snap",
                  "position", "user_row", "snap_in_pool", "snap_channel", "pick_bank_idx",
                  "s1", "s2", "s3", "s10"):
            per_state[f"{arm}_{k}"] = G[k]
        for k in ("cos1", "m", "n_band_abs", "n_band_range"):
            per_state[f"{arm}_rand_{k}"] = GR[k]
        np.savez_compressed(os.path.join(PAPERC, "c0_geometry_per_state.npz"), **per_state)
        log(f"[{arm}] median cos1 {mc1:.5f} | median m {mm:.5f} | cell "
            f"{A['registered_reading_rule']['cell']} | random floor median cos1 "
            f"{A['random_floor']['cos1']['median']:.5f} m {A['random_floor']['m']['median']:.5f}")

    flush()
    log(f"[done] wrote {p_json} and c0_geometry_per_state.npz in {out['minutes']:.1f} min")
    return out


def main():
    ap = argparse.ArgumentParser(description="Paper C stage C0 -- see docs/design/DS_C0_GEOMETRY.md")
    ap.add_argument("--ckpt", default=E1_CKPT)
    ap.add_argument("--arms", default="pool,catalogue")
    ap.add_argument("--chunk", type=int, default=1000, help="states per score chunk (lossless)")
    ap.add_argument("--batch", type=int, default=1000, help="rollout batch (users)")
    ap.add_argument("--snapshot", default=None)
    ap.add_argument("--concept_ckpt", default=None)
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()
    args.arms = [a for a in args.arms.split(",") if a]
    assert all(a in ("pool", "catalogue") for a in args.arms)
    from strategy_channel_suite import CERT_TOWER, CERT_CONC
    args.snapshot = args.snapshot or CERT_TOWER
    args.concept_ckpt = args.concept_ckpt or CERT_CONC
    log("#" * 100)
    log("# PAPER C / C0 -- geometry of E1's emitted vector. VALIDATION ONLY; no training; the "
        "10,000 TEST users are not built, not loaded, not touched.")
    log("#" * 100)
    return run(args)


if __name__ == "__main__":
    main()
