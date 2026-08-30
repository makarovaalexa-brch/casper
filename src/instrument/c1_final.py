r"""PAPER C / STAGE C1-FINAL -- train the policy to EMIT off-catalogue directions on Paper B's COLD
tree, with held-out STATES, a replay term, a scale term, and two BLOCKING gates.

Design sheet: `docs/design/DS_C1_FINAL.md` (PRE-REGISTERED 2026-08-30, BINDING -- read it first).
Scoping: `docs/design/C1_FINAL_SCOPING.md`. Predecessor + caveats: `docs/results/C1_PILOT.md`.

WHAT THIS FIXES, RELATIVE TO THE PILOT
--------------------------------------
The pilot aimed the head at an off-grid point at 42 states and succeeded, but it was not a policy: at
ordinary states it collapsed to 4-6 distinct questions with 0.0000 snap agreement, its emissions were
38x out of scale, and with 42 nested states it could not test generalisation. C1-final:

  * TEACHER    `coldtree_v2_train10000.json` -- 144 nodes, 72 of them SEARCHED with a FULL exact
               whole-pool gain table in `coldtree_v2_train10000_sweeps/`. Blend targets at those 72
               node states; the teacher's ONE-HOT everywhere else (tail slots + below-floor
               inheritors). A MIXED target set, declared.
  * REPLAY     the incumbent distillation cross-entropy at every non-blend state, over the existing
               85,289-user cold trajectory bank `m6/traj_dj/cold_*.npz` -- free supervision, and the
               specific thing whose absence collapsed the pilot.
  * SCALE      `lambda * (||z_mr|| / ||t_mr|| - 1)^2`, lambda = 0.1, mean-removed norms. The pilot's
               cosine-only loss left the norm unconstrained; that is why its emissions were
               unrenderable.
  * HELD-OUT STATES  primary = the whole `root.like.*` subtree (15 held out / 57 trained, prefixes
               disjoint so nothing nests); secondary = depth >= 5 (30 / 42). Random state splits are
               FORBIDDEN: node cohorts nest and a random split leaks.

TWO IMPLEMENTATION RULINGS, STATED HERE BECAUSE THE SHEET DOES NOT SETTLE THEM
------------------------------------------------------------------------------
1. `head.k` / `head.b` ARE FROZEN. The blend target is a constant built out of those two modules, and
   the warm start's own cosine (reading 1b) is measured against the same bank. If the replay CE were
   allowed to move `head.k`/`head.b`, the space in which "cos(z_hat, target)" is defined would drift
   during training and the registered comparison would be between numbers from two different banks.
   In the pilot this was a property of the loss; here it has to be a decision, so it is recorded as
   one. Trainable: `enc.*` and `head.q.*`.
2. A HELD-OUT STATE RECEIVES NO LOSS AT ALL -- not the blend target, and not the incumbent CE either.
   Supervising it with the teacher's one-hot would actively pull the emission back onto the grid at
   exactly the states reading 1 scores, which would make the reading meaningless.

FIREWALL. Train-side states only, on the 85,289 cold-bank users. Nothing in this file can build a
test environment; the 10,000 canonical TEST users are not built, not loaded, not touched.
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
import glob
import json
import time

import numpy as np
import torch

from policy_net import Policy, build_phi, log, paired_bootstrap
from c0_geometry import (PAPERC, N_BANK, sha256, git_block, savez_atomic, index_map_assert,
                         bank_vectors, augment, dist, wilson)
from c1_pilot import build_targets, ceiling, state_geometry, geom_summary, cohort_hash

B = os.path.join(_ROOT, "experiments", "battery")
TREE_JSON = os.path.join(B, "coldtree_v2_train10000.json")
SWEEPS = os.path.join(B, "coldtree_v2_train10000_sweeps")
BANK_DIR = os.path.join(B, "m6", "traj_dj")
WARM = os.path.join(B, "m4", "m7dj_policy.pt")

TOTAL_L = 15                       # this tree's budget. policy_net.TOTAL_L is E1's 16 -- not ours.
SEED = 98765
N_EXT = 1004                       # m7dj's phi table: the 1,000-candidate pool + 4 anchor concepts
# The canonical snap for THIS split is the TRAIN m=0 intercept, banked as `base` in the root sweep.
# It is NOT the validation 0.15686538118354684 the rest of the C-series asserts.
TRAIN_M0_FULL = 0.15830706760250668
N_TRAIN = 140768
# The pilot's split-half direction stability, carried as a LABELLED PRIOR when this tree has no
# per-user gain sidecar to recompute it from (`C1_PILOT.md` repaired reading 2).
PILOT_SPLIT_HALF_MEANREMOVED = 0.656


def free_ram_gb():
    import ctypes

    class MS(ctypes.Structure):
        _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
    m = MS()
    m.dwLength = ctypes.sizeof(MS)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
    return m.ullAvailPhys / 1e9


# ============================================================================ the trajectory bank
def load_cold_bank():
    """The 85,289-user cold arm of `m6/traj_dj`, concatenated in slice order.

    REUSED, never rebuilt: a rebuild is ~1.42 GB against ~2.4 GB free and is not authorised. The
    `rows` field of each slice is LOCAL to the slice, so the canonical train uid is `lo + rows`
    exactly as `m4b_policy.consolidate` reconstructs it.
    """
    parts = sorted(glob.glob(os.path.join(BANK_DIR, "cold_*.npz")),
                   key=lambda p: int(os.path.basename(p)[5:].split("_")[0]))
    assert parts, f"no cold slices under {BANK_DIR}"
    lab, out, val, uid, n, d = [], [], [], [], 0, None
    for p in parts:
        z = np.load(p)
        if not len(z["rows"]):
            continue
        lo = int(os.path.basename(p)[5:].split("_")[0])
        assert int(z["anchor"]) == -1, "the cold arm must carry no anchor"
        assert int(z["n_fallback"]) == 0, f"{os.path.basename(p)}: n_fallback != 0"
        lab.append(z["label_idx"]); out.append(z["outcome"]); val.append(z["valid"])
        uid.append(lo + z["rows"].astype(np.int64))
        d = int(z["z"].shape[2]); n += len(z["rows"])
    lab = np.concatenate(lab); out = np.concatenate(out)
    val = np.concatenate(val); uid = np.concatenate(uid)
    assert len(np.unique(uid)) == len(uid), "a train user appears twice in the cold bank"
    assert lab.shape[1] == TOTAL_L, f"the bank is {lab.shape[1]} positions, expected {TOTAL_L}"
    need = n * TOTAL_L * d * 4 / 1e9
    have = free_ram_gb()
    assert have > need * 1.4, (
        f"the cold bank needs {need:.2f} GB resident but only {have:.2f} GB is free -- REFUSING to "
        f"start a run that will die part-way (HARD RULE 11d)")
    zz = np.empty((n, TOTAL_L, d), np.float32)
    at = 0
    for p in parts:
        z = np.load(p)
        k = len(z["rows"])
        if k:
            zz[at:at + k] = z["z"]; at += k
    assert at == n
    log(f"[bank] {n:,} cold users x {TOTAL_L} positions ({need:.2f} GB resident, "
        f"{have:.2f} GB free before); {len(np.unique(lab))} distinct label questions")
    return {"lab": lab, "out": out, "valid": val, "uid": uid, "z": zz, "n": n, "d": d,
            "n_slices": len(parts), "gb_resident": need}


# ============================================================================ the tree's states
def descend(nodes, bank):
    """Which NODE state every (user, position) of the bank sits at, verified against the bank's own
    labels.

    The bank was produced by `tree_route.route`, which emits the tree's own `path + steps` and never
    reconstructs it. Here the descent is re-derived from the banked TERNARY OUTCOMES alone and the
    resulting question sequence is asserted equal to the banked `label_idx` for every user -- so a
    wrong branch order, a wrong depth or a stale tree cannot pass silently (memory
    `cross-population-index-match-is-silent`: a partial match is indistinguishable from a subset).

    Returns `node_at[(n, TOTAL_L)] int32` = index into `node_ids` for positions on the descended
    chain, -1 for tail positions and for users who escaped into an unfitted branch.
    """
    node_ids = sorted(nodes)
    nidx = {k: i for i, k in enumerate(node_ids)}
    n = bank["n"]
    node_at = np.full((n, TOTAL_L), -1, np.int32)
    seq_of = np.full((n, TOTAL_L), -1, np.int64)      # the tree's own question sequence, as tuples id
    unfitted = np.zeros(n, bool)
    stack = [("root", np.arange(n, dtype=np.int64))]
    term = np.full(n, -1, np.int32)
    while stack:
        nid, idx = stack.pop()
        nd = nodes[nid]
        d = int(nd["depth"])
        node_at[idx, d] = nidx[nid]
        if nd.get("split"):
            o = bank["out"][idx, d]
            for b, name in enumerate(("like", "dislike", "noanswer")):
                sel = idx[o == b]
                if not len(sel):
                    continue
                child = f"{nid}.{name}"
                if child in nodes:
                    stack.append((child, sel))
                else:
                    # a branch this tree's 10,000-user cohort never populated: `tree_route` gives
                    # these users the parent's `path + tail`. No node, hence no state, hence one-hot.
                    unfitted[sel] = True
                    term[sel] = nidx[nid]
        else:
            term[idx] = nidx[nid]
    return node_at, term, unfitted, node_ids, nidx


def verify_labels(nodes, node_ids, node_at, term, unfitted, bank, pool_index):
    """Assert the descent reproduces the banked labels at EVERY (user, position)."""
    n = bank["n"]
    exp = np.full((n, TOTAL_L), -1, np.int64)
    for ti, nid in enumerate(node_ids):
        m = term == ti
        if not m.any():
            continue
        nd = nodes[nid]
        own_fit = [tuple(q) for q in (nd.get("path") or [])] + \
                  [tuple(q) for q in (nd.get("steps") or [])]
        own_unf = [tuple(q) for q in (nd.get("path") or [])] + \
                  [tuple(q) for q in (nd.get("tail") or nd.get("steps") or [])]
        for arr, which in ((own_fit, ~unfitted), (own_unf, unfitted)):
            mm = m & which
            if not mm.any():
                continue
            assert len(arr) >= TOTAL_L, f"{nid}: own sequence is {len(arr)} < {TOTAL_L}"
            assert len(set(arr[:TOTAL_L])) == TOTAL_L, f"{nid}: repeats a question"
            exp[mm] = np.asarray([pool_index[q] for q in arr[:TOTAL_L]], np.int64)[None, :]
    assert (exp >= 0).all(), "some user was never assigned a terminal node"
    bad = int((exp != bank["lab"].astype(np.int64)).sum())
    assert bad == 0, (
        f"the re-derived descent disagrees with the banked labels at {bad:,} of "
        f"{n * TOTAL_L:,} (user, position) cells -- STOP, this is a harness/tree mismatch")
    return {"n_cells_checked": int(n * TOTAL_L), "n_mismatch": 0,
            "n_users_unfitted_branch": int(unfitted.sum()),
            "unfitted_rate": float(unfitted.mean())}


def load_sweeps(nodes, pool_index):
    """The 72 FULL exact whole-pool gain tables, with the scoping's contract check re-run as a gate."""
    st, chk = [], []
    for p in sorted(glob.glob(os.path.join(SWEEPS, "node_*.npz"))):
        nid = os.path.basename(p)[5:-4]
        assert nid in nodes, f"sweep {nid} has no node"
        nd = nodes[nid]
        assert nd.get("splitter") is not None, f"{nid}: sweep on an unsearched node"
        z = np.load(p, allow_pickle=True)
        cand = np.asarray(z["cand"], np.int64)
        gain = np.asarray(z["gain"], np.float64)
        prefix = [tuple(int(x) for x in q) for q in np.asarray(z["prefix"], np.int64).reshape(-1, 2)]
        assert prefix == [tuple(int(x) for x in q) for q in nd["path"]], f"{nid}: sweep prefix != path"
        assert len(cand) == 1000 - len(prefix), f"{nid}: {len(cand)} candidates, not a whole pool"
        j = int(np.argmax(gain))
        sp = (int(nd["splitter"][0]), int(nd["splitter"][1]))
        assert tuple(cand[j]) == sp, f"{nid}: argmax(gain) {tuple(cand[j])} != splitter {sp}"
        dg = abs(float(gain[j]) - float(nd["splitter_gain"]))
        assert dg < 1e-12, f"{nid}: max(gain) differs from splitter_gain by {dg:g}"
        chk.append(dg)
        st.append({"sid": nid, "nid": nid, "depth": int(nd["depth"]),
                   "prefix": prefix,
                   "pool_prefix_idx": [pool_index[q] for q in prefix],
                   "gains": {pool_index[(int(a), int(b))]: float(g)
                             for (a, b), g in zip(cand, gain)},
                   "n_fit_users": int(nd["n_users"]),
                   "n_measured_in_pool": int(len(cand))})
    st.sort(key=lambda s: s["sid"])
    assert len(st) == 72, f"expected 72 searched-node sweeps, got {len(st)}"
    log(f"[sweeps] 72/72 full exact tables; argmax==splitter at 72/72, "
        f"max |max(gain) - splitter_gain| = {max(chk):.3g}")
    return st, float(max(chk))


# ============================================================================ batching
def make_batch(bank, phi_ext, rows, pos, n_pool):
    """Tensors for a batch of (row, position) examples, in m4b's own history layout.

    The cold arm carries no anchor, so history slot 0 is the FIRST ASKED QUESTION and the width is
    TOTAL_L, exactly as `m4b_policy.make_batch` builds it for a bank whose `anchor` is -1.
    """
    bn = len(rows)
    lab, out = bank["lab"], bank["out"]
    z = np.ascontiguousarray(bank["z"][rows, pos]).astype(np.float32)
    hq = np.zeros((bn, TOTAL_L), np.int64)
    ho = np.zeros((bn, TOTAL_L), np.int64)
    hm = np.zeros((bn, TOTAL_L), np.float32)
    for j in range(TOTAL_L - 1):
        m = pos > j
        if not m.any():
            break
        hq[m, j] = lab[rows[m], j]
        ho[m, j] = out[rows[m], j]
        hm[m, j] = 1.0
    asked = np.zeros((bn, n_pool), bool)
    fl = hm.ravel() > 0
    asked[np.repeat(np.arange(bn), TOTAL_L)[fl], hq.ravel()[fl]] = True
    y = lab[rows, pos].astype(np.int64)
    asked[np.arange(bn), y] = False        # the target is never masked out of its own softmax
    pf = np.stack([pos, TOTAL_L - pos], 1).astype(np.float32)
    return (torch.from_numpy(z), torch.from_numpy(phi_ext[hq]), torch.from_numpy(ho),
            torch.from_numpy(hm), torch.from_numpy(pf), torch.from_numpy(asked),
            torch.from_numpy(y))


def emit_u(pol, bank, phi_ext, rows, pos, n_pool, batch=2048):
    """u(h) for a set of (row, position) states, through the policy's OWN modules."""
    outs = []
    with torch.no_grad():
        for s in range(0, len(rows), batch):
            sl = slice(s, min(s + batch, len(rows)))
            z, hp, ho, hm, pf, _am, _y = make_batch(bank, phi_ext, rows[sl], pos[sl], n_pool)
            outs.append(pol.head.q(pol.enc(z, hp, ho, hm, pf)).numpy())
    return np.concatenate(outs).astype(np.float32)


def snap_argmax(pol, bank, phi_ext, rows, pos, n_pool, batch=2048):
    """The policy's own pick over the 1,000-candidate pool at a set of states."""
    phi_c = torch.from_numpy(phi_ext[:n_pool])
    outs = []
    with torch.no_grad():
        for s in range(0, len(rows), batch):
            sl = slice(s, min(s + batch, len(rows)))
            # `make_batch`'s asked mask IS the inference mask: a question is asked at most once, so
            # the target at position p is never in the 0..p-1 history and its re-opening is a no-op.
            z, hp, ho, hm, pf, am, _y = make_batch(bank, phi_ext, rows[sl], pos[sl], n_pool)
            outs.append(pol(z, hp, ho, hm, pf, phi_c, am).argmax(1).numpy())
    return np.concatenate(outs)


def top1(pol, bank, phi_ext, rows, pos, n_pool, batch=2048):
    phi_c = torch.from_numpy(phi_ext[:n_pool])
    hit = 0
    with torch.no_grad():
        for s in range(0, len(rows), batch):
            sl = slice(s, min(s + batch, len(rows)))
            z, hp, ho, hm, pf, am, y = make_batch(bank, phi_ext, rows[sl], pos[sl], n_pool)
            hit += int((pol(z, hp, ho, hm, pf, phi_c, am).argmax(1) == y).sum())
    return hit / max(len(rows), 1)


# ============================================================================ the fine-tune
def finetune(pol, bank, phi_ext, tr_rows, tr_pos, tr_state, Tt, mu, n_pool, args, hook):
    """Cosine-to-blend at blend states + the incumbent distillation CE elsewhere + the scale term.

    `tr_state[j]` is the blend-state index of example j, or -1 for an ordinary (replay) state.
    """
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    for p in pol.parameters():
        p.requires_grad_(False)
    train = list(pol.enc.parameters()) + list(pol.head.q.parameters())
    for p in train:                    # head.k / head.b stay FROZEN -- see the module docstring
        p.requires_grad_(True)
    opt = torch.optim.AdamW(train, lr=args.lr, weight_decay=1e-4)
    phi_c = torch.from_numpy(phi_ext[:n_pool])
    Tn = torch.from_numpy((Tt / (np.linalg.norm(Tt, axis=1, keepdims=True) + 1e-30)
                           ).astype(np.float32))
    mu_t = torch.from_numpy(mu.astype(np.float32))
    Tmr = torch.from_numpy(np.linalg.norm(Tt - mu[None, :], axis=1).astype(np.float32))
    rng = np.random.default_rng(args.seed)
    N = len(tr_rows)
    hist = []
    with torch.no_grad():                  # head.k / head.b are frozen: their bank is a CONSTANT
        kc = pol.head.k(phi_c).clone()
        bc = pol.head.b(phi_c).squeeze(-1).clone()
    for ep in range(1, args.epochs + 1):
        perm = rng.permutation(N)
        t0 = time.time()
        tot = cos_s = ce_s = sc_s = 0.0
        n_seen = n_bl = n_ce = 0
        for s in range(0, N, args.batch):
            e = perm[s:s + args.batch]
            r, p, st = tr_rows[e], tr_pos[e], tr_state[e]
            z, hp, ho, hm, pf, am, y = make_batch(bank, phi_ext, r, p, n_pool)
            h = pol.enc(z, hp, ho, hm, pf)
            u = pol.head.q(h)
            zh = torch.cat([u, torch.ones(len(e), 1)], 1)
            bl = torch.from_numpy(np.flatnonzero(st >= 0))
            ce = torch.from_numpy(np.flatnonzero(st < 0))
            terms = []
            if len(bl):
                si = torch.from_numpy(st[st >= 0].astype(np.int64))
                zb = zh[bl]
                c = (torch.nn.functional.normalize(zb, dim=1) * Tn[si]).sum(1)
                ratio = (zb - mu_t[None, :]).norm(dim=1) / Tmr[si].clamp(min=1e-30)
                lb = args.alpha * (1.0 - c) + args.lam * (ratio - 1.0) ** 2
                terms.append(lb.sum())
                cos_s += float(c.sum()); sc_s += float(((ratio - 1.0) ** 2).sum())
                n_bl += len(bl)
            if len(ce):
                lg = (u[ce] @ kc.T + bc[None, :]).masked_fill(am[ce], float("-inf"))
                lc = torch.nn.functional.cross_entropy(lg, y[ce], reduction="sum")
                terms.append(lc)
                ce_s += float(lc); n_ce += len(ce)
            loss = sum(terms) / len(e)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss) * len(e); n_seen += len(e)
            if (s // args.batch) % 400 == 399:
                log(f"    ep{ep} {n_seen:,}/{N:,} loss {tot / n_seen:.5f} "
                    f"cos {cos_s / max(n_bl, 1):.5f} ce {ce_s / max(n_ce, 1):.4f} "
                    f"({(time.time() - t0) / 60:.1f}m)")
        rec = {"epoch": ep, "loss": tot / n_seen, "mean_cos_train_blend": cos_s / max(n_bl, 1),
               "mean_ce_replay": ce_s / max(n_ce, 1),
               "mean_scale_penalty": sc_s / max(n_bl, 1),
               "n_blend_examples": int(n_bl), "n_replay_examples": int(n_ce),
               "minutes": round((time.time() - t0) / 60, 2)}
        hist.append(rec)
        log(f"  [ep{ep}] loss {rec['loss']:.5f} | train-blend cos {rec['mean_cos_train_blend']:.5f} "
            f"| replay CE {rec['mean_ce_replay']:.4f} | scale pen "
            f"{rec['mean_scale_penalty']:.4f} | {rec['minutes']:.1f}m")
        hook(ep, hist)
    for p in pol.parameters():
        p.requires_grad_(False)
    return hist


# ============================================================================ readings
def per_state_emission(pol, bank, phi_ext, states, srows, spos, n_pool, Tt, mu):
    """Per-state emitted vector (cohort mean of z_hat) and per-user cosine to the state's target."""
    n = len(states)
    zbar = np.zeros((n, Tt.shape[1]), np.float32)
    ubar = np.zeros((n, Tt.shape[1] - 1), np.float32)
    cos_user_mean = np.zeros(n); disp = np.zeros(n); ratio_med = np.zeros(n)
    per_user = {}
    Tn = Tt / (np.linalg.norm(Tt, axis=1, keepdims=True) + 1e-30)
    Tmr = np.linalg.norm(Tt - mu[None, :], axis=1)
    for i in range(n):
        U = emit_u(pol, bank, phi_ext, srows[i], spos[i], n_pool)
        Z = np.concatenate([U, np.ones((U.shape[0], 1), np.float32)], 1)
        Zn = Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-30)
        cu = Zn @ Tn[i]
        per_user[i] = cu.astype(np.float32)
        cos_user_mean[i] = float(cu.mean())
        zbar[i] = Z.mean(0); ubar[i] = U.mean(0)
        disp[i] = float((Zn @ (zbar[i] / (np.linalg.norm(zbar[i]) + 1e-30))).mean())
        ratio_med[i] = float(np.median(np.linalg.norm(Z - mu[None, :], axis=1) / max(Tmr[i], 1e-30)))
    return {"zbar": zbar, "ubar": ubar, "cos_own": cos_user_mean, "within_state": disp,
            "norm_ratio_median": ratio_med, "per_user": per_user}


def cos_to(Z, T):
    Zn = Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-30)
    Tn = T / (np.linalg.norm(T, axis=1, keepdims=True) + 1e-30)
    return (Zn * Tn).sum(1)


def reading1(ev, ev0, Tt, held, rng):
    """GENERALISATION. At held-out states: cos(z_hat, own target) against (a) a seeded within-split
    random OTHER held-out state's target and (b) the warm start's own cos at the same states."""
    h = np.asarray(held, np.int64)
    k = len(h)
    assert k >= 2
    other = np.empty(k, np.int64)
    for i in range(k):
        c = h[rng.integers(0, k)]
        while c == h[i]:
            c = h[rng.integers(0, k)]
        other[i] = c
    own_state = cos_to(ev["zbar"][h], Tt[h])
    oth_state = cos_to(ev["zbar"][h], Tt[other])
    warm_state = cos_to(ev0["zbar"][h], Tt[h])
    # user-row unit: every user-row of every held-out state, paired against the WARM START on the
    # same row. (The random-other-state arm stays state-level: it compares two per-state vectors.)
    ou = np.concatenate([ev["per_user"][int(i)] for i in h])
    wu = np.concatenate([ev0["per_user"][int(i)] for i in h])
    return {
        "n_held_out_states": int(k),
        "unit": "state (paired bootstrap over the held-out states); user-row arm reported alongside",
        "cos_own_target": dist(own_state),
        "cos_random_other_held_out_state_target": dist(oth_state),
        "cos_warm_start_own_target": dist(warm_state),
        "a_own_minus_other": paired_bootstrap(own_state - oth_state),
        "b_own_minus_warm_start": paired_bootstrap(own_state - warm_state),
        "n_states_own_beats_other": int((own_state > oth_state).sum()),
        "n_states_own_beats_warm": int((own_state > warm_state).sum()),
        "user_row_unit": {
            "n_user_rows": int(len(ou)),
            "cos_own": dist(ou), "cos_warm": dist(wu),
            "b_own_minus_warm_start": paired_bootstrap(ou - wu)},
        "seeded_other_state_map": {int(a): int(b) for a, b in zip(h, other)}}


# ============================================================================ the run
def run(args):
    from strategy_channel_suite import build_greedy_env, contract_selftest, genv_mean
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
    log(f"[warm] m7dj_policy.pt epoch={blob['epoch']} ext={len(blob['ext'])} "
        f"anchor_encoding={blob['anchor_encoding']} sha256 {warm_sha[:12]}...")

    # ---- the environment, the pool, the canonical TRAIN snap --------------------------------
    E = build_greedy_env(args.snapshot, "train", args.concept_ckpt, "paperC/c1final",
                         cohort_slice=(0, args.snap_users))
    contract_selftest(E, E["rows"])
    rows_env = E["rows"]
    assert len(rows_env) == args.snap_users
    f0, t0_ = genv_mean(E, [], rows_env)
    assert f0 == TRAIN_M0_FULL, (
        f"CANONICAL SNAP FAILED: train m=0 full {f0!r}, banked {TRAIN_M0_FULL!r}. This is the TRAIN "
        f"intercept for this split -- NOT the validation 0.15686538118354684.")
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
    log(f"[bank] {Ebank.shape[0]:,} x {Ebank.shape[1]} descriptors; pool2full exact "
        f"(max|diff| {map_err:g}); bank mean direction ||mu|| {np.linalg.norm(mu):.4f}")

    # ---- the tree, its sweeps, and the bank --------------------------------------------------
    tree = json.load(open(TREE_JSON))
    assert tree["split"] == "train" and int(tree["total_L"]) == TOTAL_L
    assert tree["test_touched"] is False
    nodes = tree["nodes"]
    log(f"[tree] coldtree_v2_train10000: {len(nodes)} nodes, depth {tree['depth_reached']}, "
        f"L={tree['total_L']}, min_leaf {tree['min_leaf']}, fitted on {tree['n_cohort']:,} TRAIN users")
    states, gain_chk = load_sweeps(nodes, pool_index)
    bank = load_cold_bank()
    node_at, term, unfitted, node_ids, nidx = descend(nodes, bank)
    contract = verify_labels(nodes, node_ids, node_at, term, unfitted, bank, pool_index)
    log(f"[contract] the re-derived descent reproduces ALL {contract['n_cells_checked']:,} banked "
        f"labels; {contract['n_users_unfitted_branch']} users "
        f"({contract['unfitted_rate']:.5%}) sit in an unfitted branch and stay one-hot")

    # blend-state index per (user, position): the searched node whose OWN depth this position is
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
    log(f"[states] {n_blend_cells:,} of {bank['n'] * TOTAL_L:,} bank cells "
        f"({n_blend_cells / (bank['n'] * TOTAL_L):.1%}) are one of the 72 searched node states; "
        f"bank occupancy min {occ.min():,} median {int(np.median(occ)):,} max {occ.max():,}")
    assert occ.min() > 0, "a searched node state has no bank users"

    # the (row, position) lists per state, for the emission readings
    srows, spos = [], []
    for i in range(len(states)):
        r, p = np.nonzero(blend == i)
        srows.append(r.astype(np.int64)); spos.append(p.astype(np.int64))

    # ---- the blend targets -------------------------------------------------------------------
    Tt, TKt, meta = build_targets(states, Epool, Kpool, pool_index, args.m)
    cl = ceiling(Tt)
    G0 = {k: np.asarray(v) for k, v in state_geometry(
        Tt, states, Ebank, pool2full, chunk=args.chunk, label="target").items()}
    rng = np.random.default_rng(SEED)
    R = rng.standard_normal(Tt.shape).astype(np.float32)
    R *= (np.linalg.norm(Tt, axis=1) / np.linalg.norm(R, axis=1))[:, None]
    GR = {k: np.asarray(v) for k, v in state_geometry(
        R, states, Ebank, pool2full, chunk=args.chunk, label="rand").items()}
    w1 = np.asarray([m["w1"] for m in meta])
    stage0 = {"m": args.m, "n_states": len(states),
              "target_cos1": dist(G0["cos1"]), "target_margin": dist(G0["m"]),
              "top1_weight_w1": dist(w1),
              "n_states_w1_above_0.5": int((w1 > 0.5).sum()),
              "analytic_ceiling": dist(cl), "frac_ceiling_ge_0.90": float((cl >= 0.90).mean()),
              "target_snap_equals_top_gain_candidate": float(np.mean(
                  [G0["snap"][i] == pool2full[meta[i]["top_m_pool_idx"][0]]
                   for i in range(len(states))])),
              "random_floor": {"cos1": dist(GR["cos1"]), "m": dist(GR["m"])},
              "premise_gate_fires": bool(float(np.median(G0["cos1"])) > 0.99)}
    log(f"[stage0] median target-cos1 {np.median(G0['cos1']):.5f} | median margin "
        f"{np.median(G0['m']):.5f} | median w1 {np.median(w1):.4f} | premise gate fires "
        f"{stage0['premise_gate_fires']}")

    out = {"analysis": "paperC_C1_final", "design_sheet": "docs/design/DS_C1_FINAL.md",
           "citable": True, "test_touched": False, "split": "train_states",
           "teacher": os.path.basename(TREE_JSON),
           "warm_start": {"path": os.path.relpath(WARM, _ROOT), "sha256": warm_sha,
                          "epoch": int(blob["epoch"]), "n_ext": len(ext),
                          "anchor_encoding": blob["anchor_encoding"],
                          "dev_top1_banked": blob["dev_top1"]},
           "git": GB,
           "hyperparameters_not_tuned": {"epochs": args.epochs, "batch": args.batch, "lr": args.lr,
                                         "alpha": args.alpha, "lambda": args.lam, "m": args.m,
                                         "optimiser": "AdamW wd=1e-4", "seed": args.seed,
                                         "frozen": ["head.k", "head.b"],
                                         "note": "chosen once, before the run, never swept"},
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
               "n_users": bank["n"], "n_slices": bank["n_slices"],
               "n_cells": int(bank["n"] * TOTAL_L), "n_blend_cells": n_blend_cells,
               "bank_occupancy_per_searched_state": dist(occ)},
           "target": {"kind": "MIXED: gain-weighted top-5 blend at the 72 searched node states; the "
                              "teacher's ONE-HOT at every tail state and every below-floor "
                              "inheritor state. Only the 72 are blend states.",
                      "space": "homogeneous [k(phi), b(phi)] (C0's primary), over the 1,000-candidate"
                               " pool; mean-removal uses the FULL 19,390-entry bank mean",
                      "mu_norm": float(np.linalg.norm(mu))},
           "ingredient_stability": {
               "per_user_gain_sidecars_exist_for_this_tree": False,
               "checked": "coldtree_v2_train10000_per_user.npz holds per-user NDCG by leaf/budget/"
                          "arm, NOT per-user per-candidate gains; the sweeps bank an AGGREGATE gain "
                          "per candidate. Split-half direction stability of the blend targets is "
                          "therefore NOT computable on this tree.",
               "carried_prior_split_half_meanremoved_cos": PILOT_SPLIT_HALF_MEANREMOVED,
               "prior_source": "C1_PILOT.md repaired reading 2 (35 states, the PILOT's tree) -- a "
                               "LABELLED PRIOR, not a measurement on this tree",
               "node_cohorts": dist([s["n_fit_users"] for s in states]),
               "n_nodes_below_500_fit_users": int(sum(s["n_fit_users"] < 500 for s in states)),
               "binding_wording": "we train toward a MEASURED direction; we do NOT claim the five "
                                  "ingredients are the right five"},
           "stage0_premise": stage0,
           "cells": {}}
    p_json = os.path.join(PAPERC, f"c1_final{args.out_tag}.json")
    p_npz = os.path.join(PAPERC, f"c1_final{args.out_tag}_per_state.npz")
    npz = {"target_m5": Tt, "target_cos1": G0["cos1"], "target_margin": G0["m"],
           "target_snap": G0["snap"], "target_ceiling": cl, "bank_occupancy": occ,
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

    # ---- the warm start's own behaviour, measured once ---------------------------------------
    rows_all, pos_all = np.nonzero(bank["valid"])
    rows_all = rows_all.astype(np.int64); pos_all = pos_all.astype(np.int64)
    st_all = blend[rows_all, pos_all]
    dev = bank["uid"][rows_all] >= args.dev_lo
    _r = np.random.default_rng(SEED + 1)
    dsel = np.flatnonzero(dev)
    if len(dsel) > 40000:
        dsel = dsel[_r.choice(len(dsel), 40000, replace=False)]
    warm_dev = top1(pol0, bank, phi_ext, rows_all[dsel], pos_all[dsel], n_pool)
    log(f"[contract] warm start's teacher-forced top-1 on {len(dsel):,} held-out COLD dev states "
        f"({args.dev_lo:,}+): {warm_dev:.5f} (m7dj banked cold dev top-1 "
        f"{blob['dev_top1']['cold']:.5f})")
    assert warm_dev > 0.99, (
        f"the rebuilt pipeline scores the warm start at {warm_dev:.5f} on its own cold dev states, "
        f"far below the banked {blob['dev_top1']['cold']:.5f} -- the phi/pool/batch reconstruction "
        f"is wrong, STOP")
    out["gates"]["warm_start_cold_dev_top1_rebuilt"] = {
        "got": warm_dev, "m7dj_banked_cold_dev_top1": float(blob["dev_top1"]["cold"]),
        "n_states": int(len(dsel)),
        "note": "not bit-comparable -- m7dj's number came off a 40k sample of the SEVEN-bank dev "
                "index, this is a 40k sample of the COLD dev states only"}

    # gate-3 population: 1,000 ORDINARY (non-blend) states, drawn once, shared by every cell
    grng = np.random.default_rng(SEED)
    ord_pool = np.flatnonzero(st_all < 0)
    gsel = ord_pool[grng.choice(len(ord_pool), args.n_gate, replace=False)]
    g_rows, g_pos = rows_all[gsel], pos_all[gsel]
    a0 = snap_argmax(pol0, bank, phi_ext, g_rows, g_pos, n_pool)
    n_distinct_warm = int(len(np.unique(a0)))
    log(f"[gate3] warm start emits {n_distinct_warm} distinct questions over the "
        f"{args.n_gate:,} ordinary non-blend states")
    out["gate3_population"] = {"n_states": int(args.n_gate),
                               "n_distinct_questions_warm_start": n_distinct_warm,
                               "bar_distinct": 0.75 * n_distinct_warm, "bar_agreement": 0.90}
    npz["gate3_rows"] = g_rows; npz["gate3_pos"] = g_pos; npz["gate3_warm_argmax"] = a0

    ev0 = per_state_emission(pol0, bank, phi_ext, states, srows, spos, n_pool, Tt, mu)
    out["warm_start_emission"] = {
        "cos_to_target": dist(ev0["cos_own"]),
        "norm_ratio_median_over_states": dist(ev0["norm_ratio_median"]),
        "within_state_cos_to_mean": dist(ev0["within_state"])}
    npz["warm_zbar"] = ev0["zbar"]; npz["warm_cos_own"] = ev0["cos_own"]
    log(f"[warm] emission at the 72 states: median cos to target "
        f"{np.median(ev0['cos_own']):+.5f}, median mean-removed norm ratio "
        f"{np.median(ev0['norm_ratio_median']):.4f}")
    flush()

    # ---- the two registered cells ------------------------------------------------------------
    depth = np.asarray([s["depth"] for s in states])
    sids = [s["sid"] for s in states]
    SPLITS = {
        "primary_root_like_subtree": np.asarray(
            [i for i, s in enumerate(sids) if s == "root.like" or s.startswith("root.like.")]),
        "secondary_depth_ge_5": np.flatnonzero(depth >= 5)}
    assert len(SPLITS["primary_root_like_subtree"]) == 15
    assert len(SPLITS["secondary_depth_ge_5"]) == 30

    for cell, held in SPLITS.items():
        log("#" * 100)
        log(f"# CELL {cell}: {len(held)} states held out, {len(states) - len(held)} trained")
        heldset = set(int(x) for x in held)
        trained = np.asarray([i for i in range(len(states)) if i not in heldset])
        keep = ~np.isin(st_all, held)          # held-out blend states receive NO loss at all
        tr_rows, tr_pos, tr_state = rows_all[keep], pos_all[keep], st_all[keep]
        log(f"  [data] {len(tr_rows):,} training examples "
            f"({int((tr_state >= 0).sum()):,} blend / {int((tr_state < 0).sum()):,} replay); "
            f"{int((~keep).sum()):,} held-out-state examples EXCLUDED from the loss")
        pol = copy.deepcopy(pol0)
        cellrec = {"n_held_out_states": int(len(held)), "n_trained_states": int(len(trained)),
                   "held_out_sids": [sids[int(i)] for i in held],
                   "n_train_examples": int(len(tr_rows)),
                   "n_blend_examples": int((tr_state >= 0).sum()),
                   "n_replay_examples": int((tr_state < 0).sum()),
                   "n_examples_excluded_heldout": int((~keep).sum()),
                   "heldout_states_receive": "no loss (neither the blend target nor the incumbent CE)"}
        out["cells"][cell] = cellrec
        flush()

        def hook(ep, hist, _c=cellrec):
            _c["epochs"] = hist
            flush()

        hist = finetune(pol, bank, phi_ext, tr_rows, tr_pos, tr_state, Tt, mu, n_pool, args, hook)
        cellrec["epochs"] = hist

        ev = per_state_emission(pol, bank, phi_ext, states, srows, spos, n_pool, Tt, mu)
        npz[f"zbar_{cell}"] = ev["zbar"]; npz[f"cos_own_{cell}"] = ev["cos_own"]
        npz[f"norm_ratio_{cell}"] = ev["norm_ratio_median"]

        # READING 1 -- generalisation (the headline)
        cellrec["reading1_generalisation"] = reading1(ev, ev0, Tt, held,
                                                      np.random.default_rng(SEED))
        r1 = cellrec["reading1_generalisation"]
        log(f"  [reading1] held-out cos own {r1['cos_own_target']['median']:.5f} | vs random other "
            f"{r1['cos_random_other_held_out_state_target']['median']:.5f} "
            f"(diff {r1['a_own_minus_other']['mean']:+.5f} "
            f"[{r1['a_own_minus_other']['lo']:+.5f}, {r1['a_own_minus_other']['hi']:+.5f}] "
            f"excl0={r1['a_own_minus_other']['excludes_zero']}) | vs warm start "
            f"{r1['cos_warm_start_own_target']['median']:.5f} "
            f"(diff {r1['b_own_minus_warm_start']['mean']:+.5f} "
            f"[{r1['b_own_minus_warm_start']['lo']:+.5f}, {r1['b_own_minus_warm_start']['hi']:+.5f}] "
            f"excl0={r1['b_own_minus_warm_start']['excludes_zero']})")

        # READING 2 -- fit at TRAINED blend states (reported, not a gate)
        ct, c0t = ev["cos_own"][trained], ev0["cos_own"][trained]
        cellrec["reading2_fit_trained_states"] = {
            "unit": "state", "n_states": int(len(trained)),
            "cos_own_target": dist(ct), "warm_start_cos": dist(c0t),
            "minus_warm_start": paired_bootstrap(ct - c0t),
            "frac_states_cos_ge_0.90": float((ct >= 0.90).mean()),
            "within_state_cos_to_mean": dist(ev["within_state"][trained])}
        log(f"  [reading2] trained-state fit: median cos "
            f"{cellrec['reading2_fit_trained_states']['cos_own_target']['median']:.5f}, "
            f"{cellrec['reading2_fit_trained_states']['frac_states_cos_ge_0.90']:.3f} of states "
            f">= 0.90")

        # READING 3 -- NO-COLLAPSE GATE (hard, blocking)
        a1 = snap_argmax(pol, bank, phi_ext, g_rows, g_pos, n_pool)
        nd = int(len(np.unique(a1)))
        agree = float((a0 == a1).mean())
        g3 = {"n_states": int(args.n_gate),
              "n_distinct_questions_warm_start": n_distinct_warm,
              "n_distinct_questions_finetuned": nd,
              "distinct_ratio": nd / max(n_distinct_warm, 1),
              "distinct_bar": 0.75, "distinct_pass": bool(nd >= 0.75 * n_distinct_warm),
              "snap_agreement_vs_warm_start": agree,
              "snap_agreement_ci": wilson(int((a0 == a1).sum()), int(len(a0))),
              "agreement_bar": 0.90, "agreement_pass": bool(agree >= 0.90)}
        g3["PASS"] = bool(g3["distinct_pass"] and g3["agreement_pass"])
        cellrec["reading3_no_collapse_GATE"] = g3
        npz[f"gate3_argmax_{cell}"] = a1
        log(f"  [GATE 3 no-collapse] distinct {nd}/{n_distinct_warm} "
            f"(ratio {g3['distinct_ratio']:.3f}, bar 0.75) {'PASS' if g3['distinct_pass'] else 'FAIL'}"
            f" | snap agreement {agree:.4f} (bar 0.90) "
            f"{'PASS' if g3['agreement_pass'] else 'FAIL'} => "
            f"{'PASS' if g3['PASS'] else '*** FAIL ***'}")

        # READING 4 -- RENDERABILITY GATE
        med_all = float(np.median(ev["norm_ratio_median"]))
        g4 = {"median_norm_ratio_all_72": med_all,
              "median_norm_ratio_trained": float(np.median(ev["norm_ratio_median"][trained])),
              "median_norm_ratio_held_out": float(np.median(ev["norm_ratio_median"][held])),
              "warm_start_median": float(np.median(ev0["norm_ratio_median"])),
              "band": [0.5, 2.0], "distribution": dist(ev["norm_ratio_median"])}
        g4["PASS"] = bool(0.5 <= med_all <= 2.0)
        cellrec["reading4_renderability_GATE"] = g4
        log(f"  [GATE 4 renderability] median mean-removed norm ratio {med_all:.4f} "
            f"(band [0.5, 2.0]) {'PASS' if g4['PASS'] else '*** FAIL ***'} "
            f"| trained {g4['median_norm_ratio_trained']:.4f} held-out "
            f"{g4['median_norm_ratio_held_out']:.4f} | warm start {g4['warm_start_median']:.4f}")

        # READING 5 -- off-grid geometry of the emissions
        Gg = {k: np.asarray(v) for k, v in state_geometry(
            ev["zbar"], states, Ebank, pool2full, chunk=args.chunk, label=cell).items()}
        Zmr = ev["zbar"] - mu[None, :].astype(np.float32)
        Bmr = Ebank - mu[None, :].astype(np.float32)
        cos1_mr = np.asarray([float(Zmr[i] @ Bmr[Gg["snap"][i]] /
                                    (np.linalg.norm(Zmr[i]) * np.linalg.norm(Bmr[Gg["snap"][i]])
                                     + 1e-30)) for i in range(len(states))])
        Tmr_v = Tt - mu[None, :].astype(np.float32)
        tcos1_mr = np.asarray([float(Tmr_v[i] @ Bmr[G0["snap"][i]] /
                                     (np.linalg.norm(Tmr_v[i]) * np.linalg.norm(Bmr[G0["snap"][i]])
                                      + 1e-30)) for i in range(len(states))])
        cellrec["reading5_offgrid_geometry"] = {
            "emitted": geom_summary(Gg), "target": geom_summary(G0),
            "random_floor": geom_summary(GR),
            "emitted_cos1_mean_removed": dist(cos1_mr),
            "target_cos1_mean_removed": dist(tcos1_mr),
            "abs_cos1_diff_emitted_vs_target": dist(np.abs(Gg["cos1"] - G0["cos1"])),
            "same_nearest_bank_question_as_target": float((Gg["snap"] == G0["snap"]).mean()),
            "held_out_only": {"emitted_cos1": dist(Gg["cos1"][held]),
                              "emitted_margin": dist(Gg["m"][held]),
                              "same_nearest_as_target": float(
                                  (Gg["snap"][held] == G0["snap"][held]).mean())}}
        npz[f"emit_cos1_{cell}"] = Gg["cos1"]; npz[f"emit_margin_{cell}"] = Gg["m"]
        npz[f"emit_snap_{cell}"] = Gg["snap"]
        log(f"  [reading5] emitted median cos1 {np.median(Gg['cos1']):.5f} (target "
            f"{np.median(G0['cos1']):.5f}, random floor {np.median(GR['cos1']):.5f}) | margin "
            f"{np.median(Gg['m']):.5f} | mean-removed cos1 {np.median(cos1_mr):.5f} "
            f"(target {np.median(tcos1_mr):.5f})")

        cellrec["VERDICT"] = ("BLOCKING GATE FAILED -- the run fails regardless of readings 1-2"
                              if not (g3["PASS"] and g4["PASS"]) else
                              "both blocking gates pass")
        flush()

    out["headline"] = {
        "blocking_gates": {c: {"gate3_no_collapse": out["cells"][c]["reading3_no_collapse_GATE"]["PASS"],
                               "gate4_renderability": out["cells"][c]["reading4_renderability_GATE"]["PASS"]}
                           for c in out["cells"]},
        "all_blocking_gates_pass": bool(all(
            out["cells"][c]["reading3_no_collapse_GATE"]["PASS"] and
            out["cells"][c]["reading4_renderability_GATE"]["PASS"] for c in out["cells"]))}
    flush()
    log(f"[done] wrote {p_json} in {out['minutes']:.1f} minutes")
    return out


def main():
    ap = argparse.ArgumentParser(description="Paper C stage C1-final -- see docs/design/DS_C1_FINAL.md")
    ap.add_argument("--m", type=int, default=5, help="blend width (registered m=5)")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--lam", type=float, default=0.1, help="the registered scale term's weight")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--n_gate", type=int, default=1000, help="gate-3 ordinary states")
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
    assert args.i_have_author_goahead, "REFUSING TO RUN without --i_have_author_goahead"
    from strategy_channel_suite import CERT_TOWER, CERT_CONC
    args.snapshot = args.snapshot or CERT_TOWER
    args.concept_ckpt = args.concept_ckpt or CERT_CONC
    log("#" * 100)
    log("# PAPER C / C1-FINAL -- blend targets at the 72 searched node states of "
        "coldtree_v2_train10000, with replay, a scale term and held-out STATES. TRAIN-SIDE ONLY; "
        "the 10,000 canonical TEST users are not built, not loaded, not touched.")
    log("#" * 100)
    return run(args)


if __name__ == "__main__":
    main()
