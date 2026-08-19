r"""M4b STAGE 2 -- ONE POLICY OVER THE SEVEN-TREE FOREST, distilled on TRAIN users.

Design sheet: `docs/design/PAPER_B_POLICY_DESIGN.md` (PART A binding: the architecture is
`policy_net.Policy`, re-imported and never re-implemented). Committed before any run (HARD RULE 10).

WHAT THIS IS. M4a showed that at an opener state a DEDICATED teacher beats the incumbent cold tree.
Seven trees is not a deployable artifact -- it is seven lookup structures behind a hand-written
router. This distils all seven into ONE parametric policy whose only input is the state, so the
"which tree" decision disappears into the state itself. That is the method claim of the chapter; the
trees are the existence proof.

* THE FALSIFIER, declared before the run: INTERFERENCE. Fidelity to the cold tree and to each of the
six teachers is reported TOGETHER at every checkpoint. If learning the teachers costs the cold tree,
the single-policy claim fails at seven teachers and is reported dead, not retried. (L2a established
one policy can hold the tree plus four STATIC genre orders at 0.999969 fidelity with zero
interference; seven BRANCHING teachers is the real test of that.)

THE FIREWALL, and one deliberate departure from the design sheet
-----------------------------------------------------------------
Training states come from the 140,768 TRAIN users only (D.6b: no NDCG is ever measured on a train
user; the trees are frozen and merely APPLIED). The design sheet says "select on val". This file does
NOT: it selects on a held-out TRAIN slice (`--dev_lo`), because validation is where the arm-C
comparison against the trees is scored, and selecting there would contaminate it. Stricter than the
sheet, in the direction of caution. The 10,000 canonical TEST users are never built, loaded or named
-- there is no code path in this file that can construct a test environment.

THE VOLUNTEERED ANCHOR IS AN ANSWERED QUESTION, NOT A LATENT NUDGE
-------------------------------------------------------------------
(L1, memory `asif-asked-encoding-verified`.) A seeded teacher's anchor is folded into `z` by the
generator. That ALONE is the encoding L1 measured at 0.00% correct branching. The policy must also
see it as an ANSWERED QUESTION: history token + LIKE outcome + masked out of the candidate
vocabulary. The cohort is the anchor's endorsers under the env's SEL rule, so LIKE is its outcome by
construction. Reconstructed here from each bank's `anchor` field -- no regeneration needed.

FALLBACK POSITIONS ARE MASKED OUT OF THE LOSS (author, 2026-08-19)
-------------------------------------------------------------------
Where a user routed outside a tree's fitted support, the banked label is `tree_route`'s continuation
rule, not the teacher's decision. Training on it would fit the fallback instead of the tree, so those
positions carry `valid=False` and never enter the loss. Positions BEFORE the escape are genuine
teacher labels and are kept. Evaluation never masks -- see `tree_route`.
"""
from __future__ import annotations

import os
import sys

_H = os.path.dirname(os.path.abspath(__file__))
_R = os.path.abspath(os.path.join(_H, "..", ".."))
sys.path.insert(0, _H)

import argparse
import json
import time

import numpy as np
import torch
import torch.nn as nn

B = os.path.join(_R, "experiments", "battery")
TRAJ = os.path.join(B, "m4", "traj")
OUT = os.path.join(B, "m4")
SEED = 98765
TOTAL_L = 15
N_TRAIN = 140768
TREES = ["cold", "arthouse", "dialogue", "marvel", "gore", "teenagers", "inspiring"]


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def consolidate(tree):
    """Concatenate a tree's per-slice npz into ONE on-disk float32 memmap for z, plus a small meta
    npz. Done once and reused: the seven z tensors together are ~2.6 GB and must never all be
    resident (HARD RULE 11c -- bound every cache, prefer float32)."""
    zp = os.path.join(TRAJ, f"z_{tree}.f32")
    mp = os.path.join(TRAJ, f"meta_{tree}.npz")
    if os.path.exists(zp) and os.path.exists(mp):
        m = np.load(mp)
        n, d = len(m["uid"]), int(m["d"])
        return {"lab": m["label_idx"], "out": m["outcome"], "valid": m["valid"], "uid": m["uid"],
                "anchor": int(m["anchor"]), "d": d,
                "z": np.memmap(zp, np.float32, "r", shape=(n, TOTAL_L, d))}
    pref = tree + "_"
    parts = sorted((f for f in os.listdir(TRAJ) if f.startswith(pref) and f.endswith(".npz")),
                   key=lambda f: int(f[len(pref):].split("_")[0]))
    assert parts, f"no banked slices for {tree}"
    lab, out, val, uid = [], [], [], []
    anchor, d, n = None, None, 0
    for f in parts:
        lo = int(f[len(pref):].split("_")[0])
        z = np.load(os.path.join(TRAJ, f))
        if len(z["rows"]) == 0:
            continue
        lab.append(z["label_idx"])
        out.append(z["outcome"])
        val.append(z["valid"])
        uid.append(lo + z["rows"].astype(np.int64))
        anchor = int(z["anchor"])
        d = int(z["z"].shape[2])
        n += len(z["rows"])
    lab = np.concatenate(lab)
    out = np.concatenate(out)
    val = np.concatenate(val)
    uid = np.concatenate(uid)
    assert len(np.unique(uid)) == len(uid), f"{tree}: a train user appears twice in the bank"
    mm = np.memmap(zp, np.float32, "w+", shape=(n, TOTAL_L, d))
    at = 0
    for f in parts:
        z = np.load(os.path.join(TRAJ, f))
        k = len(z["rows"])
        if k:
            mm[at:at + k] = z["z"]
            at += k
    assert at == n
    mm.flush()
    del mm
    np.savez_compressed(mp, label_idx=lab, outcome=out, valid=val, uid=uid, anchor=anchor, d=d)
    log(f"  [{tree}] consolidated {n:,} users x {TOTAL_L} positions (anchor {anchor})")
    return consolidate(tree)


def build_index(banks, lo, hi):
    """Flat (tree, row, position) index over VALID positions whose user lies in [lo, hi)."""
    out = []
    for ti, t in enumerate(TREES):
        d = banks[t]
        sel = (d["uid"] >= lo) & (d["uid"] < hi)
        rr, pp = np.nonzero(d["valid"] & sel[:, None])
        out.append(np.stack([np.full(len(rr), ti, np.int32), rr.astype(np.int32),
                             pp.astype(np.int32)], 1))
    return np.concatenate(out)


def make_batch(banks, phi_ext, idx, a_phi, a_pool, n_pool, d_z):
    """Tensors for one batch of (tree, row, position) examples.

    History layout, width TOTAL_L: slot 0 is the SEEDED ANCHOR as an answered question (outcome
    LIKE) for a teacher, and unused for the cold tree; slots 1.. are the questions asked so far. The
    anchor is also masked out of the candidate table -- it is folded evidence, never a question to
    re-ask.
    """
    bn = len(idx)
    z = np.zeros((bn, d_z), np.float32)
    hq = np.zeros((bn, TOTAL_L), np.int64)
    ho = np.zeros((bn, TOTAL_L), np.int64)
    hm = np.zeros((bn, TOTAL_L), np.float32)
    asked = np.zeros((bn, n_pool), bool)
    y = np.zeros(bn, np.int64)
    pos = np.zeros((bn, 2), np.float32)
    for j in range(bn):
        ti, r, p = int(idx[j, 0]), int(idx[j, 1]), int(idx[j, 2])
        t = TREES[ti]
        dd = banks[t]
        z[j] = dd["z"][r, p]
        s = 0
        if dd["anchor"] >= 0:
            hq[j, 0] = a_phi[t]
            ho[j, 0] = 0
            hm[j, 0] = 1.0
            s = 1
            if a_pool[t] >= 0:
                asked[j, a_pool[t]] = True
        for k in range(p):
            q = int(dd["lab"][r, k])
            hq[j, s + k] = q
            ho[j, s + k] = int(dd["out"][r, k])
            hm[j, s + k] = 1.0
            asked[j, q] = True
        y[j] = int(dd["lab"][r, p])
        pos[j] = (p, TOTAL_L - p)
    asked[np.arange(bn), y] = False        # the target is never masked out of its own softmax
    return (torch.from_numpy(z), torch.from_numpy(phi_ext[hq]), torch.from_numpy(ho),
            torch.from_numpy(hm), torch.from_numpy(pos), torch.from_numpy(asked),
            torch.from_numpy(y))


def main():
    ap = argparse.ArgumentParser(description="M4b stage 2 -- one policy over the forest")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--dev_lo", type=int, default=120000, help="train users >= this are the DEV set")
    ap.add_argument("--arch", default="equivariant", choices=["equivariant", "pointer", "flat"])
    ap.add_argument("--state", default="both", choices=["z", "hist", "both"])
    ap.add_argument("--shuffle_control", action="store_true", help="permute labels; must NOT fit")
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--tag", default="m4b")
    ap.add_argument("--i_have_author_goahead", action="store_true")
    a = ap.parse_args()
    assert a.i_have_author_goahead, "REFUSING TO RUN without --i_have_author_goahead"
    torch.set_num_threads(a.threads)
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    rng = np.random.default_rng(SEED)

    from strategy_channel_suite import build_greedy_env, CERT_TOWER, CERT_CONC
    from ndcg_tree import build_pool
    from policy_net import Policy, build_phi

    log("#" * 100)
    log("# M4b STAGE 2 -- ONE POLICY OVER THE SEVEN-TREE FOREST. TRAIN STATES ONLY.")
    log("#" * 100)

    banks = {t: consolidate(t) for t in TREES}
    for t in TREES:
        d = banks[t]
        log(f"[bank] {t:<11} {len(d['uid']):>7,} users - {int(d['valid'].sum()):>9,} valid "
            f"positions - {int((~d['valid']).sum()):>5,} masked - anchor {d['anchor']}")

    # The pool and the descriptors, built ONCE. The pool is cohort-independent (item popularity comes
    # from the full train matrix, concept stats from a global file) -- ASSERTED below, not assumed.
    E = build_greedy_env(CERT_TOWER, "train", CERT_CONC, f"{a.tag}_phi", cohort_slice=(0, 20000))
    pool, _ip, _cp = build_pool(E, 500, 500)
    n_pool = len(pool)
    pidx = {tuple(q): i for i, q in enumerate(pool)}
    for t in TREES:
        mx = int(banks[t]["lab"].max())
        assert mx < n_pool, f"{t}: label index {mx} outside a pool of {n_pool} -- the pool moved"

    # phi over the pool, EXTENDED with any anchor that is not itself a pool candidate, so an anchor
    # can be a history token even when it is masked out of the candidate vocabulary.
    ext = list(pool)
    a_phi, a_pool = {}, {}
    for t in TREES:
        c = banks[t]["anchor"]
        if c < 0:
            a_phi[t], a_pool[t] = 0, -1
            continue
        a_pool[t] = pidx.get((1, c), -1)
        if (1, c) in pidx:
            a_phi[t] = pidx[(1, c)]
        else:
            a_phi[t] = len(ext)
            ext.append((1, c))
    phi_ext, norm = build_phi(E, ext)
    d_phi = int(phi_ext.shape[1])
    log(f"[phi] pool {n_pool} candidates - phi table {len(ext)} rows x {d_phi} "
        f"({len(ext) - n_pool} anchors appended as history-only tokens)")
    log("[anchor] pool membership: " + ", ".join(
        f"{t}={'in' if a_pool[t] >= 0 else 'OUT'}" for t in TREES if banks[t]["anchor"] >= 0))
    del E

    tr_idx = build_index(banks, 0, a.dev_lo)
    dv_idx = build_index(banks, a.dev_lo, N_TRAIN)
    log(f"[split] {len(tr_idx):,} training examples (users < {a.dev_lo:,}) - "
        f"{len(dv_idx):,} dev examples (users >= {a.dev_lo:,}). VALIDATION IS NOT TOUCHED.")

    if a.shuffle_control:
        log("[control] SHUFFLE ARM: labels permuted within each tree. This must NOT fit.")
        for t in TREES:
            lb = banks[t]["lab"]
            banks[t]["lab"] = lb.ravel()[rng.permutation(lb.size)].reshape(lb.shape)

    d_z = int(banks["cold"]["d"])
    pol = Policy(d_z, d_phi, n_pool, arch=a.arch, state=a.state, use_pos=True)
    opt = torch.optim.AdamW(pol.parameters(), lr=a.lr, weight_decay=1e-4)
    log(f"[policy] arch={a.arch} state={a.state} params={pol.n_params():,}")

    phi_cand = torch.from_numpy(phi_ext[:n_pool])

    def fidelity(idx, cap=40000):
        """Teacher-forced top-1 agreement, PER TREE -- the interference falsifier."""
        pol.eval()
        sub = idx if len(idx) <= cap else idx[rng.choice(len(idx), cap, replace=False)]
        hit = {t: [0, 0] for t in TREES}
        with torch.no_grad():
            for s in range(0, len(sub), 2048):
                bb = sub[s:s + 2048]
                z, hp, ho, hm, ps, am, y = make_batch(banks, phi_ext, bb, a_phi, a_pool,
                                                     n_pool, d_z)
                pr = pol(z, hp, ho, hm, ps, phi_cand, am).argmax(1)
                ok = (pr == y).numpy()
                for ti in np.unique(bb[:, 0]):
                    m = bb[:, 0] == ti
                    hit[TREES[ti]][0] += int(ok[m].sum())
                    hit[TREES[ti]][1] += int(m.sum())
        pol.train()
        return {t: (hit[t][0] / hit[t][1] if hit[t][1] else float("nan")) for t in TREES}

    best, hist = -1.0, []
    for ep in range(1, a.epochs + 1):
        perm = rng.permutation(len(tr_idx))
        t0, run, seen = time.time(), 0.0, 0
        for s in range(0, len(perm), a.batch):
            bb = tr_idx[perm[s:s + a.batch]]
            z, hp, ho, hm, ps, am, y = make_batch(banks, phi_ext, bb, a_phi, a_pool, n_pool, d_z)
            loss = nn.functional.cross_entropy(pol(z, hp, ho, hm, ps, phi_cand, am), y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            run += float(loss) * len(bb)
            seen += len(bb)
            if (s // a.batch) % 400 == 399:
                log(f"  ep{ep} {seen:,}/{len(perm):,} loss {run / seen:.4f} "
                    f"({(time.time() - t0) / 60:.1f}m)")
        f = fidelity(dv_idx)
        mean = float(np.mean([f[t] for t in TREES]))
        hist.append({"epoch": ep, "loss": run / max(seen, 1), "dev_top1": f, "mean": mean})
        log(f"[ep{ep}] loss {run / seen:.4f} - dev top-1 " +
            " ".join(f"{t}={f[t]:.4f}" for t in TREES) + f" - mean {mean:.4f}")
        json.dump({"analysis": "m4b_forest_policy", "split": "train_states", "test_touched": False,
                   "shuffle_control": bool(a.shuffle_control), "arch": a.arch, "state": a.state,
                   "dev_lo": a.dev_lo, "n_train_ex": int(len(tr_idx)), "n_dev_ex": int(len(dv_idx)),
                   "pool": [[int(x), int(y)] for x, y in pool], "history": hist},
                  open(os.path.join(OUT, f"{a.tag}_policy.json"), "w"), indent=1)
        if mean > best:
            best = mean
            torch.save({"sd": pol.state_dict(), "arch": a.arch, "state": a.state, "d_z": d_z,
                        "d_phi": d_phi, "n_pool": n_pool, "epoch": ep, "dev_top1": f,
                        "pool": [[int(x), int(y)] for x, y in pool],
                        "ext": [[int(x), int(y)] for x, y in ext],
                        "norm": [norm[0].tolist(), norm[1].tolist()],
                        "a_phi": a_phi, "a_pool": a_pool},
                       os.path.join(OUT, f"{a.tag}_policy.pt"))
            log(f"  [ckpt] epoch {ep} kept (mean dev top-1 {mean:.4f})")
    log("[done]")


if __name__ == "__main__":
    main()
