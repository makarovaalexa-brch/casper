r"""WORKSTREAM E1 / STEP 1 -- TRAJECTORY GENERATION: apply the FROZEN tree to the TRAIN users.

Design sheet: `docs/design/PAPER_B_POLICY_DESIGN.md`, and the binding cohort ruling is D.6b:

  > **AUTHOR SAID (2026-08-13):** *"so we train policy on train dataset, but we do not regrow the tree,
  > and we select model variant or chkp on val. is that right?"* -- confirmed.

WHAT THIS PRODUCES
------------------
For each of the 140,768 TRAIN users: the one deterministic 16-question interview the frozen tree gives
them (<= 5 branch questions on the ternary like / dislike / did-not-answer rule, then the reached
leaf's banked greedy tail), and a training example at EVERY position 0..15 including the empty state
-- 16 per user, ~2.25M in total, uniform over interview lengths.

Each example is (state -> label):
  state  = the tower+concept latent `z_t` (the recommender's own belief after folding the answers so
           far), the EXPLICIT ternary answer history, and the position / remaining budget;
  label  = the question the tree asks at that state, as an index into the 1,000-candidate pool.

THE TREE IS NEVER REGROWN. It is loaded frozen and only APPLIED. No NDCG is measured on a train user
anywhere in this file -- that is precisely what makes using them legitimate (the tower was trained on
them, so any NDCG there would be memorisation). Traversal needs only the user's own answers.

THE ANSWER-DISTRIBUTION FIX, AND WHY IT IS NOT OPTIONAL
------------------------------------------------------
Validation users answer only out of their 80% fold-in half. Train users have no such split on disk, so
answering from 100% of their ratings would give train interviews a systematically HIGHER answer rate
than the deployment distribution -- and answerability is the single most load-bearing property in this
chapter. `train_split.py` applies the SAME per-user 80/20 recipe, SAME seed 98765, gated byte-faithful
against the shipped `validation_te.csv`, and the 20% is then simply discarded. `--mode probe` measures
the resulting answer rates against the validation cohort's and REFUSES to proceed if they diverge
beyond `--ar_tol`.

FIREWALL
--------
The 10,000 canonical TEST users are never built, never loaded, never touched. `--split` accepts
`train` only and the validation cohort is built only as the reference for the probe's answer-rate
comparison and for the banked-partition contract check. There is no code path here that can name the
test cohort.

HARD RULE 11
------------
`z` is written straight into ONE preallocated memmap (`z_train.npy`, 140,768 x 16 x 200 float32 =
1.80 GB) slice by slice, and each chunk's small arrays go to their own sidecar the moment the chunk
finishes. A kill costs at most the chunk in flight. The manifest is contract-signed and a resume across
a changed contract is REFUSED, never merged.
"""
import os
import sys
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
sys.path.insert(0, _HERE)

import json
import time
import hashlib
import argparse
import numpy as np

from strategy_channel_suite import (log, build_greedy_env, set_active_rows, genv_mean, genv_latent,
                                    contract_selftest, CERT_TOWER, CERT_CONC, pool_contract)
from ndcg_tree import build_pool, leak_assert, split_rows, branch_of, BRANCHES

OUTDIR = os.path.join(_ROOT, "experiments", "battery", "policy_traj")
TREE_JSON = os.path.join(_ROOT, "experiments", "battery", "ndcgtree.json")
TREE16_JSON = os.path.join(_ROOT, "experiments", "battery", "ndcgtree16.json")
TOTAL_L = 16
N_TRAIN = 140768
OUTCOME = {"like": 0, "dislike": 1, "noanswer": 2}            # NEVER two. policy_net.OUTCOME, same map
BRANCH_OF_CODE = {v: k for k, v in OUTCOME.items()}
# The canonical policy-B m=0 intercept on the validation cohort. Every new regime in this project is
# snapped here first (`surprise-means-debug-the-harness`); it is also the regression gate on the
# `genv_latent` extraction, since `genv_eval` now routes through it.
VAL_M0_FULL = 0.15686538


def _sha(o):
    return hashlib.sha1(json.dumps(o, separators=(",", ":")).encode()).hexdigest()


def load_tree():
    tr = json.load(open(TREE_JSON))
    t16 = json.load(open(TREE16_JSON))
    assert t16["total_L"] == TOTAL_L, f"ndcgtree16 total_L is {t16['total_L']}"
    assert tr.get("test_touched") is False and t16.get("test_touched") is False
    assert tr["split"] == "validation" and t16["split"] == "validation"
    assert not tr.get("SANITY", False), "the SANITY tree is not the citable tree"
    return tr, t16


# ============================================================================ traversal
def descend(E, rows, tr, tag):
    """Apply the frozen tree to `rows`. Returns (leaf_rows, partitions).

    `leaf_rows[nid]` is the exact user list that lands in leaf `nid`; `partitions[nid]` is the
    recomputed ternary child-size dict at every internal node. Nothing is sampled and nothing is
    dropped -- `split_rows` asserts the three branches partition the node exactly.
    """
    nodes = tr["nodes"]
    leaf_rows, partitions, answer_rates = {}, {}, {}
    stack = [("root", list(rows))]
    while stack:
        nid, rs = stack.pop()
        rec = nodes[nid]
        if not rec["split"]:
            leaf_rows[nid] = rs
            continue
        q = tuple(int(x) for x in rec["splitter"])
        kids, ar, _z = split_rows(E, rs, q)
        partitions[nid] = {k: len(v) for k, v in kids.items()}
        answer_rates[nid] = ar
        for br in BRANCHES:
            stack.append((f"{nid}.{br}", kids[br]))
    got = sum(len(v) for v in leaf_rows.values())
    assert got == len(rows), f"{tag}: traversal lost users ({got} != {len(rows)})"
    return leaf_rows, partitions, answer_rates


def gate_partitions_against_banked(tr, partitions, answer_rates):
    """G-T1 (validation only). The recomputed ternary partition and answer rate at every internal node
    must reproduce the banked `child_sizes` / `answer_rate` EXACTLY. This is the free contract check
    that separates "the tree I am applying" from "a tree that looks like it" -- and it is the same
    gate `tree_minleaf_audit.py` passed 22/22."""
    nodes = tr["nodes"]
    n = 0
    for nid, got in partitions.items():
        want = nodes[nid]["child_sizes"]
        assert got == want, f"G-T1 PARTITION MISMATCH at {nid}: recomputed {got} != banked {want}"
        assert round(answer_rates[nid], 4) == round(float(nodes[nid]["answer_rate"]), 4), \
            f"G-T1 ANSWER-RATE MISMATCH at {nid}"
        n += 1
    log(f"[G-T1] {n}/{n} internal nodes reproduce their banked ternary child_sizes and answer_rate "
        f"EXACTLY -- this is the frozen tree, applied, not a lookalike")
    return n


# ============================================================================ per-cohort trajectories
def build_trajectories(E, rows, tr, t16, pool_index, want_z=True, tag="cohort"):
    """The 16 (state, label) examples per user. Returns a dict of arrays, all row-aligned to `rows`.

      label_idx (n,16) int16   pool index of the question asked at position t == the LABEL at state t
      outcome   (n,16) int8    like / dislike / did-not-answer of that question, per user
      leaf      (n,)   int16   index into `leaf_ids`
      z         (n,16,d) f32   the tower+concept latent after folding positions 0..t-1 (t=0 = empty)

    z_t is produced by `genv_latent`, which `genv_eval` itself now calls -- the policy's state and the
    scored interview are the same object, not two implementations of one idea.
    """
    leaf_rows, partitions, ars = descend(E, rows, tr, tag)
    leaves = {l["nid"]: l for l in t16["leaf_table"]}
    leaf_ids = sorted(leaves)
    n = len(rows)
    d = E["ctx"].d
    pos_of = {int(r): i for i, r in enumerate(rows)}
    label_idx = np.full((n, TOTAL_L), -1, np.int16)
    outcome = np.full((n, TOTAL_L), -1, np.int8)
    leaf_v = np.full(n, -1, np.int16)
    z = np.zeros((n, TOTAL_L, d), np.float32) if want_z else None

    t0 = time.time()
    for li, nid in enumerate(leaf_ids):
        rs = leaf_rows.get(nid, [])
        if not rs:
            log(f"[{tag}] leaf {nid}: 0 users")
            continue
        seq = [(int(a), int(b)) for a, b in leaves[nid]["own_sequence"]]
        assert len(seq) == TOTAL_L and len(set(seq)) == TOTAL_L, \
            f"leaf {nid} sequence is not 16 distinct questions"
        depth = int(leaves[nid]["depth"])
        path = [(int(a), int(b)) for a, b in leaves[nid]["path"]]
        assert seq[:depth] == path, f"leaf {nid}: own_sequence prefix != the node's path"
        jj = np.asarray([pos_of[int(r)] for r in rs], np.int64)
        leaf_v[jj] = li
        for t, q in enumerate(seq):
            assert q in pool_index, f"leaf {nid} position {t}: question {q} is outside the pool"
            label_idx[jj, t] = pool_index[q]
            lab, _ar, _nz = branch_of(E, rs, q)
            outcome[jj, t] = np.asarray([OUTCOME[str(x)] for x in lab], np.int8)
        # the branch outcomes at positions < depth MUST be the branches that routed the user here
        want = [b for b in nid.split(".")[1:]]
        for t, b in enumerate(want):
            assert (outcome[jj, t] == OUTCOME[b]).all(), \
                f"leaf {nid}: position {t} outcome disagrees with the path branch '{b}' -- the " \
                f"traversal and the answer rule are not the same rule"
        if want_z:
            set_active_rows(E, rs)
            for t in range(TOTAL_L):
                z[jj, t] = genv_latent(E, seq[:t], rs).numpy()
        log(f"[{tag}] leaf {nid}: {len(rs)} users, depth {depth} "
            f"({(time.time() - t0) / 60:.1f}m elapsed)")
    set_active_rows(E, rows)
    assert (label_idx >= 0).all() and (outcome >= 0).all() and (leaf_v >= 0).all(), \
        f"{tag}: not every user has all 16 positions filled"
    return {"label_idx": label_idx, "outcome": outcome, "leaf": leaf_v, "leaf_ids": leaf_ids,
            "z": z, "partitions": partitions, "answer_rates": ars,
            "leaf_sizes": {k: len(v) for k, v in leaf_rows.items()}}


# ============================================================================ outcome labels
def outcome_labels(E, rows):
    """THE THREE-WAY OUTCOME OF EVERY CATALOGUE QUESTION FOR EVERY USER IN `rows`.

    Design sheet PART F (the two-head design). The outcome of question q for user u is DETERMINISTIC
    given u's fold-in half under the project's structural answer rule -- `ndcg_tree.branch_of`, the
    same rule the tree branches on -- so no model call and no sampling is involved. It is measured,
    not surrogate, and it is not a function of the held-out target.

      item i     absent from the fold-in lookup -> NOANSWER; level >= LIKE_MIN_LEVEL (7, i.e. >= 4.0
                 stars) -> LIKE; otherwise DISLIKE.
      concept c  not `foldable` -> NOANSWER; signed value > 0 -> LIKE; otherwise DISLIKE.

    HARD RULE 1 / 11: nothing is sampled -- ALL 18,359 items and ALL 1,031 concepts for every user.
    The item side is stored SPARSELY (a CSR of the ANSWERED items only; everything absent is
    NOANSWER by the rule above), because the dense 140,768 x 18,359 int8 block would be 2.6 GB for a
    matrix that is ~99.2% one constant. The concept side is genuinely dense (~80% answered) and is
    stored dense at 1 byte per cell.
    """
    from ndcg_tree import LIKE_MIN_LEVEL
    lvl = E["lvl"]
    ridx = np.asarray(rows, np.int64)
    ptr = np.zeros(len(ridx) + 1, np.int64)
    ids, outs = [], []
    for j, r in enumerate(ridx):
        d = lvl[int(r)]
        if d:
            k = np.fromiter(d.keys(), np.int64, len(d))
            v = np.fromiter(d.values(), np.int64, len(d))
            o = np.argsort(k)
            k, v = k[o], v[o]
            ids.append(k.astype(np.int32))
            outs.append(np.where(v >= LIKE_MIN_LEVEL, OUTCOME["like"], OUTCOME["dislike"]).astype(np.int8))
            ptr[j + 1] = ptr[j] + len(k)
        else:
            ptr[j + 1] = ptr[j]
    item_id = np.concatenate(ids) if ids else np.zeros(0, np.int32)
    item_out = np.concatenate(outs) if outs else np.zeros(0, np.int8)
    ok = E["foldable"][ridx]                                   # (n, C) bool
    pos = E["V"][ridx] > 0
    conc = np.where(~ok, OUTCOME["noanswer"],
                    np.where(pos, OUTCOME["like"], OUTCOME["dislike"])).astype(np.int8)
    return {"item_ptr": ptr, "item_id": item_id, "item_out": item_out, "conc_out": conc}


def gate_outcome_labels(E, rows, lab, pool, n_check=24):
    """G-O1. The harvested label table must reproduce `branch_of` -- the tree's OWN answer rule -- for
    every user, on a set of questions drawn from BOTH channels. Exact equality, no tolerance: the
    labels and the tree's branches are the same rule or the harvest is a lookalike."""
    from ndcg_tree import branch_of
    ni = E["ctx"].ni
    qs = [pool[i] for i in np.linspace(0, len(pool) - 1, n_check).astype(int)]
    for q in qs:
        want = np.asarray([OUTCOME[str(x)] for x in branch_of(E, rows, q)[0]], np.int8)
        got = lookup_outcomes(lab, np.arange(len(rows)), int(q[0]), int(q[1]), ni)
        assert np.array_equal(got, want), f"G-O1 OUTCOME LABEL MISMATCH at question {tuple(q)}"
    return len(qs)


def lookup_outcomes(lab, jj, ch, iid, ni):
    """The outcome of ONE question for the rows `jj` of a harvested label block."""
    if ch == 1:
        return lab["conc_out"][jj, iid].astype(np.int8)
    out = np.full(len(jj), OUTCOME["noanswer"], np.int8)
    for a, j in enumerate(jj):
        lo, hi = int(lab["item_ptr"][j]), int(lab["item_ptr"][j + 1])
        k = np.searchsorted(lab["item_id"][lo:hi], iid)
        if k < hi - lo and int(lab["item_id"][lo + k]) == iid:
            out[a] = lab["item_out"][lo + k]
    return out


# ============================================================================ answer-rate diagnostics
def answer_stats(E, rows, traj, pool, item_pool, conc_pool):
    """The numbers that decide whether the train cohort's answer distribution matches validation's."""
    ridx = np.asarray(rows, np.int64)
    out = traj["outcome"]
    answered = (out != OUTCOME["noanswer"])
    lvl = E["lvl"]
    it = {int(i) for _c, i in item_pool}
    item_ar = float(np.mean([len(lvl[int(r)].keys() & it) for r in ridx]) / len(it))
    cc = np.asarray([int(c) for _c, c in conc_pool], np.int64)
    conc_ar = float(E["foldable"][np.ix_(ridx, cc)].mean())
    return {"n_users": len(rows),
            "interview_answer_rate": float(answered.mean()),
            "interview_answer_rate_by_position": [float(x) for x in answered.mean(0)],
            "users_answering_nothing": float((answered.sum(1) == 0).mean()),
            "mean_answered_of_16": float(answered.sum(1).mean()),
            "outcome_mix": {k: float((out == v).mean()) for k, v in OUTCOME.items()},
            "pool_item_answer_rate": item_ar, "pool_concept_answer_rate": conc_ar,
            "channel_mix_of_labels": float(np.mean(
                [1.0 if pool[int(j)][0] == 1 else 0.0 for j in traj["label_idx"].ravel()]))}


def compare_stats(v, t, tol):
    """The Step-0 refusal. A train cohort whose interviews answer at a different rate is not the
    deployment distribution, and a policy trained on it is trained on states that cannot occur."""
    rows = []
    ok = True
    for k in ("interview_answer_rate", "mean_answered_of_16", "users_answering_nothing",
              "pool_item_answer_rate", "pool_concept_answer_rate"):
        dv = float(t[k]) - float(v[k])
        rel = abs(dv) / max(abs(float(v[k])), 1e-9)
        rows.append((k, float(v[k]), float(t[k]), dv, rel))
        if k in ("interview_answer_rate", "pool_item_answer_rate", "pool_concept_answer_rate") \
                and rel > tol:
            ok = False
    log("  " + "answer-rate comparison (validation vs train)".upper())
    log(f"  {'quantity':34s} {'val':>10s} {'train':>10s} {'diff':>10s} {'rel':>8s}")
    for k, a, b, dv, rel in rows:
        log(f"  {k:34s} {a:10.4f} {b:10.4f} {dv:+10.4f} {rel * 100:7.2f}%")
    return ok, rows


# ============================================================================ the run
def build_env(args, split, cohort_slice, label):
    E = build_greedy_env(args.snapshot, split, args.concept_ckpt, label, cohort_slice=cohort_slice)
    contract_selftest(E, E["rows"])
    leak_assert(E, E["rows"], label)
    return E


def canonical_snap(E):
    """G-E1e, and the regression gate on the `genv_latent` extraction: the m=0 policy-B intercept on
    validation must reproduce 0.15686538 EXACTLY. If it does not, the ruler is wrong and nothing
    downstream means anything."""
    f, t = genv_mean(E, [], E["rows"])
    log(f"[G-E1e] validation m=0 intercept: full={f!r} tail={t!r} (canonical full {VAL_M0_FULL})")
    assert round(f, 8) == VAL_M0_FULL, (
        f"G-E1e CANONICAL SNAP FAILED: m=0 full is {f!r}, the canonical policy-B value is "
        f"{VAL_M0_FULL}. STOP -- the ruler is wrong.")
    return f, t


def probe(args):
    """Step 0. No heavy compute: one validation env (contract checks + the canonical snap + the banked
    partition gate + the reference answer rates) and one train slice (the same, plus timing)."""
    tr, t16 = load_tree()
    Ev = build_env(args, "validation", None, "policytraj/val")
    assert len(Ev["rows"]) == 10000, f"validation cohort is {len(Ev['rows'])} users"
    canonical_snap(Ev)
    pool, item_pool, conc_pool = build_pool(Ev, args.n_items, args.n_concepts)
    assert len(pool) == tr["n_pool"] == 1000
    pool_index = {tuple(q): i for i, q in enumerate(pool)}
    pool_sha = _sha([[int(a), int(b)] for a, b in pool])
    log(f"[pool] {len(item_pool)} items + {len(conc_pool)} concepts, sha1 {pool_sha[:12]}")

    t0 = time.time()
    tv = build_trajectories(Ev, Ev["rows"], tr, t16, pool_index, want_z=False, tag="val")
    gate_partitions_against_banked(tr, tv["partitions"], tv["answer_rates"])
    sv = answer_stats(Ev, Ev["rows"], tv, pool, item_pool, conc_pool)
    val_leaf = {k: v for k, v in sorted(tv["leaf_sizes"].items())}
    log(f"[val] trajectories (no z) in {time.time() - t0:.1f}s")
    del Ev

    lo, hi = 0, int(args.n_probe)
    Et = build_env(args, "train", (lo, hi), f"policytraj/train[{lo}:{hi}]")
    assert len(Et["rows"]) == hi - lo, (
        f"train slice has {len(Et['rows'])} scorable users of {hi - lo} -- a train user with no "
        f"target half would break the 80/20 accounting")
    pool_t, ip_t, cp_t = build_pool(Et, args.n_items, args.n_concepts)
    assert pool_t == pool, "the train env's candidate pool is not byte-identical to the validation env's"
    log("[pool] train env pool is BYTE-IDENTICAL to the validation env's (it is derived from "
        "train.csv counts and the frozen concept stats, so it cannot move with the cohort)")

    t1 = time.time()
    tt = build_trajectories(Et, Et["rows"], tr, t16, pool_index, want_z=False, tag="train")
    t_nz = time.time() - t1
    st = answer_stats(Et, Et["rows"], tt, pool, item_pool, conc_pool)
    ok, rowsc = compare_stats(sv, st, args.ar_tol)

    # z cost, measured on one leaf-sized slice of the real thing
    t2 = time.time()
    sub = Et["rows"][:args.n_z_probe]
    set_active_rows(Et, sub)
    leaf0 = sorted({l["nid"] for l in t16["leaf_table"]})[0]
    seq0 = [(int(a), int(b)) for a, b in
            [l for l in t16["leaf_table"] if l["nid"] == leaf0][0]["own_sequence"]]
    for t in range(TOTAL_L):
        genv_latent(Et, seq0[:t], sub)
    z_sec = time.time() - t2
    set_active_rows(Et, Et["rows"])

    per_1k_nz = t_nz / (hi - lo) * 1000
    per_1k_z = z_sec / len(sub) * 1000
    per_1k = per_1k_nz + per_1k_z
    d = Et["ctx"].d
    z_bytes = N_TRAIN * TOTAL_L * d * 4
    import shutil
    free = shutil.disk_usage(_ROOT).free
    log("=" * 100)
    log(f"[cost] traversal+outcomes: {per_1k_nz:.1f} s / 1,000 train users")
    log(f"[cost] z (16 folds/user):  {per_1k_z:.1f} s / 1,000 train users")
    log(f"[cost] TOTAL:              {per_1k:.1f} s / 1,000 -> "
        f"{per_1k * N_TRAIN / 1000 / 3600:.2f} h for all {N_TRAIN:,} train users")
    log(f"[cost] z cache: {N_TRAIN:,} x {TOTAL_L} x {d} float32 = {z_bytes / 1e9:.2f} GB; "
        f"disk free {free / 1e9:.2f} GB")
    log("=" * 100)

    blob = {"analysis": "policy_traj_step0_probe", "design_sheet":
            "docs/design/PAPER_B_POLICY_DESIGN.md", "citable": True,
            "test_touched": False, "cohorts": ["validation (reference)", f"train[{lo}:{hi}]"],
            "canonical_snap_val_m0_full": VAL_M0_FULL,
            "G-T1_banked_partition_gate": "22/22 internal nodes exact",
            "pool_sha1": pool_sha, "pool_identical_train_vs_val": True,
            "val": sv, "train": st, "answer_rate_ok": bool(ok), "ar_tol": args.ar_tol,
            "comparison": [{"quantity": k, "val": a, "train": b, "diff": dv, "rel": r}
                           for k, a, b, dv, r in rowsc],
            "val_leaf_sizes": val_leaf, "train_leaf_sizes": tt["leaf_sizes"],
            "train_partitions": tt["partitions"], "train_answer_rates": tt["answer_rates"],
            "banked_partitions": {k: v["child_sizes"] for k, v in tr["nodes"].items() if v["split"]},
            "cost": {"sec_per_1k_traversal": per_1k_nz, "sec_per_1k_z": per_1k_z,
                     "sec_per_1k_total": per_1k,
                     "projected_hours_all_train": per_1k * N_TRAIN / 1000 / 3600,
                     "z_cache_bytes": z_bytes, "disk_free_bytes": free},
            "tower": Et["tower"], "conc_ckpt": Et["conc_ckpt"], "mask_policy": "B"}
    os.makedirs(OUTDIR, exist_ok=True)
    p = os.path.join(OUTDIR, "step0_probe.json")
    with open(p + ".tmp", "w") as fh:
        json.dump(blob, fh, indent=2, default=float)
    os.replace(p + ".tmp", p)
    log(f"[probe] wrote {p}")
    if not ok:
        raise SystemExit(
            "STEP 0 REFUSAL: the train cohort's answer rates do not match the validation cohort's "
            f"within {args.ar_tol:.1%}. Reporting rather than proceeding (the brief's instruction).")
    return blob


def gen(args):
    """Step 1. All 140,768 train users, chunked, resumable, one preallocated z memmap."""
    tr, t16 = load_tree()
    os.makedirs(OUTDIR, exist_ok=True)
    # the pool and the contract come from a validation env, so the artifact states the same ruler the
    # tree does; the train envs then assert byte-identity against it.
    Ev = build_env(args, "validation", None, "policytraj/val")
    canonical_snap(Ev)
    pool, item_pool, conc_pool = build_pool(Ev, args.n_items, args.n_concepts)
    pool_index = {tuple(q): i for i, q in enumerate(pool)}
    pool_sha = _sha([[int(a), int(b)] for a, b in pool])
    val_pc = pool_contract(Ev["pool"], Ev["rows"], policy="B")
    d = Ev["ctx"].d

    # --- the VALIDATION trajectories, with z. These are NOT training data: they are the held-out
    # --- states G-E1a measures top-1 agreement on, and the tree's own realised interviews that the
    # --- policy's rollout is compared against (deviation position, per-position agreement, Pep
    # --- divergence). Produced here so the validation env is built once for both jobs.
    vpath = os.path.join(OUTDIR, "val_traj.npz")
    if not os.path.exists(vpath):
        tv = build_trajectories(Ev, Ev["rows"], tr, t16, pool_index, want_z=True, tag="val")
        gate_partitions_against_banked(tr, tv["partitions"], tv["answer_rates"])
        lv = outcome_labels(Ev, Ev["rows"])
        nq = gate_outcome_labels(Ev, Ev["rows"], lv, pool)
        log(f"[G-O1] validation outcome labels reproduce branch_of EXACTLY on {nq} questions "
            f"x {len(Ev['rows'])} users")
        np.savez_compressed(os.path.join(OUTDIR, "ans_val.npz.tmp.npz"), **lv)
        os.replace(os.path.join(OUTDIR, "ans_val.npz.tmp.npz"),
                   os.path.join(OUTDIR, "ans_val.npz"))
        zv = os.path.join(OUTDIR, "z_val.npy")
        np.save(zv + ".tmp.npy", tv["z"])
        os.replace(zv + ".tmp.npy", zv)
        np.savez_compressed(vpath + ".tmp.npz", uid=np.asarray(Ev["rows"], np.int32),
                            label_idx=tv["label_idx"], outcome=tv["outcome"], leaf=tv["leaf"],
                            leaf_ids=np.asarray(tv["leaf_ids"]),
                            pool=np.asarray([[int(a), int(b)] for a, b in pool], np.int32))
        os.replace(vpath + ".tmp.npz", vpath)
        log(f"[val] wrote {os.path.basename(vpath)} + z_val.npy "
            f"({len(Ev['rows'])} users x {TOTAL_L} positions)")
    del Ev

    sig = {"v": 2, "outcome_labels": "three-way branch_of over the full catalogue",
           "split": "train", "n_train": N_TRAIN, "total_L": TOTAL_L, "d": d,
           "tower": os.path.basename(args.snapshot), "conc_ckpt": os.path.basename(args.concept_ckpt),
           "mask_policy": "B", "pool_sha1": pool_sha, "n_pool": len(pool),
           "tree": "ndcgtree.json+ndcgtree16.json", "tree_nodes": len(tr["nodes"]),
           "val_pool_contract": val_pc, "answer_half": "liang 80/20 seed 98765 (train_split.py)",
           "chunk": int(args.chunk)}
    man_path = os.path.join(OUTDIR, "manifest.json")
    man = json.load(open(man_path)) if os.path.exists(man_path) else None
    if man is not None and man.get("sig") != sig:
        log(f"[manifest] REFUSING to resume: the contract changed.\n  banked {man.get('sig')}\n"
            f"  now    {sig}")
        man = None
    if man is None:
        man = {"sig": sig, "done": {}, "started": time.strftime("%Y-%m-%d %H:%M:%S")}

    z_path = os.path.join(OUTDIR, "z_train.npy")
    if not os.path.exists(z_path):
        import shutil
        need = N_TRAIN * TOTAL_L * d * 4
        free = shutil.disk_usage(_ROOT).free
        assert free > need * 1.05, (
            f"REFUSING to start: the z cache needs {need / 1e9:.2f} GB and only {free / 1e9:.2f} GB "
            f"is free. A long run that fills the disk loses everything (HARD RULE 11).")
        Z = np.lib.format.open_memmap(z_path, mode="w+", dtype=np.float32,
                                      shape=(N_TRAIN, TOTAL_L, d))
        del Z
        log(f"[z] preallocated {z_path} ({need / 1e9:.2f} GB)")
    Z = np.lib.format.open_memmap(z_path, mode="r+")
    assert Z.shape == (N_TRAIN, TOTAL_L, d), f"z memmap shape {Z.shape}"

    t00 = time.time()
    chunks = [(lo, min(lo + int(args.chunk), N_TRAIN)) for lo in range(0, N_TRAIN, int(args.chunk))]
    for lo, hi in chunks:
        key = f"{lo}_{hi}"
        if man["done"].get(key):
            log(f"[chunk {key}] already banked -- skipping")
            continue
        E = build_env(args, "train", (lo, hi), f"policytraj/train[{lo}:{hi}]")
        assert len(E["rows"]) == hi - lo, f"train slice {key}: {len(E['rows'])} scorable users"
        pool_t, _i, _c = build_pool(E, args.n_items, args.n_concepts)
        assert pool_t == pool, f"chunk {key}: the candidate pool is not the validation env's"
        t0 = time.time()
        traj = build_trajectories(E, E["rows"], tr, t16, pool_index, want_z=True, tag=f"train{key}")
        stats = answer_stats(E, E["rows"], traj, pool, item_pool, conc_pool)
        rows_abs = np.asarray(E["rows"], np.int64) + lo          # env rows are slice-local
        assert rows_abs.min() >= lo and rows_abs.max() < hi
        Z[rows_abs] = traj["z"]
        Z.flush()
        meta = os.path.join(OUTDIR, f"meta_{key}.npz")
        np.savez_compressed(meta + ".tmp.npz", uid=rows_abs.astype(np.int32),
                            label_idx=traj["label_idx"], outcome=traj["outcome"],
                            leaf=traj["leaf"])
        os.replace(meta + ".tmp.npz", meta)
        # PART F: the three-way outcome of EVERY catalogue question for every user in the chunk.
        # Zero NDCG cost -- it is the structural answer rule, evaluated. Gated against `branch_of`
        # on both channels before it is banked.
        lab = outcome_labels(E, E["rows"])
        nq = gate_outcome_labels(E, E["rows"], lab, pool)
        ansp = os.path.join(OUTDIR, f"ans_{key}.npz")
        np.savez_compressed(ansp + ".tmp.npz", uid=rows_abs.astype(np.int32), **lab)
        os.replace(ansp + ".tmp.npz", ansp)
        log(f"[chunk {key}] outcome labels: {len(lab['item_id']):,} answered item cells "
            f"({len(lab['item_id']) / len(E['rows']):.1f}/user), concept block "
            f"{lab['conc_out'].shape}; G-O1 exact on {nq} questions")
        man["done"][key] = {"n_users": hi - lo, "minutes": round((time.time() - t0) / 60, 2),
                            "leaf_sizes": traj["leaf_sizes"], "stats": stats,
                            "partitions": traj["partitions"]}
        man["leaf_ids"] = traj["leaf_ids"]
        man["pool"] = [[int(a), int(b)] for a, b in pool]
        with open(man_path + ".tmp", "w") as fh:
            json.dump(man, fh, indent=1, default=float)
        os.replace(man_path + ".tmp", man_path)
        log(f"[chunk {key}] DONE in {(time.time() - t0) / 60:.1f}m; "
            f"{len(man['done'])}/{len(chunks)} chunks, {(time.time() - t00) / 60:.1f}m elapsed")
        del E
    log(f"[gen] all {len(chunks)} chunks banked in {(time.time() - t00) / 60:.1f}m")
    return man


def main():
    ap = argparse.ArgumentParser(description="E1 trajectory generation. See "
                                             "docs/design/PAPER_B_POLICY_DESIGN.md D.6b")
    ap.add_argument("--mode", required=True, choices=["probe", "gen"])
    ap.add_argument("--split", default="train", choices=["train"],
                    help="FIREWALL: train only. The 10,000 TEST users are never built here, and "
                         "`test` is not an accepted value.")
    ap.add_argument("--snapshot", default=CERT_TOWER)
    ap.add_argument("--concept_ckpt", default=CERT_CONC)
    ap.add_argument("--n_items", type=int, default=500)
    ap.add_argument("--n_concepts", type=int, default=500)
    ap.add_argument("--n_probe", type=int, default=10000, help="probe: train users in the slice")
    ap.add_argument("--n_z_probe", type=int, default=2000, help="probe: users timed for the z folds")
    ap.add_argument("--ar_tol", type=float, default=0.05,
                    help="probe: maximum RELATIVE answer-rate gap train vs validation before refusing")
    ap.add_argument("--chunk", type=int, default=10000)
    ap.add_argument("--full_threads", action="store_true",
                    help="consumed by strategy_channel_suite AT IMPORT (it reads sys.argv)")
    args = ap.parse_args()
    assert args.split == "train", "FIREWALL: this file never touches the test cohort"
    log("#" * 110)
    log("# WORKSTREAM E1 -- TRAJECTORY GENERATION (the frozen tree, APPLIED to the train users)")
    log(f"# mode={args.mode}")
    log("# FIREWALL: the 10,000 canonical TEST users are NOT built, NOT loaded, NOT touched.")
    log("# NO NDCG IS EVER MEASURED ON A TRAIN USER. The tree is never regrown.")
    log("# HARD RULE 1: nothing is sampled -- every user, every position, every candidate.")
    log("#" * 110)
    return probe(args) if args.mode == "probe" else gen(args)


if __name__ == "__main__":
    main()
