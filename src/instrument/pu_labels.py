r"""PART K -- PER-USER GAIN DISTILLATION, STEP 1: THE LABELLING PASS.

Design sheet: `docs/design/PAPER_B_POLICY_DESIGN.md` **PART K is binding** (commit b75d57c). Parts
H, I and J record why every alternative died; nothing here may re-introduce one of them.

**NOTHING HERE HAS BEEN RUN.** Committed before any run (HARD RULE 10). `--selftest` is pure
(no data, no environment, no fold, no fit) and `--cost_only` builds nothing. The pilot runs only on
the author's explicit go, behind `--i_have_author_goahead`.

WHAT THIS PRODUCES, AND WHY IT IS AFFORDABLE
--------------------------------------------
Every prior attempt inherited the tree's frame: one label = the COHORT argmax at ONE shared state,
priced at ~2.4 h (1,000 candidates x 10,000 users). `genv_eval_multi` batches HETEROGENEOUS per-user
sequences in one pass, so if each user sits in their OWN state, one sweep of the 1,000 candidates
costs the same as ONE cohort sweep and yields ~8,000 distinct states, each with a full per-user
measured gain vector. That is the author's sentence -- *the best NDCG uplift for that user in that
circumstance* -- as a dataset (K.0).

    per-user gain(c, u) = NDCG@10(state_u + c) - NDCG@10(state_u)

measured, both objectives (full and tail), for all 1,000 candidates, for every user in the cohort.

WHAT IS NOT SAMPLED, AND THE ONE REDUCTION THAT IS DECLARED
------------------------------------------------------------
Candidates: never -- all 1,000, every user, every round. Answers: never -- the structural rule, the
user's own fold-in half. USERS: the FIT cohort is 8,000 of the 10,000 validation users and that
split is the FIREWALL (K.2), not a convenience. `--pilot` restricts to 1,000 of the 8,000 and that
is the ONLY user reduction in this file -- it is K.4's pre-registered pilot, it is recorded in the
artifact, and there is no free `--n_users` knob to widen it silently (HARD RULE 1).

THE FIREWALL (K.2)
------------------
The 10,000 validation users are split 8,000 FIT / 2,000 HOLDOUT by user, seed 98765, fixed once by
`fit_holdout_split` and written into every artifact this chain produces. Labels are computed on FIT
only. Every behavioural claim is proven on HOLDOUT, whose states and labels are never touched. The
10,000 canonical TEST users are NOT built, NOT loaded, NOT touched: `--split test` is refused by the
parser, and `test_touched=false` is stamped on the artifact.

THE STATE MIXTURE (K.3) -- declared here, weights recorded in the artifact
---------------------------------------------------------------------------
Each FIT user gets ONE state, drawn from the deployment mixture:

  tree   a PREFIX of that user's own frozen-tree trajectory, length t ~ U{1..15}. The empty state
         (t = 0) is EXCLUDED: E1 already handles it with the tree's root and relabelling behaviour
         the policy already has is the phase-3 failure mode Part I.0 warns about.
  vol    k ~ U{1..8} VOLUNTEERED items, popularity-weighted from the user's rated fold-in half, via
         `policy_resume.draw_volunteered` -- R's own audited draw, not a lookalike. The asked-history
         is EMPTY and position t = 0: the belief has moved, nothing was asked. R2 measured that this
         exact pairing is what E1 never saw.
  rand   a random partial interview: m ~ U{1..8} candidates drawn uniformly from the 1,000-pool,
         asked in that order, answered by the structural rule.

Mixture weights are CLI flags with declared defaults (equal thirds), recorded in the artifact.
Round 2+ adds the policy's own rollout states -- the affordable DAgger step -- and is NOT this file.

PERSISTENCE (HARD RULE 11 + persist the finest-grained quantity)
-----------------------------------------------------------------
One append-only `.f32`, one record per candidate, each record = `n_users` float32 FULL followed by
`n_users` float32 TAIL, in the cohort's fixed row order. **Record 0 is the BASE** (the state alone),
so the file is self-contained: a gain is a subtraction the reader performs, never a number the
writer rounds. Therefore

    bytes == (1 + n_candidates_done) * 2 * n_users * 4

and that arithmetic is the resume cursor -- asserted, not trusted. Alongside it a contract-signed
JSON sidecar carries the pool sha1, the DERIVED `pool_contract`, the FIT/HOLDOUT split, the full
per-user state assignment (type + prefix), the composition and the progress. A resume across a
changed contract is REFUSED by `mask_contract.resume_ok` (an absent field is a refusal, never a
pass). A kill costs one candidate.

GATES, ALL BEFORE THE FIRST CANDIDATE
--------------------------------------
  canonical snap   the m=0 policy-B intercept must reproduce 0.15686538 EXACTLY, or the ruler is
                   wrong and nothing downstream means anything.
  leak             0/users may have a held-out target inside the routing/answer lookup.
  banked headline  `greedy_static_policyB_combined` must rescore to its banked 0.25561016113602253
                   on THIS env -- the free contract check on the whole chain
                   (`banked-global-greedy-is-a-policy-A-artifact`).
  composition      state-type mix, prefix-length histogram, users per state type, PRINTED BEFORE the
                   first candidate is evaluated. Mandatory, not decorative (K.4).

COST -- STATED, NOT GUESSED
----------------------------
Measured on this box in phase 3: `s/eval ~= 0.0857 + 1.57 * (n/1000)`. Pilot (1,000 candidates at
n=1,000) ~= 1.66 s/eval ~= 27 min. Full round at n=8,000 ~= 12.6 s/eval ~= 3.5 h. `--cost_only`
prints both and builds nothing. The run re-measures its own rate and prints measured-vs-model,
because a cost model is a property of THIS box on THIS night and is allowed to be wrong out loud.

USAGE
  python src/instrument/pu_labels.py --selftest
  python src/instrument/pu_labels.py --cost_only [--pilot]
  python src/instrument/pu_labels.py --pilot --threads 6 --i_have_author_goahead
  python src/instrument/pu_labels.py --mode compose          # re-print composition, no env
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

from policy_net import OUTDIR, SEED, TOTAL_L, STATIC_JSON, STATIC_VAL_FULL, VAL_M0_FULL, log
from policy_labels import atomic_write
from mask_contract import resume_ok

PU_DIR = os.path.join(OUTDIR, "pu_labels")
GAINS_F32 = "pu_gains.f32"
SIDECAR = "pu_labels.json"
STATE_NPZ = "pu_state.npz"

# ------------------------------------------------------------------ the firewall, fixed once (K.2)
N_VAL = 10000
N_FIT = 8000
N_HOLDOUT = 2000
SPLIT_SEED = 98765                 # the canonical seed; the split is fixed ONCE and never re-drawn
PILOT_N = 1000                     # K.4, the ONLY authorised user reduction in this file
PILOT_SEED = 98766                 # SPLIT_SEED + 1; declared, so the pilot cohort is reproducible

# ------------------------------------------------------------------ the state mixture (K.3)
STATE_TYPES = ("tree", "vol", "rand")
TREE_T_MIN, TREE_T_MAX = 1, 15     # t = 0 (the empty state) is excluded -- see the module docstring
VOL_K_MIN, VOL_K_MAX = 1, 8
RAND_M_MIN, RAND_M_MAX = 1, 8

# ------------------------------------------------------------------ the cost model (phase 3, measured)
COST_A = 0.0857                    # fixed per-eval overhead, seconds
COST_B = 1.57                      # seconds per 1,000 scored users

N_ITEMS_POOL = 500
N_CONC_POOL = 500


def cost_model(n_users):
    """Seconds per `genv_eval_multi` pass over `n_users`. Phase-3 measurement, THIS box."""
    return COST_A + COST_B * (n_users / 1000.0)


def sha1_of(obj):
    return hashlib.sha1(json.dumps(obj, sort_keys=True, default=float).encode()).hexdigest()


# ============================================================== THE FIREWALL SPLIT (K.2), fixed once
def fit_holdout_split(rows, seed=SPLIT_SEED):
    """The 8,000 / 2,000 by-user split of the validation cohort. Deterministic, disjoint, exhaustive.

    Fixed ONCE and written into every artifact of this chain. All labels, all training and all model
    selection live on FIT; every behavioural claim is proven on HOLDOUT, whose states and labels are
    never touched (K.2). It is a function of `rows` and `seed` alone, so any file can recompute it
    and assert against the sha1 rather than trusting a stored list.
    """
    r = np.asarray(rows, np.int64)
    assert len(r) == N_VAL, f"the firewall split is defined on {N_VAL} validation users, got {len(r)}"
    assert len(np.unique(r)) == len(r), "duplicate rows handed to the firewall split"
    perm = np.random.default_rng(seed).permutation(len(r))
    fit = np.sort(r[perm[:N_FIT]])
    hold = np.sort(r[perm[N_FIT:]])
    assert len(fit) == N_FIT and len(hold) == N_HOLDOUT
    assert not np.intersect1d(fit, hold).size, "FIT and HOLDOUT overlap -- the firewall is broken"
    assert np.array_equal(np.sort(np.concatenate([fit, hold])), np.sort(r)), "the split lost a user"
    info = {"seed": int(seed), "n_fit": N_FIT, "n_holdout": N_HOLDOUT,
            "fit_sha1": sha1_of([int(x) for x in fit]),
            "holdout_sha1": sha1_of([int(x) for x in hold]),
            "rule": ("numpy.default_rng(98765).permutation over the canonical validation row order; "
                     "first 8,000 = FIT, last 2,000 = HOLDOUT; both sorted. Fixed once (K.2).")}
    return fit, hold, info


def pilot_cohort(fit, n=PILOT_N, seed=PILOT_SEED):
    """K.4's 1,000 FIT users. A seeded draw FROM FIT, sorted, recorded in the artifact."""
    f = np.asarray(fit, np.int64)
    sel = np.random.default_rng(seed).choice(len(f), size=n, replace=False)
    return np.sort(f[sel])


# ============================================================== the append-only .f32 (HARD RULE 11)
def rec_bytes(n_users):
    """One record = n_users float32 FULL then n_users float32 TAIL."""
    return 2 * int(n_users) * 4


def n_records(path, n_users):
    """Records fully on disk. A torn tail (a kill mid-write) is reported, never silently accepted."""
    if not os.path.exists(path):
        return 0, 0
    sz = os.path.getsize(path)
    rb = rec_bytes(n_users)
    return sz // rb, sz % rb


def truncate_torn(path, n_users):
    """Drop a partial trailing record. A kill costs ONE candidate, and it costs it visibly."""
    n, rem = n_records(path, n_users)
    if rem:
        log(f"[resume] {os.path.basename(path)} has a TORN trailing record ({rem} bytes) -- "
            f"truncating to {n} whole records. A kill costs one candidate, by design.")
        with open(path, "r+b") as fh:
            fh.truncate(n * rec_bytes(n_users))
    return n


def append_record(path, full, tail):
    """Append one record and FLUSH IT. HARD RULE 11(a): written the moment it completes."""
    f = np.ascontiguousarray(np.asarray(full, np.float32))
    t = np.ascontiguousarray(np.asarray(tail, np.float32))
    assert f.shape == t.shape and f.ndim == 1
    with open(path, "ab") as fh:
        fh.write(f.tobytes()); fh.write(t.tobytes())
        fh.flush(); os.fsync(fh.fileno())


def read_record(path, j, n_users):
    """Record `j` as (full, tail). Record 0 is the BASE; candidate i is record i + 1."""
    rb = rec_bytes(n_users)
    a = np.fromfile(path, dtype=np.float32, count=2 * n_users, offset=j * rb)
    assert len(a) == 2 * n_users, f"record {j} is short -- the .f32 is truncated"
    return a[:n_users].astype(np.float64), a[n_users:].astype(np.float64)


def read_all(path, n_users, n_cand):
    """(base_full, base_tail, G_full, G_tail) with G = per-user GAINS, candidates x users."""
    n, rem = n_records(path, n_users)
    assert rem == 0, f"{os.path.basename(path)} has a torn trailing record ({rem} bytes)"
    assert n == n_cand + 1, (
        f"{os.path.basename(path)} holds {n} records; a complete pass over {n_cand} candidates is "
        f"{n_cand + 1} (record 0 is the base). Refusing to read a partial pass as a complete one.")
    A = np.fromfile(path, dtype=np.float32).reshape(n, 2, n_users).astype(np.float64)
    bf, bt = A[0, 0], A[0, 1]
    return bf, bt, A[1:, 0] - bf[None, :], A[1:, 1] - bt[None, :]


# ============================================================== the state assignment (K.3)
def assign_states(E, rows, tree_asked, tree_outs, pool, weights, seed=SEED):
    """ONE state per user, drawn from the declared deployment mixture. Returns a dict of arrays.

    `tree_asked` / `tree_outs` are the frozen tree's realised interviews for these users (the banked
    validation trajectory), so a `tree` state is a genuine prefix of a trajectory that occurred, with
    the outcomes the user actually gave -- not a re-simulation.
    """
    import policy_resume as PR
    from ndcg_tree import branch_of_per_user
    rows = np.asarray(rows, np.int64)
    n = len(rows)
    rng = np.random.default_rng(seed)
    w = np.asarray([weights[t] for t in STATE_TYPES], np.float64)
    assert (w >= 0).all() and w.sum() > 0, f"bad mixture weights {weights}"
    w = w / w.sum()
    stype = rng.choice(len(STATE_TYPES), size=n, p=w)

    item_seqs = [[] for _ in range(n)]
    conc_seqs = [[] for _ in range(n)]
    hist_q = np.full((n, TOTAL_L - 1), -1, np.int64)     # POOL INDICES of the asked questions
    hist_out = np.full((n, TOTAL_L - 1), -1, np.int64)
    klen = np.zeros(n, np.int64)

    # ---- tree prefixes: a prefix of the user's OWN frozen-tree trajectory --------------------------
    ix = np.flatnonzero(stype == 0)
    if len(ix):
        t = rng.integers(TREE_T_MIN, TREE_T_MAX + 1, size=len(ix))
        for j, u in enumerate(ix):
            tt = int(t[j])
            qq = [int(x) for x in tree_asked[u, :tt]]
            hist_q[u, :tt] = qq
            hist_out[u, :tt] = [int(x) for x in tree_outs[u, :tt]]
            klen[u] = tt
            item_seqs[u] = [pool[q][1] for q in qq if pool[q][0] == 0]
            conc_seqs[u] = [pool[q][1] for q in qq if pool[q][0] == 1]

    # ---- volunteered seeds: R's own audited draw, per k, on the users assigned that k ---------------
    ix = np.flatnonzero(stype == 1)
    vol_k = np.zeros(n, np.int64)
    draws = {}
    if len(ix):
        kk = rng.integers(VOL_K_MIN, VOL_K_MAX + 1, size=len(ix))
        vol_k[ix] = kk
        for k in range(VOL_K_MIN, VOL_K_MAX + 1):
            sub = ix[kk == k]
            if not len(sub):
                continue
            # the SAME function R and R2 published, so the draw is the audited one; the seed is
            # offset per k so two k-groups are not handed the same permutation stream.
            vv, info = PR.draw_volunteered(E, [int(rows[u]) for u in sub], k, seed=seed + 1000 + k)
            draws[str(k)] = info
            for u, v in zip(sub, vv):
                item_seqs[u] = [int(i) for i in v]
                klen[u] = len(v)                      # the asked-history stays EMPTY, by design
        # a volunteered state carries NO asked-history and position 0: the belief moved, nothing was
        # asked. That pairing is exactly what E1 never saw (R2), and it is what deployment presents.

    # ---- random partial interviews: uniform over the 1,000-pool ------------------------------------
    ix = np.flatnonzero(stype == 2)
    if len(ix):
        m = rng.integers(RAND_M_MIN, RAND_M_MAX + 1, size=len(ix))
        for j, u in enumerate(ix):
            mm = int(m[j])
            qq = [int(x) for x in rng.choice(len(pool), size=mm, replace=False)]
            hist_q[u, :mm] = qq
            klen[u] = mm
            item_seqs[u] = [pool[q][1] for q in qq if pool[q][0] == 0]
            conc_seqs[u] = [pool[q][1] for q in qq if pool[q][0] == 1]
        # the answers, by the STRUCTURAL rule off the user's own fold-in half -- the same
        # `branch_of_per_user` the tree branches on, gated against `branch_of` by the caller.
        for j in range(RAND_M_MAX):
            sel = ix[np.asarray([hist_q[u, j] >= 0 for u in ix])]
            if not len(sel):
                continue
            qs = [pool[int(hist_q[u, j])] for u in sel]
            o = branch_of_per_user(E, [int(rows[u]) for u in sel], qs)
            hist_out[sel, j] = np.asarray(o, np.int64)

    assert (hist_out[hist_q >= 0] >= 0).all(), "an asked question in a state has no recorded answer"
    return {"rows": rows, "stype": stype, "klen": klen, "vol_k": vol_k,
            "hist_q": hist_q, "hist_out": hist_out,
            "item_seqs": item_seqs, "conc_seqs": conc_seqs, "vol_draws": draws}


def composition(A, pool):
    """THE MANDATORY PRINT (K.4). State-type mix, prefix-length histogram, users per state type."""
    n = len(A["rows"])
    st = A["stype"]; kl = A["klen"]
    comp = {"n_users": int(n),
            "by_state_type": {t: int((st == i).sum()) for i, t in enumerate(STATE_TYPES)},
            "frac_by_state_type": {t: float((st == i).mean()) for i, t in enumerate(STATE_TYPES)},
            "prefix_length_histogram": {str(int(v)): int(c) for v, c in
                                        zip(*np.unique(kl, return_counts=True))},
            "prefix_length_mean": float(kl.mean()),
            "prefix_length_by_state_type": {
                t: {"mean": float(kl[st == i].mean()) if (st == i).any() else None,
                    "histogram": ({str(int(v)): int(c) for v, c in
                                   zip(*np.unique(kl[st == i], return_counts=True))}
                                  if (st == i).any() else {})}
                for i, t in enumerate(STATE_TYPES)},
            "items_per_user_mean": float(np.mean([len(s) for s in A["item_seqs"]])),
            "concepts_per_user_mean": float(np.mean([len(s) for s in A["conc_seqs"]])),
            "n_users_with_empty_state": int(sum(1 for u in range(n)
                                                if not A["item_seqs"][u] and not A["conc_seqs"][u])),
            "n_distinct_states": len({(tuple(A["item_seqs"][u]), tuple(A["conc_seqs"][u]))
                                      for u in range(n)}),
            "volunteered_draw_diagnostics": A["vol_draws"]}
    return comp


def print_composition(comp):
    log("=" * 106)
    log("COMPOSITION OF THE STATE ASSIGNMENT -- printed BEFORE the first candidate (K.4, mandatory)")
    log(f"  users                {comp['n_users']:,}")
    log(f"  distinct states      {comp['n_distinct_states']:,} "
        f"({comp['n_distinct_states'] / max(comp['n_users'], 1) * 100:.1f}% of users sit somewhere "
        f"no other user sits)")
    for t in STATE_TYPES:
        b = comp["prefix_length_by_state_type"][t]
        log(f"  {t:<5} {comp['by_state_type'][t]:>6,} users "
            f"({comp['frac_by_state_type'][t] * 100:5.1f}%)  prefix mean "
            f"{('%.2f' % b['mean']) if b['mean'] is not None else '--':>5}  hist {b['histogram']}")
    log(f"  prefix length        mean {comp['prefix_length_mean']:.2f}, "
        f"histogram {comp['prefix_length_histogram']}")
    log(f"  folded evidence      {comp['items_per_user_mean']:.2f} items + "
        f"{comp['concepts_per_user_mean']:.2f} concepts per user")
    log(f"  empty states         {comp['n_users_with_empty_state']} "
        f"(a volunteered draw can come up short on a thin fold-in half; counted, not dropped)")
    log("=" * 106)


# ============================================================== the labelling pass
def run_labels(args):
    assert args.i_have_author_goahead, (
        "REFUSING TO LABEL. Pass --i_have_author_goahead only when the author has said go. The K.4 "
        "pilot runs on the author's explicit go and nothing else in Part K runs without one.")
    import torch
    from strategy_channel_suite import (build_greedy_env, contract_selftest, genv_eval, genv_mean,
                                        genv_eval_multi, CERT_TOWER, CERT_CONC)
    from ndcg_tree import build_pool, leak_assert, branch_of_selftest
    from policy_net import load_traj
    np.random.seed(SEED); torch.manual_seed(SEED)
    t00 = time.time()
    os.makedirs(PU_DIR, exist_ok=True)

    log("#" * 106)
    log("# PART K -- PER-USER GAIN LABELLING. VALIDATION ONLY, FIT HALF ONLY.")
    log("# The 10,000 canonical TEST users are NOT built, NOT loaded, NOT touched.")
    log("# The 2,000 HOLDOUT users are NOT labelled: every behavioural claim is proven on them.")
    log("#" * 106)

    E = build_greedy_env(args.snapshot, "validation", args.concept_ckpt, "pu_labels")
    contract_selftest(E, E["rows"])
    leak_assert(E, E["rows"], "pu_labels/val")
    rows_all = E["rows"]
    assert len(rows_all) == N_VAL, f"validation cohort is {len(rows_all)} users, expected {N_VAL}"

    # ---- GATE 1: the canonical snap. Before anything is trusted. ---------------------------------
    f0, t0_ = genv_mean(E, [], rows_all)
    assert round(f0, 8) == VAL_M0_FULL, (
        f"CANONICAL SNAP FAILED: m=0 full is {f0!r}, canonical {VAL_M0_FULL}. STOP -- the ruler is "
        f"wrong and nothing downstream means anything.")
    log(f"[gate] canonical m=0 snap full={f0!r} tail={t0_!r} -- exact")

    # ---- the pool, byte-identical to the tree's -----------------------------------------------
    Dva = load_traj(train=False)
    pool = Dva["pool"]
    pool_e, _ip, _cp = build_pool(E, N_ITEMS_POOL, N_CONC_POOL)
    assert pool_e == pool, "the env's pool is not the trajectory bank's pool"
    assert len(pool) == N_ITEMS_POOL + N_CONC_POOL
    assert list(np.asarray(Dva["uid"], np.int64)) == list(np.asarray(rows_all, np.int64)), \
        "the validation trajectory rows are not the env's rows, in order"
    branch_of_selftest(E, rows_all[:2000], [pool[0], pool[500], (1, 114)])
    pool_sha1 = sha1_of([[int(a), int(b)] for a, b in pool])

    # ---- GATE 2: the banked headline. A free contract check on the whole chain. -------------------
    static_seq = [(int(a), int(b)) for a, b in json.load(open(STATIC_JSON))["sequence"]]
    sf, _stl = genv_eval(E, static_seq, rows_all)
    got = float(np.nanmean(sf))
    assert abs(got - STATIC_VAL_FULL) < 1e-12, (
        f"BANKED-HEADLINE ASSERT FAILED: greedy_static_policyB_combined rescores to {got!r}, "
        f"banked {STATIC_VAL_FULL}. The env is not the env those numbers were measured on.")
    log(f"[gate] greedy_static_policyB_combined K=16 reproduces EXACTLY: full={got!r}")

    # ---- the firewall split, fixed once ----------------------------------------------------------
    fit, hold, split_info = fit_holdout_split(rows_all)
    log(f"[firewall] {len(fit):,} FIT / {len(hold):,} HOLDOUT by user, seed {SPLIT_SEED}; "
        f"fit sha1 {split_info['fit_sha1'][:12]} holdout sha1 {split_info['holdout_sha1'][:12]}")
    if args.pilot:
        cohort = pilot_cohort(fit)
        log(f"[pilot] K.4: {len(cohort):,} of the {len(fit):,} FIT users, seeded draw "
            f"(rng {PILOT_SEED}). This is the ONLY user reduction in this file and it is the "
            f"pre-registered pilot, not a convenience.")
    else:
        cohort = np.asarray(fit, np.int64)
        log(f"[full] all {len(cohort):,} FIT users")
    assert not np.intersect1d(cohort, hold).size, "the labelling cohort touches HOLDOUT"
    n_users = len(cohort)
    rows = [int(r) for r in cohort]

    # ---- the state assignment --------------------------------------------------------------------
    weights = {"tree": args.mix_tree, "vol": args.mix_vol, "rand": args.mix_rand}
    pos_in_all = {int(r): j for j, r in enumerate(rows_all)}
    sub = np.asarray([pos_in_all[int(r)] for r in rows], np.int64)
    A = assign_states(E, cohort, np.asarray(Dva["label_idx"], np.int64)[sub],
                      np.asarray(Dva["outcome"], np.int64)[sub], pool, weights, seed=args.seed)
    comp = composition(A, pool)
    print_composition(comp)

    # ---- the contract, and the resume decision ----------------------------------------------------
    contract = {"v": 1, "part": "K", "split": "validation", "cohort": "FIT",
                "pilot": bool(args.pilot), "n_users": int(n_users),
                "users_sha1": sha1_of([int(x) for x in cohort]),
                "tower": E["tower"], "conc_ckpt": E["conc_ckpt"],
                "pool_sha1": pool_sha1, "pool_contract": E["pool_contract"],
                "n_candidates": len(pool), "objective": "full_and_tail_ndcg10",
                "state_seed": int(args.seed), "split_info": split_info,
                "mixture_weights": weights,
                "state_assignment_sha1": sha1_of(
                    [[int(A["stype"][u]), int(A["klen"][u]),
                      [int(i) for i in A["item_seqs"][u]], [int(c) for c in A["conc_seqs"][u]]]
                     for u in range(n_users)])}
    gpath = os.path.join(PU_DIR, GAINS_F32)
    spath = os.path.join(PU_DIR, SIDECAR)
    npzp = os.path.join(PU_DIR, STATE_NPZ)
    done = 0
    if os.path.exists(spath):
        banked = json.load(open(spath))
        if resume_ok(banked.get("contract"), contract, log, what="pu_labels sidecar"):
            done = truncate_torn(gpath, n_users)
            assert done == banked.get("n_records_on_disk", done), (
                f"the sidecar claims {banked.get('n_records_on_disk')} records, the .f32 holds "
                f"{done}. Refusing to resume a disagreement.")
            log(f"[resume] contract matches; {max(done - 1, 0)} of {len(pool)} candidates banked")
        else:
            raise SystemExit(
                "RESUME REFUSED: an artifact under a DIFFERENT contract is already in "
                f"{PU_DIR}. Move it aside deliberately -- this script will not overwrite a banked "
                "measurement, and it will not merge two contracts (HARD RULE 11b).")
    else:
        assert not os.path.exists(gpath), (
            f"{GAINS_F32} exists with no sidecar -- an unsigned measurement is not resumable.")

    # ---- record 0: the BASE, and the state latent the trainer will consume ------------------------
    item_seqs = [list(map(int, A["item_seqs"][u])) for u in range(n_users)]
    conc_seqs = [list(map(int, A["conc_seqs"][u])) for u in range(n_users)]
    t_base = time.time()
    bf, bt, Zi_base, z_state = genv_eval_multi(E, rows, item_seqs, conc_seqs,
                                               return_Zi=True, return_z=True)
    log(f"[base] state-alone NDCG: full {float(np.nanmean(bf)):.6f} tail {float(np.nanmean(bt)):.6f} "
        f"({time.time() - t_base:.1f}s, cost model says {cost_model(n_users):.2f}s)")
    if done == 0:
        append_record(gpath, bf, bt)
        done = 1
    else:
        b0f, b0t = read_record(gpath, 0, n_users)
        assert np.allclose(b0f, bf, atol=0, rtol=0) and np.allclose(b0t, bt, atol=0, rtol=0), (
            "the banked BASE record does not reproduce bit-for-bit on this env. The contract matched "
            "but the numbers did not -- that is a harness bug, not a resume.")
        log("[resume] the banked base record reproduces BIT-FOR-BIT")

    # the state latent + descriptors, so the trainer needs no environment at all
    from policy_net import build_phi
    phi_t, norm = build_phi(E, pool)
    np.savez_compressed(
        npzp, rows=np.asarray(rows, np.int64), stype=A["stype"].astype(np.int8),
        klen=A["klen"].astype(np.int16), vol_k=A["vol_k"].astype(np.int16),
        hist_q=A["hist_q"].astype(np.int32), hist_out=A["hist_out"].astype(np.int8),
        z=z_state.numpy().astype(np.float32), base_full=bf.astype(np.float32),
        base_tail=bt.astype(np.float32), phi=phi_t.astype(np.float32),
        norm_mu=norm[0].astype(np.float32), norm_sd=norm[1].astype(np.float32),
        pool=np.asarray([[int(a), int(b)] for a, b in pool], np.int64),
        item_pad=_pad([np.asarray(s, np.int64) for s in item_seqs]),
        conc_pad=_pad([np.asarray(s, np.int64) for s in conc_seqs]))
    log(f"[state] wrote {STATE_NPZ}: z {tuple(z_state.shape)}, phi {phi_t.shape} -- the trainer "
        f"needs no environment")

    def _flush(n_done, extra=None):
        blob = {"analysis": "pu_labels_part_K", "design_sheet":
                "docs/design/PAPER_B_POLICY_DESIGN.md PART K", "citable": True,
                "test_touched": False, "contract": contract,
                "n_records_on_disk": int(n_done),
                "n_candidates_done": int(n_done - 1), "n_candidates": len(pool),
                "record_layout": ("append-only float32: record j = n_users FULL then n_users TAIL, "
                                  "in the cohort's fixed row order; record 0 is the BASE (the state "
                                  "alone). bytes == (1 + n_candidates_done) * 2 * n_users * 4"),
                "gains_file": GAINS_F32, "state_file": STATE_NPZ,
                "base_full": float(np.nanmean(bf)), "base_tail": float(np.nanmean(bt)),
                "composition": comp,
                "state_assignment": [
                    {"uid": int(rows[u]), "state_type": STATE_TYPES[int(A["stype"][u])],
                     "k": int(A["klen"][u]),
                     "hist_q": [int(x) for x in A["hist_q"][u] if x >= 0],
                     "hist_out": [int(x) for x in A["hist_out"][u] if x >= 0],
                     "items": item_seqs[u], "concepts": conc_seqs[u]} for u in range(n_users)],
                "cost_model": {"formula": "s_per_eval = 0.0857 + 1.57 * n_users/1000",
                               "predicted_s_per_eval": cost_model(n_users)},
                "minutes": round((time.time() - t00) / 60, 2)}
        if extra:
            blob.update(extra)
        atomic_write(spath, blob)

    _flush(done)

    # ---- the sweep: every candidate, every user, one pass each ------------------------------------
    n_cand = len(pool)
    log(f"[sweep] {n_cand} candidates x {n_users:,} users; {n_cand - (done - 1)} candidates left; "
        f"projection {(n_cand - (done - 1)) * cost_model(n_users) / 60:.1f} min at the model rate")
    t_sweep = time.time(); n_ev = 0
    already = np.zeros(n_cand, np.int64)
    for ci in range(done - 1, n_cand):
        ch, qid = int(pool[ci][0]), int(pool[ci][1])
        t0 = time.time()
        if ch == 0:
            cseq = conc_seqs
            iseq, n_dup = [], 0
            for u in range(n_users):
                s = item_seqs[u]
                if qid in s:
                    n_dup += 1
                    iseq.append(s)                # a re-ask buys nothing: gain is EXACTLY 0
                else:
                    iseq.append(s + [qid])
            f, t = genv_eval_multi(E, rows, iseq, cseq)
        else:
            iseq, n_dup = item_seqs, 0
            cseq = []
            for u in range(n_users):
                s = conc_seqs[u]
                if qid in s:
                    n_dup += 1
                    cseq.append(s)
                else:
                    cseq.append(s + [qid])
            # every CONCEPT candidate leaves the item prefix untouched, so the whole 500-concept
            # block shares ONE tower fold -- exact, not an approximation (genv_eval_multi's own
            # documented optimisation, handed the baseline pass's Zi).
            f, t = genv_eval_multi(E, rows, iseq, cseq, Zi_items=Zi_base)
        append_record(gpath, f, t)
        already[ci] = n_dup
        n_ev += 1
        if (ci + 1) % 25 == 0 or ci + 1 == n_cand:
            el = time.time() - t_sweep
            _flush(ci + 2, {"measured_s_per_eval": el / max(n_ev, 1)})
            g = float(np.nanmean(f - bf))
            log(f"[sweep {ci + 1}/{n_cand}] {'item' if ch == 0 else 'conc'} {qid} "
                f"mean gain {g:+.6f} | {el / 60:.1f}m, measured {el / max(n_ev, 1):.2f} s/eval "
                f"(model {cost_model(n_users):.2f}) | eta "
                f"{(n_cand - ci - 1) * el / max(n_ev, 1) / 60:.1f}m")

    # ---- the finished artifact -------------------------------------------------------------------
    bf2, bt2, Gf, Gt = read_all(gpath, n_users, n_cand)
    nz = (np.abs(Gf) > 0).sum(0)
    _flush(n_cand + 1, {
        "complete": True,
        "measured_s_per_eval": (time.time() - t_sweep) / max(n_ev, 1),
        "gain_summary": {
            "mean_gain_full": float(np.nanmean(Gf)), "mean_gain_tail": float(np.nanmean(Gt)),
            "best_candidate_mean_gain_full": float(np.nanmax(np.nanmean(Gf, 1))),
            "per_user_nnz_mean": float(nz.mean()), "per_user_nnz_median": float(np.median(nz)),
            "per_user_nnz_min": int(nz.min()), "per_user_nnz_max": int(nz.max()),
            "frac_candidates_exactly_zero": float((np.abs(Gf) == 0).mean()),
            "n_users_with_all_zero_gains": int((nz == 0).sum()),
            "note": ("an EXACT zero is an unanswerable or already-asked candidate: a refusal / UNSEEN "
                     "fold moves nothing, so answerability is taught by the label without being "
                     "encoded anywhere (K.1).")},
        "candidate_already_in_state_counts": [int(x) for x in already]})
    log(f"[done] {n_cand} candidates in {(time.time() - t_sweep) / 60:.1f} min; per-user nonzero "
        f"gains mean {nz.mean():.1f} of {n_cand}; {float((np.abs(Gf) == 0).mean()) * 100:.1f}% of "
        f"all (candidate, user) cells are EXACTLY zero -- that is the answerability signal")
    return True


def _pad(seqs):
    w = max((len(s) for s in seqs), default=0)
    out = np.full((len(seqs), max(w, 1)), -1, np.int64)
    for j, s in enumerate(seqs):
        out[j, :len(s)] = s
    return out


# ============================================================== compose check (no environment)
def compose_check(args):
    """Re-print the composition of a banked assignment and the per-user nnz. Reads only; no env."""
    spath = os.path.join(PU_DIR, SIDECAR)
    assert os.path.exists(spath), f"no {SIDECAR} in {PU_DIR} -- nothing to compose-check"
    b = json.load(open(spath))
    assert b["test_touched"] is False
    c = b["composition"]
    print_composition(c)
    n_users = int(b["contract"]["n_users"]); n_cand = int(b["contract"]["n_candidates"])
    gpath = os.path.join(PU_DIR, GAINS_F32)
    assert os.path.exists(gpath), f"no {GAINS_F32} in {PU_DIR} -- the sidecar has no measurement"
    n, rem = n_records(gpath, n_users)
    log(f"[compose] {os.path.getsize(gpath):,} bytes = {n} records (remainder {rem}); the arithmetic "
        f"says (1 + {n - 1}) * 2 * {n_users} * 4 = {(1 + n - 1) * rec_bytes(n_users):,}")
    assert rem == 0, "torn trailing record"
    ok = bool(b.get("complete")) and n == n_cand + 1
    if "gain_summary" in b:
        g = b["gain_summary"]
        log(f"[compose] per-user nonzero gains: mean {g['per_user_nnz_mean']:.1f} "
            f"median {g['per_user_nnz_median']:.0f} min {g['per_user_nnz_min']} "
            f"max {g['per_user_nnz_max']} of {n_cand}; "
            f"{g['n_users_with_all_zero_gains']} users have an ALL-ZERO gain vector")
        assert g["n_users_with_all_zero_gains"] < n_users, (
            "every user has an all-zero gain vector -- there is nothing to distil and the pass is "
            "broken, not informative")
    if not ok:
        raise SystemExit(f"COMPOSITION CHECK FAILED: the pass is incomplete "
                         f"({n - 1} of {n_cand} candidates). Later steps are NOT run.")
    log("[compose] PASS -- the labelling pass is complete and its record arithmetic checks out")
    return True


# ============================================================== cost (builds nothing)
def cost_only(args):
    n_pilot, n_full = PILOT_N, N_FIT
    n_cand = N_ITEMS_POOL + N_CONC_POOL
    log("COST MODEL (measured on THIS box, phase 3): s_per_eval = 0.0857 + 1.57 * n_users/1000")
    for label, n in (("pilot  (K.4)", n_pilot), ("full round", n_full)):
        s = cost_model(n)
        tot = (n_cand + 1) * s
        log(f"  {label}: n={n:,} users x {n_cand} candidates (+1 base) -> {s:.3f} s/eval -> "
            f"{tot / 60:.1f} min ({tot / 3600:.2f} h)")
    log(f"  .f32 size: pilot {(n_cand + 1) * rec_bytes(n_pilot) / 1e6:.1f} MB, "
        f"full {(n_cand + 1) * rec_bytes(n_full) / 1e6:.1f} MB")
    log("  costs are a property of THIS box; the run re-measures its own rate and prints "
        "measured-vs-model, so a wrong model is wrong out loud")
    return True


# ============================================================== selftests (pure: no data, no env)
def selftest():
    import tempfile
    ok = []

    # (1) the firewall split is deterministic for seed 98765, disjoint, and exhaustive
    rows = np.arange(N_VAL) * 3 + 7                       # any canonical row vector
    f1, h1, i1 = fit_holdout_split(rows)
    f2, h2, i2 = fit_holdout_split(rows)
    assert np.array_equal(f1, f2) and np.array_equal(h1, h2), "the firewall split is not deterministic"
    assert i1 == i2
    assert not np.intersect1d(f1, h1).size, "FIT and HOLDOUT overlap"
    assert len(f1) == N_FIT and len(h1) == N_HOLDOUT
    assert np.array_equal(np.sort(np.concatenate([f1, h1])), np.sort(rows))
    f3, _h3, _ = fit_holdout_split(rows, seed=SPLIT_SEED + 1)
    assert not np.array_equal(f1, f3), "the split does not depend on the seed -- it is not a draw"
    p1, p2 = pilot_cohort(f1), pilot_cohort(f1)
    assert np.array_equal(p1, p2) and len(p1) == PILOT_N
    assert np.isin(p1, f1).all(), "the pilot cohort leaves FIT"
    assert not np.intersect1d(p1, h1).size, "the pilot cohort touches HOLDOUT"
    ok.append("firewall split deterministic, disjoint, exhaustive; pilot inside FIT")

    # (2) the resume contract refuses a mutated pool field, and refuses a MISSING one
    base = {"pool_sha1": "a" * 40, "n_users": 1000, "tower": "t.pt"}
    assert resume_ok(dict(base), dict(base), lambda *a: None)
    mut = dict(base); mut["pool_sha1"] = "b" * 40
    assert not resume_ok(mut, dict(base), lambda *a: None), "a mutated pool sha1 was accepted"
    miss = {k: v for k, v in base.items() if k != "pool_sha1"}
    assert not resume_ok(miss, dict(base), lambda *a: None), "a MISSING pool field was accepted"
    extra = dict(base); extra["x"] = 1
    assert not resume_ok(extra, dict(base), lambda *a: None), "an unknown field was accepted"
    ok.append("resume refuses a mutated, a missing and an unknown contract field")

    # (3) the .f32 record arithmetic, on real bytes
    n_u, n_c = 37, 5
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "t.f32")
        rng = np.random.default_rng(0)
        bf = rng.random(n_u); bt = rng.random(n_u)
        append_record(p, bf, bt)
        F, T = [], []
        for c in range(n_c):
            f = bf + (c + 1) * 0.01; t = bt + (c + 1) * 0.02
            append_record(p, f, t); F.append(f); T.append(t)
        assert os.path.getsize(p) == (1 + n_c) * 2 * n_u * 4, "the record arithmetic is wrong"
        n, rem = n_records(p, n_u)
        assert n == n_c + 1 and rem == 0
        g0f, g0t = read_record(p, 0, n_u)
        assert np.allclose(g0f, bf.astype(np.float32)) and np.allclose(g0t, bt.astype(np.float32))
        rbf, rbt, Gf, Gt = read_all(p, n_u, n_c)
        assert Gf.shape == (n_c, n_u) and Gt.shape == (n_c, n_u)
        assert np.allclose(Gf, np.asarray(F, np.float32) - bf.astype(np.float32), atol=1e-6)
        assert np.allclose(Gt, np.asarray(T, np.float32) - bt.astype(np.float32), atol=1e-6)
        # a torn tail is DETECTED and truncated to whole records, costing exactly one candidate
        with open(p, "ab") as fh:
            fh.write(b"\x00" * 13)
        assert n_records(p, n_u)[1] == 13
        assert truncate_torn(p, n_u) == n_c + 1
        assert n_records(p, n_u) == (n_c + 1, 0)
        try:
            read_all(p, n_u, n_c + 3)
            raise AssertionError("read_all accepted a partial pass as a complete one")
        except AssertionError as e:
            assert "Refusing to read a partial pass" in str(e)
    ok.append("append-only .f32: bytes == (1 + n_done) * 2 * n_users * 4, base subtraction, "
              "torn-tail truncation, partial-pass refusal")

    for s in ok:
        log(f"[selftest] PASS: {s}")
    log("[selftest] pu_labels: nothing was loaded, folded, scored or fitted")
    return True


def main():
    ap = argparse.ArgumentParser(
        description="Part K step 1: per-user gain labelling. See docs/design/PAPER_B_POLICY_DESIGN.md")
    ap.add_argument("--selftest", action="store_true",
                    help="pure checks: the firewall split, the resume contract, the .f32 arithmetic. "
                         "No data, no environment, no fold, no fit.")
    ap.add_argument("--cost_only", action="store_true",
                    help="print the projected wall time from the measured per-eval model. Builds "
                         "NO environment and touches nothing.")
    ap.add_argument("--mode", default="label", choices=["label", "compose"],
                    help="compose = re-print the banked composition and check the record "
                         "arithmetic. Reads only; no environment.")
    ap.add_argument("--pilot", action="store_true",
                    help=f"K.4: {PILOT_N} FIT users, one state each, all 1,000 candidates. The ONLY "
                         f"authorised user reduction in this file.")
    ap.add_argument("--split", default="validation",
                    help="validation ONLY. Any other value is refused: the single unspent TEST look "
                         "is reserved for experiment Q (K.2).")
    ap.add_argument("--mix_tree", type=float, default=1 / 3,
                    help="mixture weight for tree-trajectory prefixes (K.3). Declared, recorded in "
                         "the artifact, never tuned post hoc.")
    ap.add_argument("--mix_vol", type=float, default=1 / 3,
                    help="mixture weight for volunteered-evidence seeds k in {1..8}")
    ap.add_argument("--mix_rand", type=float, default=1 / 3,
                    help="mixture weight for random partial interviews")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--snapshot", default=None)
    ap.add_argument("--concept_ckpt", default=None)
    ap.add_argument("--threads", type=int, default=None,
                    help="thread count, consumed by strategy_channel_suite AT IMPORT")
    ap.add_argument("--full_threads", action="store_true",
                    help="consumed by strategy_channel_suite AT IMPORT (it reads sys.argv)")
    ap.add_argument("--i_have_author_goahead", action="store_true",
                    help="required to label. The K.4 pilot runs on the author's explicit go.")
    args = ap.parse_args()
    if args.split != "validation":
        raise SystemExit(
            f"REFUSED: --split {args.split!r}. This file builds VALIDATION environments only. The "
            f"10,000 canonical TEST users are not built, not loaded, not touched; the single unspent "
            f"test look is reserved for experiment Q (design sheet K.2).")
    if args.selftest:
        return selftest()
    if args.cost_only:
        return cost_only(args)
    if args.mode == "compose":
        return compose_check(args)
    from strategy_channel_suite import CERT_TOWER, CERT_CONC
    args.snapshot = args.snapshot or CERT_TOWER
    args.concept_ckpt = args.concept_ckpt or CERT_CONC
    return run_labels(args)


if __name__ == "__main__":
    main()
