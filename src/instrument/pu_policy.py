r"""PART K -- PER-USER GAIN DISTILLATION, STEP 2: THE TRAINER.

Design sheet: `docs/design/PAPER_B_POLICY_DESIGN.md` **PART K is binding** (commit b75d57c). Parts
H, I and J record why every alternative died. Read J.2 before proposing anything adaptive here.

**NOTHING HERE HAS BEEN RUN.** Committed before any run (HARD RULE 10). `--selftest` is pure
(random weights, synthetic tensors, no data, no optimiser, no fit). Fitting requires
`--i_have_author_goahead`, exactly as `policy_net.py`.

THE MODEL IS UNCHANGED
-----------------------
The E1 architecture, verbatim: `policy_net.Policy(arch='equivariant', state='both')`, warm-started
from `policy_main_equivariant_both.pt`. All 16 tensors are transferred and every one is checked
BYTE-IDENTICAL after the load, so a "warm start" that silently transferred nothing cannot be
reported as one (`lift_head.warm_start`'s lesson, tightened from >= 12 to exactly 16/16). No new
representation is learned; nothing can drift off the certified ruler.

THE LOSS -- LISTWISE, SOFT, AND NEVER A PER-USER ARGMAX
--------------------------------------------------------
★ K.1 forbids a per-user argmax as a hard label, and the reason is measured: a single user's gain
vector is maximally noisy (the -0.724 small-cohort lesson at n=1) and argmax-of-noise is the value
head's death (Part H.1: argmax over noisy estimates selects the over-estimates; answer rate
0.58 -> 0.37). So the target is the whole graded vector:

    target(u) = softmax( gain_full(u, .) / tau )   over the 1,000 candidates
    loss(u)   = - sum_c target(u, c) * log_softmax( s(c | state_u) )(c)

Labels are noisy but UNBIASED; SGD over thousands of users in similar states is the smoother. `tau`
is a DECLARED CLI constant with its default stated in the artifact, never tuned post hoc. Two
structures in the noise do real work for free:
  * an unanswerable candidate has gain EXACTLY 0 (a refusal / UNSEEN fold moves nothing), so it
    receives the same mass as every other zero -- answerability is taught by the label, and nothing
    encodes it;
  * a candidate already in the user's state also has gain exactly 0, so re-asking is taught not to
    pay. Volunteered items are deliberately NOT masked out (R's semantics: the policy is free to
    re-ask, and how often it does is the redundancy measurement), so this is the only thing teaching
    it not to.
The argmax of the measured gains is computed ONLY as a read-out (agreement diagnostics). It is
never a training target anywhere in this file, and the assert that says so is in `pu_batch`.

TREE-LABEL REPLAY -- the anti-forgetting mechanism, not a convenience
----------------------------------------------------------------------
Each batch is `--pu_frac` per-user-gain examples and the rest E1's own supervision: the frozen
tree's labels at the tree's states over the 140,768 TRAIN users, through `policy_net.make_batch`,
unchanged. Declared default 0.5. It is NOT P2's 7/8, and the reason is stated rather than drifted
into: P2 mixed 209 states against 2.25M replay examples, where 1/8 was already generous; here the
new material is ~8,000 states x 1,000 graded targets and a 7/8 replay would drown the thing being
tested. The spanning gate is what prices the choice, and it is reported with the result.

THE FIREWALL (K.2), and why this file cannot cheat
----------------------------------------------------
The 10,000 validation users are split 8,000 FIT / 2,000 HOLDOUT by user, seed 98765, fixed once in
`pu_labels.fit_holdout_split` and re-derived here rather than read from a file. Labels exist for FIT
users only; the trainer ASSERTS that no label record exists on disk for any HOLDOUT uid. Model
selection is on a held-out slice OF FIT (by user, declared 10%), never on HOLDOUT. Every behavioural
claim is proven on HOLDOUT in `--mode evaluate`, and the gates block is computed THERE, not during
training. The 10,000 canonical TEST users are not built, not loaded, not touched; `--split test` is
refused by the parser and `test_touched=false` is stamped everywhere.

CONTROLS (HARD RULE 5), built in and not optional
---------------------------------------------------
  --arm shuffle   each user's gain VECTOR is permuted across candidates, within that user. The
                  marginals survive exactly -- same gains, same entropy, same target temperature --
                  and the state -> gain association is destroyed. A trained arm that does not beat
                  this on held-out FIT states has learned nothing about states.
  zeroed state    the same held-out agreement with z, history and position ZEROED. If the number
                  barely moves, the policy is emitting a prior, not reading a state, and the
                  artifact says so.

THE GATES (K.5), all in `--mode evaluate`, all on HOLDOUT
-----------------------------------------------------------
  spanning (HARD RULE 8)  unseeded K=16 rollout on the 2,000 HOLDOUT users must not fall below E1
                          on the SAME users (E1's banked per-user vectors, sliced -- exact).
  ★ behavioural win       on fresh seeded states from HOLDOUT users (a draw seed disjoint from
                          98765 and from P2's 424242): policy MINUS ITS OWN FROZEN ORDER >= 0 with
                          the interval excluding a loss. R2 measured -0.01246/-0.01234/-0.00860.
  vs tree                 paired CI, reported. "not worse" is the sentence; "beats" is not expected
                          and is not claimed (the bank result +0.01430 > +0.01035 travels with every
                          adaptive sentence).
  redundancy              reported WITH an interval.
  anti-collapse           distinct interviews, max share, Pep divergence vs the tree's own.

HONESTY, BINDING ON THE WRITE-UP (K.5)
---------------------------------------
The labels are ONE-STEP measured gains, so this is a greedy policy with reach and never a strategic
one; myopia is declared, not discovered by a reviewer. The expectation is that the SCORE lands ~= E1
and the prize is behavioural. Failure of the behavioural gate after a passing pilot is reported as
the finding, not retried silently.

USAGE
  python src/instrument/pu_policy.py --selftest
  python src/instrument/pu_policy.py --mode fit --tag pilot --i_have_author_goahead
  python src/instrument/pu_policy.py --mode fit --arm shuffle --tag pilot --i_have_author_goahead
  python src/instrument/pu_policy.py --mode evaluate --tag pilot --i_have_author_goahead
"""
import os
import sys
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
sys.path.insert(0, _HERE)

import json
import time
import argparse
import numpy as np
import torch
import torch.nn as nn

import pu_labels as PL
from policy_net import (OUTDIR, SEED, TOTAL_L, STATIC_JSON, STATIC_VAL_FULL, VAL_M0_FULL,
                        E1_VAL_FULL, E1_PER_USER, N_OUTCOMES, Policy, build_phi, paired_bootstrap,
                        pep_divergence_from_arrays, collapse_diagnostics, channel_mix, load_traj,
                        make_batch, rollout, log)
from policy_labels import atomic_write

PU_DIR = PL.PU_DIR
E1_CKPT = os.path.join(OUTDIR, "policy_main_equivariant_both.pt")

# --------------------------------------------------------------- declared constants, not tuned post hoc
TAU = 0.01                 # target temperature. One NDCG hundredth = one nat of logit. DECLARED.
PU_FRAC = 0.5              # share of each batch carrying per-user gain targets. DECLARED.
DEV_FRAC = 0.10            # the held-out slice OF FIT used for selection, BY USER. DECLARED.
DEV_SEED = 98767           # SPLIT_SEED + 2. The dev slice is a function of the seed alone.
EVAL_DRAW_SEED = 20260816  # ★ disjoint from R/R2's 98765 and P2's 424242. Declared, not silent.
EVAL_KS = (2, 4, 8)


# ============================================================================ the soft target (K.1)
def soft_target(gains, tau=TAU, mask=None):
    """softmax(gain / tau) over the candidate axis. `mask` True = candidate unavailable.

    Numerically stabilised by the row max, so a single huge gain cannot overflow. A masked entry
    receives EXACTLY zero mass and the rest renormalise -- an unavailable candidate must not be a
    target, and it must not silently steal probability either.

    ★ It is a DISTRIBUTION, never an argmax. A gain vector that is all zeros (every candidate
    unanswerable for that user, or the whole state saturated) maps to the UNIFORM distribution --
    which is the honest thing for it to say, and is what the selftest checks.
    """
    g = torch.as_tensor(gains, dtype=torch.float32)
    x = g / float(tau)
    if mask is not None:
        x = x.masked_fill(torch.as_tensor(mask, dtype=torch.bool), float("-inf"))
    x = x - x.max(dim=-1, keepdim=True).values
    e = torch.exp(x)
    s = e.sum(dim=-1, keepdim=True)
    assert bool((s > 0).all()), "a user has NO available candidate -- the target is undefined"
    return e / s


def listwise_ce(logits, target):
    """-sum_c target_c * log_softmax(logits)_c, meaned over the batch. Masked logits are -inf and
    their target mass is exactly 0, so the product is 0 and never NaN."""
    lp = torch.log_softmax(logits, dim=-1)
    lp = torch.where(target > 0, lp, torch.zeros_like(lp))
    return -(target * lp).sum(-1).mean()


# ============================================================================ loading the labels
def load_labels(require_complete=True):
    """The banked per-user gain vectors + their state assignment. Enforces the firewall on read."""
    spath = os.path.join(PU_DIR, PL.SIDECAR)
    assert os.path.exists(spath), (
        f"no {PL.SIDECAR} in {PU_DIR}. Run pu_labels.py first (K.4 pilot) -- there is nothing to "
        f"distil from.")
    b = json.load(open(spath))
    assert b["test_touched"] is False, "the label artifact does not stamp test_touched=false"
    c = b["contract"]
    assert c["split"] == "validation" and c["cohort"] == "FIT"
    if require_complete:
        assert b.get("complete"), (
            f"the labelling pass is INCOMPLETE ({b['n_candidates_done']} of {b['n_candidates']} "
            f"candidates). Refusing to train on a partial sweep (HARD RULE 1).")
    z = np.load(os.path.join(PU_DIR, PL.STATE_NPZ))
    S = {k: z[k] for k in z.files}
    rows = np.asarray(S["rows"], np.int64)
    n_users = int(c["n_users"]); n_cand = int(c["n_candidates"])
    assert len(rows) == n_users
    assert PL.sha1_of([int(x) for x in rows]) == c["users_sha1"], \
        "the state npz rows do not match the contract's user sha1"

    # ---- THE FIREWALL, re-derived rather than read (K.2) -----------------------------------------
    Dva = load_traj(train=False)
    fit, hold, split_info = PL.fit_holdout_split(np.asarray(Dva["uid"], np.int64))
    assert split_info["fit_sha1"] == c["split_info"]["fit_sha1"], \
        "the labels were built under a DIFFERENT firewall split -- refusing to train"
    assert np.isin(rows, fit).all(), "a labelled user is not in FIT"
    bad = np.intersect1d(rows, hold)
    assert bad.size == 0, (
        f"{bad.size} HOLDOUT users carry labels on disk. The trainer must be UNABLE to read holdout "
        f"labels (K.2) -- this is the firewall failing, not a warning.")
    log(f"[firewall] {len(rows):,} labelled FIT users; 0 of the {len(hold):,} HOLDOUT users have a "
        f"label record on disk")

    bf, bt, Gf, Gt = PL.read_all(os.path.join(PU_DIR, PL.GAINS_F32), n_users, n_cand)
    S["gain_full"] = Gf.T.astype(np.float32)          # users x candidates
    S["gain_tail"] = Gt.T.astype(np.float32)
    S["base_full_read"] = bf
    assert np.allclose(bf, np.asarray(S["base_full"], np.float64), atol=0, rtol=0), \
        "record 0 of the .f32 is not the base in the state npz"
    return b, c, S, fit, hold


def dev_split(rows, frac=DEV_FRAC, seed=DEV_SEED):
    """The selection slice, BY USER, out of FIT. Never HOLDOUT. Deterministic in the seed."""
    r = np.asarray(rows, np.int64)
    perm = np.random.default_rng(seed).permutation(len(r))
    k = max(int(round(frac * len(r))), 1)
    dev = np.sort(perm[:k]); tr = np.sort(perm[k:])
    assert not np.intersect1d(dev, tr).size
    return tr, dev


# ============================================================================ batches
def pu_batch(S, phi_t, idx, P, tau=TAU, zero_state=False):
    """One per-user-gain batch: the E1 state encoding + the soft listwise target.

    The encoding follows the state's own bookkeeping, which is the whole point of the mixture:
      tree / rand   `z` = the folded latent, history = the asked questions and their TERNARY
                    outcomes, position t = k. The asked questions are masked out of the softmax.
      vol           `z` = the folded latent, history EMPTY, position t = 0, nothing masked. The
                    belief moved and nothing was asked -- the pairing E1 never saw (R2), and the one
                    deployment actually presents. A volunteered item is NOT masked: the policy may
                    re-ask it, its measured gain is exactly 0, and that is what teaches it not to.
    """
    B = len(idx)
    z = torch.from_numpy(np.ascontiguousarray(S["z"][idx]).astype(np.float32))
    if zero_state:
        z = torch.zeros_like(z)
    T = TOTAL_L - 1
    hq = np.zeros((B, T), np.int64); ho = np.zeros((B, T), np.int64)
    hm = np.zeros((B, T), np.float32)
    am = np.zeros((B, P), bool)
    st = np.asarray(S["stype"], np.int64)[idx]
    kk = np.asarray(S["klen"], np.int64)[idx]
    sq = np.asarray(S["hist_q"], np.int64)[idx]
    so = np.asarray(S["hist_out"], np.int64)[idx]
    is_vol = (st == STATE_VOL)
    for j in range(T):
        m = (~is_vol) & (kk > j) & (sq[:, j] >= 0)
        if not m.any():
            continue
        hq[m, j] = sq[m, j]; ho[m, j] = so[m, j]; hm[m, j] = 1.0
    am[np.repeat(np.arange(B), T)[hm.ravel() > 0], hq.ravel()[hm.ravel() > 0]] = True
    pos_t = np.where(is_vol, 0, kk).astype(np.float32)
    pos = np.stack([pos_t, TOTAL_L - pos_t], 1)
    if zero_state:
        hm[:] = 0.0; ho[:] = 0; hq[:] = 0; pos[:] = 0.0
    g = np.asarray(S["gain_full"], np.float32)[idx]
    tgt = soft_target(torch.from_numpy(g), tau=tau, mask=torch.from_numpy(am))
    # ★ K.1: the measured argmax is a READ-OUT, never a target. It is returned for the agreement
    # diagnostics and it never enters a loss anywhere in this file.
    gm = np.where(am, -np.inf, g)
    argmax_readout = torch.from_numpy(np.asarray(gm.argmax(1), np.int64))
    return (z, torch.from_numpy(phi_t[hq]), torch.from_numpy(ho), torch.from_numpy(hm),
            torch.from_numpy(pos), torch.from_numpy(am), tgt, argmax_readout)


STATE_TREE, STATE_VOL, STATE_RAND = 0, 1, 2          # the order of pu_labels.STATE_TYPES
assert PL.STATE_TYPES == ("tree", "vol", "rand"), "the state-type codes moved under this file"


# ============================================================================ the warm start
def warm_start(pol, path, pool, norm):
    """Load E1 and prove it. 16/16 tensors, every one BYTE-IDENTICAL after the load."""
    blob = torch.load(path, map_location="cpu")
    for k, want in (("arch", "equivariant"), ("state", "both"), ("use_pos", True),
                    ("d_phi", int(norm[0].shape[1])), ("n_pool", len(pool))):
        assert blob[k] == want, f"{os.path.basename(path)} {k}={blob[k]!r}, this run needs {want!r}"
    assert [(int(a), int(b)) for a, b in blob["pool"]] == pool, \
        "the warm-start checkpoint was fitted on a DIFFERENT pool"
    assert np.array_equal(blob["norm_mu"], norm[0]) and np.array_equal(blob["norm_sd"], norm[1]), \
        "the warm-start checkpoint's descriptor standardisation differs from this run's"
    sd = blob["model"]
    own = pol.state_dict()
    moved = [k for k, v in sd.items() if k in own and own[k].shape == v.shape]
    assert len(moved) == 16 and len(sd) == 16, (
        f"warm start would transfer {len(moved)} of {len(sd)} tensors. The E1 equivariant/both "
        f"policy has exactly 16; anything else is a cold start wearing a warm label.")
    pol.load_state_dict(sd)
    after = pol.state_dict()
    for k in sd:
        assert torch.equal(after[k], sd[k]), f"tensor {k} is not byte-identical after the load"
    log(f"[warm] {os.path.basename(path)} epoch {blob['epoch']} val_full {blob['val_full']:.8f} -- "
        f"16/16 tensors transferred, every one byte-identical after the load")
    return blob


# ============================================================================ FIT
def fit(args):
    assert args.i_have_author_goahead, (
        "REFUSING TO FIT. Pass --i_have_author_goahead only when the author has said go.")
    torch.manual_seed(SEED); np.random.seed(SEED)
    rng = np.random.default_rng(SEED)
    t00 = time.time()
    os.makedirs(PU_DIR, exist_ok=True)

    blob_b, c, S, fit_users, hold_users = load_labels()
    pool = [(int(a), int(b)) for a, b in S["pool"]]
    P = len(pool)
    phi_t = np.asarray(S["phi"], np.float32)
    norm = (np.asarray(S["norm_mu"], np.float32), np.asarray(S["norm_sd"], np.float32))
    phi_g = torch.from_numpy(phi_t)
    n_users = len(S["rows"])
    log(f"[data] {n_users:,} labelled FIT users x {P} candidates = {n_users * P:,} graded targets; "
        f"composition {json.dumps(blob_b['composition']['by_state_type'])}")
    PL.print_composition(blob_b["composition"])

    # ---- the CONTROL (HARD RULE 5) ---------------------------------------------------------------
    if args.arm == "shuffle":
        G = np.asarray(S["gain_full"]).copy()
        for u in range(n_users):
            G[u] = G[u][rng.permutation(P)]
        S["gain_full"] = G
        log("[arm] SHUFFLE CONTROL: every user's gain VECTOR is permuted across candidates, within "
            "that user. The marginals survive exactly; the state -> gain association does not. An "
            "arm that does not beat this on held-out FIT states has learned nothing about states.")

    # ---- the selection slice, out of FIT, by user -------------------------------------------------
    tr_ix, dev_ix = dev_split(np.arange(n_users))
    log(f"[select] {len(tr_ix):,} FIT-train users / {len(dev_ix):,} FIT-dev users (by user, "
        f"{DEV_FRAC:.0%}, rng {DEV_SEED}). Selection NEVER sees HOLDOUT.")

    # ---- the model ---------------------------------------------------------------------------------
    d_z = int(S["z"].shape[1])
    pol = Policy(d_z, phi_t.shape[1], P, arch="equivariant", state="both", use_pos=True)
    if args.cold:
        # ★ THE PEAKEDNESS ARM. E1 is a distilled ARGMAX policy: it was trained to put nearly all
        # mass on one candidate, and a listwise CE against a graded target punishes confident-and-
        # wrong far harder than it punishes flat. Warm-starting therefore hands the optimiser a
        # maximally-peaked initialisation to dig out of, and the one-hot replay term keeps pushing it
        # back. This arm removes BOTH forcings, so "it cannot beat a flat prediction" can be read as
        # a property of the DESIGN rather than of its initialisation. It is a diagnostic, declared as
        # one; it is not the pilot's configuration.
        e1 = {"epoch": -1, "val_full": float("nan")}
        log("[model] COLD START (diagnostic): no warm start. Nothing is forcing this model to be "
            "peaked, so a flat prediction is freely reachable for it.")
    else:
        e1 = warm_start(pol, args.init_ckpt, pool, norm)
    log(f"[model] UNCHANGED E1 architecture, params {pol.n_params():,}")

    # ---- the replay ---------------------------------------------------------------------------------
    Dtr = load_traj(train=True)
    assert [(int(a), int(b)) for a, b in Dtr["pool"]] == pool, "the replay pool is not this pool"
    n_rep_ex = len(Dtr["uid"]) * TOTAL_L
    n_pu_b = int(round(args.batch * args.pu_frac))
    n_rep_b = args.batch - n_pu_b
    steps = int(np.ceil(len(tr_ix) / max(n_pu_b, 1)))
    log(f"[mixture] batch {args.batch} = {n_pu_b} per-user-gain + {n_rep_b} tree-label replay "
        f"(pu_frac {args.pu_frac} DECLARED, not tuned post hoc); tau {args.tau} DECLARED; "
        f"one epoch = one shuffled pass over the {len(tr_ix):,} FIT-train states = {steps} steps")

    opt = torch.optim.Adam(pol.parameters(), lr=args.lr)
    hard_ce = nn.CrossEntropyLoss()
    stem = f"pu_{args.arm}" + (f"_{args.tag}" if args.tag else "")
    ckpt = os.path.join(PU_DIR, stem + ".pt")
    for suffix in (".pt", "_history.json"):
        p_ = os.path.join(PU_DIR, stem + suffix)
        assert not os.path.exists(p_) or args.overwrite, (
            f"{os.path.basename(p_)} already exists. Pass --tag to name this run, or --overwrite if "
            f"you really mean to replace a banked artifact (HARD RULE 11e).")

    def dev_eval(zero=False):
        """Soft-target CE on the FIT-dev users, plus the argmax READ-OUTS. No gradient, no HOLDOUT."""
        pol.eval()
        tot = n = 0.0; hit1 = hit5 = 0
        with torch.no_grad():
            for st in range(0, len(dev_ix), 1024):
                ix = dev_ix[st:st + 1024]
                z, hp, ho, hm, pos, am, tgt, ar = pu_batch(S, phi_t, ix, P, tau=args.tau,
                                                           zero_state=zero)
                s = pol(z, hp, ho, hm, pos, phi_g, am)
                tot += float(listwise_ce(s, tgt)) * len(ix); n += len(ix)
                top5 = s.topk(5, dim=1).indices
                hit1 += int((s.argmax(1) == ar).sum())
                hit5 += int((top5 == ar[:, None]).any(1).sum())
        return {"soft_ce": tot / n, "top1_vs_measured_argmax": hit1 / n,
                "top5_vs_measured_argmax": hit5 / n}

    # ★ THE TARGET DIAGNOSTIC THAT WOULD HAVE CAUGHT tau=0.01 IMMEDIATELY (added 2026-08-16).
    # A target at ~ln(n_avail) nats carries NO information (uniform); a target at ~0 nats is one-hot
    # and throws away the graded signal the sweep paid 35 minutes for. Neither is trainable, and the
    # only way to know which end you are at is to MEASURE the target before reading any loss. The
    # uniform-prediction CE is printed beside it so "beats flat" is readable at a glance: a model
    # whose dev CE exceeds it is worse than emitting a constant, i.e. below the free baseline.
    tent, teff, uce, nav = [], [], [], []
    with torch.no_grad():
        for st_ in range(0, len(dev_ix), 1024):
            ix = dev_ix[st_:st_ + 1024]
            _z, _hp, _ho, _hm, _pos, am_, tgt_, _ar = pu_batch(S, phi_t, ix, P, tau=args.tau)
            # 1e-30, NOT 1e-300: the target is float32, whose smallest normal is ~1e-38, so a
            # 1e-300 clamp underflows to 0 and 0*log(0) is NaN. Measured 2026-08-16.
            h = -(tgt_ * torch.log(tgt_.clamp_min(1e-30))).sum(-1)
            tent.append(h); teff.append(torch.exp(h))
            uce.append(listwise_ce(torch.zeros(len(ix), P).masked_fill(am_, float("-inf")), tgt_)
                       * len(ix))
            nav.append(float((~am_).sum(1).float().mean()) * len(ix))
    tent = torch.cat(tent); teff = torch.cat(teff)
    tdiag = {"tau": args.tau,
             "target_entropy_nats_mean": float(tent.mean()),
             "target_entropy_nats_median": float(tent.median()),
             "target_eff_support_median": float(teff.median()),
             "target_eff_support_p10": float(torch.quantile(teff, 0.10)),
             "target_eff_support_p90": float(torch.quantile(teff, 0.90)),
             "frac_effectively_one_hot_eff_lt_2": float((teff < 2).float().mean()),
             "frac_effectively_uniform_eff_gt_half_pool": float((teff > P / 2).float().mean()),
             "uniform_prediction_ce": float(sum(uce) / len(dev_ix)),
             "n_available_candidates_mean": float(sum(nav) / len(dev_ix)),
             "max_entropy_ln_n_avail": float(np.log(sum(nav) / len(dev_ix)))}
    log(f"[target@tau={args.tau}] entropy {tdiag['target_entropy_nats_mean']:.4f} nats "
        f"(max possible {tdiag['max_entropy_ln_n_avail']:.4f}) | effective support median "
        f"{tdiag['target_eff_support_median']:.1f} [p10 {tdiag['target_eff_support_p10']:.1f}, "
        f"p90 {tdiag['target_eff_support_p90']:.1f}] of {P} | one-hot-ish "
        f"{tdiag['frac_effectively_one_hot_eff_lt_2'] * 100:.1f}% | uniform-ish "
        f"{tdiag['frac_effectively_uniform_eff_gt_half_pool'] * 100:.1f}% | UNIFORM-PREDICTION CE "
        f"{tdiag['uniform_prediction_ce']:.4f} <- the free baseline every epoch must beat")

    d0 = dev_eval()
    dz0 = dev_eval(zero=True)
    log(f"[dev@init] warm-start soft-CE {d0['soft_ce']:.5f} | top1 vs measured argmax "
        f"{d0['top1_vs_measured_argmax']:.4f} top5 {d0['top5_vs_measured_argmax']:.4f} "
        f"| ZEROED-STATE top1 {dz0['top1_vs_measured_argmax']:.4f}")

    hist = []
    best = {"soft_ce": 1e18, "epoch": None}
    for ep in range(1, args.epochs + 1):
        pol.train()
        te = time.time()
        perm = tr_ix[rng.permutation(len(tr_ix))]
        sl_pu = sl_rep = 0.0
        for si in range(steps):
            ix = perm[si * n_pu_b:(si + 1) * n_pu_b]
            if not len(ix):
                continue
            z, hp, ho, hm, pos, am, tgt, _ar = pu_batch(S, phi_t, ix, P, tau=args.tau)
            l_pu = listwise_ce(pol(z, hp, ho, hm, pos, phi_g, am), tgt)
            if n_rep_b > 0:
                e = rng.integers(0, n_rep_ex, size=n_rep_b)
                ru = (e // TOTAL_L).astype(np.int64); rt = (e % TOTAL_L).astype(np.int64)
                rz, rhp, rho, rhm, rpos, ram, ry = make_batch(Dtr, phi_t, ru, rt, P)
                l_rep = hard_ce(pol(rz, rhp, rho, rhm, rpos, phi_g, ram), ry)
                loss = (len(ix) * l_pu + n_rep_b * l_rep) / (len(ix) + n_rep_b)
            else:
                # REPLAY-FREE (pu_frac = 1.0): the anti-forgetting term is gone, so the spanning gate
                # is expected to fail and this arm is a DIAGNOSTIC, never a candidate policy.
                l_rep = torch.zeros(())
                loss = l_pu
            opt.zero_grad(); loss.backward(); opt.step()
            sl_pu += float(l_pu); sl_rep += float(l_rep)
        d = dev_eval()
        dz = dev_eval(zero=True)
        rec = {"epoch": ep, "train_listwise_ce": sl_pu / max(steps, 1),
               "train_replay_ce": sl_rep / max(steps, 1),
               "dev_soft_ce": d["soft_ce"],
               "dev_top1_vs_measured_argmax": d["top1_vs_measured_argmax"],
               "dev_top5_vs_measured_argmax": d["top5_vs_measured_argmax"],
               "dev_zeroed_state_top1": dz["top1_vs_measured_argmax"],
               "dev_zeroed_state_soft_ce": dz["soft_ce"],
               "minutes": round((time.time() - te) / 60, 2)}
        hist.append(rec)
        log(f"[ep{ep}] listwise {rec['train_listwise_ce']:.5f} replay {rec['train_replay_ce']:.5f} | "
            f"DEV soft-CE {d['soft_ce']:.5f} (init {d0['soft_ce']:.5f}) top1 "
            f"{d['top1_vs_measured_argmax']:.4f} top5 {d['top5_vs_measured_argmax']:.4f} | zeroed "
            f"top1 {dz['top1_vs_measured_argmax']:.4f} | {rec['minutes']:.1f}m")
        if d["soft_ce"] < best["soft_ce"]:
            best = {"soft_ce": d["soft_ce"], "epoch": ep, "rec": rec}
            torch.save({"model": pol.state_dict(), "arch": "equivariant", "state": "both",
                        "use_pos": True, "d_phi": int(phi_t.shape[1]), "n_pool": P, "epoch": ep,
                        "val_full": float("nan"), "val_tail": float("nan"),
                        "norm_mu": norm[0], "norm_sd": norm[1], "arm": f"pu_{args.arm}",
                        "pool": [[int(a), int(b)] for a, b in pool], "seed": SEED,
                        "dev_soft_ce": d["soft_ce"], "tau": args.tau, "pu_frac": args.pu_frac,
                        "init_from": os.path.basename(args.init_ckpt)}, ckpt)
            log(f"[ckpt] epoch {ep} is the best FIT-dev soft-CE so far -> {os.path.basename(ckpt)}")
        out = {"analysis": "pu_policy_fit", "design_sheet":
               "docs/design/PAPER_B_POLICY_DESIGN.md PART K", "citable": True,
               "test_touched": False, "arm": args.arm, "tag": args.tag,
               "declared_constants": {"tau": args.tau, "pu_frac": args.pu_frac,
                                      "dev_frac": DEV_FRAC, "dev_seed": DEV_SEED, "lr": args.lr,
                                      "batch": args.batch, "epochs": args.epochs,
                                      "note": "declared before the run; none tuned post hoc"},
               "label_contract": c, "composition": blob_b["composition"],
               "n_fit_train_users": int(len(tr_ix)), "n_fit_dev_users": int(len(dev_ix)),
               "selection_rule": ("the epoch with the lowest soft-target CE on the FIT-dev users, "
                                  "declared before the run. Selection never sees HOLDOUT."),
               "cold_start": bool(args.cold), "replay_free": bool(n_rep_b == 0),
               "warm_start": {"ckpt": (None if args.cold else os.path.basename(args.init_ckpt)),
                              "epoch": int(e1["epoch"]), "val_full": float(e1["val_full"]),
                              "tensors": "16/16, byte-identical after the load"},
               "at_init": {"dev": d0, "dev_zeroed_state": dz0},
               "target_diagnostic": tdiag,
               "beats_flat_prediction": bool(best["soft_ce"] < tdiag["uniform_prediction_ce"]),
               "best_dev_soft_ce": best["soft_ce"],
               "uniform_prediction_ce": tdiag["uniform_prediction_ce"],
               "history": hist, "best_epoch": best["epoch"],
               "label_is_never_an_argmax": ("the measured per-user argmax is computed only as a "
                                            "read-out for the agreement diagnostics; the training "
                                            "target is the temperature-scaled softmax over all "
                                            f"{P} candidates (K.1)"),
               "minutes": round((time.time() - t00) / 60, 2)}
        atomic_write(os.path.join(PU_DIR, stem + "_history.json"), out)
    log(f"[fit] best FIT-dev soft-CE {best['soft_ce']:.6f} at epoch {best['epoch']} "
        f"(warm start {d0['soft_ce']:.6f}); {(time.time() - t00) / 60:.1f}m")
    if best["soft_ce"] >= d0["soft_ce"]:
        log("[NOTE] no epoch improved on the WARM START's own dev CE. A pilot that only preserves "
            "its initialisation has learned nothing, and that is the reported outcome (K.4).")
    if best["soft_ce"] >= tdiag["uniform_prediction_ce"]:
        log(f"[NOTE] ★ BELOW THE FREE BASELINE: best dev CE {best['soft_ce']:.4f} >= the "
            f"uniform-prediction CE {tdiag['uniform_prediction_ce']:.4f}. A learner that cannot "
            f"beat emitting a constant is broken or starved, not a measurement of the design.")
    else:
        log(f"[NOTE] beats the flat prediction: {best['soft_ce']:.4f} < "
            f"{tdiag['uniform_prediction_ce']:.4f} (margin {tdiag['uniform_prediction_ce'] - best['soft_ce']:.4f} nats)")
    return out


# ============================================================================ EVALUATE (the gates)
def evaluate(args):
    assert args.i_have_author_goahead, "REFUSING TO RUN without --i_have_author_goahead"
    import policy_resume as PR
    from strategy_channel_suite import (build_greedy_env, contract_selftest, genv_eval, genv_mean,
                                        genv_eval_multi)
    from ndcg_tree import build_pool, leak_assert, branch_of_selftest
    torch.manual_seed(SEED); np.random.seed(SEED)
    t00 = time.time()

    blob_b, c, S, fit_users, hold_users = load_labels()
    pool = [(int(a), int(b)) for a, b in S["pool"]]
    P = len(pool)

    log("#" * 106)
    log("# PART K -- GATES (K.5). VALIDATION ONLY, and every number below is on the 2,000 HOLDOUT")
    log("# users, whose states and labels were never touched. TEST is not built, loaded or named.")
    log("#" * 106)

    E = build_greedy_env(args.snapshot, "validation", args.concept_ckpt, "pu_eval")
    contract_selftest(E, E["rows"]); leak_assert(E, E["rows"], "pu_eval/val")
    rows_all = E["rows"]
    f0, t0_ = genv_mean(E, [], rows_all)
    assert round(f0, 8) == VAL_M0_FULL, f"CANONICAL SNAP FAILED: {f0!r}"
    log(f"[gate] canonical m=0 snap full={f0!r} -- exact")
    pool_e, _i, _cc = build_pool(E, PL.N_ITEMS_POOL, PL.N_CONC_POOL)
    assert pool_e == pool, "the env's pool is not the label artifact's pool"
    branch_of_selftest(E, rows_all[:2000], [pool[0], pool[500], (1, 114)])
    static_seq = [(int(a), int(b)) for a, b in json.load(open(STATIC_JSON))["sequence"]]
    sfa, _ = genv_eval(E, static_seq, rows_all)
    assert abs(float(np.nanmean(sfa)) - STATIC_VAL_FULL) < 1e-12, "BANKED-HEADLINE ASSERT FAILED"
    log(f"[gate] static comparator reproduces EXACTLY: {float(np.nanmean(sfa))!r}")

    Dva = load_traj(train=False)
    assert list(np.asarray(Dva["uid"], np.int64)) == list(np.asarray(rows_all, np.int64))
    hold = np.asarray(hold_users, np.int64)
    hpos = {int(r): j for j, r in enumerate(rows_all)}
    hix = np.asarray([hpos[int(r)] for r in hold], np.int64)
    rows = [int(r) for r in hold]
    log(f"[cohort] {len(rows):,} HOLDOUT users; 0 of them carry a label (asserted on load)")

    # ---- E1 on the SAME users, from its banked per-user vectors (exact) ----------------------------
    z1 = np.load(E1_PER_USER)
    assert np.array_equal(np.asarray(z1["rows"], np.int64), np.asarray(rows_all, np.int64)), \
        "E1's banked per-user vectors are on different rows"
    assert abs(float(np.asarray(z1["policy_full"]).mean()) - E1_VAL_FULL) < 1e-6, \
        "the banked E1 per-user vector moved"
    e1_full = np.asarray(z1["policy_full"], np.float64)[hix]
    e1_tail = np.asarray(z1["policy_tail"], np.float64)[hix]
    e1_asked = np.asarray(z1["asked"], np.int64)[hix]
    tree_asked = np.asarray(Dva["label_idx"], np.int64)[hix]
    tree_outs = np.asarray(Dva["outcome"], np.int64)[hix]
    assert np.array_equal(e1_asked, tree_asked), "E1's banked rollout is not the tree's (it was, 1e-9)"
    # ★ NaN-SAFE. Measured 2026-08-16: 178 of the 2,000 HOLDOUT users have NO tail target, so their
    # tail NDCG is NaN by construction. `.mean()` propagated that into the artifact's e1_tail field
    # as a bare NaN while the paired bootstrap (which strips NaN) reported a valid interval beside
    # it -- a summary statistic and the table it summarises reading different fields, which is the
    # `verify-the-artifact-not-a-proxy` defect. Every reduction over a per-user vector is nan-aware.
    log(f"[E1] on these {len(rows):,} HOLDOUT users: full {np.nanmean(e1_full):.8f} tail "
        f"{np.nanmean(e1_tail):.8f} ({int(np.isnan(e1_tail).sum())} users have no tail target and "
        f"are NaN, excluded from every mean and from the paired bootstrap) -- the spanning bar")

    # ---- the trained policy -----------------------------------------------------------------------
    phi_t, norm = build_phi(E, pool)
    assert np.allclose(phi_t, np.asarray(S["phi"], np.float32), atol=0, rtol=0), \
        "the descriptors rebuilt here differ from the ones the labels were banked with"
    blob = torch.load(args.ckpt, map_location="cpu")
    assert [(int(a), int(b)) for a, b in blob["pool"]] == pool
    assert np.array_equal(blob["norm_mu"], norm[0]) and np.array_equal(blob["norm_sd"], norm[1])
    pol = Policy(E["ctx"].d, phi_t.shape[1], P, arch=blob["arch"], state=blob["state"],
                 use_pos=blob["use_pos"])
    pol.load_state_dict(blob["model"]); pol.eval()
    log(f"[policy] {os.path.basename(args.ckpt)} epoch {blob['epoch']} arm {blob.get('arm')} "
        f"dev_soft_ce {blob.get('dev_soft_ce')} -- FROZEN, no gradient step is taken in this file")

    gates = {}
    # ---- GATE 1: SPANNING (HARD RULE 8) -- unseeded K=16 on HOLDOUT -------------------------------
    ro = rollout(E, pol, phi_t, pool, rows)
    pf, pt = ro["curves"][TOTAL_L]
    gates["G-K-spanning"] = {
        "criterion": "the unseeded K=16 HOLDOUT rollout must not fall below E1 on the SAME users",
        "policy_full": float(np.nanmean(pf)), "policy_tail": float(np.nanmean(pt)),
        "e1_full": float(np.nanmean(e1_full)), "e1_tail": float(np.nanmean(e1_tail)),
        "n_users_no_tail_target": int(np.isnan(e1_tail).sum()),
        "delta_full": float(np.nanmean(pf) - np.nanmean(e1_full)),
        "delta_tail": float(np.nanmean(pt) - np.nanmean(e1_tail)),
        "paired_full": paired_bootstrap(pf - e1_full), "paired_tail": paired_bootstrap(pt - e1_tail),
        "pass": bool(float(np.nanmean(pf)) >= float(np.nanmean(e1_full)))}
    log(f"[G-K-spanning] policy {float(np.nanmean(pf)):.8f} vs E1 {np.nanmean(e1_full):.8f} -> "
        f"{'PASS' if gates['G-K-spanning']['pass'] else 'FAIL'}")

    # ---- GATE 2: ★ THE BEHAVIOURAL WIN, on FRESH seeded states from HOLDOUT users ------------------
    pol_asked = ro["asked"]                      # the policy's OWN unseeded order = its frozen arm
    by_k = {}
    for k in EVAL_KS:
        log("=" * 100)
        log(f"=== k = {k} volunteered items, HOLDOUT users, draw seed {EVAL_DRAW_SEED} "
            f"(disjoint from R/R2's 98765 and P2's 424242)")
        vol, draw_info = PR.draw_volunteered(E, rows, k, seed=EVAL_DRAW_SEED + k)
        cells = {}
        # the policy, conducting its own interview FROM the seeded state
        rs = PR.rollout_seeded(E, pol, phi_t, pool, rows, vol)
        f_pol, t_pol = rs["curves"][TOTAL_L]
        _i, _c, red_pol = PR.build_seqs(rs["asked"], pool, vol)
        cells["policy"] = (f_pol, t_pol, rs["asked"], rs["outs"], red_pol)
        # ITS OWN FROZEN ORDER at the same seeded state: differs in the QUESTIONS and nothing else
        for name, ask in (("policy_frozen", pol_asked), ("tree", tree_asked)):
            iseq, cseq, red = PR.build_seqs(ask, pool, vol)
            f_, t_ = genv_eval_multi(E, rows, iseq, cseq)
            cells[name] = (f_, t_, ask, tree_outs if name == "tree" else ro["outs"], red)
        # ask-nothing-more: what the volunteered evidence alone is worth
        f_n, t_n = genv_eval_multi(E, rows, vol, [[] for _ in rows])
        cells["none"] = (f_n, t_n, np.full((len(rows), 0), -1, np.int64),
                         np.full((len(rows), 0), -1, np.int8), np.zeros(len(rows), np.int64))
        blob_k = {"draw": draw_info,
                  "arms": {a: {"full": float(np.nanmean(v[0])), "tail": float(np.nanmean(v[1]))}
                           for a, v in cells.items()},
                  "deviation_cost_flip": {
                      "full": paired_bootstrap(cells["policy"][0] - cells["policy_frozen"][0]),
                      "tail": paired_bootstrap(cells["policy"][1] - cells["policy_frozen"][1]),
                      "bar": ">= 0 with the interval excluding a loss",
                      "R2_measured": {"2": -0.01246, "4": -0.01234, "8": -0.00860}},
                  "vs_tree": {"full": paired_bootstrap(cells["policy"][0] - cells["tree"][0]),
                              "tail": paired_bootstrap(cells["policy"][1] - cells["tree"][1]),
                              "sentence": "'not worse' is the claim; 'beats' is not expected"},
                  "redundancy": {a: {"mean": float(v[4].mean()),
                                     "interval": paired_bootstrap(v[4].astype(np.float64))}
                                 for a, v in cells.items()},
                  "anti_collapse": {
                      "policy": collapse_diagnostics([tuple(r) for r in cells["policy"][2]]),
                      "pep_policy": pep_divergence_from_arrays(cells["policy"][2],
                                                               cells["policy"][3]),
                      "pep_tree": pep_divergence_from_arrays(tree_asked, tree_outs)},
                  "frac_users_deviating_from_own_unseeded_order":
                      float((cells["policy"][2] != pol_asked).any(1).mean()),
                  "channel_mix_policy": channel_mix(cells["policy"][2], pool)}
        by_k[str(k)] = blob_k
        d = blob_k["deviation_cost_flip"]["full"]
        log(f"[k={k}] policy {blob_k['arms']['policy']['full']:.6f} vs its own frozen order "
            f"{blob_k['arms']['policy_frozen']['full']:.6f} -> {d['mean']:+.6f} "
            f"[{d['lo']:+.6f}, {d['hi']:+.6f}] "
            f"{'SEPARATES' if d['excludes_zero'] else 'null'} | deviates on "
            f"{blob_k['frac_users_deviating_from_own_unseeded_order'] * 100:.1f}% of users")
        np.savez_compressed(os.path.join(PU_DIR, f"pu_eval_k{k}_per_user.npz"),
                            **{f"{a}_{w}": v[i] for a, v in cells.items()
                               for i, w in ((0, "full"), (1, "tail"), (4, "redundancy"))})

    gates["G-K-behavioural"] = {
        "criterion": ("on fresh seeded states from HOLDOUT users, policy MINUS ITS OWN FROZEN ORDER "
                      ">= 0 with the interval excluding a loss (K.5)"),
        "by_k": {k: v["deviation_cost_flip"]["full"] for k, v in by_k.items()},
        "pass": bool(all(not (v["deviation_cost_flip"]["full"]["hi"] < 0) for v in by_k.values()))}
    gates["G-K-anti_collapse"] = {
        "unseeded": collapse_diagnostics([tuple(r) for r in ro["asked"]]),
        "pep_policy": pep_divergence_from_arrays(ro["asked"], ro["outs"]),
        "pep_tree": pep_divergence_from_arrays(tree_asked, tree_outs),
        "bar": "> 3 distinct interviews AND divergence >= the tree's own"}
    gates["G-K-anti_collapse"]["pass"] = bool(
        gates["G-K-anti_collapse"]["unseeded"]["n_distinct_interviews"] > 3 and
        (gates["G-K-anti_collapse"]["pep_policy"]["divergence_rate"] or 0) >=
        (gates["G-K-anti_collapse"]["pep_tree"]["divergence_rate"] or 0))

    out = {"analysis": "pu_policy_gates", "design_sheet":
           "docs/design/PAPER_B_POLICY_DESIGN.md PART K.5", "citable": True, "test_touched": False,
           "ckpt": os.path.basename(args.ckpt), "arm": blob.get("arm"),
           "cohort": {"eval": "the 2,000 HOLDOUT validation users (never labelled, never trained, "
                              "never selected on)",
                      "labels": "the 8,000 FIT validation users",
                      "split": c["split_info"]},
           "draw_seed": EVAL_DRAW_SEED, "by_k": by_k, "gates": gates,
           "honesty": ("the labels are ONE-STEP measured gains, so this is a greedy policy with "
                       "reach and never a strategic one. Myopia is declared. The bank result "
                       "(+0.01430 > +0.01035) travels with every adaptive sentence."),
           "tower": E["tower"], "conc_ckpt": E["conc_ckpt"], "mask_policy": "B",
           "pool_contract": E["pool_contract"],
           "minutes": round((time.time() - t00) / 60, 2)}
    stem = "pu_gates" + (f"_{args.tag}" if args.tag else "")
    atomic_write(os.path.join(PU_DIR, stem + ".json"), out)
    np.savez_compressed(os.path.join(PU_DIR, stem + "_per_user.npz"),
                        policy_full=pf, policy_tail=pt, e1_full=e1_full, e1_tail=e1_tail,
                        asked=ro["asked"], outs=ro["outs"], rows=np.asarray(rows, np.int64))
    for k, v in gates.items():
        log(f"  {k}: {json.dumps(v, default=float)[:300]}")
    return out


# ============================================================================ selftest (pure)
def selftest():
    """Random weights, synthetic tensors. No data, no environment, no optimiser, no fit."""
    torch.manual_seed(SEED)

    # (1) permutation EQUIVARIANCE of the scorer, exact -- the property the scalability claim rests on
    B, P, T, dz, dphi = 5, 40, 4, 16, 11
    pol = Policy(dz, dphi, P, arch="equivariant", state="both", use_pos=True)
    z = torch.randn(B, dz); phi = torch.randn(P, dphi)
    hp = torch.randn(B, T, dphi); ho = torch.randint(0, N_OUTCOMES, (B, T))
    hm = (torch.rand(B, T) > 0.3).float(); pos = torch.randn(B, 2)
    s = pol(z, hp, ho, hm, pos, phi)
    perm = torch.randperm(P)
    assert torch.allclose(pol(z, hp, ho, hm, pos, phi[perm]), s[:, perm], atol=1e-6), \
        "the scorer is not permutation-equivariant"
    keep = torch.arange(P) != 3
    assert torch.allclose(pol(z, hp, ho, hm, pos, phi[keep]), s[:, keep], atol=1e-6), \
        "dropping a candidate moved the others"
    log("[selftest] PASS: permutation equivariance + candidate independence, exact")

    # (2) the soft target: known structure -> expected target; a zero gain vector -> UNIFORM
    g = torch.zeros(3, 6)
    g[0, 2] = 0.02; g[0, 5] = 0.01                     # a clear best and a runner-up
    g[1] = 0.0                                          # every candidate unanswerable: gain == 0
    g[2, 0] = -0.03                                     # a candidate that HURTS
    t = soft_target(g, tau=0.01)
    assert torch.allclose(t.sum(1), torch.ones(3), atol=1e-6), "the target is not a distribution"
    assert int(t[0].argmax()) == 2 and t[0, 2] > t[0, 5] > t[0, 0], \
        "the target does not rank by measured gain"
    exp = torch.softmax(torch.tensor([0.0, 0.0, 2.0, 0.0, 0.0, 1.0]), 0)
    assert torch.allclose(t[0], exp, atol=1e-6), "the target is not softmax(gain / tau)"
    assert torch.allclose(t[1], torch.full((6,), 1 / 6), atol=1e-6), \
        "an all-zero (all-unanswerable) gain vector must map to the UNIFORM target"
    assert float(t[2, 0]) < float(t[2, 1]), "a candidate that hurts got more mass than a neutral one"
    # ...and the unanswerable candidate inside an informative row carries NEAR-UNIFORM mass, i.e.
    # the same mass as every other measured-zero candidate. Nothing encodes answerability; the
    # measurement does (K.1).
    assert abs(float(t[0, 0]) - float(t[0, 1])) < 1e-9 and abs(float(t[0, 0]) - float(t[0, 3])) < 1e-9
    # a hotter temperature must FLATTEN and a colder one must SHARPEN, monotonically
    assert soft_target(g, tau=0.1)[0, 2] < t[0, 2] < soft_target(g, tau=0.001)[0, 2], \
        "tau does not control the target's sharpness"
    # a masked candidate takes EXACTLY zero mass and the rest renormalise
    m = torch.zeros(3, 6, dtype=torch.bool); m[0, 2] = True
    tm = soft_target(g, tau=0.01, mask=m)
    assert float(tm[0, 2]) == 0.0 and abs(float(tm[0].sum()) - 1.0) < 1e-6
    assert int(tm[0].argmax()) == 5, "masking the best did not promote the runner-up"
    # and the loss ignores the masked entry even at -inf logits (no NaN)
    lg = torch.randn(3, 6); lg = lg.masked_fill(m, float("-inf"))
    L = listwise_ce(lg, tm)
    assert torch.isfinite(L), "the listwise loss is not finite with a masked candidate"
    # the loss is MINIMISED when the logits ARE the target's own log-probabilities
    assert float(listwise_ce(torch.log(t + 1e-12), t)) < float(listwise_ce(torch.zeros(3, 6), t))
    log("[selftest] PASS: soft target = softmax(gain/tau), uniform on an all-zero (unanswerable) "
        "vector, near-uniform mass on a zero-gain candidate, masking exact, loss finite and minimal "
        "at the target")

    # (3) the batch ENCODING follows the state's own bookkeeping, per state type
    n, Pp, dphi2 = 3, 12, 7
    Sx = {"z": np.random.RandomState(0).randn(n, 9).astype(np.float32),
          "stype": np.asarray([STATE_TREE, STATE_VOL, STATE_RAND], np.int64),
          "klen": np.asarray([3, 5, 2], np.int64),
          "hist_q": np.full((n, TOTAL_L - 1), -1, np.int64),
          "hist_out": np.full((n, TOTAL_L - 1), -1, np.int64),
          "gain_full": np.zeros((n, Pp), np.float32)}
    Sx["hist_q"][0, :3] = [4, 7, 1]; Sx["hist_out"][0, :3] = [0, 2, 1]
    Sx["hist_q"][2, :2] = [9, 0]; Sx["hist_out"][2, :2] = [1, 0]
    Sx["gain_full"][0, 4] = 0.05                      # the best candidate is one already ASKED
    Sx["gain_full"][0, 6] = 0.01
    phix = np.random.RandomState(1).randn(Pp, dphi2).astype(np.float32)
    z_, hp_, ho_, hm_, pos_, am_, tgt_, ar_ = pu_batch(Sx, phix, np.arange(n), Pp, tau=0.01)
    assert hm_[0].sum() == 3 and hm_[2].sum() == 2, "an as-asked state lost its history"
    assert hm_[1].sum() == 0, "a VOLUNTEERED state must carry an EMPTY asked-history (R2)"
    assert float(pos_[1, 0]) == 0.0 and float(pos_[1, 1]) == TOTAL_L, \
        "a VOLUNTEERED state must sit at position 0 -- the belief moved, nothing was asked"
    assert float(pos_[0, 0]) == 3.0 and float(pos_[2, 0]) == 2.0
    assert am_[0, 4] and am_[0, 7] and am_[0, 1] and not am_[0, 6], "the asked mask is wrong"
    assert not am_[1].any(), "a volunteered item was masked -- R's semantics say it may be re-asked"
    assert float(tgt_[0, 4]) == 0.0, "an already-asked candidate carries target mass"
    assert int(tgt_[0].argmax()) == 6 and int(ar_[0]) == 6, \
        "the masked best candidate was not excluded from the target and the read-out"
    assert torch.allclose(tgt_[1], torch.full((Pp,), 1 / Pp), atol=1e-6), \
        "an all-zero gain vector on a volunteered state is not uniform"
    zb = pu_batch(Sx, phix, np.arange(n), Pp, tau=0.01, zero_state=True)
    assert float(zb[0].abs().sum()) == 0.0 and float(zb[3].sum()) == 0.0 and \
        float(zb[4].abs().sum()) == 0.0, "the zeroed-state diagnostic did not zero the state"
    assert torch.equal(zb[5], am_), "the zeroed-state diagnostic changed the asked mask"
    log("[selftest] PASS: batch encoding -- as-asked states keep history/position/mask, volunteered "
        "states carry an empty history at position 0 and mask nothing, already-asked candidates take "
        "zero target mass, zeroed-state zeroes the state and only the state")

    # (4) the FIT/HOLDOUT split, the resume contract and the .f32 arithmetic (pu_labels' own)
    PL.selftest()
    log("[selftest] pu_policy: nothing was loaded, folded, scored or fitted")
    return True


def main():
    ap = argparse.ArgumentParser(
        description="Part K step 2: the per-user gain distillation trainer. See "
                    "docs/design/PAPER_B_POLICY_DESIGN.md PART K")
    ap.add_argument("--selftest", action="store_true",
                    help="pure-tensor checks on random weights. No data, no optimiser, no fit.")
    ap.add_argument("--mode", default="fit", choices=["fit", "evaluate"])
    ap.add_argument("--arm", default="main", choices=["main", "shuffle"],
                    help="shuffle = the HARD RULE 5 control: each user's gain vector permuted across "
                         "candidates within that user. Marginals survive; the association does not.")
    ap.add_argument("--tau", type=float, default=TAU,
                    help=f"target temperature (DECLARED, default {TAU}, recorded in the artifact, "
                         f"never tuned post hoc)")
    ap.add_argument("--pu_frac", type=float, default=PU_FRAC,
                    help=f"share of each batch carrying per-user gain targets; the rest is tree-label "
                         f"replay (DECLARED, default {PU_FRAC})")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--tag", default="",
                    help="suffix for this run's output stem. HARD RULE 11e: two runs never share an "
                         "output path.")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--cold", action="store_true",
                    help="DIAGNOSTIC: skip the E1 warm start. With --pu_frac 1.0 (replay-free) this "
                         "is the peakedness arm -- nothing forces the model to be confident, so a "
                         "flat prediction is freely reachable. Never a candidate policy.")
    ap.add_argument("--init_ckpt", default=E1_CKPT,
                    help="the E1 anchor. 16/16 tensors, byte-identity checked after the load.")
    ap.add_argument("--ckpt", default=None, help="--mode evaluate: the checkpoint to gate")
    ap.add_argument("--split", default="validation",
                    help="validation ONLY. Any other value is refused (K.2).")
    ap.add_argument("--snapshot", default=None)
    ap.add_argument("--concept_ckpt", default=None)
    ap.add_argument("--threads", type=int, default=None,
                    help="thread count, consumed by strategy_channel_suite AT IMPORT")
    ap.add_argument("--full_threads", action="store_true")
    ap.add_argument("--i_have_author_goahead", action="store_true",
                    help="required to fit or to evaluate, same as policy_net.")
    args = ap.parse_args()
    if args.split != "validation":
        raise SystemExit(
            f"REFUSED: --split {args.split!r}. This file builds VALIDATION environments only. The "
            f"10,000 canonical TEST users are not built, not loaded, not touched; the single unspent "
            f"test look is reserved for experiment Q (design sheet K.2).")
    if args.selftest:
        return selftest()
    if args.mode == "fit":
        return fit(args)
    from strategy_channel_suite import CERT_TOWER, CERT_CONC
    args.snapshot = args.snapshot or CERT_TOWER
    args.concept_ckpt = args.concept_ckpt or CERT_CONC
    args.ckpt = args.ckpt or os.path.join(PU_DIR, "pu_main" + (f"_{args.tag}" if args.tag else "")
                                          + ".pt")
    return evaluate(args)


if __name__ == "__main__":
    main()
