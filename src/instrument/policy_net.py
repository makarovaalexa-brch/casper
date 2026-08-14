r"""WORKSTREAM E / TODO 1.16 -- THE PARAMETRIC POLICY: architecture + E1 imitation training.

Design sheet: `docs/design/PAPER_B_POLICY_DESIGN.md` (PART A is binding; read it first).

**NOTHING HERE HAS BEEN RUN.** Committed before any run per HARD RULE 10. `--selftest` runs pure-tensor
invariance checks on random weights (no data, no optimiser, no fit); a training run needs the author's
explicit go-ahead, which as of 2026-08-13 has NOT been given ("do not train yet, just document all
ideas").

WHAT THE POLICY IS
------------------
    s(q | state) = g_theta(phi_q, h_psi(state))        one shared parameter set, applied per candidate
    pi(state)    = argmax over the pool minus the already-asked

`phi_q` is a FROZEN descriptor built from the certified stack -- the decoder row for an item, the cv3
concept embedding for a concept, plus popularity / answer-rate / polarity and a channel one-hot. No new
representation is learned, so nothing can drift off the certified ruler.

WHY EQUIVARIANT AND NOT A 1,000-WAY HEAD -- three reasons, in order of force:
  1. IT IS THE SCALABILITY CLAIM. A 1,000-way output head is a lookup table over THIS pool: it cannot
     score a candidate it never saw, so the policy is exactly as un-scalable as the tree and the
     author's stated motive evaporates. A shared scorer runs on the full 18,359 items + 1,031 concepts
     as 19,390 dot products in ONE forward pass. That is "scales better", measurably.
  2. JUNE 2026 MEASURED THE ALTERNATIVE FAILING. H3 was proven: a flat MLP actor lacks per-candidate
     inductive bias; the equivariant per-item actor is what took cloning from top-1 14% to 49% and
     produced the best realizable score of that campaign.
  3. STATISTICS. 172 distinct label questions over 202 states. A 1,000-way head has 828 outputs that
     never appear as a positive; a shared scorer has none.

"DID NOT ANSWER" IS A THIRD OUTCOME AND IS NEVER COLLAPSED TO TWO
-----------------------------------------------------------------
A two-outcome encoding would erase the property this chapter turns on. Measured stakes: 8 of the tree's
15 stopped nodes were stopped by the DID-NOT-ANSWER branch of a concept 92.7-100% of the node could
answer; the root's three children sit at 0.374 / 0.320 / 0.087, so answering AT ALL is worth ~+0.25
against +0.048 for the largest question-choice effect anywhere in the tree; ~14% of users answer
nothing. So: a three-way one-hot per asked question, a distinct learned "asked, no answer" token, and
an explicit unanswered-turn count that `z_t` cannot express.

FIREWALL
--------
Train, validate and select on VALIDATION states only. The 10,000 canonical TEST users are never loaded
here -- there is no code path in this file that can build a test environment. The held-out gap is
reported by the separate test-look driver, once per gate, never from here.

THE GATES THIS FILE COMPUTES (design sheet A.9)
-----------------------------------------------
  G-E1a  top-1 agreement with the tree's label on held-out policy-dev states, target >= 50%
         (June: the teacher that WORKED cloned at 49%; the one that failed cloned at 14%)
  G-E1c  anti-collapse -- distinct realised interviews > 3, and the Pep divergence rate (two users
         answer the same question differently => the follow-up differs) at least the tree's own.
         This is the exact June failure (BC collapsed to one fixed 17-item list) and the exact live
         failure already caught once: `conlinucb_bs_static` had ONE distinct interview.
  G-E1d  shuffle control -- `--arm shuffle` trains on permuted labels and must NOT clear G-E1b.
G-E1b (reproduce the tree's test gain within its CI) and G-E2 are TEST-side and live in the test-look
driver, not here.
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

OUTDIR = os.path.join(_ROOT, "experiments", "battery")
SEED = 98765
N_OUTCOMES = 3                                   # like / dislike / did-not-answer. NEVER 2.
OUTCOME = {"like": 0, "dislike": 1, "noanswer": 2}


def log(*a):
    print(time.strftime("[%H:%M:%S]"), *a, flush=True)


# ============================================================================ the network
class StateEncoder(nn.Module):
    """h_psi(state) from THREE inputs, per design sheet A.8. All three by default; the ablation
    (`--state z|hist|both`) is one of E1's two cheap owed experiments.

      z      the tower latent after folding the answers so far -- the recommender's own belief.
      hist   the answer history explicitly: for each asked question, phi_q tensored with a THREE-way
             outcome one-hot, mean-pooled, plus the unanswered-turn count. `z` is a lossy summary
             trained for reconstruction, not for question selection, and by construction it barely
             moves when a question is asked and NOT answered (UNSEEN / refusal tokens) -- which is
             precisely the signal this chapter turns on.
      pos    position t and remaining budget 16 - t. NOTE (design sheet A.4): the tree is
             budget-unaware, so the like-for-like comparison is the variant WITHOUT this, and both
             must be reported.
    """

    def __init__(self, d_z, d_phi, d_h=256, mode="both", use_pos=True):
        super().__init__()
        assert mode in ("z", "hist", "both")
        self.mode, self.use_pos = mode, use_pos
        self.d_phi = d_phi
        d_in = 0
        if mode in ("z", "both"):
            d_in += d_z
        if mode in ("hist", "both"):
            self.hist = nn.Linear(d_phi * N_OUTCOMES, d_h)
            d_in += d_h + N_OUTCOMES + 1          # pooled history + per-outcome counts + n_asked
        if use_pos:
            d_in += 2
        self.mlp = nn.Sequential(nn.Linear(d_in, d_h), nn.ReLU(), nn.Linear(d_h, d_h))
        self.d_out = d_h

    def forward(self, z, hist_phi, hist_out, hist_mask, pos):
        """z (B,d_z) | hist_phi (B,T,d_phi) | hist_out (B,T) long in {0,1,2} | hist_mask (B,T) float
        | pos (B,2)."""
        parts = []
        if self.mode in ("z", "both"):
            parts.append(z)
        if self.mode in ("hist", "both"):
            B, T, _ = hist_phi.shape
            oh = torch.nn.functional.one_hot(hist_out, N_OUTCOMES).to(hist_phi.dtype)   # (B,T,3)
            tok = (hist_phi.unsqueeze(-1) * oh.unsqueeze(-2)).reshape(B, T, -1)         # phi (x) out
            m = hist_mask.unsqueeze(-1)
            pooled = (tok * m).sum(1) / m.sum(1).clamp(min=1.0)
            cnt = (oh * m).sum(1)                                                       # (B,3)
            parts += [torch.relu(self.hist(pooled)), cnt, hist_mask.sum(1, keepdim=True)]
        if self.use_pos:
            parts.append(pos)
        return self.mlp(torch.cat(parts, -1))


class EquivariantScorer(nn.Module):
    """s(q | state) = <u_theta(h), v_theta(phi_q)> + b_theta(phi_q).

    Permutation-equivariant by construction: every candidate goes through the SAME v_theta / b_theta,
    and no term couples two candidates. Adding, removing or reordering candidates cannot change any
    other candidate's score -- unit-tested in `selftest`, not asserted in a comment.

    NOTE on the MCMIPL dueling borrow (POSITION_VS_MODERN_CRS.md B.1 #1): a state-only value term
    f_V(s) CANCELS EXACTLY in a softmax policy, so it buys nothing in E1. Its correct home is E3, and
    there RLOO's leave-one-out baseline is the same variance fix with zero extra parameters. Not
    implemented here, deliberately.
    """

    def __init__(self, d_state, d_phi, d_k=128, d_h=256):
        super().__init__()
        self.q = nn.Sequential(nn.Linear(d_state, d_h), nn.ReLU(), nn.Linear(d_h, d_k))
        self.k = nn.Sequential(nn.Linear(d_phi, d_h), nn.ReLU(), nn.Linear(d_h, d_k))
        self.b = nn.Linear(d_phi, 1)

    def forward(self, h, phi, asked_mask=None):
        """h (B,d_state) | phi (P,d_phi) | asked_mask (B,P) bool, True = already asked.
        Returns logits (B,P) with already-asked candidates at -inf (a question is never asked twice)."""
        s = self.q(h) @ self.k(phi).T + self.b(phi).squeeze(-1)[None, :]
        if asked_mask is not None:
            s = s.masked_fill(asked_mask, float("-inf"))
        return s


class PointerScorer(EquivariantScorer):
    """THE ABLATION, not the primary (design sheet B.7). Adds candidate-candidate attention so the
    policy can express "do not ask two near-duplicates" -- which greedy expresses only by refolding.
    It is a strict superset of the equivariant scorer and therefore strictly more able to overfit 202
    distinct targets, which is why it is the ablation and not the default."""

    def __init__(self, d_state, d_phi, d_k=128, d_h=256, heads=4):
        super().__init__(d_state, d_phi, d_k, d_h)
        self.attn = nn.MultiheadAttention(d_k, heads, batch_first=True)

    def forward(self, h, phi, asked_mask=None):
        kv = self.k(phi)[None]                                   # (1,P,d_k)
        kv, _ = self.attn(kv, kv, kv)
        s = self.q(h) @ kv[0].T + self.b(phi).squeeze(-1)[None, :]
        if asked_mask is not None:
            s = s.masked_fill(asked_mask, float("-inf"))
        return s


class FlatHead(nn.Module):
    """CONTROL ARM ONLY (HARD RULE 2, "start simple"): a 1,000-way head, i.e. a lookup table over this
    pool. Its failure was MEASURED in June 2026 (H3) and reproducing it on the new stack is the anchor.
    It cannot make the scalability claim at all -- it has no way to score a candidate outside the pool.
    NEVER the primary."""

    def __init__(self, d_state, n_pool, d_h=256):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(d_state, d_h), nn.ReLU(), nn.Linear(d_h, n_pool))

    def forward(self, h, phi, asked_mask=None):
        s = self.mlp(h)
        if asked_mask is not None:
            s = s.masked_fill(asked_mask, float("-inf"))
        return s


class Policy(nn.Module):
    def __init__(self, d_z, d_phi, n_pool, arch="equivariant", state="both", use_pos=True,
                 d_k=128, d_h=256):
        super().__init__()
        self.enc = StateEncoder(d_z, d_phi, d_h=d_h, mode=state, use_pos=use_pos)
        if arch == "equivariant":
            self.head = EquivariantScorer(self.enc.d_out, d_phi, d_k, d_h)
        elif arch == "pointer":
            self.head = PointerScorer(self.enc.d_out, d_phi, d_k, d_h)
        elif arch == "flat":
            self.head = FlatHead(self.enc.d_out, n_pool, d_h)
        else:
            raise SystemExit(f"unknown --arch {arch}")
        self.arch = arch

    def forward(self, z, hist_phi, hist_out, hist_mask, pos, phi, asked_mask=None):
        return self.head(self.enc(z, hist_phi, hist_out, hist_mask, pos), phi, asked_mask)

    def n_params(self):
        return sum(p.numel() for p in self.parameters())


# ============================================================================ candidate descriptors
def build_phi(E, pool, norm=None):
    """The FROZEN per-candidate descriptor. Nothing here is learned, so nothing can drift off the
    certified ruler.

    `norm` = the (mu, sd) standardisation to apply. It is FITTED on the training pool and then PASSED
    IN for every later pool -- above all for the full 18,359-item + 1,031-concept catalogue of the
    scalability demonstration (design sheet D.4). Re-fitting it per pool would silently move every
    descriptor, so the out-of-vocab picks would be an artefact of a changed input scale rather than a
    property of the trained policy.

      item i     -> decoder row W_d[i] (d), log1p(train count), 0, 0, channel one-hot (1,0)
      concept c  -> cv3 concept embedding emb.weight[c] (d), 0, answer rate, signed-value std,
                    channel one-hot (0,1)

    Both channels are scored by ONE head, so no channel prior is hard-coded -- which matters because
    Act 4's whole claim is that answerability, not a prior, orders the channels (the tree's label set
    is 45.4% item / 54.6% concept, user-weighted).
    """
    import numpy as _np
    from strategy_channel_suite import CONCSTATS, N_TRAIN_USERS
    ctx = E["ctx"]
    Wd = ctx.Wd.detach().numpy().astype(_np.float32)             # (n_items, d)
    cemb = E["net"].emb.weight.detach().numpy().astype(_np.float32)   # (n_concepts, d)
    z = _np.load(CONCSTATS)
    c_ar = (z["n"] / N_TRAIN_USERS).astype(_np.float32)
    c_sd = _np.asarray(z["std"], _np.float32)
    cnt = _np.asarray(ctx.cnt, _np.float64)
    d = Wd.shape[1]
    phi = _np.zeros((len(pool), d + 3 + 2), _np.float32)
    for j, (ch, i) in enumerate(pool):
        if int(ch) == 0:
            phi[j, :d] = Wd[int(i)]
            phi[j, d] = _np.log1p(cnt[int(i)])
            phi[j, d + 3] = 1.0
        else:
            phi[j, :d] = cemb[int(i)]
            phi[j, d + 1] = c_ar[int(i)]
            phi[j, d + 2] = c_sd[int(i)]
            phi[j, d + 4] = 1.0
    if norm is None:
        mu, sd = phi.mean(0, keepdims=True), phi.std(0, keepdims=True) + 1e-6
    else:
        mu, sd = norm                                        # the TRAINING pool's statistics
    return (phi - mu) / sd, (mu, sd)


# ============================================================================ gates / diagnostics
def top1_agreement(pred, label):
    """G-E1a. June's calibration: the teacher that WORKED cloned at 49% top-1; the one that failed at
    14%. The gate is >= 50% on held-out policy-dev states."""
    return float((np.asarray(pred) == np.asarray(label)).mean())


def collapse_diagnostics(interviews):
    """G-E1c. `interviews` = list of per-user question tuples (the realised 16-question orders).

    Two numbers, and both must be reported:
      distinct   how many DIFFERENT interviews the policy produced. `conlinucb_bs_static` produced
                 ONE -- a static spanner order wearing an adaptive label. The gate is > 3.
      max_share  the fraction of users on the single most common interview. A policy at 0.99 has
                 collapsed whatever its `distinct` count says.
    """
    from collections import Counter
    c = Counter(tuple(x) for x in interviews)
    n = max(len(interviews), 1)
    return {"n_distinct_interviews": len(c), "max_share": max(c.values()) / n,
            "top5": [(list(k), v) for k, v in c.most_common(5)]}


def pep_divergence(pairs):
    """G-E1c, second half -- the Pep (arXiv:2602.15012) adaptivity diagnostic:

        of the pairs of users who were asked the same question and answered it DIFFERENTLY, what
        fraction received a DIFFERENT next question?

    `pairs` = iterable of (answer_u, answer_v, next_u, next_v) at a common asked question. Report it
    against the TREE's own value on the same cohort, never against Pep's published range (abstract-only
    numbers; the borrow list forbids quoting them as a bar).

    ⚠ It detects COLLAPSE. It does NOT license a routing claim: D2 measured that our tree diverges
    WITHOUT routing (answer-congruence null, matched-availability p=0.82, branch-label shuffle p=0.61).
    """
    n = d = 0
    for au, av, nu, nv in pairs:
        if au == av:
            continue
        n += 1
        d += int(nu != nv)
    return {"n_pairs": n, "divergence_rate": (d / n) if n else None}


# ============================================================================ self-test (no fit)
def selftest():
    """Pure-tensor invariance checks on RANDOM weights. No data, no optimiser, no fitting of any kind.
    This is a unit test, not a smoke run."""
    torch.manual_seed(SEED)
    B, P, T, dz, dphi = 7, 40, 5, 16, 11
    for arch in ("equivariant", "pointer", "flat"):
        for state in ("z", "hist", "both"):
            pol = Policy(dz, dphi, P, arch=arch, state=state)
            z = torch.randn(B, dz); phi = torch.randn(P, dphi)
            hp = torch.randn(B, T, dphi)
            ho = torch.randint(0, N_OUTCOMES, (B, T))
            hm = (torch.rand(B, T) > 0.3).float()
            pos = torch.randn(B, 2)
            s = pol(z, hp, ho, hm, pos, phi)
            assert s.shape == (B, P), f"{arch}/{state}: logits {tuple(s.shape)} != {(B, P)}"
            # (1) masking: an already-asked candidate is -inf and can never be argmax
            am = torch.zeros(B, P, dtype=torch.bool); am[:, 0] = True
            sm = pol(z, hp, ho, hm, pos, phi, am)
            assert torch.isinf(sm[:, 0]).all() and (sm.argmax(1) != 0).all(), \
                f"{arch}/{state}: the asked mask does not exclude a candidate"
            if arch == "equivariant":
                # (2) permutation EQUIVARIANCE, exact
                perm = torch.randperm(P)
                sp = pol(z, hp, ho, hm, pos, phi[perm])
                assert torch.allclose(sp, s[:, perm], atol=1e-6), "scores are not permutation-equivariant"
                # (3) INDEPENDENCE: dropping a candidate leaves the others' scores untouched
                keep = torch.arange(P) != 3
                sd = pol(z, hp, ho, hm, pos, phi[keep])
                assert torch.allclose(sd, s[:, keep], atol=1e-6), "a dropped candidate moved the others"
            if state in ("hist", "both"):
                # (4) the THREE outcomes are genuinely distinct: relabelling did-not-answer as
                #     dislike must CHANGE the state. A two-outcome encoding would pass silently.
                ho2 = ho.clone(); ho2[ho2 == OUTCOME["noanswer"]] = OUTCOME["dislike"]
                if not torch.equal(ho, ho2):
                    s2 = pol(z, hp, ho2, hm, pos, phi)
                    assert not torch.allclose(s2, s, atol=1e-6), \
                        "collapsing did-not-answer into dislike did not change the state -- the " \
                        "three-outcome encoding is not load-bearing"
    log("[selftest] PASS: shapes, asked-masking, permutation equivariance, candidate independence, "
        "and three-outcome distinctness -- on random weights, nothing was fitted")
    return True


# ============================================================================ the data
TRAJ_DIR = os.path.join(OUTDIR, "policy_traj")
STATIC_JSON = os.path.join(OUTDIR, "greedy_static_policyB_combined.json")
TREE16_JSON = os.path.join(OUTDIR, "ndcgtree16.json")
# Banked headline numbers, asserted rather than trusted (`banked-global-greedy-is-a-policy-A-artifact`:
# when a new arm nests in a banked one, assert the banked number -- a free contract check).
STATIC_VAL_FULL = 0.25561016113602253
STATIC_VAL_TAIL = 0.12366492106552922
TREE_VAL_GAIN_FULL = 0.027939556166529655            # ndcgtree16.json cohort_weighted["16"]
TREE_VAL_GAIN_TAIL = 0.010548197664320469
VAL_M0_FULL = 0.15686538                              # G-E1e canonical snap
TOTAL_L = 16
# E1's measured validation result -- the SPANNING REGRESSION GATE of design sheet G ("no stage may
# score below the previous stage's number"). Phase 3 fine-tuning must not fall below this.
E1_VAL_FULL = 0.283549718
E1_VAL_GAIN_FULL = 0.027939557
# bank-alone on the SAME cohort and ruler (F.4: the bar is the BANK, not the tree). Its TEST value is
# +0.01430 full / +0.02199 tail; this is its VALIDATION value, and the two are never compared directly.
STATIC_ITEMS_VAL_FULL = 0.2428081265667196
STATIC_ITEMS_VAL_TAIL = 0.10111517752587719
DEEP_LABELS = os.path.join(OUTDIR, "policy_deep_labels.json")
E1_PER_USER = os.path.join(OUTDIR, "policy_main_equivariant_both_gates_per_user.npz")
DEEPER_PER_USER = os.path.join(OUTDIR, "policy_deeper_tree_control_per_user.npz")


def apply_deep_labels(D, path, pool, shuffle=False, seed=SEED):
    """PHASE 3 / design sheet G.B -- the labels one level BELOW the tree's floor stops.

    For every user the frozen tree reaches a leaf at depth d, asks the leaf's own splitter at position
    d, and the user's ternary answer to it selects one of 42 below-floor states. `policy_deep.py
    --mode labels` holds the EXACT MEASURED-NDCG argmax at each of those states. This function
    REPLACES the label at position d+1 with it.

    ★ WHY ONLY POSITION d+1, AND WHY EVERY OTHER POSITION KEEPS THE TREE'S LABEL. The cached state
    (`z_t` and the ternary history) is valid only where the extended trajectory and the tree's
    trajectory coincide -- and they coincide at every position up to and including d+1, because the
    first d+1 questions are identical. From d+2 onward the extended trajectory would carry a different
    asked list, so its `z` is not the one on disk; training there would be training on a state that
    never occurred. So positions != d+1 keep the tree's labels on the tree's states, and THAT is the
    replay: ~15/16 of the 2.25M examples are the original E1 supervision, which is the anti-forgetting
    mechanism the author's requirement 1 asks for and is not optional.

    `shuffle=True` is the HARD RULE 5 control: the 42 branch questions are permuted ACROSS branches,
    so every user still gets one extra deliberately-chosen question at exactly the same position --
    just not the one measured for their own state.
    """
    dl = json.load(open(path))
    assert dl["test_touched"] is False and dl["split"] == "validation"
    pool_index = {(int(a), int(b)): i for i, (a, b) in enumerate(pool)}
    gmap = {}
    for r in dl["states"]:
        q = (int(r["question"][0]), int(r["question"][1]))
        assert pool_index[q] == int(r["pool_idx"]), "the deep label's pool index does not match"
        gmap[(r["leaf"], int(r["branch_idx"]))] = int(r["pool_idx"])
    if shuffle:
        rng = np.random.default_rng(seed)
        keys = sorted(gmap)
        vals = [gmap[k] for k in keys]
        perm = rng.permutation(len(keys))
        gmap = {keys[i]: int(vals[perm[i]]) for i in range(len(keys))}
    leaf_ids = [str(s) for s in D["leaf_ids"]]
    depth = {nid: (0 if nid == "root" else nid.count(".")) for nid in leaf_ids}
    lab = np.asarray(D["label_idx"], np.int64).copy()
    outc = np.asarray(D["outcome"], np.int64)
    leaf = np.asarray(D["leaf"], np.int64)
    # A (leaf, branch) can legitimately have NO measured state: on validation the branch was empty
    # (the splitter's answer rate is 1.000, so the did-not-answer child has no users) and the sweep
    # therefore never scored it. TRAIN is a different population and does put users there. Those users
    # have no measured question for their state, so they FALL THROUGH to the parent's tail -- they
    # keep the tree's label. The assert is not loosened; the empty branch is made representable, and
    # the fall-through is counted and reported rather than being silent.
    declared_missing = {(r["leaf"], int(r["branch_idx"]))
                        for r in dl.get("branches_without_a_measured_state", [])}
    n_rep, pos_hist = 0, np.zeros(TOTAL_L, np.int64)
    n_fallthrough, fallthrough_detail = 0, []
    for li, nid in enumerate(leaf_ids):
        d = depth[nid]
        assert d + 1 < TOTAL_L, f"leaf {nid} is at depth {d}; there is no position {d + 1}"
        m = leaf == li
        for br in range(3):
            mm = m & (outc[:, d] == br)
            if not mm.any():
                continue
            g = gmap.get((nid, br))
            if g is None:
                assert (nid, br) in declared_missing, (
                    f"no deep label for leaf {nid} branch {br}, and the label artifact does NOT "
                    f"declare that branch as unmeasured. A missing label that nobody declared is a "
                    f"bug, not a fall-through.")
                n_fallthrough += int(mm.sum())
                fallthrough_detail.append({"leaf": nid, "branch": br, "n_train_users": int(mm.sum())})
                continue                                     # keep the tree's label at d+1
            lab[mm, d + 1] = g
            n_rep += int(mm.sum()); pos_hist[d + 1] += int(mm.sum())
    return lab.astype(np.int16), {"n_examples_relabelled": int(n_rep),
                                  "n_examples_fallthrough_unmeasured_branch": int(n_fallthrough),
                                  "fallthrough_detail": fallthrough_detail,
                                  "frac_of_all_examples": float(n_rep) / (lab.shape[0] * TOTAL_L),
                                  "by_position": [int(x) for x in pos_hist],
                                  "n_deep_states": len(gmap), "shuffled": bool(shuffle),
                                  "n_distinct_deep_questions": len(set(gmap.values()))}


def load_traj(train=True):
    """The trajectory bank produced by `policy_traj.py`.

    TRAIN side: the 140,768 train users, 16 (state -> label) examples each, ~2.25M. VALIDATION side:
    the same object on the 10,000 validation users -- NOT training data. It is (a) the held-out states
    G-E1a measures top-1 agreement on and (b) the tree's own realised interviews the policy's rollout
    is compared against.
    """
    man = json.load(open(os.path.join(TRAJ_DIR, "manifest.json")))
    pool = [(int(a), int(b)) for a, b in man["pool"]]
    if not train:
        z = np.load(os.path.join(TRAJ_DIR, "val_traj.npz"))
        vpool = [(int(a), int(b)) for a, b in z["pool"]]
        assert vpool == pool, "the validation trajectory pool is not the training pool"
        Z = np.load(os.path.join(TRAJ_DIR, "z_val.npy"), mmap_mode="r")
        assert Z.shape[0] == len(z["uid"]) and Z.shape[1] == TOTAL_L
        return {"uid": z["uid"], "label_idx": z["label_idx"], "outcome": z["outcome"],
                "leaf": z["leaf"], "leaf_ids": [str(s) for s in z["leaf_ids"]], "Z": Z,
                "pool": pool, "sig": man["sig"]}
    n_done = sum(v["n_users"] for v in man["done"].values())
    assert n_done == man["sig"]["n_train"], (
        f"the trajectory bank is INCOMPLETE: {n_done:,} of {man['sig']['n_train']:,} train users. "
        f"Refusing to train on a partial cohort (HARD RULE 1).")
    uid, lab, out, leaf = [], [], [], []
    for key in sorted(man["done"], key=lambda k: int(k.split("_")[0])):
        z = np.load(os.path.join(TRAJ_DIR, f"meta_{key}.npz"))
        uid.append(z["uid"]); lab.append(z["label_idx"]); out.append(z["outcome"]); leaf.append(z["leaf"])
    uid = np.concatenate(uid); lab = np.concatenate(lab)
    out = np.concatenate(out); leaf = np.concatenate(leaf)
    assert np.array_equal(np.sort(uid), np.arange(man["sig"]["n_train"])), \
        "the train uids are not exactly 0..140,767 -- a user is missing or duplicated"
    o = np.argsort(uid)
    Z = np.load(os.path.join(TRAJ_DIR, "z_train.npy"), mmap_mode="r")
    assert Z.shape == (man["sig"]["n_train"], TOTAL_L, man["sig"]["d"])
    return {"uid": uid[o], "label_idx": lab[o], "outcome": out[o], "leaf": leaf[o], "Z": Z,
            "pool": pool, "sig": man["sig"], "leaf_ids": man["leaf_ids"]}


def make_batch(D, phi_t, u, t, n_pool, labels=None):
    """Tensors for a batch of (user, position) examples. The history is EXPLICIT and TERNARY."""
    B = len(u)
    Z = D["Z"]; lab = D["label_idx"]; out = D["outcome"]
    z = torch.from_numpy(np.ascontiguousarray(Z[u, t]).astype(np.float32))
    T = TOTAL_L - 1
    hq = np.zeros((B, T), np.int64); ho = np.zeros((B, T), np.int64)
    hm = np.zeros((B, T), np.float32)
    for j in range(T):
        m = t > j
        if not m.any():
            break
        hq[m, j] = lab[u[m], j]
        ho[m, j] = out[u[m], j]
        hm[m, j] = 1.0
    asked = np.zeros((B, n_pool), bool)
    asked[np.repeat(np.arange(B), T)[hm.ravel() > 0], hq.ravel()[hm.ravel() > 0]] = True
    y = lab[u, t].astype(np.int64) if labels is None else labels
    # A question is never asked twice, so the already-asked are -inf. Under the SHUFFLE control the
    # permuted target can land on an already-asked candidate; that one entry is re-opened, because a
    # control whose loss is +inf measures nothing.
    asked[np.arange(B), y] = False
    pos = np.stack([t, TOTAL_L - t], 1).astype(np.float32)
    return (z, torch.from_numpy(phi_t[hq]), torch.from_numpy(ho), torch.from_numpy(hm),
            torch.from_numpy(pos), torch.from_numpy(asked), torch.from_numpy(y))


# ============================================================================ the validation rollout
def rollout(E, pol, phi_t, pool, rows, L=TOTAL_L, batch=1000, score_at=(TOTAL_L,)):
    """THE POLICY'S OWN INTERVIEWS. It picks its own questions; it does not replay the tree's.

    Every turn: the policy scores every candidate from the current state, the argmax over the not-yet-
    asked is asked, the user's own ternary answer is taken by the SAME rule the tree branches on
    (`branch_of_per_user`, gated against `branch_of`), and the answer is folded. The reported NDCG
    comes off the existing scoring funnel (`genv_eval_multi`), not a second scorer -- the intermediate
    turns pass `score=False`, which skips only the decoder matmul nobody reads.
    """
    from strategy_channel_suite import genv_latent, genv_eval_multi
    from ndcg_tree import branch_of_per_user
    n = len(rows)
    P = len(pool)
    asked = np.full((n, L), -1, np.int64)
    outs = np.full((n, L), -1, np.int8)
    asked_mask = np.zeros((n, P), bool)
    z = genv_latent(E, [], rows).numpy()
    curves = {}
    phi_g = torch.from_numpy(phi_t)
    pol.eval()
    for t in range(L):
        hq = np.zeros((n, TOTAL_L - 1), np.int64); ho = np.zeros((n, TOTAL_L - 1), np.int64)
        hm = np.zeros((n, TOTAL_L - 1), np.float32)
        for j in range(t):
            hq[:, j] = asked[:, j]; ho[:, j] = outs[:, j]; hm[:, j] = 1.0
        pos = np.tile(np.asarray([[t, TOTAL_L - t]], np.float32), (n, 1))
        with torch.no_grad():
            for st in range(0, n, batch):
                sl = slice(st, min(st + batch, n))
                s = pol(torch.from_numpy(z[sl]), torch.from_numpy(phi_t[hq[sl]]),
                        torch.from_numpy(ho[sl]), torch.from_numpy(hm[sl]),
                        torch.from_numpy(pos[sl]), phi_g, torch.from_numpy(asked_mask[sl]))
                asked[sl, t] = s.argmax(1).numpy()
        outs[:, t] = branch_of_per_user(E, rows, [pool[int(j)] for j in asked[:, t]])
        asked_mask[np.arange(n), asked[:, t]] = True
        need_score = (t + 1) in score_at
        iseq = [[pool[int(j)][1] for j in asked[u, :t + 1] if pool[int(j)][0] == 0] for u in range(n)]
        cseq = [[pool[int(j)][1] for j in asked[u, :t + 1] if pool[int(j)][0] == 1] for u in range(n)]
        r = genv_eval_multi(E, rows, iseq, cseq, return_z=(t + 1 < L), score=need_score)
        if need_score:
            curves[t + 1] = (r[0], r[1])
        if t + 1 < L:
            z = r[-1].numpy()
    assert (asked >= 0).all() and (outs >= 0).all()
    assert all(len(set(asked[u].tolist())) == L for u in range(min(n, 2000))), \
        "a question was asked twice inside one interview"
    return {"asked": asked, "outs": outs, "curves": curves}


def teacher_forced_agreement(pol, phi_t, D, n_pool, batch=2048):
    """G-E1a. Top-1 agreement with the tree's label at the tree's OWN states, on the held-out
    validation users -- the states are the teacher's, so this measures imitation, not rollout luck.
    Reported overall and per position (design sheet B.9: early failure = answerability, late failure =
    fragmentation)."""
    n = len(D["uid"])
    hit = np.zeros(TOTAL_L); tot = np.zeros(TOTAL_L)
    pol.eval()
    with torch.no_grad():
        for t in range(TOTAL_L):
            for st in range(0, n, batch):
                u = np.arange(st, min(st + batch, n))
                z, hp, ho, hm, pos, am, y = make_batch(D, phi_t, u, np.full(len(u), t), n_pool)
                s = pol(z, hp, ho, hm, pos, torch.from_numpy(phi_t), am)
                hit[t] += float((s.argmax(1) == y).sum()); tot[t] += len(u)
    return {"top1": float(hit.sum() / tot.sum()),
            "top1_by_position": [float(a / b) for a, b in zip(hit, tot)]}


# ============================================================================ statistics
def paired_bootstrap(diff, n_boot=10000, seed=SEED):
    """The paired percentile bootstrap over per-user DIFFERENCES. NEVER the unpaired
    `1.96*sqrt(2)*SE` form -- that is the signature the chapter's own tie claim died on
    (`paired-test-kills-the-tie-claim`)."""
    d = np.asarray(diff, np.float64)
    d = d[~np.isnan(d)]
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(n_boot, len(d)))
    bs = d[idx].mean(1)
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return {"mean": float(d.mean()), "lo": float(lo), "hi": float(hi), "n": int(len(d)),
            "excludes_zero": bool(lo > 0 or hi < 0)}


def pep_divergence_from_arrays(asked, outs):
    """G-E1c second half, exactly and in O(n): over all pairs of users asked the SAME question at the
    same position who ANSWERED IT DIFFERENTLY, the fraction that received a DIFFERENT next question.

    ⚠ It detects COLLAPSE. It licenses no routing claim -- D2 measured that the tree diverges WITHOUT
    routing (answer-congruence null, p=0.82 / p=0.61).
    """
    from collections import Counter
    n, L = asked.shape
    tot = same = 0
    for t in range(L - 1):
        byq = {}
        for u in range(n):
            byq.setdefault(int(asked[u, t]), {}).setdefault(int(outs[u, t]), Counter())[
                int(asked[u, t + 1])] += 1
        for _q, groups in byq.items():
            ks = sorted(groups)
            for i in range(len(ks)):
                for j in range(i + 1, len(ks)):
                    a, b = groups[ks[i]], groups[ks[j]]
                    tot += sum(a.values()) * sum(b.values())
                    same += sum(a[k] * b[k] for k in a if k in b)
    return {"n_pairs": int(tot), "divergence_rate": float(1.0 - same / tot) if tot else None}


def channel_mix(asked, pool):
    return float(np.mean([1.0 if pool[int(j)][0] == 1 else 0.0 for j in asked.ravel()]))


# ============================================================================ training
def train(args):
    """E1 imitation. Cross-entropy over the 1,000-candidate pool against the tree's exact-NDCG argmax,
    fitted on the 140,768 TRAIN users and SELECTED on real interview NDCG over the 10,000 VALIDATION
    users (author ruling D.6b). The 8,000/2,000 in-sample dev split of A.2.1 is superseded and gone.

    ⚠ AUTHOR GATE: `--i_have_author_goahead`. The author said "start on policy" (2026-08-13), which
    authorises E1 training on validation-SELECTED, train-FITTED data. It does NOT authorise a test
    look, E2, or regrowing the tree.
    """
    assert args.i_have_author_goahead, (
        "REFUSING TO TRAIN. Pass --i_have_author_goahead only when the author has said go.")
    from strategy_channel_suite import (build_greedy_env, contract_selftest, genv_eval, genv_mean,
                                        CERT_TOWER, CERT_CONC)
    from ndcg_tree import build_pool, leak_assert, branch_of_selftest
    torch.manual_seed(SEED); np.random.seed(SEED)
    rng = np.random.default_rng(SEED)
    t00 = time.time()

    Dtr = load_traj(train=True)
    Dva = load_traj(train=False)
    pool = Dtr["pool"]; P = len(pool)
    log(f"[data] train {len(Dtr['uid']):,} users x {TOTAL_L} = {len(Dtr['uid']) * TOTAL_L:,} examples; "
        f"val {len(Dva['uid']):,} users; pool {P}; "
        f"{len(np.unique(Dtr['label_idx']))} distinct label questions behind them "
        f"(design sheet A.5: the label set's real information content is 202 states / 172 questions)")

    # ---- the environment, the ruler, and the comparator ------------------------------------------
    E = build_greedy_env(args.snapshot, "validation", args.concept_ckpt, f"policynet/{args.arm}")
    contract_selftest(E, E["rows"])
    leak_assert(E, E["rows"], "policynet/val")
    rows = E["rows"]
    assert len(rows) == 10000
    assert list(np.asarray(Dva["uid"], np.int64)) == list(np.asarray(rows, np.int64)), \
        "the validation trajectory rows are not the env's rows, in order"
    f0, t0_ = genv_mean(E, [], rows)
    assert round(f0, 8) == VAL_M0_FULL, f"G-E1e canonical snap FAILED: {f0!r}"
    log(f"[G-E1e] m=0 intercept full={f0!r} tail={t0_!r} -- canonical, exact")
    pool_e, item_pool, conc_pool = build_pool(E, 500, 500)
    assert pool_e == pool, "the env's pool is not the trajectory bank's pool"
    branch_of_selftest(E, rows[:2000], [pool[0], pool[500], (1, 114)])

    static_seq = [(int(a), int(b)) for a, b in json.load(open(STATIC_JSON))["sequence"]]
    sf, stl = genv_eval(E, static_seq, rows)
    assert abs(float(np.nanmean(sf)) - STATIC_VAL_FULL) < 1e-12, (
        f"BANKED-HEADLINE ASSERT FAILED: greedy_static_policyB_combined rescores to "
        f"{float(np.nanmean(sf))!r}, banked {STATIC_VAL_FULL}")
    log(f"[banked] greedy_static_policyB_combined K=16 on validation reproduces EXACTLY: "
        f"full={float(np.nanmean(sf))!r} tail={float(np.nanmean(stl))!r}")

    phi_t, norm = build_phi(E, pool)
    phi_g = torch.from_numpy(phi_t)
    log(f"[phi] {phi_t.shape[1]}-dim frozen descriptors for {P} candidates")

    pol = Policy(E["ctx"].d, phi_t.shape[1], P, arch=args.arch, state=args.state,
                 use_pos=not args.no_pos)
    log(f"[model] arch={args.arch} state={args.state} pos={not args.no_pos} "
        f"params={pol.n_params():,}")
    init_from = None
    if args.init_ckpt:
        # PHASE 3 fine-tune: start from the ANCHOR (design sheet G.A) rather than from scratch, so the
        # spanning regression gate has something to regress against and the replay has something to
        # protect. Every structural field is asserted -- a checkpoint that does not match is refused,
        # never silently adapted.
        b = torch.load(args.init_ckpt, map_location="cpu")
        for k, want in (("arch", args.arch), ("state", args.state), ("use_pos", not args.no_pos),
                        ("d_phi", int(phi_t.shape[1])), ("n_pool", P)):
            assert b[k] == want, f"--init_ckpt {k}={b[k]!r}, this run needs {want!r}"
        assert [(int(a), int(c)) for a, c in b["pool"]] == pool, "--init_ckpt was fitted on a "\
                                                                 "different pool"
        assert np.array_equal(b["norm_mu"], norm[0]) and np.array_equal(b["norm_sd"], norm[1]), \
            "--init_ckpt's descriptor standardisation differs from this run's"
        pol.load_state_dict(b["model"])
        init_from = {"ckpt": os.path.basename(args.init_ckpt), "epoch": int(b["epoch"]),
                     "val_full": float(b["val_full"]), "arm": b.get("arm")}
        log(f"[init] fine-tuning from {init_from['ckpt']} (its epoch {init_from['epoch']}, "
            f"val full {init_from['val_full']:.6f})")
    opt = torch.optim.Adam(pol.parameters(), lr=args.lr)
    lossf = nn.CrossEntropyLoss()

    N = len(Dtr["uid"])
    n_ex = N * TOTAL_L
    labels_all = None
    deep_info = None
    if args.arm == "shuffle":                                # G-E1d, HARD RULE 5
        lab = Dtr["label_idx"].copy()
        for t in range(TOTAL_L):                             # permute users WITHIN a position
            lab[:, t] = lab[rng.permutation(N), t]
        labels_all = lab
        log("[arm] SHUFFLE CONTROL: the label at every (user, position) is another user's label at "
            "the same position. The states are untouched. This arm must NOT clear G-E1b.")
    elif args.arm in ("deep", "deepshuffle"):                # PHASE 3, design sheet G.B
        labels_all, deep_info = apply_deep_labels(Dtr, args.deep_labels, pool,
                                                  shuffle=(args.arm == "deepshuffle"))
        log(f"[arm] {'DEEP-SHUFFLE CONTROL' if args.arm == 'deepshuffle' else 'DEEP'}: "
            f"{deep_info['n_examples_relabelled']:,} of {n_ex:,} examples "
            f"({deep_info['frac_of_all_examples'] * 100:.2f}%) carry a below-floor label; the other "
            f"{100 - deep_info['frac_of_all_examples'] * 100:.2f}% are the ORIGINAL tree labels -- "
            f"that is the replay, and it is the anti-forgetting mechanism, not a convenience.")
        log(f"[arm] relabelled positions: "
            f"{[t for t, c in enumerate(deep_info['by_position']) if c]}")

    steps = int(np.ceil(n_ex / args.batch))
    best = {"full": -1e9}
    hist = []
    # HARD RULE 11(e): a unique output stem per run. Two runs never share an output path.
    stem = (f"policy_{args.arm}_{args.arch}_{args.state}{'_nopos' if args.no_pos else ''}"
            f"{('_' + args.tag) if args.tag else ''}")
    ckpt = os.path.join(OUTDIR, stem + ".pt")
    for suffix in (".pt", "_history.json", "_gates.json"):
        p_ = os.path.join(OUTDIR, stem + suffix)
        assert not os.path.exists(p_) or args.overwrite, (
            f"{os.path.basename(p_)} already exists. Pass --tag to name this run, or --overwrite if "
            f"you really mean to replace a banked artifact (HARD RULE 11e).")
    for ep in range(1, args.epochs + 1):
        pol.train()
        perm = rng.permutation(n_ex)
        tot = 0.0; seen = 0; hit = 0
        te = time.time()
        for si in range(steps):
            e = perm[si * args.batch:(si + 1) * args.batch]
            if not len(e):
                continue
            u = (e // TOTAL_L).astype(np.int64); t = (e % TOTAL_L).astype(np.int64)
            y = labels_all[u, t].astype(np.int64) if labels_all is not None else None
            z, hp, ho, hm, pos, am, yy = make_batch(Dtr, phi_t, u, t, P, labels=y)
            s = pol(z, hp, ho, hm, pos, phi_g, am)
            loss = lossf(s, yy)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss) * len(e); seen += len(e)
            hit += int((s.argmax(1) == yy).sum())
            if (si + 1) % 500 == 0:
                log(f"[ep{ep} {si + 1}/{steps}] loss {tot / seen:.4f} top1 {hit / seen:.4f} "
                    f"({(time.time() - te) / 60:.1f}m)")
        tr_loss, tr_top1 = tot / seen, hit / seen
        ag = teacher_forced_agreement(pol, phi_t, Dva, P)
        ro = rollout(E, pol, phi_t, pool, rows)
        vf, vt = ro["curves"][TOTAL_L]
        rec = {"epoch": ep, "train_loss": tr_loss, "train_top1": tr_top1,
               "val_top1_teacher_forced": ag["top1"], "val_full": float(np.nanmean(vf)),
               "val_tail": float(np.nanmean(vt)),
               "val_gain_full_vs_static": float(np.nanmean(vf)) - STATIC_VAL_FULL,
               "n_distinct_interviews": len({tuple(r) for r in ro["asked"]}),
               "minutes": round((time.time() - te) / 60, 2)}
        hist.append(rec)
        log(f"[ep{ep}] loss {tr_loss:.4f} | train top1 {tr_top1:.4f} | VAL top1 {ag['top1']:.4f} | "
            f"VAL full {rec['val_full']:.5f} (static {STATIC_VAL_FULL:.5f}, "
            f"gain {rec['val_gain_full_vs_static']:+.5f}) | tail {rec['val_tail']:.5f} | "
            f"{rec['n_distinct_interviews']} distinct interviews | {rec['minutes']:.1f}m")
        if rec["val_full"] > best["full"]:
            best = {"full": rec["val_full"], "epoch": ep, "rec": rec,
                    "vf": vf, "vt": vt, "asked": ro["asked"], "outs": ro["outs"], "ag": ag}
            torch.save({"model": pol.state_dict(), "arch": args.arch, "state": args.state,
                        "use_pos": not args.no_pos, "d_phi": int(phi_t.shape[1]), "n_pool": P,
                        "epoch": ep, "val_full": rec["val_full"], "val_tail": rec["val_tail"],
                        "norm_mu": norm[0], "norm_sd": norm[1], "arm": args.arm,
                        "pool": [[int(a), int(b)] for a, b in pool], "seed": SEED}, ckpt)
            log(f"[ckpt] epoch {ep} is the best so far -> {os.path.basename(ckpt)}")
        json.dump({"analysis": "policy_net_E1", "arm": args.arm, "arch": args.arch,
                   "state": args.state, "use_pos": not args.no_pos, "history": hist,
                   "init_from": init_from, "deep_labels": deep_info,
                   "test_touched": False, "selection": "real interview NDCG on 10,000 val users"},
                  open(os.path.join(OUTDIR, stem + "_history.json"), "w"), indent=1, default=float)

    # ---- the gates -------------------------------------------------------------------------------
    vf, vt = best["vf"], best["vt"]
    tree_ro = {"asked": Dva["label_idx"], "outs": Dva["outcome"]}
    dev = np.full(len(rows), TOTAL_L, np.int64)              # first position the policy deviates
    for t in range(TOTAL_L):
        m = (best["asked"][:, t] != Dva["label_idx"][:, t]) & (dev == TOTAL_L)
        dev[m] = t
    # ...and the LAST position at which it still differs, which is the other half of "did phase 3
    # actually extend anything": E1 had frac_never_deviates = 1.0000, so both numbers were degenerate.
    diff_pos = (best["asked"] != Dva["label_idx"])
    last = np.full(len(rows), -1, np.int64)
    for t in range(TOTAL_L):
        last[diff_pos[:, t]] = t
    per_pos_dev = [float(diff_pos[:, t].mean()) for t in range(TOTAL_L)]
    gates = {
        "G-E1e_canonical_snap": {"val_m0_full": f0, "expected": VAL_M0_FULL, "pass": True},
        "G-E1a_fidelity": {"top1": best["ag"]["top1"], "bar": 0.50,
                           "pass": best["ag"]["top1"] >= 0.50,
                           "by_position": best["ag"]["top1_by_position"]},
        "G-E1b_gain_vs_static": {
            "full": paired_bootstrap(vf - sf), "tail": paired_bootstrap(vt - stl),
            "comparator": "greedy_static_policyB_combined (validation, K=16)",
            "tree_val_gain_full": TREE_VAL_GAIN_FULL, "tree_val_gain_tail": TREE_VAL_GAIN_TAIL},
        "G-E1c_anti_collapse": {
            "n_distinct_interviews": len({tuple(r) for r in best["asked"]}),
            "max_share": float(max(np.unique([hash(tuple(r)) for r in best["asked"]],
                                             return_counts=True)[1]) / len(rows)),
            "pep_divergence_policy": pep_divergence_from_arrays(best["asked"], best["outs"]),
            "pep_divergence_tree": pep_divergence_from_arrays(tree_ro["asked"], tree_ro["outs"]),
            "bar": "> 3 distinct interviews AND divergence >= the tree's own"},
        "covariate_shift": {
            "mean_first_deviation_position": float(dev.mean()),
            "frac_never_deviates": float((dev == TOTAL_L).mean()),
            "deviation_position_histogram": [int((dev == t).sum()) for t in range(TOTAL_L + 1)],
            "last_deviation_position_histogram": [int((last == t).sum()) for t in range(TOTAL_L)],
            "mean_last_deviation_position_among_deviators":
                float(last[last >= 0].mean()) if (last >= 0).any() else None,
            "frac_users_differing_at_each_position": per_pos_dev,
            "n_positions_differing_mean": float(diff_pos.sum(1).mean()),
            # ★ THE DECIDING MEASUREMENT for the pre-registered comparator asymmetry (policy_deep.py
            # module docstring, 2026-08-14). The deeper-tree control may differ from the tree by
            # EXACTLY ONE question per user; the policy may differ at any position past the floor
            # stop. If the policy also moves ~1 position the asymmetry is theoretical and the
            # comparison is clean; if it moves many, the comparison measures the policy's extra
            # FREEDOM as much as its extra SKILL, and the write-up must say so.
            "n_positions_differing_histogram": [int((diff_pos.sum(1) == k).sum())
                                                for k in range(TOTAL_L + 1)],
            "n_positions_differing_median": float(np.median(diff_pos.sum(1))),
            "n_positions_differing_among_deviators_mean":
                float(diff_pos.sum(1)[diff_pos.sum(1) > 0].mean()) if diff_pos.any() else 0.0,
            "control_differs_from_tree_by_at_most": 1,
            "asymmetry_note": ("the deeper-tree control may differ from the frozen tree by EXACTLY "
                               "ONE question per user, by construction. The policy is under no such "
                               "constraint. Read the histogram above before reading any policy-minus-"
                               "deeper-tree number."),
            "reference": ("E1 measured frac_never_deviates = 1.0000 -- a perfect imitator never leaves "
                          "the teacher's trajectory. If phase 3 does not move this number it has not "
                          "extended anything, and that is what the artifact must say.")},
        "channel_mix": {"policy_concept_share": channel_mix(best["asked"], pool),
                        "tree_concept_share": channel_mix(Dva["label_idx"], pool)},
    }
    # ---- PHASE 3 (design sheet G): the spanning regression gate and the comparisons that decide it --
    pol_full = float(np.nanmean(vf))
    gates["G-B-regression_spanning"] = {
        "criterion": "no stage may score below the previous stage's validation number (HARD RULE 8)",
        "stage_A_E1_val_full": E1_VAL_FULL, "this_stage_val_full": pol_full,
        "delta": pol_full - E1_VAL_FULL, "pass": bool(pol_full >= E1_VAL_FULL)}
    if os.path.exists(E1_PER_USER):
        z1 = np.load(E1_PER_USER)
        tree_f = np.asarray(z1["policy_full"], np.float64)      # E1 == the tree, exactly (1e-9)
        tree_t = np.asarray(z1["policy_tail"], np.float64)
        gates["gain_vs_tree"] = {"full": paired_bootstrap(vf - tree_f),
                                 "tail": paired_bootstrap(vt - tree_t),
                                 "comparator": "the frozen NDCG tree's own realised interviews "
                                               "(E1's banked per-user vectors, validation, K=16)"}
    if os.path.exists(DEEPER_PER_USER):
        z2 = np.load(DEEPER_PER_USER)
        gates["gain_vs_deeper_tree"] = {
            "full": paired_bootstrap(vf - np.asarray(z2["deeper_full"], np.float64)),
            "tail": paired_bootstrap(vt - np.asarray(z2["deeper_tail"], np.float64)),
            "comparator": "the DEEPER TREE control (policy_deeper_tree_control.json)",
            "why": ("★ THE HEADLINE. If this does not separate, the policy is a re-parameterisation "
                    "of a deeper tree and phase 3 claims nothing.")}
    gates["bank_alone_val"] = {
        "full": STATIC_VAL_FULL - STATIC_ITEMS_VAL_FULL,
        "tail": STATIC_VAL_TAIL - STATIC_ITEMS_VAL_TAIL,
        "policy_gain_vs_static_full": pol_full - STATIC_VAL_FULL,
        "note": ("F.4: the real bar is BANK-ALONE, not the tree. Bank-alone is +0.01430 full / "
                 "+0.02199 tail ON TEST; the numbers here are its VALIDATION value on this cohort. A "
                 "validation gain is never compared to a test bar directly -- the tree's own measured "
                 "val->test shrinkage on full is 2.70x, which is where the pre-registered +0.011 "
                 "validation no-go bar of G-E2-pre comes from.")}
    gates["G-E1c_anti_collapse"]["pass"] = bool(
        gates["G-E1c_anti_collapse"]["n_distinct_interviews"] > 3 and
        (gates["G-E1c_anti_collapse"]["pep_divergence_policy"]["divergence_rate"] or 0) >=
        (gates["G-E1c_anti_collapse"]["pep_divergence_tree"]["divergence_rate"] or 0))
    out = {"analysis": "policy_net_E1_gates", "design_sheet":
           "docs/design/PAPER_B_POLICY_DESIGN.md", "citable": True, "test_touched": False,
           "cohorts": {"fit": "140,768 TRAIN users (states + frozen-tree labels; NO NDCG)",
                       "select": "10,000 VALIDATION users (real interview NDCG)"},
           "arm": args.arm, "arch": args.arch, "state": args.state, "use_pos": not args.no_pos,
           "init_from": init_from, "deep_labels": deep_info,
           "best_epoch": best["epoch"], "history": hist, "gates": gates,
           "static_val": {"full": float(np.nanmean(sf)), "tail": float(np.nanmean(stl))},
           "policy_val": {"full": float(np.nanmean(vf)), "tail": float(np.nanmean(vt))},
           "tower": E["tower"], "conc_ckpt": E["conc_ckpt"], "mask_policy": "B",
           "pool_contract": E["pool_contract"], "minutes": round((time.time() - t00) / 60, 2)}
    p = os.path.join(OUTDIR, stem + "_gates.json")
    np.savez_compressed(p.replace(".json", "_per_user.npz"), policy_full=vf, policy_tail=vt,
                        static_full=sf, static_tail=stl, asked=best["asked"], outs=best["outs"],
                        rows=np.asarray(rows, np.int64))
    with open(p + ".tmp", "w") as fh:
        json.dump(out, fh, indent=2, default=float)
    os.replace(p + ".tmp", p)
    log(f"[gates] wrote {os.path.basename(p)}")
    for k, v in gates.items():
        log(f"  {k}: {json.dumps(v, default=float)[:300]}")
    return out


# ============================================================================ the scalability demo
def scalability(args):
    """STEP 4 (design sheet D.4). Score the FULL 18,359-item + 1,031-concept catalogue in ONE forward
    pass at inference, with NO retraining, and report what the policy picks OUTSIDE its 1,000-candidate
    training pool.

    This is the author's stated motive made measurable, and per D.4/A.11 it stands independently of
    whether any gain separates. The vocabulary becomes a TEST-TIME INPUT rather than a parameter:
    every candidate is scored by the same dot product against the emitted query vector, so a question
    the policy never saw in training is scorable the moment it has a descriptor. A 1,000-way head
    could not do this at all.

    The descriptor standardisation is the TRAINING pool's (`norm` from the checkpoint) -- re-fitting it
    on the catalogue would move every input and the out-of-pool picks would be an artefact of scale.
    """
    from strategy_channel_suite import build_greedy_env, contract_selftest, CERT_TOWER, CERT_CONC
    from ndcg_tree import build_pool, leak_assert
    blob = torch.load(args.ckpt, map_location="cpu")
    E = build_greedy_env(args.snapshot, "validation", args.concept_ckpt, "policynet/scale")
    contract_selftest(E, E["rows"]); leak_assert(E, E["rows"], "policynet/scale")
    rows = E["rows"]
    pool = [(int(a), int(b)) for a, b in blob["pool"]]
    trained = set(pool)
    full_pool = ([(0, i) for i in range(E["ctx"].ni)] + [(1, c) for c in range(E["C"])])
    log(f"[scale] training pool {len(pool)} -> inference catalogue {len(full_pool)} "
        f"({E['ctx'].ni} items + {E['C']} concepts), no retraining")
    norm = (blob["norm_mu"], blob["norm_sd"])
    phi_small, _ = build_phi(E, pool, norm=norm)
    phi_full, _ = build_phi(E, full_pool, norm=norm)
    pol = Policy(E["ctx"].d, phi_full.shape[1], len(full_pool), arch=blob["arch"],
                 state=blob["state"], use_pos=blob["use_pos"])
    pol.load_state_dict(blob["model"])
    assert blob["arch"] != "flat", (
        "the flat 1,000-way head is the CONTROL arm and cannot score a candidate outside its pool -- "
        "the scalability claim is not available to it, by construction")

    ro_s = rollout(E, pol, phi_small, pool, rows)
    ro_f = rollout(E, pol, phi_full, full_pool, rows)
    picks = ro_f["asked"]
    oop = np.asarray([[0 if full_pool[int(j)] in trained else 1 for j in r] for r in picks])
    from collections import Counter
    c = Counter(full_pool[int(j)] for j in picks.ravel() if full_pool[int(j)] not in trained)
    fs, ts = ro_f["curves"][TOTAL_L]
    ss, st = ro_s["curves"][TOTAL_L]
    # ANSWERABILITY of what the open vocabulary makes askable. The 1,000-candidate pool is the top-500
    # items BY POPULARITY and the top-500 concepts BY ANSWER RATE, so everything outside it is, by
    # construction, less answerable -- and a burned turn costs the interview a question. This is the
    # measurement that says whether an opened vocabulary is being spent on unanswerable questions.
    ans_f = (ro_f["outs"] != OUTCOME["noanswer"])
    ans_s = (ro_s["outs"] != OUTCOME["noanswer"])
    oopm = oop.astype(bool)
    out = {"analysis": "policy_E1_scalability", "design_sheet":
           "docs/design/PAPER_B_POLICY_DESIGN.md D.4", "citable": True, "test_touched": False,
           "ckpt": os.path.basename(args.ckpt), "n_train_pool": len(pool),
           "n_inference_catalogue": len(full_pool),
           "out_of_pool_pick_rate": float(oop.mean()),
           "out_of_pool_by_position": [float(x) for x in oop.mean(0)],
           "users_with_any_out_of_pool_pick": float((oop.sum(1) > 0).mean()),
           "top_out_of_pool_questions": [{"channel": "item" if q[0] == 0 else "concept",
                                          "id": int(q[1]), "n_picks": int(n)}
                                         for q, n in c.most_common(25)],
           "n_distinct_out_of_pool_questions": len(c),
           "val_full_1000_pool": float(np.nanmean(ss)), "val_tail_1000_pool": float(np.nanmean(st)),
           "val_full_full_catalogue": float(np.nanmean(fs)),
           "val_tail_full_catalogue": float(np.nanmean(ts)),
           "paired_full_catalogue_minus_1000": paired_bootstrap(fs - ss),
           "answer_rate_1000_pool": float(ans_s.mean()),
           "answer_rate_full_catalogue": float(ans_f.mean()),
           "answer_rate_of_in_pool_picks": float(ans_f[~oopm].mean()),
           "answer_rate_of_out_of_pool_picks": float(ans_f[oopm].mean()),
           "channel_mix_full_catalogue": channel_mix(picks, full_pool),
           "note": ("The catalogue arm is a DIFFERENT action space from the tree's, so its NDCG is not "
                    "comparable to the tree's headline and is reported as a description of what the "
                    "policy does when the vocabulary is opened, not as a gain against any comparator.")}
    p = os.path.join(OUTDIR, "policy_scalability.json")
    with open(p + ".tmp", "w") as fh:
        json.dump(out, fh, indent=2, default=float)
    os.replace(p + ".tmp", p)
    log(f"[scale] out-of-pool picks {out['out_of_pool_pick_rate'] * 100:.2f}% of all questions; "
        f"{out['n_distinct_out_of_pool_questions']} distinct never-trained questions asked; "
        f"full catalogue val full {out['val_full_full_catalogue']:.5f} vs "
        f"{out['val_full_1000_pool']:.5f} on the training pool")
    log(f"[scale] wrote {p}")
    return out


def main():
    ap = argparse.ArgumentParser(description="Workstream E parametric policy. See "
                                             "docs/design/PAPER_B_POLICY_DESIGN.md")
    ap.add_argument("--selftest", action="store_true",
                    help="pure-tensor invariance checks on random weights. No data, no fit.")
    ap.add_argument("--arch", default="equivariant", choices=["equivariant", "pointer", "flat"])
    ap.add_argument("--state", default="both", choices=["z", "hist", "both"])
    ap.add_argument("--no_pos", action="store_true",
                    help="drop the position/remaining-budget feature. The tree is budget-unaware, so "
                         "THIS is the like-for-like variant (design sheet A.4) and both are reported.")
    ap.add_argument("--arm", default="main", choices=["main", "shuffle", "deep", "deepshuffle"],
                    help="deep = PHASE 3 (design sheet G.B): the labels one level BELOW the tree's "
                         "floor stops, MIXED with the original tree labels (the replay is the "
                         "anti-forgetting mechanism and is not optional). deepshuffle = its HARD "
                         "RULE 5 control, the 42 branch questions permuted across branches.")
    ap.add_argument("--deep_labels", default=DEEP_LABELS,
                    help="policy_deep_labels.json, from `policy_deep.py --mode labels`")
    ap.add_argument("--init_ckpt", default=None,
                    help="fine-tune from this checkpoint instead of from scratch (phase 3 starts "
                         "from the E1 anchor). Every structural field is asserted.")
    ap.add_argument("--tag", default="",
                    help="suffix for this run's output stem. HARD RULE 11e: two runs never share an "
                         "output path.")
    ap.add_argument("--overwrite", action="store_true",
                    help="permit replacing an existing artifact with this stem. Off by default.")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--snapshot", default=None)
    ap.add_argument("--concept_ckpt", default=None)
    ap.add_argument("--full_threads", action="store_true",
                    help="consumed by strategy_channel_suite AT IMPORT (it reads sys.argv)")
    ap.add_argument("--threads", type=int, default=None,
                    help="thread count, consumed by strategy_channel_suite AT IMPORT")
    ap.add_argument("--i_have_author_goahead", action="store_true",
                    help="required to train.")
    ap.add_argument("--scale", action="store_true",
                    help="STEP 4: score the FULL catalogue at inference from a trained checkpoint")
    ap.add_argument("--ckpt", default=os.path.join(OUTDIR, "policy_main_equivariant_both.pt"))
    args = ap.parse_args()
    if args.selftest:
        return selftest()
    if args.scale:
        from strategy_channel_suite import CERT_TOWER as _T, CERT_CONC as _C
        args.snapshot = args.snapshot or _T
        args.concept_ckpt = args.concept_ckpt or _C
        return scalability(args)
    from strategy_channel_suite import CERT_TOWER, CERT_CONC
    args.snapshot = args.snapshot or CERT_TOWER
    args.concept_ckpt = args.concept_ckpt or CERT_CONC
    log("#" * 110)
    log("# WORKSTREAM E1 -- THE PARAMETRIC POLICY. FIT on the 140,768 TRAIN users' states + the")
    log("# FROZEN tree's labels; SELECT on real interview NDCG over the 10,000 VALIDATION users.")
    log("# FIREWALL: the 10,000 canonical TEST users are NOT built, NOT loaded, NOT touched.")
    log("#" * 110)
    return train(args)


if __name__ == "__main__":
    main()
