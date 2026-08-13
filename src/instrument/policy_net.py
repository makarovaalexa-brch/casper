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
def build_phi(E, pool):
    """The FROZEN per-candidate descriptor. Nothing here is learned, so nothing can drift off the
    certified ruler.

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
    mu, sd = phi.mean(0, keepdims=True), phi.std(0, keepdims=True) + 1e-6
    return (phi - mu) / sd


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


# ============================================================================ training (NOT RUN)
def train(args):
    """E1 imitation. Cross-entropy over the 1,000-candidate pool against the tree's exact-NDCG argmax.

    ⚠ AUTHOR GATE: as of 2026-08-13 the standing instruction is "do not train yet, just document all
    ideas." This function refuses to run without `--i_have_author_goahead`, so the file can be
    committed and reviewed without any risk of a stray run (HARD RULE 10 wants the code committed
    BEFORE anything runs; it does not want a run).
    """
    assert args.i_have_author_goahead, (
        "REFUSING TO TRAIN. The standing author instruction (2026-08-13) is 'do not train yet, just "
        "document all ideas'. Pass --i_have_author_goahead only when that has changed.")
    assert os.path.exists(args.labels), f"label npz missing: {args.labels}"
    blob = json.load(open(args.labels.replace(".npz", ".json")))
    assert blob["split"] == "validation" and blob["test_touched"] is False, \
        "FIREWALL: the label set is not a validation-only artifact"
    torch.manual_seed(SEED); np.random.seed(SEED)
    Z = np.load(args.labels)
    ex, pool = Z["examples"], Z["pool"]
    rows = Z["rows"]

    # --- the dev split. See design sheet A.2.1: this is an IN-SAMPLE early-stopping signal, because
    # --- the tree was grown on these same users and a node's label used all of the node's users.
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(len(rows))
    dev_rows = set(int(rows[i]) for i in perm[:args.n_dev])
    is_dev = np.asarray([int(r) in dev_rows for r in ex[:, 0]])
    log(f"[split] {len(rows) - args.n_dev} train / {args.n_dev} dev users (seed {SEED}); "
        f"{int((~is_dev).sum())} train / {int(is_dev.sum())} dev examples. CONTAMINATION STATED: "
        f"the labels were computed on all 10,000 users, dev included -- see design sheet A.2.1.")

    if args.arm == "shuffle":                        # G-E1d, HARD RULE 5
        lab = ex[:, 2].copy()
        rng.shuffle(lab)
        ex = ex.copy(); ex[:, 2] = lab
        log("[arm] SHUFFLE CONTROL: labels permuted. This arm must NOT clear G-E1b.")

    raise SystemExit(
        "E1's training loop needs the per-example state tensors (z_t, answer history, position), which "
        "are produced by the scoring environment and are NOT in the label npz. That builder is the "
        "next commit; it is deliberately not stubbed here, because a stub that silently fabricates a "
        "state is exactly the class of defect this project has been burned by. Nothing above this "
        "line touches the test cohort, and nothing above this line fits anything.")


def main():
    ap = argparse.ArgumentParser(description="Workstream E parametric policy. See "
                                             "docs/design/PAPER_B_POLICY_DESIGN.md")
    ap.add_argument("--selftest", action="store_true",
                    help="pure-tensor invariance checks on random weights. No data, no fit.")
    ap.add_argument("--labels", default=os.path.join(OUTDIR, "policy_labels_e1.npz"))
    ap.add_argument("--arch", default="equivariant", choices=["equivariant", "pointer", "flat"])
    ap.add_argument("--state", default="both", choices=["z", "hist", "both"])
    ap.add_argument("--no_pos", action="store_true",
                    help="drop the position/remaining-budget feature. The tree is budget-unaware, so "
                         "THIS is the like-for-like variant (design sheet A.4) and both are reported.")
    ap.add_argument("--arm", default="main", choices=["main", "shuffle"])
    ap.add_argument("--n_dev", type=int, default=2000)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--i_have_author_goahead", action="store_true",
                    help="required to train. The standing instruction (2026-08-13) is DO NOT TRAIN.")
    args = ap.parse_args()
    if args.selftest:
        return selftest()
    return train(args)


if __name__ == "__main__":
    main()
