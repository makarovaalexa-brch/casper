r"""PAPER B / PART F -- THE THREE-WAY OUTCOME HEAD:  P(like | dislike | no-answer  |  state, question).

Design sheet: `docs/design/PAPER_B_POLICY_DESIGN.md` **PART F is the governing section** (the two-head
design, author-proposed, Fable-reviewed). Parts D and E give context; where F contradicts them, F wins.

> **AUTHOR SAID (2026-08-13):** *"nah, plz train on all and see if it even works. after than we can
> build curves and stuff. how long to train one version?"*

So THIS run trains on ALL questions and ALL train users. F.5's held-out-QUESTION generalisation test
and the 10/25/50/100% sample-efficiency curve are **DEFERRED** by that instruction and are not built
here. What IS built is the cheap version of the author's real question: does a state-conditioned head
beat a state-free per-question outcome prior?

WHY THREE OUTCOMES AND NEVER TWO
--------------------------------
F.1: the root's three children sit at 0.374 / 0.320 / 0.087 and ARM N measured a "no" worth +0.023 at
k=2. A binary answered/not encoding throws away the like/dislike distinction exactly where it pays,
and it would not match the tree's own ternary branching.

INPUTS -- AND THE GUARD THAT IS LOAD-BEARING
--------------------------------------------
Interview-visible state ONLY: the ternary answer history so far (three-way one-hot per asked question,
the distinct "asked, no answer" outcome, the per-outcome counts and the unanswered-turn count), the
tower latent `z_t`, and depth / remaining budget -- plus the scored question's FROZEN descriptor
(decoder row for an item, cv3 embedding for a concept, lifted into a shared space by the per-channel
lifts of D.3(a)).

**NEVER any profile feature, and never the user's rating of the question being scored.**
Answerability is a structural rule partly computable from the profile, so a head that sees
profile-like input learns an arithmetic identity and the result is hollow. `guard_no_profile` asserts
it structurally: at position 0 the state must be BIT-IDENTICAL across every user, because an empty
interview has revealed nothing. A profile leak of any kind breaks that equality.

LABELS
------
The outcome of question q for user u is DETERMINISTIC given u's fold-in half under the project's
structural answer rule (`ndcg_tree.branch_of`: values = real ratings for items, SEL behavioural signed
value for concepts; answerability = the transparent structural rule). No model call, no sampling. They
are harvested by `policy_traj.py --mode gen` and gated there against `branch_of` itself (G-O1).

The consequence, stated because it defines the task: **the outcome is a function of (u, q) and NOT of
the state**, so the head's job is genuinely predictive -- infer the reaction to an unseen question from
the answers so far. A head that ignores the state degenerates to a per-question prior, which is exactly
the popularity baseline below. That baseline is the whole point of this experiment.

COVERAGE -- NO SAMPLING (HARD RULE 1)
-------------------------------------
Every state scores ALL 19,390 catalogue candidates (18,359 items + 1,031 concepts) in one matmul, and
the loss is taken over all of them. Labels are stored sparsely (answered items as a CSR, concepts
dense) rather than as a 140,768 x 19,390 dense block, so the full coverage costs 0.25 GB, not 2.7 GB.
Nothing is subsampled: not users, not positions, not candidates.

FIREWALL
--------
The 10,000 TEST users are never built, never loaded, never touched; `--split test` does not exist in
this file's parser. **No NDCG is measured anywhere here** -- that is what makes fitting on the 140,768
TRAIN users legitimate (the tower memorised their ratings, so NDCG there would be meaningless, but an
observed answer outcome is not a model output). Every artifact stamps `test_touched=false`.
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
import torch.nn.functional as F

from policy_net import StateEncoder, build_phi, N_OUTCOMES, OUTCOME, SEED, log

OUTDIR = os.path.join(_ROOT, "experiments", "battery")
TRAJ_DIR = os.path.join(OUTDIR, "policy_traj")
TOTAL_L = 16
N_TRAIN = 140768
VAL_M0_FULL = 0.15686538
NB = 2048                          # probability bins for the streaming AUC / ECE accumulators
OUT_NAMES = ["like", "dislike", "noanswer"]


# ============================================================================ the head
class OutcomeHead(nn.Module):
    """P(outcome | state, question), three ways, over the whole catalogue in one matmul.

        logits[b, o, p] = < u_o(h_b), v(phi_p) > + c[p, o]

    The query net emits N_OUTCOMES query vectors from the state; the key net lifts the frozen
    descriptor into the same space; `c` is a per-candidate per-outcome bias read off the descriptor
    (this is what lets a per-question PRIOR be expressed at all -- and the popularity baseline is
    exactly the model that has ONLY that term, which is why the term is kept and the comparison is
    honest rather than rigged).

    Equivariant by construction: every candidate goes through the same `v` / `c`, nothing couples two
    candidates, so the head runs unchanged on a vocabulary it never saw. That is the same property the
    E1 policy head has, and it is what makes F.5's deferred held-out-question test possible later.
    """

    def __init__(self, d_state, d_phi, d_k=64, d_h=256):
        super().__init__()
        self.d_k = d_k
        self.q = nn.Sequential(nn.Linear(d_state, d_h), nn.ReLU(),
                               nn.Linear(d_h, d_k * N_OUTCOMES))
        self.k = nn.Sequential(nn.Linear(d_phi, d_h), nn.ReLU(), nn.Linear(d_h, d_k))
        self.c = nn.Linear(d_phi, N_OUTCOMES)

    def keys(self, phi):
        """The candidate side, computed ONCE per optimiser step and reused across the batch."""
        return self.k(phi), self.c(phi).T.contiguous()      # (P,d_k), (3,P)

    def forward(self, h, phi=None, kv=None):
        """h (B, d_state) -> logits (3, B, P).

        The OUTCOME-MAJOR layout is not cosmetic. (B,P,3) forces a permute of a 60-million-element
        tensor every step and makes the cross-entropy materialise another one; (3,B,P) lets the loss
        be `logsumexp - gather` over three contiguous (B,P) planes, which is the difference between
        ~90 and ~35 minutes per epoch on this box, measured.
        """
        V, C = self.keys(phi) if kv is None else kv
        B = h.shape[0]
        Q = self.q(h).view(B, N_OUTCOMES, self.d_k).transpose(0, 1)      # (3,B,d_k)
        return torch.baddbmm(C[:, None, :], Q, V.T.expand(N_OUTCOMES, -1, -1))

    def n_params(self):
        return sum(p.numel() for p in self.parameters())


class OutcomeModel(nn.Module):
    def __init__(self, d_z, d_phi, state="both", use_pos=True, d_k=64, d_h=256):
        super().__init__()
        self.enc = StateEncoder(d_z, d_phi, d_h=d_h, mode=state, use_pos=use_pos)
        self.head = OutcomeHead(self.enc.d_out, d_phi, d_k=d_k, d_h=d_h)

    def forward(self, z, hist_phi, hist_out, hist_mask, pos, phi=None, kv=None):
        return self.head(self.enc(z, hist_phi, hist_out, hist_mask, pos), phi=phi, kv=kv)

    def n_params(self):
        return sum(p.numel() for p in self.parameters())


# ============================================================================ the data
def load_bank():
    """The harvest: states from `policy_traj.py --mode gen`, outcome labels from the same pass."""
    man = json.load(open(os.path.join(TRAJ_DIR, "manifest.json")))
    assert man["sig"].get("outcome_labels"), (
        "this trajectory bank predates the outcome-label harvest -- regenerate with "
        "`policy_traj.py --mode gen` (HARD RULE 11: a contract change is a refusal, not a merge)")
    n_done = sum(v["n_users"] for v in man["done"].values())
    assert n_done == man["sig"]["n_train"], (
        f"the bank is INCOMPLETE: {n_done:,} of {man['sig']['n_train']:,} train users. Refusing to "
        f"train on a partial cohort (HARD RULE 1).")
    keys = sorted(man["done"], key=lambda k: int(k.split("_")[0]))
    uid, lab, out = [], [], []
    iptr = [np.zeros(1, np.int64)]
    iid, iout, conc = [], [], []
    base = 0
    for key in keys:
        z = np.load(os.path.join(TRAJ_DIR, f"meta_{key}.npz"))
        a = np.load(os.path.join(TRAJ_DIR, f"ans_{key}.npz"))
        assert np.array_equal(z["uid"], a["uid"]), f"chunk {key}: state and label rows disagree"
        uid.append(z["uid"]); lab.append(z["label_idx"]); out.append(z["outcome"])
        iptr.append(a["item_ptr"][1:] + base)
        base += int(a["item_ptr"][-1])
        iid.append(a["item_id"]); iout.append(a["item_out"]); conc.append(a["conc_out"])
    uid = np.concatenate(uid)
    o = np.argsort(uid)
    assert np.array_equal(uid[o], np.arange(man["sig"]["n_train"])), \
        "the train uids are not exactly 0..140,767 -- a user is missing or duplicated"
    assert np.array_equal(uid, np.arange(man["sig"]["n_train"])), \
        "the chunks are not in uid order; the CSR concatenation assumes they are"
    Z = np.load(os.path.join(TRAJ_DIR, "z_train.npy"), mmap_mode="r")
    assert Z.shape == (man["sig"]["n_train"], TOTAL_L, man["sig"]["d"])
    D = {"uid": uid, "label_idx": np.concatenate(lab), "outcome": np.concatenate(out), "Z": Z,
         "item_ptr": np.concatenate(iptr), "item_id": np.concatenate(iid),
         "item_out": np.concatenate(iout), "conc_out": np.concatenate(conc),
         "pool": [(int(a), int(b)) for a, b in man["pool"]], "sig": man["sig"]}
    assert len(D["item_ptr"]) == len(uid) + 1
    return D


def guard_no_profile(D, n=20000):
    """★ THE GUARD, structural rather than declarative.

    At position 0 the interview has revealed NOTHING, so the state must be identical for every user.
    If any profile feature had leaked into the state -- the user's ratings, their popularity profile,
    their latent -- position-0 states would differ across users and this equality would break. It is
    a one-line check that no amount of careful reading of the feature list can replace.
    """
    z0 = np.asarray(D["Z"][:n, 0], np.float32)
    d = np.abs(z0 - z0[0:1]).max()
    assert d == 0.0, (
        f"PROFILE LEAK: the position-0 state is not identical across users (max |diff| = {d:.3e}). "
        f"The empty interview has revealed nothing, so anything that distinguishes two users at "
        f"position 0 is a profile feature. STOP.")
    log(f"[guard] no-profile guard PASSES: the position-0 state is bit-identical across "
        f"{n:,} users (max |diff| = 0.0) -- the state carries interview evidence only")
    return True


def label_block(D, u, P, ni):
    """The dense (B, P) int8 outcome block for a batch of users, built from the sparse bank.

    Everything defaults to NOANSWER, which is the rule ("absent from the fold-in lookup"), then the
    answered item cells are scattered in and the concept block is gathered. No candidate is dropped:
    this IS all 19,390 of them.
    """
    B = len(u)
    y = np.full((B, P), OUTCOME["noanswer"], np.int8)
    ptr, iid, iout = D["item_ptr"], D["item_id"], D["item_out"]
    lo = ptr[u]; hi = ptr[u + 1]
    cnt = (hi - lo).astype(np.int64)
    if cnt.sum():
        rep = np.repeat(np.arange(B), cnt)
        idx = np.repeat(lo, cnt) + (np.arange(cnt.sum()) - np.repeat(np.cumsum(cnt) - cnt, cnt))
        y[rep, iid[idx]] = iout[idx]
    y[:, ni:] = D["conc_out"][u]
    return y


def state_batch(D, phi_t, u, t):
    """z + the EXPLICIT ternary history + position, exactly the E1 state encoder's contract."""
    B = len(u)
    z = torch.from_numpy(np.ascontiguousarray(D["Z"][u, t]).astype(np.float32))
    T = TOTAL_L - 1
    hq = np.zeros((B, T), np.int64); ho = np.zeros((B, T), np.int64)
    hm = np.zeros((B, T), np.float32)
    for j in range(T):
        m = t > j
        if not m.any():
            break
        hq[m, j] = D["label_idx"][u[m], j]
        ho[m, j] = D["outcome"][u[m], j]
        hm[m, j] = 1.0
    pos = np.stack([t, TOTAL_L - t], 1).astype(np.float32)
    return (z, torch.from_numpy(phi_t[hq]), torch.from_numpy(ho), torch.from_numpy(hm),
            torch.from_numpy(pos), hq, hm)


def asked_mask(hq, hm, B, P):
    """Already-asked candidates are excluded from the loss and from every metric. Their outcome is
    literally in the input, so scoring them would be measuring a copy, not a prediction. It is at
    most 15 of 19,390 cells (0.08%) but it is excluded on both sides so nothing is inflated."""
    m = np.ones((B, P), bool)
    sel = hm.ravel() > 0
    if sel.any():
        m[np.repeat(np.arange(B), hq.shape[1])[sel], hq.ravel()[sel]] = False
    return torch.from_numpy(m)


def masked_ce(s, yt, keep):
    """Cross-entropy over the (3,B,P) logits, averaged over the kept cells.

    Written out rather than called through `F.cross_entropy` because the library form needs a
    (N,3)-shaped copy of a 60-million-element tensor and a boolean gather on top of it. This form
    touches three contiguous planes and nothing else.
    """
    lse = torch.logsumexp(s, 0)                                   # (B,P)
    picked = s.gather(0, yt[None]).squeeze(0)                     # (B,P)
    l = (lse - picked) * keep
    return l.sum() / keep.sum()


# ============================================================================ the pool index map
def pool_to_catalogue(pool, ni):
    """The 1,000-candidate pool's index inside the FULL catalogue ordering
    [items 0..ni-1] ++ [concepts 0..C-1]. The harvest's history is stored as pool indices."""
    return np.asarray([(i if ch == 0 else ni + i) for ch, i in pool], np.int64)


# ============================================================================ baselines
def base_rates(D, fit_users, P, ni):
    """The two state-free predictors, computed on the FIT users only.

      global   P(o), one distribution for the whole catalogue -- the floor.
      pop      P(o | q), the per-question outcome prior. ★ THIS IS THE BASELINE THAT MATTERS.
               It ignores the interview entirely. If the head does not beat it, the head does not
               work, and that is what gets reported (HARD RULE 6) -- no tuning until it passes.

    Both are computed EXACTLY over every (fit user, candidate) pair, by counting, with a Laplace-1
    smoothing so a never-answered candidate has a defined distribution rather than a zero.
    """
    n = len(fit_users)
    cnt = np.zeros((P, N_OUTCOMES), np.float64)
    cnt[:ni, OUTCOME["noanswer"]] = n                      # items start from "nobody rated anything"
    ptr, iid, iout = D["item_ptr"], D["item_id"], D["item_out"]
    for st in range(0, n, 20000):
        u = fit_users[st:st + 20000]
        lo = ptr[u]; hi = ptr[u + 1]
        c = (hi - lo).astype(np.int64)
        if c.sum():
            idx = np.repeat(lo, c) + (np.arange(c.sum()) - np.repeat(np.cumsum(c) - c, c))
            np.add.at(cnt, (iid[idx].astype(np.int64), iout[idx].astype(np.int64)), 1.0)
            np.add.at(cnt, (iid[idx].astype(np.int64), OUTCOME["noanswer"]), -1.0)
        cc = D["conc_out"][u]
        for o in range(N_OUTCOMES):
            cnt[ni:, o] += (cc == o).sum(0).astype(np.float64)
    assert abs(cnt.sum() - n * P) < 1e-6, "the outcome counts do not partition the (user, candidate) grid"
    pop = (cnt + 1.0) / (cnt.sum(1, keepdims=True) + N_OUTCOMES)
    g = cnt.sum(0); glob = g / g.sum()
    return {"pop": pop.astype(np.float32), "global": glob.astype(np.float32),
            "counts_total": g, "n_fit_users": int(n)}


# ============================================================================ streaming metrics
class Acc:
    """Per-position, per-class streaming accumulators. Everything the report needs comes off these:
    one-vs-rest AUC (rank formula over probability bins), ECE (same bins), accuracy, mean CE.

    Binning is the HARD RULE 1 non-lossy alternative to sorting 9.3 billion scores: 2,048 bins over
    [0,1] resolve AUC to ~1e-4 and ECE is a binned quantity by definition. No sample is dropped.
    """

    def __init__(self):
        self.pos = np.zeros((TOTAL_L, N_OUTCOMES, NB), np.float64)
        self.neg = np.zeros((TOTAL_L, N_OUTCOMES, NB), np.float64)
        self.sp = np.zeros((TOTAL_L, N_OUTCOMES, NB), np.float64)
        self.correct = np.zeros(TOTAL_L, np.float64)
        self.n = np.zeros(TOTAL_L, np.float64)
        self.ce = np.zeros(TOTAL_L, np.float64)
        self.conf = np.zeros((N_OUTCOMES, N_OUTCOMES), np.float64)

    def add(self, t, probs, yt, keep):
        """probs (3,B,P) float32 | yt (B,P) long | keep (B,P) bool. `t` is one position."""
        kf = keep.ravel()
        yy = yt.ravel()[kf]
        M = int(yy.numel())
        self.n[t] += M
        pred = probs.argmax(0).ravel()[kf]
        self.correct[t] += float((pred == yy).sum())
        py = probs.reshape(N_OUTCOMES, -1)[yy, torch.nonzero(kf, as_tuple=True)[0]]
        self.ce[t] += float(-torch.log(py.clamp_min(1e-12)).sum())
        self.conf += np.bincount((yy * N_OUTCOMES + pred).numpy(),
                                 minlength=N_OUTCOMES ** 2).reshape(N_OUTCOMES, N_OUTCOMES)
        for o in range(N_OUTCOMES):
            p = probs[o].ravel()[kf]
            b = (p * NB).long().clamp_(0, NB - 1).numpy()
            ispos = (yy == o).numpy()
            self.pos[t, o] += np.bincount(b[ispos], minlength=NB)
            self.neg[t, o] += np.bincount(b[~ispos], minlength=NB)
            self.sp[t, o] += np.bincount(b, weights=p.double().numpy(), minlength=NB)

    def report(self):
        P_ = self.pos.sum(0); N_ = self.neg.sum(0); S_ = self.sp.sum(0)
        auc, ece = [], []
        for o in range(N_OUTCOMES):
            np_, nn_ = P_[o], N_[o]
            cum = np.concatenate([[0.0], np.cumsum(nn_)[:-1]])
            a = float((np_ * (cum + nn_ / 2.0)).sum() / max(np_.sum() * nn_.sum(), 1e-9))
            auc.append(a)
            tot = np_ + nn_
            m = tot > 0
            ece.append(float((tot[m] / tot.sum() *
                              np.abs(S_[o][m] / tot[m] - np_[m] / tot[m])).sum()))
        pa, pn = self.pos.sum(1), self.neg.sum(1)           # per position, all classes pooled
        return {
            "n_scored": float(self.n.sum()),
            "accuracy": float(self.correct.sum() / max(self.n.sum(), 1)),
            "mean_ce": float(self.ce.sum() / max(self.n.sum(), 1)),
            "auc_ovr": {OUT_NAMES[o]: auc[o] for o in range(N_OUTCOMES)},
            "auc_macro": float(np.mean(auc)),
            "ece": {OUT_NAMES[o]: ece[o] for o in range(N_OUTCOMES)},
            "ece_macro": float(np.mean(ece)),
            "accuracy_by_position": [float(c / max(n, 1)) for c, n in zip(self.correct, self.n)],
            "ce_by_position": [float(c / max(n, 1)) for c, n in zip(self.ce, self.n)],
            "auc_by_position": self._auc_by_position(),
            "confusion_true_by_pred": self.conf.tolist(),
        }

    def _auc_by_position(self):
        out = []
        for t in range(TOTAL_L):
            a = []
            for o in range(N_OUTCOMES):
                np_, nn_ = self.pos[t, o], self.neg[t, o]
                cum = np.concatenate([[0.0], np.cumsum(nn_)[:-1]])
                a.append(float((np_ * (cum + nn_ / 2.0)).sum() / max(np_.sum() * nn_.sum(), 1e-9)))
            out.append(float(np.mean(a)))
        return out


# ============================================================================ evaluation
def evaluate(model, D, users, phi_t, P, ni, pop=None, glob=None, batch=384, full=True, log_every=40):
    """One pass over (held-out user x 16 positions x 19,390 candidates). When `pop`/`glob` are given
    they are scored on EXACTLY the same cells through the same accumulator, so the comparison is
    cell-for-cell rather than table-against-table."""
    accs = {"head": Acc()}
    if pop is not None:
        accs["pop"] = Acc(); accs["global"] = Acc()
    pop_t = torch.from_numpy(np.ascontiguousarray(pop.T)) if pop is not None else None   # (3,P)
    glob_t = torch.from_numpy(np.ascontiguousarray(glob)) if glob is not None else None
    phi_g = torch.from_numpy(phi_t)
    model.eval()
    t0 = time.time()
    with torch.no_grad():
        kv = model.head.keys(phi_g)
        nb = 0
        for st in range(0, len(users), batch):
            u = users[st:st + batch]
            yt = torch.from_numpy(label_block(D, u, P, ni)).long()
            for t in range(TOTAL_L):
                tt = np.full(len(u), t, np.int64)
                z, hp, ho, hm, ps, hq, hmask = state_batch(D, phi_t, u, tt)
                keep = asked_mask(hq, hmask, len(u), P)
                pr = torch.softmax(model(z, hp, ho, hm, ps, kv=kv), 0)
                accs["head"].add(t, pr, yt, keep)
                if pop_t is not None and full:
                    accs["pop"].add(t, pop_t[:, None, :].expand(-1, len(u), -1), yt, keep)
                    accs["global"].add(t, glob_t[:, None, None].expand(-1, len(u), P), yt, keep)
            nb += 1
            if nb % log_every == 0:
                log(f"  [eval] {st + len(u):,}/{len(users):,} users "
                    f"({(time.time() - t0) / 60:.1f}m)")
    return {k: v.report() for k, v in accs.items()}


# ============================================================================ training
def train(args):
    assert args.i_have_author_goahead, "REFUSING TO TRAIN without the author's go-ahead."
    torch.manual_seed(SEED); np.random.seed(SEED)
    rng = np.random.default_rng(SEED)
    if args.threads:
        torch.set_num_threads(int(args.threads))
    t00 = time.time()

    D = load_bank()
    guard_no_profile(D)
    N = len(D["uid"])

    # ---- the catalogue, its descriptors, and the canonical snap ----------------------------------
    from strategy_channel_suite import (build_greedy_env, contract_selftest, genv_mean,
                                        CERT_TOWER, CERT_CONC)
    from ndcg_tree import leak_assert
    E = build_greedy_env(args.snapshot, "validation", args.concept_ckpt, "outcomehead/desc")
    contract_selftest(E, E["rows"]); leak_assert(E, E["rows"], "outcomehead/desc")
    f0, t0_ = genv_mean(E, [], E["rows"])
    assert round(f0, 8) == VAL_M0_FULL, f"canonical snap FAILED: {f0!r}"
    log(f"[snap] validation m=0 intercept full={f0!r} tail={t0_!r} -- canonical, exact. "
        f"(The env is built ONLY to read the frozen descriptors and to snap the ruler; NO NDCG is "
        f"measured in this file.)")
    ni, C = E["ctx"].ni, E["C"]
    catalogue = [(0, i) for i in range(ni)] + [(1, c) for c in range(C)]
    P = len(catalogue)
    phi_t, norm = build_phi(E, catalogue)
    d_z = E["ctx"].d
    del E
    log(f"[catalogue] {P:,} candidates ({ni:,} items + {C:,} concepts), "
        f"{phi_t.shape[1]}-dim frozen descriptors")

    # the history is stored as POOL indices; the head scores CATALOGUE indices. Map once.
    p2c = pool_to_catalogue(D["pool"], ni)
    D["label_idx"] = p2c[D["label_idx"]].astype(np.int64)
    assert D["label_idx"].max() < P

    # ---- the split: held-out USERS, questions NOT held out (the author's instruction) ------------
    perm = rng.permutation(N)
    heldout = np.sort(perm[:args.n_heldout]); fit = np.sort(perm[args.n_heldout:])
    assert len(np.intersect1d(fit, heldout)) == 0
    log(f"[split] fit {len(fit):,} train users / held-out {len(heldout):,} train users. "
        f"ALL {P:,} questions are in training -- the held-out-QUESTION test of F.5 is DEFERRED by "
        f"the author's instruction ('train on all and see if it even works').")

    # ---- the baselines, computed before anything is fitted ---------------------------------------
    tb = time.time()
    B = base_rates(D, fit, P, ni)
    log(f"[baseline] global outcome base rates like/dislike/noanswer = "
        f"{B['global'][0]:.5f} / {B['global'][1]:.5f} / {B['global'][2]:.5f} "
        f"({(time.time() - tb) / 60:.1f}m)")

    # ---- the model -------------------------------------------------------------------------------
    model = OutcomeModel(d_z, phi_t.shape[1], state=args.state, use_pos=not args.no_pos,
                         d_k=args.d_k, d_h=args.d_h)
    log(f"[model] state={args.state} pos={not args.no_pos} d_k={args.d_k} "
        f"params={model.n_params():,}")
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    phi_g = torch.from_numpy(phi_t)

    # ---- the shuffle control (HARD RULE 5) -------------------------------------------------------
    lab_user = np.arange(N)
    if args.arm == "shuffle":
        lab_user = rng.permutation(N)
        log("[arm] SHUFFLE CONTROL: the outcome labels of user u are taken from a DIFFERENT user. "
            "The states, the questions and every per-question marginal are untouched, so the "
            "control's ceiling is exactly the popularity baseline. It must not beat it.")

    n_ex = len(fit) * TOTAL_L
    steps = int(np.ceil(n_ex / args.batch))
    hist = []
    best = {"ce": 1e9}
    tag = f"{args.arm}_{args.state}{'_nopos' if args.no_pos else ''}"
    ckpt = os.path.join(OUTDIR, f"outcome_head_{tag}.pt")
    for ep in range(1, args.epochs + 1):
        model.train()
        pe = rng.permutation(n_ex)
        tot = 0.0; seen = 0; hit = 0
        te = time.time()
        for si in range(steps):
            e = pe[si * args.batch:(si + 1) * args.batch]
            if not len(e):
                continue
            u = fit[e // TOTAL_L]; t = (e % TOTAL_L).astype(np.int64)
            z, hp, ho, hm, ps, hq, hmask = state_batch(D, phi_t, u, t)
            yt = torch.from_numpy(label_block(D, lab_user[u], P, ni)).long()
            keep = asked_mask(hq, hmask, len(u), P)
            s = model(z, hp, ho, hm, ps, phi=phi_g)
            loss = masked_ce(s, yt, keep)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss) * len(e); seen += len(e)
            with torch.no_grad():
                hit += float((((s.argmax(0) == yt) & keep).sum() / keep.sum())) * len(e)
            if (si + 1) % args.log_every == 0:
                el = (time.time() - te) / 60
                log(f"[ep{ep} {si + 1}/{steps}] loss {tot / seen:.5f} acc {hit / seen:.5f} "
                    f"({el:.1f}m, projected {el / (si + 1) * steps:.1f}m/epoch)")
        tr_min = (time.time() - te) / 60
        ev = evaluate(model, D, heldout, phi_t, P, ni, batch=args.eval_batch, full=False)["head"]
        rec = {"epoch": ep, "train_loss": tot / seen, "train_acc": hit / seen,
               "heldout_ce": ev["mean_ce"], "heldout_acc": ev["accuracy"],
               "train_minutes": round(tr_min, 2),
               "epoch_minutes": round((time.time() - te) / 60, 2)}
        hist.append(rec)
        log(f"[ep{ep}] train loss {rec['train_loss']:.5f} | HELD-OUT ce {ev['mean_ce']:.5f} "
            f"acc {ev['accuracy']:.5f} | {rec['train_minutes']:.1f}m train, "
            f"{rec['epoch_minutes']:.1f}m total")
        if ev["mean_ce"] < best["ce"]:
            best = {"ce": ev["mean_ce"], "epoch": ep}
            torch.save({"model": model.state_dict(), "state": args.state,
                        "use_pos": not args.no_pos, "d_k": args.d_k, "d_h": args.d_h,
                        "d_phi": int(phi_t.shape[1]), "d_z": d_z, "epoch": ep,
                        "norm_mu": norm[0], "norm_sd": norm[1], "arm": args.arm,
                        "heldout_ce": ev["mean_ce"], "seed": SEED}, ckpt)
            log(f"[ckpt] epoch {ep} is best so far -> {os.path.basename(ckpt)}")
        json.dump({"analysis": "outcome_head_history", "arm": args.arm, "history": hist,
                   "test_touched": False},
                  open(os.path.join(OUTDIR, f"outcome_head_{tag}_history.json"), "w"),
                  indent=1, default=float)

    # ---- the full metric suite, on the SELECTED checkpoint, head vs both baselines ----------------
    blob = torch.load(ckpt, map_location="cpu")
    model.load_state_dict(blob["model"])
    log(f"[final] scoring the full metric suite on the selected checkpoint (epoch {blob['epoch']}): "
        f"head vs the per-question popularity prior vs the global base rate, cell for cell")
    res = evaluate(model, D, heldout, phi_t, P, ni, pop=B["pop"], glob=B["global"],
                   batch=args.eval_batch, full=True)

    beats = (res["head"]["auc_macro"] > res["pop"]["auc_macro"] and
             res["head"]["mean_ce"] < res["pop"]["mean_ce"])
    out = {"analysis": "outcome_head_partF", "design_sheet":
           "docs/design/PAPER_B_POLICY_DESIGN.md PART F", "citable": True,
           "test_touched": False, "ndcg_measured": False,
           "author_instruction": "train on ALL questions and ALL train users; held-out-question "
                                 "test and sample-efficiency curve DEFERRED",
           "arm": args.arm, "state": args.state, "use_pos": not args.no_pos,
           "n_catalogue_candidates": P, "n_items": ni, "n_concepts": C,
           "n_fit_users": len(fit), "n_heldout_users": len(heldout),
           "questions_held_out": 0, "params": model.n_params(),
           "best_epoch": blob["epoch"], "history": hist,
           "global_base_rates": {OUT_NAMES[o]: float(B["global"][o]) for o in range(N_OUTCOMES)},
           "results": res,
           "beats_popularity_baseline": bool(beats),
           "cost_minutes_total": round((time.time() - t00) / 60, 2)}
    p = os.path.join(OUTDIR, f"outcome_head_{tag}.json")
    with open(p + ".tmp", "w") as fh:
        json.dump(out, fh, indent=2, default=float)
    os.replace(p + ".tmp", p)
    np.savez_compressed(os.path.join(OUTDIR, f"outcome_head_{tag}_pop.npz"), pop=B["pop"],
                        glob=B["global"], counts=B["counts_total"])
    log("=" * 100)
    for k in ("head", "pop", "global"):
        r = res[k]
        log(f"[{k:6s}] acc {r['accuracy']:.5f} | CE {r['mean_ce']:.5f} | AUC macro "
            f"{r['auc_macro']:.5f} (like {r['auc_ovr']['like']:.4f} / dislike "
            f"{r['auc_ovr']['dislike']:.4f} / noanswer {r['auc_ovr']['noanswer']:.4f}) | "
            f"ECE macro {r['ece_macro']:.5f}")
    log(f"[VERDICT] beats the per-question popularity baseline: {beats}")
    log(f"[cost] {out['cost_minutes_total']:.1f} minutes end to end")
    log(f"[out] {p}")
    return out


# ============================================================================ self-test (no data)
def selftest():
    """Pure-tensor / pure-numpy checks of the three things that could silently corrupt a number:
    the sparse->dense label reconstruction, the hand-written masked cross-entropy, and the binned
    AUC / ECE accumulator. No harvest, no fit, no environment."""
    rng = np.random.default_rng(0)
    n, ni, C = 40, 30, 9
    P = ni + C
    dense = np.full((n, P), OUTCOME["noanswer"], np.int8)
    ptr = [0]; ids = []; outs = []
    for u in range(n):
        k = rng.choice(ni, size=rng.integers(0, 8), replace=False)
        k.sort()
        o = rng.integers(0, 2, size=len(k)).astype(np.int8)
        dense[u, k] = o
        ids.append(k.astype(np.int32)); outs.append(o); ptr.append(ptr[-1] + len(k))
    dense[:, ni:] = rng.integers(0, 3, size=(n, C)).astype(np.int8)
    D = {"item_ptr": np.asarray(ptr, np.int64),
         "item_id": np.concatenate(ids) if ids else np.zeros(0, np.int32),
         "item_out": np.concatenate(outs) if outs else np.zeros(0, np.int8),
         "conc_out": dense[:, ni:].copy()}
    u = rng.permutation(n)[:17]
    assert np.array_equal(label_block(D, u, P, ni), dense[u]), \
        "label_block does not reconstruct the dense outcome table from the sparse bank"

    # base_rates must partition the grid and match a brute-force count
    B_ = base_rates(D, np.arange(n), P, ni)
    want = np.stack([(dense == o).sum(0) for o in range(N_OUTCOMES)], 1).astype(np.float64)
    got = B_["pop"] * (n + N_OUTCOMES) - 1.0
    assert np.abs(got - want).max() < 1e-6, "the per-question outcome prior is not the exact count"

    # masked cross-entropy == the library form on the kept cells
    s = torch.randn(N_OUTCOMES, 6, P)
    y = torch.randint(0, N_OUTCOMES, (6, P))
    keep = (torch.rand(6, P) > 0.2).float()
    want_l = torch.nn.functional.cross_entropy(
        s.permute(1, 2, 0).reshape(-1, N_OUTCOMES)[keep.ravel() > 0], y.ravel()[keep.ravel() > 0])
    assert abs(float(masked_ce(s, y, keep)) - float(want_l)) < 1e-5, \
        "masked_ce disagrees with the library cross-entropy on the kept cells"

    # the binned AUC reproduces the exact rank AUC to the bin resolution
    a = Acc()
    pr = torch.softmax(torch.randn(N_OUTCOMES, 6, P), 0)
    a.add(0, pr, y, torch.ones(6, P, dtype=torch.bool))
    r = a.report()
    for o in range(N_OUTCOMES):
        sc = pr[o].ravel().numpy(); lb = (y.ravel().numpy() == o)
        rk = np.argsort(np.argsort(sc)) + 1.0
        exact = (rk[lb].sum() - lb.sum() * (lb.sum() + 1) / 2) / (lb.sum() * (~lb).sum())
        assert abs(exact - r["auc_ovr"][OUT_NAMES[o]]) < 2e-3, \
            f"binned AUC {r['auc_ovr'][OUT_NAMES[o]]:.5f} != exact rank AUC {exact:.5f}"
    log("[selftest] PASS: sparse->dense labels, exact per-question priors, masked cross-entropy "
        "against the library form, and binned AUC against the exact rank AUC")
    return True


# ============================================================================ bench
def bench(args):
    """MEASURE the per-step cost before committing to a config -- the author asked how long one
    version takes and a projection is not an answer."""
    if args.threads:
        torch.set_num_threads(int(args.threads))
    P, d_phi, d_z = 19390, 205, 200
    phi = torch.randn(P, d_phi)
    m = OutcomeModel(d_z, d_phi, d_k=args.d_k, d_h=args.d_h)
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    B = args.batch
    z = torch.randn(B, d_z); hp = torch.randn(B, 15, d_phi)
    ho = torch.randint(0, 3, (B, 15)); hm = (torch.rand(B, 15) > .5).float()
    ps = torch.randn(B, 2)
    y = torch.randint(0, 3, (B, P))
    keep = torch.ones(B, P)
    log(f"[bench] threads={torch.get_num_threads()} batch={B} d_k={args.d_k} P={P} "
        f"params={m.n_params():,}")
    for it in range(4):
        t0 = time.time()
        s = m(z, hp, ho, hm, ps, phi=phi)
        loss = masked_ce(s, y, keep)
        opt.zero_grad(); loss.backward(); opt.step()
        dt = time.time() - t0
        n_ex = 140768 * 16
        log(f"[bench] step {it}: {dt:.2f}s -> {dt * n_ex / B / 60:.1f} min/epoch "
            f"(full {n_ex:,} examples)")
    t0 = time.time()
    with torch.no_grad():
        kv = m.head.keys(phi)
        for _ in range(4):
            torch.softmax(m(z, hp, ho, hm, ps, kv=kv), -1)
    dt = (time.time() - t0) / 4
    log(f"[bench] forward-only {dt:.2f}s/batch -> {dt * 10000 * 16 / B / 60:.1f} min per "
        f"10,000-user x 16-position eval pass (before the metric accumulators)")
    return True


def main():
    ap = argparse.ArgumentParser(description="Part F three-way outcome head. "
                                             "See docs/design/PAPER_B_POLICY_DESIGN.md PART F")
    ap.add_argument("--bench", action="store_true", help="measure per-step cost; no data, no fit")
    ap.add_argument("--selftest", action="store_true", help="pure-tensor checks; no data, no fit")
    ap.add_argument("--arm", default="main", choices=["main", "shuffle"])
    ap.add_argument("--state", default="both", choices=["z", "hist", "both"])
    ap.add_argument("--no_pos", action="store_true")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--eval_batch", type=int, default=384)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--d_k", type=int, default=64)
    ap.add_argument("--d_h", type=int, default=256)
    ap.add_argument("--n_heldout", type=int, default=10000,
                    help="held-out TRAIN users. Questions are NOT held out this round (author).")
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--log_every", type=int, default=100)
    ap.add_argument("--snapshot", default=None)
    ap.add_argument("--concept_ckpt", default=None)
    ap.add_argument("--full_threads", action="store_true",
                    help="consumed by strategy_channel_suite AT IMPORT (it reads sys.argv)")
    ap.add_argument("--i_have_author_goahead", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        return selftest()
    if args.bench:
        return bench(args)
    from strategy_channel_suite import CERT_TOWER, CERT_CONC
    args.snapshot = args.snapshot or CERT_TOWER
    args.concept_ckpt = args.concept_ckpt or CERT_CONC
    log("#" * 110)
    log("# PART F -- THE THREE-WAY OUTCOME HEAD: P(like | dislike | no-answer | state, question)")
    log("# FIT on the 140,768 TRAIN users; ALL 19,390 catalogue candidates per state, no sampling.")
    log("# FIREWALL: the 10,000 TEST users are NOT built, NOT loaded, NOT touched.")
    log("# NO NDCG IS MEASURED ANYWHERE IN THIS FILE. Labels are the structural answer rule.")
    log("#" * 110)
    return train(args)


if __name__ == "__main__":
    main()
