r"""interview_curriculum.py -- interview-shaped training examples: the curriculum half of the
interview-native retrain.

SPEC: docs/design/RETRAIN_DESIGN_SHEET.md sections 5 and 9. NO TRAINING HAPPENS HERE. This module only
builds examples and is unit-tested against the measured evaluation statistics before any run.

WHY IT EXISTS. The current generator (`make_input_target`) draws interview examples as k items sampled
UNIFORMLY AT RANDOM from the user's own RATED items. A real interview asks strategy-selected items, most
of which the user has never rated. So the model has never seen:
  * a question the user cannot answer (6.4 of 8 come back unanswered under HELF),
  * a popularity-skewed or otherwise strategy-ordered ask sequence,
  * any correlation between what is asked and what gets answered,
  * an empty evidence set at all.
Train and test are different distributions. This module makes them the same one.

THE STRATEGY FAMILY, not the six eval arms. Each example samples its own asker:

    ask_score(i) = a*zs(log pop_i) + b*zs(H_i) + c*zs(H0_i) + d*zs(HELF_i) + e*noise_i

FOUR informative legs, not three: Rashid's two entropies are SEPARATE axes. An earlier version collapsed
H and H0 into one leg chosen by a coin flip, which made the named strategies unrepresentable -- the unit
test caught it immediately (entropy0 simulated at 0.04 answers/8 against a measured 3.10, because the
coin was landing on pure entropy, whose true rate really is ~0). Popularity, pure-entropy, entropy0 and
HELF are now exact corners of the simplex, which G-HELDOUT also depends on.

NO STRATEGY IS HELD OUT (author ruling, 2026-07-30, overruling an earlier decision of mine). The
heuristics are public formulas computed from TRAINING data, so training on them is not leakage -- it is
what a deployed system would do, and withholding one only weakens us on the comparison that matters.
The real generalisation test lives in Paper B, whose LEARNED question-selection policies are unseen by
construction. So the family spans the full simplex, corners included.

ANSWERABILITY is the transparent structural rule: an asked item is answered iff the user has rated it.
No LLM-derived answerability (banned for new work, 2026-07-22 audit). Answered -> the real half-star
level. Unanswered -> the UNSEEN channel, which the exposure branch consumes.
"""
import numpy as np

NLEV = 10
BUDGETS = (1, 2, 4, 8, 16, 32)
# Regime mixture (design sheet section 5). The k=0 bucket IS the prior-anchoring mechanism.
# 2026-07-31: full-profile share raised 0.35 -> 0.45 after the epoch-2 G-FULL abort. The certified
# recipe trains 50% of examples in the dropout regime; starving it to 35% while ALSO over-truncating
# them (drop_max 0.8 vs 0.5) cost 0.014-0.018 full-profile NDCG. The interview share stays substantial.
P_FULL, P_INTERVIEW, P_K0, P_K1 = 0.45, 0.45, 0.05, 0.05


def set_mixture(p_full, p_interview, p_k0, p_k1):
    """Override the regime mixture at runtime (the trainer's --mix flag writes through this).

    AUTHOR DIRECTIVE 2026-08-01: "the curriculum should be interview heavy, with realistic interviews.
    interviews are a priority, trying to not lose more than 1-2 points on full." So the mixture spends
    on interviews, and the full-profile share exists only to hold that budget. Reference: at 45/45 the
    run lost 0.0066 full profile (0.3482 -> 0.3416), well inside a 1-2 point allowance, so there is room
    to shift.

    NOTE this is the ONLY sanctioned way to change the mixture -- an earlier run shifted it to 30/60
    while ALSO dropping warm_lr_scale to 0.01, which nearly froze the taste path; interview learning
    went flat and the mixture change could not be evaluated on its own. Change the mixture OR the
    learning rate, not both."""
    global P_FULL, P_INTERVIEW, P_K0, P_K1
    tot = p_full + p_interview + p_k0 + p_k1
    assert abs(tot - 1.0) < 1e-9, f"mixture must sum to 1, got {tot}"
    P_FULL, P_INTERVIEW, P_K0, P_K1 = p_full, p_interview, p_k0, p_k1
def zs(x):
    x = np.asarray(x, np.float64)
    return (x - x.mean()) / max(x.std(), 1e-9)


class StrategyFamily:
    """Samples an ask-order per example from a continuous family spanning the literature's selectors."""

    def __init__(self, cnt, H, H0, helf, exclude_heldout=False):
        self.pop = zs(np.log1p(cnt))
        self.H = zs(H)
        self.H0 = zs(H0)
        self.helf = zs(helf)
        self.ni = len(self.pop)
        self.exclude_heldout = exclude_heldout

    P_PURE_NOISE = 0.08      # the eval RANDOM arms live outside the Dirichlet simplex -- cover them

    def sample_weights(self, rng):
        """(a,b,c,d,e) >= 0; a..d L1-normalised over the four informative legs. `exclude_heldout` is
        retained only as a switch for diagnostics; the shipped curriculum spans the full simplex."""
        for _ in range(32):
            w = rng.dirichlet([0.8, 0.8, 0.8, 0.8])       # mass on corners AND interiors
            e = float(rng.uniform(0.0, 0.5))
            if self.exclude_heldout and w[3] > 0.80 and max(w[0], w[1], w[2]) < 0.12:
                continue                                   # too close to pure HELF -- held out
            return float(w[0]), float(w[1]), float(w[2]), float(w[3]), e
        return 0.25, 0.25, 0.25, 0.25, 0.1

    def build_bank(self, rng, n_orders=512, kmax=32):
        """Precompute a bank of ask-orders. Sampling a fresh order per EXAMPLE costs an 18,359-element
        gaussian draw for the noise leg -- 1.2M randn per batch of 64, which dominated the training step
        (~3 s/step, measured). The bank is regenerated every epoch, so across a run the model still sees
        thousands of distinct askers; only WITHIN an epoch are orders reused. No user or item is dropped:
        this reduces strategy resampling, not data."""
        return [self.order(rng, kmax) for _ in range(n_orders)]

    def order(self, rng, k, weights=None):
        """Top-k ask order under a sampled (or supplied) weighting.

        With probability P_PURE_NOISE the score is pure noise. Dirichlet weights are never exactly zero
        and the noise leg is capped, so a RANDOM ask-order is outside the family's support -- yet the
        eval random arms live there, and G-NOHARM (stay at the prior on useless questions) is defined on
        exactly that regime. Without this the model would never train on it."""
        if weights is None and rng.random() < self.P_PURE_NOISE:
            sc = rng.standard_normal(self.ni)
            kk = min(k, self.ni)
            idx = np.argpartition(-sc, kk - 1)[:kk]
            return idx[np.argsort(-sc[idx])].astype(np.int64)
        a, b, c, d, e = weights if weights is not None else self.sample_weights(rng)
        s = a * self.pop + b * self.H + c * self.H0 + d * self.helf
        if e > 0:
            s = s + e * rng.standard_normal(self.ni)
        kk = min(k, self.ni)
        idx = np.argpartition(-s, kk - 1)[:kk]
        return idx[np.argsort(-s[idx])].astype(np.int64)


def make_interview_example(u, rng, fam, kmax_full=None, bank=None):
    """One interview-regime example.

    Returns (inp_sids, inp_levels, inp_vals, unseen_sids, targets, negatives) or None.
    `unseen_sids` is the NEW channel: items that were ASKED and came back unanswered. They are never
    targets (targets are rated), which is precisely why the existing NLL can teach the exposure branch
    without any auxiliary loss.
    """
    its = np.asarray(u["items"], np.int64)
    liked = np.asarray(u["liked"], np.int64)
    if len(its) < 4 or len(liked) < 2:
        return None
    # Hold out targets FIRST, so the interview can never ask about a target and leak it.
    ntg = max(1, len(liked) // 3)
    tg = rng.choice(liked, size=ntg, replace=False)
    keep = ~np.isin(its, tg)
    pool_s = its[keep]
    pool_l = np.asarray(u["levels"], np.int64)[keep]
    pool_v = np.asarray(u["vals"], np.float32)[keep]
    if len(pool_s) < 1:
        return None

    k = int(rng.choice(BUDGETS)) if kmax_full is None else kmax_full
    asked = (bank[rng.integers(len(bank))][:k] if bank is not None else fam.order(rng, k))
    rated = {int(s): j for j, s in enumerate(pool_s)}          # the answerability rule: rated <=> answerable
    ans_idx, unseen = [], []
    for i in asked:
        j = rated.get(int(i))
        if j is None:
            unseen.append(int(i))
        else:
            ans_idx.append(j)
    ans_idx = np.asarray(ans_idx, np.int64)
    inp_s = pool_s[ans_idx] if len(ans_idx) else np.empty(0, np.int64)
    inp_l = pool_l[ans_idx] if len(ans_idx) else np.empty(0, np.int64)
    inp_v = pool_v[ans_idx] if len(ans_idx) else np.empty(0, np.float32)

    negs = np.setdiff1d(u.get("disliked", np.empty(0, np.int64)), inp_s, assume_unique=False)
    return (inp_s, inp_l, inp_v, np.asarray(unseen, np.int64), tg, negs)


def make_empty_example(u, rng, n_keep=0):
    """The k=0 / k=1 bucket. THE PRIOR-ANCHORING MECHANISM: with no evidence the NLL on held-out likes
    trains the empty-set decode toward the population marginal directly. No loss term, no blend."""
    its = np.asarray(u["items"], np.int64)
    liked = np.asarray(u["liked"], np.int64)
    if len(liked) < 2:
        return None
    if n_keep == 0:
        inp_s = np.empty(0, np.int64); inp_l = np.empty(0, np.int64); inp_v = np.empty(0, np.float32)
        tg = liked
    else:
        j = rng.choice(len(its), size=min(n_keep, len(its)), replace=False)
        inp_s = its[j]; inp_l = np.asarray(u["levels"], np.int64)[j]
        inp_v = np.asarray(u["vals"], np.float32)[j]
        tg = np.setdiff1d(liked, inp_s, assume_unique=False)
        if len(tg) == 0:
            return None
    negs = np.setdiff1d(u.get("disliked", np.empty(0, np.int64)), inp_s, assume_unique=False)
    return (inp_s, inp_l, inp_v, np.empty(0, np.int64), tg, negs)


def draw(u, rng, fam, make_full, bank=None):
    """Regime-mixture draw. `make_full` is the EXISTING full-profile/dropout generator, kept unchanged
    so the full-profile objective is protected (design sheet: 35% of examples)."""
    r = rng.random()
    if r < P_FULL:
        out = make_full(u, rng)
        if out is None:
            return None
        inp, lv, sv, tgt, negs = out
        return (inp, lv, sv, np.empty(0, np.int64), tgt, negs)
    if r < P_FULL + P_INTERVIEW:
        return make_interview_example(u, rng, fam, bank=bank)
    if r < P_FULL + P_INTERVIEW + P_K0:
        return make_empty_example(u, rng, n_keep=0)
    return make_empty_example(u, rng, n_keep=1)
