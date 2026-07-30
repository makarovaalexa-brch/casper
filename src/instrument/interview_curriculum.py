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

HELF IS HELD OUT (author decision, 2026-07-30). The exact HELF weight vector is excluded from the
training family and reserved for G-HELDOUT. HELF is the strategy on which Golbandi's lookup beats us, so
the headline comparison then runs on an ask-order the model has never trained against. Strictly harder,
and the only version of the claim worth making.

ANSWERABILITY is the transparent structural rule: an asked item is answered iff the user has rated it.
No LLM-derived answerability (banned for new work, 2026-07-22 audit). Answered -> the real half-star
level. Unanswered -> the UNSEEN channel, which the exposure branch consumes.
"""
import numpy as np

NLEV = 10
BUDGETS = (1, 2, 4, 8, 16, 32)
# Regime mixture (design sheet section 5). The k=0 bucket IS the prior-anchoring mechanism.
P_FULL, P_INTERVIEW, P_K0, P_K1 = 0.35, 0.55, 0.05, 0.05

# The HELD-OUT configuration: pure HELF. Never sampled for training; used only by G-HELDOUT.
HELDOUT_WEIGHTS = (0.0, 0.0, 0.0, 1.0, 0.0)     # (pop, H, H0, HELF, noise) -- pure HELF


def zs(x):
    x = np.asarray(x, np.float64)
    return (x - x.mean()) / max(x.std(), 1e-9)


class StrategyFamily:
    """Samples an ask-order per example from a continuous family spanning the literature's selectors."""

    def __init__(self, cnt, H, H0, helf, exclude_heldout=True):
        self.pop = zs(np.log1p(cnt))
        self.H = zs(H)
        self.H0 = zs(H0)
        self.helf = zs(helf)
        self.ni = len(self.pop)
        self.exclude_heldout = exclude_heldout

    def sample_weights(self, rng):
        """(a,b,c,d,e) >= 0; a..d L1-normalised over the four informative legs. Rejects the held-out
        HELF corner so the model never trains on the strategy G-HELDOUT tests."""
        for _ in range(32):
            w = rng.dirichlet([0.8, 0.8, 0.8, 0.8])       # mass on corners AND interiors
            e = float(rng.uniform(0.0, 0.5))
            if self.exclude_heldout and w[3] > 0.80 and max(w[0], w[1], w[2]) < 0.12:
                continue                                   # too close to pure HELF -- held out
            return float(w[0]), float(w[1]), float(w[2]), float(w[3]), e
        return 0.25, 0.25, 0.25, 0.25, 0.1

    def order(self, rng, k, weights=None):
        """Top-k ask order under a sampled (or supplied) weighting."""
        a, b, c, d, e = weights if weights is not None else self.sample_weights(rng)
        s = a * self.pop + b * self.H + c * self.H0 + d * self.helf
        if e > 0:
            s = s + e * rng.standard_normal(self.ni)
        kk = min(k, self.ni)
        idx = np.argpartition(-s, kk - 1)[:kk]
        return idx[np.argsort(-s[idx])].astype(np.int64)


def make_interview_example(u, rng, fam, kmax_full=None):
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
    asked = fam.order(rng, k)
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


def draw(u, rng, fam, make_full):
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
        return make_interview_example(u, rng, fam)
    if r < P_FULL + P_INTERVIEW + P_K0:
        return make_empty_example(u, rng, n_keep=0)
    return make_empty_example(u, rng, n_keep=1)
