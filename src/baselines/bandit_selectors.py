r"""bandit_selectors.py -- GLM-UCB and Thompson sampling as INTERVIEW QUESTION SELECTORS over the
frozen-basis Gaussian posterior of `belief_mf`.

ROLE IN PAPER B (Act 1, roster item 3). `belief_mf` is already in the table as a CONSUMER -- it eats
whatever questions a static arm chose. These two selectors turn the same posterior into the *asking*
side, which is what makes the row "the bandit family" (NICF's own baselines; the Interactive
Collaborative Filtering lineage) rather than "one Bayesian recommender". Nothing is retrained: the
frozen iALS item factors and the empirical Gaussian prior are exactly the ones the consumer row uses,
and the conjugate update is `belief_mf.GaussianBelief` itself -- imported, not reimplemented.

THE MODEL (belief_mf, verbatim; f = 200 frozen iALS factors)
    prior        u ~ N(mu_0, Sigma_0)                 mu_0, Sigma_0 = mean/cov of TRAIN user factors
    observation  r_i = <u, y_i> + eps,  eps ~ N(0, s2)
    posterior    A_t     = Sigma_0^-1 + (1/s2) * sum_{i in S_t} y_i y_i^T      (PRECISION)
                 mu_t    = A_t^-1 ( Sigma_0^-1 mu_0 + (1/s2) * sum_{i in S_t} r_i y_i )
                 Cov_t   = A_t^-1

ACQUISITION 1 -- GLM-UCB  (`GLMUCBSelector`)
    q_t = argmax_{i not yet asked}   <y_i, mu_t>  +  alpha * sqrt( y_i^T A_t^-1 y_i )
  Lineage and the honest naming. This is the IDENTITY-LINK (Gaussian) member of the GLM-UCB family of
  Filippi, Cappe, Garivier & Szepesvari (NeurIPS 2010, "Parametric Bandits: The Generalized Linear
  Case"), whose general form is  argmax_i  mu_link(<y_i, theta_hat>) + rho(t) * ||y_i||_{A^-1}.  With a
  Gaussian likelihood the link is the identity and the confidence set is exactly the posterior
  ellipsoid, so the rule collapses to the posterior-mean-plus-width form written above -- which is
  precisely the "linear UCB over a probabilistic-MF posterior" used for interactive recommendation by
  Zhao, Zhang & Wang, "Interactive Collaborative Filtering" (CIKM 2013), Sec. 4, and reused as a
  standard baseline by the NICF line (Zou et al., "Towards Question-Based Recommender Systems",
  SIGIR 2020). We keep the name GLM-UCB because that is the roster's name, and we state the link here
  so a reviewer is never surprised: no logistic/probit link is fitted, because the belief_mf
  observation model that Paper B is holding fixed is Gaussian.
  `alpha` is the one free constant. It is NOT tuned on test. Default 1.0 = the one-sigma posterior
  width, i.e. the natural scale of this parameterisation; if it is ever tuned it must be tuned on the
  10k VALIDATION users and the chosen value printed.

ACQUISITION 2 -- Thompson sampling  (`ThompsonSelector`)
    u~_t ~ N(mu_t, A_t^-1)  drawn INDEPENDENTLY PER USER PER TURN
    q_t = argmax_{i not yet asked}  <y_i, u~_t>
  This is the ConTS / ICF-TS read-out: same posterior, sample instead of mean-plus-bonus (Chapelle &
  Li NeurIPS 2011 for the general algorithm; Zhao et al. CIKM 2013 Sec. 4 and Li et al.'s
  conversational Thompson-sampling line for the recommender instantiation). `belief_mf`'s own header
  already documents that its posterior IS the ConTS core and that at full profile the draw collapses
  to the mean; here, at interview budgets, it does not collapse and the draw is the point.
  The draw is seeded per (seed, turn, user) so the run is reproducible regardless of how the harness
  chunks or groups users.

WHAT THE SELECTOR OBSERVES. Answered questions only, with the SIGNED value contract the belief_mf
consumer row uses: r_i = level_to_sv(level) (+1 at 5.0 stars, 0 at 2.75, -1 at 0.5). An asked-but-
unrated item burns the question and contributes NOTHING to the posterior -- belief_mf has no unknown
channel, which is a real property of the method and not an omission here.

CANDIDATE POOL. The full 18,359-item train vocabulary minus the items already asked of that user --
i.e. the identical universe the static global arms walk (`interview_strategies.build_orders` ranks all
n_items). No shortlist, no popularity pre-filter, no sampling: the argmax is over every catalogue item
at every turn. `pool=` can restrict it, but any restriction is a data reduction and needs sign-off.

COST, STATED HONESTLY. A_t and therefore the whole UCB width vector depend only on WHICH items a user
answered, never on the answers' values, and mu_t depends only on the answered (item, value) pairs. So
both are cached on the interview STATE and computed once per distinct state, not once per user: at
turn 0 every user shares one state. The expensive term is the width, sqrt(diag(Y A^-1 Y^T)), which is
one 200x200 Cholesky solve against Y^T (200 x 18,359) ~ 1.5 GFLOP ~ 0.1 s per distinct state. Worst
case (every user in a distinct state by the late turns) that is ~10^3 s per turn at k=16. That is the
price of an exact, un-subsampled sweep and it is paid, not avoided. Turn timings are printed by the
harness so the bill is visible.
"""
import numpy as np
from scipy import linalg as sla

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                "..", "instrument")))
from train_tower_t2 import level_to_sv          # the SIGNED answer contract, shared with belief_mf
from adaptive_harness import Selector

NEG = -np.inf


def _group_by_state(users, answered):
    """users -> {answered-tuple: [user ids]}. The posterior is a function of this key alone."""
    groups = {}
    for u in users:
        groups.setdefault(tuple(answered[u]), []).append(int(u))
    return groups


def _obs(key):
    """answered-tuple -> (item index array, signed value array) under the belief_mf contract."""
    if not key:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float64)
    cols = np.asarray([i for (i, _lv) in key], dtype=np.int64)
    vals = level_to_sv(np.asarray([lv for (_i, lv) in key], dtype=np.int64)).astype(np.float64)
    return cols, vals


class _PosteriorSelector(Selector):
    """Shared plumbing: state grouping, the posterior cache, the ask-mask and the argmax."""

    def __init__(self, belief, pool=None):
        self.belief = belief
        self.Y = belief.Y                                  # (n_items x f) frozen iALS item factors
        self.pool = None if pool is None else np.asarray(pool, dtype=np.int64)
        self._mu = {}
        self._A = {}
        self._n_states = []

    def reset(self, ctx):
        super().reset(ctx)
        self._mu, self._A, self._n_states = {}, {}, []
        if self.pool is not None:
            ctx["log"](f"[{self.name}] CANDIDATE POOL RESTRICTED to {len(self.pool)} items -- this is "
                       f"a data reduction and must not be reported without sign-off")

    def _posterior(self, key):
        if key not in self._mu:
            cols, vals = _obs(key)
            mu, A = self.belief.posterior(cols, vals)
            self._mu[key], self._A[key] = mu, A
        return self._mu[key], self._A[key]

    def _pick(self, base_scores, users, asked, out, idx_of):
        """base_scores: (n_items,) acquisition shared by the group. Mask each user's asked items and
        take the argmax. Kept as a separate step so the pool/mask logic exists exactly once."""
        for u in users:
            s = base_scores.copy()
            if self.pool is not None:
                keep = np.full(s.shape, NEG, dtype=s.dtype)
                keep[self.pool] = s[self.pool]
                s = keep
            s[np.asarray(asked[u], dtype=np.int64)] = NEG
            out[idx_of[u]] = int(np.argmax(s))

    def stats(self):
        return {"distinct_states_per_turn": self._n_states}


class GLMUCBSelector(_PosteriorSelector):
    r"""argmax_i  <y_i, mu_t> + alpha * sqrt(y_i^T A_t^-1 y_i)   over items not yet asked."""
    name = "glm_ucb"

    def __init__(self, belief, alpha=1.0, pool=None):
        super().__init__(belief, pool=pool)
        self.alpha = float(alpha)
        self._width = {}

    def reset(self, ctx):
        super().reset(ctx)
        self._width = {}

    def _wkey(self, key):
        """The cache key for the WIDTH: the answered ITEM SET, order- and value-free (A depends only
        on which items were answered). A hook rather than an inline expression because subclasses with
        a richer interview state -- the conversational bandits, whose state is (item answers, key-term
        answers) -- must widen this key while reusing the Cholesky machinery below verbatim."""
        return tuple(sorted(i for (i, _lv) in key))

    def _width_vec(self, key):
        """sqrt(diag(Y A^-1 Y^T)) for every catalogue item. Keyed on the ANSWERED ITEM SET only,
        because A is value-free -- a coarser and therefore cheaper key than the posterior mean's."""
        wkey = self._wkey(key)
        if wkey not in self._width:
            _mu, A = self._posterior(key)
            c, low = sla.cho_factor(A, lower=True)
            B = sla.cho_solve((c, low), self.Y.T)           # (f x n_items) = A^-1 Y^T
            v = np.einsum("ij,ji->i", self.Y, B)            # y_i^T A^-1 y_i
            self._width[wkey] = np.sqrt(np.maximum(v, 0.0))
        return self._width[wkey]

    def select(self, turn, users, asked, answered):
        groups = _group_by_state(users, answered)
        self._n_states.append(len(groups))
        out = np.empty(len(users), dtype=np.int64)
        idx_of = {int(u): j for j, u in enumerate(users.tolist())}
        for key, us in groups.items():
            mu, _A = self._posterior(key)
            base = self.Y @ mu + self.alpha * self._width_vec(key)
            self._pick(base, us, asked, out, idx_of)
        return out


class ThompsonSelector(_PosteriorSelector):
    r"""u~ ~ N(mu_t, A_t^-1) per user per turn; argmax_i <y_i, u~> over items not yet asked.

    The draw uses the Cholesky of the PRECISION: A = L L^T  =>  A^-1 = L^-T L^-1, so
    u~ = mu + L^-T z with z ~ N(0, I) has exactly covariance A^-1. No 200x200 inverse is formed."""
    name = "thompson"

    def __init__(self, belief, seed=98765, pool=None, chunk=512):
        super().__init__(belief, pool=pool)
        self.seed = int(seed)
        self.chunk = int(chunk)
        self._chol = {}

    def reset(self, ctx):
        super().reset(ctx)
        self._chol = {}

    def _cholesky(self, key):
        if key not in self._chol:
            _mu, A = self._posterior(key)
            self._chol[key] = np.linalg.cholesky(A)         # lower L with A = L L^T
        return self._chol[key]

    def select(self, turn, users, asked, answered):
        groups = _group_by_state(users, answered)
        self._n_states.append(len(groups))
        out = np.empty(len(users), dtype=np.int64)
        idx_of = {int(u): j for j, u in enumerate(users.tolist())}
        f = self.Y.shape[1]
        for key, us in groups.items():
            mu, _A = self._posterior(key)
            L = self._cholesky(key)
            for s in range(0, len(us), self.chunk):
                blk = us[s:s + self.chunk]
                # Per-(seed, turn, user) RNG: reproducible whatever the grouping/chunking turns out
                # to be, which a single shared stream would not be.
                Z = np.stack([np.random.default_rng([self.seed, int(turn), int(u)])
                              .standard_normal(f) for u in blk], axis=0)          # (b x f)
                U = mu[None, :] + sla.solve_triangular(L.T, Z.T, lower=False).T   # (b x f)
                S = U @ self.Y.T                                                  # (b x n_items)
                for r, u in enumerate(blk):
                    self._pick(S[r], [u], asked, out, idx_of)
        return out
