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

ACQUISITION 3 -- BALD  (`BALDSelector`)
    q_t = argmax_i  I(u ; r_i | S_t) = H[E_u p(r_i|u)] - E_u H[p(r_i|u)]
  Houlsby, Huszar, Ghahramani & Lengyel, arXiv:1112.5745. UNDER THE belief_mf GAUSSIAN LIKELIHOOD THIS
  COLLAPSES TO 0.5*log(1 + sig_i^2/s2), a monotone function of the GLM-UCB width -- i.e. to pure
  exploration, GLM-UCB with the mean term deleted. That degeneracy is the reportable fact, it is
  measured by `--acq-check`, and the class docstring says why it happens and which likelihood would
  break it. Same disclosure discipline as GLM-UCB's identity link. Because the deleted mean term was
  GLM-UCB's ONLY value-dependent half, BALD here is value-free in exactly the sense acquisition 4 is.

ACQUISITION 4 -- A-OPTIMAL DESIGN  (`AOptimalSelector`)
    q_t = argmin_i tr(A_{t+1}(i)^-1),  A_{t+1}(i) = A_t + (1/s2) y_i y_i^T
        = argmax_i ||A_t^-1 y_i||^2 / (s2 + y_i^T A_t^-1 y_i)          [Sherman-Morrison, exact]
  Yu, Bi & Tresp (ICML 2006) transductive experimental design; Anava et al. (WWW 2015) for the
  recommender seed-set instantiation. The closed form reads both quadratic forms off the SAME
  B = A^-1 Y^T the width already needs, so no per-candidate inverse exists anywhere. It is a
  VALUE-FREE (classically static) criterion -- it adapts only to which questions turned out to be
  answerable -- and must be reported beside DRE rather than beside the adaptive arms.

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

MEMORY, STATED JUST AS HONESTLY (rewritten 2026-08-07 after an OOM; HARD RULE 11c).
The caching above is what makes the sweep affordable and it is ALSO what killed a 4-hour
conversational-bandit run: the state fans out toward one state per user (2,662 distinct states by turn
14 of 10,000 users, ~7,500 states cumulative over a k=16 interview), and the caches were unbounded
plain dicts holding, per state, a 200x200 float64 precision (320 KB) plus an 18,359 float64 width
vector (147 KB). ~3.5 GB, and the process died allocating 28 MB. Three changes, none of which alters a
selected item:

  (1) THE PRECISION IS NO LONGER CACHED AT ALL. A = Sigma_0^-1 + (1/s2) Y_S^T Y_S is a <=32x200 Gram
      plus an add -- microseconds -- while storing it was the single largest term in the bill. Only
      the posterior MEAN (200 floats, 1.6 KB) is kept, because computing it costs a 200x200 solve.
  (2) THE WIDTH VECTORS ARE CACHED AS float32. They are consumed by an argmax and by an addition to
      an O(1)-scaled mean term, never accumulated into a sum, so float32's ~1e-7 relative error is
      six orders of magnitude below the spacing of competing item scores. This is CHECKED, not
      asserted: `python src/baselines/bandit_selectors.py --dtype-check` re-runs a real-basis width
      at both precisions and compares the SELECTED ITEMS, and the harness's own smoke can be diffed
      end-to-end through its interview sidecar.
LAUNCH RECIPE for the two 2026-08-08 acquisitions (HARD RULE 11d/e: one heavy job at a time -- check
free RAM and `Get-CimInstance Win32_Process` for a live twin before starting).

  # algebra first: BALD's Gaussian degeneracy and A-optimal's Sherman-Morrison identity, both against
  # brute force. Seconds, synthetic, no data load.
  python src/baselines/bandit_selectors.py --acq-check

  # code path: first 200 test users, stamped SMOKE_NOT_CITABLE by the harness.
  python src/baselines/adaptive_harness.py \
      --selectors bald,aopt,glm_ucb,fixedorder:entropy0,fixedorder:greedy_items \
      --budgets 1,2,4,8,16 --only ours --arch i26 \
      --snapshot .cache/instrument/t2i26c_ep31.pt --no_floor --smoke_users 200 \
      --label task23_bald_aopt

  # THE RUN: all 10,000 canonical Liang test users, policy-B pool (the arm_n default -- there is no
  # flag; `--eddi_topm 0`-style truncation does not exist for these arms and the candidate set is the
  # full 18,359-item train vocabulary), NO-FLOOR, certified tower ep31 as the consumer.
  # The two COMPARATORS ride in the SAME run through `fixedorder:` for one reason: a later paired
  # bootstrap of (bald - entropy0) or (aopt - greedy) is then exact BY CONSTRUCTION -- same cohort,
  # same answer lookup, same scoring cursor, per-user vectors in one npz -- instead of being justified
  # across two files the way gtport_greedy_rescore had to. `--per_user` is what banks those vectors;
  # without it no paired test is possible later and the unpaired 1.96*sqrt(2)*SE form is banned.
  python src/baselines/adaptive_harness.py \
      --selectors bald,aopt,glm_ucb,fixedorder:entropy0,fixedorder:greedy_items \
      --budgets 1,2,4,8,16 --only ours --arch i26 \
      --snapshot .cache/instrument/t2i26c_ep31.pt \
      --no_floor --per_user --label bald_aopt_v1
  # -> experiments/baselines/interview/adaptive_table_bald_aopt_v1.json
  #    + ..._per_user.npz + ..._interviews/ (per-turn, contract-signed, resumable)

  (3) EVERY PER-STATE CACHE IS AN `LRUCache` WITH AN EXPLICIT CAP (`max_states`, default 4,096 ->
      ~300 MB for the width cache). Eviction costs recomputation, never correctness: every cached
      value is a pure function of its key. Hit/miss/eviction counts land in `stats()` so a cap set
      too low is visible as a hit rate, not as a mysterious slowdown.
"""
from collections import OrderedDict

import numpy as np
from scipy import linalg as sla

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                "..", "instrument")))
from train_tower_t2 import level_to_sv          # the SIGNED answer contract, shared with belief_mf
from adaptive_harness import Selector

NEG = -np.inf

# The dtype of every CACHED score/width vector. float32 halves the dominant memory term; see the
# header's MEMORY section and `--dtype-check` for the evidence that it does not move a selection.
# `CASPER_WIDTH_DTYPE=float64` restores the pre-2026-08-07 precision. It exists for exactly one
# purpose: running the SAME interview at both precisions and diffing the realised question sequences,
# which is the end-to-end version of `--dtype-check`. It is not a tuning knob.
WIDTH_DTYPE = np.float64 if os.environ.get("CASPER_WIDTH_DTYPE") == "float64" else np.float32
# Default cap on the number of interview STATES any one per-state cache may hold (HARD RULE 11c).
# 4,096 x 18,359 float32 = 300 MB for the width cache, the largest of them.
DEFAULT_MAX_STATES = 4096


class LRUCache:
    """A bounded dict with least-recently-used eviction, and counters.

    Deliberately tiny and dependency-free. The only contract that matters is that eviction is SAFE:
    every value stored here is a pure function of its key (the posterior mean, the width vector and
    the Cholesky factor all depend on the interview state and nothing else), so a miss costs time and
    never changes a result. `maxsize <= 0` means unbounded, which is the old behaviour and is kept
    only so a diagnostic run can reproduce it deliberately."""

    def __init__(self, maxsize=DEFAULT_MAX_STATES, name=""):
        self.maxsize = int(maxsize)
        self.name = name
        self._d = OrderedDict()
        self.hits = self.misses = self.evictions = 0

    def get(self, key):
        if key in self._d:
            self._d.move_to_end(key)
            self.hits += 1
            return self._d[key]
        self.misses += 1
        return None

    def put(self, key, val):
        self._d[key] = val
        self._d.move_to_end(key)
        while self.maxsize > 0 and len(self._d) > self.maxsize:
            self._d.popitem(last=False)
            self.evictions += 1
        return val

    def clear(self):
        self._d.clear()
        self.hits = self.misses = self.evictions = 0

    def __len__(self):
        return len(self._d)

    def stats(self):
        n = self.hits + self.misses
        return {"size": len(self._d), "cap": self.maxsize, "hits": self.hits,
                "misses": self.misses, "evictions": self.evictions,
                "hit_rate": round(self.hits / n, 4) if n else None}


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


def _a_inv_Yt(A, Y):
    """B = A^-1 Y^T, (f x n_items), by one Cholesky factorisation and one solve.

    THE one place the posterior covariance ever meets the catalogue. Three acquisitions read different
    quadratic forms off the SAME B -- the UCB/BALD width diag(Y A^-1 Y^T) = einsum('ij,ji->i', Y, B),
    and A-optimal's ||A^-1 y_i||^2 = einsum('ji,ji->i', B, B) -- so factoring it out is what stops a
    second acquisition from becoming a second 1.5-GFLOP solve of its own. No 200x200 inverse is formed.
    Extracted verbatim from GLMUCBSelector._width_vec (same cho_factor/cho_solve calls, same order), so
    every width this function now produces is bit-identical to the pre-2026-08-08 one."""
    c, low = sla.cho_factor(A, lower=True)
    return sla.cho_solve((c, low), Y.T)


class _PosteriorSelector(Selector):
    """Shared plumbing: state grouping, the posterior cache, the ask-mask and the argmax."""

    def __init__(self, belief, pool=None, max_states=DEFAULT_MAX_STATES):
        self.belief = belief
        self.Y = belief.Y                                  # (n_items x f) frozen iALS item factors
        self.pool = None if pool is None else np.asarray(pool, dtype=np.int64)
        self.max_states = int(max_states)
        # ONLY the posterior mean is cached (200 floats): the precision is rebuilt on demand because
        # storing it, at 320 KB per state, was the largest single term in the OOM. See the header.
        self._mu = LRUCache(self.max_states, "mu")
        self._n_states = []

    def reset(self, ctx):
        super().reset(ctx)
        self._mu.clear()
        self._n_states = []
        if self.pool is not None:
            ctx["log"](f"[{self.name}] CANDIDATE POOL RESTRICTED to {len(self.pool)} items -- this is "
                       f"a data reduction and must not be reported without sign-off")

    def _posterior(self, key):
        """(mu_t, A_t). The MEAN comes from the bounded cache; the PRECISION is recomputed every call.

        `belief.precision(cols)` is `belief.posterior(cols, vals)`'s own first two lines, so the A
        returned here is bit-identical to the one the old cache held -- it is the same expression on
        the same inputs, just not kept alive between calls."""
        cols, vals = _obs(key)
        mu = self._mu.get(key)
        if mu is None:
            mu, A = self.belief.posterior(cols, vals)
            self._mu.put(key, mu)
            return mu, A
        return mu, self.belief.precision(cols)

    def _wkey(self, key):
        """The cache key for any VALUE-FREE per-state quantity (the UCB width, the BALD score, the
        A-optimal gain): the answered ITEM SET, order- and value-free, because A depends only on which
        items were answered. A hook rather than an inline expression because subclasses with a richer
        interview state -- the conversational bandits, whose state is (item answers, key-term answers)
        -- must widen this key while reusing the Cholesky machinery verbatim. Lives on the shared base
        (moved up from GLMUCBSelector 2026-08-08) because three acquisitions now key on it."""
        return tuple(sorted(i for (i, _lv) in key))

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

    def _cache_stats(self):
        """Every bounded cache on this object, by attribute name. Reported so a cap that is too small
        shows up as a hit rate in the run JSON instead of as an unexplained slowdown."""
        return {n: c.stats() for n, c in sorted(vars(self).items()) if isinstance(c, LRUCache)}

    def stats(self):
        return {"distinct_states_per_turn": self._n_states, "cache": self._cache_stats(),
                "max_states": self.max_states, "width_dtype": np.dtype(WIDTH_DTYPE).name}


class GLMUCBSelector(_PosteriorSelector):
    r"""argmax_i  <y_i, mu_t> + alpha * sqrt(y_i^T A_t^-1 y_i)   over items not yet asked."""
    name = "glm_ucb"

    def __init__(self, belief, alpha=1.0, pool=None, max_states=DEFAULT_MAX_STATES):
        super().__init__(belief, pool=pool, max_states=max_states)
        self.alpha = float(alpha)
        self._width = LRUCache(self.max_states, "width")

    def reset(self, ctx):
        super().reset(ctx)
        self._width.clear()

    def _width_vec(self, key):
        """sqrt(diag(Y A^-1 Y^T)) for every catalogue item. Keyed on the ANSWERED ITEM SET only,
        because A is value-free -- a coarser and therefore cheaper key than the posterior mean's."""
        wkey = self._wkey(key)
        w = self._width.get(wkey)
        if w is None:
            _mu, A = self._posterior(key)
            B = _a_inv_Yt(A, self.Y)                        # (f x n_items) = A^-1 Y^T
            v = np.einsum("ij,ji->i", self.Y, B)            # y_i^T A^-1 y_i
            # The solve stays float64; only the CACHED result is narrowed (header, MEMORY (2)).
            w = self._width.put(wkey, np.sqrt(np.maximum(v, 0.0)).astype(WIDTH_DTYPE))
        return w

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


class BALDSelector(GLMUCBSelector):
    r"""BALD -- Bayesian Active Learning by Disagreement (Houlsby, Huszar, Ghahramani & Lengyel,
    arXiv:1112.5745, 2011). Acquisition = the mutual information between the PARAMETERS and the
    predicted answer:

        a_BALD(i) = I(u ; r_i | S_t) = H[ E_u p(r_i|u) ]  -  E_u H[ p(r_i|u) ]

    i.e. prefer the question on which the posterior DISAGREES WITH ITSELF -- high predictive entropy
    that is not explained by observation noise -- rather than the question whose predictive is merely
    uncertain. That distinction is the whole content of the paper.

    ================================================================================================
    WHICH LIKELIHOOD WE ASSUME, AND WHAT BALD REDUCES TO UNDER IT (read this before citing the row)
    ================================================================================================
    We assume the likelihood Paper B is holding fixed, `belief_mf`'s and nobody else's:

        r_i = <u, y_i> + eps,     eps ~ N(0, s2),     u | S_t ~ N(mu_t, A_t^-1)

    -- a Gaussian observation with a variance s2 that does not depend on u or on i. Then, writing
    sig_i^2 = y_i^T A_t^-1 y_i for the posterior variance of the noiseless utility:

        p(r_i | u)      = N(<u,y_i>, s2)          =>  H[p(r_i|u)]      = 0.5*log(2*pi*e*s2)   (const)
        p(r_i | S_t)    = N(<mu_t,y_i>, sig_i^2 + s2)
                                                  =>  H[p(r_i|S_t)]    = 0.5*log(2*pi*e*(sig_i^2+s2))

        a_BALD(i) = 0.5 * log( 1 + sig_i^2 / s2 )                                        [in nats]

    SO, PLAINLY: under a homoscedastic Gaussian likelihood BALD COLLAPSES TO A PURE VARIANCE RULE. The
    second term -- the expected conditional entropy, which is the entire point of BALD, the thing that
    makes it prefer disagreement over noise -- is a CONSTANT here, because a linear-Gaussian model puts
    the same aleatoric noise on every question. `log(1 + sig^2/s2)` is strictly increasing in sig^2, so

        argmax_i a_BALD(i)  ==  argmax_i sig_i^2  ==  argmax_i width_i  ==  GLM-UCB with alpha -> inf

    and this row is behaviourally "GLM-UCB's exploration term with the exploit term deleted". It is
    NOT a new source of signal on this model, and it must never be sold as one. We implement and report
    it anyway for two reasons: (i) it is a named, cited acquisition a reviewer will ask about, and the
    honest answer "it degenerates HERE, and here is the run" is worth more than an omission; (ii) the
    degeneracy is a property of the LIKELIHOOD, not of BALD -- it is exactly the same disclosure we
    made for GLM-UCB's identity link. The identity is not asserted, it is MEASURED:
    `python src/baselines/bandit_selectors.py --acq-check` re-runs both argmaxes and diffs them.

    THE SECOND CONSEQUENCE, WHICH IS THE ONE THAT MATTERS FOR THE TABLE. sig_i^2 is a function of A_t,
    and A_t is VALUE-FREE -- it contains which items were answered and not what the answers were (see
    GaussianBelief's note). The posterior MEAN is the only value-dependent half of GLM-UCB, and BALD is
    precisely GLM-UCB with that half deleted. So BALD on this model is not an answer-adaptive policy at
    all: like `AOptimalSelector` it adapts only to WHICH questions proved answerable, and with an
    always-answerable population it would degenerate to a single static order. Its distinct-ask-prefix
    count will track A-optimal's, and both belong beside DRE rather than beside glm_ucb/thompson in any
    adaptive-vs-static table. This is not a defect of the implementation; it is what BALD IS under a
    homoscedastic Gaussian likelihood, and a table that files it as adaptive is overclaiming.

    WHERE IT WOULD NOT COLLAPSE (stated so the reduction is not mistaken for a general claim): any
    likelihood whose conditional entropy depends on u -- a Bernoulli/probit like/dislike head, or a
    three-outcome answerability model of the kind `eddi_selector.EDDIThreeOutcomeSelector` carries. We
    do not fit one here, because swapping the observation model would make this a different method's
    row rather than BALD on our posterior.

    We score in NATS and keep the log rather than shortcutting to sig^2: the number in the log is then
    a real mutual information, comparable across turns, and a reader can check the units. `alpha` is
    inherited from GLMUCBSelector and is INERT -- the mean term never enters this acquisition."""
    name = "bald"

    def __init__(self, belief, pool=None, max_states=DEFAULT_MAX_STATES):
        super().__init__(belief, alpha=0.0, pool=pool, max_states=max_states)
        self.s2 = 1.0 / float(belief.inv_s2)

    def _bald_vec(self, key):
        """0.5*log(1 + sig_i^2/s2), nats, per catalogue item. Reuses the CACHED width, so a run with
        both `glm_ucb` and `bald` in it pays for the Cholesky solve once per state, not twice."""
        w = self._width_vec(key).astype(np.float64)
        return 0.5 * np.log1p(w * w / self.s2)

    def select(self, turn, users, asked, answered):
        groups = _group_by_state(users, answered)
        self._n_states.append(len(groups))
        out = np.empty(len(users), dtype=np.int64)
        idx_of = {int(u): j for j, u in enumerate(users.tolist())}
        for key, us in groups.items():
            self._pick(self._bald_vec(key), us, asked, out, idx_of)
        return out

    def stats(self):
        s = super().stats()
        s["bald_obs_noise_s2"] = self.s2
        s["bald_reduces_to"] = ("0.5*log(1+sig^2/s2) under the Gaussian belief_mf likelihood -- a "
                                "MONOTONE function of the GLM-UCB width, i.e. pure exploration")
        s["bald_design_class"] = ("VALUE-FREE, like aopt: the width depends on WHICH items were "
                                  "answered, never on the answers. Report beside DRE, not beside "
                                  "the adaptive arms.")
        return s


class AOptimalSelector(_PosteriorSelector):
    r"""A-OPTIMAL (transductive) EXPERIMENTAL DESIGN -- Yu, Bi & Tresp, "Active Learning via
    Transductive Experimental Design" (ICML 2006); applied to recommender seed-set selection by Anava,
    Golan, Golbandi, Karnin, Lempel, Rokhlenko & Somekh, "Budget-Constrained Item Cold-Start Handling
    in Collaborative Filtering Recommenders via Optimal Design" (WWW 2015). Acquisition: ask the
    question that minimises the TRACE of the posterior covariance after folding it in --

        q_t = argmin_{i not yet asked}  tr( A_{t+1}(i)^-1 ),     A_{t+1}(i) = A_t + (1/s2) y_i y_i^T

    i.e. minimise the total posterior variance of the user vector (A-optimality), as against
    D-optimality's log-determinant or E-optimality's largest eigenvalue.

    THE CLOSED FORM (why this is near-free rather than 18,359 matrix inversions per turn). By
    Sherman-Morrison, with c = 1/s2,

        A_{t+1}(i)^-1 = A_t^-1  -  c * (A_t^-1 y_i)(A_t^-1 y_i)^T / (1 + c * y_i^T A_t^-1 y_i)

    and taking the trace of a rank-one term is just a squared norm:

        tr(A_{t+1}(i)^-1) = tr(A_t^-1)  -  ||A_t^-1 y_i||^2 / (s2 + y_i^T A_t^-1 y_i)

    tr(A_t^-1) is the same for every candidate, so MINIMISING the post-fold trace is exactly

        q_t = argmax_i  g_i,      g_i = ||A_t^-1 y_i||^2 / (s2 + sig_i^2)

    Both quadratic forms are columns of ONE solve B = A_t^-1 Y^T (`_a_inv_Yt`): g's numerator is the
    squared column norm of B, its denominator the same diag(Y A^-1 Y^T) the UCB width uses. So the
    whole 18,359-candidate criterion costs one 200x200 Cholesky plus two einsums per interview STATE --
    the same bill GLM-UCB already pays -- and no per-candidate inverse is ever formed. The identity
    against a brute-force explicit inverse is CHECKED, not asserted: `--acq-check`.

    NOTE what g_i is NOT: it is not the width. ||A^-1 y||^2 = y^T A^-2 y weights the posterior's small
    eigendirections quadratically where the width weights them linearly, so A-optimal leans harder into
    the least-determined directions of taste and the two rankings genuinely differ.

    ================================================================================================
    STATIC OR SEQUENTIAL? -- SEQUENTIAL, BUT VALUE-FREE. Say it this way and no other.
    ================================================================================================
    Classical A-optimal design is a STATIC criterion: A_{t+1} does not contain the answers, only the
    QUESTIONS, so a greedy A-optimal sequence can be computed offline before the interview starts and
    handed to every user identically. That is how Anava et al. use it -- a budgeted seed SET.

    Our implementation is the greedy sequential form, and on our harness it is not quite static, for
    exactly one reason: an asked-but-UNRATED item burns the turn and enters no measurement (the
    belief_mf contract -- there is no unknown channel), so A_t differs between a user who could answer
    the opener and one who could not. The criterion therefore adapts to WHICH questions were
    answerable and never to WHAT was answered. Two consequences to report, not to hide:
      * the distinct-ask-prefix count grows with turn, but only up to the number of distinct
        ANSWERABILITY patterns -- far below an answer-dependent policy's fan-out. That number is the
        diagnostic, and it is printed per turn by the harness;
      * with a hypothetical always-answerable user population this arm would be exactly one static
        order. So it belongs BESIDE DRE, in the learned/derived-static family, and not beside the
        adaptive arms -- a table that lists it as adaptive is overclaiming.
    The candidate's own contribution is folded in ASSUMING IT WILL BE ANSWERED, which is the classical
    criterion as published; A-optimal has no answerability model and we do not give it one."""
    name = "aopt"

    def __init__(self, belief, pool=None, max_states=DEFAULT_MAX_STATES):
        super().__init__(belief, pool=pool, max_states=max_states)
        self.s2 = 1.0 / float(belief.inv_s2)
        self._gain = LRUCache(self.max_states, "aopt_gain")

    def reset(self, ctx):
        super().reset(ctx)
        self._gain.clear()

    def _gain_vec(self, key):
        """g_i = ||A^-1 y_i||^2 / (s2 + y_i^T A^-1 y_i), the trace REDUCTION candidate i would buy.
        Keyed on the answered ITEM SET (`_wkey`) because, like the width, A is value-free."""
        gk = self._wkey(key)
        g = self._gain.get(gk)
        if g is None:
            _mu, A = self._posterior(key)
            B = _a_inv_Yt(A, self.Y)                        # (f x n_items) = A^-1 Y^T
            num = np.einsum("ji,ji->i", B, B)               # ||A^-1 y_i||^2 = y_i^T A^-2 y_i
            den = self.s2 + np.einsum("ij,ji->i", self.Y, B)
            # float32 for the CACHED vector, as for the width: consumed by an argmax, never summed.
            g = self._gain.put(gk, (num / den).astype(WIDTH_DTYPE))
        return g

    def select(self, turn, users, asked, answered):
        groups = _group_by_state(users, answered)
        self._n_states.append(len(groups))
        out = np.empty(len(users), dtype=np.int64)
        idx_of = {int(u): j for j, u in enumerate(users.tolist())}
        for key, us in groups.items():
            self._pick(self._gain_vec(key), us, asked, out, idx_of)
        return out

    def stats(self):
        s = super().stats()
        s["aopt_obs_noise_s2"] = self.s2
        s["aopt_design_class"] = ("greedy A-optimal, VALUE-FREE: adapts only to WHICH questions were "
                                  "answerable, never to the answers. Report beside DRE, not beside "
                                  "the adaptive arms.")
        return s


class ThompsonSelector(_PosteriorSelector):
    r"""u~ ~ N(mu_t, A_t^-1) per user per turn; argmax_i <y_i, u~> over items not yet asked.

    The draw uses the Cholesky of the PRECISION: A = L L^T  =>  A^-1 = L^-T L^-1, so
    u~ = mu + L^-T z with z ~ N(0, I) has exactly covariance A^-1. No 200x200 inverse is formed."""
    name = "thompson"

    def __init__(self, belief, seed=98765, pool=None, chunk=512, max_states=DEFAULT_MAX_STATES):
        super().__init__(belief, pool=pool, max_states=max_states)
        self.seed = int(seed)
        self.chunk = int(chunk)
        self._chol = LRUCache(self.max_states, "chol")

    def reset(self, ctx):
        super().reset(ctx)
        self._chol.clear()

    def _cholesky(self, key):
        L = self._chol.get(key)
        if L is None:
            _mu, A = self._posterior(key)
            # float64: this factor is SOLVED AGAINST to draw the Thompson sample, not argmaxed, so it
            # is not one of the vectors the header narrows.
            L = self._chol.put(key, np.linalg.cholesky(A))  # lower L with A = L L^T
        return L

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


# ============================================================== float32 width: the selection-identity check
def basis_cache_path(train, hp=None):
    """Path of the contract-signed frozen-iALS basis sidecar for this train matrix + hyperparameters.

    The basis is a 10-minute seeded iALS fit that every bandit arm shares, so banking it is what makes
    RESUMING a crashed multi-hour run cheap (HARD RULE 11b). The contract is in the FILENAME: a
    different train matrix or a different hp gives a different path, so a stale basis can never be
    loaded for the wrong run -- it can only be missed."""
    import belief_mf
    hp = hp or belief_mf.MF_HP
    import os as _os
    sig = (f"{train.shape[0]}x{train.shape[1]}_nnz{train.nnz}_f{hp['factors']}_r{hp['reg']}"
           f"_a{hp['alpha']}_it{hp['iterations']}")
    return _os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".cache",
                         "baselines", f"belief_mf_basis_{sig}.npz")


def load_or_fit_belief(train, n_items, log=print):
    """`belief_mf.fit(train, n_items).belief`, with the frozen basis banked to `basis_cache_path`.

    On a MISS this runs the published path verbatim and banks (Y, mu0, Sig0_used) off the resulting
    object. On a HIT it rebuilds `GaussianBelief` from those three arrays with prior_scale=1.0 and
    jitter=0.0 -- Sig0_used is ALREADY the scaled-and-jittered matrix the fresh object holds, so the
    reconstruction is bit-identical rather than merely equivalent."""
    import belief_mf
    p = os.path.abspath(basis_cache_path(train))
    if os.path.exists(p):
        z = np.load(p)
        bel = belief_mf.GaussianBelief(z["Y"], z["mu0"], z["Sig0_used"],
                                       belief_mf.DEFAULTS["obs_noise"], 1.0, 0.0)
        log(f"[belief_mf] frozen basis from cache {os.path.basename(p)} "
            f"(Y={bel.Y.shape}); the 10-minute iALS fit is NOT repeated")
        return bel
    bel = belief_mf.fit(train, n_items, log=log).belief
    os.makedirs(os.path.dirname(p), exist_ok=True)
    np.savez(p + ".tmp.npz", Y=bel.Y, mu0=bel.mu0, Sig0_used=bel.Sig0)
    os.replace(p + ".tmp.npz", p)
    log(f"[belief_mf] frozen basis BANKED -> {p}")
    return bel


def dtype_check(belief=None, n_items=18359, f=200, states=(0, 1, 2, 3, 4, 8, 16, 32), alpha=1.0,
                seed=98765, log=print):
    """Does caching the width vector in float32 ever change the SELECTED ITEM? (HARD RULE 11c.)

    Runs the exact acquisition the selector runs -- `Y @ mu + alpha * width` under a per-user asked
    mask -- twice, once with the width kept in float64 and once narrowed to float32, and compares the
    argmax. Reports the number of disagreements and the worst relative width error, so the answer is
    a measurement rather than an assertion. `belief=None` builds a synthetic basis of the real shape;
    pass the real `GaussianBelief` (see `load_or_fit_belief`) for the citable version."""
    rng = np.random.default_rng(seed)
    if belief is None:
        import belief_mf
        Y = rng.standard_normal((n_items, f)) * 0.09      # ~ the real mean ||y_a|| = 0.094
        U = rng.standard_normal((2000, f)) * 0.5
        belief = belief_mf.GaussianBelief(Y, U.mean(0), np.cov(U, rowvar=False), 1.0, 1.0, 1e-6)
        log(f"[dtype-check] SYNTHETIC basis {Y.shape} (shape-realistic, values are not the real ones)")
    Y = belief.Y
    n_items, f = Y.shape
    n_dis = 0
    n_cmp = 0
    worst_rel = 0.0
    worst_gap = np.inf
    for s in states:
        for rep in range(8):
            cols = rng.choice(n_items, size=s, replace=False) if s else np.empty(0, dtype=np.int64)
            vals = rng.uniform(-1.0, 1.0, size=s)
            mu, A = belief.posterior(cols.astype(np.int64), vals)
            c, low = sla.cho_factor(A, lower=True)
            B = sla.cho_solve((c, low), Y.T)
            w64 = np.sqrt(np.maximum(np.einsum("ij,ji->i", Y, B), 0.0))
            w32 = w64.astype(np.float32)
            worst_rel = max(worst_rel, float(np.max(np.abs(w32.astype(np.float64) - w64)
                                                    / np.maximum(w64, 1e-30))))
            base = Y @ mu
            s64 = base + alpha * w64
            s32 = base + alpha * w32
            for _u in range(25):                          # 25 distinct asked-masks per state
                asked = rng.choice(n_items, size=max(s, 1), replace=False)
                a64, a32 = s64.copy(), s32.copy()
                a64[asked] = NEG
                a32[asked] = NEG
                i64, i32 = int(np.argmax(a64)), int(np.argmax(a32))
                n_cmp += 1
                n_dis += int(i64 != i32)
                top2 = np.partition(a64, -2)[-2:]
                worst_gap = min(worst_gap, float(abs(top2[1] - top2[0])))
    log(f"[dtype-check] {n_cmp} argmaxes over {len(states)} state sizes x 8 draws x 25 masks: "
        f"{n_dis} DISAGREEMENTS between a float64 and a float32 width")
    log(f"[dtype-check] worst relative width error {worst_rel:.3e}; smallest top-1/top-2 score gap "
        f"seen {worst_gap:.3e} -- the error must be well under the gap for the argmax to be safe")
    assert n_dis == 0, f"float32 width changed {n_dis}/{n_cmp} selections -- do NOT narrow the cache"
    return {"comparisons": n_cmp, "disagreements": n_dis, "worst_rel_width_err": worst_rel,
            "min_top1_top2_gap": worst_gap}


def acq_check(belief=None, n_items=2000, f=40, states=(0, 1, 2, 4, 8, 16), s2=1.0, seed=98765,
              log=print):
    """The two claims the new acquisitions rest on, MEASURED rather than asserted (HARD RULE 5).

    (1) BALD DEGENERACY. Under the Gaussian likelihood, a_BALD = 0.5*log(1+sig^2/s2) is monotone in the
        UCB width, so BALD's argmax must equal a pure-width argmax on every state and every asked-mask.
        A disagreement would mean the derivation in BALDSelector's docstring is wrong.
    (2) A-OPTIMAL CLOSED FORM. tr(A^-1) - g_i must equal the BRUTE-FORCE trace of the explicitly
        inverted A + (1/s2) y_i y_i^T, item by item. This is the Sherman-Morrison step, and it is the
        one place an algebra slip would be invisible in the output (a wrong-but-smooth ranking).
        Deliberately run at a SMALL f and n so the brute force -- 2,000 explicit 40x40 inverses per
        state -- is affordable; the identity is dimension-free.

    Synthetic basis by default (shape-realistic is irrelevant here: this is algebra, not data); pass
    the real GaussianBelief for the citable version."""
    rng = np.random.default_rng(seed)
    if belief is None:
        import belief_mf
        Y = rng.standard_normal((n_items, f)) * 0.09
        U = rng.standard_normal((2000, f)) * 0.5
        belief = belief_mf.GaussianBelief(Y, U.mean(0), np.cov(U, rowvar=False), s2, 1.0, 1e-6)
        log(f"[acq-check] SYNTHETIC basis {Y.shape}, s2={s2}")
    Y = belief.Y
    n_items, f = Y.shape
    s2 = 1.0 / float(belief.inv_s2)
    n_bald_cmp = n_bald_dis = 0
    worst_trace_err = 0.0
    n_aopt_argmax_dis = 0
    for s in states:
        cols = rng.choice(n_items, size=s, replace=False) if s else np.empty(0, dtype=np.int64)
        vals = rng.uniform(-1.0, 1.0, size=s)
        _mu, A = belief.posterior(cols.astype(np.int64), vals)
        B = _a_inv_Yt(A, Y)
        sig2 = np.einsum("ij,ji->i", Y, B)
        # ---- (1) BALD vs pure width
        bald = 0.5 * np.log1p(np.maximum(sig2, 0.0) / s2)
        width = np.sqrt(np.maximum(sig2, 0.0))
        for _rep in range(25):
            mask = rng.choice(n_items, size=max(s, 1), replace=False)
            b, w = bald.copy(), width.copy()
            b[mask] = NEG
            w[mask] = NEG
            n_bald_cmp += 1
            n_bald_dis += int(int(np.argmax(b)) != int(np.argmax(w)))
        # ---- (2) A-optimal closed form vs brute force
        g = np.einsum("ji,ji->i", B, B) / (s2 + sig2)
        Ainv = np.linalg.inv(A)
        tr0 = float(np.trace(Ainv))
        closed = tr0 - g
        brute = np.empty(n_items)
        for i in range(n_items):
            y = Y[i]
            brute[i] = float(np.trace(np.linalg.inv(A + (1.0 / s2) * np.outer(y, y))))
        worst_trace_err = max(worst_trace_err,
                              float(np.max(np.abs(closed - brute) / np.abs(brute))))
        n_aopt_argmax_dis += int(int(np.argmax(g)) != int(np.argmin(brute)))
    log(f"[acq-check] BALD: {n_bald_dis}/{n_bald_cmp} argmax disagreements against a PURE-WIDTH rule "
        f"-- 0 confirms BALD reduces to exploration under the Gaussian likelihood")
    log(f"[acq-check] A-optimal: worst relative error of the Sherman-Morrison post-fold trace vs an "
        f"EXPLICIT inverse = {worst_trace_err:.3e} over {len(states)} states x {n_items} candidates; "
        f"{n_aopt_argmax_dis} argmax/argmin disagreements")
    assert n_bald_dis == 0, "BALD did not match the pure-width rule -- the derivation is wrong"
    assert worst_trace_err < 1e-9, f"Sherman-Morrison trace off by {worst_trace_err:.3e}"
    assert n_aopt_argmax_dis == 0, "A-optimal closed form picked a different item than brute force"
    return {"bald_comparisons": n_bald_cmp, "bald_disagreements": n_bald_dis,
            "aopt_worst_rel_trace_err": worst_trace_err,
            "aopt_argmax_disagreements": n_aopt_argmax_dis}


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--dtype-check", dest="dtype_check", action="store_true",
                    help="float32-vs-float64 width: do the SELECTED ITEMS ever differ?")
    ap.add_argument("--acq-check", dest="acq_check", action="store_true",
                    help="BALD's Gaussian degeneracy + A-optimal's Sherman-Morrison identity, both "
                         "measured against brute force")
    ap.add_argument("--real-basis", action="store_true",
                    help="run the dtype check on the REAL frozen iALS basis (loads/fits it)")
    _a = ap.parse_args()
    if _a.acq_check:
        acq_check()
        print("[acq-check] PASS")
    elif _a.dtype_check:
        _bel = None
        if _a.real_basis:
            from arm_n import load_arm_n
            _D = load_arm_n(log=print)
            _bel = load_or_fit_belief(_D["train"], _D["n_items"], log=print)
        dtype_check(belief=_bel)
        print("[dtype-check] PASS")
    else:
        print(__doc__)
