r"""conversational_bandit_selectors.py -- THE KEY-TERM (conversational) BANDIT FAMILY as interview
question selectors: ConUCB (WWW'20), ConLinUCB-BS / -MCR (AAAI'23), CLiME/CLiSK (KDD'25).

WHY THIS FAMILY AND NO OTHER (audit: external_literature/findings/bandit_rl_crs_baselines.md sec. 1).
Every other adaptive line in the CRS literature either needs a known target item, an omniscient
simulator, or a candidate pool seeded by the target's own attribute. The conversational-bandit line is
the ONLY published adaptive family whose action space is ours: at each round it asks about a KEY-TERM
(an attribute / genre / tag / concept) OR about an ITEM, in one algorithm, with an explicit model of
how key-term feedback moves the item-level parameter estimate -- and with no target item and no
simulator (reward is replayed from real ratings). Their METRIC (cumulative regret) is not ours and
their numbers are not orderable against ours; the POLICY is what ports, and that is what this file
ports. Nothing here is a reproduction of a published number.

  ConUCB       Zhang, Xie, Li & Lui, WWW 2020, arXiv:1906.01219                        [full text V]
  ConLinUCB    Wang, Liu, Li & Lui, AAAI 2023 pp.10288-10295, arXiv:2303.00315         [full text V]
               reference numpy code: github.com/ZhiyongWangWzy/ConLinUCB
  CLiSK/CLiME  Liu, Li, Dai & Lui, KDD 2025, arXiv:2505.21393                          [full text V]
  Hier-UCB     Zuo, Hu, Yu, Li, Zhao & Joe-Wong, arXiv:2209.06129                      [abstract V]
               -- the source of the `representative` key-term option below.

================================================================================================
ONE ACTION SPACE: HOW AN ITEM QUESTION AND A CONCEPT QUESTION ARE MADE COMMENSURABLE
================================================================================================
This is the scientific core of the family and the thing a reviewer will go straight to, so it is
stated in full rather than implied.

Three separate commensurabilities have to hold, and each is arranged explicitly:

(1) THE FEATURE. Both action types must produce a vector in the SAME space, because both enter the
    same normal equations. Our space is the f=200 frozen iALS item-factor basis `Y` that
    `belief_mf.GaussianBelief` already uses -- an item question contributes its own row `x_a = Y[a]`;
    a key-term question contributes `xtilde_k`, an f-vector built from the concept's member items
    (three constructions, see KEY-TERM REPRESENTATION below). There is no second basis and no
    projection step.

(2) THE ANSWER. Both action types must produce a scalar on the SAME scale, or the regression is
    silently reweighted by action type. Ours are already matched by construction and this is not a
    coincidence -- both channels were designed on [-1, +1]:
        item question    r_a    = level_to_sv(half-star level)     +1 at 5.0*, 0 at 2.75*, -1 at 0.5*
        concept question rtil_k = the signed implicit-watch value V  (signed_answers.signed_values;
                                  NPMI + w_val*VAL clipped to [-1,1], meh band forced to 0)
    Neither is imputed, neither comes from a simulator or an LLM, and neither is target-aware.

(3) THE SELECTION INDEX. Both action types must be scored by ONE functional, or "which type to ask"
    is decided outside the algorithm. The unifying quantity in this family is the posterior width in
    the design-matrix norm, ||phi||_{M^-1}: ConLinUCB-MCR selects key-terms by EXACTLY that
    (`argmax_k alpha_t ||xtilde_k||_{M_t^-1}`), and the LinUCB arm index carries the same term as its
    exploration bonus. So `turn_rule="width"` puts both types in one argmax over
    `alpha * ||phi||_{M^-1}` and lets the algorithm decide the turn type itself.
    THE HONEST CAVEAT, stated because it matters: ConUCB and ConLinUCB do NOT do this. They decide the
    turn type by a FIXED SCHEDULE `b(t)` (below) and only then run a type-specific rule. That fixed
    schedule is the thing every successor attacks (Hier-UCB's entire contribution; CLiME's adaptive
    trigger). So `turn_rule` is a constructor option, each arm DEFAULTS to its own paper's rule, and
    `turn_rule="width"` is available on all of them as the one-action-space variant. Any run must say
    which was used; it is logged and it lands in `stats()`.

    *** MEASURED, AND IT MATTERS (SMOKE, first 200 test users, 2026-08-05 -- code-path run, not a
    citable number): under `turn_rule="width"` the arms ask ZERO key-terms at EVERY budget up to 16. ***
    The reason is structural, not a bug, and it is the same effect this project already found from the
    other direction ("a divisive ITEM is one few have seen"): the width criterion maximises over 18,359
    item rows and only ~1,031 key-term rows, and the least-explored direction in a 200-dim posterior is
    always some obscure item, never a concept aggregating hundreds of films. A pure-uncertainty rule
    therefore CANNOT choose a concept on a catalogue this size. The harness prints a loud
    "DEGENERATED to item-only" line whenever an arm asks no key-terms, precisely so such a row can
    never be read as "concepts did not help". Treat `turn_rule="width"` as a measured negative result
    about uncertainty-only arbitration, and run the comparison arms on their papers' schedule.

WHAT AN UNANSWERABLE QUESTION COSTS. Both types burn the turn and contribute nothing:
    item    -- asked but not in the user's fold-in-side rated history  -> no row in M, no row in b
    concept -- BAND_REFUSE (expected member exposure E < TAU=1.5) or fewer than 2 rated members
               -> no row in M, no row in b
This is the contract that the whole ICF/bandit lineage cannot express: ICF (CIKM'13) defines
`theta_hit = 1 if the rating is no less than 4, and 0 otherwise`, so an item the user never saw scores
identically to an item they hated. Burning and being wrong are the same event under their metric. Here
they are not.

================================================================================================
THE ESTIMATORS, WITH THE EQUATIONS AND THE DEVIATIONS
================================================================================================

ConLinUCB (AAAI'23) -- the ONE-STEP estimator, and the reason this file exists:

    M_t = sum_tau x_{a_tau} x_{a_tau}^T  +  sum_tau xtilde_{k_tau} xtilde_{k_tau}^T  +  beta I
    b_t = sum_tau x_{a_tau} r_{a_tau}    +  sum_tau xtilde_{k_tau} rtilde_{k_tau}
    theta_t = M_t^-1 b_t                                                       [V, AAAI'23]

  i.e. A KEY-TERM ANSWER ENTERS THE SAME NORMAL EQUATIONS AS AN ITEM ANSWER, with the key-term's
  feature vector in place of the item's. That is our signed concept fold, in bandit clothing, and it
  is what makes "one action space" a property of the estimator rather than of the wrapper.

  * ConLinUCB-MCR key-term rule:  k in argmax_k alpha_t ||xtilde_k||_{M_t^-1}          [V, AAAI'23]
    (alpha_t is a positive scalar shared by all k, so it cannot change the argmax; it is kept in the
    code for faithfulness to the paper's statement, not for effect.)
  * ConLinUCB-BS key-term rule: draw uniformly from a precomputed BARYCENTRIC SPANNER of the key-term
    feature set -- a fixed, user-independent exploration basis                          [V, AAAI'23]
  * Arm rule (both): the LinUCB index over the one-step estimator,
        a in argmax_a <x_a, theta_t> + alpha ||x_a||_{M_t^-1}
    (Li, Chu, Langford & Schapire, WWW 2010, arXiv:1003.0146, with ICF's transposition -- one frozen
    item factor per arm, one posterior over the user -- exactly as `bandit_selectors.GLMUCBSelector`
    already documents.)

  DEVIATION, stated once: the paper's ridge `beta I` is replaced BY DEFAULT with our empirical
  population precision `Sigma_0^-1` from `belief_mf.GaussianBelief` (the prior whose mean and
  covariance are the TRAIN user factors' mean and covariance). `beta I` is the isotropic special case
  of that, and the population prior is the choice this project already made and already has published
  provenance for (IGCF, arXiv:2309.01453, meta-learned Bayesian prior). Pass `beta=<float>` to get the
  paper's literal isotropic ridge instead; both are exposed and the choice is logged.

CLiME (KDD'25) -- key-terms aligned with the SMALL-EIGENVALUE eigendirections of the design matrix:

    k in argmax_k |xtilde_k^T v_i|,  v_i the eigenvector(s) of M_t with the smallest eigenvalue(s) [V]

  CLiSK's Gaussian-smoothed key-term contexts (`k = argmax_k <xtilde~_k, theta_t>` with noise added to
  the key-term features) are cheap and are available as `smooth_sigma > 0` (CLiSK-ME when combined).
  Their diagnosis of ConUCB is the reason this arm exists: ConUCB's O(d sqrt(T) log T) is "no better
  than not conversing at all" [V].

ConUCB (WWW'20) -- the original TWO-STAGE estimator, item-level fit SHRUNK toward a key-term fit:

    theta_t   = argmin_theta  lam * sum (theta^T x_a - r_a)^2 + (1-lam) ||theta - theta~_t||^2   [V]
              => theta_t = M_t^-1 ( lam sum x_a r_a + (1-lam) theta~_t ),
                 M_t = lam sum x_a x_a^T + (1-lam) I
    theta~_t  = argmin_theta~ sum_k sum_tau ( theta~^T xtilde_k - rtilde_k )^2 + lamtil ||theta~||^2
              => theta~_t = Mtil_t^-1 btil_t,  Mtil_t = sum xtilde xtilde^T + lamtil I           [V]
    PROPAGATION (Eq. 4, verbatim):
        E[rtilde_{k,t}] = sum_a ( w_{a,k} / sum_a' w_{a',k} ) E[r_{a,t}],  sum_k w_{a,k} = 1      [V]
      -- "a key-term's reward is the weighted average of its member items' rewards". This is exactly
      what `keyterm_features(..., mode="uniform_member_mean")` builds, and it is why that mode is the
      DEFAULT for the ConUCB arm specifically: using any other key-term vector under ConUCB would
      contradict the estimator it is derived from.
    KEY-TERM SELECTION (Eq. 8, verbatim):
        k = argmax_k' || X_t M_t^-1 Mtil_{t-1}^-1 xtilde_{k',t} ||_2^2
                      / ( 1 + xtilde_{k',t}^T Mtil_{t-1}^-1 xtilde_{k',t} )                      [V]
      X_t stacks the candidate arms' features, so the numerator is v^T (X_t^T X_t) v with
      v = M_t^-1 Mtil_{t-1}^-1 xtilde_k -- one 200x200 Gram matrix, precomputed once per pool.
    CONVERSATION FREQUENCY: b(t) = 5*floor(log t); a key-term turn happens whenever b(t) > b(t-1).
      The paper tests b(t)=5*floor(log t) and b(t)=5*floor(t/50) [V]; the former is the default here.

================================================================================================
KEY-TERM REPRESENTATION -- three papers, three answers, and running them adjudicates OUR design
================================================================================================
`keyterm_repr` is a constructor option with three settings. This is deliberately the most swappable
thing in the file, because the three settings are three published and MUTUALLY INCONSISTENT claims
about what a key-term's feature vector should be, and our own concept-fold design has been assuming
one of them:

  (a) "uniform_member_mean"  ConUCB Eq. 4's own definition: the (weight-normalised) average of the
                             member items' feature vectors. Weights w_{a,k} = 1{a in members(k)} /
                             |concepts(a)|, which satisfies the paper's `sum_k w_{a,k} = 1` per arm.
  (b) "representative"       Hier-UCB (arXiv:2209.06129) reports, from its own survey plus real data,
                             that "key-term rewards are mainly affected by rewards of REPRESENTATIVE
                             items" -- NOT by a uniform average over members. Implemented as the mean
                             over the top-`n_rep` members by TRAIN interaction count. `n_rep` is a
                             free constant (default 20) and is NOT tuned on test.
  (c) "whitened_centroid"    OURS: the certified concept channel's direction `d_c` --
                             `belief_layer.build_concept_dirs`, i.e. centre the basis, strip its top-1
                             principal component (the popularity axis), average over members,
                             normalise. Transported here by applying the IDENTICAL recipe to the
                             bandit's own basis `Y` instead of to the tower decoder `Wd`, because a
                             direction in the tower's decoder space is not a vector in the bandit's
                             item-factor space and pretending otherwise would be the commensurability
                             error this whole file exists to avoid. Same recipe, different basis --
                             that deviation is deliberate and is logged.

SCALE. (a) and (b) inherit the member items' magnitudes; (c) is unit-norm by construction. Since
`xtilde xtilde^T` enters M, a scale difference alone would change how much a concept answer moves the
posterior, and the three-way comparison would confound DIRECTION with MAGNITUDE. So `norm_match=True`
(default) rescales every `xtilde_k` to the mean member ||Y_a||, making the three settings differ only
in direction. `norm_match=False` leaves the raw scales; both are logged.

The in-use setting is logged at reset() and returned by stats().

================================================================================================
FREE CONSTANTS -- all of them, with provenance, NONE tuned on test
================================================================================================
  alpha       1.0    UCB exploration width. Matches `bandit_selectors.GLMUCBSelector`'s default (one
                     posterior sigma) so the conversational arms and the item-only bandit arm sit on
                     the same exploration scale and the comparison is about the ACTION SPACE, not
                     about alpha. LinUCB/ConLinUCB state alpha = 1 + sqrt(ln(2/delta)/2) (= 1.96 at
                     delta=0.05); pass alpha=1.96 for the paper's literal constant. If alpha is ever
                     tuned it must be tuned on the 10k VALIDATION users and the value printed.
  beta        None   ConLinUCB's ridge. None => the empirical population precision Sigma_0^-1 (see
                     DEVIATION above). A float gives the paper's literal beta*I; the papers' own
                     default is 1.0.
  lam         0.5    ConUCB's trust weight between the item-level fit and the key-term-level fit.
                     *** The paper's chosen value is NOT verified in our audit. *** 0.5 is the neutral
                     midpoint of the paper's stated range lam in [0,1], chosen for that reason and for
                     no other. Do not report it as ConUCB's value.
  lam_til     1.0    ConUCB's key-term-level ridge lamtil. Standard unit ridge; also unverified as the
                     paper's own choice.
  b_schedule  "5logt" ConUCB's conversation frequency b(t) = 5*floor(log t)  [V verbatim]. The
                     alternative the paper also tests is b(t) = 5*floor(t/50) ("5t50"); at our budgets
                     (k <= 16) that one never fires, which is itself worth stating.
  n_eig       1      CLiME: how many of the smallest eigendirections of M_t the key-term score is
                     taken against. The paper writes a single v_i; >1 aggregates by max |xtilde^T v_i|.
  smooth_sigma 0.0   CLiSK's Gaussian key-term-context smoothing (0 = off = pure CLiME).
  n_rep       20     "representative" key-term option: how many top-popularity members define the
                     representative set. Hier-UCB gives no number (abstract only), so this is ours.
  spanner_iters 50   Barycentric-spanner refinement passes (Awerbuch-Kleinberg style greedy
                     determinant maximisation); it terminates early when no swap improves |det|.

================================================================================================
COST, AND WHY IT IS AFFORDABLE
================================================================================================
Exactly as `bandit_selectors` documents: M_t and therefore the whole width vector depend only on WHICH
questions were answered, never on the answers' values, and theta depends only on the (question, value)
pairs. Both are cached on the interview STATE -- here the state key is the pair (item answers, concept
answers) -- so the 200x200 Cholesky and its solve against Y^T (200 x 18,359, ~1.5 GFLOP, ~0.1 s) are
paid once per DISTINCT state, not once per user. At turn 0 every user shares one state. The key-term
width needs a second solve against Xtilde^T (200 x ~1,030), which is free by comparison. Distinct-state
counts per turn are reported in stats() and printed by the harness, so the bill is visible.

  python src/baselines/conversational_bandit_selectors.py --smoke     # synthetic code-path check
"""
import os
import sys
import time
import numpy as np
from scipy import linalg as sla

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_ROOT, "src", "instrument"))

from adaptive_harness import Selector                                    # noqa: E402
from bandit_selectors import (GLMUCBSelector, LRUCache, DEFAULT_MAX_STATES,   # noqa: E402
                              WIDTH_DTYPE, _obs, NEG)

CACHE_DIR = os.path.join(_ROOT, ".cache", "baselines")
KEYTERM_MODES = ("uniform_member_mean", "representative", "whitened_centroid")
B_SCHEDULES = ("5logt", "5t50", "always", "never")


# ===================================================================== key-term feature construction
def _whitened_centroid_dirs(Y, member_sets, n_concepts):
    """`belief_layer.build_concept_dirs`'s recipe applied to the BANDIT's basis Y.

    Imported rather than re-derived would be preferable, but `build_concept_dirs` takes a torch tensor
    and a {tag: sids} dict keyed on genome tag ids; this takes the same steps on the numpy basis this
    module already holds. The steps are identical and are asserted against the torch function in the
    smoke: centre, strip the top-1 principal component (the popularity axis), mean over members,
    normalise."""
    Yc = Y - Y.mean(axis=0, keepdims=True)
    # top-1 PC via the Gram matrix (f x f); f=200 so a full eigh is instant and exact.
    G = Yc.T @ Yc
    w, V = np.linalg.eigh(G)
    pc = V[:, -1]
    pc = pc / max(np.linalg.norm(pc), 1e-12)
    Yw = Yc - np.outer(Yc @ pc, pc)                                      # popularity-stripped
    X = np.zeros((n_concepts, Y.shape[1]), dtype=np.float64)
    for c in range(n_concepts):
        mem = member_sets[c]
        v = Yw[mem].mean(axis=0)
        X[c] = v / max(np.linalg.norm(v), 1e-12)
    return X


def keyterm_features(Y, member_sets, cnt, mode="uniform_member_mean", n_rep=20, norm_match=True,
                     log=print):
    """(C x f) key-term feature matrix in the bandit's own item-factor basis. See the module header's
    KEY-TERM REPRESENTATION section for what each mode claims and who claims it."""
    assert mode in KEYTERM_MODES, f"unknown keyterm_repr '{mode}'; known {KEYTERM_MODES}"
    C, f = len(member_sets), Y.shape[1]
    if mode == "whitened_centroid":
        X = _whitened_centroid_dirs(Y, member_sets, C)
    else:
        X = np.zeros((C, f), dtype=np.float64)
        if mode == "uniform_member_mean":
            # ConUCB Eq. 4's weights: w_{a,k} = 1{a in members(k)} / |concepts(a)|, so sum_k w_{a,k}=1
            # per arm exactly as the paper requires; xtilde_k = sum_a w_{a,k} x_a / sum_a w_{a,k}.
            deg = np.zeros(Y.shape[0], dtype=np.float64)
            for c in range(C):
                deg[member_sets[c]] += 1.0
            wdeg = 1.0 / np.maximum(deg, 1.0)
            for c in range(C):
                mem = member_sets[c]
                w = wdeg[mem]
                X[c] = (w[:, None] * Y[mem]).sum(axis=0) / max(w.sum(), 1e-12)
        else:                                                            # "representative"
            for c in range(C):
                mem = member_sets[c]
                top = mem[np.argsort(-cnt[mem])[:min(n_rep, len(mem))]]
                X[c] = Y[top].mean(axis=0)
    if norm_match:
        # Remove magnitude from the three-way comparison: every key-term enters M at the mean member
        # item's scale, so the modes differ ONLY in direction. See the header's SCALE paragraph.
        for c in range(C):
            tgt = float(np.linalg.norm(Y[member_sets[c]], axis=1).mean())
            nrm = float(np.linalg.norm(X[c]))
            if nrm > 1e-12:
                X[c] *= tgt / nrm
    log(f"[conv_bandit] key-term features: mode={mode} C={C} f={f} norm_match={norm_match} "
        f"n_rep={n_rep if mode == 'representative' else '-'} "
        f"mean||xtilde||={float(np.linalg.norm(X, axis=1).mean()):.4f} "
        f"(mean||y_a||={float(np.linalg.norm(Y, axis=1).mean()):.4f})")
    return X


def barycentric_spanner(X, iters=50, log=print):
    """A 2-approximate BARYCENTRIC SPANNER of the row set of X (Awerbuch & Kleinberg's greedy
    determinant maximisation), as used by ConLinUCB-BS [V, AAAI'23].

    A barycentric spanner is a subset B of f rows such that every row of X is in the span of B with
    coefficients in [-1,1]. The construction: start from any full-rank subset (built here by greedy
    max-|projection-residual| -- deterministic, no RNG), then repeatedly replace row j of B by whichever
    row of X maximises |det| when swapped in; stop when no swap improves |det| by more than a factor of
    2 (that factor IS the 2-approximation guarantee) or after `iters` passes.

    THE SWAP IS DONE BY A RANK-1 DETERMINANT UPDATE, not by recomputing determinants. Replacing row j
    of B by x multiplies the determinant by exactly `x^T (B^-1)[:, j]`, so one f x f inverse per j
    gives the improvement factor for ALL C candidate rows in a single mat-vec. The naive version --
    C x f full determinants per pass -- is ~10^5 200x200 slogdets per pass and does not finish.

    Work happens in an orthonormal basis of the key-term row space (from the same greedy pass), so a
    rank-deficient key-term set is handled by spanning its actual rank rather than by failing on a
    singular matrix.

    The result is a FIXED, USER-INDEPENDENT set of key-terms chosen purely for spanning -- which is
    what makes ConLinUCB-BS double as a genuinely different STATIC concept bank, next to popularity /
    entropy / polar."""
    C, f = X.shape
    kmax = min(f, C)
    Q, chosen = [], []
    R = X.copy()
    for _ in range(kmax):
        nrm = np.linalg.norm(R, axis=1)
        nrm[chosen] = -1.0
        i = int(np.argmax(nrm))
        if nrm[i] <= 1e-9 * max(1.0, float(np.linalg.norm(X, axis=1).max())):
            break
        chosen.append(i)
        q = R[i] / nrm[i]
        Q.append(q)
        R = R - np.outer(R @ q, q)
    k = len(chosen)
    if k < kmax:
        log(f"[conv_bandit] spanner: key-term features have rank {k} < f={f}; the spanner is that "
            f"rank (a real property of the concept bank, not a truncation)")
    Qm = np.asarray(Q)                                                   # (k x f) orthonormal
    Z = X @ Qm.T                                                         # (C x k) row-space coords
    B = np.asarray(chosen, dtype=np.int64)
    Bm = Z[B]                                                            # (k x k), invertible
    swaps = 0
    for _ in range(int(iters)):
        improved = False
        Bi = np.linalg.inv(Bm)
        for j in range(k):
            ratio = np.abs(Z @ Bi[:, j])                                 # |det(new)/det(old)| per row
            i = int(np.argmax(ratio))
            if ratio[i] > 2.0:                                           # the 2-approximation rule
                Bm[j] = Z[i]
                B[j] = i
                Bi = np.linalg.inv(Bm)
                improved = True
                swaps += 1
        if not improved:
            break
    _s, logdet = np.linalg.slogdet(Bm)
    log(f"[conv_bandit] barycentric spanner: {k} key-terms, {swaps} swaps, log|det|={logdet:.2f}")
    return B


# ===================================================================== the concept answer table
def build_concept_stack(D, snapshot, keyterm_repr="uniform_member_mean", n_rep=20, norm_match=True,
                        Y=None, log=print, cache=True, apply_neg_cap=True):
    """Everything the key-term half of the action space needs, for the harness's OWN test cohort.

    Returns a dict with:
      n_concepts, tags, member_sets  the genome bank (relevance >= 0.5, >= 30 member items) -- the
                                     IDENTICAL bank definition `concept_fold` / `concept_distill_train`
                                     use, so the certified concept channel and this selector talk about
                                     the same objects.
    *** THE ANSWER CONVENTION IS A CONTRACT, NOT A DETAIL (2026-08-07). *** `apply_neg_cap=True` (the
    default, and what every run before 2026-08-08 used) applies `signed_values`' POPULATION-wide
    negative-mass cap across all 1,031 concepts at once. `apply_neg_cap=False` is the CERTIFIED-ERA
    convention -- what `strategy_channel_suite.build_greedy_env` uses, and what the concept channel
    cv3 was TRAINED under -- and it must be paired with `signed_answers.cap_negatives` over the
    REVEALED token list at fold time (the caller does that; see `adaptive_harness`'s
    `--concept_neg_cap 0` path and `convbandit_greedy_rescore.py`). The same 16 greedy concepts score
    .2473/.1154 under the certified-era convention and .2298/.0977 under the capped one, so numbers
    from the two conventions are NEVER differenced. Every run states which it used.

      V, ANSWERABLE                  (n_test_users x C) signed implicit-watch values and the
                                     answerability mask, from `signed_answers.signed_values` on the
                                     users' fold-in-side rated history `g_te_tr` (held-out targets
                                     already removed upstream by arm_n). BEHAVIOURAL: no simulator, no
                                     LLM, no target.
      Xtilde                         (C x f) key-term features in the bandit basis Y.

    The genome member matrix is built from `proc/unique_sid.txt` directly (movieId -> sid is exactly
    that file's line order), NOT via `reproduce_partition`, so this costs one pandas read of
    genome-scores.csv and is then cached to .cache/baselines/.
    """
    import pandas as pd
    from scipy import sparse
    from signed_answers import load_prereg, signed_values, BAND_REFUSE
    from concept_fold import build_member_matrix

    ni = D["n_items"]
    ck = os.path.join(CACHE_DIR, "conv_bandit_genome_bank.npz")
    if cache and os.path.exists(ck):
        z = np.load(ck, allow_pickle=True)
        tags = z["tags"].tolist()
        starts, flat = z["starts"], z["members_flat"]
        member_sets = {i: flat[starts[i]:starts[i + 1]] for i in range(len(tags))}
        log(f"[conv_bandit] genome bank from cache: {len(tags)} concepts")
    else:
        gpath = os.path.join(_ROOT, "data", "movielens", "genome-scores.csv")
        usid = [int(x) for x in open(os.path.join(_ROOT, "data", "ml-25m", "proc",
                                                  "unique_sid.txt")).read().split()]
        assert len(usid) == ni, f"unique_sid.txt has {len(usid)} rows but n_items={ni}"
        m2s = {m: i for i, m in enumerate(usid)}
        g = pd.read_csv(gpath)
        g = g[g["relevance"] >= 0.5]
        g = g[g["movieId"].isin(m2s)]
        g["sid"] = g["movieId"].map(m2s).astype(np.int64)
        members = {int(t): np.sort(sub["sid"].values) for t, sub in g.groupby("tagId")}
        members = {t: mm for t, mm in members.items() if len(mm) >= 30}
        tags = sorted(members.keys())
        member_sets = {i: members[t] for i, t in enumerate(tags)}
        os.makedirs(CACHE_DIR, exist_ok=True)
        starts = np.cumsum([0] + [len(member_sets[i]) for i in range(len(tags))]).astype(np.int64)
        np.savez(ck, tags=np.asarray(tags), starts=starts,
                 members_flat=np.concatenate([member_sets[i] for i in range(len(tags))]))
        log(f"[conv_bandit] genome bank built: {len(tags)} concepts (relevance>=0.5, >=30 members)")

    C = len(tags)
    Mm = build_member_matrix({i: member_sets[i] for i in range(C)}, list(range(C)), ni)
    cnt = np.asarray(D["train"].sum(axis=0)).ravel().astype(np.float64)
    pexp = (cnt @ np.asarray(Mm.todense())) / max(cnt.sum(), 1e-9)
    prereg, item_mean = load_prereg()

    G = D["g_te_tr"].tocsr()
    items_list, stars_list = [], []
    for u in range(G.shape[0]):
        s, e = G.indptr[u], G.indptr[u + 1]
        items_list.append(G.indices[s:e].astype(np.int64))
        stars_list.append(G.data[s:e].astype(np.float32))
    t0 = time.time()
    V, FOLD, BAND, ANS = signed_values(items_list, stars_list, item_mean, Mm, pexp, ni, prereg,
                                       apply_neg_cap=apply_neg_cap)
    answerable = ANS & (BAND != BAND_REFUSE)
    log(f"[conv_bandit] concept answers for {G.shape[0]} test users x {C} concepts in "
        f"{time.time() - t0:.1f}s: answerable rate {float(answerable.mean()):.3f}, "
        f"mean |v| on answerable {float(np.abs(V[answerable]).mean()):.3f}; "
        f"ANSWER CONVENTION = "
        f"{'population neg-cap (apply_neg_cap=True)' if apply_neg_cap else 'CERTIFIED-ERA (apply_neg_cap=False; cap_negatives over the revealed list at fold time)'}")

    out = {"n_concepts": C, "tags": tags, "member_sets": member_sets, "Mm": Mm,
           "V": V.astype(np.float64), "answerable": answerable, "band": BAND,
           "cnt": cnt, "prereg": prereg, "apply_neg_cap": bool(apply_neg_cap)}
    if Y is not None:
        out["Xtilde"] = keyterm_features(Y, member_sets, cnt, mode=keyterm_repr, n_rep=n_rep,
                                         norm_match=norm_match, log=log)
    return out


def concept_answer(stack, u, c):
    """The BEHAVIOURAL concept answer: (answerable, value). An unanswerable concept burns the turn."""
    if not bool(stack["answerable"][u, c]):
        return False, 0.0
    return True, float(stack["V"][u, c])


# ===================================================================== the selectors
class _ConvBanditBase(GLMUCBSelector):
    """Shared machinery for every arm in the family.

    INHERITS FROM `GLMUCBSelector` ON PURPOSE, and that is the whole point of the class hierarchy: its
    `_width_vec` is a Cholesky factorisation of the posterior PRECISION followed by one solve against
    Y^T -- which is exactly `||x_a||_{M_t^-1}` for every catalogue item, cached on the answered-set key.
    ConLinUCB's M_t *is* that precision once the key-term rows are in it, so the only thing this class
    has to change is WHICH rows go into the precision. `_posterior` is overridden to append the
    key-term rows; `_width_vec` is then inherited verbatim and computes the right thing.

    ACTION ENCODING (the harness contract). The harness's action space is
        [0, n_items)                 -> ask about that ITEM
        [n_items, n_items+C)         -> ask about key-term c = a - n_items
    and `asked[u]` carries the raw action ids, so the harness's no-re-ask assert covers both types and
    the budget is genuinely shared between them. The harness hands the concept ANSWERS back through
    `ctx["conc_state"]["answered"]`, a list it mutates in place (the `select` signature is fixed and
    shared with every other selector, so a fourth argument was not an option).

    CANDIDATE POOL: taken from `ctx["cand_pool"]` if the harness supplies one and from the full train
    vocabulary otherwise. Nothing is hard-coded here -- a protocol change on the harness side (e.g.
    masking every rated fold-in item at scoring time) arrives through ctx without touching this file.
    """
    name = "conv_bandit"

    def __init__(self, belief, stack, alpha=1.0, beta=None, turn_rule=None,
                 b_schedule="5logt", seed=98765, pool=None, max_states=DEFAULT_MAX_STATES):
        super().__init__(belief, alpha=alpha, pool=pool, max_states=max_states)
        self.stack = stack
        self.Xt = np.ascontiguousarray(stack["Xtilde"])                  # (C x f)
        self.C = int(stack["n_concepts"])
        self.beta = None if beta is None else float(beta)
        self.turn_rule = turn_rule or self.DEFAULT_TURN_RULE
        assert self.turn_rule in ("schedule", "width"), self.turn_rule
        self.b_schedule = b_schedule
        assert b_schedule in B_SCHEDULES, f"unknown b_schedule {b_schedule}"
        self.seed = int(seed)
        self._ktw = LRUCache(self.max_states, "ktw")
        self._n_kt = []
        self.n_items = None

    # ------------------------------------------------------------------ state / posterior
    def reset(self, ctx):
        super().reset(ctx)
        self._ktw.clear()
        self._n_kt = []
        self.n_items = int(ctx["n_items"])
        self.conc = ctx["conc_state"]
        if self.pool is None and ctx.get("cand_pool") is not None:
            self.pool = np.asarray(ctx["cand_pool"], dtype=np.int64)
        ctx["log"](f"[{self.name}] key-term repr={self.stack.get('_repr', '?')} C={self.C} "
                   f"alpha={self.alpha} beta={'population-prior' if self.beta is None else self.beta} "
                   f"turn_rule={self.turn_rule} b_schedule={self.b_schedule} "
                   f"pool={'FULL vocabulary' if self.pool is None else f'{len(self.pool)} items'}")

    def _group(self, users, answered):
        """State = (item answers, concept answers). The posterior is a function of this pair alone."""
        groups = {}
        ca = self.conc["answered"]
        for u in users:
            groups.setdefault((tuple(answered[u]), tuple(ca[u])), []).append(int(u))
        return groups

    def _rows(self, key):
        """(Phi, r): the stacked design rows and observations of BOTH action types.
        Phi = [ x_{a_1} ... x_{a_p} ; xtilde_{k_1} ... xtilde_{k_q} ],  r = [ r_a ... ; rtilde_k ... ].
        This IS ConLinUCB's `sum x x^T + sum xtilde xtilde^T` / `sum x r + sum xtilde rtilde`."""
        ikey, ckey = key
        cols, vals = _obs(ikey)                                          # item side, shared contract
        Phi = [self.Y[cols]] if len(cols) else []
        r = [vals] if len(cols) else []
        if ckey:
            cid = np.asarray([c for (c, _v) in ckey], dtype=np.int64)
            cv = np.asarray([v for (_c, v) in ckey], dtype=np.float64)
            Phi.append(self.Xt[cid])
            r.append(cv)
        if not Phi:
            return (np.zeros((0, self.Y.shape[1])), np.zeros(0))
        return np.vstack(Phi), np.concatenate(r)

    def _posterior(self, key):
        """(theta, M). ConLinUCB's one-step estimator, with our two ridge options.

        The arithmetic is `belief_mf.GaussianBelief.posterior`'s, line for line, with an explicit
        feature matrix in place of an index into Y -- because a key-term feature is not a catalogue
        row and cannot be addressed by an item id. Nothing else differs:
            beta is None : M = Sigma_0^-1 + (1/s2) Phi^T Phi     (the population prior; the default)
            beta float   : M = beta I     + (1/s2) Phi^T Phi     (ConLinUCB's literal ridge)

        MEMORY (2026-08-07, HARD RULE 11c): M is REBUILT on every call and only the 200-float mean is
        cached. Phi has at most k <= 16 rows, so `Phi.T @ Phi` is a ~1 ms Gram, while keeping M alive
        cost 320 KB per interview state and was the largest single term in the 2026-08-05 OOM. The
        arithmetic below is unchanged, line for line.
        """
        bel = self.belief
        Phi, r = self._rows(key)
        f = self.Y.shape[1]
        if self.beta is None:
            P0, rhs0 = bel.Sig0_inv, bel.Sig0_inv_mu0
        else:
            P0, rhs0 = self.beta * np.eye(f), np.zeros(f)
        if Phi.shape[0] == 0:
            mu = np.linalg.solve(P0, rhs0) if self.beta is not None else bel.mu0
            return mu, P0
        M = P0 + bel.inv_s2 * (Phi.T @ Phi)
        mu = self._mu.get(key)
        if mu is None:
            b = rhs0 + bel.inv_s2 * (Phi.T @ r)
            mu = self._mu.put(key, np.linalg.solve(M, b))
        return mu, M

    def _wkey(self, key):
        """Widen `GLMUCBSelector`'s width cache key to the two-type interview state. M is still
        value-free, so the key is (answered item set, answered key-term set) and the inherited
        Cholesky-solve-against-Y^T body is reused unchanged."""
        return (tuple(sorted(i for (i, _l) in key[0])), tuple(sorted(c for (c, _v) in key[1])))

    def _width_vec_kt(self, key):
        """||xtilde_k||_{M^-1} for every key-term. Second Cholesky solve, against Xtilde^T (200 x ~1k)
        rather than Y^T (200 x 18,359), so it costs ~5% of the item width and is cached the same way."""
        wkey = self._wkey(key)
        w = self._ktw.get(wkey)
        if w is None:
            _mu, M = self._posterior(key)
            c, low = sla.cho_factor(M, lower=True)
            B = sla.cho_solve((c, low), self.Xt.T)
            v = np.einsum("ij,ji->i", self.Xt, B)
            w = self._ktw.put(wkey, np.sqrt(np.maximum(v, 0.0)).astype(WIDTH_DTYPE))
        return w

    # ------------------------------------------------------------------ turn type
    def _b(self, t):
        """ConUCB's conversation-frequency function b(t) [V verbatim: b(t)=5*floor(log t)]."""
        if self.b_schedule == "always":
            return int(t)                       # every turn is a key-term turn (the static BS arm)
        if self.b_schedule == "never":
            return 0
        t = max(int(t), 1)                      # log t is undefined at t=0; the paper's t is 1-based
        if self.b_schedule == "5logt":
            return 5 * int(np.floor(np.log(t)))
        return 5 * int(np.floor(t / 50.0))      # "5t50"

    def _scheduled_keyterm(self, turn):
        """A key-term turn happens whenever b(t) > b(t-1); t is 1-based, as in the paper."""
        t = int(turn) + 1
        return self._b(t) > self._b(t - 1)

    # ------------------------------------------------------------------ arm-specific scores
    def _keyterm_scores(self, key, mu):
        raise NotImplementedError

    def _item_scores(self, key, mu):
        """The LinUCB index over the one-step estimator (ConLinUCB's arm rule)."""
        return self.Y @ mu + self.alpha * self._width_vec(key)

    # ------------------------------------------------------------------ selection
    def _pick_mixed(self, item_s_fn, kt_s, u, asked_u, force_kt):
        """One argmax over the mixed action space, with each user's own asked-mask applied to BOTH
        halves. `force_kt` is the turn-type decision: True = this turn is a key-term turn (the paper's
        schedule), None = let the width criterion choose (the one-action-space variant).

        `item_s_fn` is a ZERO-ARGUMENT CALLABLE, not an array, and that is the whole reason a key-term
        turn is now cheap: on a scheduled key-term turn the 18,359-item acquisition -- one 200x200
        Cholesky solve against Y^T per interview state -- is never needed, so it is never computed. It
        is still computed, on demand and memoised per state, in the one case where it IS needed: every
        key-term has already been asked, so the turn falls back to an item. Selections are unchanged;
        only the work not done differs."""
        a = np.asarray(asked_u, dtype=np.int64)
        it = a[a < self.n_items]
        kt = a[a >= self.n_items] - self.n_items
        # MUST be a copy: `kt_s` is shared by every user in the state group and is masked per user.
        s_k = np.array(kt_s, dtype=np.float64, copy=True)    # also lifts a float32 width cache
        if len(kt):
            s_k[kt] = NEG
        bk = int(np.argmax(s_k))
        if force_kt is True and s_k[bk] > NEG:
            return self.n_items + bk

        s_i = np.asarray(item_s_fn(), dtype=np.float64).copy()
        if self.pool is not None:
            keep = np.full(s_i.shape, NEG, dtype=s_i.dtype)
            keep[self.pool] = s_i[self.pool]
            s_i = keep
        if len(it):
            s_i[it] = NEG
        bi = int(np.argmax(s_i))
        if force_kt is not None:                             # True with every key-term exhausted, or False
            return bi
        return (self.n_items + bk) if s_k[bk] >= s_i[bi] else bi

    def select(self, turn, users, asked, answered):
        groups = self._group(users, answered)
        self._n_states.append(len(groups))
        out = np.empty(len(users), dtype=np.int64)
        idx_of = {int(u): j for j, u in enumerate(users.tolist())}
        n_kt = 0
        force = self._scheduled_keyterm(turn) if self.turn_rule == "schedule" else None
        for key, us in groups.items():
            mu, _M = self._posterior(key)
            kt_s = self._keyterm_scores(key, mu)
            box = {}                                          # memoises the item acquisition per state

            def item_s_fn(_key=key, _mu=mu, _box=box):
                if "v" not in _box:
                    _box["v"] = self._item_scores(_key, _mu)
                return _box["v"]

            for u in us:
                q = self._pick_mixed(item_s_fn, kt_s, u, asked[u], force)
                out[idx_of[u]] = q
                n_kt += int(q >= self.n_items)
        self._n_kt.append(n_kt)
        return out

    def stats(self):
        s = dict(super().stats())
        s.update({"keyterm_asks_per_turn": self._n_kt, "keyterm_repr": self.stack.get("_repr"),
                  "turn_rule": self.turn_rule, "b_schedule": self.b_schedule,
                  "alpha": self.alpha, "beta": self.beta, "n_concepts": self.C})
        return s


class ConLinUCBMCRSelector(_ConvBanditBase):
    r"""ConLinUCB-MCR (Wang, Liu, Li & Lui, AAAI 2023) -- one-step estimator, MAXIMUM-CONFIDENCE-REGION
    key-term rule.

        theta_t = M_t^-1 b_t,   M_t = sum x x^T + sum xtilde xtilde^T + beta I
        key-term:  k in argmax_k  alpha_t * ||xtilde_k||_{M_t^-1}                    [V, AAAI'23]
        item:      a in argmax_a  <x_a, theta_t> + alpha * ||x_a||_{M_t^-1}
    """
    name = "conlinucb_mcr"
    DEFAULT_TURN_RULE = "schedule"

    def _keyterm_scores(self, key, mu):
        # alpha_t is a positive scalar shared by all k -- it cannot change the argmax among key-terms,
        # but it DOES matter under turn_rule="width", where key-terms and items are compared on the
        # same axis. Using self.alpha there is the honest choice: both halves get the same width
        # coefficient, which is precisely what makes the comparison a comparison.
        return self.alpha * self._width_vec_kt(key)


class ConLinUCBBSSelector(_ConvBanditBase):
    r"""ConLinUCB-BS (same paper) -- key-terms drawn uniformly from a precomputed BARYCENTRIC SPANNER
    of the key-term feature set: a fixed, user-independent exploration basis [V, AAAI'23].

    Because the spanner is fixed and user-independent, THIS ARM DOUBLES AS A STATIC CONCEPT BANK --
    a spanning-optimal one, which is a genuinely different criterion from popularity / entropy /
    polar / expinfo, and is essentially free to compute. `key_term_only=True` runs it as exactly that
    (every turn a key-term, no item questions), which is the static concept arm; the default runs it
    as the paper's algorithm, with the item half handled by ConLinUCB's arm rule.

    The uniform draw is seeded per (seed, turn) so the realised concept order is reproducible; users
    within a turn share the draw, which is what "user-independent" means here.
    """
    name = "conlinucb_bs"
    DEFAULT_TURN_RULE = "schedule"

    def __init__(self, belief, stack, key_term_only=False, spanner_iters=50, **kw):
        super().__init__(belief, stack, **kw)
        self.key_term_only = bool(key_term_only)
        self.spanner_iters = int(spanner_iters)
        self._spanner = None
        if key_term_only:
            self.b_schedule = "always"
            self.turn_rule = "schedule"
            self.name = "conlinucb_bs_static"

    def reset(self, ctx):
        super().reset(ctx)
        if self._spanner is None:
            self._spanner = barycentric_spanner(self.Xt, iters=self.spanner_iters, log=ctx["log"])
        self._turn = 0

    def _keyterm_scores(self, key, mu):
        """Uniform over the spanner: a random permutation gives every spanner member a distinct score
        so the masked argmax realises a uniform draw over the UNASKED spanner members."""
        rng = np.random.default_rng([self.seed, int(self._turn)])
        s = np.full(self.C, NEG)
        s[self._spanner] = rng.permutation(len(self._spanner)).astype(np.float64)
        return s

    def select(self, turn, users, asked, answered):
        self._turn = int(turn)
        return super().select(turn, users, asked, answered)


class CLiMESelector(_ConvBanditBase):
    r"""CLiME (Liu, Li, Dai & Lui, KDD 2025) -- key-terms aligned with the SMALL-EIGENVALUE
    eigendirections of the design matrix, i.e. the directions the posterior knows least about.

        k in argmax_k | xtilde_k^T v_i |,   v_i the smallest-eigenvalue eigenvector(s) of M_t   [V]

    CLiSK's smoothed key-term contexts (`k = argmax_k <xtilde~_k, theta_t>`, xtilde~ = xtilde + Gaussian
    noise) are available via `smooth_sigma > 0`; combined with the eigendirection rule that is CLiSK-ME.
    The smoothing noise is seeded per (seed, turn) so a run is reproducible.

    DEFAULT TURN RULE IS "width", NOT "schedule": CLiME's contribution over ConUCB/ConLinUCB is
    precisely an ADAPTIVE conversation trigger ("ask when preference uncertainty is high") rather than
    a fixed b(t). Our width rule is the closest faithful realisation we can state without inventing a
    constant we could not verify: ask a key-term exactly when the best key-term's posterior width
    exceeds the best item's, both measured in the SAME ||.||_{M^-1} norm. This is a documented
    deviation, not the paper's literal trigger; pass turn_rule="schedule" for the ConUCB schedule.
    """
    name = "clime"
    DEFAULT_TURN_RULE = "width"

    def __init__(self, belief, stack, n_eig=1, smooth_sigma=0.0, **kw):
        super().__init__(belief, stack, **kw)
        self.n_eig = int(n_eig)
        self.smooth_sigma = float(smooth_sigma)
        self._eig = LRUCache(self.max_states, "eig")

    def reset(self, ctx):
        super().reset(ctx)
        self._eig.clear()
        self._turn = 0

    def _small_eigvecs(self, key):
        wkey = self._wkey(key)
        Vv = self._eig.get(wkey)
        if Vv is None:
            _mu, M = self._posterior(key)
            w, V = np.linalg.eigh(M)                                     # ascending eigenvalues
            Vv = self._eig.put(wkey, V[:, :self.n_eig])                  # (f x n_eig), smallest first
        return Vv

    def _keyterm_scores(self, key, mu):
        Vv = self._small_eigvecs(key)
        X = self.Xt
        if self.smooth_sigma > 0:
            rng = np.random.default_rng([self.seed, int(self._turn), 7])
            X = X + self.smooth_sigma * rng.standard_normal(X.shape)     # CLiSK smoothing
        s = np.abs(X @ Vv).max(axis=1)
        if self.turn_rule == "width":
            # Under the shared-axis variant the key-term score has to live on the item index's scale.
            # |xtilde^T v_min| is not a width, so rescale it onto the key-term width range: the RANK
            # order is CLiME's, the magnitude is put on the ||.||_{M^-1} axis it must compete on.
            w = self.alpha * self._width_vec_kt(key)
            order = np.argsort(np.argsort(s))
            s = np.sort(w)[order]
        return s


class ConUCBSelector(_ConvBanditBase):
    r"""ConUCB (Zhang, Xie, Li & Lui, WWW 2020) -- the original TWO-STAGE estimator: an item-level ridge
    fit SHRUNK toward a key-term-level fit.

        theta~_t = Mtil_t^-1 btil_t,   Mtil_t = sum xtilde xtilde^T + lamtil I
        M_t      = lam sum x_a x_a^T + (1-lam) I
        theta_t  = M_t^-1 ( lam sum x_a r_a + (1-lam) theta~_t )                              [V]
        key-term (Eq. 8):
            k = argmax_k ||X_t M_t^-1 Mtil_{t-1}^-1 xtilde_k||^2
                         / ( 1 + xtilde_k^T Mtil_{t-1}^-1 xtilde_k )                          [V]
        item: LinUCB index over (theta_t, M_t)
        conversation frequency: b(t) = 5*floor(log t)                                         [V]

    PROPAGATION (Eq. 4): `E[rtilde_k] = sum_a (w_{a,k}/sum w) E[r_a]` -- the key-term's expected reward
    IS the weight-normalised mean of its members' rewards. That equation is the DEFINITION of this
    arm's key-term feature, so this arm pins `keyterm_repr="uniform_member_mean"` unless the caller
    overrides it deliberately: running ConUCB on a `representative` or `whitened_centroid` key-term
    vector is a hybrid, not ConUCB, and must be labelled as such.

    `||X_t v||^2` with X_t the candidate arms' feature matrix is computed as `v^T (Y_pool^T Y_pool) v`
    -- one 200x200 Gram matrix built once per pool -- rather than by forming an 18,359-vector per
    key-term, which would be the same number 1,030 times over.
    """
    name = "conucb"
    DEFAULT_TURN_RULE = "schedule"

    def __init__(self, belief, stack, lam=0.5, lam_til=1.0, **kw):
        super().__init__(belief, stack, **kw)
        self.lam = float(lam)
        self.lam_til = float(lam_til)
        self._G = None
        self._mtil = LRUCache(self.max_states, "mtil")

    def reset(self, ctx):
        super().reset(ctx)
        self._mtil.clear()
        Ypool = self.Y if self.pool is None else self.Y[self.pool]
        self._G = Ypool.T @ Ypool                                        # X_t^T X_t, f x f
        ctx["log"](f"[{self.name}] lam={self.lam} (NOT the paper's verified value -- neutral midpoint) "
                   f"lam_til={self.lam_til}; Gram over "
                   f"{Ypool.shape[0]} candidate arms")

    def _keyterm_fit(self, key):
        """(theta~, Mtil^-1) from the key-term-level ridge over the ANSWERED key-terms."""
        ckey = key[1]
        got = self._mtil.get(ckey)
        if got is None:
            f = self.Y.shape[1]
            Mt = self.lam_til * np.eye(f)
            bt = np.zeros(f)
            if ckey:
                cid = np.asarray([c for (c, _v) in ckey], dtype=np.int64)
                cv = np.asarray([v for (_c, v) in ckey], dtype=np.float64)
                Xk = self.Xt[cid]
                Mt = Mt + Xk.T @ Xk
                bt = Xk.T @ cv
            Mti = np.linalg.inv(Mt)
            got = self._mtil.put(ckey, (Mti @ bt, Mti))
        return got

    def _posterior(self, key):
        """ConUCB's SHRUNK item-level fit. Overridden so the inherited `_width_vec` (Cholesky solve
        against Y^T) computes ||x_a||_{M_t^-1} against ConUCB's own M_t, not ConLinUCB's.

        As in the base class (HARD RULE 11c) M is rebuilt per call -- it is a <=16x200 Gram plus a
        scaled identity -- and only the mean is cached."""
        f = self.Y.shape[1]
        cols, vals = _obs(key[0])
        theta_til, _Mti = self._keyterm_fit(key)
        M = (1.0 - self.lam) * np.eye(f)
        if len(cols):
            M = M + self.lam * (self.Y[cols].T @ self.Y[cols])
        mu = self._mu.get(key)
        if mu is None:
            b = (1.0 - self.lam) * theta_til
            if len(cols):
                b = b + self.lam * (self.Y[cols].T @ vals)
            mu = self._mu.put(key, np.linalg.solve(M, b))
        return mu, M

    def _keyterm_scores(self, key, mu):
        _theta_til, Mti = self._keyterm_fit(key)
        _m, M = self._posterior(key)
        Vk = np.linalg.solve(M, (Mti @ self.Xt.T))                       # (f x C) = M^-1 Mtil^-1 xtil
        num = np.einsum("ij,ji->i", Vk.T @ self._G, Vk)                  # ||X_t v_k||^2
        den = 1.0 + np.einsum("ij,ji->i", self.Xt, Mti @ self.Xt.T)
        s = num / np.maximum(den, 1e-12)
        if self.turn_rule == "width":
            w = self.alpha * self._width_vec_kt(key)
            s = np.sort(w)[np.argsort(np.argsort(s))]                    # see CLiME's note
        return s

    def stats(self):
        s = dict(super().stats())
        s.update({"lam": self.lam, "lam_til": self.lam_til})
        return s


# ===================================================================== concept-aware consumer
def make_concept_aware_predict(enc, Wd, bd, L_csr, conc_answered, net, batch=256):
    """A predict_fn(X_csr)->scores for `metrics.evaluate` that folds the ITEM answers through the
    certified tower AND the CONCEPT answers through the certified concept channel.

    WHY THIS IS NEEDED AND WHAT IT DELIBERATELY DOES NOT TOUCH. The harness's scoring path
    (`run_interview_table.score_arm`) is item-only: a concept question would burn a turn and reach no
    consumer, so a conversational arm would be measured against a consumer that structurally cannot
    hear half of its action space. That is not a fair test of the family, it is a test of our harness.
    `score_arm` itself is NOT modified -- for any consumer other than "ours" it simply calls
    `obj(X)`, so this returns an object that fits that existing contract exactly, with a cursor over
    rows in the order `metrics.evaluate` walks them (the same discipline `cursor_predict` uses).

    `net` is a `concept_fold.ConceptFoldNet` -- the CERTIFIED concept channel (models/concept_cv3_ep08.pt
    at time of writing). `conc_answered[u]` is the list of (concept, value) the user could answer, plus
    the refusal flags for the ones they could not: a refusal is folded as a REFUSAL TOKEN (v=0, ref=1),
    which is exactly how `concept_cold_check.cold_interviews` folds a burned concept question, so a
    burned turn is visible to the net rather than invisible.
    """
    import torch
    from concept_fold import fold_items_frozen
    state = {"pos": 0}

    def predict(X_csr):
        pos = state["pos"]
        rows = X_csr.shape[0]
        Ls = L_csr[pos:pos + rows]
        state["pos"] = pos + rows
        out = np.zeros((rows, Wd.shape[0]), dtype=np.float32)
        with torch.no_grad():
            for st in range(0, rows, batch):
                idx = list(range(st, min(st + batch, rows)))
                seqs = []
                for r in idx:
                    s, e = Ls.indptr[r], Ls.indptr[r + 1]
                    seqs.append((Ls.indices[s:e].astype(np.int64),
                                 (Ls.data[s:e] - 1.0).astype(np.int64)))
                Zi = fold_items_frozen(enc, seqs, batch=len(seqs))
                cids, cvals, crefs = [], [], []
                for r in idx:
                    toks = conc_answered[pos + r]
                    cids.append([int(c) for (c, _v, _f) in toks])
                    cvals.append([float(v) for (_c, v, _f) in toks])
                    crefs.append([float(f) for (_c, _v, f) in toks])
                z = net(Zi, cids, cvals, crefs) if any(cids) else Zi
                out[idx] = (z @ Wd.T + bd).numpy().astype(np.float32)
        return out
    predict.reset = lambda: state.update(pos=0)
    return predict


# ===================================================================== smoke
def _smoke():
    """SYNTHETIC code-path check. Not the canonical data, not a number, not citable."""
    import belief_mf
    print("[conv_bandit][SMOKE] synthetic basis + synthetic concept bank (code path only)")
    rng = np.random.RandomState(0)
    ni, f, C, n_users = 400, 16, 25, 12
    Y = rng.randn(ni, f) * 0.3
    U = rng.randn(300, f) * 0.5
    bel = belief_mf.GaussianBelief(Y, U.mean(0), np.cov(U, rowvar=False), 1.0, 1.0, 1e-6)
    member_sets = {c: np.sort(rng.choice(ni, size=rng.randint(30, 60), replace=False)) for c in range(C)}
    cnt = rng.rand(ni) * 100
    stacks = {}
    for mode in KEYTERM_MODES:
        Xt = keyterm_features(Y, member_sets, cnt, mode=mode, n_rep=8)
        assert Xt.shape == (C, f) and np.isfinite(Xt).all()
        stacks[mode] = {"n_concepts": C, "member_sets": member_sets, "Xtilde": Xt, "_repr": mode,
                        "V": (rng.rand(n_users, C) * 2 - 1),
                        "answerable": rng.rand(n_users, C) > 0.4}
    # norm_match really removes magnitude
    for mode in KEYTERM_MODES:
        nr = np.linalg.norm(stacks[mode]["Xtilde"], axis=1)
        ref = np.asarray([np.linalg.norm(Y[member_sets[c]], axis=1).mean() for c in range(C)])
        assert np.allclose(nr, ref, rtol=1e-6), f"norm_match failed for {mode}"
    print(f"[SMOKE] key-term features PASS for all {len(KEYTERM_MODES)} representations "
          f"(norm-matched to the mean member ||y_a||)")
    B = barycentric_spanner(stacks["uniform_member_mean"]["Xtilde"], iters=5)
    assert len(B) == min(f, C) and len(set(B.tolist())) == len(B), "spanner must be a distinct basis"
    print(f"[SMOKE] barycentric spanner PASS ({len(B)} distinct key-terms, full rank)")

    # a tiny end-to-end drive of every arm through the real Selector contract
    stack = stacks["whitened_centroid"]
    conc = {"asked": [[] for _ in range(n_users)], "answered": [[] for _ in range(n_users)]}
    ctx = {"n_users": n_users, "n_items": ni, "q_max": 6, "log": print, "conc_state": conc,
           "cand_pool": None}
    arms = {"conlinucb_mcr": ConLinUCBMCRSelector(bel, stack, turn_rule="width"),
            "conlinucb_bs": ConLinUCBBSSelector(bel, stack, spanner_iters=3),
            "conlinucb_bs_static": ConLinUCBBSSelector(bel, stack, key_term_only=True,
                                                       spanner_iters=3),
            "clime": CLiMESelector(bel, stack, smooth_sigma=0.01),
            "conucb": ConUCBSelector(bel, stack)}
    lvl = [{int(i): int(rng.randint(0, 10)) for i in rng.choice(ni, 40, replace=False)}
           for _ in range(n_users)]
    for nm, sel in arms.items():
        c2 = {"asked": [[] for _ in range(n_users)], "answered": [[] for _ in range(n_users)]}
        ctx["conc_state"] = c2
        sel.reset(ctx)
        asked = [[] for _ in range(n_users)]
        answered = [[] for _ in range(n_users)]
        n_kt = 0
        for t in range(6):
            q = sel.select(t, np.arange(n_users), asked, answered)
            assert q.shape == (n_users,)
            for u, a in zip(range(n_users), q.tolist()):
                assert 0 <= a < ni + C, f"{nm} out of range {a}"
                assert a not in asked[u], f"{nm} re-asked action {a}"
                asked[u].append(int(a))
                if a < ni:
                    if a in lvl[u]:
                        answered[u].append((int(a), lvl[u][a]))
                else:
                    c = a - ni
                    n_kt += 1
                    c2["asked"][u].append(c)
                    if bool(stack["answerable"][u, c]):
                        c2["answered"][u].append((c, float(stack["V"][u, c])))
        st = sel.stats()
        print(f"[SMOKE] {nm:22s} PASS  key-term asks/turn={st['keyterm_asks_per_turn']} "
              f"distinct states/turn={st['distinct_states_per_turn']} total_kt={n_kt}")
    assert sum(arms["conlinucb_bs_static"].stats()["keyterm_asks_per_turn"]) == n_users * 6, \
        "key_term_only arm must ask ONLY key-terms"
    print("[conv_bandit][SMOKE] COMPLETE")


if __name__ == "__main__":
    if "--smoke" in sys.argv:
        _smoke()
    else:
        print(__doc__)
