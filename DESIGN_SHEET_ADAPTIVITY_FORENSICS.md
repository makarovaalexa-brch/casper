> # ⚠ SUPERSEDED — DO NOT EXECUTE §2a / §4c FROM THIS FILE
> **This sheet's central thesis (the adaptive prize lives in the ANSWERABILITY channel and we deleted it)
> was REFUTED on 2026-07-14 by our own prior results.** See **`WHY_ADAPTIVITY_DIED.md` (v2)** for the
> refutation and the surviving explanation (adaptivity <= H(Theta); a strong collaborative recommender
> leaves ~no model uncertainty).
> - **E0** already tested answerability routing WITH THE PRIVILEGED TRUE TABLE => **-0.0003 TIE**.
> - **REFUSAL_RESULT.md** already built two of the three proposed "wires" (2026-06-29) => **TIE tail, LOSE full**.
> - The "94% answerable" is an **equilibrium of good asking over an OPEN pool**, NOT an engineered filter;
>   the code "smoking gun" cited below is in a **Paper-D literature-baseline block**, not the policy path.
> - **The §4c p_unknown sweep (E3) is INCOHERENT** (high experienced refusal and good refusal prediction are
>   mutually exclusive at the optimum; and a fixed-sequence comparator guarantees the gap by wasted-turn
>   arithmetic = the P4C artifact, 2nd edition). **DO NOT RUN IT.**
>
> **STILL VALID from this file:** the Golbandi-tree baseline (§2, §2b) — but run it as **E4-DUAL** (two rulers +
> unknown-branch ablation) per `WHY_ADAPTIVITY_DIED.md` §4, which turns it into the DISCRIMINATOR between the
> two theses. The HARD RULE #1 flag on Golbandi's popular-only splitter reduction (§2b) also stands.
> **Live run order: `DESIGN_SHEET_E2_SNAPK.md` (SNAP-K, then E2), then E4-DUAL, then H(Theta)-lite.**

# DESIGN SHEET — WHY DID ADAPTIVITY DIE? (Golbandi baseline + the mixture belief)  2026-07-14
# [SUPERSEDED 2026-07-14 — see banner above and WHY_ADAPTIVITY_DIED.md v2]
Status: DESIGN. Golbandi baseline APPROVED by author ("do run golbandi as a baseline, by all means").
Runs queued behind pb2/pb3 — NOT concurrent (CPU contention corrupted the 2026-07-13 timings).

## 0. THE QUESTION WE HAVE NEVER ANSWERED
Eight learned policies TIED static entropy. Realizable adaptive prize = -0.0003. Yet we ourselves
REPLICATED Golbandi's adaptive decision-tree cold-start win, and the field reports adaptive wins routinely.
**We have never established why.** Two rival explanations, never separated:

- **E-WEAK-ACTOR**: adaptive value EXISTS in our arena; our policies were too weak to capture it.
- **E-NO-FUEL**: adaptive value DOES NOT EXIST in our arena; our belief/arena removed it by construction,
  and the -0.0003 is a correct measurement of a world that assumed answerability away.

These have OPPOSITE implications. E-WEAK-ACTOR => fix the POLICY (DAD/pointer/transformer).
E-NO-FUEL => fix the BELIEF (heteroscedastic / mixture). We must not guess.

## 1. THE THEORY (the suspected root cause)
**Non-adaptivity theorem (to be stated exactly + cited by the forensics agent).** For a LINEAR-GAUSSIAN
model, the posterior covariance `Sigma` depends only on WHICH designs were chosen, NOT on the ANSWERS.
=> the optimal design sequence is precomputable => ADAPTIVITY IS WORTH EXACTLY ZERO.
Our belief = single unimodal Gaussian over z + linear measurements => **our eight failures may be ONE
THEOREM measured eight times.** This is the leading hypothesis and it must be tested, not assumed.

**What breaks the theorem** (each is a candidate mechanism, and each is testable):
1. **Answer-dependent precision (heteroscedasticity)** — `lambda_t = f(answer)`: a `no_clue`/refusal yields
   ~zero precision about taste. Then `Sigma` DEPENDS ON THE ANSWER. This is ANSWERABILITY.
2. **Mixture / latent-class belief** — answers reweight COMPONENTS; the posterior SHAPE becomes
   answer-dependent. This is COHORT IDENTIFICATION (what Golbandi's tree nodes are).
3. Nonlinear likelihood (our decoder IS a softmax — the Gaussian belief is an approximation).

## 2. THE DECISIVE EXPERIMENT — GOLBANDI ON OUR OWN ARENA (approved)
Build Golbandi's adaptive tree ON OUR data, OUR ruler, OUR realistic answerer. This SEPARATES the two
explanations, which no amount of literature can do:
- **Tree BEATS static entropy on our arena** => adaptive fuel EXISTS here => E-WEAK-ACTOR. Our policies
  were weak. Fix the actor.
- **Tree ALSO TIES** => no fuel => E-NO-FUEL. The arena/belief killed adaptivity, NOT our optimizers.
  The E0 no-go gets STRONGER and the fix is the BELIEF.
Either result is publishable and either kills a year of ambiguity. **Pre-registered: we report whichever.**

### 2a. THE MECHANISM ABLATION (the crux)
Golbandi splits users THREE ways: **loved / hated / UNKNOWN**. The `unknown` branch IS an answerability
channel — the next question depends on whether the user COULD ANSWER AT ALL.
**Run the tree WITH and WITHOUT the `unknown` branch** (2-way vs 3-way split).
- Adaptive gap COLLAPSES without it => **ANSWERABILITY IS THE SOURCE OF ADAPTIVE VALUE**, demonstrated on
  our own data. That converts our biggest null result into a MECHANISM.
- Gap survives => the value is cohort-identification (the partition), not answerability. Also decisive.

### 2b. SCALE (the author asked: can we even build a tree this big?)
YES. Golbandi built these at NETFLIX scale (~480k users / ~17.7k items — forensics agent to VERIFY),
LARGER than our ML-25M cut (161,112 train users / 18,430 items). Cost is driven by NONZEROS, not
users x items: per-node split statistics are sparse products (sum/count of member profiles per candidate
splitter), computed per level.
**HARD RULE #1 FLAG**: Golbandi restricts candidate SPLITTER items to the most POPULAR ones. That is a
TOP-N REDUCTION. We do NOT do this silently. Default = ALL items, via CHUNKED sparse products (non-lossy,
slower). The popular-only variant may be run ONLY as an explicit, author-signed ABLATION — where it is
independently interesting, since restricting to popular splitters is itself an ANSWERABILITY PRIOR.

## 3. THE MODEL THE AUTHOR ASKED FOR: NON-GAUSSIAN **AND** CONTINUOUS-COMPATIBLE
The tree's POWER = cohort identification. The tree's LIMIT = discrete (splits on catalog items; can never
emit a continuous query, so it is useless for Paper C). We want both. The answer:

### CONJUGATE MIXTURE-OF-GAUSSIANS BELIEF ("Golbandi's cohorts, made soft and continuous")
`p(z) = sum_k pi_k N(z; m_k, S_k)`. Measurement `y = e^T z + noise` (precision `lambda`, possibly
answer-dependent). Then the posterior is **AGAIN A MIXTURE WITH THE SAME K** (conjugate, closed form):
```
component update:  (m_k, S_k) -> Kalman update              [exact]
WEIGHT UPDATE:     pi_k' ∝ pi_k * N(y; e^T m_k, e^T S_k e + 1/lambda)      <-- DEPENDS ON THE ANSWER
```
**That weight line is the escape from the theorem.** The posterior SHAPE is answer-dependent => the best
next question is answer-dependent => ADAPTIVITY HAS VALUE, for a stated reason, not a hope.

Why it is the right object for US:
- **ADAPTIVE**: answer-dependent component weights (mechanism 2 above); compose with answer-dependent
  `lambda` (mechanism 1) and we have BOTH escapes at once — exactly what Golbandi has.
- **COARSE-TO-FINE, EMERGENT**: the most informative direction is the principal axis of the (mixture)
  covariance. Early = broad/genre-scale; as it shrinks, the remaining uncertain directions are fine.
  Coarse-to-fine is a CONSEQUENCE of D-optimal design on a shrinking ellipsoid, not a curriculum.
- **CONTINUOUS-SNAPPED**: the query is a DIRECTION `e` in the latent space — any unit vector, then SNAPPED
  to the askable bank. Same space as items/concepts. Paper C's continuous actor works unchanged.
- **EIG IS AN EXACT FINITE SUM**: our answer space is TINY and DISCRETE (knowledge x value + refusal ~ 13
  outcomes). So `EIG(e) = sum_outcomes P(o|belief) * [H(prior) - H(posterior|o)]` is computed by EXACT
  ENUMERATION — no Monte Carlo, no nested estimator.
- **A CONTINUOUS ACTOR WITH NO RL**: EIG(e) is DIFFERENTIABLE in `e` => gradient-ascend the query direction.
  After eight failed RL policies, a closed-form objective is not a small thing.
- **Components INIT from cohorts** (cluster user z's from full profiles) = Golbandi's node profiles, soft.

**Cost**: K components x d=512. Full covariances are heavy, but interviews are SHORT (n <= 20), so keep
`S_k = diagonal + rank-n` (Woodbury) — cheap exactly where the belief matters.

**RISKS / what would kill it**: (a) the linear-Gaussian measurement model is an approximation of a softmax
decoder — the belief is a MODEL, not the truth; its value is empirical. (b) K is a knob; it must be chosen
on a val split, not tuned on test. (c) If the mixture collapses to one effective component (all mass on one
k), adaptivity dies again — MEASURE effective K (perplexity of pi) per turn and report it. (d) A mixture
could beat the single Gaussian purely by being a better RECOMMENDER, with no adaptivity gain at all — so
the adaptive gap must be measured AGAINST A STATIC POLICY ON THE SAME MIXTURE BELIEF (tie-by-construction).

## 4. GATES (pre-registered)
- **G-tree**: Golbandi tree vs static entropy on OUR arena/answerer/ruler. Report the sign HONESTLY.
- **G-unknown**: 3-way (loved/hated/unknown) vs 2-way tree. Does the adaptive gap need answerability?
- **G-mixture-adaptive**: adaptive EIG vs STATIC schedule ON THE SAME MIXTURE BELIEF (symmetric baseline —
  no comparator asymmetry; this is the trap that produced the P4C artifact).
- **G-effective-K**: does the posterior actually use >1 component? (If not, the mechanism is absent.)
- **G-strength**: the mixture belief must not COST full-profile NDCG (>= the single-Gaussian model).
- **G-snap**: snap loss (unsnapped continuous EIG-optimal direction vs snapped-to-bank).

## 4b. FINDINGS FROM THE 2026-07-14 LIT SWEEP (5 agents) — THE ANSWER TO "WHY"

**THE ANSWERER IS NOT THE PROBLEM. (A claim I got WRONG and retract.)** I first concluded from three
diagnostics that our answerer models knowledge as "popularity + a scalar user-competence term + a coin flip,
with no manifold structure". **ALL THREE TESTS WERE BROKEN** — (1) a variance decomposition of a BINARY
variable has ZERO power by construction (`Var_u(R_uq) = p_q(1-p_q)` identically, structure or not);
(2) the k-means collapsed to ONE cluster and the CONTROL FAILED (taste showed no structure, which is
impossible); (3) the taste/knowledge correlation conditioned on ANSWERED cells, where "answered" is the
constant 1. READ THE CODE INSTEAD (`dans_build.py:396-410`): the knowledge model conditions on
`c_align = tag_genre @ taste_u` and `genre_align` (**knowledge IS coupled to taste**), on `coprox`/`comax`/
`concept_raw=rmc` (**co-knowledge MANIFOLD structure**), and on per-user buffness traits. The fuel EXISTS.

**SO WHY IS THE PRIZE ZERO? Four suppressors, none of them the answerer:**
1. **THE POLICY DODGES.** Our arena self-selects to **94% answerable** (p_unknown ~ 0.06). Golbandi's tree
   runs at **67% unknown** (Karimi et al., UMUAI 2014, Table 1 — verified). We closed the channel 10x.
2. **ANSWERABILITY NEVER ENTERS THE PRECISION.** `Sigma_t^-1 = Sigma_{t-1}^-1 + 1[answered_t] * lambda q q^T`
   — that indicator is a RANDOM, OBSERVED outcome; it is what makes Sigma answer-dependent. We drop it.
3. **OUR OBJECTIVE IS AN INFORMATION SURROGATE, NOT THE TASK LOSS.**
4. **WE MEASURE IN THE CAPPED REGIME** (fixed-budget NDCG@8) instead of questions-to-target.

**THE THEOREM (cite properly).** Krause & Guestrin, *Nonmyopic Active Learning of Gaussian Processes*, ICML
2007, §4 + **Theorem 1**: for any objective that is a functional of the posterior covariance alone, closed-loop
= open-loop, and **the value of adaptivity is upper-bounded by H(Theta), the entropy of MODEL UNCERTAINTY.**
Know your model => adaptivity is worth zero. (Also Krause/Singh/Guestrin JMLR 2008 §3.1; Rainforth et al.,
*Modern Bayesian Experimental Design*, Statistical Science 2024, §2.3.)

**WORDING CORRECTION (would be caught in review).** "Adaptivity has ZERO value" is exactly true ONLY when the
objective is a functional of Sigma alone. Ours is NDCG — NONLINEAR IN THE POSTERIOR MEAN — so the
decision-theoretic prize is a SECOND-ORDER curvature term. **-0.0003 is exactly what a second-order term looks
like.** Correct claim: *the information-theoretic prize is provably zero; the decision-theoretic prize is
second-order and measured indistinguishable from zero.*

**GOLBANDI'S SMOKING GUN (verified in the WSDM'11 text).** His Figure 2 prints the expected final RMSE BY
BRANCH: like **0.9393** / dislike **0.9522** / unknown **0.9837**. **The residual uncertainty DEPENDS ON THE
ANSWER** — mathematically impossible under the assumptions our arena satisfies. He breaks all six; we satisfy
all six. His blending weights: w_like=5, w_dislike=1, **w_unknown=0.02** — the unknown answer carries ~1/250
the predictive weight yet drives ~2/3 of the ROUTING. **He spends his adaptivity re-planning around what the
user does NOT know.**

**⭐ THE CITATION WE DO NOT HAVE AND MUST:** Sepliarskaia, Kiseleva, Radlinski & de Rijke, **"Preference
Elicitation as an Optimization Problem", RecSys 2018** — a **STATIC questionnaire BEATS adaptive decision
trees** (3x shorter), and their own explanation is ours: *"PWDT optimizes a function that is different from the
loss function, namely weighted generalized variance."* **The field already published our result.** We can be
the first to say WHY. (Corroborating: Rashid/Karypis/Riedl, SIGKDD Explorations 2008 — IGCN, a cohort-based
ADAPTIVE method, TIES static Entropy0 OFFLINE and only wins ONLINE with real users. Cohorts are necessary,
NOT sufficient.)

**THE METRIC REGIME (M4) — we may have measured in the one regime where theory caps the prize.**
Golovin & Krause, JAIR 2011, §11: min-cost COVERAGE (questions-to-target) has an adaptivity gap of
**Theta(n/log n) — UNBOUNDED**; fixed-budget MAXIMIZATION has a gap of **e/(e-1) ~ 1.58 — a small constant**
(Asadpour et al., WINE 2008; verify the constant before publishing). **Golbandi's headline IS a
questions-to-target claim ("6 evaluations vs over 20").** Ours is fixed-budget NDCG@8.
=> **FREE EXPERIMENT, ZERO TRAINING: re-plot our EXISTING runs as questions-to-reach-target-NDCG.**

## 4c. THE ORDER PARAMETER + THE PRE-REGISTERED FALSIFIER
**Order parameter:** `Var_y[ log det Sigma_T(y) ]` — how much the ACHIEVED posterior precision varies with what
the user actually SAID. It is **EXACTLY 0** in our current arena. Claim to pre-register: adaptive value is
monotone increasing in this quantity, and ~0 when it is 0.

**THE SWEEP.** Hold recommender, answer model, budget (q=8), test users and seeds FIXED. Vary ONE knob:
`p_unknown in {0.00, 0.10, 0.25, 0.40, 0.55, 0.70}` (Golbandi ~0.67), **ENFORCED BY CONSTRUCTION OF THE
CANDIDATE POOL so the policy CANNOT DODGE by asking only head items — dodging is exactly what killed E0.**
At each level report: (a) the **static-optimal schedule RE-OPTIMIZED at that p_unknown** [BASELINE SYMMETRY —
omit this and we reproduce the R2 unfair-comparator artifact], (b) the best realizable adaptive policy,
(c) the order parameter.
**PREDICTION:** gap ~0 at p=0, growing with p, tracking Var_y[log det Sigma_T].
**FALSIFIED IF:** the gap stays ~0 at p_unknown=0.67 against a re-optimized static. That kills the
answerability hypothesis outright and hands the explanation to the OBJECTIVE (M3) and the METRIC REGIME (M4) —
and our contribution becomes the first clean MEASUREMENT of a small prize, which is still a good paper.

**Near-free H1 falsifier:** swap the Gaussian prior for a K=32 GMM, change NOTHING else. Predict greedy-EIG
under the mixture beats the static entropy schedule. If it does NOT, the mixture hypothesis is dead too.

## 5. ORDER OF WORK
1. pb2 converge -> pb3 (belief) compare. [running]
2. Golbandi tree (all-items splitters, chunked) + the `unknown` ablation. [approved, queued]
3. Mixture belief on top of the set-encoder; G-mixture-adaptive with a SYMMETRIC static baseline.
4. Only then: a learned actor (DAD-style amortized design) — and only if there is fuel to capture.
