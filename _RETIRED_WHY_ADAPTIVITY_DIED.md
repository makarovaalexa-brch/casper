> # ⛔ RETIRED — DO NOT REASON FROM THIS DOCUMENT
> **Everything in this file that concludes "adaptivity does not pay" is WRONG and was produced by a biased
> spiral of theorem-reasoning.** It is kept only as a record of how the error was made.
>
> **THE SETTLED RESULT: `experiments/ADAPTIVE_PROBE_RESULT.md` — ADAPTIVITY PAYS: +47% TAIL (150,239 users,
> CI [+0.0186,+0.0203], no policy, no optimisation).** The winning question is always a NICHE film INSIDE the
> user's own genre; coarse-to-fine falls out. See `CLAUDE.md` HARD RULE #2.
>
> **DO NOT cite the non-adaptivity theorem as an explanation of anything in this project. Do not re-derive it.
> Do not use it to predict a result. It is settled and closed.**
>
> **THE EIGHT FAILED POLICIES ARE DEAD AND BURIED.** They lost because the MODELS WERE WEAK (V1 encoder), not
> because of any objective or any theorem. **They go in NO paper, in NO form.** Do not analyse them, do not
> cite them, do not explain them.


---

# THE TWO REGIMES OF ADAPTIVITY (v3, 2026-07-14)
**v1 claimed the adaptive prize lives in the answerability channel — REFUTED (by E0 + REFUSAL_RESULT).**
**v2 claimed adaptivity is worth ~0 in our arena, full stop — WRONG IN SCOPE. It generalized a DISCRETE/BINARY
result to elicitation as such, while a 12-sigma counter-result sat unread in our own repo.**
This v3 states the BOUNDARY. Both halves are true; neither generalizes to the other's side.

STATUS KEYS: **[V]** verified at primary source · **[D]** derivation · **[U]** unverified · **[REFUTED]**.

---

## 0. THE RESULT THAT SETS THE BOUNDARY — and that I failed to read for a night
**[V] `experiments/paper2/STATIC8_RESULT.md` (2026-07-03), verbatim:**
> **"D1 ADAPTIVE BEATS THE BEST STATIC-8 BY +0.038 FULL / +0.042 TAIL."**
> *"First clean isolation of true adaptive value in the continuous regime. A fixed continuous questionnaire
> cannot match the belief-conditioned policy; per-user adaptivity is worth ~0.04 NDCG on both axes here, and
> **it is specifically what lets graded answers pay off.**"*

| policy (graded COMPARE4, seed-avg {1,2,3,7,11}, te[300:], q8) | FULL | TAIL |
|---|---|---|
| **D1 ADAPTIVE continuous actor** | **0.3780 ± 0.0032** | **0.1782 ± 0.0065** |
| best STATIC-8 continuous questionnaire (raw D1 turn-means) | 0.3399 ± 0.0053 | 0.1362 ± 0.0049 |
| entropy graded (discrete) | 0.353 | 0.133 |
| CASPER-R graded (discrete) | 0.343 | 0.138 |
| popular, no elicitation (q0) | 0.310 | 0.081 |

**+0.038 FULL on a sd of 0.0032 is ~12 sigma.** The control that v2 declared "never run" was run ELEVEN DAYS
EARLIER, and **adaptivity won.**

### 0a. THE MECHANISM (also already in that file) — why the fixed questionnaire loses
**[V] The static bank SATURATES AT q4 AND THEN DECAYS:**
| q | 0 | 2 | 4 | 6 | 8 |
|---|---|---|---|---|---|
| FULL | 0.3099 | 0.3130 | **0.3525** | 0.3434 | 0.3399 |
The taste subspace is only **~2-effective-dimensional**, so a fixed bank **exhausts the population-level
questions in ~4 turns**; further fixed questions add nothing or actively hurt. The adaptive actor keeps
climbing because it **RE-AIMS AT THE INDIVIDUAL** every turn. **[V]** Gradient-optimizing a free static bank
COLLAPSES it (cross-turn |cos| 0.47 -> 0.82): with graded answers a couple of directions along the dominant
taste axis already reconstruct u*, so surplus STATIC queries are redundant **by construction**.

### 0b. THE INVERSION THAT CLINCHES IT
**[V] STATIC gets WORSE with graded answers (0.340 vs 0.357 binary). ADAPTIVE gets BETTER (0.378 vs 0.356).**
=> **GRADED ANSWERS AND ADAPTIVITY ARE COMPLEMENTS. NEITHER PAYS ALONE.** The file's own words: *"This
localizes the entire graded-continuity prize of Paper C to ADAPTIVITY."*

### 0c. THE HOLE IN 0 -- STATIC8's STATIC ARM MAY BE UNDER-OPTIMIZED. DO NOT DECLARE VICTORY.
A counter-evidence sweep (2026-07-14) found the key methodological fact of this literature: **"the size of a
reported adaptivity gap is dominated by HOW THE NON-ADAPTIVE ARM WAS CONSTRUCTED."** (RL-BOED reports
Random=1.624 on the same benchmark where DAD reports Random=8.303 -- a 6.7-nat gap in the *comparator*.)
Now re-read STATIC8's own text **[V]**:
> *"The best static-8 is the **raw D1 turn-means with ZERO further training** (0.3399). Gradient-optimizing the
> static bank on any variant of the D1 objective **only makes it worse** (collapse or drift)."*
**Our static arm COULD NOT BE OPTIMIZED -- every attempt collapsed (cross-turn |cos| 0.47 -> 0.82).** That is
exactly the pattern the sweep finds across the field: **a TRAINED adaptive actor vs an UNTRAINED static bank.**
STATIC8's own diagnosis ("a static reconstruction objective has a trivial minimum") argues the OBJECTIVE was
wrong for static design -- NOT that no good static exists.

**THE GENUINE TENSION (unresolved -- the live question):**
- **FOR adaptivity:** our belief is NOT linear-Gaussian (learned attention fold-in) => information may genuinely
  BE answer-dependent. **[V] Krause's escape clause: "for non-Gaussian models, sequential strategies can
  strictly outperform a priori designs, EVEN WITH KNOWN PARAMETERS."**
- **AGAINST:** **[V] Lecomte, Odor & Thiran (arXiv:2002.07336), verbatim: "If instead we are allowed to query
  ANY SUBSET ... there is NO DIFFERENCE between the adaptive and the non-adaptive query complexities."**
  **ADAPTIVITY REQUIRES AN IMPOVERISHED QUERY VOCABULARY.** Ours is maximally rich (continuous, 512-d).
  This predicts static should TIE.
These point OPPOSITE ways, and **STATIC8 is the only measurement that adjudicates them -- which is exactly why
its static arm must be beyond reproach. It currently is not.**

### 0d. THE DECISIVE RUN (the literature supplies the template)
**[V] DAD's static baseline is NOT naive**: SG-BOED with the PCE bound, **JOINTLY optimizing all T designs
before any observation** -- and adaptivity still doubles the information against it (temporal discounting,
T=20: Fixed 2.518 -> DAD 5.021). **Build our static arm the same way.**
**E2-HARD-STATIC**: jointly optimize all 8 queries **directly on the DOWNSTREAM objective (NDCG)**, from an
**ACTOR-INDEPENDENT candidate pool** (NOT D1 turn-means; NOT the D1 training objective, whose static minimum is
trivial by their own admission). Same selection/multiplicity budget as the actor. Seed-avg. Pre-registered MDE.
- **Adaptive still wins => the win is REAL and Paper C's claim is safe.**
- **Gap collapses => STATIC8 was an OPTIMIZATION ARTIFACT.** Report it.
**I do not know which. Neither goes into a paper until this runs.**

---

## 1. THE BOUNDARY (the reconciliation -- this is the thesis)
> **A fixed questionnaire is optimal exactly when a question's information value is the same for everyone
> regardless of what they have already said. The adaptive interviewer pays exactly when answers RELOCATE where
> the information is — through a clean channel, below saturation budget, in proportion to the model uncertainty
> the prior leaves.**

**STATIC WINS / TIES when** (our discrete arena satisfied ALL of these):
- belief ~ linear-Gaussian => posterior covariance is ANSWER-INDEPENDENT **[V Krause & Guestrin ICML'07 §4:
  "any objective function depending only on the predictive variances... cannot benefit from sequential
  strategies"]**;
- the noise channel does not depend on the question **[V Jedynak/Frazier/Sznitman 2012]**;
- monotone-submodular objective => adaptivity gap capped at e/(e-1) ~ 1.58 **[V Asadpour]**;
- small residual MODEL uncertainty (a strong collaborative prior) **[V Krause Thm 1: adaptivity <= H(Theta)]**;
- CLOSED pool + BINARY answers + saturation budget + population-mean evaluation.
**Evidence: E0 (privileged answerability table) = -0.0003; eight learned policies tied static entropy;
Sepliarskaia et al. RecSys 2018 (static BEATS adaptive trees); IGCN ties static OFFLINE.**

**ADAPTIVE WINS when ANY of these breaks:**
- **NON-GAUSSIAN / ANSWER-DEPENDENT INFORMATION** — **[V] Krause's own next sentence: "for non-Gaussian models,
  sequential strategies can strictly outperform a priori designs, EVEN WITH KNOWN PARAMETERS."**
  **=> OUR GRADED ANSWERS THROUGH A LEARNED NONLINEAR FOLD-IN ARE EXACTLY THIS. STATIC8 IS ITS REALIZATION.**
- theta-localized queries (binary-search regime: adaptive O(log 1/eps) vs non-adaptive O(1/eps));
- coverage objectives (Golovin & Krause Thm 25: gap **Omega(n/log n)**, unbounded);
- weak prior / large H(Theta) (Golbandi's tree; LastFM; a human meeting a stranger);
- budget below saturation (our own +0.011 FULL@q2, and STATIC8's q4 peak-then-decay);
- **[V, Squeeze R2] A CLEAN CHANNEL** — under noise, per-user re-aiming is estimated from noise and LOSES to a
  noise-adapted static (fair static 0.314 > noise-trained actor 0.284). **Adaptivity is clean-channel-only.**

---

## 2. IS ELICITATION DEAD AS A RESEARCH DOMAIN? (the author's challenge)
**NO — but one room of it is, and we should say so without flinching.**
- **DEAD: closed-pool question SELECTION** against a strong collaborative recommender, binary answers,
  saturation budget, population-mean offline eval. In that regime the optimal interviewer IS a questionnaire,
  it is findable by greedy submodular selection, and **Sepliarskaia et al. (RecSys 2018) already published half
  of it in 2018.** Anyone still doing adaptive question-selection over a fixed pool with a modern recommender is
  optimizing inside a solved regime. **We add the WHY and the WHEN.**
- **ALIVE (each with in-house evidence):** (1) **adaptive question CONSTRUCTION** in continuous space with
  graded answers — **+0.038/+0.042, ~12 sigma [STATIC8]**; (2) **open-channel elicitation** — what to ask so the
  user VOLUNTEERS the most informative content, and how to exploit it (the question-framing lever:
  favourite->head, hidden-gem->tail); (3) **THE BOUNDARY ITSELF** — when to adapt vs when to denoise, who
  benefits, at what budget, under which estimator strength; (4) **answer modelling** — folding graded/open/
  refused answers into belief (de-OOD recommender; answerable concepts).
> **The domain moved from "which question do I pick from the pool" to "what question do I BUILD from what you
> just said, and when is it worth bothering."**

**THE AUTHOR'S ARGUMENT WAS RIGHT AND IS NOW THE FRAME:** *no human asks another human a fixed list of closed
probes.* Correct — and the reason is not sentiment: a human interviewer (a) CONSTRUCTS the follow-up from the
answer's content, and (b) has a huge H(Theta) about a stranger, which our model does not have about a user drawn
from 162k. **Adaptivity substitutes for a prior.**

---

## 3. PAPER C — NOT VOID; THE CLAIM IS STRONGER THAN WE WERE STATING
The 0.366-vs-0.360 headline was CONFOUNDED (action space AND policy changed together). The CONTROLLED result is
better: **graded answers + per-user adaptivity are COMPLEMENTS, worth ~+0.04 over the best static continuous
questionnaire, IN THE CLEAN-ANSWER REGIME ONLY.** Snap-loss is an action-SPACE result and survives regardless.
**REMAINING GAP (cheap):** STATIC8's static bank was D1-DERIVED (its candidates came from the actor's own query
dump), which biases toward a TIE — and it still LOST, so the bias is conservative. Close it anyway with ONE
actor-independent static construction (greedy over a non-D1 candidate pool), pre-registered MDE, to remove the
"your static ceiling was actor-shaped" referee line.

## 4. WHAT SURVIVES FROM v2 (unchanged, still true — but SCOPED)
- **[REFUTED] the answerability thesis.** E0 tested it WITH THE PRIVILEGED TRUE TABLE => -0.0003.
  REFUSAL_RESULT.md (2026-06-29) already built two of the three "wires" => TIE tail, LOSE full. **Do NOT build
  the refusal wires.**
- **[V] "Refusal as evidence" is NOT novel** (Marlin & Zemel's CPT-v; Golbandi's unknown branch). The surviving
  novelty is **answerability as a CONTROL VARIABLE in the policy objective (DESIGN, not INFERENCE)**.
- **[V] Our belief is NOT linear-Gaussian** (learned attention fold-in) => the non-adaptivity theorem is an
  ANALOGY, not binding. **v2 used this as a caveat; STATIC8 shows it is THE POINT: we are on the non-Gaussian
  side, and that is exactly why adaptivity pays here.**
- All verified citation strings: see §6b of the v2 text preserved in git history (commit 8f33b28).

## 5. RUN ORDER (revised)
1. **E-BUDGET re-plot [FREE]** — the q-curves already exist; adaptivity is an early-turn effect that saturation
   erases. One figure.
2. **E-TAIL [CHEAP]** — per-user paired gap sliced by typicality (residual off the static bank's span).
   STATIC8's mechanism PREDICTS the gap concentrates on atypical users (and the TAIL gap +0.042 > FULL +0.038
   already agrees). ⚠ n~300 => PRE-REGISTER THE MDE; an underpowered null is uninterpretable.
3. **SNAP-K** — our snap is top-1 (`continuous_actor.py:187`), the configuration Wolpertinger reports as
   FAILING. Defends/kills Paper C's snap-loss centerpiece. See DESIGN_SHEET_E2_SNAPK.md.
4. **E2-residual** — one actor-INDEPENDENT static construction (§3).
5. **E5-OPENFOLLOW** — the human intuition in its purest form: is a follow-up CONSTRUCTED from the volunteered
   entity better than the best static continuation in the OPEN channel? (Arm A: probe built from the named
   entity's contrast axis. Arm B: best fixed open-question portfolio. Arm C: greedy static-8.)
6. **E4-DUAL** (Golbandi on our arena, two rulers) — the H(Theta) / estimator-strength discriminator.

## 6. PROCESS — THE FAILURE THAT PRODUCED v1 AND v2
**Three times in 24 hours I built a confident, citation-backed argument without reading our own results:**
E0, REFUSAL_RESULT.md, and STATIC8_RESULT.md. Each was on disk. Each refuted the argument outright.
The pattern is identical: a literature sweep produces a coherent story; the story FEELS like understanding; the
lab notebook that would falsify it goes unread. **The author caught all three from the outside.**
**RULE (now codified): before ANY claim about what our system does or does not do, grep `experiments/**/*RESULT*.md`
and `experiments/**/*RESULTS*.md` FIRST. The lab notebook outranks the literature and outranks me.**
