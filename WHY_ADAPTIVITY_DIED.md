# WHY ADAPTIVITY DIED — forensic synthesis, v2 (2026-07-14, overnight)
**v1 of this document proposed that the adaptive prize lives in the ANSWERABILITY channel and that we
deleted it. THAT THESIS IS REFUTED — by our own prior experiments, which v1 failed to read.** Fable's
adversarial pass killed it; every kill was then verified at source. This v2 records the refutation, the
surviving explanation, and the run order.

STATUS KEYS: **[V]** verified at primary source · **[V-2nd]** second-hand · **[D]** derivation ·
**[U]** unverified, do not cite · **[REFUTED]** claimed and disproved.

---

## 0. THE ANSWER (v2)
**Adaptivity is worth ~0 in our arena because our recommender is too good, not because our arena is too easy.**
Krause & Guestrin's Theorem 1 **[V]**: the value of adaptivity is upper-bounded by **H(Theta)**, the entropy of
MODEL uncertainty. Our dense collaborative fold-in generalizes across users so strongly that the posterior over
"which user-model" collapses within one or two answers => **H(Theta) ~ 0 => adaptivity is worth ~0 through
EVERY channel**, answerability included. Golbandi's estimator is a *decision tree* — a weak, piecewise-constant
user model with **large H(Theta)** — asking about individual Netflix titles, a vocabulary in which any user has
seen a sliver of the catalog (hence his 67% unknown). His adaptivity pays because his *model* is uncertain.
Ours does not because ours is not.

**We wrote this down in June and forgot it** (memory, `overnight-adaptivity-study`):
> *"collaborative generalisation substitutes for adaptivity on real catalogs."*

---

## 1. [REFUTED] THE v1 THESIS — and the three checks that killed it
v1 claimed: *the prize lives in answerability; we deleted the channel by (a) a belief in which refusals teach
nothing and (b) using our own instrument to pre-filter unanswerable questions out of the pool.* All wrong.

**[V] KILL 1 — E0 tested answerability routing AT ITS CEILING and got zero.**
`experiments/E0_GONOGO_RESULTS.md:55-80`. The O-ans arms held the **TRUE per-user answerability table**
(privileged/oracle). Best realizable arm, `O-ans static+skip` (per-user refusal-skipping):
**−0.0003 (TIE)**, CI [−0.0012, +0.0007]. The channel was **not deleted — it was measured, with perfect
knowledge, at zero.** E0's own text states the mechanism v1 thought it had discovered:
> "The strong static schedule already **self-selects** near-universally-answerable questions: users can answer
> **7.53 of B's 8** questions on average (94%). There are almost no refused turns to reclaim, so per-user
> answerability routing has essentially no headroom at T=8."
=> **94% answerable is an EQUILIBRIUM OF GOOD ASKING over an OPEN pool, not an engineered exclusion.**
The G2 decomposition is already in that file: **O-full − static = +0.287 (privileged TASTE-peek);
O-ans(best) − static ~ 0 (answerability discovery).** G2 was ~100% taste-peek. v1 re-derived a false story
about the provenance of a number whose provenance was already documented.

**[V] KILL 2 — the code forensics were wrong.** v1's "smoking gun" (`continuous_actor.py:2074`,
"pre-filter to answerable") sits inside **`if os.environ.get('LITBASE'):`** — the **Paper D
literature-baseline block** (ConTS/PEBOL probes), NOT the learned-policy train/eval path. Its filter is
`len(citems[c] & half) >= 2` = **simulator ground-truth profile membership**, NOT our `head_rated`
instrument. And the MAIN path **already pays refusal costs**: `:527  unans=(av.abs()<1e-6)  # ANS==0 =>
unanswerable/wasted ask`; `:674  fold only if answerable (== CASPER-R eval)` — the ask happens, the fold does
not, the turn burns. **"We used our own instrument to dodge" is FALSE.**

**[V] KILL 3 — two of the three proposed "wires" were built a month ago and LOST.**
`experiments/paper2/REFUSAL_RESULT.md` (2026-06-29): refusal-forced training (refusal wastes the turn,
straight-through gate, `_REFUSE/TAUR/REFTMP`) => **TIE on tail (0.179 vs D1 0.178), LOSE on full (0.370 vs
0.378).** Interpretable mechanism recorded there: refusing washed-out questions sharpens tail-relevant belief
but those weak answers carried diffuse signal that ranked the popular HEAD — net wash on tail, loss on full.
The `HANDOFF.md` receipt *"refusal-as-evidence (specced, never implemented) ✗"* is accurate for **wire #1
only**; v1 presented all three as virgin territory.

**[V] KILL 4 — the expected value of the fix is bounded near zero by evidence we already own.**
Routing value ~0 (E0, *privileged*) + predictive value ~ Golbandi's own **w_unknown = 0.02** (1/250 the weight
of a like) + avoidance-training value ~0 (REFUSAL_RESULT). **Do NOT build the three wires.**

**[V] KILL 5 — Golbandi's 67% is a VOCABULARY property, not a wiring choice.** His questions are individual
Netflix *titles* (any user has seen a sliver of the catalog). Ours are concepts/genre-tags + popular items — a
vocabulary that is *naturally* answerable. "We closed the channel 10x" conflates the question DOMAIN with a bug
in our code.

---

## 2. WHAT SURVIVES FROM v1
- **[V]** Krause & Guestrin, ICML 2007, §4 + **Theorem 1** (adaptivity <= H(Theta)); Krause/Singh/Guestrin JMLR
  2008 §3.1; Jedynak/Frazier/Sznitman, J.Appl.Prob. 2012 (greedy entropy = horizon-optimal; a *deterministic*
  question set is Bayes-optimal); Rainforth et al., Stat.Sci. 2024 §2.3. These explain **why an
  information-objective policy can never beat a static schedule**, and hence why static entropy was
  unbeatable — a genuinely useful, citable explanation.
- **[V] ⭐ Sepliarskaia, Kiseleva, Radlinski & de Rijke, "Preference Elicitation as an Optimization Problem",
  RecSys 2018** — a STATIC questionnaire BEATS adaptive decision trees; their explanation: *"PWDT optimizes a
  function that is different from the loss function, namely weighted generalized variance."* **We still do not
  cite this. We must.**
- **[V] Rashid/Karypis/Riedl, SIGKDD Expl. 2008 (IGCN)**: a cohort-based ADAPTIVE method **ties static Entropy0
  OFFLINE**, winning only ONLINE. v1 filed this as a "warning"; **it is actually evidence FOR the H(Theta)
  thesis** (offline, with a good CF estimator, cohorts buy nothing).
- **[V] `ANSWERABILITY_WRITEUP.md:54`**: external answerability is "heterogeneous + taste-tracking + NOT
  stereotype + exploitable". TRUE — but *exploitable* was established for **question SELECTION quality**
  (Paper B's answerable-concepts win), NOT for an *adaptive* prize. E0 priced the adaptive part at zero.

## 2a. [REFUTED/CORRECTED] OVERCLAIMS IN v1
- **"Our eight failures were one theorem measured eight times."** FALSE. Our policies were NDCG-trained; the
  covariance-only theorem does not bind them.
- **"The static baseline is PROVABLY OPTIMAL within its class."** OVERCLAIM. Krause gives sequential = a-priori
  *at the optimum of the class*; our static entropy is a **greedy** schedule, at best (1−1/e)-near-optimal under
  submodularity we have not verified.
- **THE DEEPEST ONE: our belief is NOT linear-Gaussian.** The fold-in is a **learned attention encoder** whose
  weights depend on answer VALUES => the effective precision is **already answer-dependent**. The shipped system
  satisfies **NEITHER** hypothesis of the theorem (objective is not a variance functional; belief is not
  linear-Gaussian). **The theorem is an ANALOGY for our system, not a binding result.** Any paper spine that
  rests on "a theorem explains our eight ties" **snaps** on the first BOED reviewer.
- **"−0.0003 is exactly what a second-order term looks like."** NUMEROLOGY. It is also exactly what zero looks
  like, and exactly what a failed optimizer looks like. No diagnostic content. Cut it.
- **§4 metric-regime argument**: a cap of e/(e−1) permits a **58%** improvement; we observed **0.0%**. A cap
  190x larger than the observation **explains nothing**. §4 survives only as motivation for a free re-plot, not
  as an explanation of the tie. (Also: the submodularity precondition is NOT established for NDCG-of-a-fold-in.)

---

## 3. THE SURVIVING EXPLANATION — H(Theta) ~ 0 (collaborative generalization)
**[D, but strongly supported]** Adaptivity <= H(Theta) **[V, Krause Thm 1]**. Our collaborative fold-in
generalizes across users so effectively that after 1-2 answers there is almost no *model* uncertainty left to
resolve; the remaining uncertainty is about the *user's* taste vector, which a **static, diversified,
population-optimal schedule already resolves near-optimally.** Golbandi's tree is a weak estimator with large
H(Theta) — every answer, including "unknown", routes among cohorts, and his residual uncertainty is
answer-dependent (his Fig. 2: 0.9393 / 0.9522 / 0.9837 by branch **[V]**) *because the model class makes it so*.

**This explains everything the answerability thesis explained, PLUS the three things it could not:**
E0's −0.0003 **with the true table in hand**; IGCN tying static offline; Sepliarskaia's static beating adaptive.

**Consequence for the thesis:** the honest headline is not "we made adaptivity work." It is
> **In cold-start elicitation, the value of adaptivity is bounded by the model uncertainty your recommender
> leaves on the table. A strong collaborative recommender leaves almost none — so a static, diversified
> questionnaire is (near-)optimal, and every adaptive policy ties it. Adaptivity is not a property of the
> elicitation problem; it is a property of the WEAKNESS of the estimator you pair it with.**
That is falsifiable, explains the field's contradictory results (Golbandi/IGCN-online vs Sepliarskaia/
IGCN-offline), and is corroborated rather than contradicted by our year of nulls.

---

## 4. THE DISCRIMINATING EXPERIMENT (answerability thesis vs H(Theta) thesis)
**E4-DUAL — Golbandi's tree on our arena, on TWO rulers.** Build the tree (all items as candidate splitters,
chunked sparse; **HARD RULE #1: no top-N popular-splitter reduction without author sign-off** — Golbandi does
this and it must be an explicit, signed ablation, not a silent default).
- **(a) NATIVE ruler**: the tree's own node-cohort mean as the predictor.
- **(b) FOLDED ruler**: the SAME question sequences, answers folded into **our** encoder, scored on **our** ruler.
- Plus the **3-way vs 2-way (unknown-branch) ablation on BOTH rulers.**
**PREDICTIONS (pre-registered):**
- *Answerability thesis*: the unknown branch matters on **both** rulers.
- *H(Theta) thesis*: **any tree gain EVAPORATES on ruler (b)**, and 2-way ~ 3-way there.
**Either way we learn the answer.** This is the experiment v1 should have proposed.

**H(Theta)-LITE (cheap, no policy training):** fit the K=32 mixture prior and measure the **posterior
cohort-entropy collapse rate over turns**. If the mixture posterior collapses in 1-2 answers, **H(Theta) ~ 0
and the adaptive prize is dead in this arena regardless of any refusal wiring.** This directly estimates the
theorem's bound on our own data.

---

## 5. RUN ORDER (Fable's, adopted; author signature required)
1. **E2-HARDENED — continuous STATIC schedule vs the continuous ADAPTIVE actor** (clean answers, identical
   action space, same seeds). Adjudicates Paper C's *existing, on-deadline* claim: was the +0.006/+0.010 win
   CONTINUITY or ADAPTIVITY? **Two ways this test can LIE, both must be closed:** (i) *multiplicity asymmetry* —
   the actor is a record-the-peak best-checkpoint over many runs; a one-shot static loses to selection noise =>
   false "adaptivity is real". (ii) *construction bias* — an R2b-style schedule distilled FROM the actor
   inherits the actor's discoveries => biased toward a tie => false "continuity". **Run BOTH constructions
   (independent greedy-on-train AND actor-distilled) to bracket; seed-average; verify baselines snap to
   canonical (the corrupted-cache lesson); PRE-REGISTER THE MDE** — the effect in dispute is +0.006 (~2 sigma),
   so an underpowered "tie" is uninterpretable.
2. **E4-DUAL** (§4) — the discriminator.
3. **H(Theta)-LITE** (§4) — measures the theorem's bound directly.
4. **DO NOT BUILD THE THREE WIRES.** E0 (privileged), REFUSAL_RESULT.md, and Golbandi's own w=0.02 jointly
   predict ~zero. Revisit only if E4-DUAL shows the unknown branch matters **on our ruler**.
5. **E3 (p_unknown sweep) is INCOHERENT AS WRITTEN — do not run it.** "Enforced so the policy cannot dodge" is
   self-contradictory: either the pool leaves choices (the risk head dodges *within* it, so experienced
   p_unknown << nominal and the x-axis is a lie) or it leaves none (there is no policy to test). **High
   experienced refusal AND good refusal prediction are mutually exclusive at the optimum.** Worse, a
   re-optimized *fixed sequence* comparator eats refusals as dead turns, so "gap grows with p" is guaranteed by
   wasted-turn arithmetic — **the P4C unfair-comparator artifact, second edition.** If ever redesigned: sweep the
   question VOCABULARY (concept-heavy -> obscure-title-heavy, Golbandi-ward), report experienced p_unknown as an
   OUTCOME not a knob, and use **static+skip WITH THE SAME refusal-aware belief** as the comparator.

## 6. PUBLICATION RISKS (must be closed before writing)
- **MNAR literature is missing.** "Refusal-as-evidence" IS the missing-not-at-random literature — Marlin &
  Zemel (2007/2009); **Steck, "Training and testing of recommender systems on data missing not at random",
  KDD 2010**. Claiming novelty here is a one-line reviewer kill. **[U — verify these citations at source.]**
- Any "provably optimal baseline" phrasing (§2a) will be shredded.
- The simulator-circularity attack: recovering a refusal-prize from taste-coupling coefficients **we authored**
  (`dans_build.py`: `c_align`, `genre_align`) measures our own prior. External validation (173 LLM-judged users)
  is QUARANTINED and establishes *existence*, not *magnitude*.

## 7. PROCESS LESSONS (recorded)
1. **READ OUR OWN PRIOR RESULTS BEFORE THEORIZING.** E0 and REFUSAL_RESULT.md each independently refuted v1's
   thesis and both were on disk the whole time. A literature sweep is not a substitute for reading your own lab
   notebook.
2. **A diagnostic whose CONTROL fails is reporting on itself, not on the data.** (v1 ran three broken tests and
   reported the OPPOSITE of the truth; see git history of this file.)
3. **Read the generative code before running a diagnostic on its output.**
4. **A "smoking gun" in code must be traced to the execution path that produced the number.** v1's was in a
   Paper D baseline block that never ran in the learned-policy experiments.
