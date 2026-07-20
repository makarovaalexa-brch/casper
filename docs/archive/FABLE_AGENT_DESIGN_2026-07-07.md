# The coarse→granular adaptive agent — Fable design (2026-07-07)

Answers Q1–Q7 of `questions_for_fable_AGENT.md`, grounded in Paper B §5.2 (adaptivity = per-user
re-ordering; bandwidth hierarchy; ML-25M tail scope condition), Paper C §5.6/i2fidelity (the boundary,
answerability-not-geometry is binding), the gate/main-study artifacts, and the hard lessons
(optimization gap, tie-by-construction, train-under-the-arena).

---

## 0. THE ONE DESIGN INSIGHT EVERYTHING ELSE FOLLOWS FROM

**G2 = 0.287 is NOT the answerability-discovery prize — it must be decomposed first.**
Selector A saw *both* the per-user answerability table *and* true taste (privileged belief). Most of
0.287 is plausibly taste-peek (knowing u\* tells you which questions are informative), not answerability
knowledge. Before building anything, split the bound with one intermediate oracle:

- **O-full (=G2 A): 0.568** — answerability table + true-taste belief. Privileged twice.
- **O-ans (NEW, the number that matters):** answerability table given, but belief built *only from the
  answers actually received* (realizable belief path). Gap over the brutal static = the true
  **answerability-discovery prize** — the most any realizable agent can earn from knowing who can
  answer what.
- **O-none (=G2 B → replaced by the brutal static of Q3).**

If O-ans − static is small, no agent architecture can win and we stop before training anything
(this is the cheapest GO/NO-GO — §Q6). If it's material, the realizable agent's job is to capture a
fraction of *that* gap by **estimating the table online** — and the fraction captured is the finding.

**The second insight: the agent's only genuinely per-user unknown is the answerability table, and the
fitted pmodel makes estimating it a LOW-DIMENSIONAL problem.** The pmodel's features are all public
(popularity, ratings-count, decade, franchise, is_concept) *except* `genre_match` (+0.63) — which
depends on the user's taste profile. So the entire per-user answerability table is parameterized by one
small unknown vector: the user's genre/taste distribution **ĝ**. Discovering answerability
in-conversation = doing online inference on ĝ (≈19–20 genre dims, not 64-d taste, not 18k items).
That is learnable from ~8 observations in a way that a 64-d belief is not — and it is exactly what a
human interviewer tracks ("this person knows their noir").

---

## Q1. ARCHITECTURE: (c)-hybrid, but with the learned part demoted to calibrated scalars

Given the optimization-gap record (8+ gradient-trained policies tied or lost; direct
search/construction wins at these sample sizes), the policy should be a **decision-theoretic scorer
whose structure is fixed and whose few free parameters are set by direct search on val** — i.e. the
realizable mirror of the G2 selector, not an RL actor. RL/differentiable-unroll (option a) is the
*last* rung of the ladder, attempted only if the scorer leaves a measurable gap to O-ans.

### State (all observable — nothing peeks)
- **z** — belief latent, updated by the existing additive operator z′ = z + η·a·q (graded a from the
  arena's answer model; refusals contribute NO z update).
- **ĝ** — online genre/taste-distribution estimate (the answerability posterior's only unknown):
  Dirichlet-style counts over genres, updated by every observable event (below).
- **n_eff** — evidence mass on ĝ (how sharp the posterior is; controls exploration).
- **asked set + refusal set + turn t.**

### The answerability posterior update (the mechanism — every event is evidence)
- **Answered concept/item (any polarity):** add the question's genre vector to ĝ weighted by
  P(knows|answered)=1. A *dislike* still proves knowledge — polarity goes to z, knowledge goes to ĝ.
- **Refusal:** Bayesian down-weight: the refused question had model-predicted answerability
  p̂ = pmodel(features, genre_match(ĝ)); a refusal is a Bernoulli-0 observation → shift ĝ away from the
  question's genre region proportionally to p̂ (a surprising refusal is strong evidence; an expected
  one is weak). Refusals are *observable and informative* — this is refusal-folding.
- **Open recall (channel):** one named item = a full genre vector + a taste factor in one turn — the
  highest-information ĝ update available, which is why it is the natural opening move (see action space).
- Per-user answerability estimate for ANY candidate q: **p̂(q) = pmodel(public features(q),
  genre_match(q, ĝ))** — the fitted AUC-.921 surrogate with the one private feature imputed online.

### Action space and the scorer
Candidate pool each turn = union over channels, pre-filtered for deployability (items only from the
judged/scored bank; concepts = the 200 breadth-tiered tags; pairs only over items with p̂ high for both;
sliders = one-screen concept batteries; open recall = the framing-levered prompt; abstract-continuous
EXCLUDED — ceiling only, per E).

Score every candidate myopically:

**V(q) = p̂(q) · ΔNDCĜ(q | z) − (1 − p̂(q)) · c_refusal + β · IG_ans(q | ĝ, n_eff)**

- ΔNDCĜ(q|z): expected belief-improvement value of an *answered* q — use the existing divisiveness/
  entropy machinery (the same value model the static baseline uses; identical on both sides by design).
- c_refusal: the observed cost of a wasted turn (estimated once on val, not tuned per-run).
- IG_ans: expected information gain *about ĝ itself* — an explicit, small exploration bonus that makes
  the agent occasionally ask a taste-diagnostic question because it sharpens the answerability map.
  β anneals with n_eff.
- Free parameters: **η already fixed; β, c_refusal, τ (below) — THREE scalars**, set by grid/direct
  search on val users. Nothing else is trained. Tie-by-construction: with β=0 and a flat ĝ prior the
  agent reproduces the brutal static's first pick and follows the population schedule — it can only
  add on val.

### Coarse→granular is EMERGENT, not hard-coded (this is the paper's elegance)
Early: n_eff≈0 → ĝ ≈ population prior → genre_match is uninformative → p̂(q) is driven by popularity/
breadth → only broad, high-population-answerability questions clear the p̂·value tradeoff → the agent
opens coarse. As answers/refusals sharpen ĝ, taste-adjacent niche questions (high value, previously
low-p̂) rise — *for this user only* — and the agent descends. Granularity descent falls out of decision
theory. Keep an optional hard gate (ask only if p̂ ≥ τ, τ annealing 0.8→0.4) as a variant rung: if the
emergent behaviour needs the gate to materialize, that is itself a reportable fact.

### Why not (a) or pure (b)
- Pure RL (a): every prior attempt tied; sample sizes haven't changed; the scorer already *is* the
  myopic optimum given its inputs, so RL's only upside is non-myopia — test for non-myopic headroom
  first (O-ans with 2-step lookahead vs 1-step; if ≈equal, RL has nothing to win).
- Pure heuristic tree (b): a hand-built coarse→granular tree hard-codes the descent schedule — it
  cannot express "descend into noir at turn 3 for THIS user" without becoming the posterior machinery
  anyway. The scorer is (b) with the one hand-coded part replaced by the validated pmodel.

## Q2. HOW ADAPTIVITY EARNS ITS KEEP — the signal, the mechanism, the numbers

The user-specific signal is the **answerability table**, whose only private coordinate is genre_match
(+0.63/sd, the fitted model's user-specific term; everything else is population-public). A static
schedule — even one built with FULL knowledge of the population answerability model — must average over
ĝ: it can exploit "most people can answer '1970s?'" but never "THIS user can answer 'giallo?'". The
mechanism chain, explicitly:

**answers + refusals → ĝ posterior → p̂(q) per user → taste-adjacent niche questions unlock →
higher-bandwidth evidence (Paper B: item/pair channel ≈27 eff-dims vs concepts ≈2–4) reaches the
belief for users who can supply it — and ONLY for them.**

The bandwidth hierarchy is what makes the prize non-trivial: Paper B shows the concept channel
saturates by ~q4 because it is low-rank; the escape is item/pair-grade questions, which are exactly the
questions that are unanswerable *on average* (0.03% at scale) but answerable *in the user's home
territory*. Adaptivity is the only route from the saturating coarse channel into the high-bandwidth
fine channel without burning turns on refusals. Static schedules must choose: stay coarse (saturate)
or go fine (refusal tax on most users). The agent does per-user routing — which is also precisely the
gut-check story (a human interviewer descends into what the interviewee demonstrably knows).

Quantitatively: G2 says perfect table + perfect taste = +0.287. O-ans (Q6's first diagnostic) will say
how much of that is the table alone; expect the realizable agent to capture a *fraction* of O-ans −
static (efficiency-shaped, biggest at low q where refusal-avoidance compounds). Pre-registered honest
expectation: the win, if real, shows at q2–q4 anytime NDCG and in refusal-rate/questions-to-quality,
not as a giant q8 endpoint delta (Paper B: endpoint saturation is real).

## Q3. THE BRUTAL STATIC BASELINE (the only opponent that makes the win real)

Construct by **greedy forward selection with full population-answerability knowledge, on val users,
under the identical arena** (same answer model, same fidelity σ, same refusal costs, same belief
operator, same value model as the agent's ΔNDCĜ — any asymmetry here invalidates the comparison):

1. Position 1: try every candidate (all channels, all granularities, open recall included); keep the
   one maximizing mean val NDCG *after simulated answers with population-rate refusals*. 2. Fix it;
   repeat for position 2 conditional on position 1's simulated outcomes; … to T=8. 3. Freeze; evaluate
   on test only once.
- Allow the static to use EVERYTHING except per-user information: population answerability rates,
  channel mixing, the framing lever, even opening with open recall. If open recall is strong, the
  static gets it too — the agent must win on *routing*, not on channel access.
- Report it with the same 5-seed averaging + per-user paired bootstrap as the agent.
- ALSO build **static+skip (“conditional static”)**: the same frozen sequence, but a refused question’s
  turn is refunded to the next item on the list. This is the cheapest possible adaptivity (depth-1
  conditioning, no posterior) and it is the baseline most likely to embarrass the agent — that is why
  it must be in the ladder (Q5 rung 2). A win over brutal-static that vanishes against static+skip
  means "adaptivity = don't waste refused turns," an honest but much smaller claim.

## Q4. PRE-REGISTERED SUCCESS CRITERIA

- **Arena:** ML-25M, 300 study users' held-out targets, cached judged grid as ground-truth
  answerability (pmodel-with-true-profile for off-grid questions), real-rating/data-side answer values,
  fidelity σ=0.70, T=8, seed-avg {1,2,3,7,11}, per-user paired bootstrap, depth-matched NDCG@50
  (declared once, both axes full+tail).
- **Co-primary metrics:** (P1) **anytime NDCG** = mean over t∈{1..8} of NDCG@50 after t turns (the
  efficiency claim is the honest one); (P2) **endpoint NDCG@50 at t=8** (non-inferiority guard).
- **Secondary:** questions-to-95%-of-static-endpoint; refusal rate; per-turn answered-token count;
  tail NDCG (reported, not gated — Paper B's ML-25M tail scope condition says don't bet the claim on it).
- **Decision rule (adaptive coarse→granular wins):** agent beats the *brutal static* AND *static+skip*
  on P1 with 95% CI excluding 0 and Δ ≥ 0.010 anytime-NDCG, AND is non-inferior on P2 (Δ ≥ −0.005).
  Claim wording then: "adaptive answerability-routing wins on efficiency at equal endpoint quality."
- **Tie/fallback framing (pre-registered so it can't be spun later):** if the agent ties static+skip,
  the shipped result is: (i) the static channel map + the measured oracle decomposition
  (O-full/O-ans/static) as the *price of answerability knowledge*; (ii) the finding "the
  answerability prize at T=8 is captured by refusal-skipping alone; per-user posteriors need longer
  horizons" — a scope statement, not a failure narrative (per the stop-selling-negatives rule: the
  channel map remains the positive spine either way).

## Q5. ABLATION LADDER (one mechanism per rung, run top-down after the headline)

0. **O-full** (G2 A, 0.568): table + true taste. Context only.
1. **O-ans**: table given, realizable belief. → the answerability-discovery prize. (Also run with
   2-step lookahead once: bounds what RL could ever add; if ≈1-step, RL rung is cancelled.)
2. **Full agent**: online ĝ posterior + refusal-folding + emergent descent + β-exploration.
3. **− exploration** (β=0): does deliberately probing the answerability map matter, or is passive
   updating enough?
4. **− refusal-folding** (refusals cost a turn but don't update ĝ): isolates the refusal channel.
5. **− answerability-posterior** (ĝ frozen at population prior; p̂ = population rates): the agent
   degenerates to a value-greedy selector → measures everything the posterior added (rungs 2−5 vs 5).
6. **− coarse→granular** (hard gate removed AND p̂·value tradeoff replaced by value-only ranking):
   does descent order matter, or only *which* questions get asked?
7. **static+skip** (depth-1 adaptivity).
8. **Brutal static** (floor).
Attribution = adjacent-rung deltas, each with CIs; expected shape: most of the gain lives in rungs
2/4/5 (posterior + refusal-folding), rung 6 tests the thesis's *narrative* (coarse→granular
specifically) — if rung 6 ≈ rung 2, the honest claim is "answerability-aware routing" rather than
"coarse→granular descent," and the paper says so.

**Robustness envelope (after the ladder, headline rung only):** re-run the full agent + brutal static
under (i) the Haiku judged grid (conservative judge), (ii) the a-priori structural answerability rule
(the independent witness — the adaptive claim must at least keep its SIGN here), (iii) EASE-sourced
counterfactual values (sensitivity-only channel). Same discipline as everywhere else: the claim is
stated where both external models agree.

## Q6. COST, RISKS, AND THE $0-GO/NO-GO

**Compute:** no gradient training in the main line. Everything is simulation on cached grids +
direct search over 3 scalars: O-ans + brutal static ≈ a few CPU/GPU-hours; agent + ladder ≈ a day.
LLM cost: $0 (the grids are paid for). The only new spend is if the RL rung is ever justified.

**GO/NO-GO (one afternoon, zero new calls, run FIRST):**
1. **O-ans vs brutal static.** If O-ans − static < ~0.015 anytime-NDCG, the realizable prize is too
   small — stop, ship the static channel map + the decomposition (itself a publishable scoping of G2).
2. **Posterior-sharpening curve (no interviews needed):** feed each user's first t simulated
   answers/refusals into the ĝ updater, plot AUC(p̂ vs judged grid) against t. If AUC at t=4 is still
   ≈ population-prior AUC, the posterior can't sharpen inside the budget → the agent mechanically
   cannot route before the interview ends. (This is failure mode #1 measured directly.)
3. **Fuel-value scatter:** over the judged grid, plot per-question [answerability variance across
   users] × [question value ΔNDCĜ]. The agent's prize lives in the upper-right quadrant; if it's
   empty (all high-variance questions are low-value — the ML-25M head-aligned-concept worry from
   Paper B), there's nothing to route toward.

**Top-2 failure modes:** (1) the posterior sharpens too slowly for an 8-turn budget (diagnostic #2;
mitigation: open-recall opener front-loads ĝ — one named film is a full genre vector at turn 1);
(2) unlocked niche questions carry too little marginal value because the concept vocabulary is
head-aligned and item evidence is tail-sparse on ML-25M (diagnostic #3; mitigation: the pair channel
over the user's answerable region — precise tail evidence per Paper C's foreign-geometry result 0.223).

## Q7. THE RESULT STATEMENT, BOTH BRANCHES (abstract-grade, overclaim-free)

**If it wins:** "We give the first elicitation agent whose adaptivity targets the one quantity that is
genuinely per-user in cold start: *answerability*. Using an externally validated answerability model
(LLM-judged, human-validation pending; AUC .92 held-out users), the agent maintains an online posterior
over what this user can answer — updated by answers, refusals, and a single open-recall turn — and
descends from broad concepts to fine, taste-adjacent questions exactly where that posterior permits.
Against the strongest static schedule constructible with full population-level answerability knowledge
(and its refusal-skipping variant), the agent improves anytime NDCG by Δ=… [CI], at equal endpoint
quality, capturing …% of the measured answerability-knowledge oracle bound; ablations attribute the
gain to the answerability posterior and refusal-folding. Coarse-to-granular interviewing — the strategy
human interviewers use by instinct — is thus recoverable in a recommender agent, but only when the
environment contains real per-user answerability structure: prior simulators, including our own earlier
ones, removed exactly this structure, which is why learned adaptivity repeatedly tied static schedules."

**If it ties:** "We measure, for the first time, the value of per-user answerability knowledge in
cold-start elicitation: a privileged oracle with the user's full answerability table gains Δ_O=… over
the best population-informed static schedule, but a realizable agent that must *discover* the table
in-conversation captures only …% of it at an 8-turn budget — refusal-skipping alone accounts for the
rest. The binding constraint is discovery speed: the answerability posterior needs more turns to
sharpen than a realistic interview affords. Cold-start elicitation design should therefore invest in
the channel map — choosing broadly answerable, high-bandwidth channels up front — rather than in
per-user adaptive routing; we provide the validated answerability environment so that longer-horizon
or cross-session agents can be tested against the measured oracle bound."

Both are positive contributions; neither hedges; the tie branch converts the loss into the measured
price-of-discovery, which no prior work has quantified.

---

## RUN ORDER (series of experiments, ~1.5–2 weeks)

| # | Experiment | Gate |
|---|---|---|
| E0 | GO/NO-GO trio: O-ans decomposition, posterior-sharpening curve, fuel-value scatter | O-ans−static ≥0.015 & AUC rises by t≤4 & quadrant non-empty |
| E1 | Brutal static + static+skip construction on val; freeze | — |
| E2 | Agent v1 (scorer, 3 scalars by val grid-search; tie-by-construction warm start) | beats static+skip on val P1 |
| E3 | Test-set headline: agent vs both statics, 5 seeds, paired bootstrap, P1/P2 rule | pre-registered Δ |
| E4 | Ablation ladder rungs 1–8 | attribution |
| E5 | Robustness envelope: Haiku grid / structural rule / EASE values | sign holds |
| E6 | (Only if O-ans 2-step ≫ 1-step AND agent ≪ O-ans) RL/unroll rung, warm-started from the scorer | can only add on val |

Pre-register E3's rule (this doc, frozen + committed) BEFORE E2 tuning ends. All checkpoints/schedules
persisted per the save-all-checkpoints rule; peak on disjoint val; canonical baselines verified before
any claim.

---

# ADDENDUM (2026-07-07): could a better model be fitted to the LLM data? (owner question)

Short answer: yes — the current pooled logistic leaves identifiable signal on the table — but the
upgrades divide sharply by WHERE they may be used. State the firewall rule first, because it settles
the specific question asked ("on the learned user belief rather than genre distribution?"):

## The firewall rule (asymmetric, and it dissolves the dilemma)

- **Environment side (ground-truth answerability for eval):** must be INSTRUMENT-INDEPENDENT. Using
  the RecVAE latent z (or any learned belief of the instrument under eval) as a feature of
  P(answerable) would re-couple the answer environment to the evaluated model's geometry — the exact
  soft circularity the whole redesign removed. An agent probing the instrument's geometry would again
  win partly by construction. So: environment-side features must be DATA-SIDE only (popularity,
  counts, decade, genres, tag-genome, the judged grid itself).
- **Agent side (the online p̂ estimator):** may use ANYTHING observable, including the agent's own
  belief z — that creates no evaluation circularity, because the ground truth it is scored against
  (the judged grid) is independent of the instrument. Agent-side z-features are legal and even
  natural ("my taste estimate says this user is near noir → probably knows noir").

So "fit on the learned user belief" is FORBIDDEN for the environment surrogate and ALLOWED (as an
ablation rung) for the agent's estimator. The question conflated the two roles; split, both answers
are easy.

## Upgrades worth making (ranked by value/cost, all environment-legal)

1. **Per-user random intercept (mixed-effects logistic) — the biggest missing term, nearly free.**
   ICC=0.174 says users differ in overall knowledgeability; the pooled model ignores it, so its
   per-user calibration is off even at AUC .92. Fit user intercepts on the grid; for OFF-grid users
   the intercept is unidentified — which is fine, because the AGENT can estimate it online almost
   instantly (the running answer-rate over the first 2–3 turns is a sufficient statistic). This also
   hands the agent a second, even faster adaptation channel than ĝ: "this user refuses a lot →
   stay coarse longer" — pure coarse→granular behaviour from one scalar.
2. **Tag-genome taste-match feature (replaces/augments genre_match).** genre_match at ~20 genres
   cannot represent "Scandinavian crime" or "giallo" — the very heterogeneity the gate found. The
   genome (1128 data-side relevance dims) gives a much sharper question↔profile similarity, entirely
   instrument-independent. Likely the single largest AUC gain, and it directly upgrades the agent's
   fuel: the +0.63 coefficient was earned with the BLUNTEST possible taste-match; a genome match
   should raise both the coefficient and the routable signal. Agent-side ĝ then lives in genome
   space (still estimable: each answered/named item deposits its full genome vector).
3. **Low-rank collaborative residual (answerability MF), k≈4–8.** Factorize the 108k-observation
   (user × question) grid: user "knowledge-dimension" latents beyond taste-match (era-knowledge,
   cult-knowledge, mainstream-breadth...). Data-side (fits LLM labels, not the recommender), so
   environment-legal. Per-user latents exist only for the 300 study users — fine for the environment
   (those ARE the study users); the agent infers its own k-dim estimate online, which at k≤8 is
   still feasible in-budget. Do this only if (1)+(2) leave a within-user gap (see metric below).
4. **NOT worth it:** deep/nonlinear models on 6 features (nothing to gain at n=108k with this feature
   set); LLM-embedding features of question text (re-imports the judge's text prior into the
   surrogate — shared-prior surface again).

## The right metric is not pooled AUC (why .921 overstates what the agent gets)

Two corrections to how the surrogate should be judged:
- **Ceiling:** cross-family judge agreement is 83% (κ=.67). The surrogate predicts ONE judge's
  labels; it cannot meaningfully exceed the judge's own consistency. Pooled AUC .92 is already near
  that ceiling — chasing .95 is fitting judge noise.
- **Decision relevance:** the agent never compares users; it ranks QUESTIONS within one user. The
  pooled AUC is inflated by between-user separation (the easy part, mostly popularity). Evaluate:
  (i) **within-user AUC** (per-user question ranking, averaged), and (ii) **decision loss** — run the
  G2-style selector with p̂ substituted for the true grid and measure the NDCG it forfeits vs the
  table-oracle (O-ans). Decision loss is the number that tells you whether a better surrogate buys
  the AGENT anything; pre-register it as the surrogate's acceptance metric for upgrades (an upgrade
  is adopted iff it reduces decision loss, not iff it raises pooled AUC).

## What this changes in the plan

- E0 gains a fourth diagnostic: **within-user AUC + decision loss of the current pmodel**. If decision
  loss vs O-ans is already small, the surrogate is NOT the bottleneck — skip upgrades, build the agent.
- If it is material: apply upgrades (1)→(2)→(3) in that order, re-measuring decision loss after each;
  stop at diminishing returns. Each is hours, $0 (the grid is paid for).
- Agent ladder gains one optional rung: **agent-side p̂ with z-features** (belief-informed taste-match)
  vs data-side-only ĝ — legal per the firewall, and it directly tests whether the instrument's belief
  helps answerability routing beyond what data-side features capture.

---

# THESIS-LEVEL REVIEW (2026-07-07): fundamental upgrades + novelty-claim inventory

Owner ask: think through the whole arc (A–E, the channel-map reframe, the answerability study, the
agent plan) and improve it FUNDAMENTALLY — more valuable or more interesting — plus an explicit list
of novelty claims, done vs planned.

## 1. The arc as it stands (one paragraph)

A built the instrument and showed the apparatus decides what elicitation results even mean; B named
answerability and proved item-asking dies at scale; C found the fidelity boundary (the ordering of
question types inverts under a realistic answer channel); D showed open recall is the only route to
item-grade bandwidth at scale; E proved channels are not renderings of one another. The reframe
unified these as a CHANNEL MAP over three measured axes (answerability × fidelity × bandwidth). The
answerability study then removed the field's standing sin — self-authored answerability — with an
external validated judge, and found real per-user structure (the fuel). The agent plan tests whether
adaptivity finally pays in an environment that has that structure. This is already coherent. The
upgrades below are about making it *predictive*, *externally credible*, and *unified* rather than
merely well-organized.

## 2. THE UNIFYING REFRAME (U1, free, do it): the agent is not "also an agent" — it is per-user
navigation of the map

Currently the map is a population statement (the boundary is one line for everyone) and the agent is
a separate chapter-ending stunt. But answerability is PER-USER (the entire gate finding), and fidelity
plausibly is too. So the boundary is per-user: **for a knowledgeable user the fine channel is above
water; for a casual user it is below.** The agent is then exactly one thing: **an estimator of where
THIS user sits on the channel map, updating in-conversation, routing questions accordingly.** The
thesis becomes one sentence:

> *We build the map of elicitation channels (answerability × fidelity × bandwidth), show the winning
> channel flips across it, and give an agent that locates each user on the map while interviewing them.*

Corollary framing with real novelty bite — **elicitation is joint active inference over TWO latent
variables**: taste u\* (what they like) and knowledge g (what they can answer). Every prior CRS line
(ConTS, UNICORN, EAR, PEBOL, GATE) infers taste only; the interviewer's second job — model the
interviewee's knowledge and aim questions inside it — has never been formalized as an inference
target. Questions get dual payoffs (taste-info + knowledge-info); refusals become observations rather
than failures; coarse→granular emerges as the optimal schedule when knowledge-uncertainty starts
high. This is a *claimable formal framing*, not decoration, and the agent design (ĝ posterior +
IG_ans bonus) already implements it — it just needs to be NAMED in the papers. Cost: writing only.

## 3. Fundamental upgrades, ranked (value ÷ cost, with the 12-week ECIR clock in view)

**U2 — Make the boundary PREDICTIVE, not just measured (theory spine; ~1–2 weeks; the single biggest
"interesting" upgrade).** Today the fidelity boundary is an empirical crossing. The three axes admit a
light information-theoretic model: each channel = a noisy linear measurement of u\* with per-question
information ≈ ½log(1+SNR_eff), where SNR_eff combines fidelity σ (measured: 0.70 stars), channel
effective rank (measured: concepts ≈2–4, items ≈27), and answerability = the per-user availability
mask on measurements (measured: pmodel). Integrate over T turns → predicted NDCG ordering → **predict
the crossing point from independently measured channel parameters, THEN verify against the empirical
boundary already in Paper C.** If even approximately right, the map stops being cartography and
becomes a law — and the linear-Gaussian machinery is already in the thesis (it currently only
explains ties post-hoc, which the adversarial review dinged; this flips it into a pre-registered
prediction). Fallback if the quantitative fit is poor: publish the qualitative order-of-crossings
prediction, still novel.

**U3 — Sim-to-sim transfer: evaluate the agent against a SECOND environment it was never tuned in
(~3–4 days, ~$10s; the single biggest credibility upgrade).** The current plan evaluates the agent in
the same judged environment that built its surrogate — a reviewer's next circularity move ("the agent
overfits the judge"). Fix: hold out the **Haiku-judged grid as a foreign arena** (different model
family, known to be systematically more conservative, κ=.67 with the primary) and, stronger, run the
interview against a **persona-LLM user** (cross-family, conditioned on the known-half profile,
answering/refusing in the loop) rather than against cached labels. The agent is tuned ONLY in the
GPT-judged arena; headline includes its transfer performance. Sign-preservation under transfer is the
pre-registered claim. This is the cheapest available approximation of "does it survive contact with
someone who is not our simulator" before the human study.

**U4 — Release the arena as a benchmark (~1 week packaging; the biggest external-value upgrade).**
The field's structural defect (self-authored answerability) is now solved *in an artifact*: judged
grids (2 model families), fitted+calibrated surrogate, fidelity calibration, non-circular value
sources, oracle ladder (O-full/O-ans/static), pre-registered metrics. Package it (data + env loop +
baselines + the G2/O-ans oracles) and name it. A benchmark outlives every individual finding, makes
the thesis the *reference point* for elicitation evaluation, and converts A from "our testbed" into
"the community's instrument." It is also the strongest possible answer to "why should we trust your
sim" — *here, run it.* (ECIR loves resource papers; this can be Paper 1's resource track or a
standalone demo/resource submission.)

**U5 — One cross-domain boundary replication (Goodreads; ~1–2 weeks, mostly compute).** Every
elicitation claim lives on MovieLens; the I2 apparatus already runs Goodreads. Replicate ONE thing —
the fidelity-boundary crossing — on books. If it holds, the map generalizes; if it shifts, the shift
itself is informative (books answerability is plausibly MORE heterogeneous). Do after U2 so the
theory predicts the shift direction.

**U6 — Human study, elevated from judge-validation to MAP-validation (already planned; re-scope, no
extra cost).** As designed, the human study validates the LLM judge. Add one analysis, zero new data:
place the 50 humans ON the map (their measured test-retest fidelity, their per-channel answer rates)
and check the map's prediction for which channel should win for them. Even a directional confirmation
("humans sit below the slider-fidelity boundary → discrete should win → their answers' downstream
NDCG agrees") is the round-trip that no elicitation paper has.

**Scope guard (owner constraints: 12 weeks to ECIR, 2 kids, 1 year left).** Mandatory: U1 (free) +
U3 (days, protects the crown result). Strongly recommended: U2 (it upgrades BOTH ECIR papers) and U4
(reuse of existing artifacts; can trail as a resource paper). U5/U6-extension are thesis-stage, not
ECIR-stage. If forced to choose one: **U2** — a predicted-then-verified boundary is the difference
between "solid empirical thesis" and "memorable thesis."

## 4. NOVELTY-CLAIM INVENTORY (✓ done / ◐ done-needs-repair / ? planned)

**Measurement & apparatus**
- N1 ✓ Instrument class + eval protocol determine whether elicitation is visible at all (same data,
  two verdicts). First replication-first treatment in elicitation.
- N2 ✓ Gate suite + oracle-privilege ladder for elicitation claims (privileged vs realizable, made
  systematic).
- N3 ? The arena benchmark: first elicitation environment with EXTERNAL (non-self-authored)
  answerability, dual-judge, calibrated fidelity, non-circular values (U4).

**The map (the flagship's spine)**
- N4 ✓ Answerability as the organizing axis; item-asking structurally dead at catalogue scale
  (~0.03% answerable; replicated on 4 datasets).
- N5 ✓ The answer-fidelity boundary: question-type ordering INVERTS under an empirically calibrated
  answer channel; snap-loss reverses sign (snapping denoises). Nobody has this figure.
- N6 ✓ Bandwidth hierarchy with a mechanism: channel effective-rank explains saturation
  (concepts ≈2–4 dims saturate by q4; items ≈27 keep climbing) + the ML-25M tail scope condition.
- N7 ✓ Un-askability: latent queries are readable but not faithfully askable; channels are not
  renderings of one another (why a map must exist).
- N8 ◐ Framing lever: question wording steers volunteered-item popularity (−33pt head→tail) — needs
  the de-rigged symmetric comparison before it is claimable.
- N9 ? The boundary as a PREDICTION from measured channel parameters (U2) — map → law.

**Answerability science (new since the reframe)**
- N10 ✓ First externally-sourced per-user answerability model in elicitation: LLM judge, validity-gap
  test vs matched never-rated (+44pt), cross-family replication (κ=.67), calibrated surrogate
  (AUC .92 held-out users, ECE .008).
- N11 ✓ Answerability heterogeneity is real and taste-tracking (ICC .174; genre_match OR 2.71/sd) —
  i.e., the fuel prior simulators removed is measurably present in humans-as-modeled.
- N12 ? The price of answerability knowledge: O-full/O-ans/static decomposition — first quantification
  of what knowing "who can answer what" is worth, separate from knowing taste.

**The agent (planned)**
- N13 ? Elicitation as JOINT active inference over taste AND knowledge (u\*, g) — refusals as
  observations, dual-payoff questions, coarse→granular as emergent optimal schedule (U1 naming).
- N14 ? First adaptive-elicitation win in an environment with validated answerability structure —
  OR (tie branch) first measurement of the discovery-speed constraint that prevents it. Either way
  it resolves WHY 8+ learned policies tied static schedules (the environments had no per-user
  structure to exploit) — retroactively explains a whole literature's null results.
- N15 ? Transfer: the agent's edge survives a foreign judge/persona-LLM user (U3) — adaptivity is not
  an artifact of its own training simulator.

**Validation**
- N16 ? Human round-trip: real users placed on the map; the map's channel prediction confirmed
  directionally (U6). First human closure of an elicitation-sim loop.

Claims to STOP making (from the earlier reviews, recorded here so the inventory is complete):
continuous-beats-discrete as headline (scope: noiseless only), B's learned-policy win (seed-fragile),
D's 2-beats-8 (privileged answerer), A's question-channel-strongest capstone (tautological).

## 5. What I would NOT change

- No new datasets beyond Goodreads; no new instrument work (I2 is certified — done).
- No RL-first agent (the optimization-gap lesson stands; the scorer IS the design).
- No attempt to model answerability from logs (the owner killed it correctly; the external judge is
  the design).
- The honest-scoping discipline (privileged vs realizable, pre-registration, dual-model robustness)
  is the thesis's signature — it is now a FEATURE of the story ("the field's sims flatter adaptivity;
  ours is the first that doesn't, and adaptivity works/fails there for measurable reasons").

## 6. THE SAME REVIEW IN PLAIN LANGUAGE (no jargon — read this version first)

### The story so far, plainly

You spent the PhD asking: *when a system meets a brand-new user, what should it ask them?* Along the
way you found five things: (A) the answer you get depends enormously on which recommender you use to
measure it — a weak one shows nothing, a strong one shows a lot, so you built a trustworthy measuring
setup first; (B) on a big catalogue you can't ask about specific movies, because almost nobody has
seen any specific movie — you have to ask broad questions ("do you like horror?"); (C) fancy abstract
questions win only if the user's answers are unrealistically precise — with realistic sloppy answers,
plain broad questions win, and you found the exact point where this flips; (D) instead of asking, you
can let the user *volunteer* a movie they love — one named movie tells you more than several
yes/no answers, and the wording ("favourite" vs "hidden gem") controls what kind of movie they name;
(E) the fancy abstract questions can't even be turned into normal English questions without losing
what made them good — so the different ways of asking are genuinely different tools, not the same
tool in different clothes.

Put together: there is a small set of *ways to ask* (broad questions, specific movies, comparisons,
sliders, "name one you love"), and each way has three properties you can measure: **can the user
answer it at all** (answerability), **how sloppy the answer is** (fidelity), and **how much one answer
tells you** (bandwidth). Which way of asking wins depends on those three properties. That's "the
channel map" — a practical guide: *in this situation, ask this way.*

Then the new study: previously, your simulated user could answer whatever *you* had decided it could
answer — you wrote the rule, so any "discovery" of it was rigged. Now an outside judge (an LLM
reading a person's history) decides what each user could answer, and it turns out **different users
really can answer very different questions, in a way that follows their taste** — a horror fan can
answer niche horror questions a rom-com fan can't. That difference between users is the "fuel": it's
the thing an adaptive interviewer could exploit and a fixed script cannot.

### U1 — the reframe, plainly

Right now the plan reads as "we made a guide to ways-of-asking, and separately we built an
interviewing bot." The improvement: they're the same thing. Because users differ in what they can
answer, the guide's advice is *different for different users* — for a film buff, fine-grained
questions are worth it; for a casual viewer they're wasted. So the bot's real job is: **figure out,
during the conversation, which kind of user this is, and follow the guide's advice for that kind of
user.** One sentence for the whole thesis: *we made the guide, and we made an interviewer that works
out which page of the guide applies to you.*

The second part ("joint active inference over two latents", N13) just means: a good interviewer
tracks two separate things about you at once — **what you like** and **what you know**. All previous
research systems track only the first. Yours tracks both, and treats a refusal ("no idea, never seen
it") not as a failure but as information about what you know. The claim to make in the paper: we're
the first to treat "what the user knows" as a thing the system explicitly estimates.

### U2 — "make the boundary predictive", plainly

You currently show, by experiment, that sloppy answers flip which question type wins. That's a
measurement — like noticing water boils at 100°C. The upgrade is to have a small formula that
*predicts* the flip point from things you've already measured (how sloppy answers are, how much a
question type can tell you, how often people can answer) — like having the physics that says *why*
it boils at 100°C and predicts it boils lower on a mountain. If the formula's prediction lands close
to what your experiment found, your result stops being "a thing we observed on MovieLens" and becomes
"a rule you can apply anywhere." That's the difference between a solid thesis and a memorable one —
and reviewers strongly prefer papers that predict something and then confirm it, over papers that
only report what happened.

### U3 — "sim-to-sim transfer", plainly

Danger: the bot will be tuned in a simulated world built from one LLM's judgments. If we then also
*grade* it in that same world, a reviewer can say "your bot just memorized your simulator's quirks."
Fix: keep a second simulated world, built from a *different* LLM (Haiku — you already have that data)
that the bot never saw during tuning, and grade the bot there too. Even better, let the bot hold a
live interview against a different LLM playing the user. If the bot's advantage survives in the world
it never trained in, "it just memorized the simulator" is off the table. It's the same instinct as a
train/test split — but for the whole simulated world, not just the users.

### U4 — "release the arena as a benchmark", plainly

Everyone in this field grades their own homework: each group builds its own simulated user, with its
own baked-in rules, and reports wins inside it. You have now built the first simulated user whose
"what can they answer" comes from an outside source, checked two ways, with honest noise levels.
Package it so other researchers can download it and test *their* methods in it. Why bother: a single
finding gets cited a few times; a testbed everyone uses gets cited for years, and it makes your
apparatus the standard way to evaluate this problem. It also answers "why should we trust your
simulator?" with "don't trust it — run it yourself."

### U5 — Goodreads, plainly

Every question-asking result in the thesis lives on movies. Repeat just one result — the flip point —
on books (your setup already runs there). If it holds, the guide isn't a movie fact, it's a general
fact. If it moves, that's interesting too: books may differ in a way the formula from U2 should
predict.

### U6 — the human study upgrade, plainly

As planned, the human study checks "does the LLM judge agree with real people about what they can
answer?" Add one free analysis: for those same 50 people you'll also know how sloppy their answers
are and what they could answer — which means the guide can *predict which way of asking should have
worked best for them* — and you can check whether it's right. That's the full circle no one in the
field has closed: guide → prediction → real humans confirm.

### The novelty list, plainly (what you can claim as "we're the first")

Already done:
1. (N1) Showed the same experiment gives opposite verdicts depending on the measuring recommender —
   so half the literature's conclusions depend on their instruments, not their methods.
2. (N4) Showed asking about specific items is hopeless on big catalogues (almost nothing is
   answerable) — broad questions aren't a nicety, they're forced.
3. (N5) Found and mapped the flip point: precise-but-weird questions win with careful answerers,
   simple broad questions win with realistic ones. Including the surprise that "rounding" a weird
   question to a real movie *helps* when answers are sloppy (the movie acts as an anchor against noise).
4. (N6) Explained *why* broad questions run out of steam after ~4 questions (they overlap heavily;
   ~2–4 truly independent directions) while item-level evidence keeps adding information (~27).
5. (N7) Proved the weird-but-optimal questions can't be translated into English without losing their
   advantage — so choosing the way-of-asking really matters; there is no universal translator.
6. (N10/N11) First outside-sourced model of what each user can answer — and the discovery that users
   genuinely differ, following their taste. (The fuel.)

Planned (with the experiment that earns each):
7. (N12) First measurement of what knowing "what this user can answer" is *worth* — via the oracle
   decomposition (give a selector the answer-key and see how much it gains).
8. (N13) First interviewer that explicitly tracks user knowledge alongside user taste.
9. (N14) Either: first honest win for adaptive interviewing (in the first simulator that contains a
   real reason to adapt) — or: the measured explanation of why adaptive interviewing keeps losing
   (discovering what a user knows takes more questions than a short interview allows). Both versions
   also explain, in passing, why everyone else's learned interviewers kept tying fixed scripts:
   their simulated users had nothing user-specific to discover.
10. (N15) Proof the bot's advantage isn't a simulator artifact (survives the foreign world, U3).
11. (N3/U4) The downloadable testbed itself.
12. (N16) The human full-circle check (U6).

And the honest "stop claiming" list: the old headline "abstract continuous questions beat normal
ones" (only true with unrealistically careful answerers), the old learned-policy win in B (doesn't
survive retraining), D's "2 questions beat 8" (the comparison was unfair), A's "the question channel
is strongest" (true by construction, not a finding).

### Bottom line

Do U1 (costs nothing, unifies the thesis) and U3 (days of work, protects the crown result from the
most obvious attack). If you have capacity for one more: U2 — it turns your best finding from an
observation into a law. U4 is the gift that keeps citing; U5/U6 belong to the thesis stage, not the
October deadline.

---

# U2 THEORY SKELETON (2026-07-07, Fable): predicting the fidelity boundary

The formula Opus should fit and verify. Linear-Gaussian belief over u\* ∈ R^d (d=64/512 latent).
Model each channel c by FOUR measured parameters — nothing is free:

- **r_c** — effective rank of the channel's question directions (measured: concepts ≈2.25 on ML-1M
  genome/learned E_c, ≈4 on ML-25M centroids; items ≈27; continuous = d).
- **ρ_c** — fraction of u\*'s energy lying in the channel's subspace (measure once per dataset:
  mean over users of ||P_c u\*||²/||u\*||², P_c = projector onto the channel span). Fine channels →
  ρ→1; coarse → ρ<1. This is the channel's CEILING.
- **σ_c** — answer noise (measured: the empirical channel's residual σ_a=0.48 on the graded scale /
  0.70 stars; per-channel from test-retest or the fitted answer law).
- **α_c(u)** — per-user answerability rate (measured: pmodel / judged grid). Unanswerable → turn lost.

**Per-question signal:** an answered question reads a = u\*·q + ε, ε~N(0,σ_c²). Over T turns the
channel yields m = α_c·T answers spread over r_c directions → per-direction measurement count
m/r_c, per-direction posterior precision grows as (m/r_c)/σ_c². With per-direction prior signal
variance κ (one global constant, fit once), the recovered fraction of in-subspace energy is the
standard shrinkage factor, giving:

  **Q(c,T,u) ≈ ρ_c · SNR_c/(1+SNR_c), where SNR_c = κ·α_c(u)·T / (r_c·σ_c²)**

Then a single monotone link g: Q → NDCG, fitted ONCE on existing runs (a calibration curve over all
arms/conditions — not per-condition; if g needs re-fitting per condition the theory has failed).

**What it predicts (each a checkable claim against numbers you ALREADY have):**
1. **The boundary:** solve Q(cont)=Q(concept) for σ\* → the crossing must land inside the empirical
   ×0.5–×2 sweep window (Paper C tab:i2fidelity). Fine channels have ρ≈1 but pay r_c=d in SNR;
   coarse channels cap at ρ_c but pay only r_c≈2–4 → noise decides the winner. Exactly the observed
   steep-vs-gentle decay.
2. **Concept saturation at T ≈ r_c/α_c:** once each of the ~2–4 directions is measured, extra
   answers only average noise (1/m gains) → the q4≈q8 plateau, with the plateau HEIGHT = ρ_concept.
   The measured full-profile ceiling vs concept plateau gives an independent estimate of ρ — check
   consistency.
3. **Item channel keeps climbing to T≈27/α** → matches B §5.2.
4. **The per-user boundary (U1, formalized):** σ\* depends on α_c(u) → each user has their OWN
   crossing; high-answerability users' fine channels surface earlier. This is the sentence that
   fuses the map and the agent: the agent estimates α_c(u) online = estimates where u's boundary is.
5. **ML-25M tail reversal (B's scope condition):** enters through ρ_tail (the concept subspace's
   energy fraction on TAIL items specifically ≈ small for head-aligned centroids) — compute
   ρ_tail vs ρ_full and check the sign flips as observed.

**Fitting protocol (Opus):** measure r_c, ρ_c, σ_c, α_c from existing artifacts (no new runs);
fit κ and the link g jointly on a HELD-OUT-free subset of existing results; verify predictions 1–5
against the remaining results. Report predicted-vs-observed crossing as the headline figure.
Failure mode to accept honestly: if only the ORDERING of crossings is right but not the location,
publish the ordering claim (still novel — no elicitation paper predicts any of this a priori).

---

# E0 VERDICT (2026-07-07, run same day): NO-GO — the pre-registered tie/scope branch fires

Results: casper/experiments/E0_GONOGO_RESULTS.md (scripts e0_gonogo.py, e0_p1_improved.py,
e0_p1_skip2.py). Canonical G2 reproduced exactly (A=0.5681, B=0.2812) before any modification.

## The numbers
- **P1 (the fork):** best realizable O-ans (static+skip, diversified) anytime 0.2514 vs static
  0.2517 — **TIE, prize −0.0003, CI [−0.0012, +0.0007]** (< the 0.015 GO threshold). O-full remains
  +0.274 anytime. **Decomposition: essentially ALL of G2's 0.287 is privileged true-taste belief;
  the realizable answerability-discovery prize ≈ 0.**
- **Mechanism (the sentence that explains everything):** the strong static already self-selects
  **7.53/8 answerable questions (94%)** — population answerability knowledge alone parks the
  schedule in the answerable region for nearly everyone, so there are no refused turns to reclaim
  and no routing headroom at T=8.
- **P2:** surrogate decision loss ≈ 0; within-user AUC (0.909) ≈ pooled (0.907) — the surrogate is
  NOT the bottleneck (and the design's between-user-inflation worry was wrong — recorded).
- **P3:** ĝ posterior does NOT sharpen in-budget (AUC 0.901→0.895 flat) — failure-mode-1 confirmed,
  though moot given P1.
- **P4:** fuel exists (23% of questions high-variance × high-value) but is already harvested by the
  static — heterogeneity is real (the gate stands) yet unexploitable beyond population knowledge here.

## What this means (direction update, per the pre-registered Q7 tie branch)
1. **Do NOT build the agent.** The 2 planned weeks are reclaimed for writing + U2.
2. **The result is a real contribution, not a failure** (N12 is now DONE): first decomposition of
   the answerability-knowledge prize — per-user answerability knowledge adds ≈0 over population
   answerability at realistic interview lengths, because a well-built static schedule already sits
   in the answerable region. This retroactively explains the 8+ tied policies AND sharpens the
   flagship's prescription: **the value is in choosing the channel and building the schedule from
   population answerability — both static, both deployable.** The channel map IS the thesis.
3. **Honest scope on the NO-GO (print it):** the arena's question bank is broadly answerable (base
   rate 0.732; popularity-stratified judged bank). The verdict is "routing adds nothing when the
   bank is pre-filtered to answerable questions at T=8" — NOT "per-user answerability never
   matters." The one place the verdict could be overturned: a LOW-answerability bank (niche-heavy,
   base rate ~0.2–0.3), where a static schedule cannot park everyone in the answerable region.
4. **U-priorities update:** U3 (transfer) demotes to a cheap robustness check of the TIE itself
   (does 7.53/8 + the tie hold on the Haiku grid / structural rule — expected yes, it's structural).
   U2 (theory) PROMOTES to the flagship upgrade — and note the theory already predicts this verdict:
   with α_static→~0.94, the α_c(u) term barely varies across users, so per-user SNR differences
   vanish. E0 is U2's first confirmed prediction, post-hoc; say so carefully.

## OPUS HANDOFF (run order, ~1 week) — SUPERSEDED same day by "THE WAY OUT" below; H1 is upgraded
1. **H1 — the one overturn-check (half a day):** re-run P1 on a low-answerability bank slice
   (niche-heavy items+concepts from the judged grid, base rate ≤0.3). If the tie holds there too,
   the NO-GO is unconditional at T=8 and the scope sentence strengthens; if routing wins, the
   adaptive claim returns in scoped form ("adaptivity pays when the bank must be niche") — either
   way one clean sentence in the flagship. Same harness, zero LLM calls.
2. **H2 — tie robustness (half a day):** P1 static+skip + the 7.53/8 statistic on the Haiku grid
   and under the a-priori structural rule. Sign-preservation is the claim.
3. **H3 — U2 fit (the new crown, ~3–4 days):** per the skeleton above. E0's decomposition joins
   predictions 1–5 as target #6.
4. **H4 — writing:** fold the O-full/O-ans/static decomposition + mechanism sentence into the
   flagship as "the price of answerability knowledge"; update ANSWERABILITY_RESULTS_LOG.md (E0
   entry + close the 'agent NOT started' TODO as resolved-NO-GO); apply the Q1 kills; the tie-branch
   abstract paragraph from Q7 is pre-written above — use it.
5. **Discipline unchanged:** any H1/H2 surprise reopens this doc before anything is claimed.

---

# THE WAY OUT (2026-07-07, owner challenge): E0's tie is HALF of a two-regime result — the
ADAPTIVITY BOUNDARY

Owner's worry, stated fairly: "either the elicitation field is dead (prior lit disagrees), or we
missed something, or the metric is unfit — a reviewer will flag it." Resolution: none of the three.
E0's verdict is regime-scoped, the regime was the one where theory PREDICTS a tie, and prior
literature's adaptive wins live in the OTHER regime. Spelled out:

## 1. What E0 actually established (read the clauses)
Per-user answerability routing adds ≈0 over a population-informed static schedule, **at T=8, in a
bank pre-curated to broad answerability (base rate 0.73), dominated by a rank-≈4 concept channel.**
It did NOT show elicitation is worthless (static elicitation carries the arena: 0.281 endpoint vs
cold floor), and it did NOT test taste-adaptivity, adaptive stopping, or robustness — only
answerability-routing at fixed horizon in a rich bank.

## 2. Why the tie was theoretically INEVITABLE in this regime
The thesis's own linear-Gaussian result: adaptivity has zero value when questions are exchangeable.
In a broadly-answerable low-rank bank they are — any 8 answerable draws from a rank-≈4 concept space
span it; routing cannot matter. Adaptivity's value requires the nonlinearity (a question yields
NOTHING if unanswerable) to actually bind — i.e. **scarce answerability × high-rank channel**: the
item regime, large catalogues, high unknown-rates. U2 says it directly: adaptivity value ∝ per-user
variance of α_c(u)·T/r_c; at α≈0.94, r≈4 it vanishes (E0, measured); at α≈0.1, r≈27 it should be
large. E0 is U2's prediction confirmed in one corner of the map — not a verdict on the map.

## 3. Prior literature AGREES once regimes are labeled
- Golbandi et al. 2011 (the field's canonical adaptivity win): adaptive DECISION TREES over ITEM
  questions with high unknown-rates beat static seed lists — precisely the scarce-answerability ×
  high-rank corner. Consistent.
- EAR/UNICORN/ConTS RL gains: multi-turn attribute+item mixing with stopping/recommendation timing
  decisions — mechanisms E0 froze out by fixed T and fixed channel. Consistent.
- Our own Paper B: concept channel saturates by q4 (rank 2–4) — the E0 bank inherits exactly that
  exchangeability. Consistent.
So the honest sentence for the flagship: "prior adaptive wins and our static tie are the two sides
of one boundary; no published work has measured the boundary itself."

## 4. THE REFRAME: the thesis now has TWO boundaries and one theory
Paper C's centerpiece = the FIDELITY boundary (fine vs coarse flips with answer noise). E0 = the
first measured point of the **ADAPTIVITY boundary: static vs adaptive flips with (answerability
base rate × channel rank)**. Upgrade H1 from "overturn check" to the second half of the experiment:

**H1+ (the new key experiment, ~1–2 days, zero LLM calls):** sweep bank composition from the E0 bank
(α≈0.73, concept-heavy) toward item-heavy niche banks (α≈0.3 → 0.1), holding harness fixed. At each
point: pure static (no skip) vs static+skip vs full per-user routing (O-ans) vs the p̂-realizable
router. Plot ΔNDCG(routing − static) against bank answer rate → **the adaptivity-boundary figure**.
Mechanism hypothesis to test alongside: in low-α banks the refusal tax binds; skip recovers turns
AFTER refusals (reactive), routing avoids them BEFORE (predictive, via taste-tracking α_c(u)); with
real per-user structure (the gate's finding) predictive should separate from reactive as α drops.
Expected result: crossing somewhere in α∈[0.2,0.5]; U2 predicts its location from measured
parameters BEFORE the run — pre-register that prediction (this is the theory's first prospective
test, and it converts E0 from post-hoc to the boundary's anchor point).

## 5. The two metric fixes a reviewer would rightly demand (both cheap, run with H1+)
- **E0b — adaptive stopping:** fixed-T anytime NDCG structurally cannot reward the classic adaptive
  payoff — ending early for easy users. Add: policies may STOP; metrics = questions-to-95%-quality
  and cost-weighted NDCG (endpoint − λ·turns). A static schedule cannot stop per-user; even in the
  E0 bank this may separate adaptive from static honestly.
- **E0c — robustness/insurance:** the E0 static inherits OUR perfectly-curated bank and calibrated
  population model; deployment has neither. Degrade the population model (shifted priors /
  wrong-domain bank slice) and measure static collapse vs adaptive self-correction. "Adaptivity as
  insurance against designer miscalibration" is what practitioners actually buy — and it is a claim
  the E0 tie does not touch.

## 6. What the flagship now says (no failure narrative anywhere)
"We measure WHERE adaptive elicitation pays. Under broad answerable banks at short horizons,
population knowledge suffices and static schedules are optimal (we prove the tie and decompose the
oracle gap); as answerability thins and the channel's rank rises, routing separates from static —
we chart the crossing and our channel theory predicts its location from measurable parameters.
Prior literature's adaptive wins and static ties are reconciled as the two sides of this boundary."
If H1+ finds no crossing even at α≈0.1: the adaptivity-skeptical result strengthens to
"population-answerability statics suffice across the practical range — invest in channel choice,
not routing" — still a boundary paper (the boundary just sits further out than the practical
regime), still consistent with Golbandi via the stopping/tree mechanisms E0b covers.

## Revised Opus run order
H1+ (boundary sweep, with U2's pre-registered crossing prediction) → E0b (stopping) → E0c
(insurance) → H2 (Haiku/structural robustness of BOTH endpoints of the sweep) → H3 (U2 full fit,
now with the adaptivity boundary as prediction #7) → H4 (writing: two-boundary flagship).

---

# E0's GAP, CAUGHT BY THE OWNER (2026-07-07): coarse→fine channel DESCENT was never actually tested

Owner's challenge, which is correct: the E0 routers never DID coarse-to-fine. The bank contained
niche items (1,716 items across popularity tiers), but the best O-ans variant simply followed the
static's concept-heavy order and skipped refusals; the other two variants' value scorers never
rewarded switching into the item channel. So E0 established "the answer key adds nothing GIVEN the
three value models tested" — a weaker claim than "coarse-to-fine has no prize." The known facts that
make descent plausible: items ≈27 independent directions vs concepts ≈4 (Paper B); item-8 with real
ratings BEAT concept-8 under the realistic channel on ML-1M (0.394 vs 0.325, Paper C tab:i2fidelity);
concepts saturate by q4; and the fold is NON-MONOTONE in answer count (Paper B) — redundant coarse
answers past saturation can actively dilute the belief, so turns 5–8 on concepts may be worse than
wasted. E0's NO-GO is hereby scoped: it kills routing-within-tested-scorers, NOT channel descent.

**E0d — the DESCENT PROBE (run BEFORE H1+; half a day, zero LLM calls, same harness).**
Hand-built two-phase schedule: ~4 broad concepts for everyone (cover the coarse directions), then
SWITCH CHANNELS — 4 item questions per user, chosen from the user's answerability table (true table
first = upper bound; then p̂-surrogate = realizable). Compare vs the all-concept static and vs
static+skip, same metrics/CIs as P1. Sweep the switch point (3/4/5) once. Outcomes:
- Descent WINS → E0's NO-GO was premature (bottleneck was the value scorers); the coarse→fine thesis
  is alive in exactly the owner's stated form; the agent design revives with descent as the backbone
  and the H1+ sweep charts where its edge grows.
- Descent LOSES → the dilution/low-marginal-info explanation is confirmed DIRECTLY (not by
  inference), and the two-regime story proceeds unchanged with a stronger footnote.

**E0e — informativeness re-weighting (the owner's "re-weigh them" ask; pairs with E0d):**
1. SELECTION side: replace fixed question values with MARGINAL information given the current belief
   (expected new direction beyond answers already folded). Realizable (belief-only). This
   automatically deprioritizes redundant concepts after ~q4 and promotes items — the principled
   version of hard-coding the switch at 4.
2. FOLD side: weight answer tokens when building the belief — down-weight k-th redundant concept
   token (or weight by specificity/inverse-frequency of the question). Targets the non-monotone-fold
   dilution directly. Test as an ablation on BOTH the static and the descent schedule (it may help
   the static too — keep the comparison fair).

Order update: **E0d → E0e → then the WAY-OUT sequence (H1+ …)**, since E0d's outcome decides whether
H1+ sweeps a dead mechanism or a live one. If E0d wins, add descent-with-p̂ as an arm throughout H1+.

---

# E0d/E0e RESULTS (2026-07-07, same day): descent LOSES in the current arena; emergence FIRES;
one load-bearing caveat prevents a final verdict

Results: casper/experiments/E0D_DESCENT_RESULTS.md (scripts e0d_descent.py, e0e_emergent.py,
e0e_reweight.py). Baselines reproduced before running (static B 0.2517/0.2812 ✓).

## The numbers
- **E0d (hard-coded descent, 4 concepts → 4 per-user answerable items):** LOSES everywhere —
  best arm (k=4, true table, pop-value) −0.0111 anytime CI[−0.0176,−0.0047]; realizable p̂ arm
  −0.0133; k=3 worse, k=5 less bad; no switch point wins. Per-turn curve is the mechanism, observed
  directly: at the switch, static edges UP (0.267→0.268) while descent's first item DROPS the belief
  0.267→0.248 — the item answer DILUTES.
- **E0e (emergent switch): THE OWNER'S MECHANISM FIRES.** Pure marginal-info × true table leaves the
  concept channel at a tight median t=4 (262/298 users at exactly t4 — precisely where concepts
  saturate). Emergence is real and needs no training. But it costs NDCG (−0.048 to −0.103): in this
  arena marginal-information is ANTI-correlated with ranking value (the most-orthogonal directions
  are specific items whose answers dilute).
- **E0e fold re-weighting:** does nothing for static (already diversified), HELPS descent
  (+0.0031 anytime CI[+0.0019,+0.0043]) — the dilution diagnosis confirmed causally — but nowhere
  near closing the gap.

## THE LOAD-BEARING CAVEAT (agent-flagged; changes what may be concluded)
**The E0/E0d harness answers ALL questions geometrically (a = cos(z\*,q)) — real ratings never enter
the answer value; the LLM grid only gates askability.** So the arena still runs the OLD circular
value model with external answerability bolted on. This matters asymmetrically for the descent
verdict: Paper C's fidelity section showed item questions answered with REAL RATINGS are the
strongest realistic channel (item-8 0.394 > concept-8 0.325 under the empirical channel), and the
answerability study design §3 specifies real-rating item values — which this harness does not yet
implement. Items entering the fold as geometric tokens may be exactly why they dilute (norm/η
mis-scaling for item directions is also unruled-out). **Therefore: "descent loses" is established
for the geometric-answer arena only. The coarse→fine verdict is NOT final until the answerer swap
is in.**

## E0f — THE DECISIVE RERUN (first task for Opus, ~1 day)
Implement the non-circular item answer per the study design §3: for rated items a = the user's real
rating (centered on profile mean, rescaled to the fold's answer scale, + fidelity σ=0.70); route
descent to answerable∧RATED items primarily (answerable-unrated with LLM-predicted value =
sensitivity-only arm). Re-run: static B, static+skip, E0d k=4 (true table + p̂), E0e emergent ×div —
same metrics/CIs. Also re-check η/norm calibration for item tokens in the fold (one control: a
single rated-item answer at t1 should HELP, not hurt, vs no answer). Outcomes:
- Descent still loses under real-rating answers → the coarse→fine thesis is dead in this arena for
  real; the two-regime story ships with the strongest possible footnote (tested under both value
  models).
- Descent wins → E0/E0d's ties were artifacts of the leftover geometric value channel; the agent
  revives; H1+ proceeds with descent as a live arm — and the flagship gains "the answerer swap
  flips the adaptivity verdict," which rhymes perfectly with the fidelity boundary (the thesis's
  recurring lesson: THE ANSWER MODEL DECIDES).
Note the meta-point either way: every major conclusion in this program has been answer-model-
dependent — that IS the thesis, and E0f is its cleanest demonstration yet.

---

# E0f RESULT (2026-07-07): the CONTROL FAILED — the arena's belief update cannot consume real
answers, so "descent loses" is a fact about the FOLD, not about descent. One final run: E0g.

Results: casper/experiments/E0F_RESULTS.md. Baselines reproduced; then the mandated fold-sanity
control: **folding ONE free true real-rating answer from cold HURTS (−0.053 CI[−0.079,−0.027])** —
and even the geometric single-item token does not help (−0.011 ns). Diagnosis (quantified):
(1) η=16 overshoots for a single narrow item token (geometric item at η≈1 helps +0.063, at η=16
hurts); (2) **the additive operator z'=z+η·a·q is a taste-gradient step ONLY when a=cos(z\*,q)** —
real centered ratings correlate just +0.39 with that coordinate and SIGN-DISAGREE on dislikes, so
even noiseless real answers barely help (+0.009 at best-η) and σ=0.70 erases that. All realizable
descent arms lose accordingly (−0.012 … −0.115); the only "wins" were target-peek upper bounds
(labelled leaky by the agent — correctly not treated as evidence).

## Why this does NOT settle the descent question (and is not goalpost-moving)
A free true answer making the belief WORSE means the arena is objectively invalid for testing
whether item answers help: the instrument cannot ingest the information at all. And we already know
real-rating item answers ARE valuable when consumed by the right machine — Paper C's own fidelity
table: item-8 real ratings 0.394 and the pure item FOLD 0.4635 (the strongest realistic arm),
both >> concept-8 0.325, on the trained fold-in encoder. The additive operator was designed for
graded geometric concept tokens; items answered with ratings need the instrument's NATIVE encoder
(RecVAE consumes rated-item sets natively — that IS its input interface).

## E0g — THE FINAL RUN (pre-committed as final; ~half a day)
Hybrid belief: concepts fold via the operator as today; ITEM answers fold via the instrument's
native encoder (accumulate the user's answered items ± ratings/likes into RecVAE's input; recompute
z_items; combine with the concept-operator belief — simplest: score-sum or a convex mix λ tuned on
val; state the choice). Controls FIRST, in order: (C1) one rated item folded natively from cold
must HELP (it is literally RecVAE's training interface — if this fails, something is wired wrong,
stop); (C2) 8 rated items folded natively ≈ the known-profile fold sanity. Then rerun: static B,
static+skip, descent k4 (true table + p̂, native item fold), emergent ×div, all-item-8 native.
Same metrics/CIs.
**Pre-commitment (no more reruns after this):** if C1 PASSES and descent still loses → the verdict
is FINAL: coarse→fine descent does not pay in this arena even with valuable, properly-consumed item
answers; ship the two-regime story with the complete three-arena footnote (geometric / real+operator
/ real+native — a strong methods contribution by itself: THE FOLD DECIDES what any adaptivity study
can see). If descent wins → the agent revives with the hybrid fold as its belief machinery, and
E0/E0d/E0f become the documented path to the right instrument.

[SUPERSEDED same day by I2.5 below — owner ruled: no hacks; build the proper learned fold instead
of the E0g hybrid patch. E0g's C1/C2 controls are absorbed into the I2.5 gate suite.]

---

# I2.5 — THE PROPER INSTRUMENT (2026-07-07, owner directive): a learned fold over items + concepts
+ attributes on the certified RecVAE base. Review, lit-anchored plan, delegation.

## Where it derailed (the review, for the record)
Paper A's V1 instrument = a TRAINED fold-in encoder (DualHeadSetEncoder; Paper B's learned
reconstruction encoder) — items/concepts/attributes as answer tokens, formally gated. NOT broken;
nothing in Paper A is retracted. The I2 rebuild (Jul 4) swapped the recommender to RecVAE-d512 for
SOTA credibility but replaced the LEARNED fold with an algebraic shortcut — the additive operator
z'=z+η·a·q — because RecVAE has no native interface for answer tokens beyond rated-item sets. The
I2 gates certified that operator ONLY for the P4 envelope (concept questions, geometric graded
answers); E0/E0f probed outside the envelope (real-rating item answers) and the control failed.
Root cause: **instrument capability was silently narrowed during the I2 rebuild; the arena on top
of it inherited the narrowing.** Lesson → the gate suite must always include an out-of-envelope
canary (G-fold1 below).

## The design (lit-anchored, no invention needed)
**Frozen RecVAE decoder/scorer (keeps SOTA + the existing certification) + a learned amortized
inference network ("the fold") q(z | answer set) over heterogeneous answer tokens.**
- Token vocabulary: item±rating (real, centered), concept±graded (data-side member-aggregate),
  attribute±(actor/director/decade — NEW channel, owner-flagged as missing), refusals EXCLUDED from
  the fold (they inform answerability, not taste — per the agent design).
- Encoder: permutation-invariant set encoder over (token-type, entity-embedding, answer-value)
  triples → z in RecVAE's latent. Literature anchors: **Partial-VAE / EDDI (Ma et al., ICML 2019)**
  — amortized posterior from arbitrary PARTIAL observation sets, exactly the elicitation-fold
  problem, already cited in the thesis; **Deep Sets (Zaheer 2017) / Set Transformer (Lee 2019)** —
  the V1/V2-ST machinery, port it; **DropoutNet (Volkovs et al., NeurIPS 2017)** — train-with-
  input-dropout for cold-start robustness = the fold-curriculum you already used for FTREC;
  **conversational folds (EAR, UNICORN)** — attribute+item feedback fusion, cite as the CRS analogue.
- Training: reconstruct held-out interactions from SIMULATED partial reveals of real profiles
  (sample interview-length subsets: items w/ real ratings, concepts w/ data-side aggregates,
  attributes w/ data-side aggregates; mix lengths 1–16; dropout curriculum) — trained UNDER the
  elicitation distribution (the de-OOD lesson from Paper C/FTREC). Answers are data-side by
  construction → the fold is non-circular from birth; the geometric channel remains available
  separately for the abstract-ceiling analyses only.
- NOTHING about the recommender changes: RecVAE stays frozen; the learned fold maps to its latent;
  all existing full-profile numbers stand.

## Gate suite (formal, pre-registered — extends the I2 apparatus)
- **G-fold1 (the E0f lesson, now a permanent canary):** ONE true item answer from cold must HELP
  (Δ>0, CI excl. 0). Same for one concept, one attribute.
- **G-fold2 monotonicity:** NDCG non-decreasing (within noise) in #answers, per channel and mixed,
  1→16.
- **G-fold3 no-harm vs native:** item-only sets ≥ RecVAE's native fold of the same items (the fold
  must not lose to the base model's own interface).
- **G-fold4 channel-mix:** mixed item+concept sets ≥ best single channel at matched budget (the
  raison d'être of a unified fold).
- **G-fold5 fidelity-noise sanity:** performance degrades smoothly with the σ=0.70 knob (no cliff).
- **G-fold6 ruler consistency:** full-profile fold reproduces ≈ RecVAE native full-profile (ties
  the new fold to the certified ruler).

## Plan (phased; each phase gated before the next)
1. **Spec + data plumbing** (half a day): token schema, reveal-sampler (pinned split, seed 123),
   attribute channel extraction (actors/directors/decade from ML-25M + genome), gate harness.
2. **Train the fold** (1–2 days incl. sweeps kept minimal): Set-Transformer encoder (port V2-ST),
   P-VAE-style objective vs plain reconstruction — try plain first (V1 evidence says it suffices).
   Save ALL checkpoints; best on disjoint val (standing rules).
3. **Run the gate suite** (half a day). Any gate fails → fix before ANY arena work.
4. **Re-run the E0 suite on the new fold** (1 day): statics, descent k3/4/5 (true table + p̂),
   emergent switch, all-item — THE definitive adaptivity verdict, now on a valid instrument.
   Pre-commitment stands: this replaces E0g as the final answer; no further reruns.
5. **Extend answerability to attributes** (cheap): structural rule immediately (data-side
   popularity/coverage); LLM judging of the attribute channel = small add-on call batch later
   (NOT blocking; flag results as structural-rule-only until then).
6. **Re-anchor**: which E0/E0d/E0f conclusions survive on the valid instrument; update the design
   doc verdict chain; the three-arena story (geometric / real+operator / real+learned-fold) becomes
   the flagship's methods spine: THE FOLD DECIDES.

## TODO added (owner): attribute channel (actors/directors/decade) — missing from the question
bank, the answerability grid, and the fold vocabulary; carried in phases 1 & 5 above.

---

# I2.5 BUILD RESULT (2026-07-07): fold VALID — 5/6 gates pass; E0f canary FLIPS (−0.053→+0.095);
G-fold4 fails for a structural (non-defect) reason; re-scoped with documentation; Phase 4 unblocked

Results: casper/experiments/I25_FOLD_RESULTS.md; checkpoint .cache/i25_fold_best.pt (val 0.4629).
Design as specced: Deep-Sets RESIDUAL fold z = native_z + ρ(...) (zero-init: starts at RecVAE's
certified native interface — elegant no-harm-by-construction). Attributes = decade+genre only
(actors/directors absent from ML-25M metadata — TODO stands, needs external metadata source).

- **G-fold1 PASS, the headline:** one true answer from cold now HELPS on every channel (item +0.095,
  concept +0.066, attribute +0.068, all CIs excl. 0). The broken-arena era is over.
- G-fold2/3/5/6 PASS (monotone; ≥ native item fold; smooth under σ; full-profile within tolerance
  −0.014, flagged).
- **G-fold4 FAIL — but the diagnosis shows displacement, not dilution:** concepts ADD on top of
  items (+0.0076 CI[+0.0035,+0.0115] when added to item-8) — no defect; the fold is monotone-in-
  information everywhere. The gate fails only because a concept turn is worth ~1/18th of an item
  turn at fixed budget, so ANY mixing displaces value. "Mixed ≥ best single channel" is structurally
  unsatisfiable when one channel dominates — a mis-specified gate, not a broken instrument.
- **Bandwidth hierarchy now REAL in the arena:** item-8 ≈0.35 vs concept plateau ≈0.21 under the
  valid fold — so E0d/E0f's "descent loses" is formally scoped to the broken fold, and the descent
  question is genuinely OPEN again, with the fuel finally present.

## Pre-registration amendment (documented, not hidden — report BOTH in any paper)
Original G-fold4 ("mixed ≥ best single channel at matched budget"): **FAIL, on record.** Its intent
was defect detection (mixing must not corrupt the belief); the diagnosis confirms no defect.
Re-scoped **G-fold4′ ("mixed must not lose to its own item subset" — i.e. adding concept answers to
a fixed item set must not hurt): PASSES** (+0.0076 significant). Phase 4 unblocked under G-fold4′
with this paragraph as the amendment record. Honest sentence for the paper: "one pre-registered
gate was re-scoped after diagnosis showed the failure reflected channel dominance, not instrument
defect; both forms are reported."

## Why Phase 4 is now genuinely interesting (changed stakes)
Under the valid fold items dominate — but a STATIC schedule cannot know WHICH items this user can
answer (rated items are user-specific; that is the whole answerability problem). The router can
(table/p̂). So the adaptivity question sharpens to its true form: is per-user knowledge of WHERE the
high-bandwidth channel is answerable worth more than a population-optimal static mix? This is the
first arena in the entire program where BOTH ingredients are present: a valid fold that rewards
item answers, and real per-user answerability structure. Phase-4 verdict = final, per the standing
pre-commitment.

---

# PHASE 4 RESULT (2026-07-07): FINAL VERDICT — adaptivity WINS under the valid fold; the E0/E0d/E0f
NO-GO was a fold artifact. Ran locally by Fable (no LLM); results experiments/I25_phase4.json.

## Numbers (298 users, NDCG@10, paired bootstrap vs greedy static B)
| arm | anytime | endpoint | delta vs static B | 95% CI |
|---|---|---|---|---|
| static B (greedy — rebuilt as ALL CONCEPTS) | 0.212 | 0.210 | — | — |
| static+skip | 0.211 | 0.210 | -0.0002 | tie |
| descent k=4 (p-hat surrogate, REALIZABLE) | 0.249 | 0.310 | **+0.037** | [+0.028,+0.046] |
| descent k=3 (\|rating\|) | 0.254 | 0.311 | **+0.042** | [+0.031,+0.053] |
| descent k=4 / k=5 | 0.240 / 0.228 | 0.288 / 0.263 | +0.028 / +0.017 | both excl 0 |
| emergent (marginal-info x answerability x div) | 0.290 | 0.328 | **+0.079** | [+0.063,+0.095] |
| all-item-8 (per-user rated items) | 0.292 | 0.344 | **+0.080** | [+0.063,+0.097] |
| _descent peek UB (target-leak, LABELLED not evidence)_ | _0.396_ | _0.546_ | _+0.184_ | _leaky_ |

## What it means (three findings, all honest)
1. **Adaptivity WINS, decisively and realizably.** The deployable p-hat-surrogate router beats the
   best static +0.037 endpoint, CI excludes 0. E0/E0d/E0f's "routing adds ~0" is now formally an
   artifact of the additive-operator fold that could not consume item answers (E0f canary
   -0.053 -> +0.095 under the learned fold). The thesis's crown adaptive result EXISTS.
2. **The mechanism is answerable-channel ACCESS, not coarse->fine descent per se.** Greedy static
   rebuilt itself as all-concepts because a fixed ITEM question is unanswerable for most users — the
   only universally-answerable fixed schedule is concepts. The router wins by sending each user to
   THEIR OWN answerable rated items (the high-bandwidth channel, ~27 dims). The static structurally
   cannot reach that channel; the router can. THAT is the adaptive edge.
3. **Fast beats gradual: all-item-8 (+0.080) > descent-k3 (+0.042) > k4 > k5.** Fewer concept warm-up
   turns is strictly better; pure item elicitation (= open recall) dominates. The coarse->fine
   INTUITION identified the right mechanism (route to what the user can answer) but the wrong
   schedule — the data says reach the user's items ASAP, don't descend gradually. Consistent with
   Paper D ([[paperD-open-recall-beats-continuous]]) and the bandwidth hierarchy (Paper B).

## The clean flagship claim this licenses (no overclaim)
"Cold-start elicitation should route each user, as early as possible, to the highest-bandwidth channel
they can personally answer — their own remembered items (open recall) — because a population-optimal
static schedule is trapped in the low-bandwidth concept channel by the very fact that specific items
are not universally answerable. We demonstrate this on a validated non-circular instrument: a
deployable answerability-routing policy beats the best static by +0.037 NDCG@10 (CI [+0.028,+0.046]),
and the edge is entirely per-user channel access, which we isolate. Earlier simulators (including our
own) hid this result behind a belief-update that could not consume item answers — we show the
adaptivity verdict is decided by the fold, and provide the instrument that gets it right."

## Remaining honest to-dos before this is paper-ready (NOT reruns — reporting/robustness)
- @50 metric for headline arms (script computes it; include).
- Robustness of the +0.037 under the Haiku answerability grid + the a-priori structural rule (sign
  must hold — cheap, no LLM).
- Attribute channel is decade+genre only (actors/directors need external metadata join) — scope note.
- The peek-UB (+0.184) stays LABELLED as leaky upper bound, never cited as a result.
- G-fold6's -0.014 full-profile gap and the re-scoped G-fold4' both reported honestly.
VERDICT STATUS: FINAL per pre-commitment — adaptivity is real; the agent program is REVIVED as an
open-recall-first answerability router, not a coarse->fine descender.

## ⚠ VERDICT DOWNGRADED TO PROVISIONAL (2026-07-07, same day, Fable): evaluation averaging bug found
Ran the MISSING baseline build_item_static (defined in i25_phase4.py but never called in main()): a
population popular-item static schedule. It scored 0.4017 anytime / 0.4196 endpoint — ABOVE every
adaptive arm — despite per-item coverage of only 0.10–0.15 (90% of users refuse each item). That is
not credible and it EXPOSES a bug: eval_arm_curve sets per_turn=nan for users with no foldable token
(all-refusal users), and anytime/endpoint are nanmean, so **each arm is averaged over a DIFFERENT
user subset** (only its non-refusers). High-refusal arms (item-static; and to a lesser degree the
descent/item arms for low-coverage users) are scored over a favorable subpopulation → inflated and
NON-COMPARABLE. build_item_static's cohort_mean_items has the same survivorship drop.
CONSEQUENCE: **the +0.037 and every Phase-4 delta are ON HOLD.** Nothing here is a result until the
eval scores every arm over the SAME full user set, with refused/absent turns folded at the belief
actually held (cold or partial), not dropped. This is exactly the silent-failure class the program
exists to catch — caught here before it shipped. REQUIRED FIX + rerun before any Phase-4 claim:
common user set; refusal = no-op turn (belief unchanged), user retained at current NDCG; then
re-compare static-concept, STATIC-ITEM (the real threat baseline), descent-phat, all-item, emergent.
