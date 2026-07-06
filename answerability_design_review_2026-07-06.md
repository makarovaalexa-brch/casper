# Answerability study design — Fable review (2026-07-06)

Review of `answerability_study_design.md` (+ `scripts/llm_answerability_pilot.py`). Verdict up front:
the skeleton is right (external judge, fixed cached split, dual-model robustness, gate-before-main),
but there are **two structural holes** that would let a committee re-run the circularity attack in a new
form, **one missing gate ingredient** that could make the gate pass while the thesis still has no fuel,
and several fixable spec gaps. §7 Q-A–Q-F answered at the end.

---

## HOLE 1 (structural): the LLM value channel re-introduces circularity through a shared prior

For unrated-iconic items the LLM predicts a rating from the known profile. But the LLM's prediction is
essentially a collaborative-filtering prior learned from the same public opinion-space (reviews,
ratings-talk) that RecVAE learns from ML-25M. The cross-check — EASE trained on the known portion — is
*also* a CF prior over the same dataset. So the "two independent sources" **agree for the same reason
the evaluated instrument finds them easy**: all three encode the same collaborative structure. Agreement
between LLM and EASE therefore does NOT establish independence from the instrument under eval. This is
geometric circularity swapped for **shared-prior circularity** — softer, but a sharp reviewer will name it.

**Fix (cheap, and you get it almost for free): masked-rated-item validation.**
Take items the user DID rate in the known portion, MASK them from the shown profile, and have the LLM
predict their ratings. Now you have ground truth for exactly the "predict a rating for an item not in
the shown profile" task. Report MAE/correlation for the LLM predictor and the CF predictor on masked
items; pick the primary empirically; use the measured error as that channel's fidelity σ (see Hole 4).
Then the honest caveat writes itself: "values for never-rated items are counterfactual; both predictors
achieve MAE=… on masked rated items; headline claims are restricted to channels with real ground truth
(rated items, pairs), never-rated-item values appear only in sensitivity analyses."

**Rule to add to §4:** LLM-predicted values are *sensitivity-only*. No headline number may depend on them.

## HOLE 2 (structural): the gate can pass for the wrong reason — LLM stereotyping ≠ human knowledge structure

The fuel gate tests whether LLM-judged answerability has per-user heterogeneity that tracks taste. But an
LLM can produce exactly that pattern by **genre-matching the shown profile** (a Horror-heavy profile →
"yes" to horror questions) — i.e., stereotyping, not a model of what humans actually know. The gate as
specified cannot distinguish "the LLM models human knowledge" from "the LLM pattern-matches the prompt."
If the adaptive agent later wins by exploiting that same pattern, the win is about the LLM's
prompt-sensitivity, not about users. This is the successor to the "adaptive agent rediscovers what we
coded" trap the owner already caught once.

**Fix: the held-out-half validity check. It is a MEASUREMENT, not a pass/fail requirement on the LLM.**

Why this check, spelled out: for rated items answerability is trivially yes — the LLM isn't needed there.
The LLM's *entire job* is the extrapolation: judging questions about things NOT in the shown profile
(unrated iconic items, concepts). Every question the interview will ever ask relies on that
extrapolation. So the question the gate must answer is: does the judge carry any real USER-SPECIFIC
signal on that job, or is it just outputting popularity + genre-match — which the a-priori structural
rule already computes without an LLM?

The held-out half (which the pilot computes and never uses) is the only place this is testable, because
it is the only set of items with ground-truth "this user knows it" (they rated it) **while the LLM
cannot see it**. The check:
- Take held-out-rated items and never-rated items, **matched on popularity and genre**, and compare
  judged answerability on the two sets, **within popularity tiers**.
- **Gap exists** → the LLM infers real user-specific knowledge from taste in the visible half ("rated
  ten 90s slashers → probably knows this other 90s slasher"). That is literally the scandi-noir-fan
  hypothesis, tested. The fuel is real.
- **Gap ≈ 0** → the judge's output carries nothing beyond popularity+genre features. Not a disaster —
  but then the "external LLM judge" is decorative (the structural rule yields the same environment for
  free), and an agent "discovering per-user answerability in conversation" has nothing user-specific to
  discover. Essential to know BEFORE building the agent.
- **Partial gap is the most likely and most useful outcome:** plausibly the LLM cannot know about
  obscure items absent from the visible ratings (owner's intuition — likely correct). Then the gap will
  be small on obscure items and larger on taste-adjacent/popular ones. That is not the check failing —
  it is the check telling you **where the fuel lives** (e.g., concept/taste-adjacent level, not
  niche-item level), which directly shapes what the adaptive agent should even try to exploit.

Honest caveat to print with the result: never-rated ≠ "doesn't know" (the user may know it and never
rated it), so the gap is a ONE-SIDED signal — a positive gap proves user-signal exists; a zero gap is
ambiguous at high popularity (never-rated famous items are probably known too). Hence the within-tier
analysis: the low/mid popularity tiers are the diagnostic ones. Costs zero extra LLM calls — the
held-out items just join the question battery.

Also keep the two-model-families run (already planned); the disagreement rate is itself a reportable
uncertainty on the judge.

This turns the gate from "does the LLM emit heterogeneity" into "does the LLM emit heterogeneity that
is *about the user*" — which is the thing the thesis needs.

## HOLE 3 (gate design): statistical fuel ≠ exploitable fuel — add a second gate stage

G1 (the statistics) answers: *do users differ in what they can answer, in a way that tracks taste?*
Suppose yes. That still does not mean an adaptive agent gains anything, because the agent only profits
if knowing a user's answerability **changes which question is best, AND the change improves NDCG**.

Concrete failure mode: every user can answer the top handful of broad concept questions, and the
per-user differences live only in niche questions that carry little recommendation value anyway. G1
passes — heterogeneity real, taste-tracking, statistically solid — but the best FIXED schedule ("ask
the 5 broad concepts") is already optimal for everyone, and adaptivity gains ~0. This is exactly the
pattern the 8 failed policy attempts and the R2 static-schedule tie would predict.

**G2 — the exploitability diagnostic.** One cheap simulation, no new LLM calls, on the cached grid:
- **Selector A (answerability-aware):** per user, greedily picks the best question *given that user's
  judged answerability table*.
- **Selector B (answerability-blind):** picks the one best fixed schedule using only population-average
  answerability — same schedule for everyone.
- **ΔNDCG(A − B) is an UPPER BOUND** on what any realizable adaptive agent can extract from
  answerability discovery (A gets the answerability table for free; a real agent must spend questions
  learning it).

If Δ ≈ 0 → do not build the agent; the adaptive clause dies cheaply and the flagship ships as the
static channel map. If Δ is material → the prize exists before 2 weeks of training are spent, and Δ
becomes the **oracle-headroom number the eventual agent is measured against** — the same oracle-ladder
discipline used everywhere else in the thesis.

Pre-register both stages, with the decision rule: the adaptive claim requires G1 AND the Hole-2
validity gap AND G2 > 0 with CI excluding 0.

## HOLE 4 (spec gap): LLM-predicted values have no fidelity model

§3 gives real ratings a test-retest σ, but LLM-predicted values (unrated items, LLM-rated concepts) are
currently injected noiselessly — which quietly makes the counterfactual channel *higher-fidelity* than
the ground-truth channel. Backwards. Fix: their noise = the measured masked-item MAE (Hole 1's
validation supplies it). One sentence in §3, but without it a reviewer catches the inversion.

## HOLE 5 (stats): one fixed answerer split → split-level variance is unmeasured

Caching one canonical split is right for cost, but then every downstream CI is conditional on one random
split, and answerability judgments are split-dependent (§5 says so itself). You can't seed-average this
away on the recommender side. **Fix without exploding cost:** run 2–3 alternate splits on a ~50-user
subset, measure the split-sensitivity of the judgments (agreement rate / per-user answer-rate sd). If
low, the single-split design is defensible *with that measurement printed*. If high, you've learned the
judgments are fragile before building on them.

## HOLE 6 (protocol): held-out targets must be un-askable in interviews

Once the LLM can predict ratings for items outside the shown profile, an interview that asks about a
HELD-OUT TARGET gets an answer approximating the target's true rating → leak into the eval. §5 forbids
showing targets to the answerer but doesn't forbid *asking about* them. Add the rule: the interview
question bank excludes held-out target items for that user; the masked-item validation (Hole 1) runs as
a separate calibration pass, never inside evaluated interviews.

## Smaller holes / spec gaps

1. **"Meaningful answer" is ambiguous in the prompt.** For items it should mean "has seen it / has a
   real opinion"; for concepts "familiar enough to state a preference." Define per channel; the pilot's
   single phrasing conflates recognition with opinion-holding.
2. **"maybe" thresholding unspecified.** Pre-register: primary = maybe→refuse (conservative);
   sensitivity = maybe→yes. This moves answer rates a lot; don't leave it as a free parameter.
3. **Batched calls have order/consistency effects.** 60 questions in one call lets earlier answers
   shape later ones. Fine for cost, but measure batch-vs-single-question agreement on a small subset and
   report it.
4. **Pair answerability = "both items answerable" is an assumption, not a fact** (comparisons can be
   easier than absolute judgments — that's half the pairwise-preference literature). Keep it, but state
   it as an assumption and spot-check with a small LLM pair battery (~200 calls).
5. **Pilot ran on ML-1M; v1 arena is ML-25M.** Different era, popularity distribution, and profile
   shapes. Re-run the mini-pilot on ML-25M before spending the gate budget — a day, cents.
6. **Pilot script nits** (matter if it grows into the gate script): profile is labeled "liked/rated
   films" but includes low ratings (mislabel → judge may read everything as liked); half-profile is
   capped at 40 titles (for big users "half" is actually much less — say "a sample of size n of N");
   `response_format json_object` expects an object while the prompt demands a list; no temperature
   pinning / model-snapshot pinning for reproducibility; usage is not actually logged yet (§6 promises it).
7. **Gate sampling:** 300 users must be stratified by profile size AND taste cluster with ≥20 users per
   dominant-genre cell — the pilot's 1-user-per-genre design confounds user identity with taste, so
   taste-tracking is unpowered at anything like that shape.
8. **Model pinning:** pin the exact model snapshot for the cached grid; a mid-study model update
   invalidates the cache silently.

---

## §7 ANSWERS

**Q-A (concept value: LLM-rated vs data-side aggregate).**
Data-side aggregate = PRIMARY, tiered: use the mean of the user's real ratings over member items
(tag/genre membership, data-side) whenever ≥k members are rated; when the LLM says answerable but <k
members are rated (knows-but-never-rated), fall back to LLM-rated, flagged. Rationale: keep the LLM
confined to answerability — the thing genuinely missing from data — and keep the value channel maximally
grounded; every LLM insertion into value widens the shared-prior attack surface (Hole 1). Note the
pleasant coherence: "≥k rated members" is also the structural answerability rule, so answerable concepts
are exactly the ones with computable data-side values. Report membership-definition sensitivity
(tags vs genres). Does it change the item-vs-concept headline? Run both; claim the headline only where
both agree — that's the dual-model robustness discipline applied to value.

**Q-B (item answerability at scale).**
You don't need all ~60k ML-25M items — you need answerability over the ITEM-QUESTION BANK, and no
deployed system asks about uniformly-random obscure items anyway. So: (1) restrict item questions to a
designed bank (~500–2000 items: popularity strata × genre coverage + per-user taste-adjacent samples);
judge-all on the bank is feasible (~300 users × ~200 items each, batched). (2) Simultaneously FIT
P(answerable | features) on the judged sample — features: log-popularity, ratings-count, release
decade, genre-match-to-profile, franchise/sequel flag; logistic is enough; validate on HELD-OUT USERS
(fit on training users only — no leakage into eval users) with AUC + calibration curve. That fitted
model gives you scale (score any item) — but see Q-D for what it may NOT be used as.

**Q-C (unrated-item rating source).**
Both-with-sensitivity, but make it empirical rather than a shrug: the masked-rated-item validation
(Hole 1) measures both predictors against real ratings, picks the primary by MAE, and supplies the
fidelity σ (Hole 4). LLM-in-the-value-channel is acceptable ONLY as a sensitivity-labeled counterfactual
channel that no headline depends on. Caveat to print: "never-rated-item values are counterfactual and
unverifiable per-item; we validate both predictors on masked rated items (LLM MAE=…, CF MAE=…) and
report all affected results under each source."

**Q-D (structural model final form).**
The current text contains a trap: **if you fit the structural model's thresholds to the LLM judge, it is
no longer an independent robustness witness** — you'd have two models, one derived from the other, and
the dual-model argument collapses. Keep TWO structural artifacts with distinct roles:
- **(i) A-priori rule (the robustness witness):** items: rated OR popularity>τ; concepts: ≥k rated
  members. Set τ, k BEFORE looking at LLM output — anchor τ to something external (recognition-rate
  literature, or a round pre-registered value like top-5% popularity) — at most match the overall SCALE
  (global answer-rate), never the per-user structure. Simple and blunt is the point.
- **(ii) The fitted P(answerable|features) from Q-B (the scale surrogate):** LLM-derived by
  construction; use it to extend judgments to unjudged items; NEVER cite it as the independent model.
Headline claims must hold under (i) and under the LLM judge. (ii) is plumbing.

**Q-E (fuel-gate statistics).**
Replace the arbitrary r>0.3 with a pre-registered two-part test:
- **Heterogeneity beyond popularity:** mixed-effects logistic — can_answer ~ log-popularity + breadth +
  (1|user) + (1|question); likelihood-ratio test that user random-effect variance > 0, and report the
  variance (an ICC), not just p — with 300×~60 the p-value alone will be trivially significant.
- **Taste-tracking:** add a user×question TASTE-MATCH covariate (cosine between the user's known-profile
  genre/tag distribution and the question's genre/tag vector); test its coefficient > 0 and pre-register
  a minimum effect size — e.g., odds ratio ≥1.5 per sd of taste-match, or ΔAUC ≥0.05 over the
  popularity-only model. Report raw effect sizes regardless of the cutoff.
- **Validity gate (from Hole 2):** held-out-half vs matched never-rated gap must be positive and
  material (pre-register, e.g., ≥15pt answer-rate gap) — otherwise the judge is popularity+genre
  features in a trench coat.
- **Exploitability (G2, from Hole 3):** answerability-aware vs answerability-blind greedy selection in
  the judged environment; the adaptive-claim gate is G1 AND validity AND G2 > 0 with CI excluding 0.
Pre-register the cutoffs in the design doc BEFORE the run (you're 90% there — this doc is the
pre-registration; freeze it, date it, commit it).

**Q-F (de-risk vs de-circularize).**
Confirmed, and worth stating exactly: the LLM judge removes **self-authored** circularity (the method
can no longer exploit a rule its designers wrote), which is the specific trap the owner caught. It does
NOT remove **model-prior** dependence (the LLM is a learned proxy sharing text-sources with the world
the recommender models — Hole 1's shared-prior point). Honest claim boundary for every paper:
"results hold under two external answer models (an LLM judge and an a-priori structural rule); the LLM
judge is validated against human answers in §X [pending]." Never "realistic users," never "human-level
answerability." The human study remains the validator of the judge; the LLM study remains the load-bearer
of the recomputed results. That division of labor is defensible and, said plainly, sounds strong rather
than apologetic.

---

## Amended run order (v1, ML-25M)

1. **Freeze this doc + the design doc as the pre-registration** (cutoffs from Q-E filled in), commit.
2. **Mini-pilot on ML-25M** (~15 users, 1 day): port the pilot; fix the script nits; add the held-out
   half positive control + matched never-rated contrast. Sanity only.
3. **Gate run** (~300 users stratified, ≥20/taste-cell; question bank per Q-B; batched, cached, usage
   logged, model pinned): compute G1 + validity + G2. 2–3 days, measure cost from usage.
4. **Split-sensitivity check** (Hole 5): 50 users × 2 extra splits, report agreement.
5. **Decision:** G1+validity+G2 pass → main study (cached grid, masked-item validation pass, both
   structural artifacts, both value sources) and the adaptive agent. Any fail → static channel map,
   adaptive clause cut, and the gate result itself becomes a citable finding either way.
