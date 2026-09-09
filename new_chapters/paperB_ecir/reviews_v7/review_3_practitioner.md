# ECIR 2027 — Review 3 (industry / practitioner)

**Paper:** Objective Mismatch in Adaptive Preference Elicitation for Cold-Start Recommendation

## Summary of claims

Published cold-start selection rules optimise RMSE-era surrogates while the field now reports truncated top-k ranking; the paper argues this mismatch, not adaptivity itself, is why adaptive interviews underperform. (1) *An audit:* thirteen published methods plus classic heuristics, re-implemented under one contract — frozen certified two-tower recommender, ML-25M strong-generalisation split, answers read from each user's own held-out ratings, full catalogue with no sampling, unanswerable questions charged, 10,000 test users — all lose at every budget from 1 to 16 questions to a static order built by greedily maximising held-out NDCG@10; the authors state plainly that the static order gets a metric-driven search the ports do not, so they read this as diagnosis, not ranking. (2) *The mismatch is measured:* against exact NDCG@10 gains at 22 real interview states, three published criteria correlate at +0.010, −0.060 and −0.104 and recover only 11–34% of what a random pick leaves unrealised. (3) *The fix and its price:* measured NDCG@10 as Golbandi's split scalar gives +0.01035 full / +0.00705 tail at K=16 and is the only one of six criteria to flip the sign — but enlarging the *bank* with Tag-Genome concept questions, with no conditioning, buys more (+0.01430 / +0.02199), and choosing the answer *consumer* moves the number 2–4× more than choosing the selector.

## Strengths

- **The audit the area needed, under the right contract.** No sampled metrics, unanswerable questions charged, answers that cannot leak the target, one frozen recommender for every arm, native reproduction gates before porting, and gate failures reported rather than quietly scored (EDDI reduced and marked †, Greedy SLIM dropped).
- **The paper argues against its own headline where the data says so** — the bank beating conditioning, the null routing test, the 2.7× validation inflation, the six losing leaves.
- **Table 4 is the right experiment:** one scalar swapped, everything else fixed, plus two shuffle controls; the min-leaf-respecting shuffle grows deeper and more varied (27 leaves, 286 distinct questions against 15 and 159) and still loses by −0.04817, killing the "it just branched more" story.
- **The concept-channel decomposition is the most transferable paragraph here:** a film answer is worth 3.98× a concept answer while concepts collect 3.98× as many, and two film questions leave 5,779 of 10,000 users with nothing to say.

## Weaknesses, by severity

**1. The paper's largest number rests on an answer model no live onboarding flow can reproduce.** A concept "answer" is a signed implicit-watch statistic, "log2((nc + 1/2)/(ec + 1/2))", with answerability defined as "a user with ec < 1.5 has had no exposure and refuses rather than dislikes". A user therefore "answers" a tag question whenever they have watched enough films carrying it — nearly everyone with a profile, which is why the answer rate is 12.36 of 16. In production I ask "do you like dark humour?" and get a self-report, subject to acquiescence bias and to people being poor judges of their own attribute affinity. The paper concedes "a behavioural answer is one the user once gave, not one they would give in conversation, and that gap is not measured" — but that gap is load-bearing for the paper's strongest practical recommendation. **Fix:** a small human study comparing self-reported tag answers with the behavioural statistic, or a sensitivity analysis degrading concept answers with plausible noise and showing where +0.01430 goes to zero.

**2. The conditioning gain is fitted through, and looks specific to, the authors' own unpublished consumer.** The criterion is NDCG@10 obtained "by folding that child's answers through the frozen recommender and ranking the catalogue" — the same recommender that then scores it, described only in an anonymous technical report [3]. Table 5 reverses on the one classical consumer (Golbandi leaf, −0.01375⋆ at K=8), shrinks to +0.00161 on the accumulator, and RecVAE's +0.04122 is disclaimed by the authors as "measured against a static arm with almost no usable input". So of four consumers, one is the fitting target, one reverses, one gives a sixth of the headline, one is uninterpretable — and my ranker is not their two-tower set encoder. **Fix:** compute the criterion through one consumer and score on another, for at least one published recommender.

**3. At realistic budgets the effect is small and partly negative.** Nobody ships sixteen questions. At K=2 the tree buys +0.00528 full and **loses** on tail (−0.00262⋆); at K=4 it is +0.00891/+0.00482. Meanwhile "about 14% of users answer nothing in the first three questions", "six of the fifteen leaves lose on test, covering 39.3% of the cohort", the routing test is null ("the tree diverges without being shown to route"), and the build costs 37,191 seconds. **Fix:** make K ∈ {3,5} the primary presentation with a per-leaf win/loss distribution, and say whether you would ship the tree or the static combined order.

**4. Nothing addresses scale.** The bank is 500 items plus 500 concepts, each candidate scored by ranking the full catalogue at every node. My catalogue is not 18,359 items and my pool is not 1,000. "Serving is a lookup either way" is welcome, but I need build cost as a function of bank × cohort × catalogue and a re-fit cadence: the tree is valid only for the recommender it was grown against, and mine retrains weekly.

**5. The comparator asymmetry is bounded, but thinly.** "Giving each bandit the same freedom, one constant chosen on validation by the same metric, moves it by +0.0058 and +0.0018 against a deficit of 0.037–0.044." One scalar per bandit is not comparable freedom to a greedy search over 1,000 candidates on the reported metric. Section 3 handles this; the abstract's "every one loses, at every budget" does not.

**6. Offline ranking is the only outcome** — no engagement proxy, no abandonment measure, yet a twelve-answer and a three-answer interview are not the same product at equal NDCG@10.

## Questions for the authors

1. If a team can take one thing to production, is it the bank or the tree? Your numbers say the bank — will you say so in print?
2. How much of +0.01035 survives when the criterion is computed through one recommender and scored on another?
3. Any evidence on agreement between self-reported tag answers and the behavioural affinity statistic?
4. Why does the leaf-model consumer reverse at K=8, and can you predict which consumers reverse?
5. Would Table 1's ordering hold on a catalogue with weaker attribute metadata than the Tag Genome?

## Minor points and presentation

- Table 1 does too much (eighteen rows, two metrics per cell, five budgets, three markers); move K=1 and K=2 to an appendix. Fig. 1 is good.
- "Twelve of the twenty-three arms are value-free in that sense" is one of the paper's best observations and is buried mid-paragraph.
- The 37,191 s build cost belongs in the abstract; Section 3's "What the comparison is, and is not" should be signposted before Table 1.
- Citation [3] to an anonymous report is unavoidable under double-blind, but leaves the instrument's certification unverifiable at review time.

## Scores

- **Relevance: 5** — cold-start onboarding judged by a top-k metric is squarely ECIR, and the audit answers a question the area has been assuming past.
- **Originality: 4** — objective mismatch is not a new idea, but nobody has *measured* criterion regret against exact NDCG gains at the states a real interview visits.
- **Technical soundness: 4** — contract, native gates and the one-variable ablation are strong; undercut by the criterion running through the same recommender that scores it.
- **Quality of presentation: 3** — clear and unusually honest, but dense, and it foregrounds the least practically relevant budget.
- **Reproducibility: 4** — split, seed, answer rule, stopping rule, pool, tuning grids and gate failures all specified, with a complete artefact; the instrument is the one component described only by reference.
- **Potential impact: 3** — the bank finding will change what people build; the tree probably will not, per W2 and W3.

## Recommendation

**Weak accept**, confidence **4**.

## The one sentence I would send the authors

Your most useful result is the one you demoted — that enlarging the question bank beats conditioning because it beats the refusal rate — and the paper would be stronger if you led with it and treated the measured-metric tree as the mechanism study that explains why.

## Is the effect size large enough to justify building this?

**For the tree, no. For the bank, yes.** +0.01035 at sixteen questions is about 4% relative over an already-strong static order; it falls to +0.0053–0.0089 at the two-to-four questions I would actually ask, is negative on the tail at K=2, leaves 39.3% of the cohort in losing leaves, was overstated 2.7× on the fitting cohort, and costs ten hours of compute repaid at every retrain. That does not clear the bar for a new adaptive component with its own failure modes. The bank change is different: +0.01430 full and +0.02199 tail with no conditioning, driven by first-two-question refusals falling from 5,779 in 10,000 to 11 in 10,000 — large, cheap and mechanism-clear, subject to weakness 1.

## What I would do differently on Monday

1. **Instrument refusal first:** log per onboarding question whether the user could answer at all. If our opener leaves half of new users with nothing to say, that is a bigger lever than any policy change and needs no model work.
2. **Add tag questions to the bank and test on answer rate before ranking**, since the win comes from answers collected.
3. **Stop tuning the selector against a surrogate:** score candidate openers by measuring the reported quantity directly on a held-out cohort through the ranker that will serve, with the greedy prefix-optimal order as the internal bar — the paper's most portable idea, and it needs no tree.
4. **Re-certify the consumer** before any elicitation work lands, and **not adopt the measured-metric tree** unless the gain transfers across consumers.
