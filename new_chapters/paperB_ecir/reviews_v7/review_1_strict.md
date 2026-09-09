# ECIR 2027 — Review 1

**Paper:** Objective Mismatch in Adaptive Preference Elicitation for Cold-Start Recommendation

## Summary of claims

The paper argues that cold-start elicitation methods select questions by criteria built for an RMSE-era objective while the field now reports truncated, position-discounted top-k ranking, and that this mismatch — not adaptivity as such — explains their weak performance. (RQ1) Under one contract on one frozen recommender — behavioural answers from users' own held-out ratings, full unsampled catalogue, unanswerable questions charged, 10,000 test users — thirteen re-implemented methods and the classic heuristics all lose, at every budget from 1 to 16 questions, to a fixed order built by greedily maximising held-out NDCG@10. (RQ2) Against exact measured NDCG@10 gains at the 22 nodes of a grown tree, three published split criteria correlate at +0.010, −0.060 and −0.104 and recover only 11–34% of what a random pick forgoes. (RQ3) Putting the measured metric into Golbandi's ternary tree — CASPER — is the only one of six criteria that lifts the interview above the metric-optimised static order (+0.01035 full / +0.00705 tail at K=16), while enlarging the bank with Tag Genome concept questions, with no conditioning at all, buys more (+0.01430 / +0.02199), because sixteen concept questions collect 12.36 answers where sixteen film questions collect 3.11. The framing is a diagnosis, not a verdict on adaptivity, and the paper says so.

## Strengths

1. **The measurement contract is excellent and precisely specified**: strong-generalisation split, named seed, no negative sampling with the Krichene–Rendle justification, answered items removed from the pool, a run-time assertion that the answer lookup contains no target, learned decoder bias in scoring, and — unusually — unanswerable questions charged rather than silently skipped. Most papers here get two of these wrong.
2. **RQ2 is genuinely novel.** Persisting exact measured gains for ~1,000 candidates at 22 real interview states and scoring the surrogates against them turns a rhetorical claim into a measurement, and the root-node detail (the logistic argmax *is* the metric's argmax; variance ranks that question 250th of 1,000) is exactly the evidence the literature lacks.
3. **The paper bounds its weakest point rather than hiding it**, stating the fitting asymmetry plainly in Section 3 and quantifying it by giving the bandits the same validation freedom (+0.0058, +0.0018 against a 0.037–0.044 deficit). Other awkward results are likewise reported, not buried: six of fifteen leaves lose on test, the routing test is a pre-registered null and no routing claim is made, validation overstates at +0.02794, and the leaf-model consumer reverses the sign.
4. **The channel decomposition (p.11) is careful**: the K=16 tie is two cancelling effects — a film answer worth 3.98× a concept answer while concepts collect 3.98× as many — and at matched answers the film channel *wins*.

## Weaknesses, by severity

**W1. The criterion is optimised through the same frozen model that grades it, and that model is documented only in an unavailable anonymous report.** Each candidate "is scored by the node-size-weighted mean of its three children's cohort-mean measured held-out full NDCG@10, obtained by folding that child's answers through the frozen recommender." Held-out *users* control for user-sample overfitting, not for fitting to the instrument's idiosyncrasies. The transfer evidence is thin: RecVAE "counts for direction only", the accumulator gives +0.00161 (a sixth of the headline), and the leaf model *reverses* at K=8 (−0.01375*). One clean replication of three does not comfortably support the abstract's "holding on two of three further consumers", and reading the reversal as "an argument for certifying the consumer" converts a failed generalisation test into a recommendation. *Fix:* re-grow the tree with the criterion computed through a different consumer and score on the certified tower; failing that, scope the claim to the certified tower and give the instrument — described only via [3], "Technical report, under review" — a self-contained half-page.

**W2. Table 4's "one variable changed" is asymmetric, and at K=16 tree and comparator are not capacity-matched.** (a) Every tree in Table 4 is compared against *the same* static order, itself built by maximising the measured metric — CASPER's own criterion — so the table cannot separate "criterion X is a poor splitter" from "criterion X was measured against a rival objective's bar". (b) More seriously: "Beyond K=5 the tree's depth is exhausted and the remaining questions are chosen inside the reached leaf by the same measured-NDCG search." The K=16 arm is therefore five conditioned questions plus **fifteen leaf-specific validation-fitted tails**, against a *single* global order — and +0.01035 is quoted at exactly that budget. Six leaves losing on test and a 2.7× validation overstatement are the signature of that extra capacity. *Fix:* report the gain against a comparator allowed the same number of fitted sequences, or headline K=8 (+0.01062).

**W3. The +0.01430 bank result rests on channel-incommensurable answerability and answer granularity.** A film is answerable iff rated; a concept iff e_c ≥ 1.5, and its answer is a *continuous signed statistic*, log2((n_c+½)/(e_c+½)), over the fold-in profile. The concept channel thus gets both a much laxer answerability test and a real-valued sufficient statistic of the user's history, where the film channel gets one half-star rating — so the 12.36-versus-3.11 answer rate identified as the whole mechanism is partly definitional. Section 7 concedes answerability "is a modelling assumption ... load-bearing", but never states the asymmetry *between channels*, and it is that asymmetry which carries the claim. *Fix:* re-run concepts with the answer quantised to like/dislike/refuse and a stricter exposure threshold.

**W4. Uncertainty is quantified over users only, never over fits.** Every interval is a "paired percentile bootstrap over the per-user difference ... clustered over users", but tree, static orders and bandit constants are all fitted on one validation cohort, one seed, one split. For +0.01035 on a 0.259 base the dominant variance is plausibly the fit: rebuilding the comparator moved the same quantity from +0.01371 to +0.01035, larger than half the effect and outside the quoted CI. *Fix:* refit on three to five validation resamples and report the across-fit spread.

**W5. RQ2 judges the surrogates only at states CASPER's own tree visits** — "22 nodes ... over that node's own users". Each surrogate is scored at nodes it would never reach, where along its own trajectory its regret could be materially smaller, so the conclusion holds for *this* interview only. *Fix:* repeat Table 2 at each criterion's own nodes; the Table 4 trees presumably already exist.

**W6. The audit is not quite as complete as "every one loses".** Greedy SLIM — the authors' own "closest protocol sibling, on ML-25M with NDCG@10 and the same tail definition" — "passed only in part and is not run"; EDDI appears only in reduced form after failing its gate; Golbandi's tree, the method CASPER modifies, has no K=16 row, the budget every headline uses. Ports sitting near the answer-free floor (A-optimal .1740 against a .1626 reference; ConUCB ans@16 = 1.80) are as consistent with a port defect as with a property of the method. The abstract should say "every one we could run".

## Questions for the authors

1. If the criterion is computed through RecVAE instead of the certified tower, does CASPER still beat the static order when scored on the certified tower?
2. What is the K=16 gain against a comparator given the same number of validation-fitted sequences as the tree has leaves?
3. How much of +0.01430 survives quantised concept answers and a stricter e_c threshold?
4. Table 2 reports a mean ρ over 22 nodes — what is the spread? "Two of them point the wrong way" depends on it.
5. Table 5: RecVAE's static arm reads .0000/.0000 at K=2. Genuine empty-input score, or a degenerate port?
6. Was the reported tree the first grown, and how many depth / min-leaf / bank-size configurations were seen on validation first?

## Minor points and presentation

"Greedy prefix-optimal" is an overclaim: greedy is prefix-optimal only under submodularity, neither established nor plausible when answers interact; "greedy metric-maximising order" claims what is true. The prose is dense but disciplined, though hedged thickly enough that the claim can be hard to extract. Table 1's ans@16 reads "1.5i/9.2c" for the combined order and a scalar elsewhere; unify. Fig. 1's nine series overlap in the .21–.23 band. "twice that selector's deficit against the heuristic" (p.6) needs both numbers spelled out. Section 3 packs protocol, instrument, answers, ports and comparator under one heading. Per the chairs' instruction I have assumed a complete artefact exists and scored reproducibility on the paper's description alone.

## Scores

- **Relevance: 5** — elicitation, evaluation contracts and reproducibility are ECIR core.
- **Originality: 4** — the criterion-regret measurement at real interview states is new; static-beats-adaptive is a replication the paper properly credits.
- **Technical soundness: 3** — excellent contract, but the criterion is optimised through its own scorer (W1), the flagship ablation is asymmetric and capacity-mismatched at the headline budget (W2), and uncertainty ignores fitting variance (W4).
- **Quality of presentation: 4** — precise, quantitative, unusually self-critical; over-hedged in places.
- **Reproducibility: 4** — split, seed, cohorts, bank, grids, gate outcomes and bootstrap suffice to rebuild; one point off because the instrument lives only in an unavailable technical report.
- **Potential impact: 3** — contract and diagnosis are reusable; the positive result is +0.01035 on a 0.259 base from one seed, one split, one instrument.

## Recommendation

**Borderline**, confidence **4**. I would move to weak accept on convincing answers to Q1 and Q2.

## The one sentence I would send the authors

Your audit and your criterion-regret measurement could carry this paper on their own, but the +0.01035 headline is a tree fitted through the scorer that grades it, quoted at the one budget where it has fifteen leaf-specific fitted orders against the comparator's one — fix the capacity match, show the gain survives a change of consumer, and this becomes a paper I would argue for.
