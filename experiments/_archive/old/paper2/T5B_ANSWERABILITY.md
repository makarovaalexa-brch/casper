# T5b — Item-vs-concept elicitation under a stricter, popularity-conditioned answerability model

**Review response (Paper B).** A reviewer objected that the answerability model behind the headline
"answerable concepts beat item-asking" result is *too generous*: concepts are counted answerable
(≥2 rated member items) almost always (8/8 asked), while items are answered only when the user
actually rated them (~1.9/8). This experiment re-runs the exact item-vs-concept comparison under a
**popularity-conditioned seen-probability** answerability model and reports both regimes side by side.

- Script: `scripts/paper2/t5b_answerability_popcond.py` (additive; canonical setup untouched)
- Numbers: `data/movielens/.cache/paper2/t5b_answerability.json` (ALPHA=1.0), `..._alpha2.json` (ALPHA=2.0)
- Setup (identical to the headline `answerability_concept.py`): ML-1M, learned concept channel
  (`enc_concept`/`Ec`/`Ql_concept`), **geometric** answer, locked 150-user split (seed 123),
  `popb+Ql` ruler, HEAD = top-33% by like-count, realistic cold-start (empty start, T=8 budget over
  the full catalogue), q∈{0,2,4,8}, FULL + TAIL NDCG@10. Bootstrap 95% CIs over the 150 users (2000 resamples).

## The two answerability regimes

**`orig`** (the existing model): item answerable iff the user actually rated it (P=1 if seen);
concept answerable iff ≥2 rated member items.

**`popcond`** (NEW): a user can only *answer* about what they can *recall*, and recall is popularity-gated:

> P(recall item *j*) = σ( α · ( log(cnt_j+1) − log(cnt_med+1) ) )

a logistic in log-popularity anchored at the **median** popularity of rated items (P=0.5 there;
→1 for head items, →0 for obscure tail items). We draw a recallable subset of each user's known-half
profile once (fixed per-user seed). Then **item answerable iff recalled**, and **concept answerable iff
the user can recall ≥2 of its member items** — items and concepts treated consistently, so a concept
built from obscure items no longer gets "free" answerability. The true taste target u* is unchanged
(always the full known-half fold); only *which questions are answerable* changes.
At α=1.0: p_recall ≈ 0.98 (head) / 0.50 (median) / 0.02 (deep tail), median rated-item count = 41.

## Results — NDCG@10 at q=8 (mean [95% CI]); "ans" = answers obtained / 8

| regime | metric | pop_item (q8) | conc_pop (q8) | Δ conc_pop − pop_item @q8 |
|---|---|---|---|---|
| orig    | FULL | 0.307 [0.263, 0.354] | 0.315 [0.272, 0.360] | **+0.008 [−0.016, +0.030]  (n.s.)** |
| popcond (α=1) | FULL | 0.308 [0.265, 0.354] | 0.315 [0.272, 0.360] | **+0.007 [−0.015, +0.030]  (n.s.)** |
| orig    | TAIL | 0.085 [0.066, 0.107] | 0.116 [0.089, 0.147] | **+0.031 [+0.004, +0.060]  (sig)** |
| popcond (α=1) | TAIL | 0.086 [0.066, 0.107] | 0.116 [0.089, 0.147] | **+0.031 [+0.004, +0.060]  (sig)** |

Answers obtained (per 8 asks): pop_item **1.9** (both regimes), conc_pop **8.0** (both regimes),
conc_eig 7.9→7.8. conc_eig (myopic info-gain) stays below conc_pop as in the original result.

Full q-curve (conc_pop) — FULL: 0.293 → 0.306 → 0.312 → 0.315; TAIL: 0.088 → 0.105 → 0.109 → 0.116.
Identical under both regimes.

### Sensitivity — steeper gate (α=2.0)
Doubling the logistic slope (deep-tail recall crushed to ~0.001; median still 0.50) leaves the verdict
byte-identical: conc_pop still 8.0 answers/8, TAIL conc_pop 0.116 [0.089, 0.146] vs pop_item 0.085
→ **Δ +0.031 [+0.004, +0.060] (sig)**; FULL Δ **+0.008 [−0.016, +0.030] (n.s.)**. The broad winning
concepts remain fully answerable no matter how hard the tail is penalized.

## Verdict

**The concept-channel advantage SURVIVES the stricter, popularity-conditioned answerability model,
essentially unchanged.** Tail: conc_pop beats pop_item by **+0.031 (significant, CI excludes 0) in both
regimes**; full: +0.008/+0.007 (n.s.) in both. The numbers barely move.

**Why it survives (and why this answers the reviewer):** the winning concept policy asks *broad, frequent*
concepts. Even after popularity-gating recall, such a concept still has ≥2 recallable members in a user's
profile (its membership includes head items the user reliably recalls), so it stays answerable — ans/8 is
unchanged at 8.0. Meanwhile item-asking already relied on popular items (high recall), so it too is nearly
unaffected. In other words, the concept advantage was **never driven by generous answerability of obscure
material**; it comes from concepts being coarse aggregates that latch onto recallable popular items while
still carrying tail-relevant signal. Making answerability popularity-realistic does not erode the result.

**Honest caveats:** (1) This gate bites hardest on *rare* concepts / obscure profile items; the winning
`conc_pop` selects broad concepts by construction, so it is structurally robust to this specific stress —
that is the finding, not an evasion of it. (2) Same idealized honest-preference user simulator and single
dense dataset as the original result. (3) conc_pop ≥ conc_eig unchanged (myopic EIG still loses).
