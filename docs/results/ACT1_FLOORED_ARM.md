# Act 1, floored arm: the floor rescues the CONSUMER, not the selector

Run 2026-08-06/07, `experiments/baselines/interview/adaptive_table_floored_ep31.json`.
Identical to the NO-FLOOR table (`ACT1_ADAPTIVE_VS_STATIC.md`) in every respect except that the
Most-Popular floor is ON: a user who supplied no answer is scored on the popularity row instead of
on what the model actually predicts. Canonical Liang ML-25M, policy-B pool, 10,000 test users,
selectors entropy₀-static / GLM-UCB / Thompson / latent-only info-gain, consumers ours /
RecVAE-likes / TaNP / EDDI. MostPop prior .1626/.0262.

## Headline: the floor changes almost nothing on our tower and everything on the weak consumers

Δ = floored − no-floor, full NDCG@10.

| consumer | selector | k=1 Δ | k=16 Δ | users floored @k1 |
|---|---|---|---|---|
| **ours** | entropy₀ | **+.0023** | −.0003 | 5,633 |
| ours | GLM-UCB | +.0016 | +.0005 | 5,803 |
| ours | Thompson | +.0015 | −.0014 | 7,854 |
| ours | info-gain | +.0024 | +.0001 | 7,408 |
| **RecVAE-likes** | entropy₀ | **+.0676** | +.0024 | 5,633 |
| RecVAE-likes | Thompson | **+.1066** | +.0068 | 7,854 |
| RecVAE-likes | info-gain | **+.0969** | +.0059 | 7,408 |
| **TaNP** | entropy₀ | +.0314 | +.0015 | 5,633 |
| TaNP | Thompson | **+.0456** | +.0042 | 7,854 |
| EDDI | Thompson | +.0106 | +.0014 | 7,854 |

**On the certified tower the floor is worth ≤ .0024 anywhere, and is negative at k=16 for two
arms.** That is the whole point of the i26 exposure channel: a user who answered nothing has still
been *asked* something, those asked-but-unanswered items are folded as exposure tokens, and the
model's prediction from them is already better than the popularity row. There is nothing for the
floor to rescue.

**On the binary consumers the floor is worth up to +.1066** — an eleven-point NDCG swing at k=1 on
RecVAE-likes under Thompson. Those models have no unknown channel: no answer means no input, so
without a floor they score at or below the prior, and the floor replaces that with popularity.

## What this settles

1. **The NO-FLOOR contract is the right primary and this is now measured, not asserted.** On the
   system under test it barely moves anything (≤.0024), so nothing is being hidden by choosing it;
   on the other consumers it moves numbers by up to 0.1, so reporting *only* the floored arm would
   have manufactured a comparison in which those consumers look competitive at short budgets when in
   fact they received no usable input at all.
2. **The floor rescues the CONSUMER, never the SELECTOR.** The floored column reorders nothing among
   selectors: on our tower entropy₀ still leads at every budget (k=16: .2300 vs GLM-UCB .2166,
   Thompson .2121, info-gain .1918), so **Act 1's null holds identically under both contracts**.
   The adaptive-vs-static conclusion is contract-invariant, which is the strongest thing that can be
   said about it.
3. **The floor count is itself a diagnostic of selector answerability.** Users floored at k=1:
   entropy₀ 5,633, GLM-UCB 5,803, Thompson 7,854, info-gain 7,408. The two selectors that starve
   themselves are the two that get floored most — 78.5% of users could not answer Thompson's first
   question. That is the answer-rate story of `ACT1_ADAPTIVE_VS_STATIC.md` visible from a second,
   independent angle.

## Caveats

* Golbandi-leaf is absent by construction: it is in `CONSUMES_UNKNOWNS`, so it is never floored.
* No paired CIs on the floored/no-floor differences themselves; the ≤.0024 tower deltas are within
  the range where that would matter if any conclusion rested on them. None does.
* Both arms share every other contract element, so these Δ are attributable to the floor alone.
