# Act 1: adaptive question selection vs a static order, on the certified stack

Run 2026-08-04 15:17 → 20:22 (CPU, detached PID 21360).
`src/baselines/adaptive_harness.py --selectors entropy0-static,glm_ucb,thompson,eddi_ig
--budgets 1,2,4,8,16 --only ours,golbandi_leaf,belief_mf,recvae_likes,tanp,eddi --arch i26
--snapshot .cache/instrument/t2i26c_ep31.pt --no_floor --label i26c_ep31`.
Output `experiments/baselines/interview/adaptive_table_i26c_ep31.json`.
**10,000 test users, no sampling anywhere.** MostPop prior: 0.1626 / 0.0262.

## What was new here

Until 2026-08-04 the interview harness had **zero adaptive question selectors** — all six arms in
`interview_strategies.py` are static global orders and `golbandi_leaf` is a *consumer* of answers,
not a selector. `adaptive_harness.py` adds the sequential loop (the selector sees the answers so
far and picks the next question per user), and re-uses `run_interview_table`'s scoring, floor,
candidate-pool and contract code verbatim rather than copying it.

**Regression gate (passed before any adaptive number was produced):** all six static arms driven
through the sequential path reproduce `interview_strategies.ask()` **exactly** — Python list
equality of both `asked` and `answered`, i.e. ask order, ask identity, answerability and the
half-star level attached to every answer — for 30 (arm, budget) cells on the full 10k cohort.

**Selectors.** `glm_ucb` and `thompson` ride the *same* `belief_mf` Gaussian posterior the
belief-MF consumer scores with (frozen iALS basis, f=200; one implementation of the conjugate
update, three read-outs) — the ICF/ConTS bandit family, NICF's own baselines. `eddi_ig` is EDDI's
single-order info-gain (Ma et al. ICML'19) over our trained Partial-VAE; the Eq.-4 target form is
inapplicable here and is documented as such (it would leak evaluation targets into selection).
Candidate pool = the full 18,359-item vocabulary minus already-asked, i.e. the identical universe
the static arms walk. `alpha=1.0`, untuned.

## Headline: on our tower, every adaptive selector LOSES to the static order

Consumer `ours` (i26c ep31), full/tail NDCG@10:

| selector | k=1 | k=2 | k=4 | k=8 | k=16 | answered@16 |
|---|---|---|---|---|---|---|
| **entropy0-static** | **.1635/.0433** | **.1706/.0528** | **.1831/.0636** | **.2053/.0792** | **.2303/.0924** | 5.52 |
| glm_ucb | .1606/.0400 | .1635/.0452 | .1756/.0578 | .1902/.0673 | .2161/.0833 | 5.24 |
| thompson | .1627/.0376 | .1676/.0437 | .1748/.0551 | .1908/.0677 | .2135/.0878 | 3.39 |
| eddi_ig | .1537/.0336 | .1539/.0373 | .1618/.0432 | .1747/.0611 | .1917/.0749 | 3.84 |

Static wins at **every budget on both axes**. The ordering repeats on RecVAE-likes, TaNP and
EDDI-as-consumer. EDDI's own acquisition is worst everywhere — including when EDDI itself is the
consumer (.1493/.0488 vs .1769/.0480 static at k16), so this is not a mismatch artefact.

## The finding that complicates it: the winner FLIPS on Golbandi's consumer

| consumer | selector | k=1 | k=2 | k=4 | k=8 |
|---|---|---|---|---|---|
| golbandi_leaf | entropy0-static | .1579/.0347 | .1563/.0414 | .1680/.0670 | .1686/.0677 |
| golbandi_leaf | **thompson** | **.1682/.0399** | **.1743/.0478** | **.1844/.0616** | **.1974/.0750** |
| golbandi_leaf | eddi_ig | .1676/.0382 | .1693/.0413 | .1710/.0495 | .1817/.0643 |

On the leaf consumer (k≤8, 3^k pattern matching) Thompson sampling beats the static order at every
budget — +0.0288 full at k=8 — and even EDDI's acquisition beats it. **Which selector wins depends
on which consumer you assume**, which is Paper A's instrument-dependence spine appearing on the
selection side. Mechanism, not mystery: the leaf consumer matches a user to train users with the
*same answer pattern*, so it rewards questions that spread users across distinct patterns; a
per-user randomised selector does that by construction, while one fixed global order puts most
users in the same few cells. Reported as a finding, never buried.

## Dispersion diagnostic (collected free by the harness)

Distinct ask-prefixes over 10,000 users:

| selector | k=1 | k=2 | k=4 | k=8 | k=16 |
|---|---|---|---|---|---|
| entropy0-static | 1 | 1 | 1 | 1 | 1 |
| glm_ucb | 1 | 3 | 23 | 335 | 5,739 |
| thompson | 759 | 8,083 | 10,000 | 10,000 | 10,000 |
| eddi_ig | 1 | 2 | 7 | 99 | 1,993 |

**This is divergence without exploitability, measured directly.** The adaptive policies genuinely
branch — Thompson gives all 10,000 users a distinct interview by k=4, GLM-UCB reaches 5,739
distinct interviews by k=16 — and still lose to a single order asked of everybody. Leg-1 said the
standard banks leave nothing to adapt *to*; this says that even when a policy manufactures
per-user variety, on this consumer that variety does not pay. The two together are the honest
form of Act 2's claim.

## The mechanism the numbers point at: answerability, again

At k=16 the static order collects **5.52** answers; Thompson collects 3.39 and EDDI 3.84. The
adaptive criteria chase posterior movement or score uncertainty, both of which are largest for
items few users have seen — so they spend budget on questions users cannot answer. This is the
same failure mode Paper A measured on the item channel (2.18/8 items vs 6.02/8 concepts answered)
now appearing as a *policy* failure. Note it also explains why the flip happens on
`golbandi_leaf`: that consumer is the one that is paid in *unknowns* (it consumes the unknown
branch), so unanswerable questions still carry information for it.

## Status of these numbers

* **Point estimates only — no paired CIs yet.** The required next step before any of this is a
  claim: paired bootstrap on the same 10k users (never the `1.96*sqrt(2)*SE` unpaired form).
* `belief_mf` as a consumer is nearly flat across everything (.161–.167 full at every k and
  selector), consistent with its recorded inertness; it is not evidence about selection.
* `golbandi_leaf` is capped at k≤8 (`MAX_DEPTH`, 3^k leaf fragmentation).
* NO-FLOOR contract; the floored arm has not been produced. TaNP sits below the MostPop prior on
  zero-answer arms, which the floor exists to hide — a reason to report both arms.
* The comparator here is the **entropy0** static order, not the certified prefix-optimal greedy
  (`GREEDY_BEST_STATIC_I26C.md`, finished later the same night). Re-running these selectors
  against the greedy sequence is open work; entropy0 is the strongest of the six lit arms but the
  greedy combined arm is stronger still (.2115/.0880 at q16 with a *different* channel mix), so
  the null can only harden.
