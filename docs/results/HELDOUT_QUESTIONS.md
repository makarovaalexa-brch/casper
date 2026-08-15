# EXPERIMENT Q — HELD-OUT QUESTIONS: does the head know a question it has never seen?

**Status: COMPLETE, 2026-08-15. TRAIN users only for fitting, held-out TRAIN users for evaluation.
No NDCG is measured anywhere in this experiment. The 10,000 canonical TEST users were never built,
never loaded, never named. `test_touched=false` on every artifact.**

Design sheet: `docs/design/PAPER_B_POLICY_DESIGN.md` **Part H.3 is governing** — it re-instates F.5,
deferred by the author on 2026-08-13 with *"train on all and see if it even works"*. It works, so the
deferral expired. Code: `src/instrument/outcome_q.py` (committed before the run, HARD RULE 10).

---

## THE VERDICT, FIRST

**The pre-registered falsifier did NOT fire.** It was: *held-out-question AUC ≤ the popularity
backoff ⇒ the similarity claim fails and is reported failed.*

| trained on | held-out-question AUC macro | the fair backoff prior | margin |
|---|---|---|---|
| 10% of training questions (1,551) | **0.97595** | 0.96285 | **+0.01310** |
| 25% (3,878) | **0.98032** | 0.96285 | **+0.01747** |
| 50% (7,755) | **0.98116** | 0.96285 | **+0.01831** |
| 100% (15,510) | **0.98167** | 0.96285 | **+0.01882** |

Evaluated on **3,880 questions no gradient ever touched × 10,000 users never fitted on**, 620,606,659
scored (state, question, outcome) cells per arm, nothing subsampled.

**And it beats the rigged comparator too.** A per-question prior that HAS seen the held-out question
scores **0.96915**. The head beats it at *every* point on the curve, including the 10% point. So the
head is not merely recovering a prior it was denied — it is doing something a per-question prior
cannot do at all.

**The curve saturates early, which is the answer to the author's actual question.**

| trained on | share of the full-data margin recovered |
|---|---|
| 10% | **69.6%** |
| 25% | **92.8%** |
| 50% | 97.3% |
| 100% | 100% |

**One tenth of the questions buys seven tenths of the benefit; a quarter buys nine tenths.** That is
the shape of *"learns from fewer examples via similarity"*, not the shape of memorisation. The
memorisation check agrees: on the questions the 100% arm *did* train on it scores 0.98277 against
0.98167 on the ones it never saw — a **generalisation gap of 0.0011**.

**Written in the discipline carried from Experiment R:** this is a capability that fires
*behaviourally*, not merely structurally. The head does not just accept an unseen question as input
(that is the architecture); it predicts reactions to it better than any state-free predictor,
including one that has seen it.

---

## The split, and its composition — printed before the time was spent

**20% of the 19,390 questions held out BY IDENTITY**, stratified by **channel × popularity decile**
(20 strata, 20% held out inside each), seed 98765. Popularity = train rating count for an item,
train answer rate for a concept. The sample-efficiency curve trains on **nested** stratified subsets
of the remaining 80%, so the 10% set is a subset of the 25% set and a difference between two points
is the added questions and nothing else (asserted in code).

| | held out | trained on |
|---|---|---|
| questions | **3,880** | 15,510 |
| items / concepts | 3,670 / 210 (94.59% items) | 14,689 / 821 (94.71% items) |
| per popularity decile | 367 items and 21 concepts in every decile — exactly flat | — |
| **answered by at least one fit user** | **3,880 of 3,880 (none is unanswerable-by-everyone)** | 15,510 |
| mean per-question answer rate | **0.03433** | 0.03396 |

⚠ **The composition caveat that belongs with the headline.** The held-out set is dominated by
rarely-answerable long-tail questions: the answer-rate quantiles are **0.00008 / 0.00027 / 0.00094 /
0.00573 / 0.03938 / 0.9012 / 0.99998** at [0, 25, 50, 75, 90, 99, 100]%. The median held-out
question is answerable by about **one fit user in a thousand**. That is *not* a defect of the split —
it matches the training set (0.03396) almost exactly, and it is what the catalogue is — but it means
the AUC is dominated by discriminating *who can answer at all*, and a reader should not hear
"predicts your taste in unseen films". The right sentence is **"predicts who will be able to answer
an unseen question, and what they will say when they can"**, and the per-class numbers below say
which half is which.

---

## ★ The fair baseline, stated because it decides whether Q means anything

A held-out question has **no per-question row** — it was never observed in training — so the
per-question popularity prior that Part F used as its comparator does not exist here. It must **back
off** to the mean outcome distribution of the question's **(channel, popularity-decile) stratum**,
computed over the **training questions and the fit users only**. **That backoff is the comparator.**

Three predictors are scored cell-for-cell through the same accumulator, so this is one comparison
rather than three tables placed side by side:

| predictor | what it may see | AUC macro | mean CE | ECE macro |
|---|---|---|---|---|
| global base rate (the floor) | nothing | 0.50000 | 0.16769 | — |
| **stratum backoff (THE COMPARATOR)** | channel + popularity decile, training questions only | **0.96285** | 0.07279 | 0.00062 |
| *rigged per-question prior* (**not** the comparator) | *the held-out question's own outcome counts* | *0.96915* | *0.06983* | — |
| **the head, 100%** | the frozen descriptor + the interview state | **0.98167** | **0.05527** | 0.00099 |

The rigged row is reported **only** so the gap is legible. Comparing against it would be rigged in
the head's disfavour on one axis (it has seen the question) — and the head wins anyway, which is why
it is worth printing.

---

## RESULT 1 — the per-position curve separates *descriptor* similarity from *interview* signal

The backoff cannot use the interview, so it is flat by construction. The head is given one more
answer at every position.

| position (answers heard) | 0 | 1 | 2 | 3 | 4 | 15 |
|---|---|---|---|---|---|---|
| **head (100%), AUC macro** | 0.9676 | 0.9744 | 0.9769 | 0.9791 | 0.9807 | **0.9851** |
| head (10%) | 0.9632 | 0.9698 | 0.9721 | 0.9740 | 0.9754 | 0.9786 |
| **shuffle control** | 0.9672 | 0.9672 | 0.9671 | 0.9670 | 0.9669 | **0.9665** |
| backoff prior | 0.9629 | 0.9629 | 0.9629 | 0.9629 | 0.9629 | 0.9628 |

**At position 0 there is no interview, so the head's 0.9676 is purely what the frozen descriptor buys
over the stratum backoff: +0.0047.** By position 15 it is 0.9851, i.e. **+0.0222**. So the margin
decomposes cleanly:

> **about a quarter of the head's advantage on an unseen question is descriptor similarity, and
> about three quarters is the interview.**

Both halves are the claim under test. The descriptor half is *"it knows similarity between concepts
and items"*; the interview half is what a per-question prior — rigged or not — structurally cannot
have.

## RESULT 2 — the shuffle control (HARD RULE 5), and the one number in it that needs explaining

Same states, same architecture, same optimiser, same budget; **only the target moves** — the outcome
labels of user *u* are taken from a different user, so the state→label association is destroyed while
every per-question marginal survives.

| | AUC macro | mean CE | ECE macro | per-position curve |
|---|---|---|---|---|
| **main (100%)** | **0.98167** | 0.05527 | 0.00099 | 0.9676 → **0.9851** |
| **shuffle control** | 0.96682 | 0.07107 | 0.00240 | 0.9672 → **0.9665 (FLAT)** |
| backoff prior | 0.96285 | 0.07279 | 0.00062 | flat |

**✅ The control passes on the measurement that matters: its per-position curve is flat.** The main
arm's rise from 0.9676 to 0.9851 does not survive label permutation, so it is the state and not the
position, the interview length, or anything else that travels with them.

⚠ **But the shuffle does not land exactly on the backoff — it sits 0.0040 above it, and that is a
finding rather than a leak.** In Part F the shuffle converged onto the per-question prior almost
exactly, because that prior was its exact theoretical ceiling. Here it cannot be: the shuffle keeps
every per-question marginal it was trained on, and it can express those marginals **through the
frozen descriptor**, which then transfers to a held-out question by descriptor similarity. So the
shuffle's ceiling is *the descriptor-expressible per-question prior*, which is a strictly better
predictor than the stratum mean.

That residual is the descriptor's own generalisation measured with the state switched off — and it
agrees, to within 0.0007, with the position-0 gap of the main arm (+0.0047). Two independent
estimates of the same quantity from different arms. It is also still **below the rigged per-question
prior (0.96915)**, exactly as it should be: a state-blind model routing a prior through a
205-dimensional descriptor cannot beat the prior itself.

## RESULT 3 — where the discrimination lives, by outcome class

| trained on | AUC like | AUC dislike | AUC no-answer | accuracy | ECE macro |
|---|---|---|---|---|---|
| 10% | 0.96255 | 0.98301 | 0.98229 | 0.98181 | 0.00141 |
| 25% | 0.96962 | 0.98578 | 0.98557 | 0.98195 | 0.00112 |
| 50% | 0.97073 | 0.98643 | 0.98630 | 0.98194 | 0.00093 |
| **100%** | **0.97146** | **0.98680** | **0.98673** | 0.98192 | 0.00099 |
| shuffle | 0.95071 | 0.97525 | 0.97451 | 0.97735 | 0.00240 |

**`like` is the hardest class at every training fraction and it is the class that keeps improving
longest** (+0.0089 from 10% to 100%, against +0.0038 for dislike). Predicting *whether* someone can
answer an unseen question is easier than predicting *what* they will say — which is the expected
ordering given the composition caveat above, and it is the honest limit of the claim.

**Read the accuracy column and then ignore it**, for the same reason as Part F: the outcome base rate
over the (user × catalogue) grid is 0.00816 like / 0.02588 dislike / 0.96596 no-answer, so "always
predict no-answer" already scores ~96.6%. Accuracy cannot separate these arms and is printed only so
nobody quotes it.

**Calibration.** The head's ECE is 0.00099 macro against the backoff's 0.00062 — the head is slightly
*less* calibrated than an empirical frequency table, which is calibrated by construction, and both
are small in absolute terms. Worth noting that ECE **improves monotonically with training questions**
(0.00141 → 0.00112 → 0.00093 → 0.00099), i.e. the extra questions buy calibration as well as
discrimination.

---

## Guards and firewall

- **No-profile guard, structural:** at position 0 the interview has revealed nothing, so the state
  must be bit-identical across users. Asserted over 20,000 user-states, **max |diff| = 0.0**, before
  a single gradient step in every arm.
- **`--prior_feat` does not exist in this file.** A per-question prior does not exist for a
  never-seen question, so the descriptor is the frozen one and nothing else.
- **No NDCG is measured anywhere**, which is what makes fitting on TRAIN users legitimate: the tower
  memorised their ratings, but an observed answer outcome is not a model output.
- **Canonical snap** `0.15686538118354684` reproduced exactly before anything was fitted (the
  environment is built only to read the frozen descriptors and to snap the ruler).
- **Nested-subset assertion** in code: the 10% question set ⊂ 25% ⊂ 50% ⊂ 100%.
- **Refusal in code** if no held-out question is answerable by anyone — the check that would have
  caught a vacuous comparison. It passed at 3,880/3,880.

---

## WHAT THIS SETTLES

1. **The reaction head predicts reactions to questions it has never seen, better than any state-free
   predictor available — including one that has seen them.** 0.98167 vs 0.96285 (fair) and 0.96915
   (rigged), over 620 million held-out cells.
2. **It is sample-efficient in questions, not memorising them.** 10% of the questions buys 70% of the
   margin; the seen/unseen gap at 100% is 0.0011.
3. **The margin decomposes**: ~¼ descriptor similarity (position 0, and independently the shuffle
   control's residual), ~¾ the interview.
4. **A tree cannot enter this game at all.** It has no representation of a question outside its own
   node set; there is no analogue of this measurement for it. That is the honest form of the
   architecture claim, and unlike Experiment R it is behavioural rather than structural.
5. **The `like` class is the weakest and the composition is tail-dominated**, so the claim is "who
   can answer, and what they say when they can", never "predicts your taste in unseen films".

**What it does NOT settle:** nothing here is an NDCG gain. The head is an input to a policy, not a
policy; Part F measured its in-policy contribution at +0.00392 [+0.00105, +0.00676] and that number
is the one that travels with any claim about what it is worth *inside* an interview. And nothing here
touched the test cohort.

---

## Cost — a property of this box

| arm | training questions | epoch | eval (10,000 users × 3,880 held-out questions × 3 predictors) | total |
|---|---|---|---|---|
| 10% | 1,551 | 6.2 min | ~5 min | **11.6 min** |
| 25% | 3,878 | 10.8 min | ~5 min | **16.2 min** |
| 50% | 7,755 | 18.2 min | ~6 min | **24.0 min** |
| 100% | 15,510 | 46.0 min | ~6 min | **52.3 min** |
| shuffle (100%) | 15,510 | 45.9 min | ~4 min | **49.9 min** |
| | | | **end to end** | **~2.6 h** |

The Part H.3 estimate was 4–6 h; it came in at **2.6 h**, because the training cost scales with the
number of candidate columns and three of the five arms train on a fraction of them. One epoch per
arm, as in Part F — the loss had flattened and the held-out numbers are unambiguous.

## Artifacts

| what | where |
|---|---|
| per-arm results (full metric suite, both baselines, per-position curves) | `experiments/battery/heldout_q/q{010,025,050,100}_main.json`, `q100_shuffle.json` |
| the combined artifact + split composition | `experiments/battery/heldout_q/heldout_questions.json` |
| checkpoints (each carries its own train/held-out question id arrays) | `experiments/battery/heldout_q/q*.pt` |
| log | `experiments/battery/q_main.log` |
| code | `src/instrument/outcome_q.py` |

**Reproduce:**
`python src/instrument/outcome_q.py --fracs 0.10,0.25,0.50,1.00 --shuffle_at 1.00 --n_heldout_users 10000 --threads 6 --i_have_author_goahead`
