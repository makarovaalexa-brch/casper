# P1b — DOES THE **UNDAMAGED** POLICY RANK A QUESTION IT WAS NEVER TAUGHT?

**Status: COMPLETE, 2026-08-15. Validation only — the 10,000 canonical TEST users were never built,
never loaded, never named. `test_touched=false` on every artifact.**

Design sheet: `docs/design/PAPER_B_POLICY_DESIGN.md` Part I.1, **amended 2026-08-15** after P1
measured that removing 36 of the 172 label questions damages the policy (0.28355 → 0.23185). Code:
`src/instrument/policy_p1b.py`, committed before the run (HARD RULE 10). **No training happened here
— this scores frozen checkpoints.**

> **AUTHOR SAID (2026-08-15):** *"ok, do it."* — on being shown P1's cost and asking for a version
> that does not have to break the policy first.

---

## ★★ READ THIS FIRST — THE ON-MANIFOLD CONTROL, AND WHAT THE TABLE DOES *NOT* SAY

**Every state in groups A and B is one E1 was never trained on.** So a taught/never-taught contrast
across them confounds two different novelties — an unseen **question** and an unseen **state** — and
**P1b cannot separate them.** Without a cell where the policy is expected to succeed, the tables below
read as "it learned nothing at all", and that reading is wrong.

**Group C is that cell**: the 7 internal nodes of the frozen tree, where E1 reproduces the tree
exactly and the measured argmax *is* the tree's own splitter. Each carries the exact gain of all
1,000 candidates. E1 should rank it **1st**.

| group C — on-manifold, 7 states, all 7 measured-best questions are taught | median rank | top-1 | top-5 | mean ρ |
|---|---|---|---|---|
| **E1** | **0 (i.e. 1st)** | **1.0000 (7/7)** | **1.0000** | +0.2278 |
| shuffle (floor) | 1 | 0.2857 | 1.0000 | +0.1925 |
| popularity (dumb) | 418 | 0.0000 | 0.0000 | +0.2542 |

> **The four-line summary this licenses: E1 ranks the measured-best question FIRST at every state it
> was trained on, and 208th (full-pool) to 374th (sampled) of 1,000 at states it was not.** That is
> R2's off-manifold finding measured a second way. It is *not* "the policy learned nothing".

**And it is the instrument check on this whole file.** The pre-registered stop condition was: if E1
does not rank 1st on-manifold, the ranking harness disagrees with the policy's own argmax and every
number here is unverified. It ranks 1st at 7 of 7. The harness is verified.

**What P1b therefore can and cannot claim.** It can say the policy's ranking is excellent where it
was trained and no better than popularity where the question is one it never saw. It **cannot**
attribute that gap to the unseen vocabulary alone, because the state is unfamiliar in the same cells.
Separating the two would need measured gains at *on-manifold* states whose argmax is never-taught —
and group C has none (all 7 argmaxes are the tree's own splitters, which is what the tree being
measured-optimal means). That separation is not available from any banked artifact.

---

## ★ THE VERDICT, FIRST — THE FALSIFIER FIRED

The falsifier was: *never-taught ranking ≤ the popularity comparator ⇒ there is no similarity-based
generalisation in the ACTION policy, and it is reported failed.*

**It fired.** On the states where the measured-best question is one E1 was never taught, E1 does not
beat a fixed popularity ranking anywhere, and on the larger group it is **significantly worse**:

| never-taught states | E1 − popularity, rank of the best question | E1 − popularity, ρ over never-taught candidates |
|---|---|---|
| group A, full-pool sweeps (n=17) | −78 [−255, +103] — **null** | −0.032 [−0.089, +0.029] — **null** |
| group B, sampled sweeps (n=94) | **+68 [+9, +126] — E1 ranks it WORSE** | **−0.061 [−0.109, −0.014] — popularity correlates BETTER** |

E1 is also statistically indistinguishable from **its own label-shuffled control** on these states
(rank difference −51 [−164, +86] in A, +11 [−56, +78] in B; ρ difference +0.034 [−0.038, +0.103] and
−0.025 [−0.082, +0.033]).

**And the instrument is not blind — it sees skill where skill exists.** On the *same* states, scored
by the *same* code, when the measured-best question is one E1 *was* taught:

| taught states | E1 top-1 | E1 top-5 | E1 median rank | popularity median rank | E1 − popularity rank |
|---|---|---|---|---|---|
| group A (n=25) | **0.2400** | **0.4000** | **71** | 444 | **−253 [−376, −130]** |
| group B (n=44) | 0.0000 | 0.0227 | 192 | 503 | **−119 [−205, −31]** |

> **The sentence this licenses: the policy ranks the right question well inside the vocabulary it was
> supervised on, and no better than popularity outside it. The dot-product head is
> vocabulary-agnostic MECHANICALLY — it scores all 1,000 (and all 19,390) candidates in one pass —
> but the VALUE it assigns to a never-taught candidate carries no measured signal beyond what a
> popularity prior already carries.** That is the same conclusion E1's open-vocabulary demonstration
> reached from the other direction (`E1_POLICY_IMITATION.md` §4.2: out-of-pool picks answered 9.1% of
> the time, −0.02034 NDCG), now measured against ground truth rather than inferred from a score drop.

**One honesty note that belongs with the headline:** *nothing here ranks well in absolute terms.* The
best mean rank correlation any arm achieves against the measured gains, on any group, is **ρ ≈ +0.21**
— and that is the popularity comparator. E1's own ρ on never-taught candidates is +0.104 [−0.002,
+0.205] in group A (does not exclude zero) and +0.149 [+0.090, +0.210] in group B. The action-value
surface is weakly predicted by everything available, which is the same roughness the value line
measured directly (inter-state rank correlation ~+0.29). This is not a case of a good ranker failing
on unseen items; it is a case of a weak ranker that is weakest where it was never taught.

---

## ★ The held-out set already existed — nobody had to make it

The frozen tree uses **172 of the 1,000 pool questions** as labels. **828 pool questions were never a
training target for E1** — no gradient ever pushed the policy toward emitting one. The measured-value
sweeps banked NDCG gains at states off the tree's own trajectory. Wherever the measured best question
at such a state is one of those 828, there is a never-taught question that *is* the right answer, with
a measured ground truth, and an **undamaged** policy to ask about it. P1 removed labels to build this
situation; it was already on disk.

## The states, and their composition — printed before any number was trusted

| | **group A — full-pool** | **group B — sampled** | **group C — on-manifold** |
|---|---|---|---|
| source | the 42 phase-3 below-floor sweeps | the 138 random-prefix sweeps | the 7 tree internal nodes |
| states | **42** | **138** | **7** |
| in-pool candidates measured per state | **996.0** (essentially the whole pool) | **21.6** | **998.0** (exact, all of them) |
| cohort size, quantiles [0/25/50/75/100] | 9 / 55 / 154 / 311 / 1,438 | 204 / 301 / 404 / 1,254 / 10,000 | 1,289 / 2,463 / 2,606 / 4,188 / 10,000 |
| prefix lengths | 3–6 | 1–15 | 0–4 |
| E1 trained on this STATE? | **no** | **no** | **yes** |
| **measured best is NEVER-TAUGHT** | **17 of 42** (17 distinct questions, 53% concepts) | **94 of 138** (51 distinct questions, 65% concepts) | **0 of 7** |
| measured best is TAUGHT | 25 of 42 | 44 of 138 | **7 of 7** (they are the tree's own splitters) |

⚠ **Teacher honesty, carried on every group-B number.** Group B measured only the **50 candidates
that sweep sampled** (20 in-pool + 30 catalogue), so "the measured best" there is the best of ~22
in-pool candidates, not of 1,000. A rank against 1,000 is still meaningful — the question is where
the policy puts a question that is *known to be good* — but group B cannot say that question is the
pool's best. **Group A can**, and group A is the primary. The two are never pooled.

⚠ **Groups A and B are off the tree's trajectory manifold**, so E1 was never trained on these *states*
either. The result is therefore about two kinds of novelty at once, and the taught/never-taught
contrast within the same states is what isolates the vocabulary axis.

Group C's cohorts are not banked but are exactly reconstructible from the validation trajectory
bank's ternary outcomes — node `root.a.b` is the users who answered `a` then `b` — and each rebuilt
cohort is **asserted against the banked `n_rows`**. Its gain vectors are the level-0 sweeps CELF did
not overwrite (those 7 nodes made a single pick), and the banked pick is asserted to be the argmax of
the banked vector.

The 238 remaining states of the 425-state bank (`heap_survivor` + `pick`) are **on-manifold tree
states with only partial candidate coverage and no banked cohort rows**, so they cannot answer a rank
question. They are counted here and excluded, rather than silently dropped.

**Gates:** canonical m=0 snap `0.15686538118354684` exact; `greedy_static_policyB_combined` rescored
to `0.25561016113602253` exactly; pool identity against the trajectory bank; `branch_of_per_user ==
branch_of`; leak check 0/10,000; both checkpoints asserted to carry this pool and this descriptor
standardisation.

---

## RESULT — the full table

Rank = position of the measured-best question in the arm's ranking of the 1,000-candidate pool at
that state (0 = the arm's own top pick), cohort-mean score, already-asked candidates excluded.

### Group A — full-pool sweeps (42 states, ~996 candidates measured each)

| measured best is | arm | median rank | rank quantiles [0/25/50/75/100] | top-1 | top-5 | top-20 | mean ρ (never-taught candidates) |
|---|---|---|---|---|---|---|---|
| **NEVER-TAUGHT** (n=17) | **E1** | **208** | 23 / 59 / 208 / 445 / 978 | **0.0000** | **0.0000** | **0.0000** | +0.1036 |
| | shuffle (floor) | 262 | 63 / 187 / 262 / 549 / 870 | 0.0000 | 0.0000 | 0.0000 | +0.0700 |
| | popularity (dumb) | 383 | 55 / 148 / 383 / 583 / 816 | 0.0000 | 0.0000 | 0.0000 | **+0.1356** |
| **TAUGHT** (n=25) | **E1** | **71** | 0 / 1 / 71 / 251 / 628 | **0.2400** | **0.4000** | **0.4400** | +0.1993 |
| | shuffle | 53 | 0 / 9 / 53 / 240 / 772 | 0.0400 | 0.1200 | 0.3600 | +0.1109 |
| | popularity | 444 | 12 / 195 / 444 / 522 / 924 | 0.0000 | 0.0000 | 0.0800 | +0.2092 |

### Group B — sampled sweeps (138 states, ~22 in-pool candidates measured each)

| measured best is | arm | median rank | top-1 | top-5 | top-20 | mean ρ (never-taught candidates) |
|---|---|---|---|---|---|---|
| **NEVER-TAUGHT** (n=94) | **E1** | 374 | 0.0000 | 0.0000 | 0.0106 | +0.1489 |
| | shuffle (floor) | 378 | 0.0000 | 0.0000 | 0.0000 | +0.1742 |
| | popularity (dumb) | **244** | 0.0000 | 0.0000 | **0.0851** | **+0.2102** |
| **TAUGHT** (n=44) | **E1** | **192** | 0.0000 | 0.0227 | 0.0455 | +0.1649 |
| | shuffle | 206 | 0.0227 | 0.0682 | 0.1136 | +0.0755 |
| | popularity | 503 | 0.0000 | 0.0227 | 0.0227 | +0.1464 |

**Reading it in one paragraph.** E1's top-1 rate on the primary group goes from **0.24 on taught
questions to 0.00 on never-taught ones**, and its top-5 from **0.40 to 0.00**, while the popularity
comparator sits at 0.00 in both. Where it was taught, E1's median rank beats popularity by 253 ranks
with an interval excluding zero; where it was not, the same comparison is a null in group A and runs
*against* E1 in group B. Its own label-shuffled control is indistinguishable from it on never-taught
questions and clearly below it on taught ones (top-1 0.04 vs 0.24 in group A). Every one of those
contrasts points the same way.

**A small honest asymmetry worth printing rather than smoothing:** in group A the shuffled arm's
*median* rank on taught questions (53) is nominally better than E1's (71), while its top-1 (0.04 vs
0.24) and top-5 (0.12 vs 0.40) are far worse and the paired rank difference is a null
(−8.6 [−105, +89]). A shuffled policy learns the position-wise label marginal, and the taught
questions are exactly the high-marginal ones, so it lands them in the broad middle of its ranking
often — it just never puts them first. The top-k rates are the discriminating statistic here, not the
median.

---

## WHAT THIS SETTLES

1. **The falsifier fired.** On never-taught questions the action policy does not beat a popularity
   ranking (null in the full-pool group, significantly worse in the sampled group) and does not beat
   its own label-shuffled control. **No similarity-based generalisation in the action policy is
   demonstrated, and it is reported failed.**
2. **The measurement is not blind.** The same policy, on the same states, through the same code, is
   decisively better than popularity where it *was* taught — top-1 0.24, and −253 ranks with the
   interval excluding zero.
3. **The mechanism claim survives; the value claim does not.** The head can score any question in one
   forward pass (that is architecture, and it is real). What it cannot do is put a *useful* number on
   a question no gradient ever touched. This is the same finding as E1 §4.2, now against ground truth.
4. **Nothing ranks this surface well.** Best ρ anywhere is ≈ +0.21, and it belongs to popularity. The
   honest frame is "a weak ranker, weakest off its vocabulary", not "a good ranker that fails to
   generalise".
5. **The reaction head is a different story and stays intact.** Experiment Q measured that the
   *reaction* head predicts outcomes for questions it never saw at AUC 0.98167 against a 0.96285 fair
   backoff. **Predicting how a person will react to an unseen question generalises; deciding that it
   is the right question to ask does not.** That contrast is the chapter's honest statement of what
   the descriptor buys and where.

**Nothing about test.** No test user was built, loaded or touched.

## Cost — a property of this box

**2.7 minutes total**, all of it scoring frozen checkpoints: 180 states, 3 arms, ~1,000 candidates
each, cohort latents refolded per state. No training. The Part I.1 amendment estimated "minutes"; it
was minutes.

## Artifacts

| what | where |
|---|---|
| per-state ranks, ρ, composition, summary tables | `experiments/battery/p1b_nevertaught/p1b_never_taught.json` |
| log | `experiments/battery/p1b.log` |
| code | `src/instrument/policy_p1b.py` |

**Reproduce:** `python src/instrument/policy_p1b.py --threads 6 --i_have_author_goahead`
