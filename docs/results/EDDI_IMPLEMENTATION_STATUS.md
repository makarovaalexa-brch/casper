# EDDI implementation status — what the arm actually computes (2026-08-13)

Read-only audit, prompted by an apparent contradiction in `new_chapters/chapterB_v2/chapterB_v2.tex`:
a `\REDO` at line 701 demands the EDDI arm be re-run with a three-outcome response model, while the
paragraph at lines 728–739 reports that variant as "Implemented and measured". Evidence below is
from code, run artifacts and the git record only.

## VERDICT: **EDDI IS NOT FINE**

Two separate defects, neither discharged. The `\REDO` is **not** stale — but it is **mis-scoped**:
the three-outcome selector *is implemented* (since 2026-08-05), it has simply never been run at
contract scale. The only number behind the "implemented and measured" paragraph is a **200-user
smoke whose own artifact is stamped `"SMOKE_NOT_CITABLE": true`**, and the smoke does not
unambiguously support the sentence as written.

---

## 1. What the shipped arm computes

`src/baselines/eddi_selector.py`, `class EDDIInfoGainSelector` (`name = "eddi_ig"`, lines 95–179),
over the trained Partial-VAE `eddi_pvae` (full-profile 0.2153/0.1077, ep31).

The whole acquisition is lines 134–145:

```python
p = torch.softmax(self.model.decode(mu0[None, :])[0], dim=0)          # (n_items,)
...
kl = 0.5 * torch.sum((lv0[None, :] - lvi) + (vari + (mui - mu0[None, :]) ** 2) / var0[None, :] - 1.0, dim=1)
R[s:e] = p[s:e] * kl
```

i.e. `R(i) = p_i * KL( q(z | c_O + Phi_i) || q(z | c_O) )`.

### (a) Full Eq. 9, or one term? — **ONE TERM. The first.**

Only the latent-movement term is computed. There is no target term anywhere in the file, and no
sampling over a target variable. Confirmed independently by our own snap script,
`scripts/baselines/eddi_native_snap.py` lines 19–23 and 30:

> `src/baselines/eddi_selector.py` implements only the FIRST term of the paper's Eq. (9),
> `E_{x_i} KL[ q(z|x_i,x_O) || q(z|x_O) ]`, dropping the target-correction term …
> `eddi_latent : gain(i) = E_{x_i} KL[ q(z|x_i,x_O) || q(z|x_O) ]   <-- Eq. (9) first term = OURS.`

**The drop was deliberate but the stated reason is now known to be false.** The file's own header
(lines 20–26) justifies it as a firewall protection ("`p(z | x_phi, x_O)` is not a quantity our
protocol lets us form without leaking the evaluation targets"), and
`docs/results/BASELINE_OBJECTIVE_AUDIT.md` row 11 and its note (lines 34, 69–71) repeat that
justification verbatim and conclude the row "is not an oversight — it is the honest version of that
method under our protocol." The SNAP-1 commit `b4a4783` retracts exactly this:

> Eq. 9's second term SAMPLES its hypothetical answers from the model and leaks nothing, so the
> term was dropped for a reason that does not hold.

**The full two-term Eq. 9 is implemented NOWHERE in the repo.** Note also that the snap's *passing*
arm, `eddi_target`, is a **Gaussian predictive-variance proxy for the paper's Eq. (6)**, not the
Eq. 9 subtraction — so even the snap does not contain the object TODO 1.3 asks for.

### (b) Two-outcome or three-outcome? — **BOTH EXIST IN CODE; the shipped row is two-outcome.**

- `EDDIInfoGainSelector` (`eddi_ig`), lines 95–179: two-outcome. `select()` line 159 groups state on
  `tuple(sorted(i for (i, lv) in answered[u] if lv >= LIKE_LEVEL))` — **the LIKE set only**. The
  dislikes the user actually gave are discarded, and "not seen" and "disliked" collapse into the
  same zero-KL branch (header lines 34–40). This is the arm in the chapter table.
- `EDDIThreeOutcomeSelector` (`eddi_ig3_signed` / `eddi_ig3_zero`), lines 257–354, added 2026-08-05:
  three-outcome. `R3(i) = a_i * [ q_i * KL(c_O + Phi_i || c_O) + (1 - q_i) * KL_dislike ]` (line
  322), with `a_i` = structural answerability from the graded TRAIN matrix only
  (`structural_answerability`, lines 235–254, train-only, no LLM artifact) and state keyed on
  `(likes, dislikes)` (lines 330–332). Wired into the harness at
  `src/baselines/adaptive_harness.py:750–759`.

So "the acquisition cannot represent … the event that no answer arrives" is true **of `eddi_ig`**,
and `eddi_ig` is what every published number of ours uses.

### (c) Which variant produced which number?

**`tab:battery` `.1537/.0336 … .1917/.0749` = `eddi_ig`, two-outcome, 10,000 test users.**
`experiments/baselines/interview/a9_battery_policyB.json`, key `"latent-only info-gain"`,
`"type": "adaptive"`, `"source": "act1_per_user_vectors.npz"`:

```
"1": full 0.15371633631220374  tail 0.033625404710075815
"2": full 0.15394451759743155  tail 0.03726325490119794
"4": full 0.16180916005061272  tail 0.04320321824772285
"8": full 0.17470264232790414  tail 0.06111285331915891
"16": full 0.19168491364176013 tail 0.07493459086516495
```

Registered as EDDI's row by `src/baselines/a9_battery_policyB.py:84`:
`("latent-only info-gain", "act1", "ours", "eddi_ig", "adaptive", "in-family")`. The chapter's
`3.84` answers at k=16 is the same arm (`experiments/baselines/interview/run.log:2699`:
`INTERVIEW eddi_ig k=16 done in 12.65m; answered=3.84/16; distinct ask-prefixes=1993` — which is
also the `1{,}993` in `tab:prefixes`).

**The "implemented and measured" three-outcome result = a 200-user SMOKE.**
The only artifact in the repo containing `eddi_ig3` is
`experiments/baselines/interview/SMOKE_adaptive_table_SNAP3_smoke.json`, whose `meta` reads:

```
"n_test_users": 200, "smoke_users": 200, "SMOKE_NOT_CITABLE": true,
"selectors": ["eddi_ig", "eddi_ig3", "eddi_ig3_zero", "entropy0-static"], "budgets": [1,2,4,8]
```

(`grep -rl eddi_ig3 experiments/` returns only this file and `run.log`.) Its k=8 cells:

| arm | answered | full | tail |
|---|---|---|---|
| `eddi_ig` (two-outcome) | 1.915 | 0.17390 | 0.05996 |
| `eddi_ig3` (signed) | 1.595 | 0.18391 | **0.05167** |
| `eddi_ig3_zero` | 1.550 | 0.18638 | 0.06818 |

The commit message of `b4a4783` quotes "FEWER answers (1.92 -> 1.55 at k=8) but better quality
(+0.0125 full, +0.0082 tail)" — that is the **`_zero`** row, not the default `eddi_ig3`. The default
signed variant is **worse on tail** at k=8 (0.0517 vs 0.0600). No paired CIs, no 10,000-user cohort,
no budget 16, one consumer.

## 2. The second, independent defect: the weight is not the paper's marginal

Header lines 221–228 and commit `b4a4783` both record it: the Mult-VAE^PR decoder's `softmax`
normalises **across items**, so `p_i` is a relative next-draw probability, not the per-variable
marginal `p(x_i | x_O)` that Ma et al.'s factorised likelihood requires. Measured on the frozen
checkpoint at the empty state, `log p` spans 7.37 decades against the KL term's 1.67, with
`corr(log p, log R) = 0.965` — **our EDDI arm is a most-probable-item selector wearing
information-theoretic clothing.** This is not a rounding issue; it is the dominant term.

Caveat on provenance: that diagnostic exists only as prose in the docstring and the commit body. I
found **no artifact or script that recomputes 7.37 / 1.67 / 0.965**. It should be re-derived before
it appears in the chapter as a measured number.

## 3. The native snap — this is the load-bearing evidence

`experiments/baselines/EDDI_NATIVE_SNAP.json`, Ma et al. Fig-3 datasets (Boston/Energy/Wine),
2 seeds, PN encoder, AUIC average rank (lower better), `pooled_avg_rank`:

```
sing_eddi_target  2.3806
eddi_target       2.4111     <- their reward: reproduces EDDI ~ SING << RANDOM
rand              3.2361
eddi_latent       3.4022     <- OURS: LOSES TO RANDOM
sing_eddi_latent  3.5700
```

**Our shipped acquisition ranks below random ordering on the authors' own benchmark.** That is a
failed snap under HARD RULE 2, and the chapter is right to relabel the row "latent-only info-gain".
`docs/design/PAPER_B_TODO.md:32` already records "Six native snaps: EDDI **FAILED**".

## 4. Against the paper

We do **not** hold the EDDI PDF (`external_literature/papers/` has no `ma2019eddi*`; only the INDEX
row at line 94 and the bib entry `new_chapters/references.bib:506`). Equation content here is taken
from our own snap script, which was written against the primary text and whose `eddi_target` arm
reproduces the paper's published Table-3 ordering — so the reconstruction is validated by its own
replication, not merely asserted. Two consequences follow, both checkable:

- Eq. 9 = `E_{x_i}[ KL(q(z|x_i,x_O) || q(z|x_O)) − E_{x_phi} KL(q(z|x_i,x_phi,x_O) || q(z|x_phi,x_O)) ]`.
  Our arm is the first bracket only. Commit `b4a4783` further notes that App. C.3 of the paper
  argues *against* the first term alone (it is the BALD-style gain), so we ship the reduction the
  authors explicitly warn about, under their name.
- **Can a two-outcome response model represent an unanswerable question at all? No.** The Partial
  VAE's premise is that an unobserved variable is an *absence*, not a value. Our two-outcome arm maps
  absence and dislike onto the same "observed set unchanged" branch, so the acquisition has no slot
  in which the probability of getting no answer can appear. It is the three-outcome instantiation,
  not the two-outcome one, that is faithful to the partial-observation framing.

## 5. What a reviewer can legitimately object to

1. **"You crippled the baseline."** Fully available. We drop a term of the published criterion, we
   drop it for a reason our own snap shows is invalid, the reduced criterion loses to random on the
   authors' data, and the retained weight is a mis-instantiation that turns the arm into a
   popularity ranker. A chapter whose claim is "published adaptive methods lose to a static order"
   cannot carry EDDI's name on that row. The relabelling to "latent-only info-gain" plus the
   reported snap failure is the correct mitigation and it should stay; it is not a substitute for
   implementing the method.
2. **A smoke reported as a measurement.** Lines 731–732 assert an unqualified measured result. The
   backing artifact is 200 users, self-labelled not citable, with the two three-outcome variants
   disagreeing on the tail axis. Everything else in the chapter is 10,000 users with paired CIs;
   this one sentence is not, and nothing on the page says so.
3. **A stale justification in a results doc.** `BASELINE_OBJECTIVE_AUDIT.md` lines 69–71 still
   asserts the dropped term "is not an oversight — it is the honest version of that method under our
   protocol", which the snap falsified on 2026-08-05. That doc needs a correction (flagged, not
   made, by this audit).

## 6. Cost of discharging each item

**(i) Run the three-outcome arm at contract scale — CHEAP, no new code.**
Both selectors are wired (`adaptive_harness.py:750–759`), `score_vector` exists for both, and the
harness spec strings `eddi_ig3` / `eddi_ig3_zero` are live. Anchors: `eddi_ig` at 10,000 users cost
12.65 min of interview at k=16 with 1,993 distinct states (`run.log:2699`); scoring runs ~0.6 min per
(consumer, budget) cell. The three-outcome arm keys on `(likes, dislikes)` so it branches harder, and
`signed` does two KL sweeps per state; the smoke ratio at k=8 was 72 vs 37 states and 3.6 vs 0.39
interview-minutes. Estimate **2–6 CPU-hours** for both variants over budgets 1/2/4/8/16 on the
certified pair, plus one `act1_paired.py` bootstrap pass (~26 min, cf. `a9_battery_policyB.json`
`"seconds": 1574.7`). This is a half-day box job and should be done before the paragraph stands.

**(ii) Implement full Eq. 9 — REAL WORK, needs a design sheet.**
The second term requires an inner expectation over a designated target `x_phi`, which cold-start
ranking does not have; choosing the CF analogue (held-out-like set? decoder output distribution?
predictive variance of the top-k scores, as `eddi_native_snap.eddi_target` does on tabular data?) is
a modelling decision that changes what the row means, and the inner expectation multiplies
per-candidate cost. Roughly **1 day of build + smoke + a design sheet**, then a run of unknown but
strictly larger cost than (i). TODO 1.3 stays open.

---

## Recommended chapter edits (NOT MADE — for the author to apply)

**Line 701–703, the `\REDO`: narrow it, do not discharge it.** Current text asks for something
partly done. Replace with a marker that says the selector exists and the run is owed, e.g.:

> `\REDO{the three-outcome selector is implemented (\texttt{eddi\_ig3}) but has only been measured on
> a 200-user smoke; run it on the 10,000-user cohort with paired intervals before the mechanism claim
> and the paragraph below stand}`

**Lines 731–732 — this is the wrong sentence in the chapter.** As written:

> "It does not. Implemented and measured, the three-outcome variant collects \emph{fewer} answers
> than the two-outcome one while producing better rankings from them."

it reads as a 10,000-user result on the chapter's contract. It is a 200-user smoke, uncited, with no
CIs, and the two variants disagree on the tail. Either qualify it in place —
"On a 200-user smoke (not on this chapter's contract, and reported as provisional), the
three-outcome variant collects fewer answers … " — or run (i) above and keep the sentence unchanged.
**Do not leave it as is.**

**Lines 715–726 (the `$^\dagger$` paragraph) are correct and should not be touched.** They already
state the reduction, the failed snap, and that the leakage worry was wrong. The only thing missing
is that the *retained* weight is also non-conformant (§2 above); one clause could be added if the
author wants the full indictment on the page.

**Line 733–734** attributes the popularity behaviour to "a per-item probability term spanning seven
orders of magnitude". That is the 7.37-decade figure, which currently has no reproducible artifact
(§2). Either recompute it or drop the numeral.
