# The instrument is not up to its own spec — review, causes, and the plan to fix it

> Written 2026-07-27 after the author asked why T2′ scores 2.5/4 on the gap map. The short answer is that
> the author is right: **R3 and R4 are claimed in the chapter but not delivered by the implementation**, and
> the paper's blanket `\TODO{}` markers made "run, failed, parked" look like "not yet run". This document
> reviews what was actually done, what was abandoned and why, and states the plan.

---

## 1. What was actually done (the record, not the impression)

### R1 — full-profile accuracy. **DONE, certified.**
T2′ TEST 0.3482/0.2462, paired-bootstrap tie with the RecVAE bar, empty-interview identity exact.
No gap. (`experiments/baselines/ml25m_liang/tower_t2.json`.)

### R2 — any-length fold-in. **DONE.**
Empty set → intercept, arbitrary-length sets fold, no per-user retraining. Not in dispute.

### R4 — uncertainty-native belief. **BUILT, FITTED, RUN — AND THREE GATES FAILED ON JUL 24.**
This is the part the record obscured. `src/instrument/belief_layer.py` implements the analytic
conjugate-Gaussian layer (8 fitted scalars: 2 channels × 3 confidence precisions + s0 + vfloor), it was
calibrated on real data (`belief_fit.log`, sign proof G=2291 over 414,265 concept triples →
`.cache/instrument/belief_i25.pt`), and `run_battery_phaseB.py` ran the full covariance battery on
2026-07-24 12:22–13:45. Verdicts:

| gate | result | detail |
|---|---|---|
| G1 posterior usable | **FAIL** (1 of 3 sub-gates) | G1a trace shrinks monotonically 1881.9→1805.1 over q=0..16 **PASS**; G1b directional own/other = **11.5×** vs a 2× bar **PASS**; G1c calibration ρ=0.2393 vs 0.25 bar **FAIL**, with fitted alphas (the JSON's "INIT alphas is advisory" note is stale — the log shows fitted alphas were loaded 1s before the gate ran) |
| G4 confidence/refusal | **FAIL** | fitted precisions: know-well 0.0131, vague 27.66, refuse 0.0183 — the required ordering is inverted |
| G8 Σ load-bearing | **FAIL** (by an all-budgets bar) | order-invariance exact (dev 5e-7) **PASS**; Σ-greedy beats isotropic-cI at q4/q8/q16 (+0.0089 at q16, CI [0.0071,0.0107]) but **loses at q2**, so the all-budgets criterion failed |
| G2 formal cold curves | **PASS** | — |
| G5 concept specificity | **PASS** | — |

### R3 — channel-agnostic input. **NOT DELIVERED AS CLAIMED.**
`ConceptFoldNet.forward(z_items, conc_ids, conc_vals)` looks the direction up in
`self.emb = nn.Embedding(nc, d)` — a **trained table over the fixed 1031-concept genome vocabulary**
(`concept_fold.py:89,120`). It cannot accept an arbitrary φ ∈ R^d. The centroids only *initialise* the
table; the trained weights are what deliver the measured gains, and a genuinely new concept has none.

This matters because R3's wedge against ConTS in §4 is, verbatim, "*a fixed categorical attribute schema;
no arbitrary embedding, no continuous or open token*". **That is a description of our own implementation.**
The chapter also lists "arbitrary continuous direction tokens" as one of the three architecturally-
inexpressible legs that constitute the novelty claim. As built, we do not demonstrate it.

Mitigating fact: everything *downstream* of the lookup consumes `p ∈ R^d` (`x = cat([z, p, val])`), so the
arbitrary-embedding entry point is a small code change. The claim needs **evidence**, not just an API.

### R5 — monotonicity. **EVIDENCE EXISTS (G2 formal PASSED); not re-run on the certified tower.**
The cheapest of the four gaps.

---

## 2. What was abandoned, and why

**The belief line was parked on 2026-07-24, mid-flight, and never resumed.** STATE recorded it as
"Phase B runs after the concept line settles". What actually happened: the concept channel blew up into the
positive-only-clip regression, the signed retrain, the fold-recipe fix and the distillation study, and
consumed every cycle from Jul 24 to Jul 27. The three FAILs were never triaged.

**Why it stayed invisible:** the chapter marks all R4 material `\TODO{}`, which a reader (and, evidently,
we ourselves) read as "not yet attempted". Three failed gates were sitting behind that marker. This is the
same class of error as the broken-fold figure — the record technically said something true while conveying
something false.

---

## 3. Triage: how bad is each failure, really?

**G4 is a broken GATE, not (only) a broken model.** The gate assigns the confidence tier *by channel*:
items→know-well, concepts→vague, unrated-popular probe→refuse (`run_battery_phaseB.py:13-14`). It then
demands refuse < vague < know-well. But the fit is free to discover that a concept answer is worth far more
than a single item answer — which is exactly what it found (27.66 vs 0.013), and exactly what every other
result in the project says (concepts lead the short interview; one concept splits a cold population).
**A correct model necessarily fails this gate as written**, because the gate conflates two different axes:
*channel* (item vs concept) and *stated confidence* (know-well vs vague vs refuse). Note also that only 3
of the 6 channel×confidence cells were ever exercised — `a_item[1]` and `a_conc[0]` sit untouched at 1.0.

**G1 is a real, marginal calibration miss — not a bookkeeping artifact.** I first assumed the failing
sub-gate was an init-alpha artifact, because the JSON carries the note "*g1c on INIT alphas is advisory;
the fitted-alpha rho is the certified number*". **That note is stale and wrong for this run.** The Jul-24
log shows `[belief] loaded fitted alphas from belief_i25.pt` at 12:25:11, one second before G1 started, and
the JSON records `belief_fitted: true`. So ρ=0.2393 *is* the fitted-alpha number, against a pre-registered
bar of 0.25 — a genuine miss by 0.011. G1a and G1b pass strongly; the posterior is directionally excellent
(11.5× vs a 2× bar) and shrinks monotonically, but its *magnitude* is only weakly predictive of held-item
NLL. Note the fit ran only **2 epochs** (`belief_fit.log`: ep1 L=0.6436, ep2 L=0.6293) and the alphas were
fitted on NLL, not on the calibration objective the gate scores — so a longer fit, or fitting s0/vfloor
against calibration directly, is the obvious first remedy. The stale note must also be deleted from the
gate code so it cannot mislead again.

**G8 is a genuine but narrower result than claimed.** Σ *is* load-bearing versus an isotropic covariance at
q≥4 with a clean CI. It fails only an all-budgets bar, on q2. Honest statement: Σ earns its place at
realistic budgets; we should say so and narrow R4 rather than claim or hide.

**R3 is the one substantive capability gap.** Nothing to reinterpret; it needs building and measuring.

---

## 4. The plan

Ordered by (evidence value) / (cost). **Rule for all of it: a gate may be redesigned only if the redesign
is written down BEFORE it is run, the old result is reported alongside, and the redesign is justified on
the measurement's logic — never because it makes a number pass.**

### Step 1 — Fix the belief calibration (G1c), and delete the stale note *(medium)*
ρ=0.2393 vs the 0.25 bar is real. Two things to try, in order: (a) fit longer than 2 epochs — the fit was
still improving when it stopped (L 0.6436→0.6293); (b) fit `s0`/`vfloor` against the *calibration*
objective the gate actually scores, since the current alphas are fitted on NLL and calibration is only an
incidental consequence. If neither clears 0.25, report the near-miss and narrow R4 to "directionally
correct, magnitude weakly calibrated" — which is what G1a/G1b already license. Delete the misleading
"INIT alphas is advisory" note from `run_battery_phaseB.py:178` regardless.

### Step 2 — Redesign G4 so confidence is not a synonym for channel *(pre-register first)*
The corrected gate varies stated confidence **within** a channel and reports the channel effect separately:
- *know-well* = an item the user rated with many neighbours / high support;
- *vague* = an item or concept with thin support (low `n_c`, or a sparsely-rated item);
- *refuse* = the `e_c < τ` unmeasurable band, which must move the belief by ≈0.
The ordering bar then applies **within channel** (refuse < vague < know-well for items, and again for
concepts), and the cross-channel ratio (concepts higher-precision than items) is reported as a **result**,
not a failure. All 6 cells must be exercised. The old G4 verdict stays in the record.

### Step 3 — Restate G8 honestly and narrow R4 *(no new run)*
Report: Σ-greedy > isotropic-cI at q4/8/16, CI-clean at q16; ties/loses at q2; Σ-greedy over *random* is
thin (+0.0036 at q16). Claim: "the covariance is load-bearing for question selection at realistic budgets,"
not "Σ is essential." Investigate q2 only if cheap.

### Step 4 — Make R3 true: the zero-shot concept experiment *(the real work)*
Add an arbitrary-embedding entry point to the fold (`forward` accepts `p` directly, bypassing `emb`), then
**hold out concepts from training entirely** and fold them at inference from their whitened centroid alone.
Measure member-lift AUC and the concepts-only NDCG curve for held-out concepts versus trained ones. This is
the experiment that decides whether R3 gets a ✓, a narrowed claim ("out-of-catalogue within a fixed
vocabulary"), or a retraction of the continuous-token leg. **Until it runs, the chapter must not claim
arbitrary/continuous direction tokens as a delivered capability.**

### Step 5 — Re-run the whole battery on the certified tower *(after the belief refit)*
Refit the belief layer on `t2final_best.pt`, then Phase A + Phase B end-to-end. This is what converts R5
from ∼ to ✓ (G2 formal already passes on the ep4 snapshot) and puts every gate on the certified substrate.

### Step 6 — Fix the chapter's honesty
Replace blanket `\TODO{}` on R4 with the actual state: built, fitted, gates run, these passed, these failed,
here is why and what changed. A chapter that reports a failed gate and its diagnosis is stronger than one
that hides it behind a TODO — and the battery is *itself* a contribution, so its failures are on-topic.

---

## 5. Interim honest scorecard for T2′

| req | mark | why |
|---|---|---|
| R1 | ✓ | certified, tie with the bar |
| R2 | ✓ | any-length, no retraining |
| R3 | ∼ | fixed 1031-concept table; arbitrary/continuous tokens undemonstrated (Step 4) |
| R4 | ∼ | layer built + fitted; G1 near-pass, G4 gate broken, G8 narrowed (Steps 1–3) |
| R5 | ∼ | G2 formal passes on ep4; not re-run on the certified tower (Step 5) |

The gap map's 2.5 is therefore **correct and honest as of today** — but three of those half-marks are
recoverable, two of them cheaply.
