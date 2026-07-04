# Protocol-Inflation Demonstration (Paper A) — Sampled-Negative vs Honest Full-Catalogue

**Date:** 2026-07-04. **Branch:** master (no commits; eval-only, checkpoints reused).
**Question.** On the *same* data, instrument, users and checkpoints, how much elicitation
headroom / policy gain does the field-standard **sampled-negative** protocol (He et al. NCF
lineage: each held-out positive ranked against 100 uniformly-sampled unrated negatives;
sampled-NDCG@10 / sampled-HR@10) *manufacture* relative to our honest **full-catalogue**
protocol — and does it change the method ordering (Krichene & Rendle predict it can) or flip a
pre-stated gate verdict?

**Bottom line.** The sampled protocol does **not** manufacture a clean "the-policy-now-wins"
story on either arena. Instead it does two Krichene–Rendle-consistent things: (1) it **inflates
absolute scores** (ML-1M NDCG@10 ×1.16–1.28; Goodreads MOSTPOP alone jumps to sNDCG@10 0.66 /
sHR@10 0.82 from an honest @510 of 0.265) so a weak arena *looks* healthy in isolation, while
(2) it **compresses the elicitation deltas** (~50% on ML-1M) and **reorders methods** — on ML-1M
the popularity heuristic overtakes the entropy heuristic on **all 5 seeds**. Our pre-stated
**delta-gate does not flip to PASS** on Goodreads (stays WEAK on both cohorts), but the sign of
the cold-cohort headroom flips (−0.0030 → +0.0035) and the absolute-score impression flips from
"fail" to "looks great." **We do not claim to discover the general sampled-metrics flaw
(Krichene & Rendle 2020; Dacrema et al. 2019; Cañamares & Castells 2020); our contribution is the
elicitation-specific instantiation on identical data/instrument and the a-priori gate framework.**

---

## Configuration (identical across both protocols)

**ML-1M (Part 1).** Canonical Paper-B harness `scripts/paper2/continuous_policy2_st.py`
(EVALCKS path, `contactor` mode for the continuous actor). Frozen V1 recommender
(`enc_concept.pt` sha `27e6c72d`; `pool_entavg.npy` sha `156c072c` = the corrected cache).
Unified pool = 600 items + 761 concepts. Held-out disjoint-targets ruler: per test user likes
shuffled with rng(seed), second half = held-out target, first half + asked items excluded.
Test = `te[300:]` (304-user representative cohort), **seed-avg {1,2,3,7,11}**, q8, **graded**
answers. Methods re-evaluated (no retraining, checkpoints reused):
- `q0`/MOSTPOP = the q=0 column (u=0 ⇒ score = popb), identical for all methods.
- `conc_pop` = static popular-concept selector (binary answers).
- `entropy` = static entropy (most-divisive answerable concept; the canonical strong baseline).
- **CASPER-R** = `policy_entdistill_ep4.pt` (Paper-B learned discrete policy; Scorer `FEATS=ext,ans`).
- **D1** = continuous actor + divisiveness-field reward, `contactor` mode, graded
  (`PAPER_C_CONT_WINNER/policy_cont_actor_DIVW_WINNER.pt`).
- `fullprof` = fold the entire known profile (warm-start ceiling).

**Goodreads multi-genre composite (Part 2).** Harness `scripts/paper2/gr_comp_health.py`
(phase-1E gate). N = 188,867 items, 1500 shelf-concepts, 500 test users (n=423 usable full-test,
n=57 cold ≤10 ratings). Frozen V1 encoder `enc_v1_grcomp.pt`; decoder popb+Q·u; ridge λ=5.
Honest ruler = held-out disjoint targets, **primary K=510** (0.0027·N), NDCG@10 alongside; tail =
Cremonesi head-33%. **Pre-stated gate: encoder full-profile NDCG@510 − MOSTPOP ≥ +0.025 AND
monotone no-harm.** Methods: MOSTPOP q0, ridge, encoder full-profile, and the entropy/random/pop
concept-elicitation curves (q0/8/20, graded geometric answers).

**Sampled-negative protocol (both).** Each held-out positive ranked against **100** uniformly-
sampled negatives drawn from items the user never rated (all rated items excluded, consistent
with each harness's exclusion of asked/revealed items). sampled-NDCG@10 = 1/log₂(rank+1) if the
positive lands in the top-10 else 0 (single relevant ⇒ IDCG=1); sampled-HR@10 = 1[rank≤10]. Ties:
positive counted below equal negatives (`>=`, matches `eval_method_sample.py`). Macro-averaged over
users. **Sampling rng fixed (base SAMPSEED=0), resampled per eval-seed** (deterministic in
(seed, user)). Code added under env-gate `SAMPLED=1` (default harness behaviour unchanged).

**Faithfulness check.** With `SAMPLED=0` the harnesses reproduce canonical exactly: ML-1M q8
seed-1 conc_pop 0.343/0.128, entropy 0.362/0.139, CASPER-R 0.361/0.150 (canon 0.360/0.152),
D1 0.382/0.180 (= the GOLD seed-1); Goodreads honest headroom +0.0107 @510 (= the recorded
phase-1E +0.0106). The sampled columns are additive; the honest numbers are untouched.

---

## PART 1 — ML-1M (seed-avg n=5, te[300:], q8, graded)

### Table: full-catalogue vs sampled (NDCG@10)

| method | full NDCG@10 | sampled NDCG@10 | sampled HR@10 | abs-inflation (samp/full) | Δ vs q0 (full) | Δ vs q0 (sampled) |
|---|---|---|---|---|---|---|
| q0 / MOSTPOP | 0.3099 | 0.3978 | 0.6435 | ×1.28 | — | — |
| conc_pop (popular) | 0.3465 ±.0027 | 0.4243 ±.0031 | 0.6812 | ×1.22 | +0.0366 | +0.0265 |
| entropy (static) | 0.3609 ±.0014 | 0.4201 ±.0018 | 0.6765 | ×1.16 | +0.0509 | +0.0223 |
| **CASPER-R** (learned discrete) | 0.3601 ±.0034 | 0.4257 ±.0020 | 0.6829 | ×1.18 | +0.0502 | +0.0279 |
| **D1** (continuous, graded) | 0.3779 ±.0032 | 0.4429 ±.0039 | 0.6994 | ×1.17 | +0.0680 | +0.0451 |
| fullprof (ceiling) | 0.4074 ±.0024 | 0.4709 ±.0028 | 0.7320 | ×1.16 | +0.0975 | +0.0731 |

### (a) Inflation factor
Sampled negatives inflate the **absolute** NDCG@10 by **×1.16–1.28** (q0 inflates most, ×1.28,
because random negatives are trivial for popularity to beat). But the **elicitation deltas
compress** — sampling *deflates* the very quantity a policy paper reports:

| method | Δq0 full → Δq0 sampled | shrink |
|---|---|---|
| entropy | +0.0509 → +0.0223 | −56% |
| CASPER-R | +0.0502 → +0.0279 | −44% |
| D1 | +0.0680 → +0.0451 | −34% |
| fullprof | +0.0975 → +0.0731 | −25% |

So the *directional* story survives (bigger honest gains stay bigger), but a reader of the sampled
column would see roughly **half** the elicitation prize the honest ruler reports.

### (b) Elicitation deltas and the D1 − discrete gap (both protocols)
- **method − q0** is positive under both protocols for every method (directionally similar), but
  compressed under sampling (table above).
- **D1 − CASPER-R** (the continuity headline): full **+0.0178 ±0.0040** vs sampled
  **+0.0172 ±0.0025** — essentially **preserved** (5/5 seeds positive under both). The
  continuous-vs-discrete headline is robust to protocol.

### (c) Ordering changes (Krichene–Rendle)
- **entropy vs conc_pop — clean reversal on all 5 seeds.** Full-catalogue: entropy > popular by
  **+0.0144** (all 5 seeds positive). Sampled: popular > entropy by **−0.0042** (all 5 seeds
  negative). The sampled metric makes the **popularity heuristic beat the entropy heuristic** — a
  qualitative reversal of the Paper-A/B message that divisiveness-selection > popularity-selection.
- **entropy vs CASPER-R — minor flip.** Full: entropy (0.3609) ≈ CASPER-R (0.3601), entropy
  nominally ahead. Sampled: CASPER-R (0.4257) > entropy (0.4201). The learned policy edges ahead
  only under sampling — again a protocol-induced reorder, not a real capability change.

### (d) Tail (q8)
Sampled-tail (100 tail negatives) inflates absolute tail NDCG ×~2.4 and compresses ordering:

| method | full-cat tail NDCG@10 | sampled-tail NDCG@10 |
|---|---|---|
| conc_pop | 0.1264 | 0.3171 |
| entropy | 0.1397 | 0.3152 |
| CASPER-R | 0.1521 | 0.3238 |
| D1 | 0.1782 | 0.3472 |
| fullprof | 0.2164 | 0.3873 |

On the honest tail, entropy (0.140) clearly beats conc_pop (0.126); under sampled-tail they are
tied/reversed (0.315 vs 0.317). The tail — where Paper-B's win lives — is exactly where the
sampled protocol most erodes discrimination.

---

## PART 2 — Goodreads composite (phase-1E gate, honest @510 vs sampled @10)

### FULL-TEST cohort (n=423, avg_profile 44.1)

| model | honest @510 full | honest @10 full | sampled NDCG@10 | sampled HR@10 |
|---|---|---|---|---|
| MOSTPOP (q0) | 0.2650 | 0.1973 | **0.6638** | **0.8176** |
| ridge (λ=5) | 0.2639 | 0.1923 | 0.6659 | 0.8180 |
| **ENCODER** (full-profile) | 0.2757 | 0.2067 | 0.6730 | 0.8235 |

- **Honest headroom** encoder − MOSTPOP @510 = **+0.0107 (4.0% rel)** → below the +0.025 bar →
  pre-stated gate **WEAK/FAIL** (the recorded phase-1E verdict).
- **Sampled headroom** = **+0.0092 (1.4% rel)**, sHR +0.0059 → *smaller* than honest in both
  absolute and relative terms (**inflation ×0.9** — a mild *deflation* of the delta). Sampled-
  protocol gate (same +0.025 bar) = **WEAK. NO FLIP.**
- **The manufactured signal is in the absolute scores, not the delta.** MOSTPOP *alone* posts
  sNDCG@10 **0.66** and sHR@10 **0.82** — numbers that read as a strong, healthy recommender in
  isolation, entirely masking that the honest full-catalogue NDCG@510 is 0.265 and that this is a
  popularity-saturated arena where personalization adds ~0.01. 100 uniform negatives (mostly
  obscure tail books) are trivially out-ranked by popularity, so every model looks near-ceiling.

**Concept elicitation (entropy selector), FULL-TEST:** honest elicited gain @510 q8 = +0.0022;
sampled sNDCG@10 q8 = **+0.0010** (sHR −0.0002) — compressed toward zero. The tiny honest
elicitation prize becomes tinier and noisier under sampling, not larger.

### COLD cohort (≤10 ratings; n=57, avg_profile 3.9)

| model | honest @510 full | sampled NDCG@10 | sampled HR@10 |
|---|---|---|---|
| MOSTPOP (q0) | 0.1924 | 0.7194 | 0.8450 |
| ridge | 0.1894 | 0.7228 | 0.8494 |
| **ENCODER** | 0.1895 | 0.7229 | 0.8538 |

- **Honest headroom** encoder − MOSTPOP @510 = **−0.0030** (encoder *below* MOSTPOP on full) →
  gate **FAIL**.
- **Sampled headroom** = **+0.0035** — a **SIGN FLIP** (negative → positive) manufactured purely
  by the protocol; sHR +0.0088. Still below +0.025 → sampled gate **WEAK**, so the *pass/fail*
  verdict does not flip, but the *sign of the encoder-vs-baseline comparison* does. A paper eyeing
  "encoder beats MOSTPOP?" would read NO honestly and YES under sampled negatives.
- pop-selector no-harm: honest **HURTS** (−0.0055 @510, the held-out exclusion artifact); sampled
  masks it to ≈0.

### THE headline question — gate-verdict verdict

> *Does the +0.011 honest headroom become a large, healthy-looking headroom under sampled
> negatives — would the standard protocol have PASSED our gate on an arena the honest protocol
> failed?*

**On the pre-stated delta-gate (headroom ≥ +0.025): NO — it does not flip.** Sampled headroom is
+0.0092 (full-test) / +0.0035 (cold), both still WEAK/FAIL; sampling actually *compresses* the
delta. **But the framing matters:** our gate is an *a-priori, delta-based* bar. Most published
elicitation/RS results implicitly gate on **how healthy the absolute numbers look** — and on that
implicit bar the sampled protocol **does flip the read**: a system whose honest cold-start ranking
is NDCG@510 ≈ 0.19 (barely above popularity) presents as **sNDCG@10 0.72 / sHR@10 0.85**, and the
encoder's honest *deficit* vs MOSTPOP becomes a sampled *surplus*. Had phase-1E been evaluated the
way the field standard invites, this arena would have been reported as a healthy pass. **The
a-priori delta-gate is what protected us; the sampled protocol would have manufactured the
"looks-fine" impression that the honest ruler correctly refused.**

*(Orthogonal note: the `GOODREADS_EASE_DIAGNOSTIC` shows a full item-item EASE model finds
+0.30 NDCG@510 headroom on the SAME honest protocol/users — a **model-class** gap, a different
confound from the **protocol** gap isolated here. The sampled-negative study holds the model fixed
and varies only the metric protocol.)*

---

## Positioning (honest scope of the contribution)

The general fragility of sampled offline metrics is **established, not ours**: **Krichene & Rendle,
"On Sampled Metrics for Item Recommendation," KDD 2020** show sampled metrics are inconsistent
estimators of the full-catalogue metric and can **reverse method rankings**; **Dacrema, Cremonesi
& Jannach, RecSys 2019** ("Are we really making much progress?") show weak baselines and
evaluation choices manufacture apparent progress; **Cañamares & Castells, SIGIR 2020** analyse how
negative sampling and popularity interact to bias offline comparisons; the **100-sampled-negative
protocol** itself traces to **He et al., NCF, WWW 2017** and its lineage. We reproduce their
phenomenon, we do **not** claim to have discovered it.

**Our specific contribution** is the *elicitation-specific instantiation on identical
data / instrument / users*: holding the recommender and the conversational protocol fixed and
switching **only** the ranking-evaluation protocol, we show (i) on a mature arena (ML-1M) the
sampled protocol **reorders the elicitation baselines** (popularity overtakes entropy on all 5
seeds) and **halves the reported elicitation delta**, while the continuous-vs-discrete headline
(D1 − CASPER-R ≈ +0.017) is protocol-robust; and (ii) on a new arena (Goodreads composite) it
**cosmetically inflates absolute scores ~3×** and **flips the sign** of the cold-start
encoder-vs-baseline comparison, i.e. it would have painted a gate-failing arena as healthy. This
motivates the second half of the contribution: an **a-priori, full-catalogue, delta-based gate**
fixed *before* seeing recommender numbers is what kept the Goodreads null honest — the practice,
not just the metric, is the safeguard.

---

## Reproduce

ML-1M (writes `<csv>` with sampled columns appended; ~15 min, CPU):
```
OMP_NUM_THREADS=3 NOBC=1 EP=0 SAMPLED=1 NNEG=100 SAMPSEED=0 FEATS=ext,ans ANSF=1 \
QPTS=0,2,8 EVALSEEDS=1,2,3,7,11 TESTRANGE=300:99999 \
EVALBASE=conc_pop,entropy,fullprof,contactor EVALCKS=entdistill_ep4 \
ACTORCK=PAPER_C_CONT_WINNER/policy_cont_actor_DIVW_WINNER.pt EVALCSV=<csv> \
python scripts/paper2/continuous_policy2_st.py
```
Goodreads composite (both cohorts, ~15 min):
```
OMP_NUM_THREADS=3 SAMPLED=1 NNEG=100 SAMPSEED=0 NEVAL=500 python scripts/paper2/gr_comp_health.py
```
Aggregate ML-1M CSV → seed-avg table: the harness prints `sFULL` live; per-seed means/std and
ordering flips via the scratch `agg_inflation.py`. Sampled code is env-gated (`SAMPLED=1`); with
`SAMPLED=0` both harnesses are byte-for-byte the canonical evaluators.
