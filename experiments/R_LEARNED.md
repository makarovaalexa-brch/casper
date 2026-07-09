# r-learned -- THE LEARNED BELIEF-COMBINER ROUTER (DIRECTIONAL 173/300)

> **DIRECTIONAL BANNER** -- 173/300 users, answerer-v1 working grid NOT frozen; v2 fold
> `.cache/i25_fold_v2_best.pt`. Re-run on the frozen 300-user grid before any citation.

Date 2026-07-09. Scripts `scripts/r_learned.py` (+ `scripts/r_learned_labels.py`). NO LLM
calls; local compute. 173 study users; 180 candidates; cold NDCG@10 0.1502; fold val 0.4256. Paired per-user bootstrap BOOT=5000.

Learner: HistGradientBoosting on 262680 population (state,candidate,label)
examples from 1990 trU users; val R^2 0.0758. Label = 1-step realized
NDCG@10 gain under the certified fold-v2 sampler. Features = SELECTION-TIME population beliefs
only (see r_learned_labels.py header).

## ARM-SYMMETRY TABLE (E7 -- what each arm sees / learns from / is constructed on)

| arm | selects using | learns outcomes from | constructed on | sees study held-out targets? |
|---|---|---|---|---|
| s-item / s-mixed (SPLIT-FAIR) | greedy schedule (fixed) | cohort-mean NDCG on the CONSTRUCTION-HALF study users | the OTHER study half (out-of-sample) | YES at construction (other half's targets) |
| r-learned-free | learned gain model on population features + blind state z | 1-step NDCG gain on trU POPULATION sims ONLY | trU population (firewall) | NO -- never |
| r-learned-anchored | s-fair order, swap on learned-gain margin | trU POPULATION sims (model) + POPULATION val (margin) | s-fair schedule + population | NO -- never |

> ASYMMETRY (stated): the static is BUILT against study-user outcomes (its construction half's
> held-out targets); r-learned's combiner and margin are BUILT only on trU population simulations.
> The router therefore competes with STRICTLY LESS arena information. A win is conservative; a tie
> is the price of the firewall; a loss beyond noise (E2) triggers diagnosis.

## Pre-registered reads (printed BEFORE results)

Verdicts use the POOLED estimate (all 4 split-estimates' per-user deltas concatenated), T=12.

- **(i) FIRST FAIR LEARNED-ADAPTIVITY WIN:** r-learned-anchored vs s-fair pooled CI excludes 0
  and is POSITIVE -> the learned belief-combiner beats the split-constructed static under strictly
  less arena information.
- **(ii) DOES THE LEARNER USE VIVIDNESS?** the VIVIDNESS group's permutation importance exceeds
  the noise-floor feature's importance (+1 sd) -> the learner weights vividness once value is
  controlled (the author's hypothesis, made measurable). Reported for VALUE / VIVIDNESS /
  ANSWERABILITY side by side.
- **(iii) EXPRESSIVENESS:** r-learned-free does not lose to r-learned-anchored beyond noise.
- **(iv) E2 FLOOR:** neither router loses to s-fair beyond noise (pooled CI lower bound >= -0.01).

## Feature importances (read ii -- value vs vividness vs answerability)

Permutation importance (r2 drop) on the by-user val split; noise-floor feature = the reference floor.

| feature | importance | +/- sd | > floor? |
|---|--:|--:|---|
| turn | +0.14654 | 0.00466 | yes |
| pop_answerability | +0.03313 | 0.00662 | yes |
| taste_x_answerability | +0.01475 | 0.00146 | yes |
| norm_z | +0.01101 | 0.00097 | yes |
| taste_x_vividness | +0.00643 | 0.00112 | no |
| pop_vividness | +0.00441 | 0.00128 | no |
| noise_floor | +0.00436 | 0.00244 | FLOOR |
| taste_x_turn | +0.00300 | 0.00101 | no |
| taste | +0.00021 | 0.00092 | no |
| ch_attr | +0.00014 | 0.00021 | no |
| value_tier | +0.00000 | 0.00000 | no |
| answered_count | +0.00000 | 0.00000 | no |
| ch_item | -0.00002 | 0.00009 | no |
| ch_concept | -0.00028 | 0.00019 | no |
| pop_value | -0.00098 | 0.00108 | no |

**Grouped importance:**

| group | importance |
|---|--:|
| STATE | +0.15756 |
| ANSWERABILITY | +0.04788 |
| VIVIDNESS | +0.01085 |
| NOISE_FLOOR | +0.00436 |
| TASTE | +0.00321 |
| CHANNEL | -0.00017 |
| VALUE | -0.00098 |

- VALUE -0.00098 | VIVIDNESS +0.01085 | ANSWERABILITY +0.04788 | NOISE FLOOR +0.00436.
- **(ii) learner USES vividness:** True (vividness group importance > noise floor +1sd).

## Anchored swap margin (fit on POPULATION val, firewall)

Fitted margin = **0.01** (grid [0.0, 0.001, 0.002, 0.005, 0.01, 0.02]); pop-val scores {"0.0": 0.252, "0.001": 0.252, "0.002": 0.252, "0.005": 0.2521, "0.01": 0.2523, "0.02": 0.2511}.

## Per split-estimate (anytime NDCG@10)

### seed0_evalA

| budget | s-item | s-mixed | r-free | r-anch | anch-vs-item [CI] | free-vs-item [CI] |
|---|--:|--:|--:|--:|---|---|
| T=8 | 0.1916 | 0.1962 | 0.1892 | 0.1916 | -0.0000[-0.0321,+0.0310] | -0.0024[-0.0349,+0.0293] |
| T=12 | 0.1900 | 0.1959 | 0.1851 | 0.1863 | -0.0038[-0.0362,+0.0278] | -0.0049[-0.0392,+0.0286] |
| T=24 | 0.1860 | 0.1954 | 0.1804 | 0.1769 | -0.0091[-0.0404,+0.0215] | -0.0056[-0.0401,+0.0279] |

### seed0_evalB

| budget | s-item | s-mixed | r-free | r-anch | anch-vs-item [CI] | free-vs-item [CI] |
|---|--:|--:|--:|--:|---|---|
| T=8 | 0.2178 | 0.2343 | 0.2032 | 0.2053 | -0.0125[-0.0427,+0.0177] | -0.0146[-0.0468,+0.0186] |
| T=12 | 0.2189 | 0.2348 | 0.1962 | 0.2022 | -0.0167[-0.0490,+0.0149] | -0.0227[-0.0602,+0.0151] |
| T=24 | 0.2154 | 0.2354 | 0.1939 | 0.1898 | -0.0257[-0.0598,+0.0065] | -0.0216[-0.0649,+0.0204] |

### seed1_evalA

| budget | s-item | s-mixed | r-free | r-anch | anch-vs-item [CI] | free-vs-item [CI] |
|---|--:|--:|--:|--:|---|---|
| T=8 | 0.2136 | 0.2225 | 0.1913 | 0.1968 | -0.0168[-0.0478,+0.0148] | -0.0223[-0.0546,+0.0110] |
| T=12 | 0.2100 | 0.2225 | 0.1846 | 0.1955 | -0.0145[-0.0469,+0.0184] | -0.0255[-0.0615,+0.0115] |
| T=24 | 0.2090 | 0.2202 | 0.1768 | 0.1838 | -0.0252[-0.0606,+0.0090] | -0.0321[-0.0729,+0.0089] |

### seed1_evalB

| budget | s-item | s-mixed | r-free | r-anch | anch-vs-item [CI] | free-vs-item [CI] |
|---|--:|--:|--:|--:|---|---|
| T=8 | 0.2007 | 0.2168 | 0.1985 | 0.2007 | +0.0000[-0.0288,+0.0288] | -0.0022[-0.0321,+0.0280] |
| T=12 | 0.2017 | 0.2172 | 0.1966 | 0.2012 | -0.0006[-0.0292,+0.0281] | -0.0051[-0.0363,+0.0262] |
| T=24 | 0.2009 | 0.2183 | 0.1971 | 0.1984 | -0.0025[-0.0309,+0.0257] | -0.0039[-0.0364,+0.0284] |

## POOLED verdicts (all 4 split-estimates concatenated)

| contrast | T=8 | T=12 | T=24 |
|---|---|---|---|
| r-anchored - s-item (FAIR) | -0.0073[-0.0224,+0.0083] | -0.0089[-0.0244,+0.0073] | -0.0156[-0.0313,+0.0005] |
| r-free - s-item (FAIR) | -0.0104[-0.0261,+0.0062] | -0.0146[-0.0316,+0.0033] | -0.0158[-0.0339,+0.0027] |
| r-anchored - s-mixed (FAIR) | -0.0189[-0.0289,-0.0079] | -0.0213[-0.0320,-0.0097] | -0.0301[-0.0425,-0.0169] |
| r-free - s-mixed (FAIR) | -0.0219[-0.0326,-0.0103] | -0.0270[-0.0394,-0.0136] | -0.0303[-0.0447,-0.0148] |
| r-free - r-anchored (expressiveness) | -0.0030[-0.0056,-0.0003] | -0.0057[-0.0102,-0.0009] | -0.0002[-0.0069,+0.0069] |

## VERDICTS

- **(i) first fair learned-adaptivity win:** r-anchored - s-item @12 = -0.0089[-0.0244,+0.0073] -> **TIE**.
- **(ii) learner uses vividness:** VIVIDNESS importance +0.01085 vs noise floor +0.00436 (+1sd +0.00680) -> **True**. (VALUE -0.00098; ANSWERABILITY +0.04788.)
- **(iii) expressiveness:** r-free - r-anchored @12 = -0.0057[-0.0102,-0.0009] -> free LOSES to anchored.
- **(iv) E2 floor:** neither router loses beyond noise -> **False**.

## SYNTHESIS -- what the learned belief-combiner establishes (DIRECTIONAL 173/300)

**Headline: the LEARNED combiner TIES the split-constructed s-item static and LOSES to the stronger
s-mixed static; it does NOT deliver the first fair learned-adaptivity win. The substantive result is
in the feature importances -- the learner, given value + vividness + answerability and told to
optimize NOTHING in particular, discovers that ANSWERABILITY (population P(k>=1)) is the dominant
predictor of realized 1-step gain in this abundant arena, uses VIVIDNESS weakly-but-above-noise, and
does NOT use the population VALUE feature at all.**

1. **(i) No fair win, honest tie vs s-item, clean loss vs s-mixed.** Pooled r-anchored - s-item @12
   = **-0.0089 [-0.0244,+0.0073]** (CI includes 0 = TIE); at T=24 -0.0156 [-0.0313,+0.0005]
   (borderline). But s-mixed is the STRONGER fair static, and r-anchored - s-mixed @12 =
   **-0.0213 [-0.0320,-0.0097]** (CI excl 0 = LOSS). The router was anchored on s-item (the primary
   family / P3 s-best); anchoring on s-mixed instead would restore the tie-by-construction floor
   against it (not run). Net: on the DE-CONTAMINATED baseline the learned adaptivity ties the item
   static and loses to the mixed static -- it does not overturn "statics optimal in this arena".
   This directly measures what STATIC_CONTAMINATION could only annotate: the +0.02 "de-biased win"
   was an annotation extrapolation; the direct split-fair measurement is a tie-to-loss.

2. **(ii) The author's question, made measurable.** Grouped permutation importance (candidate-varying
   features only -- turn/norm_z/answered_count are CONSTANT across candidates at a state, so they do
   not affect selection despite dominating total variance):
   - **ANSWERABILITY +0.0479** (pop_answerability +0.033, taste_x_answerability +0.015) -- DOMINANT.
   - **VIVIDNESS +0.0109** (taste_x_vividness +0.006, pop_vividness +0.004) -- ABOVE the noise floor
     (+0.0044, +1sd +0.0068) but ~4x smaller than answerability; pop_vividness alone ~= the floor.
   - **VALUE -0.0010** (pop_value, value_tier) -- at/below the noise floor: the cold population-value
     feature is UNUSED, subsumed by answerability (popular entities are both answerable and valuable).
   Reading: "does vividness feed selection once value is controlled?" -- the learner weights vividness
   marginally above noise, but the controlled-for dominant is ANSWERABILITY, not value. The design
   "optimizes nothing for answerability per se", yet the outcome labels teach the learner that in this
   ~abundant-answerability arena the 1-step gain is decided first by whether the question is answered
   at all. This is the honest, non-circular version of the program's recurring finding.

3. **(iii) Expressiveness fails the wrong way.** r-free LOSES to r-anchored (@12 -0.0057
   [-0.0102,-0.0009], CI excl 0): the unconstrained argmax policy is WORSE than tethering to the
   fair static. The learner's per-step gain model is not expressive/accurate enough to beat its own
   anchor -- the static schedule carries sequence value the myopic 1-step model misses (the same
   optimization-gap the program has hit 8+ times). r-free - s-item stays a tie (CI includes 0) only
   because the loss to the static and the anchor's help roughly cancel.

4. **(iv) E2 floor.** Strict flag = False: the pooled r-anchored/r-free vs s-item CIs have lower
   bounds below -0.01 (they are wide, n straddles 0). The clean statement: TIE vs s-item (CI incl 0),
   LOSS vs s-mixed (CI excl 0). No E2 violation vs the anchored family; the s-mixed loss is a genuine
   deficit, addressable only by anchoring on s-mixed.

**Bottom line.** A fully firewalled learned belief-combiner -- value + vividness + answerability as
features, combination learned from population NDCG outcomes, nothing tuned for answerability -- does
not beat the de-contaminated static in this arena; it ties the item static and loses to the mixed
static. Its one positive, measurable contribution is the LEARNED WEIGHTING itself: answerability is
the dominant selection signal, vividness is used weakly (above the noise floor), and cold value is
not used -- the author's "vividness feeds selection once value is controlled" hypothesis is only
weakly supported, and it is ANSWERABILITY, not value, that the learner controls for. Consistent with
the closed STATE_2026-07-08 conclusion; the fair, learned, firewalled apparatus does not reopen it.

## DIRECTIONAL caveats

- 173/300 users, answerer-v1 grid UNFROZEN; v2 fold; re-run on the frozen 300-user grid before
  citation. All numbers provisional.
- Label environment = the certified fold-v2 sampler (build_reveal_v2) UNMODIFIED; know_well is drawn
  at population base rates (taste-independent), so a taste-conditional vividness premium is NOT
  injected -- the vividness belief earns its (weak) importance honestly, not by construction.
- Candidate universe = the same 180 CANDS entities at label-gen and eval (exact feature transfer).
- Learner = HistGradientBoosting (400 trees, leaf-31, l2=1.0) on 262,680 population examples from
  1990 trU users; val R^2 0.076 (per-example gain is noisy; ranking is what the policy uses).
- Anchored margin fit on 400 POPULATION val users (firewall) vs a pop_value-ranked population anchor;
  the same scalar margin (0.01) transfers to the study anchored policy.
- Artifacts: labels .cache/r_learned_labels/chunk_0_*.npz (14 chunks) + manifest; feature tables
  .cache/r_learned_feat.npz; model+importances .cache/r_learned_model.pkl; results
  .cache/r_learned_results.json.

