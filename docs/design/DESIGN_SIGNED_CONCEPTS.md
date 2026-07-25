# DESIGN — Signed concept answers (four-band SEL+VAL) + signed retrains of C-lite / C-full

> Author GO 2026-07-25 (after the adversarial review + the amended triple gate). This sheet is the
> pre-registered design for the signed-answer construction and the two retrains. Written and committed
> BEFORE the retrains launch (HR10 + design-sheet rule).

## 1. The problem, with archaeology

**The clip.** The concept answer value used everywhere since Jul-24 was
`v = clip(log lift / log lift_top, +0.25, 1.0)` — born in `sel_top_concepts()`
(`src/instrument/concepts_only_curve.py`, first committed in the curve/ladder commit lineage at
c89d4d2→7f45fd2), then propagated into `concept_fold.make_example`/`quick_val` (C-lite training
labels), `train_tower_t2.sel_value_to_level` (C-full concept-token levels), and
`tradeoff_ledger.build_shared` (deployment answer values). The R2-validated source construction
(`scripts/bpool_r2.py`, Jul-18: SEL watch-lift explains 0.376 of concept affinity variance, VAL
residual-rating 0.042, LLM ordinal 0.017) is **signed and continuous — it was never clipped**; the
clip entered downstream when the curve harness needed a positive graded value.

**The deployment decline mechanism.** In the realizable fixed-bank protocol (global member-mass
order, answered iff >=2 rated members), most answers deep in the bank are lift<=1 concepts. The clip
folds every one of them as a WEAK LIKE (+0.25): accumulated weak-positive poison. Measured (ledger,
clite rung, 10k COLD_SEED users): 0.1304 -> 0.1291 -> 0.1242 -> 0.1160 across q=2/4/8/16 — below the
0.1279 intercept by q8.

## 2. The triple gate (eval-only; both runs committed, `experiments/battery/signed_sel_gate.json`)

v1 (own binning) and AMENDED (exact bpool_r2 port) runs, same 10k users, snap control vs the ledger
row PASS (<=4e-5 at every q). Amended pre-registration: S = train p90 |SEL+VAL| over ALL 140,768
train users (68.3M answerable cells), TAU_refuse=1.5, LAM=3.0, train-only item means.

| arm (amended run) | q2 | q4 | q8 | q16 |
|---|---|---|---|---|
| clip-up comparator | 0.1304 | 0.1291 | 0.1242 | 0.1160 |
| **Arm1 signed (bpool_r2 port)** | **0.1317** | **0.1310** | **0.1311** | **0.1254** |
| value-permutation sanity | 0.1232 | 0.1277 | 0.1228 | 0.1153 |
| Arm2 popularity-counterfeit | 0.1298 | 0.1274 | 0.1231 | 0.1171 |
| Arm3 meh-only | 0.1317 | 0.1303 | 0.1277 | 0.1191 |

Findings: (a) signed beats clip-up at EVERY q (+0.0093 at q16, CI [0.0082, 0.0105]); (b) the
counterfeit reproduces only 12% of the gain -> **taste-real, no hard kill**; (c) value-permutation
collapses the gain (+0.0101 CI-clean) -> the VALUES carry the signal, not fold cardinality; (d)
Arm3 shows clip removal alone is NOT enough (still declines) — the negative/neutral information is
load-bearing; (e) **volume leak (KT-A3): ridge R2(fold latent @q8 -> log user volume) jumps 0.025
(clip-up) -> 0.260 (signed)** — the count/magnitude of solid-support negatives encodes user volume.
**OOD-asymmetry ruling (author):** C-lite was trained on v in [0.25,1] likes only, so
dislike/neutral inputs are out-of-envelope for the trained module — Arm1's mild q16 dip is
INCONCLUSIVE, not a kill; the retrain is the fair in-envelope test. The only hard kill (counterfeit
>=70%) did not fire.

**Precedent:** the pbC-era stack folded signed ordinal item values natively (graded dislike levels
0..4 in the tower's FiLM tables; SIGN_GAMMA_V2 hard-wires graded negation) — signed concept answers
bring the concept channel to parity with the item channel's existing sign handling.

## 3. The signed answer construction (implemented ONCE, shared: `src/instrument/signed_answers.py`)

Exact bpool_r2 components on the all-bands fold-in (held targets excluded — C2 clause;
"watched" = rated at any band, as in bpool_r2):

    n_c  = # user's watched members of c        e_c = |watches| x pexp_c   (pexp = member pop mass)
    SEL  = log2((n_c + 0.5) / (e_c + 0.5))                                  [PMI-form watch-lift]
    NPMI = SEL / (-log2((n_c + 0.5) / (nu + 1)))                            [normalized to [-1,1]]
    VAL  = (n_c / (n_c + 3)) * mean(star - TRAIN item mean)                 [shrunk residual rating]
    v_raw = NPMI + w_val * VAL,   w_val pre-registered s.t. p90|w_val*VAL| = 0.15 on train
    v     = clip(v_raw, -1, 1)

**Four bands** (thresholds pre-registered from the TRAIN v_raw distribution, stored in
`.cache/instrument/signed_answer_prereg.json` at first computation, never from eval outcomes):

| band | rule | fold |
|---|---|---|
| like | v > t_like (train p60 of positives) -> graded positive | C-lite: v; C-full: levels 6..9 |
| meh | -t_neg <= v <= t_like | neutral (C-lite v=0; C-full level 5) |
| dislike | v < -t_neg (train p25) -> graded negative | C-lite: v (<0); C-full: existing graded dislike levels 4..0 by magnitude |
| refuse | EXPO = e_c < 1.5 (bpool_r2's own rule) | no fold; burns the turn |

**Per-user volume normalization on the NEGATIVE channel (the KT-A3 mitigation, mandatory per the
leak finding):** after banding, scale each user's negative values so their total magnitude is capped:
`v_neg <- v_neg * min(1, C_NEG / sum|v_neg|)`, C_NEG = 2.0 (pre-registered). Rationale: the number
of solid-support negatives grows with user volume; capping the folded negative mass makes the
negative channel volume-invariant to first order. Acceptance re-measures the leak (below).

## 4. The retrains

- **Signed C-lite** (`concept_fold.py --train --signed --tag cfold_signed`): frozen ep4 tower,
  curriculum m ~ U{1..16} (the q16 slide was out-of-curriculum at m<=8 — extend the envelope),
  negatives/neutrals IN-envelope via the four-band signed labels, member-drop item-mask unchanged,
  ~2 h.
- **Signed C-full** (`train_tower_t2.py --train --concept_tokens --signed_concepts --select_cold
  --tag cfull_signed`): same signed channel mapped to tower levels (dislike levels are already
  first-class in the FiLM tables), concept-example curriculum m up to 16, ~3-5 h.
- Sequential, session-independent launches, stderr split, commit before each (HR10). CPU judgment:
  the S4 baseline snap (DAE/MultVAE filler) is demoted to IDLE priority for the duration — the
  author-GO retrains take precedence over a decision-independent filler; snap timings are not
  results (only its final metrics matter), so contention harms nothing that is measured.

## 5. Acceptance gates (pre-registered; run by the overnight queue after both trainings)

1. **Deployment concepts-only** (fixed bank, q=2..16): signed-retrained curve >= the signed-EVAL
   curve (>= 0.1317 at q2 held through q16; **never below the 0.1279 intercept at any q**).
2. **Counterfeit-eval on the retrained model** reproduces ~0% of the gain (the hard-kill rule
   carries over: >=70% = kill).
3. **Volume-leak R2 <= ~2x the clip-up baseline** (<= ~0.05) — the C_NEG normalization must work.
4. **Redundancy row holds** (correlated top-SEL m8 >= m4, full+tail).
5. **Per-answer m<=8 not degraded** vs current C-lite (div m2 0.1922 / m8 0.2091 - eps).
6. **Member-AUC (m=1/2/4) + pop-projection control reported** (the split-G5 columns; no bar change).
7. Ledger rows signed-clite / signed-cfull added alongside the old rungs; NO certification retrain
   until the author picks the final winner.

## 6. Decision provenance
Concept escalation arc: ArmA additive (saturates) -> FixA div-selection (correlation artifact fixed,
eval-only) -> C-lite trained fold (monotone m8, item-parity at m2, deployment decline found) ->
signed triple gate (clip poison confirmed; signed taste-real; OOD-limited) -> **this design**.
Fallbacks if acceptance fails: (i) leak gate fails -> raise/lower C_NEG grid on val (pre-registered
grid {1.0, 2.0, 4.0}); (ii) deployment gate fails on signed-C-lite but passes on signed-C-full (or
vice versa) -> the ledger decides; (iii) both fail -> concepts reclassified as short-interview
channel (m<=8), documented honestly.
