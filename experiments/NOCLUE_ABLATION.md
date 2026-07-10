# NO-CLUE ABLATION -- FOLD-V3.1 anti-surprise

Author-directed night task. Tests whether no-clue / refusal tokens can carry genuine NEGATIVE implicit signal before finalizing the decision to skip folding them. NO LLM calls; population trU users only (the 173 study/eval users untouched); deterministic; reduced threads.

## Hypothesis (author, verbatim)

> no-clue is informative exactly when knowledge was EXPECTED -- refusing a famous item your volume predicts you'd know = real negative taste/territory evidence; refusing obscurity = nothing. The current sampler gives no-clue tokens the same SURPRISE feature as consumption (low for real no-clues) -- the fold cannot express "expected but absent".

## ANTI-SURPRISE feature (formula)

New scalar input lit ONLY on no-clue (LVL_NEG implicit) tokens = expectedness of knowing:

```
E[n_E] = V * p_E            # volume-predicted engagement (fame x volume)
           p_E = popmass(E) / total_popmass ;  V = revealed volume (len known)
n_E    = # revealed members of E   (0 for a genuine refusal)
ANTI   = clip( log( (E[n_E] + 0.5) / (n_E + 0.5) ), 0, 8 )
```

HIGH when a famous/expected entity is refused (large E[n_E], n_E=0); ~0 for obscure refusals. Equals -surprise, but as a DEDICATED feature lit only on no-clue tokens it decouples "expected-but-absent" from the consumption-surprise weight (v3 found the LVL_NEG level flag inert when it shared the surprise feature).

## Curriculum (informative-refusal exposure)

Natural v2.1-answerer interview (T=24, oversampling OFF): 4.643 no-clue tokens/interview, 0.375 EXPECTED (anti>=1.0)/interview; anti mean 0.275, p90 0.889. Starved (< 0.3/int): False. Oversampling famous-off-profile probes: False (p=0.0). Oversampling lets the v2.1 knowledge model DECIDE each refusal (realistic); P2 evaluation uses NATURAL rates (oversampling OFF).

Winner checkpoint `.cache/i25_fold_v31_best.pt`: clean_frac 0.3, val NDCG@10 0.4356 @ep7.

## P1 -- author probe (no-clue pull vs no token; region-score)

| case | anti (mean) | pull vs no-token | 95% CI | n | verdict |
|---|--:|--:|---|--:|:--:|
| EXPECTED (famous refused) | 0.88 | -0.0586 | [-0.0738,-0.0426] | 398 | NEG (CI<0) |
| OBSCURE (obscure refused) | 0.35 | -0.1053 | [-0.1250,-0.0849] | 400 | FAIL |

Reading (honest): the NEGATIVE-pull half of the hypothesis is CONFIRMED -- refusing an expected/famous
entity now pulls the belief AWAY from that region (-0.0586, CI excl 0; v3's no-clue token pulled
POSITIVE +0.24 in the same probe family). The GRADIENT half is NOT confirmed as pre-registered: obscure
refusals also pull away, in fact MORE (-0.1053). Diagnostic: the model learned "no-clue = repel the
region" generically rather than scaling by anti-surprise; obscure-entity regions sit further from the
popular prior so the same repulsion moves their (lower) region scores more. The anti feature separation
in-distribution is also modest (anti mean 0.88 famous vs 0.35 obscure, both below the anti>=1.0
"expected" bar for most probes), so the feature range the probe exercises is compressed. Net: v3.1
fixes the SIGN pathology the author flagged, but does not yet express "refusing obscurity = nothing".

## P2 -- interview-level justification (T=24, natural k=0). THE DECISION NUMBER

n=800 DEV users; 3599 no-clue tokens (792 users had >=1).

| condition | NDCG@10 | delta vs (a) | 95% CI |
|---|--:|--:|---|
| (a) v3 SKIP no-clue (arena rule) | 0.3744 | -- | -- |
| (b) v3 FOLD no-clue (as-trained) | 0.3768 | +0.0024 | [-0.0014,+0.0065] |
| (c') v3.1 SKIP no-clue | 0.3823 | -- | -- |
| **(c) v3.1 FOLD no-clue + anti** | **0.3842** | **+0.0098** | **[+0.0024,+0.0173]** |

Within-v3.1 fold-vs-skip (c - c'): +0.0020 CI [-0.0022,+0.0059]. (c) vs (b): +0.0074 CI [+0.0002,+0.0147].

**DECISION -- does (c) beat (a)? True** (delta +0.0098, CI [+0.0024,+0.0173]).

Decomposition caveat (honest): the pre-registered decision number (c)-(a) is significant, but it bundles
two effects: (i) the v3.1 RETRAIN itself ((c')-(a) = +0.0079) and (ii) folding no-clue-with-anti within
v3.1 ((c)-(c') = +0.0020, CI [-0.0022,+0.0059], NOT individually significant). So most of the win is the
retrained encoder; the no-clue fold adds a small positive-but-not-significant increment ON TOP and --
unlike v3's as-trained fold (b) -- never hurts. The pre-registered rule keys on (c) vs (a), which is the
actual deployable choice (checkpoint + rule travel together), so the decision stands as registered.

## P3 -- non-regression of the decisive v3 gates

| gate | v3.1 | 95% CI | v3 ref | verdict |
|---|--:|---|--:|:--:|
| G2 GoT | +0.1611 | [+0.1242,+0.1987] | +0.1843 | PASS |
| G2b prolific | +0.4332 | [+0.4029,+0.4669] | +0.3159 | PASS |
| G7 implicit-abl | +0.0323 | [+0.0201,+0.0448] | +0.0203 | PASS |
| G4 clean gap | -0.0274 | (fold 0.5075 vs native 0.4802) | -0.0175 | PASS |

P3 all-pass: True.

## VERDICT + RECOMMENDATION

- P1 (author probe): HALF-confirmed -- expected refusals pull AWAY (-0.0586, CI [-0.0738,-0.0426] excl 0;
  the sign pathology v3 had is FIXED), but obscure refusals are NOT ~0 (-0.1053): the model repels
  refused regions generically instead of scaling by anti-surprise.
- P2 (decision): (c) beats (a) = True (delta +0.0098, CI [+0.0024,+0.0173]); most of it is the retrain
  ((c')-(a) +0.0079), the fold-vs-skip increment within v3.1 is +0.0020 (CI spans 0) and never hurts.
- P3 (non-regression): all-pass = True (G2 +0.1611, G2b +0.4332, G7 +0.0323, G4 gap -0.0274).

**RECOMMENDATION: FOLD no-clue WITH v3.1 anti-surprise (CHANGE the arena rule).** (P2 decision=True,
P3 all-pass=True, net actionable=True.) The pre-registered decision number favors switching the arena
to the v3.1 checkpoint WITH no-clue folding; the author's concern is answered with a number in both
directions: folding refusals no longer misleads (v3's positive-pull pathology is gone) and the
realistic-interview gain is real, though it comes mostly from the retrained encoder rather than the
refusal signal itself. If the arena stays on the v3 checkpoint for other reasons, the SKIP rule remains
correct for v3 (its fold-noclue delta (b)-(a) = +0.0024, CI spans 0, and its no-clue token pulls the
WRONG way in P1-style probes).
