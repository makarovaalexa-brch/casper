# Paper A design verdict (c5) — 2026-07-20, Fable

> Decision made from `external_literature/findings/paperA_recommender_landscape.md` (c1–c4 deep research)
> + the two prior findings files. Requirements R1–R5 and gates G0–G2 as in `docs/VISION.md` pillar 1.

## Verdict
**Go with the architecture we already have in design: a FROZEN shallow SOTA tower + an analytic
conjugate-Gaussian belief layer over its latent (= PrecAcc, `docs/design/DESIGN_PRECACC.md`).**
The lit review *independently converges* on this shape:

1. **R1 substrate = shallow frozen tower.** The full-profile frontier has been flat since ~2019 and is held by
   shallow linear/AE models (EASE 0.420 / RecVAE 0.442 / EDLAE, NDCG@100 ML-20M). Nothing 2023–26 beats them on
   that split. A RecVAE-class / set-encoder tower is a defensible SOTA substrate; no deep-net detour is warranted.
2. **R4+R5 mechanism = exact conjugacy over the frozen latent.** Conjugate-Gaussian (Kalman/precision) updates
   are the ONLY mechanism in the literature with a principled monotonicity story (Good 1967 / Blackwell) — and it
   holds only in expectation, so our empirical per-question G1/G2 curves are themselves a contribution. Learned
   uncertainty heads (option B below) have no such story.
3. **R3 = the genuine gap.** Every R1-bar model is structurally item-indicator — concepts, signed values, and
   continuous directions are *inexpressible* there, not merely untuned. Channel-agnostic tokens folding as linear
   observations in the frozen latent is the one leg found nowhere.

## The gap is real (c3)
No published system holds R1∧R2∧R3∧R4∧R5 ("we are aware of none" — never "there is none").
**Central threat = Biyik 2023** (Gaussian belief + attribute directions + EVOI): our wedge is (a) FROZEN
*certified* SOTA tower vs their co-trained encoder, (b) exact conjugacy vs their approximate non-Gaussian
posterior, (c) arbitrary-embedding tokens incl. out-of-catalog concepts + continuous directions vs their fixed
attribute set. **Nearest mechanism precedent = BCIE** (conjugate fold over a frozen factorization) — cite hard,
differentiate on SOTA-certification + channel breadth + the G0–G2 protocol.

## Options considered (rejected)
- **B. Amortized/learned uncertainty** (EDDI-style partial-VAE head, NP/TaNP): right R2 shape, but no
  monotonicity story, uncertainty quality unverifiable, and retraining risks G0. Keep EDDI/TaNP as baselines/cites.
- **C. Extend the linear fold-in basis** (EASE + concept columns): stays in the item-indicator world; R3
  fails architecturally; no R4. Keep EASE as the G0 bar and fold-in baseline.
- **D. LLM-as-recommender:** documented weak on collaborative signal (he2023large); LLM stays render/interpret
  only per VISION.

## Obligations the lit imposes (into the Paper A plan)
1. **Metric bridge before any "ties SOTA" claim:** our 0.4946/0.4998 are NDCG@10 ML-25M; the published bar is
   NDCG@100 ML-20M strong-generalization. Run our tower once on the Liang split (or EASE/RecVAE on ours) and snap
   to the published numbers exactly (HARD RULE 2).
2. **Baseline bank = landscape c4 tiers:** T1 floors (Pop, item-kNN, PureSVD/iALS, SLIM) · T2 R1-bar (EASE,
   RecVAE, Mult-VAE/DAE, EDLAE; SASRec/BERT4Rec as sequence fold-in; graph filters only if re-run on our split) ·
   T3 rivals (EDDI, Biyik, ConTS, BCIE, PEBOL, Golbandi tree, RBMF/Functional-MF) + random-question control.
3. **Claim discipline:** lead with the CONJUNCTION + the three inexpressible legs (signed values,
   out-of-catalog concepts, continuous directions) + G0–G2 as an acceptance protocol. Do NOT lead with
   "uncertainty-native rec" (crowded 2024–25) or anything on the pre-empted list.
4. **Open checks** (cheap, before submission): deep-set/Set-Transformer full-profile SOTA search re-check;
   BERT4Rec replicability caveat (petrov2022replicability); reconcile `austin2024pebol` vs `austin2024bayesian`
   bib key; add the 15 listed bib entries.

## Immediate consequence
The pending PrecAcc run (`scripts/train_precacc.py`, STATE.md "next action") is now also the *lit-validated*
Paper A instrument experiment: passing G0/G1/G2 on full+tail is exactly the demonstration the gap analysis says
no one has published.
