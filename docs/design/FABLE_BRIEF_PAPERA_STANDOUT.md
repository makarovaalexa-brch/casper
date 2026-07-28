# Brief for Fable: what contribution would make Paper A stand out standalone?

## The ask (one question, typed verdict please)

Paper A is currently a solid but unexciting instrument/methodology paper. An adversarial review put it at
"weak accept at ECIR, borderline RecSys, below SIGIR" **even if every queued run lands**, because the
measured delta over the best baseline is *a tie on accuracy plus capabilities*, and capabilities are hard to
sell without a number showing they pay.

**Question: what ONE additional contribution — an experiment, a reframe, or an artifact — would make Paper A
stand out on its own?** Not tidying, not fixes, not "run the queued things properly". Something that changes
what the paper *is*.

Constraint: **it must not enter Paper B territory.** Paper B owns adaptive/learned question selection
policies, the oracle-vs-realizable adaptivity gap, and anything where the contribution is "a better policy".
Paper A owns the recommender that can *be* interviewed.

Context: the thesis always intended to bundle A with B. This is exploratory — the author wants to know
whether A can stand alone, not a commitment to split.

---

## What is CERTIFIED now (measured, on the canonical ruler)

Ruler: Liang strong-generalization split on ML-25M, train 140,768 / val 10k / test 10k users, seed 98765,
18,359-item train vocab. Report full AND tail NDCG@10. Cold intercept (no answers) = **0.1279 / 0.0192**.

- **The tower ties the accuracy bar.** Frozen graded-native set-encoder T2′: TEST **0.3482 / 0.2462**
  (@100 0.4486) vs the RecVAE bar 0.3540/0.2497 — diff −0.0057, paired-bootstrap CI 0.0069, tie.
  Empty interview reproduces the frozen decoder EXACTLY (|z|=0, Spearman 1.0000). So G0 = strength + identity.
- **Bridge (C1):** our EASE 0.4203 vs published 0.420; RecVAE 0.4425 vs 0.442 (Recall@20/50 also match).
  Mult-VAE 0.4223 vs 0.426 (near-snap). Mult-DAE failed at 0.397 vs 0.419 and is dropped.
- **Baselines on our ruler (full/tail@10):** RecVAE .3540/.2497 · EASE .3476/.2441 · EDLAE .3433/.2395 ·
  Golbandi-node .3065/.1895 · iALS .2442/.1853 (uncertified floor) · item-kNN .2428/.1297 ·
  belief-MF (Bıyık/ConTS core) .2019/.1200 · Most-Popular .1345/.0226.
- **Battery, measured:** G1a posterior trace shrinks 1881.9→1805.1 monotone; G1b directional **11.5×**
  (bar 2×); G1c calibration ρ=0.2393 (diagnostic); G2 cold curve rises 0.1279→0.2724 full, 0.0192→0.1305
  tail; G3a sign-flip −0.2985 CI-clean, intensity staircase ρ=0.927; G3b value-neutralisation costs 0.2869;
  G4 refusals inert (architectural); G5 concept membership AUC 0.94 whitened vs 0.44 raw; G6 shuffled
  +0.0371 / placebo +0.0168 / wrong-user −0.0986 (a stranger is worse than silence); G8 Σ beats isotropic
  cI by +0.0089 @q16 CI-clean, order-invariance 5e-7; G9 fixed-bank rise +0.1218, monotone ρ=1.0.
- **Graded ("star") input beats binarized on SHORT interviews**, vs the fair positive-only baseline:
  **+0.0117 @k=2** (CI [+.0099,+.0135]), +0.0071 @k=4, tie by k=8. Mechanism: a signed answer encodes a
  dislike the positive-only item-indicator basis cannot express.
- **Answerability, measured and stark:** in a realizable static item interview, a user answers on average
  only **3.6 of 16** questions. In a per-user-filtered bank (oracle answerability) all 16 land and the same
  tower reaches 0.2724 instead of 0.2098. Concepts are answered far more often than items.

## ASSETS WE HAVE BUT HAVE NOT YET TURNED INTO A CONTRIBUTION

These are real, working, and currently either unreported or reported as minor:

1. **An SBERT→CF-latent adapter that makes the concept channel open-vocabulary.** One globally-fitted linear
   ridge adapter W = QᵀS(SᵀS+βI)⁻¹ from frozen all-MiniLM-L6-v2 (384-d) into the frozen CF latent. Any free-text
   phrase becomes a direction and folds like an item — **no per-concept training, no concept inventory, no
   retraining, inference-time only**. Controls that exist: paraphrase invariance ("dinosaurs" ↔ "prehistoric
   reptiles" 80% overlap@10, mean 70% over 10 pairs — so not string matching); and at scale, folding a tag's
   TEXT recovers the same recommendations as folding that tag's genome-grounded item CENTROID (50% mean
   overlap@10 vs 0.2% random, ~250×). Ridge beat InfoNCE contrastive alignment badly off-manifold
   (median rank 38 vs 206). Needs porting to the current latent; the science is done.
   Lit check verdict: mechanism is pre-empted (Gantner 2010, CB2CF 2019, Göpfert 2022 CAVs); what appears
   unoccupied is *open-vocabulary + zero per-concept supervision + used as a PREFERENCE input to a belief
   that also holds item tokens*. One BLOCKING verification outstanding (whether Göpfert/Bıyık CAVs are
   really per-concept probe-trained — currently inferred, not read from primary text).
2. **The acceptance battery itself** (G0–G9 + C1–C3) as a portable, reusable certificate for
   interview-compatible recommenders. Currently framed as internal apparatus, not as a released artifact.
3. **The "deployment currency" methodological point:** a capability gate (can the operator express dislike?)
   is vacuous unless paired with a gate on what the deployed system *actually does* down a realistic,
   user-independent question bank under a per-question budget. We have a concrete case where a
   sign-capable operator passed its capability gate while never emitting a negative answer in deployment.
4. **The answerability gap** (3.6/16 vs 16/16 above) — currently a caveat, arguably a finding.

## QUEUED WORK AND THE CLAIM EACH IS EXPECTED TO SUPPORT

| # | Work | Expected claim |
|---|---|---|
| 33 | Greedy best-static sequences (RUNNING): items-only / concepts-only / **combined** banks, built on 10k val, evaluated on disjoint 10k test, 500+500 pools | The strategy-free concept-vs-item comparison. Combined ⊇ items-only, so *combined − items* is a direct readout of what concepts add. Items-only arm already landed: 0.1405/0.0275 @q1 → 0.2098/0.0679 @q16 |
| 37 | Port the SBERT adapter to the current latent | R3 stops being "an interface" and becomes a demonstrated open-vocabulary capability |
| 43 | Open-vocabulary concept interview (PEBOL-style): questions are LLM-*rendered* free-text concepts, answers stay behavioural (never LLM) | Shows the open channel carries a *real interview*, not just a retrieval demo. Needs an "imputer" — an arbitrary phrase has no member set, so no behavioural answer exists yet |
| 40 | Package C2 (held-out behavioural answer model + noise + geometry firewall) | Non-circularity. Firewall + behavioural answerer already exist; this is packaging |
| 38 | G1c calibration: trivial-proxy control, then refit | Either the covariance beats "just count the answers" or it does not |
| 41 | Concept-as-pseudo-item ablation into EASE/RecVAE | Answers the cheap counter: "just add genre columns / signed entries" |
| 44 | Reframe: lead with set-folding + channel-agnostic input; demote the posterior to a supporting property | Removes the "your covariance buys +0.0036, it's decorative" attack |

## KNOWN WEAKNESSES (do not propose fixing these — they are already queued)

- The posterior is maintained **decoupled from the ranking path**; its only demonstrated use (question
  selection) beats random by just +0.0036 @q16. R4 is being demoted accordingly.
- Concept fold currently indexes a trained table over 1,031 genome tags (hence asset #1's importance).
- The concept channel leaks user watch-volume (R² 0.249 vs a 0.05 bar) — open acceptance condition.
- Once a question bank is restricted to answerable questions, *which* question you ask matters far less than
  *that* it is answerable — random ordering is a strong baseline. (Selection is Paper B's problem.)

## WHAT I WANT BACK

1. **The one contribution** you would add, stated as a claim sentence plus the experiment that earns it.
2. Why it is not Paper B territory.
3. What it would cost (rough — days of compute/work), and what could go wrong.
4. Your second choice, briefly.
5. Blunt: is standalone A worth it at all, or is bundling with B strictly better?
