# Adversarial review — CASPER papers A–E

_Date: 2026-07-05. Reviewer stance: hostile top-tier venue (SIGIR/RecSys/UMUAI/NeurIPS) + PhD thesis committee. Formatting/compilation ignored._

Bottom line up front: individually each paper has defensible craft and unusual honesty. But they share **one methodological substrate that a hostile reviewer will attack once and use to discredit all five**: every result is produced by a simulator whose answer model lives in the *same embedding space* as the recommender it is evaluated against, and **there is not a single human answer anywhere in the thesis**. That makes the central "wins" self-consistency exercises inside one geometry rather than findings about preference elicitation. Below: the thesis-level flaw first, then the fatal per-paper issues, then what would actually survive.

---

## The one flaw that threatens the whole thesis (hits A, B, C, D)

**Circular measurement.** In every paper the simulated user answers a question by projecting true taste `u*` onto the query *in the recommender's own latent space* (`a = sign(u*·q)` or `a = cos(u*, q)`), the belief is the encoder's fold of those answers, the recommender scores `s = b + Q·u`, and the training loss is `1 − cos(belief, u*)`. Target, answer, policy, recommender, and even one of the "belief-quality" evaluation metrics are all functions of `u*` in one 64-d space. Any method that probes coordinates of `u*` wins **by construction**. This is visible in the equations (Paper C §Background L269, §Method L287; Paper B §2 answer model). A committee member only has to establish this once to put every headline number in doubt.

**No human validation, anywhere.** Every "answerability rate," "framing lever," "graded answer," and NDCG delta is simulator output. The load-bearing *human* premises — "users can answer concepts but not items" (B), "users can name a hidden gem on command" (D), "a person answers a rendered phrase as they'd answer the latent query" (E) — are asserted and, in E's and C's own words, explicitly untested. Strip the simulator and there is no empirical result in the thesis.

Decide *now*, at thesis level, how you answer this in the defense — it will be the first question. The strongest honest answer is to reframe the contribution as **measurement-theoretic / mechanistic under a stated answer model**, and to run even a *small* human round-trip study (~50 users, rendered questions, collect graded answers). That single study would defuse the objection across all five papers.

---

## Paper C ("The Best Question Has No Name") — the paper refutes its own title

Most self-damaging, because §5.6 (the "answer-fidelity boundary," L1045–1128) *already contains the refutation* and a reviewer will quote it back verbatim:

- **The headline is an artifact of a noiseless, co-designed answer channel.** A graded query is a noiseless linear readout of `u*`; a binary query is one sign bit. "Continuous beats discrete" is close to "8 real-valued measurements reconstruct a 64-d vector better than 8 bits" — a tautology, not a discovery. The actor reaches `cos(belief, u*) ≈ 0.92` because it is regressing a vector onto its own projections.
- **Your own per-seed decomposition kills it** (L1094): `native exact 0.496 → imperfect linear 0.389 → +quantization 0.293 → +sampling 0.172`. Every realistic degradation of the *answer* — none touching query geometry — costs ~0.10 NDCG. Under the empirically fitted ML-1M channel the continuous actor collapses **0.491 → 0.150, below MOSTPOP (0.310)**, while discrete answerable questions win (concept-8 0.325, item-8 0.394). The honest finding is the *opposite* of the title.
- **Snap-loss (your "most important number") reverses sign under noise** (L1100): snapping *helps* +0.051. A quantity that flips sign the moment you use a realistic answer model is a property of the oracle, not of question geometry.
- **The one deployable, askable rung (pairs) is not significant**: tail +0.008, 95% CI [−0.007, +0.024], p=0.30. The RecVAE "sharpening" doesn't rescue this — a *more linear* instrument makes the circularity *more* exact, which is why the gap widens.
- **Suppressed evidence:** you built a learned (non-oracle) answerer and removed it (bot-play, L1245). A reviewer will ask why the one non-circular answer model you had isn't reported.

**Verdict:** reject-as-framed / major revision. Becomes legitimate if reframed around §5.6 ("continuous queries win *only* under slider-grade answers real users may not give") — but that is Paper B's thesis, not a new one.

---

## Paper B (CASPER-R / "answerability") — the axis is a definition, the win is sub-noise

- **The answerability axis is hard-coded, not discovered.** "An item is answerable only if the user rated it; a concept iff ≥2 items" (§2). The celebrated 5:1 answer rate is then just **the density of two slices of the ML-1M rating matrix**. "Rated" ≠ "could answer": a cold user can obviously answer "did you like *Titanic*?" Your own set-piece (Haiku picking famous titles and "falling into the trap") is the tell — under any realistic popularity-conditioned answerability model the item-vs-concept result narrows or inverts.
- **The learned win is +0.012 tail, a *tie* (−0.001) on full, with no std/CI/significance test, and on the wrong seed axis** — the five seeds vary the *user split*, not training init; the policy is trained once. You cannot claim a robust learned policy while never retraining (memory already flags CASPER-R as training-seed-fragile ≈ entropy).
- **§6 refutes §5:** your own saturation analysis says marginal value is ≤0.001 tail from the 6th answer and "we claim no learned endpoint win." That contradicts the §5 q8 headline. Parsimonious resolution: +0.012 is noise.
- **It doesn't replicate:** tail advantage *reverses* on ML-25M (−0.006) and is *refuted* on Goodreads (−0.084). Only the matrix-density ratio replicates.

---

## Paper D (open free-recall) — the comparison is rigged

- **Fatal asymmetry:** open-recall folds the user's *actual known-half item tokens*; every baseline is explicitly *forbidden* the profile (shared candidate pool, refuse-on-unanswerable). Your own numbers show giving PEBOL the same access jumps it +0.052/+0.076 — most of the "gap." A partial oracle vs. blinded competitors, relabeled as "bandwidth."
- **It reconstructs the ceiling:** naming 8 known-half items and folding them just re-derives `enc(known half)`, which *is* the 0.41/0.21 oracle. The info-favourable answerer hits 0.410/0.209 — the ceiling. A confession, not a win.
- **The flagship "2 questions beat 8" uses a privileged answerer** (`distinct` = argmax over `u*·Q`, i.e. collusion-by-proxy). Under realistic `popweight` recall, K=2 = 0.336/0.133, far below Paper C. The only clean result (`random5`, +0.007 tail, n=3 seeds, no CI) is marginal, and the realistic proxy *loses the tail*.
- **Recall realism is asserted and self-confirmed** on an LLM whose popularity bias *is* the assumption. The memory/decision literature you cite argues the opposite (availability heuristic). No humans tested; the name→factor resolution step (where the method dies above ~10% error) is deferred to future work.

---

## Paper E (rendering to dialogue) — proves the loop cannot be closed

- **It refutes its own title.** §4 shows the continuous query is "interpretable to read but not faithfully askable"; the askable rendering *is* the k=1 snap Paper C already rejected. Rendering = snapping. Contributions 1 and 2 are the same finding stated positively then negatively.
- **The "near-lossless" −0.008 is a slider UI, not a dialogue.** Folding phrases as separate answer tokens (the thing resembling asking questions) costs −0.073. cos 0.84 buys nothing *askable*. Also the abstract still headlines the superseded −0.008 when your own canonical five-seed number is −0.010/−0.012.
- **The marquee LLM "synthesise the axis" step has zero evaluation** — cherry-picked anecdote (n=13 for the inversion-fails probe), and axis-synthesis *discards* the orthogonal residual that §C says is the whole advantage.
- **The agent is a `\todo` placeholder + a +0.003 within-noise table.** As a standalone "deployable dialogue" paper: reject. As one honest negative-result section inside Paper C (contingent on the human study): worth keeping.

---

## Paper A (the instrument) — measures the instrument

- **The capstone "question channel is strongest" (§12.4) is tautological:** probing along the decoder's own principal directions, answered geometrically, feeds the recommender a clean readout of its own latent state. No human emits "my coordinate on decoder axis 4 is +0.7."
- **"Ties EASE (0.554)" is a full-profile warm-start number** — not the cold-start elicitation task, and RecVAE is decorative: no elicitation result actually runs on it except the circular §12.4.
- **Several headline deltas restate the training objective** (polarity loss → polarity gap grows; trained reconstruction encoder beats untrained fold-in on the encoder's own objective; G4 passes by algebraic identity with WRMF).
- **The realizable EIG win — the one result that would rescue "large oracle bound"** — is deferred to the companion paper; this chapter's own realizable evidence is null (heuristics ≈ random, LLM < random).

---

## Cross-cutting minors (apply to most)

- **No multiplicity control** across an enormous battery — small, selectively-significant deltas narrated into a monotone story (garden of forking paths). Where per-user bootstraps *are* run (C's pair rung, E's agent), significance dies; where headline claims live, they're conspicuously absent.
- **Metric depth is a free parameter** (NDCG@10 / @100 / @510 chosen per dataset).
- **Post-hoc theory-fitting:** the linear-Gaussian argument is invoked to explain adaptivity winning *and* tying, whichever happens.
- **Single domain carries every elicitation claim** (ML-1M); cross-domain runs either reverse the result or never carry an elicitation claim.

---

## What I'd actually do (priority order)

Not a fraud problem — a **framing and validation** problem, and the papers are honest enough that the fixes are visible in your own text.

1. **Run one small human study** (rendered questions → collected graded answers → downstream NDCG). Highest leverage; defuses the circularity objection across A–E at once.
2. **Reframe C around §5.6** as the honest result: *answer fidelity, not action-space continuity, governs elicitation; continuity wins only under slider-grade answers.* Demote the noiseless RecVAE "sharpening."
3. **Fix D's comparison** — either let baselines fold profile tokens or blind open-recall; report only non-privileged answerers with CIs.
4. **Report training-seed variance + paired significance** on every headline (B's +0.012, D's +0.007, E's +0.003 will likely not survive — say so).
5. **Replace "answerable iff rated"** (B) with a popularity-conditioned seen-probability model and re-run; the item-vs-concept headline probably moves.
