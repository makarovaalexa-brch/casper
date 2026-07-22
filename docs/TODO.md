# TODO — durable backlog (Paper A restart era)

> STATE.md = what is true NOW; this file = what is QUEUED, priority-ordered. Prune on completion.
> Last cleaned: 2026-07-22 (evening).

## Chapter A/B separation (author-approved rule)
**A certifies answer-INGESTION; B studies question-SELECTION.** A's interviews use only dumb fixed
policies (random control, popularity-static; Σ-greedy solely as G8 instrumentation). Everything learned/
adaptive (trees, VOI, EDDI-IG, Bıyık-EVOI, coarse-to-fine, the demoted +47% re-establishment) = Chapter B,
fought ON the certified instrument. Elicitation-line SYSTEMS appear in A only as master-table rows (their
recommender cores); their POLICIES fight in B.

## Running / imminent
- [ ] **RecVAE canonical-ruler** (training, val peak ~0.3504 @ep20; verdict vs EASE 0.3476/0.2441)
      → on completion auto-launch chain: Mult-DAE ML-20M snap → belief_mf → golbandi_node → Mult-VAE snap.
- [ ] **T2' training night** (next): graded-native tower on canonical split (`src/instrument/
      train_tower_t2.py`, committed). Include the graded-vs-binarized ablation arm (tower-level G3c
      sanity). ~74 min/epoch from scratch — expect 2–4 nights, resume-capable.
- [ ] **Tower pick at G0'**: T2' vs bar (EASE/RecVAE). **AUTHOR RULE: T3' distillation ONLY if T2' gap is
      CI-significant; within-noise ⇒ T2' is the tower.**

## Step 2 — the instrument (chapter A core)
- [ ] **Belief-layer + battery code** vs `docs/design/DESIGN_SHEET_STEP2_BELIEF.md` (v2.2: design (ii);
      concepts = Arm A latent mean shift, three-arm G5 decision; refusal = consumption observation with
      fitted α_refuse — author correction Jul 22). Then its own adversarial review → commit → run.
- [ ] **Battery run — MUST-gate tier first** (the vision set): G0, G2, G3, G5, G6 + C2. Supporting tier:
      G1c calibration detail, G4, G8-selection, G9. Continuous-token demo (random abstract embedding folds
      like a concept direction) alongside G5.
- [ ] **C2 certification** (SEL transfer; leak clause: exclude held-out targets).

## Chapter A figures (author-approved — build with the battery results)
- [ ] **FIG-1 "intensity staircase"**: ΔNDCG from folding ONE answer at each level hated→loved, per
      channel (items, concepts). The G3 figure — no elicitation paper has it.
- [ ] **FIG-2 "gap map, measured axes"**: upgrade the qualitative TikZ quadrant → scatter with
      y = measured full-profile NDCG on our ruler, x = R-checks satisfied; instrument's dot enters the
      empty top-right after the battery. Before/after variant if it reads well.
- [ ] (optional) FIG-3 Σ-trace per question with refusal steps visibly small; battery scorecard table.

## Chapter A completion checklist (what "done" means)
- [ ] All master-table planned/queued cells measured or honestly dispositioned (wave-2).
- [ ] Battery results table (pass/narrowed/fail per gate) + G2 curves (full+tail, CIs).
- [ ] Bridge prose completed with Mult-VAE/DAE snap deltas.
- [ ] UNICORN row added (bib exists: deng2021unicorn).
- [ ] Method section rewritten from design sheet v2.2; results section from battery JSONs.
- [ ] CIs on every bolded delta (paired per-user bootstrap).
- [ ] C3 stays listed as owed certification (design: `STUDY_C3_HUMAN_ROUNDTRIP.md`) — NOT a chapter blocker.

## Wave-2 baselines (nights, interleaved — table-filling, not gating)
- [ ] belief_mf + golbandi_node (reviewed, auto-queued tonight).
- [ ] EDDI training night (after T2'; deprioritized behind instrument line).
- [ ] Turbo-CF / SASRec/BERT4Rec / canonical Mult-VAE/DAE — only if the master table still needs them.
- [ ] Concept-as-pseudo-item ablation — at G2/G5 stage (it's the Arm-B comparator's cousin for EASE).

## Chapter B seeds (do NOT start — recorded so nothing is lost)
- Re-establish demoted magnitudes on the certified instrument: adaptivity (+47% re-test), answerable-
  concepts, belief-pool curves, open-recall/framing. Policy rivals: Golbandi tree, EDDI-IG, Bıyık-EVOI,
  ConTS. G9/G8 substrate from A enables clean attribution.

## Standing debts (not blocking)
- [ ] Answerability fixes (5, zero-LLM) — before answerability is EVER cited again (B-era).
- [ ] Bib merge into references.bib (`BIB_MERGE_TODO.md`); wang2025bdecf key rename optional.
- [ ] `experiments/baselines/DESIGN_SHEET.md` refresh to canonical-ruler reality.
- [ ] EXPERIMENTS.md rows for canonical-ruler results (fold in with RecVAE verdict).
- [ ] Tail-metric user-floor parameter (≥k tail targets) — set when G2 harness is built.
