# TODO — durable backlog (Paper A restart era)

> STATE.md = what is true NOW; this file = what is QUEUED, priority-ordered. Prune on completion.
> Last cleaned: 2026-07-22 (evening).

## Chapter A/B separation (author-approved rule)
**A certifies answer-INGESTION; B studies question-SELECTION.** A's interviews use only dumb fixed
policies (random control, popularity-static; Σ-greedy solely as G8 instrumentation). Everything learned/
adaptive (trees, VOI, EDDI-IG, Bıyık-EVOI, coarse-to-fine, the demoted +47% re-establishment) = Chapter B,
fought ON the certified instrument. Elicitation-line SYSTEMS appear in A only as master-table rows (their
recommender cores); their POLICIES fight in B.

## ★ AUTHOR ROADMAP (Jul 24) — in order
0. **Deep research: DONE** (findings/graded_inputs_for_ranking.md; premium claim pre-emption-mapped).
   **PARITY CORRECTION (Jul 24): the 0.419 was a MASKING ARTIFACT** — parity-correct all-bands graded =
   0.3478 vs likes-only 0.3435 (+0.004, marginal). Binary-EASE POISONED by all-rated fold (0.3446→0.2892)
   = the no-poisoning asymmetry claim. Phase-A gate verdicts unaffected (within-convention pairs).
   **In-flight, FINISH but do not extend now (author): remaining fair-fight arms + the premium-vs-k
   curve** (the likely headline: opinion value grows as evidence shrinks). Then PARK baselines until
   step 4-5.
1. **Fix G5 standalone-cold concept** (whitened dir loses the prior at k0 by 0.0045; per-k β / floor
   blend) — important, part of Step-2 design.
2. **Belief layer (Phase B)**: Σ + G1/G4/G8, formal G2, C2.
3. **Clean retrain of the tower** (one command, selection rule fixed: cold-primary s.t. full ≥ native−CI)
   + full battery RE-VERIFIED on the certified checkpoint.
4. **All outstanding baselines run** (Mult-VAE/DAE snaps + canonical, EDDI, SASRec, TaNP, Turbo-CF, RBMF).
5. **The graded/upgraded baselines** (two tiers: published-faithful biasedMF etc.; our labeled extensions
   signed-EASE etc. + the all-rated-binary control) — the fair-fight table for the graded-premium claim.

## Running / imminent
- [x] **GATING PROBE: FAILED (Jul 22 16:30) — frozen geometry DEAD.** No η gives ≥0.10 separation at
      ≤0.005 NDCG cost (best: +0.216 sep at −0.221 full). RUNG1 ceiling reproduced on canonical teacher.
      Teacher cold-curve: below pop floor at k≤2. `experiments/baselines/probe_separability.json`.
- [ ] **ESCALATION (pre-registered): T2' = warm-init TRAINABLE-decoder** — decoder+embeddings initialized
      from RecVAE ckpt but trainable; KD off; all other v3 repairs carried (γ-sign, dislike negatives,
      evidence gate, G0 split vs RecVAE full-profile). Builder implementing --warm_init mode → my review →
      commit → the training night (~74 min/epoch class). NOTE: geometry now emerges in training → concept
      directions/Σ defined on the FINAL T2' latent, not RecVAE's (design-sheet consequence, minor).
- [ ] **RecVAE canonical-ruler** (training, val peak ~0.3504 @ep20; verdict vs EASE 0.3476/0.2441)
      → on completion auto-launch chain: Mult-DAE ML-20M snap → belief_mf → golbandi_node → Mult-VAE snap.
- [ ] **T2' training night** (next): graded-native tower on canonical split (`src/instrument/
      train_tower_t2.py`, committed). **AUTHOR REVISION v2 (Jul 22): distill T1 RecVAE with FROZEN geometry** — decoder+bias and
      input item embeddings frozen from tonight's RecVAE ckpt; ONLY attention+FiLM train; loss = latent
      regression to the RecVAE encoder on binarized SUBSETS (subset-fed teacher supervises the interview
      regime k=1..8) + graded NLL; set-encoder output dim = 200; fallback --unfreeze_emb if val stalls;
      expect much faster epochs (frozen big matrices). SUPERSEDES v1 below:
      ~~KD ON from night one~~ (α_kd=0.5,
      teacher = EDLAE exported B; RecVAE outputs = alternate teacher) — distilling from the start is free;
      the old no-distillation rule now means: no SECOND tuning night if within noise of the bar.
      ⚠ KD RISK = G3 (all teachers are grade-blind): canary = the graded-vs-binarized ablation arm — if
      graded ≠> binarized on val, α_kd comes down. Prereq: `edlae.py --export_B` on the canonical split.
      ~74 min/epoch — expect 2–4 nights, resume-capable.
- [ ] **Tower pick at G0'**: T2' vs bar (RecVAE 0.3540); within-noise of the bar = good enough, move on.
- [ ] **T2' v4 = i25-REDUX (Jul 23, supersedes the phase plan below):** archive lookup found the
      pre-cleanup near-native recipe (i25_fold, native−0.014): **z = native_z(frozen RecVAE encoder on
      revealed likes) + zero-init sum-pool residual over graded tokens** — our from-scratch attention
      encoder's 0.29 plateau = the imitation gap it avoids. Builder implementing --arch i25 (frozen-
      decoder primary arm = G0-identity + full≥native at init; trainable-slow fallback if G3 fails).
      Phase-1 verdict recorded: curriculum refuted (pure-full diet stayed flat); LR vetoed by author.
- [ ] **T2' PHASE PLAN (author-designed, Jul 23 — historical; superseded by v4 above):**
      Phase 1 (RUNNING): resume-from-best (0.2913), p_interview=0.0 (pure full diet), all else frozen —
      3 epochs decide curriculum-vs-capacity (cold-val watches forgetting live). Phase 2 (only if flat):
      warm_lr_scale 1.0, single-variable. Phase 3 (always, = final recipe): reverse curriculum — anneal
      interviews back 0→0.5 as finetune (restores vision-guard ending share; cold must return ≥ today's).
      **Fork ckpt preserved:** `.cache/instrument/t2warm_v3_FORK_ep6_full02913.pt` (pre-Phase-1 best,
      full 0.2913 / tail 0.2029 / coldk2 0.1951). **AUTHOR REAFFIRMED (Jul 23): certified tower = ONE
      CLEAN RUN of the winning recipe end-to-end (curriculum as a SCHEDULE, no mid-run switches) — the
      phase experiments only DISCOVER the recipe.**
- [ ] **AUTHOR CONSULT REQUIRED (Jul 23): no recipe/LR switch at ep10 (or any point) without the author.**
      Present curve + projection + options; default = keep training. Author prefers avoiding a full
      retrain — mid-run switches need strong justification; the one-clean-run question is also re-decided
      WITH the author once the recipe settles (current run's provenance: committed code, one resume for
      logging — to be weighed then).
- [ ] **VISION GUARD for ANY T2' recipe change (author, Jul 22):** gates/requirements outrank recipes.
      Every launch config must keep: graded FiLM tokens (pb2's own proven recipe was graded); the dislike
      target-side gradient (down-weight allowed, delete never); interview-regime training ending ≥50%
      (anneal OK) with a small-k val curve logged; the evidence-gate intercept; the G3
      graded-beats-binarized canary. Teacher tricks bounded (grade-blind supervision never wins the
      endgame). Tower-internal plumbing (embeddings frozen/trainable, LRs, warm-starts) = free variables
      where the pb2 recipe wins. No mid-run changes unless the watchdog shows stall/decline.
- [ ] **REPRODUCIBILITY POLICY (author, Jul 22): the certified tower = ONE CLEAN RUN.** The current
      patched/resumed run is recipe-finding only. Once hyperparameters settle, retrain from scratch:
      single command, committed code, fixed+recorded seed, no mid-flight interventions — that checkpoint
      (and only that one) gets certified and frozen for Step 2. Exploration runs never become artifacts.
      (Bit-exact CPU replication impossible; the bar = statistical replication: same command+commit →
      same curve within noise.)

## Step 2 — the instrument (chapter A core)
- [ ] **Belief-layer + battery code** vs `docs/design/DESIGN_SHEET_STEP2_BELIEF.md` (v2.2: design (ii);
      concepts = Arm A latent mean shift, three-arm G5 decision; refusal = consumption observation with
      fitted α_refuse — author correction Jul 22). Then its own adversarial review → commit → run.
- [ ] **G2 static question-selection controls** (author-recalled Jul 23): random-k (floor) →
      popularity-order → **HELF (harmonic mean of rating-entropy and log-frequency, Rashid et al. —
      the canonical static interview selector; both stats from the train matrix, ~free)** → Σ-greedy
      (G8 arm). Learned/adaptive policies = Chapter B, fought against these.
- [ ] **Battery run — MUST-gate tier first** (the vision set): G0, G2, G3, G5, G6 + C2. Supporting tier:
      G1c calibration detail, G4, G8-selection, G9. Continuous-token demo (random abstract embedding folds
      like a concept direction) alongside G5.
- [ ] **C2 certification** (SEL transfer; leak clause: exclude held-out targets).

## Chapter A figures (author-approved — build with the battery results)
- [ ] **FIG: GRADED PREMIUM vs EVIDENCE SIZE (author-endorsed Jul 24, "great analysis")** — the curve
      k∈{2,4,8,16,full}: values-premium (graded − membership-only) and reveal-premium (membership −
      likes-only), paired CIs, parity-masked; plus the per-k flip-delta (sign-value curve). Expected
      story: opinion value GROWS as evidence shrinks — the interview-native finding. Run commissioned
      (`src/instrument/premium_vs_k.py` → experiments/battery/premium_vs_k.json, auto after fair-fight
      arms); becomes a chapter section + figure alongside the fair-fight asymmetry table (classic graded
      variants all NEGATIVE premium; tower the only non-negative one).
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
