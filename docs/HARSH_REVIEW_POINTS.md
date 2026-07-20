# HARSH REVIEW POINTS — the mistakes we must never repeat

**Purpose.** A single consolidated checklist of every concrete criticism raised against this
project across all harsh / adversarial / expert / design reviews (2026-06 → 2026-07). Each item is
a *checkable rule*, not vague advice. Sources in *(italics)*.

**How to use (BINDING — CLAUDE.md HARD RULE #5).**
- Before any **experiment**: check it against *Data handling*, *Experimental design*, *Statistical
  rigor*, *Baselines*, *Simulator circularity*.
- Before any **claim / paper edit**: check it against *Overclaiming*, *Stale numbers*, *Framing*.
- If a planned action matches a ✗ below, STOP — you are about to repeat a known mistake.

Source files consolidated here (now retired into `new_chapters/_meta/` and memory): `REVIEW_HARSH_2026-07-01`,
`REVIEW_ADVERSARIAL_2026-07-05`, `CASPER_Expert_Review_2026-06`, `INSTRUMENT_REVIEW`,
`ANSWERER_V1_DESIGN_REVIEW`, `RICH_DATASET_REVIEW`, `QUAL_REVIEW_DANS2`, `answerability_design_review`,
memory `expert-review-june-2026` + `harsh-review-and-trainseed-overturn`.

---

## ★ THE WORST REPEAT OFFENDERS (scan these first)
1. **Circular measurement.** Simulated answer, belief, loss, and metric are all functions of `u*` in
   one latent space → any method probing `u*`'s coordinates wins by construction. Never headline a
   result that is a tautology of the objective. Reframe measurement-theoretically; add the
   simulator-alignment limitation to *every* paper; run answer-noise ablations.
2. **No CIs / wrong significance unit.** Bare +0.01–0.02 deltas on ~300 users are noise; between-seed
   σ is invalid (seeds vary only the eval split). Use a **paired per-user bootstrap** on every bolded
   delta. Many "wins" die under it.
3. **Unfair / missing baselines.** Fair comparator hidden in a side table; method gets privileged
   profile access baselines are denied; the cheap heuristic that could kill the paper never built.
   Build the 2-line answerability-aware static heuristic + modal-static ablation *yourself, first*.
4. **Stale / unprovenanced numbers.** Superseded and corrupted-cache numbers survive in text; ruler
   changes (354 vs 302 vs 304 users) mid-paper. Every number → ledger row + script + date; one fixed
   ruler stated once.
5. **Zero human validation.** Every rate/lever/delta is simulator output. One ~30–50 person
   round-trip study defuses the objection across A–E.
6. **Selective reporting.** Worst seed dropped; test-set checkpoint selection; a built-then-removed
   non-circular answerer. Report all seeds, pre-register selection, disclose every exclusion.

---

## Statistical rigor
- ✗ Bare headline deltas with no CI. → **Paired per-user bootstrap** (same users, method vs baseline), one shared harness, on every bolded delta. *(REVIEW_HARSH S1; ADVERSARIAL; expert-review)*
- ✗ Between-seed σ as the significance unit — the 5 seeds share the same ~300 users (eval-split-only). → The unit is the per-user bootstrap, not seed spread. *(REVIEW_HARSH B-B4)*
- ✗ "N seeds" claimed while the model is trained **once** (`manual_seed` hardcoded). → Retrain ≥3 real training seeds before any "robust learned policy" claim. *(REVIEW_HARSH C-B5, B-B3; trainseed-overturn)*
- ✗ Checkpoint selected on the **test** set (contradicts own Algorithm 2). → 3-way split, pre-registered selection rule, + mean-over-epochs robustness row. *(REVIEW_HARSH S1, B-B2)*
- ✗ Worst seed (123, +0.002) silently dropped. → Report all seeds with per-seed deltas; footnote any exclusion. *(REVIEW_HARSH S1, B-B3)*
- ✗ No multiplicity control across a large test battery (garden of forking paths). → Correct for multiple comparisons; apply the same bootstrap that already killed C's/E's deltas to headline claims. *(ADVERSARIAL)*
- ✗ All CIs conditional on one fixed answerer split. → Run 2–3 alternate splits on ~50 users; print split-sensitivity. *(answerability HOLE 5)*
- ✗ Uncertainty computed in code (±std) but dropped from tables. → Weave ±std into every table. *(REVIEW_HARSH D-B7)*

## Baselines & comparators
- ✗ Headline vs a **weaker** comparator; the fair one sits in a side table (C: actor vs binary CASPER-R halves to +0.011/+0.020 vs the graded uent+GRAW). → Lead with the fair, apples-to-apples margin. *(REVIEW_HARSH S2, C-B1)*
- ✗ Baselines **blinded** while the method folds the user's known-half tokens (D: PEBOL +0.052/+0.076 with equal access = most of the "gap"). → Symmetric access for all; per-type answer-rate ablation + break-even p before the table ships. *(REVIEW_HARSH S2, D-B4; ADVERSARIAL D)*
- ✗ Named priors (PEBOL/EAR/SCPR/UNICORN/ConTS) **skipped**, justified by a self-contradicting oracle bound. → Run them (fair-strict PEBOL 0.317 < CASPER-R 0.360) or give an explicit incompatibility statement. Never justify skipping via a bound measured under different settings. *(REVIEW_HARSH S2, A-B1)*
- ✗ Missing the cheap baseline that could **kill** the paper: answerability-aware static heuristic (divisiveness × P(answerable)); greedy info-gain; modal-path-as-static-order ablation of the learned policy. → Build them yourself first; if they match the learned policy, the "adaptive win" story dies. *(REVIEW_HARSH B-missing, D-B5; expert Flaw E)*
- ✗ The assumption-free positive result buried (D's `random5` 0.390/0.185 beats D1 with zero machinery). → Promote it; state band-conditionality. *(REVIEW_HARSH D-B3)*

## Data handling & leakage
- ✗ **Reducing data without author sign-off** — truncated caches, top-N pools, 300-user selection, token caps have each silently changed the answer. → Non-lossy alternatives only (bucketing, streaming, sparse ops, per-item folding); if unavoidable, STOP and ask. *(HARD RULE #1)*
- ✗ Quarantine breach — the 173 LLM-judged / 300 study users in any train/val/test cohort. → Firewall study IDs out of every cohort; train on the distilled population only. *(RICH_DATASET §2; ANSWERER_V1)*
- ✗ Interviews that can ask about **held-out target items** (answer leaks the target's rating). → Exclude held-out targets from the question bank; masked-item validation is a *separate* calibration pass, never inside an evaluated interview. *(answerability HOLE 6)*
- ✗ Non-deployable eval shortcut (mask the entire known half, `s[seen]=−1e9`) undisclosed. → Disclose in limitations. *(REVIEW_HARSH D-B8)*
- ✗ Corrupted-cache artifact inflated a win (+0.015) and overstated a collapse (0.328 vs 0.343). → Always verify baselines **snap to canonical numbers** before trusting a delta. *(REVIEW_HARSH S3, C-B2)*

## Simulator circularity & validation
- ✗ The thesis rests on one circular measurement (answer = projection of `u*`; belief = fold of answers; loss = `1−cos(belief,u*)` — all in one 64-d space). → Reframe measurement-theoretically under a stated answer model; simulator-alignment limitation in every paper; answer-noise ablations. *(ADVERSARIAL thesis-flaw; REVIEW_HARSH S4)*
- ✗ **Zero human validation** anywhere. → One ~30–50 person round-trip study (rendered questions → graded answers → downstream NDCG). Highest leverage across A–E. *(ADVERSARIAL; REVIEW_HARSH §7.7)*
- ✗ A "framing lever" that is actually the simulator's own sampling rule (D hidden-gem = argmax of `u*·Q/log(cnt)`). → Noisy hidden-gem answerer (softmax / below-median-popularity); optimistic flags on every distinct/align/rating number incl. abstract. *(REVIEW_HARSH D-B2)*
- ✗ Shared-prior circularity — LLM-predicted values and EASE cross-check are both CF priors over the same space; agreement ≠ independence. → Masked-rated-item validation for real ground truth + fidelity σ; keep LLM values sensitivity-only. *(answerability HOLE 1, 4)*
- ✗ Fuel gate passing for the wrong reason (LLM genre-stereotyping mimics per-user structure). → Held-out-half validity check (held-rated vs matched never-rated within popularity tiers). *(answerability HOLE 2)*
- ✗ Statistical fuel ≠ exploitable fuel (real heterogeneity, yet a fixed schedule can stay optimal). → G2 exploitability diagnostic (answerability-aware vs -blind greedy = upper bound on adaptive gain); adaptive claim needs G1 AND validity-gap AND G2>0 with CI excluding 0, pre-registered. *(answerability HOLE 3)*
- ✗ The one **non-circular (learned) answerer** was built and removed. → Report it or justify exclusion; dropping it as "too noisy" is selective reporting. *(ADVERSARIAL C; REVIEW_HARSH B-missing)*

## Overclaiming & novelty
- ✗ Headline contribution not actually evaluated in the paper (B claims a continuous snap policy; it's in C). → Contributions must match what the paper evaluates. *(REVIEW_HARSH S2, B-B1)*
- ✗ "First X" on knife-edge results (4/5 learned recipes lose to the static heuristic; win is seed-fragile ≈ entropy). → Drop/qualify "first"; reframe as "when/why policy learning pays." *(REVIEW_HARSH B-B5; trainseed-overturn)*
- ✗ Claims contradicted by the paper's own tables (C's "discrete can't exploit graded" vs its own uent+GRAW row; A's "first selector to beat random" vs its own tab:select). → Reconcile every claim against internal tables before submission. *(REVIEW_HARSH C-B1, A-B6)*
- ✗ A paper's own refutation in a later section (C §5.6: continuity collapses below MOSTPOP under a realistic channel). → Lead with the boundary result ("continuity wins only under slider-grade answers"). *(ADVERSARIAL C)*
- ✗ Deltas that merely restate the training objective (polarity loss → polarity gap; "question channel strongest" = probing the decoder's own axes). → Don't headline a tautology. *(ADVERSARIAL A)*
- ✗ Warm-start / full-profile number sold as the cold-start result (A "ties EASE 0.554"). → Label the task each number belongs to. *(ADVERSARIAL A)*
- ✗ A **definition** sold as a discovery ("answerable iff rated" → the 5:1 rate is just rating-matrix density; "rated" ≠ "could answer"). → Popularity-conditioned seen-probability model; the item-vs-concept headline likely moves. *(ADVERSARIAL B; REVIEW_HARSH B-B7)*
- ✗ "Open/askable" that is enumeration or the rejected snap (A's 745 "open" concepts = curated Tag-Genome; E's "rendering" = the k=1 snap C rejected). → State honestly; only the SBERT bolt-on is open-vocab. *(REVIEW_HARSH A-missing; ADVERSARIAL E)*
- ✗ Illusory continuous action space (FAISS-snapped to a discrete entity → effective space discrete + aliasing). → Reframe as "embedding-space policy with discrete concept grounding." *(expert Flaw D)*

## Stale numbers & provenance
- ✗ Superseded/retracted numbers survive in text (C "+0.010"/"0.328"; B "354 users"/"six seeds"/+36%; E OMP mis-attribution; A 44%/75% vs ledger 50%/70%). → One provenance pass per paper: every number → ledger row + script + date; purge pre-fix numbers. *(REVIEW_HARSH S3; RESPONSE)*
- ✗ A claim built on a run the lab notebook itself retracted ("DO NOT put in paper until re-run"). → Cut or re-derive. *(REVIEW_HARSH E-B3)*
- ✗ **Ruler chaos** — 354 vs 302 vs 304 test users in one paper; metric depth chosen per dataset; tail undefined. → One fixed ruler + user count + metric definition, stated once, used everywhere. *(REVIEW_HARSH B-B6)*
- ✗ Central results with no reproducible script (E's un-askability table). → Commit script + recorded output before submission. *(REVIEW_HARSH S5, E-B5)*
- ✗ Small deltas quoted more precisely than the rounding (0.361 vs 0.3608; "matches" for a 7% gap). → Consistent rounding; a 7% gap is not a match. *(REVIEW_HARSH minors)*

## Experimental design & methodology
- ✗ Sample budget off by 2–3 orders (DDPG, 384-dim actions, ~1–2.5k transitions; needs 10⁵–10⁶). → Take the LLM out of the training loop (rule-based simulator → 50k+ episodes); simplify the RL. *(expert Flaw A)*
- ✗ Broken reward SNR (per-turn ΔNDCG ~0.005 vs per-user noise ~0.20; NDCG-delta telescopes). → Concept-accuracy-delta reward; eval at pool=500, min_targets=5–10; try a contextual bandit first. *(expert Flaw B)*
- ✗ Eval with no headroom (pool=100/min_targets=25 → 0.52 baseline; revealing preferences can *lower* NDCG). → Pre-register metrics; pool=500, min_targets=5–10, n≥100, report CIs. *(expert Flaw B)*
- ✗ Running without a **design sheet** (question, exact Ns, full action space, baseline symmetry, metric+MDE, every shortcut + its non-lossy alternative). → No run unless its entry pre-registers the outcomes and the decision each changes; if no decision changes, it doesn't run. *(CLAUDE.md; ANSWERER_V1 §6)*
- ✗ One experiment defining its own environment (per-experiment banks / answerability rules → circularity + post-hoc arena patches). → Build the answerer **once**, validate, freeze, version; downstream consumes only that. *(ANSWERER_V1 §0,§5,§6)*
- ✗ Pilot on a different dataset (ML-1M) than the target (ML-25M). → Re-run the mini-pilot on the target before spending gate budget. *(answerability hole 5)*
- ✗ Free parameters unpinned ("meaningful answer" conflates recognition/opinion; "maybe" threshold; batch order effects; no model-snapshot/temperature pin — a mid-study model update silently invalidates the cache). → Define per channel, pre-register thresholds, measure batch-vs-single agreement, pin the exact model snapshot. *(answerability holes 1,2,3,6,8)*
- ✗ Gate sampling confounds identity with taste (1 user/genre). → Stratify by profile size AND taste cluster, ≥20 users/cell. *(answerability hole 7)*
- ✗ Early-stopping RL on a val dip; train-then-eval-throwaway. → Track the return curve; record the peak on a disjoint val; save all checkpoints. *(MEMORY be-patient / save-all-checkpoints)*

## Recommender / model
- ✗ Polarity encoded **textually** in SBERT ("likes:X | dislikes:Y" → 0.75–0.78 sim, conflated). → Keep polarity **structural** (separate one-hot / gating), never textual; feed both polarities so the channel has variance; acceptance tests (rating flip flips ranking; like/dislike overlap <30%; monotone with length). *(expert Flaw C)*
- ✗ Bolting a sign channel onto a nonneg multinomial/implicit CF (goes inert, ΔNDCG=0.000; explicit once *hurt*: 0.269 vs 0.358). → The tension is in the **likelihood**, not the data (dislike separable, r=+0.46); dislike must act as repulsion in a taste latent with a supervised gradient path, not negative weight in a like-ranker. *(INSTRUMENT_REVIEW D/E)*
- ✗ Sparse listwise-ranking objective on a large catalog (ML-1M 3706 items → 0.003, worse than random). → Strength comes from full-matrix multinomial-denoising reconstruction + capacity, not a listwise head. *(INSTRUMENT_REVIEW A)*
- ✗ Diagnosing the prime suspect from memory, refuted by code (the false "elicitation-only vs full-matrix" hypothesis). → Cite `file:line` from actual code before asserting a mechanism; read your own results first. *(INSTRUMENT_REVIEW A; MEMORY read-own-results)*
- ✗ Distilled value channel heavily diluted (LLM masked corr 0.547 → distilled 0.177). → G5 value-signal-preservation gate: upgrade value model (neighbor CF), gate at corr ≥ ~0.4 before any policy trains. *(QUAL_REVIEW_DANS2 TODO 3)*
- ✗ Scoring with a log-count **popb** floor instead of the learned decoder bias (corr only ~0.79 → craters full-NDCG, once produced a false "full is solved"). → Use bias+Wd@μ everywhere; a reconstruction must reproduce ~0.486 full-profile before any conclusion. *(HARD RULE #4)*
- ✗ Reporting tail-only (or full-only) NDCG. → **Always both** full AND tail@10 (the Jul-14 effect was +47% tail / +6% full). *(HARD RULE #4, #2)*

## Framing, theory & writing
- ✗ Post-hoc theory-fitting — the non-adaptivity theorem invoked to explain adaptivity winning *and* tying. → NEVER argue "static is optimal so adaptivity is worthless" (settled: +47% tail from one question); the theorems assume linear-Gaussian + variance-only, which this project has neither. Prior nulls are arena-conditional. *(HARD RULES #2,#3)*
- ✗ Marketing a failure as a contribution — OR silently demoting a real loss. The 8 failed policy runs go in no paper (weak model); but a real tail loss (D −0.008) must be stated, and a within-noise +0.003 demoted, not listed. → Report faithfully: losses stated, sub-noise demoted, dead runs dropped. *(HARD RULE #3; stop-selling-negatives)*
- ✗ Single domain (ML-1M) carrying every elicitation claim while cross-domain reverses (tail −0.006 ML-25M) or refutes (−0.084 Goodreads). → State single-domain scope; a claim that reverses out-of-domain is not general. *(ADVERSARIAL)*
- ✗ Recognition vs recall conflated in "4× more sample-efficient" (naming a film ≠ clicking one; the memory literature argues against the popularity assumption). → Engage the free-recall psychology; qualify the claim. *(REVIEW_HARSH D-missing)*
- ✗ Missing/bare bibliographies + no cold-start scope paragraph (Paper D has zero citations). → Complete bibliographies; add "we extract a profile the user already possesses; truly-cold users out of scope." *(REVIEW_HARSH D-missing, A-B8)*
- ✗ Compilation/placeholder rot (`\newcommand{\AA}` breaks compile; empty `\author{}`; `\todo`; 3-entry bibs; dead figure assets; 600-word abstracts). → Compile end-to-end; remove placeholders/dead assets; halve overlong abstracts. *(REVIEW_HARSH A-B9)*
- ✗ Measuring faithfulness with the encoder the paper indicts (SBERT label-sim > 0.3, arbitrary threshold). → LLM-judge and/or human ratings + threshold sweep. *(REVIEW_HARSH E-B6/B7)*
- ✗ Wrong citations (PEBOL authors; HELF = Rashid 2008 not 2002; 20-questions = Sznitman not Zhang). → Verify every citation's authors, year, venue. *(REVIEW_HARSH C-B7; MEMORY novelty-map)*
