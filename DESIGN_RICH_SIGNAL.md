# DESIGN — Rich Two-Axis Signal Recommender (2026-07-12)
For low-token Fable adversarial loop (≤5 iters), then train, then full battery. Grounded in
RICH_DATASET_REVIEW.md (exact encodings, file:line). Builds on the signed-latent instrument
(SignedAE, full-profile 0.495 ≈ EASE) — this variant adds the SEPARATE graded channels and trains/evals
on REALISTIC distilled-answerer interviews (not raw ratings).

## 0. QUESTION
Does the rich two-axis LLM signal — implicit knowledge {no_clue/rough/know_well} + explicit value
{hated/meh/liked/loved}, with mehs & refusals — input as SEPARATE graded channels, make held-liked
NDCG@10 grow FASTER over a realistic (distilled-answerer) interview, and do BOTH channels contribute?

## 1. WHICH SET (author's question — not all movies are rated)
- TRAIN on the DISTILLED ANSWERER v2.1 applied to the POPULATION trU = 161,541 users (ZERO LLM cost).
  Held-out USER split: ~155k train / 3k val / 3k test (disjoint users).
- QUARANTINE: the 300 study users (173 with real-LLM cells) — NEVER train/val/test; used ONLY for the
  final realistic transfer eval.
- Askable universe = the fixed 2,428 questions (1,128 concepts + 500 entity/attr + 800 items),
  deterministic layout, NO pools/sampling.
- TARGET = held-out LIKED items (rating ≥ 4 from the user's RATED set), DISJOINT from the interview inputs
  (leak-free). "Not all movies rated" ⇒ target lives on the user's rated-liked subset; the interview asks
  across the full 2,428 and the answerer responds realistically (know/val, mehs, refusals).

## 2. INPUT TRANSFORMATION (exact, from RICH_DATASET_REVIEW.md)
Per answered question (know∈{0,1,2}, val∈{-1,0,1,2,3}; val defined only if know≥1; refusal=(know=0,val=-1)).
Build item-space channels (dim ni=18430); concepts/entities spread over member items (member-bag,
genome×pop weights, leak-free):
- **value_vec[i]** = CENTERED value: hated −1, meh −1/3, liked +1/3, loved +1; rated items use
  crval = star − mean(known stars). 0 if unanswered or refusal.
- **know_vec[i]** = knowledge scaled: no_clue 0, rough 0.5, know_well 1.0; 0 if unanswered.
- **ref_vec[i]** = 1 if item was ASKED and REFUSED (know=0,val=-1), else 0 — the "asked, no clue" signal
  (distinct from "not asked"), the only disinterest/absence marker the data has.
- **mask[i]** = 1 if asked/answered (any know or refusal).
Encoder input = concat[value_vec, know_vec, ref_vec, mask] (4 × ni), L2-normalized (as SignedAE).
CONCEPT conflict rule: an item covered by multiple concept answers aggregates by HIGHEST-fidelity /
knowledge (order-invariant); direct item answer overrides concept-derived.

## 3. MODEL
- Encoder: SignedAE-style deep residual MLP (5 layers, swish, LayerNorm, DenseNet residuals) over the
  4×ni input → latent z (d=512). Decoder: Linear z → item logits.
- Loss: MULTINOMIAL log-likelihood of held-liked (the strength objective; α=0 — no dislike margin, it's a
  suppression objective NDCG@liked can't reward, established).
- Crash-safe frugal checkpointing (disk check, tmp-save, best+latest only).

## 4. CURRICULUM (realistic interviews — the whole point)
Distilled answerer v2.1 generates (know,val) per asked question. Per training example:
- STRATEGY MIX (arena_core STRATEGIES / select_items): random / popularity / entropy / on-profile /
  off-profile / adversarial / mixed — strategy-agnostic.
- LENGTH: log-uniform 1 → Tmax(24) — heavy on short cold-start; also include full-profile (dense end)
  for strength.
- REFUSALS: natural (unrated → no_clue → refusal) at the answerer's rate; ALSO sweep refusal rate in eval.
- MEHS: natural (val=1) present.
- Leak-free: interview inputs disjoint from held-liked target.
Reuse i25_fold_v3_sampler / arena_core sampler machinery (the sanctioned realistic-interview generator).

## 5. ABLATION (does each channel contribute — the headline)
Train these input configs (separate models for a clean ablation):
- **C0 mask-only** [mask]: bare consumed (crude implicit ≈ current baseline).
- **C1 value** [value_vec, mask]: explicit graded value.
- **C2 know** [know_vec, ref_vec, mask]: implicit graded knowledge only.
- **C3 full** [value_vec, know_vec, ref_vec, mask]: two-axis rich.
HEADLINE METRIC: NDCG@10 BY TURN (k=1,2,4,8,16,full) on realistic answerer interviews.
Report: does C3 > C1 > C0? does knowledge add over value (C3 − C1)? does the rich signal grow FASTER
(steeper early curve)? All full-catalog, CIs.

## 6. GATES (all checks, after training)
Strength (full-profile vs EASE 0.51/RecVAE 0.526), k-curve by turn, refusal-rate robustness, IG1 genre
purity, IG2 genre polarity, IG4 graded-value sweep, hygiene (intercept/no-leak/order/falsify/monotone),
the CHANNEL ABLATION (§5), and the REALISTIC TRANSFER eval on the 173 real-LLM users.

## 8. PASS-1 AMENDMENTS (Fable) — binding

**A. CRITICAL PRE-TRAIN LEAK CHECK — VERIFIED PASS (2026-07-12).** arena_core.py:260-270: the rated
passthrough sets `star = known[int(u.bank[r])]` and `rated = f["rated_flag"]` (computed from `known` via
user_features(known)) — so ONLY the KNOWN set gets star passthrough. Held-liked items are not in `known`, so
no real star leaks into value_vec. REQUIREMENT for the training build: the interview known-set must be split
DISJOINT from the held-liked target (standard leak-free split) — assert known∩held=∅ per user in the loader.

**B. EASE-imputation circularity (the top confound).** The answerer imputes unrated/concept values from
EASE-predicted taste, so a naive C1−C0 may measure "distilling EASE through the interview," not stated
preference. Guards (all required):
- **Rated-only HEADLINE**: primary ablation uses value_vec ONLY on FID_DATA (real-star) item answers, zero
  elsewhere. CAVEAT (state it, don't hide): this cleans only the 800 bank items — concept/entity values are
  ALSO EASE-derived, so the CONCEPT channel is NOT cleaned by this; report concept-channel results as
  "EASE-derived, interpret as upper bound."
- **Control C1e**: identical model fed ease_t directly on the asked positions (no answerer). If C3 does not
  beat C1e, the "rich signal" is just EASE plumbing → report as null.
- **173 real-LLM transfer = the true arbiter**: PRE-REGISTER that the channel ordering C3>C1>C0 must reproduce
  on the 173 (caveat low power n=173). This is the honest test that realistic answers carry recoverable signal.

**C. Knowledge axis lives at CONCEPT level, not item level.** Item know≥1 is ~99% saturated + rated forced
know_well → item know_vec ≈ rated-flag (redundant with mask); item ref_vec ~1% = noise. So the ablation MUST
isolate CONCEPT/ENTITY knowledge (ablate know/ref BY QUESTION TYPE), or "knowledge contributes" is unlocatable.
Report item-knowledge and concept-knowledge separately.

**D. Ablation = SEPARATE models** (C0/C1/C2/C3 + C1e), identical arch/seed/budget — NOT eval-ablation of one
C3 model (that feeds OOD zeroed inputs, conflating OOD damage with channel info). Add SEED REPEATS and PAIRED
eval interviews (same question sequences + answers across all configs) — deltas are small; pairing is the
cheapest power win. Eval-ablation of C3 reported only as secondary.

**E. Encoding fixes:** (1) rescale crval to [−1,1] — it (±3) and centered levels (±1) must not share one
channel at incompatible scales. (2) Normalize PER-CHANNEL, not L2 over the 4×ni concat (else mask dominates
and the norm shifts with interview length). (3) Add a SOURCE-FLAG channel (concept-derived vs direct-item)
so the value channel doesn't conflate the two and the concept-dilution question stays testable.

**F. Curriculum fixes:** define the FULL-PROFILE input regime explicitly (full-profile-VIA-ANSWERER ≠ raw
ratings, so the 0.495-vs-EASE strength number is apples-to-oranges unless stated — report it as
"answerer-full-profile", a distinct anchor). Fix ONE eval strategy + length + refusal schedule, IDENTICAL
across C0–C3, for the paired comparison.

## 9. PASS-2 AMENDMENTS (Fable) — binding; design now PILOT-READY

**A2. Leak assert extends to the EASE context.** Not just star passthrough — the answerer's EASE fold-in
`t(u,i)` must condition ONLY on `known`. Loader assert: every value the answerer emits (passthrough AND
EASE-imputed) is a function of `known` alone; known∩held=∅. CAVEAT (state, don't hide): `ease_v21.npz` B was
trained on population users' FULL histories (only the 300 study held-halves were dropped), so for pop val/test
users B saw their held-liked. Diffuse at 161k, but note it as a known limitation (retraining B excluding
val/test held-halves is the clean fix if the result hinges on it).

**B2. PRIMARY STATISTIC = paired per-user EARLY-AUC.** Trapezoid area under NDCG@10-vs-turn over k∈{1,2,4,8};
report the C3−C0 delta with bootstrap CI over users×seeds, PAIRED per user (same question sequences+answers
across configs). PRE-REGISTER exactly ONE confirmatory contrast: **C3−C0 early-AUC > 0, one-sided, paired.**
Channel ORDERING (C3>C1>C0) is DESCRIPTIVE only (n=173 multi-way is underpowered). ALSO report
CEILING-NORMALIZED AUC (each curve ÷ its own answerer-full-profile NDCG) to separate SPEED from ASYMPTOTE —
else a higher-plateau C3 fakes "faster."

**C2. CAPACITY-EQUALIZED ablation (critical).** Do NOT use different arch sizes per config. Use ONE 5×ni
architecture for ALL of C0/C1/C2/C3/C1e; realize each config by ZEROING the unused channels IN THE TRAINING
DATA (zero inputs carry no gradient → identical capacity, no OOD, trained-in). Same seed/budget. Config list:
C0=mask only; C1=value+source+mask; C2=know+ref+mask; C3=all; C1e=EASE-t on asked positions+mask.
Knowledge-by-question-type: within-C3 eval masks (item-know vs concept-know), secondary.

**D2. Per-channel scaling:** NOT L2-to-unit (inflates a single turn-1 answer to unit norm). Use fixed-constant
or 1/√nnz per channel so magnitude grows with evidence and is comparable across turns.

**E2. Independence caveat:** the distilled answerer v2.1 was dial-refit on the 173's real-LLM cells, so the
transfer distribution is tuned toward those users — not label leakage, but state the independence caveat.
Source-flag channel is present in every value-bearing config (C1/C3/C1e).

## 7. OPEN QUESTIONS FOR FABLE
1. The answerer IMPUTES unrated-item values from EASE-predicted taste (know≥1 sampled). So the explicit
   value channel partly carries EASE's knowledge, not just stated preference — does this INFLATE the value
   channel's apparent contribution / risk distilling EASE? How to interpret/guard (e.g., report the
   ablation ALSO restricted to RATED-only answers, where value is real)?
2. Concept answers → member-bag into item space: right call, or a separate concept channel? (I2 Phase-1.6
   said input-space member folding can dilute — but that was for DISLIKE; here it's graded value+know.)
3. ref_vec (asked-and-refused) as a channel — useful signal or noise given item know≥1 is ~99% saturated?
4. Separate-model ablation (C0/C1/C2/C3) vs one C3 model eval-ablated — which is the honest test of
   "each channel contributes"?
5. The single most dangerous confound that would make a positive rich-signal result NOT real.
