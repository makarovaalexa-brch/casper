# BLIND ADVERSARIAL VALIDATION of PROGRAM_STATE_2026-07-10.md

Validator: blind (no prior project context, by design). Date 2026-07-10. Read-only audit of
`casper/PROGRAM_STATE_2026-07-10.md` against repository artifacts (experiments/*.md, .cache JSONs,
scripts/*.py, papers/*.tex, git log). Verdict scale: CONFIRMED / OVERSTATED / WRONG / UNVERIFIABLE.

---

## PART 1 — §2 ASSET LEDGER, claim by claim

### Instruments

**F1. "RecVAE-d512 (certified, ties EASE) — the frozen scorer."**
Not independently re-derived here; consistent with committed record (paper3 `sec:i2instrument`/
`sec:i2apparatus`; memory arc). Training code is trU-only (`scripts/instrument2/ml25m_recvae.py`).
Verdict: **CONFIRMED (by record, not re-measured).**

**F2. FOLD-V3 gate values.** Artifact: `experiments/FOLD_V3_BUILD.md` + `.cache/i25_fold_v3_gates.json`.
- GoT gate **+0.184**: artifact G2 = +0.1843 [+0.1550,+0.2167], n=290, PASS. **CONFIRMED.**
- Prolific gate **+0.316**: G2b = +0.3159 [+0.2863,+0.3442], n=400, PASS. **CONFIRMED.**
- Implicit ablation **+0.020**: G7 = +0.0203 [+0.0085,+0.0325], PASS. **CONFIRMED.**
- Clean profile beats native **+0.017 where v2 lost 0.135**: G4 fold 0.4994 vs native 0.4820 =
  +0.0174 (the artifact's gate table prints "-0.0175" — sign/convention inconsistency in the
  artifact itself, arithmetic favors the fold); v2 comparison: `REFERENCE_LADDER.md` fold-v2
  full-profile 0.3589 vs native 0.4938 = -0.1349. **CONFIRMED** (numbers), with the sign glitch noted.
- **-0.028 trade**: G5 = -0.0280. **CONFIRMED as a number, OVERSTATED as framing** — the artifact
  records G5 as a **FAILED GATE**, not a "known trade", and the synthesis **omits two further
  failures entirely**: G6 anti-saturation FAIL (max step -0.0035 vs -0.003 bound; ST contingency
  fired and also failed) and G1 item-explicit-from-cold FAIL (-0.0027, CI spans 0). The artifact's
  own verdict line is `ALL GATES PASS: False`. Decisive gates (G2/G2b) did pass per the signed
  stop rule, but "(all gated, all committed)" reads as all-pass; the true record is 12-of-16
  sub-gates with 3 documented fails + 1 diagnosed artifact.
- Cohort/robustness caveats dropped: gates run **once**, single training run per curriculum arm
  (2 arms), best-on-val checkpoint, 290–400 **synthetic** users, and the gate world is the
  **ungated sampler** (see F14). Commit exists (58ff224) — "committed" is true.
- "level-flag semantics inert (no-clue not negative)": supported — per-level pulls rough +0.236 /
  know_well +0.202 / no_clue_neg +0.244 all same sign & size. **CONFIRMED.** The no-clue/anti-surprise
  ablation (fold-v3.1) exists and is mid-run (`.cache/i25_fold_v31_*` present, partial probes) —
  synthesis correctly lists it as in-progress.

**F3. ANSWERER v2.1 "values at LLM parity (item corr 0.550 vs LLM's 0.547)".**
Numbers real but **the juxtaposition is apples-to-oranges**: 0.550 = corr(model E[stars], **LLM**
stars), LOUO over the 173 (`DANS_BUILD.md` Stage B); 0.547 = corr(**LLM** stars, **real held-out
ratings**) on the **10-user small batch** (`ANSWERER_SMALLBATCH.md:31`, echoed in
`.cache/dans/value_v21.json` as `llm_masked_benchmark_corr`). Different target variable, different
cohort. (The like-for-like model-vs-real number, v2.1 G3 corr 0.651 on 6,962 rated cells, is
actually better — the substantive point survives, the quoted "parity" measurement does not exist.)
Verdict: **OVERSTATED.**

**F4. "Gates 13/13 vs corrected targets."** **WRONG as phrased** — a conflation of two different
gates: (a) the **13/13** is ITER-2 G2 decisive statistics, measured against the **OLD,
flutter-inflated** fuel targets (full table 16/18); (b) the gate **vs corrected targets** is v2.1
G2, which has only **3 statistics** (trait ICC per channel), 3/3 within 0.02 tolerance. There is no
13/13-vs-corrected-targets result anywhere. Additional dropped caveats: the decisive-set was
recounted 10→13 with added tolerance after v1.0's 1/10 FAIL (defensibly — v1.0 still fails under
it); and `DANS_BUILD.md` honest-caveat #3 states "G2 passing was driven primarily by the two
calibration fixes", one of which (concept taste_align ×0.70 → 0.105 = target center) is literally
tuned to the target, so that match is not independent evidence.

**F5. "serves all 162k users lazily."** **OVERSTATED.** The distillation build STOPPED after gates
per contract; the 162k generation and gate G4 (20k) were explicitly **not run**
(`DANS_BUILD.md:283,595,681`). Lazy serving exists only as new arena code (`arena_core.py`), whose
answerer is not the gated model (F14).

**F6. "G1 beats baselines incl. niche."** **CONFIRMED with dropped strata** — prereg (overall +
niche) passes on all channels; two non-required strata tie/lose (entity-mid 0.625 vs 0.636;
item-high = base).

**F7. The 173-user eval world "quarantined (touched twice max)".** **OVERSTATED.** The twice-max
rule is real but **prospective only** (signed `DESIGN_SHEET_POLICY_ARENA.md`, 2026-07-09). Before
it, the 173 were the eval cohort of many verdict reads (Jul 8–9: phase4 arenas v1/v2, adaptivity
battery, habitat, fair re-runs, tree studies, r-learned, STATIC_CONTAMINATION itself — see
`STATE_2026-07-08.md`: "NONE beats the best fair static at T=12 on 173 users"). Enforcement is a
trU-only cohort constructor + docstrings ("173 never touched", `arena_core.py:7,316`;
`arena_eval.py:9,78,147,427`) — **no assert, no touch-counter**; nothing prevents loading
`answerer_v1_grid173_WORKING.json`. Also dropped everywhere in the synthesis: the grid is an
**unfrozen partial 173/300** ("DIRECTIONAL ONLY … re-run on the frozen 300-user grid before any
citation" banner on ADAPTIVITY_BATTERY_V1, STATIC_CONTAMINATION, FAIR_RERUNS, REFERENCE_LADDER,
TREE_LOUO, …), and the 173 are an availability-defined interim subset of the designed 300
(`dans_build.load_173`), not the designed cohort. "values validated vs real held-out ratings" =
the 10-user small-batch corr 0.547 + masked MAE ≈0.70 — thin for the word "validated".

**F8. Beliefs (kmap, LOUO judged-grid MF, pmodel).** kmap is leak-guarded in code
(`kmap_build.py:107-108`, `assert leak == 0`, trU only). pmodel per-arena recalibration is a
recorded standing rule. **CONFIRMED.**

### Science findings

**F9. THE FLUTTER.** Rate spread **0.66**: `SHUFFLE_PROBE.md` user 85673 spread 0.660 (others
0.540, 0.160); stars stable (MAE 0.12–0.21). Decomposition **34 / 0.3 / 4.5 / 22 / 39**:
`LLM_DECOMPOSITION.md` item k≥2 headline row = 34.3 / 0.3 / 4.5 / 21.7 / 39.2 (sum 100.1, noted).
**CONFIRMED exactly.** Dropped caveats: the probe is **3 users, 9 calls, $0.047**; the artifact's
own §(e) flags a **formal repeat-study (30–50 users × ≥3 re-sends) as still required** before any
human-transfer or policy-value claim leans on the corrected ICCs; cross-check of flutter variance
(0.046 equating vs 0.068 probe) agrees only "to the same order of magnitude". "Every LLM-judge
study in the literature is currently exposed to it" is rhetoric, not a measured or lit-verified
claim.

**F10. TERRITORY > DIAL.** Three routes exist as claimed (QUAL_REVIEW profile reading; census CV —
concept CV R² 32% generalizes, item 3% weak, entity **−28%** no signal; equating). **LARGELY
CONFIRMED**, with a dropped null: the direct era×territory test on the item channel was **not
supported** at n=9 pocket users (`DANS_BUILD.md` ADDENDUM); territory survives via co-knowledge +
genre/taste-alignment coefficients.

**F11. CONSUMPTION-AS-TASTE "now VALIDATED INSIDE a trained instrument".** G2/G2b pass decisively
supports it — **CONFIRMED on synthetic gate cohorts**, but "validated" = one single-seed gate run
inside a world whose knowledge draw hard-codes the surprise coupling being tested (sampler
K_ANS=0.9·surprise; see F14). The instrument *learned to use* the feature the world *generates
with* — partially self-referential; an external (LLM-judged or human) confirmation does not exist yet.

**F12. Rated-ness ceiling "~0.73 AUC" + recall premium "+0.058/answer".**
Ceiling: 0.725 AUC (`STATE_2026-07-08.md`, third confirmation; scope caveat — rated-ness, not true
answerability — is respected by the synthesis wording). **CONFIRMED.** Recall premium: T1b
**+0.0547 [+0.002, +0.112]** per answer slot, held-out users; later "re-confirmed +0.058 under
fold-v2" (STATE Jul-9 §4). **CONFIRMED as a number, caveats dropped**: the CI barely excludes 0,
and the result carries the DIRECTIONAL 173/300 unfrozen-grid banner.

**F13. The evaluation sins.**
- Contamination **+0.028**: `STATIC_CONTAMINATION.md` pooled s-item T=12 = +0.0283 [+0.0193,+0.0377],
  n=346 paired deltas. **CONFIRMED**, three dropped qualifiers: file header says **DIRECTIONAL
  ONLY / re-run on frozen grid before any citation**; it is **family-specific** (s-mixed only
  +0.006) and budget-varying (+0.024→+0.034); and the file's "de-biased flip to ~+0.02" corollary
  was later retracted (Jul-9: "contamination shifted both arms … the gap was never contaminated" —
  the synthesis wisely does not repeat it, but cites the artifact without noting its interpretation
  table is partly dead).
- Fuel-free pools 0.000 in-pool refusal: `FRESH_AUDIT.md` Finding 1. **CONFIRMED.**
- 86 users vs thousands: 86 = per-half construction cohort (`FRESH_AUDIT`, `STATE`); "thousands" is
  a design-sheet inference. **CONFIRMED.**
- Survivorship, value-asymmetry, OOD folds: all in the record. **CONFIRMED.**
- "E1–E7 codified": **OVERSTATED** — no canonical E1→E7 enumeration file exists; definitions are
  scattered (STATE, arena docstrings). In arena code: E1 implemented, E2 by construction, E4
  implemented, E5 procedural, **E6 (item-static-weak sanity) unimplemented**, E7 = a hand-written
  markdown table, not a verified property.

**F14. (Not in the ledger, but poisons several ledger words) THE ARENA WORLD IS NOT THE GATED
v2.1 MODEL.** `scripts/i25_fold_v3_sampler.py` — the answerer that trains fold-v3 and the arena
policies — loads only `ease_v21.npz` + `sigma_v21.json`. It does **not** load the fitted ordinal
knowledge model (`models_v21.json`: co-knowledge proximity, fame, census features) that earned the
gate passes. Its knowledge draw is a hand-built base-rate + K_ANS·surprise + trait sigmoid with
**hand-set couplings (0.9 / 0.5)**. The DANS record itself shows a base-rate+noise model FAILS G2
(v1.0: 1/10). Calling the arena world "v2.1 answerer (certified)" is **credential transfer**: the
certificate attaches to a model the arena does not run. Additionally the arena universe includes
unjudged questions (IMDb entities; run.log `nQ=1530`), where answers are pure extrapolation.

### "In progress TONIGHT"

**F15. Arena description.** Code exists and matches the b0–b4 + four-class + tie-by-construction
design (`arena_policies.py`, `arena_eval.py`). **Two overstatements**: (a) "full 2,428-question
universe" — the coded/logged universe is **nQ=1530 (200 concepts, not the judged 1,128; 30 attrs,
not 500)**, contradicting the signed design sheet (`ARENA_CODE_AUDIT.md` Finding 6); (b) "pool bugs
designed out" — the program's own blind pre-verdict audit (same date) found **critical live
defects**: b4's objective peeks at the eval user's held-out likes while E7-labelled `privileged?
no` (Finding 1); `build_b2` accepts a partial cached sequence (cache currently holds **10/24
picks**) so b2 can silently run truncated and every arm "wins" (Finding 2); answer-cache npz keys
don't encode cohort config (held-out leakage across configs, Finding 3); b2 built on **50 users**
with a capped candidate search vs scorer-A's 1000 users — "the learned static b2 at population
scale" is **not what the defaults build** (Finding 4). Whether tonight's run fixes these is
unverifiable from the repo. The synthesis honestly lists the blind audit as in-progress; it exists
with these findings.
- Fidelity-trust probe "queued but unverified": honest; `arena_fidelity_probe.py` exists, no result
  md. **CONFIRMED.**
- ">=3 training seeds on leaders": plan, not yet evidence. Every historical fold/gate result above
  is single-seed.

---

## PART 2 — §3 LOGIC AUDIT (the epistemic structure)

**F16. "The 173 is the only read with independent evidential force" — the word "independent" is
wrong, and §3 contradicts itself.** The distilled world was not merely "fit to the judge world" in
the abstract: (i) all knowledge/value channel coefficients were **final-refit on all 173** (LOUO
used only for validation; `DESIGN_SHEET_DISTILLED_ANSWERER.md` §2, `dans_stages.py`); (ii) the
**equating transform and the corrected gate targets are derived from the same 173 grid** the
headline will be read on — the world is self-equated to its own eval set; (iii) two calibrations
were **tuned until synthetic statistics hit 173-measured values** (taste_align ×0.70; entity
per-cut sigmas); (iv) the EASE backbone trains on interactions including the 173's known halves
(their held-out halves are code-guarded out — that guard is real, `dans_v21.py:520-528`); (v) the
173 were already multiply read pre-quarantine (F7), and design choices (lenient regime, fold pick,
universe) were informed by 173-derived measurements. There is no per-user memorization path into
eval (training users are disjoint trU; held-out ratings never enter any model — verified), so the
173 read **can** catch determinism/ecosystem-quirk exploits; it **cannot** certify human transfer,
and the sim-real gap table is optimistically biased by construction (the sim was calibrated to
reproduce this cohort's statistics). Only the human study is independent. Verdict: **OVERSTATED /
partial circularity the synthesis names and then talks past.**

**F17. "NDCG is grounded in real data everywhere."** Verified in code: synthetic users are real
ML-25M trU profiles split known/held; NDCG scored against real held-out ratings. **CONFIRMED.**

**F18. "Internally fair by construction (E7)".** The policy-vs-static contrast structure is fair
(shared world components enter symmetrically; b2 E2-clean). But E7 is a declaration table, not a
check (F13), and the fairness is already violated in the current code by b4's unlabelled privilege
and the b2 construction-budget asymmetry (F15) — the exact "b2 under-built ⇒ hollow wins" risk §3
itself names, present in the defaults with **no b2 ≥ historical-statics verification implemented
anywhere**. Verdict: **OVERSTATED pending the arena-audit fixes.**

**F19. Guard the synthesis misses entirely:** the certification gap between the gated v2.1 model
and the deployed sampler (F14). A policy could learn structure of the hand-set sampler (e.g. the
fixed 0.9-surprise coupling) that neither the gated model nor humans share — a quirk-exploitation
path that the 173 read would only partially catch and the sim-real table would misattribute.

---

## PART 3 — §4 PAPERS CHECK (current tex vs claims)

**F20. Paper A** (`papers/paper1_casper/casper_u_chapter.tex`): `sec:i2fold` "The fold is part of
the instrument" present (line 1453); canary gate class G8 present (line 442); 16/16 suite claim in
text; reference-ladder figure and fold-v3 capstone absent — matching their "ADD" status.
**CONFIRMED.**

**F21. Paper B** (`paper2_casper.tex`): `sec:external` (line 343) and `sec:distill` (line 420)
present; territory>dial content present (lines 428–434); learned-win retraction present at 5+ sites
including abstract ("does not survive training-seed variation"). **BUT: the tex contains ZERO
occurrences of "flutter" or "equat-"** — the claimed "THE FLUTTER + equating (new section —
arguably its headline methods contribution)" **does not exist in the file**. And "the corrected
fuel numbers" are **not** in: the gate table still cites the OLD flutter-inflated **ICC 0.174**
(line 370) that `LLM_DECOMPOSITION.md` supersedes (corrected trait ICCs 0.043/0.037/0.073).
Under §4's "current files" framing, "Now carries" is **WRONG** on these two items — a design
intention stated as fact.

**F22. Paper C** (`paper3_casper.tex`): retitled "The Best Question Has No Name"; fidelity
boundary present as `sec:i2fidelity`; ask-the-gradient absent (correctly an ADD). **CONFIRMED.**

**F23. Paper D** (`paper4_casper.tex`): synthesis is honest that de-rigging is pending; the dead
"two open questions beat eight rating-probes" still stands as a section headline (line 189), and
the rated-ness ceiling / +0.058 premium are not yet in the tex (consistent with "gains" as future).
**CONFIRMED (with the note that D currently still asserts a claim the program has internally
declared dead).**

**F24. Paper E** (`paper5_casper.tex`): unchanged skeleton; merge-into-C is a plan. **CONFIRMED.**

---

## PART 4 — §5–6 THE FINAL-STATE VISION

**F25. "S1 (floor, already secured): publishable at ECIR as A + B regardless."** **OVERSTATED.**
A hostile reviewer would likely accept **A** as near-shippable (committed instrument story, gate
suite, i2fold section in the tex). **B is not secured today**: (i) its claimed headline methods
section (flutter + equating) is unwritten (F21); (ii) its gate table carries a number
(ICC 0.174) the program's own decomposition has since corrected downward by ~4x; (iii) most of the
new answerability evidence (fuel numbers, habitat, contamination, premium) sits on the **unfrozen
173/300 grid whose own artifacts say "re-run on the frozen 300-user grid before any citation"**;
(iv) the flutter magnitude itself awaits the formal repeat-study the decomposition flags as
required; (v) the cross-family LLM check is deferred. "Secured" describes a plan whose inputs are
mostly directional. R4's own 2-week consolidation clock is the honest version of this.

---

## TOP-3 MOST CONSEQUENTIAL CORRECTIONS

1. **The certification chain to the decisive experiment is broken.** "ANSWERER v2.1 … Gates 13/13
   vs corrected targets" is doubly wrong: 13/13 was the iter-2 gate against the OLD
   flutter-inflated targets (the corrected-target gate is 3/3 on 3 statistics, tolerance-passed,
   with the pass "driven primarily by the two calibration fixes"), and the arena/fold-v3 world
   does not even run the gated model — `i25_fold_v3_sampler.py` is a hand-parameterized
   base-rate+surprise approximation of it, of the very structural class the build log shows
   failing G2. Every "certified apparatus" sentence about tonight's arena inherits this gap.

2. **The 173-user read is not "independent", and nearly every §2 number is quoted without its
   DIRECTIONAL/unfrozen banner.** The distilled world was final-refit on all 173, self-equated on
   their grid, and calibration-tuned to reproduce their statistics; the cohort was already multiply
   read before the "quarantine" (which is prospective, process-only, unenforced by any counter);
   and the grid is a partial 173/300 whose artifacts uniformly say "re-run on the frozen 300-user
   grid before any citation" — a banner the synthesis drops on the contamination +0.028, the recall
   premium +0.058 (CI [+0.002,+0.112]), the habitat facts, and the eval-world description itself.

3. **Paper B's claimed content does not exist yet, so S1 is not "already secured".** The tex has no
   flutter/equating section (0 matches) and still prints the superseded ICC 0.174; "A+B publishable
   regardless" is a forecast. Secondary corrections of the same overclaiming type: fold-v3 sold as
   "(all gated)" when its artifact records `ALL GATES PASS: False` (G5, G6, G1-item-cold fails —
   only G5 surfaces in the synthesis, relabelled a "known trade"); "serves all 162k lazily" (G4/162k
   never run); "full 2,428-question universe" (coded universe is 1530, 200 concepts); "value parity
   0.550 vs 0.547" (different targets, different cohorts); "b2 at population scale" (defaults: 50
   users, capped search, and a currently-truncated 10/24 cached sequence).

---

## INDEPENDENT ASSESSMENT

This program's underlying empirical work is unusually honest at the artifact level — pre-registered
verdicts, directional banners, self-audits that catch its own leaks (the contamination study, the
arena code audit, the retractions in STATE) — and the synthesis's *numbers* are almost all real:
of ~20 quantitative claims audited, none was fabricated and most reproduce to the digit. The
systematic failure is one level up, exactly of the predicted kind: the synthesis repeatedly
promotes single-shot, single-seed, synthetic-cohort, unfrozen-grid results into "gated / validated
/ certified / secured" assets, drops every DIRECTIONAL banner, relabels failed gates as "known
trades", describes an unwritten paper section as the paper's current content, and transfers the
v2.1 gate certificate onto a hand-built sampler the gates never saw. Where the program actually
stands: a strong, largely-committed instrument paper (A); a promising but unconsolidated
answerability/measurement paper (B) whose headline evidence needs the frozen 300-user grid, the
flutter repeat-study, and its flutter section written; a genuinely novel flutter/equating finding
resting on a 3-user probe plus one variance decomposition; a fair-fight arena that is well designed
on paper but, per its own blind audit, currently carries at least two verdict-poisoning defects
(privileged b4, truncatable b2) and runs an ungated world model; and an adaptivity thesis that has
so far *never* beaten a fair static on the only judged cohort — so the honest posture is "S1
requires ~2–6 weeks of consolidation work; S2 is live but unproven; the decisive experiment's
apparatus needs its certification gap and audit findings closed before any verdict it produces can
be believed."
