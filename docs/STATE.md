# STATE — where the project is now

> The single current-status doc. **Overwrite as things change.** Pointed to from `MEMORY.md`.
> End-state is in `VISION.md`. Last updated: **2026-07-27 (G0 CERTIFIED; Paper A number audit)**.

## ★★★ G0 PASSED ON THE CERTIFIED TOWER (2026-07-27) + full Paper-A number audit
Full record: `docs/results/PAPER_A_NUMBER_AUDIT.md`. Certified run finished (converged ep13, val flat 4).

- **T2′ certified: TEST full 0.3482 / tail 0.2462 / @100 0.4486** (`ml25m_liang/tower_t2.json`).
  **G0 tie vs the RecVAE bar 0.3540/0.2497: diff −0.0057, CI 0.0069, tie=True.** Identity half holds
  EXACTLY: empty interview → |z|=0.000000, Spearman 1.0000 vs the frozen decoder bias. Best val 0.3510.
  → R1/G0 is no longer open. The paper's abstract, `tab:g0`, `tab:master`, gap map and §5.1 now carry it.
- **C1 bridge closed, incl. a failure:** EASE 0.4203/0.420 PASS · RecVAE 0.4425/0.442 PASS ·
  **Mult-VAE 0.4223/0.426 near-snap (−.004, admitted as such)** · **Mult-DAE 0.3969/0.419 FAIL (−.022) →
  DROPPED from the bank entirely** · iALS 0.358 vs WMF 0.386 → iALS rows read as a FLOOR, not a repro.
- **Paper corrections:** Most-Popular @100 0.1961→**0.1975**; volume-leak R² 0.204→**0.249** (§6.4 and
  §6.5 had been self-inconsistent). All other baseline cells verified against JSON — OK.
- **Broken-fold retraction landed in the paper.** Fig `fig:strategychannel` REMOVED (it asserted "items win
  the long interview" = broken-fold artifact, and still had the flagged design faults). Adaptive-headroom
  magnitudes WITHDRAWN, direction re-grounded on the fold-independent D3 probe.
- **NO concept-vs-item claim in the chapter until the greedy lands.** I briefly put the fixed-fold D1 table
  in as the replacement; author caught that D1 = `item-ask: popularity order` vs `concept-ask: polarization
  order` (`answer_contrast.py:10-11`) — two hand-picked heuristics, one per channel, i.e. the SAME defect
  that made the original suite fragile. Pulled. **Best-static greedy is the agreed criterion** (combined bank
  CONTAINS items-only → combined can only lose by overfitting the build set → combined−items = a direct
  readout of what concepts add). Chapter keeps only the weak claim: both channels fold and both lift off the
  intercept.
- **Gap map clarified:** T2′ sits at x=2.5 vs the target star at 4.0. That 1.5 is THREE HALF-MARKS (R3/R4/R5
  partial), NOT failed gates — unified interface not built, belief layer not fitted, monotonicity not yet
  certified. Star moved 3.75→4.0 and the caption now says this explicitly. New `\REDO{}` macro marks superseded-but-measured numbers
  distinctly from `\TODO{}` (never run).
- **Bib:** all 35 cites resolve, 0 undefined on the final pass; added `good1967principle`,
  `blackwell1953equivalent`, `mu2018allbutthetop` (cited in prose, previously no entry). 9 raw-UTF-8 `Bıyık`
  normalised → file is 100% ASCII. Build clean via Tectonic.
- **⚠ RUN-INTEGRITY CATCH:** the in-flight greedy (PID 6512) was loading the SUPERSEDED
  `cfold_signed_best.pt` (script default) and would have re-derived the broken-fold artifact. Killed at
  23 min (nothing lost), relaunched on `cd_s1_l10_best.pt` as **PID 2652**. Fold now logged (`e745f6f`).

## ★ CURRENT (2026-07-27) — concept operator SETTLED; lever = selection/adaptivity
Full record: memory `concept-fold-distillation-and-downstream-2026-07-27`; `docs/results/
CONCEPT_DISTILL_TRAIN_RESULT.md` + `DISTILL_DOWNSTREAM_RESULT.md`; design `docs/design/
DESIGN_CONCEPT_FOLD_DISTILLATION.md`. Canonical harness, 10k COLD_SEED, intercept 0.1279/0.0192.

- **Concept fold FIXED by the RECIPE, not distillation.** Architecture unchanged (gated fold-to-point on
  FROZEN i25 tower, zero-init gate → G0 item bit-tie). The fix = signed four-band SEL (killed the
  [0.25,1] positive-only clip) + kc/member-drop curriculum (kc 1..32) + NDCG-NLL. Capture 1–6% → **~25%**
  (broad +10 over the tabular floor via composition; fine 34.6 / med 31.4 / overall 25.4% @kc4);
  deployment ~0.144 → ~0.18. **Distillation teacher = NULL** (no-distill control 25.5 / shuffle-canary
  25.9 / distilled 25.0–25.7% — within noise; random teacher = same). One ablation line, not a claim.
  **Deployment fold = λ=1.0** (only config passing the opener gate; S2 distill-then-sharpen craters it).
- **D1 concept-vs-item on the FIXED fold (SUPERSEDES the Jul-25 broken-fold suite below):** concepts BEAT
  items on **tail at EVERY budget** (@10 & @100; k8 tail .121 vs .084, +44%) and **lead full through k=4**
  (k1 .159 vs .137, k4 .197 vs .184); items overtake full only at k=8 (.211 vs .227). Caveat: realizable
  concept-ask vs pop item-ask; items still win FULL at k8. → the "items win the long interview" line in the
  Jul-25 suite was a BROKEN-FOLD artifact; the honest story is now "concepts lead short + dominate tail."
- **D2 answerer panel (fixed fold):** SEL is the answer-ceiling — REPLICATES (imputers tie within ~0.003;
  ExpoMF/PITF worse; not a broken-fold artifact). The answer model was never the lever; **SELECTION is**
  (oracle-select 0.211 vs answer-model ~0.144 @q8).
- **D3 1-level adaptivity tree:** adaptivity PAYS on the ITEM channel (Δfull +0.0015 CI[.0007,.0024],
  Δtail +0.0016 CI[.0008,.0024], significant → refutes static-optimal / HARD RULE #2); concept channel
  positive-not-significant at 1 level (needs the multi-turn policy = Chapter B).
- **Through-line:** operator settled + answer model solved → the prize is SELECTION/ADAPTIVITY (Chapter B),
  cleanly motivated by A. **Paper A remaining: G10 strategy-discrimination; full battery re-verified on the
  certified ckpt; baseline G0 table (Mult-VAE on ruler — DROP DAE, failed snap); efficiency-curve figure;
  write-up.**

## ★ Paper A = INSTRUMENT (scope-corrected 2026-07-27) — audit + star curve + certified run
Full: `docs/PAPER_CONTRIBUTION_AUDIT.md`, `docs/results/GRADED_CURVE_RESULT.md`, memory
`paper-contribution-audit-2026-07-27`. Paper A claims the INSTRUMENT; elicitation = evidence it works on
both channels, NOT a headline. Method pieces framed as PICKED (answer-model panel, distillation = due
diligence) / BUILT-ON (3 legs: graded/star, out-of-catalog concepts, direction tokens; set-encoder; belief
layer) / ARCHIVED (circular geometric answer, Kalman churn, broken clip — take working version). Circularity
is NOT a contribution (internal bug, fixed). Paper B/C + adaptivity out of scope.
- **STAR CURVE ✅ (`premium_vs_k.json`, Fig `fig:gradedcurve`, `docs/results/GRADED_CURVE_RESULT.md`):**
  stars help over binary on the SHORT interview, CI-clean — graded − likes-only (fair positive-only baseline,
  r>3.5) = **+0.0117 @k2 [+.0099,+.0135], +0.0071 @k4**, tie by k8. Mechanism: a signed answer encodes a
  dislike the item-indicator basis can't; edge washes out as collaborative signal fills in. (The "grows to
  +0.12" number vs all-as-like was a mislabeling-control artifact — demoted.) Two-panel provisional figure in
  chapterA_v2, reframed per author.
- **CERTIFIED RETRAIN ✅ DONE** — see the G0 block at the top. `t2final_best.pt`, TEST 0.3482/0.2462,
  G0 tie=True. Log: `experiments/baselines/t2final_train.out`.
- **Repo tidy ✅:** 11 pre-Jul-22 dead-end docs archived to `experiments/_archive/pre_jul22/`; dead scripts +
  cruft removed; result JSONs committed for provenance. Record fix: canonical ruler EASE 0.3476/RecVAE
  0.3540 (0.508/0.523 = retired arena).

## Paper B faithful replication — item-ask bug FIXED, real item-vs-concept table (Jul 26)
The old-Paper-B reconstruction encoder (weak biased-SVD + attention fold-in) on ML-25M Liang had a
**degenerate item-ask arm**: it masked `prof | asked`, removing popular held-out TARGETS from the
ranking → item full cratered 0.1345→0.0860 @q8 (a masking artifact). **Fix** (`excl = prof`, commit
`9a2eab7`): mask only the fold-in profile, matching the concept arm; folding was already fold-in-only.
Item is now **cold-unanswerable** (~2.3/8 answered) and **flat** (pop 0.1268, rand 0.1320 @q8), not a crater.
**Concept − item @q8** (vs pop_item): GEOMETRIC (circular) **+0.0182 full / +0.0142 tail (+64%)** —
reproduces & exceeds the old "+36%" but geom concepts (0.1450) exceed the full-profile ceiling (0.1443),
a circularity tell; BEHAVIORAL (honest SEL) **+0.0054 full / +0.0039 tail (+18%)**, ≈tie-full/+7%-tail vs
the strongest (random) item control. Honest premium is modest, tail-concentrated. Full record:
`experiments/paper2_repl/PB_REPL_RESULT.md` (JSON `pb_results.json`; buggy JSON kept as `..._BUGGY_itemmask.json`).

## The concept channel — DIRECTIONAL VERDICT RECORDED (Jul 25)
> **Partly SUPERSEDED by the Jul-27 CURRENT block above** — the concept-vs-item verdict here ran on the
> BROKEN fold; the fixed-fold D1 replaces the "items win the long interview" line (concepts are now
> tail-dominant throughout + full-lead through k4). The signed-gate / ledger / methods-lesson / OPEN
> items below still stand. Full record: `docs/results/CONCEPT_CHANNEL_RESULT.md`.

Author ruled the concept-channel story **"directionally checks out"**; recorded AS-IS (honest, open
items visible). One shared harness, 10k COLD_SEED, per-question deployment currency; intercept
(0 answers) **0.1279/0.0192**. The signed four-band SEL retrain turned the deployment curve POSITIVE.

**What's DONE:**
- **Strategy × channel suite** (`strategy_channel_suite.json`, fig `strategy_channel_curves_2026-07-25.png`;
  primary rung = signed C-lite): **concepts own the short interview** — polarization-ranked concepts
  beat every item strategy on full AND tail through q4 (q1 .1323/.0348 > items-pop .1307/.0266; q4
  .1433/.0395 > items-HELF .1391/.0384); the single best opener is a polarizing concept question.
  (⚠ the old "items win @q8/q16" reading here is a BROKEN-FOLD artifact — see Jul-27 D1 above.)
  **Adaptive prize large + unclaimed** — privileged oracle dominates every budget (@q16 .2237/.1305);
  realizable greedy closes only 2.1% full / 1.9% tail of the oracle−static gap @q16 (negative @q8)
  → Chapter B motivation.
- **Signed triple gate on the RETRAINED model** (`signed_sel_gate_signed_retrain.json`, SCORED):
  signed beats clip-up **+0.0104 @q16** CI-clean [.0075,.0134] (clip-up sinks below intercept by q8;
  signed holds .1372); popularity-counterfeit reproduces ~0% (@q16 −0.0452, no hard kill → taste, not
  volume×pop); value-permutation collapses the gain (+0.0335 CI-clean → the values carry it).
- **Ledger — signed C-LITE WINS** (`tradeoff_ledger.json`): per-answer concepts m8 sclite .1630/.0639 >
  scfull .1566/.0577; deployment sclite stays above intercept thru q16 (.1379) while **scfull craters to
  .1052 @q16**. **G0 tie holds BOTH**: sclite = frozen ep4 tower (bit-identity, ~0.3536); **scfull TEST
  full 0.3487 / tail 0.2471 vs 0.3540 bar (diff −0.0053, CI 0.0070, tie=True)** → concepts-in-tower cost
  ~nothing on items (coldk8 0.2611 ≥ 0.2584) but buy nothing on concepts.
- **Methods lesson (now in the paper):** the G3 sign-flip capability gate (flip −0.201 CI-clean) PASSED
  on a channel that, as deployed, never emitted a negative answer (clip discarded the negative half);
  only the deployment-currency gate caught it. Capability gates are vacuous if the deployed answer model
  never exercises the capability.

**What's OPEN (recorded honestly, not softened):**
- **Volume-leak gate FAILS.** KT-A3 ridge R²(signed fold latent @q8 → log user-volume) 0.025 → **0.249**
  vs the pre-registered acceptance bar **≤0.05**; the C_NEG negative-channel cap did not hold it. **GATE
  FAIL pending author ruling.** Counterfeit-clean (Arm2 ~0%) attached as waiver-consideration evidence.
  Fallback grid pre-registered: `C_NEG ∈ {1,2,4}` on val, or a volume-orthogonalised fold / invariance penalty.
- **C-lite vs C-full operating point** — C-lite wins the ledger; author call, given the split-G5 gap.
- **G5 split** — trained concept operators learn "what X-likers watch," not "members of X" (sclite
  member-AUC k1 0.630; disc Spearman 0.948 vs log-pop / 0.030 vs member-ness; pop-projection control
  PASSES +0.0087 CI-clean). Fine for recommendation, a gap for the R3 attribute-semantics claim (→ C3).

## Parked / pending
- **QUEUE (Paper A, in order) — what is actually left to run:**
  1. **Greedy best-static** (RUNNING, PID 2652, fixed fold): build 10k val / eval 10k TEST, 500 items /
     500 concepts, no refusals → the replacement `fig:strategychannel`.
  2. **Mult-VAE on the ML-25M ruler** → finishes the G0 baseline table (DAE dropped, snap failed).
  3. **Re-run the interview curves on the certified tower** (`t2final_best.pt`): the graded/star curve, and
     a concept-fold retrain on t2final so the concept channel rides the certified substrate. Everything
     marked `\REDO{}` in the chapter is blocked on this.
  4. **Volume-leak waiver ruling** (R² 0.249 vs the ≤0.05 bar) — author call, or run the pre-registered
     fallback `C_NEG ∈ {1,2,4}` / volume-orthogonalised fold. Fixes the final operating point.
  5. **G10 strategy-discrimination gate** + full battery re-verified on the certified ckpt.
  6. Efficiency-curve figure; Wave-2 belief-corner baselines; film-mute (item-cold) slice.
- S4 filler (ML-20M snap) ✅ DONE: Mult-VAE near-snap (−.004), **Mult-DAE FAILED (−.022), dropped**.
- C-lite gates (committed 70cbe78): flip −0.2012 PASS, wrong-user PASS, dup ×2 PASS / ×3 −0.005
  (slight over-count at ×3 — re-gate on the signed retrain's extended curriculum).
- Strategy ladder (Jul 24): HELF lit-rank-1 replicated CI-clean, 0 inversion flags; bib add flagged
  (golbandi2010 absent from INDEX).
- Belief layer (design (ii)) + Phase B battery built & smoked; belief shakedown fit exists
  (`belief_i25.pt`, sign proof G=2291); Phase B runs after the concept line settles.

## Older context (pre-Jul-24; see git history of this file for the full pre-reset picture)
- Canonical ruler = Liang ML-25M (CLAUDE.md); EDLAE 0.5230/0.3464 = the external bar; ep4 i25 tower
  G0-strength tie (0.3536 vs 0.3540 test).
- Phase A battery PASSED on the ep4 snapshot (G3a/G3b/G9 clean; G6 re-specified PASS; G5 → the arc
  above). Pre-Jul-22 interview-line numbers remain demoted (memory).

## Standing cautions
- Session-managed background tasks are reaped at ~1h — long runs launch via Start-Process
  (session-independent) with split stdout/stderr; watchdog via schtasks.
- The G5 split (member-AUC vs pop-projection) is an open author ruling: trained concept operators
  learn "what X-likers watch", not "members of X" — C3 human round-trip is where stated-attribute
  semantics ultimately get tested.
