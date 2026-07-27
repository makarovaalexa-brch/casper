# I2.5 — the LEARNED FOLD for the RecVAE-d512 instrument (build + gates + Phase-4 rerun)

Date 2026-07-07. Scripts `scripts/i25_lib.py` (shared machinery), `scripts/i25_fold.py` (trainer),
`scripts/i25_gates.py` (gate suite G-fold1..6), `scripts/i25_phase4.py` (E0 rerun under the learned
fold). Checkpoints `.cache/i25_fold.pt` (last), `.cache/i25_fold_best.pt` (best-on-disjoint-val),
`.cache/i25_fold_log.json` (training log). **NO LLM calls** anywhere; canonical scripts/caches
untouched. Implements FABLE_AGENT_DESIGN_2026-07-07.md §"I2.5 — THE PROPER INSTRUMENT".

STATUS: build DONE · gates RUN (5/6 PASS, **G-fold4 FAIL — diagnosed as structural, not a fold
defect**) · Phase 4 **NOT RUN** per the pre-registered any-failure-stops rule.

---

## 1. BUILD NOTES (Phases 1–2)

**Design.** Frozen RecVAE-d512 decoder/scorer (`.cache/instrument2/ml25m_recvae_d512_best.pt`,
certified; test full 0.4998). New module = a permutation-invariant **Deep-Sets residual fold**
`q(z | answer set)`:

- **Token** = (type ∈ {item, concept, attribute}, entity-embedding (512-d), answer-value).
  Per-token input to φ = `[type_onehot(3), value(1), value·emb(512)]`; φ = 2-layer MLP (512);
  sum-pool over tokens; ρ = 2-layer MLP over `[pool(512), native_z(512), log(1+ntok)]`.
- **RESIDUAL form (the load-bearing choice): z = native_z + ρ(...)**, where `native_z` = the frozen
  RecVAE encoder's own encoding of the revealed LIKED items (its native input interface), and ρ's
  output layer is **zero-initialized**. The fold therefore STARTS at the certified native item fold
  (G-fold3/6 pass at init by construction) and training only learns a beneficial delta from the
  full heterogeneous token set. A first non-residual attempt (plain concat MLP) trained but left
  item-only folds far below native (−0.076 @k=4, −0.136 full-profile on a 30-user probe at ep1) —
  the residual form was adopted and retrained from scratch.
- **Entity embeddings:** item = unit decoder row `Wn[j]`; concept = the gate's member-bag RecVAE
  encoding (top-50 by genome relevance, relevance-weighted — identical convention to
  `llm_answerability_gate.concept_dir`); attribute (NEW channel: decade-from-title + genre) =
  member-bag encoding with popularity-weighted members.
- **Answer values — DATA-SIDE ONLY (non-circular from birth):** item = the user's real rating
  centered on the revealed-set mean; concept = genome-relevance-weighted mean centered rating over
  the user's revealed member items; attribute = plain mean centered rating over revealed members
  of that decade/genre. `cos(z*, q)` appears nowhere in training or evaluation of the fold.
  Refusals are EXCLUDED from the fold (they inform answerability, not taste — per the agent design).

**Training.** Reconstruct held-out interactions from SIMULATED partial reveals of real TRAIN-user
profiles (24,803 train / 1,500 disjoint-val users sampled from the ML-25M trU cohort — the study
users live in va/te and are never trained on). Per user: rated items shuffled → known/held-out
halves (mirrors the pinned seed-123 eval split convention); per step, reveal k∈{1..16} known items
(shared-k curriculum per batch, DropoutNet/FTREC practice), tokens = item + top-6 concept + top-6
attribute aggregates of the reveal. Loss = RecVAE's multinomial log-likelihood of held-out liked
items through the FROZEN decoder (log-softmax over the catalogue, known items masked). Adam 1e-3,
batch 256, 14 epochs, deterministic seeds (torch/np seed 0). Held-out targets never appear in inputs.

**Training incident (recorded for provenance).** The first (non-residual) training run survived its
kill signal and silently raced the residual run on the same checkpoint paths: the on-disk "best"
checkpoint ended up holding non-residual ep12 weights while the log described the residual run
(best 0.4629 @ep2, whose weights were clobbered). Caught by cross-checking the checkpoint blob's
internal state/history against the log before trusting any gate numbers. Both raced files preserved
(`.cache/i25_fold_RACED_*`); training re-run cleanly as a single writer; all gates below use the
clean checkpoint. (A first gate probe against the raced checkpoint had shown G-fold1 passing
+0.099/+0.064/+0.072 — discarded, not evidence.)

**Training result (clean run):** see `.cache/i25_fold_log.json` / `_best.pt`; best checkpoint on the
1,500-user disjoint val per the standing rule. Wall ~31 min CPU.

---

## 2. GATE SUITE (Phase 3) — 298 study users, NDCG@10, per-user paired bootstrap (5000)

Fold checkpoint: clean-run best (`.cache/i25_fold_best.pt`, val 0.4629 @ep2). Backing JSON
`experiments/I25_gates.json` (+ `I25_gate4_diag.json`). Cold base NDCG(z=0) = 0.1481.

| gate | requirement | measured | verdict |
|---|---|---|---|
| **G-fold1 item** | 1 true item answer from cold HELPS | **Δ=+0.0949** CI[+0.0695,+0.1209] | **PASS** |
| **G-fold1 concept** | 1 concept answer from cold HELPS | **Δ=+0.0656** CI[+0.0467,+0.0855] | **PASS** |
| **G-fold1 attribute** | 1 attribute answer from cold HELPS | **Δ=+0.0681** CI[+0.0502,+0.0873] | **PASS** |
| **G-fold2** | monotone 1→16 per channel + mixed | item .238→.407; concept .214→.210 (flat); attr .216→.215 (flat); mixed .238→.362 — all within −0.005 band | **PASS** |
| **G-fold3** | item-only ≥ native fold, same items | k=1 +0.003 [−0.006,+0.011]; k=2 **+0.012** [+0.002,+0.022]; k=4 +0.002 [−0.008,+0.012]; k=8 +0.003 [−0.008,+0.013] — never below | **PASS** |
| **G-fold4** | mixed(4i+4c) ≥ best single @budget 8 | mixed 0.3137 vs item-8 0.3483: **Δ=−0.0732** CI[−0.0913,−0.0563] | **FAIL** |
| **G-fold5** | smooth under σ (no cliff) | σ=0/0.35/0.70/1.40 → .3594/.3579/.3559/.3546 (max step −0.0020) | **PASS** |
| **G-fold6** | full-profile fold ≈ native full-profile | learned 0.4647 vs native 0.4787, Δ=−0.0140 CI[−0.0258,−0.0023]; within the declared ±0.02 "≈" tolerance (judgment call, stated) | **PASS** |

**The E0f canary is dead on every channel: one free, TRUE, real-rating answer from cold now HELPS**
(item +0.095, vs −0.053 HURT under the old additive operator). The fidelity-noise response is smooth
and nearly flat — the fold averages noise across tokens instead of amplifying a single η-scaled spike.

### G-fold4 failure — diagnosis (scripts/i25_gate4_diag.py; same rng seed as the gate)

Paired arms on identical shuffled item order: A=item-4, B=item-4+concept-4, C=item-8,
D=item-8+concept-4. n=298.

| contrast | Δ | 95% CI | reading |
|---|---|---|---|
| B−A (4 concepts ADDED to 4 items) | +0.0020 | [−0.0018, +0.0056] | concepts never hurt |
| D−C (4 concepts ADDED to 8 items) | **+0.0076** | [+0.0035, +0.0115] | concepts HELP on top of items |
| C−A (4 more ITEMS instead) | **+0.0366** | [+0.0188, +0.0541] | item marginal ≈ 18× concept marginal |
| B−C (the gate contrast) | −0.0346 | [−0.0520, −0.0173] | displacement, not dilution |

**Verdict on the failure: NOT a fold defect.** There is no dilution anywhere — adding concept
information to any item set helps or is neutral (monotone-in-information, exactly the property E0f
showed the additive operator lacked). The gate fails by DISPLACEMENT: at matched budget a concept
turn displaces an item turn worth ~18× more. The concept channel's data-side aggregate saturates
immediately (G-fold2: 0.214 flat from k=1 to k=16 — the Paper-B rank-≈2–4 saturation reproduced by
the learned fold), while real-rating item answers keep climbing to k=16 (≈27 effective dims,
Paper B). G-fold4 as pre-registered encodes "mixing beats the best single channel", which is
structurally false in an arena whose best single channel is this dominant. The unified fold is doing
its job; the gate discovered an arena fact, not a fold bug.

**Consequence (pre-registered rule honored): Phase 4 was NOT run.** Whether to (a) re-scope G-fold4
to its defect-detecting form ("a mixed set must not lose to its own item subset" — which PASSES:
B−A ≥ 0 ns, D−C > 0 significant) and then run Phase 4, or (b) treat the arena as item-dominated and
rethink, is an owner decision (HANDOFF §9 stop-on-blocker).

---

## 3. PHASE 4 — NOT RUN (pre-registered any-failure-stops rule; G-fold4 failed)

`scripts/i25_phase4.py` is BUILT and ready: static B rebuilt greedily under the learned fold
(+ static+skip with a real turn-refund priority tail), descent k=3/4/5 (realizable |centered-rating|
ranking + p̂-surrogate ranking + a labelled per-user target-peek UPPER BOUND), emergent switch
(marginal-info × true-table gate × divisiveness), all-item-8; T=8, refusals cost a turn, belief
z_t = fold(all answered tokens so far), concepts fold through the learned encoder as data-side
aggregate tokens, NDCG@10 (+@50 for headline arms), n≈298, paired per-user bootstrap, verdict rule
"descent wins iff anytime Δ≥0.010 AND CI excludes 0". Not executed.

What the gate numbers already imply for the eventual Phase 4: with a healthy fold, 8 rated-item
answers from cold reach ≈0.35 while the concept channel plateaus at ≈0.21 — so under a valid
instrument the live question flips from "does descent beat the static" to "how early should the
interview reach rated-item elicitation (open recall)". The E0d/E0f "descent loses" verdicts are
hereby scoped to the broken fold. Third arena point for the flagship's methods spine:
geometric+additive → real+additive (fails the canary) → real+learned (canary passes). THE FOLD
DECIDES.

---

## ASSUMPTIONS / judgment calls (complete list)

1. **Deep-Sets, not Set-Transformer.** The V2-ST port was judged unnecessary for v1: the spec allows
   "Deep-Sets sum-pool MLP first (simpler is fine for v1)", and CPU-only training makes attention
   over 16 tokens × 512-d needlessly slow. The residual-around-native design is the capability
   guarantee the gates actually need.
2. **Residual fold with native-liked-items prior input.** `native_z` uses only revealed LIKED
   (r≥4) items, matching RecVAE's implicit-feedback input convention; disliked revealed items enter
   through their tokens only.
3. **Rating centering** = mean over the REVEALED set (what an interviewer can observe), not the full
   known profile — deployable by construction. Gate scripts center on the full known-half mean where
   the full profile is the revealed set (equivalent there).
4. **Attribute channel realization**: decade parsed from the title year (`answerability_main_study.item_year`)
   + the 20 ML genres. Actors/directors are NOT in ML-25M metadata (would need external joins from
   links.csv → IMDb/TMDb) — decade+genre is the in-dataset realization; stated as a scope limit.
5. **Concept aggregate uses genome relevance weights over revealed members** — data-side (the genome
   is survey/user-sourced, not instrument-derived).
6. **Curriculum k shared per batch** (not per user) for pack efficiency; reveal lengths 1–16.
7. **Training cohort** = 25k sampled trU users (not all 161k) — compute-bounded; stated, not hidden.
8. **G-fold1 item pick** = a random KNOWN LIKED item (rng seed 1) — "a genuinely answerable,
   informative single answer"; the E0f control's convention (random answerable+RATED) restricted to
   likes because a single centered-rating token for a like carries the same sign information the
   native interface consumes. G-fold2/3/5 use random KNOWN items of any rating (dislikes included as
   tokens).
9. **G-fold2 "within noise"** = non-decreasing up to a −0.005 band per step (the concept/attribute
   channels are flat-saturating; a strict ≥ would fail on ±0.002 jitter that the paired CIs show is
   noise).
10. **G-fold3 verdict rule** = learned never significantly BELOW native (CI upper ≥ −0.003); at k=2
    the learned fold is significantly ABOVE native (+0.012), because the rating VALUE (like strength)
    is information the binary native interface discards.
11. **G-fold6 "≈" tolerance** = |Δ| ≤ 0.02 (my declaration). Measured Δ=−0.0140 CI[−0.0258,−0.0023]:
    the learned full-profile fold gives up a small but significant 1.4pt to the native fold. Judged
    acceptable for a v1 ruler-consistency check (the fold is meant for partial interviews, not full
    profiles; anyone needing full-profile scores uses the native encoder). Flagged, not hidden.
12. **G-fold5 σ applied to STAR ratings before centering** (per E0F assumption 2), fresh rng per σ
    level; deterministic.
13. **Gate G-fold4's mixed composition** = k/2 items + k/2 concepts with items first (the natural
    coarse→fine order is concepts first, but token order is irrelevant to a permutation-invariant
    fold).
14. **Training incident** (§1): raced checkpoints preserved as `.cache/i25_fold_RACED_*`; the clean
    rerun reproduced the residual run's log exactly (deterministic seeds), confirming provenance.
15. **Phase-4 harness choices (built, unrun):** static+skip realized as "first T answerable entries
    of schedule++value-ranked-tail" (the true-table refund form); descent phase-2 item pool = the
    user's known-rated items (E0F assumption 4); emergent div = qᵀCov(W)q over decoder rows,
    precomputed per candidate. These matter only if Phase 4 is later unblocked.
