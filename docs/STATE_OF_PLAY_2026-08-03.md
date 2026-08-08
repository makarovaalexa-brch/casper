# STATE OF PLAY — 2026-08-03 (written for a cold start; assume the reader has no memory of the session)

Everything Paper A needs, plus the concept-channel situation in full. Numbers here are measured, with the
protocol stated, because the single most expensive recurring error on this project has been comparing
numbers from different information contracts.

---

## 1. THE TOWER IS DONE — do not retrain it

**`models/i26c_ep31/t2i26c_ep31.pt`**, sha256 `cdb74693616cb75f5e3a1796bc35054124b924c81be3fb7e7398358aa73062a0`,
committed to git with a provenance README. Pushed to origin.

Provenance is ONE hop: RecVAE (frozen decoder, hidden 600 / latent 200, own val 0.3512) → i26c.
`t2final_best.pt` (the old i25 tower) is never read.

**Training command of record:**
```
python -u src/instrument/train_i26.py --init recvae --mix 0.25,0.30,0.35,0.05,0.05 \
       --epochs 60 --tag t2i26c --steps_per_epoch 800
```
batch 128, warm_lr_scale 0.1 (WARM taste path 3e-5, NEW exposure branch + z0 3e-4). Trained to a
pre-declared plateau (6 epochs with no new val max), reached at 35 epochs. ep31 selected by the
pre-registered rule `argmax(sel + val_full)` on VAL only.

**TEST numbers, canonical Liang ML-25M ruler (140,768 train / 10k val / 10k test, 18,359 items):**

| | full@10 | tail@10 |
|---|---|---|
| RecVAE (anchor) | 0.3540 | 0.2497 |
| **i26c ep31** | **0.3496** | **0.2491** |
| i25 certified (old tower) | 0.3482 | 0.2462 |
| i26 run-A ep20 (2-hop) | 0.3416 | 0.2455 |

Paired bootstrap vs RecVAE, same 10k users, ARM A:
- full −0.0043, CI [−0.0059, −0.0028] → **separated, 1.2% relative**
- tail −0.0006, CI [−0.0020, +0.0008] → **not statistically separated**. NEVER write "tie" or "parity";
  the Jul-30 scar was exactly that.

Interview, k=8, entropy0, NO FLOOR: **0.2053 full / 0.0792 tail**.

**Empty-set behaviour:** at zero evidence the tower reproduces the popularity ranking (Spearman 0.96,
9/10 top-10 shared, 0.1615 vs popularity's 0.1626). The prior is LEARNED from the k=0 curriculum bucket —
this is why the hand-coded per-item bias was deleted and why the evaluation floor was dropped (§4).

---

## 2. PAPER A — WHAT IS DONE AND WHAT REMAINS

### 2.1 The belief layer is HALF-deleted from Paper A (`31c9d49` did the SUBSECTION only)

**CORRECTION 2026-08-03: this section previously read "DELETED (done)". That was wrong.** `31c9d49`
removed 121 lines -- the dedicated belief subsection and the mechanism claim -- but the REQUIREMENT STRIP
is untouched. Verified in the current file: **R1--R5 are still five requirements** (never renumbered to
four), V1--V5 still exist, `Sigma` appears 10 times, "belief" 53 times, the V3 covariance row is still
in the verification table, the "belief corner" framing is still in the gap analysis, and the abstract-level
contribution list at l.160 still advertises a "conjugate-Gaussian precision-accumulator belief layer".
Task #75 is the REST of this job, and it is larger than one table column.

Author ruling 2026-08-02. Not "reported as a negative result" — removed, because papers present the
working version. Justified by code, not argument: `i26_encoder.forward` folds by SUM-POOL over tokens
with a trainable no-evidence latent `z0`, and no scoring path anywhere touches a covariance. R2's
any-length order-invariant gradient-free fold is delivered by the set encoder, not by any accumulator.

Also: the evidence against the belief tested the WRONG USE. Variance-weighted decoding hurting from k=4
is uncertainty applied to RANKING, which was never proposed. The proposal is uncertainty for SELECTION →
Paper B (#76). Paper A makes no claim about it, so carries no negative result about it.

**Requirements were SUPPOSED to become FOUR — NOT DONE IN THE FILE.** The plan below is correct and
agreed; it has not been applied. The chapter still carries R1--R5 and V1--V5. Treat this as the spec for #75:
- R1 full-profile parity · R2 open any-length fold · R3 monotonicity ("does not lower beyond noise")
  · R4 sign and intensity
- V battery: V1 parity · V2 concept-fold specificity · V3 monotonicity · V4 sign/intensity. All
  "V1--V5" ranges became "V1--V4". I1/I2 keep numbers; I1's "the belief responds to information"
  became "the fold responds to information".
- `eq:obs` re-homed: the observation pair (φ,y) ∈ ℝᵈ×ℝ now defines itself in the channel-agnostic
  tokens section with no Gaussian framing.
- The uniformity argument survives, recast: a closed-form accumulator ranked by its mean reaches
  0.2019/0.1200 vs the learned fold's 0.3496/0.2491, so uniformity costs ~0.15 full NDCG. It now argues
  FOR the learned set encoder instead of defending a belief.
- Coarse-to-fine kept WITHOUT the covariance: a concept direction aligns with a large region of the
  frozen decoder's item cloud, so folding it moves many catalogue scores at once. Drop the 11.5× number.

**R3 monotonicity evidence (ep31, entropy0, floored):** k=1..16 full 0.1658 / 0.1713 / 0.1826 / 0.2047 /
0.2300; tail 0.0399 / 0.0475 / 0.0577 / 0.0770 / 0.0915. Also helf 0.1624→0.2138, popularity
0.1622→0.2221. k0→k1 passes every epoch in training. No posterior required.

**Chapter compiles clean under Tectonic** (after `\REDO` was finally DEFINED on 2026-08-03 -- it had been
used in three places but never declared, so any build touching those lines failed).

### 2.2 STILL TO DO on the chapter (#75)

1. **The R3 column in the 23-system taxonomy table** — remove it, and the "belief corner" section header.
2. **CAPTION OBLIGATION**: state plainly that ConTS and UNICORN's designed capability is
   uncertainty-native modelling, which the four surviving axes do not score. Re-scoring audit is DONE:
   dropping R3, **BCIE does NOT flip** (it holds old-R4 independently); only **ConTS and UNICORN** lose
   every mark. The bifurcation survives at four columns. But we are deleting the axis those two rivals
   won on immediately after our own version failed — say so rather than let it look like gerrymandering.
3. Remaining scattered belief mentions in the gap analysis and limitations.
4. **KNOWN TRAP**: `belief_mf` stays as a BASELINE RECOMMENDER in the interview table
   (Bıyık-belief + ConTS-mean). Do not let a global search-and-replace delete the rival's row.

### 2.3 NUMBER AUDIT (#73) — TOWER NUMBERS DONE 2026-08-03 (`5a11fe6`), CONCEPT NUMBERS STILL STALE

DONE: 0.3482/0.2462 → 0.3496/0.2491 in prose, taxonomy table, G0 table, frontier figure and the V1 gate
row; the CLAIM restated (tail parity, full separated at −0.0043, 1.2% relative — "ties the bar" retired);
the unpaired `1.96*sqrt(2)*SE` signature purged from all three places that cited 0.0069/0.0070 as paired
(true paired half-widths ±0.0016 full / ±0.0014 tail).

STILL STALE — the concept-channel section is entirely old-tower (ep4 / i25-era) numbers:
`0.1630/0.0639`, `0.1566/0.0577`, `0.1379` at q16, the `0.1052` crater. Gated on the cv5 concept run.
Three values are marked `\REDO{}` in the file: NDCG@100 on the certified checkpoint (never measured),
the graded-vs-binarised arm, and the tokens-in-tower V1 arm (both superseded tower + unpaired interval).

### 2.4 THE NEW FRAMING (#73) — the chapter's spine

"The ranking of standard elicitation heuristics INVERTS depending on which recommender consumes the
answers; therefore elicitation comparisons need a strong, documented instrument."

**Evidence, k=8, UNIFORM NO FLOOR (the primary contract, §4), 10k test users:**

FULL NDCG@10:
| recommender | helf | entropy0 | popularity | order |
|---|---|---|---|---|
| ours (ep31) | .1892 | **.2053** | .1943 | E0 > P > H |
| golbandi_leaf | **.1948** | .1686 | .1716 | **H > P > E0** |
| rbmf | .0729 | **.1644** | .1637 | E0 > P > H |
| belief_mf | .1623 | **.1634** | .1632 | E0 > P > H |
| mostpop | .1626 | | | |

TAIL NDCG@10:
| recommender | helf | entropy0 | popularity | order |
|---|---|---|---|---|
| ours (ep31) | .0725 | **.0792** | .0686 | E0 > H > P |
| golbandi_leaf | **.0792** | .0677 | .0696 | **H > P > E0** |
| rbmf | **.0396** | .0369 | .0344 | H > E0 > P |
| belief_mf | **.0331** | .0300 | .0300 | H > E0 > P |

Rashid's published order is helf > entropy0 > popularity. **Golbandi — the node model closest to what
these heuristics were validated on — is the only instrument that reproduces it.** Our best cell beats his
best by **+0.0105 on full** and **TIES exactly on tail (0.0792 both)**. The inversion holds under BOTH
floor contracts and on BOTH metrics.

**What we may NOT claim** (checked, keep the paper honest):
- NOT "helf is overrated" — on tail helf is first for three of four baselines.
- NOT a mechanism claim. "helf collapses because the model can't use non-answers" is REFUTED by our own
  controlled contrast: i25 (no unknown channel) scores .1890 on helf, i26 (with one) .1872, identical
  contract. Adding the capability made helf slightly worse.
- NOT "ours is the correct instrument" — we are 1.2% below RecVAE on full.
- The floor is not neutral in either direction; say so.
- "Beats Golbandi" is FULL-ONLY. Tail is identical. Run a paired CI on the 0.0105 before claiming it.

---

## 3. THE CONCEPT CHANNEL — FULL STATE (this is the live problem)

### 3.1 Architecture is settled: C-LITE (bolt-on over the frozen tower)

`ConceptFoldNet` (in `concept_fold.py`) takes the frozen tower's latent z plus concept answers and emits
a gated delta. The tower never sees a concept.

**The native alternative is DEAD ON EVIDENCE, not theory.** It was built (`arm C-full`,
`--concept_tokens`, default OFF, commit `5a6b582`) with the same bit-identical-item-only property, and
evaluated head-to-head in `docs/results/CONCEPT_CHANNEL_RESULT.md` §4:

| rung | q2 | q4 | q8 | q16 |
|---|---|---|---|---|
| **sclite (bolt-on)** | .1320/.0298 | .1345/.0257 | .1417/.0334 | **.1379/.0324** |
| scfull (native tokens) | .1351/.0295 | .1330/.0293 | .1315/.0342 | **.1052/.0259** CRATERS |

Native is item-safe (its G0 was 0.3487/0.2471) but craters BELOW the 0.1279 intercept by q16.
**Recorded verdict: "signed C-LITE WINS."** Author has additionally ruled: ep31 only, no new tower.

Two explanations I invented and had to retract, recorded so nobody re-invents them: "native fixes the
vocabulary" and "R2 forbids native" — BOTH WRONG. The centroid is only an INIT and the table is trained;
and nothing structural stops a native path folding an arbitrary vector. The real reason is the ledger.

### 3.2 What was rebuilt, and why

`src/instrument/concept_curriculum_v2.py` (v3 mixture) + `src/instrument/concept_train_v2.py`.
Committed. Trained as tag `cv3`, 12 epochs, on frozen ep31, 139,268 train users.

**Defects found in the OLD curriculum by reading the code (two of my four suspicions were WRONG —
recorded so nobody re-fixes a non-problem):**
- kc was ALREADY random U{1..16} — NOT a defect
- selection was ALREADY half-polarised / half-uniform — NOT a defect
- item context ALREADY existed (p_conc_only=0.35, so 65% had items) — NOT a defect
- **D1 REAL, THE MAIN ONE**: `build_cache()` drew ONE example per user under a fixed seed, cached the
  latents, and every epoch looped that frozen set. One draw per user for the whole run.
- **D3 REAL**: `cand = flatnonzero(ans & (B != BAND_REFUSE))` filtered refusal-band concepts OUT of the
  candidate pool, so the channel never saw "I don't know this concept".
- D4 REAL but mild: no rare-concept weighting.

**MEASURED ANSWERABILITY (3000 users × 1031 genome concepts with ≥30 members) — the number that drove
the redesign:**
```
median user can answer 573 of 1031 (55.6%); p10 335, p90 885
niche (bottom 25% by member count): answerable by 26.0% of users
small 45.7% | large 67.5% | broad (top 25%) 90.3%
ASK UNIFORMLY -> 57.4% answered, 42.6% REFUSAL
ASK 50 BROADEST -> 97.3% answered (2.7% refusal)
ASK 50 NICHEST  -> 16.3% answered (84% refusal)
member counts: min 30, median 291, max 13746
```
So refusal rate is a FUNCTION OF THE ASK POLICY spanning 3–84%. The old curriculum saw 2.9%.

**v3 curriculum (realised share in brackets, verified in training logs):**
- **55% INTERVIEW** [0.55] — kc ∈ {1,2,4,8,16,32} drawn UNIFORMLY over the geometric grid (= log-uniform;
  a 1/b weighting had put 50% of interviews at kc=1). Asks drawn from **ALL 1031 concepts** via a policy
  sampled per example from {broad, uniform, niche}; **answerability decided by the user's history**, so
  refusals emerge. Realised refusal share **0.029 → 0.45**.
- **30% DENSE** [0.30] — kc ~ U{1..8}, all answerable, **half rare-weighted / half uniform** (all-rare
  confounded rarity with profile size, since niche concepts are answerable mainly by heavy raters).
- **5% kc=0** [0.05] · **10% kc=1** [0.10].
- **BROAD bucket DELETED** — it revealed 50–100% of answerable concepts, but there is no warm concept
  user (nobody answers 573 concept questions), so it trained a state that never occurs.
- Rare weighting `clip(member_count, 30, 1000)^-0.5`; unclipped gave broad concepts 0.8% of picks and
  broad concepts are what an interview asks FIRST.
- Item context k ~ U{0..8} alongside.
- **KL/distillation DELETED** — lam>0 never beat lam=0 at equal budget; the teacher was NULL. This is
  what makes fresh draws affordable (no teacher fold to precompute).
- `ConceptFoldNet(use_refusal=True)`: a refusal and a genuine MEH both carry v=0.0, so refusal now has
  its own input dimension. Verified distinguishable (0.0202) and still a bit-exact no-op at init.

### 3.3 RESULTS — cold check (concepts folded CLEAN, zero items)

`src/instrument/concept_cold_check.py`, 1500 held-out train users (canonical val/test quarantined),
broad-first ask policy, real refusals, NDCG@10.

**CANARY PASSED**: with no net, spread across budgets = 0.00e+00, and m0 = **0.1267**, which matches the
historical **0.1279** intercept — so this protocol lines up with the one the old numbers used.

**SELECTED ep7** (rule: max mean NDCG@10 over m∈{1,4,8} subject to every m ≥ m0):

**RESOLVED 2026-08-03 — the "tail collapse" was A METRIC BUG IN THIS SCRIPT, not a channel property.**
The first version of `concept_cold_check.ndcg_at` scored tail by ranking ALL 18,359 items and counting
tail targets surviving in the unrestricted top-10. Canonical CASPER tail
(`run_battery_phaseA.ndcg10_from_scores`, line 114) instead **deletes head items from the candidate
list** (`S[:, head_mask] = -inf`) and ranks tail-only. Under the broken version the no-evidence prior
scores **exactly 0.0000** by construction — the 10 most popular titles fill every slot — so the whole
column was pinned near zero regardless of channel quality. Fixed; `ndcg_at`'s docstring now carries the
warning. **Every tail number below is the canonical definition.**

Matched protocol, each net on ITS OWN tower, broad policy, same 1500 users:

| budget | new ep7 full | old sclite full | **new tail** | **old tail** |
|---|---|---|---|---|
| m=0 (floor) | 0.1267 | 0.1255 | 0.0268 | 0.0213 |
| m=1 | 0.1294 | 0.1246 *(craters)* | 0.0324 | 0.0320 |
| m=2 | 0.1342 | 0.1281 | 0.0389 | 0.0347 |
| m=4 | 0.1369 | 0.1293 | 0.0485 | 0.0369 |
| m=8 | 0.1425 | 0.1317 | 0.0629 | 0.0438 |
| m=16 | **0.1489** | 0.1365 | **0.0714** | 0.0455 |

**Lift over each tower's own floor at m=16: full +0.0222 vs +0.0110 (2.0x); tail +0.0446 vs +0.0242
(1.8x).** The lift columns are the fair comparison — the two towers differ, so the absolute columns are
not strictly like-for-like.

**VERDICT: the v3 rebuild is a clean win on BOTH axes at every budget.** Monotone throughout; the old
channel craters at m=1 (.1246 below its own .1255 floor) and its published deployment curve fell back by
q16 (.1417 → .1379) where the new one keeps climbing.

**DO NOT re-derive the old .0298/.0257/.0334 figures from `CONCEPT_CHANNEL_RESULT.md` §4 and compare
them to this table** — those came from a *selected* fixed bank on the ep4 tower, a different information
contract. The matched-protocol old-channel numbers are the `old tail` column above.

**METHODS LESSON, and it is the same one for the fourth time.** Three separate false stories were
built on this one broken column before the metric was checked: (1) "tail collapsed 30x"; (2) "genome
centroids are head-aligned, so it's a scope condition on the whole concept line"; (3) "the old number
came from a greedy-selected bank, so concepts only buy tail under selection". All three were plausible,
all three cited real prior findings, and all three were wrong. **When a number is pinned at exactly
0.0000, suspect the metric before the model** — an exact zero is a structural signature, not a
measurement. Related: [[surprise-means-debug-the-harness]].

**Optional, no longer diagnostic** (`--policy uniform` / `--policy niche`): still worth one run to see
whether niche asks buy more tail than broad ones, but that is now an ask-policy question for Paper B's
selector, NOT evidence of a defect.

**ALSO UNRESOLVED — a contradiction in our own record that I have been quoting selectively:**
`254587d` says the concept-tail advantage REVERSED on ML-25M; the Jul-27 note says concepts win tail at
every budget on the corrected fold. These are opposite. Whichever holds on ep31 must be settled by the
concept-vs-item head-to-head, not by citation.

### 3.4 The capture metric is RETIRED as a gate

Capture = (student − intercept)/(teacher − intercept) where the teacher folds the user's ACTUAL RATINGS
on the concept's member films. The teacher's input strictly dominates the student's (dozens of per-item
ratings vs one signed scalar), so the ratio's ceiling is unknown and <1 by construction — it measures an
information gap, not channel quality. Confirming evidence: the same teacher was NULL as a training
target. **A denominator that fails as a training target has no business being a gate.**

**Replacement gates**: (1) PRIMARY — concept-vs-item at matched question budget (answers "should the
interviewer spend a question on a concept?"); (2) FLOOR — deployed lift over the 0.1267/0.1279 intercept;
(3) HARD REJECT — specificity triad preserved: like 0.626 / dislike 0.416 / unrelated 0.498,
SIGN ≥ +0.20, IDENTITY ≥ +0.12 (currently +0.2096 / +0.1278). **The specificity triad has NOT yet been
re-run on the new channel.** If ever a ratio is wanted, the honest denominator is a matched-information
oracle: the member-fold delta PROJECTED ONTO the concept direction.

### 3.5 Known limitation to state in the write-up

Our answerability rule ("user has ≥2 rated member films") is simulated from the SAME history that scores
the user. It is SEL-family and defensible, and it is the same family as the rest of the answer model, but
it must be stated so refusal behaviour does not silently become a circular signal.

Also: refusals can only come from concepts in the user's pool — we never ask about a concept the user has
never encountered at all. Reaching that case means changing the answerability model, which touches the
verified SEL answer.

---

## 4. PROTOCOL DECISIONS IN FORCE

**NO FLOOR is the primary interview contract** (author ruling 2026-08-02, supersedes the 2026-07-30 floor
ruling). The floor existed because a model told nothing must not score below popularity; i26 LEARNS that
prior, so the floor swapped our prediction for a near-identical one AND suppressed us: no-floor minus
floored at k=8 was +0.0031 full / +0.0093 tail on helf, positive on every informative arm. Report floored
as robustness. Honest caveat both ways: the floor penalises models that read unknowns (golbandi
0.1948 → 0.1899); no-floor penalises models with no prior (rbmf falls to 0.0013 on pure_entropy because
mu0 = 0 by construction, while belief_mf — the same model WITH an empirical prior — does not).

**PAIRED bootstrap always.** The chapter's old "CI 0.0069" was `1.96*sqrt(2)*SE`, an unpaired form that
overstated the interval ~4.6×. Both models score the SAME 10k users.

**Numbers from different information contracts are NOT ordered.** Arm-A masking vs arm-N pool vs the cold
concept protocol give different levels. This has cost real time three separate times (Golbandi 0.3894;
"i26 empty set 0.1312 vs popularity 0.1626"; the NDCG@100-vs-@10 crater gate).

---

## 5. BASELINES

- **EDDI**: PAUSED at epoch 44 of 200, checkpoint `.cache/baselines/eddi_ml25m_liang.pt`, resumes with
  the same command. Best so far test full@10 0.2153 / tail 0.1077 (val 0.2078). Author ruling: TRAIN IT
  and report best-effort with the caveat that it has no published recsys anchor (its paper is UCI, MNIST and healthcare tabular data)
  — same status RBMF, TaNP and SASRec already carry. It was still improving at the wall.
- **TaNP**: trained, checkpoint exists (`tanp_ml25m_liang.pt`), full-profile row 0.2645/0.1635.
  NOT yet in the interview table — needs an answer-contract adapter only, no training (#72).
- Interview table rows already measured on ep31: golbandi_leaf, rbmf, belief_mf.

---

## 6. TODO, ORDERED

**PAPER A — critical path, no compute needed for 1–4:**
1. **#75** finish the belief strip: R3 column in the taxonomy table, "belief corner" header, gap
   analysis, limitations, + the ConTS/UNICORN caption obligation.
2. **#73** number audit — 0.3482/0.2462 → 0.3496/0.2491 everywhere, every table and figure.
3. **#73** write the instrument-dependence framing section (spine; table in §2.4 above).
4. **#66** R5→R4 sign scoping; retitle Definition 1; requantify every "parity"/"tie" to "not
   statistically separated".
5. **#74** concept channel: run the niche/uniform policy check (§3.3), re-run the specificity triad,
   then the concept-vs-item head-to-head at matched budget with paired CIs.
6. **#72** TaNP into the interview table (adapter only). Resume EDDI.
7. **#69** bib debt: 9 owed citations; split `luo2019deepcritiquing`, which conflates DLC (Wu et al.
   RecSys'19) with CE-VAE (Luo et al. SIGIR'20) and would misattribute in print; fix the dead DLC repo
   pointer; confirm `li2022self`'s venue; mark 3 unquotable cells (BK-VAE vocab size, BK-VAE rec claim,
   all DLC sentences — code-verified only, ACM 403s).

**PAPER B — after A closes:** #76 belief for SELECTION (fix sigma tracking popularity first) · #68
organic open request (email Lukas Eberhard, TU Graz, for submission texts — highest value per effort) ·
#70 re-run the Most-Popular gate on that ruler ourselves · #56 · #52 · #53.

**STALE, delete from the list:** #55 and several completed-but-open items superseded by the retrain.

---

## 7. FILES THAT MATTER

```
models/i26c_ep31/t2i26c_ep31.pt        the tower (committed, pushed)
models/i26c_ep31/README.md             provenance, exact command, curriculum, selection rule
src/instrument/train_i26.py            tower trainer (--init recvae, --mix, --warm_lr_scale)
src/instrument/select_i26c.py          pre-registered tower selection (max sel + val_full)
src/instrument/morning_i26c.sh         the full go/no-go battery in one command
src/instrument/i26c_watchdog.ps1       keep-alive; asks the PROCESS TABLE before relaunching
src/instrument/concept_curriculum_v2.py  v3 concept curriculum (fresh draws, refusals, rare weighting)
src/instrument/concept_train_v2.py     concept trainer, pure NLL, no KL
src/instrument/concept_cold_check.py   COLD scoring (zero items) + canary
.cache/instrument/cv3_ep01..12.pt      the 12 concept checkpoints; ep7 selected
experiments/instrument/cv3_coldcheck.out   the cold results in §3.3
docs/results/CONCEPT_CHANNEL_RESULT.md     the sclite-vs-scfull ledger (§3.1)
new_chapters/chapterA_v2/chapterA_v2.tex   compiles clean, belief removed, numbers STALE
```

**Watchdog lesson worth keeping**: restarting by hand means kill → delete state file → relaunch, and in
that window the watchdog saw no state file, concluded "dead", and launched a SECOND trainer. Two ran for
ten hours on the same checkpoint files. The fix is to ask the OS process table, not the state file.

---

# NIGHT OF 2026-08-03/04 — what changed, and one retraction

## Certified artefacts (both frozen, both in models/i26c_ep31/)
- **Tower** `t2i26c_ep31.pt` — TEST 0.3496/0.2491. Unchanged.
- **Concept channel** `concept_cv3_ep08.pt` (= `.cache/instrument/cv3_ep08.pt`) — **REVERTED TO cv3 on
  2026-08-04**, author ruling: prefer the simpler architecture where results are indistinguishable. 464k
  params (hidden 256, keep_meh False) vs cv5's 1.24M. Re-selected under the FIXED ranked bank, where it
  is BETTER, not merely equal: SEL 0.1411 vs 0.1381, m16 .1498/.0673 vs .1467/.0705. Polar CIs all ten
  cells separated, full margins larger than cv5's at every budget (peak m8 +.0189 / +.0310).
  NOTE the earlier cv3 pick of ep7 was made under the random-draw bug; ep8 is the fixed-bank selection.

## RETRACTION — cv5 does NOT beat cv3
Reported at ~23:40 as "+0.0148 full / +0.0083 tail at m=16". **Wrong: contract mixing.**
`concept_cold_check --bank expinfo` had no `expinfo` branch and fell through to `crank=None` (RANDOM
draws) while still logging `bank=expinfo`. cv5 was scored on random draws, cv3 on a ranked bank.

Matched (`bank_sweep`, both checkpoints, four banks): **cv5 ≈ cv3.** Every delta inside the ±.003–.004
noise band at n=1500; several negative under `pop` and `entropy`; consistently positive but tiny under
`polar`/`expinfo`. **The whole config change bought nothing measurable.** Capacity was not binding — a 4×
wider net trained 2.5× longer moved nothing, so the flat-NLL reading was right all along. Remaining
headroom is in the QUESTION BANK and the ANSWER CHANNEL, not the network. Full table: `docs/results/
CONCEPT_VS_ITEM_BANK_SWEEP.md`. Fix + assert in `concept_cold_check.py` (every named bank must resolve to
a ranking; only `random` may be unranked).

## cv5 ep24, certified concept numbers (`expinfo` bank, floor .1277/.0267)
m1 .1323/.0422 · m2 .1351/.0455 · m4 .1372/.0508 · m8 .1449/.0634 · m16 .1467/.0705

## Other movements
- **EDDI measured**: 0.2153/0.1077, @100 0.3110 (early stop ep46, val-selected ep31). Both chapter rows
  updated — it is the R2 architecture and it lands far below the R1 bar, which strengthens the gap claim.
- **G6 gate was mis-specified**, now fixed and PASSING: it demanded the wrong-user control land NEAR the
  intercept, but the chapter's own I1 text says a stranger's answers are *worse* than no information.
  One-sided now (`PASS_no_gain`: CI upper bound below zero). Measured −0.0980 [−0.1016, −0.0944].
- **G5 was certifying dead code** — a scaled genome centroid added straight to z, the July operator,
  not the trained fold. Repointing it at the trained fold produced k0=0.0040, which is a BASE-LATENT
  MISMATCH in my harness (G5's k0 path uses the returned matrix AS the latent; ConceptFoldNet needs the
  tower's learned empty-fold latent as its running vector). Documented, not shipped. **V2 should cite the
  cold check**, which measures the same quantity correctly.
- **V4 row refreshed** on the certified tower; all arms still pass. Neutralisation endpoint is higher
  (0.1714 vs 0.1321) and the staircase lower (ρ 0.794/0.818 vs a single 0.927 previously quoted — the
  gate computes two coefficients and the chapter was quoting one that matched neither).
- **Intensity premium at full profile**: +0.0039 full (paired CI [+0.0025,+0.0053]) / +0.0020 tail
  ([+0.0007,+0.0031]), both separated. Replaced the old claim that intensity was worth ~nothing there.

## STILL OPEN
1. **Greedy best-static table** is the last old-tower result (its 0.1279/0.0192 intercept gives it away).
   568-min run, NOT launched: `strategy_channel_suite.py` builds ConceptFoldNet with `len(blob["tags"])`
   and `h=blob["hidden"]`, passing neither `conc_init` nor `use_refusal` — and cv5's blob has **no `tags`
   key**. Fix both call sites (~540, ~579), smoke, then launch.
2. **TaNP into the interview table** needs a real ADAPTER (a per-user adaptation step consuming k
   answers), not a registry line. `RECS` is at `run_interview_table.py:85`. No training required.
3. Paired CIs on concept-vs-item deltas — running as `concept_vs_item_paired.py`.
4. Rendered-PDF read-through. **Build-clean is not verification**: tonight alone a bibliography that
   silently emitted nothing, four false "five requirements" phrases, and a mislabelled bank all compiled
   without a warning.

## Rendered read-through — what it found (night of Aug 3/4)

Pages read as RENDERED: 1, 2, 3, 4, 6, 18, 19, 23, 24, 30. Every defect below compiled with zero warnings
and most survived keyword greps. **Build-clean is not verification** — this is now demonstrated, not
asserted.

| defect | why greps missed it |
|---|---|
| bibliography emitted NOTHING (one duplicate entry aborted BibTeX; all 366 citations `[?]`) | exit code 0 |
| four "five requirements" phrases after renumbering to four | searched for R5/V5, not the English word |
| `channel\'s` → `channelś` (LaTeX `\'` is an acute accent) | not a word, a glyph |
| a mislabelled question bank (`--bank expinfo` scoring random draws) | the LOG said expinfo |
| "posterior strictly shrinks (V3)", "the V3 shrinkage diagnostic" | belief claims with no "belief" |
| three "ties the strongest shallow baseline" + one "belief mean" | swept for changed NUMBERS, not the WORDS |
| fixed-bank gate called V4 in four places (it is V3) | gate names look correct in isolation |
| a "Σ-greedy selector" | names a deleted layer without naming it |
| `\date{July 2026}` | nobody greps the date |
| abstract claiming tail wins "throughout", which the CIs refute | prose, not a number |

**The pattern: a claim survives a keyword sweep when the prose paraphrases the concept.** Reading is the
only thing that finds those.

### Pages NOT yet read: 5, 7–17, 20–22, 25–29, 31–39.

### Two old-tower numbers left on p.23, deliberately unresolved
"popularity bank beats by +0.1264" and "mean 1.4 of 16 answered vs 13.5". Tonight's G9 gives the same
quantities on the certified tower but on a q=2..16 basis (1.97 of the first two answered vs 0.79; rise
+0.1035 vs +0.0051). Restate on the new basis or mark old-tower — **do not mix**.
