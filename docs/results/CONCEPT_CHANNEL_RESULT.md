# RESULT — The concept channel (signed, exposure-corrected) — durable hard record

> Author verdict (2026-07-25): the concept-channel story "directionally checks out"; recorded AS-IS,
> honest, with the open items visible. All numbers below are pulled from the committed JSONs / run
> logs, not from prose. Sources are named per block; do not re-transcribe from this file into a claim
> without re-checking the JSON.
>
> Harness: one shared harness, signed module, 10,000 COLD_SEED users, per-question deployment currency
> (every question burns budget whether answered or not). Cold intercept (0 answers):
> **full 0.12794 / tail 0.01923** (`strategy_channel_suite.json:intercept`).
> Primary rung = **sclite (signed C-lite)**, the frozen-tower trained fold-to-point.

---

## 1. The arc in one paragraph

Concepts began "completely unacceptable." A positive-only clip `[0.25,1]` on the SEL answer value
(born Jul-24 in `concepts_only_curve.py`, commit `c89d4d2`→`7f45fd2`) discarded the **signed negative
half** of the validated SEL estimator (`bpool_r2`, Jul-18: SEL = `log2((n+0.5)/(e+0.5))` = user-concept
PMI/NPMI, exposure/popularity-corrected; SEL explains 0.376 of concept-affinity variance + VAL 0.042,
vs an LLM ordinal's 0.017). Down a realizable fixed bank most answers are weak positives, so the
clip's accumulated weak-positive poison dragged the deployment curve **below the no-answer intercept**
by q16. Two fixes turned it positive: (A) **diversified selection** (`|cos|<0.5`) resolved the m8
"crater" (a correlated-sibling over-count, not a channel limit); (B) **signed four-band answers**
(graded like / meh / graded dislike / refuse-iff-unmeasurable) + a **retrained C-lite** (curriculum
`m~U{1..16}`, per-user volume-normalized negatives, frozen tower). The retrained triple gate is
CI-clean on taste; the volume-leak gate still **FAILS** (open, below). The full strategy×channel
picture: **concepts own the short interview, items win the long one, the adaptive prize is large and
unclaimed.**

---

## 2. Strategy × channel suite — the headline picture

Source: `experiments/battery/strategy_channel_suite.json`; figure
`experiments/battery/strategy_channel_curves_2026-07-25.png` (copied into the chapter as
`fig_strategy_channel_suite.png`). Primary rung = signed C-lite. full@10 / tail@10 by budget q:

| strategy | q1 | q2 | q4 | q8 | q16 |
|---|---|---|---|---|---|
| items-pop | .1307/.0266 | .1305/.0315 | .1319/.0343 | **.1673/.0504** | **.1948/.0580** |
| items-HELF | .1281/.0196 | .1304/.0286 | .1391/.0384 | .1490/.0460 | .1723/.0580 |
| items-entropy | .1280/.0195 | .1281/.0195 | .1281/.0196 | .1284/.0206 | .1295/.0219 |
| items-random-bank | .1280/.0193 | .1280/.0195 | .1284/.0198 | .1291/.0203 | .1301/.0219 |
| conc-polarization | **.1323/.0348** | **.1328/.0343** | **.1433/.0395** | .1441/.0395 | .1471/.0389 |
| conc-member-mass | .1253/.0240 | .1320/.0298 | .1345/.0257 | .1417/.0334 | .1379/.0324 |
| conc-random | .1285/.0216 | .1296/.0243 | .1337/.0292 | .1393/.0359 | .1425/.0380 |
| mixed-c4-then-items | .1253/.0240 | .1361/.0301 | .1366/.0321 | .1466/.0401 | .1775/.0498 |
| mixed-interleave | .1253/.0240 | .1294/.0295 | .1391/.0358 | .1448/.0399 | .1583/.0481 |
| ADAPTIVE-greedy (realizable) | .1253/.0240 | .1361/.0301 | .1366/.0321 | .1372/.0343 | .1397/.0343 |
| **ADAPTIVE-oracle (PRIVILEGED)** | .1537/.0638 | .1757/.0831 | .1946/.1045 | .2123/.1220 | **.2237/.1305** |

(`conc-HELF-analog` is numerically identical to `conc-polarization` in this suite.)

**Reading (each verified against the JSON):**

1. **Concepts own the short interview.** Polarization-ranked concepts beat *every* item strategy on
   **both** metrics through q4: at q1 `.1323/.0348` vs the best item (items-pop `.1307/.0266`); at q4
   `.1433/.0395` vs the best item (items-HELF `.1391/.0384`). **The single best opener is a polarizing
   concept question** — conc-polarization is the highest realizable full AND tail at q1.

2. **Items win the long interview.** items-pop is the highest realizable arm at q8 (`.1673/.0504`) and
   q16 (`.1948/.0580`); the best concept arm tops out around `.1471/.0389` (polarization q16). The
   crossover sits between q4 and q8.

3. **Static mixed schedules underperform pure items at q8+.** mixed-c4-then-items q8 `.1466` / q16
   `.1775`, mixed-interleave q16 `.1583` — all below items-pop's `.1673` / `.1948`.

4. **The adaptive prize is large and UNCLAIMED.** The privileged adaptive **oracle** dominates every
   realizable arm at every budget (q16 `.2237/.1305`, vs items-pop `.1948/.0580`). The realizable
   **greedy** selector closes only a small single-digit fraction of the oracle−static gap
   (`gap_closure_u3b_subsumed`): at q16 oracle−static full gap `0.0858` / tail `0.0981`, greedy closes
   **2.1% (full) / 1.9% (tail)**; at q8 greedy closure is **negative** (full −6.5%). Directly,
   greedy − member-mass @q8 full = **−0.0046** (CI [−0.0059, −0.0033]) — the realizable greedy does not
   even beat the static member-mass concept order at q8. → motivates **Chapter B** (a learned selector /
   Golbandi-tree lineage) on this same harness.

---

## 3. Signed-SEL triple gate — on the RETRAINED signed model

Source: `experiments/battery/signed_sel_gate_signed_retrain.json` (`status: SCORED`, 10k users, same
intercept). Comparator = the clip-up (positive-only) convention. full@10 by q:

| arm | q2 | q4 | q8 | q16 |
|---|---|---|---|---|
| clip-up comparator | .1272 | .1290 | .1269 | .1268 |
| **Arm1 signed (bpool_r2 port)** | **.1286** | **.1311** | **.1454** | **.1372** |
| Arm2 popularity-counterfeit | .1260 | .1236 | .1149 | .0816 |
| Arm1' value-permuted (sanity) | .1213 | .1180 | .1190 | .1037 |
| Arm3 meh-only | .1283 | .1314 | .1335 | .1319 |

**Verdicts (from `decision` / `kt_a3_leak_probe`):**

- **Signed beats clip-up, CI-clean.** Arm1 − clip-up @q16 = **+0.01043** (CI [+0.00753, +0.01336]).
  Clip-up sinks below the 0.1279 intercept by q8 (.1269) and stays there (.1268 @q16); signed holds
  above it (.1372 @q16). `strong_support = true`, VERDICT "STRONG SUPPORT."
- **Popularity counterfeit reproduces ~0%.** Arm2 gain @q16 = **−0.04520** (it *declines*), so the gain
  is not a volume×popularity artifact. `hard_kill_arm2 = false` (kill rule was ≥70% of Arm1's gain).
- **The values carry it.** Arm1 − value-permuted @q16 = **+0.03353** (CI [+0.03095, +0.03607]);
  permuting the values across concepts collapses the gain → it is the ANSWER VALUES, not fold
  cardinality.
- **OPEN — volume-leak gate FAILS.** KT-A3 ridge R² (fold latent @q8 → log user volume): clip-up
  **0.02365** → signed **0.24928**, vs the pre-registered acceptance bar **≤0.05**. The per-user
  `C_NEG` negative-channel cap did **not** hold the leak down. **GATE FAIL, pending author ruling.**
  The counterfeit-clean result (Arm2 ~0%) is attached as the waiver-consideration evidence: the leaked
  quantity is user *volume*, but the counterfeit control shows the ranking gain is taste, not a volume
  meter. This is NOT softened — a signed-answer build is not certified while the leak stands.
- Minor control caveat (honest): `snap_clipup_vs_ledger.PASS = false` — the retrained model's clip-up
  comparator deviates from the old ledger's clip-up row (Δ up to +0.0108 @q16). Expected (different
  model), noted not hidden.

Amendments on record (`gate.amendments`): author rulings 2026-07-25 — OOD asymmetry (arm1 worse =
INCONCLUSIVE, only arm2 ≥70% = HARD KILL); value = bpool_r2 SEL+VAL port; value-permutation sanity row
kept; KT-A4 dropped.

---

## 4. Ledger — signed C-lite (sclite) vs signed C-full (scfull)

Source: `experiments/battery/tradeoff_ledger.json` (6 rungs armA/fixab/clite/cfull/sclite/scfull);
G0 numbers from `experiments/battery/cfull_signed_train.log` + `ledger_signed_run.log`.

**Per-answer concepts-only (privileged: each user's own top-SEL concepts), full@10 / tail@10:**

| rung | m1 | m2 | m4 | m8 | redundancy-robust |
|---|---|---|---|---|---|
| **sclite (signed C-lite)** | .1318/.0256 | .1366/.0323 | .1450/.0452 | **.1630/.0639** | full ✓ / tail ✓ |
| scfull (signed C-full) | .1307/.0232 | .1339/.0281 | .1416/.0406 | .1566/.0577 | full ✓ / tail ✓ |

**Deployment concepts-only (realizable fixed bank, per-question currency), full@10 / tail@10:**

| rung | q2 | q4 | q8 | q16 | concepts beat items-pop at |
|---|---|---|---|---|---|
| **sclite** | .1320/.0298 | .1345/.0257 | .1417/.0334 | **.1379/.0324** (above intercept) | q2 |
| scfull | .1351/.0295 | .1330/.0293 | .1315/.0342 | **.1052/.0259** (CRATERS below intercept) | q2 |

**Item-G0 cost:**

- **sclite** = trained fold on the **frozen ep4 tower** → G0 holds by construction (zero-init fold,
  bit-identity at empty profile); the frozen tower's own test G0 ≈ 0.3536/0.2497, ties the 0.3540 bar.
  Its deployment items-pop curve is identical to the plain frozen tower (q16 .1948).
- **scfull** = concept tokens trained **into** the tower → G0 **TEST full@10 = 0.3487 / tail 0.2471**
  vs RecVAE bar **0.3540** (diff **−0.0053**, CI 0.0070, **tie = True**); best val full 0.3510;
  cold-start coldk8 0.2611 ≥ sclite 0.2584; deployment items-pop q16 .1988 ≥ .1948. So training
  concepts into the tower **costs ~nothing on items** — it even nudges cold-start up.

**VERDICT: signed C-LITE WINS.** sclite is higher on per-answer concepts (m8 .1630/.0639 vs
.1566/.0577) AND, decisively, in deployment: sclite stays above the intercept through q16 (.1379)
while scfull craters to .1052 (below the 0.1279 intercept) by q16. Tokens-in-tower is item-safe but
buys nothing on concepts and hurts the long concept interview. This matches the pre-signed ledger
verdict (C-lite already won there).

---

## 5. Open items (carried honestly)

1. **Volume-leak gate FAIL** (§3): signed fold latent leaks log user-volume at R² 0.249 » 0.05 bar.
   Pending author ruling; counterfeit-clean evidence attached as waiver-consideration. Fallback grid
   (pre-registered in the design sheet): `C_NEG ∈ {1.0, 2.0, 4.0}` on val, or a volume-orthogonalised
   fold / invariance penalty.
2. **C-lite vs C-full operating point** — C-lite wins the ledger, but the choice is an author call
   (the split-G5 attribute-semantics gap below is where C-full was positioned to help and did not).
3. **G5 split still open** — trained concept operators learn "what X-likers watch," not "members of X"
   (sclite member-AUC k1 0.630, disc Spearman 0.948 vs log-pop / 0.030 vs member-ness; pop-projection
   control PASSES +0.0087 CI-clean). Fine for recommendation, a real gap for the R3 attribute-semantics
   claim; C3 human round-trip is where it ultimately gets tested.
4. **Certification (t2final) PARKED** until the author picks the final winner. No cert retrain launches
   from a shakedown.

## 6. Methods lesson (for the paper)

**Capability gates can be vacuous.** The G3 sign-flip capability gate (flip = −0.201, CI-clean) passed
on a channel that, as actually driven in deployment, **never emitted a negative answer** — the clip
discarded the negative half before it ever reached the operator. The regression was caught only by a
**deployment-currency gate**: a realizable, user-independent question bank under a per-question budget,
scored at the endpoint, where an answer model that never exercises a capability is penalised by the
same declining curve that exposed the clip. This class of miss nearly certified a broken channel; the
battery now carries the deployment-currency endpoint as the acceptance form of the concept channel.

## Provenance / lineage
Design + pre-registration: `docs/design/DESIGN_SIGNED_CONCEPTS.md`. Full archaeology (every fold
operator, the exact clip provenance): `external_literature/findings/concept_fold_archaeology.md`,
`concept_folding_and_implicit_attribute_inference.md`. Running record:
memory `concept-channel-escalation-2026-07-24`. Paper: `new_chapters/chapterA_v2/chapterA_v2.tex`
§sec:concept (subsections est/fold/fail + the new results subsection).
