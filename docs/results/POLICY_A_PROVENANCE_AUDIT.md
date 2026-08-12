# Policy-A provenance audit — is anything else in the battery contaminated?

> Audit 2026-08-12. Read off CODE and ARTIFACT METADATA, not from assumption. No experiment was run;
> no recommender was scored. Two heavy jobs were live on the box throughout and were not touched.
>
> Question asked: **beyond the banked global greedy, is any other arm or reported quantity in the
> Paper-B battery contaminated by policy A — not only in scoring, but in SELECTION, or in a banked
> intermediate that is still being reused?**
>
> Prior art: `docs/results/MASK_POLICY_B_MIGRATION.md` (the migration, commit `21ca7fb`,
> 2026-08-05 16:13:07 +0100), memory note `banked-global-greedy-is-a-policy-A-artifact`,
> `docs/design/PAPER_B_TODO.md` item 1.17.

## The detection signatures used

| signature | policy A | policy B | where it lives |
|---|---|---|---|
| m=0 intercept, 10k VAL | `0.12987336107627623` | `0.15686538118354684` | `greedy_static_i26c.json:intercept.full`; `ndcgtree16_heaps/greedy_static_policyB_combined_heap.json:base` |
| m=0 intercept, 10k TEST | `0.1313911694759639 / 0.023434042774945676` | `0.15855 / 0.02819` | `greedy_static_i26c.json:intercept_eval`; `MASK_POLICY_B_MIGRATION.md` acceptance table |
| items masked per user | ~62.6 | 141.2 (nnz 1,411,761) | `MASK_POLICY_B_MIGRATION.md`; `chapterB_v2.tex` L370–371 |
| code funnel | `ctx.va_tr[ch] + ans[cj]` | `E["pool"][ch]` / `mask_X=D["pool"]` | `strategy_channel_suite.py:564`; `run_interview_table.py:298-299` |

## The decisive structural fact (verified, not assumed)

**There were two harnesses, and only one of them ever ran policy A.**

* The **arm-N / interview harness** (`src/baselines/`) has scored against `arm_n.build_pool`
  — `binarize(g_te_tr) | binarize(te_tr)`, i.e. *policy B* — since commit `12c53c1`, **2026-07-30**,
  five days *before* the migration. Every adaptive-battery cell reaches the metric through
  `run_interview_table.score_arm` → `M.evaluate(..., mask_X=D["pool"])` (`run_interview_table.py:298`),
  and `adaptive_harness.py` imports `score_arm` rather than re-implementing it (`adaptive_harness.py:76`,
  and the comment at `:264-267` states that this is deliberate). The migration commit `21ca7fb`
  **touched no file under `src/baselines/`** (verified with `git show --stat 21ca7fb`).
* The **instrument harness** (`src/instrument/`) is what was on policy A. `strategy_channel_suite.genv_eval`,
  `strategy_ladder.eval_budget`, `concept_cold_check.score` and `bank_sweep.py` were migrated on 2026-08-05.

So policy A never entered the *scoring* of any selector row. It entered exactly two places: (1) the
instrument-side artifacts built before 2026-08-05 16:13, and (2) the **content of the three greedy
orders**, which the arm-N harness then imports verbatim as `fixedorder:` comparators.

## The table

`selection touches candidate mask?` = could the rule that CHOSE the question have changed if the
candidate pool changed? "Still cited?" = appears in `docs/results/` or `new_chapters/chapterB_v2/`.

### A. The 23 rows of `BASELINE_OBJECTIVE_AUDIT.md`

| # | arm | selection touches candidate mask? | selected under | evidence field checked | still cited? | action |
|---|---|---|---|---|---|---|
| 1 | `golbandi_tree` | **no** — within-node squared error of rating prediction on `g_train` | n/a | `golbandi_tree_selector.py::_e2`; `golbandi_native.py:191-202` builds its own `va_pool = b_v + b_t` (policy B) | yes | none |
| 2 | `glm_ucb` | **no** — UCB index over `self.Y @ mu`; masks only the user's own asked items | n/a | `bandit_selectors.py:279-335` (`_pick` masks `asked`, nothing else) | yes | none |
| 3 | `thompson` | **no** — sampled posterior utility, same `_pick` | n/a | `bandit_selectors.py:_PosteriorSelector` | yes | none |
| 4 | `bald` | **no** — mutual information between parameters and answer | n/a | `bandit_selectors.py:339-395` | yes | none |
| 5 | `aopt` | **no** — trace reduction of posterior covariance (Sherman–Morrison) | n/a | `bandit_selectors.py:227-232` | yes | none |
| 6 | `conlinucb_mcr` | **no** — key-term posterior width / item UCB; `pool=` in this file is the ACTION bank, not the ranking pool | n/a | `conversational_bandit_selectors.py:485-514` (`ctx["cand_pool"]`, an action list) | yes | none |
| 7 | `conlinucb_bs` | **no** — uniform spanner draw | n/a | same | yes | none |
| 8 | `conlinucb_bs_static` | **no** — spanner draw seeded on `[seed, turn]` only; user-independent (1 distinct interview measured) | n/a | `BASELINE_OBJECTIVE_AUDIT.md` note; `ACT2_CONCEPT_BANK_PV.md` "distinct interviews = 1" | yes | none |
| 9 | `clime` | **no** — smallest eigendirection of `M` | n/a | `conversational_bandit_selectors.py` | yes | none |
| 10 | `conucb` | **no** — key-term-attributable estimation-error reduction | n/a | same | yes | none |
| 11 | `eddi` / `eddi_ig` | **no** — KL movement of the latent posterior; the target-aware term is deliberately dropped | n/a | `eddi_selector.py:16-24` header | yes | none |
| 12 | `dre` | **no** — squared reconstruction error of the full profile; seed set is a training artifact | n/a | `dre_selector.py:89-110` (`blob["itemseed"]`, `blob["pad_order"]`); bank trained 2026-08-07/08 (`DRE_TRAIN_OURS_20260807_214302.log`) | yes | none |
| 13 | `dpe_concept` | **no** — Thompson index over concept answer value | n/a | `dpe_selector.py` | yes | none |
| 14 | `dpe_oracle` | **no** — reads the user's own fold-in ratings (`D["g_te_tr"]`), never the pool | n/a | `dpe_selector.py:503` | yes | none |
| 15 | `igcn` | **no** — information gain about the cluster label | n/a | `igcn_selector.py:72` ("EXACT statement about the criterion, not a candidate-pool reduction") | yes | none |
| 16 | `greedy_slim` / `qgslim` | **no** — SLIM elastic-net squared-loss decrease | n/a | `scripts/baselines/greedyslim_native_snap.py` (native snap on the authors' own protocol) | yes | none |
| 17 | `fixedorder:entropy0` | **no** — Rashid-2008 entropy with "not rated" as an 11th band, over `g_train` | n/a | `interview_strategies.py::band_counts`/`entropies` — TRAIN users only | yes | none |
| 18 | `popularity` | **no** — train interaction count | n/a | same | yes | none |
| 19 | `helf` | **no** — harmonic mean of normalised entropy and log-frequency | n/a | same | yes | none |
| 20 | `pure_entropy` | **no** — rating-band entropy over raters | n/a | same | yes | none |
| 21 | `fixedorder:greedy_items` | **YES** — criterion is cohort-mean held-out NDCG@10 | **POLICY A** | `greedy_static_i26c.json` `intercept.full = 0.12987336107627623`, `intercept_eval = 0.1313911694759639 / 0.023434042774945676` | **yes** (chapter `tab:greedy`, `tab:battery`, L202, L384, L539; Act 1/Act 2 comparator) | **rebuild** (TODO 1.17) |
| 22 | `fixedorder:greedy_concepts` | **YES** — same | **POLICY A** | same file, arm `concepts-only` | **yes** (`tab:greedy`, `tab:battery`, Act 4, Act-2 concept-bank comparator) | **rebuild** |
| 23 | `fixedorder:greedy_combined` | **YES** — same | **POLICY A** | same file, arm `combined` | **yes** (`tab:greedy`, `tab:battery`; the NDCG-tree headline comparator) | **rebuild — in progress**, see §"the rebuild" |

**Working hypothesis: VERIFIED.** Every arm whose criterion never touches the candidate pool is
immune, and no code contradicts it. The only mask-dependent selection rule in the battery is the
NDCG-maximising greedy — which is the already-known case.

### B. Arms and quantities NOT in the 23 rows

| quantity | selection touches candidate mask? | selected under | evidence field checked | still cited? | action |
|---|---|---|---|---|---|
| **Genre group-greedy, stage 1** (`groupgreedy_genre15.json`) | YES (greedy on NDCG) | **POLICY B** | imports `build_greedy_env`/`genv_eval` from `strategy_channel_suite` (`group_greedy_genre.py:58`), which builds `pool_b = PA.ctx_pool_b(ctx)` at `:453` and scores through `E["pool"][ch]` at `:564`. Artifact mtime 2026-08-11 23:28, i.e. 6 days post-migration. Sidecar `groupgreedy_genre_ABANDONED_15step_partial.json` (2026-08-11 22:53) same env, `contract_assertion` field-by-field match recorded in the output | yes (`tab:groupgreedy`) | **its comparator is contaminated, not its own search** — see below |
| **Genre group-greedy, stage 2** (`groupgreedy_genre15_TEST.json`) | YES | **POLICY B** | re-scores both sides in-run: `per_group.Drama.curve.1.global_full@10 = 0.20358553596170048`, far above the policy-A `items-only` q1 of `0.14711`, i.e. the global order is re-scored under B, not read back | yes | same |
| **Genre arm's GLOBAL comparator** | YES | **POLICY A** | `global_sequence` in both stage-1 and stage-2 JSON is `[[0,83],[0,583],[0,19],…]` — byte-identical to `greedy_static_i26c.json` arm `items-only` | **yes** (`tab:groupgreedy`, `+0.00817 [+0.00539,+0.01097]`, "global baseline 0.24356") | **re-run against the rebuilt items order** |
| **NDCG tree** (`ndcgtree.json`, `ndcgtree16test.json`) | YES | **POLICY B** | `nodes.root.gains[0] = 0.020490556975873986` recorded as `root_gain_policyB`; all CELF heaps under `ndcgtree*_heaps/` are dated 2026-08-11/12 and their `sig` carries `contract_tag: "genv:ndcgtree(val):validation"` | yes | **comparator only** |
| **NDCG tree's comparator** | YES | **POLICY A** | `comparators.combined.sequence[0] = [1,630]`; `nesting_check.headline_first_gain_banked_policyA = 0.016569454905629777` vs the tree's own policy-B root `[1,114]` at `0.020490556975873986`. Offset **≈ +0.0009 full on the first pick alone**, and concept 114 appears nowhere in the comparator's 15 questions | yes (`CONDITIONED_ARMS_ON_TEST.md`) | **re-measure against the rebuilt combined order** |
| **Depth-1 group greedy** (`groupgreedy_d1.json`) | YES | **POLICY B** search | mtime 2026-08-11 03:12; same `genv_eval` path | its q1 is `[0,83]`, **read from** `greedy_static_i26c.json` (`group_greedy_d1.py:17,54`) | superseded by the genre arm | re-derive q1 if resurrected |
| **`greedy_polish.json`** (endpoint polish) | YES | **POLICY A** | mtime 2026-07-29 08:16 | dropped (memory `prefix-optimal-greedy-is-the-right-object`) | none — keep dropped |
| **`greedy_static.json`** (pre-i26c greedy) | YES | **POLICY A** | mtime 2026-07-28 05:38 | consumed only by `strategy_portability.py:60`, whose output `experiments/baselines/ml25m_liang/strategy_portability.json` (2026-07-29) is **cited nowhere** (grep-verified over `docs/` and `new_chapters/`) | none; if TODO 1.4 is ever done, re-point at the REBUILT order, not at `greedy_static_i26c.json` |
| **Act-2 PV, criterion `U(q\|z)`** | **NO — and it applies NO candidate mask at all** | mask-free | `dispersion_leg1.py:282-291`: `V(z) = Σ_{r≤10} exp(top_r − logsumexp(S)) · disc_r` over the **full 18,359-item** score row. No `-inf`, no pool, no `va_tr`. `act2_pv_quartiles.py` calls the same `C.V` | yes | **caveat only** — see §"the Act-2 finding" |
| **Act-2 state counts** 11 / 112 / 1,463 / 5,767 | **NO** | mask-free | read off `fixedorder:entropy0` interviews (train-statistics order); `ACT2_PV_QUARTILES.md`: "Every cell's interview and state counts are exact regardless — those need no criterion evaluation" | yes (abstract, intro, Act 2) | none |
| **Dispersion / perplexity table** (`REDDIT_DISPERSION_LEG1.md`, `tab:dispersion`) | **NO** | mask-free | same `V(z)`; doc states "the criterion is a function of the STATE alone" | yes | none |
| **Act-2 quartile trend cells** (38/48, and the concept-bank 77/96) | selection no; but the **static comparator** is `fixedorder:greedy_items` / `greedy_concepts` | comparator POLICY A | `act2pvc_gates.py:46-48` names `fixedorder:greedy_items` as the arm carried in both runs | yes | **direction is safe** (see below), **magnitudes move** |
| **G9** (`g9.json`, 2026-08-05 15:24) | no — fixed pop/random banks | both columns printed | `pool_A` and policy-B columns both in the artifact; column A bit-reproduced in the migration | yes (Paper A R3‡) | none |
| **G1/G2/G3a/G3b/G4/G5/G6/G8** instrument gates | no — fixed token sets | G3a/G3b/G6 use `ctx.allb_mask` (all rated ⇒ policy-B-equivalent); **G5 uses `mask_parity = ctx.va_tr` ⇒ policy A** | `run_battery_phaseA.py:331,422,441,642-645` | gates, not battery numbers | **caveat**: G5's absolutes are policy A; its internal contrasts are mask-matched |
| **Graded/premium curve** (`premium_vs_k*.json`, `GRADED_CURVE_RESULT.md`) | no — the revealed set is the user's own items | **POLICY A, declared** | `premium_vs_k.py:78` writes `"mask": "PARITY (liked fold-in only)"` | yes (Paper A) | none — declared, and both arms share the mask |
| **Bank sweep** (`bank_sweep.json`, `CONCEPT_VS_ITEM_BANK_SWEEP.md`) | no — banks are `pop`/`entropy`/`expinfo`/`polar` rankings | **neither A nor B — a third contract**: answered-asked-items only, *no fold-in mask at all* | migration commit message in `concept_cold_check.py`: "The previous rule masked only the ANSWERED asked items and applied no fold-in mask at all". Floor `0.1277/0.0267` (old) vs `0.2154/0.0354` (`bank_sweep_policyB.json`, 2026-08-05 15:34) | **yes, and the doc still cites the OLD file** | **re-point the doc** — see below |

## The four artifacts the author asked about explicitly

### (a) Banked interviews, cached per-user vectors, `*.partial.json`, CELF heaps

* **No `*_per_user.npz` anywhere under `experiments/` predates the migration.**
  `find experiments -name "*.npz" ! -newermt "2026-08-05 16:13"` returns **nothing**. The earliest is
  `act1_per_user_vectors.npz`, 2026-08-05 20:44 — four and a half hours after the migration commit.
* **All CELF heaps are post-migration.** `ndcgtree_heaps/`, `ndcgtree_SANITY_heaps/`,
  `ndcgtree_PREFLIGHT_heaps/`, `ndcgtree16test_heaps/`, `ndcgtree16_heaps/` — mtimes 2026-08-11 23:59
  through 2026-08-12 17:13. Their `sig` carries `contract_tag`, `split`, `objective`, `prefix`,
  `pool_sha1`, `rows_sha1` (`strategy_channel_suite.py:649-664`).
* **One interview directory predates it in name only.** `adaptive_table_i26c_ep31.json`
  (2026-08-04 20:22) records `meta` with no pool field — **its policy cannot be read off the artifact**.
  It is verified clean by code and git instead: its scoring path is `mask_X=D["pool"]`, unchanged since
  2026-07-30. That is a code-and-git verification, not an artifact-metadata one, and is labelled as such.
* **★ NEW DEFECT — the greedy resume sidecar cannot refuse a policy-A resume.**
  `greedy_static_i26c.partial.json` (2026-08-05 03:47, policy A) carries
  `sig = {tower, conc, objective, item_bank, conc_bank, n_items, n_concepts, L, n_build_users, build_head, build_tail}`
  — **no pool / policy / contract-tag field** (`strategy_channel_suite.py:978-987` builds it, `:984`
  compares it). A policy-B re-run of `strategy_channel_suite.py` writing to the same output path would
  match that signature exactly and **silently resume the policy-A picks**. The CELF heap signature
  (`celf_heap_sig`) does carry `contract_tag`; the arm-level sidecar does not.
  **Action: add `pool_policy` (or `contract_tag`) to the `sig` in `strategy_channel_suite.py:978`, and
  until then run the rebuild to a NEW output path.** The in-flight rebuild already does this
  (`ndcgtree16_heaps/greedy_static_policyB_combined_heap.json`), so nothing is currently at risk — but
  the guard is missing, not satisfied.

### (b) Act 2 — criterion states, dispersion table, PV quartiles

**The criterion applies NO candidate mask, of either policy.** `V(z)` (`scripts/reddit/dispersion_leg1.py:282-291`)
is the discounted top-10 mass of the tower's own predictive over the **entire 18,359-item catalogue**,
normalised by `logsumexp` over the entire catalogue. There is no `-inf`, no `va_tr`, no `pool`.
`act2_pv_quartiles.py` imports that same `Criterion` class (`.cache/act2_pv/contract.json` records
`"criterion": "dispersion_leg1:Criterion"`). Therefore:

* `PV(u) = U(q*_u|z_u) − U(q*_pop|z_u)`, the 11/112/1,463/5,767 state counts, the modal-q\* shares, the
  perplexity/dispersion table and the quartile cuts are **pool-invariant**. **Not contaminated.
  No rebuild.**
* PV is read at `fixedorder:entropy0` states, and entropy0's order is a train-statistics order — also
  pool-invariant. The strata are therefore stable under any pool change.
* **★ But there is a header/field mismatch worth one sentence of caveat.** `ACT2_PV_QUARTILES.md:11-12`
  and `ACT2_CONCEPT_BANK_PV.md:11-12` say *"Contract of every number below … policy-B / fixed arm-N
  candidate pool"*. That is true of the NDCG columns and **false of the PV/criterion columns**, which
  are on the unmasked catalogue. Two different rulers under one header — the same failure mode as
  "a summary statistic and the table it summarises must read the same field". **Action: split the
  contract paragraph so the criterion's pool (none) is stated separately from the metric's pool (B).**
* The **trend cells** (38/48 in the quartile doc, 77/96 in the concept-bank doc) difference an adaptive
  selector against `fixedorder:greedy_items` / `fixedorder:greedy_concepts` — **policy-A-selected
  comparators**. The *direction* is safe by the same handicap argument that protects Act 1: an order
  optimised under A and scored under B is handicapped, so a NEGATIVE adaptive−static gap measured
  against it is **conservative**. The *magnitudes* will move when the orders are rebuilt.
  **Action: re-difference after the rebuild; do not re-argue the sign.**

### (c) The genre arm's banked group orders

**Its own search was POLICY B — confirmed from code, not from the date.** `group_greedy_genre.py:58`
imports `build_greedy_env`, `genv_eval`, `greedy_certified` from `strategy_channel_suite`, whose
`build_greedy_env` sets `E["pool"] = PA.ctx_pool_b(ctx)` (`:453`) and whose `genv_eval` scores through
`ndcg10_from_scores(S, E["pool"][ch], …)` (`:564`) with the mask contract live. Stage-1 mtime
2026-08-11 23:28; the sidecar it imported completed groups from,
`groupgreedy_genre_ABANDONED_15step_partial.json`, is 2026-08-11 22:53 — both six days post-migration,
and the output records a field-by-field `contract_assertion` against that sidecar.

**Its COMPARATOR is policy A.** `global_sequence = [[0,83],[0,583],[0,19],[0,8],[0,22],…]` is the
`items-only` arm of `greedy_static_i26c.json` verbatim. So the headline
`FALLBACK-200 +0.00817 [+0.00539, +0.01097]` is *policy-B-selected group orders vs a policy-A-selected
global order*. Both sides are re-scored under B in-run (verified: Drama K=1 global = `0.20358`, not the
banked `0.14711`), so the **ruler** is shared — but the **selection contract is not symmetric**, and the
asymmetry runs **in favour of the conditioned arm**, the opposite direction from the "lower bound"
caveat the chapter already carries for `tab:battery`.

**★ The chapter states the symmetry as a fact and it is not one.** `chapterB_v2.tex:731-733`:
> "hold the objective, the optimiser, the bank, the contract and the val/test discipline fixed, and
> vary \emph{only} what the search is conditioned on."

The contract was **not** held fixed. **Action: either re-run stage 1+2 against the rebuilt items order
(TODO 1.17's plan), or amend that sentence until the re-run lands.**

### (d) The concept-bank arms

| arm | verdict |
|---|---|
| `greedy_concepts` | **POLICY A selected**, policy-B scored. Cited as the best arm we have (`.2362/.1022 @k8`, `tab:greedy`, Act 4). Rebuild queued. |
| `conlinucb_bs_static` | **clean.** One uniform spanner draw seeded on `[seed, turn]`, user-independent (measured: 1 distinct interview). No pool dependence. |
| `fixedorder:` registry | exactly two sources. (i) `GREEDY_FIXED_ORDERS` → `greedy_static_i26c.json` (`adaptive_harness.py:112,133`) — **the three contaminated orders**. (ii) `glob`, the four heuristic global orders `{popularity, pure_entropy, entropy0, helf}` (`run_interview_table.py:99`), built from `g_train` band counts — **clean**. `treestatic:` reads the same `glob`. |
| stored-order answer convention | note for completeness: `adaptive_harness.py:104-110` deliberately IGNORES the greedy JSON's `use_refusal` / `concept_ckpt` / `contract` provenance fields and takes only the `(channel, id)` pairs. So the stored order carries no policy with it either — which is why the contamination is confined to *which questions*, never to *how they are scored*. |

## Documentation defects found on the way (all 51 files in `docs/results/` swept)

The sweep's discriminator: a pre-08-05 date on a `src/baselines`-derived doc is **not** a policy-A risk
(that path was already policy B from 2026-07-30); the risk set is `src/instrument`-derived docs dated
before 2026-08-05 15:59. Ranked by consequence:

1. **★ `CONCEPT_VS_ITEM_BANK_SWEEP.md` — every absolute number is on the pre-migration THIRD contract,
   and the doc says nothing.** Header verbatim: `> Raw: experiments/instrument/bank_sweep.json`
   (2026-08-04 09:17). Protocol section states the retired rule as current fact — *"Answered items are
   masked from the candidate list; the concept arm has no items to mask"* — and the canary reads
   *"at m=0 both arms fold nothing and score bit-identically, **0.1277 / 0.0267**"*. `grep -i policy`
   on the file returns **nothing**. `bank_sweep_policyB.json` (2026-08-05 15:34, floor `0.2154 / 0.0354`)
   and `concept_vs_item_paired_policyB.json` (2026-08-05 15:58) exist on disk and are **never mentioned**.
   The migration's own instruction — *"every absolute number in that doc is on the old pool and must be
   re-read from the new file"* — is outstanding, and the forward pointer it wrote into
   `concept_cold_check.py` (`docs/results/CONCEPT_VS_ITEM_BANK_SWEEP_POLICYB.md`) **points at a file that
   does not exist**. Direction survives (margins roughly double under B); the printed numbers understate
   the result. **Fix is a re-render; the data is already on disk.**
2. **`GREEDY_BEST_STATIC_I26C.md` — prose re-cut, table not.** Finding 2 *is* corrected and correctly
   attributes `.2104 vs .1997` to policy A. But the result table above it (lines 39–45) is unlabelled
   policy A: `Empty-interview intercept on test: 0.1314 / 0.0234`, `items-only … .2104/.0761`,
   `concepts-only … .1997/.0921`, `combined … .2115/.0880`, plus the gate line `m=0 empty interview on
   val: 0.12987`. Anyone copying that table beside a policy-B number produces a cross-policy comparison.
   Finding 4 also still asserts the combined arm is *"Paper B's strongest-static comparator … the hardest
   honest bar"* with no selection-policy caveat. **Add a policy-A banner to the table, or replace it with
   the policy-B column from the migration doc, and caveat finding 4 until the rebuild lands.**
3. **`GREEDY_BEST_STATIC_RESULT.md` (Jul-27 old-tower twin) is entirely uncorrected** and carries its own
   copy of the killed claim: line 48 *"**Items win FULL at long budgets** (.2098 vs .1808 concepts @q16)"*
   and line 93 *"on full-catalogue accuracy, **item information density beats answerability**"*. Its only
   warning is external and is about the **tower**, not the mask. **Needs the same ⛔ superseded box that
   `BASELINE_INPUT_REGIME_AUDIT.md` got.**
4. **`GRADED_CURVE_RESULT.md` cites a file that does not exist.** Header points at
   `experiments/battery/premium_vs_k.json`; on disk there are only `premium_vs_k_ep4snap.json`,
   `premium_vs_k_t2final.json` and `premium_vs_k_t2i26c_ep31.json` (2026-08-08, uncited by any doc —
   **could not verify what it supersedes**). The doc is also silent about being pure policy A, which is
   *correct by design* for that comparison but unresolvable as written.
5. **Silent-but-correct contracts** (cheap one-line fixes that would have removed this whole audit's
   ambiguity): `ACT1_ADAPTIVE_VS_STATIC.md` never names the pool — its only pool sentence is about the
   *ask* universe — yet its numbers are verifiably policy B (`entropy0` k=1 `.1635/.0433` matches the
   migration gate's `0.16348/0.04328`; MostPop `0.1626/0.0262` matches the canary). `INTERVIEW_TABLE.md`
   says it only in paraphrase. `CFOLD_KEYTERM_REPR_PAIRED_CIS.md` enumerates nine verified-equal contract
   fields and **the pool is not among them** (near-zero risk — a three-way Δ inside one run family — but
   under-specified).
6. **Pre-certification instrument cluster, all silent on policy, all printing the `0.12794/0.01923`
   intercept:** `CONCEPT_CHANNEL_RESULT.md`, `CONCEPT_DISTILL_STEP0_RESULT.md`,
   `CONCEPT_DISTILL_TRAIN_RESULT.md`, `DISTILL_DOWNSTREAM_RESULT.md`, `ANSWER_DECOMPOSITION_RESULT.md`,
   `ANSWER_MODEL_CIRCULARITY_AND_IMPUTERS.md`. All already demoted on independent grounds (old tower,
   retired answer environment); none carries a mask-policy note. `DISTILL_DOWNSTREAM_RESULT.md` names
   **no artifact at all** — **could not verify its provenance.**
7. **`PAPER_B_TODO.md` 1.17 says "author has removed the in-chapter disclosure".** The disclosure is
   still in the chapter: `chapterB_v2.tex:367-391` ("Two candidate pools, and why the comparison must be
   made inside one", with `62.6` vs `141.2`, `.1314 → .1586`, and a live `\REDO{a greedy re-run whose
   selection contract is the protocol pool}`), plus `:482` ("Greedy rows are \emph{lower bounds} — the
   sequences were selected under a narrower…"). **The TODO line is stale; the chapter is honest.**

**Verified clean, per class, with the contract line each declares:** the four port docs and their paired-CI
twins (`CONVBANDIT_PORT_ON_OUR_RULER.md`, `CONVBANDIT_PAIRED_CIS.md`, `DRE_PORT_ON_OUR_RULER.md`,
`DPE_PORT_ON_OUR_RULER.md`, `GOLBANDI_PORT_ON_OUR_RULER.md`, `GOLBANDI_PORT_PAIRED_CIS.md`,
`PORT_PAIRED_CIS.md`) each carry an explicit *"policy-B pool"* line with a post-08-05 artifact on disk;
`ACT1_PAIRED_CI.md` (*"fixed arm-N candidate pool"*), `ACT1_FLOORED_ARM.md` (*"policy-B pool"*), both ACT2
docs, `CONDITIONED_ARMS_ON_TEST.md` (which proactively self-discloses the comparator residue in §7),
`ARM_N_PHASE1.md` and `PROTOCOL_DISLIKE_DISCARD.md` (which define the pool). The `SNAP3/4/5/IGCN` docs and
the whole Reddit set are **not on this ruler at all** — native reproductions inside the target papers' own
protocols, or counts on the Reddit ruler — so the mask policy does not apply; two of them (`SNAP4_*`)
independently cite policy B as external support for our ruling. `PAPER_A_NUMBER_AUDIT.md` is full-profile
certification, which the migration **retains** on policy A by design.

**Could not verify:** `MASK_CONTRACT_DIAGNOSTIC.md`'s `decisive.py` / `decisive.json` (scratchpad,
deliberately uncommitted — outside the repo, not inspected); `DISTILL_DOWNSTREAM_RESULT.md`'s source
artifact (none cited); the contents/lineage of `premium_vs_k_t2i26c_ep31.json` (no doc references it).

## The rebuild, as it stands right now

`experiments/battery/ndcgtree16_heaps/greedy_static_policyB_combined_heap.json` records
`base = 0.15686538118354684` (the policy-B m=0), `tag = "staticB"`, `nev = 1999`, and a completed
15-pick sequence `[[1,114],[1,173],[0,19],[1,748],[1,785],[0,83],[0,583],[1,243],[1,231],[1,809],
[1,455],[0,22],[0,77],[0,88],[1,18]]` with first gain `0.020490556975873986`. Its `sig` carries
`contract_tag: "genv:ndcgtree(val):validation"`, `pool_sha1`, `rows_sha1`. The combined arm is done;
items and concepts remain.

Set overlap between the rebuilt combined order and the banked policy-A combined order
(`[[1,630],[1,242],[1,173],[0,22],[1,562],[1,537],[0,8],[0,112],[0,83],[0,130],[1,754],[1,180],[0,21],
[0,19],[0,219]]`) is **5 of 15** — the two orders are substantially different objects, which is the
concrete reason the affected comparisons need re-measuring rather than a footnote.

## Verdict

**Nothing new is contaminated.** The working hypothesis holds in full: every arm whose criterion never
touches the candidate pool — entropy0, popularity, HELF, pure entropy, DRE, Greedy SLIM, the Golbandi
port, BALD, A-optimal, the runtime bandits, EDDI, IGCN, DPE, CLiME, ConUCB and the ConLinUCB family — is
immune, and I found no code that contradicts it. The arm-N harness was already on policy B a week before
the migration, so no selector row was ever scored under policy A. The Act-2 machinery is immune for a
different and stronger reason: its criterion applies no candidate mask whatsoever, so the state counts,
the dispersion/perplexity table and the PV quartiles are pool-invariant and need no rebuild.

**The contamination is exactly one object with three arms** — the banked NDCG-maximising greedy orders
— and its blast radius is *comparators*: Act 1's static rows, the Act-2 trend cells' static comparator,
the genre arm's global order, the NDCG tree's `combined` comparator, and the depth-1 group-greedy's `q1`.
Everywhere it acts as the thing the adaptive arm must beat, the handicap runs in our favour and the
reported result is conservative. In the **two** places where a policy-B-selected arm is differenced
against it — the genre arm and the tree — it runs the other way, and those two numbers are the ones that
must be re-measured rather than caveated. That was already TODO 1.17's finding; this audit adds that the
list is complete, adds the Act-2 comparator to it, and adds the resume-sidecar defect.

**Two things I could not verify from artifact metadata** and verified from code + git history instead,
labelled rather than asserted: (i) `adaptive_table_i26c_ep31.json` (2026-08-04) records no pool field —
its `meta` is `{selectors, budgets, consumers, arch, snapshot, no_floor, n_test_users, ucb_alpha,
ts_seed, eddi_cand_topm}` and nothing about masking; (ii) the group-greedy and NDCG-tree JSONs record no
`pool_policy` field either — the tree is confirmed by its recorded `root_gain_policyB`, the genre run
only by its import graph and its mtime. **Every one of these would be a one-line fix: stamp
`pool_policy` into the output blob of `strategy_channel_suite.py`, `group_greedy_genre.py`,
`ndcg_tree.py` and `run_interview_table.py`.** An artifact that cannot state its own ruler is one
migration away from this audit having to be run again.
