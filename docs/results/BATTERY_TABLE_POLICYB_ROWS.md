# A9 — the chapter's greedy static rows, rebuilt under policy B, with paired CIs

**Deliverable for the merge. Nothing in `chapterB_v2.tex` was edited by this run** (a writing agent
holds that file). Everything below is drop-in replacement text plus the verification of the sentences
that quantify over those rows.

Written 2026-08-12. Analysis `src/baselines/a9_battery_policyB.py` (commit `fdacf02`), raw
`experiments/baselines/interview/a9_battery_policyB.json`, render `src/baselines/a9_render.py`.

**This is a RE-SCORE, not a search.** The three orders are frozen artifacts (A1/A3/A4) and their
per-user TEST vectors were already banked by `src/instrument/channel_orders_on_test.py`. No
recommender was re-fit, no order re-selected, no user sampled.

Contract, identical to the rest of the table: 10,000 canonical TEST users, consumer = certified pair
(`t2i26c_ep31.pt` + `concept_cv3_ep08.pt`), **policy-B** candidate pool, NO-FLOOR, behavioural
answers, learned decoder bias, full/tail NDCG@10, no sampling. Inference: paired percentile bootstrap
over per-user differences, **10,000 resamples, seed 98765** — never the unpaired form.

---

## 0. Gates (all passed; no number below is reportable without them)

| gate | check | result |
|---|---|---|
| **banked targets** | the rebuilt orders' K=16 test figures must reproduce the banked headline | combined `0.25900 / 0.12370`, items `0.24470 / 0.10171`, concepts `0.24467 / 0.11607` — **all three exact to 5 dp** |
| **G1 cross-family** | the instrument harness (which produced the static rows) and the arm-N harness (which produced every selector row) must agree **per user**. They share exactly one arm: both items orders open with item 83, so at K=1 they are the same interview | max per-user diff `2.98e-08` raw, **`0.0e+00` after the instrument's own float32 storage round-trip**, on both axes; the tail row-compaction (10,000 → 9,281) lines up |
| **G2 cross-run** | every selector source file carrying a shared static arm must reproduce the banked act1 `entropy0` vector per user | `gtport` / `baldaopt` / `igcn` / `convrescore` all **0.0e+00** |
| **bootstrap equivalence** | the memory-bounded chunked bootstrap must equal the `gtport_paired` reference exactly | **IDENTICAL** on a real difference vector |
| **self-test** | an arm against itself | delta and both CI ends exactly `0.0` |

Cross-family pairing between the two harnesses is used here for the first time; G1 is what licenses
it. The float32 note is not cosmetic — the instrument banks float32, so bit-identity is only a
meaningful claim after the round-trip, exactly as the source run's own banked assert does it.

---

## 1. `tab:greedy` (Act 1) — replacement rows

Caption is unchanged and still correct: intercept (no questions) `0.1586/0.0282`.
**All three `emergent order` strings change**, and the combined one changes character.

```latex
items-only    & \texttt{iiiiiiiiiiiiiiii} & .1760/.0434 & .1884/.0563 & .2038/.0677 & .2235/.0860 & .2447/.1017 & 3.11 i \\
concepts-only & \texttt{cccccccccccccccc} & \textbf{.1808}/\textbf{.0537} & \textbf{.1962}/\textbf{.0701} & .2141/\textbf{.0858} & .2347/\textbf{.1074} & .2447/.1161 & 12.36 c \\
combined      & \texttt{ccicciic\,cccii\,icc} & \textbf{.1808}/\textbf{.0537} & \textbf{.1962}/\textbf{.0701} & \textbf{.2178}/.0857 & \textbf{.2386}/.1033 & \textbf{.2590}/\textbf{.1237} & 1.51 i / 9.25 c \\
```

(The combined `emergent order` is `c c i c c i i c c c c i i i c c`; break it wherever the column
width wants. The old string was `ccciccii\,iicciiii`.)

Unrounded, for anyone checking:

| arm | q=1 | q=2 | q=4 | q=8 | q=16 | answered@16 |
|---|---|---|---|---|---|---|
| items-only | .17598/.04341 | .18838/.05626 | .20376/.06767 | .22352/.08598 | .24470/.10171 | 3.108 i |
| concepts-only | .18079/.05373 | .19618/.07012 | .21411/.08577 | .23468/.10740 | .24467/.11607 | 12.360 c |
| combined | .18079/.05373 | .19618/.07012 | .21783/.08566 | .23862/.10334 | .25900/.12370 | 1.512 i / 9.245 c |

**Bolding moved.** Under the policy-A rows the combined arm carried the full-axis bold everywhere and
concepts carried the tail. Now: at q=1 and q=2 the combined order **is** the concepts order (its first
two picks are c114, c173), so those cells are equal by construction and both rows should carry the
same value — that is a nesting fact, not a tie to be adjudicated. Concepts still take the tail at q=4
and q=8 (.0858 vs .0857; .1074 vs .1033), and the combined order takes **both axes** at q=16.

**Movements worth a sentence in the merge:** the items row *falls* at every budget between 2 and 8
(e.g. q=8 .2247 → .2235 full) but *rises* on the tail (.0776 → .0860); the concepts row rises on the
tail everywhere and loses .0026 at q=16 on full; the combined row rises on both axes at every budget
from q=4 (q=16: .2556 → .2590 full, .1098 → .1237 tail). A rebuilt comparator is a *correctly
selected* comparator, not a uniformly better-scoring one.

### The per-pick gains, if `tab:greedy` or its surrounding prose wants them

Validation gains (the selection cohort; the search's own objective is cohort-mean held-out full
NDCG@10 over the 10,000 validation users). Source: `greedy_static_policyB_{combined,items,concepts}.json`.

| # | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 | 16 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **combined** | c114 | c173 | i19 | c748 | c785 | i83 | i583 | c243 | c231 | c809 | c455 | i22 | i77 | i88 | c18 | c838 |
| gain | .0205 | .0163 | .0115 | .0099 | .0070 | .0052 | .0054 | .0037 | .0034 | .0029 | .0027 | .0024 | .0024 | .0021 | .0020 | .0015 |
| **items** | i83 | i7 | i19 | i631 | i2195 | i140 | i22 | i396 | i494 | i343 | i873 | i506 | i212 | i549 | i228 | i135 |
| gain | .0174 | .0140 | .0079 | .0059 | .0054 | .0046 | .0039 | .0039 | .0038 | .0036 | .0027 | .0023 | .0022 | .0027 | .0029 | .0029 |
| **concepts** | c114 | c173 | c785 | c312 | c215 | c377 | c124 | c1010 | c233 | c12 | c912 | c294 | c488 | c467 | c753 | c851 |
| gain | .0205 | .0163 | .0108 | .0081 | .0069 | .0054 | .0039 | .0034 | .0031 | .0021 | .0015 | .0010 | .0009 | .0007 | .0006 | .0005 |

Validation curves at K=16: combined `0.25561/0.12366`, items `0.24281/0.10112`, concepts
`0.24250/0.11809`. Search cost: 2,167 / 839 / 1,546 cumulative evaluations, 106.2 / 52.7 / 134.3 min.

---

## 2. `tab:battery` — replacement rows

Only the three `optimised static` rows change. Every other row in that table is unaffected: the
provenance audit established that no selector row was ever *scored* under policy A (the arm-N harness
has been on the protocol pool since 2026-07-30), so the contamination was only ever the *content* of
these three orders.

```latex
greedy combined (item+concept) & static & \textbf{.1808/.0537} & \textbf{.1962/.0701} & \textbf{.2178}/.0857 & \textbf{.2386}/.1033 & \textbf{.2590/.1237} & 1.5i/9.2c \\
greedy items-only              & static & .1760/.0434 & .1884/.0563 & .2038/.0677 & .2235/.0860 & .2447/.1017 & 3.11 \\
greedy concepts-only           & static & \textbf{.1808/.0537} & \textbf{.1962/.0701} & .2141/\textbf{.0858} & .2347/\textbf{.1074} & .2447/\textbf{.1161} & 12.36 \\
```

**The caption's lower-bound flag is now discharged and must go.** Current text:

> Greedy rows are *lower bounds* — the sequences were selected under a narrower mask
> (Section~\ref{sec:act1}); the rebuilt sequences exist (Section~\ref{sec:act3}) and this table is
> pending their substitution.

Replacement: *"The three greedy rows are the sequences rebuilt under this table's own candidate mask
(Section~\ref{sec:act3}); selection contract and scoring contract now agree, so they are no longer
lower bounds."*

---

## 3. The narrative sentence, re-verified cell by cell

> "The optimised static comparator leads every adaptive selector at every budget, and every one of
> those thirty comparisons separates."

**HOLDS, and it holds three times over.** Reading "those thirty" as the three named adaptive
selectors (GLM-UCB, Thompson, latent-only info-gain) × 5 budgets × 2 axes:

| comparator | static leads | leads AND separates | verdict |
|---|---|---|---|
| greedy combined (policy-B) | 30/30 | **30/30** | HOLDS |
| greedy items-only (policy-B) | 30/30 | **30/30** | HOLDS |
| greedy concepts-only (policy-B) | 30/30 | **30/30** | HOLDS |

Widening "every adaptive selector" to **all twelve adaptive rows in the table** (GLM-UCB, Thompson,
latent-only info-gain, Golbandi tree, CLiME, ConLinUCB-MCR, ConLinUCB-BS, ConUCB, DPE-Hellinger,
DPE-ILD, DPE-concept, IGCN — 118 cells, Golbandi having no k=16):

| comparator | static leads | leads AND separates |
|---|---|---|
| greedy combined | 118/118 | **118/118** |
| greedy items-only | 116/118 | **114/118** |
| greedy concepts-only | 118/118 | **118/118** |

Four exceptions, all on the **tail** axis against the **items-only** order, none of them a win for a
selector:

| cell | delta | 95% CI | reading |
|---|---|---|---|
| Golbandi tree, k=4, tail | −0.00263 | [−0.00553, +0.00028] | static still ahead, does not separate |
| Golbandi tree, k=8, tail | −0.00159 | [−0.00489, +0.00167] | static still ahead, does not separate |
| IGCN, k=2, tail | +0.00060 | [−0.00169, +0.00286] | selector ahead by 0.0006, does not separate |
| IGCN, k=4, tail | +0.00241 | [−0.00015, +0.00497] | selector ahead, does not separate |

So the strongest honest form of the sentence, if the merge wants the wide version:
**against the combined order every one of the 118 comparisons separates in the static order's
favour; against the items-only order 114 of 118 do, and the four that do not are all tail cells that
fail to separate in either direction.** Against the narrow (three named selectors) reading nothing
changes at all.

### 3a. ★ A separate narrative claim BREAKS, and it is the chapter's only adaptive win

Two places say Golbandi's tree takes the tail at eight questions against the strongest static:

* line ~235: *"Against the greedy order, however, it loses full NDCG at every budget ($-.0203$,
  $-.0293$, $-.0203$, $-.0130$), taking only the tail at eight questions ($+.0068$, CI
  $[+.0039,+.0098]$)."*
* line ~589: *"against the strongest static it takes the tail at eight questions ($+.0068$
  $[+.0039,+.0098]$) while losing full at every budget."*

**That $+.0068$ was comparator weakness.** It was measured against the policy-A items order, whose
tail at k=8 is `.0776`; the rebuilt items order scores `.0860` there. Against the rebuilt order:

| Golbandi tree vs rebuilt items-only | k=1 | k=2 | k=4 | k=8 |
|---|---|---|---|---|
| full | **−0.02031** [−0.02268,−0.01794] | **−0.02698** [−0.02983,−0.02401] | **−0.01741** [−0.02054,−0.01425] | **−0.01185** [−0.01523,−0.00853] |
| tail | **−0.00828** [−0.01042,−0.00615] | **−0.00842** [−0.01126,−0.00559] | −0.00263 [−0.00553,+0.00028] | −0.00159 [−0.00489,+0.00167] |

And against the rebuilt **combined** order it loses both axes at every budget, tail k=8
**−0.01894** [−0.02255,−0.01541].

**Consequence for the chapter: no published adaptive selector beats the strong static order on
either axis at any budget.** The one exception Act 1 carefully carved out has gone. The full-NDCG
figures in that sentence also move (`−.0203, −.0293, −.0203, −.0130` → `−.0203, −.0270, −.0174,
−.0119` against items-only). This *strengthens* Act 1's null and removes the "a win over the weak one
is not a crossover" hedge's counterexample — but it must be written, not quietly dropped, because it
is a retraction of a printed separated interval.

---

## 4. Every selector against every rebuilt static order

Full grid (12 adaptive + 2 value-free + DRE + ConLinUCB-BS-static + the oracle ceiling + the six
literature heuristics) × 3 comparators × 5 budgets × 2 axes lives in
`experiments/baselines/interview/a9_battery_policyB.json` under `vs_static`, keyed
`<selector>|K<k>|<axis>|minus|<comparator>`, each with `delta`, `lo`, `hi`, `separates`, `n`,
`pairing`. Selected rows the chapter quotes:

| selector vs **rebuilt items-only**, k=16 | full | tail |
|---|---|---|
| CLiME | **−0.02294** [−0.02639,−0.01946] | **−0.01205** [−0.01544,−0.00871] |
| DRE | **−0.02619** [−0.02938,−0.02304] | **−0.01195** [−0.01492,−0.00895] |
| GLM-UCB | **−0.02808** [−0.03140,−0.02470] | **−0.01778** [−0.02098,−0.01449] |
| IGCN | **−0.02084** [−0.02406,−0.01762] | **−0.01052** [−0.01351,−0.00756] |
| Thompson | **−0.03209** [−0.03543,−0.02873] | **−0.01389** [−0.01703,−0.01074] |
| entropy$_0$ | **−0.01439** [−0.01753,−0.01120] | **−0.00929** [−0.01241,−0.00614] |

---

## 5. Other prose in the chapter that quotes the superseded rows

`grep` over `chapterB_v2.tex`. **Line numbers are from the working tree at 2026-08-12 22:37 and the
file is being edited concurrently — match on the text, not the line number.**

| line | text | what it needs |
|---|---|---|
| 190–192 | `tab:greedy` rows | §1 above |
| 205 | *"our greedy items-only sequence $.2449/.0967$"* | → `.2447/.1017` |
| 234–235 | Golbandi *"taking only the tail at eight questions ($+.0068$…)"* | **§3a — this is a retraction, not a number swap** |
| 371–383 | the two-candidate-pools paragraph: *"the greedy sequence beats entropy$_0$ … $+.0253$ / $+.0174$ under the protocol pool"* | items-only vs entropy$_0$ at q=16 is now **+0.01439 [+0.01120,+0.01753] full / +0.00929 [+0.00614,+0.01241] tail** — the full margin is smaller and **the tail margin nearly halves**; the narrow-mask half of the sentence is unaffected. Combined vs entropy$_0$ at q=16 is +0.02869 full / +0.03128 tail |
| 385–395 | *"greedy items-only $.2449/.0967$ … the best adaptive selector trails by $+.0288$ full and $+.0134$ tail"* plus the *"selected under the narrower mask, so … a lower bound"* caveat and the live `\REDO{a greedy re-run whose selection contract is the protocol pool, plus paired CIs on every gap}` | numbers → `.2447/.1017`, deficit → **+0.02808 full / +0.01778 tail** (GLM-UCB); **the lower-bound caveat and the `\REDO` are both discharged and should be deleted** |
| 486–488 | `tab:battery` caption's *"Greedy rows are lower bounds … pending their substitution"* | §2 above — delete |
| 498–500 | `tab:battery` rows | §2 above |
| 534 | *"The optimised static comparator leads every adaptive selector … thirty comparisons separates"* | **verified, no change** (§3) |
| 544 | *"greedy items-only $.2449$ against the best adaptive selector's $.2161$"* | → `.2447` against `.2166`/`.2161` (see the glm_ucb reproducibility note below) |
| 589–590 | Golbandi *"against the strongest static it takes the tail at eight questions"* | **§3a — retraction** |
| 652 | *"sixteen item questions yield $4.20$ answers where sixteen concept questions yield $12.87$"* | → **3.11** and **12.36**; note the items figure moves a long way, so the "factor of three" phrasing in the same section becomes a factor of **four** |
| 663–664 | *"the coarse and fine channels here differ in answer rate by a factor of three"* | → a factor of **four** (3.11 vs 12.36) |
| 614 | `\PLAN{… and a greedy re-run whose selection contract matches the scoring contract}` | discharged — delete that clause |
| 1051 | *"the two on test ($.25900/.12370$ against $.25565/.10982$)"* | already correct |
| 1307–1309 | Act 4's composition sentence, `c c i c c i i c c c c i i i c c`, *"concepts hold ten of the sixteen slots"* | already rebuilt — this is the string `tab:greedy` must be brought into line with |

**★ Also check, and this one is a broken claim, not a number swap.** `PAPER_B_TODO.md` item 4.6's
"availability ruled out" argument quotes the *policy-A* orders by question id (`combined asks items
22, 8, 112, 83, 130, 21, 19, 219, 80; items-only asks 83, 583, 19, 8, …`) and concludes with *"the
same holds for the root splitter: concept 114 is absent from the combined order's concepts (630, 242,
173, 562, 537, 754, 180)."* **On the rebuilt combined order that is false — concept 114 is its FIRST
pick.** It has to be: A2's contract gate is that the rebuilt order's first question is the tree's root
splitter, which is why the tree's K=1 gain against it is exactly 0.0 on all 20,000 users. Half of 4.6
survives unchanged (item 514 appears nowhere in either rebuilt order — items `83, 7, 19, 631, 2195,
140, 22, 396, 494, 343, 873, 506, 212, 549, 228, 135`; combined's items `19, 83, 583, 22, 77, 88`),
so "the right question existed and everyone had to wait for it" is still ruled out for the depth-2
gain. The root-splitter half must be deleted, and if 4.6 is quoted in the chapter the id lists must be
reprinted from the rebuilt sequences.

**One caveat that travels with two cells:** `glm_ucb` and `thompson` are not seed-stable
(`ACT1_PAIRED_CI.md` — `ials._fit_model` builds the iALS basis with no `random_state`). The vectors
used here are the banked act1 ones, so every interval is correct conditional on that basis, but any
four-decimal quote of a GLM-UCB or Thompson *absolute* carries a ±.0026 basis-to-basis component that
a within-run paired bootstrap cannot see. That is a pre-existing debt, not one A9 introduces.

---

## 6. What this does and does not settle

- **Settles:** Act 1's static rows are now selected and scored under one contract; the "lower bounds"
  hedge, the `\REDO` at line 394 and the `\PLAN` clause at line 614 are all discharged.
- **Settles, against us:** the chapter's single instance of a published adaptive selector beating the
  strong static order (Golbandi's tail at eight) does not survive the rebuilt comparator.
- **Does not settle:** the Act-2 quartile trend cells still difference against the *policy-A*
  `fixedorder:greedy_items` / `greedy_concepts` comparators (`POLICY_A_PROVENANCE_AUDIT.md` §B). Their
  direction is safe by the handicap argument; their magnitudes move. That re-difference is not A9.
- **Does not settle:** budget parity across arms (A5) and the tail-objective greedy (A6) are untouched.
