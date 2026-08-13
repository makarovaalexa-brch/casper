# `tab:battery` — the EDDI rows, for the merge (TODO 1.3(b))

**Deliverable for the writing agent. Nothing in `chapterB_v2.tex` was edited by this run.**

Written 2026-08-13. Full result + gates: `docs/results/EDDI_THREE_OUTCOME_RESULT.md`.
Raw `experiments/baselines/interview/adaptive_table_eddi3_policyB.json`; intervals
`experiments/baselines/interview/a9_battery_policyB_eddi3.json`.

Contract identical to the rest of the table: 10,000 TEST users, certified pair, policy-B pool,
NO-FLOOR, behavioural answers, learned decoder bias, no sampling. Paired percentile bootstrap,
10,000 resamples, seed 98765.

---

## 1. The existing row is UNCHANGED — do not touch it

`latent-only info-gain$^\dagger$` keeps `.1537/.0336 · .1539/.0373 · .1618/.0432 · .1747/.0611 ·
.1917/.0749` and `3.84`. This run re-derived that arm from scratch and reproduced the banked
per-user vectors at **`0.0e+00`**, plus `answered=3.84/16` and `1{,}993` distinct ask-prefixes. The
row and its `tab:prefixes` entry are confirmed, not replaced.

## 2. ONE NEW ROW to add, immediately below it

Carry **`zero`**. Reason in §4 — it is the faithful instantiation and it is also never worse than
the alternative, so the principled pick is not the flattering one.

```latex
EDDI three-outcome$^\ddagger$~\cite{ma2019eddi} & adaptive & .1607/.0363 & .1597/.0388 & .1732/.0497 & .1869/.0660 & .2041/.0834 & 2.86 \\
```

`tab:prefixes` companion row, if that table is extended: `1 & 2 & 8 & 112 & 1{,}217`.

Suggested `$^\ddagger$` footnote (or fold into the existing `$^\dagger$` note):

> $^\ddagger$ The same reduced acquisition as the row above, under a three-outcome response model
> (no answer / like / dislike) rather than a two-outcome one. It is **not** the published criterion
> either: Eq. 9's target term is still absent, and the row is retained for the mechanism comparison,
> not as a replication.

## 3. The sentence that was resting on a smoke — replacement text

Current (chapterB_v2 §~lines 728–739) reports the three-outcome variant as measured, backed by a
200-user smoke stamped not-citable whose two sub-variants disagreed on the tail. That is now a real
measurement and the sentence can be made unqualified **on the answer count**, but it needs the tail
qualification:

> It does. On the full cohort the three-outcome variant collects **2.86** answers in sixteen
> questions against the two-outcome arm's **3.84** — a quarter fewer — and ranks better from them:
> the paired difference is positive and excludes zero at **every** budget on full accuracy
> ($+.0070$, $+.0057$, $+.0114$, $+.0122$, $+.0124$) and at four of five on the tail (the two-question
> cell is a null). Pricing the event that no answer arrives is worth roughly a quarter of this arm's
> deficit against the loss-optimised static order on full accuracy and about a third on the tail —
> and it closes neither.

**Delete** the clause about the two sub-variants disagreeing on the tail: at 10,000 users they do
not. They score .0660 and .0649 at eight questions, both above the two-outcome arm's .0611.

The **`\REDO` at line 701 can now be discharged** — but only the (b) half of it. The (a) half (full
Eq. 9) is still open and the `$^\dagger$` paragraph at lines 715–726 must stay exactly as it is.

## 4. Why `zero` and not `signed` — for the author, in case it is queried

The frozen Partial-VAE has no learned token for a disliked observation, so `signed`'s
$c_O - \Phi_i$ is a documented approximation borrowed from arm N, while `zero` lets the third
outcome enter only through the answerability weight — which is what the Partial-VAE's
absence-not-a-value premise licenses. It is also **never worse**: identical at k=1, within .00005 on
full at k=16, ahead on full at k=4 and k=8, ahead on tail at every budget from k=2, and it asks
fewer questions (2.86 vs 3.16). `signed`'s numbers, if ever needed:
`.1607/.0363 · .1607/.0383 · .1655/.0442 · .1852/.0649 · .2041/.0815`, answered 3.16.

## 5. Narrative sentences — re-verified with both new arms included

Quantifying over **all** adaptive rows (now 138 cells, the two EDDI variants included): **138/138**
lead-and-separate for the static order against the rebuilt combined order, **138/138** against
concepts, **134/138** against items. The four items exceptions are the **pre-existing** Golbandi
k=4/k=8 tail and IGCN k=2/k=4 tail cells, none of which separates in either direction. **Neither
EDDI variant introduces a new exception**, so *"no published adaptive selector beats the strong
static order on either axis at any budget"* still holds — now against a better-instantiated EDDI.
