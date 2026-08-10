# SNAP-4 Greedy SLIM: the published adaptive decline IS an artefact — but not the one its authors name

Run 2026-08-09/10. Target: Proissl, Vatić & Waldschmidt, *"Greedy SLIM: A SLIM-Based Approach For
Preference Elicitation"*, **arXiv:2406.06061, CoRR only — DBLP confirms no venue**; its user study
is n=103, "many of them somehow linked to the authors". Their setup rebuilt exactly: 162,541 users
/ 59,047 items / 25,000,095 ratings, 90/10 split → 16,254 test users, graded gain `2^r−1`, IDCG
over `I_R`, Cremonesi 33% head = 296 items, PureSVD consumer.
**Nothing here is orderable against our ruler** (0.3496/0.2491) — the snap lives entirely inside
their contract.

Published, ML-25M NDCG@10 at k=5/10/15/20: QGSLIM (static) **.3390/.3594/.3709/.3752**;
QBandit (adaptive, Christakopoulou KDD'16) **.3382/.3456/.3221/.2916**.

## Snap verdict: PARTIAL, and the gap is a recommender gap

Our QBandit: full .3142/.3035/.2841/.2696, tail .0823→.1135. **The shape and the decline magnitude
reproduce** — ours falls −.0446 from its peak, theirs −.0540. The level sits ~.03 low, and the
diagnosis is that this is a *consumer* gap rather than a protocol one: **a zero-question
Most-Popular canary already scores .2960 in their setup**, so their k=5 number is only +.042 above
popularity. Their code is an OSF view-only link, so the recommender cannot be matched. Not tuned to
close it — raising SVD rank makes it measurably *worse* (d=200 → .0992).
~~QGSLIM itself was not run~~ — **RUN 2026-08-10**, see the section below. (The greedy fit had been
implemented and verified against brute force — identical 5-round selection, weights to 1.2e-08 —
but that verification was an ad-hoc in-session snippet and **no committed test re-runs it**; it is
not on disk and not in git history. Treat it as unreproduced until a test exists.) The gram cost
**1.0 min, not 16** (11.2M nnz kept at the exact λ₁/2 bound), and all 12 cells finished in 7 min.

## ★★ QGSLIM's own cells — RUN 2026-08-10. Verdict: PARTIAL (diagnosed); TAIL effectively SNAPS

`native_qgslim` label; `experiments/baselines/greedyslim/native_qgslim.json`, 12 cells, per-user
vectors banked. Selected order (20 rounds): `[2480, 584, 898, 13244, 1219, 10406, 534, 21787, 6588,
93, 1358, 1334, 35, 4670, 159, 16703, 999, 948, 4342, 20607]`.

**ARM C (their protocol), against the published targets:**

| k | ours full | published | Δ full | ours tail | published | Δ tail | rev/u |
|---|---|---|---|---|---|---|---|
| 5 | .2898 | .3390 | **−.0492** | .1069 | .1163 | −.0094 | 1.11 |
| 10 | .3270 | .3594 | −.0324 | .1477 | .1543 | −.0066 | 1.74 |
| 15 | .3447 | .3709 | −.0262 | .1619 | .1674 | −.0055 | 2.25 |
| 20 | .3518 | .3752 | −.0234 | .1767 | .1779 | **−.0012** | 2.57 |

**The tail axis snaps.** Every tail cell is inside −.0094, and k=20 is −.0012 — flat exact. On this
project's own calibration (Mult-DAE dropped for −.0221; DRE passed at −.0095) the **tail passes at
every budget**. The **full axis misses by −.023 to −.049 and the miss is largest where the answer
rate is lowest** — which is the diagnosis, not a coincidence: at k=5 only 1.11 of 5 questions are
answered (22%), and QGSLIM's recommender is `x'W` with only the k selected rows of W non-zero, so a
user who answers nothing scores an **all-zero vector and ranks at chance**. Our k=5 full (.2898)
sits **below the zero-question MostPop canary (.2960)** — the paper's §6.5 prediction, confirmed
from the other side. Either their answer rate is far higher than ours or they have an unstated
fallback. **Consumer gap, as before; not tuned to close it.**

### ★ ARM A control — the STATIC arm's shape never changed, so ARM A is not what fixed QBandit

The natural control for the ablation finding. Shape in k, full NDCG@10:

| arm | QGSLIM (static) | QBandit (adaptive) | QPop |
|---|---|---|---|
| C_theirs | .2898→.3518 **RISES** (+.0620) | .3142→.2696 DECLINES (−.0446) | DECLINES (−.0039) |
| B_no_ask_excl | .2803→.3415 **RISES** (+.0612) | .2802→.1509 DECLINES (−.1293) | DECLINES (−.0679) |
| A_heldout | .1362→.1793 **RISES** (+.0431) | .1637→.1936 RISES (+.0299) | RISES (+.0321) |

**Answer: no, the static arm does not change shape.** QGSLIM rises monotonically under **all three**
arms, including the two where both adaptive/popularity arms decline. ARM A moves QGSLIM's LEVEL
(.29 → .14, as it moves everything) but not its SHAPE, because QGSLIM never had the pathology to
begin with. That sharpens the ablation's conclusion: the decline is not a property of the metric or
of the exclusion rule as such — it is what happens when a questionnaire **chooses questions the user
is likely to rate highly and thereby eats its own targets**. A fixed, user-independent order cannot
do that. (Levels are not comparable across arms; only shape is.)

Secondary: under ARM A, QGSLIM is *below* QBandit on full at every budget (.1362/.1625/.1752/.1793
vs .1637/.1761/.1857/.1936) but *above* it on tail (.0425/.0642/.0723/.0801 vs
.0315/.0375/.0415/.0456) — the static arm trades head accuracy for tail coverage.

## ★ The ablation — the decline is an artefact, and the mechanism they name is not the one doing the damage

Their §3.3.2 requires `I_Q ∩ I_R = ∅`, and their §6.5 states: *"the main reason for the decreasing
NDCG is that the questionnaires are not allowed to recommend items that they contain as questions."*
Three outcomes were pre-registered before running. The result is outcome **(b)** — the decline
persists without the rule — and the sign of the rule's effect is the opposite of what their
explanation implies:

* Their candidate rule is already "all items with `x_ui = 0`", so `I_Q ∩ I_R = ∅` bites **only on
  asked-but-unanswered items**, which offline are guaranteed non-targets.
* **Removing the rule they blame HELPS QBandit by +.034 → +.119** (paired CIs exclude zero at every
  budget, growing with budget). It was **masking** a steeper decline: without it QBandit falls
  **−46%** instead of −14%.

**The distinction, stated fairly (refined 2026-08-10 after author challenge).** Both explanations
are about "asked items disappearing", but from **two different sets**:

* the **candidate pool** — what the recommender may rank. This is the rule their §6.5 names. In
  their setup the pool is *already* restricted to items the user has not rated, so an **answered**
  item is excluded anyway; their extra rule removes only asked-**unanswered** items, which offline
  are guaranteed non-targets. Deleting non-targets from the pool is nearly free — and indeed
  turning the rule OFF makes the decline *worse*, which is impossible if it were the cause.
* the **target set** — the held-out items the metric scores against. This is what actually shrinks,
  because a test user has **no held-out split**: their ratings are simultaneously the answers and
  the ground truth, so every answered question converts a scorable target into a known item.

So the authors identify the right **symptom** (asked items vanish) and the wrong **mechanism**
(they blame the pool; the damage is in the targets). The two are separable and we separate them.
The fair summary is therefore *"right symptom, wrong set"* — not "their cause does not reproduce",
which overstates the disagreement.

**The real cause, isolated:** their design gives a test user **no held-out split**, so every
answered question deletes one of that user's own high-gain targets. Under **ARM A** — a fixed 20%
held out *before* the interview, so the ground truth is arm-independent — **the decline disappears
entirely**: QBandit goes .1637 → .1761 → .1857 → .1936, monotone on both axes, and QPop likewise.

**Control:** at a matched answer rate (7.68 vs 7.83 revealed per user) QPop declines only −.012
against QBandit's −.045, so **adaptivity carries the effect** — it is not a popularity artefact.

## ★★ Why this matters beyond one paper: ARM A *is* our policy B

The protocol that removes the artefact is the one this project adopted independently on
2026-08-05 (`docs/results/MASK_POLICY_B_MIGRATION.md`): a candidate pool that is a property of the
**protocol**, not of what any arm happened to ask. So Greedy SLIM is a **published case study** for
the protocol paragraph of this chapter: a real, published, adaptive-degrades-with-budget curve that
is manufactured by a protocol choice and vanishes under ours.

It also changes how the paper may be cited. **A previous entry in
`external_literature/findings/candidate_masking_protocol.md` recorded the decline as "a protocol
artefact by its own authors' diagnosis". That is now corrected**: the decline *is* an artefact, but
the mechanism they name is not the one doing the damage. The honest line is **"right symptom, wrong
set"** — see the fair statement above.

## A control caught a real thing (worth keeping)

All 20 QPop items are head items, so its per-user **tail** delta must be exactly `0.0e+00`. It is —
for all 16,036 scorable users. QBandit's was `0.0000` *in the mean* but non-zero for one or two
users, because Thompson sampling draws a tail item on 0.06–0.13% of asks. The first version of the
check asserted "zero everywhere", printed CONTROL FAILED, and only the **per-user** comparison
revealed why. **A matching mean was, once again, not a licence to difference.**

## Engineering note

Batched `np.linalg.inv` costs **4.38 s** on a (256,50,50) stack against Cholesky's **0.0067 s** —
it projected 4.5 hours for a single interview, 3.6 minutes once rewritten on Cholesky. Recorded so
no future selector reaches for `inv` in an inner loop.

## Owed

* QGSLIM's own cells (the static questionnaire) — implemented and brute-force-verified, not yet run.
* The port of the static questionnaire to our ruler, if we want its row beside DRE's.
