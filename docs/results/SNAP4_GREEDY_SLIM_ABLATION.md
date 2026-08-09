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
QGSLIM itself was not run: the greedy fit is implemented and verified **exactly** against brute
force (identical 5-round selection, weights to 1.2e-08), but its 16-minute gram did not fit the
session. Resumable command recorded.

## ★ The ablation — the decline is an artefact, and their stated cause is measurably wrong

Their §3.3.2 requires `I_Q ∩ I_R = ∅`, and their §6.5 states: *"the main reason for the decreasing
NDCG is that the questionnaires are not allowed to recommend items that they contain as questions."*
Three outcomes were pre-registered before running. The result is outcome **(b)** — the decline
persists without the rule — **with the sign reversed from their explanation**:

* Their candidate rule is already "all items with `x_ui = 0`", so `I_Q ∩ I_R = ∅` bites **only on
  asked-but-unanswered items**, which offline are guaranteed non-targets.
* **Removing the rule they blame HELPS QBandit by +.034 → +.119** (paired CIs exclude zero at every
  budget, growing with budget). It was **masking** a steeper decline: without it QBandit falls
  **−46%** instead of −14%.

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
the authors' diagnosis of its cause is measurably false. The honest line is: **the decline
reproduces; the stated cause does not.**

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
