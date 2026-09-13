# Recency check for Paper B: is there newer work that scoops or threatens us? (2026-09-13)

Prompted by the supervisor's comment on the ECIR draft: *"check there are def def no newer relevant
papers!!! baselines seem pretty old"*. The draft's evaluated battery runs 2008–2025 with only one arm
newer than 2021 (CLiME, KDD'25), so the concern is fair on its face.

**Scope of this pass.** Three targeted searches aimed at the most likely threat — an attribute-aware or
decision-tree elicitation paper from 2024–2026 that already does what our C1/C2 claim. **Not an
exhaustive venue sweep**; a full 2024–2026 pass over SIGIR / RecSys / WWW / CIKM / KDD / ECIR is still
owed.

## ⭐ The one direct hit, and it HELPS us

**Gharahighehi, Nakano, Yang, Cu & Vens, "Pairwise and Attribute-Aware Decision Tree-Based Preference
Elicitation for Cold-Start Recommendation", arXiv:2510.27342, 31 Oct 2025 (KU Leuven).**

This is the closest published work to our C1 and it is *newer than anything we cite*. It extends the
decision-tree elicitation family with exactly two ideas, one of which is ours:

- (i) elicit preferences on **attributes** (genres), not only items — our concept channel;
- (ii) query **item pairs** rather than single items at each node — not ours.

It reports that both help, "particularly with a reduced number of queries".

**Why it does not scoop us, verified from the full text:**

| | Gharahighehi et al. 2025 | Paper B |
|---|---|---|
| metric | **RMSE only** — *"since it is vastly employed in the literature"*; no NDCG, no ranking metric | full + tail NDCG@10 |
| split criterion | **prediction error** E(j), squared error over lovers/haters/unknowns | the measured metric itself |
| comparators | `helf0_hybrid` and **their own** single-item tree variant | 13 published methods + a metric-optimised greedy order |
| scale | Yahoo! Music, **1,701 users / 4,793 artists / 218 genres** | 18,359 items / 10,000 test users / 1,031 concepts |
| budget | fixed 20 iterations | 1–16, reported at every budget |
| answers | simulated from historical logs | behavioural, from held-out ratings |

**So the newest work in our exact space is itself an instance of both faults we diagnose:** it optimises
and reports a surrogate (prediction error / RMSE) rather than the ranking metric the field reports, and
its only external comparator is HELF — a 2008 heuristic. It is direct evidence for C3 and C4, not a
threat to them.

**What it does cost us:** we can no longer present "elicit on attributes as well as items" as new. It
was published in Oct 2025. Our C1 must be restated as what is actually ours — not *that* attribute
questions help, but **why**: the answer rate (12.36 answers per 16 concept questions against 3.11 for
films), measured on a ranking metric at catalogue scale, against a comparator that maximises that
metric. Gharahighehi et al. should be cited in Related Work as the nearest and newest neighbour, and
again as an example under the weak-comparator observation.

## Checked and not threatening

- **PERE** — Nguyen et al., *Cold-start Recommendation by Personalized Embedding Region Elicitation*,
  UAI 2024. Already cited (`nguyen2024pere`). Reports NDCG@10 gains of 3–10% relative over prior
  elicitation methods; a region-based learned-static successor, already in our battery's family.
- **Greedy SLIM** — Proissl et al., arXiv 2406.06061 (2024). Already cited and already audited: our
  closest protocol sibling (ML-25M, NDCG@10, same tail definition), gate-failed in part and not run.
- **"Cold-Starts in Generative Recommendation: A Reproducibility Study"**, SIGIR'26. Adjacent but a
  different problem — generative/item cold start, not question selection. Worth a citation in the
  reproducibility framing at most.

## Outstanding

A full 2024–2026 venue sweep (SIGIR, RecSys, WWW, CIKM, KDD, ECIR, UMAP) has **not** been done. This
pass targeted the highest-risk shape only. Until that sweep runs, the honest internal statement is
"no scoop found on the attribute-tree axis", not "there is nothing newer".

---

# SECOND PASS (same day): the sweep was NOT complete. Three more, one MATERIAL.

The first pass searched only the attribute-tree axis and concluded "one hit". That was wrong. Four
further searches on the CRS / NDCG-in-the-selector axis turned up the following.

## ⛔ MATERIAL — a published greedy NDCG attribute selector, three years old, uncited by us

**Lin, Zhu, Wang & Caverlee, "Enhancing User Personalization in Conversational Recommenders",
arXiv:2302.06656, TheWebConf (WWW) 2023.**

From the abstract, verbatim: *"(i) a **greedy NDCG attribute selector**, to enhance user
personalization in the interactive preference elicitation process by prioritizing attributes that most
effectively represent the actual preference space of the user"*.

This overlaps the core idea of C2 — put NDCG inside the question-selection rule — and it is
**attribute-based**, so it touches C1 as well. It is not in our bibliography. A reviewer who knows this
paper and sees our claim that *"Unoccupied is the conjunction"* and *"exactly one rule optimises the
reported metric"* would have grounds to reject.

**What almost certainly still distinguishes us, NOT YET VERIFIED from full text:**

- their setting is **multi-round conversational recommendation** with a multi-groundtruth target item,
  not cold-start onboarding scored by a held-out ranking metric after a fixed question budget;
- their NDCG is, on the face of it, an **expected/estimated** quantity over the model's current belief
  within a session, where ours is a **measured cohort-mean held-out** NDCG@10 obtained by folding real
  answers through a frozen recommender and ranking the full catalogue;
- their comparators are **other conversational recommenders**, not a metric-optimised static order.

⚠ **The deciding fact — measured versus estimated NDCG — could not be read from the abstract and MUST be
checked in the full text before any novelty sentence is written.** If their NDCG is also measured and
held-out, our conjunction claim needs rewriting, not just narrowing.

## Supporting, not threatening

- **Harne, Modani, Mahapatra & Agarwal, "Dialogue to Discovery: Attribute-Aware Preference Elicitation
  for Conversational Product Search Assistants", arXiv:2606.24194, 23 Jun 2026.** Attribute elicitation
  for target-finding product search. Criterion is **uncertainty x entropy** (APU x ACE) — another
  surrogate. Catalogues of ~2,000 items, LLM baselines (BM25, SBERT, zero-shot LLM). Different problem,
  and further evidence that attribute selection is still driven by surrogates. Good citation for C3.
- **"Minimizing Live Experiments in Recommender Systems: User Simulation to Evaluate Preference
  Elicitation Policies", arXiv:2409.17436 (2024).** Evaluation methodology for elicitation policies;
  relevant to our contract framing.
- **Soft-attribute elicitation** (arXiv:2311.02085; concept activation vectors, arXiv:2202.02830).
  Adjacent to the concept channel; not yet read.

## Status

The sweep is **still not complete** — these two passes covered the attribute-tree and CRS/NDCG axes.
Not yet covered: bandit/active-learning elicitation 2024–2026, LLM elicitation, UMAP/CIKM/ECIR venues.

---

# THIRD PASS (2026-09-13): the remaining axes. Nothing competing.

The first two passes covered attribute-trees and CRS/NDCG selectors and each found something material
(Gharahighehi 2025, Lin WWW'23 — both now cited in Paper B). This pass covers what was left.

**Bandit / active-learning cold start, 2025–26.** The recent work is overwhelmingly *item* cold start
rather than user onboarding: MoToRec (AAAI'26), M2VAE (AAAI'26), MI4Rec (CIKM'25), epistemic-uncertainty
cold start (CIKM'25), item-centric exploration (RecSys'25). Adjacent but a different problem — none
selects questions for a new user or reports a post-interview held-out ranking. Also seen:
*Active learning-based multi-armed bandits for recommendation systems* (KAIS 2025) and
*Batch versus Sequential Active Learning for Recommender Systems* (arXiv 2201.07571), neither of which
compares against a metric-optimised question order.

**LLM elicitation, 2025–26.** Also non-competing on our contract. Nearest items:
- **PrefDisco** (Li et al., 2026): frontier models fail at proactive elicitation, 29% of attempts
  *worsen* alignment against generic responses. A useful citation for the LLM-elicitation sentence, not
  a rival method.
- *A Human-Centred Framework for Retrieval-Grounded Cold-Start Recommendation* (CEUR) and
  *Supporting High-Stakes Decision Making Through Interactive Preference Elicitation in the Latent
  Space* (ICLR'26): interactive elicitation, but not cold-start question selection scored on a full
  catalogue.
- *Minimizing Live Experiments in Recommender Systems* (arXiv 2409.17436, 2024): evaluation
  methodology for elicitation policies; relevant to our contract framing, already logged in pass two.

**Verdict.** Three passes. Two material finds, both folded into the paper. On the remaining axes
nothing was found that selects cold-start questions and measures the result on a held-out ranking
metric against a strong comparator. The honest statement is now "swept on four axes, two hits, both
cited" rather than "there is nothing newer" — venues were searched by topic, not enumerated
proceeding by proceeding.
