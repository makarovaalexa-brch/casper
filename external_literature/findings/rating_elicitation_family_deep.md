# The rating-elicitation / interview-tree / seed-set family — EXHAUSTIVE pass (2026-08-05)

*Sibling to `adaptive_elicitation_baselines.md` (2026-08-04), which surveyed this family at abstract
depth. This file is PASS 2: origins-to-2026, implementation grade, with the corrections that pass 1 got
wrong. **Read this file's §0 CORRECTIONS before acting on anything in the 08-04 file.***

Verification tags on every claim: **[V]** = read from full text I extracted myself · **[A]** = abstract /
landing page only · **[S]** = search-summary or third-party only. Nothing tagged [S] may be presented as
verified. Where a fetch failed it says so.

OUR CONTRACT (a baseline fits or is excluded with a stated reason): ML-25M, binary implicit (ratings
>3.5), Liang strong-generalization split (140,768 / 10,000 / 10,000 users), COLD user at interview start,
budget k ∈ {1,2,4,8,16}, questions are ITEMS (answered from real held-out ratings) or GENOME CONCEPTS,
metric full AND tail NDCG@10 on held-out items, no LLM, no simulator.

---

## 0. CORRECTIONS to the 2026-08-04 survey — read first

1. **The single biggest miss: GREEDY SLIM (arXiv:2406.06061) runs on ML-25M, reports NDCG@10, AND uses
   our exact tail definition.** [V] It is the closest contract match in the entire literature — closer
   than DRE (ML-20M, k=50) or PERE (ML-10M/20M, k=100). Its long-tail split is *verbatim* ours:
   "Following Cremonesi et al. (2010), we refer to the set of most popular items that represent 33% of
   the ratings as short-head items. The remaining items are called long-tail items." That is the same
   sentence our own tail mask implements. Pass 1 never saw this paper. Details §2.1.

2. **Greedy SLIM also publishes, on our dataset and our metric pair, the exact Act-1 result — and it
   goes our way.** [V] Static Greedy SLIM vs the adaptive QBandit (= Christakopoulou KDD'16 Thompson
   sampling, already `christakopoulou2016towards` in our INDEX), ML-25M NDCG@10:
   | k | GSLIM (static) | QBandit (adaptive) | GSLIM tail | QBandit tail |
   |---|---|---|---|---|
   | 5 | 0.3390 | 0.3382 | 0.1163 | **0.1183** |
   | 10 | **0.3594** | 0.3456 | **0.1543** | 0.1465 |
   | 15 | **0.3709** | 0.3221 | **0.1674** | 0.1488 |
   | 20 | **0.3752** | 0.2916 | **0.1779** | 0.1466 |
   The adaptive arm ties (full) or *wins* (tail) at k=5 and then **degrades monotonically** while the
   static arm keeps climbing. This is external, published, same-dataset support for Act 1's claim, and
   it also independently reproduces our own "adaptive wins only when evidence is scarce" shape.
   **Do not order these numbers against ours** (explicit ratings, no Liang split, SLIM recommender) —
   cite the *pattern*, run the *method*.

3. **The "PERE gives 3–10% relative NDCG@10" figure is NOT IN THE PAPER.** [V] Full text contains zero
   `%` characters. Recomputed against c-DPP, the nearest baseline, the real margin is **+0.59% / +1.15% /
   +2.38% / +1.25% / +3.94%** on five of six settings (only real-user ML-20M reaches +16.9%). The
   08-04 file carried this as [S]; it is now **refuted** and must be struck.

4. **PERE's own DRE column is ~10× below DRE's published numbers** (0.0382 vs DRE's own 0.5712-class
   results). [V] This is the `surprise-means-debug-the-harness` signature. **Never cite PERE's DRE row
   as evidence about DRE.**

5. **Sepliarskaia does NOT say what we lean on it for.** [V] Author list is **Sepliarskaia, Kiseleva,
   Radlinski, de Rijke** (Kiseleva was missing). **Golbandi is cited but never compared against** — the
   only tree in its experiments is PWDT (Rokach & Kisilevich 2012). Its finding is narrower and its
   stated cause is different: trees lose *because they optimize a variance surrogate instead of the true
   loss*, not because they are dynamic, and **everything ties below 5 questions** ("All algorithms
   provide the same performance for questionnaires with a length less than 5", p>0.1). At our budgets
   (k≤16, and k≤4 especially) this paper mostly supports "no measurable difference", not "static wins".
   Also: PEO in PERE's baseline table **IS this same paper** — same algorithm, new acronym.

6. **PASS 1'S HARNESS AUDIT (§8b) IS STALE — verified in code today.** It said "Act 1 currently has ZERO
   adaptive question selectors". That is no longer true. `src/baselines/adaptive_harness.py` (401 lines)
   now implements the sequential per-turn selector loop with a byte-identical regression contract against
   the static path, and registers **`glm_ucb`, `thompson`, `eddi_ig`** selectors, backed by
   `bandit_selectors.py` (GLM-UCB + Thompson over the `belief_mf` Gaussian posterior) and
   `eddi_selector.py` (EDDI single-order info gain). `rbmf_seed.py` and `golbandi_native.py` also exist.
   **Still genuinely missing:** a Golbandi adaptive *tree* selector (the split search), IGCN in adaptive
   re-clustering form, DRE, RMVA, fMF, PERE, Greedy SLIM.

7. **A whole recent adaptive-interview line was missed entirely: the Trinity College Dublin group
   (Rajapakse & Leith).** MCTS RecSys'22 → TORS'23 → RLT4Rec 2024, plus its parent Cluster-Based Bandits
   SIGIR'21. Pass 1 had RLT4Rec with "authorship unverified"; it is **Dilina Chandika Rajapakse and
   Douglas Leith, Trinity College Dublin** [V, from the PDF]. This line is UNFIT for us, but for a
   *stronger* reason than pass 1 gave — see §3.

8. **RBMF's author list in our notes was wrong.** It is **Nathan N. Liu, Xiangrui Meng, Chao Liu, Qiang
   Yang**, RecSys'11 pp. 37–44 [V].

9. **DRE has no duplicate-seed guard.** [V] Its k rows of Π are independent categoricals over the whole
   itemset and extraction is a per-row argmax — nothing prevents two rows collapsing onto the same item,
   silently shrinking the effective budget. Anyone reimplementing it must add a guard, and this is a
   legitimate, cheap criticism to make in print.

---
