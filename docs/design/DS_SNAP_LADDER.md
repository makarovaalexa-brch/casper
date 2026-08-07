# Design sheet: the SNAP LADDER — which baselines can be validated against published numbers

Author question 2026-08-05: *"a list of baselines we could replicate on their rec, but the same
dataset, with snapping to published numbers — on theirs and then on ours too."*

**The rule this serves (HARD RULE 2):** begin with the named prior and **match its reported numbers
before building anything on top**; only trust a delta once its baseline snaps. The precedent is
Paper A's own **Mult-DAE drop**: it reached NDCG@100 0.3969 against a published 0.419 on the ML-20M
bridge, failed the snap by −0.022, and was **removed from the paper** rather than reported with an
excuse. That is the standard applied below.

## The three-step protocol per baseline

1. **NATIVE SNAP** — reimplement, run in *their* setup (their dataset, their split, their metric,
   their recommender), compare to the published number. Gate: within their reported precision, or
   a stated and diagnosed gap. **A baseline that fails its native snap does not enter the paper.**
2. **PORT** — same selector, our ruler (canonical Liang ML-25M, policy-B candidate mask, behavioural
   answers, full+tail NDCG@10), still with **their** recommender as consumer. This isolates
   "their method on our data".
3. **CROSS** — same selector, **our** certified tower as consumer. This isolates "their method on
   our instrument", and is the cell Act 1 actually needs.

Every row must also carry its **own paper's static baseline** re-run in steps 2–3, or the
comparison silently swaps comparators — the failure mode this chapter is about.

---

## Tier 1 — SAME DATASET FAMILY (ML-20M/25M) + a ranking metric ⇒ a real snap is possible

| method | their consumer | their setup | published number | snap feasibility | effort (agent-h) |
|---|---|---|---|---|---|
| **Greedy SLIM** | SLIM | **ML-25M**, NDCG@10 full **and their long tail = our head-mask definition**, 90/10 user split, graded gain `2^r−1`, asked items excluded from recommendations | static: **.3390/.3594/.3709/.3752** at k=5/10/15/20; QBandit: .3382/.3456/.3221/.2916. Tail static .1163/.1543/.1674/.1779; QBandit .1183/.1465/.1488/.1466 | **HIGHEST — same dataset, same metric, same tail definition.** Differences to absorb: 90/10 vs Liang split, graded gain, asked-item exclusion (their §3.3.2). Run their protocol first, then ours | 12–20 (+6 compute) |
| **CLiSK / CLiME** | conversational bandit over item factors | **ML-25M**, genres as key-terms, cumulative regret | regret curves (figures) | **MEDIUM** — same dataset, but the metric is regret, not post-interview ranking. Snap the *regret curve* to validate the implementation, then port the selector | 8–12 |
| **DRE** | its own Gumbel-Softmax seed decoder | **ML-20M**, NDCG@10, binarised >3.5 (our binarisation), random 80/20, k=50 | \REDO verify the exact table — one secondary source implies a ~0.57-class NDCG@10, which must be confirmed from the paper before use | **HIGH** — public PyTorch, our binarisation. Needs ML-20M rebuild (we already have a Liang ML-20M bridge from Paper A) | 6–10 |
| **RMVA** | maxvol on item factors | **ML-20M**, budgets 5–100 | needs extraction | **HIGH** — ~50 lines, named baseline of both DRE and PERE | 4–6 |
| **DPE (topic bandit)** | its own | **ML-20M** | needs extraction | **MEDIUM-HIGH** — genome gives the item->topic map free. **NOT a concept-channel rival (CORRECTED 2026-08-07):** its bandit picks a topic, its QUESTION is an item. Also: the paper reports NO scalar numbers, only curves, so only orderings are snappable | 4–6 |
| **PERE** | frozen CF embedding space | ML-10M/20M, NDCG@10, budget 100, **users synthesised from the CF embeddings** | margins over its own c-DPP: +0.59/+1.15/+2.38/+1.25/+3.94%; +16.9% on real-user ML-20M | **LOW — SNAP AT YOUR PERIL.** Its DRE column is ~10× below DRE's own numbers, i.e. **their harness is broken for at least one baseline**. Snapping to a broken harness is worse than not snapping. Report as "could not be validated" if it fails | 10–20 |

## Tier 2 — DIFFERENT DATASET, but a published number we can reproduce in *their* setup first

The snap happens on their data (validating our implementation), then the selector ports to our ruler.

| method | their consumer | their setup | snap target | effort |
|---|---|---|---|---|
| **Golbandi ternary tree** | his node/leaf model | Netflix, **RMSE** | his RMSE table. **We already have a partial precedent**: `scripts/_archive/paper1_v1/golbandi_tree.py` replicated the *shape* on ML-100k (RMSE 0.9804 vs static 0.9885, adaptive winning at every depth ≥1) | 8–14 |
| **fMF** | its own latent leaves | ML/Netflix, **RMSE** | its RMSE table. ⚠ **paper bug**: Algorithm 2 prints `argmax` where Eq. 4 says `argmin` — do not code from the pseudocode | 8–14 |
| **IGCN** | user-cluster info gain | ML, RMSE/MAE | its table; must include the 150-NN re-computation per 15-item batch or it is not IGCN | 5–8 |
| **RBMF (maxvol selector)** | its own fold-in | **P@10**, budgets 5/10/20 (= ours) | its P@10 table | 3–5 |
| **MeLU candidate score** | MeLU | ML-1M | its evidence-candidate result **and its user-study aside** (popularity candidates chosen ~2× as often as theirs — our answerability collapse, in humans) | 5–8 |
| **ICF (4 policies)** | frozen MF + Gaussian posterior | cumulative precision/recall/nDCG over the probe sequence | its policy ordering — note **TS loses to LinUCB/GLM (43.7 vs ~46.4)** in their own experiment, contrary to how it is usually cited. Snapping the *ordering* is the achievable target | 4–6 |
| **EDDI** | Partial-VAE | UCI tabular (California housing etc.) | **we already have `scripts/_archive/paper1_v1/eddi_replicate.py`, reported as still runnable** — cheapest snap on the list, validates the permutation-invariant set encoder end-to-end | 2 |

## Tier 3 — NO SNAP POSSIBLE (state the reason, never force a comparison)

NICF, RLT4Rec, MCTS/CBB (cumulative reward or group-identification accuracy; no post-interview
ranking) · EAR/SCPR/UNICORN/ConTS/CRM (omniscient target-item simulator, SuccessRate@turn) · PEBOL
(100-item catalogue, LLM-simulated users — only its *acquisition* ports, and there is nothing to
snap to) · Bıyık soft attributes (16 simulated users; its "NDCG" is agreement with the simulator's
own embedding) · BED-LLM (its posterior *is* the LLM).

## Consumers we have already snapped (Paper A precedent, reuse don't redo)

RecVAE, EASE, EDLAE, Mult-VAE all carry certified full-profile numbers on the canonical split;
**Mult-DAE failed its snap and was dropped**; iALS is kept as an *uncertified floor* (0.358 vs WMF's
published 0.386) and excluded from any claim needing a certified comparator. These are the consumer
axis of the factorial — no new snapping needed.

## Recommended order (snap value per hour)

1. **EDDI tabular replicate** (2 h) — nearly free, validates our set-encoder acquisition path.
2. **RBMF P@10** (3–5 h) — small, our exact budgets.
3. **DRE on ML-20M** (6–10 h) — public code, our binarisation, the peer-reviewed static bar.
4. **ICF policy ordering** (4–6 h) — cheap, and it corrects a widely mis-cited result.
5. **Greedy SLIM on ML-25M** (12–20 h) — the highest-value snap in the list; same dataset, same
   metric, same tail definition. Also the only way to check whether their adaptive decline
   disappears once the asked-item exclusion is removed (it should — their §6.5 says so).
6. **Golbandi tree on ML-100k RMSE** (8–14 h) — validates the split search before it becomes our
   headline adaptive bar.
7. CLiSK/CLiME regret · RMVA · DPE · IGCN · fMF · MeLU.
8. **PERE last, and only if a reviewer demands a 2024 adaptive name.**

## What each snap buys, stated before running (no post-hoc rationalisation)

* A **passed** native snap licenses steps 2–3 for that method, and makes "we ran their method, not
  our sketch of it" defensible.
* A **failed** snap is itself reportable — Mult-DAE's failure is *in Paper A* as a drop, not hidden
  — and it is the single strongest defence against "your reimplementation was weak", which is the
  first objection any adaptivity null will attract.
* The **Greedy SLIM** snap has a second payoff: it is the one experiment that can show a published
  adaptive-degradation curve is a protocol artefact, by re-running it with and without the
  asked-item exclusion.
