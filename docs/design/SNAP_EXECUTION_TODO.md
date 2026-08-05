# Paper B — snap-first execution TODO (author ruling 2026-08-05)

**Ordering ruling: NATIVE SNAP FIRST, then port.** No baseline is run on our ruler until it has
been validated against its own paper's published numbers in its own setup. Rationale: the first
objection any adaptivity null attracts is "your reimplementation was weak", and the only answer is
a passed native snap. The precedent is Paper A's Mult-DAE, which missed its published number by
0.022 and was dropped from the paper rather than reported with an excuse. SNAP-1 (EDDI) has already
justified the ordering: our variant lost to *random ordering* on the authors' own data, so the CF
number we had been reporting measured our reduction, not their method.

**Persistence rule now in force (CLAUDE.md HARD RULE 11)** after a 4-hour run was lost to OOM with
nothing written: every cell to disk on completion, contract-signed resumable sidecars, bounded
caches, one heavy job at a time.

---

## The three steps, per baseline

| step | what | gate |
|---|---|---|
| **S — native snap** | their dataset, their split, their metric, their recommender | matches published within stated precision, or a diagnosed gap. **FAIL ⇒ does not enter the paper as that method's row** |
| **P — port** | our ruler (Liang ML-25M, policy-B mask, behavioural answers, full+tail NDCG@10), still **their** recommender as consumer | runs clean; MostPop gate passed |
| **X — cross** | our ruler, **our** certified tower as consumer | the cell Act 1 actually needs |

Every row must additionally carry **its own paper's static baseline** re-run at P and X, or the
comparator silently swaps — the exact failure this chapter is about.

---

## Queue

### DONE
* **SNAP-1 EDDI** — S: **PASS for their reward, FAIL for ours** (pooled AUIC rank: their reward
  2.411, SING 2.381, random 3.236, **ours 3.402**). Row relabelled "latent-only info-gain"; the
  full criterion is task #9. Verified clean: the sum-pooling conditioning identity is exact.

### NOW (native snaps, in this order)
1. **SNAP-3 DRE** (Kweon WWW'20) — public PyTorch, binarises >3.5 like us, ML-20M NDCG@10.
   *First action: extract the published table from the paper — a secondary source implies a
   ~0.57-class NDCG@10 and that is unverified.* We already have a Liang ML-20M bridge from Paper A.
   Est. 6–10 agent-h. **The peer-reviewed static bar.**
2. **SNAP-5 Golbandi tree** (WSDM'11) — split search, the piece missing from our harness. Snap the
   *shape* on ML-100k RMSE first; archived precedent `scripts/_archive/paper1_v1/golbandi_tree.py`
   got 0.9804 vs static 0.9885 with adaptive winning at every depth ≥1. Est. 8–14 agent-h.
   **The classical adaptive bar; no method in fifteen years has a clean replicated win over it at k≤16.**
3. **SNAP-2 RBMF** (RecSys'11) — maxvol selector, P@10 at budgets 5/10/20 (= ours). Est. 3–5 h.
4. **SNAP-6 ICF four policies** (CIKM'13) — snap the *policy ordering*; note their own experiment has
   **Thompson losing to LinUCB/GLM** (43.7 vs ~46.4), contrary to how it is usually cited. Est. 4–6 h.
5. **SNAP-4 Greedy SLIM** (2024) — highest value: same dataset, same metric, **same tail definition**.
   Targets static .3390/.3594/.3709/.3752 and QBandit .3382/.3456/.3221/.2916 at k=5/10/15/20.
   Second payoff: re-run **with and without their asked-item exclusion** (their §3.3.2 requires
   `I_Q ∩ I_R = ∅`; §6.5 blames it for the adaptive decline) to test whether a published
   adaptive-degradation curve is a protocol artefact. Est. 12–20 h.

### BLOCKED / REPAIR
* **ConLinUCB re-run** — the 10k run died OOM at turn 14/16. Fix first: bound the per-state width
  cache (18,359 float64 per state, states → one per user), float32 the cached vectors, and
  checkpoint interviews per cell. Then re-run. No native snap exists (regret curves only), so its
  S-step is "reproduce the regret ordering on ML-25M", optional.
* **iALS determinism** — GLM-UCB/Thompson are not reproducible across processes (±.0026, comparable
  to the CI half-widths). `random_state=98765` is already pinned, so the suspected cause is
  multithreaded reduction order; test running. If confirmed: pass `num_threads=1`, re-run both
  bandit arms. **Until then no bandit number is quotable to four decimals.**
* **EDDI full Eq. 9** (task #9) — implement both terms with `x_φ` = the unobserved catalogue vector
  (no leak: the second term samples from the model) plus proper per-item marginals.

### LATER
DPE, BALD/SEL-BALD, A-optimal, PEBOL acquisition, MeLU bank ranker, a-stratification, fMF, RMVA,
GreedyExtend, PERE (last, and only if a reviewer demands a 2024 adaptive name — its published
margins are small and its own DRE column is ~10× below DRE's).

### CHEAP, INDEPENDENT
Floored arm of Act 1 (one flag) · `--objective tail` greedy · greedy re-run with policy B as the
*selection* contract, not just the scoring contract · re-point `strategy_portability.py` at
`greedy_static_i26c.json`.

---

## Standing requirements for every snap agent

* State the published target **with its source table/figure** before running anything. If the paper
  reports only curves, say so and snap the *ordering*, not invented numbers.
* Mark every claim [V] full text / [A] abstract / [S] search summary. Never present [S] as [V].
* Report a FAILED snap as a result. Do not tune to make it pass.
* Persist per cell (HARD RULE 11). Check free RAM before launching; one heavy job at a time.
* No sampling/truncation of any evaluation cohort in anything citable; SMOKE runs must be labelled
  and stamped non-citable.
* Do not commit — the overseer reviews first.
