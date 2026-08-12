# Overnight handoff — night of 2026-08-12/13

AUTHOR SAID (verbatim, 2026-08-12 before sleeping): "i am going to sleep, plz don't stall: write up the
neural section, replicate hu & yu as fully as you can, we'll discuss in the morning."

Everything below is MINE (working notes). Read `docs/design/PAPER_B_PLAN_2026-08-12.md` first — it is the
single source of truth for the redirected chapter.

## Running overnight

| agent | task | deliverable | box |
|---|---|---|---|
| B1 / TODO 1.12 | Hu & Yu: finish the faithful replication (Part A), then the three-way split-criterion ablation (Part B) | `docs/results/HU_YU_2013_REPLICATION_AND_ABLATION.md` | **has the box** |
| positioning survey | the neural section: verify CRM/EAR/SCPR/UNICORN + LLM line from primary texts, write the position and the borrow list | `external_literature/findings/modern_crs_positioning.md`, `docs/results/POSITION_VS_MODERN_CRS.md` | desk research, no CPU |
| C1 / TODO 1.11 | HELD — building phases 2/3 and pre-writing the results doc; no runs until released | `docs/results/ACT2_SURROGATE_RANK_CORRELATION.md` | released after Hu & Yu |

Cost warning carried into the Hu & Yu brief: one 16-pick exact-NDCG tree cost ~12 h cold, so a three-way
ablation could be 30 h+. **No silent reduction is permitted** (HARD RULE 1). If it cannot finish, the
instruction is to run pairwise-rank-loss-vs-ours FIRST (variance-vs-ours is implied by the battery),
persist incrementally, and report what did not run with its measured cost.

## What landed today, in order

1. **A1** static COMBINED order rebuilt under policy B (16 picks, 2,167 evals, 104.6 min). Shares 4/16
   questions and **0/16 positions** with the policy-A order it replaces.
2. **A2** tree re-measured. **K=1 gate exactly 0.0** on all 20,000 users, both axes. Test K=16
   **+0.01035 full / +0.00705 tail** (was +0.01371/+0.02092). Two retractions: "all ten CIs exclude
   zero" (tail at K=2 now separates AGAINST the tree, −0.00262) and "tail does not shrink"
   (clean val→test is 2.70× full / 1.50× tail).
3. **1.20 routing** — the tree PARTITIONS and DIVERGES but does not demonstrably ROUTE (congruence null,
   both controls p>0.6). **No routing claim in the chapter.**
4. **Policy-A provenance audit** — contamination confined to one comparator object; Act 2's criterion is
   pool-invariant; the partial-resume signature had no pool field (fixed, guard proven 14/14).
5. **A3/A4** items and concepts orders rebuilt; genre arm re-measured (+0.00862/+0.00591 fallback-200);
   **bank beats conditioning at equal budget: +0.01430 full / +0.02199 tail vs the tree's +0.01035/+0.00705.**
6. **A10** depth audit — all 15 stops are FLOOR stops with a positive gain in hand; 0 stopped for want of
   a split. May not claim conditioning stops paying at depth 5, nor that deeper would pay.
7. **Chapter rewritten** — new title, new Act 3 (the demonstration), Act 4 reframed to answerability,
   Act 5 = reach. 26 pp, builds clean, verified in the RENDERED pdf.
8. **A9** battery rows re-scored. **★ Retracts a printed separated interval**: Golbandi's tail-at-8 win
   was comparator weakness. No published adaptive selector now beats the strong static anywhere.

## First things in the morning

- **#38** merge A9's rows + the Golbandi retraction + the stale prose into the tex (`docs/results/BATTERY_TABLE_POLICYB_ROWS.md` lists every line).
- **#37** re-difference Act 2's 48 PV-quartile cells onto the rebuilt items order — the "Golbandi's
  crossover lives in the PV=0 quartile" sentence rests on a crossover A9 has now retracted in aggregate.
- Read whatever the two overnight agents wrote, in their own files.
- Then: A5 budget consistency, A6 tail-objective greedy, A8 confirmatory pass, and E1 (the parametric
  policy) once the tree it imitates is on final numbers.
