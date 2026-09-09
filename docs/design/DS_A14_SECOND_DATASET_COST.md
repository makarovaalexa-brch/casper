# DS-A14 — what a second dataset actually costs (2026-09-09)

**Status: COSTING ONLY. Nothing proposed, nothing launched.** Written because the v7 area chair named
"Tables 2--4 replicated on a second dataset with a second certified recommender" as the one thing
that would make the paper an unambiguous accept, and called it a follow-up paper themselves.

Every wall-clock figure below marked **[R]** is a value literally recorded in an artifact or log on
this box. Figures marked **[E]** are my arithmetic on those. Nothing else is guessed, and the stages
with no recorded number are listed as such rather than filled in.

## The blocker is not compute. It is the Tag Genome.

ML-25M is unusual in shipping a dense, machine-learned **item-by-attribute relevance matrix** (the Tag
Genome, 1,031 concepts at relevance >= 0.5 with >= 30 members). The concept channel is built directly
on it (`src/instrument/belief_layer.py:616`,
`src/baselines/conversational_bandit_selectors.py:409`), and **it has no substitute**: no other
public recommender dataset ships an equivalent.

This matters because the concept bank is the *larger* of the paper's two results (+0.01430 against the
tree's +0.01035, and not close on the tail). **On any dataset without an attribute-relevance matrix,
half the paper cannot be ported at all** — not expensively, but at all. Building a Tag-Genome
equivalent is a published research contribution in its own right (Vig et al.), not a porting step.

So the choice is really three options, and they are not the same experiment.

## Option A — ML-20M. Cheap, already half-prepared, and probably does not satisfy the reviewer.

`data/ml-20m/` is already on the box **with `genome-scores.csv`**, and `data/ml-20m/proc/` already
holds a finished Liang split built bit-faithfully to Liang's own notebook (20,108 items, 116,677
train / 10,000 val / 10,000 test, seed 98765). The split step is done.

**What it buys:** a test of sensitivity to split, seed, vintage and catalogue. Published baselines
exist on exactly this benchmark, so HARD RULE 2 is satisfiable.
**What it does not buy:** domain generalisation. Same platform, same item universe, largely the same
users; ML-20M is essentially an earlier cut of ML-25M. A reviewer asking for a second dataset will
very likely say so.

## Option B — Netflix Prize. Already this project's declared plan, and it ports only half the paper.

`docs/design/PAPER_B_DESIGN.md:91` and `PAPER_B_EVIDENCE_POLICY.md:26` already name Netflix Prize
under a Liang-style protocol, with the claim scope pre-declared as **"orderings and interactions, not
magnitudes"**. That ruling stands and should be honoured rather than reopened.

Graded ratings, a comparable protocol, and published baselines (Liang's VAE-CF reports on it). **No
attribute vocabulary**, so the concept bank, the answerability mechanism and the +0.01430 result
cannot travel. RQ1, RQ2 and the tree can.

## Option C — a genuinely different domain (Goodreads shelves, Last.fm tags, Yelp categories).

Requires constructing and validating an attribute-relevance matrix from a folksonomy before any of
the pipeline can run. That is a research step with its own failure modes and its own evaluation.
**Months, and a different paper.** Not costed further here.

## Recorded cost of one full port (Option A or C; Option B drops the concept rows)

| stage | recorded wall-clock | source |
|---|---|---|
| Liang split | **not recorded** — script emits no timing | — |
| Graded all-bands reconstruction | 73 s **[R]** | `experiments/baselines/arm_n_canary.out` |
| Tag Genome -> concept bank | 4--11 s **[R]** | `experiments/battery/a3_items_train.out` |
| RecVAE teacher (frozen decoder) | 483.1 min = **8.05 h** **[R]** | `ml25m_liang/run.log:46-77` |
| i26 two-tower to ep31 | 1,146.2 min = **19.1 h** **[R]**, summed over three resume segments | `t2i26c_train.log` |
| Concept channel cv3 to ep8 | 72.4 min = **1.2 h** **[R]** | `cv3_train.out` |
| Certification gates, phases A + B | 1 h 47 + 1 h 23 + 0.2 = **3.4 h** **[R]** | `phaseA_ep31.log`, `phaseB_run.log`, gate JSONs |
| Greedy static orders, all three arms | 43,410.7 s = **12.06 h** **[R]** | `greedy_static_i26c.json` |
| Measured-NDCG tree | 37,191.3 s = **10.33 h** **[R]** | `ndcgtree.json` |
| Split-criterion ablation (grow + measure) | 32.6 m + 24.6 m = **1.0 h** **[R]** | `splitcrit_grow.log`, `splitcrit.json` |
| Battery, all arms interviewed | sum of per-arm times = 408 min = **6.8 h** **[E]** on recorded per-arm figures; nearest recorded single job **~5 h** **[R]** | `act1_paired.log` and others |
| Re-score pass over banked interviews | 26.2 min **[R]** | `a9_battery_policyB.json` |

**Pure compute, one clean pass, nothing failing: about 62 h [E], call it two and a half days of
uninterrupted machine time.** Option B removes the concept channel and the concept order, landing
nearer 50 h.

## Why 62 h of compute is not 62 h of calendar on this box

The recorded history is unambiguous and should be applied as a multiplier, not a footnote:

- **The box sleeps on battery and has eaten runs twice**: 4.1 h and **53.5 h** lost
  (`src/instrument/run_guard.py:3-22`). One incident alone is comparable to the whole port.
- **Load and thermals swing identical work by 2.7×**: 43 versus 118 minutes per epoch for the same job
  (`docs/results/OUTCOME_HEAD.md:180-183`).
- **4 physical cores, 16 GB, CPU-only PyTorch**, and no second job may run beside a fit. A 150k-user
  ablation has already OOM'd on this machine.
- **The i26 tower did not train in one go** — it is three resume segments in the log — and the
  certified checkpoint is the end of an architecture search (i25 -> i26 -> i26c), not a first attempt.
- **Certification is not guaranteed to pass.** The tower must reach parity with a strong baseline on
  the new dataset. If the recipe does not transfer, the cost is another architecture search, and that
  is the step with no bound on it.

**Realistic elapsed time for a competent port that succeeds: two to four weeks**, assuming the
instrument certifies on the first or second recipe. If it does not certify, there is no honest upper
bound, because that becomes a research problem rather than a run.

## Against the calendar

Abstract 21 Sep 2026, paper 5 Oct 2026: **twelve and twenty-six days**. Even the cheap option does not
fit, and the cheap option is the one a reviewer is least likely to accept. **The area chair's own
framing is correct and should simply be accepted: this is a follow-up paper, not a camera-ready fix.**

## If it is ever done, the order to do it in

1. **ML-20M first (Option A)**, because the split is already built and the genome is already on disk,
   so it isolates "does the pipeline reproduce itself" from "does the finding generalise". It is also
   the cheapest way to discover that the recipe does not transfer.
2. **Netflix second (Option B)**, honouring the pre-declared "orderings and interactions, not
   magnitudes" scope, and stating plainly that the bank result does not travel because the dataset
   carries no attribute vocabulary.
3. **Option C only as its own paper**, with the attribute-relevance construction as a contribution
   rather than a preliminary.
