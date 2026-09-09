# DS-A13 — the criterion-regret diagonal (PARKED, not launched)

**Status: PARKED on the to-do at the author's instruction, 2026-09-09. Nothing has run.**
This sheet exists so the job can be launched without re-deriving it. It is NOT a pre-registration
yet: the arms are settled but the cost is unbenched, and the gate and outcome table below must be
re-read before any launch.

## What the panel asked for

Three of four v7 reviewers, and the area chair's first *required* change, object to the same thing.
Table~2 (`tab:criterion`) scores the published split criteria against exact measured NDCG@10 gains
**at the 22 nodes of the tree grown by our own criterion**. The surrogates are therefore judged at
states they did not choose. Two reviewers call this a selection effect sitting on the paper's
cleanest diagnostic.

> R2: "RQ2's criterion-regret table scores the published surrogates only at the 22 nodes of the tree
> grown by CASPER's own criterion — a selection effect on the paper's cleanest diagnostic; it needs
> the diagonal (each criterion at its own tree's nodes)."

**The diagonal**: for each split criterion, grow that criterion's own tree, and at *its* nodes
compute the exact measured cohort-mean NDCG@10 gain of every candidate, then report that criterion's
agreement with the metric (mean Spearman, nodes with rho>0, argmax hit, top-10 recall, regret)
**on its own trajectory**. The current table is one column of a matrix; the diagonal is the column
that licenses the sentence the paper wants to make.

**v8 already carries the fallback the area chair explicitly permits**: the table and its caption now
say the numbers are off-trajectory and point at Table~4 for the on-trajectory question. That closes
the *disclosure*. It does not close the *objection*.

## What this is NOT

⚠ **This has nothing to do with anchor teachers or cluster teachers.** The trees here are the cold
Golbandi ternary interview trees grown on the certified tower, one per split criterion, exactly as in
the Table~4 ablation. `split_criterion_ablation.py:78` imports only
`build_pool, split_rows, branch_of, leak_assert, BRANCHES` from `ndcg_tree`, i.e. the plain ternary
primitives. `ndcg_tree.py` also carries `--anchor` / arbitrary-cohort **teacher** paths (M3, the Act-5
opener line), but the ablation does not touch them and this job must not either. See
[[teachers-are-anchor-teachers-not-cluster-teachers]] and
[[cluster-vs-endorsement-cohort-and-the-era-hole]].

## Arms

The six arms of the existing ablation, of which the first is already done:

| criterion | tree exists? | measured gains at its nodes? |
|---|---|---|
| measured NDCG@10 (ours) | yes | **yes — this is the current Table 2** |
| Hu and Yu pairwise rank loss | yes (Table 4) | no |
| Golbandi rating variance | yes (Table 4) | no |
| logistic loss as run | yes (Table 4) | no |
| shuffle, min-leaf-respecting | yes (Table 4) | optional, it is a control |
| shuffle, unrestricted | yes (Table 4) | optional |

**The trees are already grown and banked** — they were built for Table~4. The new work is *only* the
exact-gain sweep at their nodes: roughly 22 nodes x 1,000 candidates x fold-and-rank per criterion.

## The cost, and why it is not yet a number

The paper quotes **37,191 s per tree** for our own construction, which is the same
nodes-by-candidates sweep. Naively three criteria is ~31 h. **Do not launch on that figure.** This
box's cost estimates have been wrong by over 4x and under 2.6x
([[cost estimates are not properties of a method, CLAUDE.md]]), and the sweep may be cheaper here
than during tree growth because the tree structure is fixed, so no search or CELF heap is needed and
every node's cohort is known in advance.

**Bench first: one node, one criterion, timed.** Multiply out, then decide. If the benched figure
puts three criteria under about 12 h, run all three; if it lands near 30 h, run the two criteria the
paper actually names in its RQ2 sentence (pairwise rank loss and rating variance) and report the
logistic arm as off-trajectory only.

## Gate

The measured-criterion column recomputed under the new harness must reproduce the published Table~2
row for our own criterion exactly: mean rho, nodes rho>0, argmax hit, top-10 recall and regret, to
the precision printed. If it does not, the harness is wrong and nothing else may be read.

## Registered outcomes

- **The surrogates look no better on their own trajectory** -> the current table was not flattering
  itself, RQ2's claim strengthens, and the sentence loses its qualifier.
- **The surrogates look substantially better on their own trajectory** -> the off-trajectory scoring
  *was* doing work, the 11--34% figure is revised, and the paper says so.
- **Mixed across criteria** -> report per criterion; the diagonal replaces the single column.

All three change the paper. Note that the second outcome is the one that costs us something, which
is the reason to run it rather than leave the flag.

## Hard rules

HARD RULE 1 (no truncation of nodes or candidates), 10 (commit before launch), 11 (bank per node, one
heavy job at a time, check free RAM, and the box sleeps on low battery). At ~30 h this is a
multi-night job: detach with Start-Process and monitor the log, never session background bash.

## Deadline context

Abstract 21 Sep 2026, paper 5 Oct 2026. v8 already carries the disclosure the area chair accepts, so
this run is an **improvement, not a blocker**. If it is launched, it must not be the thing that
delays the submission.
