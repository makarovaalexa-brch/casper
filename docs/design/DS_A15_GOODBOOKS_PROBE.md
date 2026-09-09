# DS-A15 — goodbooks-10k as Paper B's second dataset

**Status: PROBE PASSED 2026-09-10. Replication in progress.**
Author instruction, verbatim: *"can you pick one, ideally not huge, ideally not movies, and go with
it? replicate whatever you can, stop if results stop making sense / are weak etc and switch to next
candidate."*

## Why this candidate

From `DS_A14_SECOND_DATASET_COST.md`: not films, not huge, and it ships a tag folksonomy.
10,000 books, 53,424 users, 5,976,479 ratings on a 1--5 scale, with `book_tags.csv` and `tags.csv`.
Downloaded to `data/goodbooks-10k/` (git-ignored).

⚠ **One published claim about it is wrong and was corrected against the data.** Secondary sources say
the median user has 8 ratings. The actual median is **111**, the mean 111.9, the 10th percentile 82
and the maximum 200. Every user has at least 19. The depth risk that nearly disqualified this
candidate does not exist.

## The probe, and its result

`src/instrument/a15_goodbooks_probe.py`, artifact
`experiments/goodbooks/a15_goodbooks_probe.json`. No tower, no teacher, no gates — split, mine, count.
Every definition mirrors ML-25M with a file:line reference in the script.

| quantity | goodbooks-10k | ML-25M | reading |
|---|---|---|---|
| fold-in items per user | 89.7 | 141.2 | thinner, same order |
| concepts in the bank (>= 30 members) | **1,763** | 1,031 | comparable |
| concepts per item | median 40, mean 39.7 | median 37, mean 44.5 | **calibrated to match** |
| **item answers per 16 questions** | **3.86** | **3.11** | same regime |
| **concept answers per 16 questions** | **15.82** | **12.36** | same regime |
| **ratio, concept : item** | **4.09** | **3.98** | **the finding's own quantity, reproduced** |
| users with zero item answers | 16.4% | ~17.5% | matches |
| users with zero concept answers | 0.0% | ~0.1% | matches |

**GO.** The mechanism the paper is about — item questions rarely answerable, concept questions almost
always — is present on books at almost exactly the same strength.

### The calibration, stated because it is a judgement call

A first membership rule (a tag counts if it is >= 2% of that book's tag mass) produced only 103
concepts. Popular books carry enormous tag mass, so a fractional rule is a popularity filter in
disguise. Replaced with each book's **top-50 tags by application count**, where K is calibrated to
match ML-25M's *per-item concept density*, not its bank size. The bank size then falls out on its own
at 1,763. Calibrating density is the defensible choice because it is what makes a concept question
mean a comparable thing in the two datasets. Both attempts are recorded in the script.

## The replication, and what was deliberately dropped

**The concept channel needs the two-tower instrument**, because RecVAE cannot ingest a concept answer.
Porting and re-certifying the tower is roughly 24 h of the ~62 h estimate. So the first replication
targets **RQ1 on the item channel only**, with **RecVAE itself as the consumer** — which is not a
compromise invented here, but the "RecVAE, likes only, conservative" consumer the paper already uses
in its own robustness sweep.

- `a15b_goodbooks_recvae.py` — persists the Liang split under `data/goodbooks-10k/proc/` in the same
  shape as `data/ml-25m/proc/`, then trains RecVAE. **43,424 train / 5,000 validation / 5,000 test
  users, full 10,000-book catalogue, NO truncation** — the defect that dogged the old Goodreads work
  (20,000 of 188,867 items, 418 test users) is not repeated. Resumable, banks every epoch.
  Measured: **0.6 min per epoch**, against 13.5--14 min on ML-25M.
- `a15c_goodbooks_interview.py` — the RQ1 audit. Arms: lazy-greedy NDCG-maximising order (the
  comparator), popularity, entropy0, HELF, a seeded random control, and a clairvoyant oracle ceiling.
  Orders fitted on validation, scored once on test, paired percentile bootstrap over per-user
  differences, 10,000 resamples, seed 98765.

**The claim this can and cannot support.** It can support: *under one contract on a second domain, do
published static heuristics lose to an order built by maximising the reported metric?* It cannot
support the bank result or the tree result, both of which need the concept channel and therefore the
tower.

## Stop conditions, per the author's instruction

Switch candidate, or stop, if any of these fire:
1. RecVAE fails to reach a sensible level for the dataset, leaving no strong consumer to measure
   against. A weak consumer makes every elicitation number uninterpretable.
2. The greedy order fails to beat the random control, which would mean the harness is broken rather
   than the finding absent.
3. The answer-free reference and the interview curves do not separate, i.e. asking buys nothing.
4. Any leak guard trips.

## Protocol, fixed before the numbers

min_uc = 5; like <=> rating > 3.5; per-user 80/20 fold-in/target; seed 98765; full and tail NDCG@10
with the Cremonesi short head removed for the tail; candidate pool excludes asked books; no
popularity floor; no negative sampling; unanswerable questions charged.

## Hard rules

HARD RULE 1: no truncation — full catalogue, all users surviving `min_uc`.
HARD RULE 10: every script committed before it was run.
HARD RULE 11: per-epoch checkpointing and resume; one heavy job at a time; free RAM checked at launch
(4.21 GB).
