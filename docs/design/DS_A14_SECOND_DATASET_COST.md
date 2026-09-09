# DS-A14 — what a second dataset actually costs (2026-09-09)

**Status: COSTING ONLY. Nothing proposed, nothing launched.** Written because the v7 area chair named
"Tables 2--4 replicated on a second dataset with a second certified recommender" as the one thing
that would make the paper an unambiguous accept, and called it a follow-up paper themselves.

Every wall-clock figure below marked **[R]** is a value literally recorded in an artifact or log on
this box. Figures marked **[E]** are my arithmetic on those. Nothing else is guessed, and the stages
with no recorded number are listed as such rather than filled in.

## CORRECTION, 2026-09-10. My first version of this sheet was wrong.

It claimed the Tag Genome "has no substitute" and that half the paper therefore cannot port. **Both
claims are false, and the repository already contained the refutation.** The author's instinct that
"attributes can prob be mined on most things" is correct, for a reason that is visible in our own
code.

**The concept channel does not consume graded relevance. It consumes a BINARY membership matrix.**
`belief_layer.py:632` thresholds the genome at `relevance >= 0.5` and discards the score; the matrix
is then built with `np.ones(...)` (`strategy_channel_suite.py:1422`). The only other thing derived
from it is an exposure prior `pexp` from train counts (`:455`). So what the pipeline actually needs
is: *which items belong to which concept*, plus a >= 30-member floor. **Any mined folksonomy of that
shape is a drop-in.** The Tag Genome's density and its learned relevance scores are thrown away
before the concept channel ever sees them.

**And we already mined one.** `scripts/_archive/paper2_old/gr_books.py` parses Goodreads
`popular_shelves` into exactly this object: a junk filter for process shelves ("to-read", "i-own",
"arc", "netgalley"), membership at shelf count >= 5, a **>= 30-item floor explicitly mirroring
ML-25M**, the top ~1,500 content shelves by document frequency, plus concept centroid directions.
Its own docstring says "like ML-25M".

**A Goodreads recommender already exists too.** `scripts/instrument2/gr_recvae.py` ports RecVAE to a
Goodreads arena of 188,867 items / 768,746 users / 46.1M ratings, and the results are still on disk:
`.cache/instrument2/gr_recvae_d512_TEST.json` reads **full/tail NDCG@10 of 0.4340 / 0.2507** against a
most-popular floor of 0.2045 / 0.0090. That is the parity bar a re-certified instrument would have to
clear, and it is already measured.

So the second dataset is **not blocked on a missing artifact**. The recipe exists for Goodreads. What
does not exist is any of it on the current stack.

## What genuinely remains, having corrected the above

These are real and none of them is research risk; they are work.

- **The data is gone from this box.** `.cache/goodreads/base_comp.npz` is absent and `data/` holds
  only MovieLens and the Reddit corpus. The Goodreads book graph must be re-downloaded, and the
  metadata dump carrying `popular_shelves` is large.
- **Everything Goodreads is on the retired stack.** The scripts sit in `scripts/_archive/paper2_old/`
  and predate 2026-07-22, so their numbers are demoted under
  [[pre-jul22-interview-results-demoted]] and the code speaks the old SVD-arena contract, not the
  i26 tower / cv3 concept channel contract. Porting is mechanical but it is not free.
- **The old Goodreads arena was truncated to make it run**, and this matters more than the rest.
  `gr_recvae.py` restricts to "top-N=20,000 items by train-like count" of 188,867, and evaluates on
  **418 test users**, not 10,000. Both were stated compute deviations at the time. Under HARD RULE 1
  neither may simply be inherited: a 20k-of-189k universe is exactly the kind of reduction that needs
  sign-off, and the paper's catalogue-scale claim is one of its selling points.
- **Shelf membership is coarser and dirtier than genome membership.** Shelves are user-authored,
  popularity-skewed and full of process labels, which is why `gr_books.py` needs an explicit junk
  list. The junk filter is a judgement call that would have to be defensible in print.

## What this does to the options

**Goodreads is now the strongest candidate, not the most expensive one.** It is a different domain,
different users and a different catalogue, with a mined attribute vocabulary we have already built
once and a strong recommender we have already trained once. It is the option that answers the
reviewer's actual question, and the earlier version of this sheet wrongly ruled it out.

**Netflix Prize** remains the pre-declared plan (`PAPER_B_DESIGN.md:91`,
`PAPER_B_EVIDENCE_POLICY.md:26`, scope fixed to "orderings and interactions, not magnitudes") but is
now clearly the *weaker* choice: it carries no attribute vocabulary at all and no folksonomy to mine,
so the concept-bank result genuinely cannot travel there.

**ML-20M** stays what it was: cheapest, already split, already genomed, and almost certainly not what
a reviewer means by a second dataset.

## Recorded cost of one full port (Goodreads or ML-20M; Netflix drops the concept rows)

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
uninterrupted machine time.** Netflix removes the concept channel and the concept order, landing
nearer 50 h. Goodreads adds a re-download and the shelf-mining port on top of the 62 h, and its
catalogue is ten times ML-25M's, so the greedy and tree stages would not simply inherit these times.

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

1. **Goodreads**, because it is a genuine second domain and the attribute vocabulary and the
   recommender have both been built here before. Re-download, port `gr_books.py`'s shelf mining to the
   current concept contract, retrain the tower, re-certify against the recorded 0.4340/0.2507 bar,
   then rerun. **Resolve the 20k-item universe and the 418-user cohort before anything else**, since
   inheriting either would breach HARD RULE 1 and would weaken the catalogue-scale claim.
2. **ML-20M** only as a cheap self-replication check, and never described as a second dataset.
3. **Netflix** only if an item-channel-only result is wanted, with the bank result explicitly stated
   as not portable there.
