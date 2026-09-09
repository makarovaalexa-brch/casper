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

## ADDENDUM, 2026-09-10 — "is there a smaller dataset, so we can train quicker?"

### The cost model, from recorded fields

The greedy and tree stages dominate, and both are driven by one operation: fold the cohort's answers
and rank the whole catalogue for every cohort user. The artifacts record its unit cost directly:

| build | `n_cohort` | `n_evals` | `sec_per_eval` **[R]** |
|---|---|---|---|
| `greedy_static_policyB_items` | 10,000 | 839 | **9.272** |
| `greedy_static_policyB_concepts` | 10,000 | 1,546 | **7.691** |
| `greedy_static_i26c` (combined) | 10,000 | 3,173 | **13.543** |

At 10,000 users over an 18,359-item catalogue that is **~5.0e-8 s per user-item** **[E]**, and the
number of evals is set by the 1,000-candidate pool and the budget, not by the data. So for the
eval-bound stages, **cost scales as (eval users x catalogue size)** and the candidates rank like this:

| dataset | eval users | catalogue | relative cost **[E]** |
|---|---|---|---|
| ML-25M, as run | 10,000 | 18,359 | 1.00 |
| ML-20M | 10,000 | 20,108 | 1.10 — *not smaller* |
| Goodreads, 20k universe as previously run | 10,000 | 20,000 | 1.09 — *the truncation was chosen to match* |
| Goodreads, full catalogue | 10,000 | 188,867 | **10.3** |
| goodbooks-10k | 5,000 | 10,000 | **0.27** |
| ML-1M | 1,000 | 3,706 | 0.02 |

Training scales with *train* users x catalogue instead: ML-25M is 140,768 x 18,359; goodbooks-10k
would be roughly an order of magnitude less. **Overall a goodbooks-10k port looks like 4--6x faster
[E], i.e. the ~62 h becomes low tens of hours.**

### The two candidates that are genuinely smaller

**goodbooks-10k** ([github.com/zygmuntz/goodbooks-10k](https://github.com/zygmuntz/goodbooks-10k)):
10,000 books, **53,424 users, ~6M ratings on a 1--5 scale**, and critically it ships
`book_tags.csv` (user-applied tags and shelves per book, with counts) plus `tags.csv` mapping ids to
names. That is a folksonomy of exactly the shape the concept channel needs, and `gr_books.py`'s
shelf-mining logic should transfer more or less directly.

⚠ **But its users are shallow: the median user has 8 ratings.** ML-25M gives us 141 fold-in items per
user. After an 80/20 fold-in/target split a median goodbooks user leaves about six fold-in items and
two targets. Profile density is what drives answerability, and answerability is the paper's whole
mechanism, so this is the number that decides whether the dataset can carry the experiment at all.
It cannot be assumed either way from the raw size.

**ML-1M** is already on this box (`data/ml-1m/`) and an arena was built for it once
(`scripts/instrument2/ml1m_arena.py`). It is ~50x cheaper, and **it is not usable here**: 6,040 users
cannot support anything like a 10,000/10,000 held-out split, and `movies.dat` carries only 18 genres,
against the 1,031-concept bank the paper's whole concept channel is built on. Too small in both the
ways that matter.

### The pre-flight probe, which is the actual recommendation

Do not commit any training to a candidate dataset before running a probe that needs **no tower, no
teacher and no gates** — only the split, the tag mining and some counting:

1. Build the split; count users surviving `min_uc` and report the fold-in items per user distribution
   against ML-25M's 141.
2. Mine the tag/shelf vocabulary; count concepts clearing the **>= 30 member** floor against ML-25M's
   1,031.
3. Take a popularity-ranked top-500 item pool and the answerability-ranked concept pool, and compute
   the **answer rate at sixteen questions** for both channels.

The go/no-go is step 3. ML-25M reads **3.11 item answers and 12.36 concept answers per sixteen
questions**, and the entire finding is the ratio between those. A candidate that does not land in the
same regime is not a cheaper instance of this experiment, it is a different one, and its result would
not replicate anything. **The probe is hours; the port is weeks. Never run them in that order.**

### The lever that is not a dataset

The eval cohort size is a free parameter and every eval-bound stage is linear in it. Halving the
scored cohort halves the greedy and tree stages on *any* dataset. The recorded per-cohort trees are
consistent with that but sublinear in practice — 1,822 users cost 10,210 s against 10,000 users at
37,191 s, so a 5.5x cut in users bought only 3.6x in time **[R]** — so there is real fixed overhead
and the saving should not be assumed proportional.

## ADDENDUM 2, 2026-09-10 — the candidate must be a dataset other people use

Author constraint: nothing exotic. That is the right constraint, and it interacts with two hard
requirements of *our* pipeline, which together do most of the filtering.

**Filter 1 — graded ratings.** The item answer is a half-star level on a ten-band scale, and the
tower's sign-init table is ten entries wide (`gamma(level)` = ten values, logged at every i26 build).
A play-count dataset cannot supply that answer without redefining what an item answer *is*, which
changes the instrument rather than the data. **This is not a preference; it is a contract.**

**Filter 2 — an attribute vocabulary reducible to item -> concept membership** with a >= 30-member
floor. Without it the concept bank, and therefore the larger of the paper's two results, cannot
travel.

### The standard candidates against both filters

| dataset | standing in the literature | graded ratings | attribute vocabulary | verdict |
|---|---|---|---|---|
| **Netflix Prize** | **highest for our exact problem**: Golbandi 2011, whose tree we use, ran it; Hu and Yu 2013, our nearest rival, ran it (with ML-1M and Yahoo); and Liang, RecVAE and EASE all report on it, so the protocol is native and parity baselines exist | yes, 1--5, deep profiles | **none in the dump** | **best on standing, blocked on attributes** — see the transplant below |
| **Amazon (Books or one category)** | the most-used non-MovieLens benchmark in current CF | yes, 1--5 | category paths | **strong**; standard 5-core splits are shallow, so check depth |
| **Yelp** | 15 mentions in our own index, second only to MovieLens | yes, 1--5 | business categories | **strong**; a domain shift, since "have you seen" becomes "have you been to" |
| **Goodreads (UCSD)** | moderate, and **already half-built here** | yes, 1--5 | `popular_shelves` | strong, see the correction above |
| goodbooks-10k | moderate | yes, 1--5 | `book_tags` | fast but **median 8 ratings per user** |
| **Last.fm, any size** | high in the tag literature | **NO — listening counts** | tags | **OUT on filter 1.** hetrec2011-lastfm-2k is also only **1,892 users / 17,632 artists**, far too few for the held-out split; LFM-1b/2b and Last.fm-360K are larger but still play counts |
| **MSD / Taste Profile** | in Liang's own triple with ML-20M and Netflix | **NO — play counts** | tags via the Last.fm subset | **OUT on filter 1**, same reason |
| ML-20M | high | yes | Tag Genome | not a second dataset, and not smaller (1.10x) |

So the constraint bites hard: **of the standard options, only Amazon, Yelp and Goodreads clear both
filters natively.** Last.fm and MSD fail on the answer contract, not on availability.

### The Netflix option worth costing separately

Netflix is the dataset a reviewer of *this* paper would most expect, because the interview-tree
literature we are arguing with lives there. Its only gap is attributes, and there is a clean way to
close it: **Netflix items are films, and the Tag Genome is a film vocabulary.** Join Netflix's
`movie_titles.csv` to MovieLens on title and year and transplant concept *membership* only. The
concept answer itself is computed from the user's own history over the member films
(`signed_answers.py:43`), so nothing else needs importing.

That design has a property worth having: **the concept vocabulary would be identical across the two
datasets**, which separates "does the finding depend on the users and the catalogue" from "does it
depend on the particular concepts". It must of course be stated plainly in the paper.

**The deciding number is join coverage**, and it is cheap: Netflix has 17,770 titles, and what matters
is what fraction match a genome-covered MovieLens film, and whether the surviving concepts still clear
the >= 30-member floor. An hour of work, no training, and it decides whether Netflix is a
two-channel dataset or an item-channel-only one.

### Recommended order, revised for the standardness constraint

1. **Netflix**, contingent on the title-year join clearing coverage. Highest standing with exactly the
   reviewers who will read this, protocol-native, published parity baselines, deep profiles.
2. **Amazon Books or Yelp**, whichever the depth check favours. Both clear the filters natively and
   both are unimpeachably standard.
3. **Goodreads**, cheapest in effort because the mining and a trained RecVAE already exist here, but
   with the 20k-item / 418-user truncation to resolve first.
4. **Last.fm, MSD, Gowalla, MIND: not candidates**, because the item answer contract cannot be met.

The pre-flight probe in the previous addendum applies unchanged to whichever is chosen, and the
answer-rate step remains the go/no-go.
