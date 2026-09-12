# DS-A22 — price the monitor's defects, then test the member-count hypothesis (2026-09-12)

Follows `DS_A17_GOODREADS_ARENA.md` / `docs/results/A17_GOODREADS_PORT.md`, which stopped on a negative
result: on Goodreads the concept curve runs backwards (m0 0.1877 -> m8 0.1746) while on ML-25M it rises.
A Fable analysis and my own probe since produced two competing, both-partly-supported explanations. This
sheet prices them separately and pre-registers the conditions under which each would be believed.

## The question

1. **How much of the Goodreads decline is the MONITOR rather than the channel?**
2. **Is the residual explained by CONCEPT NARROWNESS, and does restricting the ask set to the broadest
   concepts restore the channel?**

## What is already established (measured, not assumed)

- The monitor applies the TRAINING member-drop at evaluation (`concept_train_v2.py:105-107`): items
  supporting an answered concept are removed from the pool *before* the 4 context items are drawn. The
  context is therefore stripped of the taste the targets lie on, and the penalty grows with m.
- The m-levels are UNPAIRED: targets are redrawn inside each call (`:87`), so m0-vs-m8 mixes the effect
  with target-draw noise.
- The val slice is `S["users"][-1500:]` (`:63`), a positional tail of a `userId`-ordered list. On
  ML-25M activity-neutral; on Goodreads the highest UCSD ids are the lightest accounts.
- **The slice is held out of training** (`:248`, `S["users"] = S["users"][:-N_VAL]`). Therefore the
  slice CANNOT be re-drawn for the existing checkpoints without leaking. Defect 3 is fixable only for
  future runs; for cv3/cvg it is reported, not repaired. (This corrects the Fable analysis, whose
  random-slice NDCG numbers drew users the net had trained on.)
- Concept banks, per-concept size, ALL 140,768 train users, project gate (`n_c>=2 AND E>=TAU_REF`):

  | | concepts | median members | median profile | broad-policy refusal |
  |---|---|---|---|---|
  | ML-25M | 1,031 | 291 | 72 | 0.252 |
  | Goodreads | 2,408 | 80 | 47 | 0.727 |

  A19 matched per-ITEM density (37 tags/item on both) and never matched per-CONCEPT size.
- Restricting the ask set to the top-N broadest collapses Goodreads refusal: top-50 -> 0.223, at mean
  |SEL| 0.839 (ML-25M's current operating point is refusal 0.252 at |SEL| 0.616).
- `ask_pool("broad")` is NOT top-N: it samples with probability proportional to
  `clip(member_count, 30, 1000)` (`concept_curriculum_v2.py:119`). The clip flattens the whole top end.

## Instrument

New, standalone: `src/instrument/a22_concept_curve_paired.py`. It does NOT modify the training script,
so cv3's and cvg's training logs stay reproducible. It evaluates a frozen (tower, concept-net) pair on
the SAME held-out 1500 with each monitor defect switchable, so each is priced on its own:

| switch | off = corrected | on = as-shipped |
|---|---|---|
| `--drop` | context drawn from the full pool | member-drop applied first |
| `--pair` | ONE target draw per user, shared by every m | targets redrawn per m |
| `--strip` | concept net applied | concepts zeroed, same context (isolates the FOLD EFFECT) |
| `--ask` | `broad` (as shipped) | `topN:<N>` — deterministic N broadest |

Reported per m in (0,1,4,8): NDCG@10 full and tail, over >= 5 seeds, with the paired fold effect
(as-is minus stripped) and a bootstrap CI clustered on users.

## Gates — checked before any number is read

- **G1 REPRODUCTION.** With `--drop --pair off --ask broad`, seed 4242, the instrument must reproduce
  the logged curves to 1e-3: ML-25M (cv3 ep4) 0.1811 / 0.1774 / 0.1856 / 0.1916; Goodreads (cvg ep12)
  0.1877 / 0.1893 / 0.1817 / 0.1746. **A miss means the instrument is wrong and nothing below is
  readable — stop and debug.**
- **G2 LEAK.** Assert the eval users are exactly `users[-1500:]` and disjoint from the training slice.
- **G3 EXISTENTIAL CONTROL.** Permute concept ids within each user (values kept). The fold effect must
  collapse toward zero. If a scrambled interview buys the same lift, the operator is not using which
  concept was answered and every positive number here is meaningless.

## Pre-registered readings

- If the corrected protocol (`--pair`, no `--drop`) turns the Goodreads m-curve FLAT or RISING, the
  reported negative result was a protocol artifact and must be retracted.
- If it stays DECLINING, the decline is real and the ask-set arm is the next test.
- The fold effect being positive on Goodreads at m1/m4 but not accumulating is the ANSWER-RATE story;
  the fold effect being ~0 or negative at every m is the OPERATOR story.
- **ML-25M is re-reported under the identical corrected protocol, and is expected to get WEAKER.** That
  it costs us on the reference dataset is what makes the correction not a Goodreads-fitted one.

## The ask-set arm, and what would make it defensible

Restricting to the top-N broadest is measurable at eval with NO retraining (the net takes concept ids;
asking a different set only changes which ids arrive). It is OFF-DISTRIBUTION for a net trained on
broad-policy draws, so a positive result there is a lower bound and justifies a retrain; a null there
does not yet refute the hypothesis.

**N must not be chosen by reading the Goodreads curve.** Two admissible rules, both fixed on ML-25M
before any Goodreads NDCG is read:

- **R-struct:** choose N per dataset to match ML-25M's members-per-concept median (291). Honest but
  weak — Goodreads needs N ~ 900 and refusal only falls to ~0.68.
- **R-rate:** pre-register the ASK RATE as the protocol constant — choose N per dataset to hit
  ML-25M's shipped answered-share (0.748). On Goodreads that is ~ top-50. The answer rate is the
  interview's operating point, not the metric, so holding it fixed across datasets is a protocol
  statement rather than fitting to the result.

R-rate is the one to pre-register, and BOTH datasets must be re-run under it. Reporting Goodreads under
R-rate and ML-25M under the shipped policy would be exactly the asymmetry this sheet exists to prevent.

## Stop conditions

- G1 fails -> stop, debug the instrument, change nothing else.
- G3 fails -> stop; the operator is not concept-identity-sensitive and the whole channel is in question.
- The corrected Goodreads curve still declines AND the fold effect is <= 0 at every m under both ask
  sets -> the answer is (4) GENUINE, report it as a finding and do not tune further.

## Not in scope

Paper and chapter edits (author instruction). The Phase-A gate battery. Any change to TAU_REF, C_NEG,
w_val, the +-1 clip, the fold cap, or the NPMI form — the sensitivity sweep shows none of them moves the
answer-rate story, and any of them chosen after seeing this curve is untestable tuning.
