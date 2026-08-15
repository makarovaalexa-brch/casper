# EXPERIMENT G — THE GUARD: does the policy know what NOT to ask when the vocabulary is opened?

**Status: COMPLETE, 2026-08-15. Validation only — the 10,000 canonical TEST users were never built,
never loaded, never named. `test_touched=false` on every artifact.**

Design sheet: `docs/design/PAPER_B_POLICY_DESIGN.md` **Part H.4 is governing**. Code:
`src/instrument/policy_guard.py` (committed before the run, HARD RULE 10). System under test: the
**distilled policy (E1, frozen) + the reaction head (frozen)**. No value model, no lift head, no
`Σ P·V` term anywhere in this file.

---

## THE VERDICT, FIRST — two findings, and the second is the one that constrains the chapter

**1. The guard works.** Opening the vocabulary from 1,000 to 19,390 candidates costs E1
**−0.02034 [−0.02236, −0.01837]** of full NDCG@10. Using the reaction head as a hard mask on
predicted P(no-answer) recovers **57% of that deficit** at its best threshold, **+0.01159
[+0.00976, +0.01345]**, while still spending **13.0%** of its questions outside the training pool —
and those out-of-pool picks are now answered **25.7%** of the time against **9.1%** unguarded.
**The pre-registered falsifier did NOT fire:** recovery does not require driving out-of-pool picks
to ~0%.

**2. ★ The dumb guard beats it.** The pre-registered comparator was *"if popularity alone recovers
the same deficit, the learned head adds nothing and you say so."* It does not recover the same
deficit — **it recovers more.** A state-free per-question **empirical answer rate**, budget-matched
to keep exactly as many candidates alive, recovers **78.1%** (+0.01588 [+0.01401, +0.01776]) against
the learned guard's 57.0%. Head-to-head at matched budget the learned guard **loses by −0.00429
[−0.00545, −0.00312]**, and it loses at three of the four budgets where either guard helps.

> **So the honest sentence is: opening the vocabulary is survivable, but a state-conditioned learned
> guard is not what makes it survivable — a table of per-question answer rates does the job better.**
> This is the chapter's own thesis appearing for the third time: *the bank beats conditioning*
> (+0.01430 > +0.01035), *the mean value vector beat the learned lift head*, and now *the marginal
> answer rate beats the state-conditioned reaction head as a guard*.

There is one place the learned guard wins, and it is worth keeping: **at tight budgets it degrades
gracefully and the dumb guard collapses** (below).

---

## The design, as run

| arm | what it is |
|---|---|
| **pool** | E1 restricted to its 1,000-candidate training pool. **The ceiling.** Re-run here, not quoted: it reproduces the banked `0.28354971825495506` **exactly**. |
| **open_unguarded** | E1 open to all 19,390 candidates, no guard. **The deficit.** Reproduces the banked `0.26320921379832823` exactly, through E1's own rollout code, unchanged. |
| **guard@τ** | open vocabulary; a candidate whose predicted `P(no-answer \| state, q)` exceeds τ is masked out of the argmax. **State-conditioned** — every candidate is re-scored at every one of the 16 turns. |
| **dumb@m** | open vocabulary; guarded by the **state-free per-question empirical answer rate** (the prior banked with the outcome head, computed on train users), keeping the top *m*. **Budget-matched**: *m* is exactly the number of candidates the learned guard left alive on average at the threshold it is matched to, so the two guards differ **only** in whether the guard reads the interview. |

**The sweep is declared, not tuned.** All eight thresholds are reported; no point is picked post hoc.
τ = 1.0 is the unguarded arm by construction, which makes the left end of the curve a free
reproduction check against a banked number. **Zero guard releases occurred** in any arm — the guard
never masked every candidate at any state, so nothing was silently relaxed.

**Gates:** canonical m=0 snap `0.15686538118354684` exact; `branch_of_per_user == branch_of`; leak
check 0/10,000; both banked headline numbers asserted exactly; every arm's picks mapped into one
catalogue index space so the out-of-pool accounting cannot read the wrong question; the reaction
head asserted to carry no per-question prior feature.

---

## RESULT 1 — the whole threshold curve

| arm | full NDCG@10 | tail | **deficit recovered** | out-of-pool picks | answer rate (in-pool / out) | candidates alive per state | distinct interviews |
|---|---|---|---|---|---|---|---|
| **pool (ceiling)** | **0.28355** | 0.13421 | **100%** | 0.0% | 0.5789 / — | 1,000 | 15 |
| open, unguarded | 0.26321 | 0.12181 | 0% | 29.99% | 0.5523 / **0.0909** | 19,390 | 1,168 |
| guard τ=0.999 | 0.27011 | 0.12662 | 33.9% | 18.78% | 0.5607 / 0.1486 | 6,517 | 1,541 |
| guard τ=0.99 | 0.27375 | 0.12825 | 51.8% | 14.21% | 0.5802 / 0.2216 | 2,702 | 1,730 |
| **guard τ=0.98 (best)** | **0.27480** | 0.12907 | **57.0%** | **12.98%** | 0.5903 / **0.2571** | 1,996 | 2,087 |
| guard τ=0.95 | 0.27291 | 0.12832 | 47.7% | 11.35% | 0.6110 / 0.3338 | 1,331 | 3,490 |
| guard τ=0.90 | 0.27034 | 0.12880 | 35.1% | 9.74% | 0.6497 / 0.4331 | 987 | 4,963 |
| guard τ=0.80 | 0.26341 | 0.12472 | 1.0% | 8.50% | 0.7210 / 0.5739 | 744 | 6,661 |
| guard τ=0.50 | 0.23654 | 0.10971 | −131% | 7.62% | 0.8381 / 0.7796 | 499 | 7,863 |

Every guarded arm from τ=0.999 to τ=0.90 beats the unguarded arm with a paired CI excluding zero.

**The curve is single-peaked and the mechanism is legible.** Tightening the guard buys answerability
monotonically — the out-of-pool answer rate climbs 0.091 → 0.149 → 0.222 → 0.257 → 0.334 → 0.433 →
0.574 → 0.780 — but past τ≈0.98 it starts deleting *useful* questions faster than it deletes junk.
At τ=0.50 the policy is answering 83.8% of its questions and scoring **below the empty-vocabulary
baseline it started from**: an interview where every question is answerable but none is informative.
**Answerability is a constraint, not the objective**, and this curve is the cleanest measurement of
that in the chapter.

## RESULT 2 — ★ the budget-matched dumb guard, and the head-to-head

| candidates alive | learned guard (τ) | **dumb guard** | learned − dumb, paired |
|---|---|---|---|
| 6,517 | 0.27011 (τ=0.999) | 0.27042 | −0.00032 [−0.00141, +0.00081] — **null** |
| 2,702 | 0.27375 (τ=0.99) | **0.27780** | **−0.00405 [−0.00520, −0.00289]** |
| 1,996 | 0.27480 (τ=0.98) | **0.27909** | **−0.00429 [−0.00545, −0.00312]** |
| 1,331 | 0.27291 (τ=0.95) | **0.27754** | **−0.00464 [−0.00623, −0.00307]** |
| 987 | **0.27034** (τ=0.90) | 0.25924 | **+0.01111 [+0.00890, +0.01333]** |
| 744 | **0.26341** (τ=0.80) | 0.25091 | **+0.01250 [+0.00998, +0.01506]** |
| 499 | **0.23654** (τ=0.50) | 0.22755 | **+0.00899 [+0.00689, +0.01114]** |

**Read the top half and the bottom half separately, because they say different things.**

- **In the range where guarding actually pays (6,517 → 1,331 alive), the state-free table wins**,
  three of four with intervals excluding zero and the fourth a null. Its best point recovers 78.1%
  of the deficit against the learned guard's 57.0%. *A learned, state-conditioned guard does not beat
  a per-question answer-rate table at the job of declining junk.*
- **Below ~1,000 alive the ordering reverses and the learned guard wins by a wider margin than it
  ever lost by.** The dumb guard's fixed top-*m* list collapses (−19.5%, −60.5%, −175% recovery),
  because a state-free list must delete the same candidates for everybody; the learned guard keeps a
  different 1,000 for each state and degrades gracefully. **That is a real property of conditioning
  — it just does not pay at the budget where guarding is worth doing.**

**The reason the dumb guard wins where it wins is not subtle**, and it is visible in two columns: at
matched budget the dumb guard spends **fewer** questions out of pool (8.3% vs 13.0% at m=1996) and
asks far fewer distinct never-trained questions (54 vs 279). It is a *more conservative* guard for
the same nominal budget, and on this catalogue conservatism is what pays — which is the same finding
as H.4's opening premise (catalogue-wide questions are worth ≈ nothing) arriving from the other side.

## RESULT 3 — what the falsifier was watching for, and what it found instead

**Pre-registered falsifier:** *if recovery requires driving out-of-pool picks to ~0%, the claim
collapses to "the pool was already right".* **It did not fire.** The best learned guard keeps
**13.0%** of picks out of pool and the best dumb guard keeps **8.3%**, and both are far from zero.
So opening the vocabulary is not simply a mistake to be undone.

**But the catalogue does get the last word, and it is worth stating plainly.** The one arm that
*was* driven to 0% out-of-pool — `dumb@499`, which keeps only the 499 most-answerable candidates in
the entire catalogue, every one of which turns out to be inside the training pool — scores
**0.22755**, i.e. **0.056 below the 1,000-candidate pool**. So:

> **The training pool is not the set of most-answerable questions, and it beats that set decisively.**
> The pool is top-500 items by popularity ∪ top-500 concepts by answer rate, and its *composition* —
> two channels, chosen on two different criteria — is doing work that no single answerability
> ranking reproduces. "The pool was already right" is too strong; "the pool was right for reasons a
> guard cannot see" is what the numbers support.

**No arm reaches the ceiling.** The best of all fifteen arms is still **−0.00446 [−0.00550, −0.00343]**
below pool-restricted E1. Opening the vocabulary remains a net cost on this catalogue; guarding
recovers most but not all of it.

## RESULT 4 — the guard fires behaviourally, and it changes the interview

Carrying the discipline from Experiment R — *a capability that changes no behaviour is not a result*
— the guard is not a structural gesture:

| | unguarded | guard τ=0.98 | dumb m=1996 | pool |
|---|---|---|---|---|
| distinct realised interviews | 1,168 | **2,087** | 275 | 15 |
| out-of-pool picks | 29.99% | 12.98% | 8.30% | 0% |
| distinct never-trained questions asked | 189 | **279** | 54 | 0 |
| overall answer rate | 0.4139 | 0.5471 | 0.5483 | 0.5789 |

The learned guard produces **more** distinct interviews than the unguarded policy and asks **more**
distinct never-trained questions (279 vs 189) while asking far fewer out-of-pool questions overall —
i.e. it is not narrowing to a fixed safe list, it is choosing a *different* safe list per state. That
is genuine state-conditioned behaviour. It simply does not convert into score.

---

## WHAT THIS SETTLES

1. **The open-vocabulary deficit is largely recoverable.** −0.02034 → 57% recovered by the learned
   guard, **78% by a per-question answer-rate table**, both with intervals excluding zero, both
   retaining a nonzero out-of-pool rate. The system can score any question in the catalogue and
   decline most of the junk.
2. **★ The learned state-conditioned guard loses to the state-free one where it matters**
   (−0.00429 [−0.00545, −0.00312] at matched budget, best against best). Reported as the
   pre-registration required. The reaction head's value is real (Part F: AUC 0.98243, +0.00392 inside
   a policy, and Experiment Q's generalisation to unseen questions) — but **as a hard guard it is
   beaten by counting**.
3. **Conditioning does buy robustness at tight budgets** (+0.0111 / +0.0125 / +0.0090 at 987 / 744 /
   499 alive), which is the one thing a fixed list structurally cannot do. It is a real property and
   it is not worth a threshold anyone would choose.
4. **Answerability is a constraint, not an objective.** Pushed to τ=0.50 the policy answers 83.8% of
   its questions and scores 0.0470 below the pool ceiling — worse than not guarding at all.
5. **The pool's composition beats any answerability ranking**: the 499 most-answerable candidates in
   the catalogue score 0.056 below the 1,000-candidate pool.

**What it does NOT show:** that a learned guard cannot beat a table. Under HARD RULE 6 this is a
**weak-method result about this guard** — a hard threshold on one output of a head trained for a
different objective. A guard trained *as a guard*, or the head used as a soft re-weighting rather
than a hard mask, is untested here. Neither was authorised and neither was run.

**Nothing about test.** No test user was built, loaded or touched.

---

## Cost

| | measured |
|---|---|
| pool arm (1,000 candidates) | 0.8 min |
| one unguarded open arm (19,390 candidates) | 2.7 min |
| one **learned-guard** arm (reaction head re-scores 19,390 candidates at all 16 turns) | **2.9 min** |
| one dumb-guard arm (fixed mask) | 1.7 min |
| 15 arms + all comparisons, end to end | **35.9 min** |

Part H.4 estimated 2–3 h; it ran in **36 minutes**. Third cost over-estimate of the day, recorded
rather than smoothed: the guard's extra work is one forward pass of a 2.1 MB head per turn, which is
small beside the fold the rollout was already paying for.

## Artifacts

| what | where |
|---|---|
| per-arm cells + **per-user vectors** (full, tail, asked, outcomes) | `experiments/battery/vocab_guard/{pool,open_unguarded,guard_tau*,dumb_m*}.json` (+ `_per_user.npz`) |
| combined artifact, all arms, all paired CIs | `experiments/battery/vocab_guard/vocab_guard.json` |
| log | `experiments/battery/g_main.log` |
| code | `src/instrument/policy_guard.py` |

**Reproduce:** `python src/instrument/policy_guard.py --threads 6 --i_have_author_goahead`
(every arm resumes from disk under a contract check).
