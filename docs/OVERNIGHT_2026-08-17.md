# Overnight handoff — night of 2026-08-17/18

**AUTHOR SAID (verbatim, before sleeping): *"i am going to sleep, plz do not stop, keep working."***

Everything below is MINE (working notes). Governing doc: `docs/design/PAPER_B_POLICY_DESIGN.md`,
**Parts K.6 (the closing verdict) and L (the forest spec)**. Read those before acting on anything here.

## Running tonight — two agents, no contention

| agent | task | uses the box? |
|---|---|---|
| L1 → L2a | the as-if-asked encoding test, then two-teacher forest distillation | **yes** — sole occupant |
| chapter draft | prose for everything settled, into `docs/results/CHAPTER_DRAFT_SECTIONS.md` | **no** — reads only; forbidden to touch `experiments/` or the `.tex` |

Deliberately split so writing and computing proceed together. The writing agent drafts in **markdown,
not LaTeX** — the tex hazard has fired twice and unsupervised integration is not worth it.

## Where the science stands

**The policy experimental record was CLOSED this evening (Part K.6)** and then re-opened narrowly by
the author's own insight, which is Part L. The closure stands unless L overturns it.

**K.6, the resolution of the whole arc:** the tree's label is computed FROM THE STATE (a true function
of the policy's input — E1 learned it exactly, 1e-9); the per-user optimal question is computed FROM
THE HELD-OUT TARGETS, so the mapping is not a function of the input and no data supplies it. Infinite
per-user data converges back to the greedy criterion — the tree we already hold. Measured: oracle
**+0.229** · best cohort-wide question **+0.0115** · trained cold arm **+0.0153** (null vs both
controls). **~95% of the per-user prize is information the interview has not yet revealed** — which is
why interviews exist, not a modelling failure. *This is June's clairvoyant-teacher lesson in a new
costume, and I designed Part K without recognising it.*

**Part L is the author's idea, not mine**, and her sentence is the design law: **denoising IS cohort
measurement.** A grouping is admissible iff its signature is **observable at ask-time** (else
clairvoyant) and its groups are **≥~500 users** (else the −0.724 noise regime). Every failure this
week violated one; every success satisfied both.

## The two live experiments

- **L1** — volunteered STATED facts encoded as **answered history**, not belief-only. R fed evidence
  into `z` with the history empty, so the policy correctly saw "nothing asked" and was inert.
  ★ **Falsifiable prediction: a root-matching stated fact makes the policy branch exactly as the
  tree's post-root arm.** If cell 1 fails, the ENCODING is wrong, not the theory — debug the wiring.
  Redundancy must be **exactly 0 by construction**; that assert validates the mask.
- **L2a** — cold tree + the 17 **already-banked, test-grade** stated-genre orders, distilled into one
  policy with hard cohort-measured labels. ★ **Falsifier: INTERFERENCE.** If fidelity to the genre
  teacher costs fidelity to the cold tree, one policy cannot hold two teachers and the forest is dead
  at two. Both fidelities are reported together, always. T2/T3 (evidence-count and taste-popularity
  bands) authorised only if L2a passes.

## Standing constraints (unchanged, and they bind overnight)

Validation users only. **The single unspent TEST look is reserved for Q** and is not to be touched.
Certified pair frozen; policy-B mask; canonical m=0 snap `0.15686538` before any number is trusted;
paired percentile bootstrap 10,000 resamples seed 98765, never the unpaired form; `--threads 6`; one
heavy job at a time (**and exclude your own detection command when polling — that self-match has
bitten twice**); commit before running; persist per cell; print composition before spending time;
a learner below a free baseline is broken; a surprising number is a harness bug until proven otherwise.

## First things in the morning

1. **L1's verdict** — did cell 1 reproduce the tree's branch? That decides whether the encoding idea
   (and therefore L2) is real.
2. **L2a's interference number**, if it ran.
3. **The chapter draft's provenance block** — specifically *which numbers did not reproduce from their
   artifacts*. That is the most valuable output of the writing agent and the thing to read first.
4. Still reserved and unspent: **the Q test look**, to be taken at draft time, not before.
