# Fable brief — a vision for the rest of Paper B (the flexible-interview arc)

**Self-contained. Do not read the repo.** Author's request, 2026-08-17.

## What the author wants (her words, condensed but faithful)

*"We want flexible, deployable interviews. So: we want a model more capable than a single small
tree. We want it to adapt to a multitude of likely situations (if you say it is impossible to learn
from each user, maybe that negative result needs to go into the chapter, maybe not the paper). We
want those multiple situations to be supported by data, not just a lucky guess. We want to tie in
Reddit as a beautiful illustration of adaptivity."*

Her three target claims:
1. **Our policy beats all baselines** — beats static AND beats the cold NDCG tree — **when there is
   volunteered information.**
2. **Question order genuinely changes with the opener**, measured by counting trajectories. If it
   also makes semantic sense, better still.
3. **The recommender takes the volunteered info into account** and gives better-than-popularity
   recommendations from it alone.

She wants a story that ties these together. **She is unhappy with the chapter as it stands** — her
verdict: *"what we've done is just built a smallish, shallow tree on ndcg. That tree does NOT
generalise. My vision for an interview where you can ask any questions in any order and leverage
volunteered info is NOT implemented."*

## The measured record (validation unless stated; all paired percentile bootstrap, 10k, seed 98765)

Ruler: Liang-recipe ML-25M, 18,359 items, 140,768/10,000/10,000, candidate-mask policy B, certified
frozen tower `t2i26c_ep31` + concept channel `concept_cv3_ep08`. NDCG@10 full and tail.

| arm | full |
|---|---|
| ask nothing (m=0 snap) | 0.15687 |
| best static combined order, K=16 | 0.25561 |
| **cold ternary tree (the incumbent teacher)** | **0.28355** |

- **Tree on TEST vs rebuilt static comparator: +0.01035 full / +0.00705 tail.** Bank-alone (concept
  bank, no conditioning) is **+0.01430 / +0.02199** — i.e. *the bank beats the conditioning*.
- **E1** (parametric policy distilled from the tree): top-1 **1.0000 at all 16 positions**;
  `frac_never_deviates = 1.0`. An exact functional replica. Buys no gain over its teacher.
- **Tree depth is the stopping rule, not the data**: all 15 stops are min-leaf floor stops with
  positive gain in hand. `Spearman(n_users, gain) = −0.724` — smaller cohorts report larger gains,
  so the 200-user floor is roughly where measurement stops being trustworthy.
- **K.6 (per-user labels are clairvoyant):** per-user oracle **+0.19366**, best cohort-wide question
  **+0.009664**. ~95% of the per-user prize is information the interview has not yet revealed. The
  per-user optimal question is a function of held-out targets, not of the state ⇒ not learnable.
- **L1 (as-if-asked encoding):** a volunteered fact encoded as *answered history* (token + fold +
  mask) makes the policy branch as the tree's post-root arm for **100.00%** of users; per-user diff
  from banked exactly 0.0; the saved turn is worth **+0.00167 [+0.00096, +0.00238]**. Off-manifold
  it improvises and loses: **−0.047** on random concepts.
- **L2a (two-teacher forest):** one policy holds the cold tree AND 17 stated-genre orders with
  **zero interference** (T0 fidelity 1.000000→0.999969; T1 0.000369→0.999963). **But the genre
  teacher LOSES to the tree at its own states: −0.026 / −0.033 / −0.031** (Drama/Action/Comedy).
  Rebuilt over the tree's own combined pool it **still lost on holdout, 0 of 4 groups won**.
- **R (volunteered evidence, belief-only encoding):** the policy ran its unseeded interview for
  98.99 / 97.52 / 94.89% of users at k = 2/4/8 volunteered items; redundancy parity with arms that
  cannot see the evidence. **The evidence pays through the FOLD (+0.0206 full / +0.0353 tail at
  k=8) with identical questions** — i.e. the recommender uses it, the policy does not.
- **Reaction head:** predicts answers to questions it has never seen, held-out-question AUC
  **0.98167** vs a fair channel×popularity-decile backoff **0.96285**. The one capability that fires.

## Why the genre-forest route as specified is dead (author's own objection, and it is correct)

The 17 stated-genre teachers are **static orders after the genre is stated** — not trees. They are
already banked and test-grade, so they are free, but L2a measured them **losing to the cold tree at
their own states**. Distilling them would teach the policy to be worse where it is used. Also, the
author objects that genre is a **heuristic**: we have never shown genre is a good opener at all —
maybe it is mood, or decade, or something else.

## The two candidate replacements (please rule)

**(A) Latent clustering of folded evidence** — the statistical form of the author's own suggestion
(*"maybe cluster user profiles after folding in their rating? there must be some statistical ways to
cover multiple bases"*). Fold whatever the user volunteered into the tower to
get `z`, cluster that space, and grow **one tree per cluster** (adaptive, not a static order). The
signature — "which cluster does your folded evidence land in" — is computed from evidence
**at ask time**, so it satisfies the admissibility law, and it is fully statistical: no semantic
heuristic anywhere. Choosing k controls cohort size (10,000 / 20 ≈ 500, exactly the floor). It
subsumes genre: if genre is what matters, the clusters separate by genre on their own.

**(B) Mine the opener distribution from Reddit.** The corpus of real requests tells us **what people
actually volunteer, and how often** — genre, mood, decade, exemplar films, negative constraints.
That is empirical evidence for which openers deserve teachers, which is precisely the author's
"supported by data, not just a lucky guess". Reddit stops being a bolt-on illustration and becomes
the thing that **chooses the forest's axes**.

These compose: (B) says which situations are real, (A) builds a measured teacher for each without
hand-picked semantics.

## The admissibility law already established (author's, and it has held all week)

A grouping is admissible iff (i) its signature is **observable at ask time** (else clairvoyant) and
(ii) its groups are **≥ ~500 users** (else the −0.724 noise regime). L2 added a third condition
measured the hard way: (iii) **a teacher must BEAT THE INCUMBENT at its own states** — admissibility
buys learnability, not value.

## What we are asking you for

1. **A vision for the rest of Paper B** that delivers the author's three claims, or the honest
   statement of which of the three cannot be delivered and why.
2. **Rule on (A) and (B)** — and say what the decisive cheap experiment is for whichever survives.
3. **Where does the K.6 negative go** — chapter, paper, both, neither? The author has said this
   chapter should not carry limitations: *"either we show it working or we omit it."*
4. **Is claim 1 even reachable?** Beating the cold tree *at volunteered-evidence states* is the
   crux. L1 showed the tree's own machinery already handles a volunteered fact it would have asked
   about (exactly, 100.00%). So the gain must come from facts the tree does **not** ask about. Is
   that a real gap or a measure-zero one?
5. **VENUE AND PUBLISHABILITY (author's explicit ask).** Which venues does this work have a real
   chance at, and what would have to be true of the results for it to be a *good* publication rather
   than a thin one? Be concrete about the gap between what we hold now and what each venue needs.
   Prior discussion had TORS or a RecSys/ECIR reproducibility track in view, and an ECIR 2027
   deadline of 2 Oct 2026 was noted in an earlier (now superseded) plan — treat those as starting
   points to confirm or overturn, not as settled. Paper A is closed at 41pp and not yet submitted;
   this is Paper B of a thesis, so **separate the thesis-chapter bar from the conference-paper bar**
   and say what each requires.
6. **Constraints that bind:** no LLM API calls without author approval; never the Reddit API; never
   votes as graded labels; the single unspent TEST look is reserved for Q; nulls are not citable.
