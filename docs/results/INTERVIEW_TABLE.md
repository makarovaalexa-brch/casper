# The interview table: strategies × recommenders, and where Golbandi's advantage comes from

**2026-07-30.** Six lit-anchored selection strategies (Rashid 2002/2008) × five recommenders,
budgets k ∈ {1,2,4,8,16}, on the arm-N protocol pool. Fable consulted twice; both verdicts in
`experiments/baselines/interview/`. Companion: `ARM_N_PHASE1.md` (full profile).

---

## 1. Protocol

Candidate pool = the arm-N pool (every rated fold-in-side item), **fixed and strategy-independent**. A
strategy-dependent pool would delete popular items from the popularity arm's own candidate list — a
confound landing on the arm the literature ranks third. An asked-but-unrated item **burns the
question** (the lit protocol) but stays rankable.

**Most-Popular floor**, uniform: a model told nothing must not score below showing popular items.
Applied where the model **receives no input** — not where the user gave no like/dislike. Golbandi
consumes unknowns, so he always has input and is never floored. Our instrument folds only *answered*
tokens, so it is. Uniform as a rule, asymmetric in outcome, and that asymmetry is the finding of §3.

Most-Popular prior = **0.1626** (constant: it ignores answers, so it is one number, not a curve).

## 2. The table (full NDCG@10)

| strategy (answers/8) | model | k=1 | k=2 | k=4 | k=8 | k=16 |
|---|---|---|---|---|---|---|
| **HELF** (1.6) | ours | 0.1632 | 0.1668 | 0.1802 | 0.1899 | 0.2122 |
| | golbandi | **0.1665** | **0.1739** | **0.1890** | **0.1948** | — |
| | backbone | 0.1141 | 0.0991 | 0.1087 | 0.1413 | 0.1813 |
| | rbmf | 0.1165 | 0.1034 | 0.0994 | 0.1127 | 0.1481 |
| | belief-MF | 0.1620 | 0.1617 | 0.1605 | 0.1597 | 0.1633 |
| **entropy0** (3.1) | ours | **0.1646** | **0.1685** | **0.1765** | **0.1935** | 0.2146 |
| | golbandi | 0.1579 | 0.1563 | 0.1680 | 0.1686 | — |
| | backbone | 0.1284 | 0.1294 | 0.1580 | 0.1777 | **0.2238** |
| **popularity** (3.1) | ours | **0.1597** | **0.1627** | **0.1693** | **0.1775** | 0.2015 |
| | golbandi | 0.1531 | 0.1534 | 0.1445 | 0.1716 | — |
| | backbone | 0.1355 | 0.1295 | 0.1372 | 0.1817 | **0.2182** |
| **useless arms** (0.0–0.1) | ours | 0.1628 | 0.1628 | 0.1634 | 0.1642 | 0.1651 |
| | golbandi | 0.1635 | 0.1635 | 0.1643 | 0.1664 | — |
| | backbone | 0.1613 | 0.1609 | 0.1573 | **0.1512** | **0.1454** |
| | rbmf | 0.1613 | 0.1593 | 0.1563 | **0.1500** | **0.1427** |

**Golbandi cannot run past k=8**: 3^k answer patterns exhaust 140,768 training users, and the shrunk
mean degenerates to the global mean — Most-Popular in disguise, not the method.

### Findings from the table itself

1. **Strategy ranking is recommender-dependent.** HELF wins on Golbandi's node model (Rashid's published
   order); **entropy0 wins on ours**. Same questions, users, answers, pool — different conclusion. A
   paper evaluating strategies through a weak recommender draws a recommender-specific conclusion.
2. **The backbone and RBMF collapse in the scarce regime** — 0.0991 / 0.0994, *six points below the
   prior*, and they **degrade as useless questions accumulate** (0.1613 → 0.1454). An instrument that
   gets worse the more you ask is unusable.
3. **belief-MF is inert**: 0.1597–0.1665 everywhere. VERIFIED not a harness bug — 55,247 answers reach
   it, 9,220 users have ≥1, values span −1..+1, scores shift 11% relative — but Spearman against the
   pure-prior scores is **0.995**. A 200-dim Gaussian posterior with an empirical prior does not reorder
   on ~5 observations. That is a property of the ConTS line.
4. **The backbone wins the long end** (0.2238 / 0.2182 at k=16). Together with the full-profile
   −0.0057, our instrument is below its own backbone wherever evidence is abundant. Its value is the
   scarce regime plus the interface. Write that as the finding, not around it.

## 3. ★★★ WHERE GOLBANDI'S ADVANTAGE COMES FROM — the ablation

**The clue was in the table and I misread it twice.** He wins on the strategy with the FEWEST answers
(HELF, 1.6/8) and loses on those with the most (entropy0/popularity, 3.1/8). The gap is monotone in the
fraction of the interview we discard. Under HELF he uses all 8 observations; we use 1.6.

*(An earlier measurement claiming the unknown signal was worth 0.0007 was taken on the pure-entropy and
random arms, where NOBODY answers — so the unknown pattern is identical across users and carries no
information by construction. Wrong arm. It is worth 10× that where it can discriminate.)*

Three routings, same leaf machinery, same everything else:

| | HELF k=2 | k=4 | k=8 | entropy0 k=2 | k=4 | k=8 |
|---|---|---|---|---|---|---|
| **ternary** (published: like/dislike/unknown) | 0.1739 | 0.1890 | **0.1948** | 0.1563 | 0.1680 | 0.1686 |
| **no_unknown** (answered items only) | 0.1714 | 0.1823 | 0.1867 | 0.1688 | 0.1766 | **0.1879** |
| **no_sign** (dislike merged into unknown = the Liang view) | 0.1681 | 0.1741 | 0.1881 | 0.1541 | 0.1575 | 0.1624 |
| *ours, for reference* | 0.1668 | 0.1802 | **0.1899** | **0.1685** | **0.1765** | **0.1935** |

**(a) The unknown branch is the whole of his k=8 HELF advantage.** Worth **+0.0081** at k=8 on HELF
(0.1948 → 0.1867). Strip it and **we beat him: 0.1899 vs 0.1867.** At k=2/k=4 a smaller residual gap
remains (0.1714 vs 0.1668; 0.1823 vs 0.1802) — that is the shrinkage/calibration mechanism, not the
non-answer.

**(b) The unknown branch HURTS him when answers are plentiful**: −0.0193 at k=8 on entropy0
(0.1686 → 0.1879). Matching on 3^k patterns over-fragments the leaf once real answers arrive. So the
non-answer is valuable *exactly and only* in the scarce regime — the same boundary as sign, intensity,
and the backbone ablation.

**(c) ★ SIGN IS WORTH +0.0067 to +0.0149 TO A 2011 DECISION TREE.** Merging dislike into unknown — which
is *precisely* what the canonical Liang protocol does to every model — costs him 0.0149 at HELF k=4 and
0.0105 at entropy0 k=4. **This is independent corroboration of R5 from an architecture with nothing in
common with ours**, and it converts the protocol argument from an argument into a measurement: the
`r > 3.5` discard demonstrably costs a published method real accuracy in the interview regime.

## 4. What this implies

**The fix for the short end is to stop burning the non-answer.** Our fold consumes only answered
tokens; an unanswered question contributes literally nothing. Under HELF that discards 80% of the
interview. Candidate: treat "asked, unrated" as an observation with an exposure-model likelihood
(ExpoMF-style, estimable from training data), folded through the belief — inference-only if the
closed-form update admits a new observation type.

**R4 is claimed but not used at inference.** The scoring path is `z @ Wd.T + bd` — a raw encoder point
estimate. `belief_layer` is imported once, for concept directions, never for scoring. Whatever it buys
in NDCG, the claim-versus-code gap must be closed.

**The aikido option, if R2 is real:** insert the leaf like-rate as an evidence token through the open
set-valued interface. Ingesting the competitor's statistic and matching him with it converts the threat
into the R2 demonstration. If the interface cannot accept a new evidence type without retraining, R2 is
weaker than the chapter claims — better learned now.

## 5. Owed before any of this is written up

1. **Paired per-user bootstrap CIs on every cell.** The k=8 "tie" (0.1948 vs 0.1935) is currently an
   *unmeasured* claim — the same error as the full-profile parity claim corrected this morning.
2. **Gated-backbone control**: the backbone wrapped in a shrinkage-to-prior fallback. If it matches our
   robustness, the robustness headline is dead and only the low-k premium survives.
3. **(ours − backbone) vs answered-token count**, 1 → full profile. If it converges monotonically to
   −0.0057 the long-end gap is the interface cost and is structural; a mid-range dip means a fixable
   calibration problem.
4. **Answers-obtained column** beside NDCG. Legitimate context; NOT a defence — for Golbandi the
   unanswered question is not burned at all, which is an indictment of ours, not of his.
