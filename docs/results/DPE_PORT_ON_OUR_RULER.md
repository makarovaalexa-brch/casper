# DPE on our ruler: a diversity criterion that gets separably worse with every question

Run 2026-08-08, `experiments/baselines/interview/adaptive_table_dpe_v1.json` (+ `_per_user.npz`).
Selector `src/baselines/dpe_selector.py`. Contract: canonical Liang ML-25M, policy-B pool,
NO-FLOOR, behavioural answers, all 10,000 test users, consumer = certified tower i26c ep31 +
concept channel cv3 ep08, concept answers under `--concept_neg_cap 0` (certified-era).
CIs in `docs/results/PORT_PAIRED_CIS.md` (10,000 resamples, seed 98765). MostPop prior .1626/.0262.

Target: Parapar & Radlinski, *"Diverse User Preference Elicitation with Multi-Armed Bandits"*,
WSDM 2021.

## Three facts about the published method, established from its full text

1. **★ DPE does NOT ask about topics.** Its bandit selects a *topic*, then asks about an **item**
   from within that topic (Alg. 1: `next_topic ← argmax_a θ_a` … `next_item ← argmax_i δ_i`). The
   user is never asked a topic question. Our design docs had called it "our only named
   concept-channel rival"; **that was wrong and is corrected**. Its faithful row belongs in the
   **item** column, annotated *"topical selection criterion"*. `ConLinUCB-BS-static` remains the
   only published selector that actually asks key-terms.
2. **The paper reports no scalar numbers.** Table 1 is dataset statistics; every result is a curve
   (Fig. 1 ML-20M, Fig. 2 Amazon). Only *orderings* are snappable — any "DPE number" quoted
   anywhere would be manufactured.
3. **★ Its protocol has a 100% answer rate by construction.** Users are filtered to >100 ratings
   *"given that our evaluation elicits up to 75 preferences per user"*, and answers come from the
   user's own ratings. **DPE never models an unanswerable question.** This makes it the cleanest
   example in the battery of a named *adaptive* method whose protocol assumes away the phenomenon
   this chapter measures — a better citation than a baseline row.

**Setup snap PASSES, and recovered an undocumented off-by-one.** Their prose says "removing those
with fewer than 100 ratings" (keep ≥100) → 52,596 users. Keeping **strictly >100** → **51,869 =
their published count exactly**. Items 26,654 ✓, ratings/user 308 ✓, categories 20 ✓ (they keep
`(no genres listed)` as a topic). **Any DPE reproduction must use `>100`.**

## The cells

| arm | k=1 | k=2 | k=4 | k=8 | k=16 |
|---|---|---|---|---|---|
| **DPE-ILD** (Eq. 2 diversity) | .1627/.0398 | .1621/.0401 | .1613/.0397 | .1609/.0390 | **.1603/.0375** |
| DPE-Hellinger (Eq. 1) | .1627/.0398 | .1680/.0480 | .1817/.0610 | .2002/.0765 | .2192/**.0925** |
| DPE-concept *(our labelled deviation)* | .1621/.0344 | .1661/.0406 | .1729/.0499 | .1832/.0615 | .2018/.0795 |
| **DPE-oracle** *(CEILING, not a baseline)* | .1893/.0680 | .2194/.0965 | .2544/.1281 | .2930/.1597 | **.3292/.1927** |

## ★ Finding 1 — DPE-ILD is the only arm in the battery that gets separably WORSE

`k=16 − k=1` is **−.0024 [−.0032,−.0015] full** and **−.0024 [−.0034,−.0013] tail**, both separated
below zero, while all eight control arms move separably *positive* (+.040 to +.069). The magnitude
is small — report it as *"the only arm that gets separably worse"*, never as a large effect.

The mechanism is corroborated harder than expected: its **answered count goes 0.3176 at k=1 to
0.3205 at k=16 — fifteen questions bought 0.003 answers.** Maximising intra-list distance from the
profile selects items nobody has seen, so after the popularity-seeded first pick the interview
collects essentially nothing and the folded exposure tokens slightly degrade the state. This is the
same structural failure the CLiME width rule showed from the other side: an uncertainty- or
diversity-driven criterion, unweighted by answerability, walks straight into the unanswerable tail.

## ★ Finding 2 — the oracle ceiling: selection has large headroom, methods aren't reaching it

`dpe_oracle` (Eq. 3) reads the user's own ratings at selection time, so it is a **ceiling and never
a baseline** — labelled as such in code, JSON and every table. It reaches **.3292/.1927** at k=16
where the best real method (greedy items-only) reaches .2449/.0967:

* vs greedy items-only: **+.0843 [+.0804,+.0881] full**, **+.0960 [+.0917,+.1002] tail**
* vs entropy₀: +.0989 / +.1002
* separated at **all 20 cells**.

This is the strongest available answer to *"selection doesn't matter"*: it matters enormously.
The Act-1 null is therefore not a statement that the ceiling is low — it is that **published
methods capture a fraction of a large available gain**. Note also that the oracle's advantage under
our contract is substantially *answerability*: it collects 3.25 answers per 4 questions where
DPE-ILD collects 0.25.

## Finding 3 — Hellinger is the respectable arm, and the axes disagree about it

DPE-Hellinger reaches .2192/.0925 at k=16. Against entropy₀ it **loses separably on full**
(−.0111) while its **tail is a genuine dead heat** (+0.0001, CI [−.0026,+.0029], **ns**). Two arms
of the same published method land at opposite ends of the battery purely on the choice of
diversity measure — worth one sentence, since it says the criterion, not the bandit, is doing the
work.

## What is ours and labelled as such

`dpe_concept` runs DPE's Thompson bandit over the genome concept action space, so the *question* is
a concept. **This is our deviation, not DPE** — stamped in code, docs and cell metadata. It reaches
.2018/.0795, above the published concept selector (`ConLinUCB-BS-static`, .1891/.0700) and below
our greedy concepts-only sequence.

## Owed

* Re-run with `fixedorder:` comparators in the same invocation so the gaps against entropy₀ and
  greedy are exact by construction rather than admitted on a cross-run control (which passed at
  0.0e+00, but in-run is strictly better and is now available).
* The ordering snap against their own baselines (`Greedy`, `Greedy+MMR`) was **not attempted** —
  it needs WSR, which we do not implement, and substituting a recommender makes the snap a test of
  the substitute. Stated rather than faked.
