# A22 — the Goodreads "decline" was the monitor. What is left is SATURATION, not harm (2026-09-12)

Design `docs/design/DS_A22_CONCEPT_MONITOR_AND_ASKSET.md`. Instrument
`src/instrument/a22_concept_curve_paired.py`. Supersedes the negative result in
`docs/results/A17_GOODREADS_PORT.md`.

## ⛔ RETRACTION

**"On Goodreads concept answers make recommendations worse" is withdrawn.** It was an artifact of the
in-training monitor, which applies the *training* member-drop at evaluation
(`concept_train_v2.py:105-107`): items supporting an answered concept are removed from the pool before
the 4 context items are drawn, so the context is stripped of exactly the taste the targets lie on and
the penalty grows with the number of answers. Two further defects: the m-levels were unpaired (targets
redrawn per level, `:87`) and the val slice is a positional tail of a `userId`-ordered list (`:63`).

The slice defect is **not repairable for these checkpoints**: that slice is held out of training
(`:248`), so re-drawing it would evaluate users the net trained on. It is reported, not fixed.

**Gate G1 passed exactly**, all four cells to 4dp, by calling the shipped builder rather than
re-implementing it: ML-25M 0.1811 / 0.1772 / 0.1856 / 0.1916, Goodreads 0.1877 / 0.1893 / 0.1817 /
0.1746. (A17 transcribed ML-25M's m1 as 0.1774; the training log line 41 says 0.1772.)

## The corrected curves

Paired (one target draw per user, shared across m), no eval-time member-drop, 3 seeds, NDCG@10 full.
"Fold effect" is the paired within-user difference between the concept net applied and the concepts
stripped with the **context held fixed** — the honest measure of what an answer buys.

| dataset | ask set | m0 | m1 | m4 | m8 | fold effect m1/m4/m8 | scrambled m8 |
|---|---|---|---|---|---|---|---|
| ML-25M (cv3 ep4) | broad (shipped) | 0.1817 | 0.1818 | 0.1857 | **0.1854** | +0.0000 / **+0.0040** / **+0.0037** | −0.0100 |
| ML-25M | topN:562 | 0.1817 | 0.1835 | 0.1843 | 0.1842 | +0.0018 / +0.0026 / +0.0025 | −0.0082 |
| Goodreads (cvg ep12) | broad (shipped) | 0.1936 | 0.1965 | 0.1955 | **0.1948** | **+0.0029** / +0.0019 / **+0.0011** | −0.0028 |
| Goodreads | topN:61 | 0.1936 | 0.1949 | 0.1936 | **0.1859** | +0.0013 / −0.0001 / **−0.0077** | −0.0277 |

**G3 existential control passes in every cell**: permuting which concept a value is attached to (values
and refusal flags kept) drives the fold effect negative everywhere. Both operators use concept identity.

## What the corrected result actually says

- **Goodreads concept answers carry signal.** One answer buys +0.0029, and a scrambled interview is
  worse than none. The vocabulary is not the problem.
- **They do not ACCUMULATE.** ML-25M's fold effect *rises* with m (+0.0000 → +0.0040 → +0.0037);
  Goodreads' *decays* (+0.0029 → +0.0019 → +0.0011). The channel saturates after roughly the first
  answer. This is the A17 diagnostic ("one helps, accumulation fails") restated correctly: accumulation
  fails by going flat, not by doing harm.
- **The fold effect measured under member-drop is inflated by construction.** With drop on, Goodreads
  reads +0.0052/+0.0061/+0.0039 — the drop creates the deficit the concept answer then fills. Any fold
  effect quoted from the shipped monitor, on either dataset, is measuring partial repair of a wound the
  protocol inflicted. (This is where the earlier Fable analysis' positive fold numbers came from.)

## ⛔ The member-count hypothesis is REFUTED in its ask-set form

Goodreads concepts really are much narrower — median 80 members against ML-25M's 291, despite A19
matching per-*item* density exactly (37 tags/item on both). A19 matched the wrong axis. Restricting the
ask set to the broadest concepts does collapse refusal (all 140,768 train users, project gate
`n_c>=2 AND E>=TAU_REF`):

| ask set | ML-25M refusal | ML \|SEL\| | Goodreads refusal | Goodreads \|SEL\| |
|---|---|---|---|---|
| top-25 | 0.011 | 0.476 | 0.141 | 0.761 |
| top-50 | 0.027 | 0.531 | 0.223 | 0.839 |
| top-100 | 0.038 | 0.565 | 0.339 | 0.873 |
| broad (shipped) | 0.252 | 0.616 | 0.727 | 0.913 |
| uniform | 0.471 | 0.622 | 0.923 | 0.922 |

Under the **pre-registered R-rate rule** — choose N per dataset to hit ML-25M's shipped answered share
(refusal 0.252), fixed before any Goodreads NDCG was read — N* = 562 (ML-25M) and **61** (Goodreads),
both landing at refusal ≈ 0.25. **It makes Goodreads worse** (m8 fold +0.0011 → −0.0077) and costs
ML-25M a little (+0.0037 → +0.0025).

More answers, but contentless ones: Goodreads' 61 broadest concepts have a median 3,096 members, 17% of
the catalogue. This is A12's "breadth runs backwards" reproducing on a second dataset — which is a
result worth having, and it is evidence *against* answerability being the binding constraint in the
direction assumed. **`ask_pool("broad")` is not top-N**: it samples proportional to
`clip(member_count, 30, 1000)` (`concept_curriculum_v2.py:119`), and that clip is doing useful work.

## What this does and does not do to Paper B

Paper B's harness (`strategy_channel_suite.py:459-461`) uses `signed_values` with **no member-drop**, so
the +0.01430 bank result is not touched by the monitor defect. It is still exposed to the saturation
finding, since the Sep 9 decomposition traced that number to the concept answer rate. The direct test —
answers per question for concepts vs items on the deployed Goodreads harness — has not been run.

The ML-25M checkpoint selection is a separate, smaller worry: cv3 ep4 was chosen by SEL off this
monitor, and the corrected ML-25M curve is weaker than the shipped one.

## Caveats

- Top-N asking is **off-distribution** for nets trained on broad-policy draws, so its negative is a
  lower bound, not a refutation of a *rebuilt coarser bank* (which would be in-distribution). That is a
  different intervention and has not been tested.
- 3 seeds; seed sd on the m-cells is 0.002–0.005, comparable to the effects. The fold effect is paired
  within seed and therefore much tighter, which is why it is the statistic quoted.
- Goodreads m0 (0.1936) sits above ML-25M's (0.1817), so the two channels are not compared at equal
  starting difficulty.
