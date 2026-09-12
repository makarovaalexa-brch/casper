> **ADDENDA 1 AND 2 ARE SUPERSEDED AS DATASET STATEMENTS, 2026-09-12.** They ran on `cvg_ep12`
> and `cv3_ep04`, both chosen by the bent monitor and both weak. Their measurements stand; their
> reading does not. Re-run on the re-selected epochs: `docs/results/A24_RESELECTED_DIAGNOSTICS.md`.
> **The channel PORTS** (+0.0016 GR vs +0.0015 ML over all 12 epochs).

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

## ⛔ ADDENDUM — on Goodreads the ANSWERS buy nothing; the REFUSALS carry what little there is

`--no_refusals` feeds only the concepts the user actually answered, so the refusal tokens can be
priced separately (they are fed to the net: `use_refusal=True`). Corrected protocol, 3 seeds:

| | fold effect m1 / m4 / m8 |
|---|---|
| ML-25M, refusals fed | +0.0000 / **+0.0040** / **+0.0037** |
| ML-25M, answers only | −0.0005 / **+0.0036** / **+0.0033** |
| Goodreads, refusals fed | +0.0029 / +0.0019 / **+0.0011** |
| Goodreads, answers only | −0.0003 / −0.0006 / **−0.0008** |

On ML-25M stripping the refusals costs almost nothing: the ANSWERS carry the lift. On Goodreads it
reverses — the answered concepts contribute nothing and the entire (small) positive comes from the
REFUSAL PATTERN, i.e. from learning which shelves the user cannot speak to.

**This is not dilution.** A user with no tokens has a fold effect of exactly 0 by construction, so at
m1 (73% refusal) the answers-only mean is indeed diluted. But at m8, P(at least one answer)
= 1 − 0.727^8 = **0.98** — almost nobody is zeroed — and the answers-only fold effect is still
slightly negative.

### The pincer, and why no ask-policy fix works

Put together with the top-N result, the Goodreads concept vocabulary is caught between two failures:

- **Narrow shelves are informative but unanswerable.** Median concept = 80 of 18,430 books; broad-policy
  refusal 0.727 against ML-25M's 0.252.
- **Broad shelves are answerable but contentless.** Forcing the answer rate to ML-25M's by asking only
  the 61 broadest (median 3,096 members, 17% of the catalogue) drives the m8 fold effect to −0.0077.

No *ask policy* escapes a pincer in the bank itself, which is why R-rate failed. **Untested: whether a
middle band exists** — concepts large enough to answer and small enough to discriminate. That is the
next experiment and the last cheap route to making the channel work on Goodreads.

### Training never supplies the multi-answer states either

20,000 draws from the live curriculum per dataset, answered concepts per training example:

| regime (share) | ML-25M | Goodreads |
|---|---|---|
| dense (30%) | 4.50 | 4.42 |
| interview_broad (18%) | 7.25 | **2.44** |
| interview_uniform (18%) | 5.02 | **0.63** |
| interview_niche (18%) | 2.51 | **0.10** |
| **P(≥8 answers), overall** | **0.156** | **0.053** |

The dense bucket is identical by construction — it conditions on answerable concepts
(`concept_curriculum_v2.py:213`). The three *interview* regimes, the deployment-shaped 55% of training,
collapse. So the net is also trained on far fewer multi-answer interview states, which is a plausible
contributor to the saturation but cannot be the whole story, since the answers buy nothing even at m8.

## ⛔ ADDENDUM 2 — the middle band is a null, and EQUAL ANSWERS settles it

**Middle band.** Informativeness is flat at |SEL| ≈ 0.95 across every Goodreads band except the top-61
(0.843), so rank 61–200 (refusal 0.584, median 1,095 members) was the only candidate for "large enough
to answer, small enough to discriminate". It is a null:

| Goodreads ask set | fold m1 / m4 / m8 | scrambled m8 |
|---|---|---|
| broad (shipped) | +0.0029 / +0.0019 / +0.0011 | −0.0028 |
| topN:61 | +0.0013 / −0.0001 / **−0.0077** | −0.0277 |
| band:61–200 | +0.0033 / +0.0013 / +0.0013 | −0.0040 |
| *ML-25M broad, for scale* | *+0.0000 / +0.0040 / +0.0037* | *−0.0100* |

Also: the **structural rule is unusable**. Matching ML-25M's member-count IQR (124–712) on Goodreads
keeps 669 concepts at refusal **0.892** — worse than the shipped policy. Goodreads cannot be made to
look like ML-25M by any selection over its own bank.

**Equal answers, not equal questions.** Goodreads collects ~0.27 answers per broad ask against ML-25M's
~0.75, so m=8 compares 2.2 answers with 6.0. Extending the Goodreads interview:

| m asked | expected answers | NDCG | fold effect |
|---|---|---|---|
| 0 | 0.0 | 0.1936 | — |
| 8 | 2.2 | 0.1938 | +0.0002 |
| 16 | 4.4 | 0.1934 | −0.0003 |
| **22** | **6.0** (= ML-25M's m=8) | 0.1936 | **−0.0001** |
| 32 | 8.7 | 0.1914 | −0.0023 |

**At the answer count ML-25M converts into +0.0037, Goodreads converts into −0.0001.** The answer-rate
explanation — the leading implementation story, and the one the member-count hypothesis rested on — is
refuted. It is not starvation.

## Where this leaves the diagnosis

Eliminated, each by measurement rather than argument:

| explanation | verdict |
|---|---|
| monitor protocol | **CONFIRMED as the cause of the reported DECLINE**; corrected, the curve is flat |
| answer-rate starvation | **REFUTED** — equal-answer depth buys nothing |
| concept narrowness / ask policy | **REFUTED** — broad, top-N and middle band all land in the same place |
| answer-value scale, fold saturation | not implicated (halving values n.s.; 6% of Goodreads updates hit the cap vs 72% on ML-25M) |
| genuine | **the surviving explanation** |

The honest claim is now: *the concept channel's mechanism does not port to Goodreads. Under a corrected
protocol it neither helps nor harms — the fold effect is +0.001 at 8 questions and −0.0001 at the answer
count ML-25M converts into +0.0037 — and what little positive there is comes from refusals rather than
answers. This is not an answer-rate effect: it survives equalising answers, and no selection over the
shelf bank recovers it.* The item channel, arena, split, teacher and tower are untouched.

⚠ Scope the claim to **the shelf vocabulary we mined**, not to books in general. A different attribute
source (genre/subject metadata rather than user shelves) is untested.

**The one untested intervention** is a retrain under a corrected curriculum and monitor — the only
in-distribution test. Its prior is low: the dense bucket already supplies 4.4 answered concepts per
example on Goodreads, so the net has seen multi-answer states, and the answers buy nothing when present
(answers-only fold −0.0008 at m8). It is ~68 min and it would change the certified artifact set.

## ⛔⛔ ADDENDUM 3 — THE CHANNEL PORTS. Addendum 2's "does not port" is RETRACTED.

Addendum 2 concluded that the concept channel does not port to Goodreads and that explanation (4),
genuine, was what survived. **That was wrong, and the error was mine: I compared `cvg_ep12` against
`cv3_ep04` — two checkpoints BOTH chosen by the bent ruler — and then ran every diagnostic on
`cvg_ep12` alone.** A22's own finding (the monitor is bent) implies its *selections* are bent too, which
I noted for ML-25M and then failed to apply before drawing a cross-dataset conclusion.

`a23_reselect_concept_epoch.py` re-scores all 12 epochs of each run on the corrected protocol. No
retraining: the monitor is eval-only and feeds no gradients, so the checkpoints on disk are untouched.
Run twice per dataset on **independent seed sets** (4242/5242 and 777/1777), because picking the best of
12 on the same seeds that ranked them is selection on noise.

Mean paired fold effect over m in (1,4,8):

| epoch | ML s4242 | ML s777 | GR s4242 | GR s777 |
|---|---|---|---|---|
| ep1 | −0.0004 | −0.0004 | **+0.0028** | **+0.0035** |
| ep2 | +0.0003 | +0.0018 | +0.0025 | +0.0018 |
| ep3 | +0.0011 | +0.0032 | +0.0027 | +0.0016 |
| ep4 *(certified ML)* | +0.0006 | +0.0025 | +0.0016 | −0.0003 |
| ep5 | +0.0009 | +0.0010 | −0.0006 | +0.0012 |
| ep6 | +0.0012 | +0.0029 | +0.0001 | +0.0006 |
| ep7 | +0.0018 | +0.0020 | +0.0010 | +0.0017 |
| ep8 | +0.0003 | +0.0028 | +0.0008 | +0.0013 |
| ep9 | +0.0007 | +0.0014 | +0.0000 | +0.0023 |
| ep10 | −0.0002 | +0.0030 | **+0.0027** | **+0.0036** |
| ep11 | +0.0024 | +0.0020 | +0.0012 | +0.0024 |
| ep12 *(certified GR)* | +0.0023 | +0.0035 | +0.0014 | +0.0024 |

**The selection-free comparison, which is the one to quote:**

| | mean over all 12 epochs | out-of-sample best | rank stability across seeds |
|---|---|---|---|
| ML-25M | +0.0009 / +0.0021 → **+0.0015** | ep11 → **+0.0020** | Spearman 0.32 |
| Goodreads | +0.0014 / +0.0018 → **+0.0016** | ep1 → **+0.0035** | Spearman 0.51 |

**+0.0015 vs +0.0016.** The datasets are indistinguishable, and on the out-of-sample best epoch
Goodreads is ahead. **The concept channel ports.** The apparent failure was two compounding artifacts:
eval-time member-drop, which manufactured the decline, and bent-ruler checkpoint selection, which then
compared one of ML-25M's *weakest* epochs (ep4, +0.0006) against a mediocre Goodreads one (ep12).

### What this does NOT overturn

- The **retraction of the original negative** stands, and is now doubly supported.
- The **pincer measurements** stand as measurements: Goodreads concepts really are narrower (median 80
  vs 291), refusal really is 0.727 vs 0.252, and top-N asking really does hurt (−0.0077 on ep12). They
  are simply no longer evidence of a portability failure.
- The **member-count hypothesis remains refuted as a fix** — restricting the ask set makes things worse
  on both datasets.

### What is now in question, and must be re-run before anything is claimed

Every diagnostic in Addenda 1 and 2 ran on `cvg_ep12` (fold +0.0014, a below-average epoch) or on
`cv3_ep04` (+0.0006, near ML-25M's worst). **They must be repeated on the re-selected epochs**:

1. the equal-ANSWER curve (m = 8/16/22/32), which gave −0.0001 at m22 on ep12;
2. the refusals-vs-answers decomposition, which said Goodreads' lift comes entirely from refusals;
3. the ask-set arm (topN, band).

Until then, treat Addenda 1 and 2's *mechanisms* as provisional. Their *measurements* are sound; what
is unsafe is reading them as properties of the datasets rather than of two badly-chosen checkpoints.

### Method lesson

Epoch-to-epoch spread (ML −0.0004 to +0.0035) is **larger than the effect being measured**, and rank
stability across seed sets is weak-to-moderate (Spearman 0.32/0.51). Any cross-dataset claim about this
channel must be made on the mean over epochs or on an out-of-sample-selected epoch — never on a single
checkpoint, and never on one selected by a different ruler than the one reporting the result.
