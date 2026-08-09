# Paper A — Fable balanced review, 2026-08-09

Run after the week's re-measurement (graded question settled, three baseline rows re-selected, R5 moved to
the certified tower). Fable was given a self-contained brief with every headline number and every stated
limitation; it read no files, per HARD RULE 7.

## Verdict

| | |
|---|---|
| **Venue** | **ACM TORS, regular paper** |
| **Acceptance, conditional on the three fixes** | **55–60%** |
| **Acceptance as currently written** | **35–40%** |
| **First-round accept** | **<15%** — expect major revision then accept |

**Second choice** RecSys full paper (page budget forces cutting either the graded analysis or the
firewall; ~18–20% acceptance with a single-dataset methodological paper is a coin flip).
**Fallback** RecSys reproducibility track — fits, but frames the work as reproduction and undersells the
inversion. **Avoid** SIGIR and TOIS ("the proposed method loses to RecVAE" triggers a reflexive reject),
ECIR (audience fit worse than TORS), UMUAI (expects live users — our stated limitation bites hardest).

## The single most damaging question

> *"Isn't your 'inversion' just input-contract mismatch? RecVAE-likes-only literally cannot represent a
> negative answer, so HELF — which asks divisive questions and harvests 'no's — is starved by
> construction. You didn't discover that strategy quality is consumer-relative; you discovered that models
> fail when fed inputs their contract can't encode. And your fix is to use *your* model as the referee."*

Answerable, but **only by implication at present**. The rebuttal that must be made explicitly: Golbandi,
RBMF, TaNP and the conjugate belief model all consume the full signed answers and **still disagree among
themselves** (H>P>E0 vs E0>P>H vs flat). Two full-contract consumers with opposite orderings is Exhibit A.
→ task #95.

## What it rated genuinely strong

* The inversion is "the right experiment, done properly" — and the detail that **the classic consumer
  reproduces the classic published ordering** converts it from "we found a weird artifact" to "we located
  where published orderings come from." Rated as a result the field will have to cite.
* Evaluation hygiene "exceptional for this subfield": one harness, the Most-Popular gate proving the two
  columns are non-comparable, paired rather than unpaired bootstraps, and the explicit refusal to claim
  parity with RecVAE.
* **The simulation firewall is under-sold.** Discarding a +36% result because the answer simulator was
  circular addresses the most common objection to simulated-interview papers head-on.
* The RBMF centring decomposition called "a small gem" — it rehabilitates a folk claim ("ratings hurt")
  into a mechanism ("this preprocessing hurt").

## Ranked weaknesses

**Nothing fatal**, but closest to fatal is the contract-mismatch reading above.

**Must fix:** (1) mechanism for the inversion → #95; (2) the instrument is judge and contestant → #97;
(3) single dataset, load-bearing *because the claim is general* → #96; (4) **the tail non-win** — Golbandi's
HELF arm at .0792 is the best tail cell in the table and we do not beat it; state it rather than let a
reviewer find it → #97; (5) present the uniform tuning rule and its grids in an appendix so the RBMF
+0.0601 re-selection does not invite "so how tuned is everything else?"

**Nice to have:** k-sweep of the inversion (does it hold at k=2/16/32, or only k=8? — harness exists,
near-zero cost); demote the NATIVE column to an appendix since it forecloses an objection without changing
an ordering; make sure TaNP's 0.0060 cannot be quoted out of context.

## Editorial rulings

**(a) Deleting the falsified weight-vs-estimand paragraph — correct**, with one amendment: *cut the
conjecture, keep the experiment*. The .3580 vs .3127 gap is the cleanest measurement in the paper that
rating values carry +0.0454 inside a fixed pipeline. **Already satisfied**: that is the same computation as
the RBMF `G_raw` vs `B_allrated` contrast, and +0.0454 with its CI is in the chapter now, without the dead
hypothesis.

**(b) Leading with the inversion — correct**, and the alternative is worse: an instrument-first paper opens
with "here is a model that loses to RecVAE by 1%", a self-inflicted wound on page one. Fix for the
explain-before-measure dependency: the **introduction** states the three orderings and the classic-ordering
reproduction using *zero* instrument detail, naming the instrument only as "one of the six, described in
§3". The full table lands after §3. → #97.

## Explicitly NOT recommended

Chasing parity with RecVAE at full profile. The honest −1% framing is correct and pursuing it "would burn
months for a claim the paper doesn't need."
