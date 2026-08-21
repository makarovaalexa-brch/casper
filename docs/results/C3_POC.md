# C3 PoC — retrieval-anchored rendering with blinded controls (variant a)

**2026-08-21/22. Design sheet `docs/design/DS_C3_POC.md` (pre-registered, incl. the re-encode
bridge settled before the stage ran). Author instruction: "proceed as far as you can with
retrieval based"; conclusions below are Fable's own, from the unblinded join — the executing
subagents never saw `KEY.json`.**

Pipeline: `c3_poc_extract.py` (30 blinded packets: 10 real points — 5 blend targets, 5 fine-tuned
emissions — plus 10 scrambled-correspondence and 10 spectrum-matched random controls; commit
`f7d0275`) → blinded Sonnet renderer (`renders.json`) → blinded Sonnet judge on question text only
(`judgments.json`) → SBERT→concept→k(φ) re-encode (`c3_poc_reencode.py`, commit `44b375f`; tag
alignment asserted at all 1,031 positions; banked anchors reproduced exactly; model from local HF
cache, no download). All 30 questions rendered; all 30 encodable (top-4 concepts ≥ 0.30).

## The unblinded table

| type (n=10) | re-encode cos (uniform) | anchor-reference cos | renderability (self) | answerability (judge) | P(judged "real") |
|---|---|---|---|---|---|
| real | **+0.528** (med +0.472) | +0.578 | 3.20 | 4.00 | **0.20** |
| scrambled | **+0.514** (med +0.466) | +0.578 | 3.20 | 4.00 | 0.60 |
| random | +0.371 (med +0.486) | +0.345 | 2.60 | 3.40 | 0.50 |

Judge: **11/30 correct (0.367)** against an all-control baseline of 0.667; true positives 2/10;
within real/scrambled twin pairs it picked the real one and rejected its twin in **2/10** pairs.

## Conclusions (Fable)

1. **Registered prediction 2 CONFIRMED: fluency discriminates nothing.** The judge performed at or
   below chance everywhere, and its stated heuristic (sharper wording = real) pointed the wrong
   way — real packets were judged "real" *least* often (0.20). Rendered text alone carries no
   signal about whether the underlying direction was genuine. This is the audit's premise
   demonstrated on our own instrument.
2. **The scrambled control CANNOT separate under variant (a) — structurally, and now measurably.**
   Real and scrambled re-encode identically (+0.528 vs +0.514; anchor rows identical *by
   construction*). Code review before unblinding identified why: the packets scramble
   point↔context, but the anchors travel with the point, so the retrieval chain point→anchors→text
   is intact for scrambled packets. There is no way to scramble the point↔text correspondence in a
   pipeline whose LLM never sees the point. **This is the empirical version of the disqualification
   argument in `PAPER_C_DESIGN.md`**: variant (a) is a baseline row, and the correspondence control
   only becomes constructible in variant (b), where the vector conditions generation and the
   scramble happens in adapter training.
3. **Even real-vs-random barely separates, and the gap belongs to retrieval.** Random points
   re-encode at +0.371 mean (and +0.486 *median* — the mean gap is two tail packets, ids 13/20)
   because their anchors are their true nearest neighbours and the spectrum-matched draw lives in
   the real points' own 9-dim subspace. The rendered question tracks its anchor-reference tightly
   (ahead of it in 14/30); the render step neither adds nor destroys much.
4. **The one live signal is the renderer's own difficulty score.** Both renderability-1 packets are
   RANDOM points, and random averages 2.60 vs 3.20 — the blind renderer felt the incoherence the
   blind judge could not see in the finished text. A Baum–Lang answerability probe ("can this even
   be asked?") is worth a row in the real C3; grading the finished text is not.
5. **Blend targets re-encoded better than emissions** (means 0.640 vs 0.415; every emission packet
   self-scored renderability 2) — but kind is confounded with state in this draw (alternation), so
   this is a watch-item, not a finding.

## Caveats that travel with every number

n=10 per arm · the bridge routes text through CONCEPTS only, capping item-heavy points (visible in
the anchor rows) · each real point's anchor set appears twice (real + its scrambled twin), and the
judge noticed near-duplicate questions — a structural leak of pair-*existence* (not labels) in this
control layout; a future run should draw scrambled partners from a disjoint point set · everything
here is variant (a): NO faithfulness claim is licensed, per the sheet.

## What this buys the chapter

The C3/C4 harness now exists end to end (blinding protocol, re-encode bridge with asserted
alignment, anchor-reference bound), and the baseline row's structural weakness is measured rather
than asserted. The next build is variant (b) — the trained adapter — whose entire justification
this PoC just demonstrated empirically.
