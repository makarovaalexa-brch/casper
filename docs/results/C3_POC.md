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

1. **[Amended 2026-08-24] Registered prediction 2 NOT CONTRADICTED — but the run was unpowered to
   confirm it.** Judge 11/30 = 0.367, Wilson 95% CI [0.219, 0.545]: the data cannot exclude ~54%
   accuracy, so "fluency discriminates nothing" is consistent with this run, not established by it
   (HARD RULE 6). What remains suggestive: real packets were judged "real" *least* often (0.20 vs
   0.60 scrambled / 0.50 random — all three intervals overlap), i.e. no evidence the judge's
   textual heuristics track validity, and some that they anti-track. A powered version needs ~120
   packets (±0.09).
2. **The scrambled control CANNOT separate under variant (a) — structurally, and now measurably.**
   Real and scrambled re-encode identically (+0.528 vs +0.514; anchor rows identical *by
   construction*). Code review before unblinding identified why: the packets scramble
   point↔context, but the anchors travel with the point, so the retrieval chain point→anchors→text
   is intact for scrambled packets. There is no way to scramble the point↔text correspondence in a
   pipeline whose LLM never sees the point. **This is the empirical version of the disqualification
   argument in `PAPER_C_DESIGN.md`**: variant (a) is a baseline row, and the correspondence control
   only becomes constructible in variant (b), where the vector conditions generation and the
   scramble happens in adapter training.
3. **[Amended 2026-08-24, hostile-review pass] The PoC had ZERO functioning controls, not one.**
   The scrambled arm cannot separate (point 2), and the "random" arm is drawn from the real
   points' own rank-9 span (n=10 covariance), so it is a random direction INSIDE the reals'
   subspace — disclosed below, but the original conclusion still read the comparison as
   informative. It is not. What follows stands only as description, not as a control reading: Random points
   re-encode at +0.371 mean (and +0.486 *median* — the mean gap is two tail packets, ids 13/20)
   because their anchors are their true nearest neighbours and the spectrum-matched draw lives in
   the real points' own 9-dim subspace. The rendered question tracks its anchor-reference tightly
   (ahead of it in 14/30); the render step neither adds nor destroys much.
4. **[Demoted 2026-08-24 to a WATCH-ITEM, same standard as conclusion 5.]** The renderer's
   difficulty score *may* carry signal — both renderability-1 packets are random, and random
   averages 2.60 vs 3.20 — but that is an n=2 observation (P≈0.11 under exchangeability) on a
   self-rated ordinal from the agent that wrote the text, and it is the comparison most exposed to
   the (now-known-broken) random arm. The Baum–Lang answerability probe stays in next-steps on its
   own merits, not as a finding of this run.
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

## REPAIRED READING (2026-08-24) — Fable, from `experiments/paperC/c3_poc/join.json` + `byte_identity_check.json` (commit 019c060)

1. **The headline table now has a committed producer** (`c3_poc_join.py` → `join.json`) and
   intervals. The paired real−scrambled re-encode difference is **median +0.0087, real greater at
   5/10, sign test p = 0.727** — the honest form of "the scrambled arm cannot separate". On the
   median, random (+0.486) is not below real (+0.472); the mean gap was carried by the two
   artifact packets below. Bolding of the mean column is retracted.
2. **The random arm's geometry was a code bug, now fixed at source.** The spectrum-matched draw was
   rescaled in raw space but measured mean-removed: the control ran at 1.84× (up to 9.71×) the
   length of the point it stood in for *in the space every cosine is read in*, while matching the
   raw norm exactly — which is why nothing flagged it. Packets ids 13/20 (the "tail packets" of the
   original conclusion 3) are artifacts of this. `packets_v2.json` carries the corrected draw
   (rescale in the mean-removed space) but has NOT been rendered or judged; every number in this
   doc describes the v1 pipeline. Any rerun starts from v2 with scrambled partners from a disjoint
   point set.
3. **Provenance:** the run-rev vs land-rev gap found by the harness review is closed — the
   committed extract/re-encode scripts reproduce the banked packets and re-encode rows exactly
   (payload-identity, `byte_identity_check.json`); renderer/judge provenance is banked in
   `provenance.json`; `KEY.json` moved to `_quarantine/`.
4. **Net standing of the PoC after all amendments:** a pipeline demonstration with zero functioning
   controls and an unpowered judge — its two durable products are the harness itself and the
   structural finding that variant (a)'s correspondence control cannot be constructed. Both feed
   C3B; no comparison row from this run may be quoted as evidence.
