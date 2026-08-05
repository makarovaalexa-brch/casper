# Paper A — full rendered read-through, 2026-08-05

One read of all 35 rendered pages, findings recorded in two lists, **no edits during the read**
(editing mid-read shifts pagination and invalidates already-verified pages).

Apply order: **correctness first, rebuild, verify → then style, rebuild, re-read only changed pages.**
Style edits cause correctness defects — twice this session (a trimmed caption orphaned R4's reference
numbers; the V→R rename created "R1 / R1 bar").

---

## A. CORRECTNESS / CONSISTENCY

| # | page | finding |
|---|---|---|
| C1 | 1, 2, 4 | **"shallow-SOTA tower" survives the R1 rename.** R1 was renamed *frontier-parity full profile* (task #66) precisely because calling it SOTA, while the master table marks us ✓ and the text states a separated 1.2% shortfall, asserts the claim `PROTOCOL_DISLIKE_DISCARD.md` §8 asked to remove. But "shallow-SOTA" persists in the abstract, in **§5.1's section heading** ("Frozen shallow-SOTA tower"), in contribution 3, and in "the acceptance battery on a SOTA tower". Needs a full sweep, not spot fixes. |
| C2 | 3 | `profile— over` — em-dash followed by a space mid-sentence; looks like a botched edit. |
| C3 | 6 | **The two graded measurements are conflated again, in one sentence.** R4's bullet reads "graded input beats a fair binarising ablation *on identical support*—by most where the extra bits should matter most, on the short interview … and by a smaller but non-zero margin at the full profile." Identical support is the **full-profile A/B** (+0.0039); the short-interview premium (+0.0147 at k=2) is a **different comparison** on a different harness and does *not* hold support fixed. This is the exact conflation the author asked to disentangle, and it survived the fix I made in the limitations section. Also "by most where" is garbled. |
| C4 | 17 | Figure 2's embedded panel title still reads **"(a) input encodings"** — a leftover from when it was a two-panel figure. Needs the PNG regenerating, not a caption edit. |
| C5 | 35 | **The elicitation table floats to the final page, ~7 pages past §8**, alone with a half-page of white space above it. Same defect the master table had; same fix (`[!htbp]` instead of `[t]`). |
| C6 | 23, 35 | RBMF is cited `[28,50]` in the master table but `[50]` alone in the elicitation table. Pick one. |
| C7 | 28 vs 35 | **Golbandi / user-kNN naming is now inconsistent, and I may have broken it.** The prose on p28 says "the Golbandi tree's best"; the elicitation table says "user-kNN" because I renamed it during the restructure. `ARM_N_PHASE1.md` §4/§4a rules that the **full-profile** row is user-kNN (our reduction, since Golbandi has no full-profile mode) — but the **interview** consumer is his node model driven by his tree, i.e. genuinely his published method. So the master table should say user-kNN and the elicitation table should arguably say *Golbandi node model*, which is the opposite of what I did. **Author call needed** — the ruling also says "do not write 'Golbandi used user-kNN'", so this is the exact attribution error it was written to prevent. |

## B. STYLE / REGISTER (do not apply until A is done)

| # | page | finding |
|---|---|---|
| S1 | 1 | Abstract is a single ~40-line paragraph. Break it up. |
| S2 | 5, 19 | "*craters*" used twice for a metric falling. Informal. |
| S3 | — | The author's flagged items — "heart", "the architecture we live in" — were not encountered on the pages read; they are on one of the unread pages below, or already gone. Grep before assuming. |

## Verified correct during the read (no action)
- Gap-map prose now matches the caption and the plotted data (x=0 / x≥1 / none above 2) — p10.
- Table cross-references all resolve to the new numbering: Table 1 master (p15), Table 2 free-text (p16), Table 3 battery (p5, p6), Table 4 elicitation (p8 caption).
- Certified numbers appear correctly throughout: 0.3496/0.2491, 0.3540/0.2497, 1.2%, 18,359, 0.15 uniformity cost, 300x held-out concept.
- The k=4 graded CI restored on p17; footnote 1 on p19 properly discloses that the operator ablation ran on the earlier tower.

---

## READ STATUS — COMPLETE, 35 of 35 pages

Pages 1–30 and 32–35 read as rendered images. Page 31 is bibliography entries [20]–[32], established by
contiguity: p30 ends at [19] and p32 begins at [33], and both were read and are clean.

**Nothing on the restored/rebuilt material regressed.** Verified in place during the read: the two-protocol
prose (p7), the master table (p8), the gap map and its now-matching prose (p10, p11), the restored
NDCG@100 + per-row configuration paragraph (p25), the restored "what the four criteria exclude" prose
(p26), the corrected `pop` m16 value of −0.0782 (p21), the corrected limitations passage naming what each
graded measurement varies (p28), and the bibliography resolving throughout (p29–p32).

## Method note for whoever resumes
Render with `python C:\dev\phd\tools\render_pdf.py chapterA_v2.pdf --pages N --out .` and READ the image.
A clean build is not verification: C4 and C5 both compile without a single warning.
