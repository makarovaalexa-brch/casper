# PAPER A — state and remaining work, 2026-08-04

Chapter: `new_chapters/chapterA_v2/chapterA_v2.tex` → `chapterA_v2.pdf` (37 pages).
Build: `C:\dev\phd\tools\tectonic.exe chapterA_v2.tex`. Read it with
`python C:\dev\phd\tools\render_pdf.py chapterA_v2.pdf --pages 7-12 --out <scratchpad>` (PyMuPDF at
`C:\dev\phd\tools\pylibs`). **A clean build is not verification** — see §4.

---

## 1. Certified artefacts (frozen, in `models/i26c_ep31/`)

| | file | numbers |
|---|---|---|
| Tower | `t2i26c_ep31.pt` | TEST **0.3496 / 0.2491** |
| Concept channel | `concept_cv3_ep08.pt` | cold, `expinfo` bank: m1 .1327/.0412 → m16 **.1498/.0673**, floor .1277/.0267 |

Paired vs RecVAE (0.3540/0.2497), same 10k users: full **−0.0043** [−0.0059,−0.0028] **separated, 1.2%
relative**; tail **−0.0006** [−0.0020,+0.0008] **parity**. Never write "ties the bar".

**Concept channel is cv3, the SIMPLER model** (464k params, hidden 256, keep_meh False) — author ruling to
prefer simplicity, and under the fixed ranked bank it also *beats* cv5 (SEL .1411 vs .1381). The cv5 run
(hidden 1024, graded meh, 30 epochs, ~11 h) was a **null result**: every matched delta inside the
±.003–.004 noise band. Capacity was never binding.

**Tower checkpoint note:** ep23 scores 0.3505 on TEST, higher than ep31's 0.3496. ep31 is nonetheless
correct — it wins the pre-registered `argmax(val_full + sel)` rule on VALIDATION (0.4873 vs 0.4831, the
run's maximum). Switching to ep23 would be test-set selection. Do not switch.

## 2. The results the chapter carries

- **R1**: parity on tail, stated 1.2% shortfall on full. Not a tie.
- **Intensity premium, identical support**: graded 0.3496/0.2491 vs binarised 0.3457/0.2471 →
  **+0.0039 full** (paired CI [+0.0025,+0.0053]) and **+0.0020 tail** ([+0.0007,+0.0031]). Both separated.
  This holds the fold-in support FIXED and varies only the star level.
- **Adding dislikes** (a different comparison — varies *what is folded*) is worth ≈0 at full profile, and
  pays only when evidence is scarce (+0.023 at k=2). **These two are routinely confused; name what each
  varies, do not just call them "different protocols".**
- **The polar result** (sharpest in the chapter): rank the same bank by pure discrimination
  H(bands|answered) and the concept channel beats the item channel on BOTH axes at EVERY budget, all ten
  cells separated — peak m8 **+0.0189 full / +0.0310 tail**. Off that bank items win full decisively.
  Mechanism, measured: a divisive ITEM is one few have seen (4.8% answered, that arm collapses to .1352);
  a divisive CONCEPT stays 97.2% answerable.
- **Answerability, protocol-free**: a held-out user has rated **25.2%** of the hundred most popular films
  (13.1% of the top 500) and can answer **53.0%** of all 1,031 concepts. Structural: a concept needs two
  rated members out of hundreds; an item has one chance.
- **Strategy flip** (§8): five consumers, four give E0 > P > H on full; only the Golbandi tree inverts.
- **Baselines**: EDDI 0.2153/0.1077 and TaNP 0.2645/0.1635 both measured and in the tables.

## 3. Floats as they stand (9)

T1 verification criteria · T2 master table (sideways) · F1 gap map · T3 free-text phrases ·
F2 graded curve (single panel) · T4 baseline bank · T5 full-profile results · T6 verification measured ·
T7 strategy flip.

## 4. Hard-won rules

- **A clean build is not verification.** ~30 defects this session compiled with zero warnings, including a
  bibliography that silently emitted NOTHING (one duplicate entry aborted BibTeX → all 366 citations `[?]`),
  a broken sentence, an acute-accent apostrophe rendering as `ś`, and three requirements attributed to the
  wrong number. Only reading rendered pages finds these.
- **A claim survives a keyword grep when the prose paraphrases it.** "posterior shrinks", "the layer",
  "conjugacy", "tie this bar" all survived sweeps for "belief" and "ties".
- **Verify every scripted edit by grep AFTER it runs.** Four no-opped silently this session; one produced a
  commit message asserting a fix that had not happened.
- **Mechanical renames need manual repair.** Collapsing V→R created "R1 / R1 bar" and two sites naming the
  WRONG requirement.
- **Never mix information contracts.** Cost real time three times: a tail metric scored globally vs on a
  head-deleted list; `--bank expinfo` silently falling through to random draws; per-budget answer counts
  quoted as if protocol-free.
