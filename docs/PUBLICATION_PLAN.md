# PUBLICATION PLAN — conferences, venues, paper status

> Conference/journal targets, deadlines, and the A–E paper roster. The papers themselves live in
> `new_chapters/`; the system they describe is in `VISION.md`; current work is in `STATE.md`.
> Update deadlines and per-paper status here as they change.

## Constraints
- Solo; ~1 year left; **no-travel** (favour venues that don't require attendance / support remote).
- Prefer resource / short tracks and journals where a strong artifact carries the paper.

## Venue targets
- **★ TORS (ACM Trans. on Recommender Systems) — the primary target for Paper A.** See below.
- **ECIR 2027** — deadline **~2 Oct 2026** — anchor for Paper **C**; for A, only a cut-down version.
- **RecSys / ECIR reproducibility tracks** — strong fit for Paper A's certification bridge.
- **RecSys 2026** — short or resource track (Paper A as testbed/resource).
- **CIKM 2026** — resource track.
- **UMAP 2027** — ruled OUT for Paper A: no human study, and the answer model is behavioural.
- **Journals (parallel, no deadline pressure):** UMUAI, TORS.
- **Workshop fallbacks:** KaRS, IntRS.

## ★ TORS — why it is Paper A's target (settled 2026-07-29)
ACM Transactions on Recommender Systems, the field's dedicated journal (launched 2023). Scope
explicitly covers **evaluation methodology, reproducibility, and resource/benchmark contributions** —
exactly the category three independent reviews put Paper A in.

- **No deadline.** Rolling submission; we submit when the work is ready. Nothing forces the timeline.
- **No travel.** A journal satisfies the project's hard no-travel constraint outright, unlike every
  conference on the list. This alone makes it the best-matched venue we have.
- **Length is an asset.** The ~30pp that got Paper A rejected from ECIR ("too long; instrument
  without application") is unremarkable for TORS.
- **Speed is the cost.** Journal cycles run months. Against the Oct-2026 ECIR date a TORS submission
  likely will not *complete* before the thesis, but "under review at TORS" is normal and citable.
- **Not yet verified:** current editorial board and turnaround statistics. Check the journal site
  before committing.

### The blind calibration panel (2026-07-29) — why we trust the TORS read
Three drafts, one hostile reviewer, identical short rubric, all anonymised and framed as unpublished
submissions: **(A)** PEBOL, **(B)** Bıyık soft-attributes — the system our own gap analysis calls the
binding near-miss — and **(C)** our Paper A pre-restructure.

| | A (PEBOL) | B (soft attributes) | C (ours) |
|---|---|---|---|
| verdict | weak reject | weak reject | **weak reject** |
| SIGIR / RecSys main | REJECT | REJECT | REJECT |
| ECIR full | ACCEPT | ACCEPT | **REJECT** (length; instrument without application) |
| **TORS** | ACCEPT | ACCEPT | **ACCEPT** |
| reproducibility | REJECT | REJECT | **ACCEPT** (bridge called "exemplary") |

**Read:** identical verdicts to two peer-reviewed elicitation papers ⇒ we are AT the field's level, not
below it, and the earlier long-form "reject" was the reviewer running hot rather than a measurement.
TORS is the one venue every draft clears. Our unique win is the reproducibility track; our unique
loss is ECIR, for length and for deferring the payoff to a companion chapter.
**All three top objections were simulator circularity** — it is the field-wide attack, and our I2
firewall (answers = recorded ratings / raw watch counts, fold-in only) is the only real defence among
the three. We under-sell it.

## The papers (A–E → `new_chapters/`)
| # | folder | working title | maps to VISION pillar | draft state |
|---|---|---|---|---|
| **A** | `paper1_casper` | CASPER-U: unified-embedding instrument / testbed | 1 (recommender) | most developed (~1650 L, 23 figs) |
| **B** | `paper2_casper` | Set-encoder elicitation by reconstruction | 1–2 (discrete) | ~988 L, 8 figs |
| **C** | `paper3_casper` | Continuous-action preference elicitation | 3 (continuous) | ~1332 L, 5 figs |
| **D** | `paper4_casper` | "Just Ask": open-vocabulary free-recall | 4 (open questions) | ~889 L, 3 figs |
| **E** | `paper5_casper` | Latent policy → deployable LLM-rendered dialogue | 3/6 (verbalise) | skeleton (~317 L) |

## Priorities
1. **A + C** for ECIR 2027 (the anchor pair).
2. Bibliography completeness before any submission — ~8 load-bearing works still missing from
   `new_chapters/references.bib` (see `external_literature/README.md`).
3. Clear every relevant item in `docs/HARSH_REVIEW_POINTS.md` for a paper before it ships — especially
   the **human round-trip study**, which currently blocks the strongest claims across A–E.

## Reviews & response history
Full harsh/adversarial reviews and responses: `new_chapters/_meta/` · consolidated mistake-checklist:
`docs/HARSH_REVIEW_POINTS.md`.
