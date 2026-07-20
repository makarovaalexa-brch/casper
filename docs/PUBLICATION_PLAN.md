# PUBLICATION PLAN — conferences, venues, paper status

> Conference/journal targets, deadlines, and the A–E paper roster. The papers themselves live in
> `new_chapters/`; the system they describe is in `VISION.md`; current work is in `STATE.md`.
> Update deadlines and per-paper status here as they change.

## Constraints
- Solo; ~1 year left; **no-travel** (favour venues that don't require attendance / support remote).
- Prefer resource / short tracks and journals where a strong artifact carries the paper.

## Venue targets
- **ECIR 2027** — deadline **~2 Oct 2026** — primary anchor for Papers **C** and **A**.
- **RecSys 2026** — short or resource track (Paper A as testbed/resource).
- **CIKM 2026** — resource track.
- **UMAP 2027** — Paper A/B fit.
- **Journals (parallel, no deadline pressure):** UMUAI, TORS.
- **Workshop fallbacks:** KaRS, IntRS.

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
