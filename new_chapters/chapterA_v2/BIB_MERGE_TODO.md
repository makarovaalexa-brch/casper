# BIB_MERGE_TODO — chapterA_v2_new_refs.bib → references.bib

`new_chapters/references.bib` is author-owned; this repo does not edit it directly. When the author is
ready, merge the following 14 entries from `chapterA_v2_new_refs.bib` into `../references.bib` (or keep the
two-file `\bibliography{../references,chapterA_v2_new_refs}` setup permanently — either is fine, but the
duplication should not persist silently).

## Entries to merge (14)
1. `steck2019ease` — Steck, EASE^R, WWW 2019
2. `ren2020recvae` — Shenbin et al., RecVAE, WSDM 2020
3. `steck2020edlae` — Steck, EDLAE, NeurIPS 2020
4. `shen2021gfcf` — Shen et al., GF-CF, CIKM 2021
5. `choi2023bspm` — Choi et al., BSPM, SIGIR 2023
6. `park2024turbocf` — Park et al., Turbo-CF, SIGIR 2024
7. `kang2018sasrec` — Kang & McAuley, SASRec, ICDM 2018
8. `sun2019bert4rec` — Sun et al., BERT4Rec, CIKM 2019
9. `petrov2022replicability` — Petrov & Macdonald, BERT4Rec replicability, RecSys 2022
10. `lin2021tanp` — Lin et al., TaNP, WWW 2021
11. `biyik2023soft` — Bıyık et al., soft-attribute PE, arXiv 2023
12. `toroghi2023bcie` — Toroghi & Sanner, BCIE, SIGIR 2023
13. `wang2025bdecf` — Cheraghi et al., BDECF, arXiv 2025 (author list corrected 2026-07-22 —
    see below; key name `wang2025bdecf` is now a misnomer since the first author is Cheraghi, not Wang,
    but is left unchanged here since it is cited by that key throughout `chapterA_v2.tex` — rename only if
    doing a coordinated key-rename across the .tex at merge time)
14. `deng2021unicorn` — Deng et al., UNICORN, SIGIR 2021 (added 2026-07-22; NOT currently cited by
    `chapterA_v2.tex` — added per author request as a graph-RL CRS landscape entry / external_literature
    record `deng2021unicorn.md`; add a `\cite{deng2021unicorn}` in the related-work table if/when the prose
    is updated to reference it)

## Concept-channel additions (2026-07-25) — 9 more entries to merge
Added to `chapterA_v2_new_refs.bib` for the new \S "The concept channel: derivation, lineage, and failure
analysis". All are real papers recorded in `external_literature/INDEX.md` (2026-07-25 block) +
`findings/concept_folding_and_implicit_attribute_inference.md`; none were edited into the author-owned
`references.bib`. Key-collision re-checked against `references.bib`: **zero collisions**.
15. `hu2008collaborative` — Hu/Koren/Volinsky, Implicit-feedback CF (WMF), ICDM 2008
16. `liang2016expomf` — Liang et al., ExpoMF, WWW 2016
17. `qin2020attributepropensity` — Qin et al., attribute-based propensity, KDD 2020
18. `steck2018calibrated` — Steck, Calibrated Recommendations, RecSys 2018
19. `antognini2021mmvae` — Antognini & Faltings, M&Ms-VAE fast multi-step critiquing, RecSys 2021
20. `antognini2022posneg` — Antognini & Faltings, positive+negative critiquing (M&Ms-VAE++), arXiv 2022
21. `luo2021bkvae` — Luo et al., BK-VAE keyphrase-activation-vector critiquing, SIGIR 2021
22. `luo2019deepcritiquing` — Luo et al., CE-VAE deep language-based critiquing, RecSys 2019
23. `perez2018film` — Perez et al., FiLM, AAAI 2018

**Bib-detail confidence flag:** entry *keys/authors/year/venue* are taken verbatim from INDEX.md and are
reliable. *Page numbers and DOIs* were reconstructed from the findings doc + standard proceedings records and
should be spot-checked against the canonical DOI before the final camera-ready (they are not load-bearing for
compilation). arXiv-only entries (`antognini2022posneg`) carry no page range by design.

## Key-collision check (2026-07-22)
Checked every key in `chapterA_v2_new_refs.bib` against `../references.bib` — **zero collisions**, all 14
keys above are net-new to `references.bib`.

- **PEBOL / Austin et al. 2024:** previously reconciled — `references.bib` already has `austin2024bayesian`
  (Austin et al., PEBOL) and `chapterA_v2_new_refs.bib` deliberately does NOT define a duplicate
  `austin2024pebol`/`austin2024bayesian` entry (see comment at the top of that file). Verified no duplicate
  exists in either file as of this pass.
- All other keys already present in `references.bib` that `chapterA_v2.tex` cites (`li2021seamlessly`,
  `dacrema2019progress`, `liang2018variational`, `austin2024bayesian`, `ma2019eddi`, `zhao2013interactive`,
  `li2023eliciting`, `he2023large`, `cremonesi2010performance`, `golbandi2011adaptive`, `liu2011wisdom`,
  `zhou2011functional`) do not appear in `chapterA_v2_new_refs.bib` — no collision risk there either.

## Cite-resolution check (2026-07-22)
Every `\cite{...}` key used in `chapterA_v2.tex` (25 unique keys) resolves against the union of
`../references.bib` and `chapterA_v2_new_refs.bib`. No missing keys found.
