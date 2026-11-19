# TORS draft — review panel on v1, and what v2 did about it (2026-09-08)

Four independent reads of `paperA_tors` v1 (`paperA_tors_v1.tex/.pdf`, 33 pp, review mode): three simulated
TORS reviewers of increasing leniency and one supervisor/co-author fidelity audit against `chapterA_v2.tex`.
Full texts in `reviews/`. The reviewers read the PDF through text extraction, which misaligned the rows of
Table 1 in their copies; the rendered table is correct (checked on the page image), so the "row-shift"
finding raised by R1 and R2 is an artefact of their reading and is not a defect of the paper.

## Verdicts on v1

| reviewer | recommendation | relevance | originality | soundness | presentation | reproducibility | impact | confidence |
|---|---|---|---|---|---|---|---|---|
| R1 strict methodology | **Major revision** | — | — | — | — | — | — | — |
| R2 mainstream RecSys | **Major revision** | 5 | 3 | 3 | 2 | 2 | 3 | 4 |
| R3 practitioner | **Major revision** | 5 | 3 | 3 | 2 | 2 | 3 | 4 |
| Supervisor (fidelity) | **not as it stands, but close**: 0 wrong numbers, 3 sentences contradicted by their own tables | | | | | | | |

Consensus: relevance is not in doubt and the RQ1 experiment, the certification bridge and the provenance
rule are the paper's strengths; what costs it is (a) the instrument's own training is undescribed, (b) a
handful of sentences that the paper's own tables contradict, (c) undefined internal vocabulary, (d) the
abstract reporting only the favourable channel ranking, and (e) length.

## What every reviewer raised, and the v2 response

1. **Table 3's "Ours − this" tail column did not match its caption** (supervisor, R3): two tail deltas use
   the rival's full-profile-best arm, not its tail-best. **Fixed**: caption and header now say "each system at
   its own best full-profile strategy, scored on both metrics"; numbers unchanged. A first row now gives the
   graded-answer counts 1.59 / 3.10 / 3.05, which are the mechanism.
2. **"NDCG@100 orders the LIANG column identically" is false on the printed numbers** (supervisor, R3):
   RBMF, EDDI and the conjugate belief move. **Fixed**: "orders the frontier band and the first six rows
   identically and reorders only the middle of the table". *Chapter corrected to match (2026-09-08).*
3. **Two k=2 graded premiums under one description** (+0.0144 §2.2 vs +0.0102 everywhere else; all four
   reads). **Fixed**: the paper uses +0.0102 throughout (the banked curve `premium_vs_k_t2i26c_ep31.json`
   supports +0.0102 at k=2 and +0.0022 at full). *Chapter corrected to +0.0102 (2026-09-08).*
4. **+0.0022 vs +0.0039 at full profile** (R1, R3): different comparisons (graded-vs-likes-only history vs
   identical-support re-valuation). **Fixed**: each is now labelled where it appears.
5. **Two cold intercepts printed with no explanation** (0.1279/0.0192 vs 0.1277/0.0267; all four reads).
   **Fixed**: §7.6 states both, names their protocols, and says they are never compared across; §8.3 opens
   with its own intercept.
6. **The instrument's architecture and training are absent** (R1, R2, R3: "a reproducibility failure").
   **Addressed**: new §5.1 paragraph "Architecture and training of the set encoder", written from the
   training code (`src/instrument/i26_encoder.py`, `train_i26.py`, `interview_curriculum.py`, run log
   `experiments/instrument/t2i26c_run.out`, selection record `t2i26c_selection.json`): deep-sets encoder over
   the frozen d=200 latent, level-indexed scale/shift over ten half-star levels, width-512 token and residual
   networks, frozen-RecVAE anchor gated by the answered count, exposure branch for asked-but-unrated items,
   learned prior latent, zero-initialised last layers, 1.07 M trainable parameters, curriculum over
   full/dropout/short-interview/empty evidence sets, multinomial NLL on held-out targets, two learning-rate
   groups, one run, checkpoint selected on validation. **These facts are not in the chapter; the author must
   verify them against `docs/design/RETRAIN_DESIGN_SHEET.md` and the certified run's arguments before
   submission.** The "frozen vs trained on interviews" contradiction (R3) is resolved by the same paragraph:
   the substrate is frozen, the set encoder over it is trained, then the whole tower is frozen.
7. **Undefined internal vocabulary** ("triple gate", "retrained model", "clip-up", "counterfeit control",
   "certified pair", "canary", "no popularity floor", "sign/identity contrast", "T2′"; all reads). **Fixed**:
   each is defined in the paper's own words or replaced; the three concept-channel criteria are stated in
   §6.4 and referred to from §8.3; "no popularity floor" and the four bank rankings are defined in §7.6.
8. **Abstract reports only the ranking under which concepts win** (R1, R2, R3). **Fixed**: abstract,
   contribution 5, RQ4 answer and conclusion now state that under the answerability-weighted rankings the
   item channel wins full accuracy; "lifts substantially" removed.
9. **"Every deciding gap separated" overstated** (R1, R3, supervisor): rows marked ≈/flat. **Fixed**: "every gap
   that decides one of the three distinct orderings"; RBMF's 0.0001 "best" un-bolded.
10. **Five of six consumers are reconstructions; the abstract says "under the node model of Golbandi et al."**
    (R1, R2). **Fixed**: abstract says "under our reconstruction of"; §8.1 names which consumers are run as
    designed (ours, RecVAE) and marks the other four ‡ in the table.
11. **The instrument is both a row and the reference, and was trained on interviews** (R1, R2, R3). The
    chapter's own qualification ("not a controlled attribution") is kept; v2 adds it to the "strongest cells"
    paragraph and rewrites the §9 limitation (the inversion rests on the node model vs the likes-only RecVAE,
    neither trained on interviews). The reviewers asked for two controls (a set encoder trained on full
    profiles only; a frontier consumer with a fallback for unanswered questions); the author ruled them
    unnecessary (2026-09-08) and the paper does not list them as future work.
12. **HELF/entropy0 attributed to Rashid 2002** (R1; also a standing item in `HARSH_REVIEW_POINTS.md`).
    **Fixed**: the ordering is cited to Rashid, Karypis & Riedl 2008 (`rashid2008learning`, already in the shared
    bib); Rashid 2002 is kept for popularity-as-answerability. HELF, entropy0 and popularity are defined at
    first use. *Chapter corrected to cite 2008 for the ordering (2026-09-08).*
13. **Bıyık et al. 2023 author list was wrong** (R1: "Talati, Sadigh"). **Fixed** against the arXiv record
    (Bıyık, Yao, Chow, Haig, Hsu, Ghavamzadeh, Boutilier) in `refs.bib` **and in the shared
    `chapterA_v2_new_refs.bib`** (the chapter's bibliography inherits the fix at its next build).
14. **Working notes leaked into the reference list** (R1, R3): Qin 2020 and Golbandi 2010. **Fixed**: both
    notes stripped from `refs.bib` (they remain in `references.bib`, where they are harmless).
15. **Tail metric on 9,281 users unexplained** (R1, R2, R3). **Fixed**: §7.1 states that tail NDCG is defined
    for the 9,281 test users with at least one target outside the head (the harness's tail row-compaction).
16. **Related work thin on CRS and simulated users** (R2). **Addressed** with a new paragraph citing
    Christakopoulou 2016, CRM, EAR, SCPR, the Jannach survey, Kostric 2024 (TORS), Radlinski 2019 and Golbandi
    2010, all already in the shared bibliography; no reference was invented. Zhang & Balog 2020 and the
    LLM-consumer line are not in the bib and were not added.
17. **RQ4 carried ~20 numbers in prose with no table** (R2, R3). **Fixed**: Table 5 (concept − item under
    POLAR at m=1/8/16 with CIs, plus the POP row).
18. **Figure 3 was a default matplotlib plot with a stray title** (R3). **Fixed**: redrawn from the banked
    result with two panels (arms; paired premium with 95 % intervals) by `plot_graded_curve_tors.py`.
19. **Figure 1 omitted measured rows** (R1). **Fixed**: Mult-VAE, Turbo-CF, SASRec, RBMF, TaNP and EDDI added at
    their Table 1 values.
20. **R2's open half evidenced only by AUC 0.94 on the direction's own members** (R1). The like/dislike/other
    concept AUCs (0.626 / 0.416 / 0.498) moved into the R2 row of Table 4 and the sign/identity contrasts are
    defined there.
21. **Length** (R2, R3: cut a third). §3.2 (ratings decomposition, ~1 page) moved to Appendix A with a
    six-sentence summary in the body; TaNP digression halved; duplicate prior-art paragraph removed; the
    "1.8 unpaired standard errors" sentence removed. Net body length unchanged (≈32 manuscript pages) because
    the reviewers' additions (architecture, related work, Table 5, three limitations, two discussion
    paragraphs) cost what the cuts saved. TORS's own guideline is 20–35 acmsmall pages; v2 previews at 37 total
    / 33 body in acmsmall.
22. **Practitioner pay-off** (R2, R3, R1): new Discussion paragraph "Certifying another consumer" and a
    Tag-Genome-dependence limitation.
23. **Single training run** (R1, R2, R3): stated as a limitation. Not resolvable without new runs.
24. **R1 tolerance "fixed in advance"** (R1, R2): unverifiable in the paper; wording kept as the chapter has it.
    *Author decision: point to the design sheet or drop "fixed in advance".*

## Open items only the author can decide

- Verify the new architecture/training paragraph (item 6) against the design sheet and the certified run.
- Authors and affiliations (single-blind; placeholders in `\author`/`\affiliation`), the code URL, the
  acknowledgements, and the generative-AI disclosure ACM requires.
- Whether to run seed replicates (item 23) before submission.
- Whether to give τ and the four-band edges of the concept answer numerically (R3): not in the chapter.
