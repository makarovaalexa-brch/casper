# Paper B (ECIR) — what is left to finalise (opened 2026-09-13)

Author instruction 2026-09-13: *"add all these to todo. repo mirror hold for now. as soon as box frees,
go to consumers, do not do anything else. proceed with closing them, do what you can in parallel. when
everything but repo is done, ping me"*.

State at opening: builds clean, **11.38 content pages against ECIR's 12** (15 total incl. references),
no leaked references, every supervisor comment 4–24 addressed and committed.

---

## P0 — the moment the box frees, before anything else

- [x] **T6 (DONE 09-13) · A better independent consumer.** The consumers table currently compares our tower against
  RecVAE, which is our own ancestor (the tower freezes a RecVAE decoder and initialises from it), so
  the comparison can only go one way. `run_interview_table.py:90` already registers
  `rbmf`, `tanp`, `eddi` — independent, and TaNP and Partial-VAE are already trusted in the paper's
  RQ1 robustness check. Replay the same two banked interviews:
  `--only ours,tanp,eddi,rbmf,golbandi_leaf`. No re-selection, no retraining. ~30 min.
  Decide the table's contents from the numbers: `belief_mf` is NOT the answer (it moves +0.004 over
  sixteen questions — a near-inert consumer says nothing).
  **Blocked until `ndcg_tree.py` finishes (HARD RULE 11d).**

## P1 — content still missing, doable in parallel now

- [x] **T1 (DONE 09-13) · Finish the comparator audit.** `external_literature/findings/comparator_audit_2026-09.md`
  has 9 of 14. Still to extract: **Golbandi WSDM'11, Rashid 2002/2008, fMF SIGIR'11, ConUCB WWW'20,
  Sepliarskaia RecSys'18**. Then write the RQ3 paragraph. This is supervisor comments 8 and 20, and
  the last substantive content addition. Do not print a count until all fourteen are checked.
- [x] **T2 (DONE 09-13) · Finish the literature sweep.** Two passes done (attribute-tree, CRS/NDCG), each found
  something material — Gharahighehi 2025 and Lin WWW'23. **Uncovered: bandit / active-learning
  2024–26, LLM elicitation, UMAP / CIKM / ECIR venues.** Supervisor comment 3. Do not assume the
  remaining axes are empty; the base rate so far is one material hit per pass.

## P2 — agreed, not yet applied, ten minutes each

- [x] **T4 (DONE 09-13) · Five paragraph-opener flips** to finding-first, as the abstract now is:
  *One variable changed* · *Who consumes the answers…* · *Criterion regret at real states* ·
  *Enlarging the bank improves cold-start ranking* (three qualifying clauses before the number).
  (*How much is there to adapt to* is already cut.)
- [x] **T5 (DONE 09-13) · The `n_c >= 2` omission.** §Answers states the answerability rule as only
  `e_c < 1.5`; the gate in `signed_answers.py:144,175` is `n_c >= 2 AND e_c >= TAU_REF`. The paper
  states half a rule, and it is the omitted half that made the old concept-question wording overstate.

## HOLD — author instruction

- [ ] **T3 · Anonymised repository mirror.** Footnote currently reads *"Anonymised repository: link to
  be added."* **On hold.** Not to be actioned; ping the author when everything else is done.

## Opportunistic — the paper stands without it

- [ ] **T7 · Second dataset (Goodreads).** `ndcg_tree.py --total_L 16 --label ndcgtree16` running since
  00:52, now depth 7 against ML-25M's depth 5. Then the policy-B items/concepts/combined orders and
  the two test evaluations (~7 h). Retires the "one dataset" limitation.
  Port and results so far: `docs/results/A26_GOODREADS_ELICITATION.md`.

---

**Ping condition:** everything above except T3 done.


---

## Progress 2026-09-13

**Done:** T1 (comparator audit, all fourteen, now a paragraph in RQ3), T2 (third lit pass, nothing
competing on the remaining axes), T4 (opener flips), T5 (full answerability gate). Paper at
**11.75 content pages of 12**, clean build, no leaked references.

**Outstanding:** T6 (consumers) only, plus T3 on hold and T7 opportunistic.

⚠ **T6's blocker is overrunning.** `ndcg_tree.py` on Goodreads has been running since 00:52 — over
13 hours — against ML-25M's 7,061 s (about 2 h) for the same object. It is progressing, not hung, but
it is at depth 7 where ML-25M stopped at 5, and the min-leaf rule is the only stopping rule. The 73%
refusal rate keeps sending most of the cohort down `noanswer` branches that stay above 200 users, so
the tree keeps finding legal splits. No way to predict the end from here.


## Closed 2026-09-13 — everything except T3

**T6 done, with a negative outcome that is itself the answer.** RBMF reproduces RecVAE's pathology
exactly (.0001 at k=1); TaNP and EDDI have no liang-split checkpoints and were never built for it.
RecVAE is representative rather than badly chosen: nearly every published cold-start consumer takes an
item vector and cannot read a concept answer. Golbandi's leaf model is the sole exception
(`CONSUMES_UNKNOWNS`). Table unchanged, caption gains one clause.
Full: `docs/results/A27_INDEPENDENT_CONSUMERS.md`.

**Paper state:** 11.76 content pages of 12, 15 total, clean build, no leaked references, every
supervisor comment 4–24 addressed. **T1, T2, T4, T5, T6 all closed. Only T3 (repo mirror) remains,
on hold by instruction.**

**T7 (Goodreads) is a separate workstream, not a finalisation blocker.** The tree finished at 16:42
after 950.6 min: depth 11, 176 nodes, 116 leaves. On the VALIDATION cohort it beats the combined
static order by **+0.03168 full / +0.01402 tail at K=16**, three times ML-25M's +0.01035 — but
validation overstates (ML-25M's own validation figure was +0.02794 against +0.01035 on test), so this
means nothing until the policy-B orders and the two test evaluations are run (~7 h).

## Closed 2026-09-14 — T7 done; everything except T3

**T7 (Goodreads) landed on TEST and is in the paper.** Both levers transfer: CASPER +0.01100* full /
+0.01059* tail at sixteen questions against ML-25M's +0.01035*/+0.00705*, separating from four
questions onward; the bank result +0.00610*/+0.01105* against +0.01430*/+0.02199*. Full write-up
`docs/results/A28_GOODREADS_TEST_LOOK.md`. Added to the paper as a closing paragraph plus
Table 4 (a two-column dataset comparison, not a re-run of the battery), and the "one dataset"
limitation is removed.

**Paper state: exactly 12 content pages** (References begins at the top of page 13), 16 total, clean
build, no leaked references, no AI/TODO/placeholder artifacts.

Four defects found by the end-to-end read of the built PDF and fixed:
- an orphan sentence fragment in RQ1 (`[+0.02430,+0.03037] at 1.79 against 1.75 answers.`), restored
  to the answer-matched channel contrast it came from;
- the consumers paragraph said "three further consumers / three of the four" while Table 2 has five
  rows (RBMF was added to the table and the prose was never updated);
- `The gain is not uniform.` closed RQ1 but is entirely about CASPER's leaves — moved into RQ2;
- the RQ3 robustness note pointed "below" at a paragraph that is above it — now a table reference.

**Only T3 (anonymised repository mirror) remains, on hold by instruction.** The footnote still reads
"Anonymised repository: link to be added."
