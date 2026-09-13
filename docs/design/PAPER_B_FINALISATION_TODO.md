# Paper B (ECIR) — what is left to finalise (opened 2026-09-13)

Author instruction 2026-09-13: *"add all these to todo. repo mirror hold for now. as soon as box frees,
go to consumers, do not do anything else. proceed with closing them, do what you can in parallel. when
everything but repo is done, ping me"*.

State at opening: builds clean, **11.38 content pages against ECIR's 12** (15 total incl. references),
no leaked references, every supervisor comment 4–24 addressed and committed.

---

## P0 — the moment the box frees, before anything else

- [ ] **T6 · A better independent consumer.** The consumers table currently compares our tower against
  RecVAE, which is our own ancestor (the tower freezes a RecVAE decoder and initialises from it), so
  the comparison can only go one way. `run_interview_table.py:90` already registers
  `rbmf`, `tanp`, `eddi` — independent, and TaNP and Partial-VAE are already trusted in the paper's
  RQ1 robustness check. Replay the same two banked interviews:
  `--only ours,tanp,eddi,rbmf,golbandi_leaf`. No re-selection, no retraining. ~30 min.
  Decide the table's contents from the numbers: `belief_mf` is NOT the answer (it moves +0.004 over
  sixteen questions — a near-inert consumer says nothing).
  **Blocked until `ndcg_tree.py` finishes (HARD RULE 11d).**

## P1 — content still missing, doable in parallel now

- [ ] **T1 · Finish the comparator audit.** `external_literature/findings/comparator_audit_2026-09.md`
  has 9 of 14. Still to extract: **Golbandi WSDM'11, Rashid 2002/2008, fMF SIGIR'11, ConUCB WWW'20,
  Sepliarskaia RecSys'18**. Then write the RQ3 paragraph. This is supervisor comments 8 and 20, and
  the last substantive content addition. Do not print a count until all fourteen are checked.
- [ ] **T2 · Finish the literature sweep.** Two passes done (attribute-tree, CRS/NDCG), each found
  something material — Gharahighehi 2025 and Lin WWW'23. **Uncovered: bandit / active-learning
  2024–26, LLM elicitation, UMAP / CIKM / ECIR venues.** Supervisor comment 3. Do not assume the
  remaining axes are empty; the base rate so far is one material hit per pass.

## P2 — agreed, not yet applied, ten minutes each

- [ ] **T4 · Five paragraph-opener flips** to finding-first, as the abstract now is:
  *One variable changed* · *Who consumes the answers…* · *Criterion regret at real states* ·
  *Enlarging the bank improves cold-start ranking* (three qualifying clauses before the number).
  (*How much is there to adapt to* is already cut.)
- [ ] **T5 · The `n_c >= 2` omission.** §Answers states the answerability rule as only
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
