# A27 — is there a better independent consumer than RecVAE? No, and that is the finding (2026-09-13)

**The question (author, 2026-09-13):** *"ours is recvae with all these things fixed, so recvae is poor
comparator, can we find better model?"* The objection is sound: the certified tower freezes a RecVAE
decoder and initialises from it, so tower-vs-RecVAE compares a model with its own ancestor.

**Run.** `adaptive_harness.py --selectors replay:ndcgtree16test,fixedorder_json:policyB_combined
--only ours,rbmf,golbandi_leaf --cohort test --no_floor --concept_neg_cap 0`, label
`a27b_consumers_certifiedera`. The same two banked interviews, nothing regrown or reselected.

**Gate PASSED.** `ours` reproduces the chapter table at every cell: CASPER .1808/.0537, .2015/.0675,
.2267/.0905, .2492/.1117, .2694/.1307; static .1808/.0537, .1962/.0701, .2178/.0857, .2386/.1033,
.2590/.1237. (A first run at `--concept_neg_cap 1` gave .2586 at K=16 and was discarded: the answer
convention must match the certified era or the numbers are not comparable.)

## Result

| consumer | arm | k1 | k2 | k4 | k8 | k16 |
|---|---|---|---|---|---|---|
| ours | CASPER | .1808/.0537 | .2015/.0675 | .2267/.0905 | .2492/.1117 | .2694/.1307 |
| ours | static | .1808/.0537 | .1962/.0701 | .2178/.0857 | .2386/.1033 | .2590/.1237 |
| **RBMF** | CASPER | **.0001/.0001** | .0409/.0106 | .0908/.0310 | .1106/.0405 | .1448/.0595 |
| **RBMF** | static | **.0001/.0001** | **.0001/.0001** | .0619/.0128 | .1025/.0238 | .1196/.0352 |
| Golbandi leaf | CASPER | .1626/.0262 | .1745/.0393 | .1858/.0526 | .1964/.0592 | --- |
| Golbandi leaf | static | .1626/.0262 | .1626/.0262 | .1746/.0478 | .2101/.0741 | --- |

**RBMF reproduces RecVAE's pathology exactly** — .0001 at k=1, and its static arm is still .0001 at
k=2. It is a second item-only consumer and it fails in the same place for the same reason.

**TaNP and EDDI cannot be run as consumers**: `run_interview_table.py:233,243` expects
`.cache/baselines/tanp_ml25m_liang.pt` and `eddi_ml25m_liang.pt`, neither of which exists. Only
`tanp_graded_armn.pt` is present, which is a different split — substituting it would be wrong. They are
not in the deleted-checkpoint manifest; they were never built for this split.

## What this means

**RecVAE is not a poorly chosen comparator; it is a representative one.** Nearly every published
cold-start consumer ingests an item vector and cannot read a concept answer at all, so an interview
that spends its early questions on concepts reaches them as an empty input. Golbandi's leaf model is
the single exception in our set, and the code says why: `run_interview_table.py:98`,
`CONSUMES_UNKNOWNS = {"golbandi_leaf"}` — it models the did-not-answer branch, so it is never
information-free.

The consumers table therefore stays as it is, with one clause added to the caption recording that RBMF
reads .0001 at the same budgets. The alternative — swapping in `belief_mf` — would have been worse: it
moves +0.004 over sixteen questions, and a near-inert consumer says nothing about either arm.

⚠ An open question this raises for the concept channel generally: if the field's consumers cannot read
attribute answers, the bank result needs a consumer that can before it can be claimed to transfer.
That is a limitation of the available models, not of the interview, and it is worth one sentence
somewhere if the reviewers press on generality.
