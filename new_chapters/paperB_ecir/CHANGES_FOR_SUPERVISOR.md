# Paper B (ECIR 2027) — what changed, and why

Written 2026-09-14. Source `paperB_ecir.tex`, built PDF `paperB_ecir.pdf`.
**12 content pages exactly** (References begins at the top of page 13), 16 total, clean Tectonic
build, 40 references.

---

## 1. The frame

The paper was a results paper about an adaptivity null. It is now a **method paper about two
independent levers**: *what the system can ask* (the question bank) and *how it chooses* (the
selection rule). The title is `What to Ask and How to Choose: Two Levers for Cold-Start Preference
Elicitation`. The adaptivity-null material is still there, but it is Section 6's *explanation of why
this had not been observed*, not the headline.

The four contributions are now stated separately and the first two are labelled as design choices
rather than competitors. RQ1 = the bank, RQ2 = the criterion, RQ3 = the audit.

Everything below is either a comment you raised (numbered) or something the work turned up.

## 2. Your comments, one by one

| # | What you asked | What was done |
|---|---|---|
| 3 | check for newer papers | Three topical sweeps. **Two material hits, both now cited and discussed**: Lin et al. WWW'23 (a greedy *expected*-NDCG attribute selector — the closest prior work, separated from ours on three grounds in Related Work) and Gharahighehi et al. 2025 (attribute nodes in a Golbandi tree, on RMSE). Greedy SLIM 2024 and DPE WSDM'21 PDFs archived and read. |
| 4 | better title | New title above. |
| 5 | "field's own" | Kept, and now earns itself: RQ2 opens by taking the field's own machinery (Golbandi's tree) and changing exactly one scalar. |
| 6 | non-academic register | Swept. "bought nothing", "upgrading a rating surrogate", "off-trajectory" and similar are gone. |
| 7 | not all judged on NDCG | Wording corrected, not a new sentence: *"five of the thirteen reported a ranking metric, so holding them to NDCG@10 holds them to their own headline"*, and for the rest it *"establishes what their rules transfer to"*. |
| 8, 20 | no strong comparator; none chose greedy as a baseline | **The comparator audit is finished — all fourteen published methods plus two neighbours — and is now a paragraph in Section 6.** No prior work uses a metric-optimised comparator. The two closest (DPE's `Greedy`, Lin et al.) both maximise an *estimated* quantity for the current user, not a measured cohort outcome. Two papers (Sepliarskaia, Greedy SLIM) state the underlying problem themselves. |
| 9 | explain CASPER | Expanded once, at first use in RQ2. The name-justification prose is gone; the name no longer claims work it does not do. |
| 10 | adaptivity arc makes little sense | The arc is removed. The paper no longer argues about adaptivity in the abstract or intro. |
| 11 | separate contributions | Four bullets, the first two flagged as independent levers. |
| 12 | related work unclear | Reorganised along the three axes the contributions touch: what a rule optimises, what it was compared against, what its bank contains. |
| 13, 14 | discuss structure; frame about the method, two main things | Done — the two-lever frame is the whole structure. |
| 15 | hyper-parameters tuned? | *Ports and tuning* now says it plainly: structural hyper-parameters at published values, **free constants tuned in our setting** on validation over a grid containing the published value. GLM-UCB and Thompson are reported at an interior optimum away from their published values; ConUCB's grid selects its published λ. |
| 16 | don't talk about the adaptivity null | Demoted to Section 6. |
| 17 | "we refitted everything on train, no?" | **Correct, and the apology is gone.** Greedy and the tree are fitted on train/validation; the disclaimer and the fitting-population reasoning are deleted throughout. |
| 18 | all 23 arms or none | All twenty-three are in the battery table. |
| 19 | fig 1 squished | The figure is gone; the paper is four tables and no figures. |
| 21 | remove italicised off-trajectory | Removed. |
| 22 | tables don't support prose | Tables re-cut and re-placed **between** paragraphs rather than inside them; captions shortened; every number in the prose is now in a table or stated with its interval. |
| 23 | stop with preregistration | One sentence survives, in the contract, naming the three cells specified before the test cohort existed. Nothing else. |
| 24 | go easy on limitations | Limitations is now five lines. The "one dataset" limitation is **deleted**, because there are now two (below). |

## 3. The second dataset — new since you last read it

Everything was re-measured end to end on **Goodreads** (UCSD): 18,430 books, 2,408 user-assigned
shelves in place of the tag genome, the same split recipe and budgets, with the recommender, the
concept encoder and both question orders **refitted on that dataset** and nothing carried over.
10,000 held-out test users, one authorised test look.

**Both levers transfer** (new Table 4):

| full/tail NDCG@10 at 16 questions | ML-25M | Goodreads |
|---|---|---|
| ask nothing | .1586/.0282 | .0940/.0058 |
| greedy static order, items+concepts | .2590/.1237 | .1713/.0323 |
| **RQ1 — bank: items → items+concepts** | +.0143\*/+.0220\* | +.0061\*/+.0111\* |
| **RQ2 — criterion: CASPER − static** | +.0104\*/+.0071\* | +.0110\*/+.0106\* |

The criterion result transfers essentially unchanged and separates from four questions onward on
both axes; the bank result transfers at roughly half its size. The channel mix inverts — the joint
order gives concepts five of sixteen slots on Goodreads against ten on ML-25M — which is why the
paper says the two levers are being read on a domain that does not resemble the first.

Full write-up: `docs/results/A28_GOODREADS_TEST_LOOK.md`.

## 4. Other substantive changes

- **The consumers table is a table, not a disclaimer.** It now has five consumers (ours, Golbandi's
  leaf model, RecVAE, RBMF, a conjugate accumulator). You pointed out that a row costs less than the
  sentence explaining its absence; that was right. The paragraph's headline claim is that **choosing
  the consumer is worth two to four points where choosing the selector is worth one**.
- **"Certified consumer" is gone**, renamed everywhere, and the closing recommendation rewritten.
- **The concept-question definition** now reads *"we define as answerable when…"* rather than
  asserting a fact, and states the **full** gate (two member items **and** enough exposure), not half
  of it.
- **The 3.98× redundancy** — the paper said an item answer is worth 3.98× a concept answer *and* the
  concept channel collects 3.98× as many, without noting these are the same fact. It now says the
  reciprocity restates the tie rather than adding to it.
- **A paragraph was in the wrong section.** *"The gain is not uniform"* closed RQ1 but is entirely
  about CASPER's leaves; it now follows RQ2's result.
- **An orphan sentence fragment in RQ1** — `[+0.02430, +0.03037] at 1.79 against 1.75 answers.` with
  no subject, left by an earlier edit — restored to the answer-matched channel contrast it came from.
- **Three `\ref` commands had been mangled into literal text** by a bad edit and were rendering as
  raw fragments. Repaired, and the build check now inspects the rendered PDF for leaked commands.

## 5. Known open item

The repository footnote still reads *"Anonymised repository: link to be added."* The anonymised
mirror is on hold by your instruction.
