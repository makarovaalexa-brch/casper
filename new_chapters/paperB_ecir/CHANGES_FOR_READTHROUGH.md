# Paper B — what changed since you last read it yourself

For your read-through. Ordered by **how much of the paper it moved**, not chronologically.
Source `paperB_ecir.tex`, built `paperB_ecir.pdf`. **12 content pages exactly**, references from
page 13, zero overfull boxes.

Your baseline is the draft you read before sending the numbered supervisor comments. Since then:
your comments 3–24, your line-level corrections, the Goodreads work, two blind review panels, and
the errors those turned up.

**Where to look hardest:** §5 (RQ2) and §6 (RQ3) changed most. §3 (contract) and §4 (RQ1) are close
to what you last read. Table 1 and Table 2 are structurally different objects now.

---

## 1. The frame — unchanged since you approved it

Still the two-lever method paper you signed off: *what the system can ask* and *how it chooses*.
Title, RQ structure and the four contributions are as agreed. The adaptivity-null material is §6's
explanation, not the headline.

## 2. New since you last read: the second dataset

**Everything was re-measured end to end on Goodreads** — 18,430 books, 2,408 shelves, recommender,
concept encoder, both question orders and all six trees refitted there, nothing carried over.

| full/tail NDCG@10 at 16 questions | ML-25M | Goodreads |
|---|---|---|
| ask nothing | .1586/.0282 | .0940/.0058 |
| greedy static order, items+concepts | .2590/.1237 | .1713/.0323 |
| **RQ1, bank** | +.0143\*/+.0220\* | **+.0061\*/+.0111\*** |
| **RQ2, criterion** | +.0104\*/+.0070\* | **+.0110\*/+.0106\*** |

Appears as **Table 4** plus one paragraph at the end of §6. The "one dataset" limitation is deleted.

**Two cells move, and the paper says so:** Hu and Yu's ranking surrogate, indistinguishable from
Golbandi's rating surrogate on ML-25M, is 0.01286\* *worse* than it on Goodreads; and rating variance
sits marginally above the static order on the Goodreads tail.

## 3. New since you last read: the criterion ablation on both datasets

You pushed on this — the first Goodreads pass only carried our own greedy and our own tree, which is
the least contested claim. The ablation was then regrown on Goodreads too: five published criteria
and two shuffle controls, 37.8 minutes.

**Table 1 is now the cross-dataset ablation.** Its old absolute `full@10` column (recoverable as
static + gain) became a **Goodreads** column. Same six rows, same six columns — no extra space.

## 4. Your comments 3–24 — all addressed

Full detail in `CHANGES_FOR_SUPERVISOR.md`. Summary of what actually moved:

- **3** — three lit sweeps; two material hits now cited and discussed (Lin WWW'23, Gharahighehi 2025)
- **4** — new title
- **7** — wording only: "five of the audited methods reported a ranking metric, so holding them to
  NDCG@10 holds them to their own headline"
- **8, 20** — the comparator audit is finished (fifteen methods) and is a paragraph in §6
- **15** — *Ports and tuning* states that free constants were tuned in our setting, on validation,
  over a grid containing the published value
- **17** — the fitting-population apology is gone entirely
- **19** — the figure is gone; the paper is four tables and no figures
- **22** — tables re-cut and placed between paragraphs, captions shortened
- **24** — Limitations is now four lines

## 5. Your line-level corrections — all in

- concept questions are now *"we define as answerable when…"*, not asserted as fact, and the **full**
  gate is stated (two member items **and** enough exposure), not half of it
- the 3.98× reciprocity is flagged as restating the tie rather than adding to it
- "certified consumer" is gone everywhere
- the conjugate accumulator is a table row, not a disclaimer about a missing row
- non-academic register swept ("bought nothing", "upgrading", "off-trajectory")

---

## 6. ⚠ Errors found and corrected — read these carefully

These are the ones that would have been quoted back at us.

**a. Table 3's two bandit rows were stale.** Two reviewers said the p.10 GLM-UCB contrast didn't
reconcile. Recomputing from the banked per-user vectors showed **the prose was right and the table
was wrong** — GLM-UCB's true mean is .21662, the table printed .2161. Every banked row was then
checked; only these two had drifted.

| | was | now |
|---|---|---|
| GLM-UCB K=16 | .2161/.0833 | **.2166/.0839** |
| Thompson K=1 | .1627/.0376 | **.1613/.0378** |
| Thompson K=16 | .2135/.0878 | **.2126/.0878** |
| tuning deltas | +.0073/+.0023 | **+.0068/+.0032** |

**b. "138 cells… items-only in 136" corresponded to nothing.** Our own
`BATTERY_TABLE_POLICYB_ROWS.md` records **118 cells** over the twelve adaptive rows: combined
118/118, concepts-only 118/118, items-only 116/118 leading and **114/118** leading-and-separating,
four exceptions all tail. Now corrected. *This one was mine* — a reviewer derived 138 = 14×10−2 and I
changed the label to fit the number instead of checking the number.

**c. "Thompson beats the static order at every budget (Table 2)" was false as written.** It
equivocated on "the static order": true against **entropy₀-static** (+0.0288 at k=8), false against
the **greedy combined order** Table 2 shows (Thompson .1974 vs static .2101). On that consumer the
greedy order beats CASPER *and* Thompson, so **RQ3 survives** — our own loose sentence had been
manufacturing an objection against us.

**d. An orphan sentence fragment in §4** — `[+0.02430,+0.03037] at 1.79 against 1.75 answers.` with
no subject, left by an earlier edit. Restored to the answer-matched channel contrast it came from.

**e. "The best adaptive selector"** was GLM-UCB, but IGCN and CLiME both score higher. Now names
GLM-UCB.

**f. The consumers paragraph said "three further consumers / three of the four"** while the table had
five rows.

**g. "The gain is not uniform"** sat in §4 but is entirely about CASPER's leaves — moved to §5.

**h. "Five published split criteria"** in four places, where Table 1 carries **three** published
criteria and **two shuffle controls** — and contribution 3 already said three.

**i. Smaller:** Golbandi's tree cited to 2011 and the seed set to 2010; BALD's constant acquisition
attributed to *our* linear-Gaussian port rather than to Houlsby et al.; the genome filter behind
1,031 tags stated; "three pre-specified cells" are two (+0.01035 is both); a doubled "the".

## 7. Claims narrowed — check you're happy with each

- **the abstract now says "on ML-25M"** for the battery; the two levers and the ablation are on both
- **Table 3's caption discloses the candidate-bank asymmetry**: ported methods select over the whole
  catalogue, ours over 1,000 — stated as a *restriction*, since the bank is a subset of what they may
  ask
- the tree's construction is stated plainly: **conditions five times, then a per-leaf continuation**
- "no ground truth is read at selection time" → "no **test user's** targets"
- **arm counts removed throughout** — "twenty-three arms" could not be reconciled with any artifact,
  so counts are now what Table 3 shows: eleven ported selectors and three literature heuristics
- Related Work no longer claims fMF, Liu et al. and Nguyen et al. were "re-implemented and run here";
  there is no selector module for any of them
- the port gate no longer claims ports had to beat popularity, which seven of them do not

## 8. Structural

- **Table 2 dropped RecVAE and RBMF.** `.0000/.0000` is a degenerate ranking, not a measurement — an
  item-vector model handed a concept-only prefix has no input — and two of the "four of five"
  consumers rested on it. Three consumers now, no degenerate cells; the interface limitation survives
  as one sentence.
- **Both "Answer to RQn" paragraphs cut** — a fourth restatement of the abstract, the contributions
  and the conclusion, and there was no "Answer to RQ1" to match them. The conclusion carries both.
- **Code footnote:** "Anonymised repository: link to be added" replaced by release-on-publication.
  ECIR's full-paper track has no code requirement (separate Reproducibility track).

## 9. Deliberately NOT done — your calls

- the **answerability threshold sweep** (`n_c ≥ 2, e_c ≥ 1.5` never varied) — two reviewers made this
  score-deciding, and you declined it
- split-seed robustness; growing the surrogate trees on the training users; a figure; an in-paper
  description of the instrument
- the surrogate-selected static orders (`variance-static` etc.) — and note the paper now answers that
  objection from Table 3, where the three literature heuristics **are** surrogate-selected static
  orders and each loses to the metric-selected one by .0144–.0290

---

## What I verified, so you don't have to

- every **Table 1 and Table 3 cell** against its artifact at all five budgets
- 44 references, all cited and all defined; no dangling cross-references
- zero overfull boxes; no repeated words; no `??`; no leaked LaTeX
- no placeholders, no AI mentions, no identity leaks
- exactly 12 content pages

## What I could not verify, and you should judge

- whether the **ports are faithful** to their papers — five of them carry a native check, the
  heuristics and bandits do not, and the paper says so
- whether the **comparator is genuinely novel**. Two reviewers accepted it; the third was unsure,
  noting Golbandi 2010 built greedy static seed sets on the reported metric of its day, which the
  paper concedes in §2
