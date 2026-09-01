# Findings: HEWITT, TAFJORD, GEIRHOS & KIM — *Neologism Learning for Controllability and Self-Verbalization* (arXiv:2510.08506)

*Deep read 2026-08-21, full PDF (25 pages incl. appendix), for Paper C stage **C4**
(`docs/design/PAPER_C_DESIGN.md` §C4). Serves the import of **plug-in evaluation** as C4's faithfulness
metric and the citation of **machine-only synonyms**.*

> **HOUSEKEEPING — this file does NOT touch `INDEX.md` or the bib.** A later reconciliation pass adds the
> entry and the BibTeX. Nothing here is committed.

**Provenance markers:**
- **[V]** — I extracted the primary PDF myself and read the sentence/number/table cell.
- **[A]** — abstract / landing page only.
- **[C]** — my computation from [V] numbers (arithmetic check, clearly marked).

---

## 0. HEADLINE — what changes for C4

1. **The formula in `PAPER_C_DESIGN.md` §C4 has the WRONG DENOMINATOR.** We wrote
   `(x − base)/(latent − base)`. Theirs is [V, Tables 2 & 4 caption]:

   > "Concept score increase percent: `(x − base model score) / (training data score − base model score)`"

   The denominator is the **training-data** concept score — the synthetic data that *defined* the concept —
   **not** the latent (neologism). This is load-bearing: it means **the latent is scored on the same ruler
   as the text**, and can itself fall short (long-text neologism = **36%** [V]) or overshoot (**105%**,
   **103%** [V]). Our version, with the latent in the denominator, pins the latent at 100% by construction
   and cannot detect a *weak latent*. Fixing this is a design decision, not a typo (§5.1).
2. **The range we quote (−6%..86%) is the SYNONYM table only** (Table 4). The full reported spread across
   both verbalization forms is **−6% to 127%** [V]. The metric is **signed and unbounded**.
3. **They run essentially NO controls.** One reference condition — the un-prompted base model — and that is
   the metric's own zero point, not a control arm. **No scrambled control, no random-word control, no
   nearest-neighbour control, no prior-only arm.** Our three controls are an *addition* to their protocol,
   not an import from it. (Consistent with `llm_verbalisation_2026-08.md` §0.2: the scrambled control is
   absent everywhere.)
4. **Their strongest failure mode is LEAKAGE, and it lands squarely on one of our C3 arms** (§4.3).
5. **The machine-only-synonym claim rests on CROSS-MODEL TRANSFER**, not on the single-model number
   (§2). Our analogue requires a *second, independently-fitted* adapter — without it a reviewer reads our
   machine-only synonym as an adapter artefact (§5.4). **This is the biggest surprise for our design.**

---

## 1. Venue / status / provenance

- Header on every page: **"Preprint."** [V]. arXiv line: `arXiv:2510.08506v1 [cs.CL] 9 Oct 2025` [V].
- Authors: **John Hewitt, Oyvind Tafjord, Robert Geirhos, Been Kim**, all **Google DeepMind**
  (`{johnhew,oyvindt,geirhos,beenkim}@google.com`) [V].
- **Not stated as accepted anywhere.** Appendix A.1 says [V]:
  > "In accordance with the ICLR code of conduct, here is how we used LLMs in drafting this paper."

  ⇒ prepared as an **ICLR (2026) submission**; treat as **preprint, under review**. Cite as arXiv until a
  proceedings entry exists. (Same group/lineage as the ICML 2025 position paper Hewitt, Geirhos & Kim,
  *"We can't understand AI using our existing vocabulary"* [V, refs], which is the "neologism learning"
  origin they expand on.)
- Models: **Gemma-3-4B-IT** (the steered/verbalizing model), **Gemini-2.5-Pro** (LLM judge; also the
  in-context-learning comparison), **Gemini-2.5-Flash** (data generation; questionnaire summarizer;
  cross-model transfer test) [V].

---

## 2. THE MACHINE-ONLY SYNONYM — exact numbers and how it was found

**How it was found (§2, "An Aperitif"):** an *informal* experiment. They trained a neologism for
single-sentence answers (token denoted `~singlesentence`, embedding initialised to the neutral word
`" accurate"` [V, §4.1]), then asked the model, verbatim [V, A.2]:

> `List 10 synonyms for this word: ~singlesentence`

Gemma's reply asserts the word "(pronounced 'ti-ak-o-i') is Ancient Greek for 'not' or 'without'" and
lists **No, Lack, Absence, Negation, Denial, Un-, Without, Not, Zero, Void** [V, A.2]. They then plugged
`lack` back into the trained prompt slot: `<original instruction>. Give me a lack answer.`

**The numbers [V, §2]:**

| model | no additional prompt | asking for a "lack answer" | statistic |
|---|---|---|---|
| Gemma-3-4B-IT | **42.9** sentences | **15.8** sentences | mean |
| Gemini-2.5-Flash | **37** sentences | **4** sentences | median |

> "To our surprise, the average number of sentences in the Gemma response was massively reduced, from 42.9
> sentences (with no additional prompt) to 15.8 sentences (asking for a lack answer). Intrigued, we asked
> Gemini-2.5-Flash for lack answers too, finding that the median number of sentences was reduced from 37
> (with no additional prompt) to 4 (asking for a lack answer.)" [V]

Footnote 2 is the disconfirmation of the obvious tokenisation explanation [V]:
> "Lack shares no subwords with any synonyms that we're aware of (e.g., laconic)."

**Definition [V, §2]:**
> "the idea that a model can have self-verbalizations that are both causally relevant to the model, and
> unintuitive to humans, machine-only synonyms."

**★ The load-bearing structure of the claim:** it is the **transfer to a DIFFERENT model family**
(Gemma→Gemini) that licenses the word *"making 'lack' a synonym for brevity shared by machines but not
humans"* [V, §1]. On one model it would be a quirk of one embedding.

**Systematic version (Table 4, plug-in-evaluated synonyms, % gap closed) [V]:**

| Concept | syn 1 | % | syn 2 | % | syn 3 | % | syn 4 | % |
|---|---|---|---|---|---|---|---|---|
| long-text | detailed | **−6** | extensive | 11 | lengthy | 13 | prolific | 4 |
| short-text | exact | 36 | precise | 48 | accurate | 27 | identical | 57 |
| single-sentence | statement | **86** | declaration | 51 | record | 0 | text | 60 |
| use-like | solitary | 2 | lone | 1 | isolated | 1 | single | 5 |
| flattery-answer | unmatched | 17 | unrivaled | 25 | uncommon | 10 | unique | 17 |
| refusal-answer | ineffective | 23 | unproductive | 38 | inefficient | 15 | nonfunctional | 44 |
| wrong-answer | small | 14 | tiny | 24 | little | 14 | mini | 14 |

Their reading [V, §5.1]:
> "Synonym verbalizations are often machine-only synonyms; e.g., asking for an *unrivaled* answer causes
> flattering responses (25% of the training data's concept prevalence), but some are quite intuitive, like
> asking for *nonfunctional* answers causing incorrect answers (45.9% of the training data's concept
> prevalence.)"

⚠ **Two internal inconsistencies, flag before quoting:**
- The prose says 45.9% for `nonfunctional`; **Table 4 says 44** [V]. Raw score 4.7, base 1.3, training 9.1
  ⇒ (4.7−1.3)/(9.1−1.3) = **43.6%** [C] — the table is right, the prose figure is unexplained. **Quote 44%
  from the table, not 45.9% from the prose.**
- `nonfunctional` is listed under *refusal-answer* in Table 4/Table 7, but the prose calls it "incorrect
  answers" [V]. Their slip; do not repeat it.

---

## 3. THE PLUG-IN EVALUATION PROTOCOL — operationally exact

### 3.1 What is substituted for what

> "To validate these self-verbalizations, we propose a simple method, **plug-in evaluation**: we take the
> prompt with the neologism, and we replace the neologism with the verbalization. We measure whether the
> verbalization causes a similar impact of the concept." [V, §1]

> "we take a prompt and replace the neologism with the self-verbalization, and measure the extent to which
> it causes the same steering behavior." [V, §5]

The substitution is **positional and minimal**: the training template is
`<instruction> Give me a {neologism} answer.` [V, §2, Table 10 "default"], and the plug-in prompt is the
*same template with the text in the slot* — `Give me a {synonym} answer.` [V, §5.1]. **Everything else is
held fixed**: same frozen model, same 100 held-out LIMA test instructions, same scorer.

Two verbalization forms are plugged in:
- **Synonym (single word)** — Table 4. Elicited by a 4-part prompt [V, §5.1]: (red) meta-question *"Before
  you answer, give a list of 5 synonyms for {neologism}."*; (blue) a placeholder instruction, *"sometimes
  set to the empty string"*; (green) **the neologism prompt as used in training** — *"to make sure we
  trigger the neologism as it was used in training"*; (purple) a **forced response prefix**
  (`Ok, here's a list of 5 synonyms for {neologism}: "`) — *"to acquiesce to the request for synonyms,
  without biasing the model towards which synonyms to choose."*
- **Long / questionnaire verbalization (a paragraph-length instruction)** — Table 2 column 2. Twelve
  questions (Table 17) × forced response prefixes; **the transcript is then summarised by
  Gemini-2.5-Flash into a single instruction** which is what gets plugged in (full texts in Table 18) [V,
  §5.2]. Rationale [V]: *"not all questionnaire questions seem to trigger useful self-verbalizations from
  all neologisms, so we intend this evaluation of Gemini summaries to measure the total useful information
  provided across many self-verbalizations."*

### 3.2 "base" and "latent", operationally

- **base** = the **base model's default behaviour** — Gemma-3-4B-IT answering the same instructions with
  **no concept prompt at all** ("Base Data", Table 1) [V]. It is a *behaviour score of a text corpus*, not
  a vector.
- **"latent"** in our shorthand = the **neologism**: a single new **word embedding** `E_c ∈ R^d` appended
  to a frozen embedding matrix, trained by gradient descent (APO-up, a DPO variant, Eq. 2) on
  chosen/rejected response pairs, **with all other model parameters frozen** [V, §3]. It is scored by the
  *same* plug-in procedure (Table 2, "Neologism" column) — **the neologism is an ARM, not the denominator.**
- **the target / denominator** = the **training data** concept score — the synthetic responses generated by
  a strong LLM under an explicit natural-language concept prompt (Table 6) [V, §4.1]. "Training Data"
  column of Table 1.

**Table 1 — the two ends of the ruler [V]:**

| Concept | Metric | Base Data | Training Data | Δ |
|---|---|---|---|---|
| long-text | word count ↑ | 778.0 | 1511.7 | 733.7 |
| short-text | word count ↓ | 787.1 | 90.1 | −697.0 |
| single-sentence | sentence count ↓ | 42.9 | 1.2 | −41.7 |
| use-like | 'like' prevalence (%) ↑ | 0.3 | 9.0 | 8.7 |
| flattery-answer | LLM scoring (1–10) ↑ | 1.6 | 8.5 | 6.9 |
| refusal-answer | LLM scoring (1–10) ↑ | 1.3 | 9.1 | 7.8 |
| wrong-answer | LLM scoring (1–10) ↑ | 1.3 | 7.6 | 6.3 |

### 3.3 How the % is computed and aggregated

- **Per concept**, one scalar concept score per arm, computed as a **mean over the 100 held-out LIMA test
  questions** [V, §4.1: *"We evaluate on 100 different test questions also from LIMA"*; Table 1 caption:
  *"Count metrics indicate the mean count; prevalence indicates the mean fraction of words … and LLM
  scoring uses Gemini-2.5-Pro to rate responses on a scale from 1–10 … (mean score reported)"*].
- **Then normalised**: `(x − base) / (training − base)`, ×100 [V, Table 2 & 4 caption]. The signed
  denominator handles ↑/↓ concepts automatically; that is why `short-text` at word-count-**down** yields a
  positive %.
- **Aggregated across concepts by an UNWEIGHTED MEAN** — the "Average" row of Table 2 [V]. For negated
  templates in the robustness table they use `100 − score` [V, Table 14 caption].
- **No confidence intervals, no error bars, no seed replicates, no significance test anywhere in the
  paper** [V, checked all tables]. n = 100 questions per cell.
- Raw (un-normalised) scores are banked in Table 23 (neologism) and Table 24 (verbalizations) [V], which is
  what makes the arithmetic auditable. Spot-checks [C]: `detailed` 733.1 ⇒ (733.1−778.0)/733.7 = **−6.1%**
  ✓; `statement` 7.0 ⇒ (7.0−42.9)/(1.2−42.9) = **86.1%** ✓. **The formula in the caption reproduces the
  tables exactly.**

### 3.4 The headline table (Table 2) [V]

| Concept | Neologism | Long verbalization | 1st Synonym | Best Synonym |
|---|---|---|---|---|
| long-text | 36% | 39% | −1% | 24% |
| short-text | 105% | 110% | 36% | 58% |
| single-sentence | 98% | 98% | 86% | 86% |
| use-like | 103% | 32% | 2% | 5% |
| flattery-answer | 103% | 100% | 17% | 33% |
| refusal-answer | 95% | 76% | 23% | 44% |
| wrong-answer | 103% | **127%** | 13% | 24% |
| **Average** | **92%** | **83%** | **25%** | **39%** |

Their summary [V, §5.2]: *"we find that synthesized verbalizations often work as well as the trained
tokens."* And the honest hedge in §1: *"self-verbalizations are often (but not always) validated by
plug-in evaluation."*

⚠ **A prose/caption contradiction on the denominator.** §5.1 says the synonym results are *"in units of
percentage points of the concept difference between the base model **and the neologism model** that the
synonym achieves"* [V] — i.e. exactly *our* current formula. But the table caption says *training data*,
and the arithmetic [C] confirms **training data**. **The captions and the raw tables are authoritative;
§5.1's sentence is wrong.** Do not cite §5.1's phrasing as licence for our denominator.

---

## 4. Controls, comparisons, and failure modes

### 4.1 Controls they run — a short list, and mostly not controls

| Condition | Where | What it is |
|---|---|---|
| **Base model, no concept prompt** | Table 1, everywhere | the metric's **zero point**, present in every cell by construction |
| **Training-data ceiling** | Table 1 | the **100% point** — a real natural-language concept prompt (Table 6) |
| **In-context (few-shot) neologism definition** | A.8, Table 16 | *alternative method*, not a null: Gemma **46%**, Gemini-2.5-Flash **96%**, Gemini-2.5-Pro **86%** avg |
| **Few-shot prompting** | §6, Table 5 | *baseline method* for the compositionality experiment |
| **Likelihood loss vs APO-up** | A.5, Table 9 | ablation: 92% vs **70%** overall |
| **Multi-template ± hinge loss** | A.6, Tables 13/14 | robustness ablation: single 77%, multiple 86%, multiple+hinge **89%** avg |
| **AxBench "w/t concept" column** | Table 3 | base-behaviour scores (0.0–0.4 of 2.00) for complex concepts |

**★ NOT PRESENT, checked exhaustively: no scrambled/permuted control, no random-untrained-embedding
control, no random-English-word control, no nearest-neighbour control, no prior-only arm, no human
baseline, no significance testing.** The nearest thing to a prior-only manoeuvre is buried in the
questionnaire pipeline [V, §5.2]: *"We also ask an almost-identical set of questions of Gemma **without**
the neologism … The output of these two questionnaire transcripts we feed to a powerful model"* — i.e. the
model's default self-description is supplied to the summariser as a *contrast*, but it is **never evaluated
as an arm**. So the obvious question "what % does the model's default self-description close?" is
**unanswered in the paper**.

⇒ **Our three controls (scrambled correspondence, prior-only, nearest bank question) are a contribution on
top of their protocol.** Wording that survives: *"we import Hewitt et al.'s plug-in evaluation and add the
controls it does not run."* Do **not** write "following Hewitt et al. we compare against a scrambled
control".

### 4.2 Failure modes they explicitly discuss

- **Verbalizations may be junk.** [V, §5]: *"We should not assume self-verbalizations are useful—they may
  be unrelated or hallucinated."* This is the stated *motivation* for plug-in evaluation.
- **★ LEAKAGE / demonstration-instead-of-description.** [V, §5.2, on `use-like`, the worst long-form cell
  at 32% vs the neologism's 103%]:
  > "we hypothesize that Gemini's synthesis may pick up on the actual behavior of Gemma using *like* more
  > frequently in the questionnaire responses, as opposed to a meta-description."

  The summariser sees the concept **exhibited** in the transcript and can copy the surface form rather than
  describe the latent. See §5.3 — this is the most transferable warning in the paper.
- **Scaffold leakage into the instruction.** The `refusal-answer` synthesised instruction literally begins
  *"When the user's input contains the 'neologism' placeholder…"* [V, Table 18] — the verbalizer is
  describing the *harness*, not the concept, and that text still scored 76%.
- **Overshoot.** Percentages exceed 100 routinely; `wrong-answer` long verbalization hits **127%** with a
  raw LLM score of **9.3** against a training-data score of 7.6 [V, Table 24]. The metric does not
  penalise over-steering.
- **Single-word verbalizations are capacity-limited** [V, §5.2]: *"Single-word verbalizations are naturally
  limited in how much they can capture of what a neologism means to a language model."*
- **Not every probe question fires** [V, §5.2]: *"not all questionnaire questions seem to trigger useful
  self-verbalizations from all neologisms."*
- **Large embedding norms** — a *training* pathology, addressed with a hinge loss `λ_h·max(‖e‖−1, 0)`,
  `λ_h = 0.1` [V, A.6]. Relevant to us only as precedent that an unconstrained trained latent drifts
  off-distribution in norm — cf. our off-catalogue emission (C1/C2).
- **The metric inherits the judge.** Four of seven concepts are scored by a **Gemini-2.5-Pro 1–10 LLM
  judge** [V, Table 1]; AxBench uses three LLM-judge scores combined by **harmonic mean** so *"any 0 among
  the three scores will lead to an overall 0"* [V, §4.2].

### 4.3 ★ The leakage mode collides with a C3 arm

`PAPER_C_DESIGN.md` §C3 lists *"retrieval-augmented prompting from the nearest bank phrases"* as the
training-free alternative renderer. **That arm is the `use-like` failure by construction**: text retrieved
from the bank is text whose effect on the CF space we already know, so gap closure would measure the
retrieval, not the verbalisation. It is also **nearly the same object as our "nearest bank question"
C4 control**. As written, an arm and a control coincide. Resolve one of two ways: (a) keep RAG-prompting as
a *renderer* but score it only against the nearest-bank-question control it is a superset of, and state
that its gap closure is not evidence of latent faithfulness; or (b) drop it to a footnote and keep the
adapter renderer as the only faithfulness claim. Either is fine; leaving both unlabelled is not.

---

## 5. What breaks or complicates OUR instantiation

### 5.1 ★ The denominator. Their normaliser does not exist for us.

Their ruler has **three** independent objects: base behaviour, a **concept target defined by real
natural-language data**, and the latent. Our C4 has only two: no-question belief (base) and the emitted
latent `ẑ`. If we keep our `(x − base)/(latent − base)`:
- the latent is pinned at exactly 100% and **cannot be evaluated**, whereas theirs scored 36%–105% and
  *that spread is half of what makes their table informative*;
- a *weak or degenerate* latent inflates every text score (small denominator), so the metric rewards a bad
  emitter — a real failure mode given C0/C1 might yield a `ẑ` that barely moves the belief;
- we lose comparability to their −6%..127% range, which is normalised to a *different* thing.

**Recommendation (a design change, needs the author's call):** restore the third object. Define
`target` = the belief shift produced by the **measured best real question at that state** (we already bank
per-state measured gains — `policy_labels.py`, per `PAPER_C_DESIGN.md` §C1). Then `ẑ`, the rendered text,
and all three controls are scored on **one external ruler**, none of them is the denominator, and the
sentence "the text recovers X% of what a real measured question recovers" is both citable-alongside-Hewitt
and stronger than a self-normalised ratio. If the author prefers the two-object version, then say
explicitly in the paper that our normaliser differs from theirs and **never quote their range as a
comparator**.

### 5.2 ★ Their substitution is WITHIN one frozen model. Ours crosses an adapter.

In Hewitt et al. the text and the latent enter **the same forward pass at the same position** — the only
difference between arms is what occupies the slot. There is **no representational mismatch anywhere in the
protocol**. Our path is `ẑ → LLM text → frozen SBERT → frozen SBERT→CF adapter → belief`. Two extra frozen
maps sit between the text and the measurement, and **each caps gap closure below 100% for reasons that have
nothing to do with verbalisation quality**. A low number is then unattributable.

**Mitigation that must be in the design:** measure the **adapter ceiling** first. Take a **real bank
question's own text**, push it through SBERT→adapter, and score its gap closure against that same
question's *measured* effect. That number is the achievable maximum for any text on our path, and it will
not be 100%. Report every C4 figure **as a fraction of it, or alongside it**. Without it the headline is
uninterpretable. (This is cheap, answer-free, and can run before C3 — arguably it belongs in C2.)

*Their protocol's independence property, for comparison:* the verbalizer (Gemma) and the judge
(Gemini-2.5-Pro / programmatic counters) are different systems, so their scorer is not the thing being
scored. Our adapter's independence from the emitter (fitted separately, per §C4) buys us the same property
— **that argument survives and is worth stating explicitly**, it is the one place we match them cleanly.

### 5.3 The leakage mode transfers, and our renderer training is where it bites

Their `use-like` lesson generalises to: **any pipeline in which the verbalizer is exposed to the target
behaviour can copy it instead of describing the latent.** For us the exposure routes are (i) the RAG arm
(§4.3), (ii) any prompt to the LLM that includes the state's *measured* best question or the user's
history in a form correlated with the answer, and (iii) an ELM-style adapter trained on pairs where the
text side was generated *from* the item the latent points at. Route (iii) is the dangerous one because it
is invisible at evaluation time. Precommit: **the C3 adapter's training pairs must not be drawn from the
states C4 evaluates.**

### 5.4 ★ "Machine-only synonym" needs a SECOND adapter, or it is an adapter artefact

Their claim is licensed by **transfer to a different model** (Gemma→Gemini). Our analogue — text that closes
a large gap through the SBERT→CF adapter while looking wrong to humans — has an obvious deflationary
reading: **the adapter has a quirk**. Since our adapter is a single fitted ridge map, that reading is hard
to refuse. To claim a machine-only synonym at all we need the text to reproduce the effect through a
**second, independently-fitted text→CF map** (different text encoder and/or different fit), exactly
paralleling their second model. If we do not build it, the design doc's §C4 sentence should be weakened
from "machine-only synonyms" to *"we cite Hewitt et al.'s finding as the reason not to grade rendered
questions by human plausibility"* — which is what §C4 actually needs it for, and that use **is fully
supported** by the paper as read. (The plausibility argument stands on their evidence alone: `unrivaled`
→ flattery at 25%, `lack` → brevity across two model families, `detailed` → the *most* human-plausible
long-text synonym scoring **−6%**, i.e. **human plausibility and causal effect are not merely
uncorrelated here, the most plausible word was the worst**. That last inversion is the single best
sentence to quote in our §C4.)

### 5.5 Their concept score is a scalar with a direction. Ours is a belief over 18k items.

Every concept in Table 1 reduces to one monotone scalar (word count, sentence count, prevalence, judge
score) with an explicit ↑/↓, which is what makes `(x−base)/(target−base)` well-defined and signed. "Effect
on a recommender belief" is not a scalar. We must pick the scalar **before** running C4, and the choice has
a circularity trap: **projecting the belief shift onto `ẑ`'s own direction makes the latent trivially
maximal and is self-referential** in a way their concept metrics (word counts, an external judge) are not.
Safer scalars, in preference order: (1) **ΔNDCG@10 against held-out targets** — external, already our
currency, and it is what `PAPER_C_DESIGN.md` forbids *simulating* but not *measuring* where real answers
exist; (2) cosine of the belief shift to the shift induced by the **measured best real question**;
(3) projection onto `ẑ` — usable only as a diagnostic, never as the headline.

### 5.6 Smaller frictions

- **Their n and dispersion practice is below our bar.** 100 test questions, one seed, unweighted mean over
  7 concepts, no CIs. We should report per-state distributions and paired CIs. Do not adopt their bare
  "Average" row as a template — cf. the project rule that a chapter CI must be paired.
- **The metric is unbounded and signed**; a control can score negative (`detailed` −6%, `1st Synonym`
  long-text −1%) [V]. Good — it means our scrambled control *can* come out negative rather than pinned at
  zero, which is a stronger null. Say so.
- **Overshoot is unpenalised.** If our rendered text moves the belief *further* than `ẑ` in the same
  direction, the metric reads >100% and calls it faithful. Decide in advance whether >100% is reported raw
  (their choice) or as `1 − |1 − ratio|`. Their choice is more honest and easier to defend; just state it.
- **Position matters.** They keep the *exact training template* and swap only the slot contents. Our
  analogue: the rendered question must enter the belief update through **exactly the same channel** a bank
  question does (per Paper B's L1 finding: as an **answered question** — history token + fold + mask). If
  the text enters by any other route, the comparison is not a substitution and the whole import fails.
  **This is the one implementation detail most likely to be got wrong quietly.**

---

## 6. Quotable lines (all [V])

- Protocol: *"we take the prompt with the neologism, and we replace the neologism with the verbalization.
  We measure whether the verbalization causes a similar impact of the concept."* (§1)
- Motivation for the metric: *"We should not assume self-verbalizations are useful—they may be unrelated or
  hallucinated."* (§5)
- Machine-only synonyms: *"self-verbalizations that look odd or unrelated to humans, but consistently cause
  the behavior of a given neologism."* (§1)
- Honest scope: *"self-verbalizations are often (but not always) validated by plug-in evaluation."* (§1)
- Formula (table caption, Tables 2 and 4): *"Concept score increase percent:
  (x − base model score) / (training data score − base model score)"*
- Leakage: *"we hypothesize that Gemini's synthesis may pick up on the actual behavior of Gemma using like
  more frequently in the questionnaire responses, as opposed to a meta-description."* (§5.2)

---

## 7. Open items for the reconciliation pass

1. `INDEX.md` entry + BibTeX (preprint, arXiv:2510.08506v1, 9 Oct 2025, Google DeepMind) — **not done here.**
2. `PAPER_C_DESIGN.md` §C4 needs three edits, all author decisions: the **denominator** (§5.1), the
   **range** (−6%..86% → state which table it comes from; full spread is −6%..127%), and the
   **machine-only-synonym claim strength** (§5.4).
3. `llm_verbalisation_2026-08.md` §0.3 describes plug-in evaluation from the abstract; it can now be
   upgraded to [V] and cross-linked here.
4. Decide the C4 scalar (§5.5) and schedule the **adapter-ceiling measurement** (§5.2) — it is answer-free
   and belongs before C3, i.e. it is runnable now.
