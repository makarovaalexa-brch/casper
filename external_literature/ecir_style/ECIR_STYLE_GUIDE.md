# ECIR full-paper house style — a prescriptive guide

Derived from 11 ECIR 2025/2026 LNCS papers (see `INDEX.md`; corpus table at the end). Every rule below is something the corpus actually does; quoted examples are verbatim. Written for a 12-page (body, excluding references) LNCS submission such as CASPER Paper B.

## 0. LNCS mechanics observed

- **Single column**, US-letter/A4 LNCS `llncs` class, **10 pt** body (font `SFRM1000` = CM Roman 10 pt in 9/11 papers; two used Times/Nimbus 10 pt via `\usepackage{times}`), 9 pt for abstract/captions/footnotes/references, 7–8 pt inside dense tables.
- Running heads: page number + short title (odd) / "N Author et al." (even). Preprints keep them.
- **Captions**: tables are captioned **above** ("Table 2. …" with bold label and a period in classic LNCS; some use "Table 2:" via `caption` package), figures **below** ("Fig. 1. …"). Captions are full sentences and often long (3–6 lines) — they carry the bold/underline/marker legend.
- **Tables**: mixed. Copenhagen and Poliba papers use `booktabs` (toprule/midrule/bottomrule, no vertical rules); Ramos, Li and Sun use full `\hline` grids or vertical rules. Booktabs looks cleanest; either is accepted. Numbers are 4 decimals for metrics in [0,1], 1 decimal for relative % changes, thousands separated ("1,859" or "1 000 209").
- **Bold = best, underline = second best** is near-universal; one paper (Li) used colour shading, one (Dervishaj 2026) used blue/orange cell shading plus bold/underline.
- **Code URL** goes in one of three places: last sentence of the abstract ("Our code is available at https://…" — 4 papers), a page-1 footnote attached to the title or first sentence (2 papers), or the last sentence of the experimental-setup section (2 papers). One anonymised preprint wrote "[REDACTED FOR BLIND REVIEW]" in a page-1 footnote — ECIR full papers are double-blind, so submit with an anonymised link/footnote and replace it at camera-ready.
- Footnotes are used for dataset/library URLs (huggingface, RecBole, Elliot, Lenskit), implementation minutiae ("l is set to 10 in practice"), and "due to space" pointers to an online appendix. 3–15 per paper.
- Keywords line after the abstract (3–5, dot-separated). "Acknowledgments" and (2026) "Disclosure of Interests" paragraphs sit before References.
- References: LNCS `splncs04` numeric style, cited as `[3, 7]`, **32–59 references** (median ≈ 40), taking 2–4 pages.

## 1. Section skeleton and page budget (12 body pages)

| Section | Typical heading in corpus | Pages | Notes |
|---|---|---|---|
| Title/abstract/keywords | — | 0.6 | Abstract 150–250 words |
| 1 Introduction | "Introduction" (Dervishaj 2026 merges "Introduction and Related Work") | 1.5–2.0 | ends with RQs and/or contribution list |
| 2 Related Work / Background | "Related Work", "Background", "Individual user fairness" (topic-named) | 1.0–1.5 | 3–5 bold-headed paragraphs; Rampisela puts RW **after** results (§5) when the background section is itself technical |
| 3 Method / Preliminaries | named after the artefact ("CARE-CRS", "Pairwise User unFairness (PUF)", "User-Centric Data Pre-processing") | 2.0–3.0 | problem formulation + overview figure + 1–3 numbered equations |
| 4 Experimental Setup | "Experimental Setup", "Experimental Evaluation" | 1.5–2.5 | datasets table, models/baselines, metrics, protocol, implementation |
| 5 Results | "Results", "Results and Discussion", one subsection per RQ | 3.5–5.0 | the largest section; tables + 2–5 figures |
| 6 (optional) Limitations / Discussion | "Limitations", "Limitations and Broader Impact", "Further Discussion and Limitations" | 0.3–0.8 | 4/11 have a standalone section |
| 7 Conclusion | "Conclusion(s)", "Limitations and Conclusions", "Discussion and conclusions" | 0.4–0.7 | 2–3 paragraphs |

Observed: Introduction ends on p.2–3; Setup starts p.4–7; Results start p.7–9; references start p.13–14 of the preprint. Three ECIR 2026 papers use **bold run-in paragraph headings** ("**Datasets.**", "**Baselines.**", "**Effectiveness of post-training denoising (RQ1).**") instead of `\subsubsection` — this saves roughly half a page and is the dominant compact style.

## 2. Abstract recipe (150–250 words, one paragraph)

1. **One sentence of context** stating the accepted practice or premise: "Implicit feedback – the main data source for training Recommender Systems (RSs) – is inherently noisy…"; "Individual user fairness is commonly understood as treating similar users similarly."
2. **The gap**, stated as a flat negative: "no existing individual user fairness measure simultaneously accounts for both"; "this assumption lacks comprehensive empirical verification."
3. **What we do**: "In this work, we focus on post-training denoising."; "we present Pairwise User unFairness (PUF), a novel evaluation measure…"; empirical papers phrase this as the RQ ("Are multimodal embeddings truly beneficial for recommendation?").
4. **Scale of evidence with numbers**: "Experiments with a state-of-the-art CF recommender and 4 open and closed source LLMs in 3 datasets"; "across 4 datasets and 7 rankers"; "14 widely used state-of-the-art multimodal recommendation models".
5. **Headline result with a number**: "improvements up to 13% in effectiveness"; "by an average of 54% and 25% for ReDial and INSPIRED"; or a qualitative reversal: "larger LMs do not necessarily translate to better performance".
6. Optional: one-sentence implication, then **code URL**.

Numbers appear in 8/11 abstracts. No citations in the abstract. Acronyms are defined on first use even in the abstract.

## 3. Introduction recipe (5–8 paragraphs, 1.5–2 pages)

- P1: the field and why the task matters (1 paragraph, 3–6 citations).
- P2–P3: what prior work does and the specific inconsistency/gap. The corpus states gaps bluntly and cites the papers being contradicted: "But the findings are not consistent: some work reports that PLMs sometimes perform worse than SLMs in news RSs [7, 8]."; "These two limitations have been acknowledged in prior studies (e.g., [22,17,14]), but remain unaddressed in subsequent CRS research."
- P4: what we do, in one paragraph, method named. Contrast against the alternative: "Different from in-training denoising, post-training denoising does not involve changing the architecture…"
- P5: **enumerated RQs** (7/11 papers), typeset either as `RQ1 …` labels or a numbered list, one line each, phrased as yes/no or "to what extent" questions: "RQ1 Does using a larger LM in news RSs consistently lead to better recommendation accuracy?"
- P6: a **contributions list** (8/11) with 2–4 bullets ("–" itemize) or "(i)…(iv)" — see template. Reproducibility/empirical papers often replace it with a one-paragraph preview of findings with numbers: "Our results show that the reproducibility of the reported effectiveness of these models is dataset dependent, with differences up to 43% across datasets."
- Short final sentence pointing at the results, e.g. "We make code, runs and analysis available at …" (or leave the URL in the abstract).

**Contributions list template (fits the corpus):**
```
Our contributions are threefold:
– We present the first [systematic / large-scale] [study / measure / method] of X, [evaluating N models on M datasets].
– We show that [main finding with a number, e.g. "our 8B recommender achieves 15× higher accuracy than 32B baselines"].
– We [release code / provide practical guidance] [for Y].
```
Rules: every bullet starts with "We"; the first bullet claims novelty with "first"/"novel"/"the only" only when it is literally defensible (corpus authors do: "the first LLM-based post-training denoising approach for CF"); the middle bullet contains a number.

## 4. Presenting the method

- Open with a **one-paragraph overview** naming the parts and pointing to the figure: "The CARE-CRS framework (Figure 4) consists of two parts: 1) … and 2) …"
- **Overview figure** on the first or second method page in 8/11 papers (pipeline diagram, or a data-flow schematic like Ramos Fig. 2 "masks assistant turns … to train UserSim"). Captions restate the mechanism, not just the name.
- **Notation is light**: define sets in prose ("Let U denote the set of users and I the set of items"), then 1–4 numbered display equations (Dervishaj 2026 has one; Rampisela one; Zhao two; Li five). Long derivations are absent. Prompts are shown verbatim in a boxed/monospace block (3 papers).
- A **"Problem Definition/Task Formulation"** run-in paragraph precedes the method in 4 papers.
- Explain design choices with an explicit reason and cite it: "We choose logistic regression for its simplicity and interpretability…", "we use the base model versions rather than the instruct version to avoid the issue of pre-existing system alignment [13]".
- Pseudo-code appears only when the method is procedural (Slokom: two `algorithm2e` boxes).

## 5. Experimental-setup checklist (what every paper states, in this order)

1. **RQs restated** at the top of the results/experiments section if not in the intro ("We address the following research questions (RQs) in this section").
2. **Datasets table** (10/11): columns users / items / interactions / sparsity (+ domain-specific: #categories, Gini, min interactions). Name the source with a citation and a footnote URL. State preprocessing: k-core filtering, rating binarisation threshold, split ratio and whether temporal or random, and that the split is **global not per-user** if relevant ("We split datasets globally (not user-wise) to avoid data leakage [29]").
3. **Recommenders/baselines** as a comma-separated list with a citation each, grouped by family ("a) Benchmarks … b) Learning-based … c) LLM-based"). Say how each was obtained: "we use the official implementation released by the authors", "implemented via the MMRec toolbox", "we replicate UniCRS using their source code".
4. **Hyper-parameter tuning**: state the tuner and budget ("hyperopt … 50 runs of Bayesian search", "20 trials … TPE … with Recall@20 as the validation metric", "grid search"), the selection metric, early stopping, epochs, and that ranges include those of the original authors.
5. **Metrics** with cut-offs: NDCG/Recall/HR/MRR @10/20 (or @5/10), plus beyond-accuracy where relevant (coverage, Gini, calibration). Cite the metric's origin ([Järvelin & Kekäläinen 2002] for NDCG).
6. **Repetitions**: "averaged over three runs", "five different randomization seeds for both model initialization and dataset splitting", "3 denoising runs – to account for stochastic decoding".
7. **Significance test** named once here: paired t-test (5 papers, sometimes "two-tails paired t-test"), Kendall's tau for measure agreement, repeated-measures correlation for nested data.
8. **Hardware/cost** when it matters: "Denoising all 10 000 users on Yelp takes 2 hours with an NVIDIA H100 GPU"; "in excess of 12,400 GPU-hours".
9. **Toolkit** named and footnoted: RecBole, Elliot, Lenskit, recpack, Pyserini, CRSLab.
10. **Code link** if not already given.

## 6. Results-table conventions

- One **main table** with rows = methods, column groups = datasets × metrics (`\multicolumn` headers with `\cmidrule`), 4-decimal values; **bold best, underline second-best**, significance marker as superscript on the value.
- Row groups separated by `\midrule`/`\hline` (baselines / ours; or per-dataset blocks with a rotated dataset label in the first column — `\rotatebox{90}` used by 4 papers).
- **Legend in the caption**, not in the text: "Best scores are in bold, and second-best are underlined. … Statistical significance is given by the paired t-test."; markers defined in a table footnote line: "‡ p < 0.001, † p < 0.01, ∗ p < 0.05".
- Relative improvements are given as a final row ("Improvement 42.7/15.4%") or a "Change" column ("+ 5.07%"), always with the reference row stated.
- Dense tables use `\resizebox` or `\footnotesize` (7–8 pt); readability is sacrificed before rows are cut.
- **Figures**: box plots of effectiveness across variants (Sun), grouped bars per dataset (Li, Zhao), line plots vs. a swept parameter with the baseline as a dashed horizontal line (Zhao Fig. 4/5, Rampisela Fig. 2), correlation heatmaps with τ printed in cells (Rampisela, Dervishaj 2025), radar chart (Li). Subfigures labelled (a)(b)(c) by dataset/model. Legends inside panels.
- Every results subsection is **named after an RQ** ("5.1 RQ1: Impact of LMs on news recommendation accuracy") and ends with an explicit answer sentence: "(Answer to RQ1) This demonstrates that…", "In response to RQ3, these findings suggest…", or a bold summary sentence ("Overall, fig. 2 shows that, **for our denoising task, prompt optimality is LLM-specific and dataset-specific**").
- Numbers are quoted in prose with the metric and the comparator: "FREEDOM's NDCG@10 dropped by 28.0%, DRAGON's fell by 22.2%"; "up to 2.9% on MovieLens 1M (Qwen, few-shot-2 prompt)".

## 7. Statistics-reporting conventions

- The dominant pattern is **marker + threshold**, never exact p-values in prose: "∗ indicates statistical significance with p < 0.05", three-level daggers, or "(*) denotes statistical significance at p < 0.05" on a heatmap.
- **Confidence intervals are almost never reported**; variability is "±" standard deviation over seeds in the table cell ("0.1621 ± 0.0039") or "Standard deviation is denoted by ±" in the caption. Bland–Altman limits of agreement is the only interval construct in the corpus (Mohtadi).
- Always state the **pairing unit** and the comparison: "paired t-tests comparing λ values to the baseline (λ = 0)"; "between the best and second-best"; "between the original prompt with the best prompt".
- Non-significance is reported honestly and briefly: "Many of the observed differences are not statistically significant."; "no consistent statistically significant relation between the two".
- No multiple-comparison correction is mentioned anywhere in the corpus (reviewers did not require it; you may still add Bonferroni/Holm in one clause).
- Effect sizes are relative percentages over a named reference, computed on the metric, not standardized.

## 8. Limitations and conclusion pattern

- **Limitations** (when present) are 2–4 short paragraphs, each with a bold run-in name ("**Domain Specificity**", "**Constrained Conversational Scope**"), each ending with a future-work sentence. Cost/compute and single-dataset scope are the standard admissions: "we only use the MIND-small dataset … due to resource constraints".
- **Conclusion** = (i) one sentence restating what was done, (ii) 2–4 sentences restating the findings **with the numbers again** ("up to 13% improvement … whilst denoising up to 50% of the users"; "93–95% success rates"), (iii) one sentence of implication/future work. Reproducibility papers add a lesson: "the reported superiority of a new model can sometimes be an artifact of a focused tuning effort rather than an inherent architectural advantage."
- Broader-impact paragraphs are rare (1/11) and short.

## 9. Prose register

- Mean sentence length **19–24 words** (median 18–23); paragraphs 5–9 sentences.
- First person plural throughout: "we"/"our" at **0.8–1.6 per 100 words**; passive voice is used for protocol steps ("All ratings are binarized by…").
- Hedging is low (0.03–0.33 hedge words per 100 words). Claims are stated flatly and then scoped: "PUF is the only measure that can express this important distinction."; "Our method is the only post-training denoising method of user profiles in the literature."; scoping done by "in the datasets and models that we study".
- Findings are often typeset **bold** or set off as "(Answer to RQx)" sentences; the corpus is comfortable with strong verbs ("plummet by 29.5%", "collapses toward zero", "decisively outperforming").
- Section cross-references use "§ 3.1" or "Section 3.1" (both), "Table 2"/"Tab. 2", "Fig. 1"/"Figure 1", "fig. 1" (lower-case cleveref style in the Copenhagen papers).
- British and American spelling both appear; be consistent within the paper.

## 10. Ten do / don't items (with corpus quotes)

1. **Do** open the abstract with the accepted premise you are about to test, not with your method. *"Most studies suggest that (i) news RSs achieve better performance with larger pre-trained language models… However, other studies indicate that PLMs sometimes lead to worse performance than SLMs."* (Zhao)
2. **Do** put a number in the abstract and repeat it in the conclusion. *"our denoising yields improvements up to 13%"* → conclusion: *"up to 13% improvement on effectiveness across 3 datasets whilst denoising up to 50% of the users."* (Dervishaj 2026)
3. **Do** enumerate RQs and name each results subsection after one. *"4.1 RQ1: Replicability Study … 4.2 RQ2: Fair Benchmarking"* (de Gioia). **Don't** answer RQs only implicitly — write the answer sentence: *"(Answer to RQ2) Although the models are integrating multimodal embeddings, they operate in a state of textual dominance."* (Ye)
4. **Do** justify each baseline and say where its implementation came from. *"For each model, we use the official implementation released by the authors."* (Dervishaj 2025). **Don't** list baselines without citations or provenance.
5. **Do** state tuning budget and selection metric for *all* arms, and say so explicitly. *"we performed a hyperparameter optimization for all baselines by running 20 trials based on the Tree-structured Parzen Estimator (TPE) [4] strategy, with Recall@20 as the validation metric, ensuring that each algorithm could perform at its maximum potential."* (de Gioia)
6. **Do** put the bold/underline/marker legend in the caption. *"Best scores are in bold, and second-best are underlined. Blue marks scores better than at least one random method."* (Dervishaj 2026). **Don't** explain table formatting in the body text.
7. **Do** name the test and the pairing in one clause and mark cells with ∗/†/‡. *"Statistical significance is indicated by a ∗, based on paired t-tests comparing λ values to the baseline (λ = 0). Standard deviation is denoted by ±."* (Slokom). **Don't** write exact p-values in prose.
8. **Do** report non-results and reversals plainly. *"Our findings do not support the alleged trade-off between disentanglement and effectiveness [21,24,27,39], due to no consistent statistically significant relation between the two."* (Dervishaj 2025); *"Perhaps the most impactful finding of our benchmark is the re-evaluation of BISM's performance relative to the FPSR family… indicating a direct reversal of the performance hierarchy observed in the original findings."* (de Gioia)
9. **Do** move overflow to an online appendix with a footnote, keeping the trend claim in the paper. *"For space, we show only MovieLens & Amazon CDs & Vinyl. Yelp has similar trends."* (Dervishaj 2026); *"the rest are in the online appendix"* (Rampisela). **Don't** cut the datasets table or the significance markers to save space.
10. **Do** anonymise the code link for submission and keep the footnote slot. *"For replicability purposes, all embedding model settings and instructions are available online at: [REDACTED FOR BLIND REVIEW]."* (Mohtadi). **Don't** cite your own prior work in a way that identifies you (the corpus cites own work in third person, e.g. de Gioia citing Elliot [2] as "All experiments were conducted using Elliot [2]").

## 11. Corpus summary

Pages are of the arXiv PDF; "body" = pages before References. Section page ranges are the page on which each top-level section starts. #refs counted from the bibliography.

| Paper | Track | Pages (body/total) | Sections (start page) | #Tables | #Figures | #Refs | Sig. test | Code URL location |
|---|---|---|---|---|---|---|---|---|
| Li 2026 (CARE-CRS) | full | 12.5 / 18 | Intro 1 · RW 3 · Prelim 4 · Method 5 · Experiments 8 (Setup 9, RQ1 9, RQ2 11, RQ3 12) · Concl 13 | 2 | 9 | 59 | ∗ p<0.05 (test unnamed) | none in preprint |
| Ramos 2026 (Interplay) | full | 13 / 17 | Intro 1 · RW 3 · Method 4 · Setup 8 · Results 9 · Limitations 12 · Concl 13 | 3 | 3 | 32 | ∗ p<0.05 on human eval | Setup §4.1, inline URL |
| Dervishaj 2026 (denoising) | full | 12.5 / 15 | Intro+RW 1 · Method 3 · Setup 5 · Results 8 · Lim+Concl 12 | 3 | 4 | 35 | paired t-test, ∗/†/‡ | abstract |
| Ye 2026 (multimodal knockout) | full | 12.5 / 16 | Intro 2 · RW 3 · Prelim 4 · Setup 5 · Results 8 · Discussion 10 · Concl 13 | 3 | 4 | 41 | none | abstract |
| Rampisela 2026 (PUF) | full (Findings) | 12.5 / 18 | Intro 1 · Background 2 · Method 3 · Empirical analysis 4 · RW 12 · Concl 12 | 3 (+1 inline) | 4 | 56 | Kendall τ ≥ 0.9 rule; no t-test | abstract |
| Mohtadi 2026 (Q–D bias) | full | 12 / 16 | Intro 1 · RW 3 · Method+Setup 4 · Results 8 · Concl 12 | 1 | 5 | 40 | Gwet's AC1, Bland–Altman | p.1 footnote (redacted) |
| de Gioia 2026 (FPSR repro) | reproducibility | 12.5 / 15 | Intro 1 · RW 3 · Setup 5 · Results 7 (RQ1–4) · Concl 12 | 5 | 0 | 33 | ∗ p<0.05 best vs 2nd | abstract + footnote |
| Zhao 2025 (LMs in news rec) | full | 13 / 16 | Intro 1 · RW 3 · Method 4 · Setup 7 · Results 8 · Limitations 12 · Concl 13 | 3 | 7 | 39 | none (3-run means) | Setup §4, inline URL |
| Slokom 2025 (diversify) | full | 13 / 17 | Intro 1 · RW 3 · Method 4 · Setup 7 · Results 9 · Diversity 10 · Fairness 11 · Concl 12 | 3 (+2 algorithms) | 2 | 47 | paired t-test, ∗, ± SD | p.1 footnote |
| Dervishaj 2025 (disentanglement) | full | 12.5 / 16 | Intro 1 · Background 3 · Experiments 4 · Results 8 (RQ1–3, Limitations 12) · Concl 12 | 5 | 2 | 53 | RMCORR, ∗ p<0.05 | abstract |
| Sun 2025 (prompt variations) | full | 13 / 15 | Intro 1 · RW 2 · Method 4 · Results 7 (6 questions) · Discussion+Limitations 12 · Concl 13 | 5 | 1 (12 panels) | 36 | two-tailed paired t-test, ∗/† | conclusion |

Quantitative prose profile (body text): 4,900–7,400 words; mean sentence 19.6–24.4 words; "we" 0.8–1.6 per 100 words; hedge words ≤ 0.33 per 100 words; 2–13 URLs in body (footnotes); 3–4 numbered equations typical (range 0–5).
