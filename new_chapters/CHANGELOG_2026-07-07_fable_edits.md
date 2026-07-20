# Changelog — conservative fold-in edits (2026-07-07)

Scope: fold this week's validated results into Papers A and B and remove claims now known dead.
Narratives were **not** restructured (major revision scheduled with the author). No files other than
paper1/paper2 and this changelog were touched. Paper3/4/5 untouched.

Source docs read before editing:
`experiments/I25_FOLD_RESULTS.md`, `experiments/E0F_RESULTS.md`,
`ANSWERABILITY_RESULTS_LOG.md`, `ANSWERABILITY_WRITEUP.md`, `answerability_study_design.md`,
`answerability_design_review_2026-07-06.md`, `FABLE_AGENT_DESIGN_2026-07-07.md`
(§"I2.5 — THE PROPER INSTRUMENT" + "I2.5 BUILD RESULT"), memory `harsh-review-and-trainseed-overturn.md` (TRSEED).

---

## PAPER A — `paper1_casper/casper_u_chapter.tex` (instrument/testbed chapter)

### A1 + A2 — NEW subsection "The fold is part of the instrument" (`\label{sec:i2fold}`)
- **Location:** inserted at the end of the strong-instrument section (`sec:i2`), immediately before
  `\section{Discussion}`.
- **What:** New subsection introducing the algebraic belief operator `z' = z + eta*a*q`, certified only
  in the concept-geometric envelope; the E0f out-of-envelope canary failure (one free true real-rating
  item answer from cold HURTS NDCG@10 by −0.053, CI [−0.079,−0.027]); the I2.5 learned Deep-Sets residual
  fold `z = z_native + rho(answer set)` (zero-init around frozen RecVAE), data-side / non-circular; the
  canary flip under the learned fold (+0.095 item / +0.066 concept / +0.068 attribute); 5/6 gates pass.
  Includes the **6-gate table** (`\label{tab:i2fold}`, A2) with numbers/CIs from `I25_FOLD_RESULTS.md §2`.
  Includes the **G-fold4 displacement-not-defect** diagnosis (concepts ADD on top of items +0.0076 CI
  [+0.0035,+0.0115]; item marginal ≈18× concept; re-scope to G-fold4′ which passes; both forms reported),
  and the **LESSON** paragraph (gate suites need an out-of-envelope canary; the fold is part of the
  instrument and silently decides downstream verdicts). **G-fold6 reported honestly** (learned full-profile
  0.465 vs native 0.479, Δ=−0.014 CI [−0.026,−0.002], inside declared ±0.02 tolerance, flagged via `$^\dagger$`).
- **Why:** fold in the I2.5 build result + E0f canary; establish "THE FOLD DECIDES" methodological point.
- **Source:** `I25_FOLD_RESULTS.md`, `E0F_RESULTS.md`, `FABLE_AGENT_DESIGN_2026-07-07.md`.

### A3 — Canary added as a named gate class
- **Location:** acceptance suite `description` list in `sec:privladder` region (added item **G8
  out-of-envelope canary** after G7 polarity).
- **What:** new gate class requiring one free true answer from cold to help on every channel it will be
  asked over, even outside the operator's tuned regime; forward-reference to `sec:i2fold`.
- **Why:** A3 — name the canary as a gate class where the chapter lists the gate suite.
- **Source:** `FABLE_AGENT_DESIGN` gate suite (G-fold1 as permanent canary), `E0F_RESULTS.md`.

### A4 — Scoped (not removed) the additive-operator material
- **Location:** `sec:i2gates` (the "certified gate suite: 16/16" subsection), sentence about the belief
  operator's sign / polarity.
- **What:** added a clause: the algebraic operator `z'=z+eta*a*q` is "certified and valid within the
  concept-geometric envelope; outside it — for real-rating item answers — it is mis-specified and must be
  replaced by the learned fold of §sec:i2fold." Additive material retained.
- **Why:** A4 — scope, do not delete.

### Bibliography (Paper A)
- **Added two real bib entries** (used by `sec:i2fold`): `\bibitem{zaheer17}` (Deep Sets, NeurIPS 2017)
  and `\bibitem{dropoutnet17}` (Volkovs et al., DropoutNet, NeurIPS 2017). Not placeholders. Existing
  `\cite{eddi19}`, `\cite{recvae}` reused.

---

## PAPER B — `paper2_casper/paper2_casper.tex` (answerability chapter)

### B1 — NEW major section "Externally-sourced answerability" (`\label{sec:external}`)
- **Location:** inserted as a full `\section` immediately before `\section{Beyond the heuristic: CASPER-R...}`
  (`sec:learned`).
- **What:** the LLM answerability study as the chapter's second pillar: motivation (answerable-iff-rated is a
  designer-authored lower bound / self-authored-circularity trap); design (LLM judge reads known half,
  pinned gpt-5.4-mini temp 0, fixed answerer split seed 123, maybe→refuse, pre-registered 4-test gate);
  **gate table** (`\label{tab:extgate}`: ICC 0.174 p=.003; OR 2.71/sd, dAUC .106; validity gap +44.1pt;
  G2 exploitability 0.287 CI[.268,.307] as a privileged upper bound); cross-checks (Haiku κ=0.67
  sign-replicates; EASE value MAE 0.74 vs LLM 0.68); main study (fitted P(answerable|features), held-out-user
  AUC 0.921, ECE 0.008, **coefficient table** `\label{tab:extcoef}` with genre_match +0.63 as the
  user-specific fuel; masked-value MAE 0.70 = fidelity sigma); claim boundary (external, two-model-robust,
  human validation pending, NOT "realistic users"; fitted model = LLM-derived scale surrogate, a-priori rule
  = independent witness).
- **Why:** B1 — the study is Paper B's second pillar / enabling engine.
- **Source:** `ANSWERABILITY_RESULTS_LOG.md`, `ANSWERABILITY_WRITEUP.md`, `answerability_study_design.md`,
  `answerability_design_review_2026-07-06.md`.

### B2 — Killed the learned-policy (CASPER-R) endpoint win (TRSEED)
- **Locations & what:**
  - `sec:learned` opening paragraph: rewritten from "beats the heuristic … first learned-policy win" to
    "**We claim no learned endpoint win**" — the +0.012 tail margin is a single favourable *training* seed;
    over 3 training seeds the mean q8 tail margin ≈ 0 (favourable seed +0.008/q2,+0.001/q8; a second seed
    −0.007/−0.003). Section reframed as a **descriptive mechanism analysis**; mechanism content
    (belief-sharpening, front-loading, personalisation) retained as reproducible description.
  - "Reading" paragraph after `tab:learned`: rewritten — margins are single-training-seed and do not survive
    training-seed variation; belief-cosine/front-loading read as characterisation, not endpoint win.
  - `tab:learned` caption: re-captioned "single favourable training seed … **does not survive training-seed
    variation** … averaged over training seeds it matches the heuristic". Data unchanged.
  - `tab:ablation` caption: re-captioned — absolute reconstruction lift does not survive training-seed
    variation; the *ordering* (reconstruction > RL > ranking; pretraining essential) is the reproducible
    finding. Data unchanged.
  - `sec:background` footnote (seed 123): rewritten — the five seeds vary only the user split; under training
    seeds CASPER-R's q8 tail margin averages ≈0; we claim no learned endpoint win.
  - `sec:adapt`: softened "sharpens the belief and the tail ranking" → belief-sharpening reproducible, the
    endpoint tail margin it buys is not.
- **Why:** B2 — the win is training-seed-fragile (TRSEED) and does not survive; robust B = answerability.
- **Source:** memory `harsh-review-and-trainseed-overturn.md` (`experiments/paper2/TRAINSEED_RESULT.md`).
- **Note:** `sec:ansceiling` already contained "we claim no learned endpoint win in the closed-concept
  setting" (anytime-reward test); the edits above make the whole-chapter verdict consistent with it.

### B3 — Re-labelled "answerable iff rated" as the structural lower-bound model
- **Locations:** `sec:background` (geometric-answers paragraph) and `sec:learned` protocol paragraph.
- **What:** the rated/≥2-member rule is now named the "structural lower-bound answer model, one of the two
  answerability models used in this chapter", with a forward-reference to `sec:external` (the measured
  alternative).
- **Why:** B3.

### B4 — Updated abstract + contributions
- **Abstract:** rewrote the CASPER-R sentence to "we claim no learned endpoint win" + mechanism analysis;
  added the external-measurement pillar (validity gap +44pt, κ=0.67, AUC 0.92, ECE 0.008); stated the carried
  contributions = answerability axis + external measurement.
- **Intro paragraph (iii):** rewritten — external measured answer model + CASPER-R as mechanism (no endpoint
  win); carried contributions are the answerability axis and its external measurement.
- **Contributions list:** item 1 replaced with **"Externally-sourced answerability"** as the headline method
  contribution; CASPER-R demoted to a **"mechanism analysis"** item that explicitly claims no endpoint win
  (matches, not beats). Established negative-result and answerability-lever items unchanged.
- **Why:** B4.

---

## Compile status / risks
- Both files: `\begin`/`\end` balanced (Paper A 85/85 by-name matched; Paper B 28/28). New tables' column
  counts verified (tab:i2fold 4 cols; tab:extgate 4 cols; tab:extcoef 7 cols).
- All `\ref` in both files resolve to defined `\label`; all `\cite` resolve to `\bibitem` (checked by diff).
- Fixed a math-mode-only `\;` used in a text cell of `tab:extgate` → replaced with `\quad` (4 cells).
- **No LaTeX toolchain present in this environment** (no pdflatex/latexmk/xelatex), so this is a static
  check, not an actual compile. Recommend a local `pdflatex` pass before circulation.
- **Missing bib keys:** none. Two new Paper A bib entries added (`zaheer17`, `dropoutnet17`) with real
  references — verify their exact venue strings against your `.bib` conventions if a shared bibliography is
  used later. No placeholder `\cite` keys were introduced.
