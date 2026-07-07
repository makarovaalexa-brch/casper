# Changelog / verification pass — Paper A & B fold-in edits (2026-07-08)

**Status: edits are ON DISK and VERIFIED.** A prior run's edits (documented in
`CHANGELOG_2026-07-07_fable_edits.md`) had, per the handoff, previously *reported* success without
reaching disk. This pass confirms that the edits are now genuinely present in both `.tex` files,
that their numbers match the source documents, and that both files pass a static compile-integrity
check. No further content edits were required; the work described in the 07-07 changelog is real
and correct as landed.

File mtimes at verification: `casper_u_chapter.tex` 2026-07-07 23:46; `paper2_casper.tex`
2026-07-07 23:50 (both post-date the Jul-5 baseline the handoff flagged as stale → edits landed).

Sources re-read for verification: `experiments/I25_FOLD_RESULTS.md`, `experiments/E0F_RESULTS.md`,
`ANSWERABILITY_WRITEUP.md`, `ANSWERABILITY_RESULTS_LOG.md`, `STATE_2026-07-08.md`, memory TRSEED.

---

## PAPER A — `paper1_casper/casper_u_chapter.tex`  (all present & number-checked)

| task | location | verified content | source cross-check |
|---|---|---|---|
| A1 subsection "The fold is part of the instrument" | `\subsection{...}\label{sec:i2fold}` L1453, before `\section{Discussion}` L1529 | algebraic operator `z'=z+\eta a q` scoped to concept-geometric envelope; E0f canary −0.053 CI[−0.079,−0.027], corr 0.39, sign-disagree on dislikes, η overshoot; learned Deep-Sets residual `z=z_native+ρ(answer set)` zero-init around frozen RecVAE, data-side/non-circular, dropout curriculum; canary flip +0.095/+0.066/+0.068; 5/6 gates | E0F_RESULTS headline + fold-miscalibration §; I25_FOLD §1–2 |
| A1 6-gate table | `\label{tab:i2fold}` L1484–1505 | G-fold1 item +0.095[.070,.121], concept +0.066[.047,.086], attr +0.068[.050,.087]; G-fold2 .238→.407 item/.238→.362 mixed/flat conc-attr; G-fold3 k=2 +0.012[.002,.022]; G-fold4 mixed .314 vs item-8 .348 Δ−0.073[−.091,−.056] FAIL; G-fold5 max step −0.002; G-fold6 .465 vs .479 Δ−0.014[−.026,−.002] PASS† | I25_FOLD §2 table — all figures match |
| A1 G-fold4 displacement diagnosis | paragraph L1507 | concepts ADD +0.0076[.0035,.0115] on item-8, +0.0020 (spans 0) on item-4; item marginal +0.037[.019,.054] ≈18× concept; re-scope to G-fold4′ (passes +0.0076); both forms reported | I25_FOLD §"G-fold4 failure" diag table |
| A1 LESSON paragraph | L1520 | out-of-envelope canary needed; belief-update machinery is part of the instrument and silently decides downstream verdicts | I25_FOLD §3 "THE FOLD DECIDES" |
| A2 canary as named gate class | `\item[G8 out-of-envelope canary]` L442, in acceptance-gate `description` list | added after G7 polarity; forward-ref `\S\ref{sec:i2fold}` | task A2 |
| A3 scope (not delete) additive operator | L1433–1437 (`sec:i2gates`) | `z'=z+\eta a q` "certified and valid within the concept-geometric envelope … outside it … replaced by the learned fold of \S\ref{sec:i2fold}"; additive material retained | task A3 |
| A4 bib entries | `\bibitem{zaheer17}` L1622, `\bibitem{dropoutnet17}` L1623 | real refs (Deep Sets NeurIPS 2017; DropoutNet NeurIPS 2017), matching the file's `\bibitem` bibliography style | task A4 — both cited in sec:i2fold, resolve |

## PAPER B — `paper2_casper/paper2_casper.tex`  (all present & number-checked)

| task | location | verified content | source cross-check |
|---|---|---|---|
| B1 section "Externally-sourced answerability" | `\section{...}\label{sec:external}` L343, before `sec:learned` L429 | motivation (answerable-iff-rated = designer-authored lower bound / self-authored-circularity trap); design (LLM judge, known-half, gpt-5.4-mini temp 0, seed 123, maybe→refuse, pre-registered gate); claim boundary (external + two-model-robust; human validation pending; never "realistic users"; fitted model = LLM-derived scale surrogate, a-priori rule = independent witness) | ANSWERABILITY_WRITEUP §1–2,6; study_design |
| B1 gate table | `\label{tab:extgate}` L365–383 | ICC 0.174 p=.003; OR 2.71/sd, dAUC .106; validity gap +44.1pt (n=768); G2 0.287 CI[.268,.307] labelled PRIVILEGED upper bound | WRITEUP §3 — all match |
| B1 cross-checks | paragraph L385 | Haiku κ=0.67, 83.2% agreement, sign-replicates; EASE MAE 0.74 vs LLM 0.68 | WRITEUP §4 |
| B1 main study + coef table | `\label{tab:extcoef}` L403–418 | fitted P(answerable): AUC 0.921 held-out users (grouped-5-fold 0.916), ECE 0.008; genre_match +0.63 = user-specific fuel; masked-value MAE 0.70 stars = fidelity σ | WRITEUP §5 + STATE §1 |
| B2 kill learned-policy win | `sec:learned` opening L434 | "We claim no learned endpoint win"; +0.012 tail = single favourable training seed, mean over 3 seeds ≈0 (−0.001; fav +0.008/q2,+0.001/q8; 2nd −0.007/−0.003); reframed as descriptive mechanism analysis; mechanism (belief-sharpening cos 0.74 vs 0.64, front-loading, personalisation) kept | memory TRSEED; STATE §2 |
| B2 re-captions (data kept) | `tab:learned` cap L559; `tab:ablation` cap L612 | both carry "does not survive training-seed variation"; ablation *ordering* kept as reproducible | task B2 |
| B3 relabel "answerable iff rated" | `sec:background` L137; `sec:learned` protocol L453 | named "structural lower-bound answer model", cross-ref `\S\ref{sec:external}` | task B3 |
| B4 abstract + contributions | abstract L26; intro (iii) L71–75; contributions L82 (headline = Externally-sourced answerability) + L96 (CASPER-R demoted to mechanism analysis, no endpoint win) | answerability axis + external measurement = the claims; no learned-policy claim | task B4 |

---

## Final verification (mandatory grep + integrity)

Grep hits (both `.tex` files, this pass):
- Paper A: `sec:i2fold` L1453 (def) + refs; `tab:i2fold` L1484/1504; `The fold is part of the instrument` (1); `out-of-envelope canary` L442/1468/1521; `zaheer17`/`dropoutnet17` bibitems L1622/1623; `G-fold` (11 hits); `displacement, not defect` L1507.
- Paper B: `sec:external` L343 (def) + refs; `tab:extgate` L365/382; `tab:extcoef` L403/417; `Externally-sourced answerability` L82/343 (3 hits); `no learned endpoint win` L26/75/159/434 (5 hits); `does not survive training-seed variation` L435/559/612 (6 hits total); `structural lower-bound answer model` L137/454.

Compile-integrity (Python static check, `scratchpad/check.py`):
- `casper_u_chapter.tex`: `\begin`/`\end` **85/85 balanced**; unresolved `\ref` → **none**; unresolved `\cite` → **none** (40 bibitems, incl. new zaheer17/dropoutnet17).
- `paper2_casper.tex`: `\begin`/`\end` **28/28 balanced**; unresolved `\ref` → **none**; unresolved `\cite` → **none** (18 bibitems).

## Compile risks
- No LaTeX toolchain in this environment → this is a **static** check, not a real `pdflatex` run. Recommend a local compile before circulation.
- No placeholder cites introduced. New Paper A bib entries (`zaheer17`, `dropoutnet17`) are real; if a shared `.bib` is adopted later, normalise their venue strings to house style.
- New tables use text in the last column with `\quad` separators (not math-mode `\;`) per the 07-07 fix note — verified no stray math-mode-only spacing in text cells of the new tables.
