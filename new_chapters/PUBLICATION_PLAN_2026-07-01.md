# CASPER Publication Plan — 2026-07-01

Companion to `REVIEW_HARSH_2026-07-01.md` (harsh review of all 5 drafts).
Constraints: user is UK-based, two young kids, **no travel** (co-author/supervisor presents
if unavoidable); ~1 year of PhD left; defense bar ≈ **one more accepted + one more submitted**
(IJCNN 2024 Makarova et al. already published — CHECK with supervisor whether that alone
already satisfies the requirement).

---

## EXECUTIVE SUMMARY — RECOMMENDATION

**Consolidate 5 drafts → 3 publications. Two go to ECIR 2027 (Southampton, UK — deadline
2 Oct 2026, decision ~mid-Dec 2026). One goes to a journal (TORS or UMUAI) in autumn 2026.**

| # | Publication | Built from | Venue (primary) | Deadline | Decision |
|---|---|---|---|---|---|
| 1 | **Method paper**: "continuous questions win, snapping kills it, and the deepest snap-loss is un-askability" | **C** (core) + B's answerability as motivation + B's CASPER-R as baseline + E's snap/un-askability analysis | **ECIR 2027 full paper** | 2 Oct 2026 | ~mid-Dec 2026 |
| 2 | **Testbed paper**: CASPER-U instrument, oracle-bounded headroom, answerability×information quantification | **A** (as is, after fixes) | **ECIR 2027 reproducibility track** (fallback: short paper) | ~Oct 2026 (check track CfP) | ~Dec 2026–Jan 2027 |
| 3 | **Deployment paper**: open free-recall beats probing; framing lever; deployed agent; human micro-study | **D** + E's agent/deployment §§ | **Journal: ACM TORS or UMUAI** | rolling (target Oct–Nov 2026) | first decision ~Dec 2026–Feb 2027 ("feedback by spring") |
| — | Paper B standalone | **dies as a submission**; content distributed to #1 (motivation/baseline) and thesis chapter | — | — | — |

Why this shape:
- Review verdicts: B is fragile standalone (headline pending clean-split rerun; best content
  is C's motivation); E is "a very good section of another paper" (its §3 literally extends
  C's snap-loss); merged #1 is STRONGER than C alone.
- One venue (ECIR) = one UK train trip covers both conference submissions.
- #3 needs the human study + length → journal is the right format, and journal "submitted"
  status immediately satisfies the defense bar; decision arrives by spring regardless.
- Clean content boundaries between #1/#2/#3 → **no dual-submission risk** (never have
  overlapping content simultaneously at a conference and a journal).
- Two independent ECIR shots maximize P(≥1 acceptance by December), which is the
  load-bearing acceptance for the defense.

Contingency ladder for #1 if ECIR rejects (~mid-Dec): → **SIGIR 2027** (deadline ~late Jan
2027, decision ~Apr 2027, location TBA — go only if Europe, else supervisor presents)
→ or **ACM TOIS / TORS** (journal, no travel). A December rejection with 3 reviews is a
strong base for a January resubmission. For #2: → TORS. For #3 if journal major-revises:
that's normal, revise within 90 days; if rejected → UMAP 2027 (~Jan 2028 cycle) or CUI.

Timeline:
- **Jul (days)**: text-level fixes from harsh review (stale numbers, citations, fair-margin
  reframes). **COMMIT untracked manuscripts NOW** (paper4 tex, paper5/, paper2 rewrite,
  casper/continuous_actor.py, REVIEW_HARSH md).
- **Jul–Aug (calendar-bound, cannot compress)**: the 4 decisive experiments — graded-discrete
  control (C), attr-oracle@745×2-answer-models (A), clean-split rerun (B → decides whether B's
  content enters #1 as a positive or a negative result), noisy hidden-gem answerer (D);
  training-seed retrains of D1; ~30-person human micro-study (serves #3, ideally #1's
  un-askability too).
- **Aug–Sep**: supervisor sign-off; papers #1 + #2 into ECIR format; AI-use disclosure
  statements added (ACM requires disclosure in acknowledgements).
- **2 Oct 2026**: ECIR submission (#1 full, #2 repro/short).
- **Oct–Nov**: assemble #3 (D+E+human study) → TORS or UMUAI.
- **Dec**: ECIR decisions → celebrate or execute contingency ladder (SIGIR abstract ~mid-Jan).

---

## ALL VENUE VARIANTS CONSIDERED (as of 1 Jul 2026)

### Conferences

| Venue | Deadline | Event dates | Location | Travel verdict | Fit |
|---|---|---|---|---|---|
| **ECIR 2027** full/short | **2 Oct 2026** (GMT) | 21–25 Mar 2027 | **Southampton, UK** | Train ride — PRIMARY | #1 excellent (IR methodology); #2 excellent |
| **ECIR 2027** reproducibility track | ~Oct 2026 — verify on ecir2027.co.uk | same | same | same | #2 near-perfect (replication of 3 lineages + instrument) |
| **ECIR 2027** demo track | ~autumn 2026 — verify | same | same | same | Optional: E's chat agent as 4-page demo (cheap, showcases system without E's unfinished eval) |
| **SIGIR 2027** (50th anniversary) | ~late Jan 2027 (pattern; verify) | 20–24 Jul 2027 | TBA | Decide when announced | #1 backup after ECIR reject; prestigious |
| **UMAP 2027** | ~Jan 2027 (pattern) | mid-2027 | TBA (2026 was Gothenburg) | Decide when announced | #3 backup; B-flavored content fits UMAP reviewers best |
| **RecSys 2027** | ~Apr 2027 | ~late Sep 2027 | **Hawaiʻi** | AVOID unless supervisor presents | Community flagship for #1/#3, but travel kills it |
| **CIKM 2026** | PASSED (25 May 2026) | 7–11 Nov 2026 | Rome | — | off the table |
| **CIKM 2027** | ~May 2027 | late 2027 | TBA | Watch | generic backup for any of the three |
| **WSDM 2027** | ~Aug 2026 | Feb 2027 | Hong Kong | AVOID | poor travel; marginal fit |
| **CUI 2027** | ~early 2027 (verify) | mid 2027 | TBA | Watch | #3/E-content fit (conversational UI, un-askability) |
| Workshops (KaRS, IntRS, LLM4Good@UMAP…) | various | various | often hybrid/remote | Remote-friendly | fallback exposure for anything rejected twice |

Presentation policy note: post-pandemic, ECIR/ACM conferences are in-person-presentation by
default; remote is exceptional-circumstances only. Standard workaround: co-author (supervisor)
presents. Journals require no presentation → first-class option under the no-travel constraint.

### Journals (rolling, zero travel)

| Journal | Realistic time to acceptance | Fit |
|---|---|---|
| **ACM TORS** (Trans. on Recommender Systems) | first decision ~3–5 mo; accept ~9–15 mo (first decision usually major revision) | #3 primary; #2 fallback; #1 fallback. The RecSys community's journal — right audience for testbed + elicitation work |
| **UMUAI** (Springer, User Modeling & User-Adapted Interaction) | first decision ~4–8 mo; accept ~12–24 mo, often 2 rounds | #3 alternative (explicitly scopes preference elicitation + user modeling; loves long thorough papers — thesis-format friendly). SLOWEST — do not put deadline weight on it |
| **ACM TOIS** | ~12–18 mo | #1 stretch fallback (prestigious IR home) |
| **ACM TiiS** (Trans. Interactive Intelligent Systems) | ~12+ mo | niche fallback for E-flavored interactive/dialogue content |

**Key timing fact: no journal can be RELIED on for an acceptance within the remaining PhD
year. The load-bearing acceptance must come from a conference with a fixed decision date
(ECIR → Dec 2026; SIGIR/UMAP → ~Mar–Apr 2027; RecSys → ~Jun 2027, too tight).**

### Paper-fit matrix (draft → venues, best-first)

- **#1 (C+B+E-analysis)**: ECIR full → SIGIR → TOIS/TORS. (RecSys 2027 only with proxy presenter.)
- **#2 (A)**: ECIR reproducibility → TORS → UMAP.
- **#3 (D+E-agent)**: TORS or UMUAI → UMAP 2027 → CUI. (Needs the human micro-study first.)
- **B standalone**: none — dissolved into #1 + thesis chapter. (If its clean-split rerun
  surprises positively AND time allows: TORS short/UMAP later, honest "when policy learning
  pays" framing.)
- **E standalone**: none for now — un-askability could become its own CUI/CIKM paper LATER
  if the human/LLM-judge validation is done (see review §5); not on the current critical path.
  Optional cheap win: ECIR demo of the chat agent.

### Rejected variants (and why)

- **All 5 to ECIR as-is**: bandwidth (13 wks), sibling-paper/salami-slicing risk with shared
  ruler+baselines and overlapping reviewer pool, B/E not standalone-viable, B/E fit UMAP/CUI
  reviewers better.
- **One mega-paper of everything**: too big for any conference page limit; that's the thesis
  itself. (A single long journal article of C+D+E remains possible at UMUAI if all conference
  routes fail — last resort.)
- **Journal-first for the flagship**: timing risk — cannot guarantee acceptance before defense.
- **RecSys 2026 / CIKM 2026**: deadlines already passed.

### AI-disclosure requirements (all target venues)

ACM/Springer/IEEE: AI cannot be an author; LLM-generated text permitted WITH disclosure
(acknowledgements), e.g. "The authors used Claude (Anthropic) as a writing and coding
assistant throughout; all research questions, experimental designs, and conclusions are the
authors' own, and all results were produced by the authors' code and verified by the authors."
Authors bear full responsibility for LLM output (hallucinated citations = author ethics
violation). Thesis: governed by Univ. of Reading DaRC "Guidance on the responsible use of
generative AI in doctoral research" (Oct 2025 PDF) — read it and agree declaration wording
with supervisor early.
