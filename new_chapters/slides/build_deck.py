"""Build the CASPER supervisor presentation deck.

    poetry run python new_chapters/slides/build_deck.py

Emits new_chapters/slides/casper_supervisor_2026-09.pptx (16:9).
Everything is a native pptx object (text, tables, autoshapes, connectors) so the
deck stays fully editable after a Google Slides import.

EVERY number below traces to a source named in that slide's speaker notes.
Sources of record:
  Paper A  new_chapters/chapterA_v2/chapterA_v2.tex
           docs/design/GATE_BATTERY_INSTRUMENT.md
           docs/results/{CONCEPT_CHANNEL_RESULT,ANSWER_DECOMPOSITION_RESULT,
                         PAPER_A_NUMBER_AUDIT,G1C_CONTROL_RESULT}.md
           CLAUDE.md (canonical ruler / split sizes)
  Paper B  new_chapters/chapterB_v2/chapterB_v2.tex   (definitive)
           docs/design/PAPER_B_MASTER_TODO.md
           docs/results/{ACT1_PAIRED_CI,CONDITIONED_ARMS_ON_TEST,
                         HU_YU_2013_REPLICATION_AND_ABLATION,M4B_FOREST_POLICY,
                         M6L_OPENER_FITTED_TEACHER,REDDIT_LINKER_AUDITION,
                         P1_HELDOUT_ACTIONS,P1B_NEVER_TAUGHT,
                         BASELINE_OBJECTIVE_AUDIT}.md
  Paper C  new_chapters/paperC_lbr/paperC_lbr.tex, new_chapters/chapterC_v2/chapterC_v2.tex
           docs/results/{C0_GEOMETRY,C1_FINAL,C3B_ADAPTER,C3B_V2_QUESTIONS,
                         C4_PILOT,C4B_NOISE,C4C_DECODING,C4D_TRANSCRIPT}.md
  Venues   docs/PUBLICATION_PLAN.md
           docs/design/PAPER_C_PUBLICATION_ASSESSMENT_2026-08-25.md
"""
import os
from pptx.util import Inches, Pt
from pptx.enum.text import PP_ALIGN
from pptx.enum.shapes import MSO_SHAPE

import deck_lib as D
from deck_lib import (MARGIN_L, CONTENT_W, SLIDE_W, SLIDE_H,
                      INK, MUTED, RULE, BG, ACCENT, ACCENT_B, ACCENT_C,
                      BAND, textbox, notes, slide, table, box, arrow, chip,
                      bullets, footer)

L = PP_ALIGN.LEFT
C = PP_ALIGN.CENTER
R = PP_ALIGN.RIGHT

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "casper_supervisor_2026-09.pptx")

prs = D.new_deck()


def num(n):
    """Column alignment: label column left, the rest right."""
    return [L] + [R] * (n - 1)


# =====================================================================
# OPENING
# =====================================================================

# --- 1. title -------------------------------------------------------
s = prs.slides.add_slide(prs.slide_layouts[6])
bar = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, Inches(0.22), SLIDE_H)
bar.fill.solid(); bar.fill.fore_color.rgb = ACCENT
bar.line.fill.background(); bar.shadow.inherit = False
textbox(s, Inches(1.0), Inches(1.55), Inches(11.4), Inches(0.4),
        "CASPER  ·  PhD PROGRESS WALKTHROUGH", size=13, bold=True, color=ACCENT)
textbox(s, Inches(1.0), Inches(2.05), Inches(11.4), Inches(1.25),
        ["Cold-start conversational recommendation:",
         "the instrument, the interview, and the question that is not in the catalogue"],
        size=32, bold=True, color=INK, line_spacing=1.12, space_after=2)
textbox(s, Inches(1.0), Inches(3.40), Inches(11.4), Inches(0.4),
        "CASPER — Cold-start Adaptive Selection for Preference Elicitation in Recommendation",
        size=14, color=MUTED)
D._rule(s, Inches(4.05), ACCENT)
textbox(s, Inches(1.0), Inches(4.35), Inches(11.4), Inches(1.4),
        [("Paper A — the measuring instrument   ·   Paper B — what a cold interview "
          "should optimise   ·   Paper C — off-catalogue questions", {"size": 15}),
         ("", {"size": 6}),
         ("MovieLens-25M  ·  18,359-item vocabulary  ·  full and tail NDCG@10  ·  "
          "three papers, one fixed ruler", {"size": 13, "color": MUTED})],
        size=15, color=INK, space_after=3)
textbox(s, Inches(1.0), Inches(6.4), Inches(11.4), Inches(0.5),
        "September 2026", size=13, color=MUTED)
notes(s, """WHAT TO SAY
Opening frame. Three papers, one shared measurement stack, built in order: A makes the measurement
trustworthy, B uses it to answer a question about interviews, C asks what happens when the right
question does not exist in the catalogue. Roughly 50 minutes of material; if we run out of time we
stop wherever we are and continue later. There is a ~14-slide appendix behind this with the detailed
tables, so if you want to push on any number I can go straight to it.

SOURCES
Positioning and roster: docs/PUBLICATION_PLAN.md (roster reset 2026-08-18).
Ruler and vocabulary: CLAUDE.md 'Metric & reporting'; chapterA_v2.tex.""")


# --- 2. the thesis in one diagram ------------------------------------
s, y = slide(prs, "The thesis in one arc",
             "Three papers, one system — CASPER. Each hands the next a licensed capability, and "
             "nothing downstream is measured on a ruler the upstream paper did not certify.",
             kicker="orientation")

bw, bh = Inches(3.55), Inches(1.55)
gap = Inches(0.62)
x0 = MARGIN_L + Inches(0.25)
ytop = y + Inches(0.45)

box(s, x0, ytop, bw, bh,
    [("PAPER A", {"size": 12, "color": ACCENT}),
     ("The instrument", {"size": 17}),
     ("A certified cold-start recommender,", {"size": 11, "bold": False, "color": MUTED}),
     ("a fold-in operator, and one ruler", {"size": 11, "bold": False, "color": MUTED})],
    line=ACCENT)
box(s, x0 + bw + gap, ytop, bw, bh,
    [("PAPER B", {"size": 12, "color": ACCENT_B}),
     ("The interview", {"size": 17}),
     ("What should a cold interview", {"size": 11, "bold": False, "color": MUTED}),
     ("actually optimise?", {"size": 11, "bold": False, "color": MUTED})],
    line=ACCENT_B)
box(s, x0 + 2 * (bw + gap), ytop, bw, bh,
    [("PAPER C", {"size": 12, "color": ACCENT_C}),
     ("The unasked question", {"size": 17}),
     ("What if the right question is not", {"size": 11, "bold": False, "color": MUTED}),
     ("in the catalogue at all?", {"size": 11, "bold": False, "color": MUTED})],
    line=ACCENT_C)

ymid = ytop + bh // 2
arrow(s, x0 + bw, ymid, x0 + bw + gap, ymid, color=ACCENT_B, width=2)
arrow(s, x0 + 2 * bw + gap, ymid, x0 + 2 * (bw + gap), ymid, color=ACCENT_C, width=2)

cap_y = ytop + bh + Inches(0.34)
for i, (t, col) in enumerate([
        ("HANDS FORWARD  a frozen recommender, a certified fold-in operator, and one fixed ruler "
         "(full + tail NDCG@10)", ACCENT),
        ("HANDS FORWARD  the distilled interview policy E1, a 19,390-question bank, and per-state "
         "measured question values", ACCENT_B),
        ("CLAIMS  capability, not value — a direction outside the bank can be emitted, said in "
         "English, and audited", ACCENT_C)]):
    textbox(s, x0 + i * (bw + gap), cap_y, bw, Inches(1.1), t,
            size=10.5, color=col, line_spacing=1.14)

textbox(s, MARGIN_L, Inches(6.15), CONTENT_W, Inches(0.9),
        ["Common thread — a pre-registered control has overturned the headline three times:",
         ("C0's random-direction floor  ·  C3B's equivalence twin  ·  C1's untrained warm "
          "start.  Everything here is read against its own null.",
          {"size": 13, "bold": False, "color": MUTED})],
        size=14.5, bold=True, color=INK, space_after=3)
footer(s, "Sources: chapterA_v2.tex · chapterB_v2.tex · chapterC_v2.tex")
notes(s, """WHAT TO SAY
The whole of CASPER on one slide. A builds and certifies the measuring instrument, because without that
any elicitation result is unattributable — a null could be the policy or could be the recommender.
B takes that instrument and asks the real research question: what should a cold interview optimise?
C asks the capability question that falls out of B: what happens when the best direction to ask
about is not one of the questions you own.

Flag the common thread early: in this project a pre-registered control has overturned the headline
three separate times. That is the methodological character of the work and it is what I would most
like credit for.

SOURCES
Hand-offs and scope statements: chapterA_v2.tex :76, :249, :929, :1306-1307;
GATE_BATTERY_INSTRUMENT.md :86-88 (G10 certifies the ruler for every Chapter-B comparison).
E1 = Chapter B's distilled policy; bank 19,390 = 18,359 items + 1,031 concepts: chapterC_v2.tex Setup.
The three overturned headlines: C0_GEOMETRY.md (random direction lands in the same cell),
C3B_ADAPTER.md (criterion 2 fails on a favourable direction), C1_FINAL.md (untrained warm start
matches the own-beats-other margin).""")


# --- 3. the problem ---------------------------------------------------
s, y = slide(prs, "The problem: a cold user, a handful of questions, one shot",
             "A new user has no history. The system may ask a few questions. "
             "Which ones — and how much does the choice actually matter?",
             kicker="problem statement")

bw2, bh2 = Inches(2.20), Inches(0.92)
gy = y + Inches(0.35)
step = Inches(2.52)
xs = [MARGIN_L + Inches(0.10) + i * step for i in range(5)]
labels = [("USER", "no history"), ("QUESTION", "from a bank of 19,390"),
          ("ANSWER", "like / dislike / no answer"), ("BELIEF", "fold-in → latent point"),
          ("RANKING", "18,359 items, NDCG@10")]
cols = [MUTED, ACCENT_B, MUTED, ACCENT, ACCENT_C]
for i, ((h, sub), col) in enumerate(zip(labels, cols)):
    box(s, xs[i], gy, bw2, bh2,
        [(h, {"size": 12.5}), (sub, {"size": 9.5, "bold": False, "color": MUTED})],
        line=col)
    if i < 4:
        arrow(s, xs[i] + bw2, gy + bh2 // 2, xs[i + 1], gy + bh2 // 2, color=MUTED)
arrow(s, xs[3] + bw2 // 2, gy + bh2, xs[1] + bw2 // 2, gy + bh2, color=ACCENT_B, width=1.5)
textbox(s, xs[1], gy + bh2 + Inches(0.05), Inches(4.7), Inches(0.3),
        "the belief chooses the next question — this loop is Paper B",
        size=10, color=ACCENT_B, align=C)

table(s, [
    ["The setting, fixed once and reused everywhere", ""],
    ["Dataset", "MovieLens-25M, Liang strong-generalisation recipe, catalogue-restricted"],
    ["Split", "train 140,768 / val 10,000 / test 10,000 users, seed 98765, per-user 80/20"],
    ["Item vocabulary", "18,359 train vocabulary (arena catalogue 18,430 = films with ≥20 ratings)"],
    ["Question bank", "19,390 = 18,359 item questions + 1,031 concept questions"],
    ["Metric", "full AND tail NDCG@10, scored with the recommender's learned decoder bias"],
    ["Asking nothing", "0.15855 full / 0.02819 tail — the popularity prior, the floor"],
    ["Whole profile", "0.3496 full / 0.2491 tail — the same recommender given everything"],
], MARGIN_L, gy + bh2 + Inches(0.50), CONTENT_W, col_w=[24, 76],
    font_size=12, row_h=0.315, first_col_bold=True)
footer(s, "Sources: CLAUDE.md 'Metric & reporting' · chapterA_v2.tex :861 · chapterB_v2.tex tab:tree")
notes(s, """WHAT TO SAY
The loop: a cold user arrives with nothing, the system asks a question, gets an answer, folds it into
a latent belief, ranks the catalogue. The belief then chooses the next question — that feedback edge
is what Paper B is about.

Two numbers to hold on to from the table. Asking nothing scores 0.1586 full NDCG@10 — that is just
popularity, the floor. Handing the same recommender the user's entire history scores 0.3496. The
whole interview game is played inside that gap, and a good sixteen-question interview recovers
roughly a third of it. Every delta I show today should be read against that scale.

SOURCES
Split / vocabulary / metric: CLAUDE.md 'Metric & reporting (domain)'; PAPER_A_NUMBER_AUDIT.md:119
(18,359 train vocabulary vs 18,430 arena catalogue).
Bank composition: chapterC_v2.tex Setup.
Cold intercept 0.15855 / 0.02819: chapterB_v2.tex tab:tree ('no questions' row).
Full-profile 0.3496 / 0.2491: chapterA_v2.tex :861, :1426-1427.""")

# =====================================================================
# PAPER A — THE INSTRUMENT
# =====================================================================

def divider(title, sub, accent, meta):
    sd = prs.slides.add_slide(prs.slide_layouts[6])
    b = sd.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, Inches(0.22), SLIDE_H)
    b.fill.solid(); b.fill.fore_color.rgb = accent
    b.line.fill.background(); b.shadow.inherit = False
    textbox(sd, Inches(1.0), Inches(2.55), Inches(11.4), Inches(0.9), title,
            size=40, bold=True, color=INK)
    textbox(sd, Inches(1.0), Inches(3.55), Inches(11.0), Inches(0.9), sub,
            size=17, color=MUTED, line_spacing=1.15)
    textbox(sd, Inches(1.0), Inches(4.85), Inches(11.0), Inches(0.4), meta,
            size=13, bold=True, color=accent)
    return sd


sd = divider("Paper A — the instrument",
             "Before you can measure an elicitation strategy you have to certify the thing that "
             "does the measuring.",
             ACCENT, "8 slides  ·  ~12 minutes  ·  status: closed, 41pp, unsubmitted → TORS")
notes(sd, """WHAT TO SAY
Section break. Paper A is the least glamorous of the three and the one everything else stands on.
The argument is that in this literature strategy rankings invert depending on which recommender you
plug in, so a strategy comparison is only meaningful relative to a certified instrument.

SOURCES
Status: docs/PUBLICATION_PLAN.md roster table (author, 2026-08-18) — 'closed, 41pp, not yet
submitted' → TORS.""")


# --- A1: what A is ----------------------------------------------------
s, y = slide(prs, "What Paper A is, and why it has to exist",
             "Strategy rankings invert when you change the recommender. So the recommender is not "
             "background — it is the instrument, and it needs certifying first.",
             kicker="paper a · motivation", accent=ACCENT)

lw = Inches(6.35)
textbox(s, MARGIN_L, y + Inches(0.05), lw, Inches(0.3),
        "THE ARGUMENT", size=11, bold=True, color=ACCENT)
bullets(s, MARGIN_L, y + Inches(0.42), lw, [
    ("The problem.", "Elicitation papers each use their own recommender, so a "
     "strategy comparison is confounded with a model comparison."),
    ("The demonstration.", "Swap the consumer to a Golbandi leaf model and Thompson "
     "sampling flips from losing to static to beating it at every budget."),
    ("The response.", "One frozen tower, one fold-in operator, one ruler, and a "
     "battery of acceptance gates read before any result."),
    ("The consequence.", "A null in Paper B becomes attributable to the policy "
     "rather than to the instrument."),
], size=13)

rx = MARGIN_L + lw + Inches(0.45)
rw = CONTENT_W - lw - Inches(0.45)
textbox(s, rx, y + Inches(0.05), rw, Inches(0.3),
        "WHAT THE INSTRUMENT IS", size=11, bold=True, color=ACCENT)
table(s, [
    ["Component", "What it does"],
    ["Frozen tower", "shallow frontier recommender, i26c ep31, never refitted downstream"],
    ["Channel-agnostic tokens", "one representation for items, concepts, text and raw directions"],
    ["Fold-in operator", "a permutation-invariant set encoder maps evidence → a latent point; "
     "no gradient step at inference, defined for every t ≥ 0"],
    ["Concept operator", "a trained gated fold-to-point, zero-initialised so the empty interview "
     "is bit-identical to the tower"],
    ["The ruler", "full and tail NDCG@10 over the whole catalogue, learned decoder bias"],
], rx, y + Inches(0.42), rw, col_w=[33, 67], font_size=11, row_h=0.30, first_col_bold=True)

textbox(s, rx, Inches(5.55), rw, Inches(1.2),
        ["THE FLIP, MEASURED", ("Fifth consumer (Golbandi leaf model): Thompson beats the "
         "static order by +0.0100 / +0.0171 / +0.0148 / +0.0281 full at k=1,2,4,8, every "
         "interval excluding zero — the exact reverse of its ordering on the certified tower.",
         {"size": 11.5, "bold": False, "color": MUTED})],
        size=11, bold=True, color=ACCENT, space_after=3, line_spacing=1.12)
footer(s, "Sources: chapterA_v2.tex §intro, §method · chapterB_v2.tex :932-945 (the flip)")
notes(s, """WHAT TO SAY
Why this paper exists at all. In this literature every elicitation paper brings its own recommender,
so when two strategies are compared you cannot tell whether you compared strategies or models. We can
show that concretely: on our certified tower Thompson sampling loses to a simple static order at
every budget; swap the consumer to a Golbandi-style leaf model and Thompson wins at every budget,
with intervals excluding zero. Same strategies, same answers, opposite verdict.

So Paper A's job is to fix the instrument and certify it, and then Paper B's nulls mean something.

Note the design detail on the right that pays off repeatedly: the concept fold operator is
zero-initialised, so an empty interview is bit-identical to the bare tower. That means the identity
gate is inherited for free rather than being re-argued.

SOURCES
Instrument components: chapterA_v2.tex §method (:839-930), fold operator :243-249, :865-882,
:1109-1116; zero-init / bit-identical :1309-1315.
The consumer flip: chapterB_v2.tex :932-945 (Golbandi leaf-model consumer).
'Strategy rankings invert with the recommender': chapterA_v2.tex §sec:instrdep (:1596).""")


# --- A2: the ruler ----------------------------------------------------
s, y = slide(prs, "The canonical ruler",
             "One split, one metric, one scoring convention — stated once in 2026-07 and never "
             "changed since. Every number in all three papers is on it.",
             kicker="paper a · the ruler", accent=ACCENT)

table(s, [
    ["", "Decision", "Why it is the decision"],
    ["Recipe", "Liang strong-generalisation, applied verbatim to ML-25M",
     "the field's canonical protocol; lets us snap to published numbers"],
    ["Catalogue", "films with ≥20 ratings → arena 18,430; train vocabulary 18,359",
     "pre-filtered once so items and questions share one vocabulary"],
    ["Users", "ratings > 3.5, min 5 per user; train 140,768 / val 10,000 / test 10,000",
     "seed 98765; per-user 80/20 fold-in / target"],
    ["Primary metric", "full AND tail NDCG@10 (tail = the top-33 %-train-mass head "
     "mask removed, 190 head items)", "full alone hides everything popularity already gives you"],
    ["Scoring", "the recommender's own learned decoder bias — never a log-count "
     "popularity floor", "the popb floor craters full NDCG; correlation only ~0.79"],
    ["Secondary", "NDCG@100", "reported, never headlined"],
], MARGIN_L, y + Inches(0.08), CONTENT_W, col_w=[15, 47, 38],
    font_size=11.5, row_h=0.40, first_col_bold=True)

table(s, [
    ["The published-number bridge (ML-20M, NDCG@100) — the ruler is validated against print, "
     "not asserted", "ours", "published", "Δ", "verdict"],
    ["EASE (Steck)", "0.4203", "0.420", "+0.0003", "PASS"],
    ["RecVAE (Shenbin)", "0.4425", "0.442", "+0.0005", "PASS"],
    ["Mult-VAE (Liang)", "0.4223", "0.426", "−0.0037", "near-snap"],
    ["Mult-DAE (Liang)", "0.3969", "0.419", "−0.022", "FAIL → dropped from the bank"],
], MARGIN_L, Inches(4.95), CONTENT_W, col_w=[44, 12, 13, 12, 19],
    font_size=11, row_h=0.285, align=num(5), first_col_bold=True)
footer(s, "Sources: CLAUDE.md 'Metric & reporting' · chapterA_v2.tex :1368-1392 · "
          "PAPER_A_NUMBER_AUDIT.md :46-56")
notes(s, """WHAT TO SAY
The ruler. Nothing exotic — the point is that it was fixed once and never moved, and everything in
all three papers is on it.

Two things worth pausing on. First, tail NDCG. Full NDCG@10 on MovieLens is dominated by popularity;
you can score respectably by recommending the same 190 films to everybody. So we always report tail
as well, with the head mass masked out, and several results in this thesis are tail results only.

Second, the bottom table is the reproducibility bridge, and it is the bit an external reviewer liked
most. We re-ran four published systems on their own published protocol and matched their printed
numbers: EASE to three ten-thousandths, RecVAE to five. Mult-DAE we could not match — off by 0.022 —
so we dropped it from the baseline bank rather than carry an unvalidated row. That is the discipline:
if we cannot reproduce it, it does not get to be a baseline.

SOURCES
Split, vocabulary, metric definition, decoder-bias rule: CLAUDE.md 'Metric & reporting (domain)';
190 head items and tail definition ibid.; chapterA_v2.tex :1590-1593.
Bridge table: chapterA_v2.tex :1368-1392; PAPER_A_NUMBER_AUDIT.md :46-56 (ML-20M, NDCG@100;
RecVAE also R@20 .4144 vs .414 and R@50 .5525 vs .553).
popb vs learned decoder bias: CLAUDE.md; memory note popb-vs-decoder-bias-full-ndcg-bug.""")


# --- A3: the gate battery ---------------------------------------------
s, y = slide(prs, "The certification battery",
             "Eleven per-build gates and three one-time bridges, every threshold written down "
             "before the run. No gate, no progress.",
             kicker="paper a · gates", accent=ACCENT)

table(s, [
    ["Gate", "What it demands", "Threshold / result"],
    ["G0  strength & identity", "reproduces the frozen tower; empty evidence set → exactly the prior",
     "|μ₀| = 0 to precision; Spearman 1.0000 vs the frozen decoder bias"],
    ["G1  usable posterior", "posterior variance shrinks, and shrinks more along the queried direction",
     "total 1881.9 → 1805.1; directional shrink 11.5× vs a 2× bar"],
    ["G2  cold curve rises", "NDCG rises from turn 1 to turn 8 with the interval clear of zero",
     "PASS, with a one-free-true-answer canary"],
    ["G3  sign & intensity", "flipping an answer's sign must hurt; graded must beat binarised",
     "concept sign-flip −0.2012; graded − binarised +0.0039 full [+0.0025, +0.0053]"],
    ["G4  refusal boundedness", "a refusal must move the belief far less than an answer",
     "α_refuse/α_answer = 0.0007 vs a ≤0.05 bar — PASS"],
    ["G5  channel specificity", "a concept operator must lift its own members and beat its own "
     "popularity projection", "member-lift AUC 0.626; beats pop-projection by +0.0087, CI clear"],
    ["G6  existential controls", "wrong-user, shuffled and placebo evidence must buy nothing",
     "PASS"],
    ["G8  Σ load-bearing", "collapsing the covariance to cI must degrade the number", "PASS"],
    ["G9  ingestion isolation", "a fixed user-independent bank must still improve with t", "PASS"],
    ["G10 strategy discrimination", "oracle > random > adversarial-worst, each interval clean, "
     "full and tail", "PASS — this is the gate that licenses every Paper B comparison"],
    ["C1  published-number bridge", "match printed numbers on a published protocol",
     "DONE — EASE +0.0003, RecVAE +0.0005"],
    ["C2  answer-model transfer", "the gain survives held-out behavioural answers and label noise",
     "run; behavioural and stated preferences agree only ~21 %"],
    ["C3  human round-trip", "real people answer, held-out design", "NOT RUN — owed project-wide"],
], MARGIN_L, y + Inches(0.02), CONTENT_W, col_w=[21, 42, 37],
    font_size=9.8, row_h=0.315, first_col_bold=True)
footer(s, "Sources: docs/design/GATE_BATTERY_INSTRUMENT.md · chapterA_v2.tex tab:batteryresults · "
          "G1C_CONTROL_RESULT.md · CONCEPT_CHANNEL_RESULT.md")
notes(s, """WHAT TO SAY
This is the battery. Every per-build gate has been run and passes; the chapter's own summary table
reports R1–R4 plus two integrity items all PASS.

I want to be straight about three of these rather than let you find them. G4's confidence-ORDERING
sub-test is unmeasurable by design on MovieLens — there is no ground truth for how confident a user
is — so it is declared out of scope rather than passed. G1's calibration sub-gate was dropped: it
measured 0.239 against a 0.25 bar, but we then found that the popularity of the held-out targets
predicts the same quantity three times better, at −0.696, which makes the gate ill-posed rather than
failed. And C3, the human round-trip, has never been run. It is the single thing that blocks the
strongest claims across all three papers and I would like to talk about whether it is worth doing.

G10 is the one that matters for the rest of the talk: it certifies that the ruler can tell an oracle
from a random selector from an adversary, which is what makes a Paper B null attributable.

SOURCES
Gate definitions and thresholds: docs/design/GATE_BATTERY_INSTRUMENT.md (G0 :15, G1 :20, G2 :25,
G3 :32, G4 :36, G5 :45, G6 :73, G8 :77, G9 :80, G10 :84, C1 :94, C2 :96, C3 :106).
PASS/FAIL summary: chapterA_v2.tex tab:batteryresults (:1472-1478), :355.
G1 numbers: G1C_CONTROL_RESULT.md :12-45. G5 pop-projection +0.0087: CONCEPT_CHANNEL_RESULT.md
:164-167. Graded − binarised: chapterA_v2.tex :1438-1441. Sign flip −0.2012: :1160.
Behavioural-vs-stated ~21 %: GATE_BATTERY_INSTRUMENT.md :96.""")


# --- A4: the baseline bank -------------------------------------------
s, y = slide(prs, "Fifteen recommenders on one ruler",
             "The baseline bank, all rebuilt in-house and scored under identical conditions. "
             "The tower sits inside the frontier band, not above it.",
             kicker="paper a · replication", accent=ACCENT)

rows = [["System", "LIANG full", "LIANG tail", "NATIVE full", "NATIVE tail", "NDCG@100"]]
for r in [
    ("RecVAE (Shenbin 2020)", ".3540", ".2497", ".4273", ".3051", ".4537"),
    ("CASPER tower — ours (i26c ep31)", ".3496", ".2491", ".4244", ".3017", ".4492"),
    ("EASE (Steck 2019)", ".3476", ".2441", ".4135", ".2941", ".4375"),
    ("EDLAE (Steck 2020)", ".3433", ".2395", ".4110", ".2893", ".4330"),
    ("Mult-VAE (Liang 2018)", ".3202", ".2194", ".3904", ".2682", ".4303"),
    ("user-kNN (Golbandi node model)", ".3065", ".1895", ".3756", ".2377", ".3958"),
    ("RBMF (Zhou 2011)", ".2915", ".2071", ".3616", ".2572", ".3533"),
    ("iALS / PureSVD", ".2703", ".2150", ".3495", ".2566", ".3700"),
    ("TaNP (Lin 2021)", ".2645", ".1635", ".3324", ".2078", ".3661"),
    ("Turbo-CF (Park 2024)", ".2622", ".1621", ".3285", ".2034", ".3569"),
    ("Item-kNN", ".2428", ".1297", ".2946", ".1593", ".3200"),
    ("conjugate belief accumulator", ".2267", ".1300", ".2788", ".1616", ".2883"),
    ("EDDI Partial-VAE (Ma 2019)", ".2153", ".1077", ".2718", ".1383", ".3110"),
    ("SASRec (Kang 2018)", ".1486", ".0805", ".1842", ".0986", ".2465"),
    ("Most-Popular", ".1345", ".0226", ".1626", ".0262", ".1975"),
]:
    rows.append(list(r))
table(s, rows, MARGIN_L, y + Inches(0.02), CONTENT_W, col_w=[38, 12, 12, 12, 12, 14],
      font_size=10, row_h=0.275, align=num(6), first_col_bold=True)

textbox(s, MARGIN_L, Inches(6.42), CONTENT_W, Inches(0.7),
        ["The frontier is a band, not a ranking.",
         ("RecVAE − EASE is +0.0064 full ≈ 1.8 unpaired SE. The uniformity cost is the real "
          "result: the closed-form conjugate belief accumulator — the thing most elicitation "
          "papers actually use as their consumer — forfeits ~0.12 full NDCG against a learned "
          "fold.", {"size": 12, "bold": False, "color": MUTED})],
        size=13.5, bold=True, color=INK, space_after=3, line_spacing=1.12)
footer(s, "Sources: chapterA_v2.tex :514-545 (master table), :1400-1404 (@100), :880-882, :1414-1415")
notes(s, """WHAT TO SAY
This is the work-volume slide for Paper A: fifteen recommenders, every one rebuilt here and scored
under identical conditions, on two evaluation protocols plus NDCG@100.

Read it as a band, not a league table. RecVAE to EASE spans 0.0064 full, about 1.8 unpaired standard
errors — the top four are the same system as far as this measurement is concerned. Our tower sits
second and is statistically tied with RecVAE on tail.

The row I care about is fourth from the bottom: the conjugate belief accumulator at 0.2267. That is
the closed-form Gaussian updater that most elicitation papers use as their consumer, and it forfeits
about 0.12 full NDCG against a learned fold-in on identical evidence. So a paper that reports an
elicitation gain on a conjugate consumer is measuring in a regime where the consumer is throwing away
more than any interview can add back.

SOURCES
Master table both protocol columns: chapterA_v2.tex :514-545. NDCG@100 column: :1400-1404.
RecVAE − EASE band: :1414-1415. Uniformity cost: :880-882 (note :803 says 0.15 — an internal
inconsistency in the chapter worth resolving; quote 'roughly 0.12' from the results section).""")


# --- A5: the certified pair -------------------------------------------
s, y = slide(prs, "The certified pair, and what grading buys",
             "The tower ties RecVAE on tail and gives up 1.2 % relative on full — and the gap is "
             "paid back by using rating levels rather than binary likes.",
             kicker="paper a · tower results", accent=ACCENT)

lw = Inches(7.0)
table(s, [
    ["Comparison (10,000 TEST users, paired bootstrap)", "full", "tail"],
    ["RecVAE, our run, same ruler", "0.3540", "0.2497"],
    ["CASPER tower, i26c ep31 (certified)", "0.3496", "0.2491"],
    ["paired difference", "−0.0043", "−0.0006"],
    ["95 % CI", "[−0.0059, −0.0028]", "[−0.0020, +0.0008]"],
    ["reading", "1.2 % relative", "indistinguishable"],
], MARGIN_L, y + Inches(0.05), lw, col_w=[46, 27, 27],
    font_size=11.5, row_h=0.315, align=num(3), first_col_bold=True)

table(s, [
    ["The intensity premium — identical support, only the values change", "full", "tail"],
    ["graded fold-in (rating levels)", "0.3496", "0.2491"],
    ["binarised fold-in (likes only)", "0.3457", "0.2471"],
    ["premium", "+0.0039", "+0.0020"],
    ["95 % CI", "[+0.0025, +0.0053]", "[+0.0007, +0.0031]"],
], MARGIN_L, Inches(3.35), lw, col_w=[46, 27, 27],
    font_size=11.5, row_h=0.315, align=num(3), first_col_bold=True)

rx = MARGIN_L + lw + Inches(0.40)
rw = CONTENT_W - lw - Inches(0.40)
textbox(s, rx, y + Inches(0.05), rw, Inches(0.3), "WHAT THIS BUYS", size=11, bold=True, color=ACCENT)
bullets(s, rx, y + Inches(0.42), rw, [
    ("Parity is the point.", "the instrument is at the frontier, so a Paper B "
     "null cannot be dismissed as a weak recommender."),
    ("Signed, graded evidence is load-bearing.", "on binary information this tower "
     "would score 0.3457, i.e. 0.0083 behind RecVAE. Levels recover about half of that."),
    ("Checkpoint discipline.", "epoch 23 actually scores higher on TEST (0.3505) — "
     "we keep epoch 31 because it wins the pre-registered validation rule. Selecting "
     "on TEST is how instruments stop being instruments."),
], size=12.5)
footer(s, "Sources: chapterA_v2.tex :861, :1411-1413, :1426-1443 · "
          "docs/results/PAPER_A_STATE_2026-08-04.md :18-37")
notes(s, """WHAT TO SAY
The certified pair: tower i26c epoch 31 plus concept operator cv3 epoch 8, both frozen from here on.

Against RecVAE on the same ruler we tie on tail — the interval contains zero — and give up 0.0043 on
full, 1.2 % relative, with the interval excluding zero. I report the shortfall rather than rounding
it away, because the whole argument of the paper is that a certified instrument states its own
imperfections.

The second table is the reason the shortfall is acceptable. Our fold-in uses signed, graded ratings;
RecVAE's protocol uses binary likes. If we binarise our own input, holding the support identical, we
score 0.3457 — which would put us 0.0083 behind. Using the levels recovers about half of that.
So the tower is not a worse RecVAE; it is a recommender that reads a richer observation, and the
richer observation is exactly what an interview produces.

Last bullet is a discipline point I would defend: epoch 23 scores higher on TEST. We do not use it,
because the checkpoint was selected on validation under a rule written in advance.

SOURCES
Tower and RecVAE numbers, paired diffs and CIs: chapterA_v2.tex :861, :1411-1413, :1426-1430.
Intensity premium: :1438-1441; binary-information comparison :1441-1443.
Certified artefacts and epoch-23 note: docs/results/PAPER_A_STATE_2026-08-04.md :14-31.""")


# --- A6: the concept channel -------------------------------------------
s, y = slide(prs, "The concept channel: behaviour beats what an LLM says",
             "Asking about a tag needs an affinity estimate. Implicit watch behaviour carries "
             "roughly 22× the signal of an LLM's stated judgement.",
             kicker="paper a · concepts", accent=ACCENT)

lw = Inches(6.6)
table(s, [
    ["Signal source", "R² against oracle concept affinity"],
    ["SEL — signed, exposure-corrected implicit affinity (PMI)", "0.376"],
    ["SEL + VAL — plus the shrunk signed rating residual (λ = 3)", "0.398   (VAL adds +0.042)"],
    ["LLM-elicited ordinal judgement", "0.017"],
], MARGIN_L, y + Inches(0.05), lw, col_w=[62, 38],
    font_size=12, row_h=0.34, align=num(2), first_col_bold=True)

table(s, [
    ["The signed triple gate (retrained operator)", "value"],
    ["signed − clip-up at q16", "+0.01043  [+0.00753, +0.01336]"],
    ["popularity counterfeit (control) — declines", "−0.04520"],
    ["value-permuted contrast (control)", "+0.03353  [+0.03095, +0.03607]"],
    ["concept sign-flip on its own operator", "−0.2012, interval clean"],
], MARGIN_L, Inches(3.30), lw, col_w=[62, 38],
    font_size=12, row_h=0.32, align=num(2), first_col_bold=True)

rx = MARGIN_L + lw + Inches(0.40)
rw = CONTENT_W - lw - Inches(0.40)
textbox(s, rx, y + Inches(0.05), rw, Inches(0.3),
        "WHY CONCEPTS AT ALL", size=11, bold=True, color=ACCENT)
table(s, [
    ["Ranked by polarisation, q16", "full", "tail"],
    ["concept − item", "+0.0146", "+0.0265"],
    ["95 % CI", "[+.0094,+.0197]", "[+.0204,+.0325]"],
    ["peak, at q8", "+0.0189", "+0.0310"],
], rx, y + Inches(0.42), rw, col_w=[42, 29, 29],
    font_size=10.5, row_h=0.30, align=num(3), first_col_bold=True)
textbox(s, rx, Inches(3.05), rw, Inches(2.4),
        ["THE MECHANISM IS ANSWERABILITY",
         ("A polarising item bank is answered 4.8 % of the time. A polarising concept bank is "
          "answered 97.2 %.", {"size": 11.5, "bold": False, "color": INK}),
         ("", {"size": 5}),
         ("Protocol-free: a held-out user has rated 25.2 % of the top-100 films and 13.1 % of the "
          "top-500 — but can answer 53.0 % of the 1,031 concepts.",
          {"size": 11.5, "bold": False, "color": MUTED})],
        size=11, bold=True, color=ACCENT, space_after=4, line_spacing=1.15)
footer(s, "Sources: chapterA_v2.tex :1080-1084, :1181-1190, :1218-1231, :1288-1291 · "
          "CONCEPT_CHANNEL_RESULT.md :20-22, :84-101")
notes(s, """WHAT TO SAY
Two results here, and the second is the one that propagates all the way to Paper C.

First: to ask 'do you like noir films?' you need a per-user affinity for the tag. We decomposed where
that signal lives. Implicit watch behaviour — a signed, exposure-corrected PMI we call SEL — explains
0.376 of the variance in the oracle affinity. Adding the rating residual gets you to 0.398. Asking an
LLM to rate the user's affinity on an ordinal scale explains 0.017. That is roughly twenty-two times
worse, and it is why the LLM-derived answer apparatus was retired from this project in July.

Second, and this is the load-bearing one: concepts beat items as interview questions, by +0.0146 full
and +0.0265 tail at sixteen questions. The mechanism is not that concepts are more informative — they
are less informative per answer. It is that they get answered. A polarising item bank is answered
4.8 % of the time; a polarising concept bank 97.2 %. That single asymmetry drives Paper B's Act 4.

SOURCES
Decomposition R²: chapterA_v2.tex :1080-1084; CONCEPT_CHANNEL_RESULT.md :20-22.
Signed triple gate: CONCEPT_CHANNEL_RESULT.md :84-101; chapterA_v2.tex :1181-1190.
POLAR concept − item: chapterA_v2.tex :1218-1224. Answer rates 4.8 % / 97.2 %: :1228-1231.
Protocol-free answerability 25.2 / 13.1 / 53.0 %: :1288-1291. Sign flip: :1160.
LLM apparatus retirement: CLAUDE.md 'Settled'; memory answerability-audit-2026-07-22.""")


# --- A7: the fold-in machinery ----------------------------------------
s, y = slide(prs, "The fold-in machinery",
             "Evidence in, latent point out — one operator per channel, no gradient step at "
             "inference, defined from the very first question.",
             kicker="paper a · fold-in", accent=ACCENT)

bwf, bhf = Inches(2.7), Inches(1.05)
gyf = y + Inches(0.30)
xf = [MARGIN_L + Inches(0.25) + i * Inches(3.15) for i in range(4)]
for i, (h, sub, col) in enumerate([
        ("EVIDENCE SET", "{(φⱼ, yⱼ)} — a direction and a signed, graded value", MUTED),
        ("SET ENCODER", "permutation-invariant; items use the tower's own encoder", ACCENT),
        ("LATENT POINT μₜ", "one point in the recommender's own space", ACCENT),
        ("FROZEN DECODER", "sᵢ = Qᵢᵀμₜ + βᵢ over all 18,359 items", ACCENT_C)]):
    box(s, xf[i], gyf, bwf, bhf,
        [(h, {"size": 11.5}), (sub, {"size": 9.5, "bold": False, "color": MUTED})], line=col)
    if i < 3:
        arrow(s, xf[i] + bwf, gyf + bhf // 2, xf[i + 1], gyf + bhf // 2, color=MUTED)

table(s, [
    ["Design decision", "Consequence, measured"],
    ["φ is channel-agnostic: an item factor, a whitened concept centroid, a text-adapter "
     "image, or a raw policy direction all enter the same slot",
     "one code path serves items, concepts, free text and Paper C's synthesised directions"],
    ["Concepts get their own trained gated fold-to-point, zero-initialised",
     "empty interview is bit-identical to the bare tower ⇒ the identity gate is inherited free"],
    ["Concept directions are whitened before folding",
     "raw centroid member-AUC 0.44 (below chance) → whitened 0.94"],
    ["The gate replaces additive union",
     "additive union reaches 0.0991 full at 8 answers — below the 0.1279 intercept, i.e. worse "
     "than asking nothing. The gated fold reaches 0.1759 / 0.0841."],
    ["Redundancy robustness is enforced",
     "8 correlated top-SEL concepts (0.1685/0.0906) must not lose to 4 (0.1572/0.0777) — the "
     "additive operator failed exactly here"],
    ["No gradient step at inference; defined for every t ≥ 0",
     "a policy can be scored at turn 0, 1 and 16 on the same object"],
], MARGIN_L, gyf + bhf + Inches(0.35), CONTENT_W, col_w=[44, 56],
    font_size=10.8, row_h=0.44, first_col_bold=True)
footer(s, "Sources: chapterA_v2.tex :243-249, :865-882, :1094-1116, :305-308")
notes(s, """WHAT TO SAY
The fold-in operator is the piece both later papers actually consume, so it is worth two minutes.

The contract: an evidence set of direction–value pairs goes in, a single latent point comes out, and
the frozen decoder scores the whole catalogue from it. No gradient step at inference — that matters
because it means we can evaluate at turn zero and turn sixteen on the same object, and because an
interview at serving time must be cheap.

The two rows I would defend hardest. Whitening: a raw concept centroid has member-AUC 0.44 — below
chance — because the space is dominated by one shared component. Whitened, it is 0.94. Same data,
different geometry. And the gate: the obvious operator, additively unioning the concept directions,
actually scores below the no-question intercept after eight answers. It is worse than asking nothing.
That failure is why there is a learned gate rather than a formula, and it is the reason the
redundancy-robustness clause is a gate rather than a nice-to-have.

SOURCES
Operator definition: chapterA_v2.tex :243-249, :865-882 (Eq. eq:obs).
Gated fold and zero-init: :1109-1116, :1309-1315. Whitening 0.44 → 0.94: :1094-1097.
Additive-union 0.0991 vs 0.1279 intercept, gated 0.1759/0.0841, redundancy rows: :305-308, :1104-1116.
Redundancy clause as gate G5-E(e): docs/design/GATE_BATTERY_INSTRUMENT.md :51.""")


# --- A8: what A licenses + status ---------------------------------------
s, y = slide(prs, "What Paper A licenses — and where it stands",
             "A's job is to make the gap measurable. Closing it belongs to B and C.",
             kicker="paper a · hand-off and status", accent=ACCENT)

lw = Inches(7.3)
table(s, [
    ["Licensed for downstream use", "Because"],
    ["One frozen consumer for every strategy comparison",
     "G10: oracle > random > adversarial-worst, each interval clean on both axes"],
    ["A fold-in operator that ingests any channel",
     "channel-agnostic φ; the identity gate inherited by zero-init"],
    ["Concept questions as first-class actions",
     "G5 + G5-E; concepts beat items by +0.0146 full / +0.0265 tail at q16"],
    ["A continuous direction interface",
     "raw directions enter the same slot — this is exactly what Paper C emits"],
    ["Attributable nulls",
     "a Paper B null is a statement about the policy, not about the recommender"],
], MARGIN_L, y + Inches(0.05), lw, col_w=[42, 58],
    font_size=11.2, row_h=0.42, first_col_bold=True)

rx = MARGIN_L + lw + Inches(0.40)
rw = CONTENT_W - lw - Inches(0.40)
textbox(s, rx, y + Inches(0.05), rw, Inches(0.3), "STATUS", size=11, bold=True, color=ACCENT)
table(s, [
    ["", ""],
    ["State", "closed, 41 pp, not yet submitted"],
    ["Target", "TORS (ACM Trans. on Rec. Sys.)"],
    ["Why", "rolling — no deadline; no travel; length is an asset"],
    ["History", "ECIR rejected it: too long, instrument without application"],
], rx, y + Inches(0.42), rw, col_w=[26, 74], font_size=10.8, row_h=0.36,
    first_col_bold=True, header_rows=0)

textbox(s, rx, Inches(4.05), rw, Inches(2.3),
        ["THE BLIND CALIBRATION PANEL (2026-07-29)",
         ("Three anonymised drafts, one hostile reviewer, identical rubric: PEBOL, Bıyık's soft "
          "attributes, and ours. All three: weak reject overall, reject at SIGIR/RecSys, "
          "ACCEPT at TORS.", {"size": 11, "bold": False, "color": INK}),
         ("", {"size": 5}),
         ("Ours is the only one accepted on the reproducibility track — the published-number "
          "bridge was called “exemplary”. All three top objections were simulator "
          "circularity.", {"size": 11, "bold": False, "color": MUTED})],
        size=11, bold=True, color=ACCENT, space_after=4, line_spacing=1.14)
footer(s, "Sources: docs/PUBLICATION_PLAN.md :12-55, :74 · GATE_BATTERY_INSTRUMENT.md :86-88 · "
          "chapterA_v2.tex :1306-1307")
notes(s, """WHAT TO SAY
What A hands forward, and where it stands.

The hand-off list on the left is the practical answer to 'why did you spend a year on a recommender
in an elicitation thesis'. Every row is something B or C would otherwise have had to assume.

On status: the paper is closed at 41 pages and unsubmitted. Target is TORS, chosen for three
reasons — rolling submission so nothing forces the timeline, no travel, and length is unremarkable
there whereas ECIR rejected it partly for being 30 pages.

The panel at the bottom is worth a minute because it calibrated my expectations. I anonymised three
drafts — PEBOL, Bıyık's soft-attributes paper, and mine — gave a hostile reviewer one rubric, and
had it grade all three blind. Identical verdicts to two peer-reviewed papers: weak reject overall,
reject at the top conferences, accept at TORS. So we are at the field's level, not below it. The one
place we uniquely win is reproducibility. The one place we uniquely lose is ECIR, on length. And all
three papers' top objection was the same thing: simulator circularity. That is the field-wide attack,
and it is what C3 — the unrun human study — would answer.

SOURCES
Licensing statements: chapterA_v2.tex :76, :249, :929, :1306-1307, :1517-1518;
GATE_BATTERY_INSTRUMENT.md :86-88.
Concept − item at q16: chapterA_v2.tex :1218-1222.
Status, TORS rationale, panel table: docs/PUBLICATION_PLAN.md :12-34, :36-55, :74.""")

# =====================================================================
# PAPER B — WHAT SHOULD A COLD INTERVIEW OPTIMISE
# =====================================================================

sd = divider("Paper B — the interview",
             "Twenty-one selection rules from thirteen published methods, all rebuilt on one "
             "ruler — and then one variable changed.",
             ACCENT_B, "14 slides  ·  ~20 minutes  ·  status: chapter closed → ECIR 2027, 2 Oct 2026")
notes(sd, """WHAT TO SAY
Section break. This is the substantial workstream and the one going to a conference. The structure
is: audit the field on one ruler, diagnose why the field's methods underperform, then change exactly
one variable and show it was the diagnosis.

SOURCES
docs/design/PAPER_B_MASTER_TODO.md ('CHAPTER CLOSED 2026-08-30'); docs/PUBLICATION_PLAN.md roster.""")


# --- B1: the question -------------------------------------------------
s, y = slide(prs, "Paper B's question",
             "Every published selector optimises a surrogate — variance, entropy, information gain, "
             "a pairwise loss. Almost none optimises the number the field reports.",
             kicker="paper b · the question", accent=ACCENT_B)

lw = Inches(6.5)
textbox(s, MARGIN_L, y + Inches(0.05), lw, Inches(0.3),
        "THE OBSERVATION THAT STARTED IT", size=11, bold=True, color=ACCENT_B)
bullets(s, MARGIN_L, y + Inches(0.42), lw, [
    ("1 of 21 distinct selection rules", "optimises the quantity the field actually "
     "reports — ranking quality on the recommender that consumes the answers."),
    ("12 of 23 battery rows are value-free", "— they choose a question without "
     "reference to how much better the recommendations get."),
    ("So the question is:", "if you build the selector that optimises the reported "
     "metric directly, what changes — and by how much?"),
    ("And the harder question:", "is the shortfall the criterion, or the action "
     "space, or answerability?"),
], size=13)

rx = MARGIN_L + lw + Inches(0.45)
rw = CONTENT_W - lw - Inches(0.45)
textbox(s, rx, y + Inches(0.05), rw, Inches(0.3),
        "THE FIVE ACTS", size=11, bold=True, color=ACCENT_B)
table(s, [
    ["Act", "What it establishes"],
    ["1  the audit", "adaptive selectors lose to a good static order — 30/30 comparisons, "
     "and 138 cells when widened"],
    ["2  the diagnosis", "the surrogates do not track the metric; and cold users are "
     "nearly identical, so there is little to adapt to"],
    ["3  the demonstration", "change one variable — the split criterion — and the "
     "tree beats the static order on TEST"],
    ["4  answerability", "which channel wins is decided by who can answer, not by "
     "who is informative"],
    ["5  the ecological arm", "what real users volunteer unprompted, and one policy "
     "that uses it"],
], rx, y + Inches(0.42), rw, col_w=[24, 76], font_size=11, row_h=0.50, first_col_bold=True)

textbox(s, MARGIN_L, Inches(5.65), lw, Inches(1.3),
        ["THE CONTRACT, FIXED THROUGHOUT",
         ("Consumer = the certified tower i26c ep31. Protocol candidate pool: every rated "
          "fold-in item removed. Behavioural answers. All 10,000 TEST users. Nothing sampled. "
          "Answer-free reference 0.1626 / 0.0262.",
          {"size": 11.5, "bold": False, "color": MUTED})],
        size=11, bold=True, color=ACCENT_B, space_after=4, line_spacing=1.14)
footer(s, "Sources: chapterB_v2.tex :63, :93, :396, :424 · docs/results/BASELINE_OBJECTIVE_AUDIT.md")
notes(s, """WHAT TO SAY
Paper B's question in one observation. We went through the field's selection rules and asked what
each one actually optimises. Of twenty-one distinct rules, exactly one optimises the quantity the
papers report — ranking quality on the recommender that will consume the answers. Twelve of the
twenty-three battery rows are value-free: they pick a question without any reference to whether the
recommendations get better.

That is not a criticism of anyone's execution, it is a structural observation, and it suggests an
obvious experiment: build the selector that optimises the reported metric directly and see what
happens. That is Act 3. Acts 1 and 2 exist so that when Act 3 works we know why.

SOURCES
Census: chapterB_v2.tex :63, :93, :403, :1040 ('twenty-one distinct selection rules'), :396
('twelve of twenty-three rows are value-free'), :424 ('thirteen published methods');
docs/results/BASELINE_OBJECTIVE_AUDIT.md corroborates all three counts.
Contract and answer-free reference 0.1626/0.0262: chapterB_v2.tex tab:battery caption.""")


# --- B2: the battery roster --------------------------------------------
s, y = slide(prs, "The replication battery — what was rebuilt",
             "Twenty-three selector rows, twenty-one distinct rules, thirteen published methods. "
             "Every one re-implemented here and run on the same ruler, same users, same pool.",
             kicker="paper b · the work", accent=ACCENT_B)

table(s, [
    ["Family", "Rules rebuilt", "n"],
    ["Optimised static (our comparator)",
     "greedy combined · greedy items-only · greedy concepts-only", "3"],
    ["Literature-anchored heuristics",
     "entropy₀ (Rashid 2008) · popularity (Rashid 2002) · HELF (Rashid 2008) · "
     "pure entropy (Rashid 2002)", "4"],
    ["Random controls", "random over the bank · random over the catalogue", "2"],
    ["Bandit elicitors",
     "GLM-UCB (Zhao 2013) · Thompson (Zhao 2013) · ConUCB (Zhang, WWW'20) · "
     "ConLinUCB-BS-static (Wang 2023) · CLiME (Liu, KDD'25)", "5"],
    ["Information-theoretic",
     "latent-only info-gain (Ma 2019, EDDI) · EDDI three-outcome · BALD (Houlsby 2011) · "
     "A-optimal (Anava WWW'15 / Sepliarskaia 2018)", "4"],
    ["Tree and node models",
     "Golbandi decision tree (WSDM'11) · IGCN (Rashid 2008)", "2"],
    ["Learned / discriminative",
     "DRE (Kweon, WWW'20) · DPE-Hellinger and DPE-ILD (Parapar, WSDM'21)", "3"],
    ["Ceiling", "clairvoyant oracle — reads the user's own ratings at selection time; "
     "never a baseline", "1"],
], MARGIN_L, y + Inches(0.05), CONTENT_W, col_w=[24, 66, 6],
    font_size=11, row_h=0.475, align=[L, L, C], first_col_bold=True)

textbox(s, MARGIN_L, Inches(5.75), CONTENT_W, Inches(1.3),
        ["Two things this table is doing at once.",
         ("(1) It is the audit — one ruler, so the comparison is real. "
          "(2) It is the honesty ledger: the chapter also prints a coverage table of what was "
          "NOT built — ICF's four policies, SEL-BALD, PEBOL's acquisition, fMF, PERE, Greedy "
          "SLIM/QBandit, RBMF/RMVA, GreedyExtend, FacT-CRS, UpsRec, MeLU, a-stratification — "
          "each marked ✗ built, ✗ ours.",
          {"size": 12, "bold": False, "color": MUTED})],
        size=13.5, bold=True, color=INK, space_after=3, line_spacing=1.14)
footer(s, "Sources: chapterB_v2.tex tab:battery (:601-656), tab:coverage (:724-769)")
notes(s, """WHAT TO SAY
This is the slide I most want you to look at, because it is the volume of work. Twenty-three selector
rows covering twenty-one distinct rules from thirteen published methods, spanning 2002 to 2025 —
Rashid's entropy heuristics through to CLiME at KDD this year. Every one re-implemented here, not
cited, and run on identical users, an identical candidate pool, and an identical consumer.

Where a method had a native protocol we also validated it natively before porting — Golbandi, IGCN,
DRE, DPE, Hu and Yu — so a port that underperforms is a port we can defend.

And I want to flag the second table in the chapter, which is the list of what we did NOT build:
twelve more systems, each marked not-built and not-ours. I would rather print that than let a
reviewer discover it.

SOURCES
Roster and citations: chapterB_v2.tex tab:battery :601-656.
Coverage / not-built table: tab:coverage :724-769.
Counts: :63, :396, :424; BASELINE_OBJECTIVE_AUDIT.md.""")


# --- B3: battery measured, part 1 ---------------------------------------
s, y = slide(prs, "The battery measured (1/2): static orders and the classics",
             "Full / tail NDCG@10, 10,000 TEST users, protocol pool, no sampling. "
             "The optimised static order leads everything.",
             kicker="paper b · results", accent=ACCENT_B)

rows = [["selector", "type", "k=1", "k=2", "k=4", "k=8", "k=16", "ans@16"]]
for r in [
    ("greedy combined (item+concept)", "static", ".1808/.0537", ".1962/.0701",
     ".2178/.0857", ".2386/.1033", ".2590/.1237", "1.5i/9.2c"),
    ("greedy items-only", "static", ".1760/.0434", ".1884/.0563", ".2038/.0677",
     ".2235/.0860", ".2447/.1017", "3.11"),
    ("greedy concepts-only", "static", ".1808/.0537", ".1962/.0701", ".2141/.0858",
     ".2347/.1074", ".2447/.1161", "12.36"),
    ("entropy₀  (Rashid 2008)", "static", ".1635/.0433", ".1706/.0528", ".1831/.0636",
     ".2053/.0792", ".2303/.0924", "5.52"),
    ("popularity  (Rashid 2002)", "static", ".1606/.0400", ".1647/.0470", ".1746/.0579",
     ".1943/.0686", ".2228/.0848", "5.42"),
    ("HELF  (Rashid 2008)", "static", ".1603/.0354", ".1628/.0420", ".1755/.0588",
     ".1892/.0725", ".2157/.0838", "3.34"),
    ("pure entropy  (Rashid 2002)", "static", ".1582/.0319", ".1578/.0327", ".1580/.0323",
     ".1586/.0317", ".1599/.0357", "0.08"),
    ("random (bank)", "static", ".1586/.0310", ".1588/.0311", ".1588/.0316",
     ".1594/.0329", ".1607/.0353", "0.12"),
    ("random (catalogue)", "static", ".1585/.0310", ".1588/.0310", ".1589/.0318",
     ".1593/.0315", ".1612/.0341", "0.12"),
    ("GLM-UCB  (Zhao 2013)", "adaptive", ".1606/.0400", ".1635/.0452", ".1756/.0578",
     ".1902/.0673", ".2161/.0833", "5.24"),
    ("Thompson  (Zhao 2013)", "adaptive", ".1627/.0376", ".1676/.0437", ".1748/.0551",
     ".1908/.0677", ".2135/.0878", "3.39"),
    ("latent-only info-gain†  (Ma 2019)", "adaptive", ".1537/.0336", ".1539/.0373",
     ".1618/.0432", ".1747/.0611", ".1917/.0749", "3.84"),
    ("EDDI three-outcome‡  (Ma 2019)", "adaptive", ".1607/.0363", ".1597/.0388",
     ".1732/.0497", ".1869/.0660", ".2041/.0834", "2.86"),
]:
    rows.append(list(r))
table(s, rows, MARGIN_L, y + Inches(0.02), CONTENT_W, col_w=[27, 10, 12.6, 12.6, 12.6, 12.6, 12.6, 10],
      font_size=9.6, row_h=0.30, align=[L, L, R, R, R, R, R, R], first_col_bold=True)

textbox(s, MARGIN_L, Inches(6.42), CONTENT_W, Inches(0.7),
        ["Greedy items-only at k=16 beats the best adaptive selector by +0.0281 full "
         "[+0.0247, +0.0314] and +0.0178 tail [+0.0145, +0.0210].",
         ("† reduced (not published) EDDI acquisition.   ‡ the same reduced acquisition under a "
          "three-outcome response model.   ans@16 = answers actually collected in sixteen "
          "questions.", {"size": 10, "bold": False, "color": MUTED})],
        size=13, bold=True, color=INK, space_after=3, line_spacing=1.12)
footer(s, "Source: chapterB_v2.tex tab:battery (:601-656), :310-322")
notes(s, """WHAT TO SAY
First half of the battery. Read the top three rows as the comparator we built — a greedy static
order optimised directly against measured NDCG — and everything below as the field.

The headline is uncomfortable for the field and I will not soften it: at sixteen questions a fixed,
non-adaptive item order beats the best adaptive selector by 0.0281 full, interval clear of zero. The
adaptive machinery is not buying anything here.

Look at the ans@16 column, because it is the whole explanation and it is Act 4's setup. Pure entropy
collects 0.08 answers in sixteen questions — it asks about films nobody has seen. Random collects
0.12. Concepts-only collects 12.36. The selectors that look clever are asking unanswerable questions.

SOURCES
chapterB_v2.tex tab:battery :601-656 (contract in the caption: certified tower, protocol pool,
NO-FLOOR, behavioural answers, all 10,000 TEST users, no sampling).
Headline gap +0.0281 / +0.0178: :310-322.
Caveat to keep in mind if pushed: ACT1_PAIRED_CI.md records that glm_ucb and thompson cells are not
seed-stable (iALS basis re-drawn without random_state, drift up to ±0.0026) — quote those two rows
to three decimals, not four.""")


# --- B4: battery measured, part 2 ---------------------------------------
s, y = slide(prs, "The battery measured (2/2): published methods, natively validated then ported",
             "Every one of these was reproduced on its own protocol first, then ported onto our "
             "ruler. None reaches the optimised static order.",
             kicker="paper b · results", accent=ACCENT_B)

rows = [["selector", "type", "k=1", "k=2", "k=4", "k=8", "k=16", "ans@16"]]
for r in [
    ("Golbandi tree  (WSDM'11)", "adaptive", ".1557/.0351", ".1614/.0478", ".1864/.0650",
     ".2117/.0844", "—", "2.75"),
    ("IGCN  (Rashid 2008)", "adaptive", ".1685/.0407", ".1800/.0569", ".1944/.0701",
     ".2031/.0747", ".2239/.0912", "4.19"),
    ("DRE  (Kweon, WWW'20)", "learned static", ".1658/.0396", ".1693/.0465", ".1763/.0586",
     ".2074/.0789", ".2185/.0898", "4.6"),
    ("CLiME  (Liu, KDD'25)", "adaptive", ".1606/.0400", ".1634/.0452", ".1776/.0549",
     ".1914/.0701", ".2218/.0897", "4.80"),
    ("ConLinUCB-BS-static  (Wang 2023)", "static, concepts", ".1644/.0374", ".1659/.0402",
     ".1724/.0468", ".1829/.0598", ".1891/.0700", "0.00"),
    ("ConUCB  (Zhang, WWW'20)", "adaptive", ".1584/.0346", ".1591/.0364", ".1651/.0447",
     ".1805/.0574", ".1874/.0699", "1.80"),
    ("DPE-Hellinger  (Parapar, WSDM'21)", "adaptive", ".1627/.0398", ".1680/.0480",
     ".1817/.0610", ".2002/.0765", ".2192/.0925", "3.9"),
    ("DPE-ILD  (Parapar, WSDM'21)", "adaptive", ".1627/.0398", ".1621/.0401", ".1613/.0397",
     ".1609/.0390", ".1603/.0375", "0.32"),
    ("BALD  (Houlsby 2011) §", "value-free", ".1754/.0448", ".1853/.0473", ".1934/.0572",
     ".2057/.0683", ".2203/.0824", "2.80"),
    ("A-optimal  (Anava WWW'15) §", "value-free", ".1693/.0498", ".1724/.0523", ".1702/.0549",
     ".1694/.0544", ".1740/.0563", "1.81"),
    ("clairvoyant oracle — CEILING", "not a baseline", ".1893/.0680", ".2194/.0965",
     ".2544/.1281", ".2930/.1597", ".3292/.1927", "12.9"),
]:
    rows.append(list(r))
table(s, rows, MARGIN_L, y + Inches(0.02), CONTENT_W,
      col_w=[27, 12, 12.2, 12.2, 12.2, 12.2, 12.2, 9],
      font_size=9.4, row_h=0.275, align=[L, L, R, R, R, R, R, R], first_col_bold=True)

table(s, [
    ["Separation, paired bootstrap over the same 10,000 users, 10,000 resamples", ""],
    ["Optimised static vs every adaptive arm under this contract (13 adaptive rows + the "
     "three-outcome sub-variant = 138 cells)",
     "combined and concepts-only orders lead and separate in ALL 138"],
    ["items-only order", "leads 136, separates 134 — the 4 exceptions are tail cells that "
     "separate in neither direction"],
    ["DRE (WWW'20) vs the 2008 entropy heuristic, full axis",
     "−0.0118  [−0.0153, −0.0083]  — the learned selector loses to a 2008 heuristic"],
], MARGIN_L, Inches(5.6), CONTENT_W, col_w=[52, 48], font_size=10, row_h=0.30)
footer(s, "Sources: chapterB_v2.tex tab:battery, :658-667, :674-677 · § = value-free rule · "
          "Golbandi has no k=16 row (leaf fragmentation caps depth)")
notes(s, """WHAT TO SAY
Second half: the published methods, ported. Each was validated natively on its own protocol before
being ported, so when a port underperforms we can say it is the method and not our implementation.

Two rows to point at. DPE-ILD and ConLinUCB-BS flatline or collect almost no answers — ConLinUCB
collects exactly zero answers in sixteen questions under our answer model. And DRE, a learned
selector from WWW 2020, loses to Rashid's 2008 entropy heuristic by 0.0118 with the interval clear.

The separation block at the bottom is the audit's actual result. We widened the comparison to every
adaptive arm at every budget on both axes — 138 cells — and the optimised static order leads and
separates in all 138 for two of three orders, and 134 of 138 for the third, with the four exceptions
being tail cells that separate in neither direction. That is about as clean as this gets.

And the oracle row is the ceiling: 0.3292 at sixteen questions if you may read the user's own ratings
while choosing. That is not a baseline, it is the headroom.

SOURCES
chapterB_v2.tex tab:battery :601-656; separation counts :658-667; DRE vs entropy₀ :674-677.
Golbandi's missing k=16: capped by MAX_DEPTH / 3^k leaf fragmentation (chapter, and confirmed in
docs/results/GOLBANDI_PORT_ON_OUR_RULER.md).""")


# --- B5: Act 1 on TEST ---------------------------------------------------
s, y = slide(prs, "Act 1 on TEST: adaptivity does not pay here",
             "Thirty comparisons against one static order. All thirty separate, and all thirty run "
             "the same way. The ordering then repeats on four other consumers.",
             kicker="paper b · act 1", accent=ACCENT_B)

lw = Inches(7.6)
rows = [["selector (10,000 TEST users)", "k=1", "k=2", "k=4", "k=8", "k=16", "ans@16"]]
for r in [
    ("static — entropy₀", ".1635/.0433", ".1706/.0528", ".1831/.0636", ".2053/.0792",
     ".2303/.0924", "5.52"),
    ("GLM-UCB", ".1606/.0400", ".1635/.0452", ".1756/.0578", ".1902/.0673", ".2161/.0833", "5.24"),
    ("Thompson", ".1627/.0376", ".1676/.0437", ".1748/.0551", ".1908/.0677", ".2135/.0878", "3.39"),
    ("latent-only info-gain†", ".1537/.0336", ".1539/.0373", ".1618/.0432", ".1747/.0611",
     ".1917/.0749", "3.84"),
]:
    rows.append(list(r))
table(s, rows, MARGIN_L, y + Inches(0.05), lw, col_w=[26, 13, 13, 13, 13, 13, 9],
      font_size=10.2, row_h=0.33, align=[L, R, R, R, R, R, R], first_col_bold=True)

table(s, [
    ["Full-axis deficit against the static order (k=1 → k=16)", "range"],
    ["GLM-UCB", "−0.0029  →  −0.0137"],
    ["Thompson", "−0.0022  →  −0.0177"],
    ["latent-only info-gain†", "−0.0098  →  −0.0386"],
    ["All 30 comparisons separate (paired bootstrap, 10,000 resamples, "
     "self-comparison exactly zero)", "30 / 30"],
], MARGIN_L, Inches(3.35), lw, col_w=[68, 32], font_size=11, row_h=0.33,
    align=num(2), first_col_bold=True)

rx = MARGIN_L + lw + Inches(0.35)
rw = CONTENT_W - lw - Inches(0.35)
textbox(s, rx, y + Inches(0.05), rw, Inches(0.3),
        "IT IS NOT ONE CONSUMER", size=11, bold=True, color=ACCENT_B)
bullets(s, rx, y + Inches(0.42), rw, [
    ("The ordering repeats", "on RecVAE, TaNP and EDDI's own Partial-VAE."),
    ("Including the self-consistent cell", "— EDDI selects and EDDI consumes: "
     ".1493/.0488 against static's .1769/.0480 at k=16."),
    ("But it is consumer-dependent.", "On a fifth consumer (a Golbandi leaf model) "
     "Thompson beats static at every budget: +0.0100, +0.0171, +0.0148, +0.0281 full, "
     "intervals excluding zero."),
    ("Which is the point of Paper A.", "The verdict is a property of a "
     "(strategy, instrument) pair, and we say which instrument."),
], size=12)
footer(s, "Sources: chapterB_v2.tex tab:act1 (:548-578), :569-572, :932-945 · ACT1_PAIRED_CI.md")
notes(s, """WHAT TO SAY
Act 1 in detail. One static order — Rashid's 2008 entropy heuristic, which is a fair and generous
representative of the field — against three adaptive selectors, at five budgets, on both axes. Thirty
comparisons. All thirty separate in the static order's favour, and the deficit grows with budget:
GLM-UCB goes from −0.0029 at one question to −0.0137 at sixteen.

I then checked it is not an artefact of our consumer: the same ordering appears on RecVAE, on TaNP,
and on EDDI's own Partial-VAE — including the self-consistent cell where EDDI both selects and
consumes, which is the version most favourable to them.

And then the honest part, bottom right. On a fifth consumer, a Golbandi-style leaf model, Thompson
beats the static order at every budget with clean intervals. The verdict flips. That is not a problem
for the paper, it IS the paper's methodological claim, and it is why Paper A exists: an elicitation
result is a property of a strategy–instrument pair, and you have to say which instrument.

SOURCES
tab:act1 and deltas: chapterB_v2.tex :548-578. Four-consumer replication and the EDDI self-consistent
cell: :569-572. The flip: :932-945. Per-cell CIs: docs/results/ACT1_PAIRED_CI.md.""")

# --- B6: Act 2 — the criterion diagnosis --------------------------------
s, y = slide(prs, "Act 2 (i): the surrogates do not track the metric",
             "At all 22 nodes of the grown tree we measured every candidate's true NDCG gain, "
             "then asked what each published criterion would have picked.",
             kicker="paper b · act 2 · diagnosis", accent=ACCENT_B)

table(s, [
    ["split criterion", "mean ρ with measured gain", "nodes ρ>0", "argmax hit",
     "top-10 recall", "regret"],
    ["variance  (Golbandi 2011)", "+0.010", "13 / 22", "1 / 22", "0.03", "+0.01664"],
    ["pairwise rank loss  (Hu & Yu 2013)", "−0.060", "10 / 22", "0 / 22", "0.01", "+0.01738"],
    ["logistic loss, as run  (Hu & Yu)", "−0.104", "9 / 22", "1 / 22", "0.03", "+0.01282"],
    ["control: uniform random pick", "—", "—", "—", "0.10", "+0.01955"],
], MARGIN_L, y + Inches(0.05), CONTENT_W, col_w=[32, 20, 12, 12, 12, 12],
    font_size=11.5, row_h=0.36, align=num(6), first_col_bold=True)

textbox(s, MARGIN_L, Inches(3.55), CONTENT_W, Inches(0.85),
        ["Following any of these three criteria forgoes between 66 % and 89 % of the NDCG@10 that "
         "a uniform random pick would forgo.",
         ("Two of the three have a NEGATIVE rank correlation with the thing they are proxies for. "
          "Top-10 recall of 0.01–0.03 is below the 0.10 a random pick achieves.",
          {"size": 12, "bold": False, "color": MUTED})],
        size=15, bold=True, color=INK, space_after=4, line_spacing=1.14)

lw = Inches(6.5)
textbox(s, MARGIN_L, Inches(4.65), lw, Inches(0.3),
        "AT THE ROOT, CONCRETELY", size=11, bold=True, color=ACCENT_B)
bullets(s, MARGIN_L, Inches(5.0), lw, [
    ("The logistic argmax IS the metric's argmax", "— regret exactly 0."),
    ("Variance ranks that same question 250th of 1,000.", ""),
    ("Pairwise rank loss ranks it 75th.", ""),
], size=12.5)

rx = MARGIN_L + lw + Inches(0.45)
rw = CONTENT_W - lw - Inches(0.45)
textbox(s, rx, Inches(4.65), rw, Inches(0.3),
        "IS IT THE CRITERION OR THE APPROXIMATION?", size=11, bold=True, color=ACCENT_B)
bullets(s, rx, Inches(5.0), rw, [
    ("Measured properly, the pairwise loss is fine:", "at the root it tracks measured "
     "NDCG@10 at Spearman +0.784 and forgoes only +0.00459."),
    ("Hu & Yu's Taylor-approximated training-loss version", "correlates at −0.149 and "
     "forgoes +0.01473. The approximation, not the idea, is what fails."),
], size=12)
footer(s, "Sources: chapterB_v2.tex tab:criterionregret (:1056-1086), :1283-1292")
notes(s, """WHAT TO SAY
Act 2 is the diagnosis, and it has two halves. This is the first: are the surrogate criteria any good?

We took all twenty-two nodes of the grown tree, and at each node we measured the true held-out NDCG
gain of every one of about a thousand candidate questions — that is the expensive object nobody
usually has. Then we asked what each published criterion would have picked.

Two of the three have a negative rank correlation with the quantity they are standing in for.
Top-10 recall is 0.01 to 0.03 against a random pick's 0.10. And the regret column says following any
of them forgoes two-thirds to nine-tenths of what random forgoes — so they are better than random,
but barely, and not in the way their derivations suggest.

The right-hand box is the fair version, and I want it on the slide because it is a defence of Hu and
Yu rather than an attack. If you measure the pairwise rank loss properly on held-out data it tracks
NDCG at +0.784 — it is a good criterion. It is their Taylor-approximated training-loss version that
correlates at −0.149. So the failure is the approximation, not the idea.

SOURCES
tab:criterionregret and the root example: chapterB_v2.tex :1056-1086.
Measured-vs-approximated pairwise loss: :1283-1292.
NOTE: docs/results/ACT2_SURROGATE_RANK_CORRELATION.md is a design sheet with 'NO RESULT IN THIS FILE
YET' — quote the chapter table, not that file.""")


# --- B7: Act 2 (ii) dispersion -------------------------------------------
s, y = slide(prs, "Act 2 (ii): cold users are nearly identical, so there is little to adapt to",
             "How many genuinely different best-next-questions does a population contain? "
             "At cold start: about one.",
             kicker="paper b · act 2 · diagnosis", accent=ACCENT_B)

table(s, [
    ["population state", "n", "distinct best question q★", "effective # questions"],
    ["ML-25M cold, no questions asked", "10,000", "1", "1.00"],
    ["after one standard opener", "10,000", "5", "2.88"],
    ["after two standard openers", "10,000", "9", "5.04"],
    ["real free-text request, placed as a state", "4,431", "442", "95.13"],
    ["   — requests that name at least one film", "2,639", "413", "128.46"],
    ["   — text-only requests, no film named", "1,792", "171", "38.23"],
], MARGIN_L, y + Inches(0.05), CONTENT_W, col_w=[46, 14, 22, 18],
    font_size=12, row_h=0.345, align=num(4), first_col_bold=True)

lw = Inches(6.5)
textbox(s, MARGIN_L, Inches(3.60), lw, Inches(0.3),
        "THE READING", size=11, bold=True, color=ACCENT_B)
bullets(s, MARGIN_L, Inches(3.95), lw, [
    ("An adaptive selector has almost nothing to be adaptive about.", "After two "
     "openers the whole population occupies 112 distinct states and 41 % of users share "
     "a single argmax."),
    ("Concept bank alone:", "1.00 → 2.88 → 5.04.  Item bank (297 questions): "
     "1.00 → 5.48 → 37.08.  Unified bank: 1.00 → 5.04 → 74.90."),
    ("A real volunteered request is worth ~95 effective questions", "of dispersion — "
     "nineteen times what two closed openers buy."),
], size=12.5)

rx = MARGIN_L + lw + Inches(0.45)
rw = CONTENT_W - lw - Inches(0.45)
textbox(s, rx, Inches(3.60), rw, Inches(0.3),
        "BUT THE HEADROOM RUNS THE OTHER WAY", size=11, bold=True, color=ACCENT_B)
table(s, [
    ["state", "best question available", "gain"],
    ["cold, after two openers", "a concept", "+0.0283"],
    ["request-placed", "an item", "+0.0138"],
    ["request-placed", "best concept", "+0.0076"],
], rx, Inches(3.95), rw, col_w=[38, 38, 24], font_size=11, row_h=0.32, align=num(3))
textbox(s, rx, Inches(5.45), rw, Inches(1.3),
        "Dispersion and headroom are different quantities. A placed user is more "
        "distinguishable but has less left to gain — which is exactly why Act 5's opener "
        "arm is worth less than the dispersion table suggests.",
        size=11.5, color=MUTED, line_spacing=1.15)
footer(s, "Sources: chapterB_v2.tex tab:dispersion (:1103-1129), :1107-1108, :1149-1154")
notes(s, """WHAT TO SAY
Second half of the diagnosis, and this is the one that actually explains Act 1.

The question: how many genuinely different best-next-questions does a cold population contain? We
computed, for each user, which question in the bank has the highest measured gain, and then counted
distinct answers and an entropy-based effective count.

At cold start the answer is one. Every user's best first question is the same question. After one
opener, 2.88 effective questions. After two, about five, and 41 % of users still share a single
argmax. So an adaptive selector at a cold state is being clever about a population that is, from the
question bank's point of view, almost homogeneous. That is why Act 1 comes out the way it does — not
because adaptivity is a bad idea, but because there is nothing to adapt to yet.

The contrast row is the reason Act 5 exists: a real free-text request from Reddit places users at
about 95 effective questions of dispersion — nineteen times what two closed openers buy.

But the box on the right stops me overselling that: a placed user is more distinguishable and has
LESS headroom left, +0.0138 against +0.0283. Dispersion is not value.

SOURCES
tab:dispersion: chapterB_v2.tex :1103-1123; 112 states / 41 % argmax share :1127-1129;
item and unified bank rows :1107-1108; headroom comparison :1149-1154.""")


# --- B8: Act 3 — the NDCG tree -------------------------------------------
s, y = slide(prs, "Act 3: change one variable — split on measured NDCG",
             "Same tree learner, same candidates, same stopping rule. The only change is what the "
             "split criterion optimises.",
             kicker="paper b · act 3 · the demonstration", accent=ACCENT_B)

lw = Inches(6.35)
textbox(s, MARGIN_L, y + Inches(0.03), lw, Inches(0.3),
        "CONSTRUCTION", size=11, bold=True, color=ACCENT_B)
table(s, [
    ["", ""],
    ["Candidates per node", "1,000 = 500 popularity-ranked items + 500 answerability-ranked "
     "concepts; one argmax"],
    ["Objective", "node-size-weighted mean of the three children's cohort-mean measured "
     "held-out full NDCG@10"],
    ["Growth", "fitted on 10,000 VALIDATION users, then frozen"],
    ["Shape", "depth 5, 15 leaves, 22 nodes; MIN_LEAF = 200 as the only stopping rule"],
    ["Budget", "16 charged questions; leak assertion 0 of 10,000 on both cohorts"],
    ["The root splitter", "a CONCEPT, answered by 74.2 %, splitting 2,606 / 4,814 / 2,580. "
     "The best ITEM split dumps 7,045 of 10,000 into did-not-answer."],
], MARGIN_L, y + Inches(0.38), lw, col_w=[30, 70], font_size=10.8, row_h=0.42,
    first_col_bold=True, header_rows=0)

rx = MARGIN_L + lw + Inches(0.45)
rw = CONTENT_W - lw - Inches(0.45)
textbox(s, rx, y + Inches(0.03), rw, Inches(0.3),
        "ON 10,000 HELD-OUT TEST USERS", size=11, bold=True, color=ACCENT_B)
table(s, [
    ["K", "full gain over static [95 % CI]", "tail gain over static [95 % CI]"],
    ["1", "0.00000 (exact, by nesting)", "0.00000 (exact)"],
    ["2", "+0.00528 [+0.00324, +0.00725]", "−0.00262 [−0.00481, −0.00051]"],
    ["4", "+0.00891 [+0.00632, +0.01147]", "+0.00482 [+0.00214, +0.00756]"],
    ["8", "+0.01062 [+0.00775, +0.01354]", "+0.00835 [+0.00543, +0.01130]"],
    ["16", "+0.01035 [+0.00739, +0.01323]", "+0.00705 [+0.00394, +0.01012]"],
], rx, y + Inches(0.38), rw, col_w=[8, 46, 46], font_size=10.8, row_h=0.335,
    align=[C, R, R], first_col_bold=True)

textbox(s, rx, Inches(3.75), rw, Inches(1.5),
        ["Absolutes at K=16:  tree 0.26936 / 0.13074  ·  static 0.25900 / 0.12370  ·  "
         "no questions 0.15855 / 0.02819.",
         ("Seven of the eight K≥2 intervals exclude zero in the tree's favour. The eighth excludes "
          "zero AGAINST it — tail at K=2, same sign on validation. It is printed.",
          {"size": 11, "bold": False, "color": MUTED})],
        size=11.5, bold=True, color=INK, space_after=4, line_spacing=1.14)

textbox(s, MARGIN_L, Inches(5.75), CONTENT_W, Inches(1.2),
        ["What it cost, and the correction that halved it.",
         ("Validation gain +0.02794 full / +0.01055 tail became +0.01035 / +0.00705 on TEST — "
          "2.70× and 1.50× optimism. And when the static comparator was rebuilt under the "
          "correct mask the tail advantage fell from +0.02092 to +0.00705: two thirds of the "
          "reported tail advantage was comparator weakness, not method.",
          {"size": 12, "bold": False, "color": MUTED})],
        size=13.5, bold=True, color=INK, space_after=3, line_spacing=1.14)
footer(s, "Sources: chapterB_v2.tex tab:tree (:1195-1215), :1166-1185, :1366-1393 · "
          "docs/results/CONDITIONED_ARMS_ON_TEST.md")
notes(s, """WHAT TO SAY
Act 3 is the demonstration and it is deliberately boring in design: same tree learner, same candidate
pool, same stopping rule, same budget. The only thing changed is what the split criterion optimises —
measured held-out NDCG@10 instead of a surrogate.

On ten thousand held-out test users the tree beats the strong static order by +0.01035 full and
+0.00705 tail at sixteen questions, intervals clear of zero. Seven of the eight budgets separate in
the tree's favour. The eighth separates against it — tail at two questions — and it is printed in the
paper rather than dropped.

Two honesty notes I would rather say than be asked. Validation overstated the effect by 2.7× on full
and 1.5× on tail; that gap is normal and it is why the test look was reserved. And more importantly:
an earlier draft reported a much bigger tail advantage, +0.02092. When we rebuilt the static
comparator under the correct candidate mask, two thirds of that evaporated. The tree's own numbers
did not move at all — the comparator moved. The rebuilt order shares four of sixteen questions and
zero of sixteen positions with the one it replaced. That is a lesson I keep re-learning: when a
contrast moves, check whether the arm moved or the comparator did.

SOURCES
Construction: chapterB_v2.tex :1166-1185. tab:tree: :1195-1215. Costs and optimism: :1366-1393.
Comparator swap +0.01371→+0.01035 and +0.02092→+0.00705: docs/results/CONDITIONED_ARMS_ON_TEST.md.""")


# --- B9: the six-criterion ablation ---------------------------------------
s, y = slide(prs, "The six-criterion ablation: 56 intervals, one variable",
             "Six split criteria, everything else held identical. All 56 non-nested intervals in "
             "the matrix exclude zero.",
             kicker="paper b · act 3 · ablation", accent=ACCENT_B)

table(s, [
    ["split criterion (K=16, 10,000 TEST users)", "full@10", "gain vs static", "tail@10",
     "gain vs static"],
    ["measured NDCG@10  — ours", "0.26936", "+0.01035", "0.13074", "+0.00705"],
    ["pairwise rank loss  (Hu & Yu 2013)", "0.22859", "−0.03041", "0.10031", "−0.02338"],
    ["rating variance  (Golbandi 2011)", "0.22840", "−0.03061", "0.10124", "−0.02246"],
    ["logistic loss, as run  (Hu & Yu 2013)", "0.17274", "−0.08627", "0.06817", "−0.05552"],
    ["shuffle, min-leaf-respecting", "—", "−0.04817", "—", "−0.03412"],
    ["shuffle, unrestricted", "—", "−0.03934", "—", "−0.02126"],
], MARGIN_L, y + Inches(0.05), CONTENT_W, col_w=[36, 16, 16, 16, 16],
    font_size=11.5, row_h=0.345, align=num(5), first_col_bold=True)

lw = Inches(6.5)
textbox(s, MARGIN_L, Inches(3.55), lw, Inches(0.3),
        "WHAT THE MATRIX SAYS", size=11, bold=True, color=ACCENT_B)
bullets(s, MARGIN_L, Inches(3.90), lw, [
    ("The result is (measured) ≫ (pairwise) ≈ (variance)", "— not the pre-registered "
     "ordering. Pairwise vs variance is the ONLY null in the matrix: +0.00020 "
     "[−0.00331, +0.00371] full."),
    ("Every criterion that can choose at all splits the root on a concept.", ""),
    ("The min-leaf shuffle grows deeper (7 vs 5), wider (27 leaves vs 15) and more varied "
     "(286 distinct questions vs 159)", "— and still loses by 0.048. Structure is not the "
     "thing that helps."),
], size=12)

rx = MARGIN_L + lw + Inches(0.45)
rw = CONTENT_W - lw - Inches(0.45)
textbox(s, rx, Inches(3.55), rw, Inches(0.3),
        "WHAT IT COST, AND WHY NOBODY DOES IT", size=11, bold=True, color=ACCENT_B)
table(s, [
    ["", ""],
    ["Taylor construction", "scores all 1,000 candidates in ~11 s"],
    ["Our exact objective", "37,191 s to grow one tree — ~3 orders of magnitude per node"],
    ["Nothing sampled", "the pairwise loss sums over all 137,591,707 fold-in pairs"],
    ["Reproduction", "the ablation reproduces every published tree number to 2.81×10⁻⁷"],
], rx, Inches(3.90), rw, col_w=[30, 70], font_size=11, row_h=0.40,
    first_col_bold=True, header_rows=0)
textbox(s, rx, Inches(5.85), rw, Inches(0.8),
        "This is the answer to “why has nobody done this?” — the surrogates exist because "
        "the exact objective is four orders of magnitude more expensive.",
        size=11.5, color=MUTED, line_spacing=1.15)
footer(s, "Sources: chapterB_v2.tex tab:splitcrit (:1250-1313), :1328-1335 · "
          "docs/results/HU_YU_2013_REPLICATION_AND_ABLATION.md")
notes(s, """WHAT TO SAY
This is the cleanest experiment in the thesis and the one I would put in a conference talk. Six split
criteria; the tree learner, the candidate pool, the stopping rule, the budget, the consumer and the
users are all identical. Only the criterion changes. Fifty-six non-nested intervals in the matrix,
every one excluding zero.

Two findings I did not predict. First, the ordering: measured NDCG is far ahead, and then pairwise
rank loss and rating variance are statistically indistinguishable from each other — the only null in
the whole matrix. I had pre-registered that pairwise would beat variance. It does not.

Second, the shuffle control is instructive. If you grow the tree with random splits but keep the
minimum-leaf rule, you get a DEEPER, WIDER, more varied tree — seven levels, twenty-seven leaves,
286 distinct questions against our 159 — and it loses by 0.048. So the thing that helps is not
structure, variety, or personalisation. It is the criterion.

The right-hand box is the honest answer to why the field uses surrogates: our exact objective takes
about ten hours to grow one tree against eleven seconds for the Taylor construction. Three orders of
magnitude per node. The surrogates are not lazy, they are a rational response to cost — and the
contribution here is showing what the cost buys.

SOURCES
tab:splitcrit and readings: chapterB_v2.tex :1250-1313. Pairwise-vs-variance null: :1271-1275.
Cost and reproduction tolerance: :1300-1313. Hu & Yu native replication (their protocol, MAP, ML-1M,
depths 3 and 5, figure-only so pixel-matched): :1328-1335 and
docs/results/HU_YU_2013_REPLICATION_AND_ABLATION.md (finalised 2026-08-13, citable).""")


# --- B10: bank vs conditioning ---------------------------------------------
s, y = slide(prs, "Bank beats conditioning",
             "If you can change one thing about a cold interview, change WHAT it asks about — "
             "not how it decides what to ask.",
             kicker="paper b · the practical result", accent=ACCENT_B)

table(s, [
    ["change made (10,000 TEST users)", "full gain [95 % CI]", "tail gain",
     "share of the static's own lift"],
    ["BANK: items → items + concepts, equal budget", "+0.01430 [+0.01124, +0.01739]",
     "+0.02199", "16.6 %"],
    ["CONDITIONING: the NDCG tree, combined pool, K=16", "+0.01035 [+0.00739, +0.01323]",
     "+0.00705", "10.3 %"],
    ["CONDITIONING: genre-conditioned greedy, item pool, K=15", "+0.00862 [+0.00595, +0.01131]",
     "+0.00591", "10.2 %"],
], MARGIN_L, y + Inches(0.10), CONTENT_W, col_w=[38, 26, 15, 21],
    font_size=12, row_h=0.42, align=num(4), first_col_bold=True)

textbox(s, MARGIN_L, Inches(3.35), CONTENT_W, Inches(1.0),
        ["On the tail the asymmetry is 3×:  +0.02199 for changing the bank against +0.00705 "
         "for conditioning.",
         ("Both changes are real and both intervals are clean. The point is the ordering, and it "
          "is the opposite of where the field's effort goes.",
          {"size": 12.5, "bold": False, "color": MUTED})],
        size=15, bold=True, color=INK, space_after=4, line_spacing=1.14)

lw = Inches(6.5)
textbox(s, MARGIN_L, Inches(4.55), lw, Inches(0.3),
        "AND YET — THE ACTION SPACE IS NOT A FREE LUNCH", size=11, bold=True, color=ACCENT_B)
bullets(s, MARGIN_L, Inches(4.90), lw, [
    ("Enlarging the question bank further made the selectors that consume it perform WORSE.",
     "That result is why Paper C claims capability and never value."),
    ("The gain is answerability, not information.", "Concepts are less informative "
     "per answer and win anyway, because they get answered."),
], size=12.5)

rx = MARGIN_L + lw + Inches(0.45)
rw = CONTENT_W - lw - Inches(0.45)
textbox(s, rx, Inches(4.55), rw, Inches(1.9),
        ["FOR SCALE",
         ("Seed a cold user with 8 items from their own history, then run the identical 16 "
          "questions:", {"size": 11, "bold": False, "color": INK}),
         ("+0.0206 full   ·   +0.0353 tail", {"size": 14, "bold": True, "color": ACCENT_B}),
         ("The recommender's own knowledge of the user dwarfs every interview-design choice "
          "on this slide.", {"size": 11, "bold": False, "color": MUTED})],
        size=11, bold=True, color=ACCENT_B, space_after=5, line_spacing=1.15)
footer(s, "Sources: chapterB_v2.tex tab:bankdecomp (:1344-1361), :1935-1938 · "
          "memory: concept-bank-does-not-rescue-adaptive-pv")
notes(s, """WHAT TO SAY
This is the slide with the practical takeaway, and it is the one I would lead a talk with.

Three changes you can make to a cold interview, all measured on the same test users against the same
comparator. Adding a concept channel to the question bank buys +0.0143 full and +0.0220 tail.
Conditioning the question order on the answers — which is what the entire adaptive-elicitation
literature is about — buys +0.0104 full and +0.0071 tail. On the tail the bank change is three times
larger.

So: if you can change one thing, change what the system asks about, not how it decides what to ask.

Two caveats that keep me honest. Enlarging the bank further did not keep helping — beyond a point it
made the selectors consuming it worse, which is why Paper C is careful to claim capability and not
value. And for scale, the bottom-right box: giving the recommender eight of the user's real ratings
and then running the identical sixteen questions is worth +0.0206 full and +0.0353 tail. The
recommender's own knowledge dwarfs every interview-design choice on this slide.

SOURCES
tab:bankdecomp: chapterB_v2.tex :1344-1361. Seeded-user comparison: :1935-1938.
'The action space was not the constraint': memory note concept-bank-does-not-rescue-adaptive-pv;
restated in chapterC_v2.tex Positioning.""")

# --- B11: Act 4 answerability ----------------------------------------------
s, y = slide(prs, "Act 4: answerability orders the channels",
             "Three independent measurements say the same thing. What decides which channel wins "
             "is not how informative a question is — it is whether anyone can answer it.",
             kicker="paper b · act 4", accent=ACCENT_B)

table(s, [
    ["Independent measurement", "What it shows"],
    ["1.  Composition of the optimised static order",
     "the combined greedy opens  c c i c c i i c c c c i i i c c  — concepts hold 10 of 16 slots "
     "and BOTH of the first two"],
    ["2.  Per-state criterion",
     "at cold states the best question is a concept by more than 12× (+0.0238 vs +0.0019 for the "
     "best item); at request-placed states the optimum flips to an item (+0.0138 vs +0.0076)"],
    ["3.  The tree's own root",
     "the chosen concept is answered by 74.2 %; the best item split sends 70 % of the cohort into "
     "did-not-answer"],
    ["The mechanism, in one line",
     "sixteen item questions collect 3.11 answers.  Sixteen concept questions collect 12.36.  "
     "A factor of four."],
], MARGIN_L, y + Inches(0.05), CONTENT_W, col_w=[26, 74],
    font_size=11.2, row_h=0.475, first_col_bold=True)

lw = Inches(6.5)
textbox(s, MARGIN_L, Inches(4.15), lw, Inches(0.3),
        "WHICH SINGLE CHANNEL IS THE BETTER BANK? (K=16, TEST)", size=11, bold=True, color=ACCENT_B)
table(s, [
    ["contrast", "full", "tail"],
    ["concepts − items", "−0.00003 [−.00315,+.00310]", "+0.01436 [+.01093,+.01776]"],
    ["combined − items", "+0.01430", "+0.02199"],
    ["combined − concepts", "+0.01434", "+0.00763"],
], MARGIN_L, Inches(4.50), lw, col_w=[30, 36, 34], font_size=10.5, row_h=0.32, align=num(3))
textbox(s, MARGIN_L, Inches(5.90), lw, Inches(1.0),
        "A dead heat on full, a rout on tail — and concepts lead on full at every budget below "
        "16 (+0.0048, +0.0078, +0.0104, +0.0112 at k=1,2,4,8, every interval clean). Items catch "
        "up only at sixteen questions.",
        size=11.5, color=MUTED, line_spacing=1.15)

rx = MARGIN_L + lw + Inches(0.45)
rw = CONTENT_W - lw - Inches(0.45)
textbox(s, rx, Inches(4.15), rw, Inches(0.3),
        "THE PART THAT SHOULD WORRY US", size=11, bold=True, color=ACCENT_B)
bullets(s, rx, Inches(4.50), rw, [
    ("Answering AT ALL is worth ~+0.25.", "The largest question-CHOICE effect anywhere "
     "in the tree is +0.048."),
    ("1,785 users answer neither of the first two questions;", "1,438 of them cannot "
     "answer the third. ~14 % of the cohort is untouched, sitting at 0.087."),
], size=12)
footer(s, "Sources: chapterB_v2.tex :1469-1521, :834-930, :1385-1393")
notes(s, """WHAT TO SAY
Act 4 is the mechanism, and it ties back to Paper A's concept-channel result.

Three independent measurements, listed on the slide, all say the same thing: the channel that wins is
the one that gets answered. The optimised order puts concepts in ten of sixteen slots and both of the
first two. Per-state, the best cold question is a concept by more than twelve times. And the tree's
own root chooses a concept answered by three quarters of the cohort over an item that would send
seventy percent into 'did not answer'.

The one-line mechanism: sixteen item questions collect 3.11 answers; sixteen concept questions
collect 12.36.

The box bottom right is the finding that I think is genuinely important and slightly deflating for
the whole field. The difference between answering at all and answering nothing is worth about +0.25
NDCG. The largest effect of choosing a BETTER question, anywhere in our tree, is +0.048. So question
selection is a five-times-smaller lever than getting an answer out of the user. And about fourteen
percent of the cohort answers nothing in the first three questions and sits at 0.087 — they never
enter the game at all.

SOURCES
Three measurements and the 3.11/12.36 mechanism: chapterB_v2.tex :1469-1521.
Channel contrasts at K=16 and the per-budget concept lead: :1469-1521.
Policy-side answer counts (static 5.52, Thompson 3.39, three-outcome 2.86): :834-930.
Reach numbers (+0.25 vs +0.048, 1,785/1,438, 0.087): :1385-1393.""")


# --- B12: Act 5 the Reddit corpus --------------------------------------------
s, y = slide(prs, "Act 5: what people actually volunteer",
             "A corpus of real, unprompted film requests — because the whole thesis so far assumes "
             "a closed question bank, and real users do not wait to be asked.",
             kicker="paper b · act 5 · the ecological arm", accent=ACCENT_B)

lw = Inches(6.05)
textbox(s, MARGIN_L, y + Inches(0.03), lw, Inches(0.3),
        "THE CORPUS", size=11, bold=True, color=ACCENT_B)
table(s, [
    ["", ""],
    ["Size", "4,431 threads, each an unprompted free-text request with ground-truth items"],
    ["Genuine requests", "94.0 %"],
    ["Median slots per request", "5   ·   93.6 % carry ≥2 slots"],
    ["Named films", "7,572 OP-named films across 2,793 threads, 5,868 distinct titles"],
    ["Vocabulary coverage", "100.0 % of named films lie inside the 18,359-item vocabulary"],
    ["Foldable as a state", "3,078 of 4,431  (69.5 %)"],
], MARGIN_L, y + Inches(0.38), lw, col_w=[32, 68], font_size=11, row_h=0.375,
    first_col_bold=True, header_rows=0)

textbox(s, MARGIN_L, Inches(4.10), lw, Inches(0.3),
        "ENTITY LINKING IS THE RISK, SO WE MEASURED IT", size=11, bold=True, color=ACCENT_B)
table(s, [
    ["linking precision vs a 500-mention hand-labelled gold standard", "precision"],
    ["the corpus's own links", "0.572"],
    ["our two-tier pipeline (deterministic rules + LM adjudication)", "0.960"],
], MARGIN_L, Inches(4.45), lw, col_w=[74, 26], font_size=10.8, row_h=0.315, align=num(2))
textbox(s, MARGIN_L, Inches(5.60), lw, Inches(0.9),
        "The pre-registered gate was ≥0.95. Attribute mapping succeeds for 95.1 % of genres, "
        "70.2 % of eras, 66.1 % of tones — the era and tone shortfalls bound Act 5 from above "
        "and are stated as such.",
        size=11, color=MUTED, line_spacing=1.15)

rx = MARGIN_L + lw + Inches(0.40)
rw = CONTENT_W - lw - Inches(0.40)
textbox(s, rx, y + Inches(0.03), rw, Inches(0.3),
        "WHAT THEY VOLUNTEER, UNPROMPTED", size=11, bold=True, color=ACCENT_B)
table(s, [
    ["slot", "mentions it", "opens with it"],
    ["positive anchor (names a film)", "63.0 %", "63.0 %"],
    ["genre", "39.8 %", "14.7 %"],
    ["constraint", "30.7 %", "5.2 %"],
    ["tone", "21.4 %", "3.3 %"],
    ["negative anchor (a dislike)", "16.5 %", "—"],
    ["era", "9.8 %", "1.9 %"],
    ["negative genre", "4.2 %", "—"],
    ["none of the five", "11.9 %", "—"],
], rx, y + Inches(0.38), rw, col_w=[52, 24, 24], font_size=10.8, row_h=0.315, align=num(3))

textbox(s, rx, Inches(4.10), rw, Inches(2.4),
        ["WHAT PEOPLE REQUEST ≠ WHAT PEOPLE WATCH",
         ("Horror is 27.5 % of mapped genre mentions and Thriller 15.8 %. In ML-25M validation, "
          "Horror is the favourite genre of 51 users out of 10,000; Drama is the favourite of "
          "5,481.", {"size": 11.5, "bold": False, "color": INK}),
         ("", {"size": 5}),
         ("The request distribution is inverted relative to the consumption distribution — so a "
          "system tuned on watch logs is tuned on the wrong prior for what gets asked.",
          {"size": 11, "bold": False, "color": MUTED})],
        size=11, bold=True, color=ACCENT_B, space_after=4, line_spacing=1.15)
footer(s, "Sources: chapterB_v2.tex :1542-1588 · docs/results/REDDIT_LINKER_AUDITION.md · "
          "REDDIT_HE_GOLD_LABELS.md · docs/design/DS_REDDIT_LINKING.md")
notes(s, """WHAT TO SAY
Act 5 is the ecological arm and it exists because everything so far assumes a closed bank of
questions and a user who waits to be asked. Real users do not do that.

Four thousand four hundred and thirty-one threads, each a genuine unprompted request with
ground-truth recommendations attached. Ninety-four percent are real requests. The median request
carries five distinct pieces of preference information before anyone asks anything.

The linking is the methodological risk, so we measured it against a five-hundred-mention hand-labelled
gold standard: the corpus's own links are 0.572 precise; our two-tier pipeline is 0.960, against a
pre-registered 0.95 gate. And every one of the 7,572 named films falls inside our item vocabulary,
which is lucky and worth saying.

The finding I like most is bottom right. Horror is twenty-seven percent of what people ASK for. In
our validation cohort Horror is the favourite genre of fifty-one users out of ten thousand; Drama is
the favourite of five and a half thousand. The request distribution is inverted relative to the
consumption distribution. A system tuned on watch logs is holding the wrong prior for what people
will actually say to it.

SOURCES
Corpus, linking, census: chapterB_v2.tex :1542-1588 (tab:opener_census).
Linking audition and the ≥0.95 gate: docs/results/REDDIT_LINKER_AUDITION.md (rules→LM 0.960,
LM-solo 0.972); docs/design/DS_REDDIT_LINKING.md; REDDIT_HE_GOLD_LABELS.md (0.572 combined).
NOTE the chapter intro at :133-134 says '3,078 recommendation requests, 63 %' — 63.0 % belongs to
the 4,431 census; 3,078 is the foldable subset. Quote 4,431.""")


# --- B13: openers + the one policy -------------------------------------------
s, y = slide(prs, "Act 5: opener-fitted interviews, and one policy that holds the whole forest",
             "Fit a tree where you mean to serve — then distil every tree into a single policy "
             "that has to find the group itself.",
             kicker="paper b · act 5 · the policy", accent=ACCENT_B)

lw = Inches(6.7)
textbox(s, MARGIN_L, y + Inches(0.02), lw, Inches(0.3),
        "OPENER-FITTED TREES  (VALIDATION, K=15, full NDCG@10)", size=11, bold=True, color=ACCENT_B)
table(s, [
    ["group", "n", "static", "cold", "opener", "opener − cold"],
    ["art house", "473", ".26867", ".26508", ".28978", "+.02470 ★"],
    ["gore", "419", ".30394", ".30206", ".31450", "+.01244"],
    ["teenagers", "540", ".24542", ".25026", ".26192", "+.01166"],
    ["marvel", "593", ".40549", ".41570", ".42406", "+.00836"],
    ["inspiring", "1,557", ".24040", ".24626", ".24644", "+.00018"],
    ["dialogue driven", "398", ".21959", ".23058", ".22793", "−.00266"],
    ["pooled", "3,980", ".27365", ".27859", ".28547", "+.00688 ★"],
], MARGIN_L, y + Inches(0.37), lw, col_w=[26, 11, 15, 15, 15, 18],
    font_size=10.5, row_h=0.30, align=num(6), first_col_bold=True)

textbox(s, MARGIN_L, Inches(3.65), lw, Inches(0.9),
        ["The chain:  cold − static = +.00495 ★   then   opener − cold = +.00688 ★",
         ("★ = paired bootstrap interval excludes zero. Fitting the tree where you mean to serve "
          "beats an anchor teacher; then fitting it to the opener-conditioned state beats that.",
          {"size": 11, "bold": False, "color": MUTED})],
        size=12.5, bold=True, color=INK, space_after=4, line_spacing=1.14)

textbox(s, MARGIN_L, Inches(4.70), lw, Inches(0.3),
        "THE CONTROL THAT MATTERS", size=11, bold=True, color=ACCENT_B)
textbox(s, MARGIN_L, Inches(5.05), lw, Inches(1.6),
        ["Scramble the routing and it collapses on every group — .888→.148, .834→.040, "
         ".857→.038, .852→.121, .958→.094, .757→.111, against a chance floor of .167 — and the "
         "VALUE goes negative on marvel (−.01190) and inspiring (−.00572), both separating.",
         ("Honest limit: Act 5's openers are group-stamped, so routing is partly circular. The "
          "opener−cold contrast survives because the STATE is held fixed. Claim adaptivity beyond "
          "the closed tree — not routing.", {"size": 11, "bold": False, "color": ACCENT_C})],
        size=11.5, color=MUTED, space_after=5, line_spacing=1.15)

rx = MARGIN_L + lw + Inches(0.40)
rw = CONTENT_W - lw - Inches(0.40)
textbox(s, rx, y + Inches(0.02), rw, Inches(0.3),
        "★ THE SPENT TEST LOOK  —  one distilled policy, reserved TEST cohort, K=15",
        size=11, bold=True, color=ACCENT_B)
table(s, [
    ["population", "n", "policy", "cold", "policy − cold"],
    ["all users", "10,000", ".26842", ".26499", "+.00343 ★"],
    ["volunteered something", "3,877", ".28939", ".27993", "+.00946 ★"],
    ["volunteered nothing", "6,123", ".25514", ".25553", "−.00038"],
], rx, y + Inches(0.37), rw, col_w=[36, 14, 16, 16, 18], font_size=10.5, row_h=0.32,
    align=num(5), first_col_bold=True)

textbox(s, rx, Inches(2.35), rw, Inches(1.9),
        ["ONE POLICY, NOT SIX",
         ("526,542 parameters · top-1 agreement 0.991 with the interviews it imitates · routing "
          ".741–.957 · lands within ±.004 of the correct group's own interview in every group.",
          {"size": 11, "bold": False, "color": INK}),
         ("Distillation buys no accuracy. The value is that one object holds the whole forest and "
          "has to find the group itself — and value does not track routing: marvel routes best and "
          "gains least.", {"size": 10.5, "bold": False, "color": MUTED})],
        size=11, bold=True, color=ACCENT_B, space_after=4, line_spacing=1.14)

textbox(s, rx, Inches(4.45), rw, Inches(1.9),
        ["WHAT THE TEST LOOK ACTUALLY BOUGHT",
         ("A single reserved look, spent once, on a policy that must route as well as ask. "
          "Non-monotone in budget: K=1 +0.00161 ★, K=4 +0.00119 (contains zero), K=15 +0.00343 ★.",
          {"size": 10.5, "bold": False, "color": INK}),
         ("Intervals (paired bootstrap, from the run artefact): ALL K=15 +0.00343 "
          "[+0.00173, +0.00509]; warm users +0.00946 [+0.00524, +0.01376]; "
          "do-no-harm cell −0.00038 [−0.00098, +0.00012], i.e. a clean null.",
          {"size": 10, "bold": False, "color": ACCENT_C})],
        size=11, bold=True, color=ACCENT_B, space_after=4, line_spacing=1.14)
footer(s, "Sources: chapterB_v2.tex tab:volunteered, tab:policy · M4B_FOREST_POLICY.md · M6L_OPENER_FITTED_TEACHER.md · CIs: experiments/battery/m6/m6mix_B_test.json cells.ALL/warm/COLD K=15")
notes(s, """WHAT TO SAY
Act 5's result, and it is a two-step chain. First: if you fit the interview tree at the state the user
is ACTUALLY in — after they have volunteered something — rather than at a cold state, you gain. Cold
tree beats the anchor teacher by +0.00495; the opener-fitted tree beats the cold tree by +0.00688,
pooled over 3,980 users, both intervals clean. Per group it ranges from +0.0247 on art-house down to a
small negative on dialogue-driven.

Then: rather than ship six trees plus a router, we distil the whole forest into one policy of about
half a million parameters. It agrees with the interviews it imitates 99.1 % of the time at top-1, and
it has to find the group itself — routing accuracy .74 to .96. It lands within four ten-thousandths
of the correct group's own interview in every group.

On the reserved test cohort, spent once, the policy beats the cold interview by +0.00343 pooled, and
by +0.00946 on the 39 % of users who volunteer something. On users who volunteer nothing it is a null,
which is exactly right — there is nothing for it to route on.

Two honesty notes. The openers are group-stamped, so the ROUTING claim is partly circular; the
opener-minus-cold comparison survives because the state is held fixed, so the claim is adaptivity
beyond the closed tree, not routing. And the scramble control on the left is what makes the routing
non-vacuous: scramble it and it collapses to chance and the value goes negative.

SOURCES
tab:volunteered: chapterB_v2.tex :1946-1967. Scramble control: :1991-2010. tab:policy: :2042-2073.
Distillation fidelity and 526,542 parameters: docs/results/M4B_FOREST_POLICY.md.
Opener-fitted run detail: docs/results/M6L_OPENER_FITTED_TEACHER.md (which flags cluster-vs-cold as
an upper bound and cluster-vs-anchor as the clean cell).
Group-stamped openers: memory note opener-synthesis-is-group-stamped.
NO printed numeric CI exists for +0.00343 / +0.00946 — the chapter marks them with ★ only.""")


# --- B14: B status ------------------------------------------------------------
s, y = slide(prs, "Paper B: where it stands",
             "Chapter closed 30 August 2026. Acts 1–4 are on TEST and frozen; the test look is "
             "spent and replicates.",
             kicker="paper b · status", accent=ACCENT_B)

lw = Inches(7.1)
table(s, [
    ["", ""],
    ["Chapter state", "CLOSED 2026-08-30. Compiles; test look spent 2026-08-12 (Acts 1–4) and "
     "2026-08-29 (Act 5), and replicates."],
    ["Frozen scope", "Acts 1–4 are on TEST and frozen. Only Act 5 was rebuilt."],
    ["Target", "ECIR 2027 full paper — deadline 2 October 2026"],
    ["The cut", "Sections 1–3 (audit · diagnosis · one-variable demonstration) compressed to "
     "~12 pp. Estimated 40–55 % acceptance."],
    ["Second choice", "RecSys, or its reproducibility track"],
    ["Act 5", "held back as a SECOND paper (RecSys / CIKM) rather than cut"],
    ["Absorbed", "Paper D is absorbed into B — the Reddit arc and the opener census cover its "
     "substance, and its own numbers were on the retired pre-July ruler"],
], MARGIN_L, y + Inches(0.05), lw, col_w=[22, 78], font_size=11.5, row_h=0.44,
    first_col_bold=True, header_rows=0)

rx = MARGIN_L + lw + Inches(0.40)
rw = CONTENT_W - lw - Inches(0.40)
textbox(s, rx, y + Inches(0.05), rw, Inches(0.3),
        "THE MOST DANGEROUS OPEN OBJECTION", size=11, bold=True, color=ACCENT_C)
textbox(s, rx, y + Inches(0.42), rw, Inches(1.9),
        ["“Act 3 rests on ONE consumer.”",
         ("And our own §flip demonstrates consumer-dependence. It is the objection I would raise. "
          "It is fixable by SCORING — not refitting — the frozen tree through RecVAE, which is a "
          "box-night, not a redesign.", {"size": 11.5, "bold": False, "color": MUTED})],
        size=13, bold=True, color=INK, space_after=4, line_spacing=1.15)

textbox(s, rx, Inches(3.55), rw, Inches(0.3),
        "CLAIM DISCIPLINE — WHAT WE DO NOT SAY", size=11, bold=True, color=ACCENT_B)
bullets(s, rx, Inches(3.90), rw, [
    "Routing users to a group interview is not a contribution — Golbandi WSDM'11 and "
    "Cluster-Based Bandits SIGIR'21 both do it.",
    "No claim that our grouping beats k-means on NDCG.",
    "Never “reproduces the optimal question order”.",
    "Never “parity” or “tie” as a headline.",
    "No fitting-population reasoning in the paper — one flat protocol sentence.",
], size=11.5)
footer(s, "Sources: docs/design/PAPER_B_MASTER_TODO.md · docs/PUBLICATION_PLAN.md roster (2026-08-18)")
notes(s, """WHAT TO SAY
Where B stands. The chapter closed at the end of August. Acts 1 through 4 are on test and frozen —
that test look was spent in mid-August and I have not touched it since. Act 5's look was spent at the
end of August and replicated.

The plan is ECIR 2027, deadline the second of October, and the conference cut is sections one to three
only — the audit, the diagnosis, and the one-variable demonstration — compressed to about twelve pages.
Act 5 is held back as a second paper rather than crushed into this one.

The right-hand box is the objection I would raise if I were reviewing it, and I would rather raise it
myself: Act 3 rests on one consumer, and our own flip result shows that consumer choice can invert a
verdict. The fix is to SCORE the already-frozen tree through RecVAE — not refit it — which is one
box-night, and I think it should be done before submission.

The claim-discipline list at the bottom is what I have committed to NOT saying, mostly because earlier
drafts said them and they were wrong.

SOURCES
docs/design/PAPER_B_MASTER_TODO.md (chapter closed, frozen scope, the conference cut, the
one-consumer objection, the claim-discipline list); docs/PUBLICATION_PLAN.md roster table.""")

# =====================================================================
# PAPER C — THE QUESTION THAT ISN'T IN THE CATALOGUE
# =====================================================================

sd = divider("Paper C — the unasked question",
             "The policy wants to ask about a direction. The catalogue's nearest question points "
             "the other way. So: say the direction in English, and audit what comes out.",
             ACCENT_C, "10 slides  ·  ~15 minutes  ·  status: LBR draft written → opportunistic")
notes(sd, """WHAT TO SAY
Section break, and a change of register. C is the capability paper — the fun one. No benchmark to
beat; the deliverable is a mechanism plus an audit that the field does not currently run.

SOURCES
Framing and status: docs/PUBLICATION_PLAN.md roster (Paper C 'REDEFINED: absorbs old E. A CAPABILITY
paper … publishable is a bonus, not a requirement'); author steer 2026-08-30.""")


# --- C1: the question ---------------------------------------------------
s, y = slide(prs, "Paper C's question",
             "Every elicitation system in the literature picks from a list. What if the best thing "
             "to ask is not on the list?",
             kicker="paper c · the question", accent=ACCENT_C)

# loop diagram
bwc, bhc = Inches(2.45), Inches(1.0)
gyc = y + Inches(0.30)
xc = [MARGIN_L + Inches(0.30) + i * Inches(3.05) for i in range(4)]
for i, (h, sub, col) in enumerate([
        ("INTERVIEW STATE", "the answers so far", MUTED),
        ("EMITTED DIRECTION", "a point outside the bank", ACCENT_C),
        ("ENGLISH", "a rendered, askable question", ACCENT_C),
        ("FOLDED BACK", "does the text carry the direction?", ACCENT)]):
    box(s, xc[i], gyc, bwc, bhc,
        [(h, {"size": 11.5}), (sub, {"size": 9.5, "bold": False, "color": MUTED})], line=col)
    if i < 3:
        arrow(s, xc[i] + bwc, gyc + bhc // 2, xc[i + 1], gyc + bhc // 2, color=ACCENT_C)
arrow(s, xc[3] + bwc // 2, gyc + bhc, xc[1] + bwc // 2, gyc + bhc, color=ACCENT, width=1.5)
textbox(s, xc[1], gyc + bhc + Inches(0.05), Inches(6.1), Inches(0.3),
        "the audit: an independently fitted bridge, never the model grading its own answer",
        size=10, color=ACCENT, align=C)

lw = Inches(6.5)
textbox(s, MARGIN_L, Inches(3.30), lw, Inches(0.3),
        "THE CLAIM, AS A CONJUNCTION", size=11, bold=True, color=ACCENT_C)
bullets(s, MARGIN_L, Inches(3.65), lw, [
    ("The decoded point is chosen by an interview policy at an interview state",
     "— not sampled, not retrieved."),
    ("The renderer is audited against a correspondence-scrambled latent",
     "— not by round-trip cosine and not by a plausibility panel."),
    ("Faithfulness is scored causally", "— substitute the text where the latent was, "
     "push it through an independently fitted map, and see if it does the same thing."),
], size=12.5)

rx = MARGIN_L + lw + Inches(0.45)
rw = CONTENT_W - lw - Inches(0.45)
textbox(s, rx, Inches(3.30), rw, Inches(0.3),
        "EVERY LEG HAS PRIOR ART — WE CLAIM NO LEG ALONE", size=11, bold=True, color=ACCENT_C)
table(s, [
    ["prior work", "what it already does / lacks"],
    ["GeneRec (Wang 2023)", "generates objects the catalogue does not contain — but items, "
     "not questions"],
    ["ELM (Tennenholtz 2023)", "verbalises a synthesised latent — we borrow the mechanism "
     "rather than invent one"],
    ["Vendrov (AAAI 2020)", "synthesises a query continuously then SNAPS it to real items, "
     "inside recommendation, six years ago"],
    ["EAGLE (Tennenholtz 2024)", "steers an LM over an ML-25M behavioural embedding — and has "
     "NO decoder: it edits text and encodes the result. Its authors name the price: coverage is "
     "limited by the action space, with no remedy they can see."],
], rx, Inches(3.65), rw, col_w=[30, 70], font_size=10.5, row_h=0.50)

textbox(s, MARGIN_L, Inches(6.35), lw, Inches(0.9),
        ["⛔ CAPABILITY, NOT VALUE.",
         ("No claim anywhere that asking a synthesised question improves recommendations. "
          "Paper B is the reason: there, enlarging the bank made things worse.",
          {"size": 11.5, "bold": False, "color": MUTED})],
        size=12.5, bold=True, color=ACCENT_C, space_after=3, line_spacing=1.14)
footer(s, "Sources: paperC_lbr.tex §1 · chapterC_v2.tex :108-190 (positioning)")
notes(s, """WHAT TO SAY
Paper C's question. Every elicitation system in the literature, including everything in Paper B,
picks a question from a list. The question here is what happens when the best thing to ask is not on
the list.

The loop at the top is the whole paper: take the interview state, emit a direction that is not any
bank question, render it into English, and then push the English back through an independently fitted
map to ask whether the text actually carries the direction.

I want to be careful about novelty, so the right-hand table is deliberately deflating. Generating
off-catalogue objects is GeneRec's territory. Verbalising a latent is ELM's mechanism and we borrow
it. Synthesising a query and snapping it to real items was done inside recommendation in 2020. What is
unoccupied is the conjunction on the left: the point is chosen by a policy at an interview state, the
renderer is audited against a scrambled latent rather than a plausibility panel, and faithfulness is
scored causally.

The nearest relative sharpens rather than blunts it: EAGLE steers a language model over an ML-25M
behavioural embedding and has no decoder at all. Its authors say themselves that coverage is limited
by the action space and they see no remedy. They looked at decoding an off-manifold point and routed
around it. We decode, and then audit what comes out.

And the red line at the bottom holds everywhere: capability, not value. Paper B is why.

SOURCES
Claim conjunction and the neighbour analysis: chapterC_v2.tex :108-190, §positioning;
paperC_lbr.tex §1. Capability-not-value ruling: chapterC_v2.tex 'Two things we say here rather than
in a limitations paragraph'.""")


# --- C2: the arc as a map -------------------------------------------------
s, y = slide(prs, "Six experiments, in order",
             "Each one is gated, pre-registered, and read against its own null. Three of them were "
             "decided by a control rather than by the effect.",
             kicker="paper c · the arc", accent=ACCENT_C)

table(s, [
    ["", "the question it asks", "headline", "verdict"],
    ["C0  geometry",
     "does the deployed policy's emitted vector sit ON the question it snaps to?",
     "max cos₁ = 0.1203 over all 160,000 states; mean-removed cos to the snapped entry "
     "+0.488 vs +0.170 random",
     "on-grid prediction FALSIFIED — but the registered rule had no power, and we say so"],
    ["C1  emission",
     "can ONE object run the tree's ordinary interview and emit off-catalogue blends?",
     "245/245 distinct questions preserved, snap agreement 0.9950, fit cos 0.96958 at "
     "trained states",
     "both blocking gates PASS; generalisation to withheld states FAILS"],
    ["C3B  mechanism",
     "does the vector condition the text, or is the renderer ignoring it?",
     "A − A′ = +0.1114 [+0.1054, +0.1173]; 107 % of the gold phrase's own ceiling; "
     "identification 0.6778",
     "the two mechanism criteria PASS with room; the equivalence control fails on a "
     "favourable direction — composite kept as FAILURE"],
    ["C3B-v2  askability",
     "does it survive being reformatted into real questions?",
     "94.7 % [0.9398, 0.9542] well-formed; A − A′ still +0.05169 on items, +0.28625 on concepts",
     "every reading PASSES, under both decodings, on both channels"],
    ["C4  the loop",
     "does the rendered word land in the direction's own neighbourhood?",
     "median rank 91 of 1,031 where the render is distinctive; catalogue text −0.158 vs "
     "rendered word +0.204",
     "point-specificity +0.28593 [+0.17820, +0.39777]; prior floor at zero"],
    ["C4B/C4C  cost",
     "what does the round trip through English cost, and is it removable?",
     "cost +0.01802 full / +0.00672 tail; decoding alone takes distinct renders 23 → 41 and "
     "cuts the cost 2–3.5×",
     "the cost is largely HARM from confidently-wrong renders, not lost meaning"],
], MARGIN_L, y + Inches(0.02), CONTENT_W, col_w=[13, 22, 34, 31],
    font_size=9.3, row_h=0.79, first_col_bold=True)
footer(s, "Sources: docs/results/{C0_GEOMETRY, C1_FINAL, C3B_ADAPTER, C3B_V2_QUESTIONS, C4_PILOT, "
          "C4B_NOISE, C4C_DECODING}.md · paperC_lbr.tex")
notes(s, """WHAT TO SAY
The arc, and this is C's work-volume slide. Six pre-registered experiments run between 21 August and
30 August, each with a design sheet written before the run, each with gates read before any effect.

Walk it left to right. C0 asks where the deployed policy's vector actually sits — and finds it never
sits on the question it picks. C1 asks whether one object can both run the ordinary interview and
emit off-catalogue directions; it can. C3B asks the mechanism question — does the text depend on the
vector — and that is the one I would defend hardest. C3B-v2 asks whether it survives being turned
into real askable questions. C4 closes the loop. C4B and C4C price the round trip.

The right-hand column is the honest column. Note that three of the six verdicts were decided by a
control rather than by the effect: C0's registered rule turned out to have no power because a random
direction lands in the same cell; C3B's equivalence control fired in a direction that was actually
favourable to us and we kept the FAILURE label anyway, because a pre-registration does not get
reinterpreted after the data; and C1's generalisation reading was matched by the untrained network.

SOURCES
Per-experiment result docs as footed; arc summary in paperC_lbr.tex.""")


# --- C3: C0 + C1 off-catalogue emission ------------------------------------
s, y = slide(prs, "Off-catalogue emission: one object, two behaviours",
             "The policy runs the tree's ordinary interview essentially unchanged AND emits "
             "gain-weighted blends that are not any bank question.",
             kicker="paper c · c0 and c1", accent=ACCENT_C)

lw = Inches(6.35)
textbox(s, MARGIN_L, y + Inches(0.02), lw, Inches(0.3),
        "C0 — WHERE THE DEPLOYED POLICY'S VECTOR SITS", size=11, bold=True, color=ACCENT_C)
table(s, [
    ["measured at all 160,000 validation states", "emitted", "random"],
    ["median cosine to the question it snaps to", "−0.18201", "+0.06418"],
    ["MAXIMUM cosine, over every one of the 160,000 rows", "0.1203", "0.398"],
    ["states sitting on their snapped question (cos > 0.98)", "0 of 160,000", "0"],
    ["mean-removed cosine to the snapped entry", "+0.488", "+0.170"],
], MARGIN_L, y + Inches(0.37), lw, col_w=[58, 21, 21], font_size=10.5, row_h=0.315,
    align=num(3), first_col_bold=True)

textbox(s, MARGIN_L, Inches(3.35), lw, Inches(1.4),
        ["The mechanism, corrected.",
         ("The whole score vector sits far below zero — median top score −82.7 — because "
          "cross-entropy constrains only score DIFFERENCES. So the emission is a large shared "
          "common-mode offset PLUS a genuine bank-aligned direction that raw cosine hides. "
          "“A scoring functional, not a semantic point” is true only in raw space.",
          {"size": 11, "bold": False, "color": MUTED})],
        size=12.5, bold=True, color=INK, space_after=4, line_spacing=1.15)

rx = MARGIN_L + lw + Inches(0.45)
rw = CONTENT_W - lw - Inches(0.45)
textbox(s, rx, y + Inches(0.02), rw, Inches(0.3),
        "C1 — BOTH BLOCKING GATES PASS", size=11, bold=True, color=ACCENT_C)
table(s, [
    ["gate / quantity", "value", "bar"],
    ["distinct questions preserved", "245 / 245", "≥ 0.75×"],
    ["snap agreement with the banked policy", "0.9950", "≥ 0.90"],
    ["emission / target norm ratio, trained states", "1.0449", "[0.5, 2.0]"],
    ["cos(ẑ, own target), trained states", "0.96958", "—"],
    ["median top-1 blend weight (0 of 72 above 0.5)", "0.2254", "0.121 random"],
    ["within-state per-user agreement", "0.99991", "—"],
], rx, y + Inches(0.37), rw, col_w=[54, 24, 22], font_size=10.5, row_h=0.315,
    align=num(3), first_col_bold=True)

textbox(s, rx, Inches(3.05), rw, Inches(2.0),
        ["THE CONTRACT HELD THROUGHOUT",
         ("72 of 72 gain sweeps match the banked splitter at maximum error 0; the descent "
          "reproduces all 1,279,335 banked labels with zero mismatches; three hyperparameters "
          "chosen once and never swept.", {"size": 10.5, "bold": False, "color": MUTED})],
        size=11, bold=True, color=ACCENT_C, space_after=4, line_spacing=1.15)

textbox(s, MARGIN_L, Inches(5.15), CONTENT_W, Inches(1.5),
        ["⚠ THE HONEST BOUND, AND IT IS THE SAME SHAPE AS PAPER B's",
         ("At withheld states cos to the own target is −0.11948, and a STATE-BLIND CONSTANT beats "
          "the policy there by +0.8893 [+0.6555, +1.0905], winning 14 of 15. The registered "
          "generalisation statistic looked positive (+0.05926) until the UNTRAINED warm start "
          "produced a larger margin (+0.08432) at the same states.",
          {"size": 11.5, "bold": False, "color": INK}),
         ("Licensed sentence: the policy emits where the question landscape was measured.",
          {"size": 11.5, "bold": True, "color": ACCENT_C})],
        size=12.5, bold=True, color=ACCENT_C, space_after=4, line_spacing=1.15)
footer(s, "Sources: docs/results/C0_GEOMETRY.md · C1_FINAL.md (corrected 2026-09-01) · "
          "chapterC_v2.tex §C0, §C1")
notes(s, """WHAT TO SAY
Two experiments on one slide. C0 first: we took the deployed policy — Paper B's distilled E1 — and
captured the continuous vector it emits BEFORE it snaps to a bank question, at every one of 160,000
validation states. It never sits on the question it picks. The maximum cosine over all 160,000 rows is
0.12; not one row is above 0.98.

The mechanism box matters because it is a correction to our own first reading. The raw cosine is
NEGATIVE, and a random direction looks closer, which initially looked damning. The explanation is that
cross-entropy only ever constrains score differences, so the head parked a huge common-mode offset in
the vector — median top score minus eighty-three. Remove the shared component and the emission's cosine
to the entry it selects is +0.488 against a random direction's +0.170. So it is a genuine bank-aligned
direction hiding under an offset.

C1 then trains one object to do two things: run the ordinary tree interview, and emit gain-weighted
blends at its node states. Both blocking gates pass — all 245 distinct questions preserved, 99.5 %
snap agreement, and a 0.97 fit to the blend targets it was taught.

And then the bound at the bottom, which I would rather state than have found. At states it was NOT
taught, it fails — and worse, a state-blind constant beats it there by a mile. The registered
generalisation statistic looked positive until we ran it on the UNTRAINED network and got a bigger
number. Neighbouring states carry correlated targets, so 'own beats a random other' is positive before
any training at all. Same shape as Paper B's P1 and P1b results: first-rate exactly where supervised,
no better than the baseline outside. I think that is a thesis-level regularity, not an isolated
disappointment.

SOURCES
C0: docs/results/C0_GEOMETRY.md; chapterC_v2.tex §C0 (tab:c0main, tab:c0gates, §mechanism).
C1: docs/results/C1_FINAL.md (corrected 2026-09-01, state-blind-constant floor); chapterC_v2.tex §C1
(tab:c1gates, tab:c1final).""")

# --- C4: the mechanism -------------------------------------------------
s, y = slide(prs, "The mechanism: the vector conditions the text",
             "A two-layer adapter maps the direction into eight soft prefix tokens of a FROZEN "
             "distilgpt2. Its twin is trained identically on scrambled pairs.",
             kicker="paper c · c3b · the load-bearing experiment", accent=ACCENT_C)

table(s, [
    ["reading  (item channel, n = 3,670 held-out questions, 10k paired bootstrap)",
     "result", "floor / bar", "cell"],
    ["1.  A − A′  (true adapter minus correspondence-scrambled twin)",
     "+0.1114  [+0.1054, +0.1173]", "CI must exclude 0", "PASS"],
    ["     A's own level", "+0.0953  [+0.0903, +0.1003]",
     "prior −0.0028;  A′ −0.0161", "PASS"],
    ["     A as a share of the GOLD PHRASE's own ceiling (+0.0892)", "107 %",
     "bar was 40 %", "PASS"],
    ["3.  identification of the source vector against a same-channel decoy",
     "0.6778  [0.6631, 0.6931]", "chance 0.500; bar 0.5974; gold phrase 0.6624", "PASS"],
    ["2.  equivalence: A′ must sit within noise of a channel-matched prior",
     "A′ − prior = −0.0133 [−0.0185, −0.0081]", "interval must overlap", "FAIL"],
    ["composite registered label", "", "", "FAILURE — and we keep it"],
], MARGIN_L, y + Inches(0.05), CONTENT_W, col_w=[40, 24, 24, 12],
    font_size=10.8, row_h=0.40, align=[L, R, R, C], first_col_bold=True)

lw = Inches(6.5)
textbox(s, MARGIN_L, Inches(3.95), lw, Inches(0.3),
        "WHY THE FAILURE IS NOT WHAT IT LOOKS LIKE", size=11, bold=True, color=ACCENT_C)
bullets(s, MARGIN_L, Inches(4.30), lw, [
    ("The control guarded against the twin smuggling signal.", "It fired because the "
     "twin sits BELOW knowing nothing — the opposite threat."),
    ("The twin collapsed to 61 distinct texts over 3,880 vectors;", "the true arm "
     "produced 2,280."),
    ("The lesson we take, and state:", "write an equivalence control WITH a direction. "
     "A pre-registration does not get reinterpreted after the data — so the composite "
     "stays FAILURE and the two mechanism criteria are reported as passing on their own terms."),
], size=12)

rx = MARGIN_L + lw + Inches(0.45)
rw = CONTENT_W - lw - Inches(0.45)
textbox(s, rx, Inches(3.95), rw, Inches(0.3),
        "ABOVE CEILING — AND THE CAVEAT THAT TRAVELS WITH IT", size=11, bold=True, color=ACCENT_C)
textbox(s, rx, Inches(4.30), rw, Inches(2.2),
        ["The adapter's text re-encodes BETTER than the true title does, at 107 % of the gold "
         "phrase's own ceiling — while exact-phrase recovery is 0.0000 and token-F1 is 0.111. "
         "It essentially never emits the right title and carries the vector anyway.",
         ("Caveat, ours not a reviewer's: the bridge prefers concept-flavoured text, and invented "
          "titles are freer to be concept-flavoured than real ones — which is the same reason the "
          "exact-phrase floor is zero. “Above ceiling” never travels without this sentence.",
          {"size": 10.5, "bold": False, "color": ACCENT_C}),
         ("Concept channel, printed not gated: A = +0.3533 against a required +0.2002; "
          "identification 0.8476 [0.8000, 0.8952] against a 0.7714 floor.",
          {"size": 10.5, "bold": False, "color": MUTED})],
        size=11.5, color=INK, space_after=6, line_spacing=1.15)
footer(s, "Sources: docs/results/C3B_ADAPTER.md · paperC_lbr.tex tab:english panel (a) · "
          "chapterC_v2.tex :862-905")
notes(s, """WHAT TO SAY
This is the load-bearing experiment of Paper C and the one I would defend in a viva.

The setup: a two-layer MLP maps a latent direction into eight soft prefix tokens of a frozen
distilgpt2, trained on the 19,390 bank entries with a twenty-percent identity-held-out split. The
control is a twin trained identically but on channel-preserving PERMUTED pairs — so it sees the same
vectors and the same texts, just wrongly matched. If the renderer were ignoring its input, the twin
would do just as well.

It does not. The true arm beats the twin by +0.1114 with a tight interval, and — this is the part I
still find surprising — the generated text re-encodes at 107 % of what the TRUE gold phrase earns
through the same bridge. It identifies its own source vector against a decoy 68 % of the time, above
the gold phrase's own 66 %. And it does that while exact-phrase recovery is zero. It essentially never
emits the right title.

The FAILURE row: criterion two was an equivalence test — the twin should sit within noise of a prior
that knows nothing. It does not; it sits BELOW knowing nothing. That is the opposite of the threat we
wrote the control against, and it means the control fired in a direction favourable to us. We keep the
composite FAILURE label anyway, because a pre-registration you reinterpret after seeing the data is
not a pre-registration. The lesson we write down is: write equivalence controls with a direction.

Last thing — the caveat box, top right. The 'above ceiling' number needs its caveat every time it is
quoted: the bridge prefers concept-flavoured text, and invented titles are freer to be
concept-flavoured than real ones.

SOURCES
docs/results/C3B_ADAPTER.md (gate table, arm levels, twin collapse to 61 texts, exact-phrase 0.0000,
token-F1 0.111); paperC_lbr.tex tab:english panel (a); chapterC_v2.tex :862-905.""")


# --- C5: askable questions ------------------------------------------------
s, y = slide(prs, "Making them askable — and what naturalness costs",
             "Retrain the same adapter on question-shaped targets. The mechanism survives. "
             "The instrument's ceiling drops 23 %.",
             kicker="paper c · c3b-v2 · the renderer of record", accent=ACCENT_C)

lw = Inches(7.3)
table(s, [
    ["reading  (item gate channel, n = 3,670)", "best-of-8 (primary)", "greedy", "bar"],
    ["1.  A − A′, tower cosine", "+0.05169", "+0.05855", "CI excl. 0  ✓"],
    ["     95 % CI", "[+0.04595, +0.05750]", "[+0.05321, +0.06383]", ""],
    ["1.  A level  (share of the RE-MEASURED ceiling)", "0.04869  (71 %)", "0.07177  (105 %)",
     "≥ 0.02731  ✓"],
    ["2.  identification", "0.6099", "0.6392", "≥ 0.5790  ✓"],
    ["3.  question-form rate", "0.9471  [0.9398, 0.9542]", "—", "≥ 0.90  ✓"],
    ["concept channel, reported not gated:  A − A′", "+0.28625 [+0.24014, +0.33307]", "—",
     "CI excl. 0  ✓"],
    ["verdict", "SUCCESS", "SUCCESS", ""],
], MARGIN_L, y + Inches(0.05), lw, col_w=[38, 24, 24, 14], font_size=10.5, row_h=0.335,
    align=[L, R, R, C], first_col_bold=True)

rx = MARGIN_L + lw + Inches(0.35)
rw = CONTENT_W - lw - Inches(0.35)
textbox(s, rx, y + Inches(0.05), rw, Inches(0.3),
        "★ NATURALNESS COSTS VERIFIABILITY", size=11, bold=True, color=ACCENT_C)
textbox(s, rx, y + Inches(0.42), rw, Inches(2.6),
        ["Wrapping a phrase in question boilerplate lowers what a PERFECT renderer can score "
         "through the same instrument.",
         ("items   0.08920 → 0.06828   (−23 %)", {"size": 14, "bold": True, "color": ACCENT_C}),
         ("concepts   −17 %", {"size": 14, "bold": True, "color": ACCENT_C}),
         ("", {"size": 4}),
         ("“Have you seen…”, “what did you think of…” are the same words for every question in the "
          "channel. They add surface that carries no information about the direction, so the bridge "
          "reading it back gets a weaker signal from the same content.",
          {"size": 11, "bold": False, "color": MUTED})],
        size=11.5, color=INK, space_after=6, line_spacing=1.15)

textbox(s, rx, Inches(3.55), rw, Inches(2.2),
        ["THE RULE WE NOW FOLLOW",
         ("A change to the output format is a change to the instrument, and the ceiling must be "
          "RE-MEASURED before any gate is read.", {"size": 11.5, "bold": True, "color": INK}),
         ("", {"size": 4}),
         ("Without it, the bare-phrase adapter's 107 % and the question adapter's 71 % look like a "
          "regression. They are not two readings of one number. Nobody in the verbalisation "
          "literature reports a figure of this kind, and the reason is structural: it requires a "
          "same-instrument ceiling, which the field does not measure.",
          {"size": 10.5, "bold": False, "color": MUTED})],
        size=11, bold=True, color=ACCENT_C, space_after=5, line_spacing=1.15)

textbox(s, MARGIN_L, Inches(4.30), lw, Inches(2.3),
        ["94.7 % of held-out generations are well-formed questions.  5.3 % are not — and they are "
         "printed, not trimmed.",
         ("The rule was fixed in advance: ends in a question mark, at least four tokens, contains a "
          "content word that is not template boilerplate. The failures degenerate by repetition — "
          "“Did you enjoy Til, Til, Til, …”. Two priors bracket it and show the question mark is "
          "learned rather than a decoding reflex: the strict all-zero prefix emits a question mark "
          "at rate 0.00, the channel-matched prior at 1.00.",
          {"size": 11, "bold": False, "color": MUTED})],
        size=12.5, bold=True, color=INK, space_after=5, line_spacing=1.15)
footer(s, "Sources: docs/results/C3B_V2_QUESTIONS.md · paperC_lbr.tex tab:english panel (b) · "
          "chapterC_v2.tex :912-1010")
notes(s, """WHAT TO SAY
The bare-phrase adapter emits things like 'science fiction' or 'Rocky (2005)'. An interview asks
QUESTIONS. So we retrained the identical architecture on question-shaped targets — a small fixed
template family per channel, one template sampled per example per epoch so the model learns question
FORM plus this content rather than memorising one string.

Everything survives. The true arm still beats its scrambled twin, identification still clears its bar,
and 94.7 % of held-out generations are well-formed askable questions against a pre-registered 90 %
target. The 5.3 % that fail are printed rather than trimmed, and they fail by repetition.

But the finding on this slide is the box top-right, and it is my favourite small contribution.
Templating the GOLD text — not our generation, the ground truth — lowers what a perfect renderer can
possibly score through this instrument by 23 % on items and 17 % on concepts. The boilerplate is the
same words for every question in the channel, so it adds surface carrying no information about the
direction, and the bridge reading it back gets a weaker signal.

Concretely: reusing the old ceiling would have made the question adapter's 71 % look like a regression
from the bare adapter's 107 %. It is not a regression, it is a format change. So the rule we now
follow, and state as a rule: a change to the output format is a change to the instrument, and the
ceiling must be re-measured before any gate is read. Nobody in the verbalisation literature reports a
number like this, and the reason is structural — it needs a same-instrument ceiling, which is exactly
the calibration the field does not run.

SOURCES
docs/results/C3B_V2_QUESTIONS.md (gate table, 0.9471 [0.9398,0.9542], 0.08920→0.06828, the two
prior question-mark rates); paperC_lbr.tex tab:english panel (b); chapterC_v2.tex :912-1010.""")


# --- C6: triangulation -----------------------------------------------------
s, y = slide(prs, "Triangulation: where the render is distinctive, it lands among the right neighbours",
             "An independent second view — is the rendered word one of the point's own nearest "
             "concepts? Under greedy decoding the answer splits cleanly, and the split is diagnostic.",
             kicker="paper c · c4", accent=ACCENT_C)

lw = Inches(6.2)
table(s, [
    ["greedy decoding, 42 states", "n", "median rank of 1,031"],
    ["the render is a DISTINCTIVE word", "18 / 42", "91"],
    ["the render is a REPEATED word", "24 / 42", "362.5"],
    ["random-tag floor", "—", "340"],
    ["pooled over all 42 states", "42", "313.0"],
], MARGIN_L, y + Inches(0.05), lw, col_w=[52, 20, 28], font_size=11.5, row_h=0.34,
    align=num(3), first_col_bold=True)

textbox(s, MARGIN_L, Inches(3.15), lw, Inches(1.6),
        ["The reading is a generation-DIVERSITY limit, not a geometry failure.",
         ("Where greedy falls back it falls back on three stock words — horror ×9, religion ×7, "
          "movies ×4. Strict top-ten membership is 4/42 against 0/42 for random points, with "
          "overlapping Wilson intervals: a harsher bar that does not separate at this n, and we "
          "report it that way.", {"size": 11, "bold": False, "color": MUTED})],
        size=12.5, bold=True, color=INK, space_after=4, line_spacing=1.15)

rx = MARGIN_L + lw + Inches(0.45)
rw = CONTENT_W - lw - Inches(0.45)
textbox(s, rx, y + Inches(0.05), rw, Inches(0.3),
        "THE BULLSEYES — rank of the rendered word in its own point's ranking over 1,031 concepts",
        size=11, bold=True, color=ACCENT_C)
table(s, [
    ["rendered word", "rank", "what the point was built from"],
    ["action", "1", "three questions in; two unanswered, Apollo 13 disliked"],
    ["art", "4", "an art-cinema blend"],
    ["indictment", "5", "a courtroom blend — Primal Fear, A Time to Kill"],
    ["animals", "7", "the Pixar branch"],
    ["military", "21", ""],
    ["adventure", "28", "a Harry Potter blend"],
    ["science", "31", ""],
    ["noir", "54", ""],
], rx, y + Inches(0.42), rw, col_w=[24, 12, 64], font_size=11, row_h=0.325,
    align=[L, C, L], first_col_bold=True)

textbox(s, rx, Inches(4.25), rw, Inches(2.2),
        ["DECODING IS THE WHOLE LEVER",
         ("Best-of-8 by the frozen LM's own length-normalised likelihood — a rule blind to the "
          "evaluation metric by construction — takes distinct renders from 23 to 37 of 42 and cuts "
          "the translation cost roughly in half, with the faithfulness change not significant "
          "(−0.0193 [−0.097, +0.058]). It is the only Pareto point, and it is the decoding of "
          "record for anything measured.", {"size": 11, "bold": False, "color": MUTED})],
        size=11, bold=True, color=ACCENT_C, space_after=5, line_spacing=1.15)
footer(s, "Sources: docs/results/C4_PILOT.md · C4C_DECODING.md · chapterC_v2.tex :1086-1100 · "
          "paperC_lbr.tex tab:loop")
notes(s, """WHAT TO SAY
An independent second view of the same question. Forget the bridge for a moment: is the rendered word
literally one of that point's own nearest concepts?

Under greedy decoding it splits cleanly, and the split is diagnostic rather than noise. Where the
adapter emits a distinctive word — eighteen of forty-two states — the render lands at median rank 91
out of 1,031, with genuine bullseyes: 'action' at rank one, 'indictment' at five for a courtroom blend
built from Primal Fear and A Time to Kill, 'animals' at seven for the Pixar branch. Where it falls back
on one of three stock words, the median is 362, which is no better than the random-tag floor of 340.

So the failure mode is generation diversity, not geometry. And we can prove that, because changing
ONLY the decoding — same adapter, same bridge, same reference — takes distinct renders from 23 to 37
and halves the translation cost, with the faithfulness change not significant.

Note the harsher bar honestly: strict top-ten membership is 4 of 42 against 0 of 42 for random points,
and the Wilson intervals overlap. That test does not separate at this n and I say so.

SOURCES
Triangulation split and bullseyes: docs/results/C4_PILOT.md; chapterC_v2.tex :1086-1100.
Decoding table: docs/results/C4C_DECODING.md; paperC_lbr.tex tab:loop.
NOTE the label to be careful with: 362.5 is the REPEATED-render group (n=24), not the stock-word group
(n=20); both C4_PILOT.md and C4B_NOISE.md carry a 2026-08-30 correction on exactly this, and the .tex
reproduces the corrected-away wording. This deck uses the corrected label.""")


# --- C7: the hook ------------------------------------------------------------
s, y = slide(prs, "The hook: the nearest catalogue question's own text points the wrong way",
             "This is the motivating sentence turned into a number — and the single most striking "
             "measurement in the paper.",
             kicker="paper c · c4 · the result", accent=ACCENT_C)

bwh, bhh = Inches(3.6), Inches(1.55)
gyh = y + Inches(0.25)
xh0 = MARGIN_L + Inches(0.55)
box(s, xh0, gyh, bwh, bhh,
    [("THE EMITTED POINT", {"size": 12, "color": ACCENT_C}),
     ("the direction the policy wants", {"size": 10.5, "bold": False, "color": MUTED}),
     ("to ask about at this state", {"size": 10.5, "bold": False, "color": MUTED})],
    line=ACCENT_C)
box(s, xh0 + bwh + Inches(1.1), gyh - Inches(0.85), bwh, bhh,
    [("THE NEAREST CATALOGUE QUESTION", {"size": 11.5}),
     ("cos 0.93 in the emitter's own head space", {"size": 10, "bold": False, "color": MUTED}),
     ("its own gold phrase carries the direction at", {"size": 10, "bold": False, "color": MUTED}),
     ("−0.158   [−0.224, −0.089]", {"size": 15, "color": ACCENT_C})],
    line=MUTED)
box(s, xh0 + bwh + Inches(1.1), gyh + Inches(0.95), bwh, bhh,
    [("THE RENDERED OFF-CATALOGUE WORD", {"size": 11.5}),
     ("not in the bank at all", {"size": 10, "bold": False, "color": MUTED}),
     ("carries that same direction at", {"size": 10, "bold": False, "color": MUTED}),
     ("+0.204   [+0.117, +0.291]", {"size": 15, "color": ACCENT})],
    line=ACCENT)
arrow(s, xh0 + bwh, gyh + bhh // 2, xh0 + bwh + Inches(1.1), gyh - Inches(0.85) + bhh // 2,
      color=MUTED)
arrow(s, xh0 + bwh, gyh + bhh // 2, xh0 + bwh + Inches(1.1), gyh + Inches(0.95) + bhh // 2,
      color=ACCENT)

textbox(s, xh0 + 2 * bwh + Inches(1.5), gyh - Inches(0.2), Inches(3.4), Inches(2.4),
        ["A GAP OF", ("+0.362", {"size": 40, "color": ACCENT_C}),
         ("interval excluding zero", {"size": 11, "bold": False, "color": MUTED})],
        size=12, bold=True, color=INK, align=C, space_after=4)

textbox(s, MARGIN_L, Inches(4.35), CONTENT_W, Inches(0.6),
        "The direction the policy wants to ask about is expressible — and the text of the "
        "nearest catalogue question does not express it.",
        size=17, bold=True, color=INK, line_spacing=1.12)

lw = Inches(6.5)
textbox(s, MARGIN_L, Inches(5.20), lw, Inches(1.5),
        ["TWO CAVEATS TRAVEL WITH THIS NUMBER, AND THEY ARE OURS",
         ("The k(φ)-nearest entry is itself a common-mode artefact of an anisotropic space — it is "
          "frequently the SAME entry, “divorce” — so “nearest” does less work than it sounds. And "
          "the bridge favours concept-flavoured text, which is a thumb on the scale for a one-word "
          "render against a bank phrase.", {"size": 10.5, "bold": False, "color": MUTED}),
         ("The SIGN survives both. The magnitude is not to be quoted without them.",
          {"size": 10.5, "bold": True, "color": ACCENT_C})],
        size=11, bold=True, color=ACCENT_C, space_after=4, line_spacing=1.15)

rx = MARGIN_L + lw + Inches(0.45)
rw = CONTENT_W - lw - Inches(0.45)
textbox(s, rx, Inches(5.20), rw, Inches(1.5),
        ["THE REGISTERED DISCRIMINATING READING",
         ("Point-specificity — does a render of THIS point align with THIS point's reference "
          "better than a render of some other point does? +0.28593 [+0.17820, +0.39777], with the "
          "prior floor (the same adapter, latent withheld) at zero: −0.013 [−0.074, +0.053].",
          {"size": 10.5, "bold": False, "color": MUTED})],
        size=11, bold=True, color=ACCENT_C, space_after=4, line_spacing=1.15)
footer(s, "Sources: docs/results/C4_PILOT.md · chapterC_v2.tex :1068-1085 · paperC_lbr.tex §loop")
notes(s, """WHAT TO SAY
This is the slide the paper exists for.

At each interview state we take the bank entry NEAREST the emitted point — cosine 0.93, as close as the
catalogue ever gets. Then we take that entry's OWN gold phrase, the actual words a user would be shown,
and push it through the same independently fitted bridge to ask whether it carries the direction the
point was built from.

It is ANTI-aligned. Minus 0.158, interval clear of zero. The closest question in the catalogue, in
words, points away from what the policy wants to know.

The off-catalogue word our renderer produces carries that same direction at plus 0.204. A gap of 0.362.

So: the direction is expressible in English. It is just not expressible by anything in the bank. That
is the entire motivation for the paper turned into a measurement.

Two caveats I attach here rather than in limitations, because they are ours and not a reviewer's. The
'nearest' entry is partly a common-mode artefact — it is frequently the same entry, the concept
'divorce'. And the bridge prefers concept-flavoured text, which favours a one-word render over a bank
phrase. The SIGN survives both comfortably; the magnitude should never be quoted without them.

SOURCES
docs/results/C4_PILOT.md; chapterC_v2.tex :1068-1085 (including both caveats verbatim);
point-specificity +0.28593 and the prior floor: chapterC_v2.tex :1055-1067.""")

# --- C8: THE TRANSCRIPT (centrepiece) ---------------------------------------
s, y = slide(prs, "The deployment transcript",
             "One cold-start trajectory, six turns. Left: the question the frozen tree would ask. "
             "Right: the question synthesised from the direction the policy emits at that state.",
             kicker="paper c · c4d · the centrepiece", accent=ACCENT_C)

table(s, [
    ["turn", "interview state", "n", "the tree's own question",
     "the SYNTHESISED question"],
    ["1", "root", "10,000", "Do you like better than expected films?",
     "Are you in the mood for something action films?"],
    ["2", "root.dislike", "4,814", "Do you like dark humor films?",
     "Do you like it when a film is suspense?"],
    ["3", "root.dislike²", "3,815", "Do you like alien films?",
     "How do you feel about sci-fi movies?"],
    ["4", "root.dislike³", "2,409", "Have you seen Star Wars: Episode V …?",
     "Do you like it when a film is movie?"],
    ["5", "… .noanswer", "1,335", "Have you seen Crimson Tide (1995)?",
     "Do you like it when a film is detective drama?"],
    ["6", "… .noanswer²", "1,108", "Have you seen Star Wars: Episode VI …?",
     "How do you feel about children films?"],
], MARGIN_L, y + Inches(0.05), CONTENT_W, col_w=[5, 15, 8, 36, 36],
    font_size=11.5, row_h=0.435, align=[C, L, R, L, L], first_col_bold=True)

lw = Inches(6.5)
textbox(s, MARGIN_L, Inches(3.95), lw, Inches(0.3),
        "TURNS 3, 5 AND 6 CARRY THE ARGUMENT", size=11, bold=True, color=ACCENT)
bullets(s, MARGIN_L, Inches(4.30), lw, [
    ("sci-fi movies", "against the tree's “alien films”"),
    ("detective drama", "against the tree's “have you seen Crimson Tide (1995)?”"),
    ("children films", "against the tree's “have you seen Star Wars: Episode VI?”"),
], size=13)
textbox(s, MARGIN_L, Inches(5.85), lw, Inches(0.9),
        "Same neighbourhood, different — and more askable — object. That is the whole "
        "capability claim, in six lines.",
        size=12.5, bold=True, color=INK, line_spacing=1.15)

rx = MARGIN_L + lw + Inches(0.45)
rw = CONTENT_W - lw - Inches(0.45)
textbox(s, rx, Inches(3.95), rw, Inches(0.3),
        "TURNS 2 AND 4 ARE THE TEMPLATING FAILURE", size=11, bold=True, color=ACCENT_C)
textbox(s, rx, Inches(4.30), rw, Inches(2.4),
        ["“Do you like it when a film is movie?” — boilerplate round a nearly contentless phrase; "
         "turn 4's blend is a diffuse movie-business / best-cinematography direction.",
         ("2 of 6 = 33 %, against C3B-v2's measured 5.3 %. Structural, not a contradiction: 5.3 % "
          "was measured on single clean bank entities and the transcript renders BLENDS (median "
          "top-1 weight 0.225).", {"size": 11, "bold": False, "color": MUTED}),
         ("Six turns is a warning, not a number.",
          {"size": 11.5, "bold": True, "color": ACCENT_C})],
        size=11.5, color=INK, space_after=5, line_spacing=1.14)

textbox(s, MARGIN_L, Inches(6.85), CONTENT_W, Inches(0.4),
        "⛔ The artifact's NDCG column — in which the synthesised arm wins 5 of 6 turns — is "
        "declared-circular, unregistered, n=6 on one path. Not a result, never reported as one.",
        size=10.5, bold=True, color=ACCENT_C, line_spacing=1.1)
footer(s, "Source: docs/results/C4D_TRANSCRIPT.md (renderer: v2 question adapter, best-of-8 by LM "
          "likelihood; cohort = all 10,000 validation users, majority path) · paperC_lbr.tex tab:demo")
notes(s, """WHAT TO SAY
This is the centrepiece. Everything before it was a measurement; this is the thing itself.

One cold-start trajectory walked down the majority branch of the frozen tree, all ten thousand
validation users at the root, narrowing to eleven hundred by turn six. Left column: the question the
tree would actually ask. Right column: the question synthesised from the direction the policy emits at
that same state, rendered by the question adapter under best-of-8 decoding.

Turn three is my favourite. The tree asks 'do you like alien films?' — because 'alien' happens to be a
tag in the bank. The synthesised question asks 'how do you feel about sci-fi movies?'. Same
neighbourhood, more askable object. Turn five: the tree asks whether you have seen Crimson Tide, a 1995
submarine thriller most people have not seen. The synthesised question asks about detective drama. Turn
six: Star Wars Episode VI against 'children films'.

And then turns two and four, which I am showing you rather than cropping. 'Do you like it when a film
is movie?' is boilerplate wrapped round a nearly contentless phrase. Two of six is thirty-three
percent, against the 5.3 % malformation rate we measured. That is not a contradiction — the 5.3 % was
measured on single clean bank entities, and the transcript renders BLENDS, where the median top-1
weight is 0.225. A template family fitted to single phrases degrades on mixtures. Six turns is a
warning, not a number, and it is the top of my fix list.

Last line, and I want to be explicit: the artifact does contain an NDCG column where the synthesised
arm wins five of six turns. It is circular, unregistered, n equals six on one path. It is not a result
and I will not report it as one.

SOURCES
docs/results/C4D_TRANSCRIPT.md — all six turns and cohort sizes verbatim; the 33 %-vs-5.3 % analysis;
the ⛔ ruling on the NDCG column; gates (m7dj rebuild bit-identical, 72/72 sweeps, tree_route match
1.0000 on all 1,108 users, leak check 0/10,000, TEST untouched, no LLM API).
Also printed as paperC_lbr.tex tab:demo.""")


# --- C9: machine-only synonyms ------------------------------------------------
s, y = slide(prs, "Machine-only synonyms — and why a plausibility panel cannot grade this",
             "The adapter almost never emits the right title. Its text reproduces the vector's "
             "effect anyway. Both facts are true at once.",
             kicker="paper c · the methodological finding", accent=ACCENT_C)

table(s, [
    ["the source vector's true question is about…", "…and the adapter renders"],
    ["The Shawshank Redemption", "“Have you seen Pulp Fiction (1995)?”"],
    ["Ghost Story (1981)", "“Have you seen Battlestar Galactica (1980)?”"],
    ["Desperate Hours (1955)", "“What did you think of Bunted, The (1941)?”"],
], MARGIN_L, y + Inches(0.05), CONTENT_W, col_w=[42, 58], font_size=13.5, row_h=0.40,
    first_col_bold=True)

textbox(s, MARGIN_L, Inches(3.15), CONTENT_W, Inches(0.6),
        "Neighbouring era, neighbouring register, wrong film — fully fluent, fully askable, "
        "and wrong.",
        size=14, bold=True, color=INK)

lw = Inches(6.5)
textbox(s, MARGIN_L, Inches(3.85), lw, Inches(0.3),
        "THE NUMBERS UNDERNEATH", size=11, bold=True, color=ACCENT_C)
table(s, [
    ["quantity", "value"],
    ["token-F1 between generated content words and the gold phrase", "0.055"],
    ["exact content match", "0.0003"],
    ["identification of its own source vector against a decoy", "0.6099   (chance 0.500)"],
    ["what the TRUE question itself achieves", "0.6316"],
], MARGIN_L, Inches(4.20), lw, col_w=[68, 32], font_size=11, row_h=0.33, align=num(2))

rx = MARGIN_L + lw + Inches(0.45)
rw = CONTENT_W - lw - Inches(0.45)
textbox(s, rx, Inches(3.85), rw, Inches(0.3),
        "WHY THE LOAD-BEARING EVIDENCE IS A CONTROL AND A CEILING, NOT A HUMAN SCORE",
        size=11, bold=True, color=ACCENT_C)
textbox(s, rx, Inches(4.25), rw, Inches(2.5),
        ["A rater with no access to the source vector has one verdict available: fluent, natural, "
         "answerable — and about the wrong film. They mark it wrong. The instrument, which sees the "
         "vector, records that the text reproduces its effect at 61 %. Both are correct.",
         ("The asymmetry runs the other way too, and that is what should worry the field: a renderer "
          "that ignored its input entirely and asked “Have you seen The Godfather (1972)?” every "
          "time would score WELL with raters and at CHANCE with the instrument.",
          {"size": 11, "bold": False, "color": ACCENT_C}),
         ("So a plausibility panel cannot separate our renderer from a confabulator in either "
          "direction, and adding raters does not fix it — they would have to be shown the latent.",
          {"size": 11, "bold": False, "color": MUTED})],
        size=11, color=INK, space_after=5, line_spacing=1.14)

textbox(s, MARGIN_L, Inches(6.35), lw, Inches(0.7),
        "These are Hewitt et al.'s machine-only synonyms appearing unprompted: text a person "
        "would grade wrong that reproduces the vector's semantic effect better than the true "
        "surface form does.",
        size=11.5, color=MUTED, line_spacing=1.15)
footer(s, "Sources: docs/results/C3B_V2_QUESTIONS.md · C3B_ADAPTER.md · chapterC_v2.tex :1015-1042")
notes(s, """WHAT TO SAY
This is the methodological finding, and it is the one I would most like to argue about with a reviewer.

Three held-out examples. The source vector's true question is about The Shawshank Redemption; the
adapter renders 'Have you seen Pulp Fiction?'. Ghost Story from 1981 becomes Battlestar Galactica from
1980. Desperate Hours from 1955 becomes an invented title from 1941. Neighbouring era, neighbouring
register, wrong film — and completely fluent and askable.

The numbers: token overlap with the gold phrase is 0.055, exact content match is three ten-thousandths.
It essentially never says the right thing. And yet its text identifies its own source vector against a
decoy 61 % of the time, where the TRUE question manages 63 %.

Now the argument I want to make. A human rater shown 'Have you seen Pulp Fiction?' with no access to
the source vector has exactly one verdict available: it is a good question about the wrong film. They
mark it wrong. The instrument, which does see the vector, records that the text carries the vector's
effect. Both are right; they are measuring different things.

And here is the part that should worry the field: the asymmetry runs the other way too. A renderer that
ignored its input completely and asked 'have you seen The Godfather?' every single time would score
beautifully with raters and at chance with the instrument. So a plausibility panel cannot separate our
renderer from a confabulator in EITHER direction, and hiring more raters does not fix it — you would
have to show them the latent, at which point they are running our bridge by hand and worse.

That is why the load-bearing evidence in this paper is a control and a ceiling rather than a human
score. It is a measured reason, not a budget excuse. It is also, to be fair, the strongest argument
FOR the one human study I do want: on answerability, not on quality.

SOURCES
Three examples and the numbers: docs/results/C3B_V2_QUESTIONS.md (held-out, best-of-8; a fourth
example exists — Time Traveler's Wife → My Little Pony — the paper prints three).
Bare-phrase precursors (Rocky Balboa → 'Rocky (2005)', Zelig → 'Theatrical Man (1962)'):
C3B_ADAPTER.md. The panel argument verbatim: chapterC_v2.tex :1015-1042.
Machine-only synonyms concept: Hewitt et al. 2025 (neologism), cited in the chapter.""")


# --- C10: C status ------------------------------------------------------------
s, y = slide(prs, "Paper C: where it stands",
             "A capability paper with no benchmark to beat. The LBR draft is written; publication "
             "is a bonus, not a requirement.",
             kicker="paper c · status", accent=ACCENT_C)

lw = Inches(7.1)
table(s, [
    ["", ""],
    ["State", "Six experiments run and banked; the short-form draft is written "
     "(new_chapters/paperC_lbr/, dated September 2026)"],
    ["Role", "REDEFINED 2026-08-18: absorbs old Paper E. A capability paper — fun techniques, "
     "no benchmark to beat, ideally truly original. PEBOL-shaped."],
    ["Author's steer", "“a sexy LLM piece to top off the thesis… not load-bearing… interesting "
     "stuff and less lengthy field audits.” The spine is the DEMO; the 12-system audit shrinks "
     "to one related-work paragraph plus one control."],
    ["Venue", "opportunistic — no deadline pressure"],
    ["Cost to a submittable draft", "~3–4 focused days of work plus 2–3 machine-nights, "
     "plus an author pass"],
    ["Also lands as", "the thesis capstone chapter (chapterC_v2), which is longer and keeps "
     "the audit"],
], MARGIN_L, y + Inches(0.05), lw, col_w=[24, 76], font_size=11.2, row_h=0.52,
    first_col_bold=True, header_rows=0)

rx = MARGIN_L + lw + Inches(0.40)
rw = CONTENT_W - lw - Inches(0.40)
textbox(s, rx, y + Inches(0.05), rw, Inches(0.3),
        "WHAT IS ESTABLISHED", size=11, bold=True, color=ACCENT_C)
bullets(s, rx, y + Inches(0.42), rw, [
    "The policy emits off-catalogue directions where the question landscape was measured.",
    "A trained renderer says them in English, and 94.7 % are well-formed questions.",
    "The text — unlike text from a scrambled latent — reproduces the direction's causal effect.",
    "The rendered word lands in the direction's own neighbourhood where the nearest "
    "catalogue question's text points the other way.",
], size=11.5, marker_color=ACCENT)

textbox(s, rx, Inches(4.15), rw, Inches(2.5),
        ["WHAT MAY NOT BE CLAIMED",
         ("· that the policy emits sensibly at ARBITRARY interview states — C1's withheld-subtree "
          "test says it does not, and a state-blind constant beats it there;",
          {"size": 10.5, "bold": False, "color": MUTED}),
         ("· that the blend's five ingredients are the right five — they are unstable draws from a "
          "near-flat gain profile;", {"size": 10.5, "bold": False, "color": MUTED}),
         ("· ⛔ that asking a synthesised question improves anyone's recommendations. No version of "
          "that sentence appears anywhere.", {"size": 10.5, "bold": True, "color": ACCENT_C})],
        size=11, bold=True, color=ACCENT_C, space_after=5, line_spacing=1.15)
footer(s, "Sources: docs/PUBLICATION_PLAN.md roster · "
          "docs/design/PAPER_C_PUBLICATION_ASSESSMENT_2026-08-25.md · author steer 2026-08-30")
notes(s, """WHAT TO SAY
Where C stands. Six experiments run, all banked, and the short-form draft is written.

The role changed in August: C absorbs the old Paper E, and it is explicitly a capability paper rather
than a benchmark paper. My own steer was that this should be the fun piece that tops off the thesis —
interesting techniques rather than another lengthy field audit — so the spine is the demo and the
twelve-system audit shrinks to a related-work paragraph plus one control.

Venue is opportunistic. There is no deadline pressure on it, and publishable is a bonus rather than a
requirement — but the assessment says a submittable draft is about three or four days of work plus a
couple of machine-nights, which is cheap.

The two lists on the right are the scope. What is established is the four-part conjunction. What may
not be claimed is the three-part list underneath, and the red one is absolute: no version of 'asking a
synthesised question improves recommendations' appears anywhere in this work.

SOURCES
docs/PUBLICATION_PLAN.md roster table (Paper C redefined, opportunistic);
docs/design/PAPER_C_PUBLICATION_ASSESSMENT_2026-08-25.md (cost estimate, venue thought);
author steer 2026-08-30 (memory: paperC-is-the-fun-capstone-not-an-audit-paper);
scope clauses: chapterC_v2.tex §C1 'What is established, and what may not be claimed'.""")

# =====================================================================
# CLOSE
# =====================================================================

# --- Z1: the thesis claim ------------------------------------------------
s, y = slide(prs, "The thesis, in one claim",
             "", kicker="close")

textbox(s, MARGIN_L, y + Inches(0.05), CONTENT_W, Inches(1.5),
        ["A cold interview is limited by what the user can answer and by what the system "
         "is able to ask —",
         "not by how cleverly it chooses from a fixed list."],
        size=25, bold=True, color=INK, line_spacing=1.18, space_after=4)

table(s, [
    ["", "the claim", "the evidence, on 10,000 held-out TEST users"],
    ["A", "An elicitation result is a property of a strategy AND an instrument, so the "
     "instrument must be certified first.",
     "the same two strategies invert their ordering when the consumer changes; "
     "the tower ties RecVAE on tail and the ruler snaps to four published numbers"],
    ["B", "The binding constraint is the criterion and the channel, not the "
     "conditioning.",
     "21 selection rules on one ruler: a static order beats every adaptive arm in 138 of 138 "
     "cells · changing only the split criterion buys +0.01035 full / +0.00705 tail · "
     "changing the BANK buys 3× more tail than conditioning does"],
    ["", "And answerability, not informativeness, orders the channels.",
     "16 item questions collect 3.11 answers; 16 concept questions collect 12.36 · answering "
     "at all is worth ~+0.25 against a best-question effect of +0.048"],
    ["C", "The action space can be opened: a direction outside the bank can be emitted, "
     "said in English, and audited.",
     "245/245 questions preserved while emitting blends · the vector conditions the text at "
     "+0.1114 · 94.7 % askable · the nearest catalogue question's own text is anti-aligned "
     "at −0.158 where the rendered word carries it at +0.204"],
], MARGIN_L, Inches(3.20), CONTENT_W, col_w=[4, 34, 62], font_size=11, row_h=0.72,
    first_col_bold=True)

textbox(s, MARGIN_L, Inches(6.65), CONTENT_W, Inches(0.5),
        "And the method that produced all of it: pre-register the reading rule, then run the "
        "rule against its own null before reading the effect.",
        size=13, bold=True, color=ACCENT, line_spacing=1.1)
footer(s, "Sources: as per each paper's section")
notes(s, """WHAT TO SAY
If you take one sentence away, take the top one. A cold interview is limited by what the user can
answer and by what the system is able to ask — not by how cleverly it picks from a fixed list. Every
result in the three papers is a version of that.

A says you cannot measure any of this without certifying the instrument, and demonstrates the point by
showing the verdict flip when the consumer changes. B says the binding constraints are the criterion
and the channel: a static order beats every adaptive arm in 138 of 138 cells; changing only the split
criterion buys about a point of NDCG; and changing what you ask about buys three times more tail than
changing how you decide. C says the action space itself can be opened — a direction that is not any
question in the bank can be emitted, said, and audited.

And the last line is the methodological thread I would want on my viva record: pre-register the
reading rule, and then run that rule against its own null before you look at the effect. Doing that
cost me three headlines and it is the reason I trust what is left.

SOURCES
Each claim traces to its own section; see the per-slide notes for A, B and C.""")


# --- Z2: publication plan as options ------------------------------------
s, y = slide(prs, "Publication plan — as options",
             "Roster reset 2026-08-18. One deadline binds; nothing else does.",
             kicker="close · decisions I would like your view on")

table(s, [
    ["", "state", "OPTION 1 (preferred)", "OPTION 2", "OPTION 3"],
    ["A  instrument", "closed, 41 pp, unsubmitted",
     "TORS — rolling, no deadline, no travel; length is an asset; the one venue every "
     "draft in the blind panel cleared",
     "RecSys or ECIR reproducibility track — our unique win in the panel",
     "hold until the human round-trip study exists, then submit stronger"],
    ["B  interview", "chapter CLOSED 2026-08-30; test look spent and replicates",
     "★ ECIR 2027 full paper, deadline 2 Oct 2026 — sections 1–3 cut to ~12 pp "
     "(audit · diagnosis · one-variable demonstration)",
     "RecSys, or its reproducibility track",
     "add the RecVAE re-scoring of the frozen tree first, to kill the "
     "one-consumer objection"],
    ["B′  Act 5", "held back, not cut",
     "a SECOND paper — RecSys or CIKM — on the Reddit arc, the opener-fitted trees "
     "and the one distilled policy", "fold into the thesis chapter only", "—"],
    ["C  off-catalogue", "six experiments banked; LBR draft written; no deadline pressure",
     "RecSys LBR — the demo-led short form the draft is already written as",
     "IntRS or KaRS workshop", "arXiv + thesis capstone chapter, and stop there"],
    ["D  free-recall", "ABSORBED INTO B — the Reddit arc and opener census cover its "
     "substance; its own numbers were on the retired pre-July ruler", "—", "—", "—"],
    ["E  LLM dialogue", "MERGED INTO C", "—", "—", "—"],
], MARGIN_L, y + Inches(0.02), CONTENT_W, col_w=[11, 19, 28, 22, 20],
    font_size=9.6, row_h=0.72, first_col_bold=True)

textbox(s, MARGIN_L, Inches(6.15), CONTENT_W, Inches(1.0),
        ["The only hard date on this slide is 2 October 2026, and it is Paper B's.",
         ("Standing constraints: solo, roughly a year left, NO TRAVEL — which is most of why TORS "
          "and short/workshop tracks dominate the options. Bibliography completeness (~8 "
          "load-bearing works still missing) blocks any submission.",
          {"size": 11.5, "bold": False, "color": MUTED})],
        size=14, bold=True, color=INK, space_after=4, line_spacing=1.14)
footer(s, "Sources: docs/PUBLICATION_PLAN.md (roster reset 2026-08-18, TORS rationale, blind panel) · "
          "docs/design/PAPER_B_MASTER_TODO.md · PAPER_C_PUBLICATION_ASSESSMENT_2026-08-25.md")
notes(s, """WHAT TO SAY
This is the slide I would most like your view on, and I have written it as options rather than a plan.

Paper A: my preference is TORS. Rolling submission so nothing forces the timeline, no travel, and the
length that got it rejected from ECIR is unremarkable there. It was also the one venue that all three
drafts in my blind calibration panel cleared. Option two is a reproducibility track, which is the one
place we uniquely won. Option three is to hold it until the human study exists — I do not favour that,
because 'under review at TORS' is citable now.

Paper B is the one with a date: ECIR 2027, second of October. The cut is sections one to three only.
The thing I would like to do first is option three — score the already-frozen tree through RecVAE, one
machine-night, which removes the strongest objection I can see.

Act 5 I want to hold back as a second paper rather than crush into B. That is a judgement call and I
would like a second opinion on it.

Paper C has no deadline and is explicitly a bonus. The draft is already written in short form, so
RecSys LBR is the natural home, with workshops or just arXiv-plus-chapter as fallbacks.

D and E are gone as separate papers — D absorbed into B, E merged into C.

Two standing constraints frame all of this: no travel, and the bibliography is incomplete — about
eight load-bearing works are still missing and that blocks any submission.

SOURCES
docs/PUBLICATION_PLAN.md :11-19 (venue targets), :21-34 (TORS rationale), :36-55 (blind panel),
:66-92 (roster reset and what survives from D), :96-99 (blockers).
docs/design/PAPER_B_MASTER_TODO.md (the conference cut, Act 5 as a second paper, the one-consumer
objection). docs/design/PAPER_C_PUBLICATION_ASSESSMENT_2026-08-25.md (venue thought, cost).""")


# --- Z3: what's next ------------------------------------------------------
s, y = slide(prs, "What's next",
             "Ordered by what unblocks the most, not by what is most interesting.",
             kicker="close")

table(s, [
    ["", "the work", "why", "cost"],
    ["1", "Score — not refit — Paper B's frozen NDCG tree through RecVAE",
     "kills the strongest objection to Act 3 (“it rests on one consumer”), which our own flip "
     "result makes fair", "one machine-night"],
    ["2", "The ECIR cut: sections 1–3 of Chapter B to ~12 pp",
     "the only hard deadline on the roster — 2 October 2026", "the binding task"],
    ["3", "Close the bibliography (~8 load-bearing works)",
     "blocks every submission, A included", "days"],
    ["4", "Paper C short-form finish: shrink the audit, lead with the transcript",
     "the draft exists; the assessment costs it at 3–4 days plus 2–3 machine-nights",
     "cheap, no deadline"],
    ["5", "Fix the transcript's templating failure (2 of 6 malformed on blends)",
     "the template family was fitted to single phrases and degrades on mixtures — the only "
     "visible defect in C's centrepiece", "small"],
    ["6", "★ The human round-trip study (gate C3)",
     "never run; owed project-wide; it is the one thing blocking the strongest claims across "
     "A, B and C — and all three top objections in the blind panel were simulator circularity. "
     "C argues for a study on ANSWERABILITY rather than on quality.",
     "the open question for this meeting"],
], MARGIN_L, y + Inches(0.05), CONTENT_W, col_w=[4, 33, 47, 16], font_size=10.8, row_h=0.68,
    first_col_bold=True)

textbox(s, MARGIN_L, Inches(5.55), CONTENT_W, Inches(1.4),
        ["Two things I would like to decide today.",
         ("(1) Is Act 5 a second paper or a chapter section?   "
          "(2) Is the human round-trip study worth the time it costs, and if so, is "
          "answerability the right thing to test — or does it need to be quality?",
          {"size": 13, "bold": False, "color": MUTED})],
        size=15, bold=True, color=INK, space_after=5, line_spacing=1.16)
footer(s, "Sources: docs/design/PAPER_B_MASTER_TODO.md · docs/PUBLICATION_PLAN.md :96-99 · "
          "GATE_BATTERY_INSTRUMENT.md :106 · docs/HARSH_REVIEW_POINTS.md")
notes(s, """WHAT TO SAY
What's next, ordered by what unblocks the most rather than by what I most want to do.

One and two are the ECIR path: re-score the frozen tree through RecVAE to kill the one-consumer
objection, then do the twelve-page cut. Three is the bibliography, which blocks everything including
Paper A. Four and five are Paper C, which is cheap and has no deadline.

Six is the real question, and it is the one I would like your judgement on. The human round-trip study
has never been run. It is a standing gate — C3 in the battery — it is owed project-wide, and it is the
single thing blocking the strongest claims in all three papers. In the blind reviewer panel, all three
papers' top objection was simulator circularity, which is exactly what it answers.

Paper C actually gives an argument about what shape it should take: a plausibility panel cannot grade
a rendered question, for the reasons on the machine-only-synonyms slide. So the study I would argue for
is on ANSWERABILITY — can a real person answer this question at all — rather than on quality. That is
cheaper and it is the thing our simulator is actually assuming.

So the two decisions I would like today: is Act 5 a second paper or a chapter section, and is the human
study worth the time — and if so, answerability or quality?

SOURCES
Item 1 (the one-consumer objection and its fix): docs/design/PAPER_B_MASTER_TODO.md.
Item 3 (~8 missing works): docs/PUBLICATION_PLAN.md :96-99.
Item 5: docs/results/C4D_TRANSCRIPT.md (33 % vs 5.3 %).
Item 6: GATE_BATTERY_INSTRUMENT.md :106 (C3 owed, holdout design mandatory, satisfaction survey
explicitly not acceptable); PUBLICATION_PLAN.md :36-55 (all three top objections were circularity);
chapterC_v2.tex limitations (the one human study we argue FOR, on answerability).""")

# =====================================================================
# APPENDIX
# =====================================================================

sd = divider("Appendix",
             "Detailed per-experiment tables, methods detail, and the honest bounds. "
             "Not shown by default — these exist so a probing question gets an instant answer.",
             MUTED, "13 slides  ·  not part of the ~50 minutes")
notes(sd, """WHAT TO SAY
Section break — do not present. Jump here if a specific number is queried. A1-A2 cover Paper A's
gate ledger and concept channel; B1-B4 cover Paper B's diagnostics, the group arm, the routing null
and the distilled policy; C1-C3 cover Paper C's geometry, emission and translation cost; M1-M4 cover
methods, the honest bounds, the not-built ledger and a one-page number index.

SOURCES
Each appendix slide names its own source docs in its footer and notes.""")


# --- APP 1: A gate battery, the open items ---------------------------------
s, y = slide(prs, "Appendix A1 — Paper A: gates that are open, dropped, or unmeasurable",
             "The battery's own failure ledger. Every per-build gate passes; these are the "
             "qualifications.",
             kicker="appendix · paper a", accent=MUTED)
table(s, [
    ["item", "status", "detail"],
    ["G4 confidence ORDERING sub-test", "unmeasurable by design",
     "no ground truth for user confidence on MovieLens. The refusal-boundedness gate itself "
     "passes at 0.018/27.66 = 0.0007 against a ≤0.05 bar. A Jul-24 'G4 FAIL' was "
     "re-dispositioned."],
    ["G1c calibration sub-gate", "DROPPED, not failed",
     "measured Spearman(σ, held-out NLL) = 0.2393 against a 0.25 bar — but the POPULARITY of the "
     "held-out targets predicts the same NLL at ρ = −0.6964, three times better, and profile size "
     "at 0.5195. The gate is ill-posed. G1a and G1b pass: total posterior variance "
     "1881.9 → 1805.1; directional shrinkage 11.5× against a 2× bar."],
    ["Volume-leak gate (KT-A3)", "FAILS, pending author ruling",
     "ridge R² from the fold latent at q8 to log user volume: clip-up 0.02365 → signed 0.24928, "
     "against a pre-registered ≤0.05 bar"],
    ["G5 split", "open",
     "trained concept operators learn 'what X-likers watch' rather than 'members of X': "
     "member-AUC 0.630 at k=1, discrimination Spearman 0.948 against log-popularity but 0.030 "
     "against member-ness. The popularity-projection control PASSES at +0.0087, CI clean."],
    ["C3 human round-trip", "NEVER RUN", "owed project-wide; holdout design mandatory; a "
     "satisfaction survey is explicitly not acceptable"],
    ["Chapter internal inconsistency", "flagged",
     "the conjugate accumulator's uniformity cost is 'roughly 0.12 full NDCG' at :881 and "
     "'0.15' at :803. Quote 0.12, from the results section."],
], MARGIN_L, y + Inches(0.02), CONTENT_W, col_w=[19, 15, 66], font_size=9.6, row_h=0.75)
footer(s, "Sources: GATE_BATTERY_INSTRUMENT.md · G1C_CONTROL_RESULT.md · CONCEPT_CHANNEL_RESULT.md "
          ":103-108, :164-167 · docs/EXPERIMENTS.md :17 · chapterA_v2.tex :803, :881")
notes(s, """WHAT TO SAY (only if asked)
The failure ledger for Paper A's battery. Every per-build gate passes; these six items are the
qualifications I would volunteer rather than let a reviewer find.

The one that is genuinely open is the volume-leak gate: the fold latent at eight questions predicts
the log of the user's rating volume at R² 0.249 against a pre-registered 0.05 bar. It means the
latent is partly encoding 'how much this person rates' rather than 'what they like'. It is awaiting a
ruling and it is on the honest-bounds list.

SOURCES as footed.""")


# --- APP 2: A concept-channel ledger ---------------------------------------
s, y = slide(prs, "Appendix A2 — Paper A: concept channel, fold operator, and the ledgers "
                  "that must not be mixed",
             "", kicker="appendix · paper a", accent=MUTED)
table(s, [
    ["quantity", "value"],
    ["Fold-operator ablation (old tower): additive union — member AUC / full@8 answers",
     "0.820 / 0.0991  (BELOW the 0.1279 intercept — worse than asking nothing)"],
    ["Fold-operator ablation: gated fold — member AUC / full@8 / tail@8",
     "0.626 / 0.1759 / 0.0841  (up from 0.1322 / 0.0503 at one answer)"],
    ["Redundancy robustness: 8 correlated top-SEL concepts vs 4",
     "0.1685/0.0906  vs  0.1572/0.0777 — must not decline; the additive operator fails here "
     "(m8 0.0991 vs m4 0.1349, 'the correlation crater')"],
    ["Whitening: raw centroid member-AUC → whitened", "0.44 (below chance) → 0.94"],
    ["Open-vocabulary: a held-out concept direction predicted from its NAME alone",
     "top-10 overlap 0.167 vs random 0.0005 (>300× chance); cosine 0.638 to the true direction"],
    ["Concept specificity (R2 row): like→member AUC / dislike / other concept",
     "0.626 / 0.416 / 0.498 (chance); sign contrast +0.2096 [+0.1986,+0.2208]; "
     "identity contrast +0.1278 [+0.1195,+0.1361]"],
    ["Answer-model decomposition: selection oracle over the polarisation bank at q8",
     "+0.0691 [0.0656, 0.0726]; the utility oracle sits +0.1381 above behavioural SEL; "
     "answers collected per q8 — concept 7.93 vs item 3.03; no imputer beats SEL"],
    ["⚠ THREE LEDGERS THAT MUST NOT BE MIXED ON A SLIDE",
     "(1) CONCEPT_CHANNEL_RESULT.md's strategy×channel suite is on the SUPERSEDED i25 fold and "
     "its 'items win the long interview' conclusion was RETRACTED. "
     "(2) GREEDY_BEST_STATIC_I26C.md is policy A (intercept .1314/.0234). "
     "(3) MASK_POLICY_B_MIGRATION.md is policy B (intercept .15855/.02819) — the one this deck uses."],
], MARGIN_L, y + Inches(0.02), CONTENT_W, col_w=[38, 62], font_size=9.8, row_h=0.66)
footer(s, "Sources: chapterA_v2.tex :305-308, :941-947, :1094-1116, :1473-1499 · "
          "ANSWER_DECOMPOSITION_RESULT.md :85-90 · MASK_POLICY_B_MIGRATION.md :108-127")
notes(s, """WHAT TO SAY (only if asked)
Detail behind the concept-channel and fold-in slides. The last row is a note to myself as much as
anything: there are three different ledgers of strategy-by-channel numbers in this project, on three
different folds, and the first one's headline conclusion was retracted. The deck uses policy B
throughout, which is why the cold intercept reads 0.15855.

SOURCES as footed.""")


# --- APP 3: B Act 1 branching + the flip -----------------------------------
s, y = slide(prs, "Appendix B1 — Paper B: how adaptive were the adaptive selectors, and the flip",
             "", kicker="appendix · paper b", accent=MUTED)
lw = Inches(6.6)
textbox(s, MARGIN_L, y + Inches(0.02), lw, Inches(0.3),
        "DISTINCT ASK-PREFIXES OVER 10,000 USERS", size=11, bold=True, color=ACCENT_B)
table(s, [
    ["selector", "k=1", "k=2", "k=4", "k=8", "k=16"],
    ["static (entropy₀)", "1", "1", "1", "1", "1"],
    ["GLM-UCB", "1", "3", "23", "335", "5,739"],
    ["Thompson", "759", "8,083", "10,000", "10,000", "10,000"],
    ["latent-only info-gain†", "1", "2", "7", "99", "1,993"],
    ["EDDI three-outcome‡", "1", "2", "8", "112", "1,217"],
], MARGIN_L, y + Inches(0.37), lw, col_w=[30, 14, 14, 14, 14, 14], font_size=11, row_h=0.315,
    align=num(6), first_col_bold=True)
textbox(s, MARGIN_L, Inches(3.05), lw, Inches(1.2),
        "Thompson is fully personalised from the first question and still loses. So the Act 1 "
        "result is not 'the adaptive arms failed to be adaptive' — they were adaptive, and it "
        "did not help.",
        size=12, color=MUTED, line_spacing=1.15)
textbox(s, MARGIN_L, Inches(4.20), lw, Inches(0.3),
        "THREE-OUTCOME vs TWO-OUTCOME RESPONSE MODEL (paired, in-run)",
        size=11, bold=True, color=ACCENT_B)
table(s, [
    ["k", "1", "2", "4", "8", "16"],
    ["full gain", "+.0070", "+.0057", "+.0114", "+.0122", "+.0124"],
], MARGIN_L, Inches(4.55), lw, col_w=[30, 14, 14, 14, 14, 14], font_size=11, row_h=0.315,
    align=num(6), first_col_bold=True)
textbox(s, MARGIN_L, Inches(5.30), lw, Inches(1.3),
        "All five intervals exclude zero; tail separates at 4 of 5 budgets (k=2 is a null, "
        "+.0015 [−.0001,+.0031]). It closes ~¼ of the full deficit (−.0530 → −.0406) and ~⅓ of "
        "the tail (−.0268 → −.0183) — real, and nowhere near enough. EDDI's native validation "
        "FAILED: pooled average rank 3.402 against a random ordering's 3.236.",
        size=11, color=MUTED, line_spacing=1.15)

rx = MARGIN_L + lw + Inches(0.40)
rw = CONTENT_W - lw - Inches(0.40)
textbox(s, rx, y + Inches(0.02), rw, Inches(0.3),
        "THE FIFTH CONSUMER — WHERE THE VERDICT FLIPS", size=11, bold=True, color=ACCENT_C)
table(s, [
    ["Golbandi leaf model as consumer", "k=1", "k=2", "k=4", "k=8"],
    ["Thompson − static, full", "+.0100", "+.0171", "+.0148", "+.0281"],
], rx, y + Inches(0.37), rw, col_w=[40, 15, 15, 15, 15], font_size=10.5, row_h=0.34,
    align=num(5), first_col_bold=True)
textbox(s, rx, Inches(2.30), rw, Inches(2.3),
        ["Every interval excludes zero. Tail is split: Thompson separates AGAINST static at k=4 "
         "(−.0064 [−.0095,−.0033]) and for it at 1, 2 and 8. Latent-only info-gain at k=8: "
         "+.0131 full / −.0034 tail, both separating. GLM-UCB does not participate — +.0007 full, "
         "+.0008 tail, neither separating.",
         ("This is the single most important methodological result in the thesis and it is why "
          "Paper A exists.", {"size": 11, "bold": True, "color": ACCENT_C})],
        size=11, color=MUTED, space_after=6, line_spacing=1.15)
textbox(s, rx, Inches(4.85), rw, Inches(1.8),
        ["THE PREPRINT ARTEFACT, REBUILT",
         ("A published adaptive-degradation effect: switching their candidate rule off makes the "
          "decline STEEPER (−46 % vs −14 %), and under an arm-independent held-out split the "
          "adaptive arm rises monotonically .1637 → .1761 → .1857 → .1936. Their popularity "
          "control declines −.012 against the adaptive arm's −.045 at a matched answer rate.",
          {"size": 10.5, "bold": False, "color": MUTED})],
        size=11, bold=True, color=ACCENT_B, space_after=5, line_spacing=1.15)
footer(s, "Sources: chapterB_v2.tex tab:prefixes (:810-826), :834-930, :932-945, :517-539")
notes(s, """WHAT TO SAY (only if asked)
Three things. First, the branching table is the pre-emptive answer to 'maybe your adaptive selectors
were not actually adaptive' — Thompson emits a distinct prefix for 759 of 10,000 users at the FIRST
question and is fully personalised by the fourth. They were adaptive. It did not help.

Second, the three-outcome response model. Letting the selector model 'did not answer' as a real
outcome rather than a missing value is worth about +0.012 full at sixteen questions, closes a quarter
of the deficit, and is nowhere near enough to change the verdict. Worth noting EDDI's native
validation failed on its own protocol.

Third, the flip. On a Golbandi leaf-model consumer the entire Act 1 ordering reverses.

SOURCES as footed.""")


# --- APP 4: B group-conditioned greedy --------------------------------------
s, y = slide(prs, "Appendix B2 — Paper B: the group-conditioned greedy arm, and what "
                  "validation overstated",
             "", kicker="appendix · paper b", accent=MUTED)
lw = Inches(6.9)
table(s, [
    ["genre-conditioned greedy, K=15, TEST", "full", "tail"],
    ["fallback-200 arm", "+0.00862 [+0.00595, +0.01131]", "+0.00591 [+0.00326, +0.00862]"],
    ["all-groups arm", "+0.01015", "+0.00719"],
    ["rebuilt global baseline", "0.24295", "— (≈3.5 % relative gain)"],
    ["Drama (54.8 % of the cohort) — a NULL", "+0.00150 [−0.00187, +0.00486]", ""],
    ["tail at K=8 — also a null", "", "+0.00203 [−0.00073, +0.00482]"],
    ["validation overstates by", "1.80×", "1.02×"],
], MARGIN_L, y + Inches(0.05), lw, col_w=[38, 32, 30], font_size=10.8, row_h=0.36,
    align=num(3), first_col_bold=True)
textbox(s, MARGIN_L, Inches(3.30), lw, Inches(1.9),
        ["The shape of the result matters more than the headline.",
         ("Sci-Fi, Adventure, Thriller and Action gain +0.017 to +0.090. Drama — the majority "
          "group at 5,481 of 10,000 — is a null. Fantasy, at n=11, is −0.116 on test. So the "
          "pooled gain is carried entirely by the minority groups, and a system that only served "
          "Drama users would see nothing at all. Genre distribution: Drama 5,481 / Action 1,582 / "
          "Comedy 1,475; 453 test users routed back by fallback-200.",
          {"size": 11, "bold": False, "color": MUTED})],
        size=12.5, bold=True, color=INK, space_after=5, line_spacing=1.15)
rx = MARGIN_L + lw + Inches(0.40)
rw = CONTENT_W - lw - Inches(0.40)
textbox(s, rx, y + Inches(0.05), rw, Inches(0.3),
        "TREE DEPTH IS THE STOPPING RULE, NOT THE DATA", size=11, bold=True, color=ACCENT_B)
textbox(s, rx, y + Inches(0.42), rw, Inches(3.0),
        ["Admissible candidates fall from 876 of 1,000 at the root to ZERO at n=660. The "
         "answer-rate window [400/n, 1−200/n] closes at n=600.",
         ("Of the 15 stopped nodes: 5 stop on size at n<400, 3 on size at 400≤n<600, 7 on shape, "
          "and ZERO for want of a positive-gain split. Every stop had a gain of +0.00672 to "
          "+0.04251 in hand.", {"size": 10.5, "bold": False, "color": MUTED}),
         ("8 of 15 binding children are the did-not-answer branch of a concept that 92.7–100 % of "
          "the node could answer. At the root the EXCLUDED candidates are MORE answerable than "
          "the surviving ones (median 0.986 vs 0.309).",
          {"size": 10.5, "bold": False, "color": MUTED}),
         ("So the tree's depth measures MIN_LEAF, not the information in the data.",
          {"size": 11, "bold": True, "color": ACCENT_C})],
        size=11.5, color=INK, space_after=6, line_spacing=1.15)
textbox(s, rx, Inches(4.55), rw, Inches(2.1),
        ["OVERFITTING, AND WHERE THE TEST GAIN COMES FROM",
         ("Six of fifteen leaves LOSE on test, covering 39.3 % of users (3,933), against none on "
          "validation. Two crossed to negative when the comparator was rebuilt. Largest positive "
          "leaf gain +0.0584. The overfitting signature does not survive: leaf-size↔gain Spearman "
          "−0.48 (p=0.071) on validation, +0.10 (p=0.71) on test.",
          {"size": 10.5, "bold": False, "color": MUTED})],
        size=11, bold=True, color=ACCENT_B, space_after=5, line_spacing=1.15)
footer(s, "Sources: chapterB_v2.tex :956-1001, :1366-1432 · docs/results/CONDITIONED_ARMS_ON_TEST.md")
notes(s, """WHAT TO SAY (only if asked)
Two things. The group-conditioned arm was the first to beat the strong static comparator, but its gain
is carried entirely by minority genres — Drama, which is more than half the cohort, is a flat null.
That is a real limitation on deployment value.

And the depth analysis: the tree is depth five because MIN_LEAF is 200, not because the data ran out.
Zero of the fifteen stopped nodes stopped for want of a positive-gain split; every one had between
0.007 and 0.043 of gain sitting there unclaimed. The depth is measuring our stopping rule.

SOURCES as footed.""")


# --- APP 5: B routing null + Reddit groups ----------------------------------
s, y = slide(prs, "Appendix B3 — Paper B: routing is a null, and the Reddit groups",
             "", kicker="appendix · paper b", accent=MUTED)
lw = Inches(6.3)
textbox(s, MARGIN_L, y + Inches(0.02), lw, Inches(0.3),
        "ROUTING SPECIALISATION IS A NULL", size=11, bold=True, color=ACCENT_C)
bullets(s, MARGIN_L, y + Inches(0.38), lw, [
    ("105 leaf pairs; median free-pick overlap 0;", "80 of 105 pairs share nothing; the "
     "median leaf differs in 14 of 15 positions; 180 free slots yield 154 distinct questions."),
    ("Observed overlap 0.267 against a matched-chance 0.935", "— 3.5× BELOW chance, "
     "i.e. leaves are more different than chance, not more specialised."),
    ("Congruence does not fire:", "like-branch 0.7705 vs a matched control 0.7691 "
     "[0.7572, 0.7821], p = 0.82; like-minus-dislike +0.0054 against a shuffle band of "
     "[−0.0200, +0.0200], p = 0.61; only 4 of 7 nodes positive."),
    ("One secondary cell separates", "(root, items only: 0.4348 vs 0.3379 "
     "[0.2768, 0.4053], p = 0.004) — reported as a hypothesis: 1 of 7 nodes, 1 of 6 secondary "
     "tests, no multiplicity correction."),
], size=11)
textbox(s, MARGIN_L, Inches(5.25), lw, Inches(1.4),
        "Only 9 of the 15 global static questions appear anywhere in the tree, and 4 leaves "
        "share none of them. So the tree is genuinely doing something different from the static "
        "order — it just is not 'specialising by taste' in any way we can demonstrate.",
        size=11, color=MUTED, line_spacing=1.15)

rx = MARGIN_L + lw + Inches(0.40)
rw = CONTENT_W - lw - Inches(0.40)
textbox(s, rx, y + Inches(0.02), rw, Inches(0.3),
        "THE SIX REQUEST GROUPS  (k=20, fitted on 140,768 train users)",
        size=11, bold=True, color=ACCENT_B)
table(s, [
    ["group", "most distinctive films", "n", "train", "requests", "ratio"],
    ["inspiring", "Dark Knight Rises, Interstellar, Django", "9,743", "6.9 %", "20.2 %", "2.9×"],
    ["dialogue driven", "Birdman, Whiplash, Grand Budapest", "7,475", "5.3 %", "16.8 %", "3.2×"],
    ["gore", "Dredd, Watchmen, The Hateful Eight", "5,825", "4.1 %", "16.0 %", "3.9×"],
    ["marvel", "Iron Man 3, Avengers: Age of Ultron", "6,882", "4.9 %", "11.5 %", "2.4×"],
    ["school", "Aquamarine, Princess Diaries 2, Frozen", "4,945", "3.5 %", "6.1 %", "1.7×"],
    ["art house", "Holy Motors, The White Ribbon", "4,576", "3.3 %", "5.1 %", "1.6×"],
    ["six combined", "", "", "28.0 %", "75.7 %", "2.7×"],
], rx, y + Inches(0.38), rw, col_w=[20, 40, 12, 10, 11, 7], font_size=9.5, row_h=0.325,
    align=[L, L, R, R, R, R], first_col_bold=True)

textbox(s, rx, Inches(3.55), rw, Inches(3.1),
        ["28 % OF USERS RECEIVE 76 % OF THE REQUESTS.",
         ("Four chronological groups clear selection but fail agreement, receiving 0.7–1.3 % of "
          "requests each; the LARGEST training group (21,477 users, 15.3 %) receives 1.5 %.",
          {"size": 10.5, "bold": False, "color": MUTED}),
         ("Groups are not disjoint: membership overlaps at 1.36 groups per user before the argmax; "
          "marvel and inspiring share 10 of 12 characteristic words; 31 % of films appear in more "
          "than one pool. ARI of the request partition against the assignment is +0.269, and 5 of "
          "8 request groups map cleanly.", {"size": 10.5, "bold": False, "color": MUTED}),
         ("Separability rises 0.371 → 0.956 once grouped by the STATE. Route atlas: a "
          "discriminant fitted on a disjoint half assigns 0.907 of held-out users correctly "
          "against a chance floor of 0.167.", {"size": 10.5, "bold": False, "color": MUTED}),
         ("⛔ Raw-latent k-means gives RELEASE-YEAR buckets — 14 of 20 clusters span under 2.5 "
          "years. Concept-decoding halves that to 6 of 12.",
          {"size": 10.5, "bold": True, "color": ACCENT_C})],
        size=11.5, bold=True, color=INK, space_after=5, line_spacing=1.14)
footer(s, "Sources: chapterB_v2.tex :1434-1467, tab:clusters (:1692-1716), :1627-1677")
notes(s, """WHAT TO SAY (only if asked)
Two negative results I keep in the paper.

Routing specialisation is a null. Leaves ask almost entirely different questions — 80 of 105 pairs
share nothing — but we cannot show that the differences track taste. The congruence test comes back
at p = 0.82. One secondary cell separates and is reported as a hypothesis with the multiplicity
problem stated.

And the grouping: 28 % of users generate 76 % of the requests, so the request population is wildly
unrepresentative of the watching population. The groups are not disjoint. And the warning at the
bottom is one I paid for: naive k-means on the raw latent gives you release-year buckets, not taste
groups.

SOURCES as footed.""")

# --- APP 6: B opener synthesis + policy per-group ---------------------------
s, y = slide(prs, "Appendix B4 — Paper B: opener synthesis, and the distilled policy by group",
             "", kicker="appendix · paper b", accent=MUTED)
lw = Inches(6.4)
textbox(s, MARGIN_L, y + Inches(0.02), lw, Inches(0.3),
        "DO SYNTHESISED OPENERS LOOK LIKE REAL ONES?", size=11, bold=True, color=ACCENT_B)
table(s, [
    ["property", "real requests", "synthesised"],
    ["names a film", "0.630", "0.580"],
    ["names a genre", "0.398", "0.363"],
    ["names a tone", "0.214", "0.314"],
    ["internal coherence  (null: independent draw +0.172)", "+0.224", "+0.289"],
    ["popularity percentile of named films", "93.3", "96.4"],
], MARGIN_L, y + Inches(0.38), lw, col_w=[52, 24, 24], font_size=10.8, row_h=0.325,
    align=num(3), first_col_bold=True)
textbox(s, MARGIN_L, Inches(3.15), lw, Inches(1.9),
        ["Close on structure, and too popular and too coherent on the margins.",
         ("Centroid lift against the frequency ratio correlates at Spearman +0.56 to +0.93, but "
          "top-100 overlap is only 24–65 % — the ordering agrees, the membership does not.",
          {"size": 11, "bold": False, "color": MUTED}),
         ("Explicitly a LOWER bound: era openers are largely unavailable in our vocabulary "
          "(era mapping succeeds for only 70.2 % of mentions) and affect words are unsayable "
          "through this bank.", {"size": 11, "bold": False, "color": ACCENT_C})],
        size=12, bold=True, color=INK, space_after=5, line_spacing=1.15)

rx = MARGIN_L + lw + Inches(0.40)
rw = CONTENT_W - lw - Inches(0.40)
textbox(s, rx, y + Inches(0.02), rw, Inches(0.3),
        "THE ONE DISTILLED POLICY, BY GROUP  (reserved TEST, K=15, full NDCG@10)",
        size=11, bold=True, color=ACCENT_B)
table(s, [
    ["group", "n", "routing", "policy", "its own tree", "cold", "policy − cold"],
    ["art house", "464", ".843", ".29621", ".30046", ".27964", "+.01657 ★"],
    ["dialogue driven", "395", ".792", ".23299", ".22940", ".21894", "+.01404 ★"],
    ["inspiring", "1,526", ".855", ".25195", ".25534", ".24338", "+.00857 ★"],
    ["gore", "391", ".816", ".33406", ".33208", ".32595", "+.00811"],
    ["teenagers", "502", ".741", ".24405", ".24179", ".23712", "+.00692"],
    ["marvel", "599", ".957", ".42555", ".42534", ".41933", "+.00622"],
], rx, y + Inches(0.38), rw, col_w=[22, 10, 12, 14, 15, 13, 14], font_size=9.8, row_h=0.325,
    align=num(7), first_col_bold=True)

textbox(s, rx, Inches(2.75), rw, Inches(3.9),
        ["THREE THINGS THIS TABLE SAYS",
         ("1. The single policy lands within ±.004 of the correct group's OWN interview in every "
          "group — while also having to find the group itself.",
          {"size": 10.8, "bold": False, "color": MUTED}),
         ("2. Value does not track routing. marvel routes best (.957) and gains least (+.006); "
          "dialogue driven routes worst but gains second most.",
          {"size": 10.8, "bold": False, "color": MUTED}),
         ("3. The per-group ranking MOVES between validation and test: art house falls from "
          "+.02470 to +.01657, while dialogue driven rises from −.00266 to +.01404. Quote the "
          "POOLED contrast, not a group.", {"size": 10.8, "bold": True, "color": ACCENT_C}),
         ("Development (validation) estimates of the deployment mix, for comparison: +0.00333 ★, "
          "+0.00892 ★, −0.00037. Non-monotone in budget: K=1 +0.00161 ★, K=4 +0.00119 "
          "[−0.0003, +0.0027], K=15 +0.00343 ★.",
          {"size": 10.8, "bold": False, "color": MUTED})],
        size=11, bold=True, color=ACCENT_B, space_after=6, line_spacing=1.15)
footer(s, "Sources: chapterB_v2.tex tab:opener_match (:1838-1860), tab:policy (:2042-2073)")
notes(s, """WHAT TO SAY (only if asked)
Two things. The opener-synthesis check: our synthesised openers match real ones closely on structure —
how often they name a film, a genre, a tone — but are too popular and too coherent. And it is a lower
bound, because eras and affect words are largely unsayable through our bank.

And the per-group policy table, whose main lesson is the third bullet: the per-group ranking moves
between validation and test in both directions. Art house halves; dialogue driven goes from negative
to the second-best group. Only the pooled contrast is quotable.

SOURCES as footed.""")


# --- APP 7: C0 geometry detail ----------------------------------------------
s, y = slide(prs, "Appendix C1 — Paper C: C0 geometry in full",
             "", kicker="appendix · paper c", accent=MUTED)
lw = Inches(7.2)
table(s, [
    ["statistic, median over all 160,000 validation rows", "emitted ẑ", "IQR", "random direction"],
    ["cos₁ — cosine to the question it snaps to", "−0.18201", "[−0.32830, −0.06928]", "+0.06418"],
    ["margin ratio m = (s₁−s₂)/(s₁−s₁₀)", "0.40985", "[0.21470, 0.57909]", "0.26678"],
    ["entropy of softmax(s) over the top-50", "0.00549", "[0.00004, 0.41493]", "0.90049"],
    ["states with ≥3 candidates within 5 % of s₁", "20.8 %", "—", "32.6 %"],
    ["states with cos₁ > 0.98", "0 of 160,000", "—", "0"],
    ["states with cos₁ > 0", "4.76 %", "—", "—"],
    ["MAXIMUM cos₁ over all states", "0.1203", "—", "0.398"],
    ["cos₁, weighted by distinct STATE instead of user-row", "−0.260", "—", "—"],
    ["margin ratio m, state-weighted", "0.286", "—", "0.267 (floor)"],
], MARGIN_L, y + Inches(0.05), lw, col_w=[46, 18, 22, 14], font_size=10, row_h=0.315,
    align=num(4), first_col_bold=True)

textbox(s, MARGIN_L, Inches(3.90), lw, Inches(2.6),
        ["THE REGISTERED READING HAD NO POWER, AND WE SAY SO IN THE PAPER",
         ("The confirmation cell was a conjunction of two strict thresholds and the falsification "
          "cell a disjunction of two lax ones — so the norm-matched RANDOM direction lands in "
          "'between candidates' too. The cell is a property of the rule, not a characterisation "
          "of the policy.", {"size": 11, "bold": False, "color": MUTED}),
         ("The sentence originally carried — 'the prediction is falsified, and not narrowly' — is "
          "RETRACTED. What stands unconditionally: the maximum cos₁ over all 160,000 rows is "
          "0.1203, so no emission sits on its snapped question anywhere.",
          {"size": 11, "bold": True, "color": ACCENT_C})],
        size=11, bold=True, color=ACCENT_C, space_after=6, line_spacing=1.15)

rx = MARGIN_L + lw + Inches(0.35)
rw = CONTENT_W - lw - Inches(0.35)
textbox(s, rx, y + Inches(0.05), rw, Inches(0.3), "GATES", size=11, bold=True, color=ACCENT_C)
textbox(s, rx, y + Inches(0.40), rw, Inches(2.6),
        ["Snap identity: the argmax reproduces E1's own pick — 0 mismatches of 160,000, top-1 "
         "agreement 1.0000.",
         ("Banked rollout: regenerated interviews are BYTE-IDENTICAL on all 10,000 × 16; val full "
          "0.28355 and tail 0.13421 reproduce exactly.", {"size": 10.5, "bold": False}),
         ("Canonical intercept: full 0.15686538, tail 0.02747733 — exact.",
          {"size": 10.5, "bold": False}),
         ("Scale control: a random direction through the identical code path, reported beside "
          "every statistic.", {"size": 10.5, "bold": False})],
        size=10.5, color=MUTED, space_after=5, line_spacing=1.15)
textbox(s, rx, Inches(3.35), rw, Inches(3.3),
        ["THE MECHANISM, AND THE UNIT",
         ("Median top score −82.7; 95.2 % of states have a negative top score; median s₁−s₂ = 7.81 "
          "and s₁−s₁₀ = 17.50 against |s₁| ≈ 83. Cross-entropy constrains only DIFFERENCES, so "
          "nothing penalised a large common mode and the head placed one there.",
          {"size": 10.5, "bold": False, "color": MUTED}),
         ("Mean-removed, the emission's cosine to the entry it selects is +0.488, IQR "
          "[0.369, 0.587], positive at 98.9 % of rows, against +0.170 for a norm-matched random "
          "direction.", {"size": 10.5, "bold": False, "color": INK}),
         ("Unit: 160,000 rows = 31,874 distinct asked-history states inside 6,836 distinct "
          "interviews. 99.95 % of cos₁ variance is BETWEEN states, so the emission is essentially "
          "a function of the answer history, not of the user. Cluster bootstrap, cluster = user.",
          {"size": 10.5, "bold": False, "color": MUTED})],
        size=11, bold=True, color=ACCENT_C, space_after=6, line_spacing=1.15)
footer(s, "Sources: docs/results/C0_GEOMETRY.md · chapterC_v2.tex :456-648")
notes(s, """WHAT TO SAY (only if asked)
The full C0 table. The reason this slide exists is the red box: the registered reading rule turned out
to have no power, because a random direction satisfies it too. We retracted the sentence that depended
on it and kept only the maximum-statistic claim, which needs no reading rule at all.

The mechanism box is the correction that makes C0 usable: raw cosine is negative because the head
parked a huge common mode in the vector, which cross-entropy never penalises. Mean-remove it and there
is a genuine bank-aligned direction underneath — +0.488 against a random +0.170.

SOURCES as footed.""")


# --- APP 8: C1 detail --------------------------------------------------------
s, y = slide(prs, "Appendix C2 — Paper C: C1 in full, and the state-blind floor",
             "", kicker="appendix · paper c", accent=MUTED)
lw = Inches(7.4)
table(s, [
    ["", "primary cell (root.like.* withheld; 57 trained, 15 held out)",
     "secondary cell (depth ≥5 withheld; 42 trained, 30 held out)"],
    ["distinct questions preserved", "245 / 245  (1.000)", "244 / 245  (0.996)"],
    ["snap agreement with the banked policy", "0.9950", "0.9960"],
    ["norm ratio — all / trained / held-out", "1.0629 / 1.0449 / 3.4798",
     "1.1056 / 1.0541 / 3.4213"],
    ["cos(ẑ, own target), TRAINED states", "0.96958  (min 0.932; 57/57 ≥ 0.90)",
     "0.97455  (42/42)"],
    ["cos(ẑ, own target), HELD-OUT states", "−0.11948", "−0.01069"],
    ["(a) own − random other held-out target  [registered]", "+0.05926 [+0.04269, +0.07539]",
     "+0.10114 [+0.08436, +0.11798]"],
    ["(a′) the SAME statistic on the UNTRAINED warm start",
     "+0.08432 [+0.04738, +0.12680]  ← larger", "+0.10441 [+0.08950, +0.11953]  ← ties"],
    ["(b) own − the warm start's cosine, same states", "+0.25954 [+0.09535, +0.45041]",
     "+0.15696 [+0.06792, +0.25538]"],
], MARGIN_L, y + Inches(0.02), lw, col_w=[36, 32, 32], font_size=9.8, row_h=0.375,
    align=[L, R, R], first_col_bold=True)

table(s, [
    ["the state-blind constant floor (2026-09-01 correction)", "constant", "this policy",
     "paired difference"],
    ["trained states, raw cosine", "0.9531", "0.9696 / 0.9745", "policy +0.015"],
    ["held-out, primary", "0.9502", "0.0609", "constant +0.8893 [+0.6555, +1.0905], wins 14/15"],
    ["held-out, secondary", "0.9623", "0.0351", "constant +0.9271 [+0.8254, +1.0194], wins 30/30"],
], MARGIN_L, Inches(3.85), lw, col_w=[36, 16, 18, 30], font_size=9.8, row_h=0.325,
    align=num(4), first_col_bold=True)

textbox(s, MARGIN_L, Inches(5.35), lw, Inches(1.4),
        ["Why the targets are so easy for a constant to hit: they form a tight cone — median raw "
         "pairwise cosine 0.920 — because measured-best directions across states share a dominant "
         "common component (0.95 raw to a state-blind constant, 0.63 mean-removed), even though "
         "the top-1 CANDIDATES differ at 0.247. The state-specific share of mean-removed target "
         "energy is 0.5915 [0.5558, 0.6270] — substantial, and unrecoverable by a global linear "
         "probe, a profile parameterisation, or local kNN transfer.",
        ], size=11, color=MUTED, line_spacing=1.15)

rx = MARGIN_L + lw + Inches(0.30)
rw = CONTENT_W - lw - Inches(0.30)
textbox(s, rx, y + Inches(0.02), rw, Inches(0.3), "SCOPE", size=11, bold=True, color=ACCENT_C)
textbox(s, rx, y + Inches(0.38), rw, Inches(6.2),
        ["72 of 144 tree nodes emit; fully-emitting paths run to depth 12.",
         ("The 42-state population used downstream is STRUCTURALLY SELECTED: a full exact gain "
          "table exists at a state precisely when the tree-grower left no truncated tail there, so "
          "missingness is deterministic in the process under study. 27 of the 42 sit under the "
          "root's dislike branch — roughly 2:1 over-representation — and the cohorts nest: 36,197 "
          "user-rows from 10,000 users, with five states each an ancestor of twenty or more of the "
          "others.", {"size": 10.5, "bold": False, "color": MUTED}),
         ("The blend's INGREDIENTS are unstable: mean top-five overlap with itself under bootstrap "
          "is 1.905 of 5; all five survive a majority of resamples at ZERO of 35 states; two "
          "disjoint halves of a cohort pick top-five sets with median overlap zero. The DIRECTION "
          "is more stable but not stable — split-half cosine 0.987 raw, 0.656 mean-removed, IQR "
          "[0.090, 0.974].", {"size": 10.5, "bold": False, "color": MUTED}),
         ("Licensed sentence: “a blend whose ingredients are unstable draws from a near-flat "
          "measured gain profile, and whose direction is moderately stable.”",
          {"size": 10.5, "bold": True, "color": ACCENT_C})],
        size=11, color=INK, space_after=6, line_spacing=1.15)
footer(s, "Sources: docs/results/C1_FINAL.md (corrected 2026-09-01) · chapterC_v2.tex :649-786")
notes(s, """WHAT TO SAY (only if asked)
The full C1 ledger, including the correction that landed on 1 September.

The row that matters is (a′). The registered generalisation statistic, (a), was positive — own target
beats a random other held-out target by +0.059. Then the executing agent added a null we had not
registered: run the identical statistic on the UNTRAINED warm start. It comes back LARGER, at +0.084.
Neighbouring states in one tree carry correlated targets, so 'own beats a random other' is positive
before any training at all. Reported as registered, (a) would have read as evidence of extrapolation.

The state-blind constant table is the sharper version of the same point: at held-out states a single
fixed vector, ignoring the state entirely, beats the trained policy by almost 0.9 cosine and wins
14 of 15 and 30 of 30.

The scope box on the right is the population caveat: the 42 downstream states are structurally
selected, not a sample, and we never describe them as a population.

SOURCES as footed.""")

# --- APP 9: C4B / C4C -------------------------------------------------------
s, y = slide(prs, "Appendix C3 — Paper C: what the round trip costs, and that decoding removes it",
             "", kicker="appendix · paper c", accent=MUTED)

table(s, [
    ["decoding  (42 nested states, 36,197 user-rows)", "distinct / 42", "cos(u_text, R_s)",
     "Δ vs greedy", "med. rank", "cost L − T, full"],
    ["greedy (banked)", "23", "+0.2039", "—", "313",
     "+0.01802 [+0.01668, +0.01940]"],
    ["sample T = 0.7", "38", "+0.1034", "−0.1005 [−0.197, −0.004]", "441",
     "+0.00520 [+0.00397, +0.00638]"],
    ["sample T = 1.0", "41", "+0.0942", "−0.1097 [−0.220, −0.002]", "464", "+0.00746"],
    ["best-of-8 by LM likelihood  — DECODING OF RECORD", "37", "+0.1846",
     "−0.0193 [−0.097, +0.058]  n.s.", "395", "+0.00884 [+0.00747, +0.01025]"],
], MARGIN_L, y + Inches(0.02), CONTENT_W, col_w=[30, 11, 14, 20, 10, 15],
    font_size=10, row_h=0.36, align=num(6), first_col_bold=True)

lw = Inches(6.5)
textbox(s, MARGIN_L, Inches(3.15), lw, Inches(0.3),
        "THE COST, AND ITS DECOMPOSITION", size=11, bold=True, color=ACCENT_C)
bullets(s, MARGIN_L, Inches(3.50), lw, [
    ("Translating a direction into English and reading it back costs",
     "+0.01802 full [+0.01668, +0.01940] and +0.00672 tail [+0.00541, +0.00805]."),
    ("Real and non-uniform:", "21 of 42 states have an interval above zero and 5 have "
     "one BELOW it — states where the English beats the direction. Best is 'theatrical' at "
     "−0.079 [−0.119, −0.040]."),
    ("Not an artefact of the two arms folding the same content:", "the overlap |L ∩ T| "
     "is zero at 38 of 42 states, mean 0.19."),
    ("Concentrated on the collapsed renders:", "18 unique-render states pay +0.00633; "
     "24 repeated-render states pay +0.02628 — a factor of 4.2."),
], size=11)

rx = MARGIN_L + lw + Inches(0.45)
rw = CONTENT_W - lw - Inches(0.45)
textbox(s, rx, Inches(3.15), rw, Inches(0.3),
        "THE PREDICTION WAS CONFIRMED AND THE MECHANISM REFUTED",
        size=11, bold=True, color=ACCENT_C)
textbox(s, rx, Inches(3.50), rw, Inches(3.1),
        ["We predicted the cost was concentrated on mode-collapsed renders — it is. We attached "
         "the mechanism 'lost meaning' — that is wrong.",
         ("Absolute rows, declared-circular, printed and never headlined: base (fold nothing) "
          "0.16571 · L (fold the direction's own top concepts) 0.17443 · T greedy (fold the "
          "concepts recovered from the text) 0.15641 · T sampled 0.16924.",
          {"size": 10.5, "bold": False, "color": MUTED}),
         ("Greedy's stock words fold BELOW the no-fold baseline. So the cost is largely HARM from "
          "confidently-wrong concepts, not information loss — and sampling removes harm rather "
          "than confusion.", {"size": 11, "bold": True, "color": ACCENT_C}),
         ("Best-of-8 repairs greedy's collapses (religion→scifi 620→61; religion→animals 479→54; "
          "hollywood→ocean 963→133; science→science fiction 31→7) while losing hits greedy keeps "
          "(art 4 → sela 873; indictment 5 → standard-of-life 972). It is chosen on the frozen "
          "LM's own length-normalised likelihood — a rule blind to the evaluation metric by "
          "construction.", {"size": 10.5, "bold": False, "color": MUTED})],
        size=11, color=INK, space_after=6, line_spacing=1.15)

textbox(s, MARGIN_L, Inches(6.30), lw, Inches(0.8),
        "Reproduction gate: the greedy arm was REGENERATED rather than read, and reproduced the "
        "banked run exactly — 42 renders bit-identical, cosines to maximum absolute difference 0, "
        "triangulation split and translation cost identical.",
        size=10.5, color=MUTED, line_spacing=1.15)
footer(s, "Sources: docs/results/C4B_NOISE.md · C4C_DECODING.md · paperC_lbr.tex tab:loop")
notes(s, """WHAT TO SAY (only if asked)
The translation cost, and the experiment that reinterpreted it.

Two arms under the same simulator and the same certified fold operator, differing only in what gets
folded: arm L folds the direction's own top concepts, arm T folds the concepts recovered from the
rendered text. The circular component is common to both, so only the difference is read. The cost is
+0.018 full and +0.007 tail.

We predicted it was concentrated on the mode-collapsed renders, and it is — a factor of four. We
attached the mechanism 'the English loses meaning'. Testing that by changing ONLY the decoding
confirmed the number and refuted the mechanism: greedy's stock words fold BELOW the no-fold baseline
of 0.16571. That means they are not losing information, they are actively harming — a confidently
wrong concept is worse than no concept. Sampling removes harm, not confusion.

Best-of-8 by the language model's own likelihood is the decoding of record: it is the only Pareto
point, and critically its selection rule is blind to our evaluation metric by construction, so it
cannot be accused of selecting on the outcome.

SOURCES as footed.""")


# --- APP 10: methods -----------------------------------------------------------
s, y = slide(prs, "Appendix M1 — Methods: the answer model, circularity, and the "
                  "statistical conventions",
             "", kicker="appendix · methods", accent=MUTED)
lw = Inches(7.1)
table(s, [
    ["", ""],
    ["What an answer IS", "recorded ratings (items) and a signed, exposure-corrected behavioural "
     "affinity, SEL (concepts). Three outcomes: like / dislike / no answer."],
    ["What may NOT enter an answer", "no LLM-derived answer or answerability grid. The July-2026 "
     "audit found them popularity-dominated with 3 of 4 gates null-passing and the distilled "
     "value ≈ EASE. No model may train on, gate on, or cite them."],
    ["Answerability", "a transparent STRUCTURAL rule — rated ∨ popularity > τ for items; "
     "≥ k rated members for concepts. Not a learned model."],
    ["The firewall (Paper A's I2)", "answers are recorded behaviour and fold-in only. This is the "
     "project's defence against the field-wide circularity attack, and the blind panel says we "
     "under-sell it."],
    ["Where circularity IS declared", "exactly one place: Paper C's translation-cost measurement, "
     "which needs an answer model by construction. Both arms run under the SAME simulator, only "
     "the DIFFERENCE is read, absolute rows are printed and never headlined, and the declaration "
     "is in the paper's own voice."],
    ["Paper C's contract", "the simulated answerer is used to put the policy at interview states "
     "and for nothing else. No other quantity in Paper C comes from simulating an answer and "
     "observing that the belief improved."],
], MARGIN_L, y + Inches(0.05), lw, col_w=[24, 76], font_size=10.5, row_h=0.62,
    first_col_bold=True, header_rows=0)

rx = MARGIN_L + lw + Inches(0.40)
rw = CONTENT_W - lw - Inches(0.40)
textbox(s, rx, y + Inches(0.05), rw, Inches(0.3),
        "STATISTICAL CONVENTIONS", size=11, bold=True, color=ACCENT)
bullets(s, rx, y + Inches(0.42), rw, [
    ("Paired percentile bootstrap,", "10,000 resamples, seed 98765, clustered over users. "
     "Self-comparison must come out exactly zero."),
    ("Every statistic carries an interval.", "A point estimate without one is a "
     "description, not a result."),
    ("Every statistic declares its aggregation unit.", "Interview states nest inside "
     "users and inside each other, so a user-row proportion and a state proportion are "
     "different quantities. The other weighting is reported wherever a threshold depends on it."),
    ("Every registered reading rule is run against its own null AT DESIGN TIME.",
     "A rule that puts a norm-matched random direction in the same cell as the emission "
     "has no power to characterise the emission."),
    ("Wilson intervals for proportions;", "cluster bootstrap with the user as cluster "
     "for medians."),
    ("★ in this deck = the paired interval excludes zero.", ""),
], size=10.8, marker_color=ACCENT)
footer(s, "Sources: CLAUDE.md 'Settled' + HARD RULES · chapterA_v2.tex §firewall (:1552-1595) · "
          "chapterC_v2.tex Setup · chapterB_v2.tex table captions")
notes(s, """WHAT TO SAY (only if asked)
The methods contract. The single most important row is 'what may not enter an answer'. In July we
audited the LLM-derived answer and answerability artifacts and found them popularity-dominated, with
three of four gates passing on a null and the distilled value function essentially equal to EASE. They
were retired, and no model since may train on, gate on, or cite them. Answers are recorded behaviour;
answerability is a transparent structural rule.

That firewall is our defence against the circularity attack, which the blind panel says was the top
objection to all three papers it graded — and which it says we under-sell.

There is exactly one place circularity is declared rather than avoided: Paper C's translation cost,
which needs an answer model by construction. Both arms run under the same simulator, only the
difference is read, and the absolute rows are printed but never headlined.

SOURCES as footed.""")


# --- APP 11: honest bounds ------------------------------------------------------
s, y = slide(prs, "Appendix M2 — The honest bounds, in one place",
             "Everything I would want a hostile reviewer to have found already.",
             kicker="appendix · limitations", accent=MUTED)
table(s, [
    ["", "the bound", "what it costs the claim"],
    ["P1 / P1b  (Paper B, validation only)",
     "The pre-registered falsifier FIRED. At 38,336 states whose true label was a held-out "
     "question the retrained policy emits it 0.0000 at top-1 and top-5; a fixed popularity "
     "ranking beats it at 1.45 % top-5. On the undamaged policy, off-manifold, E1 ranks the "
     "measured-best question 208th–374th of 1,000.",
     "'the policy emits questions it was never taught' is NOT available and is reported failed. "
     "Neither test isolates the vocabulary axis: P1b's cells are at unseen STATES, and P1 damages "
     "the policy to create the test."],
    ["C1 emission scope  (Paper C)",
     "At withheld states cos to the own target is −0.11948 and a state-blind constant beats the "
     "policy by +0.8893, winning 14/15. The registered generalisation statistic is matched by the "
     "UNTRAINED warm start.",
     "the licensed sentence is 'the policy emits where the question landscape was measured'. No "
     "number is quoted from a held-out-state emission."],
    ["One consumer  (Paper B Act 3)",
     "The tree result rests on one consumer, and our own flip section shows consumer-dependence.",
     "the strongest open objection. Fixable by SCORING the frozen tree through RecVAE."],
    ["Act 5 openers are group-stamped",
     "routing is partly circular and the scramble control cannot detect it; opener−cold survives "
     "because the STATE is held fixed.",
     "claim adaptivity beyond the closed tree, NOT routing."],
    ["The answer model is inferred, not observed",
     "answerability is a structural rule, and it is load-bearing for Act 4's entire result. "
     "The written-request corpus bounds open-answer richness from above.",
     "gate C3, the human round-trip, has never been run — it is what would close this."],
    ["Reproducibility caveat  (Paper B)",
     "the GLM-UCB and Thompson cells are not seed-stable (iALS basis re-drawn without a "
     "random_state); drift up to ±0.0026, comparable to the CI half-widths.",
     "quote those two rows to three decimals, not four."],
    ["Single seed  (Paper C adapters)",
     "one training seed per adapter arm.",
     "our own checklist forbids the word 'robust' until there are two, and we do not use it."],
], MARGIN_L, y + Inches(0.02), CONTENT_W, col_w=[16, 47, 37], font_size=9.2, row_h=0.66)
footer(s, "Sources: P1_HELDOUT_ACTIONS.md · P1B_NEVER_TAUGHT.md · C1_FINAL.md · "
          "PAPER_B_MASTER_TODO.md · ACT1_PAIRED_CI.md · C3B_ADAPTER.md · docs/HARSH_REVIEW_POINTS.md")
notes(s, """WHAT TO SAY (only if asked, or volunteer if the meeting is going well)
This is the slide I would put up if you asked 'what is wrong with all this'. It is the consolidated
list, and every item on it was found by our own control rather than by a reviewer.

The first two rows are the same finding at two levels. In Paper B, a policy trained with questions
held out does not emit those questions — the falsifier fired and the claim is reported as failed. In
Paper C, a policy trained at some interview states does not emit sensibly at others, and a state-blind
constant beats it there. Both are the same regularity: these systems are excellent exactly where they
were supervised and no better than baseline outside it. I think that is a thesis-level finding rather
than two disappointments, and it is worth writing up as one.

The row I would act on first is 'one consumer'. It is cheap to fix and it is the objection I would
raise myself.

SOURCES as footed.""")


# --- APP 12: what was NOT built / not run -----------------------------------
s, y = slide(prs, "Appendix M3 — What was not built, and what is retracted",
             "The ledger of absences, so nothing looks like an oversight.",
             kicker="appendix · scope", accent=MUTED)
lw = Inches(6.5)
textbox(s, MARGIN_L, y + Inches(0.02), lw, Inches(0.3),
        "SELECTORS NOT BUILT  (Paper B prints this table itself)", size=11, bold=True, color=ACCENT_B)
textbox(s, MARGIN_L, y + Inches(0.38), lw, Inches(1.5),
        "ICF (4 policies) · SEL-BALD · PEBOL acquisition · fMF · PERE · Greedy SLIM / QBandit · "
        "RBMF / RMVA · GreedyExtend · The FacT / FacT-CRS · UpsRec · MeLU · a-stratification "
        "— each marked ✗ built, ✗ ours.",
        size=11.5, color=MUTED, line_spacing=1.18)
textbox(s, MARGIN_L, Inches(3.05), lw, Inches(0.3),
        "EXPERIMENTS NOT RUN", size=11, bold=True, color=ACCENT_B)
bullets(s, MARGIN_L, Inches(3.40), lw, [
    ("Gate C3 — the human round-trip study.", "Owed project-wide. Blocks the strongest "
     "claims across A, B and C."),
    ("Hu & Yu replication at depths 8 and 10,", "the Part-A pairwise arm, the fMF RMSE "
     "panel, and Netflix / Yahoo — unrun, and stated as unrun."),
    ("Act 5's 15-minute routing probe", "— required before Act 5 ships standalone."),
    ("Paper B Act 3 through a second consumer", "— the recommended next action."),
], size=11.5)

rx = MARGIN_L + lw + Inches(0.45)
rw = CONTENT_W - lw - Inches(0.45)
textbox(s, rx, y + Inches(0.02), rw, Inches(0.3),
        "RETRACTED, AND WHY THE RETRACTION IS THE POINT", size=11, bold=True, color=ACCENT_C)
table(s, [
    ["retracted claim", "what killed it"],
    ["'items win the long interview' (Paper A)",
     "it ran on the superseded i25 fold; retracted at the G0 audit"],
    ["the tree takes the tail at K=8, +0.0068 (Paper B)",
     "the COMPARATOR was selected under a narrower mask. Re-differenced: −0.0016, does not "
     "separate. The tree's own numbers never moved."],
    ["+0.02092 tail advantage (Paper B Act 3)",
     "two thirds of it was comparator weakness; the rebuilt order shares 4 of 16 questions and "
     "0 of 16 positions with the one it replaced"],
    ["'the prediction is falsified, and not narrowly' (C0)",
     "the registered rule had no power — a random direction lands in the same cell"],
    ["the emission cells of the C4 pilot",
     "34/42 item-hot and 32/42 concept-hot renders were the EMPTY STRING; emission "
     "mean-removed norm was 38.3× the target's"],
    ["the C3 PoC's 'success' cell",
     "zero functioning controls and an unpowered judge; variant (a) was structurally "
     "unconstructible"],
    ["the +47 % adaptivity magnitude, and all pre-July interview numbers",
     "they ran on the retired LLM answer environment. Directions survive as priors; "
     "magnitudes must re-establish."],
], rx, y + Inches(0.38), rw, col_w=[42, 58], font_size=9.5, row_h=0.62)
footer(s, "Sources: chapterB_v2.tex tab:coverage, :352-361 · CONDITIONED_ARMS_ON_TEST.md · "
          "C0_GEOMETRY.md · C4_PILOT.md · C3_POC.md · CLAUDE.md 'Settled'")
notes(s, """WHAT TO SAY (only if asked)
Two ledgers. What was not built — Paper B prints this itself, twelve more systems each marked not-built
and not-ours, because I would rather print it than have it discovered. And what was retracted.

I keep the retraction list deliberately, because the pattern in it is the argument for the method.
Look at the second and third rows: in both cases a headline moved and the cause was the COMPARATOR
moving, not the arm. The tree's own numbers never changed. That is a failure mode I have now hit three
times and it is the reason the deck's rule is: when a contrast moves, check whether the arm or the
comparator moved.

SOURCES as footed.""")


# --- APP 13: one-page number index ------------------------------------------
s, y = slide(prs, "Appendix M4 — The numbers, on one page",
             "For fast lookup during questions.",
             kicker="appendix · index", accent=MUTED)
lw = Inches(6.5)
table(s, [
    ["PAPER A", ""],
    ["tower TEST full / tail NDCG@10", "0.3496 / 0.2491"],
    ["RecVAE, same ruler", "0.3540 / 0.2497   (paired: −0.0043 full, −0.0006 tail)"],
    ["graded − binarised", "+0.0039 [+0.0025,+0.0053] / +0.0020 [+0.0007,+0.0031]"],
    ["concept affinity R²: SEL / +VAL / LLM", "0.376 / 0.398 / 0.017"],
    ["concept − item at q16 (POLAR)", "+0.0146 full / +0.0265 tail"],
    ["polarising bank answer rate: item / concept", "4.8 % / 97.2 %"],
    ["published-number bridge", "EASE +0.0003, RecVAE +0.0005 (ML-20M NDCG@100)"],
], MARGIN_L, y + Inches(0.02), lw, col_w=[52, 48], font_size=9.6, row_h=0.265,
    align=num(2), first_col_bold=True)
table(s, [
    ["PAPER B", ""],
    ["battery", "23 rows · 21 distinct rules · 13 published methods"],
    ["static beats every adaptive arm", "138 of 138 cells (combined and concepts-only orders)"],
    ["greedy items-only − best adaptive, k=16", "+0.0281 [+0.0247,+0.0314] full"],
    ["NDCG tree − static on TEST, K=16", "+0.01035 [+0.00739,+0.01323] / "
     "+0.00705 [+0.00394,+0.01012]"],
    ["split-criterion ablation", "56 of 56 intervals exclude zero"],
    ["bank change vs conditioning (tail)", "+0.02199 vs +0.00705"],
    ["answers per 16 questions: item / concept", "3.11 / 12.36"],
    ["Reddit corpus", "4,431 threads · 63.0 % name a film · linking precision 0.960"],
    ["distilled policy on TEST, K=15", "+0.00343 ★ pooled · +0.00946 ★ volunteers"],
], MARGIN_L, Inches(4.20), lw, col_w=[52, 48], font_size=9.6, row_h=0.265,
    align=num(2), first_col_bold=True)

rx = MARGIN_L + lw + Inches(0.45)
rw = CONTENT_W - lw - Inches(0.45)
table(s, [
    ["PAPER C", ""],
    ["C0 max cos₁ over 160,000 states", "0.1203  (mean-removed +0.488 vs +0.170 random)"],
    ["C1 gates", "245/245 distinct · snap agreement 0.9950 · fit 0.96958"],
    ["C1 held-out states", "−0.11948; state-blind constant wins by +0.8893"],
    ["C3B  A − A′", "+0.1114 [+0.1054,+0.1173]"],
    ["C3B  A as share of gold-phrase ceiling", "107 %  (ceiling +0.0892)"],
    ["C3B  identification", "0.6778 [0.6631,0.6931]  (chance 0.500)"],
    ["C3B-v2 question-form rate", "0.9471 [0.9398,0.9542]"],
    ["C3B-v2 templating cost to the ceiling", "0.08920 → 0.06828  (−23 % items, −17 % concepts)"],
    ["C4 triangulation, distinctive renders", "median rank 91 of 1,031  (n = 18/42)"],
    ["C4 catalogue text vs rendered word", "−0.158 [−0.224,−0.089]  vs  +0.204 [+0.117,+0.291]"],
    ["C4 point-specificity", "+0.28593 [+0.17820,+0.39777]  (prior floor at zero)"],
    ["C4B translation cost", "+0.01802 [+0.01668,+0.01940] full · +0.00672 tail"],
    ["C4C decoding: distinct renders / cost", "23 → 41 · cost cut 2.0–3.5×"],
    ["C4D transcript", "6 turns · 4 of 6 well-formed · NDCG column unquotable"],
], rx, y + Inches(0.02), rw, col_w=[46, 54], font_size=9.6, row_h=0.265,
    align=num(2), first_col_bold=True)
footer(s, "★ = paired interval excludes zero.  Full provenance in each slide's speaker notes.")
notes(s, """WHAT TO SAY (only if asked)
Lookup page. Every number here appears on an earlier slide with its source named in that slide's
notes.

SOURCES: see the per-slide notes for A, B and C sections.""")

# >>>SLIDES<<<

for w in D.relayout(prs):
    print("  ! " + w)

prs.save(OUT)
print("wrote %s  (%d slides)" % (OUT, len(prs.slides._sldIdLst)))
