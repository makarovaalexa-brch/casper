"""Build the CASPER supervisor presentation deck.

    poetry run python new_chapters/slides/build_deck.py

Emits new_chapters/slides/casper_supervisor_2026-09.pptx (16:9).
Everything is a native pptx object (text, tables, autoshapes, connectors) so the
deck stays fully editable after a Google Slides import.

EVERY number below traces to a source named in that slide's speaker notes.
Sources of record:
  Chapter 1  new_chapters/chapterA_v2/chapterA_v2.tex
           docs/design/GATE_BATTERY_INSTRUMENT.md
           docs/results/{CONCEPT_CHANNEL_RESULT,ANSWER_DECOMPOSITION_RESULT,
                         PAPER_A_NUMBER_AUDIT,G1C_CONTROL_RESULT}.md
           CLAUDE.md (canonical ruler / split sizes)
  Chapter 2  new_chapters/chapterB_v2/chapterB_v2.tex   (definitive)
           docs/design/PAPER_B_MASTER_TODO.md
           docs/results/{ACT1_PAIRED_CI,CONDITIONED_ARMS_ON_TEST,
                         HU_YU_2013_REPLICATION_AND_ABLATION,M4B_FOREST_POLICY,
                         M6L_OPENER_FITTED_TEACHER,REDDIT_LINKER_AUDITION,
                         P1_HELDOUT_ACTIONS,P1B_NEVER_TAUGHT,
                         BASELINE_OBJECTIVE_AUDIT}.md
  Chapter 3  new_chapters/paperC_lbr/paperC_lbr.tex, new_chapters/chapterC_v2/chapterC_v2.tex
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
                      bullets, footer, figure)

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
textbox(s, Inches(1.0), Inches(2.55), Inches(11.4), Inches(1.4),
        ["Cold-start conversational recommendation:",
         "the instrument, the interview, and the question that is not in the catalogue"],
        size=32, bold=True, color=INK, line_spacing=1.14, space_after=2)
textbox(s, Inches(1.0), Inches(4.35), Inches(11.4), Inches(0.4),
        "Aleksandra Makarova", size=17, color=INK)
textbox(s, Inches(1.0), Inches(4.90), Inches(11.4), Inches(0.4),
        "September 2026", size=15, color=MUTED)


# --- 2. the thesis in one diagram ------------------------------------
s, y = slide(prs, "Thesis structure",
             "CASPER — Cold-start Adaptive Selection for Preference Elicitation in Recommendation. "
             "Three chapters on one measurement stack: each hands the next a licensed capability, and "
             "nothing downstream is measured on a ruler the upstream chapter did not certify.")

bw, bh = Inches(3.55), Inches(1.55)
gap = Inches(0.62)
x0 = MARGIN_L + Inches(0.25)
ytop = y + Inches(0.45)

box(s, x0, ytop, bw, bh,
    [("CHAPTER 1", {"size": 12, "color": ACCENT}),
     ("The instrument", {"size": 17}),
     ("A certified cold-start recommender,", {"size": 11, "bold": False, "color": MUTED}),
     ("a fold-in operator, and one ruler", {"size": 11, "bold": False, "color": MUTED})],
    line=ACCENT)
box(s, x0 + bw + gap, ytop, bw, bh,
    [("CHAPTER 2", {"size": 12, "color": ACCENT_B}),
     ("The interview", {"size": 17}),
     ("What a cold interview", {"size": 11, "bold": False, "color": MUTED}),
     ("should optimise", {"size": 11, "bold": False, "color": MUTED})],
    line=ACCENT_B)
box(s, x0 + 2 * (bw + gap), ytop, bw, bh,
    [("CHAPTER 3", {"size": 12, "color": ACCENT_C}),
     ("Questions outside the catalogue", {"size": 15}),
     ("Emitting and rendering a direction", {"size": 11, "bold": False, "color": MUTED}),
     ("that is not in the question bank", {"size": 11, "bold": False, "color": MUTED})],
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
        ["A pre-registered control has overturned the headline three times:",
         ("C0's random-direction floor  ·  C3B's equivalence twin  ·  C1's untrained warm "
          "start.  Everything here is read against its own null.",
          {"size": 13, "bold": False, "color": MUTED})],
        size=14.5, bold=True, color=INK, space_after=3)
footer(s, "Sources: chapterA_v2.tex · chapterB_v2.tex · chapterC_v2.tex")


# --- 3. the problem ---------------------------------------------------
s, y = slide(prs, "The cold-start elicitation problem",
             "A new user has no history. The system may ask a few questions before it recommends. "
             "Which questions to ask, and how much the choice matters.")

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
        "the belief chooses the next question — this loop is Chapter 2",
        size=10, color=ACCENT_B, align=C)

table(s, [
    ["The setting, fixed once and reused everywhere", ""],
    ["Dataset", "MovieLens-25M, Liang strong-generalisation recipe, catalogue-restricted"],
    ["Split", "train 140,768 / val 10,000 / test 10,000 users, seed 98765, per-user 80/20"],
    ["Item vocabulary", "18,359 train vocabulary (arena catalogue 18,430 = films with ≥20 ratings)"],
    ["Question bank", "19,390 = 18,359 item questions + 1,031 concept questions"],
    ["Metric", "full AND tail NDCG@10, scored with the recommender's learned decoder bias"],
    ["Asking nothing", "0.15855 full / 0.02819 tail — the popularity prior"],
    ["Whole profile", "0.3496 full / 0.2491 tail — the same recommender given everything"],
], MARGIN_L, gy + bh2 + Inches(0.50), CONTENT_W, col_w=[24, 76],
    font_size=12, row_h=0.315, first_col_bold=True)
footer(s, "Sources: CLAUDE.md 'Metric & reporting' · chapterA_v2.tex :861 · chapterB_v2.tex tab:tree")

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


sd = divider("Chapter 1 — the measurement instrument",
             "Before an elicitation strategy can be measured, the recommender that consumes the "
             "answers has to be certified.",
             ACCENT, "9 slides  ·  ~12 minutes  ·  status: closed, 41 pp, unsubmitted, target TORS")


# --- A1: what A is ----------------------------------------------------
s, y = slide(prs, "Why Chapter 1 exists",
             "Strategy rankings invert when the recommender changes. The recommender is therefore "
             "part of the measurement, and has to be certified first.", accent=ACCENT)

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
    ("The consequence.", "A null in Chapter 2 becomes attributable to the policy "
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
        ["THE INVERSION, MEASURED", ("Fifth consumer (Golbandi leaf model): Thompson beats the "
         "static order by +0.0100 / +0.0171 / +0.0148 / +0.0281 full at k=1,2,4,8, every "
         "interval excluding zero — the exact reverse of its ordering on the certified tower.",
         {"size": 11.5, "bold": False, "color": MUTED})],
        size=11, bold=True, color=ACCENT, space_after=3, line_spacing=1.12)
footer(s, "Sources: chapterA_v2.tex §intro, §method · chapterB_v2.tex :932-945 (the flip)")


# --- A2: the ruler ----------------------------------------------------
s, y = slide(prs, "The canonical ruler",
             "One split, one metric, one scoring convention, fixed in July 2026 and unchanged "
             "since. Every number in the three chapters is measured on it.", accent=ACCENT)

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
    ["Published-number bridge (ML-20M, NDCG@100)", "ours", "published", "Δ", "verdict"],
    ["EASE (Steck)", "0.4203", "0.420", "+0.0003", "PASS"],
    ["RecVAE (Shenbin)", "0.4425", "0.442", "+0.0005", "PASS"],
    ["Mult-VAE (Liang)", "0.4223", "0.426", "−0.0037", "near-snap"],
    ["Mult-DAE (Liang)", "0.3969", "0.419", "−0.022", "FAIL → dropped from the bank"],
], MARGIN_L, Inches(4.95), CONTENT_W, col_w=[44, 12, 13, 12, 19],
    font_size=11, row_h=0.285, align=num(5), first_col_bold=True)
footer(s, "Sources: CLAUDE.md 'Metric & reporting' · chapterA_v2.tex :1368-1392 · "
          "PAPER_A_NUMBER_AUDIT.md :46-56")


# --- A3: the gate battery ---------------------------------------------
s, y = slide(prs, "The certification battery",
             "Eleven per-build gates and three one-time bridges. Every threshold was written down "
             "before the run.", accent=ACCENT)

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
     "full and tail", "PASS — the gate that licenses every Chapter 2 comparison"],
    ["C1  published-number bridge", "match printed numbers on a published protocol",
     "DONE — EASE +0.0003, RecVAE +0.0005"],
    ["C2  answer-model transfer", "the gain survives held-out behavioural answers and label noise",
     "run; behavioural and stated preferences agree only ~21 %"],
    ["C3  human round-trip", "real people answer, held-out design", "NOT RUN — owed project-wide"],
], MARGIN_L, y + Inches(0.02), CONTENT_W, col_w=[21, 42, 37],
    font_size=9.8, row_h=0.315, first_col_bold=True)
footer(s, "Sources: docs/design/GATE_BATTERY_INSTRUMENT.md · chapterA_v2.tex tab:batteryresults · "
          "G1C_CONTROL_RESULT.md · CONCEPT_CHANNEL_RESULT.md")


# --- A4: the baseline bank -------------------------------------------
s, y = slide(prs, "Recommender comparison on the canonical split",
             "Fifteen systems, all rebuilt in-house and scored under identical conditions. "
             "The CASPER tower sits inside the frontier band.", accent=ACCENT)

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

textbox(s, MARGIN_L, Inches(6.05), CONTENT_W, Inches(0.9),
        ["Reading the table.",
         ("RecVAE − EASE is +0.0064 full, about 1.8 unpaired SE, so the top four are within "
          "measurement of each other. The closed-form conjugate belief accumulator — the "
          "consumer most elicitation work uses — forfeits about 0.12 full NDCG against a "
          "learned fold.", {"size": 12, "bold": False, "color": MUTED})],
        size=13.5, bold=True, color=INK, space_after=3, line_spacing=1.12)
footer(s, "Sources: chapterA_v2.tex :514-545 (master table), :1400-1404 (@100), :880-882, :1414-1415")


# --- A4b: the gap map -------------------------------------------------
s, y = slide(prs, "Ranking quality against interview compatibility",
             "The same fifteen systems placed on two axes: measured full-profile NDCG@10 on our "
             "ruler, and how many of the three interview requirements a system meets.",
             accent=ACCENT)

fw = Inches(6.55)
figure(s, "a_gap_map.png", MARGIN_L + Inches(0.15), y + Inches(0.10), width=fw,
       caption="Figure from Chapter 1 (chapterA_v2.tex, fig:gapmap). Systems in the grey band "
               "have no full-profile ranker at all and are placed on the x axis only.")

rx = MARGIN_L + fw + Inches(0.55)
rw = CONTENT_W - fw - Inches(0.55)
textbox(s, rx, y + Inches(0.10), rw, Inches(0.3),
        "THE THREE REQUIREMENTS", size=11, bold=True, color=ACCENT)
table(s, [
    ["", ""],
    ["R2", "ranks the whole catalogue from an evidence set, with no refit at inference"],
    ["R3", "accepts item, concept and free-text evidence through one interface"],
    ["R4", "is defined at every turn t ≥ 0, including the empty interview"],
], rx, y + Inches(0.45), rw, col_w=[10, 90], font_size=11, row_h=0.44,
    first_col_bold=True, header_rows=0)

textbox(s, rx, Inches(4.05), rw, Inches(2.4),
        ["The two axes are close to independent.",
         ("The strongest full-profile rankers — RecVAE, EASE, EDLAE — meet none of the three, "
          "because they were built to score a fixed profile rather than to be interviewed. The "
          "systems built for interviews sit low on the y axis. The gap in the top-right corner "
          "is what Chapter 1 fills.",
          {"size": 11.5, "bold": False, "color": MUTED}),
         ("", {"size": 5}),
         ("The grey band holds systems for which no full-profile NDCG can be computed at all: "
          "ConTS, UNICORN, Bıyık's soft attributes, ICF, BCIE, PEBOL, GATE and LLM-CRS.",
          {"size": 11, "bold": False, "color": MUTED})],
        size=13, bold=True, color=INK, space_after=4, line_spacing=1.15)
footer(s, "Source: chapterA_v2.tex fig:gapmap (p. 14 of the compiled chapter), tab:master")


# --- A5: the certified pair -------------------------------------------
s, y = slide(prs, "The certified tower against RecVAE, and the effect of graded input",
             "The tower ties RecVAE on tail and gives up 1.2 % relative on full. Using rating "
             "levels rather than binary likes recovers about half of that.", accent=ACCENT)

lw = Inches(6.3)
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

textbox(s, MARGIN_L, Inches(4.90), lw, Inches(0.3),
        "WHAT THIS LICENSES", size=11, bold=True, color=ACCENT)
bullets(s, MARGIN_L, Inches(5.25), lw, [
    ("The instrument is at the frontier,", "so a Chapter 2 null cannot be "
     "dismissed as a weak recommender."),
    ("Signed, graded evidence is load-bearing.", "on binary information this tower "
     "would score 0.3457, i.e. 0.0083 behind RecVAE."),
    ("Checkpoint selection.", "epoch 23 scores higher on TEST (0.3505); epoch 31 "
     "is kept because it wins the pre-registered validation rule."),
], size=12)

rx = MARGIN_L + lw + Inches(0.40)
rw = CONTENT_W - lw - Inches(0.40)
figure(s, "a_graded_curve.png", rx, y + Inches(0.10), width=Inches(5.20),
       caption="Figure from Chapter 1 (chapterA_v2.tex, fig:gradedcurve): full NDCG@10 against "
               "interview length under graded and binarised input. The two encodings converge "
               "as the interview lengthens; the premium is largest on the short interview.")
footer(s, "Sources: chapterA_v2.tex :861, :1411-1413, :1426-1443, fig:gradedcurve · "
          "docs/results/PAPER_A_STATE_2026-08-04.md :18-37")


# --- A6: the concept channel -------------------------------------------
s, y = slide(prs, "Concept affinity: behavioural signal against LLM judgement",
             "Asking about a tag requires a per-user affinity estimate. Implicit watch behaviour "
             "carries roughly 22× the signal of an LLM's stated judgement.", accent=ACCENT)

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


# --- A7: the fold-in machinery ----------------------------------------
s, y = slide(prs, "The fold-in operator",
             "Evidence in, latent point out. One operator per channel, no gradient step at "
             "inference, defined from the first question onwards.", accent=ACCENT)

fw = Inches(6.55)
gyf = figure(s, "a_architecture.png", MARGIN_L, y + Inches(0.06), width=fw,
             caption="Figure from Chapter 1 (chapterA_v2.tex, fig:architecture): four evidence "
                     "channels enter one observation slot; a permutation-invariant fold returns "
                     "one latent point, which the frozen decoder scores.")

rx = MARGIN_L + fw + Inches(0.45)
rw = CONTENT_W - fw - Inches(0.45)
textbox(s, rx, y + Inches(0.06), rw, Inches(0.3),
        "THE FREE-TEXT CHANNEL, WORKED THROUGH", size=11, bold=True, color=ACCENT)
table(s, [
    ["phrase typed by the user", "nearest concept", "cos", "top films returned"],
    ["“the quiet dignity of losing”", "redemption", "0.47", "Umberto D. · Tyson · Red Army"],
    ["“people trapped by a decision\nthey made years ago”", "history", "0.39",
     "Music Box · Judgment at Nuremberg"],
    ["“two people talking for the\nwhole film”", "dialogue", "0.54",
     "Sweet Smell of Success · Trouble in Paradise"],
    ["“movies about grief”", "feel good movie", "0.54",
     "Rachel, Rachel · Mean Creek · The Woodsman"],
], rx, y + Inches(0.41), rw, col_w=[30, 18, 8, 44], font_size=9.6, row_h=0.40,
    align=[L, L, R, L], first_col_bold=True)

textbox(s, rx, Inches(3.82), rw, Inches(0.55),
        "A held-out concept direction predicted from its name alone reaches top-10 overlap "
        "0.167 against a random baseline of 0.0005, and cosine 0.638 to the true direction.",
        size=10.5, color=MUTED, line_spacing=1.15)

table(s, [
    ["Design decision", "Consequence, measured"],
    ["φ is channel-agnostic: an item factor, a whitened concept centroid, a text-adapter "
     "image, or a raw policy direction all enter the same slot",
     "one code path serves items, concepts, free text and Chapter 3's synthesised directions"],
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
], MARGIN_L, Inches(4.55), CONTENT_W, col_w=[44, 56],
    font_size=10.0, row_h=0.32, first_col_bold=True)
footer(s, "Sources: chapterA_v2.tex :243-249, :865-882, :1094-1116, :305-308, "
          "fig:architecture, tab:freetext (:959-977)")
notes(s, "NOTE TO AUTHOR: Chapter 1 contains no worked fold-in example — no sample user, no "
         "answer sequence, no before/after top-N. The nearest concrete artefact is tab:freetext "
         "(a phrase to ranking demo), which is what this slide uses. If you want a genuine "
         "answer-sequence walkthrough on this slide, it has to be produced; nothing in the "
         "chapter can be quoted for it.")


# --- A8: what A licenses + status ---------------------------------------
s, y = slide(prs, "What Chapter 1 hands forward, and its status",
             "Chapter 1 makes the gap measurable. Closing it is the work of Chapters 2 and 3.", accent=ACCENT)

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
     "raw directions enter the same slot — what Chapter 3 emits"],
    ["Attributable nulls",
     "a Chapter 2 null is a statement about the policy, not about the recommender"],
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
        ["BLIND CALIBRATION PANEL (2026-07-29)",
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

# =====================================================================
# PAPER B — WHAT SHOULD A COLD INTERVIEW OPTIMISE
# =====================================================================

sd = divider("Chapter 2 — the cold interview",
             "Twenty-one selection rules from thirteen published methods, rebuilt on one ruler; "
             "then one variable changed.",
             ACCENT_B, "16 slides  ·  ~20 minutes  ·  status: closed, target ECIR 2027, 2 October 2026")


# --- B1: the question -------------------------------------------------
s, y = slide(prs, "The question Chapter 2 asks",
             "Published selectors optimise a surrogate: variance, entropy, information gain, a "
             "pairwise loss. Almost none optimises the quantity the field reports.", accent=ACCENT_B)

lw = Inches(6.5)
textbox(s, MARGIN_L, y + Inches(0.05), lw, Inches(0.3),
        "THE CENSUS OF SELECTION RULES", size=11, bold=True, color=ACCENT_B)
bullets(s, MARGIN_L, y + Inches(0.42), lw, [
    ("1 of 21 distinct selection rules", "optimises the quantity the field actually "
     "reports — ranking quality on the recommender that consumes the answers."),
    ("12 of 23 battery rows are value-free", "— they choose a question without "
     "reference to how much better the recommendations get."),
    ("The experiment this suggests:", "build the selector that optimises the "
     "reported metric directly, and measure what changes."),
    ("The diagnostic question:", "whether the shortfall is the criterion, the action "
     "space, or answerability."),
], size=13)

rx = MARGIN_L + lw + Inches(0.45)
rw = CONTENT_W - lw - Inches(0.45)
textbox(s, rx, y + Inches(0.05), rw, Inches(0.3),
        "THE FIVE SECTIONS OF THE CHAPTER", size=11, bold=True, color=ACCENT_B)
table(s, [
    ["Section", "What it establishes"],
    ["the audit", "adaptive selectors lose to a good static order — 30 of 30 comparisons, "
     "and 138 cells when widened"],
    ["the diagnosis", "the surrogates do not track the metric, and cold users are "
     "nearly identical, so there is little to adapt to"],
    ["the demonstration", "changing one variable, the split criterion, makes the tree "
     "beat the static order on TEST"],
    ["answerability", "which channel wins is decided by who can answer, not by "
     "who is informative"],
    ["unprompted requests", "what real users volunteer without being asked, and one "
     "policy that uses it"],
], rx, y + Inches(0.42), rw, col_w=[24, 76], font_size=11, row_h=0.50, first_col_bold=True)

figure(s, "b_contract.png", MARGIN_L, Inches(4.28), width=Inches(5.45),
       caption="Figure from Chapter 2 (chapterB_v2.tex, fig:contract): the measurement contract "
               "every row in the battery runs under.")

textbox(s, rx, Inches(4.45), rw, Inches(2.1),
        ["THE CONTRACT, FIXED THROUGHOUT",
         ("Consumer = the certified tower i26c ep31. Protocol candidate pool: every rated "
          "fold-in item removed. Behavioural answers. All 10,000 TEST users. Nothing sampled. "
          "Answer-free reference 0.1626 / 0.0262.",
          {"size": 11.5, "bold": False, "color": MUTED})],
        size=11, bold=True, color=ACCENT_B, space_after=4, line_spacing=1.14)
footer(s, "Sources: chapterB_v2.tex :63, :93, :396, :424, fig:contract · "
          "docs/results/BASELINE_OBJECTIVE_AUDIT.md")


# --- B2: the battery roster --------------------------------------------
s, y = slide(prs, "The replication battery",
             "Twenty-three selector rows, twenty-one distinct rules, thirteen published methods. "
             "Every one re-implemented here and run on the same ruler, users and candidate pool.", accent=ACCENT_B)

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
        ["The table has two jobs.",
         ("It is the audit — one ruler, so the comparison is like-for-like. "
          "It is also the coverage ledger: the chapter prints a second table of what was "
          "NOT built — ICF's four policies, SEL-BALD, PEBOL's acquisition, fMF, PERE, Greedy "
          "SLIM/QBandit, RBMF/RMVA, GreedyExtend, FacT-CRS, UpsRec, MeLU, a-stratification — "
          "each marked ✗ built, ✗ ours.",
          {"size": 12, "bold": False, "color": MUTED})],
        size=13.5, bold=True, color=INK, space_after=3, line_spacing=1.14)
footer(s, "Sources: chapterB_v2.tex tab:battery (:601-656), tab:coverage (:724-769)")


# --- B3: battery measured, part 1 ---------------------------------------
s, y = slide(prs, "Battery results, 1 of 2: static orders and heuristics",
             "Full / tail NDCG@10, 10,000 TEST users, protocol candidate pool, no sampling. "
             "The optimised static order leads every row.", accent=ACCENT_B)

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


# --- B4: battery measured, part 2 ---------------------------------------
s, y = slide(prs, "Battery results, 2 of 2: published methods, validated natively then ported",
             "Each method was reproduced on its own protocol first, then ported onto our ruler. "
             "None reaches the optimised static order.", accent=ACCENT_B)

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


# --- B4b: the budget curves --------------------------------------------------
s, y = slide(prs, "Full NDCG@10 against question budget",
             "The same battery drawn as curves. Ten thousand TEST users, one contract, "
             "logarithmic budget axis.", accent=ACCENT_B)

fw = Inches(6.60)
figure(s, "b_budget_curves.png", MARGIN_L, y + Inches(0.08), width=fw,
       caption="Figure from Chapter 2 (chapterB_v2.tex, fig:budgetcurves). Black lines are the "
               "three optimised static orders; coloured lines are the literature; the dashed "
               "grey line is the clairvoyant oracle ceiling.")

rx = MARGIN_L + fw + Inches(0.50)
rw = CONTENT_W - fw - Inches(0.50)
textbox(s, rx, y + Inches(0.08), rw, Inches(0.3),
        "THREE THINGS THE CURVES SHOW", size=11, bold=True, color=ACCENT_B)
bullets(s, rx, y + Inches(0.44), rw, [
    ("The three static orders are the top three lines at every budget.",
     "No adaptive arm crosses them anywhere on the range."),
    ("The gap to the oracle does not close.",
     "At K=16 the ceiling is 0.3292 against the best measured arm's 0.2590, so about "
     "70 % of the available headroom is still unclaimed."),
    ("The ordering is stable in K.",
     "Golbandi's tree is the one line that changes rank with budget, and it stops at "
     "K=8 because leaf fragmentation caps its depth."),
], size=11.5)

textbox(s, rx, Inches(4.70), rw, Inches(1.9),
        ["The curves are the audit result in one picture.",
         ("A fixed, non-adaptive question order beats every adaptive selector in the "
          "literature at every budget measured, under one consumer, one candidate pool "
          "and one answer model.",
          {"size": 11.5, "bold": False, "color": MUTED})],
        size=13, bold=True, color=INK, space_after=4, line_spacing=1.15)
footer(s, "Source: chapterB_v2.tex fig:budgetcurves (p. 12 of the compiled chapter), tab:battery")


# --- B5: the audit on TEST ---------------------------------------------------
s, y = slide(prs, "Adaptive selectors against a static order, on the test users",
             "Thirty comparisons against one static order. All thirty separate, and all thirty in "
             "the same direction. The ordering repeats on four other consumers.", accent=ACCENT_B)

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
    ("This is what Chapter 1 is for.", "The verdict is a property of a "
     "strategy-and-instrument pair, and the instrument is stated."),
], size=12)
footer(s, "Sources: chapterB_v2.tex tab:act1 (:548-578), :569-572, :932-945 · ACT1_PAIRED_CI.md")

# --- B6: diagnosis — the split criteria --------------------------------
s, y = slide(prs, "Split criteria compared with measured NDCG gain",
             "At all 22 nodes of the grown tree, every candidate's true NDCG gain was measured, "
             "then each published criterion's own pick was recorded.", accent=ACCENT_B)

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
        "THE CRITERION AGAINST ITS APPROXIMATION", size=11, bold=True, color=ACCENT_B)
bullets(s, rx, Inches(5.0), rw, [
    ("Measured properly, the pairwise loss is fine:", "at the root it tracks measured "
     "NDCG@10 at Spearman +0.784 and forgoes only +0.00459."),
    ("Hu & Yu's Taylor-approximated training-loss version", "correlates at −0.149 and "
     "forgoes +0.01473. The approximation, not the idea, is what fails."),
], size=12)
footer(s, "Sources: chapterB_v2.tex tab:criterionregret (:1056-1086), :1283-1292")


# --- B7: diagnosis — dispersion -------------------------------------------
s, y = slide(prs, "Question dispersion across cold users",
             "The number of genuinely different best-next-questions a population contains. "
             "At cold start it is about one.", accent=ACCENT_B)

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
        "READING THE TABLE", size=11, bold=True, color=ACCENT_B)
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
        "HEADROOM RUNS THE OTHER WAY", size=11, bold=True, color=ACCENT_B)
table(s, [
    ["state", "best question available", "gain"],
    ["cold, after two openers", "a concept", "+0.0283"],
    ["request-placed", "an item", "+0.0138"],
    ["request-placed", "best concept", "+0.0076"],
], rx, Inches(3.95), rw, col_w=[38, 38, 24], font_size=11, row_h=0.32, align=num(3))
textbox(s, rx, Inches(5.45), rw, Inches(1.3),
        "Dispersion and headroom are different quantities. A placed user is more "
        "distinguishable but has less left to gain, which is why the opener arm is worth "
        "less than the dispersion table suggests.",
        size=11.5, color=MUTED, line_spacing=1.15)
footer(s, "Sources: chapterB_v2.tex tab:dispersion (:1103-1129), :1107-1108, :1149-1154")


# --- B8: the NDCG tree -------------------------------------------
s, y = slide(prs, "Splitting the interview tree on measured NDCG",
             "Same tree learner, same candidates, same stopping rule. The only change is what the "
             "split criterion optimises.", accent=ACCENT_B)

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
        ["Cost, and the comparator correction.",
         ("Validation gain +0.02794 full / +0.01055 tail became +0.01035 / +0.00705 on TEST — "
          "2.70× and 1.50× optimism. And when the static comparator was rebuilt under the "
          "correct mask the tail advantage fell from +0.02092 to +0.00705: two thirds of the "
          "reported tail advantage was comparator weakness, not method.",
          {"size": 12, "bold": False, "color": MUTED})],
        size=13.5, bold=True, color=INK, space_after=3, line_spacing=1.14)
footer(s, "Sources: chapterB_v2.tex tab:tree (:1195-1215), :1166-1185, :1366-1393 · "
          "docs/results/CONDITIONED_ARMS_ON_TEST.md")


# --- B9: the six-criterion ablation ---------------------------------------
s, y = slide(prs, "The six-criterion split ablation",
             "Six split criteria, everything else held identical. All 56 non-nested intervals in "
             "the matrix exclude zero.", accent=ACCENT_B)

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
        "WHAT THE MATRIX SHOWS", size=11, bold=True, color=ACCENT_B)
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
        "COMPUTE COST OF THE EXACT OBJECTIVE", size=11, bold=True, color=ACCENT_B)
table(s, [
    ["", ""],
    ["Taylor construction", "scores all 1,000 candidates in ~11 s"],
    ["Our exact objective", "37,191 s to grow one tree — ~3 orders of magnitude per node"],
    ["Nothing sampled", "the pairwise loss sums over all 137,591,707 fold-in pairs"],
    ["Reproduction", "the ablation reproduces every published tree number to 2.81×10⁻⁷"],
], rx, Inches(3.90), rw, col_w=[30, 70], font_size=11, row_h=0.40,
    first_col_bold=True, header_rows=0)
textbox(s, rx, Inches(5.85), rw, Inches(0.8),
        "The surrogates exist because the exact objective is three to four orders of "
        "magnitude more expensive per node.",
        size=11.5, color=MUTED, line_spacing=1.15)
footer(s, "Sources: chapterB_v2.tex tab:splitcrit (:1250-1313), :1328-1335 · "
          "docs/results/HU_YU_2013_REPLICATION_AND_ABLATION.md")


# --- B10: bank vs conditioning ---------------------------------------------
s, y = slide(prs, "Changing the question bank compared with conditioning",
             "Three changes to a cold interview, measured on the same test users against the "
             "same comparator.", accent=ACCENT_B)

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
        ["On the tail the ratio is about 3×:  +0.02199 for changing the bank against "
         "+0.00705 for conditioning.",
         ("Both changes are real and both intervals exclude zero. The ordering is the opposite "
          "of where the field's effort goes.",
          {"size": 12.5, "bold": False, "color": MUTED})],
        size=15, bold=True, color=INK, space_after=4, line_spacing=1.14)

lw = Inches(6.5)
textbox(s, MARGIN_L, Inches(4.55), lw, Inches(0.3),
        "TWO QUALIFICATIONS", size=11, bold=True, color=ACCENT_B)
bullets(s, MARGIN_L, Inches(4.90), lw, [
    ("Enlarging the question bank further made the selectors that consume it perform worse.",
     "This is why Chapter 3 claims capability and not value."),
    ("The gain is answerability, not information.", "Concepts are less informative "
     "per answer and win anyway, because they get answered."),
], size=12.5)

rx = MARGIN_L + lw + Inches(0.45)
rw = CONTENT_W - lw - Inches(0.45)
textbox(s, rx, Inches(4.55), rw, Inches(1.9),
        ["FOR SCALE, A SEEDED USER",
         ("Seed a cold user with 8 items from their own history, then run the identical 16 "
          "questions:", {"size": 11, "bold": False, "color": INK}),
         ("+0.0206 full   ·   +0.0353 tail", {"size": 14, "bold": True, "color": ACCENT_B}),
         ("The recommender's own knowledge of the user dwarfs every interview-design choice "
          "on this slide.", {"size": 11, "bold": False, "color": MUTED})],
        size=11, bold=True, color=ACCENT_B, space_after=5, line_spacing=1.15)
footer(s, "Sources: chapterB_v2.tex tab:bankdecomp (:1344-1361), :1935-1938 · "
          "memory: concept-bank-does-not-rescue-adaptive-pv")

# --- B11: answerability ----------------------------------------------
s, y = slide(prs, "Answerability and the ordering of channels",
             "Three independent measurements agree: which channel wins is decided by whether the "
             "question can be answered, not by how informative it is.", accent=ACCENT_B)

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
    ["The mechanism",
     "sixteen item questions collect 3.11 answers; sixteen concept questions collect 12.36, "
     "a factor of four"],
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
        "A null on full and a clear concept win on tail. Concepts also lead on full at every "
        "budget below 16 (+0.0048, +0.0078, +0.0104, +0.0112 at k=1,2,4,8, every interval "
        "excluding zero); items catch up only at sixteen questions.",
        size=11.5, color=MUTED, line_spacing=1.15)

rx = MARGIN_L + lw + Inches(0.45)
rw = CONTENT_W - lw - Inches(0.45)
textbox(s, rx, Inches(4.15), rw, Inches(0.3),
        "REACH AGAINST QUESTION CHOICE", size=11, bold=True, color=ACCENT_B)
bullets(s, rx, Inches(4.50), rw, [
    ("Answering AT ALL is worth ~+0.25.", "The largest question-CHOICE effect anywhere "
     "in the tree is +0.048."),
    ("1,785 users answer neither of the first two questions;", "1,438 of them cannot "
     "answer the third. ~14 % of the cohort is untouched, sitting at 0.087."),
], size=12)
footer(s, "Sources: chapterB_v2.tex :1469-1521, :834-930, :1385-1393")


# --- B12: the request corpus --------------------------------------------
s, y = slide(prs, "A corpus of unprompted film requests",
             "Everything above assumes a closed question bank and a user who waits to be asked. "
             "This section measures what users volunteer without being asked.", accent=ACCENT_B)

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

textbox(s, MARGIN_L, Inches(4.30), lw, Inches(0.3),
        "FIVE REQUESTS FROM THE CORPUS, VERBATIM", size=11, bold=True, color=ACCENT_B)
table(s, [
    ["the request as posted", "slots found"],
    ["“[Request] I'm looking for thrillers like 10 Cloverfield Lane "
     "(I really loved that film).”", "film + genre"],
    ["“[REQUEST] good horror/suspense movie on Netflix. i can't find "
     "anything good please help”", "genre + constraint"],
    ["“Films like; 12 Angry Men. Whether it's films told on one set, or films "
     "heavy dialogue. Cheers!”", "film + 2 constraints"],
    ["“Looking for a Horror movie with a similar tone to Evil Dead 2… a super gory "
     "horror movie with a slightly tongue-and-cheek tone.”", "film + genre + 2 tone"],
    ["“[Request] I need something that's gonna punch me in the face. Something with "
     "unexpected plot twist…”", "2 tone, no film"],
], MARGIN_L, Inches(4.65), lw, col_w=[70, 30], font_size=9.4, row_h=0.40)

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
        ["REQUEST DISTRIBUTION AGAINST CONSUMPTION DISTRIBUTION",
         ("Horror is 27.5 % of mapped genre mentions and Thriller 15.8 %. In ML-25M validation, "
          "Horror is the favourite genre of 51 users out of 10,000; Drama is the favourite of "
          "5,481.", {"size": 11.5, "bold": False, "color": INK}),
         ("", {"size": 4}),
         ("The request distribution is inverted relative to the consumption distribution, so a "
          "system tuned on watch logs holds the wrong prior for what gets asked.",
          {"size": 11, "bold": False, "color": MUTED})],
        size=11, bold=True, color=ACCENT_B, space_after=4, line_spacing=1.15)

textbox(s, rx, Inches(5.75), rw, Inches(1.1),
        ["ENTITY LINKING, MEASURED AGAINST A HAND-LABELLED STANDARD",
         ("Against 500 hand-labelled mentions the corpus's own links are 0.572 precise; the "
          "two-tier pipeline (deterministic rules + LM adjudication) is 0.960, against a "
          "pre-registered ≥0.95 gate. Attribute mapping succeeds for 95.1 % of genres, "
          "70.2 % of eras and 66.1 % of tones; the era and tone shortfalls bound this "
          "section from above and are stated as such.",
          {"size": 10, "bold": False, "color": MUTED})],
        size=10.5, bold=True, color=ACCENT_B, space_after=3, line_spacing=1.12)
footer(s, "Sources: chapterB_v2.tex :1542-1588 · experiments/battery/m1b_census.json · "
          "data/reddit_he/slots/full_input.jsonl (verbatim requests) · REDDIT_LINKER_AUDITION.md")
notes(s, "NOTE TO AUTHOR: the five requests are quoted verbatim from "
         "data/reddit_he/slots/full_input.jsonl (lines 16, 102, 126, 157, 65), including their "
         "original spelling and the [Request] tags. Two are shortened at an ellipsis. Confirm "
         "you are content to show public forum text verbatim in a supervisor deck; the "
         "alternative is to paraphrase, which costs the point that these are real.")


# --- B13: openers + the one policy -------------------------------------------
s, y = slide(prs, "Opener-fitted interviews and the distilled policy",
             "Each tree is fitted at the state the user is actually in; the six trees are then "
             "distilled into a single policy that has to identify the group itself.", accent=ACCENT_B)

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
        ["The two steps:  cold − static = +.00495 ★   then   opener − cold = +.00688 ★",
         ("★ = paired bootstrap interval excludes zero. A tree fitted at the serving state beats "
          "an anchor teacher; fitting it to the opener-conditioned state beats that in turn.",
          {"size": 11, "bold": False, "color": MUTED})],
        size=12.5, bold=True, color=INK, space_after=4, line_spacing=1.14)

figure(s, "b_act5_pipeline.png", MARGIN_L, Inches(4.82), width=Inches(5.20),
       caption="Figure from Chapter 2 (chapterB_v2.tex, fig:act5pipeline): how a synthesised "
               "opener state is built and what is run from it.")

rx = MARGIN_L + lw + Inches(0.40)
rw = CONTENT_W - lw - Inches(0.40)
textbox(s, rx, y + Inches(0.02), rw, Inches(0.3),
        "TEST-SET RESULTS  —  one distilled policy, reserved TEST cohort, K=15",
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
        ["THE TEST-SET CONTRAST IN DETAIL",
         ("A single reserved look, spent once, on a policy that must route as well as ask. "
          "Non-monotone in budget: K=1 +0.00161 ★, K=4 +0.00119 (contains zero), K=15 +0.00343 ★.",
          {"size": 10.5, "bold": False, "color": INK}),
         ("Intervals (paired bootstrap, from the run artefact): ALL K=15 +0.00343 "
          "[+0.00173, +0.00509]; warm users +0.00946 [+0.00524, +0.01376]; "
          "do-no-harm cell −0.00038 [−0.00098, +0.00012], i.e. a clean null.",
          {"size": 10, "bold": False, "color": ACCENT_C})],
        size=11, bold=True, color=ACCENT_B, space_after=4, line_spacing=1.14)
footer(s, "Sources: chapterB_v2.tex tab:volunteered, tab:policy, fig:act5pipeline · "
          "M4B_FOREST_POLICY.md · M6L_OPENER_FITTED_TEACHER.md · CIs: "
          "experiments/battery/m6/m6mix_B_test.json cells.ALL/warm/COLD K=15")


# --- B13b: the routing space + example question paths -------------------------
s, y = slide(prs, "The routing space, and three example question paths",
             "What the router reads, what happens when it is scrambled, and what the six group "
             "interviews actually ask.", accent=ACCENT_B)

lw = Inches(7.45)
figure(s, "b_route_atlas.png", MARGIN_L, y + Inches(0.06), width=Inches(7.00),
       caption="Figure from Chapter 2 (chapterB_v2.tex, fig:atlas): the same 3,980 users in two "
               "spaces. Left, the full rating profile, where the six groups are not separable. "
               "Right, the folded opener state, where a discriminant fitted on a disjoint half "
               "assigns 0.907 of held-out users correctly against a chance floor of 0.167.")

textbox(s, MARGIN_L, Inches(5.52), lw, Inches(0.3),
        "THE SCRAMBLE CONTROL", size=11, bold=True, color=ACCENT_B)
textbox(s, MARGIN_L, Inches(5.85), lw, Inches(1.1),
        ["Scrambling the routing collapses it on every group — .888→.148, .834→.040, .857→.038, "
         ".852→.121, .958→.094, .757→.111, against a chance floor of .167 — and the value goes "
         "negative on marvel (−.01190) and inspiring (−.00572), both separating.",
         ("Stated limit: the synthesised openers are group-stamped, so the routing is partly "
          "circular and this control cannot detect that. The opener-minus-cold contrast survives "
          "because the state is held fixed. The claim is adaptivity beyond the closed tree, not "
          "routing.", {"size": 10, "bold": False, "color": ACCENT_C})],
        size=10.3, color=MUTED, space_after=4, line_spacing=1.13)

rx = MARGIN_L + lw + Inches(0.45)
rw = CONTENT_W - lw - Inches(0.45)
textbox(s, rx, y + Inches(0.10), rw, Inches(0.3),
        "THREE QUESTION PATHS, AS THE TREES ASK THEM", size=11, bold=True, color=ACCENT_B)
textbox(s, rx, y + Inches(0.48), rw, Inches(3.6),
        [("art house   n = 587, depth 2", {"size": 11, "bold": True, "color": ACCENT_B}),
         ("“black and white” → no → “fight scenes” → no → “disturbing”",
          {"size": 11, "color": INK}),
         ("", {"size": 6}),
         ("teenagers   n = 786, depth 3", {"size": 11, "bold": True, "color": ACCENT_B}),
         ("“fantasy world” → yes → Harry Potter: Deathly Hallows Pt 2;   no → “80s”",
          {"size": 11, "color": INK}),
         ("", {"size": 6}),
         ("gore   n = 747, depth 3", {"size": 11, "bold": True, "color": ACCENT_B}),
         ("“shallow” → yes → Batman Begins;   no → “fun movie” → no → “based on a true story”",
          {"size": 11, "color": INK})],
        size=11, color=INK, space_after=3, line_spacing=1.16)

textbox(s, rx, Inches(4.55), rw, Inches(2.4),
        ["The first two paths read as thematic; the third does not.",
         ("The chapter states this plainly: an earlier reading called all six coherent, and that "
          "was an over-read. The tree optimises separation, not meaning.",
          {"size": 10.5, "bold": False, "color": MUTED}),
         ("", {"size": 5}),
         ("For contrast, the genre-conditioned static orders ask Drama users about Clear and "
          "Present Danger first, Comedy users about Return of the Jedi, and Romance users about "
          "Speed — the same question of everyone in the group, whatever they answer.",
          {"size": 10.5, "bold": False, "color": MUTED})],
        size=11.5, bold=True, color=INK, space_after=4, line_spacing=1.15)
footer(s, "Sources: chapterB_v2.tex fig:atlas, :1991-2010 · docs/results/M3_FOREST_TREES.md "
          ":87-92 (paths), :60-65 (cohorts) · experiments/battery/m3v2_*.json")
notes(s, "NOTE TO AUTHOR: two provenance points on this slide. (1) The question paths are from "
         "M3_FOREST_TREES.md, which is the ANCHOR-teacher forest, whereas the table on the "
         "previous slide is the OPENER-FITTED forest — the opener-fitted teenagers root is the "
         "item 'Harry Potter and the Deathly Hallows: Part 1' (answer rate 0.299), not the "
         "concept 'fantasy world'. Check you are happy showing the anchor-teacher paths here, or "
         "tell me to regenerate them from the opener-fitted trees. (2) The cohort sizes on this "
         "slide (587 / 786 / 747) are the M3 cohorts and differ from the previous slide's "
         "validation ns (473 / 540 / 419); they are different populations, not a typo.")


# --- B14: B status ------------------------------------------------------------
s, y = slide(prs, "Chapter 2 status",
             "Closed 30 August 2026. Sections 1–4 are on TEST and frozen; the test look is "
             "spent and replicates.", accent=ACCENT_B)

lw = Inches(7.1)
table(s, [
    ["", ""],
    ["Chapter state", "CLOSED 2026-08-30. Compiles; test look spent 2026-08-12 (sections 1–4) "
     "and 2026-08-29 (the unprompted-request section), and replicates."],
    ["Frozen scope", "the audit, diagnosis, demonstration and answerability sections are on "
     "TEST and frozen. Only the unprompted-request section was rebuilt."],
    ["Target", "ECIR 2027 full paper — deadline 2 October 2026"],
    ["The cut", "Sections 1–3 (audit · diagnosis · one-variable demonstration) compressed to "
     "~12 pp. Estimated 40–55 % acceptance."],
    ["Second choice", "RecSys, or its reproducibility track"],
    ["Unprompted requests", "held back as a second submission (RecSys / CIKM) rather than cut"],
    ["Absorbed", "the free-recall workstream is absorbed into B — the request corpus and the "
     "opener census cover its substance, and its own numbers were on the retired pre-July ruler"],
], MARGIN_L, y + Inches(0.05), lw, col_w=[22, 78], font_size=11.5, row_h=0.44,
    first_col_bold=True, header_rows=0)

rx = MARGIN_L + lw + Inches(0.40)
rw = CONTENT_W - lw - Inches(0.40)
textbox(s, rx, y + Inches(0.05), rw, Inches(0.3),
        "THE STRONGEST OPEN OBJECTION", size=11, bold=True, color=ACCENT_C)
textbox(s, rx, y + Inches(0.42), rw, Inches(1.9),
        ["The demonstration rests on one consumer.",
         ("The chapter's own inversion section demonstrates consumer-dependence. It is fixable by "
          "scoring — not refitting — the frozen tree through RecVAE: one machine-night, not a "
          "redesign.", {"size": 11.5, "bold": False, "color": MUTED})],
        size=13, bold=True, color=INK, space_after=4, line_spacing=1.15)

textbox(s, rx, Inches(3.55), rw, Inches(0.3),
        "CLAIMS NOT MADE", size=11, bold=True, color=ACCENT_B)
bullets(s, rx, Inches(3.90), rw, [
    "Routing users to a group interview is not a contribution — Golbandi WSDM'11 and "
    "Cluster-Based Bandits SIGIR'21 both do it.",
    "No claim that our grouping beats k-means on NDCG.",
    "Never “reproduces the optimal question order”.",
    "Never “parity” or “tie” as a headline.",
    "No fitting-population reasoning — one flat protocol sentence.",
], size=11.5)
footer(s, "Sources: docs/design/PAPER_B_MASTER_TODO.md · docs/PUBLICATION_PLAN.md roster (2026-08-18)")

# =====================================================================
# PAPER C — THE QUESTION THAT ISN'T IN THE CATALOGUE
# =====================================================================

sd = divider("Chapter 3 — questions outside the catalogue",
             "The policy emits a direction; the nearest catalogue question points elsewhere. "
             "The direction is rendered into English and the result is audited.",
             ACCENT_C, "10 slides  ·  ~15 minutes  ·  status: short-form draft written, venue opportunistic")


# --- C1: the question ---------------------------------------------------
s, y = slide(prs, "The question Chapter 3 asks",
             "Every elicitation system in the literature picks a question from a list. This "
             "chapter asks what happens when the best thing to ask is not on the list.", accent=ACCENT_C)

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
        "THE CLAIM, STATED AS A CONJUNCTION", size=11, bold=True, color=ACCENT_C)
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
        "PRIOR ART FOR EACH LEG SEPARATELY", size=11, bold=True, color=ACCENT_C)
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

textbox(s, MARGIN_L, Inches(6.10), lw, Inches(0.9),
        ["CAPABILITY, NOT VALUE.",
         ("There is no claim anywhere that asking a synthesised question improves "
          "recommendations. Chapter 2 is the reason: there, enlarging the bank made things worse.",
          {"size": 11.5, "bold": False, "color": MUTED})],
        size=12.5, bold=True, color=ACCENT_C, space_after=3, line_spacing=1.14)
footer(s, "Sources: paperC_lbr.tex §1 · chapterC_v2.tex :108-190 (positioning)")


# --- C2: the experiment map -------------------------------------------------
s, y = slide(prs, "Experiments in Chapter 3",
             "Each is gated, pre-registered, and read against its own null. Three of the six "
             "verdicts were decided by a control rather than by the effect.", accent=ACCENT_C)

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
notes(s, "NOTE TO AUTHOR: Chapter 3 contributes no figures to this deck because neither "
         "chapterC_v2.tex nor paperC_lbr.tex contains a single plot or diagram — the only "
         "figure environment in either file is fig:transcript, which is an empty placeholder "
         "reading 'Placeholder for the deployment-trajectory transcript'. Chapter 3's slides "
         "are therefore tables, generated text and the native loop diagram. If you want a "
         "figure here, the obvious candidates are the decoding Pareto plot (C4C) and the "
         "triangulation rank distribution; both would have to be drawn.")


# --- C3: C0 + C1 off-catalogue emission ------------------------------------
s, y = slide(prs, "Where the emitted vector sits, and the emission gates",
             "One object runs the tree's ordinary interview essentially unchanged and emits "
             "gain-weighted blends that are not any bank question.", accent=ACCENT_C)

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
        ["The mechanism, after correction.",
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

textbox(s, MARGIN_L, Inches(4.88), CONTENT_W, Inches(1.5),
        ["THE SCOPE BOUND — THE SAME SHAPE AS CHAPTER 2's HELD-OUT-ACTION RESULT",
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

# --- C4: the mechanism -------------------------------------------------
s, y = slide(prs, "Testing whether the vector conditions the generated text",
             "A two-layer adapter maps the direction into eight soft prefix tokens of a frozen "
             "distilgpt2. Its control twin is trained identically on scrambled pairs.", accent=ACCENT_C)

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
        "WHAT THE FAILING CRITERION MEASURED", size=11, bold=True, color=ACCENT_C)
bullets(s, MARGIN_L, Inches(4.30), lw, [
    ("The control guarded against the twin smuggling signal.", "It fired because the "
     "twin sits BELOW knowing nothing — the opposite threat."),
    ("The twin collapsed to 61 distinct texts over 3,880 vectors;", "the true arm "
     "produced 2,280."),
    ("The rule adopted:", "write an equivalence control with a direction. A "
     "pre-registration is not reinterpreted after the data, so the composite stays FAILURE and "
     "the two mechanism criteria are reported as passing on their own terms."),
], size=12)

rx = MARGIN_L + lw + Inches(0.45)
rw = CONTENT_W - lw - Inches(0.45)
textbox(s, rx, Inches(3.95), rw, Inches(0.3),
        "SCORING ABOVE THE GOLD-PHRASE CEILING, AND ITS CAVEAT", size=11, bold=True, color=ACCENT_C)
textbox(s, rx, Inches(4.30), rw, Inches(2.2),
        ["The adapter's text re-encodes BETTER than the true title does, at 107 % of the gold "
         "phrase's own ceiling — while exact-phrase recovery is 0.0000 and token-F1 is 0.111. "
         "It essentially never emits the right title and carries the vector anyway.",
         ("Caveat: the bridge prefers concept-flavoured text, and invented titles are freer to "
          "be concept-flavoured than real ones, which is also why the exact-phrase floor is zero. "
          "The 107 % figure is not quoted without this sentence.",
          {"size": 10.5, "bold": False, "color": ACCENT_C}),
         ("Concept channel, printed not gated: A = +0.3533 against a required +0.2002; "
          "identification 0.8476 [0.8000, 0.8952] against a 0.7714 floor.",
          {"size": 10.5, "bold": False, "color": MUTED})],
        size=11.5, color=INK, space_after=6, line_spacing=1.15)
footer(s, "Sources: docs/results/C3B_ADAPTER.md · paperC_lbr.tex tab:english panel (a) · "
          "chapterC_v2.tex :862-905")


# --- C5: askable questions ------------------------------------------------
s, y = slide(prs, "Question-shaped output, and its effect on the measured ceiling",
             "The same adapter retrained on question-shaped targets. The mechanism survives, and "
             "the instrument's own ceiling drops by 23 %.", accent=ACCENT_C)

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
        "QUESTION BOILERPLATE LOWERS THE CEILING", size=11, bold=True, color=ACCENT_C)
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

textbox(s, rx, Inches(4.20), rw, Inches(2.2),
        ["THE RULE ADOPTED",
         ("A change to the output format is a change to the instrument, and the ceiling must be "
          "RE-MEASURED before any gate is read.", {"size": 11.5, "bold": True, "color": INK}),
         ("", {"size": 4}),
         ("Without it, the bare-phrase adapter's 107 % and the question adapter's 71 % read as a "
          "regression. They are readings against two different ceilings. No figure of this kind "
          "appears in the verbalisation literature, because it requires a same-instrument ceiling "
          "that the field does not measure.",
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


# --- C6: triangulation -----------------------------------------------------
s, y = slide(prs, "Rank of the rendered word among the point's nearest concepts",
             "An independent second view: whether the rendered word is one of the point's own "
             "nearest concepts. Under greedy decoding the states split into two groups.", accent=ACCENT_C)

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
        ["The limit is generation diversity, not geometry.",
         ("Where greedy falls back it falls back on three stock words — horror ×9, religion ×7, "
          "movies ×4. Strict top-ten membership is 4/42 against 0/42 for random points, with "
          "overlapping Wilson intervals: a harsher bar that does not separate at this n, and we "
          "report it that way.", {"size": 11, "bold": False, "color": MUTED})],
        size=12.5, bold=True, color=INK, space_after=4, line_spacing=1.15)

rx = MARGIN_L + lw + Inches(0.45)
rw = CONTENT_W - lw - Inches(0.45)
textbox(s, rx, y + Inches(0.05), rw, Inches(0.3),
        "RENDERED WORD, AND ITS RANK IN ITS OWN POINT'S 1,031-CONCEPT RANKING",
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
        ["DECODING IS THE LEVER",
         ("Best-of-8 by the frozen LM's own length-normalised likelihood — a rule blind to the "
          "evaluation metric by construction — takes distinct renders from 23 to 37 of 42 and cuts "
          "the translation cost roughly in half, with the faithfulness change not significant "
          "(−0.0193 [−0.097, +0.058]). It is the only Pareto point, and it is the decoding of "
          "record for anything measured.", {"size": 11, "bold": False, "color": MUTED})],
        size=11, bold=True, color=ACCENT_C, space_after=5, line_spacing=1.15)
footer(s, "Sources: docs/results/C4_PILOT.md · C4C_DECODING.md · chapterC_v2.tex :1086-1100 · "
          "paperC_lbr.tex tab:loop")


# --- C7: catalogue text vs rendered word ------------------------------------------------------------
s, y = slide(prs, "Nearest catalogue question compared with the emitted direction",
             "At each interview state, whether the nearest bank entry's own gold phrase carries "
             "the direction the point was built from.", accent=ACCENT_C)

bwh, bhh = Inches(3.6), Inches(1.38)
gyh = y + Inches(0.72)
xh0 = MARGIN_L + Inches(0.55)
box(s, xh0, gyh, bwh, bhh,
    [("THE EMITTED POINT", {"size": 12, "color": ACCENT_C}),
     ("the direction the policy wants", {"size": 10.5, "bold": False, "color": MUTED}),
     ("to ask about at this state", {"size": 10.5, "bold": False, "color": MUTED})],
    line=ACCENT_C)
box(s, xh0 + bwh + Inches(1.1), gyh - Inches(0.70), bwh, bhh,
    [("THE NEAREST CATALOGUE QUESTION", {"size": 11.5}),
     ("cos 0.93 in the emitter's own head space", {"size": 10, "bold": False, "color": MUTED}),
     ("its own gold phrase carries the direction at", {"size": 10, "bold": False, "color": MUTED}),
     ("−0.158   [−0.224, −0.089]", {"size": 15, "color": ACCENT_C})],
    line=MUTED)
box(s, xh0 + bwh + Inches(1.1), gyh + Inches(0.70), bwh, bhh,
    [("THE RENDERED OFF-CATALOGUE WORD", {"size": 11.5}),
     ("not in the bank at all", {"size": 10, "bold": False, "color": MUTED}),
     ("carries that same direction at", {"size": 10, "bold": False, "color": MUTED}),
     ("+0.204   [+0.117, +0.291]", {"size": 15, "color": ACCENT})],
    line=ACCENT)
arrow(s, xh0 + bwh, gyh + bhh // 2, xh0 + bwh + Inches(1.1), gyh - Inches(0.70) + bhh // 2,
      color=MUTED)
arrow(s, xh0 + bwh, gyh + bhh // 2, xh0 + bwh + Inches(1.1), gyh + Inches(0.70) + bhh // 2,
      color=ACCENT)

textbox(s, xh0 + 2 * bwh + Inches(1.5), gyh - Inches(0.15), Inches(3.4), Inches(2.0),
        ["A GAP OF", ("+0.362", {"size": 40, "color": ACCENT_C}),
         ("interval excluding zero", {"size": 11, "bold": False, "color": MUTED})],
        size=12, bold=True, color=INK, align=C, space_after=4)

textbox(s, MARGIN_L, Inches(4.55), CONTENT_W, Inches(0.6),
        "The direction is expressible in English; the text of the nearest catalogue "
        "question does not express it.",
        size=17, bold=True, color=INK, line_spacing=1.12)

lw = Inches(6.5)
textbox(s, MARGIN_L, Inches(5.20), lw, Inches(1.5),
        ["TWO CAVEATS ATTACHED TO THIS NUMBER",
         ("The nearest entry is partly a common-mode artefact of an anisotropic space — it is "
          "frequently the same entry, the concept “divorce” — so “nearest” does less work than "
          "it sounds. The bridge also favours concept-flavoured text, which advantages a one-word "
          "render over a bank phrase.", {"size": 10.5, "bold": False, "color": MUTED}),
         ("The sign survives both. The magnitude is not quoted without them.",
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

# --- C8: the example interview ---------------------------------------
s, y = slide(prs, "Example interview",
             "One cold-start trajectory, six turns. Left: the question the frozen tree would ask. "
             "Right: the question synthesised from the direction the policy emits at that state.", accent=ACCENT_C)

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
textbox(s, MARGIN_L, Inches(4.40), lw, Inches(0.3),
        "TURNS 3, 5 AND 6", size=11, bold=True, color=ACCENT)
bullets(s, MARGIN_L, Inches(4.78), lw, [
    ("sci-fi movies", "against the tree's “alien films”"),
    ("detective drama", "against the tree's “have you seen Crimson Tide (1995)?”"),
    ("children films", "against the tree's “have you seen Star Wars: Episode VI?”"),
], size=13)
textbox(s, MARGIN_L, Inches(5.95), lw, Inches(0.7),
        "The same neighbourhood, but a more askable object. That is the capability claim, "
        "in six lines.",
        size=12.5, bold=True, color=INK, line_spacing=1.15)

rx = MARGIN_L + lw + Inches(0.45)
rw = CONTENT_W - lw - Inches(0.45)
textbox(s, rx, Inches(4.40), rw, Inches(0.3),
        "TURNS 2 AND 4: THE TEMPLATING FAILURE", size=11, bold=True, color=ACCENT_C)
textbox(s, rx, Inches(4.78), rw, Inches(2.0),
        ["“Do you like it when a film is movie?” — boilerplate round a nearly contentless phrase; "
         "turn 4's blend is a diffuse movie-business / best-cinematography direction.",
         ("2 of 6 = 33 %, against C3B-v2's measured 5.3 %. Structural, not a contradiction: 5.3 % "
          "was measured on single clean bank entities and the transcript renders BLENDS (median "
          "top-1 weight 0.225).", {"size": 11, "bold": False, "color": MUTED}),
         ("Six turns is an indication, not a measurement.",
          {"size": 11.5, "bold": True, "color": ACCENT_C})],
        size=11.5, color=INK, space_after=5, line_spacing=1.14)

textbox(s, MARGIN_L, Inches(6.58), CONTENT_W, Inches(0.4),
        "The run artefact's NDCG column, in which the synthesised arm wins 5 of 6 turns, is "
        "declared-circular, unregistered, and n=6 on one path. It is not reported as a result.",
        size=10.5, bold=True, color=ACCENT_C, line_spacing=1.1)
footer(s, "Source: docs/results/C4D_TRANSCRIPT.md (renderer: v2 question adapter, best-of-8 by LM "
          "likelihood; cohort = all 10,000 validation users, majority path) · paperC_lbr.tex tab:demo")


# --- C9: machine-only synonyms ------------------------------------------------
s, y = slide(prs, "Generated text that does not match the source phrase",
             "The adapter almost never emits the right title, and its text reproduces the "
             "vector's effect anyway. Both measurements hold at once.", accent=ACCENT_C)

table(s, [
    ["the source vector's true question is about…", "…and the adapter renders"],
    ["The Shawshank Redemption", "“Have you seen Pulp Fiction (1995)?”"],
    ["Ghost Story (1981)", "“Have you seen Battlestar Galactica (1980)?”"],
    ["Desperate Hours (1955)", "“What did you think of Bunted, The (1941)?”"],
], MARGIN_L, y + Inches(0.05), CONTENT_W, col_w=[42, 58], font_size=13.5, row_h=0.40,
    first_col_bold=True)

textbox(s, MARGIN_L, Inches(3.15), CONTENT_W, Inches(0.6),
        "Neighbouring era, neighbouring register, wrong film: fluent, askable, and incorrect "
        "as a surface form.",
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
        "WHY THE EVIDENCE IS A CONTROL AND A CEILING RATHER THAN A HUMAN SCORE",
        size=11, bold=True, color=ACCENT_C)
textbox(s, rx, Inches(4.25), rw, Inches(2.5),
        ["A rater with no access to the source vector has one verdict available: fluent, natural, "
         "answerable — and about the wrong film. They mark it wrong. The instrument, which sees the "
         "vector, records that the text reproduces its effect at 61 %. Both are correct.",
         ("The asymmetry runs the other way as well: a renderer that ignored its input entirely "
          "and asked “Have you seen The Godfather (1972)?” every time would score well with "
          "raters and at chance with the instrument.",
          {"size": 11, "bold": False, "color": ACCENT_C}),
         ("So a plausibility panel cannot separate our renderer from a confabulator in either "
          "direction, and adding raters does not fix it — they would have to be shown the latent.",
          {"size": 11, "bold": False, "color": MUTED})],
        size=11, color=INK, space_after=5, line_spacing=1.14)

textbox(s, MARGIN_L, Inches(6.35), lw, Inches(0.7),
        "These are Hewitt et al.'s machine-only synonyms, appearing unprompted: text a person "
        "would grade as wrong that reproduces the vector's semantic effect better than the true "
        "surface form does.",
        size=11.5, color=MUTED, line_spacing=1.15)
footer(s, "Sources: docs/results/C3B_V2_QUESTIONS.md · C3B_ADAPTER.md · chapterC_v2.tex :1015-1042")


# --- C10: C status ------------------------------------------------------------
s, y = slide(prs, "Chapter 3 status",
             "A capability result with no benchmark to beat. The short-form draft is written; "
             "publication is optional.", accent=ACCENT_C)

lw = Inches(7.1)
table(s, [
    ["", ""],
    ["State", "Six experiments run and banked; the short-form draft is written "
     "(new_chapters/paperC_lbr/, dated September 2026)"],
    ["Role", "redefined 2026-08-18: absorbs the former LLM-dialogue workstream. A capability "
     "result rather than a benchmark result."],
    ["Author's direction", "the spine is the worked example; the 12-system audit shrinks to one "
     "related-work paragraph plus one control."],
    ["Venue", "opportunistic — no deadline pressure"],
    ["Cost to a submittable draft", "~3–4 focused days of work plus 2–3 machine-nights, "
     "plus an author pass"],
    ["Also serves as", "the final thesis chapter (chapterC_v2), which is longer and keeps "
     "the audit"],
], MARGIN_L, y + Inches(0.05), lw, col_w=[24, 76], font_size=11.2, row_h=0.52,
    first_col_bold=True, header_rows=0)

rx = MARGIN_L + lw + Inches(0.40)
rw = CONTENT_W - lw - Inches(0.40)
textbox(s, rx, y + Inches(0.05), rw, Inches(0.3),
        "ESTABLISHED", size=11, bold=True, color=ACCENT_C)
bullets(s, rx, y + Inches(0.42), rw, [
    "The policy emits off-catalogue directions where the question landscape was measured.",
    "A trained renderer says them in English, and 94.7 % are well-formed questions.",
    "The text — unlike text from a scrambled latent — reproduces the direction's causal effect.",
    "The rendered word lands in the direction's own neighbourhood where the nearest "
    "catalogue question's text points the other way.",
], size=11.5, marker_color=ACCENT)

textbox(s, rx, Inches(4.15), rw, Inches(2.5),
        ["NOT CLAIMED",
         ("· that the policy emits sensibly at ARBITRARY interview states — C1's withheld-subtree "
          "test says it does not, and a state-blind constant beats it there;",
          {"size": 10.5, "bold": False, "color": MUTED}),
         ("· that the blend's five ingredients are the right five — they are unstable draws from a "
          "near-flat gain profile;", {"size": 10.5, "bold": False, "color": MUTED}),
         ("· that asking a synthesised question improves anyone's recommendations. No version of "
          "that sentence appears anywhere.", {"size": 10.5, "bold": True, "color": ACCENT_C})],
        size=11, bold=True, color=ACCENT_C, space_after=5, line_spacing=1.15)
footer(s, "Sources: docs/PUBLICATION_PLAN.md roster · "
          "docs/design/PAPER_C_PUBLICATION_ASSESSMENT_2026-08-25.md · author steer 2026-08-30")

# =====================================================================
# CLOSE
# =====================================================================

# --- Z1: the thesis claim ------------------------------------------------
s, y = slide(prs, "Summary of the three chapters",
             "")

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
        "Method used throughout: pre-register the reading rule, then run the rule against its "
        "own null before reading the effect.",
        size=13, bold=True, color=ACCENT, line_spacing=1.1)
footer(s, "Sources: as per each paper's section")


# --- Z2: publication plan as options ------------------------------------
s, y = slide(prs, "Publication options",
             "Roster reset 2026-08-18. One deadline binds; nothing else does.")

table(s, [
    ["", "state", "OPTION 1 (preferred)", "OPTION 2", "OPTION 3"],
    ["A  instrument", "closed, 41 pp, unsubmitted",
     "TORS — rolling, no deadline, no travel; length is not a penalty there; the one venue "
     "every draft in the blind panel cleared",
     "RecSys or ECIR reproducibility track — our unique win in the panel",
     "hold until the human round-trip study exists, then submit stronger"],
    ["B  interview", "closed 2026-08-30; test look spent and replicates",
     "ECIR 2027 full paper, deadline 2 Oct 2026 — sections 1–3 cut to ~12 pp "
     "(audit · diagnosis · one-variable demonstration)",
     "RecSys, or its reproducibility track",
     "add the RecVAE re-scoring of the frozen tree first, to kill the "
     "one-consumer objection"],
    ["B′  unprompted requests", "held back, not cut",
     "a second submission — RecSys or CIKM — on the request corpus, the opener-fitted trees "
     "and the distilled policy", "fold into the thesis chapter only", "—"],
    ["C  off-catalogue", "six experiments banked; short-form draft written; no deadline",
     "RecSys LBR — the short form the draft is already written as",
     "IntRS or KaRS workshop", "arXiv plus the thesis chapter, and stop there"],
    ["D  free-recall", "absorbed into B — the request corpus and opener census cover its "
     "substance; its own numbers were on the retired pre-July ruler", "—", "—", "—"],
    ["E  LLM dialogue", "merged into C", "—", "—", "—"],
], MARGIN_L, y + Inches(0.02), CONTENT_W, col_w=[11, 19, 28, 22, 20],
    font_size=9.6, row_h=0.72, first_col_bold=True)

textbox(s, MARGIN_L, Inches(6.15), CONTENT_W, Inches(1.0),
        ["The only hard date on this slide is 2 October 2026, and it belongs to Chapter 2.",
         ("Standing constraints: solo, roughly a year left, NO TRAVEL — which is most of why TORS "
          "and short/workshop tracks dominate the options. Bibliography completeness (~8 "
          "load-bearing works still missing) blocks any submission.",
          {"size": 11.5, "bold": False, "color": MUTED})],
        size=14, bold=True, color=INK, space_after=4, line_spacing=1.14)
footer(s, "Sources: docs/PUBLICATION_PLAN.md (roster reset 2026-08-18, TORS rationale, blind panel) · "
          "docs/design/PAPER_B_MASTER_TODO.md · PAPER_C_PUBLICATION_ASSESSMENT_2026-08-25.md")


# --- Z3: what's next ------------------------------------------------------
s, y = slide(prs, "Next steps",
             "Ordered by what unblocks the most.")

table(s, [
    ["", "the work", "why", "cost"],
    ["1", "Score — not refit — Chapter 2's frozen NDCG tree through RecVAE",
     "answers the strongest objection to the demonstration, that it rests on one consumer, "
     "which the chapter's own inversion result makes fair", "one machine-night"],
    ["2", "The ECIR cut: sections 1–3 of Chapter 2 to ~12 pp",
     "the only hard deadline on the roster — 2 October 2026", "the binding task"],
    ["3", "Close the bibliography (~8 load-bearing works)",
     "blocks every submission, A included", "days"],
    ["4", "Chapter 3 short form: shrink the audit, lead with the example interview",
     "the draft exists; the assessment costs it at 3–4 days plus 2–3 machine-nights",
     "cheap, no deadline"],
    ["5", "Fix the example interview's templating failure (2 of 6 malformed on blends)",
     "the template family was fitted to single phrases and degrades on mixtures", "small"],
    ["6", "The human round-trip study (gate C3)",
     "never run; owed project-wide; it is the one thing blocking the strongest claims across "
     "A, B and C — and all three top objections in the blind panel were simulator circularity. "
     "C argues for a study on ANSWERABILITY rather than on quality.",
     "the open question for this meeting"],
], MARGIN_L, y + Inches(0.05), CONTENT_W, col_w=[4, 33, 47, 16], font_size=10.8, row_h=0.68,
    first_col_bold=True)

textbox(s, MARGIN_L, Inches(5.55), CONTENT_W, Inches(1.4),
        ["Two decisions to take.",
         ("(1) Is the unprompted-request work a second submission or a chapter section?   "
          "(2) Is the human round-trip study worth the time it costs, and if so, is "
          "answerability the right thing to test — or does it need to be quality?",
          {"size": 13, "bold": False, "color": MUTED})],
        size=15, bold=True, color=INK, space_after=5, line_spacing=1.16)
footer(s, "Sources: docs/design/PAPER_B_MASTER_TODO.md · docs/PUBLICATION_PLAN.md :96-99 · "
          "GATE_BATTERY_INSTRUMENT.md :106 · docs/HARSH_REVIEW_POINTS.md")

# =====================================================================
# APPENDIX
# =====================================================================

sd = divider("Appendix",
             "Per-experiment tables, methods detail, and the stated bounds. Not presented by "
             "default; here so that a question about a specific number can be answered.",
             MUTED, "13 slides  ·  not part of the ~50 minutes")


# --- APP 1: A gate battery, the open items ---------------------------------
s, y = slide(prs, "Appendix 1.1 — Chapter 1: gates that are open, dropped, or unmeasurable",
             "Every per-build gate passes. These are the qualifications.", accent=MUTED)
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


# --- APP 2: A concept-channel ledger ---------------------------------------
s, y = slide(prs, "Appendix 1.2 — Chapter 1: concept channel, fold operator, and three "
                  "ledgers that must not be mixed",
             "", accent=MUTED)
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
    ["THREE LEDGERS THAT MUST NOT BE MIXED",
     "(1) CONCEPT_CHANNEL_RESULT.md's strategy×channel suite is on the SUPERSEDED i25 fold and "
     "its 'items win the long interview' conclusion was RETRACTED. "
     "(2) GREEDY_BEST_STATIC_I26C.md is policy A (intercept .1314/.0234). "
     "(3) MASK_POLICY_B_MIGRATION.md is policy B (intercept .15855/.02819) — the one this deck uses."],
], MARGIN_L, y + Inches(0.02), CONTENT_W, col_w=[38, 62], font_size=9.8, row_h=0.66)
footer(s, "Sources: chapterA_v2.tex :305-308, :941-947, :1094-1116, :1473-1499 · "
          "ANSWER_DECOMPOSITION_RESULT.md :85-90 · MASK_POLICY_B_MIGRATION.md :108-127")


# --- APP 3: B branching + the inversion -----------------------------------
s, y = slide(prs, "Appendix 2.1 — Chapter 2: branching of the adaptive selectors, and the "
                  "consumer inversion",
             "", accent=MUTED)
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
        "Thompson is fully personalised from the first question and still loses. So the audit's "
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
        "THE FIFTH CONSUMER — WHERE THE VERDICT INVERTS", size=11, bold=True, color=ACCENT_C)
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
         ("This is the methodological result that Chapter 1 is built to address.",
          {"size": 11, "bold": True, "color": ACCENT_C})],
        size=11, color=MUTED, space_after=6, line_spacing=1.15)
textbox(s, rx, Inches(4.85), rw, Inches(1.8),
        ["A PUBLISHED EFFECT, REBUILT",
         ("A published adaptive-degradation effect: switching their candidate rule off makes the "
          "decline STEEPER (−46 % vs −14 %), and under an arm-independent held-out split the "
          "adaptive arm rises monotonically .1637 → .1761 → .1857 → .1936. Their popularity "
          "control declines −.012 against the adaptive arm's −.045 at a matched answer rate.",
          {"size": 10.5, "bold": False, "color": MUTED})],
        size=11, bold=True, color=ACCENT_B, space_after=5, line_spacing=1.15)
footer(s, "Sources: chapterB_v2.tex tab:prefixes (:810-826), :834-930, :932-945, :517-539")


# --- APP 4: B group-conditioned greedy --------------------------------------
s, y = slide(prs, "Appendix 2.2 — Chapter 2: the group-conditioned greedy arm, and what "
                  "validation overstated",
             "", accent=MUTED)
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
        ["Where the pooled gain comes from.",
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


# --- APP 5: B routing null + Reddit groups ----------------------------------
s, y = slide(prs, "Appendix 2.3 — Chapter 2: leaf specialisation is a null, and the six "
                  "request groups",
             "", accent=MUTED)
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
         ("Raw-latent k-means gives release-year buckets — 14 of 20 clusters span under 2.5 "
          "years. Concept-decoding halves that to 6 of 12.",
          {"size": 10.5, "bold": True, "color": ACCENT_C})],
        size=11.5, bold=True, color=INK, space_after=5, line_spacing=1.14)
footer(s, "Sources: chapterB_v2.tex :1434-1467, tab:clusters (:1692-1716), :1627-1677")

# --- APP 6: B opener synthesis + policy per-group ---------------------------
s, y = slide(prs, "Appendix 2.4 — Chapter 2: opener synthesis, and the distilled policy "
                  "by group",
             "", accent=MUTED)
lw = Inches(6.4)
textbox(s, MARGIN_L, y + Inches(0.02), lw, Inches(0.3),
        "SYNTHESISED OPENERS AGAINST REAL ONES", size=11, bold=True, color=ACCENT_B)
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
        ["THREE READINGS OF THIS TABLE",
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


# --- APP 7: C0 geometry detail ----------------------------------------------
s, y = slide(prs, "Appendix 3.1 — Chapter 3: C0 geometry in full",
             "", accent=MUTED)
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
        ["THE REGISTERED READING RULE HAD NO POWER, AND IS REPORTED AS SUCH",
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


# --- APP 8: C1 detail --------------------------------------------------------
s, y = slide(prs, "Appendix 3.2 — Chapter 3: C1 in full, and the state-blind floor",
             "", accent=MUTED)
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

# --- APP 9: C4B / C4C -------------------------------------------------------
s, y = slide(prs, "Appendix 3.3 — Chapter 3: the cost of the round trip, and the effect "
                  "of decoding",
             "", accent=MUTED)

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
        "PREDICTION CONFIRMED, MECHANISM REFUTED",
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


# --- APP 10: methods -----------------------------------------------------------
s, y = slide(prs, "Appendix M1 — Methods: the answer model, circularity, and the "
                  "statistical conventions",
             "", accent=MUTED)
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
    ["The firewall (Chapter 1's I2)", "answers are recorded behaviour and fold-in only. This is the "
     "project's defence against the field-wide circularity attack, and the blind panel says we "
     "under-sell it."],
    ["Where circularity is declared", "exactly one place: Chapter 3's translation-cost measurement, "
     "which needs an answer model by construction. Both arms run under the SAME simulator, only "
     "the DIFFERENCE is read, absolute rows are printed and never headlined, and the declaration "
     "is in the paper's own voice."],
    ["Chapter 3's contract", "the simulated answerer is used to put the policy at interview "
     "states and for nothing else. No other quantity in Chapter 3 comes from simulating an answer and "
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


# --- APP 11: honest bounds ------------------------------------------------------
s, y = slide(prs, "Appendix M2 — The stated bounds, in one place",
             "The consolidated limitations list. Every item was found by an internal control.", accent=MUTED)
table(s, [
    ["", "the bound", "what it costs the claim"],
    ["P1 / P1b  (Chapter 2, validation only)",
     "The pre-registered falsifier FIRED. At 38,336 states whose true label was a held-out "
     "question the retrained policy emits it 0.0000 at top-1 and top-5; a fixed popularity "
     "ranking beats it at 1.45 % top-5. On the undamaged policy, off-manifold, E1 ranks the "
     "measured-best question 208th–374th of 1,000.",
     "'the policy emits questions it was never taught' is NOT available and is reported failed. "
     "Neither test isolates the vocabulary axis: P1b's cells are at unseen STATES, and P1 damages "
     "the policy to create the test."],
    ["C1 emission scope  (Chapter 3)",
     "At withheld states cos to the own target is −0.11948 and a state-blind constant beats the "
     "policy by +0.8893, winning 14/15. The registered generalisation statistic is matched by the "
     "UNTRAINED warm start.",
     "the licensed sentence is 'the policy emits where the question landscape was measured'. No "
     "number is quoted from a held-out-state emission."],
    ["One consumer  (Chapter 2 demonstration)",
     "The tree result rests on one consumer, and our own flip section shows consumer-dependence.",
     "the strongest open objection. Fixable by SCORING the frozen tree through RecVAE."],
    ["The synthesised openers are group-stamped",
     "routing is partly circular and the scramble control cannot detect it; opener−cold survives "
     "because the STATE is held fixed.",
     "the claim is adaptivity beyond the closed tree, not routing."],
    ["The answer model is inferred, not observed",
     "answerability is a structural rule, and it is load-bearing for the entire "
     "answerability result. "
     "The written-request corpus bounds open-answer richness from above.",
     "gate C3, the human round-trip, has never been run — it is what would close this."],
    ["Reproducibility caveat  (Chapter 2)",
     "the GLM-UCB and Thompson cells are not seed-stable (iALS basis re-drawn without a "
     "random_state); drift up to ±0.0026, comparable to the CI half-widths.",
     "quote those two rows to three decimals, not four."],
    ["Single seed  (Chapter 3 adapters)",
     "one training seed per adapter arm.",
     "the word 'robust' is not used anywhere in this work for a single-seed result."],
], MARGIN_L, y + Inches(0.02), CONTENT_W, col_w=[16, 47, 37], font_size=9.2, row_h=0.66)
footer(s, "Sources: P1_HELDOUT_ACTIONS.md · P1B_NEVER_TAUGHT.md · C1_FINAL.md · "
          "PAPER_B_MASTER_TODO.md · ACT1_PAIRED_CI.md · C3B_ADAPTER.md · docs/HARSH_REVIEW_POINTS.md")


# --- APP 12: what was NOT built / not run -----------------------------------
s, y = slide(prs, "Appendix M3 — What was not built, and what is retracted",
             "The ledger of absences and of withdrawn claims.", accent=MUTED)
lw = Inches(6.5)
textbox(s, MARGIN_L, y + Inches(0.02), lw, Inches(0.3),
        "SELECTORS NOT BUILT  (Chapter 2 prints this table itself)", size=11, bold=True, color=ACCENT_B)
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
    ("The 15-minute routing probe", "— required before the unprompted-request section "
     "is submitted on its own."),
    ("Chapter 2's demonstration through a second consumer", "— the recommended next action."),
], size=11.5)

rx = MARGIN_L + lw + Inches(0.45)
rw = CONTENT_W - lw - Inches(0.45)
textbox(s, rx, y + Inches(0.02), rw, Inches(0.3),
        "RETRACTED CLAIMS", size=11, bold=True, color=ACCENT_C)
table(s, [
    ["retracted claim", "what killed it"],
    ["'items win the long interview' (Chapter 1)",
     "it ran on the superseded i25 fold; retracted at the G0 audit"],
    ["the tree takes the tail at K=8, +0.0068 (Chapter 2)",
     "the COMPARATOR was selected under a narrower mask. Re-differenced: −0.0016, does not "
     "separate. The tree's own numbers never moved."],
    ["+0.02092 tail advantage (Chapter 2 demonstration)",
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


# --- APP 13: one-page number index ------------------------------------------
s, y = slide(prs, "Appendix M4 — The numbers, on one page",
             "For lookup during questions.", accent=MUTED)
lw = Inches(6.5)
table(s, [
    ["CHAPTER 1", ""],
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
    ["CHAPTER 2", ""],
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
    ["CHAPTER 3", ""],
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
    ["C4D example interview", "6 turns · 4 of 6 well-formed · NDCG column unquotable"],
], rx, y + Inches(0.02), rw, col_w=[46, 54], font_size=9.6, row_h=0.265,
    align=num(2), first_col_bold=True)
footer(s, "★ = paired interval excludes zero.  Full provenance in each slide's speaker notes.")

# >>>SLIDES<<<

for w in D.relayout(prs):
    print("  ! " + w)

prs.save(OUT)
print("wrote %s  (%d slides)" % (OUT, len(prs.slides._sldIdLst)))
