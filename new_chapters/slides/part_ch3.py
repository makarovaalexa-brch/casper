"""Chapter 3 (generated questions) slides for the supervisor deck.

Sources: build_deck.py :1406-1979 and :2410-2595 (themselves built from
new_chapters/paperC_lbr/paperC_lbr.tex, chapterC_v2.tex, docs/results/C*.md).
Style: STYLE.md.  Every number is copied verbatim from the source.
"""
from pptx.util import Inches, Pt
from pptx.enum.text import PP_ALIGN

from deck_lib2 import (slide, section_slide, notes, textbox, bullets, caption,
                       table, obox, wbox, container, bubble, arrow, icon, person,
                       database, MARGIN_L, CONTENT_W, GREY)

I = Inches
L, C, R = PP_ALIGN.LEFT, PP_ALIGN.CENTER, PP_ALIGN.RIGHT


def _num(n):
    return [L] + [R] * (n - 1)


# ---------------------------------------------------------------- main -----
def build_ch3(prs):
    section_slide(prs, "Chapter 3: generated questions",
                  "questions the catalogue does not contain")

    # 1 research question ---------------------------------------------------
    s, y = slide(prs, "Chapter 3: research question", [
        "Every elicitation system picks a question from a list",
        "Best question may not be on the list",
        "Policy emits a direction in latent space, not a catalogue index",
        "Direction rendered into an English question, then audited",
        "Capability claim only, no recommendation-value claim",
    ], body_box=(0.34, 1.26, 9.32, 2.1))
    table(s, [
        ["prior work", "what it does / lacks"],
        ["GeneRec (Wang 2023)", "generates off-catalogue items, not questions"],
        ["ELM (Tennenholtz 2023)", "verbalises a synthesised latent; mechanism borrowed"],
        ["Vendrov (AAAI 2020)", "continuous query, then snapped to real items"],
        ["EAGLE (Tennenholtz 2024)", "steers an LM over ML-25M embedding; no decoder"],
    ], MARGIN_L, I(3.45), I(9.32), col_w=[28, 72], font_size=11, row_h=0.28,
        first_col_bold=True)
    notes(s, "Chapter 3 is the capstone: every elicitation system in the literature selects from a bank.\n"
             "Question: what happens when the best thing to ask is not in the bank.\n"
             "The policy emits a vector; a renderer turns it into English; an independent bridge audits it.\n"
             "Prior work covers each leg separately: off-catalogue items, verbalised latents, snapped queries, decoder-free steering.\n"
             "No claim that asking a synthesised question improves recommendations; Chapter 2 showed enlarging the bank hurt.")

    # 2 emission loop flow chart -------------------------------------------
    s, y = slide(prs, "Emission loop")
    _emission_chart(s)
    caption(s, MARGIN_L, I(4.98), I(9.3),
            "Figure 3.1. Emitted route (Chapter 3) beside the catalogue route (Chapter 2).")
    notes(s, "Top lane: the interview state goes to the policy, which emits a vector rather than an index.\n"
             "An adapter maps the vector into eight soft prefix tokens of a frozen distilgpt2, which writes the question.\n"
             "Bottom lane: the Chapter 2 tree picks an entry from the 19,390-question bank.\n"
             "Both questions go to the user; answers fold into the frozen tower and are scored on the same ruler.\n"
             "Chapter 3 measures whether the text carries the direction, not whether it improves NDCG.")

    # 3 experiments map ----------------------------------------------------
    s, y = slide(prs, "Experiments")
    table(s, [
        ["", "question", "headline", "verdict"],
        ["C0 geometry", "does the emitted vector sit on the question it snaps to?",
         "max cos 0.1203 over 160,000 states; mean-removed cos +0.488 vs +0.170 random",
         "on-grid prediction falsified; registered rule had no power"],
        ["C1 emission", "can one object run the tree's interview and emit off-catalogue blends?",
         "245/245 distinct questions; snap agreement 0.9950; fit cos 0.96958 at trained states",
         "both blocking gates pass; generalisation to withheld states fails"],
        ["C3B mechanism", "does the vector condition the text?",
         "A − A′ = +0.1114 [+0.1054, +0.1173]; 107 % of gold-phrase ceiling; identification 0.6778",
         "two mechanism criteria pass; equivalence control fails; composite kept as failure"],
        ["C3B-v2 askability", "does it survive reformatting into real questions?",
         "94.7 % [0.9398, 0.9542] well-formed; A − A′ +0.05169 items, +0.28625 concepts",
         "all readings pass, both decodings, both channels"],
        ["C4 loop", "does the rendered word land in the direction's neighbourhood?",
         "median rank 91 of 1,031 (distinctive renders); catalogue text −0.158 vs rendered word +0.204",
         "point-specificity +0.28593 [+0.17820, +0.39777]; prior floor at zero"],
        ["C4B / C4C cost", "what does the round trip through English cost?",
         "+0.01802 full / +0.00672 tail; decoding takes distinct renders 23 → 41, cuts cost 2–3.5×",
         "cost is mostly harm from confidently-wrong renders"],
    ], MARGIN_L, I(1.28), I(9.32), col_w=[13, 25, 34, 28], font_size=9, row_h=0.5,
        first_col_bold=True)
    notes(s, "Six experiments, each pre-registered and gated, each read against its own null.\n"
             "C0 and C1 are about the vector: where it sits and whether one object can emit blends without breaking the interview.\n"
             "C3B and C3B-v2 are about the renderer: does the vector condition the text, and does that survive question form.\n"
             "C4 closes the loop: the rendered word is read back and compared with the nearest catalogue question.\n"
             "C4B and C4C price the round trip and show decoding is the lever.\n"
             "Three of the six verdicts were decided by a control rather than by the effect.")

    # 4 C0 geometry ----------------------------------------------------------
    s, y = slide(prs, "Emitted vector: geometry")
    table(s, [
        ["all 160,000 validation states", "emitted", "random"],
        ["median cosine to the question it snaps to", "−0.18201", "+0.06418"],
        ["maximum cosine over all rows", "0.1203", "0.398"],
        ["states on their snapped question (cos > 0.98)", "0 of 160,000", "0"],
        ["mean-removed cosine to the snapped entry", "+0.488", "+0.170"],
    ], MARGIN_L, I(1.3), I(9.32), col_w=[58, 21, 21], font_size=11, row_h=0.3,
        align=_num(3), first_col_bold=True)
    bullets(s, MARGIN_L, I(3.25), I(9.32), [
        "No emission sits on the question it snaps to",
        "Whole score vector far below zero: median top score −82.7",
        "Cross-entropy constrains only score differences, so a large common mode is free",
        "Emission = shared offset + genuine bank-aligned direction (visible mean-removed)",
        "Registered confirmation / falsification rule had no power; reported as such",
    ], size=16)
    notes(s, "C0 asks whether the deployed policy's vector is a point on the grid or something else.\n"
             "Maximum cosine to the snapped question is 0.12 over all 160,000 states, so never on-grid.\n"
             "The raw vector is dominated by a common-mode offset because cross-entropy only constrains differences.\n"
             "Mean-removed, the emission points at the entry it selects: +0.488 against +0.170 for random.\n"
             "The registered reading rule turned out to have no power; the retraction is in the appendix.")

    # 5 C1 emission gates --------------------------------------------------
    s, y = slide(prs, "Emission gates")
    table(s, [
        ["gate / quantity", "value", "bar"],
        ["distinct questions preserved", "245 / 245", "≥ 0.75×"],
        ["snap agreement with the banked policy", "0.9950", "≥ 0.90"],
        ["emission / target norm ratio, trained states", "1.0449", "[0.5, 2.0]"],
        ["cos(ẑ, own target), trained states", "0.96958", "—"],
        ["cos(ẑ, own target), held-out states", "−0.11948", "—"],
        ["state-blind constant vs policy, held-out", "+0.8893 [+0.6555, +1.0905], 14/15", "—"],
        ["median top-1 blend weight", "0.2254", "0.121 random"],
    ], MARGIN_L, I(1.3), I(9.32), col_w=[50, 32, 18], font_size=11, row_h=0.28,
        align=_num(3), first_col_bold=True)
    bullets(s, MARGIN_L, I(3.65), I(9.32), [
        "One object runs the tree's interview unchanged and emits gain-weighted blends",
        "72 of 72 gain sweeps match the banked splitter; 1,279,335 labels reproduced",
        "Both blocking gates pass",
        "Withheld subtree: a state-blind constant beats the policy",
        "Scope: the policy emits where the question landscape was measured",
    ], size=16)
    notes(s, "C1 trains the emitting policy on the Chapter 2 cold tree with blend targets at 72 nodes.\n"
             "The interview contract holds: 245 of 245 distinct questions preserved, snap agreement 0.995.\n"
             "At trained states the emission matches its target at cosine 0.97.\n"
             "At withheld states it does not: a state-blind constant wins 14 of 15, by +0.89.\n"
             "Same shape as Chapter 2's held-out-action result. Licensed sentence: emits where measured.")

    # 6 C3B mechanism --------------------------------------------------------
    s, y = slide(prs, "Vector conditions text")
    table(s, [
        ["reading (item channel, n = 3,670 held-out)", "result", "floor / bar", "cell"],
        ["A − A′ (true adapter minus scrambled twin)", "+0.1114 [+0.1054, +0.1173]",
         "CI excludes 0", "pass"],
        ["A level", "+0.0953 [+0.0903, +0.1003]", "prior −0.0028; A′ −0.0161", "pass"],
        ["A as share of gold-phrase ceiling (+0.0892)", "107 %", "bar 40 %", "pass"],
        ["identification vs same-channel decoy", "0.6778 [0.6631, 0.6931]",
         "chance 0.500; bar 0.5974", "pass"],
        ["equivalence: A′ within noise of prior", "A′ − prior = −0.0133 [−0.0185, −0.0081]",
         "intervals overlap", "fail"],
        ["composite registered label", "", "", "failure, kept"],
    ], MARGIN_L, I(1.3), I(9.32), col_w=[38, 28, 22, 12], font_size=10, row_h=0.3,
        align=[L, R, R, C], first_col_bold=True)
    bullets(s, MARGIN_L, I(3.6), I(9.32), [
        "Two-layer adapter → 8 soft prefix tokens of a frozen distilgpt2",
        "Control twin trained identically on correspondence-scrambled pairs",
        "Twin collapsed to 61 distinct texts over 3,880 vectors; true arm 2,280",
        "Equivalence control fired because the twin sits below knowing nothing",
        "Concept channel, not gated: A = +0.3533 vs required +0.2002",
    ], size=16)
    notes(s, "C3B asks whether the vector conditions the text or the renderer ignores it.\n"
             "The true adapter beats its scrambled twin by +0.11, and re-encodes at 107 percent of the gold phrase's own ceiling.\n"
             "Identification of the source vector against a decoy is 0.68 against chance 0.5.\n"
             "The equivalence control failed in the unexpected direction: the twin sits below a prior.\n"
             "Pre-registration is not reinterpreted after the data, so the composite stays failure; the two mechanism criteria pass on their own terms.\n"
             "Caveat on the 107 percent: the bridge prefers concept-flavoured text, and invented titles are freer to be that.")

    # 7 C3B-v2 question-shaped output -------------------------------------
    s, y = slide(prs, "Question-shaped output")
    table(s, [
        ["reading (item channel, n = 3,670)", "best-of-8 (primary)", "greedy", "bar"],
        ["A − A′, tower cosine", "+0.05169 [+0.04595, +0.05750]",
         "+0.05855 [+0.05321, +0.06383]", "CI excl. 0"],
        ["A level (share of re-measured ceiling)", "0.04869 (71 %)", "0.07177 (105 %)", "≥ 0.02731"],
        ["identification", "0.6099", "0.6392", "≥ 0.5790"],
        ["question-form rate", "0.9471 [0.9398, 0.9542]", "—", "≥ 0.90"],
        ["concept channel, not gated: A − A′", "+0.28625 [+0.24014, +0.33307]", "—", "CI excl. 0"],
        ["verdict", "success", "success", ""],
    ], MARGIN_L, I(1.3), I(9.32), col_w=[36, 27, 25, 12], font_size=10, row_h=0.3,
        align=[L, R, R, C], first_col_bold=True)
    bullets(s, MARGIN_L, I(3.6), I(9.32), [
        "Same adapter retrained on question-shaped targets",
        "Boilerplate lowers the instrument ceiling: items 0.08920 → 0.06828 (−23 %), concepts −17 %",
        "Rule adopted: a format change is an instrument change; re-measure the ceiling first",
        "5.3 % malformed, printed not trimmed; failures degenerate by repetition",
        "Question mark is learned: all-zero prefix emits it at 0.00, channel prior at 1.00",
    ], size=16)
    notes(s, "C3B-v2 retrains the same adapter to write actual questions rather than bare phrases.\n"
             "The mechanism survives: 94.7 percent well-formed, A minus A-prime still positive on both channels.\n"
             "Wrapping a phrase in boilerplate lowers what a perfect renderer can score through the same instrument by 23 percent.\n"
             "Without re-measuring the ceiling, 107 percent and 71 percent would read as a regression; they are against different ceilings.\n"
             "No figure of this kind appears in the verbalisation literature.")

    # 8 rendered word rank -------------------------------------------------
    s, y = slide(prs, "Rendered word rank")
    table(s, [
        ["greedy decoding, 42 states", "n", "median rank"],
        ["render is a distinctive word", "18 / 42", "91"],
        ["render is a repeated word", "24 / 42", "362.5"],
        ["random-tag floor", "—", "340"],
        ["pooled over all 42 states", "42", "313.0"],
    ], MARGIN_L, I(1.3), I(4.4), col_w=[56, 18, 26], font_size=10.5, row_h=0.3,
        align=_num(3), first_col_bold=True)
    table(s, [
        ["rendered word", "rank", "what the point was built from"],
        ["action", "1", "three questions in; Apollo 13 disliked"],
        ["art", "4", "an art-cinema blend"],
        ["indictment", "5", "courtroom blend: Primal Fear, A Time to Kill"],
        ["animals", "7", "the Pixar branch"],
        ["military", "21", ""],
        ["adventure", "28", "a Harry Potter blend"],
        ["science", "31", ""],
        ["noir", "54", ""],
    ], I(5.0), I(1.3), I(4.66), col_w=[26, 12, 62], font_size=10, row_h=0.26,
        align=[L, C, L], first_col_bold=True)
    bullets(s, MARGIN_L, I(3.85), I(9.32), [
        "Limit is generation diversity, not geometry",
        "Greedy fallbacks: horror ×9, religion ×7, movies ×4",
        "Best-of-8 by LM likelihood: distinct renders 23 → 37 of 42, cost roughly halved",
        "Faithfulness change under best-of-8 not significant: −0.0193 [−0.097, +0.058]",
    ], size=16)
    notes(s, "Second, independent view: is the rendered word one of the point's own nearest concepts.\n"
             "Where greedy produces a distinctive word, median rank is 91 of 1,031; the random floor is 340.\n"
             "Where greedy falls back on stock words, rank is at the floor.\n"
             "Best-of-8 by the frozen LM's own likelihood, a rule blind to the metric, removes most collapses.\n"
             "That is the decoding of record for everything measured after it.")

    # 9 catalogue question vs emitted direction -----------------------------
    s, y = slide(prs, "Catalogue question vs emitted direction")
    _gap_chart(s)
    bullets(s, MARGIN_L, I(3.55), I(9.32), [
        "Gap +0.362, interval excluding zero",
        "Direction is expressible in English; the nearest catalogue text does not express it",
        "Point-specificity +0.28593 [+0.17820, +0.39777]; prior floor −0.013 [−0.074, +0.053]",
        "Caveats: nearest entry is often the same one (\"divorce\"); bridge favours concept text",
        "Sign survives both caveats; magnitude not quoted without them",
    ], size=15)
    notes(s, "At each interview state, take the emitted point and its nearest bank entry in the emitter's own head space, cosine 0.93.\n"
             "The bank entry's own gold phrase carries the direction at minus 0.158.\n"
             "The rendered off-catalogue word, not in the bank at all, carries it at plus 0.204.\n"
             "The registered discriminating reading is point-specificity: a render of this point aligns with this point's reference better than a render of another point does.\n"
             "Two caveats attached; the sign survives both.")

    # 10 example interview --------------------------------------------------
    s, y = slide(prs, "Example interview")
    _transcript_chart(s)
    notes(s, "One cold-start trajectory, six turns, majority path through the tree, all 10,000 validation users.\n"
             "Left: the synthesised question; right: what the tree would have asked at the same state.\n"
             "Turns 3, 5 and 6: sci-fi against alien films, detective drama against Crimson Tide, children films against Star Wars VI. Same neighbourhood, more askable.\n"
             "Turns 2 and 4 are the templating failure: boilerplate around a bare phrase.\n"
             "2 of 6 against 5.3 percent measured on bank questions: blends are diffuse, so the bank-question rate overstates.\n"
             "The run's NDCG column is declared-circular, unregistered, n = 6; it is not shown.")

    # 11 rendering failures ---------------------------------------------------
    s, y = slide(prs, "Rendering failures")
    table(s, [
        ["source vector's true question is about", "adapter renders"],
        ["The Shawshank Redemption", "Have you seen Pulp Fiction (1995)?"],
        ["Ghost Story (1981)", "Have you seen Battlestar Galactica (1980)?"],
        ["Desperate Hours (1955)", "What did you think of Bunted, The (1941)?"],
    ], MARGIN_L, I(1.3), I(9.32), col_w=[42, 58], font_size=12, row_h=0.32,
        first_col_bold=True)
    table(s, [
        ["quantity", "value"],
        ["token-F1, generated content words vs gold phrase", "0.055"],
        ["exact content match", "0.0003"],
        ["identification of own source vector vs decoy", "0.6099 (chance 0.500)"],
        ["what the true question itself achieves", "0.6316"],
    ], MARGIN_L, I(2.75), I(5.6), col_w=[68, 32], font_size=10.5, row_h=0.27, align=_num(2))
    bullets(s, I(6.2), I(2.75), I(3.46), [
        "Neighbouring era and register, wrong film",
        "Fluent, askable, incorrect as a surface form",
        "Text reproduces the vector's effect anyway",
        "Machine-only synonyms (Hewitt et al.)",
        "A plausibility panel cannot separate this from a confabulator",
    ], size=13, height=I(2.3))
    notes(s, "The adapter almost never emits the right title, and its text reproduces the vector's effect anyway.\n"
             "Shawshank becomes Pulp Fiction; Ghost Story becomes Battlestar Galactica.\n"
             "Token-F1 0.055, exact match essentially zero; identification 0.61 against chance 0.5, true question 0.63.\n"
             "A rater without the vector marks these wrong; the instrument, which sees the vector, records the effect. Both are correct.\n"
             "A renderer that asked about The Godfather every time would score well with raters and at chance with the instrument.\n"
             "This is why the evidence is a control and a ceiling rather than a human score.")

    # 12 status ------------------------------------------------------------------
    s, y = slide(prs, "Chapter 3: status", [
        "Six experiments run and banked; short-form draft written (September 2026)",
        "Established",
        ("policy emits off-catalogue directions where the landscape was measured", 1),
        ("renderer writes them as English questions, 94.7 % well-formed", 1),
        ("text reproduces the direction's effect; scrambled-latent text does not", 1),
        ("rendered word lands in the direction's neighbourhood; catalogue text points away", 1),
        "Not claimed",
        ("emission at arbitrary states; blend ingredients; recommendation value", 1),
        "Venue opportunistic; also the final thesis chapter",
    ], size=16)
    notes(s, "Chapter 3 is a capability result with no benchmark to beat.\n"
             "Established: emission where measured, English rendering, causal faithfulness, loop closure.\n"
             "Not claimed: generalisation to arbitrary states, stability of the five ingredients, any recommendation value.\n"
             "Cost to a submittable draft: about three to four focused days plus two to three machine-nights.\n"
             "NOTE TO AUTHOR: publication is optional; the thesis chapter version keeps the 12-system audit the short draft drops.")


# ------------------------------------------------------------- charts ------
def _emission_chart(s):
    """Two lanes (emitted vs catalogue route), shared user and scorer."""
    bh = I(0.55)
    # lane A: emitted route
    container(s, I(0.34), I(1.28), I(7.0), I(1.22), "Chapter 3: emitted question", size=11)
    ya = I(1.72)
    b1 = obox(s, I(0.5), ya, I(1.15), bh, "interview state", size=10)
    b2 = obox(s, I(1.95), ya, I(1.35), bh, ["policy", "emits a vector"], size=10, dark=True)
    b3 = obox(s, I(3.6), ya, I(1.45), bh, ["adapter +", "frozen distilgpt2"], size=10)
    bubble(s, I(5.35), I(1.62), I(1.85), I(0.72), "How do you feel about sci-fi movies?",
           size=9.5, tail="right")
    arrow(s, I(1.65), ya + bh // 2, I(1.95), ya + bh // 2)
    arrow(s, I(3.3), ya + bh // 2, I(3.6), ya + bh // 2)
    arrow(s, I(5.05), ya + bh // 2, I(5.35), ya + bh // 2)
    # lane B: catalogue route
    container(s, I(0.34), I(2.62), I(7.0), I(1.22), "Chapter 2: catalogue question", size=11)
    yb = I(3.06)
    obox(s, I(0.5), yb, I(1.15), bh, "interview state", size=10)
    obox(s, I(1.95), yb, I(1.35), bh, ["interview tree", "picks an index"], size=10)
    database(s, I(3.9), yb - I(0.02), I(0.5), I(0.58))
    textbox(s, I(3.55), yb + I(0.56), I(1.25), I(0.3), "question bank", size=9, align=C)
    bubble(s, I(5.35), I(2.96), I(1.85), I(0.72), "Do you like alien films?",
           size=9.5, tail="right")
    arrow(s, I(1.65), yb + bh // 2, I(1.95), yb + bh // 2)
    arrow(s, I(3.3), yb + bh // 2, I(3.9), yb + bh // 2)
    arrow(s, I(4.4), yb + bh // 2, I(5.35), yb + bh // 2)
    # robot icons in lane containers
    icon(s, "robot", I(6.6), I(1.3), I(0.5))
    icon(s, "robot", I(6.6), I(2.64), I(0.5))
    # user, right column
    person(s, I(8.25), I(2.05), I(0.65))
    textbox(s, I(7.95), I(2.75), I(1.25), I(0.3), "user", size=11, align=C)
    arrow(s, I(7.34), I(2.0), I(8.2), I(2.3))
    arrow(s, I(7.34), I(3.3), I(8.2), I(2.6))
    bubble(s, I(7.6), I(3.2), I(1.9), I(0.5), "Yes / No / not seen", size=9.5, tail="up")
    # shared scorer row
    yc = I(4.1)
    c1 = obox(s, I(2.6), yc, I(1.4), bh, ["fold-in operator", "answers → point"], size=10)
    c2 = obox(s, I(4.35), yc, I(1.4), bh, ["frozen tower", "recommender"], size=10)
    c3 = wbox(s, I(6.1), yc, I(1.4), bh, ["ranked list", "NDCG@10"], size=10)
    arrow(s, I(4.0), yc + bh // 2, I(4.35), yc + bh // 2)
    arrow(s, I(5.75), yc + bh // 2, I(6.1), yc + bh // 2)
    arrow(s, I(8.4), I(3.7), I(4.0), yc + I(0.05), label="answers", label_dy=I(-0.05),
          label_dx=I(1.2))
    textbox(s, I(0.4), yc + I(0.05), I(2.1), I(0.5), ["same ruler", "for both routes"],
            size=10, color=GREY)


def _gap_chart(s):
    bh = I(0.62)
    obox(s, I(0.5), I(1.95), I(2.3), bh, ["emitted point", "the direction to ask about"],
         size=10, dark=True)
    wbox(s, I(3.9), I(1.35), I(3.2), I(0.8),
         ["nearest catalogue question (cos 0.93)", "its gold phrase carries the direction at",
          "−0.158 [−0.224, −0.089]"], size=10)
    obox(s, I(3.9), I(2.45), I(3.2), I(0.8),
         ["rendered off-catalogue word", "carries the same direction at",
          "+0.204 [+0.117, +0.291]"], size=10)
    arrow(s, I(2.8), I(2.26), I(3.9), I(1.75))
    arrow(s, I(2.8), I(2.26), I(3.9), I(2.85))
    textbox(s, I(7.4), I(1.7), I(2.2), I(1.2), ["gap", "+0.362", "interval excludes 0"],
            size=14, align=C)
    icon(s, "robot", I(0.55), I(2.7), I(0.55))


def _transcript_chart(s):
    turns = [
        ("1", "Are you in the mood for something action films?",
         "Do you like better than expected films?", False),
        ("2", "Do you like it when a film is suspense?", "Do you like dark humor films?", True),
        ("3", "How do you feel about sci-fi movies?", "Do you like alien films?", False),
        ("4", "Do you like it when a film is movie?",
         "Have you seen Star Wars: Episode V …?", True),
        ("5", "Do you like it when a film is detective drama?",
         "Have you seen Crimson Tide (1995)?", False),
        ("6", "How do you feel about children films?",
         "Have you seen Star Wars: Episode VI …?", False),
    ]
    textbox(s, I(1.55), I(1.22), I(3.9), I(0.25), "synthesised question", size=11, bold=True)
    textbox(s, I(5.75), I(1.22), I(3.9), I(0.25), "the tree's question", size=11, bold=True)
    y0 = I(1.5)
    rh = I(0.56)
    for k, (t, synth, tree, fail) in enumerate(turns):
        y = y0 + rh * k
        textbox(s, I(0.34), y + I(0.12), I(0.45), I(0.3), "turn " + t, size=9, color=GREY)
        icon(s, "robot", I(0.95), y + I(0.04), I(0.42))
        text = [synth, "templating failure"] if fail else synth
        bubble(s, I(1.55), y + I(0.02), I(3.9), I(0.48), text, size=9.5 if not fail else 9,
               tail="left")
        wbox(s, I(5.75), y + I(0.05), I(3.9), I(0.42), tree, size=9.5)
    caption(s, MARGIN_L, I(4.92), I(9.3),
            "Figure 3.2. Six-turn interview, majority path, 10,000 validation users; "
            "v2 question adapter, best-of-8 decoding.")


# ------------------------------------------------------------ appendix -----
def build_appendix_ch3(prs):
    # 3.1 C0 in full ------------------------------------------------------
    s, y = slide(prs, "Appendix 3.1: C0 geometry in full")
    table(s, [
        ["median over 160,000 validation rows", "emitted ẑ", "IQR", "random"],
        ["cos1, cosine to the snapped question", "−0.18201", "[−0.32830, −0.06928]", "+0.06418"],
        ["margin ratio m = (s1−s2)/(s1−s10)", "0.40985", "[0.21470, 0.57909]", "0.26678"],
        ["entropy of softmax(s) over top-50", "0.00549", "[0.00004, 0.41493]", "0.90049"],
        ["states with ≥3 candidates within 5 % of s1", "20.8 %", "—", "32.6 %"],
        ["states with cos1 > 0.98", "0 of 160,000", "—", "0"],
        ["states with cos1 > 0", "4.76 %", "—", "—"],
        ["maximum cos1 over all states", "0.1203", "—", "0.398"],
        ["cos1, weighted by distinct state", "−0.260", "—", "—"],
        ["margin ratio m, state-weighted", "0.286", "—", "0.267 (floor)"],
    ], MARGIN_L, I(1.28), I(9.32), col_w=[46, 18, 22, 14], font_size=9.5, row_h=0.25,
        align=_num(4), first_col_bold=True)
    bullets(s, MARGIN_L, I(3.95), I(9.32), [
        "Gates: snap identity 0 mismatches of 160,000; banked rollout byte-identical on 10,000 × 16",
        "Canonical intercept exact: full 0.15686538, tail 0.02747733",
        "Median top score −82.7; 95.2 % of states negative; mean-removed cos to selected entry +0.488",
        "160,000 rows = 31,874 states in 6,836 interviews; 99.95 % of cos1 variance between states",
        "Registered reading rule had no power: random direction lands in the same cell; sentence retracted",
    ], size=11)
    notes(s, "Reference slide for C0.\n"
             "The confirmation cell was a conjunction of two strict thresholds and the falsification cell a disjunction of two lax ones; a random direction lands in 'between candidates' too.\n"
             "What stands unconditionally: maximum cosine 0.1203, so no emission sits on its snapped question.\n"
             "Cluster bootstrap with cluster = user.")

    # 3.2 C1 in full ------------------------------------------------------------
    s, y = slide(prs, "Appendix 3.2: C1 in full")
    table(s, [
        ["", "primary (root.like.* withheld; 57 / 15)", "secondary (depth ≥5 withheld; 42 / 30)"],
        ["distinct questions preserved", "245 / 245 (1.000)", "244 / 245 (0.996)"],
        ["snap agreement", "0.9950", "0.9960"],
        ["norm ratio all / trained / held-out", "1.0629 / 1.0449 / 3.4798", "1.1056 / 1.0541 / 3.4213"],
        ["cos(ẑ, own target), trained", "0.96958 (min 0.932; 57/57 ≥ 0.90)", "0.97455 (42/42)"],
        ["cos(ẑ, own target), held-out", "−0.11948", "−0.01069"],
        ["(a) own − random other held-out target", "+0.05926 [+0.04269, +0.07539]",
         "+0.10114 [+0.08436, +0.11798]"],
        ["(a′) same statistic, untrained warm start", "+0.08432 [+0.04738, +0.12680], larger",
         "+0.10441 [+0.08950, +0.11953], ties"],
        ["(b) own − warm start's cosine", "+0.25954 [+0.09535, +0.45041]",
         "+0.15696 [+0.06792, +0.25538]"],
    ], MARGIN_L, I(1.28), I(9.32), col_w=[34, 33, 33], font_size=9, row_h=0.24,
        align=[L, R, R], first_col_bold=True)
    table(s, [
        ["state-blind constant floor", "constant", "policy", "paired difference"],
        ["trained states, raw cosine", "0.9531", "0.9696 / 0.9745", "policy +0.015"],
        ["held-out, primary", "0.9502", "0.0609", "constant +0.8893 [+0.6555, +1.0905], 14/15"],
        ["held-out, secondary", "0.9623", "0.0351", "constant +0.9271 [+0.8254, +1.0194], 30/30"],
    ], MARGIN_L, I(3.62), I(9.32), col_w=[30, 14, 18, 38], font_size=9, row_h=0.22,
        align=_num(4), first_col_bold=True)
    bullets(s, MARGIN_L, I(4.55), I(9.32), [
        "Targets form a tight cone: median pairwise cosine 0.920; state-specific share of mean-removed energy 0.5915 [0.5558, 0.6270]",
        "72 of 144 nodes emit; ingredients unstable (top-5 self-overlap 1.905 of 5); direction split-half cosine 0.987 raw, 0.656 mean-removed",
    ], size=10)
    notes(s, "Reference slide for C1 and the 2026-09-01 correction.\n"
             "The registered generalisation statistic looked positive until the untrained warm start produced a larger margin at the same states.\n"
             "A state-blind constant beats the policy at held-out states on both splits.\n"
             "Global linear probes, a profile parameterisation and local kNN transfer all fail to recover the state-specific component.")

    # 3.3 round trip cost and decoding ------------------------------------------
    s, y = slide(prs, "Appendix 3.3: round trip cost")
    table(s, [
        ["decoding (42 states, 36,197 user-rows)", "distinct / 42", "cos(u_text, R_s)",
         "Δ vs greedy", "median rank", "cost L − T, full"],
        ["greedy (banked)", "23", "+0.2039", "—", "313", "+0.01802 [+0.01668, +0.01940]"],
        ["sample T = 0.7", "38", "+0.1034", "−0.1005 [−0.197, −0.004]", "441",
         "+0.00520 [+0.00397, +0.00638]"],
        ["sample T = 1.0", "41", "+0.0942", "−0.1097 [−0.220, −0.002]", "464", "+0.00746"],
        ["best-of-8 by LM likelihood (record)", "37", "+0.1846", "−0.0193 [−0.097, +0.058] n.s.",
         "395", "+0.00884 [+0.00747, +0.01025]"],
    ], MARGIN_L, I(1.28), I(9.32), col_w=[28, 10, 13, 21, 10, 18], font_size=9, row_h=0.36,
        align=_num(6), first_col_bold=True)
    bullets(s, MARGIN_L, I(3.2), I(9.32), [
        "Round trip costs +0.01802 full [+0.01668, +0.01940], +0.00672 tail [+0.00541, +0.00805]",
        "21 of 42 states above zero, 5 below (English beats the direction; best 'theatrical' −0.079 [−0.119, −0.040])",
        "Not shared-content: |L ∩ T| zero at 38 of 42 states",
        "Unique-render states pay +0.00633; repeated-render states +0.02628 (×4.2)",
        "Absolute rows (declared-circular, not headlined): base 0.16571, L 0.17443, T greedy 0.15641, T sampled 0.16924",
        "Greedy stock words fold below the no-fold baseline: harm, not lost meaning",
        "Greedy arm regenerated, not read: 42 renders bit-identical to the banked run",
    ], size=11)
    notes(s, "Reference slide for C4B and C4C.\n"
             "Prediction confirmed: the cost concentrates on mode-collapsed renders. Mechanism refuted: it is harm from confidently wrong concepts, not lost meaning.\n"
             "Best-of-8 repairs greedy's collapses (religion to scifi, 620 to 61) while losing some hits (art 4 to 873).\n"
             "Chosen on the frozen LM's own likelihood, blind to the evaluation metric by construction.")
