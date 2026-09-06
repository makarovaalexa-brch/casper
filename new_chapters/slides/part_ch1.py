"""Chapter 1 (the measurement instrument) — slides in the author's style.

Exposes build_ch1(prs) and build_appendix_ch1(prs). Numbers from build_deck.py
(itself built from chapterA_v2.tex); nothing invented.
"""
from pptx.util import Inches, Pt
from pptx.enum.text import PP_ALIGN
from deck_lib2 import (new_deck, slide, section_slide, notes, table, figure, caption,
                       bullets, textbox, obox, wbox, container, arrow, icon, person,
                       database, MARGIN_L, CONTENT_W, BODY_TOP, GREY)

I = Inches
L, R, C = PP_ALIGN.LEFT, PP_ALIGN.RIGHT, PP_ALIGN.CENTER


def num(n):
    return [L] + [R] * (n - 1)


# ------------------------------------------------------------------ main ----
def build_ch1(prs):
    section_slide(prs, "Chapter 1: the measurement instrument")

    # 1 research question ---------------------------------------------------
    s, y = slide(prs, "Chapter 1: research question", [
        "Elicitation papers each use their own recommender",
        ("strategy comparison confounded with model comparison", 1),
        "Strategy rankings invert when the recommender changes",
        ("Golbandi leaf model as consumer: Thompson beats static order", 1),
        ("+0.0100 / +0.0171 / +0.0148 / +0.0281 full at k = 1, 2, 4, 8", 1),
        ("same selector loses to static order on the certified tower", 1),
        "Response: frozen tower, fold-in operator, fixed metric",
        "Acceptance gates fixed before any result is read",
        "Consequence: a Chapter 2 null is attributable to the policy",
    ])
    notes(s, "Every elicitation paper trains its own recommender, so a strategy comparison is also a model comparison.\n"
             "The inversion is measured: swap the consumer for a Golbandi leaf model and Thompson sampling goes from losing to the static order to beating it at every budget.\n"
             "So the recommender is part of the instrument and has to be certified first.\n"
             "Chapter 1 builds and certifies that instrument; Chapters 2 and 3 use it unchanged.")

    # 2 pipeline flow chart -------------------------------------------------
    s, _ = slide(prs, "Instrument: pipeline")
    _pipeline_chart(s)
    notes(s, "The user answers item questions, concept questions, or types free text.\n"
             "All channels enter one observation slot; the fold-in operator maps the evidence set to one latent point, with no gradient step at inference.\n"
             "The frozen tower scores the whole catalogue from that point; NDCG@10 full and tail is the ruler.\n"
             "The gate battery sits beside the pipeline: it is read before any downstream number.")

    # 3 canonical ruler -----------------------------------------------------
    s, y = slide(prs, "Canonical ruler")
    table(s, [
        ["", "Decision", "Reason"],
        ["Recipe", "Liang strong-generalisation split, applied verbatim to ML-25M",
         "field's canonical protocol; snaps to published numbers"],
        ["Catalogue", "films with ≥ 20 ratings: arena 18,430; train vocabulary 18,359",
         "items and questions share one vocabulary"],
        ["Users", "ratings > 3.5, min 5 per user; train 140,768 / val 10,000 / test 10,000",
         "seed 98765; per-user 80/20 fold-in / target"],
        ["Primary metric", "full and tail NDCG@10 (tail: 190 head items removed)",
         "full alone hides what popularity already gives"],
        ["Scoring", "recommender's own learned decoder bias", "popularity floor craters full NDCG"],
        ["Secondary", "NDCG@100", "reported, never headlined"],
    ], MARGIN_L, y + I(0.05), CONTENT_W, col_w=[15, 47, 38], font_size=11, row_h=0.42,
        first_col_bold=True)
    caption(s, MARGIN_L, I(4.55), CONTENT_W, "Table 1.1. The ruler, fixed July 2026 and unchanged since.")
    notes(s, "One split, one metric, one scoring convention. Every number in all three chapters is on this ruler.\n"
             "Liang's recipe verbatim so published numbers can be matched.\n"
             "Tail NDCG removes the 190 head items that carry a third of the training mass; full alone rewards popularity.\n"
             "Scoring uses the model's own decoder bias, not a popularity floor.")

    # 4 published-number bridge ----------------------------------------------
    s, y = slide(prs, "Published-number bridge", [
        "Each baseline reproduced on its own published protocol first",
        "ML-20M, NDCG@100, Liang protocol",
        "Mult-DAE fails the bridge and is dropped from the bank",
    ], body_box=(0.34, 1.26, 9.32, 1.4))
    table(s, [
        ["System", "ours", "published", "Δ", "verdict"],
        ["EASE (Steck 2019)", "0.4203", "0.420", "+0.0003", "pass"],
        ["RecVAE (Shenbin 2020)", "0.4425", "0.442", "+0.0005", "pass"],
        ["Mult-VAE (Liang 2018)", "0.4223", "0.426", "−0.0037", "near-snap"],
        ["Mult-DAE (Liang 2018)", "0.3969", "0.419", "−0.022", "fail, dropped"],
    ], MARGIN_L, I(2.75), I(7.6), col_w=[36, 14, 16, 14, 20], font_size=12, row_h=0.32,
        align=num(5), first_col_bold=True)
    caption(s, MARGIN_L, I(4.5), CONTENT_W, "Table 1.2. Bridge to printed numbers before porting to our ruler.")
    notes(s, "Before any system is scored on our ruler it is reproduced on its own published protocol.\n"
             "EASE and RecVAE snap to within 0.0005 of the printed numbers.\n"
             "Mult-DAE misses by 0.022 and is dropped rather than carried with an unexplained gap.\n"
             "The blind calibration panel later called this bridge exemplary.")

    # 5-6 certification battery ---------------------------------------------
    s, y = slide(prs, "Certification battery: gates G0 to G5")
    table(s, [
        ["Gate", "Demand", "Threshold / result"],
        ["G0  strength and identity", "reproduces the frozen tower; empty evidence set = the prior",
         "|μ0| = 0 to precision; Spearman 1.0000 vs frozen decoder bias"],
        ["G1  usable posterior", "posterior variance shrinks, more along the queried direction",
         "total 1881.9 → 1805.1; directional shrink 11.5× vs 2× bar"],
        ["G2  cold curve rises", "NDCG rises from turn 1 to turn 8, interval clear of zero",
         "pass, with a one-free-true-answer canary"],
        ["G3  sign and intensity", "flipped answer sign must hurt; graded must beat binarised",
         "sign-flip −0.2012; graded − binarised +0.0039 full [+0.0025, +0.0053]"],
        ["G4  refusal boundedness", "a refusal moves the belief far less than an answer",
         "α_refuse / α_answer = 0.0007 vs ≤ 0.05 bar"],
        ["G5  channel specificity", "concept operator lifts its members, beats its popularity projection",
         "member-lift AUC 0.626; +0.0087 over pop-projection, CI clear"],
    ], MARGIN_L, y + I(0.05), CONTENT_W, col_w=[22, 40, 38], font_size=10, row_h=0.46,
        first_col_bold=True)
    caption(s, MARGIN_L, I(4.6), CONTENT_W, "Table 1.3. Per-build gates, thresholds written before the run (1 of 2).")
    notes(s, "Eleven per-build gates and three one-time bridges; every threshold was written down before the run.\n"
             "G0: the empty interview reproduces the frozen tower exactly.\n"
             "G3: flipping an answer's sign must hurt, and graded ratings must beat binary likes.\n"
             "G4: a refusal barely moves the belief, ratio 0.0007 against a 0.05 bar.")

    s, y = slide(prs, "Certification battery: G6 to C3")
    table(s, [
        ["Gate", "Demand", "Result"],
        ["G6  existential controls", "wrong-user, shuffled and placebo evidence buy nothing", "pass"],
        ["G8  Σ load-bearing", "collapsing the covariance to cI degrades the number", "pass"],
        ["G9  ingestion isolation", "a fixed user-independent bank still improves with t", "pass"],
        ["G10 strategy discrimination", "oracle > random > adversarial-worst, intervals clean, full and tail",
         "pass; licenses every Chapter 2 comparison"],
        ["C1  published-number bridge", "match printed numbers on a published protocol",
         "done; EASE +0.0003, RecVAE +0.0005"],
        ["C2  answer-model transfer", "gain survives held-out behavioural answers and label noise",
         "run; behavioural and stated preferences agree ~21 %"],
        ["C3  human round-trip", "real people answer, held-out design", "not run; owed project-wide"],
    ], MARGIN_L, y + I(0.05), CONTENT_W, col_w=[24, 46, 30], font_size=10.5, row_h=0.40,
        first_col_bold=True)
    caption(s, MARGIN_L, I(4.6), CONTENT_W, "Table 1.4. Per-build gates and one-time bridges (2 of 2).")
    notes(s, "G10 is the gate that matters downstream: oracle beats random beats adversarial-worst, both axes, intervals clean. Without it no strategy comparison would be licensed.\n"
             "C2: behavioural and stated preferences agree only about a fifth of the time, which is why answers come from behaviour.\n"
             "C3, the human round-trip, has not been run; it is owed across the whole project.\n"
             "NOTE TO AUTHOR: say whether the C3 study is planned before the thesis submission.")

    # 7 recommender comparison ----------------------------------------------
    s, y = slide(prs, "Recommender comparison")
    rows = [["System", "Liang full", "Liang tail", "native full", "native tail", "NDCG@100"]]
    for r in [
        ("RecVAE (Shenbin 2020)", ".3540", ".2497", ".4273", ".3051", ".4537"),
        ("CASPER tower, i26c ep31", ".3496", ".2491", ".4244", ".3017", ".4492"),
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
    table(s, rows, MARGIN_L, y + I(0.02), CONTENT_W, col_w=[38, 12, 12, 12, 12, 14],
          font_size=9.5, row_h=0.215, align=num(6), first_col_bold=True, bold_cells={(2, 0)})
    caption(s, MARGIN_L, I(4.85), CONTENT_W, "Table 1.5. Fifteen systems rebuilt in-house, one split, 10,000 test users.")
    notes(s, "Fifteen systems, all rebuilt in-house and scored under identical conditions.\n"
             "The top four are within measurement of each other: RecVAE minus EASE is +0.0064 full, about 1.8 unpaired SE.\n"
             "The CASPER tower sits inside that frontier band.\n"
             "The conjugate belief accumulator, the consumer most elicitation work uses, gives up about 0.12 full NDCG against a learned fold.")

    # 8 gap map -------------------------------------------------------------
    s, y = slide(prs, "Gap map")
    fw = I(5.6)
    figure(s, "a_gap_map.png", MARGIN_L, y + I(0.02), width=fw,
           cap="Figure 1.1. Full-profile NDCG@10 against interview requirements met.")
    rx = MARGIN_L + fw + I(0.3)
    rw = CONTENT_W - fw - I(0.3)
    bullets(s, rx, y + I(0.05), rw, [
        "R2: ranks the whole catalogue from an evidence set, no refit",
        "R3: item, concept and free-text evidence through one interface",
        "R4: defined at every turn, including the empty interview",
        "Strongest rankers meet none; interview systems rank low",
        "Grey band: no full-profile ranker at all",
    ], size=13, height=I(3.0))
    notes(s, "Same fifteen systems on two axes: ranking quality on our ruler, and how many of the three interview requirements a system meets.\n"
             "RecVAE, EASE and EDLAE were built to score a fixed profile, so they meet none of the requirements.\n"
             "Systems built for interviews sit low on the y axis.\n"
             "The grey band holds ConTS, UNICORN, soft attributes, ICF, BCIE, PEBOL, GATE and LLM-CRS: no full-profile number can be computed for them.\n"
             "The top-right corner is what Chapter 1 fills.")

    # 9 tower vs RecVAE -----------------------------------------------------
    s, y = slide(prs, "Certified tower vs RecVAE", [
        "Tail indistinguishable; full 1.2 % relative behind",
        "Frontier instrument: a Chapter 2 null is not a weak recommender",
        "Epoch 31 kept by the pre-registered validation rule (epoch 23 scores 0.3505 on test)",
    ], body_box=(0.34, 1.26, 9.32, 1.5))
    table(s, [
        ["10,000 test users, paired bootstrap", "full", "tail"],
        ["RecVAE, our run, same ruler", "0.3540", "0.2497"],
        ["CASPER tower, i26c ep31", "0.3496", "0.2491"],
        ["paired difference", "−0.0043", "−0.0006"],
        ["95 % CI", "[−0.0059, −0.0028]", "[−0.0020, +0.0008]"],
    ], MARGIN_L, I(2.85), I(7.2), col_w=[46, 27, 27], font_size=12, row_h=0.3,
        align=num(3), first_col_bold=True)
    caption(s, MARGIN_L, I(4.5), CONTENT_W, "Table 1.6. Certified tower against RecVAE.")
    notes(s, "Against RecVAE on the same ruler the tower ties on tail and gives up 1.2 percent relative on full.\n"
             "That is a frontier instrument, so a null in Chapter 2 cannot be blamed on a weak recommender.\n"
             "Epoch 23 scores slightly higher on test; epoch 31 is kept because it wins the pre-registered validation rule.")

    # 10 graded input -------------------------------------------------------
    s, y = slide(prs, "Graded input")
    fw = I(5.3)
    figure(s, "a_graded_curve.png", MARGIN_L, y + I(0.02), width=fw,
           cap="Figure 1.2. Full NDCG@10 vs interview length, graded and binarised.")
    rx = MARGIN_L + fw + I(0.3)
    rw = CONTENT_W - fw - I(0.3)
    table(s, [
        ["identical support", "full", "tail"],
        ["graded fold-in", "0.3496", "0.2491"],
        ["binarised fold-in", "0.3457", "0.2471"],
        ["premium", "+0.0039", "+0.0020"],
        ["95 % CI", "[+.0025, +.0053]", "[+.0007, +.0031]"],
    ], rx, y + I(0.05), rw, col_w=[40, 30, 30], font_size=10, row_h=0.3,
        align=num(3), first_col_bold=True)
    bullets(s, rx, I(3.2), rw, [
        "Rating levels recover about half the RecVAE gap",
        "Premium largest on the short interview",
        "Signed, graded evidence is load-bearing",
    ], size=13)
    notes(s, "Same evidence, only the values change: rating levels instead of binary likes.\n"
             "The premium is +0.0039 full, interval clear of zero; about half of the gap to RecVAE.\n"
             "The two encodings converge as the interview lengthens, so the premium matters most where cold start lives.")

    # 11 concept affinity ---------------------------------------------------
    s, y = slide(prs, "Concept affinity: SEL vs LLM", [
        "Concept questions need a per-user affinity estimate",
        "Implicit watch behaviour: about 22× the signal of an LLM judgement",
        "Polarising concept bank answered 97.2 %; item bank 4.8 %",
        "Held-out user: 53.0 % of 1,031 concepts answerable; 25.2 % of top-100 films rated",
    ], body_box=(0.34, 1.26, 9.32, 1.9))
    table(s, [
        ["Signal source", "R² vs oracle concept affinity"],
        ["SEL: signed, exposure-corrected implicit affinity (PMI)", "0.376"],
        ["SEL + VAL: plus shrunk signed rating residual (λ = 3)", "0.398 (VAL adds +0.042)"],
        ["LLM-elicited ordinal judgement", "0.017"],
    ], MARGIN_L, I(3.25), I(8.0), col_w=[62, 38], font_size=11.5, row_h=0.3,
        align=num(2), first_col_bold=True)
    caption(s, MARGIN_L, I(4.6), CONTENT_W, "Table 1.7. Sources of concept affinity.")
    notes(s, "To ask about a tag we need to know how much each user likes it.\n"
             "Signed, exposure-corrected watch behaviour explains 0.376 of the oracle affinity; an LLM's stated judgement explains 0.017.\n"
             "Concepts matter because they are answerable: a polarising concept bank is answered 97 percent of the time, a polarising item bank under 5.\n"
             "This is why the LLM answer apparatus was retired.")

    # 12 concept channel gates ----------------------------------------------
    s, y = slide(prs, "Concept channel: gates")
    table(s, [
        ["Signed triple gate (retrained operator)", "value"],
        ["signed − clip-up at q16", "+0.01043 [+0.00753, +0.01336]"],
        ["popularity counterfeit (control)", "−0.04520, declines"],
        ["value-permuted contrast (control)", "+0.03353 [+0.03095, +0.03607]"],
        ["concept sign-flip on its own operator", "−0.2012, interval clean"],
    ], MARGIN_L, y + I(0.05), CONTENT_W, col_w=[55, 45], font_size=11.5, row_h=0.3,
        align=num(2), first_col_bold=True)
    table(s, [
        ["Concept − item, ranked by polarisation", "full", "tail"],
        ["q16", "+0.0146", "+0.0265"],
        ["95 % CI", "[+.0094, +.0197]", "[+.0204, +.0325]"],
        ["peak, q8", "+0.0189", "+0.0310"],
    ], MARGIN_L, I(3.15), I(6.5), col_w=[46, 27, 27], font_size=11.5, row_h=0.3,
        align=num(3), first_col_bold=True)
    caption(s, MARGIN_L, I(4.55), CONTENT_W, "Table 1.8. Concept operator controls, and concepts against items.")
    notes(s, "The signed gate: real signed evidence beats clip-up by +0.010 at sixteen questions, while a popularity counterfeit declines.\n"
             "Flipping the sign of a concept answer costs 0.20; the operator is reading the sign.\n"
             "Concept questions beat item questions by +0.0146 full and +0.0265 tail at sixteen questions, peak at eight.\n"
             "This licenses concepts as first-class actions for Chapter 2.")

    # 13 fold-in operator ---------------------------------------------------
    s, y = slide(prs, "Fold-in operator")
    fw = I(7.8)
    yb = figure(s, "a_architecture.png", MARGIN_L, y + I(0.0), width=fw,
                cap="Figure 1.3. Four evidence channels enter one slot; a permutation-invariant fold returns one latent point.")
    bullets(s, MARGIN_L, yb + I(0.05), CONTENT_W, [
        "Channel-agnostic: item factor, whitened concept centroid, text image, raw direction",
        "Concept operator gated, zero-initialised: empty interview bit-identical to the tower",
        "Whitening: concept member-AUC 0.44 raw → 0.94",
        "Additive union at 8 answers 0.0991 full, below the 0.1279 intercept; gated fold 0.1759",
    ], size=13)
    notes(s, "Evidence in, latent point out; no gradient step at inference; defined from turn zero.\n"
             "One slot for all channels, which is what lets Chapter 3 feed a raw synthesised direction into the same operator.\n"
             "The gate replaced additive union, which at eight answers scored below asking nothing.\n"
             "Redundancy check: eight correlated concepts must not lose to four; the additive operator failed exactly there.")

    # 14 free-text example --------------------------------------------------
    s, y = slide(prs, "Free-text channel: example")
    table(s, [
        ["phrase typed by the user", "nearest concept", "cos", "top films returned"],
        ["“the quiet dignity of losing”", "redemption", "0.47", "Umberto D. · Tyson · Red Army"],
        ["“people trapped by a decision they made years ago”", "history", "0.39",
         "Music Box · Judgment at Nuremberg"],
        ["“two people talking for the whole film”", "dialogue", "0.54",
         "Sweet Smell of Success · Trouble in Paradise"],
        ["“movies about grief”", "feel good movie", "0.54",
         "Rachel, Rachel · Mean Creek · The Woodsman"],
    ], MARGIN_L, y + I(0.05), CONTENT_W, col_w=[34, 18, 8, 40], font_size=11.5, row_h=0.42,
        align=[L, L, R, L], first_col_bold=True)
    bullets(s, MARGIN_L, I(3.75), CONTENT_W, [
        "Last row: nearest concept wrong, films still on topic",
        "Held-out concept direction from its name alone: top-10 overlap 0.167 vs 0.0005 random",
    ], size=14)
    caption(s, MARGIN_L, I(4.7), CONTENT_W, "Table 1.9. Typed phrase to concept to ranking.")
    notes(s, "A typed phrase is mapped to the nearest concept direction and folded in like any other answer.\n"
             "The first three rows land on sensible concepts and return films on topic.\n"
             "The last row is the honest failure: grief maps to feel-good movie, yet the films returned are still about grief.\n"
             "A held-out concept predicted from its name alone reaches cosine 0.638 to the true direction.\n"
             "NOTE TO AUTHOR: Chapter 1 has no worked answer-sequence example; this is the nearest concrete artefact.")

    # 15 status -------------------------------------------------------------
    s, y = slide(prs, "Chapter 1: status", [
        "Closed, 41 pages, not yet submitted; target TORS (rolling, no deadline)",
        "ECIR rejection: too long, instrument without application",
        "Blind calibration panel, July 2026: PEBOL, soft attributes, ours",
        ("all three weak reject overall; accept at TORS", 1),
        ("ours the only accept on the reproducibility track", 1),
        ("all three top objections: simulator circularity", 1),
    ], body_box=(0.34, 1.26, 9.32, 2.2))
    table(s, [
        ["Handed to Chapters 2 and 3", "Licensed by"],
        ["one frozen consumer for every strategy comparison", "G10, intervals clean on both axes"],
        ["fold-in operator that ingests any channel", "channel-agnostic slot; identity gate by zero-init"],
        ["concept questions as first-class actions", "G5; concepts beat items +0.0146 full / +0.0265 tail"],
        ["continuous direction interface", "raw directions enter the same slot (Chapter 3)"],
    ], MARGIN_L, I(3.55), CONTENT_W, col_w=[46, 54], font_size=11, row_h=0.27,
        first_col_bold=True)
    notes(s, "Chapter 1 is closed at 41 pages; ECIR rejected it as too long and application-free, which is why TORS is the target.\n"
             "A blind panel with a hostile reviewer rated ours alongside PEBOL and soft attributes: all weak reject at conference venues, all accept at TORS.\n"
             "What goes forward: the frozen consumer, the channel-agnostic fold-in, concept actions, and the direction interface Chapter 3 uses.\n"
             "NOTE TO AUTHOR: confirm the TORS submission timing you want to state.")


# ----------------------------------------------------------- flow chart ----
def _pipeline_chart(s):
    # user + channels (left)
    person(s, I(0.55), I(1.45), I(0.6))
    textbox(s, I(0.3), I(2.1), I(1.1), I(0.3), "user", size=11, align=C)
    wbox(s, I(0.35), I(2.6), I(1.55), I(0.42), "item answer: “Yes”", size=9)
    wbox(s, I(0.35), I(3.15), I(1.55), I(0.42), "concept answer: “dark humour”", size=9)
    wbox(s, I(0.35), I(3.7), I(1.55), I(0.42), "free text: “movies about grief”", size=9)
    # container
    container(s, I(2.3), I(1.35), I(7.35), I(3.75))
    icon(s, "robot", I(2.4), I(1.4), I(0.55))
    textbox(s, I(2.3), I(1.95), I(0.8), I(0.3), "CASPER", size=10, bold=True, align=C)
    # channel boxes
    b_item = obox(s, I(3.3), I(1.55), I(1.45), I(0.5), ["item channel", "item factor"], size=9)
    b_conc = obox(s, I(3.3), I(2.35), I(1.45), I(0.5), ["concept channel", "whitened centroid"], size=9)
    b_text = obox(s, I(3.3), I(3.15), I(1.45), I(0.5), ["text channel", "adapter image"], size=9)
    b_dir = obox(s, I(3.3), I(3.95), I(1.45), I(0.5), ["raw direction", "(Chapter 3)"], size=9)
    b_fold = obox(s, I(5.25), I(2.3), I(1.5), I(0.9), ["fold-in operator", "set → latent point", "no gradient step"], size=9)
    b_tower = obox(s, I(7.25), I(2.3), I(1.35), I(0.9), ["frozen tower", "i26c ep31"], size=9)
    b_rank = wbox(s, I(7.25), I(3.65), I(1.35), I(0.7), ["ranked catalogue", "NDCG@10 full / tail"], size=9)
    b_gate = obox(s, I(5.25), I(3.65), I(1.5), I(0.7), ["gate battery", "G0–G10, C1–C3"], size=9, dark=True)
    database(s, I(8.85), I(1.5), I(0.45), I(0.5))
    textbox(s, I(8.6), I(2.02), I(0.95), I(0.3), "18,359 items", size=8, align=C)
    # arrows: user answers -> channels
    arrow(s, I(1.9), I(2.81), I(3.3), I(1.8))
    arrow(s, I(1.9), I(3.36), I(3.3), I(2.6))
    arrow(s, I(1.9), I(3.91), I(3.3), I(3.4))
    # channels -> fold
    for yy in (1.8, 2.6, 3.4, 4.2):
        arrow(s, I(4.75), I(yy), I(5.25), I(2.75))
    arrow(s, I(6.75), I(2.75), I(7.25), I(2.75))
    arrow(s, I(7.92), I(3.2), I(7.92), I(3.65))
    arrow(s, I(6.0), I(3.2), I(6.0), I(3.65), head=False)
    arrow(s, I(8.85), I(1.9), I(8.3), I(2.3))
    caption(s, MARGIN_L, I(5.15), CONTENT_W, "Figure 1.4. The instrument: evidence channels, fold-in, frozen tower, ruler, and the gate battery.", size=10)


# -------------------------------------------------------------- appendix ----
def build_appendix_ch1(prs):
    s, y = slide(prs, "Appendix 1.1: open gates")
    table(s, [
        ["item", "status", "detail"],
        ["G4 confidence-ordering sub-test", "unmeasurable by design",
         "no ground truth for user confidence on MovieLens; refusal-boundedness itself passes at 0.0007 vs ≤ 0.05"],
        ["G1c calibration sub-gate", "dropped, not failed",
         "Spearman(σ, held-out NLL) 0.2393 vs 0.25 bar; popularity of the targets predicts the same NLL at ρ = −0.6964, profile size at 0.5195; gate ill-posed. G1a, G1b pass"],
        ["volume-leak gate (KT-A3)", "fails, pending ruling",
         "ridge R² from fold latent at q8 to log user volume: clip-up 0.02365 → signed 0.24928, vs ≤ 0.05 bar"],
        ["G5 split", "open",
         "operators learn 'what X-likers watch' rather than 'members of X': member-AUC 0.630 at k = 1; Spearman 0.948 vs log-popularity, 0.030 vs member-ness; pop-projection control passes +0.0087"],
        ["C3 human round-trip", "never run", "owed project-wide; holdout design mandatory; satisfaction survey not acceptable"],
        ["internal inconsistency", "flagged",
         "conjugate accumulator uniformity cost: 0.12 full at :881, 0.15 at :803; quote 0.12"],
    ], MARGIN_L, y + I(0.02), CONTENT_W, col_w=[22, 16, 62], font_size=9, row_h=0.5, first_col_bold=True)
    notes(s, "Every per-build gate passes; these are the qualifications.\n"
             "The volume-leak gate fails against its pre-registered bar and awaits a ruling.\n"
             "NOTE TO AUTHOR: rule on the volume-leak gate (KT-A3) and fix the 0.12 vs 0.15 inconsistency in chapterA_v2.tex.")

    s, y = slide(prs, "Appendix 1.2: concept channel ledger")
    table(s, [
        ["quantity", "value"],
        ["fold ablation (old tower): additive union, member AUC / full@8", "0.820 / 0.0991, below the 0.1279 intercept"],
        ["fold ablation: gated fold, member AUC / full@8 / tail@8", "0.626 / 0.1759 / 0.0841 (from 0.1322 / 0.0503 at one answer)"],
        ["redundancy: 8 correlated top-SEL concepts vs 4", "0.1685 / 0.0906 vs 0.1572 / 0.0777; additive fails (m8 0.0991 vs m4 0.1349)"],
        ["whitening: raw centroid member-AUC → whitened", "0.44 → 0.94"],
        ["open vocabulary: held-out concept from its name", "top-10 overlap 0.167 vs 0.0005 random; cosine 0.638"],
        ["specificity: like → member AUC / dislike / other concept", "0.626 / 0.416 / 0.498; sign +0.2096 [+0.1986, +0.2208]; identity +0.1278 [+0.1195, +0.1361]"],
        ["answer-model decomposition, polarisation bank, q8", "selection oracle +0.0691 [0.0656, 0.0726]; utility oracle +0.1381 above SEL; answers per q8: concept 7.93 vs item 3.03"],
        ["three ledgers, not to be mixed", "(1) strategy×channel suite on the superseded i25 fold, 'items win' retracted; (2) policy A, intercept .1314/.0234; (3) policy B, intercept .15855/.02819, the one used here"],
    ], MARGIN_L, y + I(0.02), CONTENT_W, col_w=[40, 60], font_size=9, row_h=0.42, first_col_bold=True)
    notes(s, "Reference numbers for the concept channel, in one place.\n"
             "Three result ledgers exist on different folds and policies; only policy B's numbers are used in this deck.")


if __name__ == "__main__":
    import os, sys
    prs = new_deck()
    build_ch1(prs)
    build_appendix_ch1(prs)
    out = sys.argv[1] if len(sys.argv) > 1 else "part_ch1_test.pptx"
    prs.save(out)
    print("saved", out, len(prs.slides._sldIdLst), "slides")
