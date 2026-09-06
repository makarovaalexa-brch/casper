"""Intro, closing and methods-appendix slides for the supervisor deck.

Style: STYLE.md.  Numbers: build_deck.py (itself built from the chapter .tex).
"""
from pptx.util import Inches, Pt
from pptx.enum.text import PP_ALIGN
from deck_lib2 import (new_deck, title_slide, section_slide, slide, notes, textbox,
                       bullets, caption, table, obox, wbox, container, bubble, arrow,
                       person, icon, MARGIN_L, CONTENT_W)

I = Inches
L, C, R = PP_ALIGN.LEFT, PP_ALIGN.CENTER, PP_ALIGN.RIGHT
CASPER = "CASPER: Cold-start Adaptive Selection for Preference Elicitation in Recommendation"


def _num(n):
    return [L] + [R] * (n - 1)


# ===================================================================== intro
def build_intro(prs):
    # 1. title
    title_slide(prs, "Learning preference elicitation strategies for cold-start "
                     "conversational recommendation",
                "Aleksandra Makarova", "September 2026", subtitle=CASPER)

    # 2. research goals
    s, _ = slide(prs, "Research goals", [
        "Cold-start user: no history, a few questions before the first recommendation",
        "Three questions, one measurement stack",
        ("Chapter 1: certified cold-start recommender, fold-in operator, fixed metric", 1),
        ("Chapter 2: what a cold interview should optimise", 1),
        ("Chapter 3: questions outside the catalogue, emitted and rendered in English", 1),
        "Every headline read against a pre-registered control",
        "Each chapter measured on the instrument the previous one certified",
    ])
    notes(s, "\n".join([
        "Setting: a new user with no history; the system may ask a few questions first",
        "Three chapters, each hands the next a certified capability",
        "Chapter 1 builds the instrument, Chapter 2 asks what the interview should optimise, "
        "Chapter 3 opens the action space beyond the question bank",
        "Method throughout: pre-register the reading rule, run it against its own null first",
        "Controls overturned the headline three times: C0 random-direction floor, C3B twin, "
        "C1 untrained warm start",
    ]))

    # 3. the cold-start problem
    s, y = slide(prs, "The cold-start problem", [
        "New user, no rating history",
        "Ask a question, fold the answer into a belief, choose the next question",
        "Rank 18,359 items after a handful of answers",
    ], size=16, body_box=(0.34, 1.2, 9.32, 1.3))
    table(s, [
        ["Setting, fixed once", ""],
        ["Dataset", "MovieLens-25M, Liang strong-generalisation recipe, catalogue-restricted"],
        ["Split", "train 140,768 / val 10,000 / test 10,000 users, seed 98765, per-user 80/20"],
        ["Item vocabulary", "18,359 train vocabulary (catalogue 18,430 = films with ≥ 20 ratings)"],
        ["Question bank", "19,390 = 18,359 item questions + 1,031 concept questions"],
        ["Metric", "full and tail NDCG@10, scored with the learned decoder bias"],
        ["Asking nothing", "0.15855 full / 0.02819 tail (popularity prior)"],
        ["Whole profile", "0.3496 full / 0.2491 tail (same recommender, full history)"],
    ], MARGIN_L, I(2.55), CONTENT_W, col_w=[22, 78], font_size=10.5, row_h=0.26,
        first_col_bold=True)
    notes(s, "\n".join([
        "The loop: question, answer (like / dislike / no answer), fold-in to a latent point, "
        "next question chosen from the belief",
        "One dataset, one split, one metric for all three chapters; fixed July 2026",
        "Two anchors: asking nothing scores 0.159 full, the whole profile 0.350; the interview "
        "lives between them",
        "Tail NDCG = ranking outside the 190 head items; always reported beside full",
    ]))

    # 4. CASPER components (flow chart)
    s, _ = slide(prs, "CASPER components")
    person(s, I(0.5), I(1.75), I(0.6))
    textbox(s, I(0.3), I(2.42), I(1.0), I(0.3), "user", size=11, align=C)
    bubble(s, I(1.3), I(1.42), I(1.85), I(0.6), "Do you like dark humour films?", size=10, tail="right")
    bubble(s, I(1.3), I(2.45), I(1.85), I(0.45), "Yes", size=10, tail="left")
    arrow(s, I(3.15), I(2.68), I(4.25), I(1.78), label="answer", size=9, label_dy=I(0.02), label_dx=I(-0.5))

    container(s, I(3.35), I(1.3), I(6.3), I(3.68))
    icon(s, "robot", I(3.42), I(1.35), I(0.6))
    textbox(s, I(3.37), I(1.95), I(0.7), I(0.3), "CASPER", size=10, bold=True, align=C)

    w, h = I(1.2), I(0.62)
    xs = [I(4.25), I(5.63), I(7.01), I(8.39)]
    y1, y2, y3 = I(1.45), I(2.55), I(3.65)
    obox(s, xs[0], y1, w, h, ["fold-in", "operator"], size=10)
    obox(s, xs[1], y1, w, h, ["interview state", "(latent point)"], size=10)
    obox(s, xs[2], y1, w, h, ["question selector", "(tree / policy)"], size=10)
    obox(s, xs[3], y1, w, h, ["question bank", "18,359 items", "1,031 concepts"], size=9)
    obox(s, xs[2], y2, w, h, ["vector emitter", "(new direction)"], size=10)
    obox(s, xs[3], y2, w, h, ["text renderer", "(language model)"], size=10)
    obox(s, xs[1], y3, w, h, ["frozen tower", "recommender"], size=10)
    wbox(s, xs[2], y3, w, h, ["ranked list", "NDCG@10 full / tail"], size=10)
    obox(s, xs[0], I(4.42), I(5.34), I(0.4),
         "fixed evaluation: one split, full and tail NDCG@10", size=10, dark=True)

    mid1 = y1 + h // 2
    arrow(s, xs[0] + w, mid1, xs[1], mid1)
    arrow(s, xs[1] + w, mid1, xs[2], mid1)
    arrow(s, xs[2] + w, mid1, xs[3], mid1)
    arrow(s, xs[1] + w // 2, y1 + h, xs[1] + w // 2, y3)                       # state -> tower
    arrow(s, xs[1] + w, y1 + h, xs[2], y2 + h // 2)                            # state -> emitter
    arrow(s, xs[2] + w, y2 + h // 2, xs[3], y2 + h // 2)                       # emitter -> renderer
    arrow(s, xs[1] + w, y3 + h // 2, xs[2], y3 + h // 2)                       # tower -> ranked list
    arrow(s, xs[3] + w // 2, y1 + h, xs[3] + w // 2, y2 - I(0.02), head=False)
    arrow(s, xs[3] + w // 2, y2 + h, xs[3] + w // 2, y3 + h // 2, head=False)
    arrow(s, xs[3] + w // 2, y3 + h // 2, xs[2] + w + I(0.02), y3 + h // 2, head=False)
    textbox(s, xs[3] - I(0.05), y3 + I(0.05), w + I(0.1), I(0.5),
            ["question", "→ user"], size=9, align=C)

    textbox(s, I(0.34), I(3.55), I(2.9), I(1.3), [
        "Chapter 1: fold-in, tower, ruler",
        "Chapter 2: selector, question bank",
        "Chapter 3: emitter, renderer",
    ], size=11)
    notes(s, "\n".join([
        "Answers fold into a latent point; the same point is the interview state and the "
        "tower's input",
        "Chapter 2's selector picks the next question from the 19,390-question bank",
        "Chapter 3 replaces the bank lookup: a vector emitter proposes a direction, a language "
        "model renders it as a question",
        "All chapters scored on the same split and metric: full and tail NDCG@10 over the whole catalogue",
        "NOTE TO AUTHOR: the two example utterances are illustrative, not from a transcript",
    ]))

    # 5. thesis structure
    s, y = slide(prs, "Thesis structure")
    table(s, [
        ["Chapter", "Question", "Main result", "Hands forward / status"],
        ["1  The instrument",
         "can an elicitation result be measured at all without certifying the recommender",
         "strategy rankings invert when the consumer changes; certified tower ties RecVAE on tail",
         "frozen recommender, certified fold-in operator, fixed metric; chapter closed"],
        ["2  The interview",
         "what should a cold interview optimise",
         "published selectors optimise a surrogate; a measured-NDCG split beats the static order, "
         "+0.01035 full (TEST)",
         "distilled policy E1, 19,390-question bank, per-state question values; closed, "
         "test look spent"],
        ["3  Questions outside the catalogue",
         "can a direction the bank does not contain be emitted, said in English, and audited",
         "245/245 questions preserved while emitting blends; vector conditions text +0.1114; "
         "94.7 % askable",
         "capability claim, not value; six experiments banked, short form drafted"],
    ], MARGIN_L, I(1.35), CONTENT_W, col_w=[16, 26, 32, 26], font_size=10.5, row_h=0.55,
        first_col_bold=True)
    notes(s, "\n".join([
        "Chapter 1 is closed and unsubmitted; Chapter 2 closed on 30 August with the test look "
        "spent; Chapter 3 has six experiments banked",
        "Each chapter hands the next a certified object: recommender and ruler, then policy and "
        "bank, then the emitter",
        "Chapter 3 claims capability, not value: circularity of the answer model is declared "
        "there and only there",
    ]))


# ===================================================================== close
def build_close(prs):
    s, _ = slide(prs, "Summary: three chapters")
    table(s, [
        ["", "Claim", "Evidence (10,000 held-out TEST users)"],
        ["1", "an elicitation result is a property of a strategy and an instrument; "
              "certify the instrument first",
         "two strategies invert their ordering when the consumer changes; tower ties RecVAE on "
         "tail; ruler snaps to four published numbers"],
        ["2", "the binding constraint is the split criterion and the channel, not the conditioning",
         "21 rules on one instrument: static order beats every published adaptive arm, 138 of 138 "
         "cells; measured-NDCG split +0.01035 full / +0.00705 tail; bank change buys 3× more "
         "tail than conditioning"],
        ["2", "answerability, not informativeness, orders the channels",
         "16 item questions collect 3.11 answers, 16 concept questions 12.36; answering at all "
         "≈ +0.25 vs best-question effect +0.048"],
        ["3", "the action space can be opened: a direction outside the bank, emitted, said in "
              "English, audited",
         "245/245 questions preserved; vector conditions text +0.1114; 94.7 % askable; nearest "
         "catalogue text anti-aligned −0.158 where the rendered word carries +0.204"],
    ], MARGIN_L, I(1.35), CONTENT_W, col_w=[4, 36, 60], font_size=10.5, row_h=0.55,
        first_col_bold=True)
    notes(s, "\n".join([
        "A cold interview is limited by what the user can answer and what the system can ask, "
        "not by how cleverly it picks from a fixed list",
        "Chapter 1: the instrument is part of the measurement",
        "Chapter 2: published selectors optimise the wrong objective; conditioning pays once "
        "the split optimises measured NDCG",
        "Chapter 3: the bank is not the boundary of what can be asked",
        "Common method: pre-registered reading rule, run against its own null before reading",
    ]))

    s, _ = slide(prs, "Publication options")
    table(s, [
        ["", "State", "Option 1 (preferred)", "Option 2", "Option 3"],
        ["1  instrument", "closed, 41 pp, unsubmitted",
         "TORS: rolling, no deadline, no travel; length not penalised",
         "RecSys or ECIR reproducibility track",
         "hold until the human round-trip study exists"],
        ["2  interview", "closed 2026-08-30; test look spent, replicates",
         "ECIR 2027 full paper, deadline 2 Oct 2026; sections 1–3 cut to ~12 pp",
         "RecSys, or its reproducibility track",
         "add RecVAE re-scoring of the frozen tree first"],
        ["2′  unprompted requests", "held back, not cut",
         "second submission (RecSys or CIKM): request corpus, opener-fitted trees, distilled policy",
         "thesis chapter section only", "—"],
        ["3  off-catalogue", "six experiments banked; short form drafted; no deadline",
         "RecSys LBR (the form the draft already takes)",
         "IntRS or KaRS workshop", "arXiv plus the thesis chapter"],
    ], MARGIN_L, I(1.3), CONTENT_W, col_w=[13, 21, 30, 19, 17], font_size=9.5, row_h=0.5,
        first_col_bold=True)
    textbox(s, MARGIN_L, I(4.6), CONTENT_W, I(0.45), [
        "Only hard date: 2 October 2026 (Chapter 2)",
        "Constraints: solo, about a year left, no travel; ~8 load-bearing references still missing",
    ], size=11)
    notes(s, "\n".join([
        "Roster reset on 18 August; options, not decisions",
        "Chapters 4 and 5 of the old plan: free-recall absorbed into Chapter 2, LLM dialogue "
        "merged into Chapter 3",
        "Chapter 1 to TORS: rolling, no travel, length not a penalty; cleared every draft in "
        "the blind panel",
        "Chapter 2 to ECIR 2027: the one binding deadline, 2 October; cut sections 1–3 to 12 pages",
        "The unprompted-request work: second submission or chapter section",
        "Chapter 3 as a RecSys late-breaking result or a workshop paper",
        "Bibliography completeness blocks any submission",
        "NOTE TO AUTHOR: confirm the 'no travel' constraint should be stated aloud",
    ]))

    s, _ = slide(prs, "Next steps", [
        "Score Chapter 2's frozen NDCG tree through RecVAE (one machine-night)",
        "ECIR cut: Chapter 2 sections 1–3 to ~12 pp, deadline 2 Oct 2026",
        "Close the bibliography (~8 load-bearing works)",
        "Chapter 3 short form: shrink the audit, lead with the example interview",
        "Fix the templating failure on blends (2 of 6 malformed)",
        "Human round-trip study (gate C3): never run, blocks the strongest claims in all chapters",
        "Decisions: request work as second submission or section; round-trip study on "
        "answerability or on quality",
    ], size=16)
    notes(s, "\n".join([
        "Ordered by what unblocks the most",
        "RecVAE scoring answers the one-consumer objection; scoring only, no refit",
        "The ECIR cut is the binding task",
        "The round-trip study is owed project-wide; all three top objections in the blind panel "
        "were simulator circularity",
        "Two decisions for this meeting: the request work's home, and whether the study tests "
        "answerability or quality",
    ]))


# ============================================================ methods appendix
def build_appendix_methods(prs):
    section_slide(prs, "Appendix", "methods, bounds, retractions, numbers")

    # M1a — answer model and circularity
    s, _ = slide(prs, "Appendix M1: answer model")
    table(s, [
        ["Answer", "recorded ratings (items); signed, exposure-corrected behavioural affinity "
                   "SEL (concepts); like / dislike / no answer"],
        ["Not admitted", "LLM-derived answer or answerability grids: popularity-dominated, "
                         "3 of 4 gates null-passing, distilled value ≈ EASE (July 2026 audit)"],
        ["Answerability", "structural rule: rated ∨ popularity > τ (items); ≥ k rated members "
                          "(concepts); not a learned model"],
        ["Firewall (Chapter 1, I2)", "answers are recorded behaviour and fold-in only"],
        ["Declared circularity", "one place: Chapter 3 translation cost; both arms share the "
                                 "simulator, only the difference is read, absolute rows never headlined"],
        ["Chapter 3 contract", "simulated answerer places the policy at interview states, "
                               "nothing else"],
    ], MARGIN_L, I(1.35), CONTENT_W, col_w=[22, 78], font_size=11, row_h=0.5,
        first_col_bold=True, header_rows=0)
    notes(s, "\n".join([
        "Answers are recorded behaviour; nothing LLM-derived enters an answer",
        "Answerability is a transparent rule, not a model",
        "Circularity is declared in exactly one place and read as a difference",
    ]))

    # M1b — statistical conventions
    s, _ = slide(prs, "Appendix M1: statistical conventions", [
        "Paired percentile bootstrap: 10,000 resamples, seed 98765, clustered over users",
        "Self-comparison must come out exactly zero",
        "Every statistic carries an interval and declares its aggregation unit (user vs state)",
        "Every registered reading rule run against its own null at design time",
        "Wilson intervals for proportions; cluster bootstrap (user as cluster) for medians",
        "★ in this deck: the paired interval excludes zero",
    ], size=15)
    notes(s, "\n".join([
        "One bootstrap convention across chapters",
        "A point estimate without an interval is a description, not a result",
        "States nest inside users, so user-row and state proportions are different quantities",
    ]))

    # M2 — stated bounds
    s, _ = slide(prs, "Appendix M2: stated bounds")
    table(s, [
        ["Bound", "Finding", "Cost to the claim"],
        ["P1 / P1b (Chapter 2, validation)",
         "policy emits a held-out question 0.0000 at top-1 / top-5 over 38,336 states; "
         "popularity ranking 1.45 % top-5; E1 ranks the measured-best question 208th–374th of 1,000",
         "'emits questions it was never taught' not available"],
        ["C1 emission scope (Chapter 3)",
         "withheld states: cos to own target −0.11948; state-blind constant wins by +0.8893, "
         "14/15; untrained warm start matches the registered statistic",
         "policy emits where the landscape was measured; no held-out emission quoted"],
        ["One consumer (Chapter 2)", "tree result rests on one consumer; flip section shows "
                                     "consumer-dependence",
         "fixable by scoring the frozen tree through RecVAE"],
        ["Group-stamped openers", "routing partly circular; scramble control cannot detect it",
         "claim adaptivity beyond the closed tree, not routing"],
        ["Inferred answer model", "structural answerability rule is load-bearing; request corpus "
                                  "bounds open-answer richness from above",
         "gate C3 (human round-trip) never run"],
        ["Seed stability (Chapter 2)", "GLM-UCB and Thompson cells drift up to ±0.0026 "
                                       "(iALS basis re-drawn)",
         "quote those rows to three decimals"],
        ["Single seed (Chapter 3 adapters)", "one training seed per arm", "'robust' not used"],
    ], MARGIN_L, I(1.3), CONTENT_W, col_w=[20, 50, 30], font_size=9.5, row_h=0.48,
        first_col_bold=True)
    notes(s, "\n".join([
        "Every bound on this list was found by an internal control",
        "P1/P1b and C1 bound the generalisation claims in Chapters 2 and 3",
        "One consumer is the strongest open objection; the fix is a scoring pass, not a refit",
    ]))

    # M3a — not built, not run
    s, _ = slide(prs, "Appendix M3: not built, not run", [
        "Selectors not built (Chapter 2 prints the table): ICF, SEL-BALD, PEBOL, fMF, PERE, "
        "Greedy SLIM / QBandit, RBMF / RMVA, GreedyExtend, FacT / FacT-CRS, UpsRec, MeLU, "
        "a-stratification",
        "Gate C3, the human round-trip study: owed project-wide",
        "Hu & Yu replication at depths 8 and 10; Part-A pairwise arm; fMF RMSE panel; "
        "Netflix / Yahoo",
        "The 15-minute routing probe, required before the request section ships alone",
        "Chapter 2's demonstration through a second consumer",
    ], size=14)
    notes(s, "\n".join([
        "The ledger of absences, stated as unrun rather than implied",
        "The routing probe and the second consumer are the two cheap items",
    ]))

    # M3b — retracted claims
    s, _ = slide(prs, "Appendix M3: retracted claims")
    table(s, [
        ["Retracted claim", "What killed it"],
        ["'items win the long interview' (Chapter 1)", "ran on the superseded i25 fold; "
                                                       "retracted at the G0 audit"],
        ["tree takes the tail at K=8, +0.0068 (Chapter 2)",
         "comparator selected under a narrower mask; re-differenced −0.0016, does not separate"],
        ["+0.02092 tail advantage (Chapter 2)",
         "two thirds was comparator weakness; rebuilt order shares 4 of 16 questions, "
         "0 of 16 positions"],
        ["'the prediction is falsified, and not narrowly' (C0)",
         "registered rule had no power; a random direction lands in the same cell"],
        ["C4 pilot emission cells", "34/42 item-hot and 32/42 concept-hot renders were the empty "
                                    "string; mean-removed norm 38.3× the target's"],
        ["C3 PoC 'success' cell", "zero functioning controls, unpowered judge; variant (a) "
                                  "structurally unconstructible"],
        ["+47 % adaptivity magnitude; all pre-July interview numbers",
         "retired LLM answer environment; directions survive as priors, magnitudes re-establish"],
    ], MARGIN_L, I(1.35), CONTENT_W, col_w=[40, 60], font_size=10.5, row_h=0.42)
    notes(s, "\n".join([
        "Seven withdrawn claims, each with the control that withdrew it",
        "Directions from the retired environment survive as priors only",
    ]))

    # M4a — numbers, Chapters 1 and 2
    s, _ = slide(prs, "Appendix M4: numbers (1, 2)")
    lw = I(4.1)
    table(s, [
        ["Chapter 1", ""],
        ["tower TEST full / tail NDCG@10", "0.3496 / 0.2491"],
        ["RecVAE, same ruler", "0.3540 / 0.2497 (paired −0.0043 / −0.0006)"],
        ["graded − binarised", "+0.0039 [+0.0025, +0.0053] / +0.0020 [+0.0007, +0.0031]"],
        ["concept affinity R²: SEL / +VAL / LLM", "0.376 / 0.398 / 0.017"],
        ["concept − item at q16 (POLAR)", "+0.0146 full / +0.0265 tail"],
        ["polarising bank answer rate: item / concept", "4.8 % / 97.2 %"],
        ["published-number bridge", "EASE +0.0003, RecVAE +0.0005 (ML-20M NDCG@100)"],
    ], MARGIN_L, I(1.3), lw, col_w=[52, 48], font_size=9, row_h=0.3,
        align=_num(2), first_col_bold=True)
    rx = MARGIN_L + lw + I(0.3)
    table(s, [
        ["Chapter 2", ""],
        ["battery", "23 rows · 21 rules · 13 published methods"],
        ["static beats every adaptive arm", "138 of 138 cells"],
        ["greedy items − best adaptive, k=16", "+0.0281 [+0.0247, +0.0314] full"],
        ["NDCG tree − static on TEST, K=16", "+0.01035 [+0.00739, +0.01323] / "
                                             "+0.00705 [+0.00394, +0.01012]"],
        ["split-criterion ablation", "56 of 56 intervals exclude zero"],
        ["bank change vs conditioning (tail)", "+0.02199 vs +0.00705"],
        ["answers per 16 questions: item / concept", "3.11 / 12.36"],
        ["Reddit corpus", "4,431 threads · 63.0 % name a film · linking precision 0.960"],
        ["distilled policy on TEST, K=15", "+0.00343 ★ pooled · +0.00946 ★ volunteers"],
    ], rx, I(1.3), CONTENT_W - lw - I(0.3), col_w=[42, 58], font_size=9, row_h=0.3,
        align=_num(2), first_col_bold=True)
    notes(s, "Lookup page for questions; ★ = paired interval excludes zero")

    # M4b — numbers, Chapter 3
    s, _ = slide(prs, "Appendix M4: numbers (3)")
    table(s, [
        ["Chapter 3", ""],
        ["C0 max cos₁ over 160,000 states", "0.1203 (mean-removed +0.488 vs +0.170 random)"],
        ["C1 gates", "245/245 distinct · snap agreement 0.9950 · fit 0.96958"],
        ["C1 held-out states", "−0.11948; state-blind constant wins by +0.8893"],
        ["C3B  A − A′", "+0.1114 [+0.1054, +0.1173]"],
        ["C3B  A as share of gold-phrase ceiling", "107 % (ceiling +0.0892)"],
        ["C3B  identification", "0.6778 [0.6631, 0.6931] (chance 0.500)"],
        ["C3B-v2 question-form rate", "0.9471 [0.9398, 0.9542]"],
        ["C3B-v2 templating cost to the ceiling", "0.08920 → 0.06828 (−23 % items, −17 % concepts)"],
        ["C4 triangulation, distinctive renders", "median rank 91 of 1,031 (n = 18/42)"],
        ["C4 catalogue text vs rendered word", "−0.158 [−0.224, −0.089] vs +0.204 [+0.117, +0.291]"],
        ["C4 point-specificity", "+0.28593 [+0.17820, +0.39777] (prior floor at zero)"],
        ["C4B translation cost", "+0.01802 [+0.01668, +0.01940] full · +0.00672 tail"],
        ["C4C decoding: distinct renders / cost", "23 → 41 · cost cut 2.0–3.5×"],
        ["C4D example interview", "6 turns · 4 of 6 well-formed · NDCG column unquotable"],
    ], MARGIN_L, I(1.3), I(8.2), col_w=[44, 56], font_size=9.5, row_h=0.24,
        align=_num(2), first_col_bold=True)
    notes(s, "Lookup page for questions; C4B cost is relative only (declared-circular answer model)")


if __name__ == "__main__":
    import os, sys
    prs = new_deck()
    build_intro(prs)
    build_close(prs)
    build_appendix_methods(prs)
    out = sys.argv[1] if len(sys.argv) > 1 else "part_intro_test.pptx"
    prs.save(out)
    print("wrote", out, len(prs.slides._sldIdLst), "slides")
