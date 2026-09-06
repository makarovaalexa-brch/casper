from deck_lib2 import *
from pptx.util import Inches

prs = new_deck()
title_slide(prs, "Learning preference elicitation strategies for cold-start conversational recommendation",
            "Aleksandra Makarova", "September 2026", subtitle="CASPER: Cold-start Adaptive Selection for Preference Elicitation in Recommendation")

s, y = slide(prs, "Research goals", [
    "Cold-start user: no history, a few questions allowed",
    "Which questions to ask, and how much it matters",
    "Three chapters, one measurement stack",
    ("Chapter 1: the instrument", 1),
    ("Chapter 2: question selection", 1),
    ("Chapter 3: generated questions", 1),
])
notes(s, "Talking points: ...")

s, y = slide(prs, "Recommender comparison: canonical split")
table(s, [
    ["System", "Full NDCG@10", "Tail NDCG@10", "Interview-ready"],
    ["RecVAE", "0.4321", "0.1234", "no"],
    ["CASPER tower (i26c)", "0.4270", "0.1236", "yes"],
    ["EASE", "0.4210", "0.1100", "no"],
    ["MultVAE", "0.4100", "0.1050", "no"],
], MARGIN_L, y, Inches(7.5), col_w=[40, 20, 20, 20], font_size=12,
   align=[PP_ALIGN.LEFT, PP_ALIGN.RIGHT, PP_ALIGN.RIGHT, PP_ALIGN.CENTER])
caption(s, MARGIN_L, Inches(4.6), Inches(9), "Table 1. Fifteen systems rebuilt in-house; two shown.")

# ---- component flow chart
s, y = slide(prs, "CASPER components")
I = Inches
# user side
person(s, I(0.45), I(1.7), I(0.6))
textbox(s, I(0.25), I(2.35), I(1.0), I(0.3), "user", size=11, align=PP_ALIGN.CENTER)
bubble(s, I(1.3), I(1.45), I(1.9), I(0.55), "Do you like dark humour films?", size=10, tail="right")
bubble(s, I(1.3), I(2.35), I(1.9), I(0.45), "Yes", size=10, tail="left")
# container
container(s, I(3.4), I(1.35), I(6.25), I(3.6))
icon(s, "robot", I(3.5), I(1.4), I(0.6))
textbox(s, I(3.45), I(2.0), I(0.7), I(0.3), "CASPER", size=11, bold=True, align=PP_ALIGN.CENTER)
# boxes
b_sel = obox(s, I(4.3), I(1.5), I(1.6), I(0.6), ["question selector", "(tree / policy)"], size=10)
b_bank = obox(s, I(6.3), I(1.5), I(1.5), I(0.6), ["question bank", "18,359 items + 1,031 concepts"], size=9)
b_emit = obox(s, I(8.0), I(1.5), I(1.5), I(0.6), ["vector emitter", "+ text renderer"], size=10)
b_fold = obox(s, I(4.3), I(2.75), I(1.6), I(0.6), ["fold-in operator", "(answers → latent point)"], size=10)
b_tower = obox(s, I(6.3), I(2.75), I(1.5), I(0.6), ["frozen tower", "recommender"], size=10)
b_rank = wbox(s, I(8.0), I(2.75), I(1.5), I(0.6), ["ranked list", "NDCG@10 full / tail"], size=10)
b_state = obox(s, I(4.3), I(4.0), I(1.6), I(0.55), "interview state", size=10)
b_ruler = obox(s, I(6.3), I(4.0), I(3.2), I(0.55), ["measurement stack: one split, one metric, fixed"], size=10, dark=True)
arrow(s, I(5.9), I(1.8), I(6.3), I(1.8))
arrow(s, I(7.8), I(1.8), I(8.0), I(1.8))
arrow(s, I(5.9), I(3.05), I(6.3), I(3.05))
arrow(s, I(7.8), I(3.05), I(8.0), I(3.05))
arrow(s, I(5.1), I(3.35), I(5.1), I(4.0))
arrow(s, I(4.3), I(4.27), I(3.9), I(4.27), curved=False)
arrow(s, I(3.25), I(1.75), I(4.3), I(1.75), label=None)
arrow(s, I(3.25), I(2.6), I(4.3), I(2.95))
textbox(s, I(0.34), I(4.6), I(3), I(0.6), ["Chapter 1: tower, fold-in, ruler", "Chapter 2: selector, bank", "Chapter 3: emitter, renderer"], size=10)

prs.save("test_style.pptx")
print("ok")
