"""Chapter 3, simplified: from the emission loop to the claims slide.

Standalone file for manual insertion; reuses the three native drawings from part_ch3.
"""
from pptx.util import Inches as I
from pptx.enum.text import PP_ALIGN
from deck_lib2 import (new_deck, slide, notes, table, caption, bullets, textbox, MARGIN_L, CONTENT_W)
from part_ch3 import _emission_chart, _gap_chart, _transcript_chart

L, R = PP_ALIGN.LEFT, PP_ALIGN.RIGHT


def build(prs):
    # 1 emission loop -------------------------------------------------------
    s, y = slide(prs, "Emission loop")
    _emission_chart(s)
    notes(s, "Chapter 2's selector picks an index into the question bank. Chapter 3's policy emits a vector in film space instead, and a small adapter on a frozen language model writes it out as a question.\n"
             "Everything downstream is unchanged: the answer folds in, the frozen tower re-ranks, the same protocol scores it.\n"
             "The two lanes are the comparison the chapter runs: emitted question against the catalogue question at the same state.")

    # 2 what was tested -----------------------------------------------------
    s, y = slide(prs, "What was tested", [
        "Does the policy emit directions that are not in the catalogue",
        "Does the vector actually control the text the language model writes",
        "Can the text come out as a well-formed question",
        "Does the written word land where the vector points",
        "Capability claims only: no claim that the questions recommend better",
    ])
    notes(s, "Four questions, each with its own control, and one scope rule.\n"
             "The scope rule is the declared circularity: answers in Chapter 3 come from a geometric simulator, so only relative comparisons are read and no NDCG value is claimed.")

    # 3 new directions -------------------------------------------------------
    s, y = slide(prs, "The policy emits new directions", [
        "One policy runs the tree's interview and emits a blend of the best questions at each node",
        "245 of 245 distinct questions preserved: no collapse to one output",
        "No emission sits on a catalogue question: 0 of 160,000 states; nearest at cosine −0.18 (random +0.06)",
        "Scope: emits where the question landscape was measured; in a withheld subtree a constant beats it",
    ], size=16)
    notes(s, "The policy is a small network on the interview state; its target at each node is the gain-weighted blend of the five best measured questions, so the target itself is off the catalogue.\n"
             "First control: does it still tell states apart? 245 of 245 distinct questions survive, snap agreement with the banked tree 0.995.\n"
             "Second: is the emission a catalogue question in disguise? At none of 160,000 validation states does it sit on the question it snaps to.\n"
             "Scope, stated plainly: it works at states where the question landscape was measured. On a withheld subtree a state-blind constant beats it by a wide margin, so no claim of extrapolation is made.")

    # 4 vector controls text ---------------------------------------------------
    s, y = slide(prs, "The vector controls the text")
    table(s, [
        ["item channel, 3,670 held-out vectors", "effect of the written text on the ranking"],
        ["true adapter", "+0.0953 [+0.0903, +0.1003]"],
        ["scrambled twin (trained on shuffled vector–text pairs)", "−0.0161"],
        ["difference", "+0.1114 [+0.1054, +0.1173]"],
        ["the true phrase itself (ceiling)", "+0.0892; adapter reaches 107 %"],
        ["picks out its own vector against a decoy", "68 % (chance 50 %)"],
    ], MARGIN_L, y + I(0.05), CONTENT_W, col_w=[58, 42], font_size=12, row_h=0.36, align=[L, R], bold_cells={(3, 1)})
    bullets(s, MARGIN_L, I(3.75), CONTENT_W, [
        "Adapter: two-layer map from the vector to 8 prefix tokens of a frozen distilgpt2",
        "Read-out: fold the written text back in and measure what it does to the ranking",
        "Same text from the scrambled twin does nothing, so the effect comes from the vector",
    ], size=13)
    notes(s, "The mechanism test. The adapter maps the vector to eight soft tokens that prefix a frozen small language model; the model writes a phrase.\n"
             "To measure whether the phrase carries the vector, fold the phrase back into the recommender and read the effect on the ranking against what the true phrase for that vector achieves.\n"
             "The control is a twin adapter trained identically on vector-text pairs with the correspondence shuffled: same architecture, same data, no relation between input and output.\n"
             "The true adapter's text reaches 107 percent of what the true phrase achieves; the twin's text does nothing. The difference is +0.111 with a tight interval.\n"
             "A second reading: shown its own vector and a decoy, the text identifies its own at 68 percent against 50 chance.")

    # 5 as a question ------------------------------------------------------------
    s, y = slide(prs, "Written as a question")
    table(s, [
        ["adapter retrained on question-shaped targets", ""],
        ["well-formed questions", "94.7 % [93.9, 95.4]"],
        ["effect still traced to the vector (true minus twin)", "+0.052 [+0.046, +0.058]"],
        ["cost of the question template on the measurable effect", "−23 %"],
    ], MARGIN_L, y + I(0.05), I(6.0), col_w=[70, 30], font_size=12, row_h=0.36, align=[L, R])
    textbox(s, I(6.6), y + I(0.05), I(3.05), I(0.3), "Examples", size=13, bold=False)
    bullets(s, I(6.6), y + I(0.4), I(3.05), [
        "How do you feel about sci-fi movies?",
        "Do you like it when a film is detective drama?",
        "Do you like it when a film is suspense?  (template failure)",
    ], size=11, height=I(1.8))
    bullets(s, MARGIN_L, I(3.6), CONTENT_W, [
        "Naturalness costs verifiability: the boilerplate carries no taste, so the measurable share of the text shrinks",
        "Failures are the template wrapped round a bare phrase, worse on blends than on single questions (2 of 6 in the example interview)",
    ], size=13)
    notes(s, "Retraining the same adapter on question-shaped targets gives 95 percent well-formed questions, and the vector still conditions them: the true-minus-twin difference stays clear of zero.\n"
             "The price is measurable: the question template is words that carry no taste, so the ceiling of what any text can do drops by 23 percent. A format change is an instrument change, so the ceiling was re-measured under the new format before reading anything.\n"
             "Failures look like the third example: the template wrapped round a bare tag with no agreement. They are rarer on single questions than on the blended directions the policy actually emits.")

    # 6 example interview --------------------------------------------------------
    s, y = slide(prs, "Example interview")
    _transcript_chart(s)
    notes(s, "Six turns down the largest branch of the tree, validation users only.\n"
             "Left, the synthesised question at each state; right, what the tree would have asked there.\n"
             "Turns 3, 5 and 6 are the point: sci-fi where the tree asks about alien films, detective drama where it asks about Crimson Tide, children's films where it asks about Star Wars VI. Same neighbourhood, a more askable object.\n"
             "Turns 2 and 4 are the template failure, shown rather than hidden.\n"
             "No NDCG is quoted for this transcript: the answers come from the simulator and the value comparison is not licensed.")

    # 7 rendered word vs catalogue question ----------------------------------------
    s, y = slide(prs, "Rendered word vs catalogue question")
    _gap_chart(s)
    bullets(s, MARGIN_L, I(3.55), CONTENT_W, [
        "The nearest catalogue question's text points away from the direction; the rendered word points along it",
        "Example: a courtroom blend (Primal Fear, A Time to Kill) renders as “indictment”, rank 5 of 1,031 tags",
        "The direction is expressible in English, and the catalogue does not express it",
    ], size=13)
    notes(s, "Closing the loop: take the emitted point, find the nearest catalogue question, and compare two texts: that question's own phrase, and the word the adapter writes for the point.\n"
             "Fold each text back in. The catalogue phrase carries the direction at minus 0.158; the rendered word at plus 0.204; the gap's interval excludes zero.\n"
             "So the nearest thing on the list is not what the point means, and the written word is.\n"
             "The example: a point built from a courtroom blend renders as 'indictment', which the direction ranks fifth of a thousand tags.\n"
             "Two caveats travel with the number: the nearest entry is often the same one, and the bridge favours concept text; the sign survives both.")

    # 8 rendering failures --------------------------------------------------------
    s, y = slide(prs, "Where the rendering goes wrong")
    table(s, [
        ["the vector is about", "the adapter writes"],
        ["The Shawshank Redemption", "Have you seen Pulp Fiction (1995)?"],
        ["Ghost Story (1981)", "Have you seen Battlestar Galactica (1980)?"],
        ["Desperate Hours (1955)", "What did you think of The Bunted (1941)?"],
    ], MARGIN_L, y + I(0.05), CONTENT_W, col_w=[40, 60], font_size=12, row_h=0.36, first_col_bold=True)
    bullets(s, MARGIN_L, I(3.0), CONTENT_W, [
        "Right era and register, wrong film: fluent and askable, incorrect as a surface form",
        "The text still reproduces the vector's effect (identification 0.61 vs 0.63 for the true question)",
        "Exact match with the true phrase 0.03 %: the model writes synonyms only a machine would accept",
        "A human plausibility panel cannot tell this from confabulation, which is why the audit is behavioural",
    ], size=13)
    notes(s, "The honest slide. For item questions the adapter writes a fluent film question about the wrong film, usually a neighbour in era and register.\n"
             "Measured behaviourally the text still carries the vector: identification 0.61 against 0.63 for the true question. Measured as words it almost never matches.\n"
             "These are machine-only synonyms: Pulp Fiction stands in for Shawshank because the two films sit together in the space, not because a person would accept the substitution.\n"
             "It is why the chapter audits by folding text back in rather than by asking people whether the question sounds plausible; a panel would pass a confabulator.")

    # 9 claims -----------------------------------------------------------------------
    s, y = slide(prs, "Chapter 3: what is claimed", [
        "Established",
        ("the policy emits off-catalogue directions where the question landscape was measured", 1),
        ("the renderer writes them as English questions, 94.7 % well-formed", 1),
        ("the text reproduces the direction's effect; text from a scrambled model does not", 1),
        ("the rendered word lands in the direction's neighbourhood; the catalogue text points away", 1),
        "Not claimed",
        ("emission at states outside the fitted tree", 1),
        ("any recommendation value: the answer model is a declared-circular simulator", 1),
    ])
    notes(s, "Four things established, each against its own control, and two things deliberately not claimed.\n"
             "The value question is left open on purpose: the answer simulator is geometric and shares the recommender's space, so an NDCG number would be circular. Only relative comparisons with both arms on the same simulator are read.")


if __name__ == "__main__":
    import sys
    prs = new_deck()
    build(prs)
    out = sys.argv[1] if len(sys.argv) > 1 else "slides_ch3_simple.pptx"
    prs.save(out)
    print("saved", out, len(prs.slides))
