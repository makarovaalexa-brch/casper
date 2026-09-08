"""Chapter 2 (question selection) slides for the supervisor deck.

Style: STYLE.md.  Numbers: chapterB_v2.tex via build_deck.py (lines 571-1405,
2173-2409).  Exposes build_ch2(prs) and build_appendix_ch2(prs).
"""
from pptx.util import Inches, Pt
from pptx.enum.text import PP_ALIGN
from deck_lib2 import (slide, section_slide, notes, table, textbox, bullets, caption,
                       figure, obox, wbox, container, bubble, arrow, icon, person,
                       MARGIN_L, CONTENT_W, BODY_TOP, GREY)

L, C, R = PP_ALIGN.LEFT, PP_ALIGN.CENTER, PP_ALIGN.RIGHT
I = Inches


def num(n):
    return [L] + [R] * (n - 1)


def fig_cap(s, n, text, top=I(4.72)):
    caption(s, MARGIN_L, top, CONTENT_W, f"Figure 2.{n}. {text}")


def tab_cap(s, n, text, top=I(4.72)):
    caption(s, MARGIN_L, top, CONTENT_W, f"Table 2.{n}. {text}")


# ======================================================================= main
def build_ch2(prs):
    section_slide(prs, "Chapter 2: question selection",
                  "twenty-one selection rules from thirteen published methods, one ruler")

    # -- research question --------------------------------------------------
    s, y = slide(prs, "Chapter 2: research question", [
        "cold interview: which question next, and by what criterion",
        "published selectors optimise surrogates: variance, entropy, info gain, pairwise loss",
        "1 of 21 distinct rules optimises the metric the field reports",
        "12 of 23 battery rows value-free",
        "experiment: optimise the reported metric directly, measure the change",
        "diagnostic: is the shortfall the criterion, the action space, or answerability",
    ], body_box=(0.34, 1.2, 9.32, 3.9))
    notes(s, "Chapter 2 asks what a cold interview should optimise.\n"
             "Census of the literature: almost every selector uses a surrogate objective.\n"
             "Only one of twenty-one distinct rules optimises ranking quality on the consumer.\n"
             "Twelve of twenty-three rows pick a question without any value estimate.\n"
             "So the experiment: build the selector that optimises the reported metric and see what moves.\n"
             "Then diagnose where the shortfall lives.")

    # -- contract -------------------------------------------------------------
    s, y = slide(prs, "Measurement contract", [
        "consumer: certified tower i26c ep31, frozen",
        "protocol candidate pool: rated fold-in items removed",
        "behavioural answers from the user's own ratings",
        "all 10,000 TEST users, nothing sampled",
        "answer-free reference 0.1626 full / 0.0262 tail",
    ], size=16, body_box=(0.34, 1.2, 4.3, 3.6))
    figure(s, "b_contract.png", I(4.75), I(1.35), width=I(4.9))
    caption(s, I(4.75), I(3.2), I(4.9), "Figure 2.1. The measurement contract (chapterB_v2, fig:contract).")
    notes(s, "Every row in the chapter runs under one contract.\n"
             "The consumer is the Chapter 1 tower; it is never refitted.\n"
             "Answers come from the user's own recorded ratings, so no LLM answer model.\n"
             "Candidate pool removes the items already folded in, so nothing is scored on its own evidence.\n"
             "The reference with no questions asked is 0.1626 full and 0.0262 tail.")

    # -- interview loop flow chart --------------------------------------------
    s, y = slide(prs, "Interview loop")
    _interview_loop(s)
    fig_cap(s, 2, "One turn of the cold interview: select, ask, fold, re-rank, score.", top=I(4.95))
    notes(s, "One turn of the loop, from the system's side.\n"
             "The selector picks a question from the bank; the tree or a policy decides which.\n"
             "The user answers, or cannot answer; both are recorded.\n"
             "The answer is folded into the latent state through the frozen encoder.\n"
             "The recommender re-ranks the full catalogue and NDCG@10 is measured on the held-out half.\n"
             "The measured gain feeds the next selection.")

    # -- battery roster ----------------------------------------------------------
    s, y = slide(prs, "Replication battery")
    table(s, [
        ["Family", "Rules rebuilt", "n"],
        ["Optimised static (comparator)", "greedy combined · greedy items-only · greedy concepts-only", "3"],
        ["Literature heuristics", "entropy0 (Rashid 2008) · popularity (Rashid 2002) · HELF (Rashid 2008) · pure entropy (Rashid 2002)", "4"],
        ["Random controls", "random over the bank · random over the catalogue", "2"],
        ["Bandit elicitors", "GLM-UCB (Zhao 2013) · Thompson (Zhao 2013) · ConUCB (Zhang, WWW'20) · ConLinUCB-BS-static (Wang 2023) · CLiME (Liu, KDD'25)", "5"],
        ["Information-theoretic", "latent-only info-gain (Ma 2019, EDDI) · EDDI three-outcome · BALD (Houlsby 2011) · A-optimal (Anava WWW'15 / Sepliarskaia 2018)", "4"],
        ["Tree and node models", "Golbandi decision tree (WSDM'11) · IGCN (Rashid 2008)", "2"],
        ["Learned / discriminative", "DRE (Kweon, WWW'20) · DPE-Hellinger and DPE-ILD (Parapar, WSDM'21)", "3"],
        ["Ceiling", "clairvoyant oracle, reads the user's own ratings at selection time", "1"],
    ], MARGIN_L, I(1.25), CONTENT_W, col_w=[24, 70, 6], font_size=10, row_h=0.34,
        align=[L, L, C], first_col_bold=True)
    bullets(s, MARGIN_L, I(4.55), CONTENT_W, [
        "23 rows, 21 distinct rules, 13 published methods, all re-implemented on one instrument",
    ], size=14)
    tab_cap(s, 1, "The battery (chapterB_v2, tab:battery). A second table in the chapter lists what was not built.", top=I(4.95))
    notes(s, "The battery: twenty-three rows, twenty-one distinct rules, thirteen published methods.\n"
             "Every one re-implemented here, run on the same users, ruler and candidate pool.\n"
             "The optimised static orders are the comparator, not a method.\n"
             "The clairvoyant oracle reads the user's own ratings, so it is a ceiling, never a baseline.\n"
             "The chapter also prints a coverage table of what was not built: ICF, SEL-BALD, PEBOL, fMF, PERE and others.")

    # -- battery results 1 ---------------------------------------------------------
    s, y = slide(prs, "Battery results: static orders")
    rows = [["selector", "type", "k=1", "k=2", "k=4", "k=8", "k=16", "ans@16"]]
    rows += [list(r) for r in [
        ("greedy combined (item+concept)", "static", ".1808/.0537", ".1962/.0701", ".2178/.0857", ".2386/.1033", ".2590/.1237", "1.5i/9.2c"),
        ("greedy items-only", "static", ".1760/.0434", ".1884/.0563", ".2038/.0677", ".2235/.0860", ".2447/.1017", "3.11"),
        ("greedy concepts-only", "static", ".1808/.0537", ".1962/.0701", ".2141/.0858", ".2347/.1074", ".2447/.1161", "12.36"),
        ("entropy0 (Rashid 2008)", "static", ".1635/.0433", ".1706/.0528", ".1831/.0636", ".2053/.0792", ".2303/.0924", "5.52"),
        ("popularity (Rashid 2002)", "static", ".1606/.0400", ".1647/.0470", ".1746/.0579", ".1943/.0686", ".2228/.0848", "5.42"),
        ("HELF (Rashid 2008)", "static", ".1603/.0354", ".1628/.0420", ".1755/.0588", ".1892/.0725", ".2157/.0838", "3.34"),
        ("pure entropy (Rashid 2002)", "static", ".1582/.0319", ".1578/.0327", ".1580/.0323", ".1586/.0317", ".1599/.0357", "0.08"),
        ("random (bank)", "static", ".1586/.0310", ".1588/.0311", ".1588/.0316", ".1594/.0329", ".1607/.0353", "0.12"),
        ("random (catalogue)", "static", ".1585/.0310", ".1588/.0310", ".1589/.0318", ".1593/.0315", ".1612/.0341", "0.12"),
        ("GLM-UCB (Zhao 2013)", "adaptive", ".1606/.0400", ".1635/.0452", ".1756/.0578", ".1902/.0673", ".2161/.0833", "5.24"),
        ("Thompson (Zhao 2013)", "adaptive", ".1627/.0376", ".1676/.0437", ".1748/.0551", ".1908/.0677", ".2135/.0878", "3.39"),
        ("latent-only info-gain† (Ma 2019)", "adaptive", ".1537/.0336", ".1539/.0373", ".1618/.0432", ".1747/.0611", ".1917/.0749", "3.84"),
        ("EDDI three-outcome‡ (Ma 2019)", "adaptive", ".1607/.0363", ".1597/.0388", ".1732/.0497", ".1869/.0660", ".2041/.0834", "2.86"),
    ]]
    table(s, rows, MARGIN_L, I(1.22), CONTENT_W, col_w=[27, 10, 12.6, 12.6, 12.6, 12.6, 12.6, 10],
          font_size=9, row_h=0.235, align=[L, L, R, R, R, R, R, R], first_col_bold=True)
    bullets(s, MARGIN_L, I(4.62), CONTENT_W, [
        "full / tail NDCG@10, 10,000 TEST users; items-only beats best adaptive at k=16 by +0.0281 full [+0.0247, +0.0314]",
    ], size=12)
    tab_cap(s, 2, "† reduced EDDI acquisition · ‡ same, three-outcome response model · ans@16 = answers collected in 16 questions.", top=I(4.95))
    notes(s, "Full over tail NDCG@10 at five budgets, ten thousand test users.\n"
             "The three optimised static orders lead every row at every budget.\n"
             "The 2008 heuristics sit next; the bandits and info-gain arms below them.\n"
             "Items-only static beats the best adaptive selector at k=16 by 0.028 full, interval well clear of zero.\n"
             "Last column: how many answers sixteen questions actually collect. Concepts collect four times as many as items.")

    # -- battery results 2 ---------------------------------------------------------
    s, y = slide(prs, "Battery results: published methods")
    rows = [["selector", "type", "k=1", "k=2", "k=4", "k=8", "k=16", "ans@16"]]
    rows += [list(r) for r in [
        ("Golbandi tree (WSDM'11)", "adaptive", ".1557/.0351", ".1614/.0478", ".1864/.0650", ".2117/.0844", "—", "2.75"),
        ("IGCN (Rashid 2008)", "adaptive", ".1685/.0407", ".1800/.0569", ".1944/.0701", ".2031/.0747", ".2239/.0912", "4.19"),
        ("DRE (Kweon, WWW'20)", "learned static", ".1658/.0396", ".1693/.0465", ".1763/.0586", ".2074/.0789", ".2185/.0898", "4.6"),
        ("CLiME (Liu, KDD'25)", "adaptive", ".1606/.0400", ".1634/.0452", ".1776/.0549", ".1914/.0701", ".2218/.0897", "4.80"),
        ("ConLinUCB-BS-static (Wang 2023)", "static, concepts", ".1644/.0374", ".1659/.0402", ".1724/.0468", ".1829/.0598", ".1891/.0700", "0.00"),
        ("ConUCB (Zhang, WWW'20)", "adaptive", ".1584/.0346", ".1591/.0364", ".1651/.0447", ".1805/.0574", ".1874/.0699", "1.80"),
        ("DPE-Hellinger (Parapar, WSDM'21)", "adaptive", ".1627/.0398", ".1680/.0480", ".1817/.0610", ".2002/.0765", ".2192/.0925", "3.9"),
        ("DPE-ILD (Parapar, WSDM'21)", "adaptive", ".1627/.0398", ".1621/.0401", ".1613/.0397", ".1609/.0390", ".1603/.0375", "0.32"),
        ("BALD (Houlsby 2011) §", "value-free", ".1754/.0448", ".1853/.0473", ".1934/.0572", ".2057/.0683", ".2203/.0824", "2.80"),
        ("A-optimal (Anava WWW'15) §", "value-free", ".1693/.0498", ".1724/.0523", ".1702/.0549", ".1694/.0544", ".1740/.0563", "1.81"),
        ("clairvoyant oracle (ceiling)", "not a baseline", ".1893/.0680", ".2194/.0965", ".2544/.1281", ".2930/.1597", ".3292/.1927", "12.9"),
    ]]
    table(s, rows, MARGIN_L, I(1.22), CONTENT_W, col_w=[27, 12, 12.2, 12.2, 12.2, 12.2, 12.2, 9],
          font_size=9, row_h=0.235, align=[L, L, R, R, R, R, R, R], first_col_bold=True)
    bullets(s, MARGIN_L, I(4.22), CONTENT_W, [
        "each method reproduced on its own protocol first, then ported to our ruler",
        "none reaches the optimised static order; Golbandi has no k=16 row (leaf fragmentation caps depth)",
    ], size=12)
    tab_cap(s, 3, "§ = value-free rule (chapterB_v2, tab:battery).", top=I(4.95))
    notes(s, "The second half of the battery: the tree, node and learned methods.\n"
             "Each was first validated on its own published protocol, then ported onto our ruler.\n"
             "None reaches the optimised static order at any budget.\n"
             "The clairvoyant oracle shows the headroom: 0.329 full at sixteen questions against 0.259 for the best real arm.\n"
             "Golbandi's tree stops at eight questions because leaf fragmentation caps its depth.")

    # -- separation ---------------------------------------------------------------------
    s, y = slide(prs, "Battery results: separation")
    table(s, [
        ["contrast (paired bootstrap, 10,000 users, 10,000 resamples)", "result"],
        ["optimised static vs every adaptive arm: 13 adaptive rows + three-outcome variant, 138 cells",
         "combined and concepts-only orders lead and separate in all 138"],
        ["items-only order", "leads 136, separates 134; 4 exceptions are tail cells separating in neither direction"],
        ["DRE (WWW'20) vs the 2008 entropy heuristic, full axis", "−0.0118 [−0.0153, −0.0083]"],
    ], MARGIN_L, I(1.25), CONTENT_W, col_w=[52, 48], font_size=11, row_h=0.42)
    bullets(s, MARGIN_L, I(3.25), CONTENT_W, [
        "a fixed question order leads every adaptive selector at every budget",
        "the learned selector loses to a 2008 heuristic",
        "same ordering on RecVAE, TaNP and EDDI's own Partial-VAE",
    ], size=15)
    notes(s, "Are the differences real? Paired bootstrap over the same ten thousand users.\n"
             "One hundred and thirty-eight cells: the combined and concepts-only static orders lead and separate in all of them.\n"
             "Items-only leads in 136 and separates in 134.\n"
             "DRE, the learned selector from WWW 2020, loses to the 2008 entropy heuristic.\n"
             "The ordering repeats on three other consumers.")

    # -- budget curves ---------------------------------------------------------------------
    s, y = slide(prs, "NDCG against budget", [
        "three static orders are the top three lines at every budget",
        "no adaptive arm crosses them anywhere",
        "gap to the oracle does not close: 0.3292 vs 0.2590 at K=16",
        "ordering stable in K; Golbandi's tree the one line changing rank",
    ], size=14, body_box=(5.5, 1.2, 4.2, 3.6))
    figure(s, "b_budget_curves.png", MARGIN_L, I(1.25), width=I(5.05))
    fig_cap(s, 3, "Full NDCG@10 against question budget, 10,000 TEST users (chapterB_v2, fig:budgetcurves).", top=I(4.4))
    notes(s, "The battery as curves, logarithmic budget axis.\n"
             "Black lines are the three optimised static orders; colours are the literature; dashed grey is the oracle ceiling.\n"
             "The static orders are the top three lines at every budget.\n"
             "The gap to the oracle stays open: about seventy percent of available headroom is unclaimed at sixteen questions.\n"
             "The ordering does not move with budget.")

    # -- adaptive vs static on test ------------------------------------------------------------
    s, y = slide(prs, "Adaptive vs static: test users")
    rows = [["selector (10,000 TEST users)", "k=1", "k=2", "k=4", "k=8", "k=16", "ans@16"]]
    rows += [list(r) for r in [
        ("static, entropy0", ".1635/.0433", ".1706/.0528", ".1831/.0636", ".2053/.0792", ".2303/.0924", "5.52"),
        ("GLM-UCB", ".1606/.0400", ".1635/.0452", ".1756/.0578", ".1902/.0673", ".2161/.0833", "5.24"),
        ("Thompson", ".1627/.0376", ".1676/.0437", ".1748/.0551", ".1908/.0677", ".2135/.0878", "3.39"),
        ("latent-only info-gain†", ".1537/.0336", ".1539/.0373", ".1618/.0432", ".1747/.0611", ".1917/.0749", "3.84"),
    ]]
    table(s, rows, MARGIN_L, I(1.22), CONTENT_W, col_w=[26, 13, 13, 13, 13, 13, 9],
          font_size=10, row_h=0.27, align=[L, R, R, R, R, R, R], first_col_bold=True)
    table(s, [
        ["full-axis deficit against the static order, k=1 → k=16", "range"],
        ["GLM-UCB", "−0.0029 → −0.0137"],
        ["Thompson", "−0.0022 → −0.0177"],
        ["latent-only info-gain†", "−0.0098 → −0.0386"],
        ["comparisons separating (paired bootstrap)", "30 / 30"],
    ], MARGIN_L, I(2.75), I(5.4), col_w=[68, 32], font_size=10, row_h=0.26, align=num(2), first_col_bold=True)
    bullets(s, I(6.0), I(2.75), I(3.66), [
        "ordering repeats on RecVAE, TaNP, Partial-VAE",
        "EDDI selecting and consuming: .1493/.0488 vs static .1769/.0480 at k=16",
        "fifth consumer (Golbandi leaf model): Thompson beats static, +0.0100 to +0.0281 full",
        "verdict is a property of the strategy and instrument pair",
    ], size=11)
    tab_cap(s, 4, "Section 1 audit (chapterB_v2, tab:act1).", top=I(4.75))
    notes(s, "The audit proper: three adaptive selectors against one static order, on the test users.\n"
             "All thirty comparisons separate and all thirty run the same way.\n"
             "The result holds on three other consumers, including the self-consistent EDDI cell.\n"
             "But on a fifth consumer, a Golbandi leaf model, Thompson beats static at every budget.\n"
             "So the verdict belongs to the strategy and the instrument together, which is why Chapter 1 certifies the instrument.")

    # -- split criteria --------------------------------------------------------------------------
    s, y = slide(prs, "Split criteria: measured gain")
    table(s, [
        ["split criterion", "mean ρ with measured gain", "nodes ρ>0", "argmax hit", "top-10 recall", "regret"],
        ["variance (Golbandi 2011)", "+0.010", "13 / 22", "1 / 22", "0.03", "+0.01664"],
        ["pairwise rank loss (Hu & Yu 2013)", "−0.060", "10 / 22", "0 / 22", "0.01", "+0.01738"],
        ["logistic loss, as run (Hu & Yu)", "−0.104", "9 / 22", "1 / 22", "0.03", "+0.01282"],
        ["control: uniform random pick", "—", "—", "—", "0.10", "+0.01955"],
    ], MARGIN_L, I(1.25), CONTENT_W, col_w=[32, 20, 12, 12, 12, 12], font_size=11, row_h=0.3,
        align=num(6), first_col_bold=True)
    bullets(s, MARGIN_L, I(3.15), CONTENT_W, [
        "true NDCG gain of every candidate measured at all 22 nodes of the grown tree",
        "two of three criteria negatively correlated with the quantity they proxy",
        "criteria forgo 66–89 % of what a random pick forgoes; top-10 recall below random",
        "root: logistic argmax = metric argmax; variance ranks it 250th of 1,000, pairwise 75th",
        "pairwise loss measured exactly tracks NDCG at Spearman +0.784; the Taylor approximation gives −0.149",
    ], size=13)
    tab_cap(s, 5, "Criterion against measured gain at 22 nodes (chapterB_v2, tab:criterionregret).", top=I(4.9))
    notes(s, "Diagnosis, part one: do the published split criteria track measured gain?\n"
             "At every node of the grown tree we measured the true NDCG gain of every candidate, then recorded each criterion's pick.\n"
             "Two of the three have a negative rank correlation with the thing they stand in for.\n"
             "Their top-ten recall is below what a random pick achieves.\n"
             "The pairwise loss is fine when computed exactly; it is the Taylor approximation that fails.")

    # -- dispersion --------------------------------------------------------------------------------
    s, y = slide(prs, "Question dispersion")
    table(s, [
        ["population state", "n", "distinct best question", "effective # questions"],
        ["ML-25M cold, no questions asked", "10,000", "1", "1.00"],
        ["after one standard opener", "10,000", "5", "2.88"],
        ["after two standard openers", "10,000", "9", "5.04"],
        ["real free-text request, placed as a state", "4,431", "442", "95.13"],
        ["   requests naming at least one film", "2,639", "413", "128.46"],
        ["   text-only requests, no film named", "1,792", "171", "38.23"],
    ], MARGIN_L, I(1.25), I(6.2), col_w=[46, 14, 22, 18], font_size=10.5, row_h=0.28,
        align=num(4), first_col_bold=True)
    table(s, [
        ["state", "best question", "gain"],
        ["cold, after two openers", "a concept", "+0.0283"],
        ["request-placed", "an item", "+0.0138"],
        ["request-placed", "best concept", "+0.0076"],
    ], I(6.8), I(1.25), I(2.86), col_w=[42, 34, 24], font_size=10, row_h=0.28, align=num(3))
    bullets(s, MARGIN_L, I(3.5), CONTENT_W, [
        "cold users have about one best next question; little to adapt to",
        "after two openers: 112 distinct states, 41 % of users share a single argmax",
        "a volunteered request is worth ~95 effective questions of dispersion",
        "dispersion and headroom differ: placed users are distinguishable but have less left to gain",
    ], size=13)
    tab_cap(s, 6, "Left: dispersion (chapterB_v2, tab:dispersion). Right: best available question per state.", top=I(4.9))
    notes(s, "Diagnosis, part two: how many different best questions does a cold population contain?\n"
             "With no questions asked, one. After two openers, five effective questions.\n"
             "A real volunteered request placed as a state gives ninety-five.\n"
             "So an adaptive selector at cold start has almost nothing to be adaptive about.\n"
             "But headroom runs the other way: a placed user has less left to gain.")

    # -- NDCG tree -----------------------------------------------------------------------------------
    s, y = slide(prs, "Tree split on measured NDCG")
    table(s, [
        ["candidates per node", "1,000 = 500 popularity-ranked items + 500 answerability-ranked concepts"],
        ["objective", "node-size-weighted mean of the children's cohort-mean held-out full NDCG@10"],
        ["growth", "fitted on 10,000 VALIDATION users, then frozen"],
        ["shape", "depth 5, 15 leaves, 22 nodes; MIN_LEAF = 200 the only stopping rule"],
        ["budget", "16 charged questions; leak assertion 0 of 10,000 on both cohorts"],
        ["root splitter", "a concept, answered by 74.2 %, splitting 2,606 / 4,814 / 2,580"],
    ], MARGIN_L, I(1.25), I(4.6), col_w=[30, 70], font_size=9.5, row_h=0.36, first_col_bold=True, header_rows=0)
    table(s, [
        ["K", "full gain over static [95 % CI]", "tail gain over static [95 % CI]"],
        ["1", "0.00000 (exact, by nesting)", "0.00000 (exact)"],
        ["2", "+0.00528 [+0.00324, +0.00725]", "−0.00262 [−0.00481, −0.00051]"],
        ["4", "+0.00891 [+0.00632, +0.01147]", "+0.00482 [+0.00214, +0.00756]"],
        ["8", "+0.01062 [+0.00775, +0.01354]", "+0.00835 [+0.00543, +0.01130]"],
        ["16", "+0.01035 [+0.00739, +0.01323]", "+0.00705 [+0.00394, +0.01012]"],
    ], I(5.15), I(1.25), I(4.51), col_w=[8, 46, 46], font_size=9.5, row_h=0.3,
        align=[C, R, R], first_col_bold=True)
    bullets(s, MARGIN_L, I(3.85), CONTENT_W, [
        "same learner, same candidates, same stopping rule; only the split criterion changes",
        "10,000 TEST users: 7 of 8 K≥2 intervals favour the tree; tail at K=2 goes against it",
        "absolutes at K=16: tree 0.26936 / 0.13074, static 0.25900 / 0.12370",
        "validation gain +0.02794 full became +0.01035 on TEST (2.70× optimism)",
    ], size=12, height=I(1.0))
    tab_cap(s, 7, "Left: construction. Right: gain over the static order on TEST (chapterB_v2, tab:tree).", top=I(4.95))
    notes(s, "The demonstration: change one variable, the split criterion, to the measured metric.\n"
             "Same tree learner, same one thousand candidates per node, same stopping rule.\n"
             "On ten thousand held-out test users the tree beats the static order from two questions onward.\n"
             "At sixteen questions the gain is 0.0104 full, interval 0.0074 to 0.0132.\n"
             "One interval goes the other way, tail at K=2; it is printed.\n"
             "Validation overstated the gain by 2.7 times, which is why the test look matters.")

    # -- ablation ---------------------------------------------------------------------------------
    s, y = slide(prs, "Split criteria: ablation")
    table(s, [
        ["split criterion (K=16, 10,000 TEST users)", "full@10", "gain vs static", "tail@10", "gain vs static"],
        ["measured NDCG@10 (ours)", "0.26936", "+0.01035", "0.13074", "+0.00705"],
        ["pairwise rank loss (Hu & Yu 2013)", "0.22859", "−0.03041", "0.10031", "−0.02338"],
        ["rating variance (Golbandi 2011)", "0.22840", "−0.03061", "0.10124", "−0.02246"],
        ["logistic loss, as run (Hu & Yu 2013)", "0.17274", "−0.08627", "0.06817", "−0.05552"],
        ["shuffle, min-leaf-respecting", "—", "−0.04817", "—", "−0.03412"],
        ["shuffle, unrestricted", "—", "−0.03934", "—", "−0.02126"],
    ], MARGIN_L, I(1.25), CONTENT_W, col_w=[36, 16, 16, 16, 16], font_size=10.5, row_h=0.28,
        align=num(5), first_col_bold=True)
    bullets(s, MARGIN_L, I(3.45), CONTENT_W, [
        "measured far above pairwise ≈ variance; pairwise vs variance the only null, +0.00020 [−0.00331, +0.00371]",
        "all 56 non-nested intervals exclude zero; every criterion splits the root on a concept",
        "min-leaf shuffle deeper (7 vs 5), wider (27 leaves), more varied (286 questions), still loses by 0.048",
        "cost: Taylor scores 1,000 candidates in ~11 s; exact objective 37,191 s per tree",
    ], size=12)
    tab_cap(s, 8, "Six-criterion ablation (chapterB_v2, tab:splitcrit).", top=I(4.95))
    notes(s, "Six split criteria, everything else identical.\n"
             "Measured NDCG is far ahead; pairwise and variance tie; the logistic loss as published collapses.\n"
             "The shuffle control grows a deeper, wider, more varied tree and still loses by 0.048, so structure is not what helps.\n"
             "The reason surrogates exist: the exact objective costs three orders of magnitude more per node.\n"
             "The ablation reproduces every published tree number to under one in a million.")

    # -- bank vs conditioning --------------------------------------------------------------------
    s, y = slide(prs, "Question bank vs conditioning")
    table(s, [
        ["change made (10,000 TEST users)", "full gain [95 % CI]", "tail gain", "share of static's lift"],
        ["bank: items → items + concepts, equal budget", "+0.01430 [+0.01124, +0.01739]", "+0.02199", "16.6 %"],
        ["conditioning: NDCG tree, combined pool, K=16", "+0.01035 [+0.00739, +0.01323]", "+0.00705", "10.3 %"],
        ["conditioning: genre-conditioned greedy, item pool, K=15", "+0.00862 [+0.00595, +0.01131]", "+0.00591", "10.2 %"],
    ], MARGIN_L, I(1.25), CONTENT_W, col_w=[38, 26, 15, 21], font_size=10.5, row_h=0.36,
        align=num(4), first_col_bold=True)
    bullets(s, MARGIN_L, I(3.05), CONTENT_W, [
        "both changes real, both intervals exclude zero",
        "on tail, changing the bank is worth ~3× conditioning: +0.02199 vs +0.00705",
        "enlarging the bank further made the selectors that consume it worse",
        "the gain is answerability: concepts less informative per answer, but answered",
        "for scale, 8 seed items from the user's own history: +0.0206 full / +0.0353 tail",
    ], size=13)
    tab_cap(s, 9, "Three changes to a cold interview, same users, same comparator (chapterB_v2, tab:bankdecomp).", top=I(4.95))
    notes(s, "Three changes to the same cold interview on the same test users.\n"
             "Changing the question bank from items to items plus concepts is worth more than either form of conditioning.\n"
             "On the tail the ratio is about three to one.\n"
             "Both changes are real; the ordering is just the opposite of where the field's effort goes.\n"
             "For scale: eight items from the user's own history dwarf every interview-design choice on the slide.")

    # -- answerability -----------------------------------------------------------------------------
    s, y = slide(prs, "Answerability")
    table(s, [
        ["measurement", "what it shows"],
        ["composition of the optimised static order", "c c i c c i i c c c c i i i c c: concepts hold 10 of 16 slots and both of the first two"],
        ["per-state criterion", "cold states: best concept +0.0238 vs best item +0.0019; request-placed: item +0.0138 vs concept +0.0076"],
        ["the tree's own root", "chosen concept answered by 74.2 %; best item split sends 70 % into did-not-answer"],
        ["mechanism", "16 item questions collect 3.11 answers; 16 concept questions collect 12.36"],
    ], MARGIN_L, I(1.25), CONTENT_W, col_w=[28, 72], font_size=10.5, row_h=0.38, first_col_bold=True)
    table(s, [
        ["contrast (K=16, TEST)", "full", "tail"],
        ["concepts − items", "−0.00003 [−.00315, +.00310]", "+0.01436 [+.01093, +.01776]"],
        ["combined − items", "+0.01430", "+0.02199"],
        ["combined − concepts", "+0.01434", "+0.00763"],
    ], MARGIN_L, I(3.35), I(5.6), col_w=[30, 36, 34], font_size=10, row_h=0.26, align=num(3))
    bullets(s, I(6.2), I(3.35), I(3.46), [
        "answering at all worth ~+0.25; largest question-choice effect +0.048",
        "1,785 users answer neither opener; ~14 % of cohort untouched at 0.087",
    ], size=11)
    tab_cap(s, 10, "Three independent measurements of the channel ordering (chapterB_v2, sections 4.1–4.3).", top=I(4.8))
    notes(s, "Which channel wins is decided by who can answer, not by what is informative.\n"
             "Three independent measurements agree: the static order's composition, the per-state criterion, and the tree's own root.\n"
             "Sixteen concept questions collect four times the answers of sixteen item questions.\n"
             "Concepts and items tie on full at sixteen questions; concepts win on tail and at every smaller budget.\n"
             "Answering at all is worth about 0.25; the biggest question-choice effect anywhere is 0.048.")

    # -- reddit corpus ----------------------------------------------------------------------------------
    s, y = slide(prs, "Unprompted requests: corpus")
    table(s, [
        ["size", "4,431 threads, each an unprompted free-text request with ground-truth items"],
        ["genuine requests", "94.0 %"],
        ["median slots per request", "5; 93.6 % carry ≥2 slots"],
        ["named films", "7,572 across 2,793 threads, 5,868 distinct titles"],
        ["vocabulary coverage", "100.0 % of named films inside the 18,359-item vocabulary"],
        ["foldable as a state", "3,078 of 4,431 (69.5 %)"],
        ["entity linking", "0.960 precision on 500 hand-labelled mentions (gate ≥0.95)"],
    ], MARGIN_L, I(1.25), I(5.6), col_w=[34, 66], font_size=10, row_h=0.3, first_col_bold=True, header_rows=0)
    table(s, [
        ["slot", "mentions", "opens with"],
        ["positive anchor (names a film)", "63.0 %", "63.0 %"],
        ["genre", "39.8 %", "14.7 %"],
        ["constraint", "30.7 %", "5.2 %"],
        ["tone", "21.4 %", "3.3 %"],
        ["negative anchor", "16.5 %", "—"],
        ["era", "9.8 %", "1.9 %"],
        ["negative genre", "4.2 %", "—"],
        ["none of the five", "11.9 %", "—"],
    ], I(6.2), I(1.25), I(3.46), col_w=[52, 24, 24], font_size=9.5, row_h=0.26, align=num(3))
    bullets(s, MARGIN_L, I(3.85), CONTENT_W, [
        "what users volunteer without being asked",
        "Horror 27.5 % of genre mentions, Thriller 15.8 %; in ML-25M Horror is the favourite of 51 users in 10,000, Drama of 5,481",
        "request distribution inverted relative to consumption; a watch-log prior is the wrong prior for what gets asked",
    ], size=12)
    tab_cap(s, 11, "Left: the corpus. Right: slot census (chapterB_v2, section 5).", top=I(4.95))
    notes(s, "Everything so far assumes a closed bank and a user who waits to be asked.\n"
             "This section measures what people volunteer: 4,431 real requests with ground-truth items.\n"
             "Nearly two thirds open by naming a film; a third name a constraint; a fifth a tone.\n"
             "Every named film is inside our vocabulary, and seventy percent of requests fold into a state.\n"
             "The genres people ask for are inverted relative to what they watch: Horror dominates requests and is rare in the logs.")

    # -- reddit examples ----------------------------------------------------------------------------------
    s, y = slide(prs, "Unprompted requests: examples")
    table(s, [
        ["the request as posted", "slots found"],
        ["“[Request] I'm looking for thrillers like 10 Cloverfield Lane (I really loved that film).”", "film + genre"],
        ["“[REQUEST] good horror/suspense movie on Netflix. i can't find anything good please help”", "genre + constraint"],
        ["“Films like; 12 Angry Men. Whether it's films told on one set, or films heavy dialogue. Cheers!”", "film + 2 constraints"],
        ["“Looking for a Horror movie with a similar tone to Evil Dead 2… a super gory horror movie with a slightly tongue-and-cheek tone.”", "film + genre + 2 tone"],
        ["“[Request] I need something that's gonna punch me in the face. Something with unexpected plot twist…”", "2 tone, no film"],
    ], MARGIN_L, I(1.25), CONTENT_W, col_w=[74, 26], font_size=12, row_h=0.5)
    bullets(s, MARGIN_L, I(4.3), CONTENT_W, [
        "five requests verbatim; the last names no film, the text-only case",
    ], size=14)
    tab_cap(s, 12, "Five requests from the corpus, original spelling.", top=I(4.8))
    notes(s, "Five requests as posted.\n"
             "Most anchor on a film and add a genre or a constraint.\n"
             "The last one names no film at all: two tone words and nothing else. That is the text-only case the closed bank cannot serve.\n"
             "NOTE TO AUTHOR: quoted verbatim from data/reddit_he/slots/full_input.jsonl (lines 16, 102, 126, 157, 65), original spelling and [Request] tags; two shortened at an ellipsis. Confirm you are content to show public forum text verbatim; paraphrasing costs the point that they are real.")

    # -- opener-fitted --------------------------------------------------------------------------------------
    s, y = slide(prs, "Opener-fitted interviews")
    table(s, [
        ["group", "n", "static", "cold", "opener", "opener − cold"],
        ["art house", "464", ".27251", ".27964", ".30046", "+.02082 *"],
        ["inspiring", "1,526", ".24274", ".24338", ".25534", "+.01196 *"],
        ["dialogue driven", "395", ".21750", ".21894", ".22940", "+.01045"],
        ["gore", "391", ".32950", ".32595", ".33208", "+.00613"],
        ["marvel", "599", ".40369", ".41933", ".42534", "+.00601"],
        ["teenagers", "502", ".24170", ".23712", ".24179", "+.00466"],
        ["pooled", "3,877", ".27721", ".27993", ".29035", "+.01041 *"],
    ], MARGIN_L, I(1.25), I(5.4), col_w=[26, 11, 15, 15, 15, 18], font_size=10, row_h=0.27,
        align=num(6), first_col_bold=True)
    bullets(s, I(6.0), I(1.25), I(3.66), [
        "Reserved TEST users, K=15, full NDCG@10",
        "each tree fitted at the state the user is in",
        "cold − static = +.00272 (interval includes zero)",
        "opener − cold = +.01041 *",
        "* = paired bootstrap interval excludes zero",
    ], size=12)
    figure(s, "b_act5_pipeline.png", MARGIN_L, I(3.55), width=I(5.2))
    fig_cap(s, 4, "How a synthesised opener state is built and what is run from it (chapterB_v2, fig:act5pipeline).", top=I(4.95))
    notes(s, "Section 5 proper: fit the interview tree at the state the user is actually in.\n"
             "Six request groups mined from the corpus; a synthesised opener puts each MovieLens user into a group state.\n"
             "A tree fitted at the cold state already beats the anchor teacher; fitting at the opener state beats that again.\n"
             "Pooled gain 0.0104 on test, interval excludes zero; validation gave 0.0069 with a different per-group ranking, so only the pooled figure is claimed.\n"
             "The pipeline figure shows the synthesis and the four arms run from the folded state.")

    # -- distilled policy -----------------------------------------------------------------------------------
    s, y = slide(prs, "Distilled policy: test results")
    table(s, [
        ["population (reserved TEST, K=15)", "n", "policy", "cold", "policy − cold [95 % CI]"],
        ["all users", "10,000", ".26842", ".26499", "+.00343 [+.00173, +.00509] *"],
        ["volunteered something", "3,877", ".28939", ".27993", "+.00946 [+.00524, +.01376] *"],
        ["volunteered nothing", "6,123", ".25514", ".25553", "−.00038 [−.00098, +.00012]"],
    ], MARGIN_L, I(1.25), CONTENT_W, col_w=[34, 12, 14, 14, 26], font_size=10.5, row_h=0.32,
        align=num(5), first_col_bold=True)
    bullets(s, MARGIN_L, I(2.85), CONTENT_W, [
        "six trees distilled into one policy, 526,542 parameters",
        "top-1 agreement 0.991 with the interviews it imitates; routing .741–.957",
        "within ±.004 of the correct group's own interview in every group",
        "value does not track routing: marvel routes best, gains least",
        "single reserved test look, spent once; non-monotone in budget: K=1 +0.00161 *, K=4 +0.00119, K=15 +0.00343 *",
        "do-no-harm cell on users who volunteered nothing is a clean null",
    ], size=12.5)
    tab_cap(s, 13, "One distilled policy on the reserved test cohort (chapterB_v2, tab:policy).", top=I(4.95))
    notes(s, "The six trees are distilled into a single policy that has to find the group itself.\n"
             "On the reserved test users it beats the cold tree by 0.0034 overall and 0.0095 on users who volunteered something.\n"
             "On users who volunteered nothing it does no harm: a clean null.\n"
             "Distillation buys no accuracy; the value is one object holding the whole forest.\n"
             "Quote the pooled contrast, not a group: the per-group ranking moves between validation and test.")

    # -- routing space ---------------------------------------------------------------------------------------
    s, y = slide(prs, "Routing space")
    figure(s, "b_route_atlas.png", MARGIN_L, I(1.22), width=I(6.9))
    bullets(s, I(7.4), I(1.25), I(2.26), [
        "left: full profile, groups not separable",
        "right: opener state, discriminant on a disjoint half assigns 0.907 held-out (chance 0.167)",
        "scramble control collapses routing on every group, e.g. .958 → .094",
        "claim: adaptivity beyond the closed tree, not routing",
    ], size=10, height=I(3.2))
    fig_cap(s, 5, "The same 3,980 users in two spaces (chapterB_v2, fig:atlas).", top=I(4.75))
    notes(s, "What the router reads. Left, the full rating profile: taste is a continuum and the six groups are not separable.\n"
             "Right, the folded opener state: a discriminant fitted on a disjoint half assigns ninety percent of held-out users correctly.\n"
             "Scrambling the routing collapses it on every group and turns the value negative on marvel and inspiring.\n"
             "Stated limit: the synthesised openers are group-stamped, so routing is partly circular; the opener-minus-cold contrast survives because the state is held fixed.\n"
             "The claim is adaptivity beyond the closed tree, not routing.")

    # -- example paths -----------------------------------------------------------------------------------
    s, y = slide(prs, "Interview tree: example paths")
    _paths(s)
    notes(s, "What the group interviews actually ask, root to leaf.\n"
             "Art house: black and white, fight scenes, disturbing. Teenagers: fantasy world, then Harry Potter or the eighties.\n"
             "The first two read as thematic; the gore path does not. The chapter says so: the tree optimises separation, not meaning.\n"
             "For contrast, the genre-conditioned static orders ask everyone in a group the same question: Romance users about Speed.\n"
             "NOTE TO AUTHOR: paths are from M3_FOREST_TREES.md, the anchor-teacher forest, whereas the opener-fitted table is a different forest (its teenagers root is the item Harry Potter and the Deathly Hallows: Part 1). Cohort sizes 587 / 786 / 747 are the M3 cohorts, not the validation ns 473 / 540 / 419. Say if you want the paths regenerated from the opener-fitted trees.")

    # -- status -----------------------------------------------------------------------------------------
    s, y = slide(prs, "Chapter 2: status")
    table(s, [
        ["state", "closed 30 August 2026; test look spent 12 and 29 August, replicates"],
        ["frozen", "audit, diagnosis, demonstration, answerability on TEST"],
        ["target", "ECIR 2027 full paper, deadline 2 October 2026; sections 1–3 at ~12 pp"],
        ["second choice", "RecSys, or its reproducibility track"],
        ["unprompted requests", "held back as a second submission (RecSys / CIKM)"],
        ["open objection", "demonstration rests on one consumer; fixable by scoring the frozen tree through RecVAE"],
    ], MARGIN_L, I(1.25), CONTENT_W, col_w=[24, 76], font_size=11, row_h=0.34, first_col_bold=True, header_rows=0)
    bullets(s, MARGIN_L, I(3.85), CONTENT_W, [
        "claims not made: routing to a group interview (Golbandi WSDM'11, CBB SIGIR'21 do it); beating k-means; parity or tie",
        "finding: published adaptive selectors optimise the wrong objective; conditioning on the right one beats the static order",
    ], size=12.5, height=I(1.1))
    notes(s, "Chapter 2 is closed and the test look is spent.\n"
             "Target is ECIR 2027, deadline 2 October, with the first three sections compressed to twelve pages.\n"
             "The unprompted-request section is held back as a second paper rather than cut.\n"
             "The strongest open objection: one consumer. Scoring the frozen tree through RecVAE is one machine-night.\n"
             "The one-line finding: published adaptive selectors optimise the wrong objective; conditioning on the right one wins.")


# ------------------------------------------------------------------ charts --
def _interview_loop(s):
    # user
    person(s, I(0.55), I(2.15), I(0.6))
    textbox(s, I(0.3), I(2.85), I(1.1), I(0.3), "user", size=11, align=C)
    bubble(s, I(1.45), I(1.35), I(2.1), I(0.6), "Do you like better than expected films?", size=10, tail="right")
    bubble(s, I(1.45), I(2.85), I(1.3), I(0.45), "Yes", size=10, tail="left")
    # container
    container(s, I(3.75), I(1.3), I(5.9), I(3.55))
    icon(s, "robot", I(3.85), I(1.4), I(0.55))
    textbox(s, I(3.75), I(1.97), I(0.75), I(0.3), "CASPER", size=10, bold=True, align=C)
    b_tree = obox(s, I(4.7), I(1.45), I(1.55), I(0.62), ["interview tree", "or policy"], size=10)
    b_bank = obox(s, I(6.55), I(1.45), I(1.35), I(0.62), ["question bank", "items + concepts"], size=10)
    b_q = wbox(s, I(8.15), I(1.45), I(1.35), I(0.62), ["question", "asked"], size=10)
    b_ans = wbox(s, I(8.15), I(2.6), I(1.35), I(0.62), ["answer or", "cannot answer"], size=10)
    b_fold = obox(s, I(6.55), I(2.6), I(1.35), I(0.62), ["fold-in", "(frozen encoder)"], size=10)
    b_state = obox(s, I(4.7), I(2.6), I(1.55), I(0.62), ["interview state", "(latent point)"], size=10)
    b_rec = obox(s, I(4.7), I(3.75), I(1.55), I(0.62), ["frozen tower", "re-ranks catalogue"], size=10)
    b_ndcg = obox(s, I(6.55), I(3.75), I(2.95), I(0.62), ["NDCG@10 full / tail", "on the held-out half"], size=10, dark=True)
    # arrows
    arrow(s, I(6.25), I(1.76), I(6.55), I(1.76))
    arrow(s, I(7.9), I(1.76), I(8.15), I(1.76))
    arrow(s, I(8.82), I(2.07), I(8.82), I(2.6))
    arrow(s, I(8.15), I(2.91), I(7.9), I(2.91))
    arrow(s, I(6.55), I(2.91), I(6.25), I(2.91))
    arrow(s, I(5.47), I(3.22), I(5.47), I(3.75))
    arrow(s, I(6.25), I(4.06), I(6.55), I(4.06))
    arrow(s, I(5.47), I(2.6), I(5.47), I(2.07), label="next turn", label_dx=I(0.95), label_dy=I(-0.12))
    arrow(s, I(3.55), I(1.6), I(4.7), I(1.6))
    arrow(s, I(2.75), I(3.05), I(4.7), I(2.95))


def _paths(s):
    tb = textbox(s, MARGIN_L, I(1.22), I(5.2), I(3.5), [
        "art house (n = 587, depth 2)",
        ("“black and white” → no → “fight scenes” → no → “disturbing”", 1),
        "teenagers (n = 786, depth 3)",
        ("“fantasy world” → yes → Harry Potter: Deathly Hallows Pt 2; no → “80s”", 1),
        "gore (n = 747, depth 3)",
        ("“shallow” → yes → Batman Begins; no → “fun movie” → no → “based on a true story”", 1),
    ], size=13)
    # a small tree sketch for the teenagers path
    x0 = I(6.0)
    r = obox(s, x0 + I(0.85), I(1.3), I(1.6), I(0.5), "“fantasy world”", size=10)
    a = obox(s, x0, I(2.45), I(1.55), I(0.62), ["Harry Potter:", "Deathly Hallows Pt 2"], size=9)
    b = obox(s, x0 + I(1.85), I(2.45), I(1.45), I(0.5), "“80s”", size=10)
    arrow(s, x0 + I(1.35), I(1.8), x0 + I(0.8), I(2.45), label="yes", label_dx=I(-0.35), label_dy=I(-0.35))
    arrow(s, x0 + I(2.0), I(1.8), x0 + I(2.55), I(2.45), label="no", label_dx=I(0.4), label_dy=I(-0.35))
    caption(s, x0, I(3.25), I(3.4), "Figure 2.6. The teenagers path as a tree.")
    bullets(s, MARGIN_L, I(3.75), CONTENT_W, [
        "first two paths thematic; the third is not: the tree optimises separation, not meaning",
        "genre-static contrast: Drama users asked about Clear and Present Danger first, Comedy about Return of the Jedi, Romance about Speed",
    ], size=12.5)


# =================================================================== appendix
def build_appendix_ch2(prs):
    # -- 2.1 -----------------------------------------------------------------------
    s, y = slide(prs, "Appendix 2.1: adaptive branching")
    table(s, [
        ["distinct ask-prefixes over 10,000 users", "k=1", "k=2", "k=4", "k=8", "k=16"],
        ["static (entropy0)", "1", "1", "1", "1", "1"],
        ["GLM-UCB", "1", "3", "23", "335", "5,739"],
        ["Thompson", "759", "8,083", "10,000", "10,000", "10,000"],
        ["latent-only info-gain†", "1", "2", "7", "99", "1,993"],
        ["EDDI three-outcome‡", "1", "2", "8", "112", "1,217"],
    ], MARGIN_L, I(1.25), I(5.6), col_w=[34, 13, 13, 13, 13, 14], font_size=10, row_h=0.27,
        align=num(6), first_col_bold=True)
    table(s, [
        ["Golbandi leaf model as consumer", "k=1", "k=2", "k=4", "k=8"],
        ["Thompson − static, full", "+.0100", "+.0171", "+.0148", "+.0281"],
    ], I(6.2), I(1.25), I(3.46), col_w=[44, 14, 14, 14, 14], font_size=9.5, row_h=0.27,
        align=num(5), first_col_bold=True)
    table(s, [
        ["three- vs two-outcome response model, full gain", "k=1", "k=2", "k=4", "k=8", "k=16"],
        ["paired, in-run", "+.0070", "+.0057", "+.0114", "+.0122", "+.0124"],
    ], MARGIN_L, I(3.4), I(5.6), col_w=[34, 13, 13, 13, 13, 14], font_size=10, row_h=0.27,
        align=num(6), first_col_bold=True)
    bullets(s, I(6.2), I(2.1), I(3.46), [
        "fifth consumer: every interval excludes zero; tail split",
        "GLM-UCB does not participate: +.0007 full, +.0008 tail",
    ], size=10)
    bullets(s, MARGIN_L, I(4.15), CONTENT_W, [
        "Thompson fully personalised from the first question and still loses: adaptive, and it did not help",
        "three-outcome model closes ~¼ of the full deficit (−.0530 → −.0406), ~⅓ of tail; all five intervals exclude zero",
        "EDDI native validation failed: pooled average rank 3.402 vs 3.236 for a random ordering",
    ], size=11.5, height=I(0.9))
    notes(s, "Appendix: the adaptive arms did branch. Thompson is fully personalised from question one and still loses.\n"
             "The three-outcome response model helps EDDI by a quarter of its deficit, nowhere near enough.\n"
             "The fifth consumer, a Golbandi leaf model, is where the verdict inverts: Thompson beats static at every budget.")

    # -- 2.2 -----------------------------------------------------------------------
    s, y = slide(prs, "Appendix 2.2: genre conditioning")
    table(s, [
        ["genre-conditioned greedy, K=15, TEST", "full", "tail"],
        ["fallback-200 arm", "+0.00862 [+0.00595, +0.01131]", "+0.00591 [+0.00326, +0.00862]"],
        ["all-groups arm", "+0.01015", "+0.00719"],
        ["rebuilt global baseline", "0.24295", "≈3.5 % relative gain"],
        ["Drama (54.8 % of the cohort)", "+0.00150 [−0.00187, +0.00486]", ""],
        ["tail at K=8", "", "+0.00203 [−0.00073, +0.00482]"],
        ["validation overstates by", "1.80×", "1.02×"],
    ], MARGIN_L, I(1.25), CONTENT_W, col_w=[38, 32, 30], font_size=10.5, row_h=0.28,
        align=num(3), first_col_bold=True)
    bullets(s, MARGIN_L, I(3.35), CONTENT_W, [
        "pooled gain carried by minority groups: Sci-Fi, Adventure, Thriller, Action +0.017 to +0.090; Drama a null",
        "tree depth is the stopping rule: admissible candidates fall from 876 of 1,000 at the root to zero at n=660",
        "of 15 stopped nodes, zero stop for want of a positive-gain split",
        "six of fifteen leaves lose on test (39.3 % of users), none on validation",
    ], size=11.5)
    tab_cap(s, "A1", "Group-conditioned greedy on TEST (chapterB_v2, sections 3.2, 3.5).", top=I(4.95))
    notes(s, "Appendix: the second conditioning arm, greedy orders per favourite genre.\n"
             "Real pooled gain, but the majority group, Drama, is a null; minority genres carry it.\n"
             "The tree's depth measures the minimum-leaf rule, not the information in the data.\n"
             "Six of fifteen leaves lose on test; validation showed none.")

    # -- 2.3 -----------------------------------------------------------------------
    s, y = slide(prs, "Appendix 2.3: request groups")
    table(s, [
        ["group", "most distinctive films", "n", "train", "requests", "ratio"],
        ["inspiring", "Dark Knight Rises, Interstellar, Django", "9,743", "6.9 %", "20.2 %", "2.9×"],
        ["dialogue driven", "Birdman, Whiplash, Grand Budapest", "7,475", "5.3 %", "16.8 %", "3.2×"],
        ["gore", "Dredd, Watchmen, The Hateful Eight", "5,825", "4.1 %", "16.0 %", "3.9×"],
        ["marvel", "Iron Man 3, Avengers: Age of Ultron", "6,882", "4.9 %", "11.5 %", "2.4×"],
        ["school", "Aquamarine, Princess Diaries 2, Frozen", "4,945", "3.5 %", "6.1 %", "1.7×"],
        ["art house", "Holy Motors, The White Ribbon", "4,576", "3.3 %", "5.1 %", "1.6×"],
        ["six combined", "", "", "28.0 %", "75.7 %", "2.7×"],
    ], MARGIN_L, I(1.25), CONTENT_W, col_w=[18, 40, 10, 10, 12, 10], font_size=10, row_h=0.27,
        align=[L, L, R, R, R, R], first_col_bold=True)
    bullets(s, MARGIN_L, I(3.55), CONTENT_W, [
        "k=20 groups fitted on 140,768 train users; 28 % of users receive 76 % of requests",
        "largest training group (21,477 users, 15.3 %) receives 1.5 % of requests",
        "separability 0.371 → 0.956 once grouped by the state; raw-latent k-means gives release-year buckets",
        "leaf specialisation is a null: overlap 0.267 vs matched chance 0.935; congruence p = 0.82",
    ], size=11.5)
    tab_cap(s, "A2", "The six request groups (chapterB_v2, tab:clusters).", top=I(4.95))
    notes(s, "Appendix: the six request groups, mined from training users and matched to the corpus.\n"
             "Twenty-eight percent of users receive three quarters of the requests.\n"
             "Raw k-means on the latent gives release-year buckets; grouping by the state is what separates.\n"
             "Leaf specialisation by taste is a null: leaves differ more than chance, not more purposefully.")

    # -- 2.4 -----------------------------------------------------------------------
    s, y = slide(prs, "Appendix 2.4: openers and policy")
    table(s, [
        ["property", "real requests", "synthesised"],
        ["names a film", "0.630", "0.580"],
        ["names a genre", "0.398", "0.363"],
        ["names a tone", "0.214", "0.314"],
        ["internal coherence (null +0.172)", "+0.224", "+0.289"],
        ["popularity percentile of named films", "93.3", "96.4"],
    ], MARGIN_L, I(1.25), I(4.3), col_w=[56, 22, 22], font_size=9.5, row_h=0.26, align=num(3), first_col_bold=True)
    table(s, [
        ["group", "n", "routing", "policy", "own tree", "cold", "policy − cold"],
        ["art house", "464", ".843", ".29621", ".30046", ".27964", "+.01657 *"],
        ["dialogue driven", "395", ".792", ".23299", ".22940", ".21894", "+.01404 *"],
        ["inspiring", "1,526", ".855", ".25195", ".25534", ".24338", "+.00857 *"],
        ["gore", "391", ".816", ".33406", ".33208", ".32595", "+.00811"],
        ["teenagers", "502", ".741", ".24405", ".24179", ".23712", "+.00692"],
        ["marvel", "599", ".957", ".42555", ".42534", ".41933", "+.00622"],
    ], I(4.85), I(1.25), I(4.81), col_w=[22, 10, 12, 14, 14, 13, 15], font_size=9, row_h=0.26,
        align=num(7), first_col_bold=True)
    bullets(s, MARGIN_L, I(3.3), CONTENT_W, [
        "synthesised openers close on structure, too popular and too coherent on the margins; a stated lower bound",
        "the policy lands within ±.004 of the group's own tree while also finding the group",
        "per-group ranking moves between validation and test: art house +.02470 → +.01657, dialogue driven −.00266 → +.01404",
        "quote the pooled contrast, not a group",
    ], size=11.5)
    tab_cap(s, "A3", "Left: synthesised vs real openers (tab:opener_match). Right: policy by group, reserved TEST, K=15 (tab:policy).", top=I(4.95))
    notes(s, "Appendix: how close the synthesised openers are to real ones, and the distilled policy per group.\n"
             "Openers match on structure; they are slightly too popular and too coherent, so the arm is a lower bound.\n"
             "The policy sits within four thousandths of each group's own tree while routing itself.\n"
             "Group rankings move between validation and test, so only the pooled contrast is quoted.")
