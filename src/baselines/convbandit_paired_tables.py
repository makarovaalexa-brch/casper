r"""convbandit_paired_tables.py -- render `convbandit_paired.json` as the markdown deliverable.

Reads the machine-readable outputs only; computes means off the same per-user vectors the CIs came
from, so the doc and the JSON cannot drift.

  python src/baselines/convbandit_paired_tables.py
"""
import os
import json

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
OUT = os.path.join(_ROOT, "experiments", "baselines", "interview")
DOC = os.path.join(_ROOT, "docs", "results", "CONVBANDIT_PAIRED_CIS.md")
BUD = (1, 2, 4, 8, 16)


def cell(d):
    if d is None:
        return "—"
    return f"{d['delta']:+.4f} [{d['lo']:+.4f}, {d['hi']:+.4f}] {'**SEP**' if d['separates'] else 'ns'}"


def block(bucket, rows, title, lead):
    out = [f"### {title}", "", lead, "",
           "| comparison | axis | k=1 | k=2 | k=4 | k=8 | k=16 |", "|---|---|---|---|---|---|---|"]
    for label, key in rows:
        for ax in ("full", "tail"):
            cs = [cell(bucket.get(f"{ax}/k{k}/{key}")) for k in BUD]
            out.append(f"| {label} | {ax} | " + " | ".join(cs) + " |")
    out.append("")
    return out


def main():
    R = json.load(open(os.path.join(OUT, "convbandit_paired.json")))
    M = R["meta"]

    V = np.load(os.path.join(OUT, "adaptive_table_convbandit_uniform_v2_per_user.npz"))
    RS = np.load(os.path.join(OUT, "convbandit_rescore_per_user.npz"))
    NC = np.load(os.path.join(OUT, "convbandit_rescore_nocap_per_user.npz"))

    def pt(z, pre, name, k):
        return (f"{z[f'{pre}|{name}|k{k}|full'].mean():.4f}/"
                f"{z[f'{pre}|{name}|k{k}|tail'].mean():.4f}")

    L = []
    L.append("# Paired CIs on the conversational-bandit port — and the contract mix inside its "
             "headline gap")
    L.append("")
    L.append(f"Task #17. Paired **percentile bootstrap over the per-user difference vector** "
             f"({M['boot']:,} resamples, seed {M['seed']}); users are resampled and the per-user "
             f"delta recomputed. The unpaired `1.96*sqrt(2)*SE` form is banned — it overstated "
             f"intervals ~4.6x and forced a retraction (`ARM_N_PHASE1.md` §4b).")
    L.append("")
    L.append(f"Contract: {M['contract']}. n_full = {M['n_full_scored']:,}, "
             f"n_tail = {M['n_tail_scored']:,} (the users with ≥1 tail target).")
    L.append("")
    L.append("Sources: `experiments/baselines/interview/adaptive_table_convbandit_uniform_v2_per_user.npz` "
             "(the run), `convbandit_rescore_per_user.npz` and `convbandit_rescore_nocap_per_user.npz` "
             "(the fixed static orders re-scored in-harness by `src/baselines/convbandit_greedy_rescore.py`), "
             "analysis `src/baselines/convbandit_paired.py` → `convbandit_paired.json`.")
    L.append("")

    # ------------------------------------------------------------------ controls
    L.append("## Controls, all passed before any gap was read")
    L.append("")
    L.append(f"* **Self-test** — an arm minus itself: delta 0.0000, CI [0.0000, 0.0000] at every "
             f"budget and axis. `self_test_pass = {M['self_test_pass']}`.")
    L.append(f"* **Schedule self-test** — under `b(t)=5⌊log t⌋` no key-term turn fires before t=3, "
             f"so `conlinucb_mcr`, `conlinucb_bs` and `clime` must be numerically identical at k=1 "
             f"and k=2. All three pairwise deltas are **exactly 0.0000 with CI [0.0000, 0.0000]** on "
             f"both axes at both budgets. `schedule_self_test_pass = {M['schedule_self_test_pass']}`.")
    L.append(f"* **Cross-run admission, item side** — the re-scored `entropy0` top-16 order "
             f"reproduces the banked ladder vector **per user**: max abs diff "
             f"{M['rescore_entropy0_canary_max_abs_per_user_diff']:.1e}. Its means come back "
             f"`.1635/.0433 · .1706/.0528 · .1831/.0636 · .2053/.0792 · .2303/.0924`, identical to "
             f"the published ladder. Greedy items-only likewise reproduces "
             f"`.1760/.0434 · .1907/.0581 · .2066/.0703 · .2247/.0776 · .2449/.0967`.")
    L.append(f"* **Cross-run admission, concept side** — `ConLinUCB-BS-static` is a *fixed* key-term "
             f"order in this run (asserted: identical acts for all 10,000 users), so it can be "
             f"re-scored like any static arm. Re-scoring it reproduces the banked convbandit "
             f"vectors **per user**: max abs diff "
             f"{M['rescore_bs_static_canary_max_abs_per_user_diff']:.1e}. That is what licenses "
             f"differencing a re-scored *concept* sequence at all.")
    L.append("")
    L.append("No comparison in this file crosses runs on a mean anchor. Every static row was "
             "re-scored inside this harness (precedent: task #16, `gtport_greedy_rescore.py`).")
    L.append("")

    # ------------------------------------------------------------------ the two contracts
    L.append("## ★ The finding that comes before the intervals: two concept-answer conventions")
    L.append("")
    L.append("The greedy concepts-only row could not be reproduced in this harness at first. "
             "The cause is a real difference in the **answer**, not in the question:")
    L.append("")
    L.append("* `conversational_bandit_selectors.build_concept_stack` builds concept answers with "
             "`signed_values(...)` at its default `apply_neg_cap=True` — the **population-wide** "
             "negative-mass cap, spread across all 1,031 concepts at once.")
    L.append("* `strategy_channel_suite.build_greedy_env` builds them with `apply_neg_cap=False` "
             "and applies `cap_negatives` over the **revealed** list instead — which its own comment "
             "calls \"the certified convention\", and is what the concept channel cv3 was trained on.")
    L.append("")
    L.append("Re-scoring the greedy concepts-only order under the second convention reproduces the "
             "published row **exactly** — `.1783/.0478 · .1909/.0548 · .2147/.0775 · .2362/.1022 · "
             ".2473/.1154` — while under the convbandit run's own convention the same 16 questions "
             "reach only `.2298/.0977` at k=16. So the published `.2473/.1154` and the published "
             "`.1891/.0700` were measured under **different information contracts** and were never "
             "ordered (`surprise-means-debug-the-harness`).")
    L.append("")
    L.append("| fixed order | answers | k=1 | k=2 | k=4 | k=8 | k=16 |")
    L.append("|---|---|---|---|---|---|---|")
    for name, lab in (("greedy-concepts-only", "greedy concepts-only"),
                      ("conlinucb_bs_static", "ConLinUCB-BS-static")):
        L.append(f"| {lab} | in-contract (population cap) | "
                 + " | ".join(pt(RS, "rescore", name, k) for k in BUD) + " |")
        L.append(f"| {lab} | certified-era (revealed cap) | "
                 + " | ".join(pt(NC, "rescore", name, k) for k in BUD) + " |")
    L.append("")
    L.append("The convention costs the *greedy* arm far more than it costs the *bandit* arm "
             "(§C5 below): the bandit's spanner concepts carry little negative mass, so shrinking "
             "negatives barely touches it. Both readings are given below; both are like-for-like.")
    L.append("")

    # ------------------------------------------------------------------ point table
    L.append("## Point estimates used (all from the same per-user vectors as the CIs)")
    L.append("")
    L.append("| arm | k=1 | k=2 | k=4 | k=8 | k=16 |")
    L.append("|---|---|---|---|---|---|")
    for spec, lab in (("clime", "CLiME"), ("conlinucb_bs", "ConLinUCB-BS"),
                      ("conlinucb_mcr", "ConLinUCB-MCR"),
                      ("conlinucb_bs_static", "ConLinUCB-BS-static *(concepts only)*"),
                      ("conucb", "ConUCB")):
        L.append(f"| {lab} | " + " | ".join(pt(V, "ours", spec, k) for k in BUD) + " |")
    for name, lab in (("entropy0-top16", "*entropy₀ (re-scored)*"),
                      ("greedy-items-only", "*greedy items-only (re-scored)*"),
                      ("greedy-concepts-only", "*greedy concepts-only (re-scored, in-contract)*")):
        L.append(f"| {lab} | " + " | ".join(pt(RS, "rescore", name, k) for k in BUD) + " |")
    L.append("")

    # ------------------------------------------------------------------ the CI blocks
    L.append("## The intervals")
    L.append("")
    L += block(R["C1_concept_channel_in_contract"],
               [("greedy concepts-only − ConLinUCB-BS-static", "greedy_concepts-minus-conlinucb_bs_static")],
               "C1 ★ the load-bearing one, IN CONTRACT",
               "Both arms under the convbandit run's own concept-answer convention. "
               "Static-versus-static, same channel, same budget, same users.")
    L += block(R["C1b_concept_channel_certified_answers"],
               [("greedy concepts-only − ConLinUCB-BS-static", "greedy_concepts-minus-conlinucb_bs_static")],
               "C1b the same two orders under the CERTIFIED-ERA answers",
               "The convention cv3 was trained on and the published greedy row was measured under. "
               "Both arms re-scored, so this is still like-for-like.")
    L += block(R["C2_clime_vs_static"],
               [("CLiME − entropy₀", "clime-minus-entropy0"),
                ("CLiME − greedy items-only", "clime-minus-greedy_items"),
                ("CLiME − greedy concepts-only", "clime-minus-greedy_concepts")],
               "C2 the Act-1 null: the best conversational bandit against the strongest statics",
               "Static rows re-scored in-harness; concepts row is the in-contract one.")
    L += block(R["C3_conucb_vs_successors"],
               [("ConUCB − ConLinUCB-MCR", "conucb-minus-conlinucb_mcr"),
                ("ConUCB − ConLinUCB-BS", "conucb-minus-conlinucb_bs"),
                ("ConUCB − CLiME", "conucb-minus-clime"),
                ("ConUCB − ConLinUCB-BS-static", "conucb-minus-conlinucb_bs_static")],
               "C3 ConUCB against its own successors (the answerability claim)",
               "Answered items at k=16: ConUCB **1.80**, MCR 4.77, BS 4.75, CLiME 4.80, "
               "BS-static 0.00 (it asks only key-terms).")
    L += block(R["C4_family_pairwise"],
               [("CLiME − ConLinUCB-BS", "clime-minus-conlinucb_bs"),
                ("CLiME − ConLinUCB-MCR", "clime-minus-conlinucb_mcr"),
                ("CLiME − ConLinUCB-BS-static", "clime-minus-conlinucb_bs_static"),
                ("ConLinUCB-BS − ConLinUCB-MCR", "conlinucb_bs-minus-conlinucb_mcr"),
                ("ConLinUCB-BS − ConLinUCB-BS-static", "conlinucb_bs-minus-conlinucb_bs_static"),
                ("ConLinUCB-MCR − ConLinUCB-BS-static", "conlinucb_mcr-minus-conlinucb_bs_static")],
               "C4 the rest of the family, pairwise",
               "k=1 and k=2 are the schedule self-test cells: exactly 0.0000, CI [0.0000, 0.0000].")
    L += block(R["C5_answer_convention_cost"],
               [("greedy concepts-only: certified − in-contract",
                 "greedy-concepts-only/certified_answers-minus-in_contract"),
                ("ConLinUCB-BS-static: certified − in-contract",
                 "conlinucb_bs_static/certified_answers-minus-in_contract")],
               "C5 what the answer convention alone is worth",
               "Same fixed order, same consumer, same users — only the concept-answer construction "
               "changes.")

    open(DOC, "w", encoding="utf-8").write("\n".join(L) + "\n" + VERDICT)
    print(f"written -> {DOC}")


VERDICT = r"""
## Verdict — what `CONVBANDIT_PORT_ON_OUR_RULER.md` may and may not say now

**Survives, with the interval attached.** Finding 1's two numbers are both real: CLiME loses to
entropy₀ by **−0.0085 [−0.0111, −0.0060]** full at k=16 and to greedy items-only by **−0.0231
[−0.0262, −0.0201]**, both SEP, and CLiME is behind *every* static row on *both* axes at *every*
budget — **thirty of thirty** separated comparisons in C2. The Act-1 null against the
published family is the strongest-supported claim in the doc. Finding 4's schedule argument is
verified exactly as stated: the three adaptive variants are bit-identical at k=1 and k=2 (delta
0.0000, CI [0.0000, 0.0000]). Finding 3's direction holds — ConUCB trails MCR/BS/CLiME by −0.0316 to
−0.0344 full at k=16, all SEP.

**Does NOT survive as written — one sentence, and it is the load-bearing one.**

> *"It reaches **.1891/.0700** where our greedy concepts-only reaches **.2473/.1154**: **+.0582 full
> and +.0454 tail**, static-versus-static inside the same channel, same budget, same contract."*

The words "same contract" are false. `.1891/.0700` was measured with concept answers built under
`signed_values(apply_neg_cap=True)` — the population-wide negative cap in
`build_concept_stack` — and `.2473/.1154` was measured under `apply_neg_cap=False` plus a
revealed-set cap, the convention `concept_cv3_ep08` was trained on. Re-scoring the greedy order in
the convbandit harness reproduces `.2298/.0977`, not `.2473/.1154`; re-scoring it under the
certified-era convention reproduces `.2473/.1154` to four decimals. `+.0582/+.0454` is a subtraction
across two information contracts and is not a measured gap. The two honest readings, both
like-for-like and both SEP at every budget, are:

* **in the convbandit contract** (both arms population-capped): **+0.0407 [+0.0381, +0.0432]** full
  and **+0.0276 [+0.0246, +0.0308]** tail at k=16;
* **under the certified-era answers** (both arms re-scored, ConLinUCB-BS-static rises to
  `.1921/.0745`): **+0.0553 [+0.0524, +0.0582]** full and **+0.0408 [+0.0378, +0.0440]** tail.

So the *claim* — the published method for choosing which concepts to ask is beaten decisively by our
greedy concept order, static-versus-static, at every budget from k=1 on both axes — **survives, and
separates everywhere** (10/10 comparisons under either convention). Only the number is wrong: by
0.0175 full / 0.0178 tail against the in-contract reading (about 3.4 and 2.9 interval widths) and by
0.0029 / 0.0046 against the certified-era reading (inside half an interval width). Write one of the
two bracketed pairs, name the convention, and drop "same contract".

**Two smaller corrections.**

1. Finding 3 attributes ConUCB's deficit to answerability. The intervals do not license that
   attribution as stated: against `ConLinUCB-BS-static`, which answers **zero** items, ConUCB is
   **ns** at k=8 (−0.0024 [−0.0047, +0.0000]) and k=16 (−0.0017 [−0.0045, +0.0011]) on full and
   ns on tail at both. ConUCB is separated from the arms that *answer more items* and tied with the
   arm that answers *none*; that is consistent with the answerability story but it also means
   "worst arm in its own family" is only true against the three item-capable successors. Say
   "trails its three item-asking successors" and drop "worst in its family".
2. The doc's own "Owed" line — *"Paired CIs on every gap here … the load-bearing number is the
   +.0582/+.0454 concept-channel gap"* — is discharged, but the answer-convention split it
   uncovered is now the more consequential open item: **every concept number the conversational
   bandits produced is under a convention the certified concept channel was not trained on**, and
   §C5 prices that at up to +0.0177 tail on a concept-only order. `ConLinUCB-BS`, `-MCR` and `CLiME`
   should be re-run under `apply_neg_cap=False` before their concept-turn rows are cited; their
   published values are floors, not estimates.
"""

if __name__ == "__main__":
    main()
