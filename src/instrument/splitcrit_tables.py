r"""Render `splitcrit.json` into the markdown tables that go into
`docs/results/HU_YU_2013_REPLICATION_AND_ABLATION.md`. Pure formatting -- it computes nothing and
must never compute anything, so a table and the artifact it summarises cannot disagree (memory:
`verify-the-artifact-not-a-proxy` -- "a summary statistic and the table it summarises must read the
same field").
"""
import os
import sys
import json
import argparse

_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
ORDER = ["variance", "logistic", "pairwise", "ndcg", "random_ml", "random"]
NICE = {"variance": "(a) variance (Golbandi)", "pairwise": "(b) pairwise rank loss (Hu & Yu)",
        "logistic": "(b') logistic loss (Hu & Yu, as run)", "ndcg": "**(c) measured NDCG@10 (ours)**",
        "random": "(r) random", "random_ml": "(r') random, min-leaf respecting"}


def f(x, n=5):
    return "n/a" if x is None else format(float(x), "+." + str(n) + "f")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=os.path.join(_ROOT, "experiments", "battery", "splitcrit.json"))
    a = ap.parse_args()
    b = json.load(open(a.json))
    arms = [k for k in ORDER if k in b.get("arms", {})] or list(b.get("arms", {}))
    budgets = [int(k) for k in b["budgets"]]
    out = []

    out.append("| criterion | K | val full | val tail | **test full** | 95% CI | **test tail** | "
               "95% CI | sep |")
    out.append("|---|---|---|---|---|---|---|---|---|")
    for k in arms:
        A = b["arms"][k]
        for K in budgets:
            v = A.get("cohort_val", {}).get(str(K), {})
            t = A.get("cohort_test", {}).get(str(K))
            if not t:
                continue
            sep = ("F" if t["separates_full"] else "-") + ("T" if t["separates_tail"] else "-")
            out.append("| " + (NICE.get(k, k) if K == budgets[0] else "") + " | " + str(K) + " | "
                       + f(v.get("diff_full@10")) + " | " + f(v.get("diff_tail@10")) + " | **"
                       + f(t["diff_full@10"]) + "** | ["
                       + f(t["ci95_full"][0]) + ", " + f(t["ci95_full"][1]) + "] | **"
                       + f(t["diff_tail@10"]) + "** | ["
                       + f(t["ci95_tail"][0]) + ", " + f(t["ci95_tail"][1]) + "] | " + sep + " |")
    out.append("")
    out.append("### Structure")
    out.append("")
    out.append("| criterion | depth | nodes | leaves | root splitter | root answer rate | "
               "root children (L/D/U) | item slots | concept slots | distinct questions |")
    out.append("|---|---|---|---|---|---|---|---|---|---|")
    for k in arms:
        s = b["structure"].get(k)
        if not s:
            continue
        cs = s.get("root_child_sizes") or {}
        out.append("| " + NICE.get(k, k) + " | " + str(s["depth_reached"]) + " | "
                   + str(s["n_nodes"]) + " | " + str(s["n_leaves"]) + " | "
                   + str(s["root_channel"]) + " " + str(s["root_splitter"][1]) + " | "
                   + str(s.get("root_answer_rate")) + " | "
                   + "/".join(str(cs.get(x, "?")) for x in ("like", "dislike", "noanswer")) + " | "
                   + str(s["question_slots_items"]) + " | " + str(s["question_slots_concepts"])
                   + " | " + str(s["n_distinct_questions"]) + " |")
    ov = b.get("overlap", {})
    if ov:
        out.append("")
        out.append("### Overlap of the questions each criterion chooses")
        out.append("")
        out.append("| pair | shared questions | union | shared splitters |")
        out.append("|---|---|---|---|")
        for key, n in sorted(ov.get("questions_asked", {}).items()):
            out.append("| " + key.replace("|", " vs ") + " | " + str(n) + " | "
                       + str(ov["questions_asked_union"][key]) + " | "
                       + str(ov["splitters"].get(key, "?")) + " |")
    cc = b.get("criterion_vs_measured_ndcg")
    if cc:
        out.append("")
        out.append("### Criterion vs MEASURED NDCG@10, over every candidate at every node of the "
                   "real tree")
        out.append("")
        out.append("| criterion | mean Spearman | nodes with rho>0 | argmax == NDCG argmax | "
                   "mean top-10 recall | user-weighted regret |")
        out.append("|---|---|---|---|---|---|")
        for k, v in cc["aggregate"].items():
            out.append("| " + NICE.get(k, k) + " | " + format(v["mean_spearman"], "+.3f") + " | "
                       + str(v["n_nodes_with_positive_spearman"]) + "/" + str(v["n_nodes"]) + " | "
                       + str(v["n_nodes_argmax_matches"]) + "/" + str(v["n_nodes"]) + " | "
                       + format(v["mean_top10_recall"], ".2f") + " | "
                       + format(v["user_weighted_mean_regret"], "+.5f") + " |")
        out.append("| *control: uniform random pick* | | | | | "
                   + format(cc["mean_expected_regret_of_a_uniform_random_pick"], "+.5f") + " |")
    print("\n".join(out))


if __name__ == "__main__":
    main()
