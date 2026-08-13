r"""Read `pwheldout_root_partial.json` and report separation (1) over WHATEVER has been evaluated.

The probe writes its partial every ten candidates and evaluates candidates in an INTERLEAVED
item/concept order, so any prefix is a channel-balanced sample of the 1,000-candidate pool. This
script exists so the exhibit has a usable answer at any moment rather than only on completion --
and so the answer is always reported with its own coverage attached, never as if it were the full
pool.

It computes nothing the probe does not: it reads banked per-candidate pairs of (measured NDCG@10
marginal gain, untruncated pairwise-rank-loss reduction) at the root state and correlates them.
"""
import os
import sys
import json
import argparse

import numpy as np

_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--partial", default=os.path.join(_ROOT, "experiments", "battery",
                                                      "pwheldout_root_partial.json"))
    ap.add_argument("--out", default=os.path.join(_ROOT, "experiments", "battery",
                                                  "pwheldout_root_PARTIAL_REPORT.json"))
    a = ap.parse_args()
    b = json.load(open(a.partial))
    cand = b["cand"]
    ks = sorted(cand, key=lambda k: tuple(int(x) for x in k.split(",")))
    gn = np.asarray([cand[k]["ndcg_gain"] for k in ks], np.float64)
    gp = np.asarray([cand[k]["pairwise_gain"] for k in ks], np.float64)
    ch = np.asarray([int(k.split(",")[0]) for k in ks], np.int64)
    from scipy.stats import spearmanr, kendalltau
    an, ap_ = int(np.argmax(gn)), int(np.argmax(gp))
    res = {"analysis": "separation_1_PARTIAL",
           "CITABLE_AS_FULL_POOL": len(ks) == 1000,
           "n_candidates_evaluated": len(ks),
           "coverage_of_pool": round(len(ks) / 1000.0, 4),
           "n_items": int((ch == 0).sum()), "n_concepts": int((ch == 1).sum()),
           "channel_balanced": bool(abs(int((ch == 0).sum()) - int((ch == 1).sum())) <= 2),
           "base_full_ndcg10": b["base_full"], "base_pairwise_rank_loss": b["base_pw"],
           "spearman_ndcg_vs_pairwise": float(spearmanr(gn, gp).correlation),
           "kendall_ndcg_vs_pairwise": float(kendalltau(gn, gp).correlation),
           "ndcg_argmax": [int(x) for x in ks[an].split(",")],
           "pairwise_argmax": [int(x) for x in ks[ap_].split(",")],
           "argmax_agrees": bool(an == ap_),
           "best_ndcg_gain_seen": float(gn[an]),
           "ndcg_gain_of_pairwise_argmax": float(gn[ap_]),
           "regret_of_following_the_pairwise_loss": float(gn[an] - gn[ap_]),
           "expected_regret_of_a_uniform_random_pick": float(gn[an] - gn.mean()),
           "rank_of_ndcg_argmax_under_pairwise": int(1 + (gp > gp[an]).sum()),
           "top10_recall": float(len(set(np.argsort(-gp)[:10].tolist())
                                     & set(np.argsort(-gn)[:10].tolist())) / 10.0)}
    for nm, m in (("items", ch == 0), ("concepts", ch == 1)):
        if m.sum() >= 5:
            res["spearman_" + nm] = float(spearmanr(gn[m], gp[m]).correlation)
    with open(a.out, "w") as fh:
        json.dump(res, fh, indent=2, default=float)
    print(json.dumps(res, indent=2))
    print("\nPARTIAL: " + str(len(ks)) + "/1000 candidates ("
          + str(res["n_items"]) + " items, " + str(res["n_concepts"]) + " concepts). "
          + ("Channel-balanced, so this prefix is a fair sample of the pool."
             if res["channel_balanced"] else
             "NOT channel-balanced -- report the per-channel correlations, not the pooled one."))


if __name__ == "__main__":
    main()
