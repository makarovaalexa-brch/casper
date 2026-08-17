r"""T1' ON HOLDOUT -- the verdict the rebuild exists for. SCORING ONLY, nothing is searched here.

Design sheet: `docs/design/PAPER_B_POLICY_DESIGN.md` L.3. VALIDATION ONLY, `test_touched=false`.

THE QUESTION, and why an in-sample number cannot answer it
-----------------------------------------------------------
`group_greedy_signature.py` searched a stated-genre order over the tree's own combined pool on each
group's FIT users. Its in-sample score is worthless as a verdict: a group's own order beats a global
order BY CONSTRUCTION on the cohort it was fitted to. The citable comparison is on the group's
**HOLDOUT** users, which the search never saw, against the arm L2a already measured there:

  t1prime        the stated genre + the 15 questions of the REBUILT order (items AND concepts)
  t1_item_only   the stated genre + the 15 items of the BANKED order (`groupgreedy_genre15.json`)
  tree_budget    the stated genre + the tree's own first 15 questions

All three hold 16 evidence questions. `t1_item_only` and `tree_budget` are exactly L2a's
`T1_order` and `tree_budget` cells, so they must reproduce `forest_main_gates.json` to 1e-12 before
anything is differenced -- a free chain check on the whole comparison.

THE THIRD ADMISSIBILITY CONDITION (L.3): a teacher earns its place only if it beats the teacher
already covering those states, on the same action space and the same budget. `t1prime - tree_budget`
IS that test, and the banked item-only difference sits beside it so the action-space effect is
readable at a glance.
"""
import os
import sys
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
sys.path.insert(0, _HERE)

import json
import time
import argparse
import numpy as np

from policy_net import (OUTDIR, SEED, TOTAL_L, STATIC_JSON, STATIC_VAL_FULL, VAL_M0_FULL,
                        paired_bootstrap, load_traj, log)

FOREST_DIR = os.path.join(OUTDIR, "forest")


def main():
    ap = argparse.ArgumentParser(description="T1' vs the tree at stated-genre states, on HOLDOUT")
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--full_threads", action="store_true")
    ap.add_argument("--i_have_author_goahead", action="store_true")
    args = ap.parse_args()
    assert args.i_have_author_goahead, "REFUSING TO RUN without --i_have_author_goahead"
    assert "--split" not in sys.argv, "there is no --split here: this file is validation-only"
    t00 = time.time()

    from strategy_channel_suite import (build_greedy_env, contract_selftest, genv_eval, genv_mean,
                                        genv_eval_multi, CERT_TOWER, CERT_CONC)
    from ndcg_tree import build_pool, leak_assert
    from policy_asif import fold_seqs
    from group_greedy_genre import load_item_genres, partition_by_genre
    import pu_labels as PL

    log("#" * 110)
    log("# T1' ON HOLDOUT -- scoring only. VALIDATION ONLY; TEST is not built, not loaded, not named.")
    log("#" * 110)

    E = build_greedy_env(CERT_TOWER, "validation", CERT_CONC, "t1prime_hold")
    contract_selftest(E, E["rows"])
    leak_assert(E, E["rows"], "t1prime_hold/val")
    rows = E["rows"]
    assert len(rows) == 10000
    f0, t0_ = genv_mean(E, [], rows)
    assert round(f0, 8) == VAL_M0_FULL, f"CANONICAL SNAP FAILED: {f0!r}"
    log(f"[gate] canonical m=0 snap full={f0!r} tail={t0_!r} -- exact")
    pool, _i, _c = build_pool(E, 500, 500)
    static_seq = [(int(a), int(b)) for a, b in json.load(open(STATIC_JSON))["sequence"]]
    sf, _ = genv_eval(E, static_seq, rows)
    assert abs(float(np.nanmean(sf)) - STATIC_VAL_FULL) < 1e-12, "BANKED-HEADLINE ASSERT FAILED"
    log(f"[gate] greedy_static_policyB_combined reproduces EXACTLY: {float(np.nanmean(sf))!r}")

    D = load_traj(train=False)
    assert list(np.asarray(D["uid"], np.int64)) == list(np.asarray(rows, np.int64))
    tree_asked = np.asarray(D["label_idx"], np.int64)
    pos_of = {int(r): j for j, r in enumerate(np.asarray(D["uid"], np.int64))}

    fit, hold, fw = PL.fit_holdout_split(np.asarray(rows, np.int64))
    hs = set(int(x) for x in hold.tolist())
    log(f"[firewall] {len(hold):,} HOLDOUT users -- the T1' search never saw one of them")

    sb = json.load(open(os.path.join(FOREST_DIR, "forest_states.json")))
    teachers = sb["teachers"]
    gmap = {g: int(sb["per_teacher"][g]["concept_id"]) for g in teachers}
    banked_item = {g: [(0, int(i)) for i in v["order"]] for g, v in
                   json.load(open(os.path.join(OUTDIR, "groupgreedy_genre15.json")))["groups"].items()}
    gates = json.load(open(os.path.join(FOREST_DIR, "forest_main_gates.json")))["behavioural"]
    gen, _v = load_item_genres(E["ctx"].ni)
    groups, _a, _b = partition_by_genre(E, rows, gen)

    out = {"analysis": "t1prime_holdout_verdict", "design_sheet":
           "docs/design/PAPER_B_POLICY_DESIGN.md L.3", "citable": True, "test_touched": False,
           "split": "validation", "cohort": "HOLDOUT users of each stated-genre group",
           "firewall": fw, "budget": "16 evidence questions in every arm: the stated genre + 15",
           "by_genre": {}}
    for g in teachers:
        p = os.path.join(FOREST_DIR, f"t1prime_{g.replace('-', '')}.json")
        if not os.path.exists(p):
            log(f"[{g}] no rebuilt order banked -- skipped")
            continue
        tb = json.load(open(p))
        assert tb["test_touched"] is False and tb["genre"] == g
        new_order = [(int(a), int(c)) for a, c in tb["order"]]
        grows = [int(r) for r in sorted(groups[g]) if int(r) in hs]
        cq = (1, int(gmap[g]))
        seeds = [[cq] for _ in grows]
        arms = {
            "t1prime": [list(new_order) for _ in grows],
            "t1_item_only": [list(banked_item[g]) for _ in grows],
            "tree_budget": [[pool[int(j)] for j in tree_asked[pos_of[r], :TOTAL_L - 1]]
                            for r in grows]}
        sc = {}
        for name, seqs in arms.items():
            f, t = genv_eval_multi(E, grows, *fold_seqs(seeds, seqs))
            sc[name] = (np.asarray(f, np.float64), np.asarray(t, np.float64))
            log(f"  [{g} {name}] full {float(np.nanmean(f)):.5f} tail {float(np.nanmean(t)):.5f}")

        # ---- the free chain check: two of these arms are L2a's, and must reproduce it exactly ----
        want = gates[g]
        for name, key in (("t1_item_only", "T1_order"), ("tree_budget", "tree_budget")):
            for j, ax in ((0, "full"), (1, "tail")):
                got = float(np.nanmean(sc[name][j])); w = float(want[f"{key}_{ax}"])
                assert abs(got - w) < 1e-12, (
                    f"BANKED-L2a ASSERT FAILED at {g} {name} {ax}: this run {got!r}, L2a {w!r}")
        log(f"[chain] {g}: L2a's T1_order and tree_budget cells reproduce EXACTLY")

        cmp_ = {}
        for a, b in (("t1prime", "tree_budget"), ("t1prime", "t1_item_only"),
                     ("t1_item_only", "tree_budget")):
            cmp_[f"{a}_minus_{b}"] = {
                "full": paired_bootstrap(sc[a][0] - sc[b][0]),
                "tail": paired_bootstrap(sc[a][1] - sc[b][1])}
        out["by_genre"][g] = {
            "n_holdout_users": len(grows), "concept_id": int(gmap[g]),
            "n_concepts_in_rebuilt_order": int(tb["n_concepts_in_order"]),
            "rebuilt_order": [[int(a), int(c)] for a, c in new_order],
            "in_sample_full_FIT": tb["in_sample_full"],
            "scores": {k: {"full": float(np.nanmean(v[0])), "tail": float(np.nanmean(v[1]))}
                       for k, v in sc.items()},
            "comparisons": cmp_}
        c = cmp_["t1prime_minus_tree_budget"]["full"]
        d = cmp_["t1_item_only_minus_tree_budget"]["full"]
        log(f"[{g}] n={len(grows)} REBUILT - tree {c['mean']:+.5f} [{c['lo']:+.5f}, {c['hi']:+.5f}] "
            f"| item-only - tree {d['mean']:+.5f} [{d['lo']:+.5f}, {d['hi']:+.5f}]")

    n_win = sum(1 for v in out["by_genre"].values()
                if v["comparisons"]["t1prime_minus_tree_budget"]["full"]["lo"] > 0)
    n_lose = sum(1 for v in out["by_genre"].values()
                 if v["comparisons"]["t1prime_minus_tree_budget"]["full"]["hi"] < 0)
    out["verdict"] = {
        "question": "does a teacher rebuilt on the tree's own action space beat the incumbent at its "
                    "own states, on users the search never saw?",
        "n_groups": len(out["by_genre"]), "n_groups_beating_the_tree": n_win,
        "n_groups_losing_to_the_tree": n_lose,
        "n_groups_null": len(out["by_genre"]) - n_win - n_lose}
    p = os.path.join(FOREST_DIR, "t1prime_holdout.json")
    out["minutes"] = round((time.time() - t00) / 60, 2)
    with open(p + ".tmp", "w") as fh:
        json.dump(out, fh, indent=2, default=float)
    os.replace(p + ".tmp", p)
    log(f"[verdict] {json.dumps(out['verdict'], default=float)}")
    log(f"[done] wrote {p} ({out['minutes']:.1f}m)")


if __name__ == "__main__":
    main()
