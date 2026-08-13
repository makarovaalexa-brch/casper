r"""WORKSTREAM E / PHASE 3 (design sheet PART G.B, "STAGE B -- REACH") -- ONE LEVEL BELOW THE FLOOR.

Design sheet: `docs/design/PAPER_B_POLICY_DESIGN.md`, Part G.B is binding. Read it before running
anything here; this file implements it and does not restate its argument.

WHAT PHASE 3 IS FOR
-------------------
`TREE_DEPTH_AND_MINLEAF.md` measured that the NDCG tree's depth-5 stop is a property of the STOPPING
RULE, not of the data: all 15 stops were FLOOR stops with a strictly positive gain still in hand
(+0.00672 .. +0.04251), none stopped for want of a split, and 8,398 users (84.0%) sit below a node
that had more to ask. Phase 3 asks the question that leaves open: **if you ask one more conditioned
question below every floor stop, does anything happen?** -- and it asks it TWICE, once through a
parametric policy (which has no floor because it does not partition) and once through the tree itself
with the floor lowered, because if the deeper tree captures the same gain then the honest sentence is
"the policy is a re-parameterisation of a deeper tree" and we claim nothing.

WHAT THIS FILE DOES (two modes, in order)
-----------------------------------------
  --mode labels   Reads the exact measured-NDCG sweep at the 42 below-floor states
                  (`policy_labels_deep.json`, produced by `policy_labels.py --mode sweep --states
                  deep`) and turns it into (leaf, branch) -> question. ZERO scoring cost; what it
                  pays for is the gates.
  --mode control  ★ THE MANDATORY CONTROL. Builds the DEEPER TREE -- the frozen tree with one extra
                  conditioned question below every floor stop, the floor lowered to zero -- and scores
                  its real interviews on the 10,000 validation users. Also scores a SHUFFLED deeper
                  tree (the 42 branch questions permuted across branches, HARD RULE 5), and runs the
                  order-invariance check that tells us what a K=16 comparison can and cannot see.

THE ONE APPROXIMATION, DECLARED RATHER THAN BURIED
--------------------------------------------------
A fully-grown deeper level would re-run the greedy tail from each of the 42 new child states, i.e. ~10
more greedy levels x 42 states x ~995 candidates ~ 33 h on this box. It is not affordable in this
window, so the deeper tree keeps its PARENT LEAF's greedy tail after the new question:

    tree      :  path(d) , splitter , s_{d+1} , s_{d+2} , ... , s_15                    (16 questions)
    deeper    :  path(d) , splitter , g_branch , [the same tail, minus g if it is in it,
                                                  otherwise minus its LAST pick]        (16 questions)

Budget-neutral (16 either way), branch-conditioned at position d+1, and the dropped question is the
greedy tail's WEAKEST pick (greedy is prefix-optimal). It is a real deeper tree in the only sense that
matters for the comparison -- the question at position d+1 is chosen from the child's own measured
NDCG -- but it is NOT a fully re-greedied one, and no sentence anywhere may pretend otherwise.

FIREWALL
--------
The 10,000 canonical TEST users are never built, never loaded, never touched. `--split` accepts
`validation` only. Every artifact carries `test_touched=false`. The tree is never regrown: it is
loaded frozen, and growing ONE level BELOW its existing stops is exactly what this phase is for.

HARD RULE 11 -- every scored cell is written to disk the moment it completes, cached vectors are
float32, and nothing is sampled (HARD RULE 1): every state uses every one of its users.
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

OUTDIR = os.path.join(_ROOT, "experiments", "battery")
TRAJ_DIR = os.path.join(OUTDIR, "policy_traj")
TREE_JSON = os.path.join(OUTDIR, "ndcgtree.json")
TREE16_JSON = os.path.join(OUTDIR, "ndcgtree16.json")
DEEP_SWEEP = os.path.join(OUTDIR, "policy_labels_deep.json")
DEEP_LABELS = os.path.join(OUTDIR, "policy_deep_labels.json")
STATIC_JSON = os.path.join(OUTDIR, "greedy_static_policyB_combined.json")
E1_PER_USER = os.path.join(OUTDIR, "policy_main_equivariant_both_gates_per_user.npz")

BRANCHES = ("like", "dislike", "noanswer")
TOTAL_L = 16
SEED = 98765
# Banked headline numbers, asserted rather than trusted.
VAL_M0_FULL = 0.15686538
STATIC_VAL_FULL = 0.25561016113602253
STATIC_VAL_TAIL = 0.12366492106552922
# The bank-alone comparator on the SAME cohort and ruler (F.4: the bar is the BANK, not the tree).
# greedy_static_policyB_items K=16 on validation, from its own banked curve.
STATIC_ITEMS_VAL_FULL = 0.2428081265667196
STATIC_ITEMS_VAL_TAIL = 0.10111517752587719
SCORE_AT = (1, 2, 4, 8, 16)


def log(*a):
    print(time.strftime("[%H:%M:%S]"), *a, flush=True)


def depth_of(nid):
    return 0 if nid == "root" else nid.count(".")


# ============================================================================ mode: labels
def build_labels(args):
    """(leaf, branch) -> the exact measured-NDCG argmax at that below-floor state. Zero scoring cost."""
    assert os.path.exists(DEEP_SWEEP), (
        f"{os.path.basename(DEEP_SWEEP)} is missing. Run first:\n"
        f"  python src/instrument/policy_labels.py --mode sweep --states deep --label deep "
        f"--full_threads")
    sw = json.load(open(DEEP_SWEEP))
    assert sw["split"] == "validation" and sw["test_touched"] is False
    assert sw["states_selector"] == "deep", f"this sweep is --states {sw['states_selector']}"
    man = json.load(open(os.path.join(TRAJ_DIR, "manifest.json")))
    pool = [(int(a), int(b)) for a, b in man["pool"]]
    pool_index = {q: i for i, q in enumerate(pool)}
    tr = json.load(open(TREE_JSON))
    t16 = json.load(open(TREE16_JSON))
    leaves = {l["nid"]: l for l in t16["leaf_table"]}
    tails = {nid: [(int(a), int(b)) for a, b in lf["own_sequence"]] for nid, lf in leaves.items()}

    recs, n_cov = [], 0
    for sid, r in sw["states"].items():
        assert r.get("done"), f"G-D1: state {sid} is not complete in the sweep"
        nid = r["nid"]                                     # "<leaf>.<branch>"
        leaf, br = nid.rsplit(".", 1)
        assert leaf in leaves, f"{sid}: {leaf} is not a stopped node of the frozen tree"
        assert br in BRANCHES, f"{sid}: unknown branch {br}"
        d = depth_of(leaf)
        assert int(r["pos"]) == d + 1, f"{sid}: pos {r['pos']} != depth+1 {d + 1}"
        vals = r["vals"]
        ks = list(vals)
        g = np.asarray([vals[k] for k in ks], np.float64)
        a_nf = int(np.argmax(g[:, 0]))
        q = tuple(int(x) for x in ks[a_nf].split("_"))
        assert q in pool_index, f"{sid}: argmax {q} is outside the declared pool"
        gain = float(g[a_nf, 0])
        tail_after = set(tails[leaf][d + 1:])
        recs.append({"sid": sid, "leaf": leaf, "branch": br, "branch_idx": BRANCHES.index(br),
                     "depth": d, "pos": d + 1, "n_users": int(r["n_users"]),
                     "floor_broken": bool(int(r["n_users"]) < int(tr["min_leaf"])),
                     "question": [int(q[0]), int(q[1])], "pool_idx": int(pool_index[q]),
                     "gain_ndcg_full": gain,
                     "gain_ndcg_tail": float(g[a_nf, 1]),
                     "argmax_dcg_full": r.get("argmax_dcg_full"),
                     "dcg_argmax_agrees": bool(r.get("argmax_agrees")),
                     "argmax_already_in_leaf_tail": bool(q in tail_after),
                     "n_candidates": len(ks), "base_ndcg_full": float(r["base"][0])})
        n_cov += int(r["n_users"])

    # ---- gates ----------------------------------------------------------------------------------
    by_leaf = {}
    for r in recs:
        by_leaf.setdefault(r["leaf"], []).append(r)
    for leaf, rs in by_leaf.items():
        got = sum(x["n_users"] for x in rs)
        want = int(tr["nodes"][leaf]["n_users"])
        assert got == want, (f"G-D2 COVERAGE: leaf {leaf} splits to {got} users across its branches, "
                             f"the banked node holds {want}")
    assert n_cov == 10000, f"G-D2 COVERAGE: the 42 below-floor states hold {n_cov} users, not 10,000"
    assert len(by_leaf) == len(leaves), (f"G-D1: {len(by_leaf)} of {len(leaves)} stopped nodes have "
                                         f"below-floor states")
    n_neg = sum(1 for r in recs if r["gain_ndcg_full"] <= 0)
    blob = {"analysis": "policy_deep_labels", "design_sheet":
            "docs/design/PAPER_B_POLICY_DESIGN.md G.B", "citable": True,
            "split": "validation", "test_touched": False,
            "label_definition": ("the EXACT MEASURED-NDCG@10 argmax over the 1,000-candidate pool at "
                                 "each of the 42 states one level BELOW the tree's 15 floor stops. "
                                 "No surrogate anywhere in the label."),
            "provenance": {"sweep": os.path.basename(DEEP_SWEEP),
                           "tree": os.path.basename(TREE_JSON),
                           "tails": os.path.basename(TREE16_JSON)},
            "gates": {"G-D1_all_states_done": len(recs),
                      "G-D2_users_covered": n_cov,
                      "G-D3_states_with_nonpositive_gain": n_neg,
                      "G-D4_no_test": "split=validation; the test cohort is never named here"},
            "n_states": len(recs), "n_below_floor": sum(1 for r in recs if r["floor_broken"]),
            "n_distinct_questions": len({tuple(r["question"]) for r in recs}),
            "n_argmax_already_in_leaf_tail": sum(1 for r in recs
                                                 if r["argmax_already_in_leaf_tail"]),
            "min_state_n_users": min(r["n_users"] for r in recs),
            "max_state_n_users": max(r["n_users"] for r in recs),
            "concept_share_of_labels": float(np.mean([r["question"][0] == 1 for r in recs])),
            "pool_sha1": man["sig"].get("pool_sha1"), "states": recs}
    with open(DEEP_LABELS + ".tmp", "w") as fh:
        json.dump(blob, fh, indent=2, default=float)
    os.replace(DEEP_LABELS + ".tmp", DEEP_LABELS)
    log(f"[labels] {len(recs)} below-floor states, {n_cov} users covered, "
        f"{blob['n_below_floor']} of them under the 200 floor; "
        f"{blob['n_distinct_questions']} distinct questions; "
        f"{blob['n_argmax_already_in_leaf_tail']} argmaxes were already in the leaf's own tail; "
        f"{n_neg} states have a non-positive gain")
    log(f"[labels] wrote {os.path.basename(DEEP_LABELS)}")
    return blob


# ============================================================================ sequence construction
def extend_sequence(seq_idx, d, g_idx):
    """The deeper tree's 16 questions for one user, as POOL INDICES. See the module docstring.

    `seq_idx` = the leaf's own 16-question sequence (path[:d] + splitter at d + tail).
    Returns the same length, with `g_idx` inserted at position d+1.
    """
    head = list(seq_idx[:d + 1])
    tail = list(seq_idx[d + 1:])
    if g_idx in tail:
        tail.remove(g_idx)                       # same SET as the tree -- a genuine null for this user
    else:
        tail = tail[:-1]                         # drop the greedy tail's weakest (last) pick
    out = head + [g_idx] + tail
    assert len(out) == TOTAL_L and len(set(out)) == TOTAL_L, \
        f"extended interview is {len(out)} long with {len(set(out))} distinct questions"
    return out


def score_sequences(E, rows, pool, seqs, at=SCORE_AT, tag=""):
    """Per-user (full, tail) NDCG@10 at each K in `at`, every user carrying their own sequence.

    Same funnel as everything else in this chapter (`genv_eval_multi`), same policy-B mask, same
    learned decoder bias, same certified pair. No second scorer exists in this file.
    """
    from strategy_channel_suite import genv_eval_multi
    out = {}
    for K in at:
        iseq = [[pool[int(j)][1] for j in s[:K] if pool[int(j)][0] == 0] for s in seqs]
        cseq = [[pool[int(j)][1] for j in s[:K] if pool[int(j)][0] == 1] for s in seqs]
        f, t = genv_eval_multi(E, rows, iseq, cseq)
        out[K] = (f, t)
        log(f"[score{(' ' + tag) if tag else ''}] K={K:2d} full={float(np.nanmean(f)):.8f} "
            f"tail={float(np.nanmean(t)):.8f}")
    return out


def paired_bootstrap(diff, n_boot=10000, seed=SEED):
    """The paired percentile bootstrap over per-user DIFFERENCES. NEVER the unpaired
    `1.96*sqrt(2)*SE` form (`paired-test-kills-the-tie-claim`)."""
    d = np.asarray(diff, np.float64)
    d = d[~np.isnan(d)]
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(n_boot, len(d)))
    bs = d[idx].mean(1)
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return {"mean": float(d.mean()), "lo": float(lo), "hi": float(hi), "n": int(len(d)),
            "excludes_zero": bool(lo > 0 or hi < 0)}


# ============================================================================ mode: control
def run_control(args):
    """★ THE DEEPER-TREE CONTROL. If it captures what the policy captures, we claim nothing."""
    from strategy_channel_suite import (build_greedy_env, contract_selftest, genv_eval, genv_mean,
                                        CERT_TOWER, CERT_CONC)
    from ndcg_tree import build_pool, leak_assert
    t00 = time.time()
    assert os.path.exists(DEEP_LABELS), "run --mode labels first"
    dl = json.load(open(DEEP_LABELS))
    man = json.load(open(os.path.join(TRAJ_DIR, "manifest.json")))
    pool_man = [(int(a), int(b)) for a, b in man["pool"]]

    E = build_greedy_env(args.snapshot, "validation", args.concept_ckpt, "policy_deep/control")
    contract_selftest(E, E["rows"])
    leak_assert(E, E["rows"], "policy_deep/control")
    rows = E["rows"]
    assert len(rows) == 10000
    pool, _ip, _cp = build_pool(E, 500, 500)
    assert pool == pool_man, "the env's pool is not the trajectory bank's pool"

    # ---- the two contract checks every new regime in this project starts with --------------------
    f0, t0_ = genv_mean(E, [], rows)
    assert round(f0, 8) == VAL_M0_FULL, f"CANONICAL SNAP FAILED: m=0 full is {f0!r}"
    log(f"[snap] m=0 intercept full={f0!r} tail={t0_!r} -- canonical policy-B, exact")
    static_seq = [(int(a), int(b)) for a, b in json.load(open(STATIC_JSON))["sequence"]]
    sf, stl = genv_eval(E, static_seq, rows)
    assert abs(float(np.nanmean(sf)) - STATIC_VAL_FULL) < 1e-12, (
        f"BANKED-HEADLINE ASSERT FAILED: greedy_static_policyB_combined rescores to "
        f"{float(np.nanmean(sf))!r}, banked {STATIC_VAL_FULL}")
    log(f"[banked] greedy_static_policyB_combined K=16 reproduces EXACTLY: "
        f"full={float(np.nanmean(sf))!r} tail={float(np.nanmean(stl))!r}")

    # ---- the frozen tree's own realised interviews, from the trajectory bank ----------------------
    v = np.load(os.path.join(TRAJ_DIR, "val_traj.npz"))
    assert list(np.asarray(v["uid"], np.int64)) == list(np.asarray(rows, np.int64)), \
        "the validation trajectory rows are not the env's rows, in order"
    lab = np.asarray(v["label_idx"], np.int64)              # (10000,16) pool indices
    outc = np.asarray(v["outcome"], np.int64)
    leaf = np.asarray(v["leaf"], np.int64)
    leaf_ids = [str(s) for s in v["leaf_ids"]]
    tree_seqs = [list(lab[u]) for u in range(len(rows))]

    tree_sc = score_sequences(E, rows, pool, tree_seqs, tag="tree")
    tf, tt = tree_sc[TOTAL_L]
    # regression gate: the tree's own validation interviews must reproduce E1's banked per-user vector
    if os.path.exists(E1_PER_USER):
        z = np.load(E1_PER_USER)
        d16 = np.abs(np.asarray(z["policy_full"], np.float64) - tf)
        assert float(np.nanmax(d16)) < 1e-9, (
            f"REGRESSION: the tree's K=16 per-user vector differs from E1's banked one by "
            f"{float(np.nanmax(d16)):.3e}")
        log(f"[regression] tree K=16 per-user vector == E1's banked vector, max |diff| "
            f"{float(np.nanmax(d16)):.2e}")

    # ---- ORDER-INVARIANCE: what a K=16 comparison can and cannot see ------------------------------
    rng = np.random.default_rng(SEED)
    perm_seqs = [list(np.asarray(s)[rng.permutation(TOTAL_L)]) for s in tree_seqs]
    pf, pt = score_sequences(E, rows, pool, perm_seqs, at=(TOTAL_L,), tag="tree/permuted")[TOTAL_L]
    order_max = float(np.nanmax(np.abs(pf - tf)))
    log(f"[order] permuting every user's 16 questions moves full NDCG by at most "
        f"{order_max:.3e} per user -- the fold is "
        f"{'ORDER-INVARIANT' if order_max < 1e-9 else 'ORDER-SENSITIVE'} at K=16")

    # ---- the deeper tree --------------------------------------------------------------------------
    gmap = {(r["leaf"], r["branch_idx"]): r for r in dl["states"]}
    depth = {nid: depth_of(nid) for nid in leaf_ids}
    deep_seqs, n_same_set, gidx_per_user = [], 0, np.full(len(rows), -1, np.int64)
    for u in range(len(rows)):
        nid = leaf_ids[int(leaf[u])]
        d = depth[nid]
        br = int(outc[u, d])
        r = gmap.get((nid, br))
        assert r is not None, f"user {u}: no below-floor state for leaf {nid} branch {BRANCHES[br]}"
        g = int(r["pool_idx"])
        gidx_per_user[u] = g
        s = extend_sequence(tree_seqs[u], d, g)
        n_same_set += int(set(s) == set(tree_seqs[u]))
        deep_seqs.append(s)
    log(f"[deeper] built; {n_same_set} of {len(rows)} users ({n_same_set / len(rows) * 100:.1f}%) get "
        f"the SAME QUESTION SET as the tree (their branch argmax was already in the leaf's tail)")
    deep_sc = score_sequences(E, rows, pool, deep_seqs, tag="deeper")

    # ---- the SHUFFLE control (HARD RULE 5): the 42 branch questions permuted across branches -------
    rs = np.random.default_rng(SEED)
    keys = sorted(gmap)
    vals = [gmap[k]["pool_idx"] for k in keys]
    perm = rs.permutation(len(keys))
    smap = {keys[i]: int(vals[perm[i]]) for i in range(len(keys))}
    n_fixed = sum(1 for i in range(len(keys)) if vals[perm[i]] == vals[i])
    shuf_seqs = []
    for u in range(len(rows)):
        nid = leaf_ids[int(leaf[u])]
        d = depth[nid]
        shuf_seqs.append(extend_sequence(tree_seqs[u], d, smap[(nid, int(outc[u, d]))]))
    log(f"[shuffle] the 42 branch questions permuted across branches ({n_fixed} fixed points)")
    shuf_sc = score_sequences(E, rows, pool, shuf_seqs, at=(TOTAL_L,), tag="deeper/shuffled")

    df, dt = deep_sc[TOTAL_L]
    zf, zt = shuf_sc[TOTAL_L]
    nest = {str(K): {"full_max_abs_diff": float(np.nanmax(np.abs(deep_sc[K][0] - tree_sc[K][0])))}
            for K in SCORE_AT if K in deep_sc}
    out = {"analysis": "policy_phase3_deeper_tree_control", "design_sheet":
           "docs/design/PAPER_B_POLICY_DESIGN.md G.B", "citable": True,
           "split": "validation", "test_touched": False,
           "construction": ("the frozen tree with the min-leaf floor lowered to zero for ONE level: "
                            "at each of the 15 floor-stopped nodes the node's own chosen splitter is "
                            "asked, its ternary answer branches, and each of the 42 branches asks the "
                            "EXACT MEASURED-NDCG argmax at its own state. The parent leaf's greedy "
                            "tail then completes the 16-question budget, minus the new question if it "
                            "was already in it, otherwise minus its weakest (last) pick. The tail is "
                            "NOT re-greedied below the new node -- that is ~33 h on this box and was "
                            "not paid; no sentence may describe this as a fully re-grown deeper tree."),
           "n_users": len(rows), "n_states": len(gmap),
           "n_users_same_question_set_as_tree": int(n_same_set),
           "order_invariance": {"max_abs_per_user_full_diff_under_permutation": order_max,
                                "order_invariant_at_K16": bool(order_max < 1e-9),
                                "why_it_matters": (
                                    "if the fold is order-invariant then at a fixed budget of 16 the "
                                    "deeper tree differs from the tree by exactly ONE question per "
                                    "user -- the swap of the tail's weakest pick for the branch's own "
                                    "argmax. That is the whole size of what a K=16 comparison can "
                                    "measure here, and it must be stated before the number is read.")},
           "static_val": {"full": float(np.nanmean(sf)), "tail": float(np.nanmean(stl))},
           "bank_alone_val": {"combined_minus_items_full": STATIC_VAL_FULL - STATIC_ITEMS_VAL_FULL,
                              "combined_minus_items_tail": STATIC_VAL_TAIL - STATIC_ITEMS_VAL_TAIL,
                              "note": ("bank-alone on the SAME cohort and ruler (F.4's bar). Its TEST "
                                       "value is +0.01430 full / +0.02199 tail; this is its "
                                       "VALIDATION value, and val and test numbers are never "
                                       "compared directly.")},
           "tree_val": {"full": float(np.nanmean(tf)), "tail": float(np.nanmean(tt))},
           "deeper_val": {"full": float(np.nanmean(df)), "tail": float(np.nanmean(dt))},
           "deeper_shuffled_val": {"full": float(np.nanmean(zf)), "tail": float(np.nanmean(zt))},
           "deeper_minus_tree": {"full": paired_bootstrap(df - tf), "tail": paired_bootstrap(dt - tt)},
           "deeper_minus_static": {"full": paired_bootstrap(df - sf),
                                   "tail": paired_bootstrap(dt - stl)},
           "tree_minus_static": {"full": paired_bootstrap(tf - sf), "tail": paired_bootstrap(tt - stl)},
           "shuffled_minus_tree": {"full": paired_bootstrap(zf - tf),
                                   "tail": paired_bootstrap(zt - tt)},
           "deeper_minus_shuffled": {"full": paired_bootstrap(df - zf),
                                     "tail": paired_bootstrap(dt - zt)},
           "nesting_check_by_K": nest,
           "curve_full": {str(K): float(np.nanmean(deep_sc[K][0])) for K in deep_sc},
           "curve_full_tree": {str(K): float(np.nanmean(tree_sc[K][0])) for K in tree_sc},
           "tower": E["tower"], "conc_ckpt": E["conc_ckpt"], "mask_policy": "B",
           "pool_contract": E["pool_contract"], "minutes": round((time.time() - t00) / 60, 2)}
    p = os.path.join(OUTDIR, "policy_deeper_tree_control.json")
    np.savez_compressed(p.replace(".json", "_per_user.npz"),
                        tree_full=tf.astype(np.float32), tree_tail=tt.astype(np.float32),
                        deeper_full=df.astype(np.float32), deeper_tail=dt.astype(np.float32),
                        shuffled_full=zf.astype(np.float32), shuffled_tail=zt.astype(np.float32),
                        static_full=sf.astype(np.float32), static_tail=stl.astype(np.float32),
                        deep_pick=gidx_per_user, rows=np.asarray(rows, np.int64))
    with open(p + ".tmp", "w") as fh:
        json.dump(out, fh, indent=2, default=float)
    os.replace(p + ".tmp", p)
    log(f"[control] deeper tree {out['deeper_val']['full']:.8f} vs tree "
        f"{out['tree_val']['full']:.8f} -> {out['deeper_minus_tree']['full']['mean']:+.5f} "
        f"[{out['deeper_minus_tree']['full']['lo']:+.5f}, "
        f"{out['deeper_minus_tree']['full']['hi']:+.5f}]")
    log(f"[control] wrote {os.path.basename(p)} in {out['minutes']:.1f}m")
    return out


def main():
    ap = argparse.ArgumentParser(description="Workstream E phase 3 (design sheet G.B). See "
                                             "docs/design/PAPER_B_POLICY_DESIGN.md")
    ap.add_argument("--mode", required=True, choices=["labels", "control"])
    ap.add_argument("--split", default="validation", choices=["validation"],
                    help="FIREWALL: validation only. The 10,000 TEST users are never touched here.")
    ap.add_argument("--snapshot", default=None)
    ap.add_argument("--concept_ckpt", default=None)
    ap.add_argument("--full_threads", action="store_true",
                    help="consumed by strategy_channel_suite AT IMPORT (it reads sys.argv)")
    ap.add_argument("--threads", type=int, default=None,
                    help="thread count, consumed by strategy_channel_suite AT IMPORT")
    args = ap.parse_args()
    assert args.split == "validation", "FIREWALL: this file never touches the test cohort"
    if args.mode == "labels":
        return build_labels(args)
    from strategy_channel_suite import CERT_TOWER, CERT_CONC
    args.snapshot = args.snapshot or CERT_TOWER
    args.concept_ckpt = args.concept_ckpt or CERT_CONC
    log("#" * 110)
    log("# WORKSTREAM E PHASE 3 -- THE DEEPER-TREE CONTROL (design sheet G.B).")
    log("# If the deeper tree captures what the policy captures, the honest sentence is")
    log("# 'the policy is a re-parameterisation of a deeper tree' and we claim NOTHING.")
    log("# FIREWALL: the 10,000 canonical TEST users are NOT built, NOT loaded, NOT touched.")
    log("#" * 110)
    return run_control(args)


if __name__ == "__main__":
    main()
