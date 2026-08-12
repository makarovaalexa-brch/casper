r"""TREE DEPTH / MIN-LEAF AUDIT -- PAPER_B_PLAN A10, TODO 1.18 + 1.18b.

WHAT THIS ANSWERS. The NDCG tree (`experiments/battery/ndcgtree.json`) reached depth 5 and stopped.
The chapter must not report that as "conditioning stops paying" until the STOPPING RULE has been
priced, because the rule interacts with answerability in a way that penalises exactly the questions
worth asking:

  (1) ARITHMETIC. A strictly ternary split above a MIN_LEAF=200 floor needs n >= 600 to be possible
      at all; a binary-effective split (a question 100% of the node can answer, so the DID-NOT-ANSWER
      child is empty and the floor does not apply to it) needs n >= 400.
  (2) ANSWERABILITY. A GOOD concept question is one nearly everyone can answer -- which makes its
      did-not-answer branch tiny, so the split fails min-leaf. The recorded anecdote is
      `root.like.noanswer`: 1,256 users, ample to continue, stopped because its argmax splitter
      concept 377 is answered by 97.1% of them, leaving a 37-user branch.

WHAT IS COMPUTED, AND FROM WHAT. Everything here is a LOOKUP. Nothing is scored through the
recommender; no NDCG is computed; the tower is never loaded.

  * node membership              <- `ndcgtree_per_user.npz` (`<leaf>|rows`), internal nodes are the
                                    union of their descendant leaves. The 15 leaf row sets partition
                                    0..9999 exactly, and that is asserted.
  * node records / splitters     <- `ndcgtree.json` (`nodes`)
  * per-candidate node-level gain<- `ndcgtree_heaps/node_<nid>.json`, CELF heap entries at update
                                    LEVEL 0. A level-0 entry's stored value is the exact gain of that
                                    candidate at THIS node's state. Entries at level > 0 were popped
                                    (re-evaluated deeper in the leaf's greedy tail) and their
                                    node-level value is no longer on disk -- but a popped entry was,
                                    at the moment it was popped, the maximum of the heap, so EVERY
                                    degraded candidate has node-level gain >= EVERY intact one. That
                                    gives an exact bracket rather than a guess (see `constrained`).
  * per-user answers             <- RECOMPUTED here from the data, because no artifact banks the
                                    ternary label of every candidate for every user. Items: the
                                    graded validation fold-in (all bands, held-out targets removed) --
                                    absent -> noanswer, level >= LIKE_MIN_LEVEL -> like, else dislike.
                                    Concepts: `signed_answers.signed_values(apply_neg_cap=False)` --
                                    not foldable -> noanswer, signed value > 0 -> like, else dislike.
                                    This is `ndcg_tree.branch_of` verbatim, minus the tower.

THE GATE. The recomputation is not trusted, it is CHECKED against banked ground truth before a single
number is reported:
  G1  all 1,000 root candidates must reproduce `ndcgtree_PREFLIGHT.json`'s `per_candidate` ternary
      sizes AND answer rate EXACTLY (1000/1000);
  G2  all 22 nodes must reproduce their banked `child_sizes` and `answer_rate` for the chosen
      splitter EXACTLY (22/22).
If either fails the script exits non-zero and writes nothing. A failed gate means the labels are not
the harness's labels and no number below is usable.

THE CONTROL. The headline answerability claim ("the min-leaf-feasible subset is less answerable than
the pool it is drawn from") is tested against a permutation null: within each node, reassign the
feasible/infeasible labels at random, same counts, 10,000 draws, seed 98765. Reported per node and
pooled.

OUTPUT: `experiments/battery/tree_minleaf_audit.json`. Read-only w.r.t. every other artifact.
"""
import os
import sys
import json
import time
import argparse

import numpy as np
from scipy import sparse

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_ROOT, "src", "baselines"))

BATT = os.path.join(_ROOT, "experiments", "battery")
TREE = os.path.join(BATT, "ndcgtree.json")
NPZ = os.path.join(BATT, "ndcgtree_per_user.npz")
HEAPS = os.path.join(BATT, "ndcgtree_heaps")
PREFLIGHT = os.path.join(BATT, "ndcgtree_PREFLIGHT.json")
OUT = os.path.join(BATT, "tree_minleaf_audit.json")
GENOME = os.path.join(_ROOT, "data", "movielens", "genome-scores.csv")
PROC = os.path.join(_ROOT, "data", "ml-25m", "proc")

MIN_LEAF = 200
BRANCHES = ("like", "dislike", "noanswer")
NLEV = 10
LIKE_MIN_LEVEL = 7                      # >= 4.0 stars; imported below and asserted equal
SEED = 98765
N_PERM = 10000


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


# ------------------------------------------------------------------ per-user ternary answer labels
def build_labels(n_users_expected=10000):
    """(A_item, L_item, A_conc, L_conc) boolean matrices over the 10k validation cohort.

    A = "answered" (the question is not routed to the DID-NOT-ANSWER branch); L = "like" (only
    meaningful where A). Columns are the FULL catalogue / concept bank; the candidate pool indexes
    into them. No tower, no NDCG, no candidate mask -- the branch rule reads answers only.
    """
    import metrics as M
    from train_tower_t2 import reproduce_partition
    from i26_encoder import LIKE_MIN_LEVEL as _LML
    assert int(_LML) == LIKE_MIN_LEVEL, f"LIKE_MIN_LEVEL drift: {_LML} != {LIKE_MIN_LEVEL}"

    meta = M.load_meta(PROC)
    ni = meta["n_items"]
    train = M.load_train(ni, PROC)
    cnt = np.asarray(train.sum(axis=0)).ravel().astype(np.float64)
    va_tr, va_te = M.load_val(ni, PROC)
    n = va_tr.shape[0]
    assert n == n_users_expected, f"validation cohort is {n} users, expected {n_users_expected}"

    # ---- graded ALL-BANDS validation fold-in, held-out targets removed. This is exactly
    # `run_battery_phaseA.build_real_ctx`'s `ctx.allb` construction, re-expressed here.
    unique_uid, _tr_set, _vd_set, _te_set, _ntr, raw, show2id, _usid = reproduce_partition()
    n_all = len(unique_uid)
    val_ids = unique_uid[n_all - 2 * 10000: n_all - 10000]
    uid2row = {int(u): i for i, u in enumerate(val_ids)}
    vdf = raw[raw["userId"].isin(set(val_ids.tolist()))].copy()
    del raw
    vdf["sid"] = vdf["movieId"].map(show2id)
    vdf = vdf[vdf["sid"].notna()]
    vdf["sid"] = vdf["sid"].astype(np.int64)
    vdf["lvl"] = np.clip(np.rint(vdf["rating"].values * 2).astype(np.int64) - 1, 0, NLEV - 1)
    te_sets = [set(va_te[i].indices.tolist()) for i in range(n)]
    allb = [(np.empty(0, np.int64), np.empty(0, np.int64))] * n
    for uidv, g in vdf.groupby("userId"):
        r = uid2row[int(uidv)]
        keepm = ~g["sid"].isin(te_sets[r]).values
        allb[r] = (g["sid"].values[keepm].astype(np.int64), g["lvl"].values[keepm].astype(np.int64))
    del vdf
    log(f"[labels] graded val fold-in rebuilt: {sum(len(s) for s, _ in allb)} tokens over {n} users")

    A_item = np.zeros((n, ni), bool)
    L_item = np.zeros((n, ni), bool)
    for r, (s, lv) in enumerate(allb):
        A_item[r, s] = True
        L_item[r, s[lv >= LIKE_MIN_LEVEL]] = True

    # ---- concepts: the certified signed-answer path, uncapped (build_greedy_env's own convention)
    import pandas as pd
    from concept_fold import build_member_matrix
    from signed_answers import load_prereg, signed_values
    usid = [int(x) for x in open(os.path.join(PROC, "unique_sid.txt")).read().split()]
    m2s = {m: i for i, m in enumerate(usid)}
    log("[labels] loading genome-scores.csv (relevance >= 0.5, vocab-restricted, >= 30 members)")
    g = pd.read_csv(GENOME, usecols=["movieId", "tagId", "relevance"])
    g = g[g["relevance"] >= 0.5]
    g = g[g["movieId"].isin(m2s)]
    g["sid"] = g["movieId"].map(m2s).astype(np.int64)
    members = {int(t): np.sort(sub["sid"].values) for t, sub in g.groupby("tagId")}
    members = {t: m for t, m in members.items() if len(m) >= 30}
    del g
    tags = sorted(members.keys())
    C = len(tags)
    log(f"[labels] {C} concepts (this must match the tree's concept bank width)")
    Mm = build_member_matrix(members, tags, ni)
    pexp = (cnt @ np.asarray(Mm.todense())) / max(cnt.sum(), 1e-9)
    prereg, item_mean = load_prereg()
    items_l = [np.asarray(s, np.int64) for s, _l in allb]
    stars_l = [((np.asarray(l, np.float64) + 1) / 2).astype(np.float32) for _s, l in allb]
    V, Fb, _Bb, ansm = signed_values(items_l, stars_l, item_mean, Mm, pexp, ni, prereg,
                                     apply_neg_cap=False)
    A_conc = np.asarray(Fb & ansm)
    L_conc = A_conc & (np.asarray(V) > 0)
    log(f"[labels] concept answer rate over the whole bank = {A_conc.mean():.4f}")
    return A_item, L_item, A_conc, L_conc, C


# ------------------------------------------------------------------------------------ tree reading
def node_rows(tree, npz):
    """Row sets for ALL 22 nodes. Leaves are banked; an internal node is the union of the leaves
    beneath it. Asserted against every node's banked `n_users`."""
    leaf_rows = {k.split("|")[0]: np.asarray(npz[k], np.int64) for k in npz.files
                 if k.endswith("|rows")}
    allr = np.concatenate(list(leaf_rows.values()))
    assert len(allr) == len(set(allr.tolist())) == tree["n_cohort"], \
        "the banked leaf row sets do not partition the cohort"
    out = {}
    for nid in tree["nodes"]:
        pref = nid + "."
        rr = [v for k, v in leaf_rows.items() if k == nid or k.startswith(pref)]
        assert rr, f"node {nid} has no descendant leaf rows"
        r = np.sort(np.concatenate(rr))
        assert len(r) == tree["nodes"][nid]["n_users"], \
            f"node {nid}: reconstructed {len(r)} rows vs banked n_users {tree['nodes'][nid]['n_users']}"
        out[nid] = r
    return out


def heap_gains(nid):
    """(exact node-level gains for intact candidates, set of degraded candidates, splitter gain).

    A heap entry at update level 0 was evaluated at THIS node's own state, so its stored value is the
    exact gain of asking that question here. An entry at level > 0 was popped during the leaf's greedy
    tail and re-evaluated at a longer sequence; its node-level value is gone from disk. Because CELF
    only ever pops the current maximum, and every entry is at level 0 until it is first popped, every
    degraded candidate had a node-level gain >= every intact candidate's. That is a bound, not a
    guess, and it is what `constrained` uses.
    """
    blob = json.load(open(os.path.join(HEAPS, f"node_{nid}.json")))
    intact, degraded = {}, set()
    for neg, q, u in blob["heap"]:
        k = (int(q[0]), int(q[1]))
        if int(u) == 0:
            intact[k] = -float(neg)
        else:
            degraded.add(k)
    picks = [(int(a), int(b)) for a, b in blob["seq"]]
    return intact, degraded, picks, [float(x) for x in blob["gains"]]


# --------------------------------------------------------------------------------- the audit itself
def node_table(rows, pool, A_item, L_item, A_conc, L_conc):
    """Ternary sizes of EVERY candidate over one node's users. Pure lookup; no user is dropped."""
    it = np.array([q[1] for q in pool if q[0] == 0], np.int64)
    cc = np.array([q[1] for q in pool if q[0] == 1], np.int64)
    out = {}
    for ids, A, L, ch in ((it, A_item, L_item, 0), (cc, A_conc, L_conc, 1)):
        if not len(ids):
            continue
        Asub = A[np.ix_(rows, ids)]
        Lsub = L[np.ix_(rows, ids)]
        n_ans = Asub.sum(0)
        n_like = Lsub.sum(0)
        n_dis = n_ans - n_like
        n_no = len(rows) - n_ans
        for j, i in enumerate(ids):
            out[(ch, int(i))] = (int(n_like[j]), int(n_dis[j]), int(n_no[j]))
    return out


def perm_p(ar, feasible, n_perm=N_PERM, seed=SEED):
    """Permutation null for 'the feasible subset is less answerable than the pool'.

    Statistic = mean answer rate of the feasible subset minus the mean over the whole pool. Under the
    null, feasibility carries no information about answerability, so reassigning the SAME NUMBER of
    feasible labels at random must reproduce the observed gap. Two-sided p.
    """
    k = int(feasible.sum())
    if k == 0 or k == len(ar):
        return None
    obs = float(ar[feasible].mean() - ar.mean())
    rng = np.random.default_rng(seed)
    null = np.empty(n_perm)
    idx = np.arange(len(ar))
    for b in range(n_perm):
        null[b] = ar[rng.choice(idx, size=k, replace=False)].mean() - ar.mean()
    p = float((np.abs(null) >= abs(obs) - 1e-15).mean())
    return {"observed_gap": obs, "perm_p_two_sided": p, "null_sd": float(null.std()),
            "z": float(obs / max(null.std(), 1e-12)), "n_perm": n_perm}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_perm", type=int, default=N_PERM)
    args = ap.parse_args()
    t0 = time.time()

    tree = json.load(open(TREE))
    assert tree["min_leaf"] == MIN_LEAF and tree["citable"], "unexpected tree artifact"
    npz = np.load(NPZ)
    pre = json.load(open(PREFLIGHT))
    pool = [(int(c["chan"]), int(c["id"])) for c in pre["per_candidate"]]
    assert len(pool) == tree["n_pool"] == 1000, "preflight pool != tree pool width"

    A_item, L_item, A_conc, L_conc, C = build_labels(tree["n_cohort"])
    rowsets = node_rows(tree, npz)

    # ------------------------------------------------------------------ GATE 1: the root, 1000/1000
    root_rows = rowsets["root"]
    tab_root = node_table(root_rows, pool, A_item, L_item, A_conc, L_conc)
    bad = []
    for c in pre["per_candidate"]:
        q = (int(c["chan"]), int(c["id"]))
        got = tab_root[q]
        if list(got) != [int(x) for x in c["sizes"]] or \
                round(got[0] + got[1], 10) != round(c["answer_rate"] * len(root_rows), 10):
            bad.append({"q": list(q), "banked": c["sizes"], "recomputed": list(got)})
    log(f"[GATE 1] root candidates reproducing the banked preflight partition EXACTLY: "
        f"{len(pool) - len(bad)}/{len(pool)}")
    assert not bad, f"GATE 1 FAILED on {len(bad)} candidates, e.g. {bad[:3]}"

    # ------------------------------------------------------------------ GATE 2: every node's splitter
    bad2 = []
    for nid, rec in tree["nodes"].items():
        q = (int(rec["splitter"][0]), int(rec["splitter"][1]))
        r = rowsets[nid]
        s = node_table(r, [q], A_item, L_item, A_conc, L_conc)[q]
        got = {"like": s[0], "dislike": s[1], "noanswer": s[2]}
        ar = round((s[0] + s[1]) / len(r), 4)
        if got != {k: int(v) for k, v in rec["child_sizes"].items()} or ar != round(rec["answer_rate"], 4):
            bad2.append({"nid": nid, "banked": rec["child_sizes"], "recomputed": got,
                         "banked_ar": rec["answer_rate"], "recomputed_ar": ar})
    log(f"[GATE 2] nodes reproducing their banked splitter partition EXACTLY: "
        f"{len(tree['nodes']) - len(bad2)}/{len(tree['nodes'])}")
    assert not bad2, f"GATE 2 FAILED on {bad2}"

    # ---------------------------------------------------------------------------- per-node analysis
    per_node = {}
    for nid, rec in sorted(tree["nodes"].items(), key=lambda kv: (kv[1]["depth"], kv[0])):
        r = rowsets[nid]
        n = len(r)
        prefix = {(int(a), int(b)) for a, b in rec["path"]}
        pool_n = [q for q in pool if q not in prefix]
        tab = tab_root if nid == "root" else node_table(r, pool_n, A_item, L_item, A_conc, L_conc)
        qs = list(pool_n)
        sizes = np.array([tab[q] for q in qs], np.int64)              # (m,3) like/dislike/noanswer
        minc = sizes.min(1)
        feas = minc >= MIN_LEAF
        ar = (sizes[:, 0] + sizes[:, 1]) / n
        chan = np.array([q[0] for q in qs])
        binding = np.argmin(sizes, 1)                                 # which branch is smallest

        intact, degraded, picks, gains = heap_gains(nid)
        splitq = (int(rec["splitter"][0]), int(rec["splitter"][1]))
        g = np.array([intact.get(q, np.nan) for q in qs])
        g[[i for i, q in enumerate(qs) if q == splitq]] = float(rec["splitter_gain"])
        known = np.isfinite(g)
        # best feasible candidate: exact if no DEGRADED candidate is feasible, else bracketed
        feas_known = feas & known
        best_known = float(g[feas_known].max()) if feas_known.any() else None
        deg_feas = [q for q in qs if q in degraded and tab[q] and min(tab[q]) >= MIN_LEAF]
        n_pos_known = int((g[known] > 0).sum())
        # Every degraded candidate was popped at level 0 while it was the heap maximum, so its
        # node-level gain is >= every intact candidate's. That turns "unknown" into a two-sided
        # bracket on what CONSTRAINED SEARCH would have found at this node.
        max_intact = float(g[known].max()) if known.any() else None
        lo = [x for x in (best_known, (max_intact if deg_feas else None)) if x is not None]
        best_feas_lower = max(lo) if lo else None
        best_feas_upper = float(rec["splitter_gain"]) if (deg_feas or (
            best_known is not None and bool(feas[[i for i, q in enumerate(qs)
                                                  if q == splitq][0]]))) else best_known
        # PER-CHILD ROUTING: the split is allowed and any child under the floor inherits the
        # parent's greedy tail (which is exactly what the whole node gets today). A node can
        # continue iff some candidate puts at least TWO children over the floor.
        k_ge = (sizes >= MIN_LEAF).sum(1)
        best_k = int(k_ge.max())
        split_k = int((np.array(tab[splitq]) >= MIN_LEAF).sum())
        big = np.array(tab[splitq])
        routed = int(big[big < MIN_LEAF].sum())

        # arithmetic eligibility
        ternary_possible = n >= 3 * MIN_LEAF
        binary_possible = n >= 2 * MIN_LEAF
        if not rec["split"]:
            if not binary_possible:
                cause = "SIZE_impossible(n<400)"
            elif not ternary_possible:
                cause = "SIZE_ternary_impossible(400<=n<600)"
            else:
                cause = "SHAPE"
        else:
            cause = None

        def _st(v):
            v = np.asarray(v, np.float64)
            return None if not len(v) else {
                "n": int(len(v)), "mean": float(v.mean()), "median": float(np.median(v)),
                "min": float(v.min()), "max": float(v.max()), "p10": float(np.quantile(v, .1)),
                "p90": float(np.quantile(v, .9))}

        per_node[nid] = {
            "depth": rec["depth"], "n_users": n, "split": bool(rec["split"]),
            "stop_reason_banked": rec["stop_reason"], "stop_cause": cause,
            "arith": {"ternary_possible_n>=600": bool(ternary_possible),
                      "binary_effective_possible_n>=400": bool(binary_possible)},
            "splitter": list(splitq), "splitter_channel": rec["channel"],
            "splitter_gain": float(rec["splitter_gain"]),
            "splitter_answer_rate": float(rec["answer_rate"]),
            "splitter_min_child": int(rec["min_child"]),
            "splitter_feasible": bool(min(tab[splitq]) >= MIN_LEAF),
            "n_candidates": len(qs),
            "n_feasible": int(feas.sum()), "frac_excluded": float(1 - feas.mean()),
            "n_feasible_by_channel": {"item": int(feas[chan == 0].sum()),
                                      "concept": int(feas[chan == 1].sum())},
            "n_candidates_by_channel": {"item": int((chan == 0).sum()),
                                        "concept": int((chan == 1).sum())},
            # The floor turns into a WINDOW on the answer rate: a candidate needs its
            # did-not-answer child over the floor (answer rate <= 1 - 200/n) AND both answered
            # children over it (answer rate >= 400/n, and only if the like/dislike split is even).
            "answer_rate_window": {"ceiling_1_minus_minleaf_over_n": 1 - MIN_LEAF / n,
                                   "floor_2x_minleaf_over_n": 2 * MIN_LEAF / n,
                                   "window_is_empty": bool(1 - MIN_LEAF / n < 2 * MIN_LEAF / n),
                                   "n_concepts_above_ceiling": int(
                                       ((chan == 1) & (ar > 1 - MIN_LEAF / n)).sum()),
                                   "n_concepts_below_floor": int(
                                       ((chan == 1) & (ar < 2 * MIN_LEAF / n)).sum()),
                                   "n_items_below_floor": int(
                                       ((chan == 0) & (ar < 2 * MIN_LEAF / n)).sum())},
            "binding_branch_of_infeasible": {
                b: int(((~feas) & (binding == i)).sum()) for i, b in enumerate(BRANCHES)},
            "binding_branch_of_infeasible_concepts": {
                b: int(((~feas) & (binding == i) & (chan == 1)).sum()) for i, b in enumerate(BRANCHES)},
            "answer_rate": {"all": _st(ar), "feasible": _st(ar[feas]),
                            "infeasible": _st(ar[~feas]),
                            "concepts_all": _st(ar[chan == 1]),
                            "concepts_feasible": _st(ar[feas & (chan == 1)]),
                            "items_all": _st(ar[chan == 0]),
                            "items_feasible": _st(ar[feas & (chan == 0)])},
            "answerability_penalty": perm_p(ar, feas, args.n_perm),
            "answerability_penalty_concepts": (
                perm_p(ar[chan == 1], feas[chan == 1], args.n_perm) if (chan == 1).any() else None),
            "constrained": {
                "n_gains_known_exactly": int(known.sum()),
                "n_gains_degraded": int(len(qs) - known.sum()),
                "n_positive_gain_among_known": n_pos_known,
                "best_feasible_gain_known": best_known,
                "n_degraded_and_feasible": len(deg_feas),
                "best_feasible_gain_is_exact": bool(len(deg_feas) == 0),
                "cost_vs_unconstrained_upper_bound": (
                    None if best_feas_lower is None else
                    float(rec["splitter_gain"] - best_feas_lower)),
                "max_intact_gain_any_candidate": max_intact,
                "best_feasible_gain_lower_bound": best_feas_lower,
                "best_feasible_gain_upper_bound": best_feas_upper,
                "would_continue": bool(feas.any())},
            "per_child_routing": {
                "children_over_floor_chosen_splitter": split_k,
                "users_routed_to_parent_tail_chosen_splitter": routed,
                "max_children_over_floor_over_pool": best_k,
                "n_candidates_with_2plus_children_over_floor": int((k_ge >= 2).sum()),
                "would_continue": bool(best_k >= 2)},
        }
        log(f"[node {nid:<46}] d={rec['depth']} n={n:<5} split={int(rec['split'])} "
            f"cause={str(cause):<32} feasible={int(feas.sum()):>4}/{len(qs)} "
            f"AR all={ar.mean():.3f} feas={ar[feas].mean() if feas.any() else float('nan'):.3f}")

    # ------------------------------------------------------------------------------------- pooled
    stopped = [v for v in per_node.values() if not v["split"]]
    counts = {}
    for v in stopped:
        counts[v["stop_cause"]] = counts.get(v["stop_cause"], 0) + 1
    pooled_ar_all, pooled_ar_feas, pooled_gap = [], [], []
    for nid, v in per_node.items():
        pooled_ar_all.append(v["answer_rate"]["all"]["mean"])
        if v["answer_rate"]["feasible"]:
            pooled_ar_feas.append(v["answer_rate"]["feasible"]["mean"])
            pooled_gap.append(v["answerability_penalty"]["observed_gap"])
    out = {
        "analysis": "tree_depth_and_minleaf_audit",
        "todo": "PAPER_B_PLAN A10 / PAPER_B_TODO 1.18 + 1.18b",
        "tree_source": os.path.basename(TREE), "min_leaf": MIN_LEAF,
        "n_nodes": len(per_node), "n_leaves": len(stopped),
        "depth_reached": tree["depth_reached"],
        "gates": {"root_candidates_matching_preflight": len(pool),
                  "nodes_matching_banked_splitter_partition": len(tree["nodes"]),
                  "leaf_rows_partition_cohort": True},
        "stop_cause_counts": counts,
        "stop_cause_counts_by_depth": {
            str(d): {c: sum(1 for v in stopped if v["depth"] == d and v["stop_cause"] == c)
                     for c in sorted(counts)}
            for d in sorted({v["depth"] for v in stopped})},
        "n_stopped_for_want_of_positive_gain": sum(
            1 for v in stopped if v["constrained"]["n_positive_gain_among_known"] == 0
            and v["splitter_gain"] <= 0),
        "pooled": {"mean_answer_rate_all_candidates": float(np.mean(pooled_ar_all)),
                   "mean_answer_rate_feasible": float(np.mean(pooled_ar_feas)),
                   "mean_gap": float(np.mean(pooled_gap)),
                   "n_nodes_with_negative_gap": int(sum(1 for x in pooled_gap if x < 0)),
                   "n_nodes_scored": len(pooled_gap)},
        "alternative_rules": {
            "constrained_search": {
                "stopped_nodes_that_would_continue": sorted(
                    nid for nid, v in per_node.items()
                    if not v["split"] and v["constrained"]["would_continue"]),
                "n_stopped_nodes_with_zero_feasible_candidates": sum(
                    1 for v in stopped if not v["constrained"]["would_continue"]),
                "users_in_stopped_nodes_that_would_continue": sum(
                    v["n_users"] for v in stopped if v["constrained"]["would_continue"])},
            "per_child_routing": {
                "stopped_nodes_that_would_continue": sorted(
                    nid for nid, v in per_node.items()
                    if not v["split"] and v["per_child_routing"]["would_continue"]),
                "n_stopped_nodes_that_would_continue": sum(
                    1 for v in stopped if v["per_child_routing"]["would_continue"]),
                "users_in_stopped_nodes_that_would_continue": sum(
                    v["n_users"] for v in stopped if v["per_child_routing"]["would_continue"]),
                "users_conditioned_further_using_the_chosen_splitter": sum(
                    v["n_users"] - v["per_child_routing"][
                        "users_routed_to_parent_tail_chosen_splitter"]
                    for v in stopped
                    if v["per_child_routing"]["children_over_floor_chosen_splitter"] >= 2),
                "users_routed_to_parent_tail": sum(
                    v["per_child_routing"]["users_routed_to_parent_tail_chosen_splitter"]
                    for v in stopped
                    if v["per_child_routing"]["children_over_floor_chosen_splitter"] >= 2)}},
        "per_node": per_node,
        "seconds": round(time.time() - t0, 1),
    }
    with open(OUT + ".tmp", "w") as fh:
        json.dump(out, fh, indent=2, default=float)
    os.replace(OUT + ".tmp", OUT)
    log(f"[done] wrote {os.path.basename(OUT)} in {(time.time() - t0) / 60:.1f}m")
    log(f"[done] stop causes {counts}; pooled answer-rate gap "
        f"{out['pooled']['mean_gap']:+.4f} ({out['pooled']['n_nodes_with_negative_gap']}/"
        f"{out['pooled']['n_nodes_scored']} nodes negative)")


if __name__ == "__main__":
    main()
