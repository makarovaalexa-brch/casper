r"""TREE ROUTING / SPECIALISATION MEASUREMENT (TODO 1.20, plan D2) -- BANKED ARTIFACTS ONLY.

Pre-registered in docs/design/TREE_ROUTING_SPECIALISATION_DESIGN.md. Nothing here scores the
recommender: the tree structure, every leaf's 15-question order and the ternary partition of all
1,000 pool candidates at the root are already on disk, and the concept definitions come from the
genome file on the tree's own path (relevance >= 0.5, vocab-restricted, >= 30 members,
tags = sorted(members.keys()) -- run_battery_phaseA.load_genome / tradeoff_ledger.build_shared).

Two estimands and two controls; the decision rule is fixed in the design sheet.
"""
import os
import sys
import json
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))

TREE = os.path.join(_ROOT, "experiments", "battery", "ndcgtree.json")
PREFLIGHT = os.path.join(_ROOT, "experiments", "battery", "ndcgtree_PREFLIGHT.json")
GENOME = os.path.join(_ROOT, "data", "movielens", "genome-scores.csv")
USID = os.path.join(_ROOT, "data", "ml-25m", "proc", "unique_sid.txt")
OUT = os.path.join(_ROOT, "experiments", "battery", "tree_routing_specialisation.json")
CACHE = os.path.join(_ROOT, ".cache", "instrument", "genome_relevance_dense.npy")
SEED = 98765
NDRAW = 2000
BRANCHES = ("like", "dislike", "noanswer")


def log(m):
    print(m, flush=True)


# ------------------------------------------------------------------ genome (concept definitions)
def load_genome_dense():
    """Dense (n_items, 1129) relevance matrix, column j = tagId j (col 0 unused). Cached."""
    usid = [int(x) for x in open(USID).read().split()]
    if os.path.exists(CACHE):
        R = np.load(CACHE)
        assert R.shape[0] == len(usid), "cached genome matrix has the wrong item vocabulary"
        return R, usid
    import pandas as pd
    m2s = {m: i for i, m in enumerate(usid)}
    R = np.zeros((len(usid), 1129), np.float32)
    for ch in pd.read_csv(GENOME, chunksize=2_000_000):
        ch = ch[ch["movieId"].isin(m2s)]
        R[ch["movieId"].map(m2s).values.astype(np.int64),
          ch["tagId"].values.astype(np.int64)] = ch["relevance"].values.astype(np.float32)
    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    np.save(CACHE, R)
    return R, usid


def build_vectors(R):
    """Item vector = raw genome row. Concept vector = mean genome row over its members
    (relevance >= 0.5), i.e. the same member set the concept channel is built from."""
    M = R[:, 1:] >= 0.5                                   # (ni, 1128) membership, tagId = col + 1
    n_mem = M.sum(0)
    tags = [t for t in range(1, 1129) if n_mem[t - 1] >= 30]
    assert len(tags) == 1031, f"concept vocabulary drift: {len(tags)} != 1031"
    V_item = R[:, 1:].astype(np.float64)
    V_conc = np.zeros((len(tags), 1128), np.float64)
    for ci, t in enumerate(tags):
        V_conc[ci] = V_item[M[:, t - 1]].mean(0)
    return V_item, V_conc, tags, n_mem


def unit(v):
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.maximum(n, 1e-12)


# ------------------------------------------------------------------ tree structure
def load_tree():
    b = json.load(open(TREE))
    nodes = b["nodes"]
    assert b["citable"] and b["split"] == "validation"
    return b, nodes


def q_key(q):
    return (int(q[0]), int(q[1]))


def subtree_questions(nodes, nid):
    """Every question asked strictly inside the subtree rooted at `nid`, deduplicated, with the
    inherited ancestor path removed. A node's own questions are `steps` (splitter for an internal
    node, the full greedy tail for a leaf)."""
    out, stack = [], [nid]
    while stack:
        n = stack.pop()
        rec = nodes[n]
        for q in rec["steps"]:
            out.append(q_key(q))
        if rec.get("split"):
            stack += [f"{n}.{k}" for k in BRANCHES]
    path = {q_key(q) for q in nodes[nid]["path"]}
    seen, ded = set(), []
    for q in out:
        if q not in path and q not in seen:
            seen.add(q)
            ded.append(q)
    return ded


# ------------------------------------------------------------------ S1 divergence
def s1_divergence(blob, nodes):
    leaves = [lt["nid"] for lt in blob["leaf_table"]]
    seq = {n: [q_key(q) for q in nodes[n]["own_sequence"]] for n in leaves}
    free = {n: [q_key(q) for q in nodes[n]["steps"]] for n in leaves}
    glob = [q_key(q) for q in blob["comparators"]["combined"]["sequence"]]
    full_ovl, free_ovl, posdiff, pairs = [], [], [], []
    for i in range(len(leaves)):
        for j in range(i + 1, len(leaves)):
            a, b = leaves[i], leaves[j]
            fo = len(set(seq[a]) & set(seq[b]))
            ro = len(set(free[a]) & set(free[b]))
            pd_ = sum(1 for k in range(15) if seq[a][k] != seq[b][k])
            full_ovl.append(fo); free_ovl.append(ro); posdiff.append(pd_)
            pairs.append({"a": a, "b": b, "full_overlap": fo, "free_overlap": ro,
                          "free_len_a": len(free[a]), "free_len_b": len(free[b]),
                          "positions_differing": pd_})
    union_all = sorted({q for n in leaves for q in seq[n]})
    union_free = sorted({q for n in leaves for q in free[n]})

    def d(a):
        a = np.asarray(a, np.float64)
        return {"n": int(a.size), "min": float(a.min()), "q25": float(np.percentile(a, 25)),
                "median": float(np.median(a)), "mean": float(a.mean()),
                "q75": float(np.percentile(a, 75)), "max": float(a.max()),
                "hist": {str(int(v)): int((a == v).sum()) for v in np.unique(a)}}
    mean_free_len = float(np.mean([len(free[n]) for n in leaves]))
    return {
        "n_leaves": len(leaves),
        "n_pairs": len(pairs),
        "full_sequence_overlap_of_15": d(full_ovl),
        "free_pick_overlap": d(free_ovl),
        "positions_differing_of_15": d(posdiff),
        "mean_free_pick_length": mean_free_len,
        "union_all_questions": len(union_all),
        "union_free_questions": len(union_free),
        "chance_overlap_random_pool_1000": 15 * 15 / 1000.0,
        "chance_free_overlap_matched_availability": mean_free_len ** 2 / len(union_free),
        "overlap_with_global_static": {n: len(set(seq[n]) & set(glob)) for n in leaves},
        "global_static_in_union": len(set(glob) & set(union_all)),
        "per_pair": pairs,
    }


# ------------------------------------------------------------------ matched control table
def matched_table(pf):
    """Channel + root-answer-rate quintile + root-divisiveness quintile, from the BANKED preflight
    partition of all 1,000 candidates."""
    cand = pf["per_candidate"]
    for c in cand:
        like, dis, _no = c["sizes"]
        c["div"] = min(like, dis) / max(like + dis, 1)
    out = {}
    for ch in (0, 1):
        sub = [c for c in cand if c["chan"] == ch]
        ar = np.asarray([c["answer_rate"] for c in sub])
        dv = np.asarray([c["div"] for c in sub])
        arq = np.quantile(ar, [0.2, 0.4, 0.6, 0.8])
        dvq = np.quantile(dv, [0.2, 0.4, 0.6, 0.8])
        for c in sub:
            c["ar_bin"] = int(np.searchsorted(arq, c["answer_rate"]))
            c["dv_bin"] = int(np.searchsorted(dvq, c["div"]))
        out[ch] = sub
    cells, cells_ar = {}, {}
    for ch, sub in out.items():
        for c in sub:
            cells.setdefault((ch, c["ar_bin"], c["dv_bin"]), []).append(c["id"])
            cells_ar.setdefault((ch, c["ar_bin"]), []).append(c["id"])
    idx = {(c["chan"], c["id"]): c for sub in out.values() for c in sub}
    return cells, cells_ar, idx


def congruence(sim_fn, s, qs):
    if not qs:
        return None
    return float(np.mean([sim_fn(s, q) for q in qs]))


def main():
    rng = np.random.RandomState(SEED)
    R, usid = load_genome_dense()
    V_item, V_conc, tags, n_mem = build_vectors(R)
    U_item, U_conc = unit(V_item), unit(V_conc)
    log(f"[genome] {len(tags)} concepts, {V_item.shape[0]} items, "
        f"{int((R.sum(1) > 0).sum())} items with genome rows")

    def vec(q):
        ch, i = q
        return U_conc[i] if ch == 1 else U_item[i]

    def sim(a, b):
        return float(np.dot(vec(a), vec(b)))

    blob, nodes = load_tree()
    pf = json.load(open(PREFLIGHT))
    cells, cells_ar, cidx = matched_table(pf)
    pool_items = [c["id"] for c in pf["per_candidate"] if c["chan"] == 0]
    pool_conc = [c["id"] for c in pf["per_candidate"] if c["chan"] == 1]
    assert len(pool_items) == 500 and len(pool_conc) == 500
    for q in [q_key(x) for lt in blob["leaf_table"] for x in lt["sequence"]]:
        assert q[1] in (pool_items if q[0] == 0 else pool_conc), f"asked question {q} outside the pool"
    log("[check] every question the tree asks is inside the banked 1,000-candidate pool")

    s1 = s1_divergence(blob, nodes)
    log(f"[S1] free-pick pairwise overlap median {s1['free_pick_overlap']['median']:.2f} "
        f"(mean {s1['free_pick_overlap']['mean']:.2f}, max {s1['free_pick_overlap']['max']:.0f}) of "
        f"~{s1['mean_free_pick_length']:.1f}; chance under matched availability "
        f"{s1['chance_free_overlap_matched_availability']:.2f}")

    # ---------------------------------------------------------------- S2 congruence
    split_nodes = [n for n, r in nodes.items() if r.get("split")]
    split_nodes.sort(key=lambda n: (nodes[n]["depth"], -nodes[n]["n_users"]))
    per_node, widened = [], 0
    for t in split_nodes:
        rec = nodes[t]
        s = q_key(rec["splitter"])
        row = {"nid": t, "depth": rec["depth"], "n_users": rec["n_users"],
               "splitter": list(s), "channel": rec["channel"], "child_sizes": rec["child_sizes"]}
        for b in BRANCHES:
            cid = f"{t}.{b}"
            qs = subtree_questions(nodes, cid)
            nxt = q_key(nodes[cid]["steps"][0])
            row[f"{b}_questions"] = [list(q) for q in qs]
            row[f"{b}_n"] = len(qs)
            row[f"{b}_congruence"] = congruence(sim, s, qs)
            row[f"{b}_next_only"] = sim(s, nxt)
            row[f"{b}_n_concepts"] = sum(1 for q in qs if q[0] == 1)
        row["delta_like_minus_dislike"] = row["like_congruence"] - row["dislike_congruence"]
        per_node.append(row)

    def pooled(key):
        v = [r[key] for r in per_node]
        w = [r["child_sizes"][key.split("_")[0]] for r in per_node] if key.endswith(
            "congruence") else None
        out = {"unweighted_mean": float(np.mean(v)), "per_node": v}
        if w:
            out["user_weighted_mean"] = float(np.average(v, weights=w))
        return out

    s2 = {"per_node": per_node,
          "pooled_like": pooled("like_congruence"),
          "pooled_dislike": pooled("dislike_congruence"),
          "pooled_noanswer": pooled("noanswer_congruence"),
          "pooled_delta_like_minus_dislike": {
              "unweighted_mean": float(np.mean([r["delta_like_minus_dislike"] for r in per_node])),
              "per_node": [r["delta_like_minus_dislike"] for r in per_node]}}
    log(f"[S2] pooled congruence like {s2['pooled_like']['unweighted_mean']:.4f} / dislike "
        f"{s2['pooled_dislike']['unweighted_mean']:.4f} / noanswer "
        f"{s2['pooled_noanswer']['unweighted_mean']:.4f}; "
        f"delta {s2['pooled_delta_like_minus_dislike']['unweighted_mean']:+.4f}")

    # ---------------------------------------------------------------- C1 matched control
    def draw_match(q):
        nonlocal widened
        c = cidx[q]
        key = (q[0], c["ar_bin"], c["dv_bin"])
        pool = cells.get(key, [])
        if len(pool) < 5:
            pool = cells_ar[(q[0], c["ar_bin"])]
            widened += 1
        return (q[0], int(pool[rng.randint(len(pool))]))

    ctrl = {b: [] for b in BRANCHES}
    ctrl_delta = []
    for _ in range(NDRAW):
        vals = {b: [] for b in BRANCHES}
        for r in per_node:
            s = (r["splitter"][0], r["splitter"][1])
            for b in BRANCHES:
                qs = [tuple(q) for q in r[f"{b}_questions"]]
                vals[b].append(float(np.mean([sim(s, draw_match(q)) for q in qs])))
        for b in BRANCHES:
            ctrl[b].append(float(np.mean(vals[b])))
        ctrl_delta.append(float(np.mean(np.asarray(vals["like"]) - np.asarray(vals["dislike"]))))

    def ci(a, obs):
        a = np.asarray(a, np.float64)
        return {"mean": float(a.mean()), "p2.5": float(np.percentile(a, 2.5)),
                "p97.5": float(np.percentile(a, 97.5)),
                "observed": float(obs),
                "observed_minus_control_mean": float(obs - a.mean()),
                "p_two_sided": float((np.sum(np.abs(a - a.mean()) >= abs(obs - a.mean())) + 1)
                                     / (len(a) + 1)),
                "outside_central_95": bool(obs < np.percentile(a, 2.5)
                                           or obs > np.percentile(a, 97.5))}
    c1 = {b: ci(ctrl[b], s2[f"pooled_{b}"]["unweighted_mean"]) for b in BRANCHES}
    c1["delta_like_minus_dislike"] = ci(
        ctrl_delta, s2["pooled_delta_like_minus_dislike"]["unweighted_mean"])
    c1["n_draws"] = NDRAW
    c1["n_widened_cells"] = widened
    log(f"[C1] matched control: like obs {c1['like']['observed']:.4f} vs control "
        f"{c1['like']['mean']:.4f} [{c1['like']['p2.5']:.4f},{c1['like']['p97.5']:.4f}] "
        f"outside95={c1['like']['outside_central_95']} p={c1['like']['p_two_sided']:.4f}")

    # ---------------------------------------------------------------- C2 branch-label shuffle
    sh_delta, sh_like = [], []
    for _ in range(NDRAW):
        dl, lk = [], []
        for r in per_node:
            perm = rng.permutation(3)
            sets = [r[f"{b}_congruence"] for b in BRANCHES]
            p = [sets[i] for i in perm]
            dl.append(p[0] - p[1])
            lk.append(p[0])
        sh_delta.append(float(np.mean(dl)))
        sh_like.append(float(np.mean(lk)))
    c2 = {"delta_like_minus_dislike": ci(
              sh_delta, s2["pooled_delta_like_minus_dislike"]["unweighted_mean"]),
          "like_congruence": ci(sh_like, s2["pooled_like"]["unweighted_mean"]),
          "n_permutations": NDRAW,
          "note": ("within each split node the three children's question sets are permuted across "
                   "the labels {like, dislike, noanswer}; congruence is recomputed against the same "
                   "splitter. Tests whether congruence is tied to WHICH answer made the branch.")}
    log(f"[C2] label shuffle: delta obs {c2['delta_like_minus_dislike']['observed']:+.4f} vs shuffle "
        f"[{c2['delta_like_minus_dislike']['p2.5']:+.4f},"
        f"{c2['delta_like_minus_dislike']['p97.5']:+.4f}] "
        f"p={c2['delta_like_minus_dislike']['p_two_sided']:.4f}")

    # ---------------------------------------------------------------- root, in full + secondary
    root = per_node[0]
    assert root["nid"] == "root"
    t114 = tags[root["splitter"][1]]
    rootsec = {"root_concept_index": root["splitter"][1], "root_tagId": int(t114),
               "root_n_members": int((R[:, t114] >= 0.5).sum())}
    for b in BRANCHES:
        qs = [tuple(q) for q in root[f"{b}_questions"]]
        its = [q[1] for q in qs if q[0] == 0]
        rootsec[f"{b}_direct_relevance_to_root_tag_items"] = {
            "n_items": len(its),
            "mean": float(np.mean([R[i, t114] for i in its])) if its else None}
    allpool_rel = float(np.mean([R[i, t114] for i in pool_items]))
    rootsec["pool_mean_direct_relevance_to_root_tag"] = allpool_rel
    # SECONDARY, and it gets the SAME matched control -- an uncontrolled number is not a result.
    dr_ctrl = {b: [] for b in BRANCHES}
    dr_delta = []
    for _ in range(NDRAW):
        m = {}
        for b in BRANCHES:
            its = [tuple(q) for q in root[f"{b}_questions"] if q[0] == 0]
            m[b] = float(np.mean([R[draw_match(q)[1], t114] for q in its])) if its else np.nan
            dr_ctrl[b].append(m[b])
        dr_delta.append(m["like"] - m["dislike"])
    rootsec["matched_control"] = {
        b: ci(dr_ctrl[b], rootsec[f"{b}_direct_relevance_to_root_tag_items"]["mean"])
        for b in BRANCHES}
    rootsec["matched_control"]["like_minus_dislike"] = ci(
        dr_delta, rootsec["like_direct_relevance_to_root_tag_items"]["mean"]
        - rootsec["dislike_direct_relevance_to_root_tag_items"]["mean"])
    rootsec["control_note"] = ("SECONDARY. Pre-registered as interpretable colour, not as a decision "
                               "statistic: the admissibility rule in the design sheet is the S2 "
                               "congruence pair (C1 + C2), and this number cannot license a routing "
                               "claim on its own. It carries the same matched control anyway.")

    routing_ok = bool(c1["like"]["observed"] > c1["like"]["p97.5"]
                      and c2["delta_like_minus_dislike"]["observed"]
                      > c2["delta_like_minus_dislike"]["p97.5"])
    out = {"analysis": "tree_routing_specialisation", "todo": "1.20", "plan_item": "D2",
           "design_sheet": "docs/design/TREE_ROUTING_SPECIALISATION_DESIGN.md",
           "source_tree": os.path.basename(TREE), "banked_only": True, "rescored": False,
           "seed": SEED, "similarity": ("cosine in the 1,128-dim genome tag-relevance space; item = "
                                        "raw genome row, concept = mean genome row over its members "
                                        "(relevance >= 0.5)"),
           "S1_divergence": s1, "S2_congruence": s2,
           "C1_matched_control": c1, "C2_label_shuffle": c2,
           "root_secondary": rootsec,
           "decision_rule": ("routing admissible only if pooled like-congruence exceeds the C1 "
                             "matched control's 97.5th percentile AND the pooled like-minus-dislike "
                             "delta exceeds the C2 shuffle's 97.5th percentile"),
           "ROUTING_CLAIM_ADMISSIBLE": routing_ok}
    with open(OUT, "w") as fh:
        json.dump(out, fh, indent=2, default=float)
    log(f"[done] ROUTING_CLAIM_ADMISSIBLE = {routing_ok} -> {os.path.basename(OUT)}")
    return out


if __name__ == "__main__":
    main()
