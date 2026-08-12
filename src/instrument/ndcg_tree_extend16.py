r"""BUDGET 15 -> 16, AND A POLICY-B STATIC COMPARATOR — label `ndcgtree16`.

Three jobs in one pass, all cheap for one reason: `greedy_certified` persists its lazy CELF heap, so
every search below RESUMES a banked heap instead of re-paying a ~1,000-candidate seed pass.

WHY 16. The 15-question budget was inherited from the genre experiment's turn accounting, where the
genre question consumed one of 16 turns. Nothing in this tree consumes an extra turn, so the budget
should be 16 — `max(BUDGETS)`, and the length the static orders were actually built to. Every
comparison stays question-for-question at equal K; this only stops the budget being arbitrary.

TASK 1 — REBUILD THE STATIC COMBINED ORDER UNDER POLICY B, TO 16 PICKS.
  The banked `greedy_static_i26c.json` combined arm was SELECTED under policy A. Confirmed, not
  inferred: its `intercept_eval` (0.13139/0.02343) is exactly what `MASK_POLICY_B_MIGRATION.md`
  records as the policy-A intercept, against policy B's 0.1586/0.0282. Comparing a policy-B tree
  against a policy-A-selected order put a selection-protocol offset (+0.00093 at K=1) inside every
  K>=2 number.
  The tree's ROOT node holds a completed CELF heap over the full 1,000-candidate combined pool on all
  10,000 validation users with an empty prefix — which IS precisely the seed pass of a policy-B
  static global greedy. So the rebuild costs the lazy re-evaluations only.
  HARD RULE 11(e): the root sidecar is COPIED to this job's own path first. Two runs never share an
  output file, and the tree's own artifact is left byte-for-byte intact.

TASK 2 — EXTEND EACH OF THE 15 LEAF ORDERS FROM 15 TO 16 QUESTIONS, by resuming each leaf's banked
  heap for one more pick. A resumed greedy must reproduce its own prefix exactly; every leaf asserts
  that its first (15 - depth) picks are UNCHANGED, and that the search actually resumed rather than
  silently reseeding cold (a cold reseed would reset the evaluation counter, so the returned count
  must be >= the banked one).

TASK 3 — RE-MEASURE ON VALIDATION at K in {1,2,4,8,16} against the REBUILT comparator, with the old
  policy-A one reported beside it so the weakened-comparator contribution is visible.
  THE GATE: the tree's root splitter and the rebuilt comparator's first question are both chosen by
  the same argmax over the same pool, cohort and objective, so they MUST be identical and the K=1
  difference MUST be exactly 0.0. That is the mechanical nesting guarantee; it was broken only by the
  policy mismatch. Anything else means the rebuild is wrong, and this script stops rather than
  reports.

TEST IS NOT BUILT, NOT LOADED AND NOT TOUCHED. The test look is a separate authorised step.
"""
import os
import sys
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
sys.path.insert(0, _HERE)

import json
import time
import shutil
import argparse
import numpy as np

from strategy_channel_suite import (log, build_greedy_env, set_active_rows, greedy_certified,
                                    genv_eval, genv_answered, contract_selftest,
                                    CERT_TOWER, CERT_CONC)
from ndcg_tree import (build_pool, split_rows, leak_assert, read_global, BRANCHES,
                       HEADLINE_ARM, RULER_NOTE)

OUTDIR = os.path.join(_ROOT, "experiments", "battery")
TREE_JSON = os.path.join(OUTDIR, "ndcgtree.json")
TREE_HEAPS = os.path.join(OUTDIR, "ndcgtree_heaps")
GLOBAL_JSON = os.path.join(OUTDIR, "greedy_static_i26c.json")

LABEL = "ndcgtree16"
STATIC_LABEL = "greedy_static_policyB_combined"
TOTAL_L = 16
BUDGETS = (1, 2, 4, 8, 16)

# The tree's heaps were signed with the env label the tree used. `celf_heap_sig` includes
# `contract_tag` = f"genv:{env_label}:{split}", so this job MUST build its environment under the same
# env label or every resume would be refused as a contract mismatch. That is a wart in the signature
# (a display label leaking into a contract), not a licence to fake anything: the tower, the concept
# checkpoint, the split and the cohort are all identical and are asserted separately below. The
# OUTPUT paths of this job are distinct (HARD RULE 11(e)).
ENV_LABEL = "ndcgtree(val)"

SUPERSEDES = (
    "This order SUPERSEDES the `combined` arm of greedy_static_i26c.json as the static comparator. "
    "That arm was SELECTED under the retired policy-A candidate mask (its intercept_eval "
    "0.13139/0.02343 is the policy-A intercept; policy B is 0.1586/0.0282), so comparing a policy-B "
    "arm against it carried a selection-protocol offset -- measured at +0.00093 full at K=1 -- inside "
    "every K>=2 number. This rebuild uses the SAME pool (500 pop items + 500 answerable concepts, "
    "items first), the SAME objective (cohort-mean held-out full NDCG@10), the SAME 10,000 canonical "
    "validation users and the SAME certified stack, under POLICY B. It reuses the completed CELF heap "
    "the tree's root node built, which is exactly this search's seed pass.")


def _leaf_rows(E, nodes, nid, rows_all):
    """Re-derive a leaf's user set by ROUTING from the root with the banked splitters. Deterministic,
    and cheaper than storing 10,000 row ids per node. The count is asserted against the tree's own
    record, so a silent divergence is impossible."""
    parts = nid.split(".")
    assert parts[0] == "root", nid
    rows = list(rows_all)
    cur = "root"
    for br in parts[1:]:
        assert br in BRANCHES, f"bad branch '{br}' in {nid}"
        q = tuple(int(x) for x in nodes[cur]["splitter"])
        kids, _ar, _z = split_rows(E, rows, q)
        rows = kids[br]
        cur = cur + "." + br
    assert cur == nid
    return rows


def _curve(E, seq, rows, budgets):
    out = {}
    for K in budgets:
        f, t = genv_eval(E, seq[:K], rows)
        ia, ca = genv_answered(E, seq[:K], rows)
        out[str(K)] = {"full@10": float(np.nanmean(f)), "tail@10": float(np.nanmean(t)),
                       "items_answered": round(ia, 3), "concepts_answered": round(ca, 3)}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot", default=CERT_TOWER)
    ap.add_argument("--concept_ckpt", default=CERT_CONC)
    ap.add_argument("--n_items", type=int, default=500)
    ap.add_argument("--n_concepts", type=int, default=500)
    ap.add_argument("--objective", default="full", choices=["full", "tail"])
    ap.add_argument("--total_L", type=int, default=TOTAL_L)
    ap.add_argument("--label", default=LABEL)
    ap.add_argument("--full_threads", action="store_true",
                    help="consumed by strategy_channel_suite AT IMPORT (it reads sys.argv)")
    args = ap.parse_args()

    t00 = time.time()
    OUT = os.path.join(OUTDIR, f"{args.label}.json")
    PARTIAL = os.path.join(OUTDIR, f"{args.label}_partial.json")
    NPZ = os.path.join(OUTDIR, f"{args.label}_per_user.npz")
    HEAPDIR = os.path.join(OUTDIR, f"{args.label}_heaps")
    STATIC_OUT = os.path.join(OUTDIR, f"{STATIC_LABEL}.json")
    os.makedirs(HEAPDIR, exist_ok=True)
    total_L = args.total_L

    for nm, p in (("snapshot", args.snapshot), ("concept_ckpt", args.concept_ckpt),
                  ("tree json", TREE_JSON), ("tree heap dir", TREE_HEAPS),
                  ("global greedy json", GLOBAL_JSON)):
        assert os.path.isabs(p) and os.path.exists(p), f"{nm} missing or not absolute: {p}"

    tree = json.load(open(TREE_JSON))
    assert tree["citable"] and not tree["SANITY"], "the tree artifact is a sanity run"
    assert tree["split"] == "validation" and not tree["test_touched"]
    assert tree["total_L"] == 15, f"tree was built at total_L={tree['total_L']}, expected 15"
    assert tree["tower"] == os.path.basename(args.snapshot)
    assert tree["concept_ckpt"] == os.path.basename(args.concept_ckpt)
    nodes = tree["nodes"]
    leaves_meta = tree["leaf_table"]
    log("#" * 110)
    log(f"# BUDGET {tree['total_L']} -> {total_L}. The TREE STRUCTURE IS NOT REGROWN: same {tree['n_nodes']} "
        f"nodes, same {tree['n_leaves']} leaves, same depth {tree['depth_reached']}. Only each leaf's "
        f"greedy tail gains ONE more question, and the static comparator is rebuilt under POLICY B.")
    log("# " + SUPERSEDES)
    log("# TEST IS NOT BUILT, NOT LOADED, NOT TOUCHED.")
    log("#" * 110)

    E = build_greedy_env(args.snapshot, "validation", args.concept_ckpt, ENV_LABEL)
    assert E["contract_tag"] == f"genv:{ENV_LABEL}:validation", E["contract_tag"]
    rows_all = E["rows"]
    contract_selftest(E, rows_all)
    assert len(rows_all) == tree["n_cohort"], \
        f"cohort drift: {len(rows_all)} now vs {tree['n_cohort']} in the tree"
    leak_assert(E, rows_all, "validation")

    pool, item_pool, conc_pool = build_pool(E, args.n_items, args.n_concepts)
    assert len(item_pool) == tree["n_items_pool"] and len(conc_pool) == tree["n_concepts_pool"]
    comp_old, _gblob = read_global(GLOBAL_JSON, total_L)
    old_seq = comp_old[HEADLINE_ARM]["sequence"][:total_L]
    log(f"[pool] {len(item_pool)} items + {len(conc_pool)} concepts = {len(pool)} candidates "
        f"(identical to the tree's)")

    part = json.load(open(PARTIAL)) if os.path.exists(PARTIAL) else {"static": None, "leaves": {}}

    def flush():
        with open(PARTIAL + ".tmp", "w") as fh:
            json.dump(part, fh, indent=2, default=float)
        os.replace(PARTIAL + ".tmp", PARTIAL)

    # ============================================================ TASK 1: policy-B static combined
    log("=" * 110)
    log("[static] TASK 1 -- rebuilding the static COMBINED order under POLICY B, to 16 picks, by "
        "resuming the CELF heap the tree's ROOT node already built (that heap IS this search's seed "
        "pass: empty prefix, all 10,000 validation users, the full 1,000-candidate pool).")
    src_heap = os.path.join(TREE_HEAPS, "node_root.json")
    dst_heap = os.path.join(HEAPDIR, f"{STATIC_LABEL}_heap.json")
    assert os.path.exists(src_heap), f"root heap missing: {src_heap}"
    if not os.path.exists(dst_heap):
        shutil.copy2(src_heap, dst_heap)           # HARD RULE 11(e): never share an output path
        log(f"[static] COPIED {os.path.relpath(src_heap, _ROOT)} -> "
            f"{os.path.relpath(dst_heap, _ROOT)} (the tree's own sidecar is left untouched)")
    banked_nev = int(json.load(open(dst_heap))["nev"])
    set_active_rows(E, rows_all); E["memo"] = {}; E["n_evals"] = 0; E["t_evals"] = 0.0
    t0 = time.time()

    def on_pick_static(seq, gains, nev):
        part["static"] = {"sequence": [[int(a), int(b)] for a, b in seq],
                          "gains": [float(g) for g in gains], "n_evals": int(nev),
                          "minutes": round((time.time() - t0) / 60, 2), "done": False}
        flush()

    sseq, sgains, snev = greedy_certified(E, rows_all, pool, total_L, args.objective, "staticB",
                                          on_pick=on_pick_static, heap_path=dst_heap)
    assert snev >= banked_nev, (f"the static rebuild did NOT resume: returned {snev} evals vs "
                               f"{banked_nev} banked. A cold reseed would restart the counter.")
    root_q = tuple(int(x) for x in nodes["root"]["splitter"])
    assert tuple(sseq[0]) == root_q, \
        (f"the rebuilt static order opens with {tuple(sseq[0])} but the tree's root splitter is "
         f"{root_q}; both are the same argmax over the same pool/cohort/objective and MUST agree")
    assert abs(sgains[0] - float(nodes["root"]["splitter_gain"])) == 0.0, \
        (f"resume infidelity: first gain {sgains[0]!r} != the tree's banked root gain "
         f"{nodes['root']['splitter_gain']!r}")
    log(f"[static] DONE in {(time.time() - t0) / 60:.1f}m: {snev} cumulative evals "
        f"({snev - banked_nev} NEW; a cold rebuild would have paid {len(pool)} for the seed alone)")
    log(f"[static] policy-B order = {[[int(a), int(b)] for a, b in sseq]}")
    scurve = _curve(E, sseq, rows_all, BUDGETS)
    ocurve = _curve(E, old_seq, rows_all, BUDGETS)
    same_pos = sum(1 for a, b in zip(sseq, old_seq) if tuple(a) == tuple(b))
    static_blob = {
        "analysis": "static_global_greedy_combined_POLICY_B",
        "label": STATIC_LABEL, "citable": True, "supersedes": SUPERSEDES, "ruler_note": RULER_NOTE,
        "split": "validation", "test_touched": False,
        "tower": os.path.basename(args.snapshot), "concept_ckpt": os.path.basename(args.concept_ckpt),
        "objective": args.objective, "item_bank": "pop", "conc_bank": "answerable",
        "n_items_pool": len(item_pool), "n_concepts_pool": len(conc_pool), "n_pool": len(pool),
        "mask_policy": "B", "pool_contract": E["pool_contract"],
        "n_cohort": len(rows_all), "L": total_L,
        "sequence": [[int(a), int(b)] for a, b in sseq], "gains": [float(g) for g in sgains],
        "n_evals": int(snev), "n_evals_new_this_job": int(snev - banked_nev),
        "seed_pass_reused_from": "ndcgtree_heaps/node_root.json (copied, not shared)",
        "curve_policyB": scurve,
        "policyA_order_for_reference": [[int(a), int(b)] for a, b in old_seq],
        "policyA_order_rescored_under_policyB": ocurve,
        "positions_identical_to_policyA_order": same_pos,
        "minutes": round((time.time() - t0) / 60, 2)}
    with open(STATIC_OUT + ".tmp", "w") as fh:
        json.dump(static_blob, fh, indent=2, default=float)
    os.replace(STATIC_OUT + ".tmp", STATIC_OUT)
    part["static"] = dict(static_blob, done=True)
    flush()
    log(f"[static] wrote {os.path.basename(STATIC_OUT)}; {same_pos}/{total_L} positions identical to "
        f"the policy-A order")

    # ============================================================ TASK 2: extend the 15 leaves
    log("=" * 110)
    log(f"[leaves] TASK 2 -- extending {len(leaves_meta)} leaf orders from {tree['total_L']} to "
        f"{total_L} questions, each by RESUMING its own banked heap.")
    leaves = []
    for meta in leaves_meta:
        nid, depth = meta["nid"], int(meta["depth"])
        rows = _leaf_rows(E, nodes, nid, rows_all)
        assert len(rows) == meta["n_users"], \
            f"{nid}: routed {len(rows)} users, tree recorded {meta['n_users']}"
        prefix = [(int(a), int(b)) for a, b in nodes[nid]["path"]]
        assert len(prefix) == depth
        pool_n = [q for q in pool if q not in set(prefix)]
        old_steps = [(int(a), int(b)) for a, b in nodes[nid]["steps"]]
        assert len(old_steps) == tree["total_L"] - depth

        src = os.path.join(TREE_HEAPS, f"node_{nid}.json")
        dst = os.path.join(HEAPDIR, f"node_{nid}.json")
        assert os.path.exists(src), f"leaf heap missing: {src}"
        if not os.path.exists(dst):
            shutil.copy2(src, dst)
        bnev = int(json.load(open(dst))["nev"])
        set_active_rows(E, rows); E["memo"] = {}; E["n_evals"] = 0; E["t_evals"] = 0.0
        t0 = time.time()
        want = total_L - depth
        seq, gains, nev = greedy_certified(E, rows, pool_n, want, args.objective, f"{nid}/ext",
                                           prefix=prefix, heap_path=dst)
        assert len(seq) == want, f"{nid}: got {len(seq)} picks, wanted {want}"
        assert nev >= bnev, (f"{nid}: did NOT resume (returned {nev} evals vs {bnev} banked) -- a "
                             f"cold reseed would restart the counter")
        # THE FIDELITY CHECK the coordinator asked for: a resumed greedy must reproduce its own
        # prefix exactly. If a leaf's first 15-depth picks move, the resume is not faithful.
        assert [tuple(q) for q in seq[:len(old_steps)]] == old_steps, \
            (f"{nid}: RESUME INFIDELITY -- the first {len(old_steps)} picks CHANGED.\n"
             f"  banked  {old_steps}\n  resumed {[tuple(q) for q in seq[:len(old_steps)]]}")
        own = prefix + [(int(a), int(b)) for a, b in seq]
        assert len(own) == total_L
        rec = {"nid": nid, "depth": depth, "n_users": len(rows), "rows": rows,
               "path": [[int(a), int(b)] for a, b in prefix],
               "steps": [[int(a), int(b)] for a, b in seq], "gains": [float(g) for g in gains],
               "own_sequence": [[int(a), int(b)] for a, b in own],
               "new_question": [int(seq[-1][0]), int(seq[-1][1])],
               "new_question_gain": float(gains[-1]),
               "prefix_unchanged": True, "n_evals": int(nev), "n_evals_new_this_job": int(nev - bnev),
               "sec_per_eval": round(E["t_evals"] / max(E["n_evals"], 1), 3),
               "minutes": round((time.time() - t0) / 60, 2),
               "genre_face_validity": meta["genre_face_validity"]}
        leaves.append(rec)
        part["leaves"][nid] = {k: v for k, v in rec.items() if k != "rows"}
        flush()
        log(f"[leaf {nid}] n={len(rows)} d={depth}: +1 question = "
            f"{'concept' if seq[-1][0] else 'item'} {seq[-1][1]} gain {gains[-1]:+.5f} | prefix "
            f"UNCHANGED | {nev - bnev} new evals ({rec['minutes']:.1f}m)")

    # ============================================================ TASK 3: re-measure
    log("=" * 110)
    log(f"[measure] TASK 3 -- validation, K in {list(BUDGETS)}, vs the REBUILT policy-B comparator "
        f"(and the policy-A one beside it).")
    bank = {}
    for rec in leaves:
        nid, rows, own = rec["nid"], rec["rows"], [(int(a), int(b)) for a, b in rec["own_sequence"]]
        set_active_rows(E, rows); E["memo"] = {}
        curve = {}
        for K in BUDGETS:
            a = own[:K]
            assert len(a) == K
            of, ot = genv_eval(E, a, rows)
            bank[f"{nid}|{K}|own_full"] = of.astype(np.float32)
            bank[f"{nid}|{K}|own_tail"] = ot.astype(np.float32)
            row = {"tree_full@10": float(np.nanmean(of)), "tree_tail@10": float(np.nanmean(ot))}
            for arm, seq_c in (("policyB", sseq), ("policyA", old_seq)):
                b = [tuple(q) for q in seq_c[:K]]
                assert len(b) == K, f"UNEQUAL BUDGET: {arm} has {len(b)} at K={K}"
                gf, gt = genv_eval(E, b, rows)
                bank[f"{nid}|{K}|{arm}_full"] = gf.astype(np.float32)
                bank[f"{nid}|{K}|{arm}_tail"] = gt.astype(np.float32)
                row[f"{arm}_full@10"] = float(np.nanmean(gf))
                row[f"diff_full@10_vs_{arm}"] = float(np.nanmean(of - gf))
                row[f"diff_tail@10_vs_{arm}"] = float(np.nanmean(ot - gt))
                row[f"set_overlap_vs_{arm}"] = len(set(a) & set(b))
            curve[str(K)] = row
        bank[f"{nid}|rows"] = np.asarray(rows, np.int64)
        rec["curve_matched"] = curve
        np.savez_compressed(NPZ, **bank)
        c = curve[str(total_L)]
        log(f"[leaf {nid}] n={rec['n_users']} | @K={total_L} vs policyB "
            f"{c['diff_full@10_vs_policyB']:+.5f}/{c['diff_tail@10_vs_policyB']:+.5f} | vs policyA "
            f"{c['diff_full@10_vs_policyA']:+.5f}/{c['diff_tail@10_vs_policyA']:+.5f}")

    weighted = {}
    for K in BUDGETS:
        w = {}
        for arm in ("policyB", "policyA"):
            df = np.concatenate([bank[f"{r['nid']}|{K}|own_full"] - bank[f"{r['nid']}|{K}|{arm}_full"]
                                 for r in leaves])
            dt = np.concatenate([bank[f"{r['nid']}|{K}|own_tail"] - bank[f"{r['nid']}|{K}|{arm}_tail"]
                                 for r in leaves])
            w[f"diff_full@10_vs_{arm}"] = float(np.nanmean(df))
            w[f"diff_tail@10_vs_{arm}"] = float(np.nanmean(dt))
            w["n_users"] = int(np.isfinite(df).sum())
        weighted[str(K)] = w

    # ---- THE GATE ---------------------------------------------------------------------------------
    k1 = weighted["1"]["diff_full@10_vs_policyB"]
    k1t = weighted["1"]["diff_tail@10_vs_policyB"]
    log("=" * 110)
    log(f"[gate] tree root splitter {root_q} vs rebuilt policy-B comparator's first question "
        f"{tuple(sseq[0])}; cohort K=1 difference = {k1!r} full / {k1t!r} tail")
    assert tuple(sseq[0]) == root_q, "root/comparator first question disagree"
    assert k1 == 0.0 and k1t == 0.0, (
        f"THE K=1 TIE IS NOT EXACT ({k1!r}/{k1t!r}). Both arms ask the identical first question on "
        f"the identical users, so every per-user score must be bit-identical. Something in the "
        f"rebuild is wrong -- STOP.")
    log("[gate] PASS: K=1 difference is EXACTLY 0.0 on both axes. The tree is nested in the "
        "comparator at the first question, so every K>=2 difference is conditioning alone -- the "
        "+0.00093 policy-A/B selection offset is gone.")

    # ---- size vs gain -----------------------------------------------------------------------------
    nn = np.array([r["n_users"] for r in leaves], float)
    gg = np.array([r["curve_matched"][str(total_L)]["diff_full@10_vs_policyB"] for r in leaves])
    try:
        from scipy.stats import spearmanr
        rho, pval = spearmanr(nn, gg)
    except Exception:
        rho, pval = float("nan"), float("nan")
    big = nn >= 700
    size_gain = {"spearman_rho": float(rho), "p_value": float(pval),
                 "weighted_gain_leaves_ge_700": float((nn[big] * gg[big]).sum() / nn[big].sum()),
                 "weighted_gain_leaves_lt_700": float((nn[~big] * gg[~big]).sum() / nn[~big].sum()),
                 "n_users_ge_700": int(nn[big].sum()), "n_users_lt_700": int(nn[~big].sum()),
                 "unweighted_leaf_mean": float(gg.mean()),
                 "size_weighted_mean": float((nn * gg).sum() / nn.sum()),
                 "interpretation": (
                     "A NEGATIVE rho is the overfitting signature the genre experiment was built to "
                     "detect: small leaves gaining most means the per-leaf orders are fitting noise "
                     "on the SELECTION cohort. These are in-sample numbers -- the leaves' orders were "
                     "chosen on the very users being scored. Only the TEST cohort settles it.")}

    log("-" * 110)
    log(f"{'leaf':<48}{'d':>3}{'n':>7}{'vsB_full':>11}{'vsB_tail':>11}{'vsA_full':>11}{'+16th':>18}")
    for r in sorted(leaves, key=lambda r: -r["n_users"]):
        c = r["curve_matched"][str(total_L)]
        q = r["new_question"]
        log(f"{r['nid']:<48}{r['depth']:>3}{r['n_users']:>7}"
            f"{c['diff_full@10_vs_policyB']:>11.5f}{c['diff_tail@10_vs_policyB']:>11.5f}"
            f"{c['diff_full@10_vs_policyA']:>11.5f}"
            f"{('concept ' if q[0] else 'item ') + str(q[1]):>18}")
    log("-" * 110)
    for K in BUDGETS:
        w = weighted[str(K)]
        log(f"COHORT-WEIGHTED ({w['n_users']} users) @K={K:<3} vs policy-B "
            f"{w['diff_full@10_vs_policyB']:+.5f}/{w['diff_tail@10_vs_policyB']:+.5f}   "
            f"vs policy-A {w['diff_full@10_vs_policyA']:+.5f}/{w['diff_tail@10_vs_policyA']:+.5f}")
    log(f"[size-gain] spearman(n, gain@K{total_L} vs policy-B) = {rho:.3f} (p={pval:.4f}); "
        f">=700 users {size_gain['weighted_gain_leaves_ge_700']:+.5f} vs <700 "
        f"{size_gain['weighted_gain_leaves_lt_700']:+.5f}")
    log("=" * 110)

    out = {"analysis": "ndcg_objective_interview_tree_budget16_vs_policyB_static",
           "stage": "1b_extend_and_remeasure_on_validation",
           "label": args.label, "citable": True, "split": "validation", "test_touched": False,
           "budget_change": {"from": tree["total_L"], "to": total_L,
                             "why": ("15 was inherited from the genre experiment's turn accounting, "
                                     "where the genre question consumed one of 16 turns. Nothing in "
                                     "this tree consumes an extra turn, so the budget is max(BUDGETS)"
                                     " = 16, the length the static orders were built to."),
                             "tree_regrown": False,
                             "note": ("The tree STRUCTURE is untouched -- same nodes, same splitters, "
                                      "same leaves, same depth. Only each leaf's greedy tail gained "
                                      "one question, by resuming its banked CELF heap.")},
           "tower": os.path.basename(args.snapshot),
           "concept_ckpt": os.path.basename(args.concept_ckpt),
           "objective": args.objective, "mask_policy": "B",
           # DERIVED stamp (2026-08-12 audit): "mask_policy" is a label, this is the mask itself.
           "pool_contract": E["pool_contract"],
           "n_cohort": len(rows_all), "total_L": total_L, "budgets": list(BUDGETS),
           "depth_reached": tree["depth_reached"], "n_nodes": tree["n_nodes"],
           "n_leaves": len(leaves),
           "comparator": {"headline": STATIC_LABEL, "file": os.path.basename(STATIC_OUT),
                          "sequence": [[int(a), int(b)] for a, b in sseq],
                          "gains": [float(g) for g in sgains],
                          "supersedes": SUPERSEDES,
                          "positions_identical_to_policyA": same_pos,
                          "curve_policyB": scurve,
                          "policyA_order_rescored_under_policyB": ocurve},
           "k1_gate": {"root_splitter": [int(x) for x in root_q],
                       "comparator_first_question": [int(x) for x in sseq[0]],
                       "K1_diff_full": k1, "K1_diff_tail": k1t, "exactly_zero": True},
           "leaf_table": [{k: v for k, v in r.items() if k != "rows"} for r in
                          sorted(leaves, key=lambda r: -r["n_users"])],
           "cohort_weighted": weighted, "size_vs_gain": size_gain,
           "per_user_npz": os.path.basename(NPZ),
           "in_sample_warning": (
               "SELECTION-COHORT NUMBERS. The leaves' orders were chosen on the same 10,000 "
               "validation users they are scored on. The TEST cohort has not been built, loaded or "
               "touched by this job."),
           "seconds": round(time.time() - t00, 1)}
    with open(OUT + ".tmp", "w") as fh:
        json.dump(out, fh, indent=2, default=float)
    os.replace(OUT + ".tmp", OUT)
    np.savez_compressed(NPZ, **bank)
    log(f"[done] wrote {os.path.basename(OUT)}, {os.path.basename(STATIC_OUT)} and "
        f"{os.path.basename(NPZ)} in {(time.time() - t00) / 60:.1f}m")
    return out


if __name__ == "__main__":
    main()
