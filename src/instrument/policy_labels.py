r"""WORKSTREAM E / TODO 1.16 -- LABEL EXTRACTION FOR THE PARAMETRIC POLICY.

Design sheet: `docs/design/PAPER_B_POLICY_DESIGN.md`. Read it before running anything here; this file
implements the sheet and does not restate its argument.

WHAT THIS PRODUCES
------------------
E1's supervision: at every state the NDCG tree visits, the question the tree chose -- which is the
EXACT MEASURED-NDCG ARGMAX at that state over the 1,000-candidate pool, the same scalar the split
search computes. **No surrogate anywhere in the label.** 202 distinct states, 16 labels per validation
user, 160,000 (user, position) examples.

TWO MODES, AND THE FIRST ONE IS FREE
------------------------------------
  --mode banked   Assembles E1's label set from artifacts that are ALREADY ON DISK (the tree's node
                  records and its leaves' greedy tails). Scoring cost: ZERO. What it pays for is the
                  GATES -- because a banked number reused without a gate is exactly how the policy-A
                  contamination survived for a month (`banked-global-greedy-is-a-policy-A-artifact`).
  --mode sweep    Pays for full 1,000-candidate exact gain VECTORS at a declared set of states. This
                  is what E2 needs (labels at states the tree never reached) and what the NDCG-vs-DCG
                  normalisation question needs (both readings come off the same per-user vectors).

BOTH NORMALISATIONS, AT ZERO MARGINAL COST (author decision, 2026-08-13)
-----------------------------------------------------------------------
Bilgic & Bennett (SIGIR 2012) find DCG-loss beats NDCG-loss FOR SELECTION, because per-query
normalisation drags selection toward low-IDCG queries. Our criterion is per-USER-normalised, so the
same pathology may apply. For binary relevance

    DCG_u@10 = NDCG_u@10 * IDCG_u@10 ,   IDCG_u@10 = sum_{i<=min(n_u,10)} 1/log2(i+1)

and IDCG_u depends only on the user's target count -- no model call, no second scoring pass. So every
sweep reduces the same per-user vector twice and reports how often the two argmaxes disagree. Measured
on our 10,000 validation users the weights span 1.000-4.544 (sd 1.161) and 12.7% of users hold <= 2
targets, so this is a large structured reweighting, not a rounding difference.

THE REPORTED METRIC DOES NOT CHANGE. HARD RULE 8: the ruler is fixed, full AND tail NDCG@10 under the
certified pair, learned decoder bias, policy-B mask. This is a question about the SELECTION criterion
and (later) the training reward, never about the report.

FIREWALL (non-negotiable, and enforced by the parser, not by a comment)
----------------------------------------------------------------------
`--split` accepts `validation` ONLY. The 10,000 canonical TEST users are never built, never loaded and
never touched by anything in this file. Prior burn: held-out evaluation is what exposed RL gaming in
June 2026, with realisable gain about one fifth of the oracle's.

HARD RULE 1 -- NOTHING IS SAMPLED. Every state uses every one of its users and every candidate in the
declared pool. `--verify_n` chooses only which candidates get RE-VERIFIED against a fresh evaluation;
it never chooses which get used.

HARD RULE 11 -- a contract-signed `*_partial.json` is rewritten after EVERY candidate, so a kill costs
at most one evaluation. Resume refuses across a changed contract via `mask_contract.resume_ok`, which
treats a MISSING field as a refusal rather than a pass. Cached per-user vectors are float32.
"""
import os
import sys
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
sys.path.insert(0, _HERE)

import json
import time
import hashlib
import argparse
import numpy as np

from strategy_channel_suite import (log, build_greedy_env, set_active_rows, genv_eval, genv_mean,
                                    contract_selftest, celf_heap_sig, CERT_TOWER, CERT_CONC,
                                    pool_contract, resume_ok)
from ndcg_tree import build_pool, leak_assert, split_rows, branch_of, BRANCHES

OUTDIR = os.path.join(_ROOT, "experiments", "battery")
TREE_JSON = os.path.join(OUTDIR, "ndcgtree.json")           # structure + node records (15Q build)
TREE16_JSON = os.path.join(OUTDIR, "ndcgtree16.json")       # the leaves' 16-question tails
TREE_HEAPS = os.path.join(OUTDIR, "ndcgtree_heaps")         # internal nodes' complete level-0 vectors
TREE16_HEAPS = os.path.join(OUTDIR, "ndcgtree16_heaps")     # leaves' heaps, extended to 16

TOTAL_L = 16
# MEASURED cost model, fitted over all 22 real tree nodes (`ndcgtree.json` sec_per_eval vs n_users):
# R^2 = 0.985, max residual 0.79 s. NOT a projection -- every point is a node this project actually
# paid for. The root anchor: 8.48 s/eval at n=10,000, i.e. 141.8 min for one 1,000-candidate sweep.
SEC_PER_EVAL_A = 0.0858
SEC_PER_EVAL_B = 0.8157


def sec_per_eval(n_users):
    return SEC_PER_EVAL_A + SEC_PER_EVAL_B * (n_users / 1000.0)


def _sha(o):
    return hashlib.sha1(json.dumps(o, separators=(",", ":")).encode()).hexdigest()


# ============================================================================ the IDCG weights
def idcg_weights(E, rows, tail=False):
    """Per-user IDCG@10 for binary relevance -- the DCG/NDCG bridge, computed with NO model call.

    `DCG_u@10 = NDCG_u@10 * IDCG_u@10` when relevance is binary, and IDCG depends only on how many
    held-out targets the user has. For the TAIL reading the count is over tail targets only (the head
    mask removed), matching exactly what `ndcg10_from_scores` scores. Users with no (tail) target get
    NaN, which is what their NDCG is too, so `nanmean` treats both readings identically -- the tail
    cohort is 9,217 validation / 9,281 test users, and that must not be silently widened to 10,000.
    """
    ctx = E["ctx"]
    ridx = np.asarray(rows, np.int64)
    te = ctx.va_te[ridx]
    if tail:
        te = te.multiply((~ctx.head_mask).astype("float32")[None, :]).tocsr()
        te.eliminate_zeros()
    cnt = np.asarray(te.getnnz(axis=1)).ravel()
    disc = 1.0 / np.log2(np.arange(2, 12))                  # positions 1..10
    cum = np.concatenate([[0.0], np.cumsum(disc)])          # cum[k] = IDCG of k targets, k<=10
    w = cum[np.minimum(cnt, 10)]
    w[cnt == 0] = np.nan
    return w


def dual_means(full, tail, w_full, w_tail):
    """The four cohort scalars, off ONE per-user evaluation. Order fixed and stated once."""
    return (float(np.nanmean(full)), float(np.nanmean(tail)),
            float(np.nanmean(full * w_full)), float(np.nanmean(tail * w_tail)))


# ============================================================================ state enumeration
def load_tree():
    for p in (TREE_JSON, TREE16_JSON):
        assert os.path.exists(p), f"tree artifact missing: {p}"
    tr = json.load(open(TREE_JSON))
    t16 = json.load(open(TREE16_JSON))
    assert t16["total_L"] == TOTAL_L, f"ndcgtree16 total_L is {t16['total_L']}"
    assert t16.get("test_touched") is False
    return tr, t16


def enumerate_states(E, tr, t16, which):
    """The state list. A STATE is (id, position, prefix, rows) -- prefix is the asked list every user
    in `rows` carries, rows is the exact set of validation users sitting there. Nothing is sampled.

      root  -- one state: empty prefix, all 10,000 users. The chapter's most consequential state (its
               argmax is the first question of every arm), and the cheapest full-vector instance that
               answers the NDCG-vs-DCG question.
      tree  -- the 202 states the tree visits at 16 questions: the 7 split states, plus for each leaf
               its 16-depth tail states.
      deep  -- E2's target: the states the tree was FORBIDDEN to reach. For each of the 15 floor-stopped
               nodes, its own chosen splitter's three children become states, INCLUDING the ones that
               broke the 200 floor. `TREE_DEPTH_AND_MINLEAF.md`: all 15 stops were floor stops with a
               positive gain (+0.0067..+0.0425) still in hand, and 8,398 users (84.0%) sit below one.
    """
    nodes = tr["nodes"]
    leaves = {l["nid"]: l for l in t16["leaf_table"]}
    rows_of = {"root": [int(r) for r in E["rows"]]}

    def rows_for(nid):
        """Descend the banked structure, recomputing each ternary partition from the data."""
        if nid in rows_of:
            return rows_of[nid]
        parent, branch = nid.rsplit(".", 1)
        prows = rows_for(parent)
        q = tuple(int(x) for x in nodes[parent]["splitter"])
        kids, ar, _z = split_rows(E, prows, q)
        # G-L3: the recomputed partition must reproduce the banked one EXACTLY.
        banked = nodes[parent]["child_sizes"]
        got = {k: len(v) for k, v in kids.items()}
        assert got == banked, (f"G-L3 PARTITION MISMATCH at {parent}: recomputed {got} != banked "
                               f"{banked} -- the answer labels this process builds are not the ones "
                               f"the tree branched on")
        assert round(ar, 4) == round(float(nodes[parent]["answer_rate"]), 4), \
            f"G-L3 ANSWER-RATE MISMATCH at {parent}: {ar:.4f} vs {nodes[parent]['answer_rate']}"
        for k, v in kids.items():
            rows_of[f"{parent}.{k}"] = v
        return rows_of[nid]

    out = []
    if which == "root":
        out.append({"sid": "root@0", "nid": "root", "pos": 0, "prefix": [],
                    "rows": rows_for("root"), "kind": "split"})
        return out, rows_of
    if which == "tree":
        for nid, r in nodes.items():
            if r["split"]:
                out.append({"sid": f"{nid}@{r['depth']}", "nid": nid, "pos": int(r["depth"]),
                            "prefix": [[int(a), int(b)] for a, b in r["path"]],
                            "rows": rows_for(nid), "kind": "split"})
        for nid, lf in leaves.items():
            d = int(lf["depth"]); seq = [[int(a), int(b)] for a, b in lf["own_sequence"]]
            assert len(seq) == TOTAL_L, f"leaf {nid} carries {len(seq)} questions"
            for k in range(d, TOTAL_L):
                out.append({"sid": f"{nid}@{k}", "nid": nid, "pos": k, "prefix": seq[:k],
                            "rows": rows_for(nid), "kind": "tail"})
        return out, rows_of
    if which == "deep":
        for nid, lf in leaves.items():
            r = nodes[nid]
            assert not r["split"], f"{nid} is not a stopped node"
            q = tuple(int(x) for x in r["splitter"])
            prefix = [[int(a), int(b)] for a, b in r["path"]] + [list(q)]
            kids, _ar, _z = split_rows(E, rows_for(nid), q)
            for br in BRANCHES:
                if not kids[br]:
                    continue                                # an empty branch is not a state
                out.append({"sid": f"{nid}.{br}@{int(r['depth']) + 1}", "nid": f"{nid}.{br}",
                            "pos": int(r["depth"]) + 1, "prefix": prefix, "rows": kids[br],
                            "kind": "below_floor",
                            "floor_broken": len(kids[br]) < int(tr["min_leaf"])})
        return out, rows_of
    raise SystemExit(f"unknown --states {which}")


# ============================================================================ the banked mode
def banked_vectors(nid, is_leaf):
    """The exact level-k gain vectors banked in a node's CELF heap sidecar.

    A heap entry `(gain, q, upd=k)` was measured at the state `prefix + seq[:k]` -- so level-0 entries
    are exact gains at the node's own state and level-k entries at the state after k tail picks. The
    entries POPPED before level k are missing from it, and they are missing NON-RANDOMLY: CELF pops the
    highest-gain candidates first, so a level-k survivor set is enriched for LOW gains. It is therefore
    a legitimate source of exact values and an ILLEGITIMATE 'sample of the pool'. Never call it one.
    """
    d = TREE16_HEAPS if is_leaf else TREE_HEAPS
    p = os.path.join(d, f"node_{nid}.json")
    assert os.path.exists(p), f"heap sidecar missing: {p}"
    b = json.load(open(p))
    lev = {}
    for a, q, u in b["heap"]:
        lev.setdefault(int(u), {})[(int(q[0]), int(q[1]))] = -float(a)
    for k, g in enumerate(b["gains"]):                       # the picked candidate at each level
        lev.setdefault(k, {})[(int(b["seq"][k][0]), int(b["seq"][k][1]))] = float(g)
    return b, lev


def gate_banked_node(E, st, blob, lev, pool, verify_n, rng):
    """G-L1 / G-L2 / G-L4 for one banked state. Exact equality throughout -- no tolerances."""
    set_active_rows(E, st["rows"])
    E["memo"] = {}
    base_now = genv_mean(E, [tuple(q) for q in st["prefix"]], st["rows"])[0]
    assert base_now == float(blob["base"]), (
        f"G-L1 BASE MISMATCH at {st['sid']}: this environment gives {base_now!r}, the banked heap "
        f"says {blob['base']!r}. These are different rulers (policy A m=0 = 0.12987336, policy B = "
        f"0.15686538) -- REFUSING the reuse.")
    g0 = lev.get(0, {})
    cands = [q for q in pool if q in g0]
    assert cands, f"{st['sid']}: no banked level-0 values at all"
    pick = tuple(int(x) for x in blob["seq"][0])
    if len(g0) >= len(pool) - len(st["prefix"]):             # a COMPLETE vector: check the argmax
        arg = max(g0.items(), key=lambda kv: kv[1])[0]
        assert arg == pick, (f"G-L4 ARGMAX MISMATCH at {st['sid']}: the complete banked vector's "
                             f"argmax is {arg}, the tree picked {pick}")
    n = min(verify_n, len(cands))
    for k in rng.choice(len(cands), size=n, replace=False):
        q = cands[int(k)]
        f, _t = genv_eval(E, [tuple(x) for x in st["prefix"]] + [q], st["rows"])
        v = float(np.nanmean(f))
        assert v == base_now + g0[q], (
            f"G-L2 BYTE-IDENTITY FAILED at {st['sid']} candidate {q}: a fresh evaluation gives {v!r}, "
            f"the banked value reconstructs to {base_now + g0[q]!r}")
    return base_now, len(g0), n


def run_banked(E, tr, t16, pool, args, out_path, rng):
    """E1's label set. ZERO scoring cost beyond the gates."""
    states, _rows_of = enumerate_states(E, tr, t16, "tree")
    log(f"[banked] {len(states)} distinct states (the 202 the tree visits at {TOTAL_L} questions)")
    nodes = tr["nodes"]
    leaves = {l["nid"]: l for l in t16["leaf_table"]}

    recs, n_gated, n_verified = [], 0, 0
    for st in states:
        nid, k = st["nid"], st["pos"]
        if st["kind"] == "split":
            label = tuple(int(x) for x in nodes[nid]["splitter"])
            gain = float(nodes[nid]["splitter_gain"])
        else:
            lf = leaves[nid]
            label = (int(lf["own_sequence"][k][0]), int(lf["own_sequence"][k][1]))
            gain = float(lf["gains"][k - int(lf["depth"])])
        assert tuple(label) in set(pool), f"{st['sid']}: label {label} is outside the declared pool"
        assert tuple(label) not in {tuple(q) for q in st["prefix"]}, \
            f"{st['sid']}: the label is already in the prefix (a question asked twice)"
        recs.append({"sid": st["sid"], "nid": nid, "pos": k, "kind": st["kind"],
                     "n_users": len(st["rows"]), "prefix": st["prefix"],
                     "label": [int(label[0]), int(label[1])], "label_gain": gain})

    # --- the gates, on the states whose heap sidecars exist (every node's own state) ---------------
    for st in states:
        if st["pos"] != (int(nodes[st["nid"]]["depth"])):
            continue                                        # a heap's level-0 is the NODE's state
        is_leaf = st["nid"] in leaves
        blob, lev = banked_vectors(st["nid"], is_leaf)
        base, n0, nv = gate_banked_node(E, st, blob, lev, pool, args.verify_n, rng)
        n_gated += 1; n_verified += nv
        log(f"[gate {st['sid']}] G-L1 base {base!r} exact | {n0} banked level-0 values | "
            f"G-L2 {nv} candidates re-evaluated, every one EXACT")

    per_user = []                                            # (row, position, label_index)
    pool_index = {tuple(q): i for i, q in enumerate(pool)}
    for st, rec in zip(states, recs):
        li = pool_index[tuple(rec["label"])]
        for r in st["rows"]:
            per_user.append((int(r), int(rec["pos"]), li))
    per_user = np.asarray(per_user, np.int32)
    assert len(per_user) == len(E["rows"]) * TOTAL_L, \
        f"expected {len(E['rows']) * TOTAL_L} (user, position) examples, built {len(per_user)}"

    npz = out_path.replace(".json", ".npz")
    np.savez_compressed(npz, examples=per_user,
                        pool=np.asarray([[int(a), int(b)] for a, b in pool], np.int32),
                        idcg_full=idcg_weights(E, E["rows"]).astype(np.float32),
                        idcg_tail=idcg_weights(E, E["rows"], tail=True).astype(np.float32),
                        rows=np.asarray(E["rows"], np.int64))
    blob = {"analysis": "policy_labels_E1_banked_argmax", "design_sheet":
            "docs/design/PAPER_B_POLICY_DESIGN.md", "mode": "banked", "citable": True,
            "split": "validation", "test_touched": False,
            "label_definition": ("the EXACT MEASURED-NDCG argmax at each state over the 1,000-candidate "
                                 "pool -- the same scalar the tree's split search computes. No "
                                 "surrogate anywhere in the label."),
            "provenance": {"splitters": os.path.basename(TREE_JSON),
                           "leaf_tails": os.path.basename(TREE16_JSON),
                           "gate_heaps": [os.path.basename(TREE_HEAPS),
                                          os.path.basename(TREE16_HEAPS)]},
            "gates": {"G-L0_pool_contract": E["pool_contract"],
                      "G-L1_base_equality": "exact ==, every gated state",
                      "G-L2_candidates_reverified": n_verified,
                      "G-L3_partition_recompute": "child_sizes + answer_rate exact, every node",
                      "G-L4_argmax_consistency": "asserted where the banked vector is complete",
                      "G-L5_no_test": "split=validation; --split test refused by the parser",
                      "n_states_gated": n_gated},
            "n_states": len(recs), "n_examples": int(len(per_user)),
            "n_distinct_labels": len({tuple(r["label"]) for r in recs}),
            "n_users": len(E["rows"]), "total_L": TOTAL_L,
            "n_pool": len(pool), "pool_sha1": _sha([[int(a), int(b)] for a, b in pool]),
            "tower": E["tower"], "conc_ckpt": E["conc_ckpt"],
            "mask_policy": "B", "pool_contract": E["pool_contract"],
            "states": recs, "per_user_npz": os.path.basename(npz),
            "CAVEAT_label_multiplicity": (
                "160,000 (user, position) examples carry only 202 distinct targets, and only 15 of "
                "them at any position >= 5. Two users in the same leaf at the same position have the "
                "same asked list but DIFFERENT answers, so the policy's input varies while its label "
                "does not. Fitting that perfectly means learning to ignore the answers -- which is "
                "why the anti-collapse gate G-E1c is mandatory, not decorative. E2 exists to break "
                "this: kernel-weighted labels are per-state, so the label count stops being 202."),
            "CAVEAT_selection_cohort": (
                "The tree was GROWN on these same 10,000 validation users, so these labels are "
                "already fitted to them. A user-level dev split does not fix it (a node's label was "
                "computed from all the node's users). See design sheet A.2.1 for the three options "
                "and their costs; the E1 artifact is an instrument check, not an out-of-sample "
                "result.")}
    with open(out_path + ".tmp", "w") as fh:
        json.dump(blob, fh, indent=2, default=float)
    os.replace(out_path + ".tmp", out_path)
    log(f"[banked] wrote {os.path.basename(out_path)} + {os.path.basename(npz)}: "
        f"{len(recs)} states, {len(per_user)} examples, "
        f"{blob['n_distinct_labels']} distinct label questions")
    return blob


# ============================================================================ the sweep mode
def run_sweep(E, tr, t16, pool, args, out_path, rng):
    """Full 1,000-candidate exact gain VECTORS at a declared state set, under BOTH normalisations."""
    states, _rows_of = enumerate_states(E, tr, t16, args.states)
    partial_path = out_path.replace(".json", "_partial.json")
    sig = {"v": 1, "split": E["split"], "objective": "full_and_tail_ndcg_and_dcg",
           "tower": E["tower"], "conc_ckpt": E["conc_ckpt"],
           "mask_pool": pool_contract(E["pool"], E["rows"], policy=E["pool_policy"]),
           "n_pool": len(pool), "pool_sha1": _sha([[int(a), int(b)] for a, b in pool]),
           "states": args.states,
           "state_sha1": _sha(sorted(s["sid"] for s in states))}
    part = json.load(open(partial_path)) if os.path.exists(partial_path) else {}
    if not resume_ok(part.get("sig"), sig, log, os.path.basename(partial_path)):
        part = {"sig": sig, "states": {}}
    part.setdefault("states", {})

    def flush():
        with open(partial_path + ".tmp", "w") as fh:
            json.dump(part, fh, default=float)
        os.replace(partial_path + ".tmp", partial_path)

    t00 = time.time()
    for si, st in enumerate(states):
        sid = st["sid"]
        rec = part["states"].setdefault(sid, {"sid": sid, "nid": st["nid"], "pos": st["pos"],
                                              "kind": st["kind"], "n_users": len(st["rows"]),
                                              "prefix": st["prefix"], "done": False, "vals": {}})
        if rec.get("done"):
            log(f"[sweep {sid}] complete in the sidecar -- reusing")
            continue
        set_active_rows(E, st["rows"]); E["memo"] = {}
        prefix = [tuple(q) for q in st["prefix"]]
        cand = [q for q in pool if tuple(q) not in set(prefix)]
        w_f = idcg_weights(E, st["rows"])
        w_t = idcg_weights(E, st["rows"], tail=True)
        bf, bt = genv_eval(E, prefix, st["rows"])
        base = dual_means(bf, bt, w_f, w_t)
        rec["base"] = list(base)
        rec["idcg_full_mean"] = float(np.nanmean(w_f))
        t0 = time.time()
        for ci, q in enumerate(cand):
            key = f"{int(q[0])}_{int(q[1])}"
            if key in rec["vals"]:
                continue
            f, t = genv_eval(E, prefix + [tuple(q)], st["rows"])
            m = dual_means(f, t, w_f, w_t)
            rec["vals"][key] = [m[i] - base[i] for i in range(4)]   # gains: nF, nT, dF, dT
            flush()                                          # HARD RULE 11(a): after EVERY candidate
            if (ci + 1) % 25 == 0:
                el = time.time() - t0
                log(f"[sweep {sid}] {ci + 1}/{len(cand)} ({el / 60:.1f}m, {el / (ci + 1):.2f}s/eval; "
                    f"model predicted {sec_per_eval(len(st['rows'])):.2f})")
        # --- the two argmaxes, and how often they disagree ---------------------------------------
        ks = list(rec["vals"])
        g = np.asarray([rec["vals"][k] for k in ks], np.float64)
        a_nf, a_df = ks[int(np.argmax(g[:, 0]))], ks[int(np.argmax(g[:, 2]))]
        rec["argmax_ndcg_full"] = a_nf
        rec["argmax_dcg_full"] = a_df
        rec["argmax_agrees"] = bool(a_nf == a_df)
        rec["spearman_ndcg_vs_dcg"] = float(np.corrcoef(
            np.argsort(np.argsort(-g[:, 0])), np.argsort(np.argsort(-g[:, 2])))[0, 1])
        rec["done"] = True
        flush()
        log(f"[sweep {sid}] DONE {len(cand)} candidates in {(time.time() - t0) / 60:.1f}m | "
            f"NDCG argmax {a_nf} vs DCG argmax {a_df} -> "
            f"{'AGREE' if rec['argmax_agrees'] else 'DISAGREE'} "
            f"(rank corr {rec['spearman_ndcg_vs_dcg']:.4f}) | "
            f"{si + 1}/{len(states)} states, {(time.time() - t00) / 60:.1f}m elapsed")

    done = [r for r in part["states"].values() if r.get("done")]
    n_dis = sum(1 for r in done if not r["argmax_agrees"])
    blob = {"analysis": "policy_labels_sweep_dual_normalisation",
            "design_sheet": "docs/design/PAPER_B_POLICY_DESIGN.md", "mode": "sweep",
            "citable": True, "split": "validation", "test_touched": False,
            "states_selector": args.states, "n_states": len(done),
            "gain_vector_layout": ["ndcg_full", "ndcg_tail", "dcg_full", "dcg_tail"],
            "normalisation_note": (
                "DCG_u@10 = NDCG_u@10 * IDCG_u@10 for binary relevance, and IDCG_u depends only on "
                "the user's target count -- so both readings come off the SAME per-user vectors at "
                "zero extra model cost. The REPORTED metric is unchanged (HARD RULE 8): full and "
                "tail NDCG@10."),
            "argmax_disagreement": {"n_states": len(done), "n_disagree": n_dis,
                                    "rate": (n_dis / len(done)) if done else None},
            "tower": E["tower"], "conc_ckpt": E["conc_ckpt"], "mask_policy": "B",
            "pool_contract": E["pool_contract"], "n_pool": len(pool), "sig": sig,
            "states": part["states"], "minutes": round((time.time() - t00) / 60, 2)}
    with open(out_path + ".tmp", "w") as fh:
        json.dump(blob, fh, default=float)
    os.replace(out_path + ".tmp", out_path)
    log(f"[sweep] wrote {os.path.basename(out_path)}: {len(done)} states, NDCG-vs-DCG argmax "
        f"DISAGREE at {n_dis}/{len(done)} states")
    return blob


# ============================================================================ the run
def main():
    ap = argparse.ArgumentParser(description="E1/E2 label extraction. See "
                                             "docs/design/PAPER_B_POLICY_DESIGN.md")
    ap.add_argument("--mode", required=True, choices=["banked", "sweep"])
    ap.add_argument("--states", default="tree", choices=["root", "tree", "deep"],
                    help="sweep only: which states to score exactly")
    ap.add_argument("--split", default="validation", choices=["validation"],
                    help="FIREWALL: validation only. The 10,000 TEST users are never touched by this "
                         "file, and `test` is not an accepted value.")
    ap.add_argument("--snapshot", default=CERT_TOWER)
    ap.add_argument("--concept_ckpt", default=CERT_CONC)
    ap.add_argument("--n_items", type=int, default=500)
    ap.add_argument("--n_concepts", type=int, default=500)
    ap.add_argument("--label", default="e1")
    ap.add_argument("--verify_n", type=int, default=8,
                    help="how many banked candidates per state get RE-EVALUATED from scratch and "
                         "asserted exactly equal. This never selects which candidates are USED.")
    ap.add_argument("--cost_only", action="store_true",
                    help="print the MEASURED-fit cost estimate for the requested state set and exit "
                         "without building an environment")
    ap.add_argument("--full_threads", action="store_true",
                    help="consumed by strategy_channel_suite AT IMPORT (it reads sys.argv). The BLAS "
                         "env vars must ALSO be exported by the launcher BEFORE python starts.")
    ap.add_argument("--threads", type=int, default=None,
                    help="thread count, consumed by strategy_channel_suite AT IMPORT. Overrides "
                         "--full_threads. 4 physical cores on this box: 6 is the measured setting.")
    args = ap.parse_args()

    assert args.split == "validation", "FIREWALL: this file never touches the test cohort"
    t00 = time.time()
    out_path = os.path.join(OUTDIR, f"policy_labels_{args.label}.json")
    os.makedirs(OUTDIR, exist_ok=True)

    if args.cost_only:
        # Structure-only estimate: no environment, no torch, seconds. Uses the banked node records.
        tr, t16 = load_tree()
        nodes = tr["nodes"]; leaves = {l["nid"]: l for l in t16["leaf_table"]}
        plan = []
        if args.states == "root":
            plan = [("root@0", 10000, 1000)]
        elif args.states == "tree":
            for nid, r in nodes.items():
                if r["split"]:
                    plan.append((f"{nid}@{r['depth']}", r["n_users"], 1000 - len(r["path"])))
            for nid, lf in leaves.items():
                for k in range(int(lf["depth"]), TOTAL_L):
                    plan.append((f"{nid}@{k}", lf["n_users"], 1000 - k))
        else:
            for nid, lf in leaves.items():
                r = nodes[nid]
                for br, n in r["child_sizes"].items():
                    if n:
                        plan.append((f"{nid}.{br}", n, 1000 - int(r["depth"]) - 1))
        tot = sum(c * sec_per_eval(n) for _s, n, c in plan)
        log(f"[cost] MEASURED fit s/eval = {SEC_PER_EVAL_A} + {SEC_PER_EVAL_B}*(n/1000) "
            f"(R^2 0.985 over all 22 real tree nodes; root anchor 8.48 s/eval at n=10,000)")
        log(f"[cost] --states {args.states}: {len(plan)} states, "
            f"{sum(c for _s, _n, c in plan)} evaluations, "
            f"{tot / 3600:.1f} h wall (cold, no banked reuse)")
        return {"states": len(plan), "evals": sum(c for _s, _n, c in plan), "hours": tot / 3600}

    log("#" * 110)
    log("# WORKSTREAM E -- LABEL EXTRACTION. Design sheet: docs/design/PAPER_B_POLICY_DESIGN.md")
    log(f"# mode={args.mode} states={args.states} label={args.label}")
    log("# FIREWALL: the 10,000 canonical TEST users are NOT built, NOT loaded, NOT touched.")
    log("# HARD RULE 1: nothing is sampled -- every state uses every user and every candidate.")
    log("#" * 110)

    tr, t16 = load_tree()
    E = build_greedy_env(args.snapshot, "validation", args.concept_ckpt, f"policylabels({args.label})")
    contract_selftest(E, E["rows"])
    leak_assert(E, E["rows"], "validation")
    assert len(E["rows"]) == 10000, f"cohort is {len(E['rows'])} users, expected 10,000"
    assert os.path.basename(args.snapshot) == tr["tower"], \
        f"tower mismatch vs the tree build: {tr['tower']}"
    assert os.path.basename(args.concept_ckpt) == tr["concept_ckpt"], \
        f"concept ckpt mismatch vs the tree build: {tr['concept_ckpt']}"

    pool, item_pool, conc_pool = build_pool(E, args.n_items, args.n_concepts)
    assert len(pool) == tr["n_pool"], f"pool is {len(pool)}, the tree used {tr['n_pool']}"
    # ---- G-L0: the DERIVED identity of the mask, not a label -------------------------------------
    banked_pc = tr.get("pool_contract")
    assert banked_pc, "the tree artifact carries no pool_contract -- an unstated contract is not a " \
                      "matching one (mask_contract.resume_ok's rule)"
    assert E["pool_contract"] == banked_pc, (
        f"G-L0 POOL CONTRACT MISMATCH: this env {E['pool_contract']} vs the tree's {banked_pc}")
    log(f"[G-L0] pool contract identical to the tree build: {json.dumps(banked_pc)}")
    log(f"[pool] {len(item_pool)} items + {len(conc_pool)} concepts = {len(pool)} candidates, the "
        f"tree's own pool, same ids in the same order. NO truncation.")

    rng = np.random.default_rng(98765)
    if args.mode == "banked":
        blob = run_banked(E, tr, t16, pool, args, out_path, rng)
    else:
        blob = run_sweep(E, tr, t16, pool, args, out_path, rng)
    log(f"[done] {(time.time() - t00) / 60:.1f}m")
    return blob


if __name__ == "__main__":
    main()
