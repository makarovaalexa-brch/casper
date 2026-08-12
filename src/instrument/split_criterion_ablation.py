r"""B1 / TODO 1.12 -- THE SPLIT-CRITERION ABLATION. Label `splitcrit`.

ONE QUESTION: inside OUR tree harness, with everything else held fixed, how much of the NDCG tree's
held-out gain comes from the SPLIT CRITERION being the measured reported metric rather than a
value-free or untruncated surrogate?

HELD FIXED (identical across every arm below, by construction and by assertion):
  * the tree code -- this file's growth loop is `ndcg_tree.py`'s, level-synchronous, and it imports
    `ndcg_tree.split_rows` / `branch_of` so the TERNARY SPLIT RULE is literally the same function;
  * MIN_LEAF = 200 as the only stopping rule, no depth cap;
  * the charged behavioural answers (item -> i26 UNSEEN exposure token, concept -> cv3 refusal token);
  * the certified pair (t2i26c_ep31 + concept_cv3_ep08);
  * the policy-B candidate mask;
  * the 1,000-candidate COMBINED pool (500 popularity-ranked items + 500 answerable concepts, items
    first) -- the same two `_ranked_pool` calls in the same order;
  * the 10,000 canonical VALIDATION users to grow on and the 10,000 disjoint canonical TEST users to
    report on;
  * the 16-question budget, compared question-for-question at K in {1, 2, 4, 8, 16};
  * the static comparator: `greedy_static_policyB_combined`, the policy-B-selected 16-question order.

VARIED: the scalar each candidate question is scored by, and nothing else.
  (a) `variance`   Golbandi WSDM'11 within-node squared error of rating prediction (Eq. 4)
  (b) `pairwise`   Hu & Yu RecSys'13 pairwise rank loss, via their Taylor/diagonal-Hessian gain
  (b') `logistic`  Hu & Yu with the logistic loss -- what their ranking experiments actually ran
  (c) `ndcg`       ours: measured held-out full NDCG@10 of the partition. NOT REGROWN HERE. It is the
                   banked `ndcgtree16` tree; this job re-measures it in the same pass so all four arms
                   are scored by one code path on one ruler, and ASSERTS that the re-measurement
                   reproduces `ndcgtree16test_policyB.json` to the bit.
  (r) `random`, `random_ml`   the shuffle controls (HARD RULE 5).

PRE-REGISTERED INTERPRETATION (fixed before the run, and not revisable after seeing it):
  * (c) > (b) > (a)  -> the mechanism claim is airtight: ranking beats rating, and MEASURING the
                        reported metric beats optimising a smooth surrogate for it.
  * (b) ~= (c)       -> the novelty NARROWS to "the exact reported metric". The field-level claim
                        (value-free / variance surrogates lose) SURVIVES with that nuance stated. This
                        outcome is reported as a narrowing and is NOT marketed away.
  * (b) > (c)        -> a major finding AGAINST us; reported immediately and prominently.

THE THREE SEPARATIONS THIS RUN CAN AND CANNOT TEST.
  (1) truncation/discount: (b) vs (c) tests it only in combination with (2), because (b) is a
      training loss and (c) is a measured value. The clean single-factor test is the queued
      `pairwise_heldout` arm (see DESIGN NOTE below), which is a ~10 h build and is NOT run here.
  (2) training loss vs measured expected value of asking: same caveat, same queued arm.
  (3) sampled negatives vs full catalogue: NOT testable inside our harness at all -- our metric is
      full-catalogue by construction. It is a statement about their EVALUATION and is verified by
      reading, not by running.
  A cheap and exact partial substitute for (1)+(2) IS run here: `--stage corr` correlates each
  criterion's score vector against the MEASURED NDCG marginal gain over all 1,000 candidates at every
  node of the real tree, read out of the banked CELF heap sidecars (whose seed entries are exact
  measured gains at that node's prefix). That is 22 nodes x 1,000 candidates of ground truth for free.

DESIGN NOTE -- THE ARM THAT IS NOT RUN. `pairwise_heldout` would score a candidate by the measured
held-out pairwise rank loss (targets vs the policy-B candidate pool) after folding, i.e. Hu & Yu's
LOSS under OUR protocol. It isolates separation (1) from separation (2) exactly. It costs what (c)
costs -- the banked NDCG tree build was 37,191 s -- because it needs the same per-candidate tower
fold. It is specified here and QUEUED, not run, because the box is committed elsewhere.

HARD RULE 1: no user, item, candidate, pair or negative is sampled, capped or truncated anywhere.
The pairwise loss sums over ALL POS x NEG fold-in pairs. HARD RULE 5: leak check, shuffle control and
canonical snap all run and all must pass before a number is written. HARD RULE 11: every node and
every measured leaf is flushed to disk as it completes.
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
import torch

from strategy_channel_suite import (log, build_greedy_env, set_active_rows, genv_eval, genv_mean,
                                    genv_answered, contract_selftest, CERT_TOWER, CERT_CONC)
from run_battery_phaseA import bootstrap_ci
from ndcg_tree import build_pool, split_rows, branch_of, leak_assert, BRANCHES
import split_criteria as SC

OUTDIR = os.path.join(_ROOT, "experiments", "battery")
STATICB_JSON = os.path.join(OUTDIR, "greedy_static_policyB_combined.json")
NDCGTREE_JSON = os.path.join(OUTDIR, "ndcgtree16.json")
NDCGSTRUCT_JSON = os.path.join(OUTDIR, "ndcgtree.json")
NDCGTEST_JSON = os.path.join(OUTDIR, "ndcgtree16test_policyB.json")

LABEL = "splitcrit"
TOTAL_L = 16
MIN_LEAF = 200
BUDGETS = (1, 2, 4, 8, 16)
BOOT, BOOT_SEED = 10_000, 98765
GROW_ARMS = ("variance", "pairwise", "logistic", "random", "random_ml")


# --------------------------------------------------------------------------- the recommender state
def genv_latent(E, seq, rows, batch=500):
    """The certified pair's per-user latent z_u after asking `seq`. Same fold path `genv_eval` uses --
    the same `_env_item_state` item fold, the same certified concept net with `cap_negatives` over the
    REVEALED set and the same refusal token -- stopping one line before the decoder. Hu & Yu's `u_P`.
    """
    from strategy_channel_suite import _env_item_state
    item_ids = [int(i) for ch, i in seq if ch == 0]
    conc_ids = [int(c) for ch, c in seq if ch == 1]
    Zi, _ = _env_item_state(E, item_ids)
    net = E["net"]
    ridx = np.asarray(rows, np.int64)
    jj = E["posmap"][ridx]
    assert (jj >= 0).all(), "genv_latent called on rows outside the env's active cohort"
    cvals_all = crefs_all = None
    if conc_ids:
        cc = np.asarray(conc_ids, np.int64)
        ok = E["foldable"][np.ix_(ridx, cc)]
        vv = np.where(ok, E["V"][np.ix_(ridx, cc)], 0.0).astype(np.float64)
        s = -np.minimum(vv, 0.0).sum(1)
        sc = np.minimum(1.0, E["cneg"] / np.maximum(s, 1e-9))
        cvals_all = np.where(vv < 0, vv * sc[:, None], vv)
        crefs_all = (~ok).astype(np.float64)
    out = np.empty((len(rows), E["ctx"].d), np.float64)
    with torch.no_grad():
        for st in range(0, len(rows), batch):
            cj = jj[st:st + batch]
            Zb = Zi[torch.from_numpy(cj)]
            if conc_ids:
                B = Zb.shape[0]
                z = net(Zb, [conc_ids] * B, cvals_all[st:st + B].tolist(),
                        crefs_all[st:st + B].tolist() if net.use_refusal else None)
            else:
                z = Zb
            out[st:st + Zb.shape[0]] = z.numpy().astype(np.float64)
    return out


# --------------------------------------------------------------------------- the criterion, one call
def score_candidates(kind, E, ev, memL, memD, pos_of, blocks, cand_idx, seq, rows_for_state, rng,
                     Wd, bd, stats):
    """Score every candidate in `cand_idx` (indices INTO THE POOL) under criterion `kind`, summed over
    the current partition `blocks` (a list of global-row-id lists whose union is the node's users).

    At a NODE there is exactly one block (the node's users), so this reduces to the plain criterion.
    In a LEAF TAIL the blocks are the refinement induced by the tail questions already asked: everyone
    is asked the same next question, but the criterion accounts for the grouping the earlier answers
    already produced. That is the direct analogue of what the NDCG arm's leaf tail does (its marginal
    gain is measured on the leaf's users in their actual post-answer states), and it is what keeps a
    partition criterion from being degenerate when it is not allowed to branch.
    """
    n_c = len(cand_idx)
    if kind in ("random", "random_ml"):
        return rng.random(n_c), {}
    tot = np.zeros(n_c, np.float64)
    info = {}
    # SINGLETON BLOCKS ARE SKIPPED, AND THAT IS EXACT, NOT AN APPROXIMATION.
    #   variance: a one-user block has n_j = 1 for every item it rates, so `s2 - s^2/n` is identically
    #             zero for every child and for the block itself -- it contributes 0 to every candidate.
    #   Hu & Yu: a one-user block always lands ENTIRELY in one branch whichever question is asked, so
    #             its gain is 0.5 * sum_k g_u[k]^2 / (h_u[k] + 2 lam) regardless of the candidate -- a
    #             constant added to all candidates, which cannot move an argmax.
    # It matters because the refining leaf tail produces mostly singletons after a few questions, and
    # the per-block cost is dominated by call overhead rather than by arithmetic.
    live = [b for b in blocks if len(b) > 1]
    info["n_blocks"] = len(blocks)
    info["n_blocks_scored"] = len(live)
    if not live:
        return tot, info
    if kind == "variance":
        for b in live:
            sub = np.asarray([pos_of[int(r)] for r in b], np.int64)
            s, e2n = SC.variance_scores(ev, memL, memD, sub, cand_idx)
            tot += s
            info["e2_node"] = info.get("e2_node", 0.0) + e2n
        return tot, info
    # ---- Hu & Yu: one latent fold + one gradient pass over ALL the node's users, then masked sums
    all_rows = [int(r) for b in live for r in b]
    Z = genv_latent(E, seq, all_rows)
    sub_all = np.asarray([pos_of[r] for r in all_rows], np.int64)
    t0 = time.time()
    G, H, npair = SC.loss_grad_hess(kind, Z, Wd, bd, ev["pos"], ev["neg"], sub_all)
    info["grad_seconds"] = round(time.time() - t0, 1)
    info["n_terms"] = npair
    off = 0
    nclip = 0
    for b in live:
        k = len(b)
        sub = sub_all[off:off + k]
        s, nc = SC.huyu_scores(G[off:off + k], H[off:off + k], memL, memD, sub, cand_idx)
        tot += s; nclip += nc; off += k
    info["hessian_negative_clips"] = nclip
    stats.setdefault("grad_calls", 0)
    stats["grad_calls"] += 1
    return tot, info


# --------------------------------------------------------------------------- growth
def grow(kind, E, ev, memL, memD, pos_of, pool, rows_all, total_L, min_leaf, seed, out_path, Wd, bd,
         minleaf_mode="all3"):
    """Level-synchronous growth, structurally identical to `ndcg_tree.py`'s. Returns (nodes, leaves)."""
    rng = np.random.default_rng(seed)
    stats = {}
    nodes, leaves = {}, []
    frontier = [{"nid": "root", "depth": 0, "path": [], "rows": [int(r) for r in rows_all],
                 "branch_from": None}]
    pool_pos = {tuple(q): i for i, q in enumerate(pool)}
    t00 = time.time()
    while frontier:
        nxt = []
        for nd in frontier:
            nid, depth, rows = nd["nid"], nd["depth"], nd["rows"]
            remaining = total_L - depth
            assert remaining >= 1, f"node {nid} at depth {depth} has no budget left"
            prefix = [(int(a), int(b)) for a, b in nd["path"]]
            cand = [pool_pos[tuple(q)] for q in pool if tuple(q) not in set(prefix)]
            t0 = time.time()
            sc, info = score_candidates(kind, E, ev, memL, memD, pos_of, [rows], cand, prefix,
                                        rows, rng, Wd, bd, stats)
            if kind == "random_ml":
                # the FAIR shuffle control: draw uniformly among candidates whose ternary split would
                # actually clear min-leaf at this node, so the control is a random CHOICE OF QUESTION
                # rather than a random failure to split.
                ok = []
                for j, ci in enumerate(cand):
                    kids, _a, _z = split_rows(E, rows, tuple(pool[ci]))
                    if min(len(v) for v in kids.values()) >= min_leaf:
                        ok.append(j)
                if ok:
                    sc = np.full(len(cand), -np.inf); sc[np.asarray(ok)] = rng.random(len(ok))
            best = int(np.argmax(sc))
            q = tuple(int(x) for x in pool[cand[best]])
            kids, ar, n_zero = split_rows(E, rows, q)
            sizes = {k: len(v) for k, v in kids.items()}
            # MIN-LEAF, AND THE ONE PLACE THIS ABLATION HAS A CHOICE TO DECLARE.
            #   `all3`     -- the floor applies to all three branches, INCLUDING an empty
            #                 DID-NOT-ANSWER branch. This is `ndcg_tree.py`'s rule verbatim and it is
            #                 the PRIMARY mode, because the brief requires min-leaf 200 held fixed and
            #                 arm (c)'s tree cannot be regrown under any other rule.
            #   `nonempty` -- the floor applies only to branches that actually receive users, so a
            #                 question EVERYONE can answer splits binarily instead of being refused.
            #                 Reported as a ROBUSTNESS CHECK among the cheap arms only, because
            #                 Golbandi's own selector constrains only the like and hate branches
            #                 (`golbandi_tree_selector._best_split`: `(cL >= min) & (cH >= min)`),
            #                 and because a criterion that prefers fully answerable questions is
            #                 otherwise penalised by a rule that has nothing to do with its objective.
            live = [v for v in sizes.values() if v > 0] if minleaf_mode == "nonempty"                 else list(sizes.values())
            small = min(live) if live else 0
            will_split = (remaining >= 2) and (small >= min_leaf) and (len(live) >= 2)
            rec = {"nid": nid, "depth": depth, "n_users": len(rows),
                   "path": [[int(a), int(b)] for a, b in prefix], "branch_from": nd["branch_from"],
                   "remaining_budget": remaining, "splitter": [q[0], q[1]],
                   "channel": "concept" if q[0] else "item",
                   "splitter_score": float(sc[best]), "child_sizes": sizes,
                   "answer_rate": round(ar, 4), "n_zero_value_answers": n_zero,
                   "min_child": int(small), "split": bool(will_split),
                   "minleaf_mode": minleaf_mode, "n_live_branches": len(live),
                   "stop_reason": (None if will_split else
                                   ("budget_exhausted (1 question left, nothing to split into)"
                                    if remaining < 2 else
                                    f"min_leaf({minleaf_mode}): smallest live child {small} < "
                                    f"{min_leaf} (live branches {len(live)})")),
                   "criterion_info": info,
                   "score_top10": [[int(pool[cand[i]][0]), int(pool[cand[i]][1]), float(sc[i])]
                                   for i in np.argsort(-sc)[:10]],
                   "seconds": round(time.time() - t0, 1)}
            log(f"[{kind}/{nid}] d{depth} n={len(rows)} -> {rec['channel']} {q[1]} "
                f"(score {sc[best]:+.6g}, answer rate {ar:.3f}); like={sizes['like']} "
                f"dislike={sizes['dislike']} noanswer={sizes['noanswer']}; "
                + ("SPLIT" if will_split else f"STOP ({rec['stop_reason']})")
                + f" [{rec['seconds']}s]")
            if will_split:
                rec["steps"] = [[q[0], q[1]]]
            else:
                # ---- the leaf tail: the SAME criterion, applied to a refining partition, until the
                # 16-question budget is spent. No branching -- everyone in the leaf is asked the same
                # question -- so min-leaf does not apply here.
                asked = [q]
                blocks = [v for v in (kids[k] for k in BRANCHES) if v]
                while len(asked) < remaining:
                    seq = prefix + asked
                    cand2 = [pool_pos[tuple(x)] for x in pool
                             if tuple(x) not in set(prefix) | set(asked)]
                    s2, _i2 = score_candidates(kind, E, ev, memL, memD, pos_of, blocks, cand2, seq,
                                               rows, rng, Wd, bd, stats)
                    qq = tuple(int(x) for x in pool[cand2[int(np.argmax(s2))]])
                    asked.append(qq)
                    nb = []
                    for b in blocks:
                        kk, _a, _z = split_rows(E, b, qq)
                        nb.extend([v for v in (kk[x] for x in BRANCHES) if v])
                    blocks = nb
                assert len(asked) == remaining
                rec["steps"] = [[int(a), int(b)] for a, b in asked]
                rec["n_tail_blocks"] = len(blocks)
                log(f"[{kind}/{nid}] leaf tail complete: {len(asked)} question(s), "
                    f"{len(blocks)} refinement blocks, order {asked}")
            nodes[nid] = rec
            with open(out_path + ".partial", "w") as fh:
                json.dump({"criterion": kind, "nodes": nodes}, fh, indent=2, default=float)
            if will_split:
                cpath = rec["path"] + [[q[0], q[1]]]
                for k in BRANCHES:
                    if minleaf_mode == "nonempty" and not kids[k]:
                        continue          # an empty branch is not a child; the split is binary
                    nxt.append({"nid": f"{nid}.{k}", "depth": depth + 1, "path": cpath,
                                "rows": kids[k], "branch_from": k})
            else:
                own = [(int(a), int(b)) for a, b in rec["path"]] + \
                      [(int(a), int(b)) for a, b in rec["steps"]]
                assert len(own) == total_L, f"leaf {nid} has {len(own)} questions, expected {total_L}"
                nd["own"] = own; nd["rec"] = rec
                rec["own_sequence"] = [[a, b] for a, b in own]
                leaves.append(nd)
        d = frontier[0]["depth"]
        log(f"[{kind}/level] depth {d}: {len(frontier)} node(s) -> {len(nxt)}; {len(leaves)} leaves; "
            f"elapsed {(time.time() - t00) / 60:.1f}m")
        frontier = nxt
    return nodes, leaves, stats


# --------------------------------------------------------------------------- measurement
def route(E, nodes, nid, rows_all):
    """Walk a frozen tree with its banked splitters. Works for either min-leaf mode: a branch that
    the grower dropped as empty simply has no node id, so it is never routed to."""
    parts = nid.split(".")
    assert parts[0] == "root", nid
    rows, cur = list(rows_all), "root"
    for br in parts[1:]:
        assert br in BRANCHES, f"bad branch '{br}' in {nid}"
        kids, _a, _z = split_rows(E, rows, tuple(int(x) for x in nodes[cur]["splitter"]))
        rows = kids[br]; cur = cur + "." + br
    assert cur == nid
    return rows


def measure(E, tag, arm, leaf_orders, nodes, rows_all, seq_static, npz_bank):
    """Score one arm's 16-question-per-user interview on one cohort, per leaf, at every budget."""
    per_leaf = {}
    assigned, seen = 0, set()
    for nid in sorted(leaf_orders, key=lambda n: n):
        rows = route(E, nodes, nid, rows_all)
        assigned += len(rows)
        s = set(rows); assert not (seen & s), f"{arm}: routing is not disjoint"
        seen |= s
        per_leaf[nid] = rows
    assert assigned == len(rows_all), f"{arm}: routing lost users ({assigned} vs {len(rows_all)})"
    curves = {}
    for nid, rows in per_leaf.items():
        if not rows:
            curves[nid] = {}
            continue
        own = [tuple(q) for q in leaf_orders[nid]]
        set_active_rows(E, rows); E["memo"] = {}
        c = {}
        for K in BUDGETS:
            a = own[:K]; b = seq_static[:K]
            assert len(a) == len(b) == K, f"UNEQUAL BUDGET at K={K}"
            of, ot = genv_eval(E, a, rows)
            gf, gt = genv_eval(E, b, rows)
            npz_bank[f"{tag}|{arm}|{nid}|{K}|own_full"] = of.astype(np.float32)
            npz_bank[f"{tag}|{arm}|{nid}|{K}|own_tail"] = ot.astype(np.float32)
            npz_bank[f"{tag}|{arm}|{nid}|{K}|st_full"] = gf.astype(np.float32)
            npz_bank[f"{tag}|{arm}|{nid}|{K}|st_tail"] = gt.astype(np.float32)
            ia, ca = genv_answered(E, a, rows)
            c[str(K)] = {"own_full@10": float(np.nanmean(of)), "own_tail@10": float(np.nanmean(ot)),
                         "static_full@10": float(np.nanmean(gf)),
                         "static_tail@10": float(np.nanmean(gt)),
                         "diff_full@10": float(np.nanmean(of - gf)),
                         "diff_tail@10": float(np.nanmean(ot - gt)),
                         "items_answered": round(ia, 3), "concepts_answered": round(ca, 3),
                         "set_overlap_vs_static": len(set(a) & set(b))}
        npz_bank[f"{tag}|{arm}|{nid}|rows"] = np.asarray(rows, np.int64)
        curves[nid] = c
        kk = str(max(BUDGETS))
        log(f"[{tag}/{arm}/{nid}] n={len(rows)} @K={kk} {c[kk]['diff_full@10']:+.5f} full / "
            f"{c[kk]['diff_tail@10']:+.5f} tail")
    set_active_rows(E, rows_all)
    return per_leaf, curves


def cohort_table(npz_bank, tag, arm, nids):
    out = {}
    for K in BUDGETS:
        use = [n for n in nids if f"{tag}|{arm}|{n}|{K}|own_full" in npz_bank]
        of = np.concatenate([npz_bank[f"{tag}|{arm}|{n}|{K}|own_full"] for n in use])
        ot = np.concatenate([npz_bank[f"{tag}|{arm}|{n}|{K}|own_tail"] for n in use])
        gf = np.concatenate([npz_bank[f"{tag}|{arm}|{n}|{K}|st_full"] for n in use])
        gt = np.concatenate([npz_bank[f"{tag}|{arm}|{n}|{K}|st_tail"] for n in use])
        df, dt = of - gf, ot - gt
        mf, cif, nf = bootstrap_ci(df, n_boot=BOOT, seed=BOOT_SEED)
        mt, cit, _ = bootstrap_ci(dt, n_boot=BOOT, seed=BOOT_SEED)
        out[str(K)] = {"arm_full@10": float(np.nanmean(of)), "arm_tail@10": float(np.nanmean(ot)),
                       "static_full@10": float(np.nanmean(gf)),
                       "static_tail@10": float(np.nanmean(gt)),
                       "diff_full@10": mf, "ci95_full": list(cif),
                       "diff_tail@10": mt, "ci95_tail": list(cit),
                       "separates_full": bool(cif[0] > 0 or cif[1] < 0),
                       "separates_tail": bool(cit[0] > 0 or cit[1] < 0),
                       "n_users": int(nf),
                       "maxabs_diff_full": float(np.abs(df[~np.isnan(df)]).max()),
                       "maxabs_diff_tail": float(np.abs(dt[~np.isnan(dt)]).max())}
    return out


# --------------------------------------------------------------------------- structure + overlap
def structure(kind, nodes, leaves, total_L):
    lv = [nodes[l["nid"]] for l in leaves]
    depth = max(r["depth"] for r in nodes.values())
    hist = {}
    for r in lv:
        hist[str(r["depth"])] = hist.get(str(r["depth"]), 0) + 1
    chan = {}
    for r in nodes.values():
        if r.get("split"):
            c = chan.setdefault(str(r["depth"]), {"item": 0, "concept": 0})
            c[r["channel"]] += 1
    qs = sorted({tuple(q) for l in leaves for q in nodes[l["nid"]]["own_sequence"]})
    splitters = sorted({tuple(r["splitter"]) for r in nodes.values() if r.get("split")})
    n_it = sum(1 for l in leaves for q in nodes[l["nid"]]["own_sequence"] if q[0] == 0)
    n_cc = sum(1 for l in leaves for q in nodes[l["nid"]]["own_sequence"] if q[0] == 1)
    return {"criterion": kind, "depth_reached": depth, "n_nodes": len(nodes), "n_leaves": len(leaves),
            "leaf_depth_histogram": hist, "split_channel_by_depth": chan,
            "root_splitter": nodes["root"]["splitter"],
            "root_channel": nodes["root"].get("channel", "concept" if nodes["root"]["splitter"][0] else "item"),
            "root_answer_rate": nodes["root"].get("answer_rate"),
            "root_child_sizes": nodes["root"].get("child_sizes"),
            "distinct_questions_asked": [[a, b] for a, b in qs],
            "n_distinct_questions": len(qs),
            "splitters": [[a, b] for a, b in splitters],
            "question_slots_items": n_it, "question_slots_concepts": n_cc,
            "leaf_sizes": {l["nid"]: nodes[l["nid"]]["n_users"] for l in leaves}}


# --------------------------------------------------------------------------- the correlation exhibit
def corr_stage(E, ev, memL, memD, pos_of, pool, rows_all, nodes_ndcg, heapdir, Wd, bd, out):
    """CRITERION vs MEASURED NDCG, over all 1,000 candidates, at every node of the REAL tree.

    The banked CELF heap sidecar of each node holds the SEED pass of that node's split search: one
    entry per candidate, -(gain), where gain is the exact measured cohort-mean NDCG@10 increase from
    adding that candidate to the node's prefix, over the node's own users. Entries whose `upd` field
    is 0 are seed values (later entries were lazily re-evaluated against a longer prefix and are
    excluded). The chosen splitter's own gain is in `gains[0]` and is added back.

    So this is every node of the real tree x ~1,000 candidates of GROUND TRUTH for the quantity our
    criterion measures and the other criteria approximate, at states the real tree actually visited,
    at zero additional NDCG compute.
    """
    from scipy.stats import spearmanr, kendalltau
    rng = np.random.default_rng(BOOT_SEED)
    pool_pos = {tuple(q): i for i, q in enumerate(pool)}
    rowsets = {"root": [int(r) for r in rows_all]}
    for nid in sorted(nodes_ndcg, key=lambda n: len(n.split("."))):
        rec = nodes_ndcg[nid]
        if rec.get("split") and nid in rowsets:
            kids, _a, _z = split_rows(E, rowsets[nid], tuple(int(x) for x in rec["splitter"]))
            for b in BRANCHES:
                rowsets[nid + "." + b] = kids[b]
    per_node = []
    for nid in sorted(nodes_ndcg, key=lambda n: (len(n.split(".")), n)):
        hp = os.path.join(heapdir, "node_" + nid + ".json")
        if not os.path.exists(hp) or nid not in rowsets:
            log("[corr] " + nid + ": no banked heap sidecar or no row set -- skipped")
            continue
        blob = json.load(open(hp))
        meas = {}
        for negg, q, upd in blob["heap"]:
            if int(upd) == 0:
                meas[(int(q[0]), int(q[1]))] = -float(negg)
        if blob["seq"]:
            meas[(int(blob["seq"][0][0]), int(blob["seq"][0][1]))] = float(blob["gains"][0])
        rows = rowsets[nid]
        prefix = [(int(a), int(b)) for a, b in nodes_ndcg[nid]["path"]]
        cand_q = sorted(q for q in meas if q in pool_pos and q not in set(prefix))
        cand = [pool_pos[q] for q in cand_q]
        y = np.asarray([meas[q] for q in cand_q], np.float64)
        best_j = int(np.argmax(y))
        row = {"nid": nid, "depth": nodes_ndcg[nid]["depth"], "n_users": len(rows),
               "n_candidates_with_measured_gain": len(cand),
               "ndcg_argmax": list(cand_q[best_j]),
               "best_measured_ndcg_gain": float(y[best_j]),
               "tree_splitter": nodes_ndcg[nid]["splitter"]}
        for kind in ("variance", "pairwise", "logistic"):
            sc, _i = score_candidates(kind, E, ev, memL, memD, pos_of, [rows], cand, prefix, rows,
                                      rng, Wd, bd, {})
            am = int(np.argmax(sc))
            row[kind] = {
                "spearman_vs_measured_ndcg": float(spearmanr(sc, y).correlation),
                "kendall": float(kendalltau(sc, y).correlation),
                "argmax": list(cand_q[am]),
                "argmax_matches_ndcg_argmax": bool(cand_q[am] == cand_q[best_j]),
                "measured_ndcg_gain_of_its_argmax": float(y[am]),
                "regret": float(y[best_j] - y[am]),
                "rank_of_ndcg_argmax_under_this_criterion": int(1 + (sc > sc[best_j]).sum()),
                "top10_recall": float(len(set(np.argsort(-sc)[:10].tolist())
                                          & set(np.argsort(-y)[:10].tolist())) / 10.0)}
        row["shuffle_control"] = {
            "spearman_of_a_permuted_criterion":
                float(spearmanr(np.arange(len(y)), y[rng.permutation(len(y))]).correlation),
            "expected_regret_of_a_uniform_random_pick": float(y[best_j] - y.mean())}
        per_node.append(row)
        log("[corr] " + nid.ljust(44) + " n=" + str(len(rows)).rjust(5) + " cands="
            + str(len(cand)).rjust(4) + " | "
            + " | ".join(k[:4] + " rho=" + format(row[k]["spearman_vs_measured_ndcg"], "+.3f")
                         + " regret=" + format(row[k]["regret"], "+.5f")
                         for k in ("variance", "pairwise", "logistic")))
    agg = {}
    for kind in ("variance", "pairwise", "logistic"):
        rr = np.asarray([r[kind]["spearman_vs_measured_ndcg"] for r in per_node])
        gg = np.asarray([r[kind]["regret"] for r in per_node])
        nn = np.asarray([r["n_users"] for r in per_node], float)
        agg[kind] = {"n_nodes": len(rr), "mean_spearman": float(rr.mean()),
                     "median_spearman": float(np.median(rr)),
                     "min_spearman": float(rr.min()), "max_spearman": float(rr.max()),
                     "n_nodes_with_positive_spearman": int((rr > 0).sum()),
                     "mean_regret": float(gg.mean()),
                     "user_weighted_mean_regret": float((nn * gg).sum() / nn.sum()),
                     "n_nodes_argmax_matches": int(sum(r[kind]["argmax_matches_ndcg_argmax"]
                                                       for r in per_node)),
                     "mean_top10_recall": float(np.mean([r[kind]["top10_recall"] for r in per_node]))}
    rand_regret = float(np.mean([r["shuffle_control"]["expected_regret_of_a_uniform_random_pick"]
                                 for r in per_node]))
    out["criterion_vs_measured_ndcg"] = {
        "per_node": per_node, "aggregate": agg,
        "mean_expected_regret_of_a_uniform_random_pick": rand_regret,
        "source": ("banked CELF heap sidecars of the ndcgtree build; seed entries (upd == 0) are the "
                   "exact measured cohort-mean NDCG@10 marginal gains at that node's prefix")}
    log("=" * 110)
    for k, v in agg.items():
        log("[corr/AGG] " + k.ljust(10) + " mean rho " + format(v["mean_spearman"], "+.3f")
            + " (positive at " + str(v["n_nodes_with_positive_spearman"]) + "/" + str(v["n_nodes"])
            + " nodes) | argmax matches NDCG's at " + str(v["n_nodes_argmax_matches"]) + "/"
            + str(v["n_nodes"]) + " | mean top-10 recall " + format(v["mean_top10_recall"], ".2f")
            + " | user-weighted regret " + format(v["user_weighted_mean_regret"], "+.5f"))
    log("[corr/AGG] control: the expected regret of a UNIFORM RANDOM pick is "
        + format(rand_regret, "+.5f") + " -- a criterion worth having must sit well below it")
    return out


# --------------------------------------------------------------------------- main
def main():
    global BUDGETS
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot", default=CERT_TOWER)
    ap.add_argument("--concept_ckpt", default=CERT_CONC)
    ap.add_argument("--label", default=LABEL)
    ap.add_argument("--arms", default=",".join(GROW_ARMS))
    ap.add_argument("--stages", default="grow,measure,corr")
    ap.add_argument("--min_leaf", type=int, default=MIN_LEAF)
    ap.add_argument("--total_L", type=int, default=TOTAL_L)
    ap.add_argument("--seed", type=int, default=BOOT_SEED)
    ap.add_argument("--minleaf_mode", default="all3", choices=["all3", "nonempty"],
                    help="all3 = the floor applies to all three branches (ndcg_tree.py's rule, and "
                         "the PRIMARY mode); nonempty = only to branches that receive users, so a "
                         "fully answerable question splits binarily. See the note in `grow`.")
    ap.add_argument("--full_threads", action="store_true")
    ap.add_argument("--smoke", action="store_true",
                    help="60-candidate pool, budget 4, min_leaf 2000. Pipeline check only, NOT citable.")
    args = ap.parse_args()

    t00 = time.time()
    sfx = "_SMOKE" if args.smoke else ""
    OUT = os.path.join(OUTDIR, args.label + sfx + ".json")
    NPZ = os.path.join(OUTDIR, args.label + sfx + "_per_user.npz")
    os.makedirs(OUTDIR, exist_ok=True)
    stages = set(args.stages.split(","))
    arms = [a for a in args.arms.split(",") if a]
    total_L = 4 if args.smoke else args.total_L
    min_leaf = 2000 if args.smoke else args.min_leaf
    BUDGETS = tuple(sorted({K for K in BUDGETS if K <= total_L} | {total_L}))

    for nm, p in (("static policy-B", STATICB_JSON), ("ndcgtree16", NDCGTREE_JSON),
                  ("ndcgtree structure", NDCGSTRUCT_JSON),
                  ("ndcgtree test", NDCGTEST_JSON), ("snapshot", args.snapshot),
                  ("concept_ckpt", args.concept_ckpt)):
        assert os.path.exists(p), nm + " missing: " + p
    stb = json.load(open(STATICB_JSON))
    nd16 = json.load(open(NDCGTREE_JSON))
    ndt = json.load(open(NDCGTEST_JSON))
    ndst = json.load(open(NDCGSTRUCT_JSON))
    assert stb["mask_policy"] == "B" and stb["split"] == "validation" and stb["objective"] == "full"
    assert stb["n_pool"] == 1000 and stb["L"] == 16
    assert stb["tower"] == os.path.basename(args.snapshot)
    # the 16-question leaf ORDERS live in ndcgtree16.json; the tree STRUCTURE (splitters, which
    # is what routing needs) lives in the original ndcgtree.json. Same object, two artifacts.
    assert nd16["total_L"] == 16 and nd16["citable"]
    assert ndst["min_leaf"] == 200 and ndst["citable"] and ndst["split"] == "validation"
    assert ndst["n_leaves"] == len(nd16["leaf_table"]) == 15
    seq_static = [(int(a), int(b)) for a, b in stb["sequence"]][:total_L]

    log("#" * 110)
    log("# B1 / TODO 1.12 -- SPLIT-CRITERION ABLATION. Everything is held fixed except the scalar a "
        "candidate question is scored by. Arms: " + ", ".join(arms) + " (+ the banked `ndcg` arm).")
    log("# static comparator = " + stb["label"] + " (" + str(total_L)
        + " questions, policy-B-selected)")
    log("# PRE-REGISTERED: (c)>(b)>(a) => mechanism airtight; (b)~=(c) => novelty narrows to 'the "
        "exact reported metric' and the field claim survives with that nuance, reported as a "
        "narrowing and NOT marketed away; (b)>(c) => a major finding against us, reported first.")
    log("#" * 110)

    out = {"analysis": "split_criterion_ablation", "label": args.label,
           "citable": not args.smoke, "SMOKE": bool(args.smoke),
           "tower": os.path.basename(args.snapshot),
           "concept_ckpt": os.path.basename(args.concept_ckpt),
           "objective": "full", "mask_policy": "B", "total_L": total_L, "min_leaf": min_leaf,
           "minleaf_mode": args.minleaf_mode,
           "budgets": list(BUDGETS), "lam_u": SC.LAM_U, "seed": args.seed,
           "static_comparator": {"label": stb["label"], "sequence": [[a, b] for a, b in seq_static]},
           "bootstrap": {"kind": "paired percentile over per-user differences", "n_resamples": BOOT,
                         "seed": BOOT_SEED},
           "arms": {}, "structure": {}}

    # ------------------------------------------------------------------ VALIDATION environment
    Ev = build_greedy_env(args.snapshot, "validation", args.concept_ckpt, "splitcrit(val)")
    rows_val = Ev["rows"]
    contract_selftest(Ev, rows_val)
    leak_assert(Ev, rows_val, "validation")
    assert len(rows_val) == 10000, str(len(rows_val)) + " validation users"
    set_active_rows(Ev, rows_val); Ev["memo"] = {}
    icpt_v = genv_mean(Ev, [], rows_val)
    log("[snap/val] m=0 intercept " + format(icpt_v[0], ".8f") + " full / "
        + format(icpt_v[1], ".8f") + " tail")
    assert abs(icpt_v[0] - 0.15686538118354684) < 1e-9, (
        "CANONICAL SNAP FAILED: validation m=0 full is " + repr(icpt_v[0]) + ", the banked policy-B "
        "value is 0.15686538118354684. The ruler moved -- STOP.")
    log("[snap/val] PASS -- reproduces the banked policy-B validation intercept exactly")

    pool, item_pool, conc_pool = build_pool(Ev, 500, 500)
    if args.smoke:
        pool = item_pool[:30] + conc_pool[:30]
    ev = SC.build_evidence(Ev, rows_val)
    SC.assert_no_leak(Ev, ev, rows_val, log)
    memL, memD = SC.build_membership(Ev, rows_val, pool, log)
    for j in (0, 1, len(pool) // 2, len(pool) - 1):
        lab, _ar, _z = branch_of(Ev, rows_val, pool[j])
        assert np.array_equal(memL[j], lab == "like")
        assert np.array_equal(memD[j], lab == "dislike")
    log("[selftest] membership matrices == ndcg_tree.branch_of on spot-checked candidates")
    pos_of = {int(r): i for i, r in enumerate(rows_val)}
    Wd = np.asarray(Ev["ctx"].Wd.detach().numpy(), np.float64)
    bd = np.asarray(Ev["ctx"].bd.detach().numpy(), np.float64).ravel()
    log("[stack] decoder Wd " + str(Wd.shape) + ", bd " + str(bd.shape) + ", latent d="
        + str(Ev["ctx"].d) + "; Hu & Yu lam_u=" + str(SC.LAM_U))

    # ------------------------------------------------------------------ GROW
    trees = {}
    for kind in arms:
        tp = os.path.join(OUTDIR, args.label + sfx + "_tree_" + kind + ".json")
        if "grow" in stages:
            log("=" * 110)
            log("[grow] criterion `" + kind + "`")
            nodes, leaves, st = grow(kind, Ev, ev, memL, memD, pos_of, pool, rows_val, total_L,
                                     min_leaf, args.seed, tp, Wd, bd, args.minleaf_mode)
            blob = {"criterion": kind, "nodes": nodes,
                    "leaf_orders": {l["nid"]: nodes[l["nid"]]["own_sequence"] for l in leaves},
                    "stats": st}
            with open(tp, "w") as fh:
                json.dump(blob, fh, indent=2, default=float)
        else:
            blob = json.load(open(tp))
        trees[kind] = blob
        out["structure"][kind] = structure(kind, blob["nodes"],
                                           [{"nid": n} for n in blob["leaf_orders"]], total_L)

    nd_nodes = ndst["nodes"]
    nd_leaves = {r["nid"]: r["own_sequence"] for r in nd16["leaf_table"]}
    trees["ndcg"] = {"criterion": "ndcg", "nodes": nd_nodes, "leaf_orders": nd_leaves}
    out["structure"]["ndcg"] = structure("ndcg", nd_nodes, [{"nid": n} for n in nd_leaves], total_L)
    all_arms = list(arms) + ["ndcg"]

    qsets = {k: {tuple(q) for q in out["structure"][k]["distinct_questions_asked"]} for k in all_arms}
    ssets = {k: {tuple(q) for q in out["structure"][k]["splitters"]} for k in all_arms}
    out["overlap"] = {
        "questions_asked": {a + "|" + b: len(qsets[a] & qsets[b])
                            for a in all_arms for b in all_arms if a < b},
        "questions_asked_union": {a + "|" + b: len(qsets[a] | qsets[b])
                                  for a in all_arms for b in all_arms if a < b},
        "splitters": {a + "|" + b: len(ssets[a] & ssets[b])
                      for a in all_arms for b in all_arms if a < b},
        "root_splitter": {k: out["structure"][k]["root_splitter"] for k in all_arms}}
    for k in all_arms:
        s = out["structure"][k]
        log("[structure] " + k.ljust(11) + " depth " + str(s["depth_reached"]) + " | "
            + str(s["n_nodes"]) + " nodes, " + str(s["n_leaves"]) + " leaves | root = "
            + str(s["root_channel"]) + " " + str(s["root_splitter"][1]) + " (answer rate "
            + str(s["root_answer_rate"]) + ", children " + str(s["root_child_sizes"]) + ") | "
            + str(s["n_distinct_questions"]) + " distinct questions")

    # ------------------------------------------------------------------ MEASURE
    bank = {}
    if "measure" in stages:
        out["intercept_val_m0"] = {"full@10": icpt_v[0], "tail@10": icpt_v[1]}
        for tag in ("val", "test"):
            if tag == "val":
                E, rows = Ev, rows_val
            else:
                E = build_greedy_env(args.snapshot, "test", args.concept_ckpt, "splitcrit(test)")
                rows = E["rows"]
                contract_selftest(E, rows); leak_assert(E, rows, "test")
                assert len(rows) == 10000, str(len(rows)) + " test users"
                set_active_rows(E, rows); E["memo"] = {}
                ic = genv_mean(E, [], rows)
                log("[snap/test] m=0 intercept " + format(ic[0], ".5f") + " full / "
                    + format(ic[1], ".5f") + " tail")
                assert abs(ic[0] - 0.15855) < 5e-5 and abs(ic[1] - 0.02819) < 5e-5, (
                    "CANONICAL SNAP FAILED on test: " + repr(ic) + " vs the banked 0.15855/0.02819")
                log("[snap/test] PASS -- reproduces MASK_POLICY_B_MIGRATION's banked test intercept")
                out["intercept_test_m0"] = {"full@10": ic[0], "tail@10": ic[1]}
            for kind in all_arms:
                log("-" * 110)
                log("[measure/" + tag + "] arm `" + kind + "`")
                _pl, curves = measure(E, tag, kind, trees[kind]["leaf_orders"],
                                      trees[kind]["nodes"], rows, seq_static, bank)
                out["arms"].setdefault(kind, {})["leaf_curves_" + tag] = curves
                out["arms"][kind]["cohort_" + tag] = cohort_table(
                    bank, tag, kind, list(trees[kind]["leaf_orders"]))
                np.savez_compressed(NPZ, **bank)
                with open(OUT + ".tmp", "w") as fh:
                    json.dump(out, fh, indent=2, default=float)
                os.replace(OUT + ".tmp", OUT)

        if not args.smoke:
            chk = {}
            for K in BUDGETS:
                for ax in ("full", "tail"):
                    got = out["arms"]["ndcg"]["cohort_test"][str(K)]["diff_" + ax + "@10"]
                    want = ndt["cohort_weighted_test"][str(K)]["diff_" + ax + "@10_vs_policyB"]
                    chk["K" + str(K) + "_" + ax] = {"recomputed": got, "banked": want,
                                                    "abs_dev": abs(got - want)}
                    assert abs(got - want) < 1e-6, (
                        "BANKED HEADLINE NOT REPRODUCED at K=" + str(K) + " " + ax + ": " + repr(got)
                        + " vs " + repr(want) + ". The `ndcg` arm re-measured by THIS code path does "
                        "not reproduce ndcgtree16test_policyB.json -- the ablation is not on the "
                        "tree's ruler. STOP.")
            out["banked_headline_assert"] = chk
            log("[banked-assert] PASS: the `ndcg` arm re-measured through this file's own code path "
                "reproduces every published test number (max deviation "
                + format(max(v["abs_dev"] for v in chk.values()), ".2e") + ").")

    # ------------------------------------------------------------------ CORRELATION EXHIBIT
    if "corr" in stages and not args.smoke:
        log("=" * 110)
        log("[corr] criterion-vs-MEASURED-NDCG over all candidates at every node of the real tree")
        set_active_rows(Ev, rows_val); Ev["memo"] = {}
        corr_stage(Ev, ev, memL, memD, pos_of, pool, rows_val, ndst["nodes"],
                   os.path.join(OUTDIR, "ndcgtree_heaps"), Wd, bd, out)

    # ------------------------------------------------------------------ report
    log("=" * 110)
    log("arm".ljust(12) + "K".rjust(4) + "VAL full".rjust(12) + "VAL tail".rjust(12)
        + "TEST full".rjust(12) + "TEST tail".rjust(12) + "sepF".rjust(6) + "sepT".rjust(6))
    for kind in all_arms:
        a = out["arms"].get(kind, {})
        for K in BUDGETS:
            v = a.get("cohort_val", {}).get(str(K), {})
            t = a.get("cohort_test", {}).get(str(K))
            if not t:
                continue
            log(kind.ljust(12) + str(K).rjust(4)
                + format(v.get("diff_full@10", float("nan")), "12.5f")
                + format(v.get("diff_tail@10", float("nan")), "12.5f")
                + format(t["diff_full@10"], "12.5f") + format(t["diff_tail@10"], "12.5f")
                + str(t["separates_full"]).rjust(6) + str(t["separates_tail"]).rjust(6))
    out["seconds"] = round(time.time() - t00, 1)
    with open(OUT + ".tmp", "w") as fh:
        json.dump(out, fh, indent=2, default=float)
    os.replace(OUT + ".tmp", OUT)
    if bank:
        np.savez_compressed(NPZ, **bank)
    log("[done] wrote " + os.path.basename(OUT) + " in " + format((time.time() - t00) / 60, ".1f")
        + "m")
    return out


if __name__ == "__main__":
    main()
