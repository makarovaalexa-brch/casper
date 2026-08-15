r"""P1b -- THE NON-CRIPPLING GENERALISATION TEST: does the WORKING policy rank questions it was
never taught, when those are the right answer?

Design sheet: `docs/design/PAPER_B_POLICY_DESIGN.md` Part I.1, amended 2026-08-15 after P1 measured
that removing 36 of the 172 label questions damages the policy (0.28355 -> 0.23185). P1 answers "can
it still reach a question you took away from it"; P1b answers the question that does not require
breaking anything first.

★ THE HELD-OUT SET ALREADY EXISTS AND NOBODY HAD TO MAKE IT. The frozen tree uses only **172 of the
1,000 pool questions** as labels, so **828 pool questions were never a training target for E1** --
no gradient ever pushed the policy toward emitting one. The measured-value sweeps banked the
NDCG gain of a large candidate set at states off the tree's own trajectory. Wherever the measured
best question at such a state is one of those 828, we have a never-taught question that IS the right
answer, with a measured ground truth, and an UNDAMAGED policy to ask about it.

THE STATES USED, AND WHY (composition printed before any number is trusted)
--------------------------------------------------------------------------
  GROUP A -- the 42 phase-3 below-floor states. **FULL-POOL sweeps: every one of ~995 pool
             candidates measured at that state**, cohorts of 200-800 users, rows banked. This is the
             only material in the project that can answer the rank question without a sampling
             caveat, and it is the primary.
  GROUP B -- the 138 random-prefix states. Only the **50 candidates that sweep sampled** (20 pool +
             30 catalogue) were measured, so a "rank of the measured argmax" here is the rank of the
             best-of-20-in-pool, not of the best of 1,000. Reported separately, never pooled with A.

  GROUP C -- ★ THE ON-MANIFOLD CONTROL, without which the other two are misreadable. The 7 internal
             nodes of the frozen tree, each with the **exact gain of all 1,000 candidates** banked.
             E1 reproduces the tree exactly here and the measured argmax IS the tree's splitter, so
             E1 must rank it **1st**. It is simultaneously the cell where the policy is expected to
             succeed and the instrument check on the whole file.

Groups A and B are OFF the tree's trajectory manifold, so E1 was never trained on those STATES
either. **That means a taught/never-taught contrast within A or B confounds two novelties — unseen
QUESTION and unseen STATE — and P1b cannot separate them.** Group C is what makes the comparison
readable: it is the only cell where the state is familiar.

ARMS AND CONTROLS (both mandatory)
----------------------------------
  E1              the unmodified anchor, `policy_main_equivariant_both.pt`, val 0.28355.
  shuffle         E1's own label-shuffled control, `policy_shuffle_equivariant_both.pt`. The FLOOR.
  popularity      a fixed popularity ranking -- "train users who can answer": `ctx.cnt[i]` for an
                  item, `answer_rate[c] * 140,768` for a concept. THE DUMB COMPARATOR: if popularity
                  finds the never-taught question as often, similarity added nothing.

**Falsifier, pre-registered:** never-taught ranking <= the popularity comparator ⇒ there is no
similarity-based generalisation in the ACTION policy, and it is reported failed.

VALIDATION ONLY. The 10,000 canonical TEST users are never built, never loaded, never named.
"""
import os
import sys
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
sys.path.insert(0, _HERE)

import glob
import json
import time
import argparse
import numpy as np
import torch

import policy_net as PN
from policy_net import (OUTDIR, SEED, TOTAL_L, STATIC_JSON, STATIC_VAL_FULL, VAL_M0_FULL,
                        Policy, build_phi, load_traj, log)

P1B_DIR = os.path.join(OUTDIR, "p1b_nevertaught")
DEEP_DIR = os.path.join(OUTDIR, "policy_labels_deep_partial")
SAMPLE_DIR = os.path.join(OUTDIR, "value_sample_partial")


def spearman(a, b):
    a = np.asarray(a, np.float64); b = np.asarray(b, np.float64)
    if len(a) < 3:
        return None
    ra = np.argsort(np.argsort(a)).astype(np.float64)
    rb = np.argsort(np.argsort(b)).astype(np.float64)
    ra -= ra.mean(); rb -= rb.mean()
    d = float(np.sqrt((ra ** 2).sum() * (rb ** 2).sum()))
    return float((ra * rb).sum() / d) if d > 0 else None


TREE_HEAPS = os.path.join(OUTDIR, "ndcgtree_heaps")
_OUTCODE = {"like": 0, "dislike": 1, "noanswer": 2}


def load_onmanifold(pool_index, Dva):
    """★ GROUP C — THE ON-MANIFOLD CONTROL, and P1b is not interpretable without it.

    Every state in groups A and B is one E1 was NEVER TRAINED ON, so a taught/never-taught split
    across them confounds two different kinds of novelty: an unseen QUESTION and an unseen STATE.
    Group C removes the second: the **7 internal nodes of the frozen tree**, where E1 reproduces the
    tree exactly and the measured argmax IS the tree's own splitter. E1 must rank it **1st**. If it
    does not, the ranking harness disagrees with the policy's own argmax and every number in this
    file is unverified — so this doubles as the instrument check.

    Their level-0 sweeps survived intact (`policy_deep_labels.json:complete_level0_vectors`: CELF
    overwrote the step-0 record wherever a node made a SEQUENCE of picks; the 7 single-pick internal
    nodes are the ones that did not), so each carries the **exact gain of all 1,000 candidates**:
    999 in the residual heap at level 0 plus the picked one.

    The cohort is not banked, but it is exactly reconstructible: node `root.a.b` is the users whose
    ternary answers at positions 0 and 1 were `a` and `b`. It is rebuilt from the validation
    trajectory bank and **asserted against the banked `n_rows`** — a free contract check.
    """
    C = []
    outc = np.asarray(Dva["outcome"], np.int64)
    uids = np.asarray(Dva["uid"], np.int64)
    for fp in sorted(glob.glob(os.path.join(TREE_HEAPS, "node_*.json"))):
        b = json.load(open(fp))
        if len(b.get("seq", [])) != 1:
            continue
        nid = os.path.basename(fp)[len("node_"):-len(".json")]
        parts = nid.split(".")
        assert parts[0] == "root"
        m = np.ones(len(uids), bool)
        for j, name in enumerate(parts[1:]):
            m &= outc[:, j] == _OUTCODE[name]
        rows = [int(u) for u in uids[m]]
        assert len(rows) == int(b["sig"]["n_rows"]), (
            f"{nid}: reconstructed cohort {len(rows)} users, banked {b['sig']['n_rows']}")
        prefix = [(int(a), int(x)) for a, x in b["sig"]["prefix"]]
        assert len(prefix) == len(parts) - 1
        pick = (int(b["seq"][0][0]), int(b["seq"][0][1]))
        g = {pool_index[pick]: float(b["gains"][0])}
        for negg, q, lvl in b["heap"]:
            assert int(lvl) == 0, f"{nid}: a heap entry is not at level 0 -- not an exact gain"
            pi = pool_index.get((int(q[0]), int(q[1])))
            if pi is not None:
                g[pi] = -float(negg)
        assert len(g) == int(b["sig"]["n_pool"]), f"{nid}: {len(g)} gains, expected the whole pool"
        assert max(g.items(), key=lambda kv: kv[1])[0] == pool_index[pick], (
            f"{nid}: the banked pick is not the argmax of the banked gain vector")
        C.append({"sid": nid, "prefix": prefix, "rows": rows, "gains": g,
                  "group": "C_on_manifold", "n_measured_in_pool": len(g),
                  "source": os.path.basename(fp), "tree_splitter": pool_index[pick]})
    return C


def load_states(pool_index):
    """Group A (full-pool, phase 3) and group B (50-candidate, random prefix)."""
    A, B = [], []
    for fp in sorted(glob.glob(os.path.join(DEEP_DIR, "*.json"))):
        if os.path.basename(fp) == "_sig.json":
            continue
        d = json.load(open(fp))
        if not d.get("done"):
            continue
        g = {}
        for q, v in d["vals"].items():
            a, b = q.split("_")
            pi = pool_index.get((int(a), int(b)))
            if pi is not None:
                g[pi] = float(v[0])                    # [0] = the NDCG-full gain, the objective
        A.append({"sid": d["sid"], "prefix": [(int(a), int(b)) for a, b in d["prefix"]],
                  "rows": [int(r) for r in d["rows"]], "gains": g, "group": "A_full_pool",
                  "n_measured_in_pool": len(g), "source": os.path.basename(fp)})
    for fp in sorted(glob.glob(os.path.join(SAMPLE_DIR, "prefix_*.json"))):
        d = json.load(open(fp))
        prefix = [(int(a), int(b)) for a, b in d["prefix"]]
        cands = [(int(a), int(b)) for a, b in d["candidates"]]
        for s in d["states"]:
            g = {}
            for q in cands:
                pi = pool_index.get(q)
                if pi is not None:
                    g[pi] = float(s["gains"][f"{q[0]}_{q[1]}"])
            B.append({"sid": f"{d['prefix_index']:03d}:{s['path']}", "prefix": prefix,
                      "rows": [int(r) for r in s["rows"]], "gains": g, "group": "B_sampled_50",
                      "n_measured_in_pool": len(g),
                      "source": os.path.basename(fp)})
    return A, B


def main():
    ap = argparse.ArgumentParser(description="P1b: never-taught question ranking, undamaged policy")
    ap.add_argument("--ckpts",
                    default=("e1=" + os.path.join(OUTDIR, "policy_main_equivariant_both.pt") +
                             ",shuffle=" + os.path.join(OUTDIR, "policy_shuffle_equivariant_both.pt")))
    ap.add_argument("--snapshot", default=None)
    ap.add_argument("--concept_ckpt", default=None)
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--full_threads", action="store_true")
    ap.add_argument("--i_have_author_goahead", action="store_true")
    args = ap.parse_args()
    assert args.i_have_author_goahead, "REFUSING TO RUN without --i_have_author_goahead"
    from strategy_channel_suite import (build_greedy_env, contract_selftest, genv_eval, genv_mean,
                                        genv_latent, CERT_TOWER, CERT_CONC, CONCSTATS,
                                        N_TRAIN_USERS)
    from ndcg_tree import build_pool, leak_assert, branch_of_per_user, branch_of_selftest
    args.snapshot = args.snapshot or CERT_TOWER
    args.concept_ckpt = args.concept_ckpt or CERT_CONC
    t00 = time.time()
    os.makedirs(P1B_DIR, exist_ok=True)
    log("#" * 110)
    log("# P1b -- CAN THE UNDAMAGED POLICY RANK A QUESTION IT WAS NEVER TAUGHT? VALIDATION ONLY.")
    log("# The 10,000 canonical TEST users are NOT built, NOT loaded, NOT touched.")
    log("#" * 110)

    Dva = load_traj(train=False)
    pool = Dva["pool"]; P = len(pool)
    pool_index = {(int(a), int(b)): i for i, (a, b) in enumerate(pool)}
    taught = set(int(x) for x in np.unique(np.asarray(Dva["label_idx"], np.int64)))
    log(f"[taught] the frozen tree uses {len(taught)} of the {P} pool questions as labels; "
        f"{P - len(taught)} pool questions were NEVER a training target for E1")

    E = build_greedy_env(args.snapshot, "validation", args.concept_ckpt, "p1b")
    contract_selftest(E, E["rows"]); leak_assert(E, E["rows"], "p1b")
    rows_all = E["rows"]
    f0, t0_ = genv_mean(E, [], rows_all)
    assert round(f0, 8) == VAL_M0_FULL, f"CANONICAL SNAP FAILED: {f0!r}"
    log(f"[gate] canonical m=0 snap full={f0!r} -- exact")
    pool_e, _i, _c = build_pool(E, 500, 500)
    assert pool_e == pool, "the env's pool is not the trajectory bank's pool"
    branch_of_selftest(E, rows_all[:2000], [pool[0], pool[500], (1, 114)])
    static_seq = [(int(a), int(b)) for a, b in json.load(open(STATIC_JSON))["sequence"]]
    sfv, _s = genv_eval(E, static_seq, rows_all)
    assert abs(float(np.nanmean(sfv)) - STATIC_VAL_FULL) < 1e-12, "BANKED-HEADLINE ASSERT FAILED"
    log(f"[gate] static comparator reproduces EXACTLY: {float(np.nanmean(sfv))!r}")

    phi_t, norm = build_phi(E, pool)
    phi_g = torch.from_numpy(phi_t)
    zc = np.load(CONCSTATS)
    c_ar = (zc["n"] / N_TRAIN_USERS).astype(np.float64)
    cnt = np.asarray(E["ctx"].cnt, np.float64)
    popv = np.asarray([cnt[int(i)] if int(c) == 0 else c_ar[int(i)] * N_TRAIN_USERS
                       for c, i in pool], np.float64)

    pols = {}
    for spec in args.ckpts.split(","):
        name, path = spec.split("=", 1)
        b = torch.load(path, map_location="cpu")
        assert [(int(a), int(bb)) for a, bb in b["pool"]] == pool
        assert np.array_equal(b["norm_mu"], norm[0]) and np.array_equal(b["norm_sd"], norm[1])
        pl = Policy(E["ctx"].d, phi_t.shape[1], P, arch=b["arch"], state=b["state"],
                    use_pos=b["use_pos"])
        pl.load_state_dict(b["model"]); pl.eval()
        pols[name] = pl
        log(f"[arm] {name}: {os.path.basename(path)} epoch {b['epoch']} val_full {b['val_full']:.6f}")

    A, B = load_states(pool_index)
    C = load_onmanifold(pool_index, Dva)
    log(f"[states] group A (full-pool phase-3 sweeps) {len(A)}; "
        f"group B (50-candidate random prefixes) {len(B)}; "
        f"group C (ON-MANIFOLD tree internal nodes, full 1,000-candidate exact gains) {len(C)}")

    # ---- ★ COMPOSITION, PRINTED BEFORE ANY TIME IS SPENT -----------------------------------------
    comp = {}
    for gname, G in (("A_full_pool", A), ("B_sampled_50", B), ("C_on_manifold", C)):
        am = [max(s["gains"].items(), key=lambda kv: kv[1])[0] for s in G]
        nt = [q for q in am if q not in taught]
        comp[gname] = {
            "n_states": len(G),
            "mean_candidates_measured_in_pool": float(np.mean([s["n_measured_in_pool"] for s in G])),
            "mean_cohort_users": float(np.mean([len(s["rows"]) for s in G])),
            "cohort_user_quantiles": [int(x) for x in np.percentile(
                [len(s["rows"]) for s in G], [0, 25, 50, 75, 100])],
            "prefix_length_histogram": {str(int(v)): int(c) for v, c in zip(
                *np.unique([len(s["prefix"]) for s in G], return_counts=True))},
            "n_states_whose_measured_argmax_is_NEVER_TAUGHT": len(nt),
            "n_states_whose_measured_argmax_is_TAUGHT": len(am) - len(nt),
            "n_distinct_never_taught_argmax_questions": len(set(nt)),
            "never_taught_argmax_channel_share_concept": float(
                np.mean([pool[q][0] == 1 for q in nt])) if nt else None}
        log(f"[composition {gname}] {comp[gname]['n_states']} states, "
            f"{comp[gname]['mean_candidates_measured_in_pool']:.0f} in-pool candidates measured each, "
            f"cohorts {comp[gname]['cohort_user_quantiles']}")
        log(f"[composition {gname}] the measured best question is NEVER-TAUGHT at "
            f"{comp[gname]['n_states_whose_measured_argmax_is_NEVER_TAUGHT']} of "
            f"{comp[gname]['n_states']} states "
            f"({comp[gname]['n_distinct_never_taught_argmax_questions']} distinct questions); "
            f"TAUGHT at {comp[gname]['n_states_whose_measured_argmax_is_TAUGHT']}")

    # ---- the per-state scoring --------------------------------------------------------------------
    per_state = []
    for gname, G in (("A_full_pool", A), ("B_sampled_50", B), ("C_on_manifold", C)):
        for si, s in enumerate(G):
            rws = s["rows"]
            n = len(rws)
            prefix = s["prefix"]
            k = len(prefix)
            z = genv_latent(E, prefix, rws).numpy()
            outs = np.stack([np.asarray(branch_of_per_user(E, rws, [q] * n), np.int64)
                             for q in prefix], 1) if k else np.zeros((n, 0), np.int64)
            hq = np.zeros((n, TOTAL_L - 1), np.int64); ho = np.zeros((n, TOTAL_L - 1), np.int64)
            hm = np.zeros((n, TOTAL_L - 1), np.float32)
            for j in range(min(k, TOTAL_L - 1)):
                hq[:, j] = pool_index[prefix[j]]; ho[:, j] = outs[:, j]; hm[:, j] = 1.0
            pos = np.tile(np.asarray([[k, TOTAL_L - k]], np.float32), (n, 1))
            asked = np.zeros((n, P), bool)
            for q in prefix:
                asked[:, pool_index[q]] = True
            cand = np.asarray(sorted(s["gains"]), np.int64)
            cand = np.asarray([q for q in cand if not asked[0, q]], np.int64)
            g = np.asarray([s["gains"][int(q)] for q in cand], np.float64)
            best = int(cand[int(np.argmax(g))])
            rec = {"sid": s["sid"], "group": gname, "n_users": n, "prefix_len": k,
                   "n_candidates_measured": int(len(cand)),
                   "measured_argmax_pool_idx": best,
                   "measured_argmax_gain": float(g.max()),
                   "measured_argmax_is_taught": bool(best in taught),
                   "channel": "item" if pool[best][0] == 0 else "concept", "arms": {}}
            for name, pl in pols.items():
                sc = np.zeros(P, np.float64)
                with torch.no_grad():
                    for st in range(0, n, 1000):
                        sl = slice(st, min(st + 1000, n))
                        o = pl(torch.from_numpy(z[sl]), torch.from_numpy(phi_t[hq[sl]]),
                               torch.from_numpy(ho[sl]), torch.from_numpy(hm[sl]),
                               torch.from_numpy(pos[sl]), phi_g,
                               torch.from_numpy(asked[sl])).numpy()
                        sc += np.nansum(np.where(np.isfinite(o), o, 0.0), 0)
                sc /= n
                sc[asked[0]] = -np.inf
                rank = int((sc > sc[best]).sum())          # 0 = the policy's own top pick
                order = np.argsort(-sc)
                nt_mask = np.asarray([int(q) not in taught for q in cand], bool)
                rec["arms"][name] = {
                    "rank_of_the_measured_best_out_of_1000": rank,
                    "top1": bool(rank == 0), "top5": bool(rank < 5), "top20": bool(rank < 20),
                    "policy_top1_is_never_taught": bool(int(order[0]) not in taught),
                    "spearman_all_measured": spearman(sc[cand], g),
                    "spearman_never_taught_only": (spearman(sc[cand][nt_mask], g[nt_mask])
                                                   if nt_mask.sum() >= 3 else None),
                    "measured_gain_of_the_policys_top1": float(
                        s["gains"].get(int(order[0]), np.nan))}
            r = popv.copy(); r[asked[0]] = -np.inf
            prank = int((r > r[best]).sum())
            nt_mask = np.asarray([int(q) not in taught for q in cand], bool)
            rec["arms"]["popularity"] = {
                "rank_of_the_measured_best_out_of_1000": prank,
                "top1": bool(prank == 0), "top5": bool(prank < 5), "top20": bool(prank < 20),
                "policy_top1_is_never_taught": bool(int(np.argmax(r)) not in taught),
                "spearman_all_measured": spearman(popv[cand], g),
                "spearman_never_taught_only": (spearman(popv[cand][nt_mask], g[nt_mask])
                                               if nt_mask.sum() >= 3 else None),
                "measured_gain_of_the_policys_top1": float(
                    s["gains"].get(int(np.argmax(r)), np.nan))}
            per_state.append(rec)
            if (si + 1) % 20 == 0 or si == len(G) - 1:
                log(f"  [{gname}] {si + 1}/{len(G)} states scored "
                    f"({(time.time() - t00) / 60:.1f}m)")

    # ---- the tables -------------------------------------------------------------------------------
    arms = ["e1", "shuffle", "popularity"] if "e1" in pols else list(pols) + ["popularity"]
    summary = {}
    for gname in ("A_full_pool", "B_sampled_50", "C_on_manifold"):
        for taught_flag, lab in ((False, "never_taught"), (True, "taught")):
            rs = [r for r in per_state
                  if r["group"] == gname and r["measured_argmax_is_taught"] == taught_flag]
            if not rs:
                continue
            cell = {"n_states": len(rs)}
            for a in arms:
                rk = np.asarray([r["arms"][a]["rank_of_the_measured_best_out_of_1000"] for r in rs])
                sp = [r["arms"][a]["spearman_all_measured"] for r in rs
                      if r["arms"][a]["spearman_all_measured"] is not None]
                spn = [r["arms"][a]["spearman_never_taught_only"] for r in rs
                       if r["arms"][a]["spearman_never_taught_only"] is not None]
                cell[a] = {
                    "median_rank": float(np.median(rk)), "mean_rank": float(rk.mean()),
                    "rank_quantiles": [int(x) for x in np.percentile(rk, [0, 25, 50, 75, 100])],
                    "top1_rate": float((rk == 0).mean()), "top5_rate": float((rk < 5).mean()),
                    "top20_rate": float((rk < 20).mean()),
                    "mean_spearman_all_measured": float(np.mean(sp)) if sp else None,
                    "mean_spearman_never_taught_candidates": float(np.mean(spn)) if spn else None,
                    "frac_states_where_its_own_top1_is_a_never_taught_question":
                        float(np.mean([r["arms"][a]["policy_top1_is_never_taught"] for r in rs]))}
            summary[f"{gname}/{lab}"] = cell
            log("=" * 100)
            log(f"[{gname} / measured best is {lab}] n = {len(rs)} states")
            for a in arms:
                c = cell[a]
                log(f"   {a:>10}: median rank {c['median_rank']:>6.0f} | top1 "
                    f"{c['top1_rate']:.4f} top5 {c['top5_rate']:.4f} top20 {c['top20_rate']:.4f} | "
                    f"rho(all) {c['mean_spearman_all_measured']} | "
                    f"rho(never-taught only) {c['mean_spearman_never_taught_candidates']}")

    out = {"analysis": "policy_p1b_never_taught", "design_sheet":
           "docs/design/PAPER_B_POLICY_DESIGN.md I.1 (amended 2026-08-15)", "citable": True,
           "test_touched": False, "split": "validation",
           "n_taught_questions": len(taught), "n_pool": P,
           "n_never_taught_pool_questions": P - len(taught),
           "composition": comp, "summary": summary, "per_state": per_state,
           "arms": {n: os.path.basename(spec.split("=", 1)[1])
                    for n, spec in zip(pols, args.ckpts.split(","))},
           "popularity_definition": "train users who can answer: ctx.cnt[i] for an item, "
                                    "answer_rate[c]*140768 for a concept",
           "minutes": round((time.time() - t00) / 60, 2)}
    p = os.path.join(P1B_DIR, "p1b_never_taught.json")
    with open(p + ".tmp", "w") as fh:
        json.dump(out, fh, indent=1, default=float)
    os.replace(p + ".tmp", p)
    log(f"[p1b] wrote {p} in {out['minutes']:.1f} minutes")
    return out


if __name__ == "__main__":
    main()
