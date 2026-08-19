r"""M4c -- DOES THE POLICY GENERALISE TO CLUSTER-ADJACENT FACTS IT WAS NEVER TAUGHT?

Author's question, 2026-08-19: the policy has now seen six opener-conditioned teachers. If it learned
"an anchored state gets a specialised interview" as a function of the STATE, then a user who
volunteers something *adjacent* to a cluster -- a different concept the policy was never taught --
should land in a similar state and be routed to that cluster's interview anyway. If instead it
memorised six anchor tokens, adjacent facts will do nothing.

Those two hypotheses are indistinguishable on the six anchor cohorts and differ everywhere else.

THE DESIGN
----------
For each anchor A:
  ADJACENT  the non-anchor concepts most co-endorsed with A, by LIFT computed on TRAIN users, so the
            cohort definition never touches the split the measurement runs on. Cohort = validation
            users who endorse the neighbour and NOT A (and no other anchor), so nothing here is an
            anchor cohort in disguise.
  FAR       concepts with the LOWEST maximum lift against any anchor, matched on answer rate to the
            adjacent set. ★ THIS CONTROL IS THE EXPERIMENT. Without it, "the policy behaves like a
            teacher" cannot be separated from "folding any concept at all makes it behave like a
            teacher".

The volunteered concept is folded free and carried as an ANSWERED QUESTION (history token + LIKE +
masked from the vocabulary) -- the L1 encoding, identical to how the six anchors are handled. Its
descriptor is built with the TRAINING pool's standardisation, never re-fitted.

READOUT
  routing   the policy's interview against all seven trees' routed interviews for the same users:
            which tree does it resemble, at the root and over the whole interview?
  value     policy vs the cold tree vs teacher-A, full and tail NDCG@10, K in {1,4,15}.

PREDICTION, declared before the run: if generalisation is real, ADJACENT cohorts route toward A's
teacher and do at least as well as the cold tree, while FAR cohorts look like the cold tree. If
adjacent and far behave alike, there is no generalisation and the policy is a token lookup -- which
is a reportable negative and must be reported, not retried.

VALIDATION ONLY. The 10,000 canonical TEST users are never built, loaded or named.
"""
from __future__ import annotations

import os
import sys

_H = os.path.dirname(os.path.abspath(__file__))
_R = os.path.abspath(os.path.join(_H, "..", ".."))
sys.path.insert(0, _H)

import argparse
import json
import time

import numpy as np
import torch

B = os.path.join(_R, "experiments", "battery")
OUT = os.path.join(B, "m4")
SEED = 98765
TOTAL_L = 15
VAL_M0_FULL = 0.15686538118354684
ANCHORS = {"arthouse": 68, "dialogue": 275, "marvel": 581,
           "gore": 417, "teenagers": 922, "inspiring": 503}
TREES = ["cold"] + list(ANCHORS)
BUDGETS = [1, 4, 15]


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def paired_boot(d, n=10000, seed=SEED):
    d = np.asarray(d, np.float64)
    d = d[np.isfinite(d)]
    if len(d) == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    bs = d[rng.integers(0, len(d), size=(n, len(d)))].mean(1)
    return float(d.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def policy_rollout_foldonly(E, pol, phi_ext, pool, rows, cid):
    """The policy's interview when the volunteered fact is FOLDED but NOT announced: the anchor slot
    stays empty, so the state is an ordinary cold state whose latent happens to carry the evidence."""
    from m4b_eval import policy_rollout
    return policy_rollout(E, pol, phi_ext, pool, rows, None, 0, -1, L=TOTAL_L, base_extra=[cid])


def main():
    ap = argparse.ArgumentParser(description="M4c -- adjacency generalisation of the forest policy")
    ap.add_argument("--ckpt", default=os.path.join(OUT, "m4b_policy.pt"))
    ap.add_argument("--n_adjacent", type=int, default=2, help="neighbours per anchor")
    ap.add_argument("--min_cohort", type=int, default=150)
    ap.add_argument("--min_base", type=float, default=0.10,
                    help="floor on a concept's train endorse-rate, so cohorts survive the filter")
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--i_have_author_goahead", action="store_true")
    a = ap.parse_args()
    assert a.i_have_author_goahead, "REFUSING TO RUN without --i_have_author_goahead"
    torch.set_num_threads(a.threads)
    np.random.seed(SEED)

    from strategy_channel_suite import (build_greedy_env, genv_mean, genv_eval_multi,
                                        CERT_TOWER, CERT_CONC)
    from ndcg_tree import build_pool
    from policy_net import Policy, build_phi
    from tree_route import route
    from m4b_eval import load_tree, policy_rollout

    log("#" * 100)
    log("# M4c -- DOES THE POLICY GENERALISE TO CLUSTER-ADJACENT FACTS? VALIDATION ONLY.")
    log("#" * 100)
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    log(f"[ckpt] epoch {ck['epoch']}")

    # ---------------------------------------------------------------- neighbours, from TRAIN only
    Et = build_greedy_env(CERT_TOWER, "train", CERT_CONC, "m4c_lift", cohort_slice=(0, 20000))
    Vt, Ft = np.asarray(Et["V"]), np.asarray(Et["foldable"])
    end_t = ((Ft > 0) & (Vt > 0)).astype(np.float32)          # (n_train_slice, C)
    base = end_t.mean(0)                                      # P(endorse c)
    lift = {}
    for an, cid in ANCHORS.items():
        m = end_t[:, cid] > 0
        joint = end_t[m].mean(0)                              # P(endorse c | endorse anchor)
        lift[an] = joint / np.maximum(base, 1e-9)
    L = np.stack([lift[an] for an in ANCHORS], 0)             # (6, C)
    maxlift = L.max(0)
    anchor_ids = set(ANCHORS.values())

    picks = []                                                # (label, kind, anchor, concept)
    for i, an in enumerate(ANCHORS):
        order = np.argsort(-L[i])
        got = 0
        for c in order:
            c = int(c)
            if c in anchor_ids or base[c] < a.min_base or base[c] > 0.60:
                continue
            picks.append({"kind": "adjacent", "anchor": an, "concept": c,
                          "lift": float(L[i, c]), "base": float(base[c])})
            got += 1
            if got >= a.n_adjacent:
                break
    ar_adj = np.mean([p["base"] for p in picks])
    far_pool = [int(c) for c in np.argsort(maxlift)
                if int(c) not in anchor_ids and a.min_base <= base[int(c)] <= 0.60]
    far_pool.sort(key=lambda c: abs(base[c] - ar_adj))        # match on answer rate
    for c in far_pool[:len(picks)]:
        picks.append({"kind": "far", "anchor": None, "concept": int(c),
                      "lift": float(maxlift[c]), "base": float(base[c])})
    del Et
    log(f"[picks] {sum(p['kind']=='adjacent' for p in picks)} adjacent, "
        f"{sum(p['kind']=='far' for p in picks)} far; mean endorse-rate "
        f"adjacent {ar_adj:.3f} / far {np.mean([p['base'] for p in picks if p['kind']=='far']):.3f}")

    # ---------------------------------------------------------------------------- the measurement
    E = build_greedy_env(CERT_TOWER, "validation", CERT_CONC, "m4c")
    rows_all = np.asarray(E["rows"], np.int64)
    f0, _ = genv_mean(E, [], rows_all)
    assert round(f0, 8) == round(VAL_M0_FULL, 8), f"CANONICAL SNAP FAILED: {f0!r}"
    log(f"[gate] canonical m=0 snap full={f0!r} -- exact")
    pool, _i, _c = build_pool(E, 500, 500)
    assert [list(q) for q in pool] == ck["pool"], "pool moved since training"
    pidx = {tuple(q): i for i, q in enumerate(pool)}
    n_pool = len(pool)

    from m3_show_paths import build_names
    cname, _iname = build_names()

    # phi over pool + every volunteered concept we need, under the TRAINING standardisation
    ext = [tuple(q) for q in ck["ext"]]
    extra = [(1, p["concept"]) for p in picks if (1, p["concept"]) not in ext]
    ext_all = ext + [q for q in dict.fromkeys(extra)]
    norm = (np.asarray(ck["norm"][0], np.float32), np.asarray(ck["norm"][1], np.float32))
    phi_ext, _ = build_phi(E, ext_all, norm=norm)
    eidx = {q: i for i, q in enumerate(ext_all)}
    pol = Policy(ck["d_z"], ck["d_phi"], n_pool, arch=ck["arch"], state=ck["state"], use_pos=True)
    pol.load_state_dict(ck["sd"])
    pol.eval()

    static = [(int(x), int(y)) for x, y in
              json.load(open(os.path.join(B, "greedy_static_policyB_combined.json")))["sequence"]]
    trees = {}
    for t in TREES:
        _d, nd = load_tree(os.path.join(B, "coldtree_v2.json" if t == "cold" else f"m3v2_{t}.json"))
        trees[t] = nd
    V, F = np.asarray(E["V"]), np.asarray(E["foldable"])
    endorse = (F > 0) & (V > 0)
    any_anchor = np.zeros(endorse.shape[0], bool)
    for cid in ANCHORS.values():
        any_anchor |= endorse[:, cid]

    results = []
    for p in picks:
        c = p["concept"]
        cohort = rows_all[endorse[rows_all, c] & ~any_anchor[rows_all]]
        nm = cname.get(c, f"concept{c}")
        if len(cohort) < a.min_cohort:
            log(f"[skip] {p['kind']:<8} '{nm}' (concept {c}): cohort {len(cohort)} < {a.min_cohort}")
            continue
        # ARM 1  fold + ANCHOR SLOT: the fact is presented exactly as a taught anchor is.
        # ARM 2  fold ONLY: the same evidence enters the latent, but the anchor slot stays empty,
        #        so the policy sees an ordinary cold state. Separating these says whether any damage
        #        comes from the NOVEL TOKEN in a slot only six values ever occupied during training,
        #        or from the fold itself.
        seq_pol = policy_rollout(E, pol, phi_ext, pool, cohort, c,
                                 eidx[(1, c)], pidx.get((1, c), -1))
        seq_fold = policy_rollout_foldonly(E, pol, phi_ext, pool, cohort, c)
        row = {"kind": p["kind"], "anchor": p["anchor"], "concept": c, "name": nm,
               "foldonly": {},
               "lift": p["lift"], "cohort": int(len(cohort)), "resembles": {}, "budgets": {}}
        # which tree does the policy's interview resemble?
        for t in TREES:
            seq_t, _dg = route(E, cohort, trees[t], TOTAL_L, static=static, mask=[(1, c)])
            root = float(np.mean([tuple(x[0]) == tuple(y[0]) for x, y in zip(seq_pol, seq_t)]))
            whole = float(np.mean([[tuple(q) for q in x] == [tuple(q) for q in y]
                                   for x, y in zip(seq_pol, seq_t)]))
            row["resembles"][t] = {"root": root, "whole": whole, "seq": seq_t}
        # value
        def ev(seqs, K):
            iseq, cseq = [], []
            for s in seqs:
                ask = s[:K]
                iseq.append(list(dict.fromkeys([q[1] for q in ask if q[0] == 0])))
                cseq.append(list(dict.fromkeys([c] + [q[1] for q in ask if q[0] == 1])))
            full, tail = genv_eval_multi(E, cohort, iseq, cseq)
            return np.asarray(full, np.float64), np.asarray(tail, np.float64)
        for K in BUDGETS:
            pf, pt = ev(seq_pol, K)
            ff, ft = ev(seq_fold, K)
            cf, ct = ev(row["resembles"]["cold"]["seq"], K)
            mf, lof, hif = paired_boot(ff - cf)
            row["foldonly"][str(K)] = {"full": float(np.nanmean(ff)),
                                       "foldonly_minus_cold_full": mf, "ci": [lof, hif]}
            m, lo, hi = paired_boot(pf - cf)
            row["budgets"][str(K)] = {"policy_full": float(np.nanmean(pf)),
                                      "cold_full": float(np.nanmean(cf)),
                                      "policy_minus_cold_full": m, "ci": [lo, hi],
                                      "policy_minus_cold_tail": paired_boot(pt - ct)[0]}
        row["foldonly_root_matches_cold"] = float(np.mean(
            [tuple(x[0]) == tuple(y[0]) for x, y in zip(seq_fold, row["resembles"]["cold"]["seq"])]))
        best = max(TREES, key=lambda t: row["resembles"][t]["root"])
        log(f"[{p['kind']:<8}] '{nm}' n={len(cohort):<5} lift={p['lift']:.2f} | "
            f"root-resembles {best} ({row['resembles'][best]['root']:.3f}), "
            f"cold {row['resembles']['cold']['root']:.3f}"
            + (f", teacher-{p['anchor']} {row['resembles'][p['anchor']]['root']:.3f}"
               if p["anchor"] else "")
            + f" | K=15 anchor-slot {row['budgets']['15']['policy_minus_cold_full']:+.5f}"
            + f" fold-only {row['foldonly']['15']['foldonly_minus_cold_full']:+.5f}"
            + f" (fold-only root==cold {row['foldonly_root_matches_cold']:.3f})")
        for t in TREES:
            row["resembles"][t].pop("seq", None)
        results.append(row)
        json.dump({"analysis": "m4c_adjacent", "split": "validation", "test_touched": False,
                   "results": results}, open(os.path.join(OUT, "m4c_adjacent.json"), "w"), indent=1)

    log("=" * 100)
    for kind in ("adjacent", "far"):
        sub = [r for r in results if r["kind"] == kind]
        if not sub:
            continue
        own = [r["resembles"][r["anchor"]]["root"] for r in sub if r["anchor"]]
        cold = [r["resembles"]["cold"]["root"] for r in sub]
        gain = [r["budgets"]["15"]["policy_minus_cold_full"] for r in sub]
        fgain = [r["foldonly"]["15"]["foldonly_minus_cold_full"] for r in sub]
        froot = [r["foldonly_root_matches_cold"] for r in sub]
        log(f"[{kind}] n={len(sub)} cohorts | mean root-agreement with the COLD tree "
            f"{np.mean(cold):.3f}" + (f", with its OWN anchor's teacher {np.mean(own):.3f}" if own else "")
            + f" | mean K=15 anchor-slot {np.mean(gain):+.5f}"
            + f", fold-only {np.mean(fgain):+.5f} (root==cold {np.mean(froot):.3f})")
    log("[done]")


if __name__ == "__main__":
    main()
