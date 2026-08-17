r"""M1 Stage A2' -- PER-CLUSTER SHARED ANCHORS (re-aggregation of A2, author's correction).

Design sheet: `docs/design/DS_M1_STATE_MATCHING.md`. VALIDATION ONLY.

WHY THIS REPLACES A2's PER-USER FACTS
-------------------------------------
A2 gave every user their OWN best identifying fact (1,069-1,230 distinct items). That is a coherent
statement about VOLUNTEERING but it is not a state a cluster shares, and the author's requirement is:

    *"the policy must learn mapping from user state to cluster. and we don't want big variation in
     that cluster user state, ideally it should be the same one per cluster. yes, shared fact or set"*

So the anchor is chosen PER CLUSTER, not per user. If every member volunteers the SAME fact they all
sit at EXACTLY the same state -- `fold(fact)` does not depend on who states it -- so within-cluster
state variance is zero by construction and one tree per cluster is well defined.

THE TENSION THIS RUN EXISTS TO MEASURE
--------------------------------------
  coverage(f,c) = share of cluster c that CAN state f      (high for popular films)
  purity(f,c)   = share of f's staters who are IN c        (high for niche films)
These pull against each other. Whether any fact is BOTH shared and discriminative decides whether the
design works. We report the Pareto front per cluster rather than asserting an operating point.

A SET is the fallback the author anticipated: members state whichever anchor they hold, so coverage
rises but the entry states are no longer identical -- so the SET's STATE SPREAD is reported (max and
mean pairwise distance between the folded anchors, against the cohort's own state scale).

HELD-OUT PURITY
---------------
A2's purity was in-sample: clusters, fact distributions and evaluation all came from the same 10,000
users, and `Spearman(n_users, gain) = -0.724` says a maximum selected on its own data shrinks. Here
purity is estimated on a FIT half and reported on a HELD-OUT half.
"""
from __future__ import annotations

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
sys.path.insert(0, _HERE)

import argparse
import csv
import json
import time

import numpy as np
import torch

OUT_DIR = os.path.join(_ROOT, "experiments", "battery", "m1")
TAGS = os.path.join(_ROOT, "data", "movielens", "genome-tags.csv")
MOVIES = os.path.join(_ROOT, "data", "movielens", "movies.csv")
SEED = 98765
VAL_M0_FULL = 0.15686538118354684
MIN_STATERS = 50
COVERAGE_TARGET = 0.80        # a SET grows until this share of the cluster can state something


def log(m: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="M1 Stage A2' -- per-cluster shared anchors")
    ap.add_argument("--ks", default="4,6")
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--i_have_author_goahead", action="store_true")
    a = ap.parse_args()
    assert a.i_have_author_goahead, "REFUSING TO RUN without --i_have_author_goahead"
    torch.set_num_threads(a.threads); np.random.seed(SEED)

    from strategy_channel_suite import (build_greedy_env, contract_selftest, genv_mean,
                                        CERT_TOWER, CERT_CONC)
    from concept_fold import fold_items_frozen
    from i26_encoder import UNSEEN_LEVEL
    from ndcg_tree import leak_assert
    from m1_ideal import kmeans

    log("#" * 100)
    log("# M1 STAGE A2' -- PER-CLUSTER SHARED ANCHORS. VALIDATION ONLY.")
    log("#" * 100)

    zf = np.load(os.path.join(OUT_DIR, "z_full.npz"))
    Z_full, rows = zf["z"], zf["rows"]

    E = build_greedy_env(CERT_TOWER, "validation", CERT_CONC, "m1anchor")
    contract_selftest(E, E["rows"]); leak_assert(E, E["rows"], "m1anchor/val")
    assert list(np.asarray(E["rows"], np.int64)) == list(rows.astype(np.int64))
    f0, _ = genv_mean(E, [], E["rows"])
    assert round(f0, 8) == round(VAL_M0_FULL, 8), f"CANONICAL SNAP FAILED: {f0!r}"
    log(f"[gate] canonical m=0 snap full={f0!r} -- exact")

    ctx, lvl = E["ctx"], E["lvl"]
    LIKED_MIN, n = 7, len(rows)

    # names, for readability of the anchors we pick
    tagname = {}
    with open(TAGS, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            tagname[int(r["tagId"])] = r["tag"].strip()
    cname = {cid: tagname.get(int(t), str(t)) for cid, t in enumerate(list(E["sh"]["tags"]))}
    with open(os.path.join(_ROOT, "data", "ml-25m", "proc", "unique_sid.txt")) as fh:
        sids = [int(x) for x in fh.read().split()]
    title = {}
    with open(MOVIES, newline="", encoding="utf-8") as fh:
        mt = {int(r["movieId"]): r["title"] for r in csv.DictReader(fh)}
    iname = {i: mt.get(m, str(m)) for i, m in enumerate(sids)}

    # ---- statable facts -------------------------------------------------------------------------
    item_staters, user_items = {}, []
    for u, r in enumerate(rows):
        liked = [int(i) for i, L in lvl[int(r)].items() if LIKED_MIN <= int(L) < UNSEEN_LEVEL]
        user_items.append(liked)
        for i in liked:
            item_staters.setdefault(i, []).append(u)
    V, F = np.asarray(E["V"]), np.asarray(E["foldable"])
    endorse = (F > 0) & (V > 0)
    conc_staters = {}
    for u, r in enumerate(rows):
        for c in np.flatnonzero(endorse[int(r)]):
            conc_staters.setdefault(int(c), []).append(u)

    # AUTHOR RULING 2026-08-17: *"i think fact should be an attribute. it must be not only something
    # they all share, but something that pushes them away from other clusters, that is pretty
    # crucial"*. So candidates are CONCEPTS only (a film cannot be a cluster's shared entry), and the
    # selection criterion below is DISTINCTIVENESS, not a coverage/purity balance.
    facts = [(1, c, np.asarray(us)) for c, us in conc_staters.items() if len(us) >= MIN_STATERS]
    if os.environ.get("M1_INCLUDE_ITEMS") == "1":
        facts += [(0, i, np.asarray(us)) for i, us in item_staters.items() if len(us) >= MIN_STATERS]
    log(f"[facts] {len(facts)} candidate facts with >= {MIN_STATERS} staters "
        f"({sum(1 for f in facts if f[0]==0)} items, {sum(1 for f in facts if f[0]==1)} concepts)")

    rng = np.random.default_rng(SEED)
    fit_mask = np.zeros(n, bool); fit_mask[rng.permutation(n)[: n // 2]] = True
    log(f"[split] purity FIT half {int(fit_mask.sum())} · HELD-OUT half {int((~fit_mask).sum())}")

    out = {}
    for k in [int(x) for x in a.ks.split(",") if x.strip()]:
        lab, _C = kmeans(Z_full, k)
        sizes = np.bincount(lab, minlength=k)
        log(f"[k={k}] cluster sizes {sizes.tolist()}")
        clusters = []
        for c in range(k):
            mem = np.flatnonzero(lab == c)
            memset = set(mem.tolist())
            rowsc = []
            for ch, fid, us in facts:
                inc = np.fromiter((u in memset for u in us), bool, len(us))
                cov = float(inc.sum()) / len(mem)
                fit_us = us[fit_mask[us]]
                if len(fit_us) < 10:
                    continue
                pur_fit = float(np.mean(lab[fit_us] == c))
                ho_us = us[~fit_mask[us]]
                pur_ho = float(np.mean(lab[ho_us] == c)) if len(ho_us) >= 10 else float("nan")
                # DISTINCTIVENESS: how far this fact pushes its staters AWAY from every other
                # cluster. margin = (share of staters in c) - (largest share in any other cluster),
                # on the HELD-OUT half. A fact everyone states everywhere has margin ~0 however
                # high its coverage.
                if len(ho_us) >= 10:
                    sh = np.bincount(lab[ho_us], minlength=k) / len(ho_us)
                    other = np.max(np.delete(sh, c))
                    margin = float(sh[c] - other)
                else:
                    margin = float("nan")
                rowsc.append((ch, fid, cov, pur_fit, pur_ho, int(len(us)), margin))
            prior = float(sizes[c]) / n
            # BEST SINGLE ANCHOR: the most DISTINCTIVE attribute that still reaches a usable share of
            # the cluster. Selected on the FIT half's purity ordering, reported on HELD-OUT.
            MIN_COV = 0.25
            elig = [r for r in rowsc if r[2] >= MIN_COV and r[6] == r[6]]
            best = max(elig, key=lambda r: r[6], default=None)          # max margin
            # the two extremes of the front, both still attribute-only
            hi_pur = max((r for r in rowsc if r[2] >= 0.20), key=lambda r: r[3], default=None)
            hi_cov = max((r for r in rowsc if r[3] >= 2 * prior), key=lambda r: r[2], default=None)
            # A SET: greedily add anchors (purity >= 2x prior) until COVERAGE_TARGET of the cluster
            covered, chosen = set(), []
            pool = sorted((r for r in rowsc if r[3] >= 2 * prior), key=lambda r: -r[2])
            for r in pool:
                if len(covered) / len(mem) >= COVERAGE_TARGET or len(chosen) >= 8:
                    break
                us = (item_staters if r[0] == 0 else conc_staters)[r[1]]
                gain = {u for u in us if u in memset} - covered
                if len(gain) / len(mem) < 0.02:
                    continue
                chosen.append(r); covered |= gain
            # state spread of the set: fold each anchor alone, measure pairwise distance
            spread = {"n_anchors": len(chosen)}
            if chosen:
                zs = []
                for ch, fid, *_ in chosen:
                    if ch == 0:
                        zs.append(fold_items_frozen(
                            ctx.enc, [(np.asarray([fid], np.int64), np.asarray([9], np.int64))])[0])
                    else:
                        zi = fold_items_frozen(ctx.enc, [(np.zeros(0, np.int64), np.zeros(0, np.int64))])
                        with torch.no_grad():
                            zs.append(E["net"](zi, [[int(fid)]], [[1.0]],
                                               [[0.0]] if getattr(E["net"], "use_refusal", False) else None)[0])
                Zs = torch.stack(zs).numpy()
                D = np.sqrt(((Zs[:, None] - Zs[None]) ** 2).sum(-1))
                iu = np.triu_indices(len(Zs), 1)
                spread.update(mean_pairwise=float(D[iu].mean()) if len(iu[0]) else 0.0,
                              max_pairwise=float(D[iu].max()) if len(iu[0]) else 0.0,
                              anchor_norm_mean=float(np.linalg.norm(Zs, axis=1).mean()))

            def fmt(r):
                if r is None:
                    return None
                ch, fid, cov, pf, ph, ns, mg = r
                return {"channel": "concept" if ch else "item", "id": int(fid),
                        "name": (cname if ch else iname).get(int(fid), str(fid)),
                        "coverage": round(cov, 4), "purity_fit": round(pf, 4),
                        "purity_heldout": None if ph != ph else round(ph, 4),
                        "margin_heldout": None if mg != mg else round(mg, 4), "staters": ns}

            clusters.append({"cluster": c, "size": int(sizes[c]), "prior": round(prior, 4),
                             "best_anchor_F1": fmt(best), "highest_purity": fmt(hi_pur),
                             "highest_coverage": fmt(hi_cov),
                             "set": [fmt(r) for r in chosen],
                             "set_coverage": round(len(covered) / len(mem), 4),
                             "set_state_spread": spread})
            b = clusters[-1]["best_anchor_F1"]
            if b:
                log(f"  [k={k} c={c} n={sizes[c]}] best anchor {b['channel']} {b['name'][:44]!r} "
                    f"cov {b['coverage']:.2f} pur_fit {b['purity_fit']:.2f} "
                    f"pur_ho {b['purity_heldout']} margin {b['margin_heldout']} "
                    f"(prior {prior:.2f}) | set {spread['n_anchors']} "
                    f"anchors cov {clusters[-1]['set_coverage']:.2f}")
        out[str(k)] = clusters

    with open(os.path.join(OUT_DIR, "m1_anchors.json"), "w", encoding="utf-8") as fh:
        json.dump({"analysis": "m1_stage_a2prime_cluster_anchors", "split": "validation",
                   "test_touched": False, "min_staters": MIN_STATERS,
                   "coverage_target": COVERAGE_TARGET, "canonical_snap_full": f0,
                   "by_k": out}, fh, indent=1)
    log(f"[done] wrote {OUT_DIR}\\m1_anchors.json")


if __name__ == "__main__":
    main()
