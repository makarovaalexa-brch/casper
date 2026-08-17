r"""M1 Stage A3 -- THE FOREST'S STARTING POINTS (author's four-step design, 2026-08-17).

    *"1. start from more to allow for merge 2. pick the optimal opener 3. pick optimal opener from
     reddit vocab, compare if taste drifts from the cluster -- perhaps it doesn't cost us much
     4. cluster based on reddit openers and compare drift. that should give us all our forest
     starting points"*

This file does steps 1-3. Step 4 (clustering the Reddit ARRIVAL states) is `m1_reddit_clusters.py`.

STEP 1 -- START HIGH, MERGE ON ANCHORABILITY
    k is not chosen; it is a RESULT. Start at `--start_k` and repeatedly merge the weakest cluster
    into its nearest neighbour until every survivor has (a) >= MIN_SIZE members and (b) an anchor
    whose HELD-OUT margin >= MIN_MARGIN. A merge is accepted only if the merged cluster's anchor is
    at least as good as the weak parent's -- otherwise those users are a genuine RESIDUAL and are
    reported as unroutable rather than glued somewhere convenient.

STEP 2 -- OPTIMAL OPENER, FULL ATTRIBUTE VOCABULARY (the ceiling)
STEP 3 -- OPTIMAL OPENER, REDDIT VOCABULARY ONLY (what plain language costs)
    The Reddit vocabulary is the set of concepts whose tag string actually appears as a mapped genre
    or tone in the M1b census -- i.e. words real people used. DRIFT is then reported three ways:
    margin lost, state distance between the two anchors' folds, and the overlap of their staters.

DISTINCTIVENESS, not coverage (author): margin = (share of an anchor's staters in this cluster)
minus (largest share in any other cluster), on a HELD-OUT half. A tag everybody states everywhere
scores ~0 however high its coverage.

VALIDATION ONLY. No TEST users are built, loaded or named. Nothing is scored: no decoder matmul,
no NDCG, no targets read.
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
import re
import time

import numpy as np
import torch

OUT_DIR = os.path.join(_ROOT, "experiments", "battery", "m1")
TAGS = os.path.join(_ROOT, "data", "movielens", "genome-tags.csv")
MAPPING = os.path.join(_ROOT, "experiments", "battery", "m1b_mapping.json")
SEED = 98765
VAL_M0_FULL = 0.15686538118354684
MIN_STATERS = 50
MIN_SIZE = 1200          # Fable's cohort floor (500 sits inside the -0.724 noise regime)
MIN_MARGIN = 0.10        # an anchor below this does not "push them away from other clusters"
MIN_COV = 0.25           # and it must still reach a usable share of its own cluster


def log(m: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def anchor_table(facts, lab, k, c, memset, n_mem, fit_mask, vocab=None):
    """Best anchor for cluster `c` by HELD-OUT margin, restricted to `vocab` if given."""
    best = None
    for cid, us in facts:
        if vocab is not None and cid not in vocab:
            continue
        ho = us[~fit_mask[us]]
        if len(ho) < 10:
            continue
        cov = float(sum(1 for u in us if u in memset)) / max(n_mem, 1)
        if cov < MIN_COV:
            continue
        sh = np.bincount(lab[ho], minlength=k) / len(ho)
        margin = float(sh[c] - np.max(np.delete(sh, c)))
        if best is None or margin > best["margin"]:
            best = {"concept": int(cid), "coverage": round(cov, 4),
                    "purity_heldout": round(float(sh[c]), 4), "margin": round(margin, 4),
                    "staters": int(len(us))}
    return best


def main() -> None:
    ap = argparse.ArgumentParser(description="M1 A3 -- forest starting points (steps 1-3)")
    ap.add_argument("--start_k", type=int, default=12)
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--i_have_author_goahead", action="store_true")
    a = ap.parse_args()
    assert a.i_have_author_goahead, "REFUSING TO RUN without --i_have_author_goahead"
    torch.set_num_threads(a.threads); np.random.seed(SEED)

    from strategy_channel_suite import (build_greedy_env, contract_selftest, genv_mean,
                                        CERT_TOWER, CERT_CONC)
    from concept_fold import fold_items_frozen
    from ndcg_tree import leak_assert
    from m1_ideal import kmeans

    log("#" * 100)
    log("# M1 STAGE A3 -- FOREST STARTING POINTS (steps 1-3). VALIDATION ONLY.")
    log("#" * 100)

    zf = np.load(os.path.join(OUT_DIR, "z_full.npz"))
    Z_full, rows = zf["z"], zf["rows"]
    n = len(rows)

    E = build_greedy_env(CERT_TOWER, "validation", CERT_CONC, "m1a3")
    contract_selftest(E, E["rows"]); leak_assert(E, E["rows"], "m1a3/val")
    assert list(np.asarray(E["rows"], np.int64)) == list(rows.astype(np.int64))
    f0, _ = genv_mean(E, [], E["rows"])
    assert round(f0, 8) == round(VAL_M0_FULL, 8), f"CANONICAL SNAP FAILED: {f0!r}"
    log(f"[gate] canonical m=0 snap full={f0!r} -- exact")

    # ---- vocabularies ---------------------------------------------------------------------------
    tagname = {}
    with open(TAGS, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            tagname[int(r["tagId"])] = r["tag"].strip()
    tags = list(E["sh"]["tags"])
    cname = {cid: tagname.get(int(t), str(t)) for cid, t in enumerate(tags)}
    byname = {}
    for cid, nm in cname.items():
        s = str(nm).lower()
        byname.setdefault(s, cid); byname.setdefault(s.replace("-", " "), cid)

    mp = json.load(open(MAPPING, encoding="utf-8"))
    reddit_words = set()
    for slot in ("genres_pos", "tone"):
        for tgt, _cnt in mp["slots"].get(slot, {}).get("mapped_targets", []):
            reddit_words.add(str(tgt).lower())
    reddit_vocab = {byname[w] for w in reddit_words if w in byname}
    log(f"[vocab] Reddit words seen in the census: {len(reddit_words)}; "
        f"of which present as concepts: {len(reddit_vocab)}")
    log(f"[vocab] examples: {sorted(cname[c] for c in list(reddit_vocab))[:15]}")

    # ---- statable concepts ----------------------------------------------------------------------
    V, F = np.asarray(E["V"]), np.asarray(E["foldable"])
    endorse = (F > 0) & (V > 0)
    conc_staters = {}
    for u, r in enumerate(rows):
        for c in np.flatnonzero(endorse[int(r)]):
            conc_staters.setdefault(int(c), []).append(u)
    facts = [(c, np.asarray(us)) for c, us in conc_staters.items() if len(us) >= MIN_STATERS]
    log(f"[facts] {len(facts)} concepts with >= {MIN_STATERS} staters; "
        f"{sum(1 for c, _ in facts if c in reddit_vocab)} of them in the Reddit vocabulary")

    rng = np.random.default_rng(SEED)
    fit_mask = np.zeros(n, bool); fit_mask[rng.permutation(n)[: n // 2]] = True

    # ---- step 1: start high, merge on anchorability ----------------------------------------------
    lab, C = kmeans(Z_full, a.start_k)
    active = list(range(a.start_k))
    history = []
    while True:
        k_eff = len(active)
        remap = {c: i for i, c in enumerate(active)}
        lab2 = np.array([remap[c] for c in lab])
        sizes = np.bincount(lab2, minlength=k_eff)
        anchors = []
        for i in range(k_eff):
            mem = np.flatnonzero(lab2 == i)
            anchors.append(anchor_table(facts, lab2, k_eff, i, set(mem.tolist()), len(mem), fit_mask))
        weak = [i for i in range(k_eff)
                if sizes[i] < MIN_SIZE or anchors[i] is None or anchors[i]["margin"] < MIN_MARGIN]
        log(f"[merge] k={k_eff} sizes={sizes.tolist()} weak={weak}")
        if not weak or k_eff <= 3:
            lab, k_eff_final, anchors_final, sizes_final = lab2, k_eff, anchors, sizes
            break
        # weakest = lowest margin (None counts as -1)
        wi = min(weak, key=lambda i: (anchors[i]["margin"] if anchors[i] else -1.0))
        cents = np.stack([Z_full[lab2 == i].mean(0) for i in range(k_eff)])
        cents /= np.maximum(np.linalg.norm(cents, axis=1, keepdims=True), 1e-9)
        sim = cents @ cents[wi]; sim[wi] = -9
        tgt = int(np.argmax(sim))
        log(f"[merge]   merging cluster {wi} (n={sizes[wi]}, "
            f"margin={anchors[wi]['margin'] if anchors[wi] else None}) into nearest {tgt}")
        history.append({"k_before": k_eff, "merged": int(wi), "into": tgt,
                        "n": int(sizes[wi]),
                        "margin": anchors[wi]["margin"] if anchors[wi] else None})
        lab = np.array([tgt if c == wi else c for c in lab2])
        active = list(range(k_eff))
        active.remove(wi)
        lab = np.array([c - 1 if c > wi else c for c in lab])
        active = list(range(k_eff - 1))

    # ---- steps 2 and 3: two vocabularies on the SAME clusters, and the drift ----------------------
    out = []
    for i in range(k_eff_final):
        mem = np.flatnonzero(lab == i); memset = set(mem.tolist())
        a_all = anchor_table(facts, lab, k_eff_final, i, memset, len(mem))
        a_red = anchor_table(facts, lab, k_eff_final, i, memset, len(mem), vocab=reddit_vocab)
        drift = None
        if a_all and a_red:
            zi = fold_items_frozen(ctx_enc := E["ctx"].enc,
                                   [(np.zeros(0, np.int64), np.zeros(0, np.int64))])
            with torch.no_grad():
                za = E["net"](zi, [[a_all["concept"]]], [[1.0]],
                              [[0.0]] if getattr(E["net"], "use_refusal", False) else None)[0]
                zr = E["net"](zi, [[a_red["concept"]]], [[1.0]],
                              [[0.0]] if getattr(E["net"], "use_refusal", False) else None)[0]
            sa = set(conc_staters[a_all["concept"]]) & memset
            sr = set(conc_staters[a_red["concept"]]) & memset
            drift = {
                "margin_lost": round(a_all["margin"] - a_red["margin"], 4),
                "purity_lost": round(a_all["purity_heldout"] - a_red["purity_heldout"], 4),
                "state_distance": round(float(torch.norm(za - zr).item()), 4),
                "anchor_norm": round(float(torch.norm(za).item()), 4),
                "stater_jaccard": round(len(sa & sr) / max(len(sa | sr), 1), 4),
                "same_concept": a_all["concept"] == a_red["concept"],
            }
        out.append({"cluster": i, "size": int(sizes_final[i]),
                    "prior": round(float(sizes_final[i]) / n, 4),
                    "anchor_full_vocab": (dict(a_all, name=cname[a_all["concept"]]) if a_all else None),
                    "anchor_reddit_vocab": (dict(a_red, name=cname[a_red["concept"]]) if a_red else None),
                    "drift": drift})
        nm_a = out[-1]["anchor_full_vocab"]; nm_r = out[-1]["anchor_reddit_vocab"]
        log(f"  [c{i} n={sizes_final[i]}] full: {nm_a['name']!r} m={nm_a['margin']} | "
            f"reddit: {nm_r['name']!r} m={nm_r['margin']} | "
            f"drift margin {drift['margin_lost'] if drift else None} "
            f"state {drift['state_distance'] if drift else None}/{drift['anchor_norm'] if drift else None}"
            if nm_a and nm_r else f"  [c{i}] full={nm_a} reddit={nm_r}")

    res = {"analysis": "m1_a3_forest_starting_points", "split": "validation", "test_touched": False,
           "start_k": a.start_k, "final_k": k_eff_final, "min_size": MIN_SIZE,
           "min_margin": MIN_MARGIN, "min_coverage": MIN_COV, "min_staters": MIN_STATERS,
           "reddit_vocab_size": len(reddit_vocab), "merge_history": history,
           "canonical_snap_full": f0, "clusters": out}
    with open(os.path.join(OUT_DIR, "m1_forest_starts.json"), "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1)
    np.savez_compressed(os.path.join(OUT_DIR, "m1_forest_labels.npz"), labels=lab, rows=rows)
    log(f"[done] final k={k_eff_final} after {len(history)} merges; wrote m1_forest_starts.json")


if __name__ == "__main__":
    main()
