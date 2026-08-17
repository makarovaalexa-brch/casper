r"""M1 Stage A5 -- TRAIN taste clusters, anchors in both vocabularies, and the FILM sanity check.

Consumes `experiments/battery/m1/train/slice_*.npz` (Stage A4). Design sheet
`docs/design/DS_M1_STATE_MATCHING.md`; author's four-step design in PAPER_B_POLICY_DESIGN.md M.

WHAT CHANGED FROM A3 (all three were defects the author caught)
--------------------------------------------------------------
1. Clusters are fitted on the 140,768 TRAIN users, not the 10,000 validation ones. The old run's
   1,200-member floor capped validation at ~8 clusters and forced blobs of 1,900-3,200 users that no
   single attribute can describe.
2. The Reddit vocabulary is the FULL-TAIL one (352 tags, `m1b_vocab_full.json`), not the 34 targets
   that came from the census's top-50 lists.
3. A cluster whose best anchor has margin <= 0 is reported as **NO ANCHOR**, never as the word that
   happened to be least bad. Printing `romance` for a cluster it does not describe was misleading.

THE FILM SANITY CHECK (author: *"a clean cluster ... with opposite movies"*)
   Each cluster's centroid is decoded through the SAME recommender that will serve it, and its
   top-scoring films are printed beside its attribute anchor. If the anchor and the films disagree,
   the cluster is not coherent and we should see that immediately rather than three stages later.

NDCG IS NEVER COMPUTED ON TRAIN USERS (D.6b). Everything here is a function of their own ratings.
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
from scipy import sparse

M1 = os.path.join(_ROOT, "experiments", "battery", "m1")
TRAIN_DIR = os.path.join(M1, "train")
VOCAB_FULL = os.path.join(_ROOT, "experiments", "battery", "m1b_vocab_full.json")
TAGS = os.path.join(_ROOT, "data", "movielens", "genome-tags.csv")
MOVIES = os.path.join(_ROOT, "data", "movielens", "movies.csv")
SEED = 98765
MIN_STATERS = 200          # train-side: 200 of 140,768 is far more permissive than val's 50 of 10,000
MIN_COV = 0.25
# SCALE-AWARE GATE. An absolute margin threshold is NOT comparable across k: a concept's staters
# spread over more clusters as k grows, so every share -- and every margin -- falls mechanically.
# Applying one number across all k systematically penalises fine granularity and drives the merge
# toward coarse blobs, which is an artifact rather than a finding. The gate is therefore expressed
# in PRIOR units: an anchor must reach LIFT_MIN times the cluster's own prior share, and its margin
# must be at least MARGIN_PRIOR_MULT priors wide. Both are dimensionless in k.
LIFT_MIN = 2.0
MARGIN_PRIOR_MULT = 0.5


def log(m: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def load_slices():
    files = sorted((f for f in os.listdir(TRAIN_DIR) if f.startswith("slice_")),
                   key=lambda f: int(f.split("_")[1]))
    assert files, f"no slices under {TRAIN_DIR}"
    Z, indices, indptr, off, C = [], [], [0], 0, None
    for f in files:
        d = np.load(os.path.join(TRAIN_DIR, f))
        Z.append(d["z"])
        C = int(d["C"])
        ip, ix = d["end_indptr"], d["end_indices"]
        indices.append(ix)
        indptr.extend((ip[1:] + off).tolist())     # ip[0] is always 0; splice on the running offset
        off += len(ix)
    Z = np.concatenate(Z, 0)
    endorse = sparse.csr_matrix((np.ones(off, np.bool_), np.concatenate(indices),
                                 np.asarray(indptr, np.int64)), shape=(Z.shape[0], C))
    assert endorse.nnz == off
    return Z, endorse


def main() -> None:
    ap = argparse.ArgumentParser(description="M1 A5 -- train taste clusters and anchors")
    ap.add_argument("--start_k", type=int, default=40)
    ap.add_argument("--min_size", type=int, default=1200)
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--i_have_author_goahead", action="store_true")
    a = ap.parse_args()
    assert a.i_have_author_goahead, "REFUSING TO RUN without --i_have_author_goahead"
    torch.set_num_threads(a.threads); np.random.seed(SEED)

    from strategy_channel_suite import build_greedy_env, CERT_TOWER, CERT_CONC
    from m1_ideal import kmeans

    log("#" * 100)
    log("# M1 STAGE A5 -- TRAIN taste clusters + anchors + film sanity check")
    log("#" * 100)

    Z, endorse = load_slices()
    n = Z.shape[0]
    log(f"[in] {n} train users · endorsement nnz {endorse.nnz} "
        f"({endorse.nnz / n:.1f} concepts/user)")

    csc = endorse.tocsc()
    staters = {c: csc.indices[csc.indptr[c]:csc.indptr[c + 1]]
               for c in range(endorse.shape[1])}
    facts = [(c, us) for c, us in staters.items() if len(us) >= MIN_STATERS]
    log(f"[facts] {len(facts)} concepts with >= {MIN_STATERS} train staters")

    vf = json.load(open(VOCAB_FULL, encoding="utf-8"))
    red_words = {w.lower() for w in vf["tags"]}
    log(f"[vocab] full-tail Reddit vocabulary: {len(red_words)} tags")

    rng = np.random.default_rng(SEED)
    fit = np.zeros(n, bool); fit[rng.permutation(n)[: n // 2]] = True

    fact_ids = np.asarray([c for c, _ in facts], np.int64)
    fact_us = [us for _, us in facts]
    fact_ho = [us[~fit[us]] for _, us in facts]
    n_fact = len(facts)

    def anchor_scan(lab, k):
        """All (concept, cluster) coverages and held-out margins in one pass.

        The per-concept Python membership test this replaces cost ~555M operations PER MERGE ROUND
        at k=40 over 992 concepts (~14k staters each) -- hours across the merge sequence. Two
        bincounts per concept give the identical quantities in about a second.
        """
        sizes = np.bincount(lab, minlength=k).astype(np.float64)
        cnt = np.zeros((n_fact, k), np.float64)
        cnt_ho = np.zeros((n_fact, k), np.float64)
        for j in range(n_fact):
            cnt[j] = np.bincount(lab[fact_us[j]], minlength=k)
            cnt_ho[j] = np.bincount(lab[fact_ho[j]], minlength=k)
        cov = cnt / np.maximum(sizes, 1)[None, :]
        tot_ho = cnt_ho.sum(1, keepdims=True)
        sh = np.divide(cnt_ho, np.maximum(tot_ho, 1e-9))
        # margin = own share minus the largest share in ANY OTHER cluster
        part = np.partition(sh, -2, axis=1)
        top1, top2 = part[:, -1], part[:, -2]
        other_max = np.where(sh == top1[:, None], top2[:, None], top1[:, None])
        margin = sh - other_max
        ok = (tot_ho[:, 0] >= 50)
        return cov, sh, margin, ok

    def best_anchor(cov, sh, margin, ok, c, vocab_mask=None):
        elig = ok & (cov[:, c] >= MIN_COV)
        if vocab_mask is not None:
            elig = elig & vocab_mask
        if not elig.any():
            return None
        idx = np.flatnonzero(elig)
        j = idx[int(np.argmax(margin[idx, c]))]
        return {"concept": int(fact_ids[j]), "coverage": round(float(cov[j, c]), 4),
                "purity_heldout": round(float(sh[j, c]), 4),
                "margin": round(float(margin[j, c]), 4), "staters": int(len(fact_us[j]))}

    lab, _C0 = kmeans(Z, a.start_k)
    history = []
    while True:
        k = int(lab.max()) + 1
        sizes = np.bincount(lab, minlength=k)
        cov, sh, margin, ok = anchor_scan(lab, k)
        anch = [best_anchor(cov, sh, margin, ok, i) for i in range(k)]
        priors = sizes / sizes.sum()
        def _ok(i):
            if sizes[i] < a.min_size or anch[i] is None:
                return False
            p = float(priors[i])
            return (anch[i]["purity_heldout"] >= LIFT_MIN * p
                    and anch[i]["margin"] >= MARGIN_PRIOR_MULT * p)
        weak = [i for i in range(k) if not _ok(i)]
        log(f"[merge] k={k} min_size={sizes.min()} weak={len(weak)}")
        if not weak or k <= 4:
            break
        wi = min(weak, key=lambda i: (anch[i]["margin"] if anch[i] else -1.0))
        cents = np.stack([Z[lab == i].mean(0) for i in range(k)])
        cents /= np.maximum(np.linalg.norm(cents, axis=1, keepdims=True), 1e-9)
        sim = cents @ cents[wi]; sim[wi] = -9
        tgt = int(np.argmax(sim))
        history.append({"k_before": k, "merged": int(wi), "into": tgt, "n": int(sizes[wi]),
                        "margin": anch[wi]["margin"] if anch[wi] else None})
        lab = np.where(lab == wi, tgt, lab)
        lab = np.where(lab > wi, lab - 1, lab)

    k = int(lab.max()) + 1
    sizes = np.bincount(lab, minlength=k)
    log(f"[merge] SETTLED at k={k}, sizes {sorted(sizes.tolist())}")

    # ---- names + decoder, for the film sanity check ---------------------------------------------
    E = build_greedy_env(CERT_TOWER, "validation", CERT_CONC, "m1a5names")
    tagname = {}
    with open(TAGS, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            tagname[int(r["tagId"])] = r["tag"].strip()
    cname = {cid: tagname.get(int(t), str(t)) for cid, t in enumerate(list(E["sh"]["tags"]))}
    red_ids = {cid for cid, nm in cname.items() if str(nm).lower() in red_words}
    log(f"[vocab] Reddit tags present as concepts: {len(red_ids)}")
    with open(os.path.join(_ROOT, "data", "ml-25m", "proc", "unique_sid.txt")) as fh:
        sids = [int(x) for x in fh.read().split()]
    with open(MOVIES, newline="", encoding="utf-8") as fh:
        mt = {int(r["movieId"]): r["title"] for r in csv.DictReader(fh)}
    iname = {i: mt.get(m, str(m)) for i, m in enumerate(sids)}
    Wd = np.asarray(E["ctx"].Wd); bd = np.asarray(E["ctx"].bd)
    sc_global = Z.mean(0) @ Wd.T + bd            # the average user's scores == the popularity axis

    cov, sh, margin, ok = anchor_scan(lab, k)
    red_mask = np.asarray([int(c) in red_ids for c in fact_ids], bool)
    log(f"[vocab] {int(red_mask.sum())} of {n_fact} candidate concepts are in the Reddit vocabulary")
    out = []
    for i in range(k):
        mem = np.flatnonzero(lab == i)
        a_all = best_anchor(cov, sh, margin, ok, i)
        a_red = best_anchor(cov, sh, margin, ok, i, vocab_mask=red_mask)
        cen = Z[mem].mean(0)
        sc = cen @ Wd.T + bd
        # DISTINCTIVE films, not popular ones: raw top-N returns the globally popular list for every
        # cluster (Shawshank appeared in 4 of 6 on the first pass). Subtract the global centroid's
        # own scores so what is reported is what this cluster likes MORE THAN AVERAGE.
        top = np.argsort(-(sc - sc_global))[:8]
        top_raw = np.argsort(-sc)[:5]
        rec = {"cluster": i, "size": int(sizes[i]), "prior": round(float(sizes[i]) / n, 4),
               "top_films_distinctive": [iname.get(int(j), str(j)) for j in top],
               "top_films_raw": [iname.get(int(j), str(j)) for j in top_raw],
               "anchor_full": (dict(a_all, name=cname[a_all["concept"]])
                               if a_all and a_all["margin"] > 0 else None),
               "anchor_reddit": (dict(a_red, name=cname[a_red["concept"]])
                                 if a_red and a_red["margin"] > 0 else None)}
        if rec["anchor_full"] and rec["anchor_reddit"]:
            rec["drift"] = {
                "margin_lost": round(rec["anchor_full"]["margin"] - rec["anchor_reddit"]["margin"], 4),
                "same_concept": rec["anchor_full"]["concept"] == rec["anchor_reddit"]["concept"]}
        out.append(rec)
        af = rec["anchor_full"]; ar = rec["anchor_reddit"]
        log(f"  [c{i} n={sizes[i]}] full={af['name']!r} m={af['margin']} " if af else
            f"  [c{i} n={sizes[i]}] full=NO ANCHOR ")
        log(f"        reddit={ar['name']!r} m={ar['margin']}" if ar else "        reddit=NO ANCHOR")
        log(f"        distinctive: {', '.join(rec['top_films_distinctive'][:5])}")
        log(f"        (popular  : {', '.join(rec['top_films_raw'][:3])})")

    res = {"analysis": "m1_a5_train_taste_clusters", "fitted_on": "train (140,768)",
           "ndcg_on_train": "never computed -- D.6b", "test_touched": False,
           "start_k": a.start_k, "final_k": k, "min_size": a.min_size,
           "gate": {"lift_min": LIFT_MIN, "margin_prior_mult": MARGIN_PRIOR_MULT,
                    "note": "scale-aware; an absolute margin is not comparable across k"},
           "min_coverage": MIN_COV, "min_staters": MIN_STATERS,
           "reddit_vocab_tags": len(red_ids), "merge_history": history, "clusters": out}
    with open(os.path.join(M1, "m1_train_clusters.json"), "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1, ensure_ascii=False)
    np.savez_compressed(os.path.join(M1, "m1_train_labels.npz"), labels=lab)
    log(f"[done] k={k} after {len(history)} merges; wrote m1_train_clusters.json")


if __name__ == "__main__":
    main()
