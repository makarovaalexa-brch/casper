r"""M1 -- SWEEP k on the de-era'd taste states; no merge ratchet.

TWO DESIGN CORRECTIONS THIS FILE CARRIES
  1. The merge loop only stopped when EVERY cluster passed, so one stubbornly unanchorable cluster
     dragged k all the way down and destroyed good clusters on the way. It is replaced by a SWEEP:
     for each k, report how many clusters pass the anchor gate and how many users they cover.
  2. The forest does not need to cover everyone. Users in unanchorable regions get the COLD TREE,
     which the policy keeps verbatim (design sheet M.1/M4), and the census says 17% of real requests
     volunteer nothing routable anyway. Partial coverage is the design, not a failure.

States are the de-era'd ones (`deera_*`): the era axis (rho +0.875 with release year) was projected
out, dropping the leading directions' year correlation from +0.362/+0.709 to +0.061/+0.169.

Gate is scale-aware (prior units) so k values are comparable: purity >= LIFT_MIN x prior AND
margin >= MARGIN_PRIOR_MULT x prior, measured on a HELD-OUT half of the train users.
NDCG is never computed on train users (D.6b).
"""
from __future__ import annotations

import os
import sys

_H = os.path.dirname(os.path.abspath(__file__))
_R = os.path.abspath(os.path.join(_H, "..", ".."))
sys.path.insert(0, _H)

import argparse
import csv
import json
import time

import numpy as np
import torch
from scipy import sparse

M1 = os.path.join(_R, "experiments", "battery", "m1")
TRAIN_DIR = os.path.join(M1, "train")
VOCAB_FULL = os.path.join(_R, "experiments", "battery", "m1b_vocab_full.json")
TAGS = os.path.join(_R, "data", "movielens", "genome-tags.csv")
MOVIES = os.path.join(_R, "data", "movielens", "movies.csv")
SEED = 98765
MIN_STATERS, MIN_COV, LIFT_MIN, MARGIN_PRIOR_MULT = 200, 0.25, 2.0, 0.5


def log(m: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def load(prefix: str):
    files = sorted((f for f in os.listdir(TRAIN_DIR) if f.startswith(prefix)),
                   key=lambda x: int(x.split("_")[1]))
    assert files, f"no {prefix}* slices under {TRAIN_DIR}"
    Z, indices, indptr, off, C = [], [], [0], 0, None
    for f in files:
        d = np.load(os.path.join(TRAIN_DIR, f))
        Z.append(d["z"])
        C = int(d["C"])
        ip, ix = d["end_indptr"], d["end_indices"]
        indices.append(ix)
        indptr.extend((ip[1:] + off).tolist())
        off += len(ix)
    Z = np.concatenate(Z, 0)
    End = sparse.csr_matrix((np.ones(off, np.bool_), np.concatenate(indices),
                             np.asarray(indptr, np.int64)), shape=(Z.shape[0], C))
    return Z, End


def main() -> None:
    ap = argparse.ArgumentParser(description="M1 sweep -- anchorable clusters vs k")
    ap.add_argument("--prefix", default="deera_")
    ap.add_argument("--ks", default="6,8,12,16,20,30,40")
    ap.add_argument("--report_k", type=int, default=0)
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--i_have_author_goahead", action="store_true")
    a = ap.parse_args()
    assert a.i_have_author_goahead, "REFUSING TO RUN without --i_have_author_goahead"
    torch.set_num_threads(a.threads)
    np.random.seed(SEED)

    from strategy_channel_suite import build_greedy_env, CERT_TOWER, CERT_CONC
    from m1_ideal import kmeans

    Z, End = load(a.prefix)
    n = Z.shape[0]
    log(f"[in] {a.prefix} states {Z.shape}; endorsement {End.nnz} ({End.nnz / n:.1f}/user)")

    csc = End.tocsc()
    facts = [(c, csc.indices[csc.indptr[c]:csc.indptr[c + 1]]) for c in range(End.shape[1])]
    facts = [(c, us) for c, us in facts if len(us) >= MIN_STATERS]
    fact_ids = np.asarray([c for c, _ in facts])
    fact_us = [us for _, us in facts]
    rng = np.random.default_rng(SEED)
    fit = np.zeros(n, bool)
    fit[rng.permutation(n)[: n // 2]] = True
    fact_ho = [us[~fit[us]] for us in fact_us]
    nf = len(facts)
    log(f"[facts] {nf} concepts with >= {MIN_STATERS} staters")

    red_words = {w.lower() for w in json.load(open(VOCAB_FULL, encoding="utf-8"))["tags"]}

    E = build_greedy_env(CERT_TOWER, "validation", CERT_CONC, "m1sweep")
    tagname = {}
    with open(TAGS, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            tagname[int(r["tagId"])] = r["tag"].strip()
    cname = {cid: tagname.get(int(t), str(t)) for cid, t in enumerate(list(E["sh"]["tags"]))}
    red_mask = np.asarray([str(cname.get(int(c), "")).lower() in red_words for c in fact_ids], bool)
    Wd = np.asarray(E["ctx"].Wd)
    bd = np.asarray(E["ctx"].bd)
    sc_global = Z.mean(0) @ Wd.T + bd
    with open(os.path.join(_R, "data", "ml-25m", "proc", "unique_sid.txt")) as fh:
        sids = [int(x) for x in fh.read().split()]
    with open(MOVIES, newline="", encoding="utf-8") as fh:
        mt = {int(r["movieId"]): r["title"] for r in csv.DictReader(fh)}
    iname = {i: mt.get(m, str(m)) for i, m in enumerate(sids)}
    log(f"[vocab] {int(red_mask.sum())} of {nf} candidates are Reddit words")

    def scan(lab, k):
        sizes = np.bincount(lab, minlength=k).astype(float)
        cnt = np.zeros((nf, k))
        cnt_ho = np.zeros((nf, k))
        for j in range(nf):
            cnt[j] = np.bincount(lab[fact_us[j]], minlength=k)
            cnt_ho[j] = np.bincount(lab[fact_ho[j]], minlength=k)
        cov = cnt / np.maximum(sizes, 1)[None, :]
        tot = cnt_ho.sum(1, keepdims=True)
        sh = cnt_ho / np.maximum(tot, 1e-9)
        part = np.partition(sh, -2, axis=1)
        t1, t2 = part[:, -1], part[:, -2]
        margin = sh - np.where(sh == t1[:, None], t2[:, None], t1[:, None])
        return sizes, cov, sh, margin, (tot[:, 0] >= 50)

    def best(cov, sh, margin, ok, c, mask=None):
        e = ok & (cov[:, c] >= MIN_COV)
        if mask is not None:
            e = e & mask
        if not e.any():
            return None
        idx = np.flatnonzero(e)
        j = idx[int(np.argmax(margin[idx, c]))]
        return {"concept": int(fact_ids[j]), "name": cname.get(int(fact_ids[j]), "?"),
                "coverage": round(float(cov[j, c]), 4), "purity": round(float(sh[j, c]), 4),
                "margin": round(float(margin[j, c]), 4)}

    rows, detail = [], {}
    for k in [int(x) for x in a.ks.split(",") if x.strip()]:
        lab, _ = kmeans(Z, k)
        sizes, cov, sh, margin, ok = scan(lab, k)
        priors = sizes / sizes.sum()
        passing = []
        for i in range(k):
            b = best(cov, sh, margin, ok, i)
            if b and b["purity"] >= LIFT_MIN * priors[i] and \
                    b["margin"] >= MARGIN_PRIOR_MULT * priors[i]:
                passing.append((i, b))
        cover = int(sum(sizes[i] for i, _ in passing))
        rows.append({"k": k, "n_pass": len(passing), "users_covered": cover,
                     "share_covered": round(cover / n, 4), "min_size": int(sizes.min()),
                     "median_size": int(np.median(sizes))})
        log(f"[k={k:>2}] {len(passing):>2}/{k} clusters anchorable · {cover:>6} users "
            f"({cover / n:.1%}) · sizes min {int(sizes.min())} median {int(np.median(sizes))}")
        if k == a.report_k:
            log(f"--- anchors at k={k} ---")
            det = []
            for i, b in passing:
                br = best(cov, sh, margin, ok, i, mask=red_mask)
                cen = Z[lab == i].mean(0)
                s = cen @ Wd.T + bd
                top = [int(j) for j in np.argsort(-(s - sc_global))[:5]]
                films = [iname.get(j, "?") for j in top]
                det.append({"cluster": i, "size": int(sizes[i]), "prior": round(float(priors[i]), 4),
                            "anchor": b, "anchor_reddit": br, "films": films})
                log(f"  c{i} n={int(sizes[i])} anchor={b['name']!r} m={b['margin']} "
                    f"purity={b['purity']} (prior {priors[i]:.3f})")
                log(f"     reddit={br['name']!r} m={br['margin']}" if br else "     reddit=NO ANCHOR")
                log(f"     films: {', '.join(films)}")
            detail[str(k)] = det

    json.dump({"analysis": "m1_sweep", "prefix": a.prefix, "fitted_on": "train (140,768)",
               "ndcg_on_train": "never computed -- D.6b", "test_touched": False,
               "gate": {"lift_min": LIFT_MIN, "margin_prior_mult": MARGIN_PRIOR_MULT,
                        "min_cov": MIN_COV, "min_staters": MIN_STATERS},
               "rows": rows, "detail": detail},
              open(os.path.join(M1, f"m1_sweep_{a.prefix.strip('_')}.json"), "w"), indent=1)
    log("[done]")


if __name__ == "__main__":
    main()
