"""pseudo_item_concept.py -- concept-answer FOLD operator (ablation harness, NOT a src baseline).

Lives in scripts/baselines/ on purpose: this is the G2 concept-channel ablation apparatus, not a
paper baseline. It provides ONE thing cleanly -- the operator that folds a stated concept answer into a
frozen item-item / autoencoder recommender (EASE or RecVAE) as the CENTROID of the concept's member
items -- plus a STUB eval entry to exercise the code path. The real G2 usage (which users, which
concepts, elicitation curve) is wired later.

CONCEPT = a MovieLens genome tag (data/movielens/genome-scores.csv, genome-tags.csv). Membership is
thresholded relevance:  members(tag) = { item : relevance(item, tag) >= tau }, restricted to the
project 18,430-item catalogue via the split's unique_sid.txt (movieId -> contiguous sid).

FOLD OPERATOR (the deliverable):
  A concept answer is a pseudo-item = the (L1-normalized) indicator over its member items:
      v_c[sid] = 1/|members(c)|   for sid in members(c),  else 0        (centroid of member items)
  Folding answers {(c, weight)} into a user's fold-in row x gives the augmented input
      x_aug = x + SUM_c weight_c * v_c
  which is then scored by the UNCHANGED recommender:
      EASE / RecVAE:  score = predict(x_aug)      (x_aug @ B for EASE; encoder(x_aug) for RecVAE)
  For EASE this means the concept contributes weight_c * (mean of member items' B-rows) -- i.e. it
  behaves exactly as if the user had liked the "average member movie". The operator is model-agnostic:
  it only produces x_aug; any predict_fn from src/baselines consumes it. Negative weights express a
  disliked concept (fold subtracts the centroid). No truncation: ALL member items are used.

Interface:
  build_concept_matrix(proc, tau) -> (C csr [n_concepts x n_items], tag_names)  # centroid rows v_c
  fold(x_csr, answers, C) -> x_aug csr                                          # the FOLD operator
  answers: list per user of [(concept_idx, weight), ...]

Usage: python scripts/baselines/pseudo_item_concept.py --smoke   # synthetic code-path (NO genome load)
       python scripts/baselines/pseudo_item_concept.py --tau 0.5 # build real C from genome, print stats
"""
import os
import sys
import argparse
import numpy as np
from scipy import sparse

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
sys.path.insert(0, os.path.join(_ROOT, "src", "baselines"))
import metrics as M  # noqa: E402

PROC = os.path.join(_ROOT, "data", "ml-25m", "proc")
GENOME_SCORES = os.path.join(_ROOT, "data", "movielens", "genome-scores.csv")
GENOME_TAGS = os.path.join(_ROOT, "data", "movielens", "genome-tags.csv")
DEFAULT_TAU = 0.5


def load_show2id(proc=PROC):
    """movieId -> contiguous sid, from the split's unique_sid.txt (item vocab order)."""
    with open(os.path.join(proc, "unique_sid.txt")) as f:
        sids = [int(line.strip()) for line in f if line.strip()]
    return {mid: i for i, mid in enumerate(sids)}, len(sids)


def build_concept_matrix(proc=PROC, tau=DEFAULT_TAU, scores_csv=GENOME_SCORES, tags_csv=GENOME_TAGS,
                         log=print):
    """C (n_concepts x n_items) L1-normalized member-centroid rows, restricted to the split catalogue.
    NO truncation: every genome row with relevance >= tau AND movieId in the catalogue is used."""
    import pandas as pd
    show2id, n_items = load_show2id(proc)
    tags = pd.read_csv(tags_csv)                             # tagId, tag
    tag_names = tags.sort_values("tagId")["tag"].tolist()
    tagid2idx = {int(t): i for i, t in enumerate(sorted(tags["tagId"].tolist()))}
    n_concepts = len(tagid2idx)
    log(f"[concept] loading genome scores (relevance>= {tau}); catalogue n_items={n_items}, "
        f"n_concepts={n_concepts}")
    rows, cols, vals = [], [], []
    # stream in chunks: genome-scores.csv is ~15M rows on ML-25M (no full-frame needed).
    for chunk in pd.read_csv(scores_csv, chunksize=2_000_000):
        m = chunk[chunk["relevance"] >= tau]
        m = m[m["movieId"].isin(show2id)]
        if len(m) == 0:
            continue
        rows.extend(tagid2idx[int(t)] for t in m["tagId"].values)
        cols.extend(show2id[int(mid)] for mid in m["movieId"].values)
        vals.extend([1.0] * len(m))
    Cbin = sparse.csr_matrix((vals, (rows, cols)), shape=(n_concepts, n_items), dtype=np.float32)
    # L1-normalize each concept row -> centroid (mean of member indicators)
    deg = np.asarray(Cbin.sum(axis=1)).ravel()
    deg[deg == 0] = 1.0
    C = (sparse.diags(1.0 / deg) @ Cbin).tocsr()
    empty = int((np.asarray(Cbin.sum(axis=1)).ravel() == 0).sum())
    log(f"[concept] built C: {n_concepts} concepts, {Cbin.nnz} memberships, {empty} empty concepts "
        f"(no member >= tau in catalogue)")
    return C, tag_names


def fold(x_csr, answers, C):
    """FOLD operator. x_csr: (b x n_items) base fold-in. answers: list (len b) of [(concept_idx, w),...].
    Returns x_aug csr = x + SUM_c w * C[c].  Model-agnostic (feed x_aug to any predict_fn)."""
    x = x_csr.tocsr().astype(np.float32).copy().tolil()
    for u, ans in enumerate(answers):
        for (cidx, w) in ans:
            row = C.getrow(cidx) * float(w)
            for j, v in zip(row.indices, row.data):
                x[u, j] += v
    return x.tocsr()


# --------------------------------------------------------------------------- STUB eval / smoke
def stub_eval(predict_fn, te_tr, te_te, answers, C, head_mask=None, batch_size=500):
    """STUB: fold `answers` into each fold-in row, then score with predict_fn and report NDCG.
    The REAL G2 protocol (concept selection, per-turn curve) is wired later; this only proves the
    fold->predict->metric path runs end to end."""
    x_aug = fold(te_tr, answers, C)
    return M.evaluate(predict_fn, x_aug, te_te, batch_size=batch_size, head_mask=head_mask)


def _smoke():
    print("[concept][SMOKE] synthetic tiny data (code-path only; NO genome load, NOT the Liang split)")
    rng = np.random.RandomState(0)
    n_users, n_items, n_concepts = 40, 130, 6   # >100 items: metrics uses NDCG@100 / Recall@50
    te_tr = (sparse.random(n_users, n_items, density=0.15, random_state=rng,
                           data_rvs=lambda s: np.ones(s)) > 0).astype(np.float32).tocsr()
    te_te = (sparse.random(n_users, n_items, density=0.1, random_state=rng,
                           data_rvs=lambda s: np.ones(s)) > 0).astype(np.float32).tocsr()
    # synthetic concept matrix: each concept = random member set, L1-normalized (mimics build output)
    Cbin = (sparse.random(n_concepts, n_items, density=0.3, random_state=rng,
                          data_rvs=lambda s: np.ones(s)) > 0).astype(np.float32).tocsr()
    deg = np.asarray(Cbin.sum(axis=1)).ravel(); deg[deg == 0] = 1.0
    C = (sparse.diags(1.0 / deg) @ Cbin).tocsr()
    answers = [[(rng.randint(n_concepts), 1.0)] if rng.rand() < 0.7 else [] for _ in range(n_users)]
    x_aug = fold(te_tr, answers, C)
    added = x_aug.nnz - te_tr.nnz
    assert x_aug.shape == te_tr.shape and added >= 0, "fold changed shape / removed mass"
    # a trivial predict_fn (item popularity) to exercise stub_eval
    pop = np.asarray(te_tr.sum(axis=0)).ravel().astype(np.float32)
    predict_fn = lambda Xc: np.tile(pop, (Xc.shape[0], 1))
    res = stub_eval(predict_fn, te_tr, te_te, answers, C)
    print(f"[concept][SMOKE] OK  fold added ~{added} nnz over {te_tr.nnz}; "
          f"stub_eval full@10={res['ndcg@10']:.4f}  (also compatible with EASE/RecVAE predict_fn)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tau", type=float, default=DEFAULT_TAU)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        _smoke(); return
    C, names = build_concept_matrix(PROC, tau=args.tau)
    deg = np.asarray((C > 0).sum(axis=1)).ravel()
    print(f"[concept] C shape={C.shape}; median members/concept={np.median(deg):.0f}; "
          f"example concepts: {names[:5]}")


if __name__ == "__main__":
    main()
