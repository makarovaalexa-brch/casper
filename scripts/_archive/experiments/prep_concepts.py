"""
prep_concepts.py -- build a genome-tag concept vocabulary aligned to the ML-20M
RecVAE sid vocabulary (INSTRUMENT 2.0 Phase-1.6, Experiment A).

Genome scores (data/movielens/genome-scores.csv: movieId,tagId,relevance) share
movieIds with ML-20M. We map movieId -> sid via proc/unique_sid.txt (line index = sid).
Select ~200 high-coverage tags (coverage = #vocab items with relevance>=0.5) and store
a dense item_tag relevance matrix (n_items x n_sel) for downstream concept-bag folding.

Output: .cache/instrument2/concepts.npz
  item_tag  (n_items x n_sel) float32 relevance (0 where no genome / tag not selected)
  tag_ids   (n_sel,) int32 genome tagId
  tag_names (n_sel,) str
  coverage  (n_sel,) int32
EVAL-ONLY prep, no training, no commits.
"""
import os, json
import numpy as np
import pandas as pd

PROC = os.path.join("data", "ml20m", "proc")
GENOME = os.path.join("data", "movielens", "genome-scores.csv")
TAGS = os.path.join("data", "movielens", "genome-tags.csv")
OUT = os.path.join(".cache", "instrument2", "concepts.npz")
N_TAGS = 200
COV_THRESH = 0.5


def main():
    with open(os.path.join(PROC, "meta.json")) as f:
        n_items = json.load(f)["n_items"]
    # sid -> movieId (line index = sid)
    movie_ids = np.loadtxt(os.path.join(PROC, "unique_sid.txt"), dtype=np.int64)
    assert len(movie_ids) == n_items, (len(movie_ids), n_items)
    mid2sid = {int(m): i for i, m in enumerate(movie_ids)}

    print(f"[concepts] loading genome scores {GENOME} ...", flush=True)
    gs = pd.read_csv(GENOME, dtype={"movieId": np.int64, "tagId": np.int32,
                                    "relevance": np.float32})
    # keep only movies present in our sid vocabulary
    gs = gs[gs["movieId"].isin(mid2sid)]
    gs["sid"] = gs["movieId"].map(mid2sid).astype(np.int64)
    n_movies_genome = gs["movieId"].nunique()
    print(f"[concepts] genome rows in vocab: {len(gs):,} over {n_movies_genome:,} movies "
          f"({100*n_movies_genome/n_items:.1f}% of {n_items} vocab items)", flush=True)

    # coverage per tag = #vocab items with relevance >= threshold
    hi = gs[gs["relevance"] >= COV_THRESH]
    cov = hi.groupby("tagId").size().sort_values(ascending=False)
    sel = cov.head(N_TAGS)
    sel_tags = sel.index.values.astype(np.int32)
    print(f"[concepts] selected {len(sel_tags)} tags by coverage "
          f"(top cov {int(sel.iloc[0])}, min cov {int(sel.iloc[-1])})", flush=True)

    tagnames = pd.read_csv(TAGS).set_index("tagId")["tag"].to_dict()
    names = np.array([str(tagnames.get(int(t), str(t))) for t in sel_tags])

    tag2col = {int(t): j for j, t in enumerate(sel_tags)}
    item_tag = np.zeros((n_items, len(sel_tags)), dtype=np.float32)
    sub = gs[gs["tagId"].isin(tag2col)]
    cols = sub["tagId"].map(tag2col).values
    rows = sub["sid"].values
    item_tag[rows, cols] = sub["relevance"].values

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    np.savez_compressed(OUT, item_tag=item_tag, tag_ids=sel_tags,
                        tag_names=names, coverage=sel.values.astype(np.int32))
    print(f"[concepts] wrote {OUT}  item_tag {item_tag.shape} "
          f"nnz/col mean {(item_tag>0).sum(0).mean():.0f}", flush=True)
    print("  sample tags:", ", ".join(names[:12]), flush=True)


if __name__ == "__main__":
    main()
