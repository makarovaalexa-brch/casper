r"""probe_nearest_tag.py -- is a free-text probe doing open-vocabulary work, or just tag lookup?

For each probe phrase: its nearest genome tags by SBERT cosine, and -- the decisive part -- the top-10
item overlap between the phrase's direction and its nearest tag's CURATED direction. If a probe is a
paraphrase of a tag the adapter was fitted on, the phrase adds nothing: we are reading back a direction
we fitted. Probes must be reported with this number attached, or dropped.

Also fits a LEAVE-THE-NEIGHBOUR-OUT adapter per probe (drop the nearest tag, and anything above a cosine
floor, from the fit) so a probe can be scored WITHOUT its own concept in the training set.

  python src/instrument/probe_nearest_tag.py
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "6")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import sys
import json
import time
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
sys.path.insert(0, os.path.join(_ROOT, "src", "baselines"))
sys.path.insert(0, _HERE)

from train_tower_t2 import log
import run_battery_phaseA as PA
from run_battery_phaseA import build_real_ctx
from sbert_open_concepts import PROBES

OUT = os.path.join(_ROOT, "experiments", "battery", "probe_nearest_tag.json")
DATA = os.path.join(_ROOT, "data", "movielens")
DROP_COS = 0.75          # a tag this close to the probe is treated as the probe's own concept


def main():
    t0 = time.time()
    ctx = build_real_ctx(PA.SNAP_DEFAULT)
    from run_battery_phaseB import attach_belief
    attach_belief(ctx, None)
    Wd = ctx.Wd.numpy().astype(np.float64)
    n = Wd.shape[0]
    tags = list(ctx.tags)
    D = ctx.d_c.numpy().astype(np.float64)

    import csv
    tagname = {}
    with open(os.path.join(DATA, "genome-tags.csv"), newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            tagname[int(r["tagId"])] = r["tag"].strip()
    names = [tagname.get(int(t), str(t)) if str(t).isdigit() else str(t) for t in tags]

    titles = {}
    import pandas as pd
    usid = pd.read_csv(os.path.join(_ROOT, "data", "ml-25m", "proc", "unique_sid.txt"), header=None)
    movie2sid = {int(m): i for i, m in enumerate(usid[0].tolist())}
    with open(os.path.join(DATA, "movies.csv"), newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            s = movie2sid.get(int(r["movieId"]))
            if s is not None:
                titles[s] = r["title"]

    from sentence_transformers import SentenceTransformer
    sb = SentenceTransformer("all-MiniLM-L6-v2")
    S = sb.encode(names, normalize_embeddings=True, batch_size=256,
                  show_progress_bar=False).astype(np.float64)
    P = sb.encode(PROBES, normalize_embeddings=True, show_progress_bar=False).astype(np.float64)

    def ridge_fit(X, Y, beta=1.0):
        return (Y.T @ X) @ np.linalg.inv(X.T @ X + beta * np.eye(X.shape[1]))

    def topk(v, k=10):
        return [int(i) for i in np.argsort(-(Wd @ v))[:k]]

    W_all = ridge_fit(S, D)
    res = {}
    for pi, ph in enumerate(PROBES):
        sims = S @ P[pi]
        order = np.argsort(-sims)[:3]
        near = [{"tag": names[j], "cos": round(float(sims[j]), 3)} for j in order]
        exact = names[order[0]].lower() == ph.lower()

        # does the phrase just reproduce its nearest tag's curated direction?
        dir_all = W_all @ P[pi]
        ov_curated = len(set(topk(dir_all)) & set(topk(D[order[0]]))) / 10.0

        # leave-the-neighbour-out: drop every tag within DROP_COS of the probe, refit, re-ask
        keep = np.where(sims < DROP_COS)[0]
        dropped = [names[j] for j in np.where(sims >= DROP_COS)[0]]
        W_lno = ridge_fit(S[keep], D[keep])
        dir_lno = W_lno @ P[pi]
        ov_self = len(set(topk(dir_all)) & set(topk(dir_lno))) / 10.0

        res[ph] = {"nearest": near, "exact_tag_match": exact,
                   "overlap_with_nearest_tag_curated_dir": ov_curated,
                   "n_dropped_for_lno": len(dropped), "dropped": dropped[:6],
                   "overlap_full_vs_leaveneighbourout": ov_self,
                   "top5_full": [titles.get(i, str(i)) for i in topk(dir_all)[:5]],
                   "top5_lno": [titles.get(i, str(i)) for i in topk(dir_lno)[:5]]}
        flag = "EXACT TAG" if exact else ("near-dup" if near[0]["cos"] >= DROP_COS else "open")
        log(f"[{flag:9s}] {ph!r} nearest={near[0]['tag']!r} cos={near[0]['cos']:.2f} | "
            f"overlap w/ curated {ov_curated:.1f} | dropped {len(dropped)} -> LNO overlap {ov_self:.1f}")
        log(f"            full: {'; '.join(res[ph]['top5_full'][:3])}")
        log(f"            LNO : {'; '.join(res[ph]['top5_lno'][:3])}")

    json.dump({"drop_cos": DROP_COS, "probes": res, "seconds": round(time.time() - t0, 1)},
              open(OUT, "w"), indent=1)
    log(f"[probe] -> {OUT}")


if __name__ == "__main__":
    main()
