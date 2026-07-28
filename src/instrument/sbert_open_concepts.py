r"""sbert_open_concepts.py -- R3 illustration: the instrument takes an ARBITRARY CONTINUOUS DIRECTION.

CLAIM BEING ILLUSTRATED (author, 2026-07-28): not that we contribute a way to map verbal information
into a collaborative latent -- that ground is taken (Balog et al. SIGIR 2021 map an attribute phrase by
BM25 retrieval + item-embedding centroid; Gopfert 2022 / Biyik 2023 fit a per-concept supervised probe).
The claim is narrower and is about the INSTRUMENT: any entity with an embedding is a legal observation,
so the channel set is open WITHOUT BOLT-ONS and a continuous elicitation policy can drive it. Text is the
illustration, not the point.

METHOD. One globally-fitted closed-form ridge adapter from a frozen general-purpose sentence encoder into
the frozen collaborative latent:

    W = Q^T S (S^T S + beta I)^-1          Q = frozen decoder rows (n x d), S = SBERT(item text) (n x 384)

Then ANY phrase gets a direction W @ SBERT(phrase) in the same space as item rows and concept centroids.
No per-concept training, no concept inventory, no retraining -- a matrix multiply at inference.

ITEM REPRESENTATION = the RELEVANCE-WEIGHTED MEAN of an item's genome-tag embeddings (author directive:
NOT titles -- titles carry no plot meaning, so a title-only adapter is a misconceived setup; genome is
also the vocabulary the concept channel is trained on, so membership and closeness live in one semantic
space). Weighting rather than concatenating a top-N string: a top-N cutoff is arbitrary, and MiniLM
truncates at 256 word-pieces so a long string silently loses its tail. Every tag above a relevance floor
contributes, with the weight the genome assigns it.

CONTROLS (these are what make it evidence rather than anecdote):
  (a) PARAPHRASE INVARIANCE -- fold a phrase and a non-tag paraphrase, compare top-10 overlap. Defeats
      the "it is just string matching against the tag vocabulary" reading.
  (b) AT SCALE -- for many genome tags, compare the direction from the tag's TEXT against the direction
      from that tag's genome-grounded member CENTROID (the curated concept), overlap@10 vs a random
      baseline. Shows a free-text phrase reaches the curated concept without any retraining.

  python src/instrument/sbert_open_concepts.py [--min_rel 0.3] [--ridge 1.0]
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "6")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import sys
import json
import time
import argparse
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
sys.path.insert(0, os.path.join(_ROOT, "src", "baselines"))
sys.path.insert(0, _HERE)

from train_tower_t2 import log
import run_battery_phaseA as PA
from run_battery_phaseA import build_real_ctx

OUT = os.path.join(_ROOT, "experiments", "battery", "sbert_open_concepts.json")
DATA = os.path.join(_ROOT, "data", "movielens")

# Probe phrases: deliberately NOT genome tag strings, so success cannot be tag lookup.
PROBES = ["dinosaurs", "movies about grief", "heist gone wrong", "slow burn character study",
          "outer space aliens invading earth", "spy during the cold war",
          "artificial intelligence robots", "courtroom drama", "coming of age in the suburbs",
          "post apocalyptic survival"]
# (phrase, paraphrase) pairs -- the paraphrase must not be a tag string either.
PARAPHRASES = [("dinosaurs", "prehistoric reptiles"),
               ("time travel", "journeys through time"),
               ("horror", "scary frightening movies"),
               ("space", "outer space and distant planets"),
               ("war", "armed conflict and battlefields"),
               ("romance", "falling in love"),
               ("animation", "animated cartoon films"),
               ("crime", "criminals and heists"),
               ("comedy", "funny humorous films"),
               ("documentary", "non fiction real life films")]


def build_item_vectors(sid_of_movie, sb, min_rel):
    """Item representation = RELEVANCE-WEIGHTED MEAN of its genome tags' SBERT embeddings.

    Deliberately NOT a concatenated top-N tag string. Concatenation forces an arbitrary cutoff (the old
    chapter used top-18, which is a choice with nothing behind it) and all-MiniLM-L6-v2 truncates at 256
    word-pieces anyway, so a long string silently loses its tail. Weighting instead uses ALL 1,128 tags
    with the relevance the genome actually assigns them, has no truncation, and costs 1,128 encodes
    rather than 18k. A query phrase stays a single SBERT vector in the same space, so the adapter is
    unaffected.

    min_rel drops near-zero relevances (genome scores are dense: every tag has a score for every film,
    most of them meaningless). Returns {sid: (384,) unit vector}.
    """
    import csv
    from collections import defaultdict
    tagname = {}
    with open(os.path.join(DATA, "genome-tags.csv"), newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            tagname[int(r["tagId"])] = r["tag"].strip()
    tag_ids = sorted(tagname)
    log(f"[sbert] encoding {len(tag_ids)} genome tag names once...")
    T = sb.encode([tagname[t] for t in tag_ids], normalize_embeddings=True,
                  batch_size=256, show_progress_bar=False).astype(np.float64)
    trow = {t: i for i, t in enumerate(tag_ids)}

    acc = defaultdict(lambda: np.zeros(T.shape[1]))
    wsum = defaultdict(float)
    kept = 0
    with open(os.path.join(DATA, "genome-scores.csv"), newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rel = float(r["relevance"])
            if rel < min_rel:
                continue
            sid = sid_of_movie.get(int(r["movieId"]))
            if sid is None:
                continue
            acc[sid] += rel * T[trow[int(r["tagId"])]]
            wsum[sid] += rel
            kept += 1
    out = {}
    for sid, v in acc.items():
        v = v / max(wsum[sid], 1e-9)
        out[sid] = v / max(np.linalg.norm(v), 1e-12)
    log(f"[sbert] {kept:,} tag-item pairs above relevance {min_rel} -> {len(out)} item vectors")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot", default=PA.SNAP_DEFAULT)
    ap.add_argument("--min_rel", type=float, default=0.3,
                    help="drop genome relevances below this (scores are dense; most are meaningless). "
                         "All surviving tags contribute, weighted -- there is no top-N cutoff.")
    ap.add_argument("--ridge", type=float, default=1.0)
    ap.add_argument("--n_scale_tags", type=int, default=150, help="tags for the at-scale control")
    args = ap.parse_args()
    t0 = time.time()

    ctx = build_real_ctx(args.snapshot)
    # attach_belief is what puts the frozen decoder rows (ctx.Wd) and the genome members
    # (ctx.tags / ctx.members) onto the context -- needed for the adapter target AND control (b).
    from run_battery_phaseB import attach_belief
    attach_belief(ctx, None)
    Wd = ctx.Wd.numpy().astype(np.float64)           # frozen decoder rows: item directions (n x d)
    n, d = Wd.shape
    log(f"[sbert] frozen decoder rows {Wd.shape}")

    # internal sid -> movieId, via the split's unique_sid list
    sid2movie = PA.sid_to_movieid(ctx) if hasattr(PA, "sid_to_movieid") else None
    if sid2movie is None:
        import pandas as pd
        usid = pd.read_csv(os.path.join(_ROOT, "data", "ml-25m", "proc", "unique_sid.txt"), header=None)
        sid2movie = {i: int(m) for i, m in enumerate(usid[0].tolist())}
    movie2sid = {m: s for s, m in sid2movie.items()}

    titles = {}
    import csv as _csv
    with open(os.path.join(DATA, "movies.csv"), newline="", encoding="utf-8") as f:
        for r in _csv.DictReader(f):
            s = movie2sid.get(int(r["movieId"]))
            if s is not None:
                titles[s] = r["title"]

    from sentence_transformers import SentenceTransformer
    log("[sbert] loading all-MiniLM-L6-v2 (one-off download if not cached)...")
    sb = SentenceTransformer("all-MiniLM-L6-v2")
    vecs = build_item_vectors(movie2sid, sb, args.min_rel)
    sids = sorted(vecs)
    S = np.stack([vecs[s] for s in sids])
    log(f"[sbert] item vectors {S.shape} ({100*len(sids)/n:.1f}% catalogue coverage)")

    # ---- the one global adapter: W = Q^T S (S^T S + beta I)^-1 ----
    Q = Wd[sids]
    G = S.T @ S + args.ridge * np.eye(S.shape[1])
    W = (Q.T @ S) @ np.linalg.inv(G)                  # (d x 384)
    fit_cos = float(np.mean([np.dot(W @ S[i], Q[i]) /
                             (np.linalg.norm(W @ S[i]) * np.linalg.norm(Q[i]) + 1e-12)
                             for i in range(0, len(sids), 17)]))
    log(f"[sbert] adapter fitted {W.shape}; mean cos(W*SBERT(item), decoder row) = {fit_cos:.3f}")

    def direction(phrase):
        v = sb.encode([phrase], normalize_embeddings=True)[0].astype(np.float64)
        return W @ v

    def topk(vec, k=10):
        sc = Wd @ vec
        return [int(i) for i in np.argsort(-sc)[:k]]

    # ---- (0) the illustration: does an open phrase point at the right films? ----
    probes = {}
    for ph in PROBES:
        idx = topk(direction(ph))
        probes[ph] = [titles.get(i, f"sid{i}") for i in idx]
        log(f"[probe] {ph!r} -> " + "; ".join(probes[ph][:5]))

    # ---- (a) paraphrase invariance ----
    para = {}
    for a, b in PARAPHRASES:
        ta, tb = set(topk(direction(a))), set(topk(direction(b)))
        para[f"{a} | {b}"] = len(ta & tb) / 10.0
    para_mean = float(np.mean(list(para.values())))
    log(f"[control-a] paraphrase overlap@10 mean = {para_mean:.2f}  " +
        " ".join(f"{k.split(' | ')[0]}={v:.1f}" for k, v in list(para.items())[:5]))

    # ---- (b) at scale: text-derived direction vs the tag's genome-grounded member centroid ----
    tags = getattr(ctx, "tags", None)
    members = getattr(ctx, "members", None)
    scale = {}
    if tags is not None and members is not None:
        rng = np.random.default_rng(0)
        names = list(tags)[:args.n_scale_tags]
        ov, rnd = [], []
        for tg in names:
            mem = members.get(tg)
            if mem is None or len(mem) < 30:
                continue
            cent = Wd[np.asarray(mem)].mean(0)
            a = set(topk(direction(str(tg))))
            b = set(topk(cent))
            ov.append(len(a & b) / 10.0)
            rnd.append(len(a & set(rng.choice(n, 10, replace=False))) / 10.0)
        if ov:
            scale = {"n_tags": len(ov), "text_vs_centroid_overlap@10": float(np.mean(ov)),
                     "random_overlap@10": float(np.mean(rnd)),
                     "ratio": float(np.mean(ov) / max(np.mean(rnd), 1e-9))}
            log(f"[control-b] text-vs-centroid overlap@10 = {scale['text_vs_centroid_overlap@10']:.3f} "
                f"vs random {scale['random_overlap@10']:.4f}  ({scale['ratio']:.0f}x) over {len(ov)} tags")
    else:
        log("[control-b] SKIPPED: genome members not attached to ctx")

    res = {"snapshot": os.path.basename(args.snapshot), "n_items_with_text": len(sids),
           "coverage": len(sids) / n, "min_relevance": args.min_rel, "ridge": args.ridge,
           "adapter_fit_cos": fit_cos, "probes": probes,
           "paraphrase_overlap@10": para, "paraphrase_mean": para_mean,
           "at_scale": scale, "seconds": round(time.time() - t0, 1)}
    json.dump(res, open(OUT, "w"), indent=1)
    log(f"[sbert] -> {OUT}")


if __name__ == "__main__":
    main()
