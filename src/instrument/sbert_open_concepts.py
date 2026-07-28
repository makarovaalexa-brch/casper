r"""sbert_open_concepts.py -- R3 illustration: the instrument takes an ARBITRARY CONTINUOUS DIRECTION.

CLAIM BEING ILLUSTRATED (author, 2026-07-28): NOT that we contribute a way to map verbal information into
a collaborative latent -- that ground is taken (Balog et al. SIGIR 2021 map an attribute phrase by BM25
retrieval + item-embedding centroid; Gopfert 2022 / Biyik 2023 fit a per-concept supervised probe). The
claim is narrower and is about the INSTRUMENT: any entity with an embedding is a legal observation, so the
channel set is open WITHOUT BOLT-ONS and a continuous elicitation policy can drive it. Text is the
illustration, not the point.

METHOD. We already HAVE concept directions in the latent: d_c, the whitened member centroid of each genome
tag, which is what the concept channel folds. And every one of them has a name, which is text. So the
adapter is fitted directly on that correspondence -- one global closed-form ridge map

    W = D^T S (S^T S + beta I)^-1     S = SBERT(tag name) (m x 384),  D = d_c (m x 200)

after which ANY phrase gets a direction W @ SBERT(phrase) in the same space. No per-concept training, no
concept inventory, no retraining -- a matrix multiply at inference.

(Earlier drafts fitted on item text built from top-N genome tags. Both the top-N cutoff and the string
concatenation were arbitrary -- and unnecessary, since the concept directions we actually want to
reproduce already exist. Fitting on them directly is simpler and tests the right thing.)

THE TEST THAT MATTERS -- HELD-OUT CONCEPTS. The adapter is fitted on a random 80% of tags and evaluated on
the 20% it never saw. For a held-out tag we compare the direction predicted from its NAME ALONE against
its true genome-grounded direction d_c: cosine, and top-10 item overlap. This is a generalisation test,
not a similarity check: a concept the adapter was never fitted on must still land in the right place.

FURTHER CONTROLS:
  (a) PARAPHRASE INVARIANCE -- a phrase vs a non-tag paraphrase of it, top-10 overlap. Defeats the "it is
      just matching the tag string" reading.
  (b) FREE-TEXT PROBES -- phrases that are not genome tags at all, so they have no d_c to fall back on;
      qualitative, and the illustration the reader remembers.

  python src/instrument/sbert_open_concepts.py [--ridge 1.0] [--holdout 0.2]
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

OUT_T = os.path.join(_ROOT, "experiments", "battery", "sbert_open_concepts_fit%s.json")
DATA = os.path.join(_ROOT, "data", "movielens")
SEED = 4242

# Free-text probes: deliberately NOT genome tag strings, so success cannot be tag lookup.
PROBES = ["dinosaurs", "movies about grief", "heist gone wrong", "slow burn character study",
          "outer space aliens invading earth", "spy during the cold war",
          "artificial intelligence robots", "courtroom drama", "coming of age in the suburbs",
          "post apocalyptic survival"]
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
    """Item vector = RELEVANCE-WEIGHTED MEAN of its genome tags' SBERT embeddings.

    Not a concatenated top-N tag string: a top-N cutoff is arbitrary and MiniLM truncates at 256
    word-pieces, so a long string silently loses its tail. Every tag above a relevance floor contributes
    with the weight the genome assigns it. Costs 1,128 encodes (done once, shared with the tag names)
    rather than 18k. A query phrase stays a single SBERT vector in the same space.
    """
    import csv
    from collections import defaultdict
    tagname = {}
    with open(os.path.join(DATA, "genome-tags.csv"), newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            tagname[int(r["tagId"])] = r["tag"].strip()
    tag_ids = sorted(tagname)
    T = sb.encode([tagname[t] for t in tag_ids], normalize_embeddings=True,
                  batch_size=256, show_progress_bar=False).astype(np.float64)
    trow = {t: i for i, t in enumerate(tag_ids)}
    acc, wsum, kept = defaultdict(lambda: np.zeros(T.shape[1])), defaultdict(float), 0
    log(f"[sbert] streaming genome-scores.csv (relevance >= {min_rel})...")
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
    log(f"[sbert] {kept:,} tag-item pairs kept -> {len(out)} item vectors")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot", default=PA.SNAP_DEFAULT)
    ap.add_argument("--ridge", type=float, default=1.0)
    ap.add_argument("--holdout", type=float, default=0.2,
                    help="concepts-fit only: fraction of tags never fitted on")
    ap.add_argument("--fit_on", default="items", choices=["items", "concepts"],
                    help="items: fit on ~18k item vectors and evaluate on ALL concepts, which are then "
                         "never in the fitting set (best conditioning AND the cleanest generalisation "
                         "test). concepts: fit on 80%% of tag names, hold out 20%%.")
    ap.add_argument("--min_rel", type=float, default=0.3,
                    help="items-fit only: genome relevance floor (scores are dense; most are noise)")
    args = ap.parse_args()
    t0 = time.time()

    ctx = build_real_ctx(args.snapshot)
    # attach_belief puts the frozen decoder rows (ctx.Wd) and the genome concept directions
    # (ctx.tags, ctx.d_c -- whitened member centroids) onto the context.
    from run_battery_phaseB import attach_belief
    attach_belief(ctx, None)
    Wd = ctx.Wd.numpy().astype(np.float64)
    n, d = Wd.shape
    tags = list(ctx.tags)
    D = ctx.d_c.numpy().astype(np.float64) if hasattr(ctx.d_c, "numpy") else np.asarray(ctx.d_c, np.float64)
    log(f"[sbert] {len(tags)} concept directions {D.shape} over a {Wd.shape} decoder")

    # tag id -> human-readable name
    import csv
    tagname = {}
    with open(os.path.join(DATA, "genome-tags.csv"), newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            tagname[int(r["tagId"])] = r["tag"].strip()
    names = [tagname.get(int(t), str(t)) if str(t).isdigit() else str(t) for t in tags]

    # movieId -> internal sid. Required by the items fit, so it is NOT inside the cosmetic try below.
    import pandas as pd
    usid = pd.read_csv(os.path.join(_ROOT, "data", "ml-25m", "proc", "unique_sid.txt"), header=None)
    movie2sid = {int(m): i for i, m in enumerate(usid[0].tolist())}

    titles = {}
    try:
        with open(os.path.join(DATA, "movies.csv"), newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                s = movie2sid.get(int(r["movieId"]))
                if s is not None:
                    titles[s] = r["title"]
    except Exception as e:                                        # titles are cosmetic only
        log(f"[sbert] (titles unavailable: {e})")

    from sentence_transformers import SentenceTransformer
    log("[sbert] loading all-MiniLM-L6-v2 (one-off download if not cached)...")
    sb = SentenceTransformer("all-MiniLM-L6-v2")
    S = sb.encode(names, normalize_embeddings=True, batch_size=256,
                  show_progress_bar=False).astype(np.float64)
    log(f"[sbert] encoded {S.shape[0]} tag names {S.shape}")

    rng = np.random.default_rng(SEED)

    def ridge_fit(X, Y):
        """X (m x 384) -> Y (m x d).  W = Y^T X (X^T X + beta I)^-1."""
        G = X.T @ X + args.ridge * np.eye(X.shape[1])
        return (Y.T @ X) @ np.linalg.inv(G)

    if args.fit_on == "items":
        # Fit on items; EVERY concept is then outside the fitting set, so the concept evaluation below
        # is a pure generalisation test with ~18x the pairs of a concept-only fit.
        ivec = build_item_vectors(movie2sid, sb, args.min_rel)
        isids = sorted(ivec)
        Si = np.stack([ivec[i] for i in isids])
        W = ridge_fit(Si, Wd[isids])
        ho = np.arange(len(tags))                                  # all concepts are held out
        in_ref_idx = None
        log(f"[sbert] adapter fitted on {len(isids)} ITEM vectors; all {len(tags)} concepts are unseen")
    else:
        perm = rng.permutation(len(tags))
        n_ho = max(1, int(round(args.holdout * len(tags))))
        ho, fit = perm[:n_ho], perm[n_ho:]
        W = ridge_fit(S[fit], D[fit])
        in_ref_idx = fit[:len(ho)]
        log(f"[sbert] adapter fitted on {len(fit)} TAG NAMES, holding out {len(ho)}")

    def cos(a, b):
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))

    def topk(vec, k=10):
        return [int(i) for i in np.argsort(-(Wd @ vec))[:k]]

    # ---- THE TEST: held-out concepts, predicted from the NAME ALONE ----
    ho_cos, ho_ov, ho_rand = [], [], []
    per_tag = {}
    for j in ho:
        pred = W @ S[j]
        true = D[j]
        c = cos(pred, true)
        a, b = set(topk(pred)), set(topk(true))
        o = len(a & b) / 10.0
        r = len(a & set(rng.choice(n, 10, replace=False))) / 10.0
        ho_cos.append(c); ho_ov.append(o); ho_rand.append(r)
        per_tag[names[j]] = {"cos": round(c, 3), "overlap@10": o}
    held = {"n": len(ho), "cos_mean": float(np.mean(ho_cos)),
            "overlap@10_mean": float(np.mean(ho_ov)),
            "overlap@10_random": float(np.mean(ho_rand)),
            "ratio_vs_random": float(np.mean(ho_ov) / max(np.mean(ho_rand), 1e-9))}
    log(f"[HELD-OUT] {held['n']} unseen tags: cos {held['cos_mean']:.3f} | "
        f"top-10 overlap {held['overlap@10_mean']:.3f} vs random {held['overlap@10_random']:.4f} "
        f"({held['ratio_vs_random']:.0f}x)")

    # in-fit reference: how much of the fit quality is memorisation? (concepts-fit only -- under an
    # items fit there is no in-fit concept to compare against, which is the point)
    in_cos = None
    if in_ref_idx is not None:
        in_cos = float(np.mean([cos(W @ S[j], D[j]) for j in in_ref_idx]))
        log(f"[in-fit ref] cos {in_cos:.3f} (vs held-out {held['cos_mean']:.3f})")

    # probes use the same adapter under an items fit (nothing to refit); under a concepts fit we refit
    # on all tags so the qualitative examples are not handicapped by the split.
    W_all = W if args.fit_on == "items" else ridge_fit(S, D)

    def direction(p):
        return W_all @ sb.encode([p], normalize_embeddings=True)[0].astype(np.float64)

    probes = {}
    for ph in PROBES:
        idx = topk(direction(ph))
        probes[ph] = [titles.get(i, f"sid{i}") for i in idx]
        log(f"[probe] {ph!r} -> " + "; ".join(probes[ph][:5]))

    para = {}
    for a, b in PARAPHRASES:
        para[f"{a} | {b}"] = len(set(topk(direction(a))) & set(topk(direction(b)))) / 10.0
    para_mean = float(np.mean(list(para.values())))
    log(f"[control-a] paraphrase overlap@10 mean = {para_mean:.2f}")

    res = {"snapshot": os.path.basename(args.snapshot), "n_concepts": len(tags),
           "ridge": args.ridge, "holdout_frac": args.holdout,
           "fit_on": args.fit_on, "held_out_concepts": held, "in_fit_cos_reference": in_cos,
           "per_held_out_tag": per_tag, "probes": probes,
           "paraphrase_overlap@10": para, "paraphrase_mean": para_mean,
           "seconds": round(time.time() - t0, 1)}
    out = OUT_T % args.fit_on
    json.dump(res, open(out, "w"), indent=1)
    log(f"[sbert] -> {out}")


if __name__ == "__main__":
    main()
