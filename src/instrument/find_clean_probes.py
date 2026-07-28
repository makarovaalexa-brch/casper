r"""find_clean_probes.py -- find free-text probes that are genuinely OUTSIDE the concept vocabulary.

A probe is only interesting if it is not the vocabulary in disguise. Two ways it can fail:
  LEXICAL  -- a genome tag appears verbatim inside the phrase ("slow burn CHARACTER STUDY",
              "spy during the COLD WAR", "COMING OF AGE in the suburbs"). Cosine can look moderate
              while the phrase is literally the tag plus modifiers.
  SEMANTIC -- no shared words, but some tag sits very close in embedding space.

This screens a candidate list on BOTH, then prints what survives with its nearest concept and the films
it returns, so the examples in the chapter can be chosen on evidence rather than on how good they sound.

  python src/instrument/find_clean_probes.py [--max_cos 0.60]
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "6")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import sys
import re
import json
import argparse
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
sys.path.insert(0, os.path.join(_ROOT, "src", "baselines"))
sys.path.insert(0, _HERE)

from train_tower_t2 import log
import run_battery_phaseA as PA
from run_battery_phaseA import build_real_ctx

OUT = os.path.join(_ROOT, "experiments", "battery", "clean_probes.json")
DATA = os.path.join(_ROOT, "data", "movielens")

# Candidates deliberately written as descriptions rather than labels, so they are unlikely to be tags.
CANDIDATES = [
    "movies about grief",
    "a family falling apart over one dinner",
    "someone slowly losing their memory",
    "quiet films where almost nothing happens",
    "a con artist who gets conned",
    "small town secrets that come out",
    "a road trip that goes badly wrong",
    "when the monster is never shown on screen",
    "two people talking for the whole film",
    "an ordinary person pushed too far",
    "loneliness in a big city",
    "a detective who drinks too much",
    "the last day before everything changes",
    "growing old and looking back on a life",
    "somebody pretending to be someone else",
    "a wedding where everything goes wrong",
    "set entirely inside one room",
    "a mother and daughter who cannot talk to each other",
    "revenge served very slowly",
    "people trapped by a decision they made years ago",
    "the quiet dignity of losing",
    "a friendship that curdles",
    "bureaucracy grinding somebody down",
    "nostalgia for a place that no longer exists",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max_cos", type=float, default=0.60)
    ap.add_argument("--ridge", type=float, default=1.0)
    args = ap.parse_args()

    ctx = build_real_ctx(PA.SNAP_DEFAULT)
    from run_battery_phaseB import attach_belief
    attach_belief(ctx, None)
    Wd = ctx.Wd.numpy().astype(np.float64)
    tags = list(ctx.tags)
    D = ctx.d_c.numpy().astype(np.float64)

    import csv
    tagname = {}
    with open(os.path.join(DATA, "genome-tags.csv"), newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            tagname[int(r["tagId"])] = r["tag"].strip()
    names = [tagname.get(int(t), str(t)) if str(t).isdigit() else str(t) for t in tags]
    all_tag_strings = sorted({v.lower() for v in tagname.values()})

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
    W = (D.T @ S) @ np.linalg.inv(S.T @ S + args.ridge * np.eye(S.shape[1]))
    P = sb.encode(CANDIDATES, normalize_embeddings=True, show_progress_bar=False).astype(np.float64)

    def lexical_hit(phrase):
        """Any genome tag occurring as a whole-word substring of the phrase."""
        low = " " + re.sub(r"[^a-z0-9 ]", " ", phrase.lower()) + " "
        hits = []
        for t in all_tag_strings:
            if len(t) < 4:
                continue
            if " " + t + " " in low:
                hits.append(t)
        return hits

    def topk(v, k=10):
        return [int(i) for i in np.argsort(-(Wd @ v))[:k]]

    out = {}
    for i, ph in enumerate(CANDIDATES):
        lex = lexical_hit(ph)
        sims = S @ P[i]
        j = int(np.argmax(sims))
        near, cosv = names[j], float(sims[j])
        films = [titles.get(x, str(x)) for x in topk(W @ P[i])[:5]]
        clean = (not lex) and cosv <= args.max_cos
        out[ph] = {"lexical_tag_hits": lex, "nearest": near, "cos": round(cosv, 3),
                   "clean": clean, "top5": films}
        mark = "CLEAN " if clean else ("LEXICAL" if lex else "close  ")
        log(f"[{mark}] {ph!r}")
        log(f"          nearest={near!r} cos={cosv:.2f}" +
            (f"  LEXICAL HITS: {lex}" if lex else ""))
        log(f"          {'; '.join(films[:4])}")

    json.dump({"max_cos": args.max_cos, "candidates": out}, open(OUT, "w"), indent=1)
    n_clean = sum(1 for v in out.values() if v["clean"])
    log(f"[clean-probes] {n_clean}/{len(CANDIDATES)} clean -> {OUT}")


if __name__ == "__main__":
    main()
