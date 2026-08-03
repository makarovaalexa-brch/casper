# Rules tier for He-corpus mention adjudication + model-input builder for the gold audition.
# Deterministic, span-only (no context). Decides ACCEPT(he_tconst) vs DEFER per mention.
# Thresholds fixed a priori (2026-08-03, before scoring): see DECIDE().
import csv, json, re, sys
from pathlib import Path

BASE = Path(r"C:\dev\phd\casper\data\reddit_he")
BASICS = BASE / "IMDB-database" / "title.basics.tsv"
RATINGS = BASE / "IMDB-database" / "title.ratings.tsv"
GOLD = BASE / "gold_combined_500.csv"
OUT_DEC = BASE / "audition" / "rules_decisions.csv"
OUT_MODEL_INPUT = BASE / "audition" / "model_input.jsonl"

STOP = {"netflix", "hulu", "mubi", "amazon", "prime", "hbo", "disney", "imdb", "reddit",
        "youtube", "spotify", "tv", "ok", "yes", "no", "lol", "thanks", "thank you"}

def norm(s):
    s = s.lower().strip()
    s = re.sub(r"\((19|20)\d\d\)", "", s)
    s = re.sub(r"[\"'\u2018\u2019\u201c\u201d\*\[\]\(\)]", "", s)
    s = s.strip(" .,!?:;-\u2013\u2014/\\")
    return re.sub(r"\s+", " ", s)

def span_year(s):
    m = re.search(r"\((19|20)(\d\d)\)", s)
    return int(m.group(1) + m.group(2)) if m else None

rows = list(csv.DictReader(open(GOLD, encoding="utf-8")))
queries = {}
for r in rows:
    n = norm(r["mention_span"])
    qs = {n}
    if n.startswith("the "):
        qs.add(n[4:])
    else:
        qs.add("the " + n)
    queries[n] = qs
allq = set().union(*queries.values()) if queries else set()

votes = {}
with open(RATINGS, encoding="utf-8") as f:
    next(f)
    for line in f:
        p = line.rstrip("\n").split("\t")
        votes[p[0]] = int(p[2])

TYPES = {"movie", "tvSeries", "tvMiniSeries", "tvMovie", "video", "short"}
cand = {q: [] for q in allq}
with open(BASICS, encoding="utf-8") as f:
    next(f)
    for line in f:
        p = line.rstrip("\n").split("\t")
        if p[1] not in TYPES:
            continue
        np_, no_ = norm(p[2]), norm(p[3])
        hit = np_ if np_ in cand else (no_ if no_ in cand else None)
        if hit is not None:
            cand[hit].append([p[0], p[1], p[2], p[5], votes.get(p[0], 0)])

def DECIDE(r):
    n = norm(r["mention_span"])
    cs = []
    for q in queries[n]:
        cs.extend(cand.get(q, []))
    cs = sorted({c[0]: c for c in cs}.values(), key=lambda x: -x[4])
    sy = span_year(r["mention_span"])
    if sy is not None:
        hits = [c for c in cs if c[3] not in ("\\N", "") and abs(int(c[3]) - sy) <= 1]
        if len(hits) == 1 and r["he_tconst"] == hits[0][0]:
            return "ACCEPT", cs
        return "DEFER", cs
    if len(n) < 4 or n in STOP or not cs:
        return "DEFER", cs
    top = cs[0]
    second_votes = cs[1][4] if len(cs) > 1 else 0
    if top[4] >= 2000 and top[4] >= 10 * max(second_votes, 1) and r["he_tconst"] == top[0]:
        return "ACCEPT", cs
    return "DEFER", cs

OUT_DEC.parent.mkdir(exist_ok=True)
n_acc = acc_ok = 0
with open(OUT_DEC, "w", newline="", encoding="utf-8") as fdec, open(OUT_MODEL_INPUT, "w", encoding="utf-8") as fmi:
    w = csv.writer(fdec)
    w.writerow(["idx", "decision", "rules_tconst", "gold", "rules_correct"])
    for i, r in enumerate(rows):
        dec, cs = DECIDE(r)
        gold = r["label_tconst_or_NONE"]
        out_t = r["he_tconst"] if dec == "ACCEPT" else ""
        correct = (dec == "ACCEPT" and out_t == gold)
        if dec == "ACCEPT":
            n_acc += 1
            acc_ok += int(correct)
        w.writerow([i, dec, out_t, gold, int(correct) if dec == "ACCEPT" else ""])
        fmi.write(json.dumps({
            "id": i, "year": r["year"], "context": r["context"],
            "span": r["mention_span"], "sentence": r["sentence"],
            "candidates": [{"tconst": c[0], "type": c[1], "title": c[2], "year": c[3], "votes": c[4]}
                           for c in cs[:8]],
        }, ensure_ascii=False) + "\n")

N = len(rows)
print(f"rules tier on gold {N}: ACCEPT {n_acc} ({n_acc/N:.1%} coverage), precision on accepted {acc_ok}/{n_acc} = {acc_ok/max(n_acc,1):.3f}")
print(f"DEFER {N-n_acc} -> model tier")
