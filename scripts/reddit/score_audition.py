# Score the linker audition: rules tier, each model, and rules+model pipelines vs the 500 gold labels.
import csv, json, re
from pathlib import Path

BASE = Path(r"C:\dev\phd\casper\data\reddit_he")
SCRATCH = Path(r"C:\Users\Tech\AppData\Local\Temp\claude\C--dev-phd\3de8972f-dd56-4133-8d38-0a641f8090c3\scratchpad\audition")
GOLD = BASE / "gold_combined_500.csv"
DEC = BASE / "audition" / "rules_decisions.csv"
BASICS = BASE / "IMDB-database" / "title.basics.tsv"
MODELS = ["haiku", "sonnet", "opus"]

gold_rows = list(csv.DictReader(open(GOLD, encoding="utf-8")))
gold = [r["label_tconst_or_NONE"] for r in gold_rows]
he = [r["he_tconst"] for r in gold_rows]
uncertain = [("UNCERTAIN" in r["note"].upper() or "AMBIG" in r["note"].upper()) for r in gold_rows]
N = len(gold)

dec = {int(r["idx"]): r for r in csv.DictReader(open(DEC, encoding="utf-8"))}

# TITLE: resolution needs (title, year) -> tconst
title_preds = set()
raw = {}
for m in MODELS:
    preds = {}
    for b in range(5):
        p = SCRATCH / f"{m}_b{b}.csv"
        if not p.exists():
            continue
        for r in csv.DictReader(open(p, encoding="utf-8-sig")):
            preds[int(r["id"])] = r["pred"].strip()
    raw[m] = preds
    for v in preds.values():
        if v.upper().startswith("TITLE:"):
            title_preds.add(v)

def parse_title(v):
    body = v[6:].strip()
    m = re.match(r"^(.*?)\s*\((\d{4})\)\s*$", body)
    return (m.group(1).strip().lower(), int(m.group(2))) if m else (body.lower(), None)

want = {parse_title(v): v for v in title_preds}
resolve = {}
if want:
    keys = {t for (t, y) in want}
    with open(BASICS, encoding="utf-8") as f:
        next(f)
        for line in f:
            p = line.rstrip("\n").split("\t")
            if p[1] not in ("movie", "tvSeries", "tvMiniSeries", "tvMovie", "video", "short"):
                continue
            tl, ol = p[2].lower(), p[3].lower()
            if tl in keys or ol in keys:
                yr = None if p[5] in ("\\N", "") else int(p[5])
                for (t, y), orig in want.items():
                    if (tl == t or ol == t) and (y is None or (yr is not None and abs(yr - y) <= 1)):
                        # keep the most-voted resolution later; store candidates
                        resolve.setdefault(orig, []).append((p[0], yr))
# ratings for tie-break
if resolve:
    votes = {}
    with open(BASE / "IMDB-database" / "title.ratings.tsv", encoding="utf-8") as f:
        next(f)
        for line in f:
            p = line.rstrip("\n").split("\t")
            votes[p[0]] = int(p[2])
    resolve = {k: max(v, key=lambda x: votes.get(x[0], 0))[0] for k, v in resolve.items()}

def final_pred(v):
    if v.upper() == "NONE":
        return "NONE"
    if v.upper().startswith("TITLE:"):
        return resolve.get(v, "UNRESOLVED_TITLE")
    return v

def acc(pred_vec, mask=None):
    idx = [i for i in range(N) if (mask is None or mask[i])]
    ok = sum(1 for i in idx if pred_vec[i] == gold[i])
    return ok, len(idx)

print(f"gold N={N} (uncertain/ambiguous flagged: {sum(uncertain)})")
he_ok, _ = acc(he)
print(f"\nHE (their release):            {he_ok}/{N} = {he_ok/N:.3f}")

racc = [i for i in range(N) if dec[i]["decision"] == "ACCEPT"]
rdef = [i for i in range(N) if dec[i]["decision"] == "DEFER"]
r_ok = sum(1 for i in racc if he[i] == gold[i])
print(f"RULES tier: coverage {len(racc)}/{N} = {len(racc)/N:.1%}, precision on accepted {r_ok}/{len(racc)} = {r_ok/len(racc):.3f}")

for m in MODELS:
    preds = raw.get(m, {})
    if len(preds) < N:
        print(f"\n{m.upper()}: INCOMPLETE ({len(preds)}/{N} preds) — skipping")
        continue
    pv = [final_pred(preds[i]) for i in range(N)]
    m_ok, _ = acc(pv)
    m_ok_c, n_c = acc(pv, [not u for u in uncertain])
    unres = sum(1 for v in pv if v == "UNRESOLVED_TITLE")
    # pipeline: rules ACCEPT -> he tconst; DEFER -> model
    pipe = [he[i] if dec[i]["decision"] == "ACCEPT" else pv[i] for i in range(N)]
    p_ok, _ = acc(pipe)
    d_ok = sum(1 for i in rdef if pv[i] == gold[i])
    print(f"\n{m.upper()}: solo {m_ok}/{N} = {m_ok/N:.3f}  |  excl-uncertain {m_ok_c}/{n_c} = {m_ok_c/n_c:.3f}  |  unresolved TITLE: {unres}")
    print(f"   on DEFER subset ({len(rdef)}): {d_ok}/{len(rdef)} = {d_ok/len(rdef):.3f}")
    print(f"   PIPELINE rules+{m}: {p_ok}/{N} = {p_ok/N:.3f}")

# disagreement dump for error analysis (best model vs gold)
out = SCRATCH / "audition_errors.txt"
with open(out, "w", encoding="utf-8") as f:
    for m in MODELS:
        preds = raw.get(m, {})
        if len(preds) < N:
            continue
        pv = [final_pred(preds[i]) for i in range(N)]
        f.write(f"== {m} errors ==\n")
        for i in range(N):
            if pv[i] != gold[i]:
                g = gold_rows[i]
                f.write(f"#{i} [{g['year']}|{g['context']}] span<<{g['mention_span']}>> gold={gold[i]} pred={pv[i]} raw={preds[i][:60]} note={g['note'][:80]}\n")
print(f"\nerror dump -> {out}")
