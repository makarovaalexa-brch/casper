"""Assemble the He-corpus evaluation ruler from the completed adjudication (NO LLM, NO API).

Implements the ruler definition of `docs/design/DS_REDDIT_LINKING.md` (step 6 + "PIPELINE OF
RECORD") over exactly the threads covered by adjudication batches 0000-0099:

  links   = rules-tier ACCEPTs (`adjudication/accepted.csv.gz`)
            + model adjudications (`adjudication/preds/batch_00xx_preds.csv` joined back to
              `adjudication/batches/batch_00xx.jsonl` by `id`), NONE dropped, TITLE:<name> (<year>)
              resolved to a tconst by an alias-tolerant resolver over title.basics.tsv.
  catalogue = tconst -> movieId via data/movielens/links.csv, restricted to the 18,359-item
            project catalogue (census_he.load_catalogue -- the exact same lookup the census uses).
  OP_NAMED = linked items in context request / seeker_followup.
  GT       = linked items in context reply, MINUS OP_NAMED (Eberhard protocol).
  ruler    = threads with >= 5 GT items; request text = the OP's turn_order==0 raw utterance.

HARD RULE 1: nothing is sampled, capped or truncated.  Every mention of every completed batch is
carried through; the only reductions are the pre-registered ones (NONE, unresolvable TITLE,
out-of-catalogue, <5 GT), each counted in RULER_STATS.json.

Privacy: no usernames anywhere -- request text is passed through the same `redact()` used when the
batches were built (`/u/handle` -> `u/[USER]`); no author field is ever read.

Usage (from C:/dev/phd/casper):
    python scripts/reddit/assemble_ruler.py
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import random
import re
import statistics
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from census_he import (  # noqa: E402
    iter_v1_rows,
    load_catalogue,
    parse_utt,
    thread_of,
)
from prep_adjudication_batches import redact  # noqa: E402  (same handle-redaction, verbatim)
from tier_split_census import TYPES  # noqa: E402  (same title-type filter as the candidate index)

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "data" / "reddit_he"
V1_DIR = BASE / "large_v1"
ADJ = BASE / "adjudication"
BATCH_DIR = ADJ / "batches"
PRED_DIR = ADJ / "preds"
ACCEPTED = ADJ / "accepted.csv.gz"
MANIFEST = ADJ / "manifest.json"
BASICS = BASE / "IMDB-database" / "title.basics.tsv"
RATINGS = BASE / "IMDB-database" / "title.ratings.tsv"

OUT_DIR = BASE / "ruler"
OUT_CSV = OUT_DIR / "ruler_v1.csv.gz"
OUT_STATS = OUT_DIR / "RULER_STATS.json"
OUT_MD = ROOT / "docs" / "results" / "REDDIT_RULER_V1.md"

MIN_GT = 5
SEED = 98765
SEEKER_CTX = {"request", "seeker_followup"}

csv.field_size_limit(1 << 31 - 1 if sys.maxsize > 2**32 else (1 << 30))


# ------------------------------------------------------------------ alias-tolerant resolver


def nnorm(s: str) -> str:
    """Normalize a title for alias-tolerant matching.

    case / diacritics / punctuation / hyphen insensitive; `&` == `and`; apostrophes dropped so
    "dont" == "don't"; whitespace collapsed.
    """
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = s.lower().replace("&", " and ")
    s = re.sub(r"[\u2018\u2019\u02bc']", "", s)
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def title_variants(n: str) -> set[str]:
    """The normalized query plus its article variants (IMDb stores "Thing, The" forms too)."""
    qs = {n}
    if n.startswith("the "):
        qs.add(n[4:])
    else:
        qs.add("the " + n)
    if n.startswith("a "):
        qs.add(n[2:])
    for suffix in (" the", " a", " an"):
        if n.endswith(suffix):
            qs.add(n[: -len(suffix)].strip())
    return {q for q in qs if q}


TITLE_RE = re.compile(r"^TITLE:\s*(?P<name>.*?)\s*(?:\((?P<year>(?:19|20)\d\d)\))?\s*$")


def parse_title_pred(pred: str):
    m = TITLE_RE.match(pred)
    if not m or not m.group("name"):
        return None, None
    return m.group("name"), (int(m.group("year")) if m.group("year") else None)


def build_title_index(queries: set[str]) -> dict[str, list]:
    """One streaming pass over title.basics.tsv keyed by the normalized query set."""
    votes: dict[str, int] = {}
    with RATINGS.open(encoding="utf-8") as f:
        next(f)
        for line in f:
            p = line.rstrip("\n").split("\t")
            try:
                votes[p[0]] = int(p[2])
            except (IndexError, ValueError):
                continue
    print(f"ratings: {len(votes):,} titles", flush=True)

    allq: set[str] = set()
    for q in queries:
        allq |= title_variants(q)
    idx: dict[str, list] = {q: [] for q in allq}
    print(f"title queries: {len(allq):,} (from {len(queries):,} distinct predicted titles)",
          flush=True)

    seen = 0
    with BASICS.open(encoding="utf-8") as f:
        next(f)
        for line in f:
            seen += 1
            p = line.rstrip("\n").split("\t")
            if len(p) < 6 or p[1] not in TYPES:
                continue
            keys = {nnorm(p[2]), nnorm(p[3])}
            for k in keys:
                if k in idx:
                    idx[k].append((p[0], p[5], votes.get(p[0], 0)))
    nonempty = sum(1 for v in idx.values() if v)
    print(f"basics: {seen:,} rows scanned, {nonempty:,}/{len(idx):,} queries matched", flush=True)
    return idx


def resolve_title(name: str, year, idx: dict[str, list]) -> tuple[str | None, str]:
    """-> (tconst or None, reason).  Year within +-1 when given; most-voted breaks ties."""
    n = nnorm(name)
    if not n:
        return None, "empty_title"
    cands: dict[str, tuple] = {}
    for q in title_variants(n):
        for c in idx.get(q, ()):
            cands[c[0]] = c
    cs = sorted(cands.values(), key=lambda x: -x[2])
    if not cs:
        return None, "no_title_match"
    if year is not None:
        hits = [c for c in cs if c[1] not in ("\\N", "") and abs(int(c[1]) - year) <= 1]
        if not hits:
            return None, "year_mismatch"
        return hits[0][0], "resolved_with_year"
    return cs[0][0], "resolved_no_year"


# ------------------------------------------------------------------ inputs


def load_batches(n_batches: int):
    """-> (rows, thread_ids).  rows: list of dicts with id/thread_id/turn_id/context/span."""
    rows = []
    threads: set[str] = set()
    for b in range(n_batches):
        path = BATCH_DIR / f"batch_{b:04d}.jsonl"
        with path.open(encoding="utf-8") as f:
            for line in f:
                o = json.loads(line)
                rows.append({"id": o["id"], "thread_id": o["thread_id"], "turn_id": o["turn_id"],
                             "context": o["context"], "span": o["span"]})
                threads.add(o["thread_id"])
    return rows, threads


def load_preds(n_batches: int) -> dict[int, str]:
    preds: dict[int, str] = {}
    for b in range(n_batches):
        path = PRED_DIR / f"batch_{b:04d}_preds.csv"
        with path.open(encoding="utf-8-sig", newline="") as f:
            for row in csv.reader(f):
                if not row or row[0].strip() in ("id", ""):
                    continue
                pid = int(row[0])
                assert pid not in preds, f"duplicate prediction id {pid} in {path}"
                preds[pid] = (row[1] if len(row) > 1 else "").strip()
    return preds


def load_accepted(threads: set[str]):
    """Rules-tier ACCEPT rows restricted to the ruler's threads."""
    out, total = [], 0
    with gzip.open(ACCEPTED, "rt", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            total += 1
            if row["thread_id"] in threads:
                out.append((row["thread_id"], row["context"], row["span"], row["tconst"]))
    return out, total


def load_request_text(threads: set[str]) -> dict[str, str]:
    """Stream the V1 CSVs once; keep the OP's turn_order==0 raw utterance per wanted thread."""
    texts: dict[str, str] = {}
    n_rows = 0
    for split in ("train", "valid", "test"):
        path = V1_DIR / f"{split}.csv"
        if not path.exists():
            print(f"MISSING {path}", flush=True)
            continue
        for row in iter_v1_rows(path):
            n_rows += 1
            if n_rows % 400000 == 0:
                print(f"  {split}: {n_rows:,} rows, {len(texts):,}/{len(threads):,} requests",
                      flush=True)
            tid = thread_of(row["conv_id"])
            if tid not in threads or tid in texts:
                continue
            if str(row.get("is_seeker", "")).strip().lower() != "true":
                continue
            if int(row.get("turn_order") or 0) != 0:
                continue
            texts[tid] = redact(parse_utt(row.get("raw") or "").strip())
        print(f"{split}: done ({n_rows:,} rows cumulative)", flush=True)
    return texts


# ------------------------------------------------------------------ main


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-batches", type=int, default=100)
    args = ap.parse_args()
    nb = args.n_batches

    man = json.loads(MANIFEST.read_text(encoding="utf-8"))
    man_b = man["batches"][:nb]
    man_mentions = sum(b["n_mentions"] for b in man_b)
    man_threads = sum(b["n_threads"] for b in man_b)

    print(f"--- loading {nb} adjudicated batches ---", flush=True)
    brows, threads = load_batches(nb)
    preds = load_preds(nb)
    assert len(brows) == man_mentions, f"batch rows {len(brows)} != manifest {man_mentions}"
    assert len(threads) == man_threads, f"threads {len(threads)} != manifest {man_threads}"
    assert len(preds) == len(brows), f"pred rows {len(preds)} != batch rows {len(brows)}"
    assert set(preds) == {r["id"] for r in brows}, "prediction ids do not match batch ids"
    print(f"mentions {len(brows):,}  threads {len(threads):,}  (manifest: "
          f"{man_mentions:,} / {man_threads:,})", flush=True)

    # --- classify predictions
    st = Counter()
    title_queries: set[str] = set()
    for r in brows:
        p = preds[r["id"]]
        if p == "NONE" or not p:
            st["pred_none"] += 1
        elif re.fullmatch(r"tt\d{7,8}", p):
            st["pred_tconst"] += 1
        elif p.startswith("TITLE:"):
            st["pred_title"] += 1
            name, _y = parse_title_pred(p)
            if name:
                title_queries.add(nnorm(name))
        else:
            st["pred_malformed"] += 1
    print(f"predictions: tconst {st['pred_tconst']:,}  TITLE {st['pred_title']:,}  "
          f"NONE {st['pred_none']:,}  malformed {st['pred_malformed']:,}", flush=True)

    print("--- building alias-tolerant title index ---", flush=True)
    idx = build_title_index(title_queries)

    in_cat, _cat = load_catalogue()
    print(f"catalogue lookup: {len(in_cat):,} tconst keys -> {len(_cat):,} movieIds", flush=True)

    # --- model links
    links = []          # (thread_id, context, span, tconst, source)
    res_reason = Counter()
    resolve_cache: dict[tuple, tuple] = {}
    for r in brows:
        p = preds[r["id"]]
        tc = None
        if p == "NONE" or not p:
            continue
        if re.fullmatch(r"tt\d{7,8}", p):
            tc = p
            st["model_tconst_kept"] += 1
        elif p.startswith("TITLE:"):
            name, year = parse_title_pred(p)
            if name is None:
                res_reason["unparsable_title"] += 1
                st["title_unresolved"] += 1
                continue
            key = (nnorm(name), year)
            hit = resolve_cache.get(key)
            if hit is None:
                hit = resolve_title(name, year, idx)
                resolve_cache[key] = hit
            tc, reason = hit
            res_reason[reason] += 1
            if tc is None:
                st["title_unresolved"] += 1
                continue
            st["title_resolved"] += 1
        else:
            st["malformed_dropped"] += 1
            continue
        links.append((r["thread_id"], r["context"], r["span"], tc, "model"))

    # --- rules-tier links
    acc_rows, acc_total = load_accepted(threads)
    st["accepted_rows_total"] = acc_total
    st["accepted_rows_in_ruler_threads"] = len(acc_rows)
    for tid, ctx, span, tc in acc_rows:
        links.append((tid, ctx, span, tc, "rules"))

    # --- catalogue mapping
    per_thread = defaultdict(lambda: {"op": {}, "gt": {}})
    for tid, ctx, span, tc, src in links:
        st[f"links_{src}"] += 1
        mid = in_cat.get(tc)
        if mid is None:
            st[f"out_of_catalogue_{src}"] += 1
            continue
        st[f"in_catalogue_{src}"] += 1
        bucket = "op" if ctx in SEEKER_CTX else "gt"
        per_thread[tid][bucket].setdefault(mid, (ctx, span, tc))

    # --- per-thread GT / OP_NAMED
    kept, gt_sizes, op_sizes, dropped_sizes = [], [], [], []
    for tid in sorted(threads):
        d = per_thread.get(tid, {"op": {}, "gt": {}})
        op = dict(d["op"])
        gt = {m: v for m, v in d["gt"].items() if m not in op}
        if len(gt) >= MIN_GT:
            kept.append((tid, op, gt))
            gt_sizes.append(len(gt))
            op_sizes.append(len(op))
        else:
            dropped_sizes.append(len(gt))

    print(f"threads kept {len(kept):,}/{len(threads):,}", flush=True)

    # --- request text
    print("--- streaming V1 for request text ---", flush=True)
    texts = load_request_text({t for t, _, _ in kept})
    st["threads_with_request_text"] = sum(1 for t, _, _ in kept if texts.get(t))
    st["threads_missing_request_text"] = len(kept) - st["threads_with_request_text"]

    # --- outputs
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with gzip.open(OUT_CSV, "wt", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["thread_id", "n_op_named", "n_gt", "op_named_movieIds", "gt_movieIds",
                    "request_text"])
        for tid, op, gt in kept:
            w.writerow([tid, len(op), len(gt),
                        " ".join(str(m) for m in sorted(op)),
                        " ".join(str(m) for m in sorted(gt)),
                        texts.get(tid, "")])
    print(f"wrote {OUT_CSV} ({len(kept):,} threads)", flush=True)

    def dist(v):
        if not v:
            return {"n": 0}
        s = sorted(v)
        return {"n": len(s), "mean": round(sum(s) / len(s), 3), "median": statistics.median(s),
                "min": s[0], "p10": s[int(0.10 * (len(s) - 1))], "p90": s[int(0.90 * (len(s) - 1))],
                "max": s[-1], "total": sum(s),
                "hist": dict(sorted(Counter(min(x, 50) for x in s).items()))}

    stats = {
        "scope": f"adjudication batches 0000-{nb - 1:04d} (complete), threads never split",
        "min_gt": MIN_GT,
        "manifest": {"mentions": man_mentions, "threads": man_threads},
        "funnel": {
            "adjudicated_mentions_in": len(brows),
            "pred_NONE_dropped": st["pred_none"],
            "pred_malformed_dropped": st["pred_malformed"],
            "pred_tconst": st["pred_tconst"],
            "pred_TITLE": st["pred_title"],
            "title_resolved": st["title_resolved"],
            "title_unresolved_dropped": st["title_unresolved"],
            "title_resolution_rate": round(st["title_resolved"] / max(st["pred_title"], 1), 4),
            "model_links": st["links_model"],
            "rules_links_in_ruler_threads": st["links_rules"],
            "accepted_rows_total_all_threads": st["accepted_rows_total"],
            "links_total": st["links_model"] + st["links_rules"],
            "in_catalogue_model": st["in_catalogue_model"],
            "out_of_catalogue_model": st["out_of_catalogue_model"],
            "in_catalogue_rules": st["in_catalogue_rules"],
            "out_of_catalogue_rules": st["out_of_catalogue_rules"],
            "in_catalogue_rate_model": round(
                st["in_catalogue_model"] / max(st["links_model"], 1), 4),
            "in_catalogue_rate_rules": round(
                st["in_catalogue_rules"] / max(st["links_rules"], 1), 4),
            "threads_in_scope": len(threads),
            "threads_kept_ge5_gt": len(kept),
            "threads_dropped_lt5_gt": len(threads) - len(kept),
            "threads_with_request_text": st["threads_with_request_text"],
            "threads_missing_request_text": st["threads_missing_request_text"],
        },
        "title_resolution_reasons": dict(sorted(res_reason.items())),
        "gt_size_distribution_kept": dist(gt_sizes),
        "op_named_size_distribution_kept": dist(op_sizes),
        "gt_size_distribution_dropped_threads": dist(dropped_sizes),
        "op_named_coverage": {
            "threads_with_any_op_named": sum(1 for s in op_sizes if s > 0),
            "rate": round(sum(1 for s in op_sizes if s > 0) / max(len(op_sizes), 1), 4),
        },
        "outputs": {"ruler": str(OUT_CSV), "stats": str(OUT_STATS), "doc": str(OUT_MD)},
    }
    OUT_STATS.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(f"wrote {OUT_STATS}", flush=True)

    # --- reconciliation
    print("\n--- reconciliation ---")
    print(f"  manifest mentions (b0-{nb-1:04d})   {man_mentions:,}   ours {len(brows):,}")
    print(f"  manifest threads                 {man_threads:,}   ours {len(threads):,}")
    tot = st["pred_none"] + st["pred_tconst"] + st["pred_title"] + st["pred_malformed"]
    print(f"  NONE+tconst+TITLE+malformed      {tot:,}   (== mentions in)")
    print(f"  TITLE resolved+unresolved        "
          f"{st['title_resolved'] + st['title_unresolved']:,}   (== TITLE {st['pred_title']:,})")
    print(f"  model links kept                 {st['links_model']:,} "
          f"(= tconst {st['pred_tconst']:,} + resolved {st['title_resolved']:,})")
    print(f"  accepted.csv.gz rows total       {acc_total:,}; in ruler threads {len(acc_rows):,}")
    assert tot == len(brows)
    assert st["title_resolved"] + st["title_unresolved"] == st["pred_title"]
    assert st["links_model"] == st["pred_tconst"] + st["title_resolved"]
    assert st["links_rules"] == len(acc_rows)
    assert st["in_catalogue_model"] + st["out_of_catalogue_model"] == st["links_model"]
    assert st["in_catalogue_rules"] + st["out_of_catalogue_rules"] == st["links_rules"]
    assert len(kept) + len(dropped_sizes) == len(threads)
    print("ALL RECONCILIATION ASSERTS PASSED")

    # --- spot checks (10 random kept threads, seeded; handles redacted)
    rng = random.Random(SEED)
    spot = rng.sample(kept, min(10, len(kept)))
    spot_rows = []
    for tid, op, gt in spot:
        ex = [f"{redact(span)} -> {tc} -> mid {mid}"
              for mid, (ctx, span, tc) in list(sorted(gt.items()))[:3]]
        req = (texts.get(tid) or "")
        spot_rows.append({"thread_id": tid, "n_gt": len(gt), "n_op": len(op),
                          "gt_examples": ex,
                          "request_excerpt": " ".join(req.split())[:220]})

    write_md(stats, spot_rows, nb)
    print(f"wrote {OUT_MD}")

    print("\n--- headline ---")
    print(f"  threads kept          {len(kept):,} / {len(threads):,}")
    print(f"  GT size  mean {stats['gt_size_distribution_kept']['mean']}  "
          f"median {stats['gt_size_distribution_kept']['median']}  "
          f"max {stats['gt_size_distribution_kept']['max']}")
    print(f"  OP_NAMED mean {stats['op_named_size_distribution_kept']['mean']}  "
          f"coverage {100*stats['op_named_coverage']['rate']:.1f}%")
    print(f"  TITLE resolution      {100*stats['funnel']['title_resolution_rate']:.1f}%")
    return 0


def write_md(stats: dict, spot_rows: list, nb: int) -> None:
    f = stats["funnel"]
    g = stats["gt_size_distribution_kept"]
    o = stats["op_named_size_distribution_kept"]

    def pct(a, b):
        return f"{100*a/max(b,1):.1f}%"

    spot_md = "\n".join(
        f"| `{r['thread_id']}` | {r['n_gt']} | {r['n_op']} | "
        + "<br>".join(f"`{e}`" for e in r["gt_examples"])
        + f" | {r['request_excerpt'].replace('|', '/')} |"
        for r in spot_rows)

    md = f"""# Reddit He-corpus evaluation ruler v1 (2026-08-03)

Assembled by `scripts/reddit/assemble_ruler.py` from the completed adjudication
(batches 0000-{nb-1:04d}) plus the rules-tier ACCEPTs. No LLM call, no Reddit API, no sampling:
every mention of every completed batch is carried through, and each reduction below is one of the
four pre-registered ones (NONE, unresolvable TITLE, out-of-catalogue, <{stats['min_gt']} GT items).

Ruler definition (`docs/design/DS_REDDIT_LINKING.md` step 6): **GT = in-catalogue items mentioned
in REPLY turns, minus anything the OP named** (request / seeker_followup turns = OP_NAMED, the
Eberhard protocol); a thread enters the ruler when it has >= {stats['min_gt']} GT items.

## Funnel

| stage | count | share |
|---|---|---|
| adjudicated mentions in (manifest {stats['manifest']['mentions']:,}) | {f['adjudicated_mentions_in']:,} | 100% |
| - model answered NONE (dropped) | {f['pred_NONE_dropped']:,} | {pct(f['pred_NONE_dropped'], f['adjudicated_mentions_in'])} |
| - model answered a tconst | {f['pred_tconst']:,} | {pct(f['pred_tconst'], f['adjudicated_mentions_in'])} |
| - model answered TITLE:... | {f['pred_TITLE']:,} | {pct(f['pred_TITLE'], f['adjudicated_mentions_in'])} |
| -- TITLE resolved to a tconst | {f['title_resolved']:,} | {pct(f['title_resolved'], f['pred_TITLE'])} of TITLE |
| -- TITLE unresolved (dropped) | {f['title_unresolved_dropped']:,} | {pct(f['title_unresolved_dropped'], f['pred_TITLE'])} of TITLE |
| - malformed answers (dropped) | {f['pred_malformed_dropped']:,} | {pct(f['pred_malformed_dropped'], f['adjudicated_mentions_in'])} |
| **model links** | **{f['model_links']:,}** | {pct(f['model_links'], f['adjudicated_mentions_in'])} of mentions in |
| rules-tier ACCEPT links in these threads | {f['rules_links_in_ruler_threads']:,} | (of {f['accepted_rows_total_all_threads']:,} over all 35,982 eval threads) |
| **links total** | **{f['links_total']:,}** | |
| - in the 18,359-item catalogue (model) | {f['in_catalogue_model']:,} | {pct(f['in_catalogue_model'], f['model_links'])} |
| - in the 18,359-item catalogue (rules) | {f['in_catalogue_rules']:,} | {pct(f['in_catalogue_rules'], f['rules_links_in_ruler_threads'])} |
| threads in scope | {f['threads_in_scope']:,} | |
| **threads kept (>= {stats['min_gt']} GT)** | **{f['threads_kept_ge5_gt']:,}** | {pct(f['threads_kept_ge5_gt'], f['threads_in_scope'])} |
| threads dropped (< {stats['min_gt']} GT) | {f['threads_dropped_lt5_gt']:,} | {pct(f['threads_dropped_lt5_gt'], f['threads_in_scope'])} |
| kept threads with request text | {f['threads_with_request_text']:,} | {pct(f['threads_with_request_text'], f['threads_kept_ge5_gt'])} |

## Ruler shape (kept threads)

| quantity | mean | median | p10 | p90 | max | total |
|---|---|---|---|---|---|---|
| GT items / thread | {g['mean']} | {g['median']} | {g['p10']} | {g['p90']} | {g['max']} | {g['total']:,} |
| OP_NAMED items / thread | {o['mean']} | {o['median']} | {o['p10']} | {o['p90']} | {o['max']} | {o['total']:,} |

OP_NAMED coverage: {stats['op_named_coverage']['threads_with_any_op_named']:,} of
{f['threads_kept_ge5_gt']:,} kept threads name at least one in-catalogue item in the request or a
seeker follow-up ({pct(stats['op_named_coverage']['threads_with_any_op_named'], f['threads_kept_ge5_gt'])}).

## TITLE resolution outcomes

| outcome | count |
|---|---|
""" + "\n".join(f"| {k} | {v:,} |" for k, v in stats["title_resolution_reasons"].items()) + f"""

Resolver: normalized match (case / punctuation / hyphen / diacritics insensitive, `&`==`and`,
apostrophes dropped) against BOTH `primaryTitle` and `originalTitle` of `title.basics.tsv`,
restricted to the same title types the candidate index uses; article variants ("the X" / "X")
tried; year within +-1 when the answer gave one; most-voted title wins ties. `year_mismatch` =
a title matched but no candidate sat within +-1 of the stated year -- dropped, not guessed.

## Spot check (10 random kept threads, seed {SEED}; reddit handles redacted)

| thread | GT | OP | first GT links (span -> tconst -> movieId) | request excerpt |
|---|---|---|---|---|
{spot_md}

## Anomalies / notes

- Threads dropped for < {stats['min_gt']} GT items: {f['threads_dropped_lt5_gt']:,}. These threads
  entered the adjudication scope on He's OWN links (>= 5 in-catalogue reply tconsts in the
  `processed` cell); after adjudication corrected those links some fall below the bar. That is the
  ruler doing its job, not a leak -- but it means the ruler's thread count is not a fixed fraction
  of the 35,982.
- Out-of-catalogue loss is the disclosed recency truncation (corpus 2019-22, ML-25M ends late
  2019); reported above per link source rather than hidden. It is strongly asymmetric between the
  two link sources -- {pct(f['in_catalogue_rules'], f['rules_links_in_ruler_threads'])} of
  rules-tier links are in catalogue vs {pct(f['in_catalogue_model'], f['model_links'])} of model
  links -- which is expected rather than alarming: the rules tier auto-accepts unambiguous
  high-vote classics, so everything hard, new, foreign or TV-shaped was DEFERred to the model.
- {f['pred_NONE_dropped']:,} DEFER mentions ({pct(f['pred_NONE_dropped'], f['adjudicated_mentions_in'])})
  were adjudicated NONE -- i.e. the model rejected He's link outright rather than repairing it.
  That is the intended behaviour on the population where He's raw precision was 0.572.
- Ground truth is entity-anchored: it can only contain titles someone actually named in a reply.

Raw counts: `data/reddit_he/ruler/RULER_STATS.json`. Ruler: `data/reddit_he/ruler/ruler_v1.csv.gz`.
Script: `scripts/reddit/assemble_ruler.py`.
"""
    OUT_MD.write_text(md, encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())
