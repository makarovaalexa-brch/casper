"""Batch preparation for the He-corpus Sonnet adjudication run (NO LLM calls here).

Implements `docs/design/DS_REDDIT_LINKING.md` section "PIPELINE OF RECORD":

  * scope    = span-alignable mentions inside the 35,982 eval threads (>=5 in-catalogue reply
               items, OP-named removed) -- the census-parity thread set of REDDIT_HE_CENSUS.md,
               resolved exactly as `tier_split_census.py` does (unparsed `processed` cell, so the
               thread set stays bit-identical to the published 35,982).
  * tier     = `linker_rules_tier.DECIDE`, copied verbatim via `tier_split_census` (import, so
               there is literally one copy of the thresholds in play).
  * ACCEPT   -> final links, written to adjudication/accepted.csv.gz.  No model sees them.
  * DEFER    -> grouped by thread; the THREAD list is shuffled once with random.Random(98765)
               and packed, threads kept intact, into batches of ~500 mentions (a batch closes
               when adding the next thread would take it past 550).  Any prefix of completed
               batches is therefore an unbiased random subsample of eval threads with FULLY
               adjudicated ground truth.

HARD RULE 1: nothing is sampled, capped or truncated.  The thread-level shuffle above is the
authorized, documented sampling mechanism (it re-orders, it does not drop), and the totals are
asserted against the published census DEFER count.

Privacy: no username, no user hash, and no free text beyond the mention span and its enclosing
sentence appear in any output file.

Passes:
  1. `tier_split_census.py`'s pass-1 cache supplies the distinct normalized span set (it is a
     count-aggregate and carries no thread/turn ids, so it cannot produce the per-mention rows).
  2. one streaming pass over title.basics.tsv builds the span-only candidate index.
  3. one streaming pass over large_v1/{train,valid,test}.csv emits per-mention rows.

Usage (from C:/dev/phd/casper):
    poetry run python scripts/reddit/prep_adjudication_batches.py
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from census_he import (  # noqa: E402
    TCONST_RE,
    align_mentions,
    iter_v1_rows,
    load_catalogue,
    parse_utt,
    sentence_around,
    thread_of,
    year_of,
)
from tier_split_census import (  # noqa: E402  (DECIDE / norm / span_year / variants, verbatim)
    DECIDE,
    build_candidates,
    norm,
    span_year,
)

ROOT = Path(__file__).resolve().parents[2]
V1_DIR = ROOT / "data" / "reddit_he" / "large_v1"
BASE = ROOT / "data" / "reddit_he"
CACHE = BASE / "census" / "tier_split_pass1_cache.json.gz"
OUT_DIR = BASE / "adjudication"
BATCH_DIR = OUT_DIR / "batches"
ACCEPTED = OUT_DIR / "accepted.csv.gz"
MANIFEST = OUT_DIR / "manifest.json"

SEED = 98765
BATCH_TARGET = 500
BATCH_MAX = 550
TOP_CANDIDATES = 8

# published census numbers this run must reconcile to (docs/results/TIER_SPLIT_CENSUS.md)
CENSUS_EVAL_MENTIONS = 823_128
CENSUS_EVAL_ACCEPT = 465_916
CENSUS_EVAL_DEFER = 357_212
CENSUS_EVAL_THREADS = 35_982

csv.field_size_limit(1 << 31 - 1 if sys.maxsize > 2**32 else (1 << 30))


def sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


# ------------------------------------------------------------------ candidate index


def load_spans_from_cache(cache: Path) -> set[str]:
    with gzip.open(cache, "rt", encoding="utf-8") as f:
        state = json.load(f)
    spans = {k[0] for k in state["keys"]}
    print(f"pass1 cache {cache}: {len(state['keys'])} keys, {len(spans)} distinct spans",
          flush=True)
    return spans


# ------------------------------------------------------------------ streaming pass


def stream(cand: dict):
    """Yield nothing; return (accepted_rows, defer_by_thread, stats).

    Thread rows are contiguous in the CSVs, so each thread is buffered and flushed (kept or
    dropped by eval-set membership) as soon as the thread id changes.
    """
    in_cat, _cat = load_catalogue()

    dec_cache: dict[tuple, tuple] = {}   # (nrm, sy, tc) -> (decision, cand_list)
    accepted: list[tuple] = []
    defer_by_thread: dict[str, list] = {}
    stats = Counter()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    BATCH_DIR.mkdir(parents=True, exist_ok=True)

    cur = None

    def flush(t):
        if t is None:
            return
        stats["threads"] += 1
        is_eval = len(t["replier_cat"] - t["seeker_cat"]) >= 5
        if not is_eval:
            stats["mentions_non_eval"] += len(t["buf"])
            return
        stats["eval_threads"] += 1
        for rec in t["buf"]:
            stats["eval_mentions"] += 1
            if rec["decision"] == "ACCEPT":
                stats["accept"] += 1
                accepted.append((t["tid"], rec["turn_id"], rec["year"], rec["context"],
                                 sha1(rec["span"]), rec["span"], rec["tconst"]))
            else:
                stats["defer"] += 1
                defer_by_thread.setdefault(t["tid"], []).append(rec)

    for split in ["train", "valid", "test"]:
        path = V1_DIR / f"{split}.csv"
        if not path.exists():
            print(f"MISSING {path}", flush=True)
            continue
        for row in iter_v1_rows(path):
            stats["rows"] += 1
            if stats["rows"] % 200000 == 0:
                print(f"  {split}: {stats['rows']} rows, eval acc={stats['accept']} "
                      f"def={stats['defer']}", flush=True)
            tid = thread_of(row["conv_id"])
            if cur is None or cur["tid"] != tid:
                flush(cur)
                cur = {"tid": tid, "turns": set(), "seeker_cat": set(), "replier_cat": set(),
                       "buf": []}
            turn = row["turn_id"]
            if turn in cur["turns"]:
                continue
            cur["turns"].add(turn)
            stats["unique_turns"] += 1

            is_seeker = str(row.get("is_seeker", "")).strip().lower() == "true"
            order = int(row.get("turn_order") or 0)
            role = "request" if (is_seeker and order == 0) else (
                "seeker_followup" if is_seeker else "reply")
            y = year_of(row.get("utc_time"))

            proc_cell = row.get("processed") or ""
            # eval-set membership from the UNPARSED cell -- census parity (see tier_split_census)
            for tc in TCONST_RE.findall(proc_cell):
                if tc in in_cat:
                    (cur["seeker_cat"] if is_seeker else cur["replier_cat"]).add(tc)

            proc = parse_utt(proc_cell)
            if not TCONST_RE.search(proc):
                continue
            raw = parse_utt(row.get("raw") or "")
            for tc, surface_full in align_mentions(proc, raw):
                surface = surface_full[:200]
                stats["aligned_mentions"] += 1
                nrm = norm(surface)
                sy = span_year(surface)
                key = (nrm, sy, tc)
                hit = dec_cache.get(key)
                if hit is None:
                    dec, cs = DECIDE(nrm, sy, tc, cand)
                    hit = (dec, [{"tconst": c[0], "type": c[1], "title": c[2],
                                  "year": c[3], "votes": c[4]} for c in cs[:TOP_CANDIDATES]])
                    dec_cache[key] = hit
                dec, cands = hit
                rec = {"decision": dec, "turn_id": turn, "year": y, "context": role,
                       "span": surface, "tconst": tc}
                if dec != "ACCEPT":
                    rec["sentence"] = sentence_around(raw, surface_full)
                    rec["candidates"] = cands
                cur["buf"].append(rec)
        print(f"{split}: done ({stats['rows']} rows cumulative)", flush=True)
    flush(cur)
    return accepted, defer_by_thread, stats


# ------------------------------------------------------------------ outputs


def write_accepted(rows) -> None:
    with gzip.open(ACCEPTED, "wt", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["thread_id", "turn_id", "year", "context", "span_sha1", "span", "tconst"])
        w.writerows(rows)
    print(f"wrote {ACCEPTED} ({len(rows)} rows)", flush=True)


def pack_batches(defer_by_thread: dict) -> list[list[str]]:
    threads = sorted(defer_by_thread)
    random.Random(SEED).shuffle(threads)
    batches, cur, n = [], [], 0
    for t in threads:
        k = len(defer_by_thread[t])
        if cur and n + k > BATCH_MAX:
            batches.append(cur)
            cur, n = [], 0
        cur.append(t)
        n += k
    if cur:
        batches.append(cur)
    return batches


def write_batches(batches, defer_by_thread) -> tuple[list[dict], int]:
    for old in BATCH_DIR.glob("batch_*.jsonl"):
        old.unlink()
    manifest, gid = [], 0
    for bi, tids in enumerate(batches):
        path = BATCH_DIR / f"batch_{bi:04d}.jsonl"
        n = 0
        with path.open("w", encoding="utf-8", newline="\n") as f:
            for tid in tids:
                for rec in defer_by_thread[tid]:
                    obj = {
                        "id": gid, "batch": bi, "thread_id": tid, "turn_id": rec["turn_id"],
                        "year": rec["year"], "context": rec["context"], "span": rec["span"],
                        "sentence": rec["sentence"], "candidates": rec["candidates"],
                    }
                    f.write(json.dumps(obj, ensure_ascii=False) + "\n")
                    gid += 1
                    n += 1
        manifest.append({
            "batch": bi, "file": f"batches/batch_{bi:04d}.jsonl",
            "n_mentions": n, "n_threads": len(tids),
            "thread_ids_sha1": sha1("\n".join(tids)),
        })
    return manifest, gid


# ------------------------------------------------------------------ main


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=str(CACHE))
    args = ap.parse_args()

    spans = load_spans_from_cache(Path(args.cache))
    cand = build_candidates(spans)

    accepted, defer_by_thread, stats = stream(cand)

    print("\n--- stream stats ---")
    print(json.dumps(dict(sorted(stats.items())), indent=2))

    write_accepted(accepted)
    batches = pack_batches(defer_by_thread)
    manifest, total_defer = write_batches(batches, defer_by_thread)

    sizes = [m["n_mentions"] for m in manifest]
    first60 = sum(sizes[:60])
    first60_threads = sum(m["n_threads"] for m in manifest[:60])

    doc = {
        "seed": SEED,
        "shuffle": "sorted thread ids, random.Random(98765).shuffle, threads never split",
        "batch_target": BATCH_TARGET,
        "batch_max": BATCH_MAX,
        "scope": "DEFER mentions in the 35,982 census-parity eval threads",
        "rules_tier": "tier_split_census.DECIDE (== linker_rules_tier.py), thresholds unchanged",
        "accepted_file": "accepted.csv.gz",
        "totals": {
            "eval_threads": stats["eval_threads"],
            "eval_mentions": stats["eval_mentions"],
            "accepted_mentions": stats["accept"],
            "defer_mentions": total_defer,
            "defer_threads": len(defer_by_thread),
            "n_batches": len(manifest),
            "mentions_first_60_batches": first60,
            "threads_first_60_batches": first60_threads,
        },
        "census_reconciliation": {
            "census_eval_threads": CENSUS_EVAL_THREADS, "ours": stats["eval_threads"],
            "census_eval_mentions": CENSUS_EVAL_MENTIONS, "ours_mentions": stats["eval_mentions"],
            "census_accept": CENSUS_EVAL_ACCEPT, "ours_accept": stats["accept"],
            "census_defer": CENSUS_EVAL_DEFER, "ours_defer": total_defer,
        },
        "batches": manifest,
    }
    MANIFEST.write_text(json.dumps(doc, indent=2), encoding="utf-8")

    print("\n--- reconciliation vs TIER_SPLIT_CENSUS.md (eval scope) ---")
    for label, census, ours in [
        ("eval threads", CENSUS_EVAL_THREADS, stats["eval_threads"]),
        ("eval mentions", CENSUS_EVAL_MENTIONS, stats["eval_mentions"]),
        ("ACCEPT", CENSUS_EVAL_ACCEPT, stats["accept"]),
        ("DEFER", CENSUS_EVAL_DEFER, total_defer),
    ]:
        print(f"  {label:<14} census {census:>9,}   ours {ours:>9,}   diff {ours - census:+,}")

    print("\n--- batches ---")
    print(f"  n batches            {len(manifest)}")
    print(f"  mentions total       {total_defer:,}")
    print(f"  mentions/batch       min {min(sizes)}  median {sorted(sizes)[len(sizes)//2]}  "
          f"mean {sum(sizes)/len(sizes):.1f}  max {max(sizes)}")
    print(f"  threads/batch        min {min(m['n_threads'] for m in manifest)}  "
          f"max {max(m['n_threads'] for m in manifest)}")
    print(f"  first 60 batches     {first60:,} mentions / {first60_threads:,} threads "
          f"({100*first60/max(total_defer,1):.1f}% of DEFER)")
    print(f"\nwrote {MANIFEST}\nwrote {BATCH_DIR} ({len(manifest)} files)")

    assert stats["eval_threads"] == CENSUS_EVAL_THREADS, "eval thread count off census"
    assert stats["eval_mentions"] == CENSUS_EVAL_MENTIONS, "eval mention count off census"
    assert stats["accept"] == CENSUS_EVAL_ACCEPT, "ACCEPT count off census"
    assert total_defer == CENSUS_EVAL_DEFER, "DEFER count off census"
    assert total_defer == sum(sizes), "batch sizes do not sum to DEFER total"
    print("\nALL RECONCILIATION ASSERTS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
