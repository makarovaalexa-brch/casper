"""Descriptive census of the He et al. Reddit-Movie corpus (CENSUS phase, no LLM).

Two data products are censused:

  RAW   `data/reddit_he/Reddit-Movie-large/subreddit/*.jsonl`
        One JSON object per THREAD: {submission, valid_comments, head_comments,
        ghost_comments}.  Carries NO entity annotations of any kind (the release's
        `sentences.jsonl` is sentence-segmented raw text, and `IMDB-database/` is a
        gazetteer, not a linking).  Used for: per-year thread counts, reply-count
        distributions, OP-follow-up subset.

  V1    `data/reddit_he/large_v1/{train,valid,test}.csv`
        He et al.'s own "first version" processed release.  This is where their
        machine-matched movie mentions live: column `processed` is the utterance with
        every linked mention replaced in-place by an IMDb tconst, `raw` is the same
        utterance with the original surface strings.  Used for: mention counts,
        catalogue fit, ground-truth viability, the gold sample.

HARD RULES honoured here:
  * nothing is sampled, capped or truncated in any COUNT -- every file is streamed
    end to end and every record is counted (the only sample drawn is the explicitly
    requested 300-mention gold set for human labelling, written to its own file);
  * no LLM call of any kind; all linking numbers are He et al.'s own links;
  * thread-level evaluation only -- the V1 CSV is path-expanded ("conv_id" trap), so
    turns are deduplicated by reddit `turn_id` within a thread and threads by the
    submission id `t3_xxxxx`;
  * usernames are sha256-hashed (first 12 hex) in every derived file written.

Usage (from C:/dev/phd/casper):
    poetry run python scripts/reddit/census_he.py raw
    poetry run python scripts/reddit/census_he.py v1
    poetry run python scripts/reddit/census_he.py gold
"""

from __future__ import annotations

import argparse
import ast
import csv
import datetime as dt
import difflib
import hashlib
import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "reddit_he" / "Reddit-Movie-large" / "subreddit"
V1_DIR = ROOT / "data" / "reddit_he" / "large_v1"
OUT_DIR = ROOT / "data" / "reddit_he" / "census"
LINKS = ROOT / "data" / "movielens" / "links.csv"
CATALOGUE = ROOT / "data" / "ml-25m" / "proc" / "unique_sid.txt"

SUBREDDITS = ["bestofnetflix", "movies", "moviesuggestions", "netflixbestof", "truefilm"]
TCONST_RE = re.compile(r"\btt\d{7,8}\b")
IMDB_URL_RE = re.compile(r"imdb\.com/title/(tt\d{7,8})")

csv.field_size_limit(1 << 31 - 1 if sys.maxsize > 2**32 else (1 << 30))


# ---------------------------------------------------------------- helpers


def uhash(name: str | None) -> str:
    """sha256, first 12 hex -- pseudonymisation applied to every author string."""
    if name is None:
        return "NA"
    return hashlib.sha256(name.encode("utf-8")).hexdigest()[:12]


def year_of(utc) -> int | None:
    try:
        return dt.datetime.fromtimestamp(float(utc), dt.timezone.utc).year
    except (TypeError, ValueError, OSError, OverflowError):
        return None


def quantiles(values: list[int]) -> dict:
    if not values:
        return {"n": 0}
    v = sorted(values)
    n = len(v)

    def q(p: float) -> float:
        if n == 1:
            return float(v[0])
        idx = p * (n - 1)
        lo, hi = int(idx), min(int(idx) + 1, n - 1)
        return v[lo] + (v[hi] - v[lo]) * (idx - lo)

    return {
        "n": n,
        "mean": round(sum(v) / n, 3),
        "p10": q(0.10),
        "median": q(0.50),
        "p90": q(0.90),
        "max": v[-1],
        "total": sum(v),
    }


def load_catalogue() -> tuple[dict[str, int], set[int]]:
    """tconst -> movieId, restricted to the 18,359-item project catalogue."""
    cat = {int(x) for x in CATALOGUE.read_text().split()}
    tconst2mid: dict[str, int] = {}
    with LINKS.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            imdb = (row["imdbId"] or "").strip()
            if not imdb:
                continue
            mid = int(row["movieId"])
            tconst2mid[f"tt{int(imdb):07d}"] = mid
            tconst2mid[f"tt{imdb}"] = mid  # tolerate un-zero-padded forms
    in_cat = {t: m for t, m in tconst2mid.items() if m in cat}
    return in_cat, cat


# ---------------------------------------------------------------- RAW pass


def census_raw(out: Path) -> dict:
    result: dict = {}
    for sub in SUBREDDITS:
        path = RAW_DIR / f"{sub}.jsonl"
        by_year_threads: Counter = Counter()
        by_year_comments: Counter = Counter()
        replies: list[int] = []
        ghosts_total = 0
        seen_sids: set[str] = set()
        dup_sids = 0
        n_lines = 0
        n_multi_submission = 0
        # OP follow-up accounting
        op_any = 0  # OP posted >=1 comment
        op_after_other = 0  # OP replied to a comment by someone else
        op_by_year: Counter = Counter()
        op_after_other_by_year: Counter = Counter()
        threads_with_is_submitter_field = 0
        # free linking signal that needs no model: explicit imdb.com/title links
        threads_with_imdb_url = 0
        imdb_url_mentions = 0
        no_year = 0

        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                n_lines += 1
                d = json.loads(line)
                subs = d.get("submission") or {}
                if len(subs) != 1:
                    n_multi_submission += 1
                for sid, s in subs.items():
                    if sid in seen_sids:
                        dup_sids += 1
                        continue
                    seen_sids.add(sid)
                    y = year_of(s.get("created_utc"))
                    if y is None:
                        no_year += 1
                    else:
                        by_year_threads[y] += 1
                    op = s.get("author")
                    op_h = uhash(op)

                    comments = d.get("valid_comments") or {}
                    replies.append(len(comments))
                    ghosts_total += len(d.get("ghost_comments") or [])
                    if y is not None:
                        by_year_comments[y] += len(comments)

                    has_flag = False
                    op_posted = False
                    op_replied_other = False
                    urlhit = False
                    for cid, c in comments.items():
                        if "is_submitter" in c:
                            has_flag = True
                        is_op = bool(c.get("is_submitter")) or (
                            c.get("author") is not None
                            and uhash(c.get("author")) == op_h
                            and op not in (None, "[deleted]", "AutoModerator")
                        )
                        if is_op:
                            op_posted = True
                            parent = c.get("parent_id") or ""
                            if parent.startswith("t1_"):
                                pc = comments.get(parent)
                                if pc is not None:
                                    p_is_op = bool(pc.get("is_submitter")) or (
                                        pc.get("author") is not None
                                        and uhash(pc.get("author")) == op_h
                                    )
                                    if not p_is_op:
                                        op_replied_other = True
                        body = c.get("body") or ""
                        hits = IMDB_URL_RE.findall(body)
                        if hits:
                            urlhit = True
                            imdb_url_mentions += len(hits)
                    body0 = (s.get("selftext") or "") + " " + (s.get("title") or "")
                    hits0 = IMDB_URL_RE.findall(body0)
                    if hits0:
                        urlhit = True
                        imdb_url_mentions += len(hits0)

                    threads_with_is_submitter_field += int(has_flag)
                    threads_with_imdb_url += int(urlhit)
                    if op_posted:
                        op_any += 1
                        if y is not None:
                            op_by_year[y] += 1
                    if op_replied_other:
                        op_after_other += 1
                        if y is not None:
                            op_after_other_by_year[y] += 1

        n_threads = len(seen_sids)
        rec = {
            "file": str(path),
            "jsonl_lines": n_lines,
            "threads_deduped_by_submission_id": n_threads,
            "duplicate_submission_ids_dropped": dup_sids,
            "objects_with_multiple_submissions": n_multi_submission,
            "threads_missing_created_utc": no_year,
            "threads_by_year": dict(sorted(by_year_threads.items())),
            "comments_by_year": dict(sorted(by_year_comments.items())),
            "year_sum_check": sum(by_year_threads.values()) + no_year == n_threads,
            "replies_per_thread": quantiles(replies),
            "ghost_comments_total": ghosts_total,
            "threads_with_any_is_submitter_field": threads_with_is_submitter_field,
            "op_followup_threads": op_any,
            "op_followup_pct": round(100 * op_any / max(n_threads, 1), 2),
            "op_replied_to_other_threads": op_after_other,
            "op_replied_to_other_pct": round(100 * op_after_other / max(n_threads, 1), 2),
            "op_followup_by_year": dict(sorted(op_by_year.items())),
            "op_replied_to_other_by_year": dict(sorted(op_after_other_by_year.items())),
            "threads_with_explicit_imdb_url": threads_with_imdb_url,
            "explicit_imdb_url_mentions": imdb_url_mentions,
        }
        result[sub] = rec
        print(json.dumps({sub: rec}, indent=2)[:4000], flush=True)

    out.write_text(json.dumps(result, indent=2))
    print(f"\nwrote {out}")
    return result


# ---------------------------------------------------------------- V1 pass


def iter_v1_rows(path: Path):
    with path.open("r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            yield row


def thread_of(conv_id: str) -> str:
    # "t3_rt7enj_0/14" -> "t3_rt7enj"
    return conv_id.rsplit("_", 1)[0]


def census_v1(out: Path) -> dict:
    in_cat, _cat = load_catalogue()
    id2name_path = V1_DIR / "id2name.json"
    id2name = json.loads(id2name_path.read_text(encoding="utf-8")) if id2name_path.exists() else {}

    # per-thread accumulators, flushed when the thread id changes (rows for one
    # thread are contiguous because the CSV is path-expanded in order)
    thread_seen: set[str] = set()
    out_of_order = 0

    stats = {
        "rows": 0,
        "unique_turns": 0,
        "threads": 0,
        "mentions_total": 0,
        "mentions_seeker": 0,
        "mentions_replier": 0,
        "mentions_in_catalogue": 0,
        "mentions_replier_in_catalogue": 0,
    }
    distinct_items: Counter = Counter()
    distinct_items_by_year: dict[int, Counter] = defaultdict(Counter)
    mentions_by_year: Counter = Counter()
    mentions_in_cat_by_year: Counter = Counter()
    replier_mentions_by_year: Counter = Counter()
    replier_in_cat_by_year: Counter = Counter()
    threads_by_year: Counter = Counter()
    turns_per_thread: list[int] = []
    replier_turns_per_thread: list[int] = []
    linked_turns = 0
    turns_with_no_link = 0
    threads_with_any_link = 0
    gt_ge5: Counter = Counter()  # year -> threads with >=5 in-cat replier items (OP-named removed)
    gt_ge5_total = 0
    gt_sizes: list[int] = []
    per_file: dict[str, dict] = {}

    cur = None

    def flush(t):
        nonlocal threads_with_any_link, gt_ge5_total
        if t is None:
            return
        y = t["year"]
        stats["threads"] += 1
        if y is not None:
            threads_by_year[y] += 1
        turns_per_thread.append(len(t["turns"]))
        replier_turns_per_thread.append(t["replier_turns"])
        if t["any_link"]:
            threads_with_any_link += 1
        gt = (t["replier_items_cat"] - t["seeker_items_cat"])
        gt_sizes.append(len(gt))
        if len(gt) >= 5:
            gt_ge5_total += 1
            if y is not None:
                gt_ge5[y] += 1

    for split in ["train", "valid", "test"]:
        path = V1_DIR / f"{split}.csv"
        if not path.exists():
            continue
        f_rows = 0
        f_turns = 0
        f_threads = 0
        for row in iter_v1_rows(path):
            f_rows += 1
            stats["rows"] += 1
            tid = thread_of(row["conv_id"])
            if cur is None or cur["tid"] != tid:
                flush(cur)
                if cur is not None:
                    f_threads += 1
                if tid in thread_seen:
                    out_of_order += 1
                thread_seen.add(tid)
                cur = {
                    "tid": tid,
                    "year": None,
                    "turns": set(),
                    "replier_turns": 0,
                    "any_link": False,
                    "seeker_items_cat": set(),
                    "replier_items_cat": set(),
                }
            turn = row["turn_id"]
            if turn in cur["turns"]:
                continue  # same turn re-used by another extracted path
            cur["turns"].add(turn)
            f_turns += 1
            stats["unique_turns"] += 1

            y = year_of(row.get("utc_time"))
            is_seeker = str(row.get("is_seeker", "")).strip().lower() == "true"
            if cur["year"] is None and is_seeker:
                cur["year"] = y
            if cur["year"] is None:
                cur["year"] = y
            if not is_seeker:
                cur["replier_turns"] += 1

            tconsts = TCONST_RE.findall(row.get("processed") or "")
            if tconsts:
                linked_turns += 1
                cur["any_link"] = True
            else:
                turns_with_no_link += 1
            for tc in tconsts:
                stats["mentions_total"] += 1
                incat = tc in in_cat
                if y is not None:
                    mentions_by_year[y] += 1
                    distinct_items_by_year[y][tc] += 1
                distinct_items[tc] += 1
                if incat:
                    stats["mentions_in_catalogue"] += 1
                    if y is not None:
                        mentions_in_cat_by_year[y] += 1
                if is_seeker:
                    stats["mentions_seeker"] += 1
                    if incat:
                        cur["seeker_items_cat"].add(tc)
                else:
                    stats["mentions_replier"] += 1
                    if y is not None:
                        replier_mentions_by_year[y] += 1
                    if incat:
                        stats["mentions_replier_in_catalogue"] += 1
                        cur["replier_items_cat"].add(tc)
                        if y is not None:
                            replier_in_cat_by_year[y] += 1
        per_file[split] = {"rows": f_rows, "unique_turns": f_turns}
        print(f"{split}: {f_rows} rows, {f_turns} unique turns", flush=True)
    flush(cur)

    n_dist = len(distinct_items)
    n_dist_cat = sum(1 for t in distinct_items if t in in_cat)
    by_year = {}
    for y in sorted(set(list(mentions_by_year) + list(threads_by_year))):
        dy = distinct_items_by_year.get(y, Counter())
        dcat = sum(1 for t in dy if t in in_cat)
        by_year[y] = {
            "threads": threads_by_year.get(y, 0),
            "mentions": mentions_by_year.get(y, 0),
            "mentions_in_catalogue": mentions_in_cat_by_year.get(y, 0),
            "pct_mentions_in_catalogue": round(
                100 * mentions_in_cat_by_year.get(y, 0) / max(mentions_by_year.get(y, 0), 1), 2
            ),
            "distinct_items": len(dy),
            "distinct_items_in_catalogue": dcat,
            "pct_distinct_in_catalogue": round(100 * dcat / max(len(dy), 1), 2),
            "replier_mentions": replier_mentions_by_year.get(y, 0),
            "replier_mentions_in_catalogue": replier_in_cat_by_year.get(y, 0),
            "threads_ge5_in_cat_reply_items": gt_ge5.get(y, 0),
        }

    res = {
        "protocol": {
            "files": [str(V1_DIR / f"{s}.csv") for s in ["train", "valid", "test"]],
            "thread_dedup_key": "submission id = conv_id minus the trailing _<path>/<npaths>",
            "turn_dedup_key": "reddit turn_id within a thread",
            "linking_source": "He et al. V1 `processed` column (tconsts substituted in place)",
            "catalogue_map": "links.csv movieId<->imdbId, restricted to unique_sid.txt (18,359 ids)",
        },
        "totals": stats,
        "id2name_entries": len(id2name),
        "threads_seen": len(thread_seen),
        "threads_reopened_out_of_order": out_of_order,
        "per_file": per_file,
        "turns_per_thread": quantiles(turns_per_thread),
        "replier_turns_per_thread": quantiles(replier_turns_per_thread),
        "turns_with_at_least_one_link": linked_turns,
        "turns_with_no_link": turns_with_no_link,
        "pct_turns_linked": round(100 * linked_turns / max(stats["unique_turns"], 1), 2),
        "threads_with_any_link": threads_with_any_link,
        "pct_threads_with_any_link": round(
            100 * threads_with_any_link / max(stats["threads"], 1), 2
        ),
        "distinct_items": n_dist,
        "distinct_items_in_catalogue": n_dist_cat,
        "pct_distinct_in_catalogue": round(100 * n_dist_cat / max(n_dist, 1), 2),
        "pct_mentions_in_catalogue": round(
            100 * stats["mentions_in_catalogue"] / max(stats["mentions_total"], 1), 2
        ),
        "ground_truth_set_sizes": quantiles(gt_sizes),
        "threads_ge5_in_cat_reply_items": gt_ge5_total,
        "pct_threads_ge5": round(100 * gt_ge5_total / max(stats["threads"], 1), 2),
        "by_year": by_year,
        "sanity_year_sum": {
            "threads_by_year_sum": sum(threads_by_year.values()),
            "threads_total": stats["threads"],
            "mentions_by_year_sum": sum(mentions_by_year.values()),
            "mentions_total": stats["mentions_total"],
        },
    }
    out.write_text(json.dumps(res, indent=2))
    print(f"wrote {out}")
    return res


# ---------------------------------------------------------------- gold sample


def align_mentions(processed: str, raw: str) -> list[tuple[str, str]]:
    """Recover (tconst, surface span) pairs by aligning `processed` against `raw`.

    `processed` is `raw` with each linked mention replaced in place by its tconst, so a
    word-level diff yields the surface string that each tconst stands for.
    """
    p_tok = processed.split()
    r_tok = raw.split()
    sm = difflib.SequenceMatcher(a=p_tok, b=r_tok, autojunk=False)
    pairs = []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag != "replace":
            continue
        seg = p_tok[i1:i2]
        tcs = [t for t in seg if TCONST_RE.fullmatch(t.strip(".,;:!?()[]\"'"))]
        if len(tcs) == 1:
            surface = " ".join(r_tok[j1:j2]).strip()
            if surface:
                pairs.append((TCONST_RE.search(tcs[0]).group(0), surface))
    return pairs


def sentence_around(text: str, surface: str) -> str:
    idx = text.find(surface)
    if idx < 0:
        return ""
    start = max(text.rfind(".", 0, idx), text.rfind("\n", 0, idx), text.rfind("!", 0, idx)) + 1
    ends = [e for e in (text.find(".", idx), text.find("\n", idx), text.find("!", idx)) if e >= 0]
    end = min(ends) + 1 if ends else len(text)
    return " ".join(text[start:end].split())[:400]


def parse_utt(cell: str) -> str:
    """The `processed`/`raw` cells are python-literal ['USER', '<text>'] pairs."""
    try:
        v = ast.literal_eval(cell)
        if isinstance(v, (list, tuple)) and len(v) >= 2:
            return str(v[1])
    except (ValueError, SyntaxError):
        pass
    return cell


def draw_gold(out_csv: Path, target: int = 300, seed: int = 98765) -> dict:
    rng = random.Random(seed)
    per_stratum = 60  # reservoir depth per (year, context) cell; trimmed to `target` after
    reservoirs: dict[tuple, list] = defaultdict(list)
    counts: Counter = Counter()
    seen_turn_per_thread: set[str] = set()
    cur_tid = None
    v1_threads: set[str] = set()
    surface_counts: Counter = Counter()

    for split in ["train", "valid", "test"]:
        path = V1_DIR / f"{split}.csv"
        if not path.exists():
            continue
        for row in iter_v1_rows(path):
            tid = thread_of(row["conv_id"])
            if tid != cur_tid:
                cur_tid, seen_turn_per_thread = tid, set()
                v1_threads.add(tid)
            if row["turn_id"] in seen_turn_per_thread:
                continue
            seen_turn_per_thread.add(row["turn_id"])
            y = year_of(row.get("utc_time"))
            if y is None:
                continue
            is_seeker = str(row.get("is_seeker", "")).strip().lower() == "true"
            order = int(row.get("turn_order") or 0)
            ctx = "request" if (is_seeker and order == 0) else (
                "seeker_followup" if is_seeker else "reply"
            )
            proc = parse_utt(row.get("processed") or "")
            if not TCONST_RE.search(proc):
                continue
            raw = parse_utt(row.get("raw") or "")
            for tc, surface in align_mentions(proc, raw):
                key = (y, ctx)
                counts[key] += 1
                surface_counts[surface.strip(".,;:!?()[]\"' ").lower()[:80]] += 1
                res = reservoirs[key]
                item = {
                    "thread_id": tid,
                    "turn_id": row["turn_id"],
                    "user_hash": uhash(row.get("user_id")),
                    "year": y,
                    "context": ctx,
                    "he_tconst": tc,
                    "mention_span": surface[:200],
                    "sentence": sentence_around(raw, surface),
                    "label_tconst_or_NONE": "",
                }
                if len(res) < per_stratum:
                    res.append(item)
                else:
                    j = rng.randrange(counts[key])
                    if j < per_stratum:
                        res[j] = item

    strata = sorted(reservoirs)
    if not strata:
        raise SystemExit("no mentions found -- check V1 files")
    # even allocation over non-empty strata, capped by what each reservoir holds
    alloc = {k: 0 for k in strata}
    remaining = target
    while remaining > 0 and any(alloc[k] < len(reservoirs[k]) for k in strata):
        for k in strata:
            if remaining == 0:
                break
            if alloc[k] < len(reservoirs[k]):
                alloc[k] += 1
                remaining -= 1
    rows = []
    for k in strata:
        rows.extend(reservoirs[k][: alloc[k]])
    rng.shuffle(rows)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "thread_id",
        "turn_id",
        "user_hash",
        "year",
        "context",
        "he_tconst",
        "mention_span",
        "sentence",
        "label_tconst_or_NONE",
    ]
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    summary = {
        "file": str(out_csv),
        "n_rows": len(rows),
        "strata": len(strata),
        "population_mentions_alignable": sum(counts.values()),
        "allocation": {f"{k[0]}|{k[1]}": alloc[k] for k in strata},
        "population_by_stratum": {f"{k[0]}|{k[1]}": counts[k] for k in strata},
    }
    (OUT_DIR / "v1_thread_ids.txt").write_text("\n".join(sorted(v1_threads)))
    summary["v1_threads"] = len(v1_threads)
    summary["top_surface_spans"] = surface_counts.most_common(60)
    summary["distinct_surface_spans"] = len(surface_counts)
    (OUT_DIR / "gold_sample_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2)[:6000])
    return summary


def census_coverage(out: Path) -> dict:
    """Which RAW threads survive into He et al.'s processed V1 release, per subreddit."""
    v1 = set((OUT_DIR / "v1_thread_ids.txt").read_text().split())
    res = {"v1_threads": len(v1)}
    for sub in SUBREDDITS:
        kept = Counter()
        total = Counter()
        n_kept = n_tot = 0
        with (RAW_DIR / f"{sub}.jsonl").open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                d = json.loads(line)
                for sid, s in (d.get("submission") or {}).items():
                    y = year_of(s.get("created_utc"))
                    n_tot += 1
                    total[y] += 1
                    if sid in v1:
                        n_kept += 1
                        kept[y] += 1
        res[sub] = {
            "raw_threads": n_tot,
            "kept_in_v1": n_kept,
            "pct_kept": round(100 * n_kept / max(n_tot, 1), 2),
            "by_year": {
                str(y): {"raw": total[y], "kept": kept.get(y, 0)} for y in sorted(total)
            },
        }
        print(sub, res[sub]["raw_threads"], res[sub]["kept_in_v1"], res[sub]["pct_kept"], flush=True)
    out.write_text(json.dumps(res, indent=2))
    return res


# ---------------------------------------------------------------- main


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["raw", "v1", "gold", "coverage"])
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if args.phase == "raw":
        census_raw(OUT_DIR / "raw_census.json")
    elif args.phase == "v1":
        census_v1(OUT_DIR / "v1_census.json")
    elif args.phase == "coverage":
        census_coverage(OUT_DIR / "v1_coverage.json")
    else:
        draw_gold(ROOT / "data" / "reddit_he" / "gold_sample_unlabeled.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
