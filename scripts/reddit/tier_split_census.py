"""Volume-weighted rules-tier split over the FULL He Reddit corpus (no LLM, no sampling).

Prices the Opus adjudication run: the audition (`docs/results/REDDIT_LINKER_AUDITION.md`)
measured the rules tier on a 500-mention gold set whose strata deliberately over-represent
hard contexts (request / seeker_followup turns are ~2% of corpus volume but 1/3 of the gold).
The DEFER share that actually matters is volume-weighted over every mention in the corpus.

Method (HARD RULE 1 honoured -- nothing sampled, capped or truncated):
  pass 1  stream train.csv + valid.csv + test.csv end to end, dedup threads by submission id
          and turns by reddit turn_id (exactly as census_he.py), recover every (tconst, surface
          span) pair with census_he.align_mentions, and aggregate them into a counter keyed by
          (normalized span, span year, He tconst, turn role, in-eval-thread-set).  Thread rows
          are contiguous, so the eval-set flag (>=5 in-catalogue reply items, OP-named removed
          -- the 35,982-thread set of REDDIT_HE_CENSUS.md) is resolved at thread flush.
  pass 2  stream title.basics.tsv once, keyed by ALL distinct normalized spans seen in pass 1
          (plus their "the "-prefixed/stripped variants), to build the span-only candidate index.
  pass 3  apply linker_rules_tier.DECIDE to each distinct key and multiply by its count.

DECIDE / norm / span_year / STOP are copied VERBATIM from scripts/reddit/linker_rules_tier.py
(that module executes its gold-set run at import time, so it cannot be imported).  No threshold
is changed.  Pass-1 output is cached to --cache so pass 2/3 can be re-run cheaply.

Usage (from C:/dev/phd/casper):
    poetry run python scripts/reddit/tier_split_census.py
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from census_he import (  # noqa: E402  (same-dir helper reuse)
    TCONST_RE,
    align_mentions,
    iter_v1_rows,
    load_catalogue,
    parse_utt,
    thread_of,
    year_of,
)

ROOT = Path(__file__).resolve().parents[2]
V1_DIR = ROOT / "data" / "reddit_he" / "large_v1"
BASE = ROOT / "data" / "reddit_he"
BASICS = BASE / "IMDB-database" / "title.basics.tsv"
RATINGS = BASE / "IMDB-database" / "title.ratings.tsv"
OUT_JSON = BASE / "census" / "tier_split.json"
OUT_MD = ROOT / "docs" / "results" / "TIER_SPLIT_CENSUS.md"

TOKENS_PER_MENTION = 500  # audition observation, REDDIT_LINKER_AUDITION.md reading #4

csv.field_size_limit(1 << 31 - 1 if sys.maxsize > 2**32 else (1 << 30))

# ------------------------------------------------ VERBATIM from linker_rules_tier.py

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


TYPES = {"movie", "tvSeries", "tvMiniSeries", "tvMovie", "video", "short"}


def variants(n):
    qs = {n}
    if n.startswith("the "):
        qs.add(n[4:])
    else:
        qs.add("the " + n)
    return qs


def DECIDE(n, sy, he_tconst, cand):
    """linker_rules_tier.DECIDE, re-expressed over (normalized span, span year, He tconst).

    Identical thresholds and identical branch order; only the row-dict unpacking is inlined.
    """
    cs = []
    for q in variants(n):
        cs.extend(cand.get(q, ()))
    cs = sorted({c[0]: c for c in cs}.values(), key=lambda x: -x[4])
    if sy is not None:
        hits = [c for c in cs if c[3] not in ("\\N", "") and abs(int(c[3]) - sy) <= 1]
        if len(hits) == 1 and he_tconst == hits[0][0]:
            return "ACCEPT", cs
        return "DEFER", cs
    if len(n) < 4 or n in STOP or not cs:
        return "DEFER", cs
    top = cs[0]
    second_votes = cs[1][4] if len(cs) > 1 else 0
    if top[4] >= 2000 and top[4] >= 10 * max(second_votes, 1) and he_tconst == top[0]:
        return "ACCEPT", cs
    return "DEFER", cs


def defer_reason(n, sy, cs):
    """Classification only -- never feeds back into the decision.

    Precedence: an empty candidate list is reported as such even when the span is also short or
    stop-listed (in DECIDE's no-year branch those conditions short-circuit in the other order).
    """
    if not cs:
        return "empty_candidates"
    if sy is None and (len(n) < 4 or n in STOP):
        return "stoplist_or_short_span"
    return "ambiguous_candidates"


# ------------------------------------------------ pass 1: stream every V1 mention


def pass1(cache: Path) -> dict:
    in_cat, _cat = load_catalogue()

    # key -> count, key = (norm_span, span_year_or_-1, he_tconst, role, in_eval)
    keys: Counter = Counter()
    n_rows = n_turns = n_threads = 0
    n_tconst_mentions = 0        # every tconst in the `processed` CELL (census_he mentions_total)
    n_tconst_mentions_parsed = 0  # same over the PARSED utterance (catches escaped-\n cases)
    n_aligned = 0                # of those, the ones whose surface span is recoverable
    n_turns_no_year = 0
    eval_threads = 0
    eval_threads_p = 0
    cur = None

    def flush(t):
        nonlocal n_threads, eval_threads, eval_threads_p
        if t is None:
            return
        n_threads += 1
        gt = t["replier_cat"] - t["seeker_cat"]
        is_eval = len(gt) >= 5
        if is_eval:
            eval_threads += 1
        if len(t["replier_cat_p"] - t["seeker_cat_p"]) >= 5:
            eval_threads_p += 1
        for (nrm, sy, tc, role), c in t["buf"].items():
            keys[(nrm, sy, tc, role, is_eval)] += c

    for split in ["train", "valid", "test"]:
        path = V1_DIR / f"{split}.csv"
        if not path.exists():
            print(f"MISSING {path}", flush=True)
            continue
        for row in iter_v1_rows(path):
            n_rows += 1
            if n_rows % 200000 == 0:
                print(f"  {split}: {n_rows} rows, {n_aligned} aligned mentions, "
                      f"{len(keys)} distinct keys", flush=True)
            tid = thread_of(row["conv_id"])
            if cur is None or cur["tid"] != tid:
                flush(cur)
                cur = {"tid": tid, "turns": set(), "seeker_cat": set(), "replier_cat": set(),
                       "seeker_cat_p": set(), "replier_cat_p": set(), "buf": Counter()}
            turn = row["turn_id"]
            if turn in cur["turns"]:
                continue  # same turn re-used by another extracted path
            cur["turns"].add(turn)
            n_turns += 1

            is_seeker = str(row.get("is_seeker", "")).strip().lower() == "true"
            order = int(row.get("turn_order") or 0)
            role = "request" if (is_seeker and order == 0) else (
                "seeker_followup" if is_seeker else "reply")
            if year_of(row.get("utc_time")) is None:
                n_turns_no_year += 1

            proc_cell = row.get("processed") or ""
            # Eval-set membership is built from the UNPARSED cell exactly as census_he.census_v1
            # does, so the thread set here is bit-identical to the published 35,982 (the regex's
            # \b misses a tconst glued to an escaped "\n" in the cell; keeping the quirk keeps
            # parity, and `eval_threads_parsed_text` below measures what fixing it would add).
            tconsts_cell = TCONST_RE.findall(proc_cell)
            n_tconst_mentions += len(tconsts_cell)
            for tc in tconsts_cell:
                if tc in in_cat:
                    (cur["seeker_cat"] if is_seeker else cur["replier_cat"]).add(tc)

            proc = parse_utt(proc_cell)
            tconsts = TCONST_RE.findall(proc)
            if not tconsts:
                continue
            n_tconst_mentions_parsed += len(tconsts)
            for tc in tconsts:
                if tc in in_cat:
                    (cur["seeker_cat_p"] if is_seeker else cur["replier_cat_p"]).add(tc)
            raw = parse_utt(row.get("raw") or "")
            for tc, surface in align_mentions(proc, raw):
                surface = surface[:200]  # same field cap the gold sample used
                n_aligned += 1
                nrm = norm(surface)
                cur["buf"][(nrm, span_year(surface) or -1, tc, role)] += 1
        print(f"{split}: done ({n_rows} rows cumulative)", flush=True)
    flush(cur)

    state = {
        "rows": n_rows, "unique_turns": n_turns, "threads": n_threads,
        "turns_missing_year": n_turns_no_year,
        "tconst_mentions_total": n_tconst_mentions,
        "tconst_mentions_total_parsed_text": n_tconst_mentions_parsed,
        "aligned_mentions_total": n_aligned,
        "eval_threads": eval_threads,
        "eval_threads_parsed_text": eval_threads_p,
        "keys": [[k[0], k[1], k[2], k[3], int(k[4]), v] for k, v in keys.items()],
    }
    cache.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(cache, "wt", encoding="utf-8") as f:
        json.dump(state, f)
    print(f"pass1 cached -> {cache}", flush=True)
    return state


# ------------------------------------------------ pass 2: span-only candidate index


def build_candidates(all_norm_spans) -> dict:
    votes = {}
    with open(RATINGS, encoding="utf-8") as f:
        next(f)
        for line in f:
            p = line.rstrip("\n").split("\t")
            votes[p[0]] = int(p[2])
    print(f"ratings: {len(votes)} titles", flush=True)

    allq = set()
    for n in all_norm_spans:
        allq |= variants(n)
    cand = {q: [] for q in allq}
    print(f"queries: {len(allq)} (from {len(all_norm_spans)} distinct spans)", flush=True)

    seen = 0
    with open(BASICS, encoding="utf-8") as f:
        next(f)
        for line in f:
            seen += 1
            p = line.rstrip("\n").split("\t")
            if p[1] not in TYPES:
                continue
            np_, no_ = norm(p[2]), norm(p[3])
            hit = np_ if np_ in cand else (no_ if no_ in cand else None)
            if hit is not None:
                cand[hit].append([p[0], p[1], p[2], p[5], votes.get(p[0], 0)])
    nonempty = sum(1 for v in cand.values() if v)
    print(f"basics: {seen} rows scanned, {nonempty}/{len(cand)} queries matched", flush=True)
    return cand


# ------------------------------------------------ pass 3: decide + tabulate


def blank():
    return {
        "mentions": 0, "accept": 0, "defer": 0,
        "defer_empty_candidates": 0, "defer_stoplist_or_short_span": 0,
        "defer_ambiguous_candidates": 0,
        "defer_ambiguous_he_disagrees_with_confident_top": 0,
        "_spans_all": set(), "_spans_acc": set(), "_spans_def": set(),
    }


def finish(d):
    d = dict(d)
    d["distinct_spans"] = len(d.pop("_spans_all"))
    d["distinct_spans_accept"] = len(d.pop("_spans_acc"))
    d["distinct_spans_defer"] = len(d.pop("_spans_def"))
    d["accept_pct"] = round(100 * d["accept"] / max(d["mentions"], 1), 2)
    d["defer_pct"] = round(100 * d["defer"] / max(d["mentions"], 1), 2)
    return d


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=str(BASE / "census" / "tier_split_pass1_cache.json.gz"))
    ap.add_argument("--reuse-cache", action="store_true")
    args = ap.parse_args()
    cache = Path(args.cache)

    if args.reuse_cache and cache.exists():
        with gzip.open(cache, "rt", encoding="utf-8") as f:
            state = json.load(f)
        print(f"reusing pass1 cache {cache}", flush=True)
    else:
        state = pass1(cache)

    key_rows = state["keys"]
    all_spans = {k[0] for k in key_rows}
    cand = build_candidates(all_spans)

    buckets = defaultdict(blank)
    for nrm, sy, tc, role, in_eval, count in key_rows:
        sy = None if sy == -1 else sy
        dec, cs = DECIDE(nrm, sy, tc, cand)
        reason = defer_reason(nrm, sy, cs) if dec == "DEFER" else None
        confident_top = (
            reason == "ambiguous_candidates" and sy is None and cs
            and cs[0][4] >= 2000 and cs[0][4] >= 10 * max(cs[1][4] if len(cs) > 1 else 0, 1)
        )
        scopes = ["overall"] + (["eval"] if in_eval else [])
        for scope in scopes:
            for r in ("ALL", role):
                d = buckets[(scope, r)]
                d["mentions"] += count
                d["_spans_all"].add(nrm)
                if dec == "ACCEPT":
                    d["accept"] += count
                    d["_spans_acc"].add(nrm)
                else:
                    d["defer"] += count
                    d["_spans_def"].add(nrm)
                    d["defer_" + reason] += count
                    if confident_top:
                        d["defer_ambiguous_he_disagrees_with_confident_top"] += count

    res = {
        "protocol": {
            "source": [str(V1_DIR / f"{s}.csv") for s in ("train", "valid", "test")],
            "streamed": "every row of every split, no sampling, no cap (HARD RULE 1)",
            "thread_dedup_key": "submission id = conv_id minus trailing _<path>/<npaths>",
            "turn_dedup_key": "reddit turn_id within a thread",
            "mention_alignment": "census_he.align_mentions (word-diff of processed vs raw)",
            "rules_tier": "scripts/reddit/linker_rules_tier.py DECIDE, thresholds unchanged",
            "candidate_index": "title.basics.tsv primary+original titles (types "
                               + "/".join(sorted(TYPES)) + "), votes from title.ratings.tsv",
            "eval_scope": "threads with >=5 in-catalogue reply items (OP-named removed) "
                          "-- the 35,982-thread set of REDDIT_HE_CENSUS.md",
            "roles": "request = seeker turn_order 0; seeker_followup = later seeker turn; "
                     "reply = non-seeker turn",
        },
        "corpus": {k: state[k] for k in (
            "rows", "unique_turns", "threads", "turns_missing_year",
            "tconst_mentions_total", "tconst_mentions_total_parsed_text",
            "aligned_mentions_total", "eval_threads", "eval_threads_parsed_text")},
        "distinct_normalized_spans": len(all_spans),
        "distinct_keys": len(key_rows),
        "splits": {f"{scope}|{role}": finish(d) for (scope, role), d in sorted(buckets.items())},
    }
    ev = res["splits"]["eval|ALL"]
    ov = res["splits"]["overall|ALL"]
    res["cost"] = {
        "tokens_per_mention_assumption": TOKENS_PER_MENTION,
        "assumption_source": "REDDIT_LINKER_AUDITION.md reading #4 (~400-550 tok/mention "
                             "on the LLM tier under production batching); 500 = midpoint",
        "defer_mentions_eval_scope": ev["defer"],
        "estimated_opus_tokens_eval_scope": ev["defer"] * TOKENS_PER_MENTION,
        "defer_mentions_overall": ov["defer"],
        "estimated_opus_tokens_overall": ov["defer"] * TOKENS_PER_MENTION,
    }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(res, indent=2))
    write_md(res)
    print(json.dumps({"corpus": res["corpus"], "splits": res["splits"], "cost": res["cost"]},
                     indent=2))
    print(f"\nwrote {OUT_JSON}\nwrote {OUT_MD}")
    return 0


def write_md(res: dict) -> None:
    c, cost = res["corpus"], res["cost"]
    s = res["splits"]

    def row(label, d):
        return (f"| {label} | {d['mentions']:,} | {d['accept']:,} ({d['accept_pct']}%) | "
                f"{d['defer']:,} ({d['defer_pct']}%) | {d['distinct_spans']:,} | "
                f"{d['distinct_spans_accept']:,} | {d['distinct_spans_defer']:,} |")

    def dtab(label, d):
        n = max(d["defer"], 1)
        return (f"| {label} | {d['defer']:,} | "
                f"{d['defer_empty_candidates']:,} ({100*d['defer_empty_candidates']/n:.1f}%) | "
                f"{d['defer_ambiguous_candidates']:,} ({100*d['defer_ambiguous_candidates']/n:.1f}%) | "
                f"{d['defer_stoplist_or_short_span']:,} ({100*d['defer_stoplist_or_short_span']/n:.1f}%) |")

    order = [("overall|ALL", "ALL mentions"), ("overall|reply", "- reply"),
             ("overall|request", "- request"), ("overall|seeker_followup", "- seeker_followup")]
    order_e = [("eval|ALL", "ALL mentions"), ("eval|reply", "- reply"),
               ("eval|request", "- request"), ("eval|seeker_followup", "- seeker_followup")]

    md = f"""# Volume-weighted rules-tier split over the full He corpus (2026-08-03)

**Question:** what fraction of the corpus's mentions does the deterministic rules tier
(`scripts/reddit/linker_rules_tier.py`) auto-accept, and how many DEFER to the Opus tier?
The audition (`REDDIT_LINKER_AUDITION.md`) measured 48.6% ACCEPT coverage on the 500-mention
gold set, but that set is stratified evenly over (year x turn role) and so massively
over-samples request/seeker_followup turns. This census is volume-weighted: every mention in
the corpus, no sampling, no cap.

**Method.** One streaming pass over `large_v1/{{train,valid,test}}.csv` (1.83 GB + 346 MB +
187 MB), threads deduped by submission id and turns by reddit `turn_id` exactly as
`census_he.py`; each mention's surface span recovered by `align_mentions` (word-diff of
`processed` against `raw`); a second streaming pass over `title.basics.tsv` keyed by all
{res['distinct_normalized_spans']:,} distinct normalized spans builds the span-only candidate
index; `DECIDE` is then applied per distinct (span, span-year, He-tconst, role) key and
weighted by its count. Thresholds untouched.

## Corpus streamed

| quantity | count |
|---|---|
| CSV rows (path-expanded) | {c['rows']:,} |
| unique turns | {c['unique_turns']:,} |
| threads | {c['threads']:,} |
| threads in eval scope (>=5 in-catalogue reply items) | {c['eval_threads']:,} |
| tconst mentions in `processed` (census-parity count) | {c['tconst_mentions_total']:,} |
| tconst mentions in the parsed utterance text | {c['tconst_mentions_total_parsed_text']:,} |
| of those, span-alignable (adjudicable) | {c['aligned_mentions_total']:,} |

### Controls (all three passed before any number below was read)

1. **Corpus snap.** Rows / unique turns / threads / eval threads reproduce `REDDIT_HE_CENSUS.md`
   exactly: 1,669,720 / 971,185 / 58,623 / 35,982.
2. **Alignment snap.** {c['aligned_mentions_total']:,} alignable mentions reproduces the gold
   sampler's own population exactly (`census/gold_sample_summary.json`,
   `population_mentions_alignable`).
3. **Rules-tier snap.** The `DECIDE` copied here, re-run on `gold_combined_500.csv`, returns
   243/500 ACCEPT = 0.486 coverage at 0.971 precision -- the audition's published row to three
   decimals.

### A regex quirk worth recording

The two mention counts differ by {c['tconst_mentions_total_parsed_text'] - c['tconst_mentions_total']:,}
({100*(c['tconst_mentions_total_parsed_text'] - c['tconst_mentions_total'])/max(c['tconst_mentions_total_parsed_text'],1):.0f}%).
`REDDIT_HE_CENSUS.md` applies `\\btt\\d{{7,8}}\\b` to the raw CSV cell, where the list-style
replies that dominate this corpus write one title per line -- in the cell that newline is the
two literal characters `\\` `n`, so `\\ntt0177789` glues a word character onto `tt` and the `\\b`
fails. Applied to the parsed utterance the same regex finds those. The census's mention totals
(and every percentage derived from them) are therefore conservative by roughly that margin;
eval-thread membership here is deliberately kept on the census-parity count so the thread set is
bit-identical to the published 35,982, but on the parsed text it would be
{c['eval_threads_parsed_text']:,} threads.

Only span-alignable mentions can be adjudicated at all -- a mention whose surface string cannot
be recovered from the `processed`/`raw` diff has nothing to show a model. All tier numbers below
are over the {c['aligned_mentions_total']:,} alignable mentions.

## Tier split -- OVERALL (whole corpus)

| scope | mentions | ACCEPT | DEFER | distinct spans | spans w/ any ACCEPT | spans w/ any DEFER |
|---|---|---|---|---|---|---|
""" + "\n".join(row(l, s[k]) for k, l in order) + f"""

## Tier split -- EVAL SCOPE ({c['eval_threads']:,} threads with >=5 in-catalogue reply items)

| scope | mentions | ACCEPT | DEFER | distinct spans | spans w/ any ACCEPT | spans w/ any DEFER |
|---|---|---|---|---|---|---|
""" + "\n".join(row(l, s[k]) for k, l in order_e) + f"""

## Why each DEFER happened

Classification only (never feeds back into the decision). Precedence: empty candidate list
first, then stop-listed/short span, then ambiguous.

| scope | DEFER | empty candidates | ambiguous candidates | stoplist / span <4 chars |
|---|---|---|---|---|
""" + "\n".join(dtab(l, s[k]) for k, l in order) + "\n" + "\n".join(
        dtab("EVAL " + l, s[k]) for k, l in order_e) + f"""

Of the eval-scope ambiguous DEFERs, {s['eval|ALL']['defer_ambiguous_he_disagrees_with_confident_top']:,}
have a candidate that clears the confidence bar (>=2000 votes, >=10x runner-up) but is NOT the
tconst He's release assigned -- i.e. the rules tier and He actively disagree there, which is
precisely the population the audition showed He getting wrong (57.2% precision as shipped).

## Cost arithmetic for the Opus adjudication run

The audition observed ~400-550 tokens per mention on the LLM tier under production batching
(`REDDIT_LINKER_AUDITION.md`, reading #4); **500 tokens/mention** is the midpoint assumption used
here. It covers the prompt share (policy + sentence + candidate list) plus the answer, amortised
over batches of ~500 mentions per agent.

| scope | DEFER mentions | x 500 tok/mention | = estimated Opus tokens |
|---|---|---|---|
| eval scope (the run of record) | {cost['defer_mentions_eval_scope']:,} | 500 | **{cost['estimated_opus_tokens_eval_scope']:,}** ({cost['estimated_opus_tokens_eval_scope']/1e6:.1f}M) |
| whole corpus | {cost['defer_mentions_overall']:,} | 500 | {cost['estimated_opus_tokens_overall']:,} ({cost['estimated_opus_tokens_overall']/1e6:.1f}M) |

At the audition's observed band the eval-scope figure ranges
{cost['defer_mentions_eval_scope']*400/1e6:.1f}M-{cost['defer_mentions_eval_scope']*550/1e6:.1f}M tokens
(400-550 tok/mention).

## Caveats

- The tier split is computed from the span, its parenthetical year, and He's tconst only -- no
  context, exactly as the deployed rules tier works. It is a *coverage* number, not an accuracy
  number; accuracy on the ACCEPT tier was 0.971 on gold.
- ACCEPT requires agreement with He's link, so ACCEPT volume is bounded by He's own precision.
  The tier cannot accept a mention He linked wrongly -- those land in DEFER by construction.
- **{c['tconst_mentions_total_parsed_text'] - c['aligned_mentions_total']:,} mentions
  ({100*(c['tconst_mentions_total_parsed_text'] - c['aligned_mentions_total'])/max(c['tconst_mentions_total_parsed_text'],1):.0f}%
  of the parsed-text total) are not span-alignable** and are excluded from both tiers. This is
  the same phenomenon as the regex quirk above: `align_mentions` attributes a surface span only
  when a diff block contains exactly one tconst, and the list-style reply ("title\\ntconst\\ntitle
  \\ntconst...") collapses into one block holding many. Recovering them needs a line-aware
  alignment, not a bigger LLM budget -- and it would enlarge the DEFER tier, so the token
  estimate below is a floor for full coverage, not a ceiling.

Raw counts: `data/reddit_he/census/tier_split.json`. Script: `scripts/reddit/tier_split_census.py`.
"""
    OUT_MD.write_text(md, encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())
