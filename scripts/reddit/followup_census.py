"""Interview-structure census of the He et al. Reddit-Movie corpus (no LLM, no Reddit API).

Question: how much ORGANIC interviewing happens inside a request thread?  Three signals,
all measured on He et al.'s own large-V1 processed release:

  1. QUESTIONS TO OP  -- non-seeker turns that ask the seeker something (regex classes:
     seen-X / preference-probe / constraint-probe / other-question).
  2. OP ANSWERS       -- seeker turns whose PARENT is a non-seeker turn.  The V1 CSV does
     carry parent linkage: `context_turn_ids` is the ancestor chain of the path, so the
     parent of a turn is the LAST element of that list.  No proxy is needed.
     Per-thread count of these = the effective organic interview length k.
  3. ACKNOWLEDGMENTS  -- seeker turns matching thanks/will-watch/watched-it patterns, and
     the stronger "acknowledged recommendation": an acknowledging seeker turn that names a
     tconst which some non-seeker turn of the same thread also named.

HARD RULES honoured:
  * nothing sampled/capped/truncated in any COUNT -- all three V1 CSVs are streamed end to
    end and every unique turn of every thread is counted.  The only samples drawn are the
    2 x 30 example snippets for heuristic validation (explicitly requested), and those
    snippets are character-truncated for display only.
  * no LLM call, no Reddit API call.
  * thread = submission id (conv_id minus the trailing _<path>/<npaths>); turns deduped by
    reddit turn_id within a thread (the V1 CSV is path-expanded).
  * no usernames anywhere in any output; `u/handle` mentions inside snippets are redacted.

Scopes reported: ALL EVAL THREADS (the 35,982 census-parity threads with >=5 in-catalogue
replier items after removing seeker-named items) and the CLEAN SUBSET (thread ids appearing
in adjudication batches 0000-0099), plus a by-subreddit breakdown of both.

Usage (from C:/dev/phd/casper):
    poetry run python scripts/reddit/followup_census.py
"""

from __future__ import annotations

import ast
import csv
import gzip
import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
V1_DIR = ROOT / "data" / "reddit_he" / "large_v1"
RAW_DIR = ROOT / "data" / "reddit_he" / "Reddit-Movie-large" / "subreddit"
OUT_DIR = ROOT / "data" / "reddit_he" / "census"
BATCH_DIR = ROOT / "data" / "reddit_he" / "adjudication" / "batches"
DOC = ROOT / "docs" / "results" / "REDDIT_FOLLOWUP_CENSUS.md"
JSON_OUT = OUT_DIR / "followup_census.json"
SUBMAP_CACHE = OUT_DIR / "thread_subreddit.json.gz"
LINKS = ROOT / "data" / "movielens" / "links.csv"
CATALOGUE = ROOT / "data" / "ml-25m" / "proc" / "unique_sid.txt"

SUBREDDITS = ["bestofnetflix", "movies", "moviesuggestions", "netflixbestof", "truefilm"]
TCONST_RE = re.compile(r"\btt\d{7,8}\b")
SUBMISSION_ID_RE = re.compile(r'^\{"submission":\s*\{"(t3_[a-z0-9]+)"')
USER_RE = re.compile(r"(?<![\w/])/?u/[A-Za-z0-9_\-]+")

CLEAN_BATCHES = 100  # batches 0000-0099 are the completed adjudication subset
SEED = 98765
N_EXAMPLES = 30

csv.field_size_limit(1 << 31 - 1 if sys.maxsize > 2**32 else (1 << 30))

# ---------------------------------------------------------------- heuristics

# NOTE ON ORDER: a comment is assigned ONE primary class, first match wins in this order.
# Every class additionally requires the comment to contain a literal "?".
SEEN_X = re.compile(
    r"\b(?:have|has|had|did|do|does)\s+(?:you|u|ya)\s+(?:ever\s+|already\s+|guys\s+)*"
    r"(?:seen|watched|watch|see|checked\s+out|check\s+out|tried|read)\b"
    r"|\byou\s+(?:seen|watched)\b"
    r"|\bseen\s+(?:it|this|that|any\s+of\s+(?:these|them|those))\s*\?"
    r"|\bwatched\s+(?:it|this|that|any\s+of\s+(?:these|them|those))\s*\?",
    re.I,
)
PREF_PROBE = re.compile(
    r"\b(?:do|did|does|would|will|are|is|were|was|can|could)\s+(?:you|u|ya)\b[^.?!]{0,80}"
    r"\b(?:like|liked|love|loved|enjoy|enjoyed|want|prefer|mind|into|okay|ok|fine|open)\b"
    r"|\bare\s+you\s+(?:looking|after|hoping|open|okay|ok|into|in\s+the\s+mood)\b"
    r"|\bhow\s+do\s+you\s+feel\s+about\b"
    r"|\bwhat\s+(?:kind|sort|type)s?\s+of\s+(?:movie|film|thing|stuff|story|stories|vibe)"
    r"|\bwhat\s+are\s+you\s+(?:looking|after|in\s+the\s+mood)\b"
    r"|\bin\s+the\s+mood\s+for\b[^.?!]{0,60}\?"
    r"|\bwhat\s+(?:are|were)\s+(?:some\s+of\s+)?your\s+(?:favou?rite|favs?|top)\b",
    re.I,
)
CONSTRAINT_PROBE = re.compile(
    r"\bany\s+(?:dis)?likes\b"
    r"|\bany\s+(?:preferences?|genres?|restrictions?|limits?|constraints?|dealbreakers?)\b"
    r"|\bwhat\s+genre|\bwhich\s+genre|\bwhat\s+(?:kind|type|sort)s?\s+of\b"
    r"|\bsub(?:title|)s?\s+(?:ok|okay|fine|alright|a\s+problem|an\s+issue)\b"
    r"|\b(?:ok|okay|fine)\s+with\s+sub(?:title)?s\b"
    r"|\bhow\s+old\b|\bwhat\s+age\b|\bage\s+(?:range|group)\b"
    r"|\bwhat\s+have\s+you\s+(?:seen|watched|already)\b"
    r"|\bwhat\s+(?:streaming\s+)?(?:services?|platforms?|channels?)\b"
    r"|\bdo\s+you\s+have\s+(?:netflix|hulu|prime|amazon|disney|hbo|max|paramount|peacock)\b"
    r"|\bforeign\s+(?:films?|movies?)\s+(?:ok|okay|fine)\b"
    r"|\b(?:black\s+and\s+white|b&w|animated|animation|older\s+(?:films?|movies?))\s+(?:ok|okay|fine)\b"
    r"|\bhow\s+(?:many|long)\b[^.?!]{0,60}\?",
    re.I,
)
# generic interrogative: a wh-word or auxiliary somewhere before a question mark
OTHER_Q = re.compile(
    r"\b(?:what|which|who|whose|whom|when|where|why|how|do|does|did|are|is|am|was|were|"
    r"have|has|had|can|could|would|should|will|shall|any|anyone|anything)\b[^?]{0,200}\?",
    re.I,
)
QCLASSES = [
    ("seen_x", SEEN_X),
    ("preference_probe", PREF_PROBE),
    ("constraint_probe", CONSTRAINT_PROBE),
    ("other_question", OTHER_Q),
]

ACK = re.compile(
    r"\bthank(?:s|\s*you|\s*u)\b|\bthx\b|\bty\b(?!\w)"
    r"|\bwill\s+(?:definitely\s+|def\s+|for\s+sure\s+)?(?:check|watch|give|try|look)\b"
    r"|\b(?:i'?ll|ill|imma|gonna|going\s+to)\s+(?:definitely\s+|def\s+)?(?:check|watch|give|try|look)\b"
    r"|\badd(?:ed|ing)?\s+(?:it|this|that|them|these)?\s*to\s+(?:my\s+)?list\b"
    r"|\badded\s+to\s+(?:the\s+|my\s+)?list\b"
    r"|\bwatch\s*list\b|\bwatchlist\b|\bto\s+my\s+queue\b"
    r"|\b(?:i\s+)?(?:already\s+|just\s+)?(?:watched|seen)\s+(?:it|this|that|them)\b"
    r"|\bjust\s+watched\b"
    r"|\b(?:i\s+)?(?:loved|liked|enjoyed)\s+(?:it|this|that|them)\b"
    r"|\bsounds?\s+(?:good|great|perfect|interesting|fun|amazing|awesome|promising)\b"
    r"|\bgreat\s+(?:suggestion|rec|recommendation|pick|call|shout)\b"
    r"|\bgood\s+(?:suggestion|rec|recommendation|call|shout)\b"
    r"|\bexactly\s+what\s+i\s+(?:was\s+)?(?:want|need|look)",
    re.I,
)


def classify(text: str) -> str | None:
    """Primary question class of a comment, or None if it asks nothing detectable."""
    if "?" not in text:
        return None
    for name, rx in QCLASSES:
        if rx.search(text):
            return name
    return None


def redact(text: str) -> str:
    return USER_RE.sub("u/[USER]", text)


def snippet(text: str, n: int = 420) -> str:
    s = " ".join(redact(text).split())
    return s[:n] + ("…" if len(s) > n else "")


# ---------------------------------------------------------------- small utils


def parse_utt(cell: str) -> str:
    """The `processed`/`raw` cells are python-literal ['USER', '<text>'] pairs."""
    try:
        v = ast.literal_eval(cell)
        if isinstance(v, (list, tuple)) and len(v) >= 2:
            return str(v[1])
    except (ValueError, SyntaxError):
        pass
    return cell


def parse_ctx(cell: str) -> list[str]:
    try:
        v = ast.literal_eval(cell or "[]")
        if isinstance(v, (list, tuple)):
            return [str(x) for x in v]
    except (ValueError, SyntaxError):
        pass
    return []


def thread_of(conv_id: str) -> str:
    return conv_id.rsplit("_", 1)[0]


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
        "mean": round(sum(v) / n, 4),
        "median": q(0.50),
        "p75": q(0.75),
        "p90": q(0.90),
        "p99": q(0.99),
        "max": v[-1],
        "total": sum(v),
    }


def load_catalogue() -> dict[str, int]:
    cat = {int(x) for x in CATALOGUE.read_text().split()}
    t2m: dict[str, int] = {}
    with LINKS.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            imdb = (row["imdbId"] or "").strip()
            if not imdb:
                continue
            mid = int(row["movieId"])
            t2m[f"tt{int(imdb):07d}"] = mid
            t2m[f"tt{imdb}"] = mid
    return {t: m for t, m in t2m.items() if m in cat}


def load_clean_ids() -> set[str]:
    ids: set[str] = set()
    for b in range(CLEAN_BATCHES):
        p = BATCH_DIR / f"batch_{b:04d}.jsonl"
        if not p.exists():
            raise SystemExit(f"missing {p}")
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    ids.add(json.loads(line)["thread_id"])
    return ids


def load_thread_subreddit() -> dict[str, str]:
    """thread id -> subreddit, read off the raw dump (submission id is the line prefix)."""
    if SUBMAP_CACHE.exists():
        with gzip.open(SUBMAP_CACHE, "rt", encoding="utf-8") as f:
            return json.load(f)
    m: dict[str, str] = {}
    for sub in SUBREDDITS:
        path = RAW_DIR / f"{sub}.jsonl"
        n = 0
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                hit = SUBMISSION_ID_RE.match(line)
                if hit:
                    m[hit.group(1)] = sub
                    n += 1
                else:  # fall back to a full parse -- never skip a record silently
                    d = json.loads(line)
                    for sid in (d.get("submission") or {}):
                        m[sid] = sub
                        n += 1
        print(f"  subreddit map: {sub} -> {n} threads", flush=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with gzip.open(SUBMAP_CACHE, "wt", encoding="utf-8") as f:
        json.dump(m, f)
    return m


# ---------------------------------------------------------------- accumulator


class Acc:
    """All per-thread aggregates for one scope (all-eval / clean / a subreddit)."""

    def __init__(self) -> None:
        self.threads = 0
        self.q_per_thread: list[int] = []
        self.q_to_op_per_thread: list[int] = []
        self.strict_per_thread: list[int] = []
        self.threads_with_strict = 0
        self.strict_hist: Counter = Counter()
        self.strict_total = 0
        self.strict_answered = 0
        self.strict_to_op = 0
        self.strict_to_op_answered = 0
        self.opreply_per_thread: list[int] = []
        self.ack_per_thread: list[int] = []
        self.ackrec_per_thread: list[int] = []
        self.turns_per_thread: list[int] = []
        self.qclass: Counter = Counter()
        self.threads_with_q = 0
        self.threads_with_opreply = 0
        self.threads_with_ack = 0
        self.threads_with_ackrec = 0
        self.opreply_hist: Counter = Counter()
        self.q_hist: Counter = Counter()
        self.questions_total = 0
        self.questions_answered = 0
        self.questions_to_op_total = 0
        self.questions_to_op_answered = 0
        self.qmark_comments = 0  # non-OP comments containing "?"
        self.nonop_comments = 0
        self.op_turns = 0
        self.ackrec_items: Counter = Counter()  # per-thread distinct acknowledged items

    def add(self, t: dict) -> None:
        self.threads += 1
        self.turns_per_thread.append(t["n_turns"])
        self.q_per_thread.append(t["n_q"])
        self.q_to_op_per_thread.append(t["n_q_to_op"])
        self.strict_per_thread.append(t["n_strict"])
        self.threads_with_strict += t["n_strict"] > 0
        self.strict_hist[min(t["n_strict"], 5)] += 1
        self.strict_total += t["n_strict"]
        self.strict_answered += t["n_strict_answered"]
        self.strict_to_op += t["n_strict_to_op"]
        self.strict_to_op_answered += t["n_strict_to_op_answered"]
        self.opreply_per_thread.append(t["n_opreply"])
        self.ack_per_thread.append(t["n_ack"])
        self.ackrec_per_thread.append(t["n_ackrec"])
        self.qclass.update(t["qclass"])
        self.threads_with_q += t["n_q"] > 0
        self.threads_with_opreply += t["n_opreply"] > 0
        self.threads_with_ack += t["n_ack"] > 0
        self.threads_with_ackrec += t["n_ackrec"] > 0
        self.opreply_hist[min(t["n_opreply"], 5)] += 1
        self.q_hist[min(t["n_q"], 5)] += 1
        self.questions_total += t["n_q"]
        self.questions_answered += t["n_q_answered"]
        self.questions_to_op_total += t["n_q_to_op"]
        self.questions_to_op_answered += t["n_q_to_op_answered"]
        self.qmark_comments += t["n_qmark"]
        self.nonop_comments += t["n_nonop"]
        self.op_turns += t["n_op"]

    def report(self) -> dict:
        n = max(self.threads, 1)
        qt = max(self.questions_total, 1)
        return {
            "threads": self.threads,
            "turns_per_thread": quantiles(self.turns_per_thread),
            "nonop_comments": self.nonop_comments,
            "op_turns": self.op_turns,
            "questions_per_thread": quantiles(self.q_per_thread),
            "questions_to_op_per_thread": quantiles(self.q_to_op_per_thread),
            "strict_probes_per_thread": quantiles(self.strict_per_thread),
            "pct_threads_with_strict_probe": round(100 * self.threads_with_strict / n, 2),
            "strict_hist": {
                str(k) + ("+" if k == 5 else ""): self.strict_hist.get(k, 0) for k in range(6)
            },
            "strict_hist_pct": {
                str(k) + ("+" if k == 5 else ""): round(100 * self.strict_hist.get(k, 0) / n, 2)
                for k in range(6)
            },
            "strict_probes_total": self.strict_total,
            "strict_probes_answered": self.strict_answered,
            "strict_answered_rate_pct": round(
                100 * self.strict_answered / max(self.strict_total, 1), 2
            ),
            "strict_probes_to_op": self.strict_to_op,
            "strict_to_op_answered": self.strict_to_op_answered,
            "strict_to_op_answered_rate_pct": round(
                100 * self.strict_to_op_answered / max(self.strict_to_op, 1), 2
            ),
            "op_replies_per_thread": quantiles(self.opreply_per_thread),
            "acks_per_thread": quantiles(self.ack_per_thread),
            "acked_recs_per_thread": quantiles(self.ackrec_per_thread),
            "pct_threads_with_question": round(100 * self.threads_with_q / n, 2),
            "pct_threads_with_op_reply": round(100 * self.threads_with_opreply / n, 2),
            "pct_threads_with_ack": round(100 * self.threads_with_ack / n, 2),
            "pct_threads_with_acked_rec": round(100 * self.threads_with_ackrec / n, 2),
            "op_reply_hist": {
                str(k) + ("+" if k == 5 else ""): self.opreply_hist.get(k, 0) for k in range(6)
            },
            "op_reply_hist_pct": {
                str(k) + ("+" if k == 5 else ""): round(100 * self.opreply_hist.get(k, 0) / n, 2)
                for k in range(6)
            },
            "question_hist": {
                str(k) + ("+" if k == 5 else ""): self.q_hist.get(k, 0) for k in range(6)
            },
            "question_hist_pct": {
                str(k) + ("+" if k == 5 else ""): round(100 * self.q_hist.get(k, 0) / n, 2)
                for k in range(6)
            },
            "question_class_mix": dict(self.qclass),
            "question_class_mix_pct": {
                k: round(100 * v / qt, 2) for k, v in self.qclass.items()
            },
            "questions_total": self.questions_total,
            "questions_answered": self.questions_answered,
            "question_answered_rate_pct": round(100 * self.questions_answered / qt, 2),
            "questions_to_op_total": self.questions_to_op_total,
            "questions_to_op_answered": self.questions_to_op_answered,
            "questions_to_op_answered_rate_pct": round(
                100 * self.questions_to_op_answered / max(self.questions_to_op_total, 1), 2
            ),
            "nonop_comments_with_qmark": self.qmark_comments,
            "pct_qmark_comments_matched": round(
                100 * self.questions_total / max(self.qmark_comments, 1), 2
            ),
        }


# ---------------------------------------------------------------- main pass


def run() -> dict:
    in_cat = load_catalogue()
    print("catalogue tconsts:", len(in_cat), flush=True)
    clean_ids = load_clean_ids()
    print("clean subset thread ids (batches 0000-0099):", len(clean_ids), flush=True)
    submap = load_thread_subreddit()
    print("thread->subreddit entries:", len(submap), flush=True)

    rng = random.Random(SEED)
    ex_match: list[dict] = []
    ex_nonmatch: list[dict] = []
    ex_strict: list[dict] = []
    n_seen_match = 0
    n_seen_nonmatch = 0
    n_seen_strict = 0

    scopes = {"all_eval": Acc(), "clean": Acc()}
    by_sub_all: dict[str, Acc] = defaultdict(Acc)
    by_sub_clean: dict[str, Acc] = defaultdict(Acc)

    totals = {
        "rows": 0,
        "unique_turns": 0,
        "threads": 0,
        "eval_threads": 0,
        "clean_threads_found": 0,
        "clean_threads_eval": 0,
        "clean_ids_not_in_v1": 0,
        "threads_reopened_out_of_order": 0,
        "turns_with_no_parent_nonroot": 0,
    }
    thread_seen: set[str] = set()
    clean_hit: set[str] = set()

    cur = None

    def flush(t):
        nonlocal ex_match, ex_nonmatch, ex_strict
        nonlocal n_seen_match, n_seen_nonmatch, n_seen_strict
        if t is None:
            return
        totals["threads"] += 1
        tid = t["tid"]
        # eval-thread filter, identical to census_he.census_v1
        gt = t["replier_items_cat"] - t["seeker_items_cat"]
        is_eval = len(gt) >= 5
        is_clean = tid in clean_ids
        if is_clean:
            clean_hit.add(tid)
            totals["clean_threads_found"] += 1
        if not is_eval:
            return
        totals["eval_threads"] += 1

        turns = t["turns"]  # turn_id -> dict
        op_parents = {v["parent"] for v in turns.values() if v["is_seeker"] and v["parent"]}
        reply_tconsts = t["replier_tconsts"]

        n_q = n_q_to_op = n_q_ans = n_q_to_op_ans = 0
        n_strict = n_strict_ans = n_strict_to_op = n_strict_to_op_ans = 0
        n_qmark = n_nonop = n_op = 0
        n_opreply = n_ack = n_ackrec = 0
        qclass: Counter = Counter()

        for tid_turn, v in turns.items():
            text = v["text"]
            if v["is_seeker"]:
                n_op += 1
                parent = v["parent"]
                if parent and not turns.get(parent, {}).get("is_seeker", True):
                    n_opreply += 1
                if ACK.search(text):
                    n_ack += 1
                    if v["tconsts"] & reply_tconsts:
                        n_ackrec += 1
            else:
                n_nonop += 1
                has_q = "?" in text
                n_qmark += has_q
                cls = classify(text)
                answered = tid_turn in op_parents
                # "addressed at OP" = this comment's parent is the submission itself or an
                # OP turn, so the OP is the natural addressee
                p = v["parent"]
                to_op = (p == t["tid"]) or (p in turns and turns[p]["is_seeker"])
                if cls:
                    n_q += 1
                    qclass[cls] += 1
                    n_q_ans += answered
                    if to_op:
                        n_q_to_op += 1
                        n_q_to_op_ans += answered
                    if cls != "other_question":
                        n_strict += 1
                        n_strict_ans += answered
                        if to_op:
                            n_strict_to_op += 1
                            n_strict_to_op_ans += answered
                # --- example reservoirs (validation appendix only)
                if has_q:
                    if cls:
                        n_seen_match += 1
                        item = {"class": cls, "answered": bool(answered), "text": snippet(text)}
                        if len(ex_match) < N_EXAMPLES:
                            ex_match.append(item)
                        elif rng.randrange(n_seen_match) < N_EXAMPLES:
                            ex_match[rng.randrange(N_EXAMPLES)] = item
                        if cls != "other_question":
                            n_seen_strict += 1
                            if len(ex_strict) < N_EXAMPLES:
                                ex_strict.append(item)
                            elif rng.randrange(n_seen_strict) < N_EXAMPLES:
                                ex_strict[rng.randrange(N_EXAMPLES)] = item
                    else:
                        n_seen_nonmatch += 1
                        item = {"text": snippet(text)}
                        if len(ex_nonmatch) < N_EXAMPLES:
                            ex_nonmatch.append(item)
                        elif rng.randrange(n_seen_nonmatch) < N_EXAMPLES:
                            ex_nonmatch[rng.randrange(N_EXAMPLES)] = item

        rec = {
            "n_turns": len(turns),
            "n_q": n_q,
            "n_q_to_op": n_q_to_op,
            "n_q_answered": n_q_ans,
            "n_q_to_op_answered": n_q_to_op_ans,
            "n_strict": n_strict,
            "n_strict_answered": n_strict_ans,
            "n_strict_to_op": n_strict_to_op,
            "n_strict_to_op_answered": n_strict_to_op_ans,
            "n_qmark": n_qmark,
            "n_nonop": n_nonop,
            "n_op": n_op,
            "n_opreply": n_opreply,
            "n_ack": n_ack,
            "n_ackrec": n_ackrec,
            "qclass": qclass,
        }
        scopes["all_eval"].add(rec)
        sub = submap.get(tid, "unknown")
        by_sub_all[sub].add(rec)
        if is_clean:
            totals["clean_threads_eval"] += 1
            scopes["clean"].add(rec)
            by_sub_clean[sub].add(rec)

    for split in ["train", "valid", "test"]:
        path = V1_DIR / f"{split}.csv"
        if not path.exists():
            raise SystemExit(f"missing {path}")
        n_rows = 0
        with path.open("r", encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                n_rows += 1
                totals["rows"] += 1
                tid = thread_of(row["conv_id"])
                if cur is None or cur["tid"] != tid:
                    flush(cur)
                    if tid in thread_seen:
                        totals["threads_reopened_out_of_order"] += 1
                    thread_seen.add(tid)
                    cur = {
                        "tid": tid,
                        "turns": {},
                        "seeker_items_cat": set(),
                        "replier_items_cat": set(),
                        "replier_tconsts": set(),
                    }
                turn_id = row["turn_id"]
                if turn_id in cur["turns"]:
                    continue
                totals["unique_turns"] += 1
                is_seeker = str(row.get("is_seeker", "")).strip().lower() == "true"
                ctx = parse_ctx(row.get("context_turn_ids") or "")
                parent = ctx[-1] if ctx else None
                order = int(row.get("turn_order") or 0)
                if parent is None and order > 0:
                    totals["turns_with_no_parent_nonroot"] += 1
                tcs = set(TCONST_RE.findall(row.get("processed") or ""))
                cur["turns"][turn_id] = {
                    "is_seeker": is_seeker,
                    "parent": parent,
                    "order": order,
                    "text": parse_utt(row.get("raw") or ""),
                    "tconsts": tcs,
                }
                for tc in tcs:
                    if is_seeker:
                        if tc in in_cat:
                            cur["seeker_items_cat"].add(tc)
                    else:
                        cur["replier_tconsts"].add(tc)
                        if tc in in_cat:
                            cur["replier_items_cat"].add(tc)
        print(f"{split}: {n_rows} rows", flush=True)
    flush(cur)

    totals["clean_ids_not_in_v1"] = len(clean_ids - clean_hit)

    res = {
        "protocol": {
            "source": [str(V1_DIR / f"{s}.csv") for s in ["train", "valid", "test"]],
            "thread_key": "conv_id minus trailing _<path>/<npaths>",
            "turn_dedup": "reddit turn_id within a thread",
            "parent_linkage": "REAL -- parent = last element of the V1 `context_turn_ids` "
            "ancestor chain; no turn_order proxy was needed",
            "eval_filter": ">=5 distinct in-catalogue replier tconsts after removing every "
            "in-catalogue tconst the seeker named (identical to census_he.census_v1)",
            "clean_subset": f"thread ids in adjudication batches 0000-{CLEAN_BATCHES - 1:04d}",
            "text_field": "`raw` (original surface text) for question/ack matching; "
            "`processed` for tconsts",
            "sampling": "none in any count; only the 2x30 validation snippets are sampled "
            f"(reservoir, seed {SEED}) and character-truncated for display",
        },
        "totals": totals,
        "reconciliation": {
            "census_v1_threads": 58623,
            "ours_threads": totals["threads"],
            "census_eval_threads": 35982,
            "ours_eval_threads": totals["eval_threads"],
            "manifest_clean_expected": 5441,
            "ours_clean_threads_found": totals["clean_threads_found"],
            "ours_clean_threads_eval": totals["clean_threads_eval"],
            "census_rows": 1669720,
            "ours_rows": totals["rows"],
            "census_unique_turns": 971185,
            "ours_unique_turns": totals["unique_turns"],
        },
        "all_eval": scopes["all_eval"].report(),
        "clean": scopes["clean"].report(),
        "by_subreddit_all_eval": {k: v.report() for k, v in sorted(by_sub_all.items())},
        "by_subreddit_clean": {k: v.report() for k, v in sorted(by_sub_clean.items())},
        "examples_matched": ex_match,
        "examples_qmark_nonmatched": ex_nonmatch,
        "examples_strict": ex_strict,
        "example_populations": {
            "matched_question_comments": n_seen_match,
            "qmark_nonmatched_comments": n_seen_nonmatch,
            "strict_probe_comments": n_seen_strict,
        },
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    JSON_OUT.write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(f"\nwrote {JSON_OUT}")
    print(json.dumps(res["reconciliation"], indent=2))
    return res


if __name__ == "__main__":
    run()
