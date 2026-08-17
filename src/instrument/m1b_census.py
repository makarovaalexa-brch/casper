"""M1b step 1 — the REALISTIC opener census over the banked Reddit slot corpus.

Pure groupby. No model, no network, no LLM, nothing written outside experiments/battery.
Design sheet: docs/design/DS_M1_OPENER_FAMILIES.md

Input : data/reddit_he/slots/full_out_*.jsonl  (4,431 records, one per ruler thread)
Output: experiments/battery/m1b_census.json
"""
from __future__ import annotations

import json
import os
from collections import Counter, defaultdict

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SLOTS = os.path.join(REPO, "data", "reddit_he", "slots")
OUT = os.path.join(REPO, "experiments", "battery", "m1b_census.json")

# Declared in the design sheet BEFORE running. The opener is the highest-priority slot present.
PRIORITY = ["anchors_pos", "genres_pos", "tone", "era", "constraints"]
# Every slot we report a marginal for (priority ordering is a modelling choice; the marginal is not).
ALL_SLOTS = ["anchors_pos", "anchors_neg", "genres_pos", "genres_neg",
             "tone", "era", "constraints", "context", "other"]


def _present(slots: dict, name: str) -> bool:
    """A slot counts as present iff it carries at least one non-empty value."""
    v = slots.get(name)
    if v is None:
        return False
    if isinstance(v, list):
        return len([x for x in v if str(x).strip()]) > 0
    return bool(str(v).strip())


def _values(slots: dict, name: str) -> list:
    v = slots.get(name)
    if v is None:
        return []
    if isinstance(v, list):
        return [str(x).strip() for x in v if str(x).strip()]
    return [str(v).strip()] if str(v).strip() else []


def main() -> None:
    files = sorted(f for f in os.listdir(SLOTS) if f.startswith("full_out") and f.endswith(".jsonl"))
    if not files:
        raise SystemExit(f"no slot files under {SLOTS}")

    n_rows = 0
    n_requests = 0
    marginal = Counter()                      # slot -> threads carrying it at all
    opener_type = Counter()                   # slot -> threads where it IS the opener
    opener_value = defaultdict(Counter)       # slot -> value -> count (as opener)
    any_value = defaultdict(Counter)          # slot -> value -> count (anywhere)
    cooc = defaultdict(Counter)               # slot -> slot -> co-occurrence
    n_slots_hist = Counter()
    no_opener = 0
    thread_ids = set()

    for fn in files:
        with open(os.path.join(SLOTS, fn), encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                n_rows += 1
                tid = rec.get("thread_id")
                if tid:
                    thread_ids.add(tid)
                if rec.get("is_request"):
                    n_requests += 1
                slots = rec.get("slots") or {}
                n_slots_hist[int(rec.get("n_slots") or 0)] += 1

                present = [s for s in ALL_SLOTS if _present(slots, s)]
                for s in present:
                    marginal[s] += 1
                    for v in _values(slots, s):
                        any_value[s][v] += 1
                for i, a in enumerate(present):
                    for b in present[i + 1:]:
                        cooc[a][b] += 1
                        cooc[b][a] += 1

                op = next((s for s in PRIORITY if s in present), None)
                if op is None:
                    no_opener += 1
                else:
                    opener_type[op] += 1
                    for v in _values(slots, op):
                        opener_value[op][v] += 1

    denom = max(n_rows, 1)
    out = {
        "analysis": "m1b_realistic_opener_census",
        "design_sheet": "docs/design/DS_M1_OPENER_FAMILIES.md",
        "source": "data/reddit_he/slots/full_out_*.jsonl",
        "n_files": len(files),
        "n_rows": n_rows,
        "n_distinct_thread_ids": len(thread_ids),
        "n_is_request": n_requests,
        "priority_order_declared": PRIORITY,
        "n_slots_per_thread_hist": dict(sorted(n_slots_hist.items())),
        "threads_with_no_priority_slot": no_opener,
        "marginal_slot_presence": {
            s: {"threads": marginal.get(s, 0), "share": round(marginal.get(s, 0) / denom, 5)}
            for s in ALL_SLOTS
        },
        "opener_type_distribution": {
            s: {"threads": opener_type.get(s, 0), "share": round(opener_type.get(s, 0) / denom, 5)}
            for s in PRIORITY
        },
        "opener_values_top50": {
            s: opener_value[s].most_common(50) for s in PRIORITY if opener_value[s]
        },
        "any_values_top50": {
            s: any_value[s].most_common(50) for s in ALL_SLOTS if any_value[s]
        },
        "distinct_values_per_slot": {s: len(any_value[s]) for s in ALL_SLOTS if any_value[s]},
        "cooccurrence": {a: dict(c) for a, c in sorted(cooc.items())},
    }

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1, ensure_ascii=False)

    print(f"[m1b] {n_rows} rows from {len(files)} files; {len(thread_ids)} distinct threads; "
          f"{n_requests} flagged is_request")
    print(f"[m1b] threads with no priority slot: {no_opener}")
    print("[m1b] marginal slot presence:")
    for s in ALL_SLOTS:
        c = marginal.get(s, 0)
        print(f"        {s:<14} {c:>5}  {c/denom:6.2%}")
    print("[m1b] OPENER distribution (highest-priority slot present):")
    for s in PRIORITY:
        c = opener_type.get(s, 0)
        print(f"        {s:<14} {c:>5}  {c/denom:6.2%}")
    print(f"[m1b] wrote {OUT}")


if __name__ == "__main__":
    main()
