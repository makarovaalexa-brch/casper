"""M1b vocabulary, FULL TAIL -- replaces the top-50 truncation.

DEFECT BEING FIXED (found by the author, 2026-08-17): `m1b_map.py` computed its mapping over each
slot's census TOP-50 value list -- a limitation it recorded in its own artifact ("computed over the
census top-50 value list for this slot, not the full tail") -- and the resulting 34 targets were then
used as if they were THE Reddit vocabulary. The corpus actually carries 1,170 distinct genre strings
and 1,237 distinct tone strings. `explosions` was absent because it sits in the tail, not because
nobody says it. This file maps EVERY value.

Input : data/reddit_he/slots/full_out_*.jsonl  ·  data/movielens/genome-tags.csv
Output: experiments/battery/m1b_vocab_full.json
"""
from __future__ import annotations

import csv
import json
import os
import re
from collections import Counter

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
SLOTS = os.path.join(_ROOT, "data", "reddit_he", "slots")
TAGS = os.path.join(_ROOT, "data", "movielens", "genome-tags.csv")
OUT = os.path.join(_ROOT, "experiments", "battery", "m1b_vocab_full.json")

NOUNS = r"(movies|movie|films|film|flicks|flick|shows|show|series|genre|stuff|things|vibes|vibe)"


def norm(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"[‘’“”]", "'", s)
    s = re.sub(r"[^a-z0-9\s'-]", " ", s)
    s = re.sub(r"\b" + NOUNS + r"\b", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    s = re.sub(r"^(a|an|the|some|any|good|great|best|really|very|super|kinda|sorta)\s+", "", s)
    return s.strip(" -'")


def main() -> None:
    tag_lookup = {}
    with open(TAGS, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            t = r["tag"].strip()
            tag_lookup[t.lower()] = t
            tag_lookup[t.lower().replace("-", " ")] = t

    raw = Counter()          # every value seen, over ALL slots that carry preference words
    per_slot = {}
    for fn in sorted(f for f in os.listdir(SLOTS) if f.startswith("full_out") and f.endswith(".jsonl")):
        with open(os.path.join(SLOTS, fn), encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                sl = (json.loads(line).get("slots") or {})
                for slot in ("genres_pos", "tone", "other"):
                    v = sl.get(slot)
                    vals = v if isinstance(v, list) else ([v] if v else [])
                    for x in vals:
                        s = str(x).strip()
                        if s:
                            raw[s] += 1
                            per_slot.setdefault(slot, Counter())[s] += 1

    mapped, unmapped = Counter(), Counter()
    surface = {}
    for value, cnt in raw.items():
        n = norm(value)
        hit = None
        for cand in (n, n.replace(" ", "-"), n.replace("-", " ")):
            if cand in tag_lookup:
                hit = tag_lookup[cand]
                break
        if hit is None:                       # single-word head of a short phrase
            w = n.split()
            if 1 < len(w) <= 3:
                for x in w:
                    if x in tag_lookup:
                        hit = tag_lookup[x]
                        break
        if hit is None:
            unmapped[value] += cnt
        else:
            mapped[hit] += cnt
            surface.setdefault(hit, set()).add(value)

    tot = sum(mapped.values()) + sum(unmapped.values())
    out = {
        "analysis": "m1b_reddit_vocabulary_full_tail",
        "supersedes": "the 34-target vocabulary derived from m1b_mapping.json's top-50 lists",
        "slots_used": ["genres_pos", "tone", "other"],
        "distinct_raw_values": len(raw),
        "occurrences_total": tot,
        "mapped_occurrences": sum(mapped.values()),
        "mapped_share": round(sum(mapped.values()) / max(tot, 1), 4),
        "distinct_tags_matched": len(mapped),
        "tags": sorted(mapped.keys()),
        "tag_counts": mapped.most_common(),
        "unmapped_top100": unmapped.most_common(100),
        "per_slot_distinct": {k: len(v) for k, v in per_slot.items()},
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1, ensure_ascii=False)

    print(f"[vocab-full] distinct raw values {len(raw)} over {tot} occurrences")
    print(f"[vocab-full] mapped to {len(mapped)} distinct genome tags "
          f"({out['mapped_share']:.1%} of occurrences)")
    print(f"[vocab-full] top tags: {[t for t, _ in mapped.most_common(25)]}")
    for probe in ("explosions", "action", "romance", "dark", "scary", "funny", "epic"):
        print(f"    probe {probe!r:14} in vocabulary: {any(t.lower()==probe for t in mapped)}")
    print(f"[vocab-full] wrote {OUT}")


if __name__ == "__main__":
    main()
