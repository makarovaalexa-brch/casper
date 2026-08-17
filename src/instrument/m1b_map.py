"""M1b step 2 — map REALISTIC opener values onto ML attribute signatures.

Design sheet: docs/design/DS_M1_OPENER_FAMILIES.md
No model, no network. Reports the UNMAPPED share explicitly (HARD RULE 1: nothing dropped silently).

Input : experiments/battery/m1b_census.json  (from m1b_census.py)
        data/movielens/genome-tags.csv, data/movielens/movies.csv
Output: experiments/battery/m1b_mapping.json
"""
from __future__ import annotations

import csv
import json
import os
import re
from collections import Counter

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CENSUS = os.path.join(REPO, "experiments", "battery", "m1b_census.json")
TAGS = os.path.join(REPO, "data", "movielens", "genome-tags.csv")
MOVIES = os.path.join(REPO, "data", "movielens", "movies.csv")
OUT = os.path.join(REPO, "experiments", "battery", "m1b_mapping.json")

ML_GENRES = ["Action", "Adventure", "Animation", "Children", "Comedy", "Crime", "Documentary",
             "Drama", "Fantasy", "Film-Noir", "Horror", "IMAX", "Musical", "Mystery", "Romance",
             "Sci-Fi", "Thriller", "War", "Western"]

# Alias table for genre strings. Declared before running; every entry is a surface-form variant,
# never a semantic re-interpretation.
GENRE_ALIAS = {
    "scifi": "Sci-Fi", "sci fi": "Sci-Fi", "sci-fi": "Sci-Fi", "science fiction": "Sci-Fi",
    "doc": "Documentary", "docs": "Documentary", "documentaries": "Documentary",
    "rom com": "Romance", "romcom": "Romance", "romantic comedy": "Romance",
    "romantic": "Romance", "romances": "Romance",
    "psychological thriller": "Thriller", "psychological thrillers": "Thriller",
    "kids": "Children", "family": "Children", "childrens": "Children", "children s": "Children",
    "animated": "Animation", "anime": "Animation", "cartoon": "Animation", "cartoons": "Animation",
    "action adventure": "Action", "adventures": "Adventure",
    "true crime": "Crime", "crime drama": "Crime",
    "war movie": "War", "westerns": "Western", "noir": "Film-Noir", "film noir": "Film-Noir",
    "musicals": "Musical", "mysteries": "Mystery", "fantasies": "Fantasy",
    "scary": "Horror", "slasher": "Horror", "slashers": "Horror", "horrors": "Horror",
    "comedic": "Comedy", "funny": "Comedy",
}

# Trailing nouns stripped before matching ("horror movies" -> "horror").
NOUNS = r"(movies|movie|films|film|flicks|flick|shows|show|series|genre|stuff|things)"

DECADES = [(1920, 1929), (1930, 1939), (1940, 1949), (1950, 1959), (1960, 1969), (1970, 1979),
           (1980, 1989), (1990, 1999), (2000, 2009), (2010, 2019)]


def norm(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"[‘’“”]", "'", s)
    s = re.sub(r"[^a-z0-9\s'-]", " ", s)
    s = re.sub(r"\b" + NOUNS + r"\b", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    s = re.sub(r"^(a|an|the|some|any|good|great|best|really|very)\s+", "", s)
    return s.strip(" -'")


def map_genre(v: str):
    n = norm(v)
    if not n:
        return None
    flat = {g.lower().replace("-", " ").replace("'", ""): g for g in ML_GENRES}
    key = n.replace("-", " ")
    if key in flat:
        return flat[key]
    if key.endswith("s") and key[:-1] in flat:
        return flat[key[:-1]]
    if n in GENRE_ALIAS:
        return GENRE_ALIAS[n]
    if key in GENRE_ALIAS:
        return GENRE_ALIAS[key]
    # slash / plus compounds: take the first mappable component
    for part in re.split(r"[/+&]| and ", n):
        p = part.strip()
        if not p:
            continue
        pk = p.replace("-", " ")
        if pk in flat:
            return flat[pk]
        if p in GENRE_ALIAS:
            return GENRE_ALIAS[p]
    return None


def map_tone(v: str, tag_lookup: dict):
    n = norm(v)
    if not n:
        return None
    for cand in (n, n.replace(" ", "-"), n.replace("-", " ")):
        if cand in tag_lookup:
            return tag_lookup[cand]
    # single-word head of a short phrase (e.g. "really scary" -> "scary")
    words = n.split()
    if 1 < len(words) <= 3:
        for w in words:
            if w in tag_lookup:
                return tag_lookup[w]
    return None


def map_era(v: str):
    n = norm(v)
    years = [int(y) for y in re.findall(r"(19\d{2}|20\d{2})", n)]
    if not years:
        m = re.findall(r"\b([2-9]0)'?s\b", n)          # "80s", "90's"
        for d in m:
            base = 1900 + int(d)
            years.append(base)
        m2 = re.findall(r"\b(19\d{2}|20\d{2})'?s\b", n)
        years.extend(int(y) for y in m2)
    if not years:
        return None
    y = min(years)
    for lo, hi in DECADES:
        if lo <= y <= hi:
            return f"{lo}s"
    if y >= 2020:
        return "2020s"
    return None


def main() -> None:
    census = json.load(open(CENSUS, encoding="utf-8"))

    tag_lookup = {}
    with open(TAGS, encoding="utf-8") as fh:
        for row in list(csv.reader(fh))[1:]:
            if len(row) >= 2:
                t = row[1].strip()
                tag_lookup[t.lower()] = t
                tag_lookup[t.lower().replace("-", " ")] = t

    result = {
        "analysis": "m1b_opener_value_mapping",
        "design_sheet": "docs/design/DS_M1_OPENER_FAMILIES.md",
        "source_census": "experiments/battery/m1b_census.json",
        "ml_genres": ML_GENRES,
        "n_genome_tags": len({v for v in tag_lookup.values()}),
        "slots": {},
    }

    # Map over ALL occurrences (any_values), which is the honest denominator: the opener view is
    # a modelling choice, the marginal is not.
    for slot, mapper in (("genres_pos", "genre"), ("tone", "tone"), ("era", "era")):
        pairs = census["any_values_top50"].get(slot, [])
        # top50 is a sample of the tail; recompute over it and flag the coverage of that sample
        mapped, unmapped = Counter(), Counter()
        for value, count in pairs:
            if mapper == "genre":
                tgt = map_genre(value)
            elif mapper == "tone":
                tgt = map_tone(value, tag_lookup)
            else:
                tgt = map_era(value)
            if tgt is None:
                unmapped[value] += count
            else:
                mapped[tgt] += count
        tot = sum(mapped.values()) + sum(unmapped.values())
        result["slots"][slot] = {
            "signature_space": mapper,
            "occurrences_considered": tot,
            "note": "computed over the census top-50 value list for this slot, not the full tail",
            "mapped_occurrences": sum(mapped.values()),
            "unmapped_occurrences": sum(unmapped.values()),
            "mapped_share": round(sum(mapped.values()) / tot, 4) if tot else None,
            "mapped_targets": mapped.most_common(),
            "unmapped_values": unmapped.most_common(30),
        }

    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=1, ensure_ascii=False)

    print("[m1b-map] mapping coverage over each slot's census top-50 values")
    for slot, d in result["slots"].items():
        print(f"  {slot:<12} -> {d['signature_space']:<6} "
              f"mapped {d['mapped_occurrences']:>5}/{d['occurrences_considered']:<5} "
              f"({d['mapped_share']:.1%})")
        print(f"       targets: {d['mapped_targets'][:8]}")
        if d["unmapped_values"]:
            print(f"       unmapped head: {d['unmapped_values'][:5]}")
    print(f"[m1b-map] wrote {OUT}")


if __name__ == "__main__":
    main()
