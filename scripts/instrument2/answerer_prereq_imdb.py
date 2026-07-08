"""
answerer_prereq_imdb.py -- P1 IMDb attribute join for the Answerer v1. NO LLM calls. Deterministic.

Joins ML-25M item universe (18430 keepI movieIds) to IMDb offline TSV dumps via links.csv (imdbId),
and builds the ATTRIBUTE ENTITY UNIVERSE:
  directors (title.crew) >=3 films ; actors/actresses (title.principals ordering<=4) >=5 ;
  composers (principals category=composer) >=3 ; writers (principals category=writer) >=3.
Per entity: name, member movie list (dense ids), per-entity popularity = sum of member cnt.
Franchise detection: conservative title-stem grouping (TMDb collection id NOT trivially in links -> skipped).
Also emits isAdult flags for the item universe (feeds P2 adult sanity) and updates item_lists.json.

Inputs: IMDb gz in the scratchpad imdb dir (arg --imdb).
Outputs: .cache/instrument2/attr_membership.json ; patches item_lists.json adult stats.
"""
import os, sys, gzip, csv, re, json, argparse, collections
import numpy as np

BASE = 'C:/dev/phd/casper/data/movielens'
META = f'{BASE}/.cache/ml25m/meta.npz'
LINKS = f'{BASE}/links.csv'
MOVIES = f'{BASE}/movies.csv'
OUTDIR = '.cache/instrument2'

YEAR_RE = re.compile(r"\((\d{4})\)")
THRESH = {"director": 3, "actor": 5, "composer": 3, "writer": 3}


def load_universe():
    d = np.load(META)
    keepI = d['keepI'].astype(np.int64)
    cnt = d['cnt'].astype(np.float64)
    ni = int(d['ni'])
    mv = {}
    with open(MOVIES, encoding='utf-8') as f:
        next(f)
        for line in f:
            i0 = line.find(','); i1 = line.rfind(',')
            mid = int(line[:i0]); title = line[i0 + 1:i1].strip().strip('"')
            mv[mid] = title
    title = [mv.get(int(keepI[j]), f'movie{keepI[j]}') for j in range(ni)]
    return keepI, cnt, ni, title


def load_links(keepI):
    """movieId -> tconst ('tt%07d'); restricted to universe. Returns tconst2j, j2tconst."""
    mid2j = {int(m): j for j, m in enumerate(keepI)}
    tconst2j = {}; j2tconst = {}
    with open(LINKS, encoding='utf-8') as f:
        next(f)
        for line in f:
            parts = line.rstrip('\n').split(',')
            if len(parts) < 2:
                continue
            mid = int(parts[0]); imdb = parts[1].strip()
            if mid in mid2j and imdb:
                tc = 'tt' + imdb
                j = mid2j[mid]
                tconst2j[tc] = j; j2tconst[j] = tc
    return tconst2j, j2tconst


def stream_tsv(path):
    with gzip.open(path, 'rt', encoding='utf-8', newline='') as f:
        r = csv.reader(f, delimiter='\t')
        header = next(r)
        idx = {c: i for i, c in enumerate(header)}
        for row in r:
            yield row, idx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--imdb', required=True, help='dir with title.basics/crew/principals + name.basics .tsv.gz')
    args = ap.parse_args()
    IM = args.imdb

    keepI, cnt, ni, title = load_universe()
    tconst2j, j2tconst = load_links(keepI)
    print(f'[imdb] universe={ni} linked_to_imdb={len(tconst2j)} '
          f'({100*len(tconst2j)/ni:.1f}%)', flush=True)
    target = set(tconst2j.keys())

    # --- title.basics : isAdult + startYear cross-check
    isadult = {}
    for row, idx in stream_tsv(f'{IM}/title.basics.tsv.gz'):
        tc = row[idx['tconst']]
        if tc in target:
            isadult[tc] = (row[idx['isAdult']] == '1')
    n_adult = sum(1 for v in isadult.values() if v)
    print(f'[imdb] title.basics matched={len(isadult)} adult_flagged={n_adult}', flush=True)

    # --- title.crew : directors (comma nconst list)
    directors_of = collections.defaultdict(list)   # j -> [nconst]
    need_names = set()
    for row, idx in stream_tsv(f'{IM}/title.crew.tsv.gz'):
        tc = row[idx['tconst']]
        if tc not in target:
            continue
        j = tconst2j[tc]
        ds = row[idx['directors']]
        if ds and ds != r'\N':
            for nc in ds.split(','):
                directors_of[j].append(nc); need_names.add(nc)
    print(f'[imdb] title.crew: movies_with_director={len(directors_of)}', flush=True)

    # --- title.principals : cast(order<=4, actor/actress), composer, writer
    cast_of = collections.defaultdict(list)      # j -> [nconst]
    composer_of = collections.defaultdict(list)
    writer_of = collections.defaultdict(list)
    for row, idx in stream_tsv(f'{IM}/title.principals.tsv.gz'):
        tc = row[idx['tconst']]
        if tc not in target:
            continue
        j = tconst2j[tc]
        cat = row[idx['category']]; nc = row[idx['nconst']]
        ordv = row[idx['ordering']]
        if cat in ('actor', 'actress'):
            try:
                if int(ordv) <= 4:
                    cast_of[j].append(nc); need_names.add(nc)
            except ValueError:
                pass
        elif cat == 'composer':
            composer_of[j].append(nc); need_names.add(nc)
        elif cat == 'writer':
            writer_of[j].append(nc); need_names.add(nc)
    print(f'[imdb] principals: cast_movies={len(cast_of)} composer_movies={len(composer_of)} '
          f'writer_movies={len(writer_of)}', flush=True)

    # --- name.basics : nconst -> primaryName
    name_of = {}
    for row, idx in stream_tsv(f'{IM}/name.basics.tsv.gz'):
        nc = row[idx['nconst']]
        if nc in need_names:
            name_of[nc] = row[idx['primaryName']]
    print(f'[imdb] name.basics resolved {len(name_of)}/{len(need_names)} names', flush=True)

    # --- invert to entity -> member dense ids
    def invert(perj):
        e2j = collections.defaultdict(list)
        for j, ncs in perj.items():
            for nc in set(ncs):
                e2j[nc].append(j)
        return e2j
    inv = {"director": invert(directors_of), "actor": invert(cast_of),
           "composer": invert(composer_of), "writer": invert(writer_of)}

    entities = {}
    stats_per_type = {}
    for etype, e2j in inv.items():
        thr = THRESH[etype]
        kept = []
        for nc, js in e2j.items():
            if len(js) >= thr:
                js = sorted(set(js))
                pop = float(sum(cnt[j] for j in js))
                kept.append({
                    "entity_id": f"{etype}:{nc}", "nconst": nc,
                    "name": name_of.get(nc, nc), "type": etype,
                    "n_movies": len(js), "member_dense_ids": js,
                    "member_movieIds": [int(keepI[j]) for j in js],
                    "popularity": round(pop, 1),
                })
        kept.sort(key=lambda e: -e['popularity'])
        entities[etype] = kept
        stats_per_type[etype] = {"threshold_movies": thr, "n_entities": len(kept),
                                 "total_candidates_pre_threshold": len(e2j)}
        print(f'[imdb] {etype}: {len(kept)} entities (>= {thr} films) of {len(e2j)} candidates',
              flush=True)

    # --- coverage of universe: movies with no director / no cast
    no_dir = [j for j in range(ni) if j not in directors_of]
    no_cast = [j for j in range(ni) if j not in cast_of]
    no_both = [j for j in range(ni) if j not in directors_of and j not in cast_of]
    print(f'[imdb] coverage: no_director={len(no_dir)} ({100*len(no_dir)/ni:.2f}%) '
          f'no_cast={len(no_cast)} ({100*len(no_cast)/ni:.2f}%) '
          f'no_both={len(no_both)} ({100*len(no_both)/ni:.2f}%)', flush=True)

    # --- franchise detection: conservative title-stem grouping
    def stem(t):
        base = YEAR_RE.sub('', t or '').strip()
        # cut at series separators
        base = re.split(r':| - |\bPart\b|\bChapter\b|\bEpisode\b|\bVol\b', base, flags=re.I)[0]
        # strip trailing roman numerals / digits (sequel markers)
        base = re.sub(r'\s+(\b(II|III|IV|V|VI|VII|VIII|IX|X|XI|XII)\b|\d+)\s*$', '', base, flags=re.I)
        base = re.sub(r',\s*(The|A|An)$', '', base, flags=re.I)  # normalize trailing article
        return base.strip().lower()
    stem2j = collections.defaultdict(list)
    for j in range(ni):
        s = stem(title[j])
        if len(s) >= 4:
            stem2j[s].append(j)
    franchises = []
    for s, js in stem2j.items():
        if len(js) >= 2:
            js = sorted(js, key=lambda j: -cnt[j])
            franchises.append({"stem": s, "n_movies": len(js), "member_dense_ids": js,
                               "member_movieIds": [int(keepI[j]) for j in js],
                               "titles": [title[j] for j in js],
                               "popularity": round(float(sum(cnt[j] for j in js)), 1)})
    franchises.sort(key=lambda e: -e['popularity'])
    print(f'[imdb] franchise groups (>=2 shared stem): {len(franchises)}', flush=True)

    # --- spot-print 10 famous directors + actors
    spot = {"directors": [(e['name'], e['n_movies'],
                           [title[j] for j in e['member_dense_ids'][:6]])
                          for e in entities['director'][:10]],
            "actors": [(e['name'], e['n_movies'],
                        [title[j] for j in e['member_dense_ids'][:6]])
                       for e in entities['actor'][:10]]}

    out = {
        "schema": "answerer-v1 attribute membership (P1)",
        "join": {"universe": ni, "linked_to_imdb": len(tconst2j),
                 "link_pct": round(100 * len(tconst2j) / ni, 2)},
        "thresholds": THRESH,
        "stats_per_type": stats_per_type,
        "coverage": {"no_director": len(no_dir), "no_director_pct": round(100 * len(no_dir) / ni, 3),
                     "no_cast": len(no_cast), "no_cast_pct": round(100 * len(no_cast) / ni, 3),
                     "no_both": len(no_both), "no_both_pct": round(100 * len(no_both) / ni, 3)},
        "n_franchise_groups": len(franchises),
        "adult": {"n_flagged": n_adult, "matched_basics": len(isadult)},
        "spot_check": spot,
        "entities": entities,
        "franchises": franchises[:2000],
        "isadult_by_movieId": {str(int(keepI[j])): bool(isadult.get(j2tconst.get(j, ''), False))
                               for j in range(ni) if isadult.get(j2tconst.get(j, ''), False)},
    }
    os.makedirs(OUTDIR, exist_ok=True)
    json.dump(out, open(f'{OUTDIR}/attr_membership.json', 'w'), indent=1)
    print(f'[imdb] wrote {OUTDIR}/attr_membership.json', flush=True)

    # --- patch item_lists.json with adult flags per list
    ilp = f'{OUTDIR}/item_lists.json'
    if os.path.exists(ilp):
        il = json.load(open(ilp))
        adult_j = set(j for j in range(ni) if isadult.get(j2tconst.get(j, ''), False))
        for name, blob in il['lists'].items():
            ad = [j for j in blob['ids'] if j in adult_j]
            blob['stats']['n_adult_titles'] = len(ad)
            blob['stats']['adult_movieIds'] = [int(keepI[j]) for j in ad]
        il['adult_flag_pending'] = f'DONE: isAdult joined ({n_adult} adult in universe)'
        json.dump(il, open(ilp, 'w'), indent=1)
        print(f'[imdb] patched adult flags into {ilp}', flush=True)


if __name__ == '__main__':
    main()
