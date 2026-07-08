"""
answerer_prereq_items_tags.py -- P2 (item list) + P3 (tag questions) prerequisite builds
for the Answerer v1. NO LLM calls. Deterministic. Reads only local ML-25M caches.

Canonical dense-id space (from llm_answerability_gate.load_data):
  dense id j in 0..ni-1 ; movieId = keepI[j] (keepI sorted ascending) ; cnt[j] = train-like popularity.

Outputs (.cache/instrument2/):
  item_lists.json        -- top-800 / top-1000 / top-2000 dense-id lists + per-list stats
  tag_questions.json     -- all 1128 genome tags: question template, membership, priors, awkward flag
Adult flagging for items is added by answerer_prereq_imdb.py (needs title.basics isAdult).
"""
import os, re, json, collections
import numpy as np
import pandas as pd

BASE = 'C:/dev/phd/casper/data/movielens'
META = f'{BASE}/.cache/ml25m/meta.npz'
MOVIES = f'{BASE}/movies.csv'
GENOME = f'{BASE}/genome-scores.csv'
TAGS = f'{BASE}/genome-tags.csv'
OUTDIR = '.cache/instrument2'
COV_THRESH = 0.5  # project concept-membership relevance threshold (prep_concepts_ml25m.COV_THRESH)

YEAR_RE = re.compile(r"\((\d{4})\)")
# franchise regex reused verbatim from answerability_arena3_bank.py
FR_RE = re.compile(r"(\b(II|III|IV|VI|VII|VIII|IX|XI|XII)\b|:|\bPart\b|\bChapter\b|\bEpisode\b|"
                   r"\bVol\b|\b[2-9]\b)", re.I)
GENRES = ["Action", "Adventure", "Animation", "Children", "Comedy", "Crime", "Documentary",
          "Drama", "Fantasy", "Film-Noir", "Horror", "IMAX", "Musical", "Mystery", "Romance",
          "Sci-Fi", "Thriller", "War", "Western", "(no genres listed)"]

GRIDS = {
    "gate_ml25m": ".cache/instrument2/answerability_grid_ml25m.json",
    "mainstudy": ".cache/instrument2/answerability_mainstudy_grid.json",
    "arena3": ".cache/instrument2/answerability_arena3_grid.json",
}


def load_meta():
    d = np.load(META)
    keepI = d['keepI'].astype(np.int64)
    cnt = d['cnt'].astype(np.float64)
    ni = int(d['ni'])
    # titles + genres
    mv = {}
    with open(MOVIES, encoding='utf-8') as f:
        next(f)
        for line in f:
            i0 = line.find(','); i1 = line.rfind(',')
            mid = int(line[:i0]); title = line[i0 + 1:i1].strip().strip('"')
            genres = line[i1 + 1:].strip().split('|')
            mv[mid] = (title, genres)
    title = [None] * ni; dgen = [None] * ni; year = [None] * ni
    for j in range(ni):
        mid = int(keepI[j]); t, gs = mv.get(mid, (f'movie{mid}', ['(no genres listed)']))
        title[j] = t; dgen[j] = gs
        m = YEAR_RE.search(t or '')
        year[j] = int(m.group(1)) if m else None
    pr = cnt.argsort().argsort() / (ni - 1)
    return dict(keepI=keepI, cnt=cnt, ni=ni, title=title, dgen=dgen, year=year, pr=pr)


def genome_movie_set():
    """Unique movieIds that have ANY genome score (the ~13.8k genome-scored set)."""
    seen = set()
    for chunk in pd.read_csv(GENOME, usecols=['movieId'], dtype={'movieId': np.int64},
                             chunksize=2_000_000):
        seen.update(chunk['movieId'].unique().tolist())
    return seen


def is_franchise(t):
    return 1 if FR_RE.search(YEAR_RE.sub('', t or '')) else 0


def collect_judged_item_cells():
    """Return dict grid_name -> set of (user, dense_j) item cells already judged."""
    out = {}
    for name, path in GRIDS.items():
        if not os.path.exists(path):
            out[name] = set(); continue
        d = json.load(open(path))
        cells = set()
        users = d.get('users', {})
        bank = d.get('bank')  # arena3 only
        for u, rec in users.items():
            u = int(u)
            ans = rec.get('ans', {})
            if bank is not None:
                # arena3: ans index -> bank[index].j
                for idx in ans.keys():
                    bi = int(idx)
                    if bi < len(bank):
                        cells.add((u, int(bank[bi]['j'])))
            else:
                Q = rec.get('Q', [])
                for idx_str in ans.keys():
                    qi = int(idx_str)
                    if qi < len(Q) and Q[qi][0] == 'item':
                        cells.add((u, int(Q[qi][1]['j'])))
        out[name] = cells
    return out


def list_stats(D, ids, gset, judged):
    ids = list(ids)
    idset = set(ids)
    cnt = D['cnt']; year = D['year']; dgen = D['dgen']; title = D['title']; pr = D['pr']
    # decade histogram
    dec = collections.Counter()
    for j in ids:
        y = year[j]; dec[(y // 10 * 10) if y else None] += 1
    # genre coverage
    gcov = collections.Counter()
    for j in ids:
        for g in dgen[j]:
            gcov[g] += 1
    # franchise share
    fr = sum(is_franchise(title[j]) for j in ids)
    # genome presence
    no_genome = [j for j in ids if int(D['keepI'][j]) not in gset]
    # duplicates
    dup = len(ids) - len(idset)
    # min ratings-count + percentile
    counts = np.array([cnt[j] for j in ids])
    min_cnt = float(counts.min()); min_j = ids[int(np.argmin(counts))]
    min_pct = float(pr[min_j])  # percentile among all 18430
    # overlap / absorbable judged cells
    absorb = {}
    for name, cells in judged.items():
        c = sum(1 for (u, j) in cells if j in idset)
        items_here = len(set(j for (u, j) in cells if j in idset))
        absorb[name] = {"cells": c, "distinct_items": items_here}
    total_absorb_cells = sum(v['cells'] for v in absorb.values())
    return {
        "n": len(ids),
        "decade_hist": {str(k): v for k, v in sorted(dec.items(), key=lambda x: (x[0] is None, x[0]))},
        "genre_coverage": dict(gcov.most_common()),
        "franchise_count": fr, "franchise_share": round(fr / len(ids), 4),
        "n_missing_genome": len(no_genome),
        "missing_genome_movieIds": [int(D['keepI'][j]) for j in no_genome][:50],
        "n_duplicates": dup,
        "min_ratings_count": min_cnt,
        "min_ratings_count_percentile": round(min_pct, 4),
        "min_item_title": title[min_j],
        "absorbable_by_grid": absorb,
        "absorbable_cells_total": total_absorb_cells,
    }


# ------------------------------------------------------------------ P3 tag phrasing
META_KEYWORDS = ['imdb', 'criterion', 'oscar', 'afi', 'bd-r', 'national film registry',
                 'top 250', 'seen more than once', 'best picture', 'nominee', 'academy award',
                 'golden globe', 'palme', 'boxoffice', 'box office', 'trilogy', 'franchise',
                 'blu-ray', 'dvd', 'video release']
ADJ_SUFFIX = ('ing', 'ed', 'y', 'ic', 'ical', 'ful', 'ous', 'ive', 'al', 'ish', 'less',
              'able', 'ible', 'ary', 'ent', 'ant')
# small hand list of obvious adjectives that don't match suffixes cleanly
ADJ_WORDS = {'dark', 'good', 'great', 'bad', 'weird', 'odd', 'slow', 'fast', 'sad', 'grim',
             'bleak', 'tense', 'epic', 'cult', 'camp', 'noir', 'raw', 'brutal', 'violent',
             'gory', 'sweet', 'cute', 'smart', 'clever', 'complex', 'simple', 'quirky',
             'surreal', 'stylish', 'gritty', 'cheesy', 'campy', 'melancholy'}


def phrase_tag(tag):
    """Return (template, awkward_bool, reason)."""
    t = tag.strip()
    low = t.lower()
    awkward = False; reasons = []
    # meta / list membership tags
    if any(k in low for k in META_KEYWORDS):
        awkward = True; reasons.append('meta/list tag')
        return (f"Do you specifically seek out films known as '{t}'?", True, ';'.join(reasons))
    if re.search(r'\d', t):
        awkward = True; reasons.append('contains digits')
    if '(' in t or ')' in t:
        awkward = True; reasons.append('contains parentheses')
    if len(low) <= 2:
        awkward = True; reasons.append('very short (<=2 chars)')
    # adjective vs noun heuristic
    words = low.split()
    last = words[-1] if words else low
    is_adj = (low in ADJ_WORDS) or last.endswith(ADJ_SUFFIX)
    if is_adj:
        tmpl = f"Do you like {t} movies?"
    else:
        tmpl = f"Do you like movies about {t}?"
    return (tmpl, awkward, ';'.join(reasons) if reasons else '')


def build_tags(D):
    keepI = D['keepI']; cnt = D['cnt']
    mid2j = {int(m): j for j, m in enumerate(keepI)}
    tagnames = pd.read_csv(TAGS).set_index('tagId')['tag'].to_dict()
    # membership over the ITEM UNIVERSE (18430), relevance >= COV_THRESH, popularity-weighted
    memb = collections.Counter()          # tagId -> n member movies (in universe)
    popw = collections.defaultdict(float)  # tagId -> sum of cnt over member movies
    for chunk in pd.read_csv(GENOME, dtype={'movieId': np.int64, 'tagId': np.int32,
                                            'relevance': np.float32}, chunksize=2_000_000):
        hi = chunk[chunk['relevance'] >= COV_THRESH]
        hi = hi[hi['movieId'].isin(mid2j)]
        if len(hi) == 0:
            continue
        js = hi['movieId'].map(mid2j).values
        w = cnt[js]
        for tid, ww in zip(hi['tagId'].values, w):
            memb[int(tid)] += 1
            popw[int(tid)] += float(ww)
    tot_pop = float(cnt.sum())
    tags = []
    n_awk = 0
    for tid in sorted(tagnames.keys()):
        name = str(tagnames[tid])
        tmpl, awk, reason = phrase_tag(name)
        n_awk += int(awk)
        tags.append({
            "tagId": int(tid), "tag": name, "question": tmpl,
            "awkward": bool(awk), "awkward_reason": reason,
            "membership_size": int(memb.get(tid, 0)),
            "pop_weighted_membership": round(popw.get(tid, 0.0), 1),
            "answer_rate_prior": round(popw.get(tid, 0.0) / tot_pop, 6),
        })
    return tags, n_awk


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    D = load_meta()
    print(f'[prereq] ni={D["ni"]} loaded', flush=True)
    print('[prereq] scanning genome for genome-scored movie set...', flush=True)
    gset = genome_movie_set()
    print(f'[prereq] genome-scored movies (all ML-25M): {len(gset)}', flush=True)
    gset_in_univ = sum(1 for m in D['keepI'] if int(m) in gset)
    print(f'[prereq] of {D["ni"]} item-universe items, {gset_in_univ} have genome '
          f'({100*gset_in_univ/D["ni"]:.1f}%)', flush=True)

    judged = collect_judged_item_cells()
    for k, v in judged.items():
        print(f'[prereq] judged item cells [{k}]: {len(v)}', flush=True)

    order = np.argsort(-D['cnt'])  # deterministic top-by-count (stable argsort not needed; ties by id)
    lists = {}
    for N in (800, 1000, 2000):
        ids = [int(j) for j in order[:N]]
        lists[f'top{N}'] = {"ids": ids, "stats": list_stats(D, ids, gset, judged)}
        print(f'[prereq] top{N}: min_cnt={lists[f"top{N}"]["stats"]["min_ratings_count"]:.0f} '
              f'absorb={lists[f"top{N}"]["stats"]["absorbable_cells_total"]}', flush=True)

    # spot list of top-15 for eyeball
    top15 = [{"j": int(j), "movieId": int(D['keepI'][j]), "title": D['title'][j],
              "cnt": int(D['cnt'][j])} for j in order[:15]]

    out = {
        "schema": "answerer-v1 item lists (P2)",
        "dense_id_convention": "dense id j; movieId=keepI[j]; cnt[j]=train-like popularity",
        "genome_scored_movies_all": len(gset),
        "genome_coverage_of_universe": {"n": gset_in_univ, "of": D['ni'],
                                        "pct": round(100 * gset_in_univ / D['ni'], 2)},
        "primary": "top1000",
        "top15_eyeball": top15,
        "lists": lists,
        "adult_flag_pending": "isAdult filled by answerer_prereq_imdb.py",
    }
    json.dump(out, open(f'{OUTDIR}/item_lists.json', 'w'), indent=1)
    print(f'[prereq] wrote {OUTDIR}/item_lists.json', flush=True)

    tags, n_awk = build_tags(D)
    memb_sizes = np.array([t['membership_size'] for t in tags])
    tag_out = {
        "schema": "answerer-v1 tag questions (P3)",
        "relevance_threshold": COV_THRESH,
        "threshold_source": "prep_concepts_ml25m.COV_THRESH (project concept machinery)",
        "n_tags": len(tags),
        "n_awkward_flagged": n_awk,
        "membership_over": "item universe (18430), relevance>=0.5",
        "membership_size_stats": {
            "min": int(memb_sizes.min()), "max": int(memb_sizes.max()),
            "mean": round(float(memb_sizes.mean()), 1),
            "median": int(np.median(memb_sizes)),
            "n_zero_membership": int((memb_sizes == 0).sum()),
            "pctile_10_50_90": [int(np.percentile(memb_sizes, 10)),
                                int(np.percentile(memb_sizes, 50)),
                                int(np.percentile(memb_sizes, 90))],
        },
        "note_no_pruning": "pruning is empirical/pilot-only; all 1128 tags retained, awkward flagged not removed",
        "tags": tags,
    }
    json.dump(tag_out, open(f'{OUTDIR}/tag_questions.json', 'w'), indent=1)
    print(f'[prereq] wrote {OUTDIR}/tag_questions.json  ({len(tags)} tags, {n_awk} awkward)', flush=True)


if __name__ == '__main__':
    main()
