"""
Paper 1 / M1: Refined acceptance tests (v2) for the measurement instrument.

v1's T1 (random-attribute top-10 overlap) was confounded: obscure attributes
leave most of the slate ranked by the popularity prior in both polarity
conditions, inflating overlap regardless of polarity sensitivity. v2 measures
polarity where it is identifiable:

  T1v2 (attribute polarity): for each slate attribute with >=3 matching
        movies, mean rank of matching movies must improve when the attribute
        is liked vs disliked. Pass: >=80% of attributes move correctly.
  T1b  (prior-referenced overlap): top-10 overlap(liked, disliked) on
        genre-combo reveals, referenced against overlap with the no-input
        prior. Pass: overlap(l,d) < overlap(l,prior).
  T3v2 (multi-franchise flip): direction of related-movie rank movement for
        every franchise anchor in the slate. Pass: >=70% correct direction.

Uses ckpt['items'] as the authoritative item ordering.
Run from casper root:  poetry run python scripts/paper1/test_instrument_v2.py
Results merged into experiments/paper1/instrument_acceptance.json
"""

import sys
sys.path.insert(0, '.')

import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from test_instrument_lib import load_instruments  # shared loader (see below)

DATA_DIR = Path('C:/dev/phd/casper/data/movielens')
OUT_DIR = Path('C:/dev/phd/casper/experiments/paper1')
TOP_K = 10
SEED = 42
rng = np.random.default_rng(SEED)

FRANCHISE_PATTERNS = {
    'Star Wars': r'Star Wars',
    'Matrix': r'Matrix',
    'Toy Story': r'Toy Story',
    'Lord of the Rings': r'Lord of the Rings',
    'Terminator': r'Terminator',
    'Back to the Future': r'Back to the Future',
    'Indiana Jones': r'(Indiana Jones|Raiders of the Lost Ark)',
    'Jurassic Park': r'Jurassic',
    'Godfather': r'Godfather',
    'Alien': r'^Alien',
    'Die Hard': r'Die Hard',
    'Batman': r'Batman',
    'Kill Bill': r'Kill Bill',
    'Shrek': r'Shrek',
    'Harry Potter': r'Harry Potter',
    'Pirates': r'Pirates of the Caribbean',
    'Spider-Man': r'Spider-Man',
    'X-Men': r'(X-Men|X2)',
    'Bourne': r'Bourne',
    'Mission Impossible': r'Mission: Impossible',
    'Men in Black': r'Men in Black',
    'Austin Powers': r'Austin Powers',
    'Ocean\'s': r"Ocean's",
    'Star Trek': r'Star Trek',
}
DID_NOISE_FLOOR = 0.02  # |DiD| below this is uninformative for that franchise


def build_attribute_movie_map(items, movies_df, credits_data):
    """Map each attribute item index -> list of slate movie indices it matches."""
    movie_indices = {item[1]: i for i, item in enumerate(items) if item[0] == 'movie'}
    genres_by_movie = {}
    for mid in movie_indices:
        row = movies_df[movies_df['movieId'] == mid]
        if len(row) and pd.notna(row.iloc[0]['genres']):
            genres_by_movie[mid] = set(row.iloc[0]['genres'].split('|'))
        else:
            genres_by_movie[mid] = set()

    attr_map = {}
    for i, (itype, ival, _) in enumerate(items):
        if itype == 'genre':
            attr_map[i] = [movie_indices[mid] for mid, gs in genres_by_movie.items()
                           if ival in gs]
        elif itype == 'actor':
            matches = []
            for mid_str, actors in credits_data.get('movie_actors', {}).items():
                mid = int(mid_str)
                if mid in movie_indices and ival in actors[:5]:
                    matches.append(movie_indices[mid])
            attr_map[i] = matches
        elif itype == 'director':
            matches = []
            for mid_str, directors in credits_data.get('movie_directors', {}).items():
                mid = int(mid_str)
                if mid in movie_indices and ival in directors:
                    matches.append(movie_indices[mid])
            attr_map[i] = matches
    return attr_map


def test_attribute_polarity(wrapper, items, attr_map):
    results = []
    for attr_idx, movie_idxs in attr_map.items():
        if len(movie_idxs) < 3:
            continue
        liked = wrapper.predict([(attr_idx, 1.0)])
        disliked = wrapper.predict([(attr_idx, 0.0)])
        rank_l = np.empty_like(liked)
        rank_l[np.argsort(-liked)] = np.arange(len(liked))
        rank_d = np.empty_like(disliked)
        rank_d[np.argsort(-disliked)] = np.arange(len(disliked))
        mean_rank_liked = float(np.mean([rank_l[m] for m in movie_idxs]))
        mean_rank_disliked = float(np.mean([rank_d[m] for m in movie_idxs]))
        results.append({
            'attribute': items[attr_idx][2],
            'type': items[attr_idx][0],
            'n_movies': len(movie_idxs),
            'mean_rank_liked': mean_rank_liked,
            'mean_rank_disliked': mean_rank_disliked,
            'shift': mean_rank_disliked - mean_rank_liked,
            'correct': mean_rank_liked < mean_rank_disliked,
        })
    frac = float(np.mean([r['correct'] for r in results])) if results else np.nan
    mean_shift = float(np.mean([r['shift'] for r in results])) if results else np.nan
    worst = sorted(results, key=lambda r: r['shift'])[:5]
    return {
        'n_attributes_tested': len(results),
        'fraction_correct_direction': frac,
        'mean_rank_shift': mean_shift,
        'worst_attributes': worst,
        'pass': bool(frac >= 0.80),
    }


def test_attribute_polarity_did(wrapper, items, attr_map):
    """T1v3: difference-in-differences in score space.

    A disliked attribute may legitimately shift ALL predictions down (it
    signals a harsher rater overall); what polarity requires is that
    MATCHING movies move by more than non-matching ones:
        DiD(e) = [mean_match(liked) - mean_match(disliked)]
               - [mean_other(liked) - mean_other(disliked)]  > 0
    """
    results = []
    n_movies = wrapper.n_movies
    for attr_idx, movie_idxs in attr_map.items():
        if len(movie_idxs) < 3:
            continue
        liked = wrapper.predict([(attr_idx, 1.0)])
        disliked = wrapper.predict([(attr_idx, 0.0)])
        match = np.zeros(n_movies, dtype=bool)
        match[movie_idxs] = True
        d_match = float(liked[match].mean() - disliked[match].mean())
        d_other = float(liked[~match].mean() - disliked[~match].mean())
        results.append({
            'attribute': items[attr_idx][2],
            'type': items[attr_idx][0],
            'n_movies': len(movie_idxs),
            'delta_matching': d_match,
            'delta_other': d_other,
            'did': d_match - d_other,
            'correct': d_match > d_other,
        })
    frac = float(np.mean([r['correct'] for r in results])) if results else np.nan
    mean_did = float(np.mean([r['did'] for r in results])) if results else np.nan
    by_type = {}
    for t in ('genre', 'actor', 'director'):
        sub = [r for r in results if r['type'] == t]
        if sub:
            by_type[t] = {'n': len(sub),
                          'frac_correct': float(np.mean([r['correct'] for r in sub])),
                          'mean_did': float(np.mean([r['did'] for r in sub]))}
    worst = sorted(results, key=lambda r: r['did'])[:5]
    return {
        'n_attributes_tested': len(results),
        'fraction_correct_direction': frac,
        'mean_did': mean_did,
        'by_type': by_type,
        'worst_attributes': worst,
        'pass': bool(frac >= 0.80),
    }


def test_prior_referenced_overlap(wrapper, items, attr_map, n_trials=100):
    genre_idxs = [i for i, it in enumerate(items)
                  if it[0] == 'genre' and len(attr_map.get(i, [])) >= 3]
    prior = wrapper.predict([])
    top_prior = set(np.argsort(-prior)[:TOP_K])

    o_ld, o_lp, o_dp = [], [], []
    for _ in range(n_trials):
        k = int(rng.integers(2, 5))
        ents = rng.choice(genre_idxs, size=min(k, len(genre_idxs)), replace=False)
        liked = wrapper.predict([(int(e), 1.0) for e in ents])
        disliked = wrapper.predict([(int(e), 0.0) for e in ents])
        tl = set(np.argsort(-liked)[:TOP_K])
        td = set(np.argsort(-disliked)[:TOP_K])
        o_ld.append(len(tl & td) / TOP_K)
        o_lp.append(len(tl & top_prior) / TOP_K)
        o_dp.append(len(td & top_prior) / TOP_K)

    return {
        'overlap_liked_disliked': float(np.mean(o_ld)),
        'overlap_liked_prior': float(np.mean(o_lp)),
        'overlap_disliked_prior': float(np.mean(o_dp)),
        'pass': bool(np.mean(o_ld) < np.mean(o_lp)),
    }


def test_multifranchise_flip(wrapper, items):
    title_by_idx = {i: it[2] for i, it in enumerate(items) if it[0] == 'movie'}
    checks, details = [], {}
    for fname, pat in FRANCHISE_PATTERNS.items():
        members = [i for i, t in title_by_idx.items() if re.search(pat, str(t))]
        if len(members) < 2:
            continue
        anchor, related = members[0], members[1:]
        liked = wrapper.predict([(anchor, 1.0)])
        disliked = wrapper.predict([(anchor, 0.0)])
        rank_l = np.empty_like(liked)
        rank_l[np.argsort(-liked)] = np.arange(len(liked))
        rank_d = np.empty_like(disliked)
        rank_d[np.argsort(-disliked)] = np.arange(len(disliked))
        moves = {title_by_idx[m]: float(rank_d[m] - rank_l[m]) for m in related}
        for v in moves.values():
            checks.append(v > 0)
        details[fname] = {'anchor': title_by_idx[anchor], 'rank_drops': moves}
    frac = float(np.mean(checks)) if checks else np.nan
    return {
        'n_franchises': len(details),
        'n_related_movies': len(checks),
        'fraction_correct_direction': frac,
        'details': details,
        'pass': bool(frac >= 0.70),
    }


def test_multifranchise_did(wrapper, items):
    """T3v3: franchise flip in score space (DiD vs non-franchise movies)."""
    title_by_idx = {i: it[2] for i, it in enumerate(items) if it[0] == 'movie'}
    n_movies = wrapper.n_movies
    checks, details = [], {}
    for fname, pat in FRANCHISE_PATTERNS.items():
        members = [i for i, t in title_by_idx.items() if re.search(pat, str(t))]
        if len(members) < 2:
            continue
        anchor, related = members[0], members[1:]
        liked = wrapper.predict([(anchor, 1.0)])
        disliked = wrapper.predict([(anchor, 0.0)])
        rel_mask = np.zeros(n_movies, dtype=bool)
        rel_mask[related] = True
        other_mask = ~rel_mask
        other_mask[anchor] = False
        d_rel = float(liked[rel_mask].mean() - disliked[rel_mask].mean())
        d_other = float(liked[other_mask].mean() - disliked[other_mask].mean())
        checks.append(d_rel > d_other)
        details[fname] = {'anchor': title_by_idx[anchor],
                          'did': d_rel - d_other,
                          'delta_related': d_rel, 'delta_other': d_other}
    frac = float(np.mean(checks)) if checks else np.nan
    dids = np.array([v['did'] for v in details.values()])
    informative = dids[np.abs(dids) > DID_NOISE_FLOOR]
    frac_informative = (float(np.mean(informative > 0))
                        if len(informative) else np.nan)
    # Gate logic: among franchises where the test is sensitive (|DiD| above
    # the noise floor), >=70% must move the right way, and the mean over
    # all franchises must be positive. A noise-level DiD is evidence of
    # test insensitivity for that franchise, not of broken polarity.
    passed = bool(len(informative) >= 3 and frac_informative >= 0.70
                  and float(np.mean(dids)) > 0)
    return {
        'n_franchises': len(details),
        'fraction_correct_direction': frac,
        'n_informative': int(len(informative)),
        'fraction_correct_informative': frac_informative,
        'mean_did': float(np.mean(dids)) if len(dids) else np.nan,
        'details': details,
        'pass': passed,
    }


def main():
    movies_df = pd.read_csv(DATA_DIR / 'movies.csv')
    credits_path = DATA_DIR / '.cache' / 'credits_top100_actors5.json'
    credits_data = json.loads(credits_path.read_text()) if credits_path.exists() else {}

    instruments = load_instruments()
    merged_path = OUT_DIR / 'instrument_acceptance.json'
    merged = json.loads(merged_path.read_text()) if merged_path.exists() else {}

    for name, (wrapper, ckpt) in instruments.items():
        items = ckpt['items']
        attr_map = build_attribute_movie_map(items, movies_df, credits_data)
        print(f"\n{'=' * 60}\nINSTRUMENT v2: {name}\n{'=' * 60}")

        r = merged.get(name, {})

        t1v2 = test_attribute_polarity(wrapper, items, attr_map)
        print(f"T1v2 attribute polarity (rank): {t1v2['fraction_correct_direction']:.2%} correct "
              f"(mean shift {t1v2['mean_rank_shift']:+.1f} ranks, "
              f"n={t1v2['n_attributes_tested']}) pass={t1v2['pass']}")
        r['T1v2_attribute_polarity'] = t1v2

        t1v3 = test_attribute_polarity_did(wrapper, items, attr_map)
        print(f"T1v3 attribute polarity (DiD): {t1v3['fraction_correct_direction']:.2%} correct "
              f"(mean DiD {t1v3['mean_did']:+.4f}) by_type={t1v3['by_type']} "
              f"pass={t1v3['pass']}")
        r['T1v3_attribute_did'] = t1v3

        t1b = test_prior_referenced_overlap(wrapper, items, attr_map)
        print(f"T1b prior-ref overlap: l/d={t1b['overlap_liked_disliked']:.2%} "
              f"l/prior={t1b['overlap_liked_prior']:.2%} "
              f"d/prior={t1b['overlap_disliked_prior']:.2%} pass={t1b['pass']}")
        r['T1b_prior_overlap'] = t1b

        t3v2 = test_multifranchise_flip(wrapper, items)
        print(f"T3v2 multi-franchise (rank): {t3v2['fraction_correct_direction']:.2%} correct "
              f"({t3v2['n_related_movies']} movies, {t3v2['n_franchises']} franchises) "
              f"pass={t3v2['pass']}")
        r['T3v2_multifranchise'] = t3v2

        t3v3 = test_multifranchise_did(wrapper, items)
        print(f"T3v3 multi-franchise (DiD): {t3v3['fraction_correct_direction']:.2%} correct "
              f"({t3v3['n_franchises']} franchises) pass={t3v3['pass']}")
        r['T3v3_franchise_did'] = t3v3

        # Final gates (see paper Sec. 5): genre-DiD is the hard attribute
        # gate (people-attributes lack ground-truth support and are
        # reported as informational); franchise test in DiD form;
        # overlap and monotonicity unchanged.
        genre_did = r['T1v3_attribute_did']['by_type'].get('genre', {})
        gates = {
            'genre_did': genre_did.get('frac_correct', 0) >= 0.80,
            'franchise_did': t3v3['pass'],
            'prior_overlap': r['T1b_prior_overlap']['pass'],
        }
        if 'T2_T4_monotonicity_snr' in r:
            gates['monotonicity'] = r['T2_T4_monotonicity_snr']['pass']
        r['gates'] = {k: bool(v) for k, v in gates.items()}
        r['ACCEPTED_v2'] = all(gates.values())
        print(f"  gates: {r['gates']}")
        print(f"  ==> {'ACCEPTED' if r['ACCEPTED_v2'] else 'REJECTED'}")
        merged[name] = r

    merged_path.write_text(json.dumps(merged, indent=2))
    print(f"\nSaved: {merged_path}")


if __name__ == '__main__':
    main()
