"""
Yelp multi-city slate — the best real-data adaptivity demonstrator:
cities are SIZE-BALANCED and GEOGRAPHICALLY DISJOINT (a user truly cannot
rate another city's businesses), unlike size-skewed Amazon categories.
This is the real-items analog of the synthetic indicator world, but with a
NON-popularity-biased construction across balanced segments.

Slate = top-M businesses per city (targets) + 1 city-indicator tag per city
(routing question). Profiles: stars>=4 liked, <=3 disliked. A city tag is
liked if the user has >=1 liked business in it, disliked if interacted but
none liked, else unknown. Users assigned to their densest city; balanced
N per city.

Streams the 5GB review json (never loads whole). Run:
  poetry run python scripts/paper1/build_yelp_multicity.py
"""

import sys
sys.path.insert(0, '.'); sys.path.insert(0, 'scripts/paper1')
import json
from collections import defaultdict
from pathlib import Path
import numpy as np
from adaptivity_headroom import headroom, verdict

Y = Path('C:/dev/phd/casper/data/yelp')
EXP = Path('C:/dev/phd/casper/experiments/paper1')
REVIEW = Y / 'yelp_academic_dataset_review.json'
BUSINESS = Y / 'yelp_academic_dataset_business.json'
OUT = Y / 'yelp_multicity_profiles.npz'
N_CITIES = 8
M_PER_CITY = 80
N_USERS_PER_CITY = 1200
MIN_ITEMS = 6
SEED = 42
rng = np.random.default_rng(SEED)


def main():
    # 1. business -> (city, state); pick top cities by #businesses
    biz_city = {}
    city_count = defaultdict(int)
    with open(BUSINESS, encoding='utf-8') as f:
        for line in f:
            b = json.loads(line)
            c = f"{b.get('city','?')}, {b.get('state','?')}"
            biz_city[b['business_id']] = c
            city_count[c] += 1
    cities = [c for c, _ in sorted(city_count.items(), key=lambda x: -x[1])[:N_CITIES]]
    cidx = {c: i for i, c in enumerate(cities)}
    print(f"top {N_CITIES} cities: " + "; ".join(f"{c}({city_count[c]})" for c in cities))

    # 2. stream reviews: per city, count business popularity + gather (user,biz,star)
    #    first pass: business popularity within target cities
    biz_pop = defaultdict(int)
    with open(REVIEW, encoding='utf-8') as f:
        for line in f:
            r = json.loads(line)
            c = biz_city.get(r['business_id'])
            if c in cidx:
                biz_pop[r['business_id']] += 1
    # top-M businesses per city
    per_city_biz = defaultdict(list)
    for bid, n in biz_pop.items():
        per_city_biz[biz_city[bid]].append((n, bid))
    slate_biz = {}
    item_city = []
    for c in cities:
        top = sorted(per_city_biz[c], reverse=True)[:M_PER_CITY]
        for _, bid in top:
            slate_biz[bid] = len(slate_biz); item_city.append(cidx[c])
    n_targets = len(slate_biz)
    tag_idx = {ci: n_targets + ci for ci in range(len(cities))}
    n_items = n_targets + len(cities)
    print(f"slate: {n_items} ({n_targets} businesses + {len(cities)} city tags)")

    # 3. second pass: user -> {gi: liked}, and per-city liked/interacted
    user_items = defaultdict(dict)
    user_city = defaultdict(lambda: defaultdict(lambda: [0, 0]))  # ci->[liked,inter]
    with open(REVIEW, encoding='utf-8') as f:
        for line in f:
            r = json.loads(line)
            gi = slate_biz.get(r['business_id'])
            if gi is None:
                continue
            lk = 1 if r['stars'] >= 4 else 0
            user_items[r['user_id']][gi] = lk
            ci = item_city[gi]
            cl = user_city[r['user_id']][ci]; cl[1] += 1; cl[0] += lk

    # 4. assign users to densest city; balanced N per city
    best = {}
    for uid, items in user_items.items():
        if len(items) < MIN_ITEMS:
            continue
        ci = max(user_city[uid].items(), key=lambda kv: kv[1][1])[0]
        best.setdefault(ci, []).append((len(items), uid))
    chosen = []
    for ci in range(len(cities)):
        lst = sorted(best.get(ci, []), reverse=True)[:N_USERS_PER_CITY]
        chosen += [uid for _, uid in lst]
        print(f"  {cities[ci]}: {len(lst)} users")
    rng.shuffle(chosen)

    profiles = []
    for uid in chosen:
        vec = np.full(n_items, np.nan, dtype=np.float32)
        for gi, lk in user_items[uid].items():
            vec[gi] = float(lk)
        for ci, (liked, inter) in user_city[uid].items():
            if inter > 0:
                vec[tag_idx[ci]] = 1.0 if liked >= 1 else 0.0
        profiles.append(vec)
    profiles = np.stack(profiles)
    nt = (~np.isnan(profiles[:, :n_targets])).sum(1)
    print(f"users={len(profiles)} mean items/user={nt.mean():.1f}")

    idx = rng.permutation(len(profiles)); split = int(0.8*len(idx))
    items_meta = ([('biz', i, f'{cities[item_city[i]]}_{i}') for i in range(n_targets)] +
                  [('citytag', ci, f'CITY_{cities[ci]}') for ci in range(len(cities))])
    np.savez(OUT, train=profiles[idx[:split]], test=profiles[idx[split:]],
             items=np.array(items_meta, dtype=object), n_targets=n_targets)

    samp = list(profiles[rng.choice(len(profiles), min(3000, len(profiles)), replace=False)])
    h = headroom(samp); h['verdict'] = verdict(h)
    allh = json.loads((EXP/'adaptivity_headroom.json').read_text())
    allh['yelp_multicity'] = h
    (EXP/'adaptivity_headroom.json').write_text(json.dumps(allh, indent=2))
    print(f"\nYELP MULTI-CITY HEADROOM@10={h['headroom@10']:.3f} "
          f"jaccard={h['pair_jaccard_answerable']:.3f} ans_rate={h['answer_rate_mean']:.2f}")


if __name__ == '__main__':
    main()
