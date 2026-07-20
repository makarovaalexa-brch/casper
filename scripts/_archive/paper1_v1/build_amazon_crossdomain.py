"""
Build a CROSS-DOMAIN Amazon slate (the disjoint + non-bridgeable structure
that should let adaptive elicitation win). 8 categories; slate = top-M
items per category (recommendation targets) + 1 category-indicator tag per
category (the routing-question attributes). Binarise rating>=4 liked,
<=3 disliked. A category tag is 'liked' if the user has >=1 liked item in
it, 'disliked' if they interacted but liked none, else unknown.

Runs Tier-0 headroom inline (cheap) and saves profiles npz. Training is a
separate decision based on the headroom + density it reports.

Run: poetry run python scripts/paper1/build_amazon_crossdomain.py
"""

import sys
sys.path.insert(0, '.')
sys.path.insert(0, 'scripts/paper1')

import json
from pathlib import Path

import numpy as np
import pandas as pd

from adaptivity_headroom import headroom, verdict

AZ = Path('C:/dev/phd/casper/data/amazon')
EXP = Path('C:/dev/phd/casper/experiments/paper1')
OUT_NPZ = AZ / 'amazon_crossdomain_profiles.npz'
CATEGORIES = ['All_Beauty', 'Appliances', 'Digital_Music', 'Gift_Cards',
              'Magazine_Subscriptions', 'Musical_Instruments', 'Software',
              'Video_Games']
M_PER_CAT = 80          # top items per category (targets)
MIN_USER_ITEMS = 6      # keep users with >= this many slate-item ratings
SEED = 42


def main():
    item_index = {}            # (cat, asin) -> global idx
    item_cat = []              # global idx -> category id
    cat_tag_idx = {}           # cat id -> tag slate idx (assigned after items)
    # user -> {global_item_idx: liked(1/0)} and per-cat interaction tracking
    from collections import defaultdict
    user_items = defaultdict(dict)
    user_cat_liked = defaultdict(lambda: defaultdict(lambda: [0, 0]))  # cat->[liked,inter]

    for ci, cat in enumerate(CATEGORIES):
        f = AZ / f'{cat}.csv'
        if not f.exists() or f.stat().st_size < 1000:
            print(f"  skip {cat} (missing)"); continue
        df = pd.read_csv(f, usecols=['user_id', 'parent_asin', 'rating'])
        top = df['parent_asin'].value_counts().nlargest(M_PER_CAT).index
        sub = df[df['parent_asin'].isin(top)]
        for asin in top:
            if (cat, asin) not in item_index:
                item_index[(cat, asin)] = len(item_index)
                item_cat.append(ci)
        for r in sub.itertuples():
            gi = item_index[(cat, r.parent_asin)]
            lk = 1 if r.rating >= 4 else 0
            user_items[r.user_id][gi] = lk
            cl = user_cat_liked[r.user_id][ci]
            cl[1] += 1; cl[0] += lk
        print(f"  {cat}: {len(top)} items, {len(sub)} slate ratings, "
              f"users now {len(user_items)}", flush=True)
        del df, sub

    n_items_targets = len(item_index)
    # append category tags
    for ci in range(len(CATEGORIES)):
        cat_tag_idx[ci] = n_items_targets + ci
    n_items = n_items_targets + len(CATEGORIES)
    print(f"slate: {n_items} ({n_items_targets} item targets + "
          f"{len(CATEGORIES)} category tags)")

    profiles = []
    multi_cat = 0
    for uid, items in user_items.items():
        if len(items) < MIN_USER_ITEMS:
            continue
        vec = np.full(n_items, np.nan, dtype=np.float32)
        for gi, lk in items.items():
            vec[gi] = float(lk)
        cats_touched = 0
        for ci, (liked, inter) in user_cat_liked[uid].items():
            if inter > 0:
                cats_touched += 1
                vec[cat_tag_idx[ci]] = 1.0 if liked >= 1 else 0.0
        if cats_touched >= 2:
            multi_cat += 1
        profiles.append(vec)

    profiles = np.stack(profiles)
    print(f"kept {len(profiles)} users (>={MIN_USER_ITEMS} items); "
          f"{multi_cat} multi-category ({multi_cat/len(profiles):.1%})")
    print(f"mean items/user = {np.mean((~np.isnan(profiles[:, :n_items_targets])).sum(1)):.1f}, "
          f"mean liked/user = {np.mean((profiles[:, :n_items_targets]==1).sum(1)):.1f}")

    rng = np.random.default_rng(SEED)
    idx = rng.permutation(len(profiles))
    split = int(0.8 * len(idx))
    train, test = profiles[idx[:split]], profiles[idx[split:]]
    items_meta = ([('item', i, f'{CATEGORIES[item_cat[i]]}_{i}') for i in range(n_items_targets)] +
                  [('cattag', ci, f'CAT_{CATEGORIES[ci]}') for ci in range(len(CATEGORIES))])
    np.savez(OUT_NPZ, train=train, test=test,
             items=np.array(items_meta, dtype=object), n_targets=n_items_targets)

    # Tier-0 headroom (cap users for speed)
    samp = list(profiles[rng.choice(len(profiles), min(3000, len(profiles)), replace=False)])
    h = headroom(samp); h['verdict'] = verdict(h)
    h['n_targets'] = n_items_targets; h['multi_cat_frac'] = multi_cat / len(profiles)
    allh = json.loads((EXP / 'adaptivity_headroom.json').read_text()) \
        if (EXP / 'adaptivity_headroom.json').exists() else {}
    allh['amazon_crossdomain'] = h
    (EXP / 'adaptivity_headroom.json').write_text(json.dumps(allh, indent=2))
    print(f"\nAMAZON CROSS-DOMAIN HEADROOM@10 = {h['headroom@10']:.3f} "
          f"(jaccard={h['pair_jaccard_answerable']:.3f}, "
          f"static@10={h['static_cov@10']:.3f}, adaptive@10={h['adaptive_cov@10']:.3f})")
    print(f"  ans_rate={h['answer_rate_mean']:.2f}  => {h['verdict']}")
    print("  (anchors: slate1 0.169, slate2 0.241, lastfm 0.685, synthetic 0.498)")


if __name__ == '__main__':
    main()
