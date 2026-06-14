"""
Balanced cross-domain Amazon slate: EQUAL numbers of (mono-category) users
per category, removing the popularity 'head' that let static win on the
unbalanced cross-domain build. This is the real-items analog of the
synthetic indicator world: a heterogeneous catalog with no dominant user
segment, where adaptive routing should finally beat static.

Each user is assigned to their densest category; we take the top-N densest
users per category (equal N). Profile spans the union slate (640 items +
8 category tags); a user mostly answers their own category's items.

Run: poetry run python scripts/paper1/build_amazon_balanced.py
"""

import sys
sys.path.insert(0, '.'); sys.path.insert(0, 'scripts/paper1')
import json
from collections import defaultdict
from pathlib import Path
import numpy as np
import pandas as pd
from adaptivity_headroom import headroom, verdict

AZ = Path('C:/dev/phd/casper/data/amazon')
EXP = Path('C:/dev/phd/casper/experiments/paper1')
OUT_NPZ = AZ / 'amazon_balanced_profiles.npz'
CATEGORIES = ['All_Beauty', 'Appliances', 'Digital_Music', 'Gift_Cards',
              'Magazine_Subscriptions', 'Musical_Instruments', 'Software', 'Video_Games']
M_PER_CAT = 80
N_USERS_PER_CAT = 900     # balanced
MIN_IN_CAT = 6
SEED = 42


def main():
    item_index = {}; item_cat = []
    # per category: user -> {gi: liked}
    cat_user_items = {ci: defaultdict(dict) for ci in range(len(CATEGORIES))}
    for ci, cat in enumerate(CATEGORIES):
        f = AZ / f'{cat}.csv'
        if not f.exists() or f.stat().st_size < 1000:
            print(f"  skip {cat}"); continue
        df = pd.read_csv(f, usecols=['user_id', 'parent_asin', 'rating'])
        top = df['parent_asin'].value_counts().nlargest(M_PER_CAT).index
        for asin in top:
            if (cat, asin) not in item_index:
                item_index[(cat, asin)] = len(item_index); item_cat.append(ci)
        sub = df[df['parent_asin'].isin(top)]
        for r in sub.itertuples():
            gi = item_index[(cat, r.parent_asin)]
            cat_user_items[ci][r.user_id][gi] = 1 if r.rating >= 4 else 0
        print(f"  {cat}: {len(top)} items, {sub.shape[0]} ratings, "
              f"{len(cat_user_items[ci])} users", flush=True)
        del df, sub

    n_targets = len(item_index); n_items = n_targets + len(CATEGORIES)
    tag_idx = {ci: n_targets + ci for ci in range(len(CATEGORIES))}

    # assign each user to densest category; pick top-N densest per category
    user_best = {}                      # uid -> (ci, nitems)
    for ci, ui in cat_user_items.items():
        for uid, items in ui.items():
            if uid not in user_best or len(items) > user_best[uid][1]:
                user_best[uid] = (ci, len(items))
    by_cat = defaultdict(list)
    for uid, (ci, n) in user_best.items():
        if n >= MIN_IN_CAT:
            by_cat[ci].append((n, uid))
    rng = np.random.default_rng(SEED)
    chosen = []
    for ci in range(len(CATEGORIES)):
        lst = sorted(by_cat[ci], reverse=True)[:N_USERS_PER_CAT]
        chosen += [(uid, ci) for _, uid in lst]
        print(f"  cat {CATEGORIES[ci]}: {len(lst)} balanced users")
    rng.shuffle(chosen)

    profiles = []
    for uid, ci in chosen:
        vec = np.full(n_items, np.nan, dtype=np.float32)
        # gather this user's items across ALL categories (mostly their own)
        touched = set()
        for cj, ui in cat_user_items.items():
            if uid in ui:
                for gi, lk in ui[uid].items():
                    vec[gi] = float(lk)
                liked_any = any(v == 1 for v in ui[uid].values())
                vec[tag_idx[cj]] = 1.0 if liked_any else 0.0
                touched.add(cj)
        profiles.append(vec)
    profiles = np.stack(profiles)
    nt = (~np.isnan(profiles[:, :n_targets])).sum(1)
    print(f"balanced users: {len(profiles)}, mean items/user={nt.mean():.1f}, "
          f"slate={n_items} ({n_targets} targets + {len(CATEGORIES)} tags)")

    idx = rng.permutation(len(profiles)); split = int(0.8*len(idx))
    items_meta = ([('item', i, f'{CATEGORIES[item_cat[i]]}_{i}') for i in range(n_targets)] +
                  [('cattag', ci, f'CAT_{CATEGORIES[ci]}') for ci in range(len(CATEGORIES))])
    np.savez(OUT_NPZ, train=profiles[idx[:split]], test=profiles[idx[split:]],
             items=np.array(items_meta, dtype=object), n_targets=n_targets)

    samp = list(profiles[rng.choice(len(profiles), min(3000, len(profiles)), replace=False)])
    h = headroom(samp); h['verdict'] = verdict(h)
    allh = json.loads((EXP/'adaptivity_headroom.json').read_text())
    allh['amazon_balanced'] = h
    (EXP/'adaptivity_headroom.json').write_text(json.dumps(allh, indent=2))
    print(f"\nAMAZON BALANCED HEADROOM@10={h['headroom@10']:.3f} "
          f"jaccard={h['pair_jaccard_answerable']:.3f} ans_rate={h['answer_rate_mean']:.2f}")


if __name__ == '__main__':
    main()
