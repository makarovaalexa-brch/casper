"""
Paper 1 / A8: benchmark on Slate 2 (300 mid-popularity movies + genres +
genome tags), dual-head instrument.

First wave: the five non-learned policies (random, popularity, greedy
info-gain, SCPR-entropy, Thompson) -- all instrument-generic. Learned
policies (PPO/DQN/bot-play) require slate-2 training runs and join in a
second wave.

Usage (from casper root):
    poetry run python scripts/paper1/run_benchmark_slate2.py [--n-users 200]
Outputs:
    experiments/paper1/benchmark_slate2.json
    experiments/paper1/s2_episodes_<policy>.jsonl
"""

import sys
sys.path.insert(0, '.')
sys.path.insert(0, 'scripts/paper1')

import argparse
import json
import time
from pathlib import Path

import numpy as np

from test_instrument_lib import load_instrument_by_name
from train_instrument_slate2 import (build_slate, build_profiles_slate2,
                                     MIN_USER_RATINGS)
from testbed import evaluate_policy, summarize
import policies as P

EXP = Path('C:/dev/phd/casper/experiments/paper1')
SEED = 42
INSTRUMENT_VAL_CAP = 1500


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n-users', type=int, default=200)
    ap.add_argument('--n-turns', type=int, default=15)
    ap.add_argument('--policies', default=None)
    args = ap.parse_args()

    instrument, ckpt = load_instrument_by_name('instrument_slate2_dual')
    items = ckpt['items']
    n_items = len(items)
    print(f"Slate 2: {n_items} items, instrument val_loss={ckpt['val_loss']:.4f}")

    print("Building slate + splits...")
    _, n_movies, movie_pos, attr_of_movie, ratings_f = build_slate()
    user_counts = ratings_f.groupby('userId').size()
    dense = user_counts[user_counts >= MIN_USER_RATINGS].index.to_numpy()
    np.random.seed(SEED)
    shuffled = dense.copy()
    np.random.shuffle(shuffled)
    split = int(0.8 * len(shuffled))
    train_users = shuffled[:split]
    testbed_users = shuffled[split + INSTRUMENT_VAL_CAP:]
    rng = np.random.default_rng(SEED)
    eval_users = rng.choice(testbed_users, size=args.n_users, replace=False)
    stat_users = rng.choice(train_users, size=4000, replace=False)

    print("Building profiles...")
    t0 = time.time()
    eval_profiles = build_profiles_slate2(items, n_items, movie_pos,
                                          attr_of_movie, ratings_f, eval_users)
    stat_profiles = build_profiles_slate2(items, n_items, movie_pos,
                                          attr_of_movie, ratings_f, stat_users)
    p_rated = P.compute_p_rated(stat_profiles, stat_users)
    print(f"  {time.time() - t0:.0f}s | eval profiles: {len(eval_profiles)}")

    all_policies = {
        'random': lambda: P.RandomPolicy(items),
        'popularity': lambda: P.PopularityPolicy(items, p_rated),
        'greedy_infogain': lambda: P.GreedyInfoGainPolicy(items, p_rated),
        'scpr_entropy': lambda: P.SCPREntropyPolicy(items, p_rated),
        'thompson': lambda: P.ThompsonPolicy(items),
    }
    dqn2 = EXP / 'dqn_policy_slate2.pt'
    if dqn2.exists():
        all_policies['dqn'] = lambda: P.DQNPolicy(items, dqn2)
    ppo2 = Path('C:/dev/phd/casper/experiments/paper2/discrete_ppo_slate2.pt')
    if ppo2.exists():
        all_policies['ppo'] = lambda: P.PPOPolicy(items, ppo2)
    ppo2d = Path('C:/dev/phd/casper/experiments/paper2/discrete_ppo_slate2_dual.pt')
    if ppo2d.exists():
        all_policies['ppo_dual'] = lambda: P.NetPolicy(
            items, ppo2d, 'ppo', 'ppo_dual')
    bp2d = EXP / 'botplay_slate2_dual.pt'
    if bp2d.exists():
        all_policies['botplay_dual'] = lambda: P.NetPolicy(
            items, bp2d, 'reinforce', 'botplay_dual')
    bp2 = EXP / 'botplay_slate2.pt'
    if bp2.exists():
        all_policies['botplay'] = lambda: P.NetPolicy(
            items, bp2, 'reinforce', 'botplay')
    if args.policies:
        wanted = set(args.policies.split(','))
        all_policies = {k: v for k, v in all_policies.items() if k in wanted}

    results_path = EXP / 'benchmark_slate2.json'
    results = json.loads(results_path.read_text()) if results_path.exists() else {}

    for pname, factory in all_policies.items():
        print(f"\n=== slate2 {pname} ({args.n_users} users x {args.n_turns}) ===")
        policy = factory()
        t0 = time.time()
        logs = evaluate_policy(policy, list(eval_users), eval_profiles,
                               instrument, n_turns=args.n_turns, seed=SEED)
        summary = summarize(logs, args.n_turns)
        summary['wall_seconds'] = round(time.time() - t0, 1)
        results[pname] = summary
        print(f"  turn0={summary['turn0_accuracy']:.4f} "
              f"final={summary['final_accuracy']:.4f} "
              f"AUAC={summary['auac']:.4f} answer={summary['hit_rate']:.0%}")
        with open(EXP / f's2_episodes_{pname}.jsonl', 'w') as f:
            for lg in logs:
                f.write(json.dumps(lg.to_dict()) + '\n')

    results['_meta'] = {'instrument': 'instrument_slate2_dual',
                        'n_users': args.n_users, 'n_turns': args.n_turns,
                        'seed': SEED}
    results_path.write_text(json.dumps(results, indent=2))
    print(f"\nSaved {results_path}")
    for pname, s in results.items():
        if pname.startswith('_'):
            continue
        print(f"{pname:<18} final={s['final_accuracy']:.4f} "
              f"AUAC={s['auac']:.4f} answer={s['hit_rate']:.0%}")


if __name__ == '__main__':
    main()
