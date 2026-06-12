"""
Paper 1 / M4: Run the CASPER elicitation benchmark.

Usage (from casper root):
    poetry run python scripts/paper1/run_benchmark.py                 # free policies
    poetry run python scripts/paper1/run_benchmark.py --llm           # + LLM policies
    poetry run python scripts/paper1/run_benchmark.py --policies random,greedy_infogain
    poetry run python scripts/paper1/run_benchmark.py --n-users 50    # quick smoke

Outputs:
    experiments/paper1/benchmark_results.json      (summaries, merged across runs)
    experiments/paper1/episodes_<policy>.jsonl     (full per-episode logs)
"""

import sys
sys.path.insert(0, '.')
sys.path.insert(0, 'scripts/paper1')

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from test_instrument_lib import (CHECKPOINT_DIR, ExtrapolationModel,
                                 InstrumentWrapper)
from testbed import (build_profiles, get_user_splits, evaluate_policy,
                     evaluate_policy_concurrent, summarize)
import policies as P

OUT_DIR = Path('C:/dev/phd/casper/experiments/paper1')
SEED = 42
N_TURNS = 15


def load_instrument(name='instrument_v5_set'):
    from test_instrument_lib import load_instrument_by_name
    return load_instrument_by_name(name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n-users', type=int, default=200)
    ap.add_argument('--n-turns', type=int, default=N_TURNS)
    ap.add_argument('--llm', action='store_true', help='include LLM policies')
    ap.add_argument('--llm-model', default='gpt-4o-mini')
    ap.add_argument('--confirm-llm-cost', action='store_true',
                    help='required with --llm: confirms API spend is approved')
    ap.add_argument('--policies', default=None,
                    help='comma-separated subset of policy names to run')
    ap.add_argument('--instrument', default='instrument_v5_set')
    args = ap.parse_args()

    instrument, ckpt = load_instrument(args.instrument)
    items = ckpt['items']
    print(f"Instrument: {args.instrument} (val_loss={ckpt.get('val_loss'):.4f})")

    train_users, _, testbed_users = get_user_splits(items)
    rng = np.random.default_rng(SEED)
    eval_users = rng.choice(testbed_users, size=args.n_users, replace=False)

    print("Building profiles (testbed users + train sample for stats)...")
    t0 = time.time()
    eval_profiles = build_profiles(items, user_ids=eval_users,
                                   attr_min_support=3, taste_margin=None)
    stat_users = rng.choice(train_users, size=4000, replace=False)
    stat_profiles = build_profiles(items, user_ids=stat_users,
                                   attr_min_support=3, taste_margin=None)
    p_rated = P.compute_p_rated(stat_profiles, stat_users)
    print(f"  {time.time() - t0:.0f}s | eval profiles: {len(eval_profiles)}")

    all_policies = {
        'random': lambda: P.RandomPolicy(items),
        'popularity': lambda: P.PopularityPolicy(items, p_rated),
        'greedy_infogain': lambda: P.GreedyInfoGainPolicy(items, p_rated),
        'scpr_entropy': lambda: P.SCPREntropyPolicy(items, p_rated),
        'thompson': lambda: P.ThompsonPolicy(items),
    }
    dqn_path = OUT_DIR / 'dqn_policy.pt'
    if dqn_path.exists():
        all_policies['dqn'] = lambda: P.DQNPolicy(items, dqn_path)
    ppo_path = Path('C:/dev/phd/casper/experiments/paper2/discrete_ppo.pt')
    if ppo_path.exists():
        all_policies['ppo'] = lambda: P.PPOPolicy(items, ppo_path)
    botplay_path = OUT_DIR / 'botplay_policy.pt'
    if botplay_path.exists():
        all_policies['botplay_rl'] = lambda: P.BotPlayPolicy(items, botplay_path)
    botplay_v2_path = OUT_DIR / 'botplay_policy_v2.pt'
    if botplay_v2_path.exists():
        all_policies['botplay_rl_v2'] = (
            lambda: P.BotPlayV2Policy(items, botplay_v2_path))
    if args.llm:
        if not args.confirm_llm_cost:
            raise SystemExit('LLM runs are ON HOLD: pass --confirm-llm-cost '
                             'only after the user approves the API budget.')
        mtag = args.llm_model.replace('/', '-')
        for style in ('vanilla', 'strategist', 'gate'):
            all_policies[f'llm_{style}_{mtag}'] = (
                lambda s=style: P.LLMPolicy(items, style=s, model=args.llm_model))

    if args.policies:
        wanted = set(args.policies.split(','))
        all_policies = {k: v for k, v in all_policies.items() if k in wanted}

    results_path = OUT_DIR / 'benchmark_results.json'
    results = json.loads(results_path.read_text()) if results_path.exists() else {}

    for pname, factory in all_policies.items():
        print(f"\n=== {pname} ({args.n_users} users x {args.n_turns} turns) ===")
        t0 = time.time()
        if pname.startswith('llm'):
            registry = []
            logs = evaluate_policy_concurrent(factory, list(eval_users),
                                              eval_profiles, instrument,
                                              n_turns=args.n_turns, seed=SEED,
                                              policy_registry=registry)
            policy = factory()  # for metadata shape
            policy.parse_failures = sum(p.parse_failures for p in registry)
            policy.calls = sum(p.calls for p in registry)
            with open(OUT_DIR / f'llm_transcripts_{pname}.jsonl', 'w') as tf:
                for pol in registry:
                    for rec in pol.transcript:
                        tf.write(json.dumps(rec) + '\n')
        else:
            policy = factory()
            logs = evaluate_policy(policy, list(eval_users), eval_profiles,
                                   instrument, n_turns=args.n_turns, seed=SEED)
        summary = summarize(logs, args.n_turns)
        summary['wall_seconds'] = round(time.time() - t0, 1)
        if hasattr(policy, 'parse_failures'):
            summary['llm_parse_failures'] = policy.parse_failures
            summary['llm_calls'] = policy.calls
        results[pname] = summary
        print(f"  turn0={summary['turn0_accuracy']:.4f} "
              f"final={summary['final_accuracy']:.4f} "
              f"CI={summary['final_accuracy_ci95']} "
              f"AUAC={summary['auac']:.4f} hit={summary['hit_rate']:.2%}")

        with open(OUT_DIR / f'episodes_{pname}.jsonl', 'w') as f:
            for lg in logs:
                f.write(json.dumps(lg.to_dict()) + '\n')

    results['_meta'] = {
        'instrument': args.instrument,
        'n_users': args.n_users,
        'n_turns': args.n_turns,
        'seed': SEED,
    }
    results_path.write_text(json.dumps(results, indent=2))
    print(f"\nSaved {results_path}")

    print(f"\n{'policy':<18} {'final acc':<12} {'AUAC':<10} {'hit rate':<10}")
    print('-' * 50)
    for pname, s in results.items():
        if pname.startswith('_'):
            continue
        if 'final_accuracy' not in s:
            continue
        print(f"{pname:<18} {s['final_accuracy']:<12.4f} {s['auac']:<10.4f} "
              f"{s.get('hit_rate', float('nan')):<10.2%}")


if __name__ == '__main__':
    main()
