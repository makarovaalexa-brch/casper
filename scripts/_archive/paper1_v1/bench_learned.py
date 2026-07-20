"""
Benchmark trained LEARNED policies (PPO, DQN, bot-play) on an npz world,
merging rows into benchmark_<name>.json. Run after the trainers produce
checkpoints. DATASET_NAME selects the world/instrument.

Usage: DATASET_NAME=ml_stratified poetry run python scripts/paper1/bench_learned.py
"""
import os, sys, json
sys.path.insert(0, '.'); sys.path.insert(0, 'scripts/paper1'); sys.path.insert(0, 'scripts/paper2')
from collections import Counter
from pathlib import Path
import numpy as np
from test_instrument_lib import load_instrument_by_name
from testbed import SimulatedUser, run_episode, summarize
import policies as P

NAME = os.environ['DATASET_NAME']
EXP = Path('C:/dev/phd/casper/experiments/paper1')
P2 = Path('C:/dev/phd/casper/experiments/paper2')
NPZ = {'ml_stratified': 'C:/dev/phd/casper/data/movielens/ml_stratified_profiles.npz',
       'ml1m': 'C:/dev/phd/casper/data/movielens/ml1m_profiles.npz',
       'yelp_multicity': 'C:/dev/phd/casper/data/yelp/yelp_multicity_profiles.npz'}[NAME]
N_TURNS = 15; SEED = 42


def main():
    w, _ = load_instrument_by_name(f'instrument_{NAME}')
    d = np.load(NPZ, allow_pickle=True)
    items = [tuple(x) for x in d['items'].tolist()]
    test = d['test'][:300]

    cands = {}
    pp = P2 / f'discrete_ppo_{NAME}_dual.pt'
    if pp.exists(): cands['ppo'] = lambda: P.NetPolicy(items, pp, 'ppo', 'ppo')
    dq = EXP / f'dqn_policy_{NAME}.pt'
    if dq.exists(): cands['dqn'] = lambda: P.DQNPolicy(items, dq)
    bp = EXP / f'botplay_{NAME}_dual.pt'
    if bp.exists(): cands['botplay'] = lambda: P.NetPolicy(items, bp, 'reinforce', 'botplay')

    res_path = EXP / f'benchmark_{NAME}.json'
    results = json.loads(res_path.read_text()) if res_path.exists() else {}
    for name, fac in cands.items():
        pol = fac(); logs = []
        for i, prof in enumerate(test):
            logs.append(run_episode(pol, SimulatedUser(i, prof), w, N_TURNS,
                                    np.random.default_rng((SEED*7+i) % 2**31)))
        s = summarize(logs, N_TURNS)
        t1 = Counter(l.questions[0] for l in logs if l.questions)
        br = {}
        for l in logs:
            if len(l.questions) >= 2: br.setdefault(l.answers[0], Counter())[l.questions[1]] += 1
        s['branches'] = len({c.most_common(1)[0][0] for c in br.values()}) > 1
        results[name] = s
        with open(EXP / f'episodes_{NAME}_{name}.jsonl', 'w') as fo:
            for l in logs: fo.write(json.dumps(l.to_dict()) + '\n')
        print(f"  {name:<10} final={s['final_accuracy']:.4f} AUAC={s['auac']:.4f} "
              f"answer={s['hit_rate']:.0%} branches={s['branches']}", flush=True)
    res_path.write_text(json.dumps(results, indent=2))
    print(f"merged into {res_path}")


if __name__ == '__main__':
    main()
