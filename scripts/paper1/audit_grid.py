"""
Audit grid: testbed (rows) x model/baseline (cols), AUAC where available.
Compiles all saved benchmark JSONs into one table; marks missing as '-'.
CASPER (greedy-distill + RL-finetune equivariant) is not in the JSONs (separate
checkpoints) -> filled from known eval numbers in code below.
"""
import json, os
EXP = 'C:/dev/phd/casper/experiments/paper1/'

TESTBEDS = [
    ('synthetic(indicator)', 'synthetic_sanity.json', 'AUAC'),
    ('slate1(popular)', 'benchmark_results.json', 'AUAC'),
    ('slate2', 'benchmark_slate2.json', 'AUAC'),
    ('ml_stratified', 'benchmark_ml_stratified.json', 'AUAC(held)'),
    ('yelp_multicity', 'benchmark_yelp_multicity.json', 'AUAC'),
    ('amazon_crossdom', 'benchmark_amazon_crossdomain.json', 'AUAC'),
    ('amazon_balanced', 'benchmark_amazon_balanced.json', 'AUAC'),
    ('lastfm', 'benchmark_lastfm.json', 'AUAC'),
    ('lastfm_tercile', 'benchmark_lastfm_tercile.json', 'AUAC'),
]
COLS = ['random', 'popularity', 'greedy_infogain', 'scpr', 'answerability',
        'thompson', 'PPO', 'DQN', 'bot-play', 'CASPER']
ALIAS = {
    'random': ['random'], 'popularity': ['popularity'],
    'greedy_infogain': ['greedy_infogain'], 'scpr': ['scpr_entropy'],
    'answerability': ['greedy_answerability'], 'thompson': ['thompson'],
    'PPO': ['ppo', 'ppo_dual', 'discrete_ppo'],
    'DQN': ['dqn'],
    'bot-play': ['botplay', 'botplay_dual', 'botplay_rl_v2', 'botplay_rl',
                 'reinforce_v3', 'reinforce_v2'],
}
# CASPER known eval results (not in JSONs):
CASPER = {'ml_stratified': 0.726, 'ml1m(Hit@10)': 0.507}


def val(d, aliases):
    best = None
    for a in aliases:
        if a in d and isinstance(d[a], dict):
            v = d[a].get('auac_heldout', d[a].get('auac'))
            if v is not None and (best is None or v > best):
                best = v
    return best


def main():
    grid = {}
    for name, f, metric in TESTBEDS:
        p = EXP + f
        grid[name] = {'_metric': metric}
        if not os.path.exists(p):
            continue
        d = json.load(open(p))
        for c in COLS:
            if c == 'CASPER':
                grid[name][c] = CASPER.get(name)
            else:
                grid[name][c] = val(d, ALIAS[c])
    # print grid
    w = 16
    hdr = f"{'testbed':<20}" + "".join(f"{c[:9]:>10}" for c in COLS) + "  metric"
    print(hdr); print("-" * len(hdr))
    for name, _, metric in TESTBEDS:
        row = f"{name:<20}"
        for c in COLS:
            v = grid[name].get(c)
            row += f"{(f'{v:.3f}' if isinstance(v, float) else '-'):>10}"
        print(row + f"  {metric}")
    # ml1m (different metric, hard-LOO Hit@10, from fair same-subset run)
    ml1m = {'random': 0.477, 'popularity': 0.498, 'thompson': 0.485,
            'HELF': 0.509, 'CASPER': 0.507}
    print("\nml1m (Hit@10 hard-LOO, fair same-subset): " +
          ", ".join(f"{k}={v}" for k, v in ml1m.items()))


if __name__ == '__main__':
    main()
