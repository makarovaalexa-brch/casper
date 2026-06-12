"""
Bot-play REINFORCE (Makarova et al. recipe, v2 improvements) on any
world/state-mode combination -- the user's algorithm with dual beliefs.

Env vars: CASPER_WORLD=slate1|slate2, CASPER_DUAL=0|1
Run from casper root: poetry run python scripts/paper1/train_botplay_dual.py
Output: experiments/paper1/botplay_<world><_dual>.pt
"""

import sys
sys.path.insert(0, '.')
sys.path.insert(0, 'scripts/paper1')
sys.path.insert(0, 'scripts/paper2')

import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from env import ElicitationEnv, load_world

WORLD = os.environ.get('CASPER_WORLD', 'slate1')
DUAL = os.environ.get('CASPER_DUAL', '0') == '1'
_suffix = ('' if WORLD == 'slate1' else f'_{WORLD}') + ('_dual' if DUAL else '')
OUT_PATH = Path(f'C:/dev/phd/casper/experiments/paper1/botplay{_suffix}.pt')

SEED = 42
N_EPISODES = 30000
N_TURNS = 15
LR = 5e-4
GAMMA = 0.97
BETA_START, BETA_END = 0.05, 0.005
EPS_START, EPS_END = 0.20, 0.02
VAL_EVERY = 2000
LOG_EVERY = 2000

torch.manual_seed(SEED)
np.random.seed(SEED)


def main():
    world = load_world(seed=SEED, world=WORLD)
    instrument, n_items = world['instrument'], world['n_items']
    belief = None
    if DUAL and WORLD == 'slate1':
        from test_instrument_lib import load_instrument_by_name
        belief, _ = load_instrument_by_name('instrument_s1_dual')
    env = ElicitationEnv(instrument, n_items, N_TURNS,
                         belief_instrument=belief,
                         state_mode='dual' if DUAL else 'liked')
    state_dim = env.state_dim
    rng = np.random.default_rng(SEED)

    net = nn.Sequential(nn.Linear(state_dim, 512), nn.ReLU(),
                        nn.Linear(512, 256), nn.ReLU(),
                        nn.Linear(256, n_items))
    opt = torch.optim.Adam(net.parameters(), lr=LR)
    print(f"world={WORLD} dual={DUAL} state_dim={state_dim} "
          f"params={sum(p.numel() for p in net.parameters()):,}", flush=True)

    def argmax_auac(uids):
        net.eval()
        aucs = []
        for uid in uids:
            s = env.reset(world['profiles'][uid])
            accs = [env.episode_accuracy()]
            for _ in range(N_TURNS):
                with torch.no_grad():
                    logits = net(torch.from_numpy(s).unsqueeze(0))[0]
                mask = torch.full((n_items,), float('-inf'))
                rem = [i for i in range(n_items) if i not in env.asked]
                mask[rem] = 0.0
                s, _, done = env.step(int(torch.argmax(logits + mask).item()))
                accs.append(env.episode_accuracy())
                if done:
                    break
            aucs.append(float(np.mean(accs)))
        net.train()
        return float(np.mean(aucs))

    baselines = np.zeros(N_TURNS)
    best, ep_rewards = -1.0, []
    t0 = time.time()
    for ep in range(N_EPISODES):
        frac = ep / N_EPISODES
        beta = BETA_START + (BETA_END - BETA_START) * frac
        eps = EPS_START + (EPS_END - EPS_START) * frac
        uid = world['train_uids'][int(rng.integers(len(world['train_uids'])))]
        s = env.reset(world['profiles'][uid])
        lps, ents, rews = [], [], []
        for t in range(N_TURNS):
            logits = net(torch.from_numpy(s).unsqueeze(0))[0]
            mask = torch.full((n_items,), float('-inf'))
            rem = [i for i in range(n_items) if i not in env.asked]
            mask[rem] = 0.0
            logp = torch.log_softmax(logits + mask, dim=0)
            probs = logp.exp()
            a = int(rng.choice(rem)) if rng.random() < eps else \
                int(torch.multinomial(probs, 1).item())
            lps.append(logp[a])
            ents.append(-(probs * logp.clamp(min=-30)).sum())
            s, r, _ = env.step(a)
            rews.append(r)
        g = 0.0
        rets = np.zeros(N_TURNS)
        for t in reversed(range(N_TURNS)):
            g = rews[t] + GAMMA * g
            rets[t] = g
        terms = []
        for t in range(N_TURNS):
            adv = rets[t] - baselines[t]
            baselines[t] = 0.995 * baselines[t] + 0.005 * rets[t]
            terms.append(-lps[t] * adv - beta * ents[t])
        opt.zero_grad()
        torch.stack(terms).sum().backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
        opt.step()
        ep_rewards.append(sum(rews))

        if (ep + 1) % LOG_EVERY == 0:
            print(f"Ep {ep + 1:6d}/{N_EPISODES} | reward "
                  f"{np.mean(ep_rewards[-LOG_EVERY:]):+.4f} | "
                  f"{time.time() - t0:.0f}s", flush=True)
        if (ep + 1) % VAL_EVERY == 0:
            v = argmax_auac(world['val_uids'])
            mark = ''
            if v > best:
                best = v
                mark = ' *'
                torch.save({'policy_state_dict': net.state_dict(),
                            'state_mode': 'dual' if DUAL else 'liked',
                            'state_dim': state_dim, 'world': WORLD,
                            'n_items': n_items, 'episodes': ep + 1,
                            'val_auac': v}, OUT_PATH)
            print(f"  VAL AUAC {v:.4f}{mark}", flush=True)

    print(f"Best val {best:.4f}; saved {OUT_PATH}")


if __name__ == '__main__':
    main()
