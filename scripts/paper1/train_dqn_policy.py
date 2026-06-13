"""
Paper 1 / B2: UNICORN-style value-based baseline -- dueling double DQN
over the entity slate (Deng et al., SIGIR 2021 family), trained on the
Paper 2 shared environment (instrument v5, canonical labels).

Run from casper root: poetry run python scripts/paper1/train_dqn_policy.py
Output: experiments/paper1/dqn_policy.pt
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
_suffix = '' if WORLD == 'slate1' else f'_{WORLD}'
OUT_PATH = Path(f'C:/dev/phd/casper/experiments/paper1/dqn_policy{_suffix}.pt')

SEED = 42
N_EPISODES = 75000
N_TURNS = 15
GAMMA = 0.97
LR = 5e-4
BATCH = 128
BUFFER = 200_000
TARGET_SYNC = 2000        # gradient steps between target updates
EPS_START, EPS_END = 0.5, 0.05
EPS_DECAY_EPISODES = 15000
WARMUP = 2000             # transitions before learning
VAL_EVERY = 2000
LOG_EVERY = 1000

torch.manual_seed(SEED)
np.random.seed(SEED)


class Dueling(nn.Module):
    def __init__(self, state_dim, n_actions):
        super().__init__()
        self.shared = nn.Sequential(nn.Linear(state_dim, 512), nn.ReLU(),
                                    nn.Linear(512, 256), nn.ReLU())
        self.value = nn.Linear(256, 1)
        self.adv = nn.Linear(256, n_actions)

    def forward(self, x):
        h = self.shared(x)
        a = self.adv(h)
        return self.value(h) + a - a.mean(dim=-1, keepdim=True)


def main():
    world = load_world(seed=SEED, world=WORLD)
    instrument, n_items = world['instrument'], world['n_items']
    env = ElicitationEnv(instrument, n_items, N_TURNS)
    state_dim = env.state_dim
    rng = np.random.default_rng(SEED)

    q = Dueling(state_dim, n_items)
    q_t = Dueling(state_dim, n_items)
    q_t.load_state_dict(q.state_dict())
    opt = torch.optim.Adam(q.parameters(), lr=LR)

    S = np.zeros((BUFFER, state_dim), dtype=np.float32)
    A = np.zeros(BUFFER, dtype=np.int64)
    R = np.zeros(BUFFER, dtype=np.float32)
    S2 = np.zeros((BUFFER, state_dim), dtype=np.float32)
    D = np.zeros(BUFFER, dtype=np.float32)
    ASKED2 = np.zeros((BUFFER, n_items), dtype=bool)  # asked-mask of s2
    ptr, size = 0, 0

    def validate():
        q.eval()
        aucs = []
        for uid in world['val_uids']:
            s = env.reset(world['profiles'][uid])
            accs = [env.episode_accuracy()]
            for _ in range(N_TURNS):
                with torch.no_grad():
                    qa = q(torch.from_numpy(s).unsqueeze(0))[0]
                mask = torch.full((n_items,), float('-inf'))
                rem = [i for i in range(n_items) if i not in env.asked]
                mask[rem] = 0.0
                a = int(torch.argmax(qa + mask).item())
                s, _, done = env.step(a)
                accs.append(env.episode_accuracy())
                if done:
                    break
            aucs.append(float(np.mean(accs)))
        q.train()
        return float(np.mean(aucs))

    best, steps, ep_rewards = -1.0, 0, []
    t0 = time.time()
    for ep in range(N_EPISODES):
        eps = EPS_END + (EPS_START - EPS_END) * max(0, 1 - ep / EPS_DECAY_EPISODES)
        uid = world['train_uids'][int(rng.integers(len(world['train_uids'])))]
        s = env.reset(world['profiles'][uid])
        ep_r = 0.0
        for t in range(N_TURNS):
            rem = [i for i in range(n_items) if i not in env.asked]
            if rng.random() < eps:
                a = int(rng.choice(rem))
            else:
                with torch.no_grad():
                    qa = q(torch.from_numpy(s).unsqueeze(0))[0]
                mask = torch.full((n_items,), float('-inf'))
                mask[rem] = 0.0
                a = int(torch.argmax(qa + mask).item())
            asked_before_step = set(env.asked)
            s2, r, done = env.step(a)
            S[ptr], A[ptr], R[ptr], S2[ptr], D[ptr] = s, a, r, s2, float(done)
            m = np.zeros(n_items, dtype=bool)
            m[list(asked_before_step | {a})] = True
            ASKED2[ptr] = m
            ptr = (ptr + 1) % BUFFER
            size = min(size + 1, BUFFER)
            s = s2
            ep_r += r

            if size >= WARMUP:
                idx = rng.integers(0, size, BATCH)
                bs = torch.from_numpy(S[idx])
                ba = torch.from_numpy(A[idx])
                br = torch.from_numpy(R[idx])
                bs2 = torch.from_numpy(S2[idx])
                bd = torch.from_numpy(D[idx])
                bm = torch.from_numpy(ASKED2[idx])
                with torch.no_grad():
                    q_online = q(bs2)
                    q_online[bm] = -1e9        # mask already-asked actions
                    a_star = q_online.argmax(dim=1)
                    q_target = q_t(bs2).gather(1, a_star.unsqueeze(1)).squeeze(1)
                    y = br + GAMMA * (1 - bd) * q_target
                pred = q(bs).gather(1, ba.unsqueeze(1)).squeeze(1)
                loss = nn.functional.smooth_l1_loss(pred, y)
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(q.parameters(), 10.0)
                opt.step()
                steps += 1
                if steps % TARGET_SYNC == 0:
                    q_t.load_state_dict(q.state_dict())

        ep_rewards.append(ep_r)
        if (ep + 1) % LOG_EVERY == 0:
            print(f"Ep {ep + 1:6d}/{N_EPISODES} | reward(last {LOG_EVERY}) "
                  f"{np.mean(ep_rewards[-LOG_EVERY:]):+.4f} | eps={eps:.2f} "
                  f"| steps {steps} | {time.time() - t0:.0f}s", flush=True)
        if (ep + 1) % VAL_EVERY == 0:
            v = validate()
            mark = ''
            if v > best:
                best = v
                mark = ' *'
                torch.save({'q_state_dict': q.state_dict(),
                            'episodes': ep + 1, 'val_auac': v,
                            'state_dim': state_dim, 'world': WORLD,
                            'instrument': 'instrument_v5_set' if WORLD=='slate1' else 'instrument_slate2_dual'}, OUT_PATH)
            print(f"  VAL AUAC (argmax, {len(world['val_uids'])} users): "
                  f"{v:.4f}{mark}", flush=True)

    print(f"\nBest val AUAC {best:.4f}; saved {OUT_PATH}")


if __name__ == '__main__':
    main()
