"""
Paper 1 / M3: Train the bot-play REINFORCE policy (Makarova et al., IJCNN
2024 method) against the fixed instrument, on training users only.

Faithful to the published design:
- Questioner policy: feedforward net over the revealed-state one-hots,
  softmax over the slate, epsilon-greedy exploration.
- Reward: per-turn reduction in the recommendation model's BCE loss on the
  user's rated movies (the instrument plays the role of the recommendation
  network phi).
- REINFORCE with a running per-turn baseline for variance reduction.

Run from casper root:
    poetry run python scripts/paper1/train_botplay_policy.py
Output: experiments/paper1/botplay_policy.pt
"""

import sys
sys.path.insert(0, '.')
sys.path.insert(0, 'scripts/paper1')

import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from test_instrument_lib import (CHECKPOINT_DIR, ExtrapolationModel,
                                 InstrumentWrapper)
from testbed import build_profiles, get_user_splits

OUT_PATH = Path('C:/dev/phd/casper/experiments/paper1/botplay_policy.pt')

SEED = 42
N_EPISODES = 4000
N_TURNS = 15
LR = 1e-3
EPSILON_START, EPSILON_END = 0.30, 0.05
GAMMA_BASELINE = 0.99  # running-mean baseline decay
N_TRAIN_USERS = 4000
LOG_EVERY = 200

torch.manual_seed(SEED)
np.random.seed(SEED)


def load_instrument():
    from test_instrument_lib import load_instrument_by_name
    return load_instrument_by_name('instrument_v5_set')


def bce_loss(preds, profile, n_movies):
    gt = profile[:n_movies]
    filt = ~np.isnan(gt)
    p = np.clip(preds[filt], 1e-7, 1 - 1e-7)
    y = gt[filt]
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def main():
    instrument, ckpt = load_instrument()
    items = ckpt['items']
    n_items = len(items)

    train_users, _, _ = get_user_splits(items)
    rng = np.random.default_rng(SEED)
    train_users = rng.choice(train_users, size=N_TRAIN_USERS, replace=False)
    print("Building train profiles...")
    profiles = build_profiles(items, user_ids=train_users)
    uids = [u for u in train_users if u in profiles]
    print(f"  {len(uids)} profiles")

    policy = nn.Sequential(
        nn.Linear(n_items * 3, 256), nn.ReLU(),
        nn.Linear(256, n_items),
    )
    optimizer = torch.optim.Adam(policy.parameters(), lr=LR)

    baselines = np.zeros(N_TURNS)  # per-turn running-mean reward
    ep_rewards = []
    t0 = time.time()

    for ep in range(N_EPISODES):
        uid = uids[int(rng.integers(len(uids)))]
        profile = profiles[uid]
        eps = EPSILON_START + (EPSILON_END - EPSILON_START) * ep / N_EPISODES

        revealed = []
        asked = set()
        state = np.zeros((n_items, 3), dtype=np.float32)
        state[:, 2] = 1

        loss_prev = bce_loss(instrument.predict(revealed), profile,
                             instrument.n_movies)
        log_probs, rewards = [], []

        for t in range(N_TURNS):
            s = torch.from_numpy(state.flatten()).unsqueeze(0)
            logits = policy(s)[0]
            mask = torch.full((n_items,), float('-inf'))
            rem = [i for i in range(n_items) if i not in asked]
            mask[rem] = 0.0
            probs = torch.softmax(logits + mask, dim=0)

            if rng.random() < eps:
                a = int(rng.choice(rem))
            else:
                a = int(torch.multinomial(probs, 1).item())
            log_probs.append(torch.log(probs[a].clamp(min=1e-9)))
            asked.add(a)

            v = profile[a]
            if not np.isnan(v):
                pol = 1.0 if v >= 0.5 else 0.0
                revealed.append((a, pol))
                state[a, 2] = 0
                state[a, 1 if pol >= 0.5 else 0] = 1

            loss_now = bce_loss(instrument.predict(revealed), profile,
                                instrument.n_movies)
            rewards.append(loss_prev - loss_now)
            loss_prev = loss_now

        # REINFORCE update with per-turn baseline
        loss_terms = []
        for t, (lp, r) in enumerate(zip(log_probs, rewards)):
            advantage = r - baselines[t]
            baselines[t] = GAMMA_BASELINE * baselines[t] + (1 - GAMMA_BASELINE) * r
            loss_terms.append(-lp * advantage)
        optimizer.zero_grad()
        torch.stack(loss_terms).sum().backward()
        optimizer.step()

        ep_rewards.append(sum(rewards))
        if (ep + 1) % LOG_EVERY == 0:
            recent = np.mean(ep_rewards[-LOG_EVERY:])
            print(f"Episode {ep + 1:5d}/{N_EPISODES} | mean total reward "
                  f"(last {LOG_EVERY}): {recent:+.4f} | eps={eps:.2f} | "
                  f"{time.time() - t0:.0f}s", flush=True)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        'policy_state_dict': policy.state_dict(),
        'n_items': n_items,
        'episodes': N_EPISODES,
        'instrument': 'instrument_v5_set',
        'mean_reward_last500': float(np.mean(ep_rewards[-500:])),
    }, OUT_PATH)
    print(f"\nSaved {OUT_PATH}")


if __name__ == '__main__':
    main()
