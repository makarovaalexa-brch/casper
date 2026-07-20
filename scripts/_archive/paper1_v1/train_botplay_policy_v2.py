"""
Paper 1 / M3: Bot-play policy v2 -- fixing the fixed-sequence collapse.

Diagnosis of v1 (4k episodes, immediate-reward REINFORCE): the trained
policy asks the IDENTICAL question sequence to every user (100% turn-level
concentration, 15/187 entities used, turn-2 question independent of the
turn-1 answer). It learned a static ordering, not a strategy -- the same
pathology as the 2026 V3 continuous-action campaign, now reproduced under
controlled conditions.

v2 changes, each targeting a diagnosed cause:
  1. RETURN-TO-GO credit assignment (sum of future rewards, gamma=0.97):
     immediate rewards give no credit to questions that set up later gains.
  2. ENTROPY BONUS (beta annealed 0.05 -> 0.005): direct pressure against
     deterministic collapse.
  3. RICHER STATE: revealed one-hots ++ the instrument's current full-slate
     predictions, so the policy can see what is still uncertain (cf. EAR
     feeding recommender state into the policy).
  4. SCALE + SELECTION: 30k episodes; every 2k episodes the greedy (argmax)
     policy is validated on 100 instrument-validation users (never testbed
     users) and the best-AUAC checkpoint is kept.

Run from casper root:
    poetry run python scripts/paper1/train_botplay_policy_v2.py
Output: experiments/paper1/botplay_policy_v2.pt
"""

import sys
sys.path.insert(0, '.')
sys.path.insert(0, 'scripts/paper1')

import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from test_instrument_lib import load_instrument_by_name
from testbed import build_profiles, get_user_splits

OUT_PATH = Path('C:/dev/phd/casper/experiments/paper1/botplay_policy_v2.pt')

SEED = 42
N_EPISODES = 30000
N_TURNS = 15
LR = 5e-4
GAMMA = 0.97
ENTROPY_BETA_START, ENTROPY_BETA_END = 0.05, 0.005
EPSILON_START, EPSILON_END = 0.20, 0.02
N_TRAIN_USERS = 6000
VAL_EVERY = 2000
N_VAL_USERS = 100
LOG_EVERY = 1000

torch.manual_seed(SEED)
np.random.seed(SEED)


def bce_loss(preds, profile, n_movies):
    preds = preds[:n_movies]
    gt = profile[:n_movies]
    filt = ~np.isnan(gt)
    p = np.clip(preds[filt], 1e-7, 1 - 1e-7)
    y = gt[filt]
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def accuracy(preds, profile, n_movies):
    preds = preds[:n_movies]
    gt = profile[:n_movies]
    filt = ~np.isnan(gt)
    return float(np.mean((preds[filt] > 0.5) == gt[filt]))


class PolicyNet(nn.Module):
    def __init__(self, n_items):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_items * 3 + n_items, 512), nn.ReLU(),
            nn.Linear(512, 256), nn.ReLU(),
            nn.Linear(256, n_items),
        )

    def forward(self, x):
        return self.net(x)


def make_state(n_items, revealed, beliefs):
    s = np.zeros((n_items, 3), dtype=np.float32)
    s[:, 2] = 1
    for idx, pol in revealed:
        s[idx, 2] = 0
        s[idx, 1 if pol >= 0.5 else 0] = 1
    return np.concatenate([s.flatten(), beliefs.astype(np.float32)])


def run_argmax_episode(policy, instrument, profile, n_items, n_turns):
    revealed, asked, accs = [], set(), []
    beliefs = instrument.predict_full(revealed)
    accs.append(accuracy(beliefs, profile, instrument.n_movies))
    for _ in range(n_turns):
        x = torch.from_numpy(make_state(n_items, revealed, beliefs)).unsqueeze(0)
        with torch.no_grad():
            logits = policy(x)[0]
        mask = torch.full((n_items,), float('-inf'))
        rem = [i for i in range(n_items) if i not in asked]
        mask[rem] = 0.0
        a = int(torch.argmax(logits + mask).item())
        asked.add(a)
        v = profile[a]
        if not np.isnan(v):
            revealed.append((a, 1.0 if v >= 0.5 else 0.0))
        beliefs = instrument.predict_full(revealed)
        accs.append(accuracy(beliefs, profile, instrument.n_movies))
    return float(np.mean(accs))


def main():
    instrument, ckpt = load_instrument_by_name('instrument_v5_set')
    items = ckpt['items']
    n_items = len(items)

    train_users, ival_users, _ = get_user_splits(items)
    rng = np.random.default_rng(SEED)
    train_users = rng.choice(train_users, size=N_TRAIN_USERS, replace=False)
    val_users = ival_users[:N_VAL_USERS]

    print("Building profiles...")
    profiles = build_profiles(items, user_ids=np.concatenate([train_users, val_users]),
                              attr_min_support=3, taste_margin=None)
    train_uids = [u for u in train_users if u in profiles]
    val_uids = [u for u in val_users if u in profiles]
    print(f"  train {len(train_uids)}, val {len(val_uids)}")

    policy = PolicyNet(n_items)
    optimizer = torch.optim.Adam(policy.parameters(), lr=LR)
    print(f"Policy parameters: {sum(p.numel() for p in policy.parameters()):,}")

    turn_baselines = np.zeros(N_TURNS)
    best_val_auac, ep_rewards = -1.0, []
    t0 = time.time()

    for ep in range(N_EPISODES):
        frac = ep / N_EPISODES
        eps = EPSILON_START + (EPSILON_END - EPSILON_START) * frac
        beta = ENTROPY_BETA_START + (ENTROPY_BETA_END - ENTROPY_BETA_START) * frac

        uid = train_uids[int(rng.integers(len(train_uids)))]
        profile = profiles[uid]
        revealed, asked = [], set()
        beliefs = instrument.predict_full(revealed)
        loss_prev = bce_loss(beliefs, profile, instrument.n_movies)

        log_probs, entropies, rewards = [], [], []
        for t in range(N_TURNS):
            x = torch.from_numpy(make_state(n_items, revealed, beliefs)).unsqueeze(0)
            logits = policy(x)[0]
            mask = torch.full((n_items,), float('-inf'))
            rem = [i for i in range(n_items) if i not in asked]
            mask[rem] = 0.0
            logp = torch.log_softmax(logits + mask, dim=0)
            probs = logp.exp()

            if rng.random() < eps:
                a = int(rng.choice(rem))
            else:
                a = int(torch.multinomial(probs, 1).item())
            log_probs.append(logp[a])
            entropies.append(-(probs * logp.clamp(min=-30)).sum())
            asked.add(a)

            v = profile[a]
            if not np.isnan(v):
                revealed.append((a, 1.0 if v >= 0.5 else 0.0))
            beliefs = instrument.predict_full(revealed)
            loss_now = bce_loss(beliefs, profile, instrument.n_movies)
            rewards.append(loss_prev - loss_now)
            loss_prev = loss_now

        # return-to-go with per-turn baseline
        returns = np.zeros(N_TURNS)
        g = 0.0
        for t in reversed(range(N_TURNS)):
            g = rewards[t] + GAMMA * g
            returns[t] = g
        loss_terms = []
        for t in range(N_TURNS):
            adv = returns[t] - turn_baselines[t]
            turn_baselines[t] = 0.995 * turn_baselines[t] + 0.005 * returns[t]
            loss_terms.append(-log_probs[t] * adv - beta * entropies[t])
        optimizer.zero_grad()
        torch.stack(loss_terms).sum().backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 5.0)
        optimizer.step()

        ep_rewards.append(sum(rewards))
        if (ep + 1) % LOG_EVERY == 0:
            print(f"Ep {ep + 1:6d}/{N_EPISODES} | reward(last {LOG_EVERY}) "
                  f"{np.mean(ep_rewards[-LOG_EVERY:]):+.4f} | eps={eps:.2f} "
                  f"beta={beta:.3f} | {time.time() - t0:.0f}s", flush=True)

        if (ep + 1) % VAL_EVERY == 0:
            policy.eval()
            auacs = [run_argmax_episode(policy, instrument, profiles[u],
                                        n_items, N_TURNS) for u in val_uids]
            policy.train()
            val_auac = float(np.mean(auacs))
            marker = ''
            if val_auac > best_val_auac:
                best_val_auac = val_auac
                marker = ' *'
                torch.save({
                    'policy_state_dict': policy.state_dict(),
                    'arch': 'v2_beliefs_state',
                    'n_items': n_items,
                    'episodes': ep + 1,
                    'val_auac': val_auac,
                    'instrument': 'instrument_v5_set',
                }, OUT_PATH)
            print(f"  VAL AUAC (argmax, {len(val_uids)} users): "
                  f"{val_auac:.4f}{marker}", flush=True)

    print(f"\nBest val AUAC {best_val_auac:.4f}; saved {OUT_PATH}")


if __name__ == '__main__':
    main()
