"""
Paper 2a comparator: discrete PPO over the entity slate.

The ablation that answers RQ3's second half: does the continuous semantic
action space (CASPER-CA) buy anything over a plain discrete stochastic
policy with entropy regularisation, trained on identical dynamics,
state features, reward, and budget?

PPO with masked categorical over the 187 entities, GAE(lambda), entropy
bonus, identical validation protocol to train_continuous_policy.py.

Run from casper root:
    poetry run python scripts/paper2/train_discrete_ppo.py
Output: experiments/paper2/discrete_ppo.pt
"""

import sys
sys.path.insert(0, '.')
sys.path.insert(0, 'scripts/paper1')
sys.path.insert(0, 'scripts/paper2')

import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from env import ElicitationEnv, load_world

OUT_PATH = Path('C:/dev/phd/casper/experiments/paper2/discrete_ppo.pt')
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

SEED = 42
N_EPISODES = 90000
N_TURNS = 15
GAMMA = 0.97
LAM = 0.95
CLIP = 0.2
LR = 3e-4
ENTROPY_COEF_START, ENTROPY_COEF_END = 0.03, 0.003
EPOCHS_PER_BATCH = 4
EPISODES_PER_BATCH = 16
VAL_EVERY = 2000
LOG_EVERY = 1000

torch.manual_seed(SEED)
np.random.seed(SEED)


class ActorCritic(nn.Module):
    def __init__(self, state_dim, n_items):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(state_dim, 512), nn.ReLU(),
            nn.Linear(512, 256), nn.ReLU(),
        )
        self.pi = nn.Linear(256, n_items)
        self.v = nn.Linear(256, 1)

    def forward(self, s):
        h = self.shared(s)
        return self.pi(h), self.v(h).squeeze(-1)


def masked_dist(logits, asked, n_items):
    mask = torch.full((n_items,), float('-inf'))
    rem = [i for i in range(n_items) if i not in asked]
    mask[rem] = 0.0
    return torch.distributions.Categorical(logits=logits + mask)


def main():
    world = load_world(seed=SEED)
    instrument, n_items = world['instrument'], world['n_items']
    env = ElicitationEnv(instrument, n_items, N_TURNS)
    state_dim = n_items * 3 + n_items
    rng = np.random.default_rng(SEED)

    net = ActorCritic(state_dim, n_items)
    opt = torch.optim.Adam(net.parameters(), lr=LR)

    def validate():
        net.eval()
        aucs = []
        for uid in world['val_uids']:
            s = env.reset(world['profiles'][uid])
            accs = [env.episode_accuracy()]
            for _ in range(N_TURNS):
                with torch.no_grad():
                    logits, _ = net(torch.from_numpy(s).unsqueeze(0))
                mask = torch.full((n_items,), float('-inf'))
                rem = [i for i in range(n_items) if i not in env.asked]
                mask[rem] = 0.0
                e = int(torch.argmax(logits[0] + mask).item())
                s, _, done = env.step(e)
                accs.append(env.episode_accuracy())
                if done:
                    break
            aucs.append(float(np.mean(accs)))
        net.train()
        return float(np.mean(aucs))

    best_val, ep_count, ep_rewards = -1.0, 0, []
    t0 = time.time()
    while ep_count < N_EPISODES:
        frac = ep_count / N_EPISODES
        ent_coef = ENTROPY_COEF_START + (ENTROPY_COEF_END - ENTROPY_COEF_START) * frac

        # collect a batch of episodes
        S, A, LP, R, D, V, ASKED = [], [], [], [], [], [], []
        for _ in range(EPISODES_PER_BATCH):
            uid = world['train_uids'][int(rng.integers(len(world['train_uids'])))]
            s = env.reset(world['profiles'][uid])
            ep_r = 0.0
            for t in range(N_TURNS):
                st = torch.from_numpy(s)
                with torch.no_grad():
                    logits, v = net(st.unsqueeze(0))
                asked_now = frozenset(env.asked)
                dist = masked_dist(logits[0], env.asked, n_items)
                a = int(dist.sample().item())
                lp = float(dist.log_prob(torch.tensor(a)).item())
                s2, r, done = env.step(a)
                S.append(s); A.append(a); LP.append(lp); R.append(r)
                D.append(float(done)); V.append(float(v.item()))
                ASKED.append(asked_now)
                s = s2
                ep_r += r
            ep_rewards.append(ep_r)
            ep_count += 1

        # GAE
        S_t = torch.from_numpy(np.stack(S))
        A_t = torch.tensor(A)
        LP_t = torch.tensor(LP, dtype=torch.float32)
        adv = np.zeros(len(R), dtype=np.float32)
        last = 0.0
        for i in reversed(range(len(R))):
            next_v = 0.0 if D[i] else (V[i + 1] if i + 1 < len(V) else 0.0)
            delta = R[i] + GAMMA * next_v - V[i]
            last = delta + GAMMA * LAM * (0.0 if D[i] else last)
            adv[i] = last
        ret = adv + np.array(V, dtype=np.float32)
        adv_t = torch.from_numpy((adv - adv.mean()) / (adv.std() + 1e-8))
        ret_t = torch.from_numpy(ret)

        for _ in range(EPOCHS_PER_BATCH):
            logits, v = net(S_t)
            # rebuild masked distributions (masks differ per step)
            logps, ents = [], []
            for i in range(len(A)):
                dist = masked_dist(logits[i], ASKED[i], n_items)
                logps.append(dist.log_prob(A_t[i]))
                ents.append(dist.entropy())
            logp = torch.stack(logps)
            ent = torch.stack(ents).mean()
            ratio = (logp - LP_t).exp()
            pg = -torch.min(ratio * adv_t,
                            ratio.clamp(1 - CLIP, 1 + CLIP) * adv_t).mean()
            v_loss = ((v - ret_t) ** 2).mean()
            loss = pg + 0.5 * v_loss - ent_coef * ent
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
            opt.step()

        if ep_count % LOG_EVERY < EPISODES_PER_BATCH:
            print(f"Ep {ep_count:6d}/{N_EPISODES} | reward(last {LOG_EVERY}) "
                  f"{np.mean(ep_rewards[-LOG_EVERY:]):+.4f} | ent_coef "
                  f"{ent_coef:.3f} | {time.time() - t0:.0f}s", flush=True)
        if ep_count % VAL_EVERY < EPISODES_PER_BATCH:
            val = validate()
            marker = ''
            if val > best_val:
                best_val = val
                marker = ' *'
                torch.save({
                    'net_state_dict': net.state_dict(),
                    'episodes': ep_count, 'val_auac': val,
                    'instrument': 'instrument_v5_set',
                }, OUT_PATH)
            print(f"  VAL AUAC (argmax, {len(world['val_uids'])} users): "
                  f"{val:.4f}{marker}", flush=True)

    print(f"\nBest val AUAC {best_val:.4f}; saved {OUT_PATH}")


if __name__ == '__main__':
    main()
