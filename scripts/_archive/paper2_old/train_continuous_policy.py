"""
Paper 2a / CASPER-CA: continuous semantic-action elicitation policy.

Wolpertinger-style (Dulac-Arnold et al., 2015) actor-critic, with the
repairs identified in the June 2026 expert review of the original CASPER
system:

  1. CRITIC RE-RANKING (k > 1): the actor emits a proto-action in the
     SBERT entity space; the k nearest unasked entities are retrieved and
     the critic evaluates each, executing the argmax. The original
     system's k=1 hard snap created action aliasing.
  2. EXECUTED-ENTITY TRAINING: the replay buffer stores the embedding of
     the entity actually executed, never the proto-action. (The original
     stored proto-actions, decoupling Q-targets from realised actions.)
  3. TD3 stabilisation: twin critics, delayed actor updates, target
     policy smoothing -- with the smoothed target action also grounded to
     a real entity, keeping target Q on the embedding manifold.
  4. FREE EPISODES: the verifiable simulator from the Paper 1 testbed
     replaces the LLM-in-the-loop, closing the 100x sample deficit.
  5. HIGH-SNR REWARD: instrument BCE delta (the IJCNN bot-play reward).

Run from casper root:
    poetry run python scripts/paper2/train_continuous_policy.py
Output: experiments/paper2/casper_ca.pt
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

OUT_PATH = Path('C:/dev/phd/casper/experiments/paper2/casper_ca.pt')
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

SEED = 42
N_EPISODES = 30000
N_TURNS = 15
EMB_DIM = 384
K_CANDIDATES = 20          # Wolpertinger k
GAMMA = 0.97
TAU = 0.005
ACTOR_LR = 1e-4
CRITIC_LR = 3e-4
BATCH = 256
BUFFER_CAP = 400_000
WARMUP_EPISODES = 500       # random entities before learning starts
POLICY_DELAY = 2            # TD3 delayed actor updates
EXPL_NOISE = 0.2            # noise on proto-action during training
TARGET_NOISE = 0.1          # target policy smoothing
NOISE_CLIP = 0.25
VAL_EVERY = 2000
LOG_EVERY = 1000

torch.manual_seed(SEED)
np.random.seed(SEED)


class Actor(nn.Module):
    def __init__(self, state_dim, emb_dim=EMB_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 512), nn.ReLU(),
            nn.Linear(512, 512), nn.ReLU(),
            nn.Linear(512, emb_dim), nn.Tanh(),
        )

    def forward(self, s):
        a = self.net(s)
        return a / (a.norm(dim=-1, keepdim=True) + 1e-8)


class Critic(nn.Module):
    def __init__(self, state_dim, emb_dim=EMB_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + emb_dim, 512), nn.ReLU(),
            nn.Linear(512, 256), nn.ReLU(),
            nn.Linear(256, 1),
        )

    def forward(self, s, a):
        return self.net(torch.cat([s, a], dim=-1)).squeeze(-1)


class ReplayBuffer:
    def __init__(self, cap, state_dim, emb_dim=EMB_DIM):
        self.cap = cap
        self.n = 0
        self.i = 0
        self.s = np.zeros((cap, state_dim), dtype=np.float32)
        self.a = np.zeros((cap, emb_dim), dtype=np.float32)
        self.r = np.zeros(cap, dtype=np.float32)
        self.s2 = np.zeros((cap, state_dim), dtype=np.float32)
        self.d = np.zeros(cap, dtype=np.float32)
        # asked-mask of the NEXT state, needed to ground target actions
        self.mask2 = np.zeros((cap,), dtype=np.int64)  # turn index proxy

    def add(self, s, a, r, s2, d):
        j = self.i
        self.s[j], self.a[j], self.r[j], self.s2[j], self.d[j] = s, a, r, s2, d
        self.i = (self.i + 1) % self.cap
        self.n = min(self.n + 1, self.cap)

    def sample(self, batch, rng):
        idx = rng.integers(0, self.n, batch)
        t = lambda x: torch.from_numpy(x[idx])
        return t(self.s), t(self.a), t(self.r), t(self.s2), t(self.d)


def ground(proto, entity_emb_t, asked, critic, state_t, k=K_CANDIDATES):
    """Wolpertinger grounding: kNN by cosine, critic re-ranks, argmax.

    proto: [emb] tensor; entity_emb_t: [n_items, emb]; asked: set of ints;
    state_t: [state_dim] tensor. Returns entity index (int).
    """
    sims = entity_emb_t @ proto
    if asked:
        sims[list(asked)] = -1e9
    k = min(k, (sims > -1e8).sum().item())
    cand = torch.topk(sims, k).indices
    with torch.no_grad():
        q = critic(state_t.unsqueeze(0).expand(k, -1), entity_emb_t[cand])
    return int(cand[int(torch.argmax(q))].item())


def ground_batch_target(proto_b, entity_emb_t, critic, state_b):
    """Ground a batch of target proto-actions to nearest entity embeddings
    (k=8 critic re-rank) -- used for target Q computation."""
    k = 8
    sims = proto_b @ entity_emb_t.T                     # [B, n_items]
    cand = torch.topk(sims, k, dim=1).indices           # [B, k]
    B = proto_b.shape[0]
    flat_emb = entity_emb_t[cand.reshape(-1)]           # [B*k, emb]
    flat_state = state_b.repeat_interleave(k, dim=0)    # [B*k, state]
    with torch.no_grad():
        q = critic(flat_state, flat_emb).reshape(B, k)
    best = q.argmax(dim=1)                              # [B]
    return flat_emb.reshape(B, k, -1)[torch.arange(B), best]


def main():
    world = load_world(seed=SEED)
    instrument, n_items = world['instrument'], world['n_items']
    entity_emb_t = torch.from_numpy(world['entity_emb'])
    env = ElicitationEnv(instrument, n_items, N_TURNS)
    state_dim = n_items * 3 + n_items
    rng = np.random.default_rng(SEED)

    actor = Actor(state_dim)
    actor_t = Actor(state_dim)
    actor_t.load_state_dict(actor.state_dict())
    c1, c2 = Critic(state_dim), Critic(state_dim)
    c1_t, c2_t = Critic(state_dim), Critic(state_dim)
    c1_t.load_state_dict(c1.state_dict())
    c2_t.load_state_dict(c2.state_dict())
    opt_a = torch.optim.Adam(actor.parameters(), lr=ACTOR_LR)
    opt_c = torch.optim.Adam(list(c1.parameters()) + list(c2.parameters()),
                             lr=CRITIC_LR)
    buf = ReplayBuffer(BUFFER_CAP, state_dim)

    def validate():
        actor.eval()
        aucs = []
        for uid in world['val_uids']:
            s = env.reset(world['profiles'][uid])
            accs = [env.episode_accuracy()]
            for _ in range(N_TURNS):
                with torch.no_grad():
                    proto = actor(torch.from_numpy(s).unsqueeze(0))[0]
                e = ground(proto, entity_emb_t, env.asked, c1,
                           torch.from_numpy(s))
                s, _, done = env.step(e)
                accs.append(env.episode_accuracy())
                if done:
                    break
            aucs.append(float(np.mean(accs)))
        actor.train()
        return float(np.mean(aucs))

    best_val, updates, ep_rewards = -1.0, 0, []
    t0 = time.time()
    for ep in range(N_EPISODES):
        uid = world['train_uids'][int(rng.integers(len(world['train_uids'])))]
        s = env.reset(world['profiles'][uid])
        ep_r = 0.0
        for t in range(N_TURNS):
            if ep < WARMUP_EPISODES:
                rem = [i for i in range(n_items) if i not in env.asked]
                e = int(rng.choice(rem))
            else:
                with torch.no_grad():
                    proto = actor(torch.from_numpy(s).unsqueeze(0))[0]
                proto = proto + torch.from_numpy(
                    rng.normal(0, EXPL_NOISE, EMB_DIM).astype(np.float32))
                proto = proto / (proto.norm() + 1e-8)
                e = ground(proto, entity_emb_t, env.asked, c1,
                           torch.from_numpy(s))
            s2, r, done = env.step(e)
            buf.add(s, world['entity_emb'][e], r, s2, float(done))
            s = s2
            ep_r += r

            if ep >= WARMUP_EPISODES and buf.n >= BATCH:
                bs, ba, br, bs2, bd = buf.sample(BATCH, rng)
                with torch.no_grad():
                    proto2 = actor_t(bs2)
                    noise = torch.randn_like(proto2) * TARGET_NOISE
                    proto2 = proto2 + noise.clamp(-NOISE_CLIP, NOISE_CLIP)
                    proto2 = proto2 / (proto2.norm(dim=-1, keepdim=True) + 1e-8)
                    a2 = ground_batch_target(proto2, entity_emb_t, c1_t, bs2)
                    q_t = torch.min(c1_t(bs2, a2), c2_t(bs2, a2))
                    y = br + GAMMA * (1 - bd) * q_t
                q_loss = ((c1(bs, ba) - y) ** 2).mean() + \
                         ((c2(bs, ba) - y) ** 2).mean()
                opt_c.zero_grad()
                q_loss.backward()
                opt_c.step()
                updates += 1

                if updates % POLICY_DELAY == 0:
                    a_loss = -c1(bs, actor(bs)).mean()
                    opt_a.zero_grad()
                    a_loss.backward()
                    opt_a.step()
                    for net, net_t in ((actor, actor_t), (c1, c1_t), (c2, c2_t)):
                        for p, p_t in zip(net.parameters(), net_t.parameters()):
                            p_t.data.mul_(1 - TAU).add_(TAU * p.data)

        ep_rewards.append(ep_r)
        if (ep + 1) % LOG_EVERY == 0:
            print(f"Ep {ep + 1:6d}/{N_EPISODES} | reward(last {LOG_EVERY}) "
                  f"{np.mean(ep_rewards[-LOG_EVERY:]):+.4f} | updates {updates} "
                  f"| {time.time() - t0:.0f}s", flush=True)
        if (ep + 1) % VAL_EVERY == 0:
            v = validate()
            marker = ''
            if v > best_val:
                best_val = v
                marker = ' *'
                torch.save({
                    'actor_state_dict': actor.state_dict(),
                    'critic1_state_dict': c1.state_dict(),
                    'k_candidates': K_CANDIDATES,
                    'episodes': ep + 1, 'val_auac': v,
                    'instrument': 'instrument_v5_set',
                }, OUT_PATH)
            print(f"  VAL AUAC (greedy, {len(world['val_uids'])} users): "
                  f"{v:.4f}{marker}", flush=True)

    print(f"\nBest val AUAC {best_val:.4f}; saved {OUT_PATH}")


if __name__ == '__main__':
    main()
