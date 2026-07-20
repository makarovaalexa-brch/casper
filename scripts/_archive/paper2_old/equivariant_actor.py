"""
Permutation-equivariant per-item scoring actor (Deep Sets / pointer style).

Diagnosis (3 independent lit agents agree): a flat MLP -> N-way softmax actor
has no inductive bias tying logit i to item i's belief features, so under low
advantage-SNR it collapses to a state-independent (static) policy. This actor
scores EACH candidate from its own belief slice with SHARED weights + a pooled
global context, so per-user routing is the default, not something that must be
discovered from noisy gradients.

Input: env dual-state [B, 5*n] = [reveal(3n), belief(n), rated(n)].
Output: logits [B, n]. Mask asked items to -inf before softmax/argmax.

Refs: Zaheer et al. Deep Sets 2017; Vinyals et al. Pointer Networks 2015;
object-exchangeability in RL (arXiv:1905.02698); Wolpertinger (Dulac-Arnold 2015).
"""
import torch
import torch.nn as nn


class EquivariantActor(nn.Module):
    def __init__(self, n_items, state_mode='dual', d_emb=32, d_hidden=128):
        super().__init__()
        self.n_items = n_items
        self.per_item = 5 if state_mode == 'dual' else 4
        self.item_emb = nn.Embedding(n_items, d_emb)
        self.enc = nn.Sequential(nn.Linear(self.per_item + d_emb, d_hidden), nn.ReLU(),
                                 nn.Linear(d_hidden, d_hidden), nn.ReLU())
        self.ctx = nn.Sequential(nn.Linear(d_hidden, d_hidden), nn.ReLU())
        self.score = nn.Sequential(nn.Linear(d_hidden * 2, d_hidden), nn.ReLU(),
                                   nn.Linear(d_hidden, 1))
        self.register_buffer('idx', torch.arange(n_items))

    def item_features(self, state):
        B = state.shape[0]; n = self.n_items
        reveal = state[:, :3 * n].reshape(B, n, 3)
        belief = state[:, 3 * n:4 * n].unsqueeze(-1)
        feats = [reveal, belief]
        if self.per_item == 5:
            feats.append(state[:, 4 * n:5 * n].unsqueeze(-1))
        return torch.cat(feats, dim=-1)

    def forward(self, state):
        B = state.shape[0]; n = self.n_items
        f = self.item_features(state)
        e = self.item_emb(self.idx).unsqueeze(0).expand(B, -1, -1)
        h = self.enc(torch.cat([f, e], dim=-1))                       # [B,n,H]
        ctx = self.ctx(h.mean(dim=1, keepdim=True)).expand(-1, n, -1)  # [B,n,H]
        return self.score(torch.cat([h, ctx], dim=-1)).squeeze(-1)    # [B,n]


class EquivariantCritic(nn.Module):
    """Pooled state-value head over the same per-item features (for A2C/PPO)."""
    def __init__(self, n_items, state_mode='dual', d_emb=32, d_hidden=128):
        super().__init__()
        self.actor_feat = EquivariantActor(n_items, state_mode, d_emb, d_hidden)
        self.v = nn.Sequential(nn.Linear(d_hidden, d_hidden), nn.ReLU(),
                               nn.Linear(d_hidden, 1))

    def forward(self, state):
        f = self.actor_feat.item_features(state)
        B, n, _ = f.shape
        e = self.actor_feat.item_emb(self.actor_feat.idx).unsqueeze(0).expand(B, -1, -1)
        h = self.actor_feat.enc(torch.cat([f, e], dim=-1))
        return self.v(h.mean(dim=1)).squeeze(-1)
