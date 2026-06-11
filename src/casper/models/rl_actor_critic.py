"""
Actor-Critic RL agent for continuous action space (semantic embeddings).

Uses DDPG-style training to predict embeddings in SentenceBERT space.
Now supports LSTM+attention state encoding over preference sequences.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from typing import List, Optional, Tuple


class LSTMAttentionStateEncoder(nn.Module):
    """
    State encoder that processes a sequence of (item_embedding, rating) pairs.

    Architecture (from Concept model / IEEE paper):
    1. Concatenate item embedding with rating one-hot
    2. Process sequence with LSTM
    3. Apply self-attention over LSTM outputs
    4. Pool to single state representation

    Input:
        item_embeddings: [batch, seq_len, item_dim] - SBERT embeddings of items
        rating_one_hots: [batch, seq_len, 3] - [liked, disliked, not_seen]
        mask: [batch, seq_len] - True for valid positions

    Output:
        state_embedding: [batch, output_dim]
    """

    def __init__(
        self,
        item_dim: int = 384,  # SBERT dimension
        hidden_dim: int = 256,
        output_dim: int = 384,  # Match SBERT dim for compatibility
        num_layers: int = 1,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.item_dim = item_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # Input: item embedding + rating one-hot [liked, disliked, not_seen]
        input_dim = item_dim + 3

        # LSTM encoder
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # Decoder dense layers
        self.dense1 = nn.Linear(hidden_dim, hidden_dim * 2)
        self.dense2 = nn.Linear(hidden_dim * 2, hidden_dim)

        # Self-attention
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        # Output projection (concat dense + attention, project to output_dim)
        self.output_proj = nn.Linear(hidden_dim * 2, output_dim)
        self.layer_norm = nn.LayerNorm(output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        item_embeddings: torch.Tensor,
        rating_one_hots: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            item_embeddings: [batch, seq_len, item_dim]
            rating_one_hots: [batch, seq_len, 3]
            mask: [batch, seq_len] - True for valid positions

        Returns:
            state_embedding: [batch, output_dim]
        """
        batch_size, seq_len, _ = item_embeddings.shape

        # Handle empty sequence
        if seq_len == 0:
            return torch.zeros(batch_size, self.output_dim, device=item_embeddings.device)

        # Concatenate item embedding with rating one-hot
        x = torch.cat([item_embeddings, rating_one_hots], dim=-1)

        # LSTM encoding
        lstm_out, _ = self.lstm(x)  # [batch, seq_len, hidden_dim]

        # Decoder dense layers
        dense_out = F.relu(self.dense1(lstm_out))
        dense_out = F.relu(self.dense2(dense_out))

        # Self-attention
        key_padding_mask = ~mask if mask is not None else None
        attn_out, _ = self.attention(
            query=lstm_out,
            key=dense_out,
            value=dense_out,
            key_padding_mask=key_padding_mask,
        )

        # Concatenate dense and attention outputs
        combined = torch.cat([dense_out, attn_out], dim=-1)

        # Masked mean pooling
        if mask is not None:
            mask_expanded = mask.unsqueeze(-1).float()
            combined = combined * mask_expanded
            pooled = combined.sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1)
        else:
            pooled = combined.mean(dim=1)

        # Output projection
        state_emb = self.output_proj(pooled)
        state_emb = self.layer_norm(state_emb)

        return state_emb


def create_rating_one_hot(rating: str, device: torch.device = None) -> torch.Tensor:
    """
    Convert a single rating string to one-hot tensor.

    Args:
        rating: "liked", "disliked", or "unknown"/"not_seen"
        device: Target device

    Returns:
        one_hot: [3] tensor - [liked, disliked, not_seen]
    """
    one_hot = torch.zeros(3, device=device)
    if rating == "liked":
        one_hot[0] = 1.0
    elif rating == "disliked":
        one_hot[1] = 1.0
    else:  # unknown / not_seen
        one_hot[2] = 1.0
    return one_hot


def create_rating_batch(ratings: List[str], device: torch.device = None) -> torch.Tensor:
    """
    Convert list of rating strings to one-hot tensors.

    Args:
        ratings: List of "liked", "disliked", or "unknown"/"not_seen"
        device: Target device

    Returns:
        one_hot: [seq_len, 3] tensor - [liked, disliked, not_seen]
    """
    one_hot = torch.zeros(len(ratings), 3, device=device)
    for i, rating in enumerate(ratings):
        if rating == "liked":
            one_hot[i, 0] = 1.0
        elif rating == "disliked":
            one_hot[i, 1] = 1.0
        else:  # unknown / not_seen
            one_hot[i, 2] = 1.0
    return one_hot


class EmbeddingActorCritic:
    """
    Actor-Critic RL agent that predicts continuous embeddings.

    Actor: State → SentenceBERT embedding (384-dim)
    Critic: State + Embedding → Q-value

    Uses DDPG-style training for continuous action space.
    Includes target networks for stable Q-learning.

    Supports two state encoding modes:
    1. 'sbert': Single SBERT encoding of full preference text (original)
    2. 'lstm': LSTM+attention over preference sequence (Concept model style)
    """

    def __init__(
        self,
        state_dim: int = 384,  # SentenceBERT dimension
        embedding_dim: int = 384,  # SentenceBERT dimension
        hidden_dim: int = 512,
        learning_rate: float = 0.0003,  # Lower LR for stability
        exploration_noise: float = 0.2,
        state_encoder_type: str = 'sbert',  # 'sbert' or 'lstm'
        tau: float = 0.005,  # Soft target update rate
        gamma: float = 0.95,  # Discount factor (lower for short episodes)
    ):
        self.state_dim = state_dim
        self.embedding_dim = embedding_dim
        self.exploration_noise = exploration_noise
        self.state_encoder_type = state_encoder_type
        self.tau = tau
        self.gamma = gamma

        # State encoder (for LSTM mode)
        self.state_encoder = None
        if state_encoder_type == 'lstm':
            self.state_encoder = LSTMAttentionStateEncoder(
                item_dim=384,  # SBERT dimension
                hidden_dim=256,
                output_dim=state_dim,  # Output matches state_dim
                num_layers=1,
                num_heads=4,
                dropout=0.1,
            )

        # Actor network: state → embedding
        self.actor = self._build_actor(state_dim, hidden_dim, embedding_dim)
        self.actor_target = self._build_actor(state_dim, hidden_dim, embedding_dim)
        self.actor_target.load_state_dict(self.actor.state_dict())

        # Critic network: state + embedding → Q-value
        self.critic = self._build_critic(state_dim, embedding_dim, hidden_dim)
        self.critic_target = self._build_critic(state_dim, embedding_dim, hidden_dim)
        self.critic_target.load_state_dict(self.critic.state_dict())

        # Freeze target networks (no grad needed)
        for param in self.actor_target.parameters():
            param.requires_grad = False
        for param in self.critic_target.parameters():
            param.requires_grad = False

        # Optimizers - include state encoder params if using LSTM
        actor_params = list(self.actor.parameters())
        if self.state_encoder is not None:
            actor_params.extend(self.state_encoder.parameters())
        self.actor_optimizer = torch.optim.Adam(actor_params, lr=learning_rate)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=learning_rate)

        # Experience replay
        self.memory = []

        # Reward normalization (running stats)
        self.reward_mean = 0.0
        self.reward_std = 1.0
        self.reward_count = 0

        # Exploration
        self.noise_scale = exploration_noise
        self.noise_decay = 0.995
        self.noise_min = 0.05

    def _build_actor(self, state_dim, hidden_dim, embedding_dim):
        """Build actor network."""
        return nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, embedding_dim),
            nn.Tanh()  # Output in [-1, 1]
        )

    def _build_critic(self, state_dim, embedding_dim, hidden_dim):
        """Build critic network."""
        return nn.Sequential(
            nn.Linear(state_dim + embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def _soft_update(self, target_net, source_net):
        """Soft update target network: θ_target = τ*θ_source + (1-τ)*θ_target"""
        for target_param, source_param in zip(target_net.parameters(), source_net.parameters()):
            target_param.data.copy_(
                self.tau * source_param.data + (1.0 - self.tau) * target_param.data
            )

    def _update_reward_stats(self, rewards):
        """Update running reward statistics for normalization."""
        batch_mean = np.mean(rewards)
        batch_std = np.std(rewards) + 1e-8
        batch_size = len(rewards)

        # Welford's online algorithm
        if self.reward_count == 0:
            self.reward_mean = batch_mean
            self.reward_std = batch_std
        else:
            new_count = self.reward_count + batch_size
            delta = batch_mean - self.reward_mean
            self.reward_mean += delta * batch_size / new_count
            # Simple EMA for std
            self.reward_std = 0.99 * self.reward_std + 0.01 * batch_std

        self.reward_count += batch_size

    def _normalize_reward(self, reward):
        """Normalize reward using running statistics."""
        return (reward - self.reward_mean) / (self.reward_std + 1e-8)

    def encode_state_from_sequence(
        self,
        item_embeddings: np.ndarray,
        ratings: List[str],
    ) -> np.ndarray:
        """
        Encode state from a sequence of (item_embedding, rating) pairs.

        Only works when state_encoder_type='lstm'. For 'sbert' mode,
        use external SBERT encoder to get state directly.

        Args:
            item_embeddings: [seq_len, item_dim] - SBERT embeddings of items
            ratings: List of "liked", "disliked", or "unknown"/"not_seen"

        Returns:
            state: [state_dim] numpy array
        """
        if self.state_encoder is None:
            raise ValueError("encode_state_from_sequence requires state_encoder_type='lstm'")

        if len(item_embeddings) == 0:
            return np.zeros(self.state_dim)

        # Convert to tensors
        item_tensor = torch.FloatTensor(item_embeddings).unsqueeze(0)  # [1, seq_len, item_dim]
        rating_one_hot = create_rating_batch(ratings).unsqueeze(0)  # [1, seq_len, 3]
        mask = torch.ones(1, len(ratings), dtype=torch.bool)  # [1, seq_len]

        with torch.no_grad():
            state = self.state_encoder(item_tensor, rating_one_hot, mask)

        return state.squeeze(0).numpy()

    def predict_embedding(
        self,
        state: np.ndarray,
        explore: bool = True
    ) -> np.ndarray:
        """
        Predict SentenceBERT embedding from conversation state.

        Args:
            state: Conversation state embedding (384-dim)
            explore: Whether to add exploration noise

        Returns:
            Predicted embedding (384-dim)
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0)

        with torch.no_grad():
            embedding = self.actor(state_tensor).squeeze().numpy()

        # Add exploration noise
        if explore:
            noise = np.random.normal(0, self.noise_scale, size=embedding.shape)
            embedding = embedding + noise
            embedding = np.clip(embedding, -1, 1)

        # Normalize to unit vector for cosine similarity
        embedding = embedding / (np.linalg.norm(embedding) + 1e-8)

        return embedding

    def store_experience(self, state, embedding, reward, next_state, done):
        """Store experience for replay."""
        self.memory.append((state, embedding, reward, next_state, done))

        if len(self.memory) > 10000:
            self.memory.pop(0)

    def train_step(self, batch_size: int = 32):
        """
        DDPG-style actor-critic training step with target networks.

        Key stability features:
        1. Target networks for stable Q-learning
        2. Soft target updates (Polyak averaging)
        3. Reward normalization
        4. Gradient clipping
        """
        if len(self.memory) < batch_size:
            return None

        # Sample batch
        batch = random.sample(self.memory, batch_size)
        states, embeddings, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(np.array(states))
        embeddings = torch.FloatTensor(np.array(embeddings))
        rewards_np = np.array(rewards)
        next_states = torch.FloatTensor(np.array(next_states))
        dones = torch.BoolTensor(dones)

        # Update reward statistics and normalize
        self._update_reward_stats(rewards_np)
        normalized_rewards = torch.FloatTensor(
            [self._normalize_reward(r) for r in rewards_np]
        )

        # --- Update Critic ---
        # Use TARGET networks for stable Q-learning (key fix!)
        with torch.no_grad():
            next_embeddings = self.actor_target(next_states)
            next_q = self.critic_target(
                torch.cat([next_states, next_embeddings], dim=1)
            ).squeeze()
            next_q[dones] = 0.0
            target_q = normalized_rewards + self.gamma * next_q

        current_q = self.critic(torch.cat([states, embeddings], dim=1)).squeeze()
        critic_loss = F.mse_loss(current_q, target_q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)  # Stricter clipping
        self.critic_optimizer.step()

        # --- Update Actor ---
        pred_embeddings = self.actor(states)
        actor_q = self.critic(torch.cat([states, pred_embeddings], dim=1)).squeeze()
        actor_loss = -actor_q.mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
        self.actor_optimizer.step()

        # --- Soft Update Target Networks ---
        self._soft_update(self.actor_target, self.actor)
        self._soft_update(self.critic_target, self.critic)

        # Decay exploration
        if self.noise_scale > self.noise_min:
            self.noise_scale *= self.noise_decay

        return {
            'critic_loss': critic_loss.item(),
            'actor_loss': actor_loss.item(),
            'avg_q': current_q.mean().item(),
            'max_q': current_q.max().item(),
            'noise': self.noise_scale,
            'reward_mean': self.reward_mean,
            'reward_std': self.reward_std
        }

    def clear_replay_buffer(self):
        """
        Clear replay buffer while keeping network weights.

        Use this when:
        - Reward function has changed significantly
        - Starting fresh training with updated hyperparameters
        - Old experiences are no longer valid for new reward signal
        """
        old_size = len(self.memory)
        self.memory = []
        # Also reset reward normalization stats since they're based on old rewards
        self.reward_mean = 0.0
        self.reward_std = 1.0
        self.reward_count = 0
        print(f"Cleared replay buffer ({old_size} experiences) and reset reward stats")

    def save_weights(self, path: str):
        """
        Save only network weights (not replay buffer or optimizer state).

        Args:
            path: Path to save weights (e.g., 'checkpoints/rl_weights.pt')
        """
        import torch
        state = {
            'actor': self.actor.state_dict(),
            'actor_target': self.actor_target.state_dict(),
            'critic': self.critic.state_dict(),
            'critic_target': self.critic_target.state_dict(),
            'noise_scale': self.noise_scale,
        }
        if self.state_encoder is not None:
            state['state_encoder'] = self.state_encoder.state_dict()
        torch.save(state, path)
        print(f"Saved RL weights to {path}")

    def load_weights(self, path: str):
        """
        Load network weights (leaves replay buffer empty).

        Args:
            path: Path to load weights from
        """
        import torch
        state = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(state['actor'])
        self.actor_target.load_state_dict(state['actor_target'])
        self.critic.load_state_dict(state['critic'])
        self.critic_target.load_state_dict(state['critic_target'])
        self.noise_scale = state.get('noise_scale', self.noise_scale)
        if self.state_encoder is not None and 'state_encoder' in state:
            self.state_encoder.load_state_dict(state['state_encoder'])
        print(f"Loaded RL weights from {path}")
