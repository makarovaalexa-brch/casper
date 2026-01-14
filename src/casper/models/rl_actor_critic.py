"""
Actor-Critic RL agent for continuous action space (semantic embeddings).

Uses DDPG-style training to predict embeddings in SentenceBERT space.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import random


class EmbeddingActorCritic:
    """
    Actor-Critic RL agent that predicts continuous embeddings.

    Actor: State → SentenceBERT embedding (384-dim)
    Critic: State + Embedding → Q-value

    Uses DDPG-style training for continuous action space.
    """

    def __init__(
        self,
        state_dim: int = 384,  # SentenceBERT dimension
        embedding_dim: int = 384,  # SentenceBERT dimension
        hidden_dim: int = 512,
        learning_rate: float = 0.001,
        exploration_noise: float = 0.2
    ):
        self.state_dim = state_dim
        self.embedding_dim = embedding_dim
        self.exploration_noise = exploration_noise

        # Actor network: state → embedding
        self.actor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, embedding_dim),
            nn.Tanh()  # Output in [-1, 1]
        )

        # Critic network: state + embedding → Q-value
        self.critic = nn.Sequential(
            nn.Linear(state_dim + embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

        # Optimizers
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=learning_rate)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=learning_rate * 2)

        # Experience replay
        self.memory = []

        # Exploration
        self.noise_scale = exploration_noise
        self.noise_decay = 0.995
        self.noise_min = 0.05

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

    def train_step(self, batch_size: int = 32, gamma: float = 0.99):
        """Actor-critic training step."""
        if len(self.memory) < batch_size:
            return None

        # Sample batch
        batch = random.sample(self.memory, batch_size)
        states, embeddings, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(np.array(states))
        embeddings = torch.FloatTensor(np.array(embeddings))
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(np.array(next_states))
        dones = torch.BoolTensor(dones)

        # --- Update Critic ---
        with torch.no_grad():
            next_embeddings = self.actor(next_states)
            next_q = self.critic(torch.cat([next_states, next_embeddings], dim=1)).squeeze()
            next_q[dones] = 0.0
            target_q = rewards + gamma * next_q

        current_q = self.critic(torch.cat([states, embeddings], dim=1)).squeeze()
        critic_loss = F.mse_loss(current_q, target_q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.critic_optimizer.step()

        # --- Update Actor ---
        pred_embeddings = self.actor(states)
        actor_q = self.critic(torch.cat([states, pred_embeddings], dim=1)).squeeze()
        actor_loss = -actor_q.mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.actor_optimizer.step()

        # Decay exploration
        if self.noise_scale > self.noise_min:
            self.noise_scale *= self.noise_decay

        return {
            'critic_loss': critic_loss.item(),
            'actor_loss': actor_loss.item(),
            'avg_q': current_q.mean().item(),
            'noise': self.noise_scale
        }
