"""
Reward functions for CASPER RL training.

Only includes rewards based on recommender performance metrics.
"""

import numpy as np
from typing import Dict, List, Optional


class BaseReward:
    """Base class for reward functions."""

    def __init__(self):
        self.history = []

    def calculate(self, **kwargs) -> float:
        """Calculate reward. Override in subclasses."""
        raise NotImplementedError

    def reset(self):
        """Reset reward history."""
        self.history = []


class NDCGDeltaReward(BaseReward):
    """
    Reward based on NDCG@10 improvement: reward = NDCG@10(t) - NDCG@10(t-1)

    Directly optimizes ranking quality - the question's value is measured by
    how much it improves recommendation quality.

    Pros:
    - Direct optimization of ranking metric
    - Measures actual recommendation improvement from the question

    Cons:
    - Can be noisy when NDCG is stuck at 0
    - Requires recommender forward pass every turn
    """

    def __init__(self):
        super().__init__()
        self.ndcg_history = []

    def calculate(self, current_ndcg: float) -> float:
        """
        Args:
            current_ndcg: NDCG@10 at current turn

        Returns:
            Reward (delta from previous turn)
        """
        self.ndcg_history.append(current_ndcg)

        if len(self.ndcg_history) > 1:
            reward = current_ndcg - self.ndcg_history[-2]
        else:
            reward = current_ndcg  # First turn

        self.history.append(reward)
        return reward

    def reset(self):
        super().reset()
        self.ndcg_history = []


class RecommenderLossReward(BaseReward):
    """
    Reward based on recommender loss improvement: reward = -(current_loss - previous_loss)

    Uses recommender's BPR loss as reward signal - decreasing loss = better recommendations.

    Pros:
    - Smooth gradient signal
    - Directly optimizes recommender's training objective
    - Measures how much the question helps the recommender learn

    Cons:
    - SLOW (requires forward pass + loss computation every turn)
    - Loss doesn't directly correspond to ranking quality (loss != NDCG)
    """

    def __init__(self):
        super().__init__()
        self.prev_loss = None

    def calculate(self, current_loss: float) -> float:
        """
        Args:
            current_loss: BPR loss from recommender at current turn

        Returns:
            Negative loss delta (decreasing loss = positive reward)
        """
        if self.prev_loss is not None:
            reward = -(current_loss - self.prev_loss)
        else:
            reward = -current_loss  # First turn

        self.prev_loss = current_loss
        self.history.append(reward)

        return reward

    def reset(self):
        super().reset()
        self.prev_loss = None


def get_reward_function(name: str, **kwargs):
    """
    Factory function to get reward by name.

    Args:
        name: One of ["ndcg_delta", "recommender_loss"]
        **kwargs: Additional arguments for reward function

    Returns:
        Reward function instance
    """
    rewards = {
        "ndcg_delta": NDCGDeltaReward,
        "recommender_loss": RecommenderLossReward,
    }

    if name not in rewards:
        raise ValueError(f"Unknown reward function: {name}. Choose from {list(rewards.keys())}")

    return rewards[name](**kwargs)
