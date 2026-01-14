"""
Pure metric implementations for conversational recommendation evaluation.

All functions are stateless and take data as input, return metrics as output.
No logging, no orchestration - just metric computation.
"""

import numpy as np
from typing import List, Dict, Any, Tuple


def ndcg_at_k(recommended_ids: List[int], target_ids: List[int], k: int = 10) -> float:
    """
    Calculate Normalized Discounted Cumulative Gain at k.

    Args:
        recommended_ids: List of recommended item IDs (ranked)
        target_ids: List of ground truth target item IDs
        k: Cutoff for NDCG calculation

    Returns:
        NDCG@k score (0.0 to 1.0)
    """
    # Limit to top k recommendations
    recommended_ids = recommended_ids[:k]

    # Calculate DCG (Discounted Cumulative Gain)
    dcg = 0.0
    for i, item_id in enumerate(recommended_ids):
        if item_id in target_ids:
            relevance = 1.0
            # Discount by log2(position + 1), position is 1-indexed
            dcg += relevance / np.log2(i + 2)

    # Calculate IDCG (Ideal DCG)
    ideal_relevances = [1.0] * min(len(target_ids), k)
    idcg = sum(rel / np.log2(i + 2) for i, rel in enumerate(ideal_relevances))

    # Avoid division by zero
    if idcg == 0.0:
        return 0.0

    return dcg / idcg


def hit_rate_at_k(recommended_ids: List[int], target_ids: List[int], k: int = 5) -> float:
    """
    Calculate hit rate at k (proportion of targets in top-k recommendations).

    Args:
        recommended_ids: List of recommended item IDs (ranked)
        target_ids: List of ground truth target item IDs
        k: Cutoff for hit rate calculation

    Returns:
        Hit rate (0.0 to 1.0)
    """
    recommended_ids = set(recommended_ids[:k])
    target_ids = set(target_ids)

    if not target_ids:
        return 0.0

    hits = len(recommended_ids & target_ids)
    return hits / len(target_ids)


def mrr(recommended_ids: List[int], target_ids: List[int]) -> float:
    """
    Calculate Mean Reciprocal Rank.

    Args:
        recommended_ids: List of recommended item IDs (ranked)
        target_ids: List of ground truth target item IDs

    Returns:
        MRR score (0.0 to 1.0)
    """
    for i, item_id in enumerate(recommended_ids):
        if item_id in target_ids:
            return 1.0 / (i + 1)
    return 0.0


def precision_at_k(recommended_ids: List[int], target_ids: List[int], k: int = 10) -> float:
    """
    Calculate precision at k.

    Args:
        recommended_ids: List of recommended item IDs (ranked)
        target_ids: List of ground truth target item IDs
        k: Cutoff for precision calculation

    Returns:
        Precision@k (0.0 to 1.0)
    """
    recommended_ids = set(recommended_ids[:k])
    target_ids = set(target_ids)

    if not recommended_ids:
        return 0.0

    hits = len(recommended_ids & target_ids)
    return hits / k


def recall_at_k(recommended_ids: List[int], target_ids: List[int], k: int = 10) -> float:
    """
    Calculate recall at k.

    Args:
        recommended_ids: List of recommended item IDs (ranked)
        target_ids: List of ground truth target item IDs
        k: Cutoff for recall calculation

    Returns:
        Recall@k (0.0 to 1.0)
    """
    recommended_ids = set(recommended_ids[:k])
    target_ids = set(target_ids)

    if not target_ids:
        return 0.0

    hits = len(recommended_ids & target_ids)
    return hits / len(target_ids)


def conversation_efficiency(num_turns: int, num_discoveries: int, max_turns: int = 10) -> float:
    """
    Calculate conversation efficiency.

    Rewards discovering preferences quickly with fewer questions.

    Args:
        num_turns: Number of conversation turns taken
        num_discoveries: Number of held-out items discovered
        max_turns: Maximum allowed turns

    Returns:
        Efficiency score (higher = more efficient)
    """
    if num_discoveries == 0:
        return 0.0

    # Reward: discoveries per turn (higher is better)
    efficiency = num_discoveries / num_turns

    # Bonus for finishing early
    if num_turns < max_turns:
        early_bonus = (max_turns - num_turns) / max_turns
        efficiency *= (1.0 + early_bonus)

    return efficiency


def information_gain(
    preferences_t: Dict[str, List[str]],
    preferences_t_minus_1: Dict[str, List[str]]
) -> int:
    """
    Calculate information gain between two preference states.

    Measures how many new preferences were discovered in a turn.

    Args:
        preferences_t: Current preference state
        preferences_t_minus_1: Previous preference state

    Returns:
        Number of new preferences discovered
    """
    def flatten_prefs(prefs: Dict[str, List[str]]) -> set:
        items = []
        for category_items in prefs.values():
            items.extend(category_items)
        return set(items)

    current = flatten_prefs(preferences_t)
    previous = flatten_prefs(preferences_t_minus_1)

    return len(current - previous)


def aggregate_metrics(metric_values: List[float]) -> Dict[str, float]:
    """
    Aggregate metric values across multiple conversations.

    Args:
        metric_values: List of metric values

    Returns:
        Dict with mean, median, std, min, max
    """
    if not metric_values:
        return {
            'mean': 0.0,
            'median': 0.0,
            'std': 0.0,
            'min': 0.0,
            'max': 0.0
        }

    return {
        'mean': float(np.mean(metric_values)),
        'median': float(np.median(metric_values)),
        'std': float(np.std(metric_values)),
        'min': float(np.min(metric_values)),
        'max': float(np.max(metric_values))
    }
