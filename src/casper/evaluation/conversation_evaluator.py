"""
Evaluation for conversational recommendation systems.

Consumes conversation logs and computes metrics using evaluation/metrics.py.
Does NOT orchestrate conversations - see training/runner.py for that.

DESIGN: Background Recommendations Only
- Agent only asks questions (recommendations not shown to user)
- Recommendations computed in background for reward calculation
- Evaluator focuses on preference discovery quality and efficiency
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from . import metrics


class ConversationalEvaluator:
    """
    Computes evaluation metrics for conversational recommendation systems.

    Consumes conversation logs and returns metrics.
    Does NOT run conversations or interact with agents/simulators.

    Key metrics:
    - Preference discovery: How well agent learns user preferences
    - Recommendation quality: NDCG, hit rate, MRR on held-out items
    - Conversation efficiency: Questions needed to discover preferences
    """

    def __init__(
        self,
        holdout_ratio: float = 0.3,
        min_rating_threshold: float = 4.0,
        seed: int = 42
    ):
        """
        Initialize evaluator.

        Args:
            holdout_ratio: Fraction of high-rated items to hold out for testing
            min_rating_threshold: Minimum rating to consider an item "liked"
            seed: Random seed for reproducible splits
        """
        self.holdout_ratio = holdout_ratio
        self.min_rating_threshold = min_rating_threshold
        self.random_state = np.random.RandomState(seed)

    def prepare_user_profile(self, full_profile: Dict) -> Tuple[Dict, Dict]:
        """
        Split user profile into training and test sets.

        IMPORTANT: Agent is BLIND to both profiles!
        - training_profile is for USER SIMULATOR only (to generate realistic responses)
        - Agent only knows what user mentions in conversation
        - test_set is hidden from BOTH agent and simulator

        Args:
            full_profile: Complete user profile
                         Format: {'user_id': int, 'ratings': {movie: rating}, ...}

        Returns:
            training_profile: Profile with held-out movies removed (for SIMULATOR only)
            test_set: Held-out movies for evaluation (hidden from both)
        """
        if 'ratings' not in full_profile:
            raise ValueError("Profile must contain 'ratings' field")

        # Get all high-rated movies (above threshold)
        high_rated_movies = [
            movie for movie, rating in full_profile['ratings'].items()
            if rating >= self.min_rating_threshold
        ]

        # Shuffle and split
        self.random_state.shuffle(high_rated_movies)
        split_point = int(len(high_rated_movies) * (1 - self.holdout_ratio))

        training_movies = high_rated_movies[:split_point]
        test_movies = high_rated_movies[split_point:]

        # Create training profile (without test movies) - FOR SIMULATOR ONLY
        training_ratings = {
            movie: rating
            for movie, rating in full_profile['ratings'].items()
            if movie in training_movies or rating < self.min_rating_threshold
        }

        training_profile = {
            'user_id': full_profile.get('user_id'),
            'ratings': training_ratings
        }

        # Add other profile fields if present
        for key in full_profile:
            if key not in ['ratings', 'user_id']:
                training_profile[key] = full_profile[key]

        # Create test set - HIDDEN FROM BOTH
        test_set = {
            'held_out_movies': test_movies,
            'total_liked_movies': len(high_rated_movies),
            'user_id': full_profile.get('user_id')
        }

        return training_profile, test_set

    def evaluate_conversation(
        self,
        conversation: List[Dict],
        recommended_ids: List[int],
        test_set: Dict,
        training_profile: Dict
    ) -> Dict[str, Any]:
        """
        Evaluate a single conversation with explicit recommendations.

        Args:
            conversation: List of turns [{'role': 'agent'/'user', 'content': str}, ...]
            recommended_ids: Final recommendation IDs (ranked)
            test_set: Held-out movies for this user
            training_profile: Training profile used

        Returns:
            Metrics dictionary
        """
        # Get target movie IDs from test set
        # Note: This assumes test_set['held_out_movies'] contains item IDs or titles
        # If titles, caller should convert to IDs first
        target_ids = test_set.get('held_out_movie_ids', [])
        if not target_ids:
            # Fallback: use held_out_movies if movie_ids not provided
            target_ids = test_set.get('held_out_movies', [])

        # Recommendation quality metrics
        ndcg_10 = metrics.ndcg_at_k(recommended_ids, target_ids, k=10)
        ndcg_5 = metrics.ndcg_at_k(recommended_ids, target_ids, k=5)
        hit_rate_10 = metrics.hit_rate_at_k(recommended_ids, target_ids, k=10)
        hit_rate_5 = metrics.hit_rate_at_k(recommended_ids, target_ids, k=5)
        mrr_score = metrics.mrr(recommended_ids, target_ids)
        precision_10 = metrics.precision_at_k(recommended_ids, target_ids, k=10)
        recall_10 = metrics.recall_at_k(recommended_ids, target_ids, k=10)

        # Conversation metrics
        num_turns = len([turn for turn in conversation if turn.get('role') == 'user'])
        num_questions = len([turn for turn in conversation if turn.get('role') == 'agent'])

        # Discovery metrics
        recommended_set = set(recommended_ids[:10])
        target_set = set(target_ids)
        hits = recommended_set & target_set

        # Categorize recommendations
        successful_recs = list(hits)  # In test set (discovered!)
        num_discoveries = len(successful_recs)

        # Efficiency
        efficiency = metrics.conversation_efficiency(
            num_turns=num_turns,
            num_discoveries=num_discoveries,
            max_turns=20
        )

        return {
            # Recommendation quality
            'ndcg@10': ndcg_10,
            'ndcg@5': ndcg_5,
            'hit_rate@10': hit_rate_10,
            'hit_rate@5': hit_rate_5,
            'mrr': mrr_score,
            'precision@10': precision_10,
            'recall@10': recall_10,

            # Discovery
            'num_discoveries': num_discoveries,
            'discovery_rate': num_discoveries / len(target_ids) if target_ids else 0.0,
            'successful_recommendations': successful_recs,

            # Conversation
            'num_turns': num_turns,
            'num_questions': num_questions,
            'efficiency': efficiency,

            # Metadata
            'total_targets': len(target_ids),
            'user_id': test_set.get('user_id')
        }

    def evaluate_multiple_conversations(
        self,
        conversations: List[Dict],
        aggregate: bool = True
    ) -> Dict[str, Any]:
        """
        Evaluate multiple conversations and optionally aggregate results.

        Args:
            conversations: List of conversation dicts with format:
                {
                    'conversation': List[Dict],  # Conversation log
                    'recommended_ids': List[int],  # Final recommendations
                    'test_set': Dict,  # Held-out movies
                    'training_profile': Dict,  # Training profile
                }
            aggregate: Return aggregated stats or individual results

        Returns:
            Aggregated metrics dict or list of individual metrics
        """
        all_results = []

        for conv_data in conversations:
            metrics_dict = self.evaluate_conversation(
                conversation=conv_data['conversation'],
                recommended_ids=conv_data['recommended_ids'],
                test_set=conv_data['test_set'],
                training_profile=conv_data['training_profile']
            )

            all_results.append(metrics_dict)

        if aggregate:
            aggregated = self._aggregate_results(all_results)
            aggregated['detailed_results'] = all_results
            return aggregated
        else:
            return all_results

    def _aggregate_results(self, results: List[Dict]) -> Dict[str, Any]:
        """
        Aggregate metrics across multiple conversations.
        """
        if not results:
            return {}

        df = pd.DataFrame(results)

        # Define metric columns
        metric_cols = [
            'ndcg@10', 'ndcg@5', 'hit_rate@10', 'hit_rate@5',
            'mrr', 'precision@10', 'recall@10',
            'num_discoveries', 'discovery_rate',
            'num_turns', 'num_questions', 'efficiency'
        ]

        aggregated = {
            'num_conversations': len(results)
        }

        # Aggregate each metric
        for col in metric_cols:
            if col in df.columns:
                col_values = df[col].dropna()
                if len(col_values) > 0:
                    agg = metrics.aggregate_metrics(col_values.tolist())
                    aggregated[col] = agg

        # Success rates
        aggregated['success_rate'] = (df['num_discoveries'] > 0).mean()
        aggregated['perfect_rate'] = (df['discovery_rate'] == 1.0).mean()

        return aggregated
