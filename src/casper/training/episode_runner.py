"""
Episode runner for CASPER RL training.

Runs conversation episodes between agent and user simulator,
computing rewards based on LLM-based NDCG.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from casper.evaluation.llm_reward import LLMRewardCalculator


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


class EpisodeRunner:
    """
    Runs training episodes for CASPER agent.

    Each episode:
    1. Sample user from training set
    2. Prepare user profile (holdout for evaluation)
    3. Run conversation: agent asks, user responds
    4. Calculate LLM-based NDCG reward per turn
    5. Return rewards for RL training
    """

    def __init__(
        self,
        reward_calculator: LLMRewardCalculator,
        user_simulator,
        max_turns: int = 10,
        holdout_ratio: float = 0.0,  # Default: no holdout, use all movies
        min_rating: float = 4.0,
        exclude_mentioned: bool = True,  # Exclude movies user mentions from eval
        # Reward shaping parameters
        ndcg_weight: float = 1.0,  # Weight for NDCG improvement
        new_pref_bonus: float = 0.05,  # Bonus per new preference discovered
        not_seen_penalty: float = 0.05,  # Penalty per "not_seen" item added
        use_baseline: bool = False,  # If False, baseline=0 (no penalty for first turn)
        # V3: Similarity penalty - penalize exploring same embedding space regions
        similarity_threshold: float = 0.7,  # Penalize if predicted emb is > this similar to known
        similarity_penalty: float = 0.1,  # How much to penalize (multiplied by similarity)
    ):
        """
        Initialize episode runner.

        Args:
            reward_calculator: LLMRewardCalculator instance
            user_simulator: User simulator instance
            max_turns: Maximum conversation turns per episode
            holdout_ratio: Fraction of user's TOP_N movies to hold out (0 = use all)
            min_rating: Minimum rating for "liked" movies
            exclude_mentioned: If True, don't count movies user mentions as eval targets
            ndcg_weight: Weight for NDCG improvement component
            new_pref_bonus: Bonus per new liked/disliked preference
            not_seen_penalty: Penalty per new "not_seen" item
            use_baseline: If False, prev_ndcg starts at 0 (not LLM baseline)
            similarity_threshold: V3 - Penalize if predicted embedding similarity > this
            similarity_penalty: V3 - Penalty weight (multiplied by max similarity)
        """
        self.reward_calc = reward_calculator
        self.user_sim = user_simulator
        self.max_turns = max_turns
        self.holdout_ratio = holdout_ratio
        self.min_rating = min_rating
        self.exclude_mentioned = exclude_mentioned
        # Reward shaping
        self.ndcg_weight = ndcg_weight
        self.new_pref_bonus = new_pref_bonus
        self.not_seen_penalty = not_seen_penalty
        self.use_baseline = use_baseline
        # V3: Similarity penalty
        self.similarity_threshold = similarity_threshold
        self.similarity_penalty = similarity_penalty

    def _calculate_similarity_penalty(
        self,
        predicted_embedding: np.ndarray,
        known_items: list,
        encoder
    ) -> Tuple[float, float]:
        """
        Calculate similarity penalty for exploring same embedding space regions.

        V3: Penalizes the model when its predicted embedding is too similar to
        items already discovered. This encourages exploration of new concepts.

        Args:
            predicted_embedding: The embedding predicted by RL actor
            known_items: List of items already in discovered_preferences
            encoder: SentenceBERT encoder for embedding known items

        Returns:
            Tuple of (penalty_amount, max_similarity)
        """
        if not known_items or predicted_embedding is None:
            return 0.0, 0.0

        # Encode all known items
        known_embeddings = encoder.encode(known_items, convert_to_numpy=True)

        # Find max similarity to any known item
        max_sim = 0.0
        for known_emb in known_embeddings:
            sim = cosine_similarity(predicted_embedding, known_emb)
            max_sim = max(max_sim, sim)

        # Apply penalty if above threshold
        if max_sim > self.similarity_threshold:
            penalty = self.similarity_penalty * max_sim
        else:
            penalty = 0.0

        return penalty, max_sim

    def run_episode(
        self,
        agent,
        user_profile: Dict,
        seed: int = None,
        verbose: bool = False
    ) -> Dict[str, Any]:
        """
        Run a single training episode.

        Args:
            agent: CASPER agent
            user_profile: Full user profile
            seed: Random seed for holdout split
            verbose: Print debug info

        Returns:
            Episode result dict with rewards, NDCG, turn logs, etc.
        """
        # Prepare user for evaluation
        # Default: use ALL high-rated movies in TOP_N as ground truth (no holdout)
        training_profile, all_eval_ids, eval_titles = self.reward_calc.prepare_user_for_evaluation(
            user_profile,
            holdout_ratio=self.holdout_ratio,
            min_rating=self.min_rating,
            seed=seed
        )

        # Build ground truth info
        user_ground_truth = {
            'user_id': user_profile.get('user_id', 'unknown'),
            'num_ratings': len(user_profile.get('ratings', {})),
            'all_eval_target_ids': all_eval_ids,
            'eval_movie_titles': eval_titles,
            'has_valid_targets': len(all_eval_ids) > 0,
            'num_eval_targets': len(all_eval_ids)
        }

        # Reset agent state
        agent.reset_conversation()

        rewards = []
        turn_logs = []

        # Handle users with no valid evaluation targets
        if not all_eval_ids:
            for turn in range(self.max_turns):
                question, rl_debug = agent.ask_question(
                    explore=True, verbose=False, return_debug_info=True
                )
                response, tool_calls = self.user_sim.simulate_response(
                    question=question,
                    user_profile=training_profile,
                    conversation_history=agent.conversation_history,
                    return_tool_calls=True
                )
                agent.process_user_response(response, user_target_movies=None)
                rewards.append(0.0)
                turn_logs.append({
                    "turn": turn + 1,
                    "agent_question": question,
                    "user_response": response,
                    "ndcg": 0.0,
                    "reward": 0.0,
                    "note": "No valid evaluation targets in TOP_N"
                })

            return {
                'rewards': rewards,
                'final_ndcg': 0.0,
                'turn_logs': turn_logs,
                'user_ground_truth': user_ground_truth,
                'total_reward': 0.0
            }

        # Calculate baseline NDCG (before any questions)
        baseline_ndcg = self.reward_calc.calculate_baseline_ndcg(all_eval_ids)

        # Use baseline=0 for reward calculation (so first improvement is rewarded)
        # Keep baseline_ndcg for logging/comparison purposes
        prev_ndcg = baseline_ndcg if self.use_baseline else 0.0

        # Track states and embeddings for RL training
        states = []
        embeddings = []

        # Track preference counts for reward shaping
        prev_liked_count = 0
        prev_disliked_count = 0
        prev_not_seen_count = 0

        for turn in range(self.max_turns):
            # Capture state BEFORE asking question (for RL training)
            state = agent.encode_conversation_state()
            states.append(state)

            # Agent asks question
            # Turn 1 is greeting only (no RL), turn 2+ uses RL
            question, rl_debug = agent.ask_question(
                explore=True, verbose=verbose, return_debug_info=True
            )

            # Capture the embedding predicted by RL
            if agent.embedding_history:
                embeddings.append(agent.embedding_history[-1])

            # User responds
            response, tool_calls = self.user_sim.simulate_response(
                question=question,
                user_profile=training_profile,
                conversation_history=agent.conversation_history,
                return_tool_calls=True
            )

            # Process response
            agent.process_user_response(response, user_target_movies=None)

            # Turn 1: Skip reward (greeting only, RL not used)
            if turn == 0:
                rewards.append(0.0)

                # Count preferences discovered on turn 1 (for accurate logging)
                curr_liked_count = len(agent.discovered_preferences.get('liked', []))
                curr_disliked_count = len(agent.discovered_preferences.get('disliked', []))
                curr_not_seen_count = len(agent.discovered_preferences.get('not_seen', []))

                turn_logs.append({
                    "turn": turn + 1,
                    "note": "Turn 1 greeting - RL not used, no reward",

                    # 1. RL DECISION (not used on turn 1)
                    "rl_decision": rl_debug,

                    # 2. AGENT ACTION
                    "agent": {
                        "question": question
                    },

                    # 3. USER RESPONSE
                    "user": {
                        "response": response,
                        "tool_calls": tool_calls
                    },

                    # 4. PREFERENCE UPDATE
                    "preferences": {
                        "new_this_turn": {
                            "liked": curr_liked_count - prev_liked_count,
                            "disliked": curr_disliked_count - prev_disliked_count,
                            "not_seen": curr_not_seen_count - prev_not_seen_count
                        },
                        "cumulative": {
                            "liked": agent.discovered_preferences.get('liked', []),
                            "disliked": agent.discovered_preferences.get('disliked', []),
                            "not_seen": agent.discovered_preferences.get('not_seen', [])
                        }
                    },

                    # 5. REWARD (no reward for turn 1, but log what it would be)
                    "reward": {
                        "baseline_ndcg": baseline_ndcg,
                        "total": 0.0,
                        "note": "No reward on turn 1 (greeting)"
                    }
                })

                # Update counts for turn 2 delta calculation
                prev_liked_count = curr_liked_count
                prev_disliked_count = curr_disliked_count
                prev_not_seen_count = curr_not_seen_count
                continue

            # Turn 2+: Calculate reward based on NDCG improvement + preference shaping
            if self.exclude_mentioned:
                # Exclude all explicitly mentioned movies (liked, disliked, not_seen)
                mentioned = (
                    agent.discovered_preferences.get('liked', []) +
                    agent.discovered_preferences.get('disliked', []) +
                    agent.discovered_preferences.get('not_seen', [])
                )
                eval_target_ids = self.reward_calc.filter_mentioned_movies(all_eval_ids, mentioned)
            else:
                eval_target_ids = all_eval_ids

            # Get current NDCG (ignore the raw reward from calculate_turn_reward)
            current_ndcg, _ = self.reward_calc.calculate_turn_reward(
                preferences=agent.discovered_preferences,
                eval_target_ids=eval_target_ids,
                prev_ndcg=0.0  # We calculate delta ourselves
            )

            # Count current preferences
            curr_liked_count = len(agent.discovered_preferences.get('liked', []))
            curr_disliked_count = len(agent.discovered_preferences.get('disliked', []))
            curr_not_seen_count = len(agent.discovered_preferences.get('not_seen', []))

            # Calculate new preferences discovered this turn
            new_liked = curr_liked_count - prev_liked_count
            new_disliked = curr_disliked_count - prev_disliked_count
            new_not_seen = curr_not_seen_count - prev_not_seen_count
            new_useful_prefs = new_liked + new_disliked  # Both liked and disliked are useful

            # V3: Calculate similarity penalty
            # Penalize if predicted embedding is too similar to known items
            sim_penalty = 0.0
            max_sim = 0.0
            if agent.embedding_history:
                predicted_emb = agent.embedding_history[-1]
                # Get all known items (preferences discovered so far)
                known_items = (
                    agent.discovered_preferences.get('liked', []) +
                    agent.discovered_preferences.get('disliked', []) +
                    agent.discovered_preferences.get('not_seen', [])
                )
                if known_items:
                    sim_penalty, max_sim = self._calculate_similarity_penalty(
                        predicted_emb, known_items, agent.encoder
                    )

            # Calculate shaped reward
            ndcg_delta = current_ndcg - prev_ndcg
            shaped_reward = (
                self.ndcg_weight * ndcg_delta +
                self.new_pref_bonus * new_useful_prefs -
                self.not_seen_penalty * new_not_seen -
                sim_penalty  # V3: Subtract similarity penalty
            )

            rewards.append(shaped_reward)
            prev_ndcg = current_ndcg

            # Update previous counts for next turn
            prev_liked_count = curr_liked_count
            prev_disliked_count = curr_disliked_count
            prev_not_seen_count = curr_not_seen_count

            # Log turn in logical flow order:
            # 1. RL Decision → 2. Agent Question → 3. User Response → 4. Preferences → 5. Reward
            turn_logs.append({
                "turn": turn + 1,

                # 1. RL DECISION (what the model predicted)
                "rl_decision": {
                    "selected_entity": rl_debug.get('selected_entity'),
                    "candidates": rl_debug.get('nearest_entities', [])
                },

                # 2. AGENT ACTION
                "agent": {
                    "question": question
                },

                # 3. USER RESPONSE
                "user": {
                    "response": response,
                    "tool_calls": tool_calls
                },

                # 4. PREFERENCE UPDATE
                "preferences": {
                    "new_this_turn": {
                        "liked": new_liked,
                        "disliked": new_disliked,
                        "not_seen": new_not_seen
                    },
                    "cumulative": {
                        "liked": agent.discovered_preferences.get('liked', []),
                        "disliked": agent.discovered_preferences.get('disliked', []),
                        "not_seen": agent.discovered_preferences.get('not_seen', [])
                    }
                },

                # 5. REWARD CALCULATION
                "reward": {
                    "prev_ndcg": prev_ndcg - ndcg_delta,  # What it was before this turn
                    "current_ndcg": current_ndcg,
                    "ndcg_delta": ndcg_delta,
                    "pref_bonus": self.new_pref_bonus * new_useful_prefs,
                    "not_seen_penalty": self.not_seen_penalty * new_not_seen,
                    "similarity_penalty": sim_penalty,  # V3: Penalty for exploring same region
                    "max_similarity": max_sim,  # V3: Max similarity to known items
                    "total": shaped_reward
                },

                # Metadata
                "eval_targets_remaining": len(eval_target_ids)
            })

        return {
            'rewards': rewards,
            'final_ndcg': prev_ndcg,
            'turn_logs': turn_logs,
            'user_ground_truth': user_ground_truth,
            'total_reward': sum(rewards),
            'baseline_ndcg': baseline_ndcg,
            # For RL training
            'states': states,
            'embeddings': embeddings
        }

    def run_baseline_episode(
        self,
        agent,
        user_profile: Dict,
        seed: int = None
    ) -> Dict[str, Any]:
        """
        Run baseline episode (evaluate at end only, no per-turn rewards).

        Args:
            agent: Baseline agent
            user_profile: Full user profile
            seed: Random seed

        Returns:
            Episode result with final NDCG
        """
        training_profile, eval_target_ids, _ = self.reward_calc.prepare_user_for_evaluation(
            user_profile,
            holdout_ratio=self.holdout_ratio,
            min_rating=self.min_rating,
            seed=seed
        )

        if not eval_target_ids:
            return {'final_ndcg': 0.0, 'rewards': [], 'turn_logs': []}

        agent.reset_conversation()

        for turn in range(self.max_turns):
            question = agent.ask_question(verbose=False)
            response = self.user_sim.simulate_response(
                question=question,
                user_profile=training_profile,
                conversation_history=agent.conversation_history
            )
            agent.process_user_response(response)

        # Calculate final NDCG
        final_ndcg, _ = self.reward_calc.calculate_turn_reward(
            preferences=agent.discovered_preferences,
            eval_target_ids=eval_target_ids,
            prev_ndcg=0.0
        )

        return {
            'final_ndcg': final_ndcg,
            'rewards': [],
            'turn_logs': []
        }
