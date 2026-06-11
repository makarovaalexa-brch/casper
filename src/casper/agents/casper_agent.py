"""
CASPER: Continuous Action Space Preference Elicitation via Reinforcement

Main conversational agent for preference elicitation.

DESIGN: Background Recommendations Only
- Agent ONLY asks questions (never shows recommendations to user)
- After each user response, compute top-10 recommendations in BACKGROUND
- Track NDCG@10 improvement per turn as RL reward
- Recommendations used ONLY for reward calculation, never shown
- Conversation continues indefinitely (no forced ending)
- Paper focus: "Learning What to Ask" not "Learning When to Recommend"

Pipeline per turn:
1. Encode conversation state (SentenceBERT)
2. RL predicts concept embedding (actor-critic)
3. Map embedding to nearest movie entities
4. GPT generates natural question from entities
5. User responds, update state
6. [BACKGROUND] Compute recommendations → Calculate NDCG → RL reward
7. Repeat (no recommendation to user)

Architecture:
- Embedding Space: SentenceBERT (384-dim)
- RL Agent: Actor-Critic (DDPG-style)
- Question Generator: GPT
- Recommender: Two-Tower model (for reward calculation only)
"""

import numpy as np
from typing import Optional, List, Tuple

from casper.models.embedding_space import SentenceBERTEmbeddingSpace
from casper.models.rl_actor_critic import EmbeddingActorCritic
from casper.models.question_generator import QuestionGenerator
from casper.models.two_tower_recommender import TwoTowerRecommender, MovieCatalog, RecommenderTrainer
from casper.models.preference_extractor import PreferenceExtractor


class CASPERAgent:
    """
    CASPER conversational recommendation agent.

    IMPORTANT: Agent is BLIND to user profiles!
    - Agent has NO access to user rating history or preferences
    - Agent ONLY knows what user explicitly mentions in conversation
    - This ensures realistic preference elicitation (no cheating)

    Pipeline:
    1. Encode conversation state (SentenceBERT)
    2. RL predicts SentenceBERT embedding (actor-critic)
    3. Map embedding to nearest movie entities
    4. GPT generates natural question from entities
    5. User responds, update state
    6. Provide recommendations when ready

    Architecture:
    - Embedding Space: SentenceBERT (384-dim)
    - RL Agent: Actor-Critic (DDPG-style)
    - Question Generator: GPT
    - Recommender: Two-Tower model (for background reward calculation only)
    """

    def __init__(
        self,
        movielens_data_path: Optional[str] = None,
        model_name: str = "gpt-5-nano",
        load_recommender: bool = True,
        embedding_space: Optional[SentenceBERTEmbeddingSpace] = None,
        debug_preference_extraction: bool = False
    ):
        """
        Initialize CASPER agent.

        Args:
            movielens_data_path: Path to MovieLens data (defaults to data/movielens)
            model_name: DEPRECATED - Model settings now loaded from config/config.yaml
            load_recommender: Whether to initialize recommender (requires MovieLens data)
            embedding_space: Optional pre-initialized embedding space to reuse (for efficiency)
            debug_preference_extraction: If True, print detailed debug info for preference extraction
        """
        # Default MovieLens path if not provided
        if movielens_data_path is None:
            from pathlib import Path
            project_root = Path(__file__).parent.parent.parent.parent
            movielens_data_path = str(project_root / "data" / "movielens")

        print("Initializing CASPER Agent...")

        # Initialize or reuse SentenceBERT embedding space
        if embedding_space is not None:
            print("  Using shared embedding space")
            self.embedding_space = embedding_space
        else:
            print("  Creating new embedding space")
            self.embedding_space = SentenceBERTEmbeddingSpace(movielens_data_path)

        # Initialize RL agent (actor-critic)
        # state_encoder_type: 'sbert' (default) or 'lstm' (Concept model style)
        self.state_encoder_type = 'sbert'  # Can be changed via set_state_encoder_type()
        self.rl_agent = EmbeddingActorCritic(
            state_dim=384,  # SentenceBERT
            embedding_dim=384,  # SentenceBERT
            state_encoder_type=self.state_encoder_type
        )

        # Track individual preference items for LSTM encoding
        self.preference_sequence = []  # List of (item_name, rating_type) tuples

        # Question generator
        self.question_generator = QuestionGenerator()

        # Preference extractor (LLM-based)
        self.preference_extractor = PreferenceExtractor()

        # Conversation state encoder (same as embedding space)
        self.encoder = self.embedding_space.encoder

        # Initialize recommender
        self.recommender = None
        if load_recommender and movielens_data_path:
            try:
                movie_catalog = MovieCatalog(movielens_data_path)
                recommender_model = TwoTowerRecommender(state_dim=384, embedding_dim=128)
                self.recommender = RecommenderTrainer(
                    recommender=recommender_model,
                    movie_catalog=movie_catalog,
                    encoder=self.encoder
                )
                print("Recommender initialized")
            except Exception as e:
                print(f"Warning: Could not initialize recommender: {e}")
        else:
            print("Warning: Recommender not initialized (no MovieLens data)")

        # Conversation state
        self.conversation_history = []
        self.discovered_preferences = {}
        self.embedding_history = []

        # Per-turn reward tracking
        self.ndcg_history = []  # Track NDCG@10 at each turn for reward calculation
        self.reward_history = []  # Track rewards for analysis

        # Debug mode
        self.debug_preference_extraction = debug_preference_extraction

        print("CASPER Agent ready")

    def set_state_encoder_type(self, encoder_type: str):
        """
        Set the state encoder type and reinitialize RL agent.

        Args:
            encoder_type: 'sbert' (text encoding) or 'lstm' (sequence encoding)
        """
        if encoder_type not in ['sbert', 'lstm']:
            raise ValueError(f"Invalid encoder type: {encoder_type}. Use 'sbert' or 'lstm'")

        self.state_encoder_type = encoder_type
        self.rl_agent = EmbeddingActorCritic(
            state_dim=384,
            embedding_dim=384,
            state_encoder_type=encoder_type
        )
        print(f"State encoder type set to: {encoder_type}")

    def encode_conversation_state(self) -> np.ndarray:
        """
        Encode conversation state into embedding vector for RL.

        Supports two modes:
        1. 'sbert': Encode full preference text with SentenceBERT (original)
        2. 'lstm': Encode sequence of (item, rating) pairs with LSTM+attention

        STATE = Extracted preferences only (clean, structured)

        LLM extracts structured preferences from conversation:
        {"liked": ["Nolan", "Inception"], "disliked": ["horror"]}

        For SBERT mode:
        "likes: Nolan, Inception | dislikes: horror" → SentenceBERT → 384-dim

        For LSTM mode:
        [(Nolan, liked), (Inception, liked), (horror, disliked)] → LSTM+Attention → 384-dim

        Returns:
            384-dim state vector encoding extracted preferences
        """
        if not self.discovered_preferences:
            return np.zeros(384)

        if self.state_encoder_type == 'lstm':
            # LSTM mode: encode sequence of (item, rating) pairs
            return self._encode_state_lstm()
        else:
            # SBERT mode: encode full text
            return self._encode_state_sbert()

    def _encode_state_sbert(self) -> np.ndarray:
        """Encode state using full text SBERT encoding."""
        pref_text = self.preference_extractor.preferences_to_text(self.discovered_preferences)
        if not pref_text:
            return np.zeros(384)
        return self.encoder.encode(pref_text, convert_to_numpy=True)

    def _encode_state_lstm(self) -> np.ndarray:
        """Encode state using LSTM+attention over preference sequence."""
        if not self.preference_sequence:
            return np.zeros(384)

        # Get SBERT embeddings for each item
        items = [item for item, _ in self.preference_sequence]
        ratings = [rating for _, rating in self.preference_sequence]

        # Batch encode items
        item_embeddings = self.encoder.encode(items, convert_to_numpy=True)

        # Use RL agent's state encoder
        return self.rl_agent.encode_state_from_sequence(item_embeddings, ratings)

    def ask_question(self, explore: bool = True, verbose: bool = True, return_debug_info: bool = False):
        """
        Generate next question using RL + GPT.

        Args:
            explore: Whether to use exploration noise in RL
            verbose: Print debug information
            return_debug_info: If True, return (question, debug_info) tuple with RL prediction details

        Returns:
            Natural language question, or (question, debug_info) if return_debug_info=True
        """
        # Turn 1: Skip RL, use greeting template
        is_first_turn = len(self.conversation_history) == 0

        if is_first_turn:
            # No RL on turn 1 - just generate greeting
            question = self.question_generator.generate_question(
                [],  # No entities for greeting
                self.conversation_history,
                self.discovered_preferences
            )
            self.conversation_history.append(f"Agent: {question}")

            if verbose:
                print(f"\nTurn 1: Greeting (no RL)")

            if return_debug_info:
                return question, {'note': 'Turn 1 greeting - RL not used'}
            return question

        # Turn 2+: Use RL to predict concept

        # 1. Encode state
        state = self.encode_conversation_state()

        # 2. RL predicts SentenceBERT embedding
        predicted_embedding = self.rl_agent.predict_embedding(state, explore=explore)
        self.embedding_history.append(predicted_embedding)

        # 3. Find nearest movie entities, preferring broad concepts (genres, themes)
        # over specific movie titles
        nearest_entities = self.embedding_space.find_nearest_entities_prefer_broad(
            predicted_embedding,
            top_k=10,  # Get more candidates to filter from
            broad_boost=0.1  # Slight preference for genome tags over specific titles
        )

        # Filter out entities we've already asked about this episode
        if not hasattr(self, 'asked_entities'):
            self.asked_entities = set()

        filtered_entities = [
            (entity, score) for entity, score in nearest_entities
            if entity.lower() not in self.asked_entities
        ]

        # Use first non-repeated entity, or fall back to top if all repeated
        if filtered_entities:
            top_entity = filtered_entities[0][0]
            nearest_entities = filtered_entities[:3]  # Keep top 3 for logging
        else:
            top_entity = nearest_entities[0][0] if nearest_entities else None

        # Track this entity as asked
        if top_entity:
            self.asked_entities.add(top_entity.lower())

        entity_names = [top_entity] if top_entity else []

        if verbose:
            print(f"\nRL predicted embedding - Nearest entities (prefer broad):")
            for entity, score in nearest_entities[:3]:
                is_broad = self.embedding_space.is_broad_entity(entity)
                marker = "[BROAD]" if is_broad else "[SPECIFIC]"
                print(f"  {marker} {entity}: {score:.3f}")
            print(f"  Selected: {top_entity}")
            print(f"  Already asked: {len(self.asked_entities)} entities")

        # 4. Generate question from single entity (cleaner, single-focus question)
        question = self.question_generator.generate_question(
            entity_names,
            self.conversation_history,
            self.discovered_preferences
        )

        # 5. Update history
        self.conversation_history.append(f"Agent: {question}")

        # 6. Return debug info if requested (for logging enrichment)
        if return_debug_info:
            debug_info = {
                'selected_entity': top_entity,
                'nearest_entities': [
                    {
                        'entity': entity,
                        'similarity': float(score),
                        'is_broad': self.embedding_space.is_broad_entity(entity)
                    }
                    for entity, score in nearest_entities
                ]
            }
            return question, debug_info

        return question

    def process_user_response(self, response: str, user_target_movies: Optional[List[int]] = None) -> float:
        """
        Process user response and update conversation state.

        Pipeline:
        1. Add user response to conversation history
        2. Extract preferences from FULL conversation using LLM
        3. Update discovered_preferences (cumulative across all turns)
        4. Calculate reward if in training mode

        Optionally calculates per-turn reward based on NDCG@10 improvement.
        This implements the reward mechanism from the paper (Equation 2):
            r_t = NDCG@10(t) - NDCG@10(t-1)

        At each turn:
        1. Compute recommendations based on current dialogue state (not shown to user)
        2. Calculate NDCG@10 against user's target movies
        3. Reward = improvement in NDCG@10 from previous turn

        Args:
            response: User's response text
            user_target_movies: Ground truth target movies for reward calculation
                               If None, no reward is calculated (inference mode)

        Returns:
            Reward (NDCG delta) if user_target_movies provided, else 0.0
        """
        self.conversation_history.append(f"User: {response}")

        # Extract preferences from full conversation using LLM
        old_prefs = {
            'liked': set(self.discovered_preferences.get('liked', [])),
            'disliked': set(self.discovered_preferences.get('disliked', [])),
            'not_seen': set(self.discovered_preferences.get('not_seen', [])),
        }

        self.discovered_preferences = self.preference_extractor.extract_from_conversation(
            self.conversation_history,
            existing_preferences=self.discovered_preferences,
            debug=self.debug_preference_extraction
        )

        # Update preference sequence for LSTM encoding (track new items only)
        # Rating encoding: liked=[1,0,0], disliked=[0,1,0], unknown=[0,0,1]
        if self.state_encoder_type == 'lstm':
            new_liked = set(self.discovered_preferences.get('liked', [])) - old_prefs['liked']
            new_disliked = set(self.discovered_preferences.get('disliked', [])) - old_prefs['disliked']
            new_not_seen = set(self.discovered_preferences.get('not_seen', [])) - old_prefs['not_seen']

            for item in new_liked:
                self.preference_sequence.append((item, 'liked'))
            for item in new_disliked:
                self.preference_sequence.append((item, 'disliked'))
            for item in new_not_seen:
                self.preference_sequence.append((item, 'unknown'))
            # neutral is extracted but skipped - no actionable signal for recommendations

        # Calculate per-turn reward if target movies provided (training mode)
        reward = 0.0
        if user_target_movies is not None and self.recommender is not None:
            try:
                # Get recommendations based on current state (not shown to user)
                recommendations = self.recommend_movies(top_k=10)
                recommended_ids = [movie_id for movie_id, _, _ in recommendations]

                # Calculate NDCG@10
                current_ndcg = self._calculate_ndcg_at_k(recommended_ids, user_target_movies, k=10)
                self.ndcg_history.append(current_ndcg)

                # Calculate reward as delta from previous turn
                if len(self.ndcg_history) > 1:
                    reward = current_ndcg - self.ndcg_history[-2]
                else:
                    # First turn: reward is improvement from baseline (0.0)
                    reward = current_ndcg

                self.reward_history.append(reward)

            except Exception as e:
                print(f"Warning: Could not calculate per-turn reward: {e}")

        return reward

    def recommend_movies(self, top_k: int = 10) -> List[Tuple[int, str, float]]:
        """
        Get movie recommendations based on current conversation state.

        Args:
            top_k: Number of movies to recommend

        Returns:
            List of (movie_id, title, score) tuples

        Raises:
            RuntimeError: If recommender not initialized
        """
        if self.recommender is None:
            raise RuntimeError("Recommender not initialized. Provide movielens_data_path when creating agent.")

        # Get current conversation state
        state = self.encode_conversation_state()

        # Get recommendations
        recommendations = self.recommender.recommend(state, top_k=top_k)

        return recommendations

    def _calculate_ndcg_at_k(self, recommended_ids: List[int], target_ids: List[int], k: int = 10) -> float:
        """
        Calculate NDCG@k for recommendations.

        Args:
            recommended_ids: List of recommended movie IDs (ranked)
            target_ids: List of ground truth target movie IDs
            k: Cutoff for NDCG calculation

        Returns:
            NDCG@k score (0.0 to 1.0)
        """
        import numpy as np

        # Limit to top k recommendations
        recommended_ids = recommended_ids[:k]

        # Calculate DCG (Discounted Cumulative Gain)
        dcg = 0.0
        for i, movie_id in enumerate(recommended_ids):
            if movie_id in target_ids:
                # Relevance = 1 if in target set, 0 otherwise
                relevance = 1.0
                # Discount by log2(position + 1), position is 1-indexed
                dcg += relevance / np.log2(i + 2)

        # Calculate IDCG (Ideal DCG)
        # Best case: all target movies appear at top of ranking
        ideal_relevances = [1.0] * min(len(target_ids), k)
        idcg = sum(rel / np.log2(i + 2) for i, rel in enumerate(ideal_relevances))

        # Avoid division by zero
        if idcg == 0.0:
            return 0.0

        # NDCG = DCG / IDCG
        ndcg = dcg / idcg
        return ndcg

    def train_from_episode(self, use_per_turn_rewards: bool = True):
        """
        Train RL from completed conversation episode using per-turn rewards.

        Implementation based on paper (Section II.C, Equation 2):
            r_t = NDCG@10(t) - NDCG@10(t-1)

        At each turn during the episode:
        1. User responds to agent's question
        2. Recommendations computed (not shown to user)
        3. NDCG@10 calculated against ground truth
        4. Reward = improvement in NDCG@10 from previous turn

        This provides immediate feedback to the RL agent about which questions
        improve recommendation quality, allowing it to learn an optimal
        questioning strategy that maximizes NDCG@10 improvement.

        Args:
            use_per_turn_rewards: If True, uses reward_history (per-turn NDCG deltas).
                                 If False, uses final NDCG as reward for all turns.

        Returns:
            Training metrics dict or None if not enough data
        """
        if len(self.embedding_history) < 2:
            return None

        # Reconstruct states for each turn
        states = []
        for i in range(len(self.embedding_history)):
            temp_history = self.conversation_history[:i*2]
            if temp_history:
                state_text = " | ".join(temp_history)
                state = self.encoder.encode(state_text, convert_to_numpy=True)
            else:
                state = np.zeros(384)
            states.append(state)

        # Store experiences with per-turn rewards
        for i in range(len(states)):
            next_state = states[i+1] if i+1 < len(states) else states[i]
            done = (i == len(states) - 1)

            # Use per-turn reward (NDCG delta) or final NDCG for all turns
            if use_per_turn_rewards and i < len(self.reward_history):
                turn_reward = self.reward_history[i]
            elif len(self.ndcg_history) > 0:
                # Fallback: use final NDCG if per-turn rewards not available
                turn_reward = self.ndcg_history[-1]
            else:
                turn_reward = 0.0

            self.rl_agent.store_experience(
                states[i],
                self.embedding_history[i],
                turn_reward,
                next_state,
                done
            )

        # Train
        return self.rl_agent.train_step()

    def reset_conversation(self):
        """Reset conversation state for new episode."""
        self.conversation_history = []
        self.discovered_preferences = {}
        self.preference_sequence = []  # Reset preference sequence for LSTM mode
        self.embedding_history = []
        self.asked_entities = set()  # Track entities already asked about to avoid repetition
        self.ndcg_history = []
        self.reward_history = []

    def get_initial_question(self, verbose: bool = True) -> str:
        """
        Generate the first question to start a conversation.

        Returns:
            Initial question as string
        """
        return self.ask_question(explore=True, verbose=verbose)

    def get_conversation_state(self) -> dict:
        """
        Get current conversation state for logging/analysis.

        Returns:
            Dict with conversation history, discovered preferences, NDCG history, rewards
        """
        return {
            'conversation_history': self.conversation_history.copy(),
            'discovered_preferences': self.discovered_preferences.copy(),
            'ndcg_history': self.ndcg_history.copy(),
            'reward_history': self.reward_history.copy(),
            'num_turns': len(self.conversation_history) // 2  # Agent-user pairs
        }

    def get_episode_rewards(self) -> List[float]:
        """
        Get rewards for current episode.

        Returns:
            List of per-turn rewards (NDCG deltas)
        """
        return self.reward_history.copy()
