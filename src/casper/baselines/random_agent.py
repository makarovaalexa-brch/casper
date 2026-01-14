"""
Random baseline agent.

Selects random entities and generates questions. Floor performance.
"""

import numpy as np
from typing import List, Optional

from casper.models.embedding_space import SentenceBERTEmbeddingSpace
from casper.models.question_generator import QuestionGenerator
from casper.models.preference_extractor import PreferenceExtractor


class RandomAgent:
    """
    Random baseline: selects random entities for questions.

    Same architecture as CASPER but replaces RL actor with random selection.
    """

    def __init__(
        self,
        movielens_data_path: Optional[str] = None,
        embedding_space: Optional[SentenceBERTEmbeddingSpace] = None
    ):
        """
        Initialize random agent.

        Args:
            movielens_data_path: Path to MovieLens data
            embedding_space: Optional pre-initialized embedding space to reuse (for efficiency)
        """
        # Default MovieLens path
        if movielens_data_path is None:
            from pathlib import Path
            project_root = Path(__file__).parent.parent.parent.parent
            movielens_data_path = str(project_root / "data" / "movielens")

        print("Initializing Random Baseline Agent...")

        # Initialize or reuse components
        if embedding_space is not None:
            print("  Using shared embedding space")
            self.embedding_space = embedding_space
        else:
            print("  Creating new embedding space")
            self.embedding_space = SentenceBERTEmbeddingSpace(movielens_data_path)
        self.question_generator = QuestionGenerator()
        self.preference_extractor = PreferenceExtractor()
        self.encoder = self.embedding_space.encoder

        # Conversation state
        self.conversation_history = []
        self.discovered_preferences = {}

        print("Random Baseline Agent ready")

    def ask_question(self, verbose: bool = True) -> str:
        """
        Generate question by randomly selecting entities.

        Args:
            verbose: Print debug info

        Returns:
            Natural language question
        """
        # Randomly select 3 entities from embedding space
        all_entities = self.embedding_space.entity_list

        if len(all_entities) < 3:
            # Fallback if not enough entities
            selected_entities = all_entities
        else:
            selected_entities = list(np.random.choice(
                all_entities,
                size=min(3, len(all_entities)),
                replace=False
            ))

        if verbose:
            print(f"\nRandomly selected entities: {selected_entities}")

        # Generate question from random entities
        question = self.question_generator.generate_question(
            selected_entities,
            self.conversation_history,
            self.discovered_preferences
        )

        # Update history
        self.conversation_history.append(f"Agent: {question}")

        return question

    def process_user_response(self, response: str) -> float:
        """
        Process user response and update state.

        Args:
            response: User's response text

        Returns:
            Reward (always 0.0 for random agent - no learning)
        """
        self.conversation_history.append(f"User: {response}")

        # Extract preferences using LLM (same as CASPER)
        self.discovered_preferences = self.preference_extractor.extract_from_conversation(
            self.conversation_history,
            existing_preferences=self.discovered_preferences
        )

        return 0.0  # No learning, no reward

    def reset_conversation(self):
        """Reset conversation state for new episode."""
        self.conversation_history = []
        self.discovered_preferences = {}

    def get_initial_question(self, verbose: bool = True) -> str:
        """Generate the first question to start a conversation."""
        return self.ask_question(verbose=verbose)

    def get_conversation_state(self) -> dict:
        """Get current conversation state for logging/analysis."""
        return {
            'conversation_history': self.conversation_history.copy(),
            'discovered_preferences': self.discovered_preferences.copy(),
            'num_turns': len(self.conversation_history) // 2
        }
