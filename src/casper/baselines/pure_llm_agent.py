"""
Pure LLM baseline agent.

Uses GPT directly for question generation without RL strategic learning.
Shows value of RL actor vs pure LLM reasoning.
"""

from typing import Optional
from openai import OpenAI

from casper.models.preference_extractor import PreferenceExtractor
from casper.models.embedding_space import SentenceBERTEmbeddingSpace


class PureLLMAgent:
    """
    Pure LLM baseline: GPT generates questions directly from conversation.

    No RL actor, no entity selection - just LLM reasoning.
    This shows whether RL strategic learning adds value over pure LLM.
    """

    def __init__(
        self,
        movielens_data_path: Optional[str] = None,
        model_name: str = "gpt-4o-mini",
        embedding_space: Optional[SentenceBERTEmbeddingSpace] = None
    ):
        """
        Initialize pure LLM agent.

        Args:
            movielens_data_path: Path to MovieLens data
            model_name: OpenAI model to use
            embedding_space: Optional pre-initialized embedding space to reuse (for efficiency)
        """
        # Default MovieLens path
        if movielens_data_path is None:
            from pathlib import Path
            project_root = Path(__file__).parent.parent.parent.parent
            movielens_data_path = str(project_root / "data" / "movielens")

        print("Initializing Pure LLM Baseline Agent...")

        # Initialize or reuse components
        if embedding_space is not None:
            print("  Using shared embedding space")
            self.embedding_space = embedding_space
        else:
            print("  Creating new embedding space")
            self.embedding_space = SentenceBERTEmbeddingSpace(movielens_data_path)
        self.preference_extractor = PreferenceExtractor()
        self.encoder = self.embedding_space.encoder

        # LLM client
        self.client = OpenAI()
        self.model_name = model_name

        # Conversation state
        self.conversation_history = []
        self.discovered_preferences = {}

        # System prompt for question generation
        self.system_prompt = """You are a conversational movie recommendation assistant.
Your goal is to learn the user's movie preferences through asking questions.

Guidelines:
- Ask ONE clear question per turn
- Build on previous conversation naturally
- Focus on discovering their tastes (genres, directors, themes, moods)
- Be conversational and friendly
- Do NOT make recommendations yet - only ask questions to learn preferences

Output ONLY the question, nothing else."""

        print("Pure LLM Baseline Agent ready")

    def ask_question(self, verbose: bool = True) -> str:
        """
        Generate question using pure LLM (no RL).

        Args:
            verbose: Print debug info

        Returns:
            Natural language question
        """
        # Build conversation context
        if not self.conversation_history:
            # First question
            user_prompt = "Start a conversation to learn about the user's movie preferences. Ask your first question."
        else:
            # Build context from conversation history
            context = "\n".join(self.conversation_history)
            user_prompt = f"""Previous conversation:
{context}

Based on this conversation, ask the next question to learn more about the user's preferences."""

        # Generate question with LLM
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.7,
            max_tokens=150
        )

        question = response.choices[0].message.content.strip()

        if verbose:
            print(f"\nPure LLM generated question: {question}")

        # Update history
        self.conversation_history.append(f"Agent: {question}")

        return question

    def process_user_response(self, response: str) -> float:
        """
        Process user response and update state.

        Args:
            response: User's response text

        Returns:
            Reward (always 0.0 - no learning in pure LLM)
        """
        self.conversation_history.append(f"User: {response}")

        # Extract preferences using LLM (same as CASPER)
        self.discovered_preferences = self.preference_extractor.extract_from_conversation(
            self.conversation_history,
            existing_preferences=self.discovered_preferences
        )

        return 0.0  # No RL training

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
