"""
Conversation orchestration for CASPER training and evaluation.

Handles interaction between agent and user simulator.
Separate from metric computation (see evaluation/evaluator.py).

DESIGN: Background Recommendations Only
- Agent ONLY asks questions (recommendations never shown to user)
- Recommendations computed in background after each turn for reward
- No conversation ending logic (can run indefinitely)
- Focus: Preference elicitation, not recommendation delivery
"""

from typing import Dict, List, Optional
from ..agents.user_simulator import MovieLensLLMSimulator


class ConversationRunner:
    """
    Orchestrates conversations between CASPER agent and user simulator.

    Handles conversation flow, turn management, and logging.
    Separate from evaluation metrics (see evaluation/evaluator.py).
    """

    def __init__(
        self,
        user_simulator: MovieLensLLMSimulator,
        max_turns: int = 10,
        verbose: bool = False
    ):
        """
        Initialize conversation runner.

        Args:
            user_simulator: User simulator instance
            max_turns: Maximum conversation length
            verbose: Print conversation to console
        """
        self.user_simulator = user_simulator
        self.max_turns = max_turns
        self.verbose = verbose

    def run_conversation(
        self,
        agent,
        user_profile: Dict
    ) -> List[Dict]:
        """
        Run a single conversation between agent and user simulator.

        Agent ONLY asks questions - recommendations computed in background only.
        Conversation runs for max_turns (no early stopping).

        Args:
            agent: CASPER agent instance
            user_profile: User profile for simulator

        Returns:
            List of conversation turns: [{'role': 'agent'/'user', 'content': str, 'turn': int}, ...]
        """
        conversation = []
        conversation_history = []  # For simulator context

        # Agent asks first question
        agent_response = agent.get_initial_question()

        conversation.append({
            'role': 'agent',
            'content': agent_response,
            'turn': 0
        })
        conversation_history.append(f"Agent: {agent_response}")

        if self.verbose:
            print(f"Turn 0 [Agent]: {agent_response}")

        for turn_num in range(1, self.max_turns):
            # User responds using simulator
            user_response = self.user_simulator.simulate_response(
                question=agent_response,
                user_profile=user_profile,
                conversation_history=conversation_history
            )

            conversation.append({
                'role': 'user',
                'content': user_response,
                'turn': turn_num
            })
            conversation_history.append(f"User: {user_response}")

            if self.verbose:
                print(f"Turn {turn_num} [User]: {user_response}")

            # Agent processes user response and asks next question
            # Note: process_user_response() computes recommendations in BACKGROUND for reward
            # but does NOT return them - agent always asks another question
            agent.process_user_response(user_response)
            agent_response = agent.ask_question()

            conversation.append({
                'role': 'agent',
                'content': agent_response,
                'turn': turn_num + 1
            })
            conversation_history.append(f"Agent: {agent_response}")

            if self.verbose:
                print(f"Turn {turn_num + 1} [Agent]: {agent_response}")

        return conversation

    def run_episode(
        self,
        agent,
        user_profile: Dict,
        training_profile: Dict,
        test_set: Dict,
        target_movie_ids: Optional[List[int]] = None
    ) -> Dict:
        """
        Run a complete training/evaluation episode.

        NOTE: Agent is BLIND to profiles - only knows what user mentions in conversation.
        Simulator uses training_profile to generate realistic responses.

        Args:
            agent: CASPER agent instance
            user_profile: Full user profile (for simulator) - UNUSED, kept for API compatibility
            training_profile: Profile with held-out movies removed (for simulator)
            test_set: Held-out movies for evaluation
            target_movie_ids: Optional list of target movie IDs for reward calculation

        Returns:
            Episode data: {
                'conversation': conversation log,
                'training_profile': training profile used,
                'test_set': test set used,
                'rewards': list of per-turn rewards (if training),
                'final_state': agent's final state
            }
        """
        # Reset agent for new conversation
        agent.reset_conversation()

        # Run conversation
        # Agent is blind - only simulator sees training_profile
        conversation = self.run_conversation(
            agent=agent,
            user_profile=training_profile  # For simulator only
        )

        # Collect rewards if training (agent calculates internally)
        rewards = []
        if hasattr(agent, 'get_episode_rewards'):
            rewards = agent.get_episode_rewards()

        episode_data = {
            'conversation': conversation,
            'training_profile': training_profile,
            'test_set': test_set,
            'rewards': rewards,
            'final_state': agent.get_conversation_state() if hasattr(agent, 'get_conversation_state') else None
        }

        return episode_data
