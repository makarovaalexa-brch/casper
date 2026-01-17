"""
GPT-based question generator.

Converts RL-selected entities/concepts into natural language questions.
"""

from typing import List, Dict
import openai
import yaml
from pathlib import Path
from jinja2 import Template


class QuestionGenerator:
    """
    Converts selected entities/concepts into natural language questions using GPT.
    """

    def __init__(self, config_path: str = None):
        """
        Initialize question generator.

        Args:
            config_path: Path to config.yaml (defaults to config/config.yaml)
        """
        # Load config
        if config_path is None:
            project_root = Path(__file__).parent.parent.parent.parent
            config_path = project_root / "config" / "config.yaml"

        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        generator_config = config['models']['question_generator']
        self.model_name = generator_config['model_name']
        self.max_tokens = generator_config['max_tokens']
        self.temperature = generator_config['temperature']

        # Load prompts (all Jinja templates)
        prompts_dir = Path(__file__).parent.parent / "prompts"
        with open(prompts_dir / "question_generation_system.jinja", 'r') as f:
            self.system_prompt = f.read().strip()
        with open(prompts_dir / "question_generation_user.jinja", 'r') as f:
            self.user_prompt_template = Template(f.read())

    def generate_question(
        self,
        entities: List[str],
        conversation_history: List[str],
        discovered_preferences: Dict
    ) -> str:
        """
        Generate question from RL-selected entities.

        Args:
            entities: Movie entities selected by RL (e.g., ["Inception", "Nolan", "sci-fi"])
            conversation_history: Full conversation history (all turns)
            discovered_preferences: Extracted preferences dict with categories

        Returns:
            Natural language question

        Raises:
            Exception: If GPT API call fails
        """
        # Build prompt using Jinja template
        conversation_context = "\n".join(conversation_history) if conversation_history else ""

        prompt = self.user_prompt_template.render(
            entities=entities,
            conversation_context=conversation_context,
            discovered_preferences=discovered_preferences
        )

        # Build API parameters
        api_params = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt}
            ]
        }

        # Add optional parameters
        if self.max_tokens is not None:
            api_params["max_tokens"] = self.max_tokens
        if not self.model_name.startswith("gpt-5"):
            api_params["temperature"] = self.temperature

        try:
            response = openai.chat.completions.create(**api_params)
            question = response.choices[0].message.content.strip()

            if not question:
                print(f"[QuestionGen] WARNING: Empty question generated!")
                print(f"[QuestionGen]   Entities: {entities}")
                print(f"[QuestionGen]   Model: {self.model_name}")
                print(f"[QuestionGen]   Returning fallback question")
                # Fallback question
                return f"What did you think of {entities[0]}?" if entities else "What kind of movies do you enjoy?"

            return question

        except Exception as e:
            print(f"[QuestionGen] ERROR: {type(e).__name__}: {e}")
            print(f"[QuestionGen]   Entities: {entities}")
            print(f"[QuestionGen]   Model: {self.model_name}")
            print(f"[QuestionGen]   Returning fallback question")
            # Fallback question
            return f"What did you think of {entities[0]}?" if entities else "What kind of movies do you enjoy?"
