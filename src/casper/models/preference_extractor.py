"""
LLM-based preference extractor.

Extracts structured user preferences from conversational text.
Uses GPT to parse natural language responses into clean preference facts.
"""

import openai
from typing import Dict, List
import json
import yaml
from pathlib import Path
from jinja2 import Template


class PreferenceExtractor:
    """
    Extracts structured preferences from conversation using LLM.

    Converts conversation into clean structured data:
    {"liked": ["Nolan", "action"], "neutral": ["sci-fi"], "disliked": ["romance"]}
    """

    def __init__(self, config_path: str = None):
        """
        Initialize preference extractor.

        Args:
            config_path: Path to config.yaml (defaults to config/config.yaml)
        """
        # Load config
        if config_path is None:
            project_root = Path(__file__).parent.parent.parent.parent
            config_path = project_root / "config" / "config.yaml"

        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        extractor_config = config['models']['preference_extractor']
        self.model_name = extractor_config['model_name']
        self.temperature = extractor_config['temperature']
        self.max_tokens = extractor_config['max_tokens']
        self.categories = extractor_config['categories']
        self.use_json_mode = extractor_config['use_json_mode']

        # Load prompts (all Jinja templates)
        prompts_dir = Path(__file__).parent.parent / "prompts"
        with open(prompts_dir / "preference_extraction_system.jinja", 'r') as f:
            self.system_prompt = f.read().strip()
        with open(prompts_dir / "preference_extraction_user.jinja", 'r') as f:
            self.user_prompt_template = Template(f.read())

    def extract_from_conversation(
        self,
        conversation: List[str],
        existing_preferences: Dict = None,
        debug: bool = False
    ) -> Dict[str, List[str]]:
        """
        Extract preferences from conversation history.

        Args:
            conversation: List of conversation turns (all turns)
            existing_preferences: Previously extracted preferences to update
            debug: If True, print detailed debugging information

        Returns:
            Dict with keys: liked, neutral, disliked
        """
        # Build prompt
        prompt = self._build_extraction_prompt(conversation, existing_preferences)

        if debug:
            print(f"\n{'='*60}")
            print("PREFERENCE EXTRACTOR DEBUG")
            print(f"{'='*60}")
            print(f"Conversation turns: {len(conversation)}")
            print(f"Existing preferences: {existing_preferences}")
            print(f"\nFull conversation:")
            for i, turn in enumerate(conversation):
                print(f"  {i}: {turn}")
            print(f"\nPrompt to LLM:")
            print(prompt[:500] + "..." if len(prompt) > 500 else prompt)
            print(f"{'='*60}\n")

        # Prepare API call parameters
        api_params = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt}
            ]
        }

        if self.max_tokens is not None:
            api_params["max_tokens"] = self.max_tokens

        # Only add temperature for models that support it (not gpt-5-nano)
        if not self.model_name.startswith("gpt-5"):
            api_params["temperature"] = self.temperature

        # Use JSON mode if enabled
        if self.use_json_mode:
            api_params["response_format"] = {"type": "json_object"}

        # Retry logic for API failures
        max_retries = 3
        content = None

        for attempt in range(max_retries):
            try:
                response = openai.chat.completions.create(**api_params)
                content = response.choices[0].message.content.strip()

                if debug:
                    print(f"\nAPI Response (attempt {attempt+1}):")
                    print(f"  Content length: {len(content) if content else 0}")
                    print(f"  Content preview: {content[:200] if content else '(empty)'}")

                if content:
                    break  # Success
                else:
                    if attempt < max_retries - 1:
                        import time
                        time.sleep(1)  # Brief pause before retry

            except Exception as e:
                if attempt < max_retries - 1:
                    print(f"API error (attempt {attempt+1}/{max_retries}): {e}")
                    import time
                    time.sleep(2)
                else:
                    print(f"API failed after {max_retries} attempts: {e}")
                    if debug:
                        print(f"  Exception type: {type(e)}")
                        print(f"  Full error: {str(e)}")
                    return {category: [] for category in self.categories}

        # Handle empty response after all retries
        if not content:
            # Return "unknown" instead of empty list for better encoding
            return {category: [] for category in self.categories}

        try:
            preferences = json.loads(content)
        except json.JSONDecodeError as e:
            # Try to extract JSON from markdown code blocks (common LLM mistake)
            if "```json" in content:
                try:
                    json_str = content.split("```json")[1].split("```")[0].strip()
                    preferences = json.loads(json_str)
                except:
                    # Return "unknown" instead of empty
                    return {category: [] for category in self.categories}
            elif "```" in content:
                try:
                    json_str = content.split("```")[1].split("```")[0].strip()
                    preferences = json.loads(json_str)
                except:
                    # Return "unknown" instead of empty
                    return {category: [] for category in self.categories}
            else:
                # Return "unknown" if can't parse
                return {category: [] for category in self.categories}

        # Ensure all categories exist - use "unknown" if missing
        for category in self.categories:
            if category not in preferences or not preferences[category]:
                preferences[category] = []

        return preferences

    def _build_extraction_prompt(
        self,
        conversation: List[str],
        existing_preferences: Dict
    ) -> str:
        """Build prompt for GPT extraction using Jinja template."""

        # Format conversation turns
        conversation_text = "\n".join(conversation)

        # Render Jinja template
        prompt = self.user_prompt_template.render(
            conversation_turns=conversation_text,
            existing_preferences=existing_preferences
        )

        return prompt

    def preferences_to_text(self, preferences: Dict[str, List[str]]) -> str:
        """
        Convert structured preferences to text for state encoding.

        Output format optimized for SentenceBERT encoding.
        """
        parts = []

        if preferences.get("liked"):
            parts.append(f"likes: {', '.join(preferences['liked'])}")
        if preferences.get("disliked"):
            parts.append(f"dislikes: {', '.join(preferences['disliked'])}")
        if preferences.get("neutral"):
            parts.append(f"mentioned: {', '.join(preferences['neutral'])}")

        return " | ".join(parts) if parts else ""
