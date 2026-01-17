"""
MovieLens-based user simulator for conversational recommendation systems.

Uses real MovieLens user rating data + LLM to generate realistic conversational responses.
"""

import pandas as pd
import random
import yaml
import pickle
import hashlib
from typing import Dict, List, Optional
from pathlib import Path

from openai import OpenAI
import json


class UserSimulator:
    """
    Realistic user simulator using MovieLens rating data + LLM for responses.

    Each user profile is based on real MovieLens user preferences, with LLM
    generating natural conversational responses that reflect those preferences.

    Profile format: Simple dict of {movie_title: rating}
    - LLM is smart enough to understand movies without explicit genre extraction
    - Actual ratings (not compressed to liked/disliked) for richer context
    """

    def __init__(
        self,
        movielens_data_path: str,
        model_name: Optional[str] = None,
        temperature: float = 0.7,
        min_ratings: Optional[int] = None,
        max_profiles: Optional[int] = None,
        config_path: Optional[str] = None
    ):
        """
        Initialize MovieLens-based user simulator.

        Args:
            movielens_data_path: Path to MovieLens dataset directory
            model_name: OpenAI model to use (defaults to config value)
            temperature: LLM temperature (default: 0.7)
            min_ratings: Minimum ratings per user (defaults to config value)
            max_profiles: Maximum number of profiles (defaults to config value)
            config_path: Path to config.yaml (auto-detected if None)
        """
        # Load config
        if config_path is None:
            project_root = Path(__file__).parent.parent.parent.parent
            config_path = str(project_root / "config" / "config.yaml")

        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        sim_config = config['models']['user_simulator']

        # Use config defaults if params not specified
        self.model_name = model_name or sim_config['model_name']
        self.min_ratings = min_ratings or sim_config['min_ratings']
        self.max_profiles = max_profiles or sim_config['max_profiles']
        self.max_tokens = sim_config.get('max_tokens', 16000)  # Very high for gpt-5 reasoning

        # Initialize OpenAI client for function calling
        self.openai_client = OpenAI()
        self.temperature = temperature

        # Store data path for caching
        self.data_path = Path(movielens_data_path)

        # Load and process MovieLens data
        self.ratings_df = self._load_ratings(movielens_data_path)
        self.movies_df = self._load_movies(movielens_data_path)

        # Create user profiles (with caching)
        self.user_profiles = self._create_user_profiles()
        print(f"Loaded {len(self.user_profiles)} user profiles")

    def _load_ratings(self, data_path: str) -> pd.DataFrame:
        """
        Load MovieLens ratings data (CSV format only).

        MovieLens 25M uses ratings.csv with columns: userId, movieId, rating, timestamp
        """
        ratings_path = Path(data_path) / "ratings.csv"

        if not ratings_path.exists():
            raise FileNotFoundError(
                f"MovieLens ratings.csv not found at {data_path}. "
                f"Please download MovieLens 25M dataset from https://grouplens.org/datasets/movielens/25m/"
            )

        ratings_df = pd.read_csv(ratings_path)

        print(f"Loaded {len(ratings_df)} ratings from {len(ratings_df['userId'].unique())} users")
        return ratings_df

    def _load_movies(self, data_path: str) -> pd.DataFrame:
        """
        Load MovieLens movies metadata (CSV format only).

        MovieLens 25M uses movies.csv with columns: movieId, title, genres
        """
        movies_path = Path(data_path) / "movies.csv"

        if not movies_path.exists():
            raise FileNotFoundError(
                f"MovieLens movies.csv not found at {data_path}. "
                f"Please download MovieLens 25M dataset from https://grouplens.org/datasets/movielens/25m/"
            )

        movies_df = pd.read_csv(movies_path)
        return movies_df

    def _create_user_profiles(self) -> List[Dict]:
        """
        Extract user profiles from MovieLens data.

        Profile format: {movie_title: rating} - simple and LLM-friendly.
        No need to extract genres, liked/disliked lists, etc. - LLM knows movies.

        Uses disk caching to avoid reprocessing on subsequent runs.
        """
        # Check cache first
        cache_file = self._get_profiles_cache_path()

        if cache_file.exists():
            print(f"Loading user profiles from cache: {cache_file}")
            with open(cache_file, 'rb') as f:
                profiles = pickle.load(f)
            print(f"Loaded {len(profiles)} profiles from cache")
            return profiles

        # Cache miss - build profiles from scratch
        print(f"Building {self.max_profiles} user profiles from MovieLens data...")
        print(f"(This will be cached for future runs)")

        profiles = []

        # Get users with sufficient ratings
        user_counts = self.ratings_df['userId'].value_counts()
        eligible_users = user_counts[user_counts >= self.min_ratings].index

        # Limit number of profiles (deterministic order)
        eligible_users = sorted(eligible_users)[:self.max_profiles]

        for user_id in eligible_users:
            user_ratings = self.ratings_df[self.ratings_df['userId'] == user_id]

            # Merge with movie titles
            user_movies = user_ratings.merge(
                self.movies_df[['movieId', 'title']],
                on='movieId',
                how='left'
            )

            # Build simple profile: {movie: rating}
            ratings_dict = {}
            for _, row in user_movies.iterrows():
                if pd.notna(row['title']):
                    # Clean title (remove year for readability)
                    clean_title = row['title'].split('(')[0].strip()
                    ratings_dict[clean_title] = float(row['rating'])

            profile = {
                'user_id': int(user_id),
                'ratings': ratings_dict
            }

            profiles.append(profile)

        # Save to cache
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_file, 'wb') as f:
            pickle.dump(profiles, f)
        print(f"Created {len(profiles)} profiles and cached to: {cache_file}")

        return profiles

    def _get_profiles_cache_path(self) -> Path:
        """Get cache file path for user profiles."""
        # Cache based on min_ratings and max_profiles
        cache_dir = self.data_path / '.cache'
        cache_file = cache_dir / f'user_profiles_min{self.min_ratings}_max{self.max_profiles}.pkl'
        return cache_file

    def sample_user(self) -> Dict:
        """Sample a random user profile for simulation."""
        return random.choice(self.user_profiles)

    def _query_rating(self, movie_title: str, user_profile: Dict) -> Optional[float]:
        """Tool: Get rating for a specific movie."""
        return user_profile['ratings'].get(movie_title)

    def _search_ratings(
        self,
        query: str,
        user_profile: Dict,
        min_rating: Optional[float] = None,
        max_results: int = 10
    ) -> List[Dict[str, any]]:
        """Tool: Search movies by text query."""
        query_lower = query.lower()
        results = []

        for movie, rating in user_profile['ratings'].items():
            if query_lower in movie.lower():
                if min_rating is None or rating >= min_rating:
                    results.append({"movie": movie, "rating": rating})

        # Sort by rating descending
        results.sort(key=lambda x: x['rating'], reverse=True)
        return results[:max_results]

    def _get_all_ratings(
        self,
        user_profile: Dict,
        min_rating: Optional[float] = None,
        max_rating: Optional[float] = None,
        limit: int = 20
    ) -> List[Dict[str, any]]:
        """Tool: Get all ratings within a range."""
        results = []

        for movie, rating in user_profile['ratings'].items():
            if (min_rating is None or rating >= min_rating) and \
               (max_rating is None or rating <= max_rating):
                results.append({"movie": movie, "rating": rating})

        # Sort by rating descending
        results.sort(key=lambda x: x['rating'], reverse=True)
        return results[:limit]

    def simulate_response(
        self,
        question: str,
        user_profile: Dict,
        conversation_history: Optional[List[str]] = None,
        max_retries: int = 3,
        use_function_calling: bool = True,
        return_tool_calls: bool = False
    ):
        """
        Generate realistic user response with function calling to query profile.

        The LLM can now call tools to query the full rating profile:
        - query_rating(movie_title) -> Get rating for specific movie
        - search_ratings(query) -> Search movies by text
        - get_all_ratings(min_rating, max_rating) -> Get ratings in range

        Args:
            question: The current preference elicitation question
            user_profile: User profile dict with 'ratings' key
            conversation_history: Full conversation history (optional)
            max_retries: Maximum LLM retry attempts
            use_function_calling: Enable function calling (default: True)
            return_tool_calls: If True, return (response, tool_call_logs) tuple

        Returns:
            Natural language response, or (response, tool_call_logs) if return_tool_calls=True
        """
        if not use_function_calling:
            # Fallback to old method without function calling
            return self._simulate_response_legacy(
                question, user_profile, conversation_history, max_retries
            )

        # Build system prompt - natural and conversational like Reddit users
        system_prompt = f"""You're a movie fan chatting about films. You've seen and rated {len(user_profile['ratings'])} movies.

CRITICAL RULES:
1. Keep responses SHORT (2-4 sentences max, like a real chat)
2. ONLY talk about movies you've actually seen (use tools to check!)
3. If asked about a movie you haven't seen, just say "Haven't seen that one" - DON'T speculate or give long explanations
4. Be honest and conversational, not an essay writer

GOOD EXAMPLES:
- "Inception was amazing! Gave it a 5/5. Love that kind of mind-bending stuff."
- "Haven't seen that one."
- "Dark Knight was great, but I thought Interstellar was a bit long."
- "Not really into rom-coms tbh."

BAD EXAMPLES (DO NOT DO THIS):
- Multi-paragraph essays analyzing hypothetical movies [X]
- Detailed opinions about movies you haven't seen [X]
- "I haven't seen it BUT let me give you a 10-paragraph analysis anyway..." [X]

Keep it short, honest, and natural like texting a friend about movies.
"""

        # Build conversation context
        messages = [{"role": "system", "content": system_prompt}]

        if conversation_history:
            conv_text = "\n".join(conversation_history)
            messages.append({"role": "user", "content": f"Previous conversation:\n{conv_text}"})

        # Add current question
        messages.append({"role": "user", "content": question})

        # Define tools for OpenAI function calling
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "query_rating",
                    "description": "Get your rating for a specific movie. Returns the rating (0.5-5.0) or null if you haven't seen it.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "movie_title": {
                                "type": "string",
                                "description": "The movie title to query"
                            }
                        },
                        "required": ["movie_title"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "search_ratings",
                    "description": "Search your movie ratings by text query (e.g., 'Nolan', 'sci-fi'). Returns list of matching movies with ratings.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Text to search for in movie titles"
                            },
                            "min_rating": {
                                "type": "number",
                                "description": "Minimum rating filter (optional)"
                            }
                        },
                        "required": ["query"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "get_all_ratings",
                    "description": "Get your ratings within a range. Useful for finding movies you loved or disliked.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "min_rating": {
                                "type": "number",
                                "description": "Minimum rating (0.5-5.0)"
                            },
                            "max_rating": {
                                "type": "number",
                                "description": "Maximum rating (0.5-5.0)"
                            }
                        }
                    }
                }
            }
        ]

        # Track tool calls for logging (if requested)
        tool_call_logs = []

        # OpenAI function calling loop
        for attempt in range(max_retries):
            try:
                # Build API parameters
                api_params = {
                    "model": self.model_name,
                    "messages": messages,
                    "tools": tools,
                    "tool_choice": "auto"
                }

                # Add optional parameters
                if self.max_tokens is not None:
                    api_params["max_tokens"] = self.max_tokens
                if not self.model_name.startswith("gpt-5"):
                    api_params["temperature"] = self.temperature

                response = self.openai_client.chat.completions.create(**api_params)

                response_message = response.choices[0].message

                # Check if model wants to call functions
                if response_message.tool_calls:
                    # Add assistant's message to conversation
                    messages.append(response_message)

                    # Process each tool call
                    for tool_call in response_message.tool_calls:
                        function_name = tool_call.function.name
                        function_args = json.loads(tool_call.function.arguments)

                        # Call the appropriate function
                        if function_name == "query_rating":
                            result = self._query_rating(function_args["movie_title"], user_profile)
                        elif function_name == "search_ratings":
                            result = self._search_ratings(
                                function_args["query"],
                                user_profile,
                                function_args.get("min_rating")
                            )
                        elif function_name == "get_all_ratings":
                            result = self._get_all_ratings(
                                user_profile,
                                function_args.get("min_rating"),
                                function_args.get("max_rating")
                            )
                        else:
                            result = {"error": f"Unknown function: {function_name}"}

                        # Log tool call if requested
                        if return_tool_calls:
                            tool_call_logs.append({
                                'function': function_name,
                                'arguments': function_args,
                                'result': result
                            })

                        # Add function result to messages
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": json.dumps(result)
                        })

                    # Get final response after tool calls
                    final_api_params = {
                        "model": self.model_name,
                        "messages": messages
                    }

                    # Add optional parameters
                    if self.max_tokens is not None:
                        final_api_params["max_tokens"] = self.max_tokens
                    if not self.model_name.startswith("gpt-5"):
                        final_api_params["temperature"] = self.temperature

                    final_response = self.openai_client.chat.completions.create(**final_api_params)
                    final_text = final_response.choices[0].message.content.strip()

                    if return_tool_calls:
                        return final_text, tool_call_logs
                    return final_text

                else:
                    # No tool calls, return direct response
                    final_text = response_message.content.strip()
                    if return_tool_calls:
                        return final_text, []  # Empty tool call list
                    return final_text

            except Exception as e:
                error_type = type(e).__name__
                error_msg = str(e)
                print(f"WARNING: Function calling failed (attempt {attempt + 1}/{max_retries}): {error_type}: {error_msg}")

                if attempt < max_retries - 1:
                    print(f"Retrying...")
                else:
                    raise RuntimeError(
                        f"User simulator failed after {max_retries} attempts. Last error: {error_type}: {error_msg}"
                    )

    def _simulate_response_legacy(
        self,
        question: str,
        user_profile: Dict,
        conversation_history: Optional[List[str]] = None,
        max_retries: int = 3
    ) -> str:
        """
        Legacy method without function calling (fallback).
        Shows top/bottom rated movies in prompt.
        """
        # Format top 10 + bottom 5 for context
        sorted_ratings = sorted(
            user_profile['ratings'].items(),
            key=lambda x: x[1],
            reverse=True
        )

        top_movies = sorted_ratings[:10]
        bottom_movies = sorted_ratings[-5:]

        rating_text = "TOP RATED:\n" + "\n".join(
            [f"- {movie}: {rating}/5.0" for movie, rating in top_movies]
        )
        rating_text += "\n\nLOWEST RATED:\n" + "\n".join(
            [f"- {movie}: {rating}/5.0" for movie, rating in bottom_movies]
        )
        rating_text += f"\n\n(You have rated {len(user_profile['ratings'])} movies total)"

        system_prompt = self.prompts['system_prompt'].format(
            movie_ratings=rating_text
        )

        messages = [SystemMessage(content=system_prompt)]

        if conversation_history:
            conv_text = "\n".join(conversation_history)
            context_prompt = self.prompts['conversation_context'].format(
                conversation_history=conv_text
            )
            messages.append(SystemMessage(content=context_prompt))

        messages.append(HumanMessage(content=question))

        for attempt in range(max_retries):
            try:
                response = self.llm.invoke(messages)
                return response.content.strip()
            except Exception as e:
                if attempt < max_retries - 1:
                    print(f"Retrying... (attempt {attempt + 2}/{max_retries})")
                else:
                    raise RuntimeError(f"User simulator failed: {e}")
