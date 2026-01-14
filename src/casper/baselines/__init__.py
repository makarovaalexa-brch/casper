"""
Baseline agents for comparison in CASPER experiments.

Baselines for sanity checking (minimal implementations):
- RandomAgent: Random question selection (floor performance)
- PureLLMAgent: GPT without RL (shows value of strategic learning)

For publication, compare against actual CRS papers.
"""

from .random_agent import RandomAgent
from .pure_llm_agent import PureLLMAgent

__all__ = ['RandomAgent', 'PureLLMAgent']
