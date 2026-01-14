"""
RL Actor Pretraining (Supervised Learning)

Train RL actor to predict SentenceBERT embeddings from conversation context.

Goal: Learn mapping from conversation context → relevant concept embeddings
BEFORE reward-based RL training.

DATA ORGANIZATION:
- Context: User post text ("I love Inception's mind-bending plot")
- Concepts: Movie entities extracted by LLM (["Inception", "Nolan", "sci-fi", "mind-bending"])
- Target embeddings: SentenceBERT embedding for EACH concept (384-dim per concept)
- Training pairs: MULTIPLE pairs per post (one per concept, NO averaging)
  Example: 4 concepts → 4 training pairs with same context, different target embeddings

Training data: Reddit conversation pairs
Loss: MSE between predicted and target concept embeddings
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict
from pathlib import Path
import json
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import yaml

import sys
sys.path.append(str(Path(__file__).parent.parent))

from casper.models.embedding_space import SentenceBERTEmbeddingSpace
from casper.models.rl_actor_critic import EmbeddingActorCritic
from casper.models.preference_extractor import PreferenceExtractor
from casper.data.reddit_loader import RedditDataLoader


class PretrainingDataset:
    """
    Dataset for RL actor pretraining.

    Converts Reddit posts to (context, concept_embedding) training pairs.
    Creates MULTIPLE pairs per post (one per concept).
    """

    def __init__(
        self,
        reddit_data_path: str,
        embedding_space: SentenceBERTEmbeddingSpace,
        preference_extractor: PreferenceExtractor,
        encoder: SentenceTransformer
    ):
        self.embedding_space = embedding_space
        self.preference_extractor = preference_extractor
        self.encoder = encoder

        # Load Reddit data
        loader = RedditDataLoader(reddit_data_path)
        self.data = loader.load()

        # Process into training examples
        self.examples = self._create_training_examples()

        print(f"Created {len(self.examples)} training examples from {len(self.data)} posts")

    def _create_training_examples(self) -> List[Dict]:
        """
        Convert Reddit conversation pairs to training examples.

        Data format:
        - Input: User post ("I love Inception")
        - Target: Concepts from RESPONSE ("Have you seen Interstellar?")
          → ["Interstellar", "Nolan", "sci-fi"]

        Creates MULTIPLE training pairs (one per concept, no averaging):
        - ("I love Inception", embedding("Interstellar"))
        - ("I love Inception", embedding("Nolan"))
        - ("I love Inception", embedding("sci-fi"))
        """
        examples = []

        for post in tqdm(self.data, desc="Processing Reddit conversations"):
            # Get user post as context
            if "user_post" not in post or "response" not in post:
                continue

            context = post["user_post"]

            # Get concepts from response (what agent should ask about next)
            if "response_concepts" in post:
                concepts = post["response_concepts"]
            else:
                # Extract concepts using LLM
                conversation = [f"User: {post['user_post']}", f"Agent: {post['response']}"]
                preferences = self.preference_extractor.extract_from_conversation(conversation)
                concepts = preferences.get("liked", []) + preferences.get("neutral", [])

            if not context.strip() or not concepts:
                continue

            # Create MULTIPLE training pairs (one per concept)
            for concept in concepts:
                emb = self.embedding_space.get_embedding(concept)
                if emb is not None:
                    examples.append({
                        "context": context[:500],
                        "concept": concept,
                        "target_embedding": emb
                    })

        return examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


class ActorPretrainer:
    """
    Trains RL actor with supervised learning on context-concept pairs.

    After training, actor will predict embeddings in correct semantic space.
    """

    def __init__(
        self,
        rl_agent: EmbeddingActorCritic,
        embedding_space: SentenceBERTEmbeddingSpace,
        reddit_data_path: str,
        encoder: SentenceTransformer,
        config: Dict
    ):
        self.rl_agent = rl_agent
        self.embedding_space = embedding_space
        self.encoder = encoder
        self.config = config

        # Create dataset
        preference_extractor = PreferenceExtractor()
        self.dataset = PretrainingDataset(
            reddit_data_path,
            embedding_space,
            preference_extractor,
            encoder
        )

        # Training metrics
        self.train_losses = []

    def train(self):
        """
        Train RL actor with supervised learning.

        Hyperparameters loaded from config.
        """
        epochs = self.config['epochs']
        batch_size = self.config['batch_size']
        learning_rate = self.config['learning_rate']

        print(f"\n{'='*60}")
        print("RL ACTOR PRETRAINING")
        print(f"{'='*60}")
        print(f"Dataset size: {len(self.dataset)} examples")
        print(f"Epochs: {epochs}")
        print(f"Batch size: {batch_size}")
        print(f"Learning rate: {learning_rate}\n")

        optimizer = torch.optim.Adam(
            self.rl_agent.actor.parameters(),
            lr=learning_rate
        )

        for epoch in range(epochs):
            epoch_losses = []

            # Shuffle dataset
            indices = np.random.permutation(len(self.dataset))

            # Train in batches
            for i in range(0, len(indices), batch_size):
                batch_indices = indices[i:i+batch_size]
                batch_examples = [self.dataset[idx] for idx in batch_indices]

                # Encode contexts
                contexts = [ex["context"] for ex in batch_examples]
                state_embeddings = self.encoder.encode(
                    contexts,
                    convert_to_numpy=True,
                    show_progress_bar=False
                )

                # Get target embeddings
                target_embeddings = np.array([
                    ex["target_embedding"] for ex in batch_examples
                ])

                # Convert to tensors
                states = torch.FloatTensor(state_embeddings)
                targets = torch.FloatTensor(target_embeddings)

                # Forward pass
                predicted_embeddings = self.rl_agent.actor(states)

                # Normalize for cosine similarity
                predicted_normalized = F.normalize(predicted_embeddings, dim=1)
                targets_normalized = F.normalize(targets, dim=1)

                # Loss: MSE in normalized space
                loss = F.mse_loss(predicted_normalized, targets_normalized)

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.rl_agent.actor.parameters(), 1.0)
                optimizer.step()

                epoch_losses.append(loss.item())

            # Epoch summary
            avg_loss = np.mean(epoch_losses)
            self.train_losses.append(avg_loss)

            print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}")

            # Save checkpoint
            if (epoch + 1) % 5 == 0:
                self.save_checkpoint(f"actor_pretrain_epoch{epoch+1}.pt")

        print(f"\n{'='*60}")
        print("PRETRAINING COMPLETE")
        print(f"Final loss: {self.train_losses[-1]:.4f}")
        print(f"{'='*60}\n")

    def evaluate(self, num_samples: int = 5):
        """
        Evaluate trained model on sample examples.

        Shows what concepts actor predicts for different contexts.
        """
        print(f"\n{'='*60}")
        print("EVALUATION")
        print(f"{'='*60}\n")

        indices = np.random.choice(len(self.dataset), min(num_samples, len(self.dataset)), replace=False)

        for idx in indices:
            example = self.dataset[idx]

            # Encode context
            state = self.encoder.encode(example["context"], convert_to_numpy=True)

            # Predict embedding
            predicted_embedding = self.rl_agent.predict_embedding(state, explore=False)

            # Find nearest concepts
            nearest = self.embedding_space.find_nearest_entities(predicted_embedding, top_k=5)

            print(f"Context: {example['context'][:100]}...")
            print(f"Target concept: {example['concept']}")
            print(f"Predicted concepts: {[c for c, s in nearest]}")
            print(f"Similarities: {[f'{s:.3f}' for c, s in nearest]}")
            print()

    def save_checkpoint(self, filename: str):
        """Save training checkpoint."""
        checkpoint_path = Path("checkpoints") / filename
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

        torch.save({
            'actor_state_dict': self.rl_agent.actor.state_dict(),
            'train_losses': self.train_losses,
        }, checkpoint_path)

        print(f"Saved checkpoint: {checkpoint_path}")
