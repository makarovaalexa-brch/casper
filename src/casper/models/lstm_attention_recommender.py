"""
LSTM + Attention Recommender with explicit rating encoding.

Based on IEEE Paper "Learning a Strategy for Preference Elicitation in
Conversational Recommender Systems" and CSMAI-19 notebook.

Key insight: Explicit one-hot rating encoding [liked, disliked, unknown]
ensures the model can distinguish preferences from anti-preferences,
unlike SBERT text encoding which conflates "likes: X" and "dislikes: X".
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from typing import List, Tuple, Optional
import numpy as np


class LSTMAttentionUserTower(nn.Module):
    """
    User tower that processes a sequence of (concept_embedding, rating) pairs.

    Architecture (from IEEE paper / CSMAI-19):
    1. Concatenate concept embedding with rating one-hot
    2. Process sequence with LSTM
    3. Apply attention over LSTM outputs
    4. Pool to single user embedding

    Input:
        concept_embeddings: [batch, seq_len, concept_dim] - SBERT embeddings
        rating_one_hots: [batch, seq_len, 3] - [liked, disliked, unknown]

    Output:
        user_embedding: [batch, output_dim]
    """

    def __init__(
        self,
        concept_dim: int = 384,  # SBERT dimension
        hidden_dim: int = 256,
        output_dim: int = 128,
        num_layers: int = 1,
        num_heads: int = 4,
        dropout: float = 0.1,
        bidirectional: bool = False,
    ):
        super().__init__()

        self.concept_dim = concept_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # Input projection (concept + rating one-hot)
        input_dim = concept_dim + 3  # +3 for rating one-hot [liked, disliked, unknown]

        # LSTM encoder
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        lstm_output_dim = hidden_dim * (2 if bidirectional else 1)

        # Decoder dense layers (from paper)
        self.dense1 = nn.Linear(lstm_output_dim, hidden_dim * 2)
        self.dense2 = nn.Linear(hidden_dim * 2, lstm_output_dim)

        # Multi-head attention
        self.attention = nn.MultiheadAttention(
            embed_dim=lstm_output_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        # Output projection
        # Concat dense output + attention output, then project
        self.output_proj = nn.Linear(lstm_output_dim * 2, output_dim)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(output_dim)

    def forward(
        self,
        concept_embeddings: torch.Tensor,
        rating_one_hots: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            concept_embeddings: [batch, seq_len, concept_dim]
            rating_one_hots: [batch, seq_len, 3]
            mask: [batch, seq_len] - True for valid positions, False for padding

        Returns:
            user_embedding: [batch, output_dim]
        """
        batch_size, seq_len, _ = concept_embeddings.shape

        # Concatenate concept embedding with rating one-hot
        x = torch.cat([concept_embeddings, rating_one_hots], dim=-1)

        # LSTM encoding
        lstm_out, _ = self.lstm(x)  # [batch, seq_len, hidden_dim]

        # Decoder: dense layers
        dense_out = F.relu(self.dense1(lstm_out))
        dense_out = F.relu(self.dense2(dense_out))

        # Attention: encoder output attends to dense output
        # key_padding_mask: True = ignore this position
        key_padding_mask = ~mask if mask is not None else None
        attn_out, _ = self.attention(
            query=lstm_out,
            key=dense_out,
            value=dense_out,
            key_padding_mask=key_padding_mask,
        )

        # Concatenate dense and attention outputs
        combined = torch.cat([dense_out, attn_out], dim=-1)

        # Pool over sequence dimension
        if mask is not None:
            # Masked mean pooling
            mask_expanded = mask.unsqueeze(-1).float()
            combined = combined * mask_expanded
            pooled = combined.sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1)
        else:
            # Simple mean pooling
            pooled = combined.mean(dim=1)

        # Output projection
        user_emb = self.output_proj(pooled)
        user_emb = self.layer_norm(user_emb)

        return user_emb


class ItemTower(nn.Module):
    """
    Simple item tower that projects item embeddings.
    """

    def __init__(
        self,
        item_dim: int = 384,  # SBERT dimension
        output_dim: int = 128,
        hidden_dim: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(item_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim),
        )

    def forward(self, item_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Args:
            item_embeddings: [batch, item_dim] or [batch, num_items, item_dim]
        Returns:
            projected: same shape with last dim = output_dim
        """
        return self.network(item_embeddings)


class LSTMAttentionRecommender(nn.Module):
    """
    Two-tower recommender with LSTM+Attention user tower.

    User tower: Processes sequence of (concept_embedding, rating_one_hot)
    Item tower: Projects item embeddings
    Score: Dot product of user and item embeddings

    This architecture ensures explicit encoding of like/dislike ratings,
    solving the problem where SBERT conflates "likes: X" and "dislikes: X".
    """

    def __init__(
        self,
        concept_dim: int = 384,
        item_dim: int = 384,
        hidden_dim: int = 256,
        output_dim: int = 128,
        num_lstm_layers: int = 1,
        num_attention_heads: int = 4,
        dropout: float = 0.1,
        learning_rate: float = 0.001,
        temperature: float = 0.07,
        normalize: bool = True,
    ):
        super().__init__()

        self.output_dim = output_dim
        self.temperature = temperature
        self.normalize = normalize

        # User tower (LSTM + Attention)
        self.user_tower = LSTMAttentionUserTower(
            concept_dim=concept_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            num_layers=num_lstm_layers,
            num_heads=num_attention_heads,
            dropout=dropout,
        )

        # Item tower
        self.item_tower = ItemTower(
            item_dim=item_dim,
            output_dim=output_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )

        # Optimizer
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    def encode_user(
        self,
        concept_embeddings: torch.Tensor,
        rating_one_hots: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Encode user preferences to embedding."""
        user_emb = self.user_tower(concept_embeddings, rating_one_hots, mask)
        if self.normalize:
            user_emb = F.normalize(user_emb, p=2, dim=-1)
        return user_emb

    def encode_items(self, item_embeddings: torch.Tensor) -> torch.Tensor:
        """Encode items to embeddings."""
        item_emb = self.item_tower(item_embeddings)
        if self.normalize:
            item_emb = F.normalize(item_emb, p=2, dim=-1)
        return item_emb

    def predict_scores(
        self,
        concept_embeddings: torch.Tensor,
        rating_one_hots: torch.Tensor,
        item_embeddings: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Predict scores for items given user preferences.

        Args:
            concept_embeddings: [batch, seq_len, concept_dim] or [seq_len, concept_dim]
            rating_one_hots: [batch, seq_len, 3] or [seq_len, 3]
            item_embeddings: [num_items, item_dim]
            mask: [batch, seq_len] or [seq_len]

        Returns:
            scores: [batch, num_items] or [num_items]
        """
        # Handle unbatched input
        squeeze_output = False
        if concept_embeddings.dim() == 2:
            concept_embeddings = concept_embeddings.unsqueeze(0)
            rating_one_hots = rating_one_hots.unsqueeze(0)
            if mask is not None:
                mask = mask.unsqueeze(0)
            squeeze_output = True

        # Encode user
        user_emb = self.encode_user(concept_embeddings, rating_one_hots, mask)  # [batch, output_dim]

        # Encode items
        item_emb = self.encode_items(item_embeddings)  # [num_items, output_dim]

        # Dot product scores
        scores = torch.matmul(user_emb, item_emb.T)  # [batch, num_items]

        if squeeze_output:
            scores = scores.squeeze(0)

        return scores

    def train_infonce_step(
        self,
        concept_embeddings: torch.Tensor,
        rating_one_hots: torch.Tensor,
        positive_items: torch.Tensor,
        negative_items: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> float:
        """
        Train step using InfoNCE loss.

        Args:
            concept_embeddings: [batch, seq_len, concept_dim]
            rating_one_hots: [batch, seq_len, 3]
            positive_items: [batch, item_dim]
            negative_items: [batch, num_neg, item_dim] or [num_neg, item_dim]
            mask: [batch, seq_len]

        Returns:
            loss value
        """
        self.train()
        self.optimizer.zero_grad()

        batch_size = concept_embeddings.shape[0]

        # Encode user
        user_emb = self.encode_user(concept_embeddings, rating_one_hots, mask)  # [batch, output_dim]

        # Encode positive items
        pos_emb = self.encode_items(positive_items)  # [batch, output_dim]

        # Encode negative items
        if negative_items.dim() == 2:
            # Same negatives for all users: [num_neg, item_dim]
            neg_emb = self.encode_items(negative_items)  # [num_neg, output_dim]
            # Compute negative scores: [batch, num_neg]
            neg_scores = torch.matmul(user_emb, neg_emb.T) / self.temperature
        else:
            # Different negatives per user: [batch, num_neg, item_dim]
            neg_emb = self.encode_items(negative_items)  # [batch, num_neg, output_dim]
            # Compute negative scores: [batch, num_neg]
            neg_scores = torch.bmm(neg_emb, user_emb.unsqueeze(-1)).squeeze(-1) / self.temperature

        # Positive scores: [batch]
        pos_scores = (user_emb * pos_emb).sum(dim=-1) / self.temperature

        # InfoNCE loss: -log(exp(pos) / (exp(pos) + sum(exp(neg))))
        # Equivalent to cross-entropy with positive as class 0
        logits = torch.cat([pos_scores.unsqueeze(1), neg_scores], dim=1)  # [batch, 1+num_neg]
        labels = torch.zeros(batch_size, dtype=torch.long, device=logits.device)

        loss = F.cross_entropy(logits, labels)

        loss.backward()
        self.optimizer.step()

        return loss.item()


def create_rating_one_hot(ratings: List[str], device: torch.device = None) -> torch.Tensor:
    """
    Convert rating strings to one-hot tensors.

    Args:
        ratings: List of "liked", "disliked", or "unknown"
        device: Target device

    Returns:
        one_hot: [seq_len, 3] tensor
    """
    one_hot = torch.zeros(len(ratings), 3, device=device)
    for i, rating in enumerate(ratings):
        if rating == "liked":
            one_hot[i, 0] = 1.0
        elif rating == "disliked":
            one_hot[i, 1] = 1.0
        else:  # unknown
            one_hot[i, 2] = 1.0
    return one_hot


def pad_sequences(
    sequences: List[torch.Tensor],
    max_len: Optional[int] = None,
    pad_value: float = 0.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Pad sequences to same length.

    Args:
        sequences: List of [seq_len, dim] tensors
        max_len: Maximum length (default: max of all sequences)
        pad_value: Value to pad with

    Returns:
        padded: [batch, max_len, dim]
        mask: [batch, max_len] - True for valid positions
    """
    if max_len is None:
        max_len = max(seq.shape[0] for seq in sequences)

    batch_size = len(sequences)
    dim = sequences[0].shape[-1]
    device = sequences[0].device

    padded = torch.full((batch_size, max_len, dim), pad_value, device=device)
    mask = torch.zeros(batch_size, max_len, dtype=torch.bool, device=device)

    for i, seq in enumerate(sequences):
        seq_len = seq.shape[0]
        padded[i, :seq_len] = seq
        mask[i, :seq_len] = True

    return padded, mask
