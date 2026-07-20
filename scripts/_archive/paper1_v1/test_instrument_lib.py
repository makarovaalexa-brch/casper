"""
Shared instrument definitions for the CASPER elicitation testbed (Paper 1).

The instrument is a fixed, pre-trained extrapolation recommender over a
187-item slate (100 movies + 17 genres + 50 actors + 20 directors). It
exposes a single interface:

    wrapper.predict(revealed) -> np.ndarray[n_movies] of P(liked)

where revealed is a list of (item_idx, polarity) pairs, polarity 1.0=liked,
0.0=disliked. Item ordering is taken from ckpt['items'] (authoritative).
"""

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

DATA_DIR = Path('C:/dev/phd/casper/data/movielens')
CHECKPOINT_DIR = DATA_DIR / '.cache' / 'checkpoints'


class ExtrapolationModel(nn.Module):
    """One-hot (learned embedding) extrapolation model, CSMAI-19 style."""

    def __init__(self, n_items):
        super().__init__()
        self.n_items = n_items
        hidden_dim = n_items // 2
        self.embedding = nn.Embedding(n_items + 1, hidden_dim)
        self.lstm = nn.LSTM(hidden_dim + 3, hidden_dim, 1, batch_first=True)
        self.dense1 = nn.Linear(hidden_dim, n_items)
        self.dense2 = nn.Linear(n_items, hidden_dim)
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads=1, batch_first=True)
        self.output = nn.Linear(hidden_dim * 2, n_items)

    def forward(self, index_input, rating_input):
        x = self.embedding(index_input)
        x = torch.cat((x, rating_input), dim=-1)
        encoder_output, _ = self.lstm(x)
        x = torch.relu(self.dense1(encoder_output))
        x = torch.relu(self.dense2(x))
        attention, _ = self.attention(encoder_output, x, x)
        x = torch.cat((x, attention), dim=-1)
        return self.output(x)


class ConceptEmbeddingModel(nn.Module):
    """SBERT-embedding variant of the extrapolation model."""

    def __init__(self, n_items, embedding_dim=384, hidden_dim=None):
        super().__init__()
        self.n_items = n_items
        hidden_dim = hidden_dim or n_items // 2
        self.concept_proj = nn.Linear(embedding_dim, hidden_dim)
        self.lstm = nn.LSTM(hidden_dim + 3, hidden_dim, 1, batch_first=True)
        self.dense1 = nn.Linear(hidden_dim, n_items)
        self.dense2 = nn.Linear(n_items, hidden_dim)
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads=1, batch_first=True)
        self.output = nn.Linear(hidden_dim * 2, n_items)

    def forward(self, item_embeddings, rating_input):
        x = self.concept_proj(item_embeddings)
        x = torch.cat((x, rating_input), dim=-1)
        encoder_output, _ = self.lstm(x)
        x = torch.relu(self.dense1(encoder_output))
        x = torch.relu(self.dense2(x))
        attention, _ = self.attention(encoder_output, x, x)
        x = torch.cat((x, attention), dim=-1)
        return self.output(x)


class InstrumentWrapper:
    """predict(revealed) -> movie scores; revealed = [(item_idx, polarity)]."""

    def __init__(self, model, n_items, n_movies, item_embeddings=None):
        self.model = model
        self.n_items = n_items
        self.n_movies = n_movies
        self.item_embeddings = item_embeddings
        if item_embeddings is not None:
            self._emb = torch.FloatTensor(item_embeddings).unsqueeze(0)
        else:
            self._idx = torch.arange(n_items).unsqueeze(0)

    def _build_input(self, revealed):
        rating_input = torch.zeros(self.n_items, 3)
        rating_input[:, 2] = 1
        for idx, pol in revealed:
            rating_input[idx, 2] = 0
            if pol >= 0.5:
                rating_input[idx, 1] = 1
            else:
                rating_input[idx, 0] = 1
        return rating_input

    def _forward(self, rating_batch):
        b = rating_batch.shape[0]
        with torch.no_grad():
            if self.item_embeddings is not None:
                out = self.model(self._emb.expand(b, -1, -1), rating_batch)
            else:
                out = self.model(self._idx.expand(b, -1), rating_batch)
        return out[:, -1, :].sigmoid().numpy()

    def predict(self, revealed):
        """Movie scores [n_movies] given revealed [(item_idx, polarity)]."""
        out = self._forward(self._build_input(revealed).unsqueeze(0))
        return out[0, :self.n_movies]

    def predict_full(self, revealed):
        """Scores for the FULL slate [n_items] (movies + attributes)."""
        out = self._forward(self._build_input(revealed).unsqueeze(0))
        return out[0]

    def predict_batch(self, revealed_list, full=False):
        """Batched prediction for many hypothetical reveal sets at once."""
        batch = torch.stack([self._build_input(r) for r in revealed_list])
        out = self._forward(batch)
        return out if full else out[:, :self.n_movies]


class SetInstrumentWrapper:
    """Same interface as InstrumentWrapper, for the v5 set-encoder model."""

    def __init__(self, model, n_items, n_movies, max_reveal=60):
        self.model = model
        self.n_items = n_items
        self.n_movies = n_movies
        self.max_reveal = max_reveal

    def _forward(self, revealed_list):
        b = len(revealed_list)
        L = max(1, max((len(r) for r in revealed_list), default=1))
        idx = torch.zeros(b, L, dtype=torch.long)
        pol = torch.zeros(b, L, dtype=torch.long)
        pad = torch.ones(b, L, dtype=torch.bool)
        for i, revealed in enumerate(revealed_list):
            for j, (e, p) in enumerate(revealed[:self.max_reveal]):
                idx[i, j] = int(e)
                pol[i, j] = 1 if p >= 0.5 else 0
                pad[i, j] = False
        with torch.no_grad():
            return torch.sigmoid(self.model(idx, pol, pad)).numpy()

    def predict(self, revealed):
        return self._forward([revealed])[0, :self.n_movies]

    def predict_full(self, revealed):
        return self._forward([revealed])[0]

    def predict_batch(self, revealed_list, full=False):
        out = self._forward(revealed_list)
        return out if full else out[:, :self.n_movies]


def load_set_instrument(ckpt):
    """Construct the v5 set-encoder wrapper from a checkpoint dict."""
    from train_instrument_v5 import SetEncoderInstrument
    m = SetEncoderInstrument(ckpt['n_items'], ckpt.get('d_model', 128),
                             ckpt.get('n_heads', 4), ckpt.get('n_layers', 2))
    m.load_state_dict(ckpt['model_state_dict'])
    m.eval()
    return SetInstrumentWrapper(m, ckpt['n_items'], ckpt.get('n_movies', 100),
                                ckpt.get('config', {}).get('max_reveal', 60))


def load_instruments():
    """Load all available candidate instruments as {name: (wrapper, ckpt)}."""
    instruments = {}

    ckpt_path = CHECKPOINT_DIR / 'instrument_v5_set.pt'
    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, weights_only=False)
        instruments['instrument_v5_set'] = (load_set_instrument(ckpt), ckpt)

    ckpt_path = CHECKPOINT_DIR / 'instrument_v4_onehot.pt'
    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, weights_only=False)
        m = ExtrapolationModel(ckpt['n_items'])
        m.load_state_dict(ckpt['model_state_dict'])
        m.eval()
        instruments['instrument_v4_onehot'] = (
            InstrumentWrapper(m, ckpt['n_items'], ckpt.get('n_movies', 100)), ckpt)

    ckpt_path = CHECKPOINT_DIR / 'instrument_v3_onehot.pt'
    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, weights_only=False)
        m = ExtrapolationModel(ckpt['n_items'])
        m.load_state_dict(ckpt['model_state_dict'])
        m.eval()
        instruments['instrument_v3_onehot'] = (
            InstrumentWrapper(m, ckpt['n_items'], ckpt.get('n_movies', 100)), ckpt)

    ckpt_path = CHECKPOINT_DIR / 'instrument_v2_onehot.pt'
    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, weights_only=False)
        m = ExtrapolationModel(ckpt['n_items'])
        m.load_state_dict(ckpt['model_state_dict'])
        m.eval()
        instruments['instrument_v2_onehot'] = (
            InstrumentWrapper(m, ckpt['n_items'], ckpt.get('n_movies', 100)), ckpt)

    ckpt_path = CHECKPOINT_DIR / 'onehot_paper_config.pt'
    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, weights_only=False)
        m = ExtrapolationModel(ckpt['n_items'])
        m.load_state_dict(ckpt['model_state_dict'])
        m.eval()
        instruments['onehot'] = (
            InstrumentWrapper(m, ckpt['n_items'], ckpt.get('n_movies', 100)), ckpt)

    ckpt_path = CHECKPOINT_DIR / 'concept_paper_config.pt'
    emb_path = CHECKPOINT_DIR / 'concept_embeddings_paper_config.npy'
    if ckpt_path.exists() and emb_path.exists():
        ckpt = torch.load(ckpt_path, weights_only=False)
        emb = np.load(emb_path)
        hidden_dim = ckpt['model_state_dict']['concept_proj.bias'].shape[0]
        m = ConceptEmbeddingModel(ckpt['n_items'], ckpt.get('embedding_dim', 384), hidden_dim)
        m.load_state_dict(ckpt['model_state_dict'])
        m.eval()
        instruments['concept'] = (
            InstrumentWrapper(m, ckpt['n_items'], ckpt.get('n_movies', 100), emb), ckpt)

    return instruments


def load_instrument_by_name(name):
    """Load any instrument generation by checkpoint stem; returns (wrapper, ckpt)."""
    path = CHECKPOINT_DIR / f'{name}.pt'
    if not path.exists():
        raise FileNotFoundError(f"Instrument checkpoint missing: {path}")
    ckpt = torch.load(path, weights_only=False)
    if ckpt.get('arch') == 'dual_set_encoder':
        return load_dual_set_instrument(ckpt), ckpt
    if ckpt.get('arch') == 'set_encoder':
        return load_set_instrument(ckpt), ckpt
    m = ExtrapolationModel(ckpt['n_items'])
    m.load_state_dict(ckpt['model_state_dict'])
    m.eval()
    return InstrumentWrapper(m, ckpt['n_items'], ckpt.get('n_movies', 100)), ckpt


class DualSetInstrumentWrapper(SetInstrumentWrapper):
    """Dual-head set encoder: predict() returns liked beliefs (movies),
    predict_rated() returns answerability beliefs (full slate)."""

    def _forward_dual(self, revealed_list):
        b = len(revealed_list)
        L = max(1, max((len(r) for r in revealed_list), default=1))
        idx = torch.zeros(b, L, dtype=torch.long)
        pol = torch.zeros(b, L, dtype=torch.long)
        pad = torch.ones(b, L, dtype=torch.bool)
        for i, revealed in enumerate(revealed_list):
            for j, (e, p) in enumerate(revealed[:self.max_reveal]):
                idx[i, j] = int(e)
                pol[i, j] = 1 if p >= 0.5 else 0
                pad[i, j] = False
        with torch.no_grad():
            lk, rt = self.model(idx, pol, pad)
        return torch.sigmoid(lk).numpy(), torch.sigmoid(rt).numpy()

    def _forward(self, revealed_list):
        return self._forward_dual(revealed_list)[0]

    def predict_rated(self, revealed):
        return self._forward_dual([revealed])[1][0]


def load_dual_set_instrument(ckpt):
    from synthetic_sanity import DualHeadSetEncoder
    m = DualHeadSetEncoder(ckpt['n_items'], ckpt.get('d_model', 128),
                           ckpt.get('n_heads', 4), ckpt.get('n_layers', 2))
    m.load_state_dict(ckpt['model_state_dict'])
    m.eval()
    return DualSetInstrumentWrapper(m, ckpt['n_items'],
                                    ckpt.get('n_movies', ckpt['n_items']),
                                    ckpt.get('config', {}).get('max_reveal', 60))
