"""
Paper 2 shared environment: elicitation episodes against the CASPER
testbed instrument with a verifiable simulated user.

Used by both the continuous-action (Wolpertinger-style) trainer and the
discrete PPO comparator, so the two policy classes face byte-identical
dynamics, state features, and rewards.

State  : revealed one-hots [n_items*3] ++ instrument beliefs [n_items]
Action : an entity index (how it is CHOSEN differs per algorithm)
Reward : per-turn reduction in the instrument's BCE on the user's rated
         movies (the bot-play reward of Makarova et al., IJCNN 2024)
"""

import sys
sys.path.insert(0, '.')
sys.path.insert(0, 'scripts/paper1')

import numpy as np

from test_instrument_lib import load_instrument_by_name
from testbed import build_profiles, get_user_splits


def bce_loss(preds, profile, n_movies):
    preds = preds[:n_movies]
    gt = profile[:n_movies]
    filt = ~np.isnan(gt)
    p = np.clip(preds[filt], 1e-7, 1 - 1e-7)
    y = gt[filt]
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def accuracy(preds, profile, n_movies):
    preds = preds[:n_movies]
    gt = profile[:n_movies]
    filt = ~np.isnan(gt)
    return float(np.mean((preds[filt] > 0.5) == gt[filt]))


class ElicitationEnv:
    """Single-episode environment. step(entity) -> (state, reward, done).

    instrument        -- MEASUREMENT (reward/accuracy); never changes per row
    belief_instrument -- decision aid feeding the policy state (defaults to
                         the measurement instrument)
    state_mode        -- 'liked' (3n+n) or 'dual' (3n+2n, adds answerability
                         beliefs from the belief instrument's rated head)
    """

    def __init__(self, instrument, n_items, n_turns=15,
                 belief_instrument=None, state_mode='liked'):
        self.instrument = instrument
        self.belief_instrument = belief_instrument or instrument
        self.state_mode = state_mode
        self.n_items = n_items
        self.n_turns = n_turns

    @property
    def state_dim(self):
        return self.n_items * 3 + self.n_items * (2 if self.state_mode == 'dual' else 1)

    def _refresh(self):
        self.beliefs = self.belief_instrument.predict_full(self.revealed)
        if self.state_mode == 'dual':
            self.rated_beliefs = self.belief_instrument.predict_rated(self.revealed)
        if self.belief_instrument is self.instrument:
            self.meas = self.beliefs
        else:
            self.meas = self.instrument.predict_full(self.revealed)

    def reset(self, profile):
        self.profile = profile
        self.revealed = []
        self.asked = set()
        self.t = 0
        self._refresh()
        self.loss_prev = bce_loss(self.meas, profile, self.instrument.n_movies)
        return self._state()

    def _state(self):
        s = np.zeros((self.n_items, 3), dtype=np.float32)
        s[:, 2] = 1
        for idx, pol in self.revealed:
            s[idx, 2] = 0
            s[idx, 1 if pol >= 0.5 else 0] = 1
        parts = [s.flatten(), self.beliefs.astype(np.float32)]
        if self.state_mode == 'dual':
            parts.append(self.rated_beliefs.astype(np.float32))
        return np.concatenate(parts)

    def step(self, entity):
        assert entity not in self.asked
        self.asked.add(entity)
        v = self.profile[entity]
        if not np.isnan(v):
            self.revealed.append((entity, 1.0 if v >= 0.5 else 0.0))
        self._refresh()
        loss_now = bce_loss(self.meas, self.profile, self.instrument.n_movies)
        reward = self.loss_prev - loss_now
        self.loss_prev = loss_now
        self.t += 1
        done = self.t >= self.n_turns
        return self._state(), reward, done

    def episode_accuracy(self):
        return accuracy(self.meas, self.profile, self.instrument.n_movies)


def load_world(seed=42, n_train_users=6000, n_val_users=100, world='slate1'):
    """Instrument + entity embeddings + profiles + splits, shared by trainers."""
    if world == 'slate2':
        return _load_world_slate2(seed, n_train_users, n_val_users)
    instrument, ckpt = load_instrument_by_name('instrument_v5_set')
    items = ckpt['items']
    n_items = len(items)

    # Semantic action geometry: SBERT embeddings of the slate entities
    # (cached by the original concept-model training; same item ordering).
    emb_path = ('C:/dev/phd/casper/data/movielens/.cache/checkpoints/'
                'concept_embeddings_paper_config.npy')
    entity_emb = np.load(emb_path).astype(np.float32)
    entity_emb /= np.linalg.norm(entity_emb, axis=1, keepdims=True) + 1e-9
    assert entity_emb.shape[0] == n_items

    train_users, ival_users, _ = get_user_splits(items)
    rng = np.random.default_rng(seed)
    train_users = rng.choice(train_users, size=n_train_users, replace=False)
    val_users = ival_users[:n_val_users]
    profiles = build_profiles(items, user_ids=np.concatenate([train_users, val_users]),
                              attr_min_support=3, taste_margin=None)
    train_uids = [u for u in train_users if u in profiles]
    val_uids = [u for u in val_users if u in profiles]
    return {
        'instrument': instrument, 'items': items, 'n_items': n_items,
        'entity_emb': entity_emb, 'profiles': profiles,
        'train_uids': train_uids, 'val_uids': val_uids,
    }


def _load_world_slate2(seed, n_train_users, n_val_users):
    from train_instrument_slate2 import (build_slate, build_profiles_slate2,
                                         MIN_USER_RATINGS)
    instrument, ckpt = load_instrument_by_name('instrument_slate2_dual')
    items = ckpt['items']
    n_items = len(items)
    _, n_movies, movie_pos, attr_of_movie, ratings_f = build_slate()
    user_counts = ratings_f.groupby('userId').size()
    dense = user_counts[user_counts >= MIN_USER_RATINGS].index.to_numpy()
    np.random.seed(seed)
    shuffled = dense.copy()
    np.random.shuffle(shuffled)
    split = int(0.8 * len(shuffled))
    rng = np.random.default_rng(seed)
    train_users = rng.choice(shuffled[:split], size=n_train_users, replace=False)
    val_users = shuffled[split:split + n_val_users]
    profiles = build_profiles_slate2(items, n_items, movie_pos, attr_of_movie,
                                     ratings_f,
                                     np.concatenate([train_users, val_users]))
    return {
        'instrument': instrument, 'items': items, 'n_items': n_items,
        'entity_emb': None, 'profiles': profiles,
        'train_uids': [u for u in train_users if u in profiles],
        'val_uids': [u for u in val_users if u in profiles],
    }
