"""
CASPER testbed core (Paper 1 / M2): Controlled Assessment of Strategic
Preference Elicitation in Recommendation.

Design contract:
- The INSTRUMENT (a fixed pre-trained extrapolation recommender) is the
  measurement device. It is never updated during evaluation.
- The SIMULATED USER answers entity questions deterministically from a
  held-out ground-truth profile: 'liked' / 'disliked' / 'unknown'. Every
  answer is auditable against the profile (no LLM in the answering loop).
- A POLICY selects, at each turn, one not-yet-asked entity from the shared
  187-item slate. Strategy is the only varying factor.
- METRICS: per-turn accuracy and BCE of the instrument's movie predictions
  against the user's rated movies; area under the accuracy curve (AUAC);
  hit rate (fraction of questions landing on rated entities).

Users: held-out split (seed 42, same protocol as instrument training),
excluding any user index used for instrument training or model selection.
"""

import sys
sys.path.insert(0, '.')
sys.path.insert(0, 'scripts/paper1')

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path('C:/dev/phd/casper/data/movielens')
OUT_DIR = Path('C:/dev/phd/casper/experiments/paper1')
OUT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
INSTRUMENT_VAL_CAP = 1500  # first N of the held-out split were used for early stopping


# ---------------------------------------------------------------------------
# Profiles (identical derivation to train_instrument_v2.py)
# ---------------------------------------------------------------------------

def build_profiles(items, user_ids=None, max_users=None,
                   attr_min_support=3, taste_margin=0.25):
    """Per-user ground-truth vector over the slate: 1.0 liked / 0.0 disliked / nan unrated.

    Movie label: rating >= 4 (absolute, as in prior work).
    Attribute label: TASTE-RELATIVE. In a slate of popular classics, an
    absolute attribute label (mean matching rating >= 4) mostly encodes
    rater harshness, not taste: disliking the horror classics correlates
    with rating everything low. We therefore label an attribute liked /
    disliked only when the user's mean rating on matching movies deviates
    from their own overall slate mean by at least +-taste_margin, with at
    least attr_min_support matching rated movies; otherwise the attribute
    is unrated (simulator answers 'unknown').
    """
    n_items = len(items)
    movie_to_idx = {it[1]: i for i, it in enumerate(items) if it[0] == 'movie'}

    movies_df = pd.read_csv(DATA_DIR / 'movies.csv')
    credits_path = DATA_DIR / '.cache' / 'credits_top100_actors5.json'
    credits = json.loads(credits_path.read_text()) if credits_path.exists() else {}

    genres_in_slate = {it[1]: i for i, it in enumerate(items) if it[0] == 'genre'}
    actors_in_slate = {it[1]: i for i, it in enumerate(items) if it[0] == 'actor'}
    directors_in_slate = {it[1]: i for i, it in enumerate(items) if it[0] == 'director'}

    movie_attrs = {}
    for mid in movie_to_idx:
        attrs = []
        row = movies_df[movies_df['movieId'] == mid]
        if len(row) and pd.notna(row.iloc[0]['genres']):
            attrs += [genres_in_slate[g] for g in row.iloc[0]['genres'].split('|')
                      if g in genres_in_slate]
        mstr = str(mid)
        attrs += [actors_in_slate[a] for a in credits.get('movie_actors', {}).get(mstr, [])[:5]
                  if a in actors_in_slate]
        attrs += [directors_in_slate[d] for d in credits.get('movie_directors', {}).get(mstr, [])
                  if d in directors_in_slate]
        movie_attrs[mid] = attrs

    ratings = pd.read_csv(DATA_DIR / 'ratings.csv')
    ratings_f = ratings[ratings['movieId'].isin(movie_to_idx)]

    if user_ids is None:
        user_counts = ratings_f.groupby('userId').size()
        user_ids = user_counts[user_counts >= 25].index.to_numpy()
    if max_users:
        user_ids = user_ids[:max_users]

    profiles = {}
    grouped = ratings_f[ratings_f['userId'].isin(set(user_ids))].groupby('userId')
    for uid, grp in grouped:
        vec = np.full(n_items, np.nan, dtype=np.float32)
        attr_sums = np.zeros(n_items)
        attr_cnts = np.zeros(n_items)
        rating_sum, rating_cnt = 0.0, 0
        for r in grp.itertuples():
            idx = movie_to_idx[r.movieId]
            vec[idx] = 1.0 if r.rating >= 4 else 0.0
            rating_sum += r.rating
            rating_cnt += 1
            for ai in movie_attrs[r.movieId]:
                attr_sums[ai] += r.rating
                attr_cnts[ai] += 1
        user_mean = rating_sum / max(rating_cnt, 1)
        has = attr_cnts >= attr_min_support
        if taste_margin is None:
            # absolute labels (>=4) with support threshold
            means = np.full(n_items, np.nan)
            means[has] = attr_sums[has] / attr_cnts[has]
            vec[means >= 4.0] = 1.0
            vec[(means < 4.0) & has] = 0.0
        else:
            rel = np.full(n_items, np.nan)
            rel[has] = attr_sums[has] / attr_cnts[has] - user_mean
            vec[rel >= taste_margin] = 1.0
            vec[rel <= -taste_margin] = 0.0
        profiles[uid] = vec
    return profiles


def get_user_splits(items):
    """Reproduce the seed-42 split; return (train_users, instrument_val, testbed_users)."""
    movie_to_idx = {it[1]: i for i, it in enumerate(items) if it[0] == 'movie'}
    ratings = pd.read_csv(DATA_DIR / 'ratings.csv')
    ratings_f = ratings[ratings['movieId'].isin(movie_to_idx)]
    user_counts = ratings_f.groupby('userId').size()
    dense = user_counts[user_counts >= 25].index.to_numpy()
    np.random.seed(SEED)
    shuffled = dense.copy()
    np.random.shuffle(shuffled)
    split = int(0.8 * len(shuffled))
    return (shuffled[:split],
            shuffled[split:split + INSTRUMENT_VAL_CAP],
            shuffled[split + INSTRUMENT_VAL_CAP:])


# ---------------------------------------------------------------------------
# Simulated user
# ---------------------------------------------------------------------------

class SimulatedUser:
    """Deterministic, verifiable: answers from the ground-truth profile."""

    def __init__(self, uid, profile):
        self.uid = uid
        self.profile = profile

    def answer(self, entity_idx):
        v = self.profile[entity_idx]
        if np.isnan(v):
            return 'unknown'
        return 'liked' if v >= 0.5 else 'disliked'


# ---------------------------------------------------------------------------
# Episode
# ---------------------------------------------------------------------------

@dataclass
class EpisodeLog:
    uid: int
    questions: list = field(default_factory=list)   # entity indices asked
    answers: list = field(default_factory=list)     # 'liked'/'disliked'/'unknown'
    accuracy: list = field(default_factory=list)    # per turn, incl. turn 0 prior
    bce: list = field(default_factory=list)
    asked_scores: list = field(default_factory=list)  # instrument's pre-question
                                                      # score for the asked entity
                                                      # (for redundancy analysis)

    def to_dict(self):
        return {'uid': int(self.uid), 'questions': [int(q) for q in self.questions],
                'answers': self.answers,
                'accuracy': [float(a) for a in self.accuracy],
                'bce': [float(b) for b in self.bce],
                'asked_scores': [float(s) for s in self.asked_scores]}


def _movie_metrics(preds, profile, n_movies):
    gt = profile[:n_movies]
    filt = ~np.isnan(gt)
    if filt.sum() == 0:
        return np.nan, np.nan
    p = np.clip(preds[filt], 1e-7, 1 - 1e-7)
    y = gt[filt]
    acc = float(np.mean((p > 0.5) == y))
    bce = float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))
    return acc, bce


def run_episode(policy, user, instrument, n_turns, rng):
    """One elicitation dialogue. Policy sees only asked/answers, never the profile."""
    revealed = []          # [(entity_idx, polarity)] for instrument input
    log = EpisodeLog(uid=user.uid)

    preds = instrument.predict(revealed)
    acc, bce = _movie_metrics(preds, user.profile, instrument.n_movies)
    log.accuracy.append(acc)
    log.bce.append(bce)

    policy.reset(rng=rng)
    for _ in range(n_turns):
        asked = set(log.questions)
        q = policy.select(asked=asked, history=list(zip(log.questions, log.answers)),
                          instrument=instrument, revealed=list(revealed))
        if q is None or q in asked:
            break
        pre_full = instrument.predict_full(revealed)
        log.asked_scores.append(float(pre_full[q]))
        a = user.answer(q)
        log.questions.append(q)
        log.answers.append(a)
        if a == 'liked':
            revealed.append((q, 1.0))
        elif a == 'disliked':
            revealed.append((q, 0.0))
        preds = instrument.predict(revealed)
        acc, bce = _movie_metrics(preds, user.profile, instrument.n_movies)
        log.accuracy.append(acc)
        log.bce.append(bce)
    return log


# ---------------------------------------------------------------------------
# Evaluation loop
# ---------------------------------------------------------------------------

def evaluate_policy(policy, users, profiles, instrument, n_turns=15, seed=SEED,
                    progress_every=50):
    rng = np.random.default_rng(seed)
    logs = []
    t0 = time.time()
    for i, uid in enumerate(users):
        if uid not in profiles:
            continue
        user = SimulatedUser(uid, profiles[uid])
        logs.append(run_episode(policy, user, instrument, n_turns, rng))
        if progress_every and (i + 1) % progress_every == 0:
            print(f"  {policy.name}: {i + 1}/{len(users)} users "
                  f"({time.time() - t0:.0f}s)", flush=True)
    return logs


def summarize(logs, n_turns):
    """Mean per-turn curves with bootstrap CIs, AUAC, hit rate."""
    curves = []
    hits = []
    for lg in logs:
        c = lg.accuracy + [lg.accuracy[-1]] * (n_turns + 1 - len(lg.accuracy))
        curves.append(c[:n_turns + 1])
        if lg.answers:
            hits.append(np.mean([a != 'unknown' for a in lg.answers]))
    curves = np.array(curves)
    mean_curve = curves.mean(axis=0)

    # bootstrap 95% CI on the final-turn accuracy and AUAC
    rng = np.random.default_rng(0)
    n = len(curves)
    finals, auacs = [], []
    for _ in range(1000):
        idx = rng.integers(0, n, n)
        finals.append(curves[idx, -1].mean())
        auacs.append(curves[idx].mean())
    return {
        'n_users': n,
        'turn0_accuracy': float(mean_curve[0]),
        'final_accuracy': float(mean_curve[-1]),
        'final_accuracy_ci95': [float(np.percentile(finals, 2.5)),
                                float(np.percentile(finals, 97.5))],
        'auac': float(curves.mean()),
        'auac_ci95': [float(np.percentile(auacs, 2.5)),
                      float(np.percentile(auacs, 97.5))],
        'hit_rate': float(np.mean(hits)) if hits else np.nan,
        'mean_curve': [float(x) for x in mean_curve],
    }
