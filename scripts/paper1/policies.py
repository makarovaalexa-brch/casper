"""
Elicitation policies for the CASPER testbed (Paper 1 / M3).

Every policy implements:
    reset(rng)                      -- called once per episode
    select(asked, history, instrument, revealed) -> entity_idx or None
    name                            -- identifier for results

All policies choose from the same 187-entity slate; the strategy is the
only varying factor. Policies never see the user's ground-truth profile.
"""

import json
import os
import re
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# Non-learned baselines
# ---------------------------------------------------------------------------

class BasePolicy:
    name = 'base'

    def __init__(self, items):
        self.items = items
        self.n_items = len(items)

    def reset(self, rng):
        self.rng = rng

    def select(self, asked, history, instrument, revealed):
        raise NotImplementedError

    def _remaining(self, asked):
        return [i for i in range(self.n_items) if i not in asked]


class RandomPolicy(BasePolicy):
    name = 'random'

    def select(self, asked, history, instrument, revealed):
        rem = self._remaining(asked)
        return int(self.rng.choice(rem)) if rem else None


class PopularityPolicy(BasePolicy):
    """Asks entities in descending population rating frequency (fixed order)."""
    name = 'popularity'

    def __init__(self, items, p_rated):
        super().__init__(items)
        self.order = list(np.argsort(-np.asarray(p_rated)))

    def select(self, asked, history, instrument, revealed):
        for e in self.order:
            if e not in asked:
                return int(e)
        return None


class GreedyInfoGainPolicy(BasePolicy):
    """One-step expected entropy reduction of the instrument's movie predictions.

    EIG(e) = p_rated(e) * [ p_l(e) * (H - H(revealed+{e:liked}))
                          + (1-p_l(e)) * (H - H(revealed+{e:disliked})) ]
    with p_l(e) = instrument's current full-slate score for e, and
    p_rated(e) the population frequency of e being rated (train users only).
    The strongest non-learned baseline.
    """
    name = 'greedy_infogain'

    def __init__(self, items, p_rated):
        super().__init__(items)
        self.p_rated = np.asarray(p_rated)

    @staticmethod
    def _entropy(p):
        p = np.clip(p, 1e-7, 1 - 1e-7)
        return float(-(p * np.log(p) + (1 - p) * np.log(1 - p)).sum())

    def select(self, asked, history, instrument, revealed):
        rem = self._remaining(asked)
        if not rem:
            return None
        cur_full = instrument.predict_full(revealed)
        h_now = self._entropy(cur_full[:instrument.n_movies])

        hyps, meta = [], []
        for e in rem:
            hyps.append(revealed + [(e, 1.0)])
            hyps.append(revealed + [(e, 0.0)])
            meta.append(e)
        preds = instrument.predict_batch(hyps)  # [2*len(rem), n_movies]

        best_e, best_eig = None, -np.inf
        for j, e in enumerate(meta):
            h_l = self._entropy(preds[2 * j])
            h_d = self._entropy(preds[2 * j + 1])
            p_l = float(cur_full[e])
            eig = self.p_rated[e] * (p_l * (h_now - h_l) + (1 - p_l) * (h_now - h_d))
            if eig > best_eig:
                best_eig, best_e = eig, e
        return int(best_e)


# ---------------------------------------------------------------------------
# Prompt-only LLM policies
# ---------------------------------------------------------------------------

VANILLA_SYSTEM = """You are a movie recommendation assistant interviewing a user to learn their taste.
At each step you choose ONE entity to ask about (whether they like it), from a fixed menu.
Reply with the exact entity name from the menu, and nothing else."""

STRATEGIST_SYSTEM = """You are an expert preference-elicitation strategist for a movie recommender.
Your goal: in as few questions as possible, learn enough about this user's taste to predict
their ratings for 100 popular movies. At each step you choose ONE entity from a fixed menu.

Think strategically before answering:
- Broad attributes (genres) split the space fastest early on.
- Follow up on what you have learned: refine within liked areas, confirm dislikes.
- Questions the user can't answer ('unknown') are wasted turns; popular entities are safer.
- Avoid redundant questions whose answer you can already infer.

First think step by step in at most 50 words, then output the chosen entity name on the
final line in the exact form: CHOICE: <entity name>"""

GATE_SYSTEM = """You are conducting a free-form preference interview, but your questions must be
realised by selecting ONE entity from a fixed menu (the system asks 'do you like <entity>?').
Run the interview the way an expert interviewer would: open with broad probes, test edge
cases of what you believe, and probe contrasts (if they like X, check a nearby Y that
would distinguish two hypotheses about their taste).
Output the chosen entity name on the final line as: CHOICE: <entity name>"""


class LLMPolicy(BasePolicy):
    """Menu-constrained LLM questioner. Parse failures fall back to random
    (counted, reported)."""

    STYLES = {
        'vanilla': VANILLA_SYSTEM,
        'strategist': STRATEGIST_SYSTEM,
        'gate': GATE_SYSTEM,
    }

    # Persistent response cache: identical (model, style, prompt) calls are
    # answered from disk -- API spend is never repeated, and parser changes
    # can be replayed over cached responses offline.
    import threading as _t2
    _cache_lock = _t2.Lock()
    _cache = None
    CACHE_PATH = 'C:/dev/phd/casper/experiments/paper1/llm_cache.jsonl'

    @classmethod
    def _cache_get(cls, key):
        with cls._cache_lock:
            if cls._cache is None:
                cls._cache = {}
                import os
                if os.path.exists(cls.CACHE_PATH):
                    for line in open(cls.CACHE_PATH, encoding='utf-8'):
                        try:
                            rec = json.loads(line)
                            cls._cache[rec['k']] = rec['v']
                        except Exception:
                            pass
            return cls._cache.get(key)

    @classmethod
    def _cache_put(cls, key, value):
        with cls._cache_lock:
            cls._cache[key] = value
            with open(cls.CACHE_PATH, 'a', encoding='utf-8') as f:
                f.write(json.dumps({'k': key, 'v': value}) + '\n')

    # Class-level throttle shared across threads: keeps sustained demand
    # under the org TPM limit (200k TPM / ~1.6k tokens per call ~= 2 calls/s;
    # 0.55s spacing leaves headroom for retries and output tokens).
    import threading as _threading
    _rate_lock = _threading.Lock()
    _next_slot = [0.0]
    MIN_CALL_INTERVAL = 0.55

    @classmethod
    def _throttle(cls):
        import time as _time
        with cls._rate_lock:
            now = _time.time()
            slot = max(now, cls._next_slot[0])
            cls._next_slot[0] = slot + cls.MIN_CALL_INTERVAL
        wait = slot - now
        if wait > 0:
            _time.sleep(wait)

    def __init__(self, items, style='vanilla', model='gpt-4o-mini'):
        super().__init__(items)
        self.style = style
        self.model = model
        self.name = f'llm_{style}_{model.replace("/", "-")}'
        self.parse_failures = 0
        self.calls = 0
        self.transcript = []   # raw model outputs + parse outcomes
        from dotenv import load_dotenv
        load_dotenv(Path('C:/dev/phd/casper/.env'))
        self.provider = 'anthropic' if model.startswith('claude') else 'openai'
        if self.provider == 'anthropic':
            import anthropic
            self.client = anthropic.Anthropic()
        else:
            from openai import OpenAI
            self.client = OpenAI()
        # GPT-5 family: max_completion_tokens instead of max_tokens, and no
        # temperature parameter
        self.gpt5_family = model.startswith(('gpt-5', 'o'))
        self._name_to_idx = {it[2].lower(): i for i, it in enumerate(items)}

    def _menu(self, asked):
        lines = []
        for i, (itype, _, iname) in enumerate(self.items):
            if i not in asked:
                lines.append(f"- {iname} ({itype})")
        return '\n'.join(lines)

    def _history_text(self, history):
        if not history:
            return "(no questions asked yet)"
        return '\n'.join(f"Q: do you like {self.items[q][2]}? A: {a}"
                         for q, a in history)

    def select(self, asked, history, instrument, revealed):
        rem = self._remaining(asked)
        if not rem:
            return None
        user_msg = (f"Conversation so far:\n{self._history_text(history)}\n\n"
                    f"Menu of entities you may ask about next:\n{self._menu(asked)}\n\n"
                    f"Choose the single most informative next question.")
        self.calls += 1
        import hashlib
        cache_key = hashlib.sha256(
            f"{self.model}|{self.style}|{user_msg}".encode()).hexdigest()
        text = self._cache_get(cache_key)
        cached = text is not None
        for attempt in range(7):
            if text is not None:
                break
            try:
                self._throttle()
                text = self._call(user_msg)
                self._cache_put(cache_key, text)
                break
            except Exception as e:
                msg = str(e).lower()
                transient = any(k in msg for k in
                                ('rate limit', 'rate_limit', '429', 'overloaded',
                                 'timeout', 'timed out', '503', '502',
                                 'connection', 'server error', '500'))
                if transient and attempt < 6:
                    import time as _time
                    _time.sleep(2 ** attempt)
                    continue
                print(f"  [{self.name}] API error (attempt {attempt + 1}): {e}")
                break
        if text is None:
            self.parse_failures += 1
            return int(self.rng.choice(rem))

        m = re.search(r'CHOICE:\s*(.+)', text)
        cand = (m.group(1) if m else text.splitlines()[-1]).strip()
        cand = re.sub(r'\((movie|genre|actor|director)\)\s*$', '', cand).strip()
        cand = cand.strip('"\'' ).lower()

        if cand in self._name_to_idx and self._name_to_idx[cand] not in asked:
            return self._name_to_idx[cand]
        matches = [i for i in rem if cand and cand in self.items[i][2].lower()]
        if len(matches) == 1:
            return matches[0]
        self.parse_failures += 1
        return int(self.rng.choice(rem))

    def _call(self, user_msg):
        if self.provider == 'anthropic':
            resp = self.client.messages.create(
                model=self.model,
                system=self.STYLES[self.style],
                messages=[{'role': 'user', 'content': user_msg}],
                max_tokens=150,
            )
            return resp.content[0].text.strip()
        if self.gpt5_family:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[{'role': 'system', 'content': self.STYLES[self.style]},
                          {'role': 'user', 'content': user_msg}],
                max_completion_tokens=400,
            )
            return resp.choices[0].message.content.strip()
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[{'role': 'system', 'content': self.STYLES[self.style]},
                      {'role': 'user', 'content': user_msg}],
            max_tokens=150,
            temperature=0.0,
        )
        return resp.choices[0].message.content.strip()


# ---------------------------------------------------------------------------
# Learned bot-play policy (IJCNN 2024 method, retrained on this testbed)
# ---------------------------------------------------------------------------

class BotPlayPolicy(BasePolicy):
    """REINFORCE policy over the slate (Makarova et al., IJCNN 2024 design):
    state = current revealed one-hots, reward = instrument BCE-loss reduction.
    Loaded from a checkpoint produced by train_botplay_policy.py."""
    name = 'botplay_rl'

    def __init__(self, items, checkpoint_path):
        super().__init__(items)
        import torch
        import torch.nn as nn
        self.torch = torch
        ckpt = torch.load(checkpoint_path, weights_only=False)
        self.net = nn.Sequential(
            nn.Linear(self.n_items * 3, 256), nn.ReLU(),
            nn.Linear(256, self.n_items),
        )
        self.net.load_state_dict(ckpt['policy_state_dict'])
        self.net.eval()

    def select(self, asked, history, instrument, revealed):
        rem = self._remaining(asked)
        if not rem:
            return None
        state = np.zeros((self.n_items, 3), dtype=np.float32)
        state[:, 2] = 1
        for idx, pol in revealed:
            state[idx, 2] = 0
            state[idx, 1 if pol >= 0.5 else 0] = 1
        with self.torch.no_grad():
            logits = self.net(self.torch.from_numpy(state.flatten()).unsqueeze(0))[0]
        mask = self.torch.full((self.n_items,), float('-inf'))
        mask[rem] = 0.0
        return int(self.torch.argmax(logits + mask).item())


# ---------------------------------------------------------------------------
# Population statistics (train users only -- no leakage)
# ---------------------------------------------------------------------------

def compute_p_rated(profiles, user_ids):
    """Fraction of users with a non-nan label per entity."""
    vecs = np.stack([profiles[u] for u in user_ids if u in profiles])
    return (~np.isnan(vecs)).mean(axis=0)


class BotPlayV2Policy(BasePolicy):
    """Bot-play v2 (return-to-go, entropy-regularised, belief-state input)."""
    name = 'botplay_rl_v2'

    def __init__(self, items, checkpoint_path):
        super().__init__(items)
        import torch
        import torch.nn as nn
        self.torch = torch
        ckpt = torch.load(checkpoint_path, weights_only=False)
        n = self.n_items
        self.net = nn.Sequential(
            nn.Linear(n * 3 + n, 512), nn.ReLU(),
            nn.Linear(512, 256), nn.ReLU(),
            nn.Linear(256, n),
        )
        sd = {k.replace('net.', '', 1): v
              for k, v in ckpt['policy_state_dict'].items()}
        self.net.load_state_dict(sd)
        self.net.eval()

    def select(self, asked, history, instrument, revealed):
        rem = self._remaining(asked)
        if not rem:
            return None
        state = np.zeros((self.n_items, 3), dtype=np.float32)
        state[:, 2] = 1
        for idx, pol in revealed:
            state[idx, 2] = 0
            state[idx, 1 if pol >= 0.5 else 0] = 1
        beliefs = instrument.predict_full(revealed).astype(np.float32)
        x = np.concatenate([state.flatten(), beliefs])
        with self.torch.no_grad():
            logits = self.net(self.torch.from_numpy(x).unsqueeze(0))[0]
        mask = self.torch.full((self.n_items,), float('-inf'))
        mask[rem] = 0.0
        return int(self.torch.argmax(logits + mask).item())


# ---------------------------------------------------------------------------
# Lineage + Bayesian baselines (Paper 1 / B2, B5)
# ---------------------------------------------------------------------------

class SCPREntropyPolicy(BasePolicy):
    """SCPR-style weighted-entropy attribute chooser (Lei et al., KDD 2020),
    adapted to the testbed: ask the unasked entity whose own predicted
    rating is most uncertain under the instrument's current beliefs,
    weighted by population answerability. Unlike greedy info-gain, no
    lookahead -- this is the lineage's myopic heuristic."""
    name = 'scpr_entropy'

    def __init__(self, items, p_rated):
        super().__init__(items)
        self.p_rated = np.asarray(p_rated)

    def select(self, asked, history, instrument, revealed):
        rem = self._remaining(asked)
        if not rem:
            return None
        p = instrument.predict_full(revealed)
        p = np.clip(p, 1e-7, 1 - 1e-7)
        ent = -(p * np.log(p) + (1 - p) * np.log(1 - p))
        scores = self.p_rated * ent
        scores_masked = np.full(self.n_items, -np.inf)
        scores_masked[rem] = scores[rem]
        return int(np.argmax(scores_masked))


class ThompsonPolicy(BasePolicy):
    """PEBOL-style Bayesian elicitation (Austin et al., 2024): per-entity
    Beta beliefs over P(liked), initialised from the instrument prior,
    updated only by the user's answers; Thompson sampling picks the next
    query. A pure decision-theoretic baseline with no instrument lookahead."""
    name = 'thompson'

    PRIOR_STRENGTH = 4.0

    def __init__(self, items):
        super().__init__(items)

    def reset(self, rng):
        self.rng = rng
        self.alpha = None
        self.beta = None

    def select(self, asked, history, instrument, revealed):
        rem = self._remaining(asked)
        if not rem:
            return None
        if self.alpha is None:
            p0 = instrument.predict_full([])
            self.alpha = 1.0 + self.PRIOR_STRENGTH * p0
            self.beta = 1.0 + self.PRIOR_STRENGTH * (1.0 - p0)
        # update from the latest answer
        if history:
            q, a = history[-1]
            if a == 'liked':
                self.alpha[q] += 2.0
            elif a == 'disliked':
                self.beta[q] += 2.0
        theta = self.rng.beta(self.alpha, self.beta)
        theta_masked = np.full(self.n_items, -np.inf)
        theta_masked[rem] = theta[rem]
        return int(np.argmax(theta_masked))


class DQNPolicy(BasePolicy):
    """UNICORN-style value-based policy (Deng et al., SIGIR 2021 family):
    dueling DQN over the entity slate, trained by train_dqn_policy.py."""
    name = 'dqn'

    def __init__(self, items, checkpoint_path):
        super().__init__(items)
        import torch
        import torch.nn as nn
        self.torch = torch
        ckpt = torch.load(checkpoint_path, weights_only=False)
        n = self.n_items
        state_dim = n * 3 + n

        class Dueling(nn.Module):
            def __init__(self):
                super().__init__()
                self.shared = nn.Sequential(nn.Linear(state_dim, 512), nn.ReLU(),
                                            nn.Linear(512, 256), nn.ReLU())
                self.value = nn.Linear(256, 1)
                self.adv = nn.Linear(256, n)

            def forward(self, x):
                h = self.shared(x)
                a = self.adv(h)
                return self.value(h) + a - a.mean(dim=-1, keepdim=True)

        self.net = Dueling()
        self.net.load_state_dict(ckpt['q_state_dict'])
        self.net.eval()

    def _state(self, revealed, instrument):
        s = np.zeros((self.n_items, 3), dtype=np.float32)
        s[:, 2] = 1
        for idx, pol in revealed:
            s[idx, 2] = 0
            s[idx, 1 if pol >= 0.5 else 0] = 1
        beliefs = instrument.predict_full(revealed).astype(np.float32)
        return np.concatenate([s.flatten(), beliefs])

    def select(self, asked, history, instrument, revealed):
        rem = self._remaining(asked)
        if not rem:
            return None
        x = self.torch.from_numpy(self._state(revealed, instrument)).unsqueeze(0)
        with self.torch.no_grad():
            q = self.net(x)[0]
        mask = self.torch.full((self.n_items,), float('-inf'))
        mask[rem] = 0.0
        return int(self.torch.argmax(q + mask).item())


class PPOPolicy(BasePolicy):
    """PPO policy trained by scripts/paper2/train_discrete_ppo.py."""
    name = 'ppo'

    def __init__(self, items, checkpoint_path):
        super().__init__(items)
        import torch
        import torch.nn as nn
        self.torch = torch
        ckpt = torch.load(checkpoint_path, weights_only=False)
        n = self.n_items
        state_dim = n * 3 + n

        class AC(nn.Module):
            def __init__(self):
                super().__init__()
                self.shared = nn.Sequential(nn.Linear(state_dim, 512), nn.ReLU(),
                                            nn.Linear(512, 256), nn.ReLU())
                self.pi = nn.Linear(256, n)
                self.v = nn.Linear(256, 1)

        self.net = AC()
        self.net.load_state_dict(ckpt['net_state_dict'])
        self.net.eval()

    def select(self, asked, history, instrument, revealed):
        rem = self._remaining(asked)
        if not rem:
            return None
        s = np.zeros((self.n_items, 3), dtype=np.float32)
        s[:, 2] = 1
        for idx, pol in revealed:
            s[idx, 2] = 0
            s[idx, 1 if pol >= 0.5 else 0] = 1
        beliefs = instrument.predict_full(revealed).astype(np.float32)
        x = self.torch.from_numpy(
            np.concatenate([s.flatten(), beliefs])).unsqueeze(0)
        with self.torch.no_grad():
            logits = self.net.pi(self.net.shared(x))[0]
        mask = self.torch.full((self.n_items,), float('-inf'))
        mask[rem] = 0.0
        return int(self.torch.argmax(logits + mask).item())


def _build_policy_state(n_items, revealed, belief_instrument, dual):
    s = np.zeros((n_items, 3), dtype=np.float32)
    s[:, 2] = 1
    for idx, pol in revealed:
        s[idx, 2] = 0
        s[idx, 1 if pol >= 0.5 else 0] = 1
    parts = [s.flatten(),
             belief_instrument.predict_full(revealed).astype(np.float32)]
    if dual:
        parts.append(belief_instrument.predict_rated(revealed).astype(np.float32))
    return np.concatenate(parts)


class NetPolicy(BasePolicy):
    """Generic adapter for PPO/REINFORCE checkpoints carrying state metadata.
    belief_instrument: decision-aid model (defaults to the harness
    instrument); kind: 'ppo' (ActorCritic dict) or 'reinforce'."""

    def __init__(self, items, checkpoint_path, kind, name,
                 belief_instrument=None):
        super().__init__(items)
        import torch
        import torch.nn as nn
        self.torch = torch
        self.name = name
        ckpt = torch.load(checkpoint_path, weights_only=False)
        self.dual = ckpt.get('state_mode') == 'dual'
        state_dim = ckpt.get('state_dim', self.n_items * 4)
        self.belief_instrument = belief_instrument
        n = self.n_items
        if kind == 'ppo':
            class AC(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.shared = nn.Sequential(
                        nn.Linear(state_dim, 512), nn.ReLU(),
                        nn.Linear(512, 256), nn.ReLU())
                    self.pi = nn.Linear(256, n)
                    self.v = nn.Linear(256, 1)
            self.net = AC()
            self.net.load_state_dict(ckpt['net_state_dict'])
            self._logits = lambda x: self.net.pi(self.net.shared(x))
        else:
            self.net = nn.Sequential(nn.Linear(state_dim, 512), nn.ReLU(),
                                     nn.Linear(512, 256), nn.ReLU(),
                                     nn.Linear(256, n))
            self.net.load_state_dict(ckpt['policy_state_dict'])
            self._logits = lambda x: self.net(x)
        self.net.eval()

    def select(self, asked, history, instrument, revealed):
        rem = self._remaining(asked)
        if not rem:
            return None
        bi = self.belief_instrument or instrument
        x = self.torch.from_numpy(
            _build_policy_state(self.n_items, revealed, bi, self.dual)
        ).unsqueeze(0)
        with self.torch.no_grad():
            logits = self._logits(x)[0]
        mask = self.torch.full((self.n_items,), float('-inf'))
        mask[rem] = 0.0
        return int(self.torch.argmax(logits + mask).item())
