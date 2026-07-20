"""
Paper 1: synthetic hierarchical-world sanity check, v6 (dual-head).

WORLD (hierarchical, adaptivity structurally required): 8 movie clusters
x 12 movies, two GROUPS of 4 clusters; users rate only own-group movies
(p=0.7, 5% label noise; one liked/disliked polarity per cluster); plus a
group-INDICATOR entity that is always answerable. T=5 turns.
  - Adaptive optimum: ask indicator, then one movie per own-group
    cluster -> near-ceiling coverage.
  - Best static interleave: <=62% expected own-group coverage.

DISCOVERY THIS SCRIPT TESTS (v6): with a single-head instrument exposing
only P(liked), no policy could express answerability ROUTING and all
policies failed (~0.58 AUAC vs oracle ceiling 0.90). v6 gives the
instrument a dual head -- P(rated) and P(liked|rated); the rated-ness
targets are free (the nan mask) -- and exposes both belief vectors to
policies:
  - greedy_answerability: EIG weighted by the user-conditional
    P(answerable) from the rated head;
  - reinforce_v2: state = reveal one-hots ++ liked beliefs ++ rated
    beliefs.
Prediction: adaptive policies now separate from static playlists.

Run from casper root: poetry run python scripts/paper1/synthetic_sanity.py
Output: experiments/paper1/synthetic_sanity.json
"""

import sys
sys.path.insert(0, '.')
sys.path.insert(0, 'scripts/paper1')

import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

OUT = Path('C:/dev/phd/casper/experiments/paper1/synthetic_sanity.json')

SEED = 42
N_CLUSTERS = 8
MOVIES_PER_CLUSTER = 12
N_MOVIES = N_CLUSTERS * MOVIES_PER_CLUSTER + 1  # +1 group-indicator entity
INDICATOR = N_MOVIES - 1
GROUPS = {0: list(range(4)), 1: list(range(4, 8))}
RATE_P = 0.7
NOISE_P = 0.05
N_TRAIN_USERS = 6000
N_VAL_USERS = 300
N_EVAL_USERS = 300
N_TURNS = 5

rng = np.random.default_rng(SEED)
torch.manual_seed(SEED)
CLUSTER_OF = np.repeat(np.arange(N_CLUSTERS), MOVIES_PER_CLUSTER)


def make_user(rng):
    g = int(rng.integers(2))
    polarity = {c: float(rng.integers(2)) for c in GROUPS[g]}
    vec = np.full(N_MOVIES, np.nan, dtype=np.float32)
    for m in range(N_MOVIES - 1):
        c = CLUSTER_OF[m]
        if c in polarity and rng.random() < RATE_P:
            lab = polarity[c]
            if rng.random() < NOISE_P:
                lab = 1.0 - lab
            vec[m] = lab
    vec[INDICATOR] = float(g)
    return vec


def gen_users(n, rng):
    return [make_user(rng) for _ in range(n)]


# ---------------------------------------------------------------------------
# Dual-head instrument
# ---------------------------------------------------------------------------

class DualHeadSetEncoder(nn.Module):
    """Set encoder over revealed (entity, polarity) tokens; two heads:
    liked logits and rated-ness logits, both over the full slate."""

    def __init__(self, n_items, d_model=128, n_heads=4, n_layers=2):
        super().__init__()
        self.n_items = n_items
        self.item_emb = nn.Embedding(n_items, d_model)
        self.pol_emb = nn.Embedding(2, d_model)
        self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=4 * d_model,
            dropout=0.1, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.head_liked = nn.Sequential(
            nn.Linear(d_model, 2 * d_model), nn.ReLU(),
            nn.Linear(2 * d_model, n_items))
        self.head_rated = nn.Sequential(
            nn.Linear(d_model, 2 * d_model), nn.ReLU(),
            nn.Linear(2 * d_model, n_items))
        nn.init.normal_(self.cls, std=0.02)

    def forward(self, idx, pol, pad):
        b = idx.shape[0]
        tok = self.item_emb(idx) + self.pol_emb(pol)
        x = torch.cat([self.cls.expand(b, -1, -1), tok], dim=1)
        mask = torch.cat([torch.zeros(b, 1, dtype=torch.bool), pad], dim=1)
        h = self.encoder(x, src_key_padding_mask=mask)[:, 0]
        return self.head_liked(h), self.head_rated(h)


class DualWrapper:
    def __init__(self, model, n_items):
        self.model = model
        self.n_items = n_items
        self.n_movies = n_items

    def _forward(self, revealed_list):
        b = len(revealed_list)
        L = max(1, max((len(r) for r in revealed_list), default=1))
        idx = torch.zeros(b, L, dtype=torch.long)
        pol = torch.zeros(b, L, dtype=torch.long)
        pad = torch.ones(b, L, dtype=torch.bool)
        for i, revealed in enumerate(revealed_list):
            for j, (e, p) in enumerate(revealed):
                idx[i, j] = int(e)
                pol[i, j] = 1 if p >= 0.5 else 0
                pad[i, j] = False
        with torch.no_grad():
            lk, rt = self.model(idx, pol, pad)
        return torch.sigmoid(lk).numpy(), torch.sigmoid(rt).numpy()

    def predict(self, revealed):
        return self._forward([revealed])[0][0]

    def predict_rated(self, revealed):
        return self._forward([revealed])[1][0]

    def predict_batch(self, revealed_list, full=False):
        return self._forward(revealed_list)[0]


def train_instrument(train_profiles, val_profiles):
    model = DualHeadSetEncoder(N_MOVIES)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    def batchify(profiles, bs, rng):
        idxs = rng.permutation(len(profiles))
        for start in range(0, len(idxs), bs):
            chunk = [profiles[i] for i in idxs[start:start + bs]]
            L = 12
            bi = np.zeros((len(chunk), L), dtype=np.int64)
            bp = np.zeros((len(chunk), L), dtype=np.int64)
            pad = np.ones((len(chunk), L), dtype=bool)
            tgt = np.stack(chunk)
            for r, full in enumerate(chunk):
                rated = np.where(~np.isnan(full))[0]
                k = int(np.exp(rng.uniform(0, np.log(min(len(rated), L) + 1))))
                k = max(0 if rng.random() < 0.05 else 1, min(k, len(rated)))
                sel = rng.choice(rated, size=k, replace=False) if k else []
                for j, e in enumerate(sel):
                    bi[r, j] = e
                    bp[r, j] = int(full[e] >= 0.5)
                    pad[r, j] = False
            yield (torch.from_numpy(bi), torch.from_numpy(bp),
                   torch.from_numpy(pad), torch.from_numpy(tgt))

    def loss_fn(liked_logits, rated_logits, tgt):
        is_rated = ~torch.isnan(tgt)
        tgt0 = torch.where(is_rated, tgt, torch.zeros_like(tgt))
        per = nn.functional.binary_cross_entropy_with_logits(
            liked_logits, tgt0, reduction='none')
        liked_loss = (per * is_rated.float()).sum() / \
            is_rated.float().sum().clamp(min=1)
        rated_loss = nn.functional.binary_cross_entropy_with_logits(
            rated_logits, is_rated.float())
        return liked_loss + rated_loss

    best, best_state = float('inf'), None
    lrng = np.random.default_rng(SEED + 1)
    for epoch in range(60):
        model.train()
        for bi, bp, pad, tgt in batchify(train_profiles, 128, lrng):
            opt.zero_grad()
            lk, rt = model(bi, bp, pad)
            loss = loss_fn(lk, rt, tgt)
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            va, nb = 0.0, 0
            for bi, bp, pad, tgt in batchify(val_profiles, 256, lrng):
                lk, rt = model(bi, bp, pad)
                va += loss_fn(lk, rt, tgt).item()
                nb += 1
            va /= max(nb, 1)
        if va < best:
            best, best_state = va, {k: v.clone()
                                    for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    model.eval()
    print(f"  instrument val loss {best:.4f}")
    return DualWrapper(model, N_MOVIES)


# ---------------------------------------------------------------------------
# Policies
# ---------------------------------------------------------------------------

def accuracy(preds, profile):
    filt = ~np.isnan(profile)
    filt[INDICATOR] = False
    return float(np.mean((preds[filt] > 0.5) == profile[filt]))


def run_episode(select_fn, profile, instrument, n_turns):
    revealed, asked = [], set()
    accs = [accuracy(instrument.predict(revealed), profile)]
    qs, ans = [], []
    for _ in range(n_turns):
        e = select_fn(asked, revealed, instrument)
        asked.add(e)
        v = profile[e]
        a = 'unknown'
        if not np.isnan(v):
            revealed.append((e, float(v)))
            a = 'liked' if v >= 0.5 else 'disliked'
        qs.append(e)
        ans.append(a)
        accs.append(accuracy(instrument.predict(revealed), profile))
    return accs, qs, ans


def oracle_adaptive_select():
    def f(asked, revealed, instrument):
        if INDICATOR not in asked:
            return INDICATOR
        # group known from the revealed indicator answer
        g = None
        for e, pol in revealed:
            if e == INDICATOR:
                g = int(pol)
        if g is None:
            g = 0  # indicator was unanswerable (never happens)
        answered_clusters = {CLUSTER_OF[e] for e, _ in revealed if e != INDICATOR}
        for c in GROUPS[g]:
            if c not in answered_clusters:
                for m in range(N_MOVIES - 1):
                    if CLUSTER_OF[m] == c and m not in asked:
                        return m
        rem = [i for i in range(N_MOVIES) if i not in asked]
        return rem[0]
    return f


def random_select(rng):
    def f(asked, revealed, instrument):
        rem = [i for i in range(N_MOVIES) if i not in asked]
        return int(rng.choice(rem))
    return f


def static_oracle_select():
    order = []
    for k in range(4):
        order.append(GROUPS[0][k] * MOVIES_PER_CLUSTER)
        order.append(GROUPS[1][k] * MOVIES_PER_CLUSTER)

    def f(asked, revealed, instrument):
        for e in order:
            if e not in asked:
                return e
        return next(i for i in range(N_MOVIES) if i not in asked)
    return f


def _entropy(p):
    p = np.clip(p, 1e-7, 1 - 1e-7)
    return float(-(p * np.log(p) + (1 - p) * np.log(1 - p)).sum())


def greedy_select(answerability=False):
    def f(asked, revealed, instrument):
        rem = [i for i in range(N_MOVIES) if i not in asked]
        cur = instrument.predict(revealed)
        p_ans = instrument.predict_rated(revealed) if answerability else None
        h0 = _entropy(cur)
        hyps = []
        for e in rem:
            hyps.append(revealed + [(e, 1.0)])
            hyps.append(revealed + [(e, 0.0)])
        preds = instrument.predict_batch(hyps)
        best_e, best_v = rem[0], -1e9
        for j, e in enumerate(rem):
            p_l = float(cur[e])
            eig = p_l * (h0 - _entropy(preds[2 * j])) + \
                  (1 - p_l) * (h0 - _entropy(preds[2 * j + 1]))
            if answerability:
                eig *= float(p_ans[e])
            if eig > best_v:
                best_v, best_e = eig, e
        return int(best_e)
    return f


def train_reinforce_v2(instrument, train_profiles, rng):
    """Bot-play v2 recipe; state now includes BOTH belief vectors."""
    state_dim = N_MOVIES * 3 + 2 * N_MOVIES
    net = nn.Sequential(nn.Linear(state_dim, 256), nn.ReLU(),
                        nn.Linear(256, 128), nn.ReLU(),
                        nn.Linear(128, N_MOVIES))
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    GAMMA = 0.97
    N_EP = 12000
    baselines = np.zeros(N_TURNS)

    def state(revealed, liked_b, rated_b):
        s = np.zeros((N_MOVIES, 3), dtype=np.float32)
        s[:, 2] = 1
        for idx, pol in revealed:
            s[idx, 2] = 0
            s[idx, 1 if pol >= 0.5 else 0] = 1
        return np.concatenate([s.flatten(), liked_b.astype(np.float32),
                               rated_b.astype(np.float32)])

    t0 = time.time()
    for ep in range(N_EP):
        frac = ep / N_EP
        beta = 0.05 + (0.005 - 0.05) * frac
        eps = 0.2 + (0.02 - 0.2) * frac
        profile = train_profiles[int(rng.integers(len(train_profiles)))]
        revealed, asked = [], set()
        liked_b = instrument.predict(revealed)
        rated_b = instrument.predict_rated(revealed)
        acc_prev = accuracy(liked_b, profile)
        lps, ents, rews = [], [], []
        for t in range(N_TURNS):
            x = torch.from_numpy(state(revealed, liked_b, rated_b)).unsqueeze(0)
            logits = net(x)[0]
            mask = torch.full((N_MOVIES,), float('-inf'))
            rem = [i for i in range(N_MOVIES) if i not in asked]
            mask[rem] = 0.0
            logp = torch.log_softmax(logits + mask, dim=0)
            probs = logp.exp()
            a = int(rng.choice(rem)) if rng.random() < eps else \
                int(torch.multinomial(probs, 1).item())
            lps.append(logp[a])
            ents.append(-(probs * logp.clamp(min=-30)).sum())
            asked.add(a)
            v = profile[a]
            if not np.isnan(v):
                revealed.append((a, float(v)))
            liked_b = instrument.predict(revealed)
            rated_b = instrument.predict_rated(revealed)
            acc_now = accuracy(liked_b, profile)
            rews.append(acc_now - acc_prev)
            acc_prev = acc_now
        g = 0.0
        rets = np.zeros(N_TURNS)
        for t in reversed(range(N_TURNS)):
            g = rews[t] + GAMMA * g
            rets[t] = g
        terms = []
        for t in range(N_TURNS):
            adv = rets[t] - baselines[t]
            baselines[t] = 0.995 * baselines[t] + 0.005 * rets[t]
            terms.append(-lps[t] * adv - beta * ents[t])
        opt.zero_grad()
        torch.stack(terms).sum().backward()
        opt.step()
        if (ep + 1) % 3000 == 0:
            print(f"  RL ep {ep + 1}/{N_EP} ({time.time() - t0:.0f}s)", flush=True)

    net.eval()

    def f(asked, revealed, instrument):
        liked_b = instrument.predict(revealed)
        rated_b = instrument.predict_rated(revealed)
        x = torch.from_numpy(state(revealed, liked_b, rated_b)).unsqueeze(0)
        with torch.no_grad():
            logits = net(x)[0]
        mask = torch.full((N_MOVIES,), float('-inf'))
        rem = [i for i in range(N_MOVIES) if i not in asked]
        mask[rem] = 0.0
        return int(torch.argmax(logits + mask).item())
    return f


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("Generating synthetic world...")
    train_profiles = gen_users(N_TRAIN_USERS, rng)
    val_profiles = gen_users(N_VAL_USERS, rng)
    eval_profiles = gen_users(N_EVAL_USERS, rng)

    cache = Path('experiments/paper1/synthetic_instrument_v6.pt')
    if cache.exists():
        print("Loading cached dual-head instrument...")
        m = DualHeadSetEncoder(N_MOVIES)
        m.load_state_dict(torch.load(cache, weights_only=False))
        m.eval()
        instrument = DualWrapper(m, N_MOVIES)
    else:
        print("Training dual-head instrument...")
        instrument = train_instrument(train_profiles, val_profiles)
        torch.save(instrument.model.state_dict(), cache)

    # ceiling gate
    oracle_accs = []
    for profile in eval_profiles[:100]:
        g = int(profile[INDICATOR])
        reveals = [(INDICATOR, float(g))]
        for c in GROUPS[g]:
            ms = [m for m in range(N_MOVIES - 1)
                  if CLUSTER_OF[m] == c and not np.isnan(profile[m])]
            if ms:
                reveals.append((ms[0], float(profile[ms[0]])))
        oracle_accs.append(accuracy(instrument.predict(reveals), profile))
    oracle_acc = float(np.mean(oracle_accs))
    print(f"  ORACLE-REVEAL ceiling: {oracle_acc:.4f} "
          f"({'OK' if oracle_acc >= 0.85 else 'INSTRUMENT TOO WEAK'})")

    # answerability-belief gate: after revealing the indicator, the rated
    # head must separate own-group from other-group movies
    seps = []
    for profile in eval_profiles[:100]:
        g = int(profile[INDICATOR])
        ra = instrument.predict_rated([(INDICATOR, float(g))])
        own = [m for m in range(N_MOVIES - 1) if CLUSTER_OF[m] in GROUPS[g]]
        oth = [m for m in range(N_MOVIES - 1) if CLUSTER_OF[m] not in GROUPS[g]]
        seps.append(float(ra[own].mean() - ra[oth].mean()))
    sep = float(np.mean(seps))
    print(f"  ANSWERABILITY separation after indicator: {sep:+.4f} "
          f"({'OK' if sep > 0.2 else 'RATED HEAD NOT ROUTING'})")

    print("Training REINFORCE-v2 (dual beliefs)...")
    rl_select = train_reinforce_v2(instrument, train_profiles,
                                   np.random.default_rng(SEED + 2))

    policies = {
        'oracle_adaptive': oracle_adaptive_select(),
        'random': random_select(np.random.default_rng(SEED + 3)),
        'static_oracle': static_oracle_select(),
        'greedy_infogain': greedy_select(answerability=False),
        'greedy_answerability': greedy_select(answerability=True),
        'reinforce_v2': rl_select,
    }

    results = {}
    for name, sel in policies.items():
        aucs, finals, all_qs, all_ans = [], [], [], []
        for profile in eval_profiles:
            accs, qs, ans = run_episode(sel, profile, instrument, N_TURNS)
            aucs.append(float(np.mean(accs)))
            finals.append(accs[-1])
            all_qs.append(qs)
            all_ans.append(ans)
        br = {}
        for qs, ans in zip(all_qs, all_ans):
            if len(qs) >= 2:
                br.setdefault(ans[0], Counter())[qs[1]] += 1
        t2_by_answer = {a: int(c.most_common(1)[0][0]) for a, c in br.items()}
        t1 = Counter(qs[0] for qs in all_qs)
        asked_indicator = np.mean([INDICATOR in qs for qs in all_qs])
        results[name] = {
            'auac': float(np.mean(aucs)),
            'auac_se': float(np.std(aucs) / np.sqrt(len(aucs))),
            'final_acc': float(np.mean(finals)),
            't1_concentration': t1.most_common(1)[0][1] / len(all_qs),
            'unique_questions': len(set(q for qs in all_qs for q in qs)),
            't2_by_answer': t2_by_answer,
            'branches_on_first_answer': len(set(t2_by_answer.values())) > 1,
            'asked_indicator_rate': float(asked_indicator),
        }
        r = results[name]
        print(f"{name:<22} AUAC={r['auac']:.4f} final={r['final_acc']:.4f} "
              f"branches={r['branches_on_first_answer']} "
              f"indicator={r['asked_indicator_rate']:.0%} "
              f"uniq={r['unique_questions']}")

    results['_oracle_ceiling'] = oracle_acc
    results['_answerability_separation'] = sep
    results['_world'] = {
        'version': 'v6 dual-head', 'n_clusters': N_CLUSTERS,
        'movies_per_cluster': MOVIES_PER_CLUSTER, 'groups': 2,
        'rate_p': RATE_P, 'noise_p': NOISE_P,
        'n_turns': N_TURNS, 'n_eval_users': N_EVAL_USERS,
    }
    OUT.write_text(json.dumps(results, indent=2))
    print(f"Saved {OUT}")


if __name__ == '__main__':
    main()
