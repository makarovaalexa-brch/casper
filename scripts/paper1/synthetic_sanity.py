"""
Paper 1: synthetic clustered-user sanity check (ports the IJCNN 2024
synthetic experiment, extended to REQUIRE adaptivity).

World: 8 movie clusters x 12 movies. Users belong to one of two GROUPS
(clusters 0-3 or 4-7). A user rates only movies in their own group's
clusters (p=0.7 per movie, 5% label noise); each cluster has a single
liked/disliked polarity per user. Movies outside the group answer
'unknown'.

Consequences:
- An ADAPTIVE policy identifies the group with ~1 probe, then asks one
  movie per remaining own-group cluster: ~4-5 questions to ceiling.
- The best STATIC sequence must interleave both groups' clusters: at a
  budget of 6 turns it can cover at most ~3 clusters of the user's
  group. The adaptive-static gap is therefore structural, not noise.

Pipeline (all synthetic, minutes on CPU):
  1. generate world + users
  2. train a small set-encoder instrument on synthetic profiles
  3. evaluate: random | static-oracle (one per cluster, fixed) |
     greedy info-gain | REINFORCE-v2 policy (same recipe as bot-play v2)
  4. report AUAC + branching audit of the learned policy

Success criteria:
  S1: greedy (adaptive) >> static-oracle at T=6
  S2: learned policy branches on the first answer (audit) and
      approaches greedy; if instead it collapses to a playlist HERE,
      the collapse is algorithmic, not slate-induced.

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

from train_instrument_v5 import SetEncoderInstrument
from test_instrument_lib import SetInstrumentWrapper

OUT = Path('C:/dev/phd/casper/experiments/paper1/synthetic_sanity.json')

SEED = 42
N_CLUSTERS = 8
MOVIES_PER_CLUSTER = 12
N_MOVIES = N_CLUSTERS * MOVIES_PER_CLUSTER + 1  # +1 group-indicator entity
INDICATOR = N_MOVIES - 1                        # always answerable
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
    vec[INDICATOR] = float(g)  # group indicator: always answerable
    return vec


def gen_users(n, rng):
    return [make_user(rng) for _ in range(n)]


# ---------------------------------------------------------------------------
# Instrument
# ---------------------------------------------------------------------------

def train_instrument(train_profiles, val_profiles):
    model = SetEncoderInstrument(N_MOVIES, d_model=64, n_heads=4, n_layers=2)
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

    def masked_bce(pred, tgt):
        mask = torch.isnan(tgt)
        pred = torch.where(mask, torch.zeros_like(pred), pred)
        tgt = torch.where(mask, torch.zeros_like(tgt), tgt)
        per = nn.functional.binary_cross_entropy_with_logits(
            pred, tgt, reduction='none')
        per = torch.where(mask, torch.zeros_like(per), per)
        return per.sum() / (~mask).float().sum().clamp(min=1)

    best, best_state = float('inf'), None
    lrng = np.random.default_rng(SEED + 1)
    for epoch in range(25):
        model.train()
        for bi, bp, pad, tgt in batchify(train_profiles, 128, lrng):
            opt.zero_grad()
            loss = masked_bce(model(bi, bp, pad), tgt)
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            va = 0.0
            nb = 0
            for bi, bp, pad, tgt in batchify(val_profiles, 256, lrng):
                va += masked_bce(model(bi, bp, pad), tgt).item()
                nb += 1
            va /= max(nb, 1)
        if va < best:
            best, best_state = va, {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    model.eval()
    print(f"  instrument val loss {best:.4f}")
    return SetInstrumentWrapper(model, N_MOVIES, N_MOVIES, max_reveal=12)


# ---------------------------------------------------------------------------
# Policies
# ---------------------------------------------------------------------------

def accuracy(preds, profile):
    filt = ~np.isnan(profile)
    filt[INDICATOR] = False
    return float(np.mean((preds[filt] > 0.5) == profile[filt]))


def run_episode(select_fn, profile, instrument, n_turns, reset_fn=None):
    revealed, asked = [], set()
    if reset_fn:
        reset_fn()
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


def random_select(rng):
    def f(asked, revealed, instrument):
        rem = [i for i in range(N_MOVIES) if i not in asked]
        return int(rng.choice(rem))
    return f


def static_oracle_select():
    # optimal static interleave: alternate groups (A,B,A,B,...) so either
    # group's user gets ceil(T/2) own-cluster probes
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


def greedy_select():
    def entropy(p):
        p = np.clip(p, 1e-7, 1 - 1e-7)
        return float(-(p * np.log(p) + (1 - p) * np.log(1 - p)).sum())

    def f(asked, revealed, instrument):
        rem = [i for i in range(N_MOVIES) if i not in asked]
        cur = instrument.predict(revealed)
        h0 = entropy(cur)
        hyps = []
        for e in rem:
            hyps.append(revealed + [(e, 1.0)])
            hyps.append(revealed + [(e, 0.0)])
        preds = instrument.predict_batch(hyps)
        best_e, best_v = rem[0], -1e9
        for j, e in enumerate(rem):
            p_l = float(cur[e])
            # p(answerable) ~ 0.5*RATE_P population-wide; uniform, so omit
            eig = p_l * (h0 - entropy(preds[2 * j])) + \
                  (1 - p_l) * (h0 - entropy(preds[2 * j + 1]))
            if eig > best_v:
                best_v, best_e = eig, e
        return int(best_e)
    return f


def train_reinforce_v2(instrument, train_profiles, rng):
    """Same recipe as bot-play v2: return-to-go, entropy bonus, beliefs."""
    state_dim = N_MOVIES * 3 + N_MOVIES
    net = nn.Sequential(nn.Linear(state_dim, 256), nn.ReLU(),
                        nn.Linear(256, 128), nn.ReLU(),
                        nn.Linear(128, N_MOVIES))
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    GAMMA = 0.97
    N_EP = 12000
    baselines = np.zeros(N_TURNS)

    def state(revealed, beliefs):
        s = np.zeros((N_MOVIES, 3), dtype=np.float32)
        s[:, 2] = 1
        for idx, pol in revealed:
            s[idx, 2] = 0
            s[idx, 1 if pol >= 0.5 else 0] = 1
        return np.concatenate([s.flatten(), beliefs.astype(np.float32)])

    t0 = time.time()
    for ep in range(N_EP):
        frac = ep / N_EP
        beta = 0.05 + (0.005 - 0.05) * frac
        eps = 0.2 + (0.02 - 0.2) * frac
        profile = train_profiles[int(rng.integers(len(train_profiles)))]
        revealed, asked = [], set()
        beliefs = instrument.predict(revealed)
        acc_prev = accuracy(beliefs, profile)
        lps, ents, rews = [], [], []
        for t in range(N_TURNS):
            x = torch.from_numpy(state(revealed, beliefs)).unsqueeze(0)
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
            beliefs = instrument.predict(revealed)
            acc_now = accuracy(beliefs, profile)
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
        beliefs = instrument.predict(revealed)
        x = torch.from_numpy(state(revealed, beliefs)).unsqueeze(0)
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

    print("Training instrument...")
    instrument = train_instrument(train_profiles, val_profiles)

    print("Training REINFORCE-v2 policy...")
    rl_select = train_reinforce_v2(instrument, train_profiles,
                                   np.random.default_rng(SEED + 2))

    policies = {
        'random': random_select(np.random.default_rng(SEED + 3)),
        'static_oracle': static_oracle_select(),
        'greedy_infogain': greedy_select(),
        'reinforce_v2': rl_select,
    }

    results = {}
    for name, sel in policies.items():
        aucs, all_qs, all_ans = [], [], []
        for profile in eval_profiles:
            accs, qs, ans = run_episode(sel, profile, instrument, N_TURNS)
            aucs.append(float(np.mean(accs)))
            all_qs.append(qs)
            all_ans.append(ans)
        # branching audit: distinct turn-2 questions conditioned on turn-1 answer
        br = {}
        for qs, ans in zip(all_qs, all_ans):
            if len(qs) >= 2:
                br.setdefault(ans[0], Counter())[qs[1]] += 1
        t2_by_answer = {a: int(c.most_common(1)[0][0]) for a, c in br.items()}
        branches = len(set(t2_by_answer.values())) > 1
        t1 = Counter(qs[0] for qs in all_qs)
        results[name] = {
            'auac': float(np.mean(aucs)),
            'auac_se': float(np.std(aucs) / np.sqrt(len(aucs))),
            'final_acc_mean': None,
            't1_concentration': t1.most_common(1)[0][1] / len(all_qs),
            'unique_questions': len(set(q for qs in all_qs for q in qs)),
            't2_by_answer': {k: v for k, v in t2_by_answer.items()},
            'branches_on_first_answer': bool(branches),
        }
        print(f"{name:<16} AUAC={results[name]['auac']:.4f} "
              f"(se {results[name]['auac_se']:.4f}) "
              f"branches={branches} unique_qs={results[name]['unique_questions']}")

    results['_world'] = {
        'n_clusters': N_CLUSTERS, 'movies_per_cluster': MOVIES_PER_CLUSTER,
        'groups': 2, 'rate_p': RATE_P, 'noise_p': NOISE_P,
        'n_turns': N_TURNS, 'n_eval_users': N_EVAL_USERS,
    }
    OUT.write_text(json.dumps(results, indent=2))
    print(f"Saved {OUT}")


if __name__ == '__main__':
    main()
