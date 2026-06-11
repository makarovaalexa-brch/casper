"""
Paper 1/2a: improved RL on the synthetic indicator world.

Diagnosis of reinforce_v2's below-random score (0.619 final vs random
0.657, hand-built static 0.694): (a) NO model selection -- the final
post-anneal policy was used as-is; (b) stochastic-train/argmax-eval
mismatch; (c) 12k episodes with no exploration pressure toward the
indicator (1 action of 97 with zero immediate reward).

This script fields the strongest budget-matched versions:
  REINFORCE-v3: 40k episodes, count-based exploration bonus
    (0.1/sqrt(N(a)), annealed), slower entropy anneal, argmax
    validation every 2k episodes on held-out profiles, best checkpoint.
  PPO: clipped surrogate + GAE(0.95), entropy bonus, same budget --
    Paper 2a's opening question (can replay/batched RL find the
    two-step routing plan?) answered early.

Reuses the cached dual-head instrument and world from synthetic_sanity.
Run from casper root: poetry run python scripts/paper1/synthetic_rl_improved.py
Merges results into experiments/paper1/synthetic_sanity.json
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

import synthetic_sanity as W  # world, instrument, helpers

OUT = Path('C:/dev/phd/casper/experiments/paper1/synthetic_sanity.json')

SEED = 42
N_TURNS = W.N_TURNS
N_MOVIES = W.N_MOVIES
GAMMA = 0.97
torch.manual_seed(SEED)


def load_instrument():
    cache = Path('experiments/paper1/synthetic_instrument_v6.pt')
    m = W.DualHeadSetEncoder(N_MOVIES)
    m.load_state_dict(torch.load(cache, weights_only=False))
    m.eval()
    return W.DualWrapper(m, N_MOVIES)


def make_state(revealed, liked_b, rated_b):
    s = np.zeros((N_MOVIES, 3), dtype=np.float32)
    s[:, 2] = 1
    for idx, pol in revealed:
        s[idx, 2] = 0
        s[idx, 1 if pol >= 0.5 else 0] = 1
    return np.concatenate([s.flatten(), liked_b.astype(np.float32),
                           rated_b.astype(np.float32)])


STATE_DIM = N_MOVIES * 3 + 2 * N_MOVIES


def argmax_eval(net, instrument, profiles):
    aucs, finals = [], []
    for profile in profiles:
        revealed, asked = [], set()
        lb = instrument.predict(revealed)
        rb = instrument.predict_rated(revealed)
        accs = [W.accuracy(lb, profile)]
        for _ in range(N_TURNS):
            x = torch.from_numpy(make_state(revealed, lb, rb)).unsqueeze(0)
            with torch.no_grad():
                out = net(x)
                logits = out[0] if isinstance(out, tuple) else out
                logits = logits[0] if logits.dim() == 2 else logits
            mask = torch.full((N_MOVIES,), float('-inf'))
            rem = [i for i in range(N_MOVIES) if i not in asked]
            mask[rem] = 0.0
            a = int(torch.argmax(logits + mask).item())
            asked.add(a)
            v = profile[a]
            if not np.isnan(v):
                revealed.append((a, float(v)))
            lb = instrument.predict(revealed)
            rb = instrument.predict_rated(revealed)
            accs.append(W.accuracy(lb, profile))
        aucs.append(float(np.mean(accs)))
        finals.append(accs[-1])
    return float(np.mean(aucs)), float(np.mean(finals))


def behavioral_summary(net, instrument, profiles):
    all_qs, all_ans = [], []
    for profile in profiles:
        revealed, asked = [], set()
        lb = instrument.predict(revealed)
        rb = instrument.predict_rated(revealed)
        qs, ans = [], []
        for _ in range(N_TURNS):
            x = torch.from_numpy(make_state(revealed, lb, rb)).unsqueeze(0)
            with torch.no_grad():
                out = net(x)
                logits = out[0] if isinstance(out, tuple) else out
                logits = logits[0] if logits.dim() == 2 else logits
            mask = torch.full((N_MOVIES,), float('-inf'))
            rem = [i for i in range(N_MOVIES) if i not in asked]
            mask[rem] = 0.0
            a = int(torch.argmax(logits + mask).item())
            asked.add(a)
            v = profile[a]
            ansr = 'unknown'
            if not np.isnan(v):
                revealed.append((a, float(v)))
                ansr = 'liked' if v >= 0.5 else 'disliked'
            qs.append(a)
            ans.append(ansr)
            lb = instrument.predict(revealed)
            rb = instrument.predict_rated(revealed)
        all_qs.append(qs)
        all_ans.append(ans)
    br = {}
    for qs, ans in zip(all_qs, all_ans):
        br.setdefault(ans[0], Counter())[qs[1]] += 1
    t2 = {a: int(c.most_common(1)[0][0]) for a, c in br.items()}
    return {
        'branches_on_first_answer': len(set(t2.values())) > 1,
        't2_by_answer': t2,
        'unique_questions': len(set(q for qs in all_qs for q in qs)),
        'asked_indicator_rate': float(np.mean(
            [W.INDICATOR in qs for qs in all_qs])),
    }


def train_reinforce_v3(instrument, train_profiles, val_profiles, rng):
    net = nn.Sequential(nn.Linear(STATE_DIM, 256), nn.ReLU(),
                        nn.Linear(256, 128), nn.ReLU(),
                        nn.Linear(128, N_MOVIES))
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    N_EP = 40000
    baselines = np.zeros(N_TURNS)
    counts = np.ones(N_MOVIES)
    best = (-1.0, None)
    t0 = time.time()
    for ep in range(N_EP):
        frac = ep / N_EP
        beta = 0.05 * (1 - frac) + 0.01 * frac
        bonus_scale = 0.10 * max(0.0, 1 - 2 * frac)  # explore first half
        profile = train_profiles[int(rng.integers(len(train_profiles)))]
        revealed, asked = [], set()
        lb = instrument.predict(revealed)
        rb = instrument.predict_rated(revealed)
        acc_prev = W.accuracy(lb, profile)
        lps, ents, rews = [], [], []
        for t in range(N_TURNS):
            x = torch.from_numpy(make_state(revealed, lb, rb)).unsqueeze(0)
            logits = net(x)[0]
            mask = torch.full((N_MOVIES,), float('-inf'))
            rem = [i for i in range(N_MOVIES) if i not in asked]
            mask[rem] = 0.0
            logp = torch.log_softmax(logits + mask, dim=0)
            probs = logp.exp()
            a = int(torch.multinomial(probs, 1).item())
            lps.append(logp[a])
            ents.append(-(probs * logp.clamp(min=-30)).sum())
            asked.add(a)
            v = profile[a]
            if not np.isnan(v):
                revealed.append((a, float(v)))
            lb = instrument.predict(revealed)
            rb = instrument.predict_rated(revealed)
            acc_now = W.accuracy(lb, profile)
            r = acc_now - acc_prev + bonus_scale / np.sqrt(counts[a])
            counts[a] += 1
            rews.append(r)
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
        torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
        opt.step()
        if (ep + 1) % 2000 == 0:
            auac, final = argmax_eval(net, instrument, val_profiles)
            mark = ''
            if auac > best[0]:
                best = (auac, {k: v.clone() for k, v in net.state_dict().items()})
                mark = ' *'
            print(f"  R3 ep {ep + 1}/{N_EP} val AUAC {auac:.4f} "
                  f"final {final:.4f}{mark} ({time.time() - t0:.0f}s)",
                  flush=True)
    net.load_state_dict(best[1])
    net.eval()
    return net


def train_ppo(instrument, train_profiles, val_profiles, rng):
    class AC(nn.Module):
        def __init__(self):
            super().__init__()
            self.shared = nn.Sequential(nn.Linear(STATE_DIM, 256), nn.ReLU(),
                                        nn.Linear(256, 128), nn.ReLU())
            self.pi = nn.Linear(128, N_MOVIES)
            self.v = nn.Linear(128, 1)

        def forward(self, x):
            h = self.shared(x)
            return self.pi(h), self.v(h).squeeze(-1)

    net = AC()
    opt = torch.optim.Adam(net.parameters(), lr=3e-4)
    N_EP = 40000
    BATCH_EPS = 32
    LAM, CLIP = 0.95, 0.2
    counts = np.ones(N_MOVIES)
    best = (-1.0, None)
    t0 = time.time()
    ep_count = 0
    while ep_count < N_EP:
        frac = ep_count / N_EP
        ent_coef = 0.03 * (1 - frac) + 0.005 * frac
        bonus_scale = 0.10 * max(0.0, 1 - 2 * frac)
        S, A, LP, R, D, V, MASKS = [], [], [], [], [], [], []
        for _ in range(BATCH_EPS):
            profile = train_profiles[int(rng.integers(len(train_profiles)))]
            revealed, asked = [], set()
            lb = instrument.predict(revealed)
            rb = instrument.predict_rated(revealed)
            acc_prev = W.accuracy(lb, profile)
            for t in range(N_TURNS):
                s = make_state(revealed, lb, rb)
                st = torch.from_numpy(s).unsqueeze(0)
                with torch.no_grad():
                    logits, v = net(st)
                mask = torch.full((N_MOVIES,), float('-inf'))
                rem = [i for i in range(N_MOVIES) if i not in asked]
                mask[rem] = 0.0
                dist = torch.distributions.Categorical(logits=logits[0] + mask)
                a = int(dist.sample().item())
                S.append(s)
                A.append(a)
                LP.append(float(dist.log_prob(torch.tensor(a)).item()))
                MASKS.append(mask.numpy())
                asked.add(a)
                val = profile[a]
                if not np.isnan(val):
                    revealed.append((a, float(val)))
                lb = instrument.predict(revealed)
                rb = instrument.predict_rated(revealed)
                acc_now = W.accuracy(lb, profile)
                R.append(acc_now - acc_prev + bonus_scale / np.sqrt(counts[a]))
                counts[a] += 1
                acc_prev = acc_now
                D.append(float(t == N_TURNS - 1))
                V.append(float(v.item()))
            ep_count += 1
        adv = np.zeros(len(R), dtype=np.float32)
        last = 0.0
        for i in reversed(range(len(R))):
            nv = 0.0 if D[i] else (V[i + 1] if i + 1 < len(V) else 0.0)
            delta = R[i] + GAMMA * nv - V[i]
            last = delta + GAMMA * LAM * (0.0 if D[i] else last)
            adv[i] = last
        ret = adv + np.array(V, dtype=np.float32)
        S_t = torch.from_numpy(np.stack(S))
        A_t = torch.tensor(A)
        LP_t = torch.tensor(LP)
        M_t = torch.from_numpy(np.stack(MASKS))
        adv_t = torch.from_numpy((adv - adv.mean()) / (adv.std() + 1e-8))
        ret_t = torch.from_numpy(ret)
        for _ in range(4):
            logits, v = net(S_t)
            dist = torch.distributions.Categorical(logits=logits + M_t)
            logp = dist.log_prob(A_t)
            ent = dist.entropy().mean()
            ratio = (logp - LP_t).exp()
            pg = -torch.min(ratio * adv_t,
                            ratio.clamp(1 - CLIP, 1 + CLIP) * adv_t).mean()
            loss = pg + 0.5 * ((v - ret_t) ** 2).mean() - ent_coef * ent
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
            opt.step()
        if ep_count % 2000 < BATCH_EPS:
            auac, final = argmax_eval(net, instrument, val_profiles)
            mark = ''
            if auac > best[0]:
                best = (auac, {k: v.clone() for k, v in net.state_dict().items()})
                mark = ' *'
            print(f"  PPO ep {ep_count}/{N_EP} val AUAC {auac:.4f} "
                  f"final {final:.4f}{mark} ({time.time() - t0:.0f}s)",
                  flush=True)
    net.load_state_dict(best[1])
    net.eval()
    return net


def main():
    rng = np.random.default_rng(SEED + 10)
    train_profiles = W.gen_users(W.N_TRAIN_USERS, np.random.default_rng(SEED))
    val_profiles = W.gen_users(200, np.random.default_rng(SEED + 5))
    eval_profiles = W.gen_users(W.N_EVAL_USERS, np.random.default_rng(SEED + 6))
    instrument = load_instrument()

    results = json.loads(OUT.read_text()) if OUT.exists() else {}

    print("Training REINFORCE-v3 (model selection + exploration bonus)...")
    r3 = train_reinforce_v3(instrument, train_profiles, val_profiles, rng)
    auac_list, finals = [], []
    for p in eval_profiles:
        a, f = argmax_eval(r3, instrument, [p])
        auac_list.append(a)
        finals.append(f)
    res = {'auac': float(np.mean(auac_list)),
           'auac_se': float(np.std(auac_list) / np.sqrt(len(auac_list))),
           'final_acc': float(np.mean(finals))}
    res.update(behavioral_summary(r3, instrument, eval_profiles))
    results['reinforce_v3'] = res
    print(f"reinforce_v3: AUAC={res['auac']:.4f} final={res['final_acc']:.4f} "
          f"indicator={res['asked_indicator_rate']:.0%} "
          f"branches={res['branches_on_first_answer']}")

    print("Training PPO...")
    ppo = train_ppo(instrument, train_profiles, val_profiles,
                    np.random.default_rng(SEED + 20))
    auac_list, finals = [], []
    for p in eval_profiles:
        a, f = argmax_eval(ppo, instrument, [p])
        auac_list.append(a)
        finals.append(f)
    res = {'auac': float(np.mean(auac_list)),
           'auac_se': float(np.std(auac_list) / np.sqrt(len(auac_list))),
           'final_acc': float(np.mean(finals))}
    res.update(behavioral_summary(ppo, instrument, eval_profiles))
    results['ppo'] = res
    print(f"ppo: AUAC={res['auac']:.4f} final={res['final_acc']:.4f} "
          f"indicator={res['asked_indicator_rate']:.0%} "
          f"branches={res['branches_on_first_answer']}")

    OUT.write_text(json.dumps(results, indent=2))
    print(f"Saved {OUT}")


if __name__ == '__main__':
    main()
