"""
Complete the synthetic indicator-world matrix: add SCPR-entropy, Thompson,
and DQN (the baselines present on the real slates but missing on synthetic).

Reuses the cached dual-head synthetic instrument and the exact eval users
(seed+6) and dual state representation used by the existing synthetic RL
rows, so all synthetic policies are directly comparable.

Run: poetry run python scripts/paper1/synthetic_baselines.py
Merges into experiments/paper1/synthetic_sanity.json
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

import synthetic_sanity as W
import synthetic_rl_improved as R

OUT = Path('C:/dev/phd/casper/experiments/paper1/synthetic_sanity.json')
SEED = 42
rng = np.random.default_rng(SEED + 100)


def eval_selectfn(select_fn, instrument, profiles):
    aucs, finals, all_qs, all_ans = [], [], [], []
    for prof in profiles:
        accs, qs, ans = W.run_episode(select_fn, prof, instrument, W.N_TURNS)
        aucs.append(float(np.mean(accs)))
        finals.append(accs[-1])
        all_qs.append(qs)
        all_ans.append(ans)
    br = {}
    for qs, ans in zip(all_qs, all_ans):
        if len(qs) >= 2:
            br.setdefault(ans[0], Counter())[qs[1]] += 1
    t2 = {a: int(c.most_common(1)[0][0]) for a, c in br.items()}
    t1 = Counter(qs[0] for qs in all_qs)
    return {
        'auac': float(np.mean(aucs)),
        'auac_se': float(np.std(aucs) / np.sqrt(len(aucs))),
        'final_acc': float(np.mean(finals)),
        't1_concentration': t1.most_common(1)[0][1] / len(all_qs),
        'unique_questions': len(set(q for qs in all_qs for q in qs)),
        't2_by_answer': t2,
        'branches_on_first_answer': len(set(t2.values())) > 1,
        'asked_indicator_rate': float(np.mean([W.INDICATOR in qs for qs in all_qs])),
    }


def make_scpr(p_rated):
    def f(asked, revealed, instrument):
        rem = [i for i in range(W.N_MOVIES) if i not in asked]
        p = np.clip(instrument.predict(revealed), 1e-7, 1 - 1e-7)
        ent = -(p * np.log(p) + (1 - p) * np.log(1 - p))
        sc = np.full(W.N_MOVIES, -np.inf)
        sc[rem] = (p_rated * ent)[rem]
        return int(np.argmax(sc))
    return f


def make_thompson():
    S = 4.0

    def f(asked, revealed, instrument):
        rem = [i for i in range(W.N_MOVIES) if i not in asked]
        p0 = instrument.predict([])
        alpha = 1.0 + S * p0
        beta = 1.0 + S * (1.0 - p0)
        for e, pol in revealed:
            if pol >= 0.5:
                alpha[e] += 2.0
            else:
                beta[e] += 2.0
        theta = rng.beta(alpha, beta)
        tm = np.full(W.N_MOVIES, -np.inf)
        tm[rem] = theta[rem]
        return int(np.argmax(tm))
    return f


def train_dqn(instrument, train_profiles, val_profiles):
    n = W.N_MOVIES
    sd = R.STATE_DIM

    class Dueling(nn.Module):
        def __init__(self):
            super().__init__()
            self.sh = nn.Sequential(nn.Linear(sd, 256), nn.ReLU(),
                                    nn.Linear(256, 128), nn.ReLU())
            self.v = nn.Linear(128, 1)
            self.a = nn.Linear(128, n)

        def forward(self, x):
            h = self.sh(x)
            a = self.a(h)
            return self.v(h) + a - a.mean(-1, keepdim=True)

    q, qt = Dueling(), Dueling()
    qt.load_state_dict(q.state_dict())
    opt = torch.optim.Adam(q.parameters(), lr=5e-4)
    N_EP, GAMMA = 15000, 0.97
    cap = 100000
    SB = np.zeros((cap, sd), np.float32); AB = np.zeros(cap, np.int64)
    RB = np.zeros(cap, np.float32); S2 = np.zeros((cap, sd), np.float32)
    DB = np.zeros(cap, np.float32); M2 = np.zeros((cap, n), bool)
    ptr = size = steps = 0
    drng = np.random.default_rng(SEED + 7)
    best, best_state = -1.0, None
    t0 = time.time()

    def state(revealed):
        lb = instrument.predict(revealed)
        rb = instrument.predict_rated(revealed)
        return R.make_state(revealed, lb, rb)

    for ep in range(N_EP):
        eps = 0.05 + 0.45 * max(0, 1 - ep / 10000)
        prof = train_profiles[int(drng.integers(len(train_profiles)))]
        revealed, asked = [], set()
        s = state(revealed)
        for t in range(W.N_TURNS):
            rem = [i for i in range(n) if i not in asked]
            if drng.random() < eps:
                a = int(drng.choice(rem))
            else:
                with torch.no_grad():
                    qa = q(torch.from_numpy(s).unsqueeze(0))[0]
                mask = torch.full((n,), float('-inf')); mask[rem] = 0
                a = int(torch.argmax(qa + mask).item())
            ab = set(asked)
            asked.add(a)
            v = prof[a]
            acc_prev = W.accuracy(instrument.predict(revealed), prof)
            if not np.isnan(v):
                revealed.append((a, float(v)))
            s2 = state(revealed)
            r = W.accuracy(instrument.predict(revealed), prof) - acc_prev
            done = float(t == W.N_TURNS - 1)
            SB[ptr], AB[ptr], RB[ptr], S2[ptr], DB[ptr] = s, a, r, s2, done
            m = np.zeros(n, bool); m[list(ab | {a})] = True; M2[ptr] = m
            ptr = (ptr + 1) % cap; size = min(size + 1, cap); s = s2
            if size >= 1000:
                idx = drng.integers(0, size, 128)
                bs = torch.from_numpy(SB[idx]); ba = torch.from_numpy(AB[idx])
                br_ = torch.from_numpy(RB[idx]); bs2 = torch.from_numpy(S2[idx])
                bd = torch.from_numpy(DB[idx]); bm = torch.from_numpy(M2[idx])
                with torch.no_grad():
                    qo = q(bs2); qo[bm] = -1e9
                    astar = qo.argmax(1)
                    qtt = qt(bs2).gather(1, astar.unsqueeze(1)).squeeze(1)
                    y = br_ + GAMMA * (1 - bd) * qtt
                pred = q(bs).gather(1, ba.unsqueeze(1)).squeeze(1)
                loss = nn.functional.smooth_l1_loss(pred, y)
                opt.zero_grad(); loss.backward(); opt.step()
                steps += 1
                if steps % 1500 == 0:
                    qt.load_state_dict(q.state_dict())
        if (ep + 1) % 3000 == 0:
            # quick val
            def sel(asked, revealed, instrument):
                rem = [i for i in range(n) if i not in asked]
                with torch.no_grad():
                    qa = q(torch.from_numpy(state(revealed)).unsqueeze(0))[0]
                mask = torch.full((n,), float('-inf')); mask[rem] = 0
                return int(torch.argmax(qa + mask).item())
            va = np.mean([np.mean(W.run_episode(sel, p, instrument, W.N_TURNS)[0])
                         for p in val_profiles])
            if va > best:
                best = va
                best_state = {k: v.clone() for k, v in q.state_dict().items()}
            print(f"  DQN ep {ep+1}/{N_EP} val {va:.4f} ({time.time()-t0:.0f}s)",
                  flush=True)
    q.load_state_dict(best_state)
    q.eval()

    def select(asked, revealed, instrument):
        rem = [i for i in range(n) if i not in asked]
        with torch.no_grad():
            qa = q(torch.from_numpy(state(revealed)).unsqueeze(0))[0]
        mask = torch.full((n,), float('-inf')); mask[rem] = 0
        return int(torch.argmax(qa + mask).item())
    return select


def main():
    instrument = R.load_instrument()
    train_profiles = W.gen_users(W.N_TRAIN_USERS, np.random.default_rng(SEED))
    val_profiles = W.gen_users(200, np.random.default_rng(SEED + 5))
    eval_profiles = W.gen_users(W.N_EVAL_USERS, np.random.default_rng(SEED + 6))

    # p_rated from a train sample
    vecs = np.stack([train_profiles[i] for i in range(2000)])
    p_rated = (~np.isnan(vecs)).mean(0)

    results = json.loads(OUT.read_text()) if OUT.exists() else {}

    print("SCPR-entropy...")
    results['scpr_entropy'] = eval_selectfn(make_scpr(p_rated), instrument, eval_profiles)
    print("Thompson...")
    results['thompson'] = eval_selectfn(make_thompson(), instrument, eval_profiles)
    print("DQN (training)...")
    dqn_sel = train_dqn(instrument, train_profiles, val_profiles)
    results['dqn'] = eval_selectfn(dqn_sel, instrument, eval_profiles)

    OUT.write_text(json.dumps(results, indent=2))
    for k in ['scpr_entropy', 'thompson', 'dqn']:
        r = results[k]
        print(f"{k:<16} AUAC={r['auac']:.4f} final={r['final_acc']:.4f} "
              f"branches={r['branches_on_first_answer']} "
              f"indicator={r['asked_indicator_rate']:.0%}")
    print(f"Saved {OUT}")


if __name__ == '__main__':
    main()
