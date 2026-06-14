"""
METHOD (candidate A): non-myopic amortized Expected-Information-Gain policy.

Combines the three independently-recommended fixes for the PG-collapse:
  - equivariant per-item actor (inductive bias for routing; already shown to
    branch under BC),
  - DENSE target-EIG reward = reduction in the instrument's predictive entropy
    over the HELD-OUT (un-asked) target items. Computed from the instrument's
    own probabilities -> no label-sampling noise, far better SNR than the
    2%-signal held-out-BCE reward (see snr_diagnostic.py). This is the
    target-predictive EIG of Huang et al. 2024 / RL-BOED (Blau 2022).
  - group-relative (RLOO) baseline = leave-one-out mean over G rollouts of the
    same user -> cancels dominant between-user variance (Ahmadian 2024).
Non-myopic: gamma=1, return = total EIG over the horizon, so the value of
future questions propagates back (can exceed the MYOPIC greedy-infogain 0.716).

Usage: DATASET_NAME=ml_stratified poetry run python scripts/paper2/train_eig.py
"""
import os, sys, time
sys.path.insert(0, '.'); sys.path.insert(0, 'scripts/paper1'); sys.path.insert(0, 'scripts/paper2')
from collections import Counter
import numpy as np
import torch
import torch.nn as nn
from test_instrument_lib import load_instrument_by_name
from env import ElicitationEnv
from equivariant_actor import EquivariantActor

NAME = os.environ.get('DATASET_NAME', 'ml_stratified')
NPZ = {'ml_stratified': 'C:/dev/phd/casper/data/movielens/ml_stratified_profiles.npz',
       'yelp_multicity': 'C:/dev/phd/casper/data/yelp/yelp_multicity_profiles.npz'}[NAME]
N_TURNS = 15; SEED = 42
N_EPISODES = int(os.environ.get('CASPER_EPISODES', 9000))
G = int(os.environ.get('EIG_GROUP', 4))           # RLOO group size
ENTROPY_COEF = 0.01
N_TEST = 300
OUT = f'C:/dev/phd/casper/experiments/paper2/eig_{NAME}.pt'
torch.manual_seed(SEED); np.random.seed(SEED)


def target_entropy(meas, nm, excl):
    p = np.clip(meas[:nm].astype(np.float64), 1e-6, 1 - 1e-6)
    h = -(p * np.log(p) + (1 - p) * np.log(1 - p))
    if excl:
        m = np.ones(nm, bool)
        for q in excl:
            if q < nm: m[q] = False
        h = h[m]
    return float(h.sum())


def rollout(actor, env, prof, nm, rng, greedy=False):
    s = env.reset(prof)
    H_prev = target_entropy(env.meas, nm, env.asked)
    states, acts, logps, rews = [], [], [], []
    accs = [env.episode_accuracy()]
    for _ in range(N_TURNS):
        st = torch.from_numpy(s).unsqueeze(0)
        logits = actor(st)[0]
        mask = torch.full((env.n_items,), float('-inf'))
        rem = [i for i in range(env.n_items) if i not in env.asked]
        mask[rem] = 0.0
        logp_all = torch.log_softmax(logits + mask, dim=0)
        if greedy:
            a = int(torch.argmax(logp_all).item())
        else:
            a = int(torch.multinomial(logp_all.exp(), 1).item())
        states.append(s.copy()); acts.append(a); logps.append(logp_all[a])
        s, _, _ = env.step(a)
        H_now = target_entropy(env.meas, nm, env.asked)
        rews.append(H_prev - H_now); H_prev = H_now
        accs.append(env.episode_accuracy())
    return states, acts, logps, rews, float(np.nanmean(accs))


def evaluate(actor, env, profiles, uids, nm):
    actor.eval(); aucs = []; seqs = set(); br = {}
    with torch.no_grad():
        for uid in uids:
            s = env.reset(profiles[uid]); qs = []; ans = []; accs = [env.episode_accuracy()]
            for _ in range(N_TURNS):
                logits = actor(torch.from_numpy(s).unsqueeze(0))[0]
                mask = torch.full((env.n_items,), float('-inf'))
                rem = [i for i in range(env.n_items) if i not in env.asked]; mask[rem] = 0.0
                a = int(torch.argmax(logits + mask).item()); qs.append(a)
                v = profiles[uid][a]; ans.append('unknown' if np.isnan(v) else ('liked' if v >= .5 else 'disliked'))
                s, _, _ = env.step(a); accs.append(env.episode_accuracy())
            aucs.append(np.nanmean(accs)); seqs.add(tuple(qs))
            if len(qs) >= 2: br.setdefault(ans[0], Counter())[qs[1]] += 1
    branch = len({c.most_common(1)[0][0] for c in br.values()}) > 1 if br else False
    return float(np.mean(aucs)), len(seqs), branch


def main():
    w, _ = load_instrument_by_name(f'instrument_{NAME}')
    d = np.load(NPZ, allow_pickle=True)
    train, test = d['train'], d['test']
    nm = int(d['n_targets']); n_items = train.shape[1]
    belief, _ = load_instrument_by_name(f'instrument_{NAME}')
    env = ElicitationEnv(w, n_items, N_TURNS, belief_instrument=belief,
                         state_mode='dual', heldout=True)
    actor = EquivariantActor(n_items, 'dual')
    opt = torch.optim.Adam(actor.parameters(), lr=5e-4, weight_decay=1e-5)
    rng = np.random.default_rng(SEED)
    te_prof = {i: test[i] for i in range(min(N_TEST, len(test)))}
    val_uids = list(range(min(100, len(test))))
    print(f"{NAME}: EIG-RLOO, equivariant actor params={sum(p.numel() for p in actor.parameters()):,} "
          f"G={G} episodes={N_EPISODES}", flush=True)
    best = -1; t0 = time.time()
    for ep in range(0, N_EPISODES, G):
        uid = int(rng.integers(len(train))); prof = train[uid]
        group = [rollout(actor, env, prof, nm, rng) for _ in range(G)]
        returns = np.array([sum(g[3]) for g in group])
        loss = 0.0; ent = 0.0
        for gi, (states, acts, logps, rews, _) in enumerate(group):
            base = (returns.sum() - returns[gi]) / (G - 1)         # RLOO baseline
            adv = returns[gi] - base
            lp = torch.stack(logps)
            loss = loss - adv * lp.sum()
            ent = ent - (lp.exp() * lp).sum()
        loss = loss / G - ENTROPY_COEF * ent / G
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(actor.parameters(), 5.0); opt.step()
        if (ep + G) % 1000 < G:
            auac, seqs, branch = evaluate(actor, env, te_prof, val_uids, nm)
            mark = ''
            if auac > best:
                best = auac; mark = ' *'
                torch.save({'actor_state_dict': actor.state_dict(), 'arch': 'equivariant',
                            'state_mode': 'dual', 'n_items': n_items,
                            'auac_heldout': auac, 'episodes': ep + G}, OUT)
            print(f"  ep{ep+G:5d} val AUAC={auac:.4f} seqs={seqs}/{len(val_uids)} "
                  f"branch={branch} ret={returns.mean():.3f} ({time.time()-t0:.0f}s){mark}", flush=True)
    # final full-test eval of best
    ck = torch.load(OUT); actor.load_state_dict(ck['actor_state_dict'])
    auac, seqs, branch = evaluate(actor, env, te_prof, list(te_prof), nm)
    print(f"\nEIG DONE: test held-out AUAC={auac:.4f} seqs={seqs}/{len(te_prof)} branches={branch} "
          f"(vs myopic greedy 0.716, PPO-static 0.719, scpr 0.721, oracle ~0.796)")


if __name__ == '__main__':
    main()
