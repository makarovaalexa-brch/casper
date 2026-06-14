"""
METHOD (candidate B): DAgger distillation of a clairvoyant oracle into an
equivariant belief-state elicitation policy.

Precedent: Choudhury et al., "Adaptive Information Gathering via Imitation
Learning," RSS 2017 (clairvoyant-oracle imitation; near-optimal under adaptive
submodularity). Novel here: applied to CRS preference elicitation with the
recommender as instrument. Fixes the PG-collapse (SNR) problem by replacing the
noisy task reward with a dense supervised target (the oracle's action), and the
flat-MLP actor with a permutation-equivariant per-item scorer.

DAgger (Ross et al. 2011): roll out the STUDENT to visit its own belief states,
label every visited state with the clairvoyant oracle's action, aggregate,
retrain. beta-schedule mixes oracle/student execution. Checkpoints every iter
(survives the ~42min background-job kill).

Usage: DATASET_NAME=ml_stratified poetry run python scripts/paper2/train_dagger.py
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
from oracle_distill import oracle_action

NAME = os.environ.get('DATASET_NAME', 'ml_stratified')
NPZ = {'ml_stratified': 'C:/dev/phd/casper/data/movielens/ml_stratified_profiles.npz',
       'yelp_multicity': 'C:/dev/phd/casper/data/yelp/yelp_multicity_profiles.npz'}[NAME]
N_TURNS = 15; SEED = 42
N_DAGGER = int(os.environ.get('DAGGER_ITERS', 5))
USERS_PER_ITER = int(os.environ.get('DAGGER_USERS', 150))
N_TEST = 300
OUT = f'C:/dev/phd/casper/experiments/paper2/dagger_{NAME}.pt'
torch.manual_seed(SEED); np.random.seed(SEED)


def student_action(actor, s, asked, n_items):
    with torch.no_grad():
        logits = actor(torch.from_numpy(s).unsqueeze(0))[0]
    mask = torch.full((n_items,), float('-inf'))
    rem = [i for i in range(n_items) if i not in asked]
    mask[rem] = 0.0
    return int(torch.argmax(logits + mask).item())


def evaluate(actor, env, profiles, uids, nm):
    actor.eval(); aucs = []; seqs = set(); first = Counter(); br = {}
    for uid in uids:
        s = env.reset(profiles[uid]); accs = [env.episode_accuracy()]; qs = []; ans = []
        for _ in range(N_TURNS):
            a = student_action(actor, s, env.asked, env.n_items)
            qs.append(a)
            v = profiles[uid][a]
            ans.append('unknown' if np.isnan(v) else ('liked' if v >= .5 else 'disliked'))
            s, _, _ = env.step(a); accs.append(env.episode_accuracy())
        aucs.append(np.nanmean(accs)); seqs.add(tuple(qs)); first[qs[0]] += 1
        if len(qs) >= 2: br.setdefault(ans[0], Counter())[qs[1]] += 1
    branch = len({c.most_common(1)[0][0] for c in br.values()}) > 1 if br else False
    return float(np.mean(aucs)), len(seqs), len(first), branch


def main():
    w, _ = load_instrument_by_name(f'instrument_{NAME}')
    d = np.load(NPZ, allow_pickle=True)
    train, test = d['train'], d['test']
    nm = int(d['n_targets']); n_items = train.shape[1]; attr_lo = nm
    belief, _ = load_instrument_by_name(f'instrument_{NAME}')
    env = ElicitationEnv(w, n_items, N_TURNS, belief_instrument=belief,
                         state_mode='dual', heldout=True)
    actor = EquivariantActor(n_items, 'dual')
    opt = torch.optim.Adam(actor.parameters(), lr=1e-3, weight_decay=1e-5)
    lossf = nn.CrossEntropyLoss()
    rng = np.random.default_rng(SEED)
    te_prof = {i: test[i] for i in range(min(N_TEST, len(test)))}

    DX, DA = [], []
    beta = 1.0
    print(f"{NAME}: DAgger, {N_DAGGER} iters x {USERS_PER_ITER} users, "
          f"equivariant actor params={sum(p.numel() for p in actor.parameters()):,}", flush=True)
    for it in range(N_DAGGER):
        t0 = time.time()
        uids = rng.choice(len(train), USERS_PER_ITER, replace=False)
        for uid in uids:
            prof = train[uid]; s = env.reset(prof)
            for _ in range(N_TURNS):
                a_star = oracle_action(env, prof, nm, attr_lo)
                DX.append(s.copy()); DA.append(a_star)
                a_exec = a_star if rng.random() < beta else student_action(actor, s, env.asked, n_items)
                s, _, _ = env.step(a_exec)
        # train on aggregated dataset
        X = torch.from_numpy(np.array(DX, np.float32)); A = torch.from_numpy(np.array(DA, np.int64))
        actor.train()
        for ep in range(15):
            idx = torch.randperm(len(X))
            for s0 in range(0, len(idx), 256):
                b = idx[s0:s0+256]
                opt.zero_grad(); lossf(actor(X[b]), A[b]).backward(); opt.step()
        auac, seqs, t1, branch = evaluate(actor, env, te_prof, list(te_prof), nm)
        torch.save({'actor_state_dict': actor.state_dict(), 'arch': 'equivariant',
                    'state_mode': 'dual', 'n_items': n_items, 'iter': it,
                    'auac_heldout': auac}, OUT)
        beta *= 0.6
        print(f"  iter {it+1}/{N_DAGGER} |D|={len(DX)} test AUAC={auac:.4f} "
              f"seqs={seqs}/{len(te_prof)} branches={branch} ({time.time()-t0:.0f}s)", flush=True)
    print(f"\nDAGGER DONE: test held-out AUAC={auac:.4f} branches={branch} "
          f"(vs scpr 0.7210, PPO-static 0.7192, oracle ~0.796); saved {OUT}")


if __name__ == '__main__':
    main()
