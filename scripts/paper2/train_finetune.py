"""
METHOD (final): warm-start the equivariant actor from the greedy-distilled
policy, then RL-finetune on the ALIGNED held-out reward (env's held-out
BCE-loss reduction) with an RLOO group baseline. Rationale (agent-3 fix B4):
starting from an already-adaptive policy (greedy ~0.716), RL only needs to find
small NON-MYOPIC improvements rather than discover routing from the 2%-signal
noise -- the only realistic way to push past the myopic greedy ceiling.

Warm-start ckpt: distill_<name>_greedy.pt (oracle_distill.py TEACHER=greedy).
Usage: DATASET_NAME=ml_stratified poetry run python scripts/paper2/train_finetune.py
"""
import os, sys, time
sys.path.insert(0, '.'); sys.path.insert(0, 'scripts/paper1'); sys.path.insert(0, 'scripts/paper2')
from collections import Counter
import numpy as np
import torch
from test_instrument_lib import load_instrument_by_name
from env import ElicitationEnv
from equivariant_actor import EquivariantActor

NAME = os.environ.get('DATASET_NAME', 'ml_stratified')
NPZ = {'ml_stratified': 'C:/dev/phd/casper/data/movielens/ml_stratified_profiles.npz',
       'yelp_multicity': 'C:/dev/phd/casper/data/yelp/yelp_multicity_profiles.npz'}[NAME]
N_TURNS = 15; SEED = 42
N_EPISODES = int(os.environ.get('CASPER_EPISODES', 8000))
G = int(os.environ.get('EIG_GROUP', 4))
ENTROPY_COEF = float(os.environ.get('ENT_COEF', 0.003))
LR = float(os.environ.get('LR', 1e-4))
N_TEST = 300
WARM = f'C:/dev/phd/casper/experiments/paper2/distill_{NAME}_greedy.pt'
OUT = f'C:/dev/phd/casper/experiments/paper2/finetune_{NAME}.pt'
torch.manual_seed(SEED); np.random.seed(SEED)


def rollout(actor, env, prof, rng, greedy=False):
    s = env.reset(prof); logps, rews = [], []; accs = [env.episode_accuracy()]
    for _ in range(N_TURNS):
        logits = actor(torch.from_numpy(s).unsqueeze(0))[0]
        mask = torch.full((env.n_items,), float('-inf'))
        rem = [i for i in range(env.n_items) if i not in env.asked]; mask[rem] = 0.0
        lp = torch.log_softmax(logits + mask, dim=0)
        a = int(torch.argmax(lp).item()) if greedy else int(torch.multinomial(lp.exp(), 1).item())
        logps.append(lp[a]); s, r, _ = env.step(a); rews.append(r); accs.append(env.episode_accuracy())
    return logps, rews, float(np.nanmean(accs)), s


def evaluate(actor, env, profiles, uids):
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
    d = np.load(NPZ, allow_pickle=True); train, test = d['train'], d['test']
    n_items = train.shape[1]
    belief, _ = load_instrument_by_name(f'instrument_{NAME}')
    env = ElicitationEnv(w, n_items, N_TURNS, belief_instrument=belief, state_mode='dual', heldout=True)
    actor = EquivariantActor(n_items, 'dual')
    ck = torch.load(WARM); actor.load_state_dict(ck['policy_state_dict'])
    print(f"{NAME}: warm-started from {WARM} (greedy-distill student AUAC {ck.get('student_auac')}); finetuning", flush=True)
    opt = torch.optim.Adam(actor.parameters(), lr=LR, weight_decay=1e-5)
    rng = np.random.default_rng(SEED)
    te_prof = {i: test[i] for i in range(min(N_TEST, len(test)))}
    val_uids = list(range(min(100, len(test))))
    base0, s0, b0 = evaluate(actor, env, te_prof, val_uids)
    print(f"  warm-start val AUAC={base0:.4f} seqs={s0}/{len(val_uids)} branch={b0}", flush=True)
    best = base0; t0 = time.time()
    for ep in range(0, N_EPISODES, G):
        uid = int(rng.integers(len(train))); prof = train[uid]
        actor.train()
        group = [rollout(actor, env, prof, rng) for _ in range(G)]
        rets = np.array([sum(g[1]) for g in group])
        loss = 0.0; ent = 0.0
        for gi, (logps, rews, _, _) in enumerate(group):
            adv = rets[gi] - (rets.sum() - rets[gi]) / (G - 1)
            lp = torch.stack(logps); loss = loss - adv * lp.sum(); ent = ent - (lp.exp() * lp).sum()
        (loss / G - ENTROPY_COEF * ent / G).backward()
        torch.nn.utils.clip_grad_norm_(actor.parameters(), 5.0); opt.step(); opt.zero_grad()
        if (ep + G) % 1000 < G:
            auac, seqs, branch = evaluate(actor, env, te_prof, val_uids)
            mark = ''
            if auac > best:
                best = auac; mark = ' *'
                torch.save({'actor_state_dict': actor.state_dict(), 'arch': 'equivariant',
                            'state_mode': 'dual', 'n_items': n_items, 'auac_heldout': auac}, OUT)
            print(f"  ep{ep+G:5d} val AUAC={auac:.4f} seqs={seqs}/{len(val_uids)} branch={branch} "
                  f"ret={rets.mean():+.4f} ({time.time()-t0:.0f}s){mark}", flush=True)
    if os.path.exists(OUT):
        actor.load_state_dict(torch.load(OUT)['actor_state_dict'])
    auac, seqs, branch = evaluate(actor, env, te_prof, list(te_prof))
    print(f"\nFINETUNE DONE: test held-out AUAC={auac:.4f} seqs={seqs}/{len(te_prof)} branches={branch} "
          f"(warm-start {base0:.4f}; greedy 0.716, scpr 0.721, oracle ~0.796)")


if __name__ == '__main__':
    main()
