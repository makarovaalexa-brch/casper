"""
CASPER RLOO finetune: directly optimize the held-out ranking reward (Hit@10 vs
popularity-matched decoys), initialized from the clairvoyant-imitation policy.

Rationale (measured, ml_stratified, calibrated instrument, hard-LOO):
  realizable methods (popularity / belief-greedy / CASPER-CLAIR) all ~0.76 AUC,
  but the clairvoyant ORACLE hits 0.97 -> huge uncaptured headroom. BC cannot
  capture it (imitation gap: oracle action depends on the hidden target). RL does
  not imitate -- it maximizes expected reward over the target distribution, so it
  can learn the best *observable* policy. Reward = mean over turns of Hit@10
  against the same pop-matched decoys used at eval (uses target at TRAIN time only;
  the policy is amortized and target-free at test).

RLOO (REINFORCE leave-one-out): G stochastic rollouts/user, advantage =
R_g - mean_{g'!=g} R. Init from casper_clair_<name>.pt.

Env: INST_NAME, DATASET_NPZ, DATASET_NAME, EPOCHS, N_TRAIN, G, LR, ENT.
"""
import os, sys, time
sys.path.insert(0, '.'); sys.path.insert(0, 'scripts/paper1'); sys.path.insert(0, 'scripts/paper2')
import numpy as np, torch, torch.nn.functional as F
from test_instrument_lib import load_instrument_by_name
from equivariant_actor import EquivariantActor

NAME = os.environ.get('DATASET_NAME', 'ml_stratified')
INST = os.environ.get('INST_NAME', f'instrument_{NAME}_rankcal')
NPZ = os.environ['DATASET_NPZ']
T = 15; N_NEG = int(os.environ.get('N_NEG', 20)); SEED = 42
EPOCHS = int(os.environ.get('EPOCHS', 30)); N_TRAIN = int(os.environ.get('N_TRAIN', 250))
N_TEST = int(os.environ.get('N_TEST', 300)); G = int(os.environ.get('G', 6))
LR = float(os.environ.get('LR', 3e-4)); ENT = float(os.environ.get('ENT', 0.01))
TOPK = int(os.environ.get('TEACHER_POOL', 200))
INIT = f'C:/dev/phd/casper/experiments/paper2/casper_clair_{NAME}.pt'
OUT = f'C:/dev/phd/casper/experiments/paper2/casper_rloo_{NAME}.pt'
torch.manual_seed(SEED); np.random.seed(SEED)


def main():
    w, _ = load_instrument_by_name(INST)
    d = np.load(NPZ, allow_pickle=True); train, test = d['train'], d['test']
    nt = int(d['n_targets']); ni = train.shape[1]
    pop = (~np.isnan(train[:, :nt])).mean(0)
    rng = np.random.default_rng(SEED)
    pool = sorted(set(range(nt, ni)) | set(int(m) for m in np.argsort(-pop)[:TOPK]))
    poolmask = np.full(ni, -1e9, np.float32); poolmask[np.array(pool)] = 0.0
    dec = np.zeros(nt, int); o = np.argsort(pop)
    for q in range(10): dec[o[q*nt//10:(q+1)*nt//10]] = q
    by = {q: set(np.where(dec == q)[0].tolist()) for q in range(10)}

    def make_cases(src, n):
        cs = []
        for prof in src:
            liked = np.where(prof[:nt] == 1)[0]; rated = set(np.where(~np.isnan(prof[:nt]))[0].tolist())
            if len(liked) < 3: continue
            tgt = int(liked[rng.integers(len(liked))]); negp = [i for i in by[dec[tgt]] if i not in rated]
            if len(negp) < N_NEG: continue
            cs.append((prof, tgt, np.array([tgt] + list(rng.choice(negp, N_NEG, replace=False)))))
            if len(cs) >= n: break
        return cs
    testcases = make_cases(test, N_TEST)

    def state_from(rev):
        bel = np.asarray(w.predict_full(rev)); rat = np.asarray(w.predict_rated(rev))
        s = np.zeros((ni, 3), np.float32); s[:, 2] = 1
        for (e, p) in rev: s[e, 2] = 0; s[e, 1 if p >= 0.5 else 0] = 1
        return np.concatenate([s.reshape(-1), bel.astype(np.float32), rat.astype(np.float32)])

    def hit(rev, cand):
        sc = np.asarray(w.predict(rev))[cand]; return 1.0 if 1 + int((sc[1:] >= sc[0]).sum()) <= 10 else 0.0

    actor = EquivariantActor(ni, 'dual')
    actor.load_state_dict(torch.load(INIT)['actor_state_dict'])
    opt = torch.optim.Adam(actor.parameters(), LR)

    def eval_curve(cases):
        actor.eval(); curves = []
        for (prof, tgt, cand) in cases:
            rev = []; asked = {tgt}; row = [hit([], cand)]
            for _ in range(T):
                with torch.no_grad():
                    lg = actor(torch.from_numpy(state_from(rev)).unsqueeze(0))[0].numpy()
                lg = lg + poolmask; lg[list(asked)] = -1e9
                a = int(np.argmax(lg)); asked.add(a)
                if not np.isnan(prof[a]): rev.append((a, float(prof[a])))
                row.append(hit(rev, cand))
            curves.append(row)
        c = np.array(curves).mean(0); return c

    c0 = eval_curve(testcases)
    print(f"INIT (clair) AUC={c0.mean():.4f} t5={c0[5]:.3f} t15={c0[-1]:.3f}", flush=True)
    pmask_t = torch.from_numpy(poolmask)
    t0 = time.time(); best = c0.mean(); bstate = {k: v.clone() for k, v in actor.state_dict().items()}
    for ep in range(EPOCHS):
        actor.train(); cases = make_cases(train[rng.permutation(len(train))], N_TRAIN)
        ep_R = []; loss_accum = []
        opt.zero_grad()
        for (prof, tgt, cand) in cases:
            # G stochastic rollouts
            logps = [[] for _ in range(G)]; Rs = np.zeros(G)
            for g in range(G):
                rev = []; asked = {tgt}; hits = [hit([], cand)]
                for _ in range(T):
                    st = torch.from_numpy(state_from(rev)).unsqueeze(0)
                    lg = actor(st)[0] + pmask_t
                    lg[list(asked)] = -1e9
                    dist = torch.distributions.Categorical(logits=lg)
                    a = dist.sample(); logps[g].append(dist.log_prob(a) + ENT * dist.entropy())
                    ai = int(a.item()); asked.add(ai)
                    if not np.isnan(prof[ai]): rev.append((ai, float(prof[ai])))
                    hits.append(hit(rev, cand))
                Rs[g] = float(np.mean(hits))           # AUC-style reward
            ep_R.append(Rs.mean())
            base = (Rs.sum() - Rs) / (G - 1)            # leave-one-out baseline
            adv = Rs - base
            for g in range(G):
                if abs(adv[g]) < 1e-9: continue
                lp = torch.stack(logps[g]).sum()
                loss_accum.append(-(adv[g]) * lp)
        if loss_accum:
            loss = torch.stack(loss_accum).mean(); loss.backward()
            torch.nn.utils.clip_grad_norm_(actor.parameters(), 5.0); opt.step()
        c = eval_curve(testcases)
        tag = ""
        if c.mean() > best:
            best = c.mean(); bstate = {k: v.clone() for k, v in actor.state_dict().items()}; tag = " *"
        print(f"ep{ep+1}/{EPOCHS} trainR={np.mean(ep_R):.4f} | test AUC={c.mean():.4f} "
              f"t5={c[5]:.3f} t15={c[-1]:.3f} best={best:.4f}{tag} ({time.time()-t0:.0f}s)", flush=True)
    actor.load_state_dict(bstate)
    torch.save({'actor_state_dict': actor.state_dict(), 'arch': 'equivariant', 'n_items': ni}, OUT)
    cf = eval_curve(testcases)
    print(f"\nCASPER-RLOO {NAME}: turn0={cf[0]:.3f} t5={cf[5]:.3f} t15={cf[-1]:.3f} AUC={cf.mean():.4f}", flush=True)
    print(f"  refs: popularity 0.760 | belief-greedy 0.758 | clair-init {c0.mean():.4f} | oracle 0.974", flush=True)
    import json
    json.dump({'rloo_curve': [float(x) for x in cf], 'init_curve': [float(x) for x in c0]},
              open(f'C:/dev/phd/casper/experiments/paper1/casper_rloo_{NAME}_loo.json', 'w'))


if __name__ == '__main__':
    main()
