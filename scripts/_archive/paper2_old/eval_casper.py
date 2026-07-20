"""Eval a saved CASPER checkpoint under hard-LOO (full output, no grep)."""
import os, sys
sys.path.insert(0, '.'); sys.path.insert(0, 'scripts/paper1'); sys.path.insert(0, 'scripts/paper2')
import numpy as np, torch
from test_instrument_lib import load_instrument_by_name
from equivariant_actor import EquivariantActor

NAME = os.environ.get('DATASET_NAME', 'ml_stratified')
INST = os.environ.get('INST_NAME', f'instrument_{NAME}_rank')
NPZ = os.environ['DATASET_NPZ']
CK = f'C:/dev/phd/casper/experiments/paper2/casper_{NAME}.pt'
T = 15; N_NEG = int(os.environ.get('N_NEG', 20)); N_TEST = 300; SEED = 42
rng = np.random.default_rng(SEED)

w, _ = load_instrument_by_name(INST)
d = np.load(NPZ, allow_pickle=True); train, test = d['train'], d['test']
nt = int(d['n_targets']); ni = train.shape[1]
pop = (~np.isnan(train[:, :nt])).mean(0)
dec = np.zeros(nt, int); o = np.argsort(pop)
for q in range(10): dec[o[q*nt//10:(q+1)*nt//10]] = q
by = {q: set(np.where(dec == q)[0].tolist()) for q in range(10)}
cases = []
for prof in test:
    liked = np.where(prof[:nt] == 1)[0]; rated = set(np.where(~np.isnan(prof[:nt]))[0].tolist())
    if len(liked) < 3: continue
    tgt = int(liked[rng.integers(len(liked))]); negp = [i for i in by[dec[tgt]] if i not in rated]
    if len(negp) < N_NEG: continue
    cases.append((prof, tgt, np.array([tgt]+list(rng.choice(negp, N_NEG, replace=False)))))
    if len(cases) >= N_TEST: break
print(f"cases: {len(cases)}")
actor = EquivariantActor(ni, 'dual'); actor.load_state_dict(torch.load(CK)['actor_state_dict']); actor.eval()

def state_from(rev):
    bel = np.asarray(w.predict_full(rev)); rat = np.asarray(w.predict_rated(rev))
    s = np.zeros((ni, 3), np.float32); s[:, 2] = 1
    for (e, p) in rev: s[e, 2] = 0; s[e, 1 if p >= 0.5 else 0] = 1
    return np.concatenate([s.reshape(-1), bel.astype(np.float32), rat.astype(np.float32)])

def hit(rev, cand): sc = np.asarray(w.predict(rev))[cand]; return 1.0 if 1+int((sc[1:] >= sc[0]).sum()) <= 10 else 0.0
curves = []
for (prof, tgt, cand) in cases:
    rev = []; asked = {tgt}; row = [hit([], cand)]
    for _ in range(T):
        with torch.no_grad():
            lg = actor(torch.from_numpy(state_from(rev)).unsqueeze(0))[0].numpy()
        lg[list(asked)] = -1e9; lg[nt:] = -1e9
        a = int(np.argmax(lg)); asked.add(a); v = prof[a]
        if not np.isnan(v): rev.append((a, float(v)))
        row.append(hit(rev, cand))
    curves.append(row)
cur = np.array(curves).mean(0)
print(f"CASPER {NAME}: turn0={cur[0]:.3f} t5={cur[5]:.3f} t15={cur[-1]:.3f} AUC={cur.mean():.4f}")
