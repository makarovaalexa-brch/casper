"""
UNIFIED CASPER (the method), applied identically to any dataset:
  equivariant actor, distilled from the realizable greedy-infogain teacher
  (the strongest heuristic), then optional RLOO finetune; evaluated under the
  ranking hard-LOO metric (Hit@10 vs popularity-matched negatives).

Same recipe everywhere (the stratified-winning method). Keep slates ~few-hundred
items so it stays tractable.

Env: INST_NAME, DATASET_NPZ, DATASET_NAME, DEPOCHS, FINETUNE(0/1), N_TRAIN, N_TEST.
Usage: INST_NAME=instrument_ml_stratified_rank DATASET_NPZ=... DATASET_NAME=ml_stratified \
       poetry run python scripts/paper2/train_casper.py
"""
import os, sys, time
sys.path.insert(0, '.'); sys.path.insert(0, 'scripts/paper1'); sys.path.insert(0, 'scripts/paper2')
import numpy as np, torch, torch.nn.functional as F
from test_instrument_lib import load_instrument_by_name
from equivariant_actor import EquivariantActor

NAME = os.environ.get('DATASET_NAME', 'ml_stratified')
INST = os.environ.get('INST_NAME', f'instrument_{NAME}_rank')
NPZ = os.environ['DATASET_NPZ']
T = 15; N_NEG = 50; SEED = 42
N_TRAIN = int(os.environ.get('N_TRAIN', 600)); N_TEST = int(os.environ.get('N_TEST', 300))
DEPOCHS = int(os.environ.get('DEPOCHS', 12)); TOPK = int(os.environ.get('TEACHER_POOL', 200))
OUT = f'C:/dev/phd/casper/experiments/paper2/casper_{NAME}.pt'
torch.manual_seed(SEED); np.random.seed(SEED)


def main():
    w, _ = load_instrument_by_name(INST)
    d = np.load(NPZ, allow_pickle=True); train, test = d['train'], d['test']
    nt = int(d['n_targets']); ni = train.shape[1]
    pop = (~np.isnan(train[:, :nt])).mean(0); p_all = (~np.isnan(train)).mean(0)
    rng = np.random.default_rng(SEED)
    # teacher candidate pool: attributes + top-pop movies (fast greedy-infogain)
    pool = sorted(set(range(nt, ni)) | set(int(m) for m in np.argsort(-pop)[:TOPK]))

    def state_from(rev):
        bel = np.asarray(w.predict_full(rev)); rat = np.asarray(w.predict_rated(rev))
        s = np.zeros((ni, 3), np.float32); s[:, 2] = 1
        for (e, p) in rev: s[e, 2] = 0; s[e, 1 if p >= 0.5 else 0] = 1
        return np.concatenate([s.reshape(-1), bel.astype(np.float32), rat.astype(np.float32)])

    def ent(p): p = np.clip(p, 1e-6, 1 - 1e-6); return -(p*np.log2(p)+(1-p)*np.log2(1-p))

    def greedy_teacher(rev, asked, prof):
        cur = np.asarray(w.predict_full(rev)); h_now = ent(cur[:nt]).sum()
        cands = [e for e in pool if e not in asked]
        sets = []
        for e in cands:
            sets.append(rev + [(e, 1.0)]); sets.append(rev + [(e, 0.0)])
        preds = w.predict_batch(sets)  # [2*ncand, nt]
        best, beig = cands[0], -1e9
        for j, e in enumerate(cands):
            hl = ent(preds[2*j]).sum(); hd = ent(preds[2*j+1]).sum()
            pl = float(cur[e]); eig = p_all[e]*(pl*(h_now-hl)+(1-pl)*(h_now-hd))
            if eig > beig: beig, best = eig, e
        return int(best)

    # ---- collect teacher rollouts ----
    print(f"CASPER {NAME}: instrument={INST} items={ni} targets={nt}; collecting greedy-teacher", flush=True)
    X, A = [], []; t0 = time.time()
    tr = rng.choice(len(train), min(N_TRAIN, len(train)), replace=False)
    for n, ui in enumerate(tr):
        prof = train[ui]; rev = []; asked = set()
        for _ in range(T):
            a = greedy_teacher(rev, asked, prof)
            X.append(state_from(rev)); A.append(a); asked.add(a)
            v = prof[a]
            if not np.isnan(v): rev.append((a, float(v)))
        if (n+1) % 200 == 0: print(f"  {n+1}/{len(tr)} users ({time.time()-t0:.0f}s)", flush=True)
    X = np.array(X, np.float32); A = np.array(A, np.int64)
    print(f"  {len(X)} pairs", flush=True)

    # ---- distill ----
    actor = EquivariantActor(ni, 'dual'); opt = torch.optim.Adam(actor.parameters(), 1e-3, weight_decay=1e-5)
    Xt, At = torch.from_numpy(X), torch.from_numpy(A)
    for ep in range(DEPOCHS):
        actor.train(); idx = torch.randperm(len(Xt))
        for s in range(0, len(idx), 256):
            b = idx[s:s+256]; opt.zero_grad(); F.cross_entropy(actor(Xt[b]), At[b]).backward(); opt.step()
        with torch.no_grad():
            acc = (actor(Xt[:1000]).argmax(1) == At[:1000]).float().mean().item()
        print(f"  ep{ep+1}/{DEPOCHS} action-acc={acc:.3f}", flush=True)
    torch.save({'actor_state_dict': actor.state_dict(), 'arch': 'equivariant', 'n_items': ni}, OUT)

    # ---- hard-LOO eval (movie-scoped) ----
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

    def hit(rev, cand): sc = np.asarray(w.predict(rev))[cand]; return 1.0 if 1+int((sc[1:] >= sc[0]).sum()) <= 10 else 0.0
    actor.eval(); curves = []
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
    print(f"\nCASPER {NAME}: turn0={cur[0]:.3f} t5={cur[5]:.3f} t15={cur[-1]:.3f} AUC={cur.mean():.4f}", flush=True)
    import json
    json.dump({'casper_curve': [float(x) for x in cur]},
              open(f'C:/dev/phd/casper/experiments/paper1/casper_{NAME}_loo.json', 'w'))


if __name__ == '__main__':
    main()
