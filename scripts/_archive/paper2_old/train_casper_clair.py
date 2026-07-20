"""
CASPER via CLAIRVOYANT IMITATION (Choudhury et al. RSS'17 / ADVISOR imitation-gap).

Motivation (measured on ml_stratified, calibrated ranking instrument, hard-LOO):
  - instrument is monotone in TRUE reveals (0.67 -> 0.88): sound measuring device.
  - every realizable myopic heuristic LOSES to popularity:
        greedy-infogain (global)    AUC 0.68
        answ x decision-EIG         AUC 0.61
        popularity (static)         AUC 0.74
  - the clairvoyant ORACLE (answerable + target-aligned) AUC 0.97.
  => the adaptive edge exists but needs TARGET knowledge; no observable greedy
     rule recovers it. So LEARN it: distill the oracle into a non-clairvoyant
     amortized policy. The policy sees only observable state (instrument beliefs
     + answerability + reveal one-hots); the oracle teacher sees the target.

Teacher = clairvoyant oracle: at each turn pick the ANSWERABLE pool item whose
true reveal most improves the target's rank vs popularity-matched decoys.
Student = EquivariantActor, behaviour-cloned on (observable_state, oracle_action).

Env: INST_NAME, DATASET_NPZ, DATASET_NAME, N_TRAIN, N_TEST, DEPOCHS.
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
N_TRAIN = int(os.environ.get('N_TRAIN', 800)); N_TEST = int(os.environ.get('N_TEST', 300))
DEPOCHS = int(os.environ.get('DEPOCHS', 15)); TOPK = int(os.environ.get('TEACHER_POOL', 200))
OUT = f'C:/dev/phd/casper/experiments/paper2/casper_clair_{NAME}.pt'
torch.manual_seed(SEED); np.random.seed(SEED)


def main():
    w, _ = load_instrument_by_name(INST)
    d = np.load(NPZ, allow_pickle=True); train, test = d['train'], d['test']
    nt = int(d['n_targets']); ni = train.shape[1]
    pop = (~np.isnan(train[:, :nt])).mean(0)
    rng = np.random.default_rng(SEED)
    pool = sorted(set(range(nt, ni)) | set(int(m) for m in np.argsort(-pop)[:TOPK]))
    dec = np.zeros(nt, int); o = np.argsort(pop)
    for q in range(10): dec[o[q*nt//10:(q+1)*nt//10]] = q
    by = {q: set(np.where(dec == q)[0].tolist()) for q in range(10)}

    def state_from(rev):                       # OBSERVABLE state only (no target)
        bel = np.asarray(w.predict_full(rev)); rat = np.asarray(w.predict_rated(rev))
        s = np.zeros((ni, 3), np.float32); s[:, 2] = 1
        for (e, p) in rev: s[e, 2] = 0; s[e, 1 if p >= 0.5 else 0] = 1
        return np.concatenate([s.reshape(-1), bel.astype(np.float32), rat.astype(np.float32)])

    def oracle_action(rev, asked, prof, cand):  # CLAIRVOYANT: uses prof + target via cand[0]
        cands = [e for e in pool if e not in asked and not np.isnan(prof[e])]
        if not cands:
            cands = [e for e in pool if e not in asked]            # fallback (unanswerable)
            if not cands: return None
            return int(cands[0])
        sets = [rev + [(e, float(prof[e]))] for e in cands]
        preds = w.predict_batch(sets)
        best, bs = cands[0], -1e9
        for j, e in enumerate(cands):
            sc = preds[j][cand]; r = -(1 + int((sc[1:] >= sc[0]).sum()))   # -rank, higher better
            if r > bs: bs, best = r, e
        return int(best)

    # ---- collect clairvoyant rollouts ----
    print(f"CLAIR {NAME}: inst={INST} items={ni} targets={nt}; collecting oracle demos", flush=True)
    X, A = [], []; t0 = time.time()
    tr = rng.choice(len(train), min(N_TRAIN, len(train)), replace=False)
    used = 0
    for n, ui in enumerate(tr):
        prof = train[ui]; liked = np.where(prof[:nt] == 1)[0]
        rated = set(np.where(~np.isnan(prof[:nt]))[0].tolist())
        if len(liked) < 3: continue
        tgt = int(liked[rng.integers(len(liked))])
        negp = [i for i in by[dec[tgt]] if i not in rated]
        if len(negp) < N_NEG: continue
        cand = np.array([tgt] + list(rng.choice(negp, N_NEG, replace=False)))
        rev = []; asked = {tgt}                  # never ask the held-out target itself
        for _ in range(T):
            a = oracle_action(rev, asked, prof, cand)
            if a is None: break
            X.append(state_from(rev)); A.append(a); asked.add(a)
            if not np.isnan(prof[a]): rev.append((a, float(prof[a])))
        used += 1
        if used % 200 == 0: print(f"  {used} demos ({time.time()-t0:.0f}s)", flush=True)
    X = np.array(X, np.float32); A = np.array(A, np.int64)
    print(f"  {len(X)} (state,action) pairs from {used} users", flush=True)

    # ---- behaviour clone ----
    actor = EquivariantActor(ni, 'dual'); opt = torch.optim.Adam(actor.parameters(), 1e-3, weight_decay=1e-5)
    Xt, At = torch.from_numpy(X), torch.from_numpy(A)
    for ep in range(DEPOCHS):
        actor.train(); idx = torch.randperm(len(Xt))
        for s in range(0, len(idx), 256):
            b = idx[s:s+256]; opt.zero_grad(); F.cross_entropy(actor(Xt[b]), At[b]).backward(); opt.step()
        with torch.no_grad():
            acc = (actor(Xt[:2000]).argmax(1) == At[:2000]).float().mean().item()
        print(f"  ep{ep+1}/{DEPOCHS} oracle-action-acc={acc:.3f}", flush=True)
    torch.save({'actor_state_dict': actor.state_dict(), 'arch': 'equivariant', 'n_items': ni}, OUT)

    # ---- hard-LOO eval ----
    cases = []
    for prof in test:
        liked = np.where(prof[:nt] == 1)[0]; rated = set(np.where(~np.isnan(prof[:nt]))[0].tolist())
        if len(liked) < 3: continue
        tgt = int(liked[rng.integers(len(liked))]); negp = [i for i in by[dec[tgt]] if i not in rated]
        if len(negp) < N_NEG: continue
        cases.append((prof, tgt, np.array([tgt] + list(rng.choice(negp, N_NEG, replace=False)))))
        if len(cases) >= N_TEST: break

    def hit(rev, cand): sc = np.asarray(w.predict(rev))[cand]; return 1.0 if 1+int((sc[1:] >= sc[0]).sum()) <= 10 else 0.0
    actor.eval(); curves = []
    for (prof, tgt, cand) in cases:
        rev = []; asked = {tgt}; row = [hit([], cand)]
        for _ in range(T):
            with torch.no_grad():
                lg = actor(torch.from_numpy(state_from(rev)).unsqueeze(0))[0].numpy()
            lg[list(asked)] = -1e9; lg[nt:] = -1e9
            a = int(np.argmax(lg)); asked.add(a)
            if not np.isnan(prof[a]): rev.append((a, float(prof[a])))
            row.append(hit(rev, cand))
        curves.append(row)
    cur = np.array(curves).mean(0)
    print(f"\nCASPER-CLAIR {NAME}: turn0={cur[0]:.3f} t5={cur[5]:.3f} t15={cur[-1]:.3f} AUC={cur.mean():.4f}", flush=True)
    print(f"  refs: popularity 0.738 | greedy-infogain 0.680 | oracle 0.974", flush=True)
    import json
    json.dump({'casper_clair_curve': [float(x) for x in cur]},
              open(f'C:/dev/phd/casper/experiments/paper1/casper_clair_{NAME}_loo.json', 'w'))


if __name__ == '__main__':
    main()
