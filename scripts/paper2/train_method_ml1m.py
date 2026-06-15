"""
THE METHOD on ML-1M (ranking instrument, hard-LOO metric).

(1) Diagnostic: headroom from revealing the user's rated movies chosen
    popular-first vs random vs tail-first -> confirms the discriminative
    signal is in the tail (why greedy/popularity heuristics fail).
(2) Teacher (privileged, train-time): asks the user's OWN rated movies,
    tail-first (answerable + discriminative). Distill into an equivariant
    belief-state policy (it must learn to target answerable discriminative
    items from beliefs + the answerability head).
(3) Evaluate distilled policy under full-catalog hard-LOO vs thompson (0.537).

Usage: poetry run python scripts/paper2/train_method_ml1m.py
"""
import os, sys, time
sys.path.insert(0, '.'); sys.path.insert(0, 'scripts/paper1'); sys.path.insert(0, 'scripts/paper2')
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from test_instrument_lib import load_instrument_by_name
from equivariant_actor import EquivariantActor

INST = os.environ.get('INST_NAME', 'instrument_ml1m_rank')
NPZ = os.environ.get('DATASET_NPZ', 'C:/dev/phd/casper/data/movielens/ml1m_profiles.npz')
T = 15; N_NEG = 50; N_TRAIN = 1500; N_TEST = 300; SEED = 42
OUT = 'C:/dev/phd/casper/experiments/paper2/method_ml1m.pt'
torch.manual_seed(SEED); np.random.seed(SEED)


def main():
    w, _ = load_instrument_by_name(INST)
    d = np.load(NPZ, allow_pickle=True)
    train, test = d['train'], d['test']; nt = int(d['n_targets']); ni = train.shape[1]
    pop = (~np.isnan(train[:, :nt])).mean(0)
    rng = np.random.default_rng(SEED)

    # pop-matched negative machinery
    dec = np.zeros(nt, int); order = np.argsort(pop)
    for q in range(10):
        dec[order[q * nt // 10:(q + 1) * nt // 10]] = q
    by = {q: set(np.where(dec == q)[0].tolist()) for q in range(10)}

    def make_case(prof):
        liked = np.where(prof[:nt] == 1)[0]
        rated = set(np.where(~np.isnan(prof[:nt]))[0].tolist())
        if len(liked) < 3:
            return None
        tgt = int(liked[rng.integers(len(liked))])
        negpool = [i for i in by[dec[tgt]] if i not in rated]
        if len(negpool) < N_NEG:
            return None
        return tgt, np.array([tgt] + list(rng.choice(negpool, N_NEG, replace=False)))

    def hit10(rev, cand):
        s = np.asarray(w.predict(rev))[cand]
        return 1.0 if 1 + int((s[1:] >= s[0]).sum()) <= 10 else 0.0

    # ---------- (1) diagnostic: popular/random/tail reveals ----------
    print("=== (1) headroom by reveal-order (K=10 of user's rated movies) ===", flush=True)
    modes = {'popular_first': lambda r: sorted(r, key=lambda i: -pop[i]),
             'random': lambda r: list(rng.permutation(r)),
             'tail_first': lambda r: sorted(r, key=lambda i: pop[i])}
    diag = {m: [] for m in modes}
    test_cases = []
    for p in test[:N_TEST * 2]:
        c = make_case(p)
        if c:
            test_cases.append((p, *c))
        if len(test_cases) >= N_TEST:
            break
    for (prof, tgt, cand) in test_cases:
        rated = [i for i in np.where(prof[:nt] == 1)[0] if i != tgt] + \
                [i for i in np.where(prof[:nt] == 0)[0]]
        for m, keyf in modes.items():
            chosen = keyf(list(rated))[:10]
            rev = [(int(e), float(prof[e])) for e in chosen]
            diag[m].append(hit10(rev, cand))
    for m in modes:
        print(f"    {m:<14} Hit@10={np.mean(diag[m]):.3f}", flush=True)

    # ---------- (2) collect teacher data (tail-first rated) ----------
    print("=== (2) collecting teacher rollouts (tail-first rated) ===", flush=True)
    def state_from(rev):
        bel = np.asarray(w.predict_full(rev)); rat = np.asarray(w.predict_rated(rev))
        s = np.zeros((ni, 3), np.float32); s[:, 2] = 1
        for (e, p) in rev:
            s[e, 2] = 0; s[e, 1 if p >= 0.5 else 0] = 1
        return np.concatenate([s.reshape(-1), bel.astype(np.float32), rat.astype(np.float32)])
    X, A = [], []; t0 = time.time()
    tr_idx = rng.choice(len(train), min(N_TRAIN, len(train)), replace=False)
    for n, ui in enumerate(tr_idx):
        prof = train[ui]
        rated = [i for i in np.where(prof[:nt] == 1)[0]] + [i for i in np.where(prof[:nt] == 0)[0]]
        if len(rated) < 4:
            continue
        askable = sorted(rated, key=lambda i: pop[i])  # tail-first
        rev = []
        for a in askable[:T]:
            X.append(state_from(rev)); A.append(int(a))
            rev.append((int(a), float(prof[a])))
        if (n + 1) % 300 == 0:
            print(f"    {n+1}/{len(tr_idx)} users, {len(X)} pairs ({time.time()-t0:.0f}s)", flush=True)
    X = np.array(X, np.float32); A = np.array(A, np.int64)
    print(f"    {len(X)} state-action pairs", flush=True)

    # ---------- (3) BC distill into equivariant actor ----------
    print("=== (3) distilling equivariant policy ===", flush=True)
    actor = EquivariantActor(ni, 'dual')
    opt = torch.optim.Adam(actor.parameters(), lr=1e-3, weight_decay=1e-5)
    Xt, At = torch.from_numpy(X), torch.from_numpy(A)
    for ep in range(40):
        actor.train(); idx = torch.randperm(len(Xt))
        for s in range(0, len(idx), 128):
            b = idx[s:s + 128]
            opt.zero_grad(); F.cross_entropy(actor(Xt[b]), At[b]).backward(); opt.step()
        if (ep + 1) % 10 == 0:
            with torch.no_grad():
                acc = (actor(Xt[:2000]).argmax(1) == At[:2000]).float().mean().item()
            print(f"    ep{ep+1} train action-acc={acc:.3f}", flush=True)
    torch.save({'actor_state_dict': actor.state_dict(), 'arch': 'equivariant',
                'n_items': ni}, OUT)

    # ---------- (4) hard-LOO eval (full catalog) ----------
    print("=== (4) METHOD hard-LOO (full catalog) vs thompson 0.537 ===", flush=True)
    actor.eval(); curves = []
    for (prof, tgt, cand) in test_cases:
        rev = []; asked = {tgt}; row = [hit10([], cand)]
        for _ in range(T):
            s = torch.from_numpy(state_from(rev)).unsqueeze(0)
            with torch.no_grad():
                logits = actor(s)[0].numpy()
            logits[list(asked)] = -1e9
            a = int(np.argmax(logits)); asked.add(a)
            v = prof[a]
            if not np.isnan(v):
                rev.append((a, float(v)))
            row.append(hit10(rev, cand))
        curves.append(row)
    cur = np.array(curves).mean(0)
    print(f"    METHOD turn0={cur[0]:.3f} t5={cur[5]:.3f} t15={cur[-1]:.3f} AUC={cur.mean():.4f}")
    print(f"    (thompson AUC 0.485 t15 0.537 ; ceiling 0.647 ; prior 0.440)")


if __name__ == '__main__':
    main()
