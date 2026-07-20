"""
STEP 3 — side-by-side on ONE established cold-start protocol (ML-100k):
  recommenders : MF (fold-in)   vs   Instrument (set-encoder, explicit ratings)
  strategies   : random / popular / HELF   (recommender-agnostic question order)
  metrics      : RMSE (Golbandi/Zhou)  +  NDCG@10 / Recall@10 LOO+100negs (NCF)
all vs number of questions, same warm/cold split, same held-out, same seeds.

Instrument = DeepSets set-encoder over revealed (item, rating) tokens -> per-item
rating head (the accepted Paper-1 architecture; structural rating channel, NOT a
two-tower embedding model). Trained on warm users, reveal-subset + MSE.
"""
import os, sys, time
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F

base = 'C:/dev/phd/casper/data/movielens/ml-100k'
SEED = 42; D = 32; MF_EPOCHS = 30; MF_LR = 0.008; MF_REG = 0.05; REGF = 4.0
INST_D = 64; INST_EPOCHS = int(os.environ.get('INST_EPOCHS', 40)); BS = 128
N_COLD = 200; T = 20; NEG = 100
torch.manual_seed(SEED); np.random.seed(SEED); rng = np.random.default_rng(SEED)

# ---- load ----
U, I, R = [], [], []
with open(f'{base}/u.data') as f:
    for line in f:
        a = line.strip().split('\t'); U.append(int(a[0])); I.append(int(a[1])); R.append(float(a[2]))
U = np.array(U); I = np.array(I); R = np.array(R, np.float32)
uids = {x: k for k, x in enumerate(np.unique(U))}; iids = {x: k for k, x in enumerate(np.unique(I))}
u = np.array([uids[x] for x in U]); i = np.array([iids[x] for x in I]); nu, ni = len(uids), len(iids)
cold = set(rng.choice(nu, N_COLD, replace=False).tolist())
warm_mask = np.array([uu not in cold for uu in u])
ut, it, rt = u[warm_mask], i[warm_mask], R[warm_mask]; mu = float(rt.mean())
cnt = np.bincount(it, minlength=ni)
pop_order = list(np.argsort(-cnt))
mean_r = np.array([rt[it == j].mean() if cnt[j] else mu for j in range(ni)])
var_r = np.array([rt[it == j].var() if cnt[j] > 1 else 0. for j in range(ni)])
lf = np.log1p(cnt) / np.log1p(cnt.max()); helf = 2 * lf * var_r / (lf + var_r + 1e-9)
helf_order = list(np.argsort(-helf))
# warm-user rating dicts (for instrument training)
warm_ratings = {}
for k in np.where(warm_mask)[0]:
    warm_ratings.setdefault(u[k], {})[i[k]] = R[k]
warm_users = list(warm_ratings.keys())
cold_ratings = {}
for k in np.where(~warm_mask)[0]:
    cold_ratings.setdefault(u[k], {})[i[k]] = R[k]
print(f"ML-100k: {nu} users ({len(warm_users)} warm/{N_COLD} cold), {ni} movies", flush=True)

# ---- MF (warm) ----
bu = np.zeros(nu); bi = np.zeros(ni)
P = 0.1 * rng.standard_normal((nu, D)); Q = 0.1 * rng.standard_normal((ni, D))
for ep in range(MF_EPOCHS):
    o = rng.permutation(len(ut)); lr = MF_LR / (1 + 0.05 * ep)
    for s in range(0, len(o), 20000):
        b = o[s:s + 20000]; uu, ii, rr = ut[b], it[b], rt[b]
        e = (rr - (mu + bu[uu] + bi[ii] + np.sum(P[uu] * Q[ii], 1))).astype(np.float64)
        np.add.at(bu, uu, lr * (e - MF_REG * bu[uu])); np.add.at(bi, ii, lr * (e - MF_REG * bi[ii]))
        np.add.at(P, uu, lr * (e[:, None] * Q[ii] - MF_REG * P[uu]))
        np.add.at(Q, ii, lr * (e[:, None] * P[uu] - MF_REG * Q[ii]))
print("MF trained.", flush=True)

def mf_predict(revealed):
    if not revealed: return mu + bi
    A = np.array([[1.] + Q[j].tolist() for j, _ in revealed]); y = np.array([r - mu - bi[j] for j, r in revealed])
    x = np.linalg.solve(A.T @ A + REGF * np.eye(D + 1), A.T @ y)
    return mu + x[0] + bi + Q @ x[1:]

# ---- Instrument (set-encoder, explicit ratings) ----
class SetRec(nn.Module):
    def __init__(self, ni, d):
        super().__init__()
        self.item = nn.Embedding(ni, d); self.rate = nn.Embedding(6, d)
        self.enc = nn.Sequential(nn.Linear(d, d), nn.ReLU(), nn.Linear(d, d))
        self.post = nn.Sequential(nn.Linear(d, d), nn.ReLU(), nn.Linear(d, d))
        self.head = nn.Linear(d, ni); self.bi = nn.Parameter(torch.zeros(ni)); self.u0 = nn.Parameter(torch.zeros(d))
    def uh(self, items, rates, mask):
        tok = self.item(items) + self.rate(rates); h = self.enc(tok).masked_fill(mask.unsqueeze(-1), 0.)
        n = (~mask).sum(1, keepdim=True).clamp(min=1); pooled = h.sum(1) / n
        empty = (n.squeeze(1) == 0); out = self.post(pooled)
        return torch.where(empty.unsqueeze(1), self.u0.unsqueeze(0).expand_as(out), out)
    def forward(self, items, rates, mask):
        return mu + self.bi + self.head(self.uh(items, rates, mask))

inst = SetRec(ni, INST_D); opt = torch.optim.Adam(inst.parameters(), 2e-3, weight_decay=1e-6)

def inst_batch(users):
    L = 1; ctxs = []
    for uu in users:
        rd = warm_ratings[uu]; items = list(rd.keys())
        k = int(np.clip(np.exp(rng.uniform(0, np.log(max(2, len(items))))), 1, len(items) - 1)) if len(items) > 1 else 1
        ctx = list(rng.choice(items, k, replace=False)); ctxs.append((ctx, rd)); L = max(L, k)
    B = len(users); ci = np.zeros((B, L), np.int64); cr = np.zeros((B, L), np.int64); cm = np.ones((B, L), bool)
    tgt = np.zeros((B, ni), np.float32); tm = np.zeros((B, ni), np.float32)
    for b, (ctx, rd) in enumerate(ctxs):
        seen = set()
        for j, e in enumerate(ctx): ci[b, j] = e; cr[b, j] = int(round(rd[e])); cm[b, j] = False; seen.add(e)
        for e, r in rd.items():
            if e not in seen: tgt[b, e] = r; tm[b, e] = 1.
    return (torch.from_numpy(ci), torch.from_numpy(cr), torch.from_numpy(cm),
            torch.from_numpy(tgt), torch.from_numpy(tm))

t0 = time.time()
for ep in range(INST_EPOCHS):
    inst.train(); perm = rng.permutation(len(warm_users))
    for s in range(0, len(perm), BS):
        users = [warm_users[k] for k in perm[s:s + BS]]
        ci, cr, cm, tgt, tm = inst_batch(users)
        opt.zero_grad(); pred = inst(ci, cr, cm)
        loss = ((pred - tgt) ** 2 * tm).sum() / tm.sum(); loss.backward(); opt.step()
    if (ep + 1) % 10 == 0: print(f"  inst ep{ep+1}/{INST_EPOCHS} MSE={loss.item():.4f} ({time.time()-t0:.0f}s)", flush=True)
inst.eval()

def inst_predict(revealed):
    if revealed:
        ci = torch.tensor([[j for j, _ in revealed]]); cr = torch.tensor([[int(round(r)) for _, r in revealed]]); cm = torch.zeros_like(ci, dtype=torch.bool)
    else:
        ci = torch.zeros(1, 1, dtype=torch.long); cr = torch.zeros(1, 1, dtype=torch.long); cm = torch.ones(1, 1, dtype=torch.bool)
    with torch.no_grad(): return inst(ci, cr, cm)[0].numpy()

# ---- eval harness: RMSE + NDCG@10 + Recall@10 vs #questions ----
RECS = {'MF': mf_predict, 'INSTR': inst_predict}
STRATS = {'random': None, 'popular': pop_order, 'HELF': helf_order}

def evaluate():
    out = {(rn, sn): {'rmse': np.zeros(T + 1), 'ndcg': np.zeros(T + 1), 'rec': np.zeros(T + 1)}
           for rn in RECS for sn in STRATS}
    used = 0
    for cu, ratings in cold_ratings.items():
        items_u = list(ratings.keys())
        if len(items_u) < 10: continue
        iu = items_u[:]; rng.shuffle(iu); ncut = max(5, int(0.3 * len(iu)))
        test = set(iu[:ncut])
        held_i = np.array([j for j in test]); held_r = np.array([ratings[j] for j in test])
        rel = [j for j in test if ratings[j] >= 4]
        unrated = [j for j in range(ni) if j not in ratings]
        negs = list(rng.choice(unrated, NEG, replace=False)) if len(unrated) >= NEG else unrated
        for sn, order in STRATS.items():
            o = pop_order[:] if order is None else order[:]
            if order is None: rng.shuffle(o)
            o = [x for x in o if x not in test]                  # never ask test items
            for rn, predfn in RECS.items():
                ans = []; d = out[(rn, sn)]
                for t in range(T + 1):
                    pr = predfn(ans)
                    d['rmse'][t] += np.sqrt(np.mean((held_r - np.clip(pr[held_i], 1, 5)) ** 2))
                    if rel:
                        nd = rc = 0.
                        for ri in rel:
                            sc = pr[np.array([ri] + negs)]; rank = 1 + int((sc[1:] >= sc[0]).sum())
                            if rank <= 10: nd += 1. / np.log2(rank + 1); rc += 1.
                        d['ndcg'][t] += nd / len(rel); d['rec'][t] += rc / len(rel)
                    if t == T: break
                    q = o[t] if t < len(o) else None
                    if q is not None and q in ratings: ans.append((q, ratings[q]))
        used += 1
    return out, used

res, n = evaluate()
for d in res.values():
    d['rmse'] /= n; d['ndcg'] /= n; d['rec'] /= n
print(f"\nevaluated {n} cold users (random 30% held-out; ask full catalogue; {NEG} negs)\n", flush=True)

for metric, lab, better in [('rmse', 'RMSE (lower better)', 'v'), ('ndcg', 'NDCG@10 (higher better)', '^'), ('rec', 'Recall@10 (higher better)', '^')]:
    print(f"=== {lab} ===")
    print(f"{'#q':>3}", *[f"{rn}/{sn:>7}" for rn in RECS for sn in STRATS])
    for t in [0, 1, 3, 5, 10, 20]:
        row = " ".join(f"{res[(rn,sn)][metric][t]:>11.4f}" for rn in RECS for sn in STRATS)
        print(f"{t:>3} {row}")
    print()
