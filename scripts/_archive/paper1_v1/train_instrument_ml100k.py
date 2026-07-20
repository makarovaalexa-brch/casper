"""
Train the instrument PROPERLY on ML-100k and PROVE monotonicity
(more true reveals -> monotonically better RMSE and NDCG).

Fixes the hasty inline training:
  - heavy reveal-subset augmentation (S samples/user/epoch)
  - validation split + best-checkpoint + early stopping
  - dropout + weight decay (743 users is small -> overfits easily)
  - LR schedule (ReduceLROnPlateau)
Monotonicity gate: reveal K RANDOM TRUE ratings, K=0..20 -> RMSE must fall and
NDCG must rise monotonically. If not, the instrument is broken.
"""
import os, sys, time
import numpy as np, torch, torch.nn as nn

base = 'C:/dev/phd/casper/data/movielens/ml-100k'
SEED = 42; D = int(os.environ.get('D', 48)); EPOCHS = int(os.environ.get('EPOCHS', 200))
LR = float(os.environ.get('LR', 1e-3)); WD = float(os.environ.get('WD', 1e-4))
DROP = float(os.environ.get('DROP', 0.2)); S = int(os.environ.get('S', 4)); BS = 128
RANK_W = float(os.environ.get('RANK_W', 1.0))
POOL = os.environ.get('POOL', 'attn')              # attn | mean | tf
REVEAL_DIST = os.environ.get('REVEAL_DIST', 'uniform')  # uniform | logu
MAX_CTX = int(os.environ.get('MAX_CTX', 40))       # cap context length (eval only goes to 20)
PATIENCE = int(os.environ.get('PATIENCE', 20)); N_COLD = 200; NEG = 100
OUT = 'C:/dev/phd/casper/data/movielens/.cache/checkpoints/instrument_ml100k.pt'
torch.manual_seed(SEED); np.random.seed(SEED); rng = np.random.default_rng(SEED)

# ---- load + warm/cold split (same as side-by-side) ----
U, I, R = [], [], []
with open(f'{base}/u.data') as f:
    for line in f:
        a = line.strip().split('\t'); U.append(int(a[0])); I.append(int(a[1])); R.append(float(a[2]))
U = np.array(U); I = np.array(I); R = np.array(R, np.float32)
uids = {x: k for k, x in enumerate(np.unique(U))}; iids = {x: k for k, x in enumerate(np.unique(I))}
u = np.array([uids[x] for x in U]); i = np.array([iids[x] for x in I]); nu, ni = len(uids), len(iids)
cold = set(rng.choice(nu, N_COLD, replace=False).tolist())
warm_mask = np.array([uu not in cold for uu in u])
mu = float(R[warm_mask].mean())
ratings_by = {}
for k in range(len(u)): ratings_by.setdefault(u[k], {})[i[k]] = R[k]
warm_users = [uu for uu in ratings_by if uu not in cold]
rng.shuffle(warm_users); nval = 80
val_users = warm_users[:nval]; tr_users = warm_users[nval:]
print(f"ML-100k: {ni} movies; train {len(tr_users)} / val {nval} warm users; {N_COLD} cold held out", flush=True)


class SetRec(nn.Module):
    def __init__(self, ni, d, drop):
        super().__init__()
        self.item = nn.Embedding(ni, d); self.rate = nn.Embedding(6, d)
        self.enc = nn.Sequential(nn.Linear(d, d), nn.ReLU(), nn.Dropout(drop), nn.Linear(d, d))
        self.post = nn.Sequential(nn.Linear(d, d), nn.ReLU(), nn.Dropout(drop), nn.Linear(d, d))
        self.head = nn.Linear(d, ni); self.bi = nn.Parameter(torch.zeros(ni)); self.u0 = nn.Parameter(torch.zeros(d))
        self.attn_q = nn.Parameter(torch.randn(d) * 0.1); self.dscale = d ** 0.5
        if POOL == 'tf':
            self.cls = nn.Parameter(torch.randn(d) * 0.1)
            self.tf = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(d, nhead=4, dim_feedforward=2 * d, dropout=DROP, batch_first=True),
                num_layers=2)
    def uh(self, items, rates, mask):
        tok = self.item(items) + self.rate(rates)
        if POOL == 'tf':
            B = tok.shape[0]
            x = torch.cat([self.cls.expand(B, 1, -1), tok], 1)            # prepend CLS
            pad = torch.cat([torch.zeros(B, 1, dtype=torch.bool), mask], 1)  # CLS never masked
            out = self.tf(x, src_key_padding_mask=pad)
            return self.post(out[:, 0])                                   # CLS summary
        h = self.enc(tok)
        if POOL == 'attn':
            scores = (h @ self.attn_q / self.dscale).masked_fill(mask, -1e9)
            w = torch.softmax(scores, dim=1).unsqueeze(-1); pooled = (w * h).sum(1)
        else:
            hm = h.masked_fill(mask.unsqueeze(-1), 0.)
            n = (~mask).sum(1, keepdim=True).clamp(min=1); pooled = hm.sum(1) / n
        nrev = (~mask).sum(1); empty = (nrev == 0); out = self.post(pooled)
        return torch.where(empty.unsqueeze(1), self.u0.unsqueeze(0).expand_as(out), out)
    def forward(self, items, rates, mask):
        return mu + self.bi + self.head(self.uh(items, rates, mask))


def make_batch(users):
    rows = []
    for uu in users:
        rd = ratings_by[uu]; items = list(rd.keys())
        if len(items) < 2: continue
        kmax = min(len(items) - 1, MAX_CTX)
        for _ in range(S):
            if REVEAL_DIST == 'uniform':
                k = int(rng.integers(1, kmax + 1))             # flat over reveal counts, capped
            else:
                k = int(np.clip(np.exp(rng.uniform(0, np.log(len(items)))), 1, kmax))
            ctx = list(rng.choice(items, k, replace=False)); rows.append((ctx, rd))
    L = max(len(c) for c, _ in rows); B = len(rows)
    ci = np.zeros((B, L), np.int64); cr = np.zeros((B, L), np.int64); cm = np.ones((B, L), bool)
    tgt = np.zeros((B, ni), np.float32); tm = np.zeros((B, ni), np.float32)
    for b, (ctx, rd) in enumerate(rows):
        seen = set()
        for j, e in enumerate(ctx): ci[b, j] = e; cr[b, j] = int(round(rd[e])); cm[b, j] = False; seen.add(e)
        for e, r in rd.items():
            if e not in seen: tgt[b, e] = r; tm[b, e] = 1.
    return (torch.from_numpy(ci), torch.from_numpy(cr), torch.from_numpy(cm),
            torch.from_numpy(tgt), torch.from_numpy(tm))

model = SetRec(ni, D, DROP)
opt = torch.optim.Adam(model.parameters(), LR, weight_decay=WD)
sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=6)

VAL_K = int(os.environ.get('VAL_K', 12))      # reveal count for val NDCG selection

def predict_t(rd_ctx):
    if rd_ctx:
        ci = torch.tensor([[j for j, _ in rd_ctx]]); cr = torch.tensor([[int(round(r)) for _, r in rd_ctx]]); cm = torch.zeros_like(ci, dtype=torch.bool)
    else:
        ci = torch.zeros(1, 1, dtype=torch.long); cr = torch.zeros(1, 1, dtype=torch.long); cm = torch.ones(1, 1, dtype=torch.bool)
    with torch.no_grad(): return model(ci, cr, cm)[0].numpy()

def val_ndcg(K):
    model.eval(); vals = []
    for uu in val_users:
        rd = ratings_by[uu]; items = list(rd.keys())
        if len(items) < 20: continue
        ii = items[:]; rng.shuffle(ii); test = set(ii[:max(5, len(ii)//3)]); pool = [x for x in ii if x not in test]
        rel = [j for j in test if rd[j] >= 4]
        if not rel: continue
        unr = np.array([j for j in range(ni) if j not in rd])
        pr = predict_t([(e, rd[e]) for e in pool[:K]])
        nd = sum((1./np.log2(2+int((pr[unr] >= pr[ri]).sum())) if 1+int((pr[unr] >= pr[ri]).sum()) <= 10 else 0.) for ri in rel) / len(rel)
        vals.append(nd)
    return float(np.mean(vals))

best = -1.0; bstate = None; bad = 0; t0 = time.time()
for ep in range(EPOCHS):
    model.train(); rng.shuffle(tr_users)
    for s in range(0, len(tr_users), BS):
        ci, cr, cm, tgt, tm = make_batch(tr_users[s:s + BS])
        opt.zero_grad(); pred = model(ci, cr, cm)
        mse = (((pred - tgt) ** 2) * tm).sum() / tm.sum()
        pos = ((tgt >= 4) & (tm > 0)).float()
        sm = torch.zeros_like(pred); sm.scatter_add_(1, ci, (~cm).float()); seen_mask = sm > 0.5
        logp = torch.log_softmax(pred.masked_fill(seen_mask, -1e9), dim=1)
        npos = pos.sum(1).clamp(min=1)
        rank_per = -(logp * pos).sum(1) / npos
        has = pos.sum(1) > 0
        rank = rank_per[has].mean() if has.any() else torch.zeros((), requires_grad=True)
        loss = mse + RANK_W * rank
        loss.backward(); opt.step()
    vnd = val_ndcg(VAL_K); sched.step(-vnd)           # maximize val NDCG
    if vnd > best + 1e-4:
        best = vnd; bstate = {k: v.clone() for k, v in model.state_dict().items()}; bad = 0
    else:
        bad += 1
    if (ep + 1) % 10 == 0:
        print(f"  ep{ep+1} valNDCG@{VAL_K}={vnd:.4f} best={best:.4f} lr={opt.param_groups[0]['lr']:.1e} ({time.time()-t0:.0f}s)", flush=True)
    if bad >= PATIENCE:
        print(f"  early stop ep{ep+1}", flush=True); break
model.load_state_dict(bstate); model.eval()
torch.save({'state': model.state_dict(), 'd': D, 'ni': ni, 'mu': mu}, OUT)
print(f"saved {OUT}; best val NDCG@{VAL_K} {best:.4f}", flush=True)


# ---- MONOTONICITY GATE: reveal K random TRUE ratings ----
def predict(rd_ctx):
    if rd_ctx:
        ci = torch.tensor([[j for j, _ in rd_ctx]]); cr = torch.tensor([[int(round(r)) for _, r in rd_ctx]]); cm = torch.zeros_like(ci, dtype=torch.bool)
    else:
        ci = torch.zeros(1, 1, dtype=torch.long); cr = torch.zeros(1, 1, dtype=torch.long); cm = torch.ones(1, 1, dtype=torch.bool)
    with torch.no_grad(): return model(ci, cr, cm)[0].numpy()

KS = [0, 1, 2, 3, 5, 8, 12, 16, 20]
test_users = val_users  # use val warm users (have full ratings) for the clean monotonicity check
print("\n== MONOTONICITY: reveal K random TRUE ratings ==", flush=True)
print(f"{'K':>3} {'RMSE':>8} {'NDCG@10':>8} {'Recall@10':>9}")
agg = {k: {'rmse': [], 'ndcg': [], 'rec': []} for k in KS}
for uu in test_users:
    rd = ratings_by[uu]; items = list(rd.keys())
    if len(items) < 25: continue
    rng.shuffle(items); test = set(items[:max(5, len(items) // 3)]); pool = [x for x in items if x not in test]
    rel = [j for j in test if rd[j] >= 4]
    unr = np.array([j for j in range(ni) if j not in rd])      # FULL-catalogue negatives
    hi = np.array(list(test)); ht = np.array([rd[j] for j in test])
    for K in KS:
        ctx = [(e, rd[e]) for e in pool[:K]]
        pr = predict(ctx)
        agg[K]['rmse'].append(np.sqrt(np.mean((ht - np.clip(pr[hi], 1, 5)) ** 2)))
        if rel:
            nd = rc = 0.
            for ri in rel:
                rank = 1 + int((pr[unr] >= pr[ri]).sum())       # rank among all non-rated
                if rank <= 10: nd += 1. / np.log2(rank + 1); rc += 1.
            agg[K]['ndcg'].append(nd / len(rel)); agg[K]['rec'].append(rc / len(rel))
for K in KS:
    print(f"{K:>3} {np.mean(agg[K]['rmse']):>8.4f} {np.mean(agg[K]['ndcg']):>8.4f} {np.mean(agg[K]['rec']):>9.4f}")
