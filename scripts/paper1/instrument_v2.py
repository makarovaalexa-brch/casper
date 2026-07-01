"""
Instrument v2 — the agreed clean design.

Architecture: transformer set-encoder (CLS over revealed (item,rating)) -> h, with
  - RANKING head (primary): item scores, sampled-softmax ranking loss -> drives the body
  - RATING head (diagnostic): MSE, body DETACHED (stop-grad) -> calibrated RMSE read-out
    that cannot corrupt the ranking body. (Resolves the head-fight: no RANK_W tradeoff.)

Protocol: warm-train / warm-val / cold-test (fixed split). All selection on VAL
(early-stop on val NDCG). Evaluate once on cold-test. Metrics: full-catalogue
NDCG@10/Recall@10 (primary) + RMSE (diagnostic), per #questions.

Baselines (fair, q0 popularity prior, tuned on val): random, popularity, biased-MF,
WRMF (implicit ALS, ranking).

Gates (cold-test): (1) monotone in true reveals, (2) beat tuned WRMF on ranking,
(3) RMSE <= constant baseline, (4) disliked-control (disliked rank low).

Env: SEED, EPOCHS, D, PATIENCE, MAX_CTX, S, VAL_K.
"""
import os, sys, time
import numpy as np, torch, torch.nn as nn

DATASET = os.environ.get('DATASET', 'ml-100k')
base = f'C:/dev/phd/casper/data/movielens/{DATASET}'
SEED = int(os.environ.get('SEED', 42)); D = int(os.environ.get('D', 48))
EPOCHS = int(os.environ.get('EPOCHS', 120)); PATIENCE = int(os.environ.get('PATIENCE', 25))
MAX_CTX = int(os.environ.get('MAX_CTX', 40)); S = int(os.environ.get('S', 4)); BS = 64
VAL_K = int(os.environ.get('VAL_K', 12)); DROP = 0.2; N_COLD = 200; N_VAL = 80
TIE = os.environ.get('TIE', '0') == '1'        # tie ranking output to item embeddings
MFINIT = os.environ.get('MFINIT', '0') == '1'  # init item embeddings from MF factors
torch.manual_seed(SEED); np.random.seed(SEED); rng = np.random.default_rng(SEED)

# ---- data + FIXED split (cold by a fixed seed so test is comparable across runs) ----
U, I, R = [], [], []
_fn, _sep = ('u.data', '\t') if DATASET == 'ml-100k' else ('ratings.dat', '::')
with open(f'{base}/{_fn}') as f:
    for line in f:
        a = line.strip().split(_sep); U.append(int(a[0])); I.append(int(a[1])); R.append(float(a[2]))
U = np.array(U); I = np.array(I); R = np.array(R, np.float32)
uids = {x: k for k, x in enumerate(np.unique(U))}; iids = {x: k for k, x in enumerate(np.unique(I))}
u = np.array([uids[x] for x in U]); i = np.array([iids[x] for x in I]); nu, ni = len(uids), len(iids)
split_rng = np.random.default_rng(0)                       # FIXED split regardless of run seed
cold = set(split_rng.choice(nu, N_COLD, replace=False).tolist())
warm_all = [x for x in range(nu) if x not in cold]; split_rng.shuffle(warm_all)
val_users = warm_all[:N_VAL]; tr_users = warm_all[N_VAL:]
ratings_by = {}
for k in range(len(u)): ratings_by.setdefault(u[k], {})[i[k]] = R[k]
wm = np.array([uu not in cold for uu in u]); ut, it, rt = u[wm], i[wm], R[wm]; mu = float(rt.mean())
cnt = np.bincount(it, minlength=ni); pop_order = list(np.argsort(-cnt))
logpop = np.log1p(cnt).astype(np.float64); logpop /= (logpop.max() + 1e-9)
cold_ratings = {uu: ratings_by[uu] for uu in cold}
print(f"{DATASET}: {nu}u {ni}i | warm-train {len(tr_users)} / val {N_VAL} / cold-test {N_COLD} | seed {SEED}", flush=True)

# ---- biased MF (RMSE baseline) ----
bu = np.zeros(nu); bi = np.zeros(ni); P = 0.1*rng.standard_normal((nu, 32)); Q = 0.1*rng.standard_normal((ni, 32))
for ep in range(30):
    o = rng.permutation(len(ut)); lr = 0.008/(1+0.05*ep)
    for s in range(0, len(o), 20000):
        b = o[s:s+20000]; uu, ii, rr = ut[b], it[b], rt[b]
        e = (rr-(mu+bu[uu]+bi[ii]+np.sum(P[uu]*Q[ii],1))).astype(np.float64)
        np.add.at(bu, uu, lr*(e-0.05*bu[uu])); np.add.at(bi, ii, lr*(e-0.05*bi[ii]))
        np.add.at(P, uu, lr*(e[:,None]*Q[ii]-0.05*P[uu])); np.add.at(Q, ii, lr*(e[:,None]*P[uu]-0.05*Q[ii]))
def mf_predict(rev):
    if not rev: return mu+bi
    A = np.array([[1.]+Q[j].tolist() for j,_ in rev]); y = np.array([r-mu-bi[j] for j,r in rev])
    x = np.linalg.solve(A.T@A+4.0*np.eye(33), A.T@y); return mu+x[0]+bi+Q@x[1:]

# ---- WRMF (ranking baseline) with q0 popularity prior; alpha tuned on val ----
def train_wrmf(alpha, dd=32, iters=15):
    Rm = np.full((nu, ni), np.nan)
    for k in np.where(wm)[0]: Rm[u[k], i[k]] = R[k]
    pos = (Rm >= 4); obs = ~np.isnan(Rm)
    Pw = 0.01*rng.standard_normal((nu, dd)); Qw = 0.01*rng.standard_normal((ni, dd))
    tr = [x for x in range(nu) if x not in cold]
    uo = {x: np.where(obs[x])[0] for x in tr}; up = {x: np.where(pos[x])[0] for x in tr}
    tra = np.array(tr); io = [np.where(obs[tra][:, j])[0] for j in range(ni)]; ip = [np.where(pos[tra][:, j])[0] for j in range(ni)]
    Id = 0.1*np.eye(dd)
    for it_ in range(iters):
        QtQ = Qw.T@Qw
        for x in tr:
            A = QtQ+Id+alpha*(Qw[uo[x]].T@Qw[uo[x]]); Pw[x] = np.linalg.solve(A, (1+alpha)*Qw[up[x]].sum(0) if len(up[x]) else np.zeros(dd))
        PtP = Pw[tra].T@Pw[tra]
        for j in range(ni):
            wu = tra[io[j]]; A = PtP+Id+alpha*(Pw[wu].T@Pw[wu]); wp = tra[ip[j]]
            Qw[j] = np.linalg.solve(A, (1+alpha)*Pw[wp].sum(0) if len(wp) else np.zeros(dd))
    return Qw, Qw.T@Qw
def make_wrmf_predict(Qw, QtQ, alpha, pop_w):
    def f(rev):
        A = QtQ+0.1*np.eye(Qw.shape[1]); rhs = np.zeros(Qw.shape[1])
        for j, r in rev:
            q = Qw[j]; A = A+alpha*np.outer(q, q)
            if r >= 4: rhs = rhs+(1+alpha)*q
        return Qw@np.linalg.solve(A, rhs) + pop_w*logpop      # q0 prior = popularity
    return f

# ---- instrument v2 (two-head transformer) ----
class InstrumentV2(nn.Module):
    def __init__(self, ni, d):
        super().__init__()
        self.item = nn.Embedding(ni, d); self.rate = nn.Embedding(6, d); self.cls = nn.Parameter(torch.randn(d)*0.1)
        self.tf = nn.TransformerEncoder(nn.TransformerEncoderLayer(d, 4, 2*d, DROP, batch_first=True), 2)
        self.rank_bias = nn.Parameter(torch.zeros(ni))
        if not TIE: self.rank_head = nn.Linear(d, ni)
        self.rate_head = nn.Linear(d, ni); self.rate_bias = nn.Parameter(torch.zeros(ni))
    def body(self, items, rates, mask):
        tok = self.item(items)+self.rate(rates); B = tok.shape[0]
        x = torch.cat([self.cls.expand(B, 1, -1), tok], 1)
        pad = torch.cat([torch.zeros(B, 1, dtype=torch.bool), mask], 1)
        return self.tf(x, src_key_padding_mask=pad)[:, 0]
    def forward(self, items, rates, mask):
        h = self.body(items, rates, mask)
        rank = self.rank_bias + (h @ self.item.weight.t() if TIE else self.rank_head(h))
        rate = mu + self.rate_bias + self.rate_head(h.detach())   # stop-grad: MSE can't touch body
        return rank, rate

def make_batch(users):
    rows = []
    for uu in users:
        rd = ratings_by[uu]; items = list(rd.keys())
        if len(items) < 2: continue
        kmax = min(len(items)-1, MAX_CTX)
        for _ in range(S):
            k = int(rng.integers(1, kmax+1)); ctx = list(rng.choice(items, k, replace=False)); rows.append((ctx, rd))
    L = max(len(c) for c, _ in rows); B = len(rows)
    ci = np.zeros((B, L), np.int64); cr = np.zeros((B, L), np.int64); cm = np.ones((B, L), bool)
    tgt = np.zeros((B, ni), np.float32); tm = np.zeros((B, ni), np.float32)
    for b, (ctx, rd) in enumerate(rows):
        seen = set()
        for j, e in enumerate(ctx): ci[b, j] = e; cr[b, j] = int(round(rd[e])); cm[b, j] = False; seen.add(e)
        for e, r in rd.items():
            if e not in seen: tgt[b, e] = r; tm[b, e] = 1.
    return (torch.from_numpy(ci), torch.from_numpy(cr), torch.from_numpy(cm), torch.from_numpy(tgt), torch.from_numpy(tm))

model = InstrumentV2(ni, D)
if MFINIT:                                          # import MF's clean item geometry
    assert D == Q.shape[1], f"D({D}) must equal MF dim({Q.shape[1]}) for MF-init"
    with torch.no_grad(): model.item.weight.copy_(torch.tensor(Q, dtype=torch.float32))
    print("MF-init item embeddings applied", flush=True)
opt = torch.optim.Adam(model.parameters(), 1e-3, weight_decay=1e-4)
sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=6)
def inst_rank(rev):
    if rev: ci=torch.tensor([[j for j,_ in rev]]); cr=torch.tensor([[int(round(r)) for _,r in rev]]); cm=torch.zeros_like(ci,dtype=torch.bool)
    else: ci=torch.zeros(1,1,dtype=torch.long); cr=torch.zeros(1,1,dtype=torch.long); cm=torch.ones(1,1,dtype=torch.bool)
    with torch.no_grad(): return model(ci, cr, cm)[0][0].numpy()
def inst_rate(rev):
    if rev: ci=torch.tensor([[j for j,_ in rev]]); cr=torch.tensor([[int(round(r)) for _,r in rev]]); cm=torch.zeros_like(ci,dtype=torch.bool)
    else: ci=torch.zeros(1,1,dtype=torch.long); cr=torch.zeros(1,1,dtype=torch.long); cm=torch.ones(1,1,dtype=torch.bool)
    with torch.no_grad(): return model(ci, cr, cm)[1][0].numpy()

def ndcg_full(pr, rel, unr):
    if not rel: return np.nan
    return np.mean([(1./np.log2(2+int((pr[unr]>=pr[ri]).sum())) if 1+int((pr[unr]>=pr[ri]).sum())<=10 else 0.) for ri in rel])
def val_ndcg():
    model.eval(); vs = []
    for uu in val_users:
        rd = ratings_by[uu]; items = list(rd.keys())
        if len(items) < 20: continue
        ii = items[:]; rng.shuffle(ii); test = set(ii[:max(5, len(ii)//3)]); pool = [x for x in ii if x not in test]
        rel = [j for j in test if rd[j] >= 4]; unr = np.array([j for j in range(ni) if j not in rd])
        if rel: vs.append(ndcg_full(inst_rank([(e, rd[e]) for e in pool[:VAL_K]]), rel, unr))
    return float(np.mean(vs))

best = -1; bstate = None; bad = 0; t0 = time.time()
for ep in range(EPOCHS):
    model.train(); rng.shuffle(tr_users)
    for s in range(0, len(tr_users), BS):
        ci, cr, cm, tgt, tm = make_batch(tr_users[s:s+BS]); opt.zero_grad()
        rank, rate = model(ci, cr, cm)
        pos = ((tgt >= 4) & (tm > 0)).float()
        sm = torch.zeros_like(rank); sm.scatter_add_(1, ci, (~cm).float()); seen = sm > 0.5
        logp = torch.log_softmax(rank.masked_fill(seen, -1e9), 1)
        npos = pos.sum(1).clamp(min=1); rper = -(logp*pos).sum(1)/npos; has = pos.sum(1) > 0
        rloss = rper[has].mean() if has.any() else torch.zeros((), requires_grad=True)
        mse = (((rate-tgt)**2)*tm).sum()/tm.sum()
        (rloss + mse).backward(); opt.step()
    vnd = val_ndcg(); sched.step(-vnd)
    if vnd > best+1e-4: best = vnd; bstate = {k: v.clone() for k, v in model.state_dict().items()}; bad = 0
    else: bad += 1
    if (ep+1) % 10 == 0: print(f"  ep{ep+1} valNDCG={vnd:.4f} best={best:.4f} ({time.time()-t0:.0f}s)", flush=True)
    if bad >= PATIENCE: print(f"  early stop ep{ep+1}", flush=True); break
model.load_state_dict(bstate); model.eval()
torch.save({'state': model.state_dict(), 'd': D, 'ni': ni, 'mu': mu, 'pool': 'tf', 'tie': TIE},
           f"C:/dev/phd/casper/data/movielens/.cache/checkpoints/instrument_v2_{DATASET.replace('-','')}_s{SEED}.pt")

# ---- tune WRMF (alpha, pop_w) on val ----
print("tuning WRMF on val...", flush=True)
best_w = (-1, None)
for alpha in [20., 40., 80.]:
    Qw, QtQ = train_wrmf(alpha)
    for pw in [0.0, 0.5, 1.0]:
        f = make_wrmf_predict(Qw, QtQ, alpha, pw); vs = []
        for uu in val_users:
            rd = ratings_by[uu]; items = list(rd.keys())
            if len(items) < 20: continue
            ii = items[:]; rng.shuffle(ii); test = set(ii[:max(5, len(ii)//3)]); pool = [x for x in ii if x not in test]
            rel = [j for j in test if rd[j] >= 4]; unr = np.array([j for j in range(ni) if j not in rd])
            if rel: vs.append(ndcg_full(f([(e, rd[e]) for e in pool[:VAL_K]]), rel, unr))
        v = float(np.mean(vs))
        if v > best_w[0]: best_w = (v, (alpha, pw, Qw, QtQ))
alpha, pw, Qw, QtQ = best_w[1]; wrmf_predict = make_wrmf_predict(Qw, QtQ, alpha, pw)
print(f"  WRMF best: alpha={alpha} pop_w={pw} valNDCG={best_w[0]:.4f}", flush=True)

# ============ COLD-TEST EVAL ============
RECS = {'popularity': lambda rev: logpop.copy(), 'MF': mf_predict, 'WRMF': wrmf_predict, 'INSTR': inst_rank}
T = 20
agg = {rn: {'ndcg': np.zeros(T+1), 'rec': np.zeros(T+1)} for rn in RECS}
rmse_mf = np.zeros(T+1); rmse_in = np.zeros(T+1); nrm = 0
# monotonicity (random true reveals) + disliked control on INSTR
mono = {K: [] for K in [0,1,2,3,5,8,12,20]}; dis_top10 = []; lik_top10 = []
used = 0
for cu, rd in cold_ratings.items():
    items_u = list(rd.keys())
    if len(items_u) < 10: continue
    iu = items_u[:]; rng.shuffle(iu); ncut = max(5, int(0.3*len(iu)))
    test = set(iu[:ncut]); rel = [j for j in test if rd[j] >= 4]
    unr = np.array([j for j in range(ni) if j not in rd]); order = [x for x in pop_order if x not in test]
    hi = np.array(list(test)); ht = np.array([rd[j] for j in test])
    for rn, pf in RECS.items():
        ans = []
        for t in range(T+1):
            pr = pf(ans)
            if rel:
                agg[rn]['ndcg'][t] += ndcg_full(pr, rel, unr)
                agg[rn]['rec'][t] += np.mean([1. if 1+int((pr[unr]>=pr[ri]).sum())<=10 else 0. for ri in rel])
            if t == T: break
            q = order[t] if t < len(order) else None
            if q is not None and q in rd: ans.append((q, rd[q]))
    # RMSE (MF rate vs INSTR rate head), popular interview
    for store, pf in [(rmse_mf, mf_predict), (rmse_in, inst_rate)]:
        ans = []
        for t in range(T+1):
            pr = pf(ans); store[t] += np.sqrt(np.mean((ht-np.clip(pr[hi],1,5))**2))
            if t == T: break
            q = order[t] if t < len(order) else None
            if q is not None and q in rd: ans.append((q, rd[q]))
    nrm += 1
    # monotonicity (random reveals) on INSTR
    if rel:
        pool = [x for x in items_u if x not in test]; rng.shuffle(pool)
        for K in mono: mono[K].append(ndcg_full(inst_rank([(e, rd[e]) for e in pool[:K]]), rel, unr))
        # disliked control: rank held-out DISLIKED among unrated at full reveal
        dis = [j for j in test if rd[j] <= 2]
        if dis:
            pr = inst_rank([(e, rd[e]) for e in pool[:12]])
            dis_top10.append(np.mean([1. if 1+int((pr[unr]>=pr[dj]).sum())<=10 else 0. for dj in dis]))
            lik_top10.append(np.mean([1. if 1+int((pr[unr]>=pr[ri]).sum())<=10 else 0. for ri in rel]))
    used += 1
for rn in RECS:
    agg[rn]['ndcg'] /= used; agg[rn]['rec'] /= used
rmse_mf /= nrm; rmse_in /= nrm
const_rmse = float(np.sqrt(np.mean([(r-mu)**2 for rd in cold_ratings.values() for r in rd.values()])))

print(f"\n===== COLD-TEST (n={used}), popular interview, full-catalogue =====", flush=True)
print("NDCG@10:  #q " + " ".join(f"{rn:>10}" for rn in RECS))
for t in [0,1,3,5,10,20]: print(f"  q{t:<2} " + " ".join(f"{agg[rn]['ndcg'][t]:>10.4f}" for rn in RECS))
print("Recall@10:#q " + " ".join(f"{rn:>10}" for rn in RECS))
for t in [0,1,3,5,10,20]: print(f"  q{t:<2} " + " ".join(f"{agg[rn]['rec'][t]:>10.4f}" for rn in RECS))
print(f"RMSE: MF {rmse_mf[0]:.3f}->{rmse_mf[20]:.3f} | INSTR {rmse_in[0]:.3f}->{rmse_in[20]:.3f} | const-baseline {const_rmse:.3f}")

print("\n===== GATE CARD =====", flush=True)
mc = [np.mean(mono[K]) for K in [0,1,2,3,5,8,12,20]]
g1 = all(mc[i+1] >= mc[i]-0.003 for i in range(len(mc)-1))
print(f"(1) MONOTONE (random reveals NDCG): {[round(x,4) for x in mc]} -> {'PASS' if g1 else 'FAIL'}")
g2 = agg['INSTR']['ndcg'][20] >= agg['WRMF']['ndcg'][20]
print(f"(2) RANKING vs tuned WRMF @q20: INSTR {agg['INSTR']['ndcg'][20]:.4f} vs WRMF {agg['WRMF']['ndcg'][20]:.4f} -> {'PASS' if g2 else 'FAIL'}")
g3 = rmse_in[20] <= const_rmse
print(f"(3) RMSE calibration: INSTR {rmse_in[20]:.3f} <= const {const_rmse:.3f} -> {'PASS' if g3 else 'FAIL'}")
g4 = (np.mean(dis_top10) < np.mean(lik_top10)) if dis_top10 else False
print(f"(4) DISLIKED-control: disliked top10 {np.mean(dis_top10):.3f} < liked top10 {np.mean(lik_top10):.3f} -> {'PASS' if g4 else 'FAIL'}")
print(f"\nVERDICT: {'ALL PASS' if all([g1,g2,g3,g4]) else 'FAILS — rethink architecture'}")
