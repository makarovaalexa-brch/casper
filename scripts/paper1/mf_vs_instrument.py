"""
Honest side-by-side on ONE identical protocol: MF vs the fixed instrument.
Same ML-100k warm/cold split, same cold-start interview (popular order),
SAME metrics computed identically for both: RMSE + FULL-CATALOGUE NDCG@10/Recall@10.
Answers: is the instrument actually a poor recommender, or was 0.044 just the
full-catalogue scale?
"""
import os, sys, time
import numpy as np, torch, torch.nn as nn

base = 'C:/dev/phd/casper/data/movielens/ml-100k'
CKPT = 'C:/dev/phd/casper/data/movielens/.cache/checkpoints/instrument_ml100k.pt'
SEED = 42; D = 32; MF_EPOCHS = 30; MF_LR = 0.008; MF_REG = 0.05; REGF = 4.0; N_COLD = 200; T = 20
torch.manual_seed(SEED); np.random.seed(SEED); rng = np.random.default_rng(SEED)

U, I, R = [], [], []
with open(f'{base}/u.data') as f:
    for line in f:
        a = line.strip().split('\t'); U.append(int(a[0])); I.append(int(a[1])); R.append(float(a[2]))
U = np.array(U); I = np.array(I); R = np.array(R, np.float32)
uids = {x: k for k, x in enumerate(np.unique(U))}; iids = {x: k for k, x in enumerate(np.unique(I))}
u = np.array([uids[x] for x in U]); i = np.array([iids[x] for x in I]); nu, ni = len(uids), len(iids)
cold = set(rng.choice(nu, N_COLD, replace=False).tolist())
wm = np.array([uu not in cold for uu in u]); ut, it, rt = u[wm], i[wm], R[wm]; mu = float(rt.mean())
cnt = np.bincount(it, minlength=ni); pop_order = list(np.argsort(-cnt))
ratings_by = {}
for k in range(len(u)): ratings_by.setdefault(u[k], {})[i[k]] = R[k]
cold_ratings = {uu: ratings_by[uu] for uu in ratings_by if uu in cold}

# ---- MF on warm ----
bu = np.zeros(nu); bi = np.zeros(ni); P = 0.1*rng.standard_normal((nu, D)); Q = 0.1*rng.standard_normal((ni, D))
for ep in range(MF_EPOCHS):
    o = rng.permutation(len(ut)); lr = MF_LR/(1+0.05*ep)
    for s in range(0, len(o), 20000):
        b = o[s:s+20000]; uu, ii, rr = ut[b], it[b], rt[b]
        e = (rr-(mu+bu[uu]+bi[ii]+np.sum(P[uu]*Q[ii],1))).astype(np.float64)
        np.add.at(bu, uu, lr*(e-MF_REG*bu[uu])); np.add.at(bi, ii, lr*(e-MF_REG*bi[ii]))
        np.add.at(P, uu, lr*(e[:,None]*Q[ii]-MF_REG*P[uu])); np.add.at(Q, ii, lr*(e[:,None]*P[uu]-MF_REG*Q[ii]))
def mf_predict(rev):
    if not rev: return mu+bi
    A = np.array([[1.]+Q[j].tolist() for j,_ in rev]); y = np.array([r-mu-bi[j] for j,r in rev])
    x = np.linalg.solve(A.T@A+REGF*np.eye(D+1), A.T@y); return mu+x[0]+bi+Q@x[1:]
print("MF trained.", flush=True)

# ---- WRMF (implicit ALS, ranking-trained MF) on warm: FAIR ranking baseline ----
WD = 32; ALPHA = 40.0; WREG = 0.1; WITERS = 15
Rmat = np.full((nu, ni), np.nan)
for k in np.where(wm)[0]: Rmat[u[k], i[k]] = R[k]
warm_idx = [uu for uu in range(nu) if uu not in cold]
posM = (Rmat >= 4); obsM = ~np.isnan(Rmat)
Pw = 0.01*rng.standard_normal((nu, WD)); Qw = 0.01*rng.standard_normal((ni, WD))
uo = {uu: np.where(obsM[uu])[0] for uu in warm_idx}; up = {uu: np.where(posM[uu])[0] for uu in warm_idx}
io = [np.where(obsM[warm_idx][:, j])[0] for j in range(ni)]  # indices into warm_idx
warm_arr = np.array(warm_idx)
ip = [np.where(posM[warm_idx][:, j])[0] for j in range(ni)]
Idw = WREG*np.eye(WD)
for it in range(WITERS):
    QtQ = Qw.T@Qw
    for uu in warm_idx:
        oi = uo[uu]; A = QtQ + Idw + ALPHA*(Qw[oi].T@Qw[oi])
        rhs = (1+ALPHA)*Qw[up[uu]].sum(0) if len(up[uu]) else np.zeros(WD)
        Pw[uu] = np.linalg.solve(A, rhs)
    PtP = Pw[warm_arr].T@Pw[warm_arr]
    for j in range(ni):
        wu = warm_arr[io[j]]; A = PtP + Idw + ALPHA*(Pw[wu].T@Pw[wu])
        wp = warm_arr[ip[j]]; rhs = (1+ALPHA)*Pw[wp].sum(0) if len(wp) else np.zeros(WD)
        Qw[j] = np.linalg.solve(A, rhs)
QwtQw = Qw.T@Qw
def wrmf_predict(rev):
    A = QwtQw + Idw; rhs = np.zeros(WD)
    for j, r in rev:
        q = Qw[j]; A = A + ALPHA*np.outer(q, q)
        if r >= 4: rhs = rhs + (1+ALPHA)*q
    return Qw @ np.linalg.solve(A, rhs)
print("WRMF trained.", flush=True)

# ---- load fixed instrument ----
ck = torch.load(CKPT); mu_i = ck['mu']; Di = ck['d']; POOL = ck.get('pool', 'tf'); DROP = 0.2
class SetRec(nn.Module):
    def __init__(s, ni, d):
        super().__init__(); s.item=nn.Embedding(ni,d); s.rate=nn.Embedding(6,d)
        s.enc=nn.Sequential(nn.Linear(d,d),nn.ReLU(),nn.Dropout(DROP),nn.Linear(d,d))
        s.post=nn.Sequential(nn.Linear(d,d),nn.ReLU(),nn.Dropout(DROP),nn.Linear(d,d))
        s.head=nn.Linear(d,ni); s.bi=nn.Parameter(torch.zeros(ni)); s.u0=nn.Parameter(torch.zeros(d))
        s.attn_q=nn.Parameter(torch.randn(d)*0.1); s.dscale=d**0.5
        if POOL=='tf':
            s.cls=nn.Parameter(torch.randn(d)*0.1)
            s.tf=nn.TransformerEncoder(nn.TransformerEncoderLayer(d,nhead=4,dim_feedforward=2*d,dropout=DROP,batch_first=True),num_layers=2)
    def uh(s, items, rates, mask):
        tok=s.item(items)+s.rate(rates)
        if POOL=='tf':
            B=tok.shape[0]; x=torch.cat([s.cls.expand(B,1,-1),tok],1)
            pad=torch.cat([torch.zeros(B,1,dtype=torch.bool),mask],1)
            return s.post(s.tf(x,src_key_padding_mask=pad)[:,0])
        h=s.enc(tok)
        if POOL=='attn':
            sc=(h@s.attn_q/s.dscale).masked_fill(mask,-1e9); w=torch.softmax(sc,1).unsqueeze(-1); pooled=(w*h).sum(1)
        else:
            hm=h.masked_fill(mask.unsqueeze(-1),0.); n=(~mask).sum(1,keepdim=True).clamp(min=1); pooled=hm.sum(1)/n
        nrev=(~mask).sum(1); empty=(nrev==0); out=s.post(pooled)
        return torch.where(empty.unsqueeze(1), s.u0.unsqueeze(0).expand_as(out), out)
    def forward(s, items, rates, mask): return mu_i + s.bi + s.head(s.uh(items, rates, mask))
inst = SetRec(ni, Di); inst.load_state_dict(ck['state']); inst.eval()
def inst_predict(rev):
    if rev:
        ci=torch.tensor([[j for j,_ in rev]]); cr=torch.tensor([[int(round(r)) for _,r in rev]]); cm=torch.zeros_like(ci,dtype=torch.bool)
    else:
        ci=torch.zeros(1,1,dtype=torch.long); cr=torch.zeros(1,1,dtype=torch.long); cm=torch.ones(1,1,dtype=torch.bool)
    with torch.no_grad(): return inst(ci,cr,cm)[0].numpy()

# ---- eval: identical full-catalogue protocol, popular interview ----
RECS = {'MF': mf_predict, 'WRMF': wrmf_predict, 'INSTR': inst_predict}
agg = {rn: {'rmse': np.zeros(T+1), 'ndcg': np.zeros(T+1), 'rec': np.zeros(T+1)} for rn in RECS}
used = 0
for cu, rd in cold_ratings.items():
    items_u = list(rd.keys())
    if len(items_u) < 10: continue
    iu = items_u[:]; rng.shuffle(iu); ncut = max(5, int(0.3*len(iu)))
    test = set(iu[:ncut]); hi = np.array(list(test)); ht = np.array([rd[j] for j in test])
    rel = [j for j in test if rd[j] >= 4]; unr = np.array([j for j in range(ni) if j not in rd])
    order = [x for x in pop_order if x not in test]
    for rn, pf in RECS.items():
        ans = []
        for t in range(T+1):
            pr = pf(ans)
            agg[rn]['rmse'][t] += np.sqrt(np.mean((ht-np.clip(pr[hi],1,5))**2))
            if rel:
                nd = rc = 0.
                for ri in rel:
                    rank = 1 + int((pr[unr] >= pr[ri]).sum())
                    if rank <= 10: nd += 1./np.log2(rank+1); rc += 1.
                agg[rn]['ndcg'][t] += nd/len(rel); agg[rn]['rec'][t] += rc/len(rel)
            if t == T: break
            q = order[t] if t < len(order) else None
            if q is not None and q in rd: ans.append((q, rd[q]))
    used += 1
for rn in RECS:
    for k in agg[rn]: agg[rn][k] /= used
print(f"evaluated {used} cold users; full-catalogue ranking ({ni} items), popular interview\n", flush=True)
a = agg
print("=== NDCG@10 (higher better) — the fair ranking gate ===")
print(f"{'#q':>3} {'MF':>9} {'WRMF':>9} {'INSTR':>9}")
for t in [0,1,3,5,10,20]:
    print(f"{t:>3} {a['MF']['ndcg'][t]:>9.4f} {a['WRMF']['ndcg'][t]:>9.4f} {a['INSTR']['ndcg'][t]:>9.4f}")
print("\n=== Recall@10 (higher better) ===")
print(f"{'#q':>3} {'MF':>9} {'WRMF':>9} {'INSTR':>9}")
for t in [0,1,3,5,10,20]:
    print(f"{t:>3} {a['MF']['rec'][t]:>9.4f} {a['WRMF']['rec'][t]:>9.4f} {a['INSTR']['rec'][t]:>9.4f}")
print("\n=== RMSE (lower better; WRMF predicts preference not ratings -> N/A) ===")
print(f"{'#q':>3} {'MF':>9} {'INSTR':>9}")
for t in [0,1,3,5,10,20]:
    print(f"{t:>3} {a['MF']['rmse'][t]:>9.4f} {a['INSTR']['rmse'][t]:>9.4f}")
print(f"\nrandom-guess full-catalogue NDCG@10 ~ {10/ni*1/np.log2(2):.4f} (scale reference)")
