"""
Replicate Deep Rating Elicitation (DRE, Kim et al. 2024) paradigm on ML-100k:
FIXED seed set of k items + co-trained decoder. Learned seed selection
(Gumbel-softmax) vs POP++ (popular seeds) vs RAN++ (random seeds) vs MOSTPOP.
Metric: NDCG@10 / P@10 on held-out (full-catalogue, cold-test users).
"""
import os, time, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
base='C:/dev/phd/casper/data/movielens/ml-100k'
K=int(os.environ.get('K',20)); EP=int(os.environ.get('EP',60)); H=128; N_COLD=200
DIV=float(os.environ.get('DIV',1.0))   # diversity penalty: prevent seed-row collapse
rng=np.random.default_rng(0); torch.manual_seed(0)
U,I,R=[],[],[]
with open(f'{base}/u.data') as f:
    for line in f:
        a=line.strip().split('\t'); U.append(int(a[0])); I.append(int(a[1])); R.append(float(a[2]))
U=np.array(U); I=np.array(I); R=np.array(R,np.float32)
uids={x:k for k,x in enumerate(np.unique(U))}; iids={x:k for k,x in enumerate(np.unique(I))}
u=np.array([uids[x] for x in U]); i=np.array([iids[x] for x in I]); nu,ni=len(uids),len(iids)
cold=set(np.random.default_rng(0).choice(nu,N_COLD,replace=False).tolist()); warm=[x for x in range(nu) if x not in cold]
Rm=np.full((nu,ni),np.nan,np.float32)
for k in range(len(u)): Rm[u[k],i[k]]=R[k]
cnt=np.array([(~np.isnan(Rm[warm,j])).sum() for j in range(ni)]); pop_topk=list(np.argsort(-cnt)[:K])
mu=np.nanmean(R)
Rfill=np.nan_to_num(Rm,nan=0.0); Mobs=(~np.isnan(Rm)).astype(np.float32)
Liked=(((Rm>=4)&(~np.isnan(Rm))).astype(np.float32))
Wtr=np.array(warm)
def ndcg_p(pred, rd, test, rel, unr):
    # rank held-out liked among unrated; NDCG@10 + P@10
    nd=0.; hit=0
    for ri in rel:
        rank=1+int((pred[unr]>=pred[ri]).sum())
        if rank<=10: nd+=1./np.log2(rank+1)
    nd/=len(rel)
    top=unr[np.argsort(-pred[unr])[:10]]  # precision proxy: of top-10 unrated, frac that are held-out likes
    p=len(set(top.tolist())&set(rel))/10.
    return nd, p
# cold cases
cases=[]
for cu in cold:
    rd={int(j):float(Rm[cu,j]) for j in np.where(~np.isnan(Rm[cu]))[0]}
    items=list(rd.keys())
    if len(items)<10: continue
    ii=items[:]; rng.shuffle(ii); test=set(ii[:max(5,int(0.3*len(ii)))]); rel=[j for j in test if rd[j]>=4]
    if not rel: continue
    cases.append((rd,test,rel,np.array([j for j in range(ni) if j not in rd])))
print(f"cold cases {len(cases)}, K={K}",flush=True)

class Decoder(nn.Module):
    def __init__(s):
        super().__init__(); s.net=nn.Sequential(nn.Linear(K,H),nn.ReLU(),nn.Linear(H,H),nn.ReLU(),nn.Linear(H,ni))
        s.bias=nn.Parameter(torch.zeros(ni))   # item bias (~popularity) -> proper ranking
    def forward(s,x): return s.net(x)+s.bias
def train_fixed(seed_items, name):
    seed=np.array(seed_items); dec=Decoder(); opt=torch.optim.Adam(dec.parameters(),1e-3,weight_decay=1e-5)
    Xin=torch.tensor(Rfill[Wtr][:,seed]); Lk=torch.tensor(Liked[Wtr])
    sm=torch.zeros(ni,dtype=torch.bool); sm[seed]=True
    for ep in range(EP):
        idx=torch.randperm(len(Wtr))
        for s in range(0,len(idx),256):
            b=idx[s:s+256]; opt.zero_grad(); pred=dec(Xin[b])
            pm=Lk[b].clone(); pm[:,sm]=0
            logp=torch.log_softmax(pred.masked_fill(sm,-1e9),1)
            npos=pm.sum(1).clamp(min=1); per=-(logp*pm).sum(1)/npos; has=pm.sum(1)>0
            loss=per[has].mean() if has.any() else (pred*0).sum()
            loss.backward(); opt.step()
    dec.eval()
    nd=p=0.
    for (rd,test,rel,unr) in cases:
        x=torch.tensor([[rd.get(int(j),0.0) for j in seed]])
        with torch.no_grad(): pr=dec(x)[0].numpy()
        a,b=ndcg_p(pr,rd,test,rel,unr); nd+=a; p+=b
    nd/=len(cases); p/=len(cases); print(f"  {name:<10} NDCG@10={nd:.4f} P@10={p:.4f}",flush=True); return nd
# DRE: learned seed selection (Gumbel-softmax) + decoder, co-trained
class DRE(nn.Module):
    def __init__(s):
        super().__init__(); s.logits=nn.Parameter(torch.randn(K,ni)*0.01); s.dec=Decoder()
    def forward(s, Rb, tau):
        Sg=F.gumbel_softmax(s.logits, tau=tau, hard=True, dim=1)  # STRAIGHT-THROUGH hard (matches test-time single-item elicitation)
        x=Rb@Sg.t()                     # [B,K] elicited ratings on hard-selected seeds
        return s.dec(x)
    def div_pen(s):                     # penalize >1 selection row piling on the same item -> distinct seeds
        col=torch.softmax(s.logits,1).sum(0); return ((col-1.0).clamp(min=0)**2).sum()
def train_dre():
    m=DRE(); opt=torch.optim.Adam(m.parameters(),1e-3,weight_decay=1e-5)
    Rb_all=torch.tensor(Rfill[Wtr]); Lk=torch.tensor(Liked[Wtr])
    for ep in range(EP):
        tau=max(0.4,2.0*np.exp(-0.05*ep)); idx=torch.randperm(len(Wtr))
        for s in range(0,len(idx),256):
            b=idx[s:s+256]; opt.zero_grad(); pred=m(Rb_all[b],tau)
            pm=Lk[b]; logp=torch.log_softmax(pred,1)
            npos=pm.sum(1).clamp(min=1); per=-(logp*pm).sum(1)/npos; has=pm.sum(1)>0
            loss=(per[has].mean() if has.any() else (pred*0).sum())+DIV*m.div_pen()
            loss.backward(); opt.step()
    m.eval()
    seed=list(dict.fromkeys(m.logits.argmax(1).tolist()))   # hard seeds (dedup)
    while len(seed)<K:                                       # pad if dup
        for j in np.argsort(-cnt):
            if j not in seed: seed.append(int(j)); break
    seed=seed[:K]
    nd=p=0.
    for (rd,test,rel,unr) in cases:
        x=torch.tensor([[rd.get(int(j),0.0) for j in seed]])
        with torch.no_grad(): pr=m.dec(x)[0].numpy()
        a,b=ndcg_p(pr,rd,test,rel,unr); nd+=a; p+=b
    nd/=len(cases); p/=len(cases)
    print(f"  {'DRE':<10} NDCG@10={nd:.4f} P@10={p:.4f}  (distinct seeds {len(set(seed))})",flush=True); return nd
print("=== DRE replication (fixed-seed elicitation + decoder) ===",flush=True)
# MOSTPOP (no elicitation)
ndp=p=0.
for (rd,test,rel,unr) in cases:
    pr=cnt.astype(np.float32); a,b=ndcg_p(pr,rd,test,rel,unr); ndp+=a; p+=b
print(f"  {'MOSTPOP':<10} NDCG@10={ndp/len(cases):.4f} P@10={p/len(cases):.4f}  (no elicitation)",flush=True)
train_fixed(list(rng.choice(ni,K,replace=False)),'RAN++')
train_fixed(pop_topk,'POP++')
train_dre()
