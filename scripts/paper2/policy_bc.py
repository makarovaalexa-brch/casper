"""
PAPER B, first experiment: can a LEARNED policy capture the oracle headroom (0.30->0.49) that heuristics miss?
Approach = imitation / behaviour cloning of the greedy oracle (Wolpertinger-style continuous action snapped to a
candidate pool). Frozen calibrated instrument (Q_svd/bi_svd). State = current folded taste vector u_t (the
instrument's sufficient statistic). Action = a query vector; candidates scored by inner product; pick argmax.

Pipeline: (1) generate oracle trajectories on TRAIN users over a fixed candidate pool; (2) BC-train an MLP
state->query head with cross-entropy toward the oracle's choice; (3) evaluate on TEST: learned policy vs pool-oracle
(ceiling) vs HELF vs random.  If learned >> heuristics, the headroom is capturable -> Paper B has a result.
"""
import os, numpy as np, torch, torch.nn as nn, scipy.linalg as sla
base='C:/dev/phd/casper/data/movielens'; ml=f'{base}/ml-1m'; D=64; LAM=5.0
T=int(os.environ.get('T',8)); POOL=int(os.environ.get('POOL',200)); NTR=int(os.environ.get('NTR',700)); rng=np.random.default_rng(0)
U,I,R=[],[],[]
with open(f'{ml}/ratings.dat') as f:
    for line in f:
        a=line.strip().split('::'); U.append(int(a[0])); I.append(int(a[1])); R.append(float(a[2]))
U=np.array(U);I=np.array(I);R=np.array(R,np.float32)
uids={x:k for k,x in enumerate(np.unique(U))}; iids={x:k for k,x in enumerate(np.unique(I))}; nu,ni=len(uids),len(iids)
uu=np.array([uids[x] for x in U]); ii=np.array([iids[x] for x in I])
rat_by_u={}
for k in range(len(uu)): rat_by_u.setdefault(uu[k],[]).append((ii[k],float(R[k])))
likes_by_u={x:[j for j,r in v if r>=4] for x,v in rat_by_u.items()}
keep=[x for x in range(nu) if len(likes_by_u.get(x,[]))>=5]; rng.shuffle(keep); n=len(keep)
trU=keep[:int(0.8*n)]; te=keep[int(0.9*n):]
cnt=np.zeros(ni)
for x in trU:
    for j in likes_by_u.get(x,[]): cnt[j]+=1
popb=np.log(cnt+1.0).astype(np.float32)
sm=c=0.
for x in trU:
    for j,r in rat_by_u[x]: sm+=r; c+=1
mu=sm/c
Q=np.load(f'{base}/.cache/Q_svd.npy'); bi=np.load(f'{base}/.cache/bi_svd.npy')
pool=np.array(np.argsort(-cnt)[:POOL]); Qp=Q[pool]; poolpos={int(p):k for k,p in enumerate(pool)}
helf_cnt=cnt.copy()  # for HELF we need entropy; reuse popularity order for a simple 'pop within pool' baseline + HELF below
H5=np.zeros((ni,5)); trset=set(trU)
for k in range(len(uu)):
    if uu[k] in trset: H5[ii[k],min(max(int(round(R[k])),1),5)-1]+=1
pm=H5/np.clip(H5.sum(1,keepdims=True),1,None); ent=-(pm*np.log(pm+1e-12)).sum(1)
lf=np.log(cnt+1)/np.log(cnt.max()+1); Hn=ent/np.log(5); helf=2*lf*Hn/(lf+Hn+1e-9)
def foldin(F,y): F=np.array(F); return np.linalg.solve(F.T@F+LAM*np.eye(D),F.T@np.array(y,np.float32)) if len(F) else np.zeros(D)
def ndcg(score,rel,excl):
    s=score.copy(); s[list(excl)]=-1e9; top=np.argsort(-s)[:10]; rs=set(rel)
    return sum(1./np.log2(p+2) for p,t in enumerate(top) if t in rs)/(sum(1./np.log2(p+2) for p in range(min(10,len(rel))))+1e-12)
_rs=np.random.default_rng(123)
def split(x):
    items=list(dict(rat_by_u[x]))
    if len(items)<6: return None
    il=items[:]; _rs.shuffle(il); return set(il[:len(il)//2]), set(il[len(il)//2:])  # test, profile
# ---------- 1) oracle trajectories on TRAIN users ----------
def gen_oracle(users, maxu):
    States=[]; Acts=[]
    for x in users[:maxu]:
        sp=split(x)
        if sp is None: continue
        test,prof=sp; rd=dict(rat_by_u[x]); tlike=set(j for j in test if rd[j]>=4)
        if not tlike: continue
        F=[];y=[];asked=set()
        for t in range(T):
            u=foldin(F,y)
            best=None
            for k,j in enumerate(pool):
                if k in asked: continue
                jj=int(j); tr=(rd[jj]-mu-bi[jj]) if jj in prof else None
                uu2=foldin(F+([Q[jj]] if tr is not None else []), y+([tr] if tr is not None else []))
                a=ndcg(popb+Q@uu2,tlike,prof|{jj})
                if best is None or a>best[0]: best=(a,k,jj,tr)
            _,k,jj,tr=best; States.append(u.astype(np.float32)); Acts.append(k)
            asked.add(k)
            if tr is not None: F.append(Q[jj]); y.append(tr)
    return np.array(States,np.float32), np.array(Acts,np.int64)
print(f"generating oracle trajectories on {min(NTR,len(trU))} train users (pool={POOL}, T={T})...",flush=True)
S,A=gen_oracle(trU,NTR); print(f"  {len(S)} (state,action) pairs",flush=True)
# ---------- 2) BC train: MLP state->query; scores = Qp @ query; CE vs oracle action ----------
Qpt=torch.tensor(Qp)
net=nn.Sequential(nn.Linear(D,256),nn.ReLU(),nn.Linear(256,256),nn.ReLU(),nn.Linear(256,D))
opt=torch.optim.Adam(net.parameters(),lr=1e-3,weight_decay=1e-5)
St=torch.tensor(S); At=torch.tensor(A); lossf=nn.CrossEntropyLoss()
for ep in range(int(os.environ.get('EP',40))):
    perm=torch.randperm(len(St))
    tot=0.
    for b0 in range(0,len(St),256):
        bi_=perm[b0:b0+256]; qv=net(St[bi_]); logits=qv@Qpt.t(); l=lossf(logits,At[bi_])
        opt.zero_grad(); l.backward(); opt.step(); tot+=l.item()*len(bi_)
    if (ep+1)%10==0:
        with torch.no_grad(): acc=(net(St)@Qpt.t()).argmax(1).eq(At).float().mean().item()
        print(f"  ep{ep+1} loss={tot/len(St):.3f} oracle-match-acc={acc:.3f}",flush=True)
net.eval()
# ---------- 3) evaluate on TEST ----------
NEVAL=int(os.environ.get('NEVAL',250))
def run(kind):
    ND=np.zeros(T+1); m=0
    for x in te[:NEVAL]:
        sp=split(x)
        if sp is None: continue
        test,prof=sp; rd=dict(rat_by_u[x]); tlike=set(j for j in test if rd[j]>=4)
        if not tlike: continue
        nd=np.zeros(T+1); nd[0]=ndcg(popb.copy(),tlike,prof)
        F=[];y=[];asked=set()
        for t in range(1,T+1):
            u=foldin(F,y)
            if kind=='policy':
                with torch.no_grad(): sc=(net(torch.tensor(u.astype(np.float32)))@Qpt.t()).numpy()
                for k in asked: sc[k]=-1e9;
                k=int(sc.argmax())
            elif kind=='oracle':
                best=None
                for k2,j in enumerate(pool):
                    if k2 in asked: continue
                    jj=int(j); tr=(rd[jj]-mu-bi[jj]) if jj in prof else None
                    a=ndcg(popb+Q@foldin(F+([Q[jj]] if tr is not None else []),y+([tr] if tr is not None else [])),tlike,prof|{jj})
                    if best is None or a>best[0]: best=(a,k2)
                k=best[1]
            elif kind=='helf': k=int(max((k2 for k2 in range(len(pool)) if k2 not in asked),key=lambda k2:helf[pool[k2]]))
            else: rem=[k2 for k2 in range(len(pool)) if k2 not in asked]; k=int(rng.choice(rem))
            asked.add(k); jj=int(pool[k]); tr=(rd[jj]-mu-bi[jj]) if jj in prof else None
            if tr is not None: F.append(Q[jj]); y.append(tr)
            nd[t]=ndcg(popb+Q@foldin(F,y),tlike,prof|{jj for jj in [int(pool[a]) for a in asked]})
        ND+=nd; m+=1
    return ND/m,m
print(f"\n=== TEST (pool={POOL}, {len(te)} users) ===",flush=True)
for kind in ['random','helf','policy','oracle']:
    nd,m=run(kind); print(f"{kind:<8}: "+" ".join(f"{v:.3f}" for v in nd)+f" | delta {nd[T]-nd[0]:+.3f}",flush=True)
