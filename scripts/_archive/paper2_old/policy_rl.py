"""
PAPER B core: BC-pretrain on the best non-oracle policy (HELF) then RL-finetune (REINFORCE) on the REAL reward.
Why this can beat imitation: BC of the privileged oracle fails (its actions need held-out info -> imitation gap);
RL instead optimises the reward from OBSERVABLE state, so it learns the best *realisable* state->action mapping,
marginalising the privileged info. Reward is computed on TRAIN users' held-out split (legitimate training signal);
the policy itself only ever sees the observable taste state u_t, so it is realisable at test time.

Frozen calibrated instrument (Q_svd/bi_svd). State = u_t. Action = query vector (Wolpertinger), candidates = fixed
pool scored by inner product, masked by asked. Eval: random / HELF / BC-init / RL vs the privileged oracle ceiling.
"""
import os, numpy as np, torch, torch.nn as nn, scipy.linalg as sla
base='C:/dev/phd/casper/data/movielens'; ml=f'{base}/ml-1m'; D=64; LAM=5.0
T=int(os.environ.get('T',8)); POOL=int(os.environ.get('POOL',120)); rng=np.random.default_rng(0); torch.manual_seed(0)
NTR=int(os.environ.get('NTR',400)); NEVAL=int(os.environ.get('NEVAL',250))
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
title={}
with open(f'{ml}/movies.dat',encoding='latin-1') as f:
    for line in f:
        pp=line.strip().split('::')
        if int(pp[0]) in iids: title[iids[int(pp[0])]]=pp[1]
pool=np.array(np.argsort(-cnt)[:POOL]); Qp=Q[pool]; Qpt=torch.tensor(Qp)
H5=np.zeros((ni,5)); trset=set(trU)
for k in range(len(uu)):
    if uu[k] in trset: H5[ii[k],min(max(int(round(R[k])),1),5)-1]+=1
pm=H5/np.clip(H5.sum(1,keepdims=True),1,None); ent=-(pm*np.log(pm+1e-12)).sum(1)
lf=np.log(cnt+1)/np.log(cnt.max()+1); Hn=ent/np.log(5); helf=2*lf*Hn/(lf+Hn+1e-9)
helf_pool=helf[pool]                                   # HELF score per pool slot (teacher)
def foldin(F,y): F=np.array(F); return np.linalg.solve(F.T@F+LAM*np.eye(D),F.T@np.array(y,np.float32)) if len(F) else np.zeros(D)
_W=1./np.log2(np.arange(2,12))
def ndcg(score,rel,excl):
    s=score.copy(); s[list(excl)]=-1e9; top=np.argpartition(-s,10)[:10]; top=top[np.argsort(-s[top])]
    return sum(_W[p] for p,t in enumerate(top) if int(t) in rel)/(_W[:min(10,len(rel))].sum()+1e-12)
SPL={}; _rs=np.random.default_rng(123)            # FIXED per-user split, precomputed once, SHARED across all methods
for _x in keep:
    _items=list(dict(rat_by_u[_x]))
    if len(_items)>=6: _il=_items[:]; _rs.shuffle(_il); SPL[_x]=(set(_il[:len(_il)//2]), set(_il[len(_il)//2:]))
def split(x): return SPL.get(x)
class Net(nn.Module):
    def __init__(s): super().__init__(); s.f=nn.Sequential(nn.Linear(D,256),nn.ReLU(),nn.Linear(256,256),nn.ReLU(),nn.Linear(256,D))
    def forward(s,u): return s.f(u)@Qpt.t()             # logits over pool
net=Net(); opt=torch.optim.Adam(net.parameters(),lr=3e-4,weight_decay=1e-5)
# ---------- BC pretrain on HELF (teacher) ----------
def gen_helf(users,maxu):
    S=[];A=[]
    for x in users[:maxu]:
        sp=split(x)
        if sp is None: continue
        test,prof=sp; rd=dict(rat_by_u[x]); asked=set(); F=[];y=[]
        for t in range(T):
            u=foldin(F,y); cand=[k for k in range(POOL) if k not in asked]
            k=max(cand,key=lambda k:helf_pool[k]); S.append(u.astype(np.float32)); A.append(k); asked.add(k)
            jj=int(pool[k]); tr=(rd[jj]-mu-bi[jj]) if jj in prof else None
            if tr is not None: F.append(Q[jj]); y.append(tr)
    return torch.tensor(np.array(S,np.float32)), torch.tensor(np.array(A,np.int64))
Sb,Ab=gen_helf(trU,NTR); lossf=nn.CrossEntropyLoss()
for ep in range(int(os.environ.get('BCEP',30))):
    perm=torch.randperm(len(Sb))
    for b0 in range(0,len(Sb),256):
        bidx=perm[b0:b0+256]; l=lossf(net(Sb[bidx]),Ab[bidx]); opt.zero_grad(); l.backward(); opt.step()
print(f"BC pretrain done on HELF ({len(Sb)} pairs).",flush=True)
# ---------- RL finetune (REINFORCE with baseline + entropy) ----------
optrl=torch.optim.Adam(net.parameters(),lr=1e-4,weight_decay=1e-5)
def rollout(x, greedy=False):
    sp=split(x)
    if sp is None: return None
    test,prof=sp; rd=dict(rat_by_u[x]); tlike=set(j for j in test if rd[j]>=4)
    if not tlike: return None
    F=[];y=[];asked=set(); logps=[]; ents=[]; nd0=ndcg(popb.copy(),tlike,prof)
    for t in range(T):
        u=torch.tensor(foldin(F,y).astype(np.float32)); logits=net(u)
        mask=torch.full((POOL,),-1e9); idx=[k for k in range(POOL) if k not in asked]; mask[idx]=0.
        dist=torch.distributions.Categorical(logits=logits+mask)
        k=int(dist.probs.argmax()) if greedy else int(dist.sample())
        logps.append(dist.log_prob(torch.tensor(k))); ents.append(dist.entropy()); asked.add(k)
        jj=int(pool[k]); tr=(rd[jj]-mu-bi[jj]) if jj in prof else None
        if tr is not None: F.append(Q[jj]); y.append(tr)
    rel=tlike; excl=prof|{int(pool[k]) for k in asked}
    ndf=ndcg(popb+Q@foldin(F,y),rel,excl)
    return ndf-nd0, torch.stack(logps).sum(), torch.stack(ents).sum()
RLEP=int(os.environ.get('RLEP',8)); baseline=0.0
for ep in range(RLEP):
    rng.shuffle(trU); rewards=[]; loss=0.; cntb=0
    for x in trU[:NTR]:
        r=rollout(x)
        if r is None: continue
        adv=r[0]-baseline; loss=loss - adv*r[1] - 0.01*r[2]; rewards.append(r[0]); cntb+=1
        if cntb%64==0:
            optrl.zero_grad(); (loss/64).backward(); optrl.step(); loss=0.
    if cntb%64!=0 and isinstance(loss,torch.Tensor): optrl.zero_grad(); loss.backward(); optrl.step()
    baseline=0.9*baseline+0.1*np.mean(rewards)
    print(f"  RL ep{ep+1} mean train reward(deltaNDCG)={np.mean(rewards):+.4f}",flush=True)
# ---------- eval ----------
def evalp(kind):
    ND=np.zeros(T+1); m=0
    for x in te[:NEVAL]:
        sp=split(x)
        if sp is None: continue
        test,prof=sp; rd=dict(rat_by_u[x]); tlike=set(j for j in test if rd[j]>=4)
        if not tlike: continue
        nd=np.zeros(T+1); nd[0]=ndcg(popb.copy(),tlike,prof); F=[];y=[];asked=set(); tr_=[]
        for t in range(1,T+1):
            u=foldin(F,y)
            if kind in ('bc','rl'):
                with torch.no_grad(): logits=net(torch.tensor(u.astype(np.float32))).numpy()
                for k in asked: logits[k]=-1e9
                k=int(logits.argmax())
            elif kind=='helf': k=max((k for k in range(POOL) if k not in asked),key=lambda k:helf_pool[k])
            elif kind=='oracle':
                best=None
                for k2 in range(POOL):
                    if k2 in asked: continue
                    jj=int(pool[k2]); tr=(rd[jj]-mu-bi[jj]) if jj in prof else None
                    a=ndcg(popb+Q@foldin(F+([Q[jj]] if tr is not None else []),y+([tr] if tr is not None else [])),tlike,prof|{jj})
                    if best is None or a>best[0]: best=(a,k2)
                k=best[1]
            else: rem=[k for k in range(POOL) if k not in asked]; k=int(rng.choice(rem))
            asked.add(k); jj=int(pool[k]); tr=(rd[jj]-mu-bi[jj]) if jj in prof else None
            if tr is not None: F.append(Q[jj]); y.append(tr)
            nd[t]=ndcg(popb+Q@foldin(F,y),tlike,prof|{int(pool[a]) for a in asked})
            tr_.append(f"{title.get(jj,'?')[:24]}[{'L' if (jj in prof and rd[jj]>=4) else ('D' if jj in prof else 'unseen')}]")
        if int(os.environ.get('VERBOSE',0)) and kind=='rl' and m<int(os.environ.get('VERBOSE',0)): print(f"   [rl u{x}] "+" | ".join(tr_),flush=True)
        ND+=nd; m+=1
    return ND/m,m
print(f"\n=== TEST (pool={POOL}, {min(NEVAL,len(te))} users) ===",flush=True)
for kind in ['random','helf','rl','oracle']:
    nd,m=evalp(kind); tag={'rl':'RL (BC+REINFORCE)'}.get(kind,kind)
    print(f"{tag:<18}: "+" ".join(f"{v:.3f}" for v in nd)+f" | delta {nd[T]-nd[0]:+.3f}",flush=True)
