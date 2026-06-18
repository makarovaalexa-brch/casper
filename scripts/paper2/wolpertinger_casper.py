"""
WOLPERTINGER applied to CASPER (the thesis method). The replication (wolpertinger_replicate.py) validated the
machinery, so any failure here is attributable to the task, not the implementation.

Env (per cold user, T steps), frozen calibrated instrument: state s_t = folded taste vector u_t; action = which pool
item to ask. Wolpertinger selection: actor(s)->proto p in R^d; kNN = k nearest UNASKED pool items to p (cosine over
Q); critic Q(s,a) scores them; pick argmax-Q. Transition: fold the user's true answer (leakage-free: taste residual
if in profile, else no-op), exclude asked -> u_{t+1}. Reward r_t = NDCG@10(t)-NDCG@10(t-1) on the user's held-out
likes (dense; train-time only). DDPG: critic TD-regression with target nets; actor maximises Q(s,actor(s)).

Optional BC warm-start of the actor on HELF (WARM=1). Eval greedy vs HELF / random / oracle on a SHARED fixed split.
"""
import os, random, numpy as np, torch, torch.nn as nn, scipy.linalg as sla
from collections import deque
base='C:/dev/phd/casper/data/movielens'; ml=f'{base}/ml-1m'; D=64; LAM=5.0
T=int(os.environ.get('T',8)); POOL=int(os.environ.get('POOL',120)); K=int(os.environ.get('K',20))
NTR=int(os.environ.get('NTR',400)); NEVAL=int(os.environ.get('NEVAL',250)); EPISODES=int(os.environ.get('EPISODES',6000))
GAM=0.95; TAU=0.01; rng=np.random.default_rng(0); torch.manual_seed(0); random.seed(0)
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
pool=np.array(np.argsort(-cnt)[:POOL]); Qp=torch.tensor(Q[pool]); Qpn=Qp/Qp.norm(dim=1,keepdim=True)
H5=np.zeros((ni,5)); trset=set(trU)
for k in range(len(uu)):
    if uu[k] in trset: H5[ii[k],min(max(int(round(R[k])),1),5)-1]+=1
pm=H5/np.clip(H5.sum(1,keepdims=True),1,None); ent=-(pm*np.log(pm+1e-12)).sum(1)
lf=np.log(cnt+1)/np.log(cnt.max()+1); Hn=ent/np.log(5); helf=2*lf*Hn/(lf+Hn+1e-9); helf_pool=helf[pool]
SPL={}; _rs=np.random.default_rng(123)
for x in keep:
    items=list(dict(rat_by_u[x]))
    if len(items)>=6: il=items[:]; _rs.shuffle(il); SPL[x]=(set(il[:len(il)//2]),set(il[len(il)//2:]))
def foldin(F,y): F=np.array(F); return np.linalg.solve(F.T@F+LAM*np.eye(D),F.T@np.array(y,np.float32)) if len(F) else np.zeros(D)
_W=1./np.log2(np.arange(2,12))
def ndcg(score,rel,excl):
    s=score.copy(); s[list(excl)]=-1e9; top=np.argpartition(-s,10)[:10]; top=top[np.argsort(-s[top])]
    return sum(_W[p] for p,t in enumerate(top) if int(t) in rel)/(_W[:min(10,len(rel))].sum()+1e-12)
def mk(): return nn.Sequential(nn.Linear(D,128),nn.ReLU(),nn.Linear(128,128),nn.ReLU(),nn.Linear(128,D))
def mkc(): return nn.Sequential(nn.Linear(2*D,128),nn.ReLU(),nn.Linear(128,128),nn.ReLU(),nn.Linear(128,1))
actor=mk(); actor_t=mk(); actor_t.load_state_dict(actor.state_dict())
critic=mkc(); critic_t=mkc(); critic_t.load_state_dict(critic.state_dict())
oa=torch.optim.Adam(actor.parameters(),1e-4); oc=torch.optim.Adam(critic.parameters(),1e-3)
def knn(proto, asked, k):                # k nearest UNASKED pool slots to proto (cosine)
    pn=proto/(proto.norm()+1e-9); sims=(Qpn@pn).clone()
    if asked: sims[list(asked)]=-1e9
    return sims.topk(min(k,POOL-len(asked))).indices
buf=deque(maxlen=40000)
def soft(tn,n):
    for p,pt in zip(n.parameters(),tn.parameters()): pt.data.mul_(1-TAU).add_(TAU*p.data)
# ---------- optional warm-start: actor proto -> centroid of high-HELF items (a sensible ~HELF init) ----------
if int(os.environ.get('WARM',1)):
    topslots=[int(s) for s in helf_pool.argsort()[::-1][:10]]; tgt=Qp[topslots].mean(0)
    for _ in range(300):
        s=0.1*torch.randn(64,D); l=((actor(s)-tgt)**2).mean(); oa.zero_grad(); l.backward(); oa.step()
    actor_t.load_state_dict(actor.state_dict())
def env_reset(x):
    if x not in SPL: return None
    test,prof=SPL[x]; rd=dict(rat_by_u[x]); tlike=set(j for j in test if rd[j]>=4)
    if not tlike: return None
    return dict(rd=rd,prof=prof,tlike=tlike,F=[],y=[],asked=set(),nd=ndcg(popb.copy(),tlike,prof))
def step_env(st, slot):
    jj=int(pool[slot]); tr=(st['rd'][jj]-mu-bi[jj]) if jj in st['prof'] else None
    if tr is not None: st['F'].append(Q[jj]); st['y'].append(tr)
    st['asked'].add(slot)
    u=foldin(st['F'],st['y']); nd2=ndcg(popb+Q@u,st['tlike'],st['prof']|{int(pool[a]) for a in st['asked']})
    r=nd2-st['nd']; st['nd']=nd2
    return u.astype(np.float32), r
def select(u, asked, eps):
    s=torch.tensor(u); proto=actor(s); cand=knn(proto,asked,K)
    with torch.no_grad(): q=critic(torch.cat([s.expand(len(cand),D),Qp[cand]],1)).squeeze(1)
    slot=int(cand[int(q.argmax())])
    if eps>0 and random.random()<eps: slot=int(cand[random.randrange(len(cand))])
    return slot
def update(bs=128):
    if len(buf)<bs: return
    batch=random.sample(buf,bs)
    s=torch.tensor(np.array([b[0] for b in batch])); a=torch.tensor(np.array([b[1] for b in batch]))
    r=torch.tensor(np.array([b[2] for b in batch],np.float32)); s2=torch.tensor(np.array([b[3] for b in batch]))
    done=torch.tensor(np.array([b[4] for b in batch],np.float32)); asked2=[b[5] for b in batch]
    with torch.no_grad():
        proto2=actor_t(s2); pn=proto2/(proto2.norm(dim=1,keepdim=True)+1e-9); sims=pn@Qpn.t()  # bs x POOL
        for i in range(bs):
            if asked2[i]: sims[i,list(asked2[i])]=-1e9
        cand=sims.topk(K,dim=1).indices                      # bs x K
        ac=Qp[cand]; sc=s2.unsqueeze(1).expand(-1,K,-1)      # bs x K x D
        qn=critic_t(torch.cat([sc,ac],-1)).squeeze(-1).max(1).values
        tgt=r+GAM*(1-done)*qn
    q=critic(torch.cat([s,a],1)).squeeze(1); lc=((q-tgt)**2).mean()
    oc.zero_grad(); lc.backward(); oc.step()
    la=-critic(torch.cat([s,actor(s)],1)).mean(); oa.zero_grad(); la.backward(); oa.step()
    soft(critic_t,critic); soft(actor_t,actor)
print(f"Wolpertinger-CASPER: pool={POOL} k={K} T={T} warm={os.environ.get('WARM',1)}; training {EPISODES} episodes...",flush=True)
order=[x for x in trU if x in SPL]
for epi in range(EPISODES):
    x=order[epi%len(order)];
    if epi%len(order)==0: random.shuffle(order)
    st=env_reset(x)
    if st is None: continue
    u=foldin([],[]).astype(np.float32); eps=max(0.3*(1-epi/EPISODES),0.05)
    for t in range(T):
        slot=select(u,st['asked'],eps); u2,r=step_env(st,slot)
        buf.append((u,Q[pool[slot]].astype(np.float32),r,u2,1.0 if t==T-1 else 0.0,set(st['asked'])))
        u=u2; update()
    if (epi+1)%1500==0:
        # quick train-reward probe
        print(f"  epi{epi+1} buf={len(buf)} (eps={eps:.2f})",flush=True)
# ---------- eval ----------
def evalp(kind):
    ND=np.zeros(T+1); m=0
    for x in te[:NEVAL]:
        st=env_reset(x)
        if st is None: continue
        nd=np.zeros(T+1); nd[0]=st['nd']; u=foldin([],[]).astype(np.float32)
        for t in range(1,T+1):
            if kind=='wolp': slot=select(u,st['asked'],0.0)
            elif kind=='helf': slot=max((s for s in range(POOL) if s not in st['asked']),key=lambda s:helf_pool[s])
            elif kind=='oracle':
                best=None
                for s2 in range(POOL):
                    if s2 in st['asked']: continue
                    jj=int(pool[s2]); tr=(st['rd'][jj]-mu-bi[jj]) if jj in st['prof'] else None
                    a=ndcg(popb+Q@foldin(st['F']+([Q[jj]] if tr is not None else []),st['y']+([tr] if tr is not None else [])),st['tlike'],st['prof']|{jj})
                    if best is None or a>best[0]: best=(a,s2)
                slot=best[1]
            else: slot=random.choice([s for s in range(POOL) if s not in st['asked']])
            u,_=step_env(st,slot); nd[t]=st['nd']
        ND+=nd; m+=1
    return ND/m,m
print(f"\n=== TEST (pool={POOL}, {min(NEVAL,len(te))} users) ===",flush=True)
for kind in ['random','helf','wolp','oracle']:
    nd,m=evalp(kind); tag={'wolp':'WOLPERTINGER'}.get(kind,kind)
    print(f"{tag:<14}: "+" ".join(f"{v:.3f}" for v in nd)+f" | delta {nd[T]-nd[0]:+.3f}",flush=True)
