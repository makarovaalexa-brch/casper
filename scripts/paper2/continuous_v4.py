"""
PAPER B — STARTING POINT (must work): can a continuous actor REPLICATE item-EIG? Actor emits a in R^D; SNAP to nearest
profile item; fold that item's TRUE answer (same fold as EIG => consistent). BC the actor to emit the EIG choice's
embedding. If actorP (snapped) == item-EIG, the model can learn EIG (the achievable starting point). Report FULL-FIRST
(main metric) + tail, NDCG@10 + Recall@50, vs item-EIG / random / oracle.
"""
import os, numpy as np, torch, torch.nn as nn
base='C:/dev/phd/casper/data/movielens'; ml=f'{base}/ml-1m'; D=64; T=6; NU_TR=int(os.environ.get('NU_TR',1500)); rng=np.random.default_rng(0); torch.manual_seed(0)
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
keep=[x for x in range(nu) if len(likes_by_u.get(x,[]))>=8]; rng.shuffle(keep); n=len(keep); trU=keep[:int(0.8*n)]; te=keep[int(0.9*n):]
cnt=np.zeros(ni)
for x in trU:
    for j in likes_by_u.get(x,[]): cnt[j]+=1
popb=np.log(cnt+1.0).astype(np.float32); sm=c=0.
for x in trU:
    for j,r in rat_by_u[x]: sm+=r;c+=1
mu=sm/c
Q=np.load(f'{base}/.cache/Q_svd.npy'); bi=np.load(f'{base}/.cache/bi_svd.npy'); Ql=np.load(f'{base}/.cache/Ql_unified.npy')
order_pop=np.argsort(-cnt); cum=np.cumsum(cnt[order_pop])/cnt.sum(); HEAD=set(int(j) for j in order_pop[:np.searchsorted(cum,0.33)+1]); headmask=np.zeros(ni,bool); headmask[list(HEAD)]=True
resid={x:{j:(r-mu-bi[j]) for j,r in rat_by_u[x]} for x in (trU+te)}
POS=np.mean([resid[x][j] for x in trU[:3000] for j,r in rat_by_u[x] if r>=4]); NEG=-1.0
class Enc(nn.Module):
    def __init__(s):
        super().__init__(); s.inp=nn.Sequential(nn.Linear(D+1,128),nn.ReLU(),nn.Linear(128,128),nn.ReLU()); s.att=nn.Linear(128,1); s.val=nn.Linear(128,D)
    def forward(s,t,m):
        h=s.inp(t); a=s.att(h).squeeze(-1).masked_fill(m==0,-1e9); al=torch.softmax(a,1); return (al.unsqueeze(-1)*s.val(h)).sum(1)
enc=Enc(); enc.load_state_dict(torch.load(f'{base}/.cache/enc_unified.pt')); enc.eval()
def sig(z): return 1/(1+np.exp(-z))
def enc_u_np(toks):
    if not toks: return np.zeros(D)
    arr=np.zeros((1,len(toks),D+1),np.float32); m=np.ones((1,len(toks)),np.float32)
    for q,(f,v) in enumerate(toks): arr[0,q,:D]=f; arr[0,q,D]=v
    with torch.no_grad(): return enc(torch.tensor(arr),torch.tensor(m)).numpy()[0]
def enc_u_batch(revs):
    if not revs: return np.zeros((0,D))
    mx=max(len(r) for r in revs); arr=np.zeros((len(revs),mx,D+1),np.float32); m=np.zeros((len(revs),mx),np.float32)
    for b,rev in enumerate(revs):
        for q,(f,v) in enumerate(rev): arr[b,q,:D]=f; arr[b,q,D]=v; m[b,q]=1
    with torch.no_grad(): return enc(torch.tensor(arr),torch.tensor(m)).numpy()
def eig_pick(toks, cset, rd, restp):                      # validated EIG: belief-weighted expected coverage over rest-of-profile
    ul=enc_u_batch([toks+[(Q[j],POS)] for j in cset]); ud=enc_u_batch([toks+[(Q[j],NEG)] for j in cset])
    val=0.5*sig(popb[restp]+ul@Ql[restp].T).sum(1)+0.5*sig(popb[restp]+ud@Ql[restp].T).sum(1); return cset[int(val.argmax())]
# ---- BC trajectories: (state u_t -> EIG item embedding Q[j*]) ----
print("gen item-EIG trajectories...",flush=True); S=[];Targ=[];TT=[]
sample=[x for x in trU if len(rat_by_u[x])>=12][:NU_TR]
for x in sample:
    prof=[j for j,_ in rat_by_u[x]]; restp=np.array(prof); toks=[]; asked=[]
    for t in range(T):
        cs=[j for j in prof if j not in asked]
        if not cs: break
        j=eig_pick(toks,cs,dict(rat_by_u[x]),restp); S.append(enc_u_np(toks).astype(np.float32)); Targ.append(Q[j].astype(np.float32)); TT.append(t)
        asked.append(j); toks.append((Q[j],resid[x][j]))
S=np.array(S);Targ=np.array(Targ);TT=np.array(TT); print(f"  {len(S)} pairs",flush=True)
class Actor(nn.Module):
    def __init__(s): super().__init__(); s.f=nn.Sequential(nn.Linear(D+2,256),nn.ReLU(),nn.Linear(256,256),nn.ReLU(),nn.Linear(256,D))
    def forward(s,u,t): return s.f(torch.cat([u,t],1))
actor=Actor(); opt=torch.optim.Adam(actor.parameters(),1e-3)
St=torch.tensor(S); Tg=torch.tensor(Targ); TTt=torch.stack([torch.tensor(TT/8.,dtype=torch.float32),torch.tensor(TT.astype(np.float32))],1)
print("BC actor -> EIG item embedding...",flush=True)
for ep in range(60):
    idx=torch.randperm(len(St))
    for b0 in range(0,len(St),512):
        bb=idx[b0:b0+512]; pred=actor(St[bb],TTt[bb]); loss=((pred-Tg[bb])**2).mean(); opt.zero_grad(); loss.backward(); opt.step()
    if (ep+1)%20==0: print(f"  ep{ep+1} BC-MSE={loss.item():.4f}",flush=True)
actor.eval()
_W=1./np.log2(np.arange(2,12))
def metr(u,tlike,excl,tail):
    sc=(popb+Ql@u).copy(); sc[list(excl)]=-1e9
    if tail: sc[headmask]=-1e9; rel=set(t for t in tlike if not headmask[t])
    else: rel=set(tlike)
    if not rel: return None
    o=np.argsort(-sc); nd=sum(_W[p] for p,t in enumerate(o[:10]) if int(t) in rel)/(_W[:min(10,len(rel))].sum()+1e-12); rc=len(set(o[:50].tolist())&rel)/len(rel); return nd,rc
_rs=np.random.default_rng(123); SPL={}
for x in te:
    its=list(dict(rat_by_u[x]))
    if len(its)>=8: il=its[:]; _rs.shuffle(il); SPL[x]=(il[:len(il)//2], il[len(il)//2:])
TE=[x for x in te if x in SPL][:200]
def emit(u,t):
    with torch.no_grad(): return actor(torch.tensor(u[None],dtype=torch.float32),torch.tensor([[t/8.,float(t)]])).numpy()[0]
agree=[]
def run(mode,tail):
    M={q:0. for q in [0,1,2,4,8]};Rc={q:0. for q in [0,1,2,4,8]};m=0
    for x in TE:
        test,prof=SPL[x]; rd=dict(rat_by_u[x]); tlike=set(j for j in test if rd[j]>=4); prof=list(prof)
        if len(prof)<4 or not tlike or (tail and not any(not headmask[t] for t in tlike)): continue
        restp=np.array(prof); toks=[]; asked=[]
        for q in [0,1,2,4,8]:
            while len(toks)<q:
                cs=[j for j in prof if j not in asked]
                if not cs: break
                if mode=='actorSnap':
                    a=emit(enc_u_np(toks),len(toks)); Qc=Q[cs]; pick=cs[int((Qc@a/(np.linalg.norm(Qc,axis=1)+1e-9)).argmax())]   # snap to nearest profile item
                    if not tail and len(asked)==0: agree.append(int(pick==eig_pick(toks,cs,rd,restp)))
                elif mode=='item': pick=eig_pick(toks,cs,rd,restp)
                elif mode=='random': pick=cs[int(rng.integers(len(cs)))]
                else:
                    ul=enc_u_batch([toks+[(Q[j],rd[j]-mu-bi[j])] for j in cs]); best=None
                    for li,j in enumerate(cs):
                        mt=metr(ul[li],tlike,set(prof),tail); a=mt[0] if mt else -1
                        if best is None or a>best[0]: best=(a,j)
                    pick=best[1]
                asked.append(pick); toks.append((Q[pick],rd[pick]-mu-bi[pick]))
            mt=metr(enc_u_np(toks),tlike,set(prof),tail)
            if mt: M[q]+=mt[0];Rc[q]+=mt[1]
        m+=1
    return {q:M[q]/m for q in M},{q:Rc[q]/m for q in Rc},m
for tail in [False,True]:
    print(f"\n=== {'TAIL' if tail else 'FULL (MAIN)'}: NDCG@10 / Recall@50 ===",flush=True)
    for mode in ['random','item','actorSnap','oracle']:
        M,Rc,m=run(mode,tail); print(f"  {mode:<10}: NDCG "+" ".join(f"{M[q]:.3f}" for q in [0,1,2,4,8])+" | Rec "+" ".join(f"{Rc[q]:.3f}" for q in [0,1,2,4,8]),flush=True)
print(f"\nREPLICATION CHECK: actorSnap top-1 agreement with item-EIG (turn 1) = {np.mean(agree):.2f}  (1.0 = perfectly replicates EIG)",flush=True)
