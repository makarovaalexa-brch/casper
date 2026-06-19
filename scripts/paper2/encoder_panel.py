"""
Selection-baseline panel ON THE RECONSTRUCTION ENCODER (fold-in fixed = encoder). Vary WHICH items are asked:
random, popularity, entropy (Rashid'02), HELF (Rashid'08), RMVA (representative max-volume), and ADAPTIVE oracle
(greedy peek = ceiling). Eval full + tail NDCG@10, q0..q8. Question: does the ordering now make sense
(smart/adaptive > random ~ pop), and where do the realizable methods sit vs the adaptive oracle?
"""
import os, numpy as np, torch, torch.nn as nn, scipy.linalg as sla
base='C:/dev/phd/casper/data/movielens'; ml=f'{base}/ml-1m'; D=64; K=int(os.environ.get('K',12)); rng=np.random.default_rng(0); torch.manual_seed(0)
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
Q=np.load(f'{base}/.cache/Q_svd.npy'); bi=np.load(f'{base}/.cache/bi_svd.npy'); Qt=torch.tensor(Q)
ipsw=(1.0/np.clip((cnt/max(cnt.max(),1))**0.5,1e-3,1.0)).astype(np.float32)
order_pop=np.argsort(-cnt); cum=np.cumsum(cnt[order_pop])/cnt.sum(); HEAD=set(int(j) for j in order_pop[:np.searchsorted(cum,0.33)+1])
headmask=np.zeros(ni,bool); headmask[list(HEAD)]=True
H5=np.zeros((ni,5)); trset=set(trU)
for k in range(len(uu)):
    if uu[k] in trset: H5[ii[k],min(max(int(round(R[k])),1),5)-1]+=1
pm=H5/np.clip(H5.sum(1,keepdims=True),1,None); ent=-(pm*np.log(pm+1e-12)).sum(1)
lf=np.log(cnt+1)/np.log(cnt.max()+1); Hn=ent/np.log(5); helf=2*lf*Hn/(lf+Hn+1e-9)
resid_by_u={x:{j:(r-mu-bi[j]) for j,r in rat_by_u[x]} for x in (trU+te)}
class Enc(nn.Module):
    def __init__(s):
        super().__init__(); s.inp=nn.Sequential(nn.Linear(D+1,128),nn.ReLU(),nn.Linear(128,128),nn.ReLU()); s.att=nn.Linear(128,1); s.val=nn.Linear(128,D)
    def forward(s,toks,mask):
        h=s.inp(toks); a=s.att(h).squeeze(-1).masked_fill(mask==0,-1e9); al=torch.softmax(a,1); return (al.unsqueeze(-1)*s.val(h)).sum(1)
enc=Enc(); Qp=torch.nn.Parameter(Qt.clone()); opt=torch.optim.Adam(list(enc.parameters())+[Qp],1e-3,weight_decay=1e-5)
trbig=[x for x in trU if len(rat_by_u[x])>=K+1]
def make_batch(users,kk):
    toks=np.zeros((len(users),kk,D+1),np.float32); msk=np.zeros((len(users),kk),np.float32)
    tgt=np.zeros((len(users),ni),np.float32); wt=np.ones((len(users),ni),np.float32); seen=np.zeros((len(users),ni),bool)
    for b,x in enumerate(users):
        rv=[(j,resid_by_u[x][j]) for j,_ in rat_by_u[x]]; rng.shuffle(rv); rev=rv[:kk]
        for q,(j,res) in enumerate(rev): toks[b,q,:D]=Q[j]; toks[b,q,D]=res; msk[b,q]=1; seen[b,j]=True
        posw=0.
        for j in likes_by_u[x]:
            if not seen[b,j]: tgt[b,j]=1.0; wt[b,j]=ipsw[j]; posw+=ipsw[j]
        nneg=ni-int(seen[b].sum())-int(tgt[b].sum())
        if nneg>0: wt[b][(tgt[b]==0)&(~seen[b])]=posw/nneg
    return torch.tensor(toks),torch.tensor(msk),torch.tensor(tgt),torch.tensor(wt),torch.tensor(seen)
print(f"train encoder (joint) on {len(trbig)} users...",flush=True)
for ep in range(int(os.environ.get('EP',30))):
    rng.shuffle(trbig)
    for b0 in range(0,len(trbig),256):
        us=trbig[b0:b0+256]; kk=int(rng.integers(1,K+1)); toks,msk,tgt,wt,seen=make_batch(us,kk); u=enc(toks,msk); sc=u@Qp.t()
        loss=(wt*nn.functional.binary_cross_entropy_with_logits(sc,tgt,reduction='none')).masked_fill(seen,0.).mean()
        opt.zero_grad(); loss.backward(); opt.step()
enc.eval(); Ql=Qp.detach().numpy()
_W=1./np.log2(np.arange(2,12))
def ndcg_at(u,tlike,excl,tailonly):
    sc=(popb+Ql@u).copy(); sc[list(excl)]=-1e9
    if tailonly: sc[headmask]=-1e9; rel=set(t for t in tlike if not headmask[t])
    else: rel=set(tlike)
    top=np.argpartition(-sc,10)[:10]; top=top[np.argsort(-sc[top])]
    return sum(_W[p] for p,t in enumerate(top) if int(t) in rel)/(_W[:min(10,len(rel))].sum()+1e-12) if rel else None
def enc_u(rev):
    if not rev: return np.zeros(D)
    arr=np.zeros((1,len(rev),D+1),np.float32); m=np.ones((1,len(rev)),np.float32)
    for q,(j,res) in enumerate(rev): arr[0,q,:D]=Q[j]; arr[0,q,D]=res
    with torch.no_grad(): return enc(torch.tensor(arr),torch.tensor(m)).numpy()[0]
def rmva_order(prof):
    if len(prof)<2: return prof
    _,_,piv=sla.qr(Q[np.array(prof)].T,pivoting=True); return [prof[p] for p in piv]
_rs=np.random.default_rng(123); SPL={}
for x in te:
    items=list(dict(rat_by_u[x]))
    if len(items)>=6: il=items[:]; _rs.shuffle(il); SPL[x]=(il[:len(il)//2], il[len(il)//2:])
def panel(selector,tailonly,NEVAL=250):
    acc={q:0. for q in [0,1,2,4,8]}; m=0
    for x in te[:NEVAL]:
        if x not in SPL: continue
        test,prof=SPL[x]; rd=dict(rat_by_u[x]); tlike=set(j for j in test if rd[j]>=4); prof=list(prof)
        if len(prof)<4 or not tlike or (tailonly and not any(not headmask[t] for t in tlike)): continue
        if selector=='random': o=prof[:]; rng.shuffle(o); order=o
        elif selector=='pop': order=sorted(prof,key=lambda j:-cnt[j])
        elif selector=='entropy': order=sorted(prof,key=lambda j:-ent[j])
        elif selector=='helf': order=sorted(prof,key=lambda j:-helf[j])
        elif selector=='rmva': order=rmva_order(prof)
        else: order=None   # oracle
        if selector=='oracle':
            asked=[]; rev=[]
            for q in [0,1,2,4,8]:
                while len(asked)<q:
                    cands=[j for j in prof if j not in asked]; best=None
                    for j in cands:
                        v=ndcg_at(enc_u(rev+[(j,rd[j]-mu-bi[j])]),tlike,set(prof)|set(asked)|{j},tailonly)
                        if v is not None and (best is None or v>best[0]): best=(v,j)
                    j=best[1]; asked.append(j); rev.append((j,rd[j]-mu-bi[j]))
                acc[q]+=ndcg_at(enc_u(rev),tlike,set(prof),tailonly) if rev else ndcg_at(np.zeros(D),tlike,set(prof),tailonly)
        else:
            for q in [0,1,2,4,8]:
                rev=[(j,rd[j]-mu-bi[j]) for j in order[:q]]; v=ndcg_at(enc_u(rev),tlike,set(prof),tailonly)
                if v is not None: acc[q]+=v
        m+=1
    return {q:acc[q]/m for q in acc}, m
for tailonly in [False,True]:
    print(f"\n=== {'TAIL' if tailonly else 'FULL'} NDCG@10 — SELECTION panel on the encoder ===",flush=True)
    for sel in ['random','pop','entropy','helf','rmva','oracle']:
        r,m=panel(sel,tailonly); print(f"  {sel:<8}: "+" ".join(f"q{q}={r[q]:.3f}" for q in [0,1,2,4,8])+f" | +{r[8]-r[0]:+.3f}",flush=True)
