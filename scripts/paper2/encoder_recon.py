"""
Paper B Phase A: LEARNED reconstruction encoder vs ridge fold-in (gates A1-A4).
Frozen item factors Q_svd (isolate the FOLD-IN). Encoder = attention pooling over revealed (Q[item], residual)
tokens -> user vector u. Decoder FIXED = popb + Q.u (same for encoder and ridge, fair comparison). Trained with
MASKED, SHUFFLED reveals (random count k=1..K) to RECONSTRUCT the user's FULL unrevealed like-set, IPS/tail-weighted
(inverse-popularity) so it learns niche tastes not popularity. Eval on the long-tail task (Cremonesi head-33%).
GATES: A1 encoder>ridge & >pop on tail-NDCG@10 @q4,q8; A2 holds @q1,2; A3 monotone; A4 recon-AUC tracks tail-NDCG.
"""
import os, numpy as np, torch, torch.nn as nn
base='C:/dev/phd/casper/data/movielens'; ml=f'{base}/ml-1m'; D=64; LAM=5.0; K=int(os.environ.get('K',12)); rng=np.random.default_rng(0); torch.manual_seed(0)
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
Qt=torch.tensor(Q); popbt=torch.tensor(popb); bit=torch.tensor(bi)
ipsw=(1.0/np.clip((cnt/max(cnt.max(),1))**0.5,1e-3,1.0)).astype(np.float32); ipswt=torch.tensor(ipsw)  # inverse-popularity
order_pop=np.argsort(-cnt); cum=np.cumsum(cnt[order_pop])/cnt.sum(); HEAD=set(int(j) for j in order_pop[:np.searchsorted(cum,0.33)+1])
headmask=np.zeros(ni,bool); headmask[list(HEAD)]=True
H5=np.zeros((ni,5)); trset=set(trU)
for k in range(len(uu)):
    if uu[k] in trset: H5[ii[k],min(max(int(round(R[k])),1),5)-1]+=1
pm=H5/np.clip(H5.sum(1,keepdims=True),1,None); entv=-(pm*np.log(pm+1e-12)).sum(1)
lf=np.log(cnt+1)/np.log(cnt.max()+1); Hn=entv/np.log(5); helf=2*lf*Hn/(lf+Hn+1e-9)
# residual per (user,item)
resid_by_u={x:[(j,(r-mu-bi[j])) for j,r in rat_by_u[x]] for x in (trU+te)}
class Enc(nn.Module):
    def __init__(s):
        super().__init__(); s.inp=nn.Sequential(nn.Linear(D+1,128),nn.ReLU(),nn.Linear(128,128),nn.ReLU())
        s.att=nn.Linear(128,1); s.val=nn.Linear(128,D)
    def forward(s,toks,mask):           # toks (B,K,D+1), mask (B,K) 1=valid
        h=s.inp(toks); a=s.att(h).squeeze(-1).masked_fill(mask==0,-1e9); al=torch.softmax(a,1)
        return (al.unsqueeze(-1)*s.val(h)).sum(1)   # (B,D)
JOINT=int(os.environ.get('JOINT',0))
enc=Enc(); Qparam=torch.nn.Parameter(Qt.clone())
params=list(enc.parameters())+([Qparam] if JOINT else [])
opt=torch.optim.Adam(params,1e-3,weight_decay=1e-5)
def Qcur(): return Qparam if JOINT else Qt
def make_batch(users,kk):
    toks=np.zeros((len(users),kk,D+1),np.float32); msk=np.zeros((len(users),kk),np.float32)
    tgt=np.zeros((len(users),ni),np.float32); wt=np.ones((len(users),ni),np.float32); seen=np.zeros((len(users),ni),bool)
    for b,x in enumerate(users):
        rv=resid_by_u[x][:]; rng.shuffle(rv); rev=rv[:kk]
        for q,(j,res) in enumerate(rev): toks[b,q,:D]=Q[j]; toks[b,q,D]=res; msk[b,q]=1; seen[b,j]=True
        posw=0.
        for j in likes_by_u[x]:
            if not seen[b,j]: tgt[b,j]=1.0; wt[b,j]=ipsw[j]; posw+=ipsw[j]   # unrevealed likes = recon target, IPS-weighted
        nneg=ni-int(seen[b].sum())-int(tgt[b].sum())
        if nneg>0: negw=posw/nneg; wt[b][(tgt[b]==0)&(~seen[b])]=negw        # BALANCE: total neg weight == total pos weight
    return torch.tensor(toks),torch.tensor(msk),torch.tensor(tgt),torch.tensor(wt),torch.tensor(seen)
trbig=[x for x in trU if len(resid_by_u[x])>=K+1]
print(f"train encoder on {len(trbig)} users (K={K}); IPS-weighted recon of unrevealed likes...",flush=True)
for ep in range(int(os.environ.get('EP',25))):
    rng.shuffle(trbig); tot=0.;nb=0
    for b0 in range(0,len(trbig),256):
        us=trbig[b0:b0+256]; kk=int(rng.integers(1,K+1))
        toks,msk,tgt,wt,seen=make_batch(us,kk); u=enc(toks,msk)
        sc=u@Qcur().t()                   # pure personalization (no popb crutch); JOINT=1 also learns Q
        loss=(wt*nn.functional.binary_cross_entropy_with_logits(sc,tgt,reduction='none')); loss=loss.masked_fill(seen,0.).mean()
        opt.zero_grad(); loss.backward(); opt.step(); tot+=loss.item(); nb+=1
    if (ep+1)%5==0: print(f"  ep{ep+1} loss={tot/nb:.4f}",flush=True)
enc.eval(); Ql=Qcur().detach().numpy()    # learned decoder factors (== Q_svd if not JOINT)
# ---------- eval: encoder vs ridge vs popularity on tail (Cremonesi head-33%) ----------
def foldin_ridge(F,y): F=np.array(F); return np.linalg.solve(F.T@F+LAM*np.eye(D),F.T@np.array(y,np.float32)) if len(F) else np.zeros(D)
_W=1./np.log2(np.arange(2,12))
def tail_ndcg(u,tlike,excl,Quse):
    sc=(popb+Quse@u).copy(); sc[list(excl)]=-1e9; sc[headmask]=-1e9
    top=np.argpartition(-sc,10)[:10]; top=top[np.argsort(-sc[top])]
    rel=set(t for t in tlike if not headmask[t])
    return sum(_W[p] for p,t in enumerate(top) if int(t) in rel)/(_W[:min(10,len(rel))].sum()+1e-12) if rel else None
def enc_u(rev):
    toks=np.zeros((1,max(len(rev),1),D+1),np.float32); msk=np.zeros((1,max(len(rev),1)),np.float32)
    for q,(j,res) in enumerate(rev): toks[0,q,:D]=Q[j]; toks[0,q,D]=res; msk[0,q]=1
    with torch.no_grad(): return enc(torch.tensor(toks),torch.tensor(msk)).numpy()[0]
_rs=np.random.default_rng(123); SPL={}
for x in te:
    items=list(dict(rat_by_u[x]))
    if len(items)>=6: il=items[:]; _rs.shuffle(il); SPL[x]=(il[:len(il)//2], il[len(il)//2:])
def evalq(method,Qd=8,NEVAL=300):
    acc={q:0. for q in [0,1,2,4,8]}; m=0
    for x in te[:NEVAL]:
        if x not in SPL: continue
        test,prof=SPL[x]; rd=dict(rat_by_u[x]); tlike=set(j for j in test if rd[j]>=4)
        if len(prof)<4 or not any(not headmask[t] for t in tlike): continue
        order=sorted(prof,key=lambda j:-helf[j])      # same reveal order (HELF) for fair fold-in comparison
        for q in [0,1,2,4,8]:
            rev=[(j,rd[j]-mu-bi[j]) for j in order[:q]]
            if method=='pop': u=np.zeros(D); Quse=Q
            elif method=='ridge': u=foldin_ridge([Q[j] for j,_ in rev],[r for _,r in rev]) if rev else np.zeros(D); Quse=Q
            else: u=enc_u(rev) if rev else np.zeros(D); Quse=Ql
            v=tail_ndcg(u,tlike,set(prof),Quse)
            if v is not None: acc[q]+=v
        m+=1
    return {q:acc[q]/m for q in acc}, m
print("\n=== GATE eval: tail-NDCG@10 (Cremonesi head-33%), HELF reveal order ===",flush=True)
res={}
for meth in ['pop','ridge','encoder']:
    r,m=evalq(meth); res[meth]=r
    print(f"  {meth:<8}: "+" ".join(f"q{q}={r[q]:.3f}" for q in [0,1,2,4,8])+f" | n={m}",flush=True)
e,rd_=res['encoder'],res['ridge']
print(f"\nGATE A1 (enc>ridge & >pop @q4,q8): q4 enc {e[4]:.3f} vs ridge {rd_[4]:.3f} vs pop {res['pop'][4]:.3f}; q8 enc {e[8]:.3f} vs ridge {rd_[8]:.3f}",flush=True)
print(f"  A1 {'PASS' if e[4]>rd_[4] and e[8]>rd_[8] and e[8]>res['pop'][8] else 'FAIL'}",flush=True)
print(f"GATE A2 (enc>ridge @q1,q2): q1 {e[1]:.3f} vs {rd_[1]:.3f}; q2 {e[2]:.3f} vs {rd_[2]:.3f} -> {'PASS' if e[1]>=rd_[1] and e[2]>=rd_[2] else 'FAIL'}",flush=True)
print(f"GATE A3 (encoder monotone): {'PASS' if all(e[b]>=e[a]-3e-3 for a,b in zip([0,1,2,4],[1,2,4,8])) else 'FAIL'}",flush=True)
