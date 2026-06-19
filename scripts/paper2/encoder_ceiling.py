"""
What is the elicitation CEILING? Compare, on the encoder:
  q0           : popularity (no reveals)
  infogain@8   : realizable adaptive policy at 8 questions
  full-profile : fold EVERY profile item (asked everything you CAN ask) -> ceiling of profile-restricted elicitation
  oracle-unlim : greedy peek, keep adding the profile item that most raises test-NDCG until no gain (test-aware ceiling)
Full + tail NDCG@10. Question: is full-profile the ceiling, or can a test-aware oracle exceed it (some profile items
mislead the ranker)? And how far below the ceiling does the realizable policy sit?
"""
import os, numpy as np, torch, torch.nn as nn
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
print("train encoder (joint)...",flush=True)
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
def enc_u_batch(revs):
    if not revs: return np.zeros((0,D))
    mx=max(len(r) for r in revs); arr=np.zeros((len(revs),mx,D+1),np.float32); m=np.zeros((len(revs),mx),np.float32)
    for b,rev in enumerate(revs):
        for q,(j,res) in enumerate(rev): arr[b,q,:D]=Q[j]; arr[b,q,D]=res; m[b,q]=1
    with torch.no_grad(): return enc(torch.tensor(arr),torch.tensor(m)).numpy()
def enc_u(rev): return enc_u_batch([rev])[0] if rev else np.zeros(D)
_rs=np.random.default_rng(123); SPL={}
for x in te:
    items=list(dict(rat_by_u[x]))
    if len(items)>=6: il=items[:]; _rs.shuffle(il); SPL[x]=(il[:len(il)//2], il[len(il)//2:])
def run(mode,tailonly,NEVAL=250):
    acc=0.; m=0
    for x in te[:NEVAL]:
        if x not in SPL: continue
        test,prof=SPL[x]; rd=dict(rat_by_u[x]); tlike=set(j for j in test if rd[j]>=4); prof=list(prof)
        if len(prof)<4 or not tlike or (tailonly and not any(not headmask[t] for t in tlike)): continue
        if mode=='q0': v=ndcg_at(np.zeros(D),tlike,set(prof),tailonly)
        elif mode=='full': v=ndcg_at(enc_u([(j,rd[j]-mu-bi[j]) for j in prof]),tlike,set(prof),tailonly)
        elif mode=='infogain':
            asked=[]; rev=[]
            while len(asked)<8 and len([j for j in prof if j not in asked]):
                cands=[j for j in prof if j not in asked]; cu=enc_u_batch([rev+[(c,rd[c]-mu-bi[c])] for c in cands])
                cov=(1/(1+np.exp(-(cu@Ql[cands].T)))).sum(1); c=cands[int(cov.argmax())]; asked.append(c); rev.append((c,rd[c]-mu-bi[c]))
            v=ndcg_at(enc_u(rev),tlike,set(prof),tailonly)
        else:  # oracle-unlim: greedy peek add while improving
            asked=[]; rev=[]; cur=ndcg_at(np.zeros(D),tlike,set(prof),tailonly) or 0.
            while True:
                cands=[j for j in prof if j not in asked]
                if not cands: break
                cu=enc_u_batch([rev+[(c,rd[c]-mu-bi[c])] for c in cands]); best=None
                for idx,c in enumerate(cands):
                    vv=ndcg_at(cu[idx],tlike,set(prof),tailonly)
                    if vv is not None and (best is None or vv>best[0]): best=(vv,c)
                if best is None or best[0]<=cur+1e-9: break
                cur=best[0]; asked.append(best[1]); rev.append((best[1],rd[best[1]]-mu-bi[best[1]]))
            v=cur
        if v is not None: acc+=v; m+=1
    return acc/m, m
for tailonly in [False,True]:
    print(f"\n=== {'TAIL' if tailonly else 'FULL'} NDCG@10 — elicitation ceiling ===",flush=True)
    for mode in ['q0','infogain','full','oracle']:
        r,m=run(mode,tailonly); print(f"  {mode:<9}: {r:.3f}  (n={m})",flush=True)
