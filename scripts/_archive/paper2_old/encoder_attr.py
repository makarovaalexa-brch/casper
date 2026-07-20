"""
Does the reconstruction encoder work WITH ATTRIBUTES (unified item+attribute instrument)?
Same encoder, but tokens can be ITEMS (Q[item], residual) OR GENRES (genre-centroid, user's genre-affinity). Train on
MIXED item+genre reveals (masked/shuffled), reconstruct unrevealed likes (IPS-weighted), JOINT factor learning.
Eval (full + tail NDCG@10): items-only (should still beat ridge -> no regression) and GENRES-only (does folding genre
answers via the encoder beat popularity? = the unified-attribute claim). Compare genres to the additive-genre baseline.
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
Q=np.load(f'{base}/.cache/Q_svd.npy'); bi=np.load(f'{base}/.cache/bi_svd.npy'); Qt=torch.tensor(Q)
ipsw=(1.0/np.clip((cnt/max(cnt.max(),1))**0.5,1e-3,1.0)).astype(np.float32)
order_pop=np.argsort(-cnt); cum=np.cumsum(cnt[order_pop])/cnt.sum(); HEAD=set(int(j) for j in order_pop[:np.searchsorted(cum,0.33)+1])
headmask=np.zeros(ni,bool); headmask[list(HEAD)]=True
# genres
GEN=['Action','Adventure','Animation',"Children's",'Comedy','Crime','Documentary','Drama','Fantasy','Film-Noir','Horror','Musical','Mystery','Romance','Sci-Fi','Thriller','War','Western']
gid={g:k for k,g in enumerate(GEN)}; na=len(GEN); item_g=np.zeros((ni,na),bool)
with open(f'{ml}/movies.dat',encoding='latin-1') as f:
    for line in f:
        pp=line.strip().split('::'); m=int(pp[0])
        if m in iids:
            for g in pp[2].split('|'):
                if g in gid: item_g[iids[m],gid[g]]=True
gcent=np.stack([Q[np.where(item_g[:,g])[0]].mean(0) if item_g[:,g].any() else np.zeros(D) for g in range(na)]).astype(np.float32)
resid_by_u={x:{j:(r-mu-bi[j]) for j,r in rat_by_u[x]} for x in (trU+te)}
def genre_tokens(x, items_avail):       # (factor, value) for genres the user has >=2 of among items_avail
    toks=[]
    for g in range(na):
        gi=[j for j in items_avail if item_g[j,g]]
        if len(gi)>=2: toks.append((gcent[g], float(np.mean([resid_by_u[x][j] for j in gi]))))
    return toks
class Enc(nn.Module):
    def __init__(s):
        super().__init__(); s.inp=nn.Sequential(nn.Linear(D+1,128),nn.ReLU(),nn.Linear(128,128),nn.ReLU())
        s.att=nn.Linear(128,1); s.val=nn.Linear(128,D)
    def forward(s,toks,mask):
        h=s.inp(toks); a=s.att(h).squeeze(-1).masked_fill(mask==0,-1e9); al=torch.softmax(a,1)
        return (al.unsqueeze(-1)*s.val(h)).sum(1)
enc=Enc(); Qp=torch.nn.Parameter(Qt.clone()); opt=torch.optim.Adam(list(enc.parameters())+[Qp],1e-3,weight_decay=1e-5)
trbig=[x for x in trU if len(rat_by_u[x])>=K+1]
def make_batch(users,kk):
    toks=np.zeros((len(users),kk,D+1),np.float32); msk=np.zeros((len(users),kk),np.float32)
    tgt=np.zeros((len(users),ni),np.float32); wt=np.ones((len(users),ni),np.float32); seen=np.zeros((len(users),ni),bool)
    for b,x in enumerate(users):
        its=[j for j,_ in rat_by_u[x]]; rng.shuffle(its)
        gtok=genre_tokens(x,its)
        # build a mixed pool: item tokens + genre tokens; sample kk
        pool=[('i',j) for j in its]+[('g',gi) for gi in range(len(gtok))]
        rng.shuffle(pool); pick=pool[:kk]
        for q,(typ,v) in enumerate(pick):
            if typ=='i': toks[b,q,:D]=Q[v]; toks[b,q,D]=resid_by_u[x][v]; msk[b,q]=1; seen[b,v]=True
            else: toks[b,q,:D]=gtok[v][0]; toks[b,q,D]=gtok[v][1]; msk[b,q]=1
        posw=0.
        for j in likes_by_u[x]:
            if not seen[b,j]: tgt[b,j]=1.0; wt[b,j]=ipsw[j]; posw+=ipsw[j]
        nneg=ni-int(seen[b].sum())-int(tgt[b].sum())
        if nneg>0: wt[b][(tgt[b]==0)&(~seen[b])]=posw/nneg
    return torch.tensor(toks),torch.tensor(msk),torch.tensor(tgt),torch.tensor(wt),torch.tensor(seen)
print(f"train encoder on MIXED item+genre reveals, {len(trbig)} users (K={K})...",flush=True)
for ep in range(int(os.environ.get('EP',30))):
    rng.shuffle(trbig); tot=0.;nb=0
    for b0 in range(0,len(trbig),256):
        us=trbig[b0:b0+256]; kk=int(rng.integers(1,K+1)); toks,msk,tgt,wt,seen=make_batch(us,kk)
        u=enc(toks,msk); sc=u@Qp.t()
        loss=(wt*nn.functional.binary_cross_entropy_with_logits(sc,tgt,reduction='none')).masked_fill(seen,0.).mean()
        opt.zero_grad(); loss.backward(); opt.step(); tot+=loss.item(); nb+=1
    if (ep+1)%10==0: print(f"  ep{ep+1} loss={tot/nb:.4f}",flush=True)
enc.eval(); Ql=Qp.detach().numpy()
_W=1./np.log2(np.arange(2,12))
def ndcg_at(u,tlike,excl,Quse,tailonly):
    sc=(popb+Quse@u).copy(); sc[list(excl)]=-1e9
    if tailonly: sc[headmask]=-1e9; rel=set(t for t in tlike if not headmask[t])
    else: rel=set(tlike)
    top=np.argpartition(-sc,10)[:10]; top=top[np.argsort(-sc[top])]
    return sum(_W[p] for p,t in enumerate(top) if int(t) in rel)/(_W[:min(10,len(rel))].sum()+1e-12) if rel else None
def enc_u(toklist):
    if not toklist: return np.zeros(D)
    arr=np.zeros((1,len(toklist),D+1),np.float32); m=np.ones((1,len(toklist)),np.float32)
    for q,(f,v) in enumerate(toklist): arr[0,q,:D]=f; arr[0,q,D]=v
    with torch.no_grad(): return enc(torch.tensor(arr),torch.tensor(m)).numpy()[0]
helf=None  # use affinity order for genres; item order = popularity within profile
_rs=np.random.default_rng(123); SPL={}
for x in te:
    items=list(dict(rat_by_u[x]))
    if len(items)>=6: il=items[:]; _rs.shuffle(il); SPL[x]=(il[:len(il)//2], il[len(il)//2:])
def evalmode(mode,tailonly,NEVAL=300):
    acc={q:0. for q in [0,1,2,4,8]}; m=0
    for x in te[:NEVAL]:
        if x not in SPL: continue
        test,prof=SPL[x]; rd=dict(rat_by_u[x]); tlike=set(j for j in test if rd[j]>=4); prof=list(prof)
        if len(prof)<4 or not tlike or (tailonly and not any(not headmask[t] for t in tlike)): continue
        if mode=='genres':
            gt=genre_tokens(x,prof); gt=sorted(gt,key=lambda t:-abs(t[1]))   # most-opinionated genres first
            seq=gt
        else:  # items, popularity-within-profile order
            io=sorted(prof,key=lambda j:-cnt[j]); seq=[(Q[j],rd[j]-mu-bi[j]) for j in io]
        for q in [0,1,2,4,8]:
            u=enc_u(seq[:q]); v=ndcg_at(u,tlike,set(prof),Ql,tailonly)
            if v is not None: acc[q]+=v
        m+=1
    return {q:acc[q]/m for q in acc}, m
for tailonly in [False,True]:
    print(f"\n=== {'TAIL' if tailonly else 'FULL'} NDCG@10 (encoder, unified) ===",flush=True)
    for mode in ['items','genres']:
        r,m=evalmode(mode,tailonly); print(f"  {mode:<7}: "+" ".join(f"q{q}={r[q]:.3f}" for q in [0,1,2,4,8])+f" | n={m}",flush=True)
    print(f"  (q0 = popularity baseline)",flush=True)
