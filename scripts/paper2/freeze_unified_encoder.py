"""Freeze the canonical UNIFIED encoder (the one behind RESULTS PART I) so it is never lost.
Deterministic (seed 0, EP30, K12) => reproduces the PART I model. Saves enc state_dict + learned factors Ql."""
import os, numpy as np, torch, torch.nn as nn
base='C:/dev/phd/casper/data/movielens'; ml=f'{base}/ml-1m'; D=64; K=12; rng=np.random.default_rng(0); torch.manual_seed(0)
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
keep=[x for x in range(nu) if len(likes_by_u.get(x,[]))>=5]; rng.shuffle(keep); n=len(keep); trU=keep[:int(0.8*n)]
cnt=np.zeros(ni)
for x in trU:
    for j in likes_by_u.get(x,[]): cnt[j]+=1
sm=c=0.
for x in trU:
    for j,r in rat_by_u[x]: sm+=r;c+=1
mu=sm/c
Q=np.load(f'{base}/.cache/Q_svd.npy'); bi=np.load(f'{base}/.cache/bi_svd.npy'); Qt=torch.tensor(Q)
ipsw=(1.0/np.clip((cnt/max(cnt.max(),1))**0.5,1e-3,1.0)).astype(np.float32)
expo_prop=((cnt/max(cnt.max(),1))**0.5).astype(np.float32)                    # ADOPTED: MNAR exposure-propensity negatives
GEN=['Action','Adventure','Animation',"Children's",'Comedy','Crime','Documentary','Drama','Fantasy','Film-Noir','Horror','Musical','Mystery','Romance','Sci-Fi','Thriller','War','Western']
gid={g:k for k,g in enumerate(GEN)}; na=len(GEN); item_g=np.zeros((ni,na),bool)
with open(f'{ml}/movies.dat',encoding='latin-1') as f:
    for line in f:
        pp=line.strip().split('::'); m=int(pp[0])
        if m in iids:
            for g in pp[2].split('|'):
                if g in gid: item_g[iids[m],gid[g]]=True
gcent=np.stack([Q[np.where(item_g[:,g])[0]].mean(0) if item_g[:,g].any() else np.zeros(D) for g in range(na)]).astype(np.float32)
resid_by_u={x:{j:(r-mu-bi[j]) for j,r in rat_by_u[x]} for x in trU}
def gtok(x,avail):
    out=[]
    for g in range(na):
        gi=[j for j in avail if item_g[j,g]]
        if len(gi)>=2: out.append((gcent[g],float(np.mean([resid_by_u[x][j] for j in gi]))))
    return out
class Enc(nn.Module):
    def __init__(s):
        super().__init__(); s.inp=nn.Sequential(nn.Linear(D+1,128),nn.ReLU(),nn.Linear(128,128),nn.ReLU()); s.att=nn.Linear(128,1); s.val=nn.Linear(128,D)
    def forward(s,t,m):
        h=s.inp(t); a=s.att(h).squeeze(-1).masked_fill(m==0,-1e9); al=torch.softmax(a,1); return (al.unsqueeze(-1)*s.val(h)).sum(1)
enc=Enc(); Qp=torch.nn.Parameter(Qt.clone()); opt=torch.optim.Adam(list(enc.parameters())+[Qp],1e-3,weight_decay=1e-5)
trbig=[x for x in trU if len(rat_by_u[x])>=K+1]
def make_batch(us,kk):
    toks=np.zeros((len(us),kk,D+1),np.float32); msk=np.zeros((len(us),kk),np.float32); tgt=np.zeros((len(us),ni),np.float32); wt=np.ones((len(us),ni),np.float32); seen=np.zeros((len(us),ni),bool)
    for b,x in enumerate(us):
        its=[j for j,_ in rat_by_u[x]]; rng.shuffle(its); gt=gtok(x,its); pool=[('i',j) for j in its]+[('g',k) for k in range(len(gt))]; rng.shuffle(pool)
        for q,(typ,v) in enumerate(pool[:kk]):
            if typ=='i': toks[b,q,:D]=Q[v]; toks[b,q,D]=resid_by_u[x][v]; msk[b,q]=1; seen[b,v]=True
            else: toks[b,q,:D]=gt[v][0]; toks[b,q,D]=gt[v][1]; msk[b,q]=1
        posw=0.
        for j in likes_by_u[x]:
            if not seen[b,j]: tgt[b,j]=1.;wt[b,j]=ipsw[j];posw+=ipsw[j]
        negmask=(tgt[b]==0)&(~seen[b])
        if negmask.any(): pw=expo_prop[negmask]; wt[b][negmask]=posw*(pw/pw.sum())   # EXPO weighting
    return torch.tensor(toks),torch.tensor(msk),torch.tensor(tgt),torch.tensor(wt),torch.tensor(seen)
print("training canonical unified encoder...",flush=True)
for ep in range(30):
    rng.shuffle(trbig)
    for b0 in range(0,len(trbig),256):
        us=trbig[b0:b0+256]; kk=int(rng.integers(1,K+1)); t,m,tg,w,se=make_batch(us,kk); u=enc(t,m); sc=u@Qp.t()
        loss=(w*nn.functional.binary_cross_entropy_with_logits(sc,tg,reduction='none')).masked_fill(se,0.).mean(); opt.zero_grad(); loss.backward(); opt.step()
torch.save(enc.state_dict(), f'{base}/.cache/enc_unified.pt'); np.save(f'{base}/.cache/Ql_unified.npy', Qp.detach().numpy())
print(f"SAVED enc_unified.pt + Ql_unified.npy to {base}/.cache/",flush=True)
