"""
ADAPTIVE selection baselines on the reconstruction encoder, vs static + oracle, full + tail.
 (1) GOLBANDI adaptive decision tree (Golbandi/Koren/Lempel WSDM2011): greedy ternary tree (like/dislike/unknown)
     over popular candidates, built on train users to minimize within-node variance of the full-profile ridge latent;
     route each test user -> adaptive asked sequence -> fold answered items via the encoder.
 (2) INFO-GAIN policy (realizable, Phase B): at each step pick the unrevealed PROFILE item that maximizes the
     encoder's predicted like-coverage of the REST of the profile (dense reconstruction proxy; no test peek).
Static refs: random, HELF. Ceiling: oracle (greedy peek). Question: does ADAPTIVE beat random/static where static didn't?
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
print(f"train encoder (joint)...",flush=True)
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
def enc_u_batch(revs):                     # batch encode a list of reveal-lists
    if not revs: return np.zeros((0,D))
    mx=max(len(r) for r in revs); arr=np.zeros((len(revs),mx,D+1),np.float32); m=np.zeros((len(revs),mx),np.float32)
    for b,rev in enumerate(revs):
        for q,(j,res) in enumerate(rev): arr[b,q,:D]=Q[j]; arr[b,q,D]=res; m[b,q]=1
    with torch.no_grad(): return enc(torch.tensor(arr),torch.tensor(m)).numpy()
# ---------- Golbandi tree (global, on popular candidates; impurity = within-node var of full-profile ridge latent) ----------
def ridge_u(x):
    F=np.array([Q[j] for j,_ in rat_by_u[x]]); y=np.array([resid_by_u[x][j] for j,_ in rat_by_u[x]],np.float32)
    return np.linalg.solve(F.T@F+LAM*np.eye(D),F.T@y) if len(F) else np.zeros(D)
CAND=list(np.argsort(-cnt)[:80]); GTR=rng.choice(trU,size=min(2500,len(trU)),replace=False)
Ucache={x:ridge_u(x) for x in GTR}; like_of={x:set(likes_by_u[x]) for x in GTR}; rated_of={x:set(j for j,_ in rat_by_u[x]) for x in GTR}
def impurity(users):
    if len(users)<2: return 0.
    M=np.stack([Ucache[x] for x in users]); return ((M-M.mean(0))**2).sum()
def build(users,depth,used):
    node={'item':None}
    if depth==0 or len(users)<10: node['leaf']=True; return node
    best=None
    for it in CAND:
        if it in used: continue
        L=[x for x in users if it in like_of[x]]; Dd=[x for x in users if it in rated_of[x] and it not in like_of[x]]; Uk=[x for x in users if it not in rated_of[x]]
        imp=impurity(L)+impurity(Dd)+impurity(Uk)
        if best is None or imp<best[0]: best=(imp,it,L,Dd,Uk)
    _,it,L,Dd,Uk=best; node={'item':it,'leaf':False,'used':used|{it}}
    node['L']=build(L,depth-1,used|{it}); node['D']=build(Dd,depth-1,used|{it}); node['Uk']=build(Uk,depth-1,used|{it})
    return node
print("building Golbandi tree...",flush=True); TREE=build(list(GTR),8,set())
def golbandi_path(x,rd,maxq):              # adaptive asked items for test user x (path), depth maxq
    node=TREE; path=[]
    while node and not node.get('leaf') and node['item'] is not None and len(path)<maxq:
        it=node['item']; path.append(it)
        if it in rd and rd[it]>=4: node=node['L']
        elif it in rd: node=node['D']
        else: node=node['Uk']
    return path
_rs=np.random.default_rng(123); SPL={}
for x in te:
    items=list(dict(rat_by_u[x]))
    if len(items)>=6: il=items[:]; _rs.shuffle(il); SPL[x]=(il[:len(il)//2], il[len(il)//2:])
def run(selector,tailonly,NEVAL=250):
    acc={q:0. for q in [0,1,2,4,8]}; m=0
    for x in te[:NEVAL]:
        if x not in SPL: continue
        test,prof=SPL[x]; rd=dict(rat_by_u[x]); tlike=set(j for j in test if rd[j]>=4); prof=list(prof)
        if len(prof)<4 or not tlike or (tailonly and not any(not headmask[t] for t in tlike)): continue
        for q in [0,1,2,4,8]:
            if selector=='random': o=prof[:]; rng.shuffle(o); rev=[(j,rd[j]-mu-bi[j]) for j in o[:q]]
            elif selector=='helf': o=sorted(prof,key=lambda j:-helf[j]); rev=[(j,rd[j]-mu-bi[j]) for j in o[:q]]
            elif selector=='golbandi':
                path=golbandi_path(x,rd,q); rev=[(j,rd[j]-mu-bi[j]) for j in path if j in rd]  # fold answered path items
            elif selector=='infogain':                       # realizable: greedy max predicted coverage of rest of profile
                asked=[]; rev=[]
                while len(asked)<q:
                    cands=[j for j in prof if j not in asked]
                    if not cands: break
                    cu=enc_u_batch([rev+[(c,rd[c]-mu-bi[c])] for c in cands])      # (C,D)
                    rest=[j for j in prof if j not in asked]; sccov=(cu@Ql[rest].T); cov=(1/(1+np.exp(-sccov))).sum(1)  # predicted like-mass over rest
                    c=cands[int(cov.argmax())]; asked.append(c); rev.append((c,rd[c]-mu-bi[c]))
            else:                                            # oracle greedy peek
                asked=[]; rev=[]
                while len(asked)<q:
                    cands=[j for j in prof if j not in asked]; best=None
                    cu=enc_u_batch([rev+[(c,rd[c]-mu-bi[c])] for c in cands])
                    for idx,c in enumerate(cands):
                        v=ndcg_at(cu[idx],tlike,set(prof)|set(asked)|{c},tailonly)
                        if v is not None and (best is None or v>best[0]): best=(v,c)
                    c=best[1] if best else cands[0]; asked.append(c); rev.append((c,rd[c]-mu-bi[c]))
            v=ndcg_at(enc_u(rev),tlike,set(prof),tailonly)
            if v is not None: acc[q]+=v
        m+=1
    return {q:acc[q]/m for q in acc}, m
for tailonly in [False,True]:
    print(f"\n=== {'TAIL' if tailonly else 'FULL'} NDCG@10 — adaptive vs static vs oracle (encoder) ===",flush=True)
    for sel in ['random','helf','golbandi','infogain','oracle']:
        r,m=run(sel,tailonly); print(f"  {sel:<9}: "+" ".join(f"q{q}={r[q]:.3f}" for q in [0,1,2,4,8])+f" | +{r[8]-r[0]:+.3f}",flush=True)
