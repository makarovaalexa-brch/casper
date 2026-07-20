"""
PAPER B — T3 (concept FOLDING fix): retrain the encoder WITH genome-concept reveals (it was trained on items+genres
only => genome concepts are OOD). Concept token = (tag centroid, user's affinity). Train mixed item+genre+concept
reveals, reconstruct unrevealed likes (IPS). Then measure PROFILE RECONSTRUCTION (tail Recall@50 + recon-AUC) for
item vs concept asking with the NEW encoder vs the frozen one. Gate T3: concept asking value rises with proper folding.
"""
import os, time, numpy as np, torch, torch.nn as nn
base='C:/dev/phd/casper/data/movielens'; ml=f'{base}/ml-1m'; D=64; K=12; THR=0.5; MINIT=30; EP=int(os.environ.get('EP',20)); rng=np.random.default_rng(0); torch.manual_seed(0)
U,I,R=[],[],[]
with open(f'{ml}/ratings.dat') as f:
    for line in f:
        a=line.strip().split('::'); U.append(int(a[0])); I.append(int(a[1])); R.append(float(a[2]))
U=np.array(U);I=np.array(I);R=np.array(R,np.float32)
uids={x:k for k,x in enumerate(np.unique(U))}; iids={x:k for k,x in enumerate(np.unique(I))}; nu,ni=len(uids),len(iids); keepids=set(iids)
uu=np.array([uids[x] for x in U]); ii=np.array([iids[x] for x in I])
rat_by_u={}
for k in range(len(uu)): rat_by_u.setdefault(uu[k],[]).append((ii[k],float(R[k])))
likes_by_u={x:[j for j,r in v if r>=4] for x,v in rat_by_u.items()}
keep=[x for x in range(nu) if len(likes_by_u.get(x,[]))>=5]; rng.shuffle(keep); n=len(keep); trU=keep[:int(0.8*n)]; te=keep[int(0.9*n):]
cnt=np.zeros(ni)
for x in trU:
    for j in likes_by_u.get(x,[]): cnt[j]+=1
popb=np.log(cnt+1.0).astype(np.float32); sm=c=0.
for x in trU:
    for j,r in rat_by_u[x]: sm+=r;c+=1
mu=sm/c
Q=np.load(f'{base}/.cache/Q_svd.npy'); bi=np.load(f'{base}/.cache/bi_svd.npy'); Qt=torch.tensor(Q)
ipsw=(1.0/np.clip((cnt/max(cnt.max(),1))**0.5,1e-3,1.0)).astype(np.float32)
order_pop=np.argsort(-cnt); cum=np.cumsum(cnt[order_pop])/cnt.sum(); HEAD=set(int(j) for j in order_pop[:np.searchsorted(cum,0.33)+1]); headmask=np.zeros(ni,bool); headmask[list(HEAD)]=True
GEN=['Action','Adventure','Animation',"Children's",'Comedy','Crime','Documentary','Drama','Fantasy','Film-Noir','Horror','Musical','Mystery','Romance','Sci-Fi','Thriller','War','Western']
gid={g:k for k,g in enumerate(GEN)}; na=len(GEN); item_g=np.zeros((ni,na),bool)
with open(f'{ml}/movies.dat',encoding='latin-1') as f:
    for line in f:
        p=line.strip().split('::'); m=int(p[0])
        if m in iids:
            for g in p[2].split('|'):
                if g in gid: item_g[iids[m],gid[g]]=True
gcent=np.stack([Q[np.where(item_g[:,g])[0]].mean(0) for g in range(na)]).astype(np.float32); gitems=[set(np.where(item_g[:,g])[0].tolist()) for g in range(na)]
print("streaming genome...",flush=True); t0=time.time(); tagitems={}
with open(f'{base}/genome-scores.csv') as f:
    next(f)
    for line in f:
        a=line.split(','); m=int(a[0])
        if m in keepids and float(a[2])>THR: tagitems.setdefault(int(a[1]),[]).append(iids[m])
ctags=[t for t,its in tagitems.items() if len(its)>=MINIT]; nc=len(ctags)
Ac=np.stack([Q[np.array(tagitems[t])].mean(0) for t in ctags]).astype(np.float32); citems=[set(tagitems[t]) for t in ctags]
item2c=[[] for _ in range(ni)]
for ki,t in enumerate(ctags):
    for j in tagitems[t]: item2c[j].append(ki)
print(f"  {nc} concepts; item->concept map ({time.time()-t0:.0f}s)",flush=True)
resid={x:{j:(r-mu-bi[j]) for j,r in rat_by_u[x]} for x in (trU+te)}
def user_concepts(x, items):                              # concepts the user has >=2 items in -> (centroid, affinity)
    cnts={}
    for j in items:
        for ki in item2c[j]: cnts[ki]=cnts.get(ki,[])+[j]
    return [(Ac[ki], float(np.mean([resid[x][j] for j in js]))) for ki,js in cnts.items() if len(js)>=2]
def user_genres(x, items):
    out=[]
    for g in range(na):
        gi=[j for j in items if item_g[j,g]]
        if len(gi)>=2: out.append((gcent[g], float(np.mean([resid[x][j] for j in gi]))))
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
        its=[j for j,_ in rat_by_u[x]]; rng.shuffle(its)
        gt=user_genres(x,its); ct=user_concepts(x,its)
        pool=[('i',j) for j in its]+[('g',k) for k in range(len(gt))]+[('c',k) for k in range(len(ct))]; rng.shuffle(pool)
        for q,(typ,v) in enumerate(pool[:kk]):
            if typ=='i': toks[b,q,:D]=Q[v]; toks[b,q,D]=resid[x][v]; msk[b,q]=1; seen[b,v]=True
            elif typ=='g': toks[b,q,:D]=gt[v][0]; toks[b,q,D]=gt[v][1]; msk[b,q]=1
            else: toks[b,q,:D]=ct[v][0]; toks[b,q,D]=ct[v][1]; msk[b,q]=1
        posw=0.
        for j in likes_by_u[x]:
            if not seen[b,j]: tgt[b,j]=1.;wt[b,j]=ipsw[j];posw+=ipsw[j]
        nneg=ni-int(seen[b].sum())-int(tgt[b].sum())
        if nneg>0: wt[b][(tgt[b]==0)&(~seen[b])]=posw/nneg
    return torch.tensor(toks),torch.tensor(msk),torch.tensor(tgt),torch.tensor(wt),torch.tensor(seen)
print(f"retrain encoder WITH concepts ({len(trbig)} users, EP={EP})...",flush=True)
for ep in range(EP):
    rng.shuffle(trbig)
    for b0 in range(0,len(trbig),256):
        t,m,tg,w,se=make_batch(trbig[b0:b0+256],int(rng.integers(1,K+1))); u=enc(t,m); sc=u@Qp.t()
        loss=(w*nn.functional.binary_cross_entropy_with_logits(sc,tg,reduction='none')).masked_fill(se,0.).mean(); opt.zero_grad(); loss.backward(); opt.step()
    if (ep+1)%10==0: print(f"  ep{ep+1} loss={loss.item():.4f}",flush=True)
enc.eval(); Ql=Qp.detach().numpy()
def enc_u(toks):
    if not toks: return np.zeros(D)
    arr=np.zeros((1,len(toks),D+1),np.float32); mk=np.ones((1,len(toks)),np.float32)
    for q,(f,v) in enumerate(toks): arr[0,q,:D]=f; arr[0,q,D]=v
    with torch.no_grad(): return enc(torch.tensor(arr),torch.tensor(mk)).numpy()[0]
allidx=np.arange(ni)
def recall_auc(u,tlike,excl,tail):
    sc=(popb+Ql@u).copy(); sc[list(excl)]=-1e9
    cand=np.array([j for j in allidx if (j not in excl) and (not headmask[j] if tail else True)]); s=sc[cand]; pos=np.array([1 if j in tlike else 0 for j in cand]); npos=pos.sum(); nneg=len(pos)-npos
    if npos==0 or nneg==0: return None
    order=np.argsort(s); ranks=np.empty(len(s)); ranks[order]=np.arange(1,len(s)+1); auc=(ranks[pos==1].sum()-npos*(npos+1)/2)/(npos*nneg)
    top=cand[np.argsort(-s)[:50]]; rec=len(set(top.tolist())&tlike)/npos; return auc,rec
_rs=np.random.default_rng(123); SPL={}
for x in te:
    its=list(dict(rat_by_u[x]))
    if len(its)>=6: il=its[:]; _rs.shuffle(il); SPL[x]=(il[:len(il)//2], il[len(il)//2:])
TE=[x for x in te if x in SPL][:200]
def run(mode,tail):
    Rc={q:0. for q in [0,1,2,4,8]};Au={q:0. for q in [0,1,2,4,8]};m=0
    for x in TE:
        test,prof=SPL[x]; rd=dict(rat_by_u[x]); tlike=set(j for j in test if rd[j]>=4); prof=list(prof)
        if len(prof)<4 or not tlike or (tail and not any(not headmask[t] for t in tlike)): continue
        if mode=='item': cand=[(Q[j],rd[j]-mu-bi[j]) for j in prof]
        else: cand=user_concepts(x,prof)
        rng.shuffle(cand)                                  # random-answerable order (selection~=random, T1)
        for q in [0,1,2,4,8]:
            ra=recall_auc(enc_u(cand[:q]),tlike,set(prof),tail)
            if ra: Au[q]+=ra[0];Rc[q]+=ra[1]
        m+=1
    return {q:Au[q]/m for q in Au},{q:Rc[q]/m for q in Rc},m
for tail in [False,True]:
    print(f"\n=== T3 RETRAINED-encoder reconstruction ({'TAIL' if tail else 'FULL'}) ===",flush=True)
    for mode in ['item','concept']:
        Au,Rc,m=run(mode,tail); print(f"  {mode:<8}: recAUC "+" ".join(f"{Au[q]:.3f}" for q in [0,1,2,4,8])+" | Rec@50 "+" ".join(f"{Rc[q]:.3f}" for q in [0,1,2,4,8])+f" | n={m}",flush=True)
print("(compare concept Rec@50 tail to FROZEN encoder T1: 0.107->0.171)",flush=True)
