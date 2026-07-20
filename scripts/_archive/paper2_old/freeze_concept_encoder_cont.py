"""
PHASE 0 — CONTINUOUS-CAPABLE encoder. Same recipe as V1 freeze_concept_encoder.py (pooled-attention, FROM SCRATCH,
learns enc+Ql+Ec; items + 18 genres + 761 genome concepts; geometric answers; BCE + like/dislike contrastive),
PLUS a 4th token type: CONTINUOUS off-pool directions (random unit q, scaled to concept-norm, GRADED geometric
answer a = interp(NEG,POS; cos(u*,q))). This teaches the encoder to fold ARBITRARY embedding-space queries
IN-DISTRIBUTION (V1 folds them OOD => cos~0.46 plateau that caps the continuous policy). Pooled-attention kept for
A/B/C consistency. Saves {'enc','Ql','Ec'} -> enc_concept_cont.pt (V1 enc_concept.pt UNTOUCHED).
"""
import os, time, numpy as np, torch, torch.nn as nn
base='C:/dev/phd/casper/data/movielens'; ml=f'{base}/ml-1m'; D=64; K=14; rng=np.random.default_rng(0); torch.manual_seed(0)
NCT=int(os.environ.get('NCT','3'))                                              # # continuous off-pool tokens added to each user's candidate pool (post-warmup)
RTYPE=os.environ.get('RTYPE','info')                                            # 'info' = INFORMATIVE off-pool dirs (u*-aligned+noise & concept-pair interps; carry signal) ; 'rand' = orthogonal random (v1, failed=diluted)
SIGMA=float(os.environ.get('SIGMA','0.7'))                                       # off-manifold strength for u*-aligned tokens
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
expo_prop=((cnt/max(cnt.max(),1))**0.5).astype(np.float32)
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
POS=np.mean([resid_by_u[x][j] for x in trU[:3000] for j,r in rat_by_u[x] if r>=4]); NEG=np.mean([resid_by_u[x][j] for x in trU[:3000] for j,r in rat_by_u[x] if r<4])
CL=0.2; MARGIN=0.5
t0=time.time(); tagitems={}
with open(f'{base}/genome-scores.csv') as f:
    next(f)
    for line in f:
        a=line.split(','); m=int(a[0])
        if m in keepids and float(a[2])>0.5: tagitems.setdefault(int(a[1]),[]).append(iids[m])
ctags=[t for t,its in tagitems.items() if len(its)>=30]; nc=len(ctags)
Ac=np.stack([Q[np.array(tagitems[t])].mean(0) for t in ctags]).astype(np.float32)
item2c=[[] for _ in range(ni)]
for ki,t in enumerate(ctags):
    for j in tagitems[t]: item2c[j].append(ki)
print(f"  {nc} genome concepts ({time.time()-t0:.0f}s) | NCT={NCT} continuous tokens/user",flush=True)
def gtok(x,avail):
    out=[]
    for g in range(na):
        gi=[j for j in avail if item_g[j,g]]
        if len(gi)>=2: out.append((gcent[g],float(np.mean([resid_by_u[x][j] for j in gi])),-1))
    return out
USE_GEO=False; GEO={}; USTAR={}; _CN=float(np.linalg.norm(Ac,axis=1).mean())
def ctok(x,avail):
    if USE_GEO:
        out=[(c,v) for c,v in GEO.get(x,{}).items()]
    else:
        cn={}
        for j in avail:
            for ki in item2c[j]: cn.setdefault(ki,[]).append(j)
        out=[(ki,float(np.mean([resid_by_u[x][j] for j in js]))) for ki,js in cn.items() if len(js)>=2]
    rng.shuffle(out); return out[:25]
ECD=Ac.copy(); CSET={}                                                          # current concept embeddings (np) + per-user answerable-concept list (set in precompute_geo)
def rtok(x):                                                                   # CONTINUOUS off-pool tokens. info: u*-aligned+noise (carries signal) & concept-pair interps (on-manifold, off-catalog). rand: orthogonal (v1).
    if not USE_GEO or x not in USTAR: return []
    us=USTAR[x]; un=np.linalg.norm(us)+1e-9; usn=us/un; cs=CSET.get(x,[]); out=[]
    for _ in range(NCT):
        if RTYPE=='rand': d=rng.standard_normal(D).astype(np.float32)
        elif rng.random()<0.5: d=usn+SIGMA*rng.standard_normal(D).astype(np.float32)            # u*-aligned + noise: off-catalog but high |u*.q|
        elif len(cs)>=2: a,b=rng.choice(cs,2,replace=False); d=(ECD[a]+ECD[b]).astype(np.float32)   # concept-pair interpolation: on-manifold, off-catalog
        else: d=usn+SIGMA*rng.standard_normal(D).astype(np.float32)
        dn=d/(np.linalg.norm(d)+1e-9); fe=dn*_CN
        cf=float(us@dn)/un; ans=float(NEG+(POS-NEG)*(cf+1)/2)                   # graded interp (== continuous_actor rollout GRADED)
        out.append((fe.astype(np.float32),ans))
    return out
class Enc(nn.Module):
    def __init__(s):
        super().__init__(); s.inp=nn.Sequential(nn.Linear(D+1,128),nn.ReLU(),nn.Linear(128,128),nn.ReLU()); s.att=nn.Linear(128,1); s.val=nn.Linear(128,D)
    def forward(s,t,m):
        h=s.inp(t); a=s.att(h).squeeze(-1).masked_fill(m==0,-1e9); al=torch.softmax(a,1); return (al.unsqueeze(-1)*s.val(h)).sum(1)
enc=Enc(); Qp=torch.nn.Parameter(Qt.clone()); Ec=torch.nn.Parameter(torch.tensor(Ac))
opt=torch.optim.Adam(list(enc.parameters())+[Qp,Ec],1e-3,weight_decay=1e-5)
trbig=[x for x in trU if len(rat_by_u[x])>=K+1]
def precompute_geo():
    Ecd=Ec.detach().numpy(); G={}; globals()['_CN']=float(np.linalg.norm(Ecd,axis=1).mean()); globals()['ECD']=Ecd
    for b0 in range(0,len(trbig),512):
        us=trbig[b0:b0+512]; mx=60; arr=np.zeros((len(us),mx,D+1),np.float32); msk=np.zeros((len(us),mx),np.float32)
        for b,x in enumerate(us):
            its=[j for j,_ in rat_by_u[x]][:mx]
            for q,j in enumerate(its): arr[b,q,:D]=Q[j]; arr[b,q,D]=resid_by_u[x][j]; msk[b,q]=1
        with torch.no_grad(): ustar=enc(torch.tensor(arr),torch.tensor(msk)).numpy()
        for b,x in enumerate(us):
            USTAR[x]=ustar[b].astype(np.float32)                                # store full-profile taste for continuous-token answers
            cn={}
            for j,_ in rat_by_u[x]:
                for ki in item2c[j]: cn[ki]=cn.get(ki,0)+1
            cs=[ki for ki,nn in cn.items() if nn>=2]
            CSET[x]=cs                                                          # answerable concepts for this user (for concept-pair interpolation tokens)
            if not cs: G[x]={}; continue
            proj=ustar[b]@Ecd[cs].T; thr=proj.mean(); G[x]={cs[k]:(POS if proj[k]>thr else NEG) for k in range(len(cs))}
    return G
def make_batch(us,kk):
    F=np.zeros((len(us),kk,D),np.float32); V=np.zeros((len(us),kk),np.float32); msk=np.zeros((len(us),kk),np.float32)
    tgt=np.zeros((len(us),ni),np.float32); wt=np.ones((len(us),ni),np.float32); seen=np.zeros((len(us),ni),bool); cpos=[]
    for b,x in enumerate(us):
        its=[j for j,_ in rat_by_u[x]]; rng.shuffle(its)
        pool=[('i',j) for j in its]+[('g',v) for v in gtok(x,its)]+[('c',v) for v in ctok(x,its)]+[('r',v) for v in rtok(x)]; rng.shuffle(pool)
        for q,(typ,v) in enumerate(pool[:kk]):
            if typ=='i': F[b,q]=Q[v]; V[b,q]=resid_by_u[x][v]; msk[b,q]=1; seen[b,v]=True
            elif typ=='g': F[b,q]=v[0]; V[b,q]=v[1]; msk[b,q]=1
            elif typ=='r': F[b,q]=v[0]; V[b,q]=v[1]; msk[b,q]=1                  # CONTINUOUS off-pool token (feature IS the direction; no learned embedding)
            else: V[b,q]=v[1]; msk[b,q]=1; cpos.append((b,q,v[0]))
        posw=0.
        for j in likes_by_u[x]:
            if not seen[b,j]: tgt[b,j]=1.;wt[b,j]=ipsw[j];posw+=ipsw[j]
        negmask=(tgt[b]==0)&(~seen[b])
        if negmask.any(): pw=expo_prop[negmask]; wt[b][negmask]=posw*(pw/pw.sum())
    return F,V,torch.tensor(msk),torch.tensor(tgt),torch.tensor(wt),torch.tensor(seen),cpos
print("training CONTINUOUS-CAPABLE concept encoder...",flush=True)
for ep in range(int(os.environ.get('EP','30'))):
    globals()['USE_GEO']= ep>=8
    if USE_GEO: globals()['GEO']=precompute_geo()
    rng.shuffle(trbig)
    for b0 in range(0,len(trbig),256):
        us=trbig[b0:b0+256]; kk=int(rng.integers(1,K+1)); F,V,m,tg,w,se,cpos=make_batch(us,kk)
        feat=torch.tensor(F)
        if cpos:
            cf=torch.zeros(F.shape[0],kk,D); bi_=torch.tensor([p[0] for p in cpos]); qi_=torch.tensor([p[1] for p in cpos]); ci_=torch.tensor([p[2] for p in cpos])
            cf[bi_,qi_]=Ec[ci_]; feat=feat+cf
        t=torch.cat([feat,torch.tensor(V).unsqueeze(-1)],dim=-1); u=enc(t,m); sc=u@Qp.t()
        loss=(w*nn.functional.binary_cross_entropy_with_logits(sc,tg,reduction='none')).masked_fill(se,0.).mean()
        jt=torch.randint(0,ni,(256,)); o=torch.ones(256,1)
        tp=torch.zeros(256,1,D+1); tp[:,0,:D]=Qt[jt]; tp[:,0,D]=POS
        tm=torch.zeros(256,1,D+1); tm[:,0,:D]=Qt[jt]; tm[:,0,D]=NEG
        up=enc(tp,o); um=enc(tm,o); diff=(Qp[jt]*up).sum(1)-(Qp[jt]*um).sum(1)
        loss=loss+CL*torch.nn.functional.softplus(MARGIN-diff).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    if (ep+1)%5==0: print(f"  ep{ep+1} loss={loss.item():.4f}",flush=True)
_out=os.environ.get('OUT',f'enc_concept_cont_{RTYPE}.pt')
torch.save({'enc':enc.state_dict(),'Ql':Qp.detach(),'Ec':Ec.detach()}, f'{base}/.cache/{_out}')
print(f"SAVED {_out} (continuous-capable pooled-attn; enc+Ql+Ec; RTYPE={RTYPE} NCT={NCT}). V1 enc_concept.pt untouched.",flush=True)
