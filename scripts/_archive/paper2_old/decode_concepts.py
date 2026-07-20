"""
PAPER B — E-decode (USER IDEA): translate learned/useful EMBEDDING DIRECTIONS back to NL CONCEPT TOKENS.
Level 1 (no new infra): decode a direction by kNN against genome-tag CENTROIDS in the shared space -> tag NAMES.
Demos: (A) INTERPRETABILITY -- decode each user's taste direction u_full -> top tag names (are they sensible vs the
user's actual high-affinity genres?). (B) VERBALIZATION COST -- fold the RAW direction vs fold the DECODED nearest-tag
concept; gap = cost of making the question expressible. Frozen encoder.
"""
import os, time, numpy as np, torch, torch.nn as nn
base='C:/dev/phd/casper/data/movielens'; ml=f'{base}/ml-1m'; D=64; LAM=5.0; THR=0.5; MINIT=30; rng=np.random.default_rng(0)
NEVAL=int(os.environ.get('NEVAL',150))
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
Q=np.load(f'{base}/.cache/Q_svd.npy'); bi=np.load(f'{base}/.cache/bi_svd.npy'); Ql=np.load(f'{base}/.cache/Ql_unified.npy')
order_pop=np.argsort(-cnt); cum=np.cumsum(cnt[order_pop])/cnt.sum(); HEAD=set(int(j) for j in order_pop[:np.searchsorted(cum,0.33)+1]); headmask=np.zeros(ni,bool); headmask[list(HEAD)]=True
resid_by_u={x:{j:(r-mu-bi[j]) for j,r in rat_by_u[x]} for x in te}
GEN=['Action','Adventure','Animation',"Children's",'Comedy','Crime','Documentary','Drama','Fantasy','Film-Noir','Horror','Musical','Mystery','Romance','Sci-Fi','Thriller','War','Western']
gid={g:k for k,g in enumerate(GEN)}; na=len(GEN); item_g=np.zeros((ni,na),bool); title={}
with open(f'{ml}/movies.dat',encoding='latin-1') as f:
    for line in f:
        p=line.strip().split('::'); m=int(p[0])
        if m in iids:
            title[iids[m]]=p[1]
            for g in p[2].split('|'):
                if g in gid: item_g[iids[m],gid[g]]=True
# genome tag centroids + names (shared-space concept vocabulary)
tagname={}
with open(f'{base}/genome-tags.csv',encoding='latin-1') as f:
    next(f)
    for line in f: a=line.rstrip('\n').split(','); tagname[int(a[0])]=a[1]
print("streaming genome...",flush=True); t0=time.time(); tagitems={}
with open(f'{base}/genome-scores.csv') as f:
    next(f)
    for line in f:
        a=line.split(','); m=int(a[0])
        if m in keepids and float(a[2])>THR: tagitems.setdefault(int(a[1]),[]).append(iids[m])
ctags=[t for t,its in tagitems.items() if len(its)>=MINIT]
Ac=np.stack([Q[np.array(tagitems[t])].mean(0) for t in ctags]).astype(np.float32); Acn=Ac/np.clip(np.linalg.norm(Ac,axis=1,keepdims=True),1e-9,None)
citems=[set(tagitems[t]) for t in ctags]; cname=[tagname[t] for t in ctags]
print(f"  {len(ctags)} tag-concepts ({time.time()-t0:.0f}s)",flush=True)
def decode(a,k=6):                                       # embedding direction -> nearest tag NAMES (kNN over centroids)
    an=a/(np.linalg.norm(a)+1e-9); sims=Acn@an; top=np.argsort(-sims)[:k]; return [(cname[i],float(sims[i])) for i in top]
def ridge(F,y): F=np.array(F); return np.linalg.solve(F.T@F+LAM*np.eye(D),F.T@np.array(y,np.float32)) if len(F) else np.zeros(D)
class Enc(nn.Module):
    def __init__(s):
        super().__init__(); s.inp=nn.Sequential(nn.Linear(D+1,128),nn.ReLU(),nn.Linear(128,128),nn.ReLU()); s.att=nn.Linear(128,1); s.val=nn.Linear(128,D)
    def forward(s,t,m):
        h=s.inp(t); a=s.att(h).squeeze(-1).masked_fill(m==0,-1e9); al=torch.softmax(a,1); return (al.unsqueeze(-1)*s.val(h)).sum(1)
enc=Enc(); enc.load_state_dict(torch.load(f'{base}/.cache/enc_unified.pt')); enc.eval()
def enc_u(toks):
    if not toks: return np.zeros(D)
    arr=np.zeros((1,len(toks),D+1),np.float32); m=np.ones((1,len(toks)),np.float32)
    for q,(f,v) in enumerate(toks): arr[0,q,:D]=f; arr[0,q,D]=v
    with torch.no_grad(): return enc(torch.tensor(arr),torch.tensor(m)).numpy()[0]
_W=1./np.log2(np.arange(2,12))
def ndcg(u,tlike,excl,tail):
    sc=(popb+Ql@u).copy(); sc[list(excl)]=-1e9
    if tail: sc[headmask]=-1e9; rel=[t for t in tlike if not headmask[t]]
    else: rel=list(tlike)
    if not rel: return None
    top=np.argpartition(-sc,10)[:10]; top=top[np.argsort(-sc[top])]; rs=set(rel)
    return sum(_W[p] for p,t in enumerate(top) if int(t) in rs)/(_W[:min(10,len(rel))].sum()+1e-12)
_rs=np.random.default_rng(123); SPL={}
for x in te:
    its=list(dict(rat_by_u[x]))
    if len(its)>=6: il=its[:]; _rs.shuffle(il); SPL[x]=(il[:len(il)//2], il[len(il)//2:])
TE=[x for x in te if x in SPL]
# ---------- Demo A: interpretability — decode taste direction u_full -> tag names ----------
print("\n=== (A) INTERPRETABILITY: decode each user's taste direction u_full -> nearest tag concepts ===",flush=True)
for x in TE[:6]:
    test,prof=SPL[x]; rd=dict(rat_by_u[x]); prof=list(prof)
    uf=ridge([Q[j] for j in prof],[rd[j]-mu-bi[j] for j in prof])
    tags=[t for t,_ in decode(uf,5)]
    gens=sorted(range(na),key=lambda g:-np.mean([rd[j]-mu-bi[j] for j in prof if item_g[j,g]] or [0]))[:3]
    liked=[title[j] for j in sorted(prof,key=lambda j:-(rd[j]))[:3]]
    print(f"  u_full -> [{', '.join(tags)}]  | top genres: {[GEN[g] for g in gens]} | liked: {liked[:2]}",flush=True)
# ---------- Demo B: verbalization cost — raw direction fold vs decoded-tag fold ----------
def concept_answer(k,prof,rd):                           # user's affinity for tag-concept k (profile-grounded)
    inb=[j for j in prof if j in citems[k]]
    return (float(np.mean([rd[j]-mu-bi[j] for j in inb])) if len(inb)>=2 else None)
print("\n=== (B) VERBALIZATION COST: fold RAW u_full direction vs fold DECODED nearest-answerable tag ===",flush=True)
for tail in [False,True]:
    raw=dec=q0=0.;m=0
    for x in TE[:NEVAL]:
        test,prof=SPL[x]; rd=dict(rat_by_u[x]); tlike=set(j for j in test if rd[j]>=4); prof=list(prof)
        if len(prof)<4 or not tlike or (tail and not any(not headmask[t] for t in tlike)): continue
        uf=ridge([Q[j] for j in prof],[rd[j]-mu-bi[j] for j in prof])
        # raw: fold (u_full, sim-weighted answer along u_full)
        Qp=Q[prof]; sims=np.maximum((Qp@uf)/(np.linalg.norm(uf)+1e-9)/np.clip(np.linalg.norm(Qp,axis=1),1e-9,None),0); ans=float((sims*np.array([rd[j]-mu-bi[j] for j in prof])).sum()/(sims.sum()+1e-9))
        vraw=ndcg(enc_u([(uf,ans)]),tlike,set(prof),tail)
        # decoded: nearest ANSWERABLE tag to u_full; fold its centroid + the user's affinity
        order=np.argsort(-(Acn@(uf/(np.linalg.norm(uf)+1e-9))))
        vdec=None
        for k in order[:30]:
            ca=concept_answer(k,prof,rd)
            if ca is not None: vdec=ndcg(enc_u([(Ac[k],ca)]),tlike,set(prof),tail); break
        v0=ndcg(np.zeros(D),tlike,set(prof),tail)
        if vraw is not None and vdec is not None: raw+=vraw;dec+=vdec;q0+=v0;m+=1
    print(f"  {'TAIL' if tail else 'FULL'}: q0={q0/m:.3f}  raw-direction={raw/m:.3f}  decoded-tag={dec/m:.3f}  verbalization-cost={ (raw-dec)/m:+.3f}  (n={m})",flush=True)
