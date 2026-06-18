"""
ATTRIBUTE-ONLY ORACLE (genres + genome concepts): is there ANY headroom for attribute elicitation? If a privileged
oracle that greedily picks the best attribute to reveal can't beat the no-question baseline, then no realizable
attribute-elicitation baseline (EAR/SCPR/UNICORN/PEBOL...) will either -> don't build them.

Parallel to the item oracle. Calibrated biased-SVD instrument (Q_svd/bi_svd). Attribute = pseudo-item whose factor
is the taste-space centroid of its items; the user's ANSWER = realizable affinity = (relevance-weighted) mean
taste-residual (r-mu-bi) over their PROFILE items of that attribute (the honest profile answer model, leakage-free).
ORACLE = greedy privileged selection on held-out NDCG. Compare: genre-oracle, concept-oracle, both; vs realizable
affinity-order; vs q0; vs ITEM-oracle (reference ceiling).
"""
import os, numpy as np, scipy.linalg as sla
base='C:/dev/phd/casper/data/movielens'; ml=f'{base}/ml-1m'; D=64; T=int(os.environ.get('T',8)); LAM=5.0
NU=int(os.environ.get('NU',120)); NCG=int(os.environ.get('NCG',100)); rng=np.random.default_rng(0)
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
trU=set(keep[:int(0.8*n)]); te=keep[int(0.9*n):]
cnt=np.zeros(ni)
for x in trU:
    for j in likes_by_u.get(x,[]): cnt[j]+=1
popb=np.log(cnt+1.0).astype(np.float32)
sm=c=0.
for x in trU:
    for j,r in rat_by_u[x]: sm+=r; c+=1
mu=sm/c
Q=np.load(f'{base}/.cache/Q_svd.npy'); bi=np.load(f'{base}/.cache/bi_svd.npy')
# genres
GEN=['Action','Adventure','Animation',"Children's",'Comedy','Crime','Documentary','Drama','Fantasy','Film-Noir','Horror','Musical','Mystery','Romance','Sci-Fi','Thriller','War','Western']
gid={g:k for k,g in enumerate(GEN)}; item_g=np.zeros((ni,len(GEN)),bool)
with open(f'{ml}/movies.dat',encoding='latin-1') as f:
    for line in f:
        pp=line.strip().split('::'); m=int(pp[0])
        if m in iids:
            for g in pp[2].split('|'):
                if g in gid: item_g[iids[m],gid[g]]=True
# genome concepts (top NCG by coverage, excluding genre-name tags)
THR=0.5; tagname={}
with open(f'{base}/genome-tags.csv',encoding='latin-1') as f:
    next(f)
    for line in f: a=line.rstrip('\n').split(','); tagname[int(a[0])]=a[1]
tcount={}
with open(f'{base}/genome-scores.csv') as f:
    next(f)
    for line in f:
        a=line.split(','); m=int(a[0])
        if m in iids and float(a[2])>THR: tcount[int(a[1])]=tcount.get(int(a[1]),0)+1
glow=set(g.lower() for g in GEN)
ctags=[t for t,_ in sorted(tcount.items(),key=lambda kv:-kv[1]) if tagname[t] not in glow][:NCG]; cidx={t:k for k,t in enumerate(ctags)}
relg=np.zeros((ni,len(ctags)),np.float32)
with open(f'{base}/genome-scores.csv') as f:
    next(f)
    for line in f:
        a=line.split(','); m=int(a[0]); t=int(a[1])
        if m in iids and t in cidx: relg[iids[m],cidx[t]]=float(a[2])
# attribute centroids in taste space
gcent=np.stack([Q[np.where(item_g[:,g])[0]].mean(0) if item_g[:,g].any() else np.zeros(D) for g in range(len(GEN))])
ccent=np.stack([ (relg[:,c:c+1]*Q).sum(0)/(relg[:,c].sum()+1e-9) for c in range(len(ctags)) ])
print(f"{len(te)} test users; genres={len(GEN)} concepts={len(ctags)}; T={T}\n",flush=True)
SPL={}; _rs=np.random.default_rng(123)
for x in te:
    items=list(dict(rat_by_u[x]))
    if len(items)>=6: il=items[:]; _rs.shuffle(il); SPL[x]=(set(il[:len(il)//2]),set(il[len(il)//2:]))
def ndcg(score,rel,excl):
    s=score.copy(); s[list(excl)]=-1e9; top=np.argsort(-s)[:10]; rs=set(rel)
    return sum(1./np.log2(p+2) for p,t in enumerate(top) if t in rs)/ (sum(1./np.log2(p+2) for p in range(min(10,len(rel))))+1e-12)
def affinities(x, prof, rd):
    A=[]; aff=[]
    for g in range(len(GEN)):
        gi=[j for j in prof if item_g[j,g]]
        if len(gi)>=2: A.append(gcent[g]); aff.append(np.mean([rd[j]-mu-bi[j] for j in gi]))
    for c in range(len(ctags)):
        w=relg[list(prof),c]; sw=w.sum()
        if (w>THR).sum()>=2: A.append(ccent[c]); aff.append(float((w*np.array([rd[j]-mu-bi[j] for j in prof])).sum()/(sw+1e-9)))
    return A,aff
def foldin(F,y): F=np.array(F); return np.linalg.solve(F.T@F+LAM*np.eye(D),F.T@np.array(y,np.float32)) if len(F) else np.zeros(D)
def run(mode):                       # mode: 'oracle_all','oracle_gen','oracle_con','affinity_all','item_oracle'
    ND=np.zeros(T+1); m=0
    for x in te[:NU]:
        if x not in SPL: continue
        rd=dict(rat_by_u[x]); test,prof=SPL[x]; tlike=set(j for j in test if rd[j]>=4)
        if not tlike or not prof: continue
        nd=np.zeros(T+1); nd[0]=ndcg(popb.copy(),tlike,prof)
        if mode=='item_oracle':
            F=[];y=[];asked=set()
            for t in range(1,T+1):
                best=None
                for cc in prof:
                    if cc in asked: continue
                    u=foldin(F+[Q[cc]],y+[rd[cc]-mu-bi[cc]]); a=ndcg(popb+Q@u,tlike,prof)
                    if best is None or a>best[0]: best=(a,cc)
                if best is None: nd[t:]=nd[t-1]; break
                asked.add(best[1]); F.append(Q[best[1]]); y.append(rd[best[1]]-mu-bi[best[1]]); nd[t]=best[0]
            ND+=nd; m+=1; continue
        A,aff=affinities(x,prof,rd)
        if not A: ND+=nd; m+=1; continue
        # restrict pool
        idxs=list(range(len(A)))
        if mode=='oracle_gen': idxs=[i for i in idxs if i<len([1 for g in range(len(GEN)) if sum(1 for j in prof if item_g[j,g])>=2])]
        F=[];y=[];used=set()
        if mode=='affinity_all':
            order=sorted(idxs,key=lambda i:-abs(aff[i]))
            for t in range(1,T+1):
                if t-1<len(order): F.append(A[order[t-1]]); y.append(aff[order[t-1]])
                u=foldin(F,y); nd[t]=ndcg(popb+Q@u,tlike,prof)
        else:                         # greedy oracle over attribute pool
            pool=idxs
            for t in range(1,T+1):
                best=None
                for i in pool:
                    if i in used: continue
                    u=foldin(F+[A[i]],y+[aff[i]]); a=ndcg(popb+Q@u,tlike,prof)
                    if best is None or a>best[0]: best=(a,i)
                if best is None: nd[t:]=nd[t-1]; break
                used.add(best[1]); F.append(A[best[1]]); y.append(aff[best[1]]); nd[t]=best[0]
        ND+=nd; m+=1
    return ND/m,m
for mode in ['affinity_all','oracle_all','item_oracle']:
    nd,m=run(mode); print(f"{mode:<14} (n={m}): "+" ".join(f"{v:.3f}" for v in nd)+f"  | delta {nd[T]-nd[0]:+.3f}",flush=True)
