"""
Standard cold-start MF folding-in elicitation (RBMF/RMVA lineage) — the method most
likely to reproduce a genuine elicitation>popularity win on ML-1M.
  - Train implicit WRMF (ALS) on training users -> item factors Q[ni,d], item bias.
  - Cold user: given ratings on k SEED items, FOLD IN user factor p by ridge least-squares
    over the seed item factors, then rank all items by Q p (+ pop bias).
  - Seed selection arms: POPULAR, RANDOM(avg), REPRESENTATIVE (greedy max-variance / coverage).
Metric: STANDARD NDCG@10 / Recall@10, ranking all items except seeds. GT = held-out positives.
Compared against MOSTPOP (no elicitation).  [paper ML-1M: MOSTPOP 0.3921, RMVA 0.5387, DRE 0.5688]
"""
import os, time, numpy as np
base='C:/dev/phd/casper/data/movielens/ml-1m'
K=int(os.environ.get('K',50)); D=int(os.environ.get('D',64)); ALPHA=float(os.environ.get('ALPHA',20))
LAM=float(os.environ.get('LAM',0.1)); ITERS=int(os.environ.get('ITERS',15)); NRAN=int(os.environ.get('NRAN',3))
LIKE=float(os.environ.get('LIKE',4.0)); rng=np.random.default_rng(0)
U,I,R=[],[],[]
with open(f'{base}/ratings.dat') as f:
    for line in f:
        a=line.strip().split('::'); U.append(int(a[0])); I.append(int(a[1])); R.append(float(a[2]))
U=np.array(U); I=np.array(I); R=np.array(R,np.float32)
uids={x:k for k,x in enumerate(np.unique(U))}; iids={x:k for k,x in enumerate(np.unique(I))}
u=np.array([uids[x] for x in U]); i=np.array([iids[x] for x in I]); nu,ni=len(uids),len(iids)
B=np.zeros((nu,ni),np.float32)
for k in range(len(u)):
    if R[k]>=LIKE: B[u[k],i[k]]=1.0
keep=np.where(B.sum(1)>=5)[0]; rng.shuffle(keep); n=len(keep)
tr=keep[:int(0.8*n)]; va=keep[int(0.8*n):int(0.9*n)]; te=keep[int(0.9*n):]
cnt=B[tr].sum(0); pop_bias=np.log(cnt+1.0).astype(np.float32); pop_topk=list(np.argsort(-cnt)[:K])
print(f"ml-1m: {nu}u {ni}i | usable {n} train {len(tr)} test {len(te)} | K={K} d={D} alpha={ALPHA}",flush=True)

# ---- implicit WRMF (ALS) on training users ----
Btr=B[tr]; ntr=len(tr); t0=time.time()
P=0.01*rng.standard_normal((ntr,D)); Q=0.01*rng.standard_normal((ni,D)); Id=LAM*np.eye(D)
# precompute observed/positive index lists
urow=[np.where(Btr[x]>0)[0] for x in range(ntr)]; icol=[np.where(Btr[:,j]>0)[0] for j in range(ni)]
for it in range(ITERS):
    QtQ=Q.T@Q
    for x in range(ntr):
        s=urow[x]; A=QtQ+Id+ALPHA*(Q[s].T@Q[s]); P[x]=np.linalg.solve(A,(1+ALPHA)*Q[s].sum(0)) if len(s) else np.zeros(D)
    PtP=P.T@P
    for j in range(ni):
        s=icol[j]; A=PtP+Id+ALPHA*(P[s].T@P[s]); Q[j]=np.linalg.solve(A,(1+ALPHA)*P[s].sum(0)) if len(s) else np.zeros(D)
    if (it+1)%5==0: print(f"  ALS it{it+1} ({time.time()-t0:.0f}s)",flush=True)

def foldin(seedset, prefs):           # ridge fold-in: user factor from seed item factors + confidences
    Qs=Q[seedset]; c=1.0+ALPHA*prefs  # confidence
    A=(Qs*c[:,None]).T@Qs+Id; bb=(Qs*(c*prefs)[:,None]).sum(0); return np.linalg.solve(A,bb)

POPW=float(os.environ.get('POPW',0.0))   # optional popularity-bias blend at ranking
def ndcg_recall(score, rel, seeds):
    s=score.copy(); s[seeds]=-1e9; top=np.argsort(-s)[:10]; rs=set(rel)
    dcg=sum(1./np.log2(p+2) for p,t in enumerate(top) if t in rs); idcg=sum(1./np.log2(p+2) for p in range(min(10,len(rel))))
    return (dcg/idcg if idcg>0 else 0.), len(set(top.tolist())&rs)/len(rel)
def eval_arm(seeds, name):
    seeds=np.array(seeds); ss=set(seeds.tolist()); nd=rc=0.; m=0
    for ui in te:
        rel=[j for j in np.where(B[ui]>0)[0] if j not in ss]
        if not rel: continue
        prefs=B[ui][seeds]                        # 1 if cold user liked the seed, else 0
        p=foldin(seeds,prefs); score=Q@p+POPW*pop_bias
        a,b=ndcg_recall(score,rel,seeds); nd+=a; rc+=b; m+=1
    print(f"  {name:<14} NDCG@10={nd/m:.4f} Recall@10={rc/m:.4f}",flush=True); return nd/m
# representative seeds: greedy pick items whose factor directions are most diverse (max-volume proxy)
def representative(k):
    chosen=[]; cand=list(np.argsort(-cnt)[:400])    # from reasonably-answerable items
    Qn=Q/ (np.linalg.norm(Q,axis=1,keepdims=True)+1e-9)
    chosen.append(int(cand[0]))
    while len(chosen)<k:
        best=None;bj=None
        for j in cand:
            if j in chosen: continue
            sim=max(abs(float(Qn[j]@Qn[c])) for c in chosen)   # least-similar to chosen
            if best is None or sim<best: best=sim; bj=j
        chosen.append(bj)
    return chosen
def rmva(k):   # rectangular max-volume (Fonarev ICDM2016): QR with column pivoting on Q^T over answerable pool
    import scipy.linalg as sla
    cand=np.array(np.argsort(-cnt)[:400])           # answerable pool (popular-enough to be ratable)
    _,_,piv=sla.qr(Q[cand].T, pivoting=True)        # pivot order = maximal-volume rows of Q[cand]
    return [int(cand[p]) for p in piv[:k]]
print("\n=== cold-start elicitation via MF folding-in (ML-1M, standard NDCG@10) ===",flush=True)
# MOSTPOP
nd=rc=0.; m=0
for ui in te:
    rel=list(np.where(B[ui]>0)[0])
    if not rel: continue
    a,b=ndcg_recall(cnt.copy(),rel,np.array([],int)); nd+=a; rc+=b; m+=1
print(f"  {'MOSTPOP':<14} NDCG@10={nd/m:.4f} Recall@10={rc/m:.4f}  (no elicitation)  [paper 0.3921]",flush=True)
nds=[eval_arm(list(rng.choice(ni,K,replace=False)),'RANDOM') for _ in range(NRAN)]; print(f"  {'RANDOM(avg)':<14} NDCG@10={np.mean(nds):.4f}",flush=True)
eval_arm(pop_topk,'POPULAR')
rep_seeds=representative(K); eval_arm(rep_seeds,'REPRESENTATIVE')
# CEILING diagnostic: fold-in from HALF of each user's REAL positives (rich personalized interview) -> is MF the bottleneck?
rng3=np.random.default_rng(7); nd=rc=0.; m=0
for ui in te:
    pos=np.where(B[ui]>0)[0].copy(); rng3.shuffle(pos)
    half=pos[:len(pos)//2]; rest=[int(j) for j in pos[len(pos)//2:]]
    if len(half)<3 or not rest: continue
    p=foldin(half,np.ones(len(half),np.float32)); score=Q@p
    a,b=ndcg_recall(score,rest,half); nd+=a; rc+=b; m+=1
print(f"  {'WARM-HALF(MF ceiling)':<14} NDCG@10={nd/m:.4f} Recall@10={rc/m:.4f}  (fold-in from half of REAL positives)",flush=True)
# ---- POPULARITY + PERSONALIZATION blend: score = z(Qp) + w*z(pop_bias) ; sweep w (full-cat NDCG) ----
def zsc(v): return (v-v.mean())/(v.std()+1e-9)
zpop=zsc(pop_bias)
print("\n=== popularity+personalization blend  score = z(Q.p) + w*z(pop)  (REPRESENTATIVE seeds, full-cat) ===",flush=True)
rep=np.array(rep_seeds); rss=set(rep.tolist())
for w in [0.0,1.0,2.0,4.0,8.0,16.0]:
    nd=rc=0.; m=0
    for ui in te:
        rel=[j for j in np.where(B[ui]>0)[0] if j not in rss]
        if not rel: continue
        p=foldin(rep,B[ui][rep]); score=zsc(Q@p)+w*zpop
        a,b=ndcg_recall(score,rel,rep); nd+=a; rc+=b; m+=1
    tag=' <- MOSTPOP=0.4134' if w>=8 else ''
    print(f"  w={w:<4} NDCG@10={nd/m:.4f} Recall@10={rc/m:.4f}{tag}",flush=True)

# ---- SAMPLED-candidate protocol (Krichene-Rendle): rank held-out positives vs N sampled negatives ----
NEG=int(os.environ.get('NEG',100)); rng2=np.random.default_rng(123); negs={}
allset={ui:set(np.where(B[ui]>0)[0]) for ui in te}
for ui in te:
    pool=np.setdiff1d(np.arange(ni),np.array(list(allset[ui]),int),assume_unique=False)
    negs[ui]=rng2.choice(pool,min(NEG,len(pool)),replace=False)
def eval_pool(scorer,name,seeds=None):
    sset=set(np.array(seeds).tolist()) if seeds is not None else set(); nd=rc=0.; m=0
    for ui in te:
        pos=[j for j in allset[ui] if j not in sset]
        if not pos: continue
        cand=np.array([c for c in (list(pos)+list(negs[ui])) if c not in sset])
        sc=scorer(ui,seeds); order=cand[np.argsort(-sc[cand])][:10]; rs=set(pos)
        dcg=sum(1./np.log2(p+2) for p,t in enumerate(order) if t in rs); idcg=sum(1./np.log2(p+2) for p in range(min(10,len(pos))))
        nd+=(dcg/idcg if idcg>0 else 0.); rc+=len(set(order.tolist())&rs)/len(pos); m+=1
    print(f"  {name:<14} NDCG@10={nd/m:.4f} Recall@10={rc/m:.4f}",flush=True); return nd/m
# ===================== CLEAN RESULT: blend weight tuned on VAL, reported on held-out TEST =====================
def blend_ndcg(seeds, users, w):
    seeds=np.array(seeds); sset=set(seeds.tolist()); nd=rc=0.; m=0
    for ui in users:
        rel=[j for j in np.where(B[ui]>0)[0] if j not in sset]
        if not rel: continue
        p=foldin(seeds,B[ui][seeds]); score=zsc(Q@p)+w*zpop
        a,b=ndcg_recall(score,rel,seeds); nd+=a; rc+=b; m+=1
    return nd/m, rc/m
def mostpop_ndcg(users):
    nd=rc=0.; m=0
    for ui in users:
        rel=list(np.where(B[ui]>0)[0])
        if not rel: continue
        a,b=ndcg_recall(cnt.copy(),rel,np.array([],int)); nd+=a; rc+=b; m+=1
    return nd/m, rc/m
print("\n===================== CLEAN: blend weight tuned on VAL, reported on TEST (ML-1M, full-cat NDCG@10) =====================",flush=True)
mp_nd,mp_rc=mostpop_ndcg(te); print(f"  {'MOSTPOP':<16} NDCG@10={mp_nd:.4f} Recall@10={mp_rc:.4f}  (popularity baseline)",flush=True)
WGRID=[0,1,2,3,4,6,8,12,16]
for seeds,name in [(list(rng.choice(ni,K,replace=False)),'RANDOM+pop'),(pop_topk,'POPULAR+pop'),(rep_seeds,'REPRESENTATIVE+pop'),(rmva(K),'RMVA-maxvol+pop')]:
    wbest=max(WGRID,key=lambda w: blend_ndcg(seeds,va,w)[0])     # tune on VAL
    tnd,trc=blend_ndcg(seeds,te,wbest)                            # report on TEST
    print(f"  {name:<16} NDCG@10={tnd:.4f} Recall@10={trc:.4f}  (w*={wbest} tuned on val)  {'WIN' if tnd>mp_nd else 'lose'} vs pop {tnd-mp_nd:+.4f}",flush=True)

print(f"\n=== SAMPLED protocol (positives vs {NEG} sampled negatives) — same trained MF ===",flush=True)
eval_pool(lambda ui,s: cnt,'MOSTPOP')
eval_pool(lambda ui,s: Q@foldin(np.array(pop_topk),B[ui][np.array(pop_topk)]),'POPULAR',pop_topk)
eval_pool(lambda ui,s: Q@foldin(np.array(rep_seeds),B[ui][np.array(rep_seeds)]),'REPRESENTATIVE',rep_seeds)
print(f"(total {time.time()-t0:.0f}s)",flush=True)
