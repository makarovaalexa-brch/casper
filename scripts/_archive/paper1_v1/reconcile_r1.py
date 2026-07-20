"""
Q1 RECONCILIATION: why does R1 get +0.09 NDCG (RMVA 0.50 vs MOSTPOP 0.41) but the calibrated instrument only +0.025?
Controlled 2x2xK on the SAME P1 protocol (cold user, rel = all likes - seeds, exclude seeds, q0 = MOSTPOP ~0.41),
RMVA representative seeds folded all-at-once (binary p = 1 liked / 0 else), plain ridge fold-in for BOTH factor sets
so only the named variable changes. Reuses saved factors (no retrain): Q_cal.npy (WRMF, R1-style) + Q_svd.npy
(biased-SVD taste). Tunes the blend on VAL. Isolates: FACTORS (WRMF vs SVD) x SCORING (z-score vs none) x BUDGET (K).
"""
import os, numpy as np, scipy.linalg as sla
base='C:/dev/phd/casper/data/movielens'; ml=f'{base}/ml-1m'; D=64; LAM=5.0; rng=np.random.default_rng(0)
U,I,R=[],[],[]
with open(f'{ml}/ratings.dat') as f:
    for line in f:
        a=line.strip().split('::'); U.append(int(a[0])); I.append(int(a[1])); R.append(float(a[2]))
U=np.array(U);I=np.array(I);R=np.array(R,np.float32)
uids={x:k for k,x in enumerate(np.unique(U))}; iids={x:k for k,x in enumerate(np.unique(I))}; nu,ni=len(uids),len(iids)
uu=np.array([uids[x] for x in U]); ii=np.array([iids[x] for x in I])
B=np.zeros((nu,ni),np.float32)
for k in range(len(uu)):
    if R[k]>=4: B[uu[k],ii[k]]=1.0
keep=np.where(B.sum(1)>=5)[0]; rng.shuffle(keep); n=len(keep)
tr=keep[:int(0.8*n)]; va=keep[int(0.8*n):int(0.9*n)]; te=keep[int(0.9*n):]
cnt=B[tr].sum(0); popb=np.log(cnt+1.0).astype(np.float32)
def zsc(v): return (v-v.mean())/(v.std()+1e-9)
zpop=zsc(popb)
QW=np.load(f'{base}/.cache/Q_cal.npy'); QS=np.load(f'{base}/.cache/Q_svd.npy')   # WRMF, biased-SVD
def rmva(Q):
    ca=np.array(np.argsort(-cnt)[:400]); _,_,piv=sla.qr(Q[ca].T,pivoting=True); return [int(ca[p]) for p in piv]
RM={'wrmf':rmva(QW),'svd':rmva(QS)}; QQ={'wrmf':QW,'svd':QS}
def ndcg(score,rel,seeds):
    s=score.copy(); s[list(seeds)]=-1e9; top=np.argsort(-s)[:10]; rs=set(rel)
    dcg=sum(1./np.log2(p+2) for p,t in enumerate(top) if t in rs); idcg=sum(1./np.log2(p+2) for p in range(min(10,len(rel))))
    return dcg/idcg if idcg>0 else 0.
def evalK(fac, scoring, K, users, param):
    Q=QQ[fac]; seeds=np.array(RM[fac][:K]); Qs=Q[seeds]; A=Qs.T@Qs+LAM*np.eye(D); Ainv=np.linalg.solve(A,Qs.T)
    nd=0.;m=0
    for x in users:
        rel=[j for j in np.where(B[x]>0)[0] if j not in set(seeds.tolist())]
        if not rel: continue
        p=B[x][seeds]; u=Ainv@p
        if scoring=='z': sc=zsc(Q@u)+param*zpop
        else: sc=param*popb+Q@u
        nd+=ndcg(sc,rel,seeds); m+=1
    return nd/m
def mostpop(users):
    nd=0.;m=0
    for x in users:
        rel=list(np.where(B[x]>0)[0])
        if not rel: continue
        nd+=ndcg(popb.copy(),rel,set()); m+=1
    return nd/m
MP=mostpop(te); print(f"MOSTPOP (te) NDCG@10 = {MP:.4f}  [R1 anchor 0.41]\n",flush=True)
GRID={'z':[0,1,2,4,8,16],'noz':[0.5,1,2,4,8,16,32]}
print(f"{'config':<16} | "+" ".join(f"K={k}" for k in [5,15,30,50])+" | best-K delta vs MOSTPOP",flush=True)
for fac in ['wrmf','svd']:
    for scoring in ['z','noz']:
        row=[]
        for K in [5,15,30,50]:
            best=max(GRID[scoring], key=lambda pp: evalK(fac,scoring,K,va,pp))   # tune blend on VAL
            row.append((K,evalK(fac,scoring,K,te,best),best))
        cells=" ".join(f"{v:.3f}" for _,v,_ in row); bestcell=max(row,key=lambda r:r[1])
        print(f"{fac+'+'+scoring:<16} | {cells} | best {bestcell[1]:.3f}@K{bestcell[0]} ({bestcell[1]-MP:+.3f})  [param {bestcell[2]}]",flush=True)
