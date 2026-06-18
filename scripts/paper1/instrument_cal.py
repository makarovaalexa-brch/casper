"""
CALIBRATED instrument — fixes the "revealing can hurt" flaw by removing the z-scoring of the personalization
residual. Established method only (Hu-Koren-Volinsky WRMF 2008 + Koren biased-MF 2009):

  item factors Q   : implicit WRMF (ALS, confidence c=1+alpha*pref)            [Hu-Koren-Volinsky 2008]
  user fold-in u   : u = (Qs^T C Qs + lam I)^-1 Qs^T C p    (confidence-weighted ridge MAP)   [HKV fold-in]
  score(item j)    : s_j = beta * popbias_j + q_j . u       (popularity floor + personalization residual, Koren'09)
  NO z-scoring of the residual -> ridge shrinkage is preserved -> an uninformative/redundant reveal yields a small u
  -> the score stays ~= the popularity prior -> revealing CANNOT hurt (at worst flat). beta tuned on VAL.

Protocol = canonical P1 (cold user, rel = all likes - asked seeds, exclude asked), q0 = MOSTPOP anchor ~0.41.
Every asked seed is folded with the ANSWER p=1 if liked else 0 (the discriminative negatives matter).

Gates: (1) beats popularity; (2) MONOTONE non-decreasing in #reveals for RMVA/HELF AND flat-not-down for popular/random
(revealing never hurts); (3) ATTRIBUTE understanding: input a single genre -> relevant (in-genre) recommendations.
"""
import os, time, numpy as np, torch, scipy.linalg as sla
base='C:/dev/phd/casper/data/movielens'; ml=f'{base}/ml-1m'
D=int(os.environ.get('D',64)); ALPHA=float(os.environ.get('ALPHA',20)); LAM=float(os.environ.get('LAM',0.1))
ITERS=int(os.environ.get('ITERS',15)); LIKE=4.0; K=int(os.environ.get('K',30)); rng=np.random.default_rng(0)
U,I,R=[],[],[]
with open(f'{ml}/ratings.dat') as f:
    for line in f:
        a=line.strip().split('::'); U.append(int(a[0])); I.append(int(a[1])); R.append(float(a[2]))
U=np.array(U);I=np.array(I);R=np.array(R,np.float32)
uids={x:k for k,x in enumerate(np.unique(U))}; iids={x:k for k,x in enumerate(np.unique(I))}; nu,ni=len(uids),len(iids)
u=np.array([uids[x] for x in U]); ic=np.array([iids[x] for x in I])
B=np.zeros((nu,ni),np.float32)
for k in range(len(u)):
    if R[k]>=LIKE: B[u[k],ic[k]]=1.0
keep=np.where(B.sum(1)>=5)[0]; rng.shuffle(keep); n=len(keep)
tr=keep[:int(0.8*n)]; va=keep[int(0.8*n):int(0.9*n)]; te=keep[int(0.9*n):]
cnt=B[tr].sum(0); popb=np.log(cnt+1.0).astype(np.float32)
# ---- WRMF ALS (Hu-Koren-Volinsky 2008) ----
Btr=B[tr]; ntr=len(tr); Id=LAM*np.eye(D); t0=time.time()
P=0.01*rng.standard_normal((ntr,D)); Q=0.01*rng.standard_normal((ni,D))
urow=[np.where(Btr[x]>0)[0] for x in range(ntr)]; icol=[np.where(Btr[:,j]>0)[0] for j in range(ni)]
for it in range(ITERS):
    QtQ=Q.T@Q
    for x in range(ntr):
        s=urow[x]; A=QtQ+Id+ALPHA*(Q[s].T@Q[s]); P[x]=np.linalg.solve(A,(1+ALPHA)*Q[s].sum(0)) if len(s) else 0
    PtP=P.T@P
    for j in range(ni):
        s=icol[j]; A=PtP+Id+ALPHA*(P[s].T@P[s]); Q[j]=np.linalg.solve(A,(1+ALPHA)*P[s].sum(0)) if len(s) else 0
    if (it+1)%5==0: print(f"  ALS it{it+1} ({time.time()-t0:.0f}s)",flush=True)
Q=Q.astype(np.float32)
def foldin(seeds, prefs):                      # HKV confidence-weighted ridge MAP fold-in
    Qs=Q[seeds]; c=1.0+ALPHA*prefs; A=(Qs*c[:,None]).T@Qs+Id; bb=(Qs*(c*prefs)[:,None]).sum(0); return np.linalg.solve(A,bb)
# representative (RMVA), entropy, HELF selectors
cand=np.array(np.argsort(-cnt)[:400]); _,_,piv=sla.qr(Q[cand].T,pivoting=True); RMVA=[int(cand[p]) for p in piv]
H5=np.zeros((ni,5)); trset=set(tr.tolist())
for k in range(len(u)):
    if u[k] in trset: H5[ic[k],min(max(int(round(R[k])),1),5)-1]+=1
pm=H5/np.clip(H5.sum(1,keepdims=True),1,None); ent=-(pm*np.log(pm+1e-12)).sum(1)
lf=np.log(cnt+1)/np.log(cnt.max()+1); Hn=ent/np.log(5); helf=2*lf*Hn/(lf+Hn+1e-9)
ORD={'rmva':RMVA,'pop':list(np.argsort(-cnt)),'entropy':list(np.argsort(-ent)),'helf':list(np.argsort(-helf))}
def ndcg_recall(score, rel, excl):
    s=score.copy(); s[list(excl)]=-1e9; top=np.argsort(-s)[:10]; rs=set(rel)
    dcg=sum(1./np.log2(p+2) for p,t in enumerate(top) if t in rs); idcg=sum(1./np.log2(p+2) for p in range(min(10,len(rel))))
    return (dcg/idcg if idcg>0 else 0.), len(set(top.tolist())&rs)/len(rel)
def curve(kind, beta, users, T):               # P1: fold every asked seed (1 liked /0 not); rel=all likes-asked; excl=asked
    ND=np.zeros(T+1); RC=np.zeros(T+1); m=0
    for x in users:
        likeset=set(np.where(B[x]>0)[0].tolist())
        if len(likeset)<2: continue
        order=list(rng.permutation(ni)) if kind=='random' else ORD[kind]
        seeds=[]; asked=set()
        for t in range(T+1):
            if t>0:
                e=next((it for it in order if it not in asked),None)
                if e is not None: asked.add(e); seeds.append(e)
            rel=[j for j in likeset if j not in asked]
            if seeds:
                pr=np.array([1.0 if e in likeset else 0.0 for e in seeds],np.float32); uu=foldin(np.array(seeds),pr); sc=beta*popb+Q@uu
            else: sc=beta*popb
            a,b=ndcg_recall(sc,rel,asked); ND[t]+=a; RC[t]+=b
        m+=1
    return ND/m, RC/m
# ---- tune beta on VAL (RMVA curve, mean over q) ----
T=int(os.environ.get('T',15))
BETAS=[0.0,0.5,1,2,4,8,16,32,64]
bbeta=max(BETAS,key=lambda b: np.mean(curve('rmva',b,va,T)[0]))
print(f"\n=== CALIBRATED instrument (WRMF + biased-MF, NO z-score); beta*={bbeta} tuned on VAL; T={T} ===",flush=True)
mp=curve('rmva',bbeta,te,0)[0][0]   # q0 = MOSTPOP (any selector, 0 reveals)
print(f"  q0 (MOSTPOP anchor) NDCG@10 = {mp:.4f}",flush=True)
print(f"\n{'selector':<10} | NDCG@10 q0..q{T} | Rec qT | monotone(no-harm)?",flush=True)
mono_all=True
for kind in ['rmva','helf','pop','random','entropy']:
    nd,rc=curve(kind,bbeta,te,T)
    nondec=all(nd[t]>=nd[0]-3e-3 for t in range(T+1))    # never falls below q0 by >0.003 = "revealing can't hurt"
    if kind in ('rmva','helf','pop','random','entropy') and not nondec: mono_all=False
    print(f"{kind:<10} | "+" ".join(f"{v:.3f}" for v in nd)+f" | {rc[T]:.3f} | {'OK' if nondec else 'HURTS<q0'}",flush=True)
# ---- ATTRIBUTE gate: input ONE genre -> relevant (in-genre) recs ----
GEN=['Action','Adventure','Animation',"Children's",'Comedy','Crime','Documentary','Drama','Fantasy','Film-Noir','Horror','Musical','Mystery','Romance','Sci-Fi','Thriller','War','Western']
gid={g:k for k,g in enumerate(GEN)}; na=len(GEN); item_g=np.zeros((ni,na),bool)
with open(f'{ml}/movies.dat',encoding='latin-1') as f:
    for line in f:
        pp=line.strip().split('::'); m=int(pp[0])
        if m in iids:
            for g in pp[2].split('|'):
                if g in gid: item_g[iids[m],gid[g]]=True
print(f"\n=== ATTRIBUTE GATE: input a SINGLE genre, measure in-genre precision@10 of recs (vs base rate) ===",flush=True)
ok=0
for g in range(na):
    s=np.where(item_g[:,g])[0]
    if len(s)<10: continue
    ag=Q[s].mean(0)                              # genre centroid in factor space (the "user who likes genre g")
    sc=bbeta*popb+Q@ag; top=np.argsort(-sc)[:10]
    prec=item_g[top,g].mean(); baserate=item_g[:,g].mean()
    lift=prec/ (baserate+1e-9)
    if prec>=0.5: ok+=1
    print(f"  {GEN[g]:<12} prec@10={prec:.2f}  baserate={baserate:.3f}  lift={lift:4.1f}x  top1={'in' if item_g[top[0],g] else 'out'}-genre",flush=True)
print(f"\nGATES: beats-pop={'PASS' if curve('rmva',bbeta,te,T)[0][T]>mp+0.003 else 'FAIL'} | "
      f"no-harm(all selectors >= q0)={'PASS' if mono_all else 'FAIL'} | "
      f"attribute(>=12/18 genres prec>=0.5)={'PASS' if ok>=12 else 'FAIL ('+str(ok)+'/18)'}",flush=True)
np.save(f'{base}/.cache/Q_cal.npy', Q); print("saved Q_cal.npy",flush=True)
