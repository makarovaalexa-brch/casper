"""
ML-25M PHASE-1 step A: build the biased-SVD item factors (Q, bi) on the ML-25M subsample and run the
ridge-fold-in acceptance suite (monotonicity 0..T reveals, q0 MOSTPOP anchor, full-profile fold reference,
held-out no-harm). Mirrors scripts/paper1/instrument_svd.py exactly, but reads the pre-made 25M subsample
data/processed/ratings_subset.pkl (10k users x 57606 items) instead of ml-1m/ratings.dat.

SUBSAMPLE PROTOCOL:
  users : all 10k in the subsample (each already >=490 ratings; cold-start comes from the reveal protocol,
          matching ML-1M where fold-in reveals a few items at a time). split 80/10/10 seed0 (train/val/test).
  items : catalog = items with >= MINIT total ratings among all users (collaborative-signal filter, cold-start
          realistic: drops the ~41k ultra-rare items with no CF signal). remapped dense.
  like  : rating >= 4.0 (0.5-5 scale).
Saves: .cache/ml25m/Q_svd.npy, bi_svd.npy, meta.npz (uu,ii,rr,cnt,mu, split, iids order).
"""
import os, time, numpy as np, pickle, scipy.linalg as sla
base='C:/dev/phd/casper/data/movielens'; out=f'{base}/.cache/ml25m'; os.makedirs(out,exist_ok=True)
D=int(os.environ.get('D',64)); LAMF=float(os.environ.get('LAMF',0.05)); LR=float(os.environ.get('LR',0.01))
EP=int(os.environ.get('EP',15)); LIKE=4.0; T=int(os.environ.get('T',20)); MINIT=int(os.environ.get('MINIT',20))
NEVAL=int(os.environ.get('NEVAL',400)); rng=np.random.default_rng(0)
t0=time.time()
df=pickle.load(open('C:/dev/phd/casper/data/processed/ratings_subset.pkl','rb'))
U=df.userId.to_numpy(); I=df.movieId.to_numpy(); R=df.rating.to_numpy().astype(np.float32)
print(f"loaded {len(R)} ratings ({time.time()-t0:.0f}s)",flush=True)
# item catalog filter: >= MINIT total ratings
uniqI,icnt=np.unique(I,return_counts=True); keepI=uniqI[icnt>=MINIT]
iids={x:k for k,x in enumerate(np.sort(keepI))}; ni=len(iids)
mask=np.isin(I,keepI); U=U[mask]; I=I[mask]; R=R[mask]
uids={x:k for k,x in enumerate(np.unique(U))}; nu=len(uids)
uu=np.array([uids[x] for x in U],np.int32); ii=np.array([iids[x] for x in I],np.int32)
print(f"catalog: {ni} items (>= {MINIT} ratings), {nu} users, {len(R)} ratings kept ({time.time()-t0:.0f}s)",flush=True)
likes_by_u={}; rat_by_u={}
for k in range(len(uu)):
    rat_by_u.setdefault(uu[k],[]).append((ii[k],R[k]))
    if R[k]>=LIKE: likes_by_u.setdefault(uu[k],[]).append(ii[k])
keep=[x for x in range(nu) if len(likes_by_u.get(x,[]))>=5]; rng.shuffle(keep); n=len(keep)
trU=set(keep[:int(0.8*n)]); va=keep[int(0.8*n):int(0.9*n)]; te=keep[int(0.9*n):]
print(f"split: train {len(trU)} / val {len(va)} / test {len(te)}",flush=True)
cnt=np.zeros(ni)
for x in trU:
    for j in likes_by_u.get(x,[]): cnt[j]+=1
popb=np.log(cnt+1.0).astype(np.float32)
# ---- biased SVD via mini-batch SGD on TRAIN users' explicit ratings ----
mrows=np.array([uu[k] in trU for k in range(len(uu))])
ru=uu[mrows]; ri=ii[mrows]; rr=R[mrows].astype(np.float32); mu=float(rr.mean())
truser={x:k for k,x in enumerate(sorted(trU))}; ruD=np.array([truser[x] for x in ru]); ntr=len(truser)
bu=np.zeros(ntr,np.float32); bi=np.zeros(ni,np.float32)
P=(0.1*rng.standard_normal((ntr,D))).astype(np.float32); Q=(0.1*rng.standard_normal((ni,D))).astype(np.float32)
idx=np.arange(len(rr)); print(f"training biased-SVD: {len(rr)} train ratings, D={D}, EP={EP}",flush=True)
for ep in range(EP):
    rng.shuffle(idx)
    for b0 in range(0,len(idx),16384):
        bidx=idx[b0:b0+16384]; us=ruD[bidx]; it=ri[bidx]
        pred=mu+bu[us]+bi[it]+np.sum(P[us]*Q[it],1); e=(rr[bidx]-pred).astype(np.float32)
        np.add.at(bu,us,LR*(e-LAMF*bu[us])); np.add.at(bi,it,LR*(e-LAMF*bi[it]))
        gP=LR*(e[:,None]*Q[it]-LAMF*P[us]); gQ=LR*(e[:,None]*P[us]-LAMF*Q[it]); np.add.at(P,us,gP); np.add.at(Q,it,gQ)
    if (ep+1)%5==0 or ep==0:
        tr=np.sqrt(np.mean((rr-(mu+bu[ruD]+bi[ri]+np.sum(P[ruD]*Q[ri],1)))**2))
        print(f"  ep{ep+1} train RMSE={tr:.4f} ({time.time()-t0:.0f}s)",flush=True)
np.save(f'{out}/Q_svd.npy',Q); np.save(f'{out}/bi_svd.npy',bi)
np.savez(f'{out}/meta.npz',uu=uu,ii=ii,rr=R.astype(np.float32),cnt=cnt,mu=mu,ni=ni,nu=nu,
         trU=np.array(sorted(trU)),va=np.array(va),te=np.array(te),keepI=np.sort(keepI))
print(f"SAVED Q_svd.npy bi_svd.npy meta.npz ({time.time()-t0:.0f}s)",flush=True)
# ---- selectors + ridge fold-in acceptance ----
ca=np.array(np.argsort(-cnt)[:400]); _,_,piv=sla.qr(Q[ca].T,pivoting=True); RMVA=[int(ca[p]) for p in piv]
H5=np.zeros((ni,10))
for k in range(len(uu)):
    if uu[k] in trU: H5[ii[k],min(max(int(round(R[k]*2)),1),10)-1]+=1
pm=H5/np.clip(H5.sum(1,keepdims=True),1,None); ent=-(pm*np.log(pm+1e-12)).sum(1)
lf=np.log(cnt+1)/np.log(cnt.max()+1); Hn=ent/np.log(10); helf=2*lf*Hn/(lf+Hn+1e-9)
ORD={'rmva':RMVA,'pop':list(np.argsort(-cnt)),'entropy':list(np.argsort(-ent)),'helf':list(np.argsort(-helf))}
def foldin(seeds,targ,lam):
    Qs=Q[seeds]; A=Qs.T@Qs+lam*np.eye(D); return np.linalg.solve(A,Qs.T@targ)
def ndcg_recall(score,rel,excl):
    s=score.copy(); s[list(excl)]=-1e9; top=np.argsort(-s)[:10]; rs=set(rel)
    dcg=sum(1./np.log2(p+2) for p,t in enumerate(top) if t in rs); idcg=sum(1./np.log2(p+2) for p in range(min(10,len(rel))))
    return (dcg/idcg if idcg>0 else 0.), len(set(top.tolist())&rs)/max(len(rel),1)
# held-out split for no-harm + full-profile
SPL={}; _rs=np.random.default_rng(123)
for x in (va+te):
    lk=list(likes_by_u.get(x,[]))
    if len(lk)>=4: ll=lk[:]; _rs.shuffle(ll); SPL[x]=set(ll[len(ll)//2:])
def curve(kind,beta,lam,users,T):
    ND=np.zeros(T+1); RC=np.zeros(T+1); m=0
    for x in users:
        likeset=set(likes_by_u.get(x,[])); rd={j:r for j,r in rat_by_u.get(x,[])}
        if len(likeset)<2: continue
        order=list(rng.permutation(ni)) if kind=='random' else ORD[kind]
        seeds=[]; targ=[]; asked=set()
        for t in range(T+1):
            if t>0:
                e=next((it for it in order if it not in asked),None)
                if e is not None:
                    asked.add(e)
                    if e in rd: seeds.append(e); targ.append(rd[e]-mu-bi[e])
            rel=[j for j in likeset if j not in asked]
            sc=beta*popb+(Q@foldin(np.array(seeds),np.array(targ,np.float32),lam) if seeds else 0.0)
            a,b=ndcg_recall(sc,rel,asked); ND[t]+=a; RC[t]+=b
        m+=1
    return ND/max(m,1), RC/max(m,1), m
def curve_ho(kind,beta,lam,users,T):
    ND=np.zeros(T+1); m=0
    for x in users:
        if x not in SPL: continue
        rd={j:r for j,r in rat_by_u.get(x,[])}; test=SPL[x]; profile=set(rd)-test; rel=list(test)
        if len(rel)<2: continue
        order=list(rng.permutation(ni)) if kind=='random' else ORD[kind]
        seeds=[]; targ=[]; asked=set()
        for t in range(T+1):
            if t>0:
                e=next((it for it in order if it not in asked and it not in test),None)
                if e is not None:
                    asked.add(e)
                    if e in rd: seeds.append(e); targ.append(rd[e]-mu-bi[e])
            sc=beta*popb+(Q@foldin(np.array(seeds),np.array(targ,np.float32),lam) if seeds else 0.0)
            a,_=ndcg_recall(sc,rel,profile|asked); ND[t]+=a
        m+=1
    return ND/max(m,1), m
# full-profile reference: fold ALL profile items (non-test), NDCG on held-out test likes
def fullprof(beta,lam,users):
    ND=[]; RCv=[]
    for x in users:
        if x not in SPL: continue
        rd={j:r for j,r in rat_by_u.get(x,[])}; test=SPL[x]; profile=[j for j in rd if j not in test]; rel=list(test)
        if len(rel)<2 or len(profile)<2: continue
        seeds=np.array(profile); targ=np.array([rd[j]-mu-bi[j] for j in profile],np.float32)
        sc=beta*popb+Q@foldin(seeds,targ,lam)
        a,b=ndcg_recall(sc,rel,set(profile)); ND.append(a); RCv.append(b)
    return float(np.mean(ND)), float(np.mean(RCv)), len(ND)
# light tune (beta,lam) on val
vt=va[:120]; best=None
for beta in [4,8,16]:
    for lam in [5.0,20.0,50.0]:
        g,mv=curve_ho('rmva',beta,lam,vt,8); gain=g[8]-g[0]
        nh=all(g[t]>=g[0]-3e-3 for t in range(9))
        if (best is None) or (nh and gain>best[0]): best=(gain,beta,lam,nh)
beta,lam=best[1],best[2]; print(f"\n[VAL] beta={beta} lam={lam} rmva-gain@8={best[0]:+.4f} no-harm={best[3]}",flush=True)
teN=te[:NEVAL]
q0=curve('rmva',beta,lam,te[:50],0)[0][0]
print(f"\n=== ML-25M biased-SVD ridge-fold instrument; beta={beta} lam={lam}; T={T}; ni={ni} ===",flush=True)
print(f"q0 (MOSTPOP anchor) NDCG@10 = {q0:.4f}",flush=True)
print(f"\n{'selector':<9}| NDCG@10  q0 .. qT (every 4) | RecT | monotone?",flush=True)
for kind in ['rmva','helf','pop','entropy','random']:
    nd,rc,m=curve(kind,beta,lam,teN,T); mono=all(nd[t]>=nd[0]-3e-3 for t in range(T+1))
    show=" ".join(f"{nd[t]:.3f}" for t in range(0,T+1,4))
    print(f"{kind:<9}| {show} | {rc[T]:.3f} | {'OK' if mono else 'HURTS'}  (n={m})",flush=True)
fp,fpr,fpn=fullprof(beta,lam,te)
print(f"\nFULL-PROFILE fold reference (all profile items -> held-out likes): NDCG@10={fp:.4f} Rec@10={fpr:.4f} (n={fpn})",flush=True)
print(f"\n=== HELD-OUT no-harm (fixed disjoint targets; q0={curve_ho('pop',beta,lam,teN,0)[0][0]:.4f}) ===",flush=True)
for kind in ['rmva','helf','pop','random']:
    nd,m=curve_ho(kind,beta,lam,teN,T); nh=all(nd[t]>=nd[0]-3e-3 for t in range(T+1))
    show=" ".join(f"{nd[t]:.3f}" for t in range(0,T+1,4))
    print(f"{kind:<9}| {show} | {'OK' if nh else 'HURTS'} (n={m})",flush=True)
print("\nDONE",flush=True)
