"""
ML-25M PHASE-1B step A (FULL DATASET, corrects phase-1's 10k-heaviest-raters subset).
Reads the official full ratings.csv (25M ratings, all 162k users) and mirrors scripts/paper1/instrument_svd.py.

FULL PROTOCOL (no activity sampling):
  users : ALL users. Split by user: fixed rng(0) shuffle, hold out the LAST 1000 -> 500 val / 500 test
          (mirrors ML-1M te[:300]/te[300:]); everyone else trains. Cold-start = the reveal protocol.
  items : catalog = items with >= MINIT(20) total ratings (stated preprocessing filter, NOT sampling). dense remap.
  like  : rating >= 4.0 (0.5-5 scale).
  factors: biased SVD (Koren09) mini-batch SGD, D=64 LAMF=0.05 LR=0.01 EP=15 (identical recipe to ML-1M).
Overwrites .cache/ml25m/{Q_svd.npy,bi_svd.npy,meta.npz}.
"""
import os, time, numpy as np, pandas as pd, scipy.linalg as sla
base='C:/dev/phd/casper/data/movielens'; out=f'{base}/.cache/ml25m'; os.makedirs(out,exist_ok=True)
RCSV='C:/dev/phd/casper/data/movielens/ratings.csv'
D=int(os.environ.get('D',64)); LAMF=float(os.environ.get('LAMF',0.05)); LR=float(os.environ.get('LR',0.01))
EP=int(os.environ.get('EP',15)); LIKE=4.0; T=int(os.environ.get('T',20)); MINIT=int(os.environ.get('MINIT',20))
NEVAL=int(os.environ.get('NEVAL',500)); rng=np.random.default_rng(0)
t0=time.time()
df=pd.read_csv(RCSV,usecols=['userId','movieId','rating'],
               dtype={'userId':np.int32,'movieId':np.int32,'rating':np.float32})
U=df.userId.to_numpy(); I=df.movieId.to_numpy(); R=df.rating.to_numpy(); del df
print(f"loaded {len(R)} ratings ({time.time()-t0:.0f}s)",flush=True)
# item catalog filter: >= MINIT total ratings
uniqI,icnt=np.unique(I,return_counts=True); keepI=uniqI[icnt>=MINIT]
iids={int(x):k for k,x in enumerate(np.sort(keepI))}; ni=len(iids)
mask=np.isin(I,keepI); U=U[mask]; I=I[mask]; R=R[mask]
uniqU=np.unique(U); uids={int(x):k for k,x in enumerate(uniqU)}; nu=len(uids)
# vectorized dense remap
umap=np.zeros(int(uniqU.max())+1,np.int32); umap[uniqU]=np.arange(nu)
imap=np.zeros(int(keepI.max())+1,np.int32); imap[np.sort(keepI)]=np.arange(ni)
uu=umap[U]; ii=imap[I]; R=R.astype(np.float32)
print(f"catalog: {ni} items (>= {MINIT} ratings), {nu} users, {len(R)} ratings kept ({time.time()-t0:.0f}s)",flush=True)
# rating-count distribution (per user, after item filter)
ucnt=np.bincount(uu,minlength=nu)
print(f"ratings/user: median={np.median(ucnt):.0f} mean={ucnt.mean():.1f} min={ucnt.min()} max={ucnt.max()} "
      f"p25={np.percentile(ucnt,25):.0f} p75={np.percentile(ucnt,75):.0f}",flush=True)
# --- split users: rng(0) shuffle, last 1000 held out -> 500 val / 500 test, rest train ---
perm=rng.permutation(nu); hold=perm[-1000:]; va=hold[:500].tolist(); te=hold[500:].tolist()
trU=set(perm[:-1000].tolist())
trU_mask=np.zeros(nu,bool); trU_mask[list(trU)]=True
print(f"split: train {len(trU)} / val {len(va)} / test {len(te)}",flush=True)
tr_row=trU_mask[uu]                       # bool per rating: is a train-user row
like_row=(R>=LIKE)
# popularity over TRAIN-user likes
cnt=np.bincount(ii[tr_row & like_row],minlength=ni).astype(np.float64)
popb=np.log(cnt+1.0).astype(np.float32)
# ---- biased SVD via mini-batch SGD on TRAIN users' explicit ratings ----
ru=uu[tr_row]; ri=ii[tr_row]; rr=R[tr_row]; mu=float(rr.mean())
truniq=np.unique(ru); trmap=np.zeros(nu,np.int32); trmap[truniq]=np.arange(len(truniq))
ruD=trmap[ru]; ntr=len(truniq)
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
np.savez(f'{out}/meta.npz',uu=uu,ii=ii,rr=R,cnt=cnt,mu=mu,ni=ni,nu=nu,
         trU=np.array(sorted(trU)),va=np.array(va),te=np.array(te),keepI=np.sort(keepI))
print(f"SAVED Q_svd.npy bi_svd.npy meta.npz ({time.time()-t0:.0f}s)",flush=True)
# ---- eval dicts: ONLY val+test cohort (memory-lean) ----
likes_by_u={}; rat_by_u={}
evalset=set(va)|set(te)
sel=np.isin(uu,np.array(sorted(evalset)))
euu=uu[sel]; eii=ii[sel]; eR=R[sel]
for k in range(len(euu)):
    x=int(euu[k]); rat_by_u.setdefault(x,[]).append((int(eii[k]),float(eR[k])))
    if eR[k]>=LIKE: likes_by_u.setdefault(x,[]).append(int(eii[k]))
# ---- selectors ----
ca=np.array(np.argsort(-cnt)[:400]); _,_,piv=sla.qr(Q[ca].T,pivoting=True); RMVA=[int(ca[p]) for p in piv]
H5=np.zeros((ni,10))
tri=ii[tr_row]; trr=R[tr_row]
b5=np.clip(np.round(trr*2).astype(int),1,10)-1
np.add.at(H5,(tri,b5),1.0)
pm=H5/np.clip(H5.sum(1,keepdims=True),1,None); ent=-(pm*np.log(pm+1e-12)).sum(1)
lf=np.log(cnt+1)/np.log(cnt.max()+1); Hn=ent/np.log(10); helf=2*lf*Hn/(lf+Hn+1e-9)
ORD={'rmva':RMVA,'pop':list(np.argsort(-cnt)),'entropy':list(np.argsort(-ent)),'helf':list(np.argsort(-helf))}
def foldin(seeds,targ,lam):
    Qs=Q[seeds]; A=Qs.T@Qs+lam*np.eye(D); return np.linalg.solve(A,Qs.T@targ)
def ndcg_recall(score,rel,excl):
    s=score.copy(); s[list(excl)]=-1e9; top=np.argsort(-s)[:10]; rs=set(rel)
    dcg=sum(1./np.log2(p+2) for p,t in enumerate(top) if t in rs); idcg=sum(1./np.log2(p+2) for p in range(min(10,len(rel))))
    return (dcg/idcg if idcg>0 else 0.), len(set(top.tolist())&rs)/max(len(rel),1)
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
vt=va[:120]; best=None
for beta in [4,8,16]:
    for lam in [5.0,20.0,50.0]:
        g,mv=curve_ho('rmva',beta,lam,vt,8); gain=g[8]-g[0]
        nh=all(g[t]>=g[0]-3e-3 for t in range(9))
        if (best is None) or (nh and gain>best[0]): best=(gain,beta,lam,nh)
beta,lam=best[1],best[2]; print(f"\n[VAL] beta={beta} lam={lam} rmva-gain@8={best[0]:+.4f} no-harm={best[3]}",flush=True)
teN=te[:NEVAL]
q0=curve('rmva',beta,lam,te[:50],0)[0][0]
print(f"\n=== ML-25M FULL biased-SVD ridge-fold instrument; beta={beta} lam={lam}; T={T}; ni={ni} ===",flush=True)
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
