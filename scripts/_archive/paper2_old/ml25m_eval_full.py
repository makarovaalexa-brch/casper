"""
ML-25M PHASE-1B instrument-health suite (FULL DATASET). Loads prep.npz + Q_svd/bi_svd, builds selectors
(RMVA/HELF/POP/ENTROPY), runs the ridge fold-in acceptance: q0 MOSTPOP anchor, 0..T reveal monotonicity,
held-out no-harm, full-profile fold reference. Full-catalogue NDCG@10 (+ tail NDCG@10 = relevant restricted
to non-popular items, rank>500). Tunes (beta,lam) on val. Mirrors scripts/paper1/instrument_svd.py.
"""
import os, time, numpy as np, scipy.linalg as sla
out='C:/dev/phd/casper/data/movielens/.cache/ml25m'; t0=time.time()
Q=np.load(f'{out}/Q_svd.npy'); bi=np.load(f'{out}/bi_svd.npy'); D=Q.shape[1]
P=np.load(f'{out}/prep.npz')
uu=P['uu']; ii=P['ii']; R=P['R']; cnt=P['cnt']; mu=float(P['mu']); ni=int(P['ni']); H5=P['H5']
va=P['va'].tolist(); te=P['te'].tolist()
LIKE=4.0; T=int(os.environ.get('T',20)); NEVAL=int(os.environ.get('NEVAL',500))
popb=np.log(cnt+1.0).astype(np.float32)
poprank=np.empty(ni,int); poprank[np.argsort(-cnt)]=np.arange(ni)   # 0=most popular
TAILCUT=500
# eval dicts for va+te only
evalset=np.array(sorted(set(va)|set(te))); sel=np.isin(uu,evalset)
euu=uu[sel]; eii=ii[sel]; eR=R[sel]
likes_by_u={}; rat_by_u={}
for k in range(len(euu)):
    x=int(euu[k]); rat_by_u.setdefault(x,[]).append((int(eii[k]),float(eR[k])))
    if eR[k]>=LIKE: likes_by_u.setdefault(x,[]).append(int(eii[k]))
print(f"eval cohort dicts built ({time.time()-t0:.0f}s)",flush=True)
ca=np.array(np.argsort(-cnt)[:400]); _,_,piv=sla.qr(Q[ca].T,pivoting=True); RMVA=[int(ca[p]) for p in piv]
pm=H5/np.clip(H5.sum(1,keepdims=True),1,None); ent=-(pm*np.log(pm+1e-12)).sum(1)
lf=np.log(cnt+1)/np.log(cnt.max()+1); Hn=ent/np.log(10); helf=2*lf*Hn/(lf+Hn+1e-9)
ORD={'rmva':RMVA,'pop':list(np.argsort(-cnt)),'entropy':list(np.argsort(-ent)),'helf':list(np.argsort(-helf))}
rng=np.random.default_rng(0)
def foldin(seeds,targ,lam):
    Qs=Q[seeds]; A=Qs.T@Qs+lam*np.eye(D); return np.linalg.solve(A,Qs.T@targ)
def ndcg(score,rel,excl):
    if len(rel)==0: return 0.
    s=score.copy(); s[list(excl)]=-1e9; top=np.argsort(-s)[:10]; rs=set(rel)
    dcg=sum(1./np.log2(p+2) for p,t in enumerate(top) if t in rs); idcg=sum(1./np.log2(p+2) for p in range(min(10,len(rel))))
    return dcg/idcg if idcg>0 else 0.
def recall(score,rel,excl):
    if len(rel)==0: return 0.
    s=score.copy(); s[list(excl)]=-1e9; top=set(np.argsort(-s)[:10].tolist()); return len(top&set(rel))/max(len(rel),1)
SPL={}; _rs=np.random.default_rng(123)
for x in (va+te):
    lk=list(likes_by_u.get(x,[]))
    if len(lk)>=4: ll=lk[:]; _rs.shuffle(ll); SPL[x]=set(ll[len(ll)//2:])
def curve(kind,beta,lam,users,T):
    ND=np.zeros(T+1); NDt=np.zeros(T+1); RC=np.zeros(T+1); m=0
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
            relt=[j for j in rel if poprank[j]>=TAILCUT]
            sc=beta*popb+(Q@foldin(np.array(seeds),np.array(targ,np.float32),lam) if seeds else 0.0)
            ND[t]+=ndcg(sc,rel,asked); NDt[t]+=ndcg(sc,relt,asked); RC[t]+=recall(sc,rel,asked)
        m+=1
    return ND/max(m,1), NDt/max(m,1), RC/max(m,1), m
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
            ND[t]+=ndcg(sc,rel,profile|asked)
        m+=1
    return ND/max(m,1), m
def fullprof(beta,lam,users):
    ND=[]; NDt=[]; RCv=[]
    for x in users:
        if x not in SPL: continue
        rd={j:r for j,r in rat_by_u.get(x,[])}; test=SPL[x]; profile=[j for j in rd if j not in test]; rel=list(test)
        if len(rel)<2 or len(profile)<2: continue
        relt=[j for j in rel if poprank[j]>=TAILCUT]
        seeds=np.array(profile); targ=np.array([rd[j]-mu-bi[j] for j in profile],np.float32)
        sc=beta*popb+Q@foldin(seeds,targ,lam)
        ND.append(ndcg(sc,rel,set(profile))); NDt.append(ndcg(sc,relt,set(profile))); RCv.append(recall(sc,rel,set(profile)))
    return float(np.mean(ND)), float(np.mean(NDt)), float(np.mean(RCv)), len(ND)
vt=va[:120]; best=None
for beta in [4,8,16]:
    for lam in [5.0,20.0,50.0]:
        g,mv=curve_ho('rmva',beta,lam,vt,8); gain=g[8]-g[0]
        nh=all(g[t]>=g[0]-3e-3 for t in range(9))
        if (best is None) or (nh and gain>best[0]): best=(gain,beta,lam,nh)
beta,lam=best[1],best[2]; print(f"[VAL] beta={beta} lam={lam} rmva-gain@8={best[0]:+.4f} no-harm={best[3]}",flush=True)
teN=te[:NEVAL]
q0=curve('rmva',beta,lam,te[:50],0)[0][0]
print(f"\n=== ML-25M FULL biased-SVD ridge-fold instrument; beta={beta} lam={lam}; T={T}; ni={ni} ===",flush=True)
print(f"q0 (MOSTPOP anchor) NDCG@10 = {q0:.4f}",flush=True)
print(f"\n{'selector':<9}| NDCG@10 q0..qT(/4) | tailNDCG q0..qT(/4) | RecT | mono?",flush=True)
res={}
for kind in ['rmva','helf','pop','entropy','random']:
    nd,ndt,rc,m=curve(kind,beta,lam,teN,T); mono=all(nd[t]>=nd[0]-3e-3 for t in range(T+1))
    show=" ".join(f"{nd[t]:.3f}" for t in range(0,T+1,4)); showt=" ".join(f"{ndt[t]:.3f}" for t in range(0,T+1,4))
    print(f"{kind:<9}| {show} | {showt} | {rc[T]:.3f} | {'OK' if mono else 'HURTS'} (n={m})",flush=True)
    res[kind]=(nd,ndt)
fp,fpt,fpr,fpn=fullprof(beta,lam,te)
print(f"\nFULL-PROFILE fold reference: NDCG@10={fp:.4f} tailNDCG@10={fpt:.4f} Rec@10={fpr:.4f} (n={fpn})",flush=True)
print(f"\nHEADROOM (full-profile - MOSTPOP q0): full {fp-res['pop'][0][0]:+.4f} | tail {fpt-res['pop'][1][0]:+.4f}",flush=True)
print(f"RMVA elicitation gain @8: full {res['rmva'][0][8]-res['rmva'][0][0]:+.4f} | tail {res['rmva'][1][8]-res['rmva'][1][0]:+.4f}",flush=True)
print(f"\n=== HELD-OUT no-harm (fixed disjoint targets; q0={curve_ho('pop',beta,lam,teN,0)[0][0]:.4f}) ===",flush=True)
for kind in ['rmva','helf','pop','random']:
    nd,m=curve_ho(kind,beta,lam,teN,T); nh=all(nd[t]>=nd[0]-3e-3 for t in range(T+1))
    show=" ".join(f"{nd[t]:.3f}" for t in range(0,T+1,4))
    print(f"{kind:<9}| {show} | {'OK' if nh else 'HURTS'} (n={m})",flush=True)
print(f"\nDONE ({time.time()-t0:.0f}s)",flush=True)
