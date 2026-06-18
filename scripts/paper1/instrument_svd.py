"""
CALIBRATED instrument v2 — BIASED matrix factorization (Koren 2009, the canonical method), so factors carry only
TASTE (popularity absorbed by the item bias). This gives the no-harm property for free:
  model        : r_ui ~= mu + b_i + q_i . p_u            (explicit ratings 1-5; biased SVD, SGD)
  taste resid  : a popular item rated as-expected has (r - mu - b_i) ~= 0  -> contributes ~nothing to the fold-in
  fold-in      : u = (Qs^T Qs + lam I)^-1 Qs^T (r_s - mu - b_s)     (ridge MAP over the asked+RATED items)
  score        : s_j = beta * popbias_j + q_j . u        (popularity FLOOR keeps the MOSTPOP anchor; taste residual)
  NO z-scoring. Redundant reveals -> taste residual ~0 -> u~0 -> score ~= prior  => revealing cannot hurt.
Protocol = canonical P1 (rel = all likes - asked, exclude asked), q0 = MOSTPOP ~0.41. beta, lam tuned on VAL.
Gates: beats-pop; no-harm (all selectors >= q0); attribute (single genre -> in-genre recs).
"""
import os, time, numpy as np, scipy.linalg as sla
base='C:/dev/phd/casper/data/movielens'; ml=f'{base}/ml-1m'
D=int(os.environ.get('D',64)); LAMF=float(os.environ.get('LAMF',0.05)); LR=float(os.environ.get('LR',0.01))
EP=int(os.environ.get('EP',25)); LIKE=4.0; T=int(os.environ.get('T',15)); rng=np.random.default_rng(0)
U,I,R=[],[],[]
with open(f'{ml}/ratings.dat') as f:
    for line in f:
        a=line.strip().split('::'); U.append(int(a[0])); I.append(int(a[1])); R.append(float(a[2]))
U=np.array(U);I=np.array(I);R=np.array(R,np.float32)
uids={x:k for k,x in enumerate(np.unique(U))}; iids={x:k for k,x in enumerate(np.unique(I))}; nu,ni=len(uids),len(iids)
uu=np.array([uids[x] for x in U]); ii=np.array([iids[x] for x in I])
likes_by_u={}
for k in range(len(uu)):
    if R[k]>=LIKE: likes_by_u.setdefault(uu[k],[]).append(ii[k])
rat_by_u={}
for k in range(len(uu)): rat_by_u.setdefault(uu[k],[]).append((ii[k],R[k]))
keep=[x for x in range(nu) if len(likes_by_u.get(x,[]))>=5]; rng.shuffle(keep); n=len(keep)
trU=set(keep[:int(0.8*n)]); va=keep[int(0.8*n):int(0.9*n)]; te=keep[int(0.9*n):]
cnt=np.zeros(ni)
for x in trU:
    for j in likes_by_u.get(x,[]): cnt[j]+=1
popb=np.log(cnt+1.0).astype(np.float32)
# ---- biased SVD via mini-batch SGD on TRAIN users' explicit ratings ----
mask=np.array([uu[k] in trU for k in range(len(uu))])
ru=uu[mask]; ri=ii[mask]; rr=R[mask].astype(np.float32); mu=float(rr.mean())
# remap train users to dense ids for bu
truser={x:k for k,x in enumerate(sorted(trU))}; ruD=np.array([truser[x] for x in ru]); ntr=len(truser)
bu=np.zeros(ntr,np.float32); bi=np.zeros(ni,np.float32)
P=(0.1*rng.standard_normal((ntr,D))).astype(np.float32); Q=(0.1*rng.standard_normal((ni,D))).astype(np.float32)
idx=np.arange(len(rr)); t0=time.time()
for ep in range(EP):
    rng.shuffle(idx)
    for b0 in range(0,len(idx),16384):
        bidx=idx[b0:b0+16384]; us=ruD[bidx]; it=ri[bidx]
        pred=mu+bu[us]+bi[it]+np.sum(P[us]*Q[it],1); e=(rr[bidx]-pred).astype(np.float32)
        np.add.at(bu, us, LR*(e-LAMF*bu[us]))
        np.add.at(bi, it, LR*(e-LAMF*bi[it]))
        gP=LR*(e[:,None]*Q[it]-LAMF*P[us]); gQ=LR*(e[:,None]*P[us]-LAMF*Q[it])
        np.add.at(P, us, gP); np.add.at(Q, it, gQ)
    if (ep+1)%5==0:
        tr_rmse=np.sqrt(np.mean((rr-(mu+bu[ruD]+bi[ri]+np.sum(P[ruD]*Q[ri],1)))**2))
        print(f"  ep{ep+1} train RMSE={tr_rmse:.4f} ({time.time()-t0:.0f}s)",flush=True)
# selectors
ca=np.array(np.argsort(-cnt)[:400]); _,_,piv=sla.qr(Q[ca].T,pivoting=True); RMVA=[int(ca[p]) for p in piv]
H5=np.zeros((ni,5))
for k in range(len(uu)):
    if uu[k] in trU: H5[ii[k],min(max(int(round(R[k])),1),5)-1]+=1
pm=H5/np.clip(H5.sum(1,keepdims=True),1,None); ent=-(pm*np.log(pm+1e-12)).sum(1)
lf=np.log(cnt+1)/np.log(cnt.max()+1); Hn=ent/np.log(5); helf=2*lf*Hn/(lf+Hn+1e-9)
ORD={'rmva':RMVA,'pop':list(np.argsort(-cnt)),'entropy':list(np.argsort(-ent)),'helf':list(np.argsort(-helf))}
def foldin(seeds, targ, lam):
    Qs=Q[seeds]; A=Qs.T@Qs+lam*np.eye(D); return np.linalg.solve(A, Qs.T@targ)
def ndcg_recall(score, rel, excl):
    s=score.copy(); s[list(excl)]=-1e9; top=np.argsort(-s)[:10]; rs=set(rel)
    dcg=sum(1./np.log2(p+2) for p,t in enumerate(top) if t in rs); idcg=sum(1./np.log2(p+2) for p in range(min(10,len(rel))))
    return (dcg/idcg if idcg>0 else 0.), len(set(top.tolist())&rs)/len(rel)
def curve(kind, beta, lam, users, T):
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
                    if e in rd: seeds.append(e); targ.append(rd[e]-mu-bi[e])   # taste residual of the answer
            rel=[j for j in likeset if j not in asked]
            sc=beta*popb + (Q@foldin(np.array(seeds),np.array(targ,np.float32),lam) if seeds else 0.0)
            a,b=ndcg_recall(sc,rel,asked); ND[t]+=a; RC[t]+=b
        m+=1
    return ND/m, RC/m
SPL={}; _rs=np.random.default_rng(123)
for x in (va+te):
    lk=list(likes_by_u.get(x,[]))
    if len(lk)>=4: ll=lk[:]; _rs.shuffle(ll); SPL[x]=set(ll[len(ll)//2:])
def curve_ho(kind, beta, lam, users, T):     # HELD-OUT: fixed disjoint targets -> proper 'revealing can't hurt' test
    ND=np.zeros(T+1); RC=np.zeros(T+1); m=0
    for x in users:
        if x not in SPL: continue
        rd={j:r for j,r in rat_by_u.get(x,[])}; test=SPL[x]
        profile=set(rd)-test; plike=set(likes_by_u.get(x,[]))-test; rel=list(test)
        if len(rel)<2: continue
        order=list(rng.permutation(ni)) if kind=='random' else ORD[kind]
        seeds=[]; targ=[]; asked=set()
        for t in range(T+1):
            if t>0:
                e=next((it for it in order if it not in asked),None)
                if e is not None:
                    asked.add(e)
                    if e in rd: seeds.append(e); targ.append(rd[e]-mu-bi[e])
            sc=beta*popb + (Q@foldin(np.array(seeds),np.array(targ,np.float32),lam) if seeds else 0.0)
            a,b=ndcg_recall(sc,rel,profile|asked); ND[t]+=a; RC[t]+=b
        m+=1
    return ND/m, RC/m
# tune (beta, lam) on VAL: among configs that are NO-HARM for ALL selectors (held-out), maximize RMVA gain.
vt=va[:120]   # cap tuning users for speed
def noharm_all(beta,lam):
    for k in ['pop','random','helf','rmva']:
        c=curve_ho(k,beta,lam,vt,T)
        if any(c[0][t]<c[0][0]-3e-3 for t in range(T+1)): return False
    return True
best=None; bestany=None
for beta in [4,8,16,32]:
    for lam in [5.0,20.0,50.0]:
        g=curve_ho('rmva',beta,lam,vt,T)[0]; gain=g[T]-g[0]
        if bestany is None or gain>bestany[0]: bestany=(gain,beta,lam)
        if noharm_all(beta,lam) and (best is None or gain>best[0]): best=(gain,beta,lam)
if best is not None: beta,lam=best[1],best[2]; print(f"  [VAL] no-harm-constrained: beta={beta} lam={lam} rmva-gain={best[0]:+.4f}",flush=True)
else: beta,lam=bestany[1],bestany[2]; print(f"  [VAL] NO config is fully no-harm; best-gain beta={beta} lam={lam}",flush=True)
print(f"\n=== BIASED-SVD instrument (taste factors + pop floor, NO z-score); beta*={beta} lam*={lam}; T={T} ===",flush=True)
q0=curve('rmva',beta,lam,te,0)[0][0]; print(f"  q0 (MOSTPOP anchor) NDCG@10 = {q0:.4f}",flush=True)
print(f"\n{'selector':<10} | NDCG@10 q0..q{T} | Rec qT | no-harm?",flush=True)
noharm=True
for kind in ['rmva','helf','pop','random','entropy']:
    nd,rc=curve(kind,beta,lam,te,T); nh=all(nd[t]>=nd[0]-3e-3 for t in range(T+1))
    if not nh: noharm=False
    print(f"{kind:<10} | "+" ".join(f"{v:.3f}" for v in nd)+f" | {rc[T]:.3f} | {'OK' if nh else 'HURTS'}",flush=True)
# attribute gate
GEN=['Action','Adventure','Animation',"Children's",'Comedy','Crime','Documentary','Drama','Fantasy','Film-Noir','Horror','Musical','Mystery','Romance','Sci-Fi','Thriller','War','Western']
gid={g:k for k,g in enumerate(GEN)}; na=len(GEN); item_g=np.zeros((ni,na),bool)
with open(f'{ml}/movies.dat',encoding='latin-1') as f:
    for line in f:
        pp=line.strip().split('::'); m=int(pp[0])
        if m in iids:
            for g in pp[2].split('|'):
                if g in gid: item_g[iids[m],gid[g]]=True
print(f"\n=== NO-HARM, proper test: HELD-OUT fixed targets (q0={curve_ho('pop',beta,lam,te,0)[0][0]:.4f}); revealing must not drop below q0 ===",flush=True)
noharm_ho=True
for kind in ['rmva','helf','pop','random','entropy']:
    nd,rc=curve_ho(kind,beta,lam,te,T); nh=all(nd[t]>=nd[0]-3e-3 for t in range(T+1))
    if not nh: noharm_ho=False
    print(f"{kind:<10} | "+" ".join(f"{v:.3f}" for v in nd)+f" | {rc[T]:.3f} | {'OK' if nh else 'HURTS'}",flush=True)
print(f"\n=== ATTRIBUTE GATE: single genre -> in-genre precision@10 ===",flush=True)
okf=okt=0
for g in range(na):
    s=np.where(item_g[:,g])[0]
    if len(s)<10: continue
    ag=Q[s].mean(0); top_f=np.argsort(-(beta*popb+Q@ag))[:10]; top_t=np.argsort(-(Q@ag))[:10]
    pf=item_g[top_f,g].mean(); pt=item_g[top_t,g].mean()
    if pf>=0.5: okf+=1
    if pt>=0.5: okt+=1
    print(f"  {GEN[g]:<12} pure-taste prec@10={pt:.2f}  with-floor={pf:.2f}  base={item_g[:,g].mean():.3f}",flush=True)
rmva_end=curve('rmva',beta,lam,te,T)[0][T]
print(f"\nGATES: beats-pop={'PASS' if rmva_end>q0+0.003 else 'FAIL'} | no-harm P1(all>=q0)={'PASS' if noharm else 'FAIL'} | "
      f"no-harm HELD-OUT(all>=q0)={'PASS' if noharm_ho else 'FAIL'} | attribute pure-taste(>=12/18)={'PASS' if okt>=12 else 'FAIL ('+str(okt)+'/18)'} | with-floor {okf}/18",flush=True)
np.save(f'{base}/.cache/Q_svd.npy',Q); np.save(f'{base}/.cache/bi_svd.npy',bi); print("saved.",flush=True)
