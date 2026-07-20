"""
Is the item-panel's failure-to-beat-pop just an UNTUNED blend weight? Ruler requires tuning the pop/personalization
blend on VAL. Lean (ratings+Q only), CANONICAL P1 (rel=all likes-seeds, exclude seeds, q0=MOSTPOP). Sweep W (pop
weight) for pop/rmva/eig/random; report q8 NDCG/Recall. Also tries a CONFIDENCE-CAPPED scorer (no per-step z-blowup):
score = W*z(pop) + min(conf, CAP)*z(Q.u). If a tuned W (or cap) makes RMVA/EIG beat 0.411, the panel scorer was the
problem, not elicitation.
"""
import os, numpy as np, torch, scipy.linalg as sla
base='C:/dev/phd/casper/data/movielens'; ml=f'{base}/ml-1m'; D=64; LAM=1.0; T=8; NU=int(os.environ.get('NU',200))
rng=np.random.default_rng(0)
U,I,R=[],[],[]
with open(f'{ml}/ratings.dat') as f:
    for line in f:
        a=line.strip().split('::'); U.append(int(a[0])); I.append(int(a[1])); R.append(float(a[2]))
U=np.array(U);I=np.array(I);R=np.array(R,np.float32)
uids={x:k for k,x in enumerate(np.unique(U))}; iids={x:k for k,x in enumerate(np.unique(I))}; nu,ni=len(uids),len(iids)
u=np.array([uids[x] for x in U]); ic=np.array([iids[x] for x in I])
rated={}
for k in range(len(u)): rated.setdefault(u[k],[]).append((ic[k],R[k]))
keep=[x for x in range(nu) if sum(1 for it,r in rated[x] if r>=4)>=5]; rng.shuffle(keep)
trU=keep[:int(0.8*len(keep))]; va=keep[int(0.8*len(keep)):int(0.9*len(keep))]; te=keep[int(0.9*len(keep)):]
cnt=np.zeros(ni); hist=np.zeros((ni,5))
for x in trU:
    for it,r in rated[x]:
        if r>=4: cnt[it]+=1
        hist[it,min(max(int(round(r)),1),5)-1]+=1
zpop=(np.log(cnt+1.)-np.log(cnt+1.).mean())/(np.log(cnt+1.).std()+1e-9)
pm=hist/np.clip(hist.sum(1,keepdims=True),1,None); ent=-(pm*np.log(pm+1e-12)).sum(1)
lf=np.log(cnt+1)/np.log(cnt.max()+1); Hn=ent/np.log(5); helf=2*lf*Hn/(lf+Hn+1e-9)
Q=torch.load(f'{base}/.cache/checkpoints/instrument_u_ml1m.pt')['Q'].numpy().astype(np.float32)
_cand=np.array(np.argsort(-cnt)[:400]); _,_,_piv=sla.qr(Q[_cand].T,pivoting=True); RMVA=[int(_cand[p]) for p in _piv]
pool=list(np.argsort(-cnt)[:120])
def foldin(X,y): return np.linalg.solve(X.T@X+LAM*np.eye(D), X.T@y) if len(X) else np.zeros(D)
def zc(v): return (v-v.mean())/(v.std()+1e-9)
def ndrc(scv,rel,excl):
    if not len(rel): return None
    s=scv.copy(); s[list(excl)]=-1e9; top=np.argsort(-s)[:10]; rs=set(int(t) for t in rel)
    idcg=sum(1./np.log2(p+2) for p in range(min(10,len(rel))))
    return (sum(1./np.log2(p+2) for p,t in enumerate(top) if int(t) in rs)/idcg if idcg else 0.,
            len(set(int(t) for t in top)&rs)/len(rel))
def run(kind,users,W,CAP):
    ND=np.zeros(T+1);RC=np.zeros(T+1);m=0
    for x in users[:NU]:
        recs=rated[x]; rd={it:(r-3)/2 for it,r in recs}; likes=[it for it,r in recs if r>=4]
        if len(likes)<2: continue
        rows=[];y=[];asked=set()
        order={'random':None,'pop':list(np.argsort(-cnt)),'entropy':list(np.argsort(-ent)),'helf':list(np.argsort(-helf)),'rmva':RMVA}.get(kind)
        if kind=='random': order=list(rng.permutation(ni))
        for t in range(T+1):
            if t>0:
                if kind=='eig':
                    uu=foldin(np.array(rows),np.array(y)) if rows else np.zeros(D); base=W*zpop+(min(len(rows)/(len(rows)+5.),CAP))*zc(Q@uu) if rows else W*zpop
                    bt=set(np.argsort(-base)[:10]); z=Q@uu if rows else np.zeros(ni); pl=1/(1+np.exp(-2*(z-np.median(z)))); best=None;e=pool[0]
                    for it in pool[:40]:
                        if it in asked: continue
                        ul=zc(Q@foldin(np.array(rows+[Q[it]]),np.array(y+[1.0]))); ud=zc(Q@foldin(np.array(rows+[Q[it]]),np.array(y+[-1.0])))
                        ch=pl[it]*len(set(np.argsort(-ul)[:10])^bt)+(1-pl[it])*len(set(np.argsort(-ud)[:10])^bt)
                        if best is None or ch>best: best=ch;e=it
                else: e=next((it for it in order if it not in asked),None)
                if e is not None:
                    asked.add(e)
                    if e in rd: rows.append(Q[e]); y.append(rd[e])
            rel=[j for j in likes if j not in asked]
            uu=foldin(np.array(rows),np.array(y)) if rows else np.zeros(D)
            scv=W*zpop+min(len(rows)/(len(rows)+5.),CAP)*zc(Q@uu) if rows else W*zpop
            r=ndrc(scv,rel,asked)
            if r: ND[t]+=r[0];RC[t]+=r[1]
        m+=1
    return ND/m,RC/m
print(f"P1 item panel; tune W (and conf-cap) on VAL, report TEST. users te={min(NU,len(te))}\n",flush=True)
for kind in ['pop','rmva','eig']:
    best=None
    for W in [1,2,4,8,16,32]:
        for CAP in [1.0,0.5,0.25]:
            nd,_=run(kind,va,W,CAP)
            if best is None or nd[T]>best[0]: best=(nd[T],W,CAP)
    W,CAP=best[1],best[2]; nd,rc=run(kind,te,W,CAP)
    print(f"{kind:<6} VAL-best W={W} cap={CAP} | TEST NDCG q0={nd[0]:.3f} q8={nd[T]:.3f} | Rec@10 q8={rc[T]:.3f} | {'BEATS pop-anchor' if nd[T]>nd[0]+0.005 else 'no gain over q0'}",flush=True)
