"""
Resolve 'is popularity the ceiling?': fold the WHOLE profile (realizable HELF/random order) to full depth and see
whether NDCG/RMSE climb to the oracle level (~0.46/0.87) or plateau near popularity (~0.28/0.98). ML-1M calibrated
instrument, profile mode. If realizable full-fold reaches the oracle -> personalization works fine with VOLUME of
reveals; the oracle's edge is just EFFICIENCY (fewer questions). If it plateaus low -> realizable is fundamentally capped.
"""
import os, numpy as np
base='C:/dev/phd/casper/data/movielens'; ml=f'{base}/ml-1m'; D=64; LAM=5.0; NEVAL=int(os.environ.get('NEVAL',300)); rng=np.random.default_rng(0)
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
trU=keep[:int(0.8*n)]; te=keep[int(0.9*n):]
cnt=np.zeros(ni)
for x in trU:
    for j in likes_by_u.get(x,[]): cnt[j]+=1
popb=np.log(cnt+1.0).astype(np.float32)
sm=c=0.
for x in trU:
    for j,r in rat_by_u[x]: sm+=r; c+=1
mu=sm/c
Q=np.load(f'{base}/.cache/Q_svd.npy'); bi=np.load(f'{base}/.cache/bi_svd.npy')
H5=np.zeros((ni,5)); trset=set(trU)
for k in range(len(uu)):
    if uu[k] in trset: H5[ii[k],min(max(int(round(R[k])),1),5)-1]+=1
pm=H5/np.clip(H5.sum(1,keepdims=True),1,None); entv=-(pm*np.log(pm+1e-12)).sum(1)
lf=np.log(cnt+1)/np.log(cnt.max()+1); Hn=entv/np.log(5); helf=2*lf*Hn/(lf+Hn+1e-9)
def foldin(F,y): F=np.array(F); return np.linalg.solve(F.T@F+LAM*np.eye(D),F.T@np.array(y,np.float32)) if len(F) else np.zeros(D)
_W=1./np.log2(np.arange(2,12))
def ndcg(u,tlike,excl):
    sc=popb+Q@u; sc[list(excl)]=-1e9; top=np.argpartition(-sc,10)[:10]; top=top[np.argsort(-sc[top])]
    return sum(_W[p] for p,t in enumerate(top) if int(t) in tlike)/(_W[:min(10,len(tlike))].sum()+1e-12) if tlike else 0.
def rmse(u,ty_items,ty): pred=mu+bi[ty_items]+Q[ty_items]@u; return float(np.sqrt(np.mean((pred-ty)**2)))
_rs=np.random.default_rng(123); SPL={}
for x in te:
    items=list(dict(rat_by_u[x]))
    if len(items)>=6: il=items[:]; _rs.shuffle(il); SPL[x]=(il[:len(il)//2], il[len(il)//2:])
MILE=[0,1,2,4,8,16,9999]
def run(selector):
    accN={k:0. for k in MILE}; accR={k:0. for k in MILE}; m=0
    for x in te[:NEVAL]:
        if x not in SPL: continue
        test,prof=SPL[x]; rd=dict(rat_by_u[x]); tlike=set(j for j in test if rd[j]>=4)
        if len(prof)<4 or not tlike: continue
        ty_items=np.array(test); ty=np.array([rd[j] for j in test],np.float32); excl=set(prof)
        if selector=='helf': order=sorted(prof,key=lambda j:-helf[j])
        else: order=prof[:]; rng.shuffle(order)
        F=[];y=[]; depth={};
        # record at each milestone depth
        def rec(d):
            u=foldin(F,y); return ndcg(u,tlike,excl), rmse(u,ty_items,ty)
        nd0,rm0=rec(0);
        seen={0:(nd0,rm0)}
        for t in range(1,len(order)+1):
            j=order[t-1]; F.append(Q[j]); y.append(rd[j]-mu-bi[j])
            seen[t]=None  # mark; compute lazily at milestones
        # compute milestone values
        full=len(order)
        for k in MILE:
            d=min(k,full)
            Fd=[Q[order[i]] for i in range(d)]; yd=[rd[order[i]]-mu-bi[order[i]] for i in range(d)]
            u=foldin(Fd,yd); accN[k]+=ndcg(u,tlike,excl); accR[k]+=rmse(u,ty_items,ty)
        m+=1
    return {k:accN[k]/m for k in MILE},{k:accR[k]/m for k in MILE},m
print(f"Fold-the-whole-profile (ML-1M, profile mode, {min(NEVAL,len(te))} users). depth 9999=full profile.\n",flush=True)
for sel in ['random','helf']:
    N,Rm,m=run(sel)
    print(f"[{sel}] NDCG@10 by depth: "+" ".join(f"q{k if k<9999 else 'ALL'}={N[k]:.3f}" for k in MILE),flush=True)
    print(f"        RMSE     by depth: "+" ".join(f"q{k if k<9999 else 'ALL'}={Rm[k]:.3f}" for k in MILE),flush=True)
print("\n(reference: q0=MOSTPOP; oracle@q8 ~ NDCG 0.46 / RMSE 0.87)",flush=True)
