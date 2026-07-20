"""
Audit: is the TRUE-ORACLE's huge NDCG@10 headroom REAL elicitation value, or a top-10 gaming artifact?
ML-1M calibrated instrument, profile-mode elicitation. For each metric (NDCG@10, Recall@50, RMSE) compare
random / HELF (realizable) vs TRUE-ORACLE greedy ON THAT METRIC (peeks at held-out). If oracle-minus-HELF is large
on NDCG@10 but ~0 on RMSE/Recall@50 -> the headroom is a top-heavy artifact, not genuine prediction improvement.
Held-out = half of each user's RATED items; tlike=held-out likes; ty=their ratings (for RMSE).
"""
import os, numpy as np
base='C:/dev/phd/casper/data/movielens'; ml=f'{base}/ml-1m'; D=64; LAM=5.0; T=8; NEVAL=int(os.environ.get('NEVAL',250)); rng=np.random.default_rng(0)
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
def metric_val(metric,u,tlike,excl,ty_items,ty):
    if metric=='rmse':
        pred=mu+bi[ty_items]+Q[ty_items]@u; return float(np.sqrt(np.mean((pred-ty)**2)))
    sc=popb+Q@u; sc=sc.copy(); sc[list(excl)]=-1e9
    if metric=='ndcg10':
        top=np.argpartition(-sc,10)[:10]; top=top[np.argsort(-sc[top])]
        return sum(_W[p] for p,t in enumerate(top) if int(t) in tlike)/(_W[:min(10,len(tlike))].sum()+1e-12) if tlike else 0.
    top=set(int(t) for t in np.argpartition(-sc,50)[:50]); return len(top&tlike)/len(tlike) if tlike else 0.  # recall50
def better(metric,a,b): return (a<b) if metric=='rmse' else (a>b)   # rmse: lower better
_rs=np.random.default_rng(123); SPL={}
for x in te:
    items=list(dict(rat_by_u[x]))
    if len(items)>=6: il=items[:]; _rs.shuffle(il); SPL[x]=(il[:len(il)//2], il[len(il)//2:])
def run(metric, selector):
    cv=np.zeros(T+1); m=0
    for x in te[:NEVAL]:
        if x not in SPL: continue
        test,prof=SPL[x]; rd=dict(rat_by_u[x]); tlike=set(j for j in test if rd[j]>=4)
        if len(prof)<4 or (metric!='rmse' and not tlike): continue
        ty_items=np.array(test); ty=np.array([rd[j] for j in test],np.float32); excl=set(prof)
        F=[];y=[];asked=set()
        cv[0]+=metric_val(metric,foldin(F,y),tlike,excl,ty_items,ty)
        last=cv[0]
        for t in range(1,T+1):
            cands=[j for j in prof if j not in asked]
            if not cands:
                for tt in range(t,T+1): cv[tt]+=last
                break
            if selector=='helf': e=max(cands,key=lambda j:helf[j])
            elif selector=='random': e=cands[int(rng.integers(len(cands)))]
            else:
                best=None
                for j in cands:
                    v=metric_val(metric,foldin(F+[Q[j]],y+[rd[j]-mu-bi[j]]),tlike,excl|asked|{j},ty_items,ty)
                    if best is None or better(metric,v,best[0]): best=(v,j)
                e=best[1]
            asked.add(e); F.append(Q[e]); y.append(rd[e]-mu-bi[e])
            val=metric_val(metric,foldin(F,y),tlike,excl|asked,ty_items,ty); cv[t]+=val; last=val
        m+=1
    return cv/m, m
for metric in ['ndcg10','recall50','rmse']:
    print(f"\n=== {metric} (q0 -> q{T}) ===",flush=True); out={}
    for sel in ['random','helf','oracle']:
        cv,m=run(metric,sel); out[sel]=cv
        print(f"  {sel:<8}(n={m}): "+" ".join(f"{v:.3f}" for v in cv)+f" | delta {cv[T]-cv[0]:+.3f}",flush=True)
    g=out['oracle'][T]-out['helf'][T]
    print(f"  ORACLE - HELF @q{T} = {g:+.3f}  ({'LARGE: real selection signal' if abs(g)>0.03 else 'small: NO real selection value -> NDCG headroom was top-10 artifact'})",flush=True)
