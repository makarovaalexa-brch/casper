"""
Is there REALIZABLE headroom in PROFILE mode (where elicitation is real)? Decides whether a learned profile-mode
policy is worth building. Per user: test = held-out likes (SPL); the rest is the profile, split into
pa (fold/answer source) and pv (profile-internal validation, the realizable selection proxy). We ask T items from pa
and measure NDCG on the TRUE held-out test. Three selectors:
  HELF            : order pa by HELF score (realizable, no peeking)             [baseline]
  REALIZABLE-ORACLE: greedily pick the pa item maximizing pv-NDCG (no test peek) [realizable CEILING]
  TRUE-ORACLE     : greedily pick the pa item maximizing TEST-NDCG (privileged)  [upper bound]
If REALIZABLE-ORACLE ~= HELF << TRUE-ORACLE: no realizable headroom -> a learned policy can't help even here.
If REALIZABLE-ORACLE approaches TRUE-ORACLE: headroom exists -> build the learned policy.
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
SPL={}; _rs=np.random.default_rng(123)
for x in keep:
    items=list(dict(rat_by_u[x]))
    if len(items)>=6: il=items[:]; _rs.shuffle(il); SPL[x]=(set(il[:len(il)//2]),set(il[len(il)//2:]))
def foldin(F,y): F=np.array(F); return np.linalg.solve(F.T@F+LAM*np.eye(D),F.T@np.array(y,np.float32)) if len(F) else np.zeros(D)
_W=1./np.log2(np.arange(2,12))
def ndcg(score,rel,excl):
    if not rel: return 0.
    s=score.copy(); s[list(excl)]=-1e9; top=np.argpartition(-s,10)[:10]; top=top[np.argsort(-s[top])]
    return sum(_W[p] for p,t in enumerate(top) if int(t) in rel)/(_W[:min(10,len(rel))].sum()+1e-12)
def run(kind):
    ND=np.zeros(T+1); m=0
    for x in te[:NEVAL]:
        if x not in SPL: continue
        test,prof=SPL[x]; rd=dict(rat_by_u[x]); tlike=set(j for j in test if rd[j]>=4)
        prof=list(prof);
        if len(tlike)<1 or len(prof)<4: continue
        po=prof[:]; rng.shuffle(po); cut=len(po)//2; pa=po[:cut]; pv=set(j for j in po[cut:] if rd[j]>=4)  # answer source / proxy-val likes
        if not pa or not pv: continue
        excl_base=set(prof)
        F=[];y=[];asked=set(); nd=np.zeros(T+1); nd[0]=ndcg(popb.copy(),tlike,excl_base)
        for t in range(1,T+1):
            cands=[j for j in pa if j not in asked]
            if not cands: nd[t:]=nd[t-1]; break
            if kind=='helf': e=max(cands,key=lambda j:helf[j])
            else:
                tgt = tlike if kind=='true' else pv
                best=None
                for j in cands:
                    uu2=foldin(F+[Q[j]],y+[rd[j]-mu-bi[j]]); a=ndcg(popb+Q@uu2,tgt,excl_base|asked|{j})
                    if best is None or a>best[0]: best=(a,j)
                e=best[1]
            asked.add(e); F.append(Q[e]); y.append(rd[e]-mu-bi[e])
            nd[t]=ndcg(popb+Q@foldin(F,y),tlike,excl_base|asked)
        ND+=nd; m+=1
    return ND/m,m
print(f"PROFILE-mode realizable-ceiling test ({min(NEVAL,len(te))} users); NDCG@10 on TRUE held-out:\n",flush=True)
for kind in ['helf','realizable','true']:
    nd,m=run(kind); tag={'realizable':'REALIZABLE-ORACLE (pv-greedy)','true':'TRUE-ORACLE (test-peek)'}.get(kind,'HELF')
    print(f"{tag:<30}: "+" ".join(f"{v:.3f}" for v in nd)+f" | delta {nd[T]-nd[0]:+.3f}",flush=True)
