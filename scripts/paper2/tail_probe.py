"""
Is the 'no realizable elicitation value' a NDCG@10-on-dense artifact? Re-rank on the LONG TAIL (exclude the top-POP
popular items from candidates AND targets), so the top-K reflects taste not popularity. If realizable elicitation
(fold profile likes, HELF/random order) now shows a LARGE NDCG gain on the tail, personalization/elicitation is real
and visible -> dense top-10 was the wrong lens. ML-1M calibrated instrument, profile mode.
"""
import os, numpy as np
base='C:/dev/phd/casper/data/movielens'; ml=f'{base}/ml-1m'; D=64; LAM=5.0; NEVAL=int(os.environ.get('NEVAL',300))
POPX=int(os.environ.get('POPX',300)); rng=np.random.default_rng(0)
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
TOPPOP=set(int(j) for j in np.argsort(-cnt)[:POPX])     # blockbusters to EXCLUDE from tail ranking
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
def ndcg_tail(u,tlike,excl):
    sc=popb+Q@u; sc=sc.copy(); sc[list(excl|TOPPOP)]=-1e9    # remove blockbusters from candidates
    top=np.argpartition(-sc,10)[:10]; top=top[np.argsort(-sc[top])]
    return sum(_W[p] for p,t in enumerate(top) if int(t) in tlike)/(_W[:min(10,len(tlike))].sum()+1e-12) if tlike else 0.
_rs=np.random.default_rng(123); SPL={}
for x in te:
    items=list(dict(rat_by_u[x]))
    if len(items)>=6: il=items[:]; _rs.shuffle(il); SPL[x]=(il[:len(il)//2], il[len(il)//2:])
MILE=[0,1,2,4,8,16,9999]
def run(selector):
    acc={k:0. for k in MILE}; m=0
    for x in te[:NEVAL]:
        if x not in SPL: continue
        test,prof=SPL[x]; rd=dict(rat_by_u[x])
        tlike=set(j for j in test if rd[j]>=4 and j not in TOPPOP)    # TAIL targets only
        if len(prof)<4 or not tlike: continue
        excl=set(prof)
        if selector=='helf': order=sorted(prof,key=lambda j:-helf[j])
        elif selector in ('oracle','realizable'): order=None
        else: order=prof[:]; rng.shuffle(order)
        if selector=='realizable':                          # greedy on profile-internal validation (no test peek)
            po=list(prof); rng.shuffle(po); cut=max(2,len(po)//2); pa=po[:cut]; pv=set(j for j in po[cut:] if rd[j]>=4 and j not in TOPPOP)
            F=[];y=[];as2=set(); cur=ndcg_tail(foldin(F,y),tlike,excl)
            if 0 in acc: acc[0]+=cur
            for d in range(1,17):
                cands=[j for j in pa if j not in as2]
                if not cands or not pv:
                    for k in MILE:
                        if k>=d: acc[k]+=cur
                    break
                best=None
                for j in cands:
                    v=ndcg_tail(foldin(F+[Q[j]],y+[rd[j]-mu-bi[j]]),pv,excl|as2|{j})   # score on pv proxy
                    if best is None or v>best[0]: best=(v,j)
                as2.add(best[1]); F.append(Q[best[1]]); y.append(rd[best[1]]-mu-bi[best[1]])
                cur=ndcg_tail(foldin(F,y),tlike,excl|as2)   # but REPORT on true test tail
                if d in acc: acc[d]+=cur
            acc[9999]+=cur; m+=1; continue
        if selector=='oracle':
            F=[];y=[];as2=set(); DEPTH=16
            cur=ndcg_tail(foldin(F,y),tlike,excl)
            if 0 in acc: acc[0]+=cur
            for d in range(1,DEPTH+1):                      # single incremental greedy pass; record at milestones
                cands=[j for j in prof if j not in as2]
                if not cands:
                    for k in MILE:
                        if k>=d: acc[k]+=cur
                    break
                best=None
                for j in cands:
                    v=ndcg_tail(foldin(F+[Q[j]],y+[rd[j]-mu-bi[j]]),tlike,excl|as2|{j})
                    if best is None or v>best[0]: best=(v,j)
                as2.add(best[1]); F.append(Q[best[1]]); y.append(rd[best[1]]-mu-bi[best[1]]); cur=best[0]
                if d in acc: acc[d]+=cur
            acc[9999]+=cur
        else:
            full=len(order)
            for k in MILE:
                d=min(k,full); Fd=[Q[order[i]] for i in range(d)]; yd=[rd[order[i]]-mu-bi[order[i]] for i in range(d)]
                acc[k]+=ndcg_tail(foldin(Fd,yd),tlike,excl)
        m+=1
    return {k:acc[k]/m for k in MILE},m
print(f"TAIL-restricted NDCG@10 (exclude top-{POPX} popular), ML-1M profile mode, {min(NEVAL,len(te))} users:\n",flush=True)
for sel in ['random','helf','realizable','oracle']:
    N,m=run(sel); print(f"[{sel:<10}] "+" ".join(f"q{k if k<9999 else 'ALL'}={N[k]:.3f}" for k in MILE)+f" | n={m}",flush=True)
