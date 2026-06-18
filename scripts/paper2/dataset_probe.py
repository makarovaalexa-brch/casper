"""
Replicate the decisive probe on OTHER (established) datasets: does strategic elicitation have REALIZABLE headroom?
For a dataset of explicit (user,item,rating): k-core filter -> biased-SVD instrument (Koren'09, SGD) -> PROFILE-mode
realizable-ceiling test (HELF vs REALIZABLE-ORACLE[profile-internal val, no test peek] vs TRUE-ORACLE[test peek]).
If REALIZABLE-ORACLE >> HELF -> selection-friendly dataset (build the learned policy); if ~= HELF -> same negative.

Usage: DATA=amazon:Digital_Music | amazon:All_Beauty | ml100k | ml1m
"""
import os, numpy as np
base='C:/dev/phd/casper/data'; D=int(os.environ.get('D',32)); LAMF=0.05; LR=0.01; EP=int(os.environ.get('EP',25))
LAM=5.0; T=8; KCORE=int(os.environ.get('KCORE',5)); rng=np.random.default_rng(0)
DATA=os.environ.get('DATA','amazon:Digital_Music')
def load():
    rows=[]
    if DATA.startswith('amazon:'):
        cat=DATA.split(':')[1]
        with open(f'{base}/amazon/{cat}.csv',encoding='utf-8') as f:
            next(f)
            for line in f:
                a=line.rstrip('\n').split(',')
                if len(a)>=3:
                    try: rows.append((a[0],a[1],float(a[2])))
                    except: pass
    elif DATA=='ml100k':
        with open(f'{base}/movielens/ml-100k/u.data') as f:
            for line in f: a=line.split('\t'); rows.append((a[0],a[1],float(a[2])))
    elif DATA=='ml1m':
        with open(f'{base}/movielens/ml-1m/ratings.dat') as f:
            for line in f: a=line.strip().split('::'); rows.append((a[0],a[1],float(a[2])))
    return rows
rows=load(); LIKE=4.0
print(f"DATA={DATA}: {len(rows)} raw ratings",flush=True)
# k-core filter (iterate users>=KCORE ratings & items>=KCORE)
from collections import Counter
for _ in range(8):
    uc=Counter(r[0] for r in rows); ic=Counter(r[1] for r in rows)
    rows=[r for r in rows if uc[r[0]]>=KCORE and ic[r[1]]>=KCORE]
us=sorted(set(r[0] for r in rows)); it=sorted(set(r[1] for r in rows))
uid={u:k for k,u in enumerate(us)}; iid={i:k for k,i in enumerate(it)}; nu,ni=len(us),len(it)
print(f"  after {KCORE}-core: {len(rows)} ratings, {nu} users, {ni} items, density {len(rows)/(nu*ni+1e-9):.4f}",flush=True)
uu=np.array([uid[r[0]] for r in rows]); ii=np.array([iid[r[1]] for r in rows]); rr=np.array([r[2] for r in rows],np.float32)
rat_by_u={}
for k in range(len(uu)): rat_by_u.setdefault(int(uu[k]),[]).append((int(ii[k]),float(rr[k])))
likes_by_u={x:[j for j,r in v if r>=LIKE] for x,v in rat_by_u.items()}
keep=[x for x in range(nu) if len(likes_by_u.get(x,[]))>=5]; rng.shuffle(keep); n=len(keep)
if n<50: print("  too few usable users (>=5 likes); skip."); raise SystemExit
trU=set(keep[:int(0.8*n)]); te=keep[int(0.9*n):]
cnt=np.zeros(ni)
for x in trU:
    for j in likes_by_u.get(x,[]): cnt[j]+=1
popb=np.log(cnt+1.0).astype(np.float32)
mask=np.array([int(uu[k]) in trU for k in range(len(uu))]); mu=float(rr[mask].mean())
print(f"  usable users(>=5 likes): {n}; test={len(te)}; training biased-SVD (D={D})...",flush=True)
# biased SVD SGD
truser={x:k for k,x in enumerate(sorted(trU))}; ntr=len(truser)
rid=np.array([truser[int(uu[k])] for k in range(len(uu)) if mask[k]]); rit=ii[mask]; rrt=rr[mask]
bu=np.zeros(ntr,np.float32); bi=np.zeros(ni,np.float32)
P=(0.1*rng.standard_normal((ntr,D))).astype(np.float32); Q=(0.1*rng.standard_normal((ni,D))).astype(np.float32)
idx=np.arange(len(rrt))
for ep in range(EP):
    rng.shuffle(idx)
    for b0 in range(0,len(idx),16384):
        bx=idx[b0:b0+16384]; uS=rid[bx]; iS=rit[bx]
        pred=mu+bu[uS]+bi[iS]+np.sum(P[uS]*Q[iS],1); e=(rrt[bx]-pred).astype(np.float32)
        np.add.at(bu,uS,LR*(e-LAMF*bu[uS])); np.add.at(bi,iS,LR*(e-LAMF*bi[iS]))
        gP=LR*(e[:,None]*Q[iS]-LAMF*P[uS]); gQ=LR*(e[:,None]*P[uS]-LAMF*Q[iS]); np.add.at(P,uS,gP); np.add.at(Q,iS,gQ)
H5=np.zeros((ni,5))
for k in range(len(uu)):
    if int(uu[k]) in trU: H5[int(ii[k]),min(max(int(round(rr[k])),1),5)-1]+=1
pm=H5/np.clip(H5.sum(1,keepdims=True),1,None); entv=-(pm*np.log(pm+1e-12)).sum(1)
lf=np.log(cnt+1)/np.log(cnt.max()+1); Hn=entv/np.log(5); helf=2*lf*Hn/(lf+Hn+1e-9)
def foldin(F,y): F=np.array(F); return np.linalg.solve(F.T@F+LAM*np.eye(D),F.T@np.array(y,np.float32)) if len(F) else np.zeros(D)
_W=1./np.log2(np.arange(2,12))
def ndcg(score,rel,excl):
    if not rel: return 0.
    s=score.copy(); s[list(excl)]=-1e9; top=np.argpartition(-s,min(10,len(s)-1))[:10]; top=top[np.argsort(-s[top])]
    return sum(_W[p] for p,t in enumerate(top) if int(t) in rel)/(_W[:min(10,len(rel))].sum()+1e-12)
_rs=np.random.default_rng(123); SPL={}
for x in te:
    items=list(dict(rat_by_u[x]))
    if len(items)>=6: il=items[:]; _rs.shuffle(il); SPL[x]=(set(il[:len(il)//2]),set(il[len(il)//2:]))
def run(kind,NEVAL=400):
    ND=np.zeros(T+1); m=0
    for x in te[:NEVAL]:
        if x not in SPL: continue
        test,prof=SPL[x]; rd=dict(rat_by_u[x]); tlike=set(j for j in test if rd[j]>=LIKE); prof=list(prof)
        if len(tlike)<1 or len(prof)<4: continue
        po=prof[:]; rng.shuffle(po); cut=len(po)//2; pa=po[:cut]; pv=set(j for j in po[cut:] if rd[j]>=LIKE)
        if not pa or (kind=='realizable' and not pv): continue
        excl=set(prof); F=[];y=[];asked=set(); nd=np.zeros(T+1); nd[0]=ndcg(popb.copy(),tlike,excl)
        for t in range(1,T+1):
            cands=[j for j in pa if j not in asked]
            if not cands: nd[t:]=nd[t-1]; break
            if kind=='helf': e=max(cands,key=lambda j:helf[j])
            else:
                tgt=tlike if kind=='true' else pv; best=None
                for j in cands:
                    a=ndcg(popb+Q@foldin(F+[Q[j]],y+[rd[j]-mu-bi[j]]),tgt,excl|asked|{j})
                    if best is None or a>best[0]: best=(a,j)
                e=best[1]
            asked.add(e); F.append(Q[e]); y.append(rd[e]-mu-bi[e]); nd[t]=ndcg(popb+Q@foldin(F,y),tlike,excl|asked)
        ND+=nd; m+=1
    return ND/m,m
print(f"\n=== {DATA} PROFILE-mode realizable-ceiling (NDCG@10 on held-out) ===",flush=True)
res={}
for kind in ['helf','realizable','true']:
    nd,m=run(kind); res[kind]=nd; tag={'realizable':'REALIZABLE-ORACLE','true':'TRUE-ORACLE'}.get(kind,'HELF')
    print(f"{tag:<18}(n={m}): "+" ".join(f"{v:.3f}" for v in nd)+f" | delta {nd[T]-nd[0]:+.3f}",flush=True)
rg=res['realizable'][T]-res['helf'][T]; gap=res['true'][T]-res['helf'][T]
print(f"\nVERDICT: realizable-over-HELF={rg:+.3f}; privileged-oracle-gap={gap:+.3f}; "
      f"{'SELECTION-FRIENDLY (build policy)' if rg>0.02 else 'same negative (no realizable headroom)'}",flush=True)
