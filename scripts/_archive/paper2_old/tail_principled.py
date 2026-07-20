"""
Confirm the long-tail elicitation finding under ESTABLISHED protocols (not the ad-hoc top-300):
 (A) Cremonesi/Koren/Turrin RecSys2010 long-tail: define the short HEAD as the most-popular items covering a fraction
     HEADMASS of total interaction mass; evaluate NDCG@10 on the LONG TAIL (head excluded from candidates+targets).
     Sweep HEADMASS in {0.2,0.33,0.5} to show robustness to the cutoff.
 (B) IPS-debiased Recall@10 (Schnabel ICML2016 / Yang RecSys2018): propensity p_i ∝ (pop_i)^0.5; per-user
     self-normalized IPS-Recall downweights popular hits -> rewards niche relevant items. Full catalogue (no exclusion).
For each protocol: random / HELF (realizable) vs TRUE-ORACLE (greedy on that metric, incremental to q8). ML-1M
calibrated instrument, profile mode. If realizable gain + oracle headroom persist across cutoffs & under IPS, real.
"""
import os, numpy as np
base='C:/dev/phd/casper/data/movielens'; ml=f'{base}/ml-1m'; D=64; LAM=5.0; T=8; NEVAL=int(os.environ.get('NEVAL',220)); rng=np.random.default_rng(0)
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
# Cremonesi head sets by cumulative interaction mass
order_pop=np.argsort(-cnt); cum=np.cumsum(cnt[order_pop])/cnt.sum()
HEAD={hm:set(int(j) for j in order_pop[:np.searchsorted(cum,hm)+1]) for hm in [0.2,0.33,0.5]}
for hm in HEAD: print(f"  head(mass {hm}) = {len(HEAD[hm])} items ({100*len(HEAD[hm])/ni:.1f}% of catalogue)",flush=True)
prop=(cnt/max(cnt.max(),1))**0.5; prop=np.clip(prop,1e-3,1.0); ipsw=1.0/prop   # IPS weights (downweight popular)
def foldin(F,y): F=np.array(F); return np.linalg.solve(F.T@F+LAM*np.eye(D),F.T@np.array(y,np.float32)) if len(F) else np.zeros(D)
_W=1./np.log2(np.arange(2,12))
def metric(kind,u,tlike,excl):
    sc=popb+Q@u; sc=sc.copy()
    if kind.startswith('lt'):
        hm=float('0.'+kind[2:]); sc[list(excl|HEAD[hm])]=-1e9
        top=np.argpartition(-sc,10)[:10]; top=top[np.argsort(-sc[top])]
        rel=set(t for t in tlike if t not in HEAD[hm])
        return sum(_W[p] for p,t in enumerate(top) if int(t) in rel)/(_W[:min(10,len(rel))].sum()+1e-12) if rel else None
    else:  # ips recall@10 (full catalogue)
        sc[list(excl)]=-1e9; top=set(int(t) for t in np.argpartition(-sc,10)[:10])
        num=sum(ipsw[t] for t in tlike if t in top); den=sum(ipsw[t] for t in tlike)
        return num/den if den>0 else None
_rs=np.random.default_rng(123); SPL={}
for x in te:
    items=list(dict(rat_by_u[x]))
    if len(items)>=6: il=items[:]; _rs.shuffle(il); SPL[x]=(il[:len(il)//2], il[len(il)//2:])
def run(kind, selector):
    q0s=0.; q8s=0.; m=0
    for x in te[:NEVAL]:
        if x not in SPL: continue
        test,prof=SPL[x]; rd=dict(rat_by_u[x]); tlike=set(j for j in test if rd[j]>=4)
        if len(prof)<4: continue
        v0=metric(kind,foldin([],[]),tlike,set(prof))
        if v0 is None: continue
        F=[];y=[];asked=set()
        if selector=='helf': order=sorted(prof,key=lambda j:-helf[j])
        elif selector=='random': o=prof[:]; rng.shuffle(o); order=o
        else: order=None
        for t in range(T):
            if selector=='oracle':
                cands=[j for j in prof if j not in asked];
                if not cands: break
                best=None
                for j in cands:
                    v=metric(kind,foldin(F+[Q[j]],y+[rd[j]-mu-bi[j]]),tlike,set(prof)|asked|{j})
                    if v is not None and (best is None or v>best[0]): best=(v,j)
                e=best[1] if best else cands[0]
            else:
                if t>=len(order): break
                e=order[t]
            asked.add(e); F.append(Q[e]); y.append(rd[e]-mu-bi[e])
        v8=metric(kind,foldin(F,y),tlike,set(prof)|asked)
        q0s+=v0; q8s+=(v8 if v8 is not None else v0); m+=1
    return q0s/m, q8s/m, m
print(f"\nML-1M profile mode, {NEVAL} users. q0 -> q8 under established protocols:",flush=True)
for kind in ['lt2','lt33','lt5','ips']:
    name={'lt2':'Cremonesi long-tail (head 20% mass)','lt33':'Cremonesi long-tail (head 33% mass)','lt5':'Cremonesi long-tail (head 50% mass)','ips':'IPS-debiased Recall@10'}[kind]
    print(f"\n--- {name} ---",flush=True)
    res={}
    for sel in ['random','helf','oracle']:
        a,b,m=run(kind,sel); res[sel]=(a,b); print(f"  {sel:<8}: q0={a:.3f} q8={b:.3f} | delta {b-a:+.3f}",flush=True)
    print(f"  oracle-minus-HELF @q8 = {res['oracle'][1]-res['helf'][1]:+.3f}",flush=True)
