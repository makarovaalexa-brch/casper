"""
Diagnostics for three questions:
 Q1: why do random/HELF INCREASE in Paper A (full_breakdown) but are FLAT in Paper B? Hypothesis: Paper A asks from
     the user's PROFILE (answerable items -> every Q folds in signal); Paper B asks from a fixed GLOBAL pool of
     popular items (mostly UNSEEN -> no fold-in -> flat). Test both asking modes on the SAME users.
 Q2: why is q0 different (0.305 vs 0.277)? Hypothesis: different user subset size (604 vs 250). Report both.
 Q3: is the MOSTPOP prior an artificially-hard intercept? Sweep beta (popularity-floor weight) in PROFILE mode and
     see whether a weaker prior reveals MORE elicitation value (bigger delta) or just degrades (law: pure
     personalization loses). Calibrated instrument; shared fixed split.
"""
import os, numpy as np, torch, scipy.linalg as sla
base='C:/dev/phd/casper/data/movielens'; ml=f'{base}/ml-1m'; D=64; LAM=5.0; T=8; POOL=120; rng=np.random.default_rng(0)
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
pool=np.array(np.argsort(-cnt)[:POOL])
H5=np.zeros((ni,5)); trset=set(trU)
for k in range(len(uu)):
    if uu[k] in trset: H5[ii[k],min(max(int(round(R[k])),1),5)-1]+=1
pm=H5/np.clip(H5.sum(1,keepdims=True),1,None); ent=-(pm*np.log(pm+1e-12)).sum(1)
lf=np.log(cnt+1)/np.log(cnt.max()+1); Hn=ent/np.log(5); helf=2*lf*Hn/(lf+Hn+1e-9)
SPL={}; _rs=np.random.default_rng(123)
for x in keep:
    items=list(dict(rat_by_u[x]))
    if len(items)>=6: il=items[:]; _rs.shuffle(il); SPL[x]=(set(il[:len(il)//2]),set(il[len(il)//2:]))
def foldin(F,y): F=np.array(F); return np.linalg.solve(F.T@F+LAM*np.eye(D),F.T@np.array(y,np.float32)) if len(F) else np.zeros(D)
_W=1./np.log2(np.arange(2,12))
def ndcg(score,rel,excl):
    s=score.copy(); s[list(excl)]=-1e9; top=np.argpartition(-s,10)[:10]; top=top[np.argsort(-s[top])]
    return sum(_W[p] for p,t in enumerate(top) if int(t) in rel)/(_W[:min(10,len(rel))].sum()+1e-12)
def curve(mode, selector, beta, users):
    ND=np.zeros(T+1); answered=0.; m=0
    for x in users:
        if x not in SPL: continue
        test,prof=SPL[x]; rd=dict(rat_by_u[x]); tlike=set(j for j in test if rd[j]>=4)
        if not tlike or not prof: continue
        if mode=='profile': cands=list(prof)
        else: cands=list(pool)
        if selector=='helf': order=sorted(cands,key=lambda j:-helf[j])
        else: o=cands[:]; rng.shuffle(o); order=o
        F=[];y=[];asked=set(); nd=np.zeros(T+1); nd[0]=ndcg(beta*popb,tlike,prof); na=0
        for t in range(1,T+1):
            if t-1<len(order):
                e=order[t-1]; asked.add(e)
                if e in prof: F.append(Q[e]); y.append(rd[e]-mu-bi[e]); na+=1   # answerable iff in profile
            u=foldin(F,y); nd[t]=ndcg(beta*popb+Q@u,tlike,prof|asked)
        ND+=nd; answered+=na; m+=1
    return ND/m, answered/m, m
print("=== Q1: asking from PROFILE (answerable) vs GLOBAL pool (mostly unseen); 250 users, beta=1 ===",flush=True)
for mode in ['profile','global']:
    for sel in ['random','helf']:
        nd,na,m=curve(mode,sel,1.0,te[:250])
        print(f"  {mode:<8} {sel:<6}: q0={nd[0]:.3f} q4={nd[4]:.3f} q8={nd[8]:.3f} | delta {nd[8]-nd[0]:+.3f} | avg answered/8 = {na:.2f}",flush=True)
print("\n=== Q2: q0 (MOSTPOP on held-out) by user-subset size ===",flush=True)
for nu_ in [250,604]:
    nd,_,m=curve('profile','random',1.0,te[:nu_]); print(f"  first {nu_} te users (n={m}): q0={nd[0]:.4f}",flush=True)
print("\n=== Q3: prior strength sweep (PROFILE mode, HELF), does a WEAKER prior reveal more elicitation value? ===",flush=True)
for beta in [0.0,0.25,0.5,1.0,2.0,4.0,8.0]:
    nd,_,m=curve('profile','helf',beta,te[:250]); print(f"  beta={beta:<4}: q0={nd[0]:.3f} q8={nd[8]:.3f} | delta {nd[8]-nd[0]:+.3f}",flush=True)
