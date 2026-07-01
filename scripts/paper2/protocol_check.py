"""
Protocol reconciliation: the SAME model-independent MostPop (popularity) scorer, evaluated under different
PROTOCOLS on the SAME locked test users, to show our q0=0.293 vs the published 0.41 gap is PROTOCOL, not RS weakness.
  PROT-A (published/mf_foldin): rel = ALL the user's likes ; exclude nothing ; NDCG@10 / Recall@10
  PROT-B: rel = HELD-HALF likes ; exclude the known half ; NDCG@10 / Recall@10   (isolates the held-half effect)
  PROT-C (OURS): rel = HELD-HALF likes ; exclude the known half ; NDCG@10 / Recall@50  (== our ruler's q0)
"""
import numpy as np
base='C:/dev/phd/casper/data/movielens'; ml=f'{base}/ml-1m'; rng=np.random.default_rng(0)
U,I,Rr=[],[],[]
with open(f'{ml}/ratings.dat') as f:
    for line in f:
        a=line.strip().split('::'); U.append(int(a[0])); I.append(int(a[1])); Rr.append(float(a[2]))
U=np.array(U);I=np.array(I);Rr=np.array(Rr,np.float32)
uids={x:k for k,x in enumerate(np.unique(U))}; iids={x:k for k,x in enumerate(np.unique(I))}; nu,ni=len(uids),len(iids)
uu=np.array([uids[x] for x in U]); ii=np.array([iids[x] for x in I])
rat_by_u={}
for k in range(len(uu)): rat_by_u.setdefault(uu[k],[]).append((ii[k],float(Rr[k])))
likes_by_u={x:[j for j,r in v if r>=4] for x,v in rat_by_u.items()}
keep=[x for x in range(nu) if len(likes_by_u.get(x,[]))>=5]; rng.shuffle(keep); nK=len(keep); trU=keep[:int(0.8*nK)]; te=keep[int(0.9*nK):]
cnt=np.zeros(ni)
for x in trU:
    for j in likes_by_u.get(x,[]): cnt[j]+=1
POP=np.log(cnt+1.0).astype(np.float32)                                            # MostPop scorer (popularity, model-independent)
_rs=np.random.default_rng(123); SPL={}
for x in te:
    its=list(dict(rat_by_u[x]))
    if len(its)>=6: il=its[:]; _rs.shuffle(il); SPL[x]=(set(il[:len(il)//2]), il[len(il)//2:])
TE=[x for x in te if x in SPL][:150]
_W=1./np.log2(np.arange(2,12))
def ndcg_rec(score,rel,excl,RK):
    s=score.copy()
    if excl: s[list(excl)]=-1e9
    if not rel: return None
    o=np.argsort(-s); nd=sum(_W[p] for p,t in enumerate(o[:10]) if int(t) in rel)/(_W[:min(10,len(rel))].sum()+1e-12)
    rc=len(set(o[:RK].tolist())&rel)/len(rel); return nd,rc
def run(proto):
    nd=rc=0.;m=0
    for x in TE:
        profset,test=SPL[x]; rd=dict(rat_by_u[x])
        if proto=='A':                                                            # ALL likes, exclude nothing
            rel=set(j for j,r in rat_by_u[x] if r>=4); excl=set(); RK=10
        elif proto=='B':                                                          # held-half likes, exclude known half, R@10
            rel=set(j for j in test if rd[j]>=4); excl=profset; RK=10
        else:                                                                     # OURS: held-half likes, exclude known half, R@50
            rel=set(j for j in test if rd[j]>=4); excl=profset; RK=50
        r=ndcg_rec(POP,rel,excl,RK)
        if r: nd+=r[0];rc+=r[1];m+=1
    return nd/m,rc/m,m
print("=== MostPop (popularity, SAME scorer) under 3 PROTOCOLS on the locked 150 test users ===",flush=True)
for p,desc in [('A','published/mf_foldin: ALL likes, no exclusion, R@10'),('B','held-half likes, exclude known half, R@10'),('C','OURS: held-half, exclude known half, R@50')]:
    nd,rc,m=run(p); print(f"  PROT-{p}: NDCG@10={nd:.4f}  Recall@{10 if p!='C' else 50}={rc:.4f}   ({desc})",flush=True)
print("\n(expect: A ~0.41 reproduces published MostPop ; C ~0.29 == our ruler q0 -> the gap is PROTOCOL, not RS)",flush=True)
