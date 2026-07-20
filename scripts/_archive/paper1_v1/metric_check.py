"""
METRIC ALIGNMENT CHECK — is the s4 NDCG protocol on the lit-anchored ruler?
Same data, same popularity vector, same NDCG formula. Only the PROTOCOL differs.
 (CANONICAL = mf_foldin.py / DRE-Kweon anchor): rel = ALL of the test user's likes minus seeds; exclude ONLY seeds.
 (S4 panel):                                    rel = random HALF of likes (SPL); exclude the WHOLE profile.
If canonical MOSTPOP ~= 0.41 and s4 MOSTPOP ~= 0.326 on identical inputs, the gap is 100% protocol drift.
"""
import numpy as np
base='C:/dev/phd/casper/data/movielens'; ml=f'{base}/ml-1m'
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
# --- s4's user filter+split: keep>=8 likes (s4 uses >=8); canonical uses >=5. Test BOTH user pools. ---
def runpool(minlikes):
    keep=[x for x in range(nu) if sum(1 for it,r in rated[x] if r>=4)>=minlikes]; rng2=np.random.default_rng(0); rng2.shuffle(keep)
    trU=keep[:int(0.8*len(keep))]; te=keep[int(0.9*len(keep)):]
    cnt=np.zeros(ni)
    for x in trU:
        for it,r in rated[x]:
            if r>=4: cnt[it]+=1
    def ndcg(score,rel,excl):
        if not len(rel): return None
        s=score.copy(); s[list(excl)]=-1e9; top=np.argsort(-s)[:10]; rs=set(int(t) for t in rel)
        dcg=sum(1./np.log2(p+2) for p,t in enumerate(top) if int(t) in rs); idcg=sum(1./np.log2(p+2) for p in range(min(10,len(rel))))
        return dcg/idcg if idcg>0 else 0.
    # fixed split like s4
    rs=np.random.default_rng(123); SPL={}
    for x in te:
        lk=[it for it,r in rated[x] if r>=4]
        if len(lk)>=4: ll=lk[:]; rs.shuffle(ll); SPL[x]=set(ll[len(ll)//2:])
    canon=[]; s4=[]
    for x in te:
        if x not in SPL: continue
        rd={it:r for it,r in rated[x]}; likes=[it for it,r in rated[x] if r>=4]
        # CANONICAL: rel=all likes, seeds empty -> exclude nothing
        c=ndcg(cnt.copy(),likes,[])
        if c is not None: canon.append(c)
        # S4: rel=SPL (half likes), exclude whole profile (all rated minus test)
        test=SPL[x]; profile=set(rd)-test
        v=ndcg(cnt.copy(),list(test),profile)
        if v is not None: s4.append(v)
    print(f"  users>={minlikes}: n_te={len(canon)} | CANONICAL MOSTPOP NDCG@10={np.mean(canon):.4f} | S4-protocol MOSTPOP NDCG@10={np.mean(s4):.4f}")
print("MOSTPOP under two protocols (identical data + popularity vector):")
runpool(5)
runpool(8)
print("\nlit anchor (DRE/Kweon WWW2020, reproduced in mf_foldin.py): MOSTPOP NDCG@10 ~ 0.39-0.41")
