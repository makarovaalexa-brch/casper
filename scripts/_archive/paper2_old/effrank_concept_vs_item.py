"""Exp1 extra: effective-rank comparison of the 761 concept direction vectors vs the 600 pool-item
direction vectors (both unit-normalized). Quantifies why the concept channel saturates at ~4-5 answers
while items keep adding: a lower-rank / more-concentrated concept subspace covers fewer independent
directions of taste variation. Reports participation ratio (PR) and top-k variance explained."""
import numpy as np
base='C:/dev/phd/casper/data/movielens'
Q=np.load(f'{base}/.cache/Q_svd.npy'); Ec=np.load(f'{base}/.cache/Ec_concept.npy')
# rebuild PITEMS = top-600 popular train items (identical to continuous_policy2.py)
U,I,Rr=[],[],[]
with open(f'{base}/ml-1m/ratings.dat') as f:
    for line in f:
        a=line.strip().split('::'); U.append(int(a[0])); I.append(int(a[1])); Rr.append(float(a[2]))
U=np.array(U);I=np.array(I);Rr=np.array(Rr,np.float32)
uids={x:k for k,x in enumerate(np.unique(U))}; iids={x:k for k,x in enumerate(np.unique(I))}; nu,ni=len(uids),len(iids)
uu=np.array([uids[x] for x in U]); ii=np.array([iids[x] for x in I])
rat_by_u={}
for k in range(len(uu)): rat_by_u.setdefault(uu[k],[]).append((ii[k],float(Rr[k])))
likes_by_u={x:[j for j,r in v if r>=4] for x,v in rat_by_u.items()}
rng=np.random.default_rng(0)
keep=[x for x in range(nu) if len(likes_by_u.get(x,[]))>=5]; rng.shuffle(keep)
nK=len(keep); trU=keep[:int(0.8*nK)]
cnt=np.zeros(ni)
for x in trU:
    for j in likes_by_u.get(x,[]): cnt[j]+=1
PITEMS=list(np.argsort(-cnt)[:600])
Vitem=Q[np.array(PITEMS)]                                                       # (600,64) pool-item factor vectors
def analyze(name,M):
    Mn=M/(np.linalg.norm(M,axis=1,keepdims=True)+1e-9)                          # unit vectors
    C=Mn.T@Mn/Mn.shape[0]                                                        # (64,64) second-moment (uncentered covariance of unit dirs)
    w=np.linalg.eigvalsh(C)[::-1]; w=np.clip(w,0,None); s=w.sum()
    pr=(s*s)/(np.sum(w*w)+1e-12)                                                 # participation ratio = effective dimensionality
    cum=np.cumsum(w)/s
    tk={k:float(cum[k-1]) for k in (1,2,3,4,5,8,10,16,32)}
    # also center then do PCA-style (variance about the mean direction)
    Mc=Mn-Mn.mean(0,keepdims=True); Cc=Mc.T@Mc/Mn.shape[0]
    wc=np.linalg.eigvalsh(Cc)[::-1]; wc=np.clip(wc,0,None); sc=wc.sum()
    prc=(sc*sc)/(np.sum(wc*wc)+1e-12); cumc=np.cumsum(wc)/sc
    tkc={k:float(cumc[k-1]) for k in (1,2,3,4,5,8,10,16,32)}
    print(f"\n=== {name} ({M.shape[0]} unit vectors in R^{M.shape[1]}) ===")
    print(f"  UNCENTERED 2nd-moment:  participation ratio (eff. rank) = {pr:.2f} / {M.shape[1]}")
    print("   top-k var explained: "+"  ".join(f"k={k}:{tk[k]*100:4.1f}%" for k in (1,2,3,4,5,8,16,32)))
    print(f"  CENTERED (about mean):  participation ratio (eff. rank) = {prc:.2f} / {M.shape[1]}")
    print("   top-k var explained: "+"  ".join(f"k={k}:{tkc[k]*100:4.1f}%" for k in (1,2,3,4,5,8,16,32)))
    return pr,prc,tk,tkc
print("EFFECTIVE-RANK: concept direction vectors vs pool-item direction vectors (both unit-normalized)")
analyze("CONCEPTS (Ec, 761 tags)",Ec)
analyze("POOL ITEMS (Q[top-600 popular])",Vitem)
