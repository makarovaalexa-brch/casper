"""
DIAGNOSTIC: WHY is polarity weak + why does everything collapse to blockbusters + are attributes an appendage?
Tests three hypotheses on the base biased-SVD factors (Q_svd/bi_svd) + the data:
 H1 BLOCKBUSTER ATTRACTOR: do popular items have larger factor norms ||Q[i]||? (large norm => high score for many u)
 H2 POLARITY UNDER-TRAINED BY DATA: what fraction of ratings/reveals are dislikes? mean |residual| like vs dislike?
 H3 ATTRIBUTE-AS-CENTROID: ||gcent[g]|| (genre coherence) vs base rate -> explains which genres condition well.
Also: how much of the score is the popularity FLOOR vs the personalization term (signal-to-floor ratio).
"""
import numpy as np
base='C:/dev/phd/casper/data/movielens'; ml=f'{base}/ml-1m'
U,I,R=[],[],[]
with open(f'{ml}/ratings.dat') as f:
    for line in f:
        a=line.strip().split('::'); U.append(int(a[0])); I.append(int(a[1])); R.append(float(a[2]))
U=np.array(U);I=np.array(I);R=np.array(R,np.float32)
uids={x:k for k,x in enumerate(np.unique(U))}; iids={x:k for k,x in enumerate(np.unique(I))}; nu,ni=len(uids),len(iids)
uu=np.array([uids[x] for x in U]); ii=np.array([iids[x] for x in I])
cnt=np.zeros(ni); np.add.at(cnt,ii,(R>=4).astype(float))
Q=np.load(f'{base}/.cache/Q_svd.npy'); bi=np.load(f'{base}/.cache/bi_svd.npy'); mu=float(R.mean()); D=Q.shape[1]
nrm=np.linalg.norm(Q,axis=1)
print("=== H1 BLOCKBUSTER ATTRACTOR: factor norm vs popularity ===")
rated=cnt>0; lp=np.log(cnt+1)
print(f"  corr(||Q_i||, log-popularity) = {np.corrcoef(nrm[rated],lp[rated])[0,1]:+.3f}")
op=np.argsort(-cnt)
print(f"  mean ||Q|| top-50 popular = {nrm[op[:50]].mean():.3f} | items ranked 500-1000 = {nrm[op[500:1000]].mean():.3f} | tail(rank>2000) = {nrm[op[2000:]][cnt[op[2000:]]>0].mean():.3f}")
print(f"  mean |item bias b_i| top-50 pop = {np.abs(bi[op[:50]]).mean():.3f} | tail = {np.abs(bi[op[2000:]][cnt[op[2000:]]>0]).mean():.3f}")
print("=== H2 POLARITY UNDER-TRAINED BY DATA ===")
print(f"  ratings: like(>=4) {np.mean(R>=4)*100:.0f}%  neutral(=3) {np.mean(R==3)*100:.0f}%  dislike(<3) {np.mean(R<3)*100:.0f}%")
resid=R-mu-bi[ii]
print(f"  mean taste-residual: likes(>=4) = {resid[R>=4].mean():+.3f} | dislikes(<3) = {resid[R<3].mean():+.3f} | |resid| like={np.abs(resid[R>=4]).mean():.3f} dislike={np.abs(resid[R<3]).mean():.3f}")
# in an avg profile, how many dislikes? (per user)
cu={}
for k in range(len(uu)): cu.setdefault(uu[k],[]).append(R[k])
fr=[np.mean(np.array(v)<3) for v in cu.values() if len(v)>=5]
print(f"  avg fraction of a user's ratings that are dislikes(<3) = {np.mean(fr)*100:.0f}%  => reconstruction target (LIKES only) ignores these")
print("=== H3 ATTRIBUTE-AS-CENTROID: genre coherence (||gcent||) explains which genres work ===")
GEN=['Action','Adventure','Animation',"Children's",'Comedy','Crime','Documentary','Drama','Fantasy','Film-Noir','Horror','Musical','Mystery','Romance','Sci-Fi','Thriller','War','Western']
gid={g:k for k,g in enumerate(GEN)}; na=len(GEN); item_g=np.zeros((ni,na),bool)
with open(f'{ml}/movies.dat',encoding='latin-1') as f:
    for line in f:
        pp=line.strip().split('::'); m=int(pp[0])
        if m in iids:
            for g in pp[2].split('|'):
                if g in gid: item_g[iids[m],gid[g]]=True
print("  genre        | #items | ||gcent|| | mean-item-||Q|| | coherence(=||mean||/mean||.||)")
rows=[]
for g in range(na):
    s=np.where(item_g[:,g])[0]
    if len(s)<2: continue
    gc=Q[s].mean(0); coh=np.linalg.norm(gc)/ (nrm[s].mean()+1e-9)
    rows.append((coh,GEN[g],len(s),np.linalg.norm(gc),nrm[s].mean()))
for coh,g,nn,gcn,mn in sorted(rows,reverse=True):
    print(f"  {g:<12} | {nn:5d}  |  {gcn:.3f}   |    {mn:.3f}       |  {coh:.2f}")
print("  (high coherence = tight cluster in factor space = genre 'like' conditions well; low = diffuse => weak)")
print("=== SIGNAL-TO-FLOOR: per-user, std of personalization (Q@u) vs the popularity floor (beta*popb) ===")
# fold a typical full profile via ridge, compare spread of Q@u to spread of popb
LAM=5.0; rng=np.random.default_rng(0)
import numpy.linalg as la
popb=np.log(cnt+1.0); rat_by_u={}
for k in range(len(uu)): rat_by_u.setdefault(uu[k],[]).append((ii[k],R[k]))
spreads=[]
for x in list(rat_by_u)[:500]:
    its=rat_by_u[x]
    if len(its)<5: continue
    F=np.array([Q[j] for j,_ in its]); y=np.array([r-mu-bi[j] for j,r in its],np.float32)
    u=la.solve(F.T@F+LAM*np.eye(D),F.T@y); spreads.append(np.std(Q@u))
print(f"  std(Q@u) over catalogue (mean over users) = {np.mean(spreads):.3f}  vs  std(popb) = {np.std(popb):.3f}  vs  std(beta*popb), beta=8 = {8*np.std(popb):.3f}")
print(f"  => with beta=8 the popularity floor spread is ~{8*np.std(popb)/np.mean(spreads):.0f}x the personalization spread (blockbuster attractor confirmed)")
