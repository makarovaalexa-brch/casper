import numpy as np
base='C:/dev/phd/casper/data/movielens/ml-1m'; rng=np.random.default_rng(0)
U,I,R=[],[],[]
for line in open(base+'/ratings.dat'):
    a=line.strip().split('::'); U.append(int(a[0])); I.append(int(a[1])); R.append(float(a[2]))
U=np.array(U); I=np.array(I); R=np.array(R,np.float32)
uids={x:k for k,x in enumerate(np.unique(U))}; nu=len(uids); uu=np.array([uids[x] for x in U])
rat={}
for k in range(len(uu)): rat.setdefault(uu[k],[]).append((I[k],R[k]))
likes={x:[j for j,r in v if r>=4] for x,v in rat.items()}
nrat=[len(rat[x]) for x in range(nu)]
keep=[x for x in range(nu) if len(likes.get(x,[]))>=5]; rng.shuffle(keep); nK=len(keep)
trU=keep[:int(0.8*nK)]; te=keep[int(0.9*nK):]
trbig=[x for x in trU if len(rat[x])>=14 and len(likes[x])>=6]
print('total ML-1M users      :', nu)
print('min ratings per user   :', min(nrat), '(ML-1M floor=20, so the >=14 rated filter drops NONE)')
print('keep (>=5 likes)       :', nK)
print('trU (80% split)        :', len(trU))
print('trbig (TRAINED ON)     :', len(trbig), '= trU with >=6 likes (essentially all of trU)')
print('te (10% held-out)      :', len(te))
print('frac of all users trained:', round(len(trbig)/nu,3))
