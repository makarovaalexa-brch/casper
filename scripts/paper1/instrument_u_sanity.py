"""
Interpretable sanity checks for the CASPER-U instrument (loads instrument_u_ml1m.pt + genres).
Prints HUMAN-READABLE evidence the recommender does sensible things:
  1. reveal a single GENRE 'like' -> top movie titles should be that genre
  2. POLARITY: 'like sci-fi' vs 'dislike sci-fi' should move sci-fi items in opposite directions
  3. ITEM COHERENCE: reveal 'liked <movie>' -> nearest recommendations should be similar films
"""
import numpy as np, torch
base='C:/dev/phd/casper/data/movielens/ml-1m'; D=64; LAM=1.0
U,I,R=[],[],[]
with open(f'{base}/ratings.dat') as f:
    for line in f:
        a=line.strip().split('::'); U.append(int(a[0])); I.append(int(a[1])); R.append(float(a[2]))
I=np.array(I); R=np.array(R,np.float32); U=np.array(U)
iids={x:k for k,x in enumerate(np.unique(I))}; ni=len(iids); inv={v:k for k,v in iids.items()}
cnt=np.zeros(ni)
for k in range(len(I)):
    if R[k]>=4: cnt[iids[I[k]]]+=1
Q=torch.load(f'{base}/../.cache/checkpoints/instrument_u_ml1m.pt')['Q'].numpy().astype(np.float32)
title={}; genres_of={}
GEN=['Action','Adventure','Animation',"Children's",'Comedy','Crime','Documentary','Drama','Fantasy','Film-Noir','Horror','Musical','Mystery','Romance','Sci-Fi','Thriller','War','Western']
gid={g:k for k,g in enumerate(GEN)}; na=len(GEN); item_g=np.zeros((ni,na),bool)
with open(f'{base}/movies.dat',encoding='latin-1') as f:
    for line in f:
        p=line.strip().split('::'); mid=int(p[0])
        if mid in iids:
            title[iids[mid]]=p[1]; gs=p[2].split('|'); genres_of[iids[mid]]=gs
            for g in gs:
                if g in gid: item_g[iids[mid],gid[g]]=True
A=np.zeros((na,D),np.float32)
for g in range(na):
    it=np.where(item_g[:,g])[0]
    if len(it): c=Q[it].mean(0); A[g]=c/(np.linalg.norm(c)+1e-9)*np.linalg.norm(Q,axis=1).mean()
def foldin(rows,y): return np.linalg.solve(rows.T@rows+LAM*np.eye(D), rows.T@y)
def top(score, excl, k=8):
    s=score.copy(); s[list(excl)]=-1e9; return np.argsort(-s)[:k]
POPMIN=np.argsort(-cnt)  # for readability restrict to films with some popularity

print("="*70); print("SANITY 1 — reveal ONE genre 'like' -> top recommendations (should match genre)"); print("="*70)
for gname in ['Sci-Fi','Horror','Romance',"Children's",'War']:
    g=gid[gname]; u=foldin(A[g:g+1],np.array([1.0])); sc=Q@u
    # restrict to reasonably popular films so titles are recognizable
    sc=np.where(cnt>=20,sc,-1e9)
    tops=np.argsort(-sc)[:6]; frac=np.mean([gname in genres_of.get(t,[]) for t in tops])
    print(f"\n  like '{gname}'  (top-6, {frac*100:.0f}% are {gname}):")
    for t in tops: print(f"     - {title.get(t,'?')}  [{'|'.join(genres_of.get(t,[]))}]")

print("\n"+"="*70); print("SANITY 2 — POLARITY: like vs dislike a genre (mean score-percentile of that genre's items)"); print("="*70)
for gname in ['Sci-Fi','Horror','Romance']:
    g=gid[gname]; gi=np.where(item_g[:,g])[0]
    up=foldin(A[g:g+1],np.array([1.0])); dn=foldin(A[g:g+1],np.array([-1.0]))
    pl=(Q@up).argsort().argsort()/ni; mn=(Q@dn).argsort().argsort()/ni
    print(f"  {gname:<10} LIKE -> {pl[gi].mean():.2f} percentile | DISLIKE -> {mn[gi].mean():.2f} percentile  (like should be >> dislike)")

print("\n"+"="*70); print("SANITY 3 — ITEM COHERENCE: reveal 'liked <film>' -> nearest recommendations"); print("="*70)
def find(sub):
    for t,nm in title.items():
        if sub.lower() in nm.lower(): return t
    return None
for q in ['Toy Story','Star Wars: Episode IV','Silence of the Lambs','Aladdin']:
    t=find(q)
    if t is None: continue
    u=foldin(Q[t:t+1],np.array([1.0])); sc=Q@u; sc=np.where(cnt>=20,sc,-1e9)
    tops=top(sc,{t},6)
    print(f"\n  liked '{title[t]}' [{'|'.join(genres_of.get(t,[]))}] -> recommends:")
    for x in tops: print(f"     - {title.get(x,'?')}  [{'|'.join(genres_of.get(x,[]))}]")
