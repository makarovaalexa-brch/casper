"""Paper E: RICHER phrase bank -> denser, better coverage, still interpretable. Adds to v1 (tags/genres/people/eras):
 (a) TAG-PAIRS (co-occurring, "dark + funny") = interpretable compounds that densify the space;
 (b) more PEOPLE (>=3 films);
 (c) tag-labeled ITEM-CLUSTER centroids (KMeans) = GUARANTEE coverage of every region the recommender uses.
Saves .cache/phrasebank.npz (overwrites; v1 backed up as phrasebank_v1_2542.npz). Run: python scripts/paper5/build_phrasebank_rich.py"""
import os, json, codecs, re, numpy as np
from collections import defaultdict, Counter
from itertools import combinations
from sklearn.cluster import KMeans
base='C:/dev/phd/casper/data/movielens'; ml=f'{base}/ml-1m'; D=64
I=[int(l.split('::')[1]) for l in open(f'{ml}/ratings.dat')]
iids={x:k for k,x in enumerate(np.unique(np.array(I)))}; ni=len(iids)
Q=np.load(f'{base}/.cache/Q_svd.npy').astype(np.float32)
vecs=[];labels=[];kinds=[]
def add(v,l,k):
    if v is not None and np.linalg.norm(v)>1e-6: vecs.append(v.astype(np.float32));labels.append(l);kinds.append(k)

# genome tags + per-movie top tags (for pairs) ----------------------------------
import csv
tagname={int(r[0]):r[1] for r in csv.reader(open(f'{base}/genome-tags.csv')) if r and r[0].isdigit()}
mov_tags=defaultdict(list)   # movie idx -> [(tagid,rel)]
tagmov=defaultdict(list)
with open(f'{base}/genome-scores.csv') as f:
    next(f)
    for line in f:
        a=line.split(','); m=int(a[0]); tg=int(a[1]); rel=float(a[2])
        if m in iids and rel>0.5: j=iids[m]; mov_tags[j].append((tg,rel)); tagmov[tg].append(j)
for tg,js in tagmov.items():
    if len(js)>=3: add(Q[np.array(sorted(set(js)))].mean(0), tagname.get(tg,f'tag{tg}'), 'tag')
ntag=len(vecs); print(f"tags: {ntag}")

# tag PAIRS (co-occurring, interpretable compounds) -----------------------------
pair=Counter()
for j,tl in mov_tags.items():
    top=[tg for tg,_ in sorted(tl,key=lambda x:-x[1])[:18]]                     # top-18 tags/movie -> pairs
    for a,b in combinations(sorted(top),2): pair[(a,b)]+=1
tset={tg:set(js) for tg,js in tagmov.items()}
for (a,b),c in pair.most_common(2600):
    if c<15: break
    inter=sorted(tset[a]&tset[b])
    if len(inter)>=10: add(Q[np.array(inter)].mean(0), f"{tagname.get(a,a)} + {tagname.get(b,b)}", 'tagpair')
print(f"+tagpairs: {len(vecs)-ntag}")

# genres + genre-avoid handled elsewhere; genres here ---------------------------
gfilms=defaultdict(list)
for line in codecs.open(f'{ml}/movies.dat','r','latin-1'):
    a=line.rstrip('\n').split('::')
    if len(a)>=3 and int(a[0]) in iids:
        for g in a[2].split('|'): gfilms[g].append(iids[int(a[0])])
for g,v in gfilms.items(): add(Q[np.array(sorted(set(v)))].mean(0), g, 'genre')

# people (>=3 films) ------------------------------------------------------------
try:
    cr=json.load(open(f'{base}/.cache/credits_ml1m_actors5.json')); pf=defaultdict(list)
    for src,kd in [('movie_actors','actor'),('movie_directors','director')]:
        for mid,ppl in cr[src].items():
            if int(mid) in iids:
                for p in ppl: pf[(kd,p)].append(iids[int(mid)])
    for (kd,p),v in pf.items():
        if len(set(v))>=3: add(Q[np.array(sorted(set(v)))].mean(0), p, kd)
except Exception as e: print('people skip',e)

# eras --------------------------------------------------------------------------
dec=defaultdict(list)
for line in codecs.open(f'{ml}/movies.dat','r','latin-1'):
    a=line.rstrip('\n').split('::'); mm=re.search(r'\((\d{4})\)',a[1] if len(a)>1 else '')
    if mm and int(a[0]) in iids: dec[f"{int(mm.group(1))//10*10}s films"].append(iids[int(a[0])])
for d,v in dec.items():
    if len(v)>=20: add(Q[np.array(sorted(set(v)))].mean(0), d, 'era')

# tag-labeled ITEM-CLUSTER centroids = coverage guarantee -----------------------
nc=len(vecs); TAGV=np.stack(vecs[:ntag]); TAGU=TAGV/(np.linalg.norm(TAGV,axis=1,keepdims=True)+1e-9); TAGL=labels[:ntag]
K=int(os.environ.get('KCLUST','900')); km=KMeans(n_clusters=K,n_init=4,random_state=0).fit(Q)
for c in range(K):
    cen=km.cluster_centers_[c]; cu=cen/(np.linalg.norm(cen)+1e-9)
    top=np.argsort(-(TAGU@cu))[:3]; add(cen, "≈ "+" · ".join(TAGL[t] for t in top), 'cluster')
print(f"+clusters: {len(vecs)-nc}")

vec=np.stack(vecs); unit=vec/(np.linalg.norm(vec,axis=1,keepdims=True)+1e-9)
np.savez(f'{base}/.cache/phrasebank.npz', vec=vec, unit=unit, label=np.array(labels,dtype=object), kind=np.array(kinds))
print(f"\nRICH PHRASE BANK: {len(vec)} | kinds {dict(Counter(kinds))}")
Qn=Q/(np.linalg.norm(Q,axis=1,keepdims=True)+1e-9); samp=Qn[np.random.default_rng(0).choice(ni,2000,replace=False)]; nn=(samp@unit.T).max(1)
print(f"coverage (catalogue-dir -> nearest phrase cos): mean {nn.mean():.3f}  p10 {np.percentile(nn,10):.3f}  min {nn.min():.3f}")
print("sample pairs:", ', '.join(np.array(labels)[np.array(kinds)=='tagpair'][:8]))
print("sample clusters:", ', '.join(np.array(labels)[np.array(kinds)=='cluster'][:5]))
