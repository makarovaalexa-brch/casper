"""Paper E interpretability: build a RICH, broad, movie-related PHRASE BANK in the recommender's R^64 factor space, so the
learned policy's continuous (Paper-C) queries can SNAP to a nearby named phrase. Sources: genome tags (~1100 nuanced
descriptors), genres, people (actors/directors), eras. Each phrase = centroid of its matched movies' factors (exactly how
genome concepts were built). Saves phrasebank.npz {vec[N,D], unit[N,D], label[N], kind[N]} + coverage report.
Run: python scripts/paper5/build_phrasebank.py"""
import os, json, csv, codecs, numpy as np
from collections import defaultdict
base='C:/dev/phd/casper/data/movielens'; ml=f'{base}/ml-1m'; D=64
# ml-1m index + factors
I=[int(l.split('::')[1]) for l in open(f'{ml}/ratings.dat')]
iids={x:k for k,x in enumerate(np.unique(np.array(I)))}; ni=len(iids)
Q=np.load(f'{base}/.cache/Q_svd.npy').astype(np.float32)
vecs=[]; labels=[]; kinds=[]
def add(vec,label,kind):
    if vec is not None and np.linalg.norm(vec)>1e-6: vecs.append(vec.astype(np.float32)); labels.append(label); kinds.append(kind)

# ---- genome tags (the rich core): tag -> relevance-weighted centroid of movies (rel>0.5) ----
tagname={}
for r in csv.reader(open(f'{base}/genome-tags.csv')):
    if r and r[0].isdigit(): tagname[int(r[0])]=r[1]
tagmov=defaultdict(list)
with open(f'{base}/genome-scores.csv') as f:
    next(f)
    for line in f:
        a=line.split(','); m=int(a[0]); tg=int(a[1]); rel=float(a[2])
        if m in iids and rel>0.5: tagmov[tg].append((iids[m],rel))
for tg,lst in tagmov.items():
    if len(lst)>=3:
        idx=np.array([j for j,_ in lst]); w=np.array([r for _,r in lst],np.float32); w/=w.sum()
        add((Q[idx]*w[:,None]).sum(0), tagname.get(tg,f'tag{tg}'), 'tag')
print(f"genome tags: {len(vecs)}")

# ---- genres ----
gfilms=defaultdict(list)
for line in codecs.open(f'{ml}/movies.dat','r','latin-1'):
    a=line.rstrip('\n').split('::')
    if len(a)>=3 and int(a[0]) in iids:
        for g in a[2].split('|'): gfilms[g].append(iids[int(a[0])])
for g,v in gfilms.items(): add(Q[np.array(sorted(set(v)))].mean(0), g, 'genre')
ng=len(vecs); print(f"+genres: {ng-len(tagmov)} ... total {ng}")

# ---- people (actors+directors with >=4 films) ----
try:
    cr=json.load(open(f'{base}/.cache/credits_ml1m_actors5.json')); pf=defaultdict(list)
    for src,kd in [('movie_actors','actor'),('movie_directors','director')]:
        for mid,ppl in cr[src].items():
            if int(mid) in iids:
                for p in ppl: pf[(kd,p)].append(iids[int(mid)])
    for (kd,p),v in pf.items():
        if len(set(v))>=4: add(Q[np.array(sorted(set(v)))].mean(0), p, kd)
except Exception as e: print('people skip',e)

# ---- eras (decade of release, from title year) ----
import re
decfilms=defaultdict(list)
for line in codecs.open(f'{ml}/movies.dat','r','latin-1'):
    a=line.rstrip('\n').split('::'); mm=re.search(r'\((\d{4})\)',a[1] if len(a)>1 else '')
    if mm and int(a[0]) in iids: decfilms[f"{int(mm.group(1))//10*10}s films"].append(iids[int(a[0])])
for d,v in decfilms.items():
    if len(v)>=20: add(Q[np.array(sorted(set(v)))].mean(0), d, 'era')

vec=np.stack(vecs); unit=vec/(np.linalg.norm(vec,axis=1,keepdims=True)+1e-9)
np.savez(f'{base}/.cache/phrasebank.npz', vec=vec, unit=unit, label=np.array(labels,dtype=object), kind=np.array(kinds))
from collections import Counter
print(f"\nPHRASE BANK: {len(vec)} phrases  | kinds: {dict(Counter(kinds))}")
# ---- coverage: how densely does the bank tile the space the recommender uses? ----
Qn=Q/(np.linalg.norm(Q,axis=1,keepdims=True)+1e-9)
# nearest-phrase cosine for every CATALOGUE item direction (proxy for "can we name any point the policy reaches?")
samp=Qn[np.random.default_rng(0).choice(ni,2000,replace=False)]
nn=(samp@unit.T).max(1)
print(f"coverage (catalogue-dir -> nearest phrase cos): mean {nn.mean():.3f}  p10 {np.percentile(nn,10):.3f}  min {nn.min():.3f}")
print("sample tags:", ', '.join(np.array(labels)[np.array(kinds)=='tag'][:12]))
