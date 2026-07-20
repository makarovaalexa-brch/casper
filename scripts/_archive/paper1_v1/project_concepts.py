"""
S3 (open concepts) — project ARBITRARY natural-language concepts into the instrument's taste space via the SBERT
adapter W, and show they move recommendations correctly (the 'general concepts' part of the unified claim).
  W = Q^T S (S^T S + beta I)^{-1}     (ridge map: SBERT(item text) -> item factor)
  concept factor = W . SBERT(phrase); fold in 'like concept' (u ∝ concept factor); rank items.
Checks: (1) adapter alignment (W·SBERT(item) recovers the item's neighbours);
        (2) arbitrary concept phrases -> sensible top films;
        (3) a concept that names a genre lifts that genre's items (matches the centroid mechanism, G5-style).
"""
import os, time, numpy as np, torch
from sentence_transformers import SentenceTransformer
base='C:/dev/phd/casper/data/movielens/ml-1m'; D=64; BETA=float(os.environ.get('BETA',10.0))
# ---- data + Q (from S2 checkpoint) ----
I=[]; R=[];
with open(f'{base}/ratings.dat') as f:
    for line in f:
        a=line.strip().split('::'); I.append(int(a[1])); R.append(float(a[2]))
I=np.array(I); R=np.array(R,np.float32)
iids={x:k for k,x in enumerate(np.unique(I))}; ni=len(iids)
cnt=np.zeros(ni)
for k in range(len(I)):
    if R[k]>=4: cnt[iids[I[k]]]+=1
Q=torch.load(f'{base}/../.cache/checkpoints/instrument_u_ml1m.pt')['Q'].numpy().astype(np.float32)
title={}; gtext={}; item_g={}
GEN=['Action','Adventure','Animation',"Children's",'Comedy','Crime','Documentary','Drama','Fantasy','Film-Noir','Horror','Musical','Mystery','Romance','Sci-Fi','Thriller','War','Western']
with open(f'{base}/movies.dat',encoding='latin-1') as f:
    for line in f:
        p=line.strip().split('::'); mid=int(p[0])
        if mid in iids:
            k=iids[mid]; title[k]=p[1]; gtext[k]=p[2].replace('|',', '); item_g[k]=set(p[2].split('|'))
# ---- SBERT item text + embeddings ----
print("loading SBERT (all-MiniLM-L6-v2)...",flush=True); t0=time.time()
sb=SentenceTransformer('all-MiniLM-L6-v2')
texts=[f"{title.get(k,'?')}. Genres: {gtext.get(k,'')}." for k in range(ni)]
print(f"encoding {ni} item texts...",flush=True)
S=sb.encode(texts,batch_size=128,show_progress_bar=False,normalize_embeddings=True).astype(np.float32)
print(f"SBERT done ({time.time()-t0:.0f}s), dim={S.shape[1]}",flush=True)
# ---- fit adapter W: SBERT -> Q  (ridge), on items with enough signal ----
fit=np.where(cnt>=5)[0]                                   # fit on items with some collaborative signal
Sf=S[fit]; Qf=Q[fit]
W=(Qf.T@Sf)@np.linalg.inv(Sf.T@Sf+BETA*np.eye(S.shape[1]))   # [D,384]
def concept_factor(phrase):
    s=sb.encode([phrase],normalize_embeddings=True).astype(np.float32)[0]; return W@s
def top_titles(score,k=8,popmin=20):
    s=np.where(cnt>=popmin,score,-1e9); idx=np.argsort(-s)[:k]
    return [(title.get(j,'?'),gtext.get(j,'')) for j in idx]
# ---- (1) alignment: held-out items, does W.SBERT(item) recover the item itself / neighbours ----
held=fit[::7][:500]
pred=(W@S[held].T).T                                      # [n,D] predicted factors
cos=np.sum(pred*Q[held],1)/(np.linalg.norm(pred,axis=1)*np.linalg.norm(Q[held],axis=1)+1e-9)
print(f"\n(1) adapter alignment: mean cosine(W·SBERT(item), Q[item]) on held-out = {cos.mean():.3f}",flush=True)
# ---- (2) arbitrary concepts -> top films ----
print("\n(2) ARBITRARY CONCEPTS -> top recommendations (fold-in 'like concept'):",flush=True)
for phrase in ["dark psychological thriller","feel-good family animated film","epic space adventure",
               "classic film noir detective story","brutal war epic","romantic comedy",
               "scary slasher horror","gritty crime mob movie","sweeping historical romance"]:
    cf=concept_factor(phrase); u=cf/(np.linalg.norm(cf)**2+1.0); sc=Q@u
    print(f"\n  concept: \"{phrase}\"",flush=True)
    for nm,gs in top_titles(sc,6): print(f"     - {nm}  [{gs}]",flush=True)
# ---- (3) concept that names a genre lifts that genre's items (SBERT vs centroid) ----
print("\n(3) genre-naming concept lift (score-percentile of that genre's items):",flush=True)
for gname,phrase in [('Sci-Fi','science fiction movie'),('Horror','horror movie'),('War','war movie'),('Animation','animated cartoon')]:
    cf=concept_factor(phrase); u=cf/(np.linalg.norm(cf)**2+1.0); sc=Q@u
    pct=sc.argsort().argsort()/ni; gi=[k for k in range(ni) if gname in item_g.get(k,set())]
    print(f"  \"{phrase}\" -> {gname} items percentile {np.mean([pct[k] for k in gi]):.2f} (vs global 0.50)",flush=True)
np.save(f'{base}/../.cache/checkpoints/sbert_adapter_W.npy',W); print("\nsaved sbert_adapter_W.npy",flush=True)
