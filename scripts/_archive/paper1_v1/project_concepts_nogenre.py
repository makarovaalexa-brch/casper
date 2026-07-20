"""
Rigorous open-concept test: concept phrases containing NO genre words, plus a title-only ablation so the SBERT
adapter cannot key on genre tokens in the item text. If thematically-correct films still surface, the unified
space genuinely understands open concepts (not just genre keywords).
"""
import os, time, numpy as np, torch
from sentence_transformers import SentenceTransformer
base='C:/dev/phd/casper/data/movielens/ml-1m'; D=64; BETA=float(os.environ.get('BETA',10.0))
I=[]; R=[]
with open(f'{base}/ratings.dat') as f:
    for line in f:
        a=line.strip().split('::'); I.append(int(a[1])); R.append(float(a[2]))
I=np.array(I); R=np.array(R,np.float32); iids={x:k for k,x in enumerate(np.unique(I))}; ni=len(iids)
cnt=np.zeros(ni)
for k in range(len(I)):
    if R[k]>=4: cnt[iids[I[k]]]+=1
Q=torch.load(f'{base}/../.cache/checkpoints/instrument_u_ml1m.pt')['Q'].numpy().astype(np.float32)
title={}; gtext={}
with open(f'{base}/movies.dat',encoding='latin-1') as f:
    for line in f:
        p=line.strip().split('::'); mid=int(p[0])
        if mid in iids: title[iids[mid]]=p[1]; gtext[iids[mid]]=p[2].replace('|',', ')
sb=SentenceTransformer('all-MiniLM-L6-v2'); fit=np.where(cnt>=5)[0]
def fit_adapter(texts):
    S=sb.encode(texts,batch_size=128,show_progress_bar=False,normalize_embeddings=True).astype(np.float32)
    W=(Q[fit].T@S[fit])@np.linalg.inv(S[fit].T@S[fit]+BETA*np.eye(384)); return W
print("encoding (title+genres) and (title-only)...",flush=True); t0=time.time()
W_tg=fit_adapter([f"{title.get(k,'?')}. Genres: {gtext.get(k,'')}." for k in range(ni)])
W_to=fit_adapter([f"{title.get(k,'?')}" for k in range(ni)])
print(f"adapters fit ({time.time()-t0:.0f}s)",flush=True)
def top(W,phrase,k=6,popmin=20):
    cf=W@sb.encode([phrase],normalize_embeddings=True).astype(np.float32)[0]; u=cf/(np.linalg.norm(cf)**2+1.0)
    sc=np.where(cnt>=popmin,Q@u,-1e9); return [(title.get(j,'?'),gtext.get(j,'')) for j in np.argsort(-sc)[:k]]
CONCEPTS=["time travel","a daring heist","dinosaurs","boxing","outer space aliens invading earth",
          "coming of age in high school","a courtroom trial","escape from prison","spy during the cold war",
          "a wedding","talking animals","a shark in the ocean","survival in the wilderness","artificial intelligence robots"]
for label,W in [("TITLE+GENRES adapter",W_tg),("TITLE-ONLY adapter (no genre tokens)",W_to)]:
    print("\n"+"="*72+f"\n{label}\n"+"="*72,flush=True)
    for ph in CONCEPTS:
        print(f"\n  \"{ph}\"",flush=True)
        for nm,gs in top(W,ph): print(f"     - {nm}  [{gs}]",flush=True)
