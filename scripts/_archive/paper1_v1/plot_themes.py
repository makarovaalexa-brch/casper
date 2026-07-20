"""
Add PLOT THEMES to item text (MovieLens Tag Genome) and replicate the established text->CF projection.
- Enrich item text: title + top-K Tag-Genome themes (fine-grained: 'dinosaurs','time travel','talking animals'...).
- Projection A: RIDGE adapter  W = Q^T S (S^T S + bI)^-1   (our linear baseline).
- Projection B: CONTRASTIVE MLP (InfoNCE) aligning SBERT(item) -> CF factor Q  -- the established approach
  (RLMRec, Ren et al. WWW 2024; CLCRec, Wei et al. MM 2021; A-LLMRec, Kim et al. KDD 2024). Cite + replicate.
Test: no-genre concepts -> rank of the EXPECTED film, for (title+genres) vs (title+themes), ridge vs contrastive.
"""
import os, time, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from sentence_transformers import SentenceTransformer
base='C:/dev/phd/casper/data/movielens'; ml=f'{base}/ml-1m'; D=64; BETA=10.0; KT=int(os.environ.get('KT',18))
I=[]
with open(f'{ml}/ratings.dat') as f:
    for line in f: I.append(int(line.split('::')[1]))
iids={x:k for k,x in enumerate(np.unique(np.array(I)))}; ni=len(iids); keepids=set(iids)
Q=torch.load(f'{base}/.cache/checkpoints/instrument_u_ml1m.pt')['Q'].numpy().astype(np.float32)
title={}; gtext={}
with open(f'{ml}/movies.dat',encoding='latin-1') as f:
    for line in f:
        p=line.strip().split('::'); m=int(p[0])
        if m in iids: title[iids[m]]=p[1]; gtext[iids[m]]=p[2].replace('|',', ')
# ---- Tag Genome themes ----
tagname={}
with open(f'{base}/genome-tags.csv',encoding='latin-1') as f:
    next(f)
    for line in f:
        a=line.rstrip('\n').split(','); tagname[int(a[0])]=a[1]
print("streaming genome-scores (435MB)...",flush=True); t0=time.time()
scores={}
with open(f'{base}/genome-scores.csv') as f:
    next(f)
    for line in f:
        a=line.split(','); m=int(a[0])
        if m in keepids:
            scores.setdefault(m,[]).append((float(a[2]),int(a[1])))
print(f"  parsed genome for {len(scores)} movies ({time.time()-t0:.0f}s)",flush=True)
themes={}
for m,lst in scores.items():
    top=sorted(lst,reverse=True)[:KT]; themes[iids[m]]=", ".join(tagname[t] for r,t in top)
cov=len(themes)/ni; print(f"  theme coverage {cov*100:.0f}% of items",flush=True)
txt_g=[f"{title.get(k,'?')}. Genres: {gtext.get(k,'')}." for k in range(ni)]
txt_t=[f"{title.get(k,'?')}. Themes: {themes.get(k, gtext.get(k,''))}." for k in range(ni)]
sb=SentenceTransformer('all-MiniLM-L6-v2')
print("encoding...",flush=True)
Sg=sb.encode(txt_g,batch_size=128,normalize_embeddings=True,show_progress_bar=False).astype(np.float32)
St=sb.encode(txt_t,batch_size=128,normalize_embeddings=True,show_progress_bar=False).astype(np.float32)
fitm=np.where(np.linalg.norm(Q,axis=1)>1e-6)[0]
def ridge(S): return (Q[fitm].T@S[fitm])@np.linalg.inv(S[fitm].T@S[fitm]+BETA*np.eye(S.shape[1]))
def contrastive(S, ep=60, tau=0.07):           # RLMRec/CLCRec-style InfoNCE alignment SBERT->Q
    Sm=torch.tensor(S[fitm]); Qm=F.normalize(torch.tensor(Q[fitm]),dim=1)
    net=nn.Sequential(nn.Linear(S.shape[1],256),nn.ReLU(),nn.Linear(256,D))
    opt=torch.optim.Adam(net.parameters(),1e-3,weight_decay=1e-5); n=len(fitm)
    for e in range(ep):
        perm=torch.randperm(n)
        for b in range(0,n,512):
            idx=perm[b:b+512]; z=F.normalize(net(Sm[idx]),dim=1); t=Qm[idx]
            logits=z@t.T/tau; loss=F.cross_entropy(logits,torch.arange(len(idx)))
            opt.zero_grad(); loss.backward(); opt.step()
    net.eval()
    proj=lambda s: net(torch.tensor(s)).detach().numpy()
    return proj
Wg=ridge(Sg); Wt=ridge(St); print("ridge fit. training contrastive...",flush=True); fct=contrastive(St)
def rank_expected(scorefn, concept, expect):
    sc=scorefn(concept); order=np.argsort(-sc)
    pos=next((r for r,i in enumerate(order) if any(e.lower() in title[i].lower() for e in expect)),None)
    return order[:5], pos
def s_ridge(W,S_unused): return lambda c: Q@(W@sb.encode([c],normalize_embeddings=True).astype(np.float32)[0])
def s_contr(proj): return lambda c: Q@proj(sb.encode([c],normalize_embeddings=True).astype(np.float32))[0]
CONS=[("dinosaurs",["Jurassic","Lost World"]),("time travel",["Back to the Future","Terminator","Twelve Monkeys"]),
      ("a shark attacking swimmers",["Jaws"]),("talking animals",["Babe","Dolittle","Lion King"]),
      ("boxing",["Rocky","Raging Bull"]),("escape from prison",["Shawshank","Great Escape","Alcatraz"]),
      ("artificial intelligence",["2001","Terminator","Matrix"]),("a heist robbery",["Heat","Usual Suspects","Reservoir","Italian Job"])]
arms=[("TITLE+GENRES (ridge)",s_ridge(Wg,Sg)),("TITLE+THEMES (ridge)",s_ridge(Wt,St)),("TITLE+THEMES (contrastive)",s_contr(fct))]
for label,fn in arms:
    print("\n"+"="*68+f"\n{label}\n"+"="*68,flush=True); ranks=[]
    for c,exp in CONS:
        top,pos=rank_expected(fn,c,exp); ranks.append(pos if pos is not None else 9999)
        print(f"  \"{c}\" (expect {exp[0]}; rank {pos}): "+"; ".join(title[i] for i in top[:4]),flush=True)
    hit=[r for r in ranks if r<9999]; print(f"  -> median expected-film rank: {int(np.median(hit)) if hit else 'NA'}; found {len(hit)}/{len(CONS)} in top-ranking",flush=True)
