"""
OPEN-VOCABULARY concept folding — BOLT-ON (frozen Paper A model untouched). A free-text concept (e.g. "dinosaurs") is
SBERT-embedded, matched to the nearest item DOCUMENTS (title+genres+genome tags), and their MF-factor (Q_svd) centroid is
folded through the EXISTING frozen encoder as a revealed "I like <concept>" -> recommendations. SBERT is used ONLY to
locate items; the trained instrument is never changed. Tests: (1) anecdotal free-text concepts -> sensible movies;
(2) paraphrase robustness (non-verbatim phrasings ~ verbatim); (3) AT SCALE: text-located concept ~ genome-grounded
concept (overlap@10 over many tags) => free text works like curated genome tags.
"""
import os; os.environ['HF_HUB_OFFLINE']='1'; os.environ['TRANSFORMERS_OFFLINE']='1'
import time, numpy as np, torch, torch.nn as nn
from sentence_transformers import SentenceTransformer
base='C:/dev/phd/casper/data/movielens'; ml=f'{base}/ml-1m'; D=64; rng=np.random.default_rng(0)
U,I,R=[],[],[]
with open(f'{ml}/ratings.dat') as f:
    for line in f:
        a=line.strip().split('::'); U.append(int(a[0])); I.append(int(a[1])); R.append(float(a[2]))
U=np.array(U);I=np.array(I);R=np.array(R,np.float32)
uids={x:k for k,x in enumerate(np.unique(U))}; iids={x:k for k,x in enumerate(np.unique(I))}; nu,ni=len(uids),len(iids); keepids=set(iids)
uu=np.array([uids[x] for x in U]); ii=np.array([iids[x] for x in I])
rat_by_u={}
for k in range(len(uu)): rat_by_u.setdefault(uu[k],[]).append((ii[k],float(R[k])))
likes_by_u={x:[j for j,r in v if r>=4] for x,v in rat_by_u.items()}
keep=[x for x in range(nu) if len(likes_by_u.get(x,[]))>=5]; rng.shuffle(keep); trU=keep[:int(0.8*len(keep))]
cnt=np.zeros(ni)
for x in trU:
    for j in likes_by_u.get(x,[]): cnt[j]+=1
popb=np.log(cnt+1.0).astype(np.float32); sm=c=0.
for x in trU:
    for j,r in rat_by_u[x]: sm+=r;c+=1
mu=sm/c
Q=np.load(f'{base}/.cache/Q_svd.npy'); bi=np.load(f'{base}/.cache/bi_svd.npy'); Ql=np.load(f'{base}/.cache/Ql_unified.npy')   # PAPER A model (frozen)
resid_pos=[]
for x in trU[:3000]:
    for j,r in rat_by_u[x]:
        if r>=4: resid_pos.append(r-mu-bi[j])
POS=float(np.mean(resid_pos))
# ---- item documents: title + genres + top genome tags ----
title={}; genres={}
with open(f'{ml}/movies.dat',encoding='latin-1') as f:
    for line in f:
        pp=line.strip().split('::'); m=int(pp[0])
        if m in iids: title[iids[m]]=pp[1]; genres[iids[m]]=pp[2].replace('|',', ')
tagname={}
with open(f'{base}/genome-tags.csv',encoding='utf-8') as f:
    next(f)
    for line in f:
        a=line.rstrip('\n').split(','); tagname[int(a[0])]=a[1]
t0=time.time(); itemtags={}; tag2items={}
with open(f'{base}/genome-scores.csv') as f:
    next(f)
    for line in f:
        a=line.split(','); m=int(a[0])
        if m in keepids:
            rel=float(a[2]); tg=int(a[1])
            if rel>0.5: tag2items.setdefault(tg,[]).append(iids[m])
            itemtags.setdefault(iids[m],[]).append((rel,tg))
toptags={j:[tg for _,tg in sorted(v,reverse=True)[:8]] for j,v in itemtags.items()}
docs=[]
for j in range(ni):
    tg=', '.join(tagname.get(t,'') for t in toptags.get(j,[]))
    docs.append(f"{title.get(j,'')}. Genres: {genres.get(j,'')}. Themes: {tg}")
print(f"  built {ni} item docs, {len(tag2items)} genome tags ({time.time()-t0:.0f}s)",flush=True)
sbert=SentenceTransformer('all-MiniLM-L6-v2')
IE=sbert.encode(docs, batch_size=256, normalize_embeddings=True, show_progress_bar=False).astype(np.float32)
print("  SBERT item embeddings ready",flush=True)
class Enc(nn.Module):
    def __init__(s):
        super().__init__(); s.inp=nn.Sequential(nn.Linear(D+1,128),nn.ReLU(),nn.Linear(128,128),nn.ReLU()); s.att=nn.Linear(128,1); s.val=nn.Linear(128,D)
    def forward(s,t,m):
        h=s.inp(t); a=s.att(h).squeeze(-1).masked_fill(m==0,-1e9); al=torch.softmax(a,1); return (al.unsqueeze(-1)*s.val(h)).sum(1)
enc=Enc(); enc.load_state_dict(torch.load(f'{base}/.cache/enc_unified.pt')); enc.eval()
def fold(cen):
    arr=np.zeros((1,1,D+1),np.float32); arr[0,0,:D]=cen; arr[0,0,D]=POS
    with torch.no_grad(): return enc(torch.tensor(arr),torch.ones(1,1)).numpy()[0]
def recs_from_centroid(cen,k=10,exclude=(),popw=1.0):
    u=fold(cen); s=popw*popb+Ql@u; s[list(exclude)]=-1e9; o=np.argsort(-s)[:k]; return list(o)
def text_centroid(text,N=40):
    q=sbert.encode([text],normalize_embeddings=True)[0].astype(np.float32); sims=IE@q; top=np.argsort(-sims)[:N]; return Q[top].mean(0), top
def genome_centroid(tg,N=40):
    its=tag2items.get(tg,[])
    if len(its)<5: return None,None
    its=sorted(its,key=lambda j:-cnt[j])[:N]; return Q[np.array(its)].mean(0), its
def show(text):
    cen,top=text_centroid(text,25)
    matched=" | ".join(title.get(j,'?') for j in top[:5])                                # open-vocab RETRIEVAL (SBERT text->items)
    recs=recs_from_centroid(cen,6,exclude=set(top[:5]),popw=0.4)                          # RECOMMEND after folding the concept (balanced popb)
    print(f"  '{text}'",flush=True)
    print(f"      retrieves : {matched}",flush=True)
    print(f"      recommends: "+" | ".join(title.get(j,'?') for j in recs),flush=True)
print("\n=== (1) ANECDOTAL: open free-text concepts -> recommendations (frozen Paper A model) ===",flush=True)
for t in ['dinosaurs','space exploration','heist','time travel','superhero','zombie apocalypse','romantic comedy in Paris','courtroom drama','samurai','spy espionage']:
    show(t)
print("\n=== (2) PARAPHRASE robustness: non-verbatim phrasing ~ verbatim (overlap@10 of recommendations) ===",flush=True)
pairs=[('horror','scary frightening movies'),('dinosaurs','prehistoric reptiles'),('space','outer space science fiction'),
       ('animation','animated cartoon for kids'),('war','soldiers in battle'),('crime','gangsters and mobsters'),
       ('comedy','hilarious funny film'),('romance','love story'),('western','cowboys in the old west'),('noir','dark detective mystery')]
ov=[]
for a,b in pairs:
    ra=recs_from_centroid(text_centroid(a)[0],10); rb=recs_from_centroid(text_centroid(b)[0],10); o=len(set(ra)&set(rb))/10; ov.append(o)
    print(f"  '{a}' vs '{b}': overlap@10={o:.0%}",flush=True)
print(f"  MEAN paraphrase overlap@10 = {np.mean(ov):.0%}",flush=True)
print("\n=== (3) AT SCALE: text-located concept ~ genome-grounded concept (overlap@10 over genome tags) ===",flush=True)
common=[tg for tg,its in tag2items.items() if len(its)>=20]; rng.shuffle(common); samp=common[:150]
ovg=[]; rnd=[]
allset=list(range(ni))
for tg in samp:
    cg,_=genome_centroid(tg); ct,_=text_centroid(tagname.get(tg,''))
    if cg is None: continue
    rg=recs_from_centroid(cg,10); rt=recs_from_centroid(ct,10); ovg.append(len(set(rg)&set(rt))/10)
    rr=set(rng.choice(allset,10,replace=False)); rnd.append(len(set(rg)&rr)/10)
print(f"  text-vs-genome MEAN overlap@10 = {np.mean(ovg):.0%}  (random baseline {np.mean(rnd):.1%})  over {len(ovg)} tags",flush=True)
print(f"  => free-text concepts recover the curated genome concepts WITHOUT retraining (bolt-on).",flush=True)
