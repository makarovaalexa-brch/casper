"""
OUT-OF-SAMPLE concept elicitation: arbitrary NON-GENOME multi-word concept phrases, grounded by SBERT.
- concept factor = SBERT-similarity-weighted centroid of item Q-factors (soft attribute).
- answer (profile-only, leakage-free) = SBERT-sim-weighted debiased rating + CONFIDENCE GATE (abstain='don't know').
- corrected recipe: EAR-style ADDITIVE channel + BOUNDED weight; affinity selection.
Controls: GENRES (graded-lift additive bounded). Reports NDCG@10/Recall@10 q0..qT + monotone.
Movie text enriched with genome themes (items legitimately have content); the CONCEPTS are out-of-vocab (not genome tags).
"""
import os, numpy as np, torch
from sentence_transformers import SentenceTransformer
base='C:/dev/phd/casper/data/movielens'; ml=f'{base}/ml-1m'; D=64
W=float(os.environ.get('W',2.0)); WA=float(os.environ.get('WA',0.4)); T=int(os.environ.get('T',10))
NU=int(os.environ.get('NU',120)); SUP=int(os.environ.get('SUP',5)); SIMTHR=float(os.environ.get('SIMTHR',0.30)); rng=np.random.default_rng(0)
I=[];R=[];Uu=[]
with open(f'{ml}/ratings.dat') as f:
    for line in f:
        a=line.strip().split('::'); Uu.append(int(a[0])); I.append(int(a[1])); R.append(float(a[2]))
I=np.array(I);R=np.array(R,np.float32);Uu=np.array(Uu)
uids={x:k for k,x in enumerate(np.unique(Uu))}; iids={x:k for k,x in enumerate(np.unique(I))}; ni=len(iids); nu=len(uids)
u=np.array([uids[x] for x in Uu]); ic=np.array([iids[x] for x in I])
rated={}
for k in range(len(u)): rated.setdefault(u[k],[]).append((ic[k],R[k]))
keep=[x for x in range(nu) if sum(1 for it,r in rated[x] if r>=4)>=8]; rng.shuffle(keep)
trU=keep[:int(0.8*len(keep))]; te=keep[int(0.9*len(keep)):]
cnt=np.zeros(ni)
for x in trU:
    for it,r in rated[x]:
        if r>=4: cnt[it]+=1
zpop=(np.log(cnt+1.0)-np.log(cnt+1.0).mean())/(np.log(cnt+1.0).std()+1e-9)
Q=torch.load(f'{base}/.cache/checkpoints/instrument_u_ml1m.pt')['Q'].numpy().astype(np.float32); qn=np.linalg.norm(Q,axis=1).mean()
GEN=['Action','Adventure','Animation',"Children's",'Comedy','Crime','Documentary','Drama','Fantasy','Film-Noir','Horror','Musical','Mystery','Romance','Sci-Fi','Thriller','War','Western']
gid={g:k for k,g in enumerate(GEN)}; na=len(GEN); item_g=np.zeros((ni,na),bool); title={}
with open(f'{ml}/movies.dat',encoding='latin-1') as f:
    for line in f:
        pp=line.strip().split('::'); m=int(pp[0])
        if m in iids:
            title[iids[m]]=pp[1]
            for g in pp[2].split('|'):
                if g in gid: item_g[iids[m],gid[g]]=True
Ag=np.zeros((na,D),np.float32); Pg=item_g[cnt>0].mean(0)
for g in range(na):
    s=np.where(item_g[:,g])[0]; Ag[g]=Q[s].mean(0)/(np.linalg.norm(Q[s].mean(0))+1e-9)*qn if len(s) else 0
# movie text + genome themes
tagname={}
with open(f'{base}/genome-tags.csv',encoding='latin-1') as f:
    next(f)
    for line in f: a=line.rstrip('\n').split(','); tagname[int(a[0])]=a[1]
sc_={}
with open(f'{base}/genome-scores.csv') as f:
    next(f)
    for line in f:
        a=line.split(','); m=int(a[0])
        if m in iids: sc_.setdefault(m,[]).append((float(a[2]),int(a[1])))
themes={iids[m]:", ".join(tagname[t] for r,t in sorted(v,reverse=True)[:12]) for m,v in sc_.items()}
mtext=[f"{title.get(k,'?')}. Themes: {themes.get(k,'')}." for k in range(ni)]
sb=SentenceTransformer('all-MiniLM-L6-v2')
print("encoding movies + OOS concepts...",flush=True)
ME=sb.encode(mtext,batch_size=128,normalize_embeddings=True,show_progress_bar=False).astype(np.float32)
CONCEPTS=["mind-bending plot twist","feel-good underdog story","slow-burn character study","visually stunning cinematography",
 "based on a true story","coming of age in adolescence","dark and morally ambiguous","quirky offbeat humour",
 "epic historical sweep","tense psychological cat and mouse","heartwarming story about family bonds","dystopian surveillance state",
 "hard-boiled noir detective mystery","satirical social commentary","tearjerker doomed romance","gritty urban crime saga",
 "whimsical fantasy adventure","cerebral hard science fiction","campy B-movie fun","intimate small-scale indie drama"]
CE=sb.encode(CONCEPTS,normalize_embeddings=True).astype(np.float32); nC=len(CONCEPTS)
sim=ME@CE.T                                            # [ni,nC] cosine
Ac=np.zeros((nC,D),np.float32)
for c in range(nC):
    w=np.clip(sim[:,c],0,None); w=np.where(w>SIMTHR,w,0)
    Ac[c]=(w[:,None]*Q).sum(0)/(w.sum()+1e-9); Ac[c]=Ac[c]/(np.linalg.norm(Ac[c])+1e-9)*qn
print("done.",flush=True)
def ndcg_rc(scv,rel_,excl):
    if not len(rel_): return 0.,0.
    s=scv.copy(); s[list(excl)]=-1e9; top=np.argsort(-s)[:10]; rs=set(int(t) for t in rel_)
    idcg=sum(1./np.log2(p+2) for p in range(min(10,len(rel_))))
    return (sum(1./np.log2(p+2) for p,t in enumerate(top) if int(t) in rs)/idcg if idcg else 0.,
            len(set(int(t) for t in top)&rs)/len(rel_))
def zc(v): return (v-v.mean())/(v.std()+1e-9)
def run(kind, sel='aff'):
    NDt=np.zeros(T+1); RCt=np.zeros(T+1); m_=0
    for x in te[:NU]:
        recs=rated[x]; allit=np.array([it for it,r in recs]); allr=np.array([r for it,r in recs],np.float32)
        likes=[it for it,r in recs if r>=4]
        if len(likes)<4: continue
        ll=likes[:]; rng.shuffle(ll); test=set(ll[len(ll)//2:])
        pm=np.array([int(i) not in test for i in allit]); pit=allit[pm]; pr=allr[pm]; profile=set(int(i) for i in pit); umean=pr.mean()
        rel_=list(test)
        ans={}
        if kind=='genre':
            for g in range(na):
                ing=item_g[pit,g]; pu=ing.mean() if len(pit) else 0.; ans[g]=float(np.tanh(np.log((pu+1e-3)/(Pg[g]+1e-3)+1e-6))); A=Ag
        else:
            simP=sim[pit]                                # [np,nC]
            for c in range(nC):
                w=np.clip(simP[:,c],0,None); supp=int((w>SIMTHR).sum())
                if supp<SUP: ans[c]=None; continue       # 'don't know'
                ans[c]=float(np.clip((w*(pr-umean)).sum()/(w.sum()+1e-6)/1.5,-1,1)); A=Ac
        cand=[e for e in ans if ans[e] is not None]
        order=sorted(cand,key=lambda e:-abs(ans[e])) if sel=='aff' else list(rng.permutation(cand))   # affinity vs random
        cpref=np.zeros(D); a0,b0=ndcg_rc(W*zpop,rel_,profile); curve=[a0]; rc=[b0]
        for e in order:
            if len(curve)>T: break
            cpref=cpref+ans[e]*A[e]; scv=W*zpop+WA*zc(Q@cpref)
            a,b=ndcg_rc(scv,rel_,profile); curve.append(a); rc.append(b)
        while len(curve)<=T: curve.append(curve[-1]); rc.append(rc[-1])
        NDt+=np.array(curve[:T+1]); RCt+=np.array(rc[:T+1]); m_+=1
    return NDt/m_, RCt/m_
def mono(c): return all(c[t]>=c[t-1]-0.004 for t in range(1,len(c)))
print(f"\nML-1M cold-test {min(NU,len(te))} users, additive WA={WA}, T={T}",flush=True)
print(f"{'arm':<22} | NDCG@10 q0..qT | mono? | Rec@10 qT",flush=True)
for kind,label in [('genre','GENRES'),('oos','OOS-CONCEPTS')]:
    for sel in ['aff','rand']:
        nd,rc=run(kind,sel); print(f"{label+' ('+sel+')':<22} | "+" ".join(f"{v:.3f}" for v in nd)+f" | {'YES' if mono(nd) else 'no'} | {rc[T]:.3f}",flush=True)
