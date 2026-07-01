"""
S4 Panel A - CHEAP / no-train elicitation baselines on the CASPER-U instrument (realizable; policy never sees the
answer before choosing). Item + attribute based. Cold-test users; NDCG@10/Recall@10 vs #questions.
Policies:
  RANDOM-item, POPULARITY-item, ENTROPY-item (Rashid'02), HELF=pop x entropy (Rashid'08),
  RANDOM-genre, POPULAR-genre, MAXENT-genre (SCPR ask-heuristic),
  UNCERTAINTY-item (uncertainty sampling), EIG-MIXED (greedy 1-step expected information gain over items+genres) [must-include].
Answer model: item -> user's centered rating if rated, else 'not seen' (wasted turn); genre -> +/-1 from profile.
"""
import os, numpy as np, torch
base='C:/dev/phd/casper/data/movielens'; ml=f'{base}/ml-1m'; D=64; LAM=1.0; W=float(os.environ.get('W',4.0))
T=int(os.environ.get('T',6)); NU=int(os.environ.get('NU',120)); POOL=int(os.environ.get('POOL',150)); rng=np.random.default_rng(0)
I=[];R=[];Uu=[]
with open(f'{ml}/ratings.dat') as f:
    for line in f:
        a=line.strip().split('::'); Uu.append(int(a[0])); I.append(int(a[1])); R.append(float(a[2]))
I=np.array(I);R=np.array(R,np.float32);Uu=np.array(Uu)
uids={x:k for k,x in enumerate(np.unique(Uu))}; iids={x:k for k,x in enumerate(np.unique(I))}; ni=len(iids); nu=len(uids)
u=np.array([uids[x] for x in Uu]); ic=np.array([iids[x] for x in I])
rated={}
for k in range(len(u)): rated.setdefault(u[k],[]).append((ic[k],(R[k]-3.)/2.,1.0 if R[k]>=4 else 0.0))
keep=[x for x in range(nu) if sum(t[2] for t in rated[x])>=8]; rng.shuffle(keep); tr=keep[:int(0.8*len(keep))]; te=keep[int(0.9*len(keep)):]
# warm item stats: like-count, rating entropy, variance, HELF
cnt=np.zeros(ni); ratesum={};
hist=np.zeros((ni,5))
for x in tr:
    for it,cr,lk in rated[x]:
        if lk>0: cnt[it]+=1
        r=int(round(cr*2+3)); hist[it,min(max(r,1),5)-1]+=1
popb=np.log(cnt+1.0).astype(np.float32); zpop=(popb-popb.mean())/(popb.std()+1e-9)
pmat=hist/np.clip(hist.sum(1,keepdims=True),1,None); ent=-(pmat*np.log(pmat+1e-12)).sum(1)
lf=np.log(cnt+1)/np.log(cnt.max()+1); Hn=ent/np.log(5); helf=2*lf*Hn/(lf+Hn+1e-9)
Q=torch.load(f'{base}/.cache/checkpoints/instrument_u_ml1m.pt')['Q'].numpy().astype(np.float32); qn=np.linalg.norm(Q,axis=1).mean()
GEN=['Action','Adventure','Animation',"Children's",'Comedy','Crime','Documentary','Drama','Fantasy','Film-Noir','Horror','Musical','Mystery','Romance','Sci-Fi','Thriller','War','Western']
gid={g:k for k,g in enumerate(GEN)}; na=len(GEN); item_g=np.zeros((ni,na),bool)
with open(f'{ml}/movies.dat',encoding='latin-1') as f:
    for line in f:
        pp=line.strip().split('::'); m=int(pp[0])
        if m in iids:
            for g in pp[2].split('|'):
                if g in gid: item_g[iids[m],gid[g]]=True
Ag=np.zeros((na,D),np.float32)
for g in range(na):
    s=np.where(item_g[:,g])[0]; Ag[g]=Q[s].mean(0)/(np.linalg.norm(Q[s].mean(0))+1e-9)*qn if len(s) else 0
pool=list(np.argsort(-cnt)[:POOL])                # answerable candidate pool (popular items)
# ---- fine genome concepts ----
THR=0.5; MINIT=30; tagname={}
with open(f'{base}/genome-tags.csv',encoding='latin-1') as f:
    next(f)
    for line in f: a=line.rstrip('\n').split(','); tagname[int(a[0])]=a[1]
tagit={}
with open(f'{base}/genome-scores.csv') as f:
    next(f)
    for line in f:
        a=line.split(','); m=int(a[0])
        if m in iids and float(a[2])>THR: tagit.setdefault(int(a[1]),[]).append(iids[m])
glow=set(g.lower() for g in GEN)
ctags=[t for t,its in tagit.items() if len(its)>=MINIT and tagname[t] not in glow]
nc=len(ctags); Ac=np.zeros((nc,D),np.float32); cset=[]; ccnt=np.zeros(nc)
for k,t in enumerate(ctags):
    s=np.array(tagit[t]); Ac[k]=Q[s].mean(0)/(np.linalg.norm(Q[s].mean(0))+1e-9)*qn; cset.append(set(s.tolist())); ccnt[k]=len(s)
cpool=list(np.argsort(-ccnt)[:100])               # capped concept pool for EIG
print(f"{nc} fine genome concepts loaded",flush=True)
def foldin(rows,y): return np.linalg.solve(rows.T@rows+LAM*np.eye(D), rows.T@y) if len(rows) else np.zeros(D)
def score(u_,m):
    sc=(W*zpop).copy()
    if m>0: z=Q@u_; z=(z-z.mean())/(z.std()+1e-9); sc=sc+(m/(m+5.))*z
    return sc
def nd_rc(sc,rel,excl):
    if not len(rel): return 0.,0.
    s=sc.copy(); s[list(excl)]=-1e9; top=np.argsort(-s)[:10]; rs=set(int(t) for t in rel)
    idcg=sum(1./np.log2(pos+2) for pos in range(min(10,len(rel))))
    dcg=sum(1./np.log2(pos+2) for pos,t in enumerate(top) if int(t) in rs)
    return (dcg/idcg if idcg else 0.), len(set(int(t) for t in top)&rs)/len(rel)
def run(policy):
    NDt=np.zeros(T+1); RCt=np.zeros(T+1); m_=0
    for x in te[:NU]:
        recs=rated[x]; rd={it:cr for it,cr,lk in recs}; likes=[it for it,cr,lk in recs if lk>0]
        if len(likes)<4: continue
        ll=likes[:]; rng.shuffle(ll); test=set(ll[len(ll)//2:])          # held-out targets
        profile=set(rd.keys())-test                                      # known items: excluded from ranking
        lg=set(g for g in range(na) if sum(item_g[it,g] for it in likes)>=2)
        ls=set(likes); lc=set(c for c in range(nc) if len(ls & cset[c])>=2)
        rows=[];y=[];asked_i=set();asked_g=set();asked_c=set(); rel=list(test)
        st=policy(rd,lg)
        a,b=nd_rc(score(np.zeros(D),0),rel,profile); NDt[0]+=a; RCt[0]+=b
        for t in range(T):
            kind,e=st.pick(rows,y,asked_i,asked_g,asked_c)
            if kind=='item':
                asked_i.add(e)
                if e in rd and e not in test: rows.append(Q[e]); y.append(rd[e])  # answerable profile item
            elif kind=='genre':
                asked_g.add(e); rows.append(Ag[e]); y.append(1.0 if e in lg else -1.0)
            else:
                asked_c.add(e); rows.append(Ac[e]); y.append(1.0 if e in lc else -1.0)
            uu=foldin(np.array(rows),np.array(y)) if rows else np.zeros(D)
            a,b=nd_rc(score(uu,len(rows)),rel,profile); NDt[t+1]+=a; RCt[t+1]+=b
        m_+=1
    return NDt/m_, RCt/m_
# ---- policy objects ----
class Seq:                                       # static item order
    def __init__(s,order): s.o=order;
    def pick(s,rows,y,ai,ag,ac=None):
        for it in s.o:
            if it not in ai: return 'item',it
        return 'item',0
def P_random(rd,lg): o=list(rng.permutation(ni)); return Seq(o)
def P_pop(rd,lg): return Seq(list(np.argsort(-cnt)))
def P_ent(rd,lg): return Seq(list(np.argsort(-ent)))
def P_helf(rd,lg): return Seq(list(np.argsort(-helf)))
class GenreSeq:
    def __init__(s,order): s.o=order
    def pick(s,rows,y,ai,ag,ac=None):
        for g in s.o:
            if g not in ag: return 'genre',g
        return 'genre',0
def P_rgenre(rd,lg): return GenreSeq(list(rng.permutation(na)))
def P_pgenre(rd,lg):
    gc=cnt@item_g; return GenreSeq(list(np.argsort(-gc)))
class MaxEntGenre:                               # ask genre that best bisects current top candidates
    def pick(s,rows,y,ai,ag,ac=None):
        uu=foldin(np.array(rows),np.array(y)) if rows else np.zeros(D)
        sc=score(uu,len(rows)); topN=np.argsort(-sc)[:200]; frac=item_g[topN].mean(0)
        best=None;bg=0
        for g in range(na):
            if g in ag: continue
            d=abs(frac[g]-0.5)
            if best is None or d<best: best=d;bg=g
        return 'genre',bg
def P_maxent(rd,lg): return MaxEntGenre()
class Uncertain:                                 # uncertainty sampling over answerable pool
    def pick(s,rows,y,ai,ag,ac=None):
        uu=foldin(np.array(rows),np.array(y)) if rows else np.zeros(D)
        z=Q@uu; pl=1/(1+np.exp(-2*(z-np.median(z))))
        best=None;bi=pool[0]
        for it in pool:
            if it in ai: continue
            d=abs(pl[it]-0.5)
            if best is None or d<best: best=d;bi=it
        return 'item',bi
def P_uncert(rd,lg): return Uncertain()
class EIG:                                        # greedy 1-step expected info gain over items(pool)+genres (realizable)
    def pick(s,rows,y,ai,ag,ac=None):
        uu=foldin(np.array(rows),np.array(y)) if rows else np.zeros(D); base=score(uu,len(rows))
        bt=np.argsort(-base)[:10];
        def util(sc): top=np.argsort(-sc)[:10]; return len(set(top)&set(bt))  # proxy utility = stability? use entropy instead
        z=Q@uu; plike=1/(1+np.exp(-2*(z-np.median(z))))
        best=None;pick=('item',pool[0])
        for it in pool:
            if it in ai: continue
            ul=score(foldin(np.array(rows+[Q[it]]),np.array(y+[1.0])),len(rows)+1)
            ud=score(foldin(np.array(rows+[Q[it]]),np.array(y+[-1.0])),len(rows)+1)
            # expected change in top-10 set (information) vs current
            ch=plike[it]*len(set(np.argsort(-ul)[:10])^set(bt))+(1-plike[it])*len(set(np.argsort(-ud)[:10])^set(bt))
            if best is None or ch>best: best=ch;pick=('item',it)
        for g in range(na):
            if g in ag: continue
            pg=plike[np.where(item_g[:,g])[0]].mean() if item_g[:,g].any() else 0.5
            ul=score(foldin(np.array(rows+[Ag[g]]),np.array(y+[1.0])),len(rows)+1)
            ud=score(foldin(np.array(rows+[Ag[g]]),np.array(y+[-1.0])),len(rows)+1)
            ch=pg*len(set(np.argsort(-ul)[:10])^set(bt))+(1-pg)*len(set(np.argsort(-ud)[:10])^set(bt))
            if ch>best: best=ch;pick=('genre',g)
        for c in cpool:
            if c in ac: continue
            pc=plike[list(cset[c])].mean() if cset[c] else 0.5
            ul=score(foldin(np.array(rows+[Ac[c]]),np.array(y+[1.0])),len(rows)+1)
            ud=score(foldin(np.array(rows+[Ac[c]]),np.array(y+[-1.0])),len(rows)+1)
            ch=pc*len(set(np.argsort(-ul)[:10])^set(bt))+(1-pc)*len(set(np.argsort(-ud)[:10])^set(bt))
            if ch>best: best=ch;pick=('concept',c)
        return pick
def P_eig(rd,lg): return EIG()
class ConceptSeq:
    def __init__(s,order): s.o=order
    def pick(s,rows,y,ai,ag,ac=None):
        for c in s.o:
            if c not in ac: return 'concept',c
        return 'concept',0
def P_pconcept(rd,lg): return ConceptSeq(list(np.argsort(-ccnt)))
import os as _os
if _os.environ.get('FOCUS','0')=='1':
    POLS=[('RANDOM-item',P_random),('POPULARITY-item',P_pop),('POPULAR-genre',P_pgenre),('POPULAR-concept',P_pconcept),('EIG-MIXED',P_eig)]
else:
    POLS=[('RANDOM-item',P_random),('POPULARITY-item',P_pop),('ENTROPY-item',P_ent),('HELF-item',P_helf),
      ('RANDOM-genre',P_rgenre),('POPULAR-genre',P_pgenre),('MAXENT-genre',P_maxent),('POPULAR-concept',P_pconcept),
      ('UNCERTAINTY-item',P_uncert),('EIG-MIXED',P_eig)]
print(f"S4 Panel A (cheap) | ML-1M cold-test, {min(NU,len(te))} users, T={T}, W={W}",flush=True)
print(f"{'policy':<18} | NDCG@10 per q: "+" ".join(f"q{t}" for t in range(T+1))+" | Rec@10 qT",flush=True)
for name,pol in POLS:
    nd,rc=run(pol)
    print(f"{name:<18} | "+" ".join(f"{nd[t]:.3f}" for t in range(T+1))+f" | {rc[T]:.3f}",flush=True)
