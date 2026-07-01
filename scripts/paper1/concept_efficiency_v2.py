"""
Are FINE genome-style CONCEPTS (not just 18 genres) worth a lot for some users?
Per-user ORACLE interview on the instrument with four question types:
  ITEMS | GENRES(18) | GENOME-CONCEPTS (fine Tag-Genome themes) | MIXED(all).
Concept factor = normalized centroid of Q over items with genome relevance>thr (so it lives in taste space).
User 'likes' a concept if >=2 of their liked items are strongly about it. Reports per-turn NDCG curves + the
fraction of users for whom genome-concepts beat movies, with the winning concept named for illustrative users.
"""
import os, time, numpy as np, torch
base='C:/dev/phd/casper/data/movielens'; ml=f'{base}/ml-1m'; D=64; LAM=1.0; W=4.0
T=int(os.environ.get('T',4)); NU=int(os.environ.get('NU',120)); THR=0.5; MINIT=30; rng=np.random.default_rng(0)
I=[];R=[];Uu=[]
with open(f'{ml}/ratings.dat') as f:
    for line in f:
        a=line.strip().split('::'); Uu.append(int(a[0])); I.append(int(a[1])); R.append(float(a[2]))
I=np.array(I);R=np.array(R,np.float32);Uu=np.array(Uu)
uids={x:k for k,x in enumerate(np.unique(Uu))}; movieid=np.unique(I); iids={x:k for k,x in enumerate(movieid)}; ni=len(iids); nu=len(uids)
u=np.array([uids[x] for x in Uu]); ic=np.array([iids[x] for x in I]); keepids=set(iids)
rated={}
for k in range(len(u)): rated.setdefault(u[k],[]).append((ic[k],(R[k]-3.)/2.,1.0 if R[k]>=4 else 0.0))
keep=[x for x in range(nu) if sum(t[2] for t in rated[x])>=8]; rng.shuffle(keep); te=keep[int(0.9*len(keep)):]
cnt=np.zeros(ni)
for x in keep[:int(0.8*len(keep))]:
    for it,_,lk in rated[x]:
        if lk>0: cnt[it]+=1
popb=np.log(cnt+1.0).astype(np.float32); zpop=(popb-popb.mean())/(popb.std()+1e-9)
Q=torch.load(f'{base}/.cache/checkpoints/instrument_u_ml1m.pt')['Q'].numpy().astype(np.float32); qn=np.linalg.norm(Q,axis=1).mean()
# genres
GEN=['Action','Adventure','Animation',"Children's",'Comedy','Crime','Documentary','Drama','Fantasy','Film-Noir','Horror','Musical','Mystery','Romance','Sci-Fi','Thriller','War','Western']
gid={g:k for k,g in enumerate(GEN)}; na=len(GEN); item_g=np.zeros((ni,na),bool)
with open(f'{ml}/movies.dat',encoding='latin-1') as f:
    for line in f:
        p=line.strip().split('::'); m=int(p[0])
        if m in iids:
            for g in p[2].split('|'):
                if g in gid: item_g[iids[m],gid[g]]=True
Ag=np.zeros((na,D),np.float32)
for g in range(na):
    s=np.where(item_g[:,g])[0]; Ag[g]=Q[s].mean(0)/(np.linalg.norm(Q[s].mean(0))+1e-9)*qn if len(s) else 0
# genome concepts
tagname={}
with open(f'{base}/genome-tags.csv',encoding='latin-1') as f:
    next(f)
    for line in f: a=line.rstrip('\n').split(','); tagname[int(a[0])]=a[1]
print("streaming genome...",flush=True); t0=time.time(); tagitems={}
with open(f'{base}/genome-scores.csv') as f:
    next(f)
    for line in f:
        a=line.split(','); m=int(a[0])
        if m in keepids and float(a[2])>THR: tagitems.setdefault(int(a[1]),[]).append(iids[m])
ctags=[t for t,its in tagitems.items() if len(its)>=MINIT and tagname[t] not in set(g.lower() for g in GEN)]
Ac=np.zeros((len(ctags),D),np.float32); cit=[]
for k,t in enumerate(ctags):
    s=np.array(tagitems[t]); Ac[k]=Q[s].mean(0)/(np.linalg.norm(Q[s].mean(0))+1e-9)*qn; cit.append(set(s.tolist()))
print(f"  {len(ctags)} genome concepts ({time.time()-t0:.0f}s)",flush=True)
def foldin(rows,y): return np.linalg.solve(rows.T@rows+LAM*np.eye(D), rows.T@y) if len(rows) else np.zeros(D)
def ndcg(u_,rel,excl,m):
    sc=(W*zpop).copy()
    if m>0: z=Q@u_; z=(z-z.mean())/(z.std()+1e-9); sc=sc+(m/(m+5.))*z
    sc=sc.copy(); sc[list(excl)]=-1e9; top=np.argsort(-sc)[:10]; rs=set(rel)
    idcg=sum(1./np.log2(p+2) for p in range(min(10,len(rel))))
    return sum(1./np.log2(p+2) for p,t in enumerate(top) if t in rs)/idcg if idcg else 0.
def oracle(user, mode):
    recs=rated[user]; rd={it:cr for it,cr,lk in recs}; likes=[it for it,cr,lk in recs if lk>0]; rng.shuffle(likes)
    test=set(likes[:len(likes)//2]); lset=set(likes)
    icand=[it for it,cr,lk in recs if it not in test]
    lg=set(g for g in range(na) if sum(item_g[it,g] for it in likes)>=2)
    lc=set(k for k in range(len(ctags)) if len(lset & cit[k])>=2)
    rows=[];yv=[];used=set();excl=set();best_name=None;curve=[ndcg(np.zeros(D),list(test),excl,0)]
    for t in range(T):
        best=None;pick=None
        if mode in('item','mixed'):
            for it in icand:
                if ('i',it) in used: continue
                g=ndcg(foldin(np.array(rows+[Q[it]]),np.array(yv+[rd[it]])),list(test),excl|{it},len(rows)+1)
                if best is None or g>best: best=g;pick=('i',it)
        if mode in('genre','mixed'):
            for gg in range(na):
                if ('g',gg) in used: continue
                y=1. if gg in lg else -1.
                g=ndcg(foldin(np.array(rows+[Ag[gg]]),np.array(yv+[y])),list(test),excl,len(rows)+1)
                if best is None or g>best: best=g;pick=('g',gg)
        if mode in('concept','mixed'):
            for cc in range(len(ctags)):
                if ('c',cc) in used: continue
                y=1. if cc in lc else -1.
                g=ndcg(foldin(np.array(rows+[Ac[cc]]),np.array(yv+[y])),list(test),excl,len(rows)+1)
                if best is None or g>best: best=g;pick=('c',cc)
        if pick is None: break
        used.add(pick); k,idx=pick
        if k=='i': rows.append(Q[idx]);yv.append(rd[idx]);excl.add(idx)
        elif k=='g': rows.append(Ag[idx]);yv.append(1. if idx in lg else -1.)
        else: rows.append(Ac[idx]);yv.append(1. if idx in lc else -1.); best_name=tagname[ctags[idx]] if best_name is None else best_name
        curve.append(best)
    return np.array(curve+[curve[-1]]*(T+1-len(curve))), best_name
users=te[:NU]; cur={m:[] for m in('item','genre','concept','mixed')}; names={}
for n,x in enumerate(users):
    for m in cur:
        c,nm=oracle(x,m); cur[m].append(c)
        if m=='concept': names[x]=nm
    if (n+1)%40==0: print(f"  {n+1}/{len(users)}",flush=True)
C={m:np.array(cur[m]) for m in cur}
print("\n=== ORACLE efficiency: NDCG@10 vs #questions (avg over %d users) ==="%len(users),flush=True)
print(f"{'#q':>3} {'ITEMS':>8} {'GENRES':>8} {'GENOME':>8} {'MIXED':>8}",flush=True)
for t in range(T+1): print(f"{t:>3} {C['item'][:,t].mean():>8.4f} {C['genre'][:,t].mean():>8.4f} {C['concept'][:,t].mean():>8.4f} {C['mixed'][:,t].mean():>8.4f}",flush=True)
for t in [1,T]:
    wc=(C['concept'][:,t]>C['item'][:,t]+1e-4); wg=(C['genre'][:,t]>C['item'][:,t]+1e-4)
    print(f"\n@ {t} q: GENOME-concepts>items {100*wc.mean():.0f}% (vs genres>items {100*wg.mean():.0f}%); "
          f"mixed>items {100*(C['mixed'][:,t]>C['item'][:,t]+1e-4).mean():.0f}%; mean gap when concept wins "
          f"{(C['concept'][:,t]-C['item'][:,t])[wc].mean() if wc.any() else float('nan'):.3f}",flush=True)
d=C['concept'][:,T]-C['item'][:,T]; topu=np.argsort(-d)[:6]
print("\nillustrative users (genome-concept vs items @%d q; winning concept):"%T,flush=True)
for j in topu: print(f"  user {users[j]}: concept {C['concept'][j,T]:.3f} vs items {C['item'][j,T]:.3f}  [{names.get(users[j])}]",flush=True)
