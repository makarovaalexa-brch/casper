"""
PAPER B MOTIVATION: (A) ANSWERABILITY by question type + (B) CONCEPT INFO-VALUE, on the canonical calibrated instrument
(biased-SVD Q_svd/bi + popularity floor, NO z-score). Three question TYPES:
  items   : "do you like film j?"  answerable iff j in the user's profile.
  genres  : "do you like genre g?" answerable iff user has >=2 rated items in g (18 genres).
  concepts: "do you like <tag>?"   answerable iff user has >=2 rated items with genome relevance>THR (~fine tags).
(A) answer-rate = P(a randomly-asked question of this type is answerable) = (answerable in reach)/(pool size).
(B) info-value = ORACLE-greedy NDCG@10 (full+tail) vs #ANSWERED questions, per type (ceiling info each type carries).
Story: items carry info but are almost never answerable; genres/concepts are reliably answerable; concepts add fine
info genres can't -> the realizable value (info x answerability) favours the continuous concept action space (Paper B).
"""
import os, time, numpy as np
base='C:/dev/phd/casper/data/movielens'; ml=f'{base}/ml-1m'; D=64; LAM=5.0; THR=0.5; MINIT=30; NU=int(os.environ.get('NU',150)); rng=np.random.default_rng(0)
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
keep=[x for x in range(nu) if len(likes_by_u.get(x,[]))>=5]; rng.shuffle(keep); n=len(keep); trU=keep[:int(0.8*n)]; te=keep[int(0.9*n):]
cnt=np.zeros(ni)
for x in trU:
    for j in likes_by_u.get(x,[]): cnt[j]+=1
popb=np.log(cnt+1.0).astype(np.float32); sm=c=0.
for x in trU:
    for j,r in rat_by_u[x]: sm+=r;c+=1
mu=sm/c
Q=np.load(f'{base}/.cache/Q_svd.npy'); bi=np.load(f'{base}/.cache/bi_svd.npy'); qn=np.linalg.norm(Q,axis=1).mean()
order_pop=np.argsort(-cnt); cum=np.cumsum(cnt[order_pop])/cnt.sum(); HEAD=set(int(j) for j in order_pop[:np.searchsorted(cum,0.33)+1]); headmask=np.zeros(ni,bool); headmask[list(HEAD)]=True
resid_by_u={x:{j:(r-mu-bi[j]) for j,r in rat_by_u[x]} for x in te}
# genres
GEN=['Action','Adventure','Animation',"Children's",'Comedy','Crime','Documentary','Drama','Fantasy','Film-Noir','Horror','Musical','Mystery','Romance','Sci-Fi','Thriller','War','Western']
gid={g:k for k,g in enumerate(GEN)}; na=len(GEN); item_g=np.zeros((ni,na),bool)
with open(f'{ml}/movies.dat',encoding='latin-1') as f:
    for line in f:
        p=line.strip().split('::'); m=int(p[0])
        if m in iids:
            for g in p[2].split('|'):
                if g in gid: item_g[iids[m],gid[g]]=True
Ag=np.stack([Q[np.where(item_g[:,g])[0]].mean(0)/(np.linalg.norm(Q[np.where(item_g[:,g])[0]].mean(0))+1e-9)*qn for g in range(na)]).astype(np.float32)
gitems=[set(np.where(item_g[:,g])[0].tolist()) for g in range(na)]
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
Ac=np.stack([Q[np.array(tagitems[t])].mean(0)/(np.linalg.norm(Q[np.array(tagitems[t])].mean(0))+1e-9)*qn for t in ctags]).astype(np.float32)
citems=[set(tagitems[t]) for t in ctags]; ncon=len(ctags)
print(f"  {ncon} genome concepts ({time.time()-t0:.0f}s)",flush=True)
_W=1./np.log2(np.arange(2,12))
def ridge(rows,y): rows=np.array(rows); return np.linalg.solve(rows.T@rows+LAM*np.eye(D),rows.T@np.array(y,np.float32)) if len(rows) else np.zeros(D)
def ndcg(u,rel,excl,tail):
    sc=(popb+Q@u).copy(); sc[list(excl)]=-1e9
    if tail: sc[headmask]=-1e9; rel=[t for t in rel if not headmask[t]]
    if not rel: return None
    top=np.argpartition(-sc,10)[:10]; top=top[np.argsort(-sc[top])]; rs=set(rel)
    return sum(_W[p] for p,t in enumerate(top) if int(t) in rs)/(_W[:min(10,len(rel))].sum()+1e-12)
_rs=np.random.default_rng(123); SPL={}
for x in te:
    its=list(dict(rat_by_u[x]))
    if len(its)>=6: il=its[:]; _rs.shuffle(il); SPL[x]=(il[:len(il)//2], il[len(il)//2:])
TE=[x for x in te if x in SPL][:NU]
# ---------- (A) ANSWERABILITY ----------
ai={'items':[],'genres':[],'concepts':[]}; ar={'items':[],'genres':[],'concepts':[]}
for x in TE:
    test,prof=SPL[x]; profset=set(prof)
    nit=len(prof); ng=sum(1 for g in range(na) if len(profset&gitems[g])>=2); nc=sum(1 for k in range(ncon) if len(profset&citems[k])>=2)
    ai['items'].append(nit); ai['genres'].append(ng); ai['concepts'].append(nc)
    ar['items'].append(nit/ni); ar['genres'].append(ng/na); ar['concepts'].append(nc/ncon)
print("\n=== (A) ANSWERABILITY (avg over %d users) ==="%len(TE),flush=True)
print(f"{'type':>9} | {'pool size':>9} | {'answerable in reach':>19} | {'answer-rate if asked at random':>30}",flush=True)
for t,pool in [('items',ni),('genres',na),('concepts',ncon)]:
    print(f"{t:>9} | {pool:>9} | {np.mean(ai[t]):>19.1f} | {100*np.mean(ar[t]):>29.1f}%",flush=True)
# ---------- (B) CONCEPT INFO-VALUE (oracle-greedy per ANSWERED question) ----------
def afty(items_in):  # graded affinity = mean residual over the user's items in the genre/concept
    return float(np.mean([resid_by_u_x[j] for j in items_in]))
def interview(x,mode,tail,T=8):
    test,prof=SPL[x]; rd=dict(rat_by_u[x]); profset=set(prof); rel=[j for j in test if rd[j]>=4]
    if not rel or (tail and not any(not headmask[t] for t in rel)): return None
    global resid_by_u_x; resid_by_u_x=resid_by_u[x]
    # candidate ANSWERABLE questions of this type, as (factor,value)
    cand=[]
    if mode=='items':
        for j in prof: cand.append((Q[j], rd[j]-mu-bi[j], j))
    if mode=='genres':
        for g in range(na):
            inb=[j for j in prof if item_g[j,g]]
            if len(inb)>=2: cand.append((Ag[g], afty(inb), ('g',g)))
    if mode=='concepts':
        for k in range(ncon):
            inb=[j for j in prof if j in citems[k]]
            if len(inb)>=2: cand.append((Ac[k], afty(inb), ('c',k)))
    rows=[];yv=[];used=set();excl=set(profset); curve=[ndcg(np.zeros(D),rel,excl,tail)]
    for t in range(T):
        best=None
        for idx,(f,v,key) in enumerate(cand):
            if idx in used: continue
            g=ndcg(ridge(rows+[f],yv+[v]),rel,excl,tail)
            if g is not None and (best is None or g>best[0]): best=(g,idx)
        if best is None: break
        used.add(best[1]); f,v,key=cand[best[1]]; rows.append(f);yv.append(v)
        curve.append(best[0])
    while len(curve)<T+1: curve.append(curve[-1])
    return np.array(curve)
for tail in [False,True]:
    print(f"\n=== (B) INFO-VALUE: oracle-greedy NDCG@10 vs #ANSWERED ({'TAIL' if tail else 'FULL'}) ===",flush=True)
    res={}
    for mode in ['items','genres','concepts']:
        cs=[interview(x,mode,tail) for x in TE]; cs=[c for c in cs if c is not None]; res[mode]=np.mean(cs,0)
    print(f"{'#q':>3} {'ITEMS':>8} {'GENRES':>8} {'CONCEPTS':>9}",flush=True)
    for t in [0,1,2,4,8]: print(f"{t:>3} {res['items'][t]:>8.3f} {res['genres'][t]:>8.3f} {res['concepts'][t]:>9.3f}",flush=True)
