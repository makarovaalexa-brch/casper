"""
S4 Panel A (CORRECTED) — cheap/no-train elicitation baselines on the instrument, leakage-free, with the corrected
attribute mechanism. Items -> ridge fold-in u_item. Attributes/concepts -> EAR-style ADDITIVE channel + BOUNDED
weight, graded base-rate-normalized answer, affinity selection, profile-only (no leakage).
Score = W*z(pop) + conf_item*z(Q.u_item) + WA*z(Q.attr_pref).
Item policies: RANDOM, POPULARITY, ENTROPY(Rashid'02), HELF(Rashid'08), EIG-item(greedy info-gain).
Attribute/concept: GENRES(lift), GENOME-CONCEPTS(relwtd), OOS-CONCEPTS(SBERT, out-of-vocab).
Reports NDCG@10 / Recall@10 at q1,q3,q5.
"""
import os, numpy as np, torch, scipy.linalg as sla
from sentence_transformers import SentenceTransformer
base='C:/dev/phd/casper/data/movielens'; ml=f'{base}/ml-1m'; D=64; LAM=1.0
W=float(os.environ.get('W',2.0)); WA=float(os.environ.get('WA',0.4)); T=int(os.environ.get('T',6))
NU=int(os.environ.get('NU',120)); POOL=int(os.environ.get('POOL',120)); SUP=5; SIMTHR=0.30; rng=np.random.default_rng(0)
I=[];R=[];Uu=[]
with open(f'{ml}/ratings.dat') as f:
    for line in f:
        a=line.strip().split('::'); Uu.append(int(a[0])); I.append(int(a[1])); R.append(float(a[2]))
I=np.array(I);R=np.array(R,np.float32);Uu=np.array(Uu)
uids={x:k for k,x in enumerate(np.unique(Uu))}; iids={x:k for k,x in enumerate(np.unique(I))}; ni=len(iids); nu=len(uids)
u=np.array([uids[x] for x in Uu]); ic=np.array([iids[x] for x in I])
rated={}
for k in range(len(u)): rated.setdefault(u[k],[]).append((ic[k],R[k]))
keep=[x for x in range(nu) if sum(1 for it,r in rated[x] if r>=4)>=5]; rng.shuffle(keep)  # canonical filter (mf_foldin)
trU=keep[:int(0.8*len(keep))]; va=keep[int(0.8*len(keep)):int(0.9*len(keep))]; te=keep[int(0.9*len(keep)):]
cnt=np.zeros(ni); hist=np.zeros((ni,5))
for x in trU:
    for it,r in rated[x]:
        if r>=4: cnt[it]+=1
        hist[it,min(max(int(round(r)),1),5)-1]+=1
zpop=(np.log(cnt+1.)-np.log(cnt+1.).mean())/(np.log(cnt+1.).std()+1e-9)
pm=hist/np.clip(hist.sum(1,keepdims=True),1,None); ent=-(pm*np.log(pm+1e-12)).sum(1)
lf=np.log(cnt+1)/np.log(cnt.max()+1); Hn=ent/np.log(5); helf=2*lf*Hn/(lf+Hn+1e-9)
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
# genome concepts (in-catalogue) + OOS concepts (SBERT)
THR=0.5; tagname={}
with open(f'{base}/genome-tags.csv',encoding='latin-1') as f:
    next(f)
    for line in f: a=line.rstrip('\n').split(','); tagname[int(a[0])]=a[1]
tcount={}; topth={}
with open(f'{base}/genome-scores.csv') as f:
    next(f)
    for line in f:
        a=line.split(','); m=int(a[0])
        if m in iids:
            if float(a[2])>THR: tcount[int(a[1])]=tcount.get(int(a[1]),0)+1
            topth.setdefault(m,[]).append((float(a[2]),int(a[1])))
glow=set(g.lower() for g in GEN); NCg=150
ctags=[t for t,_ in sorted(tcount.items(),key=lambda kv:-kv[1]) if tagname[t] not in glow][:NCg]; cidx={t:k for k,t in enumerate(ctags)}
relg=np.zeros((ni,NCg),np.float32)
with open(f'{base}/genome-scores.csv') as f:
    next(f)
    for line in f:
        a=line.split(','); m=int(a[0]); t=int(a[1])
        if m in iids and t in cidx: relg[iids[m],cidx[t]]=float(a[2])
Acg=np.zeros((NCg,D),np.float32)
for c in range(NCg):
    s=np.where(relg[:,c]>THR)[0]; Acg[c]=Q[s].mean(0)/(np.linalg.norm(Q[s].mean(0))+1e-9)*qn if len(s) else 0
themes={iids[m]:", ".join(tagname[t] for r,t in sorted(v,reverse=True)[:12]) for m,v in topth.items()}
sb=SentenceTransformer('all-MiniLM-L6-v2')
ME=sb.encode([f"{title.get(k,'?')}. Themes: {themes.get(k,'')}." for k in range(ni)],batch_size=128,normalize_embeddings=True,show_progress_bar=False).astype(np.float32)
OOS=["mind-bending plot twist","feel-good underdog story","slow-burn character study","based on a true story",
 "dark and morally ambiguous","quirky offbeat humour","epic historical sweep","tense psychological cat and mouse",
 "heartwarming family bonds","dystopian surveillance state","hard-boiled noir mystery","satirical social commentary",
 "tearjerker doomed romance","gritty urban crime saga","whimsical fantasy adventure","cerebral hard science fiction"]
CE=sb.encode(OOS,normalize_embeddings=True).astype(np.float32); NCo=len(OOS); simo=ME@CE.T
Aco=np.zeros((NCo,D),np.float32)
for c in range(NCo):
    w=np.where(simo[:,c]>SIMTHR,np.clip(simo[:,c],0,None),0); Aco[c]=(w[:,None]*Q).sum(0)/(w.sum()+1e-9); Aco[c]=Aco[c]/(np.linalg.norm(Aco[c])+1e-9)*qn
pool=list(np.argsort(-cnt)[:POOL])
_cand=np.array(np.argsort(-cnt)[:400]); _,_,_piv=sla.qr(Q[_cand].T,pivoting=True); RMVA=[int(_cand[p]) for p in _piv]  # representative items
print(f"loaded. ML-1M cold-test {min(NU,len(te))} users; W={W} WA={WA} T={T}",flush=True)
SPL={}; _rs=np.random.default_rng(123)                # FIXED held-out split per user, shared across ALL policies
for x in te:
    lk=[it for it,r in rated[x] if r>=4]
    if len(lk)>=4: ll=lk[:]; _rs.shuffle(ll); SPL[x]=set(ll[len(ll)//2:])
def foldin(X,y): return np.linalg.solve(X.T@X+LAM*np.eye(D), X.T@y) if len(X) else np.zeros(D)
def zc(v): return (v-v.mean())/(v.std()+1e-9)
def ndrc(scv,rel_,excl):
    if not len(rel_): return 0.,0.
    s=scv.copy(); s[list(excl)]=-1e9; top=np.argsort(-s)[:10]; rs=set(int(t) for t in rel_)
    idcg=sum(1./np.log2(p+2) for p in range(min(10,len(rel_))))
    return (sum(1./np.log2(p+2) for p,t in enumerate(top) if int(t) in rs)/idcg if idcg else 0.,
            len(set(int(t) for t in top)&rs)/len(rel_))
def split(x):
    recs=rated[x]; allit=np.array([it for it,r in recs]); allr=np.array([r for it,r in recs],np.float32)
    test=SPL[x]; pmask=np.array([int(i) not in test for i in allit]); return allit,allr,test,allit[pmask],allr[pmask]
# ---------- ITEM policies (fold-in) ----------
def run_item(kind, W=W, cap=1.0, users=None):
    # CANONICAL P1 + VALIDATED fold-in (instrument_u line 97 / mf_foldin line 59): fold EVERY asked seed with the
    # ANSWER y=1 if the user liked it else 0 (NOT skip unrated). The "0" answers carry the discriminative signal that
    # makes representative-item elicitation work. conf = n_liked/(n_liked+5). q0 == MOSTPOP ~0.41; no leakage.
    if users is None: users=te
    ND=np.zeros(T+1); RC=np.zeros(T+1); m_=0
    for x in users[:NU]:
        recs=rated[x]; likeset=set(it for it,r in recs if r>=4); likes=list(likeset)
        if len(likes)<2: continue
        rows=[];y=[];asked=set()
        if kind=='random': order=list(rng.permutation(ni))
        elif kind=='pop': order=list(np.argsort(-cnt))
        elif kind=='entropy': order=list(np.argsort(-ent))
        elif kind=='helf': order=list(np.argsort(-helf))
        elif kind=='rmva': order=RMVA
        for t in range(T+1):
            if t>0:
                if kind=='eig':
                    nL=sum(y); uu=foldin(np.array(rows),np.array(y)) if rows else np.zeros(D); base=W*zpop+ (nL/(nL+5.))*zc(Q@uu) if nL>0 else W*zpop
                    bt=set(np.argsort(-base)[:10]); z=Q@uu if rows else np.zeros(ni); pl=1/(1+np.exp(-2*(z-np.median(z)))); best=None;e=pool[0]
                    for it in pool[:40]:
                        if it in asked: continue
                        ul=zc(Q@foldin(np.array(rows+[Q[it]]),np.array(y+[1.0]))); ud=zc(Q@foldin(np.array(rows+[Q[it]]),np.array(y+[0.0])))
                        ch=pl[it]*len(set(np.argsort(-ul)[:10])^bt)+(1-pl[it])*len(set(np.argsort(-ud)[:10])^bt)
                        if best is None or ch>best: best=ch;e=it
                else:
                    e=next((it for it in order if it not in asked),None)
                if e is not None:
                    asked.add(e)
                    rows.append(Q[e]); y.append(1.0 if e in likeset else 0.0)   # fold in the ANSWER (1 liked / 0 not-liked-or-unseen)
            rel_=[j for j in likes if j not in asked]
            nL=sum(y)
            uu=foldin(np.array(rows),np.array(y)) if rows else np.zeros(D)
            scv=W*zpop+ (nL/(nL+5.))*zc(Q@uu) if nL>0 else W*zpop
            a,b=ndrc(scv,rel_,asked); ND[t]+=a; RC[t]+=b
        m_+=1
    return ND/m_,RC/m_
# ---------- ATTRIBUTE/CONCEPT policies (additive + bounded weight) ----------
def run_attr(kind):
    A = Ag if kind=='genre' else (Acg if kind=='genome' else Aco); ND=np.zeros(T+1); RC=np.zeros(T+1); m_=0
    for x in te[:NU]:
        allit,allr,test,pit,pr=split(x); profile=set(int(i) for i in pit); rel_=list(test); umean=pr.mean()
        if len(rel_)<2: continue
        ans={}
        if kind=='genre':
            for g in range(na):
                pu=item_g[pit,g].mean() if len(pit) else 0.; ans[g]=float(np.tanh(np.log((pu+1e-3)/(Pg[g]+1e-3)+1e-6)))
        elif kind=='genome':
            rp=relg[pit]
            for c in range(NCg):
                w=rp[:,c];
                if (w>THR).sum()<SUP: ans[c]=None
                else: ans[c]=float(np.clip((w*(pr-umean)).sum()/(w.sum()+1e-6)/1.5,-1,1))
        else:
            sp=simo[pit]
            for c in range(NCo):
                w=np.clip(sp[:,c],0,None)
                if (w>SIMTHR).sum()<SUP: ans[c]=None
                else: ans[c]=float(np.clip((w*(pr-umean)).sum()/(w.sum()+1e-6)/1.5,-1,1))
        order=sorted([e for e in ans if ans[e] is not None],key=lambda e:-abs(ans[e]))
        pref=np.zeros(D); a,b=ndrc(W*zpop,rel_,profile); ND[0]+=a; RC[0]+=b
        for t in range(T):
            if t<len(order):
                e=order[t]; pref=pref+ans[e]*A[e]
            scv=W*zpop+WA*zc(Q@pref) if np.any(pref) else W*zpop
            a,b=ndrc(scv,rel_,profile); ND[t+1]+=a; RC[t+1]+=b
        m_+=1
    return ND/m_,RC/m_
def run_item_ho(kind, W=W, cap=1.0, users=None):
    # UNIFIED HELD-OUT (same protocol as attributes): FIXED disjoint test targets (SPL); askable=global pool; answer
    # 1 if the item is a PROFILE like else 0 (leakage-free); exclude profile+asked. Targets never removed by asking,
    # so asking cannot 'saw off the branch' -> popularity goes FLAT not down. q0 = MOSTPOP-on-heldout (~0.31) =
    # SAME prior as attributes -> items & attributes directly comparable.
    if users is None: users=te
    ND=np.zeros(T+1); RC=np.zeros(T+1); m_=0
    for x in users[:NU]:
        if x not in SPL: continue
        recs=rated[x]; test=SPL[x]; profile=set(it for it,r in recs)-test
        plike=set(it for it,r in recs if r>=4 and it not in test); rel_=list(test)
        if len(rel_)<2: continue
        rows=[];y=[];asked=set()
        if kind=='random': order=list(rng.permutation(ni))
        elif kind=='pop': order=list(np.argsort(-cnt))
        elif kind=='entropy': order=list(np.argsort(-ent))
        elif kind=='helf': order=list(np.argsort(-helf))
        elif kind=='rmva': order=RMVA
        for t in range(T+1):
            if t>0:
                e=next((it for it in order if it not in asked),None)
                if e is not None:
                    asked.add(e); rows.append(Q[e]); y.append(1.0 if e in plike else 0.0)
            nL=sum(y); uu=foldin(np.array(rows),np.array(y)) if rows else np.zeros(D)
            scv=W*zpop+(nL/(nL+5.))*zc(Q@uu) if nL>0 else W*zpop
            a,b=ndrc(scv,rel_,profile|asked); ND[t]+=a; RC[t]+=b
        m_+=1
    return ND/m_,RC/m_
WGRID=[0.5,1,2,4,8,16,32]   # widened so the tuner can reach the pop-dominant blend
print("\n=== ITEM elicitation — CANONICAL P1 (rel=all likes-seeds, exclude seeds); q0=MOSTPOP anchor ~0.41 ===",flush=True)
print("    blend (W, conf-cap) tuned on VAL per policy, reported on TEST. q0 line = popularity baseline (no questions).",flush=True)
print(f"\n{'policy':<16} | (W,cap) | NDCG@10  q0 .... q{T}  | Rec@10 qT | vs q0",flush=True)
for name,kind,tune in [('RANDOM-item','random',False),('POPULARITY-item','pop',True),('ENTROPY-item','entropy',True),
                       ('HELF-item','helf',True),('RMVA-item','rmva',True),('EIG-item','eig',False)]:
    if tune:
        bestp=None
        for Wv in WGRID:
            vnd,_=run_item(kind,Wv,1.0,va)
            obj=float(np.mean(vnd))                          # tune on the WHOLE curve (robust), not the noisy endpoint
            if bestp is None or obj>bestp[0]: bestp=(obj,Wv,1.0)
        Wv,cp=bestp[1],bestp[2]
    else:
        Wv,cp=(2,1.0) if kind=='random' else (4,1.0)        # EIG: fixed W (full sweep too slow); random: untuned
    nd,rc=run_item(kind,Wv,cp,te)
    d=nd[T]-nd[0]; tag=f"+{d:.3f} BEATS" if d>0.005 else (f"{d:.3f} below" if d<-0.005 else "~flat")
    print(f"{name:<16} | ({Wv},{cp}) | "+" ".join(f"{v:.3f}" for v in nd)+f" | {rc[T]:.3f} | {tag}",flush=True)
print("\n=== ATTRIBUTE/CONCEPT elicitation — leakage-free held-out P2 (rel=held-out likes, exclude profile); q0 anchor ~0.32 ===",flush=True)
print("    (attributes MUST use held-out: an answer from the profile would otherwise score its own targets)",flush=True)
print(f"\n{'policy':<16} | NDCG@10  q0 .... q{T}  | Rec@10 qT | vs q0",flush=True)
for name,arg in [('GENRES','genre'),('GENOME-CONCEPTS','genome'),('OOS-CONCEPTS','oos')]:
    nd,rc=run_attr(arg); d=nd[T]-nd[0]; tag=f"+{d:.3f} BEATS" if d>0.005 else (f"{d:.3f} below" if d<-0.005 else "~flat")
    print(f"{name:<16} | "+" ".join(f"{v:.3f}" for v in nd)+f" | {rc[T]:.3f} | {tag}",flush=True)

print("\n=== UNIFIED HELD-OUT (items AND attributes, ONE protocol, fixed disjoint targets); q0 anchor ~0.31 ===",flush=True)
print("    targets never removed by asking -> no exclusion artifact; items & attributes on the SAME prior.",flush=True)
print(f"\n{'policy':<16} | (W) | NDCG@10  q0 .... q{T}  | Rec@10 qT | vs q0",flush=True)
for name,kind in [('RANDOM-item','random'),('POPULARITY-item','pop'),('ENTROPY-item','entropy'),('HELF-item','helf'),('RMVA-item','rmva')]:
    if kind=='random': Wv=2
    else:
        bestp=None
        for wv in WGRID:
            vnd,_=run_item_ho(kind,wv,1.0,va); obj=float(np.mean(vnd))
            if bestp is None or obj>bestp[0]: bestp=(obj,wv)
        Wv=bestp[1]
    nd,rc=run_item_ho(kind,Wv,1.0,te); d=nd[T]-nd[0]; tag=f"+{d:.3f} BEATS" if d>0.005 else (f"{d:.3f} below" if d<-0.005 else "~flat")
    print(f"{name:<16} | ({Wv}) | "+" ".join(f"{v:.3f}" for v in nd)+f" | {rc[T]:.3f} | {tag}",flush=True)
for name,arg in [('GENRES-attr','genre'),('GENOME-attr','genome'),('OOS-attr','oos')]:
    nd,rc=run_attr(arg); d=nd[T]-nd[0]; tag=f"+{d:.3f} BEATS" if d>0.005 else (f"{d:.3f} below" if d<-0.005 else "~flat")
    print(f"{name:<16} | (-) | "+" ".join(f"{v:.3f}" for v in nd)+f" | {rc[T]:.3f} | {tag}",flush=True)
