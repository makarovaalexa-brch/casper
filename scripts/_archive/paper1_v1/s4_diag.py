"""
S4 DIAGNOSTIC — is the item-elicitation panel a fair test of SELECTION, or is it dominated by ANSWERABILITY?
Two controls, lean (ratings + Q only; no genome/SBERT):
 (A) ANSWERABILITY: for each static item selector, avg # of the T asked items that are actually in the user's
     profile (i.e. get folded in). If popularity gets many and entropy/random get ~0, the 'flat' curves are a
     no-op artifact, not bad selection.
 (B) PROFILE-RESTRICTED (fair selection): restrict each selector's candidate set to the user's OWN rated items, so
     EVERY asked item is answered. Now answerability is equalized and we test pure ordering quality the way Rashid'02
     /Golbandi'11 do. Same recommender (pop-floor + ridge fold-in + conf-blend) and same fixed split as s4_panel.
Reports NDCG@10 and Recall@10 at q0,q4,q8 for both modes + avg fold-ins.
"""
import os, numpy as np, torch, scipy.linalg as sla
base='C:/dev/phd/casper/data/movielens'; ml=f'{base}/ml-1m'; D=64; LAM=1.0
W=float(os.environ.get('W',2.0)); T=int(os.environ.get('T',8)); NU=int(os.environ.get('NU',120)); POOL=int(os.environ.get('POOL',120))
rng=np.random.default_rng(0)
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
cnt=np.zeros(ni); hist=np.zeros((ni,5))
for x in trU:
    for it,r in rated[x]:
        if r>=4: cnt[it]+=1
        hist[it,min(max(int(round(r)),1),5)-1]+=1
zpop=(np.log(cnt+1.)-np.log(cnt+1.).mean())/(np.log(cnt+1.).std()+1e-9)
pm=hist/np.clip(hist.sum(1,keepdims=True),1,None); ent=-(pm*np.log(pm+1e-12)).sum(1)
lf=np.log(cnt+1)/np.log(cnt.max()+1); Hn=ent/np.log(5); helf=2*lf*Hn/(lf+Hn+1e-9)
Q=torch.load(f'{base}/.cache/checkpoints/instrument_u_ml1m.pt')['Q'].numpy().astype(np.float32)
_cand=np.array(np.argsort(-cnt)[:400]); _,_,_piv=sla.qr(Q[_cand].T,pivoting=True); RMVA=[int(_cand[p]) for p in _piv]
rmva_rank=np.full(ni,1e9);
for r_,it in enumerate(RMVA): rmva_rank[it]=r_
SPL={}; _rs=np.random.default_rng(123)
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
KEYS={'random':None,'pop':cnt,'entropy':ent,'helf':helf,'rmva':-rmva_rank}  # higher=asked first (rmva via -rank)
def run(kind, profile_restricted):
    ND=np.zeros(T+1); RC=np.zeros(T+1); foldins=0.; m_=0
    for x in te[:NU]:
        if x not in SPL: continue
        recs=rated[x]; rd={it:(r-3)/2 for it,r in recs}; test=SPL[x]; profile=set(rd)-test; rel_=list(test)
        if len(rel_)<2: continue
        # candidate ordering
        if profile_restricted:
            cands=[it for it in profile]                       # only answerable items
            if kind=='random': rng.shuffle(cands); order=cands
            else: order=sorted(cands,key=lambda it:-KEYS[kind][it])
        else:
            if kind=='random': order=list(rng.permutation(ni))
            else: order=list(np.argsort(-KEYS[kind]))
        rows=[];y=[];asked=set(); a,b=ndrc(W*zpop,rel_,profile); ND[0]+=a; RC[0]+=b
        nf=0
        for t in range(T):
            e=next((it for it in order if it not in asked),None)
            if e is None: pass
            else:
                asked.add(e)
                if e in rd and e not in test: rows.append(Q[e]); y.append(rd[e]); nf+=1
            uu=foldin(np.array(rows),np.array(y)) if rows else np.zeros(D)
            scv=W*zpop+(len(rows)/(len(rows)+5.))*zc(Q@uu) if rows else W*zpop
            a,b=ndrc(scv,rel_,profile); ND[t+1]+=a; RC[t+1]+=b
        foldins+=nf; m_+=1
    return ND/m_,RC/m_,foldins/m_
print(f"ML-1M cold-test {min(NU,len(te))} users; T={T} W={W}\n",flush=True)
for mode,pr in [("GLOBAL pool (current panel)",False),("PROFILE-restricted (fair selection)",True)]:
    print(f"=== {mode} ===",flush=True)
    print(f"{'sel':<10} | NDCG q0/q4/q8 | Rec q8 | avg foldins@q8",flush=True)
    for kind in ['random','pop','entropy','helf','rmva']:
        nd,rc,fi=run(kind,pr)
        print(f"{kind:<10} | {nd[0]:.3f}/{nd[4]:.3f}/{nd[8]:.3f} | {rc[8]:.3f} | {fi:.2f}",flush=True)
    print(flush=True)
