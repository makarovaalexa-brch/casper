"""
VARIANT A (explicit track): EASE (Steck 2019 closed-form linear autoencoder) on the Goodreads
COMPOSITE explicit signal, restricted to the top-N-by-count item universe. Same held-out disjoint
targets protocol as the gate. Fair triangle: MOSTPOP, ENCODER (enc_v1_grcomp), EASE all evaluated on
the SAME restricted universe / same users / same targets. lambda sweep {1,10,100,500} selected on the
500-user val cohort; test numbers reported at the val-selected lambda.
No training of the encoder, no commits.
"""
import os, time, numpy as np, torch, torch.nn as nn, scipy.sparse as sp
t0=time.time(); GR='C:/dev/phd/casper/.cache/goodreads'
torch.set_num_threads(os.cpu_count() or 4); D=64; LIKE=4.0; LAM_R=5.0
N=int(os.environ.get('NUNIV',20000)); KP=int(os.environ.get('KP',510)); Ks=[10,KP]
LAMS=[1.0,10.0,100.0,500.0]
B=np.load(f'{GR}/base_comp.npz')
uu=B['uu']; ii=B['ii']; rr=B['rr']; cnt=B['cnt']; mu=float(B['mu']); ni=int(B['ni']); nu=int(B['nu'])
trU=B['trU']; va=B['va']; te=B['te']
Q=np.load(f'{GR}/Q_svd_comp.npy').astype(np.float32); bi=np.load(f'{GR}/bi_svd_comp.npy').astype(np.float32)
popb=np.log(cnt+1.0).astype(np.float32)
# ---- restricted universe: top-N items by like-count cnt ----
univ=np.argsort(-cnt)[:N]; univ=np.sort(univ)
umask=np.zeros(ni,bool); umask[univ]=True
loc=-np.ones(ni,np.int64); loc[univ]=np.arange(N)     # dense item -> local universe idx
print(f"[universe] N={N} of {ni} items; like-count share kept={cnt[univ].sum()/cnt.sum()*100:.1f}%",flush=True)
# ---- rating dicts for eval users (val+te) + SPL held-out targets (rng123) ----
evalset=np.array(sorted(set(va.tolist())|set(te.tolist()))); selm=np.isin(uu,evalset)
euu=uu[selm]; eii=ii[selm]; eR=rr[selm]; rat_by_u={}
for k in range(len(euu)): rat_by_u.setdefault(int(euu[k]),[]).append((int(eii[k]),float(eR[k])))
_rs=np.random.default_rng(123); SPL={}
for x in (va.tolist()+te.tolist()):
    lk=[j for j,r in rat_by_u.get(x,[]) if r>=LIKE]
    if len(lk)>=4: ll=lk[:]; _rs.shuffle(ll); SPL[x]=set(ll[len(ll)//2:])
print(f"eval dicts built ({time.time()-t0:.0f}s)",flush=True)
# ---- EASE train matrix: train users x universe items, binarized like (rating>=4) ----
trset=np.zeros(nu,bool); trset[trU]=True
rowmask=(rr>=LIKE)&umask[ii]&trset[uu]
Xr=uu[rowmask]; Xc=loc[ii[rowmask]]
X=sp.csr_matrix((np.ones(len(Xr),np.float32),(Xr,Xc)),shape=(nu,N))
nnz=X.nnz; ntrain_u=np.asarray((X.sum(1)>0)).sum()
print(f"[EASE X] {nnz} like-nnz over {ntrain_u} train users x {N} items ({time.time()-t0:.0f}s)",flush=True)
def ease_B(lam):
    G=np.asarray((X.T@X).todense(),dtype=np.float64)
    G[np.diag_indices_from(G)]+=lam
    P=np.linalg.inv(G); d=np.diag(P).copy()
    Bm=-P/d[None,:]; Bm[np.diag_indices_from(Bm)]=0.0
    del G,P
    return Bm.astype(np.float32)
# ---- encoder ----
class Enc(nn.Module):
    def __init__(s):
        super().__init__(); s.inp=nn.Sequential(nn.Linear(D+1,128),nn.ReLU(),nn.Linear(128,128),nn.ReLU())
        s.att=nn.Linear(128,1); s.val=nn.Linear(128,D)
    def forward(s,toks,mask):
        h=s.inp(toks); a=s.att(h).squeeze(-1).masked_fill(mask==0,-1e9); al=torch.softmax(a,1)
        return (al.unsqueeze(-1)*s.val(h)).sum(1)
enc=Enc(); enc.load_state_dict(torch.load(f'{GR}/enc_v1_grcomp.pt')); enc.eval()
def enc_u(toks):
    if not toks: return np.zeros(D,np.float32)
    tk=np.zeros((1,len(toks),D+1),np.float32); mk=np.ones((1,len(toks)),np.float32)
    for q,(f,res) in enumerate(toks): tk[0,q,:D]=f; tk[0,q,D]=res
    with torch.no_grad(): return enc(torch.tensor(tk),torch.tensor(mk)).numpy()[0]
def ridge_u(rev):
    if not rev: return np.zeros(D,np.float32)
    F=np.array([f for f,_ in rev],np.float32); y=np.array([r for _,r in rev],np.float32)
    return np.linalg.solve(F.T@F+LAM_R*np.eye(D),F.T@y)
W=1./np.log2(np.arange(2,KP+2))
def ndcg_scores(sc,rel,K):
    if not rel: return None
    top=np.argpartition(-sc,K)[:K]; top=top[np.argsort(-sc[top])]
    rs=set(rel); dcg=sum(W[p] for p,t in enumerate(top) if int(t) in rs); idcg=W[:min(K,len(rel))].sum()
    return dcg/idcg if idcg>0 else 0.
def eval_all(users, Bm):
    """Return dict model->{K:ndcg} averaged. Models: MOSTPOP, ridge, ENCODER, EASE. Universe-restricted."""
    acc={m:{K:0. for K in Ks} for m in ['MOSTPOP','ridge','ENCODER','EASE']}; m=0
    negbase=np.full(ni,-1e9,np.float64)
    for x in users:
        if x not in SPL: continue
        rd=dict(rat_by_u[x]); test=SPL[x]
        profile=[j for j in rd if (j not in test) and umask[j]]
        rel=[t for t in test if umask[t]]
        if len(rel)<1 or len(profile)<1: continue
        m+=1; excl=profile
        seeds=[(Q[j],rd[j]-mu-bi[j]) for j in profile]
        uE=enc_u(seeds); uR=ridge_u(seeds)
        # base scores over universe only
        scP=negbase.copy(); scP[univ]=popb[univ]
        scE=negbase.copy(); scE[univ]=popb[univ]+Q[univ]@uE
        scR=negbase.copy(); scR[univ]=popb[univ]+Q[univ]@uR
        # EASE: sum of B rows for profile items -> local scores
        ea=np.zeros(N,np.float64)
        for j in profile: ea+=Bm[loc[j]]
        scS=negbase.copy(); scS[univ]=ea
        for s in (scP,scE,scR,scS): s[excl]=-1e9
        for K in Ks:
            acc['MOSTPOP'][K]+=ndcg_scores(scP,rel,K) or 0
            acc['ENCODER'][K]+=ndcg_scores(scE,rel,K) or 0
            acc['ridge'][K]+=ndcg_scores(scR,rel,K) or 0
            acc['EASE'][K]+=ndcg_scores(scS,rel,K) or 0
    for mm in acc:
        for K in Ks: acc[mm][K]/=max(m,1)
    return acc,m
print(f"\n=== VARIANT A: COMPOSITE EXPLICIT EASE; universe N={N}; K={KP} ===",flush=True)
# lambda sweep on val
best=None
for lam in LAMS:
    Bm=ease_B(lam); acc,mval=eval_all(va.tolist(),Bm)
    v=acc['EASE'][KP]
    print(f"[val lam={lam:>5.0f}] EASE @{KP}={v:.4f} @10={acc['EASE'][10]:.4f} (n={mval}) ({time.time()-t0:.0f}s)",flush=True)
    if best is None or v>best[1]: best=(lam,v,Bm)
    else: del Bm
lam_sel=best[0]; Bm=best[2]
print(f">>> val-selected lambda={lam_sel}",flush=True)
accT,mte=eval_all(te.tolist(),Bm)
print(f"\n-- TEST (n={mte}), restricted universe N={N}, val-selected lam={lam_sel} --",flush=True)
for mm in ['MOSTPOP','ridge','ENCODER','EASE']:
    print(f"  {mm:<9}| @10={accT[mm][10]:.4f} | @{KP}={accT[mm][KP]:.4f}",flush=True)
print(f"\n>>> HEADROOM over MOSTPOP @{KP}: ENCODER={accT['ENCODER'][KP]-accT['MOSTPOP'][KP]:+.4f} | "
      f"EASE={accT['EASE'][KP]-accT['MOSTPOP'][KP]:+.4f} | ridge={accT['ridge'][KP]-accT['MOSTPOP'][KP]:+.4f}",flush=True)
print(f">>> HEADROOM over MOSTPOP @10 : ENCODER={accT['ENCODER'][10]-accT['MOSTPOP'][10]:+.4f} | "
      f"EASE={accT['EASE'][10]-accT['MOSTPOP'][10]:+.4f}",flush=True)
print(f"DONE ({time.time()-t0:.0f}s)",flush=True)
