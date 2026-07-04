"""
Dimension-sweep eval: ridge-fold full-profile headroom over MOSTPOP on the SAME
protocol/universe as gr_ease_composite.py (composite arena, K=510 primary + K=10,
restricted top-N universe, rng(123) disjoint held-out like-targets, val/te cohorts).
Loads Q_svd_comp_d{D}.npy (bias bi_svd_comp_d{D}.npy). Ridge lambda swept {1,5,25}
on the val cohort; test reported at val-selected lambda. No EASE (anchor lives in the
diagnostic). No commits.
Env: D (required), NUNIV(20000), KP(510).
"""
import os, time, numpy as np
t0=time.time(); GR='C:/dev/phd/casper/.cache/goodreads'
D=int(os.environ['D']); LIKE=4.0
N=int(os.environ.get('NUNIV',20000)); KP=int(os.environ.get('KP',510)); Ks=[10,KP]
LAMS=[float(x) for x in os.environ.get('LAMS','1,5,25').split(',')]
B=np.load(f'{GR}/base_comp.npz')
uu=B['uu']; ii=B['ii']; rr=B['rr']; cnt=B['cnt']; mu=float(B['mu']); ni=int(B['ni']); nu=int(B['nu'])
trU=B['trU']; va=B['va']; te=B['te']
Q=np.load(f'{GR}/Q_svd_comp_d{D}.npy').astype(np.float32); bi=np.load(f'{GR}/bi_svd_comp_d{D}.npy').astype(np.float32)
assert Q.shape[1]==D, f"Q dim {Q.shape} != D={D}"
popb=np.log(cnt+1.0).astype(np.float32)
# restricted universe: top-N items by like-count (identical to EASE diagnostic)
univ=np.argsort(-cnt)[:N]; univ=np.sort(univ)
umask=np.zeros(ni,bool); umask[univ]=True
print(f"[D={D}] universe N={N} of {ni}; like-count share kept={cnt[univ].sum()/cnt.sum()*100:.1f}% ({time.time()-t0:.0f}s)",flush=True)
# rating dicts + rng123 disjoint held-out targets (IDENTICAL protocol to EASE diagnostic)
evalset=np.array(sorted(set(va.tolist())|set(te.tolist()))); selm=np.isin(uu,evalset)
euu=uu[selm]; eii=ii[selm]; eR=rr[selm]; rat_by_u={}
for k in range(len(euu)): rat_by_u.setdefault(int(euu[k]),[]).append((int(eii[k]),float(eR[k])))
_rs=np.random.default_rng(123); SPL={}
for x in (va.tolist()+te.tolist()):
    lk=[j for j,r in rat_by_u.get(x,[]) if r>=LIKE]
    if len(lk)>=4: ll=lk[:]; _rs.shuffle(ll); SPL[x]=set(ll[len(ll)//2:])
print(f"eval dicts built ({time.time()-t0:.0f}s)",flush=True)
W=1./np.log2(np.arange(2,KP+2))
def ndcg(sc,rel,K):
    if not rel: return None
    top=np.argpartition(-sc,K)[:K]; top=top[np.argsort(-sc[top])]
    rs=set(rel); dcg=sum(W[p] for p,t in enumerate(top) if int(t) in rs); idcg=W[:min(K,len(rel))].sum()
    return dcg/idcg if idcg>0 else 0.
def ridge_u(rev,lam):
    if not rev: return np.zeros(D,np.float32)
    F=np.array([f for f,_ in rev],np.float32); y=np.array([r for _,r in rev],np.float32)
    return np.linalg.solve(F.T@F+lam*np.eye(D),F.T@y)
def eval_cohort(users, lam):
    accP={K:0. for K in Ks}; accR={K:0. for K in Ks}; m=0
    negbase=np.full(ni,-1e9,np.float64)
    for x in users:
        if x not in SPL: continue
        rd=dict(rat_by_u[x]); test=SPL[x]
        profile=[j for j in rd if (j not in test) and umask[j]]
        rel=[t for t in test if umask[t]]
        if len(rel)<1 or len(profile)<1: continue
        m+=1; excl=profile
        seeds=[(Q[j],rd[j]-mu-bi[j]) for j in profile]
        uR=ridge_u(seeds,lam)
        scP=negbase.copy(); scP[univ]=popb[univ]
        scR=negbase.copy(); scR[univ]=popb[univ]+Q[univ]@uR
        for s in (scP,scR): s[excl]=-1e9
        for K in Ks:
            accP[K]+=ndcg(scP,rel,K) or 0
            accR[K]+=ndcg(scR,rel,K) or 0
    for K in Ks: accP[K]/=max(m,1); accR[K]/=max(m,1)
    return accP,accR,m
print(f"\n=== DSWEEP EVAL D={D}; universe N={N}; K={KP} ===",flush=True)
best=None
for lam in LAMS:
    aP,aR,mv=eval_cohort(va.tolist(),lam)
    hv=aR[KP]-aP[KP]
    print(f"[val lam={lam:>5.1f}] ridge @{KP}={aR[KP]:.4f} MOSTPOP={aP[KP]:.4f} headroom={hv:+.4f} (n={mv}) ({time.time()-t0:.0f}s)",flush=True)
    if best is None or hv>best[1]: best=(lam,hv)
lam_sel=best[0]
print(f">>> val-selected ridge lambda={lam_sel}",flush=True)
aP,aR,mt=eval_cohort(te.tolist(),lam_sel)
print(f"\n-- TEST (n={mt}) D={D} universe N={N} lam={lam_sel} --",flush=True)
print(f"  MOSTPOP | @10={aP[10]:.4f} | @{KP}={aP[KP]:.4f}",flush=True)
print(f"  ridge   | @10={aR[10]:.4f} | @{KP}={aR[KP]:.4f}",flush=True)
print(f">>> HEADROOM over MOSTPOP: @{KP}={aR[KP]-aP[KP]:+.4f} | @10={aR[10]-aP[10]:+.4f}",flush=True)
try:
    with open(f'{GR}/Q_svd_comp_d{D}_peak.txt') as f: rmse=f.read().strip()
except Exception: rmse='(rmse log missing)'
print(f">>> RESULT_ROW D={D} headroom510={aR[KP]-aP[KP]:+.5f} headroom10={aR[10]-aP[10]:+.5f} lam={lam_sel} | {rmse}",flush=True)
print(f"DONE ({time.time()-t0:.0f}s)",flush=True)
