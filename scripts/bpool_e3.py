"""E3 = the Jul-14 GENRE-TREE replication INSIDE the Kalman belief pool.
Design (disjoint train/eval halves of the cached val users, symmetric for both arms):
  q1 = ONE static concept for everyone (variance-greedy argmax dᵀSig0 d among broadly-answerable concepts).
  Cluster users by their REALIZED q1 answer (quantile of the q1 affinity estimate) into G groups.
  q2 chosen EMPIRICALLY by mean tail-NDCG on the TRAIN half:
     - STATIC arm  : one q2* = argmax over all train users (single cluster).
     - ADAPTIVE arm: per-cluster q2*(g) = argmax over train users in cluster g.
  Evaluate BOTH arms on the EVAL half. Adaptive prize = adaptive tail@q2 - static tail@q2.
  MECHANISM: report each cluster's q2* identity + member-count (breadth). Jul-14 fingerprint = the
  per-cluster q2 is a NICHER concept (fewer members) than the single static q2 (a broad population axis).
NO candidate cap (all concepts answerable by >=COV of users are candidates). Probe on the 2500-val cache;
a HEADLINE claim needs ALL users + author sign-off (HARD RULE #1).
"""
import os,sys,time
sys.path.insert(0,'scripts')
import numpy as np, torch, torch.nn as nn
import set_mn as S
from set_mn import RSD
from signed_latent import load_arena_base, ndcg10
from reconciled import ConceptBank
def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}",flush=True)
torch.set_num_threads(os.cpu_count())
S.set_grading("ordinal"); NC=S.NC; D=512; OUT=".cache/set_mn"; CACHE=OUT+"/bpool_cache.npz"
COV=0.30      # candidate concept must be answerable by >=30% of users
G=4           # number of q1-answer clusters
MAXQ=2        # this experiment is about q1->q2 (the tree depth-2)
base=load_arena_base(); ni=base['ni']; head=base['headmask']; cnt=base['cnt']
pa=torch.load(OUT+'/paord_best.pt',map_location='cpu'); Wt=nn.Linear(D,ni); Wt.load_state_dict(pa['decoder']); Wd=Wt.weight.detach().numpy().astype(np.float64)
cb=ConceptBank(ni,cnt); Mbin=cb.Mbin.tocsr().astype(np.float64); csz=np.array([len(Mbin[c].indices) for c in range(NC)])
mm=Wd.mean(0); Wc=Wd-mm; rng=np.random.default_rng(0); _,_,Vt=np.linalg.svd(Wc[rng.choice(ni,4000,replace=False)],full_matrices=False); v1=Vt[0]; Wcw=Wc-(Wc@v1)[:,None]*v1[None,:]
Dc=np.zeros((NC,D))
for c in range(NC):
    idx=Mbin[c].indices
    if len(idx)>=20: x=Wcw[idx].mean(0); n=np.linalg.norm(x); Dc[c]=x/n if n>0 else 0
popb=np.log(cnt.astype(np.float64)+1.0)
# ---- load cached beliefs + calibration ----
assert os.path.exists(CACHE), "run bpool_c2f.py first to build the belief cache"
z=np.load(CACHE,allow_pickle=True); Sig0=z['Sig0']; Bc=z['Bc']; EDG=z['EDG']; Yc=z['Yc']; S2=z['S2']; usable=z['usable']; US=list(z['US'])
log(f"cache loaded: {len(US)} val users, {int(usable.sum())} usable concepts")
# US[i] = (hlt, cc, sel_cc, val_cc, aki, akr, hlall, us)
def tt_of(cc,sel,val,c):     # q1 affinity estimate for one concept c for one user (c must be in cc)
    j=np.where(cc==c)[0][0]; return Bc[c,0]*sel[j]+Bc[c,1]*val[j]+Bc[c,2]
def lvl_of(cc,sel,val,c):
    j=np.where(cc==c)[0][0]; tt=Bc[c,0]*sel[j]+Bc[c,1]*val[j]+Bc[c,2]; return int(np.searchsorted(EDG[c],tt))
def ndt(sc,hl): return ndcg10(sc.copy(),list(hl),set(),head,True)
def ndf(sc,hl): return ndcg10(sc.copy(),list(hl),set(),head,False)
# ---- coverage over ALL users, pick q1 (static, variance-greedy on Sig0 among broadly-answerable) ----
NU=len(US); cover=np.zeros(NC)
for hlt,cc,sel,val,aki,akr,hlall,us in US: cover[cc]+=1
cover/=NU
elig=np.where((cover>=0.80)&usable)[0]
vg=np.array([Dc[c]@(Sig0@Dc[c]) for c in elig]); q1=int(elig[np.argmax(vg)])
log(f"q1 (static, variance-greedy, >=80% answerable) = concept {q1}  members={csz[q1]}  cover={cover[q1]:.2f}  dSd={vg.max():.3f}")
# ---- keep q1-answerable users; split train/eval disjoint ----
rows=[u for u in US if q1 in u[1]]
log(f"q1-answerable users: {len(rows)}/{NU} ({len(rows)/NU:.2f})")
rs=np.random.default_rng(7); perm=rs.permutation(len(rows)); half=len(rows)//2
tr=[rows[i] for i in perm[:half]]; ev=[rows[i] for i in perm[half:]]
# ---- q1 update for a user -> (mu1, Sig1_s2level) ; returns mu1 and the realized q1 level ----
def q1update(u):
    hlt,cc,sel,val,aki,akr,hlall,us=u; k1=lvl_of(cc,sel,val,q1); y1=Yc[q1,k1]; s2=max(S2[q1,k1],1e-3)
    d=Dc[q1]; Sd=Sig0@d; kk=Sd/(d@Sd+s2); mu1=kk*(y1-0.0)  # mu0=0
    Sig1=Sig0-np.outer(kk,Sd); return mu1,Sig1,k1
# cluster edges by q1 affinity on TRAIN
ttr=np.array([tt_of(u[1],u[2],u[3],q1) for u in tr]); edges=np.quantile(ttr,np.linspace(0,1,G+1)[1:-1])
def clust(u): return int(np.searchsorted(edges,tt_of(u[1],u[2],u[3],q1)))
# candidate concepts (broadly answerable, != q1)
cand=np.array([c for c in np.where((cover>=COV)&usable)[0] if c!=q1]); log(f"{len(cand)} candidate q2 concepts (cover>={COV})")
# ---- precompute per-user q1 state + base tail/full ndcg (q1-only) ----
def prep(users):
    P=[]
    for u in users:
        hlt,cc,sel,val,aki,akr,hlall,us=u; mu1,Sig1,k1=q1update(u)
        base=popb+Wd@mu1; bt=ndt(base,hlt); bf=ndf(base,hlall)
        ccset={int(c):j for j,c in enumerate(cc)}
        P.append(dict(hlt=hlt,hlall=hlall,cc=cc,sel=sel,val=val,mu1=mu1,Sig1=Sig1,base=base,bt=bt,bf=bf,ccset=ccset,g=None))
    return P
log("prepping train ..."); PT=prep(tr);
for p in PT: p['g']=int(np.searchsorted(edges,Bc[q1,0]*p['sel'][np.where(p['cc']==q1)[0][0]]+Bc[q1,1]*p['val'][np.where(p['cc']==q1)[0][0]]+Bc[q1,2]))
log("prepping eval ...");  PE=prep(ev)
for p in PE: p['g']=int(np.searchsorted(edges,Bc[q1,0]*p['sel'][np.where(p['cc']==q1)[0][0]]+Bc[q1,1]*p['val'][np.where(p['cc']==q1)[0][0]]+Bc[q1,2]))
# ---- score q2=c for user p (if answerable) -> tail ndcg after q1+q2 ; else base ----
def q2tail(p,c):
    j=p['ccset'].get(int(c))
    if j is None: return p['bt']
    d=Dc[c]; Sig1=p['Sig1']; tt=Bc[c,0]*p['sel'][j]+Bc[c,1]*p['val'][j]+Bc[c,2]; k=int(np.searchsorted(EDG[c],tt)); y=Yc[c,k]; s2=max(S2[c,k],1e-3)
    Sd=Sig1@d; kk=Sd/(d@Sd+s2); mu2=p['mu1']+kk*(y-d@p['mu1']); return ndt(popb+Wd@mu2,p['hlt'])
def q2full(p,c):
    j=p['ccset'].get(int(c))
    if j is None: return p['bf']
    d=Dc[c]; Sig1=p['Sig1']; tt=Bc[c,0]*p['sel'][j]+Bc[c,1]*p['val'][j]+Bc[c,2]; k=int(np.searchsorted(EDG[c],tt)); y=Yc[c,k]; s2=max(S2[c,k],1e-3)
    Sd=Sig1@d; kk=Sd/(d@Sd+s2); mu2=p['mu1']+kk*(y-d@p['mu1']); return ndf(popb+Wd@mu2,p['hlall'])
# ---- TRAIN: score every candidate on all-train (static) and per-cluster (adaptive) ----
byc={g:[p for p in PT if p['g']==g] for g in range(G)}
log("train sizes "+str({g:len(byc[g]) for g in range(G)}))
t0=time.time()
sc_all=np.zeros(len(cand)); sc_g=np.zeros((G,len(cand)))
for ci,c in enumerate(cand):
    tv=np.array([q2tail(p,c) for p in PT]); sc_all[ci]=tv.mean()
    for g in range(G):
        idx=[i for i,p in enumerate(PT) if p['g']==g]; sc_g[g,ci]=tv[idx].mean() if idx else -1
    if ci%50==0: log(f"  scored {ci}/{len(cand)} ({time.time()-t0:.0f}s)")
q2_static=int(cand[np.argmax(sc_all)])
q2_ad={g:int(cand[np.argmax(sc_g[g])]) for g in range(G)}
log(f"STATIC q2* = concept {q2_static} members={csz[q2_static]}  train-tail {sc_all.max():.4f}")
for g in range(G): log(f"  cluster {g} (n={len(byc[g])}) q2*={q2_ad[g]} members={csz[q2_ad[g]]}  train-tail {sc_g[g].max():.4f}  {'<-- differs' if q2_ad[g]!=q2_static else '(==static)'}")
# ---- EVAL both arms ----
base_t=np.mean([p['bt'] for p in PE]); base_f=np.mean([p['bf'] for p in PE])
st_t=np.mean([q2tail(p,q2_static) for p in PE]); st_f=np.mean([q2full(p,q2_static) for p in PE])
ad_t=np.mean([q2tail(p,q2_ad[p['g']]) for p in PE]); ad_f=np.mean([q2full(p,q2_ad[p['g']]) for p in PE])
# popb-only floor for reference
pf=np.mean([ndf(popb,p['hlall']) for p in PE]); pt=np.mean([ndt(popb,p['hlt']) for p in PE])
log("="*70)
log(f"EVAL  popb-only            TAIL {pt:.4f}  FULL {pf:.4f}")
log(f"EVAL  q1-only              TAIL {base_t:.4f}  FULL {base_f:.4f}")
log(f"EVAL  q1+STATIC-q2         TAIL {st_t:.4f}  FULL {st_f:.4f}   (d vs q1 {st_t-base_t:+.4f})")
log(f"EVAL  q1+ADAPTIVE-q2       TAIL {ad_t:.4f}  FULL {ad_f:.4f}   (d vs q1 {ad_t-base_t:+.4f})")
log(f"ADAPTIVITY PRIZE (ad-static) TAIL {ad_t-st_t:+.4f}  FULL {ad_f-st_f:+.4f}")
# mechanism: are per-cluster q2 nicher than static?
log(f"MECHANISM: static-q2 members={csz[q2_static]} ; per-cluster q2 members {[csz[q2_ad[g]] for g in range(G)]} (fewer=nicher=Jul-14 fingerprint)")
log("done")
