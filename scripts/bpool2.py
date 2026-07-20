"""BELIEF-POOL v2 (Fable-designed). Full-Sigma Kalman, EMPIRICAL anisotropic prior (mu0,Sig0 from population
user beliefs), decoder-side whitened concept dirs, per-level sigma2. CELL-1 = OPERATOR ISOLATION: oracle
t_c=<u*,d_c> (u* = paord full-profile belief) -> tests if the update itself is non-degrading given perfect
answers. Arms: greedy-Sigma order, random order, ALL-MEH control (t=prior-consistent -> must stay FLAT).
Per-step TAIL NDCG curve + delta. u* also gives the ceiling (full-profile fold)."""
import os,sys,time
sys.path.insert(0,'scripts')
import numpy as np, torch, torch.nn as nn
import set_mn as S
from set_mn import SetEncoder, RSD
from signed_latent import load_arena_base, ndcg10
from reconciled import ConceptBank
import arena_core as AC
def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}",flush=True)
torch.set_num_threads(os.cpu_count())
S.set_grading("ordinal"); NC=S.NC; NLEV=S.NLEV; D=512; OUT=".cache/set_mn"; LO=4.0
base=load_arena_base(); ni=base['ni']; head=base['headmask']; cnt=base['cnt']
dd=np.load('data/movielens/.cache/ml25m/meta.npz'); uu=dd['uu'].astype(np.int64); ii=dd['ii'].astype(np.int64); rr=dd['rr'].astype(np.float64)
o=np.argsort(uu,kind='stable'); uu,ii,rr=uu[o],ii[o],rr[o]; bnd=np.searchsorted(uu,np.arange(uu[-1]+2))
pa=torch.load(OUT+'/paord_best.pt',map_location='cpu'); Wd_t=nn.Linear(D,ni); Wd_t.load_state_dict(pa['decoder']); Wd=Wd_t.weight.detach().numpy().astype(np.float64); bd=Wd_t.bias.detach().numpy().astype(np.float64)
cb=ConceptBank(ni,cnt); Mbin=cb.Mbin.tocsr(); csz=np.array([len(Mbin[c].indices) for c in range(NC)])
m=Wd.mean(0); Wc=Wd-m; rng=np.random.default_rng(0); Us,Sg,Vt=np.linalg.svd(Wc[rng.choice(ni,4000,replace=False)],full_matrices=False); v1=Vt[0]; Wcw=Wc-(Wc@v1)[:,None]*v1[None,:]
Dc=np.zeros((NC,D)); 
for c in range(NC):
    idx=Mbin[c].indices
    if len(idx)>=20: x=Wcw[idx].mean(0); n=np.linalg.norm(x); Dc[c]=x/n if n>0 else 0
popb=np.log(cnt.astype(np.float64)+1.0)
# paord encoder for user beliefs u*
enc=SetEncoder(ni+NC, token_mode='wmat', pool='attn', nlev=NLEV, nknow=3, n_items=ni); enc.load_state_dict(torch.load(OUT+'/wmat_ep1.pt',map_location='cpu')['student']); enc.eval()
def belief(items,levels):
    L=max(1,len(items)); ids=np.zeros((1,L),np.int64); lv=np.zeros((1,L),np.int64); kn=np.zeros((1,L),np.int64); pad=np.ones((1,L),bool)
    if len(items): ids[0,:len(items)]=items; lv[0,:len(items)]=levels; kn[0,:len(items)]=2; pad[0,:len(items)]=False
    with torch.no_grad(): return enc(torch.from_numpy(ids),torch.zeros(1,L),torch.from_numpy(pad),torch.from_numpy(lv),torch.from_numpy(kn))[0].numpy().astype(np.float64)
def sv2lv(x): return S.sv_to_level((x-2.75)/2.25).astype(np.int64)
def load(tag):
    uids=np.load(RSD+f'/mm_{tag}_uids.npy'); KN=np.load(RSD+f'/mm_{tag}_know.npy'); VL=np.load(RSD+f'/mm_{tag}_val.npy')
    ans=(KN[:,:NC]>=1)&(VL[:,:NC]>=0); cval=np.clip(VL[:,:NC],0,3); return uids,ans,cval
# ---- prior mu0,Sig0 from TRAIN population beliefs (fold full profiles of a sample) ----
uidsT,ansT,cvalT=load('train'); usable=(csz>=20)
rngs=np.random.default_rng(1); samp=rngs.choice(len(uidsT),4000,replace=False)
UT=[]
for r in samp:
    uid=uidsT[r]; a,b=bnd[int(uid)],bnd[int(uid)+1]; its,rat=ii[a:b],rr[a:b]
    if len(its)<8: continue
    UT.append(belief(its.astype(np.int64), sv2lv(rat)))
UT=np.array(UT); mu0=UT.mean(0); Sig0=np.cov(UT.T)+1e-3*np.eye(D)
log(f"prior from {len(UT)} train beliefs; cold popb+<mu0,Wd> vs popb")
# per-level sigma2 + realizable t(level): bucket <u*,d_c> by answer level over train sample
aff_by_lvl={0:[],1:[],2:[],3:[]}
for k,r in enumerate(samp[:1500]):
    uid=uidsT[r]; a,b=bnd[int(uid)],bnd[int(uid)+1]; its,rat=ii[a:b],rr[a:b]
    if len(its)<8: continue
    u=belief(its.astype(np.int64),sv2lv(rat)); ac=np.where(ansT[r]&usable)[0]
    for c in ac[:200]:
        aff_by_lvl[int(cvalT[r,c])].append(u@Dc[c])
TLVL={L:float(np.mean(v)) if v else 0.0 for L,v in aff_by_lvl.items()}
SLVL={L:float(np.var(v)) if len(v)>1 else 1.0 for L,v in aff_by_lvl.items()}
log(f"t(level) {[(L,round(TLVL[L],3)) for L in range(4)]}  sig2(level) {[(L,round(SLVL[L],3)) for L in range(4)]}")
# ---- val cohort with oracle u* ----
uidsV,ansV,cvalV=load('val'); US=[]
for r,uid in enumerate(uidsV):
    a,b=bnd[int(uid)],bnd[int(uid)+1]; its,rat=ii[a:b],rr[a:b]
    if len(its)<8: continue
    ru=np.random.default_rng(AC.SEED*1_000_003+int(uid)); p=ru.permutation(len(its)); h=len(its)//2
    ki,kr,hi,hr=its[p[:h]],rat[p[:h]],its[p[h:]],rat[p[h:]]; hl=hi[hr>=LO]; hlt=hl[~head[hl]]
    if len(ki)<4 or len(hlt)==0: continue
    cc=np.where(ansV[r]&usable)[0]
    if len(cc)==0: continue
    ustar=belief(its.astype(np.int64),sv2lv(rat))          # full-profile belief = oracle taste (labeled ceiling)
    lv=cvalV[r,cc]
    US.append((hlt.astype(np.int64), cc, lv, ustar))
    if len(US)>=1500: break
log(f"val users {len(US)} (capped for speed)")
def ndt(sc,hl): return ndcg10(sc.copy(),list(hl),set(),head,True)
def agg(step_lists):
    return [np.mean([x for x in s if x is not None]) if s else np.nan for s in step_lists]
MAXQ=10
pt=np.mean([ndt(popb,hl) for hl,cc,lv,us in US]); ceil=np.mean([ndt(popb+Wd@us,hl) for hl,cc,lv,us in US])
log(f"popb tail {pt:.4f} | full-profile ceiling tail {ceil:.4f}")
def run(order, tmode):   # order: 'greedy'|'rand' ; tmode: 'oracle'|'meh'
    step=[[] for _ in range(MAXQ+1)]
    for hl,cc,lv,us in US:
        mu=mu0.copy(); Sig=Sig0.copy(); step[0].append(ndt(popb+Wd@mu,hl))
        pool=list(range(len(cc)))
        for j in range(min(MAXQ,len(cc))):
            if order=='greedy':
                gains=[Dc[cc[p]]@(Sig@Dc[cc[p]]) for p in pool]; pk=pool[int(np.argmax(gains))]
            else: pk=pool[int(np.random.default_rng(j).integers(len(pool)))]
            pool.remove(pk); c=cc[pk]; dvec=Dc[c]
            t = (us@dvec) if tmode=='oracle' else TLVL[int(lv[pk])]   # oracle vs meh-control uses calibrated t
            s2=SLVL[int(lv[pk])]; Sd=Sig@dvec; k=Sd/(dvec@Sd+s2); mu=mu+k*(t-dvec@mu); Sig=Sig-np.outer(k,Sd)
            step[j+1].append(ndt(popb+Wd@mu,hl))
    return agg(step)
for order,tmode,lab in [('greedy','oracle','ORACLE-t greedy'),('rand','oracle','ORACLE-t random'),('greedy','meh','CALIB-t greedy (realizable)')]:
    c=run(order,tmode); dl=[c[i]-c[i-1] for i in range(1,len(c)) if not np.isnan(c[i])]
    mono='OK' if all(x>=-2e-4 for x in dl) else 'DEGRADES'
    log(f"[{lab:<28}] "+" ".join(f"q{i}:{c[i]:.4f}" for i in range(min(9,MAXQ+1)))+f"  d@q8:{c[min(8,MAXQ)]-pt:+.4f} {mono}")
log("done")
