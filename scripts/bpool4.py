"""DEFINITIVE belief-pool run: MEMBER-DERIVED (behavioral) concept answer vs LLM, full per-step invariant curve.
Answer channels: cont(=<u*,d_c>, ceiling) ; member4 (discretize <u*,d_c> to 4 per-concept buckets = the
truthful discretized answer the AUTHOR proposed) ; meh-control (prior-consistent -> must stay flat).
Arms: greedy-Sigma order + random order. Per-step TAIL NDCG, monotonicity flag. Batched belief folding.
"""
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
pa=torch.load(OUT+'/paord_best.pt',map_location='cpu'); Wt=nn.Linear(D,ni); Wt.load_state_dict(pa['decoder']); Wd=Wt.weight.detach().numpy().astype(np.float64)
cb=ConceptBank(ni,cnt); Mbin=cb.Mbin.tocsr(); csz=np.array([len(Mbin[c].indices) for c in range(NC)])
mm=Wd.mean(0); Wc=Wd-mm; rng=np.random.default_rng(0); _,_,Vt=np.linalg.svd(Wc[rng.choice(ni,4000,replace=False)],full_matrices=False); v1=Vt[0]; Wcw=Wc-(Wc@v1)[:,None]*v1[None,:]
Dc=np.zeros((NC,D))
for c in range(NC):
    idx=Mbin[c].indices
    if len(idx)>=20: x=Wcw[idx].mean(0); n=np.linalg.norm(x); Dc[c]=x/n if n>0 else 0
popb=np.log(cnt.astype(np.float64)+1.0); usable=(csz>=20)
enc=SetEncoder(ni, token_mode='film', pool='attn', nlev=NLEV, nknow=3); enc.load_state_dict(pa['student'],strict=False); enc.eval()
def sv2lv(x): return S.sv_to_level((x-2.75)/2.25).astype(np.int64)
def beliefs(seqs):        # BATCHED: list of (items,levels) -> (B,D)
    out=[]
    for i in range(0,len(seqs),256):
        ch=seqs[i:i+256]; B=len(ch); L=max(1,max(len(s[0]) for s in ch))
        ids=np.zeros((B,L),np.int64); lv=np.zeros((B,L),np.int64); kn=np.zeros((B,L),np.int64); pad=np.ones((B,L),bool)
        for j,(it,le) in enumerate(ch):
            if len(it): ids[j,:len(it)]=it; lv[j,:len(it)]=le; kn[j,:len(it)]=2; pad[j,:len(it)]=False
        with torch.no_grad(): out.append(enc(torch.from_numpy(ids),torch.zeros(B,L),torch.from_numpy(pad),torch.from_numpy(lv),torch.from_numpy(kn)).numpy().astype(np.float64))
    return np.concatenate(out,0)
def load(t):
    u=np.load(RSD+f'/mm_{t}_uids.npy'); K=np.load(RSD+f'/mm_{t}_know.npy'); V=np.load(RSD+f'/mm_{t}_val.npy'); return u,(K[:,:NC]>=1)&(V[:,:NC]>=0),np.clip(V[:,:NC],0,3)
# prior + per-concept quartiles from TRAIN
uidsT,ansT,cvalT=load('train'); rs=np.random.default_rng(1); samp=rs.choice(len(uidsT),4000,replace=False)
Tseq=[]; Tans=[]
for r in samp:
    uid=uidsT[r]; a,b=bnd[int(uid)],bnd[int(uid)+1]; its,rat=ii[a:b],rr[a:b]
    if len(its)<8: continue
    Tseq.append((its.astype(np.int64),sv2lv(rat))); Tans.append(r)
UT=beliefs(Tseq); mu0=UT.mean(0); Sig0=np.cov(UT.T)+1e-3*np.eye(D)
aff_by_c={}
for k,r in enumerate(Tans):
    for c in np.where(ansT[r]&usable)[0][:200]: aff_by_c.setdefault(int(c),[]).append(UT[k]@Dc[c])
QT={c:np.quantile(v,[.25,.5,.75]) for c,v in aff_by_c.items() if len(v)>=20}
BM={c:[np.mean([x for x in v if x<=QT[c][0]] or [QT[c][0]]),np.mean([x for x in v if QT[c][0]<x<=QT[c][1]] or [QT[c][1]]),np.mean([x for x in v if QT[c][1]<x<=QT[c][2]] or [QT[c][2]]),np.mean([x for x in v if x>QT[c][2]] or [QT[c][2]])] for c,v in aff_by_c.items() if c in QT}
sig2=float(np.mean([np.var(v) for v in aff_by_c.values() if len(v)>5]))*0.25   # obs noise (downscaled -> trust obs)
log(f"prior+quartiles ready; {len(QT)} concepts; sig2={sig2:.3f}")
# val cohort
uidsV,ansV,cvalV=load('val'); rows=[]; Vseq=[]
for r,uid in enumerate(uidsV):
    a,b=bnd[int(uid)],bnd[int(uid)+1]; its,rat=ii[a:b],rr[a:b]
    if len(its)<8: continue
    ru=np.random.default_rng(AC.SEED*1_000_003+int(uid)); p=ru.permutation(len(its)); h=len(its)//2
    ki,hi,hr=its[p[:h]],its[p[h:]],rat[p[h:]]; hl=hi[hr>=LO]; hlt=hl[~head[hl]]
    if len(ki)<4 or len(hlt)==0: continue
    cc=np.where(ansV[r]&usable)[0]; cc=cc[np.isin(cc,list(QT.keys()))]
    if len(cc)==0: continue
    rows.append((hlt.astype(np.int64),cc)); Vseq.append((its.astype(np.int64),sv2lv(rat)))
    if len(rows)>=2500: break
UV=beliefs(Vseq); US=[(rows[i][0],rows[i][1],UV[i]) for i in range(len(rows))]
log(f"val {len(US)}")
def ndt(sc,hl): return ndcg10(sc.copy(),list(hl),set(),head,True)
pt=np.mean([ndt(popb,hl) for hl,cc,us in US]); ceil=np.mean([ndt(popb+Wd@us,hl) for hl,cc,us in US]); log(f"popb {pt:.4f} ceiling {ceil:.4f}")
MAXQ=8
def tval(mode,c,us):
    aff=us@Dc[c]
    if mode=='cont': return aff
    if mode=='meh': return mu0@Dc[c]        # prior-consistent -> control
    q=int((aff>QT[c][0])+(aff>QT[c][1])+(aff>QT[c][2])); return BM[c][q]   # member4
def run(mode,order):
    step=[[] for _ in range(MAXQ+1)]
    for hl,cc,us in US:
        mu=mu0.copy(); Sig=Sig0.copy(); step[0].append(ndt(popb+Wd@mu,hl)); pool=list(cc)
        for j in range(min(MAXQ,len(cc))):
            if order=='greedy':
                Dcand=Dc[pool]; g=np.einsum('kd,kd->k',Dcand@Sig,Dcand); c=pool[int(np.argmax(g))]
            else: c=pool[int(np.random.default_rng(j*7+1).integers(len(pool)))]
            pool.remove(c); dvec=Dc[c]; t=tval(mode,c,us); Sd=Sig@dvec; k=Sd/(dvec@Sd+sig2); mu=mu+k*(t-dvec@mu); Sig=Sig-np.outer(k,Sd)
            step[j+1].append(ndt(popb+Wd@mu,hl))
    c=[np.mean(s) for s in step]; dl=[c[i]-c[i-1] for i in range(1,len(c))]; mono='OK' if all(x>=-3e-4 for x in dl) else 'DEGRADES'
    return c,mono
for mode,order,lab in [('cont','greedy','CONT ceiling greedy'),('member4','greedy','MEMBER-4lvl greedy'),('member4','rand','MEMBER-4lvl random'),('meh','greedy','MEH control greedy')]:
    c,mono=run(mode,order); log(f"[{lab:<22}] "+" ".join(f"{i}:{c[i]:.4f}" for i in range(MAXQ+1))+f"  d@q{MAXQ}:{c[MAXQ]-pt:+.4f} {mono}")
log("done")
