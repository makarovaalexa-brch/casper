"""DISENTANGLE: is the flat realizable arm from (a) 4-level COARSENESS or (b) the LLM answer being NOISE that
doesn't track true affinity? Feed the belief pool three answer channels, fixed order, and report per-channel lift
+ how well the LLM level tracks the true (behavioral) affinity.
  cont_oracle : t=<u*,d_c>  (continuous behavioral -- ceiling)
  true_4lvl   : quantize <u*,d_c> into 4 per-concept buckets -> bucket-mean t  (PERFECT 4-level answer)
  llm_4lvl    : population t per LLM ordinal level          (realizable stated -- what we have)
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
m=Wd.mean(0); Wc=Wd-m; rng=np.random.default_rng(0); _,_,Vt=np.linalg.svd(Wc[rng.choice(ni,4000,replace=False)],full_matrices=False); v1=Vt[0]; Wcw=Wc-(Wc@v1)[:,None]*v1[None,:]
Dc=np.zeros((NC,D))
for c in range(NC):
    idx=Mbin[c].indices
    if len(idx)>=20: x=Wcw[idx].mean(0); n=np.linalg.norm(x); Dc[c]=x/n if n>0 else 0
popb=np.log(cnt.astype(np.float64)+1.0); usable=(csz>=20)
enc=SetEncoder(ni+NC, token_mode='wmat', pool='attn', nlev=NLEV, nknow=3, n_items=ni); enc.load_state_dict(torch.load(OUT+'/wmat_ep1.pt',map_location='cpu')['student']); enc.eval()
def belief(items,levels):
    L=max(1,len(items)); ids=np.zeros((1,L),np.int64); lv=np.zeros((1,L),np.int64); kn=np.zeros((1,L),np.int64); pad=np.ones((1,L),bool)
    if len(items): ids[0,:len(items)]=items; lv[0,:len(items)]=levels; kn[0,:len(items)]=2; pad[0,:len(items)]=False
    with torch.no_grad(): return enc(torch.from_numpy(ids),torch.zeros(1,L),torch.from_numpy(pad),torch.from_numpy(lv),torch.from_numpy(kn))[0].numpy().astype(np.float64)
def sv2lv(x): return S.sv_to_level((x-2.75)/2.25).astype(np.int64)
def load(t):
    u=np.load(RSD+f'/mm_{t}_uids.npy'); K=np.load(RSD+f'/mm_{t}_know.npy'); V=np.load(RSD+f'/mm_{t}_val.npy'); return u,(K[:,:NC]>=1)&(V[:,:NC]>=0),np.clip(V[:,:NC],0,3)
# prior + per-LLM-level t/sig from train
uidsT,ansT,cvalT=load('train'); rs=np.random.default_rng(1); samp=rs.choice(len(uidsT),3000,replace=False)
UT=[]; aff_lvl={0:[],1:[],2:[],3:[]}
for r in samp:
    uid=uidsT[r]; a,b=bnd[int(uid)],bnd[int(uid)+1]; its,rat=ii[a:b],rr[a:b]
    if len(its)<8: continue
    u=belief(its.astype(np.int64),sv2lv(rat)); UT.append(u)
    for c in np.where(ansT[r]&usable)[0][:150]: aff_lvl[int(cvalT[r,c])].append(u@Dc[c])
UT=np.array(UT); mu0=UT.mean(0); Sig0=np.cov(UT.T)+1e-3*np.eye(D)
TLVL={L:float(np.mean(v)) for L,v in aff_lvl.items()}; SLVL={L:float(np.var(v)) for L,v in aff_lvl.items()}
# per-concept quartile thresholds of TRUE affinity (from train users) -> for true_4lvl buckets
aff_by_c={}
for k,r in enumerate(samp[:2000]):
    uid=uidsT[r]; a,b=bnd[int(uid)],bnd[int(uid)+1]; its,rat=ii[a:b],rr[a:b]
    if len(its)<8: continue
    u=UT[k] if k<len(UT) else belief(its.astype(np.int64),sv2lv(rat))
    for c in np.where(ansT[r]&usable)[0][:150]: aff_by_c.setdefault(c,[]).append(u@Dc[c])
QT={c:np.quantile(v,[.25,.5,.75]) for c,v in aff_by_c.items() if len(v)>=20}
BMEAN={c:[np.mean([x for x in v if x<=QT[c][0]]),np.mean([x for x in v if QT[c][0]<x<=QT[c][1]]),np.mean([x for x in v if QT[c][1]<x<=QT[c][2]]),np.mean([x for x in v if x>QT[c][2]])] for c,v in aff_by_c.items() if c in QT}
# val cohort
uidsV,ansV,cvalV=load('val'); US=[]
for r,uid in enumerate(uidsV):
    a,b=bnd[int(uid)],bnd[int(uid)+1]; its,rat=ii[a:b],rr[a:b]
    if len(its)<8: continue
    ru=np.random.default_rng(AC.SEED*1_000_003+int(uid)); p=ru.permutation(len(its)); h=len(its)//2
    ki,kr,hi,hr=its[p[:h]],rat[p[:h]],its[p[h:]],rat[p[h:]]; hl=hi[hr>=LO]; hlt=hl[~head[hl]]
    if len(ki)<4 or len(hlt)==0: continue
    cc=np.where(ansV[r]&usable)[0]; cc=cc[np.isin(cc,list(QT.keys()))]
    if len(cc)==0: continue
    us=belief(its.astype(np.int64),sv2lv(rat)); US.append((hlt.astype(np.int64),cc,cvalV[r,cc],us))
    if len(US)>=1200: break
log(f"val {len(US)}")
# LLM-vs-true agreement: does LLM level track true-affinity quartile?
tru=[]; llm=[]
for hl,cc,lv,us in US:
    for j,c in enumerate(cc[:40]):
        aff=us@Dc[c]; q=int((aff>QT[c][0])+(aff>QT[c][1])+(aff>QT[c][2])); tru.append(q); llm.append(int(lv[j]))
log(f"corr(LLM level, TRUE affinity quartile) = {np.corrcoef(llm,tru)[0,1]:+.3f}  (low => LLM answer is NOISE, not coarse)")
def ndt(sc,hl): return ndcg10(sc.copy(),list(hl),set(),head,True)
pt=np.mean([ndt(popb,hl) for hl,cc,lv,us in US]); ceil=np.mean([ndt(popb+Wd@us,hl) for hl,cc,lv,us in US]); log(f"popb {pt:.4f} ceiling {ceil:.4f}")
MAXQ=8
def tval(mode,c,lv,us):
    if mode=='cont': return us@Dc[c]
    if mode=='llm': return TLVL[int(lv)]
    aff=us@Dc[c]; q=int((aff>QT[c][0])+(aff>QT[c][1])+(aff>QT[c][2])); return BMEAN[c][q]  # true_4lvl
def run(mode):
    fin=[]
    for hl,cc,lv,us in US:
        mu=mu0.copy(); Sig=Sig0.copy(); order=np.argsort(-np.abs([TLVL[int(x)] for x in lv]))[:MAXQ]
        for j in order:
            c=cc[j]; dvec=Dc[c]; t=tval(mode,c,lv[j],us); s2=SLVL[int(lv[j])]
            Sd=Sig@dvec; k=Sd/(dvec@Sd+s2); mu=mu+k*(t-dvec@mu); Sig=Sig-np.outer(k,Sd)
        fin.append(ndt(popb+Wd@mu,hl))
    return np.mean(fin)-pt
for mode in ['cont','true_4lvl','llm']:
    log(f"  {mode:>9}: lift@q{MAXQ} {run(mode):+.4f}")
log("done")
