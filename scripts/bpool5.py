"""ITEMS+CONCEPTS UNIFIED Kalman: both are observations of ONE belief u. Item question -> observe <u,Wd_i>=resid
rating (d=Wd_i); concept question -> observe <u,d_c>=member4 target (d=d_c). Greedy selects whichever question
(item OR concept) maximizes uncertainty-reduction dᵀΣd. Tests: does a mixed interview climb monotonically, and
does the info-gain selector prefer items or concepts? Reports per-step tail + the item/concept pick mix.
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
def beliefs(seqs):
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
sig2c=float(np.mean([np.var(v) for v in aff_by_c.values() if len(v)>5]))*0.25
# item obs scale: residual ratings; global mean rating
gmean=float(rr.mean()); sig2i=0.5
log(f"ready {len(QT)} concepts sig2c={sig2c:.3f}")
uidsV,ansV,cvalV=load('val'); rows=[]; Vseq=[]
for r,uid in enumerate(uidsV):
    a,b=bnd[int(uid)],bnd[int(uid)+1]; its,rat=ii[a:b],rr[a:b]
    if len(its)<8: continue
    ru=np.random.default_rng(AC.SEED*1_000_003+int(uid)); p=ru.permutation(len(its)); h=len(its)//2
    ki,kr,hi,hr=its[p[:h]],rat[p[:h]],its[p[h:]],rat[p[h:]]; hl=hi[hr>=LO]; hlt=hl[~head[hl]]
    if len(ki)<4 or len(hlt)==0: continue
    cc=np.where(ansV[r]&usable)[0]; cc=cc[np.isin(cc,list(QT.keys()))]
    if len(cc)==0: continue
    # askable items = known-half items that are HEAD (recognizable), with residual rating
    ask=ki[head[ki]]; askr=kr[head[ki]]
    rows.append((hlt.astype(np.int64),cc,ask.astype(np.int64),askr)); Vseq.append((its.astype(np.int64),sv2lv(rat)))
    if len(rows)>=2000: break
UV=beliefs(Vseq); US=[(rows[i][0],rows[i][1],rows[i][2],rows[i][3],UV[i]) for i in range(len(rows))]
log(f"val {len(US)}")
def ndt(sc,hl): return ndcg10(sc.copy(),list(hl),set(),head,True)
pt=np.mean([ndt(popb,hl) for hl,cc,ak,akr,us in US]); log(f"popb {pt:.4f}")
MAXQ=8
def run(bank):   # bank: 'concept'|'item'|'both'
    step=[[] for _ in range(MAXQ+1)]; picks={'item':0,'concept':0}
    for hl,cc,ak,akr,us in US:
        mu=mu0.copy(); Sig=Sig0.copy(); step[0].append(ndt(popb+Wd@mu,hl))
        Q=[]  # (kind, dvec, target, sig2)
        if bank in ('concept','both'):
            for c in cc: aff=us@Dc[c]; q=int((aff>QT[c][0])+(aff>QT[c][1])+(aff>QT[c][2])); Q.append(('concept',Dc[c],BM[c][q],sig2c))
        if bank in ('item','both'):
            for it,rt in zip(ak,akr): Q.append(('item',Wd[it],rt-gmean,sig2i))
        for j in range(min(MAXQ,len(Q))):
            Dcand=np.array([q[1] for q in Q]); g=np.einsum('kd,kd->k',Dcand@Sig,Dcand); pk=int(np.argmax(g))
            kind,dvec,t,s2=Q.pop(pk); Sd=Sig@dvec; k=Sd/(dvec@Sd+s2); mu=mu+k*(t-dvec@mu); Sig=Sig-np.outer(k,Sd)
            picks[kind]+=1; step[j+1].append(ndt(popb+Wd@mu,hl))
    c=[np.mean(s) for s in step]; return c,picks
for bank in ['concept','item','both']:
    c,picks=run(bank); log(f"[{bank:>7}] "+" ".join(f"{i}:{c[i]:.4f}" for i in range(MAXQ+1))+f"  d@q{MAXQ}:{c[MAXQ]-pt:+.4f}  picks {picks}")
log("done")
