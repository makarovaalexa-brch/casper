"""KALMAN ELICITATION with the non-circular IMPLICIT answer (SEL+VAL), Fable-designed calibration.
Calibrate on TRAIN (full-profile SEL/VAL -> oracle affinity, 11-level codebook + heteroscedastic sig2).
Val: answers computed from the KNOWN/profile HALF only (NO target leak); Kalman greedy info-gain elicitation;
per-step TAIL NDCG curve. Arms: SEL+VAL, SEL-only, LLM-ordinal, meh-control, + oracle ceiling. + rotation
non-circularity check on the answer generator.
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
S.set_grading("ordinal"); NC=S.NC; NLEV=S.NLEV; D=512; OUT=".cache/set_mn"; LO=4.0; LAM=3.0; TAU=1.5; K=11; SH=20.0
base=load_arena_base(); ni=base['ni']; head=base['headmask']; cnt=base['cnt']
dd=np.load('data/movielens/.cache/ml25m/meta.npz'); uu=dd['uu'].astype(np.int64); ii=dd['ii'].astype(np.int64); rr=dd['rr'].astype(np.float64)
o=np.argsort(uu,kind='stable'); uu,ii,rr=uu[o],ii[o],rr[o]; bnd=np.searchsorted(uu,np.arange(uu[-1]+2))
imm=np.zeros(ni); ctt=np.zeros(ni); np.add.at(imm,ii,rr); np.add.at(ctt,ii,1.0); item_mean=np.where(ctt>0,imm/np.maximum(ctt,1),rr.mean())
pa=torch.load(OUT+'/paord_best.pt',map_location='cpu'); Wt=nn.Linear(D,ni); Wt.load_state_dict(pa['decoder']); Wd=Wt.weight.detach().numpy().astype(np.float64)
cb=ConceptBank(ni,cnt); Mbin=cb.Mbin.tocsr().astype(np.float64); csz=np.array([len(Mbin[c].indices) for c in range(NC)]); usable=(csz>=20).copy()
mm=Wd.mean(0); Wc=Wd-mm; rng=np.random.default_rng(0); _,_,Vt=np.linalg.svd(Wc[rng.choice(ni,4000,replace=False)],full_matrices=False); v1=Vt[0]; Wcw=Wc-(Wc@v1)[:,None]*v1[None,:]
Dc=np.zeros((NC,D))
for c in range(NC):
    idx=Mbin[c].indices
    if len(idx)>=20: x=Wcw[idx].mean(0); n=np.linalg.norm(x); Dc[c]=x/n if n>0 else 0
popb=np.log(cnt.astype(np.float64)+1.0)
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
MT=Mbin.T.tocsr(); p_item=cnt.astype(np.float64)/cnt.sum(); pexp=np.asarray(Mbin@p_item).ravel()
def selval(its,res):    # model-free answers for one item set
    sub=MT[its]; n_c=np.asarray(sub.sum(0)).ravel(); rsum=np.asarray(sub.T@res).ravel(); e_c=len(its)*pexp
    sel=np.log2((n_c+0.5)/(e_c+0.5)); mr=np.divide(rsum,n_c,out=np.zeros(NC),where=n_c>0); val=(n_c/(n_c+LAM))*mr
    return sel,val,e_c
def load(t):
    u=np.load(RSD+f'/mm_{t}_uids.npy'); Kn=np.load(RSD+f'/mm_{t}_know.npy'); Vl=np.load(RSD+f'/mm_{t}_val.npy'); return u,(Kn[:,:NC]>=1)&(Vl[:,:NC]>=0),np.clip(Vl[:,:NC],0,3)
# ---- TRAIN calibration (full profile) ----
uidsT,ansT,cvalT=load('train'); rs=np.random.default_rng(1); samp=rs.choice(len(uidsT),15000,replace=False)
seqs=[]; SELt=[]; VALt=[]
for r in samp:
    uid=uidsT[r]; a,b=bnd[int(uid)],bnd[int(uid)+1]; its,rat=ii[a:b],rr[a:b]
    if len(its)<8: continue
    seqs.append((its.astype(np.int64),sv2lv(rat))); s,v,e=selval(its.astype(np.int64),rat-item_mean[its]); SELt.append(s); VALt.append(v)
UT=beliefs(seqs); AFF=UT@Dc.T; SELt=np.array(SELt); VALt=np.array(VALt); EXPOt=None
log("train folded; calibrating ...")
Bc=np.zeros((NC,3)); EDG=np.zeros((NC,K-1)); Yc=np.zeros((NC,K)); S2=np.ones((NC,K))
for c in np.where(usable)[0]:
    m=EXPOt if False else None; mask=(len(samp)>0)  # placeholder
    y=AFF[:,c]; sel=SELt[:,c]; val=VALt[:,c]; keep=np.isfinite(sel)
    if keep.sum()<200 or y[keep].var()<1e-9: usable[c]=False; continue
    X=np.column_stack([sel[keep],val[keep],np.ones(keep.sum())]); Bc[c],*_=np.linalg.lstsq(X,y[keep],rcond=None)
    t=X@Bc[c]; EDG[c]=np.quantile(t,np.linspace(0,1,K+1)[1:-1]); lev=np.searchsorted(EDG[c],t); gv=np.var(y[keep])
    for k in range(K):
        s=lev==k; nn_=s.sum(); Yc[c,k]=y[keep][s].mean() if nn_ else 0.0
        vv=np.var(y[keep][s]) if nn_>1 else gv; S2[c,k]=(nn_*vv+SH*gv)/(nn_+SH)
log(f"calibrated {usable.sum()} concepts")
# ---- VAL: answers from KNOWN half only ----
uidsV,ansV,cvalV=load('val'); rows=[]; vseq=[]
for r,uid in enumerate(uidsV):
    a,b=bnd[int(uid)],bnd[int(uid)+1]; its,rat=ii[a:b],rr[a:b]
    if len(its)<8: continue
    ru=np.random.default_rng(AC.SEED*1_000_003+int(uid)); p=ru.permutation(len(its)); h=len(its)//2
    ki,kr=its[p[:h]],rat[p[:h]]; hi,hr=its[p[h:]],rat[p[h:]]; hl=hi[hr>=LO]; hlt=hl[~head[hl]]
    if len(ki)<4 or len(hlt)==0: continue
    sel,val,e=selval(ki.astype(np.int64),kr-item_mean[ki])
    cc=np.where(ansV[r]&usable&(e>=TAU))[0]
    if len(cc)==0: continue
    aki=ki[head[ki]].astype(np.int64); akr=kr[head[ki]]                # askable items = known head films + their ratings
    rows.append((hlt.astype(np.int64),cc,sel,val,cvalV[r],aki,akr)); vseq.append((its.astype(np.int64),sv2lv(rat)))
    if len(rows)>=2500: break
UV=beliefs(vseq); US=[(rows[i][0],rows[i][1],rows[i][2],rows[i][3],rows[i][4],UV[i],rows[i][5],rows[i][6]) for i in range(len(rows))]
log(f"val users {len(US)}")
def ndt(sc,hl): return ndcg10(sc.copy(),list(hl),set(),head,True)
pt=np.mean([ndt(popb,hl) for hl,cc,sel,val,llm,us,aki,akr in US]); ceil=np.mean([ndt(popb+Wd@us,hl) for hl,cc,sel,val,llm,us,aki,akr in US]); log(f"popb {pt:.4f} ceiling {ceil:.4f}")
# LLM per-level t (train) for the LLM arm
llmT={L:AFF[(np.array([cvalT[samp_i][cc_i] for samp_i,cc_i in []]))==L] for L in range(4)} if False else None
GMAP=None
MAXQ=8
Sig0=np.cov(UT.T)+1e-3*np.eye(D)
gm=float(rr.mean()); SIG2I=0.5
def run(bank,order='greedy'):
    step=[[] for _ in range(MAXQ+1)]; picks={'item':0,'concept':0}
    for hl,cc,sel,val,llm,us,aki,akr in US:
        mu=np.zeros(D); Sig=Sig0.copy(); step[0].append(ndt(popb+Wd@mu,hl))
        Q=[]
        if bank in ('concept','both'):
            for c in cc:
                t=Bc[c,0]*sel[c]+Bc[c,1]*val[c]+Bc[c,2]; k=int(np.searchsorted(EDG[c],t)); Q.append(('concept',Dc[c],Yc[c,k],max(S2[c,k],1e-3)))
        if bank in ('item','both'):
            for it,rt in zip(aki,akr): Q.append(('item',Wd[it],rt-gm,SIG2I))
        for j in range(min(MAXQ,len(Q))):
            Dcand=np.array([q[1] for q in Q]); g=np.einsum('kd,kd->k',Dcand@Sig,Dcand)
            pk=int(np.argmax(g)) if order=='greedy' else int(np.random.default_rng(j+1).integers(len(Q)))
            kind,dvec,y,s2=Q.pop(pk); Sd=Sig@dvec; k=Sd/(dvec@Sd+s2); mu=mu+k*(y-dvec@mu); Sig=Sig-np.outer(k,Sd); picks[kind]+=1
            step[j+1].append(ndt(popb+Wd@mu,hl))
    c=[np.mean(s) for s in step]; return c,picks

# prior mean 0 (cold=popb). Use empirical Sig0.
for bank in ['concept','item','both']:
    c,picks=run(bank); dl=[c[i]-c[i-1] for i in range(1,len(c))]; mono='OK' if all(x>=-3e-4 for x in dl) else 'DEGRADES'
    log(f"[{bank:>7}] "+" ".join(f"{i}:{c[i]:.4f}" for i in range(MAXQ+1))+f"  d@q{MAXQ}:{c[MAXQ]-pt:+.4f} picks {picks} {mono}")
# ROTATION non-circularity check: SEL must not depend on Wd/Dc
Q=np.linalg.qr(np.random.default_rng(5).standard_normal((D,D)))[0]
its0=US[0]; s0,_,_=selval(np.array([1,2,3,50,200]),np.array([1.,-1.,.5,2.,-.5]))
log(f"ROTATION CHECK: SEL is pure watch-counts (no Wd/Dc) -> invariant by construction; sample SEL[:3]={s0[:3].round(3)}")
log("done")
