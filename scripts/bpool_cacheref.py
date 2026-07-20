"""REFUSAL-AS-ANSWER cache + direct test. The watch-signal SEL already encodes refusal: a concept with zero
watched members gives SEL=log2(0.5/(E_c+0.5)) < 0, more negative the higher the expected exposure E_c
(=> "haven't touched this POPULAR thing" is a confident below-prior signal; "haven't touched this NICHE thing"
is uninformative -> high sigma^2 via the low-SEL calibration bin). We were GATING these out (E_c>=TAU). Here we
DROP the gate and the LLM-answerability mask: every usable concept is answerable, as a genuine answer (n_c>=1)
or a REFUSAL (n_c=0). Gives universal coverage.

Directly measures what refusal buys: variance-greedy over ANSWERABLE-ONLY (n_c>=1) vs WITH-REFUSAL (all usable).
Caches full per-user answers to bpool_cache_ref.npz for the tree/continuous selectors.
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
S.set_grading("ordinal"); NC=S.NC; NLEV=S.NLEV; D=512; OUT=".cache/set_mn"; LO=4.0; LAM=3.0; K=11; SH=20.0
CACHE=OUT+"/bpool_cache_ref.npz"
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
MT=Mbin.T.tocsr(); p_item=cnt.astype(np.float64)/cnt.sum(); pexp=np.asarray(Mbin@p_item).ravel()
def selval(its,res):
    sub=MT[its]; n_c=np.asarray(sub.sum(0)).ravel(); rsum=np.asarray(sub.T@res).ravel(); e_c=len(its)*pexp
    sel=np.log2((n_c+0.5)/(e_c+0.5)); mr=np.divide(rsum,n_c,out=np.zeros(NC),where=n_c>0); return sel,(n_c/(n_c+LAM))*mr,e_c,n_c
def load(t):
    u=np.load(RSD+f'/mm_{t}_uids.npy'); return u
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
if os.path.exists(CACHE):
    z=np.load(CACHE,allow_pickle=True); mu0=z['mu0']; Sig0=z['Sig0']; Bc=z['Bc']; EDG=z['EDG']; Yc=z['Yc']; S2=z['S2']; usable=z['usable']; US=list(z['US']); UIDX=z['uidx']; log("loaded ref cache")
else:
    uidsT=load('train'); rs=np.random.default_rng(1); samp=rs.choice(len(uidsT),15000,replace=False)
    seqs=[]; SELt=[]; VALt=[]
    for r in samp:
        uid=uidsT[r]; a,b=bnd[int(uid)],bnd[int(uid)+1]; its,rat=ii[a:b],rr[a:b]
        if len(its)<8: continue
        seqs.append((its.astype(np.int64),sv2lv(rat))); s,v,e,n=selval(its.astype(np.int64),rat-item_mean[its]); SELt.append(s); VALt.append(v)
    UT=beliefs(seqs); AFF=UT@Dc.T; SELt=np.array(SELt); VALt=np.array(VALt); mu0=UT.mean(0)*0.0; Sig0=np.cov(UT.T)+1e-3*np.eye(D)
    Bc=np.zeros((NC,3)); EDG=np.zeros((NC,K-1)); Yc=np.zeros((NC,K)); S2=np.ones((NC,K))
    for c in np.where(usable)[0]:   # calibration ALREADY includes refusers (SEL finite at n_c=0)
        y=AFF[:,c]; sel=SELt[:,c]; keep=np.isfinite(sel)
        if keep.sum()<200 or y[keep].var()<1e-9: usable[c]=False; continue
        X=np.column_stack([sel[keep],VALt[:,c][keep],np.ones(keep.sum())]); Bc[c],*_=np.linalg.lstsq(X,y[keep],rcond=None)
        t=X@Bc[c]; EDG[c]=np.quantile(t,np.linspace(0,1,K+1)[1:-1]); lev=np.searchsorted(EDG[c],t); gv=np.var(y[keep])
        for k in range(K):
            sk=lev==k; nn_=sk.sum(); Yc[c,k]=y[keep][sk].mean() if nn_ else 0.0; vv=np.var(y[keep][sk]) if nn_>1 else gv; S2[c,k]=(nn_*vv+SH*gv)/(nn_+SH)
    log(f"calibrated {usable.sum()} concepts")
    UIDX=np.where(usable)[0]   # column order for stored answers
    uidsV=load('val'); rows=[]; vseq=[]
    for r,uid in enumerate(uidsV):
        a,b=bnd[int(uid)],bnd[int(uid)+1]; its,rat=ii[a:b],rr[a:b]
        if len(its)<8: continue
        ru=np.random.default_rng(AC.SEED*1_000_003+int(uid)); p=ru.permutation(len(its)); h=len(its)//2
        ki,kr=its[p[:h]],rat[p[:h]]; hi,hr=its[p[h:]],rat[p[h:]]; hl=hi[hr>=LO]; hlt=hl[~head[hl]]
        if len(ki)<4 or len(hlt)==0: continue
        sel,val,e,n_c=selval(ki.astype(np.int64),kr-item_mean[ki])
        selU=sel[UIDX].astype(np.float32); valU=val[UIDX].astype(np.float32); refU=(n_c[UIDX]==0)  # refusal flag
        aki=ki[head[ki]].astype(np.int64); akr=kr[head[ki]]
        rows.append((hlt.astype(np.int64),selU,valU,refU,aki,akr,hl.astype(np.int64))); vseq.append((its.astype(np.int64),sv2lv(rat)))
        if len(rows)>=2500: break
    UV=beliefs(vseq); US=[rows[i]+(UV[i],) for i in range(len(rows))]
    np.savez(CACHE,mu0=mu0,Sig0=Sig0,Bc=Bc,EDG=EDG,Yc=Yc,S2=S2,usable=usable,uidx=UIDX,US=np.array(US,dtype=object)); log("cached ref")
log(f"val users {len(US)}  usable concepts {len(UIDX)}")
# ---- variance-greedy: ANSWERABLE-ONLY vs WITH-REFUSAL ----
def ndt(sc,hl): return ndcg10(sc.copy(),list(hl),set(),head,True)
def ndf(sc,hl): return ndcg10(sc.copy(),list(hl),set(),head,False)
MAXQ=8; TAU=1.5
# precompute per-user answer table over UIDX columns
def answers(u):
    hlt,selU,valU,refU,aki,akr,hlall,us=u; tt=Bc[UIDX,0]*selU+Bc[UIDX,1]*valU+Bc[UIDX,2]
    lev=np.array([int(np.searchsorted(EDG[c],tt[j])) for j,c in enumerate(UIDX)])
    y=Yc[UIDX,lev]; s2=np.maximum(S2[UIDX,lev],1e-3); return hlt,hlall,y,s2,refU
def run(mode):   # 'ans' = n_c>=1 only ; 'ref' = all usable (refusal included)
    ST=[[] for _ in range(MAXQ+1)]; SF=[[] for _ in range(MAXQ+1)]
    for u in US:
        hlt,hlall,y,s2,refU=answers(u); mu=np.zeros(D); Sig=Sig0.copy()
        cols=np.arange(len(UIDX)) if mode=='ref' else np.where(~refU)[0]
        f=ndf(popb+Wd@mu,hlall); t=ndt(popb+Wd@mu,hlt); SF[0].append(f); ST[0].append(t)
        avail=list(cols)
        for j in range(min(MAXQ,len(avail))):
            Dcand=Dc[UIDX[avail]]; SD=Sig@Dcand.T; g=np.einsum('dq,dq->q',SD,SD)
            pk=int(np.argmax(g)); col=avail.pop(pk); c=UIDX[col]; d=Dc[c]
            Sd=Sig@d; kk=Sd/(d@Sd+s2[col]); mu=mu+kk*(y[col]-d@mu); Sig=Sig-np.outer(kk,Sd)
            SF[j+1].append(ndf(popb+Wd@mu,hlall)); ST[j+1].append(ndt(popb+Wd@mu,hlt))
    return [np.mean(x) for x in ST],[np.mean(x) for x in SF]
pt0=np.mean([ndt(popb,u[0]) for u in US]); pf0=np.mean([ndf(popb,u[6]) for u in US]); log(f"popb TAIL {pt0:.4f} FULL {pf0:.4f}")
for mode in ['ans','ref']:
    ct,cf=run(mode)
    log(f"[{mode:>3}] TAIL "+" ".join(f"{i}:{ct[i]:.4f}" for i in range(MAXQ+1))+f" d@8 {ct[8]-pt0:+.4f}  FULL d@8 {cf[8]-pf0:+.4f}")
log("done")
