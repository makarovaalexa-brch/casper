"""Model-free answer kill-test + Kalman elicitation. Non-circular RATING channel (mean residual member rating
from raw profile), discretized to 10 levels. KILL-TEST: between-level/total variance of true affinity <u*,d_c>
for each channel (rating vs LLM) -> if rating < ~0.25 discretized concepts are dead realizably. Then Kalman
per-step tail curve (greedy) for rating10 / llm / meh-control, + prior-pick diagnostic (concept vs item info).
Refusal = concept with no rated members -> skipped.
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
im=np.zeros(ni); ct=np.zeros(ni); np.add.at(im,ii,rr); np.add.at(ct,ii,1.0); item_mean=np.where(ct>0,im/np.maximum(ct,1),rr.mean())
pa=torch.load(OUT+'/paord_best.pt',map_location='cpu'); Wt=nn.Linear(D,ni); Wt.load_state_dict(pa['decoder']); Wd=Wt.weight.detach().numpy().astype(np.float64)
cb=ConceptBank(ni,cnt); Mbin=cb.Mbin.tocsr(); csz=np.array([len(Mbin[c].indices) for c in range(NC)]); MEM={c:set(Mbin[c].indices.tolist()) for c in range(NC)}
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
def ratans(its,rat,c):
    mask=np.array([int(x) in MEM[c] for x in its])
    if mask.sum()==0: return None,0
    return float(np.mean(rat[mask]-item_mean[its[mask]])), int(mask.sum())
# TRAIN: aff=<u*,d_c>, llm level, rating answer
uidsT,ansT,cvalT=load('train'); rs=np.random.default_rng(1); samp=rs.choice(len(uidsT),4000,replace=False)
seqs=[]; meta=[]
for r in samp:
    uid=uidsT[r]; a,b=bnd[int(uid)],bnd[int(uid)+1]; its,rat=ii[a:b],rr[a:b]
    if len(its)<8: continue
    seqs.append((its.astype(np.int64),sv2lv(rat))); meta.append((r,its.astype(np.int64),rat))
UT=beliefs(seqs); mu0=UT.mean(0); Sig0=np.cov(UT.T)+1e-3*np.eye(D)
AFF=[]; LLM=[]; RAT=[]
for k,(r,its,rat) in enumerate(meta):
    for c in np.where(ansT[r]&usable)[0][:120]:
        ra,nm=ratans(its,rat,int(c))
        if ra is None: continue
        AFF.append(UT[k]@Dc[int(c)]); LLM.append(int(cvalT[r,c])); RAT.append(ra)
AFF=np.array(AFF); LLM=np.array(LLM); RAT=np.array(RAT)
log(f"train pairs (rated members): {len(AFF)}")
def decomp(levels,K):
    T={}; V={}; gm=AFF.mean()
    for L in range(K):
        m=levels==L
        T[L]=AFF[m].mean() if m.sum()>2 else gm; V[L]=AFF[m].var() if m.sum()>2 else AFF.var()
    betw=np.sum([(T[int(l)]-gm)**2 for l in levels])/len(levels); within=np.mean([V[int(l)] for l in levels])
    return T,V,betw/max(betw+within,1e-9)
llmL=LLM; rq=np.quantile(RAT,np.linspace(0,1,11)[1:-1]); ratL=np.digitize(RAT,rq)
Tl,Vl,R2l=decomp(llmL,4); Tr,Vr,R2r=decomp(ratL,10)
log(f"KILL-TEST between/total var of <u*,d_c>:  LLM(4lvl) {R2l:.3f}   RATING(10lvl) {R2r:.3f}   (>0.25 => viable)")
# VAL cohort
uidsV,ansV,cvalV=load('val'); rows=[]; vseq=[]
for r,uid in enumerate(uidsV):
    a,b=bnd[int(uid)],bnd[int(uid)+1]; its,rat=ii[a:b],rr[a:b]
    if len(its)<8: continue
    ru=np.random.default_rng(AC.SEED*1_000_003+int(uid)); p=ru.permutation(len(its)); h=len(its)//2
    ki,hi,hr=its[p[:h]],its[p[h:]],rat[p[h:]]; hl=hi[hr>=LO]; hlt=hl[~head[hl]]
    if len(ki)<4 or len(hlt)==0: continue
    cc=[]; A=[]
    for c in np.where(ansV[r]&usable)[0]:
        ra,nm=ratans(its.astype(np.int64),rat,int(c))
        if ra is None: continue
        cc.append(int(c)); A.append((int(cvalV[r,c]), int(np.digitize(ra,rq))))
    if not cc: continue
    rows.append((hlt.astype(np.int64),np.array(cc),A)); vseq.append((its.astype(np.int64),sv2lv(rat)))
    if len(rows)>=2000: break
UV=beliefs(vseq); US=[(rows[i][0],rows[i][1],rows[i][2]) for i in range(len(rows))]
log(f"val users {len(US)}")
def ndt(sc,hl): return ndcg10(sc.copy(),list(hl),set(),head,True)
pt=np.mean([ndt(popb,hl) for hl,cc,A in US]); log(f"popb {pt:.4f}")
MAXQ=8
def tv(mode,c,A):
    llml,ratl=A
    if mode=='llm': return Tl[llml],max(Vl[llml],1e-3)
    if mode=='rating10': return Tr[ratl],max(Vr[ratl],1e-3)
    if mode=='meh': return mu0@Dc[c],1.0
def run(mode):
    step=[[] for _ in range(MAXQ+1)]
    for hl,cc,A in US:
        mu=mu0.copy(); Sig=Sig0.copy(); step[0].append(ndt(popb+Wd@mu,hl)); pool=list(range(len(cc)))
        for j in range(min(MAXQ,len(cc))):
            Dcand=Dc[cc[pool]]; g=np.einsum('kd,kd->k',Dcand@Sig,Dcand); pj=pool[int(np.argmax(g))]; pool.remove(pj)
            c=cc[pj]; dvec=Dc[c]; t,s2=tv(mode,c,A[pj]); Sd=Sig@dvec; k=Sd/(dvec@Sd+s2); mu=mu+k*(t-dvec@mu); Sig=Sig-np.outer(k,Sd)
            step[j+1].append(ndt(popb+Wd@mu,hl))
    c=[np.mean(s) for s in step]; dl=[c[i]-c[i-1] for i in range(1,len(c))]; return c,('OK' if all(x>=-3e-4 for x in dl) else 'DEGRADES')
for mode in ['rating10','llm','meh']:
    c,mono=run(mode); log(f"[{mode:>9}] "+" ".join(f"{i}:{c[i]:.4f}" for i in range(MAXQ+1))+f"  d@q{MAXQ}:{c[MAXQ]-pt:+.4f} {mono}")
gi=np.einsum('kd,kd->k',Wd[head]@Sig0,Wd[head])/0.5; gc=np.array([Dc[c]@(Sig0@Dc[c]) for c in range(NC) if usable[c]])
log(f"PRIOR-PICK info-gain: item p90 {np.percentile(gi,90):.2f} | concept p90 {np.percentile(gc,90):.2f} max {gc.max():.2f}")
log("done")
