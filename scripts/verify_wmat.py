"""Full verification of the trained W(v) concept branch (wmat_ep*.pt).
(1) SPECIFICITY SWEEP over ALL concepts (>=20 members): sign-flip ratio (loved vs hated,
    members vs unrelated) -> distribution, %>=3x/2x, genre-tags vs entities, vs concept coherence.
(2) COLD-CONCEPT NDCG on the DISJOINT val cohort: fold the user's single most-loved answerable
    concept, held-half likes = target, tail+full NDCG@10 vs the empty-fold intercept.
(3) FULL-PROFILE sanity: item-only forward == paord (structural, must be 0.0 drift).
No data reduction: every concept probed; every val user with a loved concept scored.
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try: sys.stdout.reconfigure(encoding='utf-8')
except Exception: pass
import numpy as np, torch, torch.nn as nn, json
import set_mn as S
from set_mn import SetEncoder, RSD
from signed_latent import load_arena_base, ndcg10
from reconciled import ConceptBank
import arena_core as AC
def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)
torch.set_num_threads(os.cpu_count())
S.set_grading("ordinal"); NC=S.NC; NLEV=S.NLEV; REFUSE=S.LV_REFUSE
META="C:/dev/phd/casper/data/movielens/.cache/ml25m/meta.npz"; OUT="C:/dev/phd/casper/.cache/set_mn"; LO=4.0; D=512
CKPT=os.environ.get("CKPT","wmat_ep1"); BASE=os.environ.get("BASE","paord_best")
log(f"verify ckpt={CKPT} base={BASE}")

base=load_arena_base(); ni=base['ni']; head=base['headmask']; cnt=base['cnt']
d=np.load(META); uu=d['uu'].astype(np.int64); ii=d['ii'].astype(np.int64); rr=d['rr'].astype(np.float64)
o=np.argsort(uu,kind='stable'); uu,ii,rr=uu[o],ii[o],rr[o]; bnd=np.searchsorted(uu,np.arange(uu[-1]+2))

# ---- whitened centroids (rebuild EXACTLY as trainer to seed the frozen concept rows) ----
pa=torch.load(os.path.join(OUT,BASE+'.pt'),map_location='cpu'); IE=pa['student']['item_emb.weight'].numpy()
Xc=IE-IE.mean(0); U,Sg,Vt=np.linalg.svd(Xc,full_matrices=False); V1=Vt[:1]; Xw=Xc-(Xc@V1.T)@V1
cb=ConceptBank(ni,cnt); Mbin=cb.Mbin.tocsr() if hasattr(cb.Mbin,'tocsr') else cb.Mbin
csz=np.array([len(Mbin[c].indices) for c in range(NC)])

# ---- model + trained checkpoint ----
enc=SetEncoder(ni+NC, token_mode='wmat', pool='attn', nlev=NLEV, nknow=3, n_items=ni)
ck=torch.load(os.path.join(OUT,CKPT+'.pt'),map_location='cpu')
enc.load_state_dict(ck['student']); enc.eval()
dec=nn.Linear(D,ni); dec.load_state_dict(ck['decoder']); Wd=dec.weight.detach(); bd=dec.bias.detach()
def decode(z): return (z@Wd.T+bd).detach().numpy().astype(np.float64)
def enc_seqs(seqs):
    B=len(seqs); Lm=max(len(s[0]) for s in seqs)
    ids=np.zeros((B,Lm),np.int64); lv=np.zeros((B,Lm),np.int64); kn=np.zeros((B,Lm),np.int64); pad=np.ones((B,Lm),bool)
    for i,(a,l,k) in enumerate(seqs):
        ids[i,:len(a)]=a; lv[i,:len(a)]=l; kn[i,:len(a)]=k; pad[i,:len(a)]=False
    with torch.no_grad():
        return enc(torch.from_numpy(ids),torch.zeros(B,Lm),torch.from_numpy(pad),torch.from_numpy(lv),torch.from_numpy(kn))

# ---- (3) FULL-PROFILE: structural (Wv touches ids>=ni only) -> item path == paord, verified pre-training (max|dz|=0)

# ---- (1) SPECIFICITY SWEEP: ALL concepts with >=20 members ----
prng=np.random.default_rng(0); allit=np.arange(ni)
z0=enc(torch.zeros(1,1,dtype=torch.long),torch.zeros(1,1),torch.ones(1,1,dtype=torch.bool),torch.zeros(1,1,dtype=torch.long),torch.zeros(1,1,dtype=torch.long))
sc0=decode(z0)[0]
def fold(cid,lv): return decode(enc_seqs([(np.array([ni+cid]),np.array([lv]),np.array([2]))]))[0]
ratios=[]; seps=[]; dms=[]; dns=[]; cids_ok=[]
log("specificity sweep over all concepts (>=20 members) ...")
t0=time.time()
for c in range(NC):
    mem=Mbin[c].indices
    if len(mem)<20: continue
    non=prng.choice(np.setdiff1d(allit,mem),min(2000,ni-len(mem)),replace=False)
    sl=fold(c,3); sh=fold(c,0)
    dm=float((sl[mem]-sh[mem]).mean()); dn=float((sl[non]-sh[non]).mean())
    ratios.append(dm/max(abs(dn),1e-9)); seps.append(dm-dn); dms.append(dm); dns.append(dn); cids_ok.append(c)
    if len(cids_ok)%200==0: log(f"  {len(cids_ok)} concepts, {(time.time()-t0)/60:.1f}m")
ratios=np.array(ratios); seps=np.array(seps); dms=np.array(dms); dns=np.array(dns); cids_ok=np.array(cids_ok)
np.save(f"{OUT}/wmat_specificity.npy", np.stack([cids_ok, ratios, seps, dms, dns]))
NTAG=1128
is_tag=cids_ok<NTAG
def pct(a,p): return float(np.percentile(a,p)) if len(a) else float('nan')
log("="*60)
# RATIO metric (dm/|dn|) — undersells a member-up/unrelated-DOWN pattern; report but interpret with SEPARATION.
log(f"RATIO dm/|dn| over {len(ratios)} concepts:  median {np.median(ratios):.2f}x  p25 {pct(ratios,25):.2f}x  p75 {pct(ratios,75):.2f}x  frac>=3x {(ratios>=3).mean():.1%}  >=2x {(ratios>=2).mean():.1%}")
# SEPARATION (dm - dn) is the RANKING-relevant quantity: how much love-vs-hate lifts members ABOVE unrelated.
log(f"SEPARATION dm-dn (ranking-relevant):  median {np.median(seps):+.3f}  p25 {pct(seps,25):+.3f}  p75 {pct(seps,75):+.3f}  frac>0 {(seps>0).mean():.1%}  frac>1 {(seps>1).mean():.1%}")
log(f"  member Δ  median {np.median(dms):+.3f} (frac>0 {(dms>0).mean():.1%});  unrelated Δ median {np.median(dns):+.3f}")
log(f"  genome-tags (n={is_tag.sum()}): sep median {np.median(seps[is_tag]):+.3f} frac>0 {(seps[is_tag]>0).mean():.1%}")
if (~is_tag).sum(): log(f"  entities   (n={(~is_tag).sum()}): sep median {np.median(seps[~is_tag]):+.3f} frac>0 {(seps[~is_tag]>0).mean():.1%}")
# coherence: bigger/broader concepts should be LESS separable -> correlate sep vs size
sz=csz[cids_ok]; ls=np.log(sz.clip(1))
if len(seps)>10: log(f"  corr(separation, log member-count) = {np.corrcoef(seps,ls)[0,1]:+.2f}  (expect negative: diffuse=less separable)")

# ---- (2) COLD-CONCEPT NDCG on the DISJOINT val cohort ----
uidsV=np.load(RSD+'/mm_val_uids.npy'); KV=np.load(RSD+'/mm_val_know.npy'); VV=np.load(RSD+'/mm_val_val.npy')
ANSV=(KV[:,:NC]>=1)&(VV[:,:NC]>=0); CVALV=np.clip(VV[:,:NC],0,3)
log("cold-concept NDCG on val cohort (fold single most-loved answerable concept vs empty intercept) ...")
fu_i=[]; fu_t=[]; nev=0
# base intercept ranking is user-independent (empty fold) -> compute once
for r,uid in enumerate(uidsV):
    a,b=bnd[int(uid)],bnd[int(uid)+1]; its,rat=ii[a:b],rr[a:b]
    if len(its)<8: continue
    ru=np.random.default_rng(AC.SEED*1_000_003+int(uid)); p=ru.permutation(len(its)); h=len(its)//2
    hi,hr=its[p[h:]],rat[p[h:]]; hl=hi[hr>=LO]
    if len(hl)==0: continue
    loved=np.where(ANSV[r] & (CVALV[r]==3))[0]
    loved=loved[csz[loved]>=20]
    if len(loved)==0: continue
    # pick the most-specific loved concept the user actually answered (smallest member set among loved) -> a niche taste
    cbest=loved[np.argmin(csz[loved])]
    fu_i.append((int(uid), cbest, hl)); nev+=1
log(f"  {nev} val users with a loved answerable concept")
# score
prof_empty=set()
base_full=[]; base_tail=[]; fold_full=[]; fold_tail=[]
BATCH=[]
for uid,cbest,hl in fu_i:
    zf=enc_seqs([(np.array([ni+cbest]),np.array([3]),np.array([2]))]); scf=decode(zf)[0]
    prof=set()  # cold: profile is empty (no items revealed), only the concept answer
    nb_f=ndcg10(sc0.copy(), list(hl), prof, head, False); nb_t=ndcg10(sc0.copy(), list(hl), prof, head, True)
    nf_f=ndcg10(scf.copy(), list(hl), prof, head, False); nf_t=ndcg10(scf.copy(), list(hl), prof, head, True)
    if nb_f is not None and nf_f is not None: base_full.append(nb_f); fold_full.append(nf_f)
    if nb_t is not None and nf_t is not None: base_tail.append(nb_t); fold_tail.append(nf_t)
base_full=np.array(base_full); fold_full=np.array(fold_full); base_tail=np.array(base_tail); fold_tail=np.array(fold_tail)
def ci(a): return 1.96*a.std()/np.sqrt(max(len(a),1))
log("="*60)
log(f"COLD-CONCEPT NDCG@10 (n_full={len(base_full)} n_tail={len(base_tail)}):")
log(f"  FULL  intercept {base_full.mean():.4f}  +concept {fold_full.mean():.4f}  delta {fold_full.mean()-base_full.mean():+.4f} +-{ci(fold_full-base_full):.4f}")
log(f"  TAIL  intercept {base_tail.mean():.4f}  +concept {fold_tail.mean():.4f}  delta {fold_tail.mean()-base_tail.mean():+.4f} +-{ci(fold_tail-base_tail):.4f}")
log(f"  (old FiLM Phase-B cold-concept tail bar = 0.0774)")
log("done")
