"""Diagnose WHY the cold single-concept fold hurt NDCG. Two parts:
(A) SELECTION SWEEP (cold, empty profile): fold ONE loved concept chosen by different rules
    (niche=smallest / broad=largest / random / known-covering / held-covering=ORACLE) -> does ANY
    selection beat the popularity intercept? Isolates whether niche-choice was the killer.
(B) WARM + CONCEPT: reveal K known items (the channel's real job = refine a profile), then optionally
    add the loved concept. items-only vs items+concept -> does the concept help ON TOP of items?
No data reduction: every val user with a loved answerable concept scored under every rule.
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try: sys.stdout.reconfigure(encoding='utf-8')
except Exception: pass
import numpy as np, torch, torch.nn as nn
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
log(f"diag ckpt={CKPT}")

base=load_arena_base(); ni=base['ni']; head=base['headmask']; cnt=base['cnt']
d=np.load(META); uu=d['uu'].astype(np.int64); ii=d['ii'].astype(np.int64); rr=d['rr'].astype(np.float64)
o=np.argsort(uu,kind='stable'); uu,ii,rr=uu[o],ii[o],rr[o]; bnd=np.searchsorted(uu,np.arange(uu[-1]+2))

pa=torch.load(os.path.join(OUT,BASE+'.pt'),map_location='cpu'); IE=pa['student']['item_emb.weight'].numpy()
cb=ConceptBank(ni,cnt); Mbin=cb.Mbin.tocsr() if hasattr(cb.Mbin,'tocsr') else cb.Mbin
csz=np.array([len(Mbin[c].indices) for c in range(NC)])
memset=[set(Mbin[c].indices.tolist()) for c in range(NC)]

enc=SetEncoder(ni+NC, token_mode='wmat', pool='attn', nlev=NLEV, nknow=3, n_items=ni)
ck=torch.load(os.path.join(OUT,CKPT+'.pt'),map_location='cpu'); enc.load_state_dict(ck['student']); enc.eval()
dec=nn.Linear(D,ni); dec.load_state_dict(ck['decoder']); Wd=dec.weight.detach(); bd=dec.bias.detach()
def decode(z): return (z@Wd.T+bd).detach().numpy().astype(np.float64)
def enc_seqs(seqs):
    B=len(seqs); Lm=max(len(s[0]) for s in seqs)
    ids=np.zeros((B,Lm),np.int64); lv=np.zeros((B,Lm),np.int64); kn=np.zeros((B,Lm),np.int64); pad=np.ones((B,Lm),bool)
    for i,(a,l,k) in enumerate(seqs):
        ids[i,:len(a)]=a; lv[i,:len(a)]=l; kn[i,:len(a)]=k; pad[i,:len(a)]=False
    with torch.no_grad():
        return enc(torch.from_numpy(ids),torch.zeros(B,Lm),torch.from_numpy(pad),torch.from_numpy(lv),torch.from_numpy(kn))
def sv2lv(rat): return S.sv_to_level((rat-2.75)/2.25).astype(np.int64)

z0=enc(torch.zeros(1,1,dtype=torch.long),torch.zeros(1,1),torch.ones(1,1,dtype=torch.bool),torch.zeros(1,1,dtype=torch.long),torch.zeros(1,1,dtype=torch.long))
sc0=decode(z0)[0]

# ---- val cohort: known/held split (same RNG as training), loved concepts ----
uidsV=np.load(RSD+'/mm_val_uids.npy'); KV=np.load(RSD+'/mm_val_know.npy'); VV=np.load(RSD+'/mm_val_val.npy')
ANSV=(KV[:,:NC]>=1)&(VV[:,:NC]>=0); CVALV=np.clip(VV[:,:NC],0,3)
US=[]
for r,uid in enumerate(uidsV):
    a,b=bnd[int(uid)],bnd[int(uid)+1]; its,rat=ii[a:b],rr[a:b]
    if len(its)<8: continue
    ru=np.random.default_rng(AC.SEED*1_000_003+int(uid)); p=ru.permutation(len(its)); h=len(its)//2
    ki,kr=its[p[:h]],rat[p[:h]]; hi,hr=its[p[h:]],rat[p[h:]]; hl=hi[hr>=LO]
    if len(ki)<4 or len(hl)==0: continue
    loved=np.where(ANSV[r] & (CVALV[r]==3))[0]; loved=loved[csz[loved]>=20]
    if len(loved)==0: continue
    US.append((int(uid), ki, kr, hl, loved))
log(f"{len(US)} val users with known items + held likes + loved concept")

def uscore(sc, hl, prof):
    return ndcg10(sc.copy(),list(hl),prof,head,False), ndcg10(sc.copy(),list(hl),prof,head,True)
def agg(pairs):
    f=np.array([p[0] for p in pairs if p[0] is not None]); t=np.array([p[1] for p in pairs if p[1] is not None])
    return f.mean(), t.mean(), len(f), len(t)

# ============ PART A: cold selection sweep ============
log("="*60); log("PART A - COLD selection sweep (empty profile, fold ONE loved concept)")
def pick(u, rule):
    uid,ki,kr,hl,loved=u
    if rule=="niche":  return loved[np.argmin(csz[loved])]
    if rule=="broad":  return loved[np.argmax(csz[loved])]
    if rule=="random": return loved[np.random.default_rng(uid).integers(len(loved))]
    if rule=="knowcov":
        kset=set(ki.tolist()); ov=[len(memset[c]&kset)/max(csz[c],1) for c in loved]; return loved[int(np.argmax(ov))]
    if rule=="heldcov":  # ORACLE (uses held likes) -> upper bound only
        hset=set(hl.tolist()); ov=[len(memset[c]&hset) for c in loved]; return loved[int(np.argmax(ov))]
res={}
# intercept (empty)
base_pairs=[uscore(sc0,u[3],set()) for u in US]; res["intercept"]=agg(base_pairs)
for rule in ["niche","broad","random","knowcov","heldcov"]:
    pr=[]
    for u in US:
        c=pick(u,rule); scf=decode(enc_seqs([(np.array([ni+c]),np.array([3]),np.array([2]))]))[0]
        pr.append(uscore(scf,u[3],set()))
    res[rule]=agg(pr); log(f"  {rule:<9} FULL {res[rule][0]:.4f} ({res[rule][0]-res['intercept'][0]:+.4f})  TAIL {res[rule][1]:.4f} ({res[rule][1]-res['intercept'][1]:+.4f})")
log(f"  [intercept  FULL {res['intercept'][0]:.4f}  TAIL {res['intercept'][1]:.4f}]  (heldcov = ORACLE upper bound)")

# ============ PART B: warm profile +/- concept ============
log("="*60); log("PART B - WARM profile (K known items) +/- loved concept (concept as REFINEMENT)")
for K in [3,5,10]:
    io=[]; ic=[]
    for u in US:
        uid,ki,kr,hl,loved=u
        rk=np.random.default_rng(uid*7+K); order=rk.permutation(len(ki))[:min(K,len(ki))]
        it=ki[order]; lv=sv2lv(kr[order]); prof=set(it.tolist())
        # items only
        z_i=enc_seqs([(it, lv, np.full(len(it),2,np.int64))]); io.append(uscore(decode(z_i)[0],hl,prof))
        # items + niche loved concept
        c=loved[np.argmin(csz[loved])]
        ids=np.concatenate([it,[ni+c]]); lvs=np.concatenate([lv,[3]]); kns=np.concatenate([np.full(len(it),2,np.int64),[2]])
        z_c=enc_seqs([(ids,lvs,kns)]); ic.append(uscore(decode(z_c)[0],hl,prof))
    fo,to,_,_=agg(io); fc,tc,_,_=agg(ic)
    log(f"  K={K:<2} items-only FULL {fo:.4f} TAIL {to:.4f} | +concept FULL {fc:.4f} ({fc-fo:+.4f}) TAIL {tc:.4f} ({tc-to:+.4f})")
log("done")
