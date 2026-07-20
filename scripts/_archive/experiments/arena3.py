"""Disciplined final: reuse cached TAIL-ND. (A) MARGIN DECOMPOSITION of the genre implicit(+0.022) vs
answerer(+0.0047) gap -- is the loss a tie-artifact of a bad argmax estimator? Pre-registered rule: if
STRICT-MAX answerer users recover >=60% of implicit per-capita prize, forced-choice genre is realizable.
(B) HONEST router comparison: select the best item/concept router on SEL, evaluate on EVA (de-inflate the
swept max), vs genre. Both sides same discipline.
"""
import os,sys,json
import numpy as np
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
from signed_latent import load_arena_base
from reconciled import ConceptBank
import arena_core as AC
def log(m): print(m,flush=True)
RS="C:/dev/phd/casper/.cache/rich_signal"; META="C:/dev/phd/casper/data/movielens/.cache/ml25m/meta.npz"; OFF_ITEM=1628
GENRES=['action','adventure','animation','comedy','crime','documentary','drama','family','fantasy','horror','musical','mystery','noir','romance','sci-fi','thriller','war','western']
base=load_arena_base(); ni=base["ni"]; head=base["headmask"]; cnt=base["cnt"]
d=np.load(META); uu=d["uu"].astype(np.int64); ii=d["ii"].astype(np.int64); rr=d["rr"].astype(np.float64)
o=np.argsort(uu,kind="stable"); uu,ii,rr=uu[o],ii[o],rr[o]; bnd=np.searchsorted(uu,np.arange(uu[-1]+2))
cb=ConceptBank(ni,cnt); cname=[str(x).lower() for x in cb.names]; gcol=[cname.index(g) if g in cname else -1 for g in GENRES]
tq=json.load(open("C:/dev/phd/casper/.cache/instrument2/tag_questions.json"))["tags"]
memb=json.load(open("C:/dev/phd/casper/.cache/instrument2/tag_membership.json"))["membership"]
gmem=[set(int(j) for j in memb.get(str(next(t["tagId"] for t in tq if str(t.get('tag')).lower()==g)),[]) if 0<=j<ni) for g in GENRES]
uids=np.load(f"{RS}/mm_val_uids.npy"); V=np.load(f"{RS}/mm_val_val.npy")
favw,keep=[],[]
for r,u in enumerate(uids):
    a,b=bnd[int(u)],bnd[int(u)+1]; its,rat=ii[a:b],rr[a:b]
    if len(its)<8: continue
    ru=np.random.default_rng(AC.SEED*1_000_003+int(u)); perm=ru.permutation(len(its)); half=len(its)//2
    kn={int(its[k]):float(rat[k]) for k in perm[:half]}
    hl=set(int(its[k]) for k in perm[half:] if rat[k]>=4); hlt=set(j for j in hl if not head[j])
    if len(kn)<4 or not hlt: continue
    kset=set(kn); cnts=[len(kset&m) for m in gmem]; favw.append(int(np.argmax(cnts)) if max(cnts)>0 else -1); keep.append(r)
keep=np.array(keep); V=V[keep]; N=len(keep); favw=np.array(favw)
ND=np.load(".cache/arena_ND_tail.npy"); assert len(ND)==N
gv=np.stack([V[:,c] if c>=0 else np.full(N,-1) for c in gcol],1)     # (N,18) answerer genre ordinals
gansw=np.where(gv.max(1)>=0, gv.argmax(1), -1)
# strict-max: unique argmax AND top exceeds 2nd by >=1 level
srt=np.sort(gv,1); strictmax=(gv.max(1)>=0)&((srt[:,-1]-srt[:,-2])>=1)
rng=np.random.default_rng(0); perm=rng.permutation(N); SEL,EVA=perm[:N//2],perm[N//2:]
gq=int(np.argmax(ND[SEL].mean(0))); STATIC=float(ND[EVA,gq].mean()); log(f"STATIC tail {STATIC:.4f}  users {N}")
def adaptive_eva(labels):  # q2 per cluster on SEL, scored on EVA; prize vs STATIC
    num=den=0.0
    for c in np.unique(labels):
        sc=SEL[labels[SEL]==c]; ec=EVA[labels[EVA]==c]
        if len(sc)<20 or len(ec)<20: continue
        bq=int(np.argmax(ND[sc].mean(0))); num+=ND[ec,bq].mean()*len(ec); den+=len(ec)
    return num/max(den,1)-STATIC
log("="*66)
log("(A) MARGIN DECOMPOSITION (genre, 18-cell partition)")
p_impl=adaptive_eva(favw); p_answ=adaptive_eva(gansw)
log(f"  implicit(most-watched)  prize {p_impl:+.4f}")
log(f"  answerer(argmax ord.)   prize {p_answ:+.4f}")
agree=(gansw==favw)&(favw>=0)
log(f"  answerer==implicit agreement: {agree.mean():.3f}")
# per-capita implicit prize on strict-max subset vs all (route strict-max users by answerer, else static)
def prize_on(mask):  # route masked users by answerer genre, unmasked -> global static q2; EVA prize
    lab=np.where(mask,gansw,999)   # 999 = a single 'unrouted' bucket -> gets static-ish
    return adaptive_eva(lab.astype(np.int64))
log(f"  strict-max users: {strictmax.mean():.3f} of cohort")
p_strict=adaptive_eva(np.where(strictmax,gansw,999).astype(np.int64))
p_diffuse=adaptive_eva(np.where((~strictmax)&(gansw>=0),gansw,999).astype(np.int64))
# per-capita recovery: prize among strict-max users routed by answerer, vs implicit prize among same users
sm_e=EVA[strictmax[EVA]]
impl_sm=adaptive_eva(np.where(strictmax,favw,999).astype(np.int64))
log(f"  prize from STRICT-MAX users (answerer-routed): {p_strict:+.4f}")
log(f"  prize from same STRICT-MAX users (implicit-routed): {impl_sm:+.4f}")
rec=p_strict/impl_sm if impl_sm>1e-9 else float('nan')
log(f"  >>> RECOVERY on strict-max = {rec:.2f}  (PRE-REGISTERED: >=0.60 => forced-choice genre REALIZABLE) <<<")
log(f"  (diffuse-answer users prize {p_diffuse:+.4f})")
log("="*66)
log("(B) HONEST router: select on SEL, eval on EVA (de-inflated). Fair vs swept-max.")
S2=SEL[:len(SEL)//2]; S1=SEL[len(SEL)//2:]   # S1 selects router, S2 selects q2, EVA evals
def eva_prize_router(labels):  # q2 per cluster on S2, eval EVA
    num=den=0.0
    for c in np.unique(labels):
        s2=S2[labels[S2]==c]; ec=EVA[labels[EVA]==c]
        if len(s2)<20 or len(ec)<20: continue
        bq=int(np.argmax(ND[s2].mean(0))); num+=ND[ec,bq].mean()*len(ec); den+=len(ec)
    return num/max(den,1)-STATIC
def sel_score_router(labels):  # router quality on S1 (q2 on S2 scored on S1)  -- crude but consistent
    num=den=0.0
    for c in np.unique(labels):
        s2=S2[labels[S2]==c]; s1=S1[labels[S1]==c]
        if len(s2)<20 or len(s1)<20: continue
        bq=int(np.argmax(ND[s2].mean(0))); num+=ND[s1,bq].mean()*len(s1); den+=len(s1)
    return num/max(den,1)
def honest_best(cols):
    best=(-9,None)
    for ci in cols:
        s=sel_score_router(V[:,ci].astype(np.int64))
        if s>best[0]: best=(s,ci)
    return best[1], eva_prize_router(V[:,best[1]].astype(np.int64))
it_c,it_p=honest_best(range(OFF_ITEM,OFF_ITEM+800)); log(f"  honest best ITEM router: EVA prize {it_p:+.4f}")
cc_c,cc_p=honest_best(range(0,OFF_ITEM)); log(f"  honest best CONCEPT router: {cb.names[cc_c]} EVA prize {cc_p:+.4f} (memb {len(cb.Mbin[cc_c].indices)})")
log(f"  genre implicit(EVA) {eva_prize_router(favw):+.4f}   genre answerer(EVA) {eva_prize_router(gansw.astype(np.int64)):+.4f}")
log("done")
