"""Follow-up to adaptive_arena: reuse cached TAIL-ND. Two tests:
(1) CLEAN GENRE: route by the user's highest-MEAN-RATED genre in the known half (a forced-choice favourite a
    real user CAN state) -> does it recover the implicit +0.022, or stay at the answerer +0.0047? Distinguishes
    'bad answerer simulation' from 'behavioral genre is fundamentally richer than a stated answer'.
(2) ENTITY OVERFIT GATE: re-sweep concepts requiring BOTH SEL and EVA to have >=MINC answerers in >=2 non-refuse
    cells -> do the niche-entity winners survive or collapse to noise?
"""
import os,sys,json
import numpy as np
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
from signed_latent import load_arena_base
from reconciled import ConceptBank
import arena_core as AC
def log(m): print(m,flush=True)
RS="C:/dev/phd/casper/.cache/rich_signal"; ITEM_LISTS="C:/dev/phd/casper/.cache/instrument2/item_lists.json"
META="C:/dev/phd/casper/data/movielens/.cache/ml25m/meta.npz"; OFF_ITEM=1628
GENRES=['action','adventure','animation','comedy','crime','documentary','drama','family','fantasy','horror','musical','mystery','noir','romance','sci-fi','thriller','war','western']
base=load_arena_base(); ni=base["ni"]; head=base["headmask"]; cnt=base["cnt"]
d=np.load(META); uu=d["uu"].astype(np.int64); ii=d["ii"].astype(np.int64); rr=d["rr"].astype(np.float64)
o=np.argsort(uu,kind="stable"); uu,ii,rr=uu[o],ii[o],rr[o]; bnd=np.searchsorted(uu,np.arange(uu[-1]+2))
cb=ConceptBank(ni,cnt); cname=[str(x).lower() for x in cb.names]
tq=json.load(open("C:/dev/phd/casper/.cache/instrument2/tag_questions.json"))["tags"]
memb=json.load(open("C:/dev/phd/casper/.cache/instrument2/tag_membership.json"))["membership"]
gmem=[]
for g in GENRES:
    tid=next(t["tagId"] for t in tq if str(t.get("tag")).lower()==g); gmem.append(set(int(j) for j in memb.get(str(tid),[]) if 0<=j<ni))
uids=np.load(f"{RS}/mm_val_uids.npy"); V=np.load(f"{RS}/mm_val_val.npy")
favw,favr,keep=[],[],[]
for r,u in enumerate(uids):
    a,b=bnd[int(u)],bnd[int(u)+1]; its,rat=ii[a:b],rr[a:b]
    if len(its)<8: continue
    ru=np.random.default_rng(AC.SEED*1_000_003+int(u)); perm=ru.permutation(len(its)); half=len(its)//2
    kn={int(its[k]):float(rat[k]) for k in perm[:half]}
    hl=set(int(its[k]) for k in perm[half:] if rat[k]>=4); hlt=set(j for j in hl if not head[j])
    if len(kn)<4 or not hlt: continue
    kset=set(kn); cnts=[len(kset&m) for m in gmem]
    favw.append(int(np.argmax(cnts)) if max(cnts)>0 else -1)
    # highest-MEAN-RATED genre in known half (a clean forced-choice favourite)
    means=[np.mean([kn[i] for i in kset&m]) if (kset&m) else -9 for m in gmem]
    favr.append(int(np.argmax(means)) if max(means)>-9 else -1)
    keep.append(r)
keep=np.array(keep); V=V[keep]; N=len(keep); favw=np.array(favw); favr=np.array(favr)
ND=np.load(".cache/arena_ND_tail.npy"); assert len(ND)==N, f"ND {len(ND)} vs users {N}"
log(f"[arena2] users={N}  ND={ND.shape}")
rng=np.random.default_rng(0); perm=rng.permutation(N); SEL,EVA=perm[:N//2],perm[N//2:]
gq=int(np.argmax(ND[SEL].mean(0))); STATIC=float(ND[EVA,gq].mean()); log(f"[arena2] STATIC tail={STATIC:.4f}")
def prize(labels,minc=20,min_nonrefuse_cells=0):
    num=den=0.0; ncl=0; nonref=0
    for c in np.unique(labels):
        sc=SEL[labels[SEL]==c]; ec=EVA[labels[EVA]==c]
        if len(sc)<minc or len(ec)<minc: continue
        if c>=0: nonref+=1
        bq=int(np.argmax(ND[sc].mean(0))); num+=ND[ec,bq].mean()*len(ec); den+=len(ec); ncl+=1
    if nonref<min_nonrefuse_cells: return None
    return num/max(den,1)-STATIC
log(f"[test1 CLEAN GENRE]")
log(f"  genre most-WATCHED (implicit)   prize {prize(favw):+.4f}")
log(f"  genre highest-RATED (forced fav) prize {prize(favr):+.4f}   <- realizable clean favourite")
# test2: robust concept sweep
for MINC in (20,60,120):
    best=(-9,None); nvalid=0
    for ci in range(OFF_ITEM):
        p=prize(V[:,ci].astype(np.int64),minc=MINC,min_nonrefuse_cells=2)
        if p is None: continue
        nvalid+=1
        if p>best[0]: best=(p,ci)
    lab=cb.names[best[1]] if best[1] is not None else '?'
    log(f"[test2 concept robust] MINC={MINC:>3} (>=2 nonrefuse cells): valid={nvalid:4d}  BEST {best[0]:+.4f} = {lab} (memb {len(cb.Mbin[best[1]].indices) if best[1] is not None else 0})")
log("done")
