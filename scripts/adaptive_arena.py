"""ROUTER ARENA (Fable-designed kill-shot): in a 2-question interview, does a CONCEPT-q1 route better than
an ITEM-q1? Both q1 = ROUTE-ONLY (cluster users by the answer), q2 = fold one bank item (ND_tail[u,q2]).
Perfectly fair: identical fold budget, only the router differs. TAIL@10 primary (HARD RULE #2), FULL secondary.
Honesty: router chosen on SEL half, per-cluster q2 chosen on SEL, both evaluated on EVA. All val users, all
800 items, all 1628 concepts as candidate routers (cached-ND numpy sweep). No sampling.

Arms:
  2a genre-IMPLICIT (watch-history favourite genre; zero-noise UPPER BOUND)
  2b genre-ANSWERER (route by answerer's genre-concept ordinal; REALIZABLE — the honest arm)
  3  every one of 800 ITEMS as router (histogram of prizes)
  C  every one of 1628 CONCEPTS as router (histogram; the surviving style/era/tone axes live here)
  control: genre coarsened to 5 cells + effective-cluster-count exp(entropy) for both channels.
"""
import os, sys, json
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from signed_latent import load_arena_base, scale_rating, SignedAE, ndcg10
from reconciled import ConceptBank
import arena_core as AC
def log(m): print(m, flush=True)
A0C="C:/dev/phd/casper/.cache/signed_latent/a0c_best.pt"; RS="C:/dev/phd/casper/.cache/rich_signal"
ITEM_LISTS="C:/dev/phd/casper/.cache/instrument2/item_lists.json"; META="C:/dev/phd/casper/data/movielens/.cache/ml25m/meta.npz"
OFF_ITEM=1628
GENRES=['action','adventure','animation','comedy','crime','documentary','drama','family','fantasy','horror','musical','mystery','noir','romance','sci-fi','thriller','war','western']

base=load_arena_base(); ni=base["ni"]; head=base["headmask"]; cnt=base["cnt"]
d=np.load(META); uu=d["uu"].astype(np.int64); ii=d["ii"].astype(np.int64); rr=d["rr"].astype(np.float64)
o=np.argsort(uu,kind="stable"); uu,ii,rr=uu[o],ii[o],rr[o]; bnd=np.searchsorted(uu,np.arange(uu[-1]+2))
bank=np.array([int(x) for x in json.load(open(ITEM_LISTS))["lists"]["top800"]["ids"]],np.int64)
cb=ConceptBank(ni,cnt); cname=[str(x).lower() for x in cb.names]
gcol=[cname.index(g) if g in cname else -1 for g in GENRES]   # genre concept COLUMN indices (0..1627)
log(f"[arena] ni={ni} bank={len(bank)} genre-cols found={sum(1 for x in gcol if x>=0)}/18")

uids=np.load(f"{RS}/mm_val_uids.npy"); K=np.load(f"{RS}/mm_val_know.npy"); V=np.load(f"{RS}/mm_val_val.npy"); CR=np.load(f"{RS}/mm_val_crval.npy")
tq=json.load(open("C:/dev/phd/casper/.cache/instrument2/tag_questions.json"))["tags"]
memb=json.load(open("C:/dev/phd/casper/.cache/instrument2/tag_membership.json"))["membership"]
gmem=[]
for g in GENRES:
    tid=next(t["tagId"] for t in tq if str(t.get("tag")).lower()==g)
    gmem.append(set(int(j) for j in memb.get(str(tid),[]) if 0<=j<ni))

held,kmean,keep,favg=[],[],[],[]
for r,u in enumerate(uids):
    a,b=bnd[int(u)],bnd[int(u)+1]; its,rat=ii[a:b],rr[a:b]
    if len(its)<8: continue
    ru=np.random.default_rng(AC.SEED*1_000_003+int(u)); perm=ru.permutation(len(its)); half=len(its)//2
    kn={int(its[k]):float(rat[k]) for k in perm[:half]}
    hl=set(int(its[k]) for k in perm[half:] if rat[k]>=4)
    hlt=set(j for j in hl if not head[j])           # TAIL held-liked
    if len(kn)<4 or not hlt: continue
    kset=set(kn); cnts=[len(kset&m) for m in gmem]
    favg.append(int(np.argmax(cnts)) if max(cnts)>0 else -1)
    held.append(hlt); kmean.append(float(np.mean(list(kn.values())))); keep.append(r)
keep=np.array(keep,np.int64); kmean=np.array(kmean); K=K[keep]; V=V[keep]; CR=CR[keep]; N=len(keep)
fav=np.array(favg,np.int64)
log(f"[arena] usable users={N}")

t=SignedAE(ni,use_mask=True); t.load_state_dict(torch.load(A0C,map_location="cpu")["model"]); t.eval()
Wd=t.decoder.weight.detach(); bd=t.decoder.bias.detach()
stars=np.arange(0.5,5.01,0.5); TOPN=25
tops=np.zeros((len(bank),len(stars),TOPN),np.int64)
with torch.no_grad():
    z0=t.encode(torch.zeros((1,ni))); s0=(z0@Wd.T+bd)[0].clone(); s0[torch.from_numpy(head)]=-1e30
    cold_top=torch.topk(s0,TOPN).indices.numpy()
    for bi,item in enumerate(bank):
        xs=torch.zeros((len(stars),ni))
        for li,st in enumerate(stars): xs[li,int(item)]=float(scale_rating(st))
        sc=t.encode(xs)@Wd.T+bd; sc[:,torch.from_numpy(head)]=-1e30      # TAIL: mask head
        tops[bi]=torch.topk(sc,TOPN,dim=-1).indices.numpy()
        if bi%200==0: log(f"  [arena] tail-fold precomputed {bi}/{len(bank)}")
log("[arena] tail rankings precomputed")

# ordinal level -> star (from data), item star levels
kmv=kmean[:,None]; star_all=CR+kmv; lvl_all=V[:,OFF_ITEM:].astype(np.int64)
L2S=np.zeros(4)
for L in range(4):
    m=np.isfinite(star_all)&(lvl_all==L); L2S[L]=float(np.nanmean(star_all[m])) if m.any() else 3.0
disc=1.0/np.log2(np.arange(2,12))
ND=np.zeros((N,len(bank)),np.float32)          # TAIL ndcg after folding ONE bank item q2
for u in range(N):
    hl=held[u]; idcg=disc[:min(10,len(hl))].sum()
    cold_nd=sum(disc[r] for r,it in enumerate([x for x in cold_top][:10]) if it in hl)/idcg
    krow=K[u,OFF_ITEM:]; vrow=V[u,OFF_ITEM:]; ans=(krow>=1)&(vrow>=0)
    sr=star_all[u]; st=np.where(np.isfinite(sr),sr,L2S[np.clip(vrow,0,3)]); st=np.clip(np.rint(st*2)/2,0.5,5.0)
    lv=np.clip(np.rint(st*2).astype(np.int64)-1,0,9)
    for bi in range(len(bank)):
        if not ans[bi]: ND[u,bi]=cold_nd; continue
        rank=[x for x in tops[bi,lv[bi]] if x!=int(bank[bi])][:10]
        ND[u,bi]=sum(disc[r] for r,it in enumerate(rank) if it in hl)/idcg
    if u%500==0: log(f"  [arena] tail-ND scored {u}/{N}")
np.save(".cache/arena_ND_tail.npy",ND)
log(f"[arena] TAIL cold intercept = {ND[:, :1].mean():.4f} (placeholder)  mean-ND={ND.mean():.4f}")

rng=np.random.default_rng(0); perm=rng.permutation(N); SEL,EVA=perm[:N//2],perm[N//2:]
gq=int(np.argmax(ND[SEL].mean(0))); STATIC=float(ND[EVA,gq].mean())
def eff_cells(labels):
    _,c=np.unique(labels[labels>-99],return_counts=True); p=c/c.sum(); return float(np.exp(-(p*np.log(p)).sum()))
def prize(labels, minc=20):
    num=den=0.0; ncl=0
    for c in np.unique(labels):
        sc=SEL[labels[SEL]==c]; ec=EVA[labels[EVA]==c]
        if len(sc)<minc or len(ec)<minc: continue
        bq=int(np.argmax(ND[sc].mean(0))); num+=ND[ec,bq].mean()*len(ec); den+=len(ec); ncl+=1
    adap=num/max(den,1); return adap-STATIC, adap, ncl, eff_cells(labels)
def sweep(colidx, tag):
    best=(-9,None); prizes=[]
    for ci in colidx:
        if ci<0: prizes.append(np.nan); continue
        lab=V[:,ci].astype(np.int64)            # 5 cells: -1 refuse / 0 hated / 1 meh / 2 liked / 3 loved
        p=prize(lab)[0]; prizes.append(p)
        if p>best[0]: best=(p,ci)
    prizes=np.array(prizes); return best, prizes

log("="*70); log(f"[arena] STATIC (single global best q2, tail) = {STATIC:.4f}")
p2a=prize(fav); log(f"[arm2a genre-IMPLICIT] prize {p2a[0]:+.4f}  adaptive {p2a[1]:.4f}  cells {p2a[2]} eff {p2a[3]:.1f}")
# arm2b: answerer favourite genre = argmax over genre concept ordinals (tie-break watch count via fav)
gv=np.stack([V[:,c] if c>=0 else np.full(N,-1) for c in gcol],1)   # (N,18) genre ordinals
gansw=np.where((gv.max(1)>=0), gv.argmax(1), fav)                  # answerer favourite genre; fallback implicit
p2b=prize(gansw.astype(np.int64)); log(f"[arm2b genre-ANSWERER realizable] prize {p2b[0]:+.4f} adaptive {p2b[1]:.4f} cells {p2b[2]} eff {p2b[3]:.1f}")
# genre coarsened to 5 (by size)
order=np.argsort([-len(m) for m in gmem]); g2c={int(order[i]):min(i//4,4) for i in range(18)}
favc=np.array([g2c.get(int(x),-1) if x>=0 else -1 for x in fav]); p2c=prize(favc)
log(f"[arm2a-coarse5] prize {p2c[0]:+.4f} adaptive {p2c[1]:.4f} cells {p2c[2]} eff {p2c[3]:.1f}")
(bi_p,bi_c),iprz=sweep(list(range(OFF_ITEM,OFF_ITEM+len(bank))),"item")
log(f"[arm3 ITEM sweep] BEST prize {bi_p:+.4f} @item-col {bi_c}  | p50 {np.nanpercentile(iprz,50):+.4f} p90 {np.nanpercentile(iprz,90):+.4f} p99 {np.nanpercentile(iprz,99):+.4f} frac>arm2b {np.nanmean(iprz>p2b[0]):.3f}")
(bc_p,bc_c),cprz=sweep(list(range(0,OFF_ITEM)),"concept")
log(f"[armC CONCEPT sweep] BEST prize {bc_p:+.4f} @concept {cb.names[bc_c] if bc_c is not None else '?'}  | p50 {np.nanpercentile(cprz,50):+.4f} p90 {np.nanpercentile(cprz,90):+.4f} p99 {np.nanpercentile(cprz,99):+.4f} frac>bestItem {np.nanmean(cprz>bi_p):.3f}")
np.savez(".cache/arena_prizes.npz", item=iprz, concept=cprz, arm2a=p2a[0], arm2b=p2b[0], static=STATIC)
# name the top concept routers
top=np.argsort(-np.nan_to_num(cprz,nan=-9))[:12]
log("[armC] top concept routers:")
for c in top: log(f"    {cb.names[c][:26]:<26} prize {cprz[c]:+.4f}  (memb {len(cb.Mbin[c].indices)})")
log("done")
