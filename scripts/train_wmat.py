"""Phase-1: train the per-value matrix W(v) on the CONCEPT branch. Concepts = FROZEN whitened member-centroids
(popularity-stripped); items keep frozen FiLM; decoder/attention/embeddings all frozen -> full-profile preserved
STRUCTURALLY. Trains ONLY Wv/bv/cv (~1.3M). Multi-combo STRATEGY curriculum (author): per user, several folds
selected by strategy (random / popular / niche / strongest) for items, concepts, and combos -- so the fold sees
STRONG questions, not just random. Leak-free: selection + answers from the KNOWN half, target = HELD half.
PT-1 (EPOCHS=1, FREEZE_ATTN=1) is the gate; then probes (sign-flip specificity, cold k-curve).
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try: sys.stdout.reconfigure(encoding='utf-8')
except Exception: pass
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F, json
import set_mn as S
from set_mn import SetEncoder, RSD
from signed_latent import load_arena_base, ndcg10
from reconciled import ConceptBank
import arena_core as AC
def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)
torch.set_num_threads(os.cpu_count())
S.set_grading("ordinal"); NC=S.NC; NLEV=S.NLEV; REFUSE=S.LV_REFUSE
META="C:/dev/phd/casper/data/movielens/.cache/ml25m/meta.npz"; OUT="C:/dev/phd/casper/.cache/set_mn"; LO=4.0; D=512
BASE=os.environ.get("BASE","paord_best"); TAG=os.environ.get("TAG","wmat"); EPOCHS=int(os.environ.get("EPOCHS","1"))
log(f"grading=ordinal | base={BASE} tag={TAG} epochs={EPOCHS}")

base=load_arena_base(); ni=base['ni']; head=base['headmask']; headarr=np.where(head)[0]; cnt=base['cnt']
d=np.load(META); uu=d['uu'].astype(np.int64); ii=d['ii'].astype(np.int64); rr=d['rr'].astype(np.float64)
o=np.argsort(uu,kind='stable'); uu,ii,rr=uu[o],ii[o],rr[o]; bnd=np.searchsorted(uu,np.arange(uu[-1]+2))
uidsT=np.load(RSD+'/mm_train_uids.npy'); KT=np.load(RSD+'/mm_train_know.npy'); VT=np.load(RSD+'/mm_train_val.npy')
ANS=(KT[:,:NC]>=1)&(VT[:,:NC]>=0); CVAL=np.clip(VT[:,:NC],0,3).astype(np.int64); CKN=np.clip(KT[:,:NC],0,2).astype(np.int64)

# ---- whitened concept centroids (mean + top-1 PC removed), FROZEN ----
pa=torch.load(os.path.join(OUT,BASE+'.pt'),map_location='cpu'); IE=pa['student']['item_emb.weight'].numpy()
Xc=IE-IE.mean(0); U,Sg,Vt=np.linalg.svd(Xc,full_matrices=False); V1=Vt[:1]; Xw=Xc-(Xc@V1.T)@V1
cb=ConceptBank(ni,cnt); Mbin=cb.Mbin.tocsr() if hasattr(cb.Mbin,'tocsr') else cb.Mbin
Cw=np.zeros((NC,512),np.float32); csz=np.zeros(NC)
for c in range(NC):
    idx=Mbin[c].indices; csz[c]=len(idx)
    if len(idx)>=20:
        v=Xw[idx].mean(0); n=np.linalg.norm(v)
        if n>0: Cw[c]=v/n
np.save(f"{OUT}/whiten_mu.npy", IE.mean(0)); np.save(f"{OUT}/whiten_pc1.npy", V1)   # ship for OOD channels
log(f"whitened concept centroids built ({(csz>=20).sum()} usable); whiten artifacts saved")

# ---- per-user known-half items(+rating) and held-half liked targets ----
rows=[]; TGT=[]; KIT=[]
for r,uid in enumerate(uidsT):
    a,b=bnd[int(uid)],bnd[int(uid)+1]; its,rat=ii[a:b],rr[a:b]
    if len(its)<8: continue
    ru=np.random.default_rng(AC.SEED*1_000_003+int(uid)); p=ru.permutation(len(its)); h=len(its)//2
    ki,kr=its[p[:h]],rat[p[:h]]; hi,hr=its[p[h:]],rat[p[h:]]; hl=hi[hr>=LO]
    if len(ki)<4 or len(hl)==0: continue
    rows.append(r); TGT.append(hl.astype(np.int64)); KIT.append((ki.astype(np.int64),kr.astype(np.float32)))
rows=np.array(rows,np.int64)
log(f"usable users {len(rows)}")

# ---- PRECOMPUTE per-user strategy orders ONCE (lever 2: no per-fold sorting) ----
itempop=cnt.astype(np.float32)
PRE=[]   # per-user: (ki, krlv, pop_desc_idx, niche_idx, strong_idx, nz, nz_by_size_desc)
for i in range(len(rows)):
    r=rows[i]; ki,kr=KIT[i]; krlv=S.sv_to_level((kr-2.75)/2.25).astype(np.int64)
    pd=np.argsort(-itempop[ki]); nd=pd[::-1].copy(); sd=np.argsort(-kr)
    nz=np.where(ANS[r])[0]; nzs=nz[np.argsort(-csz[nz])] if len(nz) else nz
    PRE.append((ki,krlv,pd.astype(np.int32),nd.astype(np.int32),sd.astype(np.int32),nz.astype(np.int32),nzs.astype(np.int32)))
log("per-user strategy orders precomputed")

# ---- model ----
enc=SetEncoder(ni+NC, token_mode='wmat', pool='attn', nlev=NLEV, nknow=3, n_items=ni)
with torch.no_grad():
    enc.item_emb.weight[:ni].copy_(pa['student']['item_emb.weight'])
    enc.item_emb.weight[ni:ni+NC].copy_(torch.from_numpy(Cw))
    enc.load_state_dict({k:v for k,v in pa['student'].items() if not k.startswith('item_emb')}, strict=False)
dec=nn.Linear(D,ni); dec.load_state_dict(pa['decoder']); Wd=dec.weight.detach(); bd=dec.bias.detach()
for p_ in enc.parameters(): p_.requires_grad_(False)
for name in ['Wv','bv']:
    getattr(enc,name).requires_grad_(True)
enc.cv.weight.requires_grad_(True)
train_params=[enc.Wv, enc.bv, enc.cv.weight]
opt=torch.optim.AdamW(train_params, lr=3e-4, weight_decay=1e-4)
ITEM_SNAP=enc.item_emb.weight[:ni].detach().clone()
log(f"trainable params: Wv/bv/cv = {sum(p.numel() for p in train_params)}")

def sv2lv(rat): return S.sv_to_level((rat-2.75)/2.25)
def decode(z): return (z@Wd.T+bd).numpy().astype(np.float64)
def enc_seqs(seqs):
    B=len(seqs); Lm=max(len(s[0]) for s in seqs)
    ids=np.zeros((B,Lm),np.int64); lv=np.zeros((B,Lm),np.int64); kn=np.zeros((B,Lm),np.int64); pad=np.ones((B,Lm),bool)
    for i,(a,l,k) in enumerate(seqs):
        ids[i,:len(a)]=a; lv[i,:len(a)]=l; kn[i,:len(a)]=k; pad[i,:len(a)]=False
    return enc(torch.from_numpy(ids),torch.zeros(B,Lm),torch.from_numpy(pad),torch.from_numpy(lv),torch.from_numpy(kn))

# ---- STRATEGY multi-combo curriculum (FAST: precomputed orders, no per-fold sorting) ----
def klog(rng, hi=20): return int(np.exp(rng.uniform(0, np.log(hi))))
def combos_for(i, rng):
    r=rows[i]; ki,krlv,pd,nd,sd,nz,nzs=PRE[i]; out=[]; ln=len(ki); nn=len(nz)
    def itemfold(order):
        k=min(klog(rng),ln); sel=order[:k]; return (ki[sel], krlv[sel], np.full(k,2,np.int64))
    if ln:
        for order in (pd, nd, sd): out.append(itemfold(order))                  # pop, niche, strong
        rk=min(klog(rng),ln); ridx=rng.choice(ln,rk,replace=False); out.append((ki[ridx],krlv[ridx],np.full(rk,2,np.int64)))  # random
    if nn:
        for order in (nzs, nzs[::-1]):                                           # size, niche(small)
            k=min(klog(rng),nn); sel=order[:k].astype(np.int64)
            extra=rng.choice(NC,max(1,k//3),replace=False)                       # + random bank -> refusals natural
            cids=np.concatenate([sel,extra]); lv=np.where(ANS[r,cids],CVAL[r,cids],REFUSE); kn=np.where(ANS[r,cids],CKN[r,cids],0)
            out.append((ni+cids, lv.astype(np.int64), kn.astype(np.int64)))
        k=min(klog(rng),nn); sel=nz[rng.choice(nn,k,replace=False)].astype(np.int64)  # random concepts
        out.append((ni+sel, CVAL[r,sel], CKN[r,sel]))
    if ln and nn:                                                               # 2 combo folds
        for _ in range(2):
            ka=min(klog(rng,10),ln); kc=min(klog(rng,10),nn)
            si=pd[:ka]; sc=nzs[:kc].astype(np.int64)                            # popular items + big concepts
            ids=np.concatenate([ki[si], ni+sc]); lv=np.concatenate([krlv[si], CVAL[r,sc]]).astype(np.int64)
            kn=np.concatenate([np.full(ka,2,np.int64), CKN[r,sc]]).astype(np.int64)
            out.append((ids.astype(np.int64), lv, kn))
    return out

# ---- probes ----
tag=[t.get('tag','?') for t in json.load(open('.cache/instrument2/tag_questions.json'))['tags']]
allit=np.arange(ni); prng=np.random.default_rng(0)
def probes():
    enc.eval()
    with torch.no_grad():
        z0=enc(torch.zeros(1,1,dtype=torch.long),torch.zeros(1,1),torch.ones(1,1,dtype=torch.bool),torch.zeros(1,1,dtype=torch.long),torch.zeros(1,1,dtype=torch.long))
    sc0=decode(z0)[0]
    def fold(cid,lv):
        with torch.no_grad(): z=enc_seqs([(np.array([ni+cid]),np.array([lv]),np.array([2]))])
        return decode(z)[0]
    log("  sign-flip specificity (loved vs hated on member films vs unrelated):")
    for cn in ['horror','sci fi','romance','comedy']:
        c=tag.index(cn); mem=Mbin[c].indices; non=prng.choice(np.setdiff1d(allit,mem),2000,replace=False)
        sl=fold(c,3); sh=fold(c,0)
        dm=(sl[mem]-sh[mem]).mean(); dn=(sl[non]-sh[non]).mean()
        log(f"    {cn:<9} loved-hated members {dm:+.3f} unrelated {dn:+.3f} ratio {dm/max(abs(dn),1e-9):+.1f}x")
    enc.train()

# ---- train ----
order=np.arange(len(rows)); CHUNK=384; MB=512
for ep in range(1,EPOCHS+1):
    enc.train(); rng=np.random.default_rng(100+ep); rng.shuffle(order); t0=time.time(); run=0.0; nb=0
    for b in range(0,len(order),CHUNK):
        seqs=[]; owners=[]
        for i in order[b:b+CHUNK]:
            for f in combos_for(i,rng):
                seqs.append(f); owners.append(i)
        oL=sorted(range(len(seqs)), key=lambda j: len(seqs[j][0]))           # LENGTH-BUCKET -> minimal padding
        for mb in range(0,len(oL),MB):
            idx=oL[mb:mb+MB]; ss=[seqs[j] for j in idx]; ow=[owners[j] for j in idx]
            z=enc_seqs(ss); logits=z@Wd.T+bd
            tgt=torch.zeros(len(ss),ni)
            for j,i in enumerate(ow): tgt[j,TGT[i]]=1.0
            nll=-((F.log_softmax(logits,-1)*tgt).sum(-1)/tgt.sum(-1).clamp_min(1)).mean()
            opt.zero_grad(); nll.backward(); opt.step(); run+=float(nll); nb+=1
        if b % (CHUNK*8) == 0 and b > 0:
            log(f"  ep{ep} u{b}/{len(order)} NLL={run/max(nb,1):.4f} {(time.time()-t0)/60:.1f}m")
        if b % (CHUNK*20) == 0 and b > 0:                                    # MID-EPOCH probe -> early gate read
            probes()
    drift=float((enc.item_emb.weight[:ni]-ITEM_SNAP).abs().max())
    log(f"[ep{ep}] NLL={run/max(nb,1):.4f} item-drift={drift:.2e} ({(time.time()-t0)/60:.1f}m)")
    torch.save({'student':enc.state_dict(),'decoder':dec.state_dict(),'epoch':ep,'grading':'ordinal'}, f"{OUT}/{TAG}_ep{ep}.pt")
    probes()
log("done")
