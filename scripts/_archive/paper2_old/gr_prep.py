"""
Goodreads Mystery/Thriller/Crime PHASE-1 prep (EXPLICIT track).
STREAM-parse goodreads_interactions_mystery_thriller_crime.json.gz ONCE (never fully decompress).
Keep interactions with rating>0 (explicit). like = rating>=4, dislike <=2 (mirror ML semantics).
Item filter: catalog = books with >=20 ratings (stated preprocessing filter, NOT sampling).
FULL user set (no activity sampling). Split users: rng(0) shuffle, last 1000 -> 500 val / 500 test,
rest train (mirrors ML-25M). Report counts + popularity concentration (top-1% items' rating share).
Saves .cache/goodreads/base.npz.
"""
import os, time, gzip, json, numpy as np, array
GR='C:/dev/phd/casper/.cache/goodreads'
INTER=f'{GR}/goodreads_interactions_mystery_thriller_crime.json.gz'
LIKE=4.0; MINIT=20; t0=time.time()

# ---- single stream pass: collect all rating>0 (user,book,rating) triples ----
umap={}; bmap={}
au=array.array('i'); ab=array.array('i'); ar=array.array('b')
n=0; nrow=0
with gzip.open(INTER,'rt',encoding='utf-8') as f:
    for line in f:
        nrow+=1
        if nrow % 2000000 == 0:
            print(f"  scanned {nrow/1e6:.0f}M rows, kept {n/1e6:.2f}M rated, users {len(umap)} books {len(bmap)} ({time.time()-t0:.0f}s)",flush=True)
        try:
            d=json.loads(line)
        except Exception:
            continue
        r=d.get('rating',0)
        if not r or r<=0: continue
        uid=d['user_id']; bid=d['book_id']
        ui=umap.get(uid)
        if ui is None: ui=len(umap); umap[uid]=ui
        bi=bmap.get(bid)
        if bi is None: bi=len(bmap); bmap[bid]=bi
        au.append(ui); ab.append(bi); ar.append(int(r)); n+=1
print(f"PASS DONE: {nrow} rows, {n} rated interactions, {len(umap)} users, {len(bmap)} books ({time.time()-t0:.0f}s)",flush=True)

U=np.frombuffer(au,dtype=np.int32).copy(); del au
B=np.frombuffer(ab,dtype=np.int32).copy(); del ab
R=np.frombuffer(ar,dtype=np.int8).astype(np.float32); del ar
# book_id (numeric string) per raw dense book index -> keep for join to books.gz
bid_by_idx=np.zeros(len(bmap),np.int64)
for bid,k in bmap.items(): bid_by_idx[k]=np.int64(bid)
del bmap; del umap

# ---- item filter: >=20 ratings ----
bcnt=np.bincount(B,minlength=len(bid_by_idx))
keepB_local=np.nonzero(bcnt>=MINIT)[0]           # raw book indices kept
ni=len(keepB_local)
imap=-np.ones(len(bid_by_idx),np.int32); imap[keepB_local]=np.arange(ni)
mask=imap[B]>=0
U=U[mask]; ii=imap[B][mask]; R=R[mask]; del B
keepB=bid_by_idx[keepB_local]                    # dense item -> goodreads book_id (int64)
# dense-remap users that survive (all users survive if they have >=1 kept rating)
uniqU=np.unique(U); nu=len(uniqU)
umap2=np.zeros(int(uniqU.max())+1,np.int32); umap2[uniqU]=np.arange(nu); uu=umap2[U]; del U
print(f"catalog: {ni} items (>= {MINIT} ratings), {nu} users, {len(R)} ratings kept ({time.time()-t0:.0f}s)",flush=True)

ucnt=np.bincount(uu,minlength=nu)
print(f"ratings/user: median={np.median(ucnt):.0f} mean={ucnt.mean():.1f} min={ucnt.min()} max={ucnt.max()} "
      f"p10={np.percentile(ucnt,10):.0f} p25={np.percentile(ucnt,25):.0f} p75={np.percentile(ucnt,75):.0f} p90={np.percentile(ucnt,90):.0f}",flush=True)

# ---- popularity concentration: top-1% items' share of ratings ----
icnt_kept=np.bincount(ii,minlength=ni).astype(np.float64)
order=np.argsort(-icnt_kept); tot=icnt_kept.sum()
top1=order[:max(1,int(round(0.01*ni)))]
share1=icnt_kept[top1].sum()/tot
top01=order[:max(1,int(round(0.001*ni)))]
share01=icnt_kept[top01].sum()/tot
# gini
sc=np.sort(icnt_kept); cumc=np.cumsum(sc); gini=1-2*(cumc.sum()/(cumc[-1]*len(sc)))+1/len(sc)
print(f"popularity concentration: top-1% items ({len(top1)}) hold {share1*100:.1f}% of ratings; "
      f"top-0.1% hold {share01*100:.1f}%; Gini={gini:.3f}",flush=True)

# ---- split users (rng(0) shuffle, last 1000 -> 500 val/500 test) ----
rng=np.random.default_rng(0)
perm=rng.permutation(nu); hold=perm[-1000:]; va=hold[:500]; te=hold[500:]; trU=np.sort(perm[:-1000])
trU_mask=np.zeros(nu,bool); trU_mask[trU]=True
print(f"split: train {len(trU)} / val {len(va)} / test {len(te)}",flush=True)

# ---- popularity over TRAIN-user likes (for popb / selectors) ----
tr_row=trU_mask[uu]; like_row=(R>=LIKE)
cnt=np.bincount(ii[tr_row & like_row],minlength=ni).astype(np.float64)
mu=float(R[tr_row].mean())
# per-item rating histogram (train rows) for entropy selector: bins 1..5 -> 5 bins
H5=np.zeros((ni,5)); b5=np.clip(R[tr_row].astype(int),1,5)-1
np.add.at(H5,(ii[tr_row],b5.astype(int)),1.0)

np.savez(f'{GR}/base.npz', uu=uu.astype(np.int32), ii=ii.astype(np.int32), rr=R,
         cnt=cnt, mu=mu, ni=ni, nu=nu, trU=trU, va=va, te=te, keepB=keepB, H5=H5)
print(f"SAVED base.npz ni={ni} nu={nu} nratings={len(R)} ({time.time()-t0:.0f}s)",flush=True)
print("DONE",flush=True)
