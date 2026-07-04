"""
Goodreads COMPOSITE (PHASE-1E) prep (EXPLICIT track). Union of THREE byGenre slices:
mystery_thriller_crime + fantasy_paranormal + history_biography.
STREAM-parse each interactions .gz ONCE (never fully decompress). Keep rating>0 (explicit),
like>=4 / dislike<=2. Users & books indexed GLOBALLY by their raw goodreads id strings, which are
shared across slices -> a user in multiple slices gets a MERGED cross-genre profile. Dedupe
(user,book) pairs across slices (a cross-shelved book appears in >1 file); keep first occurrence.
Item filter: catalogue = books with >=20 ratings (over the merged, deduped ratings). FULL user set.
Split: rng(0) shuffle, last 1000 -> 500 val / 500 test, rest train (mirrors ML-25M / mystery phase).
Reports: users/items/ratings, ratings/user quantiles, cross-genre user share, popularity concentration.
Saves .cache/goodreads/base_comp.npz.
"""
import os, time, gzip, json, numpy as np, array
GR='C:/dev/phd/casper/.cache/goodreads'
SLICES=[('mys','goodreads_interactions_mystery_thriller_crime.json.gz'),
        ('fan','goodreads_interactions_fantasy_paranormal.json.gz'),
        ('his','goodreads_interactions_history_biography.json.gz')]
LIKE=4.0; MINIT=20; t0=time.time()

umap={}; bmap={}
au=array.array('i'); ab=array.array('i'); ar=array.array('b'); asl=array.array('b')
n=0
for si,(tag,fn) in enumerate(SLICES):
    path=f'{GR}/{fn}'; nrow=0; nk=0
    print(f"--- slice {si} [{tag}] {fn} ---",flush=True)
    with gzip.open(path,'rt',encoding='utf-8') as f:
        for line in f:
            nrow+=1
            if nrow % 4000000 == 0:
                print(f"  [{tag}] scanned {nrow/1e6:.0f}M rows, kept(this) {nk/1e6:.2f}M, tot {n/1e6:.2f}M, users {len(umap)} books {len(bmap)} ({time.time()-t0:.0f}s)",flush=True)
            try: d=json.loads(line)
            except Exception: continue
            r=d.get('rating',0)
            if not r or r<=0: continue
            uid=d['user_id']; bid=d['book_id']
            ui=umap.get(uid)
            if ui is None: ui=len(umap); umap[uid]=ui
            bi=bmap.get(bid)
            if bi is None: bi=len(bmap); bmap[bid]=bi
            au.append(ui); ab.append(bi); ar.append(int(r)); asl.append(si); n+=1; nk+=1
    print(f"  [{tag}] DONE rows={nrow} kept={nk} tot_kept={n} users={len(umap)} books={len(bmap)} ({time.time()-t0:.0f}s)",flush=True)

U=np.frombuffer(au,dtype=np.int32).copy(); del au
Bk=np.frombuffer(ab,dtype=np.int32).copy(); del ab
R=np.frombuffer(ar,dtype=np.int8).astype(np.float32); del ar
SL=np.frombuffer(asl,dtype=np.int8).copy(); del asl
NB=len(bmap); NU_raw=len(umap)
bid_by_idx=np.zeros(NB,np.int64)
for bid,k in bmap.items(): bid_by_idx[k]=np.int64(bid)
del bmap; del umap
print(f"RAW: {n} rated rows (pre-dedup), {NU_raw} users, {NB} books ({time.time()-t0:.0f}s)",flush=True)

# ---- dedupe (user,book) across slices: keep first occurrence ----
key=U.astype(np.int64)*np.int64(NB)+Bk.astype(np.int64)
_,first=np.unique(key,return_index=True)   # first occurrence (original order) per unique pair
first.sort(); del key
ndup=n-len(first)
U=U[first]; Bk=Bk[first]; R=R[first]; SL=SL[first]
print(f"DEDUP: removed {ndup} duplicate (user,book) pairs across slices -> {len(R)} unique ratings ({time.time()-t0:.0f}s)",flush=True)

# ---- item filter: >=20 ratings (merged) ----
bcnt=np.bincount(Bk,minlength=NB)
keepB_local=np.nonzero(bcnt>=MINIT)[0]; ni=len(keepB_local)
imap=-np.ones(NB,np.int32); imap[keepB_local]=np.arange(ni)
mask=imap[Bk]>=0
U=U[mask]; ii=imap[Bk][mask]; R=R[mask]; SL=SL[mask]; del Bk
keepB=bid_by_idx[keepB_local]
# dense-remap surviving users
uniqU=np.unique(U); nu=len(uniqU)
umap2=np.zeros(int(uniqU.max())+1,np.int32); umap2[uniqU]=np.arange(nu); uu=umap2[U]; del U
print(f"CATALOGUE: {ni} items (>= {MINIT} ratings), {nu} users, {len(R)} ratings kept ({time.time()-t0:.0f}s)",flush=True)

ucnt=np.bincount(uu,minlength=nu)
print(f"ratings/user: median={np.median(ucnt):.0f} mean={ucnt.mean():.1f} min={ucnt.min()} max={ucnt.max()} "
      f"p25={np.percentile(ucnt,25):.0f} p75={np.percentile(ucnt,75):.0f} p90={np.percentile(ucnt,90):.0f}",flush=True)

# ---- cross-genre user share: users with kept ratings in >= 2 distinct slices ----
present=np.zeros((3,nu),bool)
for s in range(3):
    present[s, uu[SL==s]]=True
nsl=present.sum(0)
xg2=(nsl>=2).mean(); xg3=(nsl>=3).mean()
per_slice_users=[int(present[s].sum()) for s in range(3)]
print(f"cross-genre: users in >=2 slices = {xg2*100:.1f}% ({int((nsl>=2).sum())}/{nu}); >=3 slices = {xg3*100:.1f}% ({int((nsl>=3).sum())})",flush=True)
print(f"per-slice user counts (mys/fan/his) = {per_slice_users}",flush=True)
# per-slice item counts (a book's primary slice = the slice contributing most of its kept ratings)
prim=np.full(ni,-1,np.int8); sc_by_item=np.zeros((3,ni))
for s in range(3):
    np.add.at(sc_by_item[s], ii[SL==s], 1)
prim=np.argmax(sc_by_item,0).astype(np.int8)
print(f"per-slice item counts (primary; mys/fan/his) = {[int((prim==s).sum()) for s in range(3)]}",flush=True)
print(f"cross-genre ratings share: {(SL!=prim[ii]).mean()*100:.1f}% of kept ratings are on a book whose primary slice differs (cross-shelf overlap)",flush=True)

# ---- popularity concentration ----
icnt=np.bincount(ii,minlength=ni).astype(np.float64); order=np.argsort(-icnt); tot=icnt.sum()
share1=icnt[order[:max(1,int(round(0.01*ni)))]].sum()/tot
share01=icnt[order[:max(1,int(round(0.001*ni)))]].sum()/tot
sc=np.sort(icnt); cumc=np.cumsum(sc); gini=1-2*(cumc.sum()/(cumc[-1]*len(sc)))+1/len(sc)
print(f"popularity: top-1% items hold {share1*100:.1f}%; top-0.1% {share01*100:.1f}%; Gini={gini:.3f}",flush=True)

# ---- split ----
rng=np.random.default_rng(0); perm=rng.permutation(nu); hold=perm[-1000:]
va=hold[:500]; te=hold[500:]; trU=np.sort(perm[:-1000]); trU_mask=np.zeros(nu,bool); trU_mask[trU]=True
print(f"split: train {len(trU)} / val {len(va)} / test {len(te)}",flush=True)

# ---- popularity over TRAIN-user likes; mu; per-item 5-bin histogram ----
tr_row=trU_mask[uu]; like_row=(R>=LIKE)
cnt=np.bincount(ii[tr_row & like_row],minlength=ni).astype(np.float64); mu=float(R[tr_row].mean())
H5=np.zeros((ni,5)); b5=np.clip(R[tr_row].astype(int),1,5)-1; np.add.at(H5,(ii[tr_row],b5.astype(int)),1.0)

np.savez(f'{GR}/base_comp.npz', uu=uu.astype(np.int32), ii=ii.astype(np.int32), rr=R,
         cnt=cnt, mu=mu, ni=ni, nu=nu, trU=trU, va=va, te=te, keepB=keepB, H5=H5,
         prim=prim, slice_of_rating=SL.astype(np.int8))
print(f"SAVED base_comp.npz ni={ni} nu={nu} nratings={len(R)} ({time.time()-t0:.0f}s)",flush=True)
print("DONE",flush=True)
