"""
ML-25M PHASE-1B prep (FULL DATASET). Parse ratings.csv ONCE, apply item filter (>=20 ratings),
split users (rng(0) shuffle, last 1000 -> 500 val / 500 test, rest train), compute popularity/entropy,
and cache all arrays needed by the resumable trainer + eval. Reports counts + rating-count distribution.
Saves .cache/ml25m/prep.npz (and meta.npz for the concept step).
"""
import os, time, numpy as np, pandas as pd
base='C:/dev/phd/casper/data/movielens'; out=f'{base}/.cache/ml25m'; os.makedirs(out,exist_ok=True)
RCSV='C:/dev/phd/casper/data/movielens/ratings.csv'
LIKE=4.0; MINIT=20; rng=np.random.default_rng(0); t0=time.time()
df=pd.read_csv(RCSV,usecols=['userId','movieId','rating'],
               dtype={'userId':np.int32,'movieId':np.int32,'rating':np.float32})
U=df.userId.to_numpy(); I=df.movieId.to_numpy(); R=df.rating.to_numpy(); del df
print(f"loaded {len(R)} ratings ({time.time()-t0:.0f}s)",flush=True)
uniqI,icnt=np.unique(I,return_counts=True); keepI=np.sort(uniqI[icnt>=MINIT]); ni=len(keepI)
mask=np.isin(I,keepI); U=U[mask]; I=I[mask]; R=R[mask].astype(np.float32)
uniqU=np.unique(U); nu=len(uniqU)
umap=np.zeros(int(uniqU.max())+1,np.int32); umap[uniqU]=np.arange(nu)
imap=np.zeros(int(keepI.max())+1,np.int32); imap[keepI]=np.arange(ni)
uu=umap[U]; ii=imap[I]
print(f"catalog: {ni} items (>= {MINIT} ratings), {nu} users, {len(R)} ratings kept ({time.time()-t0:.0f}s)",flush=True)
ucnt=np.bincount(uu,minlength=nu)
print(f"ratings/user: median={np.median(ucnt):.0f} mean={ucnt.mean():.1f} min={ucnt.min()} max={ucnt.max()} "
      f"p10={np.percentile(ucnt,10):.0f} p25={np.percentile(ucnt,25):.0f} p75={np.percentile(ucnt,75):.0f} p90={np.percentile(ucnt,90):.0f}",flush=True)
perm=rng.permutation(nu); hold=perm[-1000:]; va=hold[:500]; te=hold[500:]; trU=np.sort(perm[:-1000])
trU_mask=np.zeros(nu,bool); trU_mask[trU]=True
print(f"split: train {len(trU)} / val {len(va)} / test {len(te)}",flush=True)
tr_row=trU_mask[uu]; like_row=(R>=LIKE)
cnt=np.bincount(ii[tr_row & like_row],minlength=ni).astype(np.float64)
# SGD train inputs
ru=uu[tr_row]; ri=ii[tr_row]; rr=R[tr_row]; mu=float(rr.mean())
truniq=np.unique(ru); trmap=np.zeros(nu,np.int32); trmap[truniq]=np.arange(len(truniq)); ruD=trmap[ru]; ntr=len(truniq)
# per-item rating histogram over train rows (for entropy selector)
H5=np.zeros((ni,10)); b5=np.clip(np.round(rr*2).astype(int),1,10)-1; np.add.at(H5,(ri,b5),1.0)
np.savez(f'{out}/prep.npz',ruD=ruD.astype(np.int32),ri=ri.astype(np.int32),rr=rr,mu=mu,ntr=ntr,ni=ni,nu=nu,
         cnt=cnt,H5=H5,uu=uu.astype(np.int32),ii=ii.astype(np.int32),R=R,
         va=va,te=te,trU=trU,keepI=keepI)
np.savez(f'{out}/meta.npz',uu=uu.astype(np.int32),ii=ii.astype(np.int32),rr=R,cnt=cnt,mu=mu,ni=ni,nu=nu,
         trU=trU,va=va,te=te,keepI=keepI)
print(f"SAVED prep.npz meta.npz  ntr={ntr} ({time.time()-t0:.0f}s)",flush=True)
print("DONE",flush=True)
