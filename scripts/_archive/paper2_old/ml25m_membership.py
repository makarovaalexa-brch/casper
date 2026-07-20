"""
ML-25M PHASE-2 prep: build concept<->item membership + per-entity divisiveness/like-rate/count,
so the B-battery (concept-vs-item asking) and the continuous actor share ONE cache.

Reads meta.npz (uu,ii,rr,cnt,keepI) + Ac_concept/ctags_concept + genome-scores.csv (tag>0.5, matching
ml25m_concepts_full.py). Writes .cache/ml25m/membership.npz:
  citems_flat,citems_off : ragged concept -> dense item ids (>=30 tagged, same 1031 concepts as Ac)
  clike, ccount          : per-concept like-rate & #(user,concept opinions >=2 tagged rated)
  cdiv                   : per-concept divisiveness = binary entropy of clike
  ilike, icount, idiv    : per-item like-rate / #raters / binary-entropy divisiveness
  PITEMS                 : top-600 popular dense item ids (the askable item pool, matches concepts_full)
"""
import os, time, numpy as np
t0=time.time(); base='C:/dev/phd/casper/data/movielens'; out=f'{base}/.cache/ml25m'; LIKE=4.0
M=np.load(f'{out}/meta.npz')
uu=M['uu']; ii=M['ii']; rr=M['rr']; cnt=M['cnt']; keepI=M['keepI']; ni=int(M['ni'])
ctags=np.load(f'{out}/ctags_concept.npy'); nc=len(ctags); ctset={int(t):k for k,t in enumerate(ctags)}
iids={int(x):k for k,x in enumerate(np.sort(keepI))}; keepids=set(iids)
print(f"loaded meta: ni={ni} nc={nc} ({time.time()-t0:.0f}s)",flush=True)
# ---- membership from genome-scores (tag>0.5, only our 1031 concepts) ----
citems=[[] for _ in range(nc)]
with open(f'{base}/genome-scores.csv') as f:
    next(f)
    for line in f:
        a=line.split(','); m=int(a[0])
        if m in keepids:
            t=int(a[1])
            if t in ctset and float(a[2])>0.5: citems[ctset[t]].append(iids[m])
print(f"membership built ({time.time()-t0:.0f}s); mean items/concept={np.mean([len(c) for c in citems]):.0f}",flush=True)
citems_off=np.zeros(nc+1,np.int64)
for k in range(nc): citems_off[k+1]=citems_off[k]+len(citems[k])
citems_flat=np.concatenate([np.array(c,np.int32) for c in citems]) if nc else np.zeros(0,np.int32)
# ---- per-item like-rate / count / divisiveness ----
raters=np.bincount(ii,minlength=ni).astype(np.float64)
likes=np.bincount(ii[rr>=LIKE],minlength=ni).astype(np.float64)
ilike=likes/np.clip(raters,1,None); icount=raters
def Hb(p): p=np.clip(p,1e-6,1-1e-6); return -(p*np.log2(p)+(1-p)*np.log2(1-p))
idiv=Hb(ilike)
# ---- per-concept like-rate / divisiveness (aggregate over tagged items' ratings) ----
# concept like-rate = fraction of ratings on tagged items that are >=LIKE (global divisiveness signal)
clike=np.zeros(nc); ccount=np.zeros(nc)
for k in range(nc):
    its=citems[k]
    if not its: continue
    a=np.array(its); nlk=likes[a].sum(); nrt=raters[a].sum()
    clike[k]=nlk/max(nrt,1); ccount[k]=nrt
cdiv=Hb(clike)
order_pop=np.argsort(-cnt); PITEMS=order_pop[:600].astype(np.int32)
np.savez(f'{out}/membership.npz', citems_flat=citems_flat, citems_off=citems_off,
         clike=clike, ccount=ccount, cdiv=cdiv, ilike=ilike, icount=icount, idiv=idiv, PITEMS=PITEMS)
print(f"SAVED membership.npz ({time.time()-t0:.0f}s)",flush=True)
print(f"  concept divisiveness: mean {cdiv.mean():.3f}  median {np.median(cdiv):.3f}",flush=True)
print(f"  item(pool) divisiveness: mean {idiv[PITEMS].mean():.3f}",flush=True)
print("DONE",flush=True)
