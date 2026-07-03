"""
ML-25M PHASE-1C: train the V1-style attention fold-in set-encoder on the FULL ML-25M split.
Mirrors scripts/paper2/encoder_recon.py (ML-1M V1): d=64 attention pooling over revealed
(Q[item], residual) tokens -> user vector u; FROZEN decoder popb + Q.u; IPS/tail-weighted BCE
reconstruction of the unrevealed like-set; masked/shuffled random-count reveals k=1..K.

RESUMABLE: keeps a checkpoint (enc weights + Adam state + epoch counter + best-val) so long
training can be chunked under the Bash timeout. Best-val selection on the 500-user val cohort
(encoder full-profile fold-in, NDCG@50 full catalogue). Durable:
  .cache/ml25m/enc_v1_ml25m.pt        (best-val encoder)
  .cache/ml25m/enc_v1_ml25m_state.pt  (resume state: last epoch enc+opt+counters)
  .cache/ml25m/enc_v1_ml25m_peak.txt  (durable peak log)

Env: EP_CHUNK (epochs to run this call, default 2), EP_MAX (total target epochs, default 20),
     K (max reveals, default 12), BATCH (default 512), SUB (uniform-random train subsample size;
     0 = full data, default 0). NEVAL_VAL (val users, default 500).
"""
import os, time, numpy as np, torch, torch.nn as nn
t0=time.time()
out='C:/dev/phd/casper/data/movielens/.cache/ml25m'
torch.set_num_threads(os.cpu_count() or 4)
D=64; LIKE=4.0
K=int(os.environ.get('K',12)); BATCH=int(os.environ.get('BATCH',512))
EP_CHUNK=int(os.environ.get('EP_CHUNK',2)); EP_MAX=int(os.environ.get('EP_MAX',20))
SUB=int(os.environ.get('SUB',0)); NEVAL_VAL=int(os.environ.get('NEVAL_VAL',500))
rng=np.random.default_rng(0); torch.manual_seed(0)

M=np.load(f'{out}/meta.npz')
uu=M['uu']; ii=M['ii']; rr=M['rr']; cnt=M['cnt']; mu=float(M['mu']); ni=int(M['ni'])
trU=M['trU']; va=M['va']; te=M['te']
Q=np.load(f'{out}/Q_svd.npy').astype(np.float32); bi=np.load(f'{out}/bi_svd.npy').astype(np.float32)
Qt=torch.tensor(Q)

# ---- CSR by user (sort rows by user id) ----
order=np.argsort(uu,kind='stable')
uu_s=uu[order]; ii_s=ii[order].astype(np.int32); rr_s=rr[order].astype(np.float32)
res_s=(rr_s-mu-bi[ii_s]).astype(np.float32)           # residual token feature
nu=int(M['nu'])
off=np.zeros(nu+1,np.int64); np.add.at(off,uu_s+1,1); off=np.cumsum(off)  # off[x]:off[x+1]
print(f"CSR built ({time.time()-t0:.0f}s); nu={nu} ni={ni} ratings={len(uu_s)}",flush=True)

# ---- priors (train-like popularity) ----
popb=np.log(cnt+1.0).astype(np.float32); popbt=torch.tensor(popb)
ipsw=(1.0/np.clip((cnt/max(cnt.max(),1))**0.5,1e-3,1.0)).astype(np.float32)
# head-33% by cumulative popularity (Cremonesi) -- used only for reporting parity; training weights use IPS
order_pop=np.argsort(-cnt); cum=np.cumsum(cnt[order_pop])/max(cnt.sum(),1)
HEAD=set(int(j) for j in order_pop[:np.searchsorted(cum,0.33)+1])
headmask=np.zeros(ni,bool); headmask[list(HEAD)]=True

# ---- train user pool (>= K+1 rated so a reveal count of K is realizable) ----
lens=(off[1:]-off[:-1]); trbig=np.array([x for x in trU if lens[x]>=K+1],dtype=np.int64)
if SUB>0 and SUB<len(trbig):
    pick=rng.choice(len(trbig),SUB,replace=False); trbig=np.sort(trbig[pick])
    print(f"[FALLBACK] uniform-random train subsample: {len(trbig)} users (full pool {int((lens[trU]>=K+1).sum())})",flush=True)
else:
    print(f"train users (>=K+1 rated): {len(trbig)} / {len(trU)} (FULL DATA)",flush=True)

class Enc(nn.Module):
    def __init__(s):
        super().__init__(); s.inp=nn.Sequential(nn.Linear(D+1,128),nn.ReLU(),nn.Linear(128,128),nn.ReLU())
        s.att=nn.Linear(128,1); s.val=nn.Linear(128,D)
    def forward(s,toks,mask):
        h=s.inp(toks); a=s.att(h).squeeze(-1).masked_fill(mask==0,-1e9); al=torch.softmax(a,1)
        return (al.unsqueeze(-1)*s.val(h)).sum(1)
enc=Enc(); opt=torch.optim.Adam(enc.parameters(),1e-3,weight_decay=1e-5)

def make_batch(users,kk):
    B=len(users); toks=np.zeros((B,kk,D+1),np.float32); msk=np.zeros((B,kk),np.float32)
    tgt=np.zeros((B,ni),np.float32); wt=np.ones((B,ni),np.float32); seen=np.zeros((B,ni),bool)
    for b,x in enumerate(users):
        a,c=off[x],off[x+1]; idx=np.arange(a,c); rng.shuffle(idx); rev=idx[:kk]
        its=ii_s[rev]; rss=res_s[rev]
        toks[b,:len(rev),:D]=Q[its]; toks[b,:len(rev),D]=rss; msk[b,:len(rev)]=1; seen[b,its]=True
        allit=ii_s[a:c]; alll=allit[rr_s[a:c]>=LIKE]
        unl=alll[~seen[b,alll]]
        if len(unl): tgt[b,unl]=1.0; wt[b,unl]=ipsw[unl]
        posw=float(ipsw[unl].sum()) if len(unl) else 0.
        seenb=seen[b]; negmask=(~seenb)&(tgt[b]==0); nneg=int(negmask.sum())
        if nneg>0 and posw>0: wt[b,negmask]=posw/nneg
    return (torch.tensor(toks),torch.tensor(msk),torch.tensor(tgt),torch.tensor(wt),torch.tensor(seen))

# ---- val: encoder full-profile fold-in NDCG@50 (full catalogue), held-out half-likes ----
_rs=np.random.default_rng(123); VSPL={}
vpool=list(va[:NEVAL_VAL])
for x in vpool:
    a,c=off[x],off[x+1]; its=ii_s[a:c]; lk=its[rr_s[a:c]>=LIKE]
    if len(lk)>=4:
        ll=lk.copy(); _rs.shuffle(ll); VSPL[x]=set(int(j) for j in ll[len(ll)//2:])
_Wv=1./np.log2(np.arange(2,52))
def enc_u_np(its,rss):
    if len(its)==0: return np.zeros(D,np.float32)
    tk=np.zeros((1,len(its),D+1),np.float32); mk=np.ones((1,len(its)),np.float32)
    tk[0,:,:D]=Q[its]; tk[0,:,D]=rss
    with torch.no_grad(): return enc(torch.tensor(tk),torch.tensor(mk)).numpy()[0]
def val_ndcg50():
    enc.eval(); tot=0.; m=0
    for x in vpool:
        if x not in VSPL: continue
        a,c=off[x],off[x+1]; its=ii_s[a:c]; rss=res_s[a:c]; rvals=rr_s[a:c]
        test=VSPL[x]; profmask=np.array([int(j) not in test for j in its])
        prof_it=its[profmask]; prof_rs=rss[profmask]
        rel=test
        if len(prof_it)<2 or not rel: continue
        u=enc_u_np(prof_it,prof_rs)
        sc=popb+Q@u; sc[prof_it]=-1e9
        top=np.argpartition(-sc,50)[:50]; top=top[np.argsort(-sc[top])]
        dcg=sum(_Wv[p] for p,t in enumerate(top) if int(t) in rel); idcg=_Wv[:min(50,len(rel))].sum()
        tot+=dcg/idcg if idcg>0 else 0.; m+=1
    enc.train(); return tot/max(m,1), m

# ---- resume ----
st=f'{out}/enc_v1_ml25m_state.pt'; bestf=f'{out}/enc_v1_ml25m.pt'; peakf=f'{out}/enc_v1_ml25m_peak.txt'
ep0=0; best=-1.0
if os.path.exists(st):
    ck=torch.load(st); enc.load_state_dict(ck['enc']); opt.load_state_dict(ck['opt'])
    ep0=ck['epoch']; best=ck['best']; print(f"[RESUME] from epoch {ep0}, best-val NDCG@50={best:.4f}",flush=True)

nrun=min(EP_CHUNK,EP_MAX-ep0)
if nrun<=0:
    print(f"[DONE] already trained {ep0}/{EP_MAX} epochs; best-val NDCG@50={best:.4f}",flush=True)
else:
    for e in range(nrun):
        ep=ep0+e; rng.shuffle(trbig); tl=0.; nb=0
        for b0 in range(0,len(trbig),BATCH):
            us=trbig[b0:b0+BATCH]; kk=int(rng.integers(1,K+1))
            toks,msk,tgt,wt,seen=make_batch(us,kk); u=enc(toks,msk)
            sc=u@Qt.t()
            loss=wt*nn.functional.binary_cross_entropy_with_logits(sc,tgt,reduction='none')
            loss=loss.masked_fill(seen,0.).mean()
            opt.zero_grad(); loss.backward(); opt.step(); tl+=loss.item(); nb+=1
        vN,vm=val_ndcg50()
        improved = vN>best
        if improved: best=vN; torch.save(enc.state_dict(),bestf)
        torch.save({'enc':enc.state_dict(),'opt':opt.state_dict(),'epoch':ep+1,'best':best},st)
        with open(peakf,'a') as f: f.write(f"ep{ep+1} loss={tl/nb:.4f} val_ndcg50={vN:.4f} (n={vm}) best={best:.4f}{' *SAVED' if improved else ''}\n")
        print(f"ep{ep+1}/{EP_MAX} loss={tl/nb:.4f} val_ndcg50={vN:.4f} (n={vm}) best={best:.4f}{' *SAVED' if improved else ''} ({time.time()-t0:.0f}s)",flush=True)
    print(f"[CHUNK DONE] ran {nrun} epochs -> now at {ep0+nrun}/{EP_MAX}; best-val NDCG@50={best:.4f} ({time.time()-t0:.0f}s)",flush=True)
