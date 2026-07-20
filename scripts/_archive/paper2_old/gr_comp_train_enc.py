"""
Goodreads MULTI-GENRE COMPOSITE: V1-style attention fold-in set-encoder (mirrors gr_train_enc.py /
ml25m_train_enc.py). d=64 attention pooling over revealed (Q[item], residual) tokens -> u; FROZEN
decoder popb + Q.u; IPS-weighted BCE reconstruction of unrevealed like-set; masked random-count reveals
k=1..K. Adam 1e-3, batch 512. RESUMABLE (chunked). Best-val on 500-user val cohort full-profile fold,
NDCG@KVAL full catalogue.
Durable: enc_v1_grcomp.pt (best), enc_v1_grcomp_state.pt (resume), enc_v1_grcomp_peak.txt.
Env: EP_CHUNK(1) EP_MAX(12) K(12) BATCH(512) KVAL(510) NEVAL_VAL(500) PROBE(0 => full; N => time N batches, no save).
"""
import os, time, numpy as np, torch, torch.nn as nn
t0=time.time(); GR='C:/dev/phd/casper/.cache/goodreads'
torch.set_num_threads(os.cpu_count() or 4); D=64; LIKE=4.0
K=int(os.environ.get('K',12)); BATCH=int(os.environ.get('BATCH',512))
EP_CHUNK=int(os.environ.get('EP_CHUNK',1)); EP_MAX=int(os.environ.get('EP_MAX',12))
KVAL=int(os.environ.get('KVAL',510)); NEVAL_VAL=int(os.environ.get('NEVAL_VAL',500))
PROBE=int(os.environ.get('PROBE',0))
rng=np.random.default_rng(0); torch.manual_seed(0)
B=np.load(f'{GR}/base_comp.npz')
uu=B['uu']; ii=B['ii']; rr=B['rr']; cnt=B['cnt']; mu=float(B['mu']); ni=int(B['ni']); nu=int(B['nu'])
trU=B['trU']; va=B['va']; te=B['te']
Q=np.load(f'{GR}/Q_svd_comp.npy').astype(np.float32); bi=np.load(f'{GR}/bi_svd_comp.npy').astype(np.float32); Qt=torch.tensor(Q)
order=np.argsort(uu,kind='stable'); uu_s=uu[order]; ii_s=ii[order].astype(np.int32); rr_s=rr[order].astype(np.float32)
res_s=(rr_s-mu-bi[ii_s]).astype(np.float32)
off=np.zeros(nu+1,np.int64); np.add.at(off,uu_s+1,1); off=np.cumsum(off)
print(f"CSR built ({time.time()-t0:.0f}s); nu={nu} ni={ni} ratings={len(uu_s)}",flush=True)
popb=np.log(cnt+1.0).astype(np.float32)
ipsw=(1.0/np.clip((cnt/max(cnt.max(),1))**0.5,1e-3,1.0)).astype(np.float32)
lens=(off[1:]-off[:-1]); trbig=np.array([x for x in trU if lens[x]>=K+1],dtype=np.int64)
# fixed deterministic order (rng0) so a rotating per-epoch window covers all users across epochs
trbig=trbig[np.random.default_rng(0).permutation(len(trbig))]
CAP=int(os.environ.get('CAP',90000))  # users/epoch (10-min foreground cap => ~555s/epoch at CAP=90k)
print(f"train users (>=K+1 rated): {len(trbig)} / {len(trU)}; CAP={CAP}/epoch (rotating window)",flush=True)
class Enc(nn.Module):
    def __init__(s):
        super().__init__(); s.inp=nn.Sequential(nn.Linear(D+1,128),nn.ReLU(),nn.Linear(128,128),nn.ReLU())
        s.att=nn.Linear(128,1); s.val=nn.Linear(128,D)
    def forward(s,toks,mask):
        h=s.inp(toks); a=s.att(h).squeeze(-1).masked_fill(mask==0,-1e9); al=torch.softmax(a,1)
        return (al.unsqueeze(-1)*s.val(h)).sum(1)
enc=Enc(); opt=torch.optim.Adam(enc.parameters(),1e-3,weight_decay=1e-5)
def make_batch(users,kk):
    Bn=len(users); toks=np.zeros((Bn,kk,D+1),np.float32); msk=np.zeros((Bn,kk),np.float32)
    tgt=np.zeros((Bn,ni),np.float32); wt=np.ones((Bn,ni),np.float32); seen=np.zeros((Bn,ni),bool)
    for b,x in enumerate(users):
        a,c=off[x],off[x+1]; idx=np.arange(a,c); rng.shuffle(idx); rev=idx[:kk]
        its=ii_s[rev]; rss=res_s[rev]
        toks[b,:len(rev),:D]=Q[its]; toks[b,:len(rev),D]=rss; msk[b,:len(rev)]=1; seen[b,its]=True
        allit=ii_s[a:c]; alll=allit[rr_s[a:c]>=LIKE]; unl=alll[~seen[b,alll]]
        if len(unl): tgt[b,unl]=1.0; wt[b,unl]=ipsw[unl]
        posw=float(ipsw[unl].sum()) if len(unl) else 0.
        negmask=(~seen[b])&(tgt[b]==0); nneg=int(negmask.sum())
        if nneg>0 and posw>0: wt[b,negmask]=posw/nneg
    return (torch.tensor(toks),torch.tensor(msk),torch.tensor(tgt),torch.tensor(wt),torch.tensor(seen))
_rs=np.random.default_rng(123); VSPL={}; vpool=list(va[:NEVAL_VAL])
for x in vpool:
    a,c=off[x],off[x+1]; its=ii_s[a:c]; lk=its[rr_s[a:c]>=LIKE]
    if len(lk)>=4: ll=lk.copy(); _rs.shuffle(ll); VSPL[x]=set(int(j) for j in ll[len(ll)//2:])
_Wv=1./np.log2(np.arange(2,KVAL+2))
def enc_u_np(its,rss):
    if len(its)==0: return np.zeros(D,np.float32)
    tk=np.zeros((1,len(its),D+1),np.float32); mk=np.ones((1,len(its)),np.float32); tk[0,:,:D]=Q[its]; tk[0,:,D]=rss
    with torch.no_grad(): return enc(torch.tensor(tk),torch.tensor(mk)).numpy()[0]
def val_ndcg():
    enc.eval(); tot=0.; m=0
    for x in vpool:
        if x not in VSPL: continue
        a,c=off[x],off[x+1]; its=ii_s[a:c]; rss=res_s[a:c]; test=VSPL[x]
        pm=np.array([int(j) not in test for j in its]); prof_it=its[pm]; prof_rs=rss[pm]; rel=test
        if len(prof_it)<2 or not rel: continue
        u=enc_u_np(prof_it,prof_rs); sc=popb+Q@u; sc[prof_it]=-1e9
        top=np.argpartition(-sc,KVAL)[:KVAL]; top=top[np.argsort(-sc[top])]
        dcg=sum(_Wv[p] for p,t in enumerate(top) if int(t) in rel); idcg=_Wv[:min(KVAL,len(rel))].sum()
        tot+=dcg/idcg if idcg>0 else 0.; m+=1
    enc.train(); return tot/max(m,1), m
st=f'{GR}/enc_v1_grcomp_state.pt'; bestf=f'{GR}/enc_v1_grcomp.pt'; peakf=f'{GR}/enc_v1_grcomp_peak.txt'
ep0=0; best=-1.0
if os.path.exists(st):
    ck=torch.load(st); enc.load_state_dict(ck['enc']); opt.load_state_dict(ck['opt']); ep0=ck['epoch']; best=ck['best']
    print(f"[RESUME] from epoch {ep0}, best-val NDCG@{KVAL}={best:.4f}",flush=True)
if PROBE>0:
    rng.shuffle(trbig); tp=time.time()
    for bi_,b0 in enumerate(range(0,PROBE*BATCH,BATCH)):
        us=trbig[b0:b0+BATCH]; kk=int(rng.integers(1,K+1))
        toks,msk,tgt,wt,seen=make_batch(us,kk); u=enc(toks,msk); sc=u@Qt.t()
        loss=wt*nn.functional.binary_cross_entropy_with_logits(sc,tgt,reduction='none')
        loss=loss.masked_fill(seen,0.).mean(); opt.zero_grad(); loss.backward(); opt.step()
    dt=time.time()-tp; nb_full=int(np.ceil(len(trbig)/BATCH))
    print(f"[PROBE] {PROBE} batches in {dt:.1f}s => {dt/PROBE:.2f}s/batch; full epoch ~{dt/PROBE*nb_full:.0f}s ({nb_full} batches)",flush=True)
    raise SystemExit
nrun=min(EP_CHUNK,EP_MAX-ep0)
if nrun<=0:
    print(f"[DONE] already trained {ep0}/{EP_MAX} epochs; best-val NDCG@{KVAL}={best:.4f}",flush=True)
else:
    for e in range(nrun):
        ep=ep0+e
        wnd=trbig[(np.arange(ep*CAP,ep*CAP+CAP))%len(trbig)].copy(); rng.shuffle(wnd)  # rotating per-epoch window
        tl=0.; nb=0
        for b0 in range(0,len(wnd),BATCH):
            us=wnd[b0:b0+BATCH]; kk=int(rng.integers(1,K+1))
            toks,msk,tgt,wt,seen=make_batch(us,kk); u=enc(toks,msk); sc=u@Qt.t()
            loss=wt*nn.functional.binary_cross_entropy_with_logits(sc,tgt,reduction='none')
            loss=loss.masked_fill(seen,0.).mean()
            opt.zero_grad(); loss.backward(); opt.step(); tl+=loss.item(); nb+=1
        vN,vm=val_ndcg(); improved=vN>best
        if improved: best=vN; torch.save(enc.state_dict(),bestf)
        torch.save({'enc':enc.state_dict(),'opt':opt.state_dict(),'epoch':ep+1,'best':best},st)
        with open(peakf,'a') as f: f.write(f"ep{ep+1} loss={tl/nb:.4f} val_ndcg{KVAL}={vN:.4f} (n={vm}) best={best:.4f}{' *SAVED' if improved else ''}\n")
        print(f"ep{ep+1}/{EP_MAX} loss={tl/nb:.4f} val_ndcg{KVAL}={vN:.4f} (n={vm}) best={best:.4f}{' *SAVED' if improved else ''} ({time.time()-t0:.0f}s)",flush=True)
    print(f"[CHUNK DONE] now at {ep0+nrun}/{EP_MAX}; best-val NDCG@{KVAL}={best:.4f} ({time.time()-t0:.0f}s)",flush=True)
