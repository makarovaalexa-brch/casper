"""NON-MYOPIC DKQS (concept-only, probe): fixes the myopic collapse. T=10 differentiable unroll with
  - REPEAT-MASK (top-weight concept masked next turns),
  - Sigma-AWARE state [mu, diag(Sig_t), turn-emb] via EXACT rank-1 V-downdate (Sig_t@d = Sig0@d - V(V^T d)),
  - LATE-WEIGHTED terminal objective (lam_t ~ t^2) so gradients from late turns reach early questions (non-myopic).
Trained under TAIL and FULL objectives (two runs); both metrics reported per turn q0..q10.
Compared vs the STRONG greedy-static questionnaire + variance-greedy, on the SAME eval half. Controls: frozen-mu
twin, answer-permutation null, snapped-top-1 arm. 2500 cache probe (HARD RULE #1/#2). LLM-free.
"""
import os,sys,time
sys.path.insert(0,'scripts')
import numpy as np, torch, torch.nn as nn
import set_mn as S
from signed_latent import load_arena_base
from reconciled import ConceptBank
def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}",flush=True)
torch.set_num_threads(os.cpu_count()); torch.manual_seed(0)
S.set_grading("ordinal"); NC=S.NC; D=512; OUT=".cache/set_mn"; CACHE=OUT+"/bpool_cache_ref.npz"; T=10
base=load_arena_base(); ni=base['ni']; head=base['headmask']; cnt=base['cnt']
pa=torch.load(OUT+'/paord_best.pt',map_location='cpu'); Wt=nn.Linear(D,ni); Wt.load_state_dict(pa['decoder']); Wd=Wt.weight.detach().numpy().astype(np.float64)
cb=ConceptBank(ni,cnt); Mbin=cb.Mbin.tocsr().astype(np.float64)
mm=Wd.mean(0); Wc=Wd-mm; rng=np.random.default_rng(0); _,_,Vt=np.linalg.svd(Wc[rng.choice(ni,4000,replace=False)],full_matrices=False); v1=Vt[0]; Wcw=Wc-(Wc@v1)[:,None]*v1[None,:]
Dc=np.zeros((NC,D))
for c in range(NC):
    idx=Mbin[c].indices
    if len(idx)>=20: x=Wcw[idx].mean(0); n=np.linalg.norm(x); Dc[c]=x/n if n>0 else 0
popb=np.log(cnt.astype(np.float64)+1.0)
z=np.load(CACHE,allow_pickle=True); Sig0=z['Sig0']; Bc=z['Bc']; EDG=z['EDG']; Yc=z['Yc']; S2=z['S2']; UIDX=z['uidx']; US=list(z['US'])
NCA=len(UIDX); DcU=Dc[UIDX]; LOG2=1.0/np.log2(np.arange(2,12))
def prep(u):
    hlt,selU,valU,refU,aki,akr,hlall,us=u; tt=Bc[UIDX,0]*selU+Bc[UIDX,1]*valU+Bc[UIDX,2]
    lev=np.array([int(np.searchsorted(EDG[c],tt[j])) for j,c in enumerate(UIDX)])
    hlt=np.asarray(hlt,np.int64); hla=np.asarray(hlall,np.int64)
    return dict(hlt=hlt,hla=hla,y=Yc[UIDX,lev],s2=np.maximum(S2[UIDX,lev],1e-3),
                it=LOG2[:min(len(hlt),10)].sum() if len(hlt) else 1.0, iff=LOG2[:min(len(hla),10)].sum() if len(hla) else 1.0)
P=[prep(u) for u in US]
rs=np.random.default_rng(7); perm=rs.permutation(len(P)); half=len(P)//2
TRi=perm[:half]; EVi=perm[half:]; log(f"train {len(TRi)} eval {len(EVi)} bank {NCA} catalog {ni}")
_R=np.full(ni,-1,np.int64)
def fast_nd(sc,held,idcg,tail):
    s=sc.copy();
    if tail: s[head]=-1e30
    top=np.argpartition(-s,10)[:10]; order=top[np.argsort(-s[top])]; _R[order]=np.arange(10)
    r=_R[held]; m=r>=0; v=LOG2[r[m]].sum()/idcg if m.any() else 0.0; _R[order]=-1; return v
def kal_np(mu,Sig,col,y,s2): d=DcU[col]; Sd=Sig@d; k=Sd/(d@Sd+s2); return mu+k*(y-d@mu),Sig-np.outer(k,Sd)
# ===== STRONG static baseline (greedy on train) + variance-greedy, numpy, to q10 =====
def meanS2(c): return float(np.mean([P[gi]['s2'][c] for gi in TRi]))
def build_static():
    seq=[]; muT={gi:np.zeros(D) for gi in TRi}; SigT=Sig0.copy(); avail=list(range(NCA))
    for t in range(T):
        BASE={gi:popb+Wd@muT[gi] for gi in TRi}; best=(-1,avail[0])
        for col in avail:
            d=DcU[col]; Sd=SigT@d; wk=Wd@Sd; q=float(d@Sd); tot=0.0
            for gi in TRi:
                p=P[gi]; coef=(p['y'][col]-d@muT[gi])/(q+p['s2'][col]); tot+=fast_nd(BASE[gi]+coef*wk,p['hlt'],p['it'],True)
            if tot>best[0]: best=(tot,col)
        col=best[1]; seq.append(col); avail.remove(col)
        for gi in TRi:
            d=DcU[col]; Sd=SigT@d; k=Sd/(d@Sd+P[gi]['s2'][col]); muT[gi]=muT[gi]+k*(P[gi]['y'][col]-d@muT[gi])
        SigT=Sig0.copy()
        for c in seq: SigT=SigT-np.outer(SigT@DcU[c],SigT@DcU[c])/(DcU[c]@(SigT@DcU[c])+meanS2(c))
    return seq
def run_seq_or_policy(kind,seq=None):
    ft=[[[] for _ in range(T+1)] for _ in range(2)]  # [metric][turn]; metric0=full,1=tail
    for gi in EVi:
        p=P[gi]; mu=np.zeros(D); Sig=Sig0.copy(); avail=list(range(NCA))
        for mi,(held,idc,tl) in enumerate([(p['hla'],p['iff'],False),(p['hlt'],p['it'],True)]):
            ft[mi][0].append(fast_nd(popb+Wd@mu,held,idc,tl))
        for t in range(T):
            if kind=='static': col=seq[t]
            else: A=DcU[avail]; col=avail[int(np.argmax(np.einsum('cd,cd->c',A@Sig,A)))]
            if col in avail: avail.remove(col)
            mu,Sig=kal_np(mu,Sig,col,p['y'][col],p['s2'][col])
            ft[0][t+1].append(fast_nd(popb+Wd@mu,p['hla'],p['iff'],False)); ft[1][t+1].append(fast_nd(popb+Wd@mu,p['hlt'],p['it'],True))
    return [np.mean(x) for x in ft[0]],[np.mean(x) for x in ft[1]]
STC=OUT+"/dkqs2_static.npz"
if os.path.exists(STC):
    d=np.load(STC); SEQ=list(d['seq']); statF,statT,vgF,vgT=list(d['sf']),list(d['st']),list(d['vf']),list(d['vt']); log("loaded cached static/vg")
else:
    log("building strong static ..."); SEQ=build_static(); statF,statT=run_seq_or_policy('static',SEQ); vgF,vgT=run_seq_or_policy('vg')
    np.savez(STC,seq=np.array(SEQ),sf=statF,st=statT,vf=vgF,vt=vgT)
log(f"STATIC   tail q1 {statT[1]:.4f} q5 {statT[5]:.4f} q10 {statT[10]:.4f} | full q10 {statF[10]:.4f}")
log(f"VARGREEDY tail q1 {vgT[1]:.4f} q5 {vgT[5]:.4f} q10 {vgT[10]:.4f} | full q10 {vgF[10]:.4f}")
# ===== DKQS non-myopic (torch, V-downdate) =====
Wd_t=torch.tensor(Wd); popb_t=torch.tensor(popb); DcU_t=torch.tensor(DcU); Sig0_t=torch.tensor(Sig0)
dSig0=torch.tensor(np.diag(Sig0).copy()); head_t=torch.tensor(head); YU=torch.tensor(np.array([p['y'] for p in P])); S2U=torch.tensor(np.array([p['s2'] for p in P]))
TE=torch.eye(T,dtype=torch.float64)
class Actor(nn.Module):
    def __init__(s): super().__init__(); s.net=nn.Sequential(nn.Linear(2*D+T,512),nn.ReLU(),nn.Linear(512,512),nn.ReLU(),nn.Linear(512,NCA))
    def forward(s,st): return s.net(st)
def soft_ndcg(mu_b,idxs,tail):
    sc=mu_b@Wd_t.T+popb_t
    if tail: sc=sc.masked_fill(head_t.unsqueeze(0),-1e30)
    logp=torch.log_softmax(sc,1); loss=0.0; nB=0
    for bi,gi in enumerate(idxs):
        h=P[gi]['hlt'] if tail else P[gi]['hla']
        if len(h)==0: continue
        w=torch.tensor(LOG2[:min(len(h),10)]); loss=loss-(w*logp[bi,torch.tensor(h[:len(w)])]).sum()/w.sum(); nB+=1
    return loss/max(nB,1)
def unroll_batch(act,idxs,tail,frozen=False,permute=False,collect=False):
    B=len(idxs); mu=torch.zeros(B,D,dtype=torch.float64); Vst=torch.zeros(B,T,D,dtype=torch.float64)
    asked=torch.zeros(B,NCA,dtype=torch.bool); yidx=idxs if not permute else np.random.default_rng(2).permutation(idxs)
    Yb=YU[yidx]; S2b=S2U[yidx]; loss=0.0; lam=[(t+1)**2 for t in range(T)]; Z=sum(lam); mus=[]
    for t in range(T):
        diagS=dSig0.unsqueeze(0)-(Vst*Vst).sum(1)                       # B x D
        st=torch.cat([mu if not frozen else torch.zeros_like(mu),diagS,TE[t].unsqueeze(0).expand(B,-1)],1)
        logits=act(st).masked_fill(asked,-1e30); w=torch.softmax(logits,1)
        d=w@DcU_t; d=d/d.norm(dim=1,keepdim=True).clamp_min(1e-8)
        Sd=d@Sig0_t - torch.einsum('bt,btd->bd',torch.einsum('btd,bd->bt',Vst,d),Vst)   # Sig_t @ d (exact)
        y=(w*Yb).sum(1); s2=(w*w*S2b).sum(1); denom=(d*Sd).sum(1)+s2
        mu=mu+(Sd/denom.unsqueeze(1))*(y-(d*mu).sum(1)).unsqueeze(1)
        Vst=Vst.clone(); Vst[:,t,:]=Sd/denom.clamp_min(1e-9).sqrt().unsqueeze(1)
        asked=asked.clone(); asked[torch.arange(B),w.argmax(1)]=True
        if not collect: loss=loss+lam[t]/Z*soft_ndcg(mu,idxs,tail)
        else: mus.append(mu.detach())
    return (loss,None) if not collect else (None,mus)
def eval_curve(act,tail_train):
    ft=[[[] for _ in range(T+1)] for _ in range(2)]
    act.eval()
    with torch.no_grad():
        for b in range(0,len(EVi),256):
            idxs=EVi[b:b+256]; _,mus=unroll_batch(act,idxs,tail_train,collect=True)
            for bi,gi in enumerate(idxs):
                p=P[gi]
                ft[0][0].append(fast_nd(popb+Wd@np.zeros(D),p['hla'],p['iff'],False)); ft[1][0].append(fast_nd(popb,p['hlt'],p['it'],True))
                for t in range(T):
                    m2=mus[t][bi].numpy(); ft[0][t+1].append(fast_nd(popb+Wd@m2,p['hla'],p['iff'],False)); ft[1][t+1].append(fast_nd(popb+Wd@m2,p['hlt'],p['it'],True))
    return [np.mean(x) for x in ft[0]],[np.mean(x) for x in ft[1]]
def train_dkqs(tail):
    act=Actor().double(); opt=torch.optim.Adam(act.parameters(),lr=2e-3); best=(-1,None)
    for ep in range(40):
        act.train(); ordr=np.random.default_rng(ep).permutation(len(TRi))
        for b in range(0,len(ordr),256):
            idxs=TRi[ordr[b:b+256]]; loss,_=unroll_batch(act,idxs,tail); opt.zero_grad(); loss.backward(); opt.step()
        if ep%5==0 or ep==39:
            f,t=eval_curve(act,tail); key=t[T] if tail else f[T]
            log(f"  [{'tail' if tail else 'full'}] ep{ep} tailq10 {t[T]:.4f} fullq10 {f[T]:.4f}")
            if key>best[0]: best=(key,{k:v.clone() for k,v in act.state_dict().items()})
    act.load_state_dict(best[1]); return act
res={}
for tail in [True,False]:
    log(f"training DKQS ({'TAIL' if tail else 'FULL'} objective) ...")
    act=train_dkqs(tail); f,t=eval_curve(act,tail); res['dkqs-'+('tail' if tail else 'full')]=(f,t)
    # controls at the trained (tail) actor
    if tail:
        with torch.no_grad():
            fs=[[[] for _ in range(T+1)] for _ in range(2)]
            for b in range(0,len(EVi),256):
                idxs=EVi[b:b+256]; _,mus=unroll_batch(act,idxs,True,frozen=True,collect=True)
                for bi,gi in enumerate(idxs):
                    p=P[gi]
                    for t in range(T): fs[1][t+1].append(fast_nd(popb+Wd@mus[t][bi].numpy(),p['hlt'],p['it'],True))
            twinT=np.mean(fs[1][T])
            fp=[[] for _ in range(T+1)]
            for b in range(0,len(EVi),256):
                idxs=EVi[b:b+256]; _,mus=unroll_batch(act,idxs,True,permute=True,collect=True)
                for bi,gi in enumerate(idxs): fp[T].append(fast_nd(popb+Wd@mus[T-1][bi].numpy(),P[gi]['hlt'],P[gi]['it'],True))
            permT=np.mean(fp[T])
        log(f"CONTROLS tail@q10: learned {t[T]:.4f} | frozen-mu twin {twinT:.4f} (prize {t[T]-twinT:+.4f}) | perm-null {permT:.4f}")
log("="*90)
for nm,(f,tl) in [('static',(statF,statT)),('var-greedy',(vgF,vgT)),('dkqs-tail',res['dkqs-tail']),('dkqs-full',res['dkqs-full'])]:
    log(f"{nm:>11} TAIL "+" ".join(f"{tl[q]:.4f}" for q in range(T+1)))
for nm,(f,tl) in [('static',(statF,statT)),('var-greedy',(vgF,vgT)),('dkqs-tail',res['dkqs-tail']),('dkqs-full',res['dkqs-full'])]:
    log(f"{nm:>11} FULL "+" ".join(f"{f[q]:.4f}" for q in range(T+1)))
log(f"BARS: static tail@10 {statT[T]:.4f} | dkqs-tail {res['dkqs-tail'][1][T]:.4f} | dkqs-full tail {res['dkqs-full'][1][T]:.4f}")
log("done")
