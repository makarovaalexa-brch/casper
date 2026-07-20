"""AMORTIZED-E3: a LEARNED continuous q2 policy (DKQS, differentiable through the exact Kalman fold, no RL).
After a static q1, state = mu1 (Sig1 shared). Actor: mu1 -> softmax weights over the q2 concept bank ->
d = normalize(sum_c w_c d_c). Blend answer y=sum w_c y_c, s2=sum w_c^2 s2_c (honesty-checked). Fold (Kalman),
soft tail-NDCG loss (listwise softmax-DCG). Trained on train half, exact tail-NDCG@q2 on eval half.
Ladder on the SAME split/cache: static-q2 (best single concept on train), variance-greedy-q2 (per-user argmax
dᵀSig1 d), per-user PEEKING oracle (ceiling). Controls: frozen-mu twin, answer-permutation null.
Probe on 2500 cache (HARD RULE #1/#2: headline needs all-users + sign-off). LLM-free.
"""
import os,sys,time
sys.path.insert(0,'scripts')
import numpy as np, torch, torch.nn as nn
import set_mn as S
from signed_latent import load_arena_base, ndcg10
from reconciled import ConceptBank
def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}",flush=True)
torch.set_num_threads(os.cpu_count()); torch.manual_seed(0)
S.set_grading("ordinal"); NC=S.NC; D=512; OUT=".cache/set_mn"; CACHE=OUT+"/bpool_cache_ref.npz"
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
NCA=len(UIDX); DcU=Dc[UIDX]                              # NCA x D
# per-user answer table (level,y,s2) over UIDX, plus held tail items
LOG2np=1.0/np.log2(np.arange(2,12))
def prep(u):
    hlt,selU,valU,refU,aki,akr,hlall,us=u; tt=Bc[UIDX,0]*selU+Bc[UIDX,1]*valU+Bc[UIDX,2]
    lev=np.array([int(np.searchsorted(EDG[c],tt[j])) for j,c in enumerate(UIDX)])
    hlt=np.asarray(hlt,np.int64); nt=min(len(hlt),10); idcg=LOG2np[:nt].sum() if nt>0 else 1.0
    return dict(hlt=hlt,hlall=hlall,y=Yc[UIDX,lev],s2=np.maximum(S2[UIDX,lev],1e-3),idcg=idcg)
_R=np.full(ni,-1,np.int64)
def fast_ndt(sc,hlt,idcg):
    s=sc.copy(); s[head]=-1e30; top=np.argpartition(-s,10)[:10]; order=top[np.argsort(-s[top])]
    _R[order]=np.arange(10); r=_R[hlt]; m=r>=0; v=LOG2np[r[m]].sum()/idcg if m.any() else 0.0; _R[order]=-1; return v
P=[prep(u) for u in US]
rs=np.random.default_rng(7); perm=rs.permutation(len(P)); half=len(P)//2
TRi=perm[:half]; EVi=perm[half:]; TR=[P[i] for i in TRi]; EV=[P[i] for i in EVi]
log(f"train {len(TR)} eval {len(EV)}  catalog {ni}  bank {NCA}")
# ---- static q1 = variance-greedy best concept (same for all) ----
vg=np.einsum('cd,dd,cd->c',DcU,Sig0,DcU) if False else np.array([DcU[c]@(Sig0@DcU[c]) for c in range(NCA)])
q1=int(np.argmax(vg)); dq1=DcU[q1]; log(f"static q1 = bank#{q1} (concept {UIDX[q1]})")
Sd1=Sig0@dq1; s2q1_bar=float(np.mean([p['s2'][q1] for p in P]))
K1=Sd1/(dq1@Sd1+s2q1_bar); Sig1=Sig0-np.outer(K1,Sd1)   # shared Sig1 (mean s2)
def mu1_of(p): d=dq1; s2=p['s2'][q1]; k=(Sig0@d)/(d@(Sig0@d)+s2); return k*(p['y'][q1]-0.0)  # mu0=0
MU1=np.array([mu1_of(p) for p in P])                    # per-user mu1
BASE=popb[None,:]+MU1@Wd.T                               # per-user base ranking (popb+Wd@mu1)
# fast tail-NDCG of question `col` for a list of global user indices (shared decoder projection wk)
def cand_tails(col,idxs):
    d=DcU[col]; Sd=Sig1@d; wk=Wd@Sd; q=float(d@Sd); out=np.empty(len(idxs))
    for j,i in enumerate(idxs):
        coef=(P[i]['y'][col]-d@MU1[i])/(q+P[i]['s2'][col]); out[j]=fast_ndt(BASE[i]+coef*wk,P[i]['hlt'],P[i]['idcg'])
    return out
def fast_from_mu(mu2,p): return fast_ndt(popb+Wd@mu2,p['hlt'],p['idcg'])
# torch tensors
Wd_t=torch.tensor(Wd); popb_t=torch.tensor(popb); DcU_t=torch.tensor(DcU); Sig1_t=torch.tensor(Sig1)
headmask_t=torch.tensor(head)
LOG2=1.0/np.log2(np.arange(2,12))
# q1-only (coef=0 => base ranking)
q1_only=np.mean([fast_ndt(BASE[EVi[i]],EV[i]['hlt'],EV[i]['idcg']) for i in range(len(EV))])
# static-q2 = best single concept on TRAIN (fast); + per-user oracle ceiling on EVAL, in one pass over cols
log("scanning bank on train+eval (fast) ...")
tr_sum=np.zeros(NCA); ev_max=np.full(len(EVi),-1.0); t0=time.time()
for col in range(NCA):
    tr_sum[col]=cand_tails(col,TRi).sum()
    ev=cand_tails(col,EVi); ev_max=np.maximum(ev_max,ev)
    if col%300==0: log(f"  col {col}/{NCA} ({time.time()-t0:.0f}s)")
q2s=int(np.argmax(tr_sum)); sv=tr_sum[q2s]/len(TRi); log(f"static-q2 = bank#{q2s} train-tail {sv:.4f}")
stat_q2=cand_tails(q2s,EVi).mean()
vgcol=int(np.argmax([DcU[c]@(Sig1@DcU[c]) for c in range(NCA)])); vg_q2=cand_tails(vgcol,EVi).mean()
orc=ev_max.mean()
log(f"LADDER  q1-only {q1_only:.4f} | static-q2 {stat_q2:.4f} (+{stat_q2-q1_only:.4f}) | var-greedy {vg_q2:.4f} (+{vg_q2-q1_only:.4f}) | ORACLE {orc:.4f} (+{orc-q1_only:.4f})")
# ---- DKQS actor ----
class Actor(nn.Module):
    def __init__(s):
        super().__init__(); s.net=nn.Sequential(nn.Linear(D,512),nn.ReLU(),nn.Linear(512,512),nn.ReLU(),nn.Linear(512,NCA))
    def forward(s,mu1): return s.net(mu1)   # logits over bank
act=Actor().double(); opt=torch.optim.Adam(act.parameters(),lr=3e-3)
MU1_t=torch.tensor(MU1); YU_t=torch.tensor(np.array([p['y'] for p in P])); S2U_t=torch.tensor(np.array([p['s2'] for p in P]))
# precompute per-user tail DCG target weights (soft listwise)
def soft_loss(mu2_b,idxs):
    sc=mu2_b@Wd_t.T + popb_t                       # B x ni
    sc=sc.masked_fill(headmask_t.unsqueeze(0),-1e30)
    logp=torch.log_softmax(sc,dim=1)               # B x ni
    loss=0.0; nB=0
    for bi,i in enumerate(idxs):
        h=P[i]['hlt'];
        if len(h)==0: continue
        w=torch.tensor(LOG2[:min(len(h),10)]); hh=torch.tensor(h[:len(w)])
        loss=loss-(w*logp[bi,hh]).sum()/w.sum(); nB+=1
    return loss/max(nB,1)
def batch_mu2(idxs,permute=False,frozen=False):
    mu1b=MU1_t[idxs]                                # B x D
    logits=act(torch.zeros_like(mu1b) if frozen else mu1b)
    w=torch.softmax(logits,dim=1)                  # B x NCA
    d=w@DcU_t; d=d/ d.norm(dim=1,keepdim=True).clamp_min(1e-8)   # B x D
    yidx=idxs if not permute else np.random.default_rng(1).permutation(idxs)
    y=(w*YU_t[yidx]).sum(1); s2=(w*w*S2U_t[yidx]).sum(1)        # B
    Sd=d@Sig1_t                                    # B x D  (Sig1 symmetric)
    denom=(d*Sd).sum(1)+s2                          # B
    coef=(y-(d*mu1b).sum(1))/denom                 # B
    mu2=mu1b+coef.unsqueeze(1)*Sd                   # B x D
    return mu2,w
log("training DKQS ...")
for ep in range(60):
    act.train(); order=np.random.default_rng(ep).permutation(len(TRi)); tot=0.0
    for b in range(0,len(order),256):
        idxs=TRi[order[b:b+256]]; mu2,_=batch_mu2(idxs); loss=soft_loss(mu2,idxs)
        opt.zero_grad(); loss.backward(); opt.step(); tot+=float(loss)
    if ep%10==0 or ep==59:
        act.eval()
        with torch.no_grad():
            mu2e,we=batch_mu2(EVi); nd=np.mean([fast_from_mu(mu2e[bi].numpy(),EV[bi]) for bi in range(len(EV))])
        log(f"  ep{ep} loss {tot/(len(order)//256+1):.4f}  eval-tail {nd:.4f} (+{nd-q1_only:.4f})  sparsity(maxw) {float(we.max(1).values.mean()):.2f}")
# final eval + controls
act.eval()
with torch.no_grad():
    mu2e,we=batch_mu2(EVi); learned=np.mean([fast_from_mu(mu2e[bi].numpy(),EV[bi]) for bi in range(len(EV))])
    mu2t,_=batch_mu2(EVi,frozen=True); twin=np.mean([fast_from_mu(mu2t[bi].numpy(),EV[bi]) for bi in range(len(EV))])
    mu2p,_=batch_mu2(EVi,permute=True); perm_=np.mean([fast_from_mu(mu2p[bi].numpy(),EV[bi]) for bi in range(len(EV))])
    snap_agree=float((we.argmax(1).numpy()==q2s).mean())
log("="*70)
log(f"LEARNED DKQS   {learned:.4f} (+{learned-q1_only:.4f})   [target E3 +0.0254; beat var-greedy +{vg_q2-q1_only:.4f}]")
log(f"  frozen-mu twin {twin:.4f} (+{twin-q1_only:.4f})  -> answer-contingent prize {learned-twin:+.4f}")
log(f"  answer-perm null {perm_:.4f} (+{perm_-q1_only:.4f})  (should ~= q1-only)")
log(f"  snap-agreement with static-q2 {snap_agree:.2f}  mean max-weight {float(we.max(1).values.mean()):.2f}")
log(f"LADDER  q1-only {q1_only:.4f} | static {stat_q2:.4f} | var-greedy {vg_q2:.4f} | LEARNED {learned:.4f} | oracle {orc:.4f}")
log("done")
