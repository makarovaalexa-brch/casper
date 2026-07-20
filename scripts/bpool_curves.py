"""UNIFIED policy comparison: random / static-questionnaire / variance-greedy / VoI / adaptive-tree / DKQS-learned,
all on the SAME refusal cache + SAME eval split, T=8, logging FULL and TAIL NDCG@10 per turn q0..q8.
Probe on 2500 cache (HARD RULE #1/#2). Concept-only shared bank; per-user belief unroll (numpy).
DKQS = the differentiable learned actor (trained on mixed belief states), applied MYOPICALLY each turn
(the 1-step-trained actor; the full T=8-trained unroll is a separate pending build -- labelled honestly).
"""
import os,sys,time
sys.path.insert(0,'scripts')
import numpy as np, torch, torch.nn as nn
import set_mn as S
from signed_latent import load_arena_base
from reconciled import ConceptBank
def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}",flush=True)
torch.set_num_threads(os.cpu_count()); torch.manual_seed(0)
S.set_grading("ordinal"); NC=S.NC; D=512; OUT=".cache/set_mn"; CACHE=OUT+"/bpool_cache_ref.npz"; T=8; MB=500
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
NCA=len(UIDX); DcU=Dc[UIDX]
LOG2=1.0/np.log2(np.arange(2,12))
def prep(u):
    hlt,selU,valU,refU,aki,akr,hlall,us=u; tt=Bc[UIDX,0]*selU+Bc[UIDX,1]*valU+Bc[UIDX,2]
    lev=np.array([int(np.searchsorted(EDG[c],tt[j])) for j,c in enumerate(UIDX)])
    hlt=np.asarray(hlt,np.int64); hla=np.asarray(hlall,np.int64)
    it=min(len(hlt),10); iff=min(len(hla),10)
    return dict(hlt=hlt,hla=hla,y=Yc[UIDX,lev],s2=np.maximum(S2[UIDX,lev],1e-3),it=LOG2[:it].sum() if it else 1.0,iff=LOG2[:iff].sum() if iff else 1.0)
P=[prep(u) for u in US]
rs=np.random.default_rng(7); perm=rs.permutation(len(P)); half=len(P)//2
TRi=perm[:half]; EVi=perm[half:]; TR=[P[i] for i in TRi]; EV=[P[i] for i in EVi]
log(f"train {len(TR)} eval {len(EV)} bank {NCA} catalog {ni}")
_R=np.full(ni,-1,np.int64)
def fast_nd(sc,held,idcg,tail):
    s=sc.copy()
    if tail: s[head]=-1e30
    top=np.argpartition(-s,10)[:10]; order=top[np.argsort(-s[top])]
    _R[order]=np.arange(10); r=_R[held]; m=r>=0; v=LOG2[r[m]].sum()/idcg if m.any() else 0.0; _R[order]=-1; return v
def kal(mu,Sig,col,y,s2):
    d=DcU[col]; Sd=Sig@d; k=Sd/(d@Sd+s2); return mu+k*(y-d@mu),Sig-np.outer(k,Sd)
def rec(mu,p,full,tail,t):
    sc=popb+Wd@mu; full[t].append(fast_nd(sc,p['hla'],p['iff'],False)); tail[t].append(fast_nd(sc,p['hlt'],p['it'],True))
# ---- generic realizable unroll over EVAL ----
def unroll(select,users,uidx):
    full=[[] for _ in range(T+1)]; tail=[[] for _ in range(T+1)]
    for gi in uidx:
        p=P[gi]; mu=np.zeros(D); Sig=Sig0.copy(); avail=list(range(NCA)); rec(mu,p,full,tail,0)
        st={}
        for t in range(T):
            col=select(mu,Sig,avail,p,t,st); avail.remove(col)
            mu,Sig=kal(mu,Sig,col,p['y'][col],p['s2'][col]); rec(mu,p,full,tail,t+1)
    return [np.mean(x) for x in full],[np.mean(x) for x in tail]
def sel_random(mu,Sig,avail,p,t,st):
    if 'r' not in st: st['r']=np.random.default_rng(hash(id(p))&0xffff)
    return avail[st['r'].integers(len(avail))]
def sel_vg(mu,Sig,avail,p,t,st):
    A=DcU[avail]; q=np.einsum('cd,cd->c',A@Sig,A); return avail[int(np.argmax(q))]
def sel_voi(mu,Sig,avail,p,t,st):
    A=DcU[avail]; SD=Sig@A.T; vq=np.einsum('dq,dq->q',A.T,SD)+np.array([p['s2'][c] for c in avail])
    sc=popb+Wd@mu; sc2=sc.copy(); sc2[head]=-1e30; top=np.argpartition(-sc2,MB)[:MB]
    lam=1.0/np.log2(2+np.arange(MB)); G=Wd[top]@SD; gain=(lam@(G*G))/vq; return avail[int(np.argmax(gain))]
# ---- static questionnaire: greedy fixed sequence on TRAIN, applied to EVAL ----
def meanS2(c): return float(np.mean([P[gi]['s2'][c] for gi in TRi]))
def build_static():
    seq=[]; muT={gi:np.zeros(D) for gi in TRi}; SigT=Sig0.copy(); avail=list(range(NCA))
    for t in range(T):
        BASE={gi:popb+Wd@muT[gi] for gi in TRi}
        best=(-1,avail[0])
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
        log(f"  static pick {t+1}: bank#{col}")
    return seq
def run_static(seq):
    full=[[] for _ in range(T+1)]; tail=[[] for _ in range(T+1)]
    for gi in EVi:
        p=P[gi]; mu=np.zeros(D); Sig=Sig0.copy(); rec(mu,p,full,tail,0)
        for t,col in enumerate(seq):
            mu,Sig=kal(mu,Sig,col,p['y'][col],p['s2'][col]); rec(mu,p,full,tail,t+1)
    return [np.mean(x) for x in full],[np.mean(x) for x in tail]
# ---- adaptive tree depth-3 (nbin3) then variance-greedy q4..8 ----
MINLEAF=40; NBIN=3
def score_cand(users,mu,Sig,col):
    by={}
    for p in users: by.setdefault(p['y'][col],[]).append(p)     # group by (discrete) answer value
    tot=0.0
    for yval,grp in by.items():
        mu2,_=kal(mu,Sig,col,yval,grp[0]['s2'][col]); s=popb+Wd@mu2; s[head]=-1e30
        top=np.argpartition(-s,10)[:10]; order=top[np.argsort(-s[top])]; _R[order]=np.arange(10)
        for p in grp:
            r=_R[p['hlt']]; m=r>=0; tot+=(LOG2[r[m]].sum()/p['it']) if m.any() else 0.0
        _R[order]=-1
    return tot/len(users)
def best_col(users,mu,Sig,avail):
    b=(-1,avail[0])
    for col in avail:
        s=score_cand(users,mu,Sig,col)
        if s>b[0]: b=(s,col)
    return b[1]
def grow(users,mu,Sig,depth,avail):
    node=dict(col=None,children={},edges=None,mu=mu,Sig=Sig)
    if depth>=3 or len(users)<MINLEAF*NBIN or not avail: return node
    col=best_col(users,mu,Sig,avail); node['col']=col; av2=[c for c in avail if c!=col]
    levs=np.array([p['y'][col] for p in users]); edges=np.quantile(levs,np.linspace(0,1,NBIN+1)[1:-1]); node['edges']=list(edges)
    for b in range(NBIN):
        sub=[p for p in users if (p['y'][col]>(-1e9 if b==0 else edges[b-1])) and (p['y'][col]<=(edges[b] if b<NBIN-1 else 1e9))]
        if len(sub)<MINLEAF: sub=users
        ybar=float(np.mean([p['y'][col] for p in sub])); mu2,Sig2=kal(mu,Sig,col,ybar,float(np.mean([p['s2'][col] for p in sub])))
        node['children'][b]=grow(sub,mu2,Sig2,depth+1,av2)
    return node
def run_tree(node):
    full=[[] for _ in range(T+1)]; tail=[[] for _ in range(T+1)]
    for gi in EVi:
        p=P[gi]; mu=np.zeros(D); Sig=Sig0.copy(); avail=list(range(NCA)); rec(mu,p,full,tail,0); n=node
        for t in range(T):
            if t<3 and n['col'] is not None:
                col=n['col']; b=int(np.searchsorted(n['edges'],p['y'][col])); nxt=n['children'].get(b,list(n['children'].values())[0])
            else:
                A=DcU[avail]; q=np.einsum('cd,cd->c',A@Sig,A); col=avail[int(np.argmax(q))]; nxt=n
            if col in avail: avail.remove(col)
            mu,Sig=kal(mu,Sig,col,p['y'][col],p['s2'][col]); rec(mu,p,full,tail,t+1); n=nxt if t<3 else n
    return [np.mean(x) for x in full],[np.mean(x) for x in tail]
# ---- DKQS learned actor (mixed-state training), myopic unroll ----
class Actor(nn.Module):
    def __init__(s): super().__init__(); s.net=nn.Sequential(nn.Linear(D,512),nn.ReLU(),nn.Linear(512,512),nn.ReLU(),nn.Linear(512,NCA))
    def forward(s,x): return s.net(x)
Wd_t=torch.tensor(Wd); popb_t=torch.tensor(popb); DcU_t=torch.tensor(DcU); Sig0_t=torch.tensor(Sig0); head_t=torch.tensor(head)
YU=np.array([p['y'] for p in P]); S2U=np.array([p['s2'] for p in P])
def train_dkqs():
    act=Actor().double(); opt=torch.optim.Adam(act.parameters(),lr=3e-3)
    # mixed states: fold k in {0,1,2} random concepts (variance-greedy-ish random) to get mu_k, shared Sig_k approx = Sig0
    rngk=np.random.default_rng(5)
    for ep in range(50):
        order=rngk.permutation(len(TRi))
        for b in range(0,len(order),256):
            idxs=TRi[order[b:b+256]]; B=len(idxs)
            # random prefix length per sample
            mu0=np.zeros((B,D));
            for bi,gi in enumerate(idxs):
                kpre=rngk.integers(0,3); cols=rngk.choice(NCA,kpre,replace=False); mu=np.zeros(D); Sig=Sig0.copy()
                for c in cols: mu,Sig=kal(mu,Sig,c,P[gi]['y'][c],P[gi]['s2'][c])
                mu0[bi]=mu
            mu0_t=torch.tensor(mu0); w=torch.softmax(act(mu0_t),1); d=w@DcU_t; d=d/d.norm(dim=1,keepdim=True).clamp_min(1e-8)
            y=(w*torch.tensor(YU[idxs])).sum(1); s2=(w*w*torch.tensor(S2U[idxs])).sum(1)
            Sd=d@Sig0_t; denom=(d*Sd).sum(1)+s2; coef=(y-(d*mu0_t).sum(1))/denom; mu2=mu0_t+coef.unsqueeze(1)*Sd
            sc=(mu2@Wd_t.T+popb_t).masked_fill(head_t.unsqueeze(0),-1e30); logp=torch.log_softmax(sc,1)
            loss=0.0; nB=0
            for bi,gi in enumerate(idxs):
                h=P[gi]['hlt']
                if len(h)==0: continue
                wq=torch.tensor(LOG2[:min(len(h),10)]); loss=loss-(wq*logp[bi,torch.tensor(h[:len(wq)])]).sum()/wq.sum(); nB+=1
            loss=loss/max(nB,1); opt.zero_grad(); loss.backward(); opt.step()
        if ep%10==0: log(f"  dkqs ep{ep} loss {float(loss):.3f}")
    return act
def run_dkqs(act):
    full=[[] for _ in range(T+1)]; tail=[[] for _ in range(T+1)]
    act.eval()
    with torch.no_grad():
        for gi in EVi:
            p=P[gi]; mu=np.zeros(D); Sig=Sig0.copy(); rec(mu,p,full,tail,0)
            for t in range(T):
                w=torch.softmax(act(torch.tensor(mu)),0).numpy(); d=w@DcU; d=d/max(np.linalg.norm(d),1e-8)
                y=float((w*p['y']).sum()); s2=float((w*w*p['s2']).sum())
                Sd=Sig@d; k=Sd/(d@Sd+s2); mu=mu+k*(y-d@mu); Sig=Sig-np.outer(k,Sd); rec(mu,p,full,tail,t+1)
    return [np.mean(x) for x in full],[np.mean(x) for x in tail]
# ---- RUN ALL ----
res={}
log("random ..."); res['random']=unroll(sel_random,P,EVi)
log("variance-greedy ..."); res['var-greedy']=unroll(sel_vg,P,EVi)
log("VoI ..."); res['voi']=unroll(sel_voi,P,EVi)
log("static questionnaire (train) ..."); seq=build_static(); res['static']=run_static(seq)
log("adaptive tree (grow) ..."); TREE=grow(TR,np.zeros(D),Sig0.copy(),0,list(range(NCA))); res['tree(+vg)']=run_tree(TREE)
log("DKQS (train) ..."); act=train_dkqs(); res['dkqs']=run_dkqs(act)
log("="*100)
order=['random','static','voi','tree(+vg)','var-greedy','dkqs']
for metric,mi in [('TAIL',1),('FULL',0)]:
    log(f"===== {metric} NDCG@10 by turn =====")
    log("policy        "+" ".join(f"q{q}    " for q in range(T+1)))
    for nm in order:
        cur=res[nm][mi]; log(f"{nm:>12}  "+" ".join(f"{cur[q]:.4f}" for q in range(T+1)))
log("done")
