"""DISCOVER-THEN-DRILL: the CONTINUOUS, clustering-free adaptivity demonstration.
Selection rule (one continuous acquisition function, no discrete tree):
  turns j<K : variance-greedy (argmax dᵀSig d) -- answer-INDEPENDENT, builds the belief (moves mu off prior).
  turns j>=K: NDCG-VoI -- reads the CURRENT ranking s=popb+Wd@mu_ref, weights info-gain toward the user's
              own borderline tail region B(mu_ref). This is where adaptivity enters: two users with different
              answers have different mu -> different B -> different next question. NO clustering.
TWIN control (isolates the value of reading the MOVED mu): identical, but the VoI reads mu_ref=0 (frozen prior)
  -> B is the popularity-top tail for everyone -> selection is answer-INDEPENDENT (static). The UPDATE always
  uses the real answer. adaptive - twin = the continuous adaptivity prize.
Also unlocks ITEMS in the drill (bank='both') to test whether concept+item WINS via the adaptive drill.
Reuses the belief cache; probe on 2500 val users (headline needs all users + sign-off, HARD RULE #1).
"""
import os,sys,time
sys.path.insert(0,'scripts')
import numpy as np, torch, torch.nn as nn
import set_mn as S
from set_mn import RSD
from signed_latent import load_arena_base, ndcg10
from reconciled import ConceptBank
def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}",flush=True)
torch.set_num_threads(os.cpu_count())
S.set_grading("ordinal"); NC=S.NC; D=512; OUT=".cache/set_mn"; CACHE=OUT+"/bpool_cache.npz"
MAXQ=8; K=2; MB=500; SIG2I=0.5
base=load_arena_base(); ni=base['ni']; head=base['headmask']; cnt=base['cnt']
pa=torch.load(OUT+'/paord_best.pt',map_location='cpu'); Wt=nn.Linear(D,ni); Wt.load_state_dict(pa['decoder']); Wd=Wt.weight.detach().numpy().astype(np.float64)
cb=ConceptBank(ni,cnt); Mbin=cb.Mbin.tocsr().astype(np.float64); csz=np.array([len(Mbin[c].indices) for c in range(NC)])
mm=Wd.mean(0); Wc=Wd-mm; rng=np.random.default_rng(0); _,_,Vt=np.linalg.svd(Wc[rng.choice(ni,4000,replace=False)],full_matrices=False); v1=Vt[0]; Wcw=Wc-(Wc@v1)[:,None]*v1[None,:]
Dc=np.zeros((NC,D))
for c in range(NC):
    idx=Mbin[c].indices
    if len(idx)>=20: x=Wcw[idx].mean(0); n=np.linalg.norm(x); Dc[c]=x/n if n>0 else 0
popb=np.log(cnt.astype(np.float64)+1.0); poprank=np.argsort(np.argsort(-cnt)); gm=None
assert os.path.exists(CACHE), "run bpool_c2f.py first"
z=np.load(CACHE,allow_pickle=True); Sig0=z['Sig0']; Bc=z['Bc']; EDG=z['EDG']; Yc=z['Yc']; S2=z['S2']; usable=z['usable']; US=list(z['US'])
import numpy as _np; gm=float(_np.load('data/movielens/.cache/ml25m/meta.npz')['rr'].astype(_np.float64).mean())
log(f"cache loaded: {len(US)} users; gm={gm:.3f}")
# US[i] = (hlt, cc, sel_cc, val_cc, aki, akr, hlall, us)
def ndft(sc,hl,hlall): return ndcg10(sc.copy(),list(hlall),set(),head,False), ndcg10(sc.copy(),list(hl),set(),head,True)
pf0=np.mean([ndft(popb,u[0],u[6])[0] for u in US]); pt0=np.mean([ndft(popb,u[0],u[6])[1] for u in US]); log(f"popb FULL {pf0:.4f} TAIL {pt0:.4f}")
def build_Q(u,bank):
    hlt,cc,sel,val,aki,akr,hlall,us=u; Q=[]
    for j,c in enumerate(cc):
        tt=Bc[c,0]*sel[j]+Bc[c,1]*val[j]+Bc[c,2]; k=int(np.searchsorted(EDG[c],tt)); Q.append(('concept',csz[c],Dc[c],Yc[c,k],max(S2[c,k],1e-3)))
    if bank=='both':
        for it,rt in zip(aki,akr):
            w=Wcw[it]; nr=np.linalg.norm(w); d=w/nr if nr>0 else w; Q.append(('item',int(poprank[it]),d,(rt-gm)/max(nr,1e-6),SIG2I))
    return Q
def voi_gain(Q,Sig,mu_ref):
    Dcand=np.array([q[2] for q in Q]); SD=Sig@Dcand.T                     # D x q
    vq=np.einsum('dq,dq->q',Dcand.T,SD)+np.array([q[4] for q in Q])
    sc=popb+Wd@mu_ref; sc2=sc.copy(); sc2[head]=-1e30; top=np.argpartition(-sc2,MB)[:MB]
    lam=1.0/np.log2(2+np.arange(MB)); A=Wd[top]@SD                         # MB x q
    return (lam@(A*A))/vq
def var_gain(Q,Sig):
    Dcand=np.array([q[2] for q in Q]); SD=Sig@Dcand.T; return np.einsum('dq,dq->q',SD,SD)
def run(bank,mode):   # mode: 'var' | 'dtd' (discover-then-drill adaptive) | 'twin' (drill frozen at prior)
    ST=[[] for _ in range(MAXQ+1)]; SF=[[] for _ in range(MAXQ+1)]; ipr=[[] for _ in range(MAXQ)]
    for u in US:
        hlt,cc,sel,val,aki,akr,hlall,us=u; mu=np.zeros(D); Sig=Sig0.copy()
        f,t=ndft(popb+Wd@mu,hlt,hlall); SF[0].append(f); ST[0].append(t); Q=build_Q(u,bank)
        for j in range(min(MAXQ,len(Q))):
            if mode=='var' or j<K: g=var_gain(Q,Sig)
            elif mode=='dtd':      g=voi_gain(Q,Sig,mu)         # reads MOVED mu -> adaptive
            else:                  g=voi_gain(Q,Sig,np.zeros(D))# twin: frozen prior -> static selection
            pk=int(np.argmax(g)); kind,brd,d,y,s2=Q.pop(pk)
            Sd=Sig@d; kk=Sd/(d@Sd+s2); mu=mu+kk*(y-d@mu); Sig=Sig-np.outer(kk,Sd)
            if kind=='item': ipr[j].append(brd)
            f,t=ndft(popb+Wd@mu,hlt,hlall); SF[j+1].append(f); ST[j+1].append(t)
    ct=[np.mean(s) for s in ST]; cf=[np.mean(s) for s in SF]; ipm=[int(np.median(x)) if x else -1 for x in ipr]
    return ct,cf,ipm
ARMS=[('concept','var'),('concept','dtd'),('concept','twin'),('both','dtd'),('both','twin')]
res={}
for bank,mode in ARMS:
    ct,cf,ipm=run(bank,mode); res[(bank,mode)]=(ct,cf,ipm)
    log(f"[{bank:>7} {mode:>4}] TAIL@8 {ct[8]:.4f} ({ct[8]-pt0:+.4f})  FULL@8 {cf[8]:.4f} ({cf[8]-pf0:+.4f})  item-poprank/turn {ipm}")
    log(f"           TAIL curve "+" ".join(f"{i}:{ct[i]:.4f}" for i in range(MAXQ+1)))
log("="*70)
log(f"CONTINUOUS ADAPTIVITY PRIZE (concept, dtd - twin): TAIL {res[('concept','dtd')][0][8]-res[('concept','twin')][0][8]:+.4f}  FULL {res[('concept','dtd')][1][8]-res[('concept','twin')][1][8]:+.4f}")
log(f"CONTINUOUS ADAPTIVITY PRIZE (both,    dtd - twin): TAIL {res[('both','dtd')][0][8]-res[('both','twin')][0][8]:+.4f}  FULL {res[('both','dtd')][1][8]-res[('both','twin')][1][8]:+.4f}")
log(f"CONCEPT+ITEM via drill (both,dtd - concept,var champ): TAIL {res[('both','dtd')][0][8]-res[('concept','var')][0][8]:+.4f}  FULL {res[('both','dtd')][1][8]-res[('concept','var')][1][8]:+.4f}")
log("done")
