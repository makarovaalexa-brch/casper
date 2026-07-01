"""
R3 — FAITHFUL replication of EDDI / Partial-VAE (Ma et al., ICML 2019) on its OWN kind of ruler
(tabular active feature acquisition; sklearn California-housing as a standard UCI-style regression).
Reproduces the paper's HEADLINE claim: information-reward active acquisition reaches low target error with
FEWER observed features than random acquisition (the EDDI info-curve dominates random).

This validates the permutation-invariant set-encoder over an arbitrary observed SUBSET (the exact mechanism
the CASPER-U instrument needs), with partial observation made in-distribution via random masking.
Reports: target RMSE vs #features acquired, for ACTIVE (EDDI info-gain) vs RANDOM. Active should dominate.
"""
import os, time, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from sklearn.datasets import fetch_california_housing
rng=np.random.default_rng(0); torch.manual_seed(0)
ZD=int(os.environ.get('ZD',10)); H=64; KEMB=16; EP=int(os.environ.get('EP',40)); NZ=20; NXI=5; NTEST=int(os.environ.get('NTEST',300))
data=fetch_california_housing(); X=data.data.astype(np.float32); y=data.target.astype(np.float32)
A=np.concatenate([X,y[:,None]],1)                      # variables = 8 features + target (col 8)
mu_=A.mean(0); sd_=A.std(0)+1e-6; A=(A-mu_)/sd_        # standardize all variables
Dv=A.shape[1]; TGT=Dv-1; FEATS=list(range(Dv-1))       # acquire among features 0..7; predict var 8
n=len(A); idx=rng.permutation(n); tr=idx[:int(0.8*n)]; te=idx[int(0.8*n):int(0.8*n)+NTEST]
Atr=torch.tensor(A[tr]); Ate=torch.tensor(A[te])
print(f"EDDI replication | California-housing | {Dv} vars (8 feat + target), train {len(tr)} test {len(te)}",flush=True)

class PartialVAE(nn.Module):
    def __init__(s):
        super().__init__()
        s.femb=nn.Embedding(Dv,KEMB)                                   # per-variable id embedding s_j
        s.enc=nn.Sequential(nn.Linear(KEMB+1,H),nn.ReLU(),nn.Linear(H,H))   # h([s_j, x_j])
        s.head=nn.Linear(H,2*ZD)
        s.dec=nn.Sequential(nn.Linear(ZD,H),nn.ReLU(),nn.Linear(H,Dv))
        s.logsig=nn.Parameter(torch.zeros(Dv))                          # per-var obs noise
    def encode(s,x,m):                                                  # x [B,Dv], m [B,Dv] observed mask (float)
        B=x.shape[0]; ids=torch.arange(Dv).expand(B,Dv)
        tok=torch.cat([s.femb(ids), x.unsqueeze(-1)],-1)               # [B,Dv,KEMB+1]
        e=s.enc(tok)*m.unsqueeze(-1)                                    # zero-out unobserved (permutation-inv sum)
        c=e.sum(1)                                                      # [B,H]
        mlv=s.head(c); return mlv[:,:ZD], mlv[:,ZD:]
    def decode(s,z): return s.dec(z)
def elbo(model,x):
    m=(torch.rand_like(x)<torch.rand(x.shape[0],1)*0.8+0.1).float()    # random observe prob per row in [0.1,0.9]
    muz,lvz=model.encode(x*m,m); z=muz+torch.randn_like(muz)*torch.exp(0.5*lvz)
    xh=model.decode(z); inv=2*model.logsig
    nll=(((x-xh)**2)/torch.exp(inv)+inv)*m                              # gaussian recon on OBSERVED vars
    kl=-0.5*(1+lvz-muz**2-lvz.exp())
    return (nll.sum(1)+kl.sum(1)).mean()
model=PartialVAE(); opt=torch.optim.Adam(model.parameters(),2e-3); t0=time.time()
for ep in range(EP):
    perm=torch.randperm(len(Atr))
    for b in range(0,len(perm),256):
        x=Atr[perm[b:b+256]]; opt.zero_grad(); l=elbo(model,x); l.backward(); opt.step()
    if (ep+1)%10==0: print(f"  ep{ep+1} elbo={l.item():.3f} ({time.time()-t0:.0f}s)",flush=True)
model.eval()
def target_pred(x,m):                                                   # predictive mean+var of target var given observed
    with torch.no_grad():
        muz,lvz=model.encode(x*m,m); zs=muz.unsqueeze(0)+torch.randn(NZ,*muz.shape)*torch.exp(0.5*lvz)
        xh=model.decode(zs.reshape(-1,ZD)).reshape(NZ,x.shape[0],Dv)[:,:,TGT]   # [NZ,B]
    return xh.mean(0), xh.var(0)+torch.exp(2*model.logsig[TGT])
def acquire(active):
    B=Ate.shape[0]; m=torch.zeros(B,Dv); m[:,TGT]=0.                    # target never observed
    x=Ate*m; rmse=[]; order_pool=[list(FEATS) for _ in range(B)]
    # step 0 (no features)
    pm,_=target_pred(x,m); rmse.append(torch.sqrt(((pm-Ate[:,TGT])**2).mean()).item())
    for step in range(len(FEATS)):
        pick=np.zeros(B,dtype=int)
        for b in range(B):
            cands=order_pool[b]
            if not active:
                pick[b]=cands[int(rng.integers(len(cands)))]
            else:
                _,curvar=target_pred(x[b:b+1],m[b:b+1]); best=-1; bi=cands[0]
                for i in cands:                                        # EDDI info reward: expected target-var reduction
                    with torch.no_grad():
                        muz,lvz=model.encode((x[b:b+1])*m[b:b+1],m[b:b+1]); zs=muz+torch.randn(NXI,ZD)*torch.exp(0.5*lvz)
                        xi=model.decode(zs)[:,i]                       # sampled values for candidate feature i
                    newvars=[]
                    for s_ in range(NXI):
                        mm=m[b:b+1].clone(); xx=x[b:b+1].clone(); mm[0,i]=1.; xx[0,i]=xi[s_]
                        _,nv=target_pred(xx,mm); newvars.append(nv.item())
                    gain=curvar.item()-float(np.mean(newvars))
                    if gain>best: best=gain; bi=i
                pick[b]=bi
            order_pool[b].remove(pick[b]); m[b,pick[b]]=1.; x[b,pick[b]]=Ate[b,pick[b]]
        pm,_=target_pred(x,m); rmse.append(torch.sqrt(((pm-Ate[:,TGT])**2).mean()).item())
    return rmse
print("\nacquiring (active=EDDI info-gain vs random)...",flush=True)
ra=acquire(True); rr=acquire(False)
print("\n=== EDDI replication: target RMSE vs #features acquired (lower=better) ===",flush=True)
print(f"{'#feat':>5} {'ACTIVE(EDDI)':>13} {'RANDOM':>8}",flush=True)
for k in range(len(ra)): print(f"{k:>5} {ra[k]:>13.4f} {rr[k]:>8.4f}{'   <- active better' if ra[k]<rr[k]-1e-4 else ''}",flush=True)
auc_a=np.mean(ra); auc_r=np.mean(rr)
print(f"\nmean-RMSE-over-curve: ACTIVE {auc_a:.4f} vs RANDOM {auc_r:.4f} -> EDDI active {'REPLICATES (dominates)' if auc_a<auc_r else 'FAILS'} ({auc_r-auc_a:+.4f})",flush=True)
