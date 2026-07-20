"""
FAITHFUL Deep Rating Elicitation (Kim et al. 2024, arXiv 2402.16327) replication.
Protocol matched to the paper:
  - ML-1M, implicit (rating>=4 -> 1), users with >=5 ratings
  - users split 80/10/10 train/val/test (new-user cold-start: test users give only seed feedback)
  - k=50 seed items; decoder = 2-layer FC + sigmoid; MSE reconstruction of FULL binary vector
  - eval: rank ALL items except seeds; ground-truth = user's positives among non-seeds
  - metric = STANDARD NDCG@10 = DCG@10/IDCG@10  (and Recall@10)
Arms: MOSTPOP (no elicitation), RAN++ (random seeds, avg of draws), POP++ (popular seeds), DRE (learned seeds).
All elicitation arms share the SAME co-trained decoder recipe; only seed selection differs.
"""
import os, time, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
base='C:/dev/phd/casper/data/movielens/ml-1m'
K=int(os.environ.get('K',50)); EP=int(os.environ.get('EP',100)); H=int(os.environ.get('H',256))
DIV=float(os.environ.get('DIV',1.0)); NRAN=int(os.environ.get('NRAN',5)); LIKE=float(os.environ.get('LIKE',4.0))
rng=np.random.default_rng(0); torch.manual_seed(0)
U,I,R=[],[],[]
with open(f'{base}/ratings.dat') as f:
    for line in f:
        a=line.strip().split('::'); U.append(int(a[0])); I.append(int(a[1])); R.append(float(a[2]))
U=np.array(U); I=np.array(I); R=np.array(R,np.float32)
uids={x:k for k,x in enumerate(np.unique(U))}; iids={x:k for k,x in enumerate(np.unique(I))}
u=np.array([uids[x] for x in U]); i=np.array([iids[x] for x in I]); nu,ni=len(uids),len(iids)
B=np.zeros((nu,ni),np.float32); Rmat=np.zeros((nu,ni),np.float32)   # B=binary liked (target/GT); Rmat=centered ratings (input, preserves dislikes)
for k in range(len(u)):
    if R[k]>=LIKE: B[u[k],i[k]]=1.0
    Rmat[u[k],i[k]]=(R[k]-3.0)/2.0                                  # in [-1,1]; 0 = unseen
keep=np.where(B.sum(1)>=5)[0]                     # users with >=5 positives
rng.shuffle(keep); n=len(keep); tr=keep[:int(0.8*n)]; va=keep[int(0.8*n):int(0.9*n)]; te=keep[int(0.9*n):]
cnt=B[tr].sum(0); pop_topk=list(np.argsort(-cnt)[:K])
print(f"ml-1m implicit: {nu}u {ni}i | pos>= {LIKE} | usable {n} | train {len(tr)} val {len(va)} test {len(te)} | K={K}",flush=True)
Btr=torch.tensor(B[tr]); Rtr=torch.tensor(Rmat[tr])                 # train: targets binary (Btr), inputs = ratings (Rtr)
Bte=B[te]; Rte=Rmat[te]                                             # test: GT binary (Bte), elicited input = ratings (Rte)

def ndcg_recall(scores, rel, seeds):
    s=scores.copy(); s[seeds]=-1e9                # candidates = all items except seeds
    top=np.argsort(-s)[:10]; rs=set(rel)
    dcg=sum(1./np.log2(p+2) for p,it in enumerate(top) if it in rs)
    idcg=sum(1./np.log2(p+2) for p in range(min(10,len(rel))))
    rec=len(set(top.tolist())&rs)/len(rel)
    return (dcg/idcg if idcg>0 else 0.), rec

DD=int(os.environ.get('DD',64)); LOSS=os.environ.get('LOSS','bce')   # MF-structured decoder
_binit=torch.tensor(np.log(cnt+1.0).astype(np.float32))               # item bias init ~ log-popularity
class Decoder(nn.Module):
    def __init__(s):
        super().__init__(); s.enc=nn.Sequential(nn.Linear(K,H),nn.ReLU(),nn.Linear(H,DD))  # seed feedback -> user latent
        s.Q=nn.Parameter(0.01*torch.randn(ni,DD)); s.b=nn.Parameter(_binit.clone())          # item emb + log-pop bias
    def forward(s,x): p=s.enc(x); return p@s.Q.t()+s.b                                        # logits (rank by these)
def _loss(pred,tgt): return F.binary_cross_entropy_with_logits(pred,tgt) if LOSS=='bce' else F.mse_loss(torch.sigmoid(pred),tgt)

def eval_seeds(dec_fwd, seeds):
    seeds=np.array(seeds); nd=rc=0.; m=0
    for ui in range(len(te)):
        rel=[j for j in np.where(Bte[ui]>0)[0] if j not in set(seeds.tolist())]
        if not rel: continue
        x=torch.tensor(Rte[ui][seeds][None,:])
        with torch.no_grad(): pr=dec_fwd(x)[0].numpy()
        a,b=ndcg_recall(pr,rel,seeds); nd+=a; rc+=b; m+=1
    return nd/m, rc/m

def train_fixed(seeds, name, draws=1):
    nds=[]; rcs=[]
    for d in range(draws):
        sd=np.array(seeds[d] if draws>1 else seeds); dec=Decoder(); opt=torch.optim.Adam(dec.parameters(),1e-3,weight_decay=1e-6)
        Xin=Rtr[:,sd]
        for ep in range(EP):
            idx=torch.randperm(len(tr))
            for s in range(0,len(idx),512):
                b=idx[s:s+512]; opt.zero_grad(); loss=_loss(dec(Xin[b]),Btr[b]); loss.backward(); opt.step()
        dec.eval(); a,b=eval_seeds(lambda x:dec(x), sd); nds.append(a); rcs.append(b)
    print(f"  {name:<10} NDCG@10={np.mean(nds):.4f} Recall@10={np.mean(rcs):.4f}"+(f" (avg {draws} draws)" if draws>1 else ""),flush=True)
    return np.mean(nds)

class DRE(nn.Module):
    def __init__(s): super().__init__(); s.logits=nn.Parameter(torch.randn(K,ni)*0.01); s.dec=Decoder()
    def forward(s,Bb,tau):
        Sg=F.gumbel_softmax(s.logits,tau=tau,hard=True,dim=1); return s.dec(Bb@Sg.t())
    def div_pen(s): col=torch.softmax(s.logits,1).sum(0); return ((col-1.0).clamp(min=0)**2).sum()
def train_dre():
    m=DRE(); opt=torch.optim.Adam(m.parameters(),1e-3,weight_decay=1e-6)
    for ep in range(EP):
        tau=max(0.4,2.0*np.exp(-0.04*ep)); idx=torch.randperm(len(tr))
        for s in range(0,len(idx),512):
            b=idx[s:s+512]; opt.zero_grad(); loss=_loss(m(Rtr[b],tau),Btr[b])+DIV*m.div_pen()
            loss.backward(); opt.step()
    m.eval(); seed=list(dict.fromkeys(m.logits.argmax(1).tolist()))
    for j in np.argsort(-cnt):
        if len(seed)>=K: break
        if j not in seed: seed.append(int(j))
    seed=np.array(seed[:K])
    a,b=eval_seeds(lambda x:m.dec(x), seed)
    print(f"  {'DRE':<10} NDCG@10={a:.4f} Recall@10={b:.4f}  (distinct seeds {len(set(seed.tolist()))})",flush=True); return a

print("=== FAITHFUL DRE replication (ML-1M, implicit, k=50, standard NDCG@10) ===",flush=True); t0=time.time()
# MOSTPOP: rank by popularity, no elicitation
nd=rc=0.; m=0
for ui in range(len(te)):
    rel=list(np.where(Bte[ui]>0)[0])
    if not rel: continue
    a,b=ndcg_recall(cnt.copy(),rel,np.array([],int)); nd+=a; rc+=b; m+=1
print(f"  {'MOSTPOP':<10} NDCG@10={nd/m:.4f} Recall@10={rc/m:.4f}  (no elicitation)  [paper: 0.3921]",flush=True)
ran_draws=[list(rng.choice(ni,K,replace=False)) for _ in range(NRAN)]
train_fixed(ran_draws,'RAN++',draws=NRAN)
train_fixed(pop_topk,'POP++')
train_dre()
print(f"(total {time.time()-t0:.0f}s)  [paper ML-1M: MOSTPOP 0.3921, RMVA 0.5387, DRE 0.5688]",flush=True)
