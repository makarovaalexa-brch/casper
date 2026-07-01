"""
CASPER on the accepted ml-100k instrument: learned elicitation policy.
Teacher = clairvoyant oracle (pick the answerable movie whose true reveal most
improves held-out NDCG). Student = transformer actor over revealed (item,rating)
-> next-question logits, distilled by behaviour cloning (Choudhury clairvoyant
imitation). Trained on WARM users, evaluated on COLD-test vs popularity.
Instrument is FROZEN.
"""
import os, time, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
base = 'C:/dev/phd/casper/data/movielens/ml-100k'
CK = 'C:/dev/phd/casper/data/movielens/.cache/checkpoints/instrument_v2_ml100k_s42.pt'
T = 15; N_TEACH = int(os.environ.get('N_TEACH', 450)); DEPOCHS = int(os.environ.get('DEPOCHS', 15))
N_COLD = 200; D = 48; rng = np.random.default_rng(0); torch.manual_seed(0)
U, I, R = [], [], []
with open(f'{base}/u.data') as f:
    for line in f:
        a = line.strip().split('\t'); U.append(int(a[0])); I.append(int(a[1])); R.append(float(a[2]))
U = np.array(U); I = np.array(I); R = np.array(R, np.float32)
uids = {x: k for k, x in enumerate(np.unique(U))}; iids = {x: k for k, x in enumerate(np.unique(I))}
u = np.array([uids[x] for x in U]); i = np.array([iids[x] for x in I]); nu, ni = len(uids), len(iids)
cnt = np.bincount(i, minlength=ni); pop_order = list(np.argsort(-cnt))
ratings_by = {}
for k in range(len(u)): ratings_by.setdefault(u[k], {})[i[k]] = R[k]
split_rng = np.random.default_rng(0); cold = set(split_rng.choice(nu, N_COLD, replace=False).tolist())
warm = [x for x in range(nu) if x not in cold]

# ---- frozen instrument ----
_ck = torch.load(CK); mu_ck = _ck['mu']; DI = _ck['d']; TIE = _ck.get('tie', False)
class Inst(nn.Module):
    def __init__(s, ni, d):
        super().__init__(); s.item=nn.Embedding(ni,d); s.rate=nn.Embedding(6,d); s.cls=nn.Parameter(torch.randn(d)*0.1)
        s.tf=nn.TransformerEncoder(nn.TransformerEncoderLayer(d,4,2*d,0.2,batch_first=True),2)
        s.rank_bias=nn.Parameter(torch.zeros(ni))
        if not TIE: s.rank_head=nn.Linear(d,ni)
        s.rate_head=nn.Linear(d,ni); s.rate_bias=nn.Parameter(torch.zeros(ni))
    def enc(s, ci, cr, cm):
        tok=s.item(ci)+s.rate(cr); B=tok.shape[0]
        x=torch.cat([s.cls.expand(B,1,-1),tok],1); pad=torch.cat([torch.zeros(B,1,dtype=torch.bool),cm],1)
        h=s.tf(x,src_key_padding_mask=pad)[:,0]
        return s.rank_bias+(h@s.item.weight.t() if TIE else s.rank_head(h))
W = Inst(ni, DI); W.load_state_dict(_ck['state'], strict=False); W.eval()
def pad_batch(revs):
    L = max(1, max(len(r) for r in revs)); B = len(revs)
    ci=np.zeros((B,L),np.int64); cr=np.zeros((B,L),np.int64); cm=np.ones((B,L),bool)
    for b,r in enumerate(revs):
        for j,(e,v) in enumerate(r): ci[b,j]=e; cr[b,j]=int(round(v)); cm[b,j]=False
    return torch.from_numpy(ci),torch.from_numpy(cr),torch.from_numpy(cm)
def inst_scores_batch(revs):
    ci,cr,cm = pad_batch([r if r else [] for r in revs])
    # empty rows: mark one padded token (already all-masked) -> fine
    with torch.no_grad(): return W.enc(ci,cr,cm).numpy()
def ndcg(pr, rel, unr):
    return np.mean([(1./np.log2(2+int((pr[unr]>=pr[ri]).sum())) if 1+int((pr[unr]>=pr[ri]).sum())<=10 else 0.) for ri in rel])

# ---- collect clairvoyant-oracle demos on warm users ----
print(f"collecting oracle demos on {N_TEACH} warm users...", flush=True)
tr = list(rng.choice(warm, min(N_TEACH, len(warm)), replace=False)); t0=time.time()
X, A = [], []
for n, cu in enumerate(tr):
    rd = ratings_by[cu]; items = list(rd.keys())
    if len(items) < 12: continue
    ii=items[:]; rng.shuffle(ii); ncut=max(5,int(0.3*len(ii))); test=set(ii[:ncut]); pool=[x for x in ii if x not in test]
    rel=[j for j in test if rd[j]>=4]
    if not rel: continue
    unr=np.array([j for j in range(ni) if j not in rd]); rev=[]; asked=set()
    for _ in range(T):
        cands=[c for c in pool if c not in asked]
        if not cands: break
        sets=[rev+[(c, rd[c])] for c in cands]; sc=inst_scores_batch(sets)
        nd=[ndcg(sc[k], rel, unr) for k in range(len(cands))]
        best=cands[int(np.argmax(nd))]
        X.append(rev.copy()); A.append(best); asked.add(best); rev.append((best, rd[best]))
    if (n+1)%100==0: print(f"  {n+1}/{len(tr)} ({time.time()-t0:.0f}s, {len(X)} demos)", flush=True)
print(f"{len(X)} demos", flush=True)

# ---- actor: transformer over revealed -> next-question logits ----
class Actor(nn.Module):
    def __init__(s, ni, d):
        super().__init__(); s.item=nn.Embedding(ni,d); s.rate=nn.Embedding(6,d); s.cls=nn.Parameter(torch.randn(d)*0.1)
        s.tf=nn.TransformerEncoder(nn.TransformerEncoderLayer(d,4,2*d,0.1,batch_first=True),2)
        s.head=nn.Linear(d,ni)
    def forward(s, ci, cr, cm):
        tok=s.item(ci)+s.rate(cr); B=tok.shape[0]
        x=torch.cat([s.cls.expand(B,1,-1),tok],1); pad=torch.cat([torch.zeros(B,1,dtype=torch.bool),cm],1)
        return s.head(s.tf(x,src_key_padding_mask=pad)[:,0])
actor = Actor(ni, D); opt = torch.optim.Adam(actor.parameters(), 1e-3, weight_decay=1e-5)
At = torch.tensor([int(x) for x in A], dtype=torch.long)
for ep in range(DEPOCHS):
    actor.train(); idx = torch.randperm(len(X))
    tot=0; nb=0
    for s in range(0, len(idx), 128):
        bi = idx[s:s+128].tolist()
        ci,cr,cm = pad_batch([X[k] for k in bi]); tgt = At[idx[s:s+128]]
        opt.zero_grad(); logit = actor(ci,cr,cm)
        # mask already-revealed in each row
        for r_, row in enumerate([X[k] for k in bi]):
            for (e,_) in row: logit[r_, e] = -1e9
        loss = F.cross_entropy(logit, tgt); loss.backward(); opt.step(); tot+=loss.item(); nb+=1
    with torch.no_grad():
        ci,cr,cm = pad_batch([X[k] for k in range(min(1000,len(X)))])
        acc = (actor(ci,cr,cm).argmax(1) == At[:min(1000,len(X))]).float().mean().item()
    print(f"  ep{ep+1}/{DEPOCHS} loss={tot/nb:.3f} train-acc={acc:.3f}", flush=True)
actor.eval()

# ---- eval on cold-test ----
cases=[]
for cu in cold:
    rd=ratings_by[cu]; items=list(rd.keys())
    if len(items)<10: continue
    ii=items[:]; rng.shuffle(ii); ncut=max(5,int(0.3*len(ii))); test=set(ii[:ncut]); rel=[j for j in test if rd[j]>=4]
    if not rel: continue
    cases.append((rd, test, rel, np.array([j for j in range(ni) if j not in rd])))
def run(name, pickfn):
    nd=np.zeros(T+1); rc=np.zeros(T+1)
    for (rd,test,rel,unr) in cases:
        rev=[]; asked=set(test)
        for t in range(T+1):
            sc=inst_scores_batch([rev])[0]
            n_=ndcg(sc,rel,unr); nd[t]+=n_
            rc[t]+=np.mean([1. if 1+int((sc[unr]>=sc[ri]).sum())<=10 else 0. for ri in rel])
            if t==T: break
            q=pickfn(rev,asked,rd)
            if q is None: continue
            asked.add(q)
            if q in rd: rev.append((q, rd[q]))
    nd/=len(cases); rc/=len(cases)
    print(f"  {name:<12} NDCG q5={nd[5]:.3f} q10={nd[10]:.3f} q15={nd[-1]:.3f} | Rec q15={rc[-1]:.3f}", flush=True)
    return nd, rc
def pick_actor(rev, asked, rd):
    ci,cr,cm = pad_batch([rev]);
    with torch.no_grad(): lg = actor(ci,cr,cm)[0].numpy()
    lg[list(asked)] = -1e9
    for (e,_) in rev: lg[e]=-1e9
    return int(np.argmax(lg))
def pick_pop(rev, asked, rd):
    return next((e for e in pop_order if e not in asked), None)
print(f"\n=== COLD-TEST ({len(cases)} users), CASPER vs popularity ===", flush=True)
run('popularity', pick_pop)
run('CASPER', pick_actor)
