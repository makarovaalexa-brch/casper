"""
Elicitation-strategy benchmark on the ACCEPTED instrument (v2 tied, MF-init, ml-100k).
Cheap strategies first: random / popularity / HELF / greedy-infogain.
Metric: per-turn full-catalogue NDCG@10 / Recall@10 on cold-test held-out likes.
The instrument is the FIXED recommender; only the question-selection varies.
"""
import os, numpy as np, torch, torch.nn as nn
base = 'C:/dev/phd/casper/data/movielens/ml-100k'
CK = 'C:/dev/phd/casper/data/movielens/.cache/checkpoints/instrument_v2_ml100k_s42.pt'
T = 20; N_COLD = 200; rng = np.random.default_rng(0)
U, I, R = [], [], []
with open(f'{base}/u.data') as f:
    for line in f:
        a = line.strip().split('\t'); U.append(int(a[0])); I.append(int(a[1])); R.append(float(a[2]))
U = np.array(U); I = np.array(I); R = np.array(R, np.float32)
uids = {x: k for k, x in enumerate(np.unique(U))}; iids = {x: k for k, x in enumerate(np.unique(I))}
u = np.array([uids[x] for x in U]); i = np.array([iids[x] for x in I]); nu, ni = len(uids), len(iids)
cnt = np.bincount(i, minlength=ni); pop_order = list(np.argsort(-cnt))
# HELF order
fl = np.array([R[i == j].mean() if cnt[j] else 3. for j in range(ni)])
var = np.array([R[i == j].var() if cnt[j] > 1 else 0. for j in range(ni)])
lf = np.log1p(cnt)/np.log1p(cnt.max()); helf = 2*lf*var/(lf+var+1e-9); helf_order = list(np.argsort(-helf))
ratings_by = {}
for k in range(len(u)): ratings_by.setdefault(u[k], {})[i[k]] = R[k]
split_rng = np.random.default_rng(0); cold = list(split_rng.choice(nu, N_COLD, replace=False))

_ck = torch.load(CK); mu_ck = _ck['mu']; D = _ck['d']; TIE = _ck.get('tie', False)
class Inst(nn.Module):
    def __init__(s, ni, d):
        super().__init__(); s.item=nn.Embedding(ni,d); s.rate=nn.Embedding(6,d); s.cls=nn.Parameter(torch.randn(d)*0.1)
        s.tf=nn.TransformerEncoder(nn.TransformerEncoderLayer(d,4,2*d,0.2,batch_first=True),2)
        s.rank_bias=nn.Parameter(torch.zeros(ni))
        if not TIE: s.rank_head=nn.Linear(d,ni)
        s.rate_head=nn.Linear(d,ni); s.rate_bias=nn.Parameter(torch.zeros(ni))
    def forward(s, it_, rt_, m):
        tok=s.item(it_)+s.rate(rt_); B=tok.shape[0]
        x=torch.cat([s.cls.expand(B,1,-1),tok],1); pad=torch.cat([torch.zeros(B,1,dtype=torch.bool),m],1)
        h=s.tf(x,src_key_padding_mask=pad)[:,0]
        return s.rank_bias+(h@s.item.weight.t() if TIE else s.rank_head(h)), mu_ck+s.rate_bias+s.rate_head(h.detach())
M = Inst(ni, D); M.load_state_dict(_ck['state'], strict=False); M.eval()
def scores(rev):
    if rev: ci=torch.tensor([[j for j,_ in rev]]); cr=torch.tensor([[int(round(r)) for _,r in rev]]); cm=torch.zeros_like(ci,dtype=torch.bool)
    else: ci=torch.zeros(1,1,dtype=torch.long); cr=torch.zeros(1,1,dtype=torch.long); cm=torch.ones(1,1,dtype=torch.bool)
    with torch.no_grad(): return M(ci,cr,cm)[0][0].numpy()

# build cold cases (fixed held-out 30%)
cases = []
for cu in cold:
    rd = ratings_by[cu]; items = list(rd.keys())
    if len(items) < 10: continue
    ii = items[:]; rng.shuffle(ii); ncut = max(5, int(0.3*len(ii)))
    test = set(ii[:ncut]); rel = [j for j in test if rd[j] >= 4]
    if not rel: continue
    unr = np.array([j for j in range(ni) if j not in rd])
    cases.append((rd, test, rel, unr))
print(f"cold cases: {len(cases)}", flush=True)

def ndcg_rec(pr, rel, unr):
    nd = rc = 0.
    for ri in rel:
        rank = 1 + int((pr[unr] >= pr[ri]).sum())
        if rank <= 10: nd += 1./np.log2(rank+1); rc += 1.
    return nd/len(rel), rc/len(rel)

def run(name, order_fn):
    nd = np.zeros(T+1); rc = np.zeros(T+1)
    for (rd, test, rel, unr) in cases:
        order = order_fn(rd, test); ans = []
        for t in range(T+1):
            a, b = ndcg_rec(scores(ans), rel, unr); nd[t] += a; rc[t] += b
            if t == T: break
            q = order[t] if t < len(order) else None
            if q is not None and q in rd: ans.append((q, rd[q]))
    nd /= len(cases); rc /= len(cases)
    print(f"  {name:<14} NDCG q0={nd[0]:.3f} q5={nd[5]:.3f} q10={nd[10]:.3f} q20={nd[20]:.3f} | "
          f"Rec q5={rc[5]:.3f} q20={rc[20]:.3f}", flush=True)
    return nd, rc

def rand_order(rd, test):
    o = [x for x in range(ni) if x not in test]; rng.shuffle(o); return o
def fixed_order(order):
    return lambda rd, test: [x for x in order if x not in test]

print("=== CHEAP elicitation strategies on accepted instrument (cold-test) ===", flush=True)
res = {}
res['random'] = run('random', rand_order)
res['popularity'] = run('popularity', fixed_order(pop_order))
res['HELF'] = run('HELF', fixed_order(helf_order))
import json
json.dump({k: {'ndcg': list(v[0]), 'rec': list(v[1])} for k, v in res.items()},
          open('C:/dev/phd/casper/experiments/paper1/elicit_cheap_ml100k.json', 'w'))
print("saved elicit_cheap_ml100k.json")
