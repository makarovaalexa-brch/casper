"""
Established acceptance tests applied to instrument v2 (ML-100k, movie-only).
  T2  reveal-monotonicity  : metrics rise with each true reveal (fine-grained K)
  T3  franchise flip       : like(anchor) ranks its sequels higher than dislike(anchor)
  Risers/Genre coherence   : reveal an iconic movie -> top risers same-genre (Crowe-style)
Loads instrument_v2_ml100k_s42.pt.
"""
import os, sys
import numpy as np, torch, torch.nn as nn
base = 'C:/dev/phd/casper/data/movielens/ml-100k'
CK = 'C:/dev/phd/casper/data/movielens/.cache/checkpoints/instrument_v2_ml100k_s42.pt'
rng = np.random.default_rng(0)

# ids (identical to instrument_v2)
U, I, R = [], [], []
with open(f'{base}/u.data') as f:
    for line in f:
        a = line.strip().split('\t'); U.append(int(a[0])); I.append(int(a[1])); R.append(float(a[2]))
U = np.array(U); I = np.array(I); R = np.array(R, np.float32)
uids = {x: k for k, x in enumerate(np.unique(U))}; iids = {x: k for k, x in enumerate(np.unique(I))}
u = np.array([uids[x] for x in U]); i = np.array([iids[x] for x in I]); nu, ni = len(uids), len(iids)
# titles + genres from u.item
GEN = ['unknown','Action','Adventure','Animation','Children','Comedy','Crime','Documentary','Drama','Fantasy','Film-Noir','Horror','Musical','Mystery','Romance','Sci-Fi','Thriller','War','Western']
title = {}; genres = {}
with open(f'{base}/u.item', encoding='latin-1') as f:
    for line in f:
        p = line.rstrip('\n').split('|')
        mid = int(p[0])
        if mid not in iids: continue
        idx = iids[mid]; title[idx] = p[1]
        flags = [int(x) for x in p[5:24]]
        genres[idx] = set(GEN[k] for k in range(19) if flags[k])
_ck = torch.load(CK); mu_ck = _ck['mu']; D = _ck['d']; TIE = _ck.get('tie', False)
ratings_by = {}
for k in range(len(u)): ratings_by.setdefault(u[k], {})[i[k]] = R[k]

class InstrumentV2(nn.Module):
    def __init__(s, ni, d):
        super().__init__()
        s.item=nn.Embedding(ni,d); s.rate=nn.Embedding(6,d); s.cls=nn.Parameter(torch.randn(d)*0.1)
        s.tf=nn.TransformerEncoder(nn.TransformerEncoderLayer(d,4,2*d,0.2,batch_first=True),2)
        s.rank_bias=nn.Parameter(torch.zeros(ni))
        if not TIE: s.rank_head=nn.Linear(d,ni)
        s.rate_head=nn.Linear(d,ni); s.rate_bias=nn.Parameter(torch.zeros(ni))
    def body(s, it_, rt_, m):
        tok=s.item(it_)+s.rate(rt_); B=tok.shape[0]
        x=torch.cat([s.cls.expand(B,1,-1),tok],1); pad=torch.cat([torch.zeros(B,1,dtype=torch.bool),m],1)
        return s.tf(x,src_key_padding_mask=pad)[:,0]
    def forward(s, it_, rt_, m):
        h=s.body(it_,rt_,m)
        rank = s.rank_bias + (h @ s.item.weight.t() if TIE else s.rank_head(h))
        return rank, mu_ck+s.rate_bias+s.rate_head(h.detach())
m = InstrumentV2(ni, D); m.load_state_dict(torch.load(CK)['state']); m.eval()
def rank_scores(rev):
    if rev: ci=torch.tensor([[j for j,_ in rev]]); cr=torch.tensor([[int(round(r)) for _,r in rev]]); cm=torch.zeros_like(ci,dtype=torch.bool)
    else: ci=torch.zeros(1,1,dtype=torch.long); cr=torch.zeros(1,1,dtype=torch.long); cm=torch.ones(1,1,dtype=torch.bool)
    with torch.no_grad(): return m(ci,cr,cm)[0][0].numpy()
def nm(idx): return title.get(idx, f'movie_{idx}')

def find(sub):
    for idx, t in title.items():
        if sub.lower() in t.lower(): return idx
    return None

print("="*78); print("RISERS / GENRE COHERENCE (Crowe-style: reveal LIKE an iconic film)"); print("="*78)
base_sc = rank_scores([])
for anchor_sub in ['Star Wars', 'Toy Story', 'Silence of the Lambs', 'Fargo', 'Nightmare on Elm']:
    a = find(anchor_sub)
    if a is None: continue
    d = rank_scores([(a, 5.0)]) - base_sc; d[a] = -1e9
    up = np.argsort(-d)[:8]
    ag = genres.get(a, set())
    share = np.mean([len(genres.get(int(e), set()) & ag) > 0 for e in np.argsort(-d)[:20]])
    print(f"\n>>> LIKE {nm(a)}  genres={sorted(ag)}")
    print(f"    top-20 risers sharing >=1 genre: {share*100:.0f}%")
    for e in up: print(f"      +{d[e]:.3f}  {nm(int(e)):<45} {sorted(genres.get(int(e),set()))}")

print("\n"+"="*78); print("T3 FRANCHISE FLIP: like(anchor) should rank sequels higher than dislike(anchor)"); print("="*78)
fr = {'Star Wars': ['Empire Strikes Back', 'Return of the Jedi'],
      'Godfather': ['Godfather: Part II'],
      'Raiders of the Lost Ark': ['Indiana Jones', 'Last Crusade', 'Temple of Doom'],
      'Back to the Future': ['Back to the Future Part'],
      'Star Trek': ['Star Trek'],
      'Die Hard': ['Die Hard 2', 'Die Hard: With']}
correct = tot = 0
for anc, subs in fr.items():
    a = find(anc)
    if a is None: continue
    rel = sorted({find(s) for s in subs} - {None, a})
    rel = [r for r in rel if r is not None]
    if not rel: continue
    sl = rank_scores([(a, 5.0)]); sd = rank_scores([(a, 1.0)])
    def rk(sc, e): return 1 + int((sc >= sc[e]).sum())   # rank among all (1=top)
    line = []
    for r in rel:
        rl = rk(sl, r); rd = rk(sd, r); ok = rl < rd; correct += ok; tot += 1
        line.append(f"{nm(r)[:28]} like#{rl} vs dislike#{rd} {'OK' if ok else 'X'}")
    print(f"\nanchor: {nm(a)}")
    for L in line: print("   ", L)
print(f"\nT3 direction-correct: {correct}/{tot} = {100*correct/max(tot,1):.0f}%  (paper gate >=70%)")

print("\n"+"="*78); print("T2 PER-REVEAL METRICS (reveal K random TRUE ratings; cold-test users)"); print("="*78)
split_rng = np.random.default_rng(0)
cold = set(split_rng.choice(nu, 200, replace=False).tolist())
def ndcg_rec(pr, rel, unr):
    nd = rc = 0.
    for ri in rel:
        rank = 1 + int((pr[unr] >= pr[ri]).sum())
        if rank <= 10: nd += 1./np.log2(rank+1); rc += 1.
    return nd/len(rel), rc/len(rel)
KS = list(range(0, 13))
agg = {K: {'nd': [], 'rc': []} for K in KS}
for cu in cold:
    rd = ratings_by[cu]; items = list(rd.keys())
    if len(items) < 20: continue
    ii = items[:]; rng.shuffle(ii); test = set(ii[:max(5, len(ii)//3)]); pool = [x for x in ii if x not in test]
    rel = [j for j in test if rd[j] >= 4]; unr = np.array([j for j in range(ni) if j not in rd])
    if not rel: continue
    for K in KS:
        nd, rc = ndcg_rec(rank_scores([(e, rd[e]) for e in pool[:K]]), rel, unr)
        agg[K]['nd'].append(nd); agg[K]['rc'].append(rc)
print(f"{'K':>3} {'NDCG@10':>9} {'Recall@10':>10}")
for K in KS: print(f"{K:>3} {np.mean(agg[K]['nd']):>9.4f} {np.mean(agg[K]['rc']):>10.4f}")
nds = [np.mean(agg[K]['nd']) for K in KS]
print(f"\nstrictly rising on {sum(nds[i+1]>=nds[i]-1e-4 for i in range(len(nds)-1))}/{len(nds)-1} steps")
