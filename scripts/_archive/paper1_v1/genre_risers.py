"""
'Feed in a genre' test: reveal LIKE for a genre's top exemplar movies (a genre-fan
profile), look at top risers -> are they that genre? Cleaner than single-movie
risers. Run on instrument (v2 tied, MF-init) AND MF for comparison.
"""
import numpy as np, torch, torch.nn as nn
base = 'C:/dev/phd/casper/data/movielens/ml-100k'
CK = 'C:/dev/phd/casper/data/movielens/.cache/checkpoints/instrument_v2_ml100k_s42.pt'
rng = np.random.default_rng(0)
U, I, R = [], [], []
with open(f'{base}/u.data') as f:
    for line in f:
        a = line.strip().split('\t'); U.append(int(a[0])); I.append(int(a[1])); R.append(float(a[2]))
U = np.array(U); I = np.array(I); R = np.array(R, np.float32)
uids = {x: k for k, x in enumerate(np.unique(U))}; iids = {x: k for k, x in enumerate(np.unique(I))}
u = np.array([uids[x] for x in U]); i = np.array([iids[x] for x in I]); nu, ni = len(uids), len(iids)
cnt = np.bincount(i, minlength=ni)
GEN = ['unknown','Action','Adventure','Animation','Children','Comedy','Crime','Documentary','Drama','Fantasy','Film-Noir','Horror','Musical','Mystery','Romance','Sci-Fi','Thriller','War','Western']
title = {}; genres = {}
with open(f'{base}/u.item', encoding='latin-1') as f:
    for line in f:
        p = line.rstrip('\n').split('|'); mid = int(p[0])
        if mid not in iids: continue
        idx = iids[mid]; title[idx] = p[1]; fl = [int(x) for x in p[5:24]]
        genres[idx] = set(GEN[k] for k in range(19) if fl[k])
by_genre = {g: [idx for idx in genres if g in genres[idx]] for g in GEN}

# instrument (tied, MF-init)
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
        return s.rank_bias+(h@s.item.weight.t() if TIE else s.rank_head(h))
M = Inst(ni, D); M.load_state_dict(_ck['state'], strict=False); M.eval()
def inst_pred(rev):
    if rev: ci=torch.tensor([[j for j,_ in rev]]); cr=torch.tensor([[int(round(r)) for _,r in rev]]); cm=torch.zeros_like(ci,dtype=torch.bool)
    else: ci=torch.zeros(1,1,dtype=torch.long); cr=torch.zeros(1,1,dtype=torch.long); cm=torch.ones(1,1,dtype=torch.bool)
    with torch.no_grad(): return M(ci,cr,cm)[0].numpy()

# MF (quick, for comparison)
mu = float(R.mean()); DD = 32
bu=np.zeros(nu); bi=np.zeros(ni); P=0.1*rng.standard_normal((nu,DD)); Q=0.1*rng.standard_normal((ni,DD))
for ep in range(30):
    o=rng.permutation(len(U)); lr=0.008/(1+0.05*ep)
    for s in range(0,len(o),20000):
        b=o[s:s+20000]; uu,ii,rr=u[b],i[b],R[b]
        e=(rr-(mu+bu[uu]+bi[ii]+np.sum(P[uu]*Q[ii],1))).astype(np.float64)
        np.add.at(bu,uu,lr*(e-0.05*bu[uu])); np.add.at(bi,ii,lr*(e-0.05*bi[ii]))
        np.add.at(P,uu,lr*(e[:,None]*Q[ii]-0.05*P[uu])); np.add.at(Q,ii,lr*(e[:,None]*P[uu]-0.05*Q[ii]))
def mf_pred(rev):
    if not rev: return mu+bi
    A=np.array([[1.]+Q[j].tolist() for j,_ in rev]); y=np.array([r-mu-bi[j] for j,r in rev])
    x=np.linalg.solve(A.T@A+4*np.eye(DD+1),A.T@y); return mu+x[0]+bi+Q@x[1:]

def genre_test(pred, name):
    print(f"\n===== {name}: feed genre (like top-6 exemplars) -> risers =====")
    for g in ['Horror','Sci-Fi','Children','Romance','Western','Documentary','Musical','Film-Noir']:
        cand = sorted(by_genre[g], key=lambda x: -cnt[x])[:6]
        if len(cand) < 3: continue
        rev = [(c, 5.0) for c in cand]; d = pred(rev) - pred([])
        for c in cand: d[c] = -1e9
        order = np.argsort(-d); share = np.mean([g in genres.get(int(e), set()) for e in order[:20]])
        tops = ", ".join(title[int(e)][:24] for e in order[:4])
        print(f"  {g:<12} genre-share@20 = {share*100:>3.0f}%   top: {tops}")

genre_test(mf_pred, "MF")
genre_test(inst_pred, "INSTRUMENT (v2 tied, MF-init)")
