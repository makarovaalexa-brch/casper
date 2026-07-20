"""Seed-averaged static LLM-asker scoring on the test ruler (te[300:] x 5 seeds), ALL metrics:
NDCG@10 + Recall@50 + MRR (full & tail), cos(belief,u*), answered/8. Same encoder + recommender as the harness.
Adaptive-haiku EXCLUDED (n=30 probe, too noisy). Robust JSON parse of the pick files."""
import os, json, glob, re, pickle, numpy as np, torch, torch.nn as nn
base='C:/dev/phd/casper/data/movielens'; D=64
d=pickle.load(open(f'{base}/.cache/llm_data_seedavg.pkl','rb'))
data=d['data']; Ec=d['Ec']; Q=d['Q']; Ql=d['Ql']; popb=d['popb']; headmask=d['headmask']; c2=d['concept_name2c']; t2=d['title2iid']
class Enc(nn.Module):
    def __init__(s):
        super().__init__(); s.inp=nn.Sequential(nn.Linear(D+1,128),nn.ReLU(),nn.Linear(128,128),nn.ReLU()); s.att=nn.Linear(128,1); s.val=nn.Linear(128,D)
    def forward(s,t,m):
        h=s.inp(t); a=s.att(h).squeeze(-1).masked_fill(m==0,-1e9); al=torch.softmax(a,1); return (al.unsqueeze(-1)*s.val(h)).sum(1)
enc=Enc(); enc.load_state_dict(torch.load(f'{base}/.cache/enc_concept.pt')); enc.eval()
def enc_u(toks):
    if not toks: return np.zeros(D,np.float32)
    arr=np.zeros((1,len(toks),D+1),np.float32); m=np.ones((1,len(toks)),np.float32)
    for q,(ff,v) in enumerate(toks): arr[0,q,:D]=ff; arr[0,q,D]=v
    with torch.no_grad(): return enc(torch.tensor(arr),torch.tensor(m)).numpy()[0]
_W=1./np.log2(np.arange(2,12))
def metr(u_,tlike,known,tail):
    s=(popb+Ql@u_).copy(); s[list(known)]=-1e9
    if tail: s[headmask]=-1e9; rel=set(t for t in tlike if not headmask[t])
    else: rel=set(tlike)
    if not rel: return None
    o=np.argsort(-s); nd=sum(_W[p] for p,t in enumerate(o[:10]) if int(t) in rel)/(_W[:min(10,len(rel))].sum()+1e-12)
    rc=len(set(o[:50].tolist())&rel)/len(rel); mrr=0.
    for p,t in enumerate(o):
        if int(t) in rel: mrr=1./(p+1); break
    return nd,rc,mrr
def resolve(entry,u):
    name=entry; low=entry.lower()
    if low.startswith('concept:'): name=entry[8:].strip()
    elif low.startswith('movie:'): name=entry[6:].strip()
    if name.lower() in c2:
        cc=c2[name.lower()]; return (Ec[cc],u['cans'][cc]) if cc in u['cans'] else None
    if name in t2:
        iid=t2[name]; return (Q[iid],u['resid'][iid]) if iid in set(u['known']) else None
    return None
def load_picks(f):
    m=re.search(r'\[.*\]', open(f,encoding='utf-8').read(), re.DOTALL); return json.loads(m.group(0)) if m else []
def score(method,build):
    A={k:[0.,0] for k in ['ndcg_f','ndcg_t','rec_f','rec_t','mrr_f','mrr_t','cos','ans']}
    def add(k,v):
        if v is not None: A[k][0]+=v; A[k][1]+=1
    for sd,recs in data.items():
        for u in recs:
            toks=build(u)
            if toks is None: continue
            bel=enc_u(toks); ustar=enc_u([(Q[j],u['resid'][j]) for j in u['known']])
            mf=metr(bel,set(u['tlike']),set(u['known']),False); mt=metr(bel,set(u['tlike']),set(u['known']),True)
            if mf: add('ndcg_f',mf[0]); add('rec_f',mf[1]); add('mrr_f',mf[2])
            if mt: add('ndcg_t',mt[0]); add('rec_t',mt[1]); add('mrr_t',mt[2])
            nb=np.linalg.norm(bel); nu=np.linalg.norm(ustar)
            add('cos', float(bel@ustar/(nb*nu)) if nb>1e-9 and nu>1e-9 else 0.); add('ans',float(len(toks)))
    m=lambda k: A[k][0]/max(A[k][1],1)
    print(f"  {method:<16} NDCG {m('ndcg_f'):.3f}/{m('ndcg_t'):.3f}  Rec {m('rec_f'):.3f}/{m('rec_t'):.3f}  MRR {m('mrr_f'):.3f}/{m('mrr_t'):.3f}  cos {m('cos'):.3f}  ans {m('ans'):.1f}/8")
n=sum(len(r) for r in data.values()); print(f"=== STATIC LLM-asker seed-avg (test te[300:], seeds {d['seeds']}, {n} user-evals) | full/tail ===")
print("  asker            NDCG@10      Rec@50       MRR          cos    ans")
score('q0 (MostPop)', lambda u: [])
score('fullprof', lambda u: [(Q[j],u['resid'][j]) for j in u['known']])
for f in sorted(glob.glob(f'{base}/.cache/llm_uni_*.txt')):
    model=os.path.basename(f)[8:-4]; picks=load_picks(f)
    score(f'static-{model}', lambda u,picks=picks: [r for r in (resolve(p,u) for p in picks) if r])
