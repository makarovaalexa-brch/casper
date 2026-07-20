"""
R2 (scoped) — EAR mechanism replication (Lei et al. WSDM 2020) on LastFM (hetrec2011).
EAR's FM scorer:  y(u, item v, confirmed attrs P_u) = u.v + Sum_{p in P_u} v.p   (one shared FM latent space).
KEY claim to reproduce: conditioning on CONFIRMED ATTRIBUTES improves cold-start recommendation vs
item-only / popularity. (This FM item+attribute scorer is exactly the mechanism CASPER-U ports.)
Data: items=artists, attributes=tags, implicit feedback=user listens. Metric: standard NDCG@10/Recall@10,
cold users, reveal attributes from a revealed half of their artists, rank the held-out half (full-catalogue).
"""
import os, time, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
base='C:/dev/phd/casper/data/lastfm'
NITEM=int(os.environ.get('NITEM',2500)); NATTR=int(os.environ.get('NATTR',50)); D=int(os.environ.get('D',32))
EP=int(os.environ.get('EP',25)); MINU=int(os.environ.get('MINU',10)); TAGMIN=int(os.environ.get('TAGMIN',5))
rng=np.random.default_rng(0); torch.manual_seed(0)
# ---- load implicit listens ----
ua=[]
with open(f'{base}/user_artists.dat') as f:
    next(f)
    for line in f:
        u,a,w=line.split('\t')[:3]; ua.append((int(u),int(a),float(w)))
arts,cnt={}, {}
for u,a,w in ua: cnt[a]=cnt.get(a,0)+1
top_art=[a for a,_ in sorted(cnt.items(),key=lambda x:-x[1])[:NITEM]]; aid={a:k for k,a in enumerate(top_art)}; ni=len(aid)
# ---- load tags per artist (attributes) ----
at_count={}                                   # (artist,tag)->#distinct users
seen=set()
with open(f'{base}/user_taggedartists.dat') as f:
    next(f)
    for line in f:
        p=line.split('\t'); u,a,t=int(p[0]),int(p[1]),int(p[2])
        if a in aid and (u,a,t) not in seen:
            seen.add((u,a,t)); at_count[(a,t)]=at_count.get((a,t),0)+1
tagfreq={}
for (a,t),c in at_count.items():
    if c>=TAGMIN: tagfreq[t]=tagfreq.get(t,0)+1
top_tags=[t for t,_ in sorted(tagfreq.items(),key=lambda x:-x[1])[:NATTR]]; tid={t:k for k,t in enumerate(top_tags)}; na=len(tid)
itemattr=[[] for _ in range(ni)]              # item -> list of attribute idx
for (a,t),c in at_count.items():
    if c>=TAGMIN and t in tid: itemattr[aid[a]].append(tid[t])
# ---- user->items (implicit, top artists only) ----
ui={}
for u,a,w in ua:
    if a in aid: ui.setdefault(u,set()).add(aid[a])
users=[u for u in ui if len(ui[u])>=MINU]; rng.shuffle(users)
ntr=int(0.8*len(users)); tr_u=users[:ntr]; te_u=users[ntr:]
pop=np.zeros(ni)
for u in tr_u:
    for v in ui[u]: pop[v]+=1
print(f"LastFM: {ni} artists, {na} attrs, {len(users)} users (train {len(tr_u)} test {len(te_u)}); avg attrs/item {np.mean([len(x) for x in itemattr]):.1f}",flush=True)

class FM(nn.Module):
    def __init__(s): super().__init__(); s.U=nn.Embedding(max(tr_u)+1,D); s.V=nn.Parameter(0.01*torch.randn(ni,D)); s.A=nn.Parameter(0.01*torch.randn(na,D)); s.b=nn.Parameter(torch.zeros(ni))
    def uvec(s,uid,attrs):                     # user repr = id embedding + sum of confirmed-attribute vectors (EAR)
        e=s.U(torch.tensor(uid))
        if attrs: e=e+s.A[torch.tensor(attrs)].sum(0)
        return e
m=FM(); opt=torch.optim.Adam(m.parameters(),5e-3,weight_decay=1e-6); t0=time.time()
trl=[(u,list(ui[u])) for u in tr_u]
for ep in range(EP):
    rng.shuffle(trl); tot=0.
    for u,items in trl:
        attrs=list({a for v in items for a in itemattr[v]})        # confirmed attributes = attrs of liked items (EAR)
        ue=m.uvec(u,attrs)                                          # u + Sum p
        pos=torch.tensor(items); neg=torch.tensor(rng.integers(0,ni,len(items)))
        sp=(ue*m.V[pos]).sum(1)+m.b[pos]; sn=(ue*m.V[neg]).sum(1)+m.b[neg]
        loss=-F.logsigmoid(sp-sn).mean(); opt.zero_grad(); loss.backward(); opt.step(); tot+=loss.item()
    if (ep+1)%5==0: print(f"  ep{ep+1} bpr={tot/len(trl):.4f} ({time.time()-t0:.0f}s)",flush=True)
m.eval()
# ---- cold-start eval: reveal half of user's artists -> confirmed attributes -> rank held-out half ----
def ndcg_recall(score, rel, excl):
    s=score.copy(); s[list(excl)]=-1e9; top=np.argsort(-s)[:10]; rs=set(rel)
    dcg=sum(1./np.log2(p+2) for p,t in enumerate(top) if t in rs); idcg=sum(1./np.log2(p+2) for p in range(min(10,len(rel))))
    return (dcg/idcg if idcg>0 else 0.), len(set(top.tolist())&rs)/len(rel)
lpop=np.log(pop+1.0)
with torch.no_grad(): Vd=m.V.detach().numpy(); Ad=m.A.detach().numpy(); bd=m.b.detach().numpy()
def zsc(v): return (v-v.mean())/(v.std()+1e-9)
res={'POPULARITY':[],'ATTR-only (Sum p.v)':[],'ATTR+pop (EAR mechanism)':[],'item-only (b_v)':[]}
W=float(os.environ.get('W',4.0))
for u in te_u:
    items=list(ui[u]); rng.shuffle(items); rev=set(items[:len(items)//2]); rel=[v for v in items[len(items)//2:]]
    if len(rev)<3 or not rel: continue
    attrs=list({a for v in rev for a in itemattr[v]})              # confirmed attributes from revealed half
    uvec=Ad[attrs].sum(0) if attrs else np.zeros(D)
    sattr=Vd@uvec                                                  # Sum_{p in P_u} v.p
    res['POPULARITY'].append(ndcg_recall(lpop,rel,rev))
    res['item-only (b_v)'].append(ndcg_recall(bd,rel,rev))
    res['ATTR-only (Sum p.v)'].append(ndcg_recall(sattr,rel,rev))
    res['ATTR+pop (EAR mechanism)'].append(ndcg_recall(zsc(sattr)+W*zsc(lpop),rel,rev))
print("\n=== EAR mechanism on LastFM (cold-start, std NDCG@10/Recall@10, full-cat) ===",flush=True)
base_nd=np.mean([x[0] for x in res['POPULARITY']])
for k,v in res.items():
    nd=np.mean([x[0] for x in v]); rc=np.mean([x[1] for x in v])
    print(f"  {k:<26} NDCG@10={nd:.4f} Recall@10={rc:.4f}  {'WIN vs pop' if nd>base_nd+1e-4 else ''}",flush=True)
print(f"\nKEY CLAIM: confirmed-attribute conditioning {'IMPROVES' if np.mean([x[0] for x in res['ATTR+pop (EAR mechanism)']])>base_nd else 'does NOT improve'} cold-start rec vs popularity.",flush=True)
