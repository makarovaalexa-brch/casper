"""
CHEAP LLM asker baseline on the calibrated instrument. The LLM sees the dialogue so far (liked/disliked) and a
NUMBERED menu of candidate movies; it picks the single most informative next movie to ask about. We fold the user's
true answer (leakage-free: taste residual if the item is in their profile, else "haven't seen" / no fold), then
score held-out NDCG. Responses are CACHED to jsonl (sha256 of model|prompt) so reruns are free. Parsing is
number-based (robust). Compared on the SAME users/menu against random-from-menu and an oracle-over-menu ceiling.

Cheap config: gpt-4o-mini, NU_LLM users, T questions, fixed menu. ~NU*T calls, cached.
"""
import os, json, hashlib, re, numpy as np, scipy.linalg as sla
from pathlib import Path
base='C:/dev/phd/casper/data/movielens'; ml=f'{base}/ml-1m'; D=64; LAM=5.0
T=int(os.environ.get('T',6)); NU_LLM=int(os.environ.get('NU_LLM',60)); MENU=int(os.environ.get('MENU',50))
MODEL=os.environ.get('LLM_MODEL','gpt-4o-mini'); CACHE='C:/dev/phd/casper/experiments/paper1/llm_asker_cache.jsonl'
rng=np.random.default_rng(0)
U,I,R=[],[],[]
with open(f'{ml}/ratings.dat') as f:
    for line in f:
        a=line.strip().split('::'); U.append(int(a[0])); I.append(int(a[1])); R.append(float(a[2]))
U=np.array(U);I=np.array(I);R=np.array(R,np.float32)
uids={x:k for k,x in enumerate(np.unique(U))}; iids={x:k for k,x in enumerate(np.unique(I))}; nu,ni=len(uids),len(iids)
uu=np.array([uids[x] for x in U]); ii=np.array([iids[x] for x in I])
rat_by_u={}
for k in range(len(uu)): rat_by_u.setdefault(uu[k],[]).append((ii[k],float(R[k])))
likes_by_u={x:[j for j,r in v if r>=4] for x,v in rat_by_u.items()}
keep=[x for x in range(nu) if len(likes_by_u.get(x,[]))>=5]; rng.shuffle(keep); n=len(keep)
trU=set(keep[:int(0.8*n)]); te=keep[int(0.9*n):]
cnt=np.zeros(ni)
for x in trU:
    for j in likes_by_u.get(x,[]): cnt[j]+=1
popb=np.log(cnt+1.0).astype(np.float32)
sm=c=0.
for x in trU:
    for j,r in rat_by_u[x]: sm+=r; c+=1
mu=sm/c
Q=np.load(f'{base}/.cache/Q_svd.npy'); bi=np.load(f'{base}/.cache/bi_svd.npy')
title={}
with open(f'{ml}/movies.dat',encoding='latin-1') as f:
    for line in f:
        pp=line.strip().split('::')
        if int(pp[0]) in iids: title[iids[int(pp[0])]]=pp[1]
# fixed menu = top-25 popular + 25 representative (RMVA), distinct
ca=np.array(np.argsort(-cnt)[:400]); _,_,piv=sla.qr(Q[ca].T,pivoting=True); RMVA=[int(ca[p]) for p in piv]
pop=list(np.argsort(-cnt)); seen=set(); menu=[]
for j in pop[:25]+RMVA:
    if j not in seen and j in title: seen.add(j); menu.append(j)
    if len(menu)>=MENU: break
menu_titles=[title[j] for j in menu]
SPL={}; _rs=np.random.default_rng(123)
for x in te:
    items=list(dict(rat_by_u[x]))
    if len(items)>=6: il=items[:]; _rs.shuffle(il); SPL[x]=(set(il[:len(il)//2]),set(il[len(il)//2:]))
def foldin(F,y): F=np.array(F); return np.linalg.solve(F.T@F+LAM*np.eye(D),F.T@np.array(y,np.float32)) if len(F) else np.zeros(D)
def ndcg(score,rel,excl):
    s=score.copy(); s[list(excl)]=-1e9; top=np.argsort(-s)[:10]; rs=set(rel)
    return sum(1./np.log2(p+2) for p,t in enumerate(top) if t in rs)/(sum(1./np.log2(p+2) for p in range(min(10,len(rel))))+1e-12)
# ---- cached LLM ----
_cache=None
def cache_get(k):
    global _cache
    if _cache is None:
        _cache={}
        if Path(CACHE).exists():
            for line in open(CACHE):
                try: r=json.loads(line); _cache[r['k']]=r['v']
                except: pass
    return _cache.get(k)
def cache_put(k,v):
    _cache[k]=v
    with open(CACHE,'a') as f: f.write(json.dumps({'k':k,'v':v})+'\n')
_client=None
def llm_call(msg):
    global _client
    if _client is None:
        from dotenv import load_dotenv; load_dotenv(Path('C:/dev/phd/casper/.env'))
        from openai import OpenAI; _client=OpenAI()
    r=_client.chat.completions.create(model=MODEL,messages=[
        {'role':'system','content':'You help a movie recommender choose the most informative next question for a NEW user. Reply with ONLY the menu number of the single best movie to ask about.'},
        {'role':'user','content':msg}],max_tokens=10,temperature=0.0)
    return r.choices[0].message.content.strip()
stats={'calls':0,'cached':0,'parsefail':0}
def llm_pick(history, asked):
    rem=[k for k in range(len(menu)) if k not in asked]
    if not rem: return None
    hist = "\n".join(f"- {menu_titles[k]}: {'liked' if v==1 else ('disliked' if v==0 else 'not seen')}" for k,v in history) or "(nothing yet)"
    mlist="\n".join(f"{k+1}. {menu_titles[k]}" for k in rem)
    msg=f"User's answers so far:\n{hist}\n\nMenu (pick one NUMBER to ask about next):\n{mlist}\n\nBest next number:"
    key=hashlib.sha256(f"{MODEL}|{msg}".encode()).hexdigest()
    txt=cache_get(key)
    if txt is None:
        try: txt=llm_call(msg); cache_put(key,txt); stats['calls']+=1
        except Exception as e: print("  API err:",str(e)[:80]); return int(rng.choice(rem))
    else: stats['cached']+=1
    mt=re.search(r'\d+', txt)
    if mt:
        k=int(mt.group())-1
        if 0<=k<len(menu) and k not in asked: return k
    stats['parsefail']+=1; return int(rng.choice(rem))
def answer(x, k, prof, rd):           # leakage-free: like/dislike if in profile, else "not seen"
    j=menu[k]
    if j in prof and j in rd: return (1 if rd[j]>=4 else 0), (rd[j]-mu-bi[j])
    return -1, None                   # not seen -> no fold
def run(kind):
    ND=np.zeros(T+1); m=0
    for x in te[:NU_LLM]:
        if x not in SPL: continue
        rd=dict(rat_by_u[x]); test,prof=SPL[x]; tlike=set(j for j in test if rd[j]>=4)
        if not tlike or not prof: continue
        nd=np.zeros(T+1); nd[0]=ndcg(popb.copy(),tlike,prof)
        F=[];y=[];asked=set();hist=[];excl=set(prof)
        for t in range(1,T+1):
            if kind=='llm': k=llm_pick(hist,asked)
            elif kind=='random': rem=[z for z in range(len(menu)) if z not in asked]; k=int(rng.choice(rem)) if rem else None
            else:  # oracle over menu
                best=None
                for z in range(len(menu)):
                    if z in asked: continue
                    av,tr=answer(x,z,prof,rd); Ft=F+([Q[menu[z]]] if tr is not None else []); yt=y+([tr] if tr is not None else [])
                    u=foldin(Ft,yt); a=ndcg(popb+Q@u,tlike,excl|{menu[z]})
                    if best is None or a>best[0]: best=(a,z)
                k=best[1] if best else None
            if k is None: nd[t:]=nd[t-1]; break
            asked.add(k); av,tr=answer(x,k,prof,rd); hist.append((k,av)); excl.add(menu[k])
            if tr is not None: F.append(Q[menu[k]]); y.append(tr)
            u=foldin(F,y); nd[t]=ndcg(popb+Q@u,tlike,excl)
        ND+=nd; m+=1
    return ND/m,m
print(f"LLM asker ({MODEL}); {min(NU_LLM,len(te))} users; T={T}; menu={len(menu)} items\n",flush=True)
for kind in ['random','llm','oracle']:
    nd,m=run(kind); print(f"{kind:<8} (n={m}): "+" ".join(f"{v:.3f}" for v in nd)+f"  | delta {nd[T]-nd[0]:+.3f}",flush=True)
print(f"\nLLM stats: {stats}",flush=True)
