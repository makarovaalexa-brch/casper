"""
LLM-SIMULATED-USER STUDY — Phase 1  (Paper C/D reality check on the simulator's answer model)

Reuses the canonical Paper-B/C harness data loading (continuous_actor.py, lines 8-56): ML-1M ratings,
SVD item factors Q, frozen attention fold-in recommender (enc_concept.pt), concept genome tags, te[300:] test split.

Three phases (each independently runnable):
  (a) FRAMING LEVER  -- does 'underrated gem' framing move the named-item popularity DOWN? (both context conditions)
  (b) PAIR AGREEMENT -- does an LLM user agree with the simulator's geometric answer sign(u*.(e_i-e_j))?
  (c) ANSWERABILITY  -- does the >=2-tagged-items answerability proxy predict LLM refusal?

All OpenAI responses cached to a jsonl keyed by (model,effort,system,user) hash -> reruns are FREE.
Default answering model = gpt-4o-mini. Cross-model subsample = gpt-5-nano (reasoning_effort='minimal').

Usage (foreground):
  python scripts/paper2/llm_user_study.py --phase a --n 200 --repeats 3 --nano-sub 30
  python scripts/paper2/llm_user_study.py --phase b --n 100
  python scripts/paper2/llm_user_study.py --phase c --n 100
  python scripts/paper2/llm_user_study.py --phase all
"""
import os, sys, re, json, time, hashlib, argparse, threading, csv
import numpy as np, torch, torch.nn as nn
from concurrent.futures import ThreadPoolExecutor, as_completed

BASE='C:/dev/phd/casper/data/movielens'; ML=f'{BASE}/ml-1m'; D=64; T=8
CACHE_DIR='C:/dev/phd/casper/experiments/paper2'
CACHE_JSONL=f'{CACHE_DIR}/llmuser_cache.jsonl'

# ------------------------------------------------------------------ OpenAI plumbing
from dotenv import load_dotenv
load_dotenv('C:/dev/phd/casper/.env')
if not os.environ.get('OPENAI_API_KEY'):
    print("FATAL: OPENAI_API_KEY missing from environment/.env"); sys.exit(2)
from openai import OpenAI
_client=OpenAI()
_PRICE={  # USD per 1M tokens (in,out) — for spend estimate only
    'gpt-4o-mini':(0.15,0.60), 'gpt-5-nano':(0.05,0.40), 'gpt-4o':(2.50,10.0)}
_spend={'in':{}, 'out':{}, 'calls':{}}
_spend_lock=threading.Lock()

# cache: key -> {content, pt, ct}
_cache={}
_cache_lock=threading.Lock()
def _load_cache():
    if os.path.exists(CACHE_JSONL):
        for line in open(CACHE_JSONL,encoding='utf-8'):
            line=line.strip()
            if not line: continue
            try: r=json.loads(line); _cache[r['key']]=r
            except Exception: pass
    print(f"  loaded {len(_cache)} cached LLM responses from {os.path.basename(CACHE_JSONL)}",flush=True)
def _key(model,effort,system,user):
    h=hashlib.sha256(f"{model}|{effort}|{system}|{user}".encode('utf-8')).hexdigest()
    return h
def llm_call(model,system,user,effort=None,max_tok=64):
    """Return assistant text. Cached. gpt-5-nano uses max_completion_tokens + reasoning_effort='minimal'."""
    eff = effort if effort is not None else ('minimal' if 'nano' in model else '')
    k=_key(model,eff,system,user)
    with _cache_lock:
        if k in _cache: return _cache[k]['content']
    msgs=[{'role':'system','content':system},{'role':'user','content':user}] if system else [{'role':'user','content':user}]
    for attempt in range(4):
        try:
            kw={'model':model,'messages':msgs}
            if 'nano' in model or model.startswith('gpt-5'):
                kw['max_completion_tokens']=max(max_tok,32); kw['reasoning_effort']=eff or 'minimal'
            else:
                kw['max_tokens']=max_tok; kw['temperature']=0.7
            r=_client.chat.completions.create(**kw)
            content=(r.choices[0].message.content or '').strip()
            pt=r.usage.prompt_tokens; ct=r.usage.completion_tokens
            with _spend_lock:
                _spend['in'][model]=_spend['in'].get(model,0)+pt
                _spend['out'][model]=_spend['out'].get(model,0)+ct
                _spend['calls'][model]=_spend['calls'].get(model,0)+1
            rec={'key':k,'model':model,'content':content,'pt':pt,'ct':ct}
            with _cache_lock:
                _cache[k]=rec
                with open(CACHE_JSONL,'a',encoding='utf-8') as f: f.write(json.dumps(rec)+'\n')
            return content
        except Exception as e:
            if attempt==3:
                print(f"  API ERROR (giving up): {str(e)[:160]}",flush=True); return ''
            time.sleep(2**attempt)
def run_batch(reqs,workers=8):
    """reqs = list of dicts each with keys model,system,user,(effort),(max_tok). Fills 'content' in place. Parallel over uncached."""
    todo=[r for r in reqs if _key(r['model'],(r.get('effort') if r.get('effort') is not None else ('minimal' if 'nano' in r['model'] else '')),r['system'],r['user']) not in _cache]
    print(f"  {len(reqs)} requests, {len(reqs)-len(todo)} cached, {len(todo)} to fetch",flush=True)
    def _do(r):
        r['content']=llm_call(r['model'],r['system'],r['user'],r.get('effort'),r.get('max_tok',64)); return r
    done=0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs=[ex.submit(_do,r) for r in todo]
        for _ in as_completed(futs):
            done+=1
            if done%200==0: print(f"    fetched {done}/{len(todo)}",flush=True)
    # fill cached ones
    for r in reqs:
        if 'content' not in r: r['content']=llm_call(r['model'],r['system'],r['user'],r.get('effort'),r.get('max_tok',64))
    return reqs

# ------------------------------------------------------------------ DATA (mirrors continuous_actor.py 8-56)
print("Loading ML-1M harness data...",flush=True)
U,I,Rr=[],[],[]
with open(f'{ML}/ratings.dat') as f:
    for line in f:
        a=line.strip().split('::'); U.append(int(a[0])); I.append(int(a[1])); Rr.append(float(a[2]))
U=np.array(U);I=np.array(I);Rr=np.array(Rr,np.float32)
uids={x:k for k,x in enumerate(np.unique(U))}; iids={x:k for k,x in enumerate(np.unique(I))}; nu,ni=len(uids),len(iids)
inv_iids={k:x for x,k in iids.items()}
uu=np.array([uids[x] for x in U]); ii=np.array([iids[x] for x in I])
rat_by_u={}
for k in range(len(uu)): rat_by_u.setdefault(uu[k],[]).append((ii[k],float(Rr[k])))
likes_by_u={x:[j for j,r in v if r>=4] for x,v in rat_by_u.items()}
keep=[x for x in range(nu) if len(likes_by_u.get(x,[]))>=5]; np.random.default_rng(0).shuffle(keep)  # canonical split = rng(0)
nK=len(keep); trU=keep[:int(0.8*nK)]; te=keep[int(0.9*nK):]
cnt=np.zeros(ni)
for x in trU:
    for j in likes_by_u.get(x,[]): cnt[j]+=1
popb=np.log(cnt+1.0).astype(np.float32); sm=c=0.
for x in trU:
    for j,r in rat_by_u[x]: sm+=r;c+=1
mu=sm/c
Q=np.load(f'{BASE}/.cache/Q_svd.npy'); bi=np.load(f'{BASE}/.cache/bi_svd.npy')
Ql=np.load(f'{BASE}/.cache/Ql_concept.npy'); Ec=np.load(f'{BASE}/.cache/Ec_concept.npy')
ctags=list(np.load(f'{BASE}/.cache/ctags_concept.npy'))
order_pop=np.argsort(-cnt); cum=np.cumsum(cnt[order_pop])/cnt.sum()
HEAD=set(int(j) for j in order_pop[:np.searchsorted(cum,0.33)+1]); headmask=np.zeros(ni,bool); headmask[list(HEAD)]=True
resid={x:{j:(r-mu-bi[j]) for j,r in rat_by_u[x]} for x in (trU+te)}
POS=np.mean([resid[x][j] for x in trU[:3000] for j,r in rat_by_u[x] if r>=4])
NEG=np.mean([resid[x][j] for x in trU[:3000] for j,r in rat_by_u[x] if r<4])

# popularity by RATING COUNT (full data) + percentile among catalog
ratecount=np.zeros(ni)
for k in range(len(ii)): ratecount[ii[k]]+=1
_order=np.argsort(ratecount)  # ascending; percentile = rank/(N-1)
pct_of_idx=np.zeros(ni)
_rank=np.empty(ni,int); _rank[_order]=np.arange(ni)
pct_of_idx=(_rank/(ni-1)*100.0)

# titles
title_of={}  # internal idx -> display title
with open(f'{ML}/movies.dat',encoding='latin-1') as f:
    for line in f:
        a=line.rstrip('\n').split('::')
        if len(a)<2: continue
        mid=int(a[0])
        if mid in iids: title_of[iids[mid]]=a[1]

def norm_title(t):
    t=re.sub(r'\(\d{4}\)','',t)               # drop year
    t=t.strip().lower().strip('"\'.')
    for art in ['the','a','an','le','la','les','il',"l'"]:
        if t.endswith(', '+art): t=art+' '+t[:-(len(art)+2)]; break
    t=re.sub(r'^(the|a|an) ','',t)            # drop leading article for matching
    t=re.sub(r'[^a-z0-9 ]','',t); t=re.sub(r'\s+',' ',t).strip()
    return t
# normalized catalog -> idx (collision: keep more-rated)
norm2idx={}
for idx,disp in title_of.items():
    nt=norm_title(disp)
    if not nt: continue
    if nt not in norm2idx or ratecount[idx]>ratecount[norm2idx[nt]]: norm2idx[nt]=idx
_norm_keys=list(norm2idx.keys())
import difflib
def match_title(raw):
    """LLM raw title -> (idx or None, matched_display or None, kind)."""
    nt=norm_title(raw)
    if not nt: return None,None,'empty'
    if nt in norm2idx:
        i=norm2idx[nt]; return i,title_of[i],'exact'
    cm=difflib.get_close_matches(nt,_norm_keys,n=1,cutoff=0.86)
    if cm:
        i=norm2idx[cm[0]]; return i,title_of[i],'fuzzy'
    return None,None,'nomatch'

# concept names (genome tags)
tagname={}
for r in csv.reader(open(f'{BASE}/genome-tags.csv',encoding='utf-8')):
    if r and r[0].isdigit(): tagname[int(r[0])]=r[1]
concept_name=[tagname.get(int(t),f'tag{t}') for t in ctags]
NC=len(ctags)
# concept -> set of items strongly tagged (>0.5), for the answerability proxy
tagset=set(int(t) for t in ctags); tagitems={}
with open(f'{BASE}/genome-scores.csv') as f:
    next(f)
    for line in f:
        a=line.split(','); m=int(a[0]); tg=int(a[1])
        if m in iids and tg in tagset and float(a[2])>0.5: tagitems.setdefault(tg,[]).append(iids[m])
citems=[set(tagitems.get(int(t),[])) for t in ctags]

# frozen recommender encoder
class Enc(nn.Module):
    def __init__(s):
        super().__init__(); s.inp=nn.Sequential(nn.Linear(D+1,128),nn.ReLU(),nn.Linear(128,128),nn.ReLU()); s.att=nn.Linear(128,1); s.val=nn.Linear(128,D)
    def forward(s,t,m):
        h=s.inp(t); a=s.att(h).squeeze(-1).masked_fill(m==0,-1e9); al=torch.softmax(a,1); return (al.unsqueeze(-1)*s.val(h)).sum(1)
enc=Enc(); enc.load_state_dict(torch.load(f'{BASE}/.cache/enc_concept.pt')); enc.eval()
for p in enc.parameters(): p.requires_grad_(False)
def enc_u_np(toks):
    if not toks: return np.zeros(D)
    arr=np.zeros((1,len(toks),D+1),np.float32); m=np.ones((1,len(toks)),np.float32)
    for q,(f,v) in enumerate(toks): arr[0,q,:D]=f; arr[0,q,D]=v
    with torch.no_grad(): return enc(torch.tensor(arr),torch.tensor(m)).numpy()[0]

# continuous actor (D+1 -> 128 -> 128 -> 64)
class Actor(nn.Module):
    def __init__(s): super().__init__(); s.f=nn.Sequential(nn.Linear(D+1,128),nn.ReLU(),nn.Linear(128,128),nn.ReLU(),nn.Linear(128,D))
    def forward(s,u,tt):
        B=u.shape[0]; tn=torch.full((B,1),float(tt)); return s.f(torch.cat([u,tn],1))
_CN=float(np.linalg.norm(Ec,axis=1).mean())

# faithful PAIRSNAP pair realization (mirrors pair_topk / pair_realize_t)
def pair_topk(qn,K=64):
    s=Q@qn
    ti=np.argpartition(-s,K)[:K]; ti=ti[np.argsort(-s[ti])]
    bj=np.argpartition(s,K)[:K]; bj=bj[np.argsort(s[bj])]
    ei=Q[ti]; ej=Q[bj]
    d=ei[:,None,:]-ej[None,:,:]; dn=np.linalg.norm(d,axis=2)
    cc=(s[ti][:,None]-s[bj][None,:])/(dn+1e-12); cc[dn<1e-5]=-2.
    k=int(np.argmax(cc)); a,b=k//K,k%K
    return int(ti[a]),int(bj[b])

PAIR_CK=f'{BASE}/.cache/policy_pairtrain_d1warm_best.pt'
ACTOR_PAIR=Actor()
_sd=torch.load(PAIR_CK); ACTOR_PAIR.load_state_dict(_sd['actor'] if isinstance(_sd,dict) and 'actor' in _sd else _sd); ACTOR_PAIR.eval()
print(f"  loaded pair-native actor {os.path.basename(PAIR_CK)}",flush=True)

print(f"  data ready: ni={ni} items, {len(te)} test users (te[300:]={len(te)-300}), NC={NC} concepts",flush=True)
TEST=te[300:]

# ------------------------------------------------------------------ profile text builders
def profile_items(x,cap=80):
    """Return list of (idx,rating) capped to <=cap, keeping high & low extremes (deterministic)."""
    rated=[(j,r) for j,r in rat_by_u[x] if j in title_of]
    rated=sorted(rated,key=lambda z:-z[1])
    if len(rated)<=cap: return rated
    hi=rated[:cap*5//8]; lo=rated[-(cap-len(hi)):]
    return hi+lo
def full_context_text(x):
    lines=[f"{title_of[j]}: {int(round(r))}/5" for j,r in profile_items(x)]
    return "This person's movie ratings (title: rating out of 5):\n"+"\n".join(lines)
def persona_text(x,model='gpt-4o-mini'):
    """3-sentence taste summary, generated ONCE per user (cached)."""
    prof=full_context_text(x)
    sysp="You summarize a movie viewer's taste."
    usr=(prof+"\n\nWrite exactly 3 sentences, in second person ('You love...'), summarizing this person's movie "
         "taste: the genres, tones, and eras they gravitate to and avoid. Do not list specific titles.")
    return llm_call(model,sysp,usr,max_tok=180)

def answer_system(x,cond,persona_cache):
    if cond=='FULL':
        body=full_context_text(x)
    else:
        body=persona_cache[x]
    return (f"You are a movie viewer with these tastes.\n{body}\n\n"
            "Answer as this person would, briefly and in character.")

# ================================================================== PHASE A: FRAMING LEVER
QTYPES={
 'love':"Name a movie you love. Reply with ONLY the movie title, nothing else.",
 'gem' :"Name an underrated film or hidden gem you love. Reply with ONLY the movie title, nothing else.",
 'dislike':"Name a movie you dislike. Reply with ONLY the movie title, nothing else."}

def phase_a(users,repeats,answer_model,nano_sub,persona_cache):
    print(f"\n=== PHASE A: FRAMING LEVER | {len(users)} users x 2 conds x 3 types x {repeats} reps | model={answer_model} ===",flush=True)
    conds=['FULL','PERSONA']
    reqs=[]
    for cond in conds:
        for x in users:
            sysp=answer_system(x,cond,persona_cache)
            for qt,qtext in QTYPES.items():
                for rep in range(repeats):
                    reqs.append({'model':answer_model,'system':sysp+f"\n[rep{rep}]",'user':qtext,'max_tok':40,
                                 'meta':('main',cond,x,qt,rep)})
    # cross-model subsample on gpt-5-nano
    nano_users=list(users)[:nano_sub]
    for cond in conds:
        for x in nano_users:
            sysp=answer_system(x,cond,persona_cache)
            for qt,qtext in QTYPES.items():
                for rep in range(repeats):
                    reqs.append({'model':'gpt-5-nano','system':sysp+f"\n[rep{rep}]",'user':qtext,'max_tok':40,'effort':'minimal',
                                 'meta':('nano',cond,x,qt,rep)})
    run_batch(reqs)
    # aggregate
    res={}  # (tag,cond,qt) -> list of (user, pct, matched, kind)
    for r in reqs:
        tag,cond,x,qt,rep=r['meta']; idx,disp,kind=match_title(r['content'])
        res.setdefault((tag,cond,qt),[]).append((x,None if idx is None else float(pct_of_idx[idx]),kind,r['content']))
    return res

# ================================================================== PHASE B: PAIR AGREEMENT
def realized_pairs(x,nturn=8):
    """Run the pair-native actor for nturn turns on user x (full-profile fold as u*); return list of (i,j,gscore)."""
    prof=[j for j,_ in rat_by_u[x]]
    ustar=enc_u_np([(Q[j],resid[x][j]) for j in prof])
    toks=[]; pairs=[]; seen=set()
    for t in range(nturn):
        u=enc_u_np(toks)
        with torch.no_grad(): qv=ACTOR_PAIR(torch.tensor(u[None],dtype=torch.float32),t/8.).numpy()[0]
        qn=(qv/(np.linalg.norm(qv)+1e-9)).astype(np.float32)
        i,j=pair_topk(qn)
        d=Q[i]-Q[j]; dn=d/(np.linalg.norm(d)+1e-9); fe=(dn*_CN).astype(np.float32)
        cf=float(ustar@fe)/(np.linalg.norm(fe)*np.linalg.norm(ustar)+1e-9)
        ans=float(NEG+(POS-NEG)*(cf+1)/2)
        toks.append((fe,ans))
        gscore=float(ustar@(Q[i]-Q[j]))       # >0 -> i preferred by geometry
        if (i,j) not in seen and i in title_of and j in title_of:
            pairs.append((i,j,gscore)); seen.add((i,j))
    return pairs,ustar

def phase_b(users,answer_model,persona_cache):
    print(f"\n=== PHASE B: PAIR AGREEMENT | {len(users)} users x 8 pairs | FULL context | model={answer_model} ===",flush=True)
    reqs=[]; upairs={}
    for x in users:
        pairs,_=realized_pairs(x,8); upairs[x]=pairs
        sysp=answer_system(x,'FULL',persona_cache)
        for pi,(i,j,g) in enumerate(pairs):
            # randomize presentation order to avoid position bias (deterministic per pair)
            flip=((i+j)%2==0)
            A,B=(title_of[j],title_of[i]) if flip else (title_of[i],title_of[j])
            usr=(f"Which of these two movies do you prefer: \"{A}\" or \"{B}\"? "
                 "Answer with just the title of the one you prefer.")
            reqs.append({'model':answer_model,'system':sysp,'user':usr,'max_tok':40,'meta':(x,pi,i,j,g,flip)})
    run_batch(reqs)
    rows=[]  # (x,i,j,gscore,llm_pref_i(bool or None),agree(bool or None))
    for r in reqs:
        x,pi,i,j,g,flip=r['meta']; mi,md,kind=match_title(r['content'])
        # decide which item the LLM picked
        pref=None
        if mi==i: pref=True
        elif mi==j: pref=False
        else:
            # fall back to substring against the two displayed titles
            c=norm_title(r['content'])
            ni_,nj_=norm_title(title_of[i]),norm_title(title_of[j])
            if c and c in ni_: pref=True
            elif c and c in nj_: pref=False
            elif ni_ in c: pref=True
            elif nj_ in c: pref=False
        geo_i=(g>0)  # geometry prefers i
        agree=None if pref is None else (pref==geo_i)
        rows.append((x,i,j,g,pref,agree,abs(g)))
    return rows

# ================================================================== PHASE C: ANSWERABILITY
def pick_concepts(x,n_ans=5,n_non=5):
    prof=set(j for j,_ in rat_by_u[x])
    ov=np.array([len(citems[c]&prof) for c in range(NC)])
    ans=[c for c in range(NC) if ov[c]>=2]
    non=[c for c in range(NC) if ov[c]==0]
    rng=np.random.default_rng(x)
    ans=sorted(ans,key=lambda c:-ov[c])[:max(n_ans*3,15)]; rng.shuffle(ans)
    rng.shuffle(non)
    return ans[:n_ans],non[:n_non]

def phase_c(users,answer_model,persona_cache):
    print(f"\n=== PHASE C: ANSWERABILITY REALISM | {len(users)} users x 10 concepts | FULL context | model={answer_model} ===",flush=True)
    reqs=[]
    for x in users:
        ans,non=pick_concepts(x)
        sysp=answer_system(x,'FULL',persona_cache)
        for c in ans+non:
            proxy_answerable=(c in ans)
            usr=(f"Do you like movies described as \"{concept_name[c]}\"? "
                 "Answer with exactly one of: 'yes', 'no', or 'not familiar enough to say'.")
            reqs.append({'model':answer_model,'system':sysp,'user':usr,'max_tok':24,'meta':(x,c,proxy_answerable)})
    run_batch(reqs)
    rows=[]  # (x,c,proxy_answerable,llm_label)
    for r in reqs:
        x,c,pa=r['meta']; t=r['content'].strip().lower()
        if 'not familiar' in t or 'not sure' in t or "can't say" in t or 'unfamiliar' in t: lab='refuse'
        elif t.startswith('yes') or t=='yes' or ' yes' in t[:6]: lab='yes'
        elif t.startswith('no') or t=='no': lab='no'
        else: lab='other'
        rows.append((x,c,pa,lab))
    return rows

# ------------------------------------------------------------------ reporting helpers
def wilcoxon(d):
    """Paired Wilcoxon signed-rank (two-sided normal approx). d = array of differences."""
    d=np.asarray([v for v in d if v!=0],float)
    n=len(d)
    if n<10: return float('nan'),n
    r=np.argsort(np.argsort(np.abs(d)))+1.0
    W=np.sum(r[d>0]); mu=n*(n+1)/4.; sig=np.sqrt(n*(n+1)*(2*n+1)/24.)
    z=(W-mu)/sig; from math import erf,sqrt
    p=2*(1-0.5*(1+erf(abs(z)/sqrt(2))))
    return p,n

def spend_report():
    tot=0.; lines=[]
    for m in _spend['calls']:
        pin,pout=_PRICE.get(m,(0,0)); ci=_spend['in'][m]/1e6*pin; co=_spend['out'][m]/1e6*pout
        tot+=ci+co
        lines.append(f"  {m}: {_spend['calls'][m]} calls, {_spend['in'][m]} in / {_spend['out'][m]} out tok -> ${ci+co:.4f}")
    return tot,lines

# ------------------------------------------------------------------ main
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--phase',default='all',choices=['a','b','c','all'])
    ap.add_argument('--n',type=int,default=200)
    ap.add_argument('--nb',type=int,default=100)
    ap.add_argument('--nc',type=int,default=100)
    ap.add_argument('--repeats',type=int,default=3)
    ap.add_argument('--nano-sub',type=int,default=30)
    ap.add_argument('--model',default='gpt-4o-mini')
    ap.add_argument('--workers',type=int,default=8)
    args=ap.parse_args()
    os.makedirs(CACHE_DIR,exist_ok=True); _load_cache()

    users=list(TEST)[:args.n]
    # personas: ONCE per user, cheap model, cached
    print(f"Building/caching {len(users)} personas ({args.model})...",flush=True)
    persona_cache={}
    def _pers(x): return x,persona_text(x,args.model)
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for x,pt in ex.map(_pers,users): persona_cache[x]=pt

    out={'config':{'phase':args.phase,'n':args.n,'repeats':args.repeats,'nano_sub':args.nano_sub,'model':args.model,
                   'test_users':len(TEST),'ni':ni,'NC':NC}}
    if args.phase in ('a','all'):
        out['A']=phase_a(users,args.repeats,args.model,args.nano_sub,persona_cache)
    if args.phase in ('b','all'):
        out['B']=phase_b(list(TEST)[:args.nb],args.model,persona_cache)
    if args.phase in ('c','all'):
        out['C']=phase_c(list(TEST)[:args.nc],args.model,persona_cache)
    tot,lines=spend_report()
    out['spend']=(tot,lines)
    import pickle
    pickle.dump(out,open(f'{CACHE_DIR}/llmuser_phase1_raw.pkl','wb'))
    print("\n=== SPEND ==="); [print(l) for l in lines]; print(f"  TOTAL ~${tot:.4f}")
    print(f"\nRaw results pickled -> {CACHE_DIR}/llmuser_phase1_raw.pkl")
    write_report(out,args)

def _pctstats(rows):
    """rows = list of (user,pct,kind,raw). Return dict with match rate + pct stats over matched."""
    n=len(rows); matched=[r for r in rows if r[1] is not None]
    pcts=np.array([r[1] for r in matched],float)
    d={'n':n,'matched':len(matched),'match_rate':len(matched)/max(n,1)}
    if len(pcts): d.update({'mean':pcts.mean(),'median':np.median(pcts),'p25':np.percentile(pcts,25),'p75':np.percentile(pcts,75)})
    else: d.update({'mean':float('nan'),'median':float('nan'),'p25':float('nan'),'p75':float('nan')})
    return d,matched

def write_report(out,args):
    L=[]; A=out.get('A'); B=out.get('B'); C=out.get('C')
    L.append("# LLM-Simulated-User Study — Phase 1 (Paper C/D simulator reality check)")
    L.append("")
    L.append("Tests whether the CASPER answer simulator's assumptions hold up against an LLM standing in for a real")
    L.append("user: (a) does 'hidden gem' framing pull named items toward the tail, (b) does an LLM user agree with the")
    L.append("geometric answer sign(u*.(e_i-e_j)), (c) does the >=2-tagged-items answerability proxy predict LLM refusal.")
    L.append("")
    L.append("## Config")
    L.append(f"- Data: ML-1M, canonical harness split (continuous_actor.py); test pool = te[300:] = {len(TEST)} users; catalog ni={ni}; concepts NC={NC}")
    L.append(f"- Answering model (default): **{args.model}** (temperature 0.7)")
    L.append(f"- Cross-model subsample: **gpt-5-nano** (reasoning_effort='minimal', max_completion_tokens) on {args.nano_sub} users")
    L.append(f"- Persona: 3-sentence taste summary generated ONCE/user by {args.model} (cached)")
    L.append(f"- FULL context = user's rated titles+ratings capped to <=80 items (high+low extremes)")
    L.append(f"- Recommender fold u* = frozen enc_concept.pt attention encoder; item factors = Q_svd (D={D})")
    L.append(f"- Pair questions: pair-native actor {os.path.basename(PAIR_CK)}, 8 realized item-pairs/user (PAIRSNAP top-K realization)")
    L.append(f"- All API responses cached to {os.path.basename(CACHE_JSONL)} (reruns free)")
    L.append("")
    # ---- Phase A
    if A:
        L.append("## (a) Framing lever — popularity percentile of the named movie (0=obscure tail, 100=most-rated head)")
        L.append("")
        for tag,label in [('main',args.model),('nano','gpt-5-nano')]:
            present=any(k[0]==tag for k in A)
            if not present: continue
            L.append(f"### {label}")
            L.append("| condition | question | n | match% | mean pct | median pct | p25 | p75 |")
            L.append("|---|---|---:|---:|---:|---:|---:|---:|")
            store={}
            for cond in ['FULL','PERSONA']:
                for qt in ['love','gem','dislike']:
                    rows=A.get((tag,cond,qt),[])
                    if not rows: continue
                    d,matched=_pctstats(rows); store[(cond,qt)]=(d,matched)
                    L.append(f"| {cond} | {qt} | {d['n']} | {d['match_rate']*100:.0f}% | {d['mean']:.1f} | {d['median']:.1f} | {d['p25']:.1f} | {d['p75']:.1f} |")
            L.append("")
            # paired love vs gem per user (mean pct per user per type)
            for cond in ['FULL','PERSONA']:
                if (cond,'love') in store and (cond,'gem') in store:
                    lm=_peruser(store[(cond,'love')][1]); gm=_peruser(store[(cond,'gem')][1])
                    common=[u for u in lm if u in gm]
                    diff=[gm[u]-lm[u] for u in common]  # gem - love ; expect NEGATIVE if gem pushes to tail
                    if diff:
                        p,nn=wilcoxon(diff); md=np.mean(diff)
                        if not np.isfinite(p): tag_=f'(n too small, n={len(common)})'
                        elif md<0 and p<0.05: tag_='(gem significantly LOWER — toward tail, as the simulator assumes)'
                        elif md>0 and p<0.05: tag_='(gem significantly HIGHER — CONTRARY to the simulator)'
                        else: tag_='(n.s.)'
                        L.append(f"- **{cond}: gem − love** paired shift = {md:+.1f} pct-points (n={len(common)} users), "
                                 f"Wilcoxon p={p:.2g} {tag_}")
            L.append("")
    # ---- Phase B
    if B:
        rows=[r for r in B if r[5] is not None]
        undecided=len(B)-len(rows)
        agree=np.mean([r[5] for r in rows]) if rows else float('nan')
        L.append("## (b) Pair agreement — LLM choice vs geometric sign(u*·(e_i−e_j))")
        L.append("")
        L.append(f"- Overall agreement: **{agree*100:.1f}%** over {len(rows)} decided pairs "
                 f"({undecided} undecided/unparsed of {len(B)} total).")
        L.append("")
        L.append("Calibration by |u*·Δ| (geometric signal strength; stronger should agree more):")
        L.append("| |u*·Δ| quartile | n | agreement |")
        L.append("|---|---:|---:|")
        mags=np.array([r[6] for r in rows]); qs=np.quantile(mags,[0.25,0.5,0.75])
        for lo,hi,lab in [(-1,qs[0],'Q1 weakest'),(qs[0],qs[1],'Q2'),(qs[1],qs[2],'Q3'),(qs[2],1e9,'Q4 strongest')]:
            sel=[r for r in rows if lo<r[6]<=hi]
            if sel: L.append(f"| {lab} | {len(sel)} | {np.mean([r[5] for r in sel])*100:.1f}% |")
        L.append("")
    # ---- Phase C
    if C:
        L.append("## (c) Answerability realism — LLM refusal vs the >=2-tagged-items proxy")
        L.append("")
        # confusion: proxy answerable (yes/no) vs refuse
        conf={(True,'refuse'):0,(True,'answer'):0,(False,'refuse'):0,(False,'answer'):0}
        other=0
        for x,c,pa,lab in C:
            if lab=='other': other+=1; continue
            key='refuse' if lab=='refuse' else 'answer'
            conf[(pa,key)]+=1
        na=conf[(True,'refuse')]+conf[(True,'answer')]; nn=conf[(False,'refuse')]+conf[(False,'answer')]
        L.append("| proxy | LLM answered (yes/no) | LLM refused | refusal rate |")
        L.append("|---|---:|---:|---:|")
        L.append(f"| answerable (>=2 tagged) | {conf[(True,'answer')]} | {conf[(True,'refuse')]} | {conf[(True,'refuse')]/max(na,1)*100:.1f}% |")
        L.append(f"| NOT answerable (0 tagged) | {conf[(False,'answer')]} | {conf[(False,'refuse')]} | {conf[(False,'refuse')]/max(nn,1)*100:.1f}% |")
        L.append("")
        tot_agree=conf[(True,'answer')]+conf[(False,'refuse')]; tot=na+nn
        L.append(f"- Proxy↔LLM agreement (answerable→answer, non→refuse): **{tot_agree/max(tot,1)*100:.1f}%** over {tot} concepts ({other} unparsed 'other').")
        L.append("")
    # ---- caveats + spend
    L.append("## Caveats (honest)")
    L.append("- **Clairvoyant reader, not rememberer.** In FULL context the LLM can *read off* the answer from the rating")
    L.append("  list rather than recalling like a real user with imperfect memory. This INFLATES pair agreement and")
    L.append("  answerability realism upward; treat them as optimistic upper bounds. PERSONA (lossy summary) is the more")
    L.append("  realistic-recall condition and is why phase (a) runs both.")
    L.append(f"- **Fuzzy title match.** LLM free-text titles matched to ML-1M by normalized-string + difflib (cutoff 0.86).")
    L.append("  Unmatched names are dropped from percentile stats (match rates reported per cell). ML-1M's ', The'")
    L.append("  comma-title format and year suffixes are normalized; rare/foreign titles may still miss.")
    L.append("- **Popularity percentile = rating COUNT** over full ML-1M (not the harness's train-like count), ranked across the catalog.")
    L.append("- u* = full-profile fold (single per-user taste vector); pair questions fold the full profile too, so actor,")
    L.append("  geometric answer, and agreement ground-truth all share one u*. PAIRSNAP eval instead folds a half-profile.")
    L.append("- gpt-5-nano is a reasoning model; 'minimal' effort gives clean 0-reasoning-token answers but is a different")
    L.append("  model family than gpt-4o-mini — cross-model rows test robustness of the framing effect, not identical wording.")
    L.append("")
    tot,slines=out['spend']
    L.append("## Spend")
    for s in slines: L.append(f"-{s}")
    L.append(f"- **TOTAL ~${tot:.4f}** (well under the $5 cap)")
    open(f'{CACHE_DIR}/LLMUSER_PHASE1_RESULT.md','w',encoding='utf-8').write("\n".join(L))
    print(f"\nReport written -> {CACHE_DIR}/LLMUSER_PHASE1_RESULT.md")

def _peruser(matched):
    """matched rows (user,pct,kind,raw) -> {user: mean pct} over that user's matched answers."""
    agg={}
    for u,pct,kind,raw in matched: agg.setdefault(u,[]).append(pct)
    return {u:float(np.mean(v)) for u,v in agg.items()}

if __name__=='__main__':
    main()
