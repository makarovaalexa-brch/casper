"""Paper E chatting UI: a deployable cold-start conversational recommender over the FROZEN Papers A-D pipeline.
Asks open free-recall questions (the Paper-D tree), grounds the user's free-text answers to catalogue factors, folds
into the belief, and renders live recommendations (titles + TMDB posters) that update each turn.
Run:  python scripts/paper5/casper_chat.py   then open http://127.0.0.1:5005
Self-contained: loads the frozen encoder + item factors + titles/credits/genres; no training."""
import os, json, csv, difflib, numpy as np, torch, torch.nn as nn, requests
from collections import defaultdict
from pathlib import Path
from flask import Flask, request, jsonify, render_template_string
try:
    from dotenv import load_dotenv; load_dotenv(Path(__file__).parent.parent.parent/'.env')
except Exception: pass

base='C:/dev/phd/casper/data/movielens'; ml=f'{base}/ml-1m'; D=64; TMDB=os.environ.get('TMDB_API_KEY')

# ---------- data ----------
U,I,R=[],[],[]
for line in open(f'{ml}/ratings.dat'):
    a=line.split('::'); U.append(int(a[0])); I.append(int(a[1])); R.append(float(a[2]))
I=np.array(I); R=np.array(R,np.float32)
iids={x:k for k,x in enumerate(np.unique(I))}; ni=len(iids); inv_iid={v:k for k,v in iids.items()}
cnt=np.zeros(ni)
for m in I: cnt[iids[m]]+=1
popb=np.log(cnt+1.).astype(np.float32); mu=float(R.mean())
order_pop=np.argsort(-cnt); cum=np.cumsum(cnt[order_pop])/cnt.sum(); headmask=np.zeros(ni,bool); headmask[order_pop[:np.searchsorted(cum,0.33)+1]]=True   # Cremonesi long-tail mask (head 33% by popularity)
Q=np.load(f'{base}/.cache/Q_svd.npy').astype(np.float32); bi=np.load(f'{base}/.cache/bi_svd.npy'); Ql=np.load(f'{base}/.cache/Ql_concept.npy').astype(np.float32)
POS=float(np.mean([R[k]-mu-bi[iids[I[k]]] for k in range(0,len(R),37) if R[k]>=4])); NEG=float(np.mean([R[k]-mu-bi[iids[I[k]]] for k in range(0,len(R),37) if R[k]<4]))
# ---- TASTE MAP: 2-D projection of the scoring space (items = Ql rows, belief = u) ----
from sklearn.decomposition import PCA
_pca=PCA(2,random_state=0).fit(Ql); QL2=_pca.transform(Ql).astype(np.float32); _mn=QL2.min(0); _rg=(QL2.max(0)-_mn)+1e-9
def to2d(v): p=(_pca.transform(np.asarray(v,np.float32)[None])[0]-_mn)/_rg; return [float(p[0]),float(p[1])]
def item2d(j): p=(QL2[j]-_mn)/_rg; return [float(p[0]),float(p[1])]
_bg=[[float((QL2[j,0]-_mn[0])/_rg[0]),float((QL2[j,1]-_mn[1])/_rg[1])] for j in np.argsort(-cnt)[:1400]]
# titles + genres (latin-1)
import codecs
title={}; genre_of=defaultdict(list); gfilms=defaultdict(list)
for line in codecs.open(f'{ml}/movies.dat','r','latin-1'):
    a=line.rstrip('\n').split('::')
    if len(a)>=3 and int(a[0]) in iids:
        j=iids[int(a[0])]; title[j]=a[1]
        for g in a[2].split('|'): genre_of[j].append(g); gfilms[g].append(j)
gcent={g:Q[np.array(sorted(set(v)))].mean(0) for g,v in gfilms.items()}
GENRES=sorted(gfilms)
import re
def normt(t):
    t=re.sub(r'\s*\(\d{4}\)\s*$','',t)                                          # strip trailing (year)
    mm=re.match(r'^(.*),\s+(The|A|An|La|Le|Les|Il|Der|Das)$',t)                 # "Matrix, The" -> "The Matrix"
    if mm: t=mm.group(2)+' '+mm.group(1)
    return ''.join(c for c in t.lower() if c.isalnum() or c==' ').split() and ' '.join(''.join(c for c in t.lower() if c.isalnum() or c==' ').split()) or ''
norm_title={ normt(t):j for j,t in title.items() }
# credits (actors/directors -> film centroids)
person_films=defaultdict(list)
try:
    cr=json.load(open(f'{base}/.cache/credits_ml1m_actors5.json'))
    for src in ('movie_actors','movie_directors'):
        for mid,ppl in cr[src].items():
            if int(mid) in iids:
                for p in ppl: person_films[p.lower()].append(iids[int(mid)])
except Exception: pass
pcent={p:Q[np.array(sorted(set(v)))].mean(0) for p,v in person_films.items() if v}
ALL_PEOPLE=sorted({p for p in person_films})
# tmdb links + poster cache
m2t={}
with open(f'{base}/links.csv') as f:
    next(f)
    for r in csv.reader(f):
        if len(r)>=3 and r[2]:
            try: m2t[int(r[0])]=int(r[2])
            except: pass
POSTER=Path(f'{base}/.cache/poster_cache.json'); poster_cache=json.load(open(POSTER)) if POSTER.exists() else {}

# ---------- frozen encoder ----------
class Enc(nn.Module):
    def __init__(s):
        super().__init__(); s.inp=nn.Sequential(nn.Linear(D+1,128),nn.ReLU(),nn.Linear(128,128),nn.ReLU()); s.att=nn.Linear(128,1); s.val=nn.Linear(128,D)
    def forward(s,t,m):
        h=s.inp(t); a=s.att(h).squeeze(-1).masked_fill(m==0,-1e9); al=torch.softmax(a,1); return (al.unsqueeze(-1)*s.val(h)).sum(1)
enc=Enc(); enc.load_state_dict(torch.load(f'{base}/.cache/enc_concept.pt')); enc.eval()
def fold(tokens):
    if not tokens: return np.zeros(D,np.float32)
    arr=np.zeros((1,len(tokens),D+1),np.float32); m=np.ones((1,len(tokens)),np.float32)
    for q,(f,v) in enumerate(tokens): arr[0,q,:D]=f; arr[0,q,D]=v
    with torch.no_grad(): return enc(torch.tensor(arr),torch.tensor(m)).numpy()[0]

def poster_url(j):
    mid=inv_iid[j]; key=str(mid)
    if key in poster_cache: pp=poster_cache[key]
    else:
        pp=None
        if TMDB and mid in m2t:
            try:
                r=requests.get(f"https://api.themoviedb.org/3/movie/{m2t[mid]}",params={'api_key':TMDB},timeout=6)
                if r.status_code==200: pp=r.json().get('poster_path')
            except Exception: pp=None
        poster_cache[key]=pp; json.dump(poster_cache,open(POSTER,'w'))
    return ("https://image.tmdb.org/t/p/w185"+pp) if pp else None

# ---------- grounding ----------
def ground(qtype, text):
    text=(text or '').strip()
    if not text: return None
    if qtype in ('genre','avoidgenre'):
        g=difflib.get_close_matches(text.title(),GENRES,n=1,cutoff=0.6)
        v=NEG if qtype=='avoidgenre' else POS
        return (gcent[g[0]], v, f"{'avoid ' if qtype=='avoidgenre' else ''}genre: {g[0]}") if g else None
    if qtype in ('actor','director'):
        p=difflib.get_close_matches(text.lower(),ALL_PEOPLE,n=1,cutoff=0.7)
        return (pcent[p[0]], POS, f"{qtype}: {p[0].title()}") if p and p[0] in pcent else None
    # movie types: prefer exact/prefix/whole-word substring, then fuzzy WITH word-overlap check (avoid far-fetched matches)
    key=normt(text); kw=set(key.split())
    if not kw: return None
    cand=[t for t in norm_title if t==key] \
       or [t for t in norm_title if t.startswith(key+' ')] \
       or [t for t in norm_title if (' '+key+' ') in (' '+t+' ')]
    if cand: cand=[min(cand,key=len)]
    else:
        cand=[]
        for t in difflib.get_close_matches(key,list(norm_title),n=4,cutoff=0.72):   # stricter cutoff
            tw=set(t.split())
            if len(kw & tw)/max(len(kw),1) >= 0.6: cand=[t]; break                  # require >=60% query words present
    if not cand: return None                                                        # honest "not in catalogue" instead of a far-fetched match
    if not cand: return None
    j=norm_title[cand[0]]; val=NEG if qtype=='hate' else POS
    return (Q[j], val, f"{title[j]}"), j

# ---------- the DEPLOYED learned policy (Paper-D anytime no-repeat asker) ----------
TYPES=['gem','fav','align','hate','genre','avoidgenre','actor','director','whatdoyoulike']; TI={t:i for i,t in enumerate(TYPES)}
class Pol(nn.Module):
    def __init__(s): super().__init__(); s.net=nn.Sequential(nn.Linear(D+1,128),nn.ReLU(),nn.Linear(128,128),nn.ReLU(),nn.Linear(128,len(TYPES)))
    def forward(s,x): return s.net(x)
pol=Pol(); _HASPOL=False
try:
    pol.load_state_dict(torch.load(f"{base}/.cache/polopen_{'_'.join(TYPES)}_wf1.0wt1.0.pt")); pol.eval(); _HASPOL=True
except Exception as e: print('no policy ckpt, using fixed order:',e)
TMPL={'gem':"Name a film you love that you think is underrated.",'fav':"Name a movie you love.",
      'align':"What's your all-time favourite film?",'hate':"A movie you really didn't enjoy?",
      'genre':"A genre you love?",'avoidgenre':"A genre you tend to avoid?",'actor':"A favourite actor of yours?",
      'director':"A favourite director?",'whatdoyoulike':"To start — what do you like? Name any film."}
GTYPE={'gem':'movie','fav':'movie','align':'movie','hate':'hate','whatdoyoulike':'movie','genre':'genre',
       'avoidgenre':'avoidgenre','actor':'actor','director':'director'}
FIXED=['whatdoyoulike','actor','gem','director','hate','genre']                  # fallback order if no policy ckpt
def pick_type(tokens, used, n):
    if not _HASPOL:
        for t in FIXED:
            if t not in used: return t
        return None
    u=fold(tokens); st=torch.tensor(np.concatenate([u,[n/8.]])[None],dtype=torch.float32)
    with torch.no_grad(): lg=pol(st)[0].numpy()
    for t in TYPES:
        if t in used: lg[TI[t]]=-1e9
    return TYPES[int(np.argmax(lg))]

# ---------- closed continuous probe (Paper C D1) + phrase bank (for snapping = interpretable closed Qs) ----------
class Actor(nn.Module):
    def __init__(s): super().__init__(); s.f=nn.Sequential(nn.Linear(D+1,128),nn.ReLU(),nn.Linear(128,128),nn.ReLU(),nn.Linear(128,D))
d1=Actor(); _HASD1=False
try: d1.load_state_dict(torch.load(f'{base}/.cache/policy_phase3_d1divw_last.pt')['actor']); d1.eval(); _HASD1=True
except Exception as e: print('no D1 actor:',e)
try:
    _pb=np.load(f'{base}/.cache/phrasebank.npz',allow_pickle=True); PV=_pb['vec'].astype(np.float32); PU=_pb['unit'].astype(np.float32); PL=list(_pb['label']); PK=list(_pb['kind'])
    _CN=float(np.linalg.norm(np.load(f'{base}/.cache/Ec_concept.npy'),axis=1).mean()); _HASPB=True
except Exception as e: PV=PU=None; PL=PK=[]; _HASPB=False; print('no phrasebank:',e)
_ISCON=np.array([PK[k] in ('tag','genre') for k in range(len(PK))]) if _HASPB else None   # concept mask (tags+genres) for the discrete policy
_SENS=['gay','lesbian','glbt','lgbt','queer','homophob','transgender','aids','racism','racist',' race','nazi','holocaust','rape','incest','pedoph','abortion','slavery','genocide','terroris','suicide','disab','autism','islam','muslim','jewish','religio']
_BAD=(np.array([any(s in PL[k].lower() for s in _SENS) for k in range(len(PK))]) if _HASPB else None)   # sensitive/identity tags -> never ask about these
_OKC=((_ISCON & ~_BAD) if _HASPB else None)                                     # concepts safe to ask about
_POPDIR=(PU.mean(0)/(np.linalg.norm(PU.mean(0))+1e-9)) if _HASPB else None       # central/popular direction (cold-start discrete target)
def closed_phrase(tokens, used_ph, turn):                                       # CLOSED = Paper-C D1 continuous query from belief -> snap to nearest UNUSED phrase (full bank)
    u=fold(tokens); x=torch.tensor(np.concatenate([u,[turn/8.]])[None],dtype=torch.float32)
    with torch.no_grad(): qv=d1.f(x).numpy()[0]
    qn=qv/(np.linalg.norm(qv)+1e-9); sc=(PU@qn).copy()
    for k in used_ph: sc[k]=-1e9
    k=int(np.argmax(sc)); return k, PL[k]
def discrete_phrase(tokens, used_ph):                                           # DISCRETE = concept-selection policy (NOT D1): belief-aligned + DIVERSITY penalty (avoid near-duplicate concepts)
    u=fold(tokens); n=np.linalg.norm(u)
    sc=(PU@(_POPDIR if n<1e-6 else u/n)).copy(); sc[~_ISCON]=-1e9
    for k in used_ph: sc=sc-0.6*np.clip(PU@PU[k],0,1); sc[k]=-1e9              # penalise concepts similar to already-asked -> diverse questions
    k=int(np.argmax(sc)); return k, PL[k]
def triangulate(tokens, used_ph, turn, K=3):                                    # CONTINUOUS: D1 query -> OMP-K signed phrase blend (NO snap). Returns reconstruction q_hat (what we fold), the K phrases + weights.
    u=fold(tokens); x=torch.tensor(np.concatenate([u,[turn/8.]])[None],dtype=torch.float32)
    with torch.no_grad(): qv=d1.f(x).numpy()[0]
    qn=qv/(np.linalg.norm(qv)+1e-9); ch=[]; res=qn.copy(); ban=set(used_ph); A=w=None
    for _ in range(K):
        pr=np.abs(PU@res); pr[~_OKC]=-1.                                       # DEMO: triangulate onto plain SAFE single concepts (clean 3-word blend)
        for k in ban|set(ch): pr[k]=-1.
        i=int(np.argmax(pr)); ch.append(i); A=PU[ch].T; w=np.linalg.lstsq(A,qn,rcond=None)[0]; res=qn-A@w
    qh=(PU[ch]*np.asarray(w)[:,None]).sum(0); qh=qh/(np.linalg.norm(qh)+1e-9)
    return qh, ch, [float(x) for x in w]
_CSYS=("You turn a movie-preference elicitation query into ONE short natural question a recommender asks a new user. "
"The query is a DIRECTION in taste space: a signed blend of descriptors weight*[descriptor]. Sign = which side (positive=TOWARD, negative=AWAY); magnitude = importance. "
"The descriptors are almost always facets of ONE underlying taste axis. Infer that axis, ask one warm conversational question about it from the pole the signs indicate, and do NOT enumerate the descriptors. "
"If they clearly share no axis, ask only about the single largest-magnitude one. Output only the question, one sentence (<=18 words).")
def snap_one(tokens, used_ph, turn, conc_only):                                # SNAP: D1 query -> single nearest phrase (concepts-only or full rich bank)
    u=fold(tokens); x=torch.tensor(np.concatenate([u,[turn/8.]])[None],dtype=torch.float32)
    with torch.no_grad(): qv=d1.f(x).numpy()[0]
    qn=qv/(np.linalg.norm(qv)+1e-9); sc=(PU@qn).copy()
    sc[_BAD]=-1e9                                                              # never snap to sensitive/identity tags
    if conc_only: sc[~_OKC]=-1e9
    for k in used_ph: sc[k]=-1e9
    k=int(np.argmax(sc)); return k, PL[k]
_SQCACHE={}
def nice_snap_q(label):                                                         # turn ONE snapped descriptor into a natural yes/no question (so snap modes aren't clunky); cached; template fallback
    if label in _SQCACHE: return _SQCACHE[label]
    q=f"Do you enjoy “{label}” films?"
    if os.environ.get('OPENAI_API_KEY') and os.environ.get('USE_LLM'):
        try:
            from openai import OpenAI; cl=OpenAI(); M=os.environ.get('LLM_MODEL','gpt-4.1')
            kw={'model':M,'messages':[{'role':'user','content':f'Turn this movie descriptor into ONE short, natural yes/no question a recommender could ask a new user. Do not quote it literally; make it conversational (<=16 words). Descriptor: "{label}"'}]}
            kw['max_completion_tokens' if M.startswith(('gpt-5','o1','o3','o4')) else 'max_tokens']=(300 if M.startswith(('gpt-5','o1','o3','o4')) else 30)
            if M.startswith(('gpt-5','o1','o3','o4')): kw['reasoning_effort']='minimal'
            q=cl.chat.completions.create(**kw).choices[0].message.content.strip().strip('"')
        except Exception: pass
    _SQCACHE[label]=q; return q
_SELCACHE={}
def cont_align(tokens, used_ph, turn):                                          # SELECT-AND-ALIGN: LLM picks a COHERENT subset of the top-6 aligned concepts + writes ONE question; we fold ONLY that subset (ask==fold)
    u=fold(tokens); x=torch.tensor(np.concatenate([u,[turn/8.]])[None],dtype=torch.float32)
    with torch.no_grad(): qv=d1.f(x).numpy()[0]
    qn=qv/(np.linalg.norm(qv)+1e-9); key=tuple(np.round(qn,2).tolist())
    if key in _SELCACHE: return _SELCACHE[key]
    al=(PU@qn).copy(); al[~_OKC]=0.                                            # only safe concepts
    for k in used_ph: al[k]=0.
    top=np.argsort(-np.abs(al))[:6]; cand=[(int(t),'+' if al[t]>0 else '−') for t in top]
    used=[cand[0][0]]; ques=f"Do you lean toward {PL[cand[0][0]]} films?"       # fallback = dominant concept
    if os.environ.get('OPENAI_API_KEY') and os.environ.get('USE_LLM'):
        try:
            from openai import OpenAI; import json as _J; cl=OpenAI(); M=os.environ.get('LLM_MODEL','gpt-4.1')
            lines="\n".join(f"{n+1}. {s} {PL[i]}" for n,(i,s) in enumerate(cand))
            sysp=("You help a movie recommender ask ONE natural yes/no question. You get up to 6 numbered candidate taste descriptors, "
                  "each marked + (user may lean TOWARD) or − (AWAY). Pick the SUBSET (1-3) that share a coherent theme and combine into "
                  "ONE natural question a person can answer; ignore the rest. Ask about film STYLE / MOOD / GENRE only — NEVER frame a "
                  "question around a person's identity, sexuality, race, religion, or disability; if a descriptor is about those, skip it. "
                  'Return JSON {"question":"<one sentence, do not quote the descriptors>","used":[<the NUMBERS you actually used>]}.')
            kw={'model':M,'response_format':{'type':'json_object'},
                'messages':[{'role':'system','content':sysp},{'role':'user','content':'Candidates:\n'+lines}]}
            kw['max_completion_tokens' if M.startswith(('gpt-5','o1','o3','o4')) else 'max_tokens']=(600 if M.startswith(('gpt-5','o1','o3','o4')) else 120)
            if M.startswith(('gpt-5','o1','o3','o4')): kw['reasoning_effort']='minimal'
            o=_J.loads(cl.chat.completions.create(**kw).choices[0].message.content); ques=o['question'].strip()
            nums=[int(n) for n in o.get('used',[]) if str(n).isdigit() and 1<=int(n)<=len(cand)]   # INDEX-based (robust)
            used=[cand[n-1][0] for n in nums] or [cand[0][0]]
        except Exception: pass
    A=PU[used].T; w=np.linalg.lstsq(A,qn,rcond=None)[0]; qh=(PU[used]*np.asarray(w)[:,None]).sum(0); qh=qh/(np.linalg.norm(qh)+1e-9)
    res=(ques,used,qh); _SELCACHE[key]=res; return res
_CRCACHE={}
def cont_render(ch, w):                                                         # render the triangulated blend as NL (good OpenAI model); cached; fallback = dominant phrase
    ckey=tuple(ch)
    if ckey in _CRCACHE: return _CRCACHE[ckey]
    blend="  ".join(f"{wi:+.2f}*[{PL[c]}]" for c,wi in zip(ch,w))
    if not (os.environ.get('OPENAI_API_KEY') and os.environ.get('USE_LLM')):
        d=ch[int(np.argmax(np.abs(w)))]; return f"Do you lean toward {PL[d]} films?"
    try:
        from openai import OpenAI; cl=OpenAI(); M=os.environ.get('LLM_MODEL','gpt-4.1')
        kw={'model':M,'messages':[{'role':'system','content':_CSYS},{'role':'user','content':'Query: '+blend}]}
        kw['max_completion_tokens' if M.startswith(('gpt-5','o1','o3','o4')) else 'max_tokens']=(400 if M.startswith(('gpt-5','o1','o3','o4')) else 40)
        if M.startswith(('gpt-5','o1','o3','o4')): kw['reasoning_effort']='minimal'
        nl=cl.chat.completions.create(**kw).choices[0].message.content.strip().strip('"'); _CRCACHE[ckey]=nl; return nl
    except Exception as e:
        d=ch[int(np.argmax(np.abs(w)))]; return f"Do you lean toward {PL[d]} films?"

# ---------- optional cheap-LLM question rendering ----------
def llm_render(qtype, belief_titles):
    if not os.environ.get('OPENAI_API_KEY') or not os.environ.get('USE_LLM'): return TMPL[qtype]
    try:
        from openai import OpenAI; cl=OpenAI()
        ctx=("So far they mentioned: "+", ".join(belief_titles)+". " if belief_titles else "")
        p=(f"You are a friendly movie chat assistant doing cold-start preference elicitation. {ctx}"
           f"Rewrite this elicitation question as ONE short, warm, natural sentence (<=16 words), no preamble: \"{TMPL[qtype]}\"")
        r=cl.chat.completions.create(model=os.environ.get('LLM_MODEL','gpt-4o-mini'),messages=[{'role':'user','content':p}],max_completion_tokens=40)
        return r.choices[0].message.content.strip().strip('"')
    except Exception: return TMPL[qtype]

def _top(s,K):                                                                 # NO synchronous poster fetch here -> /step stays fast; client lazy-loads posters via /poster/<j>
    return [{'title':title.get(int(j),f'movie {int(j)}'),'j':int(j)} for j in np.argsort(-s)[:K]]
def recommend(tokens, seen, K=6):
    u=fold(tokens); s=popb+Ql@u
    for j in seen: s[j]=-1e9
    sf=s.copy(); st=s.copy(); st[headmask]=-1e9                                 # full vs Cremonesi long-tail (head masked)
    return _top(sf,K), _top(st,K)

app=Flask(__name__)

def ground_open(ty, ans):                                                       # ground per asked type; if it fails, rescue with other entity types (user may answer a film with an actor etc.)
    g=ground(GTYPE[ty],ans)
    if g is not None: return g
    for alt in ('fav','actor','director','genre'):
        if alt==GTYPE[ty]: continue
        gg=ground(alt,ans)
        if gg is not None:
            if ty in ('hate','avoidgenre'):                                     # keep the question's sign (negative) even on a cross-type rescue
                if isinstance(gg,tuple) and len(gg)==2: (f,v,l),j=gg; return ((f,NEG,l),j)
                f,v,l=gg; return (f,NEG,l)
            return gg
    return None
BUDGET=6
def turn_kind(mode,t):                                                          # open / closed=snap-rich / snapc=snap-concept / cont=triangulated axis / align=LLM select-and-align / discrete / mixed
    if mode=='cont': return ('cont',False)
    if mode=='align': return ('align',False)
    if mode=='snapc': return ('snapc',False)
    if mode=='closed': return ('closed',False)
    if mode=='discrete': return ('closed',True)
    if mode=='mixed': return ('open',False) if t<(BUDGET+1)//2 else ('align',False)
    return ('open',False)
@app.route('/step', methods=['POST'])
def step():
    data=request.get_json(force=True); answers=data.get('answers',[]); mode=data.get('mode','open')
    if mode in ('closed','discrete','mixed','cont','align','snapc') and not (_HASD1 and _HASPB): mode='open'   # fall back if D1/bank missing
    tokens=[]; seen=set(); used=set(); used_ph=[]; transcript=[]; named=[]; named_items=[]
    def ask(t):                                                                 # the question posed at turn t (deterministic replay)
        k,_=turn_kind(mode,t)
        if k=='open':
            ty=pick_type(tokens,used,t); return ('open',ty,TMPL[ty]) if ty else (None,None,None)
        elif k=='cont':
            qh,ch,w=triangulate(tokens,used_ph,t); return ('cont',(qh,ch,w),None)   # NL rendered only for the new question (below)
        elif k=='align':
            ques,usel,qh=cont_align(tokens,used_ph,t); return ('align',(qh,usel),ques)   # LLM select-and-align (cached, deterministic replay)
        elif k=='snapc':
            ki,lab=snap_one(tokens,used_ph,t,True); return ('snapc',ki,f"Do you enjoy “{lab}” films?")
        else:
            disc=turn_kind(mode,t)[1]
            ki,lab=(discrete_phrase(tokens,used_ph) if disc else closed_phrase(tokens,used_ph,t)); return ('closed',ki,f"Do you enjoy “{lab}” films?")
    for t,ans in enumerate(answers):
        kind,ref,qtext=ask(t)
        if kind is None: break
        if kind=='open':
            used.add(ref); g=ground_open(ref,ans)
            if g is None: transcript.append({'q':qtext,'a':ans,'grounded':None,'modality':'text'}); continue
            if isinstance(g,tuple) and len(g)==2: (fac,val,lab),j=g[0],g[1]; seen.add(j); named.append(lab); named_items.append((j,val>0))
            else: fac,val,lab=g; named.append(lab)
            tokens.append((fac,val)); transcript.append({'q':qtext,'a':ans,'grounded':lab,'modality':'text'})
        elif kind=='cont':                                                      # triangulated continuous: fold the reconstruction q_hat (NOT a single phrase)
            qh,ch,w=ref; used_ph.extend(ch); a=ans.lower().strip()
            order=sorted(range(len(ch)),key=lambda i:-abs(w[i]))                # strongest-weight concept first
            blend=" ".join(f"[{'+' if w[i]>0 else '−'} {PL[ch[i]]}]" for i in order)   # 3 concepts, sign = toward/away
            if a in ('yes','y','yeah','sure','love it'): tokens.append((qh*_CN,POS)); gl=f"✓ yes · continuous query ≈ {blend}"
            elif a in ('no','n','nope','not really'): tokens.append((qh*_CN,NEG)); gl=f"✗ no · continuous query ≈ {blend}"
            else: gl=None
            transcript.append({'q':cont_render(ch,w),'a':ans,'grounded':gl,'modality':'yesno'})   # nice rendered Q in history
        elif kind=='align':                                                     # select-and-align: fold ONLY the concepts the LLM actually used (ask==fold)
            qh,used=ref; used_ph.extend(used); a=ans.lower().strip(); usedl=", ".join(PL[i] for i in used)
            if a in ('yes','y','yeah','sure','love it'): tokens.append((qh*_CN,POS)); gl=f"✓ yes · folded {len(used)} concept(s): {usedl}"
            elif a in ('no','n','nope','not really'): tokens.append((qh*_CN,NEG)); gl=f"✗ no · folded {len(used)} concept(s): {usedl}"
            else: gl=None
            transcript.append({'q':qtext,'a':ans,'grounded':gl,'modality':'yesno'})
        else:
            used_ph.append(ref); a=ans.lower().strip(); lab=PL[ref]
            if a in ('yes','y','yeah','sure','love it'): tokens.append((PU[ref]*_CN,POS)); gl=f"✓ likes “{lab}”"
            elif a in ('no','n','nope','not really'): tokens.append((PU[ref]*_CN,NEG)); gl=f"✗ not “{lab}”"
            else: gl=None
            transcript.append({'q':nice_snap_q(lab),'a':ans,'grounded':gl,'modality':'yesno'})   # nice rendered Q in history
    nturn=len(answers); done=nturn>=BUDGET; nq=None; modality='text'
    if not done:
        kind,ref,qtext=ask(nturn)
        if kind is None: done=True
        elif kind=='cont': qh,ch,w=ref; nq=cont_render(ch,w); modality='yesno'   # triangulated blend -> LLM natural question
        elif kind in ('closed','snapc'): nq=nice_snap_q(PL[ref]); modality='yesno'   # single snapped phrase -> natural question (consistent phrasing across modes)
        elif kind=='open': nq=llm_render(ref,[n for n in named if ':' not in n][:3]); modality='text'
        else: nq=qtext; modality='yesno'
    full,tail=(recommend(tokens,seen) if tokens else ([],[]))
    tmap={'traj':[to2d(fold(tokens[:k])) for k in range(1,len(tokens)+1)],   # belief trajectory (moves each answer)
          'items':[{'xy':item2d(j),'like':lk,'title':title.get(j,'')} for (j,lk) in named_items],   # named movies light up
          'recs':[{'xy':item2d(r['j']),'title':r['title']} for r in full[:5]]} if tokens else {'traj':[],'items':[],'recs':[]}
    return jsonify({'recs_full':full,'recs_tail':tail,'next_question':nq,'modality':modality,'turn':nturn,'done':done,
                    'transcript':transcript,'mode':mode,'policy':'learned' if _HASPOL else 'fixed','map':tmap})

PAGE="""<!doctype html><html><head><meta charset=utf-8><title>CASPER chat</title>
<style>body{font-family:-apple-system,Segoe UI,Roboto,sans-serif;max-width:980px;margin:18px auto;color:#222}
.wrap{display:flex;gap:24px;align-items:flex-start}.chat{flex:0 1 auto;max-height:60vh;overflow-y:auto;padding-right:6px}.recs{flex:1}
.bub{background:#eef4fb;padding:10px 14px;border-radius:14px;margin:8px 0;max-width:85%}
.me{background:#1f77b4;color:#fff;margin-left:auto}.g{font-size:12px;color:#16794a;margin:2px 0 8px}
input{width:70%;padding:10px;border:1px solid #ccc;border-radius:8px}
button{padding:10px 16px;border:0;background:#1f77b4;color:#fff;border-radius:8px;cursor:pointer;margin-right:6px}
select{padding:6px;border-radius:8px}
.card{display:inline-block;width:110px;margin:6px;vertical-align:top;font-size:12px;text-align:center}
.card img{width:104px;height:156px;object-fit:cover;border-radius:8px;background:#ddd}
.ph{width:104px;height:156px;border-radius:8px;background:#dde3ea;display:flex;align-items:center;justify-content:center;color:#889;font-size:11px;padding:4px}
h2{margin:4px 0}</style></head><body>
<h2>CASPER — cold-start conversational recommender</h2>
<div style="margin-bottom:10px;font-size:14px">Mode:
<select id=mode onchange="setMode(this.value)">
 <option value=open>Open recall — name things you like</option>
 <option value=mixed>Mixed — recall, then refine</option>
 <option value=align>Continuous — LLM select-and-align (fold only what's asked)</option>
 <option value=cont>Continuous — triangulated axis-synthesis (fold all 3)</option>
 <option value=snapc>Snap → single concept (nearest tag/genre)</option>
 <option value=closed>Snap → single rich-vocab phrase</option>
 <option value=discrete>Discrete — belief-aligned concept policy</option>
</select></div>
<div class=wrap>
<div style="flex:1.1;display:flex;flex-direction:column">
 <div class=chat id=chat></div>
 <div id=txtin style="margin-top:10px"><input id=inp placeholder="type a movie / actor / genre…"><button onclick="send()">Send</button><button id=undo onclick="undo()" style="background:#ccc;color:#333;display:none">↩ Undo</button></div>
 <div id=ynin style="margin-top:10px;display:none">
 <button onclick="ans('yes')">👍 Yes</button><button onclick="ans('no')" style="background:#888">👎 No</button><button onclick="ans('skip')" style="background:#ccc;color:#333">Skip</button><button onclick="undo()" style="background:#ccc;color:#333">↩ Undo</button>
 </div>
</div>
<div class=recs>
<b>🍿 Top picks</b><div id=recsF><i style="color:#889">answer a question to begin…</i></div>
<b>💎 Hidden gems (long tail)</b><div id=recsT></div>
</div></div>
<div style="margin-top:14px"><b>🗺️ Taste map</b> <span style="font-size:12px;color:#889">— your belief moves as you answer (blue path); 🟢 liked / 🔴 disliked films light up; 🟠 = current picks</span>
<canvas id=map width=920 height=320 style="border:1px solid #e6ebf2;border-radius:8px;width:100%;margin-top:6px"></canvas></div>
<script>
var answers=[], mode='open', BG=null;
fetch('/map').then(function(r){return r.json();}).then(function(d){BG=d.bg; drawMap({traj:[],items:[],recs:[]});});
function drawMap(m){
 var cv=document.getElementById('map'); if(!cv)return; var x=cv.getContext('2d'), W=cv.width, H=cv.height;
 x.clearRect(0,0,W,H);
 function px(p){return [24+p[0]*(W-48), H-24-p[1]*(H-48)];}
 if(BG){ x.fillStyle='#e6ebf2'; BG.forEach(function(p){var q=px(p);x.beginPath();x.arc(q[0],q[1],1.5,0,7);x.fill();}); }
 if(m.traj && m.traj.length){
  x.strokeStyle='#1f77b4'; x.lineWidth=2; x.beginPath();
  m.traj.forEach(function(p,i){var q=px(p); if(i===0)x.moveTo(q[0],q[1]); else x.lineTo(q[0],q[1]);}); x.stroke();
  m.traj.forEach(function(p,i){var q=px(p),lastp=(i===m.traj.length-1); x.fillStyle=lastp?'#0b3d91':'#7aa8d6'; x.beginPath(); x.arc(q[0],q[1],lastp?6:3,0,7); x.fill(); if(lastp){x.fillStyle='#0b3d91';x.font='bold 12px sans-serif';x.fillText('you are here',q[0]+9,q[1]-6);} });
 }
 (m.items||[]).forEach(function(it){var q=px(it.xy); x.fillStyle=it.like?'#16a34a':'#dc2626'; x.beginPath();x.arc(q[0],q[1],5,0,7);x.fill(); x.fillStyle='#334';x.font='11px sans-serif';x.fillText((it.title||'').slice(0,20),q[0]+7,q[1]+3);});
 (m.recs||[]).forEach(function(it){var q=px(it.xy); x.strokeStyle='#f59e0b';x.lineWidth=2;x.beginPath();x.arc(q[0],q[1],7,0,7);x.stroke();});
}
function esc(s){return (s||'').replace(/</g,'&lt;').replace(/>/g,'&gt;');}
function fill(el,arr){
 el.innerHTML='';
 arr.forEach(function(m){
  var c=document.createElement('div'); c.className='card';
  var im=document.createElement('img'); im.src='/poster/'+m.j; im.loading='lazy';
  im.onerror=function(){ var p=document.createElement('div'); p.className='ph'; p.textContent=m.title; if(im.parentNode)c.replaceChild(p,im); };
  var t=document.createElement('div'); t.textContent=m.title;
  c.appendChild(im); c.appendChild(t); el.appendChild(c);
 });
}
function render(d){
 var c=document.getElementById('chat'); c.innerHTML='';
 d.transcript.forEach(function(t){
  c.innerHTML+='<div class=bub>'+esc(t.q)+'</div><div class="bub me">'+esc(t.a)+'</div>';
  c.innerHTML+= t.grounded ? '<div class=g>'+esc(t.grounded)+'</div>' : '<div class=g style="color:#b00">could not match — skipped</div>';
 });
 if(d.next_question){ c.innerHTML+='<div class=bub>'+esc(d.next_question)+'</div>'; }
 else { c.innerHTML+='<div class=bub>That is enough to go on — enjoy your picks! 🍿</div>'; }
 var rf=document.getElementById('recsF'), rt=document.getElementById('recsT');
 if(!(d.recs_full||[]).length){ rf.innerHTML='<i style="color:#889">answer a question to begin…</i>'; rt.innerHTML=''; }
 else { fill(rf,d.recs_full); fill(rt,d.recs_tail); }
 var txt=document.getElementById('txtin'), yn=document.getElementById('ynin');
 if(d.done){ txt.style.display='block'; yn.style.display='none'; document.getElementById('inp').style.display='none'; }
 else if(d.modality==='yesno'){ txt.style.display='none'; yn.style.display='block'; }
 else { txt.style.display='block'; yn.style.display='none'; var i=document.getElementById('inp'); i.style.display=''; i.value=''; i.focus(); }
 document.getElementById('undo').style.display = answers.length? 'inline-block':'none';
 if(d.map) drawMap(d.map);
 c.scrollTop=c.scrollHeight;
}
async function step(){
 var res=await fetch('/step',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({answers:answers,mode:mode})});
 render(await res.json());
}
function send(){ var v=document.getElementById('inp').value.trim(); if(!v)return; answers.push(v); step(); }
function ans(v){ answers.push(v); step(); }
function undo(){ if(answers.length){ answers.pop(); step(); } }
function setMode(m){ mode=m; answers=[]; step(); }
document.addEventListener('keydown',function(e){ if(e.key==='Enter' && document.getElementById('txtin').style.display!=='none') send(); });
step();
</script></body></html>"""

from flask import redirect, Response
@app.route('/poster/<int:j>')
def poster(j):                                                                 # lazy poster (cached) -> redirect to TMDB image; 1x1 gif if none (keeps /step fast)
    url=poster_url(j) if 0<=j<ni else None
    return redirect(url,302) if url else ('',404)
@app.route('/map')
def mapbg(): return jsonify({'bg':_bg})                                          # static background item cloud (fetched once)
@app.route('/')
def home(): return render_template_string(PAGE)

if __name__=='__main__':
    print(f"loaded {ni} movies, {len(pcent)} people, {len(GENRES)} genres, POS={POS:.3f} NEG={NEG:.3f}; TMDB={'yes' if TMDB else 'no posters'}")
    app.run(port=5005,debug=False)
