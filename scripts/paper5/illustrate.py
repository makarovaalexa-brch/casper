"""Paper E illustration: for a few continuous queries show (1) the OMP-3 phrase blend, (2) the LLM's NL question,
(3) the GEOMETRIC nearest concept (the snap) + its cos to q, (4) the concept the LLM's WORDING fuzzy-matches to (SBERT
text-sim on labels) + how far THAT concept sits from q geometrically. Tests: does the verbalised interpretation map to a
concept that is far from q in embedding space? Usage: python scripts/paper5/illustrate.py [N]"""
import os, sys, numpy as np
base='C:/dev/phd/casper/data/movielens'
for _l in open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),'..','.env')):
    if _l.startswith('OPENAI_API_KEY='): os.environ['OPENAI_API_KEY']=_l.split('=',1)[1].strip().strip('"')
import openai; from sentence_transformers import SentenceTransformer
d=np.load(f'{base}/.cache/qdump.npz',allow_pickle=True); QV=d['q'].astype(np.float32)
pb=np.load(f'{base}/.cache/phrasebank.npz',allow_pickle=True); PU=pb['unit'].astype(np.float32); PL=list(pb['label']); PK=list(pb['kind'])
CONC=np.array([i for i,k in enumerate(PK) if k in('tag','genre')])              # "existing concepts" = plain genome tags + genres
sb=SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2'); PLE=sb.encode(PL,normalize_embeddings=True,show_progress_bar=False)
cl=openai.OpenAI(); MODEL=os.environ.get('MODEL','gpt-4.1-mini')
SYS=("You turn a movie-preference 'elicitation query' into ONE short natural question a recommender asks a new user. "
"The query is a DIRECTION in taste space: a signed blend of descriptors weight*[descriptor]. Sign=which side (positive=TOWARD, negative=AWAY); magnitude=importance. "
"The descriptors are almost always facets of ONE underlying taste axis. Infer that single axis, ask one conversational question about it from the pole the signs indicate, and do NOT enumerate the descriptors. "
"If they clearly share no axis, ask only about the single largest-magnitude one. Output only the question, one sentence.")
def omp(q,K):
    ch=[]; res=q.copy(); A=None; w=None
    for _ in range(K):
        pr=PU@res
        if ch: pr[ch]=0.
        i=int(np.argmax(np.abs(pr))); ch.append(i); A=PU[ch].T; w=np.linalg.lstsq(A,q,rcond=None)[0]; res=q-A@w
    return ch,w
import json as _json
RPATH=f'{base}/../../experiments/paper5/renders_{MODEL}.json'                   # shared persisted cache with LLMDROP
try: RC=_json.load(open(RPATH))
except Exception: RC={}
def render(q):
    ch,w=omp(q,3); um="Query: "+"  ".join(f"{wi:+.2f}*[{PL[c]}]" for c,wi in zip(ch,w))
    if um in RC: return RC[um],ch,w
    if os.environ.get('NOLLM'): return None,ch,w
    try:
        r=cl.chat.completions.create(model=MODEL,messages=[{'role':'system','content':SYS},{'role':'user','content':um}],max_tokens=60)
        nl=r.choices[0].message.content.strip(); RC[um]=nl; _json.dump(RC,open(RPATH,'w'),indent=0,ensure_ascii=False); return nl,ch,w
    except Exception as e: return None,ch,w
N=int(sys.argv[1]) if len(sys.argv)>1 else 8
rng=np.random.default_rng(7); pick=rng.choice(len(QV),N,replace=False)
for i in pick:
    q=QV[i]; qn=q/(np.linalg.norm(q)+1e-9); nl,ch,w=render(q)
    ks=int(np.argmax(PU@qn)); snap_cos=float(PU[ks]@qn)                          # geometric snap (nearest phrase to q)
    kc=int(CONC[np.argmax(PU[CONC]@qn)]); csnap=float(PU[kc]@qn)                 # nearest plain CONCEPT to q
    print("="*90)
    print("BLEND :", "  ".join(f"{wi:+.2f}*[{PL[c]}]" for c,wi in zip(ch,w)))
    if nl: print("LLM Q :", nl)
    print(f"geom snap (nearest phrase) : [{PL[ks]}]  cos={snap_cos:.2f}")
    print(f"geom nearest CONCEPT       : [{PL[kc]}]  cos={csnap:.2f}   {'<-- no single concept is close' if csnap<0.55 else ''}")
    if nl:
        e=sb.encode([nl],normalize_embeddings=True)[0]; kt=int(CONC[np.argmax(PLE[CONC]@e)]); tcos=float(PU[kt]@qn)
        print(f"LLM-wording -> CONCEPT      : [{PL[kt]}]  (text-match) ... its cos to q = {tcos:.2f}   {'<-- FAR from q in embedding space' if tcos<0.4 else ''}")
