"""Paper E LLM round-trip: continuous query q -> OMP phrase blend -> LLM renders NL question -> re-embed NL (SBERT) ->
recover phrases -> reconstruct q_hat -> round-trip fidelity cos(q,q_hat) + phrase-recovery recall. Quantifies the
"language bottleneck" of naming a continuous elicitation query, per model tier. DRYRUN=1 => just estimate token cost.
Usage: QDUMP first (TRIANGULATE=1 QDUMP=path), then: MODEL=gpt-4o-mini python scripts/paper5/roundtrip.py <dump.npz>
       DRYRUN=1 NQ=500 python scripts/paper5/roundtrip.py <dump.npz>   (cost estimate only, no API)"""
import os, sys, json, numpy as np
base='C:/dev/phd/casper/data/movielens'
dump=sys.argv[1] if len(sys.argv)>1 else f'{base}/.cache/qdump.npz'
d=np.load(dump,allow_pickle=True); QV=d['q'].astype(np.float32); BL=list(d['blends']); LABELS=list(d['labels'])
pb=np.load(f'{base}/.cache/phrasebank.npz',allow_pickle=True); PU=pb['unit'].astype(np.float32)
NQ=int(os.environ.get('NQ','500')); QV=QV[:NQ]; BL=BL[:NQ]
SYSTEM=("You turn a movie-preference 'elicitation query' into ONE short, natural question a recommender asks a new user.\n"
"The query is a DIRECTION in taste space, given as a signed blend of descriptors: weight * [descriptor].\n"
" - Sign = which side to ask from. Positive => lean TOWARD it; negative => AWAY.\n - Magnitude = how much it matters.\n"
"The descriptors are almost always facets of ONE underlying taste axis. Your job:\n"
" 1. Infer the single axis they share (serious<->escapist, calm<->intense, mainstream<->arthouse...).\n"
" 2. Ask one conversational question about THAT axis, from the pole the signs point to.\n"
" 3. Do NOT list or name the descriptors. Never enumerate them literally.\n"
"If the descriptors clearly do NOT share an axis, ignore the small ones and ask only about the single largest-magnitude one.\n"
"Output: the question only. One sentence. Friendly, concrete, answerable by anyone.")
def user_msg(b): return "Query: "+"  ".join(f"{w:+.2f}*[{l}]" for w,l in zip(b['w'],b['labels']))

# ---- token-cost estimate (no API needed) --------------------------------------
def toks(s): return int(len(s)/4)+1                                            # ~4 chars/token
sys_t=toks(SYSTEM); usr_t=int(np.mean([toks(user_msg(b)) for b in BL])); out_t=25
PRICE={ 'gpt-4.1-mini':(0.40,1.60),'gpt-4.1':(2.0,8.0),'gpt-5':(1.25,10.0),         # $/1M (in,out), OpenAI July 2026
        'gpt-5.4':(2.50,15.0),'gpt-5.5':(5.0,30.0),                                # GPT-5.4 / GPT-5.5 (flagship, Apr 2026)
        'claude-haiku-4-5':(0.80,4.0),'claude-sonnet-4-6':(3.0,15.0),'claude-opus-4-8':(15.0,75.0)}
print(f"=== token-cost estimate: {len(BL)} queries | system {sys_t} tok, user ~{usr_t} tok, output ~{out_t} tok ===")
print(f"    per query: {sys_t+usr_t} in + {out_t} out  |  total {len(BL)*(sys_t+usr_t)/1e3:.0f}K in + {len(BL)*out_t/1e3:.0f}K out")
for m,(pi,po) in PRICE.items():
    c=len(BL)*((sys_t+usr_t)*pi+out_t*po)/1e6; print(f"    {m:<20} ${c:.4f}   (full sweep of all {len(PRICE)} models ~= ${sum(len(BL)*((sys_t+usr_t)*a+out_t*b)/1e6 for a,b in PRICE.values()):.3f})")
if os.environ.get('DRYRUN'):
    print("\n[DRYRUN] sample prompt:\n--- SYSTEM ---\n"+SYSTEM+"\n--- USER ---\n"+user_msg(BL[0])); sys.exit(0)

# ---- render + score (needs API key) -------------------------------------------
from sentence_transformers import SentenceTransformer
sb=SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2'); PLE=sb.encode(LABELS,normalize_embeddings=True,show_progress_bar=False)
MODEL=os.environ.get('MODEL','gpt-4o-mini')
def render(b):
    msg=[{'role':'system','content':SYSTEM},{'role':'user','content':user_msg(b)}]
    if MODEL.startswith('claude'):
        import anthropic; cl=anthropic.Anthropic(); r=cl.messages.create(model=MODEL,max_tokens=60,system=SYSTEM,messages=[{'role':'user','content':user_msg(b)}]); return r.content[0].text.strip()
    import openai; cl=openai.OpenAI()
    kw={'model':MODEL,'messages':msg}; kw['max_completion_tokens' if ('gpt-5' in MODEL or 'o1' in MODEL) else 'max_tokens']=60
    return cl.chat.completions.create(**kw).choices[0].text if False else cl.chat.completions.create(**kw).choices[0].message.content.strip()
def qhat_from_nl(nl):
    e=sb.encode([nl],normalize_embeddings=True)[0]; sims=PLE@e; top=np.argsort(-sims)[:5]
    qh=(PU[top]*sims[top][:,None]).sum(0); qh=qh/(np.linalg.norm(qh)+1e-9); return qh,set(int(t) for t in top)
RT=[]; RCL=[]; CEIL=[]; EX=[]
for q,b in zip(QV,BL):
    qn=q/(np.linalg.norm(q)+1e-9)
    A=PU[b['idx'][:5]].T; w=np.linalg.lstsq(A,qn,rcond=None)[0]; qc=A@w; CEIL.append(float(qc@qn/(np.linalg.norm(qc)+1e-9)))   # OMP-5 ceiling
    try: nl=render(b)
    except Exception as e: print("render err",e); continue
    qh,top=qhat_from_nl(nl); RT.append(float(qh@qn)); RCL.append(len(set(b['idx'][:3])&top)/3.)
    if len(EX)<8: EX.append((nl,user_msg(b)))
print(f"\n=== ROUND-TRIP  MODEL={MODEL}  (n={len(RT)}) ===")
print(f"  OMP-5 ceiling (no language):   cos {np.mean(CEIL):.3f}")
print(f"  round-trip fidelity via NL:    cos {np.mean(RT):.3f}   (gap-from-ceiling = language bottleneck {np.mean(CEIL)-np.mean(RT):+.3f})")
print(f"  phrase-recovery recall@3:      {np.mean(RCL):.3f}")
print("  examples (NL <- blend):")
for nl,bl in EX: print(f"    Q: {nl}\n       <- {bl}")
