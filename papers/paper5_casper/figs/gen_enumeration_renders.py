import os, json, numpy as np
base='C:/dev/phd/casper/data/movielens'
for _l in open('C:/dev/phd/casper/.env'):
    if _l.startswith('OPENAI_API_KEY='): os.environ['OPENAI_API_KEY']=_l.split('=',1)[1].strip().strip('"')
import openai
qd=np.load(f'{base}/.cache/qdump.npz',allow_pickle=True); QV=qd['q'].astype(np.float32)
pb=np.load(f'{base}/.cache/phrasebank.npz',allow_pickle=True); PU=pb['unit'].astype(np.float32); PL=list(pb['label'])
# NAIVE enumeration prompt: literally list every descriptor, no axis inference (the failure mode)
SYS=("You convert a movie-preference query into ONE question for a new user. The query is a signed blend of descriptors "
"weight*[descriptor]. Mention EVERY descriptor explicitly in the question, in order; do not abstract or infer a single "
"underlying theme. Output only the question, one sentence.")
cl=openai.OpenAI(); MODEL='gpt-4o-mini'
def omp(q,K):
    ch=[]; res=q.copy(); w=None
    for _ in range(K):
        pr=PU@res
        if ch: pr[ch]=0.
        i=int(np.argmax(np.abs(pr))); ch.append(i); A=PU[ch].T; w=np.linalg.lstsq(A,q,rcond=None)[0]; res=q-A@w
    return ch,w
out={}
for i in [2,5,7,10,0,1]:
    ch,w=omp(QV[i],3); um="Query: "+"  ".join(f"{wi:+.2f}*[{PL[c]}]" for c,wi in zip(ch,w))
    r=cl.chat.completions.create(model=MODEL,messages=[{'role':'system','content':SYS},{'role':'user','content':um}],max_tokens=60)
    nl=r.choices[0].message.content.strip(); out[um]=nl; print(f"[{i}] {um}\n   -> {nl}\n")
json.dump(out,open('C:/dev/phd/papers/paper5_casper/figs/enumeration_renders.json','w',encoding='utf-8'),indent=1,ensure_ascii=False)
