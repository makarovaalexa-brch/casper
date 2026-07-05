import os, json, numpy as np
base='C:/dev/phd/casper/data/movielens'
for _l in open('C:/dev/phd/casper/.env'):
    if _l.startswith('OPENAI_API_KEY='): os.environ['OPENAI_API_KEY']=_l.split('=',1)[1].strip().strip('"')
import openai
d=np.load(f'{base}/.cache/qviz.npz',allow_pickle=True); QS=d['questions']
qd=np.load(f'{base}/.cache/qdump.npz',allow_pickle=True); QV=qd['q'].astype(np.float32)
pb=np.load(f'{base}/.cache/phrasebank.npz',allow_pickle=True); PU=pb['unit'].astype(np.float32); PL=list(pb['label'])
SYS=("You turn a movie-preference 'elicitation query' into ONE short natural question a recommender asks a new user. "
"The query is a DIRECTION in taste space: a signed blend of descriptors weight*[descriptor]. Sign=which side (positive=TOWARD, negative=AWAY); magnitude=importance. "
"The descriptors are almost always facets of ONE underlying taste axis. Infer that single axis, ask one conversational question about it from the pole the signs indicate, and do NOT enumerate the descriptors. "
"If they clearly share no axis, ask only about the single largest-magnitude one. Output only the question, one sentence.")
cl=openai.OpenAI(); MODEL='gpt-4o-mini'
def omp(q,K):
    ch=[]; res=q.copy(); w=None
    for _ in range(K):
        pr=PU@res
        if ch: pr[ch]=0.
        i=int(np.argmax(np.abs(pr))); ch.append(i); A=PU[ch].T; w=np.linalg.lstsq(A,q,rcond=None)[0]; res=q-A@w
    return ch,w
def render(q,K=3):
    ch,w=omp(q,K); um="Query: "+"  ".join(f"{wi:+.2f}*[{PL[c]}]" for c,wi in zip(ch,w))
    r=cl.chat.completions.create(model=MODEL,messages=[{'role':'system','content':SYS},{'role':'user','content':um}],max_tokens=60)
    return um, r.choices[0].message.content.strip(), ch, w
out={'interview_user':12,'turns':[]}
u=12
for t in range(8):
    um,nl,ch,w=render(QS[u*8+t])
    blend=[[float(wi),PL[c]] for c,wi in zip(ch,w)]
    out['turns'].append({'t':t,'blend':blend,'nl':nl})
    print(f"t{t}: {nl}")
# worked example: qdump idx 10 (Tim Burton)
um,nl,ch,w=render(QV[10])
out['worked_example']={'idx':10,'blend':[[float(wi),PL[c]] for c,wi in zip(ch,w)],'nl':nl}
print("WORKED:",nl)
json.dump(out,open('C:/dev/phd/papers/paper3_casper/figs/interview_renders.json','w',encoding='utf-8'),indent=1,ensure_ascii=False)
print("saved")
