"""#13 opener story: which concept does the learned policy open with, and where does it rank on
each single heuristic? Claim to check: opener is #1 on NO single heuristic but top-1% on two at once.
Standalone: reads the fresh tree dump + cached heuristic vectors (no harness import)."""
import json, csv, numpy as np
base='C:/dev/phd/casper/data/movielens'
ctags=list(np.load(f'{base}/.cache/ctags_concept.npy')); NC=len(ctags); NI=600
nm={}
for row in csv.reader(open(f'{base}/genome-tags.csv',encoding='utf-8')):
    if row and row[0].isdigit(): nm[int(row[0])]=row[1]
cn=lambda ci: nm.get(int(ctags[ci]),'?')
tree=json.load(open(f'{base}/.cache/tree_entdistill_ep4.json'))
from collections import Counter
openers=[p[0][0] for p in tree if p]
oc=Counter(openers); op,c=oc.most_common(1)[0]
print(f"OPENER: concept idx {op} = '{cn(op)}'  ({c}/{len(openers)} = {100*c/len(openers):.0f}% of users)\n")
ENT=np.load(f'{base}/.cache/pool_entavg.npy')[0][NI:]                         # raw divisiveness (binary entropy)
IG=np.load(f'{base}/.cache/pool_ig2.npy')[NI:]                                # population info-gain
catalog=set(int(l.split('::')[1]) for l in open(f'{base}/ml-1m/ratings.dat'))
tagset=set(int(t) for t in ctags); cfm={}
f=open(f'{base}/genome-scores.csv'); next(f)
for line in f:
    a=line.split(',')
    if int(a[0]) in catalog and int(a[1]) in tagset and float(a[2])>0.5: cfm[int(a[1])]=cfm.get(int(a[1]),0)+1
COV=np.array([cfm.get(int(t),0) for t in ctags],float)
def pct(vec,ci): return 100.*(vec<vec[ci]).mean()
def rank(vec,ci): return int((vec>vec[ci]).sum())+1
for name,vec in [('divisiveness',ENT),('info-gain',IG),('coverage',COV)]:
    print(f"  {name:13}: opener pctile {pct(vec,op):5.1f}%  rank {rank(vec,op):3d}/{NC}  | argmax='{cn(int(vec.argmax()))}'")
print()
for name,vec in [('divisiveness',ENT),('info-gain',IG),('coverage',COV)]:
    print(f"  top-5 {name:13}: {[cn(int(t)) for t in np.argsort(-vec)[:5]]}")
# dump concept vocabulary for the LLM-asker subagents (#17)
open(f'{base}/.cache/concept_vocab.txt','w',encoding='utf-8').write('\n'.join(cn(c) for c in range(NC)))
print(f"\nwrote concept_vocab.txt ({NC} concepts)")
