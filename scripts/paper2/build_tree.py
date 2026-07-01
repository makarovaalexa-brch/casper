"""Build the modal decision tree of the entdistill winner from its real trajectory dump (fixes the stale fig:tree).
Each path = [[concept_idx, +-1], ...]. Print opener + most-asked concepts + a depth-3 modal branch tree."""
import json, csv, numpy as np
from collections import Counter
base='C:/dev/phd/casper/data/movielens'
ctags=list(np.load(f'{base}/.cache/ctags_concept.npy'))
nm={}
for row in csv.reader(open(f'{base}/genome-tags.csv',encoding='utf-8')):
    if row and row[0].isdigit(): nm[int(row[0])]=row[1]
cn=lambda cc: nm.get(int(ctags[cc]),'?')
T=[t for t in json.load(open(f'{base}/.cache/tree_entdistill_ep4.json')) if t]
print(f"{len(T)} user paths | distinct trajectories {len(set(tuple((c,a) for c,a in p) for p in T))} | vocab {len(set(c for p in T for c,a in p))}")
allc=Counter(c for p in T for c,a in p)
print("most-asked concepts:", [(cn(c),n) for c,n in allc.most_common(8)])
def branch(paths, depth, pre=''):
    if not paths or depth==0: return
    cc=Counter(p[0][0] for p in paths).most_common(1)[0][0]
    here=[p for p in paths if p[0][0]==cc]; lik=[p[1:] for p in here if p[0][1]>0]; dis=[p[1:] for p in here if p[0][1]<0]
    print(f'{pre}Q "{cn(cc)}?"  (asked by {len(here)})')
    if depth>1:
        print(f'{pre}|- LIKE ({len(lik)})'); branch([p for p in lik if p], depth-1, pre+'|    ')
        print(f'{pre}+- DISLIKE ({len(dis)})'); branch([p for p in dis if p], depth-1, pre+'     ')
print("\n--- modal decision tree (depth 4) ---")
branch(T, 4)
