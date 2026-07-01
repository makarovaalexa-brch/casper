"""Build the UNIFIED menu the LLM asker chooses from: top-600 popular ML-1M movie TITLES + 761 concept tags.
Restricted to the ML catalog (no newer/obscure misses). MOVIE:/CONCEPT: prefixes so the LLM (and our mapper) know the type."""
import numpy as np, csv
from collections import Counter
base='C:/dev/phd/casper/data/movielens'; ml=f'{base}/ml-1m'
cnt=Counter()
for line in open(f'{ml}/ratings.dat'):
    a=line.split('::'); cnt[int(a[1])]+=1
top=[m for m,_ in cnt.most_common(600)]
title={}
for line in open(f'{ml}/movies.dat',encoding='latin-1'):
    a=line.strip().split('::'); title[int(a[0])]=a[1]
titles=[title[m] for m in top if m in title]
ctags=list(np.load(f'{base}/.cache/ctags_concept.npy'))
nm={}
for row in csv.reader(open(f'{base}/genome-tags.csv',encoding='utf-8')):
    if row and row[0].isdigit(): nm[int(row[0])]=row[1]
concepts=[nm.get(int(t),'?') for t in ctags]
with open(f'{base}/.cache/unified_menu.txt','w',encoding='utf-8') as f:
    f.write("# You may ask about any entry below. MOVIE = 'have you seen and liked this film?'  CONCEPT = 'do you like this kind of movie?'\n")
    for t in titles: f.write(f"MOVIE: {t}\n")
    for c in concepts: f.write(f"CONCEPT: {c}\n")
print(f"unified menu: {len(titles)} movies + {len(concepts)} concepts -> unified_menu.txt")
