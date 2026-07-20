"""
Pure-SBERT diagnostic: does raw SBERT (all-MiniLM-L6-v2) know the concept<->movie relationship from TEXT ALONE,
before any adapter W or collaborative space Q? Rank movies by cosine(SBERT(concept), SBERT(movie_text)) for
TITLE-ONLY and TITLE+GENRES. For each concept we name an EXPECTED film and report its rank.
If SBERT finds them -> failure is in the W/Q translation. If not -> the title text is insufficient (need synopsis).
"""
import numpy as np
from sentence_transformers import SentenceTransformer
base='C:/dev/phd/casper/data/movielens/ml-1m'
title=[]; gtext=[]
with open(f'{base}/movies.dat',encoding='latin-1') as f:
    for line in f:
        p=line.strip().split('::'); title.append(p[1]); gtext.append(p[2].replace('|',', '))
title=np.array(title); ni=len(title)
sb=SentenceTransformer('all-MiniLM-L6-v2')
T_to=sb.encode(list(title),batch_size=128,normalize_embeddings=True,show_progress_bar=False)
T_tg=sb.encode([f"{t}. Genres: {g}." for t,g in zip(title,gtext)],batch_size=128,normalize_embeddings=True,show_progress_bar=False)
def rank(emb,concept,expect):
    q=sb.encode([concept],normalize_embeddings=True)[0]; sc=emb@q; order=np.argsort(-sc)
    pos=next((r for r,i in enumerate(order) if any(e.lower() in title[i].lower() for e in expect)),None)
    return order[:6], pos
CONS=[("dinosaurs",["Jurassic","Lost World"]),
      ("time travel",["Back to the Future","Terminator","Twelve Monkeys","Groundhog"]),
      ("a shark attacking swimmers at the beach",["Jaws"]),
      ("talking farm animals",["Babe","Dolittle"]),
      ("boxing",["Rocky","Raging Bull"]),
      ("escape from prison",["Shawshank","Great Escape","Alcatraz"]),
      ("aliens visiting earth",["E.T.","Independence Day","Close Encounters","Men in Black","Alien"]),
      ("james bond spy",["GoldenEye","Russia with Love","Goldfinger","Dr. No","Tomorrow Never Dies","World Is Not Enough"]),
      ("a wedding",["Wedding"]),
      ("artificial intelligence",["2001","Terminator","Matrix","Ghost in the Shell","A.I."])]
for label,emb in [("TITLE-ONLY",T_to),("TITLE+GENRES",T_tg)]:
    print("\n"+"="*70+f"\nPURE SBERT cosine retrieval -- {label}\n"+"="*70)
    for c,exp in CONS:
        top,pos=rank(emb,c,exp)
        print(f"\n  \"{c}\"  (expect {exp[:3]}... ; first-hit rank: {pos if pos is not None else '>all'})")
        for i in top: print(f"     - {title[i]}")
