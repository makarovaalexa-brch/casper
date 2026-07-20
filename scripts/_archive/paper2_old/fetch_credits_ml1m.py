"""Paper D metadata phase: fetch top-5 actors + director(s) for ALL ml-1m movies via TMDB (links.csv movieId->tmdbId).
Resumable: merges into the existing credits cache. Run: python scripts/paper2/fetch_credits_ml1m.py"""
import os, json, time, requests
from pathlib import Path
try:
    from dotenv import load_dotenv; load_dotenv(Path(__file__).parent.parent.parent/'.env')
except Exception: pass

base=Path('C:/dev/phd/casper/data/movielens'); K=os.environ.get('TMDB_API_KEY')
assert K, "no TMDB_API_KEY"
OUT=base/'.cache'/'credits_ml1m_actors5.json'

# ml-1m movieIds
mids=set()
for line in open(base/'ml-1m'/'ratings.dat'):
    mids.add(int(line.split('::')[1]))
# links movieId->tmdbId
m2t={}
with open(base/'links.csv') as f:
    next(f)
    for line in f:
        a=line.strip().split(',')
        if len(a)>=3 and a[2]:
            try: m2t[int(a[0])]=int(a[2])
            except: pass

# resume from existing partial + seed from the top100 file
ma={}; md={}
if OUT.exists():
    d=json.load(open(OUT)); ma={int(k):v for k,v in d.get('movie_actors',{}).items()}; md={int(k):v for k,v in d.get('movie_directors',{}).items()}
seed=base/'.cache'/'credits_top100_actors5.json'
if seed.exists():
    d=json.load(open(seed))
    for k,v in d.get('movie_actors',{}).items():
        if int(k) in mids and int(k) not in ma: ma[int(k)]=v
    for k,v in d.get('movie_directors',{}).items():
        if int(k) in mids and int(k) not in md: md[int(k)]=v

todo=[m for m in sorted(mids) if m not in ma and m in m2t]
print(f"ml-1m movies {len(mids)} | have credits {len(ma)} | with tmdbId+todo {len(todo)} | no-tmdbId {len([m for m in mids if m not in m2t])}",flush=True)

def fetch(tid):
    try:
        r=requests.get(f"https://api.themoviedb.org/3/movie/{tid}/credits",params={'api_key':K},timeout=10)
        return r.json() if r.status_code==200 else None
    except Exception: return None

for i,m in enumerate(todo):
    c=fetch(m2t[m])
    if c:
        cast=sorted(c.get('cast',[]),key=lambda x:x.get('order',999))
        ma[m]=[a['name'] for a in cast[:5]]
        md[m]=[p['name'] for p in c.get('crew',[]) if p.get('job')=='Director']
    else:
        ma[m]=ma.get(m,[]); md[m]=md.get(m,[])
    if (i+1)%50==0 or i==len(todo)-1:
        allA=sorted({a for v in ma.values() for a in v}); allD=sorted({a for v in md.values() for a in v})
        json.dump({'movie_actors':{str(k):v for k,v in ma.items()},'movie_directors':{str(k):v for k,v in md.items()},
                   'all_actors':allA,'all_directors':allD},open(OUT,'w'))
        print(f"  {i+1}/{len(todo)} | {len(ma)} movies, {len(allA)} actors, {len(allD)} directors",flush=True)
    time.sleep(0.2)
print("==DONE==",flush=True)
