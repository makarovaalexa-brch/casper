"""Threaded TMDB credits fetch for ml-1m (resumes from credits_ml1m_actors5.json). 16 workers."""
import os, json, time, requests
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
try:
    from dotenv import load_dotenv; load_dotenv(Path(__file__).parent.parent.parent/'.env')
except Exception: pass

base=Path('C:/dev/phd/casper/data/movielens'); K=os.environ.get('TMDB_API_KEY'); assert K
OUT=base/'.cache'/'credits_ml1m_actors5.json'

mids=set(int(l.split('::')[1]) for l in open(base/'ml-1m'/'ratings.dat'))
m2t={}
with open(base/'links.csv') as f:
    next(f)
    for line in f:
        a=line.strip().split(',')
        if len(a)>=3 and a[2]:
            try: m2t[int(a[0])]=int(a[2])
            except: pass
ma={}; md={}
if OUT.exists():
    d=json.load(open(OUT)); ma={int(k):v for k,v in d.get('movie_actors',{}).items()}; md={int(k):v for k,v in d.get('movie_directors',{}).items()}
todo=[m for m in sorted(mids) if m not in ma and m in m2t]
print(f"have {len(ma)} | todo {len(todo)}",flush=True)

def fetch(m):
    try:
        r=requests.get(f"https://api.themoviedb.org/3/movie/{m2t[m]}/credits",params={'api_key':K},timeout=12)
        if r.status_code!=200: return m,[],[]
        c=r.json(); cast=sorted(c.get('cast',[]),key=lambda x:x.get('order',999))
        return m,[a['name'] for a in cast[:5]],[p['name'] for p in c.get('crew',[]) if p.get('job')=='Director']
    except Exception: return m,[],[]

done=0
with ThreadPoolExecutor(max_workers=16) as ex:
    futs=[ex.submit(fetch,m) for m in todo]
    for f in as_completed(futs):
        m,a,d=f.result(); ma[m]=a; md[m]=d; done+=1
        if done%200==0 or done==len(todo):
            allA=sorted({a for v in ma.values() for a in v}); allD=sorted({a for v in md.values() for a in v})
            json.dump({'movie_actors':{str(k):v for k,v in ma.items()},'movie_directors':{str(k):v for k,v in md.items()},
                       'all_actors':allA,'all_directors':allD},open(OUT,'w'))
            print(f"  {done}/{len(todo)} | {len(ma)} movies, {len(allA)} actors, {len(allD)} directors",flush=True)
print("==DONE==",flush=True)
