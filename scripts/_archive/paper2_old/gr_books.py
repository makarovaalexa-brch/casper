"""
Goodreads MTC: stream books.gz ONCE. For catalogue items (dense-mapped via base.npz keepB):
  (a) collect title + description  -> books_meta.npz (for SBERT).
  (b) parse popular_shelves -> concept vocabulary (junk-filter, frequency floor, top ~1500 content
      shelves), build concept->item membership, and concept centroid directions Ac over Q (like ML-25M).
Saves books_meta.npz, concepts.npz (ctags names, citems_flat/off, Ac).
"""
import os, time, gzip, json, re, numpy as np
GR='C:/dev/phd/casper/.cache/goodreads'
BOOKS=f'{GR}/goodreads_books_mystery_thriller_crime.json.gz'
t0=time.time()
Bz=np.load(f'{GR}/base.npz'); keepB=Bz['keepB']; ni=int(Bz['ni'])
bid2idx={int(b):k for k,b in enumerate(keepB)}
Q=np.load(f'{GR}/Q_svd.npy'); D=Q.shape[1]
MINCOUNT_SHELF=5      # per-book: a shelf counts as membership if its count>=5
CONC_FLOOR=30         # a shelf is a concept if it tags >=30 catalogue items (mirror ML-25M >=30)
TOPC=1500             # keep top-1500 content shelves by document frequency

# junk / process shelves (exact) + substring rules
JUNK_EXACT=set("""to-read currently-reading owned favorites kindle library default wish-list to-buy
series dnf abandoned favourites read reads books ebook ebooks e-book audiobook audiobooks audio
audible my-books my-library owned-books books-i-own to-read-fiction general stuff all owned-tbr
tbr wishlist want-to-read have have-read re-read reread unfinished did-not-finish unread
i-own book-club giveaways giveaway netgalley arc arcs to-read-owned my-book shelfari-favorites
not-read finished maybe on-hold borrowed calibre kindle-unlimited kindle-books""".split())
JUNK_SUB=['to-read','currently','wish','to-buy','own','library','kindle','ebook','e-book','audio',
          'audible','my-','books-i','favorit','default','dnf','did-not-finish','abandon','unfinished',
          'unread','re-read','reread','tbr','want-to','have-read','not-read','net-galley','netgalley',
          'arc','giveaway','borrowed','on-hold','calibre','shelfari','read-in-','books-read',
          'reading-challenge','challenge','goodreads','number-','-star','star-','stars','rated','review']
YEAR=re.compile(r'^\d{2,4}$')
def is_junk(s):
    if s in JUNK_EXACT: return True
    if YEAR.match(s): return True
    if len(s)<=2: return True
    for sub in JUNK_SUB:
        if sub in s: return True
    return False

titles=[None]*ni; descs=[None]*ni
book_shelves=[None]*ni   # per catalogue item: list of shelf names (count>=MINCOUNT_SHELF, non-junk)
df={}                    # shelf -> #catalogue books that have it
nseen=0; nmatch=0
with gzip.open(BOOKS,'rt',encoding='utf-8') as f:
    for line in f:
        nseen+=1
        if nseen % 50000 == 0:
            print(f"  scanned {nseen} books, matched {nmatch}/{ni} ({time.time()-t0:.0f}s)",flush=True)
        try: d=json.loads(line)
        except Exception: continue
        try: bid=int(d['book_id'])
        except Exception: continue
        k=bid2idx.get(bid)
        if k is None: continue
        nmatch+=1
        titles[k]=(d.get('title') or '').strip()
        descs[k]=(d.get('description') or '').strip()
        sh=d.get('popular_shelves') or []
        names=[]
        for e in sh:
            try: c=int(e.get('count',0))
            except Exception: c=0
            nm=(e.get('name') or '').strip().lower()
            if c>=MINCOUNT_SHELF and nm and not is_junk(nm):
                names.append(nm)
        book_shelves[k]=names
        for nm in set(names): df[nm]=df.get(nm,0)+1
print(f"books scan done: matched {nmatch}/{ni} catalogue items ({time.time()-t0:.0f}s)",flush=True)

# ---- concept vocab: floor + top-TOPC by DF ----
cand=[(s,c) for s,c in df.items() if c>=CONC_FLOOR]
cand.sort(key=lambda x:-x[1])
cand=cand[:TOPC]
ctags=[s for s,_ in cand]; cset={s:k for k,s in enumerate(ctags)}; nc=len(ctags)
print(f"concept vocab: {nc} content shelves (>= {CONC_FLOOR} items, top {TOPC} by DF) ({time.time()-t0:.0f}s)",flush=True)
print("  DF range: max={} min(kept)={}".format(cand[0][1], cand[-1][1]),flush=True)
# membership concept -> item ids
citems=[[] for _ in range(nc)]
for k in range(ni):
    for nm in (book_shelves[k] or []):
        j=cset.get(nm)
        if j is not None: citems[j].append(k)
# dedup per concept
citems=[sorted(set(c)) for c in citems]
Ac=np.stack([Q[np.array(c)].mean(0) if c else np.zeros(D,np.float32) for c in citems]).astype(np.float32)
off=np.zeros(nc+1,np.int64)
for k in range(nc): off[k+1]=off[k]+len(citems[k])
flat=np.concatenate([np.array(c,np.int32) for c in citems]) if nc else np.zeros(0,np.int32)

# examples
print("  example concepts (name: #items):",flush=True)
ex_idx=list(range(0,min(nc,10)))+list(range(nc//2,nc//2+5))+list(range(max(0,nc-5),nc))
for j in ex_idx[:20]:
    print(f"    {ctags[j]}: {len(citems[j])}",flush=True)

np.savez(f'{GR}/concepts.npz', ctags=np.array(ctags,dtype=object), citems_flat=flat, citems_off=off, Ac=Ac,
         allow_pickle=True)
# titles/descs (object arrays)
np.savez(f'{GR}/books_meta.npz', titles=np.array(titles,dtype=object), descs=np.array(descs,dtype=object),
         allow_pickle=True)
missing=sum(1 for t in titles if not t)
covered=sum(1 for c in citems if c)
mean_items=np.mean([len(c) for c in citems])
print(f"coverage: {nmatch}/{ni} items have book meta ({missing} missing titles); "
      f"concepts nonempty={covered}/{nc}, mean items/concept={mean_items:.0f}",flush=True)
# fraction of catalogue items with >=1 concept
has_conc=sum(1 for k in range(ni) if any(cset.get(nm) is not None for nm in (book_shelves[k] or [])))
print(f"catalogue items with >=1 concept shelf: {has_conc}/{ni} ({has_conc/ni*100:.1f}%)",flush=True)
print("SAVED concepts.npz books_meta.npz",flush=True)
print("DONE",flush=True)
