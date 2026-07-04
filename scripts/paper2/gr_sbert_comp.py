"""
Goodreads COMPOSITE: SBERT content embeddings (all-MiniLM-L6-v2). REUSE the mystery embeddings
(item_sbert.npy aligned to base.npz keepB) for any composite item whose book_id was in the mystery
catalogue; embed only the NEW items (fantasy/history) from books_meta_comp title+desc. Same text
recipe (title + first 200 desc chars, normalized). Saves item_sbert_comp.npy (ni x 384).
"""
import os, time, numpy as np
os.environ.setdefault('HF_HUB_OFFLINE','1'); os.environ.setdefault('TRANSFORMERS_OFFLINE','1')
import torch; torch.set_num_threads(os.cpu_count() or 8)
from sentence_transformers import SentenceTransformer
GR='C:/dev/phd/casper/.cache/goodreads'; t0=time.time()
Bc=np.load(f'{GR}/base_comp.npz'); keepB_c=Bc['keepB'].astype(np.int64); ni=len(keepB_c)
# old mystery embeddings + book-id map
oldB=np.load(f'{GR}/base.npz')['keepB'].astype(np.int64); oldE=np.load(f'{GR}/item_sbert.npy')
oldmap={int(b):r for r,b in enumerate(oldB)}
dim=oldE.shape[1]
E=np.zeros((ni,dim),np.float32); reuse=np.zeros(ni,bool)
for k in range(ni):
    r=oldmap.get(int(keepB_c[k]))
    if r is not None: E[k]=oldE[r]; reuse[k]=True
nnew=int((~reuse).sum())
print(f"reuse {int(reuse.sum())}/{ni} mystery embeddings; embed {nnew} new items ({time.time()-t0:.0f}s)",flush=True)
M=np.load(f'{GR}/books_meta_comp.npz',allow_pickle=True); titles=M['titles']; descs=M['descs']
def txt(t,d):
    t=(t or '').strip(); d=(d or '').strip()
    return (f"{t}. {d[:200]}" if d else (t or "unknown"))
new_idx=np.nonzero(~reuse)[0]
texts=[txt(titles[k],descs[k]) for k in new_idx]
# RESUMABLE: partial progress in item_sbert_comp_part.npz (Enew rows + done count)
PART=f'{GR}/item_sbert_comp_part.npz'
Enew=np.zeros((len(new_idx),dim),np.float32); done=0
if os.path.exists(PART):
    P=np.load(PART); Enew=P['Enew']; done=int(P['done'])
    print(f"[RESUME] {done}/{len(new_idx)} new items already embedded",flush=True)
sb=SentenceTransformer('all-MiniLM-L6-v2'); CH=4096
BUDGET=float(os.environ.get('BUDGET_S',0))  # 0 = run to completion
for b0 in range(done,len(texts),CH):
    e=sb.encode(texts[b0:b0+CH],batch_size=128,normalize_embeddings=True,show_progress_bar=False)
    Enew[b0:b0+len(e)]=e.astype(np.float32); done=b0+len(e)
    np.savez(PART,Enew=Enew,done=done)
    print(f"  embedded {done}/{len(texts)} new ({time.time()-t0:.0f}s)",flush=True)
    if BUDGET>0 and time.time()-t0>BUDGET:
        print(f"[BUDGET] stopping at {done}/{len(texts)}; rerun to resume",flush=True); raise SystemExit
for i,k in enumerate(new_idx): E[k]=Enew[i]
np.save(f'{GR}/item_sbert_comp.npy',E)
if os.path.exists(PART): os.remove(PART)
print(f"SAVED item_sbert_comp.npy shape={E.shape} (reused {int(reuse.sum())}, new {nnew}) ({time.time()-t0:.0f}s)\nDONE",flush=True)
