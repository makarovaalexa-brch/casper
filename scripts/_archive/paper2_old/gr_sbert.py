"""
Goodreads MTC: SBERT content embeddings (all-MiniLM-L6-v2) over title + description, dense item order.
Saves item_sbert.npy (ni x 384, normalized). Content matrix (parity with CASPER SBERT pipeline).
Manual chunked encode with progress; short text (title + first ~200 desc chars) to bound seq length.
"""
import os, time, numpy as np
os.environ.setdefault('HF_HUB_OFFLINE','1'); os.environ.setdefault('TRANSFORMERS_OFFLINE','1')
import torch
torch.set_num_threads(os.cpu_count() or 8)
from sentence_transformers import SentenceTransformer
GR='C:/dev/phd/casper/.cache/goodreads'; t0=time.time()
M=np.load(f'{GR}/books_meta.npz',allow_pickle=True); titles=M['titles']; descs=M['descs']; ni=len(titles)
def txt(t,d):
    t=(t or '').strip(); d=(d or '').strip()
    return (f"{t}. {d[:200]}" if d else (t or "unknown"))
texts=[txt(titles[k],descs[k]) for k in range(ni)]
print(f"embedding {ni} items, threads={torch.get_num_threads()} ({time.time()-t0:.0f}s)",flush=True)
sb=SentenceTransformer('all-MiniLM-L6-v2')
CH=4096; parts=[]
for b0 in range(0,ni,CH):
    e=sb.encode(texts[b0:b0+CH],batch_size=128,normalize_embeddings=True,show_progress_bar=False)
    parts.append(e.astype(np.float32))
    print(f"  {min(b0+CH,ni)}/{ni} ({time.time()-t0:.0f}s)",flush=True)
E=np.concatenate(parts,0)
np.save(f'{GR}/item_sbert.npy',E)
print(f"SAVED item_sbert.npy shape={E.shape} ({time.time()-t0:.0f}s)",flush=True)
print("DONE",flush=True)
