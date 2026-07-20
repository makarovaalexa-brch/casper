"""CORRECT concept-only cold-start eval for Phase-B checkpoints.
The in-script eval in set_mn.cmd_pb is nan: its users (umapV, the 3000 answerer-val ids) have ZERO overlap
with the SPLv cohort it pairs them against (an ID-space regression). This reuses the SAME split as the tree
(vhead's per-user permutation seed) so held targets and concept answers align on the same user rows.

Concept-only COLD: no item evidence; reveal k concepts (random), answered by the distilled answerer (refusal
burns the turn, level 14), fold as concept tokens (id = ni + cid, level 10+ordinal, separate knowledge emb),
decode, TAIL NDCG@10 vs the user's held-out liked items. Reports intercept (no concepts) + k=1,2,4,8,16,32.
Usage: python scripts/concept_eval.py <tag>_ep<N>   (checkpoint stem under .cache/set_mn/)
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch, torch.nn as nn
import set_mn as S
from set_mn import SetEncoder, concept_answers, RSD
from signed_latent import load_arena_base, ndcg10
import arena_core as AC

def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)
torch.set_num_threads(os.cpu_count())
META = "C:/dev/phd/casper/data/movielens/.cache/ml25m/meta.npz"
LO = 4.0
stem = sys.argv[1] if len(sys.argv) > 1 else "pbC_ep1"
GRADING = sys.argv[2] if len(sys.argv) > 2 else "ordinal"       # paord-based checkpoints are ordinal
S.set_grading(GRADING); NC = S.NC; NLEV = S.NLEV
log(f"grading={GRADING} NLEV={NLEV} refuse={S.LV_REFUSE} concept_offset={S.CLEVEL_OFFSET}")

base = load_arena_base(); ni = base["ni"]; head = base["headmask"]; headarr = np.where(head)[0]
d = np.load(META); uu = d["uu"].astype(np.int64); ii = d["ii"].astype(np.int64); rr = d["rr"].astype(np.float64)
o = np.argsort(uu, kind="stable"); uu, ii, rr = uu[o], ii[o], rr[o]
bnd = np.searchsorted(uu, np.arange(uu[-1] + 2))

# VAL answerer cohort (held out from the 150k train answerer) -- concept columns are [0:NC)
uidsV = np.load(RSD + "/mm_val_uids.npy"); KV = np.load(RSD + "/mm_val_know.npy"); VV = np.load(RSD + "/mm_val_val.npy")
log(f"val answerer users {len(uidsV)}  concept cols {NC}")

# per-user split IDENTICAL to vhead.build: evidence half excluded, held liked (>=LO) non-head = targets
recs = []
for r, uid in enumerate(uidsV):
    a, b = bnd[int(uid)], bnd[int(uid) + 1]
    its, rat = ii[a:b], rr[a:b]
    if len(its) < 8: continue
    ru = np.random.default_rng(AC.SEED * 1_000_003 + int(uid))
    p = ru.permutation(len(its)); h = len(its) // 2
    ki = its[p[:h]]                                   # evidence half -> excluded from ranking (profset)
    hi, hr = its[p[h:]], rat[p[h:]]
    hl = hi[hr >= LO]                                 # ALL held-liked (head+tail); ndcg10(tail=True) filters head
    if len(ki) < 4 or len(hl) == 0 or (~head[hl]).sum() == 0: continue   # need >=1 TAIL target too
    recs.append((r, set(int(x) for x in ki), hl))
log(f"usable eval users {len(recs)}")

ck = torch.load(f"C:/dev/phd/casper/.cache/set_mn/{stem}.pt", map_location="cpu")
NT = ck["student"]["item_emb.weight"].shape[0]
enc = SetEncoder(NT, token_mode="film", pool="attn", nlev=NLEV, nknow=0)   # paord has no know_emb; pool=attn FORCED
enc.load_state_dict(ck["student"], strict=False); enc.eval()
dec = nn.Linear(512, ni); dec.load_state_dict(ck["decoder"]); Wd = dec.weight.detach(); bd = dec.bias.detach()
log(f"loaded {stem}.pt  tokens {NT}  full={ck.get('full')} tail={ck.get('tail')}")

def decode(z): return (z @ Wd.T + bd).numpy().astype(np.float64)
rng = np.random.default_rng(7)

# intercept: empty set -> z0
with torch.no_grad():
    z0 = enc(torch.zeros((1, 1), dtype=torch.long), torch.zeros((1, 1)),
             torch.ones((1, 1), dtype=torch.bool), torch.zeros((1, 1), dtype=torch.long),
             torch.zeros((1, 1), dtype=torch.long))
# intercept
sc0 = decode(z0)[0]
ff_full = [ndcg10(sc0, list(hl), profset, head, False) for (_, profset, hl) in recs]
ff_tail = [ndcg10(sc0, list(hl), profset, head, True) for (_, profset, hl) in recs]
ff_full = [x for x in ff_full if x is not None]; ff_tail = [x for x in ff_tail if x is not None]
log(f"intercept (no concepts)  FULL={np.mean(ff_full):.4f}  TAIL={np.mean(ff_tail):.4f}   "
    f"(prior full baselines: intercept 0.2551, k32 0.304)")

for k in (1, 2, 4, 8, 16, 32):
    vf = []; vt = []
    for b in range(0, len(recs), 256):
        ch = recs[b:b + 256]; B = len(ch)
        ids = np.zeros((B, k), np.int64); lv = np.zeros((B, k), np.int64); kk = np.zeros((B, k), np.int64)
        pad = np.zeros((B, k), bool)
        for r, (row, profset, hl) in enumerate(ch):
            cids = rng.choice(NC, size=k, replace=False)
            a, c = concept_answers(row, KV, VV, cids)
            ids[r] = cids + ni; lv[r] = a; kk[r] = c
            pad[r] = (a == S.LV_REFUSE)                          # DROP refused (padded out); still counts toward k
        with torch.no_grad():
            z = enc(torch.from_numpy(ids), torch.zeros((B, k)), torch.from_numpy(pad),
                    torch.from_numpy(lv), torch.from_numpy(kk))
        sc = decode(z)
        for r, (row, profset, hl) in enumerate(ch):
            a = ndcg10(sc[r], list(hl), profset, head, False); b2 = ndcg10(sc[r], list(hl), profset, head, True)
            if a is not None: vf.append(a)
            if b2 is not None: vt.append(b2)
    log(f"k={k:2d} concepts  FULL={np.mean(vf):.4f}  TAIL={np.mean(vt):.4f}")
