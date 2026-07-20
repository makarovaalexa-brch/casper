"""Diagnostics on the trained fusion recommender:
1. k-curve: FULL + TAIL NDCG@10 vs number of folded concepts, for three selection strategies --
   RANDOM (with refusals folded), REAL (from the user's own non-refused concepts), ORACLE (best-individual top-k).
2. Paper-B sign-flip SPECIFICITY: flip one concept love<->hate; the ranking change must concentrate on the
   concept's MEMBER films, not unrelated items (targeted negation, not global).
Leak-free: concepts from the KNOWN half, targets = HELD half (uid-seeded split).
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch, torch.nn as nn
import set_mn as S; S.set_grading("ordinal")
from set_mn import SetEncoder, RSD
from signed_latent import load_arena_base, ndcg10
from reconciled import ConceptBank
import arena_core as AC
def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)
torch.set_num_threads(os.cpu_count())
META = "C:/dev/phd/casper/data/movielens/.cache/ml25m/meta.npz"; LO = 4.0; NC = S.NC; REFUSE = S.LV_REFUSE
CK = sys.argv[1] if len(sys.argv) > 1 else "fusj_ep5"; NU = int(os.environ.get("NU", "1500")); NUO = int(os.environ.get("NUO", "150"))

base = load_arena_base(); ni = base["ni"]; head = base["headmask"]
d = np.load(META); uu = d["uu"].astype(np.int64); ii = d["ii"].astype(np.int64); rr = d["rr"].astype(np.float64)
o = np.argsort(uu, kind="stable"); uu, ii, rr = uu[o], ii[o], rr[o]; bnd = np.searchsorted(uu, np.arange(uu[-1] + 2))
uids = np.load(RSD + "/mm_val_uids.npy"); K = np.load(RSD + "/mm_val_know.npy"); V = np.load(RSD + "/mm_val_val.npy")
ANS = (K[:, :NC] >= 1) & (V[:, :NC] >= 0); CVAL = np.clip(V[:, :NC], 0, 3).astype(np.int64); CKN = np.clip(K[:, :NC], 0, 2).astype(np.int64)
recs = []
for r, uid in enumerate(uids):
    a, b = bnd[int(uid)], bnd[int(uid) + 1]; its, rat = ii[a:b], rr[a:b]
    if len(its) < 8: continue
    ru = np.random.default_rng(AC.SEED * 1_000_003 + int(uid)); p = ru.permutation(len(its)); h = len(its) // 2
    ki = its[p[:h]]; hi, hr = its[p[h:]], rat[p[h:]]; hl = hi[hr >= LO]
    if len(ki) < 4 or len(hl) == 0: continue
    recs.append((r, hl.astype(np.int64)))
log(f"val eval users {len(recs)}  ckpt {CK}")

ck = torch.load(f".cache/set_mn/{CK}.pt", map_location="cpu")
enc = SetEncoder(ni + NC, token_mode="xattn", pool="attn", nlev=S.NLEV, nknow=3); enc.load_state_dict(ck["student"]); enc.eval()
dec = nn.Linear(512, ni); dec.load_state_dict(ck["decoder"]); Wd = dec.weight.detach(); bd = dec.bias.detach()
cb = ConceptBank(ni, base["cnt"]); M = cb.Mw.tocsr() if hasattr(cb.Mw, "tocsr") else cb.Mw
def decode(z): return (z @ Wd.T + bd).numpy().astype(np.float64)
def fold(ids, lv, kn):
    with torch.no_grad():
        z = enc(torch.tensor([ids]), torch.zeros(1, len(ids)), torch.zeros(1, len(ids), dtype=torch.bool), torch.tensor([lv]), torch.tensor([kn]))
    return decode(z)[0]
with torch.no_grad():
    z0 = decode(enc(torch.zeros(1,1,dtype=torch.long),torch.zeros(1,1),torch.ones(1,1,dtype=torch.bool),torch.zeros(1,1,dtype=torch.long),torch.zeros(1,1,dtype=torch.long)))[0]

def ndcg_both(sc, hl):
    return ndcg10(sc, list(hl), set(), head, False), ndcg10(sc, list(hl), set(), head, True)

# ---- k-curve ----
KS = [0, 1, 2, 4, 8, 16, 32]
rng = np.random.default_rng(7)
print(f"\n{'strategy':<8} {'k':>3} {'FULL':>8} {'TAIL':>8}")
def toks_for(r, cids):
    ids = []; lv = []; kn = []
    for c in cids:
        if ANS[r, c]: ids.append(ni+int(c)); lv.append(int(CVAL[r,c])); kn.append(int(CKN[r,c]))
        else: ids.append(ni+int(c)); lv.append(REFUSE); kn.append(0)      # refusal FOLDED as signal
    return ids, lv, kn
for strat in ["random", "real"]:
    for k in KS:
        ff = []; tt = []
        for r, hl in recs[:NU]:
            if k == 0: sc = z0
            else:
                if strat == "random": cids = rng.choice(NC, size=k, replace=False)
                else:                                                     # REAL: from the user's non-refused concepts
                    nz = np.where(ANS[r])[0]
                    if len(nz) == 0: continue
                    cids = rng.choice(nz, size=min(k, len(nz)), replace=False)
                ids, lv, kn = toks_for(r, cids); sc = fold(ids, lv, kn)
            a, b = ndcg_both(sc, hl)
            if a is not None: ff.append(a)
            if b is not None: tt.append(b)
        print(f"{strat:<8} {k:>3} {(np.mean(ff) if ff else float('nan')):>8.4f} {(np.mean(tt) if tt else float('nan')):>8.4f}")

# ---- ORACLE (best-individual top-k), smaller sample ----
log(f"oracle on {NUO} users (rank all {NC} concepts by single-fold TAIL, take top-k) ...")
for k in [1, 2, 4, 8, 16, 32]:
    ff = []; tt = []
    for r, hl in recs[:NUO]:
        singles = np.full(NC, -1.0)
        # batch single-concept folds
        for b0 in range(0, NC, 512):
            cc = np.arange(b0, min(b0+512, NC))
            B = len(cc); ids = (ni+cc)[:,None]; lv = np.where(ANS[r,cc], CVAL[r,cc], REFUSE)[:,None]; kn = np.where(ANS[r,cc], CKN[r,cc], 0)[:,None]
            with torch.no_grad():
                z = enc(torch.from_numpy(ids), torch.zeros(B,1), torch.zeros(B,1,dtype=torch.bool), torch.from_numpy(lv), torch.from_numpy(kn))
            sc = decode(z)
            for j, c in enumerate(cc):
                v = ndcg10(sc[j], list(hl), set(), head, True); singles[c] = v if v is not None else -1
        top = np.argsort(-singles)[:k]
        ids, lv, kn = toks_for(r, top); sc = fold(ids, lv, kn)
        a, b = ndcg_both(sc, hl)
        if a is not None: ff.append(a)
        if b is not None: tt.append(b)
    print(f"{'oracle':<8} {k:>3} {(np.mean(ff) if ff else float('nan')):>8.4f} {(np.mean(tt) if tt else float('nan')):>8.4f}")

# ---- Paper-B SIGN-FLIP SPECIFICITY ----
import json
tagnames = [t.get('tag','?') for t in json.load(open('.cache/instrument2/tag_questions.json'))['tags']]
print("\n=== SIGN-FLIP SPECIFICITY (fold one concept LOVED vs HATED; |score change| on MEMBERS vs UNRELATED) ===")
allitems = np.arange(ni)
for cn in ['horror','animation','sci fi','romance','documentary']:
    c = tagnames.index(cn) if cn in tagnames else -1
    if c < 0: continue
    mem = np.array(M[c].indices)
    non = np.setdiff1d(allitems, mem)
    non = np.random.default_rng(0).choice(non, min(2000, len(non)), replace=False)
    sl = fold([ni+c], [3], [2]); sh = fold([ni+c], [0], [2])
    dmem = np.abs(sl[mem]-sh[mem]).mean(); dnon = np.abs(sl[non]-sh[non]).mean()
    print(f"  {cn:<12} |love-hate| on members {dmem:.3f}  on unrelated {dnon:.3f}  ratio {dmem/max(dnon,1e-9):.1f}x  (want >>1)")
log("done")
