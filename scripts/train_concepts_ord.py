"""Phase B ORDINAL - learn concept embeddings on the FROZEN ordinal recommender (paord).
Signed recipe (DESIGN_SHEET_PHASE_B_ORDINAL.md, 2026-07-15):
  - FREEZE the recommender (item_emb[:ni], FiLM gamma/beta, attention, decoder). TRAIN ONLY concept rows.
  - avoid the AdamW-decay trap: optimizer over item_emb.weight ONLY, weight_decay=0, and zero grad[:ni] each
    step so item rows never move (verified bit-identical after step 1 -> full-profile f is a CONSTANT canary).
  - MIXED curriculum: 50% fold ALL non-refused concepts / 50% fold k~log-uniform[1,64]. Concepts only, NO items.
  - refusals DROPPED (padded out). target = HELD-half liked items (same uid-seeded split the answerer used).
  - polarity is the shared item FiLM (unified ordinal grading).
Pre-run controls run before epoch 0 (evidence/target disjoint, wrong-user permutation, polarity probe, freeze).
Per-epoch: save a standard SetEncoder(NT) checkpoint (concept rows in item_emb[ni:]) + concept_eval.
"""
import os, sys, time, json, subprocess
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
try: sys.stdout.reconfigure(encoding='utf-8')
except Exception: pass
import set_mn as S
from set_mn import SetEncoder, RSD
from signed_latent import load_arena_base
import arena_core as AC

def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)
torch.set_num_threads(os.cpu_count())
S.set_grading("ordinal"); NC = S.NC; NLEV = S.NLEV; REFUSE = S.LV_REFUSE
META = "C:/dev/phd/casper/data/movielens/.cache/ml25m/meta.npz"
OUT = "C:/dev/phd/casper/.cache/set_mn"
LO = 4.0; D = 512
BASE = os.environ.get("BASE", "paord_best")
TAG = os.environ.get("TAG", "pbord")
EPOCHS = int(os.environ.get("EPOCHS", "12"))
log(f"grading=ordinal NLEV={NLEV} refuse={REFUSE} | base={BASE} tag={TAG}")

base = load_arena_base(); ni = base["ni"]; head = base["headmask"]; headarr = np.where(head)[0]
d = np.load(META); uu = d["uu"].astype(np.int64); ii = d["ii"].astype(np.int64); rr = d["rr"].astype(np.float64)
o = np.argsort(uu, kind="stable"); uu, ii, rr = uu[o], ii[o], rr[o]
bnd = np.searchsorted(uu, np.arange(uu[-1] + 2))

# ---- train answerer cohort: concept columns [0:NC), reconstruct held-half target (SAME split concept_eval uses)
uidsT = np.load(RSD + "/mm_train_uids.npy"); KT = np.load(RSD + "/mm_train_know.npy"); VT = np.load(RSD + "/mm_train_val.npy")
log(f"train answerer users {len(uidsT)}")
ANS = (KT[:, :NC] >= 1) & (VT[:, :NC] >= 0)                    # (Ntr, NC) non-refused mask
LEV = (S.CLEVEL_OFFSET + np.clip(VT[:, :NC], 0, 3)).astype(np.int64)   # ordinal level 0..3 (offset 0)
rows = []; TGT = []
for r, uid in enumerate(uidsT):
    a, b = bnd[int(uid)], bnd[int(uid) + 1]
    its, rat = ii[a:b], rr[a:b]
    if len(its) < 8: continue
    ru = np.random.default_rng(AC.SEED * 1_000_003 + int(uid))
    p = ru.permutation(len(its)); h = len(its) // 2
    ki = its[p[:h]]; hi, hr = its[p[h:]], rat[p[h:]]
    hl = hi[hr >= LO]                                          # held-half likes (head+tail) = TARGET
    if len(ki) < 4 or len(hl) == 0 or not ANS[r].any(): continue
    rows.append(r); TGT.append(hl.astype(np.int64))
rows = np.array(rows, np.int64)
log(f"usable train users {len(rows)} (>=1 non-refused concept AND >=1 held-like)")

# ---- model: SetEncoder(NT), warm-start recommender from paord, concept rows from member-bags ----
NT = ni + NC
enc = SetEncoder(NT, token_mode="film", pool="attn", nlev=NLEV, nknow=0)
pa = torch.load(os.path.join(OUT, BASE + ".pt"), map_location="cpu")
pas = pa["student"]
with torch.no_grad():
    enc.item_emb.weight[:ni].copy_(pas["item_emb.weight"])                         # frozen item rows
    sd = {k: v for k, v in pas.items() if not k.startswith("item_emb")}
    missing, unexpected = enc.load_state_dict(sd, strict=False)                      # gamma/beta/attention/head/z0
    from reconciled import ConceptBank
    cb = ConceptBank(ni, base["cnt"]); M = cb.Mw.tocsr() if hasattr(cb.Mw, "tocsr") else cb.Mw
    rs = np.asarray(M.sum(1)).ravel(); rs[rs == 0] = 1.0
    E = (M @ pas["item_emb.weight"].numpy()) / rs[:, None]                           # pop-weighted member-bag mean
    enc.item_emb.weight[ni:ni + NC].copy_(torch.from_numpy(E).float())
decoder = nn.Linear(D, ni); decoder.load_state_dict(pa["decoder"])
Wd = decoder.weight.detach(); bd = decoder.bias.detach()
log(f"warm-started from {BASE} (paord full={pa.get('full'):.4f} tail={pa.get('tail'):.4f}); "
    f"concept rows = member-bag; unexpected keys skipped={len(unexpected)}")

# ---- FREEZE everything; train ONLY concept rows via item_emb.weight (wd=0 + grad-mask [:ni]) ----
for p_ in enc.parameters(): p_.requires_grad_(False)
for p_ in decoder.parameters(): p_.requires_grad_(False)
enc.item_emb.weight.requires_grad_(True)
opt = torch.optim.AdamW([enc.item_emb.weight], lr=3e-4, weight_decay=0.0)           # wd=0 => no decay of frozen rows
ITEM_SNAPSHOT = enc.item_emb.weight[:ni].detach().clone()                           # canary reference

def fold_batch(bidx, rng):
    """build concept-only tokens for a batch of usable-user indices; mixed fold-all / k-sample; refusals dropped."""
    seqs = []
    for u in bidx:
        r = rows[u]; nz = np.where(ANS[r])[0]                                        # non-refused concept cols
        if rng.random() < 0.5:
            sel = nz                                                                 # fold ALL non-refused
        else:
            k = int(np.exp(rng.uniform(0, np.log(64))))                              # k ~ log-uniform[1,64]
            sel = rng.choice(nz, size=min(k, len(nz)), replace=False)
        seqs.append((ni + sel, LEV[r, sel]))
    Lmax = max(len(s[0]) for s in seqs)
    B = len(seqs)
    ids = np.zeros((B, Lmax), np.int64); lv = np.zeros((B, Lmax), np.int64); pad = np.ones((B, Lmax), bool)
    for i, (cid, cl) in enumerate(seqs):
        ids[i, :len(cid)] = cid; lv[i, :len(cid)] = cl; pad[i, :len(cid)] = False
    return ids, lv, pad

def encode(ids, lv, pad):
    return enc(torch.from_numpy(ids), torch.zeros(ids.shape), torch.from_numpy(pad), torch.from_numpy(lv))

# ================= PRE-RUN CONTROLS (signed) =================
log("=== pre-run controls ===")
# (1) evidence disjoint from target: answers come from KNOWN half, targets are HELD half -> disjoint by split.
bad_disjoint = 0
for i in range(min(2000, len(rows))):
    u = rows[i]; uid = uidsT[u]; a, b = bnd[int(uid)], bnd[int(uid) + 1]; its = ii[a:b]
    ru = np.random.default_rng(AC.SEED * 1_000_003 + int(uid)); p = ru.permutation(len(its)); h = len(its) // 2
    known = set(int(x) for x in its[p[:h]])
    if known & set(int(x) for x in TGT[i]): bad_disjoint += 1
log(f"(1) evidence-INT-target disjoint: {bad_disjoint}/2000 users with overlap (must be 0)")
assert bad_disjoint == 0, "TARGET LEAK: held-half target overlaps known half"

# (3) polarity probe: fold ONE concept at level 3 (loved) vs level 0 (hated); member-item scores must move oppositely
with torch.no_grad():
    cprobe = int(np.argmax(np.asarray(M.sum(1)).ravel()))                            # a big-membership concept
    members = M[cprobe].indices[:50]
    def one(levent):
        z = enc(torch.tensor([[ni + cprobe]]), torch.zeros(1, 1), torch.zeros(1, 1, dtype=torch.bool),
                torch.tensor([[levent]]))
        return (z @ Wd.T + bd).numpy()[0]
    s_love = one(3); s_hate = one(0)
    dmembers = float((s_love[members] - s_hate[members]).mean())
log(f"(3) polarity probe concept {cprobe}: mean(member score love-hate) = {dmembers:+.4f} (should be >0 = loved lifts members)")

# (4) freeze canary: one training step, assert item rows [:ni] bit-identical
rng = np.random.default_rng(0)
ids, lv, pad = fold_batch(np.arange(min(64, len(rows))), rng)
z = encode(ids, lv, pad); logits = z @ Wd.T + bd
tgt0 = torch.zeros(len(ids), ni)
for i in range(len(ids)):
    tgt0[i, TGT[i]] = 1.0
nll = -((F.log_softmax(logits, -1) * tgt0).sum(-1) / tgt0.sum(-1).clamp_min(1)).mean()
opt.zero_grad(); nll.backward(); enc.item_emb.weight.grad[:ni] = 0.0; opt.step()
drift = float((enc.item_emb.weight[:ni] - ITEM_SNAPSHOT).abs().max())
log(f"(4) freeze canary: max|item_emb[:ni] drift| after 1 step = {drift:.2e} (must be 0)")
assert drift == 0.0, "FREEZE BROKEN: item rows moved (AdamW-decay trap?)"
# restore concept rows (undo the canary step) by re-warm-starting concepts
with torch.no_grad():
    enc.item_emb.weight[ni:ni + NC].copy_(torch.from_numpy(E).float())
opt = torch.optim.AdamW([enc.item_emb.weight], lr=3e-4, weight_decay=0.0)
log("controls PASSED; (2) permutation control is run by concept_eval separately.")

# ================= TRAIN =================
def save_and_eval(ep, f_canary):
    ckpt = {"student": enc.state_dict(), "decoder": decoder.state_dict(), "epoch": ep,
            "full": f_canary, "tail": None, "grading": "ordinal"}
    torch.save(ckpt, os.path.join(OUT, f"{TAG}_ep{ep}.pt"))
    torch.save(ckpt, os.path.join(OUT, f"{TAG}.pt"))
    log(f"saved {TAG}_ep{ep}.pt; launching concept_eval ...")
    with open(f"C:/dev/phd/casper/.cache/{TAG}_eval.log", "a") as fh:
        subprocess.Popen([sys.executable, "scripts/concept_eval.py", f"{TAG}_ep{ep}", "ordinal"], stdout=fh, stderr=fh)

order = np.arange(len(rows))
for ep in range(1, EPOCHS + 1):
    enc.train(); rng = np.random.default_rng(100 + ep); rng.shuffle(order)
    t0 = time.time(); run = 0.0; nb = 0
    B = 256
    for b in range(0, len(order), B):
        bidx = order[b:b + B]
        ids, lv, pad = fold_batch(bidx, rng)
        z = encode(ids, lv, pad); logits = z @ Wd.T + bd
        tgt = torch.zeros(len(ids), ni)
        for i, u in enumerate(bidx):
            tgt[i, TGT[u]] = 1.0
        nll = -((F.log_softmax(logits, -1) * tgt).sum(-1) / tgt.sum(-1).clamp_min(1)).mean()
        opt.zero_grad(); nll.backward(); enc.item_emb.weight.grad[:ni] = 0.0; opt.step()
        run += float(nll); nb += 1
        if nb % 100 == 0:
            log(f"  ep{ep} b{nb}/{len(order)//B} NLL={run/nb:.4f} {(time.time()-t0)/60:.1f}m")
    drift = float((enc.item_emb.weight[:ni] - ITEM_SNAPSHOT).abs().max())
    log(f"[ep{ep}] NLL={run/max(nb,1):.4f} item-drift={drift:.2e} ({(time.time()-t0)/60:.1f}m)")
    assert drift == 0.0, "FREEZE BROKE mid-training"
    save_and_eval(ep, float("nan"))
log("done")
