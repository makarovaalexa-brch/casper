"""FUSION pre-test: cross-attention token on the ordinal recommender. Frozen backbone, train ONLY the fusion
(xfuse: Wq/Wk/Wv/Wo, Eval, Econf) for a short run; unified item+concept curriculum; REFUSALS FOLDED as a signal
(no-clue confidence), NOT dropped. Warm-start from paord (items) + member-bag concepts; Wo=0 => step-0 == paord
(containment assert). Then probes: containment, mismatch-binding (loved-A + hated-B move A,B oppositely),
concept-hate sinks member films, item-hate stays weak-positive. Predicts BOTH gates before a full retrain.
Env: EPOCHS (default 1), FREEZE_BACKBONE (default 1 => pre-test; 0 => full joint run also trains concept rows).
"""
import os, sys, time, subprocess
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try: sys.stdout.reconfigure(encoding='utf-8')
except Exception: pass
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
import set_mn as S
from set_mn import SetEncoder, RSD
from signed_latent import load_arena_base, ndcg10
import arena_core as AC

def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)
torch.set_num_threads(os.cpu_count())
S.set_grading("ordinal"); NC = S.NC; NLEV = S.NLEV; REFUSE = S.LV_REFUSE
META = "C:/dev/phd/casper/data/movielens/.cache/ml25m/meta.npz"; OUT = "C:/dev/phd/casper/.cache/set_mn"
LO = 4.0; D = 512
BASE = os.environ.get("BASE", "paord_best"); TAG = os.environ.get("TAG", "fus")
EPOCHS = int(os.environ.get("EPOCHS", "1")); FREEZE_BACKBONE = os.environ.get("FREEZE_BACKBONE", "1") == "1"
log(f"grading=ordinal NLEV={NLEV} | base={BASE} tag={TAG} epochs={EPOCHS} freeze_backbone={FREEZE_BACKBONE}")

base = load_arena_base(); ni = base["ni"]; head = base["headmask"]; headarr = np.where(head)[0]
d = np.load(META); uu = d["uu"].astype(np.int64); ii = d["ii"].astype(np.int64); rr = d["rr"].astype(np.float64)
o = np.argsort(uu, kind="stable"); uu, ii, rr = uu[o], ii[o], rr[o]; bnd = np.searchsorted(uu, np.arange(uu[-1] + 2))

uidsT = np.load(RSD + "/mm_train_uids.npy"); KT = np.load(RSD + "/mm_train_know.npy"); VT = np.load(RSD + "/mm_train_val.npy")
ANS = (KT[:, :NC] >= 1) & (VT[:, :NC] >= 0)                       # non-refused concept mask
CVAL = np.clip(VT[:, :NC], 0, 3).astype(np.int64)                 # concept value 0..3 (refused -> forced to REFUSE below)
CKN = np.clip(KT[:, :NC], 0, 2).astype(np.int64)                  # concept confidence 0..2
rows = []; TGT = []; KHALF = []
for r, uid in enumerate(uidsT):
    a, b = bnd[int(uid)], bnd[int(uid) + 1]; its, rat = ii[a:b], rr[a:b]
    if len(its) < 8: continue
    ru = np.random.default_rng(AC.SEED * 1_000_003 + int(uid)); p = ru.permutation(len(its)); h = len(its) // 2
    ki, kr = its[p[:h]], rat[p[:h]]; hi, hr = its[p[h:]], rat[p[h:]]
    hl = hi[hr >= LO]
    if len(ki) < 4 or len(hl) == 0: continue
    rows.append(r); TGT.append(hl.astype(np.int64)); KHALF.append((ki.astype(np.int64), kr))
rows = np.array(rows, np.int64)
log(f"usable train users {len(rows)}")

# ---- model ----
enc = SetEncoder(ni + NC, token_mode="xattn", pool="attn", nlev=NLEV, nknow=3)
pa = torch.load(os.path.join(OUT, BASE + ".pt"), map_location="cpu"); pas = pa["student"]
with torch.no_grad():
    enc.item_emb.weight[:ni].copy_(pas["item_emb.weight"])
    sd = {k: v for k, v in pas.items() if not k.startswith("item_emb")}
    enc.load_state_dict(sd, strict=False)                        # gamma/beta/attention/head/z0; xfuse+know fresh
    from reconciled import ConceptBank
    cb = ConceptBank(ni, base["cnt"]); M = cb.Mw.tocsr() if hasattr(cb.Mw, "tocsr") else cb.Mw
    rs = np.asarray(M.sum(1)).ravel(); rs[rs == 0] = 1.0
    Emb = (M @ pas["item_emb.weight"].numpy()) / rs[:, None]
    enc.item_emb.weight[ni:ni + NC].copy_(torch.from_numpy(Emb).float())
decoder = nn.Linear(D, ni); decoder.load_state_dict(pa["decoder"]); Wd = decoder.weight.detach(); bd = decoder.bias.detach()
log(f"warm from {BASE} (full={pa.get('full'):.4f}); concept rows=member-bag; xfuse Wo=0")

# freeze backbone; train xfuse (+ concept rows if not FREEZE_BACKBONE)
for p_ in enc.parameters(): p_.requires_grad_(False)
for p_ in enc.xfuse.parameters(): p_.requires_grad_(True)
train_params = list(enc.xfuse.parameters())
if not FREEZE_BACKBONE:
    enc.item_emb.weight.requires_grad_(True); train_params += [enc.item_emb.weight]
opt = torch.optim.AdamW(train_params, lr=3e-4, weight_decay=0.0)
ITEM_SNAP = enc.item_emb.weight[:ni].detach().clone()

def decode(z): return (z @ Wd.T + bd).numpy().astype(np.float64)
def enc_tokens(seqs):   # seqs: list of (ids[np], lvs[np], kn[np])
    B = len(seqs); Lm = max(len(s[0]) for s in seqs)
    ids = np.zeros((B, Lm), np.int64); lvs = np.zeros((B, Lm), np.int64); kn = np.zeros((B, Lm), np.int64); pad = np.ones((B, Lm), bool)
    for i, (a, l, k) in enumerate(seqs):
        ids[i, :len(a)] = a; lvs[i, :len(a)] = l; kn[i, :len(a)] = k; pad[i, :len(a)] = False
    return enc(torch.from_numpy(ids), torch.zeros(B, Lm), torch.from_numpy(pad), torch.from_numpy(lvs), torch.from_numpy(kn))

# ---- CONTAINMENT ASSERT: full-profile at init must == paord (xfuse=0) ----
def eval_full(nmax=3000):
    ff = []; rng = np.random.default_rng(0)
    for i in rng.choice(len(rows), min(nmax, len(rows)), replace=False):
        ki, kr = KHALF[i]
        if len(ki) < 3: continue
        lv = S.sv_to_level((kr - 2.75) / 2.25); kn = np.full(len(ki), 2, np.int64)   # rated => know-well
        with torch.no_grad():
            z = enc_tokens([(ki, lv, kn)])
        sc = decode(z)[0]
        v = ndcg10(sc, list(TGT[i]), set(int(x) for x in ki), head, False)
        if v is not None: ff.append(v)
    return float(np.mean(ff))
f0 = eval_full(); log(f"CONTAINMENT full-profile at init = {f0:.4f} (paord = {pa.get('full'):.4f}; must match)")

def eval_cold_concept(k=16, nmax=2000):
    """cold-start: fold k random concepts (answered + REFUSALS folded as signal), TAIL NDCG@10 vs held likes."""
    rng = np.random.default_rng(1); vals = []
    for i in rng.choice(len(rows), min(nmax, len(rows)), replace=False):
        r = rows[i]; cids = rng.choice(NC, size=k, replace=False); ids = []; lv = []; kn = []
        for c in cids:
            if ANS[r, c]: ids.append(ni + int(c)); lv.append(int(CVAL[r, c])); kn.append(int(CKN[r, c]))
            else: ids.append(ni + int(c)); lv.append(REFUSE); kn.append(0)
        with torch.no_grad():
            z = enc_tokens([(np.array(ids), np.array(lv), np.array(kn))])
        v = ndcg10(decode(z)[0], list(TGT[i]), set(), head, True)
        if v is not None: vals.append(v)
    return float(np.mean(vals))
log(f"cold-concept TAIL@10 k16 at init (member-bag, untrained) = {eval_cold_concept():.4f}  (old FiLM Phase-B = 0.0774)")

# ---- curriculum: per user, fold a mix of items (known-half) + concepts (incl REFUSALS as no-clue) ----
def fold_user(i, rng):
    r = rows[i]; toks_ids = []; toks_lv = []; toks_kn = []
    mode = rng.random()
    if mode < 0.5:                                               # include some item folds
        ki, kr = KHALF[i]
        m = rng.random(len(ki)) < 0.5
        if m.any():
            its = ki[m]; lv = S.sv_to_level((kr[m] - 2.75) / 2.25)
            toks_ids += (its).tolist(); toks_lv += lv.tolist(); toks_kn += [2] * len(its)
    if mode > 0.2:                                               # include concepts (some refused, folded as signal)
        k = int(np.exp(rng.uniform(0, np.log(48))))
        cids = rng.choice(NC, size=min(k, NC), replace=False)
        for c in cids:
            if ANS[r, c]:
                toks_ids.append(ni + int(c)); toks_lv.append(int(CVAL[r, c])); toks_kn.append(int(CKN[r, c]))
            else:                                                # REFUSAL folded: value=REFUSE, confidence=no-clue(0)
                toks_ids.append(ni + int(c)); toks_lv.append(REFUSE); toks_kn.append(0)
    if not toks_ids: return None
    return (np.array(toks_ids, np.int64), np.array(toks_lv, np.int64), np.array(toks_kn, np.int64))

# ---- train ----
order = np.arange(len(rows))
for ep in range(1, EPOCHS + 1):
    enc.train(); rng = np.random.default_rng(100 + ep); rng.shuffle(order); t0 = time.time(); run = 0.0; nb = 0
    B = 128
    for b in range(0, len(order), B):
        seqs = []; owners = []
        for i in order[b:b + B]:
            f = fold_user(i, rng)
            if f is not None: seqs.append(f); owners.append(i)
        if not seqs: continue
        z = enc_tokens(seqs); logits = z @ Wd.T + bd
        tgt = torch.zeros(len(seqs), ni)
        for j, i in enumerate(owners): tgt[j, TGT[i]] = 1.0
        nll = -((F.log_softmax(logits, -1) * tgt).sum(-1) / tgt.sum(-1).clamp_min(1)).mean()
        opt.zero_grad(); nll.backward();
        if not FREEZE_BACKBONE: enc.item_emb.weight.grad[:ni] = 0.0
        opt.step(); run += float(nll); nb += 1
        if nb % 100 == 0: log(f"  ep{ep} b{nb}/{len(order)//B} NLL={run/nb:.4f} {(time.time()-t0)/60:.1f}m")
    drift = float((enc.item_emb.weight[:ni] - ITEM_SNAP).abs().max())
    log(f"[ep{ep}] NLL={run/max(nb,1):.4f} item-drift={drift:.2e} full={eval_full():.4f} "
        f"cold-concept-tail-k16={eval_cold_concept():.4f} ({(time.time()-t0)/60:.1f}m)")
    torch.save({"student": enc.state_dict(), "decoder": decoder.state_dict(), "epoch": ep, "grading": "ordinal",
                "full": eval_full()}, os.path.join(OUT, f"{TAG}_ep{ep}.pt"))

# ---- PROBES ----
log("=== probes ===")
import csv, json
keepI = d["keepI"]; mid2title = {int(r["movieId"]): r["title"] for r in csv.DictReader(open("data/movielens/movies.csv", encoding="utf-8"))}
tagnames = [t.get('tag', '?') for t in json.load(open('.cache/instrument2/tag_questions.json'))['tags']]
def cidx(name): return tagnames.index(name) if name in tagnames else -1
def members(c): return np.array(M[c].indices)
def fold_score(tok):   # tok list of (id,lv,kn)
    ids = np.array([t[0] for t in tok]); lv = np.array([t[1] for t in tok]); kn = np.array([t[2] for t in tok])
    with torch.no_grad(): z = enc_tokens([(ids, lv, kn)])
    return decode(z)[0]
with torch.no_grad():
    z0score = decode(enc(torch.zeros(1,1,dtype=torch.long), torch.zeros(1,1), torch.ones(1,1,dtype=torch.bool), torch.zeros(1,1,dtype=torch.long), torch.zeros(1,1,dtype=torch.long)))[0]
for cn in ['horror','animation','sci fi']:
    c = cidx(cn)
    if c < 0: continue
    mem = members(c); mem = mem[~head[mem]][:200]
    if len(mem) < 5: continue
    love = fold_score([(ni+c, 3, 2)]); hate = fold_score([(ni+c, 0, 2)]); refu = fold_score([(ni+c, REFUSE, 0)])
    dl = (love[mem]-z0score[mem]).mean(); dh = (hate[mem]-z0score[mem]).mean(); dr = (refu[mem]-z0score[mem]).mean()
    log(f"  concept '{cn}': member-score vs z0  LOVED {dl:+.3f}  REFUSED {dr:+.3f}  HATED {dh:+.3f}  "
        f"=> loved>refused>hated? {'YES' if dl>dr>dh else 'NO'}")
# mismatch-binding: loved-horror + hated-romance, must move horror UP and romance DOWN independently
ch, cr = cidx('horror'), cidx('romance')
if ch >= 0 and cr >= 0:
    mh = members(ch); mh = mh[~head[mh]][:200]; mr = members(cr); mr = mr[~head[mr]][:200]
    s = fold_score([(ni+ch, 3, 2), (ni+cr, 0, 2)])
    log(f"  MISMATCH-BIND (loved-horror + hated-romance): horror delta {(s[mh]-z0score[mh]).mean():+.3f} (want >0)  "
        f"romance delta {(s[mr]-z0score[mr]).mean():+.3f} (want <0)")
# item-hate must stay weak-positive: fold a popular item hated, check its decoder-neighbours vs z0
W = Wd.numpy()
for it in [0, 107]:
    nbr = np.argsort(-(W @ W[it]))[1:21]
    h = fold_score([(it, 0, 2)]); dh = (h[nbr]-z0score[nbr]).mean()
    log(f"  ITEM {it} hated: neighbour delta vs z0 {dh:+.3f} (want >=0, weak-positive preserved)")
log("done")
