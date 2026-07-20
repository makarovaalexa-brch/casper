"""Depth-2 greedy tree on the FROZEN pb2 recommender. The probe, extended past depth 1.
NO surrogate, NO frozen precision, NO action-space cut: pb2 does every fold, all 800 candidates.
Cheap because within a tree node every user shares the belief -> one fold per (question, answer-level),
and each user's NDCG is a ranked-membership against the shared top-10.

Pilot: NPILOT users (author-requested 20k), credit-neutral masking (Paper B: an asked-answered item
leaves BOTH the ranking and the target set). Reports BASE / STATIC-1 / STATIC-2 / ADAPTIVE-2, TAIL NDCG@10.
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch, torch.nn as nn
import vhead
from set_mn import SetEncoder

def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)
torch.set_num_threads(os.cpu_count())
_W = 1.0 / np.log2(np.arange(2, 12))                                   # DCG@10 discounts

N = int(os.environ.get("NPILOT", "20000"))
base, ni, head, bank, held_t, LV, ANS = vhead.build()                 # SAME split/answer-map as value-head build
held_t = held_t[:N]; LV = LV[:N]; ANS = ANS[:N]
headset = set(int(x) for x in np.where(head)[0])
log(f"pilot users {N}  bank {len(bank)}  catalog {ni}")

ck = torch.load(vhead.PB2, map_location="cpu")
NLEV = ck["student"]["gamma.weight"].shape[0]
enc = SetEncoder(ni, token_mode="film", pool="attn", nlev=NLEV, nknow=0)
enc.load_state_dict(ck["student"], strict=False); enc.eval()
for p in enc.parameters(): p.requires_grad_(False)
dec = nn.Linear(vhead.D, ni); dec.load_state_dict(ck["decoder"])
Wd = dec.weight.detach(); bd = dec.bias.detach()
z0 = enc.z0.detach().reshape(1, -1)

def decode(z): return (z @ Wd.T + bd).numpy().astype(np.float64)

def encode_seqs(seqs):
    B = len(seqs); mL = max((len(s) for s in seqs), default=0)
    if mL == 0: return z0.expand(B, -1)
    ids = torch.zeros(B, mL, dtype=torch.long); lvs = torch.zeros(B, mL, dtype=torch.long)
    pad = torch.ones(B, mL, dtype=torch.bool)
    for i, s in enumerate(seqs):
        for k, (it, lv) in enumerate(s):
            ids[i, k] = it; lvs[i, k] = lv; pad[i, k] = False
    with torch.no_grad():
        return enc(ids, torch.zeros(B, mL), pad, lvs)

# TAIL targets per user (drop head once)
TGT = []
for u in range(N):
    t = np.asarray(held_t[u], np.int64); TGT.append(t[~np.isin(t, np.where(head)[0])])

def ranked_top10(score, mask_ids):
    s = score.copy(); s[head] = -1e30
    if mask_ids: s[list(mask_ids)] = -1e30
    idx = np.argpartition(-s, 10)[:10]
    return idx[np.argsort(-s[idx])]                                    # ordered top-10 catalog ids

def user_ndcg(order10, targets, extra_excl):
    rel = targets if extra_excl is None else targets[targets != extra_excl]
    if len(rel) == 0: return None
    rs = set(int(x) for x in rel)
    dcg = sum(_W[p] for p, t in enumerate(order10) if int(t) in rs)
    return dcg / (_W[:min(10, len(rel))].sum() + 1e-12)

def node_candidate_values(U, T, path_ids, path_idx):
    """MEAN tail-NDCG@10 over users U after asking each candidate j (refusers keep the node score)."""
    base_score = decode(encode_seqs([T]))[0]
    base_order = ranked_top10(base_score, path_ids)                   # for refusers (candidate not masked)
    vals = np.full(len(bank), -1.0)
    for j in range(len(bank)):
        if j in path_idx: continue
        cid = int(bank[j]); ans = ANS[U, j]; lvl = LV[U, j]
        present = np.unique(lvl[ans]) if ans.any() else np.array([], np.int64)
        order = {}
        if len(present):
            S = decode(encode_seqs([T + [(cid, int(l))] for l in present]))
            for k, l in enumerate(present):
                order[int(l)] = ranked_top10(S[k], path_ids | {cid})  # credit-neutral: mask candidate too
        acc = 0.0; m = 0
        for idx, u in enumerate(U):
            if ans[idx]:
                nd = user_ndcg(order[int(lvl[idx])], TGT[u], cid)
            else:
                nd = user_ndcg(base_order, TGT[u], None)
            if nd is not None: acc += nd; m += 1
        vals[j] = acc / m if m else -1.0
    return vals

t0 = time.time()
allU = np.arange(N)

# BASE: no question
base_order_root = ranked_top10(decode(z0)[0], set())
b_acc = [user_ndcg(base_order_root, TGT[u], None) for u in range(N)]
BASE = np.mean([x for x in b_acc if x is not None])
log(f"BASE (no question)          tail-NDCG@10 = {BASE:.4f}")

# ROOT: best static Q1
log("scoring 800 candidates at the ROOT ...")
rootv = node_candidate_values(allU, [], set(), set())
j1 = int(np.argmax(rootv)); STATIC1 = rootv[j1]
log(f"Q1* = bank[{j1}] cat={int(bank[j1])}  STATIC-1 tail-NDCG@10 = {STATIC1:.4f}  ({time.time()-t0:.0f}s)")

# children by Q1 answer level (+ refusers)
ans1 = ANS[allU, j1]; lvl1 = LV[allU, j1]
children = []
for l in np.unique(lvl1[ans1]):
    Uc = allU[ans1 & (lvl1 == l)]
    children.append((int(l), Uc, [(int(bank[j1]), int(l))]))
Uref = allU[~ans1]
if len(Uref): children.append((-1, Uref, []))                        # refusers: unchanged state
log(f"Q1 splits {N} users into {len(children)} children: " +
    ", ".join(f"lvl{l}:{len(U)}" for l, U, _ in children))

# score each candidate Q2 within each child
child_vals = []
for l, Uc, Tc in children:
    v = node_candidate_values(Uc, Tc, {int(bank[j1])}, {j1})
    child_vals.append((l, Uc, v))
    j2 = int(np.argmax(v))
    log(f"  child lvl{l:+d} n={len(Uc):6d}: Q2* = bank[{j2}] cat={int(bank[j2])}  val={v[j2]:.4f}")

# ADAPTIVE-2: best Q2 per child ; STATIC-2: one Q2 maximizing the global weighted mean
w = np.array([len(Uc) for _, Uc, _ in child_vals], float)
ADAPT2 = sum(len(Uc) * child_vals[i][2].max() for i, (_, Uc, _) in enumerate(child_vals)) / N
V = np.vstack([child_vals[i][2] for i in range(len(child_vals))])     # (nchild, 800)
global_mean = (w[:, None] * V).sum(0) / N
j2s = int(np.argmax(global_mean)); STATIC2 = global_mean[j2s]
log("")
log(f"BASE (0 q)          {BASE:.4f}")
log(f"STATIC-1 (Q1 only)  {STATIC1:.4f}   ({STATIC1-BASE:+.4f} vs base)")
log(f"STATIC-2 (Q1+fixed) {STATIC2:.4f}   Q2*=bank[{j2s}] cat={int(bank[j2s])}")
log(f"ADAPTIVE-2 (tree)   {ADAPT2:.4f}   ({ADAPT2-STATIC2:+.4f} vs STATIC-2)   [the adaptivity prize]")
log(f"total {time.time()-t0:.0f}s")
