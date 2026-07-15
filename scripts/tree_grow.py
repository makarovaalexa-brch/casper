"""Greedy elicitation TREE on the frozen pb2 recommender = Golbandi's tree, but:
  - node predictor is the 0.485 set-encoder (not an item mean),
  - answers are graded (not ternary),
  - the split criterion is TAIL NDCG@10, a RANKING loss (Koren's flagged-open extension).
Grows depth by depth over ALL users. Reports ADAPTIVE vs STATIC tail-NDCG@10 and wall-clock AFTER EACH TURN.
Within a node every user shares the belief -> one fold per (question, answer-level); per-user NDCG is a
vectorized sparse membership. NO surrogate, NO action-space cut: pb2 folds all 800 candidates.
Credit-neutral masking (Paper B): an asked-answered item leaves BOTH the ranking and the target set;
a REFUSED item stays recommendable (user doesn't know it) and only burns the turn.
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch, torch.nn as nn
from scipy.sparse import csr_matrix
import vhead
from set_mn import SetEncoder

def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)
torch.set_num_threads(os.cpu_count())
_W = 1.0 / np.log2(np.arange(2, 12))
DMAX = int(os.environ.get("DMAX", "4"))
N = int(os.environ.get("NPILOT", "0"))                       # 0 => ALL users
OUTF = os.environ.get("OUTF", "C:/dev/phd/casper/.cache/tree_grow_result.npz")

base, ni, head, bank, held_t, LV, ANS = vhead.build()
if N and N < len(held_t):
    held_t = held_t[:N]; LV = LV[:N]; ANS = ANS[:N]
N = len(held_t)
headarr = np.where(head)[0]
# BUNDLE answers into 4 buckets (author, Jul 15): {1-2 disliked}{3 meh}{4 liked}{5 loved}.
# Branch AND fold on the bucket (one representative FiLM level each -> all users in a node share the token,
# which keeps the fold shared). LV is the half-star level 0..9 (star=(LV+1)/2). 4-way fan-out => deeper tree.
_rs = np.clip(np.rint((LV.astype(np.float64) + 1) / 2.0), 1, 5).astype(np.int64)     # whole star 1..5
LV = np.select([_rs <= 2, _rs == 3, _rs == 4, _rs == 5], [3, 5, 7, 9]).astype(LV.dtype)  # {2*,3*,4*,5*}
BUCKET = {3: "disliked(1-2*)", 5: "meh(3*)", 7: "liked(4*)", 9: "loved(5*)"}
log(f"users {N}  bank {len(bank)}  catalog {ni}  DMAX {DMAX}  [answers bundled -> 4 buckets]")

# titles: internal catalog id -> movieId (keepI) -> title (movies.csv)
import csv as _csv
_keepI = np.load("data/movielens/.cache/ml25m/meta.npz", allow_pickle=True)["keepI"]
_mid2title = {}
with open("data/movielens/movies.csv", encoding="utf-8") as f:
    for r in _csv.DictReader(f): _mid2title[int(r["movieId"])] = r["title"]
def title_of(catid): return _mid2title.get(int(_keepI[int(catid)]), f"cat{catid}")

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
        for k, (it, lv) in enumerate(s): ids[i, k] = it; lvs[i, k] = lv; pad[i, k] = False
    with torch.no_grad(): return enc(ids, torch.zeros(B, mL), pad, lvs)

# ---- tail target incidence matrix (N x ni), credit-neutral bookkeeping ----
rows, cols = [], []
for u in range(N):
    t = np.asarray(held_t[u], np.int64); t = t[~np.isin(t, headarr)]
    rows.extend([u] * len(t)); cols.extend(t.tolist())
M = csr_matrix((np.ones(len(rows), np.float32), (rows, cols)), shape=(N, ni)).tocsr()
cnt = np.asarray(M.sum(1)).ravel().astype(np.int64)
def idcg_of(c): return _W[:min(10, int(c))].sum() if c > 0 else 0.0
IDCG = np.array([idcg_of(c) for c in cnt])
TGTB = (M[:, bank] > 0).toarray()                                    # (N,800) bool: does user target bank[j]?
log(f"target matrix nnz {M.nnz}  users w/ >=1 tail target {int((cnt>0).sum())}")

def ranked_top10(score, mask_ids):
    s = score.copy(); s[headarr] = -1e30
    if mask_ids: s[list(mask_ids)] = -1e30
    idx = np.argpartition(-s, 10)[:10]
    return idx[np.argsort(-s[idx])]

def hits(MU, local_rows, order10):
    """DCG@10 for each local row against an ordered 10-id list (vectorized on the node submatrix)."""
    sub = MU[local_rows][:, order10]
    return np.asarray(sub.multiply(_W).sum(1)).ravel()

def score_node(U, T, path_ids, path_idx):
    """MEAN tail-NDCG@10 over U after asking each candidate j. ALL candidate folds batched into ONE
    encode + ONE decode (the per-candidate torch-call overhead was the bottleneck)."""
    MU = M[U].tocsr(); cntU = cnt[U]; IDCGU = IDCG[U]; nU = len(U)
    base_order = ranked_top10(decode(encode_seqs([T]))[0], path_ids)
    cand = [j for j in range(len(bank)) if j not in path_idx]
    ansU = {}; lvlU = {}; seqs = []; meta = []                  # meta: (j, cid, level, row_in_S)
    for j in cand:
        cid = int(bank[j]); ans = ANS[U, j]; lvl = LV[U, j]; ansU[j] = ans; lvlU[j] = lvl
        for l in (np.unique(lvl[ans]) if ans.any() else ()):
            meta.append((j, cid, int(l), len(seqs))); seqs.append(T + [(cid, int(l))])
    S = decode(encode_seqs(seqs)) if seqs else np.zeros((0, ni))
    order_of = {(j, l): ranked_top10(S[ri], path_ids | {cid}) for (j, cid, l, ri) in meta}
    vals = np.full(len(bank), -1.0)
    for j in cand:
        cid = int(bank[j]); ans = ansU[j]; lvl = lvlU[j]; dcg = np.empty(nU)
        ref = ~ans
        if ref.any(): dcg[ref] = hits(MU, np.where(ref)[0], base_order)     # refuser: candidate NOT masked
        for l in (np.unique(lvl[ans]) if ans.any() else ()):
            sel = np.where(ans & (lvl == l))[0]; dcg[sel] = hits(MU, sel, order_of[(j, int(l))])
        idcg = IDCGU.copy()
        tc = TGTB[U, j] & ans                                                # answered & targets bank[j]
        if tc.any():
            wi = np.where(tc)[0]; idcg[wi] = [idcg_of(cntU[i] - 1) for i in wi]
        valid = idcg > 0
        vals[j] = (dcg[valid] / idcg[valid]).mean() if valid.any() else -1.0
    return vals

def eval_choice(U, T, path_ids, j):
    """Per-user tail-NDCG@10 (aligned to U) after asking candidate j at this node. For the chosen question."""
    MU = M[U].tocsr(); cntU = cnt[U]; nU = len(U)
    base_order = ranked_top10(decode(encode_seqs([T]))[0], path_ids)
    cid = int(bank[j]); ans = ANS[U, j]; lvl = LV[U, j]
    present = np.unique(lvl[ans]) if ans.any() else ()
    order = {}
    if len(present):
        S = decode(encode_seqs([T + [(cid, int(l))] for l in present]))
        for k, l in enumerate(present): order[int(l)] = ranked_top10(S[k], path_ids | {cid})
    dcg = np.empty(nU); ref = ~ans
    if ref.any(): dcg[ref] = hits(MU, np.where(ref)[0], base_order)
    for l in present:
        sel = np.where(ans & (lvl == l))[0]; dcg[sel] = hits(MU, sel, order[int(l)])
    idcg = IDCG[U].copy(); tc = TGTB[U, j] & ans
    if tc.any():
        wi = np.where(tc)[0]; idcg[wi] = [idcg_of(cntU[i] - 1) for i in wi]
    nd = np.full(nU, np.nan); valid = idcg > 0; nd[valid] = dcg[valid] / idcg[valid]
    return nd

# self-check: vectorized vs reference on a small sample at the root
def _ref_value(U, j):
    cid = int(bank[j]); base_order = ranked_top10(decode(encode_seqs([[]]))[0], set())
    ans = ANS[U, j]; lvl = LV[U, j]; present = np.unique(lvl[ans]) if ans.any() else []
    ordr = {}
    if len(present):
        S = decode(encode_seqs([[(cid, int(l))] for l in present]))
        for k, l in enumerate(present): ordr[int(l)] = ranked_top10(S[k], {cid})
    acc = 0.0; m = 0
    for idx, u in enumerate(U):
        tg = np.asarray(held_t[u], np.int64); tg = tg[~np.isin(tg, headarr)]
        if ans[idx]:
            rel = tg[tg != cid]; O = ordr[int(lvl[idx])]
        else:
            rel = tg; O = base_order
        if len(rel) == 0: continue
        dcg = sum(_W[p] for p, t in enumerate(O) if int(t) in set(int(x) for x in rel))
        acc += dcg / (_W[:min(10, len(rel))].sum()); m += 1
    return acc / m if m else -1.0
Usmp = np.arange(min(N, 2000))
_vv = score_node(Usmp, [], set(), set())
for jt in (0, 5, 37):
    assert abs(_vv[jt] - _ref_value(Usmp, jt)) < 1e-9, f"self-check FAIL j={jt}"
log("self-check PASSED (vectorized NDCG == reference)")

# ---- grow the tree, report after each turn ----
TIME_BUDGET = float(os.environ.get("TIME_BUDGET", "14400"))          # seconds (default 4h)
STATIC_MAXD = int(os.environ.get("STATIC_MAXD", "4"))                # compute the static baseline this deep
LOG_MIN = max(300, N // 200)                                         # only log nodes at least this big
SH = {3: "D", 5: "M", 7: "L", 9: "V"}                               # disliked/meh/liked/loVed
def bpath(T): return "/".join(SH.get(l, "?") for _, l in T) or "root"
t0 = time.time()
_bo = ranked_top10(decode(z0)[0], set())
_bh = hits(M, np.arange(N), _bo); _bv = IDCG > 0
BASE = (_bh[_bv] / IDCG[_bv]).mean()
log(f"turn 0  BASE tail-NDCG@10 = {BASE:.4f}")

allU = np.arange(N)
ADAPT_user = np.full(N, np.nan); STAT_user = np.full(N, np.nan)
rootv = score_node(allU, [], set(), set()); j1 = int(np.argmax(rootv))
pu1 = eval_choice(allU, [], set(), j1); ADAPT_user[:] = pu1; STAT_user[:] = pu1
log(f"turn 1  Q1*=bank[{j1}] cat={int(bank[j1])} \"{title_of(bank[j1])}\"  tail-NDCG@10={rootv[j1]:.4f}  "
    f"(+{rootv[j1]-BASE:.4f})  [{time.time()-t0:.0f}s]  STATIC==ADAPTIVE at depth 1")

def children_of(node, j):
    U, T, pid, pidx = node; cid = int(bank[j]); ans = ANS[U, j]; lvl = LV[U, j]; out = []
    for l in np.unique(lvl[ans]):
        sel = ans & (lvl == l)
        out.append((U[sel], T + [(cid, int(l))], pid | {cid}, pidx | {j}))
    if (~ans).any(): out.append((U[~ans], T, pid, pidx | {j}))     # refuser: state unchanged, can't re-ask
    return out

def ci(diff):
    d = diff[np.isfinite(diff)]
    return d.mean(), 1.96 * d.std() / np.sqrt(len(d))

front0 = children_of((allU, [], set(), set()), j1)
adapt_front = list(front0); stat_front = list(front0)
depth_stats = [(1, rootv[j1], rootv[j1], 0.0)]
for d in range(2, DMAX + 1):
    if time.time() - t0 > TIME_BUDGET:
        log(f"time budget {TIME_BUDGET:.0f}s reached before depth {d}; stopping"); break
    td = time.time()
    # ADAPTIVE: each node picks its own best next question
    new_adapt = []; logged = 0
    for node in adapt_front:
        if len(node[0]) == 0: continue
        v = score_node(*node); j = int(np.argmax(v))
        ADAPT_user[node[0]] = eval_choice(*node[:3], j)
        if len(node[0]) >= LOG_MIN and logged < 40:
            log(f"    d{d} n={len(node[0]):6d} path={bpath(node[1]):<14} -> Q bank[{j:3d}] "
                f"rank{j}/800 \"{title_of(bank[j])}\" val={v[j]:.4f}"); logged += 1
        new_adapt += children_of(node, j)
    ADAPT = np.nanmean(ADAPT_user)
    # STATIC baseline (only through STATIC_MAXD): one question for the whole depth
    if d <= STATIC_MAXD:
        Vs = []; ws = []
        for node in stat_front:
            if len(node[0]) == 0: continue
            Vs.append(score_node(*node)); ws.append(len(node[0]))
        G = (np.array(ws)[:, None] * np.vstack(Vs)).sum(0) / N; js = int(np.argmax(G))
        new_stat = []
        for node in stat_front:
            if len(node[0]): STAT_user[node[0]] = eval_choice(*node[:3], js); new_stat += children_of(node, js)
        stat_front = new_stat; STAT = np.nanmean(STAT_user)
        mdiff, hw = ci(ADAPT_user - STAT_user)
        log(f"turn {d}  ADAPTIVE {ADAPT:.4f}  STATIC {STAT:.4f}  prize {mdiff:+.4f} +/- {hw:.4f}  "
            f"(Qd_static bank[{js}] \"{title_of(bank[js])}\")  nodes {len(new_adapt)}  "
            f"[{time.time()-td:.0f}s, tot {time.time()-t0:.0f}s]")
    else:
        log(f"turn {d}  ADAPTIVE {ADAPT:.4f}  (+{ADAPT-BASE:.4f} vs base)  static frozen@d{STATIC_MAXD}  "
            f"nodes {len(new_adapt)}  [{time.time()-td:.0f}s, tot {time.time()-t0:.0f}s]")
    depth_stats.append((d, ADAPT, STAT if d <= STATIC_MAXD else np.nan, time.time() - t0))
    adapt_front = new_adapt
    np.savez(OUTF, ADAPT_user=ADAPT_user, STAT_user=STAT_user,
             depth_stats=np.array(depth_stats), j1=j1, reached=d)
    if not adapt_front: log("frontier exhausted"); break
log(f"done. reached depth {depth_stats[-1][0]}  final ADAPTIVE {depth_stats[-1][1]:.4f}  in {time.time()-t0:.0f}s")
