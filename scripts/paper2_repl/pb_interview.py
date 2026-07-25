"""pb_interview.py -- STEP 3/3. FAITHFUL short-interview battery (geometric answers, item+concept,
static-pop + EIG selection), ported from scripts/_archive/paper2_old/answerability_concept.py to the
canonical ML-25M ruler. Reports the cold q0 + per-question curve q0..q8 (full AND tail NDCG@10) on
ALL 10k canonical test users.

FAITHFUL answer model = GEOMETRIC (no LLM, no SEL clip):
  u* = enc_u(the user's fold-in item tokens (Q[j], r-mu-bi[j]))            [true taste from known half]
  concept c answerable iff >=2 fold-in members; answer = POS if u*.Ec[c] > per-user mean, else NEG
  item j answerable iff j in fold-in; answer token = (Q[j], r-mu-bi[j])    [graded residual]
Decoder = the method's own: score = popb + Qp @ u  (popb=log(train-like-count+1); Qp=learned factors).
Ruler = canonical: full+tail NDCG@10, head mask from proc train mass (identical to run_ml25m_liang),
fold-in items masked, targets=test_te; tail masks head items + drops head targets.

Selectors (T=8, wasted-turn: asking consumes a turn, fold only if answerable):
  pop_item  : items by popularity (pool = top-1500 popular), faithful ITEMC=order_pop[:1500]
  conc_pop  : concepts by #members (frequency)
  eig_item  : greedy expected-coverage EIG over cand=[top-1500 not asked][:500], x P(answerable)  [infogain_items]
  conc_eig  : greedy answerability-aware concept EIG over all concepts x P(answerable)             [infogain_concepts]
  rand_item : random item selection (weak-selector reference)
Controls: (i) answer-permutation existential control on pop_item (partner-shuffled belief -> lift must vanish);
  (ii) canonical-snap: full-profile fold NDCG@10 (all fold-in items) reported as the warm reference;
  (iii) leak: fold-in and targets disjoint by construction; EIG uses belief only (ref=top-pop), never targets.

Outputs -> experiments/paper2_repl/pb_results.json
Env: NEVAL(10000) MODES(pop_item,conc_pop,eig_item,conc_eig,rand_item)
"""
import os, time, json, numpy as np, torch, torch.nn as nn

ROOT = 'C:/dev/phd/casper'; OUT = f'{ROOT}/.cache/paper2_repl'; RESD = f'{ROOT}/experiments/paper2_repl'
os.makedirs(RESD, exist_ok=True)
torch.set_num_threads(os.cpu_count() or 4); D = 64; T = 8
NEVAL = int(os.environ.get('NEVAL', 10000))
MODES = os.environ.get('MODES', 'pop_item,conc_pop,eig_item,conc_eig,rand_item').split(',')
REVEALS = list(range(T + 1))
rng = np.random.default_rng(0)
t0 = time.time()
def log(m): print(f"[{time.time()-t0:6.0f}s] {m}", flush=True)

Z = np.load(f'{OUT}/pb_data.npz')
Q = Z['Q'].astype(np.float32); bi = Z['bi'].astype(np.float32); mu = float(Z['mu'])
cnt = Z['cnt']; ni = int(Z['ni']); ntr = int(Z['ntr'])
POS = float(Z['POS']); NEG = float(Z['NEG'])
ccount = Z['ccount']; citems_flat = Z['citems_flat']; citems_off = Z['citems_off']
headmask = Z['headmask'].astype(bool)
popb = np.log(cnt + 1.0).astype(np.float32)
Qp = np.load(f'{OUT}/Qp.npy').astype(np.float32)
Ec = np.load(f'{OUT}/Ec.npy').astype(np.float32); nc = Ec.shape[0]
order_pop = np.argsort(-cnt)
ITEMC = list(order_pop[:1500])                          # faithful pop item pool
R = np.array(order_pop[:500])                           # EIG reference set (top-pop)
CONC_POP_ORD = list(np.argsort(-ccount))
item2c = [[] for _ in range(ni)]
for c in range(nc):
    for j in citems_flat[citems_off[c]:citems_off[c + 1]]: item2c[int(j)].append(c)
item2c = [np.array(v, np.int32) for v in item2c]

# ---------- encoder ----------
class Enc(nn.Module):
    def __init__(s):
        super().__init__(); s.inp = nn.Sequential(nn.Linear(D + 1, 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU())
        s.att = nn.Linear(128, 1); s.val = nn.Linear(128, D)
    def forward(s, t, m):
        h = s.inp(t); a = s.att(h).squeeze(-1).masked_fill(m == 0, -1e9); al = torch.softmax(a, 1)
        return (al.unsqueeze(-1) * s.val(h)).sum(1)
enc = Enc(); enc.load_state_dict(torch.load(f'{OUT}/enc.pt')); enc.eval()
def sig(z): return 1.0 / (1.0 + np.exp(-z))
def enc_u(rows):
    if not rows: return np.zeros(D, np.float32)
    tk = np.zeros((1, len(rows), D + 1), np.float32); mk = np.ones((1, len(rows)), np.float32)
    for q, (f, v) in enumerate(rows): tk[0, q, :D] = f; tk[0, q, D] = v
    with torch.no_grad(): return enc(torch.tensor(tk), torch.tensor(mk)).numpy()[0]
def enc_fold(base_rows, cand_feats, ansval):             # vectorized: append 1 candidate token to a shared base
    L = len(base_rows); M = cand_feats.shape[0]
    tk = np.zeros((M, L + 1, D + 1), np.float32); mk = np.ones((M, L + 1), np.float32)
    if L:
        base = np.zeros((L, D + 1), np.float32)
        for q, (f, v) in enumerate(base_rows): base[q, :D] = f; base[q, D] = v
        tk[:, :L, :] = base[None]
    tk[:, L, :D] = cand_feats; tk[:, L, D] = ansval
    with torch.no_grad(): return enc(torch.tensor(tk), torch.tensor(mk)).numpy()

# ---------- concept answerability prior pe_ans (exact over train users) ----------
uu_s = Z['uu_s']; ii_s = Z['ii_s']; off = Z['off']
pe_cnt = np.zeros(nc)
for x in range(ntr):
    a, c = off[x], off[x + 1]
    cn = {}
    for j in ii_s[a:c]:
        for ki in item2c[int(j)]: cn[ki] = cn.get(ki, 0) + 1
    for ki, ct in cn.items():
        if ct >= 2: pe_cnt[ki] += 1
pe_ans = (pe_cnt / max(ntr, 1)).astype(np.float32) + 1e-6
p_seen = (cnt / max(ntr, 1)).astype(np.float32)          # item answerability prior
log(f"pe_ans built; concepts>=0.5 answerable: {(pe_ans>=0.5).sum()}/{nc}")

# ---------- test users ----------
fu = Z['foldin_uid']; fs = Z['foldin_sid']; fr = Z['foldin_rat']; tu = Z['target_uid']; ts = Z['target_sid']
FOLD = {}; TGT = {}
for u, s, r in zip(fu, fs, fr): FOLD.setdefault(int(u), []).append((int(s), float(r)))
for u, s in zip(tu, ts): TGT.setdefault(int(u), set()).add(int(s))
users = [u for u in FOLD if u in TGT][:NEVAL]
log(f"test users with fold-in + targets: {len(users)}")

_W = 1.0 / np.log2(np.arange(2, 12))
def ndcg10(u, rel, excl, tail):
    sc = (popb + Qp @ u).astype(np.float64).copy(); sc[list(excl)] = -1e30
    if tail: sc[headmask] = -1e30; rel = [t for t in rel if not headmask[t]]
    if not rel: return None
    top = np.argpartition(-sc, 10)[:10]; top = top[np.argsort(-sc[top])]
    rs = set(rel); idcg = _W[:min(10, len(rel))].sum()
    return sum(_W[p] for p, t in enumerate(top) if int(t) in rs) / (idcg + 1e-12)

def geom_answers(fold_rows):
    """concept -> POS/NEG geometric answer; fold_rows = [(sid, resid)] fold-in items."""
    ustar = enc_u([(Q[s], v) for s, v in fold_rows])
    prof = set(s for s, _ in fold_rows)
    cn = {}
    for s, v in fold_rows:
        for ki in item2c[s]: cn.setdefault(int(ki), []).append(s)
    ac = [c for c, js in cn.items() if len(js) >= 2]
    if not ac: return {}
    proj = ustar @ Ec[ac].T; thr = float(proj.mean())
    return {ac[k]: (POS if proj[k] > thr else NEG) for k in range(len(ac))}

def run(mode):
    curveF = {q: 0. for q in REVEALS}; curveT = {q: 0. for q in REVEALS}
    m = 0; mt = 0; ans_tok = {q: 0. for q in REVEALS}
    final_u = []; final_targets = []; final_excl = []; final_tailok = []
    for x in users:
        fold_rows = FOLD[x]; prof = set(s for s, _ in fold_rows)
        residmap = {s: (v - mu - bi[s]) for s, v in fold_rows}
        rel = list(TGT[x]); rel_t = [t for t in rel if not headmask[t]]; ht = 1 if rel_t else 0
        m += 1; mt += ht
        cans = geom_answers(fold_rows) if mode.startswith('conc') else None
        toks = []; asked = set(); ei = 0
        if mode == 'rand_item':
            order = list(rng.permutation(ITEMC))
        elif mode == 'pop_item' or mode == 'eig_item':
            order = list(ITEMC)
        else:
            order = list(CONC_POP_ORD)
        for q in REVEALS:
            if q > 0:
                if mode == 'eig_item':
                    cs = [j for j in ITEMC if j not in asked][:500]
                    if cs:
                        u0 = enc_u(toks); ca = np.array(cs)
                        p = sig(popb[ca] + Qp[ca] @ u0)
                        uP = enc_fold(toks, Q[ca], POS); uN = enc_fold(toks, Q[ca], NEG)
                        cov = p * sig(popb[R] + uP @ Qp[R].T).sum(1) + (1 - p) * sig(popb[R] + uN @ Qp[R].T).sum(1)
                        j = cs[int((p_seen[ca] * cov).argmax())]; asked.add(j)
                        if j in residmap: toks.append((Q[j], residmap[j]))
                elif mode == 'conc_eig':
                    cs = [c for c in range(nc) if c not in asked]
                    if cs:
                        u0 = enc_u(toks); ca = np.array(cs)
                        pc = sig(Ec[ca] @ u0)
                        uP = enc_fold(toks, Ec[ca], POS); uN = enc_fold(toks, Ec[ca], NEG)
                        cov = pc * sig(popb[R] + uP @ Qp[R].T).sum(1) + (1 - pc) * sig(popb[R] + uN @ Qp[R].T).sum(1)
                        c = cs[int((pe_ans[ca] * cov).argmax())]; asked.add(c)
                        if cans and c in cans: toks.append((Ec[c], cans[c]))
                elif mode.startswith('conc'):
                    e = None
                    while ei < len(order):
                        cand = order[ei]; ei += 1
                        if cand in asked: continue
                        e = cand; break
                    if e is not None:
                        asked.add(e)
                        if cans and e in cans: toks.append((Ec[e], cans[e]))
                else:   # pop_item / rand_item
                    e = None
                    while ei < len(order):
                        cand = order[ei]; ei += 1
                        if cand in asked: continue
                        e = cand; break
                    if e is not None:
                        asked.add(e)
                        if e in residmap: toks.append((Q[e], residmap[e]))
            ans_tok[q] += len(toks)
            u = enc_u(toks); excl = prof | asked
            vF = ndcg10(u, rel, excl, False)
            if vF is not None: curveF[q] += vF
            if ht:
                vT = ndcg10(u, rel, excl, True)
                if vT is not None: curveT[q] += vT
        final_u.append(u); final_targets.append(rel); final_excl.append(prof | asked); final_tailok.append(ht)
    cF = {q: curveF[q] / m for q in REVEALS}; cT = {q: curveT[q] / max(mt, 1) for q in REVEALS}
    at = {q: ans_tok[q] / m for q in REVEALS}
    return cF, cT, at, m, mt, (final_u, final_targets, final_excl, final_tailok)

# ---------- canonical-snap: full-profile fold NDCG@10 (warm reference the cold curve climbs toward) ----------
def full_profile():
    F = 0.; Tt = 0.; m = 0; mt = 0
    for x in users:
        fold_rows = FOLD[x]; prof = set(s for s, _ in fold_rows)
        u = enc_u([(Q[s], v - mu - bi[s]) for s, v in fold_rows])
        rel = list(TGT[x]); ht = 1 if any(not headmask[t] for t in rel) else 0
        vF = ndcg10(u, rel, prof, False)
        if vF is not None: F += vF; m += 1
        if ht:
            vT = ndcg10(u, rel, prof, True)
            if vT is not None: Tt += vT; mt += 1
    return F / m, Tt / max(mt, 1), m, mt

log("=== canonical-snap: full-profile fold ===")
fpF, fpT, fpm, fpmt = full_profile()
log(f"FULL-PROFILE fold NDCG@10: full={fpF:.4f} tail={fpT:.4f} (n={fpm}/{fpmt})")

RES = {'meta': {'n_test_users': len(users), 'nc': int(nc), 'ni': int(ni),
                'full_profile_full': fpF, 'full_profile_tail': fpT,
                'POS': POS, 'NEG': NEG, 'n_head': int(headmask.sum())},
       'curves': {}}
pop_final = None
for mode in MODES:
    log(f"--- {mode} ---")
    cF, cT, at, m, mt, fin = run(mode)
    RES['curves'][mode] = {'full': [cF[q] for q in REVEALS], 'tail': [cT[q] for q in REVEALS],
                           'ans_tok': [at[q] for q in REVEALS], 'n': m, 'n_tail': mt}
    log(f"  full  q0..q8: " + " ".join(f"{cF[q]:.4f}" for q in REVEALS))
    log(f"  tail  q0..q8: " + " ".join(f"{cT[q]:.4f}" for q in REVEALS))
    log(f"  ans_tok q8={at[T]:.2f}  n={m} n_tail={mt}")
    if mode == 'pop_item': pop_final = fin

# ---------- existential control: answer-permutation on pop_item ----------
if pop_final is not None:
    fu_, ft_, fe_, fok_ = pop_final
    perm = rng.permutation(len(fu_))
    sF = 0.; sT = 0.; m = 0; mt = 0
    for i in range(len(fu_)):
        u = fu_[perm[i]]                                  # partner's belief vs MY targets -> lift must vanish
        rel = ft_[i]; excl = fe_[i]
        vF = ndcg10(u, rel, excl, False)
        if vF is not None: sF += vF; m += 1
        if fok_[i]:
            vT = ndcg10(u, rel, excl, True)
            if vT is not None: sT += vT; mt += 1
    RES['control_shuffle_pop_item'] = {'full_q8': sF / m, 'tail_q8': sT / max(mt, 1)}
    log(f"CONTROL (answer-permutation on pop_item @q8): full={sF/m:.4f} tail={sT/max(mt,1):.4f} "
        f"(vs real q8 full={RES['curves']['pop_item']['full'][T]:.4f} tail={RES['curves']['pop_item']['tail'][T]:.4f})")

json.dump(RES, open(f'{RESD}/pb_results.json', 'w'), indent=2)
log(f"SAVED {RESD}/pb_results.json")
print("DONE")
