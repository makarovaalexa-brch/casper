"""pb_interview.py -- STEP 3/3. Short-interview battery on the canonical ML-25M ruler, run under TWO
answer models on the SAME trained encoder (author ruling 2026-07-25, Fable HIGH conf):

  BLOCK A  GEOMETRIC answers (u*.Ec vs per-user mean -> POS/NEG) -- faithful reproduction of the OLD
           +36% setting, but a CIRCULAR measurement (shared-representation self-preference). Label:
           "simulator-sensitivity / privileged (recommender-geometry), UNCITABLE".
  BLOCK B  BEHAVIORAL answers -- the honest, recommender-INDEPENDENT signed-SEL construction
           (src/instrument/signed_answers.py): v = clip(NPMI + w_val*VAL, -1, 1) from the fold-in
           history (NPMI watch-lift over popularity base-rate + shrunk residual over TRAIN item means),
           four bands (like/meh/dislike/refuse), C_NEG per-user negative cap. This is the HEADLINE.

Both blocks fold the SAME trained concept embedding Ec[c]; only the token VALUE differs -> the delta
between them IS the demonstration that the answer model drives concept success. Everything else
(selectors static-pop + EIG, answer-model-agnostic; controls; leak; canonical-snap) unchanged.
Item-asking arms are answer-model-independent (item answer = real rating residual) -> computed ONCE.

Decoder = method's own score = popb + Qp@u. Ruler = canonical full+tail NDCG@10, head mask from proc
train mass, fold-in masked, targets=test_te. All 10k test users, all items, all 1031 concepts.

Outputs -> experiments/paper2_repl/pb_results.json
"""
import os, sys, time, json, numpy as np, torch, torch.nn as nn
from scipy import sparse

ROOT = 'C:/dev/phd/casper'; OUT = f'{ROOT}/.cache/paper2_repl'; RESD = f'{ROOT}/experiments/paper2_repl'
sys.path.insert(0, f'{ROOT}/src/instrument')
from signed_answers import _components, TAU_REF, LAM_VAL, C_NEG, W_VAL_TARGET_P90
os.makedirs(RESD, exist_ok=True)
torch.set_num_threads(os.cpu_count() or 4); D = 64; T = 8
NEVAL = int(os.environ.get('NEVAL', 10000))
REVEALS = list(range(T + 1))
rng = np.random.default_rng(0); t0 = time.time()
def log(m): print(f"[{time.time()-t0:6.0f}s] {m}", flush=True)

Z = np.load(f'{OUT}/pb_data.npz')
Q = Z['Q'].astype(np.float32); bi = Z['bi'].astype(np.float32); mu = float(Z['mu'])
cnt = Z['cnt']; ni = int(Z['ni']); ntr = int(Z['ntr'])
POS = float(Z['POS']); NEG = float(Z['NEG'])
ccount = Z['ccount']; citems_flat = Z['citems_flat']; citems_off = Z['citems_off']
headmask = Z['headmask'].astype(bool)
uu_s = Z['uu_s']; ii_s = Z['ii_s']; rr_s = Z['rr_s']; off = Z['off']
popb = np.log(cnt + 1.0).astype(np.float32)
Qp = np.load(f'{OUT}/Qp.npy').astype(np.float32)
Ec = np.load(f'{OUT}/Ec.npy').astype(np.float32); nc = Ec.shape[0]
order_pop = np.argsort(-cnt)
ITEMC = list(order_pop[:1500]); R = np.array(order_pop[:500]); CONC_POP_ORD = list(np.argsort(-ccount))
item2c = [[] for _ in range(ni)]
for c in range(nc):
    for j in citems_flat[citems_off[c]:citems_off[c + 1]]: item2c[int(j)].append(c)
item2c = [np.array(v, np.int32) for v in item2c]

# ---------- behavioral (signed-SEL) prereg: item_mean, pexp, Mm, w_val/t_like/t_neg (recommender-independent) ----------
rows = np.concatenate([citems_flat[citems_off[c]:citems_off[c + 1]] for c in range(nc)]).astype(np.int64)
cols = np.concatenate([np.full(citems_off[c + 1] - citems_off[c], c) for c in range(nc)]).astype(np.int64)
Mm = sparse.csr_matrix((np.ones(len(rows), np.float32), (rows, cols)), shape=(ni, nc))
pexp = np.asarray((cnt / cnt.sum()) @ Mm).ravel().astype(np.float32)
imm = np.zeros(ni); ctt = np.zeros(ni)
np.add.at(imm, ii_s, rr_s); np.add.at(ctt, ii_s, 1.0)
item_mean = np.where(ctt > 0, imm / np.maximum(ctt, 1), rr_s.mean()).astype(np.float32)

PREREG_CACHE = f'{OUT}/signed_prereg.json'; PEANS_CACHE = f'{OUT}/pe_ans.npy'
# CSR (train user -> items/stars) built ONCE; reused for prereg + pe_ans (vectorized via Xb@Mm)
if os.path.exists(PREREG_CACHE) and os.path.exists(PEANS_CACHE):
    prereg = json.load(open(PREREG_CACHE)); pe_cnt = np.load(PEANS_CACHE)
    log(f"loaded behavioral prereg w_val={prereg['w_val']:.4f} + pe_ans cache")
else:
    log("computing behavioral signed-SEL prereg + pe_ans over train users (chunked, vectorized)")
    np_s = []; va_s = []; CH = 20000; pe_cnt = np.zeros(nc)
    for st in range(0, ntr, CH):
        us = list(range(st, min(st + CH, ntr)))
        rws = []; cls = []; bd = []; rd = []
        for r, x in enumerate(us):
            a, c = off[x], off[x + 1]; its = ii_s[a:c]; sts = rr_s[a:c]
            rws.extend([r] * len(its)); cls.extend(its.tolist()); bd.extend([1.0] * len(its))
            rd.extend((sts - item_mean[its]).tolist())
        Xb = sparse.csr_matrix((np.asarray(bd, np.float32), (rws, cls)), shape=(len(us), ni))
        Xr = sparse.csr_matrix((np.asarray(rd, np.float32), (rws, cls)), shape=(len(us), ni))
        SEL, NPMI, VAL, E, ncell = _components(Xb, Xr, Mm, pexp)
        pe_cnt += (ncell >= 2).sum(0)                          # answerability prior (>=2 members), vectorized
        m = (ncell >= 2) & (E >= TAU_REF)
        np_s.append(NPMI[m].astype(np.float32)); va_s.append(VAL[m].astype(np.float32))
    NP = np.concatenate(np_s); VA = np.concatenate(va_s)
    w_val = float(W_VAL_TARGET_P90 / max(float(np.percentile(np.abs(VA), 90)), 1e-6))
    vraw = NP + w_val * VA; pos = vraw[vraw > 0]
    prereg = {'w_val': w_val, 't_like_p60pos': float(np.percentile(pos, 60)) if len(pos) else 0.2,
              't_neg_absp25': float(abs(np.percentile(vraw, 25))), 'C_NEG': C_NEG,
              'source': f'{ntr} train users, {int(len(NP))} answerable cells'}
    json.dump(prereg, open(PREREG_CACHE, 'w'), indent=2); np.save(PEANS_CACHE, pe_cnt)
    log(f"prereg: w_val={prereg['w_val']:.4f} t_like={prereg['t_like_p60pos']:.4f} t_neg={prereg['t_neg_absp25']:.4f}")

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
def enc_fold(base_rows, cand_feats, ansval):
    L = len(base_rows); M = cand_feats.shape[0]
    tk = np.zeros((M, L + 1, D + 1), np.float32); mk = np.ones((M, L + 1), np.float32)
    if L:
        base = np.zeros((L, D + 1), np.float32)
        for q, (f, v) in enumerate(base_rows): base[q, :D] = f; base[q, D] = v
        tk[:, :L, :] = base[None]
    tk[:, L, :D] = cand_feats; tk[:, L, D] = ansval
    with torch.no_grad(): return enc(torch.tensor(tk), torch.tensor(mk)).numpy()

# concept answerability prior pe_ans (from the vectorized pe_cnt above)
pe_ans = (pe_cnt / max(ntr, 1)).astype(np.float32) + 1e-6
p_seen = (cnt / max(ntr, 1)).astype(np.float32)
log(f"pe_ans ready; concepts>=0.5 answerable: {(pe_ans>=0.5).sum()}/{nc}")

# ---------- test users ----------
fu = Z['foldin_uid']; fs = Z['foldin_sid']; fr = Z['foldin_rat']; tu = Z['target_uid']; ts = Z['target_sid']
FOLD = {}; TGT = {}
for u, s, r in zip(fu, fs, fr): FOLD.setdefault(int(u), []).append((int(s), float(r)))
for u, s in zip(tu, ts): TGT.setdefault(int(u), set()).add(int(s))
users = [u for u in FOLD if u in TGT][:NEVAL]
log(f"test users: {len(users)}")

_W = 1.0 / np.log2(np.arange(2, 12))
def ndcg10(u, rel, excl, tail):
    sc = (popb + Qp @ u).astype(np.float64).copy(); sc[list(excl)] = -1e30
    if tail: sc[headmask] = -1e30; rel = [t for t in rel if not headmask[t]]
    if not rel: return None
    top = np.argpartition(-sc, 10)[:10]; top = top[np.argsort(-sc[top])]
    rs = set(rel); idcg = _W[:min(10, len(rel))].sum()
    return sum(_W[p] for p, t in enumerate(top) if int(t) in rs) / (idcg + 1e-12)

# ---------- answer models (concept c -> folded value; None=un-askable, 'REFUSE'=burn turn no fold) ----------
def geom_concept_answers(fold_rows):
    """BLOCK A (uncitable): POS/NEG by u*.Ec vs per-user mean. askable iff >=2 members."""
    ustar = enc_u([(Q[s], v - mu - bi[s]) for s, v in fold_rows])
    cn = {}
    for s, v in fold_rows:
        for ki in item2c[s]: cn.setdefault(int(ki), []).append(s)
    ac = [c for c, js in cn.items() if len(js) >= 2]
    if not ac: return {}
    proj = ustar @ Ec[ac].T; thr = float(proj.mean())
    return {ac[k]: (POS if proj[k] > thr else NEG) for k in range(len(ac))}

def behav_concept_answers(fold_rows):
    """BLOCK B (headline, recommender-independent): signed-SEL value per concept from fold-in history.
    returns {c: value} for askable (nc>=2) concepts; value=0 for meh; 'REFUSE' for E<TAU (burn turn)."""
    sids = np.array([s for s, _ in fold_rows], np.int64); stars = np.array([v for _, v in fold_rows], np.float32)
    sub = Mm[sids]
    ncell = np.asarray(sub.sum(axis=0)).ravel().astype(np.float32)
    res = stars - item_mean[sids]
    rs = np.asarray(sub.T @ res).ravel().astype(np.float32)
    nu = float(len(sids)); E = (nu * pexp).astype(np.float32)
    SEL = np.log2((ncell + 0.5) / (E + 0.5))
    NPMI = SEL / np.maximum(-np.log2((ncell + 0.5) / (nu + 1.0)), 1e-6)
    mr = np.divide(rs, ncell, out=np.zeros_like(rs), where=ncell > 0)
    VAL = (ncell / (ncell + LAM_VAL)) * mr
    V = np.clip(NPMI + prereg['w_val'] * VAL, -1.0, 1.0).astype(np.float32)
    tl = prereg['t_like_p60pos']; tn = prereg['t_neg_absp25']
    ans = {}
    for c in np.where(ncell >= 2)[0]:
        c = int(c)
        if E[c] < TAU_REF: ans[c] = 'REFUSE'; continue
        v = float(V[c])
        if -tn <= v <= tl: v = 0.0
        ans[c] = v
    return ans

def cap_neg_tokens(concept_vals):
    """C_NEG per-user negative-channel cap over revealed concept values (behavioral block only)."""
    s = sum(abs(v) for v in concept_vals if v < 0)
    if s <= C_NEG: return list(concept_vals)
    f = C_NEG / max(s, 1e-9)
    return [v * f if v < 0 else v for v in concept_vals]

# ---------- item arm (answer-model-independent) ----------
def run_item(mode):
    cF = {q: 0. for q in REVEALS}; cT = {q: 0. for q in REVEALS}; at = {q: 0. for q in REVEALS}
    m = 0; mt = 0; final = []
    for x in users:
        fold_rows = FOLD[x]; prof = set(s for s, _ in fold_rows)
        residmap = {s: (v - mu - bi[s]) for s, v in fold_rows}
        rel = list(TGT[x]); ht = 1 if any(not headmask[t] for t in rel) else 0
        m += 1; mt += ht
        toks = []; asked = set(); ei = 0
        order = list(rng.permutation(ITEMC)) if mode == 'rand_item' else list(ITEMC)
        for q in REVEALS:
            if q > 0:
                if mode == 'eig_item':
                    cs = [j for j in ITEMC if j not in asked][:500]
                    if cs:
                        u0 = enc_u(toks); ca = np.array(cs); p = sig(popb[ca] + Qp[ca] @ u0)
                        uP = enc_fold(toks, Q[ca], POS); uN = enc_fold(toks, Q[ca], NEG)
                        cov = p * sig(popb[R] + uP @ Qp[R].T).sum(1) + (1 - p) * sig(popb[R] + uN @ Qp[R].T).sum(1)
                        j = cs[int((p_seen[ca] * cov).argmax())]; asked.add(j)
                        if j in residmap: toks.append((Q[j], residmap[j]))
                else:
                    e = None
                    while ei < len(order):
                        cand = order[ei]; ei += 1
                        if cand in asked: continue
                        e = cand; break
                    if e is not None:
                        asked.add(e)
                        if e in residmap: toks.append((Q[e], residmap[e]))
            at[q] += len(toks); u = enc_u(toks); excl = prof | asked
            vF = ndcg10(u, rel, excl, False)
            if vF is not None: cF[q] += vF
            if ht:
                vT = ndcg10(u, rel, excl, True)
                if vT is not None: cT[q] += vT
        final.append((u, rel, prof | asked, ht))
    return ({q: cF[q] / m for q in REVEALS}, {q: cT[q] / max(mt, 1) for q in REVEALS},
            {q: at[q] / m for q in REVEALS}, m, mt, final)

# ---------- concept arm under a given answer model ----------
def run_concept(mode, model):
    cF = {q: 0. for q in REVEALS}; cT = {q: 0. for q in REVEALS}; at = {q: 0. for q in REVEALS}
    m = 0; mt = 0
    for x in users:
        fold_rows = FOLD[x]; prof = set(s for s, _ in fold_rows)
        rel = list(TGT[x]); ht = 1 if any(not headmask[t] for t in rel) else 0
        m += 1; mt += ht
        cans = geom_concept_answers(fold_rows) if model == 'geom' else behav_concept_answers(fold_rows)
        ctoks = []; asked = set(); ei = 0
        order = list(CONC_POP_ORD)
        for q in REVEALS:
            if q > 0:
                if mode == 'conc_eig':
                    cs = [c for c in range(nc) if c not in asked]
                    if cs:
                        base = [(Ec[c], v) for c, v in ctoks]
                        u0 = enc_u(base); ca = np.array(cs); pc = sig(Ec[ca] @ u0)
                        uP = enc_fold(base, Ec[ca], POS); uN = enc_fold(base, Ec[ca], NEG)
                        cov = pc * sig(popb[R] + uP @ Qp[R].T).sum(1) + (1 - pc) * sig(popb[R] + uN @ Qp[R].T).sum(1)
                        c = cs[int((pe_ans[ca] * cov).argmax())]; asked.add(c)
                        a = cans.get(c)
                        if a is not None and a != 'REFUSE': ctoks.append((c, float(a)))
                else:
                    e = None
                    while ei < len(order):
                        cand = order[ei]; ei += 1
                        if cand in asked: continue
                        e = cand; break
                    if e is not None:
                        asked.add(e); a = cans.get(e)
                        if a is not None and a != 'REFUSE': ctoks.append((e, float(a)))
            if model == 'behav' and ctoks:
                vals = cap_neg_tokens([v for _, v in ctoks]); base = [(Ec[ctoks[k][0]], vals[k]) for k in range(len(ctoks))]
            else:
                base = [(Ec[c], v) for c, v in ctoks]
            at[q] += len(ctoks); u = enc_u(base)
            vF = ndcg10(u, rel, prof, False)
            if vF is not None: cF[q] += vF
            if ht:
                vT = ndcg10(u, rel, prof, True)
                if vT is not None: cT[q] += vT
    return ({q: cF[q] / m for q in REVEALS}, {q: cT[q] / max(mt, 1) for q in REVEALS},
            {q: at[q] / m for q in REVEALS}, m, mt)

# ---------- canonical-snap: full-profile fold ----------
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
    return F / m, Tt / max(mt, 1), m

log("=== canonical-snap: full-profile fold ===")
fpF, fpT, fpm = full_profile()
log(f"FULL-PROFILE fold NDCG@10: full={fpF:.4f} tail={fpT:.4f} (n={fpm})")

RES = {'meta': {'n_test_users': len(users), 'nc': int(nc), 'ni': int(ni),
                'full_profile_full': fpF, 'full_profile_tail': fpT, 'n_head': int(headmask.sum()),
                'POS': POS, 'NEG': NEG, 'prereg': prereg},
       'item': {}, 'geom': {}, 'behav': {}}

pop_final = None
for mode in ['pop_item', 'eig_item', 'rand_item']:
    log(f"--- item: {mode} ---")
    cF, cT, at, m, mt, fin = run_item(mode)
    RES['item'][mode] = {'full': [cF[q] for q in REVEALS], 'tail': [cT[q] for q in REVEALS],
                         'ans_tok': [at[q] for q in REVEALS], 'n': m, 'n_tail': mt}
    log(f"  full q0..q8: " + " ".join(f"{cF[q]:.4f}" for q in REVEALS))
    log(f"  tail q0..q8: " + " ".join(f"{cT[q]:.4f}" for q in REVEALS))
    if mode == 'pop_item': pop_final = fin

for model in ['geom', 'behav']:
    for mode in ['conc_pop', 'conc_eig']:
        log(f"--- {model}: {mode} ---")
        cF, cT, at, m, mt = run_concept(mode, model)
        RES[model][mode] = {'full': [cF[q] for q in REVEALS], 'tail': [cT[q] for q in REVEALS],
                            'ans_tok': [at[q] for q in REVEALS], 'n': m, 'n_tail': mt}
        log(f"  full q0..q8: " + " ".join(f"{cF[q]:.4f}" for q in REVEALS))
        log(f"  tail q0..q8: " + " ".join(f"{cT[q]:.4f}" for q in REVEALS))

# ---------- existential control: answer-permutation on pop_item ----------
if pop_final is not None:
    perm = rng.permutation(len(pop_final))
    sF = 0.; sT = 0.; m = 0; mt = 0
    for i in range(len(pop_final)):
        u = pop_final[perm[i]][0]; _, rel, excl, ht = pop_final[i]
        vF = ndcg10(u, rel, excl, False)
        if vF is not None: sF += vF; m += 1
        if ht:
            vT = ndcg10(u, rel, excl, True)
            if vT is not None: sT += vT; mt += 1
    RES['control_shuffle_pop_item'] = {'full_q8': sF / m, 'tail_q8': sT / max(mt, 1)}
    log(f"CONTROL (answer-permutation pop_item @q8): full={sF/m:.4f} tail={sT/max(mt,1):.4f} "
        f"(vs real q8 full={RES['item']['pop_item']['full'][T]:.4f})")

json.dump(RES, open(f'{RESD}/pb_results.json', 'w'), indent=2)
log(f"SAVED {RESD}/pb_results.json")
print("DONE")
