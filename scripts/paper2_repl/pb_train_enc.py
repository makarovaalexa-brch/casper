"""pb_train_enc.py -- STEP 2/3. FAITHFUL concept-aware reconstruction encoder (learned enc + Qp + Ec),
ported from scripts/_archive/paper2_old/freeze_concept_encoder.py to the canonical ML-25M ruler.

Architecture VERBATIM (freeze_concept_encoder.Enc): attention pool over revealed tokens
  token = [ feature (D) , residual (1) ] ,  feature = frozen Q_svd[item]  OR  learned Ec[concept]
  inp: Linear(D+1,128)-ReLU-Linear(128,128)-ReLU ; att: Linear(128,1) softmax ; val: Linear(128,D)
  u = sum_k softmax(att)_k * val(h_k)
Learned: enc params, Qp (decoder factors, init Q_svd), Ec (concept embeddings, init centroid Ac).
Decoder (training): sc = u @ Qp^T ; recon target = unrevealed like-set (r>=4), IPS-weighted BCE, masked seen.
Negation margin loss (CL=0.2, MARGIN=0.5): a POS item-token must score above the same item as NEG token.
Geometric-answer warmup: first GEO_START epochs use mean-residual concept answer, then GEOMETRIC
  (concept c -> POS if u*.Ec[c] > per-user mean, else NEG ; u* = fold of the user's full item profile) --
  consistent with the eval-time answer model. Genre tokens are DROPPED (the ML-25M port did not use them).

Reveal mix per user: item tokens + up to CTOK concept tokens, random count k in [1..K]; PCONLY fraction of
users are trained concept-only so pure-concept folds are in-distribution.
Best-checkpoint on the canonical 10k VAL cohort: full-profile fold-in NDCG@10 (full). Resumable (chunked).

Outputs -> .cache/paper2_repl/{enc.pt (best), Qp.npy, Ec.npy, enc_state.pt, enc_peak.txt}
Env: EP_CHUNK(3) EP_MAX(12) GEO_START(4) K(12) CTOK(6) PCONLY(0.35) BATCH(256) NVAL(2000) FRESH(0)
"""
import os, time, numpy as np, torch, torch.nn as nn

ROOT = 'C:/dev/phd/casper'; OUT = f'{ROOT}/.cache/paper2_repl'
torch.set_num_threads(os.cpu_count() or 4)
D = 64; LIKE = 4.0
K = int(os.environ.get('K', 12)); CTOK = int(os.environ.get('CTOK', 6))
PCONLY = float(os.environ.get('PCONLY', 0.35)); BATCH = int(os.environ.get('BATCH', 256))
EP_CHUNK = int(os.environ.get('EP_CHUNK', 3)); EP_MAX = int(os.environ.get('EP_MAX', 12))
GEO_START = int(os.environ.get('GEO_START', 4)); NVAL = int(os.environ.get('NVAL', 2000))
CL = 0.2; MARGIN = 0.5
rng = np.random.default_rng(0); torch.manual_seed(0)
t0 = time.time()
def log(m): print(f"[{time.time()-t0:6.0f}s] {m}", flush=True)

Z = np.load(f'{OUT}/pb_data.npz')
Q = Z['Q'].astype(np.float32); bi = Z['bi'].astype(np.float32); mu = float(Z['mu'])
cnt = Z['cnt']; ni = int(Z['ni']); ntr = int(Z['ntr'])
POS = float(Z['POS']); NEG = float(Z['NEG'])
Ac = Z['Ac'].astype(np.float32); nc = Ac.shape[0]
citems_flat = Z['citems_flat']; citems_off = Z['citems_off']; ccount = Z['ccount']
uu_s = Z['uu_s']; ii_s = Z['ii_s']; rr_s = Z['rr_s']; res_s = Z['res_s']; off = Z['off']
popb = np.log(cnt + 1.0).astype(np.float32)
Qt = torch.tensor(Q)
# item -> concept membership
item2c = [[] for _ in range(ni)]
for c in range(nc):
    for j in citems_flat[citems_off[c]:citems_off[c + 1]]: item2c[int(j)].append(c)
item2c = [np.array(v, np.int32) for v in item2c]
# concept "frequency" order for CTOK selection (# members)
CONC_ORD = list(np.argsort(-ccount))
ipsw = np.clip((1.0 / np.clip((cnt / max(cnt.max(), 1)) ** 0.5, 1e-3, 1.0)).astype(np.float32), None, 8.0)
lens = (off[1:] - off[:-1])
trbig = np.array([x for x in range(ntr) if lens[x] >= K + 1], dtype=np.int64)
log(f"nu_train={ntr} ni={ni} nc={nc} ; train users(>=K+1)={len(trbig)}")

# val cohort (canonical) -- per-user fold-in (item,resid) + target like-set
vfu = Z['vfold_uid']; vfs = Z['vfold_sid']; vfr = Z['vfold_rat']; vtu = Z['vtarget_uid']; vts = Z['vtarget_sid']
VFOLD = {}; VT = {}
for u, s, r in zip(vfu, vfs, vfr): VFOLD.setdefault(int(u), []).append((int(s), float(r) - mu - bi[int(s)]))
for u, s in zip(vtu, vts): VT.setdefault(int(u), set()).add(int(s))
vusers = [u for u in VFOLD if u in VT][:NVAL]
_Wv = 1.0 / np.log2(np.arange(2, 12))
log(f"val users for selection: {len(vusers)}")


class Enc(nn.Module):
    def __init__(s):
        super().__init__(); s.inp = nn.Sequential(nn.Linear(D + 1, 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU())
        s.att = nn.Linear(128, 1); s.val = nn.Linear(128, D)
    def forward(s, t, m):
        h = s.inp(t); a = s.att(h).squeeze(-1).masked_fill(m == 0, -1e9); al = torch.softmax(a, 1)
        return (al.unsqueeze(-1) * s.val(h)).sum(1)


enc = Enc(); Qp = torch.nn.Parameter(Qt.clone()); Ec = torch.nn.Parameter(torch.tensor(Ac))
opt = torch.optim.Adam(list(enc.parameters()) + [Qp, Ec], float(os.environ.get('LR', '1e-3')), weight_decay=1e-5)


def user_concepts(its, rss):   # concept -> mean residual over the user's >=2 rated tagged items
    acc = {}
    for j, r in zip(its, rss):
        for c in item2c[j]: acc.setdefault(int(c), []).append(r)
    return {c: float(np.mean(v)) for c, v in acc.items() if len(v) >= 2}


USE_GEO = False; GEO = {}
def enc_u_batch(rows_list):
    if not rows_list: return np.zeros((0, D), np.float32)
    mx = max(len(r) for r in rows_list)
    tk = np.zeros((len(rows_list), mx, D + 1), np.float32); mk = np.zeros((len(rows_list), mx), np.float32)
    for b, rows in enumerate(rows_list):
        for q, (f, v) in enumerate(rows): tk[b, q, :D] = f; tk[b, q, D] = v; mk[b, q] = 1
    with torch.no_grad(): return enc(torch.tensor(tk), torch.tensor(mk)).numpy()


def precompute_geo():
    """geometric answer target: like c iff u*.Ec[c] > per-user mean ; u* = fold of full item profile."""
    Ecd = Ec.detach().numpy(); G = {}
    for b0 in range(0, len(trbig), 1024):
        us = trbig[b0:b0 + 1024]; rows = []
        for x in us:
            a, c = off[x], off[x + 1]
            rows.append([(Q[ii_s[k]], res_s[k]) for k in range(a, c)])
        ustar = enc_u_batch(rows)
        for bi_, x in enumerate(us):
            a, c = off[x], off[x + 1]
            cans = user_concepts(ii_s[a:c].tolist(), res_s[a:c].tolist())
            cs = list(cans.keys())
            if not cs: G[int(x)] = {}; continue
            proj = ustar[bi_] @ Ecd[cs].T; thr = proj.mean()
            G[int(x)] = {cs[k]: (POS if proj[k] > thr else NEG) for k in range(len(cs))}
    return G


def make_batch(users, kk):
    W = kk + CTOK; B = len(users)
    F = np.zeros((B, W, D), np.float32); V = np.zeros((B, W), np.float32); msk = np.zeros((B, W), np.float32)
    tgt = np.zeros((B, ni), np.float32); wt = np.ones((B, ni), np.float32); seen = np.zeros((B, ni), bool)
    cpos = []
    for b, x in enumerate(users):
        a, c = off[x], off[x + 1]; ids = np.arange(a, c); rng.shuffle(ids); rev = ids[:kk]
        its = ii_s[rev]; rss = res_s[rev]
        conly = (rng.random() < PCONLY)
        if USE_GEO: cans = GEO.get(int(x), {})
        else: cans = user_concepts(ii_s[a:c].tolist(), res_s[a:c].tolist())
        cs = [cc for cc in CONC_ORD if cc in cans][:CTOK]
        L = 0
        if not conly:
            for q in range(len(its)): F[b, L] = Q[its[q]]; V[b, L] = rss[q]; msk[b, L] = 1; L += 1
            seen[b, its] = True
        for cc in cs:
            V[b, L] = cans[cc]; msk[b, L] = 1; cpos.append((b, L, cc)); L += 1   # concept feature injected from Ec
        if conly and L == 0:
            for q in range(len(its)): F[b, L] = Q[its[q]]; V[b, L] = rss[q]; msk[b, L] = 1; L += 1
            seen[b, its] = True
        allit = ii_s[a:c]; alll = allit[rr_s[a:c] >= LIKE]; unl = alll[~seen[b, alll]]
        if len(unl): tgt[b, unl] = 1.0; wt[b, unl] = ipsw[unl]
        posw = float(ipsw[unl].sum()) if len(unl) else 0.
        negmask = (~seen[b]) & (tgt[b] == 0); nneg = int(negmask.sum())
        if nneg > 0 and posw > 0: wt[b, negmask] = posw / nneg
    return (F, V, torch.tensor(msk), torch.tensor(tgt), torch.tensor(wt), torch.tensor(seen), cpos)


def val_ndcg10():
    enc.eval(); Qpd = Qp.detach().numpy(); tot = 0.; m = 0
    for b0 in range(0, len(vusers), 512):
        us = vusers[b0:b0 + 512]; rows = [VFOLD[x] for x in us]
        U = enc_u_batch([[(Q[s], v) for s, v in r] for r in rows])
        for bi_, x in enumerate(us):
            rel = VT[x]; prof = set(s for s, _ in VFOLD[x])
            sc = popb + Qpd @ U[bi_]; sc[list(prof)] = -1e9
            top = np.argpartition(-sc, 10)[:10]; top = top[np.argsort(-sc[top])]
            idcg = _Wv[:min(10, len(rel))].sum()
            tot += sum(_Wv[p] for p, tt in enumerate(top) if int(tt) in rel) / (idcg + 1e-12); m += 1
    enc.train(); return tot / max(m, 1), m


st = f'{OUT}/enc_state.pt'; bestf = f'{OUT}/enc.pt'; peakf = f'{OUT}/enc_peak.txt'
ep0 = 0; best = -1.0
if os.path.exists(st) and not os.environ.get('FRESH'):
    ck = torch.load(st); enc.load_state_dict(ck['enc']); opt.load_state_dict(ck['opt'])
    with torch.no_grad(): Qp.copy_(ck['Qp']); Ec.copy_(ck['Ec'])
    ep0 = ck['epoch']; best = ck['best']; log(f"[RESUME] ep{ep0} best-val@10={best:.4f}")

nrun = min(EP_CHUNK, EP_MAX - ep0)
if nrun <= 0:
    log(f"[DONE] {ep0}/{EP_MAX} epochs; best-val@10={best:.4f}")
else:
    for e in range(nrun):
        ep = ep0 + e
        globals()['USE_GEO'] = ep >= GEO_START
        if USE_GEO:
            log(f"ep{ep+1}: precomputing geometric answers")
            globals()['GEO'] = precompute_geo()
        rng.shuffle(trbig); tl = 0.; nb = 0
        for b0 in range(0, len(trbig), BATCH):
            us = trbig[b0:b0 + BATCH]; kk = int(rng.integers(2, K + 1))
            F, V, m, tg, w, se, cpos = make_batch(us, kk)
            feat = torch.tensor(F)
            if cpos:
                cf = torch.zeros(F.shape[0], F.shape[1], D)
                bidx = torch.tensor([p[0] for p in cpos]); qidx = torch.tensor([p[1] for p in cpos]); cidx = torch.tensor([p[2] for p in cpos])
                cf[bidx, qidx] = Ec[cidx]; feat = feat + cf
            tok = torch.cat([feat, torch.tensor(V).unsqueeze(-1)], dim=-1)
            u = enc(tok, m); sc = u @ Qp.t()
            loss = (w * nn.functional.binary_cross_entropy_with_logits(sc, tg, reduction='none')).masked_fill(se, 0.).mean()
            jt = torch.randint(0, ni, (F.shape[0],))
            tp = torch.zeros(F.shape[0], 1, D + 1); tp[:, 0, :D] = Qt[jt]; tp[:, 0, D] = POS
            tm = torch.zeros(F.shape[0], 1, D + 1); tm[:, 0, :D] = Qt[jt]; tm[:, 0, D] = NEG
            up = enc(tp, torch.ones(F.shape[0], 1)); um = enc(tm, torch.ones(F.shape[0], 1))
            diff = (Qp[jt] * up).sum(1) - (Qp[jt] * um).sum(1)
            loss = loss + CL * torch.nn.functional.softplus(MARGIN - diff).mean()
            opt.zero_grad(); loss.backward(); opt.step(); tl += loss.item(); nb += 1
        vN, vm = val_ndcg10(); improved = vN > best
        if improved:
            best = vN; torch.save(enc.state_dict(), bestf)
            np.save(f'{OUT}/Qp.npy', Qp.detach().numpy()); np.save(f'{OUT}/Ec.npy', Ec.detach().numpy())
        torch.save({'enc': enc.state_dict(), 'opt': opt.state_dict(), 'Qp': Qp.detach().clone(),
                    'Ec': Ec.detach().clone(), 'epoch': ep + 1, 'best': best}, st)
        with open(peakf, 'a') as f:
            f.write(f"ep{ep+1} loss={tl/nb:.4f} val_ndcg10={vN:.4f}(n={vm}) geo={int(USE_GEO)} best={best:.4f}{' *SAVED' if improved else ''}\n")
        log(f"ep{ep+1}/{EP_MAX} loss={tl/nb:.4f} val_ndcg10={vN:.4f}(n={vm}) geo={int(USE_GEO)} best={best:.4f}{' *SAVED' if improved else ''}")
    log(f"[CHUNK DONE] now {ep0+nrun}/{EP_MAX}; best-val@10={best:.4f}")
print("DONE")
