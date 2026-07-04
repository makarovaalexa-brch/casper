"""
p4b_battery.py -- INSTRUMENT 2.0 Phase 4b (stretch): THE CROSS-DOMAIN POLICY BATTERY on the
certified RecVAE-d512 Goodreads composite instrument. Mirrors p4a_battery.py (ML-1M) and finally
tests the PRE-REGISTERED Goodreads predictions (experiments/paper2/PREREG_GOODREADS_PREDICTIONS.md).

THE cross-domain question: does the flagship (continuous + graded + adaptive elicitation) hold on
BOOKS, on a certified EASE-class instrument (RecVAE-d512, 0.78x EASE headroom)?

Instrument: .cache/instrument2/gr_recvae_d512_best.pt (frozen, P2 ckpt).
Arena: gr_recvae.load_arena() = phase-1E composite split, va=500 / te=500, restricted top-20k universe
       (where EASE-comparability holds); full-catalogue reported separately where stated.
Belief operator (P3 winner): additive z' = z + eta*a*q, z0=0 cold seed.
  eta = GR mean ||z*|| = 23  (measured; mirrors ML-1M's eta=16 ~ mean||z*||=17.2 scaling rule).
Graded answer: a = cos(z*, q), z* = enc(profile likes in universe).
Concept dirs = member-bag encodes of the shelf vocab (concepts_comp, >=20 members, pop-weighted top-50).
Metric: NDCG@510 (primary) + NDCG@10, Cremonesi head-33% tail. T=8 turns. Fixed rng(123) split
(single arena, NOT seed-averaged: the GR protocol uses one held-out split). Actor: 3 TRAIN seeds.

Stages:
  static  -- item 1: MOSTPOP / z0 floor; entropy-graded over shelf concepts (lift-selected per-user
             answerable vocab); decoder-SVD basis-8 (graded+binary); item-8 fold; concept-8 lift
             (graded+binary); concept-only k-curve (PREREG-1 saturation); effranks (PREREG-3 rank).
  train   -- item 2: D1-recipe differentiable-unroll actor; BC-warm from decoder-SVD-8; unroll
             fine-tune; val-best (va tail). Saves .cache/instrument2/p4b_actor_s{seed}.pt
  eval    -- items 2-5: actor test eval (3 train seeds) + binary; q-curve; snap-loss (concept +
             item-decoder-row); static-8 greedy control; paired per-user bootstrap; PREREG verdicts.

Usage:
  python scripts/instrument2/p4b_battery.py static
  python scripts/instrument2/p4b_battery.py train --tseed 0
  python scripts/instrument2/p4b_battery.py eval
"""
import os, sys, json, time, argparse
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gr_recvae as G
from recvae import RecVAE

DEVICE = torch.device('cpu'); torch.set_num_threads(4)
CK = '.cache/instrument2'
CKPT = f'{CK}/gr_recvae_d512_best.pt'
NU = G.NUNIV                     # 20000 restricted universe
LIKE = G.LIKE
KP = G.KP                        # 510 primary
ETA = 23.0                       # GR mean ||z*|| (measured, see docstring)
T = 8
MBAG = 50
NTR_SUB = 15000                  # train-user subsample for actor / PCA (CPU budget)
OUTJSON = f'{CK}/p4b_battery.json'
_ARENA = None


# ------------------------------------------------------------------ arena / model
def arena():
    global _ARENA
    if _ARENA is None:
        _ARENA = G.load_arena()
    return _ARENA


def load_model():
    blob = torch.load(CKPT, map_location=DEVICE)
    a = blob['args']
    m = RecVAE(a['hidden'], a['latent'], NU).to(DEVICE)
    m.load_state_dict(blob['model']); m.eval()
    return m, a['latent']


def enc_mu(model, Xd):
    with torch.no_grad():
        mu, _ = model.encoder(torch.tensor(np.asarray(Xd, np.float32)), dropout_rate=0.0)
    return mu.numpy()


def decode_np(model, Z):
    with torch.no_grad():
        return model.decoder(torch.tensor(np.asarray(Z, np.float32))).numpy().astype(np.float64)


def loc_map(ar):
    loc = -np.ones(ar['ni'], np.int64); loc[ar['univ']] = np.arange(NU)
    return loc


def head_local(ar):
    return set(np.where(ar['headmask'][ar['univ']])[0].tolist())


# ------------------------------------------------------------------ cohorts
def build_cohort(ar, loc, users):
    """Per-user {x, prof_like(local), rel_local, prof_local}. Mirrors gr_recvae eval selection."""
    out = []
    for x in users:
        if x not in ar['SPL']:
            continue
        rd = ar['rat_by_u'][x]; tst = ar['SPL'][x]
        prof = [j for j in rd if (j not in tst) and ar['umask'][j]]
        rel = [t for t in tst if ar['umask'][t]]
        prof_like = [int(loc[j]) for j in prof if rd[j] >= LIKE]
        rel_local = [int(loc[t]) for t in rel]
        prof_local = [int(loc[j]) for j in prof]
        if len(rel) < 1 or len(prof) < 1:
            continue
        out.append(dict(x=x, prof_like=prof_like, rel_local=rel_local, prof_local=prof_local))
    return out


def onehot(items, ni=NU):
    x = np.zeros((1, ni), np.float32)
    if len(items): x[0, list(items)] = 1.0
    return x


def zstar_of_cohort(model, coh):
    d = model.decoder.in_features
    Z = np.zeros((len(coh), d), np.float32)
    for i, c in enumerate(coh):
        if c['prof_like']:
            Z[i] = enc_mu(model, onehot(c['prof_like'])[0][None, :])[0]
    return Z


# ------------------------------------------------------------------ static direction designs
def decoder_svd_dirs(model, K):
    W = model.decoder.weight.detach().numpy()
    _, _, Vt = np.linalg.svd(W - W.mean(0, keepdims=True), full_matrices=False)
    return Vt[:K].astype(np.float32)


def train_zstar_subsample(model, ar, loc, nsub=NTR_SUB, seed=0):
    """Full-profile-like z* for a subsample of train users (>=4 likes in universe)."""
    uu, ii, rr = ar['uu'], ar['ii'], ar['rr']
    trset = np.zeros(ar['nu'], bool); trset[ar['trU']] = True
    mask = (rr >= LIKE) & ar['umask'][ii] & trset[uu]
    ru = uu[mask]; rc = loc[ii[mask]]
    order = np.argsort(ru, kind='stable')
    ru = ru[order]; rc = rc[order]
    bnd = np.searchsorted(ru, np.unique(ru), side='left')
    users = np.unique(ru); starts = bnd; ends = np.r_[bnd[1:], len(ru)]
    rng = np.random.default_rng(seed)
    keep = np.arange(len(users))
    # keep users with >=4 likes
    good = (ends - starts) >= 4
    keep = keep[good]
    if len(keep) > nsub:
        keep = np.sort(rng.choice(keep, nsub, replace=False))
    d = model.decoder.in_features
    Z = np.zeros((len(keep), d), np.float32)
    B = 500
    for st in range(0, len(keep), B):
        ch = keep[st:st + B]
        Xd = np.zeros((len(ch), NU), np.float32)
        for r, ui in enumerate(ch):
            cols = rc[starts[ui]:ends[ui]]
            Xd[r, cols] = 1.0
        Z[st:st + len(ch)] = enc_mu(model, Xd)
    return Z


def pca_zstar_dirs(Ztr, K):
    Zc = Ztr - Ztr.mean(0, keepdims=True)
    _, _, Vt = np.linalg.svd(Zc, full_matrices=False)
    return Vt[:K].astype(np.float32)


def concept_dirs(model, ar, loc):
    """member-bag encode concept directions (P3 W2 winner) for the shelf vocab; lift metadata.
    Returns Q (nshelf,d) unit dirs (invalid shelves = 0 row), tag_members(local), tag_ok, names,
    memb_of_item (item-local -> list of valid shelves), msize."""
    Cc = np.load('.cache/goodreads/concepts_comp.npz', allow_pickle=True)
    ctags = Cc['ctags']; off = Cc['citems_off']; flat = Cc['citems_flat']
    popb = ar['popb'][ar['univ']]
    nsh = len(ctags); d = model.decoder.in_features
    Q = np.zeros((nsh, d), np.float32)
    tag_members = []; tag_ok = np.zeros(nsh, bool); msize = np.zeros(nsh)
    for t in range(nsh):
        mem = flat[off[t]:off[t + 1]]; ml = loc[mem]; ml = ml[ml >= 0]
        tag_members.append(ml); msize[t] = len(ml)
        if len(ml) < 20:
            continue
        tag_ok[t] = True
        w = popb[ml]; w = w / (w.sum() + 1e-9)
        if len(ml) > MBAG:
            k2 = np.argsort(-w)[:MBAG]; ml2 = ml[k2]; w = w[k2]; w = w / (w.sum() + 1e-9)
        else:
            ml2 = ml
        b = np.zeros(NU, np.float32); b[ml2] = w
        z = enc_mu(model, b[None, :])[0]; Q[t] = z / (np.linalg.norm(z) + 1e-9)
    memb_of_item = [[] for _ in range(NU)]
    for t in range(nsh):
        if tag_ok[t]:
            for il in tag_members[t]:
                memb_of_item[il].append(t)
    return Q, tag_ok, np.array([str(x) for x in ctags]), memb_of_item, msize


def user_lift(coh_user, memb_of_item, msize, tag_ok):
    aff = np.zeros(len(msize))
    for il in coh_user['prof_like']:
        for t in memb_of_item[il]:
            aff[t] += 1
    lift = np.where(tag_ok, aff / (msize + 1e-9), -1.0)
    return lift, aff


# ------------------------------------------------------------------ operator + metric
def unroll_static(zstar, Q8, eta=ETA, binary=False):
    d = Q8.shape[1]; z = np.zeros(d, np.float32); nz = np.linalg.norm(zstar) + 1e-9
    for t in range(Q8.shape[0]):
        a = float(zstar @ Q8[t] / nz)
        if binary: a = float(np.sign(a))
        z = z + eta * a * Q8[t]
    return z


def eval_builder(model, ar, coh, hl, build_z, ks=(KP, 10)):
    """build_z(i, c, zs) -> final z. Returns dict of ndcg means: {'full510','tail510','full10'}."""
    acc = {f'full{KP}': 0.0, f'tail{KP}': 0.0, 'full10': 0.0}
    nf = nt = n10 = 0
    for i, c in enumerate(coh):
        z = build_z(i, c, None)
        s = decode_np(model, z[None, :])[0]
        vf = G.ndcg(s, c['rel_local'], c['prof_local'], hl, KP, False)
        vt = G.ndcg(s, c['rel_local'], c['prof_local'], hl, KP, True)
        v10 = G.ndcg(s, c['rel_local'], c['prof_local'], hl, 10, False)
        if vf is not None: acc[f'full{KP}'] += vf; nf += 1
        if vt is not None: acc[f'tail{KP}'] += vt; nt += 1
        if v10 is not None: acc['full10'] += v10; n10 += 1
    return {f'full{KP}': acc[f'full{KP}'] / max(nf, 1), f'tail{KP}': acc[f'tail{KP}'] / max(nt, 1),
            'full10': acc['full10'] / max(n10, 1)}


def PR(Mv):
    Mn = Mv / (np.linalg.norm(Mv, axis=1, keepdims=True) + 1e-9)
    C = Mn.T @ Mn / Mn.shape[0]; w = np.clip(np.linalg.eigvalsh(C)[::-1], 0, None)
    return float((w.sum() ** 2) / (np.sum(w * w) + 1e-12))


# ================================================================== STAGE static
def stage_static():
    model, d = load_model(); ar = arena(); loc = loc_map(ar); hl = head_local(ar)
    coh = build_cohort(ar, loc, ar['te'].tolist())
    print(f'[static] test cohort n={len(coh)}', flush=True)
    Zst = zstar_of_cohort(model, coh)
    zsof = lambda i: Zst[i]
    print(f'[static] mean ||z*|| = {np.linalg.norm(Zst, axis=1).mean():.3f}  (eta={ETA})', flush=True)

    svd8 = decoder_svd_dirs(model, 8)
    Qc, tag_ok, names, memb_of_item, msize = concept_dirs(model, ar, loc)
    popb_local = ar['popb'][ar['univ']].astype(np.float64)

    # global entropy ordering over concepts (documented control)
    valid = np.where(tag_ok)[0]
    Sc = decode_np(model, Qc[valid]); Pc = np.exp(Sc - Sc.max(1, keepdims=True)); Pc /= Pc.sum(1, keepdims=True)
    cent = -(Pc * np.log(Pc + 1e-12)).sum(1)
    ent8 = Qc[valid[np.argsort(-cent)[:8]]]

    res = {}

    # MOSTPOP + z0 floor
    floor = decode_np(model, np.zeros((1, d), np.float32))[0]
    mp = {f'full{KP}': 0.0, f'tail{KP}': 0.0, 'full10': 0.0}; fl = dict(mp)
    nf = nt = n10 = 0
    for c in coh:
        for tag, sc in (('mp', popb_local), ('fl', floor)):
            vf = G.ndcg(sc, c['rel_local'], c['prof_local'], hl, KP, False)
            vt = G.ndcg(sc, c['rel_local'], c['prof_local'], hl, KP, True)
            v10 = G.ndcg(sc, c['rel_local'], c['prof_local'], hl, 10, False)
            dd = mp if tag == 'mp' else fl
            if vf is not None: dd[f'full{KP}'] += vf
            if vt is not None: dd[f'tail{KP}'] += vt
            if v10 is not None: dd['full10'] += v10
        nf += 1
    for dd in (mp, fl):
        dd[f'full{KP}'] /= nf; dd[f'tail{KP}'] /= nf; dd['full10'] /= nf
    res['MOSTPOP'] = mp; res['z0_floor'] = fl

    # decoder-SVD basis-8 (graded + binary)
    res['basis8_decoderSVD'] = eval_builder(model, ar, coh, hl,
        lambda i, c, zs: unroll_static(Zst[i], svd8))
    res['basis8_decoderSVD_binary'] = eval_builder(model, ar, coh, hl,
        lambda i, c, zs: unroll_static(Zst[i], svd8, binary=True))

    # global-entropy concepts (graded) — documented control
    res['entropy_concepts_graded'] = eval_builder(model, ar, coh, hl,
        lambda i, c, zs: unroll_static(Zst[i], ent8))

    # lift-concepts-8 (per-user answerable, graded + binary)
    def lift8(i, c, binary=False):
        lift, _ = user_lift(c, memb_of_item, msize, tag_ok)
        sel = [int(t) for t in np.argsort(-lift)[:8] if lift[t] > 0]
        if not sel: return np.zeros(d, np.float32)
        return unroll_static(Zst[i], Qc[sel], binary=binary)
    res['lift_concepts8_graded'] = eval_builder(model, ar, coh, hl, lambda i, c, zs: lift8(i, c))
    res['lift_concepts8_binary'] = eval_builder(model, ar, coh, hl, lambda i, c, zs: lift8(i, c, True))

    # item-8 fold (fold 8 random profile likes)
    rng = np.random.default_rng(0)
    def item8(i, c, zs):
        lk = c['prof_like']
        if len(lk) > 8: lk = list(rng.choice(lk, 8, replace=False))
        if not lk: return np.zeros(d, np.float32)
        return enc_mu(model, onehot(lk)[0][None, :])[0]
    res['item8_fold'] = eval_builder(model, ar, coh, hl, item8)

    # ---- PREREG-1: concept-only k-curve (saturation) ----
    concept_kcurve = {}
    for k in range(1, 9):
        def cc(i, c, zs, k=k):
            lift, _ = user_lift(c, memb_of_item, msize, tag_ok)
            sel = [int(t) for t in np.argsort(-lift)[:k] if lift[t] > 0]
            if not sel: return np.zeros(d, np.float32)
            return unroll_static(Zst[i], Qc[sel])
        concept_kcurve[str(k)] = eval_builder(model, ar, coh, hl, cc)
        print(f'[static] concept k={k} full510={concept_kcurve[str(k)][f"full{KP}"]:.4f}', flush=True)

    # ---- PREREG answerability: mean answered concepts / 8 ----
    n_answerable = []
    for c in coh:
        _, aff = user_lift(c, memb_of_item, msize, tag_ok)
        n_answerable.append(min(int((aff > 0).sum()), 8))
    ans_mean = float(np.mean(n_answerable)); ans_frac_gt1 = float(np.mean(np.array(n_answerable) > 1))

    # ---- PREREG-3: effranks (I2 latent, uncentered participation ratio) ----
    W = model.decoder.weight.detach().numpy()
    pop_idx = np.argsort(-popb_local)[:600]
    Ztr = train_zstar_subsample(model, ar, loc, nsub=4000, seed=0)
    effrank = {'concept_memberbag_PR': PR(Qc[valid]),
               'item_all_PR': PR(W), 'item_top600_PR': PR(W[pop_idx]),
               'pca_trainzstar_PR': PR(Ztr - Ztr.mean(0, keepdims=True)),
               'n_valid_shelves': int(tag_ok.sum())}

    out = {k: (v if isinstance(v, dict) else v) for k, v in res.items()}
    out['concept_kcurve'] = concept_kcurve
    out['answerability'] = {'mean_answered_of8': ans_mean, 'frac_gt1': ans_frac_gt1}
    out['effrank'] = effrank
    out['eta'] = ETA; out['mean_znorm'] = float(np.linalg.norm(Zst, axis=1).mean())
    out['entropy_order_top8'] = [names[valid[j]] for j in np.argsort(-cent)[:8]]
    _save('static', out)
    print(json.dumps(out, indent=2))
    return out


# ================================================================== STAGE train
class Actor(torch.nn.Module):
    def __init__(self, d, T=T, h=512):
        super().__init__()
        self.T = T
        self.net = torch.nn.Sequential(
            torch.nn.Linear(d + T, h), torch.nn.SiLU(),
            torch.nn.Linear(h, h), torch.nn.SiLU(),
            torch.nn.Linear(h, d))

    def forward(self, z, t):
        oh = torch.zeros(z.shape[0], self.T, device=z.device); oh[:, t] = 1.0
        q = self.net(torch.cat([z, oh], dim=-1))
        return q / (q.norm(dim=-1, keepdim=True) + 1e-9)


def approx_ndcg_loss(scores, rel, temp=1.0):
    B, C = scores.shape
    diff = scores[:, None, :] - scores[:, :, None]
    pos = 1.0 + (torch.sigmoid(diff / temp).sum(dim=2) - 0.5)
    dcg = (rel / torch.log2(pos + 1.0)).sum(dim=1)
    ideal_pos = torch.arange(1, C + 1, device=scores.device, dtype=scores.dtype)
    rel_sorted, _ = torch.sort(rel, dim=1, descending=True)
    idcg = (rel_sorted / torch.log2(ideal_pos[None, :] + 1.0)).sum(dim=1) + 1e-9
    return (1.0 - dcg / idcg).mean()


def bc_warm(actor, svd8, d, steps=1200):
    tgt = torch.tensor(svd8); opt = torch.optim.Adam(actor.parameters(), lr=1e-3)
    rng = np.random.default_rng(0)
    for s in range(steps):
        B = 128
        coef = torch.tensor(rng.standard_normal((B, T)).astype(np.float32)) * 8.0
        z = coef @ tgt + 0.5 * torch.tensor(rng.standard_normal((B, d)).astype(np.float32))
        z = z * torch.tensor(rng.uniform(0, 1, (B, 1)).astype(np.float32))
        loss = 0.0
        for t in range(T):
            q = actor(z, t); loss = loss + (1.0 - (q * tgt[t]).sum(1)).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    return actor


def eval_actor_torch(actor, model, W, bdec, ar, coh, zstar_t, hl, binary=False, return_q=False):
    d = W.shape[1]; nz = zstar_t.norm(dim=1, keepdim=True) + 1e-9
    z = torch.zeros(len(coh), d); allq = []
    with torch.no_grad():
        for t in range(T):
            q = actor(z, t); a = (zstar_t * q).sum(1, keepdim=True) / nz
            if binary: a = torch.sign(a)
            z = z + ETA * a * q
            if return_q: allq.append(q.numpy().copy())
        S = (z @ W.T + bdec).numpy().astype(np.float64)
    r = _score_cohort(S, coh, hl)
    if return_q: return r + (np.stack(allq, 1),)
    return r


def _score_cohort(S, coh, hl):
    af = at = 0.0; mf = mt = 0
    for i, c in enumerate(coh):
        vf = G.ndcg(S[i], c['rel_local'], c['prof_local'], hl, KP, False)
        vt = G.ndcg(S[i], c['rel_local'], c['prof_local'], hl, KP, True)
        if vf is not None: af += vf; mf += 1
        if vt is not None: at += vt; mt += 1
    return af / max(mf, 1), at / max(mt, 1)


def stage_train(tseed):
    model, d = load_model(); ar = arena(); loc = loc_map(ar); hl = head_local(ar)
    W = torch.tensor(model.decoder.weight.detach().numpy(), dtype=torch.float32)
    bdec = torch.tensor(model.decoder.bias.detach().numpy(), dtype=torch.float32)
    svd8 = decoder_svd_dirs(model, 8)

    print(f'[train s{tseed}] building train z* subsample (n<= {NTR_SUB})...', flush=True)
    Zstar = train_zstar_subsample(model, ar, loc, nsub=NTR_SUB, seed=0)
    Zstar_t = torch.tensor(Zstar); znorm = Zstar_t.norm(dim=1, keepdim=True) + 1e-9
    with torch.no_grad():
        Steach = Zstar_t @ W.T + bdec
    K_POS = 20
    topk = torch.topk(Steach, K_POS, dim=1).indices

    val_coh = build_cohort(ar, loc, ar['va'].tolist())
    zsv = torch.tensor(zstar_of_cohort(model, val_coh))

    torch.manual_seed(1000 + tseed); np.random.seed(1000 + tseed)
    actor = Actor(d); bc_warm(actor, svd8, d)
    vf0, vt0 = eval_actor_torch(actor, model, W, bdec, ar, val_coh, zsv, hl)
    print(f'[train s{tseed}] post-BC-warm val full {vf0:.4f} tail {vt0:.4f}', flush=True)

    opt = torch.optim.Adam(actor.parameters(), lr=3e-4)
    Ntr = len(Zstar); B = 256; EPOCHS = 20; C_NEG = 108
    rng = np.random.default_rng(tseed)
    best_val = vt0; best_state = {k: v.clone() for k, v in actor.state_dict().items()}; best_ep = -1
    t0 = time.time()
    for ep in range(EPOCHS):
        actor.train(); perm = rng.permutation(Ntr)
        for st in range(0, Ntr, B):
            idx = perm[st:st + B]
            zs = Zstar_t[idx]; nz = znorm[idx]; z = torch.zeros(len(idx), d)
            for t in range(T):
                q = actor(z, t); a = (zs * q).sum(1, keepdim=True) / nz; z = z + ETA * a * q
            cosT = (z * zs).sum(1) / (z.norm(dim=1) + 1e-9) / nz.squeeze(1)
            L_rec = (1.0 - cosT).mean()
            pos = topk[idx]; neg = torch.randint(0, NU, (len(idx), C_NEG))
            cand = torch.cat([pos, neg], dim=1)
            sc = (z[:, None, :] * W[cand]).sum(2) + bdec[cand]
            rel = torch.zeros_like(sc); rel[:, :K_POS] = 1.0
            L_ndcg = approx_ndcg_loss(sc, rel)
            loss = L_rec + 0.3 * L_ndcg
            opt.zero_grad(); loss.backward(); opt.step()
        actor.eval()
        vf, vt = eval_actor_torch(actor, model, W, bdec, ar, val_coh, zsv, hl)
        if vt > best_val:
            best_val = vt; best_ep = ep; best_state = {k: v.clone() for k, v in actor.state_dict().items()}
        print(f'[train s{tseed}] ep{ep:2d} L {loss.item():.4f} (rec {L_rec.item():.3f} ndcg {L_ndcg.item():.3f}) '
              f'| val full {vf:.4f} tail {vt:.4f} | best_tail {best_val:.4f}@{best_ep} [{time.time()-t0:.0f}s]', flush=True)
    tag = f'p4b_actor_s{tseed}'
    torch.save({'state': best_state, 'd': d, 'best_ep': best_ep, 'best_val_tail': best_val},
               f'{CK}/{tag}.pt')
    print(f'[train] saved {tag}.pt best_val_tail {best_val:.4f} @ep{best_ep}', flush=True)


# ================================================================== STAGE eval
def _eval_partial(model, W, bdec, ar, coh, zst_t, hl, actor, k):
    d = W.shape[1]; nz = zst_t.norm(dim=1, keepdim=True) + 1e-9; z = torch.zeros(len(coh), d)
    with torch.no_grad():
        for t in range(k):
            q = actor(z, t); a = (zst_t * q).sum(1, keepdim=True) / nz; z = z + ETA * a * q
        S = (z @ W.T + bdec).numpy().astype(np.float64)
    return _score_cohort(S, coh, hl)


def _eval_snapped(model, W, bdec, ar, coh, zst_t, hl, actor, bank):
    d = W.shape[1]; nz = zst_t.norm(dim=1, keepdim=True) + 1e-9
    bank_t = torch.tensor(bank); z = torch.zeros(len(coh), d)
    with torch.no_grad():
        for t in range(T):
            q = actor(z, t); sims = q @ bank_t.T; qs = bank_t[sims.argmax(1)]
            a = (zst_t * qs).sum(1, keepdim=True) / nz; z = z + ETA * a * qs
        S = (z @ W.T + bdec).numpy().astype(np.float64)
    return _score_cohort(S, coh, hl)


def _greedy_static8(model, ar, loc, hl, pool, svd8, pca8, d):
    val_coh = build_cohort(ar, loc, ar['va'].tolist()); zst = zstar_of_cohort(model, val_coh)
    W = model.decoder.weight.detach().numpy(); b = model.decoder.bias.detach().numpy()
    cands = np.concatenate([pool, svd8, pca8], 0)
    cands = cands / (np.linalg.norm(cands, axis=1, keepdims=True) + 1e-9)

    def score_set(Q8):
        af = 0.0; m = 0
        for i, c in enumerate(val_coh):
            z = unroll_static(zst[i], np.stack(Q8)); s = z @ W.T + b
            vf = G.ndcg(s, c['rel_local'], c['prof_local'], hl, KP, False)
            if vf is not None: af += vf; m += 1
        return af / max(m, 1)

    chosen = []; used = set()
    for _ in range(8):
        best = -1; bi = -1
        for ci in range(len(cands)):
            if ci in used: continue
            v = score_set(chosen + [cands[ci]])
            if v > best: best = v; bi = ci
        chosen.append(cands[bi]); used.add(bi)
    return np.stack(chosen).astype(np.float32)


def _q_pool(actor, model, W, bdec, ar, loc, hl):
    val_coh = build_cohort(ar, loc, ar['va'].tolist())
    zsv = torch.tensor(zstar_of_cohort(model, val_coh))
    *_, Q_all = eval_actor_torch(actor, model, W, bdec, ar, val_coh, zsv, hl, return_q=True)
    means = Q_all.mean(0); means = means / (np.linalg.norm(means, axis=1, keepdims=True) + 1e-9)
    return means.astype(np.float32)


def stage_eval():
    model, d = load_model(); ar = arena(); loc = loc_map(ar); hl = head_local(ar)
    W = torch.tensor(model.decoder.weight.detach().numpy(), dtype=torch.float32)
    bdec = torch.tensor(model.decoder.bias.detach().numpy(), dtype=torch.float32)
    Wnp = model.decoder.weight.detach().numpy()
    Wn = Wnp / (np.linalg.norm(Wnp, axis=1, keepdims=True) + 1e-9)

    actors = {}
    for s in range(3):
        p = f'{CK}/p4b_actor_s{s}.pt'
        if os.path.exists(p):
            blob = torch.load(p, map_location=DEVICE); a = Actor(d); a.load_state_dict(blob['state']); a.eval()
            actors[f's{s}'] = (a, blob.get('best_val_tail'), blob.get('best_ep'))
    assert actors, 'no actors; run train first'

    svd8 = decoder_svd_dirs(model, 8)
    Ztr = train_zstar_subsample(model, ar, loc, nsub=4000, seed=0); pca8 = pca_zstar_dirs(Ztr, 8)
    Qc, tag_ok, names, memb_of_item, msize = concept_dirs(model, ar, loc)
    valid = np.where(tag_ok)[0]; Qc_valid = Qc[valid]

    coh = build_cohort(ar, loc, ar['te'].tolist())
    zst_np = zstar_of_cohort(model, coh); zst = torch.tensor(zst_np)

    per = {n: {} for n in actors}; per_bin = {n: {} for n in actors}
    for n, (act, _, _) in actors.items():
        f, t = eval_actor_torch(act, model, W, bdec, ar, coh, zst, hl)
        fb, tb = eval_actor_torch(act, model, W, bdec, ar, coh, zst, hl, binary=True)
        per[n] = {'full': f, 'tail': t}; per_bin[n] = {'full': fb, 'tail': tb}
        print(f'[eval] {n} graded full {f:.4f} tail {t:.4f} | binary full {fb:.4f} tail {tb:.4f}', flush=True)

    prim_name = max(actors, key=lambda n: actors[n][1] if actors[n][1] is not None else -1)
    prim = actors[prim_name][0]

    qcurve = {}
    for k in range(1, T + 1):
        f, t = _eval_partial(model, W, bdec, ar, coh, zst, hl, prim, k)
        qcurve[str(k)] = {'full': f, 'tail': t}

    fun, tun = eval_actor_torch(prim, model, W, bdec, ar, coh, zst, hl)
    fsc, tsc = _eval_snapped(model, W, bdec, ar, coh, zst, hl, prim, Qc_valid)
    fsi, tsi = _eval_snapped(model, W, bdec, ar, coh, zst, hl, prim, Wn)

    pool = _q_pool(prim, model, W, bdec, ar, loc, hl)
    Q8_greedy = _greedy_static8(model, ar, loc, hl, pool, svd8, pca8, d)
    fg, tg = _score_cohort(np.stack([unroll_static(zst_np[i], Q8_greedy) for i in range(len(coh))]) @ Wnp.T
                           + model.decoder.bias.detach().numpy(), coh, hl)

    # per-user bootstrap: actor vs strongest static (max greedy, svd8) vs concept-8 lift
    ba = {}; bs = {}; bc = {}
    with torch.no_grad():
        z = torch.zeros(len(coh), d); nz = zst.norm(dim=1, keepdim=True) + 1e-9
        for t in range(T):
            q = prim(z, t); a = (zst * q).sum(1, keepdim=True) / nz; z = z + ETA * a * q
        Sa = (z @ W.T + bdec).numpy().astype(np.float64)
    bb = model.decoder.bias.detach().numpy()
    for i, c in enumerate(coh):
        na = G.ndcg(Sa[i], c['rel_local'], c['prof_local'], hl, KP, False)
        zg = unroll_static(zst_np[i], Q8_greedy); zsv = unroll_static(zst_np[i], svd8)
        ng = G.ndcg(zg @ Wnp.T + bb, c['rel_local'], c['prof_local'], hl, KP, False)
        nsv = G.ndcg(zsv @ Wnp.T + bb, c['rel_local'], c['prof_local'], hl, KP, False)
        lift, _ = user_lift(c, memb_of_item, msize, tag_ok)
        sel = [int(tt) for tt in np.argsort(-lift)[:8] if lift[tt] > 0]
        zc = unroll_static(zst_np[i], Qc[sel]) if sel else np.zeros(d, np.float32)
        nc = G.ndcg(zc @ Wnp.T + bb, c['rel_local'], c['prof_local'], hl, KP, False)
        if None in (na, ng, nsv, nc): continue
        ba[c['x']] = na; bs[c['x']] = max(ng, nsv); bc[c['x']] = nc

    def boot(A_d, B_d, nboot=2000):
        keys = [k for k in A_d if k in B_d]
        diff = np.array([A_d[k] - B_d[k] for k in keys]); n = len(diff)
        rng = np.random.default_rng(0)
        means = np.array([diff[rng.integers(0, n, n)].mean() for _ in range(nboot)])
        lo, hi = np.percentile(means, [2.5, 97.5])
        return {'mean_diff': float(diff.mean()), 'ci95': [float(lo), float(hi)],
                'p_gt0': float((means > 0).mean()), 'n_users': int(n)}

    def sa(dd):
        return {'full': float(np.mean([dd[n]['full'] for n in dd])),
                'tail': float(np.mean([dd[n]['tail'] for n in dd])),
                'full_sd': float(np.std([dd[n]['full'] for n in dd])),
                'tail_sd': float(np.std([dd[n]['tail'] for n in dd]))}

    out = {'primary_actor': prim_name,
           'actors_graded': {n: per[n] for n in per}, 'actors_binary': {n: per_bin[n] for n in per_bin},
           'actor_seedavg_graded': sa(per), 'actor_seedavg_binary': sa(per_bin),
           'qcurve': qcurve,
           'snap_loss': {'unsnapped': {'full': fun, 'tail': tun},
                         'concept_snap': {'full': fsc, 'tail': tsc},
                         'item_snap': {'full': fsi, 'tail': tsi},
                         'delta_concept_full': fsc - fun, 'delta_concept_tail': tsc - tun,
                         'delta_item_full': fsi - fun, 'delta_item_tail': tsi - tun},
           'static8_control_greedy': {'full': fg, 'tail': tg},
           'bootstrap': {'actor_vs_strongest_static': boot(ba, bs),
                         'actor_vs_concept8': boot(ba, bc)}}
    _save('eval', out)
    print(json.dumps(out, indent=2))
    return out


# ------------------------------------------------------------------ io
def _save(stage, obj):
    all_ = {}
    if os.path.exists(OUTJSON):
        all_ = json.load(open(OUTJSON))
    all_[stage] = obj
    json.dump(all_, open(OUTJSON, 'w'), indent=2)
    print(f'[saved] {OUTJSON} :: {stage}', flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('stage', choices=['static', 'train', 'eval'])
    ap.add_argument('--tseed', type=int, default=0)
    args = ap.parse_args()
    if args.stage == 'static':
        stage_static()
    elif args.stage == 'train':
        stage_train(args.tseed)
    else:
        stage_eval()


if __name__ == '__main__':
    main()
