"""
p4a_battery.py -- INSTRUMENT 2.0 Phase 4a: THE ESSENTIAL POLICY BATTERY on the
certified RecVAE-d512 instrument (ML-1M). THE thesis question: does the flagship
(continuous + graded + adaptive elicitation) survive on a STRONG instrument?

Instrument: .cache/instrument2/ml1m_recvae_d512_best.pt (frozen).
Belief operator (P3 winner): additive  z' = z + eta * a * q,  eta=16, z0=0 cold seed.
Graded answer: a = cos(z*, q),  z* = enc(profile-half likes)  (the simulated user's taste).
Arena: ml1m_arena (byte-identical V1 splits). Eval seeds {1,2,3,7,11}; te[:300]=VAL, te[300:]=TEST.
Metric: NDCG@10 full + Cremonesi tail. T=8 turns.

Stages (append results to experiments/instrument2/PHASE4A_BATTERY.md as they land):
  static   -- item 1: MOSTPOP/z0; static entropy-over-concepts (graded); static informative
              basis-8 (decoder-SVD + PCA-of-train-z*); k=8 item-fold reference.
  train    -- item 2: D1-recipe differentiable-unroll actor; 3 train-seeds from scratch;
              val-best (SELVAL tail) selection; saves .cache/instrument2/p4a_actor_s{seed}.pt
  eval     -- items 2-5: actor test eval (5 seeds) full/tail + q-curve; binary-answer variant;
              snap-loss (concept + item-decoder-row); static-8 control (greedy from actor query
              distribution + PCA basis); paired per-user bootstrap margins.

Usage:
  python scripts/instrument2/p4a_battery.py static
  python scripts/instrument2/p4a_battery.py train --tseed 0 [--field]
  python scripts/instrument2/p4a_battery.py eval
"""
import os, sys, json, time, argparse
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ml1m_arena as A
from recvae import RecVAE

DEVICE = torch.device('cpu'); torch.set_num_threads(4)
CK = '.cache/instrument2'
CKPT = f'{CK}/ml1m_recvae_d512_best.pt'
NI = 3706
ETA = 16.0
T = 8
SEEDS = [1, 2, 3, 7, 11]
OUTJSON = f'{CK}/p4a_battery.json'


# ------------------------------------------------------------------ model helpers
def load_model():
    blob = torch.load(CKPT, map_location=DEVICE)
    a = blob['args']
    m = RecVAE(a['hidden'], a['latent'], NI).to(DEVICE)
    m.load_state_dict(blob['model']); m.eval()
    return m, a['latent']


def enc_mu(model, Xd):
    with torch.no_grad():
        mu, _ = model.encoder(torch.tensor(np.asarray(Xd, np.float32)), dropout_rate=0.0)
    return mu.numpy()


def decode_np(model, Z):
    with torch.no_grad():
        return model.decoder(torch.tensor(np.asarray(Z, np.float32))).numpy().astype(np.float64)


def onehot(items, ni=NI):
    x = np.zeros((1, ni), np.float32)
    if len(items): x[0, list(items)] = 1.0
    return x


def bag_from_likes(likes, ni=NI):
    x = np.zeros(ni, np.float32); x[list(likes)] = 1.0; return x


# ------------------------------------------------------------------ static direction designs
def decoder_svd_dirs(model, K):
    """Top-K right singular vectors of the (centered) decoder weight (ni x d): the item-relevant
    orthonormal population basis. Reproduces the P3 W1 'informative' design (0.4688 @ K=8)."""
    W = model.decoder.weight.detach().numpy()               # (ni, d)
    _, _, Vt = np.linalg.svd(W - W.mean(0, keepdims=True), full_matrices=False)
    return Vt[:K].astype(np.float32)


def pca_zstar_dirs(model, ar_seed, K):
    """Top-K principal directions of TRAIN-user full-profile encodings z* (population taste PCA)."""
    ar = A.load_arena(seed=ar_seed); rd = {x: dict(v) for x, v in ar['rat_by_u'].items()}
    trU = [x for x in ar['trU'] if sum(1 for j, r in rd[x].items() if r >= 4) >= 4]
    d = model.decoder.in_features
    Z = np.zeros((len(trU), d), np.float32)
    for st in range(0, len(trU), 500):
        ch = trU[st:st + 500]
        Xd = np.zeros((len(ch), NI), np.float32)
        for r, x in enumerate(ch):
            lk = [j for j, rr in rd[x].items() if rr >= 4]
            Xd[r, lk] = 1.0
        Z[st:st + len(ch)] = enc_mu(model, Xd)
    Zc = Z - Z.mean(0, keepdims=True)
    _, _, Vt = np.linalg.svd(Zc, full_matrices=False)
    return Vt[:K].astype(np.float32)


def concept_dirs(model):
    """member-bag encode concept directions (P3 W2 winner), lift metadata for selection."""
    C = np.load(f'{CK}/concepts_ml1m.npz', allow_pickle=True)
    item_tag = C['item_tag']; tag_mass = C['tag_mass']; names = C['tag_names']
    ntag = item_tag.shape[1]; Mbag = 50
    Q = np.zeros((ntag, model.decoder.in_features), np.float32)
    for tc in range(ntag):
        rel = item_tag[:, tc]
        top = np.argpartition(-rel, Mbag)[:Mbag]
        b = np.zeros(NI, np.float32); b[top] = rel[top]; b /= b.sum() + 1e-9
        z = enc_mu(model, b[None, :])[0]
        Q[tc] = z / (np.linalg.norm(z) + 1e-9)
    return Q, item_tag, tag_mass, names


# ------------------------------------------------------------------ answer + operator (numpy)
def graded_answer(zstar, q):
    return float(zstar @ q / (np.linalg.norm(zstar) + 1e-9))


def unroll_static(zstar, Q8, eta=ETA, binary=False):
    """Apply a FIXED set of directions Q8 (Kxd) statically; answers graded (or binary)."""
    d = Q8.shape[1]; z = np.zeros(d, np.float32); nz = np.linalg.norm(zstar) + 1e-9
    for t in range(Q8.shape[0]):
        a = float(zstar @ Q8[t] / nz)
        if binary: a = float(np.sign(a))
        z = z + eta * a * Q8[t]
    return z


# ------------------------------------------------------------------ eval a per-user z-builder
def eval_zbuilder(model, ar, users, zstar_of, build_z, tail_both=True):
    """build_z(i, x, zstar) -> final z ; returns (full, tail) seed-mean over cohort."""
    af = at = 0.0; mf = mt = 0
    per_full = []
    for i, x in enumerate(users):
        profset, tst = ar['SPL'][x]; rd = ar['rat_by_u_dict'][x]
        tlike = set(j for j in tst if rd[j] >= 4)
        if not tlike:
            continue
        zs = zstar_of[i]
        z = build_z(i, x, zs)
        s = decode_np(model, z[None, :])[0]
        nf = A.ndcg_at10(s, tlike, profset, ar['headmask'], False)
        nt = A.ndcg_at10(s, tlike, profset, ar['headmask'], True)
        if nf is not None:
            af += nf; mf += 1; per_full.append((x, nf))
        if nt is not None:
            at += nt; mt += 1
    return af / max(mf, 1), at / max(mt, 1), per_full


def build_zstar(model, ar, users):
    """z* = enc(profile-half likes) per user (the simulated taste elicitation must recover)."""
    d = model.decoder.in_features
    Z = np.zeros((len(users), d), np.float32)
    for i, x in enumerate(users):
        profset, _ = ar['SPL'][x]; rd = ar['rat_by_u_dict'][x]
        lk = [j for j in profset if rd[j] >= 4]
        if lk:
            Z[i] = enc_mu(model, bag_from_likes(lk)[None, :])[0]
    return Z


def arena_seed(sd, rd_all=None):
    ar = A.load_arena(seed=sd)
    ar['rat_by_u_dict'] = rd_all if rd_all is not None else {x: dict(v) for x, v in ar['rat_by_u'].items()}
    return ar


# ================================================================== STAGE: static baselines
def stage_static():
    model, d = load_model()
    rd_all = {x: dict(v) for x, v in A.load_arena(seed=123)['rat_by_u'].items()}
    svd8 = decoder_svd_dirs(model, 8)
    pca8 = pca_zstar_dirs(model, 123, 8)
    Qc, item_tag, tag_mass, names = concept_dirs(model)

    # per-concept decode entropy -> static global entropy ordering (top-8 highest entropy concepts)
    Sc = decode_np(model, Qc)                                # (ntag, ni) decode of each concept dir
    Pc = np.exp(Sc - Sc.max(1, keepdims=True)); Pc /= Pc.sum(1, keepdims=True)
    cent = -(Pc * np.log(Pc + 1e-12)).sum(1)
    ent_order = np.argsort(-cent)                            # static, same for all users
    ent8 = Qc[ent_order[:8]]

    res = {'MOSTPOP': {}, 'z0_floor': {}, 'entropy_concepts_graded': {},
           'lift_concepts8_graded': {}, 'lift_concepts8_binary': {},
           'basis8_decoderSVD': {}, 'basis8_PCAzstar': {}, 'item8_fold': {},
           'basis8_decoderSVD_binary': {}, 'entropy_concepts_binary': {}}
    for sd in SEEDS:
        ar = arena_seed(sd, rd_all)
        test = ar['test_users']
        zst = build_zstar(model, ar, test)

        # MOSTPOP (popb ranker) and z=0 floor
        floor = decode_np(model, np.zeros((1, d), np.float32))[0]
        popb = ar['popb'].astype(np.float64)
        af = at = 0.0; mf = mt = 0; ff = ft = 0.0; nf0 = nt0 = 0
        for x in test:
            profset, tst = ar['SPL'][x]; rd = rd_all[x]
            tlike = set(j for j in tst if rd[j] >= 4)
            if not tlike: continue
            npf = A.ndcg_at10(popb, tlike, profset, ar['headmask'], False)
            npt = A.ndcg_at10(popb, tlike, profset, ar['headmask'], True)
            nff = A.ndcg_at10(floor, tlike, profset, ar['headmask'], False)
            nft = A.ndcg_at10(floor, tlike, profset, ar['headmask'], True)
            if npf is not None: af += npf; mf += 1
            if npt is not None: at += npt; mt += 1
            if nff is not None: ff += nff; nf0 += 1
            if nft is not None: ft += nft; nt0 += 1
        _app(res['MOSTPOP'], af / mf, at / mt)
        _app(res['z0_floor'], ff / nf0, ft / nt0)

        # static informative basis-8 (graded, additive)
        for tag, Q8 in [('basis8_decoderSVD', svd8), ('basis8_PCAzstar', pca8)]:
            f, t, _ = eval_zbuilder(model, ar, test, zst,
                                    lambda i, x, zs, Q8=Q8: unroll_static(zs, Q8))
            _app(res[tag], f, t)
        # binary-answer variant of the strong static bar
        f, t, _ = eval_zbuilder(model, ar, test, zst,
                                lambda i, x, zs: unroll_static(zs, svd8, binary=True))
        _app(res['basis8_decoderSVD_binary'], f, t)

        # static GLOBAL-entropy-over-concepts (graded) + binary variant (documented control)
        f, t, _ = eval_zbuilder(model, ar, test, zst,
                                lambda i, x, zs: unroll_static(zs, ent8))
        _app(res['entropy_concepts_graded'], f, t)
        f, t, _ = eval_zbuilder(model, ar, test, zst,
                                lambda i, x, zs: unroll_static(zs, ent8, binary=True))
        _app(res['entropy_concepts_binary'], f, t)

        # uent+GRAW analogue: per-user top-8 LIFT-selected (answerable) concepts, graded/binary
        def lift8(i, x, zs, binary=False):
            profset, _ = ar['SPL'][x]; rd = rd_all[x]
            prof = [j for j in profset if rd[j] >= 4]
            aff = item_tag[prof].sum(0) if prof else np.zeros(item_tag.shape[1])
            lift = aff / (tag_mass + 1e-9)
            sel = np.argsort(-lift)[:8]
            sel = [c for c in sel if lift[c] > 0]
            if not sel: return np.zeros(d, np.float32)
            return unroll_static(zs, Qc[sel], binary=binary)
        f, t, _ = eval_zbuilder(model, ar, test, zst, lift8)
        _app(res['lift_concepts8_graded'], f, t)
        f, t, _ = eval_zbuilder(model, ar, test, zst, lambda i, x, zs: lift8(i, x, zs, True))
        _app(res['lift_concepts8_binary'], f, t)

        # k=8 item-fold reference (fold 8 random profile likes)
        rng = np.random.default_rng(sd)
        def mk_item8(i, x, zs):
            profset, _ = ar['SPL'][x]; rd = rd_all[x]
            lk = [j for j in profset if rd[j] >= 4]
            if len(lk) > 8: lk = list(rng.choice(lk, 8, replace=False))
            if not lk: return np.zeros(d, np.float32)
            return enc_mu(model, bag_from_likes(lk)[None, :])[0]
        f, t, _ = eval_zbuilder(model, ar, test, zst, mk_item8)
        _app(res['item8_fold'], f, t)
        print(f'[static] seed{sd} done', flush=True)

    out = {k: {'full': float(np.mean(v['full'])), 'tail': float(np.mean(v['tail'])),
               'full_sd': float(np.std(v['full'])), 'tail_sd': float(np.std(v['tail']))}
           for k, v in res.items()}
    out['entropy_order_top8_concepts'] = [str(names[i]) for i in ent_order[:8]]
    _save_json('static', out)
    print(json.dumps(out, indent=2))
    return out


def _app(d, f, t):
    d.setdefault('full', []).append(f); d.setdefault('tail', []).append(t)


# ================================================================== STAGE: train actor
class Actor(torch.nn.Module):
    """MLP(z_t, turn) -> unit direction q_t (d)."""
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
    """Differentiable approxNDCG over a candidate set. scores,rel: (B, C). rel binary graded."""
    B, C = scores.shape
    diff = scores[:, None, :] - scores[:, :, None]          # (B, C_i? ) -> D[b,i,j]=s_j - s_i
    # approx position of item i = 1 + sum_j sigmoid((s_j - s_i)/temp)
    pos = 1.0 + (torch.sigmoid(diff / temp).sum(dim=2) - 0.5)   # subtract self (sig(0)=0.5)
    gain = rel                                              # 2^rel-1 = rel for binary
    dcg = (gain / torch.log2(pos + 1.0)).sum(dim=1)
    # ideal dcg
    ideal_pos = torch.arange(1, C + 1, device=scores.device, dtype=scores.dtype)
    rel_sorted, _ = torch.sort(rel, dim=1, descending=True)
    idcg = (rel_sorted / torch.log2(ideal_pos[None, :] + 1.0)).sum(dim=1) + 1e-9
    return (1.0 - dcg / idcg).mean()


def bc_warm(actor, svd8, d, steps=1200):
    """BC-warm the actor to emit the static SVD-8 informative basis (per-turn), for ANY belief z.
    Scratch training collapses (documented); this gives the unroll fine-tune a strong, isolable
    starting point (adaptivity is then whatever the fine-tune adds ON TOP of the static basis)."""
    tgt = torch.tensor(svd8)                                # (T,d) unit rows
    opt = torch.optim.Adam(actor.parameters(), lr=1e-3)
    rng = np.random.default_rng(0)
    for s in range(steps):
        B = 128
        # sample beliefs spanning what occurs during unroll: scaled random combos of basis + noise
        coef = torch.tensor(rng.standard_normal((B, T)).astype(np.float32)) * 8.0
        z = coef @ tgt + 0.5 * torch.tensor(rng.standard_normal((B, d)).astype(np.float32))
        z = z * torch.tensor(rng.uniform(0, 1, (B, 1)).astype(np.float32))   # include near-cold z~0
        loss = 0.0
        for t in range(T):
            q = actor(z, t)
            loss = loss + (1.0 - (q * tgt[t]).sum(1)).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    return actor


def stage_train(tseed, use_field):
    model, d = load_model()
    W = torch.tensor(model.decoder.weight.detach().numpy(), dtype=torch.float32)  # (ni,d)
    bdec = torch.tensor(model.decoder.bias.detach().numpy(), dtype=torch.float32)
    svd8 = decoder_svd_dirs(model, 8)
    ar123 = A.load_arena(seed=123); rd_all = {x: dict(v) for x, v in ar123['rat_by_u'].items()}

    # precompute train z* (all likes) once
    trU = [x for x in ar123['trU'] if sum(1 for j, r in rd_all[x].items() if r >= 4) >= 4]
    Zstar = np.zeros((len(trU), d), np.float32)
    for st in range(0, len(trU), 500):
        ch = trU[st:st + 500]
        Xd = np.zeros((len(ch), NI), np.float32)
        for r, x in enumerate(ch):
            lk = [j for j, rr in rd_all[x].items() if rr >= 4]; Xd[r, lk] = 1.0
        Zstar[st:st + len(ch)] = enc_mu(model, Xd)
    Zstar_t = torch.tensor(Zstar)
    znorm = Zstar_t.norm(dim=1, keepdim=True) + 1e-9
    # teacher top-K relevant items per train user (from decode(z*))
    with torch.no_grad():
        Steach = Zstar_t @ W.T + bdec
    K_POS = 20
    topk = torch.topk(Steach, K_POS, dim=1).indices                     # (Ntr, K_POS)
    # divisiveness field: population variance of z*·q for a query q  == qᵀ Cov(z*) q
    Zc = Zstar_t - Zstar_t.mean(0, keepdim=True)
    Cov = (Zc.T @ Zc) / Zc.shape[0]                                     # (d,d)

    # val cohort (seed 1) for SELVAL-tail best-epoch selection
    arv = arena_seed(1, rd_all); val = arv['val_users']
    zsv = torch.tensor(build_zstar(model, arv, val))

    torch.manual_seed(1000 + tseed); np.random.seed(1000 + tseed)
    actor = Actor(d)
    # BC-warm from the static SVD-8 basis (scratch collapses -> documented), then unroll fine-tune
    bc_warm(actor, svd8, d)
    arv0 = arena_seed(1, rd_all)
    zsv0 = torch.tensor(build_zstar(model, arv0, arv0['val_users']))
    vf0, vt0 = eval_actor_torch(actor, model, W, bdec, arv0, arv0['val_users'], zsv0)
    print(f'[train s{tseed}] post-BC-warm val full {vf0:.4f} tail {vt0:.4f}', flush=True)
    opt = torch.optim.Adam(actor.parameters(), lr=3e-4)
    Ntr = len(trU); B = 256; EPOCHS = 20
    rng = np.random.default_rng(tseed)
    best_val = vt0; best_state = {k: v.clone() for k, v in actor.state_dict().items()}; best_ep = -1
    C_NEG = 108
    t0 = time.time()
    for ep in range(EPOCHS):
        actor.train(); perm = rng.permutation(Ntr)
        for st in range(0, Ntr, B):
            idx = perm[st:st + B]
            zs = Zstar_t[idx]; nz = znorm[idx]
            z = torch.zeros(len(idx), d)
            field_acc = 0.0
            for t in range(T):
                q = actor(z, t)
                a = (zs * q).sum(1, keepdim=True) / nz            # graded answer cos(z*,q)
                z = z + ETA * a * q
                if use_field:
                    field_acc = field_acc + ((q @ Cov) * q).sum(1).mean()
            # reconstruction (direction) + softNDCG
            cosT = (z * zs).sum(1) / (z.norm(dim=1) + 1e-9) / nz.squeeze(1)
            L_rec = (1.0 - cosT).mean()
            # softNDCG over candidate set: K_POS teacher-positives + C_NEG random negatives
            pos = topk[idx]                                        # (b,K_POS)
            neg = torch.randint(0, NI, (len(idx), C_NEG))
            cand = torch.cat([pos, neg], dim=1)                   # (b, K_POS+C_NEG)
            sc = (z[:, None, :] * W[cand]).sum(2) + bdec[cand]    # (b, C)
            rel = torch.zeros_like(sc); rel[:, :K_POS] = 1.0
            L_ndcg = approx_ndcg_loss(sc, rel)
            loss = L_rec + 0.3 * L_ndcg
            if use_field:
                loss = loss - 0.02 * field_acc / T                # reward divisive directions
            opt.zero_grad(); loss.backward(); opt.step()
        # val (SELVAL tail)
        actor.eval()
        vf, vt = eval_actor_torch(actor, model, W, bdec, arv, val, zsv, binary=False)
        if vt > best_val:
            best_val = vt; best_ep = ep; best_state = {k: v.clone() for k, v in actor.state_dict().items()}
        print(f'[train s{tseed}{"+field" if use_field else ""}] ep{ep:2d} '
              f'L {loss.item():.4f} (rec {L_rec.item():.3f} ndcg {L_ndcg.item():.3f}) '
              f'| val full {vf:.4f} tail {vt:.4f} | best_tail {best_val:.4f}@{best_ep} '
              f'[{time.time()-t0:.0f}s]', flush=True)
    tag = f'p4a_actor_s{tseed}' + ('_field' if use_field else '')
    torch.save({'state': best_state, 'd': d, 'best_ep': best_ep, 'best_val_tail': best_val,
                'field': use_field}, f'{CK}/{tag}.pt')
    print(f'[train] saved {tag}.pt best_val_tail {best_val:.4f} @ep{best_ep}', flush=True)


def eval_actor_torch(actor, model, W, bdec, ar, users, zstar_t, binary=False, return_q=False):
    """Run the actor over a cohort (torch), return (full, tail) NDCG. Optionally collect queries."""
    d = W.shape[1]; nz = zstar_t.norm(dim=1, keepdim=True) + 1e-9
    z = torch.zeros(len(users), d)
    allq = []
    with torch.no_grad():
        for t in range(T):
            q = actor(z, t)
            a = (zstar_t * q).sum(1, keepdim=True) / nz
            if binary: a = torch.sign(a)
            z = z + ETA * a * q
            if return_q: allq.append(q.numpy().copy())
        S = (z @ W.T + bdec).numpy().astype(np.float64)
    af = at = 0.0; mf = mt = 0
    for i, x in enumerate(users):
        profset, tst = ar['SPL'][x]; rd = ar['rat_by_u_dict'][x]
        tlike = set(j for j in tst if rd[j] >= 4)
        if not tlike: continue
        nf = A.ndcg_at10(S[i], tlike, profset, ar['headmask'], False)
        ntl = A.ndcg_at10(S[i], tlike, profset, ar['headmask'], True)
        if nf is not None: af += nf; mf += 1
        if ntl is not None: at += ntl; mt += 1
    res = (af / max(mf, 1), at / max(mt, 1))
    if return_q:
        return res + (np.stack(allq, 1),)   # (users, T, d)
    return res


# ================================================================== STAGE: eval (items 2-5)
def stage_eval():
    model, d = load_model()
    W = torch.tensor(model.decoder.weight.detach().numpy(), dtype=torch.float32)
    bdec = torch.tensor(model.decoder.bias.detach().numpy(), dtype=torch.float32)
    rd_all = {x: dict(v) for x, v in A.load_arena(seed=123)['rat_by_u'].items()}

    # load trained actors (3 seeds, scratch); field variant if present
    actors = {}
    for s in range(3):
        p = f'{CK}/p4a_actor_s{s}.pt'
        if os.path.exists(p):
            blob = torch.load(p, map_location=DEVICE); a = Actor(d); a.load_state_dict(blob['state']); a.eval()
            actors[f's{s}'] = (a, blob.get('best_val_tail'), blob.get('best_ep'))
    field_actors = {}
    for s in range(3):
        p = f'{CK}/p4a_actor_s{s}_field.pt'
        if os.path.exists(p):
            blob = torch.load(p, map_location=DEVICE); a = Actor(d); a.load_state_dict(blob['state']); a.eval()
            field_actors[f's{s}'] = a
    assert actors, 'no trained actors found; run train stage first'

    svd8 = decoder_svd_dirs(model, 8); pca8 = pca_zstar_dirs(model, 123, 8)
    Qc, item_tag, tag_mass, names = concept_dirs(model)
    Wn = W / (W.norm(dim=1, keepdim=True) + 1e-9)           # unit item-decoder-row directions
    Wn_np = Wn.numpy()
    Qc_np = Qc

    # per-seed eval of each actor (graded), binary variant, plus q-curve for the primary actor
    per_seed = {name: {'full': [], 'tail': []} for name in actors}
    per_seed_bin = {name: {'full': [], 'tail': []} for name in actors}
    per_seed_field = {name: {'full': [], 'tail': []} for name in field_actors}
    # collect per-user for bootstrap on the primary actor (seed-averaged records)
    boot_actor = {}; boot_static = {}; boot_ent = {}
    qcurve = {name: {k: {'full': [], 'tail': []} for k in range(1, T + 1)} for name in ['s0']}
    snap_concept = {'full': [], 'tail': []}; snap_item = {'full': [], 'tail': []}
    unsnapped_ref = {'full': [], 'tail': []}
    static_ctrl = {'full': [], 'tail': []}         # greedy static-8 from actor query dist (per seed)

    ent8 = _entropy_concepts8(model, Qc)

    for sd in SEEDS:
        ar = arena_seed(sd, rd_all); test = ar['test_users']
        zst_np = build_zstar(model, ar, test); zst = torch.tensor(zst_np)
        # actors graded + binary
        for name, (act, _, _) in actors.items():
            f, t = eval_actor_torch(act, model, W, bdec, ar, test, zst, binary=False)
            per_seed[name]['full'].append(f); per_seed[name]['tail'].append(t)
            fb, tb = eval_actor_torch(act, model, W, bdec, ar, test, zst, binary=True)
            per_seed_bin[name]['full'].append(fb); per_seed_bin[name]['tail'].append(tb)
        for name, act in field_actors.items():
            f, t = eval_actor_torch(act, model, W, bdec, ar, test, zst, binary=False)
            per_seed_field[name]['full'].append(f); per_seed_field[name]['tail'].append(t)

        # --- primary actor = the best-val seed among scratch actors ---
        prim_name = max(actors, key=lambda n: actors[n][1] if actors[n][1] is not None else -1)
        prim = actors[prim_name][0]

        # q-curve (turns 1..8) for the primary actor + collect emitted queries
        _, _, Q_all = eval_actor_torch(prim, model, W, bdec, ar, test, zst, return_q=True)  # (U,T,d)
        for k in range(1, T + 1):
            f, t = _eval_partial(model, W, bdec, ar, test, zst_np, prim, k)
            qcurve['s0'][k]['full'].append(f); qcurve['s0'][k]['tail'].append(t)

        # snap-loss: snap each emitted q_t to nearest concept dir / nearest item-decoder-row dir
        fsc, tsc = _eval_snapped(model, W, bdec, ar, test, zst_np, prim, Qc_np)
        fsi, tsi = _eval_snapped(model, W, bdec, ar, test, zst_np, prim, Wn_np)
        fun, tun = eval_actor_torch(prim, model, W, bdec, ar, test, zst)
        snap_concept['full'].append(fsc); snap_concept['tail'].append(tsc)
        snap_item['full'].append(fsi); snap_item['tail'].append(tsi)
        unsnapped_ref['full'].append(fun); unsnapped_ref['tail'].append(tun)

        # static-8 control: greedy forward selection from actor's query distribution (+ SVD/PCA pool)
        Q8_greedy = _greedy_static8(model, arena_seed(1, rd_all), Q_all_pool(prim, model, W, bdec, rd_all),
                                    svd8, pca8, d)
        fg, tg, _ = eval_zbuilder(model, ar, test, zst_np,
                                  lambda i, x, zs, Q8=Q8_greedy: unroll_static(zs, Q8))
        static_ctrl['full'].append(fg); static_ctrl['tail'].append(tg)

        # per-user records for bootstrap (primary actor vs strongest static vs entropy-graded)
        _collect_boot(model, W, bdec, ar, test, zst_np, prim, Q8_greedy, ent8, svd8,
                      boot_actor, boot_static, boot_ent)
        print(f'[eval] seed{sd} done', flush=True)

    def agg(dd):
        return {'full': float(np.mean(dd['full'])), 'tail': float(np.mean(dd['tail'])),
                'full_sd': float(np.std(dd['full'])), 'tail_sd': float(np.std(dd['tail']))}
    out = {'actors_graded': {n: agg(per_seed[n]) for n in per_seed},
           'actors_binary': {n: agg(per_seed_bin[n]) for n in per_seed_bin},
           'actors_field': {n: agg(per_seed_field[n]) for n in per_seed_field},
           'primary_actor': prim_name,
           'actor_seedavg_graded': agg(_pool(per_seed)),
           'actor_seedavg_binary': agg(_pool(per_seed_bin)),
           'qcurve_full': {str(k): float(np.mean(qcurve['s0'][k]['full'])) for k in range(1, T + 1)},
           'qcurve_tail': {str(k): float(np.mean(qcurve['s0'][k]['tail'])) for k in range(1, T + 1)},
           'snap_loss': {
               'unsnapped': agg(unsnapped_ref),
               'concept_snap': agg(snap_concept),
               'item_snap': agg(snap_item),
               'delta_concept_full': float(np.mean(snap_concept['full']) - np.mean(unsnapped_ref['full'])),
               'delta_concept_tail': float(np.mean(snap_concept['tail']) - np.mean(unsnapped_ref['tail'])),
               'delta_item_full': float(np.mean(snap_item['full']) - np.mean(unsnapped_ref['full'])),
               'delta_item_tail': float(np.mean(snap_item['tail']) - np.mean(unsnapped_ref['tail']))},
           'static8_control_greedy': agg(static_ctrl)}
    # paired bootstrap margins
    out['bootstrap'] = {
        'actor_vs_strongest_static': _paired_boot(boot_actor, boot_static),
        'actor_vs_entropy_graded': _paired_boot(boot_actor, boot_ent)}
    _save_json('eval', out)
    print(json.dumps(out, indent=2))
    return out


def _pool(per_seed):
    out = {'full': [], 'tail': []}
    for n in per_seed:
        out['full'] += per_seed[n]['full']; out['tail'] += per_seed[n]['tail']
    return out


def _entropy_concepts8(model, Qc):
    Sc = decode_np(model, Qc); Pc = np.exp(Sc - Sc.max(1, keepdims=True)); Pc /= Pc.sum(1, keepdims=True)
    cent = -(Pc * np.log(Pc + 1e-12)).sum(1)
    return Qc[np.argsort(-cent)[:8]]


def _eval_partial(model, W, bdec, ar, users, zst_np, actor, k):
    """NDCG using only the first k turns of the actor (q-curve)."""
    d = W.shape[1]; zst = torch.tensor(zst_np); nz = zst.norm(dim=1, keepdim=True) + 1e-9
    z = torch.zeros(len(users), d)
    with torch.no_grad():
        for t in range(k):
            q = actor(z, t); a = (zst * q).sum(1, keepdim=True) / nz; z = z + ETA * a * q
        S = (z @ W.T + bdec).numpy().astype(np.float64)
    af = at = 0.0; mf = mt = 0
    for i, x in enumerate(users):
        profset, tst = ar['SPL'][x]; rd = ar['rat_by_u_dict'][x]
        tlike = set(j for j in tst if rd[j] >= 4)
        if not tlike: continue
        nf = A.ndcg_at10(S[i], tlike, profset, ar['headmask'], False)
        ntl = A.ndcg_at10(S[i], tlike, profset, ar['headmask'], True)
        if nf is not None: af += nf; mf += 1
        if ntl is not None: at += ntl; mt += 1
    return af / max(mf, 1), at / max(mt, 1)


def _eval_snapped(model, W, bdec, ar, users, zst_np, actor, bank):
    """Snap each emitted q_t to nearest direction in `bank` (rows unit), then apply operator."""
    d = W.shape[1]; zst = torch.tensor(zst_np); nz = zst.norm(dim=1, keepdim=True) + 1e-9
    bank_t = torch.tensor(bank)                              # (M, d) unit rows
    z = torch.zeros(len(users), d)
    with torch.no_grad():
        for t in range(T):
            q = actor(z, t)                                 # (U,d) unit
            sims = q @ bank_t.T                             # (U,M)
            qs = bank_t[sims.argmax(1)]                     # snapped unit dirs
            a = (zst * qs).sum(1, keepdim=True) / nz
            z = z + ETA * a * qs
        S = (z @ W.T + bdec).numpy().astype(np.float64)
    af = at = 0.0; mf = mt = 0
    for i, x in enumerate(users):
        profset, tst = ar['SPL'][x]; rd = ar['rat_by_u_dict'][x]
        tlike = set(j for j in tst if rd[j] >= 4)
        if not tlike: continue
        nf = A.ndcg_at10(S[i], tlike, profset, ar['headmask'], False)
        ntl = A.ndcg_at10(S[i], tlike, profset, ar['headmask'], True)
        if nf is not None: af += nf; mf += 1
        if ntl is not None: at += ntl; mt += 1
    return af / max(mf, 1), at / max(mt, 1)


def Q_all_pool(actor, model, W, bdec, rd_all):
    """Collect the actor's emitted per-turn MEAN directions on the seed-1 VAL cohort as a candidate pool."""
    arv = arena_seed(1, rd_all); val = arv['val_users']
    zst = torch.tensor(build_zstar(model, arv, val))
    _, _, Q_all = eval_actor_torch(actor, model, W, bdec, arv, val, zst, return_q=True)  # (U,T,d)
    means = Q_all.mean(0)                                    # (T,d) per-turn mean direction
    means = means / (np.linalg.norm(means, axis=1, keepdims=True) + 1e-9)
    return means.astype(np.float32)


def _greedy_static8(model, arv, pool, svd8, pca8, d):
    """Greedy forward selection of 8 fixed directions maximizing VAL full-NDCG when applied
    statically (same set for all users). Candidate pool = actor per-turn means + SVD-8 + PCA-8."""
    val = arv['val_users']; zst = build_zstar(model, arv, val)
    W = model.decoder.weight.detach().numpy(); b = model.decoder.bias.detach().numpy()
    cands = np.concatenate([pool, svd8, pca8], 0)
    cands = cands / (np.linalg.norm(cands, axis=1, keepdims=True) + 1e-9)
    rd = arv['rat_by_u_dict']

    def score_set(Q8):
        af = 0.0; m = 0
        for i, x in enumerate(val):
            profset, tst = arv['SPL'][x]; rr = rd[x]
            tlike = set(j for j in tst if rr[j] >= 4)
            if not tlike: continue
            z = unroll_static(zst[i], np.stack(Q8))
            s = z @ W.T + b
            nf = A.ndcg_at10(s, tlike, profset, arv['headmask'], False)
            if nf is not None: af += nf; m += 1
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


def _collect_boot(model, W, bdec, ar, users, zst_np, actor, Q8_greedy, ent8, svd8,
                  boot_actor, boot_static, boot_ent):
    """Accumulate per-user NDCG (full) for paired bootstrap; strongest static = max(greedy, svd8)."""
    d = W.shape[1]; zst = torch.tensor(zst_np); nz = zst.norm(dim=1, keepdim=True) + 1e-9
    z = torch.zeros(len(users), d)
    with torch.no_grad():
        for t in range(T):
            q = actor(z, t); a = (zst * q).sum(1, keepdim=True) / nz; z = z + ETA * a * q
        Sa = (z @ W.T + bdec).numpy().astype(np.float64)
    Wn = model.decoder.weight.detach().numpy(); bn = model.decoder.bias.detach().numpy()
    for i, x in enumerate(users):
        profset, tst = ar['SPL'][x]; rd = ar['rat_by_u_dict'][x]
        tlike = set(j for j in tst if rd[j] >= 4)
        if not tlike: continue
        na = A.ndcg_at10(Sa[i], tlike, profset, ar['headmask'], False)
        zg = unroll_static(zst_np[i], Q8_greedy); zsv = unroll_static(zst_np[i], svd8)
        ng = A.ndcg_at10(zg @ Wn.T + bn, tlike, profset, ar['headmask'], False)
        nsv = A.ndcg_at10(zsv @ Wn.T + bn, tlike, profset, ar['headmask'], False)
        ze = unroll_static(zst_np[i], ent8)
        ne = A.ndcg_at10(ze @ Wn.T + bn, tlike, profset, ar['headmask'], False)
        if None in (na, ng, nsv, ne): continue
        key = (x,)
        boot_actor.setdefault(key, []).append(na)
        boot_static.setdefault(key, []).append(max(ng, nsv))
        boot_ent.setdefault(key, []).append(ne)


def _paired_boot(A_d, B_d, nboot=2000):
    keys = [k for k in A_d if k in B_d]
    da = np.array([np.mean(A_d[k]) for k in keys])
    db = np.array([np.mean(B_d[k]) for k in keys])
    diff = da - db; n = len(diff)
    rng = np.random.default_rng(0)
    means = np.array([diff[rng.integers(0, n, n)].mean() for _ in range(nboot)])
    lo, hi = np.percentile(means, [2.5, 97.5])
    return {'mean_diff': float(diff.mean()), 'ci95': [float(lo), float(hi)],
            'p_gt0': float((means > 0).mean()), 'n_users': int(n)}


# ------------------------------------------------------------------ io
def _save_json(stage, obj):
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
    ap.add_argument('--field', action='store_true')
    args = ap.parse_args()
    if args.stage == 'static':
        stage_static()
    elif args.stage == 'train':
        stage_train(args.tseed, args.field)
    else:
        stage_eval()


if __name__ == '__main__':
    main()
