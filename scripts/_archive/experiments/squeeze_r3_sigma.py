"""
squeeze_r3_sigma.py -- INSTRUMENT 2.0 "squeeze arc" RUNG 3-SIGMA: an UNCERTAINTY-CONDITIONED
continuous actor.

THE QUESTION (R2/R2b left open). Under the empirical noisy channel a zero-training noise-adapted
STATIC that repeat-probes the top informative axis (schedule [0,0,0,0,1,2,7,0], TEST 0.3143) BEATS
the noise-trained plain actor (0.2844). R2b showed the actor CAN tie the static (TEST 0.3177) but
only if WARM-STARTED from the winning repeat schedule (tie-by-construction) -- SGD from the neutral
SVD-8 warm start never found it => an OPTIMIZATION gap, not an expressiveness gap.

HYPOTHESIS. The plain feed-forward actor conditions only on its (noise-corrupted) belief z_t; it has
no explicit signal for "how much uncertainty remains along each informative axis," which is exactly
the quantity that says "this axis is still noisy -> probe it again." Give the actor that signal --
the per-coordinate posterior VARIANCE from a decoder-metric Kalman belief tracker (reduced to the
top-K informative decoder-SVD subspace) -- and test whether SGD can now DISCOVER the repeat-probe
strategy on its own, WITHOUT the tie-by-construction warm start (BC-warm stays neutral = SVD-8).

The reduced-Kalman covariance update is MEASUREMENT-INDEPENDENT (depends only on the directions the
actor has emitted so far), so the remaining-uncertainty feature is a legitimate, cheap per-turn
input. Under the noisy channel R is large => one probe of an axis only PARTIALLY collapses its
variance, so the sigma signal keeps flagging "axis 0 still uncertain" and rewards re-probing; under a
clean (small-R) channel one probe collapses it and the signal says "move on." That is precisely the
clean-vs-noisy behaviour split the rung is about.

Sigma feature (per axis k in top-K decoder-SVD basis): feat_k = sqrt(C_kk / lam_k) in [0,1] =
fraction of the prior std still remaining. feat=1 at t=0, -> 0 as axis k is resolved.

Frozen invariants: instrument .cache/instrument2/ml1m_recvae_d512_best.pt; operator z'=z+16*a*q,
z0=0; arena ml1m_arena; seeds {1,2,3,7,11}; te[300:] TEST (304 users); T=8; NDCG@10 full + Cremonesi
tail. Channel .cache/instrument2/p4c_channel.json. Answers scale-matched + per-user centered exactly
as p4c_answer_sources.py / squeeze_r2.py. Belief-tracker R = 32 (noisy) / 1 (clean) reusing R1's
decoder-metric noisy R; K=32 top decoder-SVD dirs (== R2's greedy pool).

Anchors (noisy channel, 5-seed TEST): plain noise-actor 0.2844/0.1010 | noise-adapted static-repeat
0.3143/0.1167 | R2b tie-construct 0.3177/0.1168. Clean anchor: additive actor 0.4901/0.2973.

Usage: OMP_NUM_THREADS=4 python scripts/instrument2/squeeze_r3_sigma.py {train|eval|clean|all}
       [--tseeds 0,1]
"""
import os, sys, json, time, argparse
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ml1m_arena as A
import p4a_battery as P
import p4c_answer_sources as C4
import squeeze_r01 as R1
import squeeze_r2 as R2
from p4a_bootstrap import boot

DEVICE = torch.device('cpu'); torch.set_num_threads(4)
CK = '.cache/instrument2'
NI = 3706
ETA = 16.0
T = 8
SEEDS = [1, 2, 3, 7, 11]
NOISE = 1.0
K = 32                    # top-K decoder-SVD dirs for the reduced-Kalman uncertainty subspace
R_NOISY = 32.0            # belief-tracker obs-noise (reuse R1's decoder-metric noisy R)
R_CLEAN = 1.0
OUT = f'{CK}/squeeze_r3_sigma.json'

_CH, EDGES, BIN_MEAN, BIN_DIST = C4.load_channel()


# ============================================================ sigma-conditioned actor
class SigmaActor(torch.nn.Module):
    """MLP(z_t, sigma_t, turn) -> unit direction q_t. sigma_t = per-axis remaining-uncertainty
    (K,) from the reduced decoder-metric Kalman covariance. Architecture otherwise matches
    P.Actor (2 hidden SiLU layers, h=512)."""
    def __init__(self, d, K, T=T, h=512):
        super().__init__()
        self.T = T; self.K = K; self.d = d
        self.net = torch.nn.Sequential(
            torch.nn.Linear(d + K + T, h), torch.nn.SiLU(),
            torch.nn.Linear(h, h), torch.nn.SiLU(),
            torch.nn.Linear(h, d))

    def forward(self, z, sig, t):
        oh = torch.zeros(z.shape[0], self.T, device=z.device); oh[:, t] = 1.0
        q = self.net(torch.cat([z, sig, oh], dim=-1))
        return q / (q.norm(dim=-1, keepdim=True) + 1e-9)


def bc_warm_sigma(actor, svd8, d, K, steps=1200):
    """BC-warm SigmaActor to emit the NEUTRAL static SVD-8 basis per turn, for ANY belief z AND
    any sigma input (feed random sigma in [0,1]^K). This is the SAME neutral warm start the plain
    noisy actor used (svd8, NOT the repeat schedule) -- so whatever repeat behaviour appears is
    DISCOVERED by SGD, not constructed."""
    tgt = torch.tensor(svd8)
    opt = torch.optim.Adam(actor.parameters(), lr=1e-3)
    rng = np.random.default_rng(0)
    for s in range(steps):
        B = 128
        coef = torch.tensor(rng.standard_normal((B, T)).astype(np.float32)) * 8.0
        z = coef @ tgt + 0.5 * torch.tensor(rng.standard_normal((B, d)).astype(np.float32))
        z = z * torch.tensor(rng.uniform(0, 1, (B, 1)).astype(np.float32))
        sig = torch.tensor(rng.uniform(0, 1, (B, K)).astype(np.float32))
        loss = 0.0
        for t in range(T):
            q = actor(z, sig, t)
            loss = loss + (1.0 - (q * tgt[t]).sum(1)).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    return actor


# ============================================================ decoder-metric reduced-Kalman geometry
def build_geometry(model):
    """Return (svd_all[:K] as V (K,d), lam[:K] reduced-subspace prior variances)."""
    d = model.decoder.in_features
    rd_all = {x: dict(v) for x, v in A.load_arena(seed=123)['rat_by_u'].items()}
    _, _, cov_pop = R1.population_zstar_cov(model, rd_all)
    svd_all = P.decoder_svd_dirs(model, d).astype(np.float64)
    Wd = model.decoder.weight.detach().numpy(); Wc = Wd - Wd.mean(0, keepdims=True)
    sv = np.linalg.svd(Wc, full_matrices=False, compute_uv=False)[:d].astype(np.float64)
    lam = sv ** 2; lam = lam / lam.sum() * np.trace(cov_pop)      # == make_prior('decoder') eigenvalues
    V = svd_all[:K].astype(np.float32)
    lamK = lam[:K].astype(np.float32)
    return V, lamK


# ============================================================ TRAIN (differentiable unroll under channel)
def train_sigma(tseed, alpha, beta, sig_a, V, lamK):
    model, d = P.load_model()
    W = torch.tensor(model.decoder.weight.detach().numpy(), dtype=torch.float32)
    bdc = torch.tensor(model.decoder.bias.detach().numpy(), dtype=torch.float32)
    svd8 = P.decoder_svd_dirs(model, 8)
    ar123 = A.load_arena(seed=123); rd_all = {x: dict(v) for x, v in ar123['rat_by_u'].items()}
    trU = [x for x in ar123['trU'] if sum(1 for j, r in rd_all[x].items() if r >= 4) >= 4]
    Zstar = np.zeros((len(trU), d), np.float32)
    for st in range(0, len(trU), 500):
        chk = trU[st:st + 500]; Xd = np.zeros((len(chk), NI), np.float32)
        for r, x in enumerate(chk):
            lk = [j for j, rr in rd_all[x].items() if rr >= 4]; Xd[r, lk] = 1.0
        Zstar[st:st + len(chk)] = P.enc_mu(model, Xd)
    Zstar_t = torch.tensor(Zstar); znorm = Zstar_t.norm(dim=1, keepdim=True) + 1e-9
    with torch.no_grad():
        Steach = Zstar_t @ W.T + bdc
    topk = torch.topk(Steach, 20, dim=1).indices

    arv = P.arena_seed(1, rd_all); val = arv['val_users']
    zsv = torch.tensor(P.build_zstar(model, arv, val))

    V_t = torch.tensor(V); lam_t = torch.tensor(lamK)            # (K,d),(K,)
    R = R_NOISY

    def unroll_val():
        actor.eval()
        with torch.no_grad():
            Uv = len(val); z = torch.zeros(Uv, d)
            C = torch.diag(lam_t).expand(Uv, K, K).clone()
            nzv = zsv.norm(dim=1) + 1e-9
            for t in range(T):
                dg = torch.diagonal(C, dim1=1, dim2=2).clamp_min(0)
                sig = torch.sqrt(dg / lam_t)
                q = actor(z, sig, t)
                s = (zsv * q).sum(1) / nzv
                a = alpha * s + beta + sig_a * torch.randn_like(s)
                z = z + ETA * a[:, None] * q
                g = q @ V_t.T
                Cg = torch.bmm(C, g.unsqueeze(2)).squeeze(2)
                Sd = (g * Cg).sum(1) + R
                C = C - torch.bmm(Cg.unsqueeze(2), Cg.unsqueeze(1)) / Sd.view(-1, 1, 1)
            S = (z @ W.T + bdc).numpy().astype(np.float64)
        af = at = 0.0; mf = mt = 0
        for i, x in enumerate(val):
            profset, tst = arv['SPL'][x]; rd = arv['rat_by_u_dict'][x]
            tl = set(j for j in tst if rd[j] >= 4)
            if not tl: continue
            nf = A.ndcg_at10(S[i], tl, profset, arv['headmask'], False)
            nt = A.ndcg_at10(S[i], tl, profset, arv['headmask'], True)
            if nf is not None: af += nf; mf += 1
            if nt is not None: at += nt; mt += 1
        return af / max(mf, 1), at / max(mt, 1)

    torch.manual_seed(3000 + tseed); np.random.seed(3000 + tseed)
    actor = SigmaActor(d, K)
    bc_warm_sigma(actor, svd8, d, K)
    opt = torch.optim.Adam(actor.parameters(), lr=3e-4)
    Ntr = len(trU); B = 256; EPOCHS = 20
    rng = np.random.default_rng(tseed)
    best_t = -1; best_state_t = {k: v.clone() for k, v in actor.state_dict().items()}; best_ep_t = -1
    best_f = -1; best_state_f = {k: v.clone() for k, v in actor.state_dict().items()}; best_ep_f = -1
    t0 = time.time()
    for ep in range(EPOCHS):
        actor.train(); perm = rng.permutation(Ntr)
        for st in range(0, Ntr, B):
            idx = perm[st:st + B]; zs = Zstar_t[idx]; nz = znorm[idx]; Bn = len(idx)
            z = torch.zeros(Bn, d)
            C = torch.diag(lam_t).expand(Bn, K, K).clone()
            for t in range(T):
                with torch.no_grad():
                    dg = torch.diagonal(C, dim1=1, dim2=2).clamp_min(0)
                    sig = torch.sqrt(dg / lam_t)                  # (Bn,K) detached
                q = actor(z, sig, t)
                s = (zs * q).sum(1, keepdim=True) / nz
                noise = sig_a * torch.randn_like(s)
                a = alpha * s + beta + noise.detach()
                z = z + ETA * a * q
                with torch.no_grad():
                    g = q.detach() @ V_t.T
                    Cg = torch.bmm(C, g.unsqueeze(2)).squeeze(2)
                    Sd = (g * Cg).sum(1) + R
                    C = C - torch.bmm(Cg.unsqueeze(2), Cg.unsqueeze(1)) / Sd.view(-1, 1, 1)
            cosT = (z * zs).sum(1) / (z.norm(dim=1) + 1e-9) / nz.squeeze(1)
            L_rec = (1.0 - cosT).mean()
            pos = topk[idx]; neg = torch.randint(0, NI, (Bn, 108))
            cand = torch.cat([pos, neg], 1)
            sc = (z[:, None, :] * W[cand]).sum(2) + bdc[cand]
            rel = torch.zeros_like(sc); rel[:, :20] = 1.0
            L = L_rec + 0.3 * P.approx_ndcg_loss(sc, rel)
            opt.zero_grad(); L.backward(); opt.step()
        vf, vt = unroll_val()
        if vt > best_t:
            best_t = vt; best_ep_t = ep; best_state_t = {k: v.clone() for k, v in actor.state_dict().items()}
        if vf > best_f:
            best_f = vf; best_ep_f = ep; best_state_f = {k: v.clone() for k, v in actor.state_dict().items()}
        print(f'[sigma-train s{tseed}] ep{ep:2d} val full {vf:.4f} tail {vt:.4f} | '
              f'best_tail {best_t:.4f}@{best_ep_t} best_full {best_f:.4f}@{best_ep_f} '
              f'[{time.time()-t0:.0f}s]', flush=True)
    torch.save({'state_tail': best_state_t, 'state_full': best_state_f, 'd': d, 'K': K,
                'best_ep_tail': best_ep_t, 'best_val_tail': best_t,
                'best_ep_full': best_ep_f, 'best_val_full': best_f,
                'alpha': alpha, 'beta': beta, 'sigma': sig_a},
               f'{CK}/p4c_actor_sigma_s{tseed}.pt')
    print(f'[sigma-train] saved p4c_actor_sigma_s{tseed}.pt', flush=True)
    return best_t, best_f


# ============================================================ sigma-actor rollout (numpy eval)
def sigma_rollout(actor, model, Wf, bf, zst, V, lamK, R, ans_fn, c, record=False):
    """Roll the sigma-actor over a cohort under ans_fn(t,q[U,d],s[U])->a[U] (raw), scale c.
    Reduced decoder-metric Kalman produces the per-turn sigma feature. Returns final Z (U,d) and
    optionally the per-turn emitted directions."""
    U, d = zst.shape; nz = np.linalg.norm(zst, axis=1) + 1e-9
    z = np.zeros((U, d), np.float32)
    C = np.broadcast_to(np.diag(lamK), (U, K, K)).astype(np.float32).copy()
    qrec = []
    for t in range(T):
        dg = np.clip(np.diagonal(C, axis1=1, axis2=2), 0, None)
        sig = np.sqrt(dg / lamK).astype(np.float32)               # (U,K)
        with torch.no_grad():
            q = actor(torch.tensor(z, dtype=torch.float32),
                      torch.tensor(sig, dtype=torch.float32), t).numpy()
        s = (zst * q).sum(1) / nz
        a = ans_fn(t, q, s)
        z = (z + ETA * (c * a)[:, None] * q).astype(np.float32)
        g = q @ V.T                                               # (U,K)
        Cg = np.einsum('ukj,uj->uk', C, g)
        Sd = (g * Cg).sum(1) + R
        C = C - (Cg[:, :, None] * Cg[:, None, :]) / Sd[:, None, None]
        if record:
            qrec.append(q.copy())
    return z, qrec


def sigma_native_scale(actor, model, Wf, bf, zst, V, lamK, R, emp_ans_fn):
    """c = rms(s)/rms(a_emp) along the NATIVE (a=s) trajectory (mirrors native_scale_for_actor)."""
    U, d = zst.shape; nz = np.linalg.norm(zst, axis=1) + 1e-9
    z = np.zeros((U, d), np.float32)
    C = np.broadcast_to(np.diag(lamK), (U, K, K)).astype(np.float32).copy()
    s_all = []; a_all = []
    for t in range(T):
        dg = np.clip(np.diagonal(C, axis1=1, axis2=2), 0, None)
        sig = np.sqrt(dg / lamK).astype(np.float32)
        with torch.no_grad():
            q = actor(torch.tensor(z, dtype=torch.float32),
                      torch.tensor(sig, dtype=torch.float32), t).numpy()
        s = (zst * q).sum(1) / nz
        s_all.append(s); a_all.append(emp_ans_fn(t, q, s))
        z = (z + ETA * s[:, None] * q).astype(np.float32)
        g = q @ V.T; Cg = np.einsum('ukj,uj->uk', C, g); Sd = (g * Cg).sum(1) + R
        C = C - (Cg[:, :, None] * Cg[:, None, :]) / Sd[:, None, None]
    rs = np.sqrt(np.mean(np.square(np.concatenate(s_all))) + 1e-12)
    ra = np.sqrt(np.mean(np.square(np.concatenate(a_all))) + 1e-12)
    return float(rs / ra) if ra > 0 else 1.0


def eval_sigma_cohort(actor, model, Wf, bf, ar, users, zst, V, lamK, R, um, seed, record=False):
    """Sigma-actor under the SAMPLED empirical channel, scale-matched. Returns (full,tail,pu,qrec)."""
    rng = np.random.default_rng(seed)
    af = lambda t, q, s: C4.sample_channel_a(s, EDGES, BIN_MEAN, BIN_DIST, NOISE, rng,
                                             center=np.array([um[u] for u in users]))
    c = sigma_native_scale(actor, model, Wf, bf, zst, V, lamK, R, af)
    z, qrec = sigma_rollout(actor, model, Wf, bf, zst, V, lamK, R, af, c, record=record)
    S = (z @ Wf.T + bf).astype(np.float64)
    afu = atu = 0.0; mf = mt = 0; pu = {}
    for i, x in enumerate(users):
        nf = C4.ndcg_user(S[i], ar, x, False); nt = C4.ndcg_user(S[i], ar, x, True)
        if nf is not None: afu += nf; mf += 1; pu[x] = nf
        if nt is not None: atu += nt; mt += 1
    return afu / max(mf, 1), atu / max(mt, 1), pu, qrec


def eval_sigma_clean(actor, model, Wf, bf, ar, users, zst, V, lamK, R):
    """Sigma-actor under the clean geometric answerer a=cos(z*,q)."""
    af = lambda t, q, s: s
    z, _ = sigma_rollout(actor, model, Wf, bf, zst, V, lamK, R, af, 1.0)
    S = (z @ Wf.T + bf).astype(np.float64)
    afu = atu = 0.0; mf = mt = 0; pu = {}
    for i, x in enumerate(users):
        nf = C4.ndcg_user(S[i], ar, x, False); nt = C4.ndcg_user(S[i], ar, x, True)
        if nf is not None: afu += nf; mf += 1; pu[x] = nf
        if nt is not None: atu += nt; mt += 1
    return afu / max(mf, 1), atu / max(mt, 1), pu


# ============================================================ schedule inspection
def inspect_schedule(qrec, svd_all8):
    """For each turn, over users: argmax |cos| to the top-8 decoder-SVD dirs + mean cos. Reveals
    which informative axis the actor probes each turn and whether it repeats."""
    sched = []
    for t in range(T):
        q = qrec[t]                                              # (U,d)
        cos = q @ svd_all8.T                                    # (U,8)
        idx = np.argmax(np.abs(cos), axis=1)                    # per-user dominant axis
        # modal axis + mean |cos| to it
        vals, cnts = np.unique(idx, return_counts=True)
        modal = int(vals[np.argmax(cnts)]); frac = float(cnts.max() / len(idx))
        meancos = float(np.abs(cos[np.arange(len(idx)), idx]).mean())
        sched.append({'turn': t, 'modal_svd_axis': modal, 'modal_frac': frac,
                      'mean_abs_cos_to_dominant': meancos})
    return sched


# ============================================================ main stages
def load_channel_law():
    """Channel effective linear law a = alpha*s + beta + N(0,sigma) (same fit as part1_trainnoisy)."""
    blob = torch.load(f'{CK}/p4c_actor_noisy_s0.pt', map_location=DEVICE)
    return blob['alpha'], blob['beta'], blob['sigma']


def stage_train(tseeds):
    model, d = P.load_model()
    V, lamK = build_geometry(model)
    alpha, beta, sig_a = load_channel_law()
    print(f'[train] channel law a={alpha:.3f}s+{beta:.3f}, sigma={sig_a:.3f}; K={K} R={R_NOISY}', flush=True)
    res = _load()
    res.setdefault('train', {})['channel_law'] = {'alpha': alpha, 'beta': beta, 'sigma': sig_a,
                                                  'K': K, 'R_noisy': R_NOISY}
    vals = {}
    for ts in tseeds:
        bt, bf = train_sigma(ts, alpha, beta, sig_a, V, lamK)
        vals[f's{ts}'] = {'best_val_tail': bt, 'best_val_full': bf}
    res['train']['val'] = vals
    _save(res)


def load_sigma_actors(d, tseeds, sel='tail'):
    key = 'state_tail' if sel == 'tail' else 'state_full'
    acts = {}
    for ts in tseeds:
        blob = torch.load(f'{CK}/p4c_actor_sigma_s{ts}.pt', map_location=DEVICE)
        a = SigmaActor(d, blob['K']); a.load_state_dict(blob[key]); a.eval()
        acts[f's{ts}'] = a
    return acts


def stage_eval(tseeds):
    t0 = time.time()
    model, d = P.load_model()
    Wf = model.decoder.weight.detach().numpy(); bf = model.decoder.bias.detach().numpy()
    W64 = Wf.astype(np.float64); b64 = bf.astype(np.float64)
    V, lamK = build_geometry(model)
    svd8 = P.decoder_svd_dirs(model, 8).astype(np.float64)
    rd_all = {x: dict(v) for x, v in A.load_arena(seed=123)['rat_by_u'].items()}

    # winning noise-adapted repeat schedule + plain noisy actors (R2 comparators, same CRN)
    r2 = json.load(open(f'{CK}/squeeze_r2_noise.json'))
    idx_r = r2['greedy']['repeat']['schedule_idx']
    poolQ = P.decoder_svd_dirs(model, R2.POOL)
    sched_r = np.stack([poolQ[j] for j in idx_r]).astype(np.float32)
    nacts = R2.load_noisy_actors(d)                              # s0,s1,s2 plain noise-trained

    res = _load()
    for sel in ['tail', 'full']:
        acts = load_sigma_actors(d, tseeds, sel=sel)
        perseed = {'sigma_actor': {'f': [], 't': []}, 'plain_actor': {'f': [], 't': []},
                   'static_repeat': {'f': [], 't': []}}
        pu_sig = {}; pu_plain = {}; pu_stat = {}
        qrec_last = None
        for sd in SEEDS:
            ar = P.arena_seed(sd, rd_all); test = ar['test_users']
            zst = P.build_zstar(model, ar, test); um = C4.user_means(ar, test, rd_all)

            # sigma-actor seed-avg over trseeds (same rng/seed convention as R2: seed=sd)
            fs = []; ts = []; puS = {}
            for j, (nm, act) in enumerate(acts.items()):
                f, t, pu, qr = eval_sigma_cohort(act, model, Wf, bf, ar, test, zst, V, lamK,
                                                 R_NOISY, um, sd, record=(sd == SEEDS[0] and j == 0))
                fs.append(f); ts.append(t)
                for x, v in pu.items(): puS.setdefault(x, []).append(v)
                if sd == SEEDS[0] and j == 0:
                    qrec_last = qr
            perseed['sigma_actor']['f'].append(float(np.mean(fs)))
            perseed['sigma_actor']['t'].append(float(np.mean(ts)))
            for x, vs in puS.items(): pu_sig.setdefault(x, []).append(float(np.mean(vs)))

            # plain noise-trained actor seed-avg (s0/s1/s2), R2 recipe
            rng_a = np.random.default_rng(sd)
            afp = lambda t, q, s, users: C4.sample_channel_a(
                s, EDGES, BIN_MEAN, BIN_DIST, NOISE, rng_a, center=np.array([um[u] for u in users]))
            fp = []; tp = []; puP = {}
            for k in ['s0', 's1', 's2']:
                act = nacts[k][0]
                c = C4.native_scale_for_actor(act, model, Wf, bf, ar, test, zst, afp)
                f, t, pu = R2.actor_noisy_peruser(act, model, Wf, bf, ar, test, zst, afp, c)
                fp.append(f); tp.append(t)
                for x, v in pu.items(): puP.setdefault(x, []).append(v)
            perseed['plain_actor']['f'].append(float(np.mean(fp)))
            perseed['plain_actor']['t'].append(float(np.mean(tp)))
            for x, vs in puP.items(): pu_plain.setdefault(x, []).append(float(np.mean(vs)))

            # noise-adapted static-repeat (R2 winner)
            rng_sr = np.random.default_rng(sd)
            fr, tr, _, pur = R2.run_static_noisy_peruser(model, W64, b64, ar, test, zst, sched_r, rng_sr, um)
            perseed['static_repeat']['f'].append(fr); perseed['static_repeat']['t'].append(tr)
            for x, v in pur.items(): pu_stat.setdefault(x, []).append(v)

            print(f'[eval:{sel}] seed{sd}: sigma {perseed["sigma_actor"]["f"][-1]:.4f}/'
                  f'{perseed["sigma_actor"]["t"][-1]:.4f} | plain {perseed["plain_actor"]["f"][-1]:.4f}/'
                  f'{perseed["plain_actor"]["t"][-1]:.4f} | static-r {fr:.4f}/{tr:.4f} '
                  f'[{time.time()-t0:.0f}s]', flush=True)

        def agg(a):
            return {'full': float(np.mean(a['f'])), 'tail': float(np.mean(a['t'])),
                    'full_sd': float(np.std(a['f'])), 'tail_sd': float(np.std(a['t'])),
                    'per_seed_full': [float(x) for x in a['f']]}

        def seedavg(pu): return {x: float(np.mean(vs)) for x, vs in pu.items()}
        A_sig = seedavg(pu_sig); A_plain = seedavg(pu_plain); A_stat = seedavg(pu_stat)

        def paired(A1, B1):
            keys = [k for k in A1 if k in B1]
            return boot([A1[k] - B1[k] for k in keys])

        block = {
            'test': {k: agg(v) for k, v in perseed.items()},
            'bootstrap': {
                'sigma_minus_plain_actor': paired(A_sig, A_plain),
                'sigma_minus_static_repeat': paired(A_sig, A_stat)},
            'learned_schedule': inspect_schedule(qrec_last, svd8) if qrec_last is not None else None}
        res[f'eval_{sel}'] = block
        _save(res)
        _print_eval(sel, block)


def stage_clean(tseeds):
    """Optional clean-channel run vs the 0.4901 additive actor anchor."""
    model, d = P.load_model()
    Wf = model.decoder.weight.detach().numpy(); bf = model.decoder.bias.detach().numpy()
    V, lamK = build_geometry(model)
    rd_all = {x: dict(v) for x, v in A.load_arena(seed=123)['rat_by_u'].items()}
    res = _load()
    for sel in ['tail', 'full']:
        acts = load_sigma_actors(d, tseeds, sel=sel)
        fss = []; tss = []
        for sd in SEEDS:
            ar = P.arena_seed(sd, rd_all); test = ar['test_users']
            zst = P.build_zstar(model, ar, test)
            fs = []; ts = []
            for nm, act in acts.items():
                f, t, _ = eval_sigma_clean(act, model, Wf, bf, ar, test, zst, V, lamK, R_CLEAN)
                fs.append(f); ts.append(t)
            fss.append(np.mean(fs)); tss.append(np.mean(ts))
            print(f'[clean:{sel}] seed{sd}: sigma {np.mean(fs):.4f}/{np.mean(ts):.4f}', flush=True)
        res.setdefault('clean', {})[sel] = {'full': float(np.mean(fss)), 'tail': float(np.mean(tss)),
                                            'full_sd': float(np.std(fss))}
        _save(res)
    print(f"[clean] additive-actor anchor 0.4901/0.2973", flush=True)


def _print_eval(sel, b):
    t = b['test']
    print(f'\n============ SQUEEZE R3-SIGMA EVAL [{sel}-selected] ============', flush=True)
    print(f"sigma-actor (uncertainty-conditioned): {t['sigma_actor']['full']:.4f} "
          f"(sd {t['sigma_actor']['full_sd']:.4f}) / tail {t['sigma_actor']['tail']:.4f}", flush=True)
    print(f"plain noise-actor (anchor 0.2844):     {t['plain_actor']['full']:.4f} / "
          f"tail {t['plain_actor']['tail']:.4f}", flush=True)
    print(f"noise-adapted static-repeat (0.3143):  {t['static_repeat']['full']:.4f} / "
          f"tail {t['static_repeat']['tail']:.4f}", flush=True)
    bp = b['bootstrap']['sigma_minus_plain_actor']; bs = b['bootstrap']['sigma_minus_static_repeat']
    print(f"sigma - plain:  dfull={bp['mean_diff']:+.4f} CI{[round(x,4) for x in bp['ci95']]} "
          f"p={bp['p_gt0']:.3f}", flush=True)
    print(f"sigma - static: dfull={bs['mean_diff']:+.4f} CI{[round(x,4) for x in bs['ci95']]} "
          f"p={bs['p_gt0']:.3f}", flush=True)
    if b['learned_schedule']:
        sc = ' '.join(f"t{r['turn']}:ax{r['modal_svd_axis']}({r['modal_frac']:.2f},"
                      f"c{r['mean_abs_cos_to_dominant']:.2f})" for r in b['learned_schedule'])
        print(f"learned schedule (modal SVD axis/turn): {sc}", flush=True)


def _save(obj): json.dump(obj, open(OUT, 'w'), indent=2); print(f'[saved] {OUT}', flush=True)
def _load(): return json.load(open(OUT)) if os.path.exists(OUT) else {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('stage', choices=['train', 'eval', 'clean', 'all'])
    ap.add_argument('--tseeds', type=str, default='0,1')
    args = ap.parse_args()
    tseeds = [int(x) for x in args.tseeds.split(',')]
    if args.stage in ('train', 'all'):
        stage_train(tseeds)
    if args.stage in ('eval', 'all'):
        stage_eval(tseeds)
    if args.stage in ('clean', 'all'):
        stage_clean(tseeds)


if __name__ == '__main__':
    main()
