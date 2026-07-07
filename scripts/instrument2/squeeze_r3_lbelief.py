"""
squeeze_r3_lbelief.py -- INSTRUMENT 2.0 "squeeze arc" RUNG 3 (learned belief):
replace the HAND-BUILT additive belief operator z' = z + eta*a*q with a LEARNED belief-update
head g_phi:(z_t, q_t, a_t, turn) -> z_{t+1}, trained under the DEPLOYMENT CHANNEL via
differentiable unroll (recon + softNDCG). Question: can a LEARNED update beat the fixed additive
operator where the hand-built decoder-metric Kalman (SQUEEZE_R3_BELIEF.md) could not?

Design -- GATED RESIDUAL, additive-initialized (the clean isolation trick):
    z_{t+1} = z_t + ETA * a * (g (x) q) + c ,   (g, c) = head(z_t, q_t, a_t, turn)
  head final layers init to ZERO  =>  g=1, c=0  =>  z_{t+1} = z_t + ETA*a*q  EXACTLY the additive
  operator at initialization. Any TEST improvement over additive is therefore unambiguously the
  *learned* update adding value (not re-tuning eta / re-inflation). The gate g can absorb the
  channel's effective step size (so NO cmul is needed under noise -- the head learns its own scale
  from the deployment channel, which is the whole point of training under the channel).

Two policy regimes per channel:
  FROZEN  -- pretrained actor (clean: p4a_actor_s0; noisy: p4c_actor_noisy_s0) is FROZEN (params
             requires_grad=False, but autograd still flows through it into the belief head). Isolates
             "a learned UPDATE on top of the SAME learned directions" -- the direct R3 question.
  JOINT   -- actor initialized from that pretrained actor AND trainable together with the head.
             The strongest realizable learned pipeline (best chance for the learned belief).

Clean channel: answers a = cos(z*, q) (native). Compare vs additive.actor 0.4901/0.2973.
Noisy channel: train with the fitted linear channel law a = alpha*s + beta + N(0,sigma) (as
  p4c_actor_noisy), eval under the SAMPLED empirical channel (per-user centered). Compare vs the
  R2 fair comparators: noise-trained actor 0.2844/0.1010 and noise-adapted static-repeat 0.3143/0.1167.

Frozen invariants: instrument .cache/instrument2/ml1m_recvae_d512_best.pt; ETA=16, z0=0; arena
ml1m_arena; seeds {1,2,3,7,11}; te[300:] TEST (304 users); T=8; NDCG@10 full + Cremonesi tail;
channel .cache/instrument2/p4c_channel.json. Artifact .cache/instrument2/squeeze_r3_lbelief.json.

Usage: OMP_NUM_THREADS=4 python scripts/instrument2/squeeze_r3_lbelief.py {clean|noisy|both}
                                                 [--epochs 16] [--fseeds 0,1] [--jseeds 0]
"""
import os, sys, json, time, argparse
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ml1m_arena as A
import p4a_battery as P
import squeeze_r01 as R1
import p4c_answer_sources as C4
import squeeze_r2 as R2

DEVICE = torch.device('cpu'); torch.set_num_threads(4)
CK = '.cache/instrument2'
NI = 3706
ETA = 16.0
T = 8
SEEDS = [1, 2, 3, 7, 11]
OUT = f'{CK}/squeeze_r3_lbelief.json'

# anchors (5-seed TEST)
ANCH_CLEAN = {'additive_actor': (0.4901, 0.2973), 'additive_svd8': (0.4709, 0.2916)}
ANCH_NOISY = {'r2_noisy_actor': (0.2844, 0.1010), 'r2_static_repeat': (0.3143, 0.1167)}


# ============================================================ learned belief-update head
class LearnedBelief(torch.nn.Module):
    """Gated residual belief update, additive-initialized.
    z_{t+1} = z_t + ETA*a*(g(x)q) + c ;  (g,c)=head(z/ETA, q, a, turn) ; init g=1, c=0."""
    def __init__(self, d, T=T, h=256):
        super().__init__()
        self.T = T; self.d = d
        self.net = torch.nn.Sequential(
            torch.nn.Linear(2 * d + 1 + T, h), torch.nn.SiLU(),
            torch.nn.Linear(h, h), torch.nn.SiLU())
        self.gate = torch.nn.Linear(h, d)
        self.corr = torch.nn.Linear(h, d)
        for lin in (self.gate, self.corr):
            torch.nn.init.zeros_(lin.weight); torch.nn.init.zeros_(lin.bias)

    def forward(self, z, q, a, t):
        # z,q: (B,d) ; a: (B,1)
        oh = torch.zeros(z.shape[0], self.T, device=z.device); oh[:, t] = 1.0
        h = self.net(torch.cat([z / ETA, q, a, oh], dim=-1))
        g = 1.0 + self.gate(h)
        c = self.corr(h)
        return z + ETA * a * (g * q) + c


# ============================================================ training data / teacher (shared)
def _train_setup(model, d):
    """Zstar_t, znorm, teacher top-20, val cohort -- mirrors p4a_battery.stage_train."""
    W = torch.tensor(model.decoder.weight.detach().numpy(), dtype=torch.float32)
    bdec = torch.tensor(model.decoder.bias.detach().numpy(), dtype=torch.float32)
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
        topk = torch.topk(Zstar_t @ W.T + bdec, 20, dim=1).indices
    arv = P.arena_seed(1, rd_all); val = arv['val_users']
    zsv = torch.tensor(P.build_zstar(model, arv, val))
    return W, bdec, rd_all, trU, Zstar_t, znorm, topk, arv, val, zsv


def _val_belief(actor, belief, W, bdec, arv, val, zsv, d, noisy, law):
    """SELVAL full/tail under the channel used in training (native clean; linear-law noisy)."""
    actor.eval(); belief.eval()
    with torch.no_grad():
        z = torch.zeros(len(val), d); nz = zsv.norm(dim=1, keepdim=True) + 1e-9
        for t in range(T):
            q = actor(z, t)
            s = (zsv * q).sum(1, keepdim=True) / nz
            if noisy:
                a = law[0] * s + law[1] + law[2] * torch.randn_like(s)
            else:
                a = s
            z = belief(z, q, a, t)
        S = (z @ W.T + bdec).numpy().astype(np.float64)
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


def train_lbelief(model, d, channel, joint, tseed, epochs):
    """Train the learned belief head (and, if joint, the actor) via differentiable unroll.
    Returns (actor, belief, best_val_full, best_val_tail, best_ep)."""
    noisy = (channel == 'noisy')
    W, bdec, rd_all, trU, Zstar_t, znorm, topk, arv, val, zsv = _train_setup(model, d)

    # pretrained actor init
    if noisy:
        blob = torch.load(f'{CK}/p4c_actor_noisy_s0.pt', map_location=DEVICE)
        law = (blob['alpha'], blob['beta'], blob['sigma'])
    else:
        blob = torch.load(f'{CK}/p4a_actor_s0.pt', map_location=DEVICE)
        law = None
    actor = P.Actor(d); actor.load_state_dict(blob['state'])
    for p in actor.parameters():
        p.requires_grad = bool(joint)

    torch.manual_seed(3000 + tseed); np.random.seed(3000 + tseed)
    belief = LearnedBelief(d)
    params = list(belief.parameters()) + (list(actor.parameters()) if joint else [])
    opt = torch.optim.Adam(params, lr=3e-4)

    Ntr = len(trU); B = 256; rng = np.random.default_rng(3000 + tseed)
    sigma = law[2] if noisy else 0.0
    best_f = -1; best_t = -1; best_ep = -1
    best = {'belief': {k: v.clone() for k, v in belief.state_dict().items()},
            'actor': {k: v.clone() for k, v in actor.state_dict().items()}}
    t0 = time.time()
    for ep in range(epochs):
        belief.train();  actor.train() if joint else actor.eval()
        perm = rng.permutation(Ntr)
        for st in range(0, Ntr, B):
            idx = perm[st:st + B]; zs = Zstar_t[idx]; nz = znorm[idx]
            z = torch.zeros(len(idx), d)
            for t in range(T):
                q = actor(z, t)
                s = (zs * q).sum(1, keepdim=True) / nz
                if noisy:
                    a = law[0] * s + law[1] + (sigma * torch.randn_like(s)).detach()
                else:
                    a = s
                z = belief(z, q, a, t)
            cosT = (z * zs).sum(1) / (z.norm(dim=1) + 1e-9) / nz.squeeze(1)
            L_rec = (1.0 - cosT).mean()
            pos = topk[idx]; neg = torch.randint(0, NI, (len(idx), 108))
            cand = torch.cat([pos, neg], 1)
            sc = (z[:, None, :] * W[cand]).sum(2) + bdec[cand]
            rel = torch.zeros_like(sc); rel[:, :20] = 1.0
            L = L_rec + 0.3 * P.approx_ndcg_loss(sc, rel)
            opt.zero_grad(); L.backward(); opt.step()
        vf, vt = _val_belief(actor, belief, W, bdec, arv, val, zsv, d, noisy, law)
        if vf > best_f:                                     # SELVAL on FULL (headline metric)
            best_f = vf; best_t = vt; best_ep = ep
            best = {'belief': {k: v.clone() for k, v in belief.state_dict().items()},
                    'actor': {k: v.clone() for k, v in actor.state_dict().items()}}
        print(f'[train {channel}/{"joint" if joint else "frozen"} s{tseed}] ep{ep:2d} '
              f'val full {vf:.4f} tail {vt:.4f} | best_full {best_f:.4f}@{best_ep} '
              f'[{time.time()-t0:.0f}s]', flush=True)
    belief.load_state_dict(best['belief']); actor.load_state_dict(best['actor'])
    belief.eval(); actor.eval()
    return actor, belief, best_f, best_t, best_ep


# ============================================================ TEST eval of the learned belief
def eval_belief_clean(actor, belief, Wf, bf, ar, users, zst, d):
    """Native cos answers + learned belief update. Returns (full, tail, peruser_full)."""
    U = len(users); z = torch.zeros(U, d)
    zt = torch.tensor(zst); nz = zt.norm(dim=1, keepdim=True) + 1e-9
    with torch.no_grad():
        for t in range(T):
            q = actor(z, t)
            s = (zt * q).sum(1, keepdim=True) / nz
            z = belief(z, q, s, t)
        S = (z.numpy() @ Wf.T + bf).astype(np.float64)
    return _score(S, ar, users)


def eval_belief_noisy(actor, belief, Wf, bf, ar, users, zst, um, d, seed):
    """Sampled empirical channel (per-user centered, NO cmul) + learned belief. -> (full,tail,pu)."""
    rng = np.random.default_rng(seed)
    U = len(users); z = torch.zeros(U, d)
    zt = torch.tensor(zst); nz = zt.norm(dim=1, keepdim=True) + 1e-9
    ctr = np.array([um[u] for u in users])
    with torch.no_grad():
        for t in range(T):
            q = actor(z, t)
            s = ((zt * q).sum(1) / nz.squeeze(1)).numpy()
            a = C4.sample_channel_a(s, R2.EDGES, R2.BIN_MEAN, R2.BIN_DIST, 1.0, rng, center=ctr)
            a_t = torch.tensor(a, dtype=torch.float32)[:, None]
            z = belief(z, q, a_t, t)
        S = (z.numpy() @ Wf.T + bf).astype(np.float64)
    return _score(S, ar, users)


def _score(S, ar, users):
    af = at = 0.0; mf = mt = 0; per = {}
    for i, x in enumerate(users):
        nf = C4.ndcg_user(S[i], ar, x, False); nt = C4.ndcg_user(S[i], ar, x, True)
        if nf is not None: af += nf; mf += 1; per[x] = nf
        if nt is not None: at += nt; mt += 1
    return af / max(mf, 1), at / max(mt, 1), per


# additive.actor per-user (clean) anchor on THIS run's cohorts (native)
def additive_actor_clean_peruser(actor, Wf, bf, ar, users, zst, d):
    U = len(users); z = torch.zeros(U, d)
    zt = torch.tensor(zst); nz = zt.norm(dim=1, keepdim=True) + 1e-9
    with torch.no_grad():
        for t in range(T):
            q = actor(z, t); a = (zt * q).sum(1, keepdim=True) / nz; z = z + ETA * a * q
        S = (z.numpy() @ Wf.T + bf).astype(np.float64)
    return _score(S, ar, users)


# ============================================================ driver
def run(channel, epochs, fseeds, jseeds):
    t0 = time.time()
    model, d = P.load_model()
    Wf = model.decoder.weight.detach().numpy(); bf = model.decoder.bias.detach().numpy()
    rd_all = {x: dict(v) for x, v in A.load_arena(seed=123)['rat_by_u'].items()}
    noisy = (channel == 'noisy')
    anchors = ANCH_NOISY if noisy else ANCH_CLEAN
    print(f'\n########## RUNG 3 LEARNED-BELIEF [{channel}] ##########', flush=True)

    # ---- train FROZEN-actor learned-belief over fseeds, val-select best ----
    frozen_runs = []
    for ts in fseeds:
        act, bel, vf, vt, ep = train_lbelief(model, d, channel, joint=False, tseed=ts, epochs=epochs)
        frozen_runs.append({'ts': ts, 'val_full': vf, 'val_tail': vt, 'ep': ep,
                            'actor': act, 'belief': bel})
        torch.save({'belief': bel.state_dict(), 'val_full': vf, 'val_tail': vt, 'ep': ep,
                    'channel': channel, 'joint': False},
                   f'{CK}/r3_lbelief_{channel}_frozen_s{ts}.pt')
    fbest = max(frozen_runs, key=lambda r: r['val_full'])
    print(f'[{channel}] FROZEN val-winner: s{fbest["ts"]} (val_full {fbest["val_full"]:.4f})', flush=True)

    # ---- train JOINT actor+belief over jseeds, val-select best ----
    joint_best = None
    if jseeds:
        joint_runs = []
        for ts in jseeds:
            act, bel, vf, vt, ep = train_lbelief(model, d, channel, joint=True, tseed=ts, epochs=epochs)
            joint_runs.append({'ts': ts, 'val_full': vf, 'val_tail': vt, 'ep': ep,
                               'actor': act, 'belief': bel})
            torch.save({'belief': bel.state_dict(), 'actor': act.state_dict(),
                        'val_full': vf, 'val_tail': vt, 'ep': ep, 'channel': channel, 'joint': True},
                       f'{CK}/r3_lbelief_{channel}_joint_s{ts}.pt')
        joint_best = max(joint_runs, key=lambda r: r['val_full'])
        print(f'[{channel}] JOINT val-winner: s{joint_best["ts"]} (val_full {joint_best["val_full"]:.4f})',
              flush=True)

    # ---- 5-seed TEST eval: learned-belief arms + anchors on THIS run's cohorts ----
    arms = {'lbelief_frozen': {'f': [], 't': [], 'pu': {}},
            'anchor_additive_actor' if not noisy else 'anchor_noise_actor': {'f': [], 't': [], 'pu': {}}}
    if joint_best is not None:
        arms['lbelief_joint'] = {'f': [], 't': [], 'pu': {}}
    if noisy:
        arms['anchor_static_repeat'] = {'f': [], 't': [], 'pu': {}}
        # noise actors + static-repeat schedule
        nacts = R2.load_noisy_actors(d)
        W64 = Wf.astype(np.float64); b64 = bf.astype(np.float64)
        r2json = json.load(open(f'{CK}/squeeze_r2_noise.json'))
        poolQ = P.decoder_svd_dirs(model, R2.POOL)
        sched_r = np.stack([poolQ[j] for j in r2json['greedy']['repeat']['schedule_idx']]).astype(np.float32)

    def accum(arm, f, t, pu):
        arms[arm]['f'].append(f); arms[arm]['t'].append(t)
        for x, v in pu.items():
            arms[arm]['pu'].setdefault(x, []).append(v)

    for sd in SEEDS:
        ar = P.arena_seed(sd, rd_all); test = ar['test_users']
        zst = P.build_zstar(model, ar, test)
        if noisy:
            um = C4.user_means(ar, test, rd_all)
            f, t, pu = eval_belief_noisy(fbest['actor'], fbest['belief'], Wf, bf, ar, test, zst, um, d, sd)
            accum('lbelief_frozen', f, t, pu)
            if joint_best is not None:
                f, t, pu = eval_belief_noisy(joint_best['actor'], joint_best['belief'], Wf, bf, ar, test, zst, um, d, sd)
                accum('lbelief_joint', f, t, pu)
            # anchor: noise-actor seed-avg (s0/s1/s2) per-user
            puA = {}; afs = []; ats = []
            for k in ['s0', 's1', 's2']:
                fa, ta, pua = R2.eval_actor_noisy(nacts[k][0], model, Wf, bf, ar, test, zst, um, sd)
                afs.append(fa); ats.append(ta)
                for x, v in pua.items(): puA.setdefault(x, []).append(v)
            accum('anchor_noise_actor', float(np.mean(afs)), float(np.mean(ats)),
                  {x: float(np.mean(vs)) for x, vs in puA.items()})
            # anchor: static-repeat
            rng_sr = np.random.default_rng(sd)
            fs, ts, _, pus = R2.run_static_noisy_peruser(model, W64, b64, ar, test, zst, sched_r, rng_sr, um)
            accum('anchor_static_repeat', fs, ts, pus)
        else:
            f, t, pu = eval_belief_clean(fbest['actor'], fbest['belief'], Wf, bf, ar, test, zst, d)
            accum('lbelief_frozen', f, t, pu)
            if joint_best is not None:
                f, t, pu = eval_belief_clean(joint_best['actor'], joint_best['belief'], Wf, bf, ar, test, zst, d)
                accum('lbelief_joint', f, t, pu)
            # anchor: additive.actor (p4a_actor_s0) native
            aact = P.Actor(d); aact.load_state_dict(torch.load(f'{CK}/p4a_actor_s0.pt', map_location=DEVICE)['state']); aact.eval()
            fa, ta, pua = additive_actor_clean_peruser(aact, Wf, bf, ar, test, zst, d)
            accum('anchor_additive_actor', fa, ta, pua)
        print(f'[{channel} TEST] seed{sd} done [{time.time()-t0:.0f}s]', flush=True)

    def agg(a):
        return {'full': float(np.mean(a['f'])), 'tail': float(np.mean(a['t'])),
                'full_sd': float(np.std(a['f'])), 'tail_sd': float(np.std(a['t'])),
                'per_seed_full': [float(x) for x in a['f']]}
    test_res = {k: agg(v) for k, v in arms.items()}

    # ---- paired bootstrap: learned-belief arms vs anchors (seed-mean per-user full) ----
    def seedavg(pu): return {x: float(np.mean(vs)) for x, vs in pu.items()}
    boot = {}
    anchor_keys = [k for k in arms if k.startswith('anchor_')]
    for lb in [k for k in arms if k.startswith('lbelief_')]:
        for anc in anchor_keys:
            boot[f'{lb}__vs__{anc}'] = R1.paired_boot(seedavg(arms[lb]['pu']), seedavg(arms[anc]['pu']))

    out = {'channel': channel, 'epochs': epochs, 'fseeds': fseeds, 'jseeds': jseeds,
           'frozen_val': [{'ts': r['ts'], 'val_full': r['val_full'], 'val_tail': r['val_tail'],
                           'ep': r['ep']} for r in frozen_runs],
           'frozen_valwin_ts': fbest['ts'],
           'joint_val': ([{'ts': joint_best['ts'], 'val_full': joint_best['val_full'],
                           'val_tail': joint_best['val_tail'], 'ep': joint_best['ep']}]
                         if joint_best is not None else []),
           'test': test_res, 'anchors_ref': anchors, 'bootstrap': boot}
    _save(channel, out)
    _print_summary(channel, out)
    return out


def _print_summary(channel, out):
    print(f'\n=========== SUMMARY [{channel}] ===========', flush=True)
    for k, v in out['test'].items():
        print(f'{k:26s} {v["full"]:.4f} (sd {v["full_sd"]:.4f}) / tail {v["tail"]:.4f}', flush=True)
    print('--- paired bootstrap (full NDCG) ---', flush=True)
    for k, v in out['bootstrap'].items():
        print(f'{k:44s} d={v["mean_diff"]:+.4f} CI[{v["ci95"][0]:+.4f},{v["ci95"][1]:+.4f}] '
              f'p(>0)={v["p_gt0"]:.3f}', flush=True)


def _save(stage, obj):
    all_ = json.load(open(OUT)) if os.path.exists(OUT) else {}
    all_[stage] = obj
    json.dump(all_, open(OUT, 'w'), indent=2)
    print(f'[saved] {OUT} :: {stage}', flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('mode', choices=['clean', 'noisy', 'both'])
    ap.add_argument('--epochs', type=int, default=16)
    ap.add_argument('--fseeds', type=str, default='0,1')
    ap.add_argument('--jseeds', type=str, default='0')
    args = ap.parse_args()
    fseeds = [int(s) for s in args.fseeds.split(',') if s != '']
    jseeds = [int(s) for s in args.jseeds.split(',') if s != '']
    if args.mode in ('clean', 'both'):
        run('clean', args.epochs, fseeds, jseeds)
    if args.mode in ('noisy', 'both'):
        run('noisy', args.epochs, fseeds, jseeds)


if __name__ == '__main__':
    main()
