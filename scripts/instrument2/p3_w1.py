"""
p3_w1.py -- INSTRUMENT 2.0 Phase-3 W1: THE LATENT BELIEF-UPDATE OPERATOR.
Elicitation in RecVAE latent space via free/continuous geometric queries.

Items fold natively (add to input bag -> re-encode). Free queries q (unit directions in z)
with a graded geometric answer a = cos(z*, q), z* = enc(full-profile likes), need a latent
update rule for (q, a). We DESIGN + BENCHMARK three operators:

  (a) ADDITIVE      z' = z + eta * a * q            (eta val-selected on a grid)
  (b) PSEUDO-DECODE q -> decode(q) top-M items (softmax-weighted by decoder score), scaled by
      the answer, ACCUMULATED into the input bag -> re-encode (keeps everything in the proven
      encoder path). Signed: a>0 folds top-M of decode(q); a<0 folds top-M of decode(-q) with
      weight |a| (a nonneg multinomial bag cannot subtract; disliking q ~ liking -q).
  (c) LEARNED HEAD  MLP([z, q, a]) -> Delta z, trained on train users; supervised target =
      enc(profile ∪ consistent evidence) - enc(profile), realized here as the residual toward
      the full-profile encode (z* - z), with a val-selected output gain.

Benchmark on ML-1M: cold start z=0; 8 graded probes with RANDOM directions and INFORMATIVE
directions (top-8 right-singular vectors of the decoder weight = the item-relevant orthonormal
design); NDCG@10 recovery vs the k=8 item-fold reference at equal turn count. Seed-avg over
{1,2,3}. Writes .cache/instrument2/p3_w1.json.

Usage: python scripts/instrument2/p3_w1.py [--seeds 1,2,3] [--train_head]
"""
import os, sys, json, time, argparse
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ml1m_arena as A
from recvae import RecVAE

DEVICE = torch.device('cpu')
torch.set_num_threads(4)
CKPT = os.path.join('.cache', 'instrument2', 'ml1m_recvae_d512_best.pt')
HEAD_CKPT = os.path.join('.cache', 'instrument2', 'p3_w1_head_ml1m.pt')
OUT = os.path.join('.cache', 'instrument2', 'p3_w1.json')


def load_model():
    blob = torch.load(CKPT, map_location=DEVICE)
    a = blob['args']
    m = RecVAE(a['hidden'], a['latent'], 3706).to(DEVICE)
    m.load_state_dict(blob['model']); m.eval()
    return m, a['latent']


def enc_mu(model, Xdense):
    with torch.no_grad():
        mu, _ = model.encoder(torch.tensor(Xdense, dtype=torch.float32), dropout_rate=0.0)
    return mu.numpy()


def decode(model, Z):
    with torch.no_grad():
        return model.decoder(torch.tensor(Z, dtype=torch.float32)).numpy().astype(np.float64)


def bag_from_likes(likes, ni):
    x = np.zeros(ni, np.float32); x[list(likes)] = 1.0; return x


# --- informative design: top-K right-singular vectors of decoder weight W (ni x d) ---
def informative_dirs(model, K):
    W = model.decoder.weight.detach().numpy()          # (ni x d)
    # right singular vectors V (d x d); take leading K, already orthonormal, unit
    _, _, Vt = np.linalg.svd(W - W.mean(0, keepdims=True), full_matrices=False)
    return Vt[:K].astype(np.float32)                    # (K x d) orthonormal unit rows


def rand_dirs(K, d, rng):
    Q = rng.standard_normal((K, d)).astype(np.float32)
    return Q / (np.linalg.norm(Q, axis=1, keepdims=True) + 1e-9)


class Head(torch.nn.Module):
    def __init__(self, d, h=512):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(2 * d + 1, h), torch.nn.SiLU(),
            torch.nn.Linear(h, h), torch.nn.SiLU(),
            torch.nn.Linear(h, d))

    def forward(self, z, q, a):
        return self.net(torch.cat([z, q, a], dim=-1))


# ---------------- operators: each maps a per-user (z, {(q,a)} stream) to final z --------------
def op_additive(model, z0, qs, ans, eta):
    z = z0.copy()
    for t in range(len(qs)):
        z = z + eta * ans[t] * qs[t]
    return z


def op_pseudodecode(model, ni, z0_likes, qs, ans, M, base_bag):
    bag = base_bag.copy()
    for t in range(len(qs)):
        a = ans[t]
        qd = qs[t] if a >= 0 else -qs[t]
        s = decode(model, qd[None, :])[0]                 # (ni,) item scores of the direction
        top = np.argpartition(-s, M)[:M]
        w = np.exp(s[top] - s[top].max()); w = w / (w.sum() + 1e-9)
        bag[top] += abs(a) * w.astype(np.float32)
    return enc_mu(model, bag[None, :])[0]


def head_apply(head, z0, qs, ans, gain):
    z = torch.tensor(z0[None, :], dtype=torch.float32)
    for t in range(len(qs)):
        q = torch.tensor(qs[t][None, :], dtype=torch.float32)
        a = torch.tensor([[float(ans[t])]], dtype=torch.float32)
        with torch.no_grad():
            z = z + gain * head(z, q, a)
    return z.numpy()[0]


# ---------------- eval a single operator over a cohort at a fixed seed ------------------------
def eval_operator(model, ar, users, zstar, make_z, tail=False):
    accf = acct = 0.0; mf = mt = 0
    for i, x in enumerate(users):
        profset, test = ar['SPL'][x]; rd = ar['rat_by_u_dict'][x]
        tlike = set(j for j in test if rd[j] >= 4)
        if not tlike:
            continue
        z = make_z(i, x, zstar[i])
        s = decode(model, z[None, :])[0]
        nf = A.ndcg_at10(s, tlike, profset, ar['headmask'], False)
        nt = A.ndcg_at10(s, tlike, profset, ar['headmask'], True)
        if nf is not None: accf += nf; mf += 1
        if nt is not None: acct += nt; mt += 1
    return accf / max(mf, 1), acct / max(mt, 1)


def build_zstar_and_bags(model, ar, users, ni):
    """z* = enc(full profile likes); base_bag = zeros (cold)."""
    zstar = np.zeros((len(users), model.decoder.in_features), np.float32)
    for i, x in enumerate(users):
        profset, _ = ar['SPL'][x]; rd = ar['rat_by_u_dict'][x]
        likes = [j for j in profset if rd[j] >= 4]
        if likes:
            zstar[i] = enc_mu(model, bag_from_likes(likes, ni)[None, :])[0]
    return zstar


def train_head(model, ar, ni, d, seeds_train=(0, 1, 2)):
    """Train the learned update head on TRAIN users. One probe per sample."""
    print('[w1] training learned head on train users...', flush=True)
    trU = ar['trU']; rdall = ar['rat_by_u_dict']
    # gather full-profile z* per train user (use ALL likes)
    likes_by = {x: [j for j, r in rdall[x].items() if r >= 4] for x in trU}
    trU = [x for x in trU if len(likes_by[x]) >= 4]
    info = informative_dirs(model, 32)
    head = Head(d); opt = torch.optim.Adam(head.parameters(), lr=1e-3)
    rng = np.random.default_rng(0)
    B = 256; STEPS = 4000
    # precompute z* for all train users in chunks
    Zstar = np.zeros((len(trU), d), np.float32)
    for st in range(0, len(trU), 500):
        chunk = trU[st:st + 500]
        Xd = np.stack([bag_from_likes(likes_by[x], ni) for x in chunk])
        Zstar[st:st + len(chunk)] = enc_mu(model, Xd)
    t0 = time.time()
    for step in range(STEPS):
        idx = rng.integers(0, len(trU), size=B)
        # current z: random subset of each user's likes (size 0..8)
        Zcur = np.zeros((B, d), np.float32); Q = np.zeros((B, d), np.float32); Ans = np.zeros((B, 1), np.float32)
        Xd = np.zeros((B, ni), np.float32)
        for b in range(B):
            x = trU[idx[b]]; lk = likes_by[x]
            s = int(rng.integers(0, min(9, len(lk) + 1)))
            if s > 0:
                sub = rng.choice(lk, size=min(s, len(lk)), replace=False)
                Xd[b, sub] = 1.0
        Zcur = enc_mu(model, Xd)
        empty = (Xd.sum(1) == 0)
        Zcur[empty] = 0.0                       # empty input -> z=0 (encoder NaNs on empty)
        for b in range(B):
            if rng.random() < 0.5:
                q = info[rng.integers(0, len(info))]
            else:
                q = rng.standard_normal(d).astype(np.float32); q /= np.linalg.norm(q) + 1e-9
            zs = Zstar[idx[b]]
            a = float(zs @ q / (np.linalg.norm(zs) + 1e-9))
            Q[b] = q; Ans[b, 0] = a
        tgt = torch.tensor(Zstar[idx] - Zcur, dtype=torch.float32)
        pred = head(torch.tensor(Zcur), torch.tensor(Q), torch.tensor(Ans))
        loss = ((pred - tgt) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        if (step + 1) % 1000 == 0:
            print(f'  head step {step+1}/{STEPS} mse {loss.item():.4f} [{time.time()-t0:.0f}s]', flush=True)
    torch.save({'head': head.state_dict(), 'd': d}, HEAD_CKPT)
    print(f'[w1] saved head -> {HEAD_CKPT}', flush=True)
    return head


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', default='1,2,3')
    ap.add_argument('--train_head', action='store_true')
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(',')]
    model, d = load_model(); ni = 3706
    info8 = informative_dirs(model, 8)

    # train (or load) the learned head once (uses seed-123 arena train split)
    ar0 = A.load_arena(seed=123); ar0['rat_by_u_dict'] = {x: dict(v) for x, v in ar0['rat_by_u'].items()}
    if args.train_head or not os.path.exists(HEAD_CKPT):
        head = train_head(model, ar0, ni, d)
    else:
        head = Head(d); head.load_state_dict(torch.load(HEAD_CKPT, map_location=DEVICE)['head']); head.eval()

    eta_grid = [4, 8, 12, 16, 24, 32]
    gain_grid = [0.25, 0.5, 0.75, 1.0]
    Mpd = 50
    results = {'random': {}, 'informative': {}}
    # accumulate over seeds
    agg = {mode: {'item8': [], 'add': [], 'pd': [], 'head': [],
                  'add_tail': [], 'pd_tail': [], 'head_tail': [], 'item8_tail': []}
           for mode in ('random', 'informative')}
    zstar_norms = []
    eta_sel = {}; gain_sel = {}
    for sd in seeds:
        ar = A.load_arena(seed=sd); ar['rat_by_u_dict'] = ar0['rat_by_u_dict']
        val = ar['val_users']; test = ar['test_users']
        zstar_v = build_zstar_and_bags(model, ar, val, ni)
        zstar_t = build_zstar_and_bags(model, ar, test, ni)
        zstar_norms.append(float(np.linalg.norm(zstar_t, axis=1).mean()))
        rng = np.random.default_rng(1000 + sd)
        # per (mode): build the 8 directions + answers per user, then eval operators
        for mode in ('random', 'informative'):
            def dirs_for(users, zstar):
                # returns list per-user of (Q(8xd), ans(8,))
                out = []
                for i, x in enumerate(users):
                    if mode == 'informative':
                        Q = info8
                    else:
                        Q = rand_dirs(8, d, rng)
                    zs = zstar[i]; nz = np.linalg.norm(zs) + 1e-9
                    a = Q @ zs / nz
                    out.append((Q, a.astype(np.float32)))
                return out
            DV = dirs_for(val, zstar_v); DT = dirs_for(test, zstar_t)

            # item-8 reference (fold 8 random profile likes)
            def mk_item8(seed):
                r = np.random.default_rng(seed)
                def f(i, x, zs):
                    profset, _ = ar['SPL'][x]; rd = ar['rat_by_u_dict'][x]
                    lk = [j for j in profset if rd[j] >= 4]
                    if len(lk) > 8: lk = list(r.choice(lk, 8, replace=False))
                    if not lk: return np.zeros(d, np.float32)
                    return enc_mu(model, bag_from_likes(lk, ni)[None, :])[0]
                return f
            i8f, i8t = eval_operator(model, ar, test, zstar_t, mk_item8(sd))
            agg[mode]['item8'].append(i8f); agg[mode]['item8_tail'].append(i8t)

            # (a) additive: val-select eta
            best_eta, best_v = eta_grid[0], -1
            for eta in eta_grid:
                def f(i, x, zs, D=DV, eta=eta): return op_additive(model, np.zeros(d, np.float32), D[i][0], D[i][1], eta)
                vf, _ = eval_operator(model, ar, val, zstar_v, f)
                if vf > best_v: best_v, best_eta = vf, eta
            def fa(i, x, zs, D=DT, eta=best_eta): return op_additive(model, np.zeros(d, np.float32), D[i][0], D[i][1], eta)
            af, at = eval_operator(model, ar, test, zstar_t, fa)
            agg[mode]['add'].append(af); agg[mode]['add_tail'].append(at); eta_sel[(mode, sd)] = best_eta

            # (b) pseudo-decode (no hyperparam beyond M): accumulate into cold bag
            def fp(i, x, zs, D=DT): return op_pseudodecode(model, ni, None, D[i][0], D[i][1], Mpd, np.zeros(ni, np.float32))
            pf, pt = eval_operator(model, ar, test, zstar_t, fp)
            agg[mode]['pd'].append(pf); agg[mode]['pd_tail'].append(pt)

            # (c) learned head: val-select gain
            best_g, best_gv = gain_grid[0], -1
            for g in gain_grid:
                def f(i, x, zs, D=DV, g=g): return head_apply(head, np.zeros(d, np.float32), D[i][0], D[i][1], g)
                vf, _ = eval_operator(model, ar, val, zstar_v, f)
                if vf > best_gv: best_gv, best_g = vf, g
            def fh(i, x, zs, D=DT, g=best_g): return head_apply(head, np.zeros(d, np.float32), D[i][0], D[i][1], g)
            hf, ht = eval_operator(model, ar, test, zstar_t, fh)
            agg[mode]['head'].append(hf); agg[mode]['head_tail'].append(ht); gain_sel[(mode, sd)] = best_g
            print(f'[w1] seed{sd} {mode}: item8 {i8f:.4f} | add(eta{best_eta}) {af:.4f} | '
                  f'pd {pf:.4f} | head(g{best_g}) {hf:.4f}', flush=True)

    out = {'zstar_norm_mean': float(np.mean(zstar_norms)), 'seeds': seeds,
           'reference_item8_seedavg_full_P2': 0.4625}
    for mode in ('random', 'informative'):
        out[mode] = {k: {'full': float(np.mean(agg[mode][k])),
                         'tail': float(np.mean(agg[mode][k + '_tail']))}
                     for k in ('item8', 'add', 'pd', 'head')}
        out[mode]['eta_sel'] = [eta_sel[(mode, s)] for s in seeds]
        out[mode]['gain_sel'] = [gain_sel[(mode, s)] for s in seeds]
    json.dump(out, open(OUT, 'w'), indent=2)
    print('\n=== W1 SUMMARY (seed-avg) ===')
    print(f"mean ||z*|| = {out['zstar_norm_mean']:.2f}")
    for mode in ('random', 'informative'):
        m = out[mode]
        print(f"[{mode:>11}] item8 {m['item8']['full']:.4f} | additive {m['add']['full']:.4f} | "
              f"pseudo-dec {m['pd']['full']:.4f} | head {m['head']['full']:.4f}")
    print(f'[saved] {OUT}')


if __name__ == '__main__':
    main()
