"""
p3_finetune_dislike.py -- INSTRUMENT 2.0 Phase-3 W2 dislike DESIGN B: two-channel encoder.
Extend the encoder input with a DISLIKE bag: h1 = LN(swish(fc1(likes_l2) + fc1_dis(dis_l2) + ...)).
fc1_dis is ZERO-INITIALIZED so with an empty dislike channel the model is BYTE-IDENTICAL to the
frozen P2 RecVAE (item-preservation is exact by construction). We then briefly fine-tune ONLY
fc1_dis (decoder + all other encoder weights frozen) so the dislike channel demotes disliked items.

Training (ML-1M): likes = denoised subset of user's rating>=4 items; dislikes = user's rating<=2
items; loss = -multinomial_LL(likes) + lam * prob_mass_on_dislikes. A few hundred steps.
Durable ckpt -> .cache/instrument2/ml1m_dis2ch_d512.pt (never overwrites P2). Reports:
  - PRESERVATION: full-profile NDCG@10 with empty dislike channel vs frozen (must be within 1%).
  - DISLIKE SPECIFICITY: feed a tag's member items as the dislike bag on a base like-profile;
    member-item percentile-rank demotion vs a non-member control (match/beat 1.6: -15.4 / -0.9).

Usage: python scripts/instrument2/p3_finetune_dislike.py [--steps 600 --lam 5.0]
"""
import os, sys, json, argparse, time
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from recvae import RecVAE, swish
import ml1m_arena as A

DEVICE = torch.device('cpu'); torch.set_num_threads(4)
CK = '.cache/instrument2'
OUT_CKPT = f'{CK}/ml1m_dis2ch_d512.pt'


def l2n(x):
    n = x.pow(2).sum(-1, keepdim=True).sqrt()
    return x / torch.clamp(n, min=1e-8)


class DisEncoder(torch.nn.Module):
    """Wraps a frozen RecVAE encoder, adds a zero-init dislike channel into the fc1 pre-activation."""
    def __init__(self, enc, input_dim):
        super().__init__()
        self.enc = enc                                   # frozen reference encoder
        # bias=False: empty dislike bag -> l2n(0)=0 -> fc1_dis(0)=0 EXACTLY (preservation is exact)
        self.fc1_dis = torch.nn.Linear(input_dim, enc.fc1.out_features, bias=False)
        torch.nn.init.zeros_(self.fc1_dis.weight)

    def forward(self, likes, dis, dropout_rate=0.0):
        e = self.enc
        xl = l2n(likes)
        xl = F.dropout(xl, p=dropout_rate, training=self.training)
        xd = l2n(dis)
        h1 = e.ln1(swish(e.fc1(xl) + self.fc1_dis(xd)))
        h2 = e.ln2(swish(e.fc2(h1) + h1))
        h3 = e.ln3(swish(e.fc3(h2) + h1 + h2))
        h4 = e.ln4(swish(e.fc4(h3) + h1 + h2 + h3))
        h5 = e.ln5(swish(e.fc5(h4) + h1 + h2 + h3 + h4))
        return e.fc_mu(h5), e.fc_logvar(h5)


def decode(model, Z):
    with torch.no_grad():
        return model.decoder(torch.tensor(np.asarray(Z, np.float32))).numpy().astype(np.float64)


def onehot(items, ni):
    x = np.zeros((1, ni), np.float32)
    if len(items): x[0, list(items)] = 1.0
    return x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--steps', type=int, default=600)
    ap.add_argument('--lam', type=float, default=5.0)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--batch', type=int, default=256)
    args = ap.parse_args()
    ni = 3706
    blob = torch.load(f'{CK}/ml1m_recvae_d512_best.pt', map_location=DEVICE)
    ma = blob['args']; model = RecVAE(ma['hidden'], ma['latent'], ni); model.load_state_dict(blob['model']); model.eval()
    for p in model.parameters(): p.requires_grad_(False)
    dis_enc = DisEncoder(model.encoder, ni)
    opt = torch.optim.Adam(dis_enc.fc1_dis.parameters(), lr=args.lr)

    ar = A.load_arena(seed=123); rd_all = {x: dict(v) for x, v in ar['rat_by_u'].items()}
    trU = [x for x in ar['trU'] if sum(1 for j, r in rd_all[x].items() if r >= 4) >= 4]
    likes_by = {x: [j for j, r in rd_all[x].items() if r >= 4] for x in trU}
    dis_by = {x: [j for j, r in rd_all[x].items() if r <= 2] for x in trU}
    rng = np.random.default_rng(0)

    print(f'[disB] fine-tuning fc1_dis only ({args.steps} steps, lam={args.lam})...', flush=True)
    t0 = time.time()
    dis_enc.train()
    for step in range(args.steps):
        idx = rng.integers(0, len(trU), size=args.batch)
        L = np.zeros((args.batch, ni), np.float32); D = np.zeros((args.batch, ni), np.float32)
        for b in range(args.batch):
            x = trU[idx[b]]; lk = likes_by[x]; dl = dis_by[x]
            L[b, lk] = 1.0
            if dl: D[b, dl] = 1.0
        Lt = torch.tensor(L); Dt = torch.tensor(D)
        mu, logvar = dis_enc(Lt, Dt, dropout_rate=0.5)
        std = torch.exp(0.5 * logvar); z = mu + torch.randn_like(std) * std
        pred = model.decoder(z)
        logp = F.log_softmax(pred, dim=1)
        # per-item-normalized recon so the dislike penalty is on a comparable scale
        mll = ((logp * Lt).sum(1) / Lt.sum(1).clamp(min=1)).mean()
        dislike_mass = (F.softmax(pred, dim=1) * Dt).sum(1).mean()
        loss = -mll + args.lam * dislike_mass
        opt.zero_grad(); loss.backward(); opt.step()
        if (step + 1) % 150 == 0:
            print(f'  step {step+1}/{args.steps} mll {mll.item():.3f} dis_mass {dislike_mass.item():.4f} '
                  f'[{time.time()-t0:.0f}s]', flush=True)
    dis_enc.eval()
    torch.save({'fc1_dis': dis_enc.fc1_dis.state_dict(), 'args': vars(args), 'base': 'ml1m_recvae_d512'}, OUT_CKPT)
    print(f'[disB] saved -> {OUT_CKPT}', flush=True)

    # ---- PRESERVATION: empty dislike channel must match frozen exactly ----
    test = ar['test_users']
    def ndcg_full(zfun):
        acc = 0.0; m = 0
        for x in test:
            profset, tst = ar['SPL'][x]; rd = rd_all[x]
            tlike = set(j for j in tst if rd[j] >= 4)
            if not tlike: continue
            lk = [j for j in profset if rd[j] >= 4]
            if not lk: continue
            z = zfun(lk)
            s = decode(model, z[None, :])[0]
            nf = A.ndcg_at10(s, tlike, profset, ar['headmask'], False)
            if nf is not None: acc += nf; m += 1
        return acc / max(m, 1)
    def z_frozen(lk):
        with torch.no_grad():
            mu, _ = model.encoder(torch.tensor(onehot(lk, ni)), dropout_rate=0.0)
        return mu.numpy()[0]
    def z_dis_empty(lk):
        with torch.no_grad():
            mu, _ = dis_enc(torch.tensor(onehot(lk, ni)), torch.zeros(1, ni), dropout_rate=0.0)
        return mu.numpy()[0]
    frozen = ndcg_full(z_frozen); disempty = ndcg_full(z_dis_empty)
    preservation = {'frozen_full': frozen, 'dis2ch_empty_full': disempty,
                    'rel_change_pct': float(100 * (disempty - frozen) / frozen),
                    'pass_within_1pct': abs(disempty - frozen) / frozen < 0.01}
    print(f'[disB] preservation: frozen {frozen:.4f} vs dis-empty {disempty:.4f} '
          f'({preservation["rel_change_pct"]:+.3f}%)', flush=True)

    # ---- DISLIKE SPECIFICITY via the two-channel: tag members in the dislike bag ----
    C = np.load(f'{CK}/concepts_ml1m.npz', allow_pickle=True)
    item_tag = C['item_tag']; names = [str(x) for x in C['tag_names']]
    want = ['horror', 'violence', 'romance', 'comedy', 'dark', 'scary', 'gory', 'action']
    tcs = [names.index(w) for w in want if w in names][:5]
    rng2 = np.random.default_rng(11)
    def pctrank(s):
        return np.argsort(np.argsort(s)) / (len(s) - 1)
    per = {}
    for tc in tcs:
        rel = item_tag[:, tc]; members = np.argpartition(-rel, 50)[:50]
        base_scores = decode(model, np.zeros((1, model.decoder.in_features), np.float32))[0]
        control = np.array([c for c in np.argsort(-base_scores) if c not in set(members.tolist())])[:50]
        dm = []; dc = []
        for x in test[:150]:
            lk = [j for j in ar['SPL'][x][0] if rd_all[x][j] >= 4]
            if len(lk) < 2: continue
            it2 = list(rng2.choice(lk, 2, replace=False))
            with torch.no_grad():
                mu0, _ = dis_enc(torch.tensor(onehot(it2, ni)), torch.zeros(1, ni), dropout_rate=0.0)
                disbag = np.zeros((1, ni), np.float32); disbag[0, members] = 1.0
                mu1, _ = dis_enc(torch.tensor(onehot(it2, ni)), torch.tensor(disbag), dropout_rate=0.0)
            r0 = pctrank(decode(model, mu0.numpy())[0]); r1 = pctrank(decode(model, mu1.numpy())[0])
            dm.append((r1[members] - r0[members]).mean()); dc.append((r1[control] - r0[control]).mean())
        per[names[tc]] = {'member_demotion_pts': float(np.mean(dm) * 100),
                          'control_demotion_pts': float(np.mean(dc) * 100)}
        print(f'  [disB spec] {names[tc]:>10}: member {np.mean(dm)*100:+.2f}pts control {np.mean(dc)*100:+.2f}pts', flush=True)
    md = float(np.mean([v['member_demotion_pts'] for v in per.values()]))
    cd = float(np.mean([v['control_demotion_pts'] for v in per.values()]))
    out = {'preservation': preservation, 'dislike_B_specificity': {
        'per_tag': per, 'mean_member_demotion_pts': md, 'mean_control_demotion_pts': cd,
        'specificity_pts': md - cd}}
    json.dump(out, open(f'{CK}/p3_dis2ch_ml1m.json', 'w'), indent=2, default=float)
    print(json.dumps(out, indent=2, default=float))


if __name__ == '__main__':
    main()
