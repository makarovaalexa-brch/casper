"""
ml1m_recvae.py -- port RecVAE (Phase-1 replicated recipe) to the CANONICAL CASPER ML-1M
arena. Train at latent d, early-stop on the val (te[:300]) full-profile fold NDCG@10 full,
then eval on the test (te[300:], 304 users): full-profile NDCG@10 full+tail + k in {0,1,2,4,8}
fold curve.

RecVAE mapping to OUR protocol (stated):
  - input vocabulary = ni=3706 items (identical V1 ordinal ids).
  - train matrix = trU users x ni, binarized implicit positives (rating>=4).
  - belief update = the amortized encoder pass ONLY (Phase-1.6 rule); k=0 -> z=prior-mean(0)
    (Phase-1.5 rule; never feed the encoder an empty vector).
  - "full-profile fold" folds the profile-half LIKES (rating>=4) of the split -- the VAE
    lineage is implicit-positive, so dislikes in the profile half are dropped (Phase-3 handles
    polarity). Candidate-exclusion still removes ALL profile items (likes+dislikes), matching V1.
  - score = decoder logits over ni; exclude profset; NDCG@10 (full + Cremonesi tail).

Usage: python scripts/instrument2/ml1m_recvae.py --latent 128 [--epochs 200] [--resume] [--eval_only]
"""
import os, sys, json, time, argparse
import numpy as np
import torch
from scipy import sparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ml1m_arena as A
from recvae import RecVAE, run_updates, make_predict_fn

CKPT_DIR = os.path.join('.cache', 'instrument2')
os.makedirs(CKPT_DIR, exist_ok=True)


def build_train_matrix(ar):
    trU, ni, likes = ar['trU'], ar['ni'], ar['likes_by_u']
    rows, cols = [], []
    for r, x in enumerate(trU):
        for j in likes.get(x, []):
            rows.append(r); cols.append(j)
    X = sparse.csr_matrix((np.ones(len(rows), np.float32), (rows, cols)),
                          dtype=np.float32, shape=(len(trU), ni))
    return X


def fold_score(model, ar, x, profset, klimit=None, rng=None):
    """Fold profile-half likes (optionally a random k-subset) -> decoder logits (ni,)."""
    likes = [j for j in profset if ar['rat_by_u_dict'][x].get(j, 0) >= 4]
    if klimit is not None:
        if klimit == 0:
            likes = []
        elif len(likes) > klimit:
            likes = list(rng.choice(likes, size=klimit, replace=False))
    xv = np.zeros((1, ar['ni']), np.float32)
    for j in likes:
        xv[0, j] = 1.0
    with torch.no_grad():
        if xv.sum() == 0:  # z = prior mean (0): decode(0) = decoder bias
            z = torch.zeros(1, model.decoder.in_features)
            s = model.decoder(z).numpy()[0]
        else:
            s = model(torch.tensor(xv), calculate_loss=False).numpy()[0]
    return s.astype(np.float64)


def eval_fullprofile(model, ar, users, klimit=None, kseed=0):
    rng = np.random.default_rng(kseed)
    accf = acct = 0.0; mf = mt = 0
    for x in users:
        profset, test = ar['SPL'][x]; rd = ar['rat_by_u_dict'][x]
        tlike = set(j for j in test if rd[j] >= 4)
        if not tlike:
            continue
        s = fold_score(model, ar, x, profset, klimit=klimit, rng=rng)
        nf = A.ndcg_at10(s, tlike, profset, ar['headmask'], False)
        if nf is not None:
            accf += nf; mf += 1
        nt = A.ndcg_at10(s, tlike, profset, ar['headmask'], True)
        if nt is not None:
            acct += nt; mt += 1
    return (accf / max(mf, 1), acct / max(mt, 1), mf, mt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--latent', type=int, default=128)
    ap.add_argument('--hidden', type=int, default=600)
    ap.add_argument('--epochs', type=int, default=200)
    ap.add_argument('--batch', type=int, default=500)
    ap.add_argument('--lr', type=float, default=5e-4)
    ap.add_argument('--gamma', type=float, default=0.005)
    ap.add_argument('--n_enc', type=int, default=3)
    ap.add_argument('--n_dec', type=int, default=1)
    ap.add_argument('--dropout', type=float, default=0.5)
    ap.add_argument('--patience', type=int, default=15)
    ap.add_argument('--seed', type=int, default=123)
    ap.add_argument('--eval_only', action='store_true')
    args = ap.parse_args()

    torch.manual_seed(98765); np.random.seed(98765)
    device = torch.device('cpu')
    ar = A.load_arena(seed=args.seed)
    ar['rat_by_u_dict'] = {x: dict(v) for x, v in ar['rat_by_u'].items()}
    ni = ar['ni']
    tag = f'ml1m_recvae_d{args.latent}'
    ckpt = os.path.join(CKPT_DIR, f'{tag}.pt'); best_ckpt = os.path.join(CKPT_DIR, f'{tag}_best.pt')
    log_path = os.path.join(CKPT_DIR, f'{tag}_log.json')
    print(f'[ml1m-recvae] d={args.latent} ni={ni} trU={len(ar["trU"])} '
          f'val={len(ar["val_users"])} test={len(ar["test_users"])}', flush=True)

    model = RecVAE(args.hidden, args.latent, ni).to(device)
    opt_enc = torch.optim.Adam(model.encoder.parameters(), lr=args.lr)
    opt_dec = torch.optim.Adam(model.decoder.parameters(), lr=args.lr)

    if args.eval_only:
        model.load_state_dict(torch.load(best_ckpt, map_location=device)['model']); model.eval()
    else:
        X = build_train_matrix(ar)
        print(f'[ml1m-recvae] train X: {X.shape} nnz={X.nnz}', flush=True)
        idxlist = np.arange(X.shape[0]); predict = make_predict_fn(model, device)
        best = -1.0; best_ep = 0; hist = []; t0 = time.time()
        for ep in range(args.epochs):
            run_updates(model, [opt_enc], X, idxlist, args.batch, device,
                        n_epochs=args.n_enc, dropout_rate=args.dropout, gamma=args.gamma, beta=None)
            model.update_prior()
            run_updates(model, [opt_dec], X, idxlist, args.batch, device,
                        n_epochs=args.n_dec, dropout_rate=0.0, gamma=args.gamma, beta=None)
            vf, vt, _, _ = eval_fullprofile(model, ar, ar['val_users'])
            hist.append({'epoch': ep + 1, 'val_ndcg10_full': vf, 'val_ndcg10_tail': vt})
            mk = ''
            if vf > best:
                best = vf; best_ep = ep + 1
                torch.save({'model': model.state_dict(), 'args': vars(args),
                            'best_ep': best_ep, 'val_full': vf, 'val_tail': vt}, best_ckpt)
                mk = ' <- BEST'
            if (ep + 1) % 5 == 0 or mk:
                print(f'  ep{ep+1:3d} val full {vf:.4f} tail {vt:.4f}{mk} '
                      f'[best {best:.4f}@{best_ep}, {(time.time()-t0):.0f}s]', flush=True)
            torch.save({'model': model.state_dict()}, ckpt)
            json.dump(hist, open(log_path, 'w'), indent=2)
            if (ep + 1) - best_ep >= args.patience:
                print(f'  early stop @ep{ep+1} (best {best:.4f}@ep{best_ep})', flush=True)
                break
        open(os.path.join(CKPT_DIR, f'{tag}_peak.txt'), 'w').write(
            f'PEAK {tag}: val full {best:.4f} @ep{best_ep}\n')
        model.load_state_dict(torch.load(best_ckpt, map_location=device)['model']); model.eval()

    # ---- TEST eval, SEED-AVERAGED over {1,2,3,7,11} (matches the V1 paper protocol) ----
    eval_seeds = [int(s) for s in os.environ.get('SEEDS', '1,2,3,7,11').split(',')]
    ks = [0, 1, 2, 4, 8]
    accf = []; acct = []; mpf = []; mpt = []; kacc = {k: ([], []) for k in ks}
    for sd in eval_seeds:
        ars = A.load_arena(seed=sd)
        ars['rat_by_u_dict'] = ar['rat_by_u_dict']
        tf, tt, mf, mt = eval_fullprofile(model, ars, ars['test_users'])
        mp = eval_mostpop(ars, ars['test_users'])
        accf.append(tf); acct.append(tt); mpf.append(mp[0]); mpt.append(mp[1])
        for k in ks:
            kf, kt, _, _ = eval_fullprofile(model, ars, ars['test_users'], klimit=k, kseed=sd)
            kacc[k][0].append(kf); kacc[k][1].append(kt)
    tf, tt = float(np.mean(accf)), float(np.mean(acct))
    mostpop = (float(np.mean(mpf)), float(np.mean(mpt)))
    print(f'\n=== d={args.latent} TEST seed-avg{eval_seeds} (te[300:]) ===', flush=True)
    print(f'  MOSTPOP        full {mostpop[0]:.4f} tail {mostpop[1]:.4f}', flush=True)
    print(f'  RecVAE fullprof full {tf:.4f} (sd {np.std(accf):.4f}) tail {tt:.4f} (sd {np.std(acct):.4f})  '
          f'(headroom over MOSTPOP: full {tf-mostpop[0]:+.4f} tail {tt-mostpop[1]:+.4f})', flush=True)
    kcurve = {}
    for k in ks:
        kf, kt = float(np.mean(kacc[k][0])), float(np.mean(kacc[k][1]))
        kcurve[k] = (kf, kt)
        print(f'  k={k:<2} full {kf:.4f} tail {kt:.4f}', flush=True)
    out = {'d': args.latent, 'test_full': tf, 'test_tail': tt, 'test_full_sd': float(np.std(accf)),
           'mostpop_full': mostpop[0], 'mostpop_tail': mostpop[1], 'seeds': eval_seeds,
           'kcurve': {str(k): v for k, v in kcurve.items()}}
    json.dump(out, open(os.path.join(CKPT_DIR, f'{tag}_TEST.json'), 'w'), indent=2)
    print(f'[saved] {tag}_TEST.json', flush=True)


def eval_mostpop(ar, users):
    accf = acct = 0.0; mf = mt = 0
    popb = ar['popb'].astype(np.float64)
    for x in users:
        profset, test = ar['SPL'][x]; rd = ar['rat_by_u_dict'][x]
        tlike = set(j for j in test if rd[j] >= 4)
        if not tlike:
            continue
        nf = A.ndcg_at10(popb, tlike, profset, ar['headmask'], False)
        if nf is not None:
            accf += nf; mf += 1
        nt = A.ndcg_at10(popb, tlike, profset, ar['headmask'], True)
        if nt is not None:
            acct += nt; mt += 1
    return (accf / max(mf, 1), acct / max(mt, 1))


if __name__ == '__main__':
    main()
