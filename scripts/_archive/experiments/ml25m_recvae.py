"""
ml25m_recvae.py -- port the Phase-1 replicated RecVAE recipe to the ML-25M INSTRUMENT 2.0 arena
(scripts/instrument2/ml25m_arena.py). Mirrors ml1m_recvae.py VERBATIM; only the arena module and
the input vocabulary (ni=18430) change.

RecVAE mapping (identical statement to ml1m_recvae, at 25M scale):
  - input vocabulary = ni=18430 items (the full-build catalogue, >=20 ratings).
  - train matrix = trU users x ni, binarized implicit positives (rating>=4).
    COMPUTE DEVIATION (stated, mirrors the Goodreads P2 port): optionally train on a fixed
    rng(0) NSUB-train-user subsample of the 161541 train users for CPU tractability (env NSUB).
    NSUB=0 -> all train users.
  - belief update = amortized encoder pass ONLY; k=0 -> z=prior-mean(0) (never feed empty input).
  - "full-profile fold" folds the profile-half LIKES (rating>=4); candidate-exclusion removes ALL
    profile items. score = decoder logits over ni; exclude profset; NDCG@10 (full + Cremonesi tail).
  - early-stop on the val (usable va) full-profile fold NDCG@10 full.
  - TEST = usable te cohort, seed-avg {1,2,3,7,11} (re-draw the profile-split per seed).

Usage: python scripts/instrument2/ml25m_recvae.py --latent 512 [--epochs 50] [--eval_only]
       NSUB=60000 MAX_MIN=600 python scripts/instrument2/ml25m_recvae.py --latent 512 --resume
"""
import os, sys, json, time, argparse
import numpy as np
import torch
from scipy import sparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ml25m_arena as A
from recvae import RecVAE, run_updates, make_predict_fn

torch.set_num_threads(4)
CKPT_DIR = os.path.join('.cache', 'instrument2')
os.makedirs(CKPT_DIR, exist_ok=True)


def build_train_matrix(ar, nsub=0, min_like=5):
    tr_u = ar['tr_u']; tr_i = ar['tr_i']; ni = ar['ni']
    # eligible train users = those with >= min_like implicit positives (matches ml1m_arena keep-rule
    # and Liang min_uc=5). CRITICAL: excludes 0-like train users whose all-zero rows L2-normalize to
    # NaN in the encoder.
    uu_uniq, ucnt = np.unique(tr_u, return_counts=True)
    elig = uu_uniq[ucnt >= min_like]
    if nsub and nsub < len(elig):
        rng = np.random.default_rng(0)
        keep = rng.choice(elig, size=nsub, replace=False)
        users = np.sort(keep)
    else:
        users = np.sort(elig)
    keepmask = np.zeros(ar['nu'], bool); keepmask[users] = True
    sel = keepmask[tr_u]
    tr_u = tr_u[sel]; tr_i = tr_i[sel]
    # dense-remap the kept train users to 0..M-1 rows
    umap = np.zeros(ar['nu'], np.int64); umap[users] = np.arange(len(users))
    rows = umap[tr_u]; cols = tr_i
    X = sparse.csr_matrix((np.ones(len(rows), np.float32), (rows, cols)),
                          dtype=np.float32, shape=(len(users), ni))
    return X


def fold_score(model, rd, ar, profset, klimit=None, rng=None):
    likes = [j for j in profset if rd.get(j, 0) >= 4]
    if klimit is not None:
        if klimit == 0:
            likes = []
        elif len(likes) > klimit:
            likes = list(rng.choice(likes, size=klimit, replace=False))
    xv = np.zeros((1, ar['ni']), np.float32)
    for j in likes:
        xv[0, j] = 1.0
    with torch.no_grad():
        if xv.sum() == 0:
            z = torch.zeros(1, model.decoder.in_features)
            s = model.decoder(z).numpy()[0]
        else:
            s = model(torch.tensor(xv), calculate_loss=False).numpy()[0]
    return s.astype(np.float64)


def eval_fullprofile(model, ar, rd_all, users, klimit=None, kseed=0):
    rng = np.random.default_rng(kseed)
    accf = acct = 0.0; mf = mt = 0
    for x in users:
        profset, test = ar['SPL'][x]; rd = rd_all[x]
        tlike = set(j for j in test if rd[j] >= 4)
        if not tlike:
            continue
        s = fold_score(model, rd, ar, profset, klimit=klimit, rng=rng)
        nf = A.ndcg_at10(s, tlike, profset, ar['headmask'], False)
        if nf is not None:
            accf += nf; mf += 1
        nt = A.ndcg_at10(s, tlike, profset, ar['headmask'], True)
        if nt is not None:
            acct += nt; mt += 1
    return (accf / max(mf, 1), acct / max(mt, 1), mf, mt)


def eval_mostpop(ar, rd_all, users):
    accf = acct = 0.0; mf = mt = 0
    popb = ar['popb'].astype(np.float64)
    for x in users:
        profset, test = ar['SPL'][x]; rd = rd_all[x]
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--latent', type=int, default=512)
    ap.add_argument('--hidden', type=int, default=600)
    ap.add_argument('--epochs', type=int, default=50)
    ap.add_argument('--batch', type=int, default=500)
    ap.add_argument('--lr', type=float, default=5e-4)
    ap.add_argument('--gamma', type=float, default=0.005)
    ap.add_argument('--n_enc', type=int, default=3)
    ap.add_argument('--n_dec', type=int, default=1)
    ap.add_argument('--dropout', type=float, default=0.5)
    ap.add_argument('--patience', type=int, default=8)
    ap.add_argument('--resume', action='store_true')
    ap.add_argument('--eval_only', action='store_true')
    args = ap.parse_args()
    NSUB = int(os.environ.get('NSUB', '0'))
    MAX_MIN = float(os.environ.get('MAX_MIN', '1e9'))

    torch.manual_seed(98765); np.random.seed(98765)
    device = torch.device('cpu')
    ar = A.load_arena(seed=123)
    rd_all = {x: dict(v) for x, v in ar['rat_by_u'].items()}
    ni = ar['ni']
    tag = f'ml25m_recvae_d{args.latent}'
    ckpt = os.path.join(CKPT_DIR, f'{tag}.pt'); best_ckpt = os.path.join(CKPT_DIR, f'{tag}_best.pt')
    log_path = os.path.join(CKPT_DIR, f'{tag}_log.json')
    print(f'[ml25m-recvae] d={args.latent} ni={ni} trU={len(ar["trU"])} NSUB={NSUB} '
          f'val={len(ar["val_users"])} test={len(ar["test_users"])}', flush=True)

    model = RecVAE(args.hidden, args.latent, ni).to(device)
    opt_enc = torch.optim.Adam(model.encoder.parameters(), lr=args.lr)
    opt_dec = torch.optim.Adam(model.decoder.parameters(), lr=args.lr)

    start_ep = 0; best = -1.0; best_ep = 0; hist = []
    if (args.resume or args.eval_only) and os.path.exists(ckpt):
        blob = torch.load(ckpt, map_location=device)
        model.load_state_dict(blob['model'])
        if 'opt_enc' in blob:
            opt_enc.load_state_dict(blob['opt_enc']); opt_dec.load_state_dict(blob['opt_dec'])
        st = blob.get('state', {}); start_ep = st.get('epoch', 0)
        best = st.get('best', -1.0); best_ep = st.get('best_ep', 0); hist = st.get('hist', [])
        print(f'[ml25m-recvae] RESUMED ep{start_ep} best {best:.4f}@ep{best_ep}', flush=True)

    if args.eval_only:
        model.load_state_dict(torch.load(best_ckpt, map_location=device)['model']); model.eval()
    else:
        X = build_train_matrix(ar, nsub=NSUB)
        print(f'[ml25m-recvae] train X: {X.shape} nnz={X.nnz}', flush=True)
        idxlist = np.arange(X.shape[0]); t0 = time.time()
        for ep in range(start_ep, args.epochs):
            run_updates(model, [opt_enc], X, idxlist, args.batch, device,
                        n_epochs=args.n_enc, dropout_rate=args.dropout, gamma=args.gamma, beta=None)
            model.update_prior()
            run_updates(model, [opt_dec], X, idxlist, args.batch, device,
                        n_epochs=args.n_dec, dropout_rate=0.0, gamma=args.gamma, beta=None)
            vf, vt, _, _ = eval_fullprofile(model, ar, rd_all, ar['val_users'])
            hist.append({'epoch': ep + 1, 'val_ndcg10_full': vf, 'val_ndcg10_tail': vt})
            mk = ''
            if vf > best:
                best = vf; best_ep = ep + 1
                torch.save({'model': model.state_dict(), 'args': vars(args),
                            'best_ep': best_ep, 'val_full': vf, 'val_tail': vt}, best_ckpt)
                mk = ' <- BEST'
            print(f'  ep{ep+1:3d} val full {vf:.4f} tail {vt:.4f}{mk} '
                  f'[best {best:.4f}@{best_ep}, {(time.time()-t0)/60:.1f}m]', flush=True)
            state = {'epoch': ep + 1, 'best': best, 'best_ep': best_ep, 'hist': hist}
            torch.save({'model': model.state_dict(), 'opt_enc': opt_enc.state_dict(),
                        'opt_dec': opt_dec.state_dict(), 'state': state, 'args': vars(args)}, ckpt)
            json.dump(hist, open(log_path, 'w'), indent=2)
            open(os.path.join(CKPT_DIR, f'{tag}_peak.txt'), 'w').write(
                f'PEAK {tag}: val full {best:.4f} @ep{best_ep} (NSUB={NSUB})\n')
            if (ep + 1) - best_ep >= args.patience:
                print(f'  early stop @ep{ep+1} (best {best:.4f}@ep{best_ep})', flush=True)
                break
            if (time.time() - t0) / 60 >= MAX_MIN:
                print(f'  wall budget {MAX_MIN}m hit @ep{ep+1}; re-run --resume to continue', flush=True)
                return
        model.load_state_dict(torch.load(best_ckpt, map_location=device)['model']); model.eval()

    # ---- TEST eval, SEED-AVERAGED {1,2,3,7,11} (re-draw profile-split per seed) ----
    eval_seeds = [int(s) for s in os.environ.get('SEEDS', '1,2,3,7,11').split(',')]
    ks = [0, 1, 2, 4, 8]
    accf = []; acct = []; mpf = []; mpt = []; kacc = {k: ([], []) for k in ks}
    for sd in eval_seeds:
        ars = A.load_arena(seed=sd)
        tf, tt, mf, mt = eval_fullprofile(model, ars, rd_all, ars['test_users'])
        mp = eval_mostpop(ars, rd_all, ars['test_users'])
        accf.append(tf); acct.append(tt); mpf.append(mp[0]); mpt.append(mp[1])
        for k in ks:
            kf, kt, _, _ = eval_fullprofile(model, ars, rd_all, ars['test_users'], klimit=k, kseed=sd)
            kacc[k][0].append(kf); kacc[k][1].append(kt)
    tf, tt = float(np.mean(accf)), float(np.mean(acct))
    mostpop = (float(np.mean(mpf)), float(np.mean(mpt)))
    print(f'\n=== d={args.latent} TEST seed-avg{eval_seeds} (te cohort) ===', flush=True)
    print(f'  MOSTPOP        full {mostpop[0]:.4f} tail {mostpop[1]:.4f}', flush=True)
    print(f'  RecVAE fullprof full {tf:.4f} (sd {np.std(accf):.4f}) tail {tt:.4f} (sd {np.std(acct):.4f})  '
          f'(headroom over MOSTPOP: full {tf-mostpop[0]:+.4f} tail {tt-mostpop[1]:+.4f})', flush=True)
    kcurve = {}
    for k in ks:
        kf, kt = float(np.mean(kacc[k][0])), float(np.mean(kacc[k][1]))
        kcurve[k] = (kf, kt)
        print(f'  k={k:<2} full {kf:.4f} tail {kt:.4f}', flush=True)
    out = {'d': args.latent, 'nsub': NSUB, 'test_full': tf, 'test_tail': tt,
           'test_full_sd': float(np.std(accf)), 'mostpop_full': mostpop[0], 'mostpop_tail': mostpop[1],
           'seeds': eval_seeds, 'kcurve': {str(k): v for k, v in kcurve.items()}}
    json.dump(out, open(os.path.join(CKPT_DIR, f'{tag}_TEST.json'), 'w'), indent=2)
    print(f'[saved] {tag}_TEST.json', flush=True)


if __name__ == '__main__':
    main()
