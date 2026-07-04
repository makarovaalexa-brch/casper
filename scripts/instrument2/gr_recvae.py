"""
gr_recvae.py -- port RecVAE (Phase-1 replicated recipe) to the Goodreads COMPOSITE arena.

Protocol mapping (stated; mirrors gr_ease_composite.py / the phase-1E gate VERBATIM):
  - arena: .cache/goodreads/base_comp.npz (ni=188,867 items, nu=768,746 users, 46.1M ratings).
  - like (implicit positive) = rating >= 4. NOTE: this DROPS dislikes (Phase-3 handles polarity),
    consistent with the Mult-VAE/RecVAE implicit-binary lineage.
  - RESTRICTED ITEM UNIVERSE = top-N=20,000 items by train-like count (76.8% of composite like-mass),
    identical to GOODREADS_EASE_DIAGNOSTIC so the RecVAE<->EASE comparison is apples-to-apples on the
    MATCHED universe (where the gate is defined). Full-catalogue RecVAE over 188k items is CPU-infeasible
    here; the restricted universe is the gate arena. (Full-catalogue view reported separately: the 20k-vocab
    logits embedded in the 188k catalogue, out-of-universe targets unreachable = a lower bound.)
  - COMPUTE-BUDGET deviation (stated, mirrors phase-1E's rotating-window V1 training): RecVAE trains on a
    fixed rng(0) subsample of NSUB train users (default 80k) that have >=1 like in the universe, not all
    767k -- to keep CPU epochs tractable. Resumable/chunked (--max_minutes, --resume).
  - eval: va=500 val / te=500 test users; rng(123) held-out disjoint targets (likes shuffled, 2nd half =
    target); profile = rated items not in target; candidate-exclusion removes profile; NDCG@K K in {10,510};
    fold-in input = profile LIKES within universe. belief = amortized encoder pass ONLY; empty -> z=0.

Usage: python scripts/instrument2/gr_recvae.py --latent 256 --max_minutes 9 --resume  (repeat until early-stop)
       python scripts/instrument2/gr_recvae.py --latent 256 --eval_only
"""
import os, sys, json, time, argparse
import numpy as np
import torch
from scipy import sparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from recvae import RecVAE, run_updates, make_predict_fn

GR = 'C:/dev/phd/casper/.cache/goodreads'
CKPT_DIR = os.path.join('.cache', 'instrument2')
os.makedirs(CKPT_DIR, exist_ok=True)
LIKE = 4.0
KP = 510
Ks = [10, KP]
NUNIV = 20000
_W = 1.0 / np.log2(np.arange(2, KP + 2))


def load_arena():
    B = np.load(f'{GR}/base_comp.npz')
    uu, ii, rr = B['uu'], B['ii'], B['rr']
    cnt = B['cnt']; ni = int(B['ni']); nu = int(B['nu'])
    trU = B['trU']; va = B['va']; te = B['te']
    univ = np.sort(np.argsort(-cnt)[:NUNIV])
    umask = np.zeros(ni, bool); umask[univ] = True
    popb = np.log(cnt + 1.0).astype(np.float32)
    # head-33% cumulative-popularity mask over the universe (Cremonesi tail)
    ucnt = cnt[univ]; order = np.argsort(-ucnt); cum = np.cumsum(ucnt[order]) / ucnt.sum()
    head_local = np.zeros(NUNIV, bool); head_local[order[:np.searchsorted(cum, 0.33) + 1]] = True
    headmask = np.zeros(ni, bool); headmask[univ[head_local]] = True
    # eval rating dicts (va+te)
    evalset = np.array(sorted(set(va.tolist()) | set(te.tolist())))
    selm = np.isin(uu, evalset)
    euu, eii, eR = uu[selm], ii[selm], rr[selm]
    rat_by_u = {}
    for k in range(len(euu)):
        rat_by_u.setdefault(int(euu[k]), []).append((int(eii[k]), float(eR[k])))
    _rs = np.random.default_rng(123); SPL = {}
    for x in (va.tolist() + te.tolist()):
        lk = [j for j, r in rat_by_u.get(x, []) if r >= LIKE]
        if len(lk) >= 4:
            ll = lk[:]; _rs.shuffle(ll); SPL[x] = set(ll[len(ll) // 2:])
    return dict(uu=uu, ii=ii, rr=rr, cnt=cnt, ni=ni, nu=nu, trU=trU, va=va, te=te,
                univ=univ, umask=umask, popb=popb, headmask=headmask,
                rat_by_u={x: dict(v) for x, v in rat_by_u.items()},
                rat_by_u_list=rat_by_u, SPL=SPL)


def build_train_matrix(ar, nsub, seed=0):
    """CSR (nsub_users x NUNIV) binarized likes within the universe."""
    uu, ii, rr = ar['uu'], ar['ii'], ar['rr']
    univ = ar['univ']; ni = ar['ni']
    loc = -np.ones(ni, np.int64); loc[univ] = np.arange(NUNIV)
    trset = np.zeros(ar['nu'], bool); trset[ar['trU']] = True
    mask = (rr >= LIKE) & ar['umask'][ii] & trset[uu]
    ru = uu[mask]; rc = loc[ii[mask]]
    # remap train users -> compact, then subsample
    uniq_u = np.unique(ru)
    rng = np.random.default_rng(seed)
    if nsub and len(uniq_u) > nsub:
        keep = np.sort(rng.choice(uniq_u, size=nsub, replace=False))
    else:
        keep = uniq_u
    remap = -np.ones(ar['nu'], np.int64); remap[keep] = np.arange(len(keep))
    sel = remap[ru] >= 0
    rows = remap[ru[sel]]; cols = rc[sel]
    X = sparse.csr_matrix((np.ones(len(rows), np.float32), (rows, cols)),
                          dtype=np.float32, shape=(len(keep), NUNIV))
    return X


def fold_z(model, ar, x, profset_local, klimit=None, rng=None):
    likes = profset_local
    if klimit is not None:
        if klimit == 0:
            likes = []
        elif len(likes) > klimit:
            likes = list(rng.choice(likes, size=klimit, replace=False))
    xv = np.zeros((1, NUNIV), np.float32)
    for j in likes:
        xv[0, j] = 1.0
    with torch.no_grad():
        if xv.sum() == 0:
            return model.decoder(torch.zeros(1, model.decoder.in_features)).numpy()[0].astype(np.float64)
        return model(torch.tensor(xv), calculate_loss=False).numpy()[0].astype(np.float64)


def ndcg(local_scores, rel_local, prof_local, head_local, K, tail):
    """local_scores: (NUNIV,). rel_local/prof_local/head_local are universe-local indices."""
    s = local_scores.copy()
    s[list(prof_local)] = -1e9
    if tail:
        s[list(head_local)] = -1e9
        rel = set(t for t in rel_local if t not in head_local)
    else:
        rel = set(rel_local)
    if not rel:
        return None
    kk = min(K, len(s))
    top = np.argpartition(-s, kk - 1)[:kk]; top = top[np.argsort(-s[top])]
    dcg = sum(_W[p] for p, t in enumerate(top) if int(t) in rel)
    idcg = _W[:min(K, len(rel))].sum() + 1e-12
    return dcg / idcg


def eval_recvae(model, ar, users, klimit=None, kseed=0):
    """Restricted-universe eval. Returns dict {K: {full,tail}} + MOSTPOP + n."""
    univ = ar['univ']; loc = -np.ones(ar['ni'], np.int64); loc[univ] = np.arange(NUNIV)
    head_local = set(np.where(ar['headmask'][univ])[0].tolist())
    popb_local = ar['popb'][univ].astype(np.float64)
    rng = np.random.default_rng(kseed)
    acc = {K: {'full': 0.0, 'tail': 0.0} for K in Ks}
    accmp = {K: {'full': 0.0, 'tail': 0.0} for K in Ks}
    nf = nt = 0
    for x in users:
        if x not in ar['SPL']:
            continue
        rd = ar['rat_by_u'][x]; test = ar['SPL'][x]
        prof = [j for j in rd if (j not in test) and ar['umask'][j]]
        rel = [t for t in test if ar['umask'][t]]
        prof_like = [loc[j] for j in prof if rd[j] >= LIKE]
        rel_local = [loc[t] for t in rel]
        prof_local = [loc[j] for j in prof]
        if len(rel) < 1 or len(prof) < 1:
            continue
        s = fold_z(model, ar, x, prof_like, klimit=klimit, rng=rng)
        got_f = got_t = False
        for K in Ks:
            vf = ndcg(s, rel_local, prof_local, head_local, K, False)
            vt = ndcg(s, rel_local, prof_local, head_local, K, True)
            mf = ndcg(popb_local, rel_local, prof_local, head_local, K, False)
            mt = ndcg(popb_local, rel_local, prof_local, head_local, K, True)
            if vf is not None:
                acc[K]['full'] += vf; accmp[K]['full'] += mf; got_f = True
            if vt is not None:
                acc[K]['tail'] += vt; accmp[K]['tail'] += mt; got_t = True
        nf += got_f; nt += got_t
    for K in Ks:
        acc[K]['full'] /= max(nf, 1); acc[K]['tail'] /= max(nt, 1)
        accmp[K]['full'] /= max(nf, 1); accmp[K]['tail'] /= max(nt, 1)
    return acc, accmp, nf, nt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--latent', type=int, default=256)
    ap.add_argument('--hidden', type=int, default=600)
    ap.add_argument('--epochs', type=int, default=60)
    ap.add_argument('--batch', type=int, default=500)
    ap.add_argument('--lr', type=float, default=5e-4)
    ap.add_argument('--gamma', type=float, default=0.005)
    ap.add_argument('--n_enc', type=int, default=3)
    ap.add_argument('--n_dec', type=int, default=1)
    ap.add_argument('--dropout', type=float, default=0.5)
    ap.add_argument('--patience', type=int, default=8)
    ap.add_argument('--nsub', type=int, default=80000)
    ap.add_argument('--max_minutes', type=float, default=9.0)
    ap.add_argument('--resume', action='store_true')
    ap.add_argument('--eval_only', action='store_true')
    args = ap.parse_args()

    torch.manual_seed(98765); np.random.seed(98765)
    device = torch.device('cpu')
    print(f'[gr-recvae] loading composite arena...', flush=True)
    ar = load_arena()
    tag = f'gr_recvae_d{args.latent}'
    ckpt = os.path.join(CKPT_DIR, f'{tag}.pt'); best_ckpt = os.path.join(CKPT_DIR, f'{tag}_best.pt')
    log_path = os.path.join(CKPT_DIR, f'{tag}_log.json')
    print(f'[gr-recvae] d={args.latent} universe={NUNIV} nsub={args.nsub} '
          f'val={sum(1 for x in ar["va"].tolist() if x in ar["SPL"])} '
          f'test={sum(1 for x in ar["te"].tolist() if x in ar["SPL"])}', flush=True)

    model = RecVAE(args.hidden, args.latent, NUNIV).to(device)
    opt_enc = torch.optim.Adam(model.encoder.parameters(), lr=args.lr)
    opt_dec = torch.optim.Adam(model.decoder.parameters(), lr=args.lr)
    state = {'epoch': 0, 'best': -1.0, 'best_ep': 0, 'hist': []}

    if args.eval_only or args.resume:
        src = best_ckpt if args.eval_only else ckpt
        if os.path.exists(src):
            blob = torch.load(src, map_location=device)
            model.load_state_dict(blob['model'])
            if not args.eval_only and 'opt_enc' in blob:
                opt_enc.load_state_dict(blob['opt_enc']); opt_dec.load_state_dict(blob['opt_dec'])
                state = blob['state']
                model.update_prior()  # restore encoder_old ~ current (prior state not persisted)
                print(f'[gr-recvae] RESUMED @ep{state["epoch"]} best {state["best"]:.4f}@{state["best_ep"]}', flush=True)

    if not args.eval_only:
        X = build_train_matrix(ar, args.nsub)
        print(f'[gr-recvae] train X: {X.shape} nnz={X.nnz}', flush=True)
        idxlist = np.arange(X.shape[0]); predict = make_predict_fn(model, device)
        t0 = time.time()
        for ep in range(state['epoch'], args.epochs):
            run_updates(model, [opt_enc], X, idxlist, args.batch, device,
                        n_epochs=args.n_enc, dropout_rate=args.dropout, gamma=args.gamma, beta=None)
            model.update_prior()
            run_updates(model, [opt_dec], X, idxlist, args.batch, device,
                        n_epochs=args.n_dec, dropout_rate=0.0, gamma=args.gamma, beta=None)
            acc, accmp, nf, nt = eval_recvae(model, ar, ar['va'].tolist())
            vf = acc[KP]['full']
            state['epoch'] = ep + 1
            state['hist'].append({'epoch': ep + 1, 'val_ndcg510_full': vf, 'val_ndcg510_tail': acc[KP]['tail']})
            mk = ''
            if vf > state['best']:
                state['best'] = vf; state['best_ep'] = ep + 1; mk = ' <- BEST'
                torch.save({'model': model.state_dict(), 'args': vars(args),
                            'best_ep': ep + 1, 'val_full': vf, 'val_tail': acc[KP]['tail']}, best_ckpt)
                open(os.path.join(CKPT_DIR, f'{tag}_peak.txt'), 'w').write(
                    f'PEAK {tag}: val @510 full {vf:.4f} tail {acc[KP]["tail"]:.4f} @ep{ep+1} '
                    f'(MOSTPOP val {accmp[KP]["full"]:.4f}, headroom {vf-accmp[KP]["full"]:+.4f})\n')
            el = (time.time() - t0) / 60.0
            print(f'  ep{ep+1:3d} val@510 full {vf:.4f} tail {acc[KP]["tail"]:.4f} '
                  f'(MP {accmp[KP]["full"]:.4f}, hr {vf-accmp[KP]["full"]:+.4f}){mk} [{el:.1f}m]', flush=True)
            torch.save({'model': model.state_dict(), 'opt_enc': opt_enc.state_dict(),
                        'opt_dec': opt_dec.state_dict(), 'state': state, 'args': vars(args)}, ckpt)
            json.dump(state['hist'], open(log_path, 'w'), indent=2)
            if (ep + 1) - state['best_ep'] >= args.patience:
                print(f'  early stop @ep{ep+1} (best {state["best"]:.4f}@ep{state["best_ep"]})', flush=True)
                break
            if el >= args.max_minutes:
                print(f'  [budget {args.max_minutes}m hit @ep{ep+1}] re-run with --resume to continue', flush=True)
                return
        model.load_state_dict(torch.load(best_ckpt, map_location=device)['model'])

    model.eval()
    # ---- TEST eval (restricted universe = gate arena) ----
    acc, accmp, nf, nt = eval_recvae(model, ar, ar['te'].tolist())
    print(f'\n=== d={args.latent} TEST (restricted universe N={NUNIV}, n_full={nf}) ===', flush=True)
    for K in Ks:
        hrf = acc[K]['full'] - accmp[K]['full']; hrt = acc[K]['tail'] - accmp[K]['tail']
        print(f'  @{K}: MOSTPOP full {accmp[K]["full"]:.4f} tail {accmp[K]["tail"]:.4f} | '
              f'RecVAE full {acc[K]["full"]:.4f} tail {acc[K]["tail"]:.4f} | '
              f'headroom full {hrf:+.4f} tail {hrt:+.4f}', flush=True)
    kcurve = {}
    for k in [0, 1, 2, 4, 8]:
        a2, _, _, _ = eval_recvae(model, ar, ar['te'].tolist(), klimit=k, kseed=0)
        kcurve[k] = {'full': a2[KP]['full'], 'tail': a2[KP]['tail']}
        print(f'  k={k:<2} @510 full {a2[KP]["full"]:.4f} tail {a2[KP]["tail"]:.4f}', flush=True)
    out = {'d': args.latent, 'universe': NUNIV, 'n': nf,
           'mostpop': {str(K): accmp[K] for K in Ks},
           'recvae': {str(K): acc[K] for K in Ks},
           'headroom_510_full': acc[KP]['full'] - accmp[KP]['full'],
           'kcurve': {str(k): v for k, v in kcurve.items()}}
    json.dump(out, open(os.path.join(CKPT_DIR, f'{tag}_TEST.json'), 'w'), indent=2)
    print(f'[saved] {tag}_TEST.json', flush=True)


if __name__ == '__main__':
    main()
