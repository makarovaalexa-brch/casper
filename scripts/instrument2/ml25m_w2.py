"""
ml25m_w2.py -- INSTRUMENT 2.0 Phase-3 W2 concept/dislike channel on the ML-25M arena.
Mirrors p3_w2.run_ml1m VERBATIM (member-bag vs learned-contrast concept direction, additive
operator eta=16, additivity 2it+2concept, dislike a<0 specificity), swapping in ml25m_arena,
ni=18430, ml25m_recvae_d512_best.pt, concepts_ml25m.npz. Adds concept answerability-liveness
(answered top-lift concepts per test user, cap 8) so G3 is a genuine ML-25M number.

Output: .cache/instrument2/p3_w2_ml25m.json
Usage: python scripts/instrument2/ml25m_w2.py [--op additive --eta 16 --Mbag 50]
"""
import os, sys, json, argparse
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from recvae import RecVAE
import ml25m_arena as A

DEVICE = torch.device('cpu'); torch.set_num_threads(4)
CK = '.cache/instrument2'
NI = 18430


def enc_mu(model, Xd):
    with torch.no_grad():
        mu, _ = model.encoder(torch.tensor(np.asarray(Xd, np.float32)), dropout_rate=0.0)
    return mu.numpy()


def decode(model, Z):
    with torch.no_grad():
        return model.decoder(torch.tensor(np.asarray(Z, np.float32))).numpy().astype(np.float64)


def apply_op(model, z0, q, a, op, eta, ni=None, base_bag=None, M=50):
    if op == 'additive':
        return z0 + eta * a * q
    bag = base_bag.copy(); qd = q if a >= 0 else -q
    s = decode(model, qd[None, :])[0]
    top = np.argpartition(-s, M)[:M]
    w = np.exp(s[top] - s[top].max()); w /= w.sum() + 1e-9
    bag[top] += abs(a) * w.astype(np.float32)
    return enc_mu(model, bag[None, :])[0]


def onehot(items, ni):
    x = np.zeros((1, ni), np.float32)
    if len(items): x[0, list(items)] = 1.0
    return x


def pctrank(s):
    order = np.argsort(np.argsort(s))
    return order / (len(s) - 1)


def load_model():
    blob = torch.load(f'{CK}/ml25m_recvae_d512_best.pt', map_location=DEVICE)
    a = blob['args']; m = RecVAE(a['hidden'], a['latent'], NI)
    m.load_state_dict(blob['model']); m.eval(); return m


def dislike_specificity(model, ni, d, ar, rd_all, item_tag, names, Q, op, eta, Mbag):
    name_list = [str(x) for x in names]
    want = ['horror', 'violence', 'romance', 'comedy', 'dark', 'scary', 'gory', 'action']
    tcs = [name_list.index(w) for w in want if w in name_list][:5]
    if not tcs:  # fallback: 5 most distinctive (lowest-mass) tags
        tcs = list(np.argsort(item_tag.sum(0))[:5])
    rng = np.random.default_rng(11); per = {}; test = ar['test_users']
    for tc in tcs:
        rel = item_tag[:, tc]
        members = np.argpartition(-rel, 50)[:50]
        base_scores = decode(model, np.zeros((1, d), np.float32))[0]
        control = np.argsort(-base_scores)[:50]
        control = np.array([c for c in control if c not in set(members.tolist())])[:50]
        dm = []; dc = []
        for x in test[:150]:
            prof = [j for j in ar['SPL'][x][0] if rd_all[x][j] >= 4]
            if len(prof) < 2: continue
            it2 = list(rng.choice(prof, 2, replace=False))
            z0 = enc_mu(model, onehot(it2, ni))[0]
            s0 = decode(model, z0[None, :])[0]; r0 = pctrank(s0)
            q = Q[tc]
            zd = apply_op(model, z0, q, -1.0, op, eta, ni, onehot(it2, ni)[0].copy(), Mbag)
            s1 = decode(model, zd[None, :])[0]; r1 = pctrank(s1)
            dm.append((r1[members] - r0[members]).mean())
            dc.append((r1[control] - r0[control]).mean())
        per[name_list[tc]] = {'member_demotion_pts': float(np.mean(dm) * 100),
                              'control_demotion_pts': float(np.mean(dc) * 100)}
    md = float(np.mean([v['member_demotion_pts'] for v in per.values()]))
    cd = float(np.mean([v['control_demotion_pts'] for v in per.values()]))
    return {'per_tag': per, 'mean_member_demotion_pts': md, 'mean_control_demotion_pts': cd,
            'specificity_pts': md - cd}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--op', default='additive', choices=['additive', 'pseudodecode'])
    ap.add_argument('--eta', type=float, default=16.0)
    ap.add_argument('--Mbag', type=int, default=50)
    args = ap.parse_args()
    op, eta, Mbag = args.op, args.eta, args.Mbag

    model = load_model(); ni = NI; d = model.decoder.in_features
    C = np.load(f'{CK}/concepts_ml25m.npz', allow_pickle=True)
    item_tag = C['item_tag']; tag_mass = C['tag_mass']; names = C['tag_names']
    ntag = item_tag.shape[1]
    ar = A.load_arena(seed=123)
    rd_all = {x: dict(v) for x, v in ar['rat_by_u'].items()}

    def bag_dir(tc):
        rel = item_tag[:, tc]
        top = np.argpartition(-rel, Mbag)[:Mbag]
        b = np.zeros(ni, np.float32); b[top] = rel[top]; b /= b.sum() + 1e-9
        z = enc_mu(model, b[None, :])[0]
        return z / (np.linalg.norm(z) + 1e-9)
    Qi = np.stack([bag_dir(t) for t in range(ntag)])

    # learned tag->z contrast on TRAIN users (subsample for speed: 4000 eligible train users)
    trU_all = ar['trU']
    rng0 = np.random.default_rng(0)
    trU = rng0.choice(trU_all, size=min(4000, len(trU_all)), replace=False)
    # need like-profiles for train users: build from base tr_u/tr_i? Those are train likes.
    # Build per-train-user likes from the arena base tr arrays.
    tr_u = ar['tr_u']; tr_i = ar['tr_i']
    keepmask = np.zeros(ar['nu'], bool); keepmask[trU] = True
    sel = keepmask[tr_u]
    likes_map = {}
    for u, it in zip(tr_u[sel].tolist(), tr_i[sel].tolist()):
        likes_map.setdefault(u, []).append(it)
    trU = [x for x in trU.tolist() if len(likes_map.get(x, [])) >= 4]
    Ztr = np.zeros((len(trU), d), np.float32); aff_tr = np.zeros((len(trU), ntag), np.float32)
    for st in range(0, len(trU), 500):
        ch = trU[st:st + 500]
        Xd = np.zeros((len(ch), ni), np.float32)
        for r, x in enumerate(ch):
            lk = likes_map[x]; Xd[r, lk] = 1.0
            aff_tr[st + r] = item_tag[lk].sum(0) / max(len(lk), 1)
        Ztr[st:st + len(ch)] = enc_mu(model, Xd)
    zbar = Ztr.mean(0); Qii = np.zeros((ntag, d), np.float32)
    for t in range(ntag):
        lift = aff_tr[:, t] / (tag_mass[t] + 1e-9)
        sel2 = lift >= np.quantile(lift, 0.8)
        if sel2.sum() >= 5:
            dz = Ztr[sel2].mean(0) - zbar; Qii[t] = dz / (np.linalg.norm(dz) + 1e-9)

    test = ar['test_users']

    def zstar_of(x):
        prof = [j for j in ar['SPL'][x][0] if rd_all[x][j] >= 4]
        return enc_mu(model, onehot(prof, ni))[0], prof

    def toplift(prof, n=2):
        aff = item_tag[prof].sum(0) if prof else np.zeros(ntag)
        lift = aff / (tag_mass + 1e-9)
        return np.argsort(-lift)[:n]

    floor = decode(model, np.zeros((1, d), np.float32))[0]
    res = {}
    for name, Q in (('member_bag', Qi), ('learned_contrast', Qii)):
        fid_f = []; fl_f = []
        for x in test:
            profset, tst = ar['SPL'][x]; rd = rd_all[x]
            tlike = set(j for j in tst if rd[j] >= 4)
            if not tlike: continue
            zs, prof = zstar_of(x)
            tc = int(toplift(prof, 1)[0])
            q = Q[tc]; a = float(zs @ q / (np.linalg.norm(zs) + 1e-9))
            z = apply_op(model, np.zeros(d, np.float32), q, a, op, eta, ni, np.zeros(ni, np.float32), Mbag)
            s = decode(model, z[None, :])[0]
            nf = A.ndcg_at10(s, tlike, profset, ar['headmask'], False)
            n0 = A.ndcg_at10(floor, tlike, profset, ar['headmask'], False)
            if nf is not None: fid_f.append(nf); fl_f.append(n0)
        res[name] = {'fidelity_concept_only': float(np.mean(fid_f)), 'floor_z0': float(np.mean(fl_f)),
                     'delta_vs_floor': float(np.mean(fid_f) - np.mean(fl_f))}
    best = 'member_bag' if res['member_bag']['delta_vs_floor'] >= res['learned_contrast']['delta_vs_floor'] else 'learned_contrast'
    Qbest = Qi if best == 'member_bag' else Qii

    # additivity + answerability liveness
    add_items = []; add_both = []; live = []
    rng = np.random.default_rng(7)
    for x in test:
        profset, tst = ar['SPL'][x]; rd = rd_all[x]
        tlike = set(j for j in tst if rd[j] >= 4)
        if not tlike: continue
        zs, prof = zstar_of(x)
        aff = item_tag[prof].sum(0) if prof else np.zeros(ntag)
        live.append(min(int((aff > 0).sum()), 8))
        lk = [j for j in prof]
        if len(lk) < 2: continue
        it2 = list(rng.choice(lk, 2, replace=False))
        z_it = enc_mu(model, onehot(it2, ni))[0]
        s_it = decode(model, z_it[None, :])[0]
        ni_it = A.ndcg_at10(s_it, tlike, profset, ar['headmask'], False)
        tcs = toplift(prof, 2)
        z_both = z_it.copy(); base_bag = onehot(it2, ni)[0].copy()
        for tc in tcs:
            q = Qbest[int(tc)]; a = float(zs @ q / (np.linalg.norm(zs) + 1e-9))
            z_both = apply_op(model, z_both, q, a, op, eta, ni, base_bag, Mbag)
        s_both = decode(model, z_both[None, :])[0]
        ni_both = A.ndcg_at10(s_both, tlike, profset, ar['headmask'], False)
        if ni_it is not None and ni_both is not None:
            add_items.append(ni_it); add_both.append(ni_both)
    additivity = {'items2': float(np.mean(add_items)), 'items2_plus_2concepts': float(np.mean(add_both)),
                  'delta': float(np.mean(add_both) - np.mean(add_items)), 'design': best}

    dislike = dislike_specificity(model, ni, d, ar, rd_all, item_tag, names, Qbest, op, eta, Mbag)
    out = {'concept_direction': res, 'best_design': best, 'additivity': additivity,
           'dislike_A': dislike, 'answerability_liveness_mean': float(np.mean(live)),
           'answerability_frac_gt1': float(np.mean(np.array(live) > 1)),
           'op': op, 'eta': eta, 'Mbag': Mbag}
    json.dump(out, open(f'{CK}/p3_w2_ml25m.json', 'w'), indent=2)
    print(json.dumps(out, indent=2))


if __name__ == '__main__':
    main()
