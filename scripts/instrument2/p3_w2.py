"""
p3_w2.py -- INSTRUMENT 2.0 Phase-3 W2: CONCEPT + DISLIKE CHANNEL (latent).
Concepts enter through the LATENT (Phase-1.6 rule: input-space pseudo-items FAILED). We convert a
concept (genome tag / GR shelf) to a unit latent direction q_concept, then apply the graded answer
through the W1 winning operator (default ADDITIVE z' = z + eta*a*q). Two concept-direction designs
are benchmarked:
  (i)  MEMBER-BAG ENCODE   q = unit(enc(lift-weighted top-M member-item bag))
  (ii) LEARNED TAG->Z      q = unit(E[z* | user likes tag] - E[z*])  (closed-form contrast on
       train users = the optimal linear indicator predictor = a learned tag->z mapping)
Per-user tag selection by LIFT = affinity/tag_mass (never raw affinity: generic-tag degeneracy).

Metrics (per HANDOFF plan):
  - FIDELITY     concept-only (z=0 + 1 top-lift concept answer) vs z=0 floor  (1.6 was BELOW floor)
  - ADDITIVITY   2 items + 2 concepts  vs  2 items alone
  - DISLIKE-A    negative answer (a<0) through the operator: member-item percentile-rank demotion
                 vs a non-member control (1.6 crude z-subtract: -15.4 members / -0.9 control)
(Dislike design B = two-channel encoder fine-tune, in p3_finetune_dislike.py.)

Usage: python scripts/instrument2/p3_w2.py --dataset ml1m [--op additive --eta 16]
       python scripts/instrument2/p3_w2.py --dataset gr   [--op additive --eta 16]
"""
import os, sys, json, argparse
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from recvae import RecVAE
import ml1m_arena as A

DEVICE = torch.device('cpu'); torch.set_num_threads(4)
CK = '.cache/instrument2'


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
    # pseudo-decode fold
    bag = base_bag.copy()
    qd = q if a >= 0 else -q
    s = decode(model, qd[None, :])[0]
    top = np.argpartition(-s, M)[:M]
    w = np.exp(s[top] - s[top].max()); w /= w.sum() + 1e-9
    bag[top] += abs(a) * w.astype(np.float32)
    return enc_mu(model, bag[None, :])[0]


# ============================ ML-1M ============================
def run_ml1m(op, eta, Mbag):
    model = load_ml1m(); ni = 3706; d = model.decoder.in_features
    C = np.load(f'{CK}/concepts_ml1m.npz', allow_pickle=True)
    item_tag = C['item_tag']; tag_mass = C['tag_mass']; names = C['tag_names']
    ntag = item_tag.shape[1]
    ar = A.load_arena(seed=123); ar['rat_by_u_dict'] = {x: dict(v) for x, v in ar['rat_by_u'].items()}

    # --- concept direction design (i): member-bag encode ---
    def bag_dir(tc):
        rel = item_tag[:, tc]
        top = np.argpartition(-rel, Mbag)[:Mbag]
        b = np.zeros(ni, np.float32); b[top] = rel[top]
        b /= b.sum() + 1e-9
        z = enc_mu(model, b[None, :])[0]
        return z / (np.linalg.norm(z) + 1e-9)
    Qi = np.stack([bag_dir(t) for t in range(ntag)])   # (ntag x d)

    # --- concept direction design (ii): learned tag->z contrast on TRAIN users ---
    trU = [x for x in ar['trU'] if sum(1 for j, r in ar['rat_by_u_dict'][x].items() if r >= 4) >= 4]
    Ztr = np.zeros((len(trU), d), np.float32); aff_tr = np.zeros((len(trU), ntag), np.float32)
    for st in range(0, len(trU), 500):
        ch = trU[st:st + 500]
        Xd = np.zeros((len(ch), ni), np.float32)
        for r, x in enumerate(ch):
            lk = [j for j, rr in ar['rat_by_u_dict'][x].items() if rr >= 4]
            Xd[r, lk] = 1.0
            aff_tr[st + r] = item_tag[lk].sum(0) / max(len(lk), 1)
        Ztr[st:st + len(ch)] = enc_mu(model, Xd)
    zbar = Ztr.mean(0)
    Qii = np.zeros((ntag, d), np.float32)
    for t in range(ntag):
        lift = aff_tr[:, t] / (tag_mass[t] + 1e-9)
        thr = np.quantile(lift, 0.8)
        sel = lift >= thr
        if sel.sum() >= 5:
            dz = Ztr[sel].mean(0) - zbar
            Qii[t] = dz / (np.linalg.norm(dz) + 1e-9)

    test = ar['test_users']
    # per-user z*, affinity, top-lift tag
    def zstar_of(x):
        lk = [j for j, r in ar['rat_by_u_dict'][x].items() if r >= 4]
        prof = [j for j in ar['SPL'][x][0] if ar['rat_by_u_dict'][x][j] >= 4]
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
            profset, tst = ar['SPL'][x]; rd = ar['rat_by_u_dict'][x]
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
        res[name] = {'fidelity_concept_only': float(np.mean(fid_f)),
                     'floor_z0': float(np.mean(fl_f)),
                     'delta_vs_floor': float(np.mean(fid_f) - np.mean(fl_f))}
    # pick better concept-direction design by fidelity
    best = 'member_bag' if res['member_bag']['delta_vs_floor'] >= res['learned_contrast']['delta_vs_floor'] else 'learned_contrast'
    Qbest = Qi if best == 'member_bag' else Qii

    # --- ADDITIVITY: 2 items vs 2 items + 2 concepts ---
    add_items = []; add_both = []
    rng = np.random.default_rng(7)
    for x in test:
        profset, tst = ar['SPL'][x]; rd = ar['rat_by_u_dict'][x]
        tlike = set(j for j in tst if rd[j] >= 4)
        if not tlike: continue
        zs, prof = zstar_of(x)
        lk = [j for j in prof]
        if len(lk) < 2: continue
        it2 = list(rng.choice(lk, 2, replace=False))
        z_it = enc_mu(model, onehot(it2, ni))[0]
        s_it = decode(model, z_it[None, :])[0]
        ni_it = A.ndcg_at10(s_it, tlike, profset, ar['headmask'], False)
        # add 2 top-lift concepts via operator, base_bag = the 2 items
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

    # --- DISLIKE-A: negative answer specificity ---
    dislike = dislike_specificity(model, ni, d, ar, item_tag, names, Qbest, op, eta, Mbag)
    out = {'concept_direction': res, 'best_design': best, 'additivity': additivity,
           'dislike_A': dislike, 'op': op, 'eta': eta, 'Mbag': Mbag}
    json.dump(out, open(f'{CK}/p3_w2_ml1m.json', 'w'), indent=2)
    print(json.dumps(out, indent=2))
    return out


def dislike_specificity(model, ni, d, ar, item_tag, names, Q, op, eta, Mbag):
    """Apply a<0 for a distinctive tag on a base profile; member vs control percentile demotion."""
    name_list = [str(x) for x in names]
    want = ['horror', 'violence', 'romance', 'comedy', 'dark', 'scary', 'gory', 'action']
    tcs = [name_list.index(w) for w in want if w in name_list][:5]
    rng = np.random.default_rng(11)
    per = {}
    test = ar['test_users']
    for tc in tcs:
        rel = item_tag[:, tc]
        members = np.argpartition(-rel, 50)[:50]
        base_scores = decode(model, np.zeros((1, d), np.float32))[0]
        control = np.argsort(-base_scores)[:50]                 # top-ranked non-members
        control = np.array([c for c in control if c not in set(members.tolist())])[:50]
        dm = []; dc = []
        for x in test[:150]:
            prof = [j for j in ar['SPL'][x][0] if ar['rat_by_u_dict'][x][j] >= 4]
            if len(prof) < 2: continue
            it2 = list(rng.choice(prof, 2, replace=False))
            z0 = enc_mu(model, onehot(it2, ni))[0]
            s0 = decode(model, z0[None, :])[0]
            r0 = pctrank(s0)
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


def pctrank(s):
    order = np.argsort(np.argsort(s))       # rank 0..ni-1
    return order / (len(s) - 1)


def onehot(items, ni):
    x = np.zeros((1, ni), np.float32)
    if len(items): x[0, list(items)] = 1.0
    return x


def load_ml1m():
    blob = torch.load(f'{CK}/ml1m_recvae_d512_best.pt', map_location=DEVICE)
    a = blob['args']; m = RecVAE(a['hidden'], a['latent'], 3706)
    m.load_state_dict(blob['model']); m.eval(); return m


# ============================ Goodreads ============================
def run_gr(op, eta, Mbag):
    import gr_recvae as G
    ar = G.load_arena()
    blob = torch.load(f'{CK}/gr_recvae_d512_best.pt', map_location=DEVICE)
    a = blob['args']; model = RecVAE(a['hidden'], a['latent'], G.NUNIV)
    model.load_state_dict(blob['model']); model.eval()
    NU = G.NUNIV; d = model.decoder.in_features
    univ = ar['univ']; loc = -np.ones(ar['ni'], np.int64); loc[univ] = np.arange(NU)
    Cc = np.load('.cache/goodreads/concepts_comp.npz', allow_pickle=True)
    ctags = Cc['ctags']; off = Cc['citems_off']; flat = Cc['citems_flat']
    # member items per shelf restricted to universe (local ids), keep shelves with >=20 members
    tag_members = []; tag_ok = []
    for t in range(len(ctags)):
        mem = flat[off[t]:off[t + 1]]
        ml = loc[mem]; ml = ml[ml >= 0]
        tag_members.append(ml)
        tag_ok.append(len(ml) >= 20)
    tag_ok = np.array(tag_ok)
    # concept direction (member-bag encode); popularity-weighted bag within universe
    popb = ar['popb'][univ]
    def bag_dir(t):
        ml = tag_members[t]
        if len(ml) == 0: return None
        b = np.zeros(NU, np.float32)
        w = popb[ml]; w = w / (w.sum() + 1e-9)
        # cap to top-Mbag by popularity to keep the bag focused
        if len(ml) > Mbag:
            keep = np.argsort(-w)[:Mbag]; ml = ml[keep]; w = w[keep]; w = w / (w.sum() + 1e-9)
        b[ml] = w
        z = enc_mu(model, b[None, :])[0]; return z / (np.linalg.norm(z) + 1e-9)
    # per-item -> membership matrix for affinity (sparse via dict)
    memb_of_item = [[] for _ in range(NU)]
    for t in range(len(ctags)):
        if tag_ok[t]:
            for il in tag_members[t]: memb_of_item[il].append(t)

    test = [x for x in ar['te'].tolist() if x in ar['SPL']]
    floor = decode(model, np.zeros((1, d), np.float32))[0]
    Qcache = {}
    def getQ(t):
        if t not in Qcache: Qcache[t] = bag_dir(t)
        return Qcache[t]

    fid = []; fl = []; live = []
    add_it = []; add_both = []
    rng = np.random.default_rng(7)
    for x in test:
        rd = ar['rat_by_u'][x]; tst = ar['SPL'][x]
        prof = [j for j in rd if (j not in tst) and ar['umask'][j] and rd[j] >= G.LIKE]
        prof_l = [loc[j] for j in prof]
        rel = [loc[t] for t in tst if ar['umask'][t]]
        if len(rel) < 1 or len(prof_l) < 1: continue
        zs = enc_mu(model, onehot(prof_l, NU))[0]
        # affinity over shelves = count of profile items that are members
        aff = np.zeros(len(ctags))
        for il in prof_l:
            for t in memb_of_item[il]: aff[t] += 1
        # lift = aff / (universe member count)
        msize = np.array([len(tag_members[t]) for t in range(len(ctags))], float) + 1e-9
        lift = np.where(tag_ok, aff / msize, -1)
        order = np.argsort(-lift)
        top1 = int(order[0]); n_answerable = int((aff > 0).sum())
        live.append(min(n_answerable, 8))     # answerable concepts among the vocab (cap 8)
        if lift[top1] <= 0:
            continue
        q = getQ(top1)
        if q is None: continue
        aq = float(zs @ q / (np.linalg.norm(zs) + 1e-9))
        z = apply_op(model, np.zeros(d, np.float32), q, aq, op, eta, NU, np.zeros(NU, np.float32), Mbag)
        s = decode(model, z[None, :])[0]
        nf = G.ndcg(s, rel, prof_l, set(np.where(ar['headmask'][univ])[0]), G.KP, False)
        n0 = G.ndcg(floor, rel, prof_l, set(np.where(ar['headmask'][univ])[0]), G.KP, False)
        if nf is not None: fid.append(nf); fl.append(n0)
        # additivity: 2 items + 2 concepts
        if len(prof_l) >= 2:
            it2 = list(rng.choice(prof_l, 2, replace=False))
            z_it = enc_mu(model, onehot(it2, NU))[0]
            s_it = decode(model, z_it[None, :])[0]
            nit = G.ndcg(s_it, rel, prof_l, set(np.where(ar['headmask'][univ])[0]), G.KP, False)
            z_both = z_it.copy(); base = onehot(it2, NU)[0].copy()
            for t in order[:2]:
                if lift[t] <= 0: continue
                q2 = getQ(int(t));
                if q2 is None: continue
                a2 = float(zs @ q2 / (np.linalg.norm(zs) + 1e-9))
                z_both = apply_op(model, z_both, q2, a2, op, eta, NU, base, Mbag)
            s_both = decode(model, z_both[None, :])[0]
            nbo = G.ndcg(s_both, rel, prof_l, set(np.where(ar['headmask'][univ])[0]), G.KP, False)
            if nit is not None and nbo is not None: add_it.append(nit); add_both.append(nbo)

    out = {'fidelity_concept_only': float(np.mean(fid)), 'floor_z0': float(np.mean(fl)),
           'delta_vs_floor': float(np.mean(fid) - np.mean(fl)),
           'additivity': {'items2': float(np.mean(add_it)), 'items2_plus_2concepts': float(np.mean(add_both)),
                          'delta': float(np.mean(add_both) - np.mean(add_it))},
           'answerability_liveness_mean': float(np.mean(live)),
           'answerability_frac_gt1': float(np.mean(np.array(live) > 1)),
           'n_test': len(fid), 'op': op, 'eta': eta}
    json.dump(out, open(f'{CK}/p3_w2_gr.json', 'w'), indent=2)
    print(json.dumps(out, indent=2))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', default='ml1m', choices=['ml1m', 'gr'])
    ap.add_argument('--op', default='additive', choices=['additive', 'pseudodecode'])
    ap.add_argument('--eta', type=float, default=16.0)
    ap.add_argument('--Mbag', type=int, default=50)
    args = ap.parse_args()
    if args.dataset == 'ml1m':
        run_ml1m(args.op, args.eta, args.Mbag)
    else:
        run_gr(args.op, args.eta, args.Mbag)


if __name__ == '__main__':
    main()
