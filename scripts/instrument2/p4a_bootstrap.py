"""
p4a_bootstrap.py -- paired per-user bootstrap for the Phase-4a key margins, done cleanly against
SINGLE static methods (not a per-user max). Primary actor = p4a_actor_s0.pt (seed-avg is within
0.0015 of it; all three train-seeds tie). Per-user NDCG (full + tail) averaged across the 5 eval
seeds {1,2,3,7,11}; paired bootstrap of actor - comparator.

Comparators:
  strongest_static  = decoder-SVD basis-8 (0.4709 full, the strong static bar)
  concept_discrete  = per-user top-8 lift-selected concepts (uent+GRAW analogue, 0.4436 full)
"""
import os, sys, json
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ml1m_arena as A
import p4a_battery as P

CK = '.cache/instrument2'


def per_user(model, W, bdec, ar, users, zst_np, actor, svd8, Qc, item_tag, tag_mass, d):
    zst = torch.tensor(zst_np); nz = zst.norm(dim=1, keepdim=True) + 1e-9
    z = torch.zeros(len(users), d)
    with torch.no_grad():
        for t in range(P.T):
            q = actor(z, t); a = (zst * q).sum(1, keepdim=True) / nz; z = z + P.ETA * a * q
        Sa = (z @ W.T + bdec).numpy().astype(np.float64)
    Wn = model.decoder.weight.detach().numpy(); bn = model.decoder.bias.detach().numpy()
    rec = {}
    for i, x in enumerate(users):
        profset, tst = ar['SPL'][x]; rd = ar['rat_by_u_dict'][x]
        tlike = set(j for j in tst if rd[j] >= 4)
        if not tlike: continue
        prof = [j for j in profset if rd[j] >= 4]
        aff = item_tag[prof].sum(0) if prof else np.zeros(item_tag.shape[1])
        lift = aff / (tag_mass + 1e-9); sel = [c for c in np.argsort(-lift)[:8] if lift[c] > 0]
        zc = P.unroll_static(zst_np[i], Qc[sel]) if sel else np.zeros(d, np.float32)
        zs = P.unroll_static(zst_np[i], svd8)
        row = {}
        for tail in (False, True):
            na = A.ndcg_at10(Sa[i], tlike, profset, ar['headmask'], tail)
            ns = A.ndcg_at10(zs @ Wn.T + bn, tlike, profset, ar['headmask'], tail)
            ncp = A.ndcg_at10(zc @ Wn.T + bn, tlike, profset, ar['headmask'], tail)
            row[tail] = (na, ns, ncp)
        rec[x] = row
    return rec


def boot(diff, nboot=5000):
    diff = np.array(diff); n = len(diff); rng = np.random.default_rng(0)
    means = np.array([diff[rng.integers(0, n, n)].mean() for _ in range(nboot)])
    lo, hi = np.percentile(means, [2.5, 97.5])
    return {'mean_diff': float(diff.mean()), 'ci95': [float(lo), float(hi)],
            'p_gt0': float((means > 0).mean()), 'n_users': int(n)}


def main():
    model, d = P.load_model()
    W = torch.tensor(model.decoder.weight.detach().numpy(), dtype=torch.float32)
    bdec = torch.tensor(model.decoder.bias.detach().numpy(), dtype=torch.float32)
    rd_all = {x: dict(v) for x, v in A.load_arena(seed=123)['rat_by_u'].items()}
    svd8 = P.decoder_svd_dirs(model, 8)
    Qc, item_tag, tag_mass, names = P.concept_dirs(model)
    blob = torch.load(f'{CK}/p4a_actor_s0.pt', map_location='cpu')
    actor = P.Actor(d); actor.load_state_dict(blob['state']); actor.eval()

    acc = {}  # x -> {tail: [ (na,ns,ncp), ... over seeds ]}
    for sd in P.SEEDS:
        ar = P.arena_seed(sd, rd_all); test = ar['test_users']
        zst_np = P.build_zstar(model, ar, test)
        rec = per_user(model, W, bdec, ar, test, zst_np, actor, svd8, Qc, item_tag, tag_mass, d)
        for x, row in rec.items():
            for tail in (False, True):
                acc.setdefault(x, {False: [], True: []})[tail].append(row[tail])
    out = {}
    for tail in (False, True):
        da_s = []; da_c = []
        for x in acc:
            arr = np.array(acc[x][tail], float)               # (nseeds,3)
            if np.isnan(arr).any(): continue
            m = arr.mean(0)                                   # (na, ns, ncp)
            da_s.append(m[0] - m[1]); da_c.append(m[0] - m[2])
        key = 'tail' if tail else 'full'
        out[key] = {'actor_vs_strongest_static_svd8': boot(da_s),
                    'actor_vs_concept_discrete_lift8': boot(da_c)}
    json.dump(out, open(f'{CK}/p4a_bootstrap.json', 'w'), indent=2)
    print(json.dumps(out, indent=2))


if __name__ == '__main__':
    main()
