"""
p4c_botplay.py -- INSTRUMENT 2.0 Phase 4c PART 4: the LEARNED bot-play answerer (ABot)
as a FOURTH answer source, alongside PART1 (empirical channel), PART2 (foreign geometry),
PART3 (raw-data pairs).

PROVENANCE. The bot-play / ABot answerer was built for Paper C on the V1 (64-d) encoder
(scripts/paper2/continuous_actor.py, commit b1aaa31, doc experiments/paper2/ABOT_CONFIDENCE_RESULT.md)
and then dropped ("bot-play/refusal DEAD") after two decisive negatives: drop-oracle headroom is
overfitting (cdd5296) and dropped-vs-kept answers carry no certainty signature (1c17130). The
checkpoint (.cache/abot.pt) is gone but the RECIPE survives verbatim: a heteroscedastic MLP
    ABot(z*, q, feats[5]) -> (mu, log_sigma^2),
trained by Gaussian heteroscedastic NLL on real ML-1M ratings; mu = the learned graded ANSWER,
1/sigma^2 = a learned confidence (familiarity). feats = [popularity/familiarity, divisiveness,
experience-max, experience-mean, taste-cos]. This script re-implements that recipe faithfully on
the CERTIFIED I2 geometry (RecVAE d512): z* = enc(profile likes), q = unit decoder item direction,
target answer = the SAME per-user-centered rating the other P4c arms fold (clip((r - user_mean)/2)).

WHAT MAKES IT THE 4TH SOURCE. PART1's empirical channel is a 1-D lookup a=f(s)+sampled-noise; the
ABot is a full LEARNED regressor of (z*, q, familiarity/experience) that predicts the answer's
CONDITIONAL MEAN mu (a denoised, richer answerer). It is in-distribution on real item directions
(item-8) and EXTRAPOLATES off-manifold for concept / continuous-actor directions -- exactly the
fidelity-boundary test.

Frozen invariants (identical to the other P4c arms): instrument ml1m_recvae_d512_best.pt; operator
z'=z+eta*a*q, eta=16, z0=0; arena ml1m_arena; seeds {1,2,3,7,11}; te[300:] TEST (304 users); T=8;
scale-matched (one global c=rms(s)/rms(a)) + per-user handling exactly as run_static_arm/run_actor_arm.

Usage: python scripts/instrument2/p4c_botplay.py {train|part4}
Artifacts: .cache/instrument2/abot_i2.pt (checkpoint), .cache/instrument2/p4c_botplay.json.
"""
import os, sys, json, time, argparse
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ml1m_arena as A
import p4a_battery as P
import p4c_answer_sources as PC  # reuse: run_static_arm, run_actor_arm, native_scale_for_actor,
#                                  cohort_ndcg, unit_rows, Designs, load_actors

DEVICE = torch.device('cpu'); torch.set_num_threads(4)
CK = '.cache/instrument2'
NI = 3706
ETA = 16.0
T = 8
SEEDS = [1, 2, 3, 7, 11]
ABOT_CK = f'{CK}/abot_i2.pt'
OUT = f'{CK}/p4c_botplay.json'
NFEAT = 5
TAU = 0.05                      # softmax temperature for the pop/div FIELD over off-manifold dirs


# ------------------------------------------------------------------ ABot (recovered recipe)
class ABot(nn.Module):
    """Heteroscedastic learned answerer: (z*, q_unit, feats[5]) -> (mu, log sigma^2).
    mu = graded answer in the centered-rating scale; 1/sigma^2 = learned confidence (familiarity)."""
    def __init__(s, d, h=128):
        super().__init__()
        s.net = nn.Sequential(nn.Linear(d + d + NFEAT, h), nn.ReLU(), nn.Linear(h, h), nn.ReLU())
        s.mu = nn.Linear(h, 1); s.ls = nn.Linear(h, 1)

    def forward(s, zstar, q, feats):
        h = s.net(torch.cat([zstar, q, feats], 1))
        return s.mu(h).squeeze(-1), s.ls(h).squeeze(-1)


# ------------------------------------------------------------------ item pop/div scalars
def item_pop_div(ar):
    """Per-item familiarity (log-count z) and divisiveness (rating-std z), from TRAIN ratings."""
    cnt = np.zeros(NI); ssum = np.zeros(NI); ssq = np.zeros(NI)
    for x in ar['trU']:
        for j, r in ar['rat_by_u'][x]:
            cnt[j] += 1; ssum[j] += r; ssq[j] += r * r
    with np.errstate(invalid='ignore', divide='ignore'):
        mean = ssum / np.maximum(cnt, 1)
        var = ssq / np.maximum(cnt, 1) - mean ** 2
    div = np.sqrt(np.clip(var, 0, None))
    div[cnt < 2] = np.nan
    pop = np.log(cnt + 1.0)
    gdiv = np.nanmean(div); div = np.where(np.isnan(div), gdiv, div)
    pop_z = ((pop - pop.mean()) / (pop.std() + 1e-9)).astype(np.float32)
    div_z = ((div - div.mean()) / (div.std() + 1e-9)).astype(np.float32)
    return pop_z, div_z


def popdiv_field(qn, D, pop_z, div_z, tau=TAU):
    """qn (K,d) unit -> (K,),(K,) smooth pop/div for ARBITRARY directions via softmax over items
    (peaks at the nearest real item; the honest off-manifold extension of the item scalars)."""
    sims = qn @ D.T                                     # (K,NI)
    sims = sims - sims.max(1, keepdims=True)
    w = np.exp(sims / tau); w /= w.sum(1, keepdims=True) + 1e-12
    return (w @ pop_z).astype(np.float32), (w @ div_z).astype(np.float32)


def exp_feats(qn, prof_dirs):
    """qn (d,) unit; prof_dirs (m,d) unit item dirs the user knows -> [max cos, mean-top5 cos]."""
    if prof_dirs is None or len(prof_dirs) == 0:
        return 0.0, 0.0
    c = np.sort(prof_dirs @ qn)[::-1]
    return float(c[0]), float(c[:min(5, len(c))].mean())


# ================================================================== TRAIN the ABot on I2 geometry
def train_abot(epochs=8):
    print('[abot] loading model + arena(123)...', flush=True)
    model, d = P.load_model()
    W = model.decoder.weight.detach().numpy()
    D = PC.unit_rows(W).astype(np.float32)
    ar = A.load_arena(seed=123)
    rd_all = {x: dict(v) for x, v in ar['rat_by_u'].items()}
    trU = [x for x in ar['trU'] if sum(1 for j, r in rd_all[x].items() if r >= 4) >= 4]
    pop_z, div_z = item_pop_div(ar)

    # z* = enc(user likes) per train user (batched); user rating-mean for centering
    print(f'[abot] encoding z* for {len(trU)} train users...', flush=True)
    Z = np.zeros((len(trU), d), np.float32); umean = np.zeros(len(trU), np.float32)
    for st in range(0, len(trU), 500):
        chk = trU[st:st + 500]; Xd = np.zeros((len(chk), NI), np.float32)
        for r, x in enumerate(chk):
            lk = [j for j, rr in rd_all[x].items() if rr >= 4]; Xd[r, lk] = 1.0
        Z[st:st + len(chk)] = P.enc_mu(model, Xd)
        for r, x in enumerate(chk):
            umean[st + r] = np.mean([rr for _, rr in ar['rat_by_u'][x]])

    # build (z*, q, feats, target) over all rated items; features via per-user vectorization
    print('[abot] building training set (real ML-1M ratings, leave-one-out experience)...', flush=True)
    Us, Qs, Fs, Ts = [], [], [], []
    for i, x in enumerate(trU):
        zs = Z[i]; nz = np.linalg.norm(zs) + 1e-9
        its = np.array([j for j, _ in ar['rat_by_u'][x]]); rts = np.array([r for _, r in ar['rat_by_u'][x]], np.float32)
        Du = D[its]                                     # (m,d) unit item dirs
        s = (Du @ zs) / nz                              # taste-cos per rated item
        G = Du @ Du.T                                    # (m,m) item-item cos
        np.fill_diagonal(G, -np.inf)                     # leave-one-out
        m = len(its)
        for k in range(m):
            c = np.sort(G[k])[::-1]; emax = float(c[0]) if m > 1 else 0.0
            emean = float(c[:min(5, m - 1)].mean()) if m > 1 else 0.0
            Us.append(zs); Qs.append(Du[k])
            Fs.append([pop_z[its[k]], div_z[its[k]], emax, emean, float(s[k])])
            Ts.append(np.clip((rts[k] - umean[i]) / 2.0, -1.0, 1.0))
    Us = torch.tensor(np.asarray(Us, np.float32)); Qs = torch.tensor(np.asarray(Qs, np.float32))
    Fs = torch.tensor(np.asarray(Fs, np.float32)); Ts = torch.tensor(np.asarray(Ts, np.float32))
    n = len(Ts); idx = np.random.default_rng(0).permutation(n); va = idx[:n // 10]; tr = idx[n // 10:]
    print(f'[abot] {n} (user,item) examples; train {len(tr)} val {len(va)}', flush=True)

    torch.manual_seed(0); abot = ABot(d)
    opt = torch.optim.Adam(abot.parameters(), lr=1e-3)
    for ep in range(epochs):
        np.random.default_rng(ep).shuffle(tr); tot = 0.0; nb = 0
        for b0 in range(0, len(tr), 4096):
            bb = tr[b0:b0 + 4096]
            mu, ls = abot(Us[bb], Qs[bb], Fs[bb]); ls = ls.clamp(-6, 4)
            nll = (0.5 * ((Ts[bb] - mu) ** 2) * torch.exp(-ls) + 0.5 * ls).mean()
            opt.zero_grad(); nll.backward(); opt.step(); tot += float(nll); nb += 1
        with torch.no_grad():
            mu, ls = abot(Us[va], Qs[va], Fs[va]); ls = ls.clamp(-6, 4)
            rmse = float(((Ts[va] - mu) ** 2).mean() ** 0.5)
            corr_mu = float(np.corrcoef(mu.numpy(), Ts[va].numpy())[0, 1])
        print(f'[abot] ep{ep}: train NLL {tot/max(nb,1):.3f} | val RMSE {rmse:.3f} corr(mu,a) {corr_mu:.3f}', flush=True)

    # ---- calibration gates (same three as the original ABot) ----
    with torch.no_grad():
        mu, ls = abot(Us[va], Qs[va], Fs[va]); ls = ls.clamp(-6, 4)
        s2 = torch.exp(ls).numpy(); err = np.abs((Ts[va] - mu).numpy())
        cal = float(np.corrcoef(s2, err)[0, 1])
        fp = Fs[va][:, 0].numpy()
        ph, pl = np.percentile(fp, 67), np.percentile(fp, 33)
        s2pop = float(s2[fp >= ph].mean()); s2niche = float(s2[fp <= pl].mean())
        corr_pop = float(np.corrcoef(s2, fp)[0, 1])
    gate = {'val_rmse': rmse, 'val_corr_mu_a': corr_mu,
            'gate1_sigma2_vs_err_corr': cal,
            'gate2_sigma2_popular': s2pop, 'gate2_sigma2_niche': s2niche, 'gate2_corr_sigma2_pop': corr_pop}
    print(f'[abot] GATE(1) sigma^2-vs-|err| corr {cal:+.3f} (want>0)', flush=True)
    print(f'[abot] GATE(2) sigma^2 POPULAR {s2pop:.3f} < NICHE {s2niche:.3f} ? corr {corr_pop:+.3f} (want popular=confident)', flush=True)
    torch.save({'state': abot.state_dict(), 'd': d, 'pop_z': pop_z, 'div_z': div_z, 'gate': gate}, ABOT_CK)
    print(f'[abot] saved {ABOT_CK}', flush=True)
    return abot, d, D, pop_z, div_z, gate


def load_abot():
    blob = torch.load(ABOT_CK, map_location=DEVICE)
    d = blob['d']; abot = ABot(d); abot.load_state_dict(blob['state']); abot.eval()
    for p in abot.parameters(): p.requires_grad_(False)
    return abot, d, blob['pop_z'], blob['div_z'], blob.get('gate', {})


# ================================================================== the bot-play a_fn(s)
def abot_mu_batch(abot, zstar, qn, feats):
    with torch.no_grad():
        mu, _ = abot(torch.tensor(zstar, dtype=torch.float32),
                     torch.tensor(qn, dtype=torch.float32),
                     torch.tensor(feats, dtype=torch.float32))
    return mu.numpy().astype(np.float32)


def make_static_afn(abot, D, pop_z, div_z, prof_dirs_by_user):
    """a_fn(x,t,q,item,zs,s) for run_static_arm. item given -> exact item pop/div; else field."""
    def a_fn(x, t_, q, item, zs, s):
        qn = (q / (np.linalg.norm(q) + 1e-9)).astype(np.float32)
        if item is not None:
            p, dv = pop_z[item], div_z[item]
        else:
            p, dv = popdiv_field(qn[None], D, pop_z, div_z)
            p, dv = float(p[0]), float(dv[0])
        emax, emean = exp_feats(qn, prof_dirs_by_user.get(x))
        feats = np.array([[p, dv, emax, emean, float(s)]], np.float32)
        return float(abot_mu_batch(abot, zs[None], qn[None], feats)[0])
    return a_fn


def make_actor_ansfn(abot, D, pop_z, div_z, prof_dirs_by_user):
    """ans_fn(t,q[U,d],s[U],users) for run_actor_arm; continuous dirs -> field pop/div (off-manifold)."""
    def ans_fn(t_, q, s, users):
        qn = q / (np.linalg.norm(q, axis=1, keepdims=True) + 1e-9)
        p, dv = popdiv_field(qn.astype(np.float32), D, pop_z, div_z)
        emax = np.zeros(len(users), np.float32); emean = np.zeros(len(users), np.float32)
        for u, x in enumerate(users):
            emax[u], emean[u] = exp_feats(qn[u].astype(np.float32), prof_dirs_by_user.get(x))
        feats = np.stack([p, dv, emax, emean, s.astype(np.float32)], 1)
        return abot_mu_batch(abot, PC._zst_cache[0], qn.astype(np.float32), feats)
    return ans_fn


# ================================================================== PART 4 eval
def prof_like_dirs(ar, users, D):
    """per-user unit decoder dirs of the user's revealed (profile-half) LIKE items."""
    out = {}
    for x in users:
        profset, _ = ar['SPL'][x]; rd = ar['rat_by_u_dict'][x]
        lk = [j for j in profset if rd[j] >= 4]
        out[x] = D[lk].astype(np.float32) if lk else None
    return out


def part4():
    model, d = P.load_model()
    W = model.decoder.weight.detach().numpy(); bdec = model.decoder.bias.detach().numpy()
    D = PC.unit_rows(W).astype(np.float32)
    dz = PC.Designs(model, d)
    rd_all = {x: dict(v) for x, v in A.load_arena(seed=123)['rat_by_u'].items()}
    actors = PC.load_actors(d)
    prim = max(actors, key=lambda n: actors[n][1] if actors[n][1] is not None else -1)
    abot, dA, pop_z, div_z, gate = load_abot()
    assert dA == d, 'ABot latent dim mismatch'
    print(f'[part4] primary actor={prim}; ABot gate={gate}', flush=True)

    arms = {'actor_seedavg': {'f': [], 't': [], 'c': []},
            'svd8_static': {'f': [], 't': [], 'c': []},
            'concept8_lift': {'f': [], 't': [], 'c': []},
            'item8': {'f': [], 't': [], 'c': []}}
    rng_item = None
    for sd in SEEDS:
        ar = P.arena_seed(sd, rd_all); test = ar['test_users']
        zst = P.build_zstar(model, ar, test)
        prof_dirs = prof_like_dirs(ar, test, D)
        a_static = make_static_afn(abot, D, pop_z, div_z, prof_dirs)
        rng_item = np.random.default_rng(sd)

        # ---- continuous actor (seed-avg over 3 train seeds), off-manifold ABot extrapolation ----
        PC._zst_cache = [zst.astype(np.float32)]          # actor ans_fn needs z* for the batch
        ans_fn = make_actor_ansfn(abot, D, pop_z, div_z, prof_dirs)
        afs = []; ats = []; cs = []
        for nm, (act, _) in actors.items():
            c = PC.native_scale_for_actor(act, model, W, bdec, ar, test, zst, ans_fn)
            f, t = PC.run_actor_arm(act, model, W, bdec, ar, test, zst, ans_fn, c=c)
            afs.append(f); ats.append(t); cs.append(c)
        arms['actor_seedavg']['f'].append(np.mean(afs)); arms['actor_seedavg']['t'].append(np.mean(ats))
        arms['actor_seedavg']['c'].append(np.mean(cs))

        # ---- SVD-8 static (field pop/div) ----
        f, t, c = PC.run_static_arm(model, W, bdec, ar, test, zst,
                                    lambda i, x, zs: (dz.svd8, None), a_static)
        arms['svd8_static']['f'].append(f); arms['svd8_static']['t'].append(t); arms['svd8_static']['c'].append(c)

        # ---- concept-8 lift (field pop/div) ----
        def cget(i, x, zs):
            sel, Q = dz.lift_concepts(x, ar, rd_all); return Q, None
        f, t, c = PC.run_static_arm(model, W, bdec, ar, test, zst, cget, a_static)
        arms['concept8_lift']['f'].append(f); arms['concept8_lift']['t'].append(t); arms['concept8_lift']['c'].append(c)

        # ---- item-8 (real item dirs -> ABot in-distribution, exact pop/div) ----
        def iget(i, x, zs):
            lk, Q = dz.item8(x, ar, rd_all, rng_item); return Q, lk
        f, t, c = PC.run_static_arm(model, W, bdec, ar, test, zst, iget, a_static)
        arms['item8']['f'].append(f); arms['item8']['t'].append(t); arms['item8']['c'].append(c)
        print(f'[part4] seed{sd} done', flush=True)

    out = {k: {'full': float(np.mean(v['f'])), 'tail': float(np.mean(v['t'])),
               'full_sd': float(np.std(v['f'])), 'scale_c': float(np.mean(v['c']))}
           for k, v in arms.items()}
    out['abot_gate'] = gate
    json.dump(out, open(OUT, 'w'), indent=2)
    print(f'[part4] saved {OUT}', flush=True)
    print(json.dumps(out, indent=2), flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('stage', choices=['train', 'part4'])
    ap.add_argument('--epochs', type=int, default=8)
    args = ap.parse_args()
    if args.stage == 'train':
        train_abot(args.epochs)
    else:
        part4()


if __name__ == '__main__':
    main()
