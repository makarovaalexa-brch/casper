"""
LastFM pipeline (the real-data adaptivity demonstrator).

Stages, each cached/idempotent and wrapped so a later failure preserves
earlier outputs:
  1. build   : slate (300 artists targets + 60 genre tags) + profiles
  2. train   : dual-head set-encoder instrument (liked + answerability)
  3. accept  : monotonicity + liked/disliked overlap (generic gates)
  4. bench   : training-free heuristics (random/popularity/greedy/scpr/
               thompson) via the shared testbed + efficiency
Tier-2 learned policies (PPO/bot-play) are a separate follow-up.

Run from casper root: poetry run python scripts/paper1/lastfm_pipeline.py
"""

import sys
sys.path.insert(0, '.')
sys.path.insert(0, 'scripts/paper1')

import json
import time
import zipfile
import urllib.request
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from synthetic_sanity import DualHeadSetEncoder
from test_instrument_lib import DualSetInstrumentWrapper
from testbed import SimulatedUser, run_episode, summarize
import policies as P

ROOT = Path('C:/dev/phd/casper')
LF = ROOT / 'data/lastfm'
CKPT = ROOT / 'data/movielens/.cache/checkpoints'
EXP = ROOT / 'experiments/paper1'
PROF_NPZ = LF / 'lastfm_profiles.npz'
INST = CKPT / 'instrument_lastfm_dual.pt'
N_ARTISTS, N_TAGS = 300, 60
N_TARGETS = N_ARTISTS
SEED = 42
N_TURNS = 15
MAX_REVEAL = 60
np.random.seed(SEED); torch.manual_seed(SEED)


# --------------------------- stage 1: build ---------------------------------
def build():
    if PROF_NPZ.exists():
        print("  build: cached"); return
    LF.mkdir(parents=True, exist_ok=True)
    if not (LF / 'user_artists.dat').exists():
        url = 'http://files.grouplens.org/datasets/hetrec2011/hetrec2011-lastfm-2k.zip'
        z = LF / 'lf.zip'; urllib.request.urlretrieve(url, z)
        zipfile.ZipFile(z).extractall(LF)
    ua = pd.read_csv(LF / 'user_artists.dat', sep='\t')          # userID artistID weight
    uta = pd.read_csv(LF / 'user_taggedartists.dat', sep='\t')   # userID artistID tagID ...

    artists = ua.groupby('artistID')['userID'].nunique().nlargest(N_ARTISTS).index.tolist()
    a_idx = {a: i for i, a in enumerate(artists)}
    # top tags by #artist-applications among slate artists
    uta_s = uta[uta['artistID'].isin(a_idx)]
    tags = uta_s.groupby('tagID')['userID'].nunique().nlargest(N_TAGS).index.tolist()
    t_idx = {t: N_ARTISTS + i for i, t in enumerate(tags)}
    # artist -> tag slate indices (a tag applies to an artist if any user tagged it)
    art_tags = {a: set() for a in a_idx}
    for r in uta_s[uta_s['tagID'].isin(t_idx)].itertuples():
        if r.artistID in a_idx and r.tagID in t_idx:
            art_tags[r.artistID].add(t_idx[r.tagID])

    n_items = N_ARTISTS + N_TAGS
    ua_s = ua[ua['artistID'].isin(a_idx)]
    profiles = {}
    for uid, grp in ua_s.groupby('userID'):
        if len(grp) < 5:
            continue
        med = grp['weight'].median()
        vec = np.full(n_items, np.nan, dtype=np.float32)
        tag_sum = np.zeros(n_items); tag_cnt = np.zeros(n_items)
        for r in grp.itertuples():
            i = a_idx[r.artistID]
            lk = 1.0 if r.weight >= med else 0.0
            vec[i] = lk
            for ti in art_tags[r.artistID]:
                tag_sum[ti] += lk; tag_cnt[ti] += 1
        has = tag_cnt >= 3
        vec[has] = (tag_sum[has] / tag_cnt[has] >= 0.5).astype(np.float32)
        profiles[uid] = vec

    uids = sorted(profiles)
    np.random.seed(SEED); np.random.shuffle(uids)
    split = int(0.8 * len(uids))
    train_u, test_u = uids[:split], uids[split:]
    items = ([('artist', int(a), f'artist_{a}') for a in artists] +
             [('tag', int(t), f'tag_{t}') for t in tags])
    np.savez(PROF_NPZ,
             train=np.stack([profiles[u] for u in train_u]),
             test=np.stack([profiles[u] for u in test_u]),
             items=np.array(items, dtype=object), n_targets=N_TARGETS)
    print(f"  build: {len(train_u)} train / {len(test_u)} test users, "
          f"{n_items} items ({N_ARTISTS} artists + {len(tags)} tags)")


# --------------------------- stage 2: train ---------------------------------
def train():
    if INST.exists():
        print("  train: cached"); return
    d = np.load(PROF_NPZ, allow_pickle=True)
    train = d['train']; val = d['test'][:300]
    n_items = train.shape[1]
    attr_idx = np.arange(N_ARTISTS, n_items)

    def batch(arr, bs, rng):
        idx = rng.permutation(len(arr))
        for s in range(0, len(idx), bs):
            chunk = arr[idx[s:s+bs]]
            L = MAX_REVEAL
            bi = np.zeros((len(chunk), L), np.int64)
            bp = np.zeros((len(chunk), L), np.int64)
            pad = np.ones((len(chunk), L), bool)
            for r, full in enumerate(chunk):
                rated = np.where(~np.isnan(full))[0]
                if len(rated) == 0:
                    continue
                cand = rated
                if rng.random() < 0.15:
                    a = np.intersect1d(rated, attr_idx)
                    if len(a): cand = a
                k = int(np.exp(rng.uniform(0, np.log(min(len(cand), L)+1))))
                k = max(1, min(k, len(cand)))
                sel = rng.choice(cand, k, replace=False)
                for j, e in enumerate(sel):
                    bi[r, j] = e; bp[r, j] = int(full[e] >= 0.5); pad[r, j] = False
            yield (torch.from_numpy(bi), torch.from_numpy(bp),
                   torch.from_numpy(pad), torch.from_numpy(chunk))

    def loss_fn(lk, rt, tgt):
        m = ~torch.isnan(tgt); t0 = torch.where(m, tgt, torch.zeros_like(tgt))
        per = nn.functional.binary_cross_entropy_with_logits(lk, t0, reduction='none')
        ll = (per * m.float()).sum() / m.float().sum().clamp(min=1)
        rl = nn.functional.binary_cross_entropy_with_logits(rt, m.float())
        return ll + rl

    model = DualHeadSetEncoder(n_items, d_model=128)
    opt = torch.optim.Adam(model.parameters(), lr=5e-4)
    lrng = np.random.default_rng(SEED)
    best, best_state, best_ep = 1e9, None, -1
    t0 = time.time()
    for ep in range(60):
        model.train()
        for bi, bp, pad, tgt in batch(train, 64, lrng):
            opt.zero_grad(); lk, rt = model(bi, bp, pad)
            loss_fn(lk, rt, tgt).backward(); opt.step()
        model.eval(); va = 0; nb = 0
        with torch.no_grad():
            for bi, bp, pad, tgt in batch(val, 128, lrng):
                lk, rt = model(bi, bp, pad); va += loss_fn(lk, rt, tgt).item(); nb += 1
        va /= max(nb, 1)
        if va < best:
            best, best_state, best_ep = va, {k: v.clone() for k, v in model.state_dict().items()}, ep
        if (ep+1) % 10 == 0:
            print(f"    ep {ep+1}/60 val {va:.4f} (best {best:.4f}@{best_ep+1}) "
                  f"{time.time()-t0:.0f}s", flush=True)
        if ep - best_ep >= 12:
            print("    early stop"); break
    model.load_state_dict(best_state)
    torch.save({'model_state_dict': model.state_dict(), 'arch': 'dual_set_encoder',
                'd_model': 128, 'n_heads': 4, 'n_layers': 2, 'n_items': n_items,
                'n_movies': N_TARGETS, 'items': d['items'].tolist(), 'val_loss': best,
                'config': {'dataset': 'lastfm', 'max_reveal': MAX_REVEAL}}, INST)
    print(f"  train: best val {best:.4f}, saved")


def load_instrument():
    ckpt = torch.load(INST, weights_only=False)
    m = DualHeadSetEncoder(ckpt['n_items'], ckpt['d_model'], ckpt['n_heads'], ckpt['n_layers'])
    m.load_state_dict(ckpt['model_state_dict']); m.eval()
    w = DualSetInstrumentWrapper(m, ckpt['n_items'], ckpt['n_movies'], MAX_REVEAL)
    return w, ckpt['items']


# --------------------------- stage 3: accept --------------------------------
def accept(wrapper, test):
    res = {}
    # monotonicity: accuracy vs #revealed (random reveals)
    rng = np.random.default_rng(SEED)
    ts = [1, 3, 5, 10, 20]
    accs = {t: [] for t in ts}
    for prof in test[:200]:
        rated = np.where(~np.isnan(prof))[0]
        if len(rated) < max(ts): continue
        order = rated.copy(); rng.shuffle(order)
        for t in ts:
            rev = [(int(e), float(prof[e])) for e in order[:t]]
            p = wrapper.predict(rev); gt = prof[:wrapper.n_movies]
            f = ~np.isnan(gt)
            accs[t].append(float(np.mean((p[f] > 0.5) == gt[f])))
    means = {t: float(np.mean(v)) for t, v in accs.items() if v}
    from scipy.stats import spearmanr
    rho = float(spearmanr(list(means), [means[t] for t in means])[0]) if len(means) > 2 else 0
    res['accuracy_by_reveals'] = means
    res['monotonic_rho'] = rho
    # liked/disliked overlap on artist targets
    rng2 = np.random.default_rng(1)
    ov = []
    attr = list(range(N_ARTISTS, wrapper.n_items))
    for _ in range(100):
        e = int(rng2.choice(attr)) if attr else int(rng2.integers(N_ARTISTS))
        tl = set(np.argsort(-wrapper.predict([(e, 1.0)]))[:10])
        td = set(np.argsort(-wrapper.predict([(e, 0.0)]))[:10])
        ov.append(len(tl & td) / 10)
    res['liked_disliked_overlap'] = float(np.mean(ov))
    res['accepted'] = bool(rho > 0.9 and np.mean(ov) < 0.5)
    print(f"  accept: monotonic_rho={rho:.3f} overlap={res['liked_disliked_overlap']:.2f} "
          f"acc@1={means.get(1,0):.3f} acc@20={means.get(20,0):.3f} "
          f"=> accepted={res['accepted']}")
    return res


# --------------------------- stage 4: benchmark -----------------------------
def benchmark(wrapper, items, train, test):
    p_rated = (~np.isnan(train)).mean(0)
    pols = {
        'random': lambda: P.RandomPolicy(items),
        'popularity': lambda: P.PopularityPolicy(items, p_rated),
        'greedy_infogain': lambda: P.GreedyInfoGainPolicy(items, p_rated),
        'scpr_entropy': lambda: P.SCPREntropyPolicy(items, p_rated),
        'thompson': lambda: P.ThompsonPolicy(items),
    }
    test = test[:300]
    rng = np.random.default_rng(SEED)
    results = {}
    for name, fac in pols.items():
        pol = fac(); logs = []
        for i, prof in enumerate(test):
            u = SimulatedUser(i, prof)
            logs.append(run_episode(pol, u, wrapper, N_TURNS,
                                    np.random.default_rng((SEED*7+i) % 2**31)))
        s = summarize(logs, N_TURNS)
        # adaptivity audit
        t1 = Counter(l.questions[0] for l in logs if l.questions)
        br = {}
        for l in logs:
            if len(l.questions) >= 2:
                br.setdefault(l.answers[0], Counter())[l.questions[1]] += 1
        t2 = {a: c.most_common(1)[0][0] for a, c in br.items()}
        s['branches'] = len(set(t2.values())) > 1
        s['t1_concentration'] = t1.most_common(1)[0][1] / len(logs)
        results[name] = s
        with open(EXP / f'episodes_lastfm_{name}.jsonl', 'w') as f:
            for l in logs: f.write(json.dumps(l.to_dict()) + '\n')
        print(f"  bench {name:<16} final={s['final_accuracy']:.4f} "
              f"AUAC={s['auac']:.4f} answer={s['hit_rate']:.0%} branches={s['branches']}")
    (EXP / 'benchmark_lastfm.json').write_text(json.dumps(results, indent=2))
    return results


def main():
    t0 = time.time()
    print("LastFM pipeline")
    try:
        print("[1/4] build"); build()
        print("[2/4] train"); train()
        d = np.load(PROF_NPZ, allow_pickle=True)
        wrapper, items = load_instrument()
        print("[3/4] accept"); acc = accept(wrapper, d['test'])
        print("[4/4] benchmark"); res = benchmark(wrapper, items, d['train'], d['test'])
        # append to status
        status = ROOT / 'OVERNIGHT_STATUS.md'
        prior = res['random']['turn0_accuracy']
        lines = ["", "## LastFM Tier-1 results (real-data adaptivity demonstrator)",
                 f"instrument accepted={acc['accepted']} (rho={acc['monotonic_rho']:.2f}, "
                 f"overlap={acc['liked_disliked_overlap']:.2f}); prior={prior:.3f}", "",
                 "| policy | final | AUAC | answer | adaptive |", "|---|---|---|---|---|"]
        for k in ['random', 'popularity', 'greedy_infogain', 'scpr_entropy', 'thompson']:
            s = res[k]
            lines.append(f"| {k} | {s['final_accuracy']:.4f} | {s['auac']:.4f} | "
                         f"{s['hit_rate']:.0%} | {s['branches']} |")
        adapt = max(res['greedy_infogain']['auac'], res['scpr_entropy']['auac'])
        stat = res['popularity']['auac']
        lines.append(f"\nAdaptive heuristic best AUAC={adapt:.4f} vs popularity "
                     f"{stat:.4f} (gap {adapt-stat:+.4f}). Tier-2 (PPO/bot-play) "
                     f"is the recommended next step.")
        with open(status, 'a', encoding='utf-8') as f:
            f.write("\n".join(lines))
        print(f"\nDONE {time.time()-t0:.0f}s; results in benchmark_lastfm.json + OVERNIGHT_STATUS.md")
    except Exception as e:
        import traceback; traceback.print_exc()
        print(f"PIPELINE ERROR: {e}")


if __name__ == '__main__':
    main()
