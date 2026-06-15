"""
Generic dataset pipeline: train dual-head instrument + accept (with oracle
ceiling, base-rate) + benchmark heuristics (incl. answerability-aware
greedy) from a prebuilt profiles npz. Reusable for any dataset.

Usage:
  DATASET_NPZ=<path> DATASET_NAME=<name> poetry run python \
      scripts/paper1/generic_pipeline.py
npz must contain: train, test, items(object), n_targets.
"""

import sys, os
sys.path.insert(0, '.')
sys.path.insert(0, 'scripts/paper1')

import json, time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from synthetic_sanity import DualHeadSetEncoder
from test_instrument_lib import DualSetInstrumentWrapper, CHECKPOINT_DIR
from testbed import SimulatedUser, run_episode, summarize
import policies as P

ROOT = Path('C:/dev/phd/casper'); EXP = ROOT / 'experiments/paper1'
NPZ = Path(os.environ['DATASET_NPZ']); NAME = os.environ['DATASET_NAME']
INST = CHECKPOINT_DIR / f"instrument_{NAME}{os.environ.get('INST_TAG','')}.pt"
N_TURNS = 15; MAX_REVEAL = 60; SEED = 42
D_MODEL = int(os.environ.get('D_MODEL', 128)); N_EPOCHS = int(os.environ.get('EPOCHS', 60))
np.random.seed(SEED); torch.manual_seed(SEED)


def train(train_arr, val_arr, n_items, n_targets):
    if INST.exists():
        print("  train cached"); return
    attr_idx = np.arange(n_targets, n_items)

    def batch(arr, bs, rng):
        for s in range(0, len(arr), bs):
            chunk = arr[rng.permutation(len(arr))[s:s+bs]] if s == 0 else arr[s:s+bs]
            bi = np.zeros((len(chunk), MAX_REVEAL), np.int64)
            bp = np.zeros((len(chunk), MAX_REVEAL), np.int64)
            pad = np.ones((len(chunk), MAX_REVEAL), bool)
            for r, full in enumerate(chunk):
                rated = np.where(~np.isnan(full))[0]
                if len(rated) == 0: continue
                cand = rated
                if rng.random() < 0.15 and len(np.intersect1d(rated, attr_idx)):
                    cand = np.intersect1d(rated, attr_idx)
                k = max(1, min(int(np.exp(rng.uniform(0, np.log(min(len(cand), MAX_REVEAL)+1)))), len(cand)))
                for j, e in enumerate(rng.choice(cand, k, replace=False)):
                    bi[r, j] = e; bp[r, j] = int(full[e] >= 0.5); pad[r, j] = False
            yield torch.from_numpy(bi), torch.from_numpy(bp), torch.from_numpy(pad), torch.from_numpy(chunk)

    def lf(lk, rt, t):
        m = ~torch.isnan(t); t0 = torch.where(m, t, torch.zeros_like(t))
        per = nn.functional.binary_cross_entropy_with_logits(lk, t0, reduction='none')
        return (per*m.float()).sum()/m.float().sum().clamp(min=1) + \
               nn.functional.binary_cross_entropy_with_logits(rt, m.float())

    model = DualHeadSetEncoder(n_items, d_model=D_MODEL)
    opt = torch.optim.Adam(model.parameters(), lr=5e-4)
    rng = np.random.default_rng(SEED); best, bs_, be = 1e9, None, -1; t0 = time.time()
    # checkpoint-resume: survive background-job kills (save every 3 epochs)
    RESUME = INST.with_suffix('.resume.pt')
    start_ep = 0
    if RESUME.exists():
        rc = torch.load(RESUME, weights_only=False)
        model.load_state_dict(rc['model']); opt.load_state_dict(rc['opt'])
        start_ep = rc['ep'] + 1; best = rc['best']; bs_ = rc['best_state']; be = rc['be']
        print(f"  RESUME from ep{start_ep} (best {best:.4f}@{be+1})", flush=True)
    for ep in range(start_ep, N_EPOCHS):
        model.train()
        for bi, bp, pad, t in batch(train_arr, 64, rng):
            opt.zero_grad(); lk, rt = model(bi, bp, pad); lf(lk, rt, t).backward(); opt.step()
        model.eval(); va = nb = 0
        with torch.no_grad():
            for bi, bp, pad, t in batch(val_arr, 128, rng):
                lk, rt = model(bi, bp, pad); va += lf(lk, rt, t).item(); nb += 1
        va /= max(nb, 1)
        if va < best: best, bs_, be = va, {k: v.clone() for k, v in model.state_dict().items()}, ep
        if (ep+1) % 5 == 0: print(f"    ep{ep+1} val{va:.4f} best{best:.4f}@{be+1} {time.time()-t0:.0f}s", flush=True)
        if (ep + 1) % 3 == 0:
            torch.save({'model': model.state_dict(), 'opt': opt.state_dict(), 'ep': ep,
                        'best': best, 'best_state': bs_, 'be': be}, RESUME)
        if ep - be >= 12: break
    model.load_state_dict(bs_)
    torch.save({'model_state_dict': model.state_dict(), 'arch': 'dual_set_encoder',
                'd_model': D_MODEL, 'n_heads': 4, 'n_layers': 2, 'n_items': n_items,
                'n_movies': n_targets, 'val_loss': best, 'config': {'dataset': NAME, 'max_reveal': MAX_REVEAL}}, INST)
    if RESUME.exists(): RESUME.unlink()
    print(f"  train best {best:.4f}")


def load_instrument():
    c = torch.load(INST, weights_only=False)
    m = DualHeadSetEncoder(c['n_items'], c['d_model'], c['n_heads'], c['n_layers'])
    m.load_state_dict(c['model_state_dict']); m.eval()
    return DualSetInstrumentWrapper(m, c['n_items'], c['n_movies'], MAX_REVEAL)


def accept(w, test, n_targets):
    rng = np.random.default_rng(SEED); ts = [1, 3, 5, 10, 20]; accs = {t: [] for t in ts}
    base = []
    for prof in test[:300]:
        gt = prof[:w.n_movies]; f = ~np.isnan(gt)
        if f.sum() == 0: continue
        base.append(float((gt[f] == 1).mean()))
        rated = np.where(~np.isnan(prof))[0]
        if len(rated) < 3: continue
        order = rated.copy(); rng.shuffle(order)
        for t in ts:
            rev = [(int(e), float(prof[e])) for e in order[:t]]
            p = w.predict(rev); accs[t].append(float(np.mean((p[f] > 0.5) == gt[f])))
    means = {t: float(np.mean(v)) for t, v in accs.items() if v}
    cps = []
    for prof in test[:300]:
        gt = prof[:w.n_movies]; f = ~np.isnan(gt)
        if f.sum() == 0: continue
        rev = [(int(e), float(prof[e])) for e in np.where(~np.isnan(prof))[0]]
        cps.append(float(np.mean((w.predict(rev)[f] > 0.5) == gt[f])))
    base_rate = float(np.mean([max(b, 1-b) for b in base]))   # majority-class accuracy
    ceil = float(np.mean(cps))
    res = {'accuracy_by_reveals': means, 'oracle_ceiling': ceil,
           'majority_base_rate': base_rate, 'lift_over_base': ceil - base_rate}
    print(f"  accept: base(majority)={base_rate:.3f} ceiling={ceil:.3f} "
          f"lift={ceil-base_rate:+.3f} acc@1={means.get(1,0):.3f} acc@20={means.get(20,0):.3f}")
    return res


def benchmark(w, items, train, test, n_targets):
    p_rated = (~np.isnan(train)).mean(0)
    pols = {'random': lambda: P.RandomPolicy(items),
            'popularity': lambda: P.PopularityPolicy(items, p_rated),
            'greedy_infogain': lambda: P.GreedyInfoGainPolicy(items, p_rated),
            'greedy_answerability': lambda: P.GreedyAnswerabilityPolicy(items),
            'scpr_entropy': lambda: P.SCPREntropyPolicy(items, p_rated),
            'thompson': lambda: P.ThompsonPolicy(items)}
    # candidate pruning for large catalogs (standard CRS): the per-candidate
    # heuristics score only attributes + top-CAND_CAP popular movies, else they
    # take hours at thousands of items. Applied to greedy/scpr only.
    CAND_CAP = int(os.environ.get('CAND_CAP', 0))
    cand_pool = None
    if CAND_CAP and w.n_items > CAND_CAP:
        attrs = list(range(n_targets, w.n_items))
        top_movies = [int(m) for m in np.argsort(-p_rated[:n_targets])[:CAND_CAP]]
        cand_pool = sorted(set(attrs) | set(top_movies))
        print(f"  candidate pruning: {len(cand_pool)} candidates "
              f"({len(attrs)} attrs + top-{CAND_CAP} movies)")
    pruned = {'greedy_infogain', 'greedy_answerability', 'scpr_entropy'}
    test = test[:300]; results = {}
    for name, fac in pols.items():
        pol = fac(); logs = []
        if cand_pool is not None and name in pruned:
            pol.cand_pool = cand_pool
        for i, prof in enumerate(test):
            logs.append(run_episode(pol, SimulatedUser(i, prof), w, N_TURNS,
                                    np.random.default_rng((SEED*7+i) % 2**31)))
        s = summarize(logs, N_TURNS)
        t1 = Counter(l.questions[0] for l in logs if l.questions)
        br = {}
        for l in logs:
            if len(l.questions) >= 2: br.setdefault(l.answers[0], Counter())[l.questions[1]] += 1
        s['branches'] = len({c.most_common(1)[0][0] for c in br.values()}) > 1
        results[name] = s
        with open(EXP / f'episodes_{NAME}_{name}.jsonl', 'w') as fo:
            for l in logs: fo.write(json.dumps(l.to_dict()) + '\n')
        print(f"  {name:<22} final={s['final_accuracy']:.4f} AUAC={s['auac']:.4f} "
              f"answer={s['hit_rate']:.0%} branches={s['branches']}")
    (EXP / f'benchmark_{NAME}.json').write_text(json.dumps(results, indent=2))
    return results


def main():
    d = np.load(NPZ, allow_pickle=True)
    train_arr, test_arr = d['train'], d['test']
    n_items = train_arr.shape[1]; n_targets = int(d['n_targets'])
    items = [tuple(x) for x in d['items'].tolist()]
    print(f"{NAME}: {len(train_arr)} train / {len(test_arr)} test, {n_items} items, {n_targets} targets")
    try:
        print("[train]"); train(train_arr, test_arr, n_items, n_targets)
        w = load_instrument()
        print("[accept]"); acc = accept(w, test_arr, n_targets)
        if os.environ.get('SKIP_BENCH'):
            print(f"[bench] skipped (SKIP_BENCH); instrument ready at {INST}"); return
        print("[bench]"); res = benchmark(w, items, train_arr, test_arr, n_targets)
        # append to status
        ga = res['greedy_answerability']['auac']; pop = res['popularity']['auac']
        rnd = res['random']['auac']
        with open(ROOT / 'OVERNIGHT_STATUS.md', 'a', encoding='utf-8') as f:
            f.write(f"\n## {NAME} results\n")
            f.write(f"ceiling={acc['oracle_ceiling']:.3f} base={acc['majority_base_rate']:.3f} "
                    f"lift={acc['lift_over_base']:+.3f}\n\n")
            f.write("| policy | final | AUAC | answer | branches |\n|---|---|---|---|---|\n")
            for k in ['random', 'popularity', 'greedy_infogain', 'greedy_answerability', 'scpr_entropy', 'thompson']:
                s = res[k]; f.write(f"| {k} | {s['final_accuracy']:.4f} | {s['auac']:.4f} | "
                                    f"{s['hit_rate']:.0%} | {s['branches']} |\n")
            f.write(f"\ngreedy_answerability AUAC {ga:.4f} vs popularity {pop:.4f} vs random {rnd:.4f} "
                    f"(answerability-routing gap {ga-pop:+.4f} vs static)\n")
        print(f"\nDONE: greedy_answerability {ga:.4f} vs popularity {pop:.4f} vs random {rnd:.4f}")
    except Exception as e:
        import traceback; traceback.print_exc(); print(f"ERROR {e}")


if __name__ == '__main__':
    main()
