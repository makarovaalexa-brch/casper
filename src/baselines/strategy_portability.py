r"""strategy_portability.py -- does the RANKING of interview strategies depend on the recommender?

THE QUESTION. The instrument chapter argues that elicitation-policy results are confounded by the
recommender underneath them. That argument is currently made from within one instrument (small policy
deltas on a strong tower) and never tested across recommenders. This runs the SAME fixed question
sequences through recommenders spanning a wide accuracy range and asks two things:

  COMPRESSION  does the spread between strategies shrink as the recommender gets stronger?
  ORDERING     does the best strategy at a given budget change with the recommender?

An ordering flip is the strong result: it would mean a policy comparison published on a weak recommender
does not transfer, which is precisely the confound an instrument exists to remove. Compression alone is
the weaker result and is still worth reporting. A null -- same order, same spread -- is also a real
finding and is reported as such.

PRE-REGISTERED DESIGN (2026-07-29).
  Ruler       : the canonical Liang ML-25M split, test cohort, full AND tail NDCG@10, scored by
                metrics.evaluate verbatim so no metric is reimplemented here.
  Strategies  : four fixed sequences, each defined WITHOUT reference to any recommender, so none is
                advantaged by the model it is scored on --
                  popularity   items by train interaction count, descending
                  entropy      items by binary entropy of their train frequency, descending
                  random       uniform sample of items, fixed seed
                  greedy       the best-static item sequence from greedy_static.json (built on the
                               tower's validation cohort; the ONLY sequence with a home-field
                               advantage, and it is reported with that caveat)
  Recommenders: most-popular, item-kNN, iALS, EASE, RecVAE -- spanning roughly 0.13 to 0.35 full
                NDCG@10 on this ruler.
  Interview   : question t is "have you seen and liked item i_t". A user answers from their FOLD-IN
                half only (targets excluded by construction). The model input at budget q is the binary
                set of items among the first q asked that the user liked. Every recommender sees the
                IDENTICAL input, so what varies is the recommender and nothing else.
  Budgets     : 1, 2, 4, 8, 16.
  Read-out    : per recommender, the strategy ordering at each budget, and the spread
                (best minus worst). Compression = spread falls as recommender accuracy rises.

  python src/baselines/strategy_portability.py [--budgets 1,2,4,8,16] [--seed 0]
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "6")
import sys
import json
import time
import argparse
import numpy as np
import torch
from scipy import sparse

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
sys.path.insert(0, _HERE)

import metrics as M
import pop, itemknn, ials, ease, recvae
from run_ml25m_liang import compute_head_mask

PROC = os.path.join(_ROOT, "data", "ml-25m", "proc")
CKPT = os.path.join(_ROOT, ".cache", "baselines")
OUT = os.path.join(_ROOT, "experiments", "baselines", "ml25m_liang", "strategy_portability.json")
GREEDY = os.path.join(_ROOT, "experiments", "battery", "greedy_static.json")


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def build_strategies(train, n_items, n_q, seed):
    """Four fixed question sequences, none of them a function of the recommender being scored."""
    cnt = np.asarray(train.sum(axis=0)).ravel().astype(np.float64)
    order_pop = np.argsort(-cnt)
    p = np.clip(cnt / max(train.shape[0], 1), 1e-9, 1 - 1e-9)
    ent = -(p * np.log2(p) + (1 - p) * np.log2(1 - p))
    order_ent = np.argsort(-ent)
    rng = np.random.RandomState(seed)
    order_rand = rng.permutation(n_items)
    S = {
        "popularity": order_pop[:n_q].tolist(),
        "entropy": order_ent[:n_q].tolist(),
        "random": order_rand[:n_q].tolist(),
    }
    if os.path.exists(GREEDY):
        g = json.load(open(GREEDY))
        arm = g["arms"].get("items-only")
        if arm:
            S["greedy(tower-built)"] = [int(i) for c, i in arm["sequence"] if c == 0][:n_q]
    return S


def interview_input(te_tr, asked):
    """Binary matrix of the items among `asked` that each user liked in their fold-in half.
    Zeroing the unasked columns of the fold-in profile is exactly 'the user answered only these
    questions'; every recommender receives this identical matrix."""
    mask = np.zeros(te_tr.shape[1], dtype=np.float32)
    mask[np.asarray(asked, dtype=np.int64)] = 1.0
    X = te_tr.multiply(mask[np.newaxis, :]).tocsr().astype(np.float32)
    X.eliminate_zeros()
    X.data[:] = 1.0
    return X


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budgets", default="1,2,4,8,16")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    budgets = [int(b) for b in args.budgets.split(",")]
    n_q = max(budgets)
    t0 = time.time()

    meta = M.load_meta(PROC); n_items = meta["n_items"]
    train = M.load_train(n_items, PROC)
    te_tr, te_te = M.load_test(n_items, PROC)
    hm, _ = compute_head_mask(train, n_items)
    log(f"ruler: {n_items} items, {te_tr.shape[0]} test users, head={int(hm.sum())}")

    strategies = build_strategies(train, n_items, n_q, args.seed)
    log(f"strategies: {list(strategies)}")

    # every recommender is fitted once and reused across all strategies and budgets
    recs = {}
    log("fitting most-popular"); recs["most-popular"] = pop.fit(train, n_items, log=log)
    log("fitting item-kNN");     recs["item-kNN"] = itemknn.fit(train, n_items, log=log)
    log("fitting iALS");         recs["iALS"] = ials.fit(train, n_items, log=log)
    log("fitting EASE");         recs["EASE"] = ease.fit(train, n_items, log=log)
    rv = os.path.join(CKPT, "recvae_ml25m_liang.pt")
    if os.path.exists(rv):
        blob = torch.load(rv, map_location="cpu")
        a = recvae._defaults()
        model = recvae.RecVAE(a.hidden, a.latent, n_items)
        st = blob["state"].get("best_state") or blob["model"]
        model.load_state_dict(st)
        recs["RecVAE"] = recvae.make_predict_fn(model)
        log(f"loaded RecVAE from checkpoint (ep{blob['state'].get('best_epoch')})")

    out = {"budgets": budgets, "seed": args.seed,
           "strategies": {k: v for k, v in strategies.items()}, "results": {}}

    for rname, predict in recs.items():
        out["results"][rname] = {}
        for sname, seq in strategies.items():
            curve = {}
            for q in budgets:
                X = interview_input(te_tr, seq[:q])
                r = M.evaluate(predict, X, te_te, batch_size=500, head_mask=hm)
                curve[str(q)] = {"full@10": float(r["ndcg@10"]), "tail@10": float(r["tail_ndcg@10"]),
                                 "mean_answered": float(X.getnnz(axis=1).mean())}
            out["results"][rname][sname] = curve
            log(f"  {rname:<12} {sname:<20} " +
                " ".join(f"q{q}={curve[str(q)]['full@10']:.4f}" for q in budgets))
        json.dump(out, open(OUT, "w"), indent=1)

    # read-out: ordering and spread per recommender per budget
    summary = {}
    for rname, per_s in out["results"].items():
        summary[rname] = {}
        for q in budgets:
            vals = {s: per_s[s][str(q)]["full@10"] for s in per_s}
            order = sorted(vals, key=lambda s: -vals[s])
            summary[rname][str(q)] = {"order": order,
                                      "spread": round(max(vals.values()) - min(vals.values()), 5)}
    out["summary"] = summary
    out["seconds"] = round(time.time() - t0, 1)
    json.dump(out, open(OUT, "w"), indent=1)

    log("=== ordering by recommender (best first) and spread ===")
    for rname in summary:
        for q in budgets:
            e = summary[rname][str(q)]
            log(f"  {rname:<12} q={q:<3} spread={e['spread']:.4f}  {' > '.join(e['order'])}")
    log(f"-> {OUT} ({out['seconds']/60:.1f}m)")


if __name__ == "__main__":
    main()
