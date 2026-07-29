"""run_queue2.py -- QUEUE-2 runner for the four master-table baselines added 2026-07-22:
  turbocf     Turbo-CF (Park et al., SIGIR 2024)             -- closed form, minutes
  rbmf_seed   RBMF / functional-MF point-LS seed             -- closed form (frozen iALS), minutes
  sasrec      SASRec (Kang & McAuley 2018) fold-in adaptation -- training job (later night)
  tanp        TaNP best-effort NP core (Lin et al. WWW 2021)  -- training job, modest

Same ruler, split, metrics and logging style as run_ml25m_liang.py -- it is the sibling runner for the
newly-ported rows. Evaluates on the CANONICAL Liang-recipe ML-25M strong-generalization split
(data/ml-25m/proc/), reusing run_ml25m_liang.compute_head_mask for the tail definition and
metrics.evaluate for scoring. Resumable: skips any baseline whose JSON already exists.

  PRIMARY:   full NDCG@10  and  tail NDCG@10   (tail = top-33%-train-mass head mask)
  SECONDARY: NDCG@100, Recall@20, Recall@50
  Neural rows (sasrec, tanp): early-stop on val FULL NDCG@10; checkpoints to
    .cache/baselines/<name>_ml25m_liang.pt (NEVER overwrites arena checkpoints).

Outputs: experiments/baselines/ml25m_liang/<name>.json + a rolling run_queue2.log (separate from the
run_ml25m_liang run.log so the two runners never interleave a single file).

Usage:  python src/baselines/run_queue2.py [--only turbocf,rbmf_seed] [--max_minutes 600]
NOTE: FULL run, no data caps. Do NOT launch the training rows (sasrec/tanp) until the review gate
clears (HARD RULE 10: training code committed first). turbocf/rbmf_seed are closed-form (minutes).
"""
import os
import sys
import json
import time
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import metrics as M
import turbocf
import rbmf_seed
import sasrec
import tanp_bestefffort as tanp
from run_ml25m_liang import compute_head_mask   # reuse the canonical tail definition (no duplication)

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
PROC = os.path.join(_ROOT, "data", "ml-25m", "proc")
OUTDIR = os.path.join(_ROOT, "experiments", "baselines", "ml25m_liang")
LOGP = os.path.join(OUTDIR, "run_queue2.log")
CKPT = os.path.join(_ROOT, ".cache", "baselines")

ORDER = ["turbocf", "rbmf_seed", "sasrec", "tanp"]


def logln(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOGP, "a") as f:
        f.write(line + "\n")


def build(name, train, n_items, D, head_mask, max_minutes, log):
    """Return (predict_fn, hp_dict). Closed-form rows call fit directly; neural rows get an evaluator
    that early-stops on val FULL NDCG@10 (the primary metric), exactly as run_ml25m_liang does."""
    va_tr, va_te = D["val"]

    def neural_eval(predict):
        m = M.evaluate(predict, va_tr, va_te, batch_size=500, head_mask=head_mask)
        return m["ndcg@10"], m

    os.makedirs(CKPT, exist_ok=True)
    ck = os.path.join(CKPT, f"{name}_ml25m_liang.pt")

    if name == "turbocf":
        # No published ML-20M/25M Turbo-CF number exists, so (alpha, s, filter) is selected on VAL.
        # A fixed guess is unsafe: outside the stable region the polynomial filter inverts the ranking
        # at this catalogue size (see the STABILITY note in turbocf.py).
        pr = turbocf.fit_sweep(train, n_items, va_tr, va_te, log=log)
        return pr, pr.hp
    if name == "rbmf_seed":
        pr = rbmf_seed.fit(train, n_items, log=log)
        return pr, pr.hp
    if name == "sasrec":
        a = sasrec._defaults(); a.max_minutes = max_minutes

        def sasrec_eval(predict):
            predict.set_split("val")      # split-aware predict (per-user timestamp order): reset cursor
            return neural_eval(predict)
        pr = sasrec.fit(train, n_items, evaluator=sasrec_eval, args=a, ckpt=ck, log=log)
        pr.set_split("test")              # arm for the final test pass done by the main loop
        return pr, pr.hp
    if name == "tanp":
        a = tanp._defaults(); a.max_minutes = max_minutes
        pr = tanp.fit(train, n_items, evaluator=neural_eval, args=a, ckpt=ck, log=log)
        return pr, pr.hp
    raise ValueError(name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None, help="comma list of baseline names")
    ap.add_argument("--max_minutes", type=float, default=1e9, help="per-neural-baseline wall budget")
    args = ap.parse_args()
    os.makedirs(OUTDIR, exist_ok=True)
    order = ORDER
    if args.only:
        want = set(args.only.split(","))
        order = [n for n in order if n in want]

    logln(f"=== ml25m_liang queue-2 run start; order={order} ===")
    if not os.path.exists(os.path.join(PROC, "meta.json")):
        raise SystemExit(f"[run] {PROC} not built. Run: python scripts/baselines/liang_split.py --data ml-25m")
    meta = M.load_meta(PROC); n_items = meta["n_items"]
    logln(f"Liang ML-25M split: n_items={n_items} meta={meta}")
    train = M.load_train(n_items, PROC)
    D = {"val": M.load_val(n_items, PROC)}
    te_tr, te_te = M.load_test(n_items, PROC)
    head_mask, cnt = compute_head_mask(train, n_items)
    logln(f"head/tail: {int(head_mask.sum())} head items (top-33% train mass) / "
          f"{int((~head_mask).sum())} tail items; train_users={train.shape[0]} nnz={train.nnz}")

    for name in order:
        outp = os.path.join(OUTDIR, f"{name}.json")
        if os.path.exists(outp):
            logln(f"[skip] {name}: {outp} exists")
            continue
        logln(f"--- {name} ---")
        t0 = time.time()
        predict, hp = build(name, train, n_items, D, head_mask, args.max_minutes, logln)
        res = M.evaluate(predict, te_tr, te_te, batch_size=500, head_mask=head_mask)
        res["seconds"] = time.time() - t0
        res["hp"] = hp
        res["n_items"] = n_items
        json.dump(res, open(outp, "w"), indent=2)
        logln(f"[done] {name}: full@10={res['ndcg@10']:.4f} tail@10={res['tail_ndcg@10']:.4f} "
              f"ndcg@100={res['ndcg@100']:.4f} r@20={res['recall@20']:.4f} r@50={res['recall@50']:.4f} "
              f"{res['seconds']/60:.1f}m")
    logln("=== ml25m_liang queue-2 run complete ===")


if __name__ == "__main__":
    main()
