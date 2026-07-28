r"""greedy_polish.py -- endpoint-targeted local search over the best-static sequences.

WHY. The lazy-greedy sequences in greedy_static.json maximise the metric at EVERY PREFIX. That is
myopic: having correctly found that concepts are the best opening, the combined arm commits its first
four slots to concepts and cannot undo them by q=16, which is why combined (0.2049) trails items-only
(0.2098) at the longest budget despite the combined bank CONTAINING the item bank. NDCG over a sequence
is not guaranteed submodular, so the greedy path is locally optimal and globally behind.

WHAT. Seeded from each existing sequence, try replacing each of the L positions with each of the top-M
candidates (ranked by their original greedy gain), scoring the FULL length-L sequence each time, and keep
any improvement. This optimises the ENDPOINT rather than every prefix. Bounded at L x M evaluations per
pass, which is the reason for M: an unbounded swap over the full 1000-candidate pool is ~53 hours.

Either the deficit closes -- and combined >= items as containment implies -- or it does not, and the
myopia explanation is wrong and something more interesting is going on.

  python src/instrument/greedy_polish.py [--top_m 50] [--passes 1]
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "6")
import sys
import json
import time
import argparse
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
sys.path.insert(0, os.path.join(_ROOT, "src", "baselines"))
sys.path.insert(0, _HERE)

from train_tower_t2 import log
import run_battery_phaseA as PA
from run_battery_phaseA import build_real_ctx
from tradeoff_ledger import build_shared, Rung
from strategy_channel_suite import _greedy_avg, _seq_answered
from concept_fold import ConceptFoldNet
import torch

IN = os.path.join(_ROOT, "experiments", "battery", "greedy_static.json")
OUT = os.path.join(_ROOT, "experiments", "battery", "greedy_polish.json")
BUDGETS = (1, 2, 4, 8, 16)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot", default=PA.SNAP_DEFAULT)
    ap.add_argument("--sclite_ckpt", default=os.path.join(_ROOT, ".cache", "instrument",
                                                          "cd_s1_l10_best.pt"))
    ap.add_argument("--top_m", type=int, default=50, help="candidates tried per position")
    ap.add_argument("--passes", type=int, default=2)
    ap.add_argument("--n_items", type=int, default=500)
    ap.add_argument("--n_concepts", type=int, default=500)
    args = ap.parse_args()
    t0 = time.time()

    prev = json.load(open(IN))
    log(f"[polish] seeding from {IN} (objective={prev.get('objective', 'full')})")

    ctx = build_real_ctx(args.snapshot)
    sh = build_shared(ctx)
    blob = torch.load(args.sclite_ckpt, map_location="cpu")
    net = ConceptFoldNet(len(blob["tags"]), d=ctx.d, h=blob["hidden"])
    net.load_state_dict(blob["net"]); net.eval()
    rung = Rung("sclite", ctx, sh, clite_net=net, val_source="signed")
    build_rows = [r for r in range(ctx.n) if ctx.va_te[r].nnz > 0]

    ctx_te = build_real_ctx(args.snapshot, split="test")
    sh_te = build_shared(ctx_te)
    net_te = ConceptFoldNet(len(blob["tags"]), d=ctx_te.d, h=blob["hidden"])
    net_te.load_state_dict(blob["net"]); net_te.eval()
    eval_rung = Rung("sclite", ctx_te, sh_te, clite_net=net_te, val_source="signed")
    eval_rows = [r for r in range(ctx_te.n) if ctx_te.va_te[r].nnz > 0]
    log(f"[polish] build={len(build_rows)} val / eval={len(eval_rows)} TEST; top_m={args.top_m}")

    # answerability ranking for the concept pool (same rule the greedy used)
    answ = sh["conc_answerable"]
    ans_rate = np.asarray(answ[build_rows].mean(0)).ravel()

    out = {"seeded_from": os.path.basename(IN), "top_m": args.top_m, "arms": {}}
    for name, arm in prev["arms"].items():
        seq = [(int(c), int(i)) for c, i in arm["sequence"]]
        # Candidate pool must be questions NOT already in the sequence -- an earlier version built it
        # from seq itself, so every candidate was skipped by the `q in cur` guard and the search did
        # zero evaluations. Rebuild from the same pools the greedy drew from.
        if name == "items-only":
            cand = [(0, int(i)) for i in sh["order_pop"][:args.n_items]]
        elif name == "concepts-only":
            cand = [(1, int(c)) for c in np.argsort(-ans_rate)[:args.n_concepts]]
        else:
            cand = ([(0, int(i)) for i in sh["order_pop"][:args.n_items]]
                    + [(1, int(c)) for c in np.argsort(-ans_rate)[:args.n_concepts]])
        inseq = set(seq)
        pool = [q for q in cand if q not in inseq][:args.top_m]
        cur = list(seq)
        log(f"[polish {name}] {len(pool)} swap candidates outside the seed sequence")
        base, _ = _greedy_avg(rung, build_rows, cur)
        log(f"[polish {name}] seed endpoint (build) = {base:.4f}")
        nev = 0
        for _ in range(args.passes):
            improved = False
            for pos in range(len(cur)):
                best_v, best_q = base, cur[pos]
                for q in pool:
                    if q == cur[pos] or q in cur:
                        continue
                    trial = list(cur); trial[pos] = q
                    v, _ = _greedy_avg(rung, build_rows, trial); nev += 1
                    if v > best_v + 1e-6:
                        best_v, best_q = v, q
                if best_q != cur[pos]:
                    cur[pos] = best_q; base = best_v; improved = True
                    log(f"[polish {name}] pos {pos} -> {best_q} | endpoint {base:.4f} ({nev} evals)")
            if not improved:
                log(f"[polish {name}] no improvement this pass; stopping")
                break
        curve = {}
        for q in BUDGETS:
            f, t = _greedy_avg(eval_rung, eval_rows, cur[:q])
            ia, ca = _seq_answered(eval_rung.sh, eval_rows, cur[:q])
            curve[str(q)] = {"full@10": f, "tail@10": t,
                             "item_answered": round(ia, 3), "conc_answered": round(ca, 3)}
        comp = "".join("c" if c == 1 else "i" for c, _ in cur)
        out["arms"][name] = {"order": comp, "curve": curve, "n_evals": nev,
                             "sequence": [[int(c), int(i)] for c, i in cur],
                             "seed_order": arm["order"],
                             "seed_q16_full": arm["curve"]["16"]["full@10"]}
        log(f"[polish {name}] order={comp} | q16 {curve['16']['full@10']:.4f}/{curve['16']['tail@10']:.4f} "
            f"(seed was {arm['curve']['16']['full@10']:.4f}) [{nev} evals]")
        json.dump(out, open(OUT, "w"), indent=1)

    out["seconds"] = round(time.time() - t0, 1)
    json.dump(out, open(OUT, "w"), indent=1)
    log(f"[polish] -> {OUT} ({out['seconds']/60:.1f}m)")


if __name__ == "__main__":
    main()
